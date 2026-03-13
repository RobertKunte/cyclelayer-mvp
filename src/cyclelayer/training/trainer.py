"""Training loop supporting single-task and multi-task (Phase 2) models.

Batch format
------------
DataLoader may yield either:
    (x, rul)                  -- standard 2-tuple
    (x, rul, theta_true)      -- 3-tuple when return_theta_true=True

Model forward may return either:
    rul                       -- scalar RUL (baselines, CycleLayerNet v0)
    (rul, theta_hat)          -- 2-tuple (CycleLayerNetV1)

TensorBoard tag layout
-----------------------
    Loss/                     -- all loss values
        train_epoch           -- total train loss per epoch
        val_epoch             -- total val loss per epoch
        train_{k}_epoch       -- component k (rul, theta, …) train per epoch
        val_{k}_epoch         -- component k val per epoch
        train_step            -- total train loss per step  [opt-in: tb_log_every_n_steps]
        {k}_step              -- component k per step       [opt-in: tb_log_every_n_steps]

    Val/                      -- per-epoch prediction plausibility (val set)
        mae                   -- mean absolute error
        rmse                  -- root mean squared error
        bias                  -- mean(pred - target); + = over-estimation
        pearson_r             -- Pearson correlation between pred and target
        pred_mean / pred_std  -- output distribution statistics
        pred_min  / pred_max  -- output range (clamp check)
        target_mean           -- constant reference (sanity check)

    Train/                    -- per-epoch prediction stats (train set, cheap)
        mae, bias             -- same as Val/ but for train batches

    Theta/                    -- health-parameter prediction (CycleLayerNetV1)
        mae_mean              -- mean MAE across all 10 channels
        mae_{name}            -- per-channel MAE  (fan_eff, HPC_flow, …)
        pred_mean_{name}      -- per-channel mean predicted value
        true_mean_{name}      -- per-channel mean true value (constant ref)

    Weights/
        param_norm            -- L2 norm of all model parameters
        grad_norm_epoch       -- mean pre-clip gradient norm (same as GradNorm/)

    GradNorm/
        train_epoch           -- mean gradient norm per epoch
        train_step            -- per-step gradient norm

    LR                        -- current learning rate
    Lambda/theta              -- scheduled lambda_theta value

Lambda-theta schedule
---------------------
    lambda_theta_start (float)       -- initial lambda value
    lambda_theta_end   (float)       -- final lambda value
    lambda_theta_warmup_epochs (int) -- epochs to apply the "warmup" phase
    lambda_theta_schedule ("step"|"linear"|"delayed")  default: "step"

    Step    : lambda = start for epoch <= warmup, else end
    Linear  : lambda linearly interpolates start->end over warmup epochs
    Delayed : lambda = 0 for warmup epochs (let RUL loss converge first),
              then linearly ramps 0->end over the next warmup epochs.
              Best for dense training (stride_train=5).

Gradient clipping
-----------------
    grad_clip_norm (float | 0 | None)
        Max L2 norm; set 0 or null to disable.  Default: 1.0.
        The un-clipped norm is always measured and logged.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cyclelayer.training.losses import CompositeLoss, PhysicsInformedLoss, RULLoss

logger = logging.getLogger(__name__)

# N-CMAPSS health parameter names (T_{split} column order).
# Used to label per-channel Theta/ scalars in TensorBoard.
_THETA_NAMES = [
    "fan_eff",  "fan_flow",
    "LPC_eff",  "LPC_flow",
    "HPC_eff",  "HPC_flow",
    "HPT_eff",  "HPT_flow",
    "LPT_eff",  "LPT_flow",
]


class Trainer:
    """Training loop with extended TensorBoard logging and checkpointing.

    Args:
        model: Model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Training configuration dict (from YAML).
        output_dir: Directory for checkpoints and TensorBoard logs.
        device: Torch device string ("cuda", "cpu", "mps"). Auto-detected if None.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        output_dir: str | Path = "runs/experiment",
        device: str | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.criterion = self._build_loss()

        lr = config.get("lr", 1e-3)
        weight_decay = config.get("weight_decay", 1e-4)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        n_epochs = config.get("epochs", 100)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        self.use_amp = config.get("use_amp", True) and self.device.type == "cuda"
        # init_scale=1024 (vs default 65536): 64× lower initial loss scaling →
        # much less float16 overflow → fewer NaN/inf gradient steps in early epochs.
        # growth_interval=100: rescale every 100 clean steps (default 2000 is too slow
        # for 1338 batches/epoch; this lets the scale recover quickly after overflow).
        self.scaler = torch.amp.GradScaler(
            init_scale=1024, growth_interval=100, enabled=self.use_amp
        )

        # Gradient clipping: configurable max L2 norm.
        # Default 1.0 — important for stability with dense training (stride=5).
        # Set to 0 or null in config to disable (norm is still measured).
        raw_clip = config.get("grad_clip_norm", 1.0)
        self.grad_clip_norm: float | None = float(raw_clip) if raw_clip else None

        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.global_step = 0

        # Per-step TensorBoard logging: 0 = disabled (default, fast).
        # Set tb_log_every_n_steps: 50 in the YAML to log step-level curves.
        # WARNING: small values (e.g. 1) create huge event files and add
        # significant overhead on network-mounted storage (Google Drive, NFS).
        self._tb_log_step: int = int(config.get("tb_log_every_n_steps", 0))

    # ------------------------------------------------------------------
    # Loss construction
    # ------------------------------------------------------------------

    def _build_loss(self) -> nn.Module:
        c = self.config
        rul_loss = RULLoss(
            mse_weight=c.get("mse_weight", 1.0),
            asymmetry=c.get("asymmetry", 0.5),
        )
        # lambda_theta_start takes precedence (schedule-aware); fall back to lambda_theta
        lambda_theta = c.get("lambda_theta_start", c.get("lambda_theta", 0.0))
        if lambda_theta > 0.0:
            return CompositeLoss(
                rul_loss=rul_loss,
                lambda_theta=lambda_theta,
                huber_delta=c.get("huber_delta", 0.1),
            )
        use_physics = c.get("use_physics_loss", False)
        if use_physics:
            return PhysicsInformedLoss(
                rul_loss=rul_loss,
                feasibility_weight=c.get("feasibility_weight", 0.1),
                monotonicity_weight=c.get("monotonicity_weight", 0.05),
            )
        return rul_loss

    # ------------------------------------------------------------------
    # Lambda-theta schedule
    # ------------------------------------------------------------------

    def _get_lambda_theta(self, epoch: int) -> float:
        """Return scheduled lambda_theta for the given epoch (1-indexed).

        Schedule options
        ----------------
        step    : start for epoch <= warmup, else end  (original default)
        linear  : linearly interpolates start->end over warmup epochs
        delayed : 0 for warmup epochs, then ramps 0->end over the NEXT
                  warmup epochs.  Best with dense training (stride=5):
                  let the RUL head converge before theta supervision.
        """
        c = self.config
        start    = c.get("lambda_theta_start", c.get("lambda_theta", 0.0))
        end      = c.get("lambda_theta_end", start)
        warmup   = c.get("lambda_theta_warmup_epochs", 0)
        schedule = c.get("lambda_theta_schedule", "step")

        if warmup <= 0 or start == end:
            return start

        if schedule == "linear":
            t = min((epoch - 1) / max(warmup, 1), 1.0)
            return start + (end - start) * t

        elif schedule == "delayed":
            # Phase 1 (epochs 1..warmup): lambda=0 (pure RUL training)
            # Phase 2 (warmup+1..2*warmup): ramp 0 -> end
            # Phase 3 (> 2*warmup): constant end
            if epoch <= warmup:
                return 0.0
            ramp_t = min((epoch - warmup) / max(warmup, 1), 1.0)
            return end * ramp_t

        else:  # "step"
            return start if epoch <= warmup else end

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, n_epochs: int | None = None) -> None:
        """Run the full training loop."""
        epochs  = n_epochs or self.config.get("epochs", 100)
        patience = self.config.get("early_stopping_patience", 15)

        val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Apply lambda-theta schedule
            if isinstance(self.criterion, CompositeLoss):
                lam = self._get_lambda_theta(epoch)
                self.criterion.lambda_theta = lam
                self.writer.add_scalar("Lambda/theta", lam, epoch)

            train_loss, train_comps, avg_grad_norm, train_stats = self._train_epoch(epoch)
            val_loss,   val_comps,   val_stats                  = self._val_epoch(epoch)
            self.scheduler.step()

            elapsed     = time.time() - t0
            current_lr  = self.scheduler.get_last_lr()[0]

            # Build compact component breakdown string for stdout
            comp_parts: list[str] = []
            if len(train_comps) > 1:
                for k, v in train_comps.items():
                    comp_parts.append(f"tr_{k}={v:.3f}")
            if len(val_comps) > 1:
                for k, v in val_comps.items():
                    comp_parts.append(f"val_{k}={v:.3f}")
            comp_str = ("  [" + "  ".join(comp_parts) + "]") if comp_parts else ""

            lam_str = ""
            if isinstance(self.criterion, CompositeLoss):
                lam_str = f"  lam_th={self.criterion.lambda_theta:.3f}"

            logger.info(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"val_mae={val_stats['mae']:.2f}  "
                f"val_r={val_stats.get('pearson_r', float('nan')):.3f}  "
                f"gnorm={avg_grad_norm:.3f}"
                f"{lam_str}  "
                f"lr={current_lr:.2e}  "
                f"({elapsed:.1f}s)"
                f"{comp_str}"
            )

            # ── TensorBoard: losses ────────────────────────────────────
            self.writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            self.writer.add_scalar("Loss/val_epoch",   val_loss,   epoch)
            self.writer.add_scalar("LR",               current_lr, epoch)
            for k, v in train_comps.items():
                self.writer.add_scalar(f"Loss/train_{k}_epoch", v, epoch)
            for k, v in val_comps.items():
                self.writer.add_scalar(f"Loss/val_{k}_epoch",   v, epoch)

            # ── TensorBoard: gradient + schedule ──────────────────────
            self.writer.add_scalar("GradNorm/train_epoch", avg_grad_norm, epoch)

            # ── TensorBoard: plausibility (Train/) ────────────────────
            for key, val in train_stats.items():
                self.writer.add_scalar(f"Train/{key}", val, epoch)

            # ── TensorBoard: plausibility (Val/) ──────────────────────
            for key, val in val_stats.items():
                self.writer.add_scalar(f"Val/{key}", val, epoch)

            # ── TensorBoard: model weight norm ────────────────────────
            param_norm = float(
                sum(p.data.norm(2).item() ** 2 for p in self.model.parameters()) ** 0.5
            )
            self.writer.add_scalar("Weights/param_norm", param_norm, epoch)

            # ── Checkpointing ─────────────────────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint("best.pt", epoch, val_loss)
            else:
                self.epochs_without_improvement += 1

            if patience and self.epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        self._save_checkpoint("last.pt", epochs, val_loss)
        self.writer.close()

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self, epoch: int
    ) -> tuple[float, dict[str, float], float, dict[str, float]]:
        """Run one training epoch.

        Returns:
            avg_loss       -- mean total loss across batches
            avg_comps      -- mean per-component losses
            avg_grad_norm  -- mean pre-clip gradient L2 norm
            stats          -- plausibility stats dict (mae, bias)
        """
        self.model.train()
        total_loss     = 0.0
        comp_totals: dict[str, float] = {}
        total_grad_norm = 0.0

        # Prediction stats accumulated per-batch (cheap, stays on device)
        total_mae  = 0.0
        total_bias = 0.0
        n_batches  = 0

        # mininterval=2s: redraw at most once every 2 s regardless of batch count.
        # This prevents 2000+ Jupyter HTML renders per epoch which add ~5 min on T4.
        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False, mininterval=2.0)
        for batch in pbar:
            batch = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch]
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, components, rul_pred, _ = self._compute_loss(batch, return_predictions=True)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Compute pre-clip gradient norm; clip_max=inf means no actual clipping
            clip_max = self.grad_clip_norm if self.grad_clip_norm else float("inf")
            grad_norm_tensor = nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=clip_max
            )
            total_grad_norm += grad_norm_tensor.item()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            for k, v in components.items():
                comp_totals[k] = comp_totals.get(k, 0.0) + v.item()

            # Batch prediction stats (no extra forward pass needed)
            rul_true = batch[1]
            err = rul_pred - rul_true
            total_mae  += err.abs().mean().item()
            total_bias += err.mean().item()
            n_batches  += 1

            self.global_step += 1
            # Per-step TB logging is gated to avoid thousands of writes per epoch.
            # Enabled only when tb_log_every_n_steps > 0 in config.
            if self._tb_log_step > 0 and (self.global_step % self._tb_log_step == 0):
                self.writer.add_scalar("Loss/train_step",     loss.item(),             self.global_step)
                self.writer.add_scalar("GradNorm/train_step", grad_norm_tensor.item(), self.global_step)
                for k, v in components.items():
                    self.writer.add_scalar(f"Loss/{k}_step",  v.item(),                self.global_step)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                gnorm=f"{grad_norm_tensor.item():.2f}",
            )

        n = len(self.train_loader)
        avg_comps = {k: v / n for k, v in comp_totals.items()}
        train_stats = {
            "mae":  total_mae  / max(n_batches, 1),
            "bias": total_bias / max(n_batches, 1),
        }
        return total_loss / n, avg_comps, total_grad_norm / n, train_stats

    @torch.no_grad()
    def _val_epoch(
        self, epoch: int
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Run one validation epoch.

        Returns:
            avg_loss -- mean total loss
            avg_comps -- mean per-component losses
            stats -- plausibility + physics stats dict
        """
        self.model.eval()
        total_loss = 0.0
        comp_totals: dict[str, float] = {}

        # Collect predictions for plausibility stats (all on CPU to avoid OOM)
        preds_list:      list[torch.Tensor] = []
        targets_list:    list[torch.Tensor] = []
        theta_hat_list:  list[torch.Tensor] = []
        theta_true_list: list[torch.Tensor] = []

        for batch in self.val_loader:
            batch = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch]
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, components, rul_pred, theta_hat = self._compute_loss(
                    batch, return_predictions=True
                )

            total_loss += loss.item()
            for k, v in components.items():
                comp_totals[k] = comp_totals.get(k, 0.0) + v.item()

            # Collect for stats (move to CPU to avoid accumulating GPU memory)
            preds_list.append(rul_pred.cpu().float())
            targets_list.append(batch[1].cpu().float())
            if theta_hat is not None:
                theta_hat_list.append(theta_hat.cpu().float())
            # Only collect batch[2] as theta_true when the model actually uses it;
            # avoids mixing up the (x, rul, ops) 3-tuple with (x, rul, theta_true).
            has_theta_model = getattr(self.model, "theta_true_dim", 0) > 0
            has_ops_model   = getattr(self.model, "ops_dim", 0) > 0
            if has_theta_model and len(batch) >= 3 and isinstance(batch[2], torch.Tensor):
                theta_true_list.append(batch[2].cpu().float())
            elif not has_theta_model and len(batch) >= 4 and isinstance(batch[2], torch.Tensor):
                # 4-tuple: batch[2] = theta_true, batch[3] = ops
                theta_true_list.append(batch[2].cpu().float())

        n = len(self.val_loader)
        avg_comps = {k: v / n for k, v in comp_totals.items()}

        # Compute and log plausibility + physics stats
        val_stats = self._compute_stats(
            epoch, preds_list, targets_list, theta_hat_list, theta_true_list
        )
        return total_loss / n, avg_comps, val_stats

    def _compute_stats(
        self,
        epoch: int,
        preds_list: list[torch.Tensor],
        targets_list: list[torch.Tensor],
        theta_hat_list: list[torch.Tensor],
        theta_true_list: list[torch.Tensor],
    ) -> dict[str, float]:
        """Compute plausibility and physics stats from collected val predictions.

        Returns a flat dict suitable for logging under the Val/ TensorBoard prefix.
        """
        p = torch.cat(preds_list).numpy()
        t = torch.cat(targets_list).numpy()

        err  = p - t
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))          # + = systematic over-estimation

        # Pearson correlation: measures whether pred tracks target trend.
        # Drops to 0 when model collapses to a constant; watch for sudden dips.
        if np.std(p) > 1e-6 and np.std(t) > 1e-6:
            r = float(np.corrcoef(p, t)[0, 1])
        else:
            r = 0.0

        stats: dict[str, float] = {
            "mae":         mae,
            "rmse":        rmse,
            "bias":        bias,
            "pearson_r":   r,
            "pred_mean":   float(np.mean(p)),
            "pred_std":    float(np.std(p)),
            "pred_min":    float(np.min(p)),
            "pred_max":    float(np.max(p)),
            "target_mean": float(np.mean(t)),   # constant reference
        }

        # ── Theta / health-parameter stats ────────────────────────────
        # Only available when model is CycleLayerNetV1 and val loader
        # has theta_true (return_theta_true=True in the dataset).
        if theta_hat_list and theta_true_list:
            th = torch.cat(theta_hat_list).numpy()   # (N, n_params)
            tt = torch.cat(theta_true_list).numpy()  # (N, n_params)
            n_params = th.shape[1]

            per_ch_mae = np.mean(np.abs(th - tt), axis=0)   # (n_params,)

            # Overall summary
            stats["theta_mae_mean"] = float(np.mean(per_ch_mae))

            # Per-channel: use known N-CMAPSS names if n_params matches
            names = _THETA_NAMES[:n_params] if n_params <= len(_THETA_NAMES) else \
                    [f"ch{i:02d}" for i in range(n_params)]
            for i, name in enumerate(names):
                stats[f"theta_mae_{name}"]       = float(per_ch_mae[i])
                stats[f"theta_pred_mean_{name}"] = float(np.mean(th[:, i]))
                stats[f"theta_true_mean_{name}"] = float(np.mean(tt[:, i]))

        return stats

    def _compute_loss(
        self,
        batch: list[torch.Tensor],
        return_predictions: bool = False,
    ) -> tuple:
        """Unified loss computation for all model/loss combinations.

        Args:
            batch: List of tensors from the DataLoader.
            return_predictions: If True, return a 4-tuple
                (loss, components, rul_pred_detached, theta_hat_detached_or_None).
                If False (default), return (loss, components).

        Handles:
          - (x, rul)                  + any model          -> RULLoss
          - (x, rul, theta_true)      + theta-input model  -> CompositeLoss
          - (x, rul, ops)             + ops model          -> RULLoss / CompositeLoss
          - (x, rul, theta_true, ops) + theta+ops model    -> CompositeLoss

        Disambiguation of 3-tuple batch[2]:
          = ops        when model.ops_dim > 0 AND model.theta_true_dim == 0
          = theta_true otherwise (existing convention)
        """
        x        = batch[0]
        rul_true = batch[1]
        theta_hat = None

        has_ops_model   = getattr(self.model, "ops_dim", 0) > 0
        has_theta_model = getattr(self.model, "theta_true_dim", 0) > 0

        # Unpack optional extras based on batch length and model type
        if len(batch) >= 4:
            # 4-tuple: (x, rul, theta_true, ops)
            theta_true = batch[2]
            ops        = batch[3]
        elif len(batch) >= 3:
            if has_ops_model and not has_theta_model:
                # 3-tuple (x, rul, ops) — ops-only baseline
                theta_true = None
                ops        = batch[2]
            else:
                # 3-tuple (x, rul, theta_true) — existing convention
                theta_true = batch[2]
                ops        = None
        else:
            theta_true = None
            ops        = None

        # Forward pass — all extra inputs as keyword arguments (backward compatible)
        fwd_kwargs: dict = {}
        if has_theta_model and theta_true is not None:
            fwd_kwargs["theta_true"] = theta_true
        if has_ops_model and ops is not None:
            fwd_kwargs["ops"] = ops

        if fwd_kwargs:
            model_out = self.model(x, **fwd_kwargs)
        else:
            model_out = self.model(x)

        multitask = isinstance(model_out, tuple)
        if multitask:
            rul_pred, theta_hat = model_out
        else:
            rul_pred = model_out

        # Loss computation
        if multitask and theta_true is not None and isinstance(self.criterion, CompositeLoss):
            loss, components = self.criterion(rul_pred, rul_true, theta_hat, theta_true)

        elif isinstance(self.criterion, PhysicsInformedLoss):
            if hasattr(self.model, "forward_with_intermediates"):
                # v0 physics model: need intermediate features
                rul_pred, _, cycle_features = self.model.forward_with_intermediates(x)
                loss, components = self.criterion(rul_pred, rul_true, cycle_features)
            else:
                loss_val   = self.criterion.rul_loss(rul_pred, rul_true)
                loss       = loss_val
                components = {"rul": loss_val.detach()}

        else:
            loss_val   = self.criterion(rul_pred, rul_true)
            loss       = loss_val
            components = {"rul": loss_val.detach()}

        if return_predictions:
            return (
                loss,
                components,
                rul_pred.detach(),
                theta_hat.detach() if theta_hat is not None else None,
            )
        return loss, components

    def _save_checkpoint(self, name: str, epoch: int, val_loss: float) -> None:
        path = self.output_dir / name
        torch.save(
            {
                "epoch":          epoch,
                "model":          self.model.state_dict(),
                "optimizer":      self.optimizer.state_dict(),
                "best_val_loss":  self.best_val_loss,
                "val_loss":       val_loss,
            },
            path,
        )
        logger.info(f"Saved checkpoint -> {path}")
