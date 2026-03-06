"""Training loop supporting single-task and multi-task (Phase 2) models.

Batch format
------------
DataLoader may yield either:
    (x, rul)                  -- standard 2-tuple
    (x, rul, theta_true)      -- 3-tuple when return_theta_true=True

Model forward may return either:
    rul                       -- scalar RUL (baselines, CycleLayerNet v0)
    (rul, theta_hat)          -- 2-tuple (CycleLayerNetV1)

Loss selection
--------------
    Model returns scalar  +  2-tuple batch  -> RULLoss or PhysicsInformedLoss
    Model returns tuple   +  3-tuple batch  -> CompositeLoss

Lambda-theta schedule
---------------------
    lambda_theta_start (float)      – initial lambda value
    lambda_theta_end   (float)      – final lambda value
    lambda_theta_warmup_epochs (int)– epochs to keep start value (step) or
                                      ramp from start→end (linear)
    lambda_theta_schedule ("step"|"linear")  default: "step"

    Step  : lambda = start for epoch ≤ warmup, else end
    Linear: lambda linearly interpolates start→end over warmup epochs

Loss breakdown logging
----------------------
When CompositeLoss is used, per-epoch component means (rul, theta) are
logged to both TensorBoard and stdout.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cyclelayer.training.losses import CompositeLoss, PhysicsInformedLoss, RULLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop with TensorBoard logging and checkpointing.

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
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.global_step = 0

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
        """Return the scheduled lambda_theta value for the given epoch (1-indexed)."""
        c = self.config
        start    = c.get("lambda_theta_start", c.get("lambda_theta", 0.0))
        end      = c.get("lambda_theta_end", start)
        warmup   = c.get("lambda_theta_warmup_epochs", 0)
        schedule = c.get("lambda_theta_schedule", "step")

        if warmup <= 0 or start == end:
            return start

        if schedule == "linear":
            # interpolate start → end over warmup epochs; clamp at end afterwards
            t = min((epoch - 1) / max(warmup, 1), 1.0)
            return start + (end - start) * t
        else:  # "step"
            return start if epoch <= warmup else end

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, n_epochs: int | None = None) -> None:
        """Run the full training loop."""
        epochs = n_epochs or self.config.get("epochs", 100)
        patience = self.config.get("early_stopping_patience", 15)

        val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Apply lambda-theta schedule
            if isinstance(self.criterion, CompositeLoss):
                lam = self._get_lambda_theta(epoch)
                self.criterion.lambda_theta = lam
                self.writer.add_scalar("Lambda/theta", lam, epoch)

            train_loss, train_comps = self._train_epoch(epoch)
            val_loss, val_comps     = self._val_epoch(epoch)
            self.scheduler.step()

            elapsed = time.time() - t0

            # Build component breakdown string for stdout
            comp_parts: list[str] = []
            if len(train_comps) > 1:
                for k, v in train_comps.items():
                    comp_parts.append(f"tr_{k}={v:.3f}")
            if len(val_comps) > 1:
                for k, v in val_comps.items():
                    comp_parts.append(f"val_{k}={v:.3f}")
            comp_str = ("  [" + "  ".join(comp_parts) + "]") if comp_parts else ""

            logger.info(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}  "
                f"({elapsed:.1f}s){comp_str}"
            )

            # TensorBoard: epoch-level totals
            self.writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            self.writer.add_scalar("Loss/val_epoch", val_loss, epoch)
            self.writer.add_scalar("Loss/train_total_epoch", train_loss, epoch)
            self.writer.add_scalar("Loss/val_total_epoch", val_loss, epoch)
            self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)
            for k, v in train_comps.items():
                self.writer.add_scalar(f"Loss/train_{k}_epoch", v, epoch)
            for k, v in val_comps.items():
                self.writer.add_scalar(f"Loss/val_{k}_epoch", v, epoch)

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

    def _train_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        comp_totals: dict[str, float] = {}

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for batch in pbar:
            batch = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch]
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, components = self._compute_loss(batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            for k, v in components.items():
                comp_totals[k] = comp_totals.get(k, 0.0) + v.item()

            self.global_step += 1
            self.writer.add_scalar("Loss/train_step", loss.item(), self.global_step)
            for k, v in components.items():
                self.writer.add_scalar(f"Loss/{k}_step", v.item(), self.global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n = len(self.train_loader)
        avg_comps = {k: v / n for k, v in comp_totals.items()}
        return total_loss / n, avg_comps

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        comp_totals: dict[str, float] = {}

        for batch in self.val_loader:
            batch = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch]
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, components = self._compute_loss(batch)
            total_loss += loss.item()
            for k, v in components.items():
                comp_totals[k] = comp_totals.get(k, 0.0) + v.item()

        n = len(self.val_loader)
        avg_comps = {k: v / n for k, v in comp_totals.items()}
        return total_loss / n, avg_comps

    def _compute_loss(
        self, batch: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Unified loss computation for all model/loss combinations.

        Handles:
          - (x, rul)                 + scalar-output model  -> RULLoss
          - (x, rul)                 + tuple-output model   -> RULLoss on rul
          - (x, rul, theta_true)     + scalar-output model  -> RULLoss
          - (x, rul, theta_true)     + tuple-output model   -> CompositeLoss
        """
        x        = batch[0]
        rul_true = batch[1]
        theta_true = batch[2] if len(batch) >= 3 else None

        if theta_true is not None and getattr(self.model, "theta_true_dim", 0) > 0:
            model_out = self.model(x, theta_true)
        else:
            model_out = self.model(x)
        multitask = isinstance(model_out, tuple)

        if multitask:
            rul_pred, theta_hat = model_out
        else:
            rul_pred = model_out

        if multitask and theta_true is not None and isinstance(self.criterion, CompositeLoss):
            return self.criterion(rul_pred, rul_true, theta_hat, theta_true)

        if isinstance(self.criterion, PhysicsInformedLoss):
            # PhysicsInformedLoss needs cycle_features from CycleLayerNet v0
            if hasattr(self.model, "forward_with_intermediates"):
                rul_pred, _, cycle_features = self.model.forward_with_intermediates(x)
                return self.criterion(rul_pred, rul_true, cycle_features)

        # Fallback: plain RULLoss
        loss = self.criterion(rul_pred, rul_true)
        return loss, {"rul": loss.detach()}

    def _save_checkpoint(self, name: str, epoch: int, val_loss: float) -> None:
        path = self.output_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "val_loss": val_loss,
            },
            path,
        )
        logger.info(f"Saved checkpoint -> {path}")
