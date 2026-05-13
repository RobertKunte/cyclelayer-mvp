"""Tiny DS02 smoke-training script for V3.1b thermal-auxiliary CycleLayerV3.

PURPOSE: small local sanity run (1–2 epochs, capped sample count) — NOT a
full training. Full training requires explicit approval per the V3.1b
plan. See docs/decisions/ADR-0012-v3-thermal-auxiliary-scope.md.

Constraints (enforced):
  * NO EPR / pressure loss (asserted by `CycleLayerV3Loss`)
  * NO C0 parameter tuning
  * NO supervised L_θ on theta_phys
  * No automatic adoption of trained weights — checkpoint stored locally

Usage:
    python scripts/train_cyclelayer_v3_thermal_aux_smoke.py
    python scripts/train_cyclelayer_v3_thermal_aux_smoke.py --config configs/cyclelayer_v3_thermal_aux.yaml --max_train_samples 5000 --epochs 1

Outputs under `artifacts/cyclelayer_v3/thermal_aux_smoke/`:
    train_log.csv         — per-epoch loss components, theta stats, T-RMSE diagnostics
    best.pt               — model state_dict at best val loss
    last.pt               — final state
    sensor_scaler.npz     — per-sensor mean/std (X_s, 14 cols)
    ops_scaler.npz        — per-ops mean/std    (W, 4 cols)
    sigma_train.json      — σ_T24/T30/T50, σ_lpt_flow computed from train units
    summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np   # noqa: E402
import torch         # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml          # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset                # noqa: E402
from cyclelayer.losses import CycleLayerV3Loss, V3LossConfig            # noqa: E402
from cyclelayer.models import units                                     # noqa: E402
from cyclelayer.models.brayton_engine import (                          # noqa: E402
    BraytonEngine, BraytonEngineConfig, InletFlowParams, MapCoefficients,
)
from cyclelayer.models.cyclelayer_v3 import CycleLayerV3, CycleLayerV3Config  # noqa: E402


# =============================================================================
# Windowed dataset wrapper over the raw per-row NCMAPSSV3Dataset
# =============================================================================
# Lazy: build windows by index references into the underlying arrays.
# Windows do NOT cross unit boundaries (per-unit indices only).

class NCMAPSSV3WindowedDataset(Dataset):
    """Sliding-window wrapper over `NCMAPSSV3Dataset` raw rows.

    Each item returns a window-tensor block plus per-window scalars:

        sensors_imp:  (T, 14)        raw Imperial sensors (X_s)
        ops_imp:      (T, 4)         raw Imperial W cols
        T2_R:         (T,)           from W
        targets_K_last: dict with T24_K, T30_K, T50_K  (last timestep, SI)
        ops_si_last:    dict T2_K, P2_Pa, alt_m, mach (last timestep, SI)
        sens_si_last:   dict Nf_rpm, Nc_rpm, Wf_kgs    (last timestep, SI)
        health_gt_last: dict HPT_eff_mod, LPT_eff_mod, LPT_flow_mod (last)
        RUL:          scalar        (last-timestep RUL, the target)
        unit_id:      scalar
    """

    def __init__(
        self,
        base: NCMAPSSV3Dataset,
        unit_ids: list[int],
        window_size: int = 50,
        stride: int = 5,
        max_samples: int | None = None,
    ) -> None:
        self.base = base
        self.window_size = window_size
        # We need direct array access; use the in-memory arrays when present.
        # Force in-memory mode for arrays so we can slice without per-row
        # HDF5 reads — synthetic smoke dataset is small enough.
        if not base.load_in_memory:
            raise RuntimeError(
                "NCMAPSSV3WindowedDataset currently requires "
                "load_in_memory=True on the underlying dataset. The smoke "
                "script loads DS02 dev split into memory once at startup; "
                "memory usage is ~600 MB."
            )
        W  = base._W     # (N, 4)
        X  = base._X     # (N, 14)
        T  = base._T     # (N, 10)
        A  = base._A     # (N, 4)  unit_id, cycle, Fc, hs
        Y  = base._Y     # (N,)    RUL

        unit_arr = A[:, 0].astype(np.int64)
        self.indices: list[tuple[int, int]] = []   # (start_row, end_row_exclusive)
        for uid in unit_ids:
            mask = unit_arr == uid
            idxs = np.nonzero(mask)[0]
            if len(idxs) < window_size:
                continue
            # Take contiguous runs only — DS02 unit rows are already
            # contiguous per cycle, but use a sliding window.
            start, end = int(idxs[0]), int(idxs[-1]) + 1
            for win_start in range(start, end - window_size + 1, stride):
                self.indices.append((win_start, win_start + window_size))

        if max_samples is not None and len(self.indices) > max_samples:
            rng = np.random.default_rng(0)
            self.indices = list(rng.choice(self.indices, size=max_samples, replace=False))
            # Convert numpy tuples back to (int, int)
            self.indices = [(int(a), int(b)) for a, b in self.indices]

        # Store array refs
        self._W, self._X, self._T, self._A, self._Y = W, X, T, A, Y

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        s, e = self.indices[idx]
        last = e - 1
        sensors_imp = torch.from_numpy(self._X[s:e]).float()     # (T, 14)
        ops_imp     = torch.from_numpy(self._W[s:e]).float()     # (T, 4)
        # Last-timestep SI
        alt_ft, mach, _, T2_R = self._W[last]
        T2_R_t  = T2_R
        P2_psia = self._X[last, 5]      # X_s column 5 = P2
        Nf_rpm  = self._X[last, 11]
        Nc_rpm  = self._X[last, 12]
        Wf_pps  = self._X[last, 13]

        ops_si_last = {
            "T2_K":  torch.tensor(float(T2_R * units.RANK_TO_K)),
            "P2_Pa": torch.tensor(float(P2_psia * units.PSIA_TO_PA)),
            "alt_m": torch.tensor(float(alt_ft * units.FT_TO_M)),
            "mach":  torch.tensor(float(mach)),
        }
        sens_si_last = {
            "Nf_rpm": torch.tensor(float(Nf_rpm)),
            "Nc_rpm": torch.tensor(float(Nc_rpm)),
            "Wf_kgs": torch.tensor(float(Wf_pps * units.PPS_TO_KGS)),
        }
        # L_temp targets (last timestep, in Kelvin) from X_s rankine columns
        targets_K_last = {
            "T24_K": torch.tensor(float(self._X[last, 0] * units.RANK_TO_K)),
            "T30_K": torch.tensor(float(self._X[last, 1] * units.RANK_TO_K)),
            "T50_K": torch.tensor(float(self._X[last, 3] * units.RANK_TO_K)),
        }
        # Health GT (last timestep) — columns 6 (HPT_eff), 8 (LPT_eff), 9 (LPT_flow)
        health_gt_last = {
            "HPT_eff_mod":  torch.tensor(float(self._T[last, 6])),
            "LPT_eff_mod":  torch.tensor(float(self._T[last, 8])),
            "LPT_flow_mod": torch.tensor(float(self._T[last, 9])),
        }
        RUL = torch.tensor(float(self._Y[last]))
        unit_id = torch.tensor(int(self._A[last, 0]), dtype=torch.long)

        return {
            "sensors_imp":     sensors_imp,
            "ops_imp":         ops_imp,
            "ops_si_last":     ops_si_last,
            "sens_si_last":    sens_si_last,
            "targets_K_last":  targets_K_last,
            "health_gt_last":  health_gt_last,
            "RUL":             RUL,
            "unit_id":         unit_id,
        }


def _collate(batch: list[dict]) -> dict:
    """Stack dicts of mixed tensors into a single batched dict."""
    def stack_simple(key: str) -> torch.Tensor:
        return torch.stack([b[key] for b in batch], dim=0)

    def stack_dict(key: str) -> dict[str, torch.Tensor]:
        keys = list(batch[0][key].keys())
        return {k: torch.stack([b[key][k] for b in batch], dim=0) for k in keys}

    return {
        "sensors_imp":     stack_simple("sensors_imp"),
        "ops_imp":         stack_simple("ops_imp"),
        "ops_si_last":     stack_dict("ops_si_last"),
        "sens_si_last":    stack_dict("sens_si_last"),
        "targets_K_last":  stack_dict("targets_K_last"),
        "health_gt_last":  stack_dict("health_gt_last"),
        "RUL":             stack_simple("RUL"),
        "unit_id":         stack_simple("unit_id"),
    }


# =============================================================================
# Train-split statistics (no leakage)
# =============================================================================

def fit_sensor_ops_scalers(base: NCMAPSSV3Dataset, train_units: list[int]) -> dict:
    """Compute per-channel mean/std on TRAIN units only.

    Returns dict with sensor_mean (14,), sensor_std (14,), ops_mean (4,),
    ops_std (4,).
    """
    A = base._A; unit_arr = A[:, 0].astype(np.int64)
    mask = np.isin(unit_arr, train_units)
    if not mask.any():
        raise ValueError(f"No train rows for units {train_units}")
    X = base._X[mask]    # (M, 14)
    W = base._W[mask]    # (M, 4)
    return {
        "sensor_mean": X.mean(axis=0),
        "sensor_std":  np.clip(X.std(axis=0), 1e-6, None),
        "ops_mean":    W.mean(axis=0),
        "ops_std":     np.clip(W.std(axis=0), 1e-6, None),
    }


def fit_temp_sigmas_K(base: NCMAPSSV3Dataset, train_units: list[int]) -> dict[str, float]:
    """Compute σ_T24/T30/T50 in Kelvin from train units.

    Used to normalise the L_temp loss.  Returns plain Python floats.
    """
    A = base._A; unit_arr = A[:, 0].astype(np.int64)
    mask = np.isin(unit_arr, train_units)
    X = base._X[mask]    # (M, 14)
    # X_s columns: T24=0, T30=1, T48=2, T50=3 (all °R)
    factor = units.RANK_TO_K
    return {
        "T24": float(np.clip(X[:, 0].std(), 1e-3, None) * factor),
        "T30": float(np.clip(X[:, 1].std(), 1e-3, None) * factor),
        "T50": float(np.clip(X[:, 3].std(), 1e-3, None) * factor),
    }


def fit_lpt_flow_sigma(base: NCMAPSSV3Dataset, train_units: list[int]) -> float:
    A = base._A; unit_arr = A[:, 0].astype(np.int64)
    mask = np.isin(unit_arr, train_units)
    Tarr = base._T[mask]
    s = float(Tarr[:, 9].std())   # column 9 = LPT_flow_mod
    return max(s, 1e-3)


# =============================================================================
# BraytonEngine constructor from YAML model.brayton_engine block
# =============================================================================

def build_brayton_from_cfg(b_cfg: dict) -> BraytonEngine:
    inlet = b_cfg["inlet_flow"]
    maps  = b_cfg["map_coeffs"]
    cfg = BraytonEngineConfig(
        inlet_flow=InletFlowParams(
            Wc_fan_design=float(inlet["Wc_fan_design"]),
            Nc_fan_design=float(inlet["Nc_fan_design"]),
            c1=float(inlet.get("c1", 0.85)),
            c2=float(inlet.get("c2", -0.20)),
            Wc_min=float(inlet.get("Wc_min", 100.0)),
            Wc_max=float(inlet.get("Wc_max", 1100.0)),
        ),
        map_coeffs=MapCoefficients(
            Nc_design_fan=float(maps["Nc_design_fan"]),
            Wc_design_fan=float(maps["Wc_design_fan"]),
            Nc_design_lpc=float(maps["Nc_design_lpc"]),
            Wc_design_lpc=float(maps["Wc_design_lpc"]),
            Nc_design_hpc=float(maps["Nc_design_hpc"]),
            Wc_design_hpc=float(maps["Wc_design_hpc"]),
            PR_design_fan=float(maps["PR_design_fan"]),
            PR_design_lpc=float(maps["PR_design_lpc"]),
            PR_design_hpc=float(maps["PR_design_hpc"]),
            eta_design_fan=float(maps["eta_design_fan"]),
            eta_design_lpc=float(maps["eta_design_lpc"]),
            eta_design_hpc=float(maps["eta_design_hpc"]),
            pr_a1=float(maps.get("pr_a1", 0.10)),
            pr_a2=float(maps.get("pr_a2", -0.05)),
            pr_b1=float(maps.get("pr_b1", -0.08)),
            pr_b2=float(maps.get("pr_b2", -0.03)),
            eta_e1=float(maps.get("eta_e1", 0.05)),
            eta_e2=float(maps.get("eta_e2", 0.03)),
        ),
        use_measured_inlet=bool(b_cfg.get("use_measured_inlet", True)),
        bpr_design=float(b_cfg.get("bpr_design", 5.5)),
        eta_design_hpt=float(b_cfg.get("eta_design_hpt", 0.90)),
        eta_design_lpt=float(b_cfg.get("eta_design_lpt", 0.92)),
    )
    return BraytonEngine(cfg)


def build_v3_from_cfg(m_cfg: dict, brayton: BraytonEngine) -> CycleLayerV3:
    enc = m_cfg["encoder"]; ops = m_cfg["ops_encoder"]
    pm  = m_cfg["param_head_phys"]; ah = m_cfg["aux_health_head"]
    pg  = m_cfg["prognostics_head"]
    cfg = CycleLayerV3Config(
        n_sensors=14, ops_dim=4, window_size=50,
        encoder_channels=list(enc["channels"]),
        encoder_kernel_size=int(enc["kernel_size"]),
        encoder_mlp_hidden=128,
        encoder_dropout=float(enc.get("dropout", 0.2)),
        encoder_out_dim=64,
        use_ops_encoder=bool(ops.get("enabled", True)),
        ops_channels=list(ops["channels"]),
        ops_out_dim=int(ops["out_dim"]),
        mask_target_sensor_prob=float(enc.get("mask_target_sensor_prob", 0.5)),
        target_sensor_indices=list(enc.get("target_sensor_indices", [0, 1, 3])),
        param_hidden=list(pm["hidden"]),
        theta_dim=int(pm["theta_dim"]),
        theta_bounds=tuple(pm["bounds"]),
        initial_theta_target=float(pm["initial_theta_target"]),
        aux_hidden=list(ah["hidden"]),
        aux_bounds=tuple(ah["output_bounds"]),
        initial_aux_target=float(ah["initial_value_target"]),
        detach_aux_to_rul=bool(ah.get("detach_for_rul", True)),
        prog_hidden=list(pg["hidden"]),
        prog_dropout=float(pg.get("dropout", 0.2)),
        max_rul=99.0,
        use_theta_in_rul=bool(pg.get("use_theta_in_rul", True)),
        detach_theta_to_rul=bool(pg.get("detach_theta_to_rul", False)),
        use_aux_in_rul=True,
    )
    return CycleLayerV3(cfg, brayton_engine=brayton)


# =============================================================================
# Training loop (tiny smoke)
# =============================================================================

def run_epoch(
    model: CycleLayerV3,
    loss_fn: CycleLayerV3Loss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    sensor_mean: torch.Tensor, sensor_std: torch.Tensor,
    ops_mean: torch.Tensor, ops_std: torch.Tensor,
    epoch: int, tag: str,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    comps_sum: dict[str, float] = {}
    rul_errs: list[float] = []
    theta_all: list[torch.Tensor] = []
    t24_errs: list[float] = []
    t30_errs: list[float] = []
    t50_errs: list[float] = []
    n_batches = 0
    epr_diag: list[float] = []

    sensor_mean_d = sensor_mean.to(device); sensor_std_d = sensor_std.to(device)
    ops_mean_d    = ops_mean.to(device);    ops_std_d    = ops_std.to(device)

    for batch in loader:
        # Move tensors to device
        sensors_imp = batch["sensors_imp"].to(device)   # (B, T, 14)
        ops_imp     = batch["ops_imp"].to(device)       # (B, T, 4)
        ops_si      = {k: v.to(device) for k, v in batch["ops_si_last"].items()}
        sens_si     = {k: v.to(device) for k, v in batch["sens_si_last"].items()}
        temp_true_K = {k: v.to(device) for k, v in batch["targets_K_last"].items()}
        lpt_flow_true = batch["health_gt_last"]["LPT_flow_mod"].to(device)
        rul_true    = batch["RUL"].to(device)

        # Normalise sensors/ops (per-channel std-mean from train units)
        sensors_norm = (sensors_imp - sensor_mean_d) / sensor_std_d
        ops_norm     = (ops_imp     - ops_mean_d)    / ops_std_d

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        out = model(sensors_norm, ops_norm, ops_si=ops_si, sens_si=sens_si)

        # Temperature predictions from BraytonEngine (last timestep — engine
        # is called once with last-timestep ops/sens → outputs are scalars
        # per batch sample, shape (B,))
        temp_preds_K = {
            "T24_K": out["brayton"]["sensors_pred_si"]["T24_K"],
            "T30_K": out["brayton"]["sensors_pred_si"]["T30_K"],
            "T50_K": out["brayton"]["sensors_pred_si"]["T50_K"],
        }

        total, comps = loss_fn(
            rul_pred=out["rul"],
            rul_true=rul_true,
            theta_phys=out["theta_phys"],
            lpt_flow_pred=out["lpt_flow_pred"],
            lpt_flow_true=lpt_flow_true,
            temp_preds_K=temp_preds_K,
            temp_true_K=temp_true_K,
        )

        if is_train:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        total_loss += float(total.item())
        for k, v in comps.items():
            comps_sum[k] = comps_sum.get(k, 0.0) + float(v.item())
        rul_errs.append(float((out["rul"] - rul_true).detach().pow(2).mean().sqrt().item()))
        theta_all.append(out["theta_phys"].detach().cpu())
        t24_errs.append(float((temp_preds_K["T24_K"] - temp_true_K["T24_K"]).abs().mean().item()))
        t30_errs.append(float((temp_preds_K["T30_K"] - temp_true_K["T30_K"]).abs().mean().item()))
        t50_errs.append(float((temp_preds_K["T50_K"] - temp_true_K["T50_K"]).abs().mean().item()))
        diag = out["brayton"]["diag"]
        # EPR diagnostic (NOT in loss)
        P50 = diag["P50"].detach()
        P2  = ops_si["P2_Pa"]
        epr_diag.append(float((P50 / P2).mean().item()))
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_comps = {k: v / max(n_batches, 1) for k, v in comps_sum.items()}
    theta = torch.cat(theta_all, dim=0)
    theta_stats = {
        "mean": float(theta.mean().item()),
        "std":  float(theta.std().item()),
        "min":  float(theta.min().item()),
        "max":  float(theta.max().item()),
        "frac_at_lo": float((theta <= 0.851).float().mean().item()),
        "frac_at_hi": float((theta >= 0.999).float().mean().item()),
    }
    avg_rmse_rul = float(np.mean(rul_errs))
    avg_t24_mae  = float(np.mean(t24_errs))
    avg_t30_mae  = float(np.mean(t30_errs))
    avg_t50_mae  = float(np.mean(t50_errs))
    avg_epr      = float(np.mean(epr_diag))
    return {
        "epoch": epoch, "tag": tag,
        "loss": avg_loss, "components": avg_comps,
        "rul_rmse": avg_rmse_rul,
        "T24_mae_K": avg_t24_mae,
        "T30_mae_K": avg_t30_mae,
        "T50_mae_K": avg_t50_mae,
        "theta_stats": theta_stats,
        "EPR_pred_mean_diagnostic_only": avg_epr,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cyclelayer_v3_thermal_aux.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples",   type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    loss_cfg_yaml = cfg["loss"]

    # CLI overrides
    if args.epochs is not None:
        train_cfg["max_epochs"] = args.epochs
    if args.max_train_samples is not None:
        train_cfg["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        train_cfg["max_val_samples"] = args.max_val_samples
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    out_dir = Path(train_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"V3.1b thermal-aux smoke training (profile: {cfg.get('profile_name', '?')})")
    print(f"  config:    {cfg_path}")
    print(f"  device:    {device}")
    print(f"  output:    {out_dir}")
    print(f"  epochs:    {train_cfg['max_epochs']}")
    print(f"  bs:        {train_cfg['batch_size']}")

    # ── Tripwire on YAML (defence in depth) ──────────────────────────────
    assert not loss_cfg_yaml.get("use_pressure_loss", False), (
        "YAML enables pressure loss — V3.1b forbidden, see ADR-0012"
    )
    assert not loss_cfg_yaml.get("use_epr_loss", False), (
        "YAML enables EPR loss — V3.1b forbidden, see ADR-0012"
    )

    # ── Data ─────────────────────────────────────────────────────────────
    hdf5_path = Path(data_cfg["hdf5_path"])
    print(f"\nLoading DS02 into memory (one-time, ~600 MB)...")
    base = NCMAPSSV3Dataset(hdf5_path, split="dev", load_in_memory=True)
    print(f"  n_rows = {len(base):,}")
    print(f"  units  = {base.unit_ids.tolist()}")

    train_units = list(data_cfg["train_units"])
    val_units   = list(data_cfg["val_units"])

    print("\nComputing train-split scalers + sigmas (no leakage)...")
    scalers = fit_sensor_ops_scalers(base, train_units)
    sigma_T = fit_temp_sigmas_K(base, train_units)
    sigma_lpt = fit_lpt_flow_sigma(base, train_units)
    print(f"  sigma_T24_K = {sigma_T['T24']:.2f}  "
          f"sigma_T30_K = {sigma_T['T30']:.2f}  "
          f"sigma_T50_K = {sigma_T['T50']:.2f}")
    print(f"  sigma_lpt_flow = {sigma_lpt:.5f}")

    # Save scalers
    np.savez(out_dir / "sensor_scaler.npz",
             mean=scalers["sensor_mean"], std=scalers["sensor_std"])
    np.savez(out_dir / "ops_scaler.npz",
             mean=scalers["ops_mean"], std=scalers["ops_std"])
    (out_dir / "sigma_train.json").write_text(json.dumps({
        "sigma_T_K": sigma_T,
        "sigma_lpt_flow": sigma_lpt,
        "train_units": train_units,
        "val_units": val_units,
    }, indent=2))

    train_ds = NCMAPSSV3WindowedDataset(
        base, train_units,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride_train"],
        max_samples=train_cfg.get("max_train_samples"),
    )
    val_ds = NCMAPSSV3WindowedDataset(
        base, val_units,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride_eval"],
        max_samples=train_cfg.get("max_val_samples"),
    )
    print(f"\n  train windows = {len(train_ds):,}")
    print(f"  val windows   = {len(val_ds):,}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Empty train or val window set — check unit lists / window size.")

    bs = int(train_cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=0, collate_fn=_collate)

    # ── Model ────────────────────────────────────────────────────────────
    brayton = build_brayton_from_cfg(model_cfg["brayton_engine"])
    model = build_v3_from_cfg(model_cfg, brayton).to(device)
    print(f"\nModel: CycleLayerV3 (V3.1b thermal-aux)")
    print(f"  trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Loss (thermal-aux) ───────────────────────────────────────────────
    loss_cfg = V3LossConfig(
        lambda_rul=float(loss_cfg_yaml["lambda_rul"]),
        lambda_temp=float(loss_cfg_yaml["lambda_temp"]),
        lambda_aux=float(loss_cfg_yaml["lambda_aux"]),
        lambda_healthy=float(loss_cfg_yaml["lambda_healthy"]),
        lambda_smooth=float(loss_cfg_yaml["lambda_smooth"]),
        mse_weight=float(loss_cfg_yaml["rul"]["mse_weight"]),
        asymmetry=float(loss_cfg_yaml["rul"]["asymmetry"]),
        temp_sensors=list(loss_cfg_yaml["temp_sensors"]),
        sigma_temp_K=sigma_T,
        sigma_lpt_flow=sigma_lpt,
        healthy_rul_threshold=float(loss_cfg_yaml["healthy_rul_threshold"]),
        use_pressure_loss=False,
        use_epr_loss=False,
    )
    loss_fn = CycleLayerV3Loss(loss_cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    # Scalers as tensors
    sensor_mean = torch.from_numpy(scalers["sensor_mean"]).float()
    sensor_std  = torch.from_numpy(scalers["sensor_std"]).float()
    ops_mean    = torch.from_numpy(scalers["ops_mean"]).float()
    ops_std     = torch.from_numpy(scalers["ops_std"]).float()

    # ── Train ────────────────────────────────────────────────────────────
    n_epochs = int(train_cfg["max_epochs"])
    history: list[dict] = []
    best_val = float("inf")
    print(f"\nStarting tiny smoke training: {n_epochs} epoch(s)...")
    t0_all = time.time()
    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        train_stats = run_epoch(
            model, loss_fn, train_loader, optimizer, device,
            sensor_mean, sensor_std, ops_mean, ops_std,
            ep, "train",
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model, loss_fn, val_loader, None, device,
                sensor_mean, sensor_std, ops_mean, ops_std,
                ep, "val",
            )
        dt = time.time() - t0
        ts = train_stats["theta_stats"]; vs = val_stats["theta_stats"]
        print(
            f"  epoch {ep}/{n_epochs}  "
            f"tr_loss={train_stats['loss']:.3f}  "
            f"val_loss={val_stats['loss']:.3f}  "
            f"tr_RMSE={train_stats['rul_rmse']:.2f}  "
            f"val_RMSE={val_stats['rul_rmse']:.2f}  "
            f"theta(tr/val mean) {ts['mean']:.4f}/{vs['mean']:.4f}  "
            f"val EPR(diag)={val_stats['EPR_pred_mean_diagnostic_only']:.3f}  "
            f"({dt:.1f}s)"
        )
        history.append({"train": train_stats, "val": val_stats, "elapsed_s": dt})
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save({
                "model": model.state_dict(),
                "scalers": {k: v.tolist() for k, v in {
                    "sensor_mean": sensor_mean, "sensor_std": sensor_std,
                    "ops_mean": ops_mean, "ops_std": ops_std,
                }.items()},
                "sigma_T_K": sigma_T,
                "sigma_lpt_flow": sigma_lpt,
                "epoch": ep,
                "val_loss": best_val,
            }, out_dir / "best.pt")
    torch.save(model.state_dict(), out_dir / "last.pt")

    # Save history CSV
    import csv
    csv_path = out_dir / "train_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "tag", "loss",
            "comp_rul", "comp_temp", "comp_aux", "comp_healthy", "comp_smooth",
            "rul_rmse", "T24_mae_K", "T30_mae_K", "T50_mae_K",
            "theta_mean", "theta_std", "theta_min", "theta_max",
            "theta_frac_at_lo", "theta_frac_at_hi",
            "EPR_pred_mean_diagnostic_only",
        ])
        for entry in history:
            for tag, s in (("train", entry["train"]), ("val", entry["val"])):
                c = s["components"]
                t = s["theta_stats"]
                writer.writerow([
                    s["epoch"], tag, s["loss"],
                    c.get("rul", ""), c.get("temp", ""), c.get("aux", ""),
                    c.get("healthy", ""), c.get("smooth", ""),
                    s["rul_rmse"], s["T24_mae_K"], s["T30_mae_K"], s["T50_mae_K"],
                    t["mean"], t["std"], t["min"], t["max"],
                    t["frac_at_lo"], t["frac_at_hi"],
                    s["EPR_pred_mean_diagnostic_only"],
                ])
    print(f"\nWrote {csv_path}")

    elapsed = time.time() - t0_all
    last_train = history[-1]["train"]; last_val = history[-1]["val"]
    md = f"""# V3.1b thermal-aux smoke training summary

*Profile:* `{cfg.get('profile_name', '?')}`
*Profile scope:* see ADR-0012 (V3.1b thermal auxiliary, NOT EPR-validated).

## Run

* config: `{cfg_path}`
* device: `{device}`
* epochs: {n_epochs}
* batch size: {bs}
* train windows: {len(train_ds):,}
* val   windows: {len(val_ds):,}
* total wall time: {elapsed:.1f} s

## Final epoch

| | train | val |
|---|---|---|
| total loss | {last_train['loss']:.4f} | {last_val['loss']:.4f} |
| L_rul     | {last_train['components'].get('rul', float('nan')):.4f} | {last_val['components'].get('rul', float('nan')):.4f} |
| L_temp    | {last_train['components'].get('temp', float('nan')):.4f} | {last_val['components'].get('temp', float('nan')):.4f} |
| L_aux     | {last_train['components'].get('aux', float('nan')):.4f} | {last_val['components'].get('aux', float('nan')):.4f} |
| L_healthy | {last_train['components'].get('healthy', float('nan')):.4f} | {last_val['components'].get('healthy', float('nan')):.4f} |
| L_smooth  | {last_train['components'].get('smooth', float('nan')):.4f} | {last_val['components'].get('smooth', float('nan')):.4f} |
| RUL RMSE  | {last_train['rul_rmse']:.3f} | {last_val['rul_rmse']:.3f} |
| T24 MAE [K] | {last_train['T24_mae_K']:.2f} | {last_val['T24_mae_K']:.2f} |
| T30 MAE [K] | {last_train['T30_mae_K']:.2f} | {last_val['T30_mae_K']:.2f} |
| T50 MAE [K] | {last_train['T50_mae_K']:.2f} | {last_val['T50_mae_K']:.2f} |
| θ mean    | {last_train['theta_stats']['mean']:.4f} | {last_val['theta_stats']['mean']:.4f} |
| θ std     | {last_train['theta_stats']['std']:.4f} | {last_val['theta_stats']['std']:.4f} |
| θ frac@lo (0.85) | {last_train['theta_stats']['frac_at_lo']:.3f} | {last_val['theta_stats']['frac_at_lo']:.3f} |
| θ frac@hi (1.00) | {last_train['theta_stats']['frac_at_hi']:.3f} | {last_val['theta_stats']['frac_at_hi']:.3f} |
| EPR mean (DIAG, not in loss) | {last_train['EPR_pred_mean_diagnostic_only']:.3f} | {last_val['EPR_pred_mean_diagnostic_only']:.3f} |

## Hard constraints honored

* No EPR / pressure in loss (asserted by `CycleLayerV3Loss`).
* No supervised L_θ on θ_phys.
* Train/val units split (no random row split).
* Test units `{data_cfg.get('test_units')}` NOT used (later evaluation only).

## Artifacts

* `best.pt`           — model state at best val loss
* `last.pt`           — final model state
* `sensor_scaler.npz` — per-channel mean/std for X_s (14 cols)
* `ops_scaler.npz`    — per-channel mean/std for W   (4 cols)
* `sigma_train.json`  — σ_T24/T30/T50 (K) and σ_lpt_flow used in L_temp / L_aux
* `train_log.csv`     — per-epoch metrics

## Next step (manual)

```bash
python scripts/evaluate_cyclelayer_v3_theta_diagnostics.py \\
    --checkpoint {out_dir / 'best.pt'} \\
    --config     {cfg_path}
```
"""
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.md'}")
    print("\nDone (smoke). No automatic adoption. Stop point reached.")


if __name__ == "__main__":
    main()
