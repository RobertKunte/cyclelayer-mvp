"""Post-hoc θ diagnostics for the V3.1b thermal-auxiliary model.

Loads a CycleLayerV3 checkpoint from the smoke run, evaluates it on the
val + test splits (DS02), and reports:

  * theta_eta_hpt_delta vs HPT_eff_mod  (Pearson / Spearman, overall + per-unit)
  * theta_eta_lpt_delta vs LPT_eff_mod  (Pearson / Spearman, overall + per-unit)
  * lpt_flow_pred       vs LPT_flow_mod (the supervised diagnostic)
  * theta saturation rate at lower/upper bounds
  * theta vs cycle plots per unit
  * theta vs GT scatter plots
  * theta vs RUL plots
  * optional partial correlation controlling for ops conditions
    (alt, Mach, TRA, T2, Nf, Nc, Wf) via linear-regression residualisation

Hard constraints:
  * GT (HPT_eff_mod, LPT_eff_mod) is EVALUATION-ONLY for theta_phys.
  * θ_phys is NOT trained against GT (V3.1b — see ADR-0012).
  * No YAML write; no DS02 tuning.

Usage:
    python scripts/evaluate_cyclelayer_v3_theta_diagnostics.py \
        --checkpoint artifacts/cyclelayer_v3/thermal_aux_smoke/best.pt \
        --config     configs/cyclelayer_v3_thermal_aux.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import torch         # noqa: E402
import yaml          # noqa: E402
from scipy import stats as scstats   # noqa: E402

from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset                # noqa: E402
from cyclelayer.models import units                                     # noqa: E402

# Reuse helpers from the smoke script (same code path, same conventions)
sys.path.insert(0, str(Path(__file__).parent))
from train_cyclelayer_v3_thermal_aux_smoke import (   # noqa: E402
    NCMAPSSV3WindowedDataset, _collate,
    build_brayton_from_cfg, build_v3_from_cfg,
)


# =============================================================================
# Inference loop — collect per-window θ, AuxHead, GT, RUL, ops
# =============================================================================

@torch.no_grad()
def collect_predictions(
    model, loader, device,
    sensor_mean: torch.Tensor, sensor_std: torch.Tensor,
    ops_mean: torch.Tensor,    ops_std: torch.Tensor,
) -> dict:
    model.eval()
    rows = {
        "theta_eta_fan": [], "theta_eta_lpc": [], "theta_eta_hpc": [],
        "theta_eta_hpt": [], "theta_eta_lpt": [],
        "lpt_flow_pred": [],
        "HPT_eff_mod_GT": [], "LPT_eff_mod_GT": [], "LPT_flow_mod_GT": [],
        "RUL": [], "unit_id": [],
        # Ops covariates for partial correlation
        "alt_ft": [], "Mach": [], "TRA_pct": [], "T2_R": [],
        "Nf_rpm": [], "Nc_rpm": [], "Wf_pps": [],
    }
    sm, sd = sensor_mean.to(device), sensor_std.to(device)
    om, od = ops_mean.to(device),    ops_std.to(device)
    for batch in loader:
        sensors_norm = (batch["sensors_imp"].to(device) - sm) / sd
        ops_norm     = (batch["ops_imp"].to(device)     - om) / od
        ops_si  = {k: v.to(device) for k, v in batch["ops_si_last"].items()}
        sens_si = {k: v.to(device) for k, v in batch["sens_si_last"].items()}
        out = model(sensors_norm, ops_norm, ops_si=ops_si, sens_si=sens_si)
        theta = out["theta_phys"].cpu().numpy()
        rows["theta_eta_fan"].extend(theta[:, 0].tolist())
        rows["theta_eta_lpc"].extend(theta[:, 1].tolist())
        rows["theta_eta_hpc"].extend(theta[:, 2].tolist())
        rows["theta_eta_hpt"].extend(theta[:, 3].tolist())
        rows["theta_eta_lpt"].extend(theta[:, 4].tolist())
        rows["lpt_flow_pred"].extend(out["lpt_flow_pred"].cpu().numpy().tolist())
        h = batch["health_gt_last"]
        rows["HPT_eff_mod_GT"].extend(h["HPT_eff_mod"].numpy().tolist())
        rows["LPT_eff_mod_GT"].extend(h["LPT_eff_mod"].numpy().tolist())
        rows["LPT_flow_mod_GT"].extend(h["LPT_flow_mod"].numpy().tolist())
        rows["RUL"].extend(batch["RUL"].numpy().tolist())
        rows["unit_id"].extend(batch["unit_id"].numpy().tolist())
        # Last-timestep ops/sens raw (for partial correlations)
        ops_imp_last = batch["ops_imp"][:, -1, :].numpy()
        sens_imp_last = batch["sensors_imp"][:, -1, :].numpy()
        rows["alt_ft"].extend(ops_imp_last[:, 0].tolist())
        rows["Mach"].extend(ops_imp_last[:, 1].tolist())
        rows["TRA_pct"].extend(ops_imp_last[:, 2].tolist())
        rows["T2_R"].extend(ops_imp_last[:, 3].tolist())
        rows["Nf_rpm"].extend(sens_imp_last[:, 11].tolist())
        rows["Nc_rpm"].extend(sens_imp_last[:, 12].tolist())
        rows["Wf_pps"].extend(sens_imp_last[:, 13].tolist())
    return rows


# =============================================================================
# Correlation helpers
# =============================================================================

def safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson + Spearman; returns NaN if degenerate."""
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(len(x))}
    pr = float(scstats.pearsonr(x, y).statistic)
    sr = float(scstats.spearmanr(x, y).statistic)
    return {"pearson": pr, "spearman": sr, "n": int(len(x))}


def partial_corr_residualised(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict[str, float]:
    """Partial correlation of x, y controlling for z (linear residuals).

    Fit OLS x = z·a + e_x, y = z·b + e_y; return Pearson(e_x, e_y).
    """
    if len(x) < 5 or z.shape[0] != len(x):
        return {"partial_pearson": float("nan"), "n": int(len(x))}
    # Add intercept
    Z = np.hstack([z, np.ones((z.shape[0], 1))])
    # Solve via lstsq
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    ex = x - Z @ bx
    ey = y - Z @ by
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return {"partial_pearson": float("nan"), "n": int(len(x))}
    pr = float(scstats.pearsonr(ex, ey).statistic)
    return {"partial_pearson": pr, "n": int(len(x))}


# =============================================================================
# Plots
# =============================================================================

def plot_theta_vs_gt_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    pairs = [
        ("theta_eta_hpt_delta", "HPT_eff_mod_GT", "θ_η_hpt − 1 vs HPT_eff_mod"),
        ("theta_eta_lpt_delta", "LPT_eff_mod_GT", "θ_η_lpt − 1 vs LPT_eff_mod"),
        ("lpt_flow_pred",       "LPT_flow_mod_GT", "lpt_flow_pred (AuxHead) vs LPT_flow_mod"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (xcol, ycol, title) in zip(axes, pairs):
        x = df[xcol].to_numpy(); y = df[ycol].to_numpy()
        sc = ax.scatter(x, y, c=df["RUL"].to_numpy(), cmap="viridis_r",
                        s=8, alpha=0.55, edgecolors="none")
        ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(title)
        ax.grid(True, alpha=0.4)
        s = safe_corr(x, y)
        ax.text(0.02, 0.97,
                f"Pearson={s['pearson']:.3f}\nSpearman={s['spearman']:.3f}\nN={s['n']}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    fig.colorbar(sc, ax=axes[-1], label="RUL")
    fig.suptitle("V3.1b thermal-aux — θ vs N-CMAPSS GT (post-hoc evaluation only)")
    fig.tight_layout()
    p = out_dir / "01_theta_vs_GT_scatter.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  saved  {p}")


def plot_theta_vs_RUL(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    cols = ["theta_eta_fan", "theta_eta_lpc", "theta_eta_hpc",
            "theta_eta_hpt", "theta_eta_lpt", "lpt_flow_pred"]
    for ax, c in zip(axes.flat, cols):
        ax.scatter(df["RUL"], df[c], s=6, alpha=0.4, c=df["unit_id"],
                   cmap="tab10")
        ax.set_xlabel("RUL"); ax.set_ylabel(c); ax.grid(True, alpha=0.4)
        ax.set_title(c)
    fig.suptitle("V3.1b thermal-aux — θ and AuxHead vs RUL (per-unit colours)")
    fig.tight_layout()
    p = out_dir / "02_theta_vs_RUL.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  saved  {p}")


def plot_theta_per_unit(df: pd.DataFrame, out_dir: Path) -> None:
    """For each unit, plot θ_eta_hpt/lpt vs window order (proxy for cycle)."""
    units_sorted = sorted(df["unit_id"].unique())
    n = len(units_sorted)
    if n == 0:
        return
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.5 * rows), sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, uid in zip(axes_flat, units_sorted):
        sub = df[df["unit_id"] == uid].sort_values("RUL", ascending=False).reset_index(drop=True)
        ax.plot(sub.index, sub["theta_eta_hpt"], "o-",
                color="tab:red",   ms=2, lw=0.7, label="θ_η_hpt")
        ax.plot(sub.index, sub["theta_eta_lpt"], "o-",
                color="tab:blue",  ms=2, lw=0.7, label="θ_η_lpt")
        ax2 = ax.twinx()
        ax2.plot(sub.index, sub["HPT_eff_mod_GT"] + 1.0, "--",
                 color="tab:red",  lw=0.7, alpha=0.6, label="HPT_eff_mod + 1")
        ax2.plot(sub.index, sub["LPT_eff_mod_GT"] + 1.0, "--",
                 color="tab:blue", lw=0.7, alpha=0.6, label="LPT_eff_mod + 1")
        ax.set_title(f"unit {uid}")
        ax.set_xlabel("window (decreasing-RUL order)"); ax.set_ylabel("θ (factor)")
        ax.set_ylim(0.84, 1.01)
        ax.grid(True, alpha=0.3)
        ax2.set_ylim(0.84, 1.01)
        ax2.set_yticks([])
        ax.legend(fontsize=6, loc="lower left")
        ax2.legend(fontsize=6, loc="lower right")
    for ax in axes_flat[len(units_sorted):]:
        ax.axis("off")
    fig.suptitle("V3.1b thermal-aux — θ trajectories vs N-CMAPSS GT (per unit)")
    fig.tight_layout()
    p = out_dir / "03_theta_per_unit.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  saved  {p}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cyclelayer_v3_thermal_aux.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples_per_unit", type=int, default=None,
                        help="Cap windows per unit during evaluation (smoke).")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Load data
    base = NCMAPSSV3Dataset(Path(data_cfg["hdf5_path"]),
                            split="dev", load_in_memory=True)
    print(f"loaded DS02 dev: {len(base):,} rows, units {base.unit_ids.tolist()}")

    val_units  = list(data_cfg["val_units"])
    test_units = list(data_cfg["test_units"])

    # Build val + test windowed datasets
    val_ds = NCMAPSSV3WindowedDataset(
        base, val_units,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride_eval"],
        max_samples=args.max_samples_per_unit,
    )
    # For test, also load from test split HDF5 keys
    base_test = NCMAPSSV3Dataset(Path(data_cfg["hdf5_path"]),
                                 split="test", load_in_memory=True)
    test_ds = NCMAPSSV3WindowedDataset(
        base_test, test_units,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride_eval"],
        max_samples=args.max_samples_per_unit,
    )
    bs = int(args.batch_size or train_cfg["batch_size"])
    val_loader  = torch.utils.data.DataLoader(val_ds,  batch_size=bs,
                                              shuffle=False, num_workers=0, collate_fn=_collate)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs,
                                              shuffle=False, num_workers=0, collate_fn=_collate)
    print(f"  val windows: {len(val_ds):,}, test windows: {len(test_ds):,}")

    # Build model and load checkpoint
    brayton = build_brayton_from_cfg(model_cfg["brayton_engine"])
    model = build_v3_from_cfg(model_cfg, brayton).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"loaded checkpoint: {args.checkpoint}")

    # Scalers
    if "scalers" in ckpt:
        sm = torch.tensor(ckpt["scalers"]["sensor_mean"]).float()
        sd = torch.tensor(ckpt["scalers"]["sensor_std"]).float()
        om = torch.tensor(ckpt["scalers"]["ops_mean"]).float()
        od = torch.tensor(ckpt["scalers"]["ops_std"]).float()
    else:
        # Fall back to YAML output dir
        sc_dir = Path(train_cfg["output_dir"])
        sn = np.load(sc_dir / "sensor_scaler.npz"); on = np.load(sc_dir / "ops_scaler.npz")
        sm, sd = torch.from_numpy(sn["mean"]).float(), torch.from_numpy(sn["std"]).float()
        om, od = torch.from_numpy(on["mean"]).float(), torch.from_numpy(on["std"]).float()

    out_dir = Path(cfg["evaluation"]["theta_diag_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect ─────────────────────────────────────────────────────────
    print("\n[val] collecting predictions...")
    val_rows  = collect_predictions(model, val_loader,  device, sm, sd, om, od)
    print("[test] collecting predictions...")
    test_rows = collect_predictions(model, test_loader, device, sm, sd, om, od)
    df_val  = pd.DataFrame(val_rows);  df_val["split"]  = "val"
    df_test = pd.DataFrame(test_rows); df_test["split"] = "test"
    df = pd.concat([df_val, df_test], ignore_index=True)
    df["theta_eta_hpt_delta"] = df["theta_eta_hpt"] - 1.0
    df["theta_eta_lpt_delta"] = df["theta_eta_lpt"] - 1.0
    df.to_csv(out_dir / "theta_predictions.csv", index=False)
    print(f"  saved  {out_dir / 'theta_predictions.csv'}  (rows={len(df)})")

    # ── Correlations ────────────────────────────────────────────────────
    def per_unit_corr(d, xcol, ycol):
        rows = []
        for uid in sorted(d["unit_id"].unique()):
            sub = d[d["unit_id"] == uid]
            r = safe_corr(sub[xcol].to_numpy(), sub[ycol].to_numpy())
            r["unit_id"] = int(uid); r["n"] = len(sub)
            rows.append(r)
        return pd.DataFrame(rows)

    results = {}
    for split_name, ds in (("val", df_val), ("test", df_test), ("all", df)):
        ds = ds.copy()
        ds["theta_eta_hpt_delta"] = ds["theta_eta_hpt"] - 1.0
        ds["theta_eta_lpt_delta"] = ds["theta_eta_lpt"] - 1.0
        ov = {
            "HPT_eff_mod":  safe_corr(ds["theta_eta_hpt_delta"].to_numpy(),
                                       ds["HPT_eff_mod_GT"].to_numpy()),
            "LPT_eff_mod":  safe_corr(ds["theta_eta_lpt_delta"].to_numpy(),
                                       ds["LPT_eff_mod_GT"].to_numpy()),
            "LPT_flow_mod": safe_corr(ds["lpt_flow_pred"].to_numpy(),
                                       ds["LPT_flow_mod_GT"].to_numpy()),
        }
        # Partial correlations controlling for ops covariates
        cov_cols = ["alt_ft", "Mach", "TRA_pct", "T2_R", "Nf_rpm", "Nc_rpm", "Wf_pps"]
        z = ds[cov_cols].to_numpy()
        pc = {
            "HPT_eff_mod_partial":  partial_corr_residualised(
                ds["theta_eta_hpt_delta"].to_numpy(),
                ds["HPT_eff_mod_GT"].to_numpy(), z),
            "LPT_eff_mod_partial":  partial_corr_residualised(
                ds["theta_eta_lpt_delta"].to_numpy(),
                ds["LPT_eff_mod_GT"].to_numpy(), z),
        }
        # Saturation rates
        theta_cols = ["theta_eta_fan", "theta_eta_lpc", "theta_eta_hpc",
                      "theta_eta_hpt", "theta_eta_lpt"]
        sat_lo = float((ds[theta_cols].to_numpy() <= 0.851).mean())
        sat_hi = float((ds[theta_cols].to_numpy() >= 0.999).mean())
        # Per-unit
        per_unit = {
            "HPT_eff_mod":  per_unit_corr(ds, "theta_eta_hpt_delta", "HPT_eff_mod_GT").to_dict("records"),
            "LPT_eff_mod":  per_unit_corr(ds, "theta_eta_lpt_delta", "LPT_eff_mod_GT").to_dict("records"),
            "LPT_flow_mod": per_unit_corr(ds, "lpt_flow_pred",       "LPT_flow_mod_GT").to_dict("records"),
        }
        results[split_name] = {
            "n":            len(ds),
            "overall":      ov,
            "partial":      pc,
            "per_unit":     per_unit,
            "theta_saturation_lo": sat_lo,
            "theta_saturation_hi": sat_hi,
            "theta_summary": {c: {
                "mean": float(ds[c].mean()),
                "std":  float(ds[c].std()),
                "min":  float(ds[c].min()),
                "max":  float(ds[c].max()),
            } for c in theta_cols},
        }
    (out_dir / "theta_correlations.json").write_text(
        json.dumps(results, indent=2, default=float)
    )
    print(f"  saved  {out_dir / 'theta_correlations.json'}")

    # ── Plots ───────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_theta_vs_gt_scatter(df, out_dir)
    plot_theta_vs_RUL(df, out_dir)
    plot_theta_per_unit(df, out_dir)

    # ── Markdown summary ────────────────────────────────────────────────
    test_ov = results["test"]["overall"]
    val_ov  = results["val"]["overall"]
    md = f"""# V3.1b thermal-aux θ diagnostics

*Post-hoc evaluation only — θ_phys was NOT trained against GT.*
See [ADR-0012](../../decisions/ADR-0012-v3-thermal-auxiliary-scope.md).

## Sample counts

| split | windows |
|---|---|
| val  | {results['val']['n']:,} |
| test | {results['test']['n']:,} |

## Correlations (post-hoc, val split)

| metric | Pearson | Spearman | N |
|---|---|---|---|
| `(θ_η_hpt − 1)` vs `HPT_eff_mod` | {val_ov['HPT_eff_mod']['pearson']:.3f} | {val_ov['HPT_eff_mod']['spearman']:.3f} | {val_ov['HPT_eff_mod']['n']} |
| `(θ_η_lpt − 1)` vs `LPT_eff_mod` | {val_ov['LPT_eff_mod']['pearson']:.3f} | {val_ov['LPT_eff_mod']['spearman']:.3f} | {val_ov['LPT_eff_mod']['n']} |
| `lpt_flow_pred` vs `LPT_flow_mod` (supervised) | {val_ov['LPT_flow_mod']['pearson']:.3f} | {val_ov['LPT_flow_mod']['spearman']:.3f} | {val_ov['LPT_flow_mod']['n']} |

## Correlations (post-hoc, test split)

| metric | Pearson | Spearman | N |
|---|---|---|---|
| `(θ_η_hpt − 1)` vs `HPT_eff_mod` | {test_ov['HPT_eff_mod']['pearson']:.3f} | {test_ov['HPT_eff_mod']['spearman']:.3f} | {test_ov['HPT_eff_mod']['n']} |
| `(θ_η_lpt − 1)` vs `LPT_eff_mod` | {test_ov['LPT_eff_mod']['pearson']:.3f} | {test_ov['LPT_eff_mod']['spearman']:.3f} | {test_ov['LPT_eff_mod']['n']} |
| `lpt_flow_pred` vs `LPT_flow_mod` (supervised) | {test_ov['LPT_flow_mod']['pearson']:.3f} | {test_ov['LPT_flow_mod']['spearman']:.3f} | {test_ov['LPT_flow_mod']['n']} |

## Partial correlations (controlling for alt, Mach, TRA, T2, Nf, Nc, Wf — test)

| metric | partial Pearson | N |
|---|---|---|
| `(θ_η_hpt − 1)` vs `HPT_eff_mod` | {results['test']['partial']['HPT_eff_mod_partial']['partial_pearson']:.3f} | {results['test']['partial']['HPT_eff_mod_partial']['n']} |
| `(θ_η_lpt − 1)` vs `LPT_eff_mod` | {results['test']['partial']['LPT_eff_mod_partial']['partial_pearson']:.3f} | {results['test']['partial']['LPT_eff_mod_partial']['n']} |

## θ saturation (fraction at bound) — test split

* near lower bound 0.85: {results['test']['theta_saturation_lo']:.3f}
* near upper bound 1.00: {results['test']['theta_saturation_hi']:.3f}

## Plots

* `01_theta_vs_GT_scatter.png` — θ-delta and AuxHead vs N-CMAPSS GT
* `02_theta_vs_RUL.png` — all five θ + AuxHead vs RUL, coloured by unit
* `03_theta_per_unit.png` — per-unit θ trajectories vs GT trend
"""
    (out_dir / "theta_diagnostics_summary.md").write_text(md, encoding="utf-8")
    print(f"  saved  {out_dir / 'theta_diagnostics_summary.md'}")
    print("\nDone. No YAML written. No DS02 tuning.")


if __name__ == "__main__":
    main()
