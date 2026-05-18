"""Diagnostic — RUL mean-collapse detection (ADR-0014, Steps 2 + 3).

Read-only.  No model changes.  No DS02 tuning.

Loads the latest C run (or `--run_dir`), runs inference on the test
units, computes:
  * full metrics + per-RUL-region metrics
  * trivial baselines (constant 50, train-mean, train-median, per-unit
    linear regression on cycle index)
  * collapse-indicator flags

Generates plots:
  1. pred-vs-true scatter
  2. residual-vs-true
  3. predicted RUL histogram overlaid with true RUL histogram
  4. binned calibration plot
  5. per-unit RUL trajectory (true + predicted vs cycle)
  6. train/val/test target distributions

Outputs land under
  artifacts/cyclelayer_v3/rul_model_sanity/<TIMESTAMP>/rul_collapse/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import torch                     # noqa: E402
import yaml                      # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# Helpers + smoke-script utilities
sys.path.insert(0, str(Path(__file__).parent))
from _v31b_diag_helpers import (   # noqa: E402
    df_to_md, find_c_run_dir, flag_collapse, get_session_dir,
    metrics_by_region, rul_metrics, RUL_REGIONS, REPO_ROOT,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset   # noqa: E402

from train_cyclelayer_v3_thermal_aux_smoke import (   # noqa: E402
    NCMAPSSV3WindowedDataset, _collate,
    build_brayton_from_cfg, build_v3_from_cfg,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None,
                    help="Path to a run dir with best.pt (auto-discovered otherwise).")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_test_samples", type=int, default=None,
                    help="Cap test windows (for fast Colab smoke).")
    ap.add_argument("--max_train_samples_for_baseline", type=int, default=200000)
    return ap.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device, scalers):
    sm, sd, om, od = scalers
    sm = sm.to(device); sd = sd.to(device)
    om = om.to(device); od = od.to(device)
    rul_pred, rul_true, unit, cyc = [], [], [], []
    for b in loader:
        sn = (b["sensors_imp"].to(device) - sm) / sd
        on = (b["ops_imp"].to(device)     - om) / od
        ops_si  = {k: v.to(device) for k, v in b["ops_si_last"].items()}
        sens_si = {k: v.to(device) for k, v in b["sens_si_last"].items()}
        out = model(sn, on, ops_si=ops_si, sens_si=sens_si)
        rul_pred.append(out["rul"].cpu().numpy())
        rul_true.append(b["RUL"].numpy())
        unit.append(b["unit_id"].numpy())
        if "aux" in b and "cycle" in b["aux"]:
            cyc.append(b["aux"]["cycle"].numpy())
        else:
            cyc.append(np.zeros(len(b["RUL"]), dtype=np.int64))
    return (np.concatenate(rul_pred), np.concatenate(rul_true),
            np.concatenate(unit), np.concatenate(cyc))


def baseline_predictions(true: np.ndarray, train_mean: float, train_median: float,
                         per_unit_linear: dict | None = None,
                         unit_ids: np.ndarray | None = None,
                         cycles: np.ndarray | None = None) -> dict:
    """Return dict of baseline_name -> predicted RUL array (same len as true)."""
    n = len(true)
    out = {
        "constant_50":           np.full(n, 50.0),
        "constant_train_mean":   np.full(n, train_mean),
        "constant_train_median": np.full(n, train_median),
        "constant_test_mean":    np.full(n, float(true.mean())),  # diagnostic only
    }
    if per_unit_linear is not None and unit_ids is not None and cycles is not None:
        # Linear regression of true_RUL vs cycle per unit, using TRAIN data.
        # For test units (not in dict), we fall back to global slope/intercept.
        pred = np.zeros(n)
        for i in range(n):
            uid = int(unit_ids[i])
            cyc = float(cycles[i])
            ab = per_unit_linear.get(uid, per_unit_linear.get("__global__"))
            if ab is None:
                pred[i] = train_mean
            else:
                a, b = ab
                pred[i] = max(0.0, min(99.0, a + b * cyc))
        out["linear_cycle"] = pred
    return out


def fit_per_unit_linear(base: NCMAPSSV3Dataset, units: list[int]) -> dict:
    """Fit `true_RUL ~ a + b * cycle` per unit on the training data.
    Also fit a global slope/intercept as fallback."""
    A = base._A; unit_arr = A[:, 0].astype(np.int64)
    Y = base._Y.astype(np.float32)
    out = {}
    all_x, all_y = [], []
    for uid in units:
        mask = unit_arr == uid
        if mask.sum() < 10:
            continue
        x = A[mask, 1].astype(np.float64)   # cycle col
        y = Y[mask].astype(np.float64)
        if np.std(x) < 1e-6:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        out[int(uid)] = (float(intercept), float(slope))
        all_x.append(x); all_y.append(y)
    if all_x:
        gx = np.concatenate(all_x); gy = np.concatenate(all_y)
        slope, intercept = np.polyfit(gx, gy, 1)
        out["__global__"] = (float(intercept), float(slope))
    return out


def fit_train_stats(base: NCMAPSSV3Dataset, units: list[int]) -> tuple[float, float, np.ndarray]:
    A = base._A; unit_arr = A[:, 0].astype(np.int64)
    mask = np.isin(unit_arr, units)
    Y = base._Y[mask].astype(np.float64)
    return float(Y.mean()), float(np.median(Y)), Y


# ── Plot helpers ─────────────────────────────────────────────────────

def plot_scatter(true, pred, m, label, out_path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(true, pred, s=4, alpha=0.30, c="tab:blue", edgecolors="none")
    ax.plot([0, 99], [0, 99], "k--", lw=0.7, label="ideal y=x")
    ax.axhline(pred.mean(), color="tab:red", lw=0.7, ls=":",
               label=f"mean pred ≈ {pred.mean():.1f}")
    ax.set_xlabel("true RUL"); ax.set_ylabel("predicted RUL")
    ax.set_title(f"{label}\nR²={m['R2']:.3f}  RMSE={m['RMSE']:.2f}  "
                 f"std_ratio={m['std_ratio']:.3f}  slope={m['slope']:.3f}")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_residual(true, pred, label, out_path):
    err = pred - true
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(true, err, s=4, alpha=0.30, edgecolors="none")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvspan(0, 20, alpha=0.10, color="red",  label="EOL RUL<20 (over-estimation matters)")
    ax.axvspan(20, 30, alpha=0.10, color="orange")
    ax.set_xlabel("true RUL"); ax.set_ylabel("residual = pred − true")
    ax.set_title(f"{label} — residuals  (mean={err.mean():+.2f}, std={err.std():.2f})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(true, pred, label, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 100, 41)
    ax.hist(true, bins=bins, alpha=0.55, label=f"true (std={true.std():.1f})",
            color="tab:gray", edgecolor="black")
    ax.hist(pred, bins=bins, alpha=0.55, label=f"pred (std={pred.std():.1f})",
            color="tab:blue", edgecolor="black")
    ax.set_xlabel("RUL"); ax.set_ylabel("count")
    ax.set_title(f"{label} — predicted vs true RUL distribution")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(true, pred, label, out_path):
    bins = np.linspace(true.min(), true.max(), 11)
    centers, mean_p, mean_t, std_p, counts = [], [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (true >= lo) & (true < hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        mean_p.append(pred[mask].mean())
        mean_t.append(true[mask].mean())
        std_p.append(pred[mask].std())
        counts.append(int(mask.sum()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(mean_t, mean_p, yerr=std_p, fmt="o-",
                     color="tab:blue", capsize=3)
    axes[0].plot([0, 99], [0, 99], "k--", lw=0.6, label="ideal")
    axes[0].set_xlabel("bin mean true RUL"); axes[0].set_ylabel("bin mean pred RUL ± std")
    axes[0].set_title(f"{label} — calibration"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(centers, counts, width=(bins[1] - bins[0]) * 0.9, edgecolor="black",
                color="tab:gray", alpha=0.7)
    axes[1].set_xlabel("RUL bin centre"); axes[1].set_ylabel("count")
    axes[1].set_title("Test bin counts"); axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_per_unit_trajectory(true, pred, units, cycles, label, out_path):
    uniq = sorted(np.unique(units).tolist())
    n = len(uniq)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.5 * rows), sharex=False)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, uid in zip(axes_flat, uniq):
        mask = units == uid
        order = np.argsort(cycles[mask])
        x = cycles[mask][order] if cycles is not None else np.arange(mask.sum())
        ax.plot(x, true[mask][order], "k-", lw=1.0, label="true RUL")
        ax.plot(x, pred[mask][order], "tab:blue", lw=0.8, alpha=0.8, label="pred RUL")
        ax.set_title(f"unit {int(uid)}  (n={int(mask.sum())})")
        ax.set_xlabel("cycle"); ax.set_ylabel("RUL")
        ax.set_ylim(-2, 102); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{label} — per-unit RUL trajectory (true vs predicted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(train_y, val_y, test_y, out_path):
    bins = np.linspace(0, 100, 41)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(train_y, bins=bins, alpha=0.55, label=f"train (n={len(train_y):,}, mean={train_y.mean():.1f})",
            color="tab:gray", edgecolor="black", density=True)
    if val_y is not None and len(val_y):
        ax.hist(val_y, bins=bins, alpha=0.55,
                label=f"val (n={len(val_y):,}, mean={val_y.mean():.1f})",
                color="tab:green", edgecolor="black", density=True)
    ax.hist(test_y, bins=bins, alpha=0.55,
            label=f"test (n={len(test_y):,}, mean={test_y.mean():.1f})",
            color="tab:red", edgecolor="black", density=True)
    ax.set_xlabel("RUL"); ax.set_ylabel("density")
    ax.set_title("RUL target distribution: train vs val vs test")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    session_dir = get_session_dir()
    out_dir = session_dir / "rul_collapse"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"session dir: {session_dir}")
    print(f"out dir:     {out_dir}")

    run_dir = find_c_run_dir(args.run_dir)
    if run_dir is None:
        msg = "No C run dir found.  Pass --run_dir."
        print(msg)
        (out_dir / "report.md").write_text(f"# RUL collapse (SKIPPED)\n\n{msg}\n",
                                            encoding="utf-8")
        return
    print(f"run dir:     {run_dir}")

    # Load config from the checkpoint if available, else from the YAML at REPO_ROOT
    cfg_path = REPO_ROOT / "configs" / "cyclelayer_v3_thermal_aux.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    train_units = list(data_cfg["train_units"])
    val_units   = list(data_cfg["val_units"])
    test_units  = list(data_cfg["test_units"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brayton = build_brayton_from_cfg(cfg["model"]["brayton_engine"])
    model = build_v3_from_cfg(cfg["model"], brayton).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    if "scalers" in ckpt:
        sm = torch.tensor(ckpt["scalers"]["sensor_mean"]).float()
        sd = torch.tensor(ckpt["scalers"]["sensor_std"]).float()
        om = torch.tensor(ckpt["scalers"]["ops_mean"]).float()
        od = torch.tensor(ckpt["scalers"]["ops_std"]).float()
    else:
        sn = np.load(run_dir / "sensor_scaler.npz"); on = np.load(run_dir / "ops_scaler.npz")
        sm = torch.from_numpy(sn["mean"]).float(); sd = torch.from_numpy(sn["std"]).float()
        om = torch.from_numpy(on["mean"]).float(); od = torch.from_numpy(on["std"]).float()

    # Load DS02
    ds02_path = Path(data_cfg["hdf5_path"])
    if not ds02_path.is_absolute():
        ds02_path = REPO_ROOT / ds02_path
    if not ds02_path.exists():
        msg = f"DS02 not at {ds02_path}"
        print(msg)
        (out_dir / "report.md").write_text(f"# RUL collapse (SKIPPED)\n\n{msg}\n",
                                            encoding="utf-8")
        return

    base_dev  = NCMAPSSV3Dataset(ds02_path, split="dev",  load_in_memory=True)
    base_test = NCMAPSSV3Dataset(ds02_path, split="test", load_in_memory=True)
    test_ds = NCMAPSSV3WindowedDataset(
        base_test, test_units,
        window_size=data_cfg["window_size"], stride=data_cfg["stride_eval"],
        max_samples=args.max_test_samples,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=_collate)
    print(f"test windows: {len(test_ds):,}")

    # ── Predictions ──────────────────────────────────────────────────
    pred, true, units, cycles = collect_predictions(
        model, test_loader, device, (sm, sd, om, od),
    )
    m_full = rul_metrics(pred, true)
    region_df = metrics_by_region(pred, true, RUL_REGIONS)
    eol_bias = float(region_df.loc[region_df["region"] == "RUL<20", "bias"].iloc[0])
    flags = flag_collapse(m_full, eol_bias)
    print(f"V3.1b C  RMSE={m_full['RMSE']:.3f}  R²={m_full['R2']:.3f}  "
          f"std_ratio={m_full['std_ratio']:.3f}  slope={m_full['slope']:.3f}")

    # ── Trivial baselines ────────────────────────────────────────────
    train_mean, train_med, train_y_all = fit_train_stats(base_dev, train_units)
    val_y = fit_train_stats(base_dev, val_units)[2] if val_units else np.array([])
    per_unit_linear = fit_per_unit_linear(base_dev, train_units)
    baselines = baseline_predictions(
        true, train_mean, train_med, per_unit_linear, units, cycles
    )

    all_rows: list[dict] = []
    metric_keys = ["RMSE", "MAE", "bias", "R2", "Pearson", "Spearman", "slope",
                   "std_true", "std_pred", "std_ratio",
                   "pred_min", "pred_max", "pred_p05", "pred_p50", "pred_p95",
                   "true_min", "true_max", "true_p05", "true_p50", "true_p95",
                   "n"]
    all_rows.append({"model": "V3.1b_C", **{k: m_full[k] for k in metric_keys}})
    for name, p in baselines.items():
        m = rul_metrics(p, true)
        all_rows.append({"model": name, **{k: m[k] for k in metric_keys}})
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(out_dir / "metrics_overall.csv", index=False)
    region_df.to_csv(out_dir / "metrics_by_region.csv", index=False)

    # ── Plots for V3.1b C ────────────────────────────────────────────
    label_C = f"V3.1b C ({run_dir.name})"
    plot_scatter(true, pred, m_full, label_C, out_dir / "01_scatter_pred_vs_true_C.png")
    plot_residual(true, pred, label_C, out_dir / "02_residual_vs_true_C.png")
    plot_histograms(true, pred, label_C, out_dir / "03_hist_pred_vs_true_C.png")
    plot_calibration(true, pred, label_C, out_dir / "04_calibration_C.png")
    plot_per_unit_trajectory(true, pred, units, cycles, label_C,
                              out_dir / "05_per_unit_trajectory_C.png")

    # Best constant baseline scatter for comparison
    best_const = max(["constant_50", "constant_train_mean",
                      "constant_train_median", "constant_test_mean"],
                     key=lambda k: rul_metrics(baselines[k], true)["R2"])
    p_b = baselines[best_const]; m_b = rul_metrics(p_b, true)
    plot_scatter(true, p_b, m_b, f"baseline {best_const}",
                 out_dir / "01b_scatter_baseline.png")

    # linear cycle baseline if present
    if "linear_cycle" in baselines:
        p_l = baselines["linear_cycle"]; m_l = rul_metrics(p_l, true)
        plot_scatter(true, p_l, m_l, "baseline linear-cycle-per-unit",
                     out_dir / "01c_scatter_linear_cycle.png")
        plot_per_unit_trajectory(true, p_l, units, cycles,
                                  "baseline linear-cycle-per-unit",
                                  out_dir / "05b_per_unit_trajectory_linear.png")

    # Target distribution
    plot_target_distribution(train_y_all, val_y, true,
                              out_dir / "06_target_distribution_train_val_test.png")

    # ── Markdown report ──────────────────────────────────────────────
    md = f"""# RUL collapse diagnostic — V3.1b C (ADR-0014)

*Read-only.  No model changes.*

* Checkpoint: `{run_dir}`
* Test windows: {len(test_ds):,}
* Test units:   {sorted(np.unique(units).tolist())}

## Overall metrics

{df_to_md(metrics_df.round(4), floatfmt='.4f')}

## V3.1b C — collapse flags

| indicator | value | threshold | flagged? |
|---|---|---|---|
| R² ≤ 0           | {m_full['R2']:.4f}     | ≤ 0    | {'YES' if flags['R2_le_0'] else 'no'} |
| std_pred/std_true < 0.3 | {m_full['std_ratio']:.4f} | < 0.3 | {'YES' if flags['std_ratio_lt_0.3'] else 'no'} |
| abs(slope) < 0.3 | {m_full['slope']:.4f}  | < 0.3  | {'YES' if flags['abs_slope_lt_0.3'] else 'no'} |
| EOL bias (RUL<20) > +10 | {eol_bias:.4f}    | > +10  | {'YES' if flags['EOL_bias_gt_+10'] else 'no'} |

## Per-region metrics (V3.1b C)

{df_to_md(region_df, floatfmt='.3f')}

## Plots

* `01_scatter_pred_vs_true_C.png`        — V3.1b C scatter
* `01b_scatter_baseline.png`             — best constant baseline scatter
* `01c_scatter_linear_cycle.png`         — per-unit linear-cycle baseline (if available)
* `02_residual_vs_true_C.png`            — V3.1b C residuals
* `03_hist_pred_vs_true_C.png`           — predicted vs true RUL histograms
* `04_calibration_C.png`                 — binned calibration
* `05_per_unit_trajectory_C.png`         — per-unit RUL trajectory
* `05b_per_unit_trajectory_linear.png`   — baseline linear-cycle trajectory
* `06_target_distribution_train_val_test.png` — RUL distributions across splits

## Interpretation

* `std_ratio = {m_full['std_ratio']:.3f}` — V3.1b C predicts a band of width
  ≈ {m_full['std_pred']:.1f} cycles versus a true range of ≈ {m_full['std_true']:.1f}.
* `R² = {m_full['R2']:.3f}` — fraction of variance explained vs the mean predictor.
  A non-positive R² means the model is **no better than predicting the test mean**.
* `slope = {m_full['slope']:.3f}` — slope of `pred = a + b·true`.  Ideal model: 1.0.
* `EOL bias = {eol_bias:+.2f}` cycles on `RUL<20` — positive means over-estimation
  at end-of-life.

Decision rule (ADR-0014):

* if `R² ≤ 0` **and** `std_ratio < 0.3` **and** `|slope| < 0.3` → **MEAN-COLLAPSE confirmed**
* else if some flags fire but not all → **WEAK / partial-collapse**
* else → **NO collapse**

V3.1b C flags fired: **{sum(flags.values())} / 4** ({', '.join(k for k, v in flags.items() if v) or 'none'}).
"""
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({
        "run_dir":    str(run_dir),
        "n_test":     int(len(true)),
        "metrics":    m_full,
        "eol_bias":   eol_bias,
        "flags":      flags,
        "best_const_baseline_R2": float(m_b["R2"]),
    }, indent=2, default=float), encoding="utf-8")
    print(f"saved {out_dir / 'report.md'}")
    print(f"saved {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
