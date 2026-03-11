"""Diagnostic plots and plausibility report for predictions.csv.

Generates a small set of PNG plots and a markdown report that make it easy
to judge whether model predictions are plausible, spot mid-trajectory spikes,
and understand how the metrics relate to the visual output.

Usage
-----
    python scripts/analyze_predictions.py \\
        --pred_csv runs/my_run/predictions.csv \\
        --out_dir  results/analysis

    # Focus on a single unit:
    python scripts/analyze_predictions.py \\
        --pred_csv runs/my_run/predictions_dev.csv \\
        --out_dir  results/analysis_unit1 \\
        --unit_id  1 \\
        --n_spikes 50

Outputs (written to --out_dir)
-------------------------------
    unit_<id>_trajectory.png   -- y_true vs y_pred + abs_error vs time_index
    error_histogram.png        -- abs_error distribution (global + per unit)
    scatter_pred_vs_true.png   -- scatter with identity line
    spike_locator.csv          -- top-N windows by abs_error per unit
    report.md                  -- explanatory text + per-unit summary stats

Colab compatibility
-------------------
Uses matplotlib Agg backend (non-interactive).  No display required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force non-interactive Agg backend before any pyplot import.
# This is required for headless environments (Colab, servers, CI).
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_unit_trajectory(df_unit: pd.DataFrame, uid: int, out_dir: Path) -> None:
    """Line plot: y_true + y_pred vs time_index, then abs_error below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df_unit["time_index"], df_unit["y_true_rul"],
             label="y_true", color="steelblue", linewidth=1.5)
    ax1.plot(df_unit["time_index"], df_unit["y_pred_rul"],
             label="y_pred", color="darkorange", linewidth=1.2, alpha=0.85)
    ax1.set_ylabel("RUL (cycles)")
    ax1.set_title(f"Unit {uid}  —  RUL trajectory  "
                  f"({len(df_unit):,} windows)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(df_unit["time_index"], df_unit["abs_error"],
                     color="crimson", alpha=0.5, linewidth=0)
    ax2.plot(df_unit["time_index"], df_unit["abs_error"],
             color="crimson", linewidth=0.8)
    ax2.set_ylabel("|error|")
    ax2.set_xlabel("time_index (window index, stride_eval steps)")
    ax2.set_title("Absolute prediction error")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"unit_{uid}_trajectory.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_error_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    """Global histogram + one subplot per unit."""
    units   = sorted(df["unit_id"].unique())
    n_units = len(units)
    ncols   = n_units + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4), squeeze=False)
    axes = axes[0]

    # Global panel
    axes[0].hist(df["abs_error"], bins=60, color="steelblue",
                 alpha=0.75, edgecolor="white", linewidth=0.3)
    axes[0].set_title("All units\n(global)")
    axes[0].set_xlabel("|error|")
    axes[0].set_ylabel("windows")

    # Per-unit panels
    for i, uid in enumerate(units):
        err = df[df["unit_id"] == uid]["abs_error"]
        axes[i + 1].hist(err, bins=40, color="darkorange",
                         alpha=0.75, edgecolor="white", linewidth=0.3)
        axes[i + 1].set_title(f"Unit {uid}")
        axes[i + 1].set_xlabel("|error|")

    fig.suptitle("Absolute error distribution", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "error_histogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter y_true vs y_pred for all windows, with identity line."""
    max_rul = max(df["y_true_rul"].max(), df["y_pred_rul"].max())
    units   = sorted(df["unit_id"].unique())

    # Color-code by unit if few units, else single color
    fig, ax = plt.subplots(figsize=(6.5, 6))
    if len(units) <= 8:
        cmap = plt.cm.get_cmap("tab10", len(units))
        for i, uid in enumerate(units):
            mask = df["unit_id"] == uid
            ax.scatter(df.loc[mask, "y_true_rul"], df.loc[mask, "y_pred_rul"],
                       s=1, alpha=0.2, color=cmap(i), label=f"u{uid}", rasterized=True)
        ax.legend(markerscale=6, loc="upper left", fontsize=8)
    else:
        ax.scatter(df["y_true_rul"], df["y_pred_rul"],
                   s=1, alpha=0.1, color="steelblue", rasterized=True)

    ax.plot([0, max_rul], [0, max_rul], "r--", linewidth=1.5, label="identity")
    ax.set_xlabel("y_true_rul")
    ax.set_ylabel("y_pred_rul")
    ax.set_title(f"Predicted vs True RUL  ({len(df):,} windows)")
    ax.set_xlim(0, max_rul * 1.05)
    ax.set_ylim(0, max_rul * 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "scatter_pred_vs_true.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _spike_locator(df: pd.DataFrame, n: int, out_dir: Path) -> pd.DataFrame:
    """Find top-N windows by abs_error per unit, save as CSV.

    This is the 'spike locator': a quick way to find the exact time steps
    where the model has catastrophic failures.  Cross-reference with the
    trajectory plot to see whether spikes cluster at certain RUL values or
    flight classes.
    """
    rows = []
    for uid, grp in df.groupby("unit_id"):
        top = grp.nlargest(n, "abs_error")[
            ["unit_id", "time_index", "y_true_rul", "y_pred_rul", "abs_error"]
        ].copy()
        rows.append(top)
    spike_df = pd.concat(rows, ignore_index=True)
    spike_df.to_csv(out_dir / "spike_locator.csv", index=False)
    return spike_df


# ---------------------------------------------------------------------------
# Per-unit summary stats (used in report)
# ---------------------------------------------------------------------------

def _unit_stats(df: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    rows = []
    for uid, grp in df.groupby("unit_id"):
        grp = grp.sort_values("time_index")
        p   = grp["y_pred_rul"].values
        t   = grp["y_true_rul"].values
        err = grp["abs_error"].values
        rel = err / np.maximum(t, 1e-6)
        within = rel <= alpha
        T = len(grp)

        rows.append({
            "unit_id":         uid,
            "n_windows":       T,
            "rmse":            float(np.sqrt(np.mean(err ** 2))),
            "mae":             float(np.mean(err)),
            "max_err":         float(np.max(err)),
            "p95_err":         float(np.percentile(err, 95)),
            "frac_within_20": float(np.mean(within)),
            "frac_last50":    float(np.mean(within[-50:])) if T >= 50 else float(np.mean(within)),
            "frac_last100":   float(np.mean(within[-100:])) if T >= 100 else float(np.mean(within)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(
    df:       pd.DataFrame,
    stats:    pd.DataFrame,
    out_dir:  Path,
    alpha:    float,
) -> None:
    """Write a self-contained markdown report explaining the analysis."""
    units    = sorted(df["unit_id"].unique())
    n_samp   = len(df)
    split    = df["split"].iloc[0] if "split" in df.columns else "unknown"
    glob_mae = float(df["abs_error"].mean())
    glob_rmse= float(np.sqrt((df["abs_error"] ** 2).mean()))

    # Stats table rows
    stat_rows = ""
    for _, r in stats.iterrows():
        stat_rows += (
            f"| {int(r.unit_id):4d} | {int(r.n_windows):7,} | "
            f"{r.rmse:6.2f} | {r.mae:6.2f} | {r.max_err:7.1f} | "
            f"{r.frac_within_20:.2%} | {r.frac_last50:.2%} |\n"
        )

    report = f"""\
# Predictions Analysis Report
Generated from: `predictions.csv`  |  Split: `{split}`  |  Alpha: {alpha}

## 1. What is predictions.csv?

`predictions.csv` contains window-level RUL predictions exported by
`scripts/evaluate.py`.

| Column | Description |
|--------|-------------|
| `unit_id` | Engine unit identifier (e.g. 1-6 in DS01-005) |
| `time_index` | 0-indexed window position within that unit's trajectory |
| `y_true_rul` | Ground-truth Remaining Useful Life (cycles) |
| `y_pred_rul` | Model-predicted RUL |
| `abs_error` | `|y_pred_rul - y_true_rul|` |
| `split` | HDF5 split (`dev` or `test`) |

This file has **{n_samp:,} rows** covering {len(units)} unit(s): {units}.

---

## 2. time_index and stride_eval

`time_index` is a 0-indexed window counter that resets to 0 at the start
of each unit's trajectory.  It is **not** measured in physical cycles:

- `time_index = 0`   → first window (highest RUL, engine is fresh)
- `time_index = T-1` → last window before failure (RUL ~ 0)

Each step corresponds to `stride_eval` cycle shifts of the sensor window.
With `stride_eval=1` (dense), every step = 1 cycle.  With `stride_eval=50`,
every step = 50 cycles.

**Best practice**: always evaluate with `stride_eval=1` so you see the full
trajectory and can compute the Prediction Horizon correctly.

---

## 3. Interpreting the metrics

### RMSE — Root Mean Squared Error  ({glob_rmse:.2f} here)
- Symmetric: treats over- and under-prediction equally.
- Useful for overall accuracy; comparable across experiments.
- Limitation: averages over all time steps including high-RUL early windows
  where the model has little prognostic value.

### S-score (NASA PHM'08 asymmetric scoring)
- Formula: `S = sum(exp(d/a) - 1)` where `d = y_hat - y`,
  `a = 10` for late predictions (over-estimate), `a = 13` for early.
- Penalises over-estimation (predicting too much RUL = delayed alarm) more.

**CRITICAL WARNING** — s_score is a SUM over N_samples.  With {n_samp:,}
windows, even a perfect model would show a large absolute value.
- `s_score_sum`: raw PHM'08 figure.  Use ONLY to compare runs with the
  **same dataset and stride**.
- `s_score_mean = s_score_sum / N_samples`: per-window; comparable across
  strides and split sizes.  This is the preferred comparison metric.
- `s_score_unit_median_mean`: median of per-unit means.  Each unit
  contributes equally regardless of trajectory length.

### Prediction Horizon (PH, alpha={alpha})
- "How many cycles before failure does the model become reliably accurate?"
- PH = None means the model **never** achieves continuous accuracy until EOL.
- `frac_within_alpha_lastK` helps diagnose why:

  | frac_last50 high (>0.8), PH=None | -> mid-trajectory spike problem |
  | frac_last50 low (<0.4), PH=None  | -> systematic bias or collapse  |

---

## 4. Per-unit summary

| unit | windows | RMSE | MAE | max_err | within_20% | last_50 |
|------|---------|------|-----|---------|------------|---------|
{stat_rows}
Global: MAE={glob_mae:.2f}, RMSE={glob_rmse:.2f}

---

## 5. Typical failure modes

### Mode 1: Mid-trajectory spikes (most common)
**Symptoms**: PH=None, frac_last50 > 0.8, but spike_locator.csv shows
a few rows with very large abs_error at intermediate time_indices.
**In plots**: trajectory plot is mostly smooth but has one or more sharp
upward spikes in the abs_error panel.
**Root cause**: model fails on specific flight classes (Fc) or operating
condition transitions.
**Fix**: longer warmup, gradient clipping, examine whether spikes correlate
with Fc column from the HDF5 A_dev array.

### Mode 2: Constant prediction (underfit / collapse)
**Symptoms**: Large RMSE, scatter shows a horizontal band at a fixed y_pred
value, frac_within_20 is low everywhere.
**In plots**: y_pred is nearly flat; abs_error is proportional to y_true.
**Root cause**: model collapsed to predicting the mean or a constant.
**Fix**: reduce LR, use "delayed" lambda schedule, increase model capacity.

### Mode 3: Good late-life, bad early-life (expected, acceptable)
**Symptoms**: PH has a non-None value, RMSE moderate, frac_last50 high.
**In plots**: abs_error is large at high y_true (left side) and small
near EOL (right side of trajectory plot).
**Root cause**: high-RUL prediction is inherently harder; the model
learns end-of-life patterns better.
**This is normal**: PH captures it correctly.

---

## 6. Files in this directory

| File | Description |
|------|-------------|
| `unit_*_trajectory.png` | Per-unit RUL trajectory + abs_error panel |
| `error_histogram.png` | Global and per-unit absolute error distributions |
| `scatter_pred_vs_true.png` | y_true vs y_pred scatter with identity line |
| `spike_locator.csv` | Top-N error time steps per unit for manual inspection |
| `report.md` | This file |
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic plots and report from predictions.csv"
    )
    parser.add_argument("--pred_csv",  required=True,
                        help="Path to predictions.csv (or predictions_dev.csv etc.)")
    parser.add_argument("--out_dir",   required=True,
                        help="Output directory for plots and report.md")
    parser.add_argument("--unit_id",   type=int, default=None,
                        help="If set, only analyse this unit (trajectory plot still generated)")
    parser.add_argument("--n_spikes",  type=int, default=20,
                        help="Top-N windows by abs_error to put in spike_locator.csv (per unit)")
    parser.add_argument("--alpha",     type=float, default=0.2,
                        help="Relative error threshold for PH / frac_within_alpha (default 0.2)")
    args = parser.parse_args()

    pred_path = Path(args.pred_csv)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {pred_path} ...")
    df = pd.read_csv(pred_path)

    required_cols = {"unit_id", "time_index", "y_true_rul", "y_pred_rul", "abs_error"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: predictions.csv is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(df):,} rows, {df['unit_id'].nunique()} units: "
          f"{sorted(df['unit_id'].unique())}")

    # Optionally filter to a single unit for detailed analysis
    df_full = df.copy()
    if args.unit_id is not None:
        if args.unit_id not in df["unit_id"].values:
            print(f"ERROR: unit_id {args.unit_id} not in CSV.", file=sys.stderr)
            sys.exit(1)
        df = df[df["unit_id"] == args.unit_id].copy()
        print(f"  Filtered to unit {args.unit_id} ({len(df):,} rows)")

    units = sorted(df["unit_id"].unique())

    # --- Trajectory plots (one per unit) ---
    print(f"Plotting {len(units)} unit trajectory plot(s) ...")
    for uid in units:
        df_u = df[df["unit_id"] == uid].sort_values("time_index")
        _plot_unit_trajectory(df_u, uid, out_dir)
    print(f"  Saved to {out_dir}/unit_*_trajectory.png")

    # --- Error histogram (uses df_full if unit_id filter was applied) ---
    df_hist = df_full if args.unit_id is not None else df
    print("Plotting error histogram ...")
    _plot_error_histogram(df_hist, out_dir)
    print(f"  Saved to {out_dir}/error_histogram.png")

    # --- Scatter ---
    print("Plotting scatter ...")
    _plot_scatter(df_hist, out_dir)
    print(f"  Saved to {out_dir}/scatter_pred_vs_true.png")

    # --- Spike locator ---
    print(f"Building spike locator (top-{args.n_spikes} per unit) ...")
    spike_df = _spike_locator(df_hist, args.n_spikes, out_dir)
    print(f"  Saved to {out_dir}/spike_locator.csv  ({len(spike_df)} rows)")

    # --- Per-unit stats ---
    stats = _unit_stats(df_hist, alpha=args.alpha)
    stats.to_csv(out_dir / "unit_stats.csv", index=False)

    # --- Report ---
    print("Writing report.md ...")
    _write_report(df_hist, stats, out_dir, alpha=args.alpha)
    print(f"  Saved to {out_dir}/report.md")

    print(f"\nDone. All outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
