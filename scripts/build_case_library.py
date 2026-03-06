"""Build a failure-case library from evaluate.py's predictions.csv output.

Two complementary views of "failure cases":

1. Unit-level EOL manifest (unit_manifest.csv)
   Criterion: absolute RUL error at the End-Of-Life window.
   EOL window = the window with the minimum y_true_rul for a given unit.
   Ranks units as worst / best / mid.

2. Window-level top-N manifests (worst_windows.csv, best_windows.csv, mid_windows.csv)
   Criterion: absolute RUL error (abs_error column) across ALL windows.
   Worst = N windows with largest error.
   Best  = N windows with smallest error.
   Mid   = N windows whose abs_error is closest to the global median.

Outputs (written to --out_dir):
    unit_manifest.csv     – per-unit EOL error + rank tag
    worst_windows.csv     – top N windows with largest abs error
    best_windows.csv      – top N windows with smallest abs error
    mid_windows.csv       – N windows closest to median abs error
    summary.json          – aggregate stats used to produce the library

Usage
-----
    python scripts/build_case_library.py \\
        --predictions runs/<run_dir>/predictions.csv \\
        --out_dir results/ds01/case_library \\
        --n 20

    # From a LOUO run (per-fold predictions are in each fold_<unit>/ subdir):
    python scripts/build_case_library.py \\
        --predictions "runs/<louo_dir>/louo_cyclelayer_v1/fold_*/predictions.csv" \\
        --out_dir results/ds01/case_library_louo \\
        --n 20
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_predictions(predictions_arg: str) -> pd.DataFrame:
    """Load one or more predictions.csv files (glob supported)."""
    paths = sorted(glob.glob(predictions_arg))
    if not paths:
        raise FileNotFoundError(f"No files matched: {predictions_arg!r}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_file"] = str(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _unit_eol_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Per-unit EOL error: abs error at the window with minimum y_true_rul."""
    records = []
    for uid, grp in df.groupby("unit_id", sort=True):
        eol_row = grp.loc[grp["y_true_rul"].idxmin()]
        records.append({
            "unit_id":      uid,
            "eol_y_true":   float(eol_row["y_true_rul"]),
            "eol_y_pred":   float(eol_row["y_pred_rul"]),
            "eol_abs_error": float(eol_row["abs_error"]),
            "n_windows":    len(grp),
            "mean_abs_error": float(grp["abs_error"].mean()),
            "source_file":  eol_row.get("source_file", ""),
        })
    manifest = pd.DataFrame(records).sort_values("eol_abs_error", ascending=False)
    manifest = manifest.reset_index(drop=True)

    n_units = len(manifest)
    n_tag   = max(1, n_units // 3)   # tag roughly top/bottom third

    manifest["rank_tag"] = "mid"
    manifest.loc[manifest.index[:n_tag], "rank_tag"] = "worst"
    manifest.loc[manifest.index[-n_tag:], "rank_tag"] = "best"
    return manifest


def _window_manifests(df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (worst_n, best_n, mid_n) DataFrames by abs_error."""
    sorted_df  = df.sort_values("abs_error", ascending=False).reset_index(drop=True)
    worst_n    = sorted_df.head(n).copy()
    best_n     = df.sort_values("abs_error", ascending=True).head(n).copy()

    median_err = df["abs_error"].median()
    mid_n = (
        df.assign(_dist=(df["abs_error"] - median_err).abs())
        .sort_values("_dist")
        .head(n)
        .drop(columns=["_dist"])
        .copy()
    )
    return worst_n, best_n, mid_n


def main() -> None:
    parser = argparse.ArgumentParser(description="Build failure-case library from predictions.csv.")
    parser.add_argument(
        "--predictions", required=True,
        help="Path (or glob) to predictions.csv file(s).",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Output directory for library artifacts.",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of cases per category (worst/best/mid windows). Default: 20.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions: {args.predictions}")
    df = _load_predictions(args.predictions)
    print(f"  Loaded {len(df):,} windows from {df['unit_id'].nunique()} units.")

    # ── Unit EOL manifest ────────────────────────────────────────────────────
    unit_manifest = _unit_eol_manifest(df)
    unit_path = out_dir / "unit_manifest.csv"
    unit_manifest.to_csv(unit_path, index=False)
    print(f"Unit manifest ({len(unit_manifest)} units) -> {unit_path}")
    print(unit_manifest[["unit_id", "eol_abs_error", "rank_tag"]].to_string(index=False))

    # ── Window-level top-N manifests ─────────────────────────────────────────
    n = min(args.n, len(df))
    worst_n, best_n, mid_n = _window_manifests(df, n)

    worst_path = out_dir / "worst_windows.csv"
    best_path  = out_dir / "best_windows.csv"
    mid_path   = out_dir / "mid_windows.csv"
    worst_n.to_csv(worst_path, index=False)
    best_n.to_csv(best_path,   index=False)
    mid_n.to_csv(mid_path,     index=False)
    print(f"\nWorst {n} windows -> {worst_path}")
    print(f"Best  {n} windows -> {best_path}")
    print(f"Mid   {n} windows -> {mid_path}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "n_windows_total": int(len(df)),
        "n_units":         int(df["unit_id"].nunique()),
        "global_rmse":     float(np.sqrt((df["abs_error"] ** 2).mean())),
        "global_mae":      float(df["abs_error"].mean()),
        "global_median_ae":float(df["abs_error"].median()),
        "worst_eol_unit":  int(unit_manifest.iloc[0]["unit_id"]),
        "worst_eol_error": float(unit_manifest.iloc[0]["eol_abs_error"]),
        "best_eol_unit":   int(unit_manifest.iloc[-1]["unit_id"]),
        "best_eol_error":  float(unit_manifest.iloc[-1]["eol_abs_error"]),
        "n_per_category":  n,
        "predictions_src": args.predictions,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {summary_path}")
    print(f"  RMSE={summary['global_rmse']:.3f}  MAE={summary['global_mae']:.3f}  "
          f"MedianAE={summary['global_median_ae']:.3f}")


if __name__ == "__main__":
    main()
