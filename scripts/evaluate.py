"""Evaluate a trained model on N-CMAPSS with per-unit metrics.

Usage:
    python scripts/evaluate.py --config configs/cyclelayer.yaml
    python scripts/evaluate.py --config configs/baseline.yaml --checkpoint runs/baseline/best.pt
    python scripts/evaluate.py --config configs/cyclelayer.yaml --split dev
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cyclelayer.data.ncmapss import NCMAPSSDataset
from cyclelayer.data.splits import load_splits, splits_exist
from cyclelayer.evaluation.metrics import (
    evaluate_all,
    ph_debug_stats,
    prediction_horizon,
    s_score_samples,
)
from scripts.train import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _ema_smooth(preds: np.ndarray, alpha: float) -> np.ndarray:
    """Apply causal exponential moving average along the first axis.

    EMA(t) = alpha * pred(t) + (1 - alpha) * EMA(t-1)
    alpha=1.0 → no smoothing (identity); alpha→0 → heavy smoothing.
    """
    if alpha >= 1.0:
        return preds
    out = preds.copy()
    for i in range(1, len(out)):
        out[i] = alpha * preds[i] + (1.0 - alpha) * out[i - 1]
    return out


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds, targets) preserving sample order (shuffle=False required).

    Automatically detects whether the model uses a separate ops path
    (getattr(model, 'ops_dim', 0) > 0) and passes ops from 3-tuple batches.
    """
    model.eval()
    has_ops_model = getattr(model, "ops_dim", 0) > 0
    preds, targets = [], []
    for batch in loader:
        x   = batch[0].to(device)
        rul = batch[1]
        if has_ops_model and len(batch) >= 3:
            ops = batch[2].to(device)   # 3-tuple (x, rul, ops) when use_ops=True
            out = model(x, ops=ops)
        else:
            out = model(x)
        if isinstance(out, tuple):   # CycleLayerNetV1 returns (rul, theta_hat)
            out = out[0]
        preds.append(out.cpu().numpy())
        targets.append(rul.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-unit evaluation of a trained model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--output", default=None, help="Override output path for metrics.json.")
    parser.add_argument(
        "--ema_alpha", type=float, default=1.0,
        help="EMA smoothing factor applied per-unit to predictions before metrics "
             "(1.0 = no smoothing, 0.3 = moderate, 0.1 = heavy). "
             "Reduces mid-trajectory spikes; does not affect raw predictions.csv.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    d  = cfg["data"]
    t  = cfg["training"]
    hdf5_path = Path(d["hdf5_path"])

    # When evaluating on the dev set, restrict to val+test units to avoid
    # evaluating on training data.
    eval_units: list[int] | None = None
    if args.split == "dev":
        split_dir = Path(cfg.get("split_dir") or d.get("split_dir") or
                         Path("splits") / hdf5_path.stem)
        if splits_exist(split_dir):
            splits = load_splits(split_dir)
            eval_units = splits["val"] + splits["test"]
            logger.info(f"Evaluating on held-out dev units: {eval_units}")
        else:
            logger.warning("No split file found; evaluating all dev units.")

    # Dataset — use stride_eval (default 1) for full per-unit trajectories
    stride_eval = d.get("stride_eval", 1)
    use_ops = d.get("use_ops", False)
    dataset = NCMAPSSDataset(
        hdf5_path=hdf5_path,
        split=args.split,
        window_size=d["window_size"],
        stride=stride_eval,
        use_virtual_sensors=d.get("use_virtual_sensors", False),
        return_theta_true=False,
        return_ops=use_ops,
        units=eval_units,
    )
    loader = DataLoader(
        dataset,
        batch_size=d["batch_size"] * 2,
        shuffle=False,   # must preserve order for unit grouping
        num_workers=d.get("num_workers", 0),
    )

    # Model
    ckpt_path = args.checkpoint or cfg["evaluation"]["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg, dataset.n_features, dataset.n_health_params, dataset.ops_dim).to(device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    # Apply sensor scaler if one was saved alongside the checkpoint
    ckpt_dir = Path(ckpt_path).parent
    sensor_scaler_path = ckpt_dir / "sensor_scaler.npz"
    if sensor_scaler_path.exists():
        sc = np.load(sensor_scaler_path)
        sc_mean, sc_std = sc["mean"], sc["std"]
        sc_std_safe = np.where(sc_std == 0, 1.0, sc_std)
        dataset._sensors = ((dataset._sensors - sc_mean) / sc_std_safe).astype(np.float32)
        logger.info(f"Sensor scaler applied from {sensor_scaler_path}")
    else:
        logger.warning(
            f"No sensor_scaler.npz found at {sensor_scaler_path} — using raw sensors. "
            "Predictions may be poor if the model was trained with normalization."
        )

    # Apply ops scaler when use_ops=True (saved during training)
    if use_ops:
        ops_scaler_path = ckpt_dir / "ops_scaler.npz"
        if ops_scaler_path.exists():
            sc = np.load(ops_scaler_path)
            sc_std_safe = np.where(sc["std"] == 0, 1.0, sc["std"])
            dataset._ops = ((dataset._ops - sc["mean"]) / sc_std_safe).astype(np.float32)
            logger.info(f"Ops scaler applied from {ops_scaler_path}")
        else:
            logger.warning(
                f"No ops_scaler.npz found at {ops_scaler_path} — using raw ops. "
                "Predictions may be poor if the model was trained with normalized ops."
            )

    # Inference
    preds, targets = run_inference(model, loader, device)
    unit_ids       = dataset.unit_ids_array   # aligned with preds (shuffle=False)
    unique_units   = np.unique(unit_ids)
    n_samples      = len(preds)

    # Global metrics
    global_metrics = evaluate_all(preds, targets)
    s_sum          = global_metrics["s_score"]
    # s_score_mean divides by the number of windows, making it comparable
    # across experiments with different strides.  s_score_sum is the raw PHM'08
    # figure and grows linearly with N_samples, so it's unreliable for comparisons.
    s_mean         = s_sum / n_samples if n_samples > 0 else float("nan")

    logger.info(f"RMSE         : {global_metrics['rmse']:.4f}")
    logger.info(f"S-score sum  : {s_sum:.2f}   (N={n_samples:,})")
    logger.info(f"S-score mean : {s_mean:.6f}  (per-window; stride-invariant)")

    # Per-unit loop: PH, S-score mean, and PH debug stats
    alpha   = cfg["evaluation"].get("alpha", 0.2)
    ph_ks   = (50, 100, 200)  # frac_within_alpha_lastK breakpoints
    ema_alpha = args.ema_alpha
    if ema_alpha < 1.0:
        logger.info(f"EMA smoothing enabled: alpha={ema_alpha}")

    ph_values:             list[float]       = []
    ph_none_count:         int               = 0
    per_unit_s_mean:       list[float]       = []
    per_unit_debug:        list[dict]        = []
    per_unit_summary:      list[dict]        = []  # for logging

    for uid in unique_units:
        mask  = unit_ids == uid
        p_u   = _ema_smooth(preds[mask], ema_alpha)
        t_u   = targets[mask]

        # Prediction horizon
        ph = prediction_horizon(p_u, t_u, alpha=alpha)
        if ph is None:
            ph_none_count += 1
        else:
            ph_values.append(float(ph))

        # Per-unit S-score mean (not sum): stride-invariant, fair per-engine comparison
        s_samp = s_score_samples(p_u, t_u)
        per_unit_s_mean.append(float(np.mean(s_samp)))

        # PH diagnostic stats
        dbg = ph_debug_stats(p_u, t_u, alpha=alpha, ks=ph_ks)
        per_unit_debug.append(dbg)

        per_unit_summary.append({
            "unit_id":  int(uid),
            "ph":       ph,
            "s_mean":   float(np.mean(s_samp)),
            "max_err":  dbg["max_abs_error"],
            "p95_err":  dbg["p95_abs_error"],
            "frac50":   dbg["frac_within_alpha_last50"],
        })

    # Aggregate PH stats
    ph_median = float(np.median(ph_values)) if ph_values else float("nan")
    ph_p10    = float(np.percentile(ph_values, 10)) if ph_values else float("nan")
    ph_p90    = float(np.percentile(ph_values, 90)) if ph_values else float("nan")

    logger.info(
        f"PH (alpha={alpha})  "
        f"median={ph_median:.1f}  p10={ph_p10:.1f}  p90={ph_p90:.1f}  "
        f"none={ph_none_count}/{len(unique_units)} units"
    )

    # Aggregate per-unit S-score stats
    # Median of per-unit means is a robust summary that won't blow up if one unit
    # has many windows.  p10/p90 show spread.
    s_unit_median = float(np.median(per_unit_s_mean)) if per_unit_s_mean else float("nan")
    s_unit_p10    = float(np.percentile(per_unit_s_mean, 10)) if per_unit_s_mean else float("nan")
    s_unit_p90    = float(np.percentile(per_unit_s_mean, 90)) if per_unit_s_mean else float("nan")

    logger.info(
        f"S-score (per-unit mean)  "
        f"median={s_unit_median:.4f}  p10={s_unit_p10:.4f}  p90={s_unit_p90:.4f}"
    )

    # Aggregate PH debug stats across units
    ph_debug_agg: dict[str, float] = {}
    for k in ph_ks:
        key  = f"frac_within_alpha_last{k}"
        vals = [d[key] for d in per_unit_debug]
        ph_debug_agg[f"ph_{key}_median"] = float(np.median(vals))
        logger.info(f"  frac_within_alpha_last{k:3d}  median={ph_debug_agg[f'ph_{key}_median']:.3f}")

    max_err_vals = [d["max_abs_error"] for d in per_unit_debug]
    p95_err_vals = [d["p95_abs_error"] for d in per_unit_debug]
    ph_debug_agg["max_abs_error_unit_median"] = float(np.median(max_err_vals))
    ph_debug_agg["p95_abs_error_unit_median"] = float(np.median(p95_err_vals))

    # Per-unit summary table to log
    logger.info("Per-unit breakdown:")
    logger.info(f"  {'unit':>4}  {'ph':>6}  {'s_mean':>9}  {'max_err':>8}  {'frac@50':>8}")
    for row in per_unit_summary:
        ph_str = f"{row['ph']:6.1f}" if row["ph"] is not None else "  None"
        logger.info(
            f"  {row['unit_id']:4d}  {ph_str}  "
            f"{row['s_mean']:9.4f}  {row['max_err']:8.2f}  {row['frac50']:8.3f}"
        )

    # Save metrics.json
    out_path = Path(
        args.output or Path(t.get("output_dir", "runs/experiment")) / "metrics.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        # --- Standard metrics (keep existing keys for backward compat) ---
        "rmse":          global_metrics["rmse"],
        "s_score":       s_sum,           # existing key name kept for consumers
        "ph_median":     ph_median,
        "ph_p10":        ph_p10,
        "ph_p90":        ph_p90,
        "ph_none_count": ph_none_count,
        "alpha":         alpha,
        "ema_alpha":     ema_alpha,
        "split":         args.split,
        "checkpoint":    str(ckpt_path),
        "epoch":         ckpt.get("epoch"),

        # --- Extended S-score fields ---
        # s_score_sum == s_score (explicit alias).  s_score_mean is the key metric
        # for cross-run comparisons because it does not scale with N_samples.
        "s_score_sum":            s_sum,
        "s_score_mean":           s_mean,
        "n_samples":              n_samples,
        # Per-unit means: more robust than per-window because each unit
        # contributes equally regardless of trajectory length.
        "s_score_unit_median_mean": s_unit_median,
        "s_score_unit_p10_mean":    s_unit_p10,
        "s_score_unit_p90_mean":    s_unit_p90,

        # --- PH debug stats (aggregated across units) ---
        # frac_within_alpha_lastK: if high + PH=None => mid-trajectory spike issue.
        # max/p95 abs_error: scale of worst mistakes per unit.
        **ph_debug_agg,
    }
    # metrics.json is written at the end of main() after cycle-avg metrics
    # are appended — see the bottom of this function.
    logger.info(f"Metrics will be saved -> {out_path}")

    # Build cycle_id array: for each window i, take the cycle of its last sample.
    # dataset._index_list[i] = (uid, w_start); last HDF5 row = unit_start + w_start
    # + window_size - 1.  Vectorised to avoid a Python loop over millions of windows.
    _uid_arr  = np.array([uid for uid, _  in dataset._index_list], dtype=np.int32)
    _ws_arr   = np.array([ws  for _,   ws in dataset._index_list], dtype=np.int64)
    _unit_starts = np.array(
        [dataset._unit_ranges[uid][0] for uid in _uid_arr], dtype=np.int64
    )
    _global_rows  = _unit_starts + _ws_arr + dataset.window_size - 1
    cycle_ids_all = dataset._cycle_arr[_global_rows].astype(np.int32)

    # Export per-window predictions CSV (cycle_id added for cycle-avg eval)
    time_index = np.zeros(n_samples, dtype=np.int64)
    for uid in unique_units:
        mask = unit_ids == uid
        time_index[mask] = np.arange(mask.sum())

    pred_df = pd.DataFrame({
        "unit_id":    unit_ids.astype(np.int32),
        "cycle_id":   cycle_ids_all,
        "time_index": time_index,
        "y_true_rul": targets.astype(np.float32),
        "y_pred_rul": preds.astype(np.float32),
        "abs_error":  np.abs(preds - targets).astype(np.float32),
        "split":      args.split,
    })
    pred_path = out_path.parent / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved -> {pred_path}  ({n_samples:,} rows)")

    # ── Cycle-averaged evaluation — Chao 2022, Eq. 9 ─────────────────────────
    # y_hat_cycle[c] = (1/m_c) * Σ y_hat[j]  for all j in flight-cycle c
    # Groups by (unit_id, cycle_id); true RUL is also averaged (constant within
    # a cycle for N-CMAPSS, so the mean equals the single value).
    df_cycle = (
        pred_df.groupby(["unit_id", "cycle_id"])[["y_true_rul", "y_pred_rul"]]
        .mean()
        .reset_index()
    )
    n_cycles      = len(df_cycle)
    cy_pred        = df_cycle["y_pred_rul"].values
    cy_true        = df_cycle["y_true_rul"].values
    rmse_cycle     = float(np.sqrt(np.mean((cy_pred - cy_true) ** 2)))
    s_cycle_arr    = s_score_samples(cy_pred, cy_true)
    s_score_cycle  = float(np.mean(s_cycle_arr))

    cycle_pred_path = out_path.parent / "predictions_cycle.csv"
    df_cycle.to_csv(cycle_pred_path, index=False)
    logger.info(
        f"Cycle-avg  RMSE={rmse_cycle:.4f}  S-score/cycle={s_score_cycle:.4f}  "
        f"({n_cycles:,} cycles)  -> {cycle_pred_path}"
    )

    # Add cycle-avg metrics to the dict
    metrics["rmse_cycle"]         = rmse_cycle
    metrics["s_score_cycle_mean"] = s_score_cycle
    metrics["n_cycles"]           = n_cycles

    # ── Within-cycle scatter metrics ──────────────────────────────────────────
    # std(y_pred) within each (unit, cycle) group measures how much the model's
    # RUL estimate varies across timesteps within a single flight cycle.
    # Low values indicate the model is insensitive to within-cycle ops changes
    # (desirable: degradation, not operating point, should drive RUL).
    wc = (
        pred_df.groupby(["unit_id", "cycle_id"])["y_pred_rul"]
        .agg(n_windows="count", pred_std="std", pred_range=lambda x: x.max() - x.min())
        .reset_index()
    )
    wc["pred_std"]   = wc["pred_std"].fillna(0.0)    # cycles with 1 window → NaN std
    wc["pred_range"] = wc["pred_range"].fillna(0.0)

    wc_std_median   = float(wc["pred_std"].median())
    wc_range_median = float(wc["pred_range"].median())

    within_cycle_path = out_path.parent / "within_cycle_scatter.csv"
    wc.to_csv(within_cycle_path, index=False)
    logger.info(
        f"Within-cycle scatter: std_median={wc_std_median:.3f}  "
        f"range_median={wc_range_median:.3f}  -> {within_cycle_path}"
    )

    metrics["within_cycle_pred_std_median"]   = wc_std_median
    metrics["within_cycle_pred_range_median"] = wc_range_median

    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved -> {out_path}")


if __name__ == "__main__":
    main()
