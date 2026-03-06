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
from cyclelayer.evaluation.metrics import evaluate_all, prediction_horizon
from scripts.train import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds, targets) preserving sample order (shuffle=False required)."""
    model.eval()
    preds, targets = [], []
    for batch in loader:
        x   = batch[0].to(device)
        rul = batch[1]
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

    # ── Dataset — use stride_eval (default 1) for full per-unit trajectories ──
    stride_eval = d.get("stride_eval", 1)
    dataset = NCMAPSSDataset(
        hdf5_path=hdf5_path,
        split=args.split,
        window_size=d["window_size"],
        stride=stride_eval,
        use_virtual_sensors=d.get("use_virtual_sensors", False),
        return_theta_true=False,
        units=eval_units,
    )
    loader = DataLoader(
        dataset,
        batch_size=d["batch_size"] * 2,
        shuffle=False,   # must preserve order for unit grouping
        num_workers=d.get("num_workers", 0),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, dataset.n_features, dataset.n_health_params).to(device)
    ckpt_path = args.checkpoint or cfg["evaluation"]["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    # ── Inference ─────────────────────────────────────────────────────────────
    preds, targets = run_inference(model, loader, device)
    unit_ids = dataset.unit_ids_array   # aligned with preds (shuffle=False)

    # ── Global metrics ────────────────────────────────────────────────────────
    global_metrics = evaluate_all(preds, targets)
    logger.info(f"RMSE    : {global_metrics['rmse']:.4f}")
    logger.info(f"S-score : {global_metrics['s_score']:.2f}")

    # ── Per-unit prediction horizon ───────────────────────────────────────────
    alpha = cfg["evaluation"].get("alpha", 0.2)
    ph_values: list[float] = []
    ph_none_count = 0
    unique_units = np.unique(unit_ids)

    for uid in unique_units:
        mask = unit_ids == uid
        ph = prediction_horizon(preds[mask], targets[mask], alpha=alpha)
        if ph is None:
            ph_none_count += 1
        else:
            ph_values.append(float(ph))

    ph_median = float(np.median(ph_values)) if ph_values else float("nan")
    ph_p10    = float(np.percentile(ph_values, 10)) if ph_values else float("nan")
    ph_p90    = float(np.percentile(ph_values, 90)) if ph_values else float("nan")

    logger.info(
        f"Prediction horizon (alpha={alpha})  "
        f"median={ph_median:.1f}  p10={ph_p10:.1f}  p90={ph_p90:.1f}  "
        f"none={ph_none_count}/{len(unique_units)} units"
    )

    # ── Save metrics.json ─────────────────────────────────────────────────────
    out_path = Path(
        args.output or Path(t.get("output_dir", "runs/experiment")) / "metrics.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "rmse":          global_metrics["rmse"],
        "s_score":       global_metrics["s_score"],
        "ph_median":     ph_median,
        "ph_p10":        ph_p10,
        "ph_p90":        ph_p90,
        "ph_none_count": ph_none_count,
        "alpha":         alpha,
        "split":         args.split,
        "checkpoint":    str(ckpt_path),
        "epoch":         ckpt.get("epoch"),
    }
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved -> {out_path}")

    # ── Export per-window predictions CSV ─────────────────────────────────────
    time_index = np.zeros(len(preds), dtype=np.int64)
    for uid in unique_units:
        mask = unit_ids == uid
        time_index[mask] = np.arange(mask.sum())

    pred_df = pd.DataFrame({
        "unit_id":    unit_ids.astype(np.int32),
        "time_index": time_index,
        "y_true_rul": targets.astype(np.float32),
        "y_pred_rul": preds.astype(np.float32),
        "abs_error":  np.abs(preds - targets).astype(np.float32),
        "split":      args.split,
    })
    pred_path = out_path.parent / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved -> {pred_path}  ({len(pred_df):,} rows)")


if __name__ == "__main__":
    main()
