"""Leave-One-Unit-Out (LOUO) cross-validation for N-CMAPSS DS01.

For each unit U in the dataset:
  - Train on all remaining units except the last (by ID), which is used as val.
  - Evaluate on the held-out unit U.
  - Record per-unit RMSE, S-score, and prediction horizon.

Aggregated results (mean +/- std, median PH) are printed and saved to
  runs/louo_<model>/results.json

Usage:
    python scripts/run_louo.py --config configs/baseline.yaml --model cnn
    python scripts/run_louo.py --config configs/cyclelayer.yaml --model cyclelayer_v1
    python scripts/run_louo.py --config configs/test_run.yaml --model cnn  # smoke test

Val unit selection
------------------
For each fold with test_unit U, the remaining units are sorted by ID.
The last remaining unit is reserved as val for early stopping; all others
are used for training.  With DS01-005 (6 units), each fold has 4 train + 1 val.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cyclelayer.data.ncmapss import NCMAPSSDataset, SubsetByUnit
from cyclelayer.data.preprocessing import StandardScaler, fit_theta_scaler
from cyclelayer.data.splits import extract_unit_ids, save_splits
from cyclelayer.evaluation.metrics import evaluate_all, prediction_horizon
from cyclelayer.training.trainer import Trainer
from scripts.train import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single LOUO fold
# ---------------------------------------------------------------------------

def run_fold(
    test_unit: int,
    all_units: list[int],
    cfg: dict,
    fold_dir: Path,
    device_str: str | None,
) -> dict:
    """Train and evaluate one LOUO fold.

    Returns a dict with keys: rmse, s_score, ph, test_unit, val_unit, train_units.
    """
    d = cfg["data"]
    t = cfg["training"]
    hdf5_path = Path(d["hdf5_path"])
    model_type = cfg["model"]["type"]
    needs_theta = model_type in ("cyclelayer_v1", "cnn_theta", "lstm_theta")

    stride_train = d.get("stride_train", d.get("stride", 1))
    stride_eval  = d.get("stride_eval", 1)

    remaining = sorted([u for u in all_units if u != test_unit])
    if len(remaining) < 2:
        # Only one remaining unit — use it for both train and val
        train_units = remaining
        val_unit    = remaining[0]
    else:
        val_unit    = remaining[-1]   # highest-ID remaining unit as val
        train_units = remaining[:-1]

    logger.info(
        f"Fold test_unit={test_unit}  val_unit={val_unit}  "
        f"train_units={train_units}"
    )

    # ── Save per-fold split files for reproducibility ─────────────────────────
    fold_split_dir = fold_dir / "splits"
    save_splits(fold_split_dir, {
        "train": train_units,
        "val":   [val_unit],
        "test":  [test_unit],
    })

    # ── Training dataset (stride_train) ───────────────────────────────────────
    base_ds = NCMAPSSDataset(
        hdf5_path=hdf5_path,
        split="dev",
        window_size=d["window_size"],
        stride=stride_train,
        use_virtual_sensors=d.get("use_virtual_sensors", False),
        return_theta_true=needs_theta,
    )

    # Theta normalisation: fit ONLY on train units (no leakage)
    theta_scaler: StandardScaler | None = None
    if needs_theta and base_ds._theta is not None:
        theta_scaler = fit_theta_scaler(base_ds, train_units)
        base_ds._theta = theta_scaler.transform(base_ds._theta).astype(np.float32)

    train_ds = SubsetByUnit(base_ds, train_units)
    val_ds   = SubsetByUnit(base_ds, [val_unit])

    logger.info(
        f"  train={len(train_ds):,} windows  val={len(val_ds):,} windows  "
        f"(stride_train={stride_train})"
    )

    train_loader = DataLoader(
        train_ds, batch_size=d["batch_size"], shuffle=True,
        num_workers=d.get("num_workers", 0), pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=d["batch_size"] * 2, shuffle=False,
        num_workers=d.get("num_workers", 0), pin_memory=True,
    )

    model = build_model(cfg, base_ds.n_features, base_ds.n_health_params)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=t,
        output_dir=fold_dir,
        device=device_str,
    )
    trainer.train()

    # ── Evaluation dataset (stride_eval=1 for full trajectory) ────────────────
    test_ds = NCMAPSSDataset(
        hdf5_path=hdf5_path,
        split="dev",
        window_size=d["window_size"],
        stride=stride_eval,
        use_virtual_sensors=d.get("use_virtual_sensors", False),
        return_theta_true=False,
        units=[test_unit],
    )

    # Apply the same theta scaler to the test dataset if needed
    if theta_scaler is not None and test_ds._theta is not None:
        test_ds._theta = theta_scaler.transform(test_ds._theta).astype(np.float32)

    # Save theta scaler for reproducibility
    if theta_scaler is not None:
        np.savez(
            fold_dir / "theta_scaler.npz",
            mean=theta_scaler.mean_,
            std=theta_scaler.std_,
        )

    test_loader = DataLoader(
        test_ds, batch_size=d["batch_size"] * 2, shuffle=False,
        num_workers=0,
    )

    # Load best checkpoint and run inference
    ckpt = torch.load(fold_dir / "best.pt", map_location=trainer.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            x   = batch[0].to(trainer.device)
            rul = batch[1]
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            preds_list.append(out.cpu().numpy())
            targets_list.append(rul.numpy())

    preds   = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    metrics = evaluate_all(preds, targets)
    alpha = cfg["evaluation"].get("alpha", 0.2)
    ph = prediction_horizon(preds, targets, alpha=alpha)
    metrics["ph"]          = float(ph) if ph is not None else None
    metrics["test_unit"]   = test_unit
    metrics["val_unit"]    = val_unit
    metrics["train_units"] = train_units

    logger.info(
        f"  -> RMSE={metrics['rmse']:.3f}  "
        f"S-score={metrics['s_score']:.1f}  "
        f"PH={'None' if ph is None else ph}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-One-Unit-Out cross-validation for N-CMAPSS DS01."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--model",
        choices=["cyclelayer", "cyclelayer_v1", "cnn", "cnn_theta", "lstm", "lstm_theta"],
        default=None,
        help="Override model.type from config.",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu / mps.")
    parser.add_argument(
        "--units", nargs="+", type=int, default=None,
        help="Run only specific test units (default: all).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["type"] = args.model

    model_type = cfg["model"]["type"]
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    hdf5_path    = Path(cfg["data"]["hdf5_path"])
    all_units    = sorted(extract_unit_ids(hdf5_path, split="dev").tolist())
    test_units   = args.units if args.units else all_units

    # LOUO output goes in a sibling directory named louo_<model>
    base_out_dir = Path(cfg["training"].get("output_dir", "runs/experiment"))
    louo_dir     = base_out_dir.parent / f"louo_{model_type}"
    louo_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"LOUO  model={model_type}  units={all_units}  test_folds={test_units}")
    logger.info(f"Output dir: {louo_dir}")

    fold_results: dict[int, dict] = {}

    for test_unit in test_units:
        logger.info(f"\n{'-' * 60}")
        fold_dir = louo_dir / f"fold_{test_unit}"

        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["training"]["output_dir"] = str(fold_dir)

        metrics = run_fold(
            test_unit=test_unit,
            all_units=all_units,
            cfg=fold_cfg,
            fold_dir=fold_dir,
            device_str=args.device,
        )
        fold_results[test_unit] = metrics

    # ── Aggregate ─────────────────────────────────────────────────────────────
    rmse_vals = [v["rmse"]    for v in fold_results.values()]
    ss_vals   = [v["s_score"] for v in fold_results.values()]
    ph_vals   = [v["ph"]      for v in fold_results.values() if v["ph"] is not None]
    ph_none   = sum(1 for v in fold_results.values() if v["ph"] is None)

    aggregate = {
        "model":        model_type,
        "n_folds":      len(fold_results),
        "rmse_mean":    float(np.mean(rmse_vals)),
        "rmse_std":     float(np.std(rmse_vals)),
        "s_score_mean": float(np.mean(ss_vals)),
        "s_score_std":  float(np.std(ss_vals)),
        "ph_median":    float(np.median(ph_vals)) if ph_vals else None,
        "ph_p10":       float(np.percentile(ph_vals, 10)) if ph_vals else None,
        "ph_p90":       float(np.percentile(ph_vals, 90)) if ph_vals else None,
        "ph_none_count": ph_none,
        "per_fold":     {str(k): v for k, v in fold_results.items()},
    }

    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"LOUO {model_type} ({len(fold_results)} folds): "
        f"RMSE={aggregate['rmse_mean']:.3f}+/-{aggregate['rmse_std']:.3f}  "
        f"S-score={aggregate['s_score_mean']:.1f}  "
        f"PH_median={aggregate['ph_median']}"
    )

    out_path = louo_dir / "results.json"
    out_path.write_text(json.dumps(aggregate, indent=2))
    logger.info(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
