"""Train a model on N-CMAPSS using unit-level train/val splits.

Usage:
    python scripts/train.py --config configs/cyclelayer.yaml
    python scripts/train.py --config configs/baseline.yaml --model cnn
    python scripts/train.py --config configs/baseline.yaml --model cnn_theta  # upper bound
    python scripts/train.py --config configs/cyclelayer.yaml --model cyclelayer_v1

Config keys (data section)
---------------------------
    stride_train (int): stride used for training windows (default: stride or 1).
    stride_eval  (int): stride used by evaluate.py for full trajectories (default 1;
                        not consumed here, documented for evaluate.py).

Theta normalization
-------------------
When the model needs theta_true (cnn_theta, lstm_theta, cyclelayer_v1):
    - A StandardScaler is fit on the train-split theta rows only.
    - It is applied in-place to base_ds._theta (normalises train, val, test rows).
    - The scaler mean/std are saved to output_dir/theta_scaler.npz.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Ensure the package is importable when run as a script (pip install -e . is preferred)
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cyclelayer.data.ncmapss import NCMAPSSDataset, SubsetByUnit
from cyclelayer.data.preprocessing import StandardScaler, fit_ops_scaler, fit_sensor_scaler, fit_theta_scaler
from cyclelayer.data.splits import extract_unit_ids, load_splits, make_unit_splits, save_splits, splits_exist
from cyclelayer.models.baselines import CNNBaseline, LSTMBaseline
from cyclelayer.models.cycle_layer import CycleLayerNet, CycleLayerNetV1
from cyclelayer.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    cfg: dict,
    n_features: int,
    n_health_params: int,
    ops_dim: int = 0,
) -> torch.nn.Module:
    """Construct a model from config dict.

    Supported types:
        cyclelayer        – CycleLayerNet v0 (BraytonCycle)
        cyclelayer_v1     – CycleLayerNetV1 (multi-task health params)
        cnn               – CNNBaseline
        cnn_theta         – CNNBaseline with theta_true concat (upper bound)
        lstm              – LSTMBaseline
        lstm_theta        – LSTMBaseline with theta_true concat (upper bound)

    Args:
        cfg: Full config dict (data + model + training sections).
        n_features: Sensor feature count (14 when use_ops=True, else 18).
        n_health_params: Number of health parameters (10 for N-CMAPSS).
        ops_dim: Operating condition channels (4 when use_ops=True, else 0).
    """
    mc = cfg["model"]
    model_type = mc["type"]
    max_rul = cfg["data"].get("max_rul", 125.0)

    if model_type == "cyclelayer":
        mc_full = dict(mc)
        mc_full["n_features"] = n_features
        # max_rul lives in cfg["data"], not cfg["model"], so inject it explicitly.
        # Without this, from_config_dict uses the dataclass default of 125.0
        # instead of the correct 99 for DS01 — causing a wrong output clamp AND
        # a wrong initial bias (62.5 instead of 49.5) from our weight init.
        mc_full.setdefault("max_rul", max_rul)
        return CycleLayerNet.from_config_dict(mc_full)

    if model_type == "cyclelayer_v1":
        mc_full = dict(mc)
        mc_full["n_features"] = n_features
        mc_full.setdefault("n_health_params", n_health_params)
        mc_full.setdefault("max_rul", max_rul)  # same fix as above
        mc_full.setdefault("ops_dim", ops_dim)
        return CycleLayerNetV1.from_config_dict(mc_full)

    use_theta = model_type.endswith("_theta")
    base_type = model_type.replace("_theta", "")
    theta_dim = n_health_params if use_theta else 0

    # OpsEncoder / fusion config (top-level model keys, only used when use_ops: true)
    ops_enc_channels  = mc.get("ops_enc_channels", None)   # None → OpsEncoder default [16,32]
    ops_enc_out_dim   = mc.get("ops_enc_out_dim", 32)
    fusion_hidden_dim = mc.get("fusion_hidden_dim", 0)

    if base_type == "cnn":
        c = mc.get("cnn", {})
        return CNNBaseline(
            n_features=n_features,
            channels=tuple(c.get("channels", [64, 128, 128])),
            kernel_size=c.get("kernel_size", 3),
            mlp_hidden=c.get("mlp_hidden", 128),
            dropout=c.get("dropout", 0.2),
            max_rul=max_rul,
            theta_true_dim=theta_dim,
            ops_dim=ops_dim,
            ops_enc_channels=ops_enc_channels,
            ops_enc_out_dim=ops_enc_out_dim,
            fusion_hidden_dim=fusion_hidden_dim,
        )
    if base_type == "lstm":
        c = mc.get("lstm", {})
        return LSTMBaseline(
            n_features=n_features,
            hidden_size=c.get("hidden_size", 64),
            n_layers=c.get("n_layers", 2),
            bidirectional=c.get("bidirectional", True),
            mlp_hidden=c.get("mlp_hidden", 64),
            dropout=c.get("dropout", 0.2),
            max_rul=max_rul,
            theta_true_dim=theta_dim,
            ops_dim=ops_dim,
            ops_enc_channels=ops_enc_channels,
            ops_enc_out_dim=ops_enc_out_dim,
            fusion_hidden_dim=fusion_hidden_dim,
        )
    raise ValueError(f"Unknown model type: {model_type!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train on N-CMAPSS with unit-level splits.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--model",
        choices=["cyclelayer", "cyclelayer_v1", "cnn", "cnn_theta", "lstm", "lstm_theta"],
        default=None,
        help="Override model.type from config.",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu / mps.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["type"] = args.model

    d = cfg["data"]
    t = cfg["training"]
    model_type = cfg["model"]["type"]

    seed = cfg.get("seed", t.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    hdf5_path = Path(d["hdf5_path"])
    dataset_name = hdf5_path.stem   # e.g. "N-CMAPSS_DS01-005"

    # ── Unit splits ───────────────────────────────────────────────────────────
    split_dir_cfg = cfg.get("split_dir") or d.get("split_dir")
    if split_dir_cfg is None:
        # Default: store alongside the run output AND in a global splits folder
        run_split_dir = Path(t.get("output_dir", "runs/experiment")) / "splits"
        global_split_dir = Path("splits") / dataset_name
        split_dir = global_split_dir
    else:
        split_dir = Path(split_dir_cfg)
        run_split_dir = None

    if splits_exist(split_dir):
        logger.info(f"Loading unit splits from {split_dir}")
        unit_splits = load_splits(split_dir)
    else:
        logger.info(f"Creating unit splits -> {split_dir}")
        all_units = extract_unit_ids(hdf5_path, split="dev")
        unit_splits = make_unit_splits(
            all_units,
            val_frac=d.get("val_split", 0.15),
            test_frac=d.get("test_split", 0.15),
            seed=seed,
        )
        save_splits(split_dir, unit_splits)
        if run_split_dir is not None and run_split_dir != split_dir:
            save_splits(run_split_dir, unit_splits)

    logger.info(
        f"Units - train: {unit_splits['train']}  "
        f"val: {unit_splits['val']}  "
        f"test: {unit_splits['test']}"
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    needs_theta = model_type in ("cyclelayer_v1", "cnn_theta", "lstm_theta")
    stride_train = d.get("stride_train", d.get("stride", 1))
    use_ops = d.get("use_ops", False)

    base_ds = NCMAPSSDataset(
        hdf5_path=hdf5_path,
        split="dev",
        window_size=d["window_size"],
        stride=stride_train,
        use_virtual_sensors=d.get("use_virtual_sensors", False),
        return_theta_true=needs_theta,
        return_ops=use_ops,
    )

    output_dir = Path(t.get("output_dir", "runs/experiment"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sensor normalization (train-split fit, applied to all rows) ────────────
    # Sensors have very different scales (alt ≈ 35000 ft, Wf ≈ 300 pph, etc.).
    # StandardScaler removes per-feature mean and variance so the CNN/LSTM sees
    # unit-normal inputs, improving gradient flow and training stability.
    n_train_rows = int(np.isin(base_ds._unit_id_arr, unit_splits["train"]).sum())
    sensor_scaler = fit_sensor_scaler(base_ds, unit_splits["train"])
    base_ds._sensors = sensor_scaler.transform(base_ds._sensors).astype(np.float32)
    logger.info(
        f"Sensor StandardScaler fitted on {n_train_rows} train rows, "
        f"applied to all {len(base_ds._sensors)} rows."
    )
    np.savez(
        output_dir / "sensor_scaler.npz",
        mean=sensor_scaler.mean_,
        std=sensor_scaler.std_,
    )

    # ── Ops normalization (train-split fit, applied to all rows) ──────────────
    # When use_ops=True, operating conditions (W: alt, Mach, TRA, T2) are in
    # _ops and excluded from _sensors.  They need their own scaler because their
    # scales differ greatly from each other (alt~35000 ft, Mach~0.8, etc.).
    if use_ops:
        ops_scaler = fit_ops_scaler(base_ds, unit_splits["train"])
        base_ds._ops = ops_scaler.transform(base_ds._ops).astype(np.float32)
        logger.info(
            f"Ops StandardScaler fitted on {n_train_rows} train rows, "
            f"applied to all {len(base_ds._ops)} rows."
        )
        np.savez(
            output_dir / "ops_scaler.npz",
            mean=ops_scaler.mean_,
            std=ops_scaler.std_,
        )

    # ── Theta normalization (train-split fit, applied to all rows) ─────────────
    if needs_theta and base_ds._theta is not None:
        theta_scaler = fit_theta_scaler(base_ds, unit_splits["train"])
        base_ds._theta = theta_scaler.transform(base_ds._theta).astype(np.float32)
        logger.info(
            f"Theta StandardScaler fitted on {n_train_rows} train rows, "
            f"applied to {len(base_ds._theta)} total rows."
        )
        np.savez(
            output_dir / "theta_scaler.npz",
            mean=theta_scaler.mean_,
            std=theta_scaler.std_,
        )

    train_ds = SubsetByUnit(base_ds, unit_splits["train"])
    val_ds   = SubsetByUnit(base_ds, unit_splits["val"])

    logger.info(
        f"Dataset  n_features={base_ds.n_features}  "
        f"n_health={base_ds.n_health_params}  "
        f"stride_train={stride_train}  "
        f"train={len(train_ds):,}  val={len(val_ds):,} windows"
    )

    n_workers = d.get("num_workers", 0)
    # persistent_workers avoids respawning subprocesses between epochs on Linux/Colab
    # (saves ~0.5s/epoch with num_workers=4).  Must be False when num_workers=0.
    persist = n_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=d["batch_size"], shuffle=True,
        num_workers=n_workers, pin_memory=True, persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=d["batch_size"] * 2, shuffle=False,
        num_workers=n_workers, pin_memory=True, persistent_workers=persist,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, base_ds.n_features, base_ds.n_health_params, base_ds.ops_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_type}  |  trainable params: {n_params:,}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=t,
        output_dir=t.get("output_dir", "runs/experiment"),
        device=args.device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
