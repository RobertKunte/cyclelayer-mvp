"""Deterministic case runner: train + evaluate one case.

Creates a unique, timestamped run directory and saves reproducibility metadata
before launching training and evaluation.

Run directory layout
--------------------
runs/<YYYYMMDD_HHMMSS>_<dataset>_<model>/
    run_meta.json        – git hash, command, seed, timestamp
    config_frozen.yaml   – frozen copy of the config used (output_dir patched)
    best.pt              – best checkpoint written by Trainer
    last.pt              – last checkpoint written by Trainer
    theta_scaler.npz     – theta scaler stats (only when model uses theta)
    metrics.json         – RMSE, S-score, PH from evaluate.py
    predictions.csv      – per-window predictions from evaluate.py
    tb/                  – TensorBoard event files

For LOUO (--louo flag), run_louo.py is called instead of train.py, and the
fold subdirectories are placed inside the run directory.

Usage
-----
    # Single training run:
    python scripts/run_cases.py --config configs/baseline.yaml --model cnn
    python scripts/run_cases.py --config configs/cyclelayer.yaml --model cyclelayer_v1

    # LOUO cross-validation:
    python scripts/run_cases.py --config configs/cyclelayer.yaml --model cyclelayer_v1 --louo

    # Specific test units only (LOUO):
    python scripts/run_cases.py --config configs/cyclelayer.yaml --model cyclelayer_v1 --louo --units 1 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
_SCRIPTS = _ROOT / "scripts"


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def _dataset_label(hdf5_path: str) -> str:
    """Extract a short label from the HDF5 path, e.g. 'DS01-005'."""
    stem = Path(hdf5_path).stem          # e.g. "N-CMAPSS_DS01-005"
    parts = stem.split("_")
    return parts[-1] if len(parts) > 1 else stem


def _make_run_dir(dataset_label: str, model_type: str, louo: bool) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "louo" if louo else "train"
    name = f"{ts}_{dataset_label}_{model_type}_{suffix}"
    run_dir = _ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _patch_and_save_config(cfg: dict, run_dir: Path, louo: bool) -> Path:
    """Write a frozen config copy with output_dir pointing into run_dir."""
    cfg = dict(cfg)
    cfg["training"] = dict(cfg.get("training", {}))
    if louo:
        # run_louo.py uses base_out_dir.parent / "louo_{model}" as louo_dir.
        # Setting output_dir = run_dir / "_louo_anchor" causes louo_dir to land
        # at run_dir / "louo_{model}" — fully inside run_dir.
        cfg["training"]["output_dir"] = str(run_dir / "_louo_anchor")
    else:
        cfg["training"]["output_dir"] = str(run_dir)
    # evaluation checkpoint always points into run_dir/best.pt
    cfg["evaluation"] = dict(cfg.get("evaluation", {}))
    cfg["evaluation"]["checkpoint"] = str(run_dir / "best.pt")

    frozen_path = run_dir / "config_frozen.yaml"
    with open(frozen_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return frozen_path


def _save_meta(run_dir: Path, args: argparse.Namespace, config_path: Path) -> None:
    meta = {
        "timestamp":   datetime.now().isoformat(),
        "git_commit":  _git_hash(),
        "command":     sys.argv,
        "config_src":  str(args.config),
        "model":       args.model,
        "louo":        args.louo,
        "device":      args.device,
        "run_dir":     str(run_dir),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def _run_training(frozen_cfg: Path, model: str, device: str | None, run_dir: Path) -> None:
    cmd = [sys.executable, str(_SCRIPTS / "train.py"),
           "--config", str(frozen_cfg),
           "--model", model]
    if device:
        cmd += ["--device", device]
    print(f"\n{'-'*60}\nTraining: {' '.join(cmd)}\n{'-'*60}")
    subprocess.run(cmd, check=True)


def _run_louo(
    frozen_cfg: Path,
    model: str,
    device: str | None,
    units: list[int] | None,
) -> None:
    cmd = [sys.executable, str(_SCRIPTS / "run_louo.py"),
           "--config", str(frozen_cfg),
           "--model", model]
    if device:
        cmd += ["--device", device]
    if units:
        cmd += ["--units"] + [str(u) for u in units]
    print(f"\n{'-'*60}\nLOUO: {' '.join(cmd)}\n{'-'*60}")
    subprocess.run(cmd, check=True)


def _run_evaluate(frozen_cfg: Path, model_type: str, device: str | None, run_dir: Path) -> None:
    ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        print(f"[WARN] Checkpoint not found at {ckpt}; skipping evaluate.")
        return
    cmd = [sys.executable, str(_SCRIPTS / "evaluate.py"),
           "--config", str(frozen_cfg),
           "--checkpoint", str(ckpt),
           "--split", "dev",
           "--output", str(run_dir / "metrics.json")]
    if device:
        cmd += ["--device", device]
    print(f"\n{'-'*60}\nEvaluate: {' '.join(cmd)}\n{'-'*60}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic case runner for cyclelayer-mvp.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--model",
        choices=["cyclelayer", "cyclelayer_v1", "cnn", "cnn_theta", "lstm", "lstm_theta"],
        required=True,
    )
    parser.add_argument("--device", default=None, help="cuda / cpu / mps.")
    parser.add_argument("--louo", action="store_true", help="Run LOUO instead of single train.")
    parser.add_argument(
        "--units", nargs="+", type=int, default=None,
        help="(LOUO only) Specific test units to evaluate.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_label = _dataset_label(cfg["data"]["hdf5_path"])
    run_dir = _make_run_dir(dataset_label, args.model, args.louo)

    # Frozen config must be written before any subprocess call
    frozen_cfg = _patch_and_save_config(cfg, run_dir, args.louo)
    _save_meta(run_dir, args, frozen_cfg)
    # Also copy original config for reference
    shutil.copy2(args.config, run_dir / "config_original.yaml")

    print(f"\nRun dir : {run_dir}")
    print(f"Git hash: {_git_hash()}")
    print(f"Model   : {args.model}  |  LOUO: {args.louo}\n")

    if args.louo:
        _run_louo(frozen_cfg, args.model, args.device, args.units)
        print(f"\nLOUO complete. Results in: {run_dir}")
    else:
        _run_training(frozen_cfg, args.model, args.device, run_dir)
        _run_evaluate(frozen_cfg, args.model, args.device, run_dir)
        print(f"\nCase complete. Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
