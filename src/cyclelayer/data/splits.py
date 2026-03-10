"""Unit-level train/val/test splitting for N-CMAPSS.

Splits are defined by unit IDs (A[:,0]) to prevent data leakage.
Window-level random splits must NOT be used.

Split files are stored as plain text (one integer per line):
    splits/<dataset_name>/train_units.txt
    splits/<dataset_name>/val_units.txt
    splits/<dataset_name>/test_units.txt
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def extract_unit_ids(hdf5_path: str | Path, split: str = "dev") -> np.ndarray:
    """Return sorted unique unit IDs from the N-CMAPSS HDF5 file.

    Args:
        hdf5_path: Path to the N-CMAPSS .h5 file.
        split: ``"dev"`` or ``"test"``.

    Returns:
        Sorted integer array of unit IDs.
    """
    with h5py.File(hdf5_path, "r") as f:
        A = f[f"A_{split}"][:]
    return np.unique(A[:, 0].astype(int))


def make_unit_splits(
    unit_ids: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Partition unit IDs deterministically into train/val/test sets.

    The partition is reproducible for a given (unit_ids, val_frac, test_frac, seed).
    Units are shuffled and then sliced; no units are shared across splits.

    Args:
        unit_ids: Array of integer unit IDs.
        val_frac: Fraction of units for validation (0.0 – 1.0).
        test_frac: Fraction of units for test (0.0 – 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` each mapping to a
        sorted list of integer unit IDs.

    Raises:
        ValueError: If fractions leave fewer than 1 unit for any split.
    """
    rng = np.random.default_rng(seed)
    ids = np.sort(unit_ids).copy()
    rng.shuffle(ids)

    n = len(ids)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"val_frac={val_frac} + test_frac={test_frac} leave no training units "
            f"(total units={n})."
        )

    train_ids = sorted(ids[:n_train].tolist())
    val_ids   = sorted(ids[n_train : n_train + n_val].tolist())
    test_ids  = sorted(ids[n_train + n_val :].tolist())

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def save_splits(out_dir: str | Path, splits: dict[str, list[int]]) -> None:
    """Persist unit splits as text files.

    Args:
        out_dir: Directory to write ``{split}_units.txt`` files.
        splits: Dict returned by :func:`make_unit_splits`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out_dir / f"{name}_units.txt").write_text("\n".join(str(i) for i in ids) + "\n")


def load_splits(split_dir: str | Path) -> dict[str, list[int]]:
    """Load unit splits from text files created by :func:`save_splits`.

    Args:
        split_dir: Directory containing ``{split}_units.txt`` files.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``.

    Raises:
        FileNotFoundError: If any of the three expected files is missing.
    """
    split_dir = Path(split_dir)
    result: dict[str, list[int]] = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}_units.txt"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        result[name] = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    return result


def splits_exist(split_dir: str | Path) -> bool:
    """Return True if all three split files exist in ``split_dir``."""
    d = Path(split_dir)
    return all((d / f"{n}_units.txt").exists() for n in ("train", "val", "test"))
