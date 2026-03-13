"""Tests for dataset splitting, window integrity, and per-unit evaluation.

All tests use synthetic in-memory data (no HDF5 file required) by
monkey-patching NCMAPSSDataset._load or constructing objects directly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cyclelayer.data.ncmapss import NCMAPSSDataset, SubsetByUnit
from cyclelayer.data.splits import (
    load_splits,
    make_unit_splits,
    save_splits,
    splits_exist,
)
from cyclelayer.evaluation.metrics import prediction_horizon


# ---------------------------------------------------------------------------
# Helpers: build a synthetic NCMAPSSDataset without HDF5
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(
    unit_counts: dict[int, int],
    window_size: int = 5,
    stride: int = 1,
    n_features: int = 4,
    n_health: int = 3,
    return_theta_true: bool = False,
    return_ops: bool = False,
    n_ops: int = 4,
) -> NCMAPSSDataset:
    """Build a synthetic dataset with controlled unit time-series lengths.

    unit_counts: {uid: n_timesteps} for each unit.
    n_ops: number of operating condition channels in _ops (matches real W=4).
    """
    rows: list[np.ndarray] = []
    rul_rows: list[float] = []
    theta_rows: list[np.ndarray] = []
    ops_rows: list[np.ndarray] = []
    A_rows: list[np.ndarray] = []

    for uid, n_steps in unit_counts.items():
        for t in range(n_steps):
            rows.append(np.full(n_features, float(uid) + t * 0.01, dtype=np.float32))
            rul_rows.append(float(n_steps - 1 - t))
            theta_rows.append(np.ones(n_health, dtype=np.float32) * uid * 0.1)
            # Synthetic ops: different scale per channel, distinct per unit
            ops_rows.append(np.array(
                [uid * 1000.0, uid * 0.1, uid * 10.0, uid * 100.0],
                dtype=np.float32
            )[:n_ops])
            A_rows.append(np.array([uid, t, 1, 1], dtype=np.float32))

    sensors_all = np.stack(rows)
    rul_all     = np.array(rul_rows, dtype=np.float32)
    theta_all   = np.stack(theta_rows)
    ops_all     = np.stack(ops_rows)
    A_all       = np.stack(A_rows)
    unit_ids    = A_all[:, 0].astype(np.int32)
    cycle_ids   = A_all[:, 1].astype(np.int32)

    # Sort by (unit, cycle) — already sorted but let's be explicit
    order = np.lexsort((A_all[:, 1], unit_ids))
    sensors_all = sensors_all[order]
    rul_all     = rul_all[order]
    theta_all   = theta_all[order]
    ops_all     = ops_all[order]
    unit_ids    = unit_ids[order]
    cycle_ids   = cycle_ids[order]

    ds = NCMAPSSDataset.__new__(NCMAPSSDataset)
    ds.hdf5_path = Path("synthetic")
    ds.split = "dev"
    ds.window_size = window_size
    ds.stride = stride
    ds.use_virtual_sensors = False
    ds.return_theta_true = return_theta_true
    ds.return_ops = return_ops
    ds.dtype = torch.float32

    ds._sensors     = sensors_all
    ds._ops         = ops_all
    ds._rul         = rul_all
    ds._theta       = theta_all
    ds._unit_id_arr = unit_ids
    ds._cycle_arr   = cycle_ids

    cumsum = 0
    ds._unit_ranges = {}
    unique_ids, counts = np.unique(unit_ids, return_counts=True)
    for uid, cnt in zip(unique_ids.tolist(), counts.tolist()):
        ds._unit_ranges[uid] = (cumsum, cumsum + cnt)
        cumsum += cnt

    ds._index_list = []
    for uid in sorted(ds._unit_ranges.keys()):
        g_start, g_end = ds._unit_ranges[uid]
        n_steps = g_end - g_start
        for w in range(0, n_steps - window_size + 1, stride):
            ds._index_list.append((uid, w))

    return ds


# ---------------------------------------------------------------------------
# Tests: unit splitting
# ---------------------------------------------------------------------------

def test_unit_split_no_overlap():
    """Train / val / test unit sets must be disjoint."""
    unit_ids = np.arange(1, 11)   # 10 units
    splits = make_unit_splits(unit_ids, val_frac=0.2, test_frac=0.2, seed=0)

    train_set = set(splits["train"])
    val_set   = set(splits["val"])
    test_set  = set(splits["test"])

    assert train_set.isdisjoint(val_set),  "train ∩ val must be empty"
    assert train_set.isdisjoint(test_set), "train ∩ test must be empty"
    assert val_set.isdisjoint(test_set),   "val ∩ test must be empty"
    assert train_set | val_set | test_set == set(unit_ids.tolist()), \
        "All units must be covered"


def test_unit_split_deterministic():
    """Same seed -> identical splits."""
    ids = np.arange(1, 7)
    s1 = make_unit_splits(ids, seed=99)
    s2 = make_unit_splits(ids, seed=99)
    assert s1 == s2


def test_unit_split_different_seeds():
    """Different seeds should (likely) produce different splits."""
    ids = np.arange(1, 13)
    s1 = make_unit_splits(ids, seed=0)
    s2 = make_unit_splits(ids, seed=1)
    # With 12 units it's astronomically unlikely that both are identical
    assert s1 != s2


def test_save_load_splits_roundtrip():
    ids = np.arange(1, 8)
    splits = make_unit_splits(ids, seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        save_splits(tmp, splits)
        assert splits_exist(tmp)
        loaded = load_splits(tmp)
    assert loaded == splits


# ---------------------------------------------------------------------------
# Tests: windows do not cross unit boundaries
# ---------------------------------------------------------------------------

def test_dataset_windows_do_not_cross_units():
    """Every window must belong to a single unit only."""
    unit_counts = {1: 20, 2: 15, 3: 25}
    ds = _make_synthetic_dataset(unit_counts, window_size=5)

    for idx in range(len(ds)):
        uid, w_start = ds._index_list[idx]
        g_start = ds._unit_ranges[uid][0] + w_start
        g_end   = g_start + ds.window_size

        # All rows in the window must belong to uid
        window_units = ds._unit_id_arr[g_start:g_end]
        assert np.all(window_units == uid), (
            f"Window {idx} crosses unit boundary: uid={uid}, units={window_units}"
        )


def test_dataset_correct_window_count():
    """Number of windows = sum over units of (n_steps - W + 1)."""
    unit_counts = {1: 10, 2: 8, 3: 12}
    W = 4
    expected = sum(n - W + 1 for n in unit_counts.values())
    ds = _make_synthetic_dataset(unit_counts, window_size=W)
    assert len(ds) == expected


def test_dataset_stride_reduces_windows():
    """Stride > 1 reduces the number of windows."""
    unit_counts = {1: 20}
    ds1 = _make_synthetic_dataset(unit_counts, window_size=5, stride=1)
    ds2 = _make_synthetic_dataset(unit_counts, window_size=5, stride=2)
    assert len(ds2) < len(ds1)


def test_dataset_getitem_shape():
    unit_counts = {1: 15}
    ds = _make_synthetic_dataset(unit_counts, window_size=5, n_features=4)
    x, rul = ds[0]
    assert x.shape == (5, 4)
    assert rul.shape == ()


def test_dataset_getitem_theta_true():
    unit_counts = {1: 15}
    ds = _make_synthetic_dataset(unit_counts, window_size=5, n_health=3, return_theta_true=True)
    x, rul, theta = ds[0]
    assert theta.shape == (3,)


# ---------------------------------------------------------------------------
# Tests: SubsetByUnit
# ---------------------------------------------------------------------------

def test_subset_by_unit_no_leakage():
    """Units assigned to val subset must not appear in train subset."""
    unit_counts = {u: 20 for u in range(1, 7)}
    ds = _make_synthetic_dataset(unit_counts, window_size=4)

    train_units = [1, 2, 3, 4]
    val_units   = [5, 6]

    train_ds = SubsetByUnit(ds, train_units)
    val_ds   = SubsetByUnit(ds, val_units)

    train_uid_set = set(train_ds.unit_ids_array.tolist())
    val_uid_set   = set(val_ds.unit_ids_array.tolist())

    assert train_uid_set == {1, 2, 3, 4}
    assert val_uid_set   == {5, 6}
    assert train_uid_set.isdisjoint(val_uid_set)


def test_subset_len_is_subset_of_base():
    unit_counts = {u: 20 for u in range(1, 5)}
    ds = _make_synthetic_dataset(unit_counts, window_size=4)
    sub = SubsetByUnit(ds, [1, 2])
    assert len(sub) < len(ds)
    assert len(sub) + len(SubsetByUnit(ds, [3, 4])) == len(ds)


# ---------------------------------------------------------------------------
# Tests: per-unit prediction horizon
# ---------------------------------------------------------------------------

def test_prediction_horizon_per_unit_synthetic():
    """prediction_horizon should return non-None for a perfect predictor."""
    # Perfect predictor: pred == target for the entire trajectory
    target = np.linspace(100, 0, 50)
    pred   = target.copy()
    ph = prediction_horizon(pred, target, alpha=0.2)
    assert ph is not None
    assert ph >= 0


def test_prediction_horizon_none_on_bad_predictor():
    """A wildly wrong predictor should return None."""
    target = np.linspace(100, 0, 50)
    pred   = np.full_like(target, 200.0)   # always 200 cycles off
    ph = prediction_horizon(pred, target, alpha=0.05)
    assert ph is None


def test_prediction_horizon_aggregate_per_unit():
    """Simulated multi-unit evaluation: aggregate ph values."""
    n_units = 4
    ph_values = []
    ph_none = 0

    for uid in range(n_units):
        n = 40 + uid * 5
        target = np.linspace(n, 0, n + 1)
        # Slightly noisy predictor
        rng = np.random.default_rng(uid)
        pred = target + rng.normal(0, target * 0.05)
        ph = prediction_horizon(pred, target, alpha=0.2)
        if ph is None:
            ph_none += 1
        else:
            ph_values.append(ph)

    # For a 5 % noise level with alpha=0.2, all units should get a horizon
    assert ph_none == 0, f"Expected 0 None, got {ph_none}"
    assert np.median(ph_values) > 0


def test_unit_ids_array_aligned_with_getitem():
    """unit_ids_array[i] must match the unit of ds[i]."""
    unit_counts = {1: 10, 2: 10, 3: 10}
    ds = _make_synthetic_dataset(unit_counts, window_size=4)
    uids = ds.unit_ids_array
    for i in range(len(ds)):
        uid_meta = uids[i]
        uid_actual, _ = ds._index_list[i]
        assert uid_meta == uid_actual
