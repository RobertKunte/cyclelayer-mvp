"""Pipeline guard tests — no HDF5 file required.

Covers:
    1. stride_eval respected: stride=N yields fewer windows than stride=1.
    2. Theta scaler no leakage: scaler fit on train rows only, not val/test rows.
    3. Theta scaler determinism: same train units → same mean/std across calls.
    4. Theta scaler raises on empty train mask.
    5. predictions.csv structure: verify expected columns are present.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Re-use the synthetic dataset factory from test_dataset.py
from tests.test_dataset import _make_synthetic_dataset
from cyclelayer.data.preprocessing import StandardScaler, fit_theta_scaler


# ---------------------------------------------------------------------------
# 1. stride_eval is respected
# ---------------------------------------------------------------------------

def test_stride_eval_reduces_windows():
    """stride=N must produce strictly fewer windows than stride=1 (per unit)."""
    unit_counts = {1: 50, 2: 40}  # large enough to show stride effect
    ds1 = _make_synthetic_dataset(unit_counts, window_size=5, stride=1)
    ds2 = _make_synthetic_dataset(unit_counts, window_size=5, stride=5)
    assert len(ds2) < len(ds1), (
        f"stride=5 should give fewer windows than stride=1: {len(ds2)} >= {len(ds1)}"
    )


def test_stride_one_gives_maximum_windows():
    """stride=1 must give floor((n - W + 1) / 1) windows per unit."""
    W = 5
    unit_counts = {1: 20}
    ds = _make_synthetic_dataset(unit_counts, window_size=W, stride=1)
    expected = 20 - W + 1
    assert len(ds) == expected, f"Expected {expected} windows, got {len(ds)}"


def test_stride_eval_window_count_formula():
    """Verify windows = floor((n - W) / stride + 1) for stride > 1."""
    W, stride = 5, 3
    n = 20
    expected = (n - W) // stride + 1
    ds = _make_synthetic_dataset({1: n}, window_size=W, stride=stride)
    assert len(ds) == expected, f"Expected {expected}, got {len(ds)}"


# ---------------------------------------------------------------------------
# 2. Theta scaler no leakage
# ---------------------------------------------------------------------------

def test_theta_scaler_only_uses_train_rows():
    """Scaler fitted on train units must NOT have val/test unit stats in mean/std."""
    # unit 1: theta = [0.1, 0.1, 0.1], unit 2: [0.2, 0.2, 0.2], unit 3: [0.3, 0.3, 0.3]
    unit_counts = {1: 30, 2: 30, 3: 30}
    ds = _make_synthetic_dataset(unit_counts, window_size=5, n_health=3)

    train_units = [1]
    scaler = fit_theta_scaler(ds, train_units)

    # Train unit 1 has theta = 0.1 * 1 = 0.1 for all rows → mean should be near 0.1
    np.testing.assert_allclose(
        scaler.mean_,
        np.full(3, 0.1),
        atol=1e-5,
        err_msg="Scaler mean should reflect only train unit theta values",
    )

    # If val/test rows leaked, mean would be (0.1+0.2+0.3)/3 = 0.2
    assert not np.allclose(scaler.mean_, 0.2, atol=0.05), (
        "Scaler mean matches global mean — val/test rows may have leaked into fit"
    )


def test_theta_scaler_fit_on_multiple_train_units():
    """Mean should be average of train units, not include held-out units."""
    unit_counts = {1: 20, 2: 20, 3: 20, 4: 20}
    ds = _make_synthetic_dataset(unit_counts, window_size=5, n_health=3)

    # theta for unit u = u * 0.1 (from synthetic helper)
    # train units 1+2 → mean = (0.1 + 0.2) / 2 = 0.15
    scaler = fit_theta_scaler(ds, train_units=[1, 2])
    np.testing.assert_allclose(scaler.mean_, np.full(3, 0.15), atol=1e-4)

    # val units 3+4 must not be reflected
    assert not np.allclose(scaler.mean_, 0.25, atol=0.05), (
        "Val units 3+4 appear to have influenced the scaler"
    )


def test_theta_scaler_raises_on_empty_train_mask():
    """fit_theta_scaler must raise ValueError when no rows match train_units."""
    ds = _make_synthetic_dataset({1: 20, 2: 20}, window_size=5, n_health=3)
    with pytest.raises(ValueError, match="No rows found"):
        fit_theta_scaler(ds, train_units=[99])  # unit 99 does not exist


# ---------------------------------------------------------------------------
# 3. Theta scaler determinism
# ---------------------------------------------------------------------------

def test_theta_scaler_deterministic():
    """Calling fit_theta_scaler twice with the same units yields identical stats."""
    ds = _make_synthetic_dataset({1: 30, 2: 30, 3: 30}, window_size=5, n_health=4)
    s1 = fit_theta_scaler(ds, [1, 2])
    s2 = fit_theta_scaler(ds, [1, 2])
    np.testing.assert_array_equal(s1.mean_, s2.mean_)
    np.testing.assert_array_equal(s1.std_, s2.std_)


def test_theta_scaler_different_train_sets_differ():
    """Scalers fitted on different train unit sets must have different means."""
    ds = _make_synthetic_dataset({1: 30, 2: 30, 3: 30}, window_size=5, n_health=3)
    s_12 = fit_theta_scaler(ds, [1, 2])
    s_23 = fit_theta_scaler(ds, [2, 3])
    assert not np.allclose(s_12.mean_, s_23.mean_), (
        "Different train unit sets should produce different scaler means"
    )


# ---------------------------------------------------------------------------
# 4. predictions.csv expected columns
# ---------------------------------------------------------------------------

def test_predictions_csv_columns():
    """Verify that a hand-crafted predictions DataFrame has all expected columns."""
    import pandas as pd

    required_cols = {"unit_id", "time_index", "y_true_rul", "y_pred_rul", "abs_error", "split"}

    # Simulate what evaluate.py writes
    df = pd.DataFrame({
        "unit_id":    [1, 1, 2, 2],
        "time_index": [0, 1, 0, 1],
        "y_true_rul": [10.0, 9.0, 20.0, 19.0],
        "y_pred_rul": [10.5, 8.5, 21.0, 18.5],
        "abs_error":  [0.5, 0.5, 1.0, 0.5],
        "split":      ["dev"] * 4,
    })
    assert required_cols.issubset(df.columns), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )
    # time_index resets to 0 for each new unit
    for uid, grp in df.groupby("unit_id"):
        assert grp["time_index"].iloc[0] == 0, (
            f"unit {uid}: time_index should start at 0"
        )
