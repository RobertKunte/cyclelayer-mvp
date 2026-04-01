"""Pipeline guard tests — no HDF5 file required.

Covers:
    1. stride_eval respected: stride=N yields fewer windows than stride=1.
    2. Theta scaler no leakage: scaler fit on train rows only, not val/test rows.
    3. Theta scaler determinism: same train units → same mean/std across calls.
    4. Theta scaler raises on empty train mask.
    5. predictions.csv structure: verify expected columns are present.
    6. Ops: return_ops separates sensors from W; shapes and scaler correct.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Re-use the synthetic dataset factory from test_dataset.py
from tests.test_dataset import _make_synthetic_dataset
from cyclelayer.data.preprocessing import StandardScaler, fit_ops_scaler, fit_sensor_scaler, fit_theta_scaler


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
# 5. Sensor scaler no leakage
# ---------------------------------------------------------------------------

def test_sensor_scaler_only_uses_train_rows():
    """Sensor scaler fitted on train units must NOT include val/test stats."""
    # Sensor values differ per unit: unit u has sensors = u (constant for all features)
    unit_counts = {1: 30, 2: 30, 3: 30}
    ds = _make_synthetic_dataset(unit_counts, window_size=5, n_features=4)

    # Verify sensors are unit-specific: row sensor value = uid + t*0.01 ≈ uid for small t
    scaler = fit_sensor_scaler(ds, train_units=[1])
    # Unit 1 sensors start at 1.0, so mean should be ≈ 1.15 (1.0 + 29*0.01/2)
    # The mean of unit 2 rows is ≈ 2.15, unit 3 ≈ 3.15
    # Global mean of all 3 units would be ≈ 2.15 — clearly different from unit-1-only mean
    global_mean_approx = 2.15
    assert not np.allclose(scaler.mean_, global_mean_approx, atol=0.2), (
        "Sensor scaler mean matches global mean — val/test rows may have leaked"
    )
    # Unit-1-only mean should be close to 1.15
    np.testing.assert_allclose(scaler.mean_, 1.145, atol=0.05)


def test_sensor_scaler_raises_on_empty_train_mask():
    """fit_sensor_scaler must raise ValueError when no rows match train_units."""
    ds = _make_synthetic_dataset({1: 20, 2: 20}, window_size=5, n_features=4)
    with pytest.raises(ValueError, match="No rows found"):
        fit_sensor_scaler(ds, train_units=[99])


def test_sensor_scaler_deterministic():
    """Calling fit_sensor_scaler twice with the same units yields identical stats."""
    ds = _make_synthetic_dataset({1: 30, 2: 30, 3: 30}, window_size=5, n_features=4)
    s1 = fit_sensor_scaler(ds, [1, 2])
    s2 = fit_sensor_scaler(ds, [1, 2])
    np.testing.assert_array_equal(s1.mean_, s2.mean_)
    np.testing.assert_array_equal(s1.std_, s2.std_)


def test_sensor_scaler_transform_normalises_train_mean():
    """After transform, train rows should have near-zero mean."""
    ds = _make_synthetic_dataset({1: 50, 2: 50}, window_size=5, n_features=4)
    scaler = fit_sensor_scaler(ds, train_units=[1, 2])
    normed = scaler.transform(ds._sensors)
    # All rows are train rows here; after standardization mean ≈ 0
    np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. predictions.csv expected columns
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


# ---------------------------------------------------------------------------
# 6. Operating conditions (ops) — return_ops, _ops field, ops scaler
# ---------------------------------------------------------------------------

def test_return_ops_false_sensor_shape_unchanged():
    """With return_ops=False (default), _sensors keeps all n_features columns."""
    n_features = 8
    ds = _make_synthetic_dataset({1: 30, 2: 20}, window_size=5, n_features=n_features)
    assert ds.n_features == n_features, (
        f"return_ops=False: expected n_features={n_features}, got {ds.n_features}"
    )
    assert ds.ops_dim == 0, "ops_dim must be 0 when return_ops=False"


def test_return_ops_true_sensor_shape_reduced():
    """With return_ops=True, _sensors = X_s only (n_features cols, not n_features+4).

    The synthetic helper stores n_features sensor cols; _ops is always n_ops cols.
    With return_ops=True, ops_dim > 0.
    """
    n_features = 6
    ds = _make_synthetic_dataset(
        {1: 30, 2: 20}, window_size=5, n_features=n_features, return_ops=True
    )
    assert ds.n_features == n_features, (
        f"return_ops=True: expected n_features={n_features}, got {ds.n_features}"
    )
    assert ds.ops_dim == 4, f"ops_dim must be 4 when return_ops=True, got {ds.ops_dim}"


def test_return_ops_getitem_returns_3tuple():
    """__getitem__ returns (x, rul, ops) 3-tuple when return_ops=True."""
    import torch
    ds = _make_synthetic_dataset(
        {1: 20}, window_size=5, n_features=6, return_ops=True
    )
    sample = ds[0]
    assert len(sample) == 3, f"Expected 3-tuple, got {len(sample)}-tuple"
    x, rul, ops = sample
    assert x.shape == (5, 6),  f"x shape: expected (5,6), got {x.shape}"
    assert ops.shape == (5, 4), f"ops shape: expected (5,4), got {ops.shape}"
    assert rul.ndim == 0, "rul must be scalar tensor"


def test_ops_stored_separately_from_sensors():
    """_ops and _sensors are distinct arrays; no column overlap."""
    n_features, n_ops = 6, 4
    ds = _make_synthetic_dataset(
        {1: 30}, window_size=5, n_features=n_features, n_ops=n_ops
    )
    assert ds._sensors.shape[1] == n_features, (
        f"_sensors cols: expected {n_features}, got {ds._sensors.shape[1]}"
    )
    assert ds._ops.shape[1] == n_ops, (
        f"_ops cols: expected {n_ops}, got {ds._ops.shape[1]}"
    )


def test_ops_scaler_no_leakage():
    """Ops scaler fitted on train units must NOT include val/test stats."""
    # Unit u has ops = [u*1000, u*0.1, u*10, u*100]
    unit_counts = {1: 30, 2: 30, 3: 30}
    ds = _make_synthetic_dataset(unit_counts, window_size=5)

    scaler = fit_ops_scaler(ds, train_units=[1])
    # Unit-1 ops: col 0 = 1000.0 for all rows → mean col0 should be ≈ 1000
    np.testing.assert_allclose(scaler.mean_[0], 1000.0, atol=1e-3,
                                err_msg="ops scaler mean[0] should reflect only unit-1 rows")
    # If val units leaked, col0 mean would be (1000+2000+3000)/3 = 2000
    assert not np.isclose(scaler.mean_[0], 2000.0, atol=100.0), (
        "Ops scaler mean matches global mean — val/test rows may have leaked"
    )


def test_ops_scaler_raises_on_empty():
    """fit_ops_scaler must raise ValueError when no rows match train_units."""
    ds = _make_synthetic_dataset({1: 20, 2: 20}, window_size=5)
    with pytest.raises(ValueError, match="No rows found"):
        fit_ops_scaler(ds, train_units=[99])


# ---------------------------------------------------------------------------
# 7. OpsEncoder + dual-encoder architecture guard tests
# ---------------------------------------------------------------------------

def test_ops_encoder_shape():
    """OpsEncoder maps (B, T, ops_dim) -> (B, out_dim)."""
    import torch
    from cyclelayer.models.encoder import OpsEncoder

    enc = OpsEncoder(ops_dim=4, channels=[16, 32], out_dim=32)
    enc.eval()
    with torch.no_grad():
        out = enc(torch.randn(2, 30, 4))
    assert out.shape == (2, 32), f"Expected (2, 32), got {out.shape}"


def test_cnn_baseline_dual_encoder_shape():
    """CNNBaseline with ops_dim + fusion runs forward and returns correct shape."""
    import torch
    from cyclelayer.models.baselines import CNNBaseline

    model = CNNBaseline(
        n_features=14,
        ops_dim=4,
        ops_enc_channels=[16, 32],
        ops_enc_out_dim=32,
        fusion_hidden_dim=64,
    )
    model.eval()
    with torch.no_grad():
        pred = model(torch.randn(2, 30, 14), ops=torch.randn(2, 30, 4))
    assert pred.shape == (2,), f"Expected (2,), got {pred.shape}"


def test_no_ops_backward_compatible():
    """CNNBaseline with ops_dim=0 must produce identical output regardless of
    whether new params are passed explicitly or left at defaults."""
    import torch
    from cyclelayer.models.baselines import CNNBaseline

    torch.manual_seed(0)
    model_default = CNNBaseline(n_features=18)          # ops_dim=0 implicitly
    torch.manual_seed(0)
    model_explicit = CNNBaseline(n_features=18, ops_dim=0,
                                 ops_enc_channels=[16, 32],
                                 ops_enc_out_dim=32,
                                 fusion_hidden_dim=64)  # ignored when ops_dim=0

    # Both should have identical parameter counts (ops path inactive)
    params_default  = sum(p.numel() for p in model_default.parameters())
    params_explicit = sum(p.numel() for p in model_explicit.parameters())
    assert params_default == params_explicit, (
        f"ops_dim=0 should produce same param count: {params_default} vs {params_explicit}"
    )

    # Copy weights and verify identical forward pass
    model_explicit.load_state_dict(model_default.state_dict())
    x = torch.randn(2, 30, 18)
    model_default.eval()
    model_explicit.eval()
    with torch.no_grad():
        assert torch.allclose(model_default(x), model_explicit(x)), (
            "ops_dim=0 forward outputs differ — backward compatibility broken"
        )


# ---------------------------------------------------------------------------
# 8. OpsResidualNet (cyclelayer_v2) guard tests
# ---------------------------------------------------------------------------

def test_ops_to_sensor_baseline_shape():
    """OpsToSensorBaseline maps (B, T, ops_dim) -> (B, T, n_sensors)."""
    import torch
    from cyclelayer.models.encoder import OpsToSensorBaseline

    baseline = OpsToSensorBaseline(ops_dim=4, n_sensors=14, hidden_dim=32, n_layers=2)
    baseline.eval()
    with torch.no_grad():
        out = baseline(torch.randn(2, 30, 4))
    assert out.shape == (2, 30, 14), f"Expected (2,30,14), got {out.shape}"


def test_ops_to_sensor_baseline_near_zero_init():
    """OpsToSensorBaseline output must be near zero at initialization."""
    import torch
    from cyclelayer.models.encoder import OpsToSensorBaseline

    torch.manual_seed(42)
    baseline = OpsToSensorBaseline(ops_dim=4, n_sensors=14, hidden_dim=32, n_layers=2)
    baseline.eval()
    with torch.no_grad():
        out = baseline(torch.randn(8, 30, 4))
    abs_mean = out.abs().mean().item()
    assert abs_mean < 0.5, (
        f"Baseline output abs mean {abs_mean:.4f} too large at init — "
        "near-zero init not applied correctly"
    )


def test_ops_residual_net_forward_shape():
    """OpsResidualNet forward returns (rul, h_deg) with correct shapes."""
    import torch
    from cyclelayer.models.physresnet import OpsResidualNet, OpsResidualNetConfig

    cfg = OpsResidualNetConfig(n_features=14, ops_dim=4, window_size=30, n_health_params=10)
    model = OpsResidualNet(cfg)
    model.eval()
    with torch.no_grad():
        rul, h_deg = model(torch.randn(2, 30, 14), torch.randn(2, 30, 4))
    assert rul.shape   == (2,),    f"rul shape: expected (2,), got {rul.shape}"
    assert h_deg.shape == (2, 10), f"h_deg shape: expected (2,10), got {h_deg.shape}"


def test_ops_residual_net_x_ref_stored():
    """forward() must store _x_ref with correct shape (B, T, n_sensors)."""
    import torch
    from cyclelayer.models.physresnet import OpsResidualNet, OpsResidualNetConfig

    cfg = OpsResidualNetConfig(n_features=14, ops_dim=4, window_size=30)
    model = OpsResidualNet(cfg)
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 30, 14), torch.randn(2, 30, 4))
    assert model._x_ref is not None, "_x_ref must be set after forward()"
    assert model._x_ref.shape == (2, 30, 14), (
        f"_x_ref shape: expected (2,30,14), got {model._x_ref.shape}"
    )


def test_ops_residual_net_backward():
    """Gradients must flow through the full OpsResidualNet graph."""
    import torch
    from cyclelayer.models.physresnet import OpsResidualNet, OpsResidualNetConfig

    cfg = OpsResidualNetConfig(n_features=14, ops_dim=4, window_size=30, n_health_params=10)
    model = OpsResidualNet(cfg)
    model.train()
    x   = torch.randn(2, 30, 14, requires_grad=False)
    ops = torch.randn(2, 30, 4,  requires_grad=False)
    rul, h_deg = model(x, ops)
    loss = rul.mean() + h_deg.mean()
    loss.backward()  # must not raise
    # At least one parameter must have a gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found after backward()"


def test_baseline_smoothness_loss_nonnegative():
    """CompositeLoss smoothness term must be >= 0."""
    import torch
    from cyclelayer.training.losses import CompositeLoss, RULLoss

    crit = CompositeLoss(
        rul_loss=RULLoss(),
        lambda_theta=0.1,
        baseline_smoothness_weight=1.0,
    )
    B, T, F = 4, 30, 14
    rul_pred   = torch.randn(B)
    rul_true   = torch.randn(B)
    theta_hat  = torch.randn(B, 10)
    theta_true = torch.randn(B, 10)
    x_ref      = torch.randn(B, T, F)
    total, comps = crit(rul_pred, rul_true, theta_hat, theta_true, x_ref=x_ref)
    assert "smooth" in comps, "smoothness component missing from loss dict"
    assert comps["smooth"].item() >= 0.0, "smoothness loss must be non-negative"


def test_forward_aux_returns_expected_keys():
    """forward_aux must return dict with keys: x_ref, x_res, h_deg, z_ops, rul."""
    import torch
    from cyclelayer.models.physresnet import OpsResidualNet, OpsResidualNetConfig

    cfg = OpsResidualNetConfig(n_features=14, ops_dim=4, window_size=30)
    model = OpsResidualNet(cfg)
    model.eval()
    with torch.no_grad():
        aux = model.forward_aux(torch.randn(2, 30, 14), torch.randn(2, 30, 4))
    expected_keys = {"x_ref", "x_res", "h_deg", "z_ops", "rul"}
    assert expected_keys == set(aux.keys()), (
        f"forward_aux keys mismatch: got {set(aux.keys())}"
    )
    assert aux["x_ref"].shape == (2, 30, 14)
    assert aux["x_res"].shape == (2, 30, 14)
    assert aux["h_deg"].shape == (2, 10)
    assert aux["z_ops"].shape == (2, 32)
    assert aux["rul"].shape   == (2,)
