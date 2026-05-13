"""Tests for V3.1b thermal-auxiliary loss (`CycleLayerV3Loss`).

Covers:
1. use_epr_loss=True raises at construction
2. use_pressure_loss=True raises at construction
3. forbidden tokens in temp_sensors raise
4. empty temp_sensors raises
5. L_temp uses ONLY T24/T30/T50 keys (Ps30/P30/EPR keys in pred dict raise)
6. L_aux normalisation by sigma_lpt_flow
7. L_healthy_prior only active on RUL > threshold
8. L_smooth on constant theta is zero
9. Total loss is finite + backward produces finite grads
"""

from __future__ import annotations

import pytest
import torch

from cyclelayer.losses import CycleLayerV3Loss, V3LossConfig


# ===========================================================================
# Construction tripwires
# ===========================================================================

def test_epr_loss_forbidden_at_construction():
    with pytest.raises(ValueError, match="EPR"):
        CycleLayerV3Loss(V3LossConfig(use_epr_loss=True))


def test_pressure_loss_forbidden_at_construction():
    with pytest.raises(ValueError, match="[Pp]ressure"):
        CycleLayerV3Loss(V3LossConfig(use_pressure_loss=True))


def test_temp_sensors_must_be_subset_of_T24_T30_T50():
    with pytest.raises(ValueError, match="subset"):
        CycleLayerV3Loss(V3LossConfig(temp_sensors=["T24", "P30"]))


def test_temp_sensors_rejects_forbidden_tokens():
    # "EPR" is not in ALLOWED set anyway, but verifying the explicit token check
    with pytest.raises(ValueError):
        CycleLayerV3Loss(V3LossConfig(temp_sensors=["EPR"]))


def test_temp_sensors_cannot_be_empty():
    with pytest.raises(ValueError, match="empty"):
        CycleLayerV3Loss(V3LossConfig(temp_sensors=[]))


def test_temp_sensors_T48_rejected():
    """T48 is dataset T48 (HPT outlet), not the documented V3.1b L_temp target."""
    with pytest.raises(ValueError, match="subset"):
        CycleLayerV3Loss(V3LossConfig(temp_sensors=["T48"]))


# ===========================================================================
# L_temp behaviour
# ===========================================================================

def test_temp_loss_uses_only_T24_T30_T50():
    """Only configured temp_sensors keys contribute; T48 etc. silently ignored."""
    loss = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=0.0, lambda_temp=1.0, lambda_aux=0.0,
        lambda_healthy=0.0, lambda_smooth=0.0,
        temp_sensors=["T24"],
        sigma_temp_K={"T24": 1.0},
    ))
    B = 4
    rul_pred = torch.zeros(B); rul_true = torch.zeros(B)
    theta = torch.ones(B, 5)
    # Only T24 should affect the loss; T48 must be silently ignored (NOT contribute).
    preds = {
        "T24_K": torch.full((B,), 10.0),
        "T30_K": torch.full((B,), 100.0),    # ignored — not in temp_sensors
        "T48_K": torch.full((B,), 1000.0),   # ignored
    }
    trues = {
        "T24_K": torch.zeros(B),
        "T30_K": torch.zeros(B),
        "T48_K": torch.zeros(B),
    }
    total, comps = loss(rul_pred, rul_true, theta,
                        temp_preds_K=preds, temp_true_K=trues)
    # L_temp = mean( ((10-0)/1)^2 ) = 100; lambda_temp=1.0 → total ≈ 100
    assert abs(total.item() - 100.0) < 1e-4, f"got {total.item()}"
    assert "temp" in comps


def test_temp_loss_rejects_pressure_keys_in_pred_dict():
    """If a P-/EPR-like key sneaks into the temp-preds dict the loss raises."""
    loss = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=0.0, lambda_temp=1.0,
        temp_sensors=["T24"],
    ))
    B = 4
    rul_pred = torch.zeros(B); rul_true = torch.zeros(B)
    theta = torch.ones(B, 5)
    bad_preds = {
        "T24_K":   torch.zeros(B),
        "Ps30_Pa": torch.zeros(B),         # forbidden token
    }
    bad_trues = {
        "T24_K":   torch.zeros(B),
        "Ps30_Pa": torch.zeros(B),
    }
    with pytest.raises(ValueError, match="forbidden"):
        loss(rul_pred, rul_true, theta,
             temp_preds_K=bad_preds, temp_true_K=bad_trues)


def test_temp_loss_uses_sigma_per_sensor():
    """Per-sensor σ scaling is applied."""
    loss = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=0.0, lambda_temp=1.0,
        temp_sensors=["T24"],
        sigma_temp_K={"T24": 10.0},
    ))
    B = 4
    rul_pred = torch.zeros(B); rul_true = torch.zeros(B)
    theta = torch.ones(B, 5)
    preds = {"T24_K": torch.full((B,), 10.0)}
    trues = {"T24_K": torch.zeros(B)}
    total, _ = loss(rul_pred, rul_true, theta,
                    temp_preds_K=preds, temp_true_K=trues)
    # ((10 - 0)/10)^2 = 1
    assert abs(total.item() - 1.0) < 1e-5


# ===========================================================================
# L_aux behaviour
# ===========================================================================

def test_aux_loss_normalised_by_sigma():
    loss = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=0.0, lambda_temp=0.0,
        lambda_aux=1.0, lambda_healthy=0.0, lambda_smooth=0.0,
        sigma_lpt_flow=0.02,
    ))
    B = 4
    rul_pred = torch.zeros(B); rul_true = torch.zeros(B)
    theta = torch.ones(B, 5)
    pred = torch.full((B,), 0.02)   # 1σ off
    true = torch.zeros(B)
    total, comps = loss(rul_pred, rul_true, theta,
                        lpt_flow_pred=pred, lpt_flow_true=true)
    # ((0.02 - 0)/0.02)^2 = 1
    assert abs(total.item() - 1.0) < 1e-5
    assert "aux" in comps


def test_aux_loss_inactive_when_lambda_zero():
    loss = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=0.0, lambda_temp=0.0,
        lambda_aux=0.0, lambda_healthy=0.0, lambda_smooth=0.0,
    ))
    B = 4
    rul_pred = torch.zeros(B); rul_true = torch.zeros(B)
    theta = torch.ones(B, 5)
    pred = torch.full((B,), 1e6); true = torch.zeros(B)
    total, comps = loss(rul_pred, rul_true, theta,
                        lpt_flow_pred=pred, lpt_flow_true=true)
    assert total.item() == 0.0
    assert "aux" not in comps


# ===========================================================================
# L_healthy_prior behaviour
# ===========================================================================

def test_healthy_prior_zero_when_no_high_rul():
    """If no sample has RUL > threshold, L_healthy is zero."""
    loss = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=0.0, lambda_temp=0.0, lambda_aux=0.0,
        lambda_healthy=1.0, lambda_smooth=0.0,
        healthy_rul_threshold=80.0,
    ))
    B = 4
    rul_pred = torch.zeros(B); rul_true = torch.full((B,), 30.0)   # all low-RUL
    theta = torch.full((B, 5), 0.85)   # very degraded — but irrelevant
    total, comps = loss(rul_pred, rul_true, theta)
    assert total.item() == 0.0
    # When mask is empty, _healthy_prior returns 0 but still records the component
    assert comps.get("healthy", torch.tensor(0.0)).item() == 0.0


def test_healthy_prior_penalises_degraded_theta_at_high_rul():
    loss = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=0.0, lambda_temp=0.0, lambda_aux=0.0,
        lambda_healthy=1.0, lambda_smooth=0.0,
        healthy_rul_threshold=80.0,
    ))
    B = 4
    rul_pred = torch.zeros(B)
    rul_true = torch.full((B,), 95.0)   # all high-RUL
    theta = torch.full((B, 5), 0.85)   # max degradation
    total, comps = loss(rul_pred, rul_true, theta)
    # ||0.85 - 1||² = 0.0225 per element; mean over 5 → 0.0225
    assert abs(total.item() - 0.0225) < 1e-5


# ===========================================================================
# L_smooth behaviour
# ===========================================================================

def test_smooth_zero_on_constant_theta_over_time():
    """Constant theta over time produces zero smoothness loss."""
    loss = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=0.0, lambda_temp=0.0, lambda_aux=0.0,
        lambda_healthy=0.0, lambda_smooth=1.0,
    ))
    B, T = 4, 8
    rul_pred = torch.zeros(B); rul_true = torch.zeros(B)
    theta_const = torch.full((B, T, 5), 0.99)
    total, comps = loss(rul_pred, rul_true, theta_const)
    assert total.item() == 0.0


def test_smooth_zero_when_no_time_axis():
    """Per-window theta (B, 5) has no time axis → smoothness 0."""
    loss = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=0.0, lambda_temp=0.0, lambda_aux=0.0,
        lambda_healthy=0.0, lambda_smooth=1.0,
    ))
    B = 4
    rul_pred = torch.zeros(B); rul_true = torch.zeros(B)
    theta = torch.ones(B, 5)
    total, _ = loss(rul_pred, rul_true, theta)
    assert total.item() == 0.0


def test_smooth_nonzero_on_changing_theta():
    """Increasing theta over time produces positive smoothness loss."""
    loss = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=0.0, lambda_temp=0.0, lambda_aux=0.0,
        lambda_healthy=0.0, lambda_smooth=1.0,
    ))
    B, T = 4, 8
    rul_pred = torch.zeros(B); rul_true = torch.zeros(B)
    theta = torch.linspace(0.85, 1.0, T).view(1, T, 1).expand(B, T, 5).contiguous()
    total, _ = loss(rul_pred, rul_true, theta)
    assert total.item() > 0.0


# ===========================================================================
# End-to-end: total, finite, backward
# ===========================================================================

def test_total_loss_finite_and_backward():
    loss = CycleLayerV3Loss(V3LossConfig())   # defaults; all components active
    B, T = 4, 8
    rul_pred = torch.randn(B, requires_grad=True)
    rul_true = torch.full((B,), 60.0)
    theta = torch.full((B, T, 5), 0.95, requires_grad=True)
    pred = torch.full((B,), 0.01, requires_grad=True)
    true = torch.zeros(B)
    preds_K = {
        "T24_K": torch.full((B,), 310.0, requires_grad=True),
        "T30_K": torch.full((B,), 920.0, requires_grad=True),
        "T50_K": torch.full((B,), 860.0, requires_grad=True),
    }
    trues_K = {
        "T24_K": torch.full((B,), 305.0),
        "T30_K": torch.full((B,), 915.0),
        "T50_K": torch.full((B,), 855.0),
    }
    cfg = V3LossConfig(
        sigma_temp_K={"T24": 5.0, "T30": 20.0, "T50": 10.0},
        sigma_lpt_flow=0.02,
    )
    loss = CycleLayerV3Loss(cfg)
    total, comps = loss(
        rul_pred, rul_true, theta,
        lpt_flow_pred=pred, lpt_flow_true=true,
        temp_preds_K=preds_K, temp_true_K=trues_K,
    )
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(rul_pred.grad).all()
    assert torch.isfinite(theta.grad).all()
    assert torch.isfinite(pred.grad).all()
    for v in preds_K.values():
        assert torch.isfinite(v.grad).all()
    expected_components = {"rul", "temp", "aux", "healthy", "smooth"}
    assert expected_components.issubset(set(comps.keys()))
