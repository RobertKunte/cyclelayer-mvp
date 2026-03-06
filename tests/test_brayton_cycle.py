"""Unit tests for the differentiable Brayton cycle layer.

Tests cover:
- Output shape and dtype
- Physically meaningful output ranges
- Gradient flow (differentiability)
- Thermal efficiency of the ideal cycle
- Behaviour with degenerate inputs (clamping stability)
"""

from __future__ import annotations

import pytest
import torch

from cyclelayer.models.brayton_cycle import BraytonCycleLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def layer() -> BraytonCycleLayer:
    return BraytonCycleLayer(n_params=5)


@pytest.fixture
def theta_batch() -> torch.Tensor:
    """Physically plausible batch of cycle parameters."""
    B = 8
    T1    = torch.full((B, 1), 288.0)    # ISA sea-level temperature [K]
    pi    = torch.full((B, 1), 20.0)     # pressure ratio
    T3    = torch.full((B, 1), 1500.0)   # TIT [K]
    eta_c = torch.full((B, 1), 0.87)
    eta_t = torch.full((B, 1), 0.90)
    return torch.cat([T1, pi, T3, eta_c, eta_t], dim=1)


# ---------------------------------------------------------------------------
# Shape & dtype
# ---------------------------------------------------------------------------

def test_output_shape(layer, theta_batch):
    out = layer(theta_batch)
    assert out.shape == (8, layer.out_features), f"Expected (8, {layer.out_features}), got {out.shape}"


def test_output_dtype(layer, theta_batch):
    out = layer(theta_batch)
    assert out.dtype == torch.float32


def test_out_features_property():
    l5  = BraytonCycleLayer(n_params=5)
    l8  = BraytonCycleLayer(n_params=8)
    assert l5.out_features == 8
    assert l8.out_features == 11   # 8 base + 3 extra


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------

def test_thermal_efficiency_positive(layer, theta_batch):
    out = layer(theta_batch)
    eta_th = out[:, 1]   # feature index 1 = η_th
    assert (eta_th > 0).all(), "Thermal efficiency should be positive."


def test_thermal_efficiency_less_than_one(layer, theta_batch):
    out = layer(theta_batch)
    eta_th = out[:, 1]
    assert (eta_th < 1).all(), "Thermal efficiency must be < 1 (2nd law)."


def test_net_work_positive(layer, theta_batch):
    out = layer(theta_batch)
    W_net_norm = out[:, 0]
    assert (W_net_norm > 0).all(), "Net specific work should be positive for a running engine."


def test_back_work_ratio_between_zero_and_one(layer, theta_batch):
    out = layer(theta_batch)
    r_bw = out[:, 2]
    assert (r_bw > 0).all() and (r_bw < 1).all(), "Back-work ratio must be in (0, 1)."


# ---------------------------------------------------------------------------
# Ideal cycle: closed-form thermal efficiency
# ---------------------------------------------------------------------------

def test_ideal_cycle_efficiency():
    """For η_c = η_t = 1, η_th should match the Carnot-like formula:
       η_th_ideal = 1 - π^(-(γ-1)/γ)
    """
    gamma = 1.4
    layer = BraytonCycleLayer(n_params=5, gamma=gamma)
    pi_val = 20.0
    exp = (gamma - 1) / gamma
    eta_th_ideal = 1.0 - pi_val ** (-exp)

    theta = torch.tensor([[288.0, pi_val, 1500.0, 1.0, 1.0]])
    out = layer(theta)
    eta_th = out[0, 1].item()

    assert abs(eta_th - eta_th_ideal) < 1e-4, (
        f"Ideal η_th mismatch: expected {eta_th_ideal:.6f}, got {eta_th:.6f}"
    )


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_gradients_flow(layer):
    theta = torch.tensor(
        [[288.0, 20.0, 1500.0, 0.87, 0.90]],
        requires_grad=True,
    )
    out = layer(theta)
    loss = out.sum()
    loss.backward()
    assert theta.grad is not None
    assert not torch.isnan(theta.grad).any(), "NaN gradients detected."
    assert not torch.isinf(theta.grad).any(), "Inf gradients detected."


# ---------------------------------------------------------------------------
# Stability with degenerate inputs
# ---------------------------------------------------------------------------

def test_stability_extreme_inputs():
    """Layer should not produce NaN/Inf for extreme but clamped inputs."""
    layer = BraytonCycleLayer(n_params=5, clamp_temperatures=True)
    theta = torch.tensor([[1.0, 0.01, 10.0, 0.01, 0.01]])   # physically invalid
    out = layer(theta)
    assert not torch.isnan(out).any(), "NaN output for extreme inputs."
    assert not torch.isinf(out).any(), "Inf output for extreme inputs."


def test_extra_params_passed_through():
    layer = BraytonCycleLayer(n_params=7)
    theta = torch.cat(
        [torch.tensor([[288.0, 20.0, 1500.0, 0.87, 0.90]]),
         torch.rand(1, 2)],
        dim=1,
    )
    out = layer(theta)
    assert out.shape[-1] == layer.out_features
