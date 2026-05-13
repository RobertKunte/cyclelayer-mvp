"""Integration tests for CycleLayerV3 (V3.1b thermal-aux).

Covers:
1. forward shape
2. no NaN on synthetic batch
3. theta in [0.85, 1.00]
4. initial theta mean near 0.99
5. AuxHead initial output near 0.0
6. target-sensor masking after normalization sets masked entries to 0.0
7. AuxHead detach to RUL behaviour
8. theta detach to RUL behaviour
9. backward pass produces finite gradients
10. construction asserts use_pressure_loss=False, use_epr_loss=False
"""

from __future__ import annotations

import pytest
import torch

from cyclelayer.models.brayton_engine import (
    BraytonEngine,
    BraytonEngineConfig,
    InletFlowParams,
    MapCoefficients,
)
from cyclelayer.models.cyclelayer_v3 import (
    AuxHealthHead,
    CycleLayerV3,
    CycleLayerV3Config,
    ParamHeadPhys,
    TargetSensorMask,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _stable_brayton_engine() -> BraytonEngine:
    """D-like thermal-stable engine matching configs/cyclelayer_v3_thermal_aux.yaml."""
    return BraytonEngine(BraytonEngineConfig(
        inlet_flow=InletFlowParams(
            Wc_fan_design=900.0,
            Nc_fan_design=2388.1,
            Wc_min=100.0, Wc_max=1100.0,
        ),
        map_coeffs=MapCoefficients(
            Nc_design_fan=2388.1, Wc_design_fan=900.0,
            Nc_design_lpc=2388.1, Wc_design_lpc=93.06,
            Nc_design_hpc=7529.45, Wc_design_hpc=51.88,
            PR_design_fan=1.6, PR_design_lpc=2.0, PR_design_hpc=12.0,
            eta_design_fan=0.92, eta_design_lpc=0.90, eta_design_hpc=0.88,
        ),
        use_measured_inlet=True,
        bpr_design=5.5,
        eta_design_hpt=0.90,
        eta_design_lpt=0.92,
    ))


def _v3_config(**overrides) -> CycleLayerV3Config:
    base = dict(
        n_sensors=14, ops_dim=4, window_size=30,
        encoder_channels=[16, 32],
        encoder_kernel_size=3,
        encoder_mlp_hidden=64,
        encoder_dropout=0.0,
        encoder_out_dim=32,
        use_ops_encoder=True,
        ops_channels=[16, 16],
        ops_out_dim=16,
        param_hidden=[32, 16],
        aux_hidden=[16],
        prog_hidden=[32, 16],
        prog_dropout=0.0,
        max_rul=99.0,
    )
    base.update(overrides)
    return CycleLayerV3Config(**base)


def _random_batch(B: int = 4, T: int = 30, n_sensors: int = 14, ops_dim: int = 4):
    sensors_norm = torch.randn(B, T, n_sensors)
    ops_norm     = torch.randn(B, T, ops_dim)
    # Plausible last-timestep SI ops/sens for BraytonEngine
    ops_si = {
        "T2_K":  torch.full((B,), 261.0),
        "P2_Pa": torch.full((B,), 55000.0),
        "alt_m": torch.full((B,), 7000.0),
        "mach":  torch.full((B,), 0.63),
    }
    sens_si = {
        "Nf_rpm": torch.full((B,), 2020.0),
        "Nc_rpm": torch.full((B,), 8200.0),
        "Wf_kgs": torch.full((B,), 1.0),
    }
    return sensors_norm, ops_norm, ops_si, sens_si


# ===========================================================================
# Tripwire tests — pressure/EPR forbidden
# ===========================================================================

def test_forbid_use_epr_loss_at_construction():
    """V3.1b: use_epr_loss=True must raise at construction."""
    cfg = _v3_config(use_epr_loss=True)
    with pytest.raises(ValueError, match=r"(?i)epr"):
        CycleLayerV3(cfg)


def test_forbid_use_pressure_loss_at_construction():
    """V3.1b: use_pressure_loss=True must raise at construction."""
    cfg = _v3_config(use_pressure_loss=True)
    with pytest.raises(ValueError, match=r"(?i)pressure"):
        CycleLayerV3(cfg)


# ===========================================================================
# Head bias-init tests
# ===========================================================================

def test_param_head_initial_theta_near_099():
    """ParamHeadPhys: initial theta mean must be near 0.99."""
    head = ParamHeadPhys(in_dim=32, hidden=[16],
                         theta_dim=5, bounds=(0.85, 1.00),
                         initial_theta_target=0.99)
    head.eval()
    with torch.no_grad():
        theta = head(torch.randn(64, 32))
    mean = theta.mean().item()
    assert abs(mean - 0.99) < 1e-3, f"initial theta mean {mean:.6f} not near 0.99"


def test_param_head_outputs_in_bounds():
    """ParamHeadPhys: outputs always in [theta_min, theta_max]."""
    head = ParamHeadPhys(in_dim=32, hidden=[16, 8],
                         theta_dim=5, bounds=(0.85, 1.00),
                         initial_theta_target=0.99)
    # Strong perturbation; large random input
    with torch.no_grad():
        theta = head(torch.randn(64, 32) * 10.0)
    assert (theta >= 0.85).all(), f"theta below lower bound: min={theta.min()}"
    assert (theta <= 1.00).all(), f"theta above upper bound: max={theta.max()}"


def test_aux_head_initial_near_zero():
    """AuxHealthHead: initial output mean must be near 0.0."""
    head = AuxHealthHead(in_dim=32, hidden=[8],
                         bounds=(-0.05, 0.02),
                         initial_value_target=0.0)
    head.eval()
    with torch.no_grad():
        out = head(torch.randn(64, 32))
    mean = out.mean().item()
    assert abs(mean - 0.0) < 1e-3, f"initial aux mean {mean:.6f} not near 0.0"


def test_aux_head_outputs_in_bounds():
    head = AuxHealthHead(in_dim=32, hidden=[8],
                         bounds=(-0.05, 0.02), initial_value_target=0.0)
    with torch.no_grad():
        out = head(torch.randn(64, 32) * 10.0)
    assert (out >= -0.05).all() and (out <= 0.02).all()


# ===========================================================================
# Target-sensor masking
# ===========================================================================

def test_target_sensor_mask_after_normalization_zero():
    """Masked target-sensor entries must become 0.0 in normalized space."""
    mask = TargetSensorMask(target_indices=[0, 1, 3], mask_prob=1.0)
    mask.train()
    x = torch.randn(64, 30, 14)
    y = mask(x)
    # With mask_prob=1.0 every sample is masked at indices 0, 1, 3
    assert torch.all(y[:, :, 0] == 0.0)
    assert torch.all(y[:, :, 1] == 0.0)
    assert torch.all(y[:, :, 3] == 0.0)
    # Non-target columns are untouched
    assert torch.equal(y[:, :, 2], x[:, :, 2])
    for idx in range(4, 14):
        assert torch.equal(y[:, :, idx], x[:, :, idx])


def test_target_sensor_mask_disabled_in_eval():
    """In eval() mode no masking occurs."""
    mask = TargetSensorMask(target_indices=[0, 1, 3], mask_prob=1.0)
    mask.eval()
    x = torch.randn(8, 30, 14)
    y = mask(x)
    assert torch.equal(y, x)


def test_target_sensor_mask_zero_prob_no_op():
    """mask_prob = 0.0 leaves tensor unchanged."""
    mask = TargetSensorMask(target_indices=[0, 1, 3], mask_prob=0.0)
    mask.train()
    x = torch.randn(8, 30, 14)
    y = mask(x)
    assert torch.equal(y, x)


def test_target_sensor_mask_independent_per_sample():
    """With p=0.5 over many samples, masked fraction per index ≈ 0.5 ± 5%."""
    torch.manual_seed(0)
    mask = TargetSensorMask(target_indices=[0], mask_prob=0.5)
    mask.train()
    B = 2000
    x = torch.ones(B, 5, 14)
    y = mask(x)
    masked_frac = (y[:, 0, 0] == 0.0).float().mean().item()
    assert 0.45 < masked_frac < 0.55, f"masking frac {masked_frac:.3f} far from 0.5"


# ===========================================================================
# Forward / shape / no-NaN tests
# ===========================================================================

def test_v3_forward_shape_no_brayton():
    """Forward without BraytonEngine returns rul, theta, aux with correct shapes."""
    model = CycleLayerV3(_v3_config())
    model.eval()
    s, o, _, _ = _random_batch(B=4, T=30)
    out = model(s, o)
    assert out["rul"].shape == (4,)
    assert out["theta_phys"].shape == (4, 5)
    assert out["lpt_flow_pred"].shape == (4,)
    assert "brayton" not in out


def test_v3_forward_shape_with_brayton():
    """Forward WITH BraytonEngine attaches brayton outputs."""
    model = CycleLayerV3(_v3_config(), brayton_engine=_stable_brayton_engine())
    model.eval()
    s, o, ops_si, sens_si = _random_batch(B=4, T=30)
    out = model(s, o, ops_si=ops_si, sens_si=sens_si)
    assert "brayton" in out
    spreds = out["brayton"]["sensors_pred_si"]
    assert set(spreds.keys()) == {"T24_K", "T30_K", "P30_Pa", "T50_K"}
    for k, v in spreds.items():
        assert v.shape == (4,)


def test_v3_forward_no_nan():
    """No NaN / Inf in any output on a synthetic batch."""
    torch.manual_seed(7)
    model = CycleLayerV3(_v3_config(), brayton_engine=_stable_brayton_engine())
    model.eval()
    s, o, ops_si, sens_si = _random_batch(B=4, T=30)
    out = model(s, o, ops_si=ops_si, sens_si=sens_si)
    for k in ("rul", "theta_phys", "lpt_flow_pred"):
        assert torch.isfinite(out[k]).all(), f"{k} contains non-finite"


# ===========================================================================
# RUL feature composition / detach tests
# ===========================================================================

def test_aux_detach_to_rul_blocks_aux_gradient():
    """When detach_aux_to_rul=True, RUL grad must not flow into AuxHead."""
    cfg = _v3_config(detach_aux_to_rul=True, use_aux_in_rul=True)
    model = CycleLayerV3(cfg)
    model.train()
    s, o, _, _ = _random_batch(B=4, T=30)
    out = model(s, o)
    rul = out["rul"]
    loss = rul.mean()
    loss.backward()
    aux_grads = [p.grad for p in model.aux_head.parameters() if p.grad is not None]
    if aux_grads:
        total = sum(g.abs().sum().item() for g in aux_grads)
        assert total == 0.0, (
            "AuxHead gradient leaked from RUL loss despite detach_aux_to_rul=True"
        )


def test_aux_no_detach_to_rul_passes_gradient():
    """When detach_aux_to_rul=False, RUL gradient flows into AuxHead."""
    cfg = _v3_config(detach_aux_to_rul=False, use_aux_in_rul=True)
    model = CycleLayerV3(cfg)
    model.train()
    s, o, _, _ = _random_batch(B=4, T=30)
    out = model(s, o)
    loss = out["rul"].mean()
    loss.backward()
    aux_grads = [p.grad for p in model.aux_head.parameters() if p.grad is not None]
    total = sum(g.abs().sum().item() for g in aux_grads)
    assert total > 0.0, "AuxHead got zero gradient despite detach_aux_to_rul=False"


def test_theta_detach_to_rul_blocks_param_gradient():
    """When detach_theta_to_rul=True, RUL grad must not flow into ParamHead."""
    cfg = _v3_config(detach_theta_to_rul=True, use_theta_in_rul=True)
    model = CycleLayerV3(cfg)
    model.train()
    s, o, _, _ = _random_batch(B=4, T=30)
    out = model(s, o)
    loss = out["rul"].mean()
    loss.backward()
    pg = [p.grad for p in model.param_head.parameters() if p.grad is not None]
    if pg:
        total = sum(g.abs().sum().item() for g in pg)
        assert total == 0.0


# ===========================================================================
# Backward pass — finite grads
# ===========================================================================

def test_backward_pass_finite_grads():
    """One tiny training step should produce finite gradients."""
    torch.manual_seed(13)
    model = CycleLayerV3(_v3_config(), brayton_engine=_stable_brayton_engine())
    model.train()
    s, o, ops_si, sens_si = _random_batch(B=4, T=30)
    out = model(s, o, ops_si=ops_si, sens_si=sens_si)
    # Cheap combined loss touching RUL + theta + aux + brayton temperature
    rul_t = torch.full((4,), 60.0)
    L = (out["rul"] - rul_t).pow(2).mean()
    L = L + (out["theta_phys"] - 1.0).pow(2).mean()
    L = L + out["lpt_flow_pred"].pow(2).mean()
    L = L + out["brayton"]["sensors_pred_si"]["T50_K"].sum() * 1e-6
    L.backward()
    any_grad = False
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite gradient encountered"
            if p.grad.abs().sum().item() > 0:
                any_grad = True
    assert any_grad, "no parameter received a non-zero gradient"
