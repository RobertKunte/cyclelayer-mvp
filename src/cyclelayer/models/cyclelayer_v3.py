"""CycleLayerV3 — V3.1b thermal auxiliary hybrid model.

Wraps the frozen building blocks (SensorEncoder, OpsEncoder, BraytonEngine,
PrognosticsHead) into a single nn.Module configured for the **thermal
auxiliary** scope defined in ADR-0012:

    * physics signal:   T24 / T30 / T50 only (NO pressure / EPR loss)
    * theta_phys:       5-dim factor [0.85, 1.00], init near 0.99, NO supervised L_theta
    * AuxHealthHead:    LPT_flow_mod delta [-0.05, 0.02], init near 0.0
    * target-sensor masking applied AFTER normalization
    * EPR / pressure losses asserted disabled at construction

This file is a NEW module. Legacy modules remain frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from cyclelayer.models.brayton_engine import BraytonEngine
from cyclelayer.models.encoder import OpsEncoder, SensorEncoder
from cyclelayer.models.prognostics import PrognosticsHead


# =============================================================================
# Target-sensor masking
# =============================================================================

class TargetSensorMask(nn.Module):
    """Randomly mask target-sensor columns to 0.0 in NORMALIZED feature space.

    Applied to the encoder input, AFTER the script-side standard
    normalization (so 0.0 in normalized space equals the per-sensor mean
    in raw space).  Prevents the encoder from trivially passing T24/T30/T50
    through to the physics-consistency loss.

    Args:
        target_indices: List of feature-axis indices to mask (X_s columns
            in DS02 order).  Default `[0, 1, 3]` masks T24, T30, T50 which
            are the V3.1b physics-loss targets.
        mask_prob: Per-sample, per-target Bernoulli mask probability.
            Applied only when `self.training == True`.
    """

    def __init__(
        self,
        target_indices: list[int],
        mask_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.target_indices = list(target_indices)
        self.mask_prob = float(mask_prob)

    def forward(self, x_norm: Tensor) -> Tensor:
        """x_norm: (B, T, F).  Returns same shape with selected (B, *, idx)
        rows zeroed (per-sample, all-timesteps).
        """
        if (not self.training) or self.mask_prob <= 0.0 or not self.target_indices:
            return x_norm
        x = x_norm.clone()
        B = x.shape[0]
        for idx in self.target_indices:
            # Per-sample independent Bernoulli draws
            mask = torch.rand(B, device=x.device) < self.mask_prob
            if mask.any():
                x[mask, :, idx] = 0.0
        return x


# =============================================================================
# Heads
# =============================================================================

class ParamHeadPhys(nn.Module):
    """5-dim head producing `theta_phys` in factor space [theta_min, theta_max].

    Activation: scaled sigmoid.
    Bias init: chosen so initial theta ≈ `initial_theta_target` (≈ 0.99 in
    V3.1b — slightly degraded from healthy 1.00, not at midpoint).

    NO supervised loss on theta_phys (V3.1b).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        theta_dim: int = 5,
        bounds: tuple[float, float] = (0.85, 1.00),
        initial_theta_target: float = 0.99,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, theta_dim))
        self.net = nn.Sequential(*layers)
        self.theta_min, self.theta_max = bounds

        # sigmoid(b) → normalized target ⇒ output = min + (max-min)*sigmoid(b)
        norm_t = (initial_theta_target - self.theta_min) / (self.theta_max - self.theta_min)
        norm_t = min(max(norm_t, 1e-6), 1.0 - 1e-6)
        bias_init = math.log(norm_t / (1.0 - norm_t))   # logit
        final: nn.Linear = self.net[-1]   # type: ignore[assignment]
        init.zeros_(final.weight)         # weight=0 → output is pure bias at init
        init.constant_(final.bias, bias_init)

    def forward(self, features: Tensor) -> Tensor:
        raw = self.net(features)
        return self.theta_min + (self.theta_max - self.theta_min) * torch.sigmoid(raw)


class AuxHealthHead(nn.Module):
    """1-dim head producing `lpt_flow_pred` in delta space [lo, hi].

    Activation: tanh-scaled.
    Bias init: chosen so initial output ≈ `initial_value_target` (≈ 0.0 in
    V3.1b — healthy delta, not midpoint of [-0.05, 0.02]).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        bounds: tuple[float, float] = (-0.05, 0.02),
        initial_value_target: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.lo, self.hi = bounds

        # output = lo + (hi - lo) * (tanh(raw)+1)/2 → tanh(raw) = 2*norm − 1
        norm_t = 2.0 * (initial_value_target - self.lo) / (self.hi - self.lo) - 1.0
        norm_t = min(max(norm_t, -1.0 + 1e-6), 1.0 - 1e-6)
        bias_init = math.atanh(norm_t)
        final: nn.Linear = self.net[-1]   # type: ignore[assignment]
        init.zeros_(final.weight)
        init.constant_(final.bias, bias_init)

    def forward(self, features: Tensor) -> Tensor:
        raw = self.net(features)
        return self.lo + (self.hi - self.lo) * (torch.tanh(raw) + 1.0) / 2.0   # (B, 1)


# =============================================================================
# CycleLayerV3 config + module
# =============================================================================

@dataclass
class CycleLayerV3Config:
    """Configuration for the V3.1b thermal-auxiliary hybrid.

    NOTE: `use_pressure_loss` and `use_epr_loss` are NOT loss-side knobs —
    they are tripwires.  V3.1b's loss module *also* asserts these are
    False.  Set False here (default) and keep them False.  Enabling either
    requires a new ADR and code change.
    """

    # Input shapes
    n_sensors: int = 14
    ops_dim: int = 4
    window_size: int = 50

    # SensorEncoder (frozen module reused as feature extractor with linear out)
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    encoder_kernel_size: int = 5
    encoder_mlp_hidden: int = 128
    encoder_dropout: float = 0.2
    encoder_out_dim: int = 64

    # OpsEncoder (frozen)
    use_ops_encoder: bool = True
    ops_channels: list[int] = field(default_factory=lambda: [16, 32])
    ops_out_dim: int = 32

    # Target-sensor masking (V3.1a Patch P6; AFTER normalization)
    mask_target_sensor_prob: float = 0.5
    target_sensor_indices: list[int] = field(default_factory=lambda: [0, 1, 3])

    # ParamHead_phys
    param_hidden: list[int] = field(default_factory=lambda: [64, 32])
    theta_dim: int = 5
    theta_bounds: tuple[float, float] = (0.85, 1.00)
    initial_theta_target: float = 0.99

    # AuxHealthHead
    aux_hidden: list[int] = field(default_factory=lambda: [32])
    aux_bounds: tuple[float, float] = (-0.05, 0.02)
    initial_aux_target: float = 0.0
    detach_aux_to_rul: bool = True

    # PrognosticsHead
    prog_hidden: list[int] = field(default_factory=lambda: [64, 32])
    prog_dropout: float = 0.2
    max_rul: float = 99.0

    # RUL feature composition
    use_theta_in_rul: bool = True
    use_aux_in_rul: bool = True
    detach_theta_to_rul: bool = False

    # FORBIDDEN flags (V3.1b — see ADR-0012)
    use_pressure_loss: bool = False
    use_epr_loss: bool = False


class CycleLayerV3(nn.Module):
    """V3.1b thermal auxiliary hybrid prognostics model.

    Architecture (frozen building blocks):

        sensors_norm (B,T,n_sensors)
              │
              ▼
        TargetSensorMask  (only T24/T30/T50 columns; training only;
                           masked = 0.0 in normalized space)
              │
              ▼
        SensorEncoder (constrain_output=False) ────► h_sens (B, encoder_out_dim)
        OpsEncoder                              ────► z_ops  (B, ops_out_dim)
              │
              ├──────────────┐
              ▼              ▼
        ParamHeadPhys    AuxHealthHead
        → theta_phys     → lpt_flow_pred
          (B, 5)            (B,)

        theta_phys ──► BraytonEngine(ops_si, sens_si, θ) ──► sensors_pred_si
                                                            (T24, T30, P30, T50)

        cat(h_sens, z_ops, [theta_phys], [lpt_flow_pred]) ──► PrognosticsHead ──► RUL

    The model returns a dict; the loss module unpacks it.

    The `theta_true_dim = 0` class attribute signals to legacy trainers that
    theta_true is NOT a model INPUT (we never feed GT health params to the
    network).  L_aux uses LPT_flow_mod GT in the loss only.
    """

    theta_true_dim: int = 0

    def __init__(
        self,
        config: CycleLayerV3Config,
        brayton_engine: BraytonEngine | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        # ── Tripwires ────────────────────────────────────────────────────
        if config.use_epr_loss:
            raise ValueError(
                "use_epr_loss=True is FORBIDDEN in V3.1b — see "
                "docs/decisions/ADR-0012-v3-thermal-auxiliary-scope.md"
            )
        if config.use_pressure_loss:
            raise ValueError(
                "use_pressure_loss=True is FORBIDDEN in V3.1b — see "
                "docs/decisions/ADR-0012-v3-thermal-auxiliary-scope.md"
            )

        self.ops_dim = config.ops_dim

        self.mask = TargetSensorMask(
            target_indices=config.target_sensor_indices,
            mask_prob=config.mask_target_sensor_prob,
        )

        self.sensor_encoder = SensorEncoder(
            n_features=config.n_sensors,
            window_size=config.window_size,
            n_theta=config.encoder_out_dim,
            cnn_channels=tuple(config.encoder_channels),
            kernel_size=config.encoder_kernel_size,
            mlp_hidden=config.encoder_mlp_hidden,
            dropout=config.encoder_dropout,
            constrain_output=False,             # raw linear output
        )

        if config.use_ops_encoder:
            self.ops_encoder: OpsEncoder | None = OpsEncoder(
                ops_dim=config.ops_dim,
                channels=list(config.ops_channels),
                out_dim=config.ops_out_dim,
            )
            feat_dim = config.encoder_out_dim + config.ops_out_dim
        else:
            self.ops_encoder = None
            feat_dim = config.encoder_out_dim

        self.param_head = ParamHeadPhys(
            in_dim=feat_dim,
            hidden=list(config.param_hidden),
            theta_dim=config.theta_dim,
            bounds=tuple(config.theta_bounds),
            initial_theta_target=config.initial_theta_target,
        )

        self.aux_head = AuxHealthHead(
            in_dim=feat_dim,
            hidden=list(config.aux_hidden),
            bounds=tuple(config.aux_bounds),
            initial_value_target=config.initial_aux_target,
        )

        # BraytonEngine is optional at construction (script wires it in).
        # When None, the model still produces theta_phys, lpt_flow_pred, RUL.
        self.brayton_engine = brayton_engine

        # PrognosticsHead input dim
        prog_in = feat_dim
        if config.use_theta_in_rul:
            prog_in += config.theta_dim
        if config.use_aux_in_rul:
            prog_in += 1

        self.prognostics = PrognosticsHead(
            in_features=prog_in,
            hidden_sizes=tuple(config.prog_hidden),
            dropout=config.prog_dropout,
            max_rul=config.max_rul,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sensors_norm: Tensor,                          # (B, T, n_sensors)
        ops_norm: Tensor | None = None,                # (B, T, ops_dim)
        ops_si: dict[str, Tensor] | None = None,
        sens_si: dict[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
            sensors_norm: Normalized sensor window (B, T, n_sensors).
            ops_norm: Normalized ops window (B, T, ops_dim) for OpsEncoder.
            ops_si: Per-sample ops dict (T2_K, P2_Pa, alt_m, mach) for
                BraytonEngine — values at the LAST timestep of the window.
                If `None`, BraytonEngine is skipped.
            sens_si: Per-sample sens dict (Nf_rpm, Nc_rpm, Wf_kgs) for
                BraytonEngine — values at the LAST timestep of the window.

        Returns:
            Dict with keys:
                rul:           (B,) predicted RUL
                theta_phys:    (B, 5)  health-modifier factors [0.85, 1.0]
                lpt_flow_pred: (B,)   AuxHead delta [-0.05, 0.02]
                h_sens:        (B, encoder_out_dim)
                z_ops:         (B, ops_out_dim) or None
                brayton:       optional dict {sensors_pred_si, diag}
        """
        sensors_in = self.mask(sensors_norm)
        h_sens = self.sensor_encoder(sensors_in)

        if self.ops_encoder is not None:
            assert ops_norm is not None, (
                "use_ops_encoder=True but ops_norm not provided"
            )
            z_ops = self.ops_encoder(ops_norm)
            features = torch.cat([h_sens, z_ops], dim=-1)
        else:
            z_ops = None
            features = h_sens

        theta_phys    = self.param_head(features)               # (B, 5)
        lpt_flow_pred = self.aux_head(features).squeeze(-1)     # (B,)

        brayton_out: dict[str, Any] | None = None
        if self.brayton_engine is not None and ops_si is not None and sens_si is not None:
            sensors_pred_si, diag = self.brayton_engine(ops_si, sens_si, theta_phys)
            brayton_out = {"sensors_pred_si": sensors_pred_si, "diag": diag}

        # RUL feature composition
        rul_parts: list[Tensor] = [features]
        if self.config.use_theta_in_rul:
            t = theta_phys.detach() if self.config.detach_theta_to_rul else theta_phys
            rul_parts.append(t)
        if self.config.use_aux_in_rul:
            a = lpt_flow_pred.detach() if self.config.detach_aux_to_rul else lpt_flow_pred
            rul_parts.append(a.unsqueeze(-1))
        rul = self.prognostics(torch.cat(rul_parts, dim=-1))

        out: dict[str, Any] = {
            "rul":            rul,
            "theta_phys":     theta_phys,
            "lpt_flow_pred":  lpt_flow_pred,
            "h_sens":         h_sens,
            "z_ops":          z_ops,
        }
        if brayton_out is not None:
            out["brayton"] = brayton_out
        return out

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config_dict(
        cls,
        d: dict[str, Any],
        brayton_engine: BraytonEngine | None = None,
    ) -> "CycleLayerV3":
        """Build from a plain dict (e.g. parsed YAML `model:` section)."""
        from dataclasses import fields
        known = {f.name for f in fields(CycleLayerV3Config)}
        cfg = CycleLayerV3Config(
            **{k: v for k, v in d.items() if k in known}
        )
        return cls(cfg, brayton_engine=brayton_engine)
