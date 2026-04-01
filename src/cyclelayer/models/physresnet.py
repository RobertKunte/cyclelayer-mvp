"""OpsResidualNet (cyclelayer_v2): physics-motivated residual factorization.

Core insight
------------
Sensor readings decompose as:

    x(t) = x_healthy(ops(t)) + x_degradation(t) + noise

The ``OpsToSensorBaseline`` module estimates x_healthy(ops) — what a healthy
engine would measure under the given operating conditions.  Subtracting this
from the actual sensor readings yields a residual that encodes *only*
degradation-driven variation, not operating-point variation.

Data flow
---------
    ops  (B, T, 4)  → OpsToSensorBaseline → x_ref  (B, T, 14)
    x_res = x_sens - x_ref                          [degradation residual]
    x_res (B, T, 14) → SensorEncoder → h_deg (B, 10)  [PRIMARY: degradation path]
    ops  (B, T, 4)  → OpsEncoder     → z_ops (B, 32)  [CONTEXT: regime path]
    cat([h_deg, z_ops]) → [fusion] → PrognosticsHead → rul (B,)

NOTE on architecture intent
---------------------------
This is an INTERMEDIATE STEP — not a full thermodynamic physics integration.
The baseline is a learned ops→sensor map, not a Brayton cycle calculation.
Future: ops could feed directly into BraytonLayer as thermodynamic boundary
conditions (T1, P1 at compressor inlet).

The theta supervision on h_deg (L_theta in CompositeLoss) prevents the
baseline from absorbing degradation signal: if the baseline over-fits
degradation, h_deg can no longer predict theta_true → L_theta increases.

forward() always returns (rul, h_deg) for multi-task training.
forward_aux() returns a diagnostic dict for visualization.
self._x_ref stores the baseline output for optional smoothness regularization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from cyclelayer.models.encoder import OpsEncoder, OpsToSensorBaseline, SensorEncoder
from cyclelayer.models.prognostics import PrognosticsHead


@dataclass
class OpsResidualNetConfig:
    """Hyper-parameters for OpsResidualNet (cyclelayer_v2)."""

    n_features: int = 14                     # sensor channels (X_s only, not W)
    ops_dim: int = 4                         # operating condition channels (W)
    window_size: int = 30
    n_health_params: int = 10                # matches N-CMAPSS T_dev columns
    # OpsToSensorBaseline settings
    baseline_hidden_dim: int = 32
    baseline_n_layers: int = 2
    # SensorEncoder (residual path)
    cnn_channels: tuple[int, ...] = (32, 64, 64)
    encoder_kernel_size: int = 3
    encoder_mlp_hidden: int = 128
    encoder_dropout: float = 0.2
    # OpsEncoder (regime context path — lighter than sensor encoder)
    ops_enc_channels: list[int] = field(default_factory=lambda: [16, 32])
    ops_enc_out_dim: int = 32
    # Fusion projection after cat([h_deg, z_ops])
    fusion_hidden_dim: int = 64
    # PrognosticsHead
    prog_hidden_sizes: tuple[int, ...] = (64, 32)
    prog_dropout: float = 0.2
    max_rul: float | None = 99.0


class OpsResidualNet(nn.Module):
    """Physics-motivated residual factorization for RUL prediction (V2).

    Separates operating-condition effects from degradation effects by
    subtracting a learned healthy-reference baseline before encoding.

    See module docstring for architecture and design notes.

    Args:
        config: OpsResidualNetConfig with all hyper-parameters.
    """

    def __init__(self, config: OpsResidualNetConfig | None = None) -> None:
        super().__init__()
        cfg = config or OpsResidualNetConfig()
        self.config = cfg
        self.ops_dim = cfg.ops_dim
        # _x_ref is set during forward() for optional smoothness regularization.
        # No detach — gradients must flow through it for the smoothness loss.
        self._x_ref: Tensor | None = None

        self.baseline = OpsToSensorBaseline(
            ops_dim=cfg.ops_dim,
            n_sensors=cfg.n_features,
            hidden_dim=cfg.baseline_hidden_dim,
            n_layers=cfg.baseline_n_layers,
        )
        self.encoder = SensorEncoder(
            n_features=cfg.n_features,
            window_size=cfg.window_size,
            n_theta=cfg.n_health_params,
            cnn_channels=cfg.cnn_channels,
            kernel_size=cfg.encoder_kernel_size,
            mlp_hidden=cfg.encoder_mlp_hidden,
            dropout=cfg.encoder_dropout,
            constrain_output=False,   # health params, not Brayton theta
        )
        self.ops_enc = OpsEncoder(
            ops_dim=cfg.ops_dim,
            channels=cfg.ops_enc_channels,
            out_dim=cfg.ops_enc_out_dim,
        )

        concat_dim = cfg.n_health_params + cfg.ops_enc_out_dim
        if cfg.fusion_hidden_dim > 0:
            self.fusion: nn.Sequential | None = nn.Sequential(
                nn.Linear(concat_dim, cfg.fusion_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.fusion_hidden_dim),
            )
            prog_in = cfg.fusion_hidden_dim
        else:
            self.fusion = None
            prog_in = concat_dim

        self.prognostics = PrognosticsHead(
            in_features=prog_in,
            hidden_sizes=cfg.prog_hidden_sizes,
            dropout=cfg.prog_dropout,
            max_rul=cfg.max_rul,
        )

    # ------------------------------------------------------------------
    # Shared computation
    # ------------------------------------------------------------------

    def _forward_all(
        self, x: Tensor, ops: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Shared computation for forward() and forward_aux().

        Args:
            x:   (B, T, n_features)   sensor readings
            ops: (B, T, ops_dim)      operating conditions

        Returns:
            (rul, h_deg, x_ref, x_res, z_ops)
        """
        x_ref = self.baseline(ops)                       # (B, T, n_features)
        x_res = x - x_ref                                # (B, T, n_features)
        h_deg = self.encoder(x_res)                      # (B, n_health_params)
        z_ops = self.ops_enc(ops)                        # (B, ops_enc_out_dim)
        h = torch.cat([h_deg, z_ops], dim=-1)
        if self.fusion is not None:
            h = self.fusion(h)
        rul = self.prognostics(h)                        # (B,)
        return rul, h_deg, x_ref, x_res, z_ops

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, ops: Tensor) -> tuple[Tensor, Tensor]:
        """Return (rul, h_deg) for multi-task training.

        Also stores self._x_ref (with gradients) for the optional baseline
        smoothness regularization term in CompositeLoss.

        Args:
            x:   (B, window_size, n_features)  sensor readings
            ops: (B, window_size, ops_dim)      operating conditions

        Returns:
            rul:   (B,)
            h_deg: (B, n_health_params)  — supervised against theta_true
        """
        rul, h_deg, x_ref, _, _ = self._forward_all(x, ops)
        self._x_ref = x_ref   # keep gradient for smoothness loss
        return rul, h_deg

    def forward_aux(self, x: Tensor, ops: Tensor) -> dict[str, Tensor]:
        """Diagnostic forward pass returning all intermediate tensors.

        Not used during training.  Use for visualization and analysis.

        Args:
            x:   (B, window_size, n_features)
            ops: (B, window_size, ops_dim)

        Returns:
            dict with keys: ``x_ref``, ``x_res``, ``h_deg``, ``z_ops``, ``rul``.
        """
        rul, h_deg, x_ref, x_res, z_ops = self._forward_all(x, ops)
        return {
            "x_ref": x_ref,
            "x_res": x_res,
            "h_deg": h_deg,
            "z_ops": z_ops,
            "rul":   rul,
        }

    @classmethod
    def from_config_dict(cls, d: dict) -> "OpsResidualNet":
        """Construct from a plain dict (e.g. loaded from YAML)."""
        from dataclasses import fields
        known = {f.name for f in fields(OpsResidualNetConfig)}
        cfg = OpsResidualNetConfig(**{k: v for k, v in d.items() if k in known})
        return cls(cfg)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
