"""CycleLayer model variants.

CycleLayerNet (v0)
    Encoder -> BraytonCycleLayer -> PrognosticsHead -> RUL
    Used for Phase 0 architecture validation.

CycleLayerNetV1 (Phase 2, multi-task)
    Encoder -> theta_hat (health params, no Brayton layer)
    theta_hat -> PrognosticsHead -> RUL
    Also returns theta_hat for supervised multi-task loss.
    Designed to be extended with a physics layer later (Phase 3+).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from cyclelayer.models.brayton_cycle import BraytonCycleLayer
from cyclelayer.models.encoder import OpsEncoder, SensorEncoder
from cyclelayer.models.prognostics import PrognosticsHead


# ---------------------------------------------------------------------------
# V0 — Brayton cycle architecture
# ---------------------------------------------------------------------------

@dataclass
class CycleLayerConfig:
    """Hyper-parameters for CycleLayerNet (v0)."""

    n_features: int = 18
    window_size: int = 30
    n_theta: int = 5                          # Brayton params, >= 5
    cnn_channels: tuple[int, ...] = (32, 64, 64)
    encoder_kernel_size: int = 3
    encoder_mlp_hidden: int = 128
    encoder_dropout: float = 0.2
    gamma: float = 1.4
    cp: float = 1005.0
    prog_hidden_sizes: tuple[int, ...] = (64, 32)
    prog_dropout: float = 0.1
    max_rul: float | None = 125.0


class CycleLayerNet(nn.Module):
    """Physics-informed RUL network (v0): Encoder + BraytonCycle + Prognostics.

    Args:
        config: CycleLayerConfig with all hyper-parameters.
    """

    def __init__(self, config: CycleLayerConfig | None = None) -> None:
        super().__init__()
        cfg = config or CycleLayerConfig()
        self.config = cfg

        self.encoder = SensorEncoder(
            n_features=cfg.n_features,
            window_size=cfg.window_size,
            n_theta=cfg.n_theta,
            cnn_channels=cfg.cnn_channels,
            kernel_size=cfg.encoder_kernel_size,
            mlp_hidden=cfg.encoder_mlp_hidden,
            dropout=cfg.encoder_dropout,
            constrain_output=True,
        )
        self.brayton = BraytonCycleLayer(
            n_params=cfg.n_theta,
            gamma=cfg.gamma,
            cp=cfg.cp,
        )
        self.prognostics = PrognosticsHead(
            in_features=self.brayton.out_features,
            hidden_sizes=cfg.prog_hidden_sizes,
            dropout=cfg.prog_dropout,
            max_rul=cfg.max_rul,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: (B, window_size, n_features)
        Returns:
            rul: (B,)
        """
        theta    = self.encoder(x)
        features = self.brayton(theta)
        return self.prognostics(features)

    def forward_with_intermediates(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (rul, theta, cycle_features)."""
        theta    = self.encoder(x)
        features = self.brayton(theta)
        rul      = self.prognostics(features)
        return rul, theta, features

    @classmethod
    def from_config_dict(cls, d: dict) -> "CycleLayerNet":
        """Construct from a plain dict (e.g. loaded from YAML)."""
        # Extract only known fields to avoid TypeError from extra YAML keys
        known = {f.name for f in CycleLayerConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = CycleLayerConfig(**{k: v for k, v in d.items() if k in known})
        return cls(cfg)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# V1 — Multi-task health-parameter supervised (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class CycleLayerV1Config:
    """Hyper-parameters for CycleLayerNetV1."""

    n_features: int = 18
    window_size: int = 30
    n_health_params: int = 10             # matches N-CMAPSS T_dev columns
    cnn_channels: tuple[int, ...] = (32, 64, 64)
    encoder_kernel_size: int = 3
    encoder_mlp_hidden: int = 128
    encoder_dropout: float = 0.2
    prog_hidden_sizes: tuple[int, ...] = (64, 32)
    prog_dropout: float = 0.1
    max_rul: float | None = 125.0
    lambda_theta: float = 0.1             # weight for theta supervision loss
    ops_dim: int = 0                      # 4 for N-CMAPSS W; 0 = no ops path
    # OpsEncoder settings (only used when ops_dim > 0)
    # Deliberately lighter than the sensor encoder — ops encode flight regime,
    # not degradation state.  Future: ops could feed into BraytonLayer directly.
    ops_enc_channels: list[int] = field(default_factory=lambda: [16, 32])
    ops_enc_out_dim: int = 32
    # fusion_hidden_dim > 0 adds a Linear+GELU+LayerNorm projection after
    # concat([theta_hat, ops_emb]) so the model learns regime-conditioned RUL.
    # 0 = skip fusion (concat fed directly to PrognosticsHead).
    fusion_hidden_dim: int = 0


class CycleLayerNetV1(nn.Module):
    """Phase 2 CycleLayer: supervised health-parameter multi-task network.

    Data flow (without ops):
        x (B, T, F)
        -> SensorEncoder
        -> theta_hat (B, n_health_params)
        -> PrognosticsHead
        -> rul (B,)

    Data flow (with ops, ops_dim > 0):
        x (B, T, F)  +  ops (B, T, ops_dim)
        -> SensorEncoder(x)           -> theta_hat (B, n_health_params)  [PRIMARY: physics path]
        -> OpsEncoder(ops)            -> ops_emb   (B, ops_enc_out_dim)  [CONTEXT: regime path]
        -> cat([theta_hat, ops_emb])
        -> [optional fusion projection]
        -> PrognosticsHead
        -> rul (B,)

    NOTE on architecture intent: theta_hat (engine health) is the PRIMARY signal.
    ops_emb provides operating regime CONTEXT — it must not become a shortcut that
    bypasses the physics encoder.  Future: ops could feed as boundary conditions
    directly into BraytonLayer (T1, P1 at compressor inlet).

    forward() always returns (rul, theta_hat) for CompositeLoss multi-task training.

    Args:
        config: CycleLayerV1Config with all hyper-parameters.
    """

    def __init__(self, config: CycleLayerV1Config | None = None) -> None:
        super().__init__()
        cfg = config or CycleLayerV1Config()
        self.config = cfg
        self.ops_dim = cfg.ops_dim

        self.encoder = SensorEncoder(
            n_features=cfg.n_features,
            window_size=cfg.window_size,
            n_theta=cfg.n_health_params,
            cnn_channels=cfg.cnn_channels,
            kernel_size=cfg.encoder_kernel_size,
            mlp_hidden=cfg.encoder_mlp_hidden,
            dropout=cfg.encoder_dropout,
            constrain_output=False,       # health params, not Brayton
        )

        # Operating-condition regime encoder (lightweight — regime signal, not degradation)
        if cfg.ops_dim > 0:
            self.ops_enc: OpsEncoder | None = OpsEncoder(
                ops_dim=cfg.ops_dim,
                channels=cfg.ops_enc_channels,
                out_dim=cfg.ops_enc_out_dim,
            )
            ops_out = cfg.ops_enc_out_dim
        else:
            self.ops_enc = None
            ops_out = 0

        # Optional fusion projection after cat([theta_hat, ops_emb])
        concat_dim = cfg.n_health_params + ops_out
        if cfg.fusion_hidden_dim > 0 and ops_out > 0:
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

    def forward(self, x: Tensor, ops: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return (rul, theta_hat).

        Args:
            x:   (B, window_size, n_features)
            ops: (B, window_size, ops_dim) or None
        Returns:
            rul:       (B,)
            theta_hat: (B, n_health_params)
        """
        theta_hat = self.encoder(x)   # physics path — PRIMARY
        h = theta_hat
        if self.ops_enc is not None and ops is not None:
            h = torch.cat([theta_hat, self.ops_enc(ops)], dim=-1)
        if self.fusion is not None:
            h = self.fusion(h)
        rul = self.prognostics(h)
        return rul, theta_hat   # theta_hat always returned for multi-task supervision

    @classmethod
    def from_config_dict(cls, d: dict) -> "CycleLayerNetV1":
        known = {f.name for f in CycleLayerV1Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = CycleLayerV1Config(**{k: v for k, v in d.items() if k in known})
        return cls(cfg)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
