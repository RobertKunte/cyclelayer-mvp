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
from cyclelayer.models.encoder import SensorEncoder
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
    ops_hidden: int = 16                  # output channels of the ops encoder


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
        -> SensorEncoder(x)        -> theta_hat (B, n_health_params)
        -> OpsEncoder(ops)         -> ops_emb   (B, ops_hidden)
        -> cat([theta_hat, ops_emb])            (B, n_health_params + ops_hidden)
        -> PrognosticsHead
        -> rul (B,)

    forward() returns (rul, theta_hat) to enable CompositeLoss.

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

        if cfg.ops_dim > 0:
            self.ops_enc = nn.Sequential(
                nn.Conv1d(cfg.ops_dim, cfg.ops_hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            prog_in = cfg.n_health_params + cfg.ops_hidden
        else:
            prog_in = cfg.n_health_params

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
        theta_hat = self.encoder(x)
        h = theta_hat
        if self.ops_dim > 0 and ops is not None:
            ops_h = self.ops_enc(ops.permute(0, 2, 1)).squeeze(-1)  # (B, ops_hidden)
            h = torch.cat([theta_hat, ops_h], dim=-1)
        rul = self.prognostics(h)
        return rul, theta_hat

    @classmethod
    def from_config_dict(cls, d: dict) -> "CycleLayerNetV1":
        known = {f.name for f in CycleLayerV1Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = CycleLayerV1Config(**{k: v for k, v in d.items() if k in known})
        return cls(cfg)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
