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


class CycleLayerNetV1(nn.Module):
    """Phase 2 CycleLayer: supervised health-parameter multi-task network.

    Data flow:
        x (B, T, F)
        -> SensorEncoder  (constrain_output=False, sigmoid -> [0,1])
        -> theta_hat (B, n_health_params)
        -> PrognosticsHead
        -> rul (B,)

    forward() returns (rul, theta_hat) to enable CompositeLoss.

    The BraytonCycleLayer is intentionally omitted in V1 to keep the
    multi-task loss independent of the physics layer.  A subsequent phase
    will insert a physics mapping between theta_hat and cycle features.

    Args:
        config: CycleLayerV1Config with all hyper-parameters.
    """

    def __init__(self, config: CycleLayerV1Config | None = None) -> None:
        super().__init__()
        cfg = config or CycleLayerV1Config()
        self.config = cfg

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
        self.prognostics = PrognosticsHead(
            in_features=cfg.n_health_params,
            hidden_sizes=cfg.prog_hidden_sizes,
            dropout=cfg.prog_dropout,
            max_rul=cfg.max_rul,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (rul, theta_hat).

        Args:
            x: (B, window_size, n_features)
        Returns:
            rul:       (B,)
            theta_hat: (B, n_health_params)
        """
        theta_hat = self.encoder(x)
        rul       = self.prognostics(theta_hat)
        return rul, theta_hat

    @classmethod
    def from_config_dict(cls, d: dict) -> "CycleLayerNetV1":
        known = {f.name for f in CycleLayerV1Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = CycleLayerV1Config(**{k: v for k, v in d.items() if k in known})
        return cls(cfg)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
