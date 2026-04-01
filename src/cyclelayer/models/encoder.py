"""Sensor Encoder: raw sensor windows → cycle or health parameters θ.

Two output modes selected via ``constrain_output``:

constrain_output=True  (default, Brayton mode):
    Maps raw outputs to physically plausible Brayton cycle parameters.
    Requires n_theta >= 5, layout [T1, pi, T3, eta_c, eta_t, ...].

constrain_output=False  (health-param mode, Phase 2):
    Returns raw MLP output (linear activation) — no sigmoid.
    Targets are StandardScaler-normalised health parameters (z-scores,
    typically in [-3, 3]). Using sigmoid here would restrict outputs to
    (0, 1) and make the Huber loss unminimisable, causing the encoder to
    saturate at ~0.5 and the RUL head to predict a constant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SensorEncoder(nn.Module):
    """1-D CNN encoder mapping sensor windows to parameter vectors θ.

    Args:
        n_features: Number of input sensor/condition channels.
        window_size: Temporal length of the input window.
        n_theta: Output dimension (>= 5 when ``constrain_output=True``).
        cnn_channels: Channel sizes for the 1-D CNN blocks.
        kernel_size: Convolutional kernel size.
        mlp_hidden: Hidden size of the MLP projection head.
        dropout: Dropout probability applied before the MLP.
        constrain_output: If True, apply Brayton physical constraints.
            If False, return raw linear output (for z-scored health params).
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        n_theta: int = 5,
        cnn_channels: tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 3,
        mlp_hidden: int = 128,
        dropout: float = 0.2,
        constrain_output: bool = True,
    ) -> None:
        super().__init__()
        if constrain_output and n_theta < 5:
            raise ValueError("n_theta must be >= 5 when constrain_output=True.")

        self.n_theta = n_theta
        self.constrain_output = constrain_output

        cnn_layers: list[nn.Module] = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        cnn_layers.append(nn.AdaptiveAvgPool1d(1))  # (B, C, 1)
        self.cnn = nn.Sequential(*cnn_layers)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, n_theta),
        )

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Encode a sensor window to θ.

        Args:
            x: Tensor of shape (B, window_size, n_features).

        Returns:
            theta: Tensor of shape (B, n_theta).
        """
        h   = self.cnn(x.permute(0, 2, 1))  # (B, C, 1)
        raw = self.mlp(h)                    # (B, n_theta)
        if self.constrain_output:
            return self._constrain_brayton(raw)
        return raw                           # linear; matches z-scored targets

    # ------------------------------------------------------------------

    @staticmethod
    def _constrain_brayton(raw: Tensor) -> Tensor:
        """Map raw outputs to physically plausible Brayton cycle ranges."""
        sig = torch.sigmoid
        T1    = 250.0  + sig(raw[:, 0:1]) * 650.0    # [250, 900] K
        pi    = 1.0    + nn.functional.softplus(raw[:, 1:2])  # (1, ∞)
        T3    = 1000.0 + sig(raw[:, 2:3]) * 1000.0   # [1000, 2000] K
        eta_c = 0.70   + sig(raw[:, 3:4]) * 0.30     # [0.70, 1.00]
        eta_t = 0.70   + sig(raw[:, 4:5]) * 0.30     # [0.70, 1.00]
        parts = [T1, pi, T3, eta_c, eta_t]
        if raw.shape[1] > 5:
            parts.append(sig(raw[:, 5:]))
        return torch.cat(parts, dim=1)   # (B, n_theta)


# ---------------------------------------------------------------------------
# OpsEncoder — operating-condition-aware temporal encoder (regime encoder)
# ---------------------------------------------------------------------------

class OpsEncoder(nn.Module):
    """Operating-condition-aware temporal encoder (regime encoder).

    Encodes the operating point (altitude, Mach, TRA, T2) as temporal context
    for the sensor encoder.  Intentionally lighter than SensorEncoder: operating
    conditions are a regime/context signal, not a degradation signal.

    NOTE: This is an INTERMEDIATE STEP — not a thermodynamic physics layer.
    Future: ops could feed directly into a BraytonLayer as boundary conditions
    (e.g. T1, P1 at the compressor inlet), reducing the latent space physically.

    Architecture: Conv1d-BN-GELU stack → temporal pool → Linear projection.
    Input:  ops (B, T, ops_dim)
    Output: z_ops (B, out_dim)

    Args:
        ops_dim: Number of operating condition channels (default 4: alt, Mach, TRA, T2).
        channels: Conv channel depths.  Defaults to [16, 32] — deliberately lighter
            than the sensor encoder [32, 64] since ops encode flight regime, not
            degradation state.
        out_dim: Output embedding dimension.
        kernel_size: Convolutional kernel size.
        dropout: Dropout probability applied before the projection.
    """

    def __init__(
        self,
        ops_dim: int = 4,
        channels: list[int] | None = None,
        out_dim: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs: list[int] = channels if channels is not None else [16, 32]
        layers: list[nn.Module] = []
        in_ch = ops_dim
        for ch in chs:
            layers += [
                nn.Conv1d(in_ch, ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            ]
            in_ch = ch
        self.cnn = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_ch, out_dim)

    def _pool(self, h: Tensor) -> Tensor:
        """Temporal pooling: (B, C, T) → (B, C).

        Current strategy: global mean (AdaptiveAvgPool).
        Future options to consider: mean+std concat, attention pooling, last-timestep.
        To extend: override this method or add a ``pooling_mode`` parameter.
        """
        return h.mean(dim=-1)

    def forward(self, ops: Tensor) -> Tensor:
        """Encode an operating-condition window.

        Args:
            ops: Tensor of shape (B, T, ops_dim).

        Returns:
            z_ops: Tensor of shape (B, out_dim).
        """
        h = self.cnn(ops.permute(0, 2, 1))  # (B, ops_dim, T) → (B, C, T)
        h = self._pool(h)                   # (B, C)
        return self.proj(self.dropout(h))   # (B, out_dim)


# ---------------------------------------------------------------------------
# OpsToSensorBaseline — ops-conditioned healthy reference predictor
# ---------------------------------------------------------------------------

class OpsToSensorBaseline(nn.Module):
    """Maps operating conditions to a healthy-reference sensor prediction.

    Estimates B(ops(t)) ≈ x_healthy(ops(t)): what a healthy engine would
    measure under the current operating conditions.  The residual
    x(t) - B(ops(t)) then encodes *only* degradation-driven variation.

    NOTE: This is NOT a thermodynamic physics layer.  It is a learned
    "ops → healthy baseline" map — an intermediate step before a full
    BraytonLayer integration.

    Architecture: Conv1d-BN-GELU stack (temporal, no pooling) → 1×1 Conv
    projection to n_sensors channels.

    Near-zero output init: the final projection weight/bias starts near zero
    so the model begins training with residual ≈ x_sens (no harmful
    subtraction at epoch 0).

    Input:  ops   (B, T, ops_dim)
    Output: x_ref (B, T, n_sensors)

    Args:
        ops_dim:    Number of operating-condition channels (default 4).
        n_sensors:  Number of sensor output channels (default 14).
        hidden_dim: Conv channel width in the hidden layers.
        n_layers:   Number of Conv-BN-GELU blocks.
        kernel_size: Convolutional kernel size (padding preserves T).
    """

    def __init__(
        self,
        ops_dim: int = 4,
        n_sensors: int = 14,
        hidden_dim: int = 32,
        n_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = ops_dim
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ]
            in_ch = hidden_dim
        self.cnn = nn.Sequential(*layers)
        # 1×1 conv: projects hidden features to n_sensors; preserves temporal dim
        self.proj = nn.Conv1d(in_ch, n_sensors, kernel_size=1)
        # Near-zero init: residual starts ≈ x_sens at epoch 0
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.proj.bias)  # type: ignore[arg-type]

    def forward(self, ops: Tensor) -> Tensor:
        """Predict healthy-reference sensor window from operating conditions.

        Args:
            ops: Tensor of shape (B, T, ops_dim).

        Returns:
            x_ref: Tensor of shape (B, T, n_sensors).
        """
        h = self.cnn(ops.permute(0, 2, 1))  # (B, C, T)
        return self.proj(h).permute(0, 2, 1)  # (B, T, n_sensors)
