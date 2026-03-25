"""Baseline models for comparison with CycleLayerNet.

Both models accept (B, window_size, n_features) and return scalar RUL (B,).
When ``theta_true_dim > 0``, a second input ``theta_true`` of shape
(B, theta_true_dim) is concatenated to the pooled representation before the
MLP head.  This enables the "upper bound" experiment (Phase 1) where ground-
truth health parameters are provided as extra signal.

forward signature:
    theta_true_dim == 0 (default):   forward(x)            -> rul
    theta_true_dim  > 0:             forward(x, theta_true) -> rul
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from cyclelayer.models.encoder import OpsEncoder


# ---------------------------------------------------------------------------
# CNN Baseline
# ---------------------------------------------------------------------------

class CNNBaseline(nn.Module):
    """1-D convolutional baseline for RUL regression.

    Args:
        n_features: Input feature channels (X_s only when ops_dim > 0).
        channels: Channel sizes for each conv block.
        kernel_size: Kernel size for all conv layers.
        mlp_hidden: Hidden size of the MLP head.
        dropout: Dropout probability.
        max_rul: Soft output clamp (None to disable).
        theta_true_dim: If > 0, concatenate a ground-truth health-parameter
            vector of this size to the CNN embedding before the MLP head.
        ops_dim: If > 0, enable a separate OpsEncoder for operating conditions
            (W: alt, Mach, TRA, T2).  ops_dim is typically 4.
        ops_enc_channels: Conv channel depths for OpsEncoder.  Defaults to
            [16, 32] (lighter than sensor path — regime signal, not degradation).
        ops_enc_out_dim: Output embedding dimension of OpsEncoder.
        fusion_hidden_dim: If > 0, apply a Linear+GELU+LayerNorm fusion projection
            after concat([sensor_emb, ops_emb]) before the MLP head.  This lets the
            model learn to interpret sensor signals under the given operating regime.
            0 = skip fusion (concat fed directly to head).
    """

    def __init__(
        self,
        n_features: int,
        channels: tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 3,
        mlp_hidden: int = 128,
        dropout: float = 0.2,
        max_rul: float | None = 125.0,
        theta_true_dim: int = 0,
        ops_dim: int = 0,
        ops_enc_channels: list[int] | None = None,
        ops_enc_out_dim: int = 32,
        fusion_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.max_rul = max_rul
        self.theta_true_dim = theta_true_dim
        self.ops_dim = ops_dim

        cnn_layers: list[nn.Module] = []
        in_ch = n_features
        for out_ch in channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        cnn_layers.append(nn.AdaptiveAvgPool1d(1))
        self.cnn = nn.Sequential(*cnn_layers)
        self._cnn_out_ch = in_ch

        # Operating-condition regime encoder (OpsEncoder)
        # Processes ops as a temporal sequence — not just a summary statistic.
        # NOTE: this is an intermediate step; ops may later feed into BraytonLayer.
        if ops_dim > 0:
            self.ops_enc: OpsEncoder | None = OpsEncoder(
                ops_dim=ops_dim,
                channels=ops_enc_channels,
                out_dim=ops_enc_out_dim,
            )
            self._ops_out = ops_enc_out_dim
        else:
            self.ops_enc = None
            self._ops_out = 0

        # Optional fusion projection: learns regime-conditioned feature interpretation
        concat_dim = in_ch + theta_true_dim + self._ops_out
        if fusion_hidden_dim > 0 and self._ops_out > 0:
            self.fusion: nn.Sequential | None = nn.Sequential(
                nn.Linear(concat_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_hidden_dim),
            )
            head_in = fusion_hidden_dim
        else:
            self.fusion = None
            head_in = concat_dim

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
            nn.Softplus(),
        )
        if max_rul is not None:
            init.xavier_uniform_(self.head[-2].weight)
            init.constant_(self.head[-2].bias, max_rul / 2.0)

    def forward(
        self,
        x: Tensor,
        theta_true: Tensor | None = None,
        ops: Tensor | None = None,
    ) -> Tensor:
        """Args:
            x:          (B, window_size, n_features)
            theta_true: (B, theta_true_dim) or None
            ops:        (B, window_size, ops_dim) or None
        Returns:
            rul: (B,)
        """
        h = self.cnn(x.permute(0, 2, 1)).squeeze(-1)  # (B, C)
        if self.theta_true_dim > 0 and theta_true is not None:
            h = torch.cat([h, theta_true], dim=-1)
        if self.ops_enc is not None and ops is not None:
            h = torch.cat([h, self.ops_enc(ops)], dim=-1)
        if self.fusion is not None:
            h = self.fusion(h)
        out = self.head(h).squeeze(-1)
        if self.max_rul is not None:
            out = out.clamp(max=self.max_rul)
        return out


# ---------------------------------------------------------------------------
# LSTM Baseline
# ---------------------------------------------------------------------------

class LSTMBaseline(nn.Module):
    """Bidirectional LSTM baseline for RUL regression.

    Args:
        n_features: Input feature channels (X_s only when ops_dim > 0).
        hidden_size: LSTM hidden state size (per direction).
        n_layers: Number of LSTM layers.
        bidirectional: If True, use a bidirectional LSTM.
        mlp_hidden: Hidden size of the MLP head.
        dropout: Dropout probability.
        max_rul: Soft output clamp (None to disable).
        theta_true_dim: If > 0, concatenate ground-truth health parameters
            to the LSTM embedding before the MLP head.
        ops_dim: If > 0, enable a separate OpsEncoder for operating conditions.
        ops_enc_channels: Conv channel depths for OpsEncoder (default [16, 32]).
        ops_enc_out_dim: Output embedding dimension of OpsEncoder.
        fusion_hidden_dim: If > 0, apply a fusion projection after concat before head.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        bidirectional: bool = True,
        mlp_hidden: int = 64,
        dropout: float = 0.2,
        max_rul: float | None = 125.0,
        theta_true_dim: int = 0,
        ops_dim: int = 0,
        ops_enc_channels: list[int] | None = None,
        ops_enc_out_dim: int = 32,
        fusion_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.max_rul = max_rul
        self.bidirectional = bidirectional
        self.theta_true_dim = theta_true_dim
        self.ops_dim = ops_dim

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        if ops_dim > 0:
            self.ops_enc: OpsEncoder | None = OpsEncoder(
                ops_dim=ops_dim,
                channels=ops_enc_channels,
                out_dim=ops_enc_out_dim,
            )
            self._ops_out = ops_enc_out_dim
        else:
            self.ops_enc = None
            self._ops_out = 0

        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        concat_dim = lstm_out_size + theta_true_dim + self._ops_out
        if fusion_hidden_dim > 0 and self._ops_out > 0:
            self.fusion: nn.Sequential | None = nn.Sequential(
                nn.Linear(concat_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_hidden_dim),
            )
            head_in = fusion_hidden_dim
        else:
            self.fusion = None
            head_in = concat_dim

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
            nn.Softplus(),
        )
        if max_rul is not None:
            init.xavier_uniform_(self.head[-2].weight)
            init.constant_(self.head[-2].bias, max_rul / 2.0)

    def forward(
        self,
        x: Tensor,
        theta_true: Tensor | None = None,
        ops: Tensor | None = None,
    ) -> Tensor:
        """Args:
            x:          (B, window_size, n_features)
            theta_true: (B, theta_true_dim) or None
            ops:        (B, window_size, ops_dim) or None
        Returns:
            rul: (B,)
        """
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, 2*hidden)
        else:
            h = h_n[-1]                                  # (B, hidden)

        if self.theta_true_dim > 0 and theta_true is not None:
            h = torch.cat([h, theta_true], dim=-1)

        if self.ops_enc is not None and ops is not None:
            h = torch.cat([h, self.ops_enc(ops)], dim=-1)

        if self.fusion is not None:
            h = self.fusion(h)

        out = self.head(h).squeeze(-1)
        if self.max_rul is not None:
            out = out.clamp(max=self.max_rul)
        return out
