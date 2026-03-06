"""Prognostics Head: Brayton cycle features → RUL prediction.

A lightweight MLP that takes the output of the BraytonCycleLayer and
produces a single scalar Remaining Useful Life (RUL) estimate.

Design choices:
- SiLU activations (smooth, non-saturating)
- Layer normalization for stable training with heterogeneous feature scales
- Softplus output to enforce non-negative RUL predictions
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PrognosticsHead(nn.Module):
    """MLP that maps cycle features to a RUL scalar.

    Args:
        in_features: Dimensionality of the input feature vector
            (= BraytonCycleLayer.out_features).
        hidden_sizes: Sequence of hidden layer widths.
        dropout: Dropout probability between hidden layers.
        max_rul: Soft upper bound applied via clamp on the output.
            Set to ``None`` to disable.
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        max_rul: float | None = 125.0,
    ) -> None:
        super().__init__()
        self.max_rul = max_rul

        layers: list[nn.Module] = []
        current = in_features
        for h in hidden_sizes:
            layers += [
                nn.LayerNorm(current),
                nn.Linear(current, h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            current = h

        layers += [
            nn.LayerNorm(current),
            nn.Linear(current, 1),
            nn.Softplus(),   # output > 0
        ]
        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------

    def forward(self, features: Tensor) -> Tensor:
        """Predict RUL from cycle feature vector.

        Args:
            features: Tensor of shape (B, in_features).

        Returns:
            rul: Tensor of shape (B,) with predicted RUL values.
        """
        out = self.net(features).squeeze(-1)  # (B,)
        if self.max_rul is not None:
            out = out.clamp(max=self.max_rul)
        return out
