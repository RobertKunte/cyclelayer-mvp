"""Differentiable Brayton Cycle Layer.

Models an ideal open Brayton cycle (gas turbine / jet engine) as a
differentiable PyTorch module.  Given thermodynamic cycle parameters θ
produced by the encoder, it computes physically-meaningful intermediate
quantities that serve as features for the prognostics head.

Ideal open Brayton cycle steps
-------------------------------
1. Isentropic compression  (1→2):  T2 = T1 * π^((γ-1)/γ) / η_c
2. Isentropic combustion   (2→3):  T3 = T2 + Q_in / (cp * ṁ)   (simplified)
3. Isentropic expansion    (3→4):  T4 = T3 * (1 - η_t * (1 - π^(-(γ-1)/γ)))
4. Heat rejection          (4→1):  back to ambient

Output features
---------------
For each sample the layer returns a fixed-size feature vector containing:
  - Specific work output W_net / (cp * T1)  (dimensionless)
  - Thermal efficiency η_th
  - Back-work ratio r_bw
  - Temperature ratios T2/T1, T3/T1, T4/T1
  - Pressure ratio π  (passed through)
  - Degradation proxy Δη = η_th_ideal - η_th
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BraytonCycleLayer(nn.Module):
    """Differentiable ideal Brayton cycle computation.

    Args:
        n_params: Number of input parameters θ from the encoder.
            Expected layout (all values > 0, already in physical range):
                0: T1  – inlet temperature [K]
                1: pi  – pressure ratio [-]
                2: T3  – turbine inlet temperature [K]
                3: eta_c – isentropic compressor efficiency (0, 1]
                4: eta_t – isentropic turbine efficiency (0, 1]
            If ``n_params > 5``, the remaining dimensions are passed through
            as-is and concatenated to the output features.
        gamma: Heat capacity ratio (default 1.4 for air).
        cp: Specific heat at constant pressure [J/(kg·K)] (default 1005).
        clamp_temperatures: If True, clamp intermediate temperatures to
            physically plausible ranges to improve numerical stability.
    """

    # Physically plausible bounds for clamping
    T_MIN, T_MAX = 200.0, 2500.0
    PI_MIN, PI_MAX = 1.01, 60.0
    ETA_MIN, ETA_MAX = 0.5, 1.0

    def __init__(
        self,
        n_params: int = 5,
        gamma: float = 1.4,
        cp: float = 1005.0,
        clamp_temperatures: bool = True,
    ) -> None:
        super().__init__()
        if n_params < 5:
            raise ValueError("n_params must be >= 5 (T1, π, T3, η_c, η_t).")
        self.n_params = n_params
        self.n_extra = n_params - 5
        self.gamma = gamma
        self.cp = cp
        self.clamp_temperatures = clamp_temperatures

        # Pre-compute constant exponent
        self._exp: float = (gamma - 1.0) / gamma  # (γ-1)/γ

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, theta: Tensor) -> Tensor:
        """Compute cycle feature vector from cycle parameters.

        Args:
            theta: Tensor of shape (..., n_params) with columns
                   [T1, π, T3, η_c, η_t, *extra].

        Returns:
            features: Tensor of shape (..., 8 + n_extra).
        """
        T1   = theta[..., 0:1]   # inlet temperature
        pi   = theta[..., 1:2]   # pressure ratio
        T3   = theta[..., 2:3]   # turbine inlet temperature
        eta_c = theta[..., 3:4]  # compressor efficiency
        eta_t = theta[..., 4:5]  # turbine efficiency

        if self.clamp_temperatures:
            T1    = T1.clamp(self.T_MIN, self.T_MAX)
            pi    = pi.clamp(self.PI_MIN, self.PI_MAX)
            T3    = T3.clamp(self.T_MIN, self.T_MAX)
            eta_c = eta_c.clamp(self.ETA_MIN, self.ETA_MAX)
            eta_t = eta_t.clamp(self.ETA_MIN, self.ETA_MAX)

        exp = self._exp

        # --- Compression (1 → 2) ---
        # Ideal: T2s = T1 * π^((γ-1)/γ)
        # Actual: T2 = T1 + (T2s - T1) / η_c
        T2s = T1 * pi.pow(exp)
        T2  = T1 + (T2s - T1) / eta_c

        # --- Expansion (3 → 4) ---
        # Ideal: T4s = T3 / π^((γ-1)/γ)
        # Actual: T4 = T3 - η_t * (T3 - T4s)
        T4s = T3 / pi.pow(exp)
        T4  = T3 - eta_t * (T3 - T4s)

        # --- Work & efficiency ---
        W_comp = self.cp * (T2 - T1)          # compressor specific work [J/kg]
        W_turb = self.cp * (T3 - T4)          # turbine specific work [J/kg]
        W_net  = W_turb - W_comp              # net specific work [J/kg]
        Q_in   = self.cp * (T3 - T2)          # heat addition [J/kg]

        # Thermal efficiency (clamp denominator for stability)
        Q_in_safe = Q_in.clamp(min=1e-3)
        eta_th = W_net / Q_in_safe

        # Back-work ratio
        W_turb_safe = W_turb.clamp(min=1e-3)
        r_bw = W_comp / W_turb_safe

        # Dimensionless net work
        W_net_norm = W_net / (self.cp * T1.clamp(min=1e-3))

        # Temperature ratios (dimensionless)
        T1_safe = T1.clamp(min=1e-3)
        tau2 = T2 / T1_safe
        tau3 = T3 / T1_safe
        tau4 = T4 / T1_safe

        features = torch.cat(
            [W_net_norm, eta_th, r_bw, tau2, tau3, tau4, pi, eta_th - (1.0 - 1.0 / pi.pow(exp))],
            dim=-1,
        )   # (..., 8)

        if self.n_extra > 0:
            features = torch.cat([features, theta[..., 5:]], dim=-1)

        return features  # (..., 8 + n_extra)

    # ------------------------------------------------------------------
    # Output dimensionality
    # ------------------------------------------------------------------

    @property
    def out_features(self) -> int:
        """Number of output features."""
        return 8 + self.n_extra
