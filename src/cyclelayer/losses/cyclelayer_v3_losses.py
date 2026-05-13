"""CycleLayerV3 thermal-auxiliary loss (V3.1b).

Composite loss for the re-scoped V3.1b model. See
[docs/decisions/ADR-0012-v3-thermal-auxiliary-scope.md] and
[docs/V3_thermal_auxiliary_plan.md] for the design context.

L_total = λ_rul · L_rul                   # asymmetric RUL
        + λ_temp · L_temp_sensor          # T24/T30/T50 ONLY (normalized)
        + λ_aux · L_aux_lpt_flow          # supervised, small, normalized
        + λ_healthy · L_healthy_prior     # weak ||θ − 1||² for RUL > threshold
        + λ_smooth · L_smooth             # temporal Δ² on θ_phys

Hard tripwires:
    * `use_pressure_loss == False`        — asserted at construction
    * `use_epr_loss == False`             — asserted at construction
    * `temp_sensors` ⊂ {"T24", "T30", "T50"} — asserted at construction
    * pressure / EPR keys must not appear in `temp_sensors` or anywhere else
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


ALLOWED_TEMP_SENSORS: frozenset[str] = frozenset({"T24", "T30", "T50"})
FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "P30", "Ps30", "P50", "P2", "P21", "P24", "P45",
    "EPR", "epr", "pressure",
})


# =============================================================================
# Config
# =============================================================================

@dataclass
class V3LossConfig:
    """Configuration for `CycleLayerV3Loss`.

    The `use_pressure_loss` and `use_epr_loss` flags MUST stay False in
    V3.1b.  Construction asserts this — there is no path to enable either
    without changing the code.  See ADR-0012.

    `temp_sensors` MUST be a subset of {"T24", "T30", "T50"}.  Construction
    rejects anything else (e.g. "P30", "EPR").
    """

    # Weights
    lambda_rul:     float = 1.0
    lambda_temp:    float = 0.1
    lambda_aux:     float = 0.02
    lambda_healthy: float = 0.05
    lambda_smooth:  float = 1.0e-3

    # Asymmetric RUL loss
    mse_weight: float = 1.0
    asymmetry:  float = 0.1

    # L_temp targets — must be a subset of ALLOWED_TEMP_SENSORS
    temp_sensors: list[str] = field(default_factory=lambda: ["T24", "T30", "T50"])

    # Per-sensor sigma for temperature normalization.  Computed from train
    # split at script startup; if None at construction, falls back to a
    # uniform σ=1.0 (which means raw-K MSE).  Smoke script computes proper
    # σ values from train data.
    sigma_temp_K: dict[str, float] | None = None

    # AuxHead sigma for L_aux normalization.  Computed from train split if
    # available, else 0.02 per spec.
    sigma_lpt_flow: float = 0.02

    # Healthy prior — applied only when RUL > threshold
    healthy_rul_threshold: float = 80.0

    # FORBIDDEN flags (V3.1b — see ADR-0012)
    use_pressure_loss: bool = False
    use_epr_loss:      bool = False


# =============================================================================
# Loss module
# =============================================================================

class CycleLayerV3Loss(nn.Module):
    """Composite thermal-auxiliary loss for V3.1b.

    Inputs to forward:
        rul_pred:       (B,)
        rul_true:       (B,)
        theta_phys:     (B, [T,] 5)        — for L_healthy + L_smooth
        lpt_flow_pred:  (B,) or (B, T)     — AuxHead output for L_aux
        lpt_flow_true:  (B,) or (B, T) or None — GT for L_aux
        temp_preds_K:   dict {"T24_K": (B,), "T30_K": (B,), "T50_K": (B,)}
        temp_true_K:    dict {"T24_K": (B,), "T30_K": (B,), "T50_K": (B,)}

    Returns:
        total: scalar loss
        components: dict[str, Tensor] with detached component scalars
    """

    def __init__(self, config: V3LossConfig) -> None:
        super().__init__()
        # ── Tripwires ────────────────────────────────────────────────────
        if config.use_epr_loss:
            raise ValueError(
                "V3.1b: use_epr_loss=True is FORBIDDEN. "
                "EPR validation moves to V4 — see ADR-0012."
            )
        if config.use_pressure_loss:
            raise ValueError(
                "V3.1b: use_pressure_loss=True is FORBIDDEN. "
                "Pressure validation moves to V4 — see ADR-0012."
            )

        bad_temp = [s for s in config.temp_sensors if s not in ALLOWED_TEMP_SENSORS]
        if bad_temp:
            raise ValueError(
                f"V3.1b: temp_sensors must be a subset of {sorted(ALLOWED_TEMP_SENSORS)}; "
                f"got forbidden entries {bad_temp}. Pressure-like keys are not allowed."
            )
        for k in config.temp_sensors:
            for forb in FORBIDDEN_KEYS:
                if forb.lower() in k.lower():
                    raise ValueError(
                        f"V3.1b: temp_sensors entry {k!r} contains forbidden token {forb!r}"
                    )
        if not config.temp_sensors:
            raise ValueError("V3.1b: temp_sensors must not be empty.")

        self.config = config

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _rul_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Asymmetric MSE: penalises over-estimation harder than under-estimation."""
        mse = F.mse_loss(pred, target, reduction="mean")
        if self.config.asymmetry > 0.0:
            over = F.relu(pred - target)
            asym = (over ** 2).mean()
            return self.config.mse_weight * mse + self.config.asymmetry * asym
        return self.config.mse_weight * mse

    def _temp_loss(
        self,
        temp_preds_K: dict[str, Tensor],
        temp_true_K:  dict[str, Tensor],
    ) -> Tensor:
        """Per-sensor σ-normalized MSE on T24 / T30 / T50 ONLY.

        Pressure / EPR keys are forbidden — if they appear in the dicts
        we raise so a future code change can't silently leak them in.
        """
        # Forbidden-key guard
        for k in list(temp_preds_K.keys()) + list(temp_true_K.keys()):
            kn = k.replace("_K", "").replace("_R", "")
            if kn not in ALLOWED_TEMP_SENSORS:
                if any(forb.lower() in k.lower() for forb in FORBIDDEN_KEYS):
                    raise ValueError(
                        f"V3.1b: forbidden pressure/EPR key in temp dict: {k!r}"
                    )
                # ignore non-target keys silently (T48 etc. can be present as diagnostic)

        terms: list[Tensor] = []
        for sname in self.config.temp_sensors:
            key = f"{sname}_K"
            if key not in temp_preds_K or key not in temp_true_K:
                continue
            p = temp_preds_K[key]
            t = temp_true_K[key]
            sigma = 1.0
            if self.config.sigma_temp_K is not None:
                sigma = float(self.config.sigma_temp_K.get(sname, 1.0))
            sigma = max(sigma, 1e-6)
            terms.append(((p - t) / sigma).pow(2).mean())
        if not terms:
            # No matching sensors found — return a finite zero on the correct device.
            ref = next(iter(temp_preds_K.values())) if temp_preds_K else None
            return torch.zeros((), device=ref.device if ref is not None else None)
        return torch.stack(terms).mean()

    def _aux_loss(
        self,
        lpt_flow_pred: Tensor,
        lpt_flow_true: Tensor,
    ) -> Tensor:
        """Normalized MSE on LPT_flow_mod: MSE((pred-GT)/σ_lpt_flow)."""
        sigma = max(float(self.config.sigma_lpt_flow), 1e-6)
        return ((lpt_flow_pred - lpt_flow_true) / sigma).pow(2).mean()

    def _healthy_prior(
        self,
        theta_phys: Tensor,        # (B, [T,] 5)
        rul_true:   Tensor,        # (B,)
    ) -> Tensor:
        """Weak prior: at high RUL, theta should be near 1.0.

        `mean(||θ − 1||²)` averaged over (sample, time, theta_dim) only on
        rows with `RUL > threshold`.  Returns 0 if no such rows in the batch.
        """
        mask = rul_true > self.config.healthy_rul_threshold
        if not mask.any():
            return torch.zeros((), device=theta_phys.device)
        # theta_phys is (B, [T,] 5).  Broadcast the per-sample mask.
        diff = (theta_phys - 1.0) ** 2
        if diff.dim() == 3:
            # (B, T, 5)
            diff_per_sample = diff.mean(dim=(1, 2))   # (B,)
        else:
            diff_per_sample = diff.mean(dim=-1)       # (B,)
        return diff_per_sample[mask].mean()

    def _smooth_loss(self, theta_phys: Tensor) -> Tensor:
        """Temporal Δ² smoothness on theta_phys.

        Requires theta_phys with a time axis: (B, T, 5).  If the model
        produces a per-window scalar theta (B, 5), smoothness is 0.
        """
        if theta_phys.dim() != 3 or theta_phys.shape[1] < 2:
            return torch.zeros((), device=theta_phys.device)
        diff = theta_phys[:, 1:, :] - theta_phys[:, :-1, :]
        return diff.pow(2).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        rul_pred:       Tensor,
        rul_true:       Tensor,
        theta_phys:     Tensor,
        lpt_flow_pred:  Tensor | None = None,
        lpt_flow_true:  Tensor | None = None,
        temp_preds_K:   dict[str, Tensor] | None = None,
        temp_true_K:    dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        cfg = self.config
        comps: dict[str, Tensor] = {}
        total = torch.zeros((), device=rul_pred.device, dtype=rul_pred.dtype)

        # L_rul (always)
        L_rul = self._rul_loss(rul_pred, rul_true)
        total = total + cfg.lambda_rul * L_rul
        comps["rul"] = L_rul.detach()

        # L_temp
        if cfg.lambda_temp > 0.0 and temp_preds_K is not None and temp_true_K is not None:
            L_temp = self._temp_loss(temp_preds_K, temp_true_K)
            total = total + cfg.lambda_temp * L_temp
            comps["temp"] = L_temp.detach()

        # L_aux
        if (cfg.lambda_aux > 0.0 and lpt_flow_pred is not None
                and lpt_flow_true is not None):
            L_aux = self._aux_loss(lpt_flow_pred, lpt_flow_true)
            total = total + cfg.lambda_aux * L_aux
            comps["aux"] = L_aux.detach()

        # L_healthy
        if cfg.lambda_healthy > 0.0:
            L_healthy = self._healthy_prior(theta_phys, rul_true)
            total = total + cfg.lambda_healthy * L_healthy
            comps["healthy"] = L_healthy.detach()

        # L_smooth
        if cfg.lambda_smooth > 0.0:
            L_smooth = self._smooth_loss(theta_phys)
            total = total + cfg.lambda_smooth * L_smooth
            comps["smooth"] = L_smooth.detach()

        return total, comps
