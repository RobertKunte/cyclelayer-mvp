"""BraytonEngine — differentiable thermodynamic engine model (V3.1b).

A pure-physics nn.Module for a 2-spool high-bypass turbofan.  No ML
components live here.  The five health thetas (theta_phys) are inputs,
not parameters — they are produced upstream by ParamHead_phys in
CycleLayerV3.

Architecture follows docs/CycleLayer_V3.1a_Master_Spec.md (Mai 2026, Rev
3.1a) § B with the V3.1b operational corrections from review:

* P1: ETA_INLET only on ISA/Ram fallback, NEVER on measured P2.
* P2: estimate_inlet_flow() is explicit, YAML-parametrised.
* P3: turbine() returns 6-tuple inc. W_turbine and shaft_residual.
* P4: theta_phys is factor [0.85, 1.00] internally; theta_phys_as_delta
       helper returns delta around 0 for GT comparison.
* P8: numerical safety BEFORE the ** EXP_T power; PR-clamp activity
       in diagnostics; ETA_DESIGN_HPT/LPT separated.
* V3.1b correction 5: turbine plausibility diagnostics added.

The closure is explicit:
    W_HPT = W_HPC                              (HPT shaft balance)
    W_LPT = W_LPC + W_Fan_total                (LPT shaft balance)

Mass balance is per-construction:
    m_byp + m_core = m_in                       (after fan split)
    m_4 = m_core + Wf                           (combustor)

This module is fully differentiable.  No numerical solvers, no Python
branches on tensor values.  All bounds via torch.clamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from cyclelayer.models.stations import (
    BPR_DESIGN,
    CP_C,
    CP_T,
    DT_HPT_MAX_K,
    DT_LPT_MAX_K,
    ETA_COMB,
    ETA_DESIGN_FAN,
    ETA_DESIGN_HPC,
    ETA_DESIGN_HPT,
    ETA_DESIGN_LPC,
    ETA_DESIGN_LPT,
    ETA_INLET,
    ETA_MAX,
    ETA_MIN,
    EXP_C,
    EXP_T,
    GAMMA_C,
    LHV,
    N_THETA_PHYS,
    PI_BURN,
    PR_MAX,
    PR_MIN,
    P_REF,
    T45_FLOOR_K,
    T50_FLOOR_K,
    T_REF,
)


# ---------------------------------------------------------------------------
# Theta semantics helpers (V3.1a Patch P4)
# ---------------------------------------------------------------------------

def theta_phys_as_delta(theta_phys: Tensor) -> Tensor:
    """Convert internal factor representation to delta around 0 for GT
    comparison with N-CMAPSS HPT_eff_mod / LPT_eff_mod (which are deltas).
    """
    return theta_phys - 1.0


def theta_phys_from_factor(factor: Tensor) -> Tensor:
    """Identity (clarity wrapper for upstream code that may want this name)."""
    return factor


# ---------------------------------------------------------------------------
# Corrected-quantity helpers (Walsh & Fletcher convention)
# ---------------------------------------------------------------------------

def corrected_flow(m_dot: Tensor, T_in: Tensor, P_in: Tensor) -> Tensor:
    """Mass flow corrected to Sea-Level ISA reference."""
    return m_dot * torch.sqrt(T_in / T_REF) / (P_in / P_REF)


def corrected_speed(N_rpm: Tensor, T_in: Tensor) -> Tensor:
    """Rotational speed corrected to Sea-Level ISA reference (rpm)."""
    return N_rpm / torch.sqrt(T_in / T_REF)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InletFlowParams:
    """Parameters for estimate_inlet_flow() (V3.1a Patch P2).

    Design fields (`Wc_fan_design`, `Nc_fan_design`) are REQUIRED — no
    defaults — to force explicit configuration from YAML / Phase C0.
    Sensitivity (`c1`, `c2`) and safety bounds (`Wc_min`, `Wc_max`) keep
    generic-literature defaults.

    Phase C validates the fixed configuration against external references;
    no DS02 tuning is allowed.
    See docs/CycleLayer_V3.1b_Master_Spec.md § C.3.
    """

    # Required — must come from YAML (computed in Phase C0 from UserGuide FC02)
    Wc_fan_design: float
    Nc_fan_design: float

    # Generic literature sensitivity (Walsh & Fletcher / Kurzke shape)
    c1: float = 0.85
    c2: float = -0.20

    # Safety bounds
    Wc_min: float = 100.0
    Wc_max: float = 1100.0


@dataclass
class MapCoefficients:
    """Component map coefficients.

    Each component (fan / LPC / HPC) is described by:
      - a design point (corrected speed Nc_design_*, corrected flow
        Wc_design_*, design pressure ratio PR_design_*) — **REQUIRED**
      - generic literature sensitivity coefficients (pr_a*, eta_e*) —
        defaults are CMAPSS / Walsh-&-Fletcher generic shape parameters
        that are kept fixed.

    Phase C validates fixed configuration; no DS02 tuning allowed.
    Values come from Phase 0 (UserGuide FC02 anchor) and are then frozen.
    See docs/CycleLayer_V3.1b_Master_Spec.md § C.3.
    """

    # ── REQUIRED design points (per component) ────────────────────────────
    Nc_design_fan: float
    Wc_design_fan: float
    Nc_design_lpc: float
    Wc_design_lpc: float
    Nc_design_hpc: float
    Wc_design_hpc: float

    # ── Design pressure ratios (Walsh & Fletcher Tab. 5.3, CMAPSS-class) ──
    PR_design_fan: float
    PR_design_lpc: float
    PR_design_hpc: float

    # ── Design efficiencies, per component (REQUIRED, no hidden defaults) ──
    # Phase C0 / C0d set these EXPLICITLY. Generic literature defaults are
    # 0.92 / 0.90 / 0.88; documented C-MAPSS values are 0.8969 / 0.9148 /
    # 0.8615 (pending Frederick TM2007-215026 source verification).
    eta_design_fan: float
    eta_design_lpc: float
    eta_design_hpc: float

    # ── Generic literature sensitivity coefficients (defaults OK) ─────────
    # PR(Wc, Nc) ≈ PR_design * (1 + a1*dN + a2*dN^2 + b1*dW + b2*dW^2)
    # eta(Wc, Nc) ≈ eta_design * (1 - e1*dN^2 - e2*dW^2)   parabolic peak
    pr_a1: float = 0.10
    pr_a2: float = -0.05
    pr_b1: float = -0.08
    pr_b2: float = -0.03
    eta_e1: float = 0.05
    eta_e2: float = 0.03


@dataclass
class BraytonEngineConfig:
    """Top-level config for BraytonEngine.

    `inlet_flow` and `map_coeffs` are REQUIRED (no `default_factory`).
    Construction without them is an explicit error — V3.1b enforces that
    design configuration always comes from YAML or programmatic test
    fixtures, never from hidden in-code defaults.

    `use_measured_inlet=True` is the P1 default for DS02 (skip ETA_INLET
    on measured P2). Set False only for the ISA/Ram fallback.
    """

    # Required — explicit configuration only
    inlet_flow: InletFlowParams
    map_coeffs: MapCoefficients

    # Defaults OK — these are not engine-specific design points
    use_measured_inlet: bool = True
    bpr_design: float = BPR_DESIGN
    eta_design_hpt: float = ETA_DESIGN_HPT   # 0.90
    eta_design_lpt: float = ETA_DESIGN_LPT   # 0.92


# ---------------------------------------------------------------------------
# Inlet flow estimator (V3.1a Patch P2)
# ---------------------------------------------------------------------------

def estimate_inlet_flow(
    T2: Tensor,
    P2: Tensor,
    Nf_rpm: Tensor,
    params: InletFlowParams,
) -> Tensor:
    """Estimate total inlet mass flow from corrected fan speed.

    Wc_fan(Nc) = Wc_design * (1 + c1·dN + c2·dN²)   where dN = (Nc - Nc_d)/Nc_d
    m_in = Wc_fan · (P2/P_REF) / sqrt(T2/T_REF)

    Args:
        T2:     Total temperature at fan inlet (K).
        P2:     Total pressure at fan inlet (Pa).
        Nf_rpm: Fan/LP shaft speed (rpm).
        params: InletFlowParams (from BraytonEngineConfig).

    Returns:
        m_in: Total inlet mass flow (kg/s).
    """
    Nc = corrected_speed(Nf_rpm, T2)
    dN = (Nc - params.Nc_fan_design) / params.Nc_fan_design

    Wc_fan = params.Wc_fan_design * (
        1.0 + params.c1 * dN + params.c2 * dN ** 2
    )
    Wc_fan = torch.clamp(Wc_fan, params.Wc_min, params.Wc_max)

    m_in = Wc_fan * (P2 / P_REF) / torch.sqrt(T2 / T_REF)
    return m_in


# ---------------------------------------------------------------------------
# Parametric component maps (placeholder — fixed coefficients, Phase A/B)
# ---------------------------------------------------------------------------

def _parametric_compressor_map(
    Wc: Tensor,
    Nc: Tensor,
    coeffs: MapCoefficients,
    kind: str,
) -> tuple[Tensor, Tensor]:
    """Simplified parametric map for fan / LPC / HPC.

    Returns (PR_nominal, eta_nominal) before theta is applied.  All design
    constants (Nc_design_*, Wc_design_*, PR_design_*) come from `coeffs`,
    not from in-code defaults — Phase C validates fixed configuration; no
    DS02 tuning allowed.  See docs/CycleLayer_V3.1b_Master_Spec.md § C.3.
    """
    if kind == "fan":
        PR_design  = coeffs.PR_design_fan
        eta_design = coeffs.eta_design_fan
        Nc_design  = coeffs.Nc_design_fan
        Wc_design  = coeffs.Wc_design_fan
    elif kind == "lpc":
        PR_design  = coeffs.PR_design_lpc
        eta_design = coeffs.eta_design_lpc
        Nc_design  = coeffs.Nc_design_lpc
        Wc_design  = coeffs.Wc_design_lpc
    elif kind == "hpc":
        PR_design  = coeffs.PR_design_hpc
        eta_design = coeffs.eta_design_hpc
        Nc_design  = coeffs.Nc_design_hpc
        Wc_design  = coeffs.Wc_design_hpc
    else:
        raise ValueError(f"Unknown compressor kind: {kind!r}")

    dN = (Nc - Nc_design) / Nc_design
    dW = (Wc - Wc_design) / Wc_design

    PR = PR_design * (
        1.0 + coeffs.pr_a1 * dN + coeffs.pr_a2 * dN ** 2
            + coeffs.pr_b1 * dW + coeffs.pr_b2 * dW ** 2
    )
    eta = eta_design * (1.0 - coeffs.eta_e1 * dN ** 2 - coeffs.eta_e2 * dW ** 2)

    return PR, eta


# ---------------------------------------------------------------------------
# BraytonEngine
# ---------------------------------------------------------------------------

class BraytonEngine(nn.Module):
    """Differentiable Brayton-cycle engine model (V3.1b).

    Pure physics. No learnable parameters. The five thetas (theta_phys)
    are inputs, supplied by ParamHead_phys in CycleLayerV3.

    Args:
        config: BraytonEngineConfig — REQUIRED. All design points must be
            supplied explicitly (V3.1b: no hidden in-code defaults).
    """

    def __init__(self, config: BraytonEngineConfig) -> None:
        super().__init__()
        self.config = config
        # Buffers (so they move with .to(device) but are not learnable).
        # Map coefficients live in the config dataclass; access via
        # self.config.map_coeffs.

    # ------------------------------------------------------------------
    # Stage components
    # ------------------------------------------------------------------

    def _inlet(
        self,
        T2_meas: Tensor | None,
        P2_meas: Tensor | None,
        alt_m:   Tensor | None,
        mach:    Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Compute (T2, P2) at fan inlet.

        V3.1a Patch P1: ETA_INLET is applied ONLY in the ISA/Ram fallback.
        On the measured path, P2_meas is already total at the fan inlet.
        """
        if self.config.use_measured_inlet:
            assert T2_meas is not None and P2_meas is not None, (
                "use_measured_inlet=True requires T2_meas and P2_meas"
            )
            return T2_meas, P2_meas
        # ISA/Ram fallback path
        assert alt_m is not None and mach is not None, (
            "use_measured_inlet=False requires alt_m and mach"
        )
        T0 = 288.15 - 0.0065 * alt_m
        p0 = 101325.0 * (T0 / 288.15) ** 5.2561
        ram_T = 1.0 + 0.5 * (GAMMA_C - 1.0) * mach ** 2
        T2 = T0 * ram_T
        P2 = p0 * ram_T ** (GAMMA_C / (GAMMA_C - 1.0))
        P2 = P2 * ETA_INLET
        return T2, P2

    def _fan(
        self,
        T2: Tensor, P2: Tensor, m_in: Tensor, Nf_rpm: Tensor,
        theta_eta_fan: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Single fan call on TOTAL inlet mass flow (V3.1 correction)."""
        Wc = corrected_flow(m_in, T2, P2)
        Nc = corrected_speed(Nf_rpm, T2)
        PR, eta_nom = _parametric_compressor_map(
            Wc, Nc, self.config.map_coeffs, kind="fan"
        )
        eta = torch.clamp(eta_nom * theta_eta_fan, ETA_MIN, ETA_MAX)
        PR  = torch.clamp(PR, PR_MIN, PR_MAX)

        T21_isen = T2 * PR ** EXP_C
        T21      = T2 + (T21_isen - T2) / eta
        P21      = P2 * PR
        W_fan_total = m_in * CP_C * (T21 - T2)
        return T21, P21, W_fan_total, eta, PR

    def _compressor(
        self,
        T_in: Tensor, P_in: Tensor, m_dot: Tensor, N_rpm: Tensor,
        theta_eta: Tensor,
        kind: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generic axial compressor — used for LPC and HPC."""
        Wc = corrected_flow(m_dot, T_in, P_in)
        Nc = corrected_speed(N_rpm, T_in)
        PR, eta_nom = _parametric_compressor_map(
            Wc, Nc, self.config.map_coeffs, kind=kind
        )
        eta = torch.clamp(eta_nom * theta_eta, ETA_MIN, ETA_MAX)
        PR  = torch.clamp(PR, PR_MIN, PR_MAX)

        T_out_isen = T_in * PR ** EXP_C
        T_out      = T_in + (T_out_isen - T_in) / eta
        P_out      = P_in * PR
        W_comp     = m_dot * CP_C * (T_out - T_in)
        return T_out, P_out, W_comp, eta, PR

    def _combustor(
        self,
        T30: Tensor, P30: Tensor, m_core: Tensor, Wf: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Energy balance combustor; pressure drop pi_burn fixed."""
        m_4 = m_core + Wf
        T4  = (m_core * CP_C * T30 + Wf * LHV * ETA_COMB) / (m_4 * CP_T)
        P4  = P30 * (1.0 - PI_BURN)
        return T4, P4, m_4

    def _turbine(
        self,
        T_in: Tensor, P_in: Tensor, m_dot: Tensor,
        W_required: Tensor,
        theta_eta: Tensor,
        eta_design: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Inverse-PR turbine with explicit closure (V3.1a Patches P3 + P8).

        Returns:
            (T_out, P_out, W_turbine, shaft_residual, eta, PR)

        The closure is enforced by construction: dT = W_required / (m·cp_t).
        eta drives the PR via dT_isen = dT / eta.
        """
        eta = torch.clamp(theta_eta * eta_design, ETA_MIN, ETA_MAX)

        # Actual temperature drop from energy balance
        dT     = W_required / (m_dot * CP_T)
        T_out  = T_in - dT

        # Isentropic temperature drop
        dT_isen    = dT / eta
        T_out_isen = T_in - dT_isen

        # V3.1a Patch P8 — numerical safety BEFORE the ** EXP_T power.
        # Without these clamps, random batches in the gradient-stability
        # test (Stage 5) produce NaN before reaching the PR clamp.
        eps = 1e-6
        T_out_isen_safe = torch.clamp(T_out_isen, min=0.05 * T_in)
        ratio_in        = torch.clamp(
            T_in / T_out_isen_safe,
            min=1.0 + eps,
            max=PR_MAX ** EXP_T,
        )

        PR    = ratio_in ** (1.0 / EXP_T)
        PR    = torch.clamp(PR, PR_MIN, PR_MAX)
        P_out = P_in / PR

        W_turbine      = m_dot * CP_T * (T_in - T_out)
        shaft_residual = W_turbine - W_required   # ≈ 0 by construction

        return T_out, P_out, W_turbine, shaft_residual, eta, PR

    # ------------------------------------------------------------------
    # forward()
    # ------------------------------------------------------------------

    def forward(
        self,
        ops_si:     Mapping[str, Tensor],
        sens_si:    Mapping[str, Tensor],
        theta_phys: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Compute predicted output sensors and diagnostics.

        Args:
            ops_si:
                Required keys: T2_K, P2_Pa.
                Optional keys (used in fallback path): alt_m, mach, TRA.
            sens_si:
                Required keys: Nf_rpm, Nc_rpm, Wf_kgs.
            theta_phys:
                Tensor of shape (..., 5). Order:
                    [eta_fan, eta_lpc, eta_hpc, eta_hpt, eta_lpt].
                Each value is a factor in [0.85, 1.00] (1.00 = healthy).

        Returns:
            sensors_pred_si:
                Dict with keys T24_K, T30_K, P30_Pa, T50_K — the four
                BraytonEngine output sensors (specified in V3.1a § A.2).
            diagnostics:
                Dict with mass and shaft balance residuals, component PRs,
                effective etas, internal-station temperatures and pressures,
                PR-clamp-fraction (P8), and turbine plausibility metrics
                (V3.1b correction 5).  Used by tests and TensorBoard logging.
        """
        if theta_phys.shape[-1] != N_THETA_PHYS:
            raise ValueError(
                f"theta_phys last dim must be {N_THETA_PHYS}, got {theta_phys.shape[-1]}"
            )

        # Inlet (P1: skip ETA_INLET on measured path)
        T2, P2 = self._inlet(
            T2_meas=ops_si.get("T2_K"),
            P2_meas=ops_si.get("P2_Pa"),
            alt_m=ops_si.get("alt_m"),
            mach=ops_si.get("mach"),
        )
        Nf, Nc, Wf = sens_si["Nf_rpm"], sens_si["Nc_rpm"], sens_si["Wf_kgs"]

        # Inlet mass flow (P2: explicit YAML-parametrised)
        m_in = estimate_inlet_flow(T2, P2, Nf, self.config.inlet_flow)

        # Fan — single call on total mass flow
        T21, P21, W_fan_total, eta_fan, PR_fan = self._fan(
            T2, P2, m_in, Nf, theta_phys[..., 0]
        )

        # Bypass split AFTER fan
        m_byp  = m_in * self.config.bpr_design / (self.config.bpr_design + 1.0)
        m_core = m_in - m_byp
        W_fan_core = W_fan_total * m_core / m_in
        W_fan_byp  = W_fan_total * m_byp  / m_in

        # Core compressors (LPC on Nf, HPC on Nc)
        T24, P24, W_lpc, eta_lpc, PR_lpc = self._compressor(
            T21, P21, m_core, Nf, theta_phys[..., 1], kind="lpc"
        )
        T30, P30, W_hpc, eta_hpc, PR_hpc = self._compressor(
            T24, P24, m_core, Nc, theta_phys[..., 2], kind="hpc"
        )

        # Combustor
        T4, P4, m_4 = self._combustor(T30, P30, m_core, Wf)

        # Turbines via explicit closure
        # HPT: W_HPT = W_HPC
        T45, P45, W_hpt, hpt_residual, eta_hpt, PR_hpt = self._turbine(
            T4, P4, m_4,
            W_required=W_hpc,
            theta_eta=theta_phys[..., 3],
            eta_design=self.config.eta_design_hpt,
        )
        # LPT: W_LPT = W_LPC + W_Fan_total
        W_lpt_required = W_lpc + W_fan_total
        T50, P50, W_lpt, lpt_residual, eta_lpt, PR_lpt = self._turbine(
            T45, P45, m_4,
            W_required=W_lpt_required,
            theta_eta=theta_phys[..., 4],
            eta_design=self.config.eta_design_lpt,
        )

        sensors_pred_si: dict[str, Tensor] = {
            "T24_K": T24, "T30_K": T30, "P30_Pa": P30, "T50_K": T50,
        }

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        diagnostics: dict[str, Tensor] = {
            # Conservation residuals (Stage 1)
            "mass_balance_inlet":   (m_in - (m_byp + m_core)).abs(),
            "mass_balance_combust": (m_4  - (m_core + Wf)).abs(),
            "shaft_HPT_residual":   hpt_residual,
            "shaft_LPT_residual":   lpt_residual,

            # Component work
            "W_fan_total": W_fan_total, "W_fan_core": W_fan_core,
            "W_fan_byp":   W_fan_byp,
            "W_lpc": W_lpc, "W_hpc": W_hpc,
            "W_hpt": W_hpt, "W_lpt": W_lpt,

            # Pressure ratios
            "PR_fan": PR_fan, "PR_lpc": PR_lpc, "PR_hpc": PR_hpc,
            "PR_hpt": PR_hpt, "PR_lpt": PR_lpt,

            # Internal stations
            "T4": T4, "P4": P4, "m_4": m_4,
            "T45": T45, "P45": P45,
            "P50": P50,
            "m_in": m_in, "m_core": m_core, "m_byp": m_byp,

            # Effective etas
            "eta_fan": eta_fan, "eta_lpc": eta_lpc, "eta_hpc": eta_hpc,
            "eta_hpt": eta_hpt, "eta_lpt": eta_lpt,

            # Overall metrics
            "P30_over_P2": P30 / P2,

            # PR-clamp-fraction monitoring (V3.1a Patch P8)
            "frac_PR_fan_clamped": ((PR_fan == PR_MIN) | (PR_fan == PR_MAX)).float().mean(),
            "frac_PR_lpc_clamped": ((PR_lpc == PR_MIN) | (PR_lpc == PR_MAX)).float().mean(),
            "frac_PR_hpc_clamped": ((PR_hpc == PR_MIN) | (PR_hpc == PR_MAX)).float().mean(),
            "frac_PR_hpt_clamped": ((PR_hpt == PR_MIN) | (PR_hpt == PR_MAX)).float().mean(),
            "frac_PR_lpt_clamped": ((PR_lpt == PR_MIN) | (PR_lpt == PR_MAX)).float().mean(),

            # Turbine plausibility (V3.1b correction 5)
            "min_T45": T45.min(),
            "min_T50": T50.min(),
            "frac_T45_below_limit":   (T45 < T45_FLOOR_K).float().mean(),
            "frac_T50_below_limit":   (T50 < T50_FLOOR_K).float().mean(),
            "frac_dT_hpt_over_limit": ((T4  - T45) > DT_HPT_MAX_K).float().mean(),
            "frac_dT_lpt_over_limit": ((T45 - T50) > DT_LPT_MAX_K).float().mean(),
        }
        return sensors_pred_si, diagnostics

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config_dict(cls, d: dict[str, Any]) -> "BraytonEngine":
        """Construct from a plain dict (e.g. loaded from YAML).

        Required keys under `inlet_flow`:
            Wc_fan_design, Nc_fan_design
        Required keys under `map_coeffs`:
            Nc_design_fan, Wc_design_fan,
            Nc_design_lpc, Wc_design_lpc,
            Nc_design_hpc, Wc_design_hpc,
            PR_design_fan, PR_design_lpc, PR_design_hpc

        Missing required fields raise TypeError with the dataclass field name —
        the YAML must be filled out before construction succeeds.  See
        configs/cyclelayer_v3.yaml.
        """
        inlet_d = d.get("inlet_flow", {}) or {}
        maps_d  = d.get("map_coeffs", {}) or {}
        cfg = BraytonEngineConfig(
            use_measured_inlet=d.get("use_measured_inlet", True),
            bpr_design=d.get("bpr_design", BPR_DESIGN),
            eta_design_hpt=d.get("eta_design_hpt", ETA_DESIGN_HPT),
            eta_design_lpt=d.get("eta_design_lpt", ETA_DESIGN_LPT),
            inlet_flow=InletFlowParams(**inlet_d),
            map_coeffs=MapCoefficients(**maps_d),
        )
        return cls(cfg)
