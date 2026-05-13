"""Thermodynamic constants and station numbering for BraytonEngine (V3.1b).

Single source of truth for all magic numbers used by the differentiable
Brayton-cycle layer.  Pure constants only — no logic, no torch tensors,
no callable functions.

Reference: docs/CycleLayer_V3.1a_Master_Spec.md (Mai 2026, Rev 3.1a) § B.1.

Station numbering follows CMAPSS / SAE-ARP755 convention for a 2-spool
high-bypass turbofan:

    0  Ambient (ISA)              T0,  p0
    2  Fan Inlet (Ram)            T2,  P2,  m_dot_2
    21 Fan Outlet, Bypass         T21, P21, m_byp
    24 LPC Outlet                 T24, P24, m_core   [SENSOR]
    30 HPC Outlet                 T30, P30, Ps30     [SENSOR]
    4  Combustor Outlet           T4,  P4,  m_core+Wf
    45 HPT Outlet                 T45, P45
    50 LPT Outlet                 T50, P50           [SENSOR]
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Thermodynamic constants (SI)
# ---------------------------------------------------------------------------

GAMMA_C: float = 1.40       # heat-capacity ratio, cold side (air)
GAMMA_T: float = 1.33       # heat-capacity ratio, hot side  (combustion gas)
CP_C:    float = 1005.0     # J/(kg·K), air
CP_T:    float = 1150.0     # J/(kg·K), combustion gas
R_AIR:   float = 287.05     # J/(kg·K)
LHV:     float = 43.0e6     # J/kg, Jet-A lower heating value
ETA_COMB: float = 0.99      # combustor efficiency

# Reference state for corrected quantities (Sea Level Static, ISA)
T_REF: float = 288.15       # K
P_REF: float = 101325.0     # Pa

# Pre-computed exponents (gamma-1)/gamma — used in isentropic relations
EXP_C: float = (GAMMA_C - 1.0) / GAMMA_C   # 0.2857
EXP_T: float = (GAMMA_T - 1.0) / GAMMA_T   # 0.2481

# ---------------------------------------------------------------------------
# Soft bounds — applied via torch.clamp inside BraytonEngine
# ---------------------------------------------------------------------------

ETA_MIN: float = 0.50
ETA_MAX: float = 0.99
PR_MIN:  float = 1.05
PR_MAX:  float = 25.0

# ---------------------------------------------------------------------------
# Inlet (only used in ISA/Ram fallback path — V3.1a Patch P1)
# ---------------------------------------------------------------------------

ETA_INLET: float = 0.98     # pressure recovery; NEVER apply on measured P2

# ---------------------------------------------------------------------------
# Bypass and combustor pressure drop
# ---------------------------------------------------------------------------

BPR_DESIGN: float = 5.5     # CFM56-class commercial high-bypass turbofan
PI_BURN:    float = 0.04    # combustor pressure drop fraction (P4 = P30·(1-π_b))

# ---------------------------------------------------------------------------
# Component-specific design efficiencies (V3.1a Patch P8)
# ---------------------------------------------------------------------------
# Separated HPT vs LPT design efficiency — they differ in real engines.
# Multiplied by theta_phys (factor [0.85, 1.0]) to get effective eta.

ETA_DESIGN_FAN: float = 0.92
ETA_DESIGN_LPC: float = 0.90
ETA_DESIGN_HPC: float = 0.88
ETA_DESIGN_HPT: float = 0.90
ETA_DESIGN_LPT: float = 0.92

# ---------------------------------------------------------------------------
# Theta layout (factor representation, V3.1a Patch P4)
# ---------------------------------------------------------------------------
# theta_phys[..., i] is a multiplicative health modifier on component eta.
# Internal range: [0.85, 1.00].  1.00 = healthy.
# For comparison with N-CMAPSS GT (delta around 0): use (theta - 1.0).

N_THETA_PHYS: int = 5

THETA_NAMES: tuple[str, ...] = (
    "eta_fan",   # index 0
    "eta_lpc",   # index 1
    "eta_hpc",   # index 2
    "eta_hpt",   # index 3 — corresponds to HPT_eff_mod in N-CMAPSS GT
    "eta_lpt",   # index 4 — corresponds to LPT_eff_mod in N-CMAPSS GT
)

# Theta bounds in factor space
THETA_MIN: float = 0.85
THETA_MAX: float = 1.00

# AuxHealthHead bounds for lpt_flow_pred (delta around 0, V3.1a Patch P4)
LPT_FLOW_DELTA_MIN: float = -0.05
LPT_FLOW_DELTA_MAX: float = 0.02

# ---------------------------------------------------------------------------
# Plausibility limits for Stage 2 / V3.1b correction 5 turbine diagnostics
# ---------------------------------------------------------------------------
# Floor values below which a turbine outlet temperature would be implausible
# for a 2-spool high-bypass turbofan in normal operation.  Used by Stage 2
# soft checks and V3.1b turbine plausibility diagnostics.

T45_FLOOR_K: float = 800.0   # below this T45 is implausible (cold turbine)
T50_FLOOR_K: float = 600.0   # below this T50 is implausible (cold exhaust)

# Maximum plausible per-component temperature drop
DT_HPT_MAX_K: float = 600.0
DT_LPT_MAX_K: float = 500.0

# Plausibility ranges from Walsh & Fletcher / CMAPSS-class engines (Stage 2).
# Used by tests, NOT enforced by clamps — these are diagnostic checks.
PR_FAN_RANGE: tuple[float, float] = (1.4, 1.7)
PR_LPC_RANGE: tuple[float, float] = (1.5, 2.5)
PR_HPC_RANGE: tuple[float, float] = (8.0, 16.0)
PR_OVERALL_RANGE: tuple[float, float] = (20.0, 40.0)
T4_RANGE_K: tuple[float, float] = (1300.0, 1900.0)
ETA_PLAUSIBLE_RANGE: tuple[float, float] = (0.7, 0.99)
