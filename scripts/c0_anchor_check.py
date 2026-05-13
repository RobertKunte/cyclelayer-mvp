"""C0 anchor check — one-shot worksheet (V3.1b Phase C0).

Purpose
-------
Run a SINGLE BraytonEngine forward pass at UserGuide Tab. 1.4 FC02 with
theta=1.0 and report the residuals for review. NO LOOPS. NO FIT HELPERS.
NO DS02 ACCESS. The single manually-selected parameter is `Wc_fan_design`;
all other design points are derived from it under the fixed assumptions
documented at the top of this file.

If you find yourself tempted to wrap this in a sweep / iteration / minimizer:
STOP. That is a tuning loop. Hard Rule 17 forbids it. Report residuals,
escalate to Robert.

Usage
-----
    python scripts/c0_anchor_check.py

This script is read-only with respect to repo state. It does NOT write to
configs/cyclelayer_v3.yaml. YAML population happens AFTER Robert reviews
the residuals printed here.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

# Ensure src/ on path when run from repo root
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from cyclelayer.data.ncmapss_v3 import load_userguide_fc02_anchor  # noqa: E402
from cyclelayer.models import units  # noqa: E402
from cyclelayer.models.brayton_engine import (  # noqa: E402
    BraytonEngine,
    BraytonEngineConfig,
    InletFlowParams,
    MapCoefficients,
    corrected_speed,
)
from cyclelayer.models.stations import (  # noqa: E402
    BPR_DESIGN,
    CP_C,
    ETA_DESIGN_FAN,
    ETA_DESIGN_HPC,
    ETA_DESIGN_HPT,
    ETA_DESIGN_LPC,
    ETA_DESIGN_LPT,
    EXP_C,
    GAMMA_C,
    P_REF,
    T_REF,
)


# =============================================================================
# Step 1 — Fixed assumptions (frozen for V3.1b; same across all of Phase C)
# =============================================================================
# Walsh & Fletcher Tab. 5.3 / Kurzke generic / Frederick 2007.
# These are CONSCIOUS DESIGN CHOICES, not silent defaults.
# They are inputs to the FC02 anchor procedure, NOT solved by it.

ASSUMPTION_BPR              = BPR_DESIGN          # 5.5  (CMAPSS-90K-class generic)
ASSUMPTION_PR_DESIGN_FAN    = 1.6                 # Walsh & Fletcher generic
ASSUMPTION_PR_DESIGN_LPC    = 2.0                 # Walsh & Fletcher generic
ASSUMPTION_PR_DESIGN_HPC    = 12.0                # CMAPSS-class generic
ASSUMPTION_ETA_DESIGN_FAN   = ETA_DESIGN_FAN      # 0.92
ASSUMPTION_ETA_DESIGN_LPC   = ETA_DESIGN_LPC      # 0.90
ASSUMPTION_ETA_DESIGN_HPC   = ETA_DESIGN_HPC      # 0.88
ASSUMPTION_ETA_DESIGN_HPT   = ETA_DESIGN_HPT      # 0.90
ASSUMPTION_ETA_DESIGN_LPT   = ETA_DESIGN_LPT      # 0.92


# =============================================================================
# Step 2 — Single manually selected parameter
# =============================================================================
# Engineering rationale (NOT a derivation from Wf and Nf):
#  - CMAPSS reference engine is 90K-thrust class (Frederick 2007)
#  - Walsh & Fletcher generic fan corrected flow at design ~ 100-1500 kg/s for
#    high-bypass turbofans; CMAPSS-90K sits upper-mid
#  - Cross-check vs FC02 fuel flow (Wf=7.085 pps -> 3.21 kg/s) under typical
#    takeoff F/A ratio ~ 0.022 implies m_core ~ 145 kg/s -> m_in ~ 940 kg/s
#  -> pick Wc_fan_design = 900 kg/s (mid of the implied window)

PICKED_WC_FAN_DESIGN = 900.0   # kg/s


# =============================================================================
# Step 3 — FC02 conditions in SI (after ram correction at M=0.25)
# =============================================================================

def fc02_conditions_si() -> dict[str, float]:
    """Compute T2, P2 at FC02 from Tsl + Mach (ISA Standard-Day + ram).

    NOTE: BraytonEngine runs in `use_measured_inlet=True`, so T2 / P2
    passed in must already be the total values at fan inlet — i.e. ram
    correction is APPLIED here, not later by the engine.
    """
    fc = load_userguide_fc02_anchor()
    Tsl_R = fc["Tsl_F"] + 459.67    # 59 + 459.67 = 518.67 R (ISA SL Std-Day)
    P0_psia = 14.696                 # ISA SL static pressure
    M = fc["XM"]

    # Ram correction (isentropic)
    ram_T = 1.0 + 0.5 * (GAMMA_C - 1.0) * M ** 2
    ram_P = ram_T ** (GAMMA_C / (GAMMA_C - 1.0))

    T2_R = Tsl_R * ram_T            # 518.67 * 1.0125 = 525.16 R
    P2_psia = P0_psia * ram_P       # 14.696 * 1.044 ~ 15.35 psia

    return {
        "alt_ft":  float(fc["alt_ft"]),
        "XM":      float(fc["XM"]),
        "TRA_pct": float(fc["TRA_pct"]),
        "T2_R":    T2_R,
        "P2_psia": P2_psia,
        "Nf_rpm":  float(fc["Nf_rpm"]),
        "Nc_rpm":  float(fc["Nc_rpm"]),
        "Wf_pps":  float(fc["Wf_pps"]),
        "T48_ref_R":  float(fc["T48_ref_R"]),
        "EPR_ref":    float(fc["EPR_ref"]),
    }


# =============================================================================
# Step 4 — Derived design points (from PICKED_WC_FAN_DESIGN + fixed assumptions)
# =============================================================================
# These are NOT additional manual choices — they follow mechanically from
# Step 1 + Step 2 + FC02 conditions.

def derive_design_points(fc02_si: dict[str, float], Wc_fan_design: float) -> dict[str, float]:
    """Derive Nc_design_*, Wc_design_lpc/hpc from the single chosen Wc_fan_design.

    Walks fan -> LPC -> HPC at the design point (FC02 with theta=1.0) using
    the fixed PR_design and eta_design assumptions. No solver, no iteration,
    closed-form arithmetic.
    """
    T2_K = fc02_si["T2_R"] * units.RANK_TO_K
    P2_Pa = fc02_si["P2_psia"] * units.PSIA_TO_PA

    # Nc_design_fan: at FC02 the corrected fan speed equals Nf / sqrt(T2/T_REF)
    Nc_design_fan = fc02_si["Nf_rpm"] / math.sqrt(T2_K / T_REF)

    # m_in_design from inlet_flow estimator at design point (dN = 0)
    # m_in = Wc_fan_design * (P2/P_REF) / sqrt(T2/T_REF)
    m_in_design = Wc_fan_design * (P2_Pa / P_REF) / math.sqrt(T2_K / T_REF)
    m_core_design = m_in_design / (ASSUMPTION_BPR + 1.0)

    # Fan thermodynamics at design
    T21_isen = T2_K * ASSUMPTION_PR_DESIGN_FAN ** EXP_C
    T21      = T2_K + (T21_isen - T2_K) / ASSUMPTION_ETA_DESIGN_FAN
    P21      = P2_Pa * ASSUMPTION_PR_DESIGN_FAN

    # Wc_design_lpc: corrected_flow at LPC inlet (m_core, T21, P21)
    Wc_design_lpc = m_core_design * math.sqrt(T21 / T_REF) / (P21 / P_REF)
    # Nc_design_lpc: LP shaft, same physical speed as fan
    Nc_design_lpc = Nc_design_fan

    # LPC thermodynamics at design
    T24_isen = T21 * ASSUMPTION_PR_DESIGN_LPC ** EXP_C
    T24      = T21 + (T24_isen - T21) / ASSUMPTION_ETA_DESIGN_LPC
    P24      = P21 * ASSUMPTION_PR_DESIGN_LPC

    # Wc_design_hpc: corrected_flow at HPC inlet
    Wc_design_hpc = m_core_design * math.sqrt(T24 / T_REF) / (P24 / P_REF)
    # Nc_design_hpc: HP shaft corrected to T24
    Nc_design_hpc = fc02_si["Nc_rpm"] / math.sqrt(T24 / T_REF)

    return {
        "Nc_design_fan":  Nc_design_fan,
        "Nc_design_lpc":  Nc_design_lpc,
        "Nc_design_hpc":  Nc_design_hpc,
        "Wc_design_fan":  Wc_fan_design,
        "Wc_design_lpc":  Wc_design_lpc,
        "Wc_design_hpc":  Wc_design_hpc,
        "PR_design_fan":  ASSUMPTION_PR_DESIGN_FAN,
        "PR_design_lpc":  ASSUMPTION_PR_DESIGN_LPC,
        "PR_design_hpc":  ASSUMPTION_PR_DESIGN_HPC,
        "eta_design_fan": ASSUMPTION_ETA_DESIGN_FAN,
        "eta_design_lpc": ASSUMPTION_ETA_DESIGN_LPC,
        "eta_design_hpc": ASSUMPTION_ETA_DESIGN_HPC,
        # Side outputs for diagnostics
        "_m_in_design":   m_in_design,
        "_m_core_design": m_core_design,
        "_T21_design_K":  T21,
        "_T24_design_K":  T24,
        "_P21_design_Pa": P21,
        "_P24_design_Pa": P24,
    }


# =============================================================================
# Step 5 — Build BraytonEngine and run forward ONCE at FC02
# =============================================================================

def build_engine(design: dict[str, float]) -> BraytonEngine:
    cfg = BraytonEngineConfig(
        inlet_flow=InletFlowParams(
            Wc_fan_design=design["Wc_design_fan"],
            Nc_fan_design=design["Nc_design_fan"],
        ),
        map_coeffs=MapCoefficients(
            Nc_design_fan=design["Nc_design_fan"],
            Wc_design_fan=design["Wc_design_fan"],
            Nc_design_lpc=design["Nc_design_lpc"],
            Wc_design_lpc=design["Wc_design_lpc"],
            Nc_design_hpc=design["Nc_design_hpc"],
            Wc_design_hpc=design["Wc_design_hpc"],
            PR_design_fan=design["PR_design_fan"],
            PR_design_lpc=design["PR_design_lpc"],
            PR_design_hpc=design["PR_design_hpc"],
        ),
        use_measured_inlet=True,
        bpr_design=ASSUMPTION_BPR,
        eta_design_hpt=ASSUMPTION_ETA_DESIGN_HPT,
        eta_design_lpt=ASSUMPTION_ETA_DESIGN_LPT,
    )
    return BraytonEngine(cfg)


def run_fc02_forward(engine: BraytonEngine, fc02_si: dict[str, float]):
    # Build SI ops/sens dicts in shape (1,) for batched forward
    ops_imp = {
        "alt_ft":  torch.tensor([fc02_si["alt_ft"]]),
        "XM":      torch.tensor([fc02_si["XM"]]),
        "TRA_pct": torch.tensor([fc02_si["TRA_pct"]]),
        "T2_R":    torch.tensor([fc02_si["T2_R"]]),
        "P2_psia": torch.tensor([fc02_si["P2_psia"]]),
    }
    sens_imp = {
        "Nf_rpm": torch.tensor([fc02_si["Nf_rpm"]]),
        "Nc_rpm": torch.tensor([fc02_si["Nc_rpm"]]),
        "Wf_pps": torch.tensor([fc02_si["Wf_pps"]]),
    }
    si = units.to_si(ops_imp, sens_imp)
    ops_si  = {"T2_K": si["T2_K"], "P2_Pa": si["P2_Pa"],
               "alt_m": si["alt_m"], "mach": si["mach"]}
    sens_si = {"Nf_rpm": si["Nf_rpm"], "Nc_rpm": si["Nc_rpm"],
               "Wf_kgs": si["Wf_kgs"]}
    theta = torch.ones(1, 5)  # healthy
    sensors_pred_si, diag = engine(ops_si, sens_si, theta)
    return sensors_pred_si, diag, ops_si


# =============================================================================
# Step 6 — Report
# =============================================================================

def main() -> None:
    fc02 = fc02_conditions_si()
    print("=" * 72)
    print("V3.1b Phase C0 — one-shot anchor check at UserGuide FC02")
    print("=" * 72)
    print(f"\nFixed assumptions (frozen):")
    print(f"  bpr_design          = {ASSUMPTION_BPR}")
    print(f"  PR_design_fan/lpc/hpc = {ASSUMPTION_PR_DESIGN_FAN} / "
          f"{ASSUMPTION_PR_DESIGN_LPC} / {ASSUMPTION_PR_DESIGN_HPC}")
    print(f"  eta_design_fan/lpc/hpc/hpt/lpt = "
          f"{ASSUMPTION_ETA_DESIGN_FAN} / {ASSUMPTION_ETA_DESIGN_LPC} / "
          f"{ASSUMPTION_ETA_DESIGN_HPC} / {ASSUMPTION_ETA_DESIGN_HPT} / "
          f"{ASSUMPTION_ETA_DESIGN_LPT}")

    print(f"\nSingle manually picked parameter:")
    print(f"  Wc_fan_design       = {PICKED_WC_FAN_DESIGN} kg/s")

    print(f"\nFC02 (post-ram, total values at fan inlet):")
    print(f"  T2 = {fc02['T2_R']:.3f} R = {fc02['T2_R'] * units.RANK_TO_K:.3f} K")
    print(f"  P2 = {fc02['P2_psia']:.3f} psia = "
          f"{fc02['P2_psia'] * units.PSIA_TO_PA:.1f} Pa")
    print(f"  Nf = {fc02['Nf_rpm']} rpm,  Nc = {fc02['Nc_rpm']} rpm,  "
          f"Wf = {fc02['Wf_pps']} pps")
    print(f"  References: T48 = {fc02['T48_ref_R']} R, EPR = {fc02['EPR_ref']:.3f}")

    design = derive_design_points(fc02, PICKED_WC_FAN_DESIGN)
    print(f"\nDerived design points (from picked Wc_fan_design + fixed assumptions):")
    for k in ("Nc_design_fan", "Wc_design_fan",
              "Nc_design_lpc", "Wc_design_lpc",
              "Nc_design_hpc", "Wc_design_hpc",
              "PR_design_fan", "PR_design_lpc", "PR_design_hpc"):
        print(f"  {k:18s} = {design[k]:.4f}")
    print(f"  (intermediate)  m_in_design  = {design['_m_in_design']:.2f} kg/s")
    print(f"  (intermediate)  m_core_design = {design['_m_core_design']:.2f} kg/s")
    print(f"  (intermediate)  T21_design   = {design['_T21_design_K']:.2f} K")
    print(f"  (intermediate)  T24_design   = {design['_T24_design_K']:.2f} K")

    engine = build_engine(design)
    sensors_pred_si, diag, ops_si = run_fc02_forward(engine, fc02)

    # Convert outputs to imperial for reference comparison
    sensors_pred_imp = units.to_imperial(sensors_pred_si)

    T45_K = float(diag["T45"].item())
    T45_R = T45_K / units.RANK_TO_K
    P50_Pa = float(diag["P50"].item())
    P2_Pa = float(ops_si["P2_Pa"].item())
    EPR_pred = P50_Pa / P2_Pa

    rel_err_T45 = abs(T45_R - fc02["T48_ref_R"]) / fc02["T48_ref_R"]
    rel_err_EPR = abs(EPR_pred - fc02["EPR_ref"]) / fc02["EPR_ref"]

    print(f"\n" + "-" * 72)
    print(f"BraytonEngine forward at FC02, theta = 1.0  (one shot, no iteration)")
    print(f"-" * 72)
    print(f"\nPRIMARY anchor target: T45 vs T48_ref (proxy)")
    print(f"  T45_pred  = {T45_K:.2f} K  =  {T45_R:.2f} R")
    print(f"  T48_ref   = {fc02['T48_ref_R']:.2f} R")
    print(f"  rel err   = {rel_err_T45 * 100:.2f} %  "
          f"{'PASS' if rel_err_T45 < 0.25 else 'FAIL'} (band = 25 %)")

    print(f"\nINDEPENDENT plausibility check: EPR = P50/P2 (CMAPSS convention)")
    print(f"  EPR_pred  = {EPR_pred:.4f}")
    print(f"  EPR_ref   = {fc02['EPR_ref']:.4f}")
    print(f"  rel err   = {rel_err_EPR * 100:.2f} %  "
          f"{'PASS' if rel_err_EPR < 0.25 else 'FAIL'} (band = 25 %)")

    print(f"\nSecondary diagnostics (sanity, not acceptance):")
    print(f"  T24_pred (R)        = "
          f"{sensors_pred_imp['T24_R'].item():.2f}  (LPC outlet)")
    print(f"  T30_pred (R)        = "
          f"{sensors_pred_imp['T30_R'].item():.2f}  (HPC outlet)")
    print(f"  P30_pred (psia)     = "
          f"{sensors_pred_imp['P30_psia'].item():.2f}  (HPC outlet, total)")
    print(f"  T50_pred (R)        = "
          f"{sensors_pred_imp['T50_R'].item():.2f}  (LPT outlet)")
    print(f"  T4 (TIT)            = {float(diag['T4'].item()):.2f} K")
    print(f"  m_in (kg/s)         = {float(diag['m_in'].item()):.2f}")
    print(f"  m_core (kg/s)       = {float(diag['m_core'].item()):.2f}")
    print(f"  PR_fan / lpc / hpc  = "
          f"{float(diag['PR_fan'].item()):.3f} / "
          f"{float(diag['PR_lpc'].item()):.3f} / "
          f"{float(diag['PR_hpc'].item()):.3f}")
    print(f"  PR_hpt / lpt        = "
          f"{float(diag['PR_hpt'].item()):.3f} / "
          f"{float(diag['PR_lpt'].item()):.3f}")
    print(f"  P30/P2              = {float(diag['P30_over_P2'].item()):.3f}")

    print(f"\nConservation residuals (should all be near zero by closure):")
    for k in ("mass_balance_inlet", "mass_balance_combust",
              "shaft_HPT_residual", "shaft_LPT_residual"):
        print(f"  {k:24s} = {float(diag[k].item()):.6e}")

    pr_clamp_active = any(
        float(diag[k].item()) > 0.0 for k in (
            "frac_PR_fan_clamped", "frac_PR_lpc_clamped",
            "frac_PR_hpc_clamped", "frac_PR_hpt_clamped",
            "frac_PR_lpt_clamped",
        )
    )
    print(f"\nPR-clamp active anywhere? {pr_clamp_active}")
    if pr_clamp_active:
        for k in ("frac_PR_fan_clamped", "frac_PR_lpc_clamped",
                  "frac_PR_hpc_clamped", "frac_PR_hpt_clamped",
                  "frac_PR_lpt_clamped"):
            print(f"  {k:24s} = {float(diag[k].item()):.3f}")

    print(f"\n" + "=" * 72)
    print(f"STOP — report to Robert before writing YAML values.")
    print(f"=" * 72)


if __name__ == "__main__":
    main()
