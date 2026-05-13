"""C0d documented C-MAPSS reference-engine initialization check (V3.1b, read-only).

Purpose
-------
After C0c showed that there are many BPR/Wc_fan combinations that fit FC02
EPR and T45, we explicitly do NOT pick the numerically-best fit. Instead we
test 4 NAMED CANDIDATES initialized from documented C-MAPSS reference-engine
values (BPR≈8.4, OPR≈36, Nf_design≈2450 rpm, Nc_design≈9300 rpm,
Wc_fan_design≈1658 kg/s, component PR / eta table) and report whether the
documented configuration is physically defensible at FC02.

Hard constraints (V3.1b Hard Rule 17 + C0d decision)
----------------------------------------------------
- read-only: NO writes to configs/cyclelayer_v3.yaml
- NO DS02 access
- NO `fit_*` helpers, NO optimizer, NO automatic parameter selection
- NO turbine eta below realistic component values (≥ 0.88)
- FC02 is an external sanity check, NOT a fitting target
- C1 across 13 User Guide flight conditions is the real next validation gate

Outputs
-------
artifacts/cyclelayer_v3/c0d_cmapss_documented_design/
  ├─ 20_FC02_station_temperatures.png
  ├─ 21_FC02_station_pressure_ratios_log.png
  ├─ 22_FC02_EPR_waterfall.png
  ├─ 23_FC02_work_balance.png
  ├─ 24_FC02_metric_comparison_bars.png
  ├─ 25_FC02_metrics_table.png
  ├─ candidates_metrics.csv
  └─ c0d_cmapss_documented_design_report.md
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import torch                      # noqa: E402

from cyclelayer.data.ncmapss_v3 import load_userguide_fc02_anchor   # noqa: E402
from cyclelayer.models import units                                  # noqa: E402
from cyclelayer.models.brayton_engine import (                       # noqa: E402
    BraytonEngine,
    BraytonEngineConfig,
    InletFlowParams,
    MapCoefficients,
)
from cyclelayer.models.stations import (                             # noqa: E402
    EXP_C, GAMMA_C,
    P_REF, T_REF,
)


# =============================================================================
# Documented C-MAPSS reference values (USER-PROVIDED — pending Frederick TM2007-215026 verification)
# =============================================================================
# Source verification status (local-only search):
#   - Saxena 2008 ("Damage Propagation Modeling..."), data/CMAPSS/Damage Propagation Modeling.pdf
#       VERIFIED: 90,000 lb thrust class
#       VERIFIED: operating envelope alt 0–40K ft, M 0–0.9, T(SL) −60..103 °F
#       NOT FOUND: numerical BPR / OPR / Nf_des / Nc_des / Wc_des / PR table / eta table
#       (Saxena cites Frederick et al., NASA/ARL TM2007-215026 ref [11] as the source)
#   - Chao 2022 ("Run-to-Failure..."), data/NCMAPSS/Run_to_Failure_Simulation_*.pdf
#       (N-CMAPSS dataset paper; engine details inherited from C-MAPSS, not re-published)
#   - Frederick 2007 NASA/ARL TM2007-215026 PDF: NOT PRESENT in local repo
#
# Until Frederick TM2007-215026 is acquired, the values below are USER-PROVIDED
# DOCUMENTED VALUES from the C0d task spec. They are marked accordingly in the
# report.

DOC_BPR              = 8.4
DOC_OPR              = 36.0
DOC_NF_DES_RPM       = 2450.0
DOC_NC_DES_RPM       = 9300.0
DOC_WCFAN_DES_KGS    = 1658.0       # ≈ 3655.5 lbm/s
DOC_PR_FAN           = 1.784
DOC_ETA_FAN          = 0.8969
DOC_PR_LPC           = 1.1035
DOC_ETA_LPC          = 0.9148
DOC_PR_HPC_TABLE     = 21.817        # the documented component-table value
DOC_ETA_HPC          = 0.8615
DOC_PR_HPT           = 4.239
DOC_ETA_HPT          = 0.9202
DOC_PR_LPT           = 5.858
DOC_ETA_LPT          = 0.930

# Implied OPR from the documented component PR table (note discrepancy)
IMPLIED_OPR_FROM_TABLE = DOC_PR_FAN * DOC_PR_LPC * DOC_PR_HPC_TABLE
# = 1.784 * 1.1035 * 21.817 ≈ 42.95 — does NOT match DOC_OPR=36.

# Old-baseline literature defaults (Walsh & Fletcher / Kurzke generic)
LIT_ETA_FAN = 0.92
LIT_ETA_LPC = 0.90
LIT_ETA_HPC = 0.88
LIT_ETA_HPT = 0.90
LIT_ETA_LPT = 0.92


# =============================================================================
# Plausibility windows (kept consistent with C0c)
# =============================================================================

T4_LO,  T4_HI   = 1300.0, 1900.0
T30_LO, T30_HI  = 600.0,  1000.0
T50_LO, T50_HI  = 650.0,  1100.0
FAR_LO, FAR_HI  = 0.015,  0.035
T45_ERR_PREFERRED = 15.0
T45_ERR_ACCEPT    = 25.0
EPR_ERR_ACCEPT    = 25.0


# =============================================================================
# FC02 conditions (external sanity check inputs; not a fitting target)
# =============================================================================

def fc02_conditions_si() -> dict[str, float]:
    fc = load_userguide_fc02_anchor()
    Tsl_R = fc["Tsl_F"] + 459.67
    P0_psia = 14.696
    M = fc["XM"]
    ram_T = 1.0 + 0.5 * (GAMMA_C - 1.0) * M ** 2
    ram_P = ram_T ** (GAMMA_C / (GAMMA_C - 1.0))
    return {
        "alt_ft":    float(fc["alt_ft"]),
        "XM":        float(fc["XM"]),
        "TRA_pct":   float(fc["TRA_pct"]),
        "T2_R":      Tsl_R * ram_T,
        "P2_psia":   P0_psia * ram_P,
        "Nf_rpm":    float(fc["Nf_rpm"]),
        "Nc_rpm":    float(fc["Nc_rpm"]),
        "Wf_pps":    float(fc["Wf_pps"]),
        "T48_ref_R": float(fc["T48_ref_R"]),
        "EPR_ref":   float(fc["EPR_ref"]),
    }


# =============================================================================
# Build a fully-explicit MapCoefficients/InletFlowParams from a candidate spec
# =============================================================================
# Candidate spec includes ALL design values explicitly. Wc_design_lpc / hpc are
# derived from station thermodynamics at the design point (SLS Std-Day) using
# the candidate's own PR / eta values — NOT in-code constants.

def derive_dependent_design(spec: dict, fc02: dict) -> dict:
    """Compute Wc_design_lpc, Wc_design_hpc, Nc_design_lpc, Nc_design_hpc
    from the candidate's documented Wc_fan_design + Nf_des + Nc_des + cold-side
    PR / eta values. No in-code defaults consulted.
    """
    # SLS Std-Day reference (T_REF, P_REF) — this is the design-point convention.
    # If a different design-point reference is preferred, document it in spec.
    T2_des_K = T_REF
    P2_des_Pa = P_REF
    Wc_fan = spec["Wc_fan_design"]

    # m_in at design (no ram correction since T2_des = T_REF, P2_des = P_REF)
    m_in_des   = Wc_fan
    m_core_des = m_in_des / (spec["bpr_design"] + 1.0)

    # Fan thermodynamics at design
    pr_fan = spec["PR_design_fan"]
    T21 = T2_des_K + (T2_des_K * pr_fan ** EXP_C - T2_des_K) / spec["eta_design_fan"]
    P21 = P2_des_Pa * pr_fan
    Wc_design_lpc = m_core_des * math.sqrt(T21 / T_REF) / (P21 / P_REF)

    # LPC thermodynamics at design
    pr_lpc = spec["PR_design_lpc"]
    T24 = T21 + (T21 * pr_lpc ** EXP_C - T21) / spec["eta_design_lpc"]
    P24 = P21 * pr_lpc
    Wc_design_hpc = m_core_des * math.sqrt(T24 / T_REF) / (P24 / P_REF)

    # Corrected design speeds (the documented values are at SLS Std-Day so
    # corrected_speed at design = design rpm)
    Nc_design_fan = spec["Nf_des_rpm"]
    Nc_design_lpc = Nc_design_fan
    Nc_design_hpc = spec["Nc_des_rpm"] / math.sqrt(T24 / T_REF)

    return {
        "Nc_design_fan": Nc_design_fan,
        "Nc_design_lpc": Nc_design_lpc,
        "Nc_design_hpc": Nc_design_hpc,
        "Wc_design_fan": Wc_fan,
        "Wc_design_lpc": Wc_design_lpc,
        "Wc_design_hpc": Wc_design_hpc,
        "PR_design_fan": pr_fan,
        "PR_design_lpc": pr_lpc,
        "PR_design_hpc": spec["PR_design_hpc"],
        "eta_design_fan": spec["eta_design_fan"],
        "eta_design_lpc": spec["eta_design_lpc"],
        "eta_design_hpc": spec["eta_design_hpc"],
        # Diagnostics
        "_m_in_design":  m_in_des,
        "_m_core_design": m_core_des,
        "_T21_design_K": T21,
        "_T24_design_K": T24,
        "_P21_design_Pa": P21,
        "_P24_design_Pa": P24,
    }


def build_engine(spec: dict, fc02: dict) -> tuple[BraytonEngine, dict]:
    design = derive_dependent_design(spec, fc02)
    cfg = BraytonEngineConfig(
        inlet_flow=InletFlowParams(
            Wc_fan_design=design["Wc_design_fan"],
            Nc_fan_design=design["Nc_design_fan"],
            # Safety bounds chosen explicitly per candidate to NOT clip the
            # documented Wc_fan_design (no hidden defaults). The InletFlowParams
            # defaults (Wc_min=100, Wc_max=1100) were sized for CFM56-class
            # and would clip the documented C-MAPSS Wc_fan_design = 1658.
            Wc_min=spec["Wc_min"],
            Wc_max=spec["Wc_max"],
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
            eta_design_fan=design["eta_design_fan"],
            eta_design_lpc=design["eta_design_lpc"],
            eta_design_hpc=design["eta_design_hpc"],
        ),
        use_measured_inlet=True,
        bpr_design=spec["bpr_design"],
        eta_design_hpt=spec["eta_design_hpt"],
        eta_design_lpt=spec["eta_design_lpt"],
    )
    return BraytonEngine(cfg), design


def run_fc02_forward(engine: BraytonEngine, fc02: dict):
    ops_imp = {
        "alt_ft":  torch.tensor([fc02["alt_ft"]]),
        "XM":      torch.tensor([fc02["XM"]]),
        "TRA_pct": torch.tensor([fc02["TRA_pct"]]),
        "T2_R":    torch.tensor([fc02["T2_R"]]),
        "P2_psia": torch.tensor([fc02["P2_psia"]]),
    }
    sens_imp = {
        "Nf_rpm": torch.tensor([fc02["Nf_rpm"]]),
        "Nc_rpm": torch.tensor([fc02["Nc_rpm"]]),
        "Wf_pps": torch.tensor([fc02["Wf_pps"]]),
    }
    si = units.to_si(ops_imp, sens_imp)
    ops_si  = {"T2_K": si["T2_K"], "P2_Pa": si["P2_Pa"],
               "alt_m": si["alt_m"], "mach": si["mach"]}
    sens_si = {"Nf_rpm": si["Nf_rpm"], "Nc_rpm": si["Nc_rpm"],
               "Wf_kgs": si["Wf_kgs"]}
    theta = torch.ones(1, 5)
    sensors_pred_si, diag = engine(ops_si, sens_si, theta)
    return sensors_pred_si, diag, ops_si, sens_si


# =============================================================================
# Candidate definitions (4 named, no automation)
# =============================================================================

def candidates() -> dict[str, dict]:
    """Return the 4 named candidates as fully-explicit specs.

    Candidate A — engine-level OPR=36 respected (PR_hpc derived to hit OPR=36)
    Candidate B — component PR table respected (PR_hpc=21.817; implied OPR≈42.95)
    Candidate C — conservative hybrid (slightly less extreme PR-split, OPR=36)
    Candidate D — old baseline (BPR=5.5, Wc=900, OPR≈38.4) — NOT FOR ADOPTION
    """
    # Candidate A: engine-level OPR respected
    pr_hpc_A = DOC_OPR / (DOC_PR_FAN * DOC_PR_LPC)
    A = dict(
        name="A_OPR36",
        label="A — documented OPR=36 (PR_hpc derived)",
        bpr_design=DOC_BPR,
        Wc_fan_design=DOC_WCFAN_DES_KGS,
        Nf_des_rpm=DOC_NF_DES_RPM,
        Nc_des_rpm=DOC_NC_DES_RPM,
        PR_design_fan=DOC_PR_FAN,
        PR_design_lpc=DOC_PR_LPC,
        PR_design_hpc=pr_hpc_A,
        eta_design_fan=DOC_ETA_FAN,
        eta_design_lpc=DOC_ETA_LPC,
        eta_design_hpc=DOC_ETA_HPC,
        eta_design_hpt=DOC_ETA_HPT,
        eta_design_lpt=DOC_ETA_LPT,
        # Safety bounds sized for documented C-MAPSS Wc_fan_design ≈ 1658
        Wc_min=200.0,
        Wc_max=2500.0,
        adopt=False,
    )

    # Candidate B: component PR table respected, implied OPR documented
    B = dict(
        name="B_PRtable",
        label=f"B — documented PR table (PR_hpc=21.817, implied OPR={IMPLIED_OPR_FROM_TABLE:.2f})",
        bpr_design=DOC_BPR,
        Wc_fan_design=DOC_WCFAN_DES_KGS,
        Nf_des_rpm=DOC_NF_DES_RPM,
        Nc_des_rpm=DOC_NC_DES_RPM,
        PR_design_fan=DOC_PR_FAN,
        PR_design_lpc=DOC_PR_LPC,
        PR_design_hpc=DOC_PR_HPC_TABLE,
        eta_design_fan=DOC_ETA_FAN,
        eta_design_lpc=DOC_ETA_LPC,
        eta_design_hpc=DOC_ETA_HPC,
        eta_design_hpt=DOC_ETA_HPT,
        eta_design_lpt=DOC_ETA_LPT,
        Wc_min=200.0,
        Wc_max=2500.0,
        adopt=False,
    )

    # Candidate C: conservative hybrid
    pr_hpc_C = DOC_OPR / (1.70 * 1.20)
    C = dict(
        name="C_conservative",
        label="C — conservative hybrid (PR_fan=1.70, PR_lpc=1.20, OPR=36)",
        bpr_design=DOC_BPR,
        Wc_fan_design=1500.0,
        Nf_des_rpm=DOC_NF_DES_RPM,
        Nc_des_rpm=DOC_NC_DES_RPM,
        PR_design_fan=1.70,
        PR_design_lpc=1.20,
        PR_design_hpc=pr_hpc_C,
        eta_design_fan=DOC_ETA_FAN,
        eta_design_lpc=DOC_ETA_LPC,
        eta_design_hpc=DOC_ETA_HPC,
        eta_design_hpt=DOC_ETA_HPT,
        eta_design_lpt=DOC_ETA_LPT,
        Wc_min=200.0,
        Wc_max=2500.0,
        adopt=False,
    )

    # Candidate D: old baseline (NOT FOR ADOPTION)
    D = dict(
        name="D_old_baseline",
        label="D — OLD baseline (BPR=5.5, Wc=900, OPR=38.4) — NOT FOR ADOPTION",
        bpr_design=5.5,
        Wc_fan_design=900.0,
        Nf_des_rpm=2450.0,    # we use the same documented Nc convention
        Nc_des_rpm=9300.0,
        PR_design_fan=1.6,
        PR_design_lpc=2.0,
        PR_design_hpc=12.0,
        eta_design_fan=LIT_ETA_FAN,
        eta_design_lpc=LIT_ETA_LPC,
        eta_design_hpc=LIT_ETA_HPC,
        eta_design_hpt=LIT_ETA_HPT,
        eta_design_lpt=LIT_ETA_LPT,
        # Old baseline used the legacy CFM56-class safety bounds
        Wc_min=100.0,
        Wc_max=1100.0,
        adopt=False,
    )

    return {"A": A, "B": B, "C": C, "D": D}


# =============================================================================
# Evaluate each candidate at FC02
# =============================================================================

def evaluate(spec: dict, fc02: dict) -> dict:
    engine, design = build_engine(spec, fc02)
    sensors_pred_si, diag, ops_si, sens_si = run_fc02_forward(engine, fc02)

    T2 = float(ops_si["T2_K"].item())
    P2 = float(ops_si["P2_Pa"].item())
    PR_fan = float(diag["PR_fan"].item());  PR_lpc = float(diag["PR_lpc"].item())
    eta_fan_eff = float(diag["eta_fan"].item())

    # Reconstruct T21 / P21 / P24 (not in diag)
    T21_isen = T2 * PR_fan ** EXP_C
    T21      = T2 + (T21_isen - T2) / eta_fan_eff
    P21      = P2 * PR_fan
    T24 = float(sensors_pred_si["T24_K"].item());  P24 = P21 * PR_lpc

    T30 = float(sensors_pred_si["T30_K"].item()); P30 = float(sensors_pred_si["P30_Pa"].item())
    T4  = float(diag["T4"].item());               P4  = float(diag["P4"].item())
    T45 = float(diag["T45"].item());              P45 = float(diag["P45"].item())
    T50 = float(sensors_pred_si["T50_K"].item());  P50 = float(diag["P50"].item())

    OPR = float(diag["P30_over_P2"].item())
    EPR = P50 / P2
    T45_R = T45 / units.RANK_TO_K
    T45_err = abs(T45_R - fc02["T48_ref_R"]) / fc02["T48_ref_R"] * 100.0
    EPR_err = abs(EPR - fc02["EPR_ref"]) / fc02["EPR_ref"] * 100.0

    m_in   = float(diag["m_in"].item())
    m_core = float(diag["m_core"].item())
    m_byp  = float(diag["m_byp"].item())
    Wf_kgs = float(sens_si["Wf_kgs"].item())
    FAR    = Wf_kgs / m_core

    W_fan_total = float(diag["W_fan_total"].item())
    W_lpc = float(diag["W_lpc"].item());  W_hpc = float(diag["W_hpc"].item())
    W_hpt = float(diag["W_hpt"].item());  W_lpt = float(diag["W_lpt"].item())

    PR_hpt = float(diag["PR_hpt"].item());  PR_lpt = float(diag["PR_lpt"].item())

    clamp_keys = ("frac_PR_fan_clamped", "frac_PR_lpc_clamped",
                  "frac_PR_hpc_clamped", "frac_PR_hpt_clamped",
                  "frac_PR_lpt_clamped")
    clamps = {k: float(diag[k].item()) for k in clamp_keys}
    clamps_active = any(v > 0.0 for v in clamps.values())

    # Wc_fan diagnostic — was the inlet-flow Wc_fan clamped?
    inlet = engine.config.inlet_flow
    Wc_fan_requested = inlet.Wc_fan_design
    # Approximate actual Wc_fan at FC02 (recompute the inlet-flow estimator path)
    from cyclelayer.models.brayton_engine import estimate_inlet_flow as _est
    Wc_fan_actual = float(_est(
        torch.tensor([T2]), torch.tensor([P2]), torch.tensor([fc02["Nf_rpm"]]),
        inlet,
    ).item())
    # The estimator de-corrects to m_in; "actual Wc" before clamp:
    # Wc_pre = Wc_fan_design * (1 + c1*dN + c2*dN^2)
    Nc_des = inlet.Nc_fan_design
    Nc_act = fc02["Nf_rpm"] / math.sqrt(T2 / T_REF)
    dN = (Nc_act - Nc_des) / Nc_des
    Wc_pre_clamp = inlet.Wc_fan_design * (1.0 + inlet.c1 * dN + inlet.c2 * dN ** 2)
    Wc_clamp_active = (Wc_pre_clamp <= inlet.Wc_min) or (Wc_pre_clamp >= inlet.Wc_max)

    # Plausibility flags
    flags = {
        "FAR_plausible":  FAR_LO <= FAR <= FAR_HI,
        "T4_plausible":   T4_LO  <= T4  <= T4_HI,
        "T30_plausible":  T30_LO <= T30 <= T30_HI,
        "T50_plausible":  T50_LO <= T50 <= T50_HI,
        "T45_preferred":  T45_err < T45_ERR_PREFERRED,
        "T45_acceptable": T45_err < T45_ERR_ACCEPT,
        "EPR_acceptable": EPR_err < EPR_ERR_ACCEPT,
        "no_PR_clamp":    not clamps_active,
        "no_Wc_clamp":    not Wc_clamp_active,
        "eta_hpt_realistic": spec["eta_design_hpt"] >= 0.88,
        "eta_lpt_realistic": spec["eta_design_lpt"] >= 0.88,
    }

    return {
        "name": spec["name"], "label": spec["label"],
        "BPR": spec["bpr_design"],
        "Wc_fan_design_kg_s": spec["Wc_fan_design"],
        "PR_fan_design": spec["PR_design_fan"],
        "PR_lpc_design": spec["PR_design_lpc"],
        "PR_hpc_design": spec["PR_design_hpc"],
        "OPR_implied":   spec["PR_design_fan"] * spec["PR_design_lpc"] * spec["PR_design_hpc"],
        "eta_fan": spec["eta_design_fan"],
        "eta_lpc": spec["eta_design_lpc"],
        "eta_hpc": spec["eta_design_hpc"],
        "eta_hpt": spec["eta_design_hpt"],
        "eta_lpt": spec["eta_design_lpt"],
        # FC02 results (effective values from the engine)
        "OPR_pred":    OPR,
        "EPR_pred":    EPR,
        "EPR_err_pct": EPR_err,
        "T45_R":       T45_R,
        "T45_err_pct": T45_err,
        "T2_K": T2, "T21_K": T21, "T24_K": T24, "T30_K": T30,
        "T4_K": T4, "T45_K": T45, "T50_K": T50,
        "P2_Pa": P2, "P21_Pa": P21, "P24_Pa": P24, "P30_Pa": P30,
        "P4_Pa": P4, "P45_Pa": P45, "P50_Pa": P50,
        "PR_hpt_pred": PR_hpt, "PR_lpt_pred": PR_lpt,
        "m_in_kg_s":   m_in, "m_core_kg_s": m_core, "m_bypass_kg_s": m_byp,
        "Wf_kg_s":     Wf_kgs, "FAR": FAR,
        "W_fan_total_MW": W_fan_total / 1e6,
        "W_lpc_MW": W_lpc / 1e6, "W_hpc_MW": W_hpc / 1e6,
        "W_hpt_MW": W_hpt / 1e6, "W_lpt_MW": W_lpt / 1e6,
        "shaft_HPT_resid_W": float(diag["shaft_HPT_residual"].item()),
        "shaft_LPT_resid_W": float(diag["shaft_LPT_residual"].item()),
        **clamps,
        "clamps_active": clamps_active,
        "Wc_fan_requested": Wc_fan_requested,
        "Wc_fan_pre_clamp": Wc_pre_clamp,
        "Wc_fan_actual":    Wc_fan_actual,
        "Wc_min":           inlet.Wc_min,
        "Wc_max":           inlet.Wc_max,
        "Wc_clamp_active":  Wc_clamp_active,
        **flags,
    }


# =============================================================================
# Plotting
# =============================================================================

OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "c0d_cmapss_documented_design"
STATIONS = ["2", "21", "24", "30", "4", "45", "50"]
CAND_COLORS = {"A_OPR36": "tab:blue", "B_PRtable": "tab:orange",
               "C_conservative": "tab:green", "D_old_baseline": "tab:red"}


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {p}")


def _stations_arrays(row: dict) -> tuple[list[float], list[float]]:
    Ts = [row[f"T{s}_K"] if f"T{s}_K" in row else None for s in STATIONS]
    Ps = [row[f"P{s}_Pa"] for s in STATIONS]
    return Ts, Ps


def plot_station_temperatures(rows: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in rows:
        Ts, _ = _stations_arrays(r)
        ax.plot(STATIONS, Ts, "o-", color=CAND_COLORS[r["name"]],
                lw=1.8, ms=7, label=r["label"])
    ax.set_xlabel("Station");  ax.set_ylabel("Total temperature [K]")
    ax.set_title("FC02 — total temperature at each station, all C0d candidates")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "20_FC02_station_temperatures.png")


def plot_station_pressure_ratios(rows: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in rows:
        _, Ps = _stations_arrays(r)
        P2 = r["P2_Pa"]
        ax.semilogy(STATIONS, [p / P2 for p in Ps], "o-",
                    color=CAND_COLORS[r["name"]], lw=1.8, ms=7,
                    label=r["label"])
    ax.axhline(1.0, color="black", lw=0.4)
    ax.set_xlabel("Station");  ax.set_ylabel("P / P2 (log)")
    ax.set_title("FC02 — total pressure ratio P/P2 at each station, all C0d candidates")
    ax.grid(True, alpha=0.4, which="both")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "21_FC02_station_pressure_ratios_log.png")


def plot_epr_waterfall(rows: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, r in zip(axes.flat, rows):
        P2 = r["P2_Pa"]
        steps = [
            ("P2",         1.0),
            ("× fan",      r["P21_Pa"] / P2),
            ("× LPC",      r["P24_Pa"] / r["P21_Pa"]),
            ("× HPC",      r["P30_Pa"] / r["P24_Pa"]),
            ("× comb",     r["P4_Pa"]  / r["P30_Pa"]),
            ("÷ HPT",      r["P45_Pa"] / r["P4_Pa"]),
            ("÷ LPT",      r["P50_Pa"] / r["P45_Pa"]),
            ("EPR",        r["P50_Pa"] / P2),
        ]
        labels = [s[0] for s in steps]
        cum = []
        running = 1.0
        for lbl, factor in steps:
            if lbl == "P2":
                cum.append(1.0)
            elif lbl == "EPR":
                cum.append(r["P50_Pa"] / P2)
            else:
                running *= factor
                cum.append(running)
        ax.plot(labels, cum, "o-", color=CAND_COLORS[r["name"]], lw=1.8, ms=7)
        for x, y in zip(labels, cum):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7)
        ax.set_yscale("log")
        ax.set_title(f"{r['label']}\nEPR={r['EPR_pred']:.3f}  "
                     f"(ref 1.261, err {r['EPR_err_pct']:.1f}%)",
                     fontsize=9)
        ax.grid(True, alpha=0.4, which="both")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
    fig.suptitle("FC02 — EPR decomposition waterfall, all C0d candidates")
    fig.tight_layout()
    _save(fig, "22_FC02_EPR_waterfall.png")


def plot_work_balance(rows: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    names = [r["name"] for r in rows]
    labels = [r["label"].split(" — ")[0] for r in rows]

    HPC = [r["W_hpc_MW"] for r in rows]
    HPT = [r["W_hpt_MW"] for r in rows]
    Fan = [r["W_fan_total_MW"] for r in rows]
    LPC = [r["W_lpc_MW"] for r in rows]
    FanLPC = [f + l for f, l in zip(Fan, LPC)]
    LPT = [r["W_lpt_MW"] for r in rows]

    x = np.arange(len(names));  w = 0.35

    axes[0].bar(x - w/2, HPC, w, label="W_HPC", color="tab:blue")
    axes[0].bar(x + w/2, HPT, w, label="W_HPT", color="tab:orange")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("Shaft work [MW]")
    axes[0].set_title("HP spool: HPC vs HPT")
    axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.4)
    for i, (a, b) in enumerate(zip(HPC, HPT)):
        axes[0].annotate(f"{a:.1f}", (i - w/2, a), xytext=(0, 4), ha="center",
                         textcoords="offset points", fontsize=7)
        axes[0].annotate(f"{b:.1f}", (i + w/2, b), xytext=(0, 4), ha="center",
                         textcoords="offset points", fontsize=7)

    axes[1].bar(x - w/2, FanLPC, w, label="W_Fan_total + W_LPC", color="tab:blue")
    axes[1].bar(x + w/2, LPT,    w, label="W_LPT",                color="tab:orange")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("Shaft work [MW]")
    axes[1].set_title("LP spool: Fan+LPC vs LPT")
    axes[1].legend(); axes[1].grid(True, axis="y", alpha=0.4)
    for i, (a, b) in enumerate(zip(FanLPC, LPT)):
        axes[1].annotate(f"{a:.1f}", (i - w/2, a), xytext=(0, 4), ha="center",
                         textcoords="offset points", fontsize=7)
        axes[1].annotate(f"{b:.1f}", (i + w/2, b), xytext=(0, 4), ha="center",
                         textcoords="offset points", fontsize=7)

    fig.suptitle("FC02 — spool work balance (closure should match by construction)")
    fig.tight_layout()
    _save(fig, "23_FC02_work_balance.png")


def plot_metric_comparison(rows: list[dict]):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    metrics = [
        ("T45_err_pct", "T45 rel err [%]",   T45_ERR_ACCEPT, "%"),
        ("EPR_err_pct", "EPR rel err [%]",   EPR_ERR_ACCEPT, "%"),
        ("FAR",         "FAR",               (FAR_LO, FAR_HI), ""),
        ("T4_K",        "T4 (TIT) [K]",      (T4_LO, T4_HI), "K"),
        ("T50_K",       "T50 [K]",           (T50_LO, T50_HI), "K"),
        ("OPR_pred",    "OPR (P30/P2)",      (25.0, 45.0), ""),
    ]
    names = [r["name"] for r in rows]
    labels = [r["label"].split(" — ")[0] for r in rows]
    colors = [CAND_COLORS[n] for n in names]
    for ax, (col, ylabel, band, suffix) in zip(axes.flat, metrics):
        vals = [r[col] for r in rows]
        ax.bar(labels, vals, color=colors, edgecolor="black")
        if isinstance(band, tuple):
            ax.axhline(band[0], color="black", lw=0.6, ls="--")
            ax.axhline(band[1], color="black", lw=0.6, ls="--")
            ax.axhspan(band[0], band[1], color="green", alpha=0.10)
        else:
            ax.axhline(band, color="black", lw=0.6, ls="--",
                       label=f"acceptance {band}{suffix}")
            ax.legend(fontsize=8)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.4)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 4), ha="center",
                        textcoords="offset points", fontsize=7)
    fig.suptitle("FC02 — metric comparison across C0d candidates "
                 "(green band = plausibility window)")
    fig.tight_layout()
    _save(fig, "24_FC02_metric_comparison_bars.png")


def plot_metrics_table(rows: list[dict]):
    cols = ["BPR", "Wc_fan_design_kg_s", "OPR_implied", "OPR_pred",
            "PR_fan_design", "PR_lpc_design", "PR_hpc_design",
            "eta_fan", "eta_lpc", "eta_hpc", "eta_hpt", "eta_lpt",
            "T45_err_pct", "EPR_err_pct",
            "T4_K", "T30_K", "T50_K",
            "FAR", "m_in_kg_s", "m_core_kg_s",
            "PR_hpt_pred", "PR_lpt_pred",
            "clamps_active", "Wc_clamp_active"]
    fmts = {
        "BPR": "{:.1f}", "Wc_fan_design_kg_s": "{:.0f}",
        "OPR_implied": "{:.2f}", "OPR_pred": "{:.2f}",
        "PR_fan_design": "{:.3f}", "PR_lpc_design": "{:.3f}", "PR_hpc_design": "{:.3f}",
        "eta_fan": "{:.4f}", "eta_lpc": "{:.4f}", "eta_hpc": "{:.4f}",
        "eta_hpt": "{:.4f}", "eta_lpt": "{:.4f}",
        "T45_err_pct": "{:.2f}", "EPR_err_pct": "{:.2f}",
        "T4_K": "{:.0f}", "T30_K": "{:.0f}", "T50_K": "{:.0f}",
        "FAR": "{:.4f}", "m_in_kg_s": "{:.1f}", "m_core_kg_s": "{:.1f}",
        "PR_hpt_pred": "{:.3f}", "PR_lpt_pred": "{:.3f}",
    }
    headers = ["candidate"] + cols
    body = []
    for r in rows:
        row_cells = [r["label"].split(" — ")[0]]
        for c in cols:
            v = r[c]
            if isinstance(v, bool):
                row_cells.append("✓" if v else "✗")
            elif c in fmts and isinstance(v, (int, float)):
                row_cells.append(fmts[c].format(v))
            else:
                row_cells.append(str(v))
        body.append(row_cells)
    fig, ax = plt.subplots(figsize=(14, 0.45 * (len(body) + 1) + 1.2))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(7)
    table.scale(1.0, 1.4)
    ax.set_title("FC02 — C0d candidate metrics table (REPORTING ONLY — not for adoption)")
    _save(fig, "25_FC02_metrics_table.png")


# =============================================================================
# Markdown report
# =============================================================================

def _row_md(row: dict, fields: list[tuple[str, str, str]]) -> str:
    """fields = list of (key, header, fmt)."""
    lines = ["| metric | value |", "|---|---|"]
    for key, header, fmt in fields:
        v = row.get(key)
        if isinstance(v, bool):
            cell = "✓" if v else "✗"
        elif isinstance(v, (int, float)) and not math.isnan(v):
            cell = fmt.format(v)
        else:
            cell = str(v)
        lines.append(f"| {header} | {cell} |")
    return "\n".join(lines)


def write_report(rows: list[dict], fc02: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "candidates_metrics.csv", index=False)
    print(f"  saved  {OUT_DIR / 'candidates_metrics.csv'}")

    # Per-candidate tables
    fields = [
        ("BPR",                "BPR",                 "{:.1f}"),
        ("Wc_fan_design_kg_s", "Wc_fan_design [kg/s]", "{:.0f}"),
        ("PR_fan_design",      "PR_fan",               "{:.3f}"),
        ("PR_lpc_design",      "PR_lpc",               "{:.3f}"),
        ("PR_hpc_design",      "PR_hpc",               "{:.3f}"),
        ("OPR_implied",        "OPR implied (PR product)", "{:.2f}"),
        ("OPR_pred",           "OPR predicted (P30/P2)",   "{:.2f}"),
        ("eta_fan",            "eta_fan",              "{:.4f}"),
        ("eta_lpc",            "eta_lpc",              "{:.4f}"),
        ("eta_hpc",            "eta_hpc",              "{:.4f}"),
        ("eta_hpt",            "eta_hpt",              "{:.4f}"),
        ("eta_lpt",            "eta_lpt",              "{:.4f}"),
        ("EPR_pred",           "EPR pred",             "{:.4f}"),
        ("EPR_err_pct",        "EPR rel err [%]",      "{:.2f}"),
        ("T45_R",              "T45 [°R]",             "{:.1f}"),
        ("T45_err_pct",        "T45 rel err vs T48_ref [%]", "{:.2f}"),
        ("T4_K",               "T4 [K]",               "{:.0f}"),
        ("T30_K",              "T30 [K]",              "{:.0f}"),
        ("T50_K",              "T50 [K]",              "{:.0f}"),
        ("FAR",                "FAR",                  "{:.4f}"),
        ("m_in_kg_s",          "m_in [kg/s]",          "{:.1f}"),
        ("m_core_kg_s",        "m_core [kg/s]",        "{:.1f}"),
        ("PR_hpt_pred",        "PR_hpt (closure)",     "{:.3f}"),
        ("PR_lpt_pred",        "PR_lpt (closure)",     "{:.3f}"),
        ("Wc_fan_requested",   "Wc_fan requested [kg/s]", "{:.1f}"),
        ("Wc_fan_pre_clamp",   "Wc_fan pre-clamp [kg/s]", "{:.1f}"),
        ("Wc_fan_actual",      "Wc_fan actual m_in [kg/s]", "{:.1f}"),
        ("Wc_min",             "Wc_min [kg/s]",        "{:.0f}"),
        ("Wc_max",             "Wc_max [kg/s]",        "{:.0f}"),
        ("Wc_clamp_active",    "Wc clamp active",      "{}"),
        ("clamps_active",      "PR clamp active",      "{}"),
        ("FAR_plausible",      "FAR plausible",        "{}"),
        ("T4_plausible",       "T4 plausible",         "{}"),
        ("T30_plausible",      "T30 plausible",        "{}"),
        ("T50_plausible",      "T50 plausible",        "{}"),
        ("T45_acceptable",     "T45 err < 25 %",       "{}"),
        ("T45_preferred",      "T45 err < 15 %",       "{}"),
        ("EPR_acceptable",     "EPR err < 25 %",       "{}"),
        ("eta_hpt_realistic",  "eta_hpt ≥ 0.88",       "{}"),
        ("eta_lpt_realistic",  "eta_lpt ≥ 0.88",       "{}"),
    ]

    sec_per_cand = ""
    for r in rows:
        sec_per_cand += f"\n### {r['label']}\n\n{_row_md(r, fields)}\n"

    # Q&A narrative
    A = next(r for r in rows if r["name"] == "A_OPR36")
    B = next(r for r in rows if r["name"] == "B_PRtable")
    C = next(r for r in rows if r["name"] == "C_conservative")
    D = next(r for r in rows if r["name"] == "D_old_baseline")

    q1 = (
        f"Old baseline (D, BPR=5.5, Wc=900): EPR err = **{D['EPR_err_pct']:.1f} %**.  "
        f"BPR=8.4 documented (A, OPR=36): EPR err = **{A['EPR_err_pct']:.1f} %**.  "
        + ("YES — moving from BPR=5.5 to BPR=8.4 with documented OPR=36 reduces the EPR mismatch."
           if A["EPR_err_pct"] < D["EPR_err_pct"]
           else "NO — moving to BPR=8.4 alone does not reduce the EPR mismatch under the documented OPR.")
    )

    q2 = (
        f"At Wc_fan_design = {DOC_WCFAN_DES_KGS} kg/s (Candidates A and B), "
        f"m_in at FC02 = {A['m_in_kg_s']:.1f} kg/s, m_core = {A['m_core_kg_s']:.1f} kg/s, "
        f"FAR = {A['FAR']:.4f} ({'plausible' if A['FAR_plausible'] else 'OUT OF RANGE'}), "
        f"T4 = {A['T4_K']:.0f} K ({'plausible' if A['T4_plausible'] else 'OUT OF RANGE'}), "
        f"T45 err = {A['T45_err_pct']:.1f} %, T50 = {A['T50_K']:.0f} K "
        f"({'plausible' if A['T50_plausible'] else 'OUT OF RANGE'})."
    )

    q3 = (
        f"Documented PR table → implied OPR = `PR_fan × PR_lpc × PR_hpc` = "
        f"{DOC_PR_FAN} × {DOC_PR_LPC} × {DOC_PR_HPC_TABLE} = "
        f"**{IMPLIED_OPR_FROM_TABLE:.2f}**.  "
        f"This does NOT match the documented engine-level OPR ≈ {DOC_OPR}.  "
        f"Likely station / design-condition mismatch in the original Frederick TM2007-215026 table "
        f"(e.g. component design point at a different OP, or a published delta vs reference). "
        f"Resolve the mismatch via source verification before adopting the component table verbatim."
    )

    # Pick recommended candidate based on PHYSICAL DEFENSIBILITY (documented values),
    # NOT numerical FC02 fit.
    plausible_doc_candidates = [
        r for r in (A, B, C)
        if r["FAR_plausible"] and r["T4_plausible"] and r["T50_plausible"]
        and r["T45_acceptable"] and r["EPR_acceptable"]
        and r["no_PR_clamp"] and r["no_Wc_clamp"]
        and r["eta_hpt_realistic"] and r["eta_lpt_realistic"]
    ]

    if plausible_doc_candidates:
        # Prefer A (engine-level OPR respected), then C, then B
        priority = {"A_OPR36": 0, "C_conservative": 1, "B_PRtable": 2}
        chosen = sorted(plausible_doc_candidates,
                        key=lambda r: priority[r["name"]])[0]
        recommended = (
            f"**{chosen['label']}** is physically defensible: it uses documented "
            f"BPR=8.4, documented Wc_fan_design, documented engine-level OPR, and "
            f"all plausibility windows hold. Recommend this candidate ONLY for **C1 "
            f"validation across the 13 User Guide flight conditions**, NOT for adoption. "
            f"If C1 succeeds, then ask Robert to adopt; if C1 fails, escalate before any YAML edit."
        )
    else:
        # Document why each documented candidate fails
        recommended = (
            "**No documented candidate (A / B / C) is fully plausible at FC02.** "
            "Failure flags per candidate:\n\n"
        )
        for r in (A, B, C):
            failed = [k for k in (
                "FAR_plausible", "T4_plausible", "T30_plausible", "T50_plausible",
                "T45_acceptable", "EPR_acceptable",
                "no_PR_clamp", "no_Wc_clamp",
                "eta_hpt_realistic", "eta_lpt_realistic",
            ) if not r[k]]
            recommended += f"- **{r['label']}**: failed = {', '.join(failed) if failed else '(none)'}\n"
        recommended += (
            "\nGiven that the documented values cannot satisfy FC02 plausibility within V3.1b's "
            "explicit-closure architecture, **EPR should be demoted from a hard pressure gate to "
            "a diagnostic until V4 introduces a flow-matching solver.** T45 anchor remains a hard "
            "gate. C1 across 13 User Guide FCs should report EPR error per FC as a diagnostic, not "
            "an acceptance criterion. No YAML change is recommended at this stage."
        )

    md = f"""# C0d Documented C-MAPSS reference-engine initialization check — FC02

*Read-only diagnostic. No YAML written. No DS02 access. No optimizer.*
*FC02 is an external sanity check, not a fitting target.*

## 1. Source verification

Local-only search of `data/`, `docs/`, and repository PDFs for the documented
C-MAPSS reference-engine values:

| Reference value | Locally verified? | Source |
|---|---|---|
| 90,000 lb thrust class | **VERIFIED** | Saxena 2008, [data/CMAPSS/Damage Propagation Modeling.pdf](data/CMAPSS/Damage Propagation Modeling.pdf), p. 2 |
| Operating envelope (alt 0–40 K ft, M 0–0.9, T_SL −60..103 °F) | **VERIFIED** | Saxena 2008, p. 2 |
| BPR ≈ {DOC_BPR} | NOT FOUND locally | Cited via Frederick TM2007-215026 ref [11] in Saxena 2008 |
| OPR ≈ {DOC_OPR} | NOT FOUND locally | Same |
| Nf design speed ≈ {DOC_NF_DES_RPM} rpm | NOT FOUND locally | Same |
| Nc design speed ≈ {DOC_NC_DES_RPM} rpm | NOT FOUND locally | Same |
| Wc_fan_design ≈ {DOC_WCFAN_DES_KGS} kg/s | NOT FOUND locally | Same |
| Component PR / eta table (Fan / LPC / HPC / HPT / LPT) | NOT FOUND locally | Same |

The numerical values are USER-PROVIDED documented reference values and are
treated as such throughout this report. **Source verification pending acquisition
of Frederick et al., NASA/ARL TM2007-215026.** No internet browsing performed.

## 2. Frozen documented values used in C0d

| Quantity | Value |
|---|---|
| BPR | {DOC_BPR} |
| OPR (engine-level) | {DOC_OPR} |
| Nf design (rpm) | {DOC_NF_DES_RPM} |
| Nc design (rpm) | {DOC_NC_DES_RPM} |
| Wc_fan_design (kg/s) | {DOC_WCFAN_DES_KGS} |
| PR_fan / eta_fan | {DOC_PR_FAN} / {DOC_ETA_FAN} |
| PR_lpc / eta_lpc | {DOC_PR_LPC} / {DOC_ETA_LPC} |
| PR_hpc / eta_hpc | {DOC_PR_HPC_TABLE} / {DOC_ETA_HPC} |
| PR_hpt / eta_hpt | {DOC_PR_HPT} / {DOC_ETA_HPT} |
| PR_lpt / eta_lpt | {DOC_PR_LPT} / {DOC_ETA_LPT} |
| Implied OPR from PR table | **{IMPLIED_OPR_FROM_TABLE:.2f}** (≠ documented OPR {DOC_OPR}) |

NOTE: V3.1b uses Nf and Nc as INPUTS at FC02; PR_hpt and PR_lpt are **closure-determined**
from the shaft balances, not free parameters. The documented PR_hpt / PR_lpt are
listed for reference only.

## 3. Candidates

**A** — engine-level OPR=36 respected; PR_hpc derived as `36/(PR_fan × PR_lpc) = {DOC_OPR / (DOC_PR_FAN * DOC_PR_LPC):.3f}`
**B** — documented PR table verbatim; PR_hpc=21.817; implied OPR ≈ {IMPLIED_OPR_FROM_TABLE:.2f}
**C** — conservative hybrid; PR_fan=1.70, PR_lpc=1.20, PR_hpc=`36/(1.70×1.20)={DOC_OPR / (1.70 * 1.20):.3f}`; Wc_fan_design = 1500 kg/s
**D** — old baseline (BPR=5.5, Wc_fan=900, OPR=38.4) — FOR COMPARISON ONLY, NOT FOR ADOPTION

## 4. Per-candidate FC02 results

{sec_per_cand}

## 5. Q&A

### Q: Does BPR=8.4 improve EPR vs the old baseline (BPR=5.5)?

{q1}

### Q: Does Wc_fan_design ≈ 1658 kg/s produce plausible mass flow / FAR / T4 / T45 / T50?

{q2}

### Q: Is the documented component-PR table consistent with engine-level OPR ≈ 36?

{q3}

### Q: Which candidate is physically most defensible for C1 testing?

{recommended}

## 6. Plot index

| # | Plot | File |
|---|---|---|
| 20 | FC02 station temperature, all candidates                | `20_FC02_station_temperatures.png` |
| 21 | FC02 station P/P2 (log), all candidates                  | `21_FC02_station_pressure_ratios_log.png` |
| 22 | FC02 EPR decomposition waterfall, per candidate          | `22_FC02_EPR_waterfall.png` |
| 23 | FC02 spool work balance (HPC vs HPT, Fan+LPC vs LPT)     | `23_FC02_work_balance.png` |
| 24 | FC02 metric comparison bars                              | `24_FC02_metric_comparison_bars.png` |
| 25 | FC02 candidate metrics table                             | `25_FC02_metrics_table.png` |

CSV: `candidates_metrics.csv` (full per-candidate metrics).

---

*Stop. No YAML written. No DS02. No automatic adoption. Awaiting Robert review.*
"""
    p = OUT_DIR / "c0d_cmapss_documented_design_report.md"
    p.write_text(md, encoding="utf-8")
    print(f"  saved  {p}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("V3.1b C0d documented C-MAPSS reference-engine initialization check at FC02")
    print(f"Output dir: {OUT_DIR}")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fc02 = fc02_conditions_si()
    cands = candidates()

    print("\nEvaluating 4 candidates at FC02...")
    rows: list[dict] = []
    for key in ("A", "B", "C", "D"):
        spec = cands[key]
        # ASCII-safe stdout (Windows cp1252 can't print U+2014, U+2248, U+2022)
        ascii_label = spec["label"].encode("ascii", "replace").decode("ascii")
        print(f"  - {ascii_label}")
        rows.append(evaluate(spec, fc02))

    print("\nGenerating plots...")
    plot_station_temperatures(rows)
    plot_station_pressure_ratios(rows)
    plot_epr_waterfall(rows)
    plot_work_balance(rows)
    plot_metric_comparison(rows)
    plot_metrics_table(rows)

    print("\nWriting markdown report...")
    write_report(rows, fc02)

    print("\n" + "=" * 72)
    print("Done. No YAML written. No DS02. No automatic adoption.")
    print("=" * 72)


if __name__ == "__main__":
    main()
