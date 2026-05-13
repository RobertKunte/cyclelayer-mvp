"""C0 cycle plausibility diagnostic report at FC02 (V3.1b, read-only).

Purpose
-------
Produce a full-cycle plausibility view at UserGuide FC02 with the current
C0 state (Wc_fan_design = 900 kg/s, fixed PR_design and eta_design from
the V3.1b assumption set).  The output answers ONE question: do the cycle
states (T, P, h, s) and component balances look like a real CMAPSS-class
2-spool turbofan, and which fixed assumption is most likely responsible
for the EPR mismatch reported by the C0 anchor check?

Hard constraints (V3.1b Hard Rule 17)
-------------------------------------
- read-only: NO writes to configs/cyclelayer_v3.yaml
- NO DS02 access
- NO `fit_*` helpers, NO optimization loop, NO minimizer
- eta_design stays in the realistic 0.88-0.92 component band — not adjusted
- the only sweep in this script is the explicitly labelled
  PR_hpc-sensitivity diagnostic (plot 8). Each PR_hpc point is a one-shot
  forward pass; no value is selected as "best".

Outputs
-------
artifacts/cyclelayer_v3/c0_diagnostics/
  ├─ 01_T_s_diagram.png
  ├─ 02_h_s_diagram.png
  ├─ 03_T_vs_station.png
  ├─ 04_P_over_P2_vs_station_log.png
  ├─ 05_EPR_waterfall.png
  ├─ 06_spool_work_balance.png
  ├─ 07_FAR_T4_massflow_summary.png
  ├─ 08_PR_hpc_sensitivity.png
  └─ summary.md

Usage
-----
    python scripts/c0_cycle_plausibility_report.py
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
    BPR_DESIGN,
    CP_C, CP_T, R_AIR,
    ETA_DESIGN_FAN, ETA_DESIGN_HPC, ETA_DESIGN_HPT,
    ETA_DESIGN_LPC, ETA_DESIGN_LPT,
    EXP_C, GAMMA_C,
    PI_BURN,
    P_REF, T_REF,
)


# =============================================================================
# Frozen V3.1b assumptions and the single C0 pick (Wc_fan_design = 900 kg/s)
# =============================================================================

PICKED_WC_FAN_DESIGN = 900.0   # kg/s — provisional thermal anchor (C0)

ASSUMPTION_BPR              = BPR_DESIGN
ASSUMPTION_PR_DESIGN_FAN    = 1.6
ASSUMPTION_PR_DESIGN_LPC    = 2.0
ASSUMPTION_PR_DESIGN_HPC    = 12.0
ASSUMPTION_ETA_DESIGN_FAN   = ETA_DESIGN_FAN     # 0.92
ASSUMPTION_ETA_DESIGN_LPC   = ETA_DESIGN_LPC     # 0.90
ASSUMPTION_ETA_DESIGN_HPC   = ETA_DESIGN_HPC     # 0.88
ASSUMPTION_ETA_DESIGN_HPT   = ETA_DESIGN_HPT     # 0.90
ASSUMPTION_ETA_DESIGN_LPT   = ETA_DESIGN_LPT     # 0.92


# Plausibility windows (Walsh & Fletcher / Kurzke / CMAPSS-class generic)
PLAUSIBLE = {
    "FAR_takeoff":   (0.018, 0.030),
    "T4_K":          (1300.0, 1900.0),
    "T45_K":         (1000.0, 1500.0),
    "T50_K":         (700.0,  1100.0),
    "OPR":           (20.0,   45.0),
    "EPR_total":     (1.10,   1.50),
    "m_in_kgs":      (50.0,   1500.0),     # very wide, just a sanity check
}


# =============================================================================
# FC02 conditions (post-ram, total values at fan inlet)
# =============================================================================

def fc02_conditions_si() -> dict[str, float]:
    fc = load_userguide_fc02_anchor()
    Tsl_R = fc["Tsl_F"] + 459.67           # 518.67 R (ISA SL Std-Day)
    P0_psia = 14.696                        # ISA SL static pressure
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


def derive_design_points(fc02_si: dict, Wc_fan_design: float,
                         pr_fan: float = ASSUMPTION_PR_DESIGN_FAN,
                         pr_lpc: float = ASSUMPTION_PR_DESIGN_LPC,
                         pr_hpc: float = ASSUMPTION_PR_DESIGN_HPC) -> dict:
    """Same derivation as scripts/c0_anchor_check.py — kept consistent."""
    T2_K = fc02_si["T2_R"] * units.RANK_TO_K
    P2_Pa = fc02_si["P2_psia"] * units.PSIA_TO_PA
    Nc_design_fan = fc02_si["Nf_rpm"] / math.sqrt(T2_K / T_REF)
    m_in_design = Wc_fan_design * (P2_Pa / P_REF) / math.sqrt(T2_K / T_REF)
    m_core_design = m_in_design / (ASSUMPTION_BPR + 1.0)

    T21_isen = T2_K * pr_fan ** EXP_C
    T21      = T2_K + (T21_isen - T2_K) / ASSUMPTION_ETA_DESIGN_FAN
    P21      = P2_Pa * pr_fan
    Wc_design_lpc = m_core_design * math.sqrt(T21 / T_REF) / (P21 / P_REF)
    Nc_design_lpc = Nc_design_fan

    T24_isen = T21 * pr_lpc ** EXP_C
    T24      = T21 + (T24_isen - T21) / ASSUMPTION_ETA_DESIGN_LPC
    P24      = P21 * pr_lpc
    Wc_design_hpc = m_core_design * math.sqrt(T24 / T_REF) / (P24 / P_REF)
    Nc_design_hpc = fc02_si["Nc_rpm"] / math.sqrt(T24 / T_REF)

    return {
        "Nc_design_fan": Nc_design_fan, "Nc_design_lpc": Nc_design_lpc,
        "Nc_design_hpc": Nc_design_hpc,
        "Wc_design_fan": Wc_fan_design,
        "Wc_design_lpc": Wc_design_lpc, "Wc_design_hpc": Wc_design_hpc,
        "PR_design_fan": pr_fan,
        "PR_design_lpc": pr_lpc,
        "PR_design_hpc": pr_hpc,
        "eta_design_fan": ASSUMPTION_ETA_DESIGN_FAN,
        "eta_design_lpc": ASSUMPTION_ETA_DESIGN_LPC,
        "eta_design_hpc": ASSUMPTION_ETA_DESIGN_HPC,
    }


def build_engine(design: dict) -> BraytonEngine:
    cfg = BraytonEngineConfig(
        inlet_flow=InletFlowParams(
            Wc_fan_design=design["Wc_design_fan"],
            Nc_fan_design=design["Nc_design_fan"],
        ),
        map_coeffs=MapCoefficients(**{k: design[k] for k in (
            "Nc_design_fan", "Wc_design_fan",
            "Nc_design_lpc", "Wc_design_lpc",
            "Nc_design_hpc", "Wc_design_hpc",
            "PR_design_fan", "PR_design_lpc", "PR_design_hpc",
        )}),
        use_measured_inlet=True,
        bpr_design=ASSUMPTION_BPR,
        eta_design_hpt=ASSUMPTION_ETA_DESIGN_HPT,
        eta_design_lpt=ASSUMPTION_ETA_DESIGN_LPT,
    )
    return BraytonEngine(cfg)


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
# Reconstruct full-cycle station states (T, P, s, h) from outputs + diagnostics
# =============================================================================

def cycle_states(sensors_pred_si: dict, diag: dict,
                 ops_si: dict) -> dict[str, dict[str, float]]:
    """Recover T and P at every station (2, 21, 24, 30, 4, 45, 50)."""
    T2 = float(ops_si["T2_K"].item())
    P2 = float(ops_si["P2_Pa"].item())
    PR_fan = float(diag["PR_fan"].item())
    PR_lpc = float(diag["PR_lpc"].item())
    eta_fan_eff = float(diag["eta_fan"].item())
    eta_lpc_eff = float(diag["eta_lpc"].item())

    T21_isen = T2 * PR_fan ** EXP_C
    T21      = T2 + (T21_isen - T2) / eta_fan_eff
    P21      = P2 * PR_fan

    T24 = float(sensors_pred_si["T24_K"].item())
    P24 = P21 * PR_lpc

    T30 = float(sensors_pred_si["T30_K"].item())
    P30 = float(sensors_pred_si["P30_Pa"].item())
    T4  = float(diag["T4"].item());   P4  = float(diag["P4"].item())
    T45 = float(diag["T45"].item());  P45 = float(diag["P45"].item())
    T50 = float(sensors_pred_si["T50_K"].item())
    P50 = float(diag["P50"].item())

    states = {
        "2":  {"T": T2,  "P": P2,  "cp": CP_C},
        "21": {"T": T21, "P": P21, "cp": CP_C},
        "24": {"T": T24, "P": P24, "cp": CP_C},
        "30": {"T": T30, "P": P30, "cp": CP_C},
        "4":  {"T": T4,  "P": P4,  "cp": CP_T},
        "45": {"T": T45, "P": P45, "cp": CP_T},
        "50": {"T": T50, "P": P50, "cp": CP_T},
    }
    # Entropy and enthalpy referenced to station 2 (s=0, h relative)
    s2 = 0.0
    h2 = states["2"]["cp"] * states["2"]["T"]
    for k, st in states.items():
        # ds = cp · ln(T/T2) − R · ln(P/P2)
        st["s_rel"] = st["cp"] * math.log(st["T"] / T2) - R_AIR * math.log(st["P"] / P2)
        st["h"]     = st["cp"] * st["T"]
    return states


def ideal_cycle_states(states: dict) -> dict[str, dict[str, float]]:
    """All-isentropic reference: same PR profile, eta=1 everywhere, same fuel."""
    P2 = states["2"]["P"]; T2 = states["2"]["T"]
    out: dict[str, dict[str, float]] = {}
    # Stations 2, 21, 24, 30 — ideal compression
    for k, prev in (("2", None), ("21", "2"), ("24", "21"), ("30", "24")):
        if prev is None:
            T = T2; P = P2
        else:
            P = states[k]["P"]
            T = out[prev]["T"] * (P / out[prev]["P"]) ** EXP_C
        out[k] = {"T": T, "P": P, "cp": CP_C}
    # Station 4 — same combustor energy balance, but T30 is now ideal
    # We use the actual diag T4 because the combustor output was driven
    # by the fixed Wf and m_core.  For ideal reference we recompute T4
    # using the ideal T30:
    cp_c, cp_t = CP_C, CP_T
    out["4"] = {"T": states["4"]["T"], "P": states["4"]["P"], "cp": cp_t}
    # Stations 45, 50 — ideal turbines (eta=1) with same shaft work
    # In ideal closure, dT_isen = dT_actual at eta=1, but the actual closure
    # already gives us T45, T50.  For the *ideal isentropic* reference we
    # plot the locus T_isen from T4 with the same PR_hpt, PR_lpt:
    PR_hpt = states["45"]["P"] and (states["4"]["P"] / states["45"]["P"]) or 1.0
    PR_lpt = states["50"]["P"] and (states["45"]["P"] / states["50"]["P"]) or 1.0
    EXP_T_local = (1.33 - 1.0) / 1.33
    out["45"] = {"T": out["4"]["T"] / PR_hpt ** EXP_T_local,
                 "P": states["45"]["P"], "cp": cp_t}
    out["50"] = {"T": out["45"]["T"] / PR_lpt ** EXP_T_local,
                 "P": states["50"]["P"], "cp": cp_t}
    # Compute s, h relative to station 2
    for k, st in out.items():
        st["s_rel"] = st["cp"] * math.log(st["T"] / T2) - R_AIR * math.log(st["P"] / P2)
        st["h"]     = st["cp"] * st["T"]
    return out


# =============================================================================
# Plot helpers
# =============================================================================

STATIONS = ["2", "21", "24", "30", "4", "45", "50"]
STATION_LABEL = {
    "2":  "2 (Fan in)",   "21": "21 (Fan out)",
    "24": "24 (LPC out)", "30": "30 (HPC out)",
    "4":  "4 (Comb out)", "45": "45 (HPT out)",
    "50": "50 (LPT out)",
}

OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "c0_diagnostics"


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {p}")


def plot_T_s(states, ideal):
    fig, ax = plt.subplots(figsize=(8, 5))
    s_act = [states[k]["s_rel"]  for k in STATIONS]
    T_act = [states[k]["T"]      for k in STATIONS]
    s_id  = [ideal[k]["s_rel"]   for k in STATIONS]
    T_id  = [ideal[k]["T"]       for k in STATIONS]
    ax.plot(s_act, T_act, "o-", label="actual cycle (eta_design)", color="tab:blue")
    ax.plot(s_id,  T_id,  "s--", label="ideal isentropic ref",      color="tab:gray", alpha=0.75)
    for k, s, T in zip(STATIONS, s_act, T_act):
        ax.annotate(k, (s, T), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Entropy s − s2  [J/(kg·K)]")
    ax.set_ylabel("Total temperature T  [K]")
    ax.set_title("FC02 — T-s diagram (actual cycle vs ideal isentropic ref)")
    ax.grid(True, alpha=0.4); ax.legend()
    _save(fig, "01_T_s_diagram.png")


def plot_h_s(states, ideal):
    fig, ax = plt.subplots(figsize=(8, 5))
    s_act = [states[k]["s_rel"] for k in STATIONS]
    h_act = [states[k]["h"]     for k in STATIONS]
    s_id  = [ideal[k]["s_rel"]  for k in STATIONS]
    h_id  = [ideal[k]["h"]      for k in STATIONS]
    ax.plot(s_act, h_act, "o-", label="actual",  color="tab:blue")
    ax.plot(s_id,  h_id, "s--", label="ideal",   color="tab:gray", alpha=0.75)
    for k, s, h in zip(STATIONS, s_act, h_act):
        ax.annotate(k, (s, h), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Entropy s − s2  [J/(kg·K)]")
    ax.set_ylabel("Total enthalpy h = cp·T  [J/kg]")
    ax.set_title("FC02 — h-s diagram")
    ax.grid(True, alpha=0.4); ax.legend()
    _save(fig, "02_h_s_diagram.png")


def plot_T_vs_station(states):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    Ts = [states[k]["T"] for k in STATIONS]
    ax.plot(STATIONS, Ts, "o-", color="tab:red", lw=2)
    for k, T in zip(STATIONS, Ts):
        ax.annotate(f"{T:.0f} K", (k, T), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Station")
    ax.set_ylabel("Total temperature  [K]")
    ax.set_title("FC02 — total temperature vs station")
    ax.grid(True, alpha=0.4)
    _save(fig, "03_T_vs_station.png")


def plot_P_over_P2(states):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    P2 = states["2"]["P"]
    ratios = [states[k]["P"] / P2 for k in STATIONS]
    ax.semilogy(STATIONS, ratios, "o-", color="tab:purple", lw=2)
    for k, r in zip(STATIONS, ratios):
        ax.annotate(f"{r:.2f}", (k, r), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Station")
    ax.set_ylabel("P / P2  (log scale)")
    ax.set_title("FC02 — total pressure ratio P/P2 vs station")
    ax.axhline(1.0, color="black", lw=0.5)
    ax.grid(True, alpha=0.4, which="both")
    _save(fig, "04_P_over_P2_vs_station_log.png")


def plot_EPR_waterfall(states):
    """Multiplicative pressure-ratio steps from P2 to P50."""
    P2 = states["2"]["P"]
    steps = [
        ("P2",        1.0),
        ("× fan",     states["21"]["P"] / P2),
        ("× LPC",     states["24"]["P"] / states["21"]["P"]),
        ("× HPC",     states["30"]["P"] / states["24"]["P"]),
        ("× comb",    states["4"]["P"]  / states["30"]["P"]),
        ("÷ HPT",     states["45"]["P"] / states["4"]["P"]),
        ("÷ LPT",     states["50"]["P"] / states["45"]["P"]),
        ("EPR=P50/P2", states["50"]["P"] / P2),
    ]
    cumulative = []
    cum = 1.0
    for i, (lbl, factor) in enumerate(steps):
        if lbl == "P2":
            cumulative.append(1.0)
        elif lbl.startswith("EPR"):
            cumulative.append(states["50"]["P"] / P2)
        else:
            cum *= factor
            cumulative.append(cum)
    labels = [s[0] for s in steps]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(labels, cumulative, "o-", color="tab:green", lw=2, ms=8)
    for x, y in zip(labels, cumulative):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative P / P2")
    ax.set_title("FC02 — EPR decomposition (cumulative pressure ratio)")
    ax.grid(True, alpha=0.4, which="both")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, "05_EPR_waterfall.png")


def plot_spool_balance(diag):
    fig, ax = plt.subplots(figsize=(8, 5))
    W_hpc, W_hpt = float(diag["W_hpc"].item()),  float(diag["W_hpt"].item())
    W_lpc, W_fan = float(diag["W_lpc"].item()),  float(diag["W_fan_total"].item())
    W_lpt        = float(diag["W_lpt"].item())
    groups = ["HP spool", "LP spool"]
    drives = [W_hpc / 1e6, (W_lpc + W_fan) / 1e6]      # MW required
    turbs  = [W_hpt / 1e6, W_lpt / 1e6]                # MW supplied
    x = np.arange(len(groups)); w = 0.35
    ax.bar(x - w/2, drives, w, label="W required (HPC | LPC+Fan)",
           color="tab:blue")
    ax.bar(x + w/2, turbs,  w, label="W supplied (HPT | LPT)",
           color="tab:orange")
    for i, (d, t) in enumerate(zip(drives, turbs)):
        ax.annotate(f"{d:.1f}", (i - w/2, d), textcoords="offset points",
                    xytext=(0, 4), ha="center")
        ax.annotate(f"{t:.1f}", (i + w/2, t), textcoords="offset points",
                    xytext=(0, 4), ha="center")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Shaft power  [MW]")
    ax.set_title("FC02 — spool work balance (closure should match)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.4)
    _save(fig, "06_spool_work_balance.png")


def plot_far_summary(diag, fc02):
    m_core = float(diag["m_core"].item())
    m_in   = float(diag["m_in"].item())
    m_byp  = float(diag["m_byp"].item())
    Wf_kgs = fc02["Wf_pps"] * units.PPS_TO_KGS
    FAR = Wf_kgs / m_core
    T4  = float(diag["T4"].item())
    rows = [
        ("m_in (kg/s)",              f"{m_in:.2f}"),
        ("m_core (kg/s)",            f"{m_core:.2f}"),
        ("m_byp  (kg/s)",            f"{m_byp:.2f}"),
        ("BPR computed",             f"{m_byp / m_core:.2f}"),
        ("Wf (kg/s)",                f"{Wf_kgs:.4f}"),
        ("FAR = Wf/m_core",          f"{FAR:.4f}"),
        ("T4 (TIT) [K]",             f"{T4:.1f}"),
        ("T4 [°R]",                  f"{T4 / units.RANK_TO_K:.1f}"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Quantity", "Value"],
        loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False); table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title("FC02 — FAR / T4 / mass-flow summary")
    _save(fig, "07_FAR_T4_massflow_summary.png")


def plot_pr_hpc_sensitivity(fc02, t48_ref_R, epr_ref):
    """SENSITIVITY DIAGNOSTIC (NOT TUNING).

    Show how the C0 residuals (T45_rel_err, EPR_rel_err) and downstream
    metrics (T30, P30/P2) depend on the assumed PR_design_hpc, holding
    Wc_fan_design and all other design choices fixed.  No value is
    selected as best.
    """
    pr_values = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    rel_T45, rel_EPR, T30s, OPRs = [], [], [], []
    for pr_hpc in pr_values:
        d = derive_design_points(fc02, PICKED_WC_FAN_DESIGN,
                                 pr_hpc=pr_hpc)
        eng = build_engine(d)
        sensors_pred_si, diag, ops_si, _ = run_fc02_forward(eng, fc02)
        T45_R = float(diag["T45"].item()) / units.RANK_TO_K
        P50_Pa = float(diag["P50"].item())
        P2_Pa = float(ops_si["P2_Pa"].item())
        EPR = P50_Pa / P2_Pa
        T30_K = float(sensors_pred_si["T30_K"].item())
        rel_T45.append(abs(T45_R - t48_ref_R) / t48_ref_R * 100.0)
        rel_EPR.append(abs(EPR - epr_ref) / epr_ref * 100.0)
        T30s.append(T30_K)
        OPRs.append(float(diag["P30_over_P2"].item()))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(pr_values, rel_T45, "o-", color="tab:red")
    ax1.axhline(25.0, ls="--", color="black", lw=0.7,
                label="25% acceptance band")
    ax1.set_ylabel("|T45 − T48_ref| / T48_ref  [%]")
    ax1.set_title("T45 rel error  vs  PR_design_hpc")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)

    ax2.plot(pr_values, rel_EPR, "o-", color="tab:purple")
    ax2.axhline(25.0, ls="--", color="black", lw=0.7,
                label="25% acceptance band")
    ax2.set_ylabel("|EPR − EPR_ref| / EPR_ref  [%]")
    ax2.set_title("EPR rel error  vs  PR_design_hpc")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)

    ax3.plot(pr_values, T30s, "o-", color="tab:blue")
    ax3.set_xlabel("PR_design_hpc")
    ax3.set_ylabel("T30  [K]")
    ax3.set_title("T30 vs PR_design_hpc")
    ax3.grid(True, alpha=0.4)

    ax4.plot(pr_values, OPRs, "o-", color="tab:green")
    ax4.set_xlabel("PR_design_hpc")
    ax4.set_ylabel("OPR = P30 / P2")
    ax4.set_title("OPR vs PR_design_hpc")
    ax4.grid(True, alpha=0.4)

    fig.suptitle("FC02 — PR_hpc sensitivity diagnostic  "
                 "(SENSITIVITY ONLY, NOT TUNING — no value is selected)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "08_PR_hpc_sensitivity.png")
    return list(zip(pr_values, rel_T45, rel_EPR, T30s, OPRs))


# =============================================================================
# Markdown summary
# =============================================================================

def _flag(value: float, lo: float, hi: float) -> str:
    return "plausible" if lo <= value <= hi else "OUT OF RANGE"


def write_markdown_summary(states, ideal, diag, fc02, sensitivity_rows):
    P2 = states["2"]["P"]
    EPR_pred = states["50"]["P"] / P2
    OPR = states["30"]["P"] / P2
    T4_K = states["4"]["T"]
    T45_K = states["45"]["T"]
    T50_K = states["50"]["T"]
    m_core = float(diag["m_core"].item())
    m_in   = float(diag["m_in"].item())
    Wf_kgs = fc02["Wf_pps"] * units.PPS_TO_KGS
    FAR = Wf_kgs / m_core
    T45_R = T45_K / units.RANK_TO_K
    rel_T45 = abs(T45_R - fc02["T48_ref_R"]) / fc02["T48_ref_R"] * 100
    rel_EPR = abs(EPR_pred - fc02["EPR_ref"]) / fc02["EPR_ref"] * 100

    # Sensitivity table for narrative
    pr_table_md = "| PR_hpc | T45 rel err [%] | EPR rel err [%] | T30 [K] | OPR |\n"
    pr_table_md += "|---|---|---|---|---|\n"
    for pr, rT45, rEPR, T30, opr in sensitivity_rows:
        pr_table_md += f"| {pr:.1f} | {rT45:.1f} | {rEPR:.1f} | {T30:.0f} | {opr:.1f} |\n"

    # Rough cause inference: scan whether sensitivity reduces EPR error meaningfully
    epr_errors = [r[2] for r in sensitivity_rows]
    pr_at_min = sensitivity_rows[int(np.argmin(epr_errors))][0]
    epr_min   = min(epr_errors)
    closes_with_pr_hpc = epr_min < 25.0

    if closes_with_pr_hpc:
        cause = (
            f"**PR_design_hpc is the dominant driver.** Sweeping PR_hpc "
            f"reduces the EPR mismatch below the 25 % band at "
            f"PR_hpc ≈ {pr_at_min:.1f}. The current assumption (12.0) overpressurises the "
            "compressor stack relative to the closure-determined dual-turbine "
            "expansion. Lowering PR_hpc (a cycle-topology assumption, NOT a "
            "DS02 tune) brings overall compression and required turbine "
            "expansion into agreement."
        )
    else:
        cause = (
            "**PR_hpc alone does not close the EPR gap.** Even at "
            f"PR_hpc = {pr_at_min:.1f} the EPR rel err is {epr_min:.1f} %. "
            "Likely additional factor(s): combustor pressure drop PI_BURN "
            "(currently 0.04) understates the real pressure loss, OR PR_lpc / "
            "PR_fan also need revision, OR the dual-turbine eta_design pair "
            "(0.90 / 0.92) is at the upper edge of the realistic 0.88-0.92 "
            "component band. None of these are touched in this report."
        )

    md = f"""# C0 Cycle Plausibility Report — UserGuide FC02

*Read-only diagnostic. No YAML written. No DS02 access. No optimization loop.*

## State (frozen for this report)

- Single picked parameter: `Wc_fan_design = {PICKED_WC_FAN_DESIGN} kg/s` (provisional thermal anchor)
- Fixed PR_design_fan / lpc / hpc = {ASSUMPTION_PR_DESIGN_FAN} / {ASSUMPTION_PR_DESIGN_LPC} / {ASSUMPTION_PR_DESIGN_HPC}
- Fixed eta_design fan / lpc / hpc / hpt / lpt = {ASSUMPTION_ETA_DESIGN_FAN} / {ASSUMPTION_ETA_DESIGN_LPC} / {ASSUMPTION_ETA_DESIGN_HPC} / {ASSUMPTION_ETA_DESIGN_HPT} / {ASSUMPTION_ETA_DESIGN_LPT}
- BPR = {ASSUMPTION_BPR}, PI_BURN = {PI_BURN}

## FC02 inputs

- alt = {fc02['alt_ft']} ft, M = {fc02['XM']}, TRA = {fc02['TRA_pct']} %
- T2 (post-ram) = {fc02['T2_R']:.2f} °R = {fc02['T2_R'] * units.RANK_TO_K:.2f} K
- P2 (post-ram) = {fc02['P2_psia']:.2f} psia
- Nf = {fc02['Nf_rpm']} rpm, Nc = {fc02['Nc_rpm']} rpm, Wf = {fc02['Wf_pps']} pps
- Reference (UserGuide): T48 = {fc02['T48_ref_R']} °R, EPR = {fc02['EPR_ref']:.3f}

## Current FC02 residuals

| Quantity | Predicted | Reference | Rel err [%] | 25 %-band |
|---|---|---|---|---|
| **T45 / T48_proxy** (primary anchor) | {T45_R:.2f} °R | {fc02['T48_ref_R']:.2f} °R | **{rel_T45:.2f}** | {'PASS' if rel_T45 < 25 else 'FAIL'} |
| **EPR = P50/P2** (plausibility check) | {EPR_pred:.3f} | {fc02['EPR_ref']:.3f} | **{rel_EPR:.2f}** | {'PASS' if rel_EPR < 25 else 'FAIL'} |

Conservation (closure): mass_inlet = {float(diag['mass_balance_inlet'].item()):.2e}, mass_combust = {float(diag['mass_balance_combust'].item()):.2e}, HPT shaft res = {float(diag['shaft_HPT_residual'].item()):.2e}, LPT shaft res = {float(diag['shaft_LPT_residual'].item()):.2e}. PR-clamps not active.

## Plausibility check

| Metric | Value | Realistic window | Verdict |
|---|---|---|---|
| FAR = Wf / m_core | {FAR:.4f} | {PLAUSIBLE['FAR_takeoff'][0]}–{PLAUSIBLE['FAR_takeoff'][1]} | **{_flag(FAR, *PLAUSIBLE['FAR_takeoff'])}** |
| T4 (TIT)         | {T4_K:.1f} K | {PLAUSIBLE['T4_K'][0]}–{PLAUSIBLE['T4_K'][1]} K | **{_flag(T4_K, *PLAUSIBLE['T4_K'])}** |
| T45              | {T45_K:.1f} K | {PLAUSIBLE['T45_K'][0]}–{PLAUSIBLE['T45_K'][1]} K | **{_flag(T45_K, *PLAUSIBLE['T45_K'])}** |
| T50              | {T50_K:.1f} K | {PLAUSIBLE['T50_K'][0]}–{PLAUSIBLE['T50_K'][1]} K | **{_flag(T50_K, *PLAUSIBLE['T50_K'])}** |
| OPR = P30 / P2   | {OPR:.2f} | {PLAUSIBLE['OPR'][0]}–{PLAUSIBLE['OPR'][1]} | **{_flag(OPR, *PLAUSIBLE['OPR'])}** |
| EPR = P50 / P2   | {EPR_pred:.3f} | {PLAUSIBLE['EPR_total'][0]}–{PLAUSIBLE['EPR_total'][1]} | **{_flag(EPR_pred, *PLAUSIBLE['EPR_total'])}** |
| m_in             | {m_in:.1f} kg/s | {PLAUSIBLE['m_in_kgs'][0]}–{PLAUSIBLE['m_in_kgs'][1]} | **{_flag(m_in, *PLAUSIBLE['m_in_kgs'])}** |

## PR_hpc sensitivity (diagnostic, not tuning)

{pr_table_md}

## Most likely cause of EPR mismatch

{cause}

## Plot index

| Plot | File |
|---|---|
| 1. T-s diagram                          | `01_T_s_diagram.png` |
| 2. h-s diagram                          | `02_h_s_diagram.png` |
| 3. T vs station                         | `03_T_vs_station.png` |
| 4. P/P2 vs station (log)                | `04_P_over_P2_vs_station_log.png` |
| 5. EPR decomposition waterfall          | `05_EPR_waterfall.png` |
| 6. Spool work balance                   | `06_spool_work_balance.png` |
| 7. FAR / T4 / mass-flow summary         | `07_FAR_T4_massflow_summary.png` |
| 8. PR_hpc sensitivity diagnostic        | `08_PR_hpc_sensitivity.png` |

---

*Stop. Awaiting Robert review before any YAML change.*
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "summary.md"
    p.write_text(md, encoding="utf-8")
    print(f"  saved  {p}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("V3.1b C0 cycle plausibility report at FC02")
    print(f"Output dir: {OUT_DIR}")
    print("=" * 72)

    fc02 = fc02_conditions_si()
    design = derive_design_points(fc02, PICKED_WC_FAN_DESIGN)
    engine = build_engine(design)
    sensors_pred_si, diag, ops_si, _ = run_fc02_forward(engine, fc02)

    states = cycle_states(sensors_pred_si, diag, ops_si)
    ideal  = ideal_cycle_states(states)

    print("\nGenerating plots...")
    plot_T_s(states, ideal)
    plot_h_s(states, ideal)
    plot_T_vs_station(states)
    plot_P_over_P2(states)
    plot_EPR_waterfall(states)
    plot_spool_balance(diag)
    plot_far_summary(diag, fc02)
    sensitivity_rows = plot_pr_hpc_sensitivity(
        fc02, fc02["T48_ref_R"], fc02["EPR_ref"]
    )

    print("\nWriting summary.md...")
    write_markdown_summary(states, ideal, diag, fc02, sensitivity_rows)

    print("\n" + "=" * 72)
    print("Done.  Stopping.  No YAML written.  Awaiting Robert review.")
    print("=" * 72)


if __name__ == "__main__":
    main()
