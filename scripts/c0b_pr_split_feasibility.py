"""C0b PR-split feasibility diagnostic at FC02 (V3.1b, read-only).

Purpose
-------
Sweep the 3-D grid of compressor pressure-ratio assumptions
(PR_design_fan × PR_design_lpc × PR_design_hpc), evaluate every combination
at UserGuide FC02 with theta=1.0, and report which combinations are
feasible against the plausibility windows.  Report TOP CANDIDATES under
several criteria but DO NOT automatically pick one.

Hard constraints (V3.1b Hard Rule 17)
-------------------------------------
- read-only: NO writes to configs/cyclelayer_v3.yaml
- NO DS02 access
- NO `fit_*` helpers, NO optimization loop, NO minimizer, NO automatic
  parameter selection
- `Wc_fan_design = 900 kg/s` stays as the provisional thermal anchor
- `eta_design_hpt = 0.90`, `eta_design_lpt = 0.92` stay in the realistic
  component band (0.88–0.92), unchanged
- The 3-D grid sweep is a SENSITIVITY DIAGNOSTIC.  Each grid point is a
  one-shot forward pass.  No iteration, no minimizer.

Outputs
-------
artifacts/cyclelayer_v3/c0b_pr_split/
  ├─ 09_EPR_err_heatmaps.png
  ├─ 10_feasible_scatter_OPR_vs_EPR.png
  ├─ 11_station_pressure_top5.png
  ├─ 12_station_temperature_top5.png
  ├─ all_candidates.csv
  ├─ feasible_candidates.csv
  └─ summary.md
"""

from __future__ import annotations

import math
from itertools import product
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
    BPR_DESIGN,
    ETA_DESIGN_FAN, ETA_DESIGN_HPC, ETA_DESIGN_HPT,
    ETA_DESIGN_LPC, ETA_DESIGN_LPT,
    EXP_C, GAMMA_C,
    P_REF, T_REF,
)


# =============================================================================
# Frozen V3.1b state
# =============================================================================

PICKED_WC_FAN_DESIGN = 900.0           # kg/s — provisional thermal anchor
ASSUMPTION_BPR              = BPR_DESIGN
ASSUMPTION_ETA_DESIGN_FAN   = ETA_DESIGN_FAN     # 0.92
ASSUMPTION_ETA_DESIGN_LPC   = ETA_DESIGN_LPC     # 0.90
ASSUMPTION_ETA_DESIGN_HPC   = ETA_DESIGN_HPC     # 0.88
ASSUMPTION_ETA_DESIGN_HPT   = ETA_DESIGN_HPT     # 0.90
ASSUMPTION_ETA_DESIGN_LPT   = ETA_DESIGN_LPT     # 0.92


# =============================================================================
# Sweep grid (as specified in C0b decision)
# =============================================================================

PR_FAN_GRID = [1.40, 1.45, 1.50, 1.55, 1.60, 1.65]
PR_LPC_GRID = [1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
PR_HPC_GRID = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]


# =============================================================================
# Plausibility windows (Walsh & Fletcher / Kurzke / CMAPSS-class generic)
# =============================================================================

OPR_LO,  OPR_HI  = 25.0, 35.0
T4_LO,   T4_HI   = 1300.0, 1900.0
T30_LO,  T30_HI  = 600.0, 1000.0
T50_LO,  T50_HI  = 700.0, 1100.0
T45_ERR_MAX_PCT  = 25.0
EPR_ERR_MAX_PCT  = 25.0
ETA_BAND_LO, ETA_BAND_HI = 0.88, 0.92

# CFM56-like OPR window (per the C0b spec)
CFM56_OPR_LO, CFM56_OPR_HI = 30.0, 33.0


# =============================================================================
# FC02 conditions and engine builder (consistent with c0_anchor_check.py)
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


def derive_design_points(fc02, Wc_fan_design, pr_fan, pr_lpc, pr_hpc) -> dict:
    T2_K = fc02["T2_R"] * units.RANK_TO_K
    P2_Pa = fc02["P2_psia"] * units.PSIA_TO_PA
    Nc_design_fan = fc02["Nf_rpm"] / math.sqrt(T2_K / T_REF)
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
    Nc_design_hpc = fc02["Nc_rpm"] / math.sqrt(T24 / T_REF)

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


def build_engine(design) -> BraytonEngine:
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


def run_fc02_forward(engine, fc02):
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
    return sensors_pred_si, diag, ops_si


# =============================================================================
# Sweep
# =============================================================================

def sweep_grid(fc02) -> pd.DataFrame:
    rows = []
    for pr_fan, pr_lpc, pr_hpc in product(PR_FAN_GRID, PR_LPC_GRID, PR_HPC_GRID):
        design = derive_design_points(fc02, PICKED_WC_FAN_DESIGN,
                                      pr_fan, pr_lpc, pr_hpc)
        engine = build_engine(design)
        sensors_pred_si, diag, ops_si = run_fc02_forward(engine, fc02)

        # Pull all metrics
        T2 = float(ops_si["T2_K"].item())
        P2 = float(ops_si["P2_Pa"].item())
        T24 = float(sensors_pred_si["T24_K"].item())
        T30 = float(sensors_pred_si["T30_K"].item())
        T50 = float(sensors_pred_si["T50_K"].item())
        P30 = float(sensors_pred_si["P30_Pa"].item())
        T4  = float(diag["T4"].item())
        T45 = float(diag["T45"].item())
        P45 = float(diag["P45"].item())
        P50 = float(diag["P50"].item())
        OPR = float(diag["P30_over_P2"].item())
        EPR = P50 / P2
        T45_R = T45 / units.RANK_TO_K

        T45_err_pct = abs(T45_R - fc02["T48_ref_R"]) / fc02["T48_ref_R"] * 100.0
        EPR_err_pct = abs(EPR - fc02["EPR_ref"]) / fc02["EPR_ref"] * 100.0

        # P21 and P24 from design (same as in BraytonEngine forward at theta=1)
        P21 = P2 * pr_fan
        P24 = P21 * pr_lpc

        # Shaft balance
        W_hpc = float(diag["W_hpc"].item())
        W_hpt = float(diag["W_hpt"].item())
        W_lpc = float(diag["W_lpc"].item())
        W_fan_total = float(diag["W_fan_total"].item())
        W_lpt = float(diag["W_lpt"].item())
        shaft_HPT_resid = float(diag["shaft_HPT_residual"].item())
        shaft_LPT_resid = float(diag["shaft_LPT_residual"].item())

        # Clamps
        clamp_frac = {
            f"frac_{k}": float(diag[f"frac_PR_{k}_clamped"].item())
            for k in ("fan", "lpc", "hpc", "hpt", "lpt")
        }
        clamps_active = any(v > 0.0 for v in clamp_frac.values())

        # Effective etas (these stay at design values when theta=1.0 + design in band)
        eta_hpt = float(diag["eta_hpt"].item())
        eta_lpt = float(diag["eta_lpt"].item())
        PR_hpt = float(diag["PR_hpt"].item())
        PR_lpt = float(diag["PR_lpt"].item())

        rows.append({
            "PR_fan": pr_fan, "PR_lpc": pr_lpc, "PR_hpc": pr_hpc,
            "OPR": OPR, "EPR": EPR,
            "T45_R": T45_R, "T45_err_pct": T45_err_pct,
            "EPR_err_pct": EPR_err_pct,
            "T24_K": T24, "T30_K": T30, "T4_K": T4, "T50_K": T50,
            "P24_over_P2": P24 / P2, "P30_over_P2": P30 / P2,
            "PR_hpt": PR_hpt, "PR_lpt": PR_lpt,
            "W_hpc_MW": W_hpc / 1e6, "W_hpt_MW": W_hpt / 1e6,
            "W_lpc_MW": W_lpc / 1e6, "W_fan_total_MW": W_fan_total / 1e6,
            "W_lpt_MW": W_lpt / 1e6,
            "shaft_HPT_resid_W": shaft_HPT_resid,
            "shaft_LPT_resid_W": shaft_LPT_resid,
            "shaft_HPT_resid_rel": abs(shaft_HPT_resid) / max(abs(W_hpc), 1.0),
            "shaft_LPT_resid_rel": abs(shaft_LPT_resid) / max(abs(W_lpc + W_fan_total), 1.0),
            "eta_hpt": eta_hpt, "eta_lpt": eta_lpt,
            **clamp_frac,
            "clamps_active": clamps_active,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Filter / candidate selection (REPORTING ONLY — no automatic adoption)
# =============================================================================

def feasibility_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["OPR"]         >= OPR_LO)            & (df["OPR"]         <= OPR_HI) &
        (df["T45_err_pct"] <  T45_ERR_MAX_PCT)   &
        (df["EPR_err_pct"] <  EPR_ERR_MAX_PCT)   &
        (df["T4_K"]        >= T4_LO)             & (df["T4_K"]        <= T4_HI) &
        (df["T30_K"]       >= T30_LO)            & (df["T30_K"]       <= T30_HI) &
        (df["T50_K"]       >= T50_LO)            & (df["T50_K"]       <= T50_HI) &
        (df["eta_hpt"]     >= ETA_BAND_LO)       & (df["eta_hpt"]     <= ETA_BAND_HI) &
        (df["eta_lpt"]     >= ETA_BAND_LO)       & (df["eta_lpt"]     <= ETA_BAND_HI) &
        (~df["clamps_active"])
    )


def best_candidates(df_all: pd.DataFrame, df_feas: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return reporting-only top picks per criterion (NOT for automatic adoption).

    When the strict-feasibility set is empty, also returns near-feasible
    candidates from the unfiltered grid so Robert can still see the
    structure of the answer.
    """
    out: dict[str, pd.DataFrame] = {}

    # 1) Min EPR error in plausible OPR (= same as feasibility OPR window)
    out["min_EPR_err"] = df_feas.sort_values("EPR_err_pct").head(5)

    # 2) Min combined T45/EPR error (feasible)
    df_combined = df_feas.copy()
    if not df_combined.empty:
        df_combined["combined_err"] = np.sqrt(
            df_combined["T45_err_pct"] ** 2 + df_combined["EPR_err_pct"] ** 2
        )
    out["min_combined_err"] = df_combined.sort_values("combined_err").head(5) \
        if not df_combined.empty else pd.DataFrame()

    # 3) Closest CFM56-like OPR window (feasible)
    cfm = df_feas[(df_feas["OPR"] >= CFM56_OPR_LO) & (df_feas["OPR"] <= CFM56_OPR_HI)]
    out["closest_CFM56_OPR"] = cfm.sort_values("EPR_err_pct").head(5) \
        if not cfm.empty else pd.DataFrame()

    # ── Near-feasible references (for diagnosis when strict-feasibility is empty) ──
    # 4) Min EPR error over the whole grid (no feasibility filter)
    out["unconstrained_min_EPR"] = df_all.sort_values("EPR_err_pct").head(5)

    # 5) Min EPR error within the OPR plausibility window (no other filters)
    in_opr = df_all[(df_all["OPR"] >= OPR_LO) & (df_all["OPR"] <= OPR_HI)]
    out["min_EPR_in_OPR_window"] = in_opr.sort_values("EPR_err_pct").head(5) \
        if not in_opr.empty else pd.DataFrame()

    return out


def per_criterion_pass_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-criterion isolated pass rate over the unfiltered grid."""
    masks = {
        f"OPR in [{OPR_LO}, {OPR_HI}]":
            (df["OPR"] >= OPR_LO) & (df["OPR"] <= OPR_HI),
        f"T45 err < {T45_ERR_MAX_PCT}%":
            df["T45_err_pct"] < T45_ERR_MAX_PCT,
        f"EPR err < {EPR_ERR_MAX_PCT}%":
            df["EPR_err_pct"] < EPR_ERR_MAX_PCT,
        f"T4 in [{T4_LO}, {T4_HI}] K":
            (df["T4_K"] >= T4_LO) & (df["T4_K"] <= T4_HI),
        f"T30 in [{T30_LO}, {T30_HI}] K":
            (df["T30_K"] >= T30_LO) & (df["T30_K"] <= T30_HI),
        f"T50 in [{T50_LO}, {T50_HI}] K":
            (df["T50_K"] >= T50_LO) & (df["T50_K"] <= T50_HI),
        "no PR clamp active":
            ~df["clamps_active"],
    }
    n = len(df)
    rows = []
    for name, m in masks.items():
        rows.append({
            "criterion": name,
            "pass":      int(m.sum()),
            "total":     n,
            "pass_rate": f"{m.sum() / n * 100:.1f}%",
        })
    return pd.DataFrame(rows)


# =============================================================================
# Plot helpers
# =============================================================================

OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "c0b_pr_split"
STATIONS = ["2", "21", "24", "30", "4", "45", "50"]


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {p}")


def plot_epr_heatmaps(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, pr_fan in zip(axes.flat, PR_FAN_GRID):
        sub = df[df["PR_fan"] == pr_fan]
        # Pivot: rows=PR_hpc, cols=PR_lpc, values=EPR_err_pct
        piv = sub.pivot_table(index="PR_hpc", columns="PR_lpc",
                              values="EPR_err_pct")
        im = ax.imshow(piv.values, origin="lower", aspect="auto",
                       cmap="viridis_r", vmin=0, vmax=80)
        ax.set_xticks(range(len(PR_LPC_GRID)))
        ax.set_xticklabels([f"{v:.1f}" for v in PR_LPC_GRID])
        ax.set_yticks(range(len(PR_HPC_GRID)))
        ax.set_yticklabels([f"{int(v)}" for v in PR_HPC_GRID])
        ax.set_xlabel("PR_lpc"); ax.set_ylabel("PR_hpc")
        ax.set_title(f"PR_fan = {pr_fan}")
        # Annotate cells with the EPR err
        for i, hp in enumerate(PR_HPC_GRID):
            for j, lp in enumerate(PR_LPC_GRID):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color="white" if v > 35 else "black", fontsize=7)
    fig.suptitle("FC02 — EPR rel error [%] heatmap, faceted by PR_fan  "
                 "(SENSITIVITY ONLY, NOT TUNING)", fontsize=11)
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="|EPR − EPR_ref| / EPR_ref [%]")
    _save(fig, "09_EPR_err_heatmaps.png")


def plot_feasibility_scatter(df: pd.DataFrame):
    feas = feasibility_mask(df)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc1 = ax.scatter(df.loc[~feas, "OPR"], df.loc[~feas, "EPR_err_pct"],
                     c="lightgray", s=14, alpha=0.5, label="infeasible")
    sc2 = ax.scatter(df.loc[feas, "OPR"], df.loc[feas, "EPR_err_pct"],
                     c=df.loc[feas, "T45_err_pct"], s=42, cmap="RdYlGn_r",
                     vmin=0, vmax=15, edgecolors="black", linewidths=0.4,
                     label="feasible")
    ax.axhline(EPR_ERR_MAX_PCT, color="black", lw=0.7, ls="--",
               label=f"{EPR_ERR_MAX_PCT}% EPR band")
    ax.axvspan(OPR_LO, OPR_HI, alpha=0.10, color="green",
               label=f"OPR band [{OPR_LO}, {OPR_HI}]")
    ax.axvspan(CFM56_OPR_LO, CFM56_OPR_HI, alpha=0.20, color="blue",
               label=f"CFM56-like OPR [{CFM56_OPR_LO}, {CFM56_OPR_HI}]")
    ax.set_xlabel("OPR = P30 / P2")
    ax.set_ylabel("EPR rel err [%]")
    ax.set_yscale("log")
    ax.set_title("FC02 — feasibility scatter:  OPR vs EPR error  "
                 "(color = T45 rel err [%], feasible only)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.4, which="both")
    if (df["EPR_err_pct"] > 0).any():
        cbar = fig.colorbar(sc2, ax=ax)
        cbar.set_label("T45 rel err [%]  (feasible)")
    _save(fig, "10_feasible_scatter_OPR_vs_EPR.png")


def plot_top5_stations(df_feas: pd.DataFrame, df_all: pd.DataFrame,
                        fc02, fname_p: str, fname_t: str):
    """Pressure and temperature plots for top-5 by min-combined-err.

    If the strict-feasibility set is empty, falls back to the top-5 within
    the OPR window [OPR_LO, OPR_HI] so the diagnostic still produces a
    plot — annotated as 'near-feasible' on the figure.
    """
    fallback_used = False
    if df_feas.empty:
        # Use the OPR-window near-feasible set instead of a blank plot.
        in_opr = df_all[(df_all["OPR"] >= OPR_LO) & (df_all["OPR"] <= OPR_HI)]
        df_feas = in_opr.copy()
        fallback_used = True
        if df_feas.empty:
            for fname in (fname_p, fname_t):
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, "no candidates in OPR window",
                        ha="center", va="center", fontsize=14)
                ax.axis("off")
                _save(fig, fname)
            return

    df = df_feas.copy()
    df["combined_err"] = np.sqrt(df["T45_err_pct"] ** 2 + df["EPR_err_pct"] ** 2)
    top = df.sort_values("combined_err").head(5)
    fb_note = "  [NEAR-FEASIBLE — strict feasibility set is empty]" if fallback_used else ""

    P_curves: list[tuple[str, list[float]]] = []
    T_curves: list[tuple[str, list[float]]] = []
    for _, row in top.iterrows():
        design = derive_design_points(fc02, PICKED_WC_FAN_DESIGN,
                                      row["PR_fan"], row["PR_lpc"], row["PR_hpc"])
        engine = build_engine(design)
        sensors_pred_si, diag, ops_si = run_fc02_forward(engine, fc02)

        T2 = float(ops_si["T2_K"].item());     P2 = float(ops_si["P2_Pa"].item())
        PR_fan = float(diag["PR_fan"].item()); PR_lpc = float(diag["PR_lpc"].item())
        eta_fan_eff = float(diag["eta_fan"].item())
        T21_isen = T2 * PR_fan ** EXP_C
        T21      = T2 + (T21_isen - T2) / eta_fan_eff
        P21      = P2 * PR_fan
        T24 = float(sensors_pred_si["T24_K"].item())
        P24 = P21 * PR_lpc
        T30 = float(sensors_pred_si["T30_K"].item()); P30 = float(sensors_pred_si["P30_Pa"].item())
        T4  = float(diag["T4"].item());               P4  = float(diag["P4"].item())
        T45 = float(diag["T45"].item());              P45 = float(diag["P45"].item())
        T50 = float(sensors_pred_si["T50_K"].item());  P50 = float(diag["P50"].item())

        Ts = [T2, T21, T24, T30, T4, T45, T50]
        Ps = [P2, P21, P24, P30, P4, P45, P50]
        label = (f"fan={row['PR_fan']:.2f}, lpc={row['PR_lpc']:.2f}, "
                 f"hpc={row['PR_hpc']:.0f}  "
                 f"OPR={row['OPR']:.1f}, EPRerr={row['EPR_err_pct']:.1f}%")
        P_curves.append((label, [p / P2 for p in Ps]))
        T_curves.append((label, Ts))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for lbl, vals in P_curves:
        ax.semilogy(STATIONS, vals, "o-", lw=1.6, ms=6, label=lbl)
    ax.axhline(1.0, color="black", lw=0.4)
    ax.set_xlabel("Station"); ax.set_ylabel("P / P2 (log)")
    ax.set_title(f"FC02 — top-5 candidates: total pressure ratio P/P2 vs station{fb_note}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.4, which="both")
    _save(fig, fname_p)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for lbl, vals in T_curves:
        ax.plot(STATIONS, vals, "o-", lw=1.6, ms=6, label=lbl)
    ax.set_xlabel("Station"); ax.set_ylabel("Total temperature [K]")
    ax.set_title(f"FC02 — top-5 candidates: total temperature vs station{fb_note}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.4)
    _save(fig, fname_t)


# =============================================================================
# Markdown summary
# =============================================================================

def _df_md(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "| (none) |\n|---|\n"
    sub = df[cols].copy()
    md = "| " + " | ".join(cols) + " |\n"
    md += "|" + "|".join(["---"] * len(cols)) + "|\n"
    for _, row in sub.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}")
            else:
                cells.append(str(v))
        md += "| " + " | ".join(cells) + " |\n"
    return md


def write_markdown(df: pd.DataFrame, candidates: dict[str, pd.DataFrame],
                   pass_rates: pd.DataFrame):
    feas = feasibility_mask(df)
    df_feas = df[feas]
    n_total = len(df)
    n_feas  = int(feas.sum())

    cols_show = ["PR_fan", "PR_lpc", "PR_hpc", "OPR",
                 "T45_err_pct", "EPR_err_pct",
                 "T4_K", "T30_K", "T50_K",
                 "PR_hpt", "PR_lpt"]

    # Identify the binding constraint(s) — those with the lowest isolated pass rate
    binding = pass_rates.copy()
    binding["pass_int"] = binding["pass"].astype(int)
    binding = binding.sort_values("pass_int")
    binding_lines = "\n".join(
        f"- **{r['criterion']}** — {r['pass']}/{r['total']} ({r['pass_rate']})"
        for _, r in binding.iterrows()
    )

    md = f"""# C0b PR-split feasibility diagnostic — UserGuide FC02

*Read-only sweep diagnostic. No YAML written. No DS02 access. No optimization.*
*Top candidates are reporting-only; no parameter set is automatically adopted.*

## Frozen state (unchanged for this sweep)

- `Wc_fan_design = {PICKED_WC_FAN_DESIGN} kg/s` (provisional thermal anchor from C0)
- `eta_design_hpt = {ASSUMPTION_ETA_DESIGN_HPT}`, `eta_design_lpt = {ASSUMPTION_ETA_DESIGN_LPT}` (in 0.88–0.92 component band)
- `eta_design_fan/lpc/hpc = {ASSUMPTION_ETA_DESIGN_FAN}/{ASSUMPTION_ETA_DESIGN_LPC}/{ASSUMPTION_ETA_DESIGN_HPC}`
- `BPR = {ASSUMPTION_BPR}`

## Sweep grid (3-D, {len(PR_FAN_GRID) * len(PR_LPC_GRID) * len(PR_HPC_GRID)} combinations)

- PR_fan ∈ {PR_FAN_GRID}
- PR_lpc ∈ {PR_LPC_GRID}
- PR_hpc ∈ {PR_HPC_GRID}

## Feasibility filter

A grid point is **feasible** if all of the following hold:

- OPR ∈ [{OPR_LO}, {OPR_HI}]
- T45 rel err < {T45_ERR_MAX_PCT} %
- EPR rel err < {EPR_ERR_MAX_PCT} %
- T4 ∈ [{T4_LO}, {T4_HI}] K
- T30 ∈ [{T30_LO}, {T30_HI}] K
- T50 ∈ [{T50_LO}, {T50_HI}] K
- eta_hpt, eta_lpt ∈ [{ETA_BAND_LO}, {ETA_BAND_HI}] (kept by construction)
- no PR clamp active anywhere

## Sweep result

| Total points | Feasible | Feasibility rate |
|---|---|---|
| {n_total} | **{n_feas}** | {n_feas / n_total * 100:.1f} % |

## Per-criterion isolated pass rates

Pass rate of each criterion **considered alone** across the full {n_total}-point grid (highest-restriction first):

{binding_lines}

The criterion with the lowest isolated pass rate is the **binding constraint** — the assumption set cannot satisfy it within the realistic eta band, regardless of how PR_fan/lpc/hpc are split.

## Top-5 candidates by criterion (REPORTING ONLY — do not auto-adopt)

### 1) Minimum EPR error (within strict feasibility window)

{_df_md(candidates['min_EPR_err'], cols_show)}

### 2) Minimum combined T45/EPR error (within strict feasibility window)

{_df_md(candidates['min_combined_err'], cols_show + ['combined_err'] if not candidates['min_combined_err'].empty else cols_show)}

### 3) Closest CFM56-like OPR window [{CFM56_OPR_LO}, {CFM56_OPR_HI}] (within strict feasibility)

{_df_md(candidates['closest_CFM56_OPR'], cols_show)}

## Near-feasible references (diagnostic — when strict feasibility is empty)

These ignore some of the feasibility filters; useful for understanding where the cycle wants to sit.

### A) Smallest EPR error over the entire grid (no filters)

{_df_md(candidates['unconstrained_min_EPR'], cols_show)}

### B) Smallest EPR error within OPR ∈ [{OPR_LO}, {OPR_HI}] (no other filters)

{_df_md(candidates['min_EPR_in_OPR_window'], cols_show)}

## CSV exports

- `all_candidates.csv` — full {n_total}-row sweep with every metric
- `feasible_candidates.csv` — the {n_feas} feasible rows only

## Plot index

| # | Plot | File |
|---|---|---|
| 9  | EPR-error heatmaps faceted by PR_fan | `09_EPR_err_heatmaps.png` |
| 10 | Feasibility scatter OPR vs EPR error (color = T45 err) | `10_feasible_scatter_OPR_vs_EPR.png` |
| 11 | Station total-pressure profile, top-5 candidates | `11_station_pressure_top5.png` |
| 12 | Station total-temperature profile, top-5 candidates | `12_station_temperature_top5.png` |

---

*Stop. No automatic parameter selection. Awaiting Robert review.*
"""
    p = OUT_DIR / "summary.md"
    p.write_text(md, encoding="utf-8")
    print(f"  saved  {p}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("V3.1b C0b PR-split feasibility diagnostic at FC02")
    print(f"Output dir: {OUT_DIR}")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fc02 = fc02_conditions_si()

    print(f"\nSweeping {len(PR_FAN_GRID) * len(PR_LPC_GRID) * len(PR_HPC_GRID)} "
          f"combinations of (PR_fan, PR_lpc, PR_hpc)...")
    df = sweep_grid(fc02)

    feas = feasibility_mask(df)
    df_feas = df[feas].copy()
    n_total = len(df); n_feas = int(feas.sum())
    print(f"  total = {n_total},  feasible = {n_feas} "
          f"({n_feas / n_total * 100:.1f} %)")

    # CSVs
    df.to_csv(OUT_DIR / "all_candidates.csv", index=False)
    df_feas.to_csv(OUT_DIR / "feasible_candidates.csv", index=False)
    print(f"  saved  {OUT_DIR / 'all_candidates.csv'}")
    print(f"  saved  {OUT_DIR / 'feasible_candidates.csv'}")

    candidates = best_candidates(df, df_feas)
    # combined_err is already attached for the feasible set; ensure it's also there
    # for the unconstrained_min_EPR view if useful (skipped — keep narrow columns)

    pass_rates = per_criterion_pass_rates(df)

    print("\nPer-criterion isolated pass rates:")
    for _, r in pass_rates.iterrows():
        print(f"  {r['criterion']:35s}  {r['pass']:3d} / {r['total']}  ({r['pass_rate']})")

    print("\nGenerating plots...")
    plot_epr_heatmaps(df)
    plot_feasibility_scatter(df)
    plot_top5_stations(df_feas, df, fc02,
                       "11_station_pressure_top5.png",
                       "12_station_temperature_top5.png")

    print("\nWriting summary.md...")
    write_markdown(df, candidates, pass_rates)

    print("\n" + "=" * 72)
    print(f"Done.  Feasible: {n_feas} / {n_total}.  No YAML written.")
    print(f"Awaiting Robert review.")
    print("=" * 72)


if __name__ == "__main__":
    main()
