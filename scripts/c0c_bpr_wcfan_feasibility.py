"""C0c BPR/Wc_fan feasibility diagnostic at FC02 (V3.1b, read-only).

Purpose
-------
After C0/C0b showed that PR-split alone cannot resolve the EPR mismatch
under CFM56-like assumptions (BPR=5.5, Wc=900, OPR≈38), this diagnostic
sweeps the engine-class assumptions: BPR, Wc_fan_design, target OPR (via
OPR-constrained PR split), and the realistic eta_design_hpt/lpt grid.
The sweep tests whether C-MAPSS-90k-class assumptions (BPR≈8.4, OPR≈36)
close the EPR gap.

Hard constraints (V3.1b Hard Rule 17)
-------------------------------------
- read-only: NO writes to configs/cyclelayer_v3.yaml
- NO DS02 access
- NO `fit_*` helpers, NO optimizer, NO automatic parameter selection
- eta_design_hpt/lpt stay in the realistic 0.88–0.92 component band
- Each grid point is a single forward pass; no iteration, no minimizer
- Top candidates are reporting-only

Outputs
-------
artifacts/cyclelayer_v3/c0c_bpr_wcfan/
  ├─ 13_heatmap_BPR_Wcfan_EPRerr.png
  ├─ 14_heatmap_BPR_Wcfan_T50.png
  ├─ 15_pareto_T45err_vs_EPRerr.png
  ├─ 16_LP_spool_decomposition_vs_BPR.png
  ├─ 17_station_pressure_top5.png
  ├─ 18_station_temperature_top5.png
  ├─ 19_top20_candidates_table.png
  ├─ all_candidates.csv
  ├─ plausible_candidates.csv
  ├─ top20_candidates.csv
  └─ c0c_bpr_wcfan_report.md
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
    ETA_DESIGN_FAN, ETA_DESIGN_HPC, ETA_DESIGN_LPC,
    EXP_C, GAMMA_C,
    P_REF, T_REF,
)


# =============================================================================
# Sweep grids (V3.1b C0c spec)
# =============================================================================

BPR_GRID         = [5.5, 6.5, 7.5, 8.4, 9.0]
WCFAN_GRID       = [900.0, 1050.0, 1200.0, 1300.0, 1400.0, 1500.0]
TARGET_OPR_GRID  = [30.0, 33.0, 36.0]
PR_FAN_GRID      = [1.55, 1.60, 1.65, 1.70]
PR_LPC_GRID      = [1.40, 1.50, 1.60, 1.70, 1.80]
ETA_HPT_GRID     = [0.88, 0.90, 0.92]
ETA_LPT_GRID     = [0.88, 0.90, 0.92]

# Cold-side compressor design etas stay at literature defaults (frozen)
ETA_DESIGN_FAN_C0c = ETA_DESIGN_FAN     # 0.92
ETA_DESIGN_LPC_C0c = ETA_DESIGN_LPC     # 0.90
ETA_DESIGN_HPC_C0c = ETA_DESIGN_HPC     # 0.88


# =============================================================================
# Rejection / plausibility windows
# =============================================================================

PR_HPC_LO, PR_HPC_HI = 6.0, 14.0           # structural (pre-forward reject)
T4_LO,  T4_HI   = 1300.0, 1900.0           # plausibility
T30_LO, T30_HI  = 600.0, 1000.0            # plausibility
T50_LO, T50_HI  = 650.0, 1100.0            # plausibility
FAR_LO, FAR_HI  = 0.015, 0.035             # plausibility

# Acceptance bands
T45_ERR_PREFERRED = 15.0   # %
T45_ERR_ALLOWED   = 25.0   # %
EPR_ERR_ALLOWED   = 25.0   # %

# Region of interest for "C-MAPSS class"
CMAPSS_BPR        = 8.4
CMAPSS_BPR_BAND   = (7.5, 9.0)
CMAPSS_OPR_BAND   = (30.0, 36.0)
CMAPSS_WCFAN_BAND = (1200.0, 1500.0)

# Old CFM56-like baseline reference (not in any sweep grid; computed once)
OLD_BASELINE = dict(
    bpr=5.5, Wc_fan_design=900.0, target_OPR=38.0,
    pr_fan=1.60, pr_lpc=2.00, pr_hpc=12.0,        # gives ~38.4 OPR
    eta_hpt=0.90, eta_lpt=0.92,
)


# =============================================================================
# FC02 conditions and engine builder
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


def derive_design_points(fc02: dict, Wc_fan: float,
                         pr_fan: float, pr_lpc: float, pr_hpc: float,
                         bpr: float) -> dict:
    T2_K = fc02["T2_R"] * units.RANK_TO_K
    P2_Pa = fc02["P2_psia"] * units.PSIA_TO_PA
    Nc_design_fan = fc02["Nf_rpm"] / math.sqrt(T2_K / T_REF)
    m_in_design = Wc_fan * (P2_Pa / P_REF) / math.sqrt(T2_K / T_REF)
    m_core_design = m_in_design / (bpr + 1.0)

    T21_isen = T2_K * pr_fan ** EXP_C
    T21      = T2_K + (T21_isen - T2_K) / ETA_DESIGN_FAN_C0c
    P21      = P2_Pa * pr_fan
    Wc_design_lpc = m_core_design * math.sqrt(T21 / T_REF) / (P21 / P_REF)
    Nc_design_lpc = Nc_design_fan

    T24_isen = T21 * pr_lpc ** EXP_C
    T24      = T21 + (T24_isen - T21) / ETA_DESIGN_LPC_C0c
    P24      = P21 * pr_lpc
    Wc_design_hpc = m_core_design * math.sqrt(T24 / T_REF) / (P24 / P_REF)
    Nc_design_hpc = fc02["Nc_rpm"] / math.sqrt(T24 / T_REF)

    return {
        "Nc_design_fan": Nc_design_fan, "Nc_design_lpc": Nc_design_lpc,
        "Nc_design_hpc": Nc_design_hpc,
        "Wc_design_fan": Wc_fan,
        "Wc_design_lpc": Wc_design_lpc, "Wc_design_hpc": Wc_design_hpc,
        "PR_design_fan": pr_fan,
        "PR_design_lpc": pr_lpc,
        "PR_design_hpc": pr_hpc,
        "eta_design_fan": ETA_DESIGN_FAN_C0c,
        "eta_design_lpc": ETA_DESIGN_LPC_C0c,
        "eta_design_hpc": ETA_DESIGN_HPC_C0c,
    }


def build_engine(design: dict, bpr: float, eta_hpt: float, eta_lpt: float) -> BraytonEngine:
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
        bpr_design=bpr,
        eta_design_hpt=eta_hpt,
        eta_design_lpt=eta_lpt,
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


def evaluate_point(fc02: dict, bpr: float, Wc_fan: float, target_OPR: float,
                   pr_fan: float, pr_lpc: float, pr_hpc: float,
                   eta_hpt: float, eta_lpt: float) -> dict:
    """Single forward pass + metric extraction; returns one row dict."""
    design = derive_design_points(fc02, Wc_fan, pr_fan, pr_lpc, pr_hpc, bpr)
    engine = build_engine(design, bpr, eta_hpt, eta_lpt)
    sensors_pred_si, diag, ops_si, sens_si = run_fc02_forward(engine, fc02)

    T2 = float(ops_si["T2_K"].item())
    P2 = float(ops_si["P2_Pa"].item())
    T24 = float(sensors_pred_si["T24_K"].item())
    T30 = float(sensors_pred_si["T30_K"].item())
    P30 = float(sensors_pred_si["P30_Pa"].item())
    T50 = float(sensors_pred_si["T50_K"].item())
    T4  = float(diag["T4"].item());  T45 = float(diag["T45"].item())
    P45 = float(diag["P45"].item());  P50 = float(diag["P50"].item())
    OPR = float(diag["P30_over_P2"].item())
    EPR = P50 / P2
    T45_R = T45 / units.RANK_TO_K

    T45_err = abs(T45_R - fc02["T48_ref_R"]) / fc02["T48_ref_R"] * 100.0
    EPR_err = abs(EPR - fc02["EPR_ref"]) / fc02["EPR_ref"] * 100.0

    P21 = P2 * pr_fan; P24 = P21 * pr_lpc

    m_in   = float(diag["m_in"].item())
    m_core = float(diag["m_core"].item())
    m_byp  = float(diag["m_byp"].item())
    Wf_kgs = float(sens_si["Wf_kgs"].item())
    FAR = Wf_kgs / m_core

    W_hpc = float(diag["W_hpc"].item());  W_hpt = float(diag["W_hpt"].item())
    W_lpc = float(diag["W_lpc"].item());  W_fan = float(diag["W_fan_total"].item())
    W_lpt = float(diag["W_lpt"].item())

    PR_hpt = float(diag["PR_hpt"].item());  PR_lpt = float(diag["PR_lpt"].item())

    clamp_frac = {f"frac_{k}": float(diag[f"frac_PR_{k}_clamped"].item())
                  for k in ("fan", "lpc", "hpc", "hpt", "lpt")}
    clamps_active = any(v > 0.0 for v in clamp_frac.values())

    return {
        "BPR": bpr, "Wc_fan_design": Wc_fan, "target_OPR": target_OPR,
        "PR_fan": pr_fan, "PR_lpc": pr_lpc, "PR_hpc": pr_hpc,
        "eta_hpt": eta_hpt, "eta_lpt": eta_lpt,
        "OPR": OPR, "EPR": EPR,
        "T45_R": T45_R, "T45_err_pct": T45_err, "EPR_err_pct": EPR_err,
        "T24_K": T24, "T30_K": T30, "T4_K": T4, "T50_K": T50,
        "P24_over_P2": P24 / P2, "P30_over_P2": P30 / P2,
        "PR_hpt": PR_hpt, "PR_lpt": PR_lpt,
        "m_in": m_in, "m_core": m_core, "m_bypass": m_byp,
        "Wf_kgs": Wf_kgs, "FAR": FAR,
        "W_fan_total_MW": W_fan / 1e6, "W_lpc_MW": W_lpc / 1e6,
        "W_hpc_MW": W_hpc / 1e6, "W_hpt_MW": W_hpt / 1e6, "W_lpt_MW": W_lpt / 1e6,
        "shaft_HPT_resid_W": float(diag["shaft_HPT_residual"].item()),
        "shaft_LPT_resid_W": float(diag["shaft_LPT_residual"].item()),
        **clamp_frac,
        "clamps_active": clamps_active,
    }


# =============================================================================
# Sweep
# =============================================================================

def sweep_grid(fc02: dict) -> tuple[pd.DataFrame, int, int]:
    """Run the full diagnostic sweep. Returns (df, n_attempted, n_pre_rejected)."""
    rows: list[dict] = []
    n_attempted = 0
    n_pre_reject = 0
    for bpr, Wc_fan, opr_t, pr_fan, pr_lpc, eta_hpt, eta_lpt in product(
        BPR_GRID, WCFAN_GRID, TARGET_OPR_GRID,
        PR_FAN_GRID, PR_LPC_GRID, ETA_HPT_GRID, ETA_LPT_GRID,
    ):
        n_attempted += 1
        pr_hpc = opr_t / (pr_fan * pr_lpc)
        if pr_hpc < PR_HPC_LO or pr_hpc > PR_HPC_HI:
            n_pre_reject += 1
            continue
        rows.append(evaluate_point(
            fc02, bpr, Wc_fan, opr_t, pr_fan, pr_lpc, pr_hpc,
            eta_hpt, eta_lpt,
        ))
    return pd.DataFrame(rows), n_attempted, n_pre_reject


# =============================================================================
# Plausibility / candidate selection (REPORTING ONLY)
# =============================================================================

def plausibility_mask(df: pd.DataFrame) -> pd.Series:
    """All structural + plausibility filters combined."""
    return (
        (df["T4_K"]        >= T4_LO)              & (df["T4_K"]        <= T4_HI) &
        (df["T30_K"]       >= T30_LO)             & (df["T30_K"]       <= T30_HI) &
        (df["T50_K"]       >= T50_LO)             & (df["T50_K"]       <= T50_HI) &
        (df["FAR"]         >= FAR_LO)             & (df["FAR"]         <= FAR_HI) &
        (df["T45_err_pct"] <  T45_ERR_ALLOWED)    &
        (df["EPR_err_pct"] <  EPR_ERR_ALLOWED)    &
        (df["eta_hpt"]     >= 0.88)               & (df["eta_hpt"]     <= 0.92) &
        (df["eta_lpt"]     >= 0.88)               & (df["eta_lpt"]     <= 0.92) &
        (~df["clamps_active"])
    )


def near_cmapss_mask(df: pd.DataFrame) -> pd.Series:
    """C-MAPSS-class region: BPR≈8.4 and OPR≈36."""
    return (
        (df["BPR"]        >= CMAPSS_BPR_BAND[0])   & (df["BPR"]        <= CMAPSS_BPR_BAND[1]) &
        (df["OPR"]        >= CMAPSS_OPR_BAND[0])   & (df["OPR"]        <= CMAPSS_OPR_BAND[1])
    )


def per_criterion_pass_rates(df: pd.DataFrame) -> pd.DataFrame:
    masks = {
        f"T4 in [{T4_LO}, {T4_HI}] K":      (df["T4_K"]  >= T4_LO)  & (df["T4_K"]  <= T4_HI),
        f"T30 in [{T30_LO}, {T30_HI}] K":   (df["T30_K"] >= T30_LO) & (df["T30_K"] <= T30_HI),
        f"T50 in [{T50_LO}, {T50_HI}] K":   (df["T50_K"] >= T50_LO) & (df["T50_K"] <= T50_HI),
        f"FAR in [{FAR_LO}, {FAR_HI}]":     (df["FAR"]   >= FAR_LO) & (df["FAR"]   <= FAR_HI),
        f"T45 err < {T45_ERR_ALLOWED}%":     df["T45_err_pct"] < T45_ERR_ALLOWED,
        f"EPR err < {EPR_ERR_ALLOWED}%":     df["EPR_err_pct"] < EPR_ERR_ALLOWED,
        "eta_hpt/lpt in [0.88, 0.92]":      (df["eta_hpt"] >= 0.88) & (df["eta_hpt"] <= 0.92) &
                                            (df["eta_lpt"] >= 0.88) & (df["eta_lpt"] <= 0.92),
        "no PR clamp active":               ~df["clamps_active"],
    }
    n = len(df)
    return pd.DataFrame([
        {"criterion": name, "pass": int(m.sum()), "total": n,
         "pass_rate": f"{m.sum() / n * 100:.1f}%"}
        for name, m in masks.items()
    ]).sort_values("pass")


def best_candidates(df_plausible: pd.DataFrame, df_all: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    # 1) Best EPR error in plausible set
    out["min_EPR_plausible"] = df_plausible.sort_values("EPR_err_pct").head(10)

    # 2) Best combined T45/EPR error in plausible set
    if not df_plausible.empty:
        d = df_plausible.copy()
        d["combined_err"] = np.sqrt(d["T45_err_pct"] ** 2 + d["EPR_err_pct"] ** 2)
        out["min_combined_plausible"] = d.sort_values("combined_err").head(10)
    else:
        out["min_combined_plausible"] = pd.DataFrame()

    # 3) Best near C-MAPSS region (BPR≈8.4 AND OPR≈30–36)
    near = df_all[near_cmapss_mask(df_all)]
    if not near.empty:
        # plausible-and-near subset preferred
        plaus_near = near[plausibility_mask(near)]
        if not plaus_near.empty:
            out["near_cmapss_plausible"] = plaus_near.sort_values("EPR_err_pct").head(10)
        else:
            out["near_cmapss_plausible"] = near.sort_values("EPR_err_pct").head(10)
    else:
        out["near_cmapss_plausible"] = pd.DataFrame()

    # 4) Top 20 sorted by EPR error (over the whole sweep)
    out["top20_by_EPR_err"] = df_all.sort_values("EPR_err_pct").head(20)

    return out


# =============================================================================
# Plots
# =============================================================================

OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "c0c_bpr_wcfan"
STATIONS = ["2", "21", "24", "30", "4", "45", "50"]


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {p}")


def _bpr_wcfan_aggregate(df: pd.DataFrame, value_col: str, agg: str = "min"):
    """Pivot to BPR (rows) x Wc_fan_design (cols) using per-cell aggregate."""
    if agg == "min":
        piv = df.pivot_table(index="BPR", columns="Wc_fan_design",
                             values=value_col, aggfunc="min")
    elif agg == "median":
        piv = df.pivot_table(index="BPR", columns="Wc_fan_design",
                             values=value_col, aggfunc="median")
    elif agg == "best_EPR":
        # value at row with min EPR_err per cell
        idx = df.groupby(["BPR", "Wc_fan_design"])["EPR_err_pct"].idxmin()
        rep = df.loc[idx]
        piv = rep.pivot_table(index="BPR", columns="Wc_fan_design",
                              values=value_col, aggfunc="first")
    else:
        raise ValueError(agg)
    return piv


def plot_heatmap_eprerr(df: pd.DataFrame):
    eprerr = _bpr_wcfan_aggregate(df, "EPR_err_pct", "min")
    t45err = _bpr_wcfan_aggregate(df, "T45_err_pct", "best_EPR")

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(eprerr.values, origin="lower", aspect="auto",
                   cmap="viridis_r", vmin=0, vmax=80)
    ax.set_xticks(range(len(eprerr.columns)))
    ax.set_xticklabels([f"{int(c)}" for c in eprerr.columns])
    ax.set_yticks(range(len(eprerr.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in eprerr.index])
    ax.set_xlabel("Wc_fan_design  [kg/s]")
    ax.set_ylabel("BPR")
    ax.set_title("FC02 — min EPR rel error [%]   (heatmap = EPR err, contour = T45 err [%] at min-EPR row)")
    cbar = fig.colorbar(im, ax=ax, label="min EPR rel err [%]")

    # Annotate each cell with the EPR err value
    for i in range(eprerr.shape[0]):
        for j in range(eprerr.shape[1]):
            v = eprerr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > 35 else "black", fontsize=7)

    # Contour the T45 err on top
    if not t45err.isna().all().all():
        try:
            X, Y = np.meshgrid(np.arange(t45err.shape[1]), np.arange(t45err.shape[0]))
            cs = ax.contour(X, Y, t45err.values, levels=[5, 10, 15, 25],
                            colors="white", linewidths=1.0, alpha=0.9)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%d%%")
        except Exception:
            pass
    _save(fig, "13_heatmap_BPR_Wcfan_EPRerr.png")


def plot_heatmap_T50(df: pd.DataFrame):
    piv = _bpr_wcfan_aggregate(df, "T50_K", "best_EPR")
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(piv.values, origin="lower", aspect="auto",
                   cmap="coolwarm", vmin=650, vmax=1100)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{int(c)}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in piv.index])
    ax.set_xlabel("Wc_fan_design  [kg/s]")
    ax.set_ylabel("BPR")
    ax.set_title("FC02 — T50 [K] at min-EPR-err point  (plausibility window 650–1100 K)")
    fig.colorbar(im, ax=ax, label="T50 [K]")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="black", fontsize=7)
    _save(fig, "14_heatmap_BPR_Wcfan_T50.png")


def plot_pareto(df: pd.DataFrame, baseline_row: dict):
    fig, ax = plt.subplots(figsize=(9, 6))
    feas = plausibility_mask(df)
    df_p = df[feas]
    sc1 = ax.scatter(df.loc[~feas, "T45_err_pct"], df.loc[~feas, "EPR_err_pct"],
                     s=8, c="lightgray", alpha=0.4, label="non-plausible")
    if not df_p.empty:
        sizes = (df_p["Wc_fan_design"] - WCFAN_GRID[0]) / (WCFAN_GRID[-1] - WCFAN_GRID[0]) * 60 + 20
        sc2 = ax.scatter(df_p["T45_err_pct"], df_p["EPR_err_pct"],
                         s=sizes, c=df_p["BPR"], cmap="viridis",
                         vmin=BPR_GRID[0], vmax=BPR_GRID[-1],
                         edgecolors="black", linewidths=0.4, alpha=0.85,
                         label="plausible")
        cbar = fig.colorbar(sc2, ax=ax)
        cbar.set_label("BPR")
    ax.scatter([baseline_row["T45_err_pct"]], [baseline_row["EPR_err_pct"]],
               s=180, marker="X", c="red", edgecolors="black", linewidths=1.4,
               label=f"OLD baseline BPR=5.5, Wc=900, OPR≈38", zorder=10)
    ax.axhline(EPR_ERR_ALLOWED, color="black", lw=0.7, ls="--",
               label=f"{EPR_ERR_ALLOWED}% EPR band")
    ax.axvline(T45_ERR_ALLOWED, color="black", lw=0.7, ls=":",
               label=f"{T45_ERR_ALLOWED}% T45 band")
    ax.set_xlabel("T45 rel err [%]")
    ax.set_ylabel("EPR rel err [%]")
    ax.set_yscale("log")
    ax.set_title("FC02 — Pareto: T45 err vs EPR err  "
                 "(color = BPR, size ~ Wc_fan_design, X = old baseline)")
    ax.grid(True, alpha=0.4, which="both")
    ax.legend(fontsize=7, loc="upper right")
    _save(fig, "15_pareto_T45err_vs_EPRerr.png")


def plot_LP_decomposition(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    # For each BPR pick the representative point: min EPR err overall in that BPR
    # (fall back if needed; this is just a representative trend, not selection)
    rep_rows = []
    for bpr in BPR_GRID:
        sub = df[df["BPR"] == bpr]
        if sub.empty:
            continue
        rep_rows.append(sub.loc[sub["EPR_err_pct"].idxmin()])
    rep = pd.DataFrame(rep_rows)
    if rep.empty:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        _save(fig, "16_LP_spool_decomposition_vs_BPR.png")
        return
    x = rep["BPR"]
    panels = [
        ("W_fan_total_MW", "W_fan_total [MW]"),
        ("W_lpc_MW",       "W_lpc [MW]"),
        ("W_lpt_MW",       "W_lpt (required) [MW]"),
        ("PR_lpt",         "PR_lpt"),
        ("T50_K",          "T50 [K]"),
    ]
    for ax, (col, lbl) in zip(axes.flat, panels):
        ax.plot(x, rep[col], "o-", color="tab:blue", lw=1.6, ms=7)
        ax.set_xlabel("BPR"); ax.set_ylabel(lbl)
        ax.set_title(lbl + " vs BPR  (representative: min-EPR-err per BPR)")
        ax.grid(True, alpha=0.4)
    axes.flat[5].axis("off")
    fig.suptitle("LP-spool decomposition vs BPR  (FC02, representative per BPR)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "16_LP_spool_decomposition_vs_BPR.png")


def _top5_curves(df_p: pd.DataFrame, df_all: pd.DataFrame, fc02: dict, fname_p: str, fname_t: str):
    fb_note = ""
    use = df_p
    if use.empty:
        use = df_all
        fb_note = "  [NEAR-FEASIBLE — plausible set is empty]"
    use = use.copy()
    use["combined_err"] = np.sqrt(use["T45_err_pct"] ** 2 + use["EPR_err_pct"] ** 2)
    top = use.sort_values("combined_err").head(5)

    P_curves: list[tuple[str, list[float]]] = []
    T_curves: list[tuple[str, list[float]]] = []
    for _, row in top.iterrows():
        design = derive_design_points(fc02, row["Wc_fan_design"],
                                      row["PR_fan"], row["PR_lpc"], row["PR_hpc"],
                                      row["BPR"])
        eng = build_engine(design, row["BPR"], row["eta_hpt"], row["eta_lpt"])
        sensors_pred_si, diag, ops_si, _ = run_fc02_forward(eng, fc02)
        T2 = float(ops_si["T2_K"].item());   P2 = float(ops_si["P2_Pa"].item())
        PR_fan = float(diag["PR_fan"].item()); PR_lpc = float(diag["PR_lpc"].item())
        eta_fan_eff = float(diag["eta_fan"].item())
        T21_isen = T2 * PR_fan ** EXP_C
        T21      = T2 + (T21_isen - T2) / eta_fan_eff
        P21      = P2 * PR_fan
        T24 = float(sensors_pred_si["T24_K"].item()); P24 = P21 * PR_lpc
        T30 = float(sensors_pred_si["T30_K"].item()); P30 = float(sensors_pred_si["P30_Pa"].item())
        T4  = float(diag["T4"].item());               P4  = float(diag["P4"].item())
        T45 = float(diag["T45"].item());              P45 = float(diag["P45"].item())
        T50 = float(sensors_pred_si["T50_K"].item());  P50 = float(diag["P50"].item())
        Ts = [T2, T21, T24, T30, T4, T45, T50]
        Ps = [P2, P21, P24, P30, P4, P45, P50]
        label = (f"BPR={row['BPR']:.1f}, Wc={int(row['Wc_fan_design'])}, "
                 f"OPR={row['OPR']:.1f},  "
                 f"EPRerr={row['EPR_err_pct']:.1f}%,  "
                 f"T45err={row['T45_err_pct']:.1f}%")
        P_curves.append((label, [p / P2 for p in Ps]))
        T_curves.append((label, Ts))

    fig, ax = plt.subplots(figsize=(11, 6))
    for lbl, vals in P_curves:
        ax.semilogy(STATIONS, vals, "o-", lw=1.6, ms=6, label=lbl)
    ax.axhline(1.0, color="black", lw=0.4)
    ax.set_xlabel("Station"); ax.set_ylabel("P / P2 (log)")
    ax.set_title(f"FC02 — top-5 candidates: total pressure P/P2 vs station{fb_note}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.4, which="both")
    _save(fig, fname_p)

    fig, ax = plt.subplots(figsize=(11, 6))
    for lbl, vals in T_curves:
        ax.plot(STATIONS, vals, "o-", lw=1.6, ms=6, label=lbl)
    ax.set_xlabel("Station"); ax.set_ylabel("Total temperature [K]")
    ax.set_title(f"FC02 — top-5 candidates: total temperature vs station{fb_note}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.4)
    _save(fig, fname_t)


def plot_top20_table(df_top20: pd.DataFrame, df_all: pd.DataFrame):
    if df_top20.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "no candidates", ha="center", va="center", fontsize=14)
        _save(fig, "19_top20_candidates_table.png")
        return
    plaus = plausibility_mask(df_all)
    cols = ["BPR", "Wc_fan_design", "OPR",
            "PR_fan", "PR_lpc", "PR_hpc",
            "eta_hpt", "eta_lpt",
            "T45_err_pct", "EPR_err_pct",
            "T4_K", "T30_K", "T50_K", "FAR"]
    df_show = df_top20[cols].copy()
    plaus_flag = plaus.reindex(df_top20.index, fill_value=False).map(
        lambda v: "✓" if v else "✗"
    )
    df_show.insert(0, "plausible", plaus_flag.values)
    # Format
    fmts = {
        "BPR": "{:.1f}", "Wc_fan_design": "{:.0f}", "OPR": "{:.1f}",
        "PR_fan": "{:.2f}", "PR_lpc": "{:.2f}", "PR_hpc": "{:.2f}",
        "eta_hpt": "{:.2f}", "eta_lpt": "{:.2f}",
        "T45_err_pct": "{:.1f}", "EPR_err_pct": "{:.1f}",
        "T4_K": "{:.0f}", "T30_K": "{:.0f}", "T50_K": "{:.0f}",
        "FAR": "{:.4f}",
    }
    for c, f in fmts.items():
        df_show[c] = df_show[c].map(lambda v: f.format(v))

    fig, ax = plt.subplots(figsize=(15, 0.36 * (len(df_show) + 1) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False); table.set_fontsize(8)
    table.scale(1.0, 1.4)
    ax.set_title("Top 20 candidates by EPR rel error (REPORTING ONLY — not for adoption)")
    _save(fig, "19_top20_candidates_table.png")


# =============================================================================
# Markdown report
# =============================================================================

def _df_md(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "| (none) |\n|---|\n"
    md = "| " + " | ".join(cols) + " |\n"
    md += "|" + "|".join(["---"] * len(cols)) + "|\n"
    for _, row in df[cols].iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, float):
                if c in ("BPR", "PR_fan", "PR_lpc"):
                    cells.append(f"{v:.2f}")
                elif c in ("PR_hpc", "OPR"):
                    cells.append(f"{v:.2f}")
                elif c == "FAR":
                    cells.append(f"{v:.4f}")
                elif c.endswith("_K"):
                    cells.append(f"{v:.0f}")
                elif "err" in c.lower():
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        md += "| " + " | ".join(cells) + " |\n"
    return md


def write_report(df_all: pd.DataFrame, df_plaus: pd.DataFrame,
                 candidates: dict[str, pd.DataFrame], pass_rates: pd.DataFrame,
                 baseline_row: dict, n_attempted: int, n_pre_reject: int):
    n_total = len(df_all)
    n_plaus = len(df_plaus)
    n_in_cmapss = int(near_cmapss_mask(df_all).sum())
    n_plaus_in_cmapss = int((near_cmapss_mask(df_all) & plausibility_mask(df_all)).sum())

    cols_show = ["BPR", "Wc_fan_design", "OPR",
                 "PR_fan", "PR_lpc", "PR_hpc",
                 "eta_hpt", "eta_lpt",
                 "T45_err_pct", "EPR_err_pct",
                 "T4_K", "T30_K", "T50_K", "FAR"]
    cols_show_top20 = ["BPR", "Wc_fan_design", "OPR",
                       "PR_fan", "PR_lpc", "PR_hpc",
                       "T45_err_pct", "EPR_err_pct",
                       "T4_K", "T50_K", "FAR"]

    bind_lines = "\n".join(
        f"- **{r['criterion']}** — {r['pass']}/{r['total']} ({r['pass_rate']})"
        for _, r in pass_rates.iterrows()
    )

    # ── Narrative answers to the three review questions ──
    # Q1: Does BPR≈8.4 reduce EPR vs BPR=5.5?
    rep_per_bpr = (
        df_all.groupby("BPR")["EPR_err_pct"].min().to_dict()
    )
    eprerr_at_55 = rep_per_bpr.get(5.5, float("nan"))
    eprerr_at_84 = rep_per_bpr.get(8.4, float("nan"))
    eprerr_at_90 = rep_per_bpr.get(9.0, float("nan"))
    q1_answer = (
        f"Across the swept grid, min EPR rel err per BPR:  "
        f"BPR=5.5 → **{eprerr_at_55:.1f} %**, BPR=8.4 → **{eprerr_at_84:.1f} %**, "
        f"BPR=9.0 → **{eprerr_at_90:.1f} %**.  "
        + ("Yes — moving from BPR=5.5 to BPR=8.4 reduces the minimum EPR rel err."
           if eprerr_at_84 < eprerr_at_55
           else "No — moving from BPR=5.5 to BPR=8.4 does not reduce the minimum EPR rel err.")
    )

    # Q2: FAR/T45 plausible at Wc=1200-1500?
    sub_q2 = df_all[(df_all["Wc_fan_design"] >= 1200) & (df_all["Wc_fan_design"] <= 1500)]
    far_in_band = ((sub_q2["FAR"] >= FAR_LO) & (sub_q2["FAR"] <= FAR_HI)).sum()
    t45_under15 = (sub_q2["T45_err_pct"] < T45_ERR_PREFERRED).sum()
    q2_answer = (
        f"Among the {len(sub_q2)} points with Wc_fan_design ∈ [1200, 1500] kg/s: "
        f"**{far_in_band} ({far_in_band / max(len(sub_q2), 1) * 100:.0f} %)** have plausible FAR, "
        f"**{t45_under15} ({t45_under15 / max(len(sub_q2), 1) * 100:.0f} %)** have T45 err < {T45_ERR_PREFERRED} %."
    )

    # Q3: Any candidate fully plausible?
    if n_plaus > 0:
        q3_answer = (
            f"**Yes** — {n_plaus} candidate(s) satisfy all plausibility criteria "
            f"(BPR/OPR/T/FAR/eta/no-clamp + T45 err < {T45_ERR_ALLOWED} % + EPR err < {EPR_ERR_ALLOWED} %). "
            "Listed below for review; **no parameter set is automatically adopted**."
        )
        recommendation = (
            "Top candidates are listed but NOT adopted. Robert reviews and chooses; "
            "YAML is updated only after explicit approval."
        )
    else:
        q3_answer = (
            "**No** — even with C-MAPSS-class assumptions (BPR up to 9.0, Wc_fan_design up to 1500 kg/s, "
            "OPR ∈ {30, 33, 36}, eta ∈ {0.88, 0.90, 0.92}), no grid point satisfies the full plausibility "
            "set including EPR err < 25 % simultaneously with all temperature/FAR windows."
        )
        recommendation = (
            "**Recommendation:** demote EPR from a hard acceptance gate to a diagnostic until V4. "
            "Reasoning: in V3.1b the LPT pressure ratio is purely closure-determined from measured "
            "Nf/Nc/Wf and the assumed eta and shaft loads; the actual LPT in CMAPSS may use a "
            "flow-matched solver (V4 roadmap) that V3.1b cannot replicate. Phase D / GasTurb cross-check "
            "stays as a separate independent plausibilization. T45 anchor remains a hard gate."
        )

    md = f"""# C0c BPR / Wc_fan feasibility diagnostic — UserGuide FC02

*Read-only sweep diagnostic. No YAML written. No DS02 access. No optimizer.*
*No parameter set is automatically adopted. Top candidates are reporting-only.*

## Purpose

After C0/C0b showed that the PR-split alone cannot resolve the EPR mismatch
under CFM56-like assumptions (BPR=5.5, Wc_fan=900, OPR≈38), this diagnostic
tests whether moving to C-MAPSS-90k-class assumptions (BPR≈8.4, Wc_fan≈1200–1500,
OPR∈{{30, 33, 36}}) closes the gap.

## Frozen state

- eta_design_fan / lpc / hpc = {ETA_DESIGN_FAN_C0c} / {ETA_DESIGN_LPC_C0c} / {ETA_DESIGN_HPC_C0c}  (literature defaults, fixed)
- eta_design_hpt, eta_design_lpt **swept** within the realistic 0.88–0.92 component band
- BraytonEngine `use_measured_inlet=True` (P1)

## Sweep grid

- BPR ∈ {BPR_GRID}
- Wc_fan_design ∈ {WCFAN_GRID} kg/s
- target_OPR ∈ {TARGET_OPR_GRID}
- PR_fan ∈ {PR_FAN_GRID}
- PR_lpc ∈ {PR_LPC_GRID}
- eta_design_hpt ∈ {ETA_HPT_GRID}
- eta_design_lpt ∈ {ETA_LPT_GRID}

PR_hpc is computed as `target_OPR / (PR_fan × PR_lpc)`, then rejected pre-forward
if outside [{PR_HPC_LO}, {PR_HPC_HI}].

## Sweep result

| Attempted | Pre-rejected (PR_hpc bound) | Forward-pass total | Plausible | Plausible & near-CMAPSS (BPR ∈ [{CMAPSS_BPR_BAND[0]}, {CMAPSS_BPR_BAND[1]}], OPR ∈ [{CMAPSS_OPR_BAND[0]}, {CMAPSS_OPR_BAND[1]}]) |
|---|---|---|---|---|
| {n_attempted} | {n_pre_reject} | **{n_total}** | **{n_plaus}** | **{n_plaus_in_cmapss}** |

## Per-criterion isolated pass rates

{bind_lines}

## Q1 — Does BPR≈8.4 reduce EPR vs BPR=5.5?

{q1_answer}

## Q2 — Wc_fan_design ∈ [1200, 1500] kg/s plausibility

{q2_answer}

## Q3 — Any fully plausible candidate?

{q3_answer}

## Old baseline reference (BPR=5.5, Wc_fan=900, OPR≈38)

| metric | value |
|---|---|
| BPR | {baseline_row['BPR']:.1f} |
| Wc_fan_design [kg/s] | {baseline_row['Wc_fan_design']:.0f} |
| OPR | {baseline_row['OPR']:.2f} |
| T45 err [%] | {baseline_row['T45_err_pct']:.2f} |
| EPR err [%] | {baseline_row['EPR_err_pct']:.2f} |
| T4 [K] | {baseline_row['T4_K']:.0f} |
| T30 [K] | {baseline_row['T30_K']:.0f} |
| T50 [K] | {baseline_row['T50_K']:.0f} |
| FAR | {baseline_row['FAR']:.4f} |

## Top candidates — REPORTING ONLY (do not auto-adopt)

### A) Best EPR error in plausible set

{_df_md(candidates['min_EPR_plausible'], cols_show)}

### B) Best combined T45/EPR error in plausible set

{_df_md(candidates['min_combined_plausible'], cols_show)}

### C) Best near C-MAPSS region (BPR ∈ [{CMAPSS_BPR_BAND[0]}, {CMAPSS_BPR_BAND[1]}] AND OPR ∈ [{CMAPSS_OPR_BAND[0]}, {CMAPSS_OPR_BAND[1]}])

{_df_md(candidates['near_cmapss_plausible'], cols_show)}

### D) Top 20 by EPR error (with plausibility flag)

See `top20_candidates.csv` and `19_top20_candidates_table.png`. Sample rows:

{_df_md(candidates['top20_by_EPR_err'].head(8), cols_show_top20)}

## Recommendation

{recommendation}

## CSV exports

- `all_candidates.csv` — full sweep with every metric
- `plausible_candidates.csv` — plausible subset only
- `top20_candidates.csv` — top 20 by EPR error

## Plot index

| # | Plot | File |
|---|---|---|
| 13 | BPR × Wc_fan heatmap, color = min EPR err, contour = T45 err | `13_heatmap_BPR_Wcfan_EPRerr.png` |
| 14 | BPR × Wc_fan heatmap, color = T50 at min-EPR cell | `14_heatmap_BPR_Wcfan_T50.png` |
| 15 | Pareto T45 err vs EPR err, color=BPR, size~Wc_fan, X=old baseline | `15_pareto_T45err_vs_EPRerr.png` |
| 16 | LP-spool decomposition (W_fan, W_lpc, W_lpt, PR_lpt, T50) vs BPR | `16_LP_spool_decomposition_vs_BPR.png` |
| 17 | Top-5 candidate station total-pressure profile | `17_station_pressure_top5.png` |
| 18 | Top-5 candidate station total-temperature profile | `18_station_temperature_top5.png` |
| 19 | Top-20 candidate table (image) | `19_top20_candidates_table.png` |

---

*Stop. No automatic parameter selection. Awaiting Robert review.*
"""
    p = OUT_DIR / "c0c_bpr_wcfan_report.md"
    p.write_text(md, encoding="utf-8")
    print(f"  saved  {p}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("V3.1b C0c BPR / Wc_fan feasibility diagnostic at FC02")
    print(f"Output dir: {OUT_DIR}")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fc02 = fc02_conditions_si()

    print("\nSweeping (this is a 7-D grid; please wait)...")
    df, n_attempted, n_pre_reject = sweep_grid(fc02)
    print(f"  attempted = {n_attempted},  pre-rejected = {n_pre_reject},  "
          f"forward-pass = {len(df)}")

    plaus_mask = plausibility_mask(df)
    df_plaus = df[plaus_mask].copy()
    print(f"  plausible = {len(df_plaus)} / {len(df)} "
          f"({len(df_plaus) / max(len(df), 1) * 100:.1f} %)")

    # Old baseline (single point, computed independently)
    base = OLD_BASELINE
    baseline_row = evaluate_point(
        fc02, base["bpr"], base["Wc_fan_design"], base["target_OPR"],
        base["pr_fan"], base["pr_lpc"], base["pr_hpc"],
        base["eta_hpt"], base["eta_lpt"],
    )

    candidates = best_candidates(df_plaus, df)
    pass_rates = per_criterion_pass_rates(df)

    # CSV exports
    df.to_csv(OUT_DIR / "all_candidates.csv", index=False)
    df_plaus.to_csv(OUT_DIR / "plausible_candidates.csv", index=False)
    candidates["top20_by_EPR_err"].to_csv(OUT_DIR / "top20_candidates.csv", index=False)
    print(f"  saved  all_candidates.csv, plausible_candidates.csv, top20_candidates.csv")

    print("\nGenerating plots...")
    plot_heatmap_eprerr(df)
    plot_heatmap_T50(df)
    plot_pareto(df, baseline_row)
    plot_LP_decomposition(df)
    _top5_curves(df_plaus, df, fc02,
                 "17_station_pressure_top5.png",
                 "18_station_temperature_top5.png")
    plot_top20_table(candidates["top20_by_EPR_err"], df)

    print("\nWriting report...")
    write_report(df, df_plaus, candidates, pass_rates,
                 baseline_row, n_attempted, n_pre_reject)

    print("\n" + "=" * 72)
    print(f"Done.  Plausible: {len(df_plaus)} / {len(df)}.  No YAML written.")
    print("=" * 72)


if __name__ == "__main__":
    main()
