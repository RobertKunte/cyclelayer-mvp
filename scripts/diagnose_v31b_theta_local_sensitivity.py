"""Diagnostic 1 — local Jacobian of BraytonEngine outputs w.r.t. each θ.

ADR-0013, Task 2. Read-only.

Goal: directly measure how each of `T24, T30, T45, T50, P30, P45, P50,
PR_hpt, PR_lpt, EPR` responds to each of `θ_η_fan, θ_η_lpc, θ_η_hpc,
θ_η_hpt, θ_η_lpt` at representative operating points.

Engineering hypothesis to verify or falsify: in V3.1b's closure (Nf, Nc,
Wf fixed inputs), `θ_η_hpt` and `θ_η_lpt` have ~0 effect on turbine
*outlet temperatures* and nonzero effect only on *pressure ratios*.  If
confirmed, this is the architectural reason `L_temp` (T-only) cannot
identify them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import torch         # noqa: E402
import yaml          # noqa: E402

from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset, load_userguide_fc02_anchor  # noqa: E402
from cyclelayer.models import units  # noqa: E402
from cyclelayer.models.stations import GAMMA_C  # noqa: E402

# Reuse builders from the smoke script
sys.path.insert(0, str(Path(__file__).parent))
from train_cyclelayer_v3_thermal_aux_smoke import build_brayton_from_cfg  # noqa: E402


OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "theta_identifiability"


def df_to_md(df: pd.DataFrame, floatfmt: str = ".3e") -> str:
    """Minimal markdown table renderer (avoids `tabulate` optional dep)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


OUTPUT_KEYS = [
    ("T24",      "T24_K",   "sensors_pred_si"),
    ("T30",      "T30_K",   "sensors_pred_si"),
    ("T45",      "T45",     "diag"),
    ("T50",      "T50_K",   "sensors_pred_si"),
    ("P30",      "P30_Pa",  "sensors_pred_si"),
    ("P45",      "P45",     "diag"),
    ("P50",      "P50",     "diag"),
    ("PR_hpt",   "PR_hpt",  "diag"),
    ("PR_lpt",   "PR_lpt",  "diag"),
    # EPR computed manually = P50 / P2_input
]
THETA_NAMES = ["eta_fan", "eta_lpc", "eta_hpc", "eta_hpt", "eta_lpt"]


def fc02_si_inputs() -> tuple[dict, dict, float]:
    fc = load_userguide_fc02_anchor()
    Tsl_R = fc["Tsl_F"] + 459.67
    P0_psia = 14.696
    M = fc["XM"]
    ram_T = 1.0 + 0.5 * (GAMMA_C - 1.0) * M ** 2
    ram_P = ram_T ** (GAMMA_C / (GAMMA_C - 1.0))
    T2_R  = Tsl_R * ram_T
    P2_psia = P0_psia * ram_P
    ops_si = {
        "T2_K":  torch.tensor([T2_R * units.RANK_TO_K]),
        "P2_Pa": torch.tensor([P2_psia * units.PSIA_TO_PA]),
        "alt_m": torch.tensor([0.0]),
        "mach":  torch.tensor([float(M)]),
    }
    sens_si = {
        "Nf_rpm": torch.tensor([float(fc["Nf_rpm"])]),
        "Nc_rpm": torch.tensor([float(fc["Nc_rpm"])]),
        "Wf_kgs": torch.tensor([float(fc["Wf_pps"]) * units.PPS_TO_KGS]),
    }
    return ops_si, sens_si, float(ops_si["P2_Pa"].item())


def ds02_row_inputs(base_ds: NCMAPSSV3Dataset, idx: int) -> tuple[dict, dict, float]:
    """Imperial → SI inputs for a single DS02 row (last-timestep convention)."""
    s = base_ds[idx]   # per-row dict
    T2_R    = float(s["ops_imp"]["T2_R"])
    P2_psia = float(s["ops_imp"]["P2_psia"])
    alt_ft  = float(s["ops_imp"]["alt_ft"])
    mach    = float(s["ops_imp"]["XM"])
    Nf_rpm  = float(s["sens_imp"]["Nf_rpm"])
    Nc_rpm  = float(s["sens_imp"]["Nc_rpm"])
    Wf_pps  = float(s["sens_imp"]["Wf_pps"])
    ops_si = {
        "T2_K":  torch.tensor([T2_R * units.RANK_TO_K]),
        "P2_Pa": torch.tensor([P2_psia * units.PSIA_TO_PA]),
        "alt_m": torch.tensor([alt_ft * units.FT_TO_M]),
        "mach":  torch.tensor([mach]),
    }
    sens_si = {
        "Nf_rpm": torch.tensor([Nf_rpm]),
        "Nc_rpm": torch.tensor([Nc_rpm]),
        "Wf_kgs": torch.tensor([Wf_pps * units.PPS_TO_KGS]),
    }
    return ops_si, sens_si, float(ops_si["P2_Pa"].item())


def jacobian_at_point(engine, ops_si: dict, sens_si: dict,
                      P2_Pa: float, theta0: torch.Tensor) -> dict:
    """Return dict {output_name: tensor[5] of d output / d theta_i} via autograd.

    Also returns elasticity = d log(out) / d log(theta) = (theta/out) * d/d theta
    """
    rows = []
    for out_name, out_key, out_group in OUTPUT_KEYS:
        theta = theta0.clone().detach().requires_grad_(True).unsqueeze(0)   # (1, 5)
        sensors_pred_si, diag = engine(ops_si, sens_si, theta)
        if out_group == "sensors_pred_si":
            y = sensors_pred_si[out_key].squeeze()
        else:
            y = diag[out_key].squeeze()
        grad = torch.autograd.grad(y, theta, retain_graph=False)[0].detach().squeeze()
        out_val = float(y.detach().item())
        rows.append({
            "output":         out_name,
            "value":          out_val,
            **{f"d_d_{n}": float(grad[i].item()) for i, n in enumerate(THETA_NAMES)},
            **{f"elasticity_{n}": (float(theta0[i].item()) / out_val * float(grad[i].item()))
               if abs(out_val) > 1e-12 else float("nan")
               for i, n in enumerate(THETA_NAMES)},
        })
    # EPR row: P50 / P2_Pa (P2 is input → constant w.r.t. theta)
    theta = theta0.clone().detach().requires_grad_(True).unsqueeze(0)
    _, diag = engine(ops_si, sens_si, theta)
    P50 = diag["P50"].squeeze()
    epr = P50 / P2_Pa
    grad = torch.autograd.grad(epr, theta, retain_graph=False)[0].detach().squeeze()
    out_val = float(epr.detach().item())
    rows.append({
        "output": "EPR",
        "value":  out_val,
        **{f"d_d_{n}": float(grad[i].item()) for i, n in enumerate(THETA_NAMES)},
        **{f"elasticity_{n}":
            (float(theta0[i].item()) / out_val * float(grad[i].item()))
            if abs(out_val) > 1e-12 else float("nan")
            for i, n in enumerate(THETA_NAMES)},
    })
    return rows


def plot_heatmap(df_avg: pd.DataFrame, out_path: Path) -> None:
    elast_cols = [f"elasticity_{n}" for n in THETA_NAMES]
    mat = df_avg[elast_cols].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(THETA_NAMES, rotation=20)
    ax.set_yticks(range(len(df_avg))); ax.set_yticklabels(df_avg["output"].tolist())
    ax.set_xlabel("theta channel")
    ax.set_ylabel("BraytonEngine output")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                txt = f"{v:.2f}" if abs(v) >= 1e-3 else f"{v:.0e}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if abs(v) > vmax * 0.5 else "black", fontsize=8)
    ax.set_title("Mean elasticity  d log(output) / d log(theta)  across operating points")
    fig.colorbar(im, ax=ax, label="elasticity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(__file__).parent.parent / "configs" / "cyclelayer_v3_thermal_aux.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    engine = build_brayton_from_cfg(cfg["model"]["brayton_engine"])

    theta0 = torch.tensor([0.95, 0.95, 0.95, 0.95, 0.95])   # mildly degraded baseline

    all_rows: list[dict] = []

    # ── (a) FC02-style design-ish point ────────────────────────────────
    print("[1] FC02 anchor point ...")
    ops_si, sens_si, P2 = fc02_si_inputs()
    for r in jacobian_at_point(engine, ops_si, sens_si, P2, theta0):
        r["point"] = "FC02"; all_rows.append(r)

    # ── (b) DS02 sample rows (lazy load — read-only) ───────────────────
    ds02 = Path(cfg["data"]["hdf5_path"])
    if not ds02.is_absolute():
        ds02 = Path(__file__).parent.parent / ds02
    if ds02.exists():
        print(f"[2] DS02 rows ({ds02}) ...")
        base = NCMAPSSV3Dataset(ds02, split="dev", load_in_memory=False)
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(base), size=min(10, len(base)), replace=False)
        for j, idx in enumerate(idxs):
            ops_si, sens_si, P2 = ds02_row_inputs(base, int(idx))
            for r in jacobian_at_point(engine, ops_si, sens_si, P2, theta0):
                r["point"] = f"DS02_row_{int(idx)}"; all_rows.append(r)
        base.close()
    else:
        print(f"[2] DS02 not at {ds02} — skipping DS02 rows.")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "local_sensitivity.csv", index=False)
    print(f"  saved {OUT_DIR / 'local_sensitivity.csv'}  ({len(df)} rows)")

    # Per-output averaged elasticity across all points
    elast_cols = [f"elasticity_{n}" for n in THETA_NAMES]
    df_avg = df.groupby("output", sort=False)[elast_cols].mean().reset_index()
    df_avg.to_csv(OUT_DIR / "local_sensitivity_mean_elasticity.csv", index=False)
    print(f"  saved {OUT_DIR / 'local_sensitivity_mean_elasticity.csv'}")

    plot_heatmap(df_avg, OUT_DIR / "local_sensitivity_heatmap.png")
    print(f"  saved {OUT_DIR / 'local_sensitivity_heatmap.png'}")

    # ── Interpretation: which thetas are observable from temperatures only? ──
    temp_outputs = ["T24", "T30", "T45", "T50"]
    pressure_outputs = ["P30", "P45", "P50", "PR_hpt", "PR_lpt", "EPR"]

    mask_T = df_avg["output"].isin(temp_outputs)
    mask_P = df_avg["output"].isin(pressure_outputs)

    summary = {}
    for theta in THETA_NAMES:
        e_T = df_avg.loc[mask_T, f"elasticity_{theta}"].abs().max()
        e_P = df_avg.loc[mask_P, f"elasticity_{theta}"].abs().max()
        summary[theta] = {
            "max_abs_elasticity_T_outputs": float(e_T),
            "max_abs_elasticity_P_outputs": float(e_P),
            "identifiable_from_T_only_threshold_1e-2": bool(e_T > 1e-2),
        }

    md = f"""# Local sensitivity — V3.1b θ identifiability (Task 2)

*Read-only diagnostic.  ADR-0013.*

## Setup

* operating points: FC02 design-ish anchor + 10 random DS02 rows (if DS02 available locally)
* θ baseline: `[0.95, 0.95, 0.95, 0.95, 0.95]` (mildly degraded)
* sensitivity: exact via `torch.autograd`
* normalized elasticity: `(θ / output) · (∂output / ∂θ)`

## Per-output mean elasticity (across all points)

{df_to_md(df_avg, floatfmt='.3e')}

## Per-θ max |elasticity|: temperature outputs vs pressure outputs

| θ | max |elasticity| on {{T24,T30,T45,T50}} | max |elasticity| on {{P30,P45,P50,PR_hpt,PR_lpt,EPR}} | T-identifiable? (|el| > 1e-2) |
|---|---|---|---|
"""
    for theta in THETA_NAMES:
        s = summary[theta]
        md += (f"| {theta} | {s['max_abs_elasticity_T_outputs']:.3e} | "
               f"{s['max_abs_elasticity_P_outputs']:.3e} | "
               f"{'YES' if s['identifiable_from_T_only_threshold_1e-2'] else '**NO**'} |\n")

    md += """
## Interpretation

* If `eta_fan / eta_lpc / eta_hpc` show meaningful elasticity on T24/T30 and `eta_hpt / eta_lpt` do NOT,
  the V3.1b closure architecturally rules out HPT/LPT efficiency identification from a T-only loss.
* HPT/LPT θ elasticity on pressure outputs (P45/P50/PR_hpt/PR_lpt/EPR) being nonzero is the
  *expected* physical behavior — and the reason V4 with pressure loss is required to identify them.

See `local_sensitivity_heatmap.png` for the elasticity grid.
"""
    (OUT_DIR / "local_sensitivity_report.md").write_text(md, encoding="utf-8")
    print(f"  saved {OUT_DIR / 'local_sensitivity_report.md'}")

    print("\n=== Quick summary ===")
    for theta in THETA_NAMES:
        s = summary[theta]
        flag = "OK from T" if s["identifiable_from_T_only_threshold_1e-2"] else "NOT id'able from T"
        print(f"  {theta:8s}  max|elast|_T={s['max_abs_elasticity_T_outputs']:.2e}  "
              f"max|elast|_P={s['max_abs_elasticity_P_outputs']:.2e}  -> {flag}")


if __name__ == "__main__":
    main()
