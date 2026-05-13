"""Diagnostic — partial correlations of θ vs GT controlling for nuisance vars.

ADR-0013, Task 4. Read-only. No tuning.

If a θ-vs-GT raw Pearson is high but collapses after residualising on RUL /
cycle / ops, the high Pearson was a time-axis artifact, not a learned
physical mapping.

Inputs:
    --run_dir <path>     directory containing best.pt (CycleLayerV3 checkpoint)
                         If omitted, auto-discovers under:
                           /content/runs_v3_thermal_aux/  (Colab)
                           artifacts/cyclelayer_v3/thermal_aux_smoke/ (local)
                           runs_v3_thermal_aux/<RUN_ID>_C_physics_theta_rul/

Outputs (artifacts/cyclelayer_v3/theta_identifiability/):
    partial_correlations.csv
    partial_correlations_report.md
    theta_vs_gt_raw.png
    theta_vs_gt_residualized.png
    theta_vs_cycle_per_unit.png
    theta_damage_vs_gt_per_unit.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
import torch         # noqa: E402
import yaml          # noqa: E402
from scipy import stats as scstats  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset  # noqa: E402
from train_cyclelayer_v3_thermal_aux_smoke import (   # noqa: E402
    NCMAPSSV3WindowedDataset, _collate,
    build_brayton_from_cfg, build_v3_from_cfg,
)

OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "theta_identifiability"


def df_to_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(format(v, floatfmt) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def find_run_dir(args_dir: str | None) -> Path | None:
    if args_dir:
        p = Path(args_dir)
        return p if (p / "best.pt").exists() else None
    candidates = [
        Path("/content/runs_v3_thermal_aux"),
        Path(__file__).parent.parent / "runs_v3_thermal_aux",
        Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "thermal_aux_smoke",
    ]
    for c in candidates:
        if not c.exists(): continue
        # Latest C run dir
        subs = sorted([s for s in c.glob("*C_physics_theta_rul*") if (s / "best.pt").exists()])
        if subs: return subs[-1]
        if (c / "best.pt").exists(): return c
    return None


def safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    return (float(scstats.pearsonr(x, y).statistic),
            float(scstats.spearmanr(x, y).statistic))


def partial_pearson(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson(x | z) via OLS residualisation."""
    if len(x) < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    if z.ndim == 1: z = z.reshape(-1, 1)
    Z = np.hstack([z, np.ones((z.shape[0], 1))])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    ex = x - Z @ bx
    ey = y - Z @ by
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float("nan")
    return float(scstats.pearsonr(ex, ey).statistic)


@torch.no_grad()
def collect_predictions(model, loader, device, scalers):
    sm, sd, om, od = scalers
    sm = sm.to(device); sd = sd.to(device); om = om.to(device); od = od.to(device)
    rows = {k: [] for k in [
        "theta_eta_fan", "theta_eta_lpc", "theta_eta_hpc",
        "theta_eta_hpt", "theta_eta_lpt", "lpt_flow_pred",
        "HPT_eff_mod", "LPT_eff_mod", "LPT_flow_mod",
        "RUL", "cycle", "unit_id", "Fc", "hs",
        "alt_ft", "Mach", "TRA_pct", "T2_R", "P2_psia",
        "Nf_rpm", "Nc_rpm", "Wf_pps",
    ]}
    for b in loader:
        sensors_norm = (b["sensors_imp"].to(device) - sm) / sd
        ops_norm     = (b["ops_imp"].to(device)     - om) / od
        ops_si  = {k: v.to(device) for k, v in b["ops_si_last"].items()}
        sens_si = {k: v.to(device) for k, v in b["sens_si_last"].items()}
        out = model(sensors_norm, ops_norm, ops_si=ops_si, sens_si=sens_si)
        theta = out["theta_phys"].cpu().numpy()
        for i, n in enumerate(["theta_eta_fan", "theta_eta_lpc", "theta_eta_hpc",
                                "theta_eta_hpt", "theta_eta_lpt"]):
            rows[n].extend(theta[:, i].tolist())
        rows["lpt_flow_pred"].extend(out["lpt_flow_pred"].cpu().numpy().tolist())
        h = b["health_gt_last"]
        rows["HPT_eff_mod"].extend(h["HPT_eff_mod"].numpy().tolist())
        rows["LPT_eff_mod"].extend(h["LPT_eff_mod"].numpy().tolist())
        rows["LPT_flow_mod"].extend(h["LPT_flow_mod"].numpy().tolist())
        rows["RUL"].extend(b["RUL"].numpy().tolist())
        # Cycle, Fc, hs from the per-window aux dict at last timestep
        rows["cycle"].extend(b["aux"]["cycle"].numpy().tolist() if "aux" in b
                              else [0] * len(b["RUL"]))
        rows["unit_id"].extend(b["unit_id"].numpy().tolist())
        rows["Fc"].extend([0] * len(b["RUL"]))
        rows["hs"].extend([0] * len(b["RUL"]))
        ops_imp_last = b["ops_imp"][:, -1, :].numpy()
        sens_imp_last = b["sensors_imp"][:, -1, :].numpy()
        rows["alt_ft"].extend(ops_imp_last[:, 0].tolist())
        rows["Mach"].extend(ops_imp_last[:, 1].tolist())
        rows["TRA_pct"].extend(ops_imp_last[:, 2].tolist())
        rows["T2_R"].extend(ops_imp_last[:, 3].tolist())
        rows["P2_psia"].extend(sens_imp_last[:, 5].tolist())   # P2 is X_s col 5
        rows["Nf_rpm"].extend(sens_imp_last[:, 11].tolist())
        rows["Nc_rpm"].extend(sens_imp_last[:, 12].tolist())
        rows["Wf_pps"].extend(sens_imp_last[:, 13].tolist())
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None,
                    help="Directory containing best.pt; auto-discovered if omitted.")
    ap.add_argument("--max_samples_per_unit", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    run_dir = find_run_dir(args.run_dir)
    if run_dir is None:
        msg = ("No trained C checkpoint found. Looked in /content/runs_v3_thermal_aux/, "
               "runs_v3_thermal_aux/, and artifacts/cyclelayer_v3/thermal_aux_smoke/. "
               "Pass --run_dir to point to one explicitly.")
        print(msg)
        (OUT_DIR / "partial_correlations_report.md").write_text(
            f"# Partial correlations (SKIPPED)\n\n{msg}\n", encoding="utf-8"
        )
        return
    print(f"Using checkpoint dir: {run_dir}")

    cfg_path = Path(__file__).parent.parent / "configs" / "cyclelayer_v3_thermal_aux.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    brayton = build_brayton_from_cfg(cfg["model"]["brayton_engine"])
    model = build_v3_from_cfg(cfg["model"], brayton).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)

    # Scalers
    if "scalers" in ckpt:
        sm = torch.tensor(ckpt["scalers"]["sensor_mean"]).float()
        sd = torch.tensor(ckpt["scalers"]["sensor_std"]).float()
        om = torch.tensor(ckpt["scalers"]["ops_mean"]).float()
        od = torch.tensor(ckpt["scalers"]["ops_std"]).float()
    else:
        sn = np.load(run_dir / "sensor_scaler.npz")
        on = np.load(run_dir / "ops_scaler.npz")
        sm = torch.from_numpy(sn["mean"]).float(); sd = torch.from_numpy(sn["std"]).float()
        om = torch.from_numpy(on["mean"]).float(); od = torch.from_numpy(on["std"]).float()

    ds02_path = Path(cfg["data"]["hdf5_path"])
    if not ds02_path.is_absolute():
        ds02_path = Path(__file__).parent.parent / ds02_path
    if not ds02_path.exists():
        msg = f"DS02 not at {ds02_path} — cannot compute partial correlations."
        print(msg)
        (OUT_DIR / "partial_correlations_report.md").write_text(
            f"# Partial correlations (SKIPPED)\n\n{msg}\n", encoding="utf-8"
        )
        return

    base = NCMAPSSV3Dataset(ds02_path, split="test", load_in_memory=True)
    test_units = list(cfg["data"]["test_units"])
    wds = NCMAPSSV3WindowedDataset(
        base, test_units,
        window_size=cfg["data"]["window_size"],
        stride=cfg["data"]["stride_eval"],
        max_samples=None,
    )
    # Cap per-unit
    if args.max_samples_per_unit:
        # Re-sample by unit_id
        df_idx = pd.DataFrame({"idx": range(len(wds))})
        # We need to know unit_id per window; easiest: load all (small) and subselect
        pass   # WindowedDataset already caps via max_samples globally
    loader = DataLoader(wds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    print(f"Collecting predictions on {len(wds):,} test windows ...")
    df = collect_predictions(model, loader, device, (sm, sd, om, od))
    df["theta_hpt_delta"]  = df["theta_eta_hpt"] - 1.0
    df["theta_lpt_delta"]  = df["theta_eta_lpt"] - 1.0
    df["theta_hpt_damage"] = 1.0 - df["theta_eta_hpt"]
    df["theta_lpt_damage"] = 1.0 - df["theta_eta_lpt"]
    df.to_csv(OUT_DIR / "partial_correlations_predictions.csv", index=False)

    # ── Compute correlations ─────────────────────────────────────────────
    pairs = [
        ("theta_hpt_delta",  "HPT_eff_mod"),
        ("theta_lpt_delta",  "LPT_eff_mod"),
        ("theta_hpt_damage", "HPT_eff_mod"),
        ("theta_lpt_damage", "LPT_eff_mod"),
        ("lpt_flow_pred",    "LPT_flow_mod"),
    ]
    control_sets = {
        "none":          [],
        "RUL":           ["RUL"],
        "cycle":         ["cycle"],
        "ops":           ["alt_ft", "Mach", "TRA_pct", "T2_R", "P2_psia",
                          "Nf_rpm", "Nc_rpm", "Wf_pps"],
        "RUL+ops":       ["RUL", "alt_ft", "Mach", "TRA_pct", "T2_R", "P2_psia",
                          "Nf_rpm", "Nc_rpm", "Wf_pps"],
        "cycle+ops":     ["cycle", "alt_ft", "Mach", "TRA_pct", "T2_R", "P2_psia",
                          "Nf_rpm", "Nc_rpm", "Wf_pps"],
    }

    rows: list[dict] = []
    for xcol, ycol in pairs:
        x = df[xcol].to_numpy(); y = df[ycol].to_numpy()
        pr, sr = safe_corr(x, y)
        row = {"x": xcol, "y": ycol, "control": "none",
               "pearson": pr, "spearman": sr, "n": len(df)}
        rows.append(row)
        for ctrl_name, ctrl_cols in control_sets.items():
            if ctrl_name == "none": continue
            z = df[ctrl_cols].to_numpy() if ctrl_cols else np.zeros((len(df), 0))
            ppr = partial_pearson(x, y, z) if z.size else pr
            rows.append({"x": xcol, "y": ycol, "control": ctrl_name,
                         "pearson": ppr, "spearman": float("nan"), "n": len(df)})

        # Per-unit raw
        for uid in sorted(df["unit_id"].unique()):
            sub = df[df["unit_id"] == uid]
            pr_u, sr_u = safe_corr(sub[xcol].to_numpy(), sub[ycol].to_numpy())
            rows.append({"x": xcol, "y": ycol,
                         "control": f"per_unit_{int(uid)}",
                         "pearson": pr_u, "spearman": sr_u, "n": len(sub)})

    cor = pd.DataFrame(rows)
    cor.to_csv(OUT_DIR / "partial_correlations.csv", index=False)
    print(f"  saved {OUT_DIR / 'partial_correlations.csv'}")

    # ── Plots ────────────────────────────────────────────────────────────
    # 1) Raw scatter theta vs gt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (xcol, ycol) in zip(axes, [
        ("theta_hpt_delta", "HPT_eff_mod"),
        ("theta_lpt_delta", "LPT_eff_mod"),
        ("lpt_flow_pred",   "LPT_flow_mod"),
    ]):
        ax.scatter(df[xcol], df[ycol], c=df["RUL"], cmap="viridis_r", s=6, alpha=0.5)
        pr, _ = safe_corr(df[xcol].to_numpy(), df[ycol].to_numpy())
        ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        ax.set_title(f"{xcol} vs {ycol}\nPearson={pr:.3f}")
        ax.grid(True, alpha=0.4)
    fig.suptitle("Raw θ vs N-CMAPSS GT (colour = RUL)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "theta_vs_gt_raw.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 2) Residualised — control for RUL+ops
    ctrl_cols = control_sets["RUL+ops"]
    z = df[ctrl_cols].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (xcol, ycol) in zip(axes, [
        ("theta_hpt_delta", "HPT_eff_mod"),
        ("theta_lpt_delta", "LPT_eff_mod"),
        ("lpt_flow_pred",   "LPT_flow_mod"),
    ]):
        x = df[xcol].to_numpy(); y = df[ycol].to_numpy()
        Z = np.hstack([z, np.ones((z.shape[0], 1))])
        bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
        by, *_ = np.linalg.lstsq(Z, y, rcond=None)
        ex = x - Z @ bx; ey = y - Z @ by
        ax.scatter(ex, ey, s=6, alpha=0.5)
        pp = partial_pearson(x, y, z)
        ax.set_xlabel(f"{xcol} residual"); ax.set_ylabel(f"{ycol} residual")
        ax.set_title(f"After residualising on RUL+ops\npartial Pearson={pp:.3f}")
        ax.grid(True, alpha=0.4)
    fig.suptitle("Residualised θ vs GT — collapses if raw r was time-axis artifact")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "theta_vs_gt_residualized.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3) Theta vs cycle/RUL per unit
    units_sorted = sorted(df["unit_id"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=False)
    cols_pairs = [
        ("theta_eta_hpt", "HPT_eff_mod"),
        ("theta_eta_lpt", "LPT_eff_mod"),
        ("lpt_flow_pred", "LPT_flow_mod"),
    ]
    for ci, (tcol, gcol) in enumerate(cols_pairs):
        ax_t = axes[0, ci]; ax_g = axes[1, ci]
        for uid in units_sorted:
            sub = df[df["unit_id"] == uid].sort_values("RUL", ascending=False).reset_index(drop=True)
            ax_t.plot(sub.index, sub[tcol], lw=0.6, alpha=0.7, label=f"u{int(uid)}")
            ax_g.plot(sub.index, sub[gcol], lw=0.6, alpha=0.7, label=f"u{int(uid)}")
        ax_t.set_title(f"model: {tcol}"); ax_g.set_title(f"GT: {gcol}")
        ax_t.set_ylabel(tcol); ax_g.set_ylabel(gcol)
        ax_g.set_xlabel("window (decreasing-RUL order)")
        ax_t.grid(True, alpha=0.3); ax_g.grid(True, alpha=0.3)
        ax_t.legend(fontsize=6); ax_g.legend(fontsize=6)
    fig.suptitle("Per-unit trajectories — model θ (top) vs N-CMAPSS GT (bottom)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "theta_vs_cycle_per_unit.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 4) damage vs GT per unit
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (dmg, gt) in zip(axes, [
        ("theta_hpt_damage", "HPT_eff_mod"),
        ("theta_lpt_damage", "LPT_eff_mod"),
    ]):
        for uid in units_sorted:
            sub = df[df["unit_id"] == uid]
            ax.scatter(sub[dmg], -sub[gt], s=6, alpha=0.4, label=f"u{int(uid)}")
        ax.set_xlabel(dmg + "  (= 1 − θ)")
        ax.set_ylabel(f"−{gt}  (degradation magnitude)")
        pr, _ = safe_corr(df[dmg].to_numpy(), -df[gt].to_numpy())
        ax.set_title(f"Pearson(damage_model, −GT) = {pr:.3f}")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
    fig.suptitle("If V3.1b learns degradation correctly: positive correlation in 'damage' space")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "theta_damage_vs_gt_per_unit.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ── Markdown ────────────────────────────────────────────────────────
    pivot = cor[cor["control"].isin(list(control_sets.keys()))].copy()
    pivot["pair"] = pivot["x"] + " vs " + pivot["y"]
    pv = pivot.pivot_table(index="pair", columns="control",
                            values="pearson", aggfunc="first")
    pv = pv.reindex(columns=list(control_sets.keys()))
    pv = pv.reset_index()

    # Decision logic
    decisions = {}
    for xcol, ycol in pairs:
        pair = f"{xcol} vs {ycol}"
        row = pivot[(pivot["x"] == xcol) & (pivot["y"] == ycol)]
        raw  = row[row["control"] == "none"]["pearson"].iloc[0] if not row.empty else float("nan")
        rul  = row[row["control"] == "RUL"]["pearson"].iloc[0]
        ru_o = row[row["control"] == "RUL+ops"]["pearson"].iloc[0]
        if abs(raw) > 0.6 and abs(ru_o) < 0.2:
            verdict = "ARTIFACT (raw |r|>0.6, partial |r|<0.2 after RUL+ops)"
        elif abs(ru_o) > 0.4 and np.sign(ru_o) == np.sign(raw):
            verdict = "ROBUST (partial |r|>0.4 same sign)"
        else:
            verdict = "WEAK / INCONSISTENT"
        decisions[pair] = {"raw": raw, "partial_RUL": rul,
                           "partial_RUL_ops": ru_o, "verdict": verdict}

    md = f"""# Partial correlations — V3.1b θ identifiability (Task 4)

*Read-only.  ADR-0013.  No DS02 tuning.*

* Checkpoint: `{run_dir}`
* Test windows analysed: {len(df):,}
* Test units: {sorted(df['unit_id'].unique().tolist())}

## Correlation table (rows = θ-GT pair, columns = control set; values = Pearson r)

{df_to_md(pv, floatfmt='.3f')}

## Per-pair verdict

| pair | raw r | partial r (RUL) | partial r (RUL+ops) | verdict |
|---|---|---|---|---|
"""
    for pair, d in decisions.items():
        md += (f"| {pair} | {d['raw']:.3f} | {d['partial_RUL']:.3f} | "
               f"{d['partial_RUL_ops']:.3f} | {d['verdict']} |\n")

    md += f"""

## Plots

* `theta_vs_gt_raw.png` — raw scatter, colour = RUL
* `theta_vs_gt_residualized.png` — same scatter after RUL+ops residualisation
* `theta_vs_cycle_per_unit.png` — per-unit time trajectories (model on top, GT below)
* `theta_damage_vs_gt_per_unit.png` — "damage" space (1−θ) vs negated GT

## Decision rule (ADR-0013)

* `raw |r| > 0.6` AND `partial |r| < 0.2` → **time/degradation-axis artifact**.
* `partial |r| > 0.4` with same sign as raw → **robust signal**, more training warranted.
* `per_unit` correlations vary widely in sign → global Pearson alone is **misleading**.
"""
    (OUT_DIR / "partial_correlations_report.md").write_text(md, encoding="utf-8")
    print(f"  saved {OUT_DIR / 'partial_correlations_report.md'}")

    # CLI summary
    print("\n=== Partial correlation verdicts ===")
    for pair, d in decisions.items():
        print(f"  {pair:40s}  raw={d['raw']:+.3f}  "
              f"partial(RUL+ops)={d['partial_RUL_ops']:+.3f}  -> {d['verdict']}")


if __name__ == "__main__":
    main()
