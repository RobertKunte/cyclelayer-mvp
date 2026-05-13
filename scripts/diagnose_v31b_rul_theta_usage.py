"""Diagnostic — Does the RUL head actually use θ_phys?  (Task 6, ADR-0013).

Read-only. No tuning.

For the trained C checkpoint, evaluate RUL on the test split under several
θ-perturbation variants and report ΔRMSE / ΔMAE / Δbias. Also reports
prognostics-head weight norms on the θ feature slots and ∂L_rul / ∂θ.

If all perturbations move RMSE by < 0.5% and ∂L_rul/∂θ is near zero,
the RUL head is ignoring θ_phys.
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
from torch.utils.data import DataLoader  # noqa: E402

from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset  # noqa: E402
from train_cyclelayer_v3_thermal_aux_smoke import (   # noqa: E402
    NCMAPSSV3WindowedDataset, _collate,
    build_brayton_from_cfg, build_v3_from_cfg,
)


OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "theta_identifiability"
THETA_NAMES = ["eta_fan", "eta_lpc", "eta_hpc", "eta_hpt", "eta_lpt"]


def _find_run_dir(args_dir: str | None) -> Path | None:
    if args_dir:
        p = Path(args_dir)
        return p if (p / "best.pt").exists() else None
    candidates = [
        Path("/content/runs_v3_thermal_aux"),
        Path(__file__).parent.parent / "runs_v3_thermal_aux",
        Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "thermal_aux_smoke",
    ]
    for c in candidates:
        if not c.exists():
            continue
        subs = sorted([s for s in c.glob("*C_physics_theta_rul*")
                       if (s / "best.pt").exists()])
        if subs:
            return subs[-1]
        if (c / "best.pt").exists():
            return c
    return None


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


@torch.no_grad()
def infer_with_theta_override(model, loader, device, scalers,
                              variant: str = "real") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sm, sd, om, od = scalers
    sm = sm.to(device); sd = sd.to(device); om = om.to(device); od = od.to(device)
    rul_pred_all, rul_true_all, unit_all = [], [], []
    for b in loader:
        sensors_norm = (b["sensors_imp"].to(device) - sm) / sd
        ops_norm     = (b["ops_imp"].to(device)     - om) / od
        ops_si  = {k: v.to(device) for k, v in b["ops_si_last"].items()}
        sens_si = {k: v.to(device) for k, v in b["sens_si_last"].items()}
        out = model(sensors_norm, ops_norm, ops_si=ops_si, sens_si=sens_si)
        theta_real = out["theta_phys"]
        unit_ids   = b["unit_id"]

        if variant == "real":
            theta_used = theta_real
        elif variant == "shuffle_batch":
            theta_used = theta_real[torch.randperm(theta_real.shape[0], device=device)]
        elif variant == "shuffle_across_units":
            # Permute samples but keep unit alignment broken
            theta_used = theta_real[torch.randperm(theta_real.shape[0], device=device)]
        elif variant == "shuffle_within_unit":
            theta_used = theta_real.clone()
            for uid in unit_ids.unique():
                mask = (unit_ids == uid)
                idx = torch.nonzero(mask).squeeze(-1)
                if len(idx) > 1:
                    perm = idx[torch.randperm(len(idx))]
                    theta_used[idx] = theta_real[perm]
        elif variant == "constant_healthy":
            theta_used = torch.full_like(theta_real, 0.99)
        elif variant == "constant_lo":
            theta_used = torch.full_like(theta_real, 0.85)
        elif variant == "zero_in_RUL_features_only":
            theta_used = torch.zeros_like(theta_real)   # outside [0.85,1.0], but RUL head doesn't clamp
        else:
            raise ValueError(variant)

        # Rebuild RUL feature vector with the overridden theta
        cfg = model.config
        h_sens = out["h_sens"]; z_ops = out["z_ops"]
        parts = [h_sens]
        if z_ops is not None: parts.append(z_ops)
        if cfg.use_theta_in_rul: parts.append(theta_used)
        if cfg.use_aux_in_rul:   parts.append(out["lpt_flow_pred"].unsqueeze(-1))
        rul = model.prognostics(torch.cat(parts, dim=-1))

        rul_pred_all.append(rul.cpu().numpy())
        rul_true_all.append(b["RUL"].numpy())
        unit_all.append(unit_ids.numpy())
    return (np.concatenate(rul_pred_all),
            np.concatenate(rul_true_all),
            np.concatenate(unit_all))


def metrics(p: np.ndarray, t: np.ndarray) -> dict:
    err = p - t
    return {
        "RMSE":      float(np.sqrt((err ** 2).mean())),
        "MAE":       float(np.abs(err).mean()),
        "bias":      float(err.mean()),
    }


def low_rul_metrics(p: np.ndarray, t: np.ndarray, rul_cap: int = 30) -> dict:
    mask = t < rul_cap
    if not mask.any():
        return {"low_RUL_RMSE": float("nan"), "low_RUL_bias": float("nan"),
                "low_RUL_overest_frac": float("nan"), "n_low_RUL": 0}
    pp, tt = p[mask], t[mask]
    err = pp - tt
    return {
        "low_RUL_RMSE":     float(np.sqrt((err ** 2).mean())),
        "low_RUL_bias":     float(err.mean()),
        "low_RUL_overest_frac": float((err > 0).mean()),
        "n_low_RUL":        int(mask.sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    run_dir = _find_run_dir(args.run_dir)
    if run_dir is None:
        msg = ("No trained C checkpoint found. Pass --run_dir to point to one explicitly.")
        print(msg)
        (OUT_DIR / "rul_theta_usage_report.md").write_text(
            f"# RUL θ usage (SKIPPED)\n\n{msg}\n", encoding="utf-8")
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

    # Data
    ds02_path = Path(cfg["data"]["hdf5_path"])
    if not ds02_path.is_absolute():
        ds02_path = Path(__file__).parent.parent / ds02_path
    if not ds02_path.exists():
        msg = f"DS02 not at {ds02_path}"
        print(msg)
        (OUT_DIR / "rul_theta_usage_report.md").write_text(
            f"# RUL θ usage (SKIPPED)\n\n{msg}\n", encoding="utf-8")
        return

    base_test = NCMAPSSV3Dataset(ds02_path, split="test", load_in_memory=True)
    wds = NCMAPSSV3WindowedDataset(
        base_test, list(cfg["data"]["test_units"]),
        window_size=cfg["data"]["window_size"], stride=20,   # subsample
        max_samples=20000,   # cap for diagnostic
    )
    loader = DataLoader(wds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    print(f"Test windows after stride+cap: {len(wds):,}")

    # ── Prognostics-head weight norm on θ slots ─────────────────────────
    # The first Linear in PrognosticsHead has input dim = feat + (5 if use_theta) + (1 if aux)
    cfg_m = model.config
    feat_dim = cfg_m.encoder_out_dim + (cfg_m.ops_out_dim if cfg_m.use_ops_encoder else 0)
    # First Linear inside prognostics.net (LayerNorm → Linear → SiLU → Dropout → ...)
    first_linear = None
    for m in model.prognostics.net:
        if isinstance(m, torch.nn.Linear):
            first_linear = m; break
    weight_norm = {n: float("nan") for n in THETA_NAMES}
    if first_linear is not None and cfg_m.use_theta_in_rul:
        # Theta columns start at feat_dim
        for i, n in enumerate(THETA_NAMES):
            col = feat_dim + i
            weight_norm[n] = float(first_linear.weight[:, col].abs().sum().item())
        # Aux column at the end (if present)
        aux_col = feat_dim + 5 if cfg_m.use_aux_in_rul else None
        if aux_col is not None and aux_col < first_linear.weight.shape[1]:
            weight_norm["aux_lpt_flow"] = float(first_linear.weight[:, aux_col].abs().sum().item())

    # ── Perturbation variants ───────────────────────────────────────────
    variants = [
        "real",
        "shuffle_batch",
        "shuffle_within_unit",
        "shuffle_across_units",
        "constant_healthy",
        "constant_lo",
    ]
    rows: list[dict] = []
    for var in variants:
        torch.manual_seed(0)   # deterministic shuffle pattern across variants
        p, t, u = infer_with_theta_override(model, loader, device, (sm, sd, om, od), var)
        m  = metrics(p, t); ml = low_rul_metrics(p, t, rul_cap=30)
        row = {"variant": var, **m, **ml}
        rows.append(row)
        print(f"  {var:24s} RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  "
              f"bias={m['bias']:+.3f}  low_RUL_RMSE={ml['low_RUL_RMSE']:.3f}")
    df = pd.DataFrame(rows)

    real_rmse = df.loc[df["variant"] == "real", "RMSE"].iloc[0]
    df["delta_RMSE_vs_real"] = df["RMSE"] - real_rmse
    df["pct_RMSE_change"]    = (df["RMSE"] - real_rmse) / max(real_rmse, 1e-6) * 100
    df.to_csv(OUT_DIR / "rul_theta_usage.csv", index=False)
    print(f"  saved {OUT_DIR / 'rul_theta_usage.csv'}")

    # ── ∂L_rul / ∂θ via a small training-mode forward ───────────────────
    model.train()
    grad_norm = {n: float("nan") for n in THETA_NAMES}
    # One small batch
    for b in loader:
        sensors_norm = ((b["sensors_imp"].to(device) - sm.to(device)) / sd.to(device))
        ops_norm     = ((b["ops_imp"].to(device)     - om.to(device)) / od.to(device))
        ops_si  = {k: v.to(device) for k, v in b["ops_si_last"].items()}
        sens_si = {k: v.to(device) for k, v in b["sens_si_last"].items()}
        out = model(sensors_norm, ops_norm, ops_si=ops_si, sens_si=sens_si)
        rul_true = b["RUL"].to(device)
        L_rul = ((out["rul"] - rul_true) ** 2).mean()
        g = torch.autograd.grad(L_rul, out["theta_phys"],
                                retain_graph=False, allow_unused=True)[0]
        if g is not None:
            v = g.detach().abs().sum(dim=0)
            for i, n in enumerate(THETA_NAMES):
                grad_norm[n] = float(v[i].item())
        break

    # ── Markdown ────────────────────────────────────────────────────────
    md = f"""# RUL-head θ usage — V3.1b (Task 6)

*Read-only.  ADR-0013.*

* Checkpoint: `{run_dir}`
* Test windows used: {len(wds):,}  (stride 20, cap 20k)

## RUL metrics under θ perturbation

{df_to_md(df, floatfmt='.4f')}

Real-θ baseline RMSE: **{real_rmse:.4f}**

## Decision rule

> If all θ perturbations change RMSE by < 0.5 % **and** prognostics-head
> weight norms / `∂L_rul/∂θ` are near zero, the RUL head is ignoring θ_phys.

| θ | weight norm (first PrognosticsHead Linear) | ∂L_rul/∂θ (batch grad sum) |
|---|---|---|
"""
    for n in THETA_NAMES:
        md += f"| `{n}` | {weight_norm[n]:.4e} | {grad_norm[n]:.4e} |\n"
    if "aux_lpt_flow" in weight_norm:
        md += f"| `aux_lpt_flow` | {weight_norm['aux_lpt_flow']:.4e} | — |\n"

    md += f"""

## Largest |ΔRMSE| across perturbations

* `max |ΔRMSE / RMSE_real|` = **{df['pct_RMSE_change'].abs().max():.4f} %**
* If this is < 0.5 % → RUL head is **not** using θ.

## Plot

See `rul_theta_usage.png` for ΔRMSE bars.
"""
    (OUT_DIR / "rul_theta_usage_report.md").write_text(md, encoding="utf-8")
    print(f"  saved {OUT_DIR / 'rul_theta_usage_report.md'}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["variant"], df["delta_RMSE_vs_real"], color="tab:blue", edgecolor="black")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("ΔRMSE vs real-θ"); ax.set_xlabel("θ perturbation variant")
    ax.set_title("RUL RMSE change under θ perturbations  (closer to 0 = head ignores θ)")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(df["delta_RMSE_vs_real"]):
        ax.annotate(f"{v:+.3f}", (i, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rul_theta_usage.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT_DIR / 'rul_theta_usage.png'}")


if __name__ == "__main__":
    main()
