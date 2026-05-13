"""Diagnostic 2 — loss-gradient pathway to each θ channel.

ADR-0013, Task 3. Read-only.

Measures `∂L_x / ∂θ_phys` separately for each loss component
(L_rul, L_temp, L_aux, L_healthy, L_smooth, L_total).  The critical
question: does L_temp flow gradient into θ_η_hpt / θ_η_lpt at all, or
are those channels gradient-isolated as the closure math predicts?

Uses a SMALL CycleLayerV3 + synthetic batch (or real DS02 batch if the
HDF5 is local).  Pressure / EPR loss is never enabled — V3.1b path only.
"""

from __future__ import annotations

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
import torch.nn as nn  # noqa: E402
import yaml          # noqa: E402

from cyclelayer.losses import CycleLayerV3Loss, V3LossConfig    # noqa: E402
from cyclelayer.models import units                              # noqa: E402

from train_cyclelayer_v3_thermal_aux_smoke import (              # noqa: E402
    build_brayton_from_cfg, build_v3_from_cfg,
    NCMAPSSV3WindowedDataset, _collate,
    fit_sensor_ops_scalers, fit_temp_sigmas_K, fit_lpt_flow_sigma,
)
from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset           # noqa: E402

OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "theta_identifiability"
THETA_NAMES = ["eta_fan", "eta_lpc", "eta_hpc", "eta_hpt", "eta_lpt"]
LOSS_TERMS  = ["L_rul", "L_temp", "L_aux", "L_healthy", "L_smooth", "L_total"]


def df_to_md(df: pd.DataFrame, floatfmt: str = ".3e") -> str:
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


def build_synthetic_batch(B: int = 8, T: int = 50):
    """Cruise-style synthetic batch in *imperial* space (matches dataloader)."""
    sensors_imp = torch.randn(B, T, 14) * 0.0
    # Plausible cruise-like raw imperial values
    sensors_imp[..., 0]  = 555.0 + torch.randn(B, T) * 5.0       # T24 R
    sensors_imp[..., 1]  = 1300.0 + torch.randn(B, T) * 20.0     # T30 R
    sensors_imp[..., 2]  = 1500.0 + torch.randn(B, T) * 30.0     # T48 R
    sensors_imp[..., 3]  = 1100.0 + torch.randn(B, T) * 15.0     # T50 R
    sensors_imp[..., 4]  = 10.0   + torch.randn(B, T) * 0.5      # P15
    sensors_imp[..., 5]  = 8.0    + torch.randn(B, T) * 0.5      # P2
    sensors_imp[..., 6]  = 10.0   + torch.randn(B, T) * 0.5      # P21
    sensors_imp[..., 7]  = 14.0   + torch.randn(B, T) * 1.0      # P24
    sensors_imp[..., 8]  = 200.0  + torch.randn(B, T) * 10.0     # Ps30
    sensors_imp[..., 9]  = 200.0  + torch.randn(B, T) * 10.0     # P40
    sensors_imp[..., 10] = 8.0    + torch.randn(B, T) * 0.5      # P50
    sensors_imp[..., 11] = 2020.0 + torch.randn(B, T) * 30.0     # Nf
    sensors_imp[..., 12] = 8200.0 + torch.randn(B, T) * 100.0    # Nc
    sensors_imp[..., 13] = 2.2    + torch.randn(B, T) * 0.1      # Wf

    ops_imp = torch.zeros(B, T, 4)
    ops_imp[..., 0] = 23000.0 + torch.randn(B, T) * 1000.0       # alt ft
    ops_imp[..., 1] = 0.63    + torch.randn(B, T) * 0.05          # Mach
    ops_imp[..., 2] = 70.0    + torch.randn(B, T) * 10.0          # TRA %
    ops_imp[..., 3] = 470.0   + torch.randn(B, T) * 10.0          # T2 R

    # Last-timestep SI inputs for BraytonEngine
    last = T - 1
    ops_si = {
        "T2_K":  ops_imp[:, last, 3] * units.RANK_TO_K,
        "P2_Pa": sensors_imp[:, last, 5] * units.PSIA_TO_PA,
        "alt_m": ops_imp[:, last, 0] * units.FT_TO_M,
        "mach":  ops_imp[:, last, 1].clone(),
    }
    sens_si = {
        "Nf_rpm": sensors_imp[:, last, 11].clone(),
        "Nc_rpm": sensors_imp[:, last, 12].clone(),
        "Wf_kgs": sensors_imp[:, last, 13] * units.PPS_TO_KGS,
    }
    targets_K_last = {
        "T24_K": sensors_imp[:, last, 0] * units.RANK_TO_K,
        "T30_K": sensors_imp[:, last, 1] * units.RANK_TO_K,
        "T50_K": sensors_imp[:, last, 3] * units.RANK_TO_K,
    }
    # Health GT (small negative deltas — synthetic "moderately degraded")
    lpt_flow_true = torch.full((B,), -0.02)
    # Mix of low + high RUL so L_healthy_prior has samples to act on
    RUL = torch.cat([torch.full((B // 2,), 30.0),
                     torch.full((B - B // 2,), 90.0)])
    return (sensors_imp, ops_imp, ops_si, sens_si,
            targets_K_last, lpt_flow_true, RUL)


def compute_grad_norms(
    model, loss_fn, batch_data, scalers_t,
) -> dict[str, dict[str, float]]:
    """Return {loss_term: {theta_name: grad_norm}} for each isolated loss term."""
    sensors_imp, ops_imp, ops_si, sens_si, targets_K_last, lpt_flow_true, RUL = batch_data
    sm, sd, om, od = scalers_t

    out = {term: {n: float("nan") for n in THETA_NAMES} for term in LOSS_TERMS}
    out["theta_at_eval"] = {n: float("nan") for n in THETA_NAMES}

    def fresh_forward():
        # Re-zero grads on encoder/heads so theta_phys grad collected here is
        # the chain-rule gradient from THIS specific loss term.
        for p in model.parameters():
            p.grad = None
        sensors_norm = (sensors_imp - sm) / sd
        ops_norm     = (ops_imp     - om) / od
        return model(sensors_norm, ops_norm, ops_si=ops_si, sens_si=sens_si)

    cfg = loss_fn.config
    # Trick: replace lambda weights to isolate each component, but the loss
    # module's component "rul", "temp", etc. require lambda > 0.  We'll
    # construct the COMPONENT directly without going through the lambda
    # weighting, then sum and take grad w.r.t. theta_phys.
    out_dict = fresh_forward()
    temp_preds = {k: out_dict["brayton"]["sensors_pred_si"][k]
                  for k in ("T24_K", "T30_K", "T50_K")}
    # Record theta_phys mean at this operating point
    theta_eval = out_dict["theta_phys"].detach().mean(dim=0)
    out["theta_at_eval"] = {n: float(theta_eval[i].item())
                            for i, n in enumerate(THETA_NAMES)}

    # Build each component term standalone
    L_rul     = loss_fn._rul_loss(out_dict["rul"], RUL)
    L_temp    = loss_fn._temp_loss(temp_preds, targets_K_last)
    L_aux     = loss_fn._aux_loss(out_dict["lpt_flow_pred"], lpt_flow_true)
    L_healthy = loss_fn._healthy_prior(out_dict["theta_phys"], RUL)
    L_smooth  = loss_fn._smooth_loss(out_dict["theta_phys"])
    L_total = (cfg.lambda_rul * L_rul + cfg.lambda_temp * L_temp
               + cfg.lambda_aux * L_aux + cfg.lambda_healthy * L_healthy
               + cfg.lambda_smooth * L_smooth)

    # For per-term grads we re-do retain_graph
    def grads_to_theta(scalar_loss, retain: bool = True) -> torch.Tensor:
        try:
            g = torch.autograd.grad(scalar_loss, out_dict["theta_phys"],
                                    retain_graph=retain, allow_unused=True)[0]
        except RuntimeError:
            # Loss is a no-grad scalar (e.g. L_smooth on (B, 5) theta with no
            # time axis returns torch.zeros((), device=...) with no grad_fn).
            return torch.zeros_like(out_dict["theta_phys"])
        return g if g is not None else torch.zeros_like(out_dict["theta_phys"])

    g_rul     = grads_to_theta(L_rul)
    g_temp    = grads_to_theta(L_temp)
    g_aux     = grads_to_theta(L_aux)
    g_healthy = grads_to_theta(L_healthy)
    g_smooth  = grads_to_theta(L_smooth)
    g_total   = grads_to_theta(L_total, retain=False)

    def norm_per_theta(g: torch.Tensor) -> dict[str, float]:
        # g shape: (B, 5) — sum over B for the channel-norm
        v = g.detach().abs().sum(dim=0)
        return {n: float(v[i].item()) for i, n in enumerate(THETA_NAMES)}

    out["L_rul"]     = norm_per_theta(g_rul)
    out["L_temp"]    = norm_per_theta(g_temp)
    out["L_aux"]     = norm_per_theta(g_aux)
    out["L_healthy"] = norm_per_theta(g_healthy)
    out["L_smooth"]  = norm_per_theta(g_smooth)
    out["L_total"]   = norm_per_theta(g_total)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(__file__).parent.parent / "configs" / "cyclelayer_v3_thermal_aux.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Build a tiny CycleLayerV3.  We override window_size to match the synthetic batch.
    brayton = build_brayton_from_cfg(cfg["model"]["brayton_engine"])
    model = build_v3_from_cfg(cfg["model"], brayton)
    model.train()   # active masking + theta computation paths

    # Build the loss with sane sigma defaults (we don't need fitted values for grad-norms)
    sigma_T = {"T24": 10.0, "T30": 30.0, "T50": 25.0}
    loss_cfg = V3LossConfig(
        lambda_rul=float(cfg["loss"]["lambda_rul"]),
        lambda_temp=float(cfg["loss"]["lambda_temp"]),
        lambda_aux=float(cfg["loss"]["lambda_aux"]),
        lambda_healthy=float(cfg["loss"]["lambda_healthy"]),
        lambda_smooth=float(cfg["loss"]["lambda_smooth"]),
        mse_weight=float(cfg["loss"]["rul"]["mse_weight"]),
        asymmetry=float(cfg["loss"]["rul"]["asymmetry"]),
        temp_sensors=list(cfg["loss"]["temp_sensors"]),
        sigma_temp_K=sigma_T,
        sigma_lpt_flow=0.02,
        healthy_rul_threshold=float(cfg["loss"]["healthy_rul_threshold"]),
        use_pressure_loss=False,
        use_epr_loss=False,
    )
    loss_fn = CycleLayerV3Loss(loss_cfg)

    # Trivial unit-σ scalers (gradient direction is independent of scale)
    sm = torch.zeros(14); sd = torch.ones(14)
    om = torch.zeros(4);  od = torch.ones(4)

    print("Running synthetic batch (B=16, T=50) ...")
    torch.manual_seed(0)
    batch_data = build_synthetic_batch(B=16, T=50)
    grads_synth = compute_grad_norms(model, loss_fn, batch_data, (sm, sd, om, od))

    rows: list[dict] = []
    for term in LOSS_TERMS:
        for n in THETA_NAMES:
            rows.append({
                "source":    "synthetic",
                "loss_term": term,
                "theta":     n,
                "grad_norm": grads_synth[term][n],
            })

    # ── Optional: real DS02 batch if local file exists ──────────────────
    ds02_path = Path(cfg["data"]["hdf5_path"])
    if not ds02_path.is_absolute():
        ds02_path = Path(__file__).parent.parent / ds02_path
    if ds02_path.exists():
        print(f"Running DS02 batch from {ds02_path.name} ...")
        base = NCMAPSSV3Dataset(ds02_path, split="dev", load_in_memory=True)
        scalers = fit_sensor_ops_scalers(base, [2, 5])
        sigma_T_real = fit_temp_sigmas_K(base, [2, 5])
        loss_cfg_real = V3LossConfig(**{**loss_cfg.__dict__,
                                        "sigma_temp_K": sigma_T_real,
                                        "sigma_lpt_flow": fit_lpt_flow_sigma(base, [2, 5])})
        loss_fn_real = CycleLayerV3Loss(loss_cfg_real)
        wds = NCMAPSSV3WindowedDataset(base, [2, 5], window_size=50, stride=20, max_samples=16)
        from torch.utils.data import DataLoader
        loader = DataLoader(wds, batch_size=16, shuffle=True, num_workers=0,
                            collate_fn=_collate)
        for b in loader:
            sm_r = torch.from_numpy(scalers["sensor_mean"]).float()
            sd_r = torch.from_numpy(scalers["sensor_std"]).float()
            om_r = torch.from_numpy(scalers["ops_mean"]).float()
            od_r = torch.from_numpy(scalers["ops_std"]).float()
            batch_real = (
                b["sensors_imp"], b["ops_imp"],
                b["ops_si_last"], b["sens_si_last"], b["targets_K_last"],
                b["health_gt_last"]["LPT_flow_mod"], b["RUL"],
            )
            grads_real = compute_grad_norms(model, loss_fn_real, batch_real,
                                            (sm_r, sd_r, om_r, od_r))
            for term in LOSS_TERMS:
                for n in THETA_NAMES:
                    rows.append({
                        "source":    "DS02",
                        "loss_term": term,
                        "theta":     n,
                        "grad_norm": grads_real[term][n],
                    })
            break   # single batch is enough — gradient pattern is structural
    else:
        print(f"DS02 not at {ds02_path} — synthetic only.")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "loss_gradient_paths.csv", index=False)
    print(f"  saved {OUT_DIR / 'loss_gradient_paths.csv'}")

    # Pivot for plot/markdown: theta × loss_term (avg across sources)
    pivot = df.pivot_table(index="loss_term", columns="theta",
                           values="grad_norm", aggfunc="mean")
    pivot = pivot.reindex(LOSS_TERMS)[THETA_NAMES]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    mat = pivot.to_numpy()
    # log scale (clip zeros for visualisation)
    safe = np.where(mat > 0, mat, np.nan)
    log_mat = np.log10(safe)
    im = ax.imshow(log_mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(THETA_NAMES, rotation=20)
    ax.set_yticks(range(len(LOSS_TERMS))); ax.set_yticklabels(LOSS_TERMS)
    ax.set_title("log10(|grad| of loss term w.r.t. theta channel)  — mean over sources")
    fig.colorbar(im, ax=ax, label="log10(|grad|)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v == 0:
                ax.text(j, i, "0", ha="center", va="center",
                        color="red", fontweight="bold", fontsize=9)
            elif np.isfinite(v):
                ax.text(j, i, f"{v:.2e}", ha="center", va="center",
                        color="white" if log_mat[i, j] < log_mat[np.isfinite(log_mat)].mean() else "black",
                        fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_gradient_paths.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT_DIR / 'loss_gradient_paths.png'}")

    # ── Verdict per theta ───────────────────────────────────────────────
    # Critical question: is ∂L_temp/∂θ_η_hpt and ∂L_temp/∂θ_η_lpt effectively zero?
    L_temp_row = pivot.loc["L_temp"]
    max_temp_grad = float(L_temp_row.abs().max())
    threshold_rel = 1e-3
    threshold_abs = 1e-8

    verdict = {}
    for n in THETA_NAMES:
        v = float(L_temp_row[n])
        rel = v / max_temp_grad if max_temp_grad > 0 else 0
        verdict[n] = {
            "L_temp_grad_norm":    v,
            "rel_to_max":          rel,
            "below_1e-3_relative": rel < threshold_rel,
            "below_1e-8_absolute": v < threshold_abs,
        }

    md = f"""# Loss gradient paths — V3.1b θ identifiability (Task 3)

*Read-only diagnostic. ADR-0013.*

## Setup

* `CycleLayerV3` built from `configs/cyclelayer_v3_thermal_aux.yaml`
* Synthetic batch (B=16, T=50) + (if local DS02 available) one real DS02 batch
* Each loss component (L_rul / L_temp / L_aux / L_healthy / L_smooth / L_total)
  computed STANDALONE and `∂/∂θ_phys` taken via `torch.autograd.grad`
* Reported value = `Σ_batch |grad|` per (loss_term, θ) cell

## Mean |grad| of each loss term w.r.t. each θ

{df_to_md(pivot.reset_index(), floatfmt='.3e')}

## V3.1b's critical question: ∂L_temp / ∂θ

| θ | grad |
|---|---|
"""
    for n in THETA_NAMES:
        v = verdict[n]
        flag = []
        if v["below_1e-3_relative"]: flag.append("rel < 1e-3")
        if v["below_1e-8_absolute"]: flag.append("abs < 1e-8")
        tag = "; ".join(flag) if flag else "active gradient"
        md += f"| `{n}` | {v['L_temp_grad_norm']:.3e}  ({tag}) |\n"

    md += f"""
## Decision rule (ADR-0013)

> If `L_temp` gradient norms for `θ_η_hpt` and `θ_η_lpt` are ~0 while
> compressor θ gradients are nonzero, then HPT/LPT efficiency θ are not
> identifiable from the current temperature loss.

* `θ_η_hpt`  L_temp grad : **{verdict['eta_hpt']['L_temp_grad_norm']:.3e}**
* `θ_η_lpt`  L_temp grad : **{verdict['eta_lpt']['L_temp_grad_norm']:.3e}**

The other components (L_aux, L_healthy, L_smooth) may still provide a
weak gradient, but they do not constrain the *physical mapping* between
θ and HPT/LPT efficiency.

See `loss_gradient_paths.png` for the full grid.
"""
    (OUT_DIR / "loss_gradient_paths_report.md").write_text(md, encoding="utf-8")
    print(f"  saved {OUT_DIR / 'loss_gradient_paths_report.md'}")

    # CLI summary
    print("\n=== L_temp gradient per theta ===")
    for n in THETA_NAMES:
        v = verdict[n]
        print(f"  {n:8s}  L_temp grad = {v['L_temp_grad_norm']:.3e}  "
              f"rel={v['rel_to_max']:.2e}  "
              f"{'(zero pathway)' if v['below_1e-8_absolute'] else ''}")


if __name__ == "__main__":
    main()
