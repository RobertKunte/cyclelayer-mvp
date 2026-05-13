"""Synthetic theta recovery test (Task 7, ADR-0013).

Controlled inverse problem on SYNTHETIC data only.  No DS02, no NCMAPSS GT,
no parameter tuning on real data.

Procedure:
  1. Pick a "true" theta vector (mildly degraded, e.g. [0.93, 0.92, 0.91, 0.94, 0.95]).
  2. Run BraytonEngine with `theta_true` at a representative operating point
     → record target outputs {T24*, T30*, T50*, P30*, P50*, PR_hpt*, PR_lpt*}.
  3. Initialise `theta_pred` (start near healthy or random) and optimise on
     two target sets:
       Case A: T24/T30/T50 only          (mirrors V3.1b L_temp)
       Case B: T24/T30/T50 + P30/P50     (T + P, the V4-style target)
  4. After convergence, compare `theta_pred` to `theta_true`.

Expected:
  * Case A: compressor theta converge; HPT/LPT theta DO NOT converge.
  * Case B: all theta converge.

If Case A confirms HPT/LPT non-recovery, V3.1b's architectural inability
to identify η_hpt/η_lpt from temperatures is proven *constructively*.
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
import yaml          # noqa: E402

from cyclelayer.models import units  # noqa: E402
from cyclelayer.models.stations import GAMMA_C  # noqa: E402
from cyclelayer.data.ncmapss_v3 import load_userguide_fc02_anchor  # noqa: E402

from train_cyclelayer_v3_thermal_aux_smoke import build_brayton_from_cfg  # noqa: E402

OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "theta_identifiability"
THETA_NAMES = ["eta_fan", "eta_lpc", "eta_hpc", "eta_hpt", "eta_lpt"]


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


def fc02_si_inputs(batch_size: int = 1):
    fc = load_userguide_fc02_anchor()
    Tsl_R = fc["Tsl_F"] + 459.67
    P0_psia = 14.696
    M = fc["XM"]
    ram_T = 1.0 + 0.5 * (GAMMA_C - 1.0) * M ** 2
    ram_P = ram_T ** (GAMMA_C / (GAMMA_C - 1.0))
    T2_R  = Tsl_R * ram_T
    P2_psia = P0_psia * ram_P
    ops_si = {
        "T2_K":  torch.full((batch_size,), T2_R * units.RANK_TO_K),
        "P2_Pa": torch.full((batch_size,), P2_psia * units.PSIA_TO_PA),
        "alt_m": torch.zeros(batch_size),
        "mach":  torch.full((batch_size,), float(M)),
    }
    sens_si = {
        "Nf_rpm": torch.full((batch_size,), float(fc["Nf_rpm"])),
        "Nc_rpm": torch.full((batch_size,), float(fc["Nc_rpm"])),
        "Wf_kgs": torch.full((batch_size,), float(fc["Wf_pps"]) * units.PPS_TO_KGS),
    }
    P2_Pa = P2_psia * units.PSIA_TO_PA
    return ops_si, sens_si, P2_Pa


def engine_outputs(engine, ops_si, sens_si, theta_5: torch.Tensor) -> dict:
    sensors_pred_si, diag = engine(ops_si, sens_si, theta_5)
    return {
        "T24":    sensors_pred_si["T24_K"].squeeze(),
        "T30":    sensors_pred_si["T30_K"].squeeze(),
        "T50":    sensors_pred_si["T50_K"].squeeze(),
        "P30":    sensors_pred_si["P30_Pa"].squeeze(),
        "P50":    diag["P50"].squeeze(),
        "PR_hpt": diag["PR_hpt"].squeeze(),
        "PR_lpt": diag["PR_lpt"].squeeze(),
        "T45":    diag["T45"].squeeze(),
        "P45":    diag["P45"].squeeze(),
    }


def loss_for_targets(out: dict, target: dict, target_keys: list[str],
                     P2_Pa: float) -> torch.Tensor:
    """Sum-squared-relative loss across the chosen target outputs."""
    L = torch.zeros((), dtype=out["T24"].dtype)
    for k in target_keys:
        # Normalise by the magnitude of the target so the gradient is balanced.
        scale = max(abs(target[k].item()) if torch.is_tensor(target[k])
                    else abs(float(target[k])), 1.0)
        diff = (out[k] - target[k]) / scale
        L = L + (diff ** 2)
    return L


def recover(
    engine, ops_si, sens_si, P2_Pa: float,
    theta_true: torch.Tensor, target_keys: list[str],
    init_theta: torch.Tensor, n_steps: int = 2000, lr: float = 1e-2,
) -> tuple[torch.Tensor, list[float], list[torch.Tensor]]:
    """Optimise theta_pred to match `target` on the given keys."""
    # Build target with theta_true
    with torch.no_grad():
        target = engine_outputs(engine, ops_si, sens_si, theta_true.unsqueeze(0))

    # Optimise an UNCONSTRAINED parameter and pass through sigmoid scaling
    # so theta stays in [0.85, 1.00] like in CycleLayerV3.
    lo, hi = 0.85, 1.00

    def to_theta(raw: torch.Tensor) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(raw)

    def to_raw(theta: torch.Tensor) -> torch.Tensor:
        norm = (theta - lo) / (hi - lo)
        norm = torch.clamp(norm, 1e-6, 1.0 - 1e-6)
        return torch.log(norm / (1.0 - norm))

    raw = to_raw(init_theta).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=lr)
    losses: list[float] = []
    theta_history: list[torch.Tensor] = []
    for step in range(n_steps):
        optimizer.zero_grad()
        theta = to_theta(raw)
        out = engine_outputs(engine, ops_si, sens_si, theta.unsqueeze(0))
        L = loss_for_targets(out, target, target_keys, P2_Pa)
        L.backward()
        optimizer.step()
        losses.append(float(L.item()))
        if step % 100 == 0 or step == n_steps - 1:
            theta_history.append(theta.detach().clone())
    theta_final = to_theta(raw).detach()
    return theta_final, losses, theta_history


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(__file__).parent.parent / "configs" / "cyclelayer_v3_thermal_aux.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    engine = build_brayton_from_cfg(cfg["model"]["brayton_engine"])

    ops_si, sens_si, P2_Pa = fc02_si_inputs(batch_size=1)
    theta_true = torch.tensor([0.93, 0.92, 0.91, 0.94, 0.95])

    init_theta = torch.tensor([0.98, 0.98, 0.98, 0.98, 0.98])

    # Case A: T only
    print("[Case A] Recovering theta from T24/T30/T50 only ...")
    theta_A, loss_A, hist_A = recover(
        engine, ops_si, sens_si, P2_Pa,
        theta_true, target_keys=["T24", "T30", "T50"],
        init_theta=init_theta, n_steps=3000, lr=2e-2,
    )
    err_A = (theta_A - theta_true).abs()
    print(f"  final loss = {loss_A[-1]:.3e}")
    print(f"  recovered theta = {theta_A.numpy()}")
    print(f"  true theta      = {theta_true.numpy()}")
    print(f"  |err|       = {err_A.numpy()}")

    # Case B: T + P
    print("[Case B] Recovering theta from T24/T30/T50 + P30/P50 ...")
    theta_B, loss_B, hist_B = recover(
        engine, ops_si, sens_si, P2_Pa,
        theta_true, target_keys=["T24", "T30", "T50", "P30", "P50"],
        init_theta=init_theta, n_steps=3000, lr=2e-2,
    )
    err_B = (theta_B - theta_true).abs()
    print(f"  final loss = {loss_B[-1]:.3e}")
    print(f"  recovered theta = {theta_B.numpy()}")
    print(f"  |err|       = {err_B.numpy()}")

    # Save table
    rows = []
    for i, n in enumerate(THETA_NAMES):
        rows.append({
            "theta":          n,
            "true":           float(theta_true[i].item()),
            "recovered_T":    float(theta_A[i].item()),
            "abs_err_T":      float(err_A[i].item()),
            "recovered_T_P":  float(theta_B[i].item()),
            "abs_err_T_P":    float(err_B[i].item()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "synthetic_recovery.csv", index=False)
    print(f"\n  saved {OUT_DIR / 'synthetic_recovery.csv'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(loss_A, label="Case A: T only");  axes[0].plot(loss_B, label="Case B: T+P")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("loss (sum sq. rel.)")
    axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(True, alpha=0.4)
    axes[0].set_title("Optimisation loss")

    x = np.arange(5); w = 0.25
    axes[1].bar(x - w, theta_true.numpy(), w, label="true theta",          color="tab:gray", edgecolor="black")
    axes[1].bar(x,     theta_A.numpy(),    w, label="recovered (T)",   color="tab:red",  edgecolor="black")
    axes[1].bar(x + w, theta_B.numpy(),    w, label="recovered (T+P)", color="tab:green",edgecolor="black")
    axes[1].set_xticks(x); axes[1].set_xticklabels(THETA_NAMES, rotation=20)
    axes[1].set_ylabel("theta value (factor)")
    axes[1].set_ylim(0.84, 1.01)
    axes[1].axhline(1.0, color="black", lw=0.4)
    axes[1].legend(); axes[1].grid(True, axis="y", alpha=0.4)
    axes[1].set_title("Recovered vs true theta")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "synthetic_recovery.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT_DIR / 'synthetic_recovery.png'}")

    # Decision logic
    threshold = 0.01
    A_recovered = {n: bool(err_A[i].item() < threshold) for i, n in enumerate(THETA_NAMES)}
    B_recovered = {n: bool(err_B[i].item() < threshold) for i, n in enumerate(THETA_NAMES)}

    md = f"""# Synthetic theta recovery — V3.1b (Task 7)

*Read-only.  ADR-0013.  Synthetic data only — no DS02 / no health GT.*

## Setup

* Operating point: FC02 (SLS, M=0.25, TRA=100 %)
* True theta        : `{theta_true.numpy().tolist()}`
* Initial theta     : `{init_theta.numpy().tolist()}` (near healthy)
* Optimiser     : Adam, lr=2e-2, 3000 steps
* theta kept in `[0.85, 1.00]` via sigmoid-scaling of an unconstrained raw param

## Results

{df_to_md(df, floatfmt='.5f')}

## Recovery summary (|theta_recovered − theta_true| < 0.01 ?)

| theta | Case A (T only) | Case B (T+P) |
|---|---|---|
"""
    for n in THETA_NAMES:
        md += f"| `{n}` | {'YES' if A_recovered[n] else '**NO**'} | {'YES' if B_recovered[n] else '**NO**'} |\n"

    md += f"""

## Final loss

* Case A (T only):  {loss_A[-1]:.3e}
* Case B (T+P):     {loss_B[-1]:.3e}

## Plot

`synthetic_recovery.png` — loss curve + bar chart of recovered vs true theta.

## Decision (ADR-0013)

* If `theta_η_hpt` and `theta_η_lpt` are **not recovered** under Case A but **are**
  recovered under Case B, then V3.1b's L_temp cannot identify HPT/LPT
  efficiency: the architecture is the bottleneck, not training data.

* Recovered in A but not B should not happen physically; if it does the
  test setup is malformed (investigate).

This is a CONSTRUCTIVE proof of the identifiability boundary — independent
of N-CMAPSS, RUL-axis time leakage, or training quality.
"""
    (OUT_DIR / "synthetic_recovery_report.md").write_text(md, encoding="utf-8")
    print(f"  saved {OUT_DIR / 'synthetic_recovery_report.md'}")

    print("\n=== Synthetic recovery summary ===")
    print("    theta          true       T-only      T+P    -> Case A | Case B")
    for i, n in enumerate(THETA_NAMES):
        print(f"  {n:8s}  {theta_true[i].item():.4f}   {theta_A[i].item():.4f}    "
              f"{theta_B[i].item():.4f}   "
              f"  {'OK' if A_recovered[n] else 'FAIL':4s}    "
              f"{'OK' if B_recovered[n] else 'FAIL':4s}")


if __name__ == "__main__":
    main()
