"""Diagnostic — tiny-overfit smoke test for V3.1b RUL head (ADR-0014, Step 5).

Falsifies H5/H6/H7: if the model CAN'T overfit a tiny train subset, the
implementation (output scaling / loss / optimizer / architecture) is
broken; if it CAN overfit tiny but collapses on full DS02, the
generalisation / training-budget / sampling is the problem.

Train a fresh `CycleLayerV3` (random init, same architecture as the
production C run) on N ∈ {256, 1024, 4096} train-unit windows for a
fixed number of epochs.  Report final train-set RMSE, R², std_ratio,
slope on the SAME windows it was trained on.

Read-only with respect to the production C checkpoint — never loaded.
Never touches test units [11, 14, 15].

Outputs under `<session>/overfit_tiny/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _v31b_diag_helpers import (   # noqa: E402
    REPO_ROOT, df_to_md, get_session_dir, rul_metrics,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset   # noqa: E402
from cyclelayer.losses import CycleLayerV3Loss, V3LossConfig   # noqa: E402

from train_cyclelayer_v3_thermal_aux_smoke import (   # noqa: E402
    NCMAPSSV3WindowedDataset, _collate,
    build_brayton_from_cfg, build_v3_from_cfg,
    fit_sensor_ops_scalers, fit_temp_sigmas_K, fit_lpt_flow_sigma,
)
from torch.utils.data import DataLoader, Subset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        REPO_ROOT / "configs" / "cyclelayer_v3_thermal_aux.yaml"))
    ap.add_argument("--sizes", default="256,1024,4096",
                    help="Comma-separated tiny-train sizes.")
    ap.add_argument("--epochs", type=int, default=200,
                    help="Epochs per size (small so tiny set can overfit).")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(n_train: int, epochs: int, batch_size: int, lr: float,
            cfg: dict, base_dev: NCMAPSSV3Dataset, train_units: list[int],
            sensor_mean: torch.Tensor, sensor_std: torch.Tensor,
            ops_mean: torch.Tensor, ops_std: torch.Tensor,
            sigma_T: dict, sigma_lpt: float,
            device: torch.device, seed: int) -> dict:
    set_seed(seed)
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg_yaml = cfg["loss"]

    # Build a windowed dataset on train units, sample first n_train windows
    full_train = NCMAPSSV3WindowedDataset(
        base_dev, train_units, window_size=data_cfg["window_size"],
        stride=data_cfg["stride_train"], max_samples=None,
    )
    n_avail = len(full_train)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n_avail, size=min(n_train, n_avail), replace=False)
    tiny = Subset(full_train, sorted(int(x) for x in idxs))
    print(f"\n--- TINY OVERFIT N={n_train} (effective {len(tiny)}) ---")

    loader = DataLoader(tiny, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=_collate)

    brayton = build_brayton_from_cfg(model_cfg["brayton_engine"])
    model = build_v3_from_cfg(model_cfg, brayton).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}")

    loss_fn = CycleLayerV3Loss(V3LossConfig(
        lambda_rul=float(loss_cfg_yaml["lambda_rul"]),
        lambda_temp=float(loss_cfg_yaml["lambda_temp"]),
        lambda_aux=float(loss_cfg_yaml["lambda_aux"]),
        lambda_healthy=float(loss_cfg_yaml["lambda_healthy"]),
        lambda_smooth=float(loss_cfg_yaml["lambda_smooth"]),
        mse_weight=float(loss_cfg_yaml["rul"]["mse_weight"]),
        asymmetry=float(loss_cfg_yaml["rul"]["asymmetry"]),
        temp_sensors=list(loss_cfg_yaml["temp_sensors"]),
        sigma_temp_K=sigma_T,
        sigma_lpt_flow=sigma_lpt,
        healthy_rul_threshold=float(loss_cfg_yaml["healthy_rul_threshold"]),
        use_pressure_loss=False, use_epr_loss=False,
    ))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    sm, sd = sensor_mean.to(device), sensor_std.to(device)
    om, od = ops_mean.to(device),    ops_std.to(device)

    history: list[dict] = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0; n_b = 0
        rul_preds, rul_trues = [], []
        for b in loader:
            sn = (b["sensors_imp"].to(device) - sm) / sd
            on = (b["ops_imp"].to(device)     - om) / od
            ops_si  = {k: v.to(device) for k, v in b["ops_si_last"].items()}
            sens_si = {k: v.to(device) for k, v in b["sens_si_last"].items()}
            temp_true = {k: v.to(device) for k, v in b["targets_K_last"].items()}
            lpt_true = b["health_gt_last"]["LPT_flow_mod"].to(device)
            rul_true = b["RUL"].to(device)

            opt.zero_grad(set_to_none=True)
            out = model(sn, on, ops_si=ops_si, sens_si=sens_si)
            temp_preds = {
                "T24_K": out["brayton"]["sensors_pred_si"]["T24_K"],
                "T30_K": out["brayton"]["sensors_pred_si"]["T30_K"],
                "T50_K": out["brayton"]["sensors_pred_si"]["T50_K"],
            }
            loss_tot, _ = loss_fn(
                rul_pred=out["rul"], rul_true=rul_true,
                theta_phys=out["theta_phys"],
                lpt_flow_pred=out["lpt_flow_pred"],
                lpt_flow_true=lpt_true,
                temp_preds_K=temp_preds, temp_true_K=temp_true,
            )
            loss_tot.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()
            total += float(loss_tot.item()); n_b += 1
            rul_preds.append(out["rul"].detach().cpu().numpy())
            rul_trues.append(rul_true.detach().cpu().numpy())
        avg = total / max(n_b, 1)
        p = np.concatenate(rul_preds); t = np.concatenate(rul_trues)
        rmse = float(np.sqrt(((p - t) ** 2).mean()))
        history.append({"epoch": ep, "loss": avg, "train_rmse": rmse})
        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(f"  epoch {ep:>4}/{epochs}  loss={avg:.4f}  train_rmse={rmse:.3f}")

    # Final pass on the same tiny set (no shuffle, no dropout)
    model.eval()
    pred_all, true_all = [], []
    with torch.no_grad():
        for b in loader:
            sn = (b["sensors_imp"].to(device) - sm) / sd
            on = (b["ops_imp"].to(device)     - om) / od
            ops_si  = {k: v.to(device) for k, v in b["ops_si_last"].items()}
            sens_si = {k: v.to(device) for k, v in b["sens_si_last"].items()}
            out = model(sn, on, ops_si=ops_si, sens_si=sens_si)
            pred_all.append(out["rul"].cpu().numpy())
            true_all.append(b["RUL"].numpy())
    pred = np.concatenate(pred_all); true = np.concatenate(true_all)
    m = rul_metrics(pred, true)
    elapsed = time.time() - t0
    print(f"  final: RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}  "
          f"std_ratio={m['std_ratio']:.3f}  slope={m['slope']:.3f}  "
          f"({elapsed:.1f}s)")
    return {
        "n_train_requested": int(n_train),
        "n_train_effective": int(len(tiny)),
        "epochs": int(epochs),
        "n_params": int(n_params),
        "elapsed_s": float(elapsed),
        "metrics_train": m,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    session_dir = get_session_dir()
    out_dir = session_dir / "overfit_tiny"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"session dir: {session_dir}")
    print(f"out dir:     {out_dir}")

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    train_units = list(data_cfg["train_units"])

    device = torch.device(args.device
                          or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds02_path = Path(data_cfg["hdf5_path"])
    if not ds02_path.is_absolute():
        ds02_path = REPO_ROOT / ds02_path
    if not ds02_path.exists():
        msg = f"DS02 not at {ds02_path}"
        print(msg)
        (out_dir / "report.md").write_text(f"# Tiny overfit (SKIPPED)\n\n{msg}\n",
                                            encoding="utf-8")
        return

    base_dev = NCMAPSSV3Dataset(ds02_path, split="dev", load_in_memory=True)

    scalers = fit_sensor_ops_scalers(base_dev, train_units)
    sigma_T  = fit_temp_sigmas_K(base_dev, train_units)
    sigma_lpt = fit_lpt_flow_sigma(base_dev, train_units)
    sensor_mean = torch.from_numpy(scalers["sensor_mean"]).float()
    sensor_std  = torch.from_numpy(scalers["sensor_std"]).float()
    ops_mean    = torch.from_numpy(scalers["ops_mean"]).float()
    ops_std     = torch.from_numpy(scalers["ops_std"]).float()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    results: list[dict] = []
    for n in sizes:
        results.append(run_one(
            n_train=n, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, cfg=cfg, base_dev=base_dev, train_units=train_units,
            sensor_mean=sensor_mean, sensor_std=sensor_std,
            ops_mean=ops_mean, ops_std=ops_std,
            sigma_T=sigma_T, sigma_lpt=sigma_lpt,
            device=device, seed=args.seed,
        ))

    # Pass/fail rule per ADR-0014:
    #   each tiny set: train RMSE < 5 (strict overfit) AND R2 > 0.7 → PASS
    #   else FAIL → implementation problem (H5 unlikely if even tiny fails)
    rows = []
    overall_pass = True
    for r in results:
        m = r["metrics_train"]
        passes = (m["RMSE"] < 5.0) and (m["R2"] is not None) and (m["R2"] > 0.7)
        if not passes:
            overall_pass = False
        rows.append({
            "n_train":   r["n_train_effective"],
            "epochs":    r["epochs"],
            "RMSE":      m["RMSE"],
            "MAE":       m["MAE"],
            "R2":        m["R2"],
            "Pearson":   m["Pearson"],
            "slope":     m["slope"],
            "std_ratio": m["std_ratio"],
            "elapsed_s": r["elapsed_s"],
            "passes_overfit": passes,
        })

    # ── Output ───────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)

    (out_dir / "history.json").write_text(json.dumps(
        [{"n_train": r["n_train_effective"], "history": r["history"]}
         for r in results], indent=2), encoding="utf-8")

    status = "PASS" if overall_pass else "FAIL"
    md = f"""# Tiny-overfit diagnostic — {status}  (ADR-0014, Step 5)

*Read-only.  No production checkpoint loaded.  Train units only
({train_units}); test units `{data_cfg['test_units']}` NEVER touched.*

## Setup

* sizes:      {sizes}
* epochs:     {args.epochs}
* batch size: {args.batch_size}
* lr:         {args.lr}
* optimiser:  AdamW (wd=1e-5)
* model:      fresh `CycleLayerV3` (production architecture, random init)
* seed:       {args.seed}
* device:     {device}

## Results (train-set metrics on the SAME windows used for training)

{df_to_md(df, floatfmt='.4f')}

## Pass rule

A size **passes** if `train RMSE < 5` **and** `R² > 0.7` on the tiny set.

## Interpretation

* All sizes pass → overfit is achievable → implementation healthy →
  collapse on full DS02 is a **generalisation / training-budget /
  sampling problem (H5)**.  Next: longer training, EOL-balanced
  sampling, possibly larger encoder.
* Any size fails (especially N=256) → **implementation problem (H6/H7)**:
  output scaling, loss, optimizer, or architecture.  Investigate
  PrognosticsHead bias init, Softplus clamp, learning rate, gradient
  flow into θ_phys / encoder.

VERDICT: **{status}**
"""
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({
        "status": status,
        "overall_pass": bool(overall_pass),
        "rows": rows,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {out_dir / 'report.md'}")
    print(f"VERDICT: {status}")


if __name__ == "__main__":
    main()
