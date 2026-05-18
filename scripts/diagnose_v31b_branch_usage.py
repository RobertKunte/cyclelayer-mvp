"""Diagnostic — RUL-head branch usage ablations (ADR-0014, Step 7).

Loads the latest V3.1b C checkpoint, runs inference on test units, and
**at the prognostics-head input** swaps out each upstream signal:

  * baseline         — unmodified
  * zero_theta       — `theta_phys` replaced by zeros
  * shuffle_theta    — `theta_phys` permuted across batch (preserves
                       marginal distribution, destroys per-sample link)
  * zero_aux         — `lpt_flow_pred` replaced by 0
  * shuffle_aux      — permuted across batch
  * zero_h_sens      — `h_sens` (sensor-encoder output) replaced by 0
  * shuffle_h_sens   — permuted across batch
  * zero_z_ops       — `z_ops` (ops-encoder output) replaced by 0
  * shuffle_z_ops    — permuted across batch
  * zero_features    — both `h_sens` and `z_ops` zeroed (RUL must rely
                       only on theta + aux → exposes whether features
                       are actually contributing)

For every ablation, reports `mean(RUL_pred)`, std, RMSE, R² vs true,
ΔRMSE vs baseline.  A branch is **unused** if zeroing/shuffling it
leaves RUL essentially unchanged (|ΔRMSE| < 0.5 cycles).

Outputs under `<session>/branch_usage/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from _v31b_diag_helpers import (   # noqa: E402
    REPO_ROOT, df_to_md, find_c_run_dir, get_session_dir, rul_metrics,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset   # noqa: E402

from train_cyclelayer_v3_thermal_aux_smoke import (   # noqa: E402
    NCMAPSSV3WindowedDataset, _collate,
    build_brayton_from_cfg, build_v3_from_cfg,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


@torch.no_grad()
def predict_with_ablations(model, loader, device, scalers, seed: int = 0):
    """Run inference; for each batch produce `len(ABLATIONS)` predicted
    RUL arrays at the prognostics-head level."""
    sm, sd, om, od = scalers
    sm = sm.to(device); sd = sd.to(device)
    om = om.to(device); od = od.to(device)

    cfg = model.config
    use_theta = bool(cfg.use_theta_in_rul)
    use_aux   = bool(cfg.use_aux_in_rul)
    detach_theta = bool(cfg.detach_theta_to_rul)
    detach_aux   = bool(cfg.detach_aux_to_rul)

    g = torch.Generator(device="cpu").manual_seed(seed)

    ABLATIONS = [
        "baseline",
        "zero_theta", "shuffle_theta",
        "zero_aux",   "shuffle_aux",
        "zero_h_sens", "shuffle_h_sens",
        "zero_z_ops",  "shuffle_z_ops",
        "zero_features",
    ]
    preds_by_name: dict[str, list[np.ndarray]] = {n: [] for n in ABLATIONS}
    true_all, unit_all, cyc_all = [], [], []
    n_dim_theta = None
    n_dim_aux = None

    for b in loader:
        sn = (b["sensors_imp"].to(device) - sm) / sd
        on = (b["ops_imp"].to(device)     - om) / od
        ops_si  = {k: v.to(device) for k, v in b["ops_si_last"].items()}
        sens_si = {k: v.to(device) for k, v in b["sens_si_last"].items()}
        true_all.append(b["RUL"].numpy())
        unit_all.append(b["unit_id"].numpy())
        if "aux" in b and "cycle" in b["aux"]:
            cyc_all.append(b["aux"]["cycle"].numpy())
        else:
            cyc_all.append(np.zeros(len(b["RUL"]), dtype=np.int64))

        out = model(sn, on, ops_si=ops_si, sens_si=sens_si)
        h_sens     = out["h_sens"]
        z_ops      = out["z_ops"]
        theta_phys = out["theta_phys"]
        aux        = out["lpt_flow_pred"]
        n_dim_theta = theta_phys.shape[1]
        n_dim_aux   = 1

        B = h_sens.shape[0]

        def _rul_from(parts: list[torch.Tensor]) -> torch.Tensor:
            return model.prognostics(torch.cat(parts, dim=-1))

        def _features(h, z):
            return h if z is None else torch.cat([h, z], dim=-1)

        def _rul_full(h, z, t, a):
            parts: list[torch.Tensor] = [_features(h, z)]
            if use_theta:
                tt = t.detach() if detach_theta else t
                parts.append(tt)
            if use_aux:
                aa = a.detach() if detach_aux else a
                parts.append(aa.unsqueeze(-1) if aa.dim() == 1 else aa)
            return _rul_from(parts)

        # baseline = stored "rul"
        preds_by_name["baseline"].append(out["rul"].cpu().numpy())

        # CPU permutation index, then to device
        perm = torch.randperm(B, generator=g).to(device)

        # zero / shuffle theta
        zt = torch.zeros_like(theta_phys)
        st = theta_phys[perm]
        preds_by_name["zero_theta"].append(
            _rul_full(h_sens, z_ops, zt, aux).cpu().numpy())
        preds_by_name["shuffle_theta"].append(
            _rul_full(h_sens, z_ops, st, aux).cpu().numpy())

        # zero / shuffle aux
        za = torch.zeros_like(aux)
        sa = aux[perm]
        preds_by_name["zero_aux"].append(
            _rul_full(h_sens, z_ops, theta_phys, za).cpu().numpy())
        preds_by_name["shuffle_aux"].append(
            _rul_full(h_sens, z_ops, theta_phys, sa).cpu().numpy())

        # zero / shuffle h_sens
        zh = torch.zeros_like(h_sens)
        sh = h_sens[perm]
        preds_by_name["zero_h_sens"].append(
            _rul_full(zh, z_ops, theta_phys, aux).cpu().numpy())
        preds_by_name["shuffle_h_sens"].append(
            _rul_full(sh, z_ops, theta_phys, aux).cpu().numpy())

        # zero / shuffle z_ops (only when present)
        if z_ops is not None:
            zz = torch.zeros_like(z_ops)
            sz = z_ops[perm]
            preds_by_name["zero_z_ops"].append(
                _rul_full(h_sens, zz, theta_phys, aux).cpu().numpy())
            preds_by_name["shuffle_z_ops"].append(
                _rul_full(h_sens, sz, theta_phys, aux).cpu().numpy())
            preds_by_name["zero_features"].append(
                _rul_full(zh, zz, theta_phys, aux).cpu().numpy())
        else:
            preds_by_name["zero_z_ops"].append(out["rul"].cpu().numpy())
            preds_by_name["shuffle_z_ops"].append(out["rul"].cpu().numpy())
            preds_by_name["zero_features"].append(
                _rul_full(zh, None, theta_phys, aux).cpu().numpy())

    preds_concat = {k: np.concatenate(v) for k, v in preds_by_name.items()}
    true = np.concatenate(true_all)
    units = np.concatenate(unit_all)
    cycles = np.concatenate(cyc_all)
    return preds_concat, true, units, cycles, {
        "theta_dim": int(n_dim_theta) if n_dim_theta is not None else 0,
        "aux_dim":   int(n_dim_aux)   if n_dim_aux   is not None else 0,
    }


def main() -> None:
    args = parse_args()
    session_dir = get_session_dir()
    out_dir = session_dir / "branch_usage"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"session dir: {session_dir}")
    print(f"out dir:     {out_dir}")

    run_dir = find_c_run_dir(args.run_dir)
    if run_dir is None:
        msg = "No C run dir found.  Pass --run_dir."
        print(msg)
        (out_dir / "report.md").write_text(f"# Branch usage (SKIPPED)\n\n{msg}\n",
                                            encoding="utf-8")
        return
    print(f"run dir: {run_dir}")

    cfg_path = REPO_ROOT / "configs" / "cyclelayer_v3_thermal_aux.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    test_units = list(data_cfg["test_units"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brayton = build_brayton_from_cfg(cfg["model"]["brayton_engine"])
    model = build_v3_from_cfg(cfg["model"], brayton).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    if "scalers" in ckpt:
        sm = torch.tensor(ckpt["scalers"]["sensor_mean"]).float()
        sd = torch.tensor(ckpt["scalers"]["sensor_std"]).float()
        om = torch.tensor(ckpt["scalers"]["ops_mean"]).float()
        od = torch.tensor(ckpt["scalers"]["ops_std"]).float()
    else:
        sn = np.load(run_dir / "sensor_scaler.npz"); on = np.load(run_dir / "ops_scaler.npz")
        sm = torch.from_numpy(sn["mean"]).float(); sd = torch.from_numpy(sn["std"]).float()
        om = torch.from_numpy(on["mean"]).float(); od = torch.from_numpy(on["std"]).float()

    ds02_path = Path(data_cfg["hdf5_path"])
    if not ds02_path.is_absolute():
        ds02_path = REPO_ROOT / ds02_path
    if not ds02_path.exists():
        msg = f"DS02 not at {ds02_path}"
        print(msg)
        (out_dir / "report.md").write_text(f"# Branch usage (SKIPPED)\n\n{msg}\n",
                                            encoding="utf-8")
        return
    base_test = NCMAPSSV3Dataset(ds02_path, split="test", load_in_memory=True)
    test_ds = NCMAPSSV3WindowedDataset(
        base_test, test_units,
        window_size=data_cfg["window_size"], stride=data_cfg["stride_eval"],
        max_samples=args.max_test_samples,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    print(f"test windows: {len(test_ds):,}")

    preds, true, units, cycles, dims = predict_with_ablations(
        model, loader, device, (sm, sd, om, od), seed=args.seed)
    base_metrics = rul_metrics(preds["baseline"], true)
    base_rmse = base_metrics["RMSE"]

    rows = []
    for name, p in preds.items():
        m = rul_metrics(p, true)
        rows.append({
            "ablation":      name,
            "RMSE":          m["RMSE"],
            "MAE":           m["MAE"],
            "R2":            m["R2"],
            "Pearson":       m["Pearson"],
            "slope":         m["slope"],
            "std_ratio":     m["std_ratio"],
            "pred_mean":     float(np.mean(p)),
            "pred_std":      float(np.std(p)),
            "dRMSE_vs_base": m["RMSE"] - base_rmse,
            "branch_unused": bool(abs(m["RMSE"] - base_rmse) < 0.5
                                  and name != "baseline"),
        })
    # Ensure baseline is first
    rows.sort(key=lambda r: (0 if r["ablation"] == "baseline" else 1,
                              -r["dRMSE_vs_base"]))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)

    # Cohesive verdict
    unused = [r["ablation"] for r in rows
              if r["branch_unused"] and r["ablation"] != "baseline"]
    used = [r["ablation"] for r in rows
            if (not r["branch_unused"]) and r["ablation"] != "baseline"]

    md = f"""# Branch-usage ablation — ADR-0014 Step 7

*Read-only.  Production checkpoint loaded; **NO** training; weights
unchanged.  Ablations performed at the prognostics-head input.*

* Checkpoint: `{run_dir}`
* Test windows: {len(test_ds):,}
* theta dim:  {dims['theta_dim']}
* aux   dim:  {dims['aux_dim']}
* Detach-to-RUL flags: theta={bool(model.config.detach_theta_to_rul)}  """\
f"""aux={bool(model.config.detach_aux_to_rul)}

## Metrics

{df_to_md(df, floatfmt='.4f')}

## Branches that appear **unused** by the RUL head (|ΔRMSE| < 0.5)

{', '.join(unused) if unused else '_(none — every branch contributes)_'}

## Branches that contribute (|ΔRMSE| ≥ 0.5)

{', '.join(used) if used else '_(none — RUL prediction is independent of upstream signals!)_'}

## Interpretation

* If `zero_theta` and `shuffle_theta` give ΔRMSE ≈ 0:  V3.1b's θ_phys
  is not used by the RUL head → consistent with ADR-0013 H1
  (unidentifiable) and explains the C ≈ D RMSE gap.
* If `zero_aux` ≈ 0 too:  AuxHead also unused.
* If `zero_h_sens` ≈ 0 and `zero_z_ops` ≈ 0 too:  RUL is **entirely
  bias-driven** — predicting a near-constant — confirming mean-collapse
  (H1, H6).
* If only `zero_features` ≈ 0 but individual branches do shift RUL:
  features cancel each other / encoder relies on noise.
"""
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({
        "run_dir": str(run_dir),
        "baseline_RMSE": base_rmse,
        "rows": rows,
        "unused_branches": unused,
        "used_branches": used,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {out_dir / 'report.md'}")
    print("VERDICT: unused branches:", unused or "(none)")


if __name__ == "__main__":
    main()
