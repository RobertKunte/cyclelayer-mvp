"""Diagnostic — simple ML baselines for DS02 RUL (ADR-0014, Step 6).

Trains classical feature-based regressors on DS02 train units only,
evaluates on test units [11, 14, 15].  Tests H7 (task harder than
current model capacity): if Ridge / Random-Forest / Hist-Gradient-Boost
crush V3.1b on the same test units, V3.1b's architecture/training is
not competitive.

Features used (per window or per row): summary statistics over a
sliding window of width = `data.window_size` from the YAML, computed
over the 14 X_s sensors + 4 W ops columns.  No θ_phys, no Brayton —
purely classical ML.

Strict constraints:
  * train ONLY on train units (no test leakage).
  * stride for windows matches the V3.1b training/eval strides.
  * test windows are sampled with `stride_eval` (1).
  * outputs RMSE / MAE / R² / Pearson / slope per model on test units.

Outputs under `<session>/simple_baselines/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _v31b_diag_helpers import (   # noqa: E402
    REPO_ROOT, df_to_md, get_session_dir, rul_metrics,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset   # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        REPO_ROOT / "configs" / "cyclelayer_v3_thermal_aux.yaml"))
    ap.add_argument("--max_train_windows", type=int, default=80000,
                    help="Cap train windows (sklearn memory).")
    ap.add_argument("--max_test_windows",  type=int, default=40000)
    ap.add_argument("--models", default="ridge,hgb,rf",
                    help="Comma-separated subset of {ridge, hgb, rf}.")
    ap.add_argument("--rf_n_estimators", type=int, default=200)
    ap.add_argument("--rf_max_depth",    type=int, default=20)
    ap.add_argument("--hgb_max_iter",    type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def make_window_features(W: np.ndarray, X: np.ndarray, A: np.ndarray,
                         Y: np.ndarray, units: list[int],
                         window_size: int, stride: int,
                         max_windows: int | None,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a (M, n_features) feature matrix per window from train units.

    Features per window (length T=window_size):
        for each of the 18 channels (14 X_s + 4 W):
            mean, std, min, max, last_value, slope  → 6 stats
        plus the cycle index at the last row (1 feature).
    Total = 18*6 + 1 = 109 features.

    Returns (X_feat, y, unit_ids, cycles_last).
    """
    unit_arr = A[:, 0].astype(np.int64)
    cyc_arr  = A[:, 1].astype(np.int64)
    feats_list: list[np.ndarray] = []
    y_list: list[float] = []
    uid_list: list[int] = []
    cyc_list: list[int] = []

    # Concatenate ops + sensors → (N, 18)
    chans = np.concatenate([X, W], axis=1)
    n_ch = chans.shape[1]

    candidate_windows: list[tuple[int, int]] = []
    for uid in units:
        m = unit_arr == uid
        idxs = np.nonzero(m)[0]
        if len(idxs) < window_size:
            continue
        start, end = int(idxs[0]), int(idxs[-1]) + 1
        for ws in range(start, end - window_size + 1, stride):
            candidate_windows.append((ws, ws + window_size))

    if max_windows is not None and len(candidate_windows) > max_windows:
        idxs = rng.choice(len(candidate_windows), size=max_windows, replace=False)
        candidate_windows = [candidate_windows[int(i)] for i in idxs]

    ramp = np.arange(window_size, dtype=np.float32)
    ramp_c = ramp - ramp.mean()
    ramp_norm2 = float((ramp_c ** 2).sum()) + 1e-9

    for (s, e) in candidate_windows:
        block = chans[s:e]              # (T, 18)
        mean_ = block.mean(axis=0)
        std_  = block.std(axis=0)
        min_  = block.min(axis=0)
        max_  = block.max(axis=0)
        last_ = block[-1]
        # OLS slope per channel using ramp_c
        slope_ = (ramp_c[:, None] * (block - mean_)).sum(axis=0) / ramp_norm2
        feat = np.concatenate([mean_, std_, min_, max_, last_, slope_],
                              axis=0)
        feat = np.concatenate([feat, np.array([cyc_arr[e - 1]], dtype=np.float32)])
        feats_list.append(feat.astype(np.float32))
        y_list.append(float(Y[e - 1]))
        uid_list.append(int(unit_arr[e - 1]))
        cyc_list.append(int(cyc_arr[e - 1]))

    if not feats_list:
        return (np.zeros((0, 6 * n_ch + 1), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64))
    return (np.stack(feats_list, axis=0),
            np.asarray(y_list, dtype=np.float32),
            np.asarray(uid_list, dtype=np.int64),
            np.asarray(cyc_list, dtype=np.int64))


def main() -> None:
    args = parse_args()
    session_dir = get_session_dir()
    out_dir = session_dir / "simple_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"session dir: {session_dir}")
    print(f"out dir:     {out_dir}")

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    train_units = list(data_cfg["train_units"])
    test_units  = list(data_cfg["test_units"])
    window_size = int(data_cfg["window_size"])
    stride_train = int(data_cfg["stride_train"])
    stride_eval  = int(data_cfg["stride_eval"])

    ds02_path = Path(data_cfg["hdf5_path"])
    if not ds02_path.is_absolute():
        ds02_path = REPO_ROOT / ds02_path
    if not ds02_path.exists():
        msg = f"DS02 not at {ds02_path}"
        print(msg)
        (out_dir / "report.md").write_text(f"# Simple baselines (SKIPPED)\n\n{msg}\n",
                                            encoding="utf-8")
        return

    print(f"DS02 at {ds02_path}")
    base_dev  = NCMAPSSV3Dataset(ds02_path, split="dev",  load_in_memory=True)
    base_test = NCMAPSSV3Dataset(ds02_path, split="test", load_in_memory=True)

    rng = np.random.default_rng(args.seed)
    print(f"\nBuilding train feature matrix (window={window_size}, stride={stride_train})...")
    t0 = time.time()
    Xtr, ytr, uid_tr, cyc_tr = make_window_features(
        W=base_dev._W, X=base_dev._X, A=base_dev._A, Y=base_dev._Y,
        units=train_units, window_size=window_size, stride=stride_train,
        max_windows=args.max_train_windows, rng=rng,
    )
    print(f"  train: X={Xtr.shape}  y={ytr.shape}  units={sorted(np.unique(uid_tr).tolist())}  "
          f"({time.time() - t0:.1f}s)")

    print(f"\nBuilding test feature matrix (window={window_size}, stride={stride_eval})...")
    t0 = time.time()
    Xte, yte, uid_te, cyc_te = make_window_features(
        W=base_test._W, X=base_test._X, A=base_test._A, Y=base_test._Y,
        units=test_units, window_size=window_size, stride=stride_eval,
        max_windows=args.max_test_windows, rng=rng,
    )
    print(f"  test:  X={Xte.shape}  y={yte.shape}  units={sorted(np.unique(uid_te).tolist())}  "
          f"({time.time() - t0:.1f}s)")

    if Xtr.size == 0 or Xte.size == 0:
        msg = "Empty feature matrix — cannot fit baselines."
        print(msg)
        (out_dir / "report.md").write_text(f"# Simple baselines (FAIL)\n\n{msg}\n",
                                            encoding="utf-8")
        return

    # Save train/test arrays for reproducibility
    np.savez(out_dir / "features.npz",
             Xtr=Xtr, ytr=ytr, uid_tr=uid_tr, cyc_tr=cyc_tr,
             Xte=Xte, yte=yte, uid_te=uid_te, cyc_te=cyc_te)

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    results: list[dict] = []
    preds_by_model: dict[str, np.ndarray] = {}

    # Standardise features for Ridge
    mu = Xtr.mean(axis=0); sd = np.clip(Xtr.std(axis=0), 1e-6, None)
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd

    if "ridge" in models:
        try:
            from sklearn.linear_model import Ridge
            print("\n[ridge] fitting...")
            t0 = time.time()
            m = Ridge(alpha=1.0, random_state=args.seed)
            m.fit(Xtr_n, ytr)
            p_te = np.clip(m.predict(Xte_n), 0.0, 99.0)
            p_tr = np.clip(m.predict(Xtr_n), 0.0, 99.0)
            met_te = rul_metrics(p_te, yte)
            met_tr = rul_metrics(p_tr, ytr)
            print(f"  ridge test: RMSE={met_te['RMSE']:.3f}  R2={met_te['R2']:.3f}  "
                  f"({time.time() - t0:.1f}s)")
            results.append({"model": "ridge",
                            "test_RMSE": met_te["RMSE"], "test_R2": met_te["R2"],
                            "test_Pearson": met_te["Pearson"],
                            "test_slope": met_te["slope"],
                            "test_std_ratio": met_te["std_ratio"],
                            "train_RMSE": met_tr["RMSE"], "train_R2": met_tr["R2"]})
            preds_by_model["ridge"] = p_te
        except Exception as exc:
            print(f"  ridge SKIPPED: {exc}")
            results.append({"model": "ridge", "error": str(exc)})

    if "hgb" in models:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            print("\n[hgb] fitting...")
            t0 = time.time()
            m = HistGradientBoostingRegressor(
                max_iter=args.hgb_max_iter,
                learning_rate=0.05,
                max_depth=None,
                random_state=args.seed,
            )
            m.fit(Xtr, ytr)
            p_te = np.clip(m.predict(Xte), 0.0, 99.0)
            p_tr = np.clip(m.predict(Xtr), 0.0, 99.0)
            met_te = rul_metrics(p_te, yte)
            met_tr = rul_metrics(p_tr, ytr)
            print(f"  hgb test: RMSE={met_te['RMSE']:.3f}  R2={met_te['R2']:.3f}  "
                  f"({time.time() - t0:.1f}s)")
            results.append({"model": "hgb",
                            "test_RMSE": met_te["RMSE"], "test_R2": met_te["R2"],
                            "test_Pearson": met_te["Pearson"],
                            "test_slope": met_te["slope"],
                            "test_std_ratio": met_te["std_ratio"],
                            "train_RMSE": met_tr["RMSE"], "train_R2": met_tr["R2"]})
            preds_by_model["hgb"] = p_te
        except Exception as exc:
            print(f"  hgb SKIPPED: {exc}")
            results.append({"model": "hgb", "error": str(exc)})

    if "rf" in models:
        try:
            from sklearn.ensemble import RandomForestRegressor
            print("\n[rf] fitting...")
            t0 = time.time()
            m = RandomForestRegressor(
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                n_jobs=-1, random_state=args.seed,
            )
            m.fit(Xtr, ytr)
            p_te = np.clip(m.predict(Xte), 0.0, 99.0)
            p_tr = np.clip(m.predict(Xtr), 0.0, 99.0)
            met_te = rul_metrics(p_te, yte)
            met_tr = rul_metrics(p_tr, ytr)
            print(f"  rf  test: RMSE={met_te['RMSE']:.3f}  R2={met_te['R2']:.3f}  "
                  f"({time.time() - t0:.1f}s)")
            results.append({"model": "rf",
                            "test_RMSE": met_te["RMSE"], "test_R2": met_te["R2"],
                            "test_Pearson": met_te["Pearson"],
                            "test_slope": met_te["slope"],
                            "test_std_ratio": met_te["std_ratio"],
                            "train_RMSE": met_tr["RMSE"], "train_R2": met_tr["R2"]})
            preds_by_model["rf"] = p_te
        except Exception as exc:
            print(f"  rf SKIPPED: {exc}")
            results.append({"model": "rf", "error": str(exc)})

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "metrics.csv", index=False)
    if preds_by_model:
        np.savez(out_dir / "test_predictions.npz",
                 yte=yte, uid_te=uid_te, cyc_te=cyc_te,
                 **{f"pred_{k}": v for k, v in preds_by_model.items()})

    # Verdict: a baseline meaningfully beats V3.1b C (RMSE 21.4) if its
    # test RMSE < 18 and R² > 0.2 — adjust thresholds as needed.
    best = None
    for r in results:
        if "test_RMSE" not in r:
            continue
        if best is None or r["test_RMSE"] < best["test_RMSE"]:
            best = r
    md_verdict = ""
    if best is not None:
        beats = (best["test_RMSE"] < 18.0) and (best.get("test_R2") or -1) > 0.2
        md_verdict = (
            f"Best simple baseline: **{best['model']}** with "
            f"test RMSE = {best['test_RMSE']:.3f}, R² = "
            f"{best.get('test_R2', float('nan')):.3f}.  "
            + ("**Beats V3.1b clearly** — V3.1b architecture/training is "
               "not competitive against classical ML on DS02."
               if beats else
               "Does NOT clearly beat V3.1b — both classical and V3.1b "
               "struggle, suggesting H7 (task hardness) or shared data "
               "issues.")
        )

    md = f"""# Simple ML baselines on DS02 — ADR-0014 Step 6

*Read-only.  Train on train_units only ({train_units}); test on
{test_units} (held-out).*

## Setup

* window_size:   {window_size}
* stride_train:  {stride_train}
* stride_eval:   {stride_eval}
* train rows ({len(train_units)} units): {Xtr.shape[0]:,}  features={Xtr.shape[1]}
* test  rows ({len(test_units)} units):  {Xte.shape[0]:,}
* feature recipe: mean / std / min / max / last / OLS-slope per channel
  (14 X_s + 4 W = 18 channels × 6 stats = 108), + cycle_last = 109 features.

## Results (test units)

{df_to_md(df, floatfmt='.4f')}

## Reference

* V3.1b C (production checkpoint): test RMSE ≈ 21.41.

{md_verdict}
"""
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({
        "n_train": int(Xtr.shape[0]),
        "n_test":  int(Xte.shape[0]),
        "models":  [r["model"] for r in results],
        "results": results,
        "v31b_C_reference_test_RMSE": 21.41,
        "best": best,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
