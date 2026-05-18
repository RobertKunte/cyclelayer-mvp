"""Diagnostic — DS02 target / window alignment audit (ADR-0014, Step 4).

Read-only.  First gate in the ADR-0014 decision logic: if any check
here FAILS, all other RUL-collapse diagnostics are invalid.

Verifies:
  * test_units (DS02 "test" split) and dev_units (DS02 "dev" split) are
    disjoint.
  * configured train_units, val_units, test_units are pairwise disjoint
    and live in the correct HDF5 split.
  * within every windowed sample produced by
    `NCMAPSSV3WindowedDataset`, all rows share one unit_id (no unit
    mixing across window boundaries).
  * the y_true returned by `__getitem__` equals the RUL at the *last*
    HDF5 row of the window (not the first / middle).
  * per-unit cycle-vs-RUL monotonicity: RUL decreases (non-strict) as
    cycle increases on every unit.
  * raw RUL distribution sanity: range, integer-ness, units, no double
    normalization, no negative values.
  * test units have a last-row RUL of 0 (or very small).
  * random sample of N=10 windows printed with: unit_id, cycle range,
    y_true, first/last sensor values, HDF5 row indices.

Outputs under `<session>/target_alignment/`:
  * `target_alignment_report.md`
  * `summary.json` (status: PASS / FAIL with reasons)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _v31b_diag_helpers import (   # noqa: E402
    REPO_ROOT, df_to_md, get_session_dir,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset   # noqa: E402

from train_cyclelayer_v3_thermal_aux_smoke import (   # noqa: E402
    NCMAPSSV3WindowedDataset,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        REPO_ROOT / "configs" / "cyclelayer_v3_thermal_aux.yaml"))
    ap.add_argument("--n_sample_windows", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ── Check helpers ────────────────────────────────────────────────────


def check_split_disjoint(train_u: list[int], val_u: list[int],
                         test_u: list[int]) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    pairs = [("train", train_u, "val", val_u),
             ("train", train_u, "test", test_u),
             ("val",   val_u,   "test", test_u)]
    ok = True
    for (an, a, bn, b) in pairs:
        inter = sorted(set(a) & set(b))
        if inter:
            ok = False
            msgs.append(f"FAIL: {an} ∩ {bn} = {inter}")
        else:
            msgs.append(f"OK: {an} ∩ {bn} = ∅")
    return ok, msgs


def check_units_in_correct_hdf5_split(
    base_dev: NCMAPSSV3Dataset, base_test: NCMAPSSV3Dataset,
    train_u: list[int], val_u: list[int], test_u: list[int],
) -> tuple[bool, list[str], list[int], list[int]]:
    dev_uids  = sorted(int(u) for u in base_dev.unit_ids.tolist())
    test_uids = sorted(int(u) for u in base_test.unit_ids.tolist())
    msgs = [f"HDF5 dev split unit_ids:  {dev_uids}",
            f"HDF5 test split unit_ids: {test_uids}",
            f"configured train_units:   {sorted(train_u)}",
            f"configured val_units:     {sorted(val_u)}",
            f"configured test_units:    {sorted(test_u)}"]
    ok = True
    # train / val must live in DEV
    for kind, units in (("train", train_u), ("val", val_u)):
        miss = [u for u in units if u not in dev_uids]
        if miss:
            ok = False
            msgs.append(f"FAIL: {kind}_units {miss} not in HDF5 dev split")
        else:
            msgs.append(f"OK: all {kind}_units present in HDF5 dev split")
    # test must live in TEST
    miss = [u for u in test_u if u not in test_uids]
    if miss:
        ok = False
        msgs.append(f"FAIL: test_units {miss} not in HDF5 test split")
    else:
        msgs.append("OK: all test_units present in HDF5 test split")
    # dev and test units must be globally disjoint (no leak across HDF5)
    leak = sorted(set(dev_uids) & set(test_uids))
    if leak:
        ok = False
        msgs.append(f"FAIL: HDF5 dev ∩ test unit_ids = {leak}")
    else:
        msgs.append("OK: HDF5 dev ∩ test unit_ids = ∅ (no DS02 split leak)")
    return ok, msgs, dev_uids, test_uids


def check_no_unit_mixing(ds: NCMAPSSV3WindowedDataset,
                         n_check: int = 1000) -> tuple[bool, list[str], int]:
    """For up to n_check sampled windows, assert that all rows inside the
    window share one unit_id."""
    rng = np.random.default_rng(0)
    n_wins = len(ds)
    take = min(n_check, n_wins)
    idxs = rng.choice(n_wins, size=take, replace=False) if n_wins else []
    bad: list[tuple[int, int]] = []
    unit_arr = ds._A[:, 0].astype(np.int64)
    for i in idxs:
        s, e = ds.indices[int(i)]
        u = unit_arr[s:e]
        if not (u == u[0]).all():
            bad.append((int(i), int(u[0])))
    msgs = [f"sampled {take} of {n_wins} windows"]
    if bad:
        msgs.append(f"FAIL: {len(bad)} window(s) cross unit boundaries — "
                    f"first 3: {bad[:3]}")
        return False, msgs, len(bad)
    msgs.append("OK: every sampled window contains a single unit_id")
    return True, msgs, 0


def check_target_at_window_end(ds: NCMAPSSV3WindowedDataset,
                               n_check: int = 200) -> tuple[bool, list[str], int]:
    """Item RUL must equal Y[end-1], NOT Y[start]."""
    rng = np.random.default_rng(1)
    take = min(n_check, len(ds))
    idxs = rng.choice(len(ds), size=take, replace=False) if len(ds) else []
    bad_end = 0; bad_start = 0
    for i in idxs:
        item = ds[int(i)]
        y_item = float(item["RUL"].item())
        s, e = ds.indices[int(i)]
        y_end_h5   = float(ds._Y[e - 1])
        y_start_h5 = float(ds._Y[s])
        if not np.isclose(y_item, y_end_h5):
            bad_end += 1
        if not np.isclose(y_item, y_start_h5) and (y_end_h5 != y_start_h5):
            # only count "matches start instead of end" when they differ
            if np.isclose(y_item, y_start_h5):
                bad_start += 1
    msgs = [f"sampled {take} windows",
            f"y_item != y_end_h5  count: {bad_end}",
            f"y_item == y_start_h5 but not y_end_h5 count: {bad_start}"]
    if bad_end == 0:
        msgs.append("OK: target equals last-row RUL (window endpoint)")
        return True, msgs, 0
    if bad_start > 0:
        msgs.append("FAIL: target matches window START instead of END")
    else:
        msgs.append("FAIL: target equals neither start nor end of window")
    return False, msgs, bad_end


def check_per_unit_rul_monotonic(
    base: NCMAPSSV3Dataset, units: list[int],
) -> tuple[bool, list[str], list[dict]]:
    """Within every unit, RUL should be non-increasing as cycle increases.
    A small fraction of equal-RUL plateaus is fine; we flag if a unit
    has *increasing* steps."""
    rows: list[dict] = []
    A = base._A; Y = base._Y
    unit_arr = A[:, 0].astype(np.int64)
    cyc_arr  = A[:, 1].astype(np.int64)
    ok = True
    for uid in sorted(units):
        m = unit_arr == uid
        if not m.any():
            rows.append({"unit": uid, "n": 0, "rul_start": float("nan"),
                         "rul_end": float("nan"), "cycle_min": int(-1),
                         "cycle_max": int(-1), "n_increasing_steps": 0,
                         "monotonic_decreasing": True})
            continue
        order = np.argsort(cyc_arr[m])
        y_seq = Y[m][order].astype(np.float64)
        c_seq = cyc_arr[m][order].astype(np.int64)
        diffs = np.diff(y_seq)
        n_inc = int((diffs > 0).sum())
        if n_inc > 0:
            ok = False
        rows.append({
            "unit": int(uid),
            "n":    int(m.sum()),
            "cycle_min": int(c_seq.min()),
            "cycle_max": int(c_seq.max()),
            "rul_start": float(y_seq[0]),
            "rul_end":   float(y_seq[-1]),
            "n_increasing_steps":  n_inc,
            "monotonic_decreasing": bool(n_inc == 0),
        })
    msgs = [
        ("OK: per-unit RUL is monotonically non-increasing"
         if ok else "FAIL: at least one unit has increasing RUL steps"),
    ]
    return ok, msgs, rows


def check_rul_range(base: NCMAPSSV3Dataset, name: str,
                    expected_max: float = 99.0) -> tuple[bool, list[str], dict]:
    Y = np.asarray(base._Y).astype(np.float64).ravel()
    info = {
        "split":  name,
        "n":      int(Y.size),
        "min":    float(Y.min()),
        "max":    float(Y.max()),
        "mean":   float(Y.mean()),
        "median": float(np.median(Y)),
        "std":    float(Y.std()),
        "is_integer": bool(np.allclose(Y, np.round(Y))),
    }
    msgs: list[str] = [
        f"{name}: n={info['n']:,}  min={info['min']:.3f}  max={info['max']:.3f}  "
        f"mean={info['mean']:.3f}  is_integer={info['is_integer']}",
    ]
    ok = True
    if info["min"] < -0.01:
        ok = False
        msgs.append(f"FAIL: {name} has negative RUL values")
    if info["max"] > 200:
        ok = False
        msgs.append(f"FAIL: {name} max RUL ({info['max']}) > 200 — "
                    "possibly raw / un-clamped")
    if info["max"] <= 1.5:
        ok = False
        msgs.append(f"FAIL: {name} max RUL ({info['max']}) <= 1.5 — "
                    "possibly double-normalized")
    return ok, msgs, info


def check_test_units_reach_eol(
    base_test: NCMAPSSV3Dataset, test_u: list[int],
) -> tuple[bool, list[str], list[dict]]:
    """Each test unit's last-cycle row should have RUL ≈ 0 (it ran to EOL)."""
    A = base_test._A; Y = base_test._Y
    unit_arr = A[:, 0].astype(np.int64)
    cyc_arr  = A[:, 1].astype(np.int64)
    rows: list[dict] = []
    ok = True
    for uid in sorted(test_u):
        m = unit_arr == uid
        if not m.any():
            ok = False
            rows.append({"unit": uid, "n": 0, "rul_at_last_cycle": float("nan"),
                         "last_cycle": -1})
            continue
        order = np.argsort(cyc_arr[m])
        y_last = float(Y[m][order][-1])
        c_last = int(cyc_arr[m][order][-1])
        rows.append({
            "unit": int(uid), "n": int(m.sum()),
            "last_cycle": c_last, "rul_at_last_cycle": y_last,
        })
        if y_last > 2.0:
            ok = False
    msgs = [("OK: every test unit ends with RUL ≤ 2"
             if ok else "FAIL: some test unit does not reach RUL ≈ 0")]
    return ok, msgs, rows


def sample_windows(ds: NCMAPSSV3WindowedDataset, n: int = 10,
                   seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    if not len(ds):
        return []
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    samples: list[dict] = []
    A = ds._A
    for i in sorted(int(x) for x in idxs):
        s, e = ds.indices[i]
        item = ds[i]
        sens = item["sensors_imp"].numpy()    # (T, 14)
        samples.append({
            "win_idx":     int(i),
            "h5_start":    int(s),
            "h5_end_excl": int(e),
            "len":         int(e - s),
            "unit_id":     int(A[e - 1, 0]),
            "cycle_first": int(A[s, 1]),
            "cycle_last":  int(A[e - 1, 1]),
            "Fc_first":    int(A[s, 2]),
            "Fc_last":     int(A[e - 1, 2]),
            "hs_first":    int(A[s, 3]),
            "hs_last":     int(A[e - 1, 3]),
            "y_true":      float(item["RUL"].item()),
            "y_h5_start":  float(ds._Y[s]),
            "y_h5_end":    float(ds._Y[e - 1]),
            "sens_first_T24_R": float(sens[0, 0]),
            "sens_last_T24_R":  float(sens[-1, 0]),
            "sens_first_T50_R": float(sens[0, 3]),
            "sens_last_T50_R":  float(sens[-1, 3]),
        })
    return samples


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    train_u = list(data_cfg["train_units"])
    val_u   = list(data_cfg["val_units"])
    test_u  = list(data_cfg["test_units"])
    window_size = int(data_cfg["window_size"])
    stride_eval = int(data_cfg["stride_eval"])
    max_rul_cfg = 99.0   # V3.1b convention (DS02 capped at 99)

    session_dir = get_session_dir()
    out_dir = session_dir / "target_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"session dir: {session_dir}")
    print(f"out dir:     {out_dir}")

    ds02_path = Path(data_cfg["hdf5_path"])
    if not ds02_path.is_absolute():
        ds02_path = REPO_ROOT / ds02_path
    if not ds02_path.exists():
        msg = f"DS02 not at {ds02_path}"
        print(msg)
        (out_dir / "target_alignment_report.md").write_text(
            f"# Target alignment (SKIPPED)\n\n{msg}\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"status": "SKIPPED", "reason": msg}, indent=2),
            encoding="utf-8")
        return

    print(f"DS02 at {ds02_path}")
    base_dev  = NCMAPSSV3Dataset(ds02_path, split="dev",  load_in_memory=True)
    base_test = NCMAPSSV3Dataset(ds02_path, split="test", load_in_memory=True)

    overall_pass = True
    results: dict[str, dict] = {}

    # 1) configured-split pairwise disjoint
    ok, msgs = check_split_disjoint(train_u, val_u, test_u)
    overall_pass &= ok
    results["split_disjoint"] = {"pass": ok, "details": msgs}
    print("\n[1] Configured-split disjointness")
    for m in msgs: print("   ", m)

    # 2) configured units in correct HDF5 split
    ok, msgs, dev_uids, test_uids = check_units_in_correct_hdf5_split(
        base_dev, base_test, train_u, val_u, test_u)
    overall_pass &= ok
    results["units_in_correct_split"] = {
        "pass": ok, "details": msgs,
        "hdf5_dev_unit_ids": dev_uids, "hdf5_test_unit_ids": test_uids,
    }
    print("\n[2] Configured units live in correct HDF5 split")
    for m in msgs: print("   ", m)

    # 3) RUL range / dtype sanity per split
    print("\n[3] RUL range / dtype sanity")
    rul_info: dict[str, dict] = {}
    for name, base in (("dev", base_dev), ("test", base_test)):
        ok, msgs, info = check_rul_range(base, name, expected_max=max_rul_cfg)
        overall_pass &= ok
        rul_info[name] = info
        for m in msgs: print("   ", m)
    results["rul_range"] = rul_info

    # 4) per-unit monotonicity (dev + test)
    print("\n[4] Per-unit RUL monotonicity")
    ok_dev, msgs_dev, mono_rows_dev = check_per_unit_rul_monotonic(
        base_dev, sorted(set(train_u) | set(val_u)))
    overall_pass &= ok_dev
    print("    [dev units]")
    for m in msgs_dev: print("     ", m)
    ok_test, msgs_test, mono_rows_test = check_per_unit_rul_monotonic(
        base_test, test_u)
    overall_pass &= ok_test
    print("    [test units]")
    for m in msgs_test: print("     ", m)
    results["monotonic_decreasing"] = {
        "pass_dev":  ok_dev,
        "pass_test": ok_test,
        "rows_dev":  mono_rows_dev,
        "rows_test": mono_rows_test,
    }

    # 5) test units reach EOL
    print("\n[5] Test units reach RUL ≈ 0 at last cycle")
    ok, msgs, eol_rows = check_test_units_reach_eol(base_test, test_u)
    overall_pass &= ok
    for m in msgs: print("   ", m)
    for r in eol_rows:
        print(f"      unit {r['unit']:>2}: n={r['n']:,}  "
              f"last_cycle={r['last_cycle']}  "
              f"rul_at_last_cycle={r['rul_at_last_cycle']:.3f}")
    results["test_units_eol"] = {"pass": ok, "rows": eol_rows}

    # 6) Build windowed datasets — needed for window-level checks
    print("\n[6] Build windowed datasets (window_size={}, stride_eval={})"
          .format(window_size, stride_eval))
    train_ds = NCMAPSSV3WindowedDataset(
        base_dev, train_u, window_size=window_size,
        stride=int(data_cfg["stride_train"]),
    )
    test_ds = NCMAPSSV3WindowedDataset(
        base_test, test_u, window_size=window_size,
        stride=stride_eval,
    )
    print(f"    train windows: {len(train_ds):,}")
    print(f"    test  windows: {len(test_ds):,}")
    results["window_counts"] = {
        "train": int(len(train_ds)), "test": int(len(test_ds)),
    }

    # 7) No unit mixing inside windows
    print("\n[7] No unit mixing inside windows")
    for tag, ds in (("train", train_ds), ("test", test_ds)):
        ok, msgs, n_bad = check_no_unit_mixing(ds, n_check=1000)
        overall_pass &= ok
        print(f"    [{tag}]")
        for m in msgs: print("     ", m)
        results.setdefault("no_unit_mixing", {})[tag] = {
            "pass": ok, "n_bad": n_bad, "details": msgs,
        }

    # 8) Target equals window endpoint
    print("\n[8] Target equals window endpoint (Y[end-1])")
    for tag, ds in (("train", train_ds), ("test", test_ds)):
        ok, msgs, n_bad = check_target_at_window_end(ds, n_check=200)
        overall_pass &= ok
        print(f"    [{tag}]")
        for m in msgs: print("     ", m)
        results.setdefault("target_at_endpoint", {})[tag] = {
            "pass": ok, "n_bad": n_bad, "details": msgs,
        }

    # 9) Random sample dump
    print(f"\n[9] Random sample of {args.n_sample_windows} test windows")
    samples = sample_windows(test_ds, n=args.n_sample_windows, seed=args.seed)
    for s in samples:
        print(f"    win={s['win_idx']:>6}  h5={s['h5_start']}-{s['h5_end_excl']} "
              f"unit={s['unit_id']:>2}  cyc=[{s['cycle_first']},{s['cycle_last']}] "
              f"y_true={s['y_true']:.2f}  y_h5_end={s['y_h5_end']:.2f} "
              f"T24_R(last)={s['sens_last_T24_R']:.1f}")
    results["sample_windows"] = samples

    # ── Markdown report ──────────────────────────────────────────────
    status = "PASS" if overall_pass else "FAIL"
    md_parts: list[str] = [
        f"# Target alignment audit — {status} (ADR-0014, Step 4)\n",
        "*Read-only.  No model changes.*\n",
        f"* Config: `{args.config}`",
        f"* DS02 file: `{ds02_path}`",
        f"* window_size = {window_size}, stride_train = "
        f"{data_cfg['stride_train']}, stride_eval = {stride_eval}",
        f"* configured train_units = {sorted(train_u)}",
        f"* configured val_units   = {sorted(val_u)}",
        f"* configured test_units  = {sorted(test_u)}",
        "",
        "## 1. Configured-split disjointness",
        ""]
    md_parts += [f"- {m}" for m in results["split_disjoint"]["details"]]
    md_parts += [
        "",
        "## 2. Configured units in correct HDF5 split",
        f"* HDF5 dev unit_ids:  {results['units_in_correct_split']['hdf5_dev_unit_ids']}",
        f"* HDF5 test unit_ids: {results['units_in_correct_split']['hdf5_test_unit_ids']}",
        ""]
    md_parts += [f"- {m}" for m in results["units_in_correct_split"]["details"]]
    md_parts += ["", "## 3. RUL range / dtype sanity", ""]
    for name, info in rul_info.items():
        md_parts.append(
            f"* **{name}**: n={info['n']:,}  min={info['min']:.3f}  "
            f"max={info['max']:.3f}  mean={info['mean']:.3f}  "
            f"median={info['median']:.3f}  std={info['std']:.3f}  "
            f"is_integer={info['is_integer']}")

    # Monotonicity tables
    import pandas as pd  # local
    mono_df_dev  = pd.DataFrame(mono_rows_dev)
    mono_df_test = pd.DataFrame(mono_rows_test)
    md_parts += [
        "",
        "## 4. Per-unit RUL monotonicity",
        "",
        "### Dev (train + val units)",
        df_to_md(mono_df_dev, floatfmt=".3f"),
        "",
        "### Test units",
        df_to_md(mono_df_test, floatfmt=".3f"),
        ""]

    md_parts += [
        "## 5. Test units reach RUL ≈ 0 at last cycle",
        "",
        df_to_md(pd.DataFrame(eol_rows), floatfmt=".3f"),
        ""]

    md_parts += [
        "## 6. Window counts",
        "",
        f"* train windows: {len(train_ds):,}",
        f"* test  windows: {len(test_ds):,}",
        ""]

    md_parts += ["## 7. No unit mixing inside windows", ""]
    for tag, info in results["no_unit_mixing"].items():
        md_parts.append(
            f"* **{tag}**: {'OK' if info['pass'] else 'FAIL'} — "
            f"{info['n_bad']} bad windows out of 1000 sampled")

    md_parts += ["", "## 8. Target equals window endpoint", ""]
    for tag, info in results["target_at_endpoint"].items():
        md_parts.append(
            f"* **{tag}**: {'OK' if info['pass'] else 'FAIL'} — "
            f"{info['n_bad']} mismatches out of 200 sampled")

    md_parts += ["", f"## 9. Random sample of {len(samples)} test windows", ""]
    if samples:
        sdf = pd.DataFrame(samples)
        md_parts.append(df_to_md(sdf, floatfmt=".2f"))

    md_parts += [
        "",
        "## Verdict",
        "",
        f"**{status}** — overall_pass = {overall_pass}.",
        "",
        ("All target / window-alignment checks pass.  Downstream RUL "
         "diagnostics can proceed."
         if overall_pass else
         "At least one alignment check FAILED.  Per ADR-0014, "
         "**downstream RUL collapse diagnostics are invalid until this is "
         "fixed.**"),
        ""]
    (out_dir / "target_alignment_report.md").write_text(
        "\n".join(md_parts), encoding="utf-8")

    (out_dir / "summary.json").write_text(json.dumps({
        "status": status,
        "overall_pass": bool(overall_pass),
        "window_size": window_size,
        "stride_train": int(data_cfg["stride_train"]),
        "stride_eval":  stride_eval,
        "train_units":  sorted(train_u),
        "val_units":    sorted(val_u),
        "test_units":   sorted(test_u),
        "hdf5_dev_unit_ids":  results["units_in_correct_split"]["hdf5_dev_unit_ids"],
        "hdf5_test_unit_ids": results["units_in_correct_split"]["hdf5_test_unit_ids"],
        "n_windows_train": int(len(train_ds)),
        "n_windows_test":  int(len(test_ds)),
        "rul_info": rul_info,
        "checks": {
            "split_disjoint":          results["split_disjoint"]["pass"],
            "units_in_correct_split":  results["units_in_correct_split"]["pass"],
            "rul_range_dev":           rul_info["dev"]["max"]  <= 200 and rul_info["dev"]["min"]  >= -0.01,
            "rul_range_test":          rul_info["test"]["max"] <= 200 and rul_info["test"]["min"] >= -0.01,
            "monotonic_decreasing_dev":  results["monotonic_decreasing"]["pass_dev"],
            "monotonic_decreasing_test": results["monotonic_decreasing"]["pass_test"],
            "test_units_eol":          results["test_units_eol"]["pass"],
            "no_unit_mixing_train":    results["no_unit_mixing"]["train"]["pass"],
            "no_unit_mixing_test":     results["no_unit_mixing"]["test"]["pass"],
            "target_at_endpoint_train": results["target_at_endpoint"]["train"]["pass"],
            "target_at_endpoint_test":  results["target_at_endpoint"]["test"]["pass"],
        },
    }, indent=2, default=float), encoding="utf-8")

    print(f"\nsaved {out_dir / 'target_alignment_report.md'}")
    print(f"saved {out_dir / 'summary.json'}")
    print(f"\nVERDICT: {status}")


if __name__ == "__main__":
    main()
