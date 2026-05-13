"""Diagnostic — N-CMAPSS DS02 health-parameter sign convention (Task 5).

ADR-0013. Read-only. No DS02 tuning. No model used.

For each DS02 unit, prints early / mid / late-life statistics of the six
efficiency modifiers + four flow modifiers. Determines whether the
documented "degradation" corresponds to GT decreasing toward negative
values (delta convention) or to GT being a factor below 1.

This sets the **expected sign** of `Pearson(θ_phys − 1, GT)` post-hoc.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset  # noqa: E402


OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "theta_identifiability"

HEALTH_COLS = [
    "fan_eff_mod",  "fan_flow_mod",
    "LPC_eff_mod",  "LPC_flow_mod",
    "HPC_eff_mod",  "HPC_flow_mod",
    "HPT_eff_mod",  "HPT_flow_mod",
    "LPT_eff_mod",  "LPT_flow_mod",
]


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


def per_unit_thirds(base: NCMAPSSV3Dataset, units: list[int]) -> pd.DataFrame:
    """For each unit and each health column: early / mid / late statistics."""
    A = base._A; unit_arr = A[:, 0].astype(np.int64)
    T = base._T   # (N, 10)
    Y = base._Y   # (N,) RUL
    rows: list[dict] = []
    for uid in units:
        mask = unit_arr == uid
        if not mask.any():
            continue
        # RUL-decreasing axis (life progresses → RUL decreases). We instead
        # use ROW INDEX (per-unit row order = chronological); high index = late life.
        idxs = np.nonzero(mask)[0]
        n = len(idxs)
        early = idxs[: int(n * 0.05)]
        mid_lo = int(n * 0.45); mid_hi = int(n * 0.55)
        mid = idxs[mid_lo:mid_hi]
        late = idxs[int(n * 0.95):]
        for i, col_name in enumerate(HEALTH_COLS):
            for tag, sl in [("early", early), ("mid", mid), ("late", late)]:
                vals = T[sl, i]
                rows.append({
                    "unit":   uid,
                    "column": col_name,
                    "phase":  tag,
                    "n":      len(vals),
                    "mean":   float(np.mean(vals)),
                    "std":    float(np.std(vals)),
                    "min":    float(np.min(vals)),
                    "max":    float(np.max(vals)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).parent.parent
    ds02_path = repo_root / "data" / "NCMAPSS" / "N-CMAPSS_DS02-006.h5"
    if not ds02_path.exists():
        msg = f"DS02 not found at {ds02_path} — skipping sign-convention check."
        print(msg)
        (OUT_DIR / "health_sign_convention.md").write_text(
            f"# Sign convention (skipped)\n\n{msg}\n", encoding="utf-8"
        )
        return

    print("Loading DS02 dev split ...")
    base_dev = NCMAPSSV3Dataset(ds02_path, split="dev", load_in_memory=True)
    print("Loading DS02 test split ...")
    base_test = NCMAPSSV3Dataset(ds02_path, split="test", load_in_memory=True)

    units_dev  = base_dev.unit_ids.tolist()
    units_test = base_test.unit_ids.tolist()
    print(f"  dev units:  {units_dev}")
    print(f"  test units: {units_test}")

    df_dev  = per_unit_thirds(base_dev,  units_dev)
    df_dev["split"]  = "dev"
    df_test = per_unit_thirds(base_test, units_test)
    df_test["split"] = "test"
    df = pd.concat([df_dev, df_test], ignore_index=True)
    df.to_csv(OUT_DIR / "health_sign_convention.csv", index=False)
    print(f"  saved {OUT_DIR / 'health_sign_convention.csv'}")

    # Aggregate: late-mean − early-mean across units → sign of degradation
    delta_rows: list[dict] = []
    for col in HEALTH_COLS:
        for split, sub in (("dev", df_dev), ("test", df_test)):
            grp = sub[sub["column"] == col].groupby("phase")["mean"].mean()
            early = grp.get("early", float("nan"))
            mid   = grp.get("mid",   float("nan"))
            late  = grp.get("late",  float("nan"))
            delta_rows.append({
                "split":           split,
                "column":          col,
                "mean_early":      early,
                "mean_mid":        mid,
                "mean_late":       late,
                "delta_late_early": late - early,
                "direction":       "decreases" if late - early < 0
                                   else ("increases" if late - early > 0 else "flat"),
            })
    df_delta = pd.DataFrame(delta_rows)
    df_delta.to_csv(OUT_DIR / "health_sign_convention_summary.csv", index=False)
    print(f"  saved {OUT_DIR / 'health_sign_convention_summary.csv'}")

    # Overall verdict per column (using test split)
    test_d = df_delta[df_delta["split"] == "test"].copy()
    verdict_lines = []
    for _, r in test_d.iterrows():
        if r["mean_early"] > 0.05 or r["mean_early"] < -0.05:
            anchor = "non-zero early"
        else:
            anchor = "near-zero early (delta around 0)"
        verdict_lines.append(
            f"- **{r['column']}**: early≈{r['mean_early']:.4f}, late≈{r['mean_late']:.4f}, "
            f"Δ={r['delta_late_early']:+.4f} → {r['direction']} over life ({anchor})"
        )

    md = f"""# N-CMAPSS DS02 health-parameter sign convention (Task 5)

*Read-only.  No model used.*

## Splits

* dev  units: {units_dev}
* test units: {units_test}

## Per-unit early/mid/late statistics

CSV: `health_sign_convention.csv` (full row-level).

## Per-column life-direction summary (averaged across units within split)

{df_to_md(df_delta, floatfmt='.5f')}

## Interpretation (test split)

{chr(10).join(verdict_lines)}

## Sign-convention conclusion

For health parameters that **decrease** from early to late life
(`mean_late < mean_early`), the documented N-CMAPSS convention is:

* `*_eff_mod` and `*_flow_mod` are **delta around 0** (healthy ≈ 0,
  degraded → negative).

Therefore the **expected physical sign** of:

* `Pearson(θ_phys − 1.0, *_eff_mod)`  → **positive** when the model
  correctly identifies degradation (both move toward negative together
  during life progression).

A *negative* observed Pearson (the experiment-matrix run reported
−0.85 for θ_η_hpt vs HPT_eff_mod) means **either** the model has
anti-learned the relationship, **or** the correlation is a time-axis
artifact whose sign is determined by other latent factors.

## Important guard (do NOT flip sign cosmetically)

Per ADR-0013: "Do not simply flip θ sign to improve correlation.
Establish the physical meaning first."  This script confirms the
physical sign expectation; it does not modify the model.
"""
    (OUT_DIR / "health_sign_convention.md").write_text(md, encoding="utf-8")
    print(f"  saved {OUT_DIR / 'health_sign_convention.md'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
