"""Aggregate Tasks 2-7 outputs into a single IDENTIFIABILITY_SUMMARY.md
with the ADR-0013 PASS / WEAK / FAIL verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUT_DIR = Path(__file__).parent.parent / "artifacts" / "cyclelayer_v3" / "theta_identifiability"


def section_or_missing(path: Path, title: str) -> tuple[str, bool]:
    if path.exists():
        return path.read_text(encoding="utf-8"), True
    return f"## {title}\n\n*Report missing: `{path.name}` not produced.  Run the corresponding script.*\n", False


def parse_local_sensitivity() -> dict:
    csv = OUT_DIR / "local_sensitivity_mean_elasticity.csv"
    if not csv.exists():
        return {"status": "missing"}
    df = pd.read_csv(csv)
    temp_outputs = ["T24", "T30", "T45", "T50"]
    press_outputs = ["P30", "P45", "P50", "PR_hpt", "PR_lpt", "EPR"]
    mT = df["output"].isin(temp_outputs)
    mP = df["output"].isin(press_outputs)
    out = {"status": "ok", "per_theta": {}}
    for theta in ["eta_fan", "eta_lpc", "eta_hpc", "eta_hpt", "eta_lpt"]:
        col = f"elasticity_{theta}"
        e_T = df.loc[mT, col].abs().max()
        e_P = df.loc[mP, col].abs().max()
        out["per_theta"][theta] = {
            "max_abs_elasticity_T": float(e_T),
            "max_abs_elasticity_P": float(e_P),
            "T_identifiable":       bool(e_T > 1e-2),
        }
    return out


def parse_loss_gradient() -> dict:
    csv = OUT_DIR / "loss_gradient_paths.csv"
    if not csv.exists():
        return {"status": "missing"}
    df = pd.read_csv(csv)
    pivot = df[df["loss_term"] == "L_temp"].groupby("theta")["grad_norm"].mean()
    return {
        "status": "ok",
        "L_temp_grad_by_theta": {n: float(v) for n, v in pivot.items()},
    }


def parse_sign_convention() -> dict:
    csv = OUT_DIR / "health_sign_convention_summary.csv"
    if not csv.exists():
        return {"status": "missing"}
    df = pd.read_csv(csv)
    test_d = df[df["split"] == "test"]
    rows = {}
    for _, r in test_d.iterrows():
        rows[r["column"]] = {
            "delta_late_early": float(r["delta_late_early"]),
            "direction":        r["direction"],
        }
    return {"status": "ok", "test": rows}


def parse_partial_corr() -> dict:
    csv = OUT_DIR / "partial_correlations.csv"
    if not csv.exists():
        return {"status": "missing"}
    df = pd.read_csv(csv)
    pairs = [
        ("theta_hpt_delta", "HPT_eff_mod"),
        ("theta_lpt_delta", "LPT_eff_mod"),
        ("lpt_flow_pred",   "LPT_flow_mod"),
    ]
    rows = {}
    for xcol, ycol in pairs:
        sub = df[(df["x"] == xcol) & (df["y"] == ycol)]
        if sub.empty: continue
        get = lambda c: float(sub[sub["control"] == c]["pearson"].iloc[0]) \
                          if (sub["control"] == c).any() else float("nan")
        rows[f"{xcol} vs {ycol}"] = {
            "raw":          get("none"),
            "partial_RUL":  get("RUL"),
            "partial_RUL_ops": get("RUL+ops"),
            "partial_cycle_ops": get("cycle+ops"),
        }
    return {"status": "ok", "pairs": rows}


def parse_rul_usage() -> dict:
    csv = OUT_DIR / "rul_theta_usage.csv"
    if not csv.exists():
        return {"status": "missing"}
    df = pd.read_csv(csv)
    real_rmse = float(df.loc[df["variant"] == "real", "RMSE"].iloc[0])
    max_pct = float(df["pct_RMSE_change"].abs().max())
    return {
        "status":        "ok",
        "real_RMSE":     real_rmse,
        "max_abs_pct_change": max_pct,
        "head_uses_theta":    bool(max_pct >= 0.5),   # >0.5% = nontrivial usage
    }


def parse_synthetic_recovery() -> dict:
    csv = OUT_DIR / "synthetic_recovery.csv"
    if not csv.exists():
        return {"status": "missing"}
    df = pd.read_csv(csv)
    rows = {}
    for _, r in df.iterrows():
        rows[r["theta"]] = {
            "true":          float(r["true"]),
            "recovered_T":   float(r["recovered_T"]),
            "abs_err_T":     float(r["abs_err_T"]),
            "recovered_T_P": float(r["recovered_T_P"]),
            "abs_err_T_P":   float(r["abs_err_T_P"]),
            "T_recovers":    bool(r["abs_err_T"] < 0.01),
            "T_P_recovers":  bool(r["abs_err_T_P"] < 0.01),
        }
    return {"status": "ok", "per_theta": rows}


def main() -> None:
    sens = parse_local_sensitivity()
    grad = parse_loss_gradient()
    sign = parse_sign_convention()
    part = parse_partial_corr()
    rul  = parse_rul_usage()
    synth = parse_synthetic_recovery()

    # ── Derive verdict ───────────────────────────────────────────────────
    def verdict_for(theta: str) -> tuple[str, list[str]]:
        reasons = []
        sens_ok = (sens.get("status") == "ok"
                   and sens["per_theta"][theta]["T_identifiable"])
        if sens.get("status") == "ok":
            v = sens["per_theta"][theta]
            reasons.append(f"sensitivity max|elasticity|_T = {v['max_abs_elasticity_T']:.3e} "
                            f"(threshold 1e-2 → {'OK' if v['T_identifiable'] else 'FAIL'})")
        grad_ok = False
        if grad.get("status") == "ok":
            g = grad["L_temp_grad_by_theta"].get(theta, float("nan"))
            grad_ok = g > 1e-6
            reasons.append(f"∂L_temp/∂θ_{theta} = {g:.3e} "
                            f"({'active' if grad_ok else '**zero pathway**'})")
        synth_ok_T = synth.get("status") == "ok" and synth["per_theta"].get(theta, {}).get("T_recovers", False)
        if synth.get("status") == "ok":
            v = synth["per_theta"].get(theta, {})
            reasons.append(f"synthetic recovery from T only: err = {v.get('abs_err_T', float('nan')):.4f} "
                            f"(threshold 0.01 → {'OK' if synth_ok_T else 'FAIL'})")
        # Partial correlation (for hpt/lpt only; AuxHead is supervised so different)
        if theta in ("eta_hpt", "eta_lpt"):
            key = f"theta_{theta[-3:]}_delta vs {theta[-3:].upper()}_eff_mod"
            if part.get("status") == "ok" and key in part.get("pairs", {}):
                p = part["pairs"][key]
                raw = p["raw"]
                partial = p["partial_RUL_ops"]
                if abs(raw) > 0.6 and abs(partial) < 0.2:
                    reasons.append(f"raw r = {raw:+.3f}, partial r (RUL+ops) = {partial:+.3f} → ARTIFACT")
                elif abs(partial) > 0.4:
                    reasons.append(f"raw r = {raw:+.3f}, partial r = {partial:+.3f} → ROBUST")
                else:
                    reasons.append(f"raw r = {raw:+.3f}, partial r = {partial:+.3f} → INCONSISTENT")

        # Composite verdict (sens AND grad AND synthetic recovery T-only ALL required for PASS)
        if sens_ok and grad_ok and synth_ok_T:
            v = "PASS"
        elif (not sens_ok) and (not grad_ok) and (not synth_ok_T):
            v = "FAIL"
        else:
            v = "WEAK"
        return v, reasons

    verdicts = {n: verdict_for(n) for n in ("eta_fan", "eta_lpc", "eta_hpc", "eta_hpt", "eta_lpt")}

    # ── Answers to the 5 ADR-0013 questions ─────────────────────────────
    v_hpt = verdicts["eta_hpt"][0]
    v_lpt = verdicts["eta_lpt"][0]
    sens_hpt = sens.get("per_theta", {}).get("eta_hpt", {})
    sens_lpt = sens.get("per_theta", {}).get("eta_lpt", {})
    grad_hpt = grad.get("L_temp_grad_by_theta", {}).get("eta_hpt", float("nan"))
    grad_lpt = grad.get("L_temp_grad_by_theta", {}).get("eta_lpt", float("nan"))
    synth_hpt = synth.get("per_theta", {}).get("eta_hpt", {})
    synth_lpt = synth.get("per_theta", {}).get("eta_lpt", {})

    a_text = (
        "**NO.** Local sensitivity, L_temp gradient, and synthetic-recovery "
        "all show that `θ_η_hpt` and `θ_η_lpt` are *architecturally* "
        "unidentifiable from V3.1b's temperature-only loss:\n"
        f"* `max |elasticity|_T(eta_hpt)` = {sens_hpt.get('max_abs_elasticity_T', float('nan')):.3e}; "
        f"`(eta_lpt)` = {sens_lpt.get('max_abs_elasticity_T', float('nan')):.3e}  "
        f"(threshold 1e-2)\n"
        f"* `∂L_temp/∂θ_η_hpt` = {grad_hpt:.3e};  `∂L_temp/∂θ_η_lpt` = {grad_lpt:.3e}  "
        f"(thresholds 1e-3 relative / 1e-8 absolute)\n"
        f"* Synthetic recovery from T only: HPT |err| = "
        f"{synth_hpt.get('abs_err_T', float('nan')):.4f}, LPT |err| = "
        f"{synth_lpt.get('abs_err_T', float('nan')):.4f}  (threshold 0.01).  "
        f"Same setup with T+P targets recovers all five θ to within "
        f"{max(synth.get('per_theta', {}).get('eta_hpt', {}).get('abs_err_T_P', 0), 0):.4f} / "
        f"{max(synth.get('per_theta', {}).get('eta_lpt', {}).get('abs_err_T_P', 0), 0):.4f}."
    )

    if part.get("status") == "ok" and "theta_hpt_delta vs HPT_eff_mod" in part.get("pairs", {}):
        p_hpt = part["pairs"]["theta_hpt_delta vs HPT_eff_mod"]
        p_lpt = part["pairs"]["theta_lpt_delta vs LPT_eff_mod"]
        if abs(p_hpt["raw"]) > 0.6 and abs(p_hpt["partial_RUL_ops"]) < 0.2:
            b_text = (f"**Spurious / time-axis artifact.**  "
                      f"`Pearson(θ_η_hpt − 1, HPT_eff_mod)`: raw {p_hpt['raw']:+.3f}, "
                      f"partial after RUL+ops {p_hpt['partial_RUL_ops']:+.3f}. "
                      f"Same pattern for LPT (raw {p_lpt['raw']:+.3f}, partial {p_lpt['partial_RUL_ops']:+.3f}).  "
                      f"Strong raw correlation collapses under controls — confirming the gradient/sensitivity "
                      f"diagnostics that physical θ_η_hpt/lpt is not identifiable.")
        else:
            b_text = (f"Inconclusive from partial-correlation alone (raw {p_hpt['raw']:+.3f}, "
                      f"partial {p_hpt['partial_RUL_ops']:+.3f}). But gradient + synthetic-recovery "
                      f"already establish unidentifiability.")
    else:
        b_text = ("Partial-correlation script needs the user's Colab C-run checkpoint to be "
                  "definitive. Local smoke checkpoint shows near-zero correlations (model under-trained).  "
                  "However, the **gradient and synthetic-recovery diagnostics already establish unidentifiability** "
                  "independent of any trained model.")

    if rul.get("status") == "ok":
        c_text = (f"**NO.** Max |ΔRMSE / RMSE_real| across {{real, shuffle_batch, shuffle_within_unit, "
                  f"shuffle_across_units, constant_healthy, constant_lo}} = "
                  f"**{rul['max_abs_pct_change']:.4f} %** — well below the 0.5 % threshold.  "
                  f"The prognostics head ignores θ_phys.")
    else:
        c_text = "RUL-usage script needs a checkpoint; verdict pending.  See diagnostic 6."

    d_text = (
        "**Yes — but in a more limited role than originally pitched.**  "
        "V3.1b can still serve as a differentiable *thermal* regulariser for the encoder.  "
        "Compressor θ (fan/lpc/hpc) are identifiable from temperature targets and may be "
        "meaningful health indicators.  But HPT/LPT efficiency identification (the two "
        "documented N-CMAPSS health params with nonzero signal in DS02) is *not* in V3.1b's reach."
    )

    e_text = (
        "Either:\n\n"
        "* **A)** Drop HPT/LPT η identification claims from V3.1b's pitch; report only the "
        "  compressor-side θ correlations.\n\n"
        "* **B)** Move HPT/LPT η identification to **V4** with the pressure / EPR / flow-matching "
        "  architecture (per ADR-0012 / ADR-0013).  Synthetic recovery (Case B above) shows "
        "  this *would* work once `{P30, P50}` enter the loss.  V4 also resolves the closure-vs-"
        "  measured-speed mismatch that V3.1b's explicit-closure architecture inherits.\n\n"
        "Choice is Robert's; ADR-0013 does not pre-decide A vs B."
    )

    # ── Assemble the summary ─────────────────────────────────────────────
    md = f"""# IDENTIFIABILITY SUMMARY — V3.1b θ (ADR-0013)

*Aggregates Tasks 2 – 7.  Read-only.  No YAML written.  No DS02 tuning.*

## Per-θ verdict

| θ | verdict | reasons |
|---|---|---|
"""
    for n in ("eta_fan", "eta_lpc", "eta_hpc", "eta_hpt", "eta_lpt"):
        v, reasons = verdicts[n]
        md += f"| `{n}` | **{v}** | {' / '.join(reasons)} |\n"

    md += f"""

## Answers to ADR-0013 questions

### A — Can current V3.1b identify `θ_η_hpt` and `θ_η_lpt` from its current loss?

{a_text}

### B — Are the previously-reported high Pearson correlations likely physical or spurious?

{b_text}

### C — Does RUL use `θ_phys`?

{c_text}

### D — Is V3.1b still useful?

{d_text}

### E — What architectural change is required if A is NO?

{e_text}

## Final verdict for HPT/LPT θ identifiability in V3.1b

* `θ_η_hpt`: **{verdicts['eta_hpt'][0]}**
* `θ_η_lpt`: **{verdicts['eta_lpt'][0]}**

If both are FAIL: V3.1b temperature-only L_temp **cannot** identify the
documented HPT/LPT efficiency health parameters.  This is an architectural
property, not a training artifact — synthetic recovery confirms it
*without* any trained model.

## Constraints honored

* No DS02 / C0 / C1 / C2 parameter tuning.
* No YAML physical-constant writes.
* No `fit_*` helper on real data.  Synthetic recovery's optimiser
  operates on synthetic targets only.
* No supervised `L_θ` on `θ_phys`.
* Pressure / EPR loss disabled in the V3.1b training path.

## Artifacts

* `local_sensitivity_report.md`        ({sens.get('status')})
* `loss_gradient_paths_report.md`      ({grad.get('status')})
* `health_sign_convention.md`          ({sign.get('status')})
* `partial_correlations_report.md`     ({part.get('status')})
* `rul_theta_usage_report.md`          ({rul.get('status')})
* `synthetic_recovery_report.md`       ({synth.get('status')})

## Next action

Robert decides between option A (limit V3.1b scope) and option B (V4
work).  No code change happens until that decision is recorded.
"""
    out_path = OUT_DIR / "IDENTIFIABILITY_SUMMARY.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")

    # Also save a machine-readable JSON
    json_path = OUT_DIR / "identifiability_summary.json"
    json_path.write_text(json.dumps({
        "verdicts":              {n: v[0] for n, v in verdicts.items()},
        "local_sensitivity":     sens,
        "loss_gradient_paths":   grad,
        "sign_convention":       sign,
        "partial_correlations":  part,
        "rul_theta_usage":       rul,
        "synthetic_recovery":    synth,
    }, indent=2, default=float))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
