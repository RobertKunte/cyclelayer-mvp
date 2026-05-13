# IDENTIFIABILITY SUMMARY — V3.1b θ (ADR-0013)

*Aggregates Tasks 2 – 7.  Read-only.  No YAML written.  No DS02 tuning.*

## Per-θ verdict

| θ | verdict | reasons |
|---|---|---|
| `eta_fan` | **PASS** | sensitivity max|elasticity|_T = 3.573e-01 (threshold 1e-2 → OK) / ∂L_temp/∂θ_eta_fan = 4.265e+01 (active) / synthetic recovery from T only: err = 0.0000 (threshold 0.01 → OK) |
| `eta_lpc` | **PASS** | sensitivity max|elasticity|_T = 2.013e-01 (threshold 1e-2 → OK) / ∂L_temp/∂θ_eta_lpc = 4.404e+01 (active) / synthetic recovery from T only: err = 0.0000 (threshold 0.01 → OK) |
| `eta_hpc` | **PASS** | sensitivity max|elasticity|_T = 5.529e-01 (threshold 1e-2 → OK) / ∂L_temp/∂θ_eta_hpc = 3.105e+01 (active) / synthetic recovery from T only: err = 0.0000 (threshold 0.01 → OK) |
| `eta_hpt` | **FAIL** | sensitivity max|elasticity|_T = 0.000e+00 (threshold 1e-2 → FAIL) / ∂L_temp/∂θ_eta_hpt = 0.000e+00 (**zero pathway**) / synthetic recovery from T only: err = 0.0400 (threshold 0.01 → FAIL) / raw r = +0.008, partial r = +0.007 → INCONSISTENT |
| `eta_lpt` | **FAIL** | sensitivity max|elasticity|_T = 0.000e+00 (threshold 1e-2 → FAIL) / ∂L_temp/∂θ_eta_lpt = 0.000e+00 (**zero pathway**) / synthetic recovery from T only: err = 0.0300 (threshold 0.01 → FAIL) / raw r = +0.010, partial r = +0.015 → INCONSISTENT |


## Answers to ADR-0013 questions

### A — Can current V3.1b identify `θ_η_hpt` and `θ_η_lpt` from its current loss?

**NO.** Local sensitivity, L_temp gradient, and synthetic-recovery all show that `θ_η_hpt` and `θ_η_lpt` are *architecturally* unidentifiable from V3.1b's temperature-only loss:
* `max |elasticity|_T(eta_hpt)` = 0.000e+00; `(eta_lpt)` = 0.000e+00  (threshold 1e-2)
* `∂L_temp/∂θ_η_hpt` = 0.000e+00;  `∂L_temp/∂θ_η_lpt` = 0.000e+00  (thresholds 1e-3 relative / 1e-8 absolute)
* Synthetic recovery from T only: HPT |err| = 0.0400, LPT |err| = 0.0300  (threshold 0.01).  Same setup with T+P targets recovers all five θ to within 0.0048 / 0.0054.

### B — Are the previously-reported high Pearson correlations likely physical or spurious?

Inconclusive from partial-correlation alone (raw +0.008, partial +0.007). But gradient + synthetic-recovery already establish unidentifiability.

### C — Does RUL use `θ_phys`?

**NO.** Max |ΔRMSE / RMSE_real| across {real, shuffle_batch, shuffle_within_unit, shuffle_across_units, constant_healthy, constant_lo} = **0.0699 %** — well below the 0.5 % threshold.  The prognostics head ignores θ_phys.

### D — Is V3.1b still useful?

**Yes — but in a more limited role than originally pitched.**  V3.1b can still serve as a differentiable *thermal* regulariser for the encoder.  Compressor θ (fan/lpc/hpc) are identifiable from temperature targets and may be meaningful health indicators.  But HPT/LPT efficiency identification (the two documented N-CMAPSS health params with nonzero signal in DS02) is *not* in V3.1b's reach.

### E — What architectural change is required if A is NO?

Either:

* **A)** Drop HPT/LPT η identification claims from V3.1b's pitch; report only the   compressor-side θ correlations.

* **B)** Move HPT/LPT η identification to **V4** with the pressure / EPR / flow-matching   architecture (per ADR-0012 / ADR-0013).  Synthetic recovery (Case B above) shows   this *would* work once `{P30, P50}` enter the loss.  V4 also resolves the closure-vs-  measured-speed mismatch that V3.1b's explicit-closure architecture inherits.

Choice is Robert's; ADR-0013 does not pre-decide A vs B.

## Final verdict for HPT/LPT θ identifiability in V3.1b

* `θ_η_hpt`: **FAIL**
* `θ_η_lpt`: **FAIL**

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

* `local_sensitivity_report.md`        (ok)
* `loss_gradient_paths_report.md`      (ok)
* `health_sign_convention.md`          (ok)
* `partial_correlations_report.md`     (ok)
* `rul_theta_usage_report.md`          (ok)
* `synthetic_recovery_report.md`       (ok)

## Next action

Robert decides between option A (limit V3.1b scope) and option B (V4
work).  No code change happens until that decision is recorded.
