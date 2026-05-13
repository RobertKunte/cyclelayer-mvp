# ADR-0013 — V3.1b θ identifiability test phase

* Status: **Accepted**
* Decision owner: Robert Kunte
* Builds on: [ADR-0012](ADR-0012-v3-thermal-auxiliary-scope.md)
* Date: 2026-05-13

## Context

The first V3.1b thermal-aux experiment matrix run (`20260513_090332`) produced
the following on the test split (units 11/14/15):

| run | RUL RMSE | Pearson(θ_η_hpt − 1, HPT_eff_mod) | Pearson(θ_η_lpt − 1, LPT_eff_mod) |
|---|---|---|---|
| A baseline       | 21.73 | — | — |
| B physics-aux    | 21.74 | **−0.667** | **−0.632** |
| C physics + θ→RUL| 21.41 | **−0.850** | **−0.740** |
| D shuffled θ     | 21.41 | — | — |

Two findings are alarming enough to demand a falsifiable test before any
publication or further training:

1. **Sign is reversed and magnitude is suspiciously high.** Strong correlation
   in the wrong direction either means the model has learned the *opposite*
   of HPT/LPT efficiency, or the strong correlation is a spurious time/RUL
   artifact (since all health parameters and θ all share the run-to-failure
   axis).
2. **D ablation shows ΔRMSE ≈ 0** — shuffling θ does not change RUL.
   So θ_phys is not used by the prognostics head at all.

We must determine *physically* whether V3.1b can identify HPT/LPT efficiency
θ before claiming θ is an interpretable health representation.

## Engineering hypothesis to falsify

In the V3.1b explicit-closure architecture with measured Nf, Nc, Wf:

```
W_HPT = W_HPC                 → ΔT_HPT = W_HPC / (m4 · cp_t)   FIXED
W_LPT = W_LPC + W_Fan_total   → ΔT_LPT = W_LPT / (m4 · cp_t)   FIXED
```

`η_hpt` and `η_lpt` affect only the *isentropic* ΔT and hence the *pressure*
ratios `PR_hpt` and `PR_lpt`. They do **not** affect turbine outlet temperatures.
Since V3.1b's `L_temp` uses only `{T24, T30, T50}`, there is no temperature
target through which `θ_η_hpt` and `θ_η_lpt` receive gradient — they are
*architecturally unidentifiable* from this loss.

This must be confirmed by direct tests, not by assumption.

## Decision — required diagnostics

Six read-only diagnostic suites + one synthetic controlled test, all under
`scripts/diagnose_v31b_*.py` and `scripts/test_v31b_synthetic_theta_recovery.py`,
each writing to `artifacts/cyclelayer_v3/theta_identifiability/`:

1. **Local sensitivity / Jacobian** — `d{T,P,PR,EPR}/dθ` at representative
   operating points (FC02, DS02 cruise sample, several DS02 rows).
2. **Loss-gradient paths** — `∂L_temp/∂θ_i`, plus the same for L_rul, L_aux,
   L_healthy, L_smooth, L_total.
3. **N-CMAPSS sign convention check** — verifies what "degraded" means
   numerically for HPT_eff_mod, LPT_eff_mod, LPT_flow_mod (and the fan/lpc/hpc
   variants).
4. **Partial correlations** — raw vs residualised correlations controlling
   for RUL, cycle, ops, and combinations. Per-unit + global.
5. **RUL-head θ usage** — RMSE under {real θ, shuffled-across-batch,
   shuffled-within-unit-over-time, shuffled-across-units, constant healthy θ,
   zero θ} on a trained C checkpoint.
6. **Synthetic recovery** — known-θ → BraytonEngine → outputs → recover θ
   by optimisation on synthetic data only, with two target sets:
   (A) `{T24, T30, T50}` only; (B) add `{P30, P50}` (or equivalently
   `PR_hpt`, `PR_lpt`, EPR).

## Decision rule (binding)

Verdict for **HPT/LPT θ identifiability** in V3.1b — one of:

* **PASS** — HPT/LPT θ show nonzero physical sensitivity to the *used* loss
  targets, retain correct-sign partial correlation > 0.4 after controls,
  per-unit trends are consistent, **and** synthetic temperature-only recovery
  works.
* **WEAK** — Some raw correlations exist but gradients or partial correlations
  are inconsistent or borderline.
* **FAIL** — HPT/LPT θ have near-zero L_temp gradient *and* synthetic
  temperature-only recovery fails *and* partial correlations collapse after
  controlling for RUL/cycle/ops.

### If FAIL

Two options for V3.1b's future scope (no third option — no parameter tuning,
no eta below 0.88):

* **A)** Limit V3.1b to *identifiable* compressor θ only (drop HPT/LPT θ
  identification claims from the V3.1b pitch).
* **B)** Move HPT/LPT η identification to V4 (with pressure / EPR / flow
  matching as part of the loss — out of V3.1b scope per ADR-0012).

The decision between A and B is Robert's. The ADR documents the test
outcome; the choice is recorded separately.

## Hard constraints during this phase

* No DS02 / C0 / C1 / C2 parameter tuning.
* No YAML physical-constant writes.
* No `fit_*` helper on real data; the synthetic recovery script is the
  *only* place an optimiser is allowed, and it operates on synthetic targets
  only.
* No supervised `L_θ` on `θ_phys`. Health-parameter GT is evaluation-only
  for `θ_phys`.
* Pressure / EPR loss remains **disabled** in the main V3.1b training path.
  An experimental diagnostic loss may exist behind a separate flag for
  Task 7's case (B), but must not be reachable from the main config.
* All new outputs are read-only diagnostics under
  `artifacts/cyclelayer_v3/theta_identifiability/`.
* Existing tests stay green.

## Stop point

After producing `IDENTIFIABILITY_SUMMARY.md`. Do **not** launch longer
training or change YAML. Robert reviews the verdict and chooses the
follow-up (A or B above).
