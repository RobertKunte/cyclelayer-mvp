# ADR-0014 — V3.1b RUL collapse diagnostic phase

* Status: **Accepted**
* Decision owner: Robert Kunte
* Builds on: [ADR-0012](ADR-0012-v3-thermal-auxiliary-scope.md), [ADR-0013](ADR-0013-v31b-theta-identifiability-tests.md)
* Date: 2026-05-13

## Problem

The V3.1b thermal-aux experiment matrix (RUN `20260513_090332`) reports

| run | test RMSE | test bias |
|---|---|---|
| A baseline           | 21.73 | +15.7 |
| B physics-aux only   | 21.74 | +15.8 |
| C physics + θ → RUL  | 21.41 | +15.4 |
| D shuffled θ         | 21.41 | +15.4 |

with a predicted-vs-true scatter that looks **almost flat**, predictions
clustered in a narrow band around the train RUL mean (≈ 50). RMSE
differences across A/B/C/D are within 0.3 cycles. ADR-0013 already
established that HPT/LPT θ is unidentifiable from L_temp and that the
prognostics head ignores θ (D ≈ C). The deeper question now: **is the RUL
model itself collapsing to a mean prediction?**

If yes, "physics aux helps RUL" is not even on the table — there is no
real RUL learning to help.

## Hypotheses to falsify

* **H1 — mean-collapse:** RUL head predicts approximately the training-set
  mean RUL irrespective of input. Pearson(pred, true) ≈ 0, slope ≈ 0,
  std(pred) << std(true).
* **H2 — target normalisation bug:** something is silently rescaling the
  RUL target or prediction (e.g. a leftover scaler, a `/max_rul`
  somewhere).
* **H3 — window/label misalignment:** target attached to wrong window
  endpoint, or windows cross unit boundaries.
* **H4 — train/test distribution mismatch:** test units sit outside the
  training RUL distribution, so the head defaults to mean.
* **H5 — under-training, not architectural failure:** the model *could*
  learn but was trained too briefly / with insufficient data.
* **H6 — RUL head ignores all features:** not just θ — h_sens and z_ops
  also bypassed; prediction is dominated by bias init.
* **H7 — task is harder than current model capacity:** DS02 RUL is a
  difficult target and the current encoder / head combination is too
  small.

## Decision rule

Run a structured **read-only** diagnostic suite that produces evidence
for each hypothesis before any architectural change or hyperparameter
tuning. Six diagnostic scripts under `scripts/diagnose_v31b_*.py` and
`scripts/test_v31b_rul_overfit_tiny.py` /
`scripts/train_simple_rul_baselines_ds02.py`, plus an aggregator. All
outputs land under
`artifacts/cyclelayer_v3/rul_model_sanity/<TIMESTAMP>/`.

| # | Test | What it falsifies / confirms |
|---|---|---|
| 2 | Collapse metrics + trivial baselines (constant 50, train-mean, train-median, per-unit linear) | H1 — direct evidence of mean-collapse |
| 3 | Plots (scatter, residuals, hist, calibration, per-unit trajectory) | H1, H3, H6 |
| 4 | Target / window alignment audit | H2, H3 |
| 5 | Tiny-overfit smoke test (256 / 1k / 4k windows) | H5 vs H6/H7 |
| 6 | Simple Ridge / RF / HGB baselines on DS02 | H7 |
| 7 | Branch usage ablations (zero / shuffle θ, aux, h_sens) | H1, H6 |
| 8 | Aggregate `RUL_MODEL_SANITY_SUMMARY.md` with PASS / WEAK / FAIL | — |

## Decision logic

```
if target_alignment_FAIL:
    STOP. Fix data/target pipeline first. All other diagnostics are invalid.

elif tiny_overfit_FAIL:
    STOP. Implementation problem (output scaling / loss / optimizer / arch).
    Debug before any V3.1b training improvements.

elif tiny_overfit_PASS and full_model_collapses:
    Generalisation / training-budget / sampling problem.
    Next: longer training, EOL-balanced sampling, larger encoder.

elif simple_baselines_clearly_beat_V3.1b:
    V3.1b architecture/training is not competitive.
    Next: revert to V2 encoder/baseline; V3.1b physics only as auxiliary diagnostic.

elif V3.1b R² > 0.4 AND beats constant + simple baselines:
    Continue V3.1b improvements.

else:
    Inconclusive — surface to Robert.
```

Default expectation given the experiment matrix: **likely H1 (mean-collapse)
or H5 (under-training)**. Diagnostics decide which.

## Hard constraints during this phase

* No model architecture changes (`cyclelayer_v3.py` frozen).
* No YAML / physical-constant changes.
* No hyperparameter tuning loops.
* No EPR / pressure loss reintroduction.
* No DS02 test leakage (the simple-baseline script trains on train units
  only; test units `[11, 14, 15]` remain held out).
* All new outputs are read-only diagnostics under
  `artifacts/cyclelayer_v3/rul_model_sanity/<TIMESTAMP>/`.
* Existing 142/142 tests stay green.

## Stop point

After `RUL_MODEL_SANITY_SUMMARY.md` is produced. **Do not** change
`CycleLayerV3`, do not add V4, do not tune. Robert reviews and chooses
the next action.
