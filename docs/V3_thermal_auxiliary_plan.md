# V3.1b Thermal Auxiliary Plan

*Repo-versioned plan. Authoritative scope and experiment matrix for the
re-scoped V3.1b. Companion to [ADR-0012](decisions/ADR-0012-v3-thermal-auxiliary-scope.md).*

## Scope (locked)

V3.1b is **a differentiable thermal auxiliary physics layer**, not a
full cycle / EPR validation model. See ADR-0012 for the decision context.

### Training signals (allowed)

| Signal | Used in loss | Notes |
|---|---|---|
| `L_rul` (asymmetric RUL) | YES — primary | Same form as v1 / V2 |
| `L_temp` on T24, T30, T50 | YES — auxiliary, weak weight | Compared in normalized sensor space; per-sensor σ from train split |
| `L_aux` on LPT_flow_mod | YES — small weight | Normalised: `MSE((pred − GT)/σ_lpt_flow)` |
| `L_healthy` weak prior on θ | YES — weak weight | `||θ_phys − 1||²` averaged over RUL > 80 samples |
| `L_smooth` temporal Δ² | YES — small weight | First difference on θ_phys (and AuxHead) over time |

### Training signals (forbidden in V3.1b)

| Signal | Forbidden because |
|---|---|
| `L_pressure` on P30 / Ps30 | V3.1b cannot satisfy P/EPR with realistic eta — see ADR-0012, C0d |
| `L_EPR` on P50 / P2 | Same |
| Supervised `L_θ` on θ_phys | θ_phys learned without GT; correlations evaluated post-hoc only |
| Supervised L on HPT_eff_mod / LPT_eff_mod via θ | Same |

The loss module **asserts** at construction that `use_pressure_loss == False`
and `use_epr_loss == False`. A code change is required to enable either —
that change requires a new ADR.

### Diagnostics only (logged, not in loss)

* T45 / T48 (T48 as proxy for T45; documented tolerance)
* P30 / Ps30 (via `pressure_proxies.Ps30_proxy`)
* P50 / P2 → EPR
* Per-component PR (PR_fan / lpc / hpc, closure-determined PR_hpt / lpt)
* Mass + shaft + combustor balance residuals (already in BraytonEngine
  diagnostics)
* PR-clamp activity, Wc-clamp activity, turbine plausibility (`min_T45`,
  `frac_T50_below_limit`, ...) — V3.1b correction 5

## Frozen configuration family

`configs/cyclelayer_v3_thermal_aux.yaml` carries the only V3.1b training
configuration. It is **labelled `thermal_regularizer_v3_1b_not_epr_validated`**
and chosen for numerical and thermal stability of the auxiliary layer; it is
**not** a C-MAPSS EPR-validated design point. Adopted from the old D-like
baseline that gave physically reasonable T50, PR_lpt and T45 anchor:

| | |
|---|---|
| BPR | 5.5 |
| Wc_fan_design | 900 kg/s |
| PR_fan / lpc / hpc | 1.60 / 2.00 / 12.00 |
| eta_design fan / lpc / hpc | 0.92 / 0.90 / 0.88 |
| eta_design hpt / lpt | 0.90 / 0.92 |

Source verification: literature Walsh & Fletcher / Kurzke generic 2-spool
turbofan. Documented C-MAPSS reference (Frederick TM2007-215026)
**deferred to V4** per ADR-0012.

## Hard constraints

1. No further C0 / C0b / C0c / C0d tuning.
2. No BPR / Wc / PR / eta optimisation against FC02, UserGuide FCs, DS02,
   or any sensor data.
3. No EPR / pressure loss.
4. No supervised L_θ on θ_phys.
5. Existing legacy modules frozen: `brayton_cycle`, `cycle_layer`,
   `physresnet`, `encoder`, `baselines`, `prognostics`.
6. Existing tests stay green (currently 108/108).
7. New modules / tests / scripts only — extend, don't break.

## Module inventory (V3.1b thermal-aux)

| File | Role | Status |
|---|---|---|
| `src/cyclelayer/models/units.py` | Imperial ↔ SI | unchanged |
| `src/cyclelayer/models/stations.py` | Constants | unchanged |
| `src/cyclelayer/models/brayton_engine.py` | Differentiable Brayton-cycle module (5 θ in) | unchanged |
| `src/cyclelayer/models/pressure_proxies.py` | Ps30_proxy (DIAGNOSTIC only in V3.1b) | unchanged |
| `src/cyclelayer/data/ncmapss_v3.py` | DS02 adapter (Imperial-named, lazy default) | minor extension: windowed wrapper |
| `src/cyclelayer/models/cyclelayer_v3.py` | **NEW.** Hybrid wrapper: encoder + ParamHead + AuxHead + Brayton + Prognostics + target-sensor masking | NEW |
| `src/cyclelayer/losses/cyclelayer_v3_losses.py` | **NEW.** Thermal-aux composite loss with EPR / pressure asserts | NEW |
| `configs/cyclelayer_v3_thermal_aux.yaml` | **NEW.** Single V3.1b training config | NEW |
| `tests/test_cyclelayer_v3_integration.py` | **NEW.** Model wrapper tests | NEW |
| `tests/test_cyclelayer_v3_losses.py` | **NEW.** Loss tests (incl. EPR / pressure forbidden) | NEW |
| `scripts/train_cyclelayer_v3_thermal_aux_smoke.py` | **NEW.** Tiny DS02 smoke training | NEW |
| `scripts/evaluate_cyclelayer_v3_theta_diagnostics.py` | **NEW.** Post-hoc θ correlation evaluation | NEW |

## Model semantics

### ParamHead_phys (5 thetas, factor representation)

* Input: encoded sensor / ops features (B, F)
* Output: `theta_phys` with shape `(B, [T,] 5)` in order
  `[eta_fan, eta_lpc, eta_hpc, eta_hpt, eta_lpt]`
* Activation: `sigmoid` scaled to `[0.85, 1.00]`
* Bias init: chosen so initial θ ≈ 0.99 (slightly degraded from healthy,
  not at midpoint)
* No supervised loss on θ_phys

### AuxHealthHead (LPT_flow_mod)

* Output: `lpt_flow_pred` scalar in `[-0.05, 0.02]` (delta around 0)
* Activation: `tanh` scaled
* Bias init: lpt_flow_pred ≈ 0.0 at init
* Supervised L_aux with small weight; normalised MSE

### Target-sensor masking

* Applied **AFTER** sensor normalization
* Masked normalized values become 0.0 (= per-sensor mean in raw space)
* During training: mask `T24`, `T30`, `T50` columns independently with
  `p = 0.5` (because these are physics-loss targets — direct visibility
  would create a sensor leak)
* During eval: no masking
* Unit test verifies normalised value is 0.0 in masked positions

### Loss components

```
L_total = λ_rul · L_rul
        + λ_temp · L_temp_sensor
        + λ_aux · L_aux_lpt_flow
        + λ_healthy · L_healthy_prior
        + λ_smooth · L_smooth

# initial weights (configurable)
λ_rul     = 1.0
λ_temp    = 0.1
λ_aux     = 0.02
λ_healthy = 0.05
λ_smooth  = 1.0e-3
```

`L_temp_sensor`:
* compares BraytonEngine `T24_K / T30_K / T50_K` against measured values
  in **normalized** space (per-sensor σ computed from train units)
* explicitly **does NOT** include P30 / Ps30 / P50 / EPR
* code asserts that pressure / EPR keys are absent

`L_aux_lpt_flow = MSE((pred − GT) / σ_lpt_flow)`, σ from train split or
config default 0.02.

`L_healthy_prior`:
* applied only to samples with `RUL > healthy_rul_threshold` (default 80)
* `mean(||θ_phys − 1||²)` — weak weight to avoid θ collapsing to constant

`L_smooth`: temporal Δ² over time axis of θ_phys (and AuxHead optionally).

## Experiment matrix (planned — not run in this turn)

| Run | Physics aux | θ → RUL | AuxHead → RUL | Notes |
|---|---|---|---|---|
| **A** baseline | no | no | no | Standard encoder + prognostics. Reference RUL accuracy. |
| **B** physics aux only | YES | no (detached) | no (detached) | Tests whether θ emerges from sensor consistency alone. |
| **C** physics + θ → RUL | YES | yes | yes (detached default) | Tests whether θ adds prognostic value. |
| **D** ablation | YES (or scrambled) | shuffled / frozen / random θ | — | Control: tests whether θ *structure* matters or any latent works. |

## Success criteria (research-scoped — not gates for adoption)

| Tier | Criterion | Note |
|---|---|---|
| Minimum | no NaN; RUL RMSE not worse than baseline by > 5 %; θ not fully saturated; θ trajectories smooth | sanity |
| Good | `Pearson(θ_η_hpt − 1, HPT_eff_mod) > 0.4`; `Pearson(θ_η_lpt − 1, LPT_eff_mod) > 0.4`; RMSE comparable to baseline | meaningful latent |
| Very good | correlations > 0.6; RUL RMSE improves; worst-case over-estimation reduced | publishable result |
| Stretch only | correlations > 0.7 | **NOT** a hard gate for V3.1b |

These are research-scoped success bands. Hard adoption gates are not in
V3.1b scope; adoption happens via V4 if at all.

## Stop points

After each implementation step:

* Run unit tests; require green.
* Optional: run smoke training (1-2 epochs, small subset) only on local
  machine; full training requires explicit approval.
* Produce status report with: changed files, test results, smoke results
  (if run), first θ diagnostics, blockers.

## Out of scope

* Adoption of any C0d-style "documented C-MAPSS" parameter set within
  the explicit-closure V3.1b architecture
* EPR / pressure validation as a V3.1b acceptance gate
* DS02 cycle-level cross-check as a V3.1b acceptance gate
* Full training without approval
