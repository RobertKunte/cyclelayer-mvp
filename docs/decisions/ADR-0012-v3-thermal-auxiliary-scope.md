# ADR-0012 — V3.1b re-scope: differentiable thermal auxiliary layer (not full cycle model)

* Status: **Accepted**
* Decision owner: Robert Kunte
* Supersedes scope assumptions of V3.1a and V3.1b § C.3 ("Phase C — three-stage validation including EPR")

## Context

V3.1b was originally specified as a full differentiable thermodynamic cycle
model with EPR (P50 / P2) as a hard validation gate at FC02 (UserGuide
NASA/TM-2007-215026 Tab. 1.4) and across the 13 remaining UserGuide flight
conditions (C1).

Four sequential read-only diagnostics (C0, C0b, C0c, C0d) systematically
explored whether the V3.1b cycle architecture can satisfy EPR plausibility:

* **C0** (single-parameter `Wc_fan_design` anchor at FC02) — T45 anchor passes
  at 0.88 % rel err; EPR fails at 68.9 %.
* **C0b** (PR-split sweep, 384 points; eta in 0.88–0.92 band) — 0 / 384
  feasible candidates; minimum EPR err within OPR ∈ [25, 35] is 35.7 %.
* **C0c** (BPR + Wc_fan + OPR-constrained PR split, 12,420 forward passes;
  eta in 0.88–0.92) — 8,468 / 12,420 satisfy combined T45/EPR/T/FAR plausibility
  windows. BUT every candidate that satisfies the joint window does so with
  a `Wc_fan_design` that disagrees with the documented C-MAPSS reference
  value, and the choice itself reduces to *fitting the model to FC02 rather
  than initialising the model from documented references* — i.e. tuning under
  another name.
* **C0d** (four named candidates initialised from the documented C-MAPSS
  reference engine: BPR=8.4, Wc_fan_design=1658 kg/s, OPR=36, documented
  PR / eta table) — **none of A / B / C is physically plausible at FC02**.
  All three produce `T50 ≈ 482 K` (below the 650 K plausibility floor) and
  closure-determined `PR_lpt ≈ 22` (real LPTs are 4–6). The structural
  reason: V3.1b's explicit-closure architecture (with measured Nf, Nc, Wf
  as inputs and fixed eta) forces the LPT to extract ~100 MW of fan-shaft
  work from only 180 kg/s of hot core flow at FC02 — an unphysical
  expansion. Lowering turbine eta below 0.88 would mathematically reduce
  the gap but is not a realistic component value.

## Decision

V3.1b is **re-scoped** from "full cycle model with EPR validation" to a
**differentiable thermal auxiliary physics layer**.

### What V3.1b is now

* A differentiable Brayton-cycle module whose outputs are used **only** as a
  thermal-consistency regulariser for the encoder and a structured latent
  for the prognostics head.
* Training uses **only temperature-sensor consistency** on T24, T30, T50.
* Theta_phys (5 efficiency factors) are learned **without health-parameter
  ground truth** — driven by sensor consistency, RUL loss, and weak
  healthy / smoothness priors. Theta-GT correlations are reported post-hoc
  as evaluation diagnostics.
* The single supervised AuxHealthHead for LPT_flow_mod remains (small
  weight, normalised loss).
* T45 / T48, P30 / Ps30, P50 / EPR may be **logged** as diagnostics where
  available, but are **not** used in any loss for V3.1b.

### What V3.1b is no longer

* **Not** a full pressure / EPR-validated cycle model.
* **Not** validated against the UserGuide 13-FC C1 acceptance gate as a
  pressure validation step.
* **Not** a candidate for adoption of the documented C-MAPSS reference
  engine numbers (BPR=8.4, OPR=36, Wc_fan_design=1658, component PR/eta
  table) within the current explicit-closure architecture.

### Why not "demote EPR"

EPR is a real engineering acceptance criterion for a full cycle model. We
are **not** demoting EPR's engineering importance. We are explicitly
declaring that V3.1b is not a full cycle model and therefore EPR validation
is not in V3.1b's scope.

EPR remains a **hard validation target for V4**.

### Main experiment question (V3.1b research scope)

> Does the simple V3.1b thermal auxiliary layer improve prognostics
> and / or learn meaningful efficiency-like latent health parameters
> (especially theta_eta_hpt, theta_eta_lpt) even though it is not an
> EPR-validated full cycle solver?

## V4 roadmap (out of V3.1b scope)

V4 will introduce the cycle-physics elements that V3.1b cannot represent:

* Split fan / core flow paths
* Bypass nozzle
* Core nozzle
* Flow matching (corrected flows balanced across components)
* Map consistency
* Shaft balances solved iteratively (not by closure on prescribed work)
* Nf / Nc / Wf reconsidered as targets / balance variables rather than
  fixed hard inputs where appropriate
* EPR validation against UserGuide 13 FCs as a hard gate
* DS02 cycle-level cross-check after EPR / pressure gate is green

## Implementation consequences

* New training config (`configs/cyclelayer_v3_thermal_aux.yaml`) labelled
  `thermal_regularizer_v3_1b_not_epr_validated`.
* Loss module (`src/cyclelayer/losses/cyclelayer_v3_losses.py`) **asserts**
  pressure / EPR losses are disabled. No way to enable them within V3.1b
  without an explicit code change.
* Existing V3 modules (BraytonEngine, MapCoefficients, units, stations,
  pressure_proxies, NCMAPSSV3Dataset) remain unchanged.
* Existing legacy modules (`brayton_cycle`, `cycle_layer`, `physresnet`,
  `encoder`, `baselines`, `prognostics`) remain frozen.
* All 108 existing tests remain green.

## Status of C0 → C0d worksheets

The four C0 diagnostic scripts and their `artifacts/` outputs are
**preserved as research record**:

* `scripts/c0_anchor_check.py`
* `scripts/c0_cycle_plausibility_report.py`
* `scripts/c0b_pr_split_feasibility.py`
* `scripts/c0c_bpr_wcfan_feasibility.py`
* `scripts/c0d_cmapss_documented_design_check.py`

They are **not part of V3.1b training**. They document why pressure
validation was deferred to V4.

## Decision date

2026-05-12

## References

* [docs/CycleLayer_V3.1b_Master_Spec.md](../CycleLayer_V3.1b_Master_Spec.md)
* [docs/V3_thermal_auxiliary_plan.md](../V3_thermal_auxiliary_plan.md)
* C0 → C0d diagnostic artifacts under `artifacts/cyclelayer_v3/`
* NASA/TM-2007-215026 (Frederick et al., C-MAPSS User Guide) — *source
  verification still pending; PDF not in repo*
* Saxena et al. 2008, "Damage Propagation Modeling..." — confirms 90 K lbf
  thrust class and operating envelope only.
