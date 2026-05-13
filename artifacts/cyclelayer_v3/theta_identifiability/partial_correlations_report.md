# Partial correlations — V3.1b θ identifiability (Task 4)

*Read-only.  ADR-0013.  No DS02 tuning.*

* Checkpoint: `C:\Users\rober\OneDrive\Dokumente\GitHub\cyclelayer-mvp\artifacts\cyclelayer_v3\thermal_aux_smoke`
* Test windows analysed: 1,253,596
* Test units: [11, 14, 15]

## Correlation table (rows = θ-GT pair, columns = control set; values = Pearson r)

| pair | none | RUL | cycle | ops | RUL+ops | cycle+ops |
|---|---|---|---|---|---|---|
| lpt_flow_pred vs LPT_flow_mod | -0.033 | -0.013 | -0.033 | -0.073 | -0.051 | -0.073 |
| theta_hpt_damage vs HPT_eff_mod | -0.008 | -0.001 | -0.008 | -0.010 | -0.007 | -0.010 |
| theta_hpt_delta vs HPT_eff_mod | 0.008 | 0.001 | 0.008 | 0.010 | 0.007 | 0.010 |
| theta_lpt_damage vs LPT_eff_mod | -0.010 | 0.001 | -0.010 | -0.028 | -0.015 | -0.028 |
| theta_lpt_delta vs LPT_eff_mod | 0.010 | -0.001 | 0.010 | 0.028 | 0.015 | 0.028 |

## Per-pair verdict

| pair | raw r | partial r (RUL) | partial r (RUL+ops) | verdict |
|---|---|---|---|---|
| theta_hpt_delta vs HPT_eff_mod | 0.008 | 0.001 | 0.007 | WEAK / INCONSISTENT |
| theta_lpt_delta vs LPT_eff_mod | 0.010 | -0.001 | 0.015 | WEAK / INCONSISTENT |
| theta_hpt_damage vs HPT_eff_mod | -0.008 | -0.001 | -0.007 | WEAK / INCONSISTENT |
| theta_lpt_damage vs LPT_eff_mod | -0.010 | 0.001 | -0.015 | WEAK / INCONSISTENT |
| lpt_flow_pred vs LPT_flow_mod | -0.033 | -0.013 | -0.051 | WEAK / INCONSISTENT |


## Plots

* `theta_vs_gt_raw.png` — raw scatter, colour = RUL
* `theta_vs_gt_residualized.png` — same scatter after RUL+ops residualisation
* `theta_vs_cycle_per_unit.png` — per-unit time trajectories (model on top, GT below)
* `theta_damage_vs_gt_per_unit.png` — "damage" space (1−θ) vs negated GT

## Decision rule (ADR-0013)

* `raw |r| > 0.6` AND `partial |r| < 0.2` → **time/degradation-axis artifact**.
* `partial |r| > 0.4` with same sign as raw → **robust signal**, more training warranted.
* `per_unit` correlations vary widely in sign → global Pearson alone is **misleading**.
