# V3.1b thermal-aux θ diagnostics

*Post-hoc evaluation only — θ_phys was NOT trained against GT.*
See [ADR-0012](../../decisions/ADR-0012-v3-thermal-auxiliary-scope.md).

## Sample counts

| split | windows |
|---|---|
| val  | 200 |
| test | 200 |

## Correlations (post-hoc, val split)

| metric | Pearson | Spearman | N |
|---|---|---|---|
| `(θ_η_hpt − 1)` vs `HPT_eff_mod` | 0.110 | 0.074 | 200 |
| `(θ_η_lpt − 1)` vs `LPT_eff_mod` | 0.083 | 0.039 | 200 |
| `lpt_flow_pred` vs `LPT_flow_mod` (supervised) | 0.005 | 0.018 | 200 |

## Correlations (post-hoc, test split)

| metric | Pearson | Spearman | N |
|---|---|---|---|
| `(θ_η_hpt − 1)` vs `HPT_eff_mod` | 0.199 | 0.109 | 200 |
| `(θ_η_lpt − 1)` vs `LPT_eff_mod` | 0.051 | 0.020 | 200 |
| `lpt_flow_pred` vs `LPT_flow_mod` (supervised) | -0.089 | -0.090 | 200 |

## Partial correlations (controlling for alt, Mach, TRA, T2, Nf, Nc, Wf — test)

| metric | partial Pearson | N |
|---|---|---|
| `(θ_η_hpt − 1)` vs `HPT_eff_mod` | -0.065 | 200 |
| `(θ_η_lpt − 1)` vs `LPT_eff_mod` | 0.121 | 200 |

## θ saturation (fraction at bound) — test split

* near lower bound 0.85: 0.000
* near upper bound 1.00: 0.000

## Plots

* `01_theta_vs_GT_scatter.png` — θ-delta and AuxHead vs N-CMAPSS GT
* `02_theta_vs_RUL.png` — all five θ + AuxHead vs RUL, coloured by unit
* `03_theta_per_unit.png` — per-unit θ trajectories vs GT trend
