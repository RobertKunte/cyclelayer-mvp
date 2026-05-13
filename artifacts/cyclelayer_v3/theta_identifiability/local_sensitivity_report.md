# Local sensitivity — V3.1b θ identifiability (Task 2)

*Read-only diagnostic.  ADR-0013.*

## Setup

* operating points: FC02 design-ish anchor + 10 random DS02 rows (if DS02 available locally)
* θ baseline: `[0.95, 0.95, 0.95, 0.95, 0.95]` (mildly degraded)
* sensitivity: exact via `torch.autograd`
* normalized elasticity: `(θ / output) · (∂output / ∂θ)`

## Per-output mean elasticity (across all points)

| output | elasticity_eta_fan | elasticity_eta_lpc | elasticity_eta_hpc | elasticity_eta_hpt | elasticity_eta_lpt |
|---|---|---|---|---|---|
| T24 | -1.367e-01 | -2.013e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| T30 | -1.327e-01 | -1.959e-01 | -5.529e-01 | 0.000e+00 | 0.000e+00 |
| T45 | -5.131e-02 | -7.557e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| T50 | 3.573e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| P30 | 2.460e-02 | 1.735e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| P45 | 1.512e-01 | 2.051e-01 | 1.622e+00 | 2.378e+00 | 0.000e+00 |
| P50 | 2.196e+00 | 5.832e-01 | 1.622e+00 | 2.378e+00 | 2.697e+00 |
| PR_hpt | -1.266e-01 | -1.877e-01 | -1.622e+00 | -2.378e+00 | 0.000e+00 |
| PR_lpt | -2.045e+00 | -3.781e-01 | 0.000e+00 | 0.000e+00 | -2.697e+00 |
| EPR | 2.196e+00 | 5.832e-01 | 1.622e+00 | 2.378e+00 | 2.697e+00 |

## Per-θ max |elasticity|: temperature outputs vs pressure outputs

| θ | max |elasticity| on {T24,T30,T45,T50} | max |elasticity| on {P30,P45,P50,PR_hpt,PR_lpt,EPR} | T-identifiable? (|el| > 1e-2) |
|---|---|---|---|
| eta_fan | 3.573e-01 | 2.196e+00 | YES |
| eta_lpc | 2.013e-01 | 5.832e-01 | YES |
| eta_hpc | 5.529e-01 | 1.622e+00 | YES |
| eta_hpt | 0.000e+00 | 2.378e+00 | **NO** |
| eta_lpt | 0.000e+00 | 2.697e+00 | **NO** |

## Interpretation

* If `eta_fan / eta_lpc / eta_hpc` show meaningful elasticity on T24/T30 and `eta_hpt / eta_lpt` do NOT,
  the V3.1b closure architecturally rules out HPT/LPT efficiency identification from a T-only loss.
* HPT/LPT θ elasticity on pressure outputs (P45/P50/PR_hpt/PR_lpt/EPR) being nonzero is the
  *expected* physical behavior — and the reason V4 with pressure loss is required to identify them.

See `local_sensitivity_heatmap.png` for the elasticity grid.
