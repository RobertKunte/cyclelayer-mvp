# Loss gradient paths — V3.1b θ identifiability (Task 3)

*Read-only diagnostic. ADR-0013.*

## Setup

* `CycleLayerV3` built from `configs/cyclelayer_v3_thermal_aux.yaml`
* Synthetic batch (B=16, T=50) + (if local DS02 available) one real DS02 batch
* Each loss component (L_rul / L_temp / L_aux / L_healthy / L_smooth / L_total)
  computed STANDALONE and `∂/∂θ_phys` taken via `torch.autograd.grad`
* Reported value = `Σ_batch |grad|` per (loss_term, θ) cell

## Mean |grad| of each loss term w.r.t. each θ

| loss_term | eta_fan | eta_lpc | eta_hpc | eta_hpt | eta_lpt |
|---|---|---|---|---|---|
| L_rul | 1.919e+01 | 1.111e+01 | 1.532e+01 | 1.392e+01 | 1.478e+01 |
| L_temp | 4.265e+01 | 4.404e+01 | 3.105e+01 | 0.000e+00 | 0.000e+00 |
| L_aux | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| L_healthy | 2.000e-03 | 2.000e-03 | 2.000e-03 | 2.000e-03 | 2.000e-03 |
| L_smooth | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| L_total | 2.033e+01 | 1.144e+01 | 1.561e+01 | 1.392e+01 | 1.478e+01 |

## V3.1b's critical question: ∂L_temp / ∂θ

| θ | grad |
|---|---|
| `eta_fan` | 4.265e+01  (active gradient) |
| `eta_lpc` | 4.404e+01  (active gradient) |
| `eta_hpc` | 3.105e+01  (active gradient) |
| `eta_hpt` | 0.000e+00  (rel < 1e-3; abs < 1e-8) |
| `eta_lpt` | 0.000e+00  (rel < 1e-3; abs < 1e-8) |

## Decision rule (ADR-0013)

> If `L_temp` gradient norms for `θ_η_hpt` and `θ_η_lpt` are ~0 while
> compressor θ gradients are nonzero, then HPT/LPT efficiency θ are not
> identifiable from the current temperature loss.

* `θ_η_hpt`  L_temp grad : **0.000e+00**
* `θ_η_lpt`  L_temp grad : **0.000e+00**

The other components (L_aux, L_healthy, L_smooth) may still provide a
weak gradient, but they do not constrain the *physical mapping* between
θ and HPT/LPT efficiency.

See `loss_gradient_paths.png` for the full grid.
