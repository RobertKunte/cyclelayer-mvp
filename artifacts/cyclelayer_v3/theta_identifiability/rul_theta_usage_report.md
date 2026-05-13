# RUL-head θ usage — V3.1b (Task 6)

*Read-only.  ADR-0013.*

* Checkpoint: `C:\Users\rober\OneDrive\Dokumente\GitHub\cyclelayer-mvp\artifacts\cyclelayer_v3\thermal_aux_smoke`
* Test windows used: 20,000  (stride 20, cap 20k)

## RUL metrics under θ perturbation

| variant | RMSE | MAE | bias | low_RUL_RMSE | low_RUL_bias | low_RUL_overest_frac | n_low_RUL | delta_RMSE_vs_real | pct_RMSE_change |
|---|---|---|---|---|---|---|---|---|---|
| real | 24.2398 | 19.9573 | 15.0052 | 33.2045 | 32.0559 | 1.0000 | 9634 | 0.0000 | 0.0000 |
| shuffle_batch | 24.2401 | 19.9567 | 14.9973 | 33.2062 | 32.0606 | 1.0000 | 9634 | 0.0003 | 0.0013 |
| shuffle_within_unit | 24.2473 | 19.9609 | 15.0041 | 33.2132 | 32.0637 | 1.0000 | 9634 | 0.0075 | 0.0311 |
| shuffle_across_units | 24.2401 | 19.9567 | 14.9973 | 33.2062 | 32.0606 | 1.0000 | 9634 | 0.0003 | 0.0013 |
| constant_healthy | 24.2398 | 19.9573 | 15.0052 | 33.2045 | 32.0559 | 1.0000 | 9634 | 0.0000 | 0.0001 |
| constant_lo | 24.2568 | 19.9690 | 15.0351 | 33.2317 | 32.0844 | 1.0000 | 9634 | 0.0169 | 0.0699 |

Real-θ baseline RMSE: **24.2398**

## Decision rule

> If all θ perturbations change RMSE by < 0.5 % **and** prognostics-head
> weight norms / `∂L_rul/∂θ` are near zero, the RUL head is ignoring θ_phys.

| θ | weight norm (first PrognosticsHead Linear) | ∂L_rul/∂θ (batch grad sum) |
|---|---|---|
| `eta_fan` | 3.5168e+00 | 1.5207e+01 |
| `eta_lpc` | 2.9812e+00 | 1.3468e+01 |
| `eta_hpc` | 3.1123e+00 | 7.4424e+00 |
| `eta_hpt` | 3.4191e+00 | 1.0057e+01 |
| `eta_lpt` | 2.9410e+00 | 7.0726e+00 |
| `aux_lpt_flow` | 2.8888e+00 | — |


## Largest |ΔRMSE| across perturbations

* `max |ΔRMSE / RMSE_real|` = **0.0699 %**
* If this is < 0.5 % → RUL head is **not** using θ.

## Plot

See `rul_theta_usage.png` for ΔRMSE bars.
