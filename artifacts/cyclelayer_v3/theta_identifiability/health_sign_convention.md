# N-CMAPSS DS02 health-parameter sign convention (Task 5)

*Read-only.  No model used.*

## Splits

* dev  units: [2, 5, 10, 16, 18, 20]
* test units: [11, 14, 15]

## Per-unit early/mid/late statistics

CSV: `health_sign_convention.csv` (full row-level).

## Per-column life-direction summary (averaged across units within split)

| split | column | mean_early | mean_mid | mean_late | delta_late_early | direction |
|---|---|---|---|---|---|---|
| dev | fan_eff_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| test | fan_eff_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| dev | fan_flow_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| test | fan_flow_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| dev | LPC_eff_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| test | LPC_eff_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| dev | LPC_flow_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| test | LPC_flow_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| dev | HPC_eff_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| test | HPC_eff_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| dev | HPC_flow_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| test | HPC_flow_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| dev | HPT_eff_mod | -0.00058 | -0.00188 | -0.01230 | -0.01172 | decreases |
| test | HPT_eff_mod | -0.00071 | -0.00164 | -0.00968 | -0.00897 | decreases |
| dev | HPT_flow_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| test | HPT_flow_mod | 0.00000 | 0.00000 | 0.00000 | 0.00000 | flat |
| dev | LPT_eff_mod | -0.00021 | -0.00089 | -0.00704 | -0.00683 | decreases |
| test | LPT_eff_mod | -0.00041 | -0.00124 | -0.00810 | -0.00769 | decreases |
| dev | LPT_flow_mod | -0.00035 | -0.00096 | -0.00551 | -0.00516 | decreases |
| test | LPT_flow_mod | -0.00055 | -0.00161 | -0.01384 | -0.01329 | decreases |

## Interpretation (test split)

- **fan_eff_mod**: early≈0.0000, late≈0.0000, Δ=+0.0000 → flat over life (near-zero early (delta around 0))
- **fan_flow_mod**: early≈0.0000, late≈0.0000, Δ=+0.0000 → flat over life (near-zero early (delta around 0))
- **LPC_eff_mod**: early≈0.0000, late≈0.0000, Δ=+0.0000 → flat over life (near-zero early (delta around 0))
- **LPC_flow_mod**: early≈0.0000, late≈0.0000, Δ=+0.0000 → flat over life (near-zero early (delta around 0))
- **HPC_eff_mod**: early≈0.0000, late≈0.0000, Δ=+0.0000 → flat over life (near-zero early (delta around 0))
- **HPC_flow_mod**: early≈0.0000, late≈0.0000, Δ=+0.0000 → flat over life (near-zero early (delta around 0))
- **HPT_eff_mod**: early≈-0.0007, late≈-0.0097, Δ=-0.0090 → decreases over life (near-zero early (delta around 0))
- **HPT_flow_mod**: early≈0.0000, late≈0.0000, Δ=+0.0000 → flat over life (near-zero early (delta around 0))
- **LPT_eff_mod**: early≈-0.0004, late≈-0.0081, Δ=-0.0077 → decreases over life (near-zero early (delta around 0))
- **LPT_flow_mod**: early≈-0.0005, late≈-0.0138, Δ=-0.0133 → decreases over life (near-zero early (delta around 0))

## Sign-convention conclusion

For health parameters that **decrease** from early to late life
(`mean_late < mean_early`), the documented N-CMAPSS convention is:

* `*_eff_mod` and `*_flow_mod` are **delta around 0** (healthy ≈ 0,
  degraded → negative).

Therefore the **expected physical sign** of:

* `Pearson(θ_phys − 1.0, *_eff_mod)`  → **positive** when the model
  correctly identifies degradation (both move toward negative together
  during life progression).

A *negative* observed Pearson (the experiment-matrix run reported
−0.85 for θ_η_hpt vs HPT_eff_mod) means **either** the model has
anti-learned the relationship, **or** the correlation is a time-axis
artifact whose sign is determined by other latent factors.

## Important guard (do NOT flip sign cosmetically)

Per ADR-0013: "Do not simply flip θ sign to improve correlation.
Establish the physical meaning first."  This script confirms the
physical sign expectation; it does not modify the model.
