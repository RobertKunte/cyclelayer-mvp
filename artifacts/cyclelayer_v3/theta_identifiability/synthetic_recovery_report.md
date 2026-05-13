# Synthetic theta recovery — V3.1b (Task 7)

*Read-only.  ADR-0013.  Synthetic data only — no DS02 / no health GT.*

## Setup

* Operating point: FC02 (SLS, M=0.25, TRA=100 %)
* True theta        : `[0.9300000071525574, 0.9200000166893005, 0.9100000262260437, 0.9399999976158142, 0.949999988079071]`
* Initial theta     : `[0.9800000190734863, 0.9800000190734863, 0.9800000190734863, 0.9800000190734863, 0.9800000190734863]` (near healthy)
* Optimiser     : Adam, lr=2e-2, 3000 steps
* theta kept in `[0.85, 1.00]` via sigmoid-scaling of an unconstrained raw param

## Results

| theta | true | recovered_T | abs_err_T | recovered_T_P | abs_err_T_P |
|---|---|---|---|---|---|
| eta_fan | 0.93000 | 0.93000 | 0.00000 | 0.93081 | 0.00081 |
| eta_lpc | 0.92000 | 0.92000 | 0.00000 | 0.91900 | 0.00100 |
| eta_hpc | 0.91000 | 0.91000 | 0.00000 | 0.91024 | 0.00024 |
| eta_hpt | 0.94000 | 0.98000 | 0.04000 | 0.94477 | 0.00477 |
| eta_lpt | 0.95000 | 0.98000 | 0.03000 | 0.94462 | 0.00538 |

## Recovery summary (|theta_recovered − theta_true| < 0.01 ?)

| theta | Case A (T only) | Case B (T+P) |
|---|---|---|
| `eta_fan` | YES | YES |
| `eta_lpc` | YES | YES |
| `eta_hpc` | YES | YES |
| `eta_hpt` | **NO** | YES |
| `eta_lpt` | **NO** | YES |


## Final loss

* Case A (T only):  0.000e+00
* Case B (T+P):     7.214e-08

## Plot

`synthetic_recovery.png` — loss curve + bar chart of recovered vs true theta.

## Decision (ADR-0013)

* If `theta_η_hpt` and `theta_η_lpt` are **not recovered** under Case A but **are**
  recovered under Case B, then V3.1b's L_temp cannot identify HPT/LPT
  efficiency: the architecture is the bottleneck, not training data.

* Recovered in A but not B should not happen physically; if it does the
  test setup is malformed (investigate).

This is a CONSTRUCTIVE proof of the identifiability boundary — independent
of N-CMAPSS, RUL-axis time leakage, or training quality.
