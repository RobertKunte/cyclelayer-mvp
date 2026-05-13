# C0 Cycle Plausibility Report — UserGuide FC02

*Read-only diagnostic. No YAML written. No DS02 access. No optimization loop.*

## State (frozen for this report)

- Single picked parameter: `Wc_fan_design = 900.0 kg/s` (provisional thermal anchor)
- Fixed PR_design_fan / lpc / hpc = 1.6 / 2.0 / 12.0
- Fixed eta_design fan / lpc / hpc / hpt / lpt = 0.92 / 0.9 / 0.88 / 0.9 / 0.92
- BPR = 5.5, PI_BURN = 0.04

## FC02 inputs

- alt = 0.0 ft, M = 0.25, TRA = 100.0 %
- T2 (post-ram) = 525.15 °R = 291.75 K
- P2 (post-ram) = 15.35 psia
- Nf = 2403.0 rpm, Nc = 9084.0 rpm, Wf = 7.085 pps
- Reference (UserGuide): T48 = 2083.0 °R, EPR = 1.261

## Current FC02 residuals

| Quantity | Predicted | Reference | Rel err [%] | 25 %-band |
|---|---|---|---|---|
| **T45 / T48_proxy** (primary anchor) | 2101.25 °R | 2083.00 °R | **0.88** | PASS |
| **EPR = P50/P2** (plausibility check) | 2.130 | 1.261 | **68.92** | FAIL |

Conservation (closure): mass_inlet = 0.00e+00, mass_combust = 0.00e+00, HPT shaft res = 8.00e+00, LPT shaft res = 4.00e+00. PR-clamps not active.

## Plausibility check

| Metric | Value | Realistic window | Verdict |
|---|---|---|---|
| FAR = Wf / m_core | 0.0224 | 0.018–0.03 | **plausible** |
| T4 (TIT)         | 1587.6 K | 1300.0–1900.0 K | **plausible** |
| T45              | 1167.4 K | 1000.0–1500.0 K | **plausible** |
| T50              | 844.8 K | 700.0–1100.0 K | **plausible** |
| OPR = P30 / P2   | 38.11 | 20.0–45.0 | **plausible** |
| EPR = P50 / P2   | 2.130 | 1.1–1.5 | **OUT OF RANGE** |
| m_in             | 934.2 kg/s | 50.0–1500.0 | **plausible** |

## PR_hpc sensitivity (diagnostic, not tuning)

| PR_hpc | T45 rel err [%] | EPR rel err [%] | T30 [K] | OPR |
|---|---|---|---|---|
| 7.0 | 0.9 | 41.0 | 772 | 22.2 |
| 8.0 | 0.9 | 47.9 | 804 | 25.4 |
| 9.0 | 0.9 | 54.0 | 834 | 28.6 |
| 10.0 | 0.9 | 59.5 | 861 | 31.8 |
| 11.0 | 0.9 | 64.4 | 886 | 34.9 |
| 12.0 | 0.9 | 68.9 | 910 | 38.1 |


## Most likely cause of EPR mismatch

**PR_hpc alone does not close the EPR gap.** Even at PR_hpc = 7.0 the EPR rel err is 41.0 %. Likely additional factor(s): combustor pressure drop PI_BURN (currently 0.04) understates the real pressure loss, OR PR_lpc / PR_fan also need revision, OR the dual-turbine eta_design pair (0.90 / 0.92) is at the upper edge of the realistic 0.88-0.92 component band. None of these are touched in this report.

## Plot index

| Plot | File |
|---|---|
| 1. T-s diagram                          | `01_T_s_diagram.png` |
| 2. h-s diagram                          | `02_h_s_diagram.png` |
| 3. T vs station                         | `03_T_vs_station.png` |
| 4. P/P2 vs station (log)                | `04_P_over_P2_vs_station_log.png` |
| 5. EPR decomposition waterfall          | `05_EPR_waterfall.png` |
| 6. Spool work balance                   | `06_spool_work_balance.png` |
| 7. FAR / T4 / mass-flow summary         | `07_FAR_T4_massflow_summary.png` |
| 8. PR_hpc sensitivity diagnostic        | `08_PR_hpc_sensitivity.png` |

---

*Stop. Awaiting Robert review before any YAML change.*
