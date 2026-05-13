# C0b PR-split feasibility diagnostic — UserGuide FC02

*Read-only sweep diagnostic. No YAML written. No DS02 access. No optimization.*
*Top candidates are reporting-only; no parameter set is automatically adopted.*

## Frozen state (unchanged for this sweep)

- `Wc_fan_design = 900.0 kg/s` (provisional thermal anchor from C0)
- `eta_design_hpt = 0.9`, `eta_design_lpt = 0.92` (in 0.88–0.92 component band)
- `eta_design_fan/lpc/hpc = 0.92/0.9/0.88`
- `BPR = 5.5`

## Sweep grid (3-D, 384 combinations)

- PR_fan ∈ [1.4, 1.45, 1.5, 1.55, 1.6, 1.65]
- PR_lpc ∈ [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
- PR_hpc ∈ [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]

## Feasibility filter

A grid point is **feasible** if all of the following hold:

- OPR ∈ [25.0, 35.0]
- T45 rel err < 25.0 %
- EPR rel err < 25.0 %
- T4 ∈ [1300.0, 1900.0] K
- T30 ∈ [600.0, 1000.0] K
- T50 ∈ [700.0, 1100.0] K
- eta_hpt, eta_lpt ∈ [0.88, 0.92] (kept by construction)
- no PR clamp active anywhere

## Sweep result

| Total points | Feasible | Feasibility rate |
|---|---|---|
| 384 | **0** | 0.0 % |

## Per-criterion isolated pass rates

Pass rate of each criterion **considered alone** across the full 384-point grid (highest-restriction first):

- **EPR err < 25.0%** — 11/384 (2.9%)
- **OPR in [25.0, 35.0]** — 158/384 (41.1%)
- **T45 err < 25.0%** — 384/384 (100.0%)
- **T4 in [1300.0, 1900.0] K** — 384/384 (100.0%)
- **T30 in [600.0, 1000.0] K** — 384/384 (100.0%)
- **T50 in [700.0, 1100.0] K** — 384/384 (100.0%)
- **no PR clamp active** — 384/384 (100.0%)

The criterion with the lowest isolated pass rate is the **binding constraint** — the assumption set cannot satisfy it within the realistic eta band, regardless of how PR_fan/lpc/hpc are split.

## Top-5 candidates by criterion (REPORTING ONLY — do not auto-adopt)

### 1) Minimum EPR error (within strict feasibility window)

| (none) |
|---|


### 2) Minimum combined T45/EPR error (within strict feasibility window)

| (none) |
|---|


### 3) Closest CFM56-like OPR window [30.0, 33.0] (within strict feasibility)

| (none) |
|---|


## Near-feasible references (diagnostic — when strict feasibility is empty)

These ignore some of the feasibility filters; useful for understanding where the cycle wants to sit.

### A) Smallest EPR error over the entire grid (no filters)

| PR_fan | PR_lpc | PR_hpc | OPR | T45_err_pct | EPR_err_pct | T4_K | T30_K | T50_K | PR_hpt | PR_lpt |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.65 | 1.30 | 7.00 | 14.89 | 2.77 | 11.56 | 1391.7 | 680.9 | 829.7 | 2.62 | 3.87 |
| 1.65 | 1.40 | 7.00 | 16.04 | 2.13 | 14.89 | 1405.4 | 697.0 | 829.7 | 2.66 | 3.99 |
| 1.65 | 1.30 | 8.00 | 17.02 | 2.77 | 17.73 | 1416.0 | 709.3 | 829.7 | 2.84 | 3.87 |
| 1.65 | 1.50 | 7.00 | 17.18 | 1.52 | 18.03 | 1418.5 | 712.2 | 829.7 | 2.70 | 4.11 |
| 1.60 | 1.30 | 7.00 | 14.45 | 3.03 | 19.48 | 1386.3 | 674.6 | 844.8 | 2.61 | 3.53 |


### B) Smallest EPR error within OPR ∈ [25.0, 35.0] (no other filters)

| PR_fan | PR_lpc | PR_hpc | OPR | T45_err_pct | EPR_err_pct | T4_K | T30_K | T50_K | PR_hpt | PR_lpt |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.65 | 1.70 | 9.00 | 25.04 | 0.38 | 35.73 | 1493.4 | 799.9 | 829.7 | 3.25 | 4.33 |
| 1.65 | 1.40 | 11.00 | 25.20 | 2.13 | 36.28 | 1493.6 | 800.2 | 829.7 | 3.53 | 3.99 |
| 1.65 | 1.30 | 12.00 | 25.53 | 2.77 | 37.03 | 1495.7 | 802.6 | 829.7 | 3.66 | 3.87 |
| 1.65 | 2.00 | 8.00 | 26.19 | 1.17 | 37.75 | 1503.6 | 811.8 | 829.7 | 3.12 | 4.64 |
| 1.65 | 1.60 | 10.00 | 26.19 | 0.93 | 37.94 | 1502.5 | 810.5 | 829.7 | 3.43 | 4.22 |


## CSV exports

- `all_candidates.csv` — full 384-row sweep with every metric
- `feasible_candidates.csv` — the 0 feasible rows only

## Plot index

| # | Plot | File |
|---|---|---|
| 9  | EPR-error heatmaps faceted by PR_fan | `09_EPR_err_heatmaps.png` |
| 10 | Feasibility scatter OPR vs EPR error (color = T45 err) | `10_feasible_scatter_OPR_vs_EPR.png` |
| 11 | Station total-pressure profile, top-5 candidates | `11_station_pressure_top5.png` |
| 12 | Station total-temperature profile, top-5 candidates | `12_station_temperature_top5.png` |

---

*Stop. No automatic parameter selection. Awaiting Robert review.*
