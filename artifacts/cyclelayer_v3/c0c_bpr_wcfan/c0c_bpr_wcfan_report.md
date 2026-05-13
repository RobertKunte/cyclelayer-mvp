# C0c BPR / Wc_fan feasibility diagnostic — UserGuide FC02

*Read-only sweep diagnostic. No YAML written. No DS02 access. No optimizer.*
*No parameter set is automatically adopted. Top candidates are reporting-only.*

## Purpose

After C0/C0b showed that the PR-split alone cannot resolve the EPR mismatch
under CFM56-like assumptions (BPR=5.5, Wc_fan=900, OPR≈38), this diagnostic
tests whether moving to C-MAPSS-90k-class assumptions (BPR≈8.4, Wc_fan≈1200–1500,
OPR∈{30, 33, 36}) closes the gap.

## Frozen state

- eta_design_fan / lpc / hpc = 0.92 / 0.9 / 0.88  (literature defaults, fixed)
- eta_design_hpt, eta_design_lpt **swept** within the realistic 0.88–0.92 component band
- BraytonEngine `use_measured_inlet=True` (P1)

## Sweep grid

- BPR ∈ [5.5, 6.5, 7.5, 8.4, 9.0]
- Wc_fan_design ∈ [900.0, 1050.0, 1200.0, 1300.0, 1400.0, 1500.0] kg/s
- target_OPR ∈ [30.0, 33.0, 36.0]
- PR_fan ∈ [1.55, 1.6, 1.65, 1.7]
- PR_lpc ∈ [1.4, 1.5, 1.6, 1.7, 1.8]
- eta_design_hpt ∈ [0.88, 0.9, 0.92]
- eta_design_lpt ∈ [0.88, 0.9, 0.92]

PR_hpc is computed as `target_OPR / (PR_fan × PR_lpc)`, then rejected pre-forward
if outside [6.0, 14.0].

## Sweep result

| Attempted | Pre-rejected (PR_hpc bound) | Forward-pass total | Plausible | Plausible & near-CMAPSS (BPR ∈ [7.5, 9.0], OPR ∈ [30.0, 36.0]) |
|---|---|---|---|---|
| 16200 | 3780 | **12420** | **8468** | **3525** |

## Per-criterion isolated pass rates

- **EPR err < 25.0%** — 8468/12420 (68.2%)
- **T45 err < 25.0%** — 11592/12420 (93.3%)
- **T4 in [1300.0, 1900.0] K** — 11916/12420 (95.9%)
- **T50 in [650.0, 1100.0] K** — 12033/12420 (96.9%)
- **FAR in [0.015, 0.035]** — 12420/12420 (100.0%)
- **T30 in [600.0, 1000.0] K** — 12420/12420 (100.0%)
- **eta_hpt/lpt in [0.88, 0.92]** — 12420/12420 (100.0%)
- **no PR clamp active** — 12420/12420 (100.0%)

## Q1 — Does BPR≈8.4 reduce EPR vs BPR=5.5?

Across the swept grid, min EPR rel err per BPR:  BPR=5.5 → **0.0 %**, BPR=8.4 → **0.0 %**, BPR=9.0 → **0.0 %**.  Yes — moving from BPR=5.5 to BPR=8.4 reduces the minimum EPR rel err.

## Q2 — Wc_fan_design ∈ [1200, 1500] kg/s plausibility

Among the 8280 points with Wc_fan_design ∈ [1200, 1500] kg/s: **8280 (100 %)** have plausible FAR, **6633 (80 %)** have T45 err < 15.0 %.

## Q3 — Any fully plausible candidate?

**Yes** — 8468 candidate(s) satisfy all plausibility criteria (BPR/OPR/T/FAR/eta/no-clamp + T45 err < 25.0 % + EPR err < 25.0 %). Listed below for review; **no parameter set is automatically adopted**.

## Old baseline reference (BPR=5.5, Wc_fan=900, OPR≈38)

| metric | value |
|---|---|
| BPR | 5.5 |
| Wc_fan_design [kg/s] | 900 |
| OPR | 38.11 |
| T45 err [%] | 0.88 |
| EPR err [%] | 68.92 |
| T4 [K] | 1588 |
| T30 [K] | 910 |
| T50 [K] | 845 |
| FAR | 0.0224 |

## Top candidates — REPORTING ONLY (do not auto-adopt)

### A) Best EPR error in plausible set

| BPR | Wc_fan_design | OPR | PR_fan | PR_lpc | PR_hpc | eta_hpt | eta_lpt | T45_err_pct | EPR_err_pct | T4_K | T30_K | T50_K | FAR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7.50 | 1300.00 | 30.86 | 1.60 | 1.60 | 11.72 | 0.88 | 0.88 | 3.75 | 0.01 | 1593 | 853 | 814 | 0.0239 |
| 9.00 | 1200.00 | 30.35 | 1.70 | 1.70 | 10.38 | 0.92 | 0.92 | 17.45 | 0.01 | 1735 | 849 | 859 | 0.0281 |
| 7.50 | 1300.00 | 37.02 | 1.65 | 1.80 | 12.12 | 0.92 | 0.88 | 5.12 | 0.01 | 1635 | 902 | 794 | 0.0239 |
| 7.50 | 1050.00 | 32.72 | 1.70 | 1.80 | 10.78 | 0.88 | 0.90 | 8.60 | 0.01 | 1645 | 868 | 822 | 0.0251 |
| 8.40 | 1200.00 | 36.44 | 1.65 | 1.60 | 13.64 | 0.88 | 0.88 | 11.56 | 0.01 | 1717 | 896 | 849 | 0.0265 |
| 8.40 | 1400.00 | 37.48 | 1.65 | 1.60 | 13.64 | 0.90 | 0.88 | 11.74 | 0.02 | 1725 | 905 | 841 | 0.0265 |
| 8.40 | 1500.00 | 31.56 | 1.65 | 1.40 | 12.99 | 0.88 | 0.92 | 10.60 | 0.02 | 1685 | 859 | 839 | 0.0265 |
| 7.50 | 1400.00 | 34.36 | 1.65 | 1.50 | 13.33 | 0.88 | 0.92 | 3.51 | 0.02 | 1617 | 881 | 791 | 0.0239 |
| 5.50 | 1300.00 | 37.03 | 1.60 | 1.70 | 13.24 | 0.90 | 0.90 | 12.80 | 0.02 | 1439 | 902 | 695 | 0.0183 |
| 5.50 | 1300.00 | 37.03 | 1.60 | 1.70 | 13.24 | 0.88 | 0.92 | 12.80 | 0.02 | 1439 | 902 | 695 | 0.0183 |


### B) Best combined T45/EPR error in plausible set

| BPR | Wc_fan_design | OPR | PR_fan | PR_lpc | PR_hpc | eta_hpt | eta_lpt | T45_err_pct | EPR_err_pct | T4_K | T30_K | T50_K | FAR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6.50 | 1050.00 | 29.74 | 1.70 | 1.80 | 9.80 | 0.90 | 0.92 | 0.31 | 0.06 | 1522 | 843 | 762 | 0.0221 |
| 6.50 | 1050.00 | 32.72 | 1.70 | 1.80 | 10.78 | 0.88 | 0.92 | 0.31 | 0.44 | 1543 | 868 | 762 | 0.0221 |
| 6.50 | 1050.00 | 32.73 | 1.65 | 1.80 | 11.11 | 0.88 | 0.88 | 0.59 | 0.13 | 1543 | 868 | 779 | 0.0221 |
| 6.50 | 1050.00 | 35.69 | 1.70 | 1.80 | 11.76 | 0.90 | 0.90 | 0.31 | 0.61 | 1563 | 892 | 762 | 0.0221 |
| 6.50 | 1050.00 | 29.76 | 1.65 | 1.80 | 10.10 | 0.90 | 0.88 | 0.59 | 0.41 | 1522 | 844 | 779 | 0.0221 |
| 6.50 | 1050.00 | 29.74 | 1.70 | 1.70 | 10.38 | 0.90 | 0.92 | 0.85 | 0.13 | 1522 | 843 | 762 | 0.0221 |
| 6.50 | 1050.00 | 32.72 | 1.70 | 1.70 | 11.42 | 0.88 | 0.92 | 0.85 | 0.44 | 1543 | 868 | 762 | 0.0221 |
| 6.50 | 1050.00 | 32.73 | 1.65 | 1.70 | 11.76 | 0.88 | 0.88 | 1.12 | 0.03 | 1543 | 868 | 779 | 0.0221 |
| 6.50 | 1050.00 | 35.69 | 1.70 | 1.70 | 12.46 | 0.90 | 0.90 | 0.85 | 0.78 | 1563 | 891 | 762 | 0.0221 |
| 6.50 | 1050.00 | 29.76 | 1.65 | 1.70 | 10.70 | 0.90 | 0.88 | 1.12 | 0.64 | 1522 | 843 | 779 | 0.0221 |


### C) Best near C-MAPSS region (BPR ∈ [7.5, 9.0] AND OPR ∈ [30.0, 36.0])

| BPR | Wc_fan_design | OPR | PR_fan | PR_lpc | PR_hpc | eta_hpt | eta_lpt | T45_err_pct | EPR_err_pct | T4_K | T30_K | T50_K | FAR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7.50 | 1300.00 | 30.86 | 1.60 | 1.60 | 11.72 | 0.88 | 0.88 | 3.75 | 0.01 | 1593 | 853 | 814 | 0.0239 |
| 9.00 | 1200.00 | 30.35 | 1.70 | 1.70 | 10.38 | 0.92 | 0.92 | 17.45 | 0.01 | 1735 | 849 | 859 | 0.0281 |
| 7.50 | 1050.00 | 32.72 | 1.70 | 1.80 | 10.78 | 0.88 | 0.90 | 8.60 | 0.01 | 1645 | 868 | 822 | 0.0251 |
| 8.40 | 1500.00 | 31.56 | 1.65 | 1.40 | 12.99 | 0.88 | 0.92 | 10.60 | 0.02 | 1685 | 859 | 839 | 0.0265 |
| 7.50 | 1400.00 | 34.36 | 1.65 | 1.50 | 13.33 | 0.88 | 0.92 | 3.51 | 0.02 | 1617 | 881 | 791 | 0.0239 |
| 8.40 | 1500.00 | 31.56 | 1.65 | 1.50 | 12.12 | 0.88 | 0.92 | 11.22 | 0.03 | 1686 | 859 | 839 | 0.0265 |
| 9.00 | 1300.00 | 30.85 | 1.65 | 1.50 | 12.12 | 0.88 | 0.90 | 16.14 | 0.04 | 1738 | 852 | 879 | 0.0281 |
| 7.50 | 1200.00 | 30.37 | 1.65 | 1.40 | 12.99 | 0.88 | 0.92 | 2.72 | 0.04 | 1588 | 847 | 797 | 0.0239 |
| 8.40 | 1200.00 | 33.40 | 1.65 | 1.70 | 11.76 | 0.90 | 0.88 | 12.12 | 0.04 | 1698 | 874 | 849 | 0.0265 |
| 7.50 | 1400.00 | 34.36 | 1.65 | 1.60 | 12.50 | 0.88 | 0.92 | 4.10 | 0.04 | 1617 | 882 | 791 | 0.0239 |


### D) Top 20 by EPR error (with plausibility flag)

See `top20_candidates.csv` and `19_top20_candidates_table.png`. Sample rows:

| BPR | Wc_fan_design | OPR | PR_fan | PR_lpc | PR_hpc | T45_err_pct | EPR_err_pct | T4_K | T50_K | FAR |
|---|---|---|---|---|---|---|---|---|---|---|
| 7.50 | 1300.00 | 30.86 | 1.60 | 1.60 | 11.72 | 3.75 | 0.01 | 1593 | 814 | 0.0239 |
| 9.00 | 1200.00 | 30.35 | 1.70 | 1.70 | 10.38 | 17.45 | 0.01 | 1735 | 859 | 0.0281 |
| 7.50 | 1300.00 | 37.02 | 1.65 | 1.80 | 12.12 | 5.12 | 0.01 | 1635 | 794 | 0.0239 |
| 7.50 | 1050.00 | 32.72 | 1.70 | 1.80 | 10.78 | 8.60 | 0.01 | 1645 | 822 | 0.0251 |
| 8.40 | 1200.00 | 36.44 | 1.65 | 1.60 | 13.64 | 11.56 | 0.01 | 1717 | 849 | 0.0265 |
| 8.40 | 1400.00 | 37.48 | 1.65 | 1.60 | 13.64 | 11.74 | 0.02 | 1725 | 841 | 0.0265 |
| 8.40 | 1500.00 | 31.56 | 1.65 | 1.40 | 12.99 | 10.60 | 0.02 | 1685 | 839 | 0.0265 |
| 7.50 | 1400.00 | 34.36 | 1.65 | 1.50 | 13.33 | 3.51 | 0.02 | 1617 | 791 | 0.0239 |


## Recommendation

Top candidates are listed but NOT adopted. Robert reviews and chooses; YAML is updated only after explicit approval.

## CSV exports

- `all_candidates.csv` — full sweep with every metric
- `plausible_candidates.csv` — plausible subset only
- `top20_candidates.csv` — top 20 by EPR error

## Plot index

| # | Plot | File |
|---|---|---|
| 13 | BPR × Wc_fan heatmap, color = min EPR err, contour = T45 err | `13_heatmap_BPR_Wcfan_EPRerr.png` |
| 14 | BPR × Wc_fan heatmap, color = T50 at min-EPR cell | `14_heatmap_BPR_Wcfan_T50.png` |
| 15 | Pareto T45 err vs EPR err, color=BPR, size~Wc_fan, X=old baseline | `15_pareto_T45err_vs_EPRerr.png` |
| 16 | LP-spool decomposition (W_fan, W_lpc, W_lpt, PR_lpt, T50) vs BPR | `16_LP_spool_decomposition_vs_BPR.png` |
| 17 | Top-5 candidate station total-pressure profile | `17_station_pressure_top5.png` |
| 18 | Top-5 candidate station total-temperature profile | `18_station_temperature_top5.png` |
| 19 | Top-20 candidate table (image) | `19_top20_candidates_table.png` |

---

*Stop. No automatic parameter selection. Awaiting Robert review.*
