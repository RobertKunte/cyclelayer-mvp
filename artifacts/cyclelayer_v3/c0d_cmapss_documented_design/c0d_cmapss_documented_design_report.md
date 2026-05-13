# C0d Documented C-MAPSS reference-engine initialization check — FC02

*Read-only diagnostic. No YAML written. No DS02 access. No optimizer.*
*FC02 is an external sanity check, not a fitting target.*

## 1. Source verification

Local-only search of `data/`, `docs/`, and repository PDFs for the documented
C-MAPSS reference-engine values:

| Reference value | Locally verified? | Source |
|---|---|---|
| 90,000 lb thrust class | **VERIFIED** | Saxena 2008, [data/CMAPSS/Damage Propagation Modeling.pdf](data/CMAPSS/Damage Propagation Modeling.pdf), p. 2 |
| Operating envelope (alt 0–40 K ft, M 0–0.9, T_SL −60..103 °F) | **VERIFIED** | Saxena 2008, p. 2 |
| BPR ≈ 8.4 | NOT FOUND locally | Cited via Frederick TM2007-215026 ref [11] in Saxena 2008 |
| OPR ≈ 36.0 | NOT FOUND locally | Same |
| Nf design speed ≈ 2450.0 rpm | NOT FOUND locally | Same |
| Nc design speed ≈ 9300.0 rpm | NOT FOUND locally | Same |
| Wc_fan_design ≈ 1658.0 kg/s | NOT FOUND locally | Same |
| Component PR / eta table (Fan / LPC / HPC / HPT / LPT) | NOT FOUND locally | Same |

The numerical values are USER-PROVIDED documented reference values and are
treated as such throughout this report. **Source verification pending acquisition
of Frederick et al., NASA/ARL TM2007-215026.** No internet browsing performed.

## 2. Frozen documented values used in C0d

| Quantity | Value |
|---|---|
| BPR | 8.4 |
| OPR (engine-level) | 36.0 |
| Nf design (rpm) | 2450.0 |
| Nc design (rpm) | 9300.0 |
| Wc_fan_design (kg/s) | 1658.0 |
| PR_fan / eta_fan | 1.784 / 0.8969 |
| PR_lpc / eta_lpc | 1.1035 / 0.9148 |
| PR_hpc / eta_hpc | 21.817 / 0.8615 |
| PR_hpt / eta_hpt | 4.239 / 0.9202 |
| PR_lpt / eta_lpt | 5.858 / 0.93 |
| Implied OPR from PR table | **42.95** (≠ documented OPR 36.0) |

NOTE: V3.1b uses Nf and Nc as INPUTS at FC02; PR_hpt and PR_lpt are **closure-determined**
from the shaft balances, not free parameters. The documented PR_hpt / PR_lpt are
listed for reference only.

## 3. Candidates

**A** — engine-level OPR=36 respected; PR_hpc derived as `36/(PR_fan × PR_lpc) = 18.287`
**B** — documented PR table verbatim; PR_hpc=21.817; implied OPR ≈ 42.95
**C** — conservative hybrid; PR_fan=1.70, PR_lpc=1.20, PR_hpc=`36/(1.70×1.20)=17.647`; Wc_fan_design = 1500 kg/s
**D** — old baseline (BPR=5.5, Wc_fan=900, OPR=38.4) — FOR COMPARISON ONLY, NOT FOR ADOPTION

## 4. Per-candidate FC02 results


### A — documented OPR=36 (PR_hpc derived)

| metric | value |
|---|---|
| BPR | 8.4 |
| Wc_fan_design [kg/s] | 1658 |
| PR_fan | 1.784 |
| PR_lpc | 1.103 |
| PR_hpc | 18.287 |
| OPR implied (PR product) | 36.00 |
| OPR predicted (P30/P2) | 35.55 |
| eta_fan | 0.8969 |
| eta_lpc | 0.9148 |
| eta_hpc | 0.8615 |
| eta_hpt | 0.9202 |
| eta_lpt | 0.9300 |
| EPR pred | 0.2648 |
| EPR rel err [%] | 79.00 |
| T45 [°R] | 1730.6 |
| T45 rel err vs T48_ref [%] | 16.92 |
| T4 [K] | 1425 |
| T30 [K] | 900 |
| T50 [K] | 482 |
| FAR | 0.0179 |
| m_in [kg/s] | 1683.8 |
| m_core [kg/s] | 179.1 |
| PR_hpt (closure) | 5.806 |
| PR_lpt (closure) | 22.200 |
| Wc_fan requested [kg/s] | 1658.0 |
| Wc_fan pre-clamp [kg/s] | 1622.2 |
| Wc_fan actual m_in [kg/s] | 1683.8 |
| Wc_min [kg/s] | 200 |
| Wc_max [kg/s] | 2500 |
| Wc clamp active | ✗ |
| PR clamp active | ✗ |
| FAR plausible | ✓ |
| T4 plausible | ✓ |
| T30 plausible | ✓ |
| T50 plausible | ✗ |
| T45 err < 25 % | ✓ |
| T45 err < 15 % | ✗ |
| EPR err < 25 % | ✗ |
| eta_hpt ≥ 0.88 | ✓ |
| eta_lpt ≥ 0.88 | ✓ |

### B — documented PR table (PR_hpc=21.817, implied OPR=42.95)

| metric | value |
|---|---|
| BPR | 8.4 |
| Wc_fan_design [kg/s] | 1658 |
| PR_fan | 1.784 |
| PR_lpc | 1.103 |
| PR_hpc | 21.817 |
| OPR implied (PR product) | 42.95 |
| OPR predicted (P30/P2) | 42.41 |
| eta_fan | 0.8969 |
| eta_lpc | 0.9148 |
| eta_hpc | 0.8615 |
| eta_hpt | 0.9202 |
| eta_lpt | 0.9300 |
| EPR pred | 0.2761 |
| EPR rel err [%] | 78.11 |
| T45 [°R] | 1730.6 |
| T45 rel err vs T48_ref [%] | 16.92 |
| T4 [K] | 1468 |
| T30 [K] | 950 |
| T50 [K] | 482 |
| FAR | 0.0179 |
| m_in [kg/s] | 1683.8 |
| m_core [kg/s] | 179.1 |
| PR_hpt (closure) | 6.643 |
| PR_lpt (closure) | 22.200 |
| Wc_fan requested [kg/s] | 1658.0 |
| Wc_fan pre-clamp [kg/s] | 1622.2 |
| Wc_fan actual m_in [kg/s] | 1683.8 |
| Wc_min [kg/s] | 200 |
| Wc_max [kg/s] | 2500 |
| Wc clamp active | ✗ |
| PR clamp active | ✗ |
| FAR plausible | ✓ |
| T4 plausible | ✓ |
| T30 plausible | ✓ |
| T50 plausible | ✗ |
| T45 err < 25 % | ✓ |
| T45 err < 15 % | ✗ |
| EPR err < 25 % | ✗ |
| eta_hpt ≥ 0.88 | ✓ |
| eta_lpt ≥ 0.88 | ✓ |

### C — conservative hybrid (PR_fan=1.70, PR_lpc=1.20, OPR=36)

| metric | value |
|---|---|
| BPR | 8.4 |
| Wc_fan_design [kg/s] | 1500 |
| PR_fan | 1.700 |
| PR_lpc | 1.200 |
| PR_hpc | 17.647 |
| OPR implied (PR product) | 36.00 |
| OPR predicted (P30/P2) | 35.57 |
| eta_fan | 0.8969 |
| eta_lpc | 0.9148 |
| eta_hpc | 0.8615 |
| eta_hpt | 0.9202 |
| eta_lpt | 0.9300 |
| EPR pred | 0.5369 |
| EPR rel err [%] | 57.42 |
| T45 [°R] | 1857.2 |
| T45 rel err vs T48_ref [%] | 10.84 |
| T4 [K] | 1492 |
| T30 [K] | 901 |
| T50 [K] | 587 |
| FAR | 0.0198 |
| m_in [kg/s] | 1523.3 |
| m_core [kg/s] | 162.1 |
| PR_hpt (closure) | 5.177 |
| PR_lpt (closure) | 12.288 |
| Wc_fan requested [kg/s] | 1500.0 |
| Wc_fan pre-clamp [kg/s] | 1467.6 |
| Wc_fan actual m_in [kg/s] | 1523.3 |
| Wc_min [kg/s] | 200 |
| Wc_max [kg/s] | 2500 |
| Wc clamp active | ✗ |
| PR clamp active | ✗ |
| FAR plausible | ✓ |
| T4 plausible | ✓ |
| T30 plausible | ✓ |
| T50 plausible | ✗ |
| T45 err < 25 % | ✓ |
| T45 err < 15 % | ✓ |
| EPR err < 25 % | ✗ |
| eta_hpt ≥ 0.88 | ✓ |
| eta_lpt ≥ 0.88 | ✓ |

### D — OLD baseline (BPR=5.5, Wc=900, OPR=38.4) — NOT FOR ADOPTION

| metric | value |
|---|---|
| BPR | 5.5 |
| Wc_fan_design [kg/s] | 900 |
| PR_fan | 1.600 |
| PR_lpc | 2.000 |
| PR_hpc | 12.000 |
| OPR implied (PR product) | 38.40 |
| OPR predicted (P30/P2) | 37.99 |
| eta_fan | 0.9200 |
| eta_lpc | 0.9000 |
| eta_hpc | 0.8800 |
| eta_hpt | 0.9000 |
| eta_lpt | 0.9200 |
| EPR pred | 2.2296 |
| EPR rel err [%] | 76.81 |
| T45 [°R] | 2132.1 |
| T45 rel err vs T48_ref [%] | 2.36 |
| T4 [K] | 1604 |
| T30 [K] | 909 |
| T50 [K] | 863 |
| FAR | 0.0229 |
| m_in [kg/s] | 914.0 |
| m_core [kg/s] | 140.6 |
| PR_hpt (closure) | 3.990 |
| PR_lpt (closure) | 4.100 |
| Wc_fan requested [kg/s] | 900.0 |
| Wc_fan pre-clamp [kg/s] | 880.6 |
| Wc_fan actual m_in [kg/s] | 914.0 |
| Wc_min [kg/s] | 100 |
| Wc_max [kg/s] | 1100 |
| Wc clamp active | ✗ |
| PR clamp active | ✗ |
| FAR plausible | ✓ |
| T4 plausible | ✓ |
| T30 plausible | ✓ |
| T50 plausible | ✓ |
| T45 err < 25 % | ✓ |
| T45 err < 15 % | ✓ |
| EPR err < 25 % | ✗ |
| eta_hpt ≥ 0.88 | ✓ |
| eta_lpt ≥ 0.88 | ✓ |


## 5. Q&A

### Q: Does BPR=8.4 improve EPR vs the old baseline (BPR=5.5)?

Old baseline (D, BPR=5.5, Wc=900): EPR err = **76.8 %**.  BPR=8.4 documented (A, OPR=36): EPR err = **79.0 %**.  NO — moving to BPR=8.4 alone does not reduce the EPR mismatch under the documented OPR.

### Q: Does Wc_fan_design ≈ 1658 kg/s produce plausible mass flow / FAR / T4 / T45 / T50?

At Wc_fan_design = 1658.0 kg/s (Candidates A and B), m_in at FC02 = 1683.8 kg/s, m_core = 179.1 kg/s, FAR = 0.0179 (plausible), T4 = 1425 K (plausible), T45 err = 16.9 %, T50 = 482 K (OUT OF RANGE).

### Q: Is the documented component-PR table consistent with engine-level OPR ≈ 36?

Documented PR table → implied OPR = `PR_fan × PR_lpc × PR_hpc` = 1.784 × 1.1035 × 21.817 = **42.95**.  This does NOT match the documented engine-level OPR ≈ 36.0.  Likely station / design-condition mismatch in the original Frederick TM2007-215026 table (e.g. component design point at a different OP, or a published delta vs reference). Resolve the mismatch via source verification before adopting the component table verbatim.

### Q: Which candidate is physically most defensible for C1 testing?

**No documented candidate (A / B / C) is fully plausible at FC02.** Failure flags per candidate:

- **A — documented OPR=36 (PR_hpc derived)**: failed = T50_plausible, EPR_acceptable
- **B — documented PR table (PR_hpc=21.817, implied OPR=42.95)**: failed = T50_plausible, EPR_acceptable
- **C — conservative hybrid (PR_fan=1.70, PR_lpc=1.20, OPR=36)**: failed = T50_plausible, EPR_acceptable

Given that the documented values cannot satisfy FC02 plausibility within V3.1b's explicit-closure architecture, **EPR should be demoted from a hard pressure gate to a diagnostic until V4 introduces a flow-matching solver.** T45 anchor remains a hard gate. C1 across 13 User Guide FCs should report EPR error per FC as a diagnostic, not an acceptance criterion. No YAML change is recommended at this stage.

## 6. Plot index

| # | Plot | File |
|---|---|---|
| 20 | FC02 station temperature, all candidates                | `20_FC02_station_temperatures.png` |
| 21 | FC02 station P/P2 (log), all candidates                  | `21_FC02_station_pressure_ratios_log.png` |
| 22 | FC02 EPR decomposition waterfall, per candidate          | `22_FC02_EPR_waterfall.png` |
| 23 | FC02 spool work balance (HPC vs HPT, Fan+LPC vs LPT)     | `23_FC02_work_balance.png` |
| 24 | FC02 metric comparison bars                              | `24_FC02_metric_comparison_bars.png` |
| 25 | FC02 candidate metrics table                             | `25_FC02_metrics_table.png` |

CSV: `candidates_metrics.csv` (full per-candidate metrics).

---

*Stop. No YAML written. No DS02. No automatic adoption. Awaiting Robert review.*
