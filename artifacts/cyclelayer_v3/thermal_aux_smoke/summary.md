# V3.1b thermal-aux smoke training summary

*Profile:* `thermal_regularizer_v3_1b_not_epr_validated`
*Profile scope:* see ADR-0012 (V3.1b thermal auxiliary, NOT EPR-validated).

## Run

* config: `configs\cyclelayer_v3_thermal_aux.yaml`
* device: `cpu`
* epochs: 1
* batch size: 64
* train windows: 1,500
* val   windows: 400
* total wall time: 1.1 s

## Final epoch

| | train | val |
|---|---|---|
| total loss | 680.6418 | 631.3753 |
| L_rul     | 678.3719 | 629.1264 |
| L_temp    | 22.4295 | 22.0789 |
| L_aux     | 1.3504 | 2.0554 |
| L_healthy | 0.0001 | 0.0000 |
| L_smooth  | 0.0000 | 0.0000 |
| RUL RMSE  | 25.036 | 24.027 |
| T24 MAE [K] | 68.40 | 68.23 |
| T30 MAE [K] | 96.94 | 95.56 |
| T50 MAE [K] | 69.13 | 67.45 |
| θ mean    | 0.9900 | 0.9900 |
| θ std     | 0.0000 | 0.0000 |
| θ frac@lo (0.85) | 0.000 | 0.000 |
| θ frac@hi (1.00) | 0.000 | 0.000 |
| EPR mean (DIAG, not in loss) | 0.900 | 0.911 |

## Hard constraints honored

* No EPR / pressure in loss (asserted by `CycleLayerV3Loss`).
* No supervised L_θ on θ_phys.
* Train/val units split (no random row split).
* Test units `[11, 14, 15]` NOT used (later evaluation only).

## Artifacts

* `best.pt`           — model state at best val loss
* `last.pt`           — final model state
* `sensor_scaler.npz` — per-channel mean/std for X_s (14 cols)
* `ops_scaler.npz`    — per-channel mean/std for W   (4 cols)
* `sigma_train.json`  — σ_T24/T30/T50 (K) and σ_lpt_flow used in L_temp / L_aux
* `train_log.csv`     — per-epoch metrics

## Next step (manual)

```bash
python scripts/evaluate_cyclelayer_v3_theta_diagnostics.py \
    --checkpoint artifacts\cyclelayer_v3\thermal_aux_smoke\best.pt \
    --config     configs\cyclelayer_v3_thermal_aux.yaml
```
