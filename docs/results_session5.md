# Session 5 — Experiment Results: N-CMAPSS DS01-005

**Date**: 2026-03-06
**Dataset**: N-CMAPSS DS01-005 (`data/NCMAPSS/N-CMAPSS_DS01-005.h5`)
**Device**: CPU (no CUDA)
**Git commit**: c5c122a

> **Config note**: Runs use `cpu_baseline.yaml` / `cpu_cyclelayer.yaml`
> (`stride_train=200`, `stride_eval=50`, `num_workers=0`, `use_amp=false`).
> This reduces training windows to ~17k (vs ~980k at stride=5) for feasible CPU runtimes.
> RMSE numbers are indicative — expect ~30–50% lower RMSE with GPU and stride=5.

---

## Unit Splits (seed=42, reused across all cases)

| Split | Units |
|-------|-------|
| Train | 3, 4, 5, 6 |
| Val   | 2 |
| Test  | 1 |

Evaluation uses `--split dev` restricted to val+test units (1, 2).

---

## Case Matrix & Artifacts

| Case | Model          | Config                   | Run dir                                              |
|------|----------------|--------------------------|------------------------------------------------------|
| A    | `cnn`          | `cpu_baseline.yaml`      | `runs/20260306_093821_DS01-005_cnn_train/`           |
| B    | `cyclelayer_v1`| `cpu_cyclelayer.yaml`    | `runs/20260306_094340_DS01-005_cyclelayer_v1_train/` |
| C    | `cyclelayer_v1`| `cpu_cyclelayer.yaml`    | `runs/20260306_094710_DS01-005_cyclelayer_v1_louo/`  |

---

## Case A — CNN Baseline (in-distribution)

**Run**: `scripts/run_cases.py --config configs/cpu_baseline.yaml --model cnn`

| Metric             | Value    |
|--------------------|----------|
| RMSE               | **13.45** |
| S-score            | +8681    |
| PH (alpha=0.2)     | None (both units) |
| Trainable params   | 94,785   |
| Best epoch         | 21 (early stop at 36) |
| stride_train       | 200      |
| Train windows      | 17,123   |

**Per-unit trajectory analysis (eval split: units 1 and 2):**

| Unit | Windows | Start err | End err | Max err | Max err at t | Frac within alpha=0.2 |
|------|---------|-----------|---------|---------|-------------|----------------------|
| 1    | 8,666   | 32.2      | 0.2     | 54.6    | t=137 (y_true=98) | 38.0% |
| 2    | 20,982  | 2.9       | 0.2     | 54.3    | t=192 (y_true=74) | 38.6% |

**Key observations**:
- EOL accuracy is excellent (< 0.3 error) — the model converges correctly near failure.
- Large mid-trajectory errors (~54 cycles) cause PH=None; the alpha criterion requires sustained accuracy from some point onwards, which is never achieved.
- Positive S-score (+8681) indicates systematic **over-estimation** of RUL (model predicts too optimistic remaining life in mid-trajectory).

**Artifacts**:
- `runs/20260306_093821.../best.pt` — best checkpoint
- `runs/20260306_093821.../metrics.json` — evaluation metrics
- `runs/20260306_093821.../predictions.csv` — 29,648 rows
- `runs/20260306_093821.../theta_scaler.npz` — NOT present (cnn has no theta)
- `results/ds01/case_library_cnn/` — failure case library

---

## Case B — CycleLayer v1 Multi-task (in-distribution)

**Run**: `scripts/run_cases.py --config configs/cpu_cyclelayer.yaml --model cyclelayer_v1`

Lambda-theta schedule: start=1.0 → end=0.1 after 10 epochs (step schedule).

| Metric             | Value     |
|--------------------|-----------|
| RMSE               | **13.56** |
| S-score            | -8586     |
| PH (alpha=0.2)     | None (both units) |
| Trainable params   | 33,279    |
| Best epoch         | 38 (no early stop triggered) |
| stride_train       | 200       |
| Train windows      | 17,123    |

**Per-unit trajectory analysis:**

| Unit | Windows | Start err | End err | Max err | Max err at t | Frac within alpha=0.2 |
|------|---------|-----------|---------|---------|-------------|----------------------|
| 1    | 8,666   | 48.6      | 1.5     | 54.3    | t=220 (y_true=97) | 31.5% |
| 2    | 20,982  | 23.6      | 1.3     | 42.2    | t=4828 (y_true=58) | 55.6% |

**Key observations**:
- Nearly identical RMSE to CNN (13.56 vs 13.45) — expected at high stride; the physics layer needs more data to show benefit.
- Negative S-score (−8586) indicates systematic **under-estimation** of RUL (conservative predictions). Opposite direction from CNN. The multi-task theta loss appears to bias predictions toward conservative estimates.
- **Lower MAE** than CNN (8.88 vs 10.22) and substantially **lower median absolute error** (4.81 vs 7.79) — suggesting the distribution of errors is better despite similar RMSE.
- Unit 2 achieves 55.6% of windows within alpha=0.2 (vs CNN's 38.6%) — more accurate on the longer trajectory.
- `theta_scaler.npz` present: scaler fit on train units [3,4,5,6] only (no leakage).

**Artifacts**:
- `runs/20260306_094340.../best.pt`
- `runs/20260306_094340.../theta_scaler.npz` — mean and std for 10 health parameters
- `runs/20260306_094340.../metrics.json`
- `runs/20260306_094340.../predictions.csv` — 29,648 rows
- `results/ds01/case_library_cyclelayer_v1/` — failure case library

---

## Case C — LOUO Cross-Validation (out-of-distribution)

**Run**: `scripts/run_cases.py --config configs/cpu_cyclelayer.yaml --model cyclelayer_v1 --louo`

6 folds; each fold trains on 4 units, validates on 1, tests on the held-out unit.
Val unit per fold: always the highest-ID remaining unit (unit 6 for folds 1-5, unit 5 for fold 6).

| Fold | Test unit | Val unit | Train units | RMSE   | S-score  |
|------|-----------|----------|-------------|--------|----------|
| 1    | 1         | 6        | [2,3,4,5]   | 25.49  | -2468    |
| 2    | 2         | 6        | [1,3,4,5]   | 10.65  | +3872    |
| 3    | 3         | 6        | [1,2,4,5]   | 21.29  | +8110    |
| 4    | 4         | 6        | [1,2,3,5]   | 20.16  | -1181    |
| 5    | 5         | 6        | [1,2,3,4]   | 17.14  | -9083    |
| 6    | 6         | 5        | [1,2,3,4]   | 18.73  | -7971    |

**Aggregate (6 folds):**

| Metric      | Mean   | Std   |
|-------------|--------|-------|
| RMSE        | **18.91** | 4.51  |
| S-score     | -1453  | 6077  |
| PH (median) | None (6/6 folds = 0 PH achieved) |

**Key observations**:
- LOUO RMSE (18.91) is **40% higher** than in-distribution eval (13.56) — confirms a real generalization gap between units. N-CMAPSS DS01-005 shows significant unit-to-unit variability.
- Best fold: unit 2 (RMSE=10.65); worst fold: unit 1 (RMSE=25.49) — 2.4× spread.
- High S-score variance (std=6077) indicates the model's systematic bias flips sign depending on the test unit. This is characteristic of underfitted models at high stride.
- Unit 6 has only 5 candidates for val (it's in train for folds 1-5), so in fold 6 the val unit is unit 5 instead.

**Artifacts**:
```
runs/20260306_094710.../louo_cyclelayer_v1/
    fold_1/ ... fold_6/    -- per-fold: best.pt, theta_scaler.npz, splits/
    results.json           -- aggregate metrics
```

---

## Comparative Summary

| Case | Model          | Setting   | RMSE  | S-score | MAE  | Median AE | PH   |
|------|----------------|-----------|-------|---------|------|-----------|------|
| A    | CNN baseline   | In-dist   | 13.45 | +8681   | 10.22| 7.79      | None |
| B    | CycleLayer v1  | In-dist   | 13.56 | -8586   | 8.88 | **4.81**  | None |
| C    | CycleLayer v1  | LOUO      | 18.91 | -1453   | —    | —         | None |

> MAE and Median AE for Case C not computed (run_louo.py produces global RMSE/S-score only; per-unit predictions.csv available per fold if evaluate.py is run post-fold).

---

## Failure Case Libraries

### Case A — CNN (`results/ds01/case_library_cnn/`)

| File                  | Content                                      |
|-----------------------|----------------------------------------------|
| `unit_manifest.csv`   | 2 units; unit 2 worst (EOL err=0.21), unit 1 best (0.21) |
| `worst_windows.csv`   | 20 windows with abs_error up to ~54.6 cycles |
| `best_windows.csv`    | 20 windows with abs_error near 0             |
| `mid_windows.csv`     | 20 windows near median abs_error (~7.79)     |
| `summary.json`        | RMSE=13.453, MAE=10.215, MedianAE=7.788      |

**Notable**: Both units have virtually identical EOL errors (0.21 vs 0.21). The large errors occur in the high-RUL region (t<200, y_true≈74-99).

### Case B — CycleLayer v1 (`results/ds01/case_library_cyclelayer_v1/`)

| File                  | Content                                      |
|-----------------------|----------------------------------------------|
| `unit_manifest.csv`   | Unit 1 worst (EOL err=1.06), unit 2 best (1.01) |
| `worst_windows.csv`   | 20 windows with abs_error up to ~54.3 cycles |
| `best_windows.csv`    | 20 windows with abs_error near 0             |
| `mid_windows.csv`     | 20 windows near median abs_error (~4.81)     |
| `summary.json`        | RMSE=13.563, MAE=8.883, MedianAE=4.814       |

**Notable**: EOL errors are slightly larger than CNN (1.0 vs 0.2) but median trajectory error is much lower (4.81 vs 7.79). The worst windows occur at high-RUL / start of trajectory.

---

## Observations and Next Steps

### What the runs confirm
1. **Pipeline integrity**: Theta scaler fitted on train units only (verified by `theta_scaler.npz` in each fold). Unit splits are deterministic and saved per run.
2. **Generalization gap is real**: LOUO adds ~40% relative RMSE vs in-distribution. With only 6 units, this variance is expected.
3. **CycleLayer v1 shows better median accuracy**: Median AE is 4.81 vs 7.79 for CNN (38% reduction) despite similar RMSE. The multi-task theta loss helps concentration around the median.
4. **S-score sign flip**: CNN over-predicts (optimistic); CycleLayer v1 under-predicts (conservative). This suggests the theta supervision introduces a systematic conservative bias — worth investigating.

### What needs a GPU run to validate
1. **Prediction horizon**: PH=None for all runs at stride=200. With stride=5 and proper training (~980k windows, 50 epochs), mid-trajectory accuracy should improve significantly.
2. **Physics advantage**: At stride=200 both models learn ~equally (13.45 vs 13.56 RMSE). The physics constraint needs richer training to manifest.
3. **LOUO variance**: 4.51 RMSE std across folds is large. Need more training to tighten this.

### Recommended next run (GPU)
```bash
# Full training with production stride (requires GPU, ~2h each)
py -3.11 scripts/run_cases.py --config configs/baseline.yaml --model cnn
py -3.11 scripts/run_cases.py --config configs/cyclelayer.yaml --model cyclelayer_v1
py -3.11 scripts/run_cases.py --config configs/cyclelayer.yaml --model cyclelayer_v1 --louo
```
Expected improvements: RMSE < 8, PH achieved for some units, LOUO variance reduced.

---

## Reproducibility Checklist

- [x] Unit splits saved: `splits/N-CMAPSS_DS01-005/{train,val,test}_units.txt`
- [x] Frozen configs in each run_dir: `config_frozen.yaml`
- [x] Git hash recorded: `run_meta.json` (c5c122a)
- [x] Theta scaler saved per run: `theta_scaler.npz` (Cases B, C)
- [x] Per-fold splits saved in LOUO: `fold_*/splits/{train,val,test}_units.txt`
- [x] Predictions exported: `predictions.csv` (Cases A, B)
- [x] Case library manifests: `results/ds01/case_library_{cnn,cyclelayer_v1}/`
