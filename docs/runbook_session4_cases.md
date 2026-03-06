# Runbook: Session 4 Case Experiments

Training and evaluation runbook for the three planned N-CMAPSS DS01-005 cases.

> **Dataset note**: The repo uses **N-CMAPSS DS01-005** (`data/NCMAPSS/N-CMAPSS_DS01-005.h5`).
> N-CMAPSS uses DS01–DS08 naming; this is distinct from the classic CMAPSS FD001–FD004 files.
> To run on a different dataset (e.g. DS04), set `data.hdf5_path` in the config accordingly.

---

## Case Matrix

| Case ID | Model          | Config                    | Script         | Lambda θ | Notes                         |
|---------|----------------|---------------------------|----------------|----------|-------------------------------|
| A       | `cnn`          | `configs/baseline.yaml`   | `train.py`     | 0        | Pure RUL baseline             |
| B       | `cyclelayer_v1`| `configs/cyclelayer.yaml` | `train.py`     | 1.0→0.1  | Multi-task; lambda schedule   |
| C       | `cyclelayer_v1`| `configs/cyclelayer.yaml` | `run_louo.py`  | 1.0→0.1  | Leave-One-Unit-Out CV (6 folds) |

---

## Prerequisites

```bash
pip install -e ".[dev]"   # once, from repo root

# Verify data file exists:
ls data/NCMAPSS/N-CMAPSS_DS01-005.h5
```

---

## Method 1 — `run_cases.py` (recommended, fully automated)

`scripts/run_cases.py` creates a timestamped run directory, writes reproducibility
metadata, calls train + evaluate, and saves all artifacts under `runs/`.

### Case A — CNN Baseline

```bash
py -3.11 scripts/run_cases.py \
    --config configs/baseline.yaml \
    --model cnn
```

**Expected run dir:** `runs/YYYYMMDD_HHMMSS_DS01-005_cnn_train/`

**Expected artifacts:**
```
runs/<run_dir>/
    run_meta.json          # git hash, command, timestamp, seed
    config_frozen.yaml     # config with output_dir patched
    config_original.yaml   # original config (unchanged)
    best.pt                # best checkpoint (by val loss)
    last.pt                # final checkpoint
    metrics.json           # RMSE, S-score, PH (stride_eval=1)
    predictions.csv        # unit_id, time_index, y_true_rul, y_pred_rul, abs_error
    tb/                    # TensorBoard event files
```

**Expected stdout (50 epochs, DS01):**
- Training: ~2–5 min on GPU, ~15–30 min on CPU
- Final logged: `RMSE: ~10–15  S-score: ~200–500  PH_median: ~N`

### Case B — CycleLayer v1 (multi-task)

```bash
py -3.11 scripts/run_cases.py \
    --config configs/cyclelayer.yaml \
    --model cyclelayer_v1
```

**Expected run dir:** `runs/YYYYMMDD_HHMMSS_DS01-005_cyclelayer_v1_train/`

**Lambda schedule:** `lambda_theta` starts at 1.0, drops to 0.1 after epoch 10 (step schedule).
TensorBoard scalar `Lambda/theta` tracks the schedule. Loss breakdown `[tr_rul=x  tr_theta=x]`
is printed each epoch.

### Case C — LOUO Cross-Validation (cyclelayer_v1)

```bash
py -3.11 scripts/run_cases.py \
    --config configs/cyclelayer.yaml \
    --model cyclelayer_v1 \
    --louo
```

**Expected run dir:** `runs/YYYYMMDD_HHMMSS_DS01-005_cyclelayer_v1_louo/`

**Expected artifacts:**
```
runs/<run_dir>/
    run_meta.json
    config_frozen.yaml
    louo_cyclelayer_v1/
        fold_1/
            splits/{train,val,test}_units.txt   # per-fold split files
            best.pt
            theta_scaler.npz                    # scaler fit on fold train units only
            tb/
        fold_2/ ... fold_6/
        results.json                            # aggregate RMSE/S-score/PH over 6 folds
```

**Specific folds only** (faster for smoke-testing):
```bash
py -3.11 scripts/run_cases.py \
    --config configs/cyclelayer.yaml \
    --model cyclelayer_v1 \
    --louo --units 1 3
```

---

## Method 2 — Manual commands (reproducible without `run_cases.py`)

### Case A

```bash
py -3.11 scripts/train.py    --config configs/baseline.yaml --model cnn
py -3.11 scripts/evaluate.py --config configs/baseline.yaml --split dev
```

Outputs: `runs/baseline/best.pt`, `runs/baseline/metrics.json`, `runs/baseline/predictions.csv`

### Case B

```bash
py -3.11 scripts/train.py    --config configs/cyclelayer.yaml --model cyclelayer_v1
py -3.11 scripts/evaluate.py --config configs/cyclelayer.yaml --split dev
```

Outputs: `runs/cyclelayer_v1/best.pt`, `runs/cyclelayer_v1/metrics.json`, `runs/cyclelayer_v1/predictions.csv`

### Case C

```bash
py -3.11 scripts/run_louo.py --config configs/cyclelayer.yaml --model cyclelayer_v1
```

Outputs: `runs/louo_cyclelayer_v1/results.json` (aggregate), per-fold artifacts inside.

---

## Build Failure-Case Library

After evaluation, build the case library from `predictions.csv`:

```bash
# Case A baseline:
py -3.11 scripts/build_case_library.py \
    --predictions runs/<run_dir>/predictions.csv \
    --out_dir results/ds01/case_library_baseline \
    --n 20

# Case B cyclelayer_v1:
py -3.11 scripts/build_case_library.py \
    --predictions runs/<run_dir>/predictions.csv \
    --out_dir results/ds01/case_library_cyclelayer \
    --n 20

# Case C LOUO (combine all fold predictions):
py -3.11 scripts/build_case_library.py \
    --predictions "runs/<run_dir>/louo_cyclelayer_v1/fold_*/predictions.csv" \
    --out_dir results/ds01/case_library_louo \
    --n 20
```

> **Note**: LOUO fold predictions.csv files are not produced automatically by `run_louo.py`.
> Run `evaluate.py` per-fold manually if needed, or add a post-processing step.

**Output manifest layout:**
```
results/ds01/case_library_<case>/
    unit_manifest.csv       # per-unit EOL error, rank_tag (worst/mid/best)
    worst_windows.csv       # top-N windows with largest abs error
    best_windows.csv        # top-N windows with smallest abs error
    mid_windows.csv         # N windows closest to median abs error
    summary.json            # aggregate stats (RMSE, MAE, median AE, n_units)
```

**Ranking criterion:**
- *Unit-level*: absolute RUL error at the **End-Of-Life window** (window with minimum `y_true_rul` per unit).
- *Window-level*: `abs_error = |y_pred_rul - y_true_rul|` globally across all windows.

---

## Smoke Test (quick end-to-end check, no HDF5 needed for unit tests)

### Guard tests (no data file required)
```bash
py -3.11 -m pytest tests/test_pipeline_guards.py -v
```

### Pipeline smoke test (requires HDF5)
Uses `test_run.yaml` (3 epochs, stride_train=100):
```bash
py -3.11 scripts/train.py --config configs/test_run.yaml --model cnn
py -3.11 scripts/evaluate.py --config configs/test_run.yaml --split dev
```

---

## Theta Scaling — Leakage-Free Verification

The `theta_scaler.npz` saved alongside each checkpoint encodes the scaler's
`mean` and `std` fitted **only on train-unit rows**.

To verify manually:
```python
import numpy as np
scaler = np.load("runs/<run_dir>/theta_scaler.npz")
print("Scaler mean:", scaler["mean"])   # should reflect train units only
print("Scaler std :", scaler["std"])
```

Unit tests in `tests/test_pipeline_guards.py` assert this property on synthetic data:
- `test_theta_scaler_only_uses_train_rows`
- `test_theta_scaler_fit_on_multiple_train_units`
- `test_theta_scaler_raises_on_empty_train_mask`

---

## TensorBoard

```bash
tensorboard --logdir runs/
```

Key scalars per run:
| Scalar                      | Description                          |
|-----------------------------|--------------------------------------|
| `Loss/train_epoch`          | Training loss (total)                |
| `Loss/val_epoch`            | Validation loss (total)              |
| `Loss/train_rul_epoch`      | RUL component of training loss       |
| `Loss/train_theta_epoch`    | Theta component of training loss     |
| `Loss/val_rul_epoch`        | RUL component of validation loss     |
| `Loss/val_theta_epoch`      | Theta component of validation loss   |
| `Lambda/theta`              | Scheduled lambda_theta value         |
| `LR`                        | Learning rate (cosine decay)         |

---

## Expected Metric Ranges (DS01-005, 50 epochs)

| Case | Model          | RMSE (approx) | S-score | PH_median |
|------|----------------|---------------|---------|-----------|
| A    | cnn            | 8–15          | 200–600 | 20–40     |
| B    | cyclelayer_v1  | 7–14          | 150–500 | 25–45     |
| C    | cyclelayer_v1 (LOUO) | 10–20  | 300–800 | 15–35     |

*Ranges are indicative; actual values depend on random seed and hardware.*

---

## Artifact Reference

| File                             | Produced by       | Used by                     |
|----------------------------------|-------------------|-----------------------------|
| `best.pt`                        | `train.py`        | `evaluate.py`, case library |
| `theta_scaler.npz`               | `train.py`        | verification, reproduce     |
| `metrics.json`                   | `evaluate.py`     | comparison table            |
| `predictions.csv`                | `evaluate.py`     | `build_case_library.py`     |
| `splits/{train,val,test}.txt`    | `train.py` / LOUO | reproduce, anti-leakage     |
| `run_meta.json`                  | `run_cases.py`    | audit trail                 |
| `config_frozen.yaml`             | `run_cases.py`    | exact reproduce             |
| `results/ds01/case_library/`     | `build_case_library.py` | analysis, paper     |
