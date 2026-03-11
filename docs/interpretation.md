# CycleLayer MVP — Metrics Interpretation Guide

## 1. Why training becomes unstable

### Symptom: best checkpoint at epoch 5, val loss explodes afterward

This is the most common failure mode with the CycleLayerNetV1 + CompositeLoss
at dense training (stride_train=5, ~200k windows/epoch).

**Root cause chain:**
1. With lambda_theta=1.0 at epoch 1, the theta Huber loss dominates early.
   L_theta is O(1) while L_rul is O(10-100) during random initialization.
2. Large combined gradients → gradient norm spikes → weights jump to a bad
   region → val loss explodes.
3. Best checkpoint is saved at epoch 3-5 before the explosion.

**Fixes applied (all configurable in YAML):**

| Fix | Config key | Recommended value |
|-----|-----------|-------------------|
| Gradient clipping | `training.grad_clip_norm` | 1.0 (default) |
| Delayed lambda warmup | `training.lambda_theta_schedule` | `"delayed"` |
| Longer warmup | `training.lambda_theta_warmup_epochs` | 20 for stride=5 |
| Lower LR | `training.lr` | 3e-4 for stride=5 (not 1e-3) |

**The "delayed" schedule** is the most important fix for dense training:
- Epochs 1..warmup: lambda_theta=0 (pure RUL training, no theta supervision)
- Epochs warmup+1..2*warmup: linearly ramps 0 → lambda_theta_end
- After 2*warmup: constant at lambda_theta_end

This lets the RUL head and encoder converge to a reasonable initialisation
before theta supervision is introduced, preventing the spike.

---

## 2. Reading the epoch log

Example log line:
```
12:34:56  INFO     Epoch  12/50  train=0.4231  val=0.5614  lr=8.43e-04  gnorm=0.847  lam_th=0.050  (23.1s)  [tr_rul=0.421  tr_theta=0.002  val_rul=0.558  val_theta=0.003]
```

| Field | Meaning |
|-------|---------|
| `train=` | Mean total loss across all training batches |
| `val=` | Mean total loss on validation set |
| `lr=` | Current learning rate (after cosine decay step) |
| `gnorm=` | Mean gradient L2 norm per batch **before** clipping |
| `lam_th=` | Current lambda_theta from the schedule |
| `tr_rul/val_rul` | RUL loss component (MSE + asymmetry) |
| `tr_theta/val_theta` | Theta Huber loss component |

**What to watch:**
- `gnorm` > 5 early in training → increase `grad_clip_norm` to 0.5 or lower LR
- `gnorm` stays at exactly `grad_clip_norm` every epoch → clipping is active,
  consider lowering LR so clipping is rarely triggered
- `val_rul` increases while `val_theta` decreases → theta supervision is
  hurting RUL generalisation; increase `lambda_theta_warmup_epochs`

---

## 3. S-score: why the number is huge

### The PHM'08 formula

```
S = sum(exp(d/a) - 1)   for d = y_hat - y
    a = 10 if d >= 0 (late / over-estimate)
    a = 13 if d <  0 (early / under-estimate)
```

This is a **SUM** over all prediction windows.

### The scale problem

With N-CMAPSS at stride_eval=1:
- DS01-005 has ~4M dev windows and ~1M test windows
- A model with mean |d|=3 cycles produces S_sum ≈ 4M × 0.3 ≈ 1.2M
- Changing stride from 1 to 5 divides N by 5 → S_sum drops 5×

This makes s_score_sum **useless for cross-experiment comparisons** unless
the stride and split size are identical.

### What to use instead

| Metric | Key in metrics.json | When to use |
|--------|---------------------|-------------|
| `s_score_sum` | `s_score` / `s_score_sum` | Comparing against PHM'08 papers (same dataset) |
| `s_score_mean` | `s_score_mean` | Comparing runs with different strides |
| `s_score_unit_median_mean` | `s_score_unit_median_mean` | Fair per-engine comparison (each unit weighted equally) |

### Typical values (DS01-005, well-trained model)
- `s_score_mean` ≈ 0.1–0.5 (per window)
- `s_score_unit_median_mean` ≈ 0.2–1.0

---

## 4. Prediction Horizon (PH) explained

### Definition
PH = the true RUL at the first time step from which the relative error
stays within alpha (default 0.2 = 20%) **continuously** until end-of-life.

```
PH = target[t_first]   where t_first = min t such that within[t:].all()
```

### Why PH = None

PH = None means no such `t_first` exists.  Three common reasons:

1. **Mid-trajectory spike** (most common with N-CMAPSS):
   The model is accurate near EOL but has a brief large error at some
   intermediate step, breaking the "continuously" requirement.

   Diagnostic: `ph_frac_within_alpha_last50_median` > 0.8 but PH = None.
   → Fix: regularise more; check if spikes correlate with Fc transitions.

2. **Systematic over/under-estimation**:
   The model consistently misses by more than alpha throughout the trajectory.
   Diagnostic: `s_score_unit_median_mean` is large positive (over-estimation).

3. **Model collapse** (predicting constant):
   Diagnostic: scatter plot shows horizontal band; `rmse` is close to
   std(y_true).

### PH debug stats in metrics.json

| Key | Meaning |
|-----|---------|
| `ph_frac_within_alpha_last50_median` | Median unit fraction: last 50 windows within alpha |
| `ph_frac_within_alpha_last100_median` | Same for last 100 windows |
| `ph_frac_within_alpha_last200_median` | Same for last 200 windows |
| `max_abs_error_unit_median` | Median (across units) of per-unit worst error |
| `p95_abs_error_unit_median` | Median (across units) of per-unit 95th-pct error |

---

## 5. Diagnostic workflow

```
                   ┌──────────────────────────────────────────┐
                   │ Run evaluate.py → metrics.json           │
                   └────────────────────┬─────────────────────┘
                                        │
                     ┌──────────────────┼────────────────────┐
                     ▼                  ▼                     ▼
              PH = None?          gnorm spikes?         s_score_mean
                     │                  │                  too large?
            ┌────────┴────────┐         │                     │
            │                 │    Lower LR or          Use per-unit
     frac_last50       frac_last50  grad_clip_norm        median_mean
      HIGH (>0.8)       LOW (<0.4)
            │                 │
     Spike problem      Bias/collapse
    analyze spike_locator  Check scatter
    for Fc correlation      plot
```

**Quick commands:**
```bash
# Train (dense, GPU recommended)
python scripts/train.py --config configs/cyclelayer.yaml --model cyclelayer_v1

# Evaluate + extended metrics
python scripts/evaluate.py --config configs/cyclelayer.yaml --split dev

# Visual diagnostics
python scripts/analyze_predictions.py \
    --pred_csv runs/cyclelayer_v1/predictions_dev.csv \
    --out_dir  results/analysis_dev

# Inspect HDF5 structure
python scripts/inspect_hdf5.py data/NCMAPSS/N-CMAPSS_DS01-005.h5 --units --stats
```

---

## 6. HDF5 data flow: from file to window

```
N-CMAPSS HDF5
 ├── W_dev   (N×4)  flight conditions    ┐
 ├── X_s_dev (N×14) measured sensors     ├─ concatenated → (N×18) input
 ├── T_dev   (N×10) health params        │  → theta_true for supervision
 ├── Y_dev   (N×1)  RUL labels           → target
 └── A_dev   (N×4)  [unit, cycle, Fc, hs]→ unit_ids for grouping
                         │
                 NCMAPSSDataset
                         │
              sliding window (size=30, stride=stride_train)
                         │
              DataLoader batch (B×18×30)
                         │
                    Encoder (1D-CNN)
                         │
                  theta_hat (B×10)
                         │
               PrognosticsHead → RUL_pred (B,)
```

The `time_index` in predictions.csv is the window index **after** striding.
- stride_eval=1: time_index corresponds directly to cycle offset from start
- stride_eval=k: multiply time_index by k to get approximate cycle offset
```
