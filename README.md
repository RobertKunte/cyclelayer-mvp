# CycleLayer MVP

Physics-informed neural network for turbofan engine **Remaining Useful Life (RUL)** prediction,
combining a data-driven sensor encoder with a differentiable **Brayton cycle layer**.

## Architecture

```
Sensor window (B, T, F)
        │
        ▼
  SensorEncoder          ← 1-D CNN + MLP
  (CNN backbone)
        │
        ▼
  θ = [T1, π, T3, η_c, η_t]    ← thermodynamic cycle parameters
        │
        ▼
  BraytonCycleLayer      ← differentiable ideal open Brayton cycle
  (physics layer)
        │
        ▼
  cycle features [W_net, η_th, r_bw, τ2, τ3, τ4, π, Δη]
        │
        ▼
  PrognosticsHead        ← MLP + Softplus
        │
        ▼
  RUL prediction (B,)
```

## Dataset

[N-CMAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) —
NASA's New Commercial Modular Aero-Propulsion System Simulation dataset (HDF5).

Place the downloaded `.h5` files in `data/`.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Train CycleLayerNet
python scripts/train.py --config configs/cyclelayer.yaml

# Train CNN baseline
python scripts/train.py --config configs/baseline.yaml --model cnn

# Evaluate
python scripts/evaluate.py --config configs/cyclelayer.yaml
```

## Run Tests

```bash
pytest
```

## Project Structure

```
src/cyclelayer/
├── data/           – N-CMAPSS loader & preprocessing
├── models/         – Encoder, BraytonCycleLayer, PrognosticsHead, Baselines
├── training/       – Trainer, loss functions
└── evaluation/     – RMSE, s-score, prediction horizon
configs/            – YAML hyper-parameter configs
scripts/            – train.py, evaluate.py
tests/              – pytest unit tests
notebooks/          – data exploration
```

## Loss Function

The physics-informed loss combines:
- **RUL MSE** with asymmetric over-estimation penalty
- **Feasibility penalty**: W\_net ≥ 0 (engine must do positive work)
- **Monotonicity penalty**: thermal efficiency η\_th is non-increasing along a degradation trajectory
