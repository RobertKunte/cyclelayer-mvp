# CLAUDE.md – Project context for Claude Code

## Project overview

**CycleLayer MVP** is a research project implementing a physics-informed neural network
for turbofan engine Remaining Useful Life (RUL) prediction.

Core idea: replace a black-box latent space with a differentiable **Brayton cycle layer**
whose outputs are interpretable thermodynamic quantities (thermal efficiency, net work, etc.).

## Key files

| Path | Purpose |
|------|---------|
| `src/cyclelayer/models/brayton_cycle.py` | Differentiable Brayton cycle – the physics core |
| `src/cyclelayer/models/encoder.py` | 1-D CNN sensor encoder → θ parameters |
| `src/cyclelayer/models/cycle_layer.py` | Full architecture + `CycleLayerConfig` |
| `src/cyclelayer/data/ncmapss.py` | N-CMAPSS HDF5 dataset loader |
| `src/cyclelayer/training/losses.py` | `PhysicsInformedLoss` with physics penalties |
| `configs/cyclelayer.yaml` | Main hyper-parameter config |

## Data

- Dataset: **N-CMAPSS** (NASA), HDF5 format
- Place files as `data/DS01.h5`, `data/DS02.h5`, etc.
- **Not committed to git** (`.gitignore` excludes `data/` and `*.h5`)

## Development commands

```bash
# Install in editable mode with dev extras
pip install -e ".[dev]"

# Run all tests
pytest

# Train
python scripts/train.py --config configs/cyclelayer.yaml

# Evaluate
python scripts/evaluate.py --config configs/cyclelayer.yaml
```

## Conventions

- Python 3.11+, `src/` layout
- Type hints on all public functions
- No implicit `torch.no_grad()` outside evaluation code
- Physics layer must remain **fully differentiable** — no `numpy` calls inside `forward()`
- Cycle parameter layout in θ is fixed: `[T1, π, T3, η_c, η_t, *extra]`
- All temperatures in Kelvin, pressure ratio dimensionless, efficiencies in (0, 1]

## Open research questions

- Does the physics constraint actually improve out-of-distribution generalisation?
- Which operating condition flight class (Fc) benefits most from the cycle layer?
- Can we extend to a two-spool turbofan model (fan + LPC + HPC)?
