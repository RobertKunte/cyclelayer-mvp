"""One-shot generator for `notebooks/colab_v3_thermal_aux.ipynb`.

Run:
    python scripts/_build_v3_colab_notebook.py

The notebook trains 4 runs (A baseline / B physics-aux only / C physics+θ→RUL),
runs an inference-time ablation as Run D (shuffled θ on the trained C model),
and produces a 4-way comparison.
"""

from __future__ import annotations

import json
from pathlib import Path


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


CELLS: list[dict] = []


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

CELLS.append(md(
    "# CycleLayer V3.1b — thermal-auxiliary experiment matrix (Colab)",
    "",
    "**Scope:** V3.1b is a differentiable thermal AUXILIARY layer, NOT a full",
    "EPR-validated cycle model. See `docs/decisions/ADR-0012-v3-thermal-auxiliary-scope.md`.",
    "",
    "Runs:",
    "* **A** — baseline RUL only (no physics, θ not in RUL head)",
    "* **B** — physics-aux only (L_temp + L_aux + priors active, θ NOT in RUL head)",
    "* **C** — physics-aux + θ feeding RUL head",
    "* **D** — *inference-time* ablation on the trained C model with **shuffled θ**",
    "  (tests whether θ structure is actually used by the prognostics head)",
    "",
    "Hard constraints honored:",
    "* No EPR / pressure loss (asserted by `CycleLayerV3Loss` at construction)",
    "* No supervised L_θ on θ_phys",
    "* No DS02 tuning / no fit_* helpers / no auto parameter selection",
    "",
    "After the comparison, an **identifiability diagnostic phase** (ADR-0013)",
    "runs 7 read-only Tasks to determine whether `θ_η_hpt` / `θ_η_lpt` are",
    "physically identifiable from V3.1b's temperature-only loss. The phase",
    "produces `IDENTIFIABILITY_SUMMARY.md` with a PASS / WEAK / FAIL verdict.",
    "",
    "## Quick-start",
    "1. Runtime → Change runtime type → **GPU** (T4 / L4 / A100 OK)",
    "2. Edit USER CONFIG in Cell 2 (repo URL, HDF5 path on Drive)",
    "3. Run all cells in order",
))


# -----------------------------------------------------------------------------
# Cell 1 — env + GPU check
# -----------------------------------------------------------------------------

CELLS.append(code(
    "import os, sys, subprocess, shutil, json, time, math, textwrap",
    "from pathlib import Path",
    "from datetime import datetime",
    "",
    "def _sh(cmd, check=True, **kw):",
    "    print(f'$ {cmd}')",
    "    return subprocess.run(cmd, shell=True, check=check, **kw)",
    "",
    "try:",
    "    _sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader',",
    "        check=False)",
    "except Exception:",
    "    print('[WARN] no GPU detected.')",
    "import torch",
    "print(f'torch {torch.__version__}  CUDA={torch.cuda.is_available()}')",
    "if torch.cuda.is_available():",
    "    print(f'device: {torch.cuda.get_device_name(0)}')",
))


# -----------------------------------------------------------------------------
# Cell 2 — USER CONFIG
# -----------------------------------------------------------------------------

CELLS.append(code(
    "# ─── USER CONFIG ────────────────────────────────────────────────────────────",
    "REPO_URL    = 'https://github.com/RobertKunte/cyclelayer-mvp.git'  # ← EDIT if needed",
    "REPO_CLONE  = True                                                # set False if repo exists",
    "REPO_ROOT   = Path('/content/cyclelayer-mvp')",
    "",
    "# DS02 HDF5 on Google Drive",
    "DRIVE_DATA_PATH = Path('/content/drive/MyDrive/cyclelayer-mvp/data/NCMAPSS/N-CMAPSS_DS02-006.h5')  # ← EDIT",
    "",
    "# Runtime ─ size of the (still smoke-sized) run.  Scale up if needed.",
    "EPOCHS              = 3",
    "MAX_TRAIN_SAMPLES   = 30000      # ~ 6 % of ~500k train windows",
    "MAX_VAL_SAMPLES     =  6000",
    "BATCH_SIZE          =   256",
    "LR                  =   1.0e-4",
    "SEED                =    42",
    "",
    "# Where to save run artifacts (on Colab disk; copied to Drive at the end)",
    "RUNS_ROOT       = Path('/content/runs_v3_thermal_aux')",
    "DRIVE_RUNS_ROOT = Path('/content/drive/MyDrive/cyclelayer_v3_runs')   # ← EDIT or None",
    "",
    "RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')",
    "print(f'RUN_ID = {RUN_ID}')",
    "for k, v in dict(EPOCHS=EPOCHS, MAX_TRAIN_SAMPLES=MAX_TRAIN_SAMPLES,",
    "                  MAX_VAL_SAMPLES=MAX_VAL_SAMPLES, BATCH_SIZE=BATCH_SIZE,",
    "                  LR=LR, SEED=SEED).items():",
    "    print(f'  {k:18s}= {v}')",
))


# -----------------------------------------------------------------------------
# Cell 3 — Install deps
# -----------------------------------------------------------------------------

CELLS.append(code(
    "_sh('pip install -q -U pip')",
    "_sh('pip install -q torch>=2.2 numpy>=1.26 scipy>=1.12 h5py>=3.10 '",
    "    'matplotlib>=3.8 pandas>=2.2 pyyaml>=6.0 tqdm>=4.66')",
    "print('dependencies installed.')",
))


# -----------------------------------------------------------------------------
# Cell 4 — Clone repo + install editable
# -----------------------------------------------------------------------------

CELLS.append(code(
    "if REPO_CLONE:",
    "    if REPO_ROOT.exists():",
    "        try:",
    "            _sh(f'git -C {REPO_ROOT} pull --ff-only')",
    "        except Exception:",
    "            shutil.rmtree(REPO_ROOT, ignore_errors=True)",
    "            _sh(f'git clone {REPO_URL} {REPO_ROOT}')",
    "    else:",
    "        _sh(f'git clone {REPO_URL} {REPO_ROOT}')",
    "else:",
    "    assert REPO_ROOT.exists(), f'REPO_ROOT not found: {REPO_ROOT}'",
    "",
    "for p in [str(REPO_ROOT / 'src'), str(REPO_ROOT), str(REPO_ROOT / 'scripts')]:",
    "    if p not in sys.path:",
    "        sys.path.insert(0, p)",
    "",
    "print('git commit:', end=' ')",
    "_sh(f'git -C {REPO_ROOT} rev-parse --short HEAD')",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'],",
    "               cwd=str(REPO_ROOT), check=True)",
    "print('package installed (editable).')",
))


# -----------------------------------------------------------------------------
# Cell 5 — Mount Drive + verify HDF5
# -----------------------------------------------------------------------------

CELLS.append(code(
    "from google.colab import drive  # noqa: E402",
    "drive.mount('/content/drive', force_remount=False)",
    "assert DRIVE_DATA_PATH.exists(), f'HDF5 file not found: {DRIVE_DATA_PATH}'",
    "print(f'HDF5: {DRIVE_DATA_PATH}  ({DRIVE_DATA_PATH.stat().st_size / 1e9:.2f} GB)')",
    "",
    "import h5py",
    "with h5py.File(DRIVE_DATA_PATH, 'r') as f:",
    "    keys = list(f.keys())",
    "    print(f'HDF5 keys (n={len(keys)}):', keys[:8], '...')",
    "    assert any('dev' in k for k in keys), f'missing _dev keys: {keys}'",
))


# -----------------------------------------------------------------------------
# Cell 6 — Pytest sanity check
# -----------------------------------------------------------------------------

CELLS.append(code(
    "result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-q'],",
    "                        cwd=str(REPO_ROOT), capture_output=True, text=True)",
    "tail = result.stdout.strip().splitlines()[-5:]",
    "for line in tail: print(line)",
    "assert result.returncode == 0, f'pytest failed; rc={result.returncode}'",
))


# -----------------------------------------------------------------------------
# Cell 7 — Import helpers from the smoke script + repo
# -----------------------------------------------------------------------------

CELLS.append(code(
    "import numpy as np",
    "import pandas as pd",
    "import yaml",
    "import torch",
    "import torch.nn as nn",
    "from torch.utils.data import DataLoader",
    "from scipy import stats as scstats",
    "import matplotlib.pyplot as plt",
    "",
    "# Repo modules",
    "from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset",
    "from cyclelayer.losses import CycleLayerV3Loss, V3LossConfig",
    "from cyclelayer.models import units",
    "from cyclelayer.models.cyclelayer_v3 import CycleLayerV3",
    "from cyclelayer.models.brayton_engine import BraytonEngine",
    "",
    "# Smoke-script helpers (windowing, builders, train-split stats)",
    "from train_cyclelayer_v3_thermal_aux_smoke import (",
    "    NCMAPSSV3WindowedDataset, _collate,",
    "    build_brayton_from_cfg, build_v3_from_cfg,",
    "    fit_sensor_ops_scalers, fit_temp_sigmas_K, fit_lpt_flow_sigma,",
    ")",
    "",
    "# Load base YAML once",
    "BASE_CFG = yaml.safe_load((REPO_ROOT / 'configs' / 'cyclelayer_v3_thermal_aux.yaml')",
    "                          .read_text(encoding='utf-8'))",
    "BASE_CFG['data']['hdf5_path'] = str(DRIVE_DATA_PATH)",
    "print('base cfg loaded; profile:', BASE_CFG.get('profile_name'))",
))


# -----------------------------------------------------------------------------
# Cell 8 — Load DS02 once into memory, compute train-split stats
# -----------------------------------------------------------------------------

CELLS.append(code(
    "print('loading DS02 dev split into memory (one-time, ~600 MB)...')",
    "t0 = time.time()",
    "BASE_DS = NCMAPSSV3Dataset(DRIVE_DATA_PATH, split='dev', load_in_memory=True)",
    "print(f'  loaded in {time.time()-t0:.1f}s; n_rows={len(BASE_DS):,}; '",
    "      f'units={BASE_DS.unit_ids.tolist()}')",
    "",
    "data_cfg = BASE_CFG['data']",
    "TRAIN_UNITS = list(data_cfg['train_units'])",
    "VAL_UNITS   = list(data_cfg['val_units'])",
    "TEST_UNITS  = list(data_cfg['test_units'])",
    "",
    "SCALERS = fit_sensor_ops_scalers(BASE_DS, TRAIN_UNITS)",
    "SIGMA_T = fit_temp_sigmas_K(BASE_DS, TRAIN_UNITS)",
    "SIGMA_LPT = fit_lpt_flow_sigma(BASE_DS, TRAIN_UNITS)",
    "print(f'sigma_T24_K={SIGMA_T[\"T24\"]:.2f}  sigma_T30_K={SIGMA_T[\"T30\"]:.2f}  '",
    "      f'sigma_T50_K={SIGMA_T[\"T50\"]:.2f}')",
    "print(f'sigma_lpt_flow={SIGMA_LPT:.5f}')",
    "",
    "# Window-level datasets shared across runs (training only — we redraw windows for D)",
    "train_ds = NCMAPSSV3WindowedDataset(",
    "    BASE_DS, TRAIN_UNITS,",
    "    window_size=data_cfg['window_size'], stride=data_cfg['stride_train'],",
    "    max_samples=MAX_TRAIN_SAMPLES,",
    ")",
    "val_ds = NCMAPSSV3WindowedDataset(",
    "    BASE_DS, VAL_UNITS,",
    "    window_size=data_cfg['window_size'], stride=data_cfg['stride_eval'],",
    "    max_samples=MAX_VAL_SAMPLES,",
    ")",
    "BASE_DS_TEST = NCMAPSSV3Dataset(DRIVE_DATA_PATH, split='test', load_in_memory=True)",
    "test_ds = NCMAPSSV3WindowedDataset(",
    "    BASE_DS_TEST, TEST_UNITS,",
    "    window_size=data_cfg['window_size'], stride=data_cfg['stride_eval'],",
    "    max_samples=MAX_VAL_SAMPLES,",
    ")",
    "print(f'train windows = {len(train_ds):,}, val = {len(val_ds):,}, test = {len(test_ds):,}')",
    "",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "print('device:', device)",
))


# -----------------------------------------------------------------------------
# Cell 9 — Run-config builder (4 variants A/B/C/D)
# -----------------------------------------------------------------------------

CELLS.append(code(
    "import copy",
    "",
    "def build_run_config(base_cfg: dict, run_name: str) -> dict:",
    "    \"\"\"Return a deep copy of base_cfg with run-specific overrides.\"\"\"",
    "    cfg = copy.deepcopy(base_cfg)",
    "    cfg['training']['max_epochs']           = EPOCHS",
    "    cfg['training']['max_train_samples']    = MAX_TRAIN_SAMPLES",
    "    cfg['training']['max_val_samples']      = MAX_VAL_SAMPLES",
    "    cfg['training']['batch_size']           = BATCH_SIZE",
    "    cfg['training']['lr']                   = LR",
    "    cfg['training']['output_dir']           = str(RUNS_ROOT / f'{RUN_ID}_{run_name}')",
    "",
    "    if run_name == 'A_baseline':",
    "        # No physics loss; θ NOT in RUL head; AuxHead NOT in RUL head",
    "        cfg['loss']['lambda_temp']    = 0.0",
    "        cfg['loss']['lambda_aux']     = 0.0",
    "        cfg['loss']['lambda_healthy'] = 0.0",
    "        cfg['loss']['lambda_smooth']  = 0.0",
    "        cfg['model']['prognostics_head']['use_theta_in_rul']   = False",
    "        cfg['model']['aux_health_head']['detach_for_rul']      = True   # not used anyway",
    "    elif run_name == 'B_physics_aux':",
    "        # Physics loss active; θ NOT in RUL head (theta emerges from sensor-only)",
    "        cfg['model']['prognostics_head']['use_theta_in_rul']   = False",
    "    elif run_name == 'C_physics_theta_rul':",
    "        # Physics + θ feeds RUL head",
    "        cfg['model']['prognostics_head']['use_theta_in_rul']   = True",
    "        cfg['model']['aux_health_head']['detach_for_rul']      = True",
    "    else:",
    "        raise ValueError(f'unknown run_name {run_name}')",
    "    return cfg",
    "",
    "RUN_CFGS = {",
    "    'A_baseline':           build_run_config(BASE_CFG, 'A_baseline'),",
    "    'B_physics_aux':        build_run_config(BASE_CFG, 'B_physics_aux'),",
    "    'C_physics_theta_rul':  build_run_config(BASE_CFG, 'C_physics_theta_rul'),",
    "}",
    "for k, v in RUN_CFGS.items():",
    "    print(f'{k:24s}  '",
    "          f'lambda_temp={v[\"loss\"][\"lambda_temp\"]}  '",
    "          f'use_theta_in_rul={v[\"model\"][\"prognostics_head\"][\"use_theta_in_rul\"]}  '",
    "          f'out={v[\"training\"][\"output_dir\"]}')",
))


# -----------------------------------------------------------------------------
# Cell 10 — Training loop (in-notebook, reuses smoke-script builders)
# -----------------------------------------------------------------------------

CELLS.append(code(
    "def build_loss_from_cfg(cfg: dict) -> CycleLayerV3Loss:",
    "    lc = cfg['loss']",
    "    return CycleLayerV3Loss(V3LossConfig(",
    "        lambda_rul     = float(lc['lambda_rul']),",
    "        lambda_temp    = float(lc['lambda_temp']),",
    "        lambda_aux     = float(lc['lambda_aux']),",
    "        lambda_healthy = float(lc['lambda_healthy']),",
    "        lambda_smooth  = float(lc['lambda_smooth']),",
    "        mse_weight     = float(lc['rul']['mse_weight']),",
    "        asymmetry      = float(lc['rul']['asymmetry']),",
    "        temp_sensors   = list(lc['temp_sensors']),",
    "        sigma_temp_K   = SIGMA_T,",
    "        sigma_lpt_flow = SIGMA_LPT,",
    "        healthy_rul_threshold = float(lc['healthy_rul_threshold']),",
    "        use_pressure_loss = False,",
    "        use_epr_loss      = False,",
    "    ))",
    "",
    "",
    "def build_model_from_cfg(cfg: dict) -> CycleLayerV3:",
    "    brayton = build_brayton_from_cfg(cfg['model']['brayton_engine'])",
    "    return build_v3_from_cfg(cfg['model'], brayton)",
    "",
    "",
    "def run_epoch(model, loss_fn, loader, optimizer, device, scalers, tag, epoch):",
    "    is_train = optimizer is not None",
    "    model.train(is_train)",
    "    sm = scalers['sensor_mean'].to(device); sd = scalers['sensor_std'].to(device)",
    "    om = scalers['ops_mean'].to(device);    od = scalers['ops_std'].to(device)",
    "    total = 0.0; comps_sum = {}; rul_errs = []; theta_all = []",
    "    t24 = []; t30 = []; t50 = []; epr = []; n = 0",
    "    for batch in loader:",
    "        sensors_norm = (batch['sensors_imp'].to(device) - sm) / sd",
    "        ops_norm     = (batch['ops_imp'].to(device)     - om) / od",
    "        ops_si  = {k: v.to(device) for k, v in batch['ops_si_last'].items()}",
    "        sens_si = {k: v.to(device) for k, v in batch['sens_si_last'].items()}",
    "        temp_true = {k: v.to(device) for k, v in batch['targets_K_last'].items()}",
    "        rul_true  = batch['RUL'].to(device)",
    "        lpt_flow_true = batch['health_gt_last']['LPT_flow_mod'].to(device)",
    "        if is_train: optimizer.zero_grad(set_to_none=True)",
    "        out = model(sensors_norm, ops_norm, ops_si=ops_si, sens_si=sens_si)",
    "        temp_preds = {k: out['brayton']['sensors_pred_si'][k]",
    "                      for k in ('T24_K', 'T30_K', 'T50_K')}",
    "        L, c = loss_fn(",
    "            rul_pred=out['rul'], rul_true=rul_true,",
    "            theta_phys=out['theta_phys'],",
    "            lpt_flow_pred=out['lpt_flow_pred'], lpt_flow_true=lpt_flow_true,",
    "            temp_preds_K=temp_preds, temp_true_K=temp_true,",
    "        )",
    "        if is_train:",
    "            L.backward()",
    "            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)",
    "            optimizer.step()",
    "        total += float(L.item())",
    "        for k, v in c.items():",
    "            comps_sum[k] = comps_sum.get(k, 0.0) + float(v.item())",
    "        rul_errs.append(float((out['rul'] - rul_true).detach().pow(2).mean().sqrt().item()))",
    "        theta_all.append(out['theta_phys'].detach().cpu())",
    "        t24.append(float((temp_preds['T24_K'] - temp_true['T24_K']).abs().mean().item()))",
    "        t30.append(float((temp_preds['T30_K'] - temp_true['T30_K']).abs().mean().item()))",
    "        t50.append(float((temp_preds['T50_K'] - temp_true['T50_K']).abs().mean().item()))",
    "        epr.append(float((out['brayton']['diag']['P50'] / ops_si['P2_Pa']).mean().item()))",
    "        n += 1",
    "    theta = torch.cat(theta_all, dim=0)",
    "    return {",
    "        'epoch': epoch, 'tag': tag,",
    "        'loss': total/max(n,1),",
    "        'components': {k: v/max(n,1) for k, v in comps_sum.items()},",
    "        'rul_rmse': float(np.mean(rul_errs)),",
    "        'T24_mae_K': float(np.mean(t24)),",
    "        'T30_mae_K': float(np.mean(t30)),",
    "        'T50_mae_K': float(np.mean(t50)),",
    "        'theta_stats': {",
    "            'mean': float(theta.mean()),'std': float(theta.std()),",
    "            'min':  float(theta.min()), 'max': float(theta.max()),",
    "            'frac_at_lo': float((theta <= 0.851).float().mean()),",
    "            'frac_at_hi': float((theta >= 0.999).float().mean()),",
    "        },",
    "        'EPR_pred_mean_diagnostic_only': float(np.mean(epr)),",
    "    }",
    "",
    "",
    "def train_run(cfg: dict, run_name: str):",
    "    torch.manual_seed(SEED)",
    "    np.random.seed(SEED)",
    "    out_dir = Path(cfg['training']['output_dir'])",
    "    out_dir.mkdir(parents=True, exist_ok=True)",
    "",
    "    model = build_model_from_cfg(cfg).to(device)",
    "    loss_fn = build_loss_from_cfg(cfg)",
    "    optimizer = torch.optim.AdamW(",
    "        model.parameters(), lr=float(cfg['training']['lr']),",
    "        weight_decay=float(cfg['training']['weight_decay']),",
    "    )",
    "    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,",
    "                              num_workers=0, collate_fn=_collate)",
    "    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,",
    "                            num_workers=0, collate_fn=_collate)",
    "",
    "    scalers = {",
    "        'sensor_mean': torch.from_numpy(SCALERS['sensor_mean']).float(),",
    "        'sensor_std':  torch.from_numpy(SCALERS['sensor_std']).float(),",
    "        'ops_mean':    torch.from_numpy(SCALERS['ops_mean']).float(),",
    "        'ops_std':     torch.from_numpy(SCALERS['ops_std']).float(),",
    "    }",
    "    np.savez(out_dir / 'sensor_scaler.npz',",
    "             mean=SCALERS['sensor_mean'], std=SCALERS['sensor_std'])",
    "    np.savez(out_dir / 'ops_scaler.npz',",
    "             mean=SCALERS['ops_mean'], std=SCALERS['ops_std'])",
    "    (out_dir / 'sigma_train.json').write_text(json.dumps({",
    "        'sigma_T_K': SIGMA_T, 'sigma_lpt_flow': SIGMA_LPT,",
    "        'train_units': TRAIN_UNITS, 'val_units': VAL_UNITS,",
    "        'test_units': TEST_UNITS,",
    "    }, indent=2))",
    "",
    "    print(f'\\n[{run_name}]  trainable_params = '",
    "          f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')",
    "    print(f'[{run_name}]  out_dir = {out_dir}')",
    "    history = []; best = float(\"inf\")",
    "    n_epochs = int(cfg['training']['max_epochs'])",
    "    for ep in range(1, n_epochs + 1):",
    "        t0 = time.time()",
    "        tr = run_epoch(model, loss_fn, train_loader, optimizer, device,",
    "                       scalers, 'train', ep)",
    "        with torch.no_grad():",
    "            va = run_epoch(model, loss_fn, val_loader, None, device,",
    "                           scalers, 'val', ep)",
    "        dt = time.time() - t0",
    "        ts = tr['theta_stats']; vs = va['theta_stats']",
    "        print(f'  ep {ep}/{n_epochs}  tr_loss={tr[\"loss\"]:.3f}  val_loss={va[\"loss\"]:.3f}  '",
    "              f'tr_RMSE={tr[\"rul_rmse\"]:.2f}  val_RMSE={va[\"rul_rmse\"]:.2f}  '",
    "              f'θ(tr/val mean)={ts[\"mean\"]:.4f}/{vs[\"mean\"]:.4f}  '",
    "              f'val_EPR(diag)={va[\"EPR_pred_mean_diagnostic_only\"]:.3f}  ({dt:.1f}s)')",
    "        history.append({'train': tr, 'val': va, 'elapsed_s': dt})",
    "        if va['loss'] < best:",
    "            best = va['loss']",
    "            torch.save({",
    "                'model': model.state_dict(),",
    "                'scalers': {k: v.tolist() for k, v in scalers.items()},",
    "                'sigma_T_K': SIGMA_T, 'sigma_lpt_flow': SIGMA_LPT,",
    "                'epoch': ep, 'val_loss': best,",
    "                'config': cfg, 'run_name': run_name,",
    "            }, out_dir / 'best.pt')",
    "    torch.save(model.state_dict(), out_dir / 'last.pt')",
    "    (out_dir / 'history.json').write_text(json.dumps(history, indent=2,",
    "                                                    default=float))",
    "    return {'model': model, 'history': history, 'out_dir': out_dir,",
    "            'best_val_loss': best, 'cfg': cfg, 'scalers': scalers}",
))


# -----------------------------------------------------------------------------
# Cells 11–13 — Train Runs A, B, C
# -----------------------------------------------------------------------------

for run_name in ("A_baseline", "B_physics_aux", "C_physics_theta_rul"):
    CELLS.append(md(f"## Run `{run_name}`"))
    CELLS.append(code(
        f"RESULTS_{run_name.upper()} = train_run(RUN_CFGS['{run_name}'], '{run_name}')",
    ))


# -----------------------------------------------------------------------------
# Cell — Run D ablation (inference-time shuffled theta on C model)
# -----------------------------------------------------------------------------

CELLS.append(md(
    "## Run `D_shuffled_theta` — inference-time ablation on the trained C model",
    "",
    "Loads the best `C_physics_theta_rul` checkpoint and runs inference twice on the test "
    "split: once normally, once with `theta_phys` randomly permuted across the batch dimension. "
    "If the RUL accuracy is unchanged, θ is not actually contributing useful information to the "
    "prognostics head. If RUL accuracy drops significantly under shuffled θ, θ encodes signal.",
))

CELLS.append(code(
    "@torch.no_grad()",
    "def infer_rul(model, loader, device, scalers, shuffle_theta: bool):",
    "    model.eval()",
    "    sm = scalers['sensor_mean'].to(device); sd = scalers['sensor_std'].to(device)",
    "    om = scalers['ops_mean'].to(device);    od = scalers['ops_std'].to(device)",
    "    rul_preds = []; rul_trues = []",
    "    for batch in loader:",
    "        sensors_norm = (batch['sensors_imp'].to(device) - sm) / sd",
    "        ops_norm     = (batch['ops_imp'].to(device)     - om) / od",
    "        ops_si  = {k: v.to(device) for k, v in batch['ops_si_last'].items()}",
    "        sens_si = {k: v.to(device) for k, v in batch['sens_si_last'].items()}",
    "        out = model(sensors_norm, ops_norm, ops_si=ops_si, sens_si=sens_si)",
    "        if shuffle_theta:",
    "            # Re-run the prognostics head with a permuted theta_phys.",
    "            # Build the same RUL feature vector the model would build internally.",
    "            B = out['theta_phys'].shape[0]",
    "            perm = torch.randperm(B, device=device)",
    "            theta_shuf = out['theta_phys'][perm]",
    "            # Reconstruct RUL feature parts (mirrors CycleLayerV3.forward)",
    "            cfg = model.config",
    "            parts = [out['h_sens']]",
    "            if out['z_ops'] is not None: parts.append(out['z_ops'])",
    "            if cfg.use_theta_in_rul: parts.append(theta_shuf)",
    "            if cfg.use_aux_in_rul:",
    "                aux = out['lpt_flow_pred']",
    "                parts.append(aux.unsqueeze(-1))",
    "            rul_p = model.prognostics(torch.cat(parts, dim=-1))",
    "        else:",
    "            rul_p = out['rul']",
    "        rul_preds.append(rul_p.cpu())",
    "        rul_trues.append(batch['RUL'])",
    "    p = torch.cat(rul_preds).numpy()",
    "    t = torch.cat(rul_trues).numpy()",
    "    return p, t",
    "",
    "",
    "def metrics(p, t):",
    "    err = p - t",
    "    return {",
    "        'RMSE': float(np.sqrt((err**2).mean())),",
    "        'MAE':  float(np.abs(err).mean()),",
    "        'bias': float(err.mean()),",
    "    }",
    "",
    "C_model    = RESULTS_C_PHYSICS_THETA_RUL['model']",
    "C_scalers  = RESULTS_C_PHYSICS_THETA_RUL['scalers']",
    "test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,",
    "                         num_workers=0, collate_fn=_collate)",
    "",
    "print('Inference on TEST split with REAL θ ...')",
    "p_real, t_real = infer_rul(C_model, test_loader, device, C_scalers, shuffle_theta=False)",
    "m_real = metrics(p_real, t_real)",
    "print(f'  REAL θ:     RMSE={m_real[\"RMSE\"]:.3f}  MAE={m_real[\"MAE\"]:.3f}  bias={m_real[\"bias\"]:+.3f}')",
    "",
    "print('Inference on TEST split with SHUFFLED θ ...')",
    "p_shuf, t_shuf = infer_rul(C_model, test_loader, device, C_scalers, shuffle_theta=True)",
    "m_shuf = metrics(p_shuf, t_shuf)",
    "print(f'  SHUFFLED θ: RMSE={m_shuf[\"RMSE\"]:.3f}  MAE={m_shuf[\"MAE\"]:.3f}  bias={m_shuf[\"bias\"]:+.3f}')",
    "",
    "delta = m_shuf['RMSE'] - m_real['RMSE']",
    "if delta > 1.0:",
    "    verdict = f'θ DOES carry useful information (shuffling raised RMSE by {delta:+.2f})'",
    "elif delta < -0.5:",
    "    verdict = f'θ may HURT prognostics (shuffling LOWERED RMSE by {-delta:.2f})'",
    "else:",
    "    verdict = f'θ has negligible effect on RUL (Δ RMSE = {delta:+.2f})'",
    "print(f'\\nVerdict: {verdict}')",
    "",
    "# Save ablation results",
    "D_out_dir = RUNS_ROOT / f'{RUN_ID}_D_shuffled_theta'",
    "D_out_dir.mkdir(parents=True, exist_ok=True)",
    "(D_out_dir / 'metrics.json').write_text(json.dumps({",
    "    'metrics_real':     m_real,",
    "    'metrics_shuffled': m_shuf,",
    "    'delta_RMSE':       delta,",
    "    'verdict':          verdict,",
    "}, indent=2))",
    "print(f'saved {D_out_dir / \"metrics.json\"}')",
))


# -----------------------------------------------------------------------------
# Cell — θ diagnostics for B and C on test split
# -----------------------------------------------------------------------------

CELLS.append(md("## θ diagnostics on the TEST split (post-hoc) — Runs B and C"))

CELLS.append(code(
    "@torch.no_grad()",
    "def collect_predictions(model, loader, device, scalers):",
    "    model.eval()",
    "    sm = scalers['sensor_mean'].to(device); sd = scalers['sensor_std'].to(device)",
    "    om = scalers['ops_mean'].to(device);    od = scalers['ops_std'].to(device)",
    "    rows = {",
    "        'theta_eta_fan': [], 'theta_eta_lpc': [], 'theta_eta_hpc': [],",
    "        'theta_eta_hpt': [], 'theta_eta_lpt': [],",
    "        'lpt_flow_pred': [],",
    "        'HPT_eff_mod_GT': [], 'LPT_eff_mod_GT': [], 'LPT_flow_mod_GT': [],",
    "        'RUL': [], 'unit_id': [], 'rul_pred': [],",
    "    }",
    "    for batch in loader:",
    "        sensors_norm = (batch['sensors_imp'].to(device) - sm) / sd",
    "        ops_norm     = (batch['ops_imp'].to(device)     - om) / od",
    "        ops_si  = {k: v.to(device) for k, v in batch['ops_si_last'].items()}",
    "        sens_si = {k: v.to(device) for k, v in batch['sens_si_last'].items()}",
    "        out = model(sensors_norm, ops_norm, ops_si=ops_si, sens_si=sens_si)",
    "        theta = out['theta_phys'].cpu().numpy()",
    "        rows['theta_eta_fan'].extend(theta[:, 0].tolist())",
    "        rows['theta_eta_lpc'].extend(theta[:, 1].tolist())",
    "        rows['theta_eta_hpc'].extend(theta[:, 2].tolist())",
    "        rows['theta_eta_hpt'].extend(theta[:, 3].tolist())",
    "        rows['theta_eta_lpt'].extend(theta[:, 4].tolist())",
    "        rows['lpt_flow_pred'].extend(out['lpt_flow_pred'].cpu().numpy().tolist())",
    "        h = batch['health_gt_last']",
    "        rows['HPT_eff_mod_GT'].extend(h['HPT_eff_mod'].numpy().tolist())",
    "        rows['LPT_eff_mod_GT'].extend(h['LPT_eff_mod'].numpy().tolist())",
    "        rows['LPT_flow_mod_GT'].extend(h['LPT_flow_mod'].numpy().tolist())",
    "        rows['RUL'].extend(batch['RUL'].numpy().tolist())",
    "        rows['unit_id'].extend(batch['unit_id'].numpy().tolist())",
    "        rows['rul_pred'].extend(out['rul'].cpu().numpy().tolist())",
    "    return pd.DataFrame(rows)",
    "",
    "def safe_corr(x, y):",
    "    x, y = np.asarray(x), np.asarray(y)",
    "    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:",
    "        return float('nan'), float('nan')",
    "    return (float(scstats.pearsonr(x, y).statistic),",
    "            float(scstats.spearmanr(x, y).statistic))",
    "",
    "diag = {}",
    "for run_name, results_var in (",
    "    ('B_physics_aux',       'RESULTS_B_PHYSICS_AUX'),",
    "    ('C_physics_theta_rul', 'RESULTS_C_PHYSICS_THETA_RUL'),",
    "):",
    "    r = globals()[results_var]",
    "    df = collect_predictions(r['model'], test_loader, device, r['scalers'])",
    "    df['theta_eta_hpt_delta'] = df['theta_eta_hpt'] - 1.0",
    "    df['theta_eta_lpt_delta'] = df['theta_eta_lpt'] - 1.0",
    "    pH, sH = safe_corr(df['theta_eta_hpt_delta'], df['HPT_eff_mod_GT'])",
    "    pL, sL = safe_corr(df['theta_eta_lpt_delta'], df['LPT_eff_mod_GT'])",
    "    pF, sF = safe_corr(df['lpt_flow_pred'],       df['LPT_flow_mod_GT'])",
    "    diag[run_name] = {'df': df,",
    "                       'HPT': (pH, sH),",
    "                       'LPT': (pL, sL),",
    "                       'LPT_flow_aux': (pF, sF)}",
    "    print(f'\\n[{run_name}]  N={len(df):,}')",
    "    print(f'  Pearson(θ_η_hpt − 1, HPT_eff_mod) = {pH:.3f}  '",
    "          f'Spearman = {sH:.3f}')",
    "    print(f'  Pearson(θ_η_lpt − 1, LPT_eff_mod) = {pL:.3f}  '",
    "          f'Spearman = {sL:.3f}')",
    "    print(f'  Pearson(lpt_flow_pred, LPT_flow_mod) = {pF:.3f}  '",
    "          f'Spearman = {sF:.3f}  [supervised]')",
    "    sat_lo = float((df[['theta_eta_fan','theta_eta_lpc','theta_eta_hpc',",
    "                        'theta_eta_hpt','theta_eta_lpt']].to_numpy() <= 0.851).mean())",
    "    sat_hi = float((df[['theta_eta_fan','theta_eta_lpc','theta_eta_hpc',",
    "                        'theta_eta_hpt','theta_eta_lpt']].to_numpy() >= 0.999).mean())",
    "    print(f'  θ saturation: lo={sat_lo:.3f}  hi={sat_hi:.3f}')",
))


# -----------------------------------------------------------------------------
# Cell — 4-way comparison table + plots
# -----------------------------------------------------------------------------

CELLS.append(md("## 4-way comparison (TEST split)"))

CELLS.append(code(
    "def final_rmse(run_results, loader):",
    "    p, t = infer_rul(run_results['model'], loader, device,",
    "                     run_results['scalers'], shuffle_theta=False)",
    "    return metrics(p, t)",
    "",
    "test_metrics = {}",
    "for run_name, results_var in (",
    "    ('A_baseline',          'RESULTS_A_BASELINE'),",
    "    ('B_physics_aux',       'RESULTS_B_PHYSICS_AUX'),",
    "    ('C_physics_theta_rul', 'RESULTS_C_PHYSICS_THETA_RUL'),",
    "):",
    "    test_metrics[run_name] = final_rmse(globals()[results_var], test_loader)",
    "test_metrics['D_shuffled_theta'] = m_shuf",
    "",
    "rows = []",
    "for run_name in ('A_baseline', 'B_physics_aux', 'C_physics_theta_rul', 'D_shuffled_theta'):",
    "    m = test_metrics[run_name]",
    "    pH = diag.get(run_name, {}).get('HPT', (float('nan'), float('nan')))[0]",
    "    pL = diag.get(run_name, {}).get('LPT', (float('nan'), float('nan')))[0]",
    "    pF = diag.get(run_name, {}).get('LPT_flow_aux', (float('nan'), float('nan')))[0]",
    "    rows.append({",
    "        'run':              run_name,",
    "        'test_RUL_RMSE':    m['RMSE'],",
    "        'test_RUL_MAE':     m['MAE'],",
    "        'test_RUL_bias':    m['bias'],",
    "        'Pearson(θ_hpt, HPT_eff_mod)': pH,",
    "        'Pearson(θ_lpt, LPT_eff_mod)': pL,",
    "        'Pearson(aux, LPT_flow_mod)':  pF,",
    "    })",
    "df_compare = pd.DataFrame(rows)",
    "print(df_compare.to_string(index=False))",
    "",
    "# Save",
    "compare_dir = RUNS_ROOT / f'{RUN_ID}_comparison'",
    "compare_dir.mkdir(parents=True, exist_ok=True)",
    "df_compare.to_csv(compare_dir / 'comparison.csv', index=False)",
    "print(f'saved {compare_dir / \"comparison.csv\"}')",
))


# -----------------------------------------------------------------------------
# Cell — Plots
# -----------------------------------------------------------------------------

CELLS.append(code(
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))",
    "",
    "ax = axes[0]",
    "names = df_compare['run'].tolist()",
    "rmse = df_compare['test_RUL_RMSE'].tolist()",
    "colors = ['tab:gray', 'tab:blue', 'tab:green', 'tab:red']",
    "ax.bar(names, rmse, color=colors, edgecolor='black')",
    "ax.set_ylabel('test RUL RMSE'); ax.set_title('Test RUL RMSE')",
    "ax.tick_params(axis='x', rotation=20)",
    "for i, v in enumerate(rmse):",
    "    ax.annotate(f'{v:.2f}', (i, v), xytext=(0, 4), ha='center',",
    "                textcoords='offset points', fontsize=9)",
    "",
    "ax = axes[1]",
    "pH = df_compare['Pearson(θ_hpt, HPT_eff_mod)'].tolist()",
    "pL = df_compare['Pearson(θ_lpt, LPT_eff_mod)'].tolist()",
    "x = np.arange(len(names)); w = 0.35",
    "ax.bar(x - w/2, pH, w, label='θ_hpt vs HPT_eff_mod', color='tab:red',  edgecolor='black')",
    "ax.bar(x + w/2, pL, w, label='θ_lpt vs LPT_eff_mod', color='tab:blue', edgecolor='black')",
    "ax.axhline(0.4, ls='--', color='black', lw=0.7, label='good band 0.4')",
    "ax.axhline(0.7, ls=':',  color='black', lw=0.7, label='stretch 0.7')",
    "ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)",
    "ax.set_ylabel('Pearson r (post-hoc)')",
    "ax.set_title('θ-delta vs N-CMAPSS GT (post-hoc)')",
    "ax.legend(fontsize=7); ax.grid(True, axis='y', alpha=0.3)",
    "",
    "ax = axes[2]",
    "# Plot the RUL pred vs RUL true scatter for run C and run D (shuffled)",
    "for tag, p, t, c in (",
    "    ('C real θ',     p_real, t_real, 'tab:green'),",
    "    ('D shuffled θ', p_shuf, t_shuf, 'tab:red'),",
    "):",
    "    ax.scatter(t, p, s=6, alpha=0.35, c=c, label=tag)",
    "ax.plot([0, 99], [0, 99], 'k--', lw=0.5)",
    "ax.set_xlabel('true RUL'); ax.set_ylabel('predicted RUL')",
    "ax.set_title('Run C vs Run D ablation')",
    "ax.legend(); ax.grid(True, alpha=0.3)",
    "",
    "fig.suptitle(f'V3.1b thermal-aux experiment matrix — {RUN_ID}')",
    "fig.tight_layout()",
    "fig.savefig(compare_dir / 'comparison.png', dpi=110, bbox_inches='tight')",
    "plt.show()",
    "print(f'saved {compare_dir / \"comparison.png\"}')",
))


# -----------------------------------------------------------------------------
# Cell — Markdown summary
# -----------------------------------------------------------------------------

CELLS.append(code(
    "summary = textwrap.dedent(f'''",
    "# V3.1b thermal-aux experiment matrix — {RUN_ID}",
    "",
    "*Read-only diagnostic. No YAML written. See ADR-0012.*",
    "",
    "## Test split (DS02 units {TEST_UNITS}) — 4-way comparison",
    "",
    "{df_compare.to_markdown(index=False)}",
    "",
    "## D ablation verdict",
    "",
    "* RMSE with REAL θ:     {m_real[\"RMSE\"]:.3f}",
    "* RMSE with SHUFFLED θ: {m_shuf[\"RMSE\"]:.3f}",
    "* Δ RMSE: {delta:+.3f}",
    "* {verdict}",
    "",
    "## Constraints honored",
    "* No EPR / pressure loss (asserted by `CycleLayerV3Loss`)",
    "* No supervised L_θ on θ_phys",
    "* No DS02 tuning / no fit_* helpers / no auto parameter selection",
    "* Test units NOT used for training (units {TEST_UNITS} held out)",
    "",
    "## Artifacts",
    "* {RUNS_ROOT / (RUN_ID + \"_A_baseline\")}",
    "* {RUNS_ROOT / (RUN_ID + \"_B_physics_aux\")}",
    "* {RUNS_ROOT / (RUN_ID + \"_C_physics_theta_rul\")}",
    "* {RUNS_ROOT / (RUN_ID + \"_D_shuffled_theta\")}",
    "* comparison.csv + comparison.png in {compare_dir}",
    "''')",
    "(compare_dir / 'comparison.md').write_text(summary, encoding='utf-8')",
    "print(summary)",
))


# -----------------------------------------------------------------------------
# Cell — Save to Drive
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Identifiability diagnostic phase (ADR-0013)
# -----------------------------------------------------------------------------

CELLS.append(md(
    "# Identifiability diagnostics (ADR-0013)",
    "",
    "Following the experiment matrix above (Run C produced strong-magnitude but",
    "*wrong-sign* Pearson and the D-ablation showed θ has no RUL effect),",
    "[ADR-0013](docs/decisions/ADR-0013-v31b-theta-identifiability-tests.md) requires",
    "a structured diagnostic suite before any further claim about HPT/LPT θ",
    "identifiability.  The cells below run all seven Tasks and aggregate the verdict.",
    "",
    "Hard constraints (enforced by the scripts):",
    "* No DS02 / C0 / C1 / C2 parameter tuning",
    "* No YAML physical-constant writes",
    "* No `fit_*` helper on real data (synthetic recovery is the only optimisation, and only on synthetic targets)",
    "* No supervised L_θ on θ_phys",
    "* Pressure / EPR loss remains disabled in the V3.1b training path",
    "",
    "Outputs land under `artifacts/cyclelayer_v3/theta_identifiability/`.",
))

CELLS.append(code(
    "# Helper — run a script via subprocess with clean output streaming",
    "def _run_diag(script, args=()):",
    "    cmd = [sys.executable, script, *map(str, args)]",
    "    print('$', ' '.join(cmd))",
    "    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)",
    "    # show last lines of stdout for context",
    "    lines = res.stdout.splitlines()",
    "    for line in lines[-25:]:",
    "        print(line)",
    "    if res.returncode != 0:",
    "        print('-- STDERR --')",
    "        for line in res.stderr.splitlines()[-30:]:",
    "            print(line)",
    "    return res.returncode == 0",
    "",
    "# Latest C-run dir (used by Tasks 4 + 6)",
    "C_RUN_DIR = RUN_DIR_C = Path(RUN_CFGS['C_physics_theta_rul']['training']['output_dir'])",
    "print(f'C run dir: {C_RUN_DIR}')",
    "print(f'C best.pt exists: {(C_RUN_DIR / \"best.pt\").exists()}')",
))

CELLS.append(md(
    "## Task 2 — Local sensitivity Jacobian",
    "",
    "Measures `d{T, P, PR, EPR} / dθ` at FC02 + DS02 sample points via autograd."
    " Establishes whether each θ has measurable effect on temperature vs"
    " pressure outputs.",
))
CELLS.append(code(
    "_run_diag('scripts/diagnose_v31b_theta_local_sensitivity.py')",
))

CELLS.append(md(
    "## Task 3 — Loss gradient pathways",
    "",
    "Measures `∂L_temp / ∂θ` (and per-component gradient norms for L_rul, L_aux,"
    " L_healthy, L_smooth, L_total) on a synthetic batch + DS02 batch.",
))
CELLS.append(code(
    "_run_diag('scripts/diagnose_v31b_loss_gradient_paths.py')",
))

CELLS.append(md(
    "## Task 5 — N-CMAPSS sign convention",
    "",
    "Confirms what 'degraded' means numerically for the 10 health modifiers"
    " across DS02 unit life. Sets the EXPECTED sign of `Pearson(θ−1, GT)`.",
))
CELLS.append(code(
    "_run_diag('scripts/inspect_ncmapss_health_sign_convention.py')",
))

CELLS.append(md(
    "## Task 4 — Partial correlations",
    "",
    "Raw vs residualised `Pearson(θ, GT)` controlling for {RUL, cycle, ops, combinations}."
    " If raw |r| > 0.6 and partial |r| < 0.2 → strong evidence of time/RUL-axis artifact.",
    "",
    "*(Uses the C run's checkpoint — auto-discovers `/content/runs_v3_thermal_aux/<RUN_ID>_C_physics_theta_rul/`.)*",
))
CELLS.append(code(
    "_run_diag('scripts/diagnose_v31b_theta_partial_correlations.py',",
    "          args=['--run_dir', str(C_RUN_DIR)])",
))

CELLS.append(md(
    "## Task 6 — Does the RUL head use θ_phys?",
    "",
    "Perturbation tests: {real, shuffle_batch, shuffle_within_unit, shuffle_across_units,"
    " constant_healthy, constant_lo}. Reports ΔRMSE per variant + the first PrognosticsHead"
    " Linear's column-norm on the θ slots.",
    "",
    "*(Uses the C run's checkpoint.)*",
))
CELLS.append(code(
    "_run_diag('scripts/diagnose_v31b_rul_theta_usage.py',",
    "          args=['--run_dir', str(C_RUN_DIR)])",
))

CELLS.append(md(
    "## Task 7 — Synthetic θ recovery (CONSTRUCTIVE TEST)",
    "",
    "On synthetic data only: known θ → BraytonEngine → outputs → optimise θ_pred to match."
    " Two target sets:",
    "* **Case A:** {T24, T30, T50} only (mirrors V3.1b L_temp)",
    "* **Case B:** {T24, T30, T50, P30, P50} (V4-style)",
    "",
    "Expected if the V3.1b architecture is the bottleneck: HPT/LPT θ do not recover in A but do in B."
    " This is **independent of any trained model or DS02** — pure constructive proof.",
))
CELLS.append(code(
    "_run_diag('scripts/test_v31b_synthetic_theta_recovery.py')",
))

CELLS.append(md(
    "## Task 8 — Aggregate verdict",
    "",
    "Combines Tasks 2-7 into `IDENTIFIABILITY_SUMMARY.md` with the ADR-0013 "
    "PASS / WEAK / FAIL verdict per θ channel and answers to the five questions.",
))
CELLS.append(code(
    "_run_diag('scripts/build_identifiability_summary.py')",
    "",
    "# Display the summary inline",
    "summary_md = REPO_ROOT / 'artifacts' / 'cyclelayer_v3' / 'theta_identifiability' / 'IDENTIFIABILITY_SUMMARY.md'",
    "if summary_md.exists():",
    "    from IPython.display import Markdown, display",
    "    display(Markdown(summary_md.read_text(encoding='utf-8')))",
    "else:",
    "    print('SUMMARY NOT FOUND')",
))


# =============================================================================
# RUL Model Sanity & Collapse Diagnostic phase (ADR-0014)
# =============================================================================

CELLS.append(md(
    "# RUL Model Sanity & Collapse Diagnostic (ADR-0014)",
    "",
    "Following the experiment matrix (A/B/C/D all RMSE ≈ 21.4, predictions clustered",
    "in a narrow band around the train mean), [ADR-0014](docs/decisions/ADR-0014-rul-collapse-diagnostic.md)",
    "requires a structured **read-only** diagnostic suite to decide whether the RUL",
    "head is collapsing to a mean, the implementation is broken, or the task is",
    "simply harder than current model capacity.",
    "",
    "Hard constraints (enforced by every script below):",
    "* No model architecture changes (`cyclelayer_v3.py` frozen)",
    "* No YAML / physical-constant changes",
    "* No hyperparameter tuning loops",
    "* No EPR / pressure loss reintroduction",
    "* No DS02 test leakage (`[11, 14, 15]` only evaluated)",
    "",
    "Outputs land under `artifacts/cyclelayer_v3/rul_model_sanity/<TIMESTAMP>/`.",
    "All six scripts share the same `<TIMESTAMP>` via the `RUL_SANITY_SESSION`",
    "env var set in the next cell.",
))

CELLS.append(code(
    "# Pin a single session id so every diagnostic writes to the same dir",
    "import os, time",
    "if not os.environ.get('RUL_SANITY_SESSION'):",
    "    os.environ['RUL_SANITY_SESSION'] = time.strftime('%Y%m%d_%H%M%S')",
    "SESSION = os.environ['RUL_SANITY_SESSION']",
    "SESSION_DIR = REPO_ROOT / 'artifacts' / 'cyclelayer_v3' / 'rul_model_sanity' / SESSION",
    "SESSION_DIR.mkdir(parents=True, exist_ok=True)",
    "print(f'RUL_SANITY_SESSION = {SESSION}')",
    "print(f'session dir = {SESSION_DIR}')",
))

CELLS.append(md(
    "## Step 4 — Target / window alignment audit",
    "",
    "First gate.  Verifies test/dev split disjoint, unit_id assignment, RUL range,",
    "per-unit monotonic decrease, window endpoint semantics, sample dump.",
    "**If this FAILS, all other diagnostics are invalid.**",
))
CELLS.append(code(
    "_run_diag('scripts/diagnose_v31b_target_alignment.py')",
))

CELLS.append(md(
    "## Step 2+3 — Collapse metrics, trivial baselines, plots",
    "",
    "Loads the production C checkpoint (auto-discovered).  Computes overall and",
    "per-RUL-region metrics, fits constant + per-unit linear baselines, produces",
    "the six diagnostic plots (scatter, residual, hist, calibration, per-unit",
    "trajectory, target distribution).",
))
CELLS.append(code(
    "_run_diag('scripts/diagnose_v31b_rul_collapse.py',",
    "          args=['--run_dir', str(C_RUN_DIR)])",
))

CELLS.append(md(
    "## Step 5 — Tiny-overfit smoke test",
    "",
    "Trains a fresh `CycleLayerV3` (random init, production architecture) on",
    "tiny subsets of train windows (256 / 1024 / 4096) for many epochs.",
    "If the model can't overfit 256 windows it's an implementation problem;",
    "if it can but collapses on full DS02 it's a generalisation problem.",
))
CELLS.append(code(
    "_run_diag('scripts/test_v31b_rul_overfit_tiny.py',",
    "          args=['--epochs', '200', '--batch_size', '64'])",
))

CELLS.append(md(
    "## Step 6 — Simple ML baselines (Ridge / HGB / RF)",
    "",
    "Classical feature-based regressors on DS02 train units → test units.",
    "Tests H7: if simple baselines crush V3.1b, the architecture/training",
    "is not competitive against classical ML.",
))
CELLS.append(code(
    "_run_diag('scripts/train_simple_rul_baselines_ds02.py',",
    "          args=['--models', 'ridge,hgb,rf',",
    "                '--max_train_windows', '60000',",
    "                '--max_test_windows', '30000'])",
))

CELLS.append(md(
    "## Step 7 — Branch-usage ablations",
    "",
    "Loads the production C checkpoint, at the prognostics-head input swaps",
    "out theta, aux, h_sens, z_ops (zero + shuffle).  A branch is **unused**",
    "if its ablation leaves RUL essentially unchanged.",
))
CELLS.append(code(
    "_run_diag('scripts/diagnose_v31b_branch_usage.py',",
    "          args=['--run_dir', str(C_RUN_DIR)])",
))

CELLS.append(md(
    "## Step 8 — Aggregate verdict",
    "",
    "Reads every diagnostic's `summary.json`, applies the ADR-0014 decision",
    "logic, writes `RUL_MODEL_SANITY_SUMMARY.md` + `.json`.  **Stop point.**",
))
CELLS.append(code(
    "_run_diag('scripts/build_rul_model_sanity_summary.py')",
    "",
    "summary_md = SESSION_DIR / 'RUL_MODEL_SANITY_SUMMARY.md'",
    "if summary_md.exists():",
    "    from IPython.display import Markdown, display",
    "    display(Markdown(summary_md.read_text(encoding='utf-8')))",
    "else:",
    "    print('SUMMARY NOT FOUND')",
))


CELLS.append(md("## Save artifacts to Drive"))

CELLS.append(code(
    "if DRIVE_RUNS_ROOT is not None:",
    "    DRIVE_RUNS_ROOT.mkdir(parents=True, exist_ok=True)",
    "    for run_name in ('A_baseline', 'B_physics_aux',",
    "                     'C_physics_theta_rul', 'D_shuffled_theta', 'comparison'):",
    "        src = RUNS_ROOT / f'{RUN_ID}_{run_name}'",
    "        if not src.exists(): continue",
    "        dst = DRIVE_RUNS_ROOT / f'{RUN_ID}_{run_name}'",
    "        if dst.exists(): shutil.rmtree(dst)",
    "        shutil.copytree(src, dst)",
    "        print(f'copied {src.name} -> {dst}')",
    "    # Also copy identifiability artifacts (read-only diagnostics)",
    "    ident_src = REPO_ROOT / 'artifacts' / 'cyclelayer_v3' / 'theta_identifiability'",
    "    if ident_src.exists():",
    "        ident_dst = DRIVE_RUNS_ROOT / f'{RUN_ID}_theta_identifiability'",
    "        if ident_dst.exists(): shutil.rmtree(ident_dst)",
    "        shutil.copytree(ident_src, ident_dst)",
    "        print(f'copied theta_identifiability/ -> {ident_dst}')",
    "    # RUL model sanity artifacts (ADR-0014)",
    "    sanity_src = REPO_ROOT / 'artifacts' / 'cyclelayer_v3' / 'rul_model_sanity'",
    "    if sanity_src.exists():",
    "        sanity_dst = DRIVE_RUNS_ROOT / f'{RUN_ID}_rul_model_sanity'",
    "        if sanity_dst.exists(): shutil.rmtree(sanity_dst)",
    "        shutil.copytree(sanity_src, sanity_dst)",
    "        print(f'copied rul_model_sanity/ -> {sanity_dst}')",
    "else:",
    "    print('DRIVE_RUNS_ROOT is None; skipping Drive copy.')",
))


# -----------------------------------------------------------------------------
# Build the .ipynb JSON
# -----------------------------------------------------------------------------

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
        "colab": {
            "provenance": [],
            "name": "colab_v3_thermal_aux.ipynb",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path(__file__).parent.parent / "notebooks" / "colab_v3_thermal_aux.ipynb"
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(NB, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB; {len(CELLS)} cells)")
