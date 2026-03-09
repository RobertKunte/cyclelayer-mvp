# =============================================================================
# CycleLayer MVP — Google Colab Training Script
# N-CMAPSS turbofan RUL prediction (physics-informed Brayton cycle layer)
#
# Compatible with: Colab (T4/A100/V100), Python 3.11+, PyTorch >= 2.2
# Usage: paste cells into a Colab notebook, or run as a script.
# =============================================================================

# %% [markdown]
# ## Cell 0 — Quick-start checklist
# 1. Runtime > Change runtime type > **GPU** (T4 is fine)
# 2. Upload your HDF5 file to Google Drive, note its path
# 3. Edit the USER CONFIG block in Cell 2 (repo URL, HDF5 path, model type)
# 4. Run all cells in order

# --- Cell: 1 — GPU / Environment Check ---
# %%
import subprocess, sys, os, shutil, time, json, textwrap
from pathlib import Path
from datetime import datetime

def _sh(cmd, **kw):
    """Run shell command, stream output, raise on failure."""
    print(f"\n$ {cmd}")
    proc = subprocess.run(cmd, shell=True, check=True, **kw)
    return proc

# GPU check
_sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
import torch
print(f"\ntorch version : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device        : {torch.cuda.get_device_name(0)}")
    print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[WARN] No GPU detected — training will be slow on CPU.")


# --- Cell: 2 — USER CONFIG (edit before running) ---
# %%

# --------------------------------------------------------------------------
# Repo setup — choose ONE of the options below (A or B).
# Option A: Clone from GitHub (first run or if repo not yet mounted)
# Option B: Repo already available (mounted drive or previously cloned)
# --------------------------------------------------------------------------
REPO_URL   = "https://github.com/RobertKunte/cyclelayer-mvp.git"  # ← EDIT
REPO_CLONE = True          # Set False if repo already exists at REPO_ROOT

REPO_ROOT  = Path("/content/cyclelayer-mvp")   # where repo lives in Colab

# --------------------------------------------------------------------------
# Dataset path — HDF5 on Google Drive
# --------------------------------------------------------------------------
DRIVE_DATA_PATH = Path("/content/drive/MyDrive/cyclelayer-mvp/data/NCMAPSS/N-CMAPSS_DS01-005.h5")  # ← EDIT

# --------------------------------------------------------------------------
# Training settings
# --------------------------------------------------------------------------
MODEL_TYPE    = "cyclelayer_v1"   # "cyclelayer_v1" | "cnn" | "lstm" | "cnn_theta"
BASE_CONFIG   = "cyclelayer.yaml" # "cyclelayer.yaml" or "baseline.yaml"
STRIDE_TRAIN  = 5                 # 5 = dense (recommended); 10 = faster; 50 = quick test
STRIDE_EVAL   = 1                 # keep at 1 for correct PH computation
EPOCHS        = 50
PATIENCE      = 15
BATCH_SIZE    = 512               # reduce to 256 if OOM
NUM_WORKERS   = 4                 # Colab typically has 4 CPU cores
SEED          = 42

# --------------------------------------------------------------------------
# OOM guard — reduce batch size automatically on CUDA OOM
# --------------------------------------------------------------------------
OOM_FALLBACK_BATCH = 256          # used if BATCH_SIZE triggers OOM

# --------------------------------------------------------------------------
# Optional: LOUO cross-validation (runs 6 folds after main training)
# --------------------------------------------------------------------------
RUN_LOUO = False                  # set True to enable

# --------------------------------------------------------------------------
# Artifact destinations
# --------------------------------------------------------------------------
RUNS_ROOT        = Path("/content/runs")
DRIVE_RUNS_ROOT  = Path("/content/drive/MyDrive/cyclelayer_runs")  # ← EDIT (or None)

print("Config:")
for k, v in dict(MODEL_TYPE=MODEL_TYPE, BASE_CONFIG=BASE_CONFIG,
                 STRIDE_TRAIN=STRIDE_TRAIN, EPOCHS=EPOCHS,
                 BATCH_SIZE=BATCH_SIZE, SEED=SEED).items():
    print(f"  {k:20s} = {v}")


# --- Cell: 3 — Install Dependencies ---
# %%
_sh("pip install -q -U pip")
_sh(
    "pip install -q "
    "torch>=2.2 "
    "numpy>=1.26 "
    "scipy>=1.12 "
    "h5py>=3.10 "
    "matplotlib>=3.8 "
    "pandas>=2.2 "
    "tensorboard>=2.16 "
    "pyyaml>=6.0 "
    "tqdm>=4.66"
)
print("Dependencies installed.")


# --- Cell: 4 — Clone / Load Repo ---
# %%
if REPO_CLONE:
    if REPO_ROOT.exists():
        print(f"Repo already exists at {REPO_ROOT}. Pulling latest...")
        _sh(f"git -C {REPO_ROOT} pull --ff-only")
    else:
        _sh(f"git clone {REPO_URL} {REPO_ROOT}")
else:
    # Option B: repo already on mounted drive — just verify
    assert REPO_ROOT.exists(), f"REPO_ROOT not found: {REPO_ROOT}"
    print(f"Using existing repo at {REPO_ROOT}")

# Make sure repo is importable
for p in [str(REPO_ROOT / "src"), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Verify key entry points exist
for script in ["scripts/train.py", "scripts/evaluate.py", "scripts/run_louo.py"]:
    assert (REPO_ROOT / script).exists(), f"Missing: {script}"

print(f"\nGit commit: ", end="")
_sh(f"git -C {REPO_ROOT} rev-parse --short HEAD")


# --- Cell: 5 — Mount Google Drive & Verify HDF5 ---
# %%
from google.colab import drive  # noqa: E402  (Colab-specific)
drive.mount("/content/drive", force_remount=False)

assert DRIVE_DATA_PATH.exists(), (
    f"\nHDF5 file not found: {DRIVE_DATA_PATH}\n"
    "Ensure the file is uploaded to Google Drive at the path set in USER CONFIG."
)

file_size_gb = DRIVE_DATA_PATH.stat().st_size / 1e9
print(f"HDF5 found: {DRIVE_DATA_PATH}  ({file_size_gb:.2f} GB)")

# Quick sanity check — list top-level keys
import h5py
with h5py.File(DRIVE_DATA_PATH, "r") as f:
    keys = list(f.keys())
    print(f"HDF5 keys : {keys}")
    assert any("dev" in k for k in keys), (
        f"Expected keys ending in '_dev'. Found: {keys}"
    )
print("HDF5 structure OK.")


# --- Cell: 6 — Build Patched Config ---
# %%
import yaml

RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{MODEL_TYPE}"
RUN_DIR = RUNS_ROOT / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

RUNTIME_CFG_DIR = Path("/content/configs_runtime")
RUNTIME_CFG_DIR.mkdir(parents=True, exist_ok=True)

# Load base config
base_cfg_path = REPO_ROOT / "configs" / BASE_CONFIG
with open(base_cfg_path) as f:
    cfg = yaml.safe_load(f)

# --- Override data section ---
cfg.setdefault("data", {})
cfg["data"]["hdf5_path"]      = str(DRIVE_DATA_PATH)
cfg["data"]["stride_train"]   = STRIDE_TRAIN
cfg["data"]["stride_eval"]    = STRIDE_EVAL
cfg["data"]["batch_size"]     = BATCH_SIZE
cfg["data"]["num_workers"]    = NUM_WORKERS

# --- Override model section ---
cfg.setdefault("model", {})
cfg["model"]["type"] = MODEL_TYPE

# --- Override training section ---
cfg.setdefault("training", {})
cfg["training"]["epochs"]                  = EPOCHS
cfg["training"]["early_stopping_patience"] = PATIENCE
cfg["training"]["output_dir"]              = str(RUN_DIR)
cfg["training"]["use_amp"]                 = torch.cuda.is_available()

# --- Override evaluation section ---
cfg.setdefault("evaluation", {})
cfg["evaluation"]["checkpoint"] = str(RUN_DIR / "best.pt")

# --- Seed ---
cfg["seed"] = SEED

# Save patched config
patched_cfg_name = f"cyclelayer_{MODEL_TYPE}_stride{STRIDE_TRAIN}.yaml"
PATCHED_CFG = RUNTIME_CFG_DIR / patched_cfg_name
with open(PATCHED_CFG, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"Patched config: {PATCHED_CFG}")
print(f"Run dir       : {RUN_DIR}")
print("\nKey overrides:")
print(f"  hdf5_path    : {cfg['data']['hdf5_path']}")
print(f"  stride_train : {cfg['data']['stride_train']}")
print(f"  stride_eval  : {cfg['data']['stride_eval']}")
print(f"  batch_size   : {cfg['data']['batch_size']}")
print(f"  model.type   : {cfg['model']['type']}")
print(f"  epochs       : {cfg['training']['epochs']}")
print(f"  use_amp      : {cfg['training']['use_amp']}")
print(f"  output_dir   : {cfg['training']['output_dir']}")


# --- Cell: 7 — Training ---
# %%

def _stream_subprocess(cmd: list[str], logfile: Path, cwd: Path) -> None:
    """Run subprocess, print output live, and tee to logfile."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(cmd)}\nLog: {logfile}\n{'='*70}")
    t0 = time.time()

    with open(logfile, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd),
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Elapsed: {elapsed/60:.1f} min  |  Return code: {proc.returncode}")

    if proc.returncode != 0:
        # OOM fallback: retry with smaller batch
        if "CUDA out of memory" in open(logfile, encoding="utf-8").read():
            print(f"\n[OOM] Retrying with batch_size={OOM_FALLBACK_BATCH} ...")
            cfg["data"]["batch_size"] = OOM_FALLBACK_BATCH
            with open(PATCHED_CFG, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            _stream_subprocess(cmd, logfile.with_suffix(".retry.log"), cwd)
        else:
            raise RuntimeError(f"Command failed (code {proc.returncode}). See {logfile}")


LOG_DIR = RUN_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

train_cmd = [
    sys.executable, "scripts/train.py",
    "--config", str(PATCHED_CFG),
    "--model", MODEL_TYPE,
]

_stream_subprocess(train_cmd, LOG_DIR / "train.log", cwd=REPO_ROOT)

# Verify checkpoint produced
assert (RUN_DIR / "best.pt").exists(), (
    f"Training did not produce best.pt in {RUN_DIR}. Check {LOG_DIR}/train.log"
)
print(f"\nCheckpoint: {RUN_DIR / 'best.pt'} ({(RUN_DIR / 'best.pt').stat().st_size / 1e6:.1f} MB)")


# --- Cell: 8 — Evaluation (test split + dev split) ---
# %%

def _evaluate(split: str, logfile: Path) -> dict:
    """Run evaluate.py for one split, return loaded metrics dict."""
    metrics_path = RUN_DIR / f"metrics_{split}.json"
    eval_cmd = [
        sys.executable, "scripts/evaluate.py",
        "--config",     str(PATCHED_CFG),
        "--checkpoint", str(RUN_DIR / "best.pt"),
        "--split",      split,
        "--output",     str(metrics_path),
    ]
    _stream_subprocess(eval_cmd, logfile, cwd=REPO_ROOT)

    # predictions.csv is written next to metrics.json by evaluate.py
    # rename it to avoid collision between splits
    default_pred = RUN_DIR / "predictions.csv"
    split_pred   = RUN_DIR / f"predictions_{split}.csv"
    if default_pred.exists() and not split_pred.exists():
        default_pred.rename(split_pred)

    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


# Evaluate on held-out dev (val + test units)
metrics_dev = _evaluate("dev", LOG_DIR / "eval_dev.log")

# Evaluate on the HDF5 test split (fully independent rows)
metrics_test = _evaluate("test", LOG_DIR / "eval_test.log")

print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
for split_name, m in [("dev", metrics_dev), ("test", metrics_test)]:
    if m:
        print(f"\n  Split: {split_name}")
        print(f"    RMSE       : {m.get('rmse', 'n/a'):.4f}")
        print(f"    S-score    : {m.get('s_score', 'n/a'):.2f}")
        print(f"    PH median  : {m.get('ph_median', 'n/a')}")
        print(f"    PH p10/p90 : {m.get('ph_p10', 'n/a')} / {m.get('ph_p90', 'n/a')}")
        print(f"    PH none    : {m.get('ph_none_count', 'n/a')}")


# --- Cell: 9 — (Optional) LOUO Cross-Validation ---
# %%
# Toggle RUN_LOUO = True in the USER CONFIG cell to enable this.

if RUN_LOUO:
    LOUO_DIR = RUN_DIR / f"louo_{MODEL_TYPE}"
    cfg_louo  = yaml.safe_load(open(PATCHED_CFG))
    # Point output_dir to a sub-anchor so louo_dir lands inside RUN_DIR
    cfg_louo["training"]["output_dir"] = str(RUN_DIR / "_louo_anchor")
    LOUO_CFG = RUNTIME_CFG_DIR / f"louo_{patched_cfg_name}"
    with open(LOUO_CFG, "w") as f:
        yaml.dump(cfg_louo, f, default_flow_style=False, sort_keys=False)

    louo_cmd = [
        sys.executable, "scripts/run_louo.py",
        "--config", str(LOUO_CFG),
        "--model",  MODEL_TYPE,
    ]
    _stream_subprocess(louo_cmd, LOG_DIR / "louo.log", cwd=REPO_ROOT)

    results_json = LOUO_DIR / "results.json"
    if results_json.exists():
        louo_results = json.load(open(results_json))
        print("\nLOUO Results:")
        print(f"  RMSE mean : {louo_results['rmse_mean']:.4f} +/- {louo_results['rmse_std']:.4f}")
        print(f"  S-score   : {louo_results['s_score_mean']:.2f} +/- {louo_results['s_score_std']:.2f}")
        print(f"  PH median : {louo_results['ph_median']}")
        print(f"  PH none   : {louo_results['ph_none_count']}/{louo_results['n_folds']} folds")
else:
    print("LOUO skipped (RUN_LOUO=False). Set RUN_LOUO=True in Cell 2 to enable.")


# --- Cell: 10 — Artifact Summary ---
# %%

def _fmt_size(path: Path) -> str:
    if path.exists():
        return f"{path.stat().st_size / 1e6:.2f} MB"
    return "(missing)"

print(f"\nArtifacts in: {RUN_DIR}")
print(f"{'File':<40} {'Size':>10}")
print("-" * 52)
for fname in [
    "best.pt", "last.pt",
    "theta_scaler.npz",
    "metrics_dev.json", "metrics_test.json",
    "predictions_dev.csv", "predictions_test.csv",
    "logs/train.log", "logs/eval_dev.log", "logs/eval_test.log",
]:
    p = RUN_DIR / fname
    status = _fmt_size(p) if p.exists() else "(absent)"
    print(f"  {fname:<38} {status:>10}")

# List TensorBoard event files
tb_dir = RUN_DIR / "tb"
if tb_dir.exists():
    tb_files = list(tb_dir.glob("events.out.*"))
    print(f"  tb/events.out.*  ({len(tb_files)} file(s))")

# Config and meta
print(f"\n  Config used  : {PATCHED_CFG}")


# --- Cell: 11 — Package & Copy to Google Drive ---
# %%
if DRIVE_RUNS_ROOT is not None:
    DRIVE_RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    # Create zip archive of the run directory
    zip_base = str(DRIVE_RUNS_ROOT / RUN_ID)
    print(f"Zipping {RUN_DIR} -> {zip_base}.zip ...")
    zip_path = shutil.make_archive(
        zip_base,
        format="zip",
        root_dir=str(RUN_DIR.parent),
        base_dir=str(RUN_DIR.name),
    )
    zip_size_mb = Path(zip_path).stat().st_size / 1e6
    print(f"Created: {zip_path}  ({zip_size_mb:.1f} MB)")

    # Also copy patched config alongside the zip for reference
    shutil.copy2(PATCHED_CFG, DRIVE_RUNS_ROOT / PATCHED_CFG.name)
    print(f"Config copy: {DRIVE_RUNS_ROOT / PATCHED_CFG.name}")
else:
    print("DRIVE_RUNS_ROOT is None — skipping Drive upload.")
    print("To save artifacts manually:")
    print(f"  from google.colab import files")
    print(f"  files.download('{RUN_DIR}/best.pt')")


# --- Cell: 12 — Optional: View TensorBoard ---
# %%
# Uncomment and run to launch TensorBoard in Colab:

# %load_ext tensorboard
# %tensorboard --logdir /content/runs

# Or equivalently:
# _sh(f"tensorboard --logdir {RUNS_ROOT} --port 6006 &")
# from google.colab.output import eval_js
# print(eval_js("google.colab.kernel.proxyPort(6006)"))


# --- Cell: 13 — Troubleshooting Notes ---
# %%
NOTES = textwrap.dedent("""
    TROUBLESHOOTING
    ===============

    1. OOM (CUDA out of memory)
       - The script automatically retries with batch_size={fallback} on OOM.
       - Manually: set BATCH_SIZE = 256 or 128 in Cell 2.
       - Last resort: set cfg["data"]["window_size"] = 20 (shorter windows).

    2. HDF5 file not found
       - Verify DRIVE_DATA_PATH in Cell 2 matches the exact path in Google Drive.
       - Google Drive path is case-sensitive on Colab.

    3. 'No module named scripts'
       - evaluate.py and run_louo.py both add REPO_ROOT to sys.path.
       - If this still fails: sys.path.insert(0, str(REPO_ROOT)) at the top of the
         failing script, then re-run.

    4. PH = None for all units
       - This is expected at high stride (stride=50+) or early epochs.
       - With stride=5 and >=50 epochs on GPU the model typically achieves PH
         for at least some units.
       - PH criterion: the model must stay within alpha=20% of true RUL
         continuously until end of life. Mid-trajectory spikes break it.

    5. Slow training
       - Check nvidia-smi to confirm GPU is used.
       - Confirm use_amp=True (automatic mixed precision) is active.
       - Reduce NUM_WORKERS if Colab shows DataLoader worker errors.

    6. LOUO very slow
       - LOUO runs 6 full training loops. Use --units to pick specific folds:
         run_louo.py ... --units 1 3  (runs only folds for test units 1 and 3)

    7. Reproducing a run
       - The frozen config is stored in the run dir and zipped to Drive.
       - Load it: yaml.safe_load(open("config_frozen.yaml"))
       - Theta scaler: np.load("theta_scaler.npz")  -> mean, std arrays
""").format(fallback=OOM_FALLBACK_BATCH)

print(NOTES)
