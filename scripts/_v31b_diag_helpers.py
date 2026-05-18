"""Shared helpers for V3.1b RUL collapse diagnostic scripts (ADR-0014).

Lightweight module — no heavy dependencies, no side effects on import.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

# Keep sys.path additions local (avoid polluting global state on import)
_THIS = Path(__file__).parent
if str(_THIS.parent / "src") not in sys.path:
    sys.path.insert(0, str(_THIS.parent / "src"))
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))


REPO_ROOT = _THIS.parent
SANITY_BASE = REPO_ROOT / "artifacts" / "cyclelayer_v3" / "rul_model_sanity"


def get_session_dir(reuse_within_seconds: int = 3600) -> Path:
    """Return a shared session dir under `rul_model_sanity/<YYYYMMDD_HHMMSS>/`.

    Behaviour:
    1. If env var `RUL_SANITY_SESSION` is set, use `SANITY_BASE/<that>`.
    2. Else reuse the most-recently-modified subdir of SANITY_BASE if it
       was touched within the last `reuse_within_seconds` (default 1 h)
       — so sequential script invocations from one Colab session share
       a dir.
    3. Else create a fresh timestamped dir.
    """
    SANITY_BASE.mkdir(parents=True, exist_ok=True)
    env = os.environ.get("RUL_SANITY_SESSION")
    if env:
        sd = SANITY_BASE / env
        sd.mkdir(exist_ok=True)
        return sd
    candidates = [d for d in SANITY_BASE.iterdir() if d.is_dir()]
    if candidates:
        latest = max(candidates, key=lambda d: d.stat().st_mtime)
        if (time.time() - latest.stat().st_mtime) < reuse_within_seconds:
            return latest
    sd = SANITY_BASE / datetime.now().strftime("%Y%m%d_%H%M%S")
    sd.mkdir(exist_ok=True)
    return sd


def find_c_run_dir(args_dir: str | None) -> Path | None:
    """Locate a V3.1b C run dir (containing best.pt).

    Search order:
      1. --run_dir CLI arg if provided
      2. /content/runs_v3_thermal_aux/<*_C_physics_theta_rul>/   (Colab)
      3. runs_v3_thermal_aux/<*_C_physics_theta_rul>/           (local)
      4. artifacts/cyclelayer_v3/thermal_aux_smoke/             (local smoke)
    """
    if args_dir:
        p = Path(args_dir)
        return p if (p / "best.pt").exists() else None
    candidates = [
        Path("/content/runs_v3_thermal_aux"),
        REPO_ROOT / "runs_v3_thermal_aux",
        REPO_ROOT / "artifacts" / "cyclelayer_v3" / "thermal_aux_smoke",
    ]
    for c in candidates:
        if not c.exists():
            continue
        # Latest C run with best.pt
        subs = sorted(
            [s for s in c.glob("*C_physics_theta_rul*") if (s / "best.pt").exists()],
            key=lambda d: d.stat().st_mtime,
        )
        if subs:
            return subs[-1]
        if (c / "best.pt").exists():
            return c
    return None


# ── Markdown helpers (avoid pandas.to_markdown's tabulate dependency) ──

def df_to_md(df, floatfmt: str = ".4f") -> str:
    import pandas as pd  # local import; helpers stay light
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt))
            elif isinstance(v, bool):
                cells.append("YES" if v else "NO")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── Metric helpers ────────────────────────────────────────────────────

def rul_metrics(pred, true) -> dict:
    """Standard regression metrics for RUL with cycle units."""
    import numpy as np
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    err = pred - true
    mse = float((err ** 2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(np.abs(err).mean())
    bias = float(err.mean())
    if np.std(pred) > 1e-12 and np.std(true) > 1e-12:
        pearson = float(np.corrcoef(pred, true)[0, 1])
        # Slope of linear regression pred = a + b*true
        slope = float(np.cov(true, pred)[0, 1] / np.var(true))
        # R² as 1 − SSE/SST (can go below 0 for poor predictors)
        ss_res = float(((true - pred) ** 2).sum())
        ss_tot = float(((true - true.mean()) ** 2).sum())
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        pearson = float("nan"); slope = float("nan"); r2 = float("nan")
    try:
        from scipy.stats import spearmanr
        if np.std(pred) > 1e-12 and np.std(true) > 1e-12:
            spear = float(spearmanr(pred, true).statistic)
        else:
            spear = float("nan")
    except Exception:
        spear = float("nan")
    std_true = float(np.std(true))
    std_pred = float(np.std(pred))
    std_ratio = std_pred / std_true if std_true > 0 else float("nan")
    p_qs = np.quantile(pred, [0.05, 0.5, 0.95]) if len(pred) else (float("nan"),) * 3
    t_qs = np.quantile(true, [0.05, 0.5, 0.95]) if len(true) else (float("nan"),) * 3
    return {
        "n":          int(len(pred)),
        "RMSE":       rmse,
        "MAE":        mae,
        "bias":       bias,
        "R2":         r2,
        "Pearson":    pearson,
        "Spearman":   spear,
        "slope":      slope,
        "std_true":   std_true,
        "std_pred":   std_pred,
        "std_ratio":  std_ratio,
        "pred_min":   float(pred.min()) if len(pred) else float("nan"),
        "pred_max":   float(pred.max()) if len(pred) else float("nan"),
        "pred_p05":   float(p_qs[0]),
        "pred_p50":   float(p_qs[1]),
        "pred_p95":   float(p_qs[2]),
        "true_min":   float(true.min()) if len(true) else float("nan"),
        "true_max":   float(true.max()) if len(true) else float("nan"),
        "true_p05":   float(t_qs[0]),
        "true_p50":   float(t_qs[1]),
        "true_p95":   float(t_qs[2]),
    }


def metrics_by_region(pred, true, regions: list[tuple[str, float, float]]) -> "pd.DataFrame":
    import numpy as np, pandas as pd
    pred = np.asarray(pred, dtype=float); true = np.asarray(true, dtype=float)
    rows = []
    for name, lo, hi in regions:
        mask = (true >= lo) & (true < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"region": name, "n": 0,
                         "RMSE": float("nan"), "MAE": float("nan"),
                         "bias": float("nan"), "mean_pred": float("nan"),
                         "mean_true": float("nan")})
            continue
        e = pred[mask] - true[mask]
        rows.append({
            "region": name,
            "n": n,
            "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAE":  float(np.abs(e).mean()),
            "bias": float(e.mean()),
            "mean_pred": float(pred[mask].mean()),
            "mean_true": float(true[mask].mean()),
        })
    return pd.DataFrame(rows)


COLLAPSE_THRESHOLDS = dict(
    r2_max_for_collapse       = 0.0,
    std_ratio_max_for_collapse = 0.3,
    slope_max_for_collapse     = 0.3,
    EOL_bias_max_for_collapse  = 10.0,    # over-estimation in true_RUL < 20
)


def flag_collapse(m: dict, eol_bias: float) -> dict[str, bool]:
    t = COLLAPSE_THRESHOLDS
    return {
        "R2_le_0":               m["R2"] is not None and m["R2"] <= t["r2_max_for_collapse"],
        "std_ratio_lt_0.3":      m["std_ratio"] < t["std_ratio_max_for_collapse"],
        "abs_slope_lt_0.3":      abs(m["slope"]) < t["slope_max_for_collapse"],
        "EOL_bias_gt_+10":       eol_bias > t["EOL_bias_max_for_collapse"],
    }


# ── Region defaults shared across scripts ─────────────────────────────

RUL_REGIONS = [
    ("RUL<10",         0.0,  10.0),
    ("RUL<20",         0.0,  20.0),
    ("RUL<30",         0.0,  30.0),
    ("30<=RUL<60",    30.0,  60.0),
    ("RUL>=60",       60.0, 1e9),
]
