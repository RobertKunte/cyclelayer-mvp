"""Evaluation metrics for turbofan RUL prognostics.

Implements the three standard metrics used in the PHM community:

RMSE          -- Root Mean Squared Error
s-score       -- Asymmetric scoring function from the PHM'08 challenge
                 (larger penalty for late predictions / over-estimation)
Prediction    -- First time step where |y_hat - y| / y < threshold
Horizon (alpha)

Extended diagnostics (added for interpretability):

s_score_samples    -- per-sample s-score values (before summing), used for
                      per-unit mean aggregation and summary statistics.
ph_debug_stats     -- per-unit PH diagnostics: frac_within_alpha_lastK,
                      max_abs_error, p95_abs_error.

S-score scale warning
---------------------
The standard s_score() is a SUM over all samples.  With N-CMAPSS at
stride_eval=1 you get ~4M windows across all units.  A single engine with
only small errors (mean |d|=3) still contributes ~300k score points.  This
makes s_score_sum sensitive to dataset size and stride.

Always report BOTH:
  - s_score_sum  : the standard summed score (comparable to PHM'08 papers)
  - s_score_mean : sum / N_samples  (comparable across experiments with
                   different strides and splits)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def rmse(
    pred: np.ndarray | Tensor,
    target: np.ndarray | Tensor,
) -> float:
    """Root Mean Squared Error.

    Args:
        pred:   Predicted RUL values, shape (N,).
        target: True RUL values, shape (N,).

    Returns:
        Scalar RMSE value.
    """
    pred, target = _to_numpy(pred), _to_numpy(target)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


# ---------------------------------------------------------------------------
# s-score (PHM'08 / NASA scoring function)
# ---------------------------------------------------------------------------

def s_score(
    pred: np.ndarray | Tensor,
    target: np.ndarray | Tensor,
) -> float:
    """Asymmetric NASA scoring function (summed over all samples).

    Computes:  S = sum(exp(d/a_i) - 1)
    where d = y_hat - y and a_i = 13 if d < 0 (early), 10 if d >= 0 (late).

    Late predictions (over-estimation of RUL) are penalised more heavily.

    WARNING: This is a SUM.  With millions of windows the absolute value
    is dominated by dataset size.  Use s_score_mean for cross-experiment
    comparisons with different strides.

    Reference: Saxena et al., "Metrics for Offline Evaluation of
        Prognostic Performance", IJPHM 2010.
    """
    return float(np.sum(s_score_samples(pred, target)))


def s_score_samples(
    pred: np.ndarray | Tensor,
    target: np.ndarray | Tensor,
) -> np.ndarray:
    """Per-sample NASA s-score values (before summing).

    Returns an array of shape (N,) where each element is
    exp(d/a) - 1 for that prediction.  Sum to get the standard s_score;
    take the mean for a stride-invariant metric; group by unit to get
    per-unit statistics.

    Args:
        pred:   Predicted RUL values, shape (N,).
        target: True RUL values, shape (N,).
    """
    pred, target = _to_numpy(pred), _to_numpy(target)
    d = pred - target
    a = np.where(d < 0, 13.0, 10.0)
    return np.exp(d / a) - 1.0


# ---------------------------------------------------------------------------
# Prediction Horizon (alpha-lambda accuracy)
# ---------------------------------------------------------------------------

def prediction_horizon(
    pred_trajectory: np.ndarray | Tensor,
    target_trajectory: np.ndarray | Tensor,
    alpha: float = 0.2,
) -> int | None:
    """Number of time steps before end-of-life where |y_hat-y|/y <= alpha.

    Finds the *first* time step (closest to failure) from which the model's
    predictions stay within alpha-percent of the true RUL continuously until
    end-of-life.

    PH = None when:
      1. Mid-trajectory spikes: the model is accurate near EOL but briefly
         exceeds the alpha band at some earlier point, breaking continuity.
      2. Systematic bias: the model consistently over- or under-estimates
         throughout the trajectory.
      3. Collapse: the model predicts a constant value (e.g., mean RUL).

    Use ph_debug_stats() to diagnose which case applies.

    Args:
        pred_trajectory:   Predicted RUL for one unit over time, shape (T,).
        target_trajectory: True RUL for the same unit, shape (T,).
        alpha:             Relative error threshold (default 0.20 = 20 %).

    Returns:
        The prediction horizon in time steps from failure (int), or
        None if the condition is never satisfied.
    """
    pred   = _to_numpy(pred_trajectory)
    target = _to_numpy(target_trajectory)

    relative_err = np.abs(pred - target) / np.maximum(target, 1e-6)
    within = relative_err <= alpha  # boolean mask

    # Find the first index from which `within` holds until end
    # i.e., the earliest t such that within[t:] is all True
    for t in range(len(within)):
        if within[t:].all():
            rul_at_t = int(round(target[t]))
            return rul_at_t

    return None


def ph_debug_stats(
    pred_trajectory: np.ndarray | Tensor,
    target_trajectory: np.ndarray | Tensor,
    alpha: float = 0.2,
    ks: tuple[int, ...] = (50, 100, 200),
) -> dict[str, float]:
    """Diagnostic statistics for a single unit's prediction trajectory.

    Complements prediction_horizon() by explaining *why* PH is None.

    frac_within_alpha_lastK
        Fraction of the last K windows where |y_hat - y| / y <= alpha.
        - High (>0.8) + PH=None  => mid-trajectory spike problem.
          The model is accurate near EOL but has a burst error at some
          earlier point, breaking the continuity requirement.
        - Low (<0.4) + PH=None   => systematic bias or collapse.

    max_abs_error / p95_abs_error
        Absolute scale of errors.  If RMSE is acceptable but PH=None,
        these help pinpoint whether there are a few catastrophic outliers.

    Args:
        pred_trajectory:   Predicted RUL for one unit, shape (T,).
        target_trajectory: True RUL for the same unit, shape (T,).
        alpha:             Same threshold used for PH computation.
        ks:                Window counts for frac_within_alpha_lastK.

    Returns:
        Dict with keys:
          frac_within_alpha_last{k}  for each k in ks
          max_abs_error
          p95_abs_error
          n_windows
    """
    pred   = _to_numpy(pred_trajectory)
    target = _to_numpy(target_trajectory)
    abs_err      = np.abs(pred - target)
    relative_err = abs_err / np.maximum(target, 1e-6)
    within       = relative_err <= alpha
    T            = len(pred)

    stats: dict[str, float] = {
        "max_abs_error": float(np.max(abs_err)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "n_windows":     float(T),
    }
    for k in ks:
        # Use all available windows if the trajectory is shorter than k
        last_k = within[-k:] if T >= k else within
        stats[f"frac_within_alpha_last{k}"] = float(np.mean(last_k))

    return stats


# ---------------------------------------------------------------------------
# Aggregate evaluation helper
# ---------------------------------------------------------------------------

def evaluate_all(
    pred: np.ndarray | Tensor,
    target: np.ndarray | Tensor,
    alpha: float = 0.2,
) -> dict[str, float | None]:
    """Compute RMSE and s-score for a set of predictions.

    Args:
        pred:   Predicted RUL values (N,).
        target: True RUL values (N,).
        alpha:  Threshold for prediction horizon (not computed here;
                requires per-unit trajectories).

    Returns:
        Dict with keys "rmse" and "s_score".
    """
    return {
        "rmse": rmse(pred, target),
        "s_score": s_score(pred, target),
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_numpy(x: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
