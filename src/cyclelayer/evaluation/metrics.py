"""Evaluation metrics for turbofan RUL prognostics.

Implements the three standard metrics used in the PHM community:

RMSE          – Root Mean Squared Error
s-score       – Asymmetric scoring function from the PHM'08 challenge
                (larger penalty for late predictions / over-estimation)
Prediction    – First time step where |ŷ - y| / y < threshold
Horizon (α)
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
    """Asymmetric NASA scoring function.

    Computes:  S = Σ (exp(d/a_i) - 1)
    where d = ŷ - y and a_i = 13 if d < 0 (early), 10 if d >= 0 (late).

    Late predictions (over-estimation of RUL) are penalised more heavily.

    Reference: Saxena et al., "Metrics for Offline Evaluation of
        Prognostic Performance", IJPHM 2010.
    """
    pred, target = _to_numpy(pred), _to_numpy(target)
    d = pred - target
    a = np.where(d < 0, 13.0, 10.0)
    return float(np.sum(np.exp(d / a) - 1.0))


# ---------------------------------------------------------------------------
# Prediction Horizon (α-λ accuracy)
# ---------------------------------------------------------------------------

def prediction_horizon(
    pred_trajectory: np.ndarray | Tensor,
    target_trajectory: np.ndarray | Tensor,
    alpha: float = 0.2,
) -> int | None:
    """Number of time steps before end-of-life where |ŷ-y|/y ≤ α.

    Finds the *first* time step (closest to failure) from which the model's
    predictions stay within α-percent of the true RUL.

    Args:
        pred_trajectory:   Predicted RUL for one unit over time, shape (T,).
        target_trajectory: True RUL for the same unit, shape (T,).
        alpha:             Relative error threshold (default 0.20 = 20 %).

    Returns:
        The prediction horizon in time steps from failure (int), or
        ``None`` if the condition is never satisfied.
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
        Dict with keys ``"rmse"`` and ``"s_score"``.
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
