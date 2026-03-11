"""Normalization and windowing utilities for N-CMAPSS data.

All functions are pure numpy/torch and operate on arrays already loaded
from the HDF5 file, so they can be applied before or after dataset creation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from cyclelayer.data.ncmapss import NCMAPSSDataset


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """Per-feature min-max normalization to [0, 1].

    Fit on training data, then apply to train/val/test.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Compute min/max from 2-D array (n_samples, n_features)."""
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.data_min_ is None:
            raise RuntimeError("Call fit() before transform().")
        lo, hi = self.feature_range
        scale = self.data_max_ - self.data_min_
        # Avoid division by zero for constant features
        scale = np.where(scale == 0, 1.0, scale)
        return lo + (X - self.data_min_) / scale * (hi - lo)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.data_min_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        lo, hi = self.feature_range
        scale = self.data_max_ - self.data_min_
        scale = np.where(scale == 0, 1.0, scale)
        return (X - lo) / (hi - lo) * scale + self.data_min_


class StandardScaler:
    """Zero-mean unit-variance standardization per feature."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        std = np.where(self.std_ == 0, 1.0, self.std_)
        return (X - self.mean_) / std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        std = np.where(self.std_ == 0, 1.0, self.std_)
        return X * std + self.mean_


def normalize(
    X: np.ndarray,
    method: str = "minmax",
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, MinMaxScaler | StandardScaler]:
    """Fit and apply normalization to a 2-D feature matrix.

    Args:
        X: Array of shape (n_samples, n_features).
        method: ``"minmax"`` or ``"standard"``.
        feature_range: Target range for min-max scaling (ignored for standard).

    Returns:
        Tuple of (normalized array, fitted scaler).
    """
    if method == "minmax":
        scaler: MinMaxScaler | StandardScaler = MinMaxScaler(feature_range)
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")
    return scaler.fit_transform(X), scaler


# ---------------------------------------------------------------------------
# Theta scaler helper
# ---------------------------------------------------------------------------

def fit_sensor_scaler(
    dataset: "NCMAPSSDataset | Any",
    train_units: list[int],
) -> StandardScaler:
    """Fit a StandardScaler on sensor inputs using only train-split rows.

    Prevents data leakage: val and test sensor rows are never seen during fit.

    Args:
        dataset: NCMAPSSDataset with _sensors (N, n_features) and
                 _unit_id_arr (N,) populated.
        train_units: Unit IDs whose rows are used for fitting.

    Returns:
        Fitted StandardScaler (not yet applied to the dataset).

    Example::

        scaler = fit_sensor_scaler(base_ds, unit_splits["train"])
        base_ds._sensors = scaler.transform(base_ds._sensors).astype(np.float32)
    """
    mask = np.isin(dataset._unit_id_arr, train_units)
    if mask.sum() == 0:
        raise ValueError(f"No rows found for train_units={train_units}.")
    return StandardScaler().fit(dataset._sensors[mask])


def fit_theta_scaler(
    dataset: "NCMAPSSDataset | Any",
    train_units: list[int],
) -> StandardScaler:
    """Fit a StandardScaler on theta_true using only train-split rows.

    Prevents data leakage: val and test theta rows are never seen during fit.

    Args:
        dataset: NCMAPSSDataset with _theta (N, n_health_params) and
                 _unit_id_arr (N,) populated.
        train_units: Unit IDs whose rows are used for fitting.

    Returns:
        Fitted StandardScaler (not yet applied to the dataset).

    Raises:
        ValueError: If the dataset has no theta data.

    Example::

        scaler = fit_theta_scaler(base_ds, unit_splits["train"])
        base_ds._theta = scaler.transform(base_ds._theta).astype(np.float32)
    """
    if dataset._theta is None:
        raise ValueError("Dataset has no theta data (T_dev key not found in HDF5).")
    mask = np.isin(dataset._unit_id_arr, train_units)
    if mask.sum() == 0:
        raise ValueError(f"No rows found for train_units={train_units}.")
    return StandardScaler().fit(dataset._theta[mask])


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def sliding_window(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping windows from a time series.

    Args:
        X: Feature matrix of shape (n_timesteps, n_features).
        y: Target vector of shape (n_timesteps,).
        window_size: Length of each window.
        stride: Step size between windows.

    Returns:
        windows: Array of shape (n_windows, window_size, n_features).
        targets: Array of shape (n_windows,) — RUL at the last step of each window.
    """
    n = len(X)
    indices = range(0, n - window_size + 1, stride)
    windows = np.stack([X[i : i + window_size] for i in indices])
    targets = np.array([y[i + window_size - 1] for i in indices])
    return windows, targets


def clip_rul(
    y: np.ndarray | Tensor,
    max_rul: float = 125.0,
) -> np.ndarray | Tensor:
    """Clip RUL targets to a maximum value (piece-wise linear health index).

    Values above ``max_rul`` are clipped to ``max_rul``, reflecting the
    assumption that the engine is healthy beyond that point.
    """
    if isinstance(y, Tensor):
        return y.clamp(max=max_rul)
    return np.clip(y, a_min=None, a_max=max_rul)
