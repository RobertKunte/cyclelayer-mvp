"""N-CMAPSS Dataset Loader (HDF5).

N-CMAPSS (New Commercial Modular Aero-Propulsion System Simulation) is a
NASA dataset for turbofan engine degradation simulation.

HDF5 structure (flat keys, no nested groups):
    W_{split}    - flight conditions    (N, 4)   float64  [alt, Mach, TRA, T2]
    X_s_{split}  - measured sensors     (N, 14)  float64
    X_v_{split}  - virtual sensors      (N, 14)  float64
    T_{split}    - health parameters    (N, 10)  float64  <- theta_true
    Y_{split}    - RUL targets          (N, 1)   int64
    A_{split}    - auxiliary            (N, 4)   float64  [unit, cycle, Fc, hs]

Health parameters (T_var): fan_eff_mod, fan_flow_mod, LPC_eff_mod,
    LPC_flow_mod, HPC_eff_mod, HPC_flow_mod, HPT_eff_mod, HPT_flow_mod,
    LPT_eff_mod, LPT_flow_mod

Reference: Arias Chao et al., "Aircraft Engine Run-to-Failure Dataset under
    Real Flight Conditions for Prognostics and Diagnostics", Data 2021.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SENSOR_NAMES_XS: list[str] = [
    "T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
    "Ps30", "P40", "P50", "Nf", "Nc", "Wf",
]
SENSOR_NAMES_XV: list[str] = [
    "T40", "P30", "P45", "W21", "W22", "W25", "W31", "W32",
    "W48", "W50", "SmFan", "SmLPC", "SmHPC", "phi",
]
OPERATING_CONDITION_NAMES: list[str] = ["alt", "Mach", "TRA", "T2"]
HEALTH_PARAM_NAMES: list[str] = [
    "fan_eff_mod", "fan_flow_mod",
    "LPC_eff_mod", "LPC_flow_mod",
    "HPC_eff_mod", "HPC_flow_mod",
    "HPT_eff_mod", "HPT_flow_mod",
    "LPT_eff_mod", "LPT_flow_mod",
]
N_HEALTH_PARAMS: int = len(HEALTH_PARAM_NAMES)  # 10


class NCMAPSSDataset(Dataset):
    """Index-based PyTorch Dataset for N-CMAPSS (no pre-built window list).

    Raw sensor arrays are loaded once into memory. Each call to
    __getitem__ extracts a window on-the-fly using a stored
    (unit_id, local_window_start) index list. Windows never cross unit
    boundaries.

    Args:
        hdf5_path: Path to the N-CMAPSS .h5 file.
        split: "dev" (training) or "test" (evaluation).
        window_size: Number of consecutive time steps per sample.
        stride: Step size between successive windows (default 1).
        use_virtual_sensors: Concatenate X_v after X_s.
        units: Subset of integer unit IDs (None = all).
        return_theta_true: If True, __getitem__ returns a 3-tuple
            (x, rul, theta_true) instead of (x, rul).
        dtype: Floating-point dtype for returned tensors.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        split: str = "dev",
        window_size: int = 30,
        stride: int = 1,
        use_virtual_sensors: bool = False,
        units: list[int] | None = None,
        return_theta_true: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.use_virtual_sensors = use_virtual_sensors
        self.return_theta_true = return_theta_true
        self.dtype = dtype

        if split not in ("dev", "test"):
            raise ValueError(f"split must be 'dev' or 'test', got '{split!r}'")

        self._sensors: np.ndarray = np.empty(0)
        self._rul: np.ndarray = np.empty(0)
        self._theta: np.ndarray | None = None
        self._unit_id_arr: np.ndarray = np.empty(0)
        self._unit_ranges: dict[int, tuple[int, int]] = {}
        self._index_list: list[tuple[int, int]] = []

        self._load(units)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, units: list[int] | None) -> None:
        sfx = f"_{self.split}"
        with h5py.File(self.hdf5_path, "r") as f:
            W   = f[f"W{sfx}"][:].astype(np.float32)
            X_s = f[f"X_s{sfx}"][:].astype(np.float32)
            rul = f[f"Y{sfx}"][:].ravel().astype(np.float32)
            A   = f[f"A{sfx}"][:].astype(np.float32)

            if self.use_virtual_sensors:
                X_v = f[f"X_v{sfx}"][:].astype(np.float32)
                sensors = np.concatenate([W, X_s, X_v], axis=1)
            else:
                sensors = np.concatenate([W, X_s], axis=1)

            theta_key = f"T{sfx}"
            theta: np.ndarray | None = (
                f[theta_key][:].astype(np.float32) if theta_key in f else None
            )

        unit_ids_raw = A[:, 0].astype(np.int32)

        # Sort by (unit_id, cycle) -> each unit occupies a contiguous block
        order = np.lexsort((A[:, 1], unit_ids_raw))
        sensors      = sensors[order]
        rul          = rul[order]
        unit_ids_raw = unit_ids_raw[order]
        if theta is not None:
            theta = theta[order]

        self._sensors     = sensors
        self._rul         = rul
        self._theta       = theta
        self._unit_id_arr = unit_ids_raw

        unique_ids, counts = np.unique(unit_ids_raw, return_counts=True)
        cumsum = 0
        for uid, cnt in zip(unique_ids.tolist(), counts.tolist()):
            self._unit_ranges[uid] = (cumsum, cumsum + cnt)
            cumsum += cnt

        selected: list[int] = (
            sorted(self._unit_ranges.keys()) if units is None
            else [u for u in units if u in self._unit_ranges]
        )

        index_list: list[tuple[int, int]] = []
        for uid in selected:
            g_start, g_end = self._unit_ranges[uid]
            n_steps = g_end - g_start
            for w_start in range(0, n_steps - self.window_size + 1, self.stride):
                index_list.append((uid, w_start))
        self._index_list = index_list

        if self.return_theta_true and self._theta is None:
            raise ValueError(
                f"return_theta_true=True but key 'T_{self.split}' not found in {self.hdf5_path}."
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Return (x, rul) or (x, rul, theta_true) for sample ``idx``.

        x shape: (window_size, n_features)
        theta_true shape (when present): (n_health_params,)
        """
        uid, w_start = self._index_list[idx]
        g_start = self._unit_ranges[uid][0] + w_start
        g_end   = g_start + self.window_size

        x   = torch.from_numpy(self._sensors[g_start:g_end].copy())
        rul = torch.tensor(float(self._rul[g_end - 1]), dtype=self.dtype)

        if self.return_theta_true:
            theta_t = torch.from_numpy(self._theta[g_end - 1].copy())
            return x, rul, theta_t

        return x, rul

    # ------------------------------------------------------------------
    # Unit-trajectory helpers
    # ------------------------------------------------------------------

    @property
    def unit_ids_array(self) -> np.ndarray:
        """Unit ID aligned with each __getitem__ index, shape (len(self),)."""
        return np.array([uid for uid, _ in self._index_list], dtype=np.int32)

    def available_units(self) -> list[int]:
        """Sorted list of unit IDs in this dataset."""
        seen: set[int] = set()
        result: list[int] = []
        for uid, _ in self._index_list:
            if uid not in seen:
                seen.add(uid)
                result.append(uid)
        return sorted(result)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        return int(self._sensors.shape[1]) if self._sensors.ndim == 2 else 0

    @property
    def n_health_params(self) -> int:
        return int(self._theta.shape[1]) if self._theta is not None else 0

    @staticmethod
    def feature_names(use_virtual_sensors: bool = False) -> list[str]:
        names = OPERATING_CONDITION_NAMES + SENSOR_NAMES_XS
        if use_virtual_sensors:
            names = names + SENSOR_NAMES_XV
        return names


# ---------------------------------------------------------------------------
# SubsetByUnit
# ---------------------------------------------------------------------------

class SubsetByUnit(Dataset):
    """Zero-copy view of NCMAPSSDataset restricted to given unit IDs.

    No data is copied; backed by the parent dataset raw arrays.
    Enables unit-level train/val splitting without duplicating data.

    Args:
        base: The parent NCMAPSSDataset.
        unit_ids: Units to include in this subset.
    """

    def __init__(self, base: NCMAPSSDataset, unit_ids: list[int]) -> None:
        super().__init__()
        self._base = base
        uid_set = set(unit_ids)
        self._indices: list[int] = [
            i for i, (uid, _) in enumerate(base._index_list) if uid in uid_set
        ]
        self._unit_ids = sorted(uid_set & set(base.available_units()))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return self._base[self._indices[idx]]

    @property
    def unit_ids_array(self) -> np.ndarray:
        """Unit ID for each sample in this subset."""
        return self._base.unit_ids_array[self._indices]

    def available_units(self) -> list[int]:
        return self._unit_ids

    @property
    def n_features(self) -> int:
        return self._base.n_features

    @property
    def n_health_params(self) -> int:
        return self._base.n_health_params
