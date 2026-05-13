"""DS02 dataset adapter for CycleLayer V3 (V3.1b).

Loads N-CMAPSS DS02 in its native Imperial units and returns explicitly-
named dicts.  No unit conversion happens here — `units.to_si()` is the
explicit downstream conversion layer (Hard Rule: every unit conversion
goes through units.py).

Column conventions (DS02, confirmed by HDF5 inspection):

    W_{split}    [N, 4]   alt (ft), Mach (-), TRA (%), T2 (°R)
    X_s_{split}  [N, 14]  T24, T30, T48, T50,                   (all °R)
                          P15, P2, P21, P24, Ps30, P40, P50,    (all psia)
                          Nf, Nc,                               (rpm)
                          Wf                                    (pps)
    T_{split}    [N, 10]  fan_eff_mod, fan_flow_mod,
                          LPC_eff_mod, LPC_flow_mod,
                          HPC_eff_mod, HPC_flow_mod,
                          HPT_eff_mod, HPT_flow_mod,
                          LPT_eff_mod, LPT_flow_mod
    A_{split}    [N, 4]   unit_id, cycle, Fc, hs
    Y_{split}    [N, 1]   RUL

NOTE on T2: the HDF5 inspector annotates T2 with "(K)" but DS02 values
(421–510) are clearly Rankine — at altitude T2 ≈ 234–283 K is plausible,
T2 ≈ 234–283 °R = 130–157 K is not.  We treat T2 as °R.

NOTE on station naming: spec calls HPT outlet "T45"; dataset uses "T48".
The adapter exposes both; downstream code maps T48 → T45 as a proxy.

NOTE on P30: dataset has Ps30 (static), not P30 (total).  The adapter
returns Ps30; downstream code uses pressure_proxies.Ps30_proxy(P30_total)
to compare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Column indices (DS02, fixed by NASA convention — verified by inspection)
# ---------------------------------------------------------------------------

# W_{split} columns (4)
_W_ALT_FT   = 0
_W_MACH     = 1
_W_TRA_PCT  = 2
_W_T2_R     = 3

# X_s_{split} columns (14)
_X_T24      = 0
_X_T30      = 1
_X_T48      = 2
_X_T50      = 3
_X_P15      = 4
_X_P2       = 5
_X_P21      = 6
_X_P24      = 7
_X_PS30     = 8
_X_P40      = 9
_X_P50      = 10
_X_NF       = 11
_X_NC       = 12
_X_WF       = 13

# T_{split} columns (10) — only the three GT we use are indexed here
_T_HPT_EFF  = 6
_T_HPT_FLOW = 7
_T_LPT_EFF  = 8
_T_LPT_FLOW = 9

# A_{split} columns (4)
_A_UNIT     = 0
_A_CYCLE    = 1
_A_FC       = 2
_A_HS       = 3


class NCMAPSSV3Dataset(Dataset):
    """DS02 adapter for CycleLayer V3.

    Returns per-row dicts with Imperial-named values:

        ops_imp     {alt_ft, XM, TRA_pct, T2_R, P2_psia}
        sens_imp    {Nf_rpm, Nc_rpm, Wf_pps}
        targets_imp {T24_R, T30_R, T48_R, T50_R,
                     P24_psia, Ps30_psia, P40_psia, P50_psia}
        health_gt   {HPT_eff_mod, HPT_flow_mod,
                     LPT_eff_mod, LPT_flow_mod}
        aux         {unit_id, cycle, RUL, Fc, hs}

    No unit conversion is performed.  Apply `units.to_si()` downstream.

    Window-based access (B, T, F) is NOT yet implemented — this adapter
    returns per-row tensors.  Phase E will wrap or extend this for window
    indexing.

    Args:
        hdf5_path: Path to the DS02 HDF5 file.
        split:     "dev" or "test".  Default "dev".
        load_in_memory: Default **False** — lazy HDF5 access (one row read
            from disk per `__getitem__`). On DS02 the dev split is 5.26 M
            rows × ~0.6 GB and a full preload would dominate startup
            time / RAM unnecessarily.  Set True only for small fixtures
            or when dataloader workers benefit from in-memory tensors.

            In lazy mode the dataset keeps an open `h5py.File`; call
            `.close()` (or use the context manager) before deleting the
            file on disk (Windows file-locking).
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        split: str = "dev",
        load_in_memory: bool = False,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        if split not in ("dev", "test"):
            raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
        self.split = split
        self.load_in_memory = load_in_memory

        with h5py.File(self.hdf5_path, "r") as f:
            keys = {f"W_{split}", f"X_s_{split}", f"T_{split}",
                    f"A_{split}", f"Y_{split}"}
            missing = keys - set(f.keys())
            if missing:
                raise KeyError(
                    f"Missing required HDF5 datasets in {self.hdf5_path}: "
                    f"{missing}"
                )
            if load_in_memory:
                self._W = f[f"W_{split}"][:].astype(np.float32)
                self._X = f[f"X_s_{split}"][:].astype(np.float32)
                self._T = f[f"T_{split}"][:].astype(np.float32)
                self._A = f[f"A_{split}"][:].astype(np.float32)
                self._Y = f[f"Y_{split}"][:].astype(np.int64).reshape(-1)
            else:
                # Streaming mode: keep the file open and index lazily.
                # Caller must keep this object alive while indexing.
                self._h5 = h5py.File(self.hdf5_path, "r")
                self._W = self._h5[f"W_{split}"]
                self._X = self._h5[f"X_s_{split}"]
                self._T = self._h5[f"T_{split}"]
                self._A = self._h5[f"A_{split}"]
                self._Y = self._h5[f"Y_{split}"]

        # Sanity checks on shape
        n = self._W.shape[0]
        assert self._X.shape == (n, 14), f"X_s shape mismatch: {self._X.shape}"
        assert self._W.shape == (n, 4),  f"W shape mismatch: {self._W.shape}"
        assert self._T.shape == (n, 10), f"T shape mismatch: {self._T.shape}"
        assert self._A.shape == (n, 4),  f"A shape mismatch: {self._A.shape}"
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        W = np.asarray(self._W[idx])            # (4,)
        X = np.asarray(self._X[idx])            # (14,)
        T = np.asarray(self._T[idx])            # (10,)
        A = np.asarray(self._A[idx])            # (4,)
        # In-memory mode: self._Y was reshaped to (N,) → scalar at idx.
        # Streaming mode: self._Y is the original (N, 1) Dataset → 1-element array.
        # .item() handles both safely.
        Y = float(np.asarray(self._Y[idx]).item())

        # ops_imp — flight conditions + measured P2 (extracted from X_s[5])
        ops_imp = {
            "alt_ft":  torch.tensor(W[_W_ALT_FT],   dtype=torch.float32),
            "XM":      torch.tensor(W[_W_MACH],     dtype=torch.float32),
            "TRA_pct": torch.tensor(W[_W_TRA_PCT],  dtype=torch.float32),
            "T2_R":    torch.tensor(W[_W_T2_R],     dtype=torch.float32),
            # P2 lives in X_s (col 5); needed by Inlet (P1: use_measured_inlet)
            "P2_psia": torch.tensor(X[_X_P2],       dtype=torch.float32),
        }

        # sens_imp — shaft speeds + fuel flow
        sens_imp = {
            "Nf_rpm": torch.tensor(X[_X_NF], dtype=torch.float32),
            "Nc_rpm": torch.tensor(X[_X_NC], dtype=torch.float32),
            "Wf_pps": torch.tensor(X[_X_WF], dtype=torch.float32),
        }

        # targets_imp — measured sensors used as L_sens targets
        targets_imp = {
            "T24_R":     torch.tensor(X[_X_T24],  dtype=torch.float32),
            "T30_R":     torch.tensor(X[_X_T30],  dtype=torch.float32),
            "T48_R":     torch.tensor(X[_X_T48],  dtype=torch.float32),
            "T50_R":     torch.tensor(X[_X_T50],  dtype=torch.float32),
            "P24_psia":  torch.tensor(X[_X_P24],  dtype=torch.float32),
            "Ps30_psia": torch.tensor(X[_X_PS30], dtype=torch.float32),
            "P40_psia":  torch.tensor(X[_X_P40],  dtype=torch.float32),
            "P50_psia":  torch.tensor(X[_X_P50],  dtype=torch.float32),
        }

        # health_gt — N-CMAPSS T_dev modifiers (delta around 0)
        health_gt = {
            "HPT_eff_mod":  torch.tensor(T[_T_HPT_EFF],  dtype=torch.float32),
            "HPT_flow_mod": torch.tensor(T[_T_HPT_FLOW], dtype=torch.float32),
            "LPT_eff_mod":  torch.tensor(T[_T_LPT_EFF],  dtype=torch.float32),
            "LPT_flow_mod": torch.tensor(T[_T_LPT_FLOW], dtype=torch.float32),
        }

        # aux — bookkeeping
        aux = {
            "unit_id": torch.tensor(int(A[_A_UNIT]),  dtype=torch.long),
            "cycle":   torch.tensor(int(A[_A_CYCLE]), dtype=torch.long),
            "Fc":      torch.tensor(int(A[_A_FC]),    dtype=torch.long),
            "hs":      torch.tensor(int(A[_A_HS]),    dtype=torch.long),
            "RUL":     torch.tensor(Y,                dtype=torch.float32),
        }

        return {
            "ops_imp":     ops_imp,
            "sens_imp":    sens_imp,
            "targets_imp": targets_imp,
            "health_gt":   health_gt,
            "aux":         aux,
        }

    @property
    def unit_ids(self) -> np.ndarray:
        """Unique unit IDs in this split (for unit-level splitting)."""
        if self.load_in_memory:
            return np.unique(self._A[:, _A_UNIT].astype(np.int64))
        return np.unique(self._A[:, _A_UNIT][:].astype(np.int64))

    def close(self) -> None:
        """Close the HDF5 handle if streaming."""
        if not self.load_in_memory and hasattr(self, "_h5"):
            self._h5.close()

    def __enter__(self) -> "NCMAPSSV3Dataset":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup; safe even if close() already ran.
        try:
            self.close()
        except Exception:
            pass


def load_userguide_fc02_anchor() -> dict[str, Any]:
    """Return UserGuide Tab. 1.4 FC02 reference values.

    The single design anchor for V3.1b. Values are from
    NASA/TM-2007-215026 Tab. 1.4 (CMAPSS reference engine, 90K-class
    2-spool turbofan).  Used in Phase C0 and C1 only.

    No unit conversion — Imperial values straight from the table.
    """
    return {
        "alt_ft":           0,
        "XM":               0.25,
        "TRA_pct":          100,
        "Tsl_F":            59,
        "Wf_pps":           7.085,
        "Nf_rpm":           2403,
        "Nc_rpm":           9084,
        "EPR_ref":          1.261,    # P50/P2 per CMAPSS convention
        "T48_ref_R":        2083,     # used as PROXY for T45
        "Net_Thrust_lbf":   66755,    # NOT validated in V3.1b
    }
