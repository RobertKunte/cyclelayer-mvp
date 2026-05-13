"""Smoke tests for the NCMAPSSV3Dataset adapter (V3.1b).

Uses a synthetic mini-HDF5 so the test is fast and self-contained.
A separate offline check verifies behavior on the real DS02 file
(skipped here when the file is not present).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from cyclelayer.data.ncmapss_v3 import NCMAPSSV3Dataset, load_userguide_fc02_anchor


# ---------------------------------------------------------------------------
# Synthetic mini-HDF5 fixture
# ---------------------------------------------------------------------------

def _write_synthetic_h5(path: Path, n_rows: int = 100, split: str = "dev") -> None:
    """Write a tiny DS02-shaped HDF5 file for testing."""
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as f:
        # W: alt(ft), Mach, TRA(%), T2(R)
        W = np.stack([
            rng.uniform(10000, 35000, size=n_rows),
            rng.uniform(0.4, 0.8,    size=n_rows),
            rng.uniform(50,  90,     size=n_rows),
            rng.uniform(420, 510,    size=n_rows),
        ], axis=1)
        f.create_dataset(f"W_{split}", data=W)

        # X_s: 14 columns (T24 R, T30 R, T48 R, T50 R, P15 psia, P2 psia,
        #                  P21, P24, Ps30, P40, P50, Nf rpm, Nc rpm, Wf pps)
        X = np.stack([
            rng.uniform(480, 610,   size=n_rows),    # T24
            rng.uniform(1080, 1450, size=n_rows),    # T30
            rng.uniform(1230, 1840, size=n_rows),    # T48
            rng.uniform(880, 1240,  size=n_rows),    # T50
            rng.uniform(6, 16,      size=n_rows),    # P15
            rng.uniform(4, 12,      size=n_rows),    # P2
            rng.uniform(6, 16,      size=n_rows),    # P21
            rng.uniform(7, 20,      size=n_rows),    # P24
            rng.uniform(80, 340,    size=n_rows),    # Ps30
            rng.uniform(82, 348,    size=n_rows),    # P40
            rng.uniform(4, 13,      size=n_rows),    # P50
            rng.uniform(1500, 2200, size=n_rows),    # Nf
            rng.uniform(7400, 8640, size=n_rows),    # Nc
            rng.uniform(0.7, 4.0,   size=n_rows),    # Wf
        ], axis=1)
        f.create_dataset(f"X_s_{split}", data=X)

        # T: 10 health modifiers (delta around 0)
        T = rng.uniform(-0.05, 0.0, size=(n_rows, 10))
        f.create_dataset(f"T_{split}", data=T)

        # A: unit_id, cycle, Fc, hs
        A = np.stack([
            rng.integers(1, 7,     size=n_rows).astype(np.float64),  # unit_id
            rng.integers(0, 100,   size=n_rows).astype(np.float64),  # cycle
            rng.integers(1, 5,     size=n_rows).astype(np.float64),  # Fc
            rng.integers(0, 2,     size=n_rows).astype(np.float64),  # hs
        ], axis=1)
        f.create_dataset(f"A_{split}", data=A)

        # Y: RUL labels
        Y = rng.integers(0, 100, size=(n_rows, 1)).astype(np.int64)
        f.create_dataset(f"Y_{split}", data=Y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dataset_loads_synthetic_h5():
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "synthetic.h5"
        _write_synthetic_h5(h5, n_rows=50)
        # Synthetic tests use in-memory explicitly to avoid Windows file-lock
        # issues with TemporaryDirectory cleanup; the lazy default is exercised
        # by test_streaming_mode_via_context_manager and the real-DS02 test.
        ds = NCMAPSSV3Dataset(h5, split="dev", load_in_memory=True)
        assert len(ds) == 50


def test_getitem_returns_expected_dict_structure():
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "synthetic.h5"
        _write_synthetic_h5(h5, n_rows=10)
        ds = NCMAPSSV3Dataset(h5, split="dev", load_in_memory=True)
        sample = ds[0]
        # Top-level keys
        assert set(sample.keys()) == {
            "ops_imp", "sens_imp", "targets_imp", "health_gt", "aux"
        }
        # ops_imp keys (incl. P2_psia from X_s)
        assert set(sample["ops_imp"].keys()) == {
            "alt_ft", "XM", "TRA_pct", "T2_R", "P2_psia"
        }
        # sens_imp keys
        assert set(sample["sens_imp"].keys()) == {"Nf_rpm", "Nc_rpm", "Wf_pps"}
        # targets_imp keys
        assert set(sample["targets_imp"].keys()) == {
            "T24_R", "T30_R", "T48_R", "T50_R",
            "P24_psia", "Ps30_psia", "P40_psia", "P50_psia",
        }
        # health_gt: 4 keys (HPT_eff_mod, HPT_flow_mod, LPT_eff_mod, LPT_flow_mod)
        assert set(sample["health_gt"].keys()) == {
            "HPT_eff_mod", "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod",
        }
        # aux keys
        assert set(sample["aux"].keys()) == {
            "unit_id", "cycle", "Fc", "hs", "RUL"
        }


def test_p2_extracted_from_x_s_not_w():
    """Critical: P2_psia must be sourced from X_s column 5, not from W."""
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "synthetic.h5"
        # Write a file with a known P2 value at row 0
        rng = np.random.default_rng(7)
        with h5py.File(h5, "w") as f:
            W = rng.uniform(10000, 35000, size=(5, 4))
            f.create_dataset("W_dev", data=W)
            X = np.zeros((5, 14), dtype=np.float64)
            X[:, 5] = 8.123   # P2 column
            X[:, 11] = 2000   # Nf column (avoid div-by-zero in any downstream)
            X[:, 12] = 8000
            X[:, 13] = 1.5
            # Set physical T/P columns to plausible values
            X[:, 0] = 500; X[:, 1] = 1300; X[:, 2] = 1700; X[:, 3] = 1100
            X[:, 4] = 10; X[:, 6] = 11; X[:, 7] = 14; X[:, 8] = 200
            X[:, 9] = 205; X[:, 10] = 8
            f.create_dataset("X_s_dev", data=X)
            f.create_dataset("T_dev",   data=np.zeros((5, 10)))
            f.create_dataset("A_dev",   data=np.ones((5, 4)))
            f.create_dataset("Y_dev",   data=np.zeros((5, 1), dtype=np.int64))
        ds = NCMAPSSV3Dataset(h5, split="dev", load_in_memory=True)
        sample = ds[0]
        assert torch.isclose(
            sample["ops_imp"]["P2_psia"],
            torch.tensor(8.123, dtype=torch.float32),
            rtol=1e-5,
        ), f"P2_psia should be 8.123 (from X_s[5]), got {sample['ops_imp']['P2_psia']}"


def test_unit_ids_property():
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "synthetic.h5"
        _write_synthetic_h5(h5, n_rows=200)
        ds = NCMAPSSV3Dataset(h5, split="dev", load_in_memory=True)
        ids = ds.unit_ids
        assert ids.dtype.kind in ("i", "u")
        # Synthetic data uses units 1..6
        assert set(ids.tolist()).issubset({1, 2, 3, 4, 5, 6})


def test_default_is_lazy_not_in_memory():
    """V3.1b clarification 3: lazy HDF5 access is the default."""
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "synthetic.h5"
        _write_synthetic_h5(h5, n_rows=20)
        # Use context manager to ensure file handle closes before TemporaryDirectory
        # tries to delete the file (Windows file-locking).
        with NCMAPSSV3Dataset(h5, split="dev") as ds:
            # Default constructor (no load_in_memory kwarg) → lazy mode
            assert ds.load_in_memory is False
            # Lazy mode keeps an h5py handle open
            assert hasattr(ds, "_h5")
            # __getitem__ must still work (reads one row from disk)
            sample = ds[0]
            assert "ops_imp" in sample
            assert "T2_R" in sample["ops_imp"]
            # RUL must be a finite scalar (verifies Y handling for (N,1) shape)
            assert torch.isfinite(sample["aux"]["RUL"]).all()


def test_streaming_mode_returns_same_values_as_in_memory():
    """Streaming and in-memory modes must produce identical row data."""
    with tempfile.TemporaryDirectory() as td:
        h5 = Path(td) / "synthetic.h5"
        _write_synthetic_h5(h5, n_rows=30)

        ds_mem = NCMAPSSV3Dataset(h5, split="dev", load_in_memory=True)
        with NCMAPSSV3Dataset(h5, split="dev", load_in_memory=False) as ds_lazy:
            for idx in (0, 5, 17, 29):
                a = ds_mem[idx]
                b = ds_lazy[idx]
                for group in ("ops_imp", "sens_imp", "targets_imp",
                              "health_gt", "aux"):
                    for k in a[group]:
                        assert torch.allclose(
                            a[group][k].float(), b[group][k].float(), rtol=1e-5
                        ), f"streaming mismatch at idx={idx} {group}.{k}"


def test_userguide_fc02_anchor_keys():
    """The FC02 anchor dict must expose all the constants used in C0/C1."""
    fc02 = load_userguide_fc02_anchor()
    expected = {
        "alt_ft", "XM", "TRA_pct", "Tsl_F",
        "Wf_pps", "Nf_rpm", "Nc_rpm",
        "EPR_ref", "T48_ref_R", "Net_Thrust_lbf",
    }
    assert expected == set(fc02.keys())
    # Spot-check the documented values
    assert fc02["XM"]      == 0.25
    assert fc02["TRA_pct"] == 100
    assert fc02["Nf_rpm"]  == 2403
    assert fc02["Nc_rpm"]  == 9084


# ---------------------------------------------------------------------------
# Optional: real DS02 file smoke test (skipped if not present)
# ---------------------------------------------------------------------------

_DS02_PATH = Path(__file__).parents[1] / "data" / "NCMAPSS" / "N-CMAPSS_DS02-006.h5"


@pytest.mark.skipif(not _DS02_PATH.exists(), reason="DS02 HDF5 not on disk")
def test_real_ds02_loads_and_first_sample_imperial_ranges():
    """If real DS02 is present: load it and check first-sample value ranges
    are in the expected Imperial ballparks."""
    ds = NCMAPSSV3Dataset(_DS02_PATH, split="dev")
    assert len(ds) > 100_000  # DS02 dev split is ~5.3M rows
    s = ds[0]
    # T2 in Rankine should be 200–600 (cruise inlet ≈ 470 R)
    assert 200 < s["ops_imp"]["T2_R"].item() < 600
    # P2 in psia should be 3–16 (cruise altitude → SLS)
    assert 3 < s["ops_imp"]["P2_psia"].item() < 16
    # Nf in rpm: ~ 1500–2200 for CMAPSS-90K-class
    assert 1000 < s["sens_imp"]["Nf_rpm"].item() < 2500
