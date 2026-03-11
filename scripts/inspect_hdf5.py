"""Print a concise description of an N-CMAPSS HDF5 file.

Outputs:
  - All dataset keys with shape, dtype, min, max, mean
  - Per-split unit IDs (from A_{split}[:,0])
  - Number of rows per split
  - Quick guide to the N-CMAPSS key naming convention

Usage
-----
    python scripts/inspect_hdf5.py data/NCMAPSS/N-CMAPSS_DS01-005.h5
    python scripts/inspect_hdf5.py data/NCMAPSS/N-CMAPSS_DS01-005.h5 --stats
    python scripts/inspect_hdf5.py data/NCMAPSS/N-CMAPSS_DS01-005.h5 --units

Design
------
Keeps memory footprint small: stats are computed by reading one column at a
time with h5py chunked access, never loading the full 2D array into RAM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError as e:
    print(f"ERROR: {e}\n  Install with: pip install h5py numpy", file=sys.stderr)
    sys.exit(1)


# N-CMAPSS dataset descriptions for the known key families
_KEY_DESCRIPTIONS: dict[str, str] = {
    "W":   "Flight conditions  (alt, Mach, TRA, T2)  [N x 4]",
    "X_s": "Measured sensors   (14 channels)          [N x 14]",
    "X_v": "Virtual sensors    (14 channels)          [N x 14]",
    "T":   "Health parameters  (10 mod factors)       [N x 10]",
    "Y":   "RUL labels         (0..max_rul)           [N x 1]",
    "A":   "Auxiliary info     (unit, cycle, Fc, hs)  [N x 4]",
}


def _col_stats(ds: "h5py.Dataset") -> dict[str, float]:
    """Compute min/max/mean of a 2-D or 1-D dataset without loading all columns."""
    n_rows = ds.shape[0]
    n_cols = ds.shape[1] if ds.ndim == 2 else 1

    col_mins, col_maxs, col_means = [], [], []
    for c in range(n_cols):
        if ds.ndim == 2:
            col = ds[:, c]
        else:
            col = ds[:]
        col_mins.append(float(np.min(col)))
        col_maxs.append(float(np.max(col)))
        col_means.append(float(np.mean(col)))

    return {
        "min":  min(col_mins),
        "max":  max(col_maxs),
        "mean": float(np.mean(col_means)),
        "n_rows": n_rows,
    }


def _unit_ids_from_A(ds: "h5py.Dataset") -> list[int]:
    """Read unit column (col 0) from A_{split} and return sorted unique IDs."""
    units = ds[:, 0].astype(int)
    return sorted(set(units.tolist()))


def _split_label(key: str) -> str:
    """Return 'dev' or 'test' from a key like 'X_s_dev'."""
    if key.endswith("_dev"):
        return "dev"
    if key.endswith("_test"):
        return "test"
    return "?"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect N-CMAPSS HDF5 structure: shapes, dtypes, stats, units."
    )
    parser.add_argument("hdf5_path", help="Path to the HDF5 file")
    parser.add_argument("--stats",  action="store_true",
                        help="Print per-dataset min/max/mean (slower for large files)")
    parser.add_argument("--units",  action="store_true",
                        help="Print unique unit IDs per split from A_{split}")
    args = parser.parse_args()

    path = Path(args.hdf5_path)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f" N-CMAPSS HDF5 Inspector")
    print(f" File : {path.resolve()}")
    print(f" Size : {path.stat().st_size / 1e9:.3f} GB")
    print(f"{'='*70}\n")

    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
        print(f"Top-level datasets ({len(keys)} total):\n")

        # Group keys by split
        dev_keys  = [k for k in keys if k.endswith("_dev")]
        test_keys = [k for k in keys if k.endswith("_test")]
        other_keys = [k for k in keys if not (k.endswith("_dev") or k.endswith("_test"))]

        for split_label, split_keys in [("dev", dev_keys), ("test", test_keys),
                                         ("other", other_keys)]:
            if not split_keys:
                continue
            print(f"  --- split: {split_label} ---")
            for key in split_keys:
                ds = f[key]
                family = key.rsplit("_dev", 1)[0].rsplit("_test", 1)[0]
                desc   = _KEY_DESCRIPTIONS.get(family, "")
                shape_str = "x".join(str(d) for d in ds.shape)
                print(f"  {key:<18}  shape=({shape_str:<12})  dtype={ds.dtype!s:<10}  {desc}")

                if args.stats:
                    try:
                        st = _col_stats(ds)
                        print(f"  {'':18}  min={st['min']:10.4f}  "
                              f"max={st['max']:10.4f}  mean={st['mean']:10.4f}")
                    except Exception as e:
                        print(f"  {'':18}  [stats error: {e}]")

            # Unit IDs from A_{split} if requested
            if args.units:
                a_key = f"A_{split_label}"
                if a_key in f:
                    uid_list = _unit_ids_from_A(f[a_key])
                    # Count windows per unit
                    unit_col = f[a_key][:, 0].astype(int)
                    counts   = {u: int((unit_col == u).sum()) for u in uid_list}
                    print(f"\n  Units in {split_label}:")
                    for u in uid_list:
                        print(f"    unit {u:2d}  :  {counts[u]:>8,} rows")
                    print()
            print()

        # Top-level summary
        if "A_dev" in f:
            n_dev  = f["A_dev"].shape[0]
            print(f"  Total rows (dev)  : {n_dev:,}")
        if "A_test" in f:
            n_test = f["A_test"].shape[0]
            print(f"  Total rows (test) : {n_test:,}")
        print()

    # N-CMAPSS naming guide
    print(f"{'='*70}")
    print(" N-CMAPSS key naming convention")
    print(f"{'='*70}")
    print("""
  <family>_<split>   where family is one of:
    W       Flight conditions   — [alt (ft), Mach, TRA (%), T2 (K)]
    X_s     Measured sensors    — 14 sensor channels (temperatures, pressures, …)
    X_v     Virtual sensors     — 14 derived channels
    T       Health parameters   — 10 multiplicative efficiency/flow modifiers
    Y       RUL labels          — remaining useful life in cycles (int)
    A       Auxiliary           — [unit_id, flight_cycle, flight_class (Fc), health_state (hs)]

  Splits: '_dev' is the training+validation set; '_test' is the hold-out set.
  Both use the same units but different operating points / flight cycles.

  Health parameters in T (column order, 1-indexed):
    1  fan_eff_mod     2  fan_flow_mod
    3  LPC_eff_mod     4  LPC_flow_mod
    5  HPC_eff_mod     6  HPC_flow_mod
    7  HPT_eff_mod     8  HPT_flow_mod
    9  LPT_eff_mod    10  LPT_flow_mod
""")


if __name__ == "__main__":
    main()
