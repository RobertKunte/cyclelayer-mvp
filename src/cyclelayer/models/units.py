"""Imperial ↔ SI conversion for N-CMAPSS DS02 (V3.1b).

N-CMAPSS publishes sensor and ops values in Imperial units (°R, psia, ft, pps,
rpm).  BraytonEngine works internally in SI.  This module is the single
hard-gated conversion layer.

Reference: docs/CycleLayer_V3.1a_Master_Spec.md (Mai 2026, Rev 3.1a) § B.0.

Conversion factors are exact where possible; numeric constants follow NIST.
Stage 0 of the validation suite tests the roundtrip identity to < 1e-6.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Conversion factors
# ---------------------------------------------------------------------------

FT_TO_M:    float = 0.3048              # exact
RANK_TO_K:  float = 5.0 / 9.0           # exact
PSIA_TO_PA: float = 6894.757293168361   # 1 psi = 6894.757... Pa (NIST)
PPS_TO_KGS: float = 0.45359237          # exact: pounds-mass per second to kg/s
RPM_TO_RAD: float = 2.0 * math.pi / 60.0


# ---------------------------------------------------------------------------
# ops + sensors → SI dict
# ---------------------------------------------------------------------------

def to_si(
    ops_imp:  Mapping[str, Tensor],
    sens_imp: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Convert N-CMAPSS imperial inputs to SI for BraytonEngine.

    Args:
        ops_imp:
            Required keys (each tensor):
              - alt_ft   altitude in feet
              - XM       Mach number (dimensionless)
              - TRA_pct  throttle resolver angle in percent
              - T2_R     inlet total temperature in Rankine
              - P2_psia  inlet total pressure in psia
        sens_imp:
            Required keys (each tensor):
              - Nf_rpm   fan/LP-spool speed in rpm
              - Nc_rpm   core/HP-spool speed in rpm
              - Wf_pps   fuel mass flow in pounds-mass per second

    Returns:
        Dict of SI-converted scalars/tensors. Keys are explicit about units
        so downstream code cannot accidentally mix systems.
    """
    return {
        "alt_m":  ops_imp["alt_ft"] * FT_TO_M,
        "mach":   ops_imp["XM"],                        # already dimensionless
        "TRA":    ops_imp["TRA_pct"] / 100.0,           # to fraction
        "T2_K":   ops_imp["T2_R"] * RANK_TO_K,
        "P2_Pa":  ops_imp["P2_psia"] * PSIA_TO_PA,
        # Speeds remain in rpm — the corrected-speed formula in stations.py
        # uses rpm directly. Conversion to rad/s only if a torque calculation
        # needs it (not in V3.1b MVP).
        "Nf_rpm": sens_imp["Nf_rpm"],
        "Nc_rpm": sens_imp["Nc_rpm"],
        "Wf_kgs": sens_imp["Wf_pps"] * PPS_TO_KGS,
    }


# ---------------------------------------------------------------------------
# BraytonEngine SI sensors → imperial dict for sensor-loss
# ---------------------------------------------------------------------------

def to_imperial(sensors_si: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Convert BraytonEngine SI outputs back to imperial for sensor-loss.

    BraytonEngine returns a dict with keys T24_K, T30_K, P30_Pa, T50_K
    (the four output sensors specified in V3.1a § A.2).
    """
    return {
        "T24_R":    sensors_si["T24_K"]  / RANK_TO_K,
        "T30_R":    sensors_si["T30_K"]  / RANK_TO_K,
        "P30_psia": sensors_si["P30_Pa"] / PSIA_TO_PA,
        "T50_R":    sensors_si["T50_K"]  / RANK_TO_K,
    }


# ---------------------------------------------------------------------------
# Scalar helpers (for tests, plotting, debug — not for hot path)
# ---------------------------------------------------------------------------

def rankine_to_kelvin(value: float | Tensor) -> float | Tensor:
    return value * RANK_TO_K


def kelvin_to_rankine(value: float | Tensor) -> float | Tensor:
    return value / RANK_TO_K


def psia_to_pa(value: float | Tensor) -> float | Tensor:
    return value * PSIA_TO_PA


def pa_to_psia(value: float | Tensor) -> float | Tensor:
    return value / PSIA_TO_PA


def ft_to_m(value: float | Tensor) -> float | Tensor:
    return value * FT_TO_M


def m_to_ft(value: float | Tensor) -> float | Tensor:
    return value / FT_TO_M


def pps_to_kgs(value: float | Tensor) -> float | Tensor:
    return value * PPS_TO_KGS


def kgs_to_pps(value: float | Tensor) -> float | Tensor:
    return value / PPS_TO_KGS
