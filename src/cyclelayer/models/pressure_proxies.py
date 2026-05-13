"""Pressure proxies for total-vs-static comparison (V3.1b).

DS02 publishes Ps30 (static), BraytonEngine produces P30 (total).
The proxy converts between them using an assumed Mach number at
station 30 (HPC discharge).

Reference: V3.1b decision "No direct P30-vs-Ps30 comparison.
Use Ps30_proxy with configurable M30_proxy = 0.30."

The relation (isentropic, total-to-static):

    P_static = P_total / (1 + 0.5·(γ-1)·M²)^(γ/(γ-1))

with γ = γ_C (cold-side air) and M an *assumed* discharge Mach number.
This is a documented proxy, not a measurement.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cyclelayer.models.stations import GAMMA_C


def total_to_static_pressure(
    P_total: Tensor,
    M_assumed: float = 0.30,
) -> Tensor:
    """Convert total pressure to static pressure via isentropic relation.

    P_static = P_total / (1 + 0.5·(γ-1)·M²)^(γ/(γ-1))

    Args:
        P_total:   Total pressure tensor (any shape).
        M_assumed: Assumed flow Mach number (default 0.30 — typical HPC
                   discharge order of magnitude). Documented proxy
                   assumption, NOT a measurement.

    Returns:
        P_static: Tensor of same shape as P_total.
    """
    factor = (1.0 + 0.5 * (GAMMA_C - 1.0) * M_assumed ** 2) ** (
        GAMMA_C / (GAMMA_C - 1.0)
    )
    return P_total / factor


def Ps30_proxy(P30_total: Tensor, M30_proxy: float = 0.30) -> Tensor:
    """Compute Ps30 proxy from BraytonEngine P30 (total).

    Used when comparing the BraytonEngine prediction (total pressure at
    HPC discharge) against DS02 Ps30 measurements (static pressure).
    The conversion assumes a fixed flow Mach number at station 30.

    Args:
        P30_total: Total pressure at HPC discharge (Pa), from BraytonEngine.
        M30_proxy: Assumed flow Mach number at station 30 (default 0.30).

    Returns:
        Ps30: Static pressure proxy (Pa), comparable to DS02 Ps30 measurement.
    """
    return total_to_static_pressure(P30_total, M30_proxy)
