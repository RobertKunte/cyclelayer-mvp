"""Tests for pressure_proxies.py (V3.1b — total-to-static conversion).

Reference: docs/CycleLayer_V3.1b_Master_Spec.md decision
"No direct P30-vs-Ps30 comparison. Use Ps30_proxy with configurable
M30_proxy = 0.30."
"""

from __future__ import annotations

import math

import torch

from cyclelayer.models.pressure_proxies import (
    Ps30_proxy,
    total_to_static_pressure,
)
from cyclelayer.models.stations import GAMMA_C


def test_proxy_at_zero_mach_equals_total():
    """At M=0 the dynamic head is zero → static = total."""
    P_total = torch.tensor([200_000.0, 100_000.0, 50_000.0])
    P_static = total_to_static_pressure(P_total, M_assumed=0.0)
    assert torch.allclose(P_static, P_total, rtol=1e-9), (
        f"M=0 must give Ps == P_total; got {P_static} vs {P_total}"
    )


def test_proxy_at_M030_typical_factor():
    """At M=0.30 the conversion factor is ≈ 0.939 (γ=1.4 air).

    Ps/P_total = 1 / (1 + 0.5·(γ-1)·M²)^(γ/(γ-1))
              = 1 / (1 + 0.2·0.09)^(3.5)
              = 1 / (1.018)^3.5
              ≈ 0.93947
    """
    expected = 1.0 / (1.0 + 0.5 * (GAMMA_C - 1.0) * 0.30 ** 2) ** (
        GAMMA_C / (GAMMA_C - 1.0)
    )
    # Sanity-check the known value
    assert math.isclose(expected, 0.93947, abs_tol=1e-3)

    P_total = torch.tensor([1_000_000.0])
    P_static = Ps30_proxy(P_total, M30_proxy=0.30)
    assert math.isclose(
        (P_static / P_total).item(), expected, rel_tol=1e-6,
    ), f"factor mismatch: {(P_static / P_total).item()} vs expected {expected}"


def test_proxy_monotonic_in_mach():
    """Higher assumed Mach → lower Ps_proxy (more dynamic head subtracted)."""
    P_total = torch.tensor([1_000_000.0])
    machs = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    statics = [total_to_static_pressure(P_total, M_assumed=m).item() for m in machs]
    diffs = [statics[i + 1] - statics[i] for i in range(len(statics) - 1)]
    assert all(d < 0 for d in diffs), (
        f"Ps_proxy must decrease monotonically in M; got {statics}"
    )
