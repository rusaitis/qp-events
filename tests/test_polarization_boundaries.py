r"""Boundary tests for polarization classification (Eq. 6-7, Fig 10).

Existing tests cover the canonical 90° and 0°/180° cases; this file
hardens the classifier against near-threshold phase shifts and noisy
inputs.

The tolerance is $\pm 30°$ around each canonical phase
(:data:`CIRCULAR_LINEAR_TOL_DEG = 30.0`):

  - 0°  / 180° / 360°   → "linear"
  - 90° / 270°          → "circular"
  - otherwise           → "mixed"
"""

from __future__ import annotations

import numpy as np
import pytest

from qp.signal.cross_correlation import (
    classify_polarization,
    phase_shift,
)
from qp.signal.polarization_config import CIRCULAR_LINEAR_TOL_DEG


@pytest.mark.parametrize(
    "phase_deg, expected",
    [
        (90.0, "circular"),
        (270.0, "circular"),
        (89.9, "circular"),
        (270.1, "circular"),
        (0.0, "linear"),
        (180.0, "linear"),
        (360.0, "linear"),
        (45.0, "mixed"),  # in the gap between linear (0±30) and circular (90±30)
        (135.0, "mixed"),  # in the gap between circular (90±30) and linear (180±30)
        (225.0, "mixed"),  # in the gap between linear (180±30) and circular (270±30)
    ],
)
def test_classification_grid(phase_deg, expected):
    assert classify_polarization(phase_deg) == expected


@pytest.mark.parametrize(
    "phase_deg",
    [
        90.0 - CIRCULAR_LINEAR_TOL_DEG + 1.0,
        90.0 + CIRCULAR_LINEAR_TOL_DEG - 1.0,
        270.0 - CIRCULAR_LINEAR_TOL_DEG + 1.0,
    ],
)
def test_near_circular_boundary_classifies_circular(phase_deg):
    assert classify_polarization(phase_deg) == "circular"


def test_phase_modulo_wraparound():
    """phase_deg arguments outside [0, 360) must wrap before classification."""
    assert classify_polarization(450.0) == "circular"  # 450 mod 360 = 90
    assert classify_polarization(-90.0) == "circular"  # -90 mod 360 = 270


class TestPhaseShiftFromSignals:
    @pytest.fixture
    def t_6h(self) -> tuple[np.ndarray, float]:
        dt = 60.0
        n = int(6 * 3600 / dt)
        return np.arange(n) * dt, dt

    def test_perfect_circular_90_phase(self, t_6h):
        t, dt = t_6h
        period = 3600.0
        f = np.cos(2 * np.pi * t / period)
        g = np.sin(2 * np.pi * t / period)  # 90° leading
        _, phase = phase_shift(f, g, dt=dt, period=period)
        assert classify_polarization(phase) == "circular"

    def test_perfect_linear_in_phase(self, t_6h):
        t, dt = t_6h
        period = 3600.0
        f = np.cos(2 * np.pi * t / period)
        g = np.cos(2 * np.pi * t / period)
        _, phase = phase_shift(f, g, dt=dt, period=period)
        assert classify_polarization(phase) == "linear"

    def test_perfect_linear_anti_phase(self, t_6h):
        t, dt = t_6h
        period = 3600.0
        f = np.cos(2 * np.pi * t / period)
        g = -np.cos(2 * np.pi * t / period)
        _, phase = phase_shift(f, g, dt=dt, period=period)
        assert classify_polarization(phase) == "linear"

    def test_noisy_circular_still_classifies_circular(self, t_6h):
        """20% Gaussian noise on top of a circular pair should not flip the
        classification — the cross-correlation peak is robust to broadband noise."""
        t, dt = t_6h
        period = 3600.0
        rng = np.random.default_rng(13)
        amp = 1.0
        sigma = 0.2 * amp
        f = amp * np.cos(2 * np.pi * t / period) + rng.normal(0.0, sigma, t.size)
        g = amp * np.sin(2 * np.pi * t / period) + rng.normal(0.0, sigma, t.size)
        _, phase = phase_shift(f, g, dt=dt, period=period)
        assert classify_polarization(phase) == "circular"


def test_tolerance_widening_swallows_mixed():
    """Doubling the tolerance should reclassify a 45° phase shift (currently
    'mixed') as 'circular' since 45 then falls within 90 ± 60."""
    assert classify_polarization(45.0) == "mixed"
    assert (
        classify_polarization(45.0, tolerance=2 * CIRCULAR_LINEAR_TOL_DEG) == "circular"
    )
