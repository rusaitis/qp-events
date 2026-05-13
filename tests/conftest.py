"""Shared pytest fixtures for the qp test suite.

Fixtures here are intended for reuse across multiple test files. Single-use
fixtures stay local to the test file that owns them.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sine_60min() -> tuple[np.ndarray, float]:
    """A pure 60-minute sinusoid sampled at 1-min cadence for 36 h.

    Returns (signal, dt_seconds). Useful as a known-period reference input
    for FFT/wavelet/ridge tests across the signal subpackage.
    """
    dt = 60.0
    n = int(36 * 3600 / dt)
    t = np.arange(n) * dt
    period = 60 * 60
    return 2.0 * np.sin(2 * np.pi * t / period), dt


@pytest.fixture
def white_noise_36h() -> tuple[np.ndarray, float]:
    """Zero-mean unit-variance white noise of the same length as sine_60min."""
    rng = np.random.default_rng(42)
    dt = 60.0
    n = int(36 * 3600 / dt)
    return rng.normal(0.0, 1.0, n), dt


@pytest.fixture(scope="session")
def saturn_field():
    """Default SaturnField instance for KMAG-dependent tests.

    Session-scoped because constructing the field warms numba JIT (~2 s);
    tests should treat the field as read-only.
    """
    from qp.fieldline.kmag_model import SaturnField

    return SaturnField()
