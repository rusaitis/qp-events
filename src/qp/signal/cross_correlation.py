"""Cross-correlation phase shift (Equations 6-7).

    tau_delay = argmax_t ((f * g)(t))       (Eq 6)
    (f * g)(tau) = integral f(t) g(t+tau) dt  (Eq 7)

Used in paper Figure 10 to classify wave polarization from the
b_perp1 / b_perp2 cross-correlation peak:
    - 90° phase shift → circular polarization (most common)
    - 180° phase shift → linear polarization

Stokes / ellipticity helpers live in :mod:`qp.signal.polarization`;
import them from there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from qp.signal.polarization_config import CIRCULAR_LINEAR_TOL_DEG

__all__ = [
    "cross_correlate",
    "phase_shift",
    "classify_polarization",
]


def cross_correlate(
    f: ArrayLike,
    g: ArrayLike,
    dt: float = 60.0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-correlation of two signals.

    Parameters
    ----------
    f, g : array_like
        Input signals (same length).
    dt : float
        Sampling interval in seconds.
    normalize : bool
        If True, normalize to [-1, 1] range.

    Returns
    -------
    lag : ndarray
        Lag times in seconds.
    xcorr : ndarray
        Cross-correlation values.
    """
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)
    N = len(f)

    xcorr = np.correlate(f, g, mode="full")

    if normalize:
        norm = np.sqrt(np.sum(f**2) * np.sum(g**2))
        if norm > 0:
            xcorr = xcorr / norm

    lag = np.arange(-(N - 1), N) * dt
    return lag, xcorr


def phase_shift(
    f: ArrayLike,
    g: ArrayLike,
    dt: float = 60.0,
    period: float = 3600.0,
) -> tuple[float, float]:
    """Determine the phase shift between two transverse components.

    Parameters
    ----------
    f, g : array_like
        The two perpendicular field components (b_perp1, b_perp2).
    dt : float
        Sampling interval in seconds.
    period : float
        Expected wave period in seconds (for phase conversion).

    Returns
    -------
    delay_sec : float
        Time delay at maximum cross-correlation (seconds).
    phase_deg : float
        Phase shift in degrees (0-360).
    """
    lag, xcorr = cross_correlate(f, g, dt, normalize=True)
    peak_idx = np.argmax(xcorr)
    delay_sec = lag[peak_idx]
    phase_deg = (delay_sec / period * 360.0) % 360.0
    return delay_sec, phase_deg


def classify_polarization(
    phase_deg: float,
    tolerance: float = CIRCULAR_LINEAR_TOL_DEG,
) -> str:
    """Classify wave polarization from phase shift.

    Returns 'circular', 'linear', or 'mixed'.
    """
    phase = phase_deg % 360.0

    for a in (90.0, 270.0):
        if abs(phase - a) < tolerance:
            return "circular"
    for a in (0.0, 180.0, 360.0):
        if abs(phase - a) < tolerance:
            return "linear"
    return "mixed"
