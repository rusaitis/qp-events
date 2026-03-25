"""Cross-correlation and polarization analysis (Equations 6-7).

    tau_delay = argmax_t ((f * g)(t))       (Eq 6)
    (f * g)(tau) = integral f(t) g(t+tau) dt  (Eq 7)

Used to determine the polarization of QP waves:
    - 90 deg phase shift = circular polarization (most common)
    - 180 deg phase shift = linear polarization
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


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

    # Convert time delay to phase
    phase_deg = (delay_sec / period * 360.0) % 360.0

    return delay_sec, phase_deg


def classify_polarization(phase_deg: float, tolerance: float = 30.0) -> str:
    """Classify wave polarization from phase shift.

    Returns 'circular', 'linear', or 'mixed'.
    """
    # Normalize to 0-360
    phase = phase_deg % 360.0

    # Check proximity to 90/270 (circular) or 0/180/360 (linear)
    circular_angles = [90.0, 270.0]
    linear_angles = [0.0, 180.0, 360.0]

    for a in circular_angles:
        if abs(phase - a) < tolerance:
            return "circular"
    for a in linear_angles:
        if abs(phase - a) < tolerance:
            return "linear"
    return "mixed"
