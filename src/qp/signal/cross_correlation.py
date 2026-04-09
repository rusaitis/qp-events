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


# ----------------------------------------------------------------------
# Phase 6.5 — Stokes / ellipticity / inclination from two perpendicular
# components.
#
# The "circular vs linear" categorical label is too lossy: a 75°
# phase shift is "mixed" but is much closer to circular than linear.
# This module gives the continuous Stokes-parameter description so
# Fig 10 can plot a histogram of ellipticities instead of a 3-way
# bar chart.
# ----------------------------------------------------------------------


def stokes_parameters(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
) -> tuple[float, float, float, float]:
    r"""Stokes parameters of a two-component time series.

    Computes the standard Stokes parameters for a quasi-monochromatic
    transverse wave from the analytic signal of two orthogonal
    components:

    .. math::

        I &= \langle |b_\perp1|^2\rangle + \langle |b_\perp2|^2\rangle \\
        Q &= \langle |b_\perp1|^2\rangle - \langle |b_\perp2|^2\rangle \\
        U &= 2 \mathrm{Re}\langle b_\perp1^* b_\perp2 \rangle \\
        V &= 2 \mathrm{Im}\langle b_\perp1^* b_\perp2 \rangle

    Where the analytic signal is built via the Hilbert transform.

    Returns
    -------
    I, Q, U, V : float
    """
    from scipy.signal import hilbert

    a = hilbert(np.asarray(b_perp1, dtype=float))
    b = hilbert(np.asarray(b_perp2, dtype=float))
    I = float(np.mean(np.abs(a) ** 2 + np.abs(b) ** 2))
    Q = float(np.mean(np.abs(a) ** 2 - np.abs(b) ** 2))
    U = float(2 * np.real(np.mean(np.conj(a) * b)))
    V = float(2 * np.imag(np.mean(np.conj(a) * b)))
    return I, Q, U, V


def ellipticity_inclination(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
) -> tuple[float, float, float]:
    r"""Ellipticity and inclination of a transverse wave packet.

    Returns
    -------
    ellipticity : float
        Signed ratio of minor / major polarization-ellipse axes.
        +1 = right-circular, -1 = left-circular, 0 = linear.
    inclination_deg : float
        Tilt angle of the major axis from the b_perp1 axis (degrees).
    polarization_fraction : float
        Fraction of total power that is polarized
        ``sqrt(Q^2 + U^2 + V^2) / I``. 1.0 = fully polarized.

    The standard formulas in terms of Stokes:
        chi = 0.5 * arcsin(V / sqrt(Q² + U² + V²))   # ellipticity angle
        psi = 0.5 * arctan2(U, Q)                     # inclination
        ellipticity = tan(chi)
    """
    I, Q, U, V = stokes_parameters(b_perp1, b_perp2)
    p = np.sqrt(Q ** 2 + U ** 2 + V ** 2)
    if p <= 0 or I <= 0:
        return 0.0, 0.0, 0.0
    chi = 0.5 * np.arcsin(np.clip(V / p, -1.0, 1.0))
    psi = 0.5 * np.arctan2(U, Q)
    return float(np.tan(chi)), float(np.degrees(psi)), float(p / I)
