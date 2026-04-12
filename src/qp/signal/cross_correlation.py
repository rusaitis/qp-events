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
from scipy.signal import hilbert, windows


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
    phase = phase_deg % 360.0

    for a in (90.0, 270.0):
        if abs(phase - a) < tolerance:
            return "circular"
    for a in (0.0, 180.0, 360.0):
        if abs(phase - a) < tolerance:
            return "linear"
    return "mixed"


# ----------------------------------------------------------------------
# Stokes / ellipticity / inclination from two perpendicular components.
# ----------------------------------------------------------------------


def _stokes_from_analytic(
    a: np.ndarray, b: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute Stokes parameters from analytic (Hilbert-transformed) signals."""
    abs_a2 = np.abs(a) ** 2
    abs_b2 = np.abs(b) ** 2
    cross = np.conj(a) * b
    si = float(np.mean(abs_a2 + abs_b2))  # Stokes I
    sq = float(np.mean(abs_a2 - abs_b2))  # Stokes Q
    su = float(2 * np.real(np.mean(cross)))  # Stokes U
    sv = float(2 * np.imag(np.mean(cross)))  # Stokes V
    return si, sq, su, sv


def _ellipticity_from_stokes(
    si: float, sq: float, su: float, sv: float,
) -> tuple[float, float, float]:
    """Derive ellipticity, inclination, polarization fraction from Stokes."""
    p = np.sqrt(sq**2 + su**2 + sv**2)
    if p <= 0 or si <= 0:
        return 0.0, 0.0, 0.0
    chi = 0.5 * np.arcsin(np.clip(sv / p, -1.0, 1.0))
    psi = 0.5 * np.arctan2(su, sq)
    return float(np.tan(chi)), float(np.degrees(psi)), float(p / si)


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
    a = hilbert(np.asarray(b_perp1, dtype=float))
    b = hilbert(np.asarray(b_perp2, dtype=float))
    return _stokes_from_analytic(a, b)


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
    """
    return _ellipticity_from_stokes(*stokes_parameters(b_perp1, b_perp2))


def stokes_parameters_tapered(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
    alpha: float = 0.25,
) -> tuple[float, float, float, float]:
    r"""Stokes parameters with a Tukey (cosine) taper to suppress edge effects.

    The Hilbert transform used in :func:`stokes_parameters` is sensitive
    to edge discontinuities in short windows (<5 oscillations). The
    Tukey taper smoothly rolls the signal to zero at both ends, removing
    the edge artifact without distorting the central portion.

    Parameters
    ----------
    b_perp1, b_perp2 : array_like
    alpha : float
        Fraction of the window inside the cosine taper (0 = rectangular,
        1 = Hann). Default 0.25 tapers the outer 12.5% on each side.
    """
    b1 = np.asarray(b_perp1, dtype=float)
    b2 = np.asarray(b_perp2, dtype=float)
    taper = windows.tukey(len(b1), alpha=alpha)
    a = hilbert(b1 * taper)
    b = hilbert(b2 * taper)
    return _stokes_from_analytic(a, b)


def ellipticity_inclination_tapered(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
    alpha: float = 0.25,
) -> tuple[float, float, float]:
    r"""Ellipticity with Tukey taper — fixes edge artifacts in short events."""
    return _ellipticity_from_stokes(
        *stokes_parameters_tapered(b_perp1, b_perp2, alpha=alpha)
    )


def per_oscillation_ellipticity(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
    dt: float = 60.0,
    period: float = 3600.0,
) -> tuple[float, float]:
    r"""Median per-oscillation ellipticity and its spread.

    Splits the event window into individual cycles of the peak period,
    computes Stokes-derived ellipticity for each cycle, and returns the
    median and IQR. This avoids washing out polarization rotation across
    a multi-hour event.

    Returns
    -------
    median_ellipticity : float
    iqr_ellipticity : float
        Inter-quartile range (75th - 25th percentile).
    """
    b1 = np.asarray(b_perp1, dtype=float)
    b2 = np.asarray(b_perp2, dtype=float)
    samples_per_cycle = int(round(period / dt))
    if samples_per_cycle < 4 or len(b1) < samples_per_cycle:
        e, _, _ = ellipticity_inclination_tapered(b1, b2)
        return e, 0.0

    n_cycles = len(b1) // samples_per_cycle
    ellipticities = []
    for i in range(n_cycles):
        lo = i * samples_per_cycle
        hi = lo + samples_per_cycle
        e, _, _ = ellipticity_inclination_tapered(b1[lo:hi], b2[lo:hi])
        if np.isfinite(e):
            ellipticities.append(e)

    if not ellipticities:
        e, _, _ = ellipticity_inclination_tapered(b1, b2)
        return e, 0.0

    arr = np.array(ellipticities)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return float(q50), float(q75 - q25)
