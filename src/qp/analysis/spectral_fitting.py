r"""Spectral slope estimation and power-law fitting.

Extracts the data-processing parts of ``cassinilib/PlotFFT.py:PowerLawFit()``,
``PowerLawSegmenter()``, ``powerBinner()``, and ``calculateSlopes()``.
All plotting logic is left behind — these are pure numerical functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class PowerLawResult:
    r"""Result of a log-log polynomial fit.

    Attributes
    ----------
    coefficients : ndarray
        Polynomial coefficients in log-space, lowest degree first.
        For degree 1: ``[intercept, slope]``.
    slope : float
        The linear slope (first-degree coefficient).
    freq_fit : ndarray or None
        Frequency array of the fitted curve (if ``return_fit=True``).
    psd_fit : ndarray or None
        Fitted PSD values (if ``return_fit=True``).
    """

    coefficients: np.ndarray
    slope: float
    freq_fit: np.ndarray | None = None
    psd_fit: np.ndarray | None = None


def power_law_fit(
    freq: ArrayLike,
    psd: ArrayLike,
    freq_range: tuple[float, float] | None = None,
    degree: int = 1,
    return_fit: bool = False,
    fit_range: tuple[float, float] | None = None,
) -> PowerLawResult:
    r"""Fit a polynomial in log-log space to a power spectrum.

    Replaces ``cassinilib/PlotFFT.py:PowerLawFit()``.

    Parameters
    ----------
    freq : array_like
        Frequency array (Hz). Must be positive.
    psd : array_like
        Power spectral density.
    freq_range : tuple[float, float], optional
        Frequency range to fit over (Hz). Default: full range (excluding f=0).
    degree : int
        Polynomial degree in log-log space. Default 1 (linear = power law).
    return_fit : bool
        If True, compute and return the fitted curve.
    fit_range : tuple[float, float], optional
        Frequency range for the output fitted curve. Default: same as
        ``freq_range``.

    Returns
    -------
    PowerLawResult
        Fit coefficients, slope, and optional fitted curve.
    """
    freq = np.asarray(freq, dtype=float)
    psd = np.asarray(psd, dtype=float)

    # Determine index range
    if freq_range is not None:
        mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
    else:
        mask = freq > 0  # exclude f=0

    log_f = np.log(freq[mask])
    log_p = np.log(psd[mask])

    fit = Polynomial.fit(log_f, log_p, degree)
    coefs = fit.convert().coef  # standard polynomial coefficients

    slope = float(coefs[1]) if len(coefs) > 1 else 0.0

    freq_fit = None
    psd_fit = None
    if return_fit:
        if fit_range is not None:
            fit_mask = (freq >= fit_range[0]) & (freq <= fit_range[1])
        else:
            fit_mask = mask
        log_f_out = np.log(freq[fit_mask])
        log_p_fit = np.polynomial.polynomial.polyval(
            log_f_out,
            np.polynomial.polynomial.polyline(coefs[0], slope),
        )
        freq_fit = freq[fit_mask]
        psd_fit = np.exp(log_p_fit)

    return PowerLawResult(
        coefficients=coefs,
        slope=slope,
        freq_fit=freq_fit,
        psd_fit=psd_fit,
    )


def spectral_slopes(
    freq: ArrayLike,
    psd: ArrayLike,
    bands: dict[str, tuple[float, float]] | None = None,
    degree: int = 1,
) -> dict[str, float]:
    r"""Compute spectral slopes in predefined frequency bands.

    Extracts the slope-computation logic from
    ``cassinilib/PlotFFT.py:calculateSlopes()`` (without the seaborn plotting).

    Parameters
    ----------
    freq : array_like
        Frequency array (Hz).
    psd : array_like
        Power spectral density.
    bands : dict[str, tuple[float, float]], optional
        Named frequency bands as ``{name: (f_low, f_high)}``.
        Default: ``{'low': (1/10800, 1/1800), 'high': (1/1800, 1/300)}``.
    degree : int
        Polynomial degree for each band fit.

    Returns
    -------
    dict[str, float]
        Spectral slope for each band.
    """
    if bands is None:
        bands = {
            "low": (1 / 10800, 1 / 1800),  # 3h–30min
            "high": (1 / 1800, 1 / 300),  # 30min–5min
        }

    slopes: dict[str, float] = {}
    for name, (f_lo, f_hi) in bands.items():
        result = power_law_fit(freq, psd, freq_range=(f_lo, f_hi), degree=degree)
        slopes[name] = result.slope

    return slopes


def bin_power_spectra(
    psd: ArrayLike,
    center_indices: list[int],
    half_width: int = 3,
) -> np.ndarray:
    r"""Estimate median power in frequency bins around specified centers.

    Replaces ``cassinilib/PlotFFT.py:powerBinner()``.

    Parameters
    ----------
    psd : array_like
        Power spectral density array.
    center_indices : list[int]
        Indices of bin centers in the PSD array.
    half_width : int
        Number of frequency bins on each side of center to include.

    Returns
    -------
    ndarray, shape (n_bins,)
        Median power in each bin.
    """
    psd = np.asarray(psd, dtype=float)
    n = len(psd)
    powers = np.empty(len(center_indices))

    for i, center in enumerate(center_indices):
        lo = max(1, center - half_width)
        hi = min(n - 1, center + half_width)
        powers[i] = np.median(psd[lo : hi + 1])

    return powers
