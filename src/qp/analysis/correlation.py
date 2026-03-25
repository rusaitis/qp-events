r"""Statistical correlation analysis for wave event properties.

Distinct from the signal-level cross-correlation in ``qp.signal.cross_correlation``
— this module operates on *collections of events*, computing correlations between
event metadata (period vs amplitude, occurrence vs local time, etc.).

Also provides a sliding-window cross-correlation phase estimator, extracted
from ``cassinilib/PlotFFT.py:calculateCorrelation()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal as sig
from scipy.stats import pearsonr, spearmanr


def event_property_correlation(
    events: Sequence[Any],
    x_key: str,
    y_key: str,
    method: str = "spearman",
) -> tuple[float, float]:
    r"""Compute correlation between two event properties.

    Parameters
    ----------
    events : sequence
        Event objects with attributes accessible via ``getattr``.
    x_key, y_key : str
        Attribute names to correlate.
    method : str
        ``'spearman'`` (default, rank-based) or ``'pearson'`` (linear).

    Returns
    -------
    correlation : float
        Correlation coefficient.
    p_value : float
        Two-sided p-value for the null hypothesis of zero correlation.

    Raises
    ------
    ValueError
        If fewer than 3 events have non-None values for both keys.
    """
    x_vals = []
    y_vals = []
    for event in events:
        xv = getattr(event, x_key, None)
        yv = getattr(event, y_key, None)
        if xv is not None and yv is not None:
            x_vals.append(float(xv))
            y_vals.append(float(yv))

    if len(x_vals) < 3:
        raise ValueError(
            f"Need at least 3 events with non-None {x_key!r} and {y_key!r}, "
            f"got {len(x_vals)}"
        )

    x = np.array(x_vals)
    y = np.array(y_vals)

    match method:
        case "spearman":
            corr, pval = spearmanr(x, y)
        case "pearson":
            corr, pval = pearsonr(x, y)
        case _:
            raise ValueError(
                f"Unknown method: {method!r}. Use 'spearman' or 'pearson'."
            )

    return float(corr), float(pval)


def sliding_phase_lag(
    y1: ArrayLike,
    y2: ArrayLike,
    dt: float = 60.0,
    window_samples: int = 121,
    max_lag_degrees: float = 180.0,
) -> np.ndarray:
    r"""Estimate phase lag between two signals using sliding cross-correlation.

    Slides a window across the two signals, computes cross-correlation in each
    window, and returns the lag (in degrees) at the correlation peak.

    Extracted from ``cassinilib/PlotFFT.py:calculateCorrelation()``.

    Parameters
    ----------
    y1, y2 : array_like
        Two signal components (e.g., $b_{\perp 1}$ and $b_{\perp 2}$).
    dt : float
        Sampling interval in seconds.
    window_samples : int
        Half-width of the sliding window in samples. Should be odd.
    max_lag_degrees : float
        Maximum lag to consider (degrees). Correlation outside this
        range is zeroed out.

    Returns
    -------
    phase_lag : ndarray
        Phase lag in degrees at each interior sample. Length is
        ``len(y1) - window_samples + 1`` (valid region only).
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    n = len(y1)
    half_w = window_samples // 2

    phase_lags = []
    for center in range(half_w, n - half_w):
        lo = center - half_w
        hi = center + half_w
        seg1 = y1[lo:hi]
        seg2 = y2[lo:hi]
        npts = len(seg1)

        ccor = sig.correlate(seg2, seg1, mode="same", method="direct")
        sr = 1.0 / dt
        delay_arr = np.linspace(-npts / sr, npts / sr, npts)
        delay_deg = delay_arr / 3600.0 * 180.0  # seconds → hours → degrees

        # Zero out correlation outside the allowed lag range
        outside = np.abs(delay_deg) > max_lag_degrees
        ccor[outside] = 0.0

        lag = delay_deg[np.argmax(ccor)]
        phase_lags.append(lag)

    return np.array(phase_lags)
