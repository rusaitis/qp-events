"""Time series preprocessing: resampling, detrending, smoothing.

Core signal operations extracted from cassinilib/NewSignal.py.
These are pure functions that operate on numpy arrays — no class state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


def uniform_resample(
    data: ArrayLike,
    times_unix: ArrayLike,
    dt: float = 60.0,
    n_samples: int | None = None,
    start_time: float | None = None,
    nan_method: str = "zero",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample irregularly-spaced data onto a uniform time grid.

    Parameters
    ----------
    data : array_like
        Signal values.
    times_unix : array_like
        Timestamps in Unix seconds (or any monotonic float).
    dt : float
        Target sampling interval in seconds.
    n_samples : int, optional
        Number of output samples. Default: spans full time range.
    start_time : float, optional
        Start time for the output grid. Default: times_unix[0].
    nan_method : str
        How to handle NaNs before interpolation:
        'zero' (replace with 0), 'mean', 'linear' (interpolate).

    Returns
    -------
    resampled_data : ndarray
    new_times : ndarray
        Uniform time grid.
    """
    data = np.asarray(data, dtype=float)
    t = np.asarray(times_unix, dtype=float)

    # Handle NaNs
    data = _handle_nans(data, nan_method)

    if start_time is None:
        start_time = t[0]
    if n_samples is None:
        n_samples = int((t[-1] - start_time) / dt)

    new_times = start_time + np.arange(n_samples) * dt

    f = interp1d(
        t, data, kind="linear", fill_value=(data[0], data[-1]), bounds_error=False
    )
    resampled = f(new_times)

    return resampled, new_times


def running_average(
    data: ArrayLike,
    window_samples: int,
) -> np.ndarray:
    """Compute running average with a uniform filter.

    Parameters
    ----------
    data : array_like
        Input signal.
    window_samples : int
        Window size in samples (will be forced to odd).
    """
    data = np.asarray(data, dtype=float)
    window_samples = _nearest_odd(window_samples)
    return uniform_filter1d(data, size=window_samples, mode="nearest")


def detrend(
    data: ArrayLike,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove running average (trend) from data.

    Returns (detrended, trend).
    """
    data = np.asarray(data, dtype=float)
    trend = running_average(data, window_samples)
    return data - trend, trend


def smooth_savgol(
    data: ArrayLike,
    window_samples: int,
    order: int = 3,
) -> np.ndarray:
    """Smooth data with Savitzky-Golay filter."""
    data = np.asarray(data, dtype=float)
    window_samples = _nearest_odd(window_samples)
    return savgol_filter(data, window_samples, order)


def block_average(
    data: ArrayLike,
    block_size: int,
) -> np.ndarray:
    r"""Bin-average an array into blocks, ignoring NaN values.

    If the array length isn't evenly divisible by ``block_size``, the final
    partial block is padded with NaN before averaging.

    Replaces ``cassinilib/Core.py:AvgArray()``.

    Parameters
    ----------
    data : array_like
        Input array.
    block_size : int
        Number of samples per block.

    Returns
    -------
    ndarray
        Array of length ``ceil(len(data) / block_size)``.
    """
    data = np.asarray(data, dtype=float)
    block_size = int(block_size)
    remainder = len(data) % block_size
    if remainder != 0:
        pad_width = block_size - remainder
        data = np.pad(data, (0, pad_width), constant_values=np.nan)
    return np.nanmean(data.reshape(-1, block_size), axis=1)


def detrend_for_fft(
    data: ArrayLike,
    dt: float = 60.0,
    window_sec: float = 3 * 3600,
) -> tuple[np.ndarray, np.ndarray]:
    """Detrend a time series for spectral analysis.

    Removes a running average with the specified window (default 3 hours,
    matching the paper's choice for resolving waves up to 3h period).

    Returns (detrended, trend).
    """
    window_samples = _nearest_odd(int(window_sec / dt))
    return detrend(data, window_samples)


# --- Helpers ---


def _nearest_odd(n: int) -> int:
    """Round to nearest odd integer (required by Savitzky-Golay)."""
    n = int(np.ceil(n))
    return n if n % 2 == 1 else n + 1


def _handle_nans(data: np.ndarray, method: str = "zero") -> np.ndarray:
    """Replace NaN values in data."""
    if not np.any(np.isnan(data)):
        return data

    data = data.copy()
    if method == "zero":
        data = np.nan_to_num(data, nan=0.0)
    elif method == "mean":
        mean_val = np.nanmean(data)
        data = np.where(np.isnan(data), mean_val, data)
    elif method == "linear":
        nans = np.isnan(data)
        if np.any(nans):
            x = np.arange(len(data))
            data[nans] = np.interp(x[nans], x[~nans], data[~nans])
    return data
