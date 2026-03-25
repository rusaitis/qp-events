"""FFT and Welch power spectral density estimation.

Extracted from cassinilib/NewSignal.py — the spectral analysis core
used for computing power spectra of 36-hour MAG data segments.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal as sig
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def welch_psd(
    data: ArrayLike,
    dt: float = 60.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    nfft: int | None = None,
    scaling: str = "density",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch power spectral density estimate.

    Parameters
    ----------
    data : array_like
        Input time series.
    dt : float
        Sampling interval in seconds (default 60s for 1-min MAG data).
    nperseg : int, optional
        Segment length for Welch averaging. Default: len(data).
    noverlap : int, optional
        Overlap between segments. Default: nperseg // 2.
    window : str
        Window function name (default 'hann' as in the paper).
    nfft : int, optional
        FFT length (zero-padding if > nperseg).
    scaling : str
        'density' for PSD (V^2/Hz), 'spectrum' for power spectrum.

    Returns
    -------
    freq : ndarray
        Frequency array in Hz.
    psd : ndarray
        Power spectral density.
    """
    data = np.asarray(data, dtype=float)
    fs = 1.0 / dt

    if nperseg is None:
        nperseg = len(data)
    if noverlap is None:
        noverlap = nperseg // 2

    freq, psd = sig.welch(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        nfft=nfft,
        detrend=False,
        scaling=scaling,
        average="mean",
    )
    return freq, psd


def spectrogram(
    data: ArrayLike,
    dt: float = 60.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    nfft: int | None = None,
    scaling: str = "density",
    mode: str = "psd",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram via short-time Fourier transform.

    Returns
    -------
    freq : ndarray
        Frequency array.
    time : ndarray
        Time array (segment centers).
    Sxx : ndarray
        Spectrogram values, shape (n_freq, n_time).
    """
    data = np.asarray(data, dtype=float)
    fs = 1.0 / dt

    if nperseg is None:
        nperseg = len(data)
    if noverlap is None:
        noverlap = 0

    freq, time, Sxx = sig.spectrogram(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        nfft=nfft,
        detrend=False,
        scaling=scaling,
        mode=mode,
    )
    return freq, time, Sxx


def estimate_background(
    psd: ArrayLike,
    freq: ArrayLike,
    savgol_window: int = 17,
    savgol_order: int = 3,
    freq_low: float = 1 / (3 * 3600),
    freq_high: float = 1 / (30 * 60),
    target_fraction: float = 0.5,
    tolerance: float = 0.01,
    max_iterations: int = 50,
) -> np.ndarray:
    """Estimate the smooth background of an FFT power spectrum.

    Works in log-log space: fits a Savitzky-Golay filter to the log(PSD)
    vs log(freq), then iteratively adjusts the level so that ~50% of the
    spectral points in the 30min-3h band lie above the background.

    Parameters
    ----------
    psd : array_like
        Power spectral density values.
    freq : array_like
        Corresponding frequency array (Hz).
    savgol_window : int
        Window length for Savitzky-Golay filter (must be odd).
    savgol_order : int
        Polynomial order for Savitzky-Golay filter.
    freq_low, freq_high : float
        Frequency bounds (Hz) for the adjustment band.
    target_fraction : float
        Target fraction of points above background (0.5 = median).
    tolerance : float
        Convergence tolerance on the fraction.
    max_iterations : int
        Maximum number of adjustment iterations.

    Returns
    -------
    background : ndarray
        Smooth background estimate, same shape as psd.
    """
    psd = np.asarray(psd, dtype=float)
    freq = np.asarray(freq, dtype=float)

    # Work in log space
    log_freq = np.log(np.where(freq > 0, freq, 1e-10))
    log_psd = np.log(np.where(psd > 0, psd, 1e-30))

    # Resample to fewer points for smooth fitting
    n_resample = savgol_window * 3
    x_new = np.linspace(log_freq[0], log_freq[-1], n_resample)
    f_interp = interp1d(log_freq, log_psd, kind="cubic", fill_value="extrapolate")
    y_resampled = f_interp(x_new)

    # Savitzky-Golay smooth in log space
    bg_smooth = savgol_filter(y_resampled, savgol_window, savgol_order, mode="nearest")

    # Find the critical frequency band for adjustment
    idx_low = np.searchsorted(freq, freq_low)
    idx_high = np.searchsorted(freq, freq_high)
    n_band = max(idx_high - idx_low, 1)

    # Iteratively adjust background level
    med_diff = np.median(y_resampled[idx_low:idx_high]) - np.median(
        bg_smooth[idx_low:idx_high]
    )
    adjust = abs(med_diff / 10)

    for _ in range(max_iterations):
        diff = y_resampled - bg_smooth
        n_above = np.count_nonzero(diff[idx_low:idx_high] > 0)
        fraction = n_above / n_band
        error = target_fraction - fraction

        if abs(error) < tolerance:
            break

        direction = -1.0 if error > 0 else 1.0
        bg_smooth += direction * adjust
        adjust *= 0.8  # shrink step

    # Interpolate back to original frequency grid
    f_bg = interp1d(x_new, bg_smooth, kind="cubic", fill_value="extrapolate")
    background = np.exp(f_bg(log_freq))

    return background
