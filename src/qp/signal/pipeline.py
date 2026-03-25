r"""Spectral analysis pipeline for MAG data segments.

Replaces the ``NewSignal.analyze()`` method chain with a single composable
function that strings together detrending, Welch PSD, background estimation,
power ratios, and optionally spectrogram and CWT.

Usage
-----
>>> from qp.signal.pipeline import analyze_segment
>>> result = analyze_segment(data, dt=60.0)
>>> result.freq   # Hz
>>> result.psd    # nT^2/Hz
>>> result.power_ratio  # signal / background
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from qp.signal.fft import estimate_background, spectrogram, welch_psd
from qp.signal.timeseries import detrend_for_fft
from qp.signal.wavelet import cwt_power


@dataclass(frozen=True, slots=True)
class SpectralResult:
    r"""Complete spectral analysis result for one field component.

    Attributes
    ----------
    freq : ndarray
        Frequency array in Hz.
    psd : ndarray
        Power spectral density ($\mathrm{nT}^2/\mathrm{Hz}$).
    background : ndarray
        Estimated smooth background PSD.
    power_ratio : ndarray
        Ratio of PSD to background ($r_i$ in the paper).
    detrended : ndarray
        The detrended time series that was analyzed.
    trend : ndarray
        The removed trend (running average).
    spectrogram_result : tuple[np.ndarray, np.ndarray, np.ndarray] | None
        (freq, time, Sxx) if computed, else None.
    cwt_result : tuple[np.ndarray, np.ndarray, np.ndarray] | None
        (freq, time, power) if computed, else None.
    """

    freq: np.ndarray
    psd: np.ndarray
    background: np.ndarray
    power_ratio: np.ndarray
    detrended: np.ndarray
    trend: np.ndarray
    spectrogram_result: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    cwt_result: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def analyze_segment(
    data: ArrayLike,
    dt: float = 60.0,
    detrend_window_sec: float = 3 * 3600,
    welch_nperseg: int | None = None,
    welch_noverlap: int | None = None,
    welch_window: str = "hann",
    welch_nfft: int | None = None,
    bg_savgol_window: int = 17,
    bg_savgol_order: int = 3,
    include_spectrogram: bool = False,
    spec_nperseg: int | None = None,
    spec_noverlap: int | None = None,
    include_cwt: bool = False,
    cwt_omega0: float = 10.0,
    cwt_n_freqs: int = 500,
) -> SpectralResult:
    r"""Run the full spectral analysis pipeline on a time series.

    Steps:
    1. Detrend by removing a running average (default 3-hour window)
    2. Compute Welch PSD
    3. Estimate smooth background
    4. Compute power ratio = PSD / background
    5. Optionally compute spectrogram
    6. Optionally compute CWT power

    Parameters
    ----------
    data : array_like
        Input time series (e.g., one MFA field component in nT).
    dt : float
        Sampling interval in seconds (default 60 s for 1-min MAG data).
    detrend_window_sec : float
        Running average window for detrending, in seconds.
        Default 3 hours, matching the paper's choice.
    welch_nperseg : int, optional
        Segment length for Welch PSD. Default: full length of data.
    welch_noverlap : int, optional
        Overlap for Welch PSD. Default: nperseg // 2.
    welch_window : str
        Window function for Welch PSD.
    welch_nfft : int, optional
        FFT length (zero-padding if > nperseg).
    bg_savgol_window : int
        Savitzky-Golay window for background estimation.
    bg_savgol_order : int
        Savitzky-Golay polynomial order.
    include_spectrogram : bool
        If True, compute STFT spectrogram.
    spec_nperseg : int, optional
        Spectrogram segment length. Default: len(data) // 4.
    spec_noverlap : int, optional
        Spectrogram overlap. Default: 0.
    include_cwt : bool
        If True, compute CWT power spectrum.
    cwt_omega0 : float
        Morlet wavelet central frequency parameter.
    cwt_n_freqs : int
        Number of CWT frequency bins.

    Returns
    -------
    SpectralResult
        Complete analysis result with PSD, background, power ratio,
        and optional spectrogram/CWT.
    """
    data = np.asarray(data, dtype=float)

    # 1. Detrend
    detrended, trend = detrend_for_fft(data, dt=dt, window_sec=detrend_window_sec)

    # 2. Welch PSD
    freq, psd = welch_psd(
        detrended,
        dt=dt,
        nperseg=welch_nperseg,
        noverlap=welch_noverlap,
        window=welch_window,
        nfft=welch_nfft,
    )

    # 3. Background estimation
    background = estimate_background(
        psd,
        freq,
        savgol_window=bg_savgol_window,
        savgol_order=bg_savgol_order,
    )

    # 4. Power ratio
    bg_safe = np.where(background > 0, background, 1e-30)
    power_ratio = psd / bg_safe

    # 5. Optional spectrogram
    spectrogram_result = None
    if include_spectrogram:
        if spec_nperseg is None:
            spec_nperseg = max(len(detrended) // 4, 64)
        spectrogram_result = spectrogram(
            detrended,
            dt=dt,
            nperseg=spec_nperseg,
            noverlap=spec_noverlap,
            window=welch_window,
        )

    # 6. Optional CWT
    cwt_result = None
    if include_cwt:
        cwt_result = cwt_power(
            detrended,
            dt=dt,
            omega0=cwt_omega0,
            n_freqs=cwt_n_freqs,
        )

    return SpectralResult(
        freq=freq,
        psd=psd,
        background=background,
        power_ratio=power_ratio,
        detrended=detrended,
        trend=trend,
        spectrogram_result=spectrogram_result,
        cwt_result=cwt_result,
    )
