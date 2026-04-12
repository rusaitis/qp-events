r"""Wavelet coherence between two transverse components.

Implements the wavelet coherence spectrum $C^2(f, t)$ and cross-wavelet
phase $\Delta\phi(f, t)$ following Torrence & Compo (1998). A real
Alfven wave produces $C \approx 1$ with a stable phase difference
($\sim 90°$ for circular, $\sim 180°$ for linear polarisation). Random
noise in two independent channels gives $C \approx 0$.

This is a much more discriminating gate than the binary
``require_both_perp`` flag from Phase 6.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from qp.signal.wavelet import morlet_cwt


def _smooth_in_time(
    W: NDArray[np.complexfloating],
    scales: NDArray[np.floating],
    dt: float,
) -> NDArray[np.complexfloating]:
    r"""Smooth a CWT matrix in time using a Gaussian whose width scales
    with the wavelet scale (Torrence & Webster 1999, Eq. 6)."""
    smoothed = np.empty_like(W)
    for i, s in enumerate(scales):
        sigma_samples = max(1.0, 0.6 * s / dt)
        smoothed[i] = gaussian_filter1d(W[i].real, sigma_samples) + \
                       1j * gaussian_filter1d(W[i].imag, sigma_samples)
    return smoothed


def _smooth_in_scale(
    W: NDArray[np.complexfloating],
    n_octaves: float = 0.6,
) -> NDArray[np.complexfloating]:
    r"""Smooth in scale using a boxcar filter spanning ~0.6 octaves."""

    n_scales = W.shape[0]
    # In log-scale space, 0.6 octaves ≈ a few rows
    width = max(3, int(n_octaves * n_scales / np.log2(n_scales + 1)))
    if width % 2 == 0:
        width += 1
    smoothed = np.empty_like(W)
    for j in range(W.shape[1]):
        smoothed[:, j] = (
            uniform_filter1d(W[:, j].real, width) +
            1j * uniform_filter1d(W[:, j].imag, width)
        )
    return smoothed


def wavelet_coherence(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
    dt: float = 60.0,
    omega0: float = 10.0,
    n_freqs: int = 300,
) -> tuple[NDArray[np.floating], NDArray[np.floating],
           NDArray[np.floating], NDArray[np.floating]]:
    r"""Wavelet coherence spectrum between two transverse components.

    Parameters
    ----------
    b_perp1, b_perp2 : array_like
        Two orthogonal transverse MFA components.
    dt : float
        Sampling interval in seconds.
    omega0 : float
        Morlet wavelet parameter.
    n_freqs : int
        Number of CWT frequency rows.

    Returns
    -------
    freq : ndarray, shape (n_freqs,)
        Frequency axis in Hz.
    coherence : ndarray, shape (n_freqs, n_time)
        Magnitude-squared coherence $|C|^2 \in [0, 1]$.
    phase_diff : ndarray, shape (n_freqs, n_time)
        Cross-wavelet phase difference in degrees.
    cwt_power_mean : ndarray, shape (n_freqs, n_time)
        Mean of |cwt1| and |cwt2| (useful for ridge overlay).
    """
    b_perp1 = np.asarray(b_perp1, dtype=float)
    b_perp2 = np.asarray(b_perp2, dtype=float)

    freq1, _, cwt1 = morlet_cwt(b_perp1, dt=dt, omega0=omega0,
                                 n_freqs=n_freqs)
    _, _, cwt2 = morlet_cwt(b_perp2, dt=dt, omega0=omega0,
                             n_freqs=n_freqs)

    # Scales for smoothing (scale = 1 / (freq * omega0 / (2π)) )
    scales = np.where(freq1 > 0, 1.0 / freq1, 1e10)

    # Cross-wavelet spectrum
    W12 = cwt1 * np.conj(cwt2)

    # Auto-spectra
    S11 = np.abs(cwt1) ** 2
    S22 = np.abs(cwt2) ** 2

    # Smooth in time and scale
    S12 = _smooth_in_time(W12, scales, dt)
    S12 = _smooth_in_scale(S12)
    S11s = _smooth_in_time(S11.astype(complex), scales, dt).real
    S11s = _smooth_in_scale(S11s.astype(complex)).real
    S22s = _smooth_in_time(S22.astype(complex), scales, dt).real
    S22s = _smooth_in_scale(S22s.astype(complex)).real

    # Coherence: |<W12>|² / (<|W1|²> <|W2|²>)
    denom = S11s * S22s
    denom = np.maximum(denom, 1e-30)
    coherence = np.abs(S12) ** 2 / denom
    coherence = np.clip(coherence, 0.0, 1.0)

    # Phase difference from the smoothed cross-spectrum
    phase_diff = np.degrees(np.angle(S12))

    cwt_power_mean = (np.abs(cwt1) + np.abs(cwt2)) / 2.0

    return freq1, coherence, phase_diff, cwt_power_mean


def ridge_coherence(
    coherence: NDArray[np.floating],
    phase_diff: NDArray[np.floating],
    cwt_freq: NDArray[np.floating],
    band_freq_min: float,
    band_freq_max: float,
    t_start_idx: int,
    t_end_idx: int,
) -> tuple[float, float]:
    r"""Mean coherence and phase difference over a ridge footprint.

    Parameters
    ----------
    coherence : ndarray, shape (n_freq, n_time)
    phase_diff : ndarray, shape (n_freq, n_time)
    cwt_freq : ndarray, shape (n_freq,)
    band_freq_min, band_freq_max : float
        Frequency bounds of the band (Hz).
    t_start_idx, t_end_idx : int
        Time index range of the ridge.

    Returns
    -------
    mean_coherence : float
        Mean coherence over the ridge footprint.
    mean_phase_deg : float
        Circular mean of phase difference (degrees).
    """
    freq_mask = (cwt_freq >= band_freq_min) & (cwt_freq < band_freq_max)
    if not freq_mask.any():
        return 0.0, 0.0

    sub_coh = coherence[freq_mask, t_start_idx:t_end_idx + 1]
    sub_phase = phase_diff[freq_mask, t_start_idx:t_end_idx + 1]

    mean_coh = float(np.nanmean(sub_coh))

    # Circular mean of phase
    phase_rad = np.radians(sub_phase)
    mean_sin = np.nanmean(np.sin(phase_rad))
    mean_cos = np.nanmean(np.cos(phase_rad))
    mean_phase = float(np.degrees(np.arctan2(mean_sin, mean_cos)))

    return mean_coh, mean_phase
