"""Morlet continuous wavelet transform for quasi-periodic event detection.

Uses the Morlet wavelet (Eq 5 in the paper):
    psi(t, omega) = c_omega * exp(-t^2/sigma^2) * exp(i*omega*t)

Implements CWT directly since scipy.signal.cwt was removed in scipy 1.15.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import fftconvolve


def morlet_cwt(
    data: ArrayLike,
    dt: float = 60.0,
    omega0: float = 10.0,
    freq_min: float | None = None,
    freq_max: float | None = None,
    n_freqs: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute continuous wavelet transform using Morlet wavelet.

    Parameters
    ----------
    data : array_like
        Input time series.
    dt : float
        Sampling interval in seconds.
    omega0 : float
        Central frequency parameter of the Morlet wavelet.
        Higher values give better frequency resolution at the cost
        of time resolution.
    freq_min : float, optional
        Minimum frequency in Hz. Default: 1/(4 hours).
    freq_max : float, optional
        Maximum frequency in Hz. Default: Nyquist/2.
    n_freqs : int
        Number of frequency bins.

    Returns
    -------
    freq : ndarray
        Frequency array in Hz, shape (n_freqs,).
    time : ndarray
        Time array in seconds, shape (N,).
    cwt_matrix : ndarray
        Complex CWT coefficients, shape (n_freqs, N).
    """
    data = np.asarray(data, dtype=float)
    fs = 1.0 / dt
    N = len(data)

    if freq_min is None:
        freq_min = 1.0 / (4 * 3600)
    if freq_max is None:
        freq_max = fs / 2

    time = np.arange(N) * dt
    freq = np.linspace(freq_min, freq_max, n_freqs)

    # Scale parameter: width in samples for each frequency
    # At scale s, the wavelet has central frequency omega0/(2*pi*s*dt)
    # So s = omega0 / (2*pi*f*dt)
    scales = omega0 / (2 * np.pi * freq * dt)

    cwt_matrix = np.empty((n_freqs, N), dtype=complex)
    for i, s in enumerate(scales):
        # Wavelet length: ~10 standard deviations of the Gaussian
        wavelet_len = min(int(10 * s) * 2 + 1, N)
        if wavelet_len < 3:
            wavelet_len = 3
        t_wav = (np.arange(wavelet_len) - (wavelet_len - 1) / 2) / s
        wavelet = (
            (np.pi ** (-0.25)) * np.exp(1j * omega0 * t_wav) * np.exp(-(t_wav**2) / 2)
        )
        wavelet /= np.sqrt(s)  # energy normalization

        # Convolve and trim to original length
        conv = fftconvolve(data, np.conj(wavelet[::-1]), mode="same")
        cwt_matrix[i] = conv

    return freq, time, cwt_matrix


def cwt_power(
    data: ArrayLike,
    dt: float = 60.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CWT power spectrum (|CWT|^2).

    Returns freq, time, power arrays.
    """
    freq, time, cwt_matrix = morlet_cwt(data, dt, **kwargs)
    return freq, time, np.abs(cwt_matrix) ** 2


def cwt_averaged_spectrum(
    data: ArrayLike,
    dt: float = 60.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute time-averaged CWT power spectrum.

    Returns freq and mean power (averaged over time).
    """
    freq, _, power = cwt_power(data, dt, **kwargs)
    return freq, np.mean(power, axis=1)
