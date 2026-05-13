r"""Morlet continuous wavelet transform for quasi-periodic event detection.

The wavelet implemented here is the standard simplified Morlet of
Torrence & Compo (1998, hereafter T&C98) Eq. (1):

.. math::

    \psi_0(\eta) = \pi^{-1/4} \exp(i \omega_0 \eta) \exp(-\eta^2/2),

with :math:`\eta = t/s` the dimensionless time and :math:`s` the scale
parameter in samples. The strict admissibility correction
:math:`-\exp(-\omega_0^2/2)` is omitted — at the repository default
:math:`\omega_0 = 10` it is :math:`\sim 2\times10^{-22}` and entirely
negligible.

Relation to the paper's Eq. (5) — which writes the Morlet as
:math:`\psi(t,\omega) = c_\omega \exp(-t^2/\sigma^2) \exp(i\omega t)` —
is :math:`\sigma = s\sqrt{2}`, :math:`\omega = \omega_0/s`, and
:math:`c_\omega = \pi^{-1/4}`. Substituting recovers the form above.

Implements CWT directly since ``scipy.signal.cwt`` was removed in
SciPy 1.15.

References
----------
Torrence, C. & Compo, G. P. (1998), "A Practical Guide to Wavelet
Analysis", *Bull. Amer. Meteor. Soc.* **79**, 61–78.
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
    r"""Compute continuous wavelet transform using the Morlet wavelet.

    Parameters
    ----------
    data : array_like
        Input time series.
    dt : float
        Sampling interval in seconds.
    omega0 : float, default 10.0
        Central frequency parameter of the Morlet wavelet
        (T&C98 :math:`\omega_0`). Sets the time-frequency resolution
        trade-off: higher :math:`\omega_0` gives narrower frequency
        bandwidth :math:`\Delta f / f \approx 1/\omega_0` and wider
        temporal extent :math:`\Delta t \propto \omega_0/f`. The
        repo default :math:`\omega_0 = 10` is above the T&C98 §3a
        choice of 6 — it is chosen to resolve the QP30/QP60/QP120
        peaks that sit at factor-2 period separations, where the
        :math:`\omega_0 = 6` bandwidth of ~17 % overlaps adjacent
        QP bands.
    freq_min : float, optional
        Minimum frequency in Hz. Default: 1/(4 hours).
    freq_max : float, optional
        Maximum frequency in Hz. Default: Nyquist/2.
    n_freqs : int
        Number of linearly-spaced frequency bins. The grid is
        deliberately linear (constant :math:`\Delta f`) rather than
        the conventional log-scale grid of T&C98 §3h — this matches
        the FFT comparison axis used elsewhere in the pipeline.
        Downstream consumers (ridge extractor, parabolic peak
        interpolator in :mod:`qp.events.ridge`) handle the resulting
        non-uniform log-period spacing correctly.

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

    # Scale parameter: width in samples for each frequency.
    # The exact Morlet Fourier-period-to-scale relation (T&C98 Table 1) is
    #     f = (omega0 + sqrt(2 + omega0**2)) / (4 pi s dt),
    # i.e. s = (omega0 + sqrt(2 + omega0**2)) / (4 pi f dt). Inverting gives
    # the form used here. The often-quoted simplified inverse
    # s_simple = omega0 / (2 pi f dt) differs from the exact value by a
    # factor (omega0 + sqrt(2 + omega0**2)) / (2 omega0) — at omega0 = 10
    # this is 1.005, a 0.5 % bias in reported peak periods.
    scales = (omega0 + np.sqrt(2.0 + omega0**2)) / (4.0 * np.pi * freq * dt)

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
