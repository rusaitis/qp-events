r"""Matched-filter detection for quasi-periodic wave packets.

The matched filter is the optimal linear detector in stationary Gaussian
noise (Neyman-Pearson lemma). Given a template $h(t)$ (sine modulated by
a Gaussian envelope) and data $d(t)$, the SNR time series is:

$$\mathrm{SNR}(t) = \frac{(d \star h)(t)}{\sigma_n \sqrt{h \cdot h}}$$

where $\sigma_n$ is the noise standard deviation estimated from data
outside the QP bands.

Pre-whitening (dividing by the power-law background in the frequency
domain) makes the noise approximately white, satisfying the matched
filter's optimality assumption.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _gaussian_windowed_sine(
    n_samples: int,
    dt: float,
    period: float,
    envelope_width: float | None = None,
) -> NDArray[np.floating]:
    r"""Construct a sine $\times$ Gaussian template centred at t=0.

    Parameters
    ----------
    n_samples : int
        Total number of samples (odd is best for symmetry).
    dt : float
        Sampling interval in seconds.
    period : float
        Oscillation period in seconds.
    envelope_width : float, optional
        Gaussian $\sigma$ in seconds. Default: ``period * 1.5``
        (covers ~3 oscillations within the FWHM).
    """
    if envelope_width is None:
        envelope_width = period * 1.5
    t = (np.arange(n_samples) - n_samples // 2) * dt
    envelope = np.exp(-0.5 * (t / envelope_width) ** 2)
    template = np.sin(2 * np.pi * t / period) * envelope
    # Normalize to unit energy
    norm = np.sqrt(np.sum(template**2))
    if norm > 0:
        template /= norm
    return template


def prewhiten(
    data: ArrayLike,
    dt: float,
    background: ArrayLike,
    freq: ArrayLike,
) -> NDArray[np.floating]:
    r"""Divide by the power-law background in the frequency domain.

    Transforms the coloured noise into approximately white noise so
    the matched filter's Gaussian assumption holds.

    Parameters
    ----------
    data : array_like
        Input time series.
    dt : float
        Sampling interval in seconds.
    background : array_like
        Power-law background estimate (one-sided PSD, same freq grid
        as from :func:`qp.signal.fft.welch_psd`).
    freq : array_like
        Frequency axis of the background.

    Returns
    -------
    whitened : ndarray
        Pre-whitened time series (same length as ``data``).
    """
    data = np.asarray(data, dtype=float)
    background = np.asarray(background, dtype=float)
    freq = np.asarray(freq, dtype=float)

    n = len(data)
    fft_data = np.fft.rfft(data)
    fft_freq = np.fft.rfftfreq(n, d=dt)

    # Interpolate the one-sided background to the FFT frequency grid
    bg_interp = np.interp(fft_freq, freq, background, left=background[0],
                          right=background[-1])
    bg_interp = np.maximum(bg_interp, 1e-30)

    # Divide amplitude by sqrt(background) to flatten the noise floor
    fft_whitened = fft_data / np.sqrt(bg_interp)
    whitened = np.fft.irfft(fft_whitened, n=n)
    return whitened


def matched_filter_snr(
    data: ArrayLike,
    dt: float = 60.0,
    period: float = 3600.0,
    envelope_width: float | None = None,
    background: ArrayLike | None = None,
    freq: ArrayLike | None = None,
) -> NDArray[np.floating]:
    r"""Compute matched-filter SNR time series for a given period.

    Parameters
    ----------
    data : array_like
        Input time series (MFA component).
    dt : float
        Sampling interval in seconds.
    period : float
        Target oscillation period in seconds.
    envelope_width : float, optional
        Gaussian envelope width. Default: ``1.5 * period``.
    background, freq : array_like, optional
        If provided, pre-whiten the data before filtering.

    Returns
    -------
    snr : ndarray
        SNR time series, same length as ``data``.
    """
    data = np.asarray(data, dtype=float)

    # Pre-whiten if background is available
    if background is not None and freq is not None:
        data = prewhiten(data, dt, background, freq)

    # Build the template — length = 6 envelope widths (captures >99.7%)
    if envelope_width is None:
        envelope_width = period * 1.5
    template_len = int(6 * envelope_width / dt)
    template_len = min(template_len, len(data))
    if template_len < 3:
        return np.zeros(len(data))
    # Make it odd for symmetry
    if template_len % 2 == 0:
        template_len += 1

    template = _gaussian_windowed_sine(template_len, dt, period,
                                        envelope_width)

    # Correlate via FFT (scipy.signal.fftconvolve with reversed template)
    # matched filter output = cross-correlation with template
    from scipy.signal import fftconvolve
    corr = fftconvolve(data, template[::-1], mode="same")

    # Noise estimate: MAD-based robust σ of the data
    med = np.median(data)
    sigma_n = 1.4826 * np.median(np.abs(data - med))
    if sigma_n <= 0:
        sigma_n = np.std(data)
    if sigma_n <= 0:
        return np.zeros(len(data))

    snr = np.abs(corr) / sigma_n
    return snr


def matched_filter_peak_snr(
    data: ArrayLike,
    dt: float,
    period: float,
    t_peak_idx: int,
    window_minutes: float = 30.0,
    background: ArrayLike | None = None,
    freq: ArrayLike | None = None,
) -> float:
    r"""Peak matched-filter SNR near a candidate event's peak time.

    Parameters
    ----------
    data : array_like
        Full segment time series.
    dt : float
        Sampling interval in seconds.
    period : float
        Event's peak period in seconds.
    t_peak_idx : int
        Sample index of the event peak.
    window_minutes : float
        Half-window around peak to search for max SNR.
    background, freq : array_like, optional
        For pre-whitening.

    Returns
    -------
    peak_snr : float
    """
    data = np.asarray(data, dtype=float)
    snr = matched_filter_snr(data, dt, period,
                              background=background, freq=freq)
    half = int(window_minutes * 60 / dt)
    lo = max(0, t_peak_idx - half)
    hi = min(len(snr), t_peak_idx + half + 1)
    if lo >= hi:
        return 0.0
    return float(np.max(snr[lo:hi]))
