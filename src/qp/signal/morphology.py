"""Waveform morphology analysis for QP wave events.

Characterises the shape of detected oscillation packets:
asymmetry (rise/fall), harmonic content, amplitude growth/decay,
frequency drift (chirp), and inter-cycle coherence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, hilbert, sosfiltfilt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bandpass(
    data: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    r"""Butterworth bandpass filter (zero-phase, forward-backward).

    Parameters
    ----------
    data : ndarray
        Input time series.
    low_hz, high_hz : float
        Passband edges in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (default 4).

    Returns
    -------
    ndarray
        Filtered signal, same length as input.
    """
    nyq = 0.5 * fs
    lo = max(low_hz / nyq, 1e-4)
    hi = min(high_hz / nyq, 0.999)
    if lo >= hi:
        return data.copy()
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, data)


# Keep alias for internal callers
_bandpass = bandpass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def band_envelope(
    data: ArrayLike,
    dt: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    r"""Amplitude envelope of a band-passed signal via Hilbert transform.

    Applies a 4th-order Butterworth band-pass filter then computes the
    instantaneous amplitude as ``|analytic_signal|``.

    Parameters
    ----------
    data : array_like
        Input time series.
    dt : float
        Sampling interval in seconds.
    low_hz, high_hz : float
        Band-pass filter edges in Hz.

    Returns
    -------
    envelope : ndarray
        Same length as ``data``.
    """
    x = np.asarray(data, dtype=float)
    filtered = _bandpass(x, low_hz, high_hz, 1.0 / dt)
    return np.abs(hilbert(filtered))


def instantaneous_frequency(
    data: ArrayLike,
    dt: float,
) -> np.ndarray:
    r"""Instantaneous frequency from the Hilbert analytic signal.

    .. math::
        f(t) = \frac{1}{2\pi}\frac{d\phi}{dt}

    where ``phi`` is the unwrapped phase of the analytic signal.

    Parameters
    ----------
    data : array_like
        Input signal (ideally already band-passed to the QP band of interest).
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    f_inst : ndarray
        Instantaneous frequency in Hz, same length as ``data``.
        Padded at the end with the last computed value.
    """
    x = np.asarray(data, dtype=float)
    analytic = hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    f_inst = np.diff(phase) / (2.0 * np.pi * dt)
    # Clip to positive frequencies (spurious sign flips at low SNR)
    f_inst = np.abs(f_inst)
    return np.concatenate([f_inst, [f_inst[-1]]])


def envelope_rise_fall(
    envelope: ArrayLike,
    dt: float,
) -> tuple[float, float, float] | None:
    r"""Rise and fall times of an amplitude envelope.

    Locates the peak of the envelope, then measures:

    * **rise time**: time from 10 % to 90 % of peak amplitude on the
      leading edge.
    * **fall time**: time from 90 % to 10 % on the trailing edge.

    Parameters
    ----------
    envelope : array_like
        Non-negative amplitude envelope.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    (rise_sec, fall_sec, rise_fall_ratio) : tuple of float, or None
        ``None`` if the envelope is too short or flat.
    ``rise_fall_ratio > 1`` means slow rise / fast fall.
    """
    env = np.asarray(envelope, dtype=float)
    n = len(env)
    if n < 4:
        return None
    peak_idx = int(np.argmax(env))
    env_max = env[peak_idx]
    env_min = env.min()
    span = env_max - env_min
    if span <= 0 or peak_idx == 0 or peak_idx == n - 1:
        return None

    lo_thresh = env_min + 0.10 * span
    hi_thresh = env_min + 0.90 * span

    # Rise: find 10 % → 90 % on the leading edge [0 .. peak_idx]
    lead = env[:peak_idx + 1]
    crossings_lo = np.where(lead >= lo_thresh)[0]
    crossings_hi = np.where(lead >= hi_thresh)[0]
    if len(crossings_lo) == 0 or len(crossings_hi) == 0:
        return None
    idx_10 = crossings_lo[0]
    idx_90 = crossings_hi[0]
    rise_sec = max((idx_90 - idx_10) * dt, dt)

    # Fall: find 90 % → 10 % on the trailing edge [peak_idx .. end]
    trail = env[peak_idx:]
    crossings_90t = np.where(trail <= hi_thresh)[0]
    crossings_10t = np.where(trail <= lo_thresh)[0]
    if len(crossings_90t) == 0:
        fall_sec = (len(trail) - 1) * dt
    elif len(crossings_10t) == 0:
        fall_sec = (len(trail) - crossings_90t[0]) * dt
    else:
        fall_sec = max((crossings_10t[0] - crossings_90t[0]) * dt, dt)

    ratio = rise_sec / fall_sec
    return float(rise_sec), float(fall_sec), float(ratio)


def harmonic_ratio(
    data: ArrayLike,
    dt: float,
    period_sec: float,
    window_bins: int = 2,
) -> float:
    r"""Ratio of power at the 2nd harmonic to the fundamental.

    A pure sine has ratio ≈ 0.  A sawtooth wave has ratio ≈ 0.25
    (power at ``2f`` is 1/4 of ``f``).

    Parameters
    ----------
    data : array_like
        Time series in the event window.
    dt : float
        Sampling interval in seconds.
    period_sec : float
        Nominal wave period in seconds.
    window_bins : int
        Number of FFT bins on each side of the target frequency to sum.

    Returns
    -------
    ratio : float
        ``P(2f) / P(f)`` — non-negative.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n < 4:
        return 0.0
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)
    if freqs[-1] <= 0:
        return 0.0

    f_fund = 1.0 / period_sec
    f_harm = 2.0 / period_sec

    def _band_power(f_target: float) -> float:
        if f_target <= 0 or f_target > freqs[-1]:
            return 0.0
        i0 = int(np.argmin(np.abs(freqs - f_target)))
        lo = max(0, i0 - window_bins)
        hi = min(len(spectrum), i0 + window_bins + 1)
        return float(spectrum[lo:hi].sum())

    p_fund = _band_power(f_fund)
    p_harm = _band_power(f_harm)
    if p_fund <= 0:
        return 0.0
    return float(p_harm / p_fund)


def amplitude_growth_rate(
    envelope: ArrayLike,
    dt: float,
    period_sec: float,
) -> float:
    r"""Per-oscillation amplitude growth rate in dB/period.

    Splits the envelope into cycles of ``period_sec``, computes the RMS
    amplitude per cycle, and fits a linear trend to ``log10(rms)`` vs
    cycle index.  Positive → growing; negative → decaying.

    Parameters
    ----------
    envelope : array_like
        Amplitude envelope (non-negative).
    dt : float
        Sampling interval in seconds.
    period_sec : float
        Wave period in seconds.

    Returns
    -------
    slope_db_per_period : float
        Linear slope in dB/period (``20 × slope_in_log10``).
    """
    env = np.asarray(envelope, dtype=float)
    spc = max(4, int(round(period_sec / dt)))
    n_cycles = len(env) // spc
    if n_cycles < 2:
        return 0.0

    chunks = env[:n_cycles * spc].reshape(n_cycles, spc)
    rms_all = np.sqrt(np.mean(chunks ** 2, axis=1))
    rms_vals = rms_all[rms_all > 0]

    if len(rms_vals) < 2:
        return 0.0

    cycles = np.arange(len(rms_vals), dtype=float)
    log_rms = np.log10(rms_vals)
    slope = float(np.polyfit(cycles, log_rms, 1)[0])
    return slope * 20.0  # convert log10/cycle → dB/cycle


def inter_cycle_coherence(
    data: ArrayLike,
    dt: float,
    period_sec: float,
) -> float:
    r"""Mean normalised dot-product between successive oscillation cycles.

    For a perfectly coherent wave every cycle is an exact copy of the
    last → value near 1.  For incoherent noise → value near 0.

    Parameters
    ----------
    data : array_like
        Time series in the event window.
    dt : float
        Sampling interval in seconds.
    period_sec : float
        Wave period in seconds (used for cycle segmentation).

    Returns
    -------
    coherence : float
        In [−1, 1]; typically [0, 1] for real Alfvén wave events.
    """
    x = np.asarray(data, dtype=float)
    spc = max(4, int(round(period_sec / dt)))
    n_cycles = len(x) // spc
    if n_cycles < 2:
        return 1.0

    chunks = x[:n_cycles * spc].reshape(n_cycles, spc)
    rms = np.sqrt(np.mean(chunks ** 2, axis=1, keepdims=True))
    valid = (rms > 0).ravel()
    if valid.sum() < 2:
        return 1.0

    normed = chunks[valid] / rms[valid]
    # Dot product between successive normalized cycles
    corrs = np.sum(normed[:-1] * normed[1:], axis=1) / normed.shape[1]
    return float(np.clip(np.mean(corrs), -1.0, 1.0))


def freq_drift_rate(
    data: ArrayLike,
    dt: float,
    low_hz: float,
    high_hz: float,
) -> float:
    r"""Linear frequency drift rate from instantaneous frequency.

    Band-passes the data, computes the instantaneous frequency via
    the Hilbert transform, and fits a linear trend ``f(t) = f_0 + α·t``.

    Parameters
    ----------
    data : array_like
        Time series in the event window.
    dt : float
        Sampling interval in seconds.
    low_hz, high_hz : float
        Band-pass edges in Hz (should match the QP band).

    Returns
    -------
    alpha : float
        Chirp rate in Hz/s.  Positive → upward frequency sweep;
        near zero → standing / non-dispersive.
    """
    x = np.asarray(data, dtype=float)
    if len(x) < 8:
        return 0.0
    filtered = _bandpass(x, low_hz, high_hz, 1.0 / dt)
    f_inst = instantaneous_frequency(filtered, dt)
    # Clip to band frequencies to reject transient edge artefacts
    f_inst = np.clip(f_inst, low_hz * 0.5, high_hz * 2.0)
    times = np.arange(len(f_inst)) * dt
    slope = float(np.polyfit(times, f_inst, 1)[0])
    return slope
