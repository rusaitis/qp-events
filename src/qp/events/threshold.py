r"""Statistical thresholding for QP wave-event detection.

This module implements the **two-stage gate** that the Phase 1 ridge
extractor consumes:

1. :func:`screen_segment_by_power_ratio` runs the FFT power ratio
   :math:`r_i = P(b_i)/P(\langle B_T\rangle_f)` from
   :mod:`qp.signal.power_ratio` and asks "is there *any* enhancement
   inside this band?". This is a **fast pass/fail** on a 36-h
   segment — it's used to skip 60-80 % of the mission cheaply before
   running the more expensive ridge extractor.

2. :func:`wavelet_sigma_mask` computes a robust per-row noise floor
   on the CWT power matrix using **median + MAD** of "background"
   period rows (those outside all QP bands), and returns a boolean
   mask of cells whose power exceeds ``median + n_sigma * 1.4826*MAD``.
   The factor 1.4826 turns MAD into a Gaussian-equivalent σ.

   Why MAD instead of the per-row mean+std: the very QP signals we
   want to detect would inflate the std and self-suppress; the
   median is unaffected by ≤50 % outliers. Restricting the noise
   estimate to *rows outside the QP bands* makes the gate even more
   robust.

Both functions are pure — they take arrays in, return masks/bools out,
and never touch I/O. The mission sweep in
:func:`scripts.sweep_events.process_segment` is responsible for
plumbing them together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qp.events.bands import (
    QP_BANDS,
    REJECT_BAND_HF,
    REJECT_BAND_LF,
    Band,
    freq_to_period,
    get_band,
)
from qp.signal.pipeline import SpectralResult


# ----------------------------------------------------------------------
# Stage 1: FFT power ratio screen
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FFTScreenResult:
    """Outcome of an FFT band screen."""

    band: str
    triggered: bool
    max_ratio: float
    peak_freq_hz: float
    peak_period_min: float


def screen_segment_by_power_ratio(
    freq: ArrayLike,
    power_ratio: ArrayLike,
    band: str | Band,
    ratio_threshold: float = 5.0,
) -> FFTScreenResult:
    r"""Pass/fail screen based on the Welch power ratio inside one band.

    Parameters
    ----------
    freq : array_like, shape (n_freq,)
        Frequency axis in Hz from :func:`qp.signal.fft.welch_psd`.
    power_ratio : array_like, shape (n_freq,)
        Power ratio :math:`r_i = P(b_i)/P(\langle B_T\rangle_f)`.
    band : str or Band
        Which QP band to check.
    ratio_threshold : float
        Trigger if any in-band frequency has ratio > threshold.
        The default of 5 is a permissive screen — Phase 2.4 calibrates
        the final value against a synthetic injection campaign.

    Returns
    -------
    FFTScreenResult
    """
    freq = np.asarray(freq, dtype=float)
    power_ratio = np.asarray(power_ratio, dtype=float)
    band_obj = get_band(band)

    in_band = (
        (freq >= band_obj.freq_min_hz)
        & (freq < band_obj.freq_max_hz)
    )
    if not in_band.any():
        return FFTScreenResult(
            band=band_obj.name,
            triggered=False,
            max_ratio=0.0,
            peak_freq_hz=0.0,
            peak_period_min=0.0,
        )

    sub_ratio = power_ratio[in_band]
    sub_freq = freq[in_band]
    idx = int(sub_ratio.argmax())
    max_ratio = float(sub_ratio[idx])
    peak_freq = float(sub_freq[idx])
    peak_period_min = (1.0 / peak_freq) / 60.0 if peak_freq > 0 else 0.0

    return FFTScreenResult(
        band=band_obj.name,
        triggered=max_ratio > ratio_threshold,
        max_ratio=max_ratio,
        peak_freq_hz=peak_freq,
        peak_period_min=peak_period_min,
    )


def screen_spectral_result(
    spectral_result: SpectralResult,
    band: str | Band,
    ratio_threshold: float = 5.0,
) -> FFTScreenResult:
    r"""Convenience wrapper that pulls ``freq`` and ``power_ratio`` out
    of a :class:`qp.signal.pipeline.SpectralResult`."""
    return screen_segment_by_power_ratio(
        spectral_result.freq,
        spectral_result.power_ratio,
        band=band,
        ratio_threshold=ratio_threshold,
    )


# ----------------------------------------------------------------------
# Stage 2: Wavelet MAD-based σ mask
# ----------------------------------------------------------------------

#: Conversion factor: σ = 1.4826 * MAD for Gaussian data.
MAD_TO_SIGMA: float = 1.4826


def _background_row_indices(cwt_freq: NDArray[np.floating]) -> NDArray[np.intp]:
    """Indices of CWT rows that fall *outside* every QP band.

    These rows feed the noise model. Rows in the rejection guard bands
    (above 12 h or below 10 min) are also excluded since their power
    is dominated by edge effects and aliasing.
    """
    periods_sec = freq_to_period(cwt_freq)
    keep = np.ones_like(cwt_freq, dtype=bool)
    # Exclude QP bands and the super_qp120 gap ([QP120.period_max, 180) min).
    # Rows inside super_qp120 are signal-bearing in the band-agnostic
    # pipeline (scenarios inject events at 170-180 min). Including them
    # in the noise model inflates the local MAD by the signal itself:
    # tier4_super_qp120 fills ~50 % of the time axis with genuine wave
    # power, driving the threshold above the signal max. Excluding
    # super_qp120 lets the threshold at 170-180 min be extrapolated
    # from genuinely-noise rows just above 180 min.
    for band in QP_BANDS.values():
        in_band = (
            (periods_sec >= band.period_min_sec)
            & (periods_sec < band.period_max_sec)
        )
        keep &= ~in_band
    qp120 = QP_BANDS["QP120"]
    super_qp120_mask = (
        (periods_sec >= qp120.period_max_sec)
        & (periods_sec < 180.0 * 60.0)
    )
    keep &= ~super_qp120_mask
    # exclude rejection guards
    keep &= periods_sec >= REJECT_BAND_HF.period_max_sec
    keep &= periods_sec < REJECT_BAND_LF.period_min_sec
    return np.flatnonzero(keep)


def wavelet_sigma_mask(
    cwt_power: ArrayLike,
    cwt_freq: ArrayLike,
    n_sigma: float = 3.0,
) -> NDArray[np.bool_]:
    r"""Boolean mask of CWT cells exceeding the per-row σ threshold.

    Noise model
    -----------
    Per-row median and MAD are computed only on rows that fall
    **outside** every QP band (and outside the high/low rejection
    guards). For rows *inside* a QP band — where the signal lives —
    the noise threshold is interpolated from the surrounding
    background rows in log-period space. This decouples the noise
    estimate from the signal we're trying to detect: a strong QP60
    packet would otherwise inflate its own row's MAD and gate itself
    out.

    Parameters
    ----------
    cwt_power : array_like, shape (n_freq, n_time)
        CWT power (``|cwt|`` or ``|cwt|^2``).
    cwt_freq : array_like, shape (n_freq,)
        Wavelet frequency axis in Hz (assumed monotonic).
    n_sigma : float, default 3.0
        Detection threshold in robust σ units (1 σ ≈ 1.4826 * MAD).

    Returns
    -------
    mask : ndarray of bool, shape (n_freq, n_time)
        ``True`` where ``cwt_power`` exceeds its row's threshold.
    """
    cwt_power = np.asarray(cwt_power, dtype=float)
    cwt_freq = np.asarray(cwt_freq, dtype=float)

    bg_rows = _background_row_indices(cwt_freq)
    if bg_rows.size == 0:
        # Degenerate axis — fall back to a single global threshold.
        med = float(np.median(cwt_power))
        mad = float(np.median(np.abs(cwt_power - med)))
        thr = med + n_sigma * MAD_TO_SIGMA * mad
        return cwt_power > thr

    # Per-row median & MAD on the background rows only — these rows
    # are uncontaminated by QP signal by construction.
    bg_medians = np.median(cwt_power[bg_rows], axis=1)
    bg_mads = np.median(
        np.abs(cwt_power[bg_rows] - bg_medians[:, None]),
        axis=1,
    )
    bg_thr = bg_medians + n_sigma * MAD_TO_SIGMA * bg_mads  # (n_bg,)

    # Interpolate the threshold to every row of the CWT in log-period
    # space. Periods are monotonically *decreasing* with frequency,
    # so interp wants strictly increasing x — work in log10(period).
    periods_sec = freq_to_period(cwt_freq)
    log_p_bg = np.log10(periods_sec[bg_rows])
    log_p_all = np.log10(periods_sec)

    # np.interp needs sorted xp; sort by log_p_bg ascending
    order = np.argsort(log_p_bg)
    log_p_bg_sorted = log_p_bg[order]
    bg_thr_sorted = bg_thr[order]
    row_thr = np.interp(
        log_p_all,
        log_p_bg_sorted,
        bg_thr_sorted,
        left=bg_thr_sorted[0],
        right=bg_thr_sorted[-1],
    )

    return cwt_power > row_thr[:, None]


# ----------------------------------------------------------------------
# Stage 3: Combined gate (FFT screen + σ mask + COI + duration)
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateConfig:
    r"""Detection gate configuration.

    The pipeline thresholds the joint transverse CWT power
    (average of both $b_{\perp 1}$ and $b_{\perp 2}$) using a robust
    MAD-based σ-mask, extracts connected ridges per QP band, then
    applies physical post-filters (min oscillations, transverse ratio,
    spectral concentration, same-band dedup).
    """

    # σ-mask threshold on joint transverse CWT power.
    # 3.5σ on averaged power ≈ 5σ on a single component (√2 noise reduction).
    n_sigma: float = 3.5
    min_duration_hours: float = 2.0
    min_pixels: int = 80
    # Quasi-periodic identification convention: ≥3 full cycles.
    min_oscillations: float = 3.0
    coi_factor: float = 1.0
    # Alfvén waves are transverse: reject if parallel CWT power exceeds
    # this fraction of transverse in-band power.
    transverse_ratio: float = 0.5
    # Legacy cross-band broadband rejection. Replaced by the ridge-local
    # FWHM check (max_fwhm_log_period) in the band-agnostic pipeline.
    spectral_concentration: float | None = None
    dedup_window_sec: float = 10800.0
    # Wavelet cross-coherence between b_perp1 and b_perp2.
    min_coherence: float | None = 0.9
    # Max FWHM in log10(period) of the ridge's spectral profile,
    # measured on a ±half-octave window around the peak. Morlet's
    # fundamental FWHM is ~0.035; broadband transients >>0.3.
    max_fwhm_log_period: float | None = 0.3


DEFAULT_GATE: GateConfig = GateConfig()
