"""Wave event detection from CWT power.

Two flavours:

1. **Legacy single-band detector** (``detect_wave_packets``,
   ``compute_event_measure``, ``collect_wave_events``) — peak-finding
   on a normalized CWT event measure inside a fixed period band. Kept
   for back-compat with the existing tests and the QP60-only Fig 9
   pipeline.

2. **Multi-band detector** (``detect_wave_packets_multi``) — runs
   :func:`qp.events.ridge.extract_ridges` over each canonical QP band
   and turns each ridge into a :class:`WavePacketPeak`. Optionally
   uses an externally supplied σ-mask from
   :mod:`qp.events.threshold`. This is the path used by the
   mission-wide sweep in Phase 3.

The legacy API stays untouched so existing scripts and tests continue
to work; new code should call ``detect_wave_packets_multi``.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

from qp.events.bands import QP_BAND_NAMES, Band
from qp.events.catalog import WaveEvent, WavePacketPeak
from qp.events.ridge import Ridge, extract_ridges
from qp.signal.wavelet import morlet_cwt

if TYPE_CHECKING:
    from qp.events.threshold_diag import BGArchive

#: Round-8 threshold-method dispatch labels.
#:
#: ``tc_chi2`` (default) — Torrence & Compo (1998) §4 AR(1)+χ²(2)
#: per-pixel test at the round-8 Bonferroni FWER. Recovers ~3× more
#: events than ``mad_row`` at the 0.1–0.3 nT detection edge at matched
#: bg-rate; methods converge above ~1 nT.
#:
#: ``mad_row`` — per-row MAD on background period rows, log-period
#: interpolated across QP bands. Legacy gate; kept for parity with the
#: pre-Phase-3 catalogue.
#:
#: ``fdr_chi2``, ``pooled`` — diagnostic-only alternatives reachable
#: from the sweep script for sensitivity studies; not recommended as a
#: production default.
ThresholdMethod = Literal["mad_row", "tc_chi2", "fdr_chi2", "pooled"]

# Imported lazily inside functions to avoid an import cycle:
#   qp.events.threshold imports qp.signal.pipeline.SpectralResult
#   qp.signal.pipeline currently does not import qp.events, so we're
#   safe to import threshold at module top, but we keep it local for
#   the functions that use it for clarity.


def _cwt_event_measure(
    cwt_power: np.ndarray,
    freq: np.ndarray,
    period_band: tuple[float, float],
    n_period_bins: int,
) -> np.ndarray:
    """Compute event measure by extracting CWT power at target periods.

    Returns the L2 norm across period bins at each time step.
    """
    target_freqs = 1.0 / np.linspace(period_band[0], period_band[1], n_period_bins)
    idx = np.array([np.argmin(np.abs(freq - f)) for f in target_freqs])
    return np.linalg.norm(cwt_power[idx, :], axis=0)


# ----------------------------------------------------------------------
# Legacy single-band detector (back-compat)
# ----------------------------------------------------------------------


def detect_wave_packets(
    data: ArrayLike,
    times: list[datetime.datetime],
    dt: float = 60.0,
    period_band: tuple[float, float] = (50 * 60, 70 * 60),
    n_period_bins: int = 5,
    min_prominence: float = 0.05,
    min_peak_distance: int = 60,
    min_peak_width: int = 100,
    min_duration_hours: float = 2.0,
    dedup_window_hours: float = 3.0,
    previous_peak_time: datetime.datetime | None = None,
) -> list[WavePacketPeak]:
    """Detect wave packets in a time series using CWT (legacy QP60 path).

    Parameters
    ----------
    data : array_like
        Field component time series (e.g., b_perp1).
    times : list of datetime
        Corresponding timestamps.
    dt : float
        Sampling interval in seconds.
    period_band : tuple
        (min_period, max_period) in seconds to target.
        Default (3000, 4200) = 50-70 min for QP60.
    n_period_bins : int
        Number of period bins within the band.
    min_prominence : float
        Minimum peak prominence in normalized CWT power.
    min_peak_distance : int
        Minimum separation between peaks in samples.
    min_peak_width : int
        Minimum peak width in samples.
    min_duration_hours : float
        Minimum wave packet duration to accept.
    dedup_window_hours : float
        Suppress duplicate detections within this window.
    previous_peak_time : datetime, optional
        Last peak from previous segment (for cross-segment dedup).

    Returns
    -------
    packets : list of WavePacketPeak
    """
    data = np.asarray(data, dtype=float)
    N = len(data)
    times = list(times)

    # Compute CWT
    freq, _, cwt_matrix = morlet_cwt(data, dt=dt, n_freqs=300)

    # Normalize CWT power
    cwt_power = np.abs(cwt_matrix)
    max_power = np.max(cwt_power)
    if max_power > 0:
        cwt_power /= max_power

    event_measure = _cwt_event_measure(cwt_power, freq, period_band, n_period_bins)

    # Find peaks
    peaks, properties = find_peaks(
        event_measure,
        height=min_prominence,
        distance=min_peak_distance,
        prominence=min_prominence,
        width=min_peak_width,
    )

    # Build WavePacketPeak objects
    packets: list[WavePacketPeak] = []
    dedup_sec = dedup_window_hours * 3600

    for peak, prom, l_ips, r_ips in zip(
        peaks,
        properties["prominences"],
        properties["left_ips"],
        properties["right_ips"],
    ):
        i_left = int(np.clip(np.round(l_ips), 0, N - 1))
        i_right = int(np.clip(np.round(r_ips), 0, N - 1))

        t_peak = times[peak]
        t_from = times[i_left]
        t_to = times[i_right]
        duration_h = (t_to - t_from).total_seconds() / 3600

        if duration_h < min_duration_hours:
            continue

        # De-duplicate against previous detections
        if previous_peak_time is not None:
            sep = abs((t_peak - previous_peak_time).total_seconds())
            if sep < dedup_sec:
                continue
        if packets:
            sep = abs((t_peak - packets[-1].peak_time).total_seconds())
            if sep < dedup_sec:
                continue

        packets.append(
            WavePacketPeak(
                peak_time=t_peak,
                prominence=float(prom),
                date_from=t_from,
                date_to=t_to,
            )
        )

    return packets


def compute_event_measure(
    data: ArrayLike,
    dt: float = 60.0,
    period_band: tuple[float, float] = (50 * 60, 70 * 60),
    n_period_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the CWT-based event measure for a time series.

    Returns (time_seconds, event_measure) for plotting/diagnostics.
    """
    data = np.asarray(data, dtype=float)
    freq, time_sec, cwt_matrix = morlet_cwt(data, dt=dt, n_freqs=300)

    cwt_power = np.abs(cwt_matrix)
    max_power = np.max(cwt_power)
    if max_power > 0:
        cwt_power /= max_power

    event_measure = _cwt_event_measure(cwt_power, freq, period_band, n_period_bins)
    return time_sec, event_measure


def collect_wave_events(
    field_data: ArrayLike,
    time_unix: ArrayLike,
    dt: float = 60.0,
    period_band: tuple[float, float] = (50 * 60, 70 * 60),
    min_snr: float = 3.0,
    min_duration_hours: float = 4.0,
    coords_krtp: np.ndarray | None = None,
    local_time: np.ndarray | None = None,
    sls5_phases: dict[str, np.ndarray] | None = None,
) -> list[WaveEvent]:
    r"""Detect wave events with full metadata (legacy single-band).

    See :func:`detect_wave_packets_multi` for the multi-band Phase 1+
    replacement.
    """
    field_data = np.asarray(field_data, dtype=float)
    time_unix = np.asarray(time_unix, dtype=float)

    # Build datetime list from POSIX timestamps
    _epoch = datetime.datetime(1970, 1, 1)
    times = [_epoch + datetime.timedelta(seconds=float(ts)) for ts in time_unix]

    # Detect packets using CWT
    packets = detect_wave_packets(
        field_data,
        times,
        dt=dt,
        period_band=period_band,
        min_prominence=min_snr,
        min_duration_hours=min_duration_hours,
    )

    # Enrich with metadata
    events: list[WaveEvent] = []
    for pkt in packets:
        peak_idx = _nearest_index(time_unix, _dt_to_unix(pkt.peak_time))
        from_idx = _nearest_index(time_unix, _dt_to_unix(pkt.date_from))
        to_idx = _nearest_index(time_unix, _dt_to_unix(pkt.date_to))

        # Amplitude: max |field| in the packet window
        amplitude = float(np.max(np.abs(field_data[from_idx : to_idx + 1])))

        # Coordinates at peak time
        r_dist = None
        mag_lat = None
        coord_krtp_val = None
        if coords_krtp is not None:
            r_dist = float(coords_krtp[peak_idx, 0])
            theta = float(coords_krtp[peak_idx, 1])
            mag_lat = 90.0 - np.degrees(theta)
            r, th, ph = (float(v) for v in coords_krtp[peak_idx])
            coord_krtp_val = (r, th, ph)

        lt = None
        if local_time is not None:
            lt = float(local_time[peak_idx])

        events.append(
            WaveEvent(
                date_from=pkt.date_from,
                date_to=pkt.date_to,
                amplitude=amplitude,
                snr=pkt.prominence,
                local_time=lt,
                mag_lat=mag_lat,
                r_distance=r_dist,
                coord_krtp=coord_krtp_val,
            )
        )

    return events


# ----------------------------------------------------------------------
# Multi-band detector (Phase 1+)
# ----------------------------------------------------------------------


def detect_wave_packets_multi(
    data: ArrayLike,
    times: list[datetime.datetime] | NDArray[np.floating],
    dt: float = 60.0,
    bands: Iterable[str | Band] = QP_BAND_NAMES,
    *,
    cwt_freq: ArrayLike | None = None,
    cwt_power: ArrayLike | None = None,
    threshold_mask: ArrayLike | None = None,
    min_duration_hours: float = 2.0,
    min_pixels: int = 50,
    coi_factor: float = 1.0,
    n_freqs: int = 300,
) -> list[WavePacketPeak]:
    r"""Multi-band wave-packet detection from a CWT scalogram.

    Pipeline
    --------
    1. Compute the CWT once (or accept a precomputed scalogram so the
       caller can reuse it across components).
    2. For each requested QP band, call
       :func:`qp.events.ridge.extract_ridges` to find connected blobs
       above the supplied threshold mask.
    3. Convert each ridge to a :class:`WavePacketPeak` with band label
       and peak period populated.

    Parameters
    ----------
    data : array_like
        Time-series of one MFA component (typically ``b_perp1``).
    times : list[datetime] or float ndarray
        Sample timestamps. If a numpy array is passed it is assumed to
        be POSIX seconds.
    dt : float
        Sampling interval in seconds (default 60 s).
    bands : iterable of str or Band
        Which QP bands to scan. Default scans all of QP30/QP60/QP120.
    cwt_freq, cwt_power : array_like, optional
        Precomputed CWT frequency axis and ``|cwt|`` power matrix. If
        either is None, the CWT is computed internally with
        ``omega0=10`` and ``n_freqs`` rows.
    threshold_mask : array_like, optional
        Boolean mask the same shape as ``cwt_power``. ``True`` means
        the cell is above the σ-threshold. If None, a fall-back
        ``cwt_power.max() / 4`` mask is used (suitable for tests, but
        the production sweep should always pass an explicit mask from
        :func:`qp.events.threshold.wavelet_sigma_mask`).
    min_duration_hours : float
        Reject ridges shorter than this in time.
    min_pixels : int
        Reject blobs with fewer than this many pixels (denoising).
    coi_factor : float
        Cone-of-influence factor passed to the ridge extractor.
    n_freqs : int
        Number of CWT rows when computing internally.

    Returns
    -------
    list[WavePacketPeak]
        Sorted by peak time. Each carries ``band`` and ``period_sec``.
    """
    data = np.asarray(data, dtype=float)
    n_time = data.size

    # Build a datetime list — accept POSIX float arrays for convenience
    if isinstance(times, np.ndarray) and times.dtype.kind in ("f", "i"):
        _epoch = datetime.datetime(1970, 1, 1)
        times_list: list[datetime.datetime] = [
            _epoch + datetime.timedelta(seconds=float(ts)) for ts in times
        ]
    else:
        times_list = list(times)
    if len(times_list) != n_time:
        raise ValueError(f"len(times)={len(times_list)} != len(data)={n_time}")

    if cwt_freq is None or cwt_power is None:
        freq, _, cwt_matrix = morlet_cwt(data, dt=dt, n_freqs=n_freqs)
        cwt_power = np.abs(cwt_matrix)
    else:
        freq = np.asarray(cwt_freq, dtype=float)
        cwt_power = np.asarray(cwt_power, dtype=float)
        if cwt_power.shape != (freq.size, n_time):
            raise ValueError(
                f"cwt_power shape {cwt_power.shape} does not match "
                f"(n_freq={freq.size}, n_time={n_time})"
            )

    min_duration_sec = min_duration_hours * 3600.0

    packets: list[WavePacketPeak] = []
    for b in bands:
        ridges = extract_ridges(
            cwt_power,
            freq,
            band=b,
            threshold_mask=threshold_mask,
            dt=dt,
            min_duration_sec=min_duration_sec,
            min_pixels=min_pixels,
            coi_factor=coi_factor,
        )
        for ridge in ridges:
            packets.append(_ridge_to_packet(ridge, times_list, n_time))

    packets.sort(key=lambda p: p.peak_time)
    return packets


def _ridge_to_packet(
    ridge: Ridge,
    times: list[datetime.datetime],
    n_time: int,
) -> WavePacketPeak:
    """Convert a :class:`Ridge` to a :class:`WavePacketPeak`."""
    t_start = times[max(0, ridge.t_start_idx)]
    t_end = times[min(n_time - 1, ridge.t_end_idx)]
    t_peak = times[ridge.peak_time_idx]
    return WavePacketPeak(
        peak_time=t_peak,
        prominence=float(ridge.peak_power),
        date_from=t_start,
        date_to=t_end,
        band=ridge.band,
        period_sec=float(ridge.peak_period_sec),
        period_fwhm_sec=float(ridge.period_fwhm_sec),
    )


# ----------------------------------------------------------------------
# Phase 2: full gate combining FFT screen + σ mask + ridge extraction
# ----------------------------------------------------------------------


def detect_with_gate(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
    times: list[datetime.datetime] | NDArray[np.floating],
    *,
    dt: float = 60.0,
    bands: Iterable[str | Band] = QP_BAND_NAMES,
    spectral_result_perp1=None,
    spectral_result_perp2=None,
    gate=None,
    cwt_n_freqs: int = 300,
) -> list[WavePacketPeak]:
    r"""Run the full Phase 2 detection gate on a single MFA segment.

    Pipeline
    --------
    1. For each band, run :func:`screen_segment_by_power_ratio` on
       both ``b_perp1`` and ``b_perp2`` Welch PSDs (caller may pass
       precomputed :class:`SpectralResult` objects to skip the
       analyze_segment step). Drop the band if neither component
       triggers.
    2. Compute the CWT of ``b_perp1`` (the dominant transverse
       component in the published paper).
    3. Build a :func:`wavelet_sigma_mask` from that CWT.
    4. Hand the (CWT, mask) pair to :func:`detect_wave_packets_multi`,
       restricted to the bands that survived step 1.

    Returns the list of :class:`WavePacketPeak` accepted by every
    stage. Polarization, coordinate, and PPO enrichment is the
    sweep script's job (see ``scripts/sweep_events_round8.py``).
    """
    # Lazy imports to keep module-level deps minimal
    from qp.events.threshold import (
        DEFAULT_GATE,
        screen_spectral_result,
        wavelet_sigma_mask,
    )
    from qp.signal.pipeline import analyze_segment

    if gate is None:
        gate = DEFAULT_GATE

    b_perp1 = np.asarray(b_perp1, dtype=float)
    b_perp2 = np.asarray(b_perp2, dtype=float)

    # Stage 1: Welch PSDs (compute lazily if not provided).
    if spectral_result_perp1 is None:
        spectral_result_perp1 = analyze_segment(
            b_perp1,
            dt=dt,
            detrend_window_sec=60.0,  # tiny — segments are pre-detrended
            welch_nperseg=12 * 60,
            welch_noverlap=6 * 60,
        )
    if spectral_result_perp2 is None:
        spectral_result_perp2 = analyze_segment(
            b_perp2,
            dt=dt,
            detrend_window_sec=60.0,
            welch_nperseg=12 * 60,
            welch_noverlap=6 * 60,
        )

    if gate.enable_fft_screen:
        triggered_bands: list[str | Band] = []
        for b in bands:
            s1 = screen_spectral_result(
                spectral_result_perp1,
                b,
                ratio_threshold=gate.fft_ratio_threshold,
            )
            s2 = screen_spectral_result(
                spectral_result_perp2,
                b,
                ratio_threshold=gate.fft_ratio_threshold,
            )
            if s1.triggered or s2.triggered:
                triggered_bands.append(b)
        if not triggered_bands:
            return []
    else:
        triggered_bands = list(bands)

    # Stage 2: CWT on both transverse components and combine via the
    # coincidence rule from Phase 6.3 (`require_both_perp`).
    #
    # We CWT both perp components and require σ-mask agreement in the
    # same (period, time) cell. This eliminates compressional or
    # single-axis contamination — a real Alfvén wave packet will fire
    # both transverse components together.
    freq, _, cwt_matrix1 = morlet_cwt(b_perp1, dt=dt, n_freqs=cwt_n_freqs)
    cwt_power1 = np.abs(cwt_matrix1)
    if gate.require_both_perp:
        _, _, cwt_matrix2 = morlet_cwt(b_perp2, dt=dt, n_freqs=cwt_n_freqs)
        cwt_power2 = np.abs(cwt_matrix2)
        mask1 = wavelet_sigma_mask(cwt_power1, freq, n_sigma=gate.n_sigma)
        mask2 = wavelet_sigma_mask(cwt_power2, freq, n_sigma=gate.n_sigma)
        mask = mask1 & mask2
        cwt_power = (cwt_power1 + cwt_power2) / 2.0
    else:
        cwt_power = cwt_power1
        mask = wavelet_sigma_mask(cwt_power, freq, n_sigma=gate.n_sigma)

    # Stage 4: ridge extraction on triggered bands only
    packets = detect_wave_packets_multi(
        b_perp1,
        times,
        dt=dt,
        bands=triggered_bands,
        cwt_freq=freq,
        cwt_power=cwt_power,
        threshold_mask=mask,
        min_duration_hours=gate.min_duration_hours,
        min_pixels=gate.min_pixels,
        coi_factor=gate.coi_factor,
    )

    # Stage 5: physical sanity — require at least N oscillations of
    # the peak period inside the packet window. A "wave packet" with
    # fewer than two oscillations is just a glitch.
    if gate.min_oscillations > 0:
        kept: list[WavePacketPeak] = []
        for p in packets:
            if p.period_sec is None or p.period_sec <= 0:
                continue
            duration_sec = (p.date_to - p.date_from).total_seconds()
            n_osc = duration_sec / p.period_sec
            if n_osc >= gate.min_oscillations:
                kept.append(p)
        packets = kept

    return packets


def _dt_to_unix(dt_obj: datetime.datetime) -> float:
    """Convert a naive datetime to POSIX timestamp (assuming UTC)."""
    epoch = datetime.datetime(1970, 1, 1)
    return (dt_obj - epoch).total_seconds()


def _nearest_index(arr: np.ndarray, value: float) -> int:
    """Find the index of the nearest value in a sorted array."""
    return int(np.argmin(np.abs(arr - value)))


def dedup_peaks_by_band(
    peaks: list[WavePacketPeak],
    dt_sec: float = 7200.0,
) -> list[WavePacketPeak]:
    """Suppress peaks that fall within ``dt_sec`` of a prior same-band peak.

    The transverse pair (b_perp1, b_perp2) frequently picks up the same
    physical wave train twice. The merge removes those near-duplicates
    while preserving genuinely close peaks in *different* bands (e.g.
    multi-harmonic events). A per-band rolling-last is required: a naïve
    ``merged[-1]``-only guard would let close same-band peaks through
    whenever a different-band peak happens to be interleaved in the
    time-sorted stream.

    Parameters
    ----------
    peaks : list[WavePacketPeak]
        Must be sorted by ``peak_time``. Not modified.
    dt_sec : float, default 7200
        Minimum separation in seconds for same-band peaks. Closer peaks
        are dropped (keeping the earlier one).

    Returns
    -------
    list[WavePacketPeak]
    """
    merged: list[WavePacketPeak] = []
    last_by_band: dict[str | None, WavePacketPeak] = {}
    for peak in peaks:
        prev = last_by_band.get(peak.band)
        if prev is not None:
            sep = abs((peak.peak_time - prev.peak_time).total_seconds())
            if sep < dt_sec:
                continue
        merged.append(peak)
        last_by_band[peak.band] = peak
    return merged


# ---------------------------------------------------------------------
# Round-8 detector — public entry point
# ---------------------------------------------------------------------
#
# Four physically distinct gates, each with a canonical (or
# search-volume-derived) threshold:
#
#   amplitude       whitened sigma-mask            Bonferroni FWER 1%
#   narrowness      spectral Q-factor              >= 3
#   transversality  MVA major-axis projection      |e_max . b_par|^2 <= 0.5
#   polarization    Stokes degree d                >= 0.7
#
# Plus three housekeeping rules: duration >= 2 h, dedup 2 h same-band,
# min_pixels 10. See the round-7/round-8 retrospectives in
# planner notes for the full lineage.

#: Family-wise error rate for the whitened CWT sigma-mask. The actual
#: sigma threshold is derived from the effective number of independent
#: time-frequency cells of the Morlet scalogram (see
#: :func:`bonferroni_n_sigma_for_cwt`).
SEGMENT_FWER_ALPHA: float = 0.01

#: Minimum spectral Q = period / FWHM. Floor for any band-limited peak.
MIN_Q_FACTOR: float = 3.0

#: Re-exports from :mod:`qp.signal.polarization_config` — the canonical
#: home for polarization thresholds. The justification block below is
#: kept here next to its consumer; the constants themselves live in the
#: config module so :mod:`qp.signal.cross_correlation` and the rest of
#: the polarization surface share a single source of truth.
#:
#: ``d = sqrt(Q^2 + U^2 + V^2) / I`` is computed from the cross-Stokes
#: parameters of the b_perp1 / b_perp2 Morlet-CWT analytic-signal
#: coefficients, averaged over the band's frequency rows and the
#: event's time window (see
#: :func:`qp.signal.polarization.degree_of_polarization`). ``d = 1``
#: is a fully polarized wave — circular, linear, or elliptical alike;
#: ``d = 0`` is unpolarized broadband noise, where the cross-terms
#: :math:`\langle z_1 z_2^* \rangle` average to zero.
#:
#: Community convention. The 0.6–0.8 range is standard for "polarized
#: event" identification in magnetospheric ULF-wave catalogues (Samson
#: 1973, *Geophys. J. R. Astr. Soc.* 34, 403; Anderson et al. 1990,
#: *JGR* 95 A6; Bortnik et al. 2007, *JGR* 112 A11; Engebretson et al.
#: 1986, *JGR* 91 A7); 0.7 is the most frequently adopted single value.
#:
#: Empirical justification for 0.7 in this catalogue —
#: ``scripts/diag_stokes_distribution.py`` re-runs the detector on 111
#: representative segments with the Stokes gate disabled and records
#: ``d`` for every candidate that survives the σ-mask + Q-factor + MVA
#: gates (83 candidates total). The distribution is bimodal-ish: a long
#: "mixed/ambiguous" tail centred near 0.55 and a sharp "clean
#: transverse Alfvén" peak near 0.9
#: (see ``Output/figures/diag_p4_stokes_distribution.png``). The 0.7
#: cut sits at the natural shoulder between the two populations — the
#: catalogue would grow by ~30 % at 0.6 (mixing in linear / partly
#: compressional candidates) and shrink by ~30 % at 0.8 (excluding
#: moderately polarized real waves). 0.7 is therefore both the most
#: defensible single value here and aligned with community convention.
from qp.signal.polarization_config import (  # noqa: E402
    MAX_MVA_PARALLEL_FRACTION,
    MIN_DEGREE_OF_POLARIZATION,
)


def _v_indep_for_cwt(
    n_time: int,
    dt: float,
    freq: np.ndarray,
    morlet_omega0: float = 10.0,
) -> float:
    r"""Effective number of independent time-frequency cells in the Morlet CWT.

    The Morlet wavelet has temporal correlation length ~1 period at
    every frequency and frequency-bandwidth :math:`\Delta f / f \approx
    1/\omega_0`. The number of *independent* time-frequency cells is

    .. math::

        V_{\mathrm{indep}} = \frac{\omega_0}{2\pi}
            \ln\!\left(\frac{f_{\max}}{f_{\min}}\right)
        \cdot
        n_t\,dt\,\bar f.
    """
    freq = np.asarray(freq, dtype=float)
    f_min = float(freq[freq > 0].min())
    f_max = float(freq.max())
    n_freq_indep = (morlet_omega0 / (2.0 * np.pi)) * np.log(f_max / f_min)
    n_time_indep = n_time * dt * float(freq.mean())
    return max(n_freq_indep * n_time_indep, 1.0)


def bonferroni_n_sigma_for_cwt(
    n_time: int,
    dt: float,
    freq: np.ndarray,
    morlet_omega0: float = 10.0,
    alpha: float = SEGMENT_FWER_ALPHA,
) -> float:
    r"""Sigma threshold for FWER control over the Morlet-CWT search volume.

    Bonferroni sets the per-pixel FP probability to
    :math:`\alpha / V_{\mathrm{indep}}` (see :func:`_v_indep_for_cwt`)
    and the threshold is the corresponding Gaussian quantile (~4.6σ at
    α = 0.01).
    """
    from scipy.stats import norm

    v_indep = _v_indep_for_cwt(n_time, dt, freq, morlet_omega0)
    return float(norm.isf(alpha / v_indep))


from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True, slots=True)
class DetectedEvent:
    """A WavePacketPeak that passed all round-8 gates, with diagnostics.

    The four gate values plus the bandpass amplitudes per component are
    retained so callers can persist them in tabular form without
    rerunning the detector. The Stokes vector and derived ellipticity /
    inclination / polarized fraction are computed once from the same
    in-band CWT slice that feeds the ``stokes_d`` gate — persisting
    them costs nothing and lets downstream consumers (Fig 10, fig11,
    sensitivity analysis) read the polarization geometry directly from
    the catalogue without rerunning the sweep.
    """

    peak: WavePacketPeak
    q_factor: float
    mva_par_frac: float
    stokes_d: float
    b_perp1_amp: float
    b_perp2_amp: float
    b_par_amp: float
    stokes_i: float
    stokes_q: float
    stokes_u: float
    stokes_v: float
    ellipticity: float
    inclination_deg: float
    polarized_fraction: float


def detect_round8(
    t: NDArray[np.floating],
    fields: NDArray[np.floating],
    dt: float = 60.0,
    *,
    n_freqs: int = 300,
    fwer_alpha: float = SEGMENT_FWER_ALPHA,
    min_q_factor: float = MIN_Q_FACTOR,
    min_stokes_d: float = MIN_DEGREE_OF_POLARIZATION,
    max_mva_par_frac: float = MAX_MVA_PARALLEL_FRACTION,
    min_duration_hours: float = 2.0,
    min_pixels: int = 10,
    epoch: datetime.datetime | None = None,
    threshold_method: ThresholdMethod = "tc_chi2",
    apply_coi_mask: bool = True,
    bg_archive: "BGArchive | None" = None,
    region: str | None = None,
) -> list[DetectedEvent]:
    """Round-8 simplified wave-event detector.

    Parameters
    ----------
    t : array_like, shape (N,)
        Time samples in seconds since ``epoch``.
    fields : array_like, shape (N, 3)
        ``[b_par, b_perp1, b_perp2]`` in MFA frame.
    dt : float, default 60
        Sampling interval, seconds.
    epoch : datetime, optional
        Reference epoch for ``t``. Default 2000-01-01 (J2000-ish).
    threshold_method : {"mad_row", "tc_chi2", "fdr_chi2", "pooled"}
        CWT amplitude gate. ``tc_chi2`` (default) — Torrence & Compo
        1998 AR(1)+χ²(2) per-pixel test at the same Bonferroni FWER
        ``fwer_alpha``. ``mad_row`` keeps the legacy per-row MAD on
        background period rows. ``fdr_chi2`` and ``pooled`` are
        diagnostic-only; ``pooled`` additionally requires ``bg_archive``
        and ``region``.
    apply_coi_mask : bool, default True
        Drop wavelet cells inside the Morlet cone of influence before
        ridge extraction. The ``min_duration_hours = 2`` cut discards
        most COI-tainted events implicitly; this is the rigorous
        version and pairs naturally with the χ²-based gate.
    bg_archive : BGArchive, optional
        Pre-computed pooled background-row statistics (see
        :mod:`qp.events.threshold_diag`). Only used when
        ``threshold_method='pooled'``.
    region : str, optional
        Plasma-region key into ``bg_archive``. Only used when
        ``threshold_method='pooled'``.

    Returns
    -------
    list of DetectedEvent
        One entry per peak that passed all four gates. The gate values
        are populated; the underlying ``WavePacketPeak`` is unchanged.
    """
    from qp.events.bands import get_band
    from qp.events.threshold import wavelet_sigma_mask
    from qp.events.threshold_diag import (
        coi_mask as _coi_mask,
        fdr_chi2_mask,
        pooled_archive_mask,
        torrence_compo_chi2_mask,
    )
    from qp.signal.polarization import (
        ellipticity_inclination_from_stokes,
        mva_major_axis_parallel_fraction,
        stokes_parameters,
    )

    if epoch is None:
        epoch = datetime.datetime(2000, 1, 1)

    b_par = np.asarray(fields[:, 0], dtype=float)
    b_perp1 = np.asarray(fields[:, 1], dtype=float)
    b_perp2 = np.asarray(fields[:, 2], dtype=float)

    # Complex CWTs of all three field components. The transverse pair
    # feeds ridge extraction and Stokes; b_par enters MVA.
    freq, _, cwt_par = morlet_cwt(b_par, dt=dt, n_freqs=n_freqs)
    _, _, cwt1 = morlet_cwt(b_perp1, dt=dt, n_freqs=n_freqs)
    _, _, cwt2 = morlet_cwt(b_perp2, dt=dt, n_freqs=n_freqs)
    power1 = np.abs(cwt1)
    power2 = np.abs(cwt2)

    n_time = power1.shape[1]

    if threshold_method == "mad_row":
        n_sigma = bonferroni_n_sigma_for_cwt(n_time, dt, freq, alpha=fwer_alpha)
        mask1 = wavelet_sigma_mask(power1, freq, n_sigma=n_sigma)
        mask2 = wavelet_sigma_mask(power2, freq, n_sigma=n_sigma)
    elif threshold_method == "tc_chi2":
        # Bonferroni-corrected per-pixel α under the same FWER budget.
        v_indep = _v_indep_for_cwt(n_time, dt, freq)
        alpha_pixel = fwer_alpha / v_indep
        mask1 = torrence_compo_chi2_mask(
            power1,
            freq,
            b_perp1,
            dt=dt,
            alpha=alpha_pixel,
        )
        mask2 = torrence_compo_chi2_mask(
            power2,
            freq,
            b_perp2,
            dt=dt,
            alpha=alpha_pixel,
        )
    elif threshold_method == "fdr_chi2":
        # FDR target is the FWER budget — apples-to-apples conservatism.
        mask1 = fdr_chi2_mask(power1, freq, b_perp1, dt=dt, q=fwer_alpha)
        mask2 = fdr_chi2_mask(power2, freq, b_perp2, dt=dt, q=fwer_alpha)
    elif threshold_method == "pooled":
        if bg_archive is None or region is None:
            raise ValueError(
                "threshold_method='pooled' requires both bg_archive and region; "
                "build the archive with scripts/build_bg_archive.py and pass "
                "the segment's plasma-region label"
            )
        n_sigma = bonferroni_n_sigma_for_cwt(n_time, dt, freq, alpha=fwer_alpha)
        mask1 = pooled_archive_mask(power1, freq, region, bg_archive, n_sigma=n_sigma)
        mask2 = pooled_archive_mask(power2, freq, region, bg_archive, n_sigma=n_sigma)
    else:  # pragma: no cover — Literal exhausts the cases
        raise ValueError(f"unknown threshold_method: {threshold_method!r}")

    if apply_coi_mask:
        coi = _coi_mask(freq, n_time, dt=dt)
        mask1 = mask1 & coi
        mask2 = mask2 & coi

    times = [epoch + datetime.timedelta(seconds=float(s)) for s in t]

    all_peaks: list[WavePacketPeak] = []
    for component, power, mask in (
        (b_perp1, power1, mask1),
        (b_perp2, power2, mask2),
    ):
        peaks = detect_wave_packets_multi(
            data=component,
            times=times,
            dt=dt,
            cwt_freq=freq,
            cwt_power=power,
            threshold_mask=mask,
            min_duration_hours=min_duration_hours,
            min_pixels=min_pixels,
        )
        all_peaks.extend(peaks)

    all_peaks.sort(key=lambda p: p.peak_time)
    merged = dedup_peaks_by_band(all_peaks, dt_sec=7200.0)

    # Per-detection physical gates.
    n_time = cwt1.shape[1]
    kept: list[DetectedEvent] = []
    for peak in merged:
        if peak.period_sec is None or peak.period_sec <= 0 or peak.band is None:
            continue
        q = peak.q_factor
        if q is None or q < min_q_factor:
            continue
        i_start = max(
            0,
            int(np.floor((peak.date_from - epoch).total_seconds() / dt)),
        )
        i_end = min(
            n_time - 1,
            int(np.ceil((peak.date_to - epoch).total_seconds() / dt)),
        )
        if i_end <= i_start:
            continue
        # Transversality: MVA on bandpass-filtered 3-component field.
        i_freq_peak = int(np.argmin(np.abs(freq - 1.0 / peak.period_sec)))
        sl = slice(i_start, i_end + 1)
        field_bp = np.column_stack(
            [
                np.real(cwt_par[i_freq_peak, sl]),
                np.real(cwt1[i_freq_peak, sl]),
                np.real(cwt2[i_freq_peak, sl]),
            ]
        )
        par_frac = mva_major_axis_parallel_fraction(field_bp, par_axis=0)
        if par_frac > max_mva_par_frac:
            continue
        # Polarization purity over the in-band TF window.
        band_obj = get_band(peak.band)
        in_band = (freq >= band_obj.freq_min_hz) & (freq < band_obj.freq_max_hz)
        if not in_band.any():
            continue
        c1_window = cwt1[in_band, sl]
        c2_window = cwt2[in_band, sl]
        s_i, s_q, s_u, s_v = stokes_parameters(
            c1_window.ravel(),
            c2_window.ravel(),
        )
        d = (np.sqrt(s_q * s_q + s_u * s_u + s_v * s_v) / s_i) if s_i > 0 else 0.0
        if d < min_stokes_d:
            continue
        ell, incl_deg, pol_frac = ellipticity_inclination_from_stokes(
            s_i,
            s_q,
            s_u,
            s_v,
        )
        # Bandpass amplitudes (RMS of the real CWT slice at peak f).
        b_perp1_amp = float(np.sqrt(np.mean(field_bp[:, 1] ** 2)))
        b_perp2_amp = float(np.sqrt(np.mean(field_bp[:, 2] ** 2)))
        b_par_amp = float(np.sqrt(np.mean(field_bp[:, 0] ** 2)))
        kept.append(
            DetectedEvent(
                peak=peak,
                q_factor=float(q),
                mva_par_frac=float(par_frac),
                stokes_d=float(d),
                b_perp1_amp=b_perp1_amp,
                b_perp2_amp=b_perp2_amp,
                b_par_amp=b_par_amp,
                stokes_i=float(s_i),
                stokes_q=float(s_q),
                stokes_u=float(s_u),
                stokes_v=float(s_v),
                ellipticity=float(ell),
                inclination_deg=float(incl_deg),
                polarized_fraction=float(pol_frac),
            )
        )
    return kept
