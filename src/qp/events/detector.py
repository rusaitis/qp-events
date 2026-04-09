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
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

from qp.events.bands import QP_BAND_NAMES, Band
from qp.events.catalog import WaveEvent, WavePacketPeak
from qp.events.ridge import Ridge, extract_ridges
from qp.signal.wavelet import morlet_cwt


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

    # Extract power at target period bins
    period_min_sec, period_max_sec = period_band
    target_periods = np.linspace(period_min_sec, period_max_sec, n_period_bins)
    target_freqs = 1.0 / target_periods

    cwt_slices = []
    for f_target in target_freqs:
        idx = np.argmin(np.abs(freq - f_target))
        cwt_slices.append(cwt_power[idx, :])

    # Event measure: norm across period bins
    event_measure = np.linalg.norm(np.asarray(cwt_slices), axis=0)

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

    target_periods = np.linspace(period_band[0], period_band[1], n_period_bins)
    target_freqs = 1.0 / target_periods

    slices = []
    for f_target in target_freqs:
        idx = np.argmin(np.abs(freq - f_target))
        slices.append(cwt_power[idx, :])

    event_measure = np.linalg.norm(np.asarray(slices), axis=0)
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
        raise ValueError(
            f"len(times)={len(times_list)} != len(data)={n_time}"
        )

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
    )


def _dt_to_unix(dt_obj: datetime.datetime) -> float:
    """Convert a naive datetime to POSIX timestamp (assuming UTC)."""
    epoch = datetime.datetime(1970, 1, 1)
    return (dt_obj - epoch).total_seconds()


def _nearest_index(arr: np.ndarray, value: float) -> int:
    """Find the index of the nearest value in a sorted array."""
    return int(np.argmin(np.abs(arr - value)))
