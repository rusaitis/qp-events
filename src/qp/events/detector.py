"""Wave event detection from CWT power.

Detects QP wave packets by:
1. Computing Morlet CWT of MFA field component
2. Extracting power at target periods (e.g., 55-65 min for QP60)
3. Computing event measure as the norm across period bins
4. Finding peaks in the event measure using scipy.signal.find_peaks

Extracted from cassinilib/PlotFFT.py:collectWaveEvents and
calculateEventSeparation.
"""

from __future__ import annotations

import datetime

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks

from qp.signal.wavelet import morlet_cwt
from qp.events.catalog import WaveEvent, WavePacketPeak


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
    """Detect wave packets in a time series using CWT.

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
    freq, time_sec, cwt_matrix = morlet_cwt(data, dt=dt, n_freqs=300)

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
    packets = []
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
    r"""Detect wave events with full metadata.

    Wraps ``detect_wave_packets()`` and enriches the results with
    spacecraft coordinates and SLS5 phase information.

    Extracted from ``cassinilib/PlotFFT.py:collectWaveEvents()``.

    Parameters
    ----------
    field_data : array_like
        A single MFA field component (e.g., $b_{\perp 1}$) in nT.
    time_unix : array_like
        POSIX timestamps for each sample.
    dt : float
        Sampling interval in seconds.
    period_band : tuple[float, float]
        Target period band in seconds. Default (3000, 4200) = QP60.
    min_snr : float
        Minimum prominence threshold (normalized CWT units).
    min_duration_hours : float
        Minimum wave packet duration to accept.
    coords_krtp : ndarray, shape (N, 3), optional
        Spacecraft position in KRTP: columns (r, theta, phi) in
        (R_S, radians, radians).
    local_time : ndarray, shape (N,), optional
        Local time in hours.
    sls5_phases : dict[str, ndarray], optional
        SLS5 phase arrays keyed by name (e.g., ``{'SLS5N': ..., 'SLS5S': ...}``).
        Each array has one value per 10-minute bin (shape N//10 or similar).

    Returns
    -------
    list[WaveEvent]
        Detected events with coordinates and SLS5 metadata.
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


def _dt_to_unix(dt_obj: datetime.datetime) -> float:
    """Convert a naive datetime to POSIX timestamp (assuming UTC)."""
    epoch = datetime.datetime(1970, 1, 1)
    return (dt_obj - epoch).total_seconds()


def _nearest_index(arr: np.ndarray, value: float) -> int:
    """Find the index of the nearest value in a sorted array."""
    return int(np.argmin(np.abs(arr - value)))
