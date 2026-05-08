"""Cached data loaders for the event review web app.

The 36-hour MFA segment archive is held in a legacy pickled `.npy` file
that requires `qp.io.register_legacy_pickle_stubs()` to be installed
before any deserialization. We register stubs at import time so every
endpoint below can rely on segments being loadable.
"""

from __future__ import annotations

import bisect
import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qp
from qp.events.sweep_loader import (
    SegmentPayload,
    segment_to_payload,
)
from qp.io import register_legacy_pickle_stubs
from qp.signal.fft import welch_psd

register_legacy_pickle_stubs()


_REGION_CODE_TO_NAME = {
    0: "magnetosphere",
    1: "magnetosheath",
    2: "solar_wind",
    9: "unknown",
}

EVENTS_PARQUET: Path = qp.OUTPUT_DIR / "events_round8.parquet"
SEGMENTS_NPY: Path = qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy"


# --------------------------------------------------------------------- #
# Cached file loads                                                     #
# --------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def load_event_table() -> pd.DataFrame:
    """Load the round-8 event catalogue once, parsing peak_time."""
    df = pd.read_parquet(EVENTS_PARQUET)
    for col in ("date_from", "date_to", "peak_time"):
        df[col] = pd.to_datetime(df[col])
    return df


@lru_cache(maxsize=1)
def _segment_array() -> np.ndarray:
    return np.load(SEGMENTS_NPY, allow_pickle=True)


@lru_cache(maxsize=64)
def get_segment_payload(seg_idx: int) -> SegmentPayload | None:
    """Return a `SegmentPayload` for the given index (cached)."""
    arr = _segment_array()
    if seg_idx < 0 or seg_idx >= len(arr):
        return None
    return segment_to_payload(seg_idx, arr[seg_idx])


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _slice_window(
    times: list[datetime.datetime],
    peak: datetime.datetime,
    hours_pad: float,
) -> tuple[int, int]:
    pad = datetime.timedelta(hours=hours_pad)
    t_lo = peak - pad
    t_hi = peak + pad
    lo = bisect.bisect_left(times, t_lo)
    hi = bisect.bisect_right(times, t_hi)
    return max(lo, 0), min(hi, len(times))


def _region_spans(
    payload: SegmentPayload,
    t_start: datetime.datetime,
    t_end: datetime.datetime,
) -> list[dict[str, Any]]:
    """Build (t0, t1, region) spans inside [t_start, t_end].

    Walks ``info["flag_times"]`` / ``info["locations"]``: each flag_time
    marks the start of a region code. A span runs from one flag_time to
    the next (clipped to the requested window).
    """
    info = payload.info
    flag_times = info.get("flag_times")
    locations = info.get("locations")
    if not flag_times or not locations:
        return []
    pairs = sorted(
        (
            (t, int(c))
            for t, c in zip(flag_times, locations)
            if isinstance(t, datetime.datetime)
        ),
        key=lambda x: x[0],
    )
    if not pairs:
        return []

    spans: list[dict[str, Any]] = []
    for i, (t, code) in enumerate(pairs):
        t0 = max(t, t_start)
        t1 = pairs[i + 1][0] if i + 1 < len(pairs) else t_end
        t1 = min(t1, t_end)
        if t1 <= t0:
            continue
        if t < t_start and t1 <= t_start:
            continue
        spans.append(
            {
                "t0": t0.isoformat(),
                "t1": t1.isoformat(),
                "region": _REGION_CODE_TO_NAME.get(code, "unknown"),
            }
        )
    if pairs[0][0] > t_start:
        spans.insert(
            0,
            {
                "t0": t_start.isoformat(),
                "t1": min(pairs[0][0], t_end).isoformat(),
                "region": "unknown",
            },
        )
    return spans


# --------------------------------------------------------------------- #
# Public API                                                            #
# --------------------------------------------------------------------- #


def event_summaries(
    band: str | None = None,
    region: str | None = None,
    sort: str = "peak_time",
) -> list[dict[str, Any]]:
    """Lightweight list of all events for the timeline / browse strip."""
    df = load_event_table()
    if band:
        df = df[df.band == band]
    if region:
        df = df[df.region == region]
    if sort in df.columns:
        df = df.sort_values(sort, kind="mergesort")
    cols = [
        "event_id", "segment_idx", "peak_time", "band", "region",
        "r_distance", "mag_lat", "local_time", "period_min",
        "duration_minutes", "q_factor", "stokes_d",
        "b_perp1_amp", "b_perp2_amp", "b_par_amp",
    ]
    out = df[cols].copy()
    out["peak_time"] = out["peak_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out.to_dict(orient="records")


def event_detail(event_id: int) -> dict[str, Any] | None:
    df = load_event_table()
    rows = df[df.event_id == int(event_id)]
    if rows.empty:
        return None
    row = rows.iloc[0].to_dict()
    for k, v in list(row.items()):
        if isinstance(v, pd.Timestamp):
            row[k] = v.isoformat()
        elif isinstance(v, (np.integer,)):
            row[k] = int(v)
        elif isinstance(v, (np.floating,)):
            row[k] = float(v) if np.isfinite(v) else None
    return row


@lru_cache(maxsize=256)
def event_waveform(event_id: int, hours_pad: float = 12.0) -> dict[str, Any] | None:
    """Return MFA waveform window centered on peak_time. LRU cached so the
    same event is served instantly on revisit (e.g., toggling zoom/detrend
    on the client doesn't refetch — but in case it ever does, this is cheap).
    """
    df = load_event_table()
    rows = df[df.event_id == int(event_id)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    seg_idx = int(row.segment_idx)
    peak = row.peak_time.to_pydatetime()
    payload = get_segment_payload(seg_idx)
    if payload is None:
        return None
    lo, hi = _slice_window(payload.times, peak, hours_pad)
    if hi - lo < 2:
        return None
    times = payload.times[lo:hi]
    b_par = payload.b_par[lo:hi]
    b_perp1 = payload.b_perp1[lo:hi]
    b_perp2 = payload.b_perp2[lo:hi]
    b_tot = np.sqrt(b_par**2 + b_perp1**2 + b_perp2**2)

    spans = _region_spans(payload, times[0], times[-1])

    return {
        "event_id": int(row.event_id),
        "segment_idx": seg_idx,
        "peak_time": peak.isoformat(),
        "date_from": row.date_from.isoformat(),
        "date_to": row.date_to.isoformat(),
        "band": row.band,
        "region": row.region,
        "epoch_s": [t.timestamp() for t in times],
        "b_par": _to_clean_list(b_par),
        "b_perp1": _to_clean_list(b_perp1),
        "b_perp2": _to_clean_list(b_perp2),
        "b_tot": _to_clean_list(b_tot),
        "region_spans": spans,
    }


@lru_cache(maxsize=256)
def event_spectrum(
    event_id: int, hours_pad: float = 12.0,
) -> dict[str, Any] | None:
    """Welch PSD over the event window (1-min cadence, 12-h window)."""
    df = load_event_table()
    rows = df[df.event_id == int(event_id)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    seg_idx = int(row.segment_idx)
    peak = row.peak_time.to_pydatetime()
    payload = get_segment_payload(seg_idx)
    if payload is None:
        return None
    lo, hi = _slice_window(payload.times, peak, hours_pad)
    if hi - lo < 64:
        return None
    n = hi - lo
    nperseg = min(n, 12 * 60)  # 12-hour Welch window matches scripts/fig04
    b_par = _detrend(payload.b_par[lo:hi])
    b_perp1 = _detrend(payload.b_perp1[lo:hi])
    b_perp2 = _detrend(payload.b_perp2[lo:hi])
    freq, psd_par = welch_psd(b_par, dt=60.0, nperseg=nperseg)
    _, psd_p1 = welch_psd(b_perp1, dt=60.0, nperseg=nperseg)
    _, psd_p2 = welch_psd(b_perp2, dt=60.0, nperseg=nperseg)
    keep = freq > 0
    f = freq[keep]
    period_min = 1.0 / f / 60.0
    # uPlot requires monotonically increasing x — period decreases with
    # frequency, so reverse so the smallest period is first.
    order = np.argsort(period_min)
    return {
        "freq_hz":      f[order].tolist(),
        "period_min":   period_min[order].tolist(),
        "psd_par":      _to_clean_list(psd_par[keep][order]),
        "psd_perp1":    _to_clean_list(psd_p1[keep][order]),
        "psd_perp2":    _to_clean_list(psd_p2[keep][order]),
        "qp_periods_min": [30.0, 60.0, 120.0],
    }


def timeline_summary() -> list[dict[str, Any]]:
    """Per-event minimal record for the top timeline strip."""
    df = load_event_table()
    out = df[["event_id", "peak_time", "band", "region"]].copy()
    out["peak_time"] = out["peak_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out.to_dict(orient="records")


@lru_cache(maxsize=1)
def region_intervals() -> list[dict[str, Any]]:
    """Mission-wide MS/SH/SW intervals from BS/MP boundary crossings.

    Walks the Jackman 2019 crossing list and emits one record per
    interval between consecutive crossings, labelled by the region the
    spacecraft sits in *after* the most recent crossing.
    """
    from qp.io.crossings import crossing_lookup_arrays

    times_unix, codes = crossing_lookup_arrays()
    if len(times_unix) == 0:
        return []
    out: list[dict[str, Any]] = []
    for i in range(len(times_unix) - 1):
        t0 = float(times_unix[i])
        t1 = float(times_unix[i + 1])
        if t1 <= t0:
            continue
        out.append({
            "epoch_start": t0,
            "epoch_end":   t1,
            "region":      _REGION_CODE_TO_NAME.get(int(codes[i]), "unknown"),
        })
    # Tail interval until end of mission (2017-09-15) keeps coverage complete.
    t_last = float(times_unix[-1])
    t_end = datetime.datetime(2017, 9, 15).timestamp()
    if t_end > t_last:
        out.append({
            "epoch_start": t_last,
            "epoch_end":   t_end,
            "region":      _REGION_CODE_TO_NAME.get(int(codes[-1]), "unknown"),
        })
    return out


# --------------------------------------------------------------------- #
# Internal utilities                                                    #
# --------------------------------------------------------------------- #


def _detrend(x: np.ndarray) -> np.ndarray:
    """Remove the linear trend; replicates the prep used in fig04."""
    n = len(x)
    if n < 3:
        return x
    t = np.arange(n, dtype=float)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def _to_clean_list(arr: np.ndarray) -> list[float | None]:
    """Convert array to list, mapping non-finite to None for JSON safety."""
    out: list[float | None] = []
    for v in arr.tolist():
        out.append(float(v) if np.isfinite(v) else None)
    return out
