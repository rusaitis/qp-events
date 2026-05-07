"""Shared loaders and ephemeris helpers for the mission-wide event sweep.

Every QP event-detection entry point (`scripts/sweep_events_round8.py`,
`scripts/fig_waveform_gallery.py`, ...) needs the same scaffolding: load the
36-hour MFA segments from disk, slice the central 24 h, look up SLS5 phase /
region / spacecraft coordinates at the wave-packet peak. Those helpers used
to live in `scripts/sweep_events.py`; they have been lifted here so a single
canonical implementation is shared by the round-8 sweep and any future
script that walks the same segment archive.

.. important::

    `load_segments` reads ``DataProducts/Cassini_MAG_MFA_36H.npy``, whose
    pickle stream references long-removed module paths
    (``data_sweeper``, ``mag_fft_sweeper``, ``cassinilib`` and
    ``cassinilib.NewSignal``). Callers must register pickle stubs **before**
    calling this function — see the ``_register_pickle_stubs()`` helper at
    the top of every consumer script. The package deliberately does *not*
    inject sentinel modules into ``sys.modules`` at import time; that side
    effect belongs to the script entry point.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import qp

__all__ = [
    "SegmentPayload",
    "load_segments",
    "segment_to_payload",
    "segment_central_window",
    "ppo_at_peak_from_info",
    "region_at_peak_from_info",
]


_REGION_CODE_TO_NAME = {
    0: "magnetosphere",
    1: "magnetosheath",
    2: "solar_wind",
    9: "unknown",
}

_INFO_KEYS_TO_KEEP = (
    "median_LT", "median_BT", "median_coords", "location",
    "locations", "flag_times",
    "SLS5N", "SLS5S", "SLS5N2", "SLS5S2", "NaN_count", "gaps",
)


@dataclass
class SegmentPayload:
    """Picklable view of a single 36-hour MFA segment.

    The native segment objects in ``Cassini_MAG_MFA_36H.npy`` are pickled
    instances of legacy classes (``SignalSnapshot`` etc.). Shipping them
    across a multiprocessing-spawn boundary would require every worker to
    re-register the pickle stubs and re-load. A `SegmentPayload` is the
    minimal subset of fields actually consumed by the sweep, in plain
    NumPy types only.
    """

    seg_idx: int
    times: list[datetime.datetime]
    b_par: np.ndarray
    b_perp1: np.ndarray
    b_perp2: np.ndarray
    coord_r: np.ndarray | None
    coord_th: np.ndarray | None
    coord_phi: np.ndarray | None
    info: dict = field(default_factory=dict)


def load_segments(year: int | None = None) -> tuple[np.ndarray, list[int]]:
    """Load all MFA segments and return (array, indices to process).

    Parameters
    ----------
    year : int or None
        If given, only segments whose first sample falls in this calendar
        year are kept; otherwise every segment is returned.

    Returns
    -------
    array : numpy.ndarray
        The full segment array (object dtype). Index with ``array[i]``.
    keep_idx : list of int
        The subset of indices that match the year filter (or all of them).

    Notes
    -----
    Callers must register pickle stubs before calling — see the module
    docstring.
    """
    products = qp.DATA_PRODUCTS
    arr = np.load(products / "Cassini_MAG_MFA_36H.npy", allow_pickle=True)
    if year is None:
        keep_idx = list(range(len(arr)))
    else:
        keep_idx = []
        for i, seg in enumerate(arr):
            t0 = seg.datetime[0]
            if isinstance(t0, datetime.datetime) and t0.year == year:
                keep_idx.append(i)
    return arr, keep_idx


def _clean_info(info: dict) -> dict:
    """Strip non-picklable / un-needed fields from a segment's ``info``."""
    out: dict = {}
    for k in _INFO_KEYS_TO_KEEP:
        if k not in info:
            continue
        v = info[k]
        if hasattr(v, "tolist"):
            try:
                out[k] = v.tolist()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def segment_to_payload(seg_idx: int, seg: Any) -> SegmentPayload | None:
    """Extract a `SegmentPayload` from a SignalSnapshot in the parent process.

    Returns ``None`` for segments that are flagged, malformed, or shorter
    than 18 hours of valid samples.
    """
    if getattr(seg, "flag", None) is not None:
        return None
    if not hasattr(seg, "FIELDS") or len(seg.FIELDS) < 4:
        return None
    if not hasattr(seg, "datetime") or len(seg.datetime) == 0:
        return None
    times = list(seg.datetime)
    n_samples = len(times)
    if n_samples < 18 * 60:
        return None
    b_par = np.asarray(seg.FIELDS[0].y, dtype=float)
    b_perp1 = np.asarray(seg.FIELDS[1].y, dtype=float)
    b_perp2 = np.asarray(seg.FIELDS[2].y, dtype=float)
    if (
        len(b_perp1) != n_samples
        or np.isnan(b_perp1).all()
        or np.isnan(b_perp2).all()
    ):
        return None
    coords = {c.name: np.asarray(c.y, dtype=float) for c in seg.COORDS}
    info = getattr(seg, "info", None) or {}
    return SegmentPayload(
        seg_idx=seg_idx,
        times=times,
        b_par=np.nan_to_num(b_par, nan=0.0),
        b_perp1=np.nan_to_num(b_perp1, nan=0.0),
        b_perp2=np.nan_to_num(b_perp2, nan=0.0),
        coord_r=coords.get("r"),
        coord_th=coords.get("th"),
        coord_phi=coords.get("phi"),
        info=_clean_info(info),
    )


def segment_central_window(
    times: list[datetime.datetime], hours_pad: int = 6,
) -> tuple[datetime.datetime, datetime.datetime]:
    """Return the central 24-hour window of a 36-hour segment.

    The 6-hour overlap on each side prevents events near segment edges
    from being double-counted across adjacent segments.
    """
    t0 = times[0]
    t_end = times[-1]
    central_start = t0 + datetime.timedelta(hours=hours_pad)
    central_end = t_end - datetime.timedelta(hours=hours_pad - 1)
    return central_start, central_end


def ppo_at_peak_from_info(
    info: dict,
    peak_time: datetime.datetime,
    seg_t0: datetime.datetime,
    n_samples: int,
) -> dict[str, float | None]:
    """Look up SLS5 N/S phases at a wave-packet peak time."""
    out: dict[str, float | None] = {
        "sls5n": None, "sls5s": None,
        "sls5n2": None, "sls5s2": None,
    }
    elapsed_min = (peak_time - seg_t0).total_seconds() / 60.0
    for key in ("SLS5N", "SLS5S", "SLS5N2", "SLS5S2"):
        arr = info.get(key)
        if arr is None or len(arr) == 0:
            continue
        cadence = n_samples / len(arr)
        idx = int(round(elapsed_min / cadence))
        idx = max(0, min(len(arr) - 1, idx))
        try:
            out[key.lower()] = float(arr[idx])
        except (TypeError, ValueError):
            pass
    return out


def region_at_peak_from_info(
    info: dict, peak_time: datetime.datetime,
) -> str:
    """Translate ``seg.info['locations']`` into a region label at peak_time."""
    locations = info.get("locations")
    flag_times = info.get("flag_times")
    if locations is None or flag_times is None:
        return "unknown"
    times = [t for t in flag_times if isinstance(t, datetime.datetime)]
    if not times:
        return "unknown"
    deltas = [abs((t - peak_time).total_seconds()) for t in times]
    idx = int(np.argmin(deltas))
    if idx >= len(locations):
        return "unknown"
    try:
        code = int(locations[idx])
    except (TypeError, ValueError):
        return "unknown"
    return _REGION_CODE_TO_NAME.get(code, "unknown")
