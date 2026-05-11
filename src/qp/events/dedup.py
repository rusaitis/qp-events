r"""Post-hoc deduplication of the round-8 event catalogue.

The round-8 detector merges peaks from the two transverse components inside
each 36-h segment with a 2-hour, same-band rule (``detector.py``). That rule
fails in two ways once the parquet is assembled:

1. **Cross-segment leakage.** Adjacent 36-h MFA segments overlap in time, so
   the same physical wave packet can be detected independently in segment
   ``N`` and segment ``N+1``. Each detection writes one row.
2. **Intra-segment leakage.** The in-segment dedup compares each new peak
   only to ``merged[-1]``. A different-band peak interleaved between two
   same-band peaks defeats the check, so both same-band peaks survive.

A single band-scoped, time-windowed cluster pass over the full catalogue
catches both. We tag rather than collapse so the original detector output
remains losslessly recoverable.

Two events are considered duplicates when

- their ``band`` matches,
- ``|Δpeak_time| ≤ dt_sec``  (default 7200 s = 2 h, mirroring the detector),
- ``|Δperiod_min| / period_min ≤ period_rel_tol`` (default 10 %).

Within a cluster the row with the largest ``q_factor`` is kept; the others
have ``is_duplicate = True``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


DEFAULT_DT_SEC: float = 7200.0
DEFAULT_PERIOD_REL_TOL: float = 0.10


def tag_duplicates(
    df: "pd.DataFrame",
    *,
    dt_sec: float = DEFAULT_DT_SEC,
    period_rel_tol: float = DEFAULT_PERIOD_REL_TOL,
) -> "pd.DataFrame":
    """Return a copy of ``df`` with an added boolean ``is_duplicate`` column.

    The input is not mutated. The returned frame preserves row order. For
    each cluster (band-scoped, time-windowed, period-tolerant), the row
    with the largest ``q_factor`` is kept (``False``) and the rest are
    flagged ``True``. Ties in ``q_factor`` break by lower row index.

    Parameters
    ----------
    df : pandas.DataFrame
        Event catalogue with at least ``band``, ``peak_time``,
        ``period_min``, ``q_factor`` columns. ``peak_time`` may be
        ISO-string or datetime; it is coerced to datetime internally.
    dt_sec : float, default 7200
        Maximum peak-time separation, seconds, for two events to be
        considered duplicates.
    period_rel_tol : float, default 0.10
        Maximum relative period mismatch :math:`|\\Delta P| / P` for two
        events to be considered duplicates.
    """
    import pandas as pd

    out = df.copy()
    out["is_duplicate"] = False
    if len(out) == 0:
        return out

    peak = pd.to_datetime(out["peak_time"]).astype("datetime64[ns]")
    order = peak.argsort(kind="stable").to_numpy()
    bands = out["band"].to_numpy()
    times_s = peak.astype("int64").to_numpy() / 1e9
    periods = out["period_min"].to_numpy(dtype=float)
    qfactors = out["q_factor"].to_numpy(dtype=float)

    # Cluster representatives: index in `out` of the current "kept" row per band.
    keepers: dict[str, int] = {}
    is_dup = [False] * len(out)

    for i in order:
        band = bands[i]
        rep = keepers.get(band)
        if rep is None:
            keepers[band] = int(i)
            continue
        dt = abs(times_s[i] - times_s[rep])
        if dt > dt_sec:
            keepers[band] = int(i)
            continue
        rel = abs(periods[i] - periods[rep]) / max(periods[rep], 1e-9)
        if rel > period_rel_tol:
            keepers[band] = int(i)
            continue
        # Duplicate: keep the higher-q_factor row, tag the other.
        if qfactors[i] > qfactors[rep]:
            is_dup[rep] = True
            keepers[band] = int(i)
        else:
            is_dup[i] = True

    out["is_duplicate"] = is_dup
    return out


def collapse_duplicates(
    df: "pd.DataFrame",
    *,
    dt_sec: float = DEFAULT_DT_SEC,
    period_rel_tol: float = DEFAULT_PERIOD_REL_TOL,
) -> "pd.DataFrame":
    """Tag-and-filter convenience wrapper around :func:`tag_duplicates`.

    Returns a new frame with duplicate rows removed and the
    ``is_duplicate`` column dropped. Row order is preserved.
    """
    tagged = tag_duplicates(df, dt_sec=dt_sec, period_rel_tol=period_rel_tol)
    kept = tagged.loc[~tagged["is_duplicate"]].drop(columns=["is_duplicate"])
    return kept.reset_index(drop=True)
