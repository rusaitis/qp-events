r"""Post-hoc deduplication of the round-8 event catalogue.

The round-8 detector merges peaks from the two transverse components inside
each 36-h segment with a 2-hour, period-proximity rule (``detector.py``).
That rule fails in two ways once the parquet is assembled:

1. **Cross-segment leakage.** Adjacent 36-h MFA segments overlap in time, so
   the same physical wave packet can be detected independently in segment
   ``N`` and segment ``N+1``. Each detection writes one row.
2. **Band-edge leakage.** A wave packet whose peak period sits near an
   octave boundary can be classified as one band in segment ``N`` (e.g.
   QP30 at 39 min) and the next band over in segment ``N+1`` (e.g. QP60
   at 41 min). A band-keyed dedup would let both rows through; the
   period-proximity dedup below catches them.

A single time-windowed, period-tolerant cluster pass over the full
catalogue catches both. We tag rather than collapse so the original
detector output remains losslessly recoverable.

Two events are considered duplicates when

- ``|Δpeak_time| ≤ dt_sec``  (default 7200 s = 2 h, mirroring the detector),
- ``|Δperiod_min| / period_min ≤ period_rel_tol`` (default 10 %).

Bands enter only as labels on the surviving rows, never as a clustering
axis — consistent with the band-agnostic in-segment detector
(:mod:`qp.events.detector`).

Within a cluster the row with the largest ``q_factor`` is kept; the others
have ``is_duplicate = True``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qp.events.detector import DEDUP_WINDOW_SEC as DEFAULT_DT_SEC

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


DEFAULT_PERIOD_REL_TOL: float = 0.10


def tag_duplicates(
    df: pd.DataFrame,
    *,
    dt_sec: float = DEFAULT_DT_SEC,
    period_rel_tol: float = DEFAULT_PERIOD_REL_TOL,
) -> pd.DataFrame:
    """Return a copy of ``df`` with an added boolean ``is_duplicate`` column.

    The input is not mutated. The returned frame preserves row order.
    Two rows are clustered when their peak times differ by ``≤ dt_sec``
    AND their periods differ by ``≤ period_rel_tol`` (relative). Within
    a cluster the row with the largest ``q_factor`` is kept (``False``);
    the rest are flagged ``True``. Ties in ``q_factor`` break by lower
    row index.

    Parameters
    ----------
    df : pandas.DataFrame
        Event catalogue with at least ``peak_time``, ``period_min``,
        ``q_factor`` columns. ``peak_time`` may be ISO-string or
        datetime; it is coerced to datetime internally. A ``band``
        column may be present but is no longer used for clustering.
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
    times_s = peak.astype("int64").to_numpy() / 1e9
    periods = out["period_min"].to_numpy(dtype=float)
    qfactors = out["q_factor"].to_numpy(dtype=float)

    # Greedy time-sorted cluster pass. For each candidate, walk back
    # through *kept* peaks within `dt_sec` and check the period
    # tolerance. The kept list (`kept_idx`) is itself time-sorted, so
    # we can stop as soon as the time gap exceeds the window.
    kept_idx: list[int] = []
    is_dup = [False] * len(out)

    for i in order:
        i_int = int(i)
        # Walk backwards through the kept tail until either a duplicate
        # match is found or the time gap exceeds dt_sec.
        match_pos: int | None = None
        for tail_pos in range(len(kept_idx) - 1, -1, -1):
            rep = kept_idx[tail_pos]
            dt_gap = abs(times_s[i_int] - times_s[rep])
            if dt_gap > dt_sec:
                break
            rel = abs(periods[i_int] - periods[rep]) / max(periods[rep], 1e-9)
            if rel <= period_rel_tol:
                match_pos = tail_pos
                break
        if match_pos is None:
            kept_idx.append(i_int)
            continue
        rep = kept_idx[match_pos]
        if qfactors[i_int] > qfactors[rep]:
            # Promote the new row, demote the previous rep.
            is_dup[rep] = True
            kept_idx[match_pos] = i_int
        else:
            is_dup[i_int] = True

    out["is_duplicate"] = is_dup
    return out


def collapse_duplicates(
    df: pd.DataFrame,
    *,
    dt_sec: float = DEFAULT_DT_SEC,
    period_rel_tol: float = DEFAULT_PERIOD_REL_TOL,
) -> pd.DataFrame:
    """Tag-and-filter convenience wrapper around :func:`tag_duplicates`.

    Returns a new frame with duplicate rows removed and the
    ``is_duplicate`` column dropped. Row order is preserved.
    """
    tagged = tag_duplicates(df, dt_sec=dt_sec, period_rel_tol=period_rel_tol)
    kept = tagged.loc[~tagged["is_duplicate"]].drop(columns=["is_duplicate"])
    return kept.reset_index(drop=True)
