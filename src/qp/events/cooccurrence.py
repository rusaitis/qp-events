r"""Multi-harmonic co-occurrence bookkeeping for round-8 events.

The detector emits one row per band per 36-h segment, so a wave train
that excites both the QP60 and QP120 FLR harmonics simultaneously
produces two parquet rows sharing the same ``segment_idx``. Nothing
in the schema flags that — downstream consumers can't tell a "pure
QP60" event from a QP60 that co-exists with a QP120 sibling.

This module adds a ``co_bands`` string column to the catalogue: a
sorted, comma-separated list of *other* bands whose
``[date_from, date_to]`` window overlaps this row's window, restricted
to rows in the same ``segment_idx``. Duplicate rows (``is_duplicate``)
are excluded from the sibling lookup since they represent the same
physical wave packet, not an independent harmonic.

The column is empty for events with no overlapping siblings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


def tag_co_bands(df: "pd.DataFrame") -> "pd.DataFrame":
    """Return a copy of ``df`` with an added ``co_bands`` string column.

    For each event row, ``co_bands`` is the sorted comma-separated list
    of unique band names found among other (non-duplicate) rows in the
    same ``segment_idx`` whose ``[date_from, date_to]`` window overlaps
    this row's window. The event's own band is excluded.

    Empty string if no co-occurring sibling exists.

    Parameters
    ----------
    df : pandas.DataFrame
        Event catalogue. Must have ``segment_idx``, ``band``,
        ``date_from``, ``date_to``. ``is_duplicate`` is honoured if
        present (duplicate rows are skipped as siblings AND keep their
        own ``co_bands`` empty since they don't represent an
        independent detection).
    """
    import pandas as pd

    out = df.copy()
    if len(out) == 0:
        out["co_bands"] = pd.Series([], dtype="string")
        return out

    date_from = pd.to_datetime(out["date_from"]).astype("datetime64[ns]")
    date_to = pd.to_datetime(out["date_to"]).astype("datetime64[ns]")
    seg = out["segment_idx"].to_numpy()
    band = out["band"].to_numpy()
    is_dup = out["is_duplicate"].to_numpy() if "is_duplicate" in out.columns else None
    t_from = date_from.astype("int64").to_numpy()
    t_to = date_to.astype("int64").to_numpy()

    co_bands: list[str] = [""] * len(out)

    # Group by segment to keep the comparison O(N_seg * k^2) where k is
    # the per-segment event count (usually 1–5). The catalogue has 1881
    # rows and ~1423 segments, so this is essentially linear.
    by_seg: dict[int, list[int]] = {}
    for i, s in enumerate(seg):
        by_seg.setdefault(int(s), []).append(i)

    for indices in by_seg.values():
        if len(indices) < 2:
            continue
        # Mark each row's siblings.
        for i in indices:
            if is_dup is not None and is_dup[i]:
                continue
            siblings: set[str] = set()
            for j in indices:
                if i == j:
                    continue
                if is_dup is not None and is_dup[j]:
                    continue
                # half-open overlap: [a0, a1) ∩ [b0, b1) ≠ ∅
                if t_from[i] < t_to[j] and t_from[j] < t_to[i]:
                    if band[j] != band[i]:
                        siblings.add(str(band[j]))
            if siblings:
                co_bands[i] = ",".join(sorted(siblings))

    out["co_bands"] = pd.Series(co_bands, index=out.index, dtype="string")
    return out
