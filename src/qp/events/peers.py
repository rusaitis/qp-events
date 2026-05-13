r"""Per-event peer bookkeeping for the round-8 catalogue.

A "peer" of an event is another non-duplicate detection in the *same*
36-h MFA segment whose detection window overlaps this event's window by
at least a configured fraction of the shorter window. Peers carry raw
period and event-id information only — no band labels. Band-aware
views of multi-harmonic co-occurrence (e.g. "QP60 + QP120 event") are
derived at analysis/view time by :func:`derive_co_bands`.

This pulls band structure out of the catalogue layer: storage records
who overlapped whom and at what period, and downstream code chooses
the band scheme (paper QP15/30/60/120, log-spaced fine bins, sliding
windows, ...) without rerunning the tagging step.

Overlap criterion
-----------------
For two detections :math:`a, b` with windows :math:`[t_a^0, t_a^1)` and
:math:`[t_b^0, t_b^1)`:

.. math::

   \mathrm{overlap} &= \max\bigl(0,\;
                          \min(t_a^1, t_b^1) - \max(t_a^0, t_b^0)\bigr) \\
   \mathrm{frac}    &= \frac{\mathrm{overlap}}{\min(t_a^1 - t_a^0,\;
                                                    t_b^1 - t_b^0)}

``b`` is a peer of ``a`` iff ``frac >= min_overlap_frac``. Same-band
peers are recorded; band membership plays no part in the criterion.
Rows flagged ``is_duplicate=True`` are excluded both as candidates and
as recipients (they represent the same physical packet caught twice in
the detector, not an independent harmonic).

Default ``min_overlap_frac = 0.5`` admits genuinely co-temporal
harmonics (typical wave-train durations 4–6 h) and rejects edge-brush
coincidences. Setting it to ``0.0`` recovers the legacy any-overlap
rule used by the deprecated ``co_bands`` tagger.

Output columns
--------------
``tag_peers`` adds three list-typed columns aligned by position:

==================  ==================  ===========================
Column              Type                Description
==================  ==================  ===========================
peer_event_ids      list[int64]         peer ``event_id`` values
peer_periods_min    list[float64]       peer ``period_min`` values
peer_overlap_frac   list[float64]       overlap fraction in [τ, 1]
==================  ==================  ===========================

Peers are sorted by ``event_id`` ascending for stable round-trip
behaviour. Empty lists when no peers — never NaN/None — so column
dtypes stay stable downstream.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable

from qp.events.bands import classify_period

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


DEFAULT_MIN_OVERLAP_FRAC: float = 0.5


def tag_peers(
    df: "pd.DataFrame",
    *,
    min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
) -> "pd.DataFrame":
    r"""Return a copy of ``df`` with peer columns added.

    Three new columns are appended (see module docstring for the
    overlap definition):

    - ``peer_event_ids``  — ``list[int]``
    - ``peer_periods_min`` — ``list[float]``
    - ``peer_overlap_frac`` — ``list[float]``

    Parameters
    ----------
    df : pandas.DataFrame
        Event catalogue. Must contain ``event_id``, ``segment_idx``,
        ``date_from``, ``date_to``, ``period_min``. The optional
        ``is_duplicate`` column, when present, is honoured: duplicate
        rows are skipped both as peer candidates and as peer recipients
        (they receive empty peer lists).
    min_overlap_frac : float, default 0.5
        Lower bound on ``overlap / min(duration_a, duration_b)`` for
        two rows to be considered peers. Must lie in [0, 1].

    Returns
    -------
    pandas.DataFrame
        A copy of the input with the three peer columns added.
        Row order is preserved.

    Raises
    ------
    ValueError
        If ``min_overlap_frac`` is outside ``[0, 1]`` or a required
        column is missing.
    """
    import pandas as pd

    if not 0.0 <= min_overlap_frac <= 1.0:
        raise ValueError(
            f"min_overlap_frac must lie in [0, 1]; got {min_overlap_frac!r}"
        )

    required = ("event_id", "segment_idx", "date_from", "date_to", "period_min")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = df.copy()
    n = len(out)
    empty_ids: list[list[int]] = [[] for _ in range(n)]
    empty_periods: list[list[float]] = [[] for _ in range(n)]
    empty_fracs: list[list[float]] = [[] for _ in range(n)]
    if n == 0:
        out["peer_event_ids"] = empty_ids
        out["peer_periods_min"] = empty_periods
        out["peer_overlap_frac"] = empty_fracs
        return out

    t_from = pd.to_datetime(out["date_from"]).astype("datetime64[ns]")
    t_to = pd.to_datetime(out["date_to"]).astype("datetime64[ns]")
    t_from_ns = t_from.astype("int64").to_numpy()
    t_to_ns = t_to.astype("int64").to_numpy()
    duration_ns = t_to_ns - t_from_ns
    seg = out["segment_idx"].to_numpy()
    event_id = out["event_id"].to_numpy()
    period_min = out["period_min"].to_numpy(dtype=float)
    is_dup = out["is_duplicate"].to_numpy() if "is_duplicate" in out.columns else None

    # Group row positions by segment so the pairwise scan is local.
    by_seg: dict[int, list[int]] = {}
    for i, s in enumerate(seg):
        by_seg.setdefault(int(s), []).append(i)

    for indices in by_seg.values():
        if len(indices) < 2:
            continue
        for i in indices:
            if is_dup is not None and bool(is_dup[i]):
                continue
            peers: list[tuple[int, float, float]] = []
            for j in indices:
                if j == i:
                    continue
                if is_dup is not None and bool(is_dup[j]):
                    continue
                # Half-open interval overlap; touching windows have
                # zero overlap, which fails any positive min_overlap_frac
                # but is still admitted when min_overlap_frac == 0
                # only if both durations are zero (degenerate).
                lo = max(t_from_ns[i], t_from_ns[j])
                hi = min(t_to_ns[i], t_to_ns[j])
                overlap_ns = hi - lo
                if overlap_ns <= 0:
                    continue
                shorter_ns = min(duration_ns[i], duration_ns[j])
                if shorter_ns <= 0:
                    continue
                frac = float(overlap_ns) / float(shorter_ns)
                if frac < min_overlap_frac:
                    continue
                peers.append((int(event_id[j]), float(period_min[j]), frac))
            if not peers:
                continue
            peers.sort(key=lambda t: t[0])
            empty_ids[i] = [p[0] for p in peers]
            empty_periods[i] = [p[1] for p in peers]
            empty_fracs[i] = [p[2] for p in peers]

    out["peer_event_ids"] = empty_ids
    out["peer_periods_min"] = empty_periods
    out["peer_overlap_frac"] = empty_fracs
    return out


def _default_classifier(period_min: float) -> str | None:
    """Map a period in minutes to a QP band name via :func:`classify_period`."""
    return classify_period(period_min * 60.0)


def derive_co_bands(
    peer_periods_min: Iterable[float],
    *,
    classifier: Callable[[float], str | None] | None = None,
    exclude_self_band: str | None = None,
) -> str:
    r"""Derive a sorted comma-separated co-band string from peer periods.

    View-stage helper. The catalogue stores raw peer periods; callers
    that want a band-label view (per the paper QP scheme or a custom
    scheme) call this on one row's ``peer_periods_min`` and pass an
    appropriate classifier.

    Parameters
    ----------
    peer_periods_min : iterable of float
        Periods of this row's peers, in minutes.
    classifier : callable, optional
        ``classifier(period_min) -> band_name | None``. Defaults to
        :func:`qp.events.bands.classify_period` (wrapped to take minutes).
    exclude_self_band : str, optional
        If given, peers classified into this band are dropped from the
        output — reproduces the legacy ``co_bands`` semantics where the
        row's own band was excluded.

    Returns
    -------
    str
        Sorted comma-separated unique band names, or ``""`` when no
        peer falls into a known band.
    """
    classify = classifier if classifier is not None else _default_classifier
    bands: set[str] = set()
    for p in peer_periods_min:
        name = classify(float(p))
        if name is None:
            continue
        if exclude_self_band is not None and name == exclude_self_band:
            continue
        bands.add(name)
    return ",".join(sorted(bands))
