r"""Bin cumulative QP event time onto the dwell-grid coordinates.

Phase 4 of the plan. The output of this module is the **numerator**
for the dwell-normalized occurrence-rate maps in Figs 7, 8 of the
paper. The denominator (cumulative spacecraft dwell time per cell)
already lives in ``Output/dwell_grid_cassini_saturn.zarr``; the bin
edges here must match it exactly.

Two binning strategies are exposed:

1. :func:`bin_events_peak_position` — fast, single-bin assignment.
   Uses the event's stored peak ``(r, mag_lat, LT)`` and dumps the
   entire event duration into that one cell. This is accurate when
   the spacecraft moves slowly compared to the event duration
   (typically true outside ~10 R_S where Cassini takes >> 4 h to
   cross a 1° latitude × 1 R_S radial cell).

2. :func:`bin_events_walking` — slower, accurate. Loads the MFA
   segments file, walks each event minute-by-minute through the
   segment's pre-computed positions, and accumulates one minute per
   matching cell. Used for the production grid where periapsis
   passes matter.

Both produce a dict of 3D ``numpy.ndarray`` of minutes, one per QP
band plus a ``"total"`` field. The driver wraps this in an
``xarray.Dataset`` and persists to
``Output/event_time_grid_v1.zarr``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from qp.dwell.grid import DwellGridConfig
from qp.events.bands import QP_BAND_NAMES
from qp.events.catalog import WaveEvent


# ----------------------------------------------------------------------
# Bin index helper (mirrors qp.dwell.grid._bin_index)
# ----------------------------------------------------------------------


def _bin_index(value: float, vmin: float, vmax: float, n: int) -> int:
    frac = (value - vmin) / (vmax - vmin)
    idx = int(math.floor(frac * n))
    return max(0, min(n - 1, idx))


def _in_range(
    r: float, lat: float, lt: float, config: DwellGridConfig,
) -> bool:
    return (
        config.r_range[0] <= r < config.r_range[1]
        and config.lat_range[0] <= lat < config.lat_range[1]
        and config.lt_range[0] <= lt < config.lt_range[1]
    )


# ----------------------------------------------------------------------
# Strategy 1 — peak-position binning
# ----------------------------------------------------------------------


@dataclass
class BinningStats:
    """Per-run accounting of how many events were placed in the grid."""

    n_total: int = 0
    n_binned: int = 0
    n_missing_coords: int = 0
    n_out_of_range: int = 0

    @property
    def fraction_binned(self) -> float:
        return self.n_binned / max(self.n_total, 1)


def bin_events_peak_position(
    events: Iterable[WaveEvent],
    config: DwellGridConfig | None = None,
    *,
    quality_weighted: bool = False,
    quality_col: str = "quality",
) -> tuple[dict[str, np.ndarray], BinningStats]:
    r"""Single-bin event-time binning.

    Each event's full duration is dumped into the cell containing its
    stored ``(r_distance, mag_lat, local_time)`` peak position. The
    units are **minutes**, matching the dwell zarr.

    Parameters
    ----------
    quality_weighted : bool, default False
        When True, each event contributes ``quality × duration_minutes``
        instead of ``duration_minutes``. The resulting grids represent
        :math:`\sum q_i \cdot \Delta t_i` per cell and must be
        normalized by the same dwell grid as the unweighted case.
    quality_col : str
        Attribute name for the quality score on each event.

    Returns
    -------
    grids : dict[str, np.ndarray]
        Keys: ``"QP30"``, ``"QP60"``, ``"QP120"``, ``"total"``.
        Each value is a 3D float array of shape
        ``(n_r, n_lat, n_lt)`` from the supplied :class:`DwellGridConfig`.
    stats : BinningStats
    """
    if config is None:
        config = DwellGridConfig()

    grids: dict[str, np.ndarray] = {
        b: np.zeros(config.shape, dtype=np.float64)
        for b in QP_BAND_NAMES
    }
    grids["total"] = np.zeros(config.shape, dtype=np.float64)

    stats = BinningStats()
    for ev in events:
        stats.n_total += 1
        if (
            ev.r_distance is None
            or ev.mag_lat is None
            or ev.local_time is None
            or ev.band is None
        ):
            stats.n_missing_coords += 1
            continue
        r = float(ev.r_distance)
        lat = float(ev.mag_lat)
        lt = float(ev.local_time)
        if not _in_range(r, lat, lt, config):
            stats.n_out_of_range += 1
            continue
        i_r = _bin_index(r, *config.r_range, config.n_r)
        i_lat = _bin_index(lat, *config.lat_range, config.n_lat)
        i_lt = _bin_index(lt, *config.lt_range, config.n_lt)
        minutes = float(ev.duration_minutes)
        if quality_weighted:
            q = getattr(ev, quality_col, None)
            weight = float(q) if q is not None and math.isfinite(q) else 0.0
            weight = max(0.0, min(1.0, weight))
            minutes = minutes * weight
        if ev.band in grids:
            grids[ev.band][i_r, i_lat, i_lt] += minutes
        grids["total"][i_r, i_lat, i_lt] += minutes
        stats.n_binned += 1

    return grids, stats


# ----------------------------------------------------------------------
# Strategy 2 — walking binning (uses segment positions)
# ----------------------------------------------------------------------


@dataclass
class SegmentPositions:
    """Minimal per-segment data needed for walking binning.

    The optional ``central_mask`` is a boolean array marking the
    minutes that belong to the segment's central 24-hour window.
    Setting it ensures events and dwell are accumulated only over
    the segment's "owned" minutes — eliminating the double-counting
    that arises when adjacent 36-h segments overlap by 12 h.
    """

    seg_idx: int
    times_unix: np.ndarray  # (N,) float seconds
    r: np.ndarray  # (N,) R_S
    mag_lat: np.ndarray  # (N,) degrees
    local_time: np.ndarray  # (N,) hours
    central_mask: np.ndarray | None = None  # (N,) bool


def bin_events_walking(
    events: Iterable[WaveEvent],
    segment_positions: dict[int, SegmentPositions],
    config: DwellGridConfig | None = None,
) -> tuple[dict[str, np.ndarray], BinningStats]:
    r"""Walking event-time binning.

    For each event, walks the matching segment's position arrays
    minute-by-minute between ``date_from`` and ``date_to`` and
    accumulates ``+1.0`` minute per visited bin. This handles
    spacecraft motion within an event correctly, which matters near
    periapsis.

    Per-band grids accumulate independently. The ``"total"`` grid is
    the **union** of all band masks per segment — i.e. minutes during
    which *any* QP band fired count once, not once per band. This
    ensures ``event_time_total ≤ dwell`` per cell.

    ``segment_positions`` is a dict keyed by ``segment_id``; events
    whose ``segment_id`` is missing fall back to the peak-position
    strategy for that single event. The fallback is accounted for
    in :class:`BinningStats`.

    Returns
    -------
    grids : dict[str, np.ndarray]
    stats : BinningStats
    """
    if config is None:
        config = DwellGridConfig()

    grids: dict[str, np.ndarray] = {
        b: np.zeros(config.shape, dtype=np.float64)
        for b in QP_BAND_NAMES
    }
    grids["total"] = np.zeros(config.shape, dtype=np.float64)

    stats = BinningStats()
    epoch = np.datetime64("1970-01-01T00:00:00")

    # Phase 1: for each segment, build per-minute boolean masks for
    # each band so we can compute the union ("total") correctly.
    seg_event_masks: dict[int, dict[str, np.ndarray]] = {}

    events_list = list(events)
    for ev in events_list:
        seg_id = ev.segment_id
        if seg_id is None or seg_id not in segment_positions:
            continue
        if ev.band is None:
            continue
        sp = segment_positions[seg_id]
        n = sp.times_unix.size
        if seg_id not in seg_event_masks:
            seg_event_masks[seg_id] = {
                b: np.zeros(n, dtype=bool) for b in QP_BAND_NAMES
            }
        t_from = (
            np.datetime64(ev.date_from) - epoch
        ).astype("timedelta64[s]").astype(float)
        t_to = (
            np.datetime64(ev.date_to) - epoch
        ).astype("timedelta64[s]").astype(float)
        time_mask = (sp.times_unix >= t_from) & (sp.times_unix <= t_to)
        if sp.central_mask is not None:
            time_mask &= sp.central_mask
        seg_event_masks[seg_id][ev.band] |= time_mask

    # Phase 2: accumulate per-band grids and the union total grid
    # using the masks built above.
    for ev in events_list:
        stats.n_total += 1
        if ev.band is None:
            stats.n_missing_coords += 1
            continue

        seg_id = ev.segment_id
        if seg_id is None or seg_id not in segment_positions:
            # Fallback to peak position
            if (
                ev.r_distance is None
                or ev.mag_lat is None
                or ev.local_time is None
            ):
                stats.n_missing_coords += 1
                continue
            r = float(ev.r_distance)
            lat = float(ev.mag_lat)
            lt = float(ev.local_time)
            if not _in_range(r, lat, lt, config):
                stats.n_out_of_range += 1
                continue
            i_r = _bin_index(r, *config.r_range, config.n_r)
            i_lat = _bin_index(lat, *config.lat_range, config.n_lat)
            i_lt = _bin_index(lt, *config.lt_range, config.n_lt)
            minutes = float(ev.duration_minutes)
            if ev.band in grids:
                grids[ev.band][i_r, i_lat, i_lt] += minutes
            grids["total"][i_r, i_lat, i_lt] += minutes
            stats.n_binned += 1
            continue

        sp = segment_positions[seg_id]
        t_from = (
            np.datetime64(ev.date_from) - epoch
        ).astype("timedelta64[s]").astype(float)
        t_to = (
            np.datetime64(ev.date_to) - epoch
        ).astype("timedelta64[s]").astype(float)
        mask = (sp.times_unix >= t_from) & (sp.times_unix <= t_to)
        if sp.central_mask is not None:
            mask = mask & sp.central_mask
        if not mask.any():
            stats.n_missing_coords += 1
            continue

        contribution = _accumulate_mask(sp, mask, config, stats)
        if contribution is None:
            continue
        if ev.band in grids:
            grids[ev.band] += contribution
        stats.n_binned += 1

    # Phase 3: accumulate the union "total" grid from the per-segment
    # band masks. Each minute that has *any* band firing contributes 1.
    for seg_id, band_masks in seg_event_masks.items():
        sp = segment_positions[seg_id]
        any_mask = np.zeros_like(next(iter(band_masks.values())))
        for m in band_masks.values():
            any_mask |= m
        if not any_mask.any():
            continue
        contribution = _accumulate_mask(sp, any_mask, config, None)
        if contribution is not None:
            grids["total"] += contribution

    return grids, stats


def _accumulate_mask(
    sp: SegmentPositions,
    mask: np.ndarray,
    config: DwellGridConfig,
    stats: BinningStats | None,
) -> np.ndarray | None:
    """Helper: convert a boolean time mask + segment positions into a
    3D contribution grid (in minutes)."""
    rs = sp.r[mask]
    lats = sp.mag_lat[mask]
    lts = sp.local_time[mask]

    finite = np.isfinite(rs) & np.isfinite(lats) & np.isfinite(lts)
    rs, lats, lts = rs[finite], lats[finite], lts[finite]
    if rs.size == 0:
        if stats is not None:
            stats.n_missing_coords += 1
        return None

    in_r = (
        (rs >= config.r_range[0]) & (rs < config.r_range[1])
        & (lats >= config.lat_range[0]) & (lats < config.lat_range[1])
        & (lts >= config.lt_range[0]) & (lts < config.lt_range[1])
    )
    rs, lats, lts = rs[in_r], lats[in_r], lts[in_r]
    if rs.size == 0:
        if stats is not None:
            stats.n_out_of_range += 1
        return None

    i_r = np.clip(
        np.floor(
            (rs - config.r_range[0])
            / (config.r_range[1] - config.r_range[0]) * config.n_r,
        ).astype(int), 0, config.n_r - 1,
    )
    i_lat = np.clip(
        np.floor(
            (lats - config.lat_range[0])
            / (config.lat_range[1] - config.lat_range[0]) * config.n_lat,
        ).astype(int), 0, config.n_lat - 1,
    )
    i_lt = np.clip(
        np.floor(
            (lts - config.lt_range[0])
            / (config.lt_range[1] - config.lt_range[0]) * config.n_lt,
        ).astype(int), 0, config.n_lt - 1,
    )

    flat = np.ravel_multi_index((i_r, i_lat, i_lt), config.shape)
    counts = np.bincount(flat, minlength=math.prod(config.shape))
    return counts.astype(np.float64).reshape(config.shape)


# ----------------------------------------------------------------------
# Companion dwell grid built with the same approximations as the
# event binner.
#
# Why this exists: the canonical dwell grid in
# Output/dwell_grid_cassini_saturn.zarr was built from per-minute KSM
# positions (offset-dipole magnetic latitude, KSM local time),
# whereas the event binner here uses KRTP latitude and per-segment
# median LT. The two coordinate conventions disagree by enough to
# make the per-cell ratio event_time / dwell_time exceed 1 in some
# cells, which is unphysical.
#
# Building a parallel "consistency" dwell grid with the same
# approximations as the events guarantees event_time ≤ dwell_time
# per cell by construction. It is what Phase 5 should divide by.
# ----------------------------------------------------------------------


def accumulate_segment_dwell(
    segment_positions: dict[int, "SegmentPositions"],
    config: DwellGridConfig | None = None,
) -> np.ndarray:
    r"""Accumulate per-minute dwell time using the **same** coordinates
    as :func:`bin_events_walking`.

    Walks every minute of every segment and adds 1.0 minute to the
    matching ``(r, mag_lat, LT)`` cell. If a segment has a
    ``central_mask`` set, only the central minutes contribute, which
    is what guarantees event_time ≤ dwell per cell when the events
    are also clipped to the central window.

    Returns the 3D grid in minutes.
    """
    if config is None:
        config = DwellGridConfig()
    grid = np.zeros(config.shape, dtype=np.float64)
    for sp in segment_positions.values():
        rs, lats, lts = sp.r, sp.mag_lat, sp.local_time
        if sp.central_mask is not None:
            rs = rs[sp.central_mask]
            lats = lats[sp.central_mask]
            lts = lts[sp.central_mask]
        finite = np.isfinite(rs) & np.isfinite(lats) & np.isfinite(lts)
        rs, lats, lts = rs[finite], lats[finite], lts[finite]
        in_r = (
            (rs >= config.r_range[0]) & (rs < config.r_range[1])
            & (lats >= config.lat_range[0]) & (lats < config.lat_range[1])
            & (lts >= config.lt_range[0]) & (lts < config.lt_range[1])
        )
        rs, lats, lts = rs[in_r], lats[in_r], lts[in_r]
        if rs.size == 0:
            continue
        i_r = np.clip(
            np.floor(
                (rs - config.r_range[0])
                / (config.r_range[1] - config.r_range[0]) * config.n_r,
            ).astype(int), 0, config.n_r - 1,
        )
        i_lat = np.clip(
            np.floor(
                (lats - config.lat_range[0])
                / (config.lat_range[1] - config.lat_range[0]) * config.n_lat,
            ).astype(int), 0, config.n_lat - 1,
        )
        i_lt = np.clip(
            np.floor(
                (lts - config.lt_range[0])
                / (config.lt_range[1] - config.lt_range[0]) * config.n_lt,
            ).astype(int), 0, config.n_lt - 1,
        )
        flat = np.ravel_multi_index(
            (i_r, i_lat, i_lt), config.shape,
        )
        counts = np.bincount(flat, minlength=math.prod(config.shape))
        grid += counts.astype(np.float64).reshape(config.shape)
    return grid


# ----------------------------------------------------------------------
# 2D pre-aggregation (LT × mag_lat) for fast plotting
# ----------------------------------------------------------------------


def aggregate_lt_mag_lat(grid_3d: np.ndarray) -> np.ndarray:
    r"""Sum a 3D ``(r, mag_lat, LT)`` grid over the radial axis.

    Returns shape ``(n_lat, n_lt)``.
    """
    return grid_3d.sum(axis=0)


# ----------------------------------------------------------------------
# xarray wrapper
# ----------------------------------------------------------------------


def grids_to_xarray(
    grids: dict[str, np.ndarray],
    config: DwellGridConfig,
    *,
    title: str = "QP event time grid",
    description: str | None = None,
    extra_attrs: dict | None = None,
) -> "xarray.Dataset":  # noqa: F821
    r"""Wrap a dict of 3D grids in an :class:`xarray.Dataset` matching
    the dwell zarr layout.

    The dataset uses the same dimension names as
    ``Output/dwell_grid_cassini_saturn.zarr`` so the two files can be
    sliced and divided side-by-side.
    """
    import xarray as xr

    coords = {
        "r": config.r_centers,
        "magnetic_latitude": config.lat_centers,
        "local_time": config.lt_centers,
        "r_edges": config.r_edges,
        "lat_edges": config.lat_edges,
        "lt_edges": config.lt_edges,
    }
    data_vars: dict[str, tuple] = {}
    for name, arr in grids.items():
        data_vars[f"event_time_{name}"] = (
            ("r", "magnetic_latitude", "local_time"),
            arr.astype(np.float32),
        )
        # 2D pre-aggregation (LT × mag_lat) for Fig 8 fast path
        agg = aggregate_lt_mag_lat(arr)
        data_vars[f"event_time_{name}_lt_mag_lat"] = (
            ("magnetic_latitude", "local_time"),
            agg.astype(np.float32),
        )

    attrs = {
        "title": title,
        "units": "minutes",
        "n_r": config.n_r,
        "n_lat": config.n_lat,
        "n_lt": config.n_lt,
    }
    if description:
        attrs["description"] = description
    if extra_attrs:
        attrs.update(extra_attrs)

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def save_event_time_zarr(
    grids: dict[str, np.ndarray],
    config: DwellGridConfig,
    path: Path,
    **xr_kwargs,
) -> Path:
    r"""Persist event-time grids to a zarr matching the dwell layout."""
    ds = grids_to_xarray(grids, config, **xr_kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Remove the old zarr to avoid mixing schemas
        import shutil
        shutil.rmtree(path)
    ds.to_zarr(path, mode="w", consolidated=False)
    return path
