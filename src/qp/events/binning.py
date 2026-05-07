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

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

from qp.dwell.grid import DwellGridConfig

if TYPE_CHECKING:
    import xarray as xr

log = logging.getLogger(__name__)
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


# ----------------------------------------------------------------------
# Full-mirror schema — for each period band, accumulate event time on
# the same axes and region splits as the canonical dwell grid at
# Output/dwell_grid_cassini_saturn.zarr. The variable names mirror the
# dwell-grid names with a ``<band>_`` prefix so the ratio computation
# is one indexing step:
#
#     event_time_per_bin = ev_zarr[f"{band}_kmag_inv_lat_closed_magnetosphere"]
#     dwell_per_bin      = dw_zarr["kmag_inv_lat_closed_magnetosphere"]
#     occurrence_rate    = event_time_per_bin / dwell_per_bin
#
# Three of the dwell schemas can be reproduced from the spacecraft
# trajectory alone (region splits, dipole invariant latitude, weak-
# field plasma-sheet proxy). The fourth — KMAG-traced invariant
# latitude — requires per-sample field-line tracing and is left for a
# follow-up.
# ----------------------------------------------------------------------


@dataclass
class SegmentPositionsExt(SegmentPositions):
    """Per-segment positions enriched with KSM cartesian + region codes.

    Adds the per-minute spacecraft Cartesian position (KSM, R_S),
    region code (Jackman 2019 crossings), and total field magnitude
    (nT). Region codes follow ``qp.dwell.regions.REGION_CODES``:
    ``0=magnetosphere, 1=magnetosheath, 2=solar_wind, 9=unknown``. The
    weak-field grid uses ``b_total_nT`` directly.
    """

    ksm_x: np.ndarray | None = None  # (N,) R_S
    ksm_y: np.ndarray | None = None  # (N,) R_S
    ksm_z: np.ndarray | None = None  # (N,) R_S
    region_codes: np.ndarray | None = None  # (N,) int (0/1/2/9)
    b_total_nT: np.ndarray | None = None  # (N,) float


def accumulate_full_mirror(
    events: Iterable[WaveEvent],
    segment_positions: dict[int, SegmentPositionsExt],
    *,
    bands: list[str] | None = None,
    band_for_period_min: Callable[[float], str | None] | None = None,
    config: DwellGridConfig | None = None,
    b_threshold_nT: float = 2.0,
) -> dict[str, np.ndarray]:
    r"""Accumulate per-band event time on every dwell-grid schema.

    Returns a flat dict of arrays, prefixed by band name. For each
    band ``B`` and the band-union ``"total"``, four families of
    variables are produced:

    1. 3D ``(r, magnetic_latitude, local_time)``, with region splits:
       ``B_total``, ``B_magnetosphere``, ``B_magnetosheath``,
       ``B_solar_wind``, ``B_unknown``.
    2. 2D ``(dipole_inv_lat, local_time)`` with the same region splits:
       ``B_dipole_inv_lat_total``, ``B_dipole_inv_lat_magnetosphere``,
       ...
    3. 2D ``(dipole_inv_lat, local_time)`` plasma-sheet proxy
       (``|B| < b_threshold_nT``):
       ``B_weak_field_total``, ``B_weak_field_magnetosphere``, ...

    All arrays are float64 minutes. The KMAG-traced invariant-latitude
    schemas are *not* produced here — they require per-sample
    field-line tracing and are added by a follow-up that consumes a
    sample-cache from :mod:`qp.fieldline.tracer`.

    Parameters
    ----------
    events : iterable of WaveEvent
        Detections to bin.
    segment_positions : dict[int, SegmentPositionsExt]
        Per-segment trajectory and region information. Events whose
        segment id is missing are dropped (counted in :class:`BinningStats`
        — but stats aren't returned here; use ``bin_events_walking``
        for fallback peak-position binning).
    bands : list[str], optional
        Band names to populate. Defaults to ``QP_BAND_NAMES``.
    band_for_period_min : callable, optional
        Function ``(period_min: float) -> str | None`` mapping a
        detected period to a band name. Defaults to using the event's
        stored ``band`` attribute (so this argument lets a single
        parquet be re-binned into different band schemes).
    config : DwellGridConfig, optional
        Same axes as the dwell grid. Defaults to
        ``DwellGridConfig()``.
    b_threshold_nT : float, default 2.0
        Plasma-sheet proxy threshold (matches the dwell grid).
    """
    from qp.dwell.grid import (
        accumulate_inv_lat_grid,
        accumulate_weak_field_grid,
        accumulate_with_regions,
    )

    if config is None:
        config = DwellGridConfig()
    if bands is None:
        bands = list(QP_BAND_NAMES)

    use_stored_band = band_for_period_min is None
    events_list = list(events)

    seg_masks: dict[int, dict[str, np.ndarray]] = {}
    epoch = np.datetime64("1970-01-01T00:00:00")
    for ev in events_list:
        seg_id = ev.segment_id
        if seg_id is None or seg_id not in segment_positions:
            continue
        if use_stored_band:
            if ev.band is None or ev.band not in bands:
                continue
            band = ev.band
        else:
            period_min = ev.period_peak_min
            if period_min is None and ev.period is not None:
                period_min = ev.period / 60.0
            if period_min is None:
                continue
            band = band_for_period_min(float(period_min))
            if band is None or band not in bands:
                continue
        sp = segment_positions[seg_id]
        n = sp.times_unix.size
        seg_masks.setdefault(seg_id, {b: np.zeros(n, dtype=bool) for b in bands})
        t_from = (
            np.datetime64(ev.date_from) - epoch
        ).astype("timedelta64[s]").astype(float)
        t_to = (
            np.datetime64(ev.date_to) - epoch
        ).astype("timedelta64[s]").astype(float)
        m = (sp.times_unix >= t_from) & (sp.times_unix <= t_to)
        if sp.central_mask is not None:
            m &= sp.central_mask
        seg_masks[seg_id][band] |= m

    grids: dict[str, np.ndarray] = {}
    region_names = ("total", "magnetosphere", "magnetosheath", "solar_wind", "unknown")
    shape_3d = config.shape
    shape_2d = (config.n_lat, config.n_lt)

    def _zero_grids_for_band(b: str) -> None:
        for r in region_names:
            grids[f"{b}_{r}"] = np.zeros(shape_3d, dtype=np.float64)
            grids[f"{b}_dipole_inv_lat_{r}"] = np.zeros(shape_2d, dtype=np.float64)
            grids[f"{b}_weak_field_{r}"] = np.zeros(shape_2d, dtype=np.float64)

    band_keys = list(bands) + ["total"]
    for b in band_keys:
        _zero_grids_for_band(b)

    for seg_id, band_to_mask in seg_masks.items():
        sp = segment_positions[seg_id]
        if (
            sp.ksm_x is None or sp.ksm_y is None or sp.ksm_z is None
            or sp.region_codes is None
        ):
            log.warning(
                "segment %s missing KSM cartesian or region codes; skipping",
                seg_id,
            )
            continue
        # Per-band masks
        for b in bands:
            mask = band_to_mask[b]
            if not mask.any():
                continue
            x = sp.ksm_x[mask]
            y = sp.ksm_y[mask]
            z = sp.ksm_z[mask]
            codes = sp.region_codes[mask]
            r3d = accumulate_with_regions(x, y, z, codes, 1.0, config)
            r2d = accumulate_inv_lat_grid(x, y, z, 1.0, codes, config)
            for r in region_names:
                grids[f"{b}_{r}"] += r3d[r]
                grids[f"{b}_dipole_inv_lat_{r}"] += r2d[r]
            if sp.b_total_nT is not None:
                bt = sp.b_total_nT[mask]
                rwf = accumulate_weak_field_grid(
                    x, y, z, bt, 1.0, b_threshold_nT, codes, config,
                )
                for r in region_names:
                    grids[f"{b}_weak_field_{r}"] += rwf[r]
        # Band-union mask for the "total" band (no double-counting)
        any_mask = np.zeros_like(next(iter(band_to_mask.values())))
        for m in band_to_mask.values():
            any_mask |= m
        if not any_mask.any():
            continue
        x = sp.ksm_x[any_mask]
        y = sp.ksm_y[any_mask]
        z = sp.ksm_z[any_mask]
        codes = sp.region_codes[any_mask]
        r3d = accumulate_with_regions(x, y, z, codes, 1.0, config)
        r2d = accumulate_inv_lat_grid(x, y, z, 1.0, codes, config)
        for r in region_names:
            grids[f"total_{r}"] += r3d[r]
            grids[f"total_dipole_inv_lat_{r}"] += r2d[r]
        if sp.b_total_nT is not None:
            bt = sp.b_total_nT[any_mask]
            rwf = accumulate_weak_field_grid(
                x, y, z, bt, 1.0, b_threshold_nT, codes, config,
            )
            for r in region_names:
                grids[f"total_weak_field_{r}"] += rwf[r]

    return grids


def full_mirror_grids_to_xarray(
    grids: dict[str, np.ndarray],
    config: DwellGridConfig,
    bands: list[str],
    *,
    title: str = "QP Event Time Grid (full-mirror schema)",
    description: str | None = None,
    extra_attrs: dict | None = None,
) -> "xr.Dataset":
    """Wrap full-mirror grids into an xarray Dataset matching the dwell layout."""
    import xarray as xr

    coords = {
        "r": config.r_centers,
        "magnetic_latitude": config.lat_centers,
        "local_time": config.lt_centers,
        "dipole_inv_lat": config.lat_centers,
        "r_edges": config.r_edges,
        "lat_edges": config.lat_edges,
        "lt_edges": config.lt_edges,
    }
    data_vars: dict[str, tuple] = {}
    region_names = ("total", "magnetosphere", "magnetosheath", "solar_wind", "unknown")
    band_keys = list(bands) + ["total"]
    for b in band_keys:
        for r in region_names:
            arr = grids.get(f"{b}_{r}")
            if arr is not None:
                data_vars[f"{b}_{r}"] = (
                    ("r", "magnetic_latitude", "local_time"),
                    arr.astype(np.float32),
                )
            arr2 = grids.get(f"{b}_dipole_inv_lat_{r}")
            if arr2 is not None:
                data_vars[f"{b}_dipole_inv_lat_{r}"] = (
                    ("dipole_inv_lat", "local_time"),
                    arr2.astype(np.float32),
                )
            arr3 = grids.get(f"{b}_weak_field_{r}")
            if arr3 is not None:
                data_vars[f"{b}_weak_field_{r}"] = (
                    ("dipole_inv_lat", "local_time"),
                    arr3.astype(np.float32),
                )

    attrs = {
        "title": title,
        "units": "minutes",
        "n_r": config.n_r,
        "n_lat": config.n_lat,
        "n_lt": config.n_lt,
        "schema": "full_mirror",
        "kmag_inv_lat_populated": False,
        "bands": list(bands),
    }
    if description:
        attrs["description"] = description
    if extra_attrs:
        attrs.update(extra_attrs)
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


# ----------------------------------------------------------------------
# KMAG-traced event-time schemas — kmag_inv_lat + kmag_eq_r per band.
#
# Implementation strategy: we subset the full trajectory to the union
# of all band masks, hand THAT to compute_invariant_latitudes_parallel
# with trace_every_n=10 (matching the dwell grid), and then accumulate
# per band by reading per-trace band membership at the subsampled
# cadence. This means the tracer only fires on event-window samples,
# so the cost scales with total event minutes, not total mission
# minutes.
# ----------------------------------------------------------------------


def accumulate_kmag_event_grids(
    band_masks: dict[str, np.ndarray],
    x_traj: np.ndarray,
    y_traj: np.ndarray,
    z_traj: np.ndarray,
    t_unix_traj: np.ndarray,
    region_codes_traj: np.ndarray,
    *,
    trace_every_n: int = 10,
    config: DwellGridConfig | None = None,
    tracing_config=None,
    field_config=None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    r"""Trace event-window samples and accumulate KMAG event-time grids.

    For each band ``B`` and the band-union ``"total"``, emit four
    families of variables, each with the standard 5-region split:

    - ``B_kmag_inv_lat_<region>``       (kmag_inv_lat, LT)
    - ``B_kmag_inv_lat_closed_<region>`` (closed lines only)
    - ``B_kmag_eq_r_<region>``          (kmag_eq_r, LT)
    - ``B_kmag_eq_r_closed_<region>``    (closed lines only)

    Parameters
    ----------
    band_masks : dict[str, ndarray]
        Per-band boolean masks over the full trajectory (length N).
    x_traj, y_traj, z_traj : ndarray
        KSM positions at every trajectory minute, R_S.
    t_unix_traj : ndarray
        POSIX timestamps at every trajectory minute. Converted
        internally to J2000 seconds.
    region_codes_traj : ndarray of int
        Region code at every trajectory minute (0/1/2/9).
    trace_every_n : int, default 10
        Subsampling cadence for tracing — must match the dwell grid.
    config : DwellGridConfig, optional
    tracing_config, field_config : qp.dwell.tracing types, optional
        If None, default ``TracingConfig`` / ``SaturnFieldConfig`` are
        used (same defaults as ``compute_dwell_grid.py``).

    Returns
    -------
    grids : dict[str, ndarray]
        Flat dict of accumulated grids, ready for
        :func:`full_mirror_grids_to_xarray_with_kmag`.
    stats : dict[str, int]
        ``{"n_traces": ..., "n_closed": ..., "n_events_traced_min": ...}``
    """
    from qp.coords.ksm import local_time
    from qp.dwell.grid import (
        accumulate_kmag_eq_r_grid,
        accumulate_traced_inv_lat_grid,
    )
    from qp.dwell.tracing import (
        SaturnFieldConfig,
        TracingConfig,
        compute_invariant_latitudes_parallel,
    )

    if config is None:
        config = DwellGridConfig()
    if tracing_config is None:
        tracing_config = TracingConfig(trace_every_n=trace_every_n)
    if field_config is None:
        field_config = SaturnFieldConfig()

    # 1. Union event-mask (any band, any minute) → defines what to trace
    bands = list(band_masks.keys())
    if not bands:
        return {}, {"n_traces": 0, "n_closed": 0, "n_events_traced_min": 0}
    union = np.zeros_like(next(iter(band_masks.values())))
    for m in band_masks.values():
        union |= m
    n_event_min = int(union.sum())
    log.info(
        "KMAG event tracing: %d event-minutes (~%d traces at every-%d cadence)",
        n_event_min, n_event_min // trace_every_n, trace_every_n,
    )

    # 2. Subset trajectory + region codes to the union mask
    keep = np.flatnonzero(union)
    x_e = x_traj[keep]
    y_e = y_traj[keep]
    z_e = z_traj[keep]
    t_e = t_unix_traj[keep]
    codes_e = region_codes_traj[keep]
    # Convert POSIX → J2000 for KMAG
    j2000_posix = 946728000.0
    t_j2000 = t_e - j2000_posix

    # 3. Trace
    result = compute_invariant_latitudes_parallel(
        x_e, y_e, z_e, t_j2000,
        config=tracing_config,
        field_config=field_config,
        region_codes=codes_e,
    )

    # 4. Subsample band masks at the same cadence as the tracer's
    #    internal subsampling (np.arange(0, n_active, trace_every_n)).
    n_active = len(keep)
    sub_idx = np.arange(0, n_active, trace_every_n)
    n_traces = len(sub_idx)
    # Original-trajectory indices of each traced position
    orig_idx_for_trace = keep[sub_idx]

    lt_sub = local_time(x_e[sub_idx], y_e[sub_idx])
    z_sub = z_e[sub_idx]
    codes_sub = codes_e[sub_idx]
    dt_trace = float(trace_every_n)

    # Per-band membership at each traced position
    per_band_mask: dict[str, np.ndarray] = {
        b: band_masks[b][orig_idx_for_trace] for b in bands
    }
    union_mask_sub = np.zeros(n_traces, dtype=bool)
    for m in per_band_mask.values():
        union_mask_sub |= m

    # 5. Accumulate per band — apply the band's mask to lt/z/codes
    #    by giving the accumulator a NaN-filled inv_lat where mask is
    #    False (the validity check drops those).
    grids: dict[str, np.ndarray] = {}

    def _accumulate_for(
        band_key: str, mask_sub: np.ndarray,
    ) -> None:
        if not mask_sub.any():
            empty_inv = np.zeros(
                (config.n_lat, config.n_lt), dtype=np.float64,
            )
            empty_eq = np.zeros((config.n_r, config.n_lt), dtype=np.float64)
            for r in ("total", "magnetosphere", "magnetosheath",
                      "solar_wind", "unknown"):
                grids[f"{band_key}_kmag_inv_lat_{r}"] = empty_inv.copy()
                grids[f"{band_key}_kmag_inv_lat_closed_{r}"] = empty_inv.copy()
                grids[f"{band_key}_kmag_eq_r_{r}"] = empty_eq.copy()
                grids[f"{band_key}_kmag_eq_r_closed_{r}"] = empty_eq.copy()
            return
        inv_n_b = np.where(mask_sub, result.inv_lat_north, np.nan)
        inv_s_b = np.where(mask_sub, result.inv_lat_south, np.nan)
        l_eq_b = np.where(mask_sub, result.l_equatorial, np.nan)
        closed_b = result.is_closed & mask_sub
        # kmag_inv_lat (all + closed-only)
        all_inv = accumulate_traced_inv_lat_grid(
            inv_n_b, inv_s_b, closed_b, lt_sub, z_sub,
            dt_minutes=dt_trace, region_codes=codes_sub, config=config,
        )
        closed_inv = accumulate_traced_inv_lat_grid(
            inv_n_b, inv_s_b, closed_b, lt_sub, z_sub,
            dt_minutes=dt_trace, region_codes=codes_sub,
            closed_only=True, config=config,
        )
        # kmag_eq_r (all + closed-only)
        all_eq = accumulate_kmag_eq_r_grid(
            l_eq_b, closed_b, lt_sub,
            dt_minutes=dt_trace, region_codes=codes_sub, config=config,
        )
        closed_eq = accumulate_kmag_eq_r_grid(
            l_eq_b, closed_b, lt_sub,
            dt_minutes=dt_trace, region_codes=codes_sub,
            closed_only=True, config=config,
        )
        for k, v in all_inv.items():
            grids[f"{band_key}_kmag_inv_lat_{k}"] = v
        for k, v in closed_inv.items():
            grids[f"{band_key}_kmag_inv_lat_closed_{k}"] = v
        for k, v in all_eq.items():
            grids[f"{band_key}_kmag_eq_r_{k}"] = v
        for k, v in closed_eq.items():
            grids[f"{band_key}_kmag_eq_r_closed_{k}"] = v

    for b in bands:
        _accumulate_for(b, per_band_mask[b])
    _accumulate_for("total", union_mask_sub)

    stats = {
        "n_traces": int(result.n_traces),
        "n_closed": int(result.n_closed),
        "n_events_traced_min": n_event_min,
    }
    return grids, stats


def kmag_event_grids_to_xarray(
    grids: dict[str, np.ndarray],
    config: DwellGridConfig,
    bands: list[str],
    *,
    title: str = "QP Event Time Grid (KMAG-traced schemas)",
    extra_attrs: dict | None = None,
) -> "xr.Dataset":
    """Wrap KMAG event grids into an xarray Dataset.

    Output dims: ``(kmag_inv_lat, local_time)`` and
    ``(kmag_eq_r, local_time)``. The two coordinate vectors reuse
    ``config.lat_centers`` and ``config.r_centers`` respectively, so
    the bin edges align exactly with the dwell grid.
    """
    import xarray as xr

    coords = {
        "local_time": config.lt_centers,
        "kmag_inv_lat": config.lat_centers,
        "kmag_eq_r": config.r_centers,
        "lat_edges": config.lat_edges,
        "lt_edges": config.lt_edges,
        "r_edges": config.r_edges,
    }
    region_names = ("total", "magnetosphere", "magnetosheath",
                    "solar_wind", "unknown")
    data_vars: dict[str, tuple] = {}
    for b in list(bands) + ["total"]:
        for r in region_names:
            for prefix in ("kmag_inv_lat", "kmag_inv_lat_closed"):
                k = f"{b}_{prefix}_{r}"
                if k in grids:
                    data_vars[k] = (
                        ("kmag_inv_lat", "local_time"),
                        grids[k].astype(np.float32),
                    )
            for prefix in ("kmag_eq_r", "kmag_eq_r_closed"):
                k = f"{b}_{prefix}_{r}"
                if k in grids:
                    data_vars[k] = (
                        ("kmag_eq_r", "local_time"),
                        grids[k].astype(np.float32),
                    )

    attrs = {
        "title": title,
        "units": "minutes",
        "schema": "kmag_event_grids",
        "bands": list(bands),
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
