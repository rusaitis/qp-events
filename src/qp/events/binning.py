r"""Bin cumulative QP event time onto the dwell-grid coordinates.

The output of this module is the **numerator** for the
dwell-normalized occurrence-rate maps in Figs 7, 8 of the paper. The
denominator (cumulative spacecraft dwell time per cell) lives in
``Output/dwell_grid_cassini_saturn.zarr``; the bin edges here must
match it exactly.

Production pipeline (``scripts/bin_events_round8.py``) uses:

- :func:`accumulate_full_mirror` — for each band, accumulate event
  time on every dwell-grid schema derivable from the spacecraft
  trajectory (3D r/mag_lat/LT with region splits, 2D dipole invariant
  latitude, 2D plasma-sheet weak-field proxy).
- :func:`accumulate_kmag_event_grids` — companion that traces the
  union event-window minutes through the KMAG field and accumulates
  on the (kmag_inv_lat, LT) and (kmag_eq_r, LT) axes.
- :func:`full_mirror_grids_to_xarray` /
  :func:`kmag_event_grids_to_xarray` — wrap the resulting dicts in
  xarray datasets matching the canonical zarr layout.

:func:`bin_events_walking` is retained as a tested reference
implementation of per-minute walking binning (used by the test suite
to pin the same-band overlap regression); it is not on the round-8
critical path.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qp.constants import J2000_POSIX
from qp.dwell.grid import DwellGridConfig, _bin_index, _compute_bins
from qp.events.bands import QP_BAND_NAMES
from qp.events.catalog import WaveEvent

if TYPE_CHECKING:
    import xarray as xr

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Coordinate quantization + event-window helpers
# ----------------------------------------------------------------------

_UNIX_EPOCH = np.datetime64("1970-01-01T00:00:00")


def _event_time_window(ev: WaveEvent) -> tuple[float, float]:
    """Return ``(t_from, t_to)`` of an event as Unix-epoch seconds (float)."""
    dt_from = np.datetime64(ev.date_from) - _UNIX_EPOCH
    dt_to = np.datetime64(ev.date_to) - _UNIX_EPOCH
    return (
        float(dt_from.astype("timedelta64[s]").astype(float)),
        float(dt_to.astype("timedelta64[s]").astype(float)),
    )


# ----------------------------------------------------------------------
# Per-band sample mask builder (round-8 driver)
# ----------------------------------------------------------------------


def build_band_masks(
    date_from: np.ndarray,
    date_to: np.ndarray,
    period_min: np.ndarray,
    t_unix: np.ndarray,
    bands: list[str],
    band_lookup,
) -> tuple[dict[str, np.ndarray], int]:
    """Build a per-sample boolean mask per band.

    A sample is True for band B if it falls inside any event whose
    period maps to band B. Per band, do one ``searchsorted`` of all
    event endpoints, then build the mask via a +1/-1 cumulative-sum
    trick (so N range-set operations collapse to O(N + M) work).

    Detector stores ``date_to`` as the **time of the last included
    sample** (``qp.events.detector._ridge_to_packet`` uses
    ``times[t_end_idx]``), so the time window is the *closed* interval
    ``[t_from, t_to]``: ``side="left"`` picks the first sample with
    ``t_unix >= t_from``; ``side="right"`` picks one past the last
    sample with ``t_unix == t_to``. A 60-sample (1-min) event therefore
    contributes 60 mask minutes, while
    ``WaveEvent.duration_minutes = t_to - t_from`` is one fewer —
    callers should account for this in self-consistency checks.

    Parameters
    ----------
    date_from, date_to : np.ndarray of np.datetime64
        Event start/end times.
    period_min : np.ndarray of float
        Period in minutes for each event.
    t_unix : np.ndarray of float
        Trajectory timestamps in Unix epoch seconds.
    bands : list[str]
        Target band names.
    band_lookup : callable
        ``(period_min: float) -> str | None`` mapping a period to a
        band name (or None to drop the event).

    Returns
    -------
    masks : dict[str, np.ndarray]
        Per-band boolean masks of shape ``t_unix.shape``.
    n_unmapped : int
        Number of events whose period did not map to any of ``bands``.
    """
    n = t_unix.size
    epoch = np.datetime64("1970-01-01T00:00:00")

    period_arr = np.asarray(period_min, dtype=float)
    t_from_all = (
        (np.asarray(date_from, dtype="datetime64[s]") - epoch)
        .astype("timedelta64[s]")
        .astype(np.int64)
        .astype(float)
    )
    t_to_all = (
        (np.asarray(date_to, dtype="datetime64[s]") - epoch)
        .astype("timedelta64[s]")
        .astype(np.int64)
        .astype(float)
    )
    band_for_row = np.array(
        [band_lookup(p) for p in period_arr],
        dtype=object,
    )
    n_unmapped = int(np.sum(band_for_row == None))  # noqa: E711

    masks: dict[str, np.ndarray] = {}
    for band in bands:
        sel = band_for_row == band
        if not sel.any():
            masks[band] = np.zeros(n, dtype=bool)
            continue
        i_lo = np.searchsorted(t_unix, t_from_all[sel], side="left")
        i_hi = np.searchsorted(t_unix, t_to_all[sel], side="right")
        delta = np.zeros(n + 1, dtype=np.int32)
        np.add.at(delta, i_lo, 1)
        np.add.at(delta, i_hi, -1)
        masks[band] = np.cumsum(delta[:-1]) > 0

    return masks, n_unmapped


# ----------------------------------------------------------------------
# Binning stats + walking binner (reference implementation; deprecated)
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
    r"""Walking event-time binning. *DEPRECATED — reference impl only.*

    The production round-8 pipeline does not call this; see
    :func:`accumulate_full_mirror` and ``scripts/bin_events_round8.py``.
    Retained because the regression tests in
    ``tests/test_event_gridding.py::TestWalkingBinnerOverlap`` pin its
    same-band overlap behaviour, and it remains a useful reference for
    per-minute walking binning against arbitrary per-segment trajectories.
    Do not introduce new callers; extend :func:`accumulate_full_mirror`
    instead.

    For each event, walks the matching segment's position arrays
    minute-by-minute between ``date_from`` and ``date_to`` and
    accumulates ``+1.0`` minute per visited bin. This handles
    spacecraft motion within an event correctly, which matters near
    periapsis.

    Both the per-band grids and the ``"total"`` grid are accumulated
    from **OR-aggregated** per-segment masks (one accumulate call per
    (segment, band) and one per (segment, union)). A minute that lies
    inside two same-band events on the same segment counts once, not
    twice. The ``"total"`` grid takes the union across bands so a
    minute with *any* QP band firing also counts once. Together these
    guarantee ``event_time ≤ dwell`` per cell.

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
        b: np.zeros(config.shape, dtype=np.float64) for b in QP_BAND_NAMES
    }
    grids["total"] = np.zeros(config.shape, dtype=np.float64)

    stats = BinningStats()

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
        t_from, t_to = _event_time_window(ev)
        time_mask = (sp.times_unix >= t_from) & (sp.times_unix <= t_to)
        if sp.central_mask is not None:
            time_mask &= sp.central_mask
        seg_event_masks[seg_id][ev.band] |= time_mask

    # Phase 2: accumulate per-band grids from the OR-aggregated masks
    # (one accumulate call per (segment, band)). Accumulating per-event
    # would double-count any minutes that two same-band events share.
    # Stats are tallied here in a separate event-level pass.
    for seg_id, band_masks in seg_event_masks.items():
        sp = segment_positions[seg_id]
        for band, mask in band_masks.items():
            if band not in grids or not mask.any():
                continue
            contribution = _accumulate_mask(sp, mask, config, None)
            if contribution is not None:
                grids[band] += contribution

    # Per-event stats + fallback peak-position binning for events whose
    # segment is missing from segment_positions.
    for ev in events_list:
        stats.n_total += 1
        if ev.band is None:
            stats.n_missing_coords += 1
            continue

        seg_id = ev.segment_id
        if seg_id is None or seg_id not in segment_positions:
            # Fallback to peak position. Use the canonical _compute_bins
            # (length-1 arrays) so the in-range / bin-index logic does
            # not diverge from the dwell-grid pipeline.
            if ev.r_distance is None or ev.mag_lat is None or ev.local_time is None:
                stats.n_missing_coords += 1
                continue
            r_arr = np.array([float(ev.r_distance)])
            lat_arr = np.array([float(ev.mag_lat)])
            lt_arr = np.array([float(ev.local_time)])
            i_r_arr, i_lat_arr, i_lt_arr, in_range = _compute_bins(
                r_arr,
                lat_arr,
                lt_arr,
                config,
            )
            if not in_range[0]:
                stats.n_out_of_range += 1
                continue
            i_r = int(i_r_arr[0])
            i_lat = int(i_lat_arr[0])
            i_lt = int(i_lt_arr[0])
            minutes = float(ev.duration_minutes)
            if ev.band in grids:
                grids[ev.band][i_r, i_lat, i_lt] += minutes
            grids["total"][i_r, i_lat, i_lt] += minutes
            stats.n_binned += 1
            continue

        # Already accumulated in the Phase-2 loop above; just check
        # that the event's time window had finite coordinates so we
        # can update the stats counters consistently with the old API.
        sp = segment_positions[seg_id]
        t_from, t_to = _event_time_window(ev)
        mask = (sp.times_unix >= t_from) & (sp.times_unix <= t_to)
        if sp.central_mask is not None:
            mask = mask & sp.central_mask
        if not mask.any():
            stats.n_missing_coords += 1
            continue
        rs = sp.r[mask]
        if not np.isfinite(rs).any():
            stats.n_missing_coords += 1
            continue
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
        (rs >= config.r_range[0])
        & (rs < config.r_range[1])
        & (lats >= config.lat_range[0])
        & (lats < config.lat_range[1])
        & (lts >= config.lt_range[0])
        & (lts < config.lt_range[1])
    )
    rs, lats, lts = rs[in_r], lats[in_r], lts[in_r]
    if rs.size == 0:
        if stats is not None:
            stats.n_out_of_range += 1
        return None

    i_r = _bin_index(rs, *config.r_range, config.n_r)
    i_lat = _bin_index(lats, *config.lat_range, config.n_lat)
    i_lt = _bin_index(lts, *config.lt_range, config.n_lt)

    flat = np.ravel_multi_index((i_r, i_lat, i_lt), config.shape)
    counts = np.bincount(flat, minlength=math.prod(config.shape))
    return counts.astype(np.float64).reshape(config.shape)


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
    field-line tracing and are added by :func:`accumulate_kmag_event_grids`.

    Parameters
    ----------
    events : iterable of WaveEvent
        Detections to bin. Each event's ``band`` attribute selects its
        target grid; events with no band, or a band not in ``bands``,
        are skipped. To re-bin into a different band scheme, set
        ``ev.band`` upstream (see ``scripts/bin_events_round8.py`` for
        the ``--bands`` CLI mechanism).
    segment_positions : dict[int, SegmentPositionsExt]
        Per-segment trajectory and region information. Events whose
        segment id is missing are dropped silently.
    bands : list[str], optional
        Band names to populate. Defaults to ``QP_BAND_NAMES``.
    config : DwellGridConfig, optional
        Same axes as the dwell grid. Defaults to
        ``DwellGridConfig()``.
    b_threshold_nT : float, default 2.0
        Plasma-sheet proxy threshold (matches the dwell grid).
    """
    from qp.dwell.grid import (
        accumulate_inv_lat_grid_cached,
        accumulate_weak_field_grid_cached,
        accumulate_with_regions_cached,
        precompute_bins,
    )

    if config is None:
        config = DwellGridConfig()
    if bands is None:
        bands = list(QP_BAND_NAMES)

    events_list = list(events)

    seg_masks: dict[int, dict[str, np.ndarray]] = {}
    for ev in events_list:
        seg_id = ev.segment_id
        if seg_id is None or seg_id not in segment_positions:
            continue
        if ev.band is None or ev.band not in bands:
            continue
        band = ev.band
        sp = segment_positions[seg_id]
        n = sp.times_unix.size
        seg_masks.setdefault(seg_id, {b: np.zeros(n, dtype=bool) for b in bands})
        t_from, t_to = _event_time_window(ev)
        m = (sp.times_unix >= t_from) & (sp.times_unix <= t_to)
        if sp.central_mask is not None:
            m &= sp.central_mask
        # OR-aggregation is load-bearing: two same-band events that share
        # any minute on the same segment must collapse to a single boolean
        # mask. Replacing |= with += or accumulating per-event downstream
        # would double-count overlap minutes and let event_time exceed
        # dwell time per cell.
        seg_masks[seg_id][band] |= m

    grids: dict[str, np.ndarray] = {}
    region_names = ("total", "magnetosphere", "magnetosheath", "solar_wind", "unknown")
    shape_3d = config.shape
    shape_2d = (config.n_lat, config.n_lt)

    def _zero_grids_for_band(b: str) -> None:
        # float32 matches the dwell-grid accumulator output and the
        # downstream zarr dtype — halves resident memory of the output
        # dict. Per-segment += accumulation in float32 loses < 1 ULP
        # per add (< 1 min total over the mission), well below the
        # event-time granularity.
        for r in region_names:
            grids[f"{b}_{r}"] = np.zeros(shape_3d, dtype=np.float32)
            grids[f"{b}_dipole_inv_lat_{r}"] = np.zeros(shape_2d, dtype=np.float32)
            grids[f"{b}_weak_field_{r}"] = np.zeros(shape_2d, dtype=np.float32)

    band_keys = list(bands) + ["total"]
    for b in band_keys:
        _zero_grids_for_band(b)

    for seg_id, band_to_mask in seg_masks.items():
        sp = segment_positions[seg_id]
        if (
            sp.ksm_x is None
            or sp.ksm_y is None
            or sp.ksm_z is None
            or sp.region_codes is None
        ):
            log.warning(
                "segment %s missing KSM cartesian or region codes; skipping",
                seg_id,
            )
            continue
        # Compute bin indices + in-range masks once per segment. The
        # four schemas (3D regions, 2D inv-lat, 2D weak-field) and N
        # bands all reuse this cache rather than re-deriving coords.
        cache = precompute_bins(sp.ksm_x, sp.ksm_y, sp.ksm_z, config)
        codes = sp.region_codes
        bt = sp.b_total_nT
        any_mask = np.zeros_like(next(iter(band_to_mask.values())))

        # Bind per-segment values (cache, codes, bt) as default args so the
        # closure captures the *current* iteration's values. Only matters if
        # _accumulate ever outlives the loop — it doesn't today, but pinning
        # the binding is cheap insurance and silences ruff B023.
        def _accumulate(
            prefix: str,
            sample_mask: np.ndarray,
            *,
            cache=cache,  # noqa: B008
            codes=codes,  # noqa: B008
            bt=bt,  # noqa: B008
        ) -> None:
            r3d = accumulate_with_regions_cached(
                cache,
                codes,
                1.0,
                mask=sample_mask,
                config=config,
            )
            r2d = accumulate_inv_lat_grid_cached(
                cache,
                codes,
                1.0,
                mask=sample_mask,
                config=config,
            )
            for r in region_names:
                grids[f"{prefix}_{r}"] += r3d[r]
                grids[f"{prefix}_dipole_inv_lat_{r}"] += r2d[r]
            if bt is not None:
                rwf = accumulate_weak_field_grid_cached(
                    cache,
                    bt,
                    1.0,
                    b_threshold_nT,
                    region_codes=codes,
                    mask=sample_mask,
                    config=config,
                )
                for r in region_names:
                    grids[f"{prefix}_weak_field_{r}"] += rwf[r]

        for b in bands:
            mask = band_to_mask[b]
            if not mask.any():
                continue
            any_mask |= mask
            _accumulate(b, mask)

        if any_mask.any():
            _accumulate("total", any_mask)

    return grids


def full_mirror_grids_to_xarray(
    grids: dict[str, np.ndarray],
    config: DwellGridConfig,
    bands: list[str],
    *,
    title: str = "QP Event Time Grid (full-mirror schema)",
    description: str | None = None,
    extra_attrs: dict | None = None,
) -> xr.Dataset:
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
    trace_every_n: int = 1,
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
    trace_every_n : int, default 1
        Subsampling cadence for tracing. Defaults to **1** (no
        subsampling) so every event-window minute contributes to the
        KMAG grids; cadence-10 subsampling silently dropped short
        events whose only minutes fell between traced positions,
        biasing ``sum(B_kmag_*) < sum(B_total)``. The event union is
        small (~5% of mission minutes), so the full-cadence cost is
        bounded and worth the bias removal.
    config : DwellGridConfig, optional
    tracing_config, field_config : qp.dwell.tracing types, optional
        If None, default ``TracingConfig`` / ``SaturnFieldConfig`` are
        used.

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
        n_event_min,
        max(1, n_event_min // trace_every_n),
        trace_every_n,
    )

    # 2. Subset trajectory + region codes to the union mask
    keep = np.flatnonzero(union)
    x_e = x_traj[keep]
    y_e = y_traj[keep]
    z_e = z_traj[keep]
    t_e = t_unix_traj[keep]
    codes_e = region_codes_traj[keep]
    # Convert POSIX → J2000 for KMAG
    t_j2000 = t_e - J2000_POSIX

    # 3. Trace
    result = compute_invariant_latitudes_parallel(
        x_e,
        y_e,
        z_e,
        t_j2000,
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
        band_key: str,
        mask_sub: np.ndarray,
    ) -> None:
        if not mask_sub.any():
            empty_inv = np.zeros(
                (config.n_lat, config.n_lt),
                dtype=np.float32,
            )
            empty_eq = np.zeros((config.n_r, config.n_lt), dtype=np.float32)
            for r in (
                "total",
                "magnetosphere",
                "magnetosheath",
                "solar_wind",
                "unknown",
            ):
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
            inv_n_b,
            inv_s_b,
            closed_b,
            lt_sub,
            z_sub,
            dt_minutes=dt_trace,
            region_codes=codes_sub,
            config=config,
        )
        closed_inv = accumulate_traced_inv_lat_grid(
            inv_n_b,
            inv_s_b,
            closed_b,
            lt_sub,
            z_sub,
            dt_minutes=dt_trace,
            region_codes=codes_sub,
            closed_only=True,
            config=config,
        )
        # kmag_eq_r (all + closed-only)
        all_eq = accumulate_kmag_eq_r_grid(
            l_eq_b,
            closed_b,
            lt_sub,
            dt_minutes=dt_trace,
            region_codes=codes_sub,
            config=config,
        )
        closed_eq = accumulate_kmag_eq_r_grid(
            l_eq_b,
            closed_b,
            lt_sub,
            dt_minutes=dt_trace,
            region_codes=codes_sub,
            closed_only=True,
            config=config,
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
) -> xr.Dataset:
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
    region_names = ("total", "magnetosphere", "magnetosheath", "solar_wind", "unknown")
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
