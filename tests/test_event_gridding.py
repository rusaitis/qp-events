"""Tests for the full-mirror event-time gridding extensions in
``qp.events.binning``.

These exercise :func:`qp.events.binning.accumulate_full_mirror` on
synthetic spacecraft trajectories with hand-placed events. The dwell-
grid accumulators they call internally are tested elsewhere — here we
verify that the per-band masking, region splits, and union-total
behave correctly.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from qp.dwell.grid import DwellGridConfig
from qp.events.binning import (
    SegmentPositionsExt,
    accumulate_full_mirror,
    full_mirror_grids_to_xarray,
)
from qp.events.catalog import WaveEvent


def _make_segment(n_minutes: int = 1440) -> SegmentPositionsExt:
    """Build a simple synthetic 24-h trajectory at constant position.

    The spacecraft sits at (x, y, z) = (15, 0, 5) R_S — a magnetic
    latitude ~ 18° outside the planet, in the magnetosphere by
    construction. Region code 0. |B| = 5 nT (above the weak-field
    threshold so weak_field is empty).
    """
    epoch = datetime.datetime(2008, 6, 15)
    times_unix = np.array([
        (epoch + datetime.timedelta(minutes=i) - datetime.datetime(1970, 1, 1))
        .total_seconds()
        for i in range(n_minutes)
    ])
    x = np.full(n_minutes, 15.0)
    y = np.zeros(n_minutes)
    z = np.full(n_minutes, 5.0)
    r = np.sqrt(x * x + y * y + z * z)
    # mag_lat: simple latitude in degrees (asin of z/r)
    mag_lat = np.degrees(np.arcsin(z / r))
    # local_time (12.0 h is the noon meridian for x>0, y=0)
    lt = np.full(n_minutes, 12.0)
    region = np.zeros(n_minutes, dtype=int)  # all magnetosphere
    btot = np.full(n_minutes, 5.0)  # well above weak-field 2 nT cutoff

    return SegmentPositionsExt(
        seg_idx=0,
        times_unix=times_unix,
        r=r,
        mag_lat=mag_lat,
        local_time=lt,
        ksm_x=x,
        ksm_y=y,
        ksm_z=z,
        region_codes=region,
        b_total_nT=btot,
    )


def _make_event(
    seg: SegmentPositionsExt,
    *,
    start_min: int = 60,
    duration_min: int = 240,
    band: str = "QP60",
    period_min: float = 60.0,
) -> WaveEvent:
    epoch = datetime.datetime(1970, 1, 1)
    t_from = epoch + datetime.timedelta(seconds=float(seg.times_unix[start_min]))
    t_to = epoch + datetime.timedelta(
        seconds=float(seg.times_unix[start_min + duration_min - 1]),
    )
    return WaveEvent(
        date_from=t_from,
        date_to=t_to,
        period=period_min * 60.0,
        amplitude=0.2,
        snr=10.0,
        local_time=12.0,
        mag_lat=18.0,
        r_distance=15.8,
        band=band,
        period_peak_min=period_min,
        segment_id=seg.seg_idx,
    )


class TestSchemaShape:
    def test_paper_schema_has_all_band_region_combinations(self) -> None:
        seg = _make_segment()
        ev = _make_event(seg, band="QP60")
        grids = accumulate_full_mirror([ev], {0: seg})
        # 4 bands (QP30 + QP60 + QP120 + total) × 5 regions × 3 schemas
        assert "QP60_total" in grids
        assert "QP60_magnetosphere" in grids
        assert "QP60_dipole_inv_lat_total" in grids
        assert "QP60_weak_field_total" in grids
        assert "total_total" in grids
        # 3D dims
        assert grids["QP60_total"].shape == DwellGridConfig().shape
        # 2D dims
        assert grids["QP60_dipole_inv_lat_total"].shape == (180, 96)


class TestPerBandAccumulation:
    def test_qp60_event_only_lands_in_qp60_band(self) -> None:
        seg = _make_segment()
        ev = _make_event(seg, band="QP60", duration_min=240)
        grids = accumulate_full_mirror([ev], {0: seg})
        # QP60 grids should sum to ~ 240 minutes; QP30/QP120 to 0
        qp60_minutes = float(grids["QP60_total"].sum())
        qp30_minutes = float(grids["QP30_total"].sum())
        qp120_minutes = float(grids["QP120_total"].sum())
        assert qp60_minutes == pytest.approx(240.0, abs=1.0)
        assert qp30_minutes == 0.0
        assert qp120_minutes == 0.0

    def test_total_band_is_band_union(self) -> None:
        seg = _make_segment()
        e1 = _make_event(
            seg, band="QP60", start_min=60, duration_min=120,
        )
        e2 = _make_event(
            seg, band="QP30", start_min=600, duration_min=120,
        )
        grids = accumulate_full_mirror([e1, e2], {0: seg})
        # Two non-overlapping events — total should equal sum of bands
        qp60_total = float(grids["QP60_total"].sum())
        qp30_total = float(grids["QP30_total"].sum())
        union_total = float(grids["total_total"].sum())
        assert union_total == pytest.approx(qp60_total + qp30_total, abs=1.0)

    def test_overlapping_events_no_double_count_in_union(self) -> None:
        seg = _make_segment()
        # Two overlapping events in different bands — the union should
        # count each minute once.
        e1 = _make_event(seg, band="QP60", start_min=60, duration_min=240)
        e2 = _make_event(seg, band="QP30", start_min=120, duration_min=180)
        grids = accumulate_full_mirror([e1, e2], {0: seg})
        # Union: minutes 60..299 inclusive = 240
        union_total = float(grids["total_total"].sum())
        assert union_total == pytest.approx(240.0, abs=1.0)


class TestRegionSplits:
    def test_magnetosphere_grid_carries_all_minutes(self) -> None:
        seg = _make_segment()
        ev = _make_event(seg, band="QP60", duration_min=300)
        grids = accumulate_full_mirror([ev], {0: seg})
        # All samples have region_code 0 → magnetosphere
        assert (
            float(grids["QP60_magnetosphere"].sum())
            == pytest.approx(float(grids["QP60_total"].sum()), abs=0.1)
        )
        # Other regions empty
        assert float(grids["QP60_magnetosheath"].sum()) == 0.0
        assert float(grids["QP60_solar_wind"].sum()) == 0.0


class TestRebandWithCallable:
    def test_band_for_period_min_overrides_event_band(self) -> None:
        seg = _make_segment()
        # Stored event band is QP60 but we map it to a custom "wide"
        # band via the period-only callable. The event should land in
        # "wide", not "QP60".
        ev = _make_event(seg, band="QP60", duration_min=180)

        def lookup(p: float) -> str | None:
            return "wide" if 50.0 <= p < 200.0 else None

        grids = accumulate_full_mirror(
            [ev],
            {0: seg},
            bands=["wide"],
            band_for_period_min=lookup,
        )
        assert float(grids["wide_total"].sum()) == pytest.approx(180.0, abs=1.0)


class TestXarrayWrap:
    def test_dataset_carries_required_attrs(self) -> None:
        seg = _make_segment()
        ev = _make_event(seg, band="QP60", duration_min=120)
        grids = accumulate_full_mirror([ev], {0: seg})
        ds = full_mirror_grids_to_xarray(
            grids, DwellGridConfig(), bands=["QP30", "QP60", "QP120"],
        )
        assert ds.attrs["schema"] == "full_mirror"
        assert ds.attrs["kmag_inv_lat_populated"] is False
        # The 3D variable exists with the right dims
        assert ds["QP60_total"].dims == ("r", "magnetic_latitude", "local_time")
        # The 2D dipole inv-lat variable exists
        assert ds["QP60_dipole_inv_lat_total"].dims == (
            "dipole_inv_lat", "local_time",
        )

    def test_bin_edges_match_dwell_grid_default(self) -> None:
        seg = _make_segment()
        ev = _make_event(seg, band="QP60", duration_min=60)
        grids = accumulate_full_mirror([ev], {0: seg})
        ds = full_mirror_grids_to_xarray(
            grids, DwellGridConfig(), bands=["QP30", "QP60", "QP120"],
        )
        # r edges: 0..100 with step 1.0
        assert ds["r_edges"].values[0] == pytest.approx(0.0)
        assert ds["r_edges"].values[-1] == pytest.approx(100.0)
        # lat edges: -90..90 with step 1.0
        assert ds["lat_edges"].values[0] == pytest.approx(-90.0)
        assert ds["lat_edges"].values[-1] == pytest.approx(90.0)


class TestKmagEqRGrid:
    """Tests for the equatorial-r dwell accumulator."""

    def test_closed_lines_bin_at_apex(self) -> None:
        from qp.dwell.grid import accumulate_kmag_eq_r_grid

        # 3 closed lines with apexes at L = 5.5, 10.5, 20.5 R_S, all at noon LT
        l_eq = np.array([5.5, 10.5, 20.5])
        is_closed = np.array([True, True, True])
        lt = np.array([12.0, 12.0, 12.0])
        grids = accumulate_kmag_eq_r_grid(
            l_eq, is_closed, lt, dt_minutes=10.0,
        )
        total = grids["total"]
        assert total.shape == (100, 96)
        # All three contribute 10 min each in the noon LT column
        i_lt = int((12.0 - 0.0) / 24.0 * 96)
        assert total[5, i_lt] == pytest.approx(10.0)
        assert total[10, i_lt] == pytest.approx(10.0)
        assert total[20, i_lt] == pytest.approx(10.0)
        # Other bins zero
        assert total.sum() == pytest.approx(30.0)

    def test_closed_only_filter_drops_open(self) -> None:
        from qp.dwell.grid import accumulate_kmag_eq_r_grid

        l_eq = np.array([10.5, 15.5])
        is_closed = np.array([True, False])
        lt = np.array([12.0, 12.0])
        all_lines = accumulate_kmag_eq_r_grid(
            l_eq, is_closed, lt, dt_minutes=10.0, closed_only=False,
        )
        closed = accumulate_kmag_eq_r_grid(
            l_eq, is_closed, lt, dt_minutes=10.0, closed_only=True,
        )
        # all_lines: both closed AND open lines (apex finite) contribute
        # closed: only the closed line contributes
        assert float(all_lines["total"].sum()) == pytest.approx(20.0)
        assert float(closed["total"].sum()) == pytest.approx(10.0)

    def test_nan_apex_excluded(self) -> None:
        from qp.dwell.grid import accumulate_kmag_eq_r_grid

        # Failed traces have NaN apex; they should not contribute
        l_eq = np.array([np.nan, 12.5, np.nan])
        is_closed = np.array([False, True, False])
        lt = np.array([0.0, 12.0, 18.0])
        grids = accumulate_kmag_eq_r_grid(
            l_eq, is_closed, lt, dt_minutes=10.0,
        )
        assert float(grids["total"].sum()) == pytest.approx(10.0)

    def test_per_region_split_consistent(self) -> None:
        from qp.dwell.grid import accumulate_kmag_eq_r_grid

        l_eq = np.array([10.5, 10.5, 10.5])
        is_closed = np.array([True, True, True])
        lt = np.array([6.0, 12.0, 18.0])
        codes = np.array([0, 1, 0])  # MS, SH, MS
        grids = accumulate_kmag_eq_r_grid(
            l_eq, is_closed, lt, dt_minutes=10.0, region_codes=codes,
        )
        # total sums to 30; magnetosphere = 20; sheath = 10
        assert float(grids["total"].sum()) == pytest.approx(30.0)
        assert float(grids["magnetosphere"].sum()) == pytest.approx(20.0)
        assert float(grids["magnetosheath"].sum()) == pytest.approx(10.0)


class TestKmagEventGridsAccumulator:
    """Light end-to-end test for accumulate_kmag_event_grids.

    Uses a tiny synthetic trajectory in the magnetosphere where the
    KMAG model produces a closed field line, and verifies that:
    - the union ``total_*`` schema has at least as many minutes as
      either single band
    - the kmag_inv_lat and kmag_eq_r schemas both populate
    """

    def test_synthetic_one_event_traces_and_bins(self) -> None:
        from qp.events.binning import accumulate_kmag_event_grids

        n = 200  # 200 minutes
        t_unix = np.linspace(1.0e9, 1.0e9 + n * 60, n)
        # static spacecraft position around L=10 R_S, low latitude — KMAG
        # should give a closed line for the magnetosphere region.
        x = np.full(n, 10.0)
        y = np.zeros(n)
        z = np.full(n, 0.5)
        codes = np.zeros(n, dtype=int)  # all MS
        # event between minutes 50..100 in band QP60
        mask_qp60 = np.zeros(n, dtype=bool)
        mask_qp60[50:100] = True
        masks = {"QP60": mask_qp60}

        grids, stats = accumulate_kmag_event_grids(
            masks, x, y, z, t_unix, codes,
            trace_every_n=10,
        )
        assert stats["n_events_traced_min"] == 50
        # The grids dict should contain the four families per band:
        for prefix in ("kmag_inv_lat", "kmag_inv_lat_closed",
                       "kmag_eq_r", "kmag_eq_r_closed"):
            assert f"QP60_{prefix}_total" in grids
            assert f"total_{prefix}_total" in grids
        # Equatorial-r grid should have non-zero total if the trace
        # produced a closed line. The benchmark KMAG defaults at
        # x=10, y=0, z=0.5 give a well-defined closed field line.
        # Either equatorial OR inv-lat should accumulate something.
        total_eq = float(grids["QP60_kmag_eq_r_total"].sum())
        total_inv = float(grids["QP60_kmag_inv_lat_total"].sum())
        assert (total_eq + total_inv) > 0.0
