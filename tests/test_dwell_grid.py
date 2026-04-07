"""Tests for qp.dwell — spherical dwell-time grid accumulation and I/O."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qp.dwell.grid import (
    DwellGridConfig,
    accumulate_dwell_time,
    accumulate_traced_inv_lat_grid,
    accumulate_weak_field_grid,
    accumulate_with_regions,
)
from qp.dwell.io import ZarrEncoding, load_zarr, save_zarr, to_xarray
from qp.dwell.tracing import TracingConfig, TracingResult


# ============================================================================
# DwellGridConfig
# ============================================================================


class TestDwellGridConfig:
    def test_default_shape(self):
        cfg = DwellGridConfig()
        assert cfg.shape == (100, 180, 96)

    def test_custom_shape(self):
        cfg = DwellGridConfig(n_r=30, n_lat=45, n_lt=24)
        assert cfg.shape == (30, 45, 24)

    def test_r_edges_count(self):
        cfg = DwellGridConfig(n_r=10)
        assert len(cfg.r_edges) == 11

    def test_r_centers_count(self):
        cfg = DwellGridConfig(n_r=10)
        assert len(cfg.r_centers) == 10

    def test_lat_centers_symmetric(self):
        cfg = DwellGridConfig(n_lat=18)
        centers = cfg.lat_centers
        np.testing.assert_allclose(centers + centers[::-1], 0.0, atol=1e-10)

    def test_lt_range(self):
        cfg = DwellGridConfig(n_lt=24)
        assert cfg.lt_centers[0] == pytest.approx(0.5)
        assert cfg.lt_centers[-1] == pytest.approx(23.5)


# ============================================================================
# accumulate_dwell_time
# ============================================================================


class TestAccumulateDwellTime:
    @pytest.fixture
    def small_config(self):
        return DwellGridConfig(
            n_r=10, n_lat=18, n_lt=12,
            r_range=(0, 30), lat_range=(-90, 90), lt_range=(0, 24),
        )

    def test_single_point(self, small_config):
        """A single point at (10, 0, 0) KSM should land in the correct bin."""
        grid = accumulate_dwell_time(
            x=[10.0], y=[0.0], z=[0.0], dt_minutes=1.0, config=small_config,
        )
        assert grid.sum() == pytest.approx(1.0)
        # r=10 → bin 3 (0-3, 3-6, 6-9, 9-12), lat≈0 → middle, LT=12h (sunward)
        assert np.count_nonzero(grid) == 1

    def test_total_time_conservation(self, small_config):
        """Total accumulated time must equal n_points × dt."""
        rng = np.random.default_rng(42)
        n = 1000
        # Random positions within grid range
        r = rng.uniform(1, 25, n)
        theta = rng.uniform(0, np.pi, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        dt = 5.0
        grid = accumulate_dwell_time(x, y, z, dt_minutes=dt, config=small_config)
        np.testing.assert_allclose(grid.sum(), n * dt, rtol=0.01)

    def test_out_of_range_excluded(self, small_config):
        """Points outside the grid range should not contribute."""
        grid = accumulate_dwell_time(
            x=[100.0], y=[0.0], z=[0.0],  # r=100, beyond r_range=(0,30)
            dt_minutes=1.0, config=small_config,
        )
        assert grid.sum() == pytest.approx(0.0)

    def test_equatorial_point_at_noon(self, small_config):
        """(x=15, y=0, z=0) → r=15, lat≈0, LT=12h."""
        grid = accumulate_dwell_time(
            x=[15.0], y=[0.0], z=[0.0], dt_minutes=1.0, config=small_config,
        )
        # Find the non-zero bin
        idx = np.argwhere(grid > 0)
        assert len(idx) == 1
        i_r, i_lat, i_lt = idx[0]
        # r=15 in range (0,30) with 10 bins → bin 5
        assert i_r == 5
        # lat≈0 → middle bin (9 for 18 bins spanning -90 to 90)
        assert i_lat in (8, 9)
        # LT=12h → bin 6 (for 12 bins spanning 0-24)
        assert i_lt == 6

    def test_midnight_point(self, small_config):
        """(x=-15, y=0, z=0) → LT=0h (midnight)."""
        grid = accumulate_dwell_time(
            x=[-15.0], y=[0.0], z=[0.0], dt_minutes=1.0, config=small_config,
        )
        idx = np.argwhere(grid > 0)
        assert len(idx) == 1
        i_lt = idx[0, 2]
        assert i_lt == 0  # LT=0h → first bin

    def test_uniform_circular_orbit(self, small_config):
        """Points uniformly around a circle should produce uniform LT distribution."""
        n = 4800
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r_orbit = 15.0
        x = r_orbit * np.cos(phi)
        y = r_orbit * np.sin(phi)
        z = np.zeros(n)

        grid = accumulate_dwell_time(x, y, z, dt_minutes=1.0, config=small_config)

        # Sum over r and lat to get LT profile
        lt_profile = grid.sum(axis=(0, 1))
        # All LT bins should have similar counts (within 20%)
        assert lt_profile.min() > 0
        ratio = lt_profile.max() / lt_profile.min()
        assert ratio < 1.3, f"LT distribution not uniform: ratio={ratio:.2f}"

    def test_symmetric_orbit_symmetric_grid(self, small_config):
        """An orbit symmetric about the equator should give symmetric lat distribution."""
        n = 2000
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r_orbit = 15.0
        x = r_orbit * np.cos(phi)
        y = r_orbit * np.sin(phi)
        z_amp = 5.0
        z = z_amp * np.sin(phi)  # symmetric about z=0

        grid = accumulate_dwell_time(x, y, z, dt_minutes=1.0, config=small_config)
        lat_profile = grid.sum(axis=(0, 2))  # sum over r and LT

        # Should be roughly symmetric about the equator
        n_lat = len(lat_profile)
        north = lat_profile[n_lat // 2:]
        south = lat_profile[:n_lat // 2][::-1]
        np.testing.assert_allclose(north, south, rtol=0.15)

    def test_default_config(self):
        """accumulate_dwell_time should work with default config."""
        grid = accumulate_dwell_time(x=[10.0], y=[0.0], z=[0.0])
        assert grid.shape == (100, 180, 96)
        assert grid.sum() == pytest.approx(1.0)


# ============================================================================
# accumulate_with_regions
# ============================================================================


class TestAccumulateWithRegions:
    @pytest.fixture
    def small_config(self):
        return DwellGridConfig(
            n_r=10, n_lat=18, n_lt=12,
            r_range=(0, 30), lat_range=(-90, 90), lt_range=(0, 24),
        )

    def test_region_separation(self, small_config):
        """MS + SH + SW + unknown should equal total."""
        x = np.array([10.0, 15.0, 20.0, 25.0])
        y = np.array([0.0, 5.0, -5.0, 0.0])
        z = np.array([0.0, 1.0, -1.0, 0.0])
        codes = np.array([0, 1, 2, 9])  # MS, SH, SW, unknown

        grids = accumulate_with_regions(x, y, z, codes, dt_minutes=1.0, config=small_config)

        total = grids["total"]
        parts = grids["magnetosphere"] + grids["magnetosheath"] + grids["solar_wind"] + grids["unknown"]
        np.testing.assert_allclose(total, parts)

    def test_all_ms(self, small_config):
        """All points in magnetosphere → magnetosphere grid equals total."""
        x = np.array([10.0, 15.0])
        y = np.array([0.0, 5.0])
        z = np.array([0.0, 1.0])
        codes = np.array([0, 0])

        grids = accumulate_with_regions(x, y, z, codes, dt_minutes=1.0, config=small_config)
        np.testing.assert_allclose(grids["magnetosphere"], grids["total"])
        assert grids["magnetosheath"].sum() == 0.0
        assert grids["solar_wind"].sum() == 0.0

    def test_returns_all_keys(self, small_config):
        grids = accumulate_with_regions(
            [10.0], [0.0], [0.0], [0], dt_minutes=1.0, config=small_config,
        )
        assert set(grids.keys()) == {"total", "magnetosphere", "magnetosheath", "solar_wind", "unknown"}


# ============================================================================
# xarray I/O
# ============================================================================


class TestXarrayIO:
    @pytest.fixture
    def sample_dataset(self):
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid = np.random.default_rng(42).uniform(0, 10, cfg.shape)
        return to_xarray({"total": grid, "magnetosphere": grid * 0.7}, cfg, attrs={"source": "test"})

    def test_dimensions(self, sample_dataset):
        # Data dimensions (excludes non-dimension coords like bin edges)
        assert set(sample_dataset["total"].dims) == {"r", "magnetic_latitude", "local_time"}

    def test_coordinate_values(self, sample_dataset):
        assert len(sample_dataset.coords["r"]) == 5
        assert len(sample_dataset.coords["magnetic_latitude"]) == 9
        assert len(sample_dataset.coords["local_time"]) == 6

    def test_data_variables(self, sample_dataset):
        assert "total" in sample_dataset.data_vars
        assert "magnetosphere" in sample_dataset.data_vars

    def test_attributes(self, sample_dataset):
        assert sample_dataset.attrs["source"] == "test"
        assert "title" in sample_dataset.attrs
        assert sample_dataset.attrs["n_r"] == 5

    def test_zarr_roundtrip(self, sample_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.zarr"
            save_zarr(sample_dataset, path)
            loaded = load_zarr(path)
            np.testing.assert_allclose(
                sample_dataset["total"].values, loaded["total"].values,
            )
            np.testing.assert_allclose(
                sample_dataset["magnetosphere"].values, loaded["magnetosphere"].values,
            )
            assert loaded.attrs["source"] == "test"

    def test_coordinate_metadata(self, sample_dataset):
        assert sample_dataset.coords["r"].attrs["units"] == "R_S"
        assert "latitude" in sample_dataset.coords["magnetic_latitude"].attrs["long_name"].lower()

    def test_bin_edges_present(self, sample_dataset):
        """Bin edges should be stored as non-dimension coordinates."""
        assert "r_edges" in sample_dataset.coords
        assert "lat_edges" in sample_dataset.coords
        assert "lt_edges" in sample_dataset.coords
        # Edges have one more element than centers
        assert len(sample_dataset.coords["r_edges"]) == 6  # n_r=5 → 6 edges
        assert len(sample_dataset.coords["lat_edges"]) == 10  # n_lat=9 → 10 edges
        assert len(sample_dataset.coords["lt_edges"]) == 7  # n_lt=6 → 7 edges


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_empty_input(self):
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6)
        grid = accumulate_dwell_time([], [], [], dt_minutes=1.0, config=cfg)
        assert grid.sum() == 0.0
        assert grid.shape == cfg.shape

    def test_pole_point(self):
        """Point at z=10, x=y=0 → lat≈90°, should land in highest lat bin."""
        cfg = DwellGridConfig(n_r=10, n_lat=18, n_lt=12, r_range=(0, 30))
        grid = accumulate_dwell_time(
            x=[0.001], y=[0.0], z=[10.0], dt_minutes=1.0, config=cfg,
        )
        idx = np.argwhere(grid > 0)
        assert len(idx) == 1
        i_lat = idx[0, 1]
        # lat≈90° → highest bin (17 for 18 bins)
        assert i_lat >= 16

    def test_large_dt(self):
        """dt_minutes=60 should accumulate 60 min per point."""
        grid = accumulate_dwell_time(x=[10.0], y=[0.0], z=[0.0], dt_minutes=60.0)
        assert grid.sum() == pytest.approx(60.0)


# ============================================================================
# TracingConfig
# ============================================================================


class TestTracingConfig:
    def test_defaults(self):
        cfg = TracingConfig()
        assert cfg.trace_every_n == 10
        assert cfg.step == 0.15
        assert cfg.max_radius == 60.0
        assert cfg.min_radius == 1.0
        assert cfg.surface_tolerance == 1.5
        assert cfg.max_steps == 20_000
        assert cfg.log_interval == 1000
        assert cfg.region_filter == (0,)
        assert cfg.n_workers == 1
        assert cfg.chunk_size is None

    def test_custom(self):
        cfg = TracingConfig(trace_every_n=120, step=0.05, max_radius=80.0)
        assert cfg.trace_every_n == 120
        assert cfg.step == 0.05
        assert cfg.max_radius == 80.0

    def test_frozen(self):
        cfg = TracingConfig()
        with pytest.raises(AttributeError):
            cfg.step = 0.5  # type: ignore[misc]


# ============================================================================
# ZarrEncoding
# ============================================================================


class TestZarrEncoding:
    def test_defaults(self):
        enc = ZarrEncoding()
        assert enc.compressor == "zstd"
        assert enc.compression_level == 3
        assert enc.chunks is None
        assert enc.dtype == "float32"

    def test_roundtrip_compressed(self):
        """Zarr save/load with zstd compression and float32."""
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid = np.random.default_rng(42).uniform(0, 10, cfg.shape)
        ds = to_xarray({"total": grid}, cfg, attrs={"source": "test"})

        enc = ZarrEncoding(compressor="zstd", compression_level=3, dtype="float32")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.zarr"
            save_zarr(ds, path, encoding=enc)
            loaded = load_zarr(path)
            # float32 round-trip has reduced precision
            np.testing.assert_allclose(
                ds["total"].values, loaded["total"].values, rtol=1e-6,
            )

    def test_roundtrip_blosc(self):
        """Zarr save/load with blosc compression."""
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid = np.ones(cfg.shape) * 42.0
        ds = to_xarray({"total": grid}, cfg)
        enc = ZarrEncoding(compressor="blosc", dtype="float64")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.zarr"
            save_zarr(ds, path, encoding=enc)
            loaded = load_zarr(path)
            np.testing.assert_allclose(
                ds["total"].values, loaded["total"].values,
            )

    def test_roundtrip_no_compression(self):
        """Zarr save/load with compression disabled."""
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid = np.ones(cfg.shape) * 7.0
        ds = to_xarray({"total": grid}, cfg)
        enc = ZarrEncoding(compressor="none", dtype="float64")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.zarr"
            save_zarr(ds, path, encoding=enc)
            loaded = load_zarr(path)
            np.testing.assert_allclose(
                ds["total"].values, loaded["total"].values,
            )

    def test_frozen(self):
        enc = ZarrEncoding()
        with pytest.raises(AttributeError):
            enc.compressor = "blosc"  # type: ignore[misc]


# ============================================================================
# Stats mode
# ============================================================================


class TestStatsMode:
    def test_stats_returns_tuple(self):
        grid, info = accumulate_dwell_time(
            x=[10.0], y=[0.0], z=[0.0], stats=True,
        )
        assert isinstance(grid, np.ndarray)
        assert isinstance(info, dict)
        assert info["n_total"] == 1
        assert info["n_in_range"] == 1
        assert info["n_out_of_range"] == 0

    def test_stats_counts_out_of_range(self):
        cfg = DwellGridConfig(n_r=10, n_lat=18, n_lt=12, r_range=(0, 30))
        grid, info = accumulate_dwell_time(
            x=[10.0, 100.0], y=[0.0, 0.0], z=[0.0, 0.0],
            config=cfg, stats=True,
        )
        assert info["n_total"] == 2
        assert info["n_in_range"] == 1
        assert info["n_out_of_range"] == 1
        assert info["r_out_high"] == 1
        assert info["r_max_observed"] == pytest.approx(100.0)

    def test_stats_false_returns_array(self):
        result = accumulate_dwell_time(x=[10.0], y=[0.0], z=[0.0], stats=False)
        assert isinstance(result, np.ndarray)

    def test_stats_empty(self):
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6)
        grid, info = accumulate_dwell_time([], [], [], config=cfg, stats=True)
        assert info["n_total"] == 0
        assert info["pct_in_range"] == 0.0


# ============================================================================
# Extended metadata in xarray
# ============================================================================


class TestExtendedMetadata:
    def test_tracing_config_in_attrs(self):
        from qp.dwell.tracing import TracingConfig
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid = np.ones(cfg.shape)
        tc = TracingConfig(step=0.05, max_radius=80.0)
        ds = to_xarray({"total": grid}, cfg, tracing_config=tc)
        assert ds.attrs["trace_step_RS"] == 0.05
        assert ds.attrs["trace_max_radius_RS"] == 80.0

    def test_field_config_in_attrs(self):
        from qp.fieldline.kmag_model import SaturnFieldConfig
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid = np.ones(cfg.shape)
        fc = SaturnFieldConfig(dp=0.02, by_imf=-0.3, bz_imf=0.2)
        ds = to_xarray({"total": grid}, cfg, field_config=fc)
        assert ds.attrs["dp_nPa"] == 0.02
        assert ds.attrs["by_imf_nT"] == -0.3
        assert ds.attrs["bz_imf_nT"] == 0.2

    def test_no_extra_attrs_when_configs_none(self):
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid = np.ones(cfg.shape)
        ds = to_xarray({"total": grid}, cfg)
        assert "trace_step_RS" not in ds.attrs
        assert "dp_nPa" not in ds.attrs


# ============================================================================
# Region naming and unknown code handling
# ============================================================================


class TestRegionCodes:
    @pytest.fixture
    def small_config(self):
        return DwellGridConfig(
            n_r=10, n_lat=18, n_lt=12,
            r_range=(0, 30), lat_range=(-90, 90), lt_range=(0, 24),
        )

    def test_unknown_code_in_total(self, small_config):
        """Unexpected region codes (e.g., 5) should still be in total."""
        x = np.array([10.0, 15.0])
        y = np.array([0.0, 5.0])
        z = np.array([0.0, 1.0])
        codes = np.array([0, 5])  # 0=MS, 5=unrecognized

        grids = accumulate_with_regions(x, y, z, codes, dt_minutes=1.0, config=small_config)
        # Total should include both points
        assert grids["total"].sum() == pytest.approx(2.0)
        # Magnetosphere should include only code 0
        assert grids["magnetosphere"].sum() == pytest.approx(1.0)
        # No per-region grid for code 5
        region_sum = sum(grids[k].sum() for k in ["magnetosphere", "magnetosheath", "solar_wind", "unknown"])
        assert region_sum == pytest.approx(1.0)  # only the code=0 point

    def test_region_names_match_crossings(self):
        """REGION_CODES in grid.py should use the same code values as crossings.py."""
        from qp.dwell.grid import REGION_CODES
        from qp.io.crossings import MS, SH, SW, UNKNOWN
        assert MS in REGION_CODES
        assert SH in REGION_CODES
        assert SW in REGION_CODES
        assert UNKNOWN in REGION_CODES

    def test_region_names_are_descriptive(self):
        """Region names should be human-readable, not abbreviations."""
        from qp.dwell.grid import REGION_CODES
        abbreviations = {"ms", "sh", "sw", "unk"}
        for name in REGION_CODES.values():
            assert name not in abbreviations, f"Expected descriptive name, got '{name}'"


# ============================================================================
# Dipole invariant latitude
# ============================================================================


class TestDipoleInvariantLatitude:
    def test_equatorial_point(self):
        """Equatorial point at r=10 → L=10, inv_lat ≈ 71.57°."""
        from qp.coords.ksm import dipole_invariant_latitude
        inv = dipole_invariant_latitude(10.0, 0.0, 0.037)  # at dipole equator
        expected = np.degrees(np.arccos(1 / np.sqrt(10)))
        assert inv == pytest.approx(expected, abs=0.5)

    def test_inside_planet(self):
        """Point inside planet (L < 1) should return NaN."""
        from qp.coords.ksm import dipole_invariant_latitude
        inv = dipole_invariant_latitude(0.5, 0.0, 0.037)
        assert np.isnan(inv)

    def test_sign_matches_hemisphere(self):
        """Northern point → positive inv lat, southern → negative."""
        from qp.coords.ksm import dipole_invariant_latitude
        north = dipole_invariant_latitude(10.0, 0.0, 5.0)
        south = dipole_invariant_latitude(10.0, 0.0, -5.0)
        assert north > 0
        assert south < 0

    def test_vectorized(self):
        """Should handle arrays."""
        from qp.coords.ksm import dipole_invariant_latitude
        x = np.array([10.0, 20.0, 0.5])
        y = np.zeros(3)
        z = np.full(3, 0.037)
        inv = dipole_invariant_latitude(x, y, z)
        assert len(inv) == 3
        assert np.isfinite(inv[0])
        assert np.isfinite(inv[1])
        assert np.isnan(inv[2])  # inside planet

    def test_higher_r_gives_higher_inv_lat(self):
        """At equator, larger r → larger L → larger inv_lat."""
        from qp.coords.ksm import dipole_invariant_latitude
        inv_5 = dipole_invariant_latitude(5.0, 0.0, 0.037)
        inv_20 = dipole_invariant_latitude(20.0, 0.0, 0.037)
        assert inv_20 > inv_5


# ============================================================================
# Invariant latitude grid accumulation
# ============================================================================


class TestAccumulateInvLatGrid:
    @pytest.fixture
    def small_config(self):
        return DwellGridConfig(
            n_r=10, n_lat=18, n_lt=12,
            r_range=(0, 30), lat_range=(-90, 90), lt_range=(0, 24),
        )

    def test_returns_total(self, small_config):
        from qp.dwell.grid import accumulate_inv_lat_grid
        result = accumulate_inv_lat_grid(
            [10.0], [0.0], [0.0], dt_minutes=1.0, config=small_config,
        )
        assert "total" in result
        assert result["total"].shape == (18, 12)

    def test_time_conservation(self, small_config):
        """Total accumulated time should be close to n_points × dt for valid points."""
        from qp.dwell.grid import accumulate_inv_lat_grid
        # Points at r=10, equator → all have L=10 > 1, all valid
        n = 100
        rng = np.random.default_rng(42)
        phi = rng.uniform(0, 2 * np.pi, n)
        x = 10.0 * np.cos(phi)
        y = 10.0 * np.sin(phi)
        z = np.full(n, 0.037)
        result = accumulate_inv_lat_grid(x, y, z, dt_minutes=1.0, config=small_config)
        assert result["total"].sum() == pytest.approx(n * 1.0)

    def test_with_region_codes(self, small_config):
        from qp.dwell.grid import accumulate_inv_lat_grid
        result = accumulate_inv_lat_grid(
            [10.0, 15.0], [0.0, 0.0], [0.0, 0.0],
            region_codes=[0, 1], config=small_config,
        )
        assert "magnetosphere" in result
        assert "magnetosheath" in result
        assert result["magnetosphere"].sum() == pytest.approx(1.0)
        assert result["magnetosheath"].sum() == pytest.approx(1.0)


# ============================================================================
# TracingResult
# ============================================================================


class TestTracingResult:
    def test_frozen(self):
        result = TracingResult(
            inv_lat_north=np.array([60.0]),
            inv_lat_south=np.array([-60.0]),
            is_closed=np.array([True]),
            l_equatorial=np.array([15.0]),
            n_traces=1,
            n_closed=1,
        )
        with pytest.raises(AttributeError):
            result.n_traces = 5  # type: ignore[misc]

    def test_fields(self):
        result = TracingResult(
            inv_lat_north=np.array([60.0, np.nan]),
            inv_lat_south=np.array([-60.0, np.nan]),
            is_closed=np.array([True, False]),
            l_equatorial=np.array([15.0, np.nan]),
            n_traces=2,
            n_closed=1,
        )
        assert result.n_traces == 2
        assert result.n_closed == 1
        assert np.isnan(result.l_equatorial[1])


# ============================================================================
# accumulate_traced_inv_lat_grid
# ============================================================================


class TestAccumulateTracedInvLatGrid:
    @pytest.fixture
    def small_config(self):
        return DwellGridConfig(
            n_r=10, n_lat=18, n_lt=12,
            r_range=(0, 30), lat_range=(-90, 90), lt_range=(0, 24),
        )

    def test_basic_accumulation(self, small_config):
        """Total dwell = n_traces × dt_minutes for all-valid, all-closed traces."""
        n = 10
        result = accumulate_traced_inv_lat_grid(
            inv_lat_north=np.full(n, 65.0),
            inv_lat_south=np.full(n, -65.0),
            is_closed=np.ones(n, dtype=bool),
            local_time=np.full(n, 12.0),
            z=np.full(n, 5.0),  # north hemisphere
            dt_minutes=60.0,
            config=small_config,
        )
        assert "total" in result
        assert result["total"].shape == (18, 12)
        assert result["total"].sum() == pytest.approx(n * 60.0)

    def test_conjugate_convention_north(self, small_config):
        """Spacecraft in north (z > 0) → uses inv_lat_north (positive)."""
        result = accumulate_traced_inv_lat_grid(
            inv_lat_north=np.array([65.0]),
            inv_lat_south=np.array([-65.0]),
            is_closed=np.array([True]),
            local_time=np.array([12.0]),
            z=np.array([5.0]),  # north hemisphere
            dt_minutes=60.0,
            config=small_config,
        )
        grid = result["total"]
        idx = np.argwhere(grid > 0)
        assert len(idx) == 1
        i_lat = idx[0, 0]
        # 65° in [-90, 90] with 18 bins → should be in upper half (bin > 9)
        assert i_lat > 9

    def test_conjugate_convention_south(self, small_config):
        """Spacecraft in south (z < 0) → uses inv_lat_south (negative)."""
        result = accumulate_traced_inv_lat_grid(
            inv_lat_north=np.array([65.0]),
            inv_lat_south=np.array([-65.0]),
            is_closed=np.array([True]),
            local_time=np.array([12.0]),
            z=np.array([-5.0]),  # south hemisphere
            dt_minutes=60.0,
            config=small_config,
        )
        grid = result["total"]
        idx = np.argwhere(grid > 0)
        assert len(idx) == 1
        i_lat = idx[0, 0]
        # -65° → should be in lower half (bin < 9)
        assert i_lat < 9

    def test_closed_only(self, small_config):
        """closed_only=True excludes open field lines."""
        n = 5
        inv_n = np.array([65.0, 65.0, np.nan, np.nan, 65.0])
        inv_s = np.array([-65.0, -65.0, np.nan, np.nan, -65.0])
        closed = np.array([True, True, False, False, True])

        all_result = accumulate_traced_inv_lat_grid(
            inv_n, inv_s, closed,
            local_time=np.full(n, 12.0), z=np.full(n, 5.0),
            dt_minutes=60.0, config=small_config,
        )
        closed_result = accumulate_traced_inv_lat_grid(
            inv_n, inv_s, closed,
            local_time=np.full(n, 12.0), z=np.full(n, 5.0),
            dt_minutes=60.0, closed_only=True, config=small_config,
        )
        # Open lines have NaN inv_lat, excluded from both. Same result.
        np.testing.assert_allclose(all_result["total"], closed_result["total"])
        # 3 closed traces × 60 min
        assert closed_result["total"].sum() == pytest.approx(3 * 60.0)

    def test_nan_excluded(self, small_config):
        """Open field lines (NaN inv_lat) contribute zero dwell."""
        result = accumulate_traced_inv_lat_grid(
            inv_lat_north=np.array([np.nan]),
            inv_lat_south=np.array([np.nan]),
            is_closed=np.array([False]),
            local_time=np.array([12.0]),
            z=np.array([5.0]),
            dt_minutes=60.0,
            config=small_config,
        )
        assert result["total"].sum() == pytest.approx(0.0)

    def test_region_separation(self, small_config):
        """Per-region grids should sum to total for known codes."""
        n = 4
        result = accumulate_traced_inv_lat_grid(
            inv_lat_north=np.full(n, 65.0),
            inv_lat_south=np.full(n, -65.0),
            is_closed=np.ones(n, dtype=bool),
            local_time=np.full(n, 12.0),
            z=np.full(n, 5.0),
            dt_minutes=60.0,
            region_codes=np.array([0, 1, 2, 9]),
            config=small_config,
        )
        assert "magnetosphere" in result
        parts = result["magnetosphere"] + result["magnetosheath"] + result["solar_wind"] + result["unknown"]
        np.testing.assert_allclose(result["total"], parts)

    def test_dt_minutes_scales(self, small_config):
        """dt_minutes=120 → each point contributes 120 min."""
        result = accumulate_traced_inv_lat_grid(
            inv_lat_north=np.array([65.0]),
            inv_lat_south=np.array([-65.0]),
            is_closed=np.array([True]),
            local_time=np.array([12.0]),
            z=np.array([5.0]),
            dt_minutes=120.0,
            config=small_config,
        )
        assert result["total"].sum() == pytest.approx(120.0)

    def test_empty_input(self, small_config):
        result = accumulate_traced_inv_lat_grid(
            inv_lat_north=np.array([]),
            inv_lat_south=np.array([]),
            is_closed=np.array([], dtype=bool),
            local_time=np.array([]),
            z=np.array([]),
            dt_minutes=60.0,
            config=small_config,
        )
        assert result["total"].sum() == 0.0
        assert result["total"].shape == (18, 12)


# ============================================================================
# accumulate_weak_field_grid
# ============================================================================


class TestAccumulateWeakFieldGrid:
    @pytest.fixture
    def small_config(self):
        return DwellGridConfig(
            n_r=10, n_lat=18, n_lt=12,
            r_range=(0, 30), lat_range=(-90, 90), lt_range=(0, 24),
        )

    def test_weak_field_filter(self, small_config):
        """Only points with btotal < 2.0 should appear."""
        x = np.array([10.0, 10.0, 10.0])
        y = np.zeros(3)
        z = np.full(3, 0.037)
        btotal = np.array([1.0, 3.0, 0.5])

        result = accumulate_weak_field_grid(
            x, y, z, btotal, dt_minutes=1.0, b_threshold=2.0, config=small_config,
        )
        # 2 of 3 points pass the threshold
        assert result["total"].sum() == pytest.approx(2.0)

    def test_all_pass_with_high_threshold(self, small_config):
        """With b_threshold=1000, all points pass."""
        from qp.dwell.grid import accumulate_inv_lat_grid
        x = np.array([10.0, 15.0])
        y = np.zeros(2)
        z = np.full(2, 0.037)
        btotal = np.array([5.0, 10.0])

        wf = accumulate_weak_field_grid(
            x, y, z, btotal, dt_minutes=1.0, b_threshold=1000.0, config=small_config,
        )
        ref = accumulate_inv_lat_grid(
            x, y, z, dt_minutes=1.0, config=small_config,
        )
        np.testing.assert_allclose(wf["total"], ref["total"])

    def test_with_region_codes(self, small_config):
        x = np.array([10.0, 10.0])
        y = np.zeros(2)
        z = np.full(2, 0.037)
        btotal = np.array([1.0, 1.0])
        codes = np.array([0, 1])

        result = accumulate_weak_field_grid(
            x, y, z, btotal, region_codes=codes, config=small_config,
        )
        assert "magnetosphere" in result
        assert result["magnetosphere"].sum() == pytest.approx(1.0)
        assert result["magnetosheath"].sum() == pytest.approx(1.0)

    def test_empty_input(self, small_config):
        result = accumulate_weak_field_grid(
            [], [], [], [], config=small_config,
        )
        assert result["total"].sum() == 0.0
        assert result["total"].shape == (18, 12)


# ============================================================================
# KMAG grids in xarray I/O
# ============================================================================


class TestKmagInvLatIO:
    def test_kmag_grids_in_dataset(self):
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid_3d = np.ones(cfg.shape)
        inv_2d = np.ones((9, 6)) * 2.0
        kmag_2d = np.ones((9, 6)) * 3.0

        ds = to_xarray(
            {"total": grid_3d}, cfg,
            inv_lat_grids={"dipole_inv_lat_total": inv_2d},
            kmag_inv_lat_grids={"kmag_inv_lat_total": kmag_2d},
        )

        assert "dipole_inv_lat" in ds.dims
        assert "kmag_inv_lat" in ds.dims
        assert "dipole_inv_lat_total" in ds.data_vars
        assert "kmag_inv_lat_total" in ds.data_vars
        np.testing.assert_allclose(ds["kmag_inv_lat_total"].values, 3.0)

    def test_kmag_zarr_roundtrip(self):
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid_3d = np.ones(cfg.shape)
        kmag_2d = np.ones((9, 6)) * 5.0

        ds = to_xarray(
            {"total": grid_3d}, cfg,
            kmag_inv_lat_grids={"kmag_inv_lat_total": kmag_2d},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.zarr"
            save_zarr(ds, path)
            loaded = load_zarr(path)
            np.testing.assert_allclose(
                loaded["kmag_inv_lat_total"].values, 5.0,
            )
            assert "kmag_inv_lat" in loaded.dims

    def test_no_kmag_grids_when_none(self):
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid_3d = np.ones(cfg.shape)
        ds = to_xarray({"total": grid_3d}, cfg, kmag_inv_lat_grids=None)
        assert "kmag_inv_lat" not in ds.dims


# ============================================================================
# Region filter and parallel tracing
# ============================================================================


class TestComputeInvariantLatitudesParallel:
    """Tests for region filtering and the multiprocessing tracer."""

    @pytest.fixture
    def synthetic_inputs(self):
        """20 samples spread across regions with varied positions."""
        rng = np.random.default_rng(42)
        n = 20
        x = rng.uniform(5, 25, n)
        y = rng.uniform(-10, 10, n)
        z = rng.uniform(-10, 10, n)
        t = np.zeros(n)  # J2000=0 for all
        codes = np.tile([0, 0, 0, 1, 2], 4)  # 60% MS, 20% MSh, 20% SW
        return x, y, z, t, codes

    def test_region_filter_skips_non_ms(self, synthetic_inputs):
        """Region filter to MS leaves non-MS slots as NaN/False."""
        from qp.dwell.tracing import (
            TracingConfig, compute_invariant_latitudes,
        )
        x, y, z, t, codes = synthetic_inputs
        cfg = TracingConfig(
            trace_every_n=1, max_steps=5000, log_interval=100,
            region_filter=(0,),
        )
        result = compute_invariant_latitudes(
            x, y, z, t, config=cfg, region_codes=codes,
        )
        # Slots where region != 0 must all be NaN / False
        non_ms = codes != 0
        assert np.all(np.isnan(result.inv_lat_north[non_ms]))
        assert np.all(np.isnan(result.inv_lat_south[non_ms]))
        assert np.all(~result.is_closed[non_ms])

    def test_region_filter_none_traces_all(self, synthetic_inputs):
        """region_filter=None ignores region_codes and traces everything."""
        from qp.dwell.tracing import (
            TracingConfig, compute_invariant_latitudes,
        )
        x, y, z, t, codes = synthetic_inputs
        cfg_none = TracingConfig(
            trace_every_n=1, max_steps=5000, log_interval=100,
            region_filter=None,
        )
        r_codes = compute_invariant_latitudes(
            x, y, z, t, config=cfg_none, region_codes=codes,
        )
        r_no_codes = compute_invariant_latitudes(
            x, y, z, t, config=cfg_none,
        )
        np.testing.assert_array_equal(r_codes.is_closed, r_no_codes.is_closed)
        np.testing.assert_allclose(
            r_codes.inv_lat_north, r_no_codes.inv_lat_north, equal_nan=True,
        )

    def test_parallel_matches_serial_no_filter(self, synthetic_inputs):
        """Parallel must produce element-wise identical results to serial."""
        from qp.dwell.tracing import (
            TracingConfig,
            compute_invariant_latitudes,
            compute_invariant_latitudes_parallel,
        )
        x, y, z, t, _ = synthetic_inputs
        cfg = TracingConfig(
            trace_every_n=1, max_steps=5000, log_interval=100,
            region_filter=None,
        )
        r_serial = compute_invariant_latitudes(x, y, z, t, config=cfg)
        r_parallel = compute_invariant_latitudes_parallel(
            x, y, z, t, config=cfg, n_workers=2,
        )
        np.testing.assert_array_equal(r_serial.is_closed, r_parallel.is_closed)
        np.testing.assert_allclose(
            r_serial.inv_lat_north, r_parallel.inv_lat_north, equal_nan=True,
        )
        np.testing.assert_allclose(
            r_serial.inv_lat_south, r_parallel.inv_lat_south, equal_nan=True,
        )
        np.testing.assert_allclose(
            r_serial.l_equatorial, r_parallel.l_equatorial, equal_nan=True,
        )
        assert r_serial.n_traces == r_parallel.n_traces
        assert r_serial.n_closed == r_parallel.n_closed

    def test_parallel_matches_serial_with_filter(self, synthetic_inputs):
        """Parallel + region filter must match serial + region filter."""
        from qp.dwell.tracing import (
            TracingConfig,
            compute_invariant_latitudes,
            compute_invariant_latitudes_parallel,
        )
        x, y, z, t, codes = synthetic_inputs
        cfg = TracingConfig(
            trace_every_n=1, max_steps=5000, log_interval=100,
            region_filter=(0,),
        )
        r_s = compute_invariant_latitudes(
            x, y, z, t, config=cfg, region_codes=codes,
        )
        r_p = compute_invariant_latitudes_parallel(
            x, y, z, t, config=cfg, region_codes=codes, n_workers=2,
        )
        np.testing.assert_array_equal(r_s.is_closed, r_p.is_closed)
        np.testing.assert_allclose(
            r_s.inv_lat_north, r_p.inv_lat_north, equal_nan=True,
        )
        np.testing.assert_allclose(
            r_s.l_equatorial, r_p.l_equatorial, equal_nan=True,
        )

    def test_parallel_n_workers_1_falls_back_to_serial(self, synthetic_inputs):
        """n_workers=1 should use the serial path (no multiprocessing)."""
        from qp.dwell.tracing import (
            TracingConfig,
            compute_invariant_latitudes,
            compute_invariant_latitudes_parallel,
        )
        x, y, z, t, _ = synthetic_inputs
        cfg = TracingConfig(
            trace_every_n=1, max_steps=5000, log_interval=100,
            region_filter=None,
        )
        r_s = compute_invariant_latitudes(x, y, z, t, config=cfg)
        r_p = compute_invariant_latitudes_parallel(
            x, y, z, t, config=cfg, n_workers=1,
        )
        np.testing.assert_array_equal(r_s.is_closed, r_p.is_closed)
        np.testing.assert_allclose(
            r_s.inv_lat_north, r_p.inv_lat_north, equal_nan=True,
        )

    def test_round_robin_chunks_partition_active_slots(self):
        """Round-robin chunking covers all active slots exactly once."""
        active_slots = np.arange(100)
        n_chunks = 8
        chunks = [active_slots[i::n_chunks] for i in range(n_chunks)]
        union = np.concatenate(chunks)
        assert sorted(union.tolist()) == active_slots.tolist()
        # No duplicates
        assert len(set(union.tolist())) == 100
