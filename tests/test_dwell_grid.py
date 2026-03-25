"""Tests for qp.dwell — spherical dwell-time grid accumulation and I/O."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qp.dwell.grid import DwellGridConfig, accumulate_dwell_time, accumulate_with_regions
from qp.dwell.io import load_zarr, save_zarr, to_xarray


# ============================================================================
# DwellGridConfig
# ============================================================================


class TestDwellGridConfig:
    def test_default_shape(self):
        cfg = DwellGridConfig()
        assert cfg.shape == (70, 90, 48)

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
        assert grid.shape == (70, 90, 48)
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
        parts = grids["ms"] + grids["sh"] + grids["sw"] + grids["unknown"]
        np.testing.assert_allclose(total, parts)

    def test_all_ms(self, small_config):
        """All points in MS → ms grid equals total."""
        x = np.array([10.0, 15.0])
        y = np.array([0.0, 5.0])
        z = np.array([0.0, 1.0])
        codes = np.array([0, 0])

        grids = accumulate_with_regions(x, y, z, codes, dt_minutes=1.0, config=small_config)
        np.testing.assert_allclose(grids["ms"], grids["total"])
        assert grids["sh"].sum() == 0.0
        assert grids["sw"].sum() == 0.0

    def test_returns_all_keys(self, small_config):
        grids = accumulate_with_regions(
            [10.0], [0.0], [0.0], [0], dt_minutes=1.0, config=small_config,
        )
        assert set(grids.keys()) == {"total", "ms", "sh", "sw", "unknown"}


# ============================================================================
# xarray I/O
# ============================================================================


class TestXarrayIO:
    @pytest.fixture
    def sample_dataset(self):
        cfg = DwellGridConfig(n_r=5, n_lat=9, n_lt=6, r_range=(0, 15))
        grid = np.random.default_rng(42).uniform(0, 10, cfg.shape)
        return to_xarray({"total": grid, "ms": grid * 0.7}, cfg, attrs={"source": "test"})

    def test_dimensions(self, sample_dataset):
        assert list(sample_dataset.dims) == ["r", "magnetic_latitude", "local_time"]

    def test_coordinate_values(self, sample_dataset):
        assert len(sample_dataset.coords["r"]) == 5
        assert len(sample_dataset.coords["magnetic_latitude"]) == 9
        assert len(sample_dataset.coords["local_time"]) == 6

    def test_data_variables(self, sample_dataset):
        assert "total" in sample_dataset.data_vars
        assert "ms" in sample_dataset.data_vars

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
                sample_dataset["ms"].values, loaded["ms"].values,
            )
            assert loaded.attrs["source"] == "test"

    def test_coordinate_metadata(self, sample_dataset):
        assert sample_dataset.coords["r"].attrs["units"] == "R_S"
        assert "latitude" in sample_dataset.coords["magnetic_latitude"].attrs["long_name"].lower()


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
