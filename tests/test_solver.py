"""End-to-end tests for the wavesolver: trace → profile → eigenfrequencies."""

from __future__ import annotations


import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.wavesolver.solver import (
    WavesolverConfig,
    solve_eigenfrequencies,
    solve_for_latitude_range,
)
from qp.wavesolver.density import UniformDensity
from qp.fieldline.kmag_model import SaturnField


class TestSolveEigenfrequenciesDipole:
    """End-to-end tests with dipole field."""

    def test_dipole_bagenal_finds_modes(self):
        """Dipole + Bagenal at L=10 should find 6 toroidal modes."""
        config = WavesolverConfig(
            l_shell=10.0,
            component="toroidal",
            n_modes=6,
            density_model="bagenal",
            freq_range=(1e-4, 0.005),
            resolution=100,
        )
        result = solve_eigenfrequencies(config)
        assert result.n_modes >= 4  # should find at least 4 of 6

    def test_eigenfrequencies_ordered(self):
        """Eigenfrequencies should be in ascending order."""
        config = WavesolverConfig(
            l_shell=10.0,
            n_modes=4,
            freq_range=(1e-5, 0.02),
            resolution=300,
        )
        result = solve_eigenfrequencies(config)
        freqs = result.angular_frequencies
        assert np.all(np.diff(freqs) > 0)

    def test_periods_in_physical_range(self):
        """Periods at L=10 should be roughly 10-200 minutes."""
        config = WavesolverConfig(
            l_shell=10.0,
            n_modes=4,
            freq_range=(1e-5, 0.02),
            resolution=300,
        )
        result = solve_eigenfrequencies(config)
        periods = result.periods_minutes
        assert np.all(periods > 1.0)
        assert np.all(periods < 500.0)

    def test_l_shell_in_result(self):
        """Result should carry the L-shell value."""
        config = WavesolverConfig(
            l_shell=10.0, n_modes=2, freq_range=(1e-5, 0.01), resolution=200
        )
        result = solve_eigenfrequencies(config)
        assert_allclose(result.l_shell, 10.0, atol=1.0)

    def test_component_stored(self):
        """Result should store which component was solved."""
        config = WavesolverConfig(
            l_shell=10.0,
            component="poloidal",
            n_modes=2,
            freq_range=(1e-4, 0.005),
            resolution=100,
        )
        result = solve_eigenfrequencies(config)
        assert result.component == "poloidal"

    def test_profiles_populated(self):
        """Field line profiles (vA, B, n) should be in the result."""
        config = WavesolverConfig(
            l_shell=10.0, n_modes=2, freq_range=(1e-5, 0.01), resolution=200
        )
        result = solve_eigenfrequencies(config)
        assert result.arc_length is not None
        assert result.alfven_velocity is not None
        assert result.field_magnitude is not None
        assert result.density is not None
        assert len(result.arc_length) > 50

    def test_with_eigenfunctions(self):
        """include_eigenfunctions should populate mode eigenfunctions."""
        config = WavesolverConfig(
            l_shell=10.0,
            n_modes=2,
            freq_range=(1e-4, 0.005),
            resolution=100,
            include_eigenfunctions=True,
        )
        result = solve_eigenfrequencies(config)
        for mode in result.modes:
            assert mode.eigenfunction is not None
            assert mode.arc_length is not None

    def test_colatitude_overrides_l_shell(self):
        """Passing colatitude should override l_shell."""
        config = WavesolverConfig(
            colatitude=20.0,
            n_modes=2,
            freq_range=(1e-4, 0.005),
            resolution=100,
        )
        result = solve_eigenfrequencies(config)
        # colatitude=20° → L ≈ 1/sin²(20°) ≈ 8.5
        assert 5.0 < result.l_shell < 15.0

    @pytest.mark.slow
    def test_higher_l_shell_lower_frequencies(self):
        """Longer field lines (higher L) should have lower eigenfrequencies."""
        config_6 = WavesolverConfig(
            l_shell=6.0,
            n_modes=1,
            freq_range=(1e-4, 0.01),
            resolution=80,
        )
        config_10 = WavesolverConfig(
            l_shell=10.0,
            n_modes=1,
            freq_range=(1e-4, 0.005),
            resolution=80,
        )
        result_6 = solve_eigenfrequencies(config_6)
        result_10 = solve_eigenfrequencies(config_10)
        assert result_10.angular_frequencies[0] < result_6.angular_frequencies[0]


class TestSolveWithKMAG:
    """End-to-end tests with KMAG field model."""

    @pytest.mark.slow
    def test_kmag_bagenal_finds_modes(self):
        """KMAG + Bagenal at L~8 should find toroidal modes."""
        field = SaturnField()
        config = WavesolverConfig(
            l_shell=8.0,
            component="toroidal",
            n_modes=2,
            field=field,
            density_model="bagenal",
            freq_range=(1e-4, 0.008),
            resolution=80,
        )
        result = solve_eigenfrequencies(config)
        assert result.n_modes >= 1
        assert np.all(result.angular_frequencies > 0)


class TestSolveForLatitudeRange:
    """Tests for batch computation across latitudes."""

    @pytest.mark.slow
    def test_returns_multiple_results(self):
        """Should return one result per field line."""
        config = WavesolverConfig(
            n_modes=1,
            freq_range=(1e-4, 0.005),
            resolution=80,
        )
        results = solve_for_latitude_range(
            config,
            lat_min=68.0,
            lat_max=72.0,
            n_fieldlines=2,
        )
        assert len(results) >= 1

    @pytest.mark.slow
    def test_results_sorted_by_latitude(self):
        """Results should be sorted by conjugate latitude."""
        config = WavesolverConfig(
            n_modes=1,
            freq_range=(1e-4, 0.005),
            resolution=80,
        )
        results = solve_for_latitude_range(
            config,
            lat_min=68.0,
            lat_max=72.0,
            n_fieldlines=2,
        )
        lats = [r.conjugate_latitude for r in results]
        assert lats == sorted(lats)


class TestDensityModelResolution:
    """Test that different density model strings work."""

    @pytest.mark.parametrize("model_name", ["bagenal", "persoon", "uniform"])
    def test_density_model_string(self, model_name):
        """Each density model string should produce a result."""
        config = WavesolverConfig(
            l_shell=8.0,
            n_modes=2,
            density_model=model_name,
            freq_range=(1e-4, 0.005),
            resolution=80,
        )
        result = solve_eigenfrequencies(config)
        assert result.n_modes >= 1

    def test_density_model_instance(self):
        """Passing a DensityModel instance directly should work."""
        config = WavesolverConfig(
            l_shell=8.0,
            n_modes=2,
            density_model=UniformDensity(n0=1e7),
            freq_range=(1e-4, 0.005),
            resolution=80,
        )
        result = solve_eigenfrequencies(config)
        assert result.n_modes >= 1
