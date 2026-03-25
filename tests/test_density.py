"""Tests for plasma density models."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.wavesolver.density import (
    BagenalDelamere,
    PersoonEtAl,
    PowerLawDensity,
    UniformDensity,
    alfven_velocity,
    AMU,
    MU0,
)


class TestBagenalDelamere:
    """Tests for Bagenal & Delamere (2011) density model."""

    @pytest.fixture
    def model(self):
        return BagenalDelamere()

    def test_equatorial_density_at_10_rs(self, model):
        """At L=10, density should be ~2-5 cm⁻³ (2-5e6 m⁻³)."""
        n = model.equatorial_density(10.0)
        assert 1e6 < n < 10e6

    def test_equatorial_density_at_5_rs(self, model):
        """At L=5, density should be ~70-90 cm⁻³."""
        n = model.equatorial_density(5.0)
        assert 50e6 < n < 100e6

    def test_density_decreases_with_distance(self, model):
        """Density should decrease from 5 Rs to 15 Rs."""
        n5 = model.equatorial_density(5.0)
        n10 = model.equatorial_density(10.0)
        n15 = model.equatorial_density(15.0)
        assert n5 > n10 > n15

    def test_scale_height_increases_with_distance(self, model):
        """Scale height should increase with L-shell."""
        h5 = model.scale_height(5.0)
        h10 = model.scale_height(10.0)
        h15 = model.scale_height(15.0)
        assert h5 < h10 < h15

    def test_field_aligned_profile_peaks_at_equator(self, model):
        """Density should peak at the equatorial crossing."""
        s = np.linspace(0, 20, 500)
        s_eq = 10.0
        n = model.field_aligned_density(10.0, s, s_eq)
        peak_idx = np.argmax(n)
        assert abs(s[peak_idx] - s_eq) < 0.5

    def test_field_aligned_profile_symmetric(self, model):
        """Profile should be symmetric about the equator."""
        s = np.linspace(0, 20, 501)
        s_eq = 10.0
        n = model.field_aligned_density(10.0, s, s_eq)
        n_flipped = n[::-1]
        assert_allclose(n, n_flipped, rtol=1e-10)

    def test_ion_mass(self, model):
        """Default ion mass should be 18 amu (water group)."""
        assert model.ion_mass_amu == 18.0
        assert_allclose(model.ion_mass_kg, 18.0 * AMU, rtol=1e-6)


class TestPersoonEtAl:
    """Tests for Persoon et al. (2013) density model."""

    @pytest.fixture
    def model(self):
        return PersoonEtAl()

    def test_equatorial_density_at_5_rs(self, model):
        """At L=5, Persoon density should be ~60-75 cm⁻³."""
        n = model.equatorial_density(5.0)
        assert 40e6 < n < 80e6

    def test_density_decreases_outward(self, model):
        """Density should decrease from L=5 to L=9."""
        assert model.equatorial_density(5.0) > model.equatorial_density(9.0)

    def test_scale_height_positive(self, model):
        """Scale height should be positive at all L."""
        for L in [3.0, 5.0, 7.0, 9.0]:
            assert model.scale_height(L) > 0


class TestPowerLawDensity:
    """Tests for Cummings & Coleman power-law model."""

    def test_equatorial_density_constant(self):
        """Equatorial density should equal n0 regardless of L."""
        model = PowerLawDensity(n0=1e6, m_index=6.0)
        assert model.equatorial_density(5.0) == 1e6
        assert model.equatorial_density(15.0) == 1e6

    def test_field_aligned_returns_n0(self):
        """Simplified field-aligned model returns constant n0."""
        model = PowerLawDensity(n0=2e6)
        s = np.linspace(0, 10, 100)
        n = model.field_aligned_density(10.0, s, 5.0)
        assert_allclose(n, 2e6)


class TestUniformDensity:
    """Tests for constant density model."""

    def test_returns_constant(self):
        model = UniformDensity(n0=5e6)
        assert model.equatorial_density(10.0) == 5e6
        s = np.linspace(0, 100, 50)
        n = model.field_aligned_density(10.0, s, 50.0)
        assert_allclose(n, 5e6)


class TestAlfvenVelocity:
    """Tests for Alfvén speed computation."""

    def test_basic_computation(self):
        """vA = B/sqrt(μ₀·n·m) for non-relativistic case."""
        B = np.array([1e-9])  # 1 nT
        n = np.array([1e6])  # 1 cm⁻³ in m⁻³
        m = 18.0 * AMU  # water group
        va = alfven_velocity(B, n, m)
        # Non-relativistic: B / sqrt(μ₀ n m)
        expected = 1e-9 / math.sqrt(MU0 * 1e6 * 18.0 * AMU)
        assert_allclose(va, expected, rtol=0.01)  # <1% from relativistic correction

    def test_increases_with_field(self):
        """Stronger field → faster Alfvén speed."""
        n = np.array([1e6])
        m = 18.0 * AMU
        va_weak = alfven_velocity(np.array([1e-9]), n, m)
        va_strong = alfven_velocity(np.array([10e-9]), n, m)
        assert va_strong > va_weak

    def test_decreases_with_density(self):
        """Higher density → slower Alfvén speed."""
        B = np.array([5e-9])
        m = 18.0 * AMU
        va_sparse = alfven_velocity(B, np.array([1e4]), m)
        va_dense = alfven_velocity(B, np.array([1e8]), m)
        assert va_sparse > va_dense

    def test_capped_at_speed_of_light(self):
        """In vacuum (n→0), vA should approach c."""
        B = np.array([1e-9])
        n = np.array([1e-20])  # near vacuum
        m = AMU
        va = alfven_velocity(B, n, m)
        assert va[0] < 3e8  # less than c
        assert va[0] > 2e8  # but close

    def test_array_input(self):
        """Should handle array inputs."""
        B = np.array([1e-9, 5e-9, 10e-9])
        n = np.array([1e6, 1e6, 1e6])
        m = 18.0 * AMU
        va = alfven_velocity(B, n, m)
        assert va.shape == (3,)
        assert va[0] < va[1] < va[2]
