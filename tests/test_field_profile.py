"""Tests for field line profile computation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.fieldline.tracer import (
    dipole_field,
    trace_fieldline_bidirectional,
    saturn_field_wrapper,
)
from qp.fieldline.kmag_model import SaturnField
from qp.wavesolver.density import (
    BagenalDelamere,
    UniformDensity,
)
from qp.wavesolver.field_profile import compute_field_line_profile


@pytest.fixture(scope="module")
def dipole_trace_10rs():
    """Bidirectional dipole field line trace from L=10 Rs equator."""
    return trace_fieldline_bidirectional(
        dipole_field,
        [10.0, 0.0, 0.0],
        step=0.1,
        max_steps=10000,
    )


@pytest.fixture(scope="module")
def dipole_profile_uniform(dipole_trace_10rs):
    """Field line profile with uniform density on a dipole trace."""
    return compute_field_line_profile(
        dipole_trace_10rs,
        UniformDensity(n0=1e6),
    )


@pytest.fixture(scope="module")
def dipole_profile_bagenal(dipole_trace_10rs):
    """Field line profile with Bagenal density on a dipole trace."""
    return compute_field_line_profile(
        dipole_trace_10rs,
        BagenalDelamere(),
    )


class TestArcLength:
    """Tests for arc length computation."""

    def test_starts_at_zero(self, dipole_profile_uniform):
        """Arc length should start at 0."""
        assert dipole_profile_uniform.arc_length[0] == 0.0

    def test_monotonically_increasing(self, dipole_profile_uniform):
        """Arc length should be strictly increasing."""
        ds = np.diff(dipole_profile_uniform.arc_length)
        assert np.all(ds > 0)

    def test_reasonable_length(self, dipole_profile_uniform):
        """A dipole field line at L=10 should be ~30-60 Rs long."""
        length = dipole_profile_uniform.length_rs
        assert 20.0 < length < 80.0


class TestLShellAndLatitude:
    """Tests for L-shell and conjugate latitude."""

    def test_l_shell_near_10(self, dipole_profile_uniform):
        """L-shell should be close to 10 for a trace starting at [10,0,0]."""
        assert_allclose(dipole_profile_uniform.l_shell, 10.0, atol=0.5)

    def test_conjugate_latitude_physical(self, dipole_profile_uniform):
        """Conjugate latitude should be between 50 and 90 degrees."""
        lat = dipole_profile_uniform.conjugate_latitude
        assert 50.0 < abs(lat) < 90.0


class TestFieldMagnitude:
    """Tests for B-field along the profile."""

    def test_field_positive(self, dipole_profile_uniform):
        """Field magnitude should be positive everywhere."""
        assert np.all(dipole_profile_uniform.field_magnitude > 0)

    def test_field_minimum_near_equator(self, dipole_profile_uniform):
        """B should be weakest near the equatorial crossing."""
        B = dipole_profile_uniform.field_magnitude
        eq_idx = dipole_profile_uniform.equator_index
        # B at equator should be less than B at both endpoints
        assert B[eq_idx] < B[0]
        assert B[eq_idx] < B[-1]

    def test_field_in_tesla(self, dipole_profile_uniform):
        """Field at L=10 equator should be ~O(10 nT) = O(1e-8 T)."""
        B_eq = dipole_profile_uniform.field_magnitude[
            dipole_profile_uniform.equator_index
        ]
        assert 1e-10 < B_eq < 1e-6


class TestDensity:
    """Tests for density profile."""

    def test_uniform_density_constant(self, dipole_profile_uniform):
        """Uniform density model should give constant density."""
        assert_allclose(dipole_profile_uniform.density, 1e6, rtol=1e-10)

    def test_bagenal_density_peaks_at_equator(self, dipole_profile_bagenal):
        """Bagenal density should peak near the equatorial crossing."""
        n = dipole_profile_bagenal.density
        eq_idx = dipole_profile_bagenal.equator_index
        # Equatorial density should be a local max
        assert n[eq_idx] > n[max(eq_idx - 20, 0)]
        assert n[eq_idx] > n[min(eq_idx + 20, len(n) - 1)]

    def test_bagenal_density_positive(self, dipole_profile_bagenal):
        """Density should be positive everywhere."""
        assert np.all(dipole_profile_bagenal.density > 0)


class TestAlfvenVelocity:
    """Tests for Alfvén velocity profile."""

    def test_va_positive(self, dipole_profile_uniform):
        """Alfvén speed should be positive everywhere."""
        assert np.all(dipole_profile_uniform.alfven_velocity_profile > 0)

    def test_va_faster_at_poles(self, dipole_profile_bagenal):
        """With Bagenal density (peaking at equator), vA should be faster near poles."""
        va = dipole_profile_bagenal.alfven_velocity_profile
        eq_idx = dipole_profile_bagenal.equator_index
        # vA near poles > vA at equator (density drops, B increases)
        assert va[5] > va[eq_idx]
        assert va[-5] > va[eq_idx]


class TestScaleFactors:
    """Tests for scale factors h1 and h2."""

    def test_h1_maximum_at_equator(self, dipole_profile_uniform):
        """h1 = r·sin(θ) should be maximum at the equator for a dipole."""
        h1 = dipole_profile_uniform.h1
        eq_idx = dipole_profile_uniform.equator_index
        # h1 at equator should be large (r=10 Rs, sin(θ)=1)
        assert h1[eq_idx] > h1[5]
        assert h1[eq_idx] > h1[-5]

    def test_h1_h2_positive(self, dipole_profile_uniform):
        """Scale factors should be positive."""
        assert np.all(dipole_profile_uniform.h1 > 0)
        assert np.all(dipole_profile_uniform.h2 > 0)


class TestSplineInterpolants:
    """Tests for the precomputed spline interpolants."""

    def test_va_spline_matches_data(self, dipole_profile_uniform):
        """vA spline should reproduce the data points."""
        p = dipole_profile_uniform
        va_interp = p.va_spline(p.arc_length)
        assert_allclose(va_interp, p.alfven_velocity_profile, rtol=1e-6)

    def test_dlnh_splines_finite(self, dipole_profile_uniform):
        """dlnh splines should return finite values."""
        p = dipole_profile_uniform
        s_mid = 0.5 * (p.arc_length[0] + p.arc_length[-1])
        assert math.isfinite(p.dlnh1B_spline(s_mid))
        assert math.isfinite(p.dlnh2B_spline(s_mid))

    def test_s_span_meters(self, dipole_profile_uniform):
        """s_span should give valid integration bounds."""
        s_min, s_max = dipole_profile_uniform.s_span_meters
        assert s_min == 0.0
        assert s_max > 0.0
        assert s_max > 1e9  # at least ~1 million km for L=10


class TestWithKMAG:
    """Test field profile with KMAG field model (integration test)."""

    def test_kmag_profile_computes(self):
        """Should successfully compute a profile from a KMAG trace."""
        field = SaturnField()
        field_func = saturn_field_wrapper(field, time=284040000.0, coord="KSM")
        trace = trace_fieldline_bidirectional(
            field_func,
            [8.0, 0.0, 0.0],
            step=0.1,
            max_steps=5000,
        )
        profile = compute_field_line_profile(trace, BagenalDelamere())

        assert profile.l_shell > 5.0
        assert len(profile.arc_length) == len(trace.positions)
        assert np.all(profile.alfven_velocity_profile > 0)
        assert np.all(np.isfinite(profile.density))
