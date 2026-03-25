"""Tests for the wavesolver Phase 1: wave equation + eigensolver.

The key validation: in a uniform medium (constant vA, no geometry effects),
the eigenfrequencies are exactly ω_n = n·π·vA / L. This provides an
analytical reference with no ambiguity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import CubicSpline

from qp.wavesolver.wave_equation import (
    _scan_boundary_errors,
    integrate_wave_equation,
    boundary_error,
    count_mode_number,
)
from qp.wavesolver.eigensolver import find_eigenfrequencies
from qp.wavesolver.result import EigenMode, EigenResult


# ============================================================================
# Fixtures: uniform medium
# ============================================================================


@pytest.fixture
def uniform_medium():
    """A uniform medium with constant vA and no geometry effects.

    Analytical eigenfrequencies: ω_n = n·π·vA/L
    for y(0)=0, y(L)=0 boundary conditions.
    """
    va_const = 1e6  # 1000 km/s Alfvén speed
    L = 1e9  # 1 million km field line length
    n_points = 500

    s = np.linspace(0, L, n_points)
    va = np.full_like(s, va_const)
    dlnh = np.zeros_like(s)  # no geometry: d/ds ln(h²B) = 0

    va_spline = CubicSpline(s, va)
    dlnh_spline = CubicSpline(s, dlnh)

    # Analytical eigenfrequencies
    analytical = [n * math.pi * va_const / L for n in range(1, 7)]

    return {
        "s_span": (0.0, L),
        "va_spline": va_spline,
        "dlnh_spline": dlnh_spline,
        "va": va_const,
        "L": L,
        "analytical_freqs": analytical,
    }


@pytest.fixture
def nonuniform_medium():
    """A medium with sinusoidal vA variation (no analytical solution,
    but eigenfrequencies should still be real and ordered).
    """
    L = 1e9
    n_points = 500
    s = np.linspace(0, L, n_points)

    # vA varies sinusoidally: faster at equator, slower at poles
    va_mean = 1e6
    va = va_mean * (1.0 + 0.5 * np.sin(np.pi * s / L))
    dlnh = np.zeros_like(s)

    return {
        "s_span": (0.0, L),
        "va_spline": CubicSpline(s, va),
        "dlnh_spline": CubicSpline(s, dlnh),
    }


# ============================================================================
# Wave equation integration tests
# ============================================================================


class TestIntegrateWaveEquation:
    """Tests for the ODE integrator."""

    def test_integration_returns_arrays(self, uniform_medium):
        """Should return s, y, dy arrays of the right shape."""
        m = uniform_medium
        s, y, dy = integrate_wave_equation(
            omega=m["analytical_freqs"][0],
            s_span=m["s_span"],
            va_spline=m["va_spline"],
            dlnh_spline=m["dlnh_spline"],
            n_points=100,
        )
        assert s.shape == (100,)
        assert y.shape == (100,)
        assert dy.shape == (100,)

    def test_initial_conditions(self, uniform_medium):
        """First point should match initial conditions y(0)=0, y'(0)=1."""
        m = uniform_medium
        s, y, dy = integrate_wave_equation(
            omega=m["analytical_freqs"][0],
            s_span=m["s_span"],
            va_spline=m["va_spline"],
            dlnh_spline=m["dlnh_spline"],
        )
        assert_allclose(y[0], 0.0, atol=1e-12)
        assert_allclose(dy[0], 1.0, atol=1e-8)

    def test_eigenfunction_at_eigenfrequency(self, uniform_medium):
        """At an exact eigenfrequency, y(L) should be small."""
        m = uniform_medium
        omega_1 = m["analytical_freqs"][0]  # fundamental
        s, y, dy = integrate_wave_equation(
            omega=omega_1,
            s_span=m["s_span"],
            va_spline=m["va_spline"],
            dlnh_spline=m["dlnh_spline"],
            n_points=1000,
            rtol=1e-10,
            atol=1e-12,
        )
        # y(L) should be close to 0 at the exact eigenfrequency
        # (limited by spline interpolation of the constant vA)
        assert abs(y[-1]) < 0.5

    def test_eigenfunction_shape_fundamental(self, uniform_medium):
        """Fundamental mode: y should be a half-sine (one peak, no internal zeros)."""
        m = uniform_medium
        omega_1 = m["analytical_freqs"][0]
        s, y, dy = integrate_wave_equation(
            omega=omega_1,
            s_span=m["s_span"],
            va_spline=m["va_spline"],
            dlnh_spline=m["dlnh_spline"],
            n_points=500,
        )
        # Interior should be all positive (half-sine)
        interior = y[10:-10]
        assert np.all(interior > -0.01)  # small tolerance for numerical error


# ============================================================================
# Boundary error tests
# ============================================================================


class TestBoundaryError:
    """Tests for the boundary error function."""

    def test_error_sign_changes_near_eigenfrequency(self, uniform_medium):
        """Boundary error should change sign near an eigenfrequency."""
        m = uniform_medium
        w = m["analytical_freqs"][0]
        err_lo = boundary_error(w * 0.95, m["s_span"], m["va_spline"], m["dlnh_spline"])
        err_hi = boundary_error(w * 1.05, m["s_span"], m["va_spline"], m["dlnh_spline"])
        assert err_lo * err_hi < 0

    def test_error_nonzero_off_eigenfrequency(self, uniform_medium):
        """Boundary error should be nonzero away from eigenfrequencies."""
        m = uniform_medium
        # Midpoint between mode 1 and mode 2
        omega_mid = 0.5 * (m["analytical_freqs"][0] + m["analytical_freqs"][1])
        err = boundary_error(
            omega_mid,
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
        )
        assert abs(err) > 1e-3

    def test_error_changes_sign_between_modes(self, uniform_medium):
        """Error function should change sign between consecutive eigenfrequencies."""
        m = uniform_medium
        for i in range(3):
            w1 = m["analytical_freqs"][i]
            # Error at eigenfrequency is ~0, but slightly off it should have
            # opposite signs on either side
            err_lo = boundary_error(
                w1 * 0.95,
                m["s_span"],
                m["va_spline"],
                m["dlnh_spline"],
            )
            err_hi = boundary_error(
                w1 * 1.05,
                m["s_span"],
                m["va_spline"],
                m["dlnh_spline"],
            )
            # Sign should change near each eigenfrequency
            assert err_lo * err_hi < 0, (
                f"No sign change near mode {i + 1}: err({w1 * 0.95:.4e})={err_lo:.4e}, "
                f"err({w1 * 1.05:.4e})={err_hi:.4e}"
            )


# ============================================================================
# Mode counting tests
# ============================================================================


class TestModeNumber:
    """Tests for mode number identification."""

    def test_fundamental_mode(self, uniform_medium):
        """Fundamental mode (n=1) eigenfunction derivative has 1 sign change."""
        m = uniform_medium
        _, _, dy = integrate_wave_equation(
            m["analytical_freqs"][0],
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
            n_points=500,
        )
        assert count_mode_number(dy) == 1

    def test_second_harmonic(self, uniform_medium):
        """Second harmonic (n=2) eigenfunction derivative has 2 sign changes."""
        m = uniform_medium
        _, _, dy = integrate_wave_equation(
            m["analytical_freqs"][1],
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
            n_points=500,
        )
        assert count_mode_number(dy) == 2

    def test_third_harmonic(self, uniform_medium):
        """Third harmonic has 3 sign changes in dy."""
        m = uniform_medium
        _, _, dy = integrate_wave_equation(
            m["analytical_freqs"][2],
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
            n_points=500,
        )
        assert count_mode_number(dy) == 3


# ============================================================================
# Eigensolver tests
# ============================================================================


class TestFindEigenfrequencies:
    """Tests for the full eigenfrequency solver."""

    def test_finds_correct_number_of_modes(self, uniform_medium):
        """Should find the requested number of modes."""
        m = uniform_medium
        modes = find_eigenfrequencies(
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
            freq_range=(1e-5, m["analytical_freqs"][5] * 1.5),
            n_modes=6,
            resolution=300,
        )
        assert len(modes) == 6

    def test_matches_analytical_uniform(self, uniform_medium):
        """Found frequencies should match ω_n = nπvA/L within 0.1%."""
        m = uniform_medium
        modes = find_eigenfrequencies(
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
            freq_range=(1e-5, m["analytical_freqs"][5] * 1.5),
            n_modes=6,
            resolution=300,
        )
        for i, mode in enumerate(modes):
            expected = m["analytical_freqs"][i]
            assert_allclose(
                mode.angular_frequency,
                expected,
                rtol=1e-3,
                err_msg=f"Mode {i + 1}: got {mode.angular_frequency:.6e}, "
                f"expected {expected:.6e}",
            )

    def test_modes_sorted_by_frequency(self, uniform_medium):
        """Modes should be returned in ascending frequency order."""
        m = uniform_medium
        modes = find_eigenfrequencies(
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
            freq_range=(1e-5, m["analytical_freqs"][3] * 1.5),
            n_modes=4,
        )
        freqs = [mode.angular_frequency for mode in modes]
        assert freqs == sorted(freqs)

    def test_with_eigenfunctions(self, uniform_medium):
        """When include_eigenfunctions=True, should populate eigenfunction data."""
        m = uniform_medium
        modes = find_eigenfrequencies(
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
            freq_range=(1e-5, m["analytical_freqs"][2] * 1.5),
            n_modes=3,
            include_eigenfunctions=True,
        )
        for mode in modes:
            assert mode.eigenfunction is not None
            assert mode.eigenfunction_derivative is not None
            assert mode.arc_length is not None
            assert len(mode.eigenfunction) == 500

    def test_nonuniform_medium_finds_modes(self, nonuniform_medium):
        """Should find modes in a non-uniform medium (no analytical check)."""
        m = nonuniform_medium
        modes = find_eigenfrequencies(
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
            freq_range=(1e-5, 5e-2),
            n_modes=3,
            resolution=300,
        )
        assert len(modes) >= 3
        # Frequencies should be positive and ordered
        for mode in modes:
            assert mode.angular_frequency > 0


# ============================================================================
# Result dataclass tests
# ============================================================================


class TestEigenResult:
    """Tests for result dataclasses."""

    def test_eigenmode_conversions(self):
        """frequency_mhz and period_minutes should be consistent."""
        mode = EigenMode(angular_frequency=0.001, mode_number=1)
        # f = ω/(2π) in Hz → mHz
        expected_mhz = 0.001 / (2 * math.pi) * 1e3
        assert_allclose(mode.frequency_mhz, expected_mhz, rtol=1e-10)
        # T = 2π/ω in seconds → minutes
        expected_min = 2 * math.pi / 0.001 / 60.0
        assert_allclose(mode.period_minutes, expected_min, rtol=1e-10)

    def test_eigenresult_properties(self):
        """EigenResult should aggregate mode properties."""
        modes = [
            EigenMode(angular_frequency=0.001, mode_number=1),
            EigenMode(angular_frequency=0.002, mode_number=2),
            EigenMode(angular_frequency=0.003, mode_number=3),
        ]
        result = EigenResult(
            modes=modes,
            l_shell=15.0,
            conjugate_latitude=70.0,
            component="toroidal",
        )
        assert result.n_modes == 3
        assert_allclose(result.angular_frequencies, [0.001, 0.002, 0.003])
        assert result.mode_numbers == [1, 2, 3]
        assert len(result.periods_minutes) == 3


# ============================================================================
# Cross-validation tests for optimization correctness
# ============================================================================


class TestRK4MatchesScipy:
    """Validate that the numba RK4 path agrees with scipy solve_ivp."""

    def test_boundary_error_rk4_vs_scipy(self, uniform_medium):
        """Fast numba path and scipy fallback should agree on boundary error."""
        m = uniform_medium
        n_samples = 5000
        s = np.linspace(m["s_span"][0], m["s_span"][1], n_samples)
        va_samples = np.ascontiguousarray(m["va_spline"](s))
        dlnh_samples = np.ascontiguousarray(m["dlnh_spline"](s))

        omega = m["analytical_freqs"][0] * 1.03  # slightly off-resonance

        err_scipy = boundary_error(
            omega,
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
        )
        err_rk4 = boundary_error(
            omega,
            m["s_span"],
            m["va_spline"],
            m["dlnh_spline"],
            va_samples=va_samples,
            dlnh_samples=dlnh_samples,
        )
        assert_allclose(err_rk4, err_scipy, rtol=0.05)


class TestVectorizedScan:
    """Validate that _scan_boundary_errors matches individual calls."""

    def test_scan_matches_loop(self, uniform_medium):
        """Vectorized scan should produce identical results to a Python loop."""
        m = uniform_medium
        n_samples = 5000
        s = np.linspace(m["s_span"][0], m["s_span"][1], n_samples)
        va_samples = np.ascontiguousarray(m["va_spline"](s))
        dlnh_samples = np.ascontiguousarray(m["dlnh_spline"](s))

        omegas = np.linspace(1e-5, m["analytical_freqs"][3] * 1.5, 50)

        # Vectorized
        errors_vec = _scan_boundary_errors(
            omegas,
            m["s_span"][0],
            m["s_span"][1],
            va_samples,
            dlnh_samples,
            0.0,
            1.0,
        )

        # Loop (calling the same underlying RK4)
        errors_loop = np.array(
            [
                boundary_error(
                    w,
                    m["s_span"],
                    m["va_spline"],
                    m["dlnh_spline"],
                    va_samples=va_samples,
                    dlnh_samples=dlnh_samples,
                )
                for w in omegas
            ]
        )

        assert_allclose(errors_vec, errors_loop, rtol=1e-12)
