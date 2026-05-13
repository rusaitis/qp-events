"""Tests for Stokes parameters and degree of polarization."""

from __future__ import annotations

import numpy as np
import pytest

from qp.signal.polarization import (
    degree_of_polarization,
    stokes_parameters,
)


def _analytic_tone(n: int, omega: float = 0.1, phase: float = 0.0) -> np.ndarray:
    """A unit-amplitude complex sinusoid (analytic signal of cos(omega t + phase))."""
    t = np.arange(n)
    return np.exp(1j * (omega * t + phase))


def test_pure_linear_along_perp1_gives_Q_equals_I():
    """Only b_perp1 oscillates: Q = I, U = V = 0, d = 1."""
    z1 = _analytic_tone(2000)
    z2 = np.zeros_like(z1)
    i_, q, u, v = stokes_parameters(z1, z2)
    np.testing.assert_allclose(q, i_, rtol=1e-10)
    np.testing.assert_allclose(u, 0.0, atol=1e-10)
    np.testing.assert_allclose(v, 0.0, atol=1e-10)
    assert degree_of_polarization(z1, z2) == pytest.approx(1.0, abs=1e-10)


def test_pure_linear_at_45_degrees_gives_U_equals_I():
    """Equal-amplitude in-phase components: Q = V = 0, U = I, d = 1."""
    z = _analytic_tone(2000)
    i_, q, u, v = stokes_parameters(z, z)
    np.testing.assert_allclose(q, 0.0, atol=1e-10)
    np.testing.assert_allclose(u, i_, rtol=1e-10)
    np.testing.assert_allclose(v, 0.0, atol=1e-10)
    assert degree_of_polarization(z, z) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("lag", [np.pi / 2, -np.pi / 2])
def test_pure_circular_polarization_gives_V_equals_I(lag: float):
    """Equal-amplitude with quadrature lag: Q = U = 0, |V| = I, d = 1."""
    z1 = _analytic_tone(2000)
    z2 = _analytic_tone(2000, phase=lag)
    i_, q, u, v = stokes_parameters(z1, z2)
    np.testing.assert_allclose(q, 0.0, atol=1e-10)
    np.testing.assert_allclose(u, 0.0, atol=1e-10)
    np.testing.assert_allclose(abs(v), i_, rtol=1e-10)
    assert degree_of_polarization(z1, z2) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("ellipticity", [0.1, 0.3, 0.7])
def test_elliptical_polarization_is_fully_polarized(ellipticity: float):
    """Pure elliptical wave (no noise) has d = 1 for any ellipticity."""
    z1 = _analytic_tone(2000)
    z2 = ellipticity * _analytic_tone(2000, phase=np.pi / 2)
    assert degree_of_polarization(z1, z2) == pytest.approx(1.0, abs=1e-10)


def test_uncorrelated_noise_has_low_polarization():
    """Independent CN(0,1) samples in both components: d ~ 1/sqrt(N)."""
    rng = np.random.default_rng(42)
    n = 20000
    z1 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    z2 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    d = degree_of_polarization(z1, z2)
    assert d < 0.05, f"expected d << 1 for independent noise, got {d}"


def test_signal_plus_noise_decays_smoothly_with_snr():
    """Adding noise to a coherent wave reduces d monotonically."""
    rng = np.random.default_rng(0)
    n = 10000
    z_sig = _analytic_tone(n)
    ds = []
    for snr in [10.0, 3.0, 1.0, 0.3]:
        noise1 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        noise2 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        z1 = snr * z_sig + noise1
        z2 = snr * 1j * z_sig + noise2  # circular
        ds.append(degree_of_polarization(z1, z2))
    # Pure-signal limit ~ 1, pure-noise limit ~ 0; monotone in between.
    for a, b in zip(ds, ds[1:], strict=False):
        assert a >= b - 0.05, f"d should be non-increasing with noise, got {ds}"
    assert ds[0] > 0.9
    assert ds[-1] < 0.5


def test_zero_input_returns_zero():
    z = np.zeros(100, dtype=complex)
    assert degree_of_polarization(z, z) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        stokes_parameters(np.zeros(10, dtype=complex), np.zeros(20, dtype=complex))


# ---------------------------------------------------------------------
# Minimum variance analysis
# ---------------------------------------------------------------------


from qp.signal.polarization import mva_intermediate_minimum_ratio


def test_mva_planar_alfven_perturbation_has_high_ratio():
    """delta B in the (b_perp1, b_perp2) plane only: lambda3 -> 0, ratio -> inf."""
    n = 1000
    t = np.linspace(0, 50 * np.pi, n)
    field = np.column_stack(
        [
            np.zeros(n),  # b_par: no perturbation
            np.cos(t),  # b_perp1
            np.sin(t),  # b_perp2 (circular polarization)
        ]
    )
    ratio = mva_intermediate_minimum_ratio(field)
    assert ratio > 1e6, f"planar perturbation should give large ratio, got {ratio}"


def test_mva_isotropic_noise_has_ratio_near_one():
    """3D Gaussian noise: all eigenvalues ~ 1, ratio ~ 1."""
    rng = np.random.default_rng(0)
    field = rng.standard_normal((20000, 3))
    ratio = mva_intermediate_minimum_ratio(field)
    assert 0.7 < ratio < 1.3, f"isotropic noise should give ratio ~ 1, got {ratio}"


def test_mva_step_in_all_components_has_low_ratio():
    """A simultaneous step in all three components has poorly-defined wave normal."""
    n = 1000
    field = np.zeros((n, 3))
    field[n // 2 :, :] = 1.0  # step in all components
    rng = np.random.default_rng(42)
    field += 0.1 * rng.standard_normal(field.shape)
    ratio = mva_intermediate_minimum_ratio(field)
    assert ratio < 5.0, f"isotropic step should fail lambda2/lambda3 >= 5, got {ratio}"


def test_mva_short_input_returns_zero():
    assert mva_intermediate_minimum_ratio(np.zeros((2, 3))) == 0.0


def test_mva_wrong_shape_raises():
    with pytest.raises(ValueError):
        mva_intermediate_minimum_ratio(np.zeros((100, 2)))


# ---------------------------------------------------------------------
# Minimum-eigenvalue fraction (rank-2 planarity, all pol modes)
# ---------------------------------------------------------------------


from qp.signal.polarization import mva_minimum_eigenvalue_fraction


def test_min_eig_linear_polarization_is_low():
    """Pure linear pol (rank 1): lambda_2 ~ lambda_3 ~ noise, but lambda_3/lambda_1 -> 0."""
    n = 1000
    rng = np.random.default_rng(0)
    t = np.linspace(0, 50 * np.pi, n)
    field = np.column_stack(
        [
            0.01 * rng.standard_normal(n),
            np.cos(t),  # only b_perp1 has the wave
            0.01 * rng.standard_normal(n),
        ]
    )
    frac = mva_minimum_eigenvalue_fraction(field)
    assert frac < 0.01, f"linear pol should give ~0, got {frac}"


def test_min_eig_circular_polarization_is_low():
    """Pure circular pol (rank 2 planar): lambda_3 -> 0, lambda_3/lambda_1 -> 0."""
    n = 1000
    t = np.linspace(0, 50 * np.pi, n)
    field = np.column_stack([np.zeros(n), np.cos(t), np.sin(t)])
    frac = mva_minimum_eigenvalue_fraction(field)
    assert frac < 1e-10, f"circular pol should give ~0, got {frac}"


def test_min_eig_independent_3axis_perturbation_is_high():
    """True rank-3 perturbation (independent variation on each axis): ratio close to 1.

    A coordinated step along (1,1,1) is rank-1, not rank-3 — its
    minimum-eigenvalue fraction is therefore small. The genuine
    rank-3 case is independent variation (e.g. 3-D Gaussian noise),
    which we test in the dedicated isotropic-noise test below; here
    we just confirm that *anisotropic* but rank-3 input still gives
    a non-trivial ratio.
    """
    rng = np.random.default_rng(42)
    n = 5000
    field = np.column_stack(
        [
            rng.standard_normal(n),  # var ~ 1
            0.7 * rng.standard_normal(n),  # var ~ 0.49
            0.5 * rng.standard_normal(n),  # var ~ 0.25
        ]
    )
    frac = mva_minimum_eigenvalue_fraction(field)
    assert 0.15 < frac < 0.4, f"anisotropic rank-3 should give ~0.25, got {frac}"


def test_min_eig_isotropic_noise_returns_unity():
    """3D Gaussian noise: lambda_1 ~ lambda_2 ~ lambda_3 ~ 1, ratio ~ 1."""
    rng = np.random.default_rng(0)
    field = rng.standard_normal((20000, 3))
    frac = mva_minimum_eigenvalue_fraction(field)
    assert 0.7 < frac < 1.3


def test_min_eig_short_input_returns_zero():
    assert mva_minimum_eigenvalue_fraction(np.zeros((2, 3))) == 0.0


def test_min_eig_wrong_shape_raises():
    with pytest.raises(ValueError):
        mva_minimum_eigenvalue_fraction(np.zeros((100, 2)))


# ---------------------------------------------------------------------
# Major-axis parallel fraction (transversality)
# ---------------------------------------------------------------------


from qp.signal.polarization import mva_major_axis_parallel_fraction


def test_major_axis_purely_transverse_alfven():
    """Pure transverse perturbation (b_par=0): major axis is perpendicular to B0."""
    n = 1000
    t = np.linspace(0, 50 * np.pi, n)
    field = np.column_stack([np.zeros(n), np.cos(t), np.sin(t)])
    frac = mva_major_axis_parallel_fraction(field)
    assert frac < 1e-10, f"transverse wave should give ~0, got {frac}"


def test_major_axis_purely_compressional():
    """Pure b_par perturbation: major axis is along B0."""
    n = 1000
    t = np.linspace(0, 50 * np.pi, n)
    rng = np.random.default_rng(0)
    field = np.column_stack(
        [
            np.cos(t),
            0.001 * rng.standard_normal(n),  # near-zero noise on b_perp1
            0.001 * rng.standard_normal(n),
        ]
    )
    frac = mva_major_axis_parallel_fraction(field)
    assert frac > 0.99, f"compressional wave should give ~1, got {frac}"


def test_major_axis_compressional_with_leakage():
    """Decoy compressional with 5% par_leakage: major axis still > 0.5 along B0."""
    n = 1000
    t = np.linspace(0, 50 * np.pi, n)
    par_leakage = 0.05
    field = np.column_stack(
        [
            np.cos(t),
            par_leakage * np.cos(t),
            par_leakage * np.sin(t),
        ]
    )
    frac = mva_major_axis_parallel_fraction(field)
    # var(par)/(var(par)+var(perp1)) ~ 1/(1+0.05^2) ~ 0.9975 along (par,perp1) axis
    assert frac > 0.9, f"compressional with leakage should give >0.9, got {frac}"


def test_major_axis_isotropic_returns_finite():
    """Isotropic noise: any direction is equally valid; result is in [0,1]."""
    rng = np.random.default_rng(1)
    field = rng.standard_normal((10000, 3))
    frac = mva_major_axis_parallel_fraction(field)
    assert 0.0 <= frac <= 1.0


def test_major_axis_choice_of_par_axis():
    """par_axis=2 selects column 2 as the parallel direction."""
    n = 1000
    t = np.linspace(0, 50 * np.pi, n)
    # Put compressional signal on column 2
    field = np.column_stack(
        [
            0.05 * np.cos(t),
            0.05 * np.sin(t),
            np.cos(t),
        ]
    )
    frac = mva_major_axis_parallel_fraction(field, par_axis=2)
    assert frac > 0.9, (
        f"compressional on column 2 with par_axis=2 should give >0.9, got {frac}"
    )


def test_major_axis_short_input_returns_zero():
    assert mva_major_axis_parallel_fraction(np.zeros((2, 3))) == 0.0


def test_major_axis_invalid_par_axis_raises():
    with pytest.raises(ValueError):
        mva_major_axis_parallel_fraction(np.zeros((100, 3)), par_axis=3)


def test_major_axis_wrong_shape_raises():
    with pytest.raises(ValueError):
        mva_major_axis_parallel_fraction(np.zeros((100, 2)))
