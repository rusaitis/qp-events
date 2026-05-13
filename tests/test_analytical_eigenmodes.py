r"""Non-uniform analytical benchmarks for the wavesolver.

Until May 2026 the only closed-form check was the uniform-:math:`v_A`
baseline (``tests/test_matrix_solver.py::test_matrix_uniform_va_matches_analytic``).
This module adds non-trivial profiles where the eigenvalues are known
in closed form, exercising both the matrix and shooter backends on
non-uniform :math:`v_A`.

The Cummings backend takes a physical trace (positions array) and is
exercised in :mod:`tests.test_cummings_solver` and the three-backend
parity tests; synthetic-:math:`v_A` cases here cover matrix and shooter
only.

Profiles
--------

**Linear** :math:`v_A(s) = v_0 (1 + \alpha s/L)` on :math:`s \in [0, L]`.
The wave equation :math:`y'' + (\omega/v_A)^2 y = 0` becomes a
Cauchy–Euler equation under the substitution :math:`u = v_A(s)`. With
homogeneous Dirichlet BCs the eigenfrequencies are

.. math::

   \omega_n = \frac{v_0\,\alpha}{L} \sqrt{
       \left(\frac{n\pi}{\ln(1+\alpha)}\right)^{\!2} + \tfrac14},
   \quad n = 1, 2, 3, \dots

In the limit :math:`\alpha \to 0` this reduces to :math:`\omega_n = n\pi v_0/L`,
the uniform baseline.

**Two-section string** :math:`v_A = v_1` on :math:`[0, L/2]`,
:math:`v_A = v_2` on :math:`[L/2, L]`. The characteristic equation is

.. math::

   k_1 \sin\!\bigl(k_2 L/2\bigr) \cos\!\bigl(k_1 L/2\bigr)
   + k_2 \sin\!\bigl(k_1 L/2\bigr) \cos\!\bigl(k_2 L/2\bigr) = 0,

with :math:`k_i = \omega/v_i`. Solved numerically by sign-change search
+ ``brentq`` to high precision.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from qp.wavesolver.eigensolver import find_eigenfrequencies
from qp.wavesolver.matrix_solver import find_eigenfrequencies_matrix


# ---------------------------------------------------------------------------
# Closed-form analytic references


def _linear_va_modes_analytic(
    v_0: float, alpha: float, L: float, n_modes: int
) -> np.ndarray:
    r"""Eigenvalues of :math:`y'' + (\omega/v_A)^2 y = 0` with linear :math:`v_A`."""
    factor = v_0 * alpha / L
    return np.array(
        [
            factor * np.sqrt((n * np.pi / np.log(1 + alpha)) ** 2 + 0.25)
            for n in range(1, n_modes + 1)
        ]
    )


def _two_section_string_modes(
    v_1: float, v_2: float, L: float, n_modes: int
) -> np.ndarray:
    r"""Eigenvalues of the two-section vibrating string."""

    def char(omega: float) -> float:
        k1 = omega / v_1
        k2 = omega / v_2
        return (
            k1 * np.sin(k2 * L / 2) * np.cos(k1 * L / 2)
            + k2 * np.sin(k1 * L / 2) * np.cos(k2 * L / 2)
        )

    omega_max = 4 * n_modes * np.pi * max(v_1, v_2) / L
    grid = np.linspace(1e-3 * omega_max / n_modes, omega_max, 20000)
    f = np.array([char(o) for o in grid])
    sign_change = np.where(f[:-1] * f[1:] < 0)[0]
    roots = []
    for i in sign_change:
        try:
            r = brentq(char, grid[i], grid[i + 1])
            roots.append(r)
        except ValueError:
            continue
        if len(roots) >= n_modes:
            break
    return np.array(roots)


# ---------------------------------------------------------------------------
# Test fixtures


def _build_uniform_profile(L: float, n_pts: int, va_fn) -> dict:
    s = np.linspace(0.0, L, n_pts)
    va = np.asarray(va_fn(s), dtype=float)
    ones = np.ones_like(s)
    return {
        "s": s,
        "h_alpha": ones,
        "B": ones,
        "va": va,
        "va_spline": CubicSpline(s, va),
        "dlnh_spline": CubicSpline(s, np.zeros_like(s)),
    }


# ---------------------------------------------------------------------------
# Linear v_A profile


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_matrix_linear_va_matches_analytic(alpha: float) -> None:
    """Matrix solver reproduces the linear-v_A Cauchy–Euler ladder."""
    L = 1.0
    v_0 = 1.0
    n_modes = 5
    prof = _build_uniform_profile(L, 5000, lambda s: v_0 * (1 + alpha * s / L))
    modes = find_eigenfrequencies_matrix(
        prof["s"], prof["h_alpha"], prof["B"], prof["va"],
        n_modes=n_modes, include_eigenfunctions=False,
    )
    found = np.array([m.angular_frequency for m in modes])
    analytic = _linear_va_modes_analytic(v_0, alpha, L, n_modes)
    assert_allclose(found, analytic, rtol=1e-3)


@pytest.mark.parametrize("alpha", [0.5, 1.0])
@pytest.mark.slow
def test_shooter_linear_va_matches_analytic(alpha: float) -> None:
    """Shooter reproduces the linear-v_A Cauchy–Euler ladder."""
    L = 1.0
    v_0 = 1.0
    n_modes = 4
    prof = _build_uniform_profile(L, 5000, lambda s: v_0 * (1 + alpha * s / L))
    analytic = _linear_va_modes_analytic(v_0, alpha, L, n_modes)
    omega_low = analytic[0] * 0.1
    omega_high = analytic[-1] * 2.0
    modes = find_eigenfrequencies(
        s_span=(0.0, L),
        va_spline=prof["va_spline"],
        dlnh_spline=prof["dlnh_spline"],
        freq_range=(omega_low, omega_high),
        n_modes=n_modes,
        resolution=400,
        tolerance=1e-9,
        include_eigenfunctions=False,
        va_samples=prof["va"],
        dlnh_samples=np.zeros_like(prof["s"]),
    )
    found = np.array([m.angular_frequency for m in modes[:n_modes]])
    assert_allclose(found, analytic, rtol=5e-3)


# ---------------------------------------------------------------------------
# Two-section string


@pytest.mark.parametrize("v_ratio", [1.5, 2.0, 3.0])
def test_matrix_two_section_string_matches_analytic(v_ratio: float) -> None:
    """Matrix solver reproduces the two-section vibrating-string ladder.

    Tolerance is 3 % rather than 1 % because the step discontinuity at
    s = L/2 is smeared over one grid cell by the matrix solver's
    cell-face averaging (``p_half = (p[i]+p[i+1])/2``).
    """
    L = 1.0
    v_1 = 1.0
    v_2 = v_1 * v_ratio
    n_modes = 4
    n_pts = 8000

    def va(s):
        return np.where(s < L / 2, v_1, v_2)

    prof = _build_uniform_profile(L, n_pts, va)
    modes = find_eigenfrequencies_matrix(
        prof["s"], prof["h_alpha"], prof["B"], prof["va"],
        n_modes=n_modes, include_eigenfunctions=False,
    )
    found = np.array([m.angular_frequency for m in modes])
    analytic = _two_section_string_modes(v_1, v_2, L, n_modes)
    assert_allclose(found, analytic, rtol=3e-2)


def test_two_section_string_collapses_to_uniform() -> None:
    """At v_1 = v_2, the two-section formula must give the uniform ladder."""
    v = 1.0
    L = 1.0
    modes = _two_section_string_modes(v, v, L, 4)
    expected = np.array([n * np.pi * v / L for n in range(1, 5)])
    assert_allclose(modes, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Cross-backend agreement on the linear-v_A profile (matrix vs shooter)


@pytest.mark.slow
def test_matrix_and_shooter_agree_on_linear_va() -> None:
    """Matrix and shooter must agree on a non-uniform analytic profile."""
    L = 1.0
    v_0 = 1.0
    alpha = 1.0
    n_modes = 4
    prof = _build_uniform_profile(L, 5000, lambda s: v_0 * (1 + alpha * s / L))

    matrix_modes = find_eigenfrequencies_matrix(
        prof["s"], prof["h_alpha"], prof["B"], prof["va"],
        n_modes=n_modes, include_eigenfunctions=False,
    )
    shoot_modes = find_eigenfrequencies(
        s_span=(0.0, L),
        va_spline=prof["va_spline"],
        dlnh_spline=prof["dlnh_spline"],
        freq_range=(0.1, 40.0),
        n_modes=n_modes,
        resolution=400,
        tolerance=1e-9,
        include_eigenfunctions=False,
        va_samples=prof["va"],
        dlnh_samples=np.zeros_like(prof["s"]),
    )
    m_freqs = np.array([m.angular_frequency for m in matrix_modes])
    s_freqs = np.array([m.angular_frequency for m in shoot_modes[:n_modes]])
    assert_allclose(s_freqs, m_freqs, rtol=5e-3)
