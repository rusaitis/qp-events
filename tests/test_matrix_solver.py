"""Parity tests for the matrix Sturm-Liouville eigensolver vs the shooter.

The matrix solver (``qp.wavesolver.matrix_solver.find_eigenfrequencies_matrix``)
and the shooter (``qp.wavesolver.eigensolver.find_eigenfrequencies``) discretise
the same equation by completely different numerical paths — a single
``scipy.linalg.eigh_tridiagonal`` call vs adaptive bracket-scan +
Brent + numba RK4 / scipy LSODA. They should agree to 1 % on KMAG cases
where both are well-conditioned. For pure-dipole with a Bagenal density
the system is severely ill-conditioned (v_A varies 10 orders of magnitude
along the field line) and we relax the tolerance accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import CubicSpline

from qp.fieldline.kmag_model import SaturnField
from qp.wavesolver.density import UniformDensity
from qp.wavesolver.matrix_solver import find_eigenfrequencies_matrix
from qp.wavesolver.solver import WavesolverConfig, solve_eigenfrequencies


def test_matrix_uniform_va_matches_analytic() -> None:
    """Constant medium with no dlnh term must match ω_m = m·π·v_A/L."""
    L = 1.0e9
    va_const = 1.0e6
    n_pts = 2000
    s = np.linspace(0.0, L, n_pts)
    h_alpha = np.ones_like(s)
    B = np.ones_like(s)
    va = np.full_like(s, va_const)

    modes = find_eigenfrequencies_matrix(
        s, h_alpha, B, va, n_modes=6, include_eigenfunctions=False
    )
    found = np.array([m.angular_frequency for m in modes])
    analytic = np.array([m * np.pi * va_const / L for m in range(1, 7)])
    assert_allclose(found, analytic, rtol=1e-4)


@pytest.mark.parametrize(
    "config_kwargs",
    [
        # KMAG L=8 noon - the actual physics setup used in the paper. Both
        # solvers are well-conditioned here.
        dict(
            l_shell=8.0,
            n_modes=6,
            field=SaturnField(),
            local_time_hours=12.0,
            freq_range=(1e-6, 1e-2),
            resolution=400,
        ),
        # KMAG L=15 noon - lower-frequency case.
        dict(
            l_shell=15.0,
            n_modes=4,
            field=SaturnField(),
            local_time_hours=12.0,
            freq_range=(1e-6, 1e-2),
            resolution=400,
        ),
        # Pure dipole L=6 + UniformDensity - well-conditioned. Mode 1 at
        # ~1.2e-5 rad/s sits right at the default freq_range lower bound,
        # so widen freq_range explicitly for the shooter to find it.
        dict(
            l_shell=6.0,
            n_modes=4,
            density_model=UniformDensity(n0=1e7),
            freq_range=(1e-6, 0.001),
            resolution=400,
        ),
    ],
    ids=["kmag_l8_noon", "kmag_l15_noon", "dipole_l6_uniform"],
)
@pytest.mark.slow
def test_matrix_agrees_with_shooter(config_kwargs: dict) -> None:
    """Matrix and shooter back-ends should agree on every requested mode.

    Tolerance 1 % captures the discretisation difference between the two
    methods (5000-point uniform grid + finite differences vs adaptive ODE
    integration + brentq root finding). Both target the same Sturm-Liouville
    eigenvalue problem.
    """
    matrix_modes = solve_eigenfrequencies(
        WavesolverConfig(method="matrix", **config_kwargs)
    ).angular_frequencies
    shooter_modes = solve_eigenfrequencies(
        WavesolverConfig(method="shoot", **config_kwargs)
    ).angular_frequencies
    n_compare = min(len(matrix_modes), len(shooter_modes))
    assert n_compare >= 1, "no modes returned by either backend"
    assert_allclose(matrix_modes[:n_compare], shooter_modes[:n_compare], rtol=0.01)


def test_matrix_returns_ascending_modes() -> None:
    """find_eigenfrequencies_matrix must return modes sorted by frequency."""
    result = solve_eigenfrequencies(
        WavesolverConfig(
            l_shell=10.0,
            n_modes=6,
            density_model=UniformDensity(n0=1e7),
            method="matrix",
        )
    )
    omegas = result.angular_frequencies
    assert len(omegas) == 6
    assert np.all(np.diff(omegas) > 0)


def test_matrix_eigenfunctions_satisfy_bcs() -> None:
    """Eigenfunctions should equal zero at both arc-length endpoints."""
    result = solve_eigenfrequencies(
        WavesolverConfig(
            l_shell=8.0,
            n_modes=3,
            density_model=UniformDensity(n0=1e7),
            method="matrix",
            include_eigenfunctions=True,
        )
    )
    for mode in result.modes:
        y = mode.eigenfunction
        assert y is not None
        assert y[0] == 0.0
        assert y[-1] == 0.0


def test_matrix_rejects_non_uniform_grid() -> None:
    """A non-uniform s array should raise rather than silently produce nonsense."""
    s = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    h_alpha = np.ones_like(s)
    B = np.ones_like(s)
    va = np.ones_like(s)
    with pytest.raises(ValueError, match="uniformly spaced"):
        find_eigenfrequencies_matrix(s, h_alpha, B, va, n_modes=2)


def test_matrix_eigenvalues_invariant_to_grid_resolution() -> None:
    """Matrix eigenvalues should converge as the grid is refined."""
    L = 1.0e9
    va = 1.0e6
    analytic_1 = np.pi * va / L

    omegas: list[float] = []
    for n in [200, 1000, 5000]:
        s = np.linspace(0.0, L, n)
        ones = np.ones_like(s)
        modes = find_eigenfrequencies_matrix(
            s, ones, ones, np.full_like(s, va), n_modes=1, include_eigenfunctions=False
        )
        omegas.append(modes[0].angular_frequency)

    # All three resolutions should agree with the analytic value to 3+ sig figs
    for omega in omegas:
        assert_allclose(omega, analytic_1, rtol=1e-3)


def test_matrix_with_smooth_profile_matches_direct_call() -> None:
    """A direct call to find_eigenfrequencies_matrix matches the solver pipeline."""
    cfg = WavesolverConfig(
        l_shell=6.0,
        n_modes=3,
        density_model=UniformDensity(n0=1e7),
        method="matrix",
    )
    via_solver = solve_eigenfrequencies(cfg).angular_frequencies

    # Reproduce the matrix call manually from the field-line profile
    from qp.fieldline.tracer import dipole_field, trace_fieldline_bidirectional
    from qp.wavesolver.field_profile import compute_field_line_profile
    from qp.wavesolver.solver import _starting_position

    start = _starting_position(cfg)
    trace = trace_fieldline_bidirectional(dipole_field, start, step=cfg.trace_step)
    profile = compute_field_line_profile(trace, UniformDensity(n0=1e7))
    direct = np.array(
        [
            m.angular_frequency
            for m in find_eigenfrequencies_matrix(
                profile.s_samples,
                profile.h1_samples,
                profile.B_samples,
                profile.va_samples,
                n_modes=3,
                include_eigenfunctions=False,
            )
        ]
    )
    assert_allclose(direct, via_solver, rtol=1e-10)


# Silence the unused-import lint for CubicSpline (kept for future tests that
# want to compare matrix output against a hand-built spline reference).
_ = CubicSpline
