"""Parity tests for the Cummings (1969) cos-θ Sturm-Liouville eigensolver.

The Cummings solver reparametrises the FLR wave equation from arc length
``s`` to ``μ = z/r`` (magnetic colatitude cosine). The eigenvalue problem
is invariant under coordinate change, so the eigenfrequencies *must* match
the arc-length matrix solver to within discretisation error. These tests
verify that invariance on real KMAG profiles and on the uniform-v_A
analytic baseline.

The Phase-3b investigation in May 2026 originally hypothesised that the
Cummings coordinate would *remove* the sub-WKB soft-wall mode 1 that the
arc-length matrix solver finds at KMAG L=8 noon. That hypothesis was
wrong — the test results below directly confirm it. The note in
``docs/notes/wavesolver_reference.md`` documents the resolution.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.fieldline.kmag_model import SaturnField
from qp.wavesolver.density import UniformDensity
from qp.wavesolver.solver import WavesolverConfig, solve_eigenfrequencies


@pytest.mark.slow
@pytest.mark.parametrize("l_shell", [8.0, 10.0, 15.0])
def test_cummings_matches_matrix_on_kmag(l_shell: float) -> None:
    """Cummings μ-coordinate eigenfrequencies match the arc-length matrix solver.

    The eigenvalue problem :math:`-(p y')' = \\omega^2 w y` with Dirichlet
    BCs is invariant under reparametrisation. The two solvers must agree
    to within finite-difference discretisation error (≤ 0.5 % observed
    on KMAG L=8/10/15 noon).
    """
    base = dict(
        l_shell=l_shell,
        n_modes=6,
        field=SaturnField(),
        local_time_hours=12.0,
    )
    matrix_freqs = solve_eigenfrequencies(
        WavesolverConfig(method="matrix", **base)  # type: ignore[arg-type]
    ).angular_frequencies
    cummings_freqs = solve_eigenfrequencies(
        WavesolverConfig(method="cummings", **base)  # type: ignore[arg-type]
    ).angular_frequencies
    assert len(matrix_freqs) == len(cummings_freqs) == 6
    assert_allclose(cummings_freqs, matrix_freqs, rtol=5e-3)


def test_cummings_uniform_density_dipole_agrees_with_matrix() -> None:
    """On a pure dipole + uniform density, Cummings and matrix solvers agree."""
    base = dict(
        l_shell=6.0,
        n_modes=4,
        density_model=UniformDensity(n0=1e7),
    )
    matrix_freqs = solve_eigenfrequencies(
        WavesolverConfig(method="matrix", **base)  # type: ignore[arg-type]
    ).angular_frequencies
    cummings_freqs = solve_eigenfrequencies(
        WavesolverConfig(method="cummings", **base)  # type: ignore[arg-type]
    ).angular_frequencies
    assert_allclose(cummings_freqs, matrix_freqs, rtol=1e-2)


def test_cummings_returns_ascending_modes() -> None:
    """Cummings backend returns eigenfrequencies in ascending order."""
    result = solve_eigenfrequencies(
        WavesolverConfig(
            l_shell=8.0,
            n_modes=4,
            density_model=UniformDensity(n0=1e7),
            method="cummings",
        )
    )
    omegas = result.angular_frequencies
    assert len(omegas) == 4
    assert np.all(np.diff(omegas) > 0)


def test_cummings_eigenfunctions_satisfy_bcs() -> None:
    """Eigenfunctions vanish at the trace endpoints."""
    result = solve_eigenfrequencies(
        WavesolverConfig(
            l_shell=8.0,
            n_modes=3,
            density_model=UniformDensity(n0=1e7),
            method="cummings",
            include_eigenfunctions=True,
        )
    )
    for mode in result.modes:
        y = mode.eigenfunction
        assert y is not None
        assert y[0] == 0.0
        assert y[-1] == 0.0
