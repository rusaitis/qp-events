"""Three-backend parity tests for the wavesolver.

Pre-existing tests cover (matrix vs shooter) and (matrix vs Cummings),
but **not** (shooter vs Cummings) — a gap closed here. These tests also
extend agreement to the full ladder (m=1..6) on real-physics setups,
where the previous coverage was limited to the lowest few modes.

Tolerance: 0.5 % for the cross-backend pairs (limited by discretisation
differences between adaptive ODE integration in the shooter vs.
finite-difference matrix on the uniform grid).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.fieldline.kmag_model import SaturnField
from qp.wavesolver.density import UniformDensity
from qp.wavesolver.solver import WavesolverConfig, solve_eigenfrequencies


def _solve(method: str, **cfg_kwargs) -> np.ndarray:
    return solve_eigenfrequencies(
        WavesolverConfig(method=method, **cfg_kwargs)
    ).angular_frequencies


@pytest.mark.slow
def test_three_backends_agree_on_kmag_l8_full_ladder() -> None:
    """All three backends agree to ≤ 0.5 % on modes 1..6 at KMAG L=8 noon."""
    base = dict(
        l_shell=8.0,
        n_modes=6,
        field=SaturnField(),
        local_time_hours=12.0,
        freq_range=(1e-6, 1e-2),
        resolution=400,
    )
    matrix = _solve("matrix", **base)
    cummings = _solve("cummings", **base)
    shoot = _solve("shoot", **base)

    n = min(len(matrix), len(cummings), len(shoot))
    assert n == 6, (
        f"expected 6 modes from each backend, got {len(matrix)}/{len(cummings)}/{len(shoot)}"
    )
    assert_allclose(cummings[:n], matrix[:n], rtol=5e-3)
    assert_allclose(shoot[:n], matrix[:n], rtol=5e-3)
    assert_allclose(shoot[:n], cummings[:n], rtol=5e-3)


@pytest.mark.slow
def test_three_backends_agree_on_dipole_l6_uniform_full_ladder() -> None:
    """All three backends agree on dipole L=6 + UniformDensity, modes 1..4.

    Pure dipole + UniformDensity has a closely-spaced ladder (mode_2/mode_1
    ≈ 2.6), so the shooter's bracket scan needs a fine ``resolution`` —
    too coarse and it skips intermediate modes. The matrix/Cummings
    backends solve all modes in one eigh_tridiagonal call and do not
    have this failure mode.
    """
    base = dict(
        l_shell=6.0,
        n_modes=4,
        density_model=UniformDensity(n0=1e7),
        freq_range=(1e-6, 3.0e-4),
        resolution=2000,
    )
    matrix = _solve("matrix", **base)
    cummings = _solve("cummings", **base)
    shoot = _solve("shoot", **base)

    n = min(len(matrix), len(cummings), len(shoot))
    assert n >= 4, (
        f"expected ≥4 modes from each backend; got {len(matrix)}/{len(cummings)}/{len(shoot)}"
    )
    assert_allclose(cummings[:n], matrix[:n], rtol=5e-3)
    assert_allclose(shoot[:n], matrix[:n], rtol=1e-2)
    assert_allclose(shoot[:n], cummings[:n], rtol=1e-2)


@pytest.mark.slow
@pytest.mark.parametrize("density_model", ["bagenal", "persoon"])
def test_three_backends_agree_across_density_models(density_model: str) -> None:
    """All three backends agree on mode 1 at KMAG L=8 across density models."""
    base = dict(
        l_shell=8.0,
        n_modes=2,
        field=SaturnField(),
        local_time_hours=12.0,
        density_model=density_model,
        freq_range=(1e-6, 1e-2),
        resolution=400,
    )
    matrix = _solve("matrix", **base)
    cummings = _solve("cummings", **base)
    shoot = _solve("shoot", **base)

    assert_allclose(cummings[0], matrix[0], rtol=5e-3)
    assert_allclose(shoot[0], matrix[0], rtol=5e-3)
    assert_allclose(shoot[0], cummings[0], rtol=5e-3)


@pytest.mark.slow
def test_three_backends_agree_at_l20_full_ladder() -> None:
    """The high-L Rusaitis-anchor case must show three-backend agreement too."""
    base = dict(
        l_shell=20.0,
        n_modes=4,
        field=SaturnField(),
        local_time_hours=12.0,
    )
    matrix = _solve("matrix", **base)
    cummings = _solve("cummings", **base)
    shoot = _solve("shoot", **base)
    n = min(len(matrix), len(cummings), len(shoot))
    assert n == 4
    assert_allclose(cummings, matrix, rtol=5e-3)
    assert_allclose(shoot, matrix, rtol=5e-3)
    assert_allclose(shoot, cummings, rtol=5e-3)
