r"""Cummings (1969) cos-θ Sturm–Liouville eigensolver.

The toroidal/poloidal wave equation in self-adjoint form is

.. math::

    -\,\frac{d}{ds}\!\left[p(s)\,\frac{dy}{ds}\right]
        \;=\; \omega^{2}\, w(s)\, y,\qquad
    p = h_\alpha^{2} B,\quad w = \frac{h_\alpha^{2} B}{v_A^{2}},

with Dirichlet boundary conditions ``y(s_min) = y(s_max) = 0``.

This module reparametrises from arc length :math:`s` to the dimensionless
:math:`\mu = z/r = \cos\theta_{\text{mag}}`, where :math:`\theta_{\text{mag}}`
is the magnetic colatitude. Substituting :math:`y'(s) = y'(\mu)/J` with
:math:`J = ds/d\mu` and multiplying through by :math:`J` puts the equation
back in self-adjoint form on the :math:`\mu` grid,

.. math::

    -\,\frac{d}{d\mu}\!\left[\frac{p}{J}\,\frac{dy}{d\mu}\right]
        \;=\; \omega^{2}\, (w\,J)\, y.

The transformation absorbs the boundary asymptotics: as the trace dips
toward the ionosphere, the field line bends sharply in s but covers a
small range in :math:`\mu`, so :math:`J = ds/d\mu \to \infty` near the
footpoints. The effective mass :math:`w_\mu = wJ` blows up there and
the effective stiffness :math:`p_\mu = p/J` shrinks — the soft-wall
mode that arises in the arc-length formulation (because
:math:`(\omega/v_A)^2 y \to 0` at high :math:`v_A`) is suppressed by
construction. Cummings (1969) used this coordinate (``mu`` is their
``cos θ``) to derive the FLR eigenmodes that match the WKB-asymptotic
ladder, which is also the ladder the Rusaitis (2021) reference table
follows.

References
----------
Cummings, W.D., O'Sullivan, R.J., Coleman, P.J. Jr. (1969),
*Standing Alfvén waves in the magnetosphere*, J. Geophys. Res., 74, 778.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import CubicSpline

from qp.wavesolver._sl_kernel import solve_sl_uniform_grid
from qp.wavesolver.result import EigenMode

log = logging.getLogger(__name__)

# mu-coordinate w_max/w_min is naturally large (Jacobian diverges near μ=±1)
_MASS_CONDITION_WARN = 1e10


def find_eigenfrequencies_cummings(
    positions: np.ndarray,
    arc_length: np.ndarray,
    h_alpha: np.ndarray,
    B: np.ndarray,
    va: np.ndarray,
    n_modes: int = 6,
    include_eigenfunctions: bool = True,
    n_grid: int = 5000,
) -> list[EigenMode]:
    r"""Solve the FLR eigenproblem on a uniform :math:`\mu=z/r` grid.

    Parameters
    ----------
    positions : ndarray, shape (M, 3)
        Cartesian positions along the trace (R_S). Used to compute
        :math:`\mu = z/r`. ``M`` need not equal the sample grid size
        — the trace is the *native* grid; the uniform :math:`\mu` grid
        is built internally.
    arc_length : ndarray, shape (M,)
        Arc length (m) on the *same* native grid as ``positions``.
    h_alpha, B, va : ndarray, shape (M,)
        Scale factor (m), field magnitude (T) and Alfvén speed (m/s) on
        the native grid (matching ``positions``).
    n_modes : int
        Number of lowest eigenmodes to return.
    include_eigenfunctions : bool
        If True, populate ``EigenMode.eigenfunction`` and ``arc_length``.
        The eigenfunction is returned on the uniform :math:`\mu` grid.
    n_grid : int
        Uniform :math:`\mu` grid resolution. 5000 matches the matrix
        solver default.

    Returns
    -------
    list[EigenMode]
        ``n_modes`` eigenmodes sorted by ascending angular frequency.

    Raises
    ------
    ValueError
        If :math:`\mu` is not monotonic along the native grid (typical
        cause: a stretched current-sheet trace that bends back along z).
    """
    # 1. μ = z/r along the trace
    r = np.linalg.norm(positions, axis=1)
    mu = positions[:, 2] / r

    # Order so μ increases from south footpoint to north footpoint.
    if mu[-1] < mu[0]:
        order = slice(None, None, -1)
        mu = mu[order]
        # arc length monotonicity must be preserved
        arc_length = arc_length[-1] - arc_length[order]
        h_alpha = h_alpha[order]
        B = B[order]
        va = va[order]

    # If μ has small non-monotonicities (current-sheet stretching), sort
    # and de-duplicate. This is a fallback; ideally the trace is monotonic
    # in μ from one footpoint to the other.
    if not np.all(np.diff(mu) > 0):
        ordering = np.argsort(mu)
        mu = mu[ordering]
        h_alpha = h_alpha[ordering]
        B = B[ordering]
        va = va[ordering]
        arc_length = arc_length[ordering]
        keep = np.concatenate([[True], np.diff(mu) > 1e-12])
        if not np.all(keep):
            mu = mu[keep]
            h_alpha = h_alpha[keep]
            B = B[keep]
            va = va[keep]
            arc_length = arc_length[keep]

    if mu.size < 8:
        raise ValueError(
            f"Cummings solver needs ≥8 monotonic μ samples; got {mu.size}"
        )

    # 2. Splines as functions of μ
    h_alpha_of_mu = CubicSpline(mu, h_alpha)
    B_of_mu = CubicSpline(mu, B)
    va_of_mu = CubicSpline(mu, va)
    s_of_mu = CubicSpline(mu, arc_length)

    # 3. Uniform μ grid + sampled fields
    mu_uniform = np.linspace(mu[0], mu[-1], n_grid)
    h_alpha_g = h_alpha_of_mu(mu_uniform)
    B_g = B_of_mu(mu_uniform)
    va_g = va_of_mu(mu_uniform)
    J_g = s_of_mu.derivative()(mu_uniform)  # ds/dμ

    if np.any(J_g <= 0):
        raise ValueError(
            "Jacobian ds/dμ is non-positive somewhere on the grid; "
            "trace is not monotonic in μ"
        )

    # 4. SL coefficients in μ space
    p_mu = h_alpha_g**2 * B_g / J_g
    w_mu = h_alpha_g**2 * B_g * J_g / va_g**2

    sl = solve_sl_uniform_grid(
        mu_uniform, p_mu, w_mu, n_modes,
        include_eigenfunctions=include_eigenfunctions,
        mass_condition_warn=_MASS_CONDITION_WARN,
        diagnostic_label="Cummings μ-coordinate",
    )

    # 5. Pack into EigenMode; report eigenfunctions on the arc-length s
    # grid (sampled via s(μ) spline) so they're consistent with the
    # matrix backend's output.
    s_uniform = s_of_mu(mu_uniform)
    modes: list[EigenMode] = []
    for k, omega in enumerate(sl.omegas):
        if include_eigenfunctions and sl.eigenfunctions_full is not None:
            y_full = sl.eigenfunctions_full[:, k]
            dy_full = np.gradient(y_full, s_uniform)
            mode_num = int(np.sum(np.sign(dy_full[:-1]) != np.sign(dy_full[1:])))
            modes.append(
                EigenMode(
                    angular_frequency=float(omega),
                    mode_number=mode_num if mode_num > 0 else k + 1,
                    eigenfunction=y_full,
                    eigenfunction_derivative=dy_full,
                    arc_length=s_uniform,
                )
            )
        else:
            modes.append(
                EigenMode(
                    angular_frequency=float(omega),
                    mode_number=k + 1,
                )
            )
    return modes
