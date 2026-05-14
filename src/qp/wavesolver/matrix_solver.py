r"""Matrix-based Sturm-Liouville eigensolver for standing Alfvén waves.

The toroidal/poloidal wave equation in self-adjoint form is

.. math::

    -\,\frac{d}{ds}\!\left[p(s)\,\frac{dy}{ds}\right]
        \;=\; \omega^{2}\, w(s)\, y,\qquad
    p = h_\alpha^{2} B,\quad w = \frac{h_\alpha^{2} B}{v_A^{2}},

with Dirichlet boundary conditions ``y(s_min) = y(s_max) = 0``.

This module discretises the operator on a uniform grid and solves the
resulting symmetric tridiagonal generalised eigenvalue problem
``A y = ω² M y``. Compared to the shooting solver in
:mod:`qp.wavesolver.eigensolver` it eliminates the bracket-scan layer,
the ``freq_range`` configuration that has historically clipped fundamentals,
and the ``y'(0) = 1`` launch condition — finding all requested modes in
one ``scipy.linalg.eigh_tridiagonal`` call.
"""

from __future__ import annotations

import logging

import numpy as np

from qp.wavesolver._sl_kernel import solve_sl_uniform_grid
from qp.wavesolver.result import EigenMode

log = logging.getLogger(__name__)


def find_eigenfrequencies_matrix(
    s: np.ndarray,
    h_alpha: np.ndarray,
    B: np.ndarray,
    va: np.ndarray,
    n_modes: int = 6,
    include_eigenfunctions: bool = True,
) -> list[EigenMode]:
    r"""Solve the standing-wave eigenproblem via finite differences.

    Parameters
    ----------
    s : ndarray, shape (N+1,)
        Arc-length grid (m), uniformly spaced.
    h_alpha : ndarray, shape (N+1,)
        Scale factor in the polarisation direction (``h1 = r·sinθ`` for
        toroidal, ``h2`` for poloidal). Same units as ``s``.
    B : ndarray, shape (N+1,)
        Magnetic field magnitude on the grid (T).
    va : ndarray, shape (N+1,)
        Alfvén speed on the grid (m/s).
    n_modes : int
        Number of lowest eigenmodes to return.
    include_eigenfunctions : bool
        If True, populate ``EigenMode.eigenfunction`` and ``arc_length``.

    Returns
    -------
    list[EigenMode]
        ``n_modes`` eigenmodes sorted by ascending angular frequency. The
        ``mode_number`` field is the count of dy/ds sign changes, which
        for clean Sturm-Liouville eigenfunctions equals the mode index
        (1-based).

    Notes
    -----
    The cell-face coefficient uses an arithmetic mean
    ``p_{i+1/2} = (p_i + p_{i+1}) / 2``. The standard tridiagonal form is
    obtained by the diagonal rescaling ``y = M^{-1/2} u``, which preserves
    tridiagonal structure because ``M = diag(w_i)``.
    """
    if h_alpha.shape != s.shape or B.shape != s.shape or va.shape != s.shape:
        raise ValueError(
            "s, h_alpha, B, va must have identical shape; got "
            f"{s.shape}, {h_alpha.shape}, {B.shape}, {va.shape}"
        )

    # SL coefficients in arc-length: p = h_α² B, w = h_α² B / v_A²
    p = h_alpha**2 * B
    w = p / va**2

    sl = solve_sl_uniform_grid(
        s,
        p,
        w,
        n_modes,
        include_eigenfunctions=include_eigenfunctions,
        diagnostic_label="arc-length",
    )

    modes: list[EigenMode] = []
    for k, omega in enumerate(sl.omegas):
        if include_eigenfunctions and sl.eigenfunctions_full is not None:
            y_full = sl.eigenfunctions_full[:, k]
            dy_full = np.gradient(y_full, s)
            mode_num = int(np.sum(np.sign(dy_full[:-1]) != np.sign(dy_full[1:])))
            modes.append(
                EigenMode(
                    angular_frequency=float(omega),
                    mode_number=mode_num if mode_num > 0 else k + 1,
                    eigenfunction=y_full,
                    eigenfunction_derivative=dy_full,
                    arc_length=s.copy(),
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
