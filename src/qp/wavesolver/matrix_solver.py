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
from scipy.linalg import eigh_tridiagonal

from qp.wavesolver.result import EigenMode

log = logging.getLogger(__name__)

# Above this w_max/w_min ratio the M^{-1/2} rescaling becomes severely
# ill-conditioned and the lowest eigenvalues may lose precision (warning
# only, not a hard failure — the physics combinations that trip this
# are typically synthetic, e.g. pure dipole field with the Bagenal-Delamere
# density model which does not match the dipole B at L≳5).
_MASS_CONDITION_WARN = 1e8


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
    n_plus_1 = len(s)
    if n_plus_1 < 5:
        raise ValueError(f"Need at least 5 grid points; got {n_plus_1}")
    if h_alpha.shape != s.shape or B.shape != s.shape or va.shape != s.shape:
        raise ValueError(
            "s, h_alpha, B, va must have identical shape; got "
            f"{s.shape}, {h_alpha.shape}, {B.shape}, {va.shape}"
        )

    # Uniform-grid check (cubic spline samples in field_profile already are)
    diffs = np.diff(s)
    h = float(diffs[0])
    if not np.allclose(diffs, h, rtol=1e-6):
        raise ValueError("s grid must be uniformly spaced")

    # Coefficients on the full grid
    p = h_alpha**2 * B           # (N+1,)
    w_full = p / va**2           # (N+1,)

    # Cell-face values p_{i+1/2}, length N
    p_half = 0.5 * (p[:-1] + p[1:])

    # Interior indexing i' = i - 1, for i = 1..N-1 (count N-1)
    p_left = p_half[:-1]   # p_{i-1/2} for i=1..N-1; length N-1
    p_right = p_half[1:]   # p_{i+1/2} for i=1..N-1; length N-1

    diag_A = (p_left + p_right) / h**2                         # (N-1,)
    off_A = -p_right[:-1] / h**2                                # (N-2,)

    # Interior mass weights w_i for i=1..N-1
    w = w_full[1:-1]
    if np.any(w <= 0):
        raise ValueError("Mass weights h_alpha²·B / v_A² must be positive")

    sqrt_w = np.sqrt(w)
    n_modes_req = max(1, min(n_modes, len(w)))

    mass_ratio = float(w.max() / w.min())
    if mass_ratio > _MASS_CONDITION_WARN:
        log.warning(
            "Mass matrix w = h_alpha^2 B / v_A^2 spans %.2e orders of magnitude; "
            "the lowest eigenvalues may lose precision. This typically indicates "
            "a non-physical field/density combination (e.g. pure dipole + "
            "Bagenal-Delamere at L>~5 where the dipole equator field is much "
            "smaller than KMAG, giving v_A_eq ~ km/s).",
            mass_ratio,
        )

    diag_t = diag_A / w
    off_t = off_A / (sqrt_w[:-1] * sqrt_w[1:])
    eigvals, eigvecs_raw = eigh_tridiagonal(
        diag_t,
        off_t,
        select="i",
        select_range=(0, n_modes_req - 1),
    )
    # Recover physical eigenfunctions y = M^{-1/2} u
    eigvecs = eigvecs_raw / sqrt_w[:, None]
    omegas = np.sqrt(np.maximum(eigvals, 0.0))

    modes: list[EigenMode] = []
    for k in range(len(omegas)):
        if include_eigenfunctions:
            y_interior = eigvecs[:, k]
            # Sign convention: positive at the first interior node
            if y_interior[0] < 0:
                y_interior = -y_interior
            y_full = np.zeros(n_plus_1)
            y_full[1:-1] = y_interior
            dy_full = np.gradient(y_full, s)
            mode_num = int(np.sum(np.sign(dy_full[:-1]) != np.sign(dy_full[1:])))
            modes.append(
                EigenMode(
                    angular_frequency=float(omegas[k]),
                    mode_number=mode_num if mode_num > 0 else k + 1,
                    eigenfunction=y_full,
                    eigenfunction_derivative=dy_full,
                    arc_length=s.copy(),
                )
            )
        else:
            modes.append(
                EigenMode(
                    angular_frequency=float(omegas[k]),
                    mode_number=k + 1,
                )
            )
    return modes
