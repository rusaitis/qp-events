r"""Shared Sturm–Liouville finite-difference eigensolver kernel.

The toroidal/poloidal Alfvén wave equation reduces to the self-adjoint
Sturm–Liouville form

.. math::

    -\,\frac{d}{d\xi}\!\left[p(\xi)\,\frac{dy}{d\xi}\right]
        \;=\; \omega^{2}\, w(\xi)\, y,

with Dirichlet BCs :math:`y(\xi_{\min}) = y(\xi_{\max}) = 0`. The
matrix and Cummings backends each build :math:`p` and :math:`w` from
physical fields (in arc length :math:`s` and in :math:`\mu = z/r`
respectively), but the discretisation kernel — cell-face averaging of
:math:`p`, tridiagonal assembly, :math:`M^{-1/2}` rescaling, and
``scipy.linalg.eigh_tridiagonal`` — is identical. This module hosts
that shared kernel so that any future change to the discretisation
applies to both backends by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SLEigenResult:
    """Output of :func:`solve_sl_uniform_grid`.

    Attributes
    ----------
    omegas : ndarray
        Angular frequencies :math:`\\omega_k = \\sqrt{\\lambda_k}` in
        ascending order.
    eigenfunctions_full : ndarray | None
        Eigenfunctions on the *full* grid (length ``N+1``) with the
        Dirichlet endpoints prepended/appended as zeros. ``None`` if
        ``include_eigenfunctions`` was False.
    """

    omegas: np.ndarray
    eigenfunctions_full: np.ndarray | None


def solve_sl_uniform_grid(
    grid: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    n_modes: int,
    *,
    include_eigenfunctions: bool = True,
    mass_condition_warn: float = 1e8,
    diagnostic_label: str = "",
) -> SLEigenResult:
    r"""Solve :math:`-(p y')' = \omega^2 w y` on a uniform grid.

    Cell-face values use the arithmetic mean
    :math:`p_{i+1/2} = (p_i + p_{i+1})/2`. The standard symmetric
    tridiagonal form is obtained by the diagonal rescaling
    :math:`y = M^{-1/2} u` (which preserves tridiagonal structure since
    :math:`M = \mathrm{diag}(w_i)`).

    Parameters
    ----------
    grid : ndarray, shape (N+1,)
        Uniformly spaced coordinate grid.
    p, w : ndarray, shape (N+1,)
        Sturm–Liouville coefficients. Both must be strictly positive on
        the interior (``i=1..N-1``).
    n_modes : int
        Number of lowest eigenmodes to compute.
    include_eigenfunctions : bool
        If True, return eigenfunctions on the full grid (endpoints
        explicitly zero).
    mass_condition_warn : float
        Log a warning when ``w.max() / w.min()`` exceeds this ratio —
        the :math:`M^{-1/2}` rescaling becomes ill-conditioned beyond
        this point and lowest eigenvalues may lose precision.
    diagnostic_label : str
        Prefix for the warning message (e.g. ``"Cummings μ-coordinate"``).

    Returns
    -------
    SLEigenResult
        ``omegas`` (length ``n_modes``) and optionally
        ``eigenfunctions_full`` (shape ``(N+1, n_modes)``).

    Raises
    ------
    ValueError
        If the grid is not uniform, shapes mismatch, ``N < 5``, or any
        interior ``w`` is non-positive.
    """
    n_plus_1 = len(grid)
    if n_plus_1 < 5:
        raise ValueError(f"Need at least 5 grid points; got {n_plus_1}")
    if p.shape != grid.shape or w.shape != grid.shape:
        raise ValueError(
            f"grid/p/w shapes must match; got {grid.shape}, {p.shape}, {w.shape}"
        )

    diffs = np.diff(grid)
    h = float(diffs[0])
    if not np.allclose(diffs, h, rtol=1e-6):
        raise ValueError("grid must be uniformly spaced")

    # Cell-face p values, length N
    p_half = 0.5 * (p[:-1] + p[1:])

    # Interior indexing i' = i - 1, for i = 1..N-1 (count N-1)
    p_left = p_half[:-1]  # p_{i-1/2} for i=1..N-1; length N-1
    p_right = p_half[1:]  # p_{i+1/2} for i=1..N-1; length N-1

    diag_A = (p_left + p_right) / h**2  # (N-1,)
    off_A = -p_right[:-1] / h**2  # (N-2,)

    # Interior mass weights
    w_int = w[1:-1]
    if np.any(w_int <= 0):
        raise ValueError("Mass weights must be strictly positive on the interior")

    sqrt_w = np.sqrt(w_int)
    n_modes_req = max(1, min(n_modes, len(w_int)))

    mass_ratio = float(w_int.max() / w_int.min())
    if mass_ratio > mass_condition_warn:
        prefix = f"{diagnostic_label} " if diagnostic_label else ""
        log.warning(
            "%smass matrix w spans %.2e orders of magnitude; "
            "the lowest eigenvalues may lose precision.",
            prefix,
            mass_ratio,
        )

    diag_t = diag_A / w_int
    off_t = off_A / (sqrt_w[:-1] * sqrt_w[1:])
    eigvals, eigvecs_raw = eigh_tridiagonal(
        diag_t,
        off_t,
        select="i",
        select_range=(0, n_modes_req - 1),
    )
    eigvecs_int = eigvecs_raw / sqrt_w[:, None]
    omegas = np.sqrt(np.maximum(eigvals, 0.0))

    if not include_eigenfunctions:
        return SLEigenResult(omegas=omegas, eigenfunctions_full=None)

    eigenfunctions_full = np.zeros((n_plus_1, len(omegas)))
    for k in range(len(omegas)):
        y_int = eigvecs_int[:, k]
        if y_int[0] < 0:
            y_int = -y_int
        eigenfunctions_full[1:-1, k] = y_int
    return SLEigenResult(omegas=omegas, eigenfunctions_full=eigenfunctions_full)
