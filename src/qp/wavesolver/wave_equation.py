r"""Standing Alfvén wave equation along a magnetic field line.

The transverse displacement $y$ satisfies:

$$\frac{d^2 y}{ds^2} + \frac{d \ln(h_i^2 B)}{ds} \frac{dy}{ds}
  + \frac{\omega^2}{v_A^2(s)} \, y = 0$$

where $s$ is arc length, $h_i$ is the scale factor ($i=1$ toroidal, $i=2$
poloidal), $B$ is field magnitude, $v_A$ is Alfvén speed, and $\omega$ is
the angular frequency.

This is converted to a first-order system for `scipy.integrate.solve_ivp`:

- $y_0 = y$ (displacement)
- $y_1 = dy/ds$ (velocity)

References
----------
- Singer, H.J. et al. (1981), J. Geophys. Res., 86, 4589 (arc-length formulation)
- Cummings, W.D. et al. (1969), J. Geophys. Res., 74, 778 (cos-θ formulation)
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


def integrate_wave_equation(
    omega: float,
    s_span: tuple[float, float],
    va_spline: CubicSpline,
    dlnh_spline: CubicSpline,
    y0: tuple[float, float] = (0.0, 1.0),
    n_points: int = 500,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Integrate the wave equation for a given frequency.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    s_span : tuple[float, float]
        Arc-length integration domain (s_min, s_max) in meters.
    va_spline : CubicSpline
        Alfvén velocity interpolant v_A(s).
    dlnh_spline : CubicSpline
        Gradient d/ds ln(h²B) interpolant.
    y0 : tuple[float, float]
        Initial conditions (y, dy/ds) at s_span[0].
        Default (0, 1) = zero displacement, unit velocity at the boundary.
    n_points : int
        Number of output points.
    method : str
        scipy.integrate.solve_ivp method.
    rtol, atol : float
        Relative and absolute tolerances.

    Returns
    -------
    s : ndarray, shape (n_points,)
        Arc-length coordinate.
    y : ndarray, shape (n_points,)
        Displacement eigenfunction.
    dy : ndarray, shape (n_points,)
        Derivative of displacement.
    """
    s_eval = np.linspace(s_span[0], s_span[1], n_points)

    def rhs(s, state):
        y, dy = state[0], state[1]
        va = float(va_spline(s))
        dlnh = float(dlnh_spline(s))
        d2y = -dlnh * dy - (omega / va) ** 2 * y
        return [dy, d2y]

    sol = solve_ivp(
        rhs,
        s_span,
        list(y0),
        method=method,
        t_eval=s_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed at ω={omega:.6e}: {sol.message}")

    return sol.t, sol.y[0], sol.y[1]


def boundary_error(
    omega: float,
    s_span: tuple[float, float],
    va_spline: CubicSpline,
    dlnh_spline: CubicSpline,
    y0: tuple[float, float] = (0.0, 1.0),
    target: float = 0.0,
    *,
    va_samples: np.ndarray | None = None,
    dlnh_samples: np.ndarray | None = None,
    **kwargs,
) -> float:
    """Evaluate the boundary condition error for a trial frequency.

    Integrates the wave equation from s_min to s_max and returns
    y(s_max) - target. Eigenfrequencies are zeros of this function.

    When ``va_samples`` and ``dlnh_samples`` are provided (pre-sampled
    on a uniform grid), uses a numba JIT RK4 integrator for ~50× speedup.
    Otherwise falls back to scipy solve_ivp.

    Parameters
    ----------
    omega : float
        Trial angular frequency (rad/s).
    target : float
        Target displacement at s_max (usually 0 for nodal BC).
    va_samples, dlnh_samples : ndarray or None
        Pre-sampled arrays on a uniform grid spanning s_span.
        If provided, the fast numba path is used.
    """
    if va_samples is not None and dlnh_samples is not None:
        y_end = _scan_boundary_errors(
            np.array([omega]),
            s_span[0], s_span[1],
            va_samples, dlnh_samples,
            y0[0], y0[1],
        )[0]
        return y_end - target

    # Fallback: scipy solve_ivp
    _, y, _ = integrate_wave_equation(
        omega,
        s_span,
        va_spline,
        dlnh_spline,
        y0=y0,
        n_points=200,
        method="LSODA",
        **kwargs,
    )
    return y[-1] - target


# ======================================================================
# Numba JIT RK4 integrator (fast path for boundary_error)
# ======================================================================


@njit(cache=True)
def _interp_uniform(
    s: float, s_min: float, ds_inv: float, samples: np.ndarray
) -> float:
    """Fast linear interpolation on a uniform grid."""
    n = len(samples)
    idx_f = (s - s_min) * ds_inv
    idx = int(idx_f)
    if idx < 0:
        return samples[0]
    if idx >= n - 1:
        return samples[n - 1]
    frac = idx_f - idx
    return samples[idx] + frac * (samples[idx + 1] - samples[idx])


@njit(cache=True)
def _scan_boundary_errors(
    omegas: np.ndarray,
    s_min: float,
    s_max: float,
    va_samples: np.ndarray,
    dlnh_samples: np.ndarray,
    y0: float,
    dy0: float,
    n_steps: int = 3000,
) -> np.ndarray:
    """Evaluate boundary errors for an array of trial frequencies.

    Integrates the wave equation via RK4 for each omega entirely in compiled
    code. Returns y(s_max) for each frequency (eigenfrequencies are zeros).
    """
    n_samp = len(va_samples)
    ds_inv = (n_samp - 1) / (s_max - s_min)
    h = (s_max - s_min) / n_steps
    n_omega = len(omegas)
    errors = np.empty(n_omega)

    for k in range(n_omega):
        omega2 = omegas[k] * omegas[k]
        y = y0
        dy = dy0
        s = s_min

        for _ in range(n_steps):
            va1 = _interp_uniform(s, s_min, ds_inv, va_samples)
            dl1 = _interp_uniform(s, s_min, ds_inv, dlnh_samples)
            ky1 = dy
            kdy1 = -dl1 * dy - omega2 / (va1 * va1) * y

            s2 = s + 0.5 * h
            y2 = y + 0.5 * h * ky1
            dy2 = dy + 0.5 * h * kdy1
            va2 = _interp_uniform(s2, s_min, ds_inv, va_samples)
            dl2 = _interp_uniform(s2, s_min, ds_inv, dlnh_samples)
            ky2 = dy2
            kdy2 = -dl2 * dy2 - omega2 / (va2 * va2) * y2

            # k3 (same s as k2 — reuse va2, dl2)
            y3 = y + 0.5 * h * ky2
            dy3 = dy + 0.5 * h * kdy2
            ky3 = dy3
            kdy3 = -dl2 * dy3 - omega2 / (va2 * va2) * y3

            s4 = s + h
            y4 = y + h * ky3
            dy4 = dy + h * kdy3
            va4 = _interp_uniform(s4, s_min, ds_inv, va_samples)
            dl4 = _interp_uniform(s4, s_min, ds_inv, dlnh_samples)
            ky4 = dy4
            kdy4 = -dl4 * dy4 - omega2 / (va4 * va4) * y4

            y += h / 6.0 * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4)
            dy += h / 6.0 * (kdy1 + 2.0 * kdy2 + 2.0 * kdy3 + kdy4)
            s += h

        errors[k] = y

    return errors


def count_mode_number(dy: np.ndarray) -> int:
    """Determine the harmonic mode number from the eigenfunction derivative.

    The mode number equals the number of sign changes in dy/ds.
    Mode 1 (fundamental) has 0 sign changes in dy, mode 2 has 1, etc.
    Following the convention: m = number of dy sign flips.
    """
    sign_changes = np.where(np.sign(dy[:-1]) != np.sign(dy[1:]))[0]
    return len(sign_changes)
