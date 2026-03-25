"""Eigenfrequency solver using bracket search + Brent's method.

Finds angular frequencies ω where the standing wave boundary condition
is satisfied, by:

1. Scanning the error function f(ω) = y(s_max; ω) over a frequency grid
2. Detecting sign changes to bracket each root
3. Refining with scipy.optimize.brentq (guaranteed superlinear convergence)
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from qp.wavesolver.result import EigenMode
from qp.wavesolver.wave_equation import (
    _scan_boundary_errors,
    boundary_error,
    count_mode_number,
    integrate_wave_equation,
)

log = logging.getLogger(__name__)


def find_eigenfrequencies(
    s_span: tuple[float, float],
    va_spline: CubicSpline,
    dlnh_spline: CubicSpline,
    freq_range: tuple[float, float] = (1e-5, 0.01),
    n_modes: int = 6,
    resolution: int = 200,
    tolerance: float = 1e-7,
    include_eigenfunctions: bool = False,
    n_eigenfunction_points: int = 500,
    va_samples: np.ndarray | None = None,
    dlnh_samples: np.ndarray | None = None,
) -> list[EigenMode]:
    """Find standing wave eigenfrequencies along a field line.

    Parameters
    ----------
    s_span : tuple[float, float]
        Arc-length integration domain (s_min, s_max) in meters.
    va_spline : CubicSpline
        Alfvén velocity interpolant v_A(s).
    dlnh_spline : CubicSpline
        Gradient d/ds ln(h²B) interpolant.
    freq_range : tuple[float, float]
        Angular frequency search range (ω_min, ω_max) in rad/s.
    n_modes : int
        Number of eigenmodes to find.
    resolution : int
        Number of frequency samples for bracket search.
    tolerance : float
        Root-finding tolerance for brentq.
    include_eigenfunctions : bool
        If True, compute and store eigenfunctions for each mode.
    n_eigenfunction_points : int
        Spatial resolution for eigenfunctions.

    Returns
    -------
    list[EigenMode]
        Found eigenmodes sorted by frequency.
    """
    w_min, w_max = freq_range

    # Adaptive bracket search: expand range if we don't find enough roots
    brackets = _find_brackets(
        s_span,
        va_spline,
        dlnh_spline,
        w_min,
        w_max,
        n_modes,
        resolution,
        va_samples=va_samples,
        dlnh_samples=dlnh_samples,
    )

    # Refine each bracket with Brent's method
    modes = []
    for w_lo, w_hi in brackets[:n_modes]:
        try:

            def _err(w):
                return boundary_error(
                    w,
                    s_span,
                    va_spline,
                    dlnh_spline,
                    va_samples=va_samples,
                    dlnh_samples=dlnh_samples,
                )

            omega = brentq(_err, w_lo, w_hi, xtol=tolerance, rtol=tolerance)
        except ValueError:
            log.warning("brentq failed for bracket [%.6e, %.6e], skipping", w_lo, w_hi)
            continue

        eigenfunction = eigenfunction_deriv = arc_length = None
        mode_number = len(modes) + 1  # default: sequential

        if include_eigenfunctions:
            s, y, dy = integrate_wave_equation(
                omega,
                s_span,
                va_spline,
                dlnh_spline,
                n_points=n_eigenfunction_points,
            )
            mode_number = count_mode_number(dy)
            eigenfunction = y
            eigenfunction_deriv = dy
            arc_length = s

        modes.append(
            EigenMode(
                angular_frequency=omega,
                mode_number=mode_number,
                eigenfunction=eigenfunction,
                eigenfunction_derivative=eigenfunction_deriv,
                arc_length=arc_length,
            )
        )

    # Sort by frequency
    modes.sort(key=lambda m: m.angular_frequency)

    # Re-assign mode numbers sequentially if eigenfunctions not computed
    if not include_eigenfunctions:
        modes = [
            EigenMode(
                angular_frequency=m.angular_frequency,
                mode_number=i + 1,
            )
            for i, m in enumerate(modes)
        ]

    return modes


def _find_brackets(
    s_span: tuple[float, float],
    va_spline: CubicSpline,
    dlnh_spline: CubicSpline,
    w_min: float,
    w_max: float,
    n_modes: int,
    resolution: int,
    max_expansions: int = 10,
    va_samples: np.ndarray | None = None,
    dlnh_samples: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """Find bracket intervals containing eigenfrequency roots.

    Scans the boundary error function over a frequency grid, detects
    sign changes, and adaptively expands the range if not enough roots
    are found.
    """
    use_fast_path = va_samples is not None and dlnh_samples is not None
    brackets: list[tuple[float, float]] = []
    last_error: float | None = None
    scan_min = w_min

    for _ in range(max_expansions):
        omegas = np.linspace(scan_min, w_max, resolution)
        if use_fast_path:
            assert va_samples is not None and dlnh_samples is not None
            errors = _scan_boundary_errors(
                omegas,
                s_span[0],
                s_span[1],
                va_samples,
                dlnh_samples,
                0.0,
                1.0,
            )
        else:
            errors = np.array(
                [boundary_error(w, s_span, va_spline, dlnh_spline) for w in omegas]
            )

        # Check for sign change at the boundary between previous and new scan
        if last_error is not None and np.sign(last_error) != np.sign(errors[0]):
            brackets.append(
                (float(omegas[0] - (omegas[1] - omegas[0])), float(omegas[0]))
            )

        # Find sign changes within this scan
        sign_changes = np.where(np.sign(errors[:-1]) != np.sign(errors[1:]))[0]
        brackets.extend((float(omegas[i]), float(omegas[i + 1])) for i in sign_changes)
        last_error = float(errors[-1])

        log.debug(
            "Bracket search [%.2e, %.2e] res=%d: found %d roots (need %d)",
            scan_min,
            w_max,
            resolution,
            len(brackets),
            n_modes,
        )

        if len(brackets) >= n_modes:
            return brackets[:n_modes]

        # Expand range: only scan the new extension next iteration
        scan_min = w_max
        if len(brackets) > 0:
            avg_spacing = (w_max - w_min) / max(len(brackets), 1)
            w_max += avg_spacing * (n_modes - len(brackets) + 1)
        else:
            w_max *= 2.0

    log.warning(
        "Found only %d/%d brackets after %d expansions",
        len(brackets),
        n_modes,
        max_expansions,
    )
    return brackets
