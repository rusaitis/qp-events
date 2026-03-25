r"""Compute physical profiles along a traced magnetic field line.

Given a ``FieldLineTrace`` (positions + B-field from the tracer) and a
``DensityModel``, this module computes everything the eigensolver needs:

- Arc length $s$
- Field magnitude $|B|$
- Number density $n(s)$
- Alfvén velocity $v_A(s)$
- Scale factors $h_1(s)$, $h_2(s)$
- Spline interpolants of $v_A$ and $d/ds \ln(h_i^2 B)$

The splines are the direct inputs to ``wave_equation.integrate_wave_equation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from qp.coords.transforms import car2sph
from qp.fieldline.tracer import FieldLineTrace
from qp.wavesolver.density import (
    SATURN_RADIUS,
    DensityModel,
    alfven_velocity,
)


@dataclass(slots=True)
class FieldLineProfile:
    """All physical quantities along a traced field line.

    Attributes
    ----------
    arc_length : ndarray
        Distance along field line from south footpoint (m).
    positions : ndarray, shape (N, 3)
        Cartesian positions (R_S).
    field_magnitude : ndarray
        |B| in Tesla.
    density : ndarray
        Number density (m⁻³).
    alfven_velocity_profile : ndarray
        Alfvén speed (m/s).
    h1 : ndarray
        Toroidal scale factor (m).
    h2 : ndarray
        Poloidal scale factor (m⁻¹ T⁻¹, but dimensionally 1/(B·h1)).
    l_shell : float
        L-shell (equatorial crossing distance in R_S).
    conjugate_latitude : float
        Invariant latitude at the footpoint (degrees).
    equator_index : int
        Index of the equatorial crossing in the arrays.

    va_spline : CubicSpline
        v_A(s) interpolant.
    dlnh1B_spline : CubicSpline
        d/ds ln(h1² B) for toroidal modes.
    dlnh2B_spline : CubicSpline
        d/ds ln(h2² B) for poloidal modes.
    """

    arc_length: np.ndarray
    positions: np.ndarray
    field_magnitude: np.ndarray
    density: np.ndarray
    alfven_velocity_profile: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    l_shell: float
    conjugate_latitude: float
    equator_index: int

    va_spline: CubicSpline
    dlnh1B_spline: CubicSpline
    dlnh2B_spline: CubicSpline

    # Pre-sampled uniform grids for numba fast path
    va_samples: np.ndarray | None = None
    dlnh1B_samples: np.ndarray | None = None
    dlnh2B_samples: np.ndarray | None = None

    @property
    def s_span_meters(self) -> tuple[float, float]:
        """Integration domain (s_min, s_max) in meters."""
        return (float(self.arc_length[0]), float(self.arc_length[-1]))

    @property
    def length_rs(self) -> float:
        """Total field line length in R_S."""
        return (self.arc_length[-1] - self.arc_length[0]) / SATURN_RADIUS


def compute_field_line_profile(
    trace: FieldLineTrace,
    density_model: DensityModel,
) -> FieldLineProfile:
    """Build a complete field line profile from a traced field line.

    Parameters
    ----------
    trace : FieldLineTrace
        Output of ``trace_fieldline_bidirectional`` — positions and B-field
        along the field line, from one footpoint to the other.
    density_model : DensityModel
        Plasma density model (e.g., ``BagenalDelamere()``).

    Returns
    -------
    FieldLineProfile
        Complete profile with precomputed spline interpolants ready for
        the eigensolver.
    """
    positions = trace.positions  # (N, 3) in R_S
    B_nT = trace.field_magnitude  # nT
    B_T = B_nT * 1e-9  # Tesla

    # 1. Arc length (cumulative distance between consecutive points, in meters)
    diffs = np.diff(positions, axis=0)  # (N-1, 3)
    segment_lengths = np.linalg.norm(diffs, axis=1) * SATURN_RADIUS  # meters
    arc_length = np.zeros(len(positions))
    arc_length[1:] = np.cumsum(segment_lengths)

    # 2. Find equatorial crossing (maximum radial distance)
    r = np.linalg.norm(positions, axis=1)
    eq_idx = int(np.argmax(r))
    l_shell = float(r[eq_idx])

    # 3. Conjugate latitude from the first footpoint
    rtp = car2sph(positions)
    theta_foot = rtp[0, 1]  # colatitude of first footpoint
    conj_lat = 90.0 - np.degrees(theta_foot)

    # 4. Arc length at the equator (for density model)
    s_equator_m = arc_length[eq_idx]
    s_equator_rs = s_equator_m / SATURN_RADIUS

    # 5. Compute density along the field line
    arc_length_rs = arc_length / SATURN_RADIUS
    n = density_model.field_aligned_density(l_shell, arc_length_rs, s_equator_rs)

    # 6. Alfvén velocity
    va = alfven_velocity(B_T, n, density_model.ion_mass_kg)

    # 7. Scale factors (analytical dipole approximation)
    #    h1 = r·sin(θ) [toroidal], h2 = 1/(|B|·h1) [poloidal]
    r_m = r * SATURN_RADIUS  # R_S → meters
    sin_theta = np.sin(rtp[:, 1])
    h1 = r_m * sin_theta
    # Clamp h1 to avoid division by zero near the poles
    h1 = np.maximum(h1, 1.0)
    h2 = 1.0 / (B_T * h1)

    # 8. Build spline interpolants
    va_spline = CubicSpline(arc_length, va)

    # d/ds ln(h1² B)  and  d/ds ln(h2² B)
    # Fit spline to ln(h²B), then take analytical derivative — smoother and
    # more accurate than numerical gradient + Savitzky-Golay smoothing.
    lnh1B_spline = CubicSpline(arc_length, np.log(h1**2 * B_T))
    lnh2B_spline = CubicSpline(arc_length, np.log(h2**2 * B_T))
    dlnh1B_spline = lnh1B_spline.derivative()
    dlnh2B_spline = lnh2B_spline.derivative()

    # 9. Pre-sample onto uniform grids for numba fast path
    N_SAMPLES = 5000
    s_uniform = np.linspace(arc_length[0], arc_length[-1], N_SAMPLES)
    va_samples = np.ascontiguousarray(va_spline(s_uniform))
    dlnh1B_samples = np.ascontiguousarray(dlnh1B_spline(s_uniform))
    dlnh2B_samples = np.ascontiguousarray(dlnh2B_spline(s_uniform))

    return FieldLineProfile(
        arc_length=arc_length,
        positions=positions,
        field_magnitude=B_T,
        density=n,
        alfven_velocity_profile=va,
        h1=h1,
        h2=h2,
        l_shell=l_shell,
        conjugate_latitude=conj_lat,
        equator_index=eq_idx,
        va_spline=va_spline,
        dlnh1B_spline=dlnh1B_spline,
        dlnh2B_spline=dlnh2B_spline,
        va_samples=va_samples,
        dlnh1B_samples=dlnh1B_samples,
        dlnh2B_samples=dlnh2B_samples,
    )
