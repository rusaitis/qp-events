"""Field line tracing utilities.

Provides RK4 field line tracing for any magnetic field model:
- `trace_fieldline`: generic tracer accepting a callable field model
- `dipole_field` / `trace_dipole_fieldline`: dipole-specific convenience functions

Extracted from mission_trace.py tracing logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from qp.coords.transforms import car2sph


# Saturn dipole moment (nT * R_S^3)
SATURN_DIPOLE_MOMENT = 380.0  # Approximately 0.2154 Gauss * R_S^3

# Saturn oblateness: (R_eq - R_pol) / R_eq ≈ 0.09796
SATURN_FLATTENING = 0.09796


class FieldModel(Protocol):
    """Protocol for a magnetic field model callable.

    Accepts a Cartesian position (3,) in some coordinate system and returns
    the B-field (3,) in nT in the same coordinate system.
    """

    def __call__(self, position: np.ndarray) -> np.ndarray: ...


@dataclass
class FieldLineTrace:
    """Result of a field line trace.

    Attributes
    ----------
    positions : ndarray, shape (N, 3)
        Traced positions in Cartesian coordinates (R_S).
    field : ndarray, shape (N, 3)
        Magnetic field at each position (nT, Cartesian).
    field_magnitude : ndarray, shape (N,)
        |B| at each position (nT).
    """

    positions: np.ndarray
    field: np.ndarray
    field_magnitude: np.ndarray

    @property
    def spherical(self) -> np.ndarray:
        """Positions in spherical (r, theta, phi)."""
        return car2sph(self.positions)

    @property
    def r(self) -> np.ndarray:
        """Radial distance at each point (R_S)."""
        return np.linalg.norm(self.positions, axis=1)

    def conjugate_latitude(self, surface_radius: float = 1.0) -> float | None:
        """Find the invariant latitude at the planet surface.

        Returns latitude in degrees, or None if no intersection found.
        """
        rtp = self.spherical
        r = rtp[:, 0]
        idx = np.argmin(np.abs(r - surface_radius))
        if np.abs(r[idx] - surface_radius) > 0.5:
            return None
        return 90.0 - np.degrees(rtp[idx, 1])


def _surface_radius(x: float, y: float, z: float, flattening: float) -> float:
    """Oblate spheroid surface radius at a given position.

    For Saturn, the surface is an oblate spheroid with equatorial radius 1 R_S
    and polar radius (1 - flattening) R_S.
    """
    if flattening == 0.0:
        return 1.0
    # Geocentric distance to the spheroid surface along the radial direction
    rho = math.sqrt(x**2 + y**2)
    r = math.sqrt(x**2 + y**2 + z**2)
    if r == 0.0:
        return 1.0 - flattening
    cos_lat = rho / r
    sin_lat = abs(z) / r
    b = 1.0 - flattening  # polar radius
    # Ellipse in (rho, z) plane: rho²/1² + z²/b² = r_surf²
    # At direction (cos_lat, sin_lat): r_surf = 1/sqrt(cos²/1² + sin²/b²)
    return 1.0 / math.sqrt(cos_lat**2 + sin_lat**2 / b**2)


def dipole_field(position: ArrayLike) -> np.ndarray:
    """Compute dipole magnetic field at a Cartesian position.

    Parameters
    ----------
    position : array_like, shape (3,) or (N, 3)
        Position in R_S (Cartesian: x, y, z).

    Returns
    -------
    B : ndarray, shape (3,) or (N, 3)
        Magnetic field in nT (Cartesian components).
    """
    pos = np.asarray(position, dtype=float)
    single = pos.ndim == 1
    if single:
        pos = pos[np.newaxis, :]

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r5 = r**5

    M = SATURN_DIPOLE_MOMENT
    # Dipole along z-axis: B = M/(r^5) * (3z*r_vec - r^2 * z_hat)
    Bx = 3 * M * x * z / r5
    By = 3 * M * y * z / r5
    Bz = M * (3 * z**2 - r**2) / r5

    B = np.stack([Bx, By, Bz], axis=-1)
    return B[0] if single else B


def trace_dipole_fieldline(
    start_position: ArrayLike,
    step: float = 0.05,
    max_steps: int = 100000,
    min_radius: float = 1.0,
    max_radius: float = 300.0,
) -> np.ndarray:
    """Trace a field line in a dipole field using RK4.

    Parameters
    ----------
    start_position : array_like, shape (3,)
        Starting position in R_S (Cartesian).
    step : float
        Step size in R_S.
    min_radius : float
        Stop tracing at this radius (planet surface).
    max_radius : float
        Stop tracing at this radius (outer boundary).

    Returns
    -------
    trace : ndarray, shape (N, 3)
        Traced field line positions.
    """
    pos = np.asarray(start_position, dtype=float).copy()
    trace = [pos.copy()]

    for _ in range(max_steps):
        B = dipole_field(pos)
        B_mag = np.linalg.norm(B)
        if B_mag == 0:
            break
        B_unit = B / B_mag

        # RK4 step along field direction
        k1 = step * B_unit
        k2 = step * _dipole_unit(pos + 0.5 * k1)
        k3 = step * _dipole_unit(pos + 0.5 * k2)
        k4 = step * _dipole_unit(pos + k3)
        pos += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        r = np.linalg.norm(pos)
        if r < min_radius or r > max_radius:
            break

        trace.append(pos.copy())

    return np.array(trace)


def trace_dipole_fieldline_bidirectional(
    start_position: ArrayLike,
    step: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """Trace a field line in both directions from the starting point."""
    forward = trace_dipole_fieldline(start_position, step=step, **kwargs)
    backward = trace_dipole_fieldline(start_position, step=-step, **kwargs)
    # Combine: backward (reversed) + forward (skip duplicate start)
    return np.concatenate([backward[::-1], forward[1:]], axis=0)


def field_line_to_spherical(trace_xyz: ArrayLike) -> np.ndarray:
    """Convert traced field line from Cartesian to spherical (r, theta, phi)."""
    return car2sph(np.asarray(trace_xyz))


def conjugate_latitude(
    trace_xyz: ArrayLike, surface_radius: float = 1.0
) -> float | None:
    """Find the conjugate (invariant) latitude where the field line
    intersects the planet surface.

    Returns latitude in degrees, or None if no intersection.
    """
    rtp = field_line_to_spherical(trace_xyz)
    r = rtp[:, 0]

    # Find where r is closest to surface_radius
    idx = np.argmin(np.abs(r - surface_radius))
    if np.abs(r[idx] - surface_radius) > 0.5:  # tolerance of 0.5 R_S
        return None

    theta = rtp[idx, 1]  # colatitude
    lat_deg = 90.0 - np.degrees(theta)
    return lat_deg


def _dipole_unit(pos: np.ndarray) -> np.ndarray:
    """Unit vector of dipole field at position."""
    B = dipole_field(pos)
    B_mag = np.linalg.norm(B)
    return B / B_mag if B_mag > 0 else np.zeros(3)


# ======================================================================
# Generic field line tracer
# ======================================================================


def trace_fieldline(
    field_func: FieldModel,
    start_position: ArrayLike,
    step: float = 0.01,
    max_steps: int = 100000,
    min_radius: float = 1.0,
    max_radius: float = 300.0,
    flattening: float = 0.0,
) -> FieldLineTrace:
    """Trace a field line using RK4 for any magnetic field model.

    Parameters
    ----------
    field_func : callable
        Function that takes position ndarray (3,) and returns B-field
        ndarray (3,) in nT. Must work in the same Cartesian coordinate
        system as the start position.
    start_position : array_like, shape (3,)
        Starting position in R_S (Cartesian).
    step : float
        Step size in R_S. Positive traces along B, negative against.
    max_steps : int
        Maximum number of integration steps.
    min_radius : float
        Stop tracing at this radial distance (planet surface).
        If flattening > 0, the oblate surface is used instead.
    max_radius : float
        Stop tracing at this radial distance (outer boundary).
    flattening : float
        Planet oblateness (0 = sphere, 0.098 = Saturn).

    Returns
    -------
    FieldLineTrace
        Traced positions, field values, and magnitudes.
    """
    pos = np.asarray(start_position, dtype=float).copy()
    positions = [pos.copy()]
    fields = []

    B = field_func(pos)
    fields.append(B.copy())

    for _ in range(max_steps):
        B = field_func(pos)
        B_mag = np.linalg.norm(B)
        if B_mag == 0:
            break
        B_hat = B / B_mag

        # RK4 integration along B-hat direction
        k1 = step * B_hat
        k2 = step * _field_unit(field_func, pos + 0.5 * k1)
        k3 = step * _field_unit(field_func, pos + 0.5 * k2)
        k4 = step * _field_unit(field_func, pos + k3)
        pos = pos + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        r = np.linalg.norm(pos)

        # Check outer boundary
        if r > max_radius:
            break

        # Check surface boundary (oblate or spherical)
        if flattening > 0:
            r_surf = _surface_radius(pos[0], pos[1], pos[2], flattening)
        else:
            r_surf = min_radius
        if r < r_surf:
            break

        B_new = field_func(pos)
        positions.append(pos.copy())
        fields.append(B_new.copy())

    pos_arr = np.array(positions)
    field_arr = np.array(fields)
    mag_arr = np.linalg.norm(field_arr, axis=1)

    return FieldLineTrace(
        positions=pos_arr,
        field=field_arr,
        field_magnitude=mag_arr,
    )


def trace_fieldline_bidirectional(
    field_func: FieldModel,
    start_position: ArrayLike,
    step: float = 0.01,
    **kwargs,
) -> FieldLineTrace:
    """Trace a field line in both directions from the starting point.

    Traces forward (along B) and backward (against B), then concatenates.
    The result runs from one footpoint through the start to the other.

    Parameters
    ----------
    field_func : callable
        Field model function (see trace_fieldline).
    start_position : array_like, shape (3,)
        Starting position in R_S.
    step : float
        Step size in R_S (always positive; sign handled internally).
    **kwargs
        Additional arguments passed to trace_fieldline.
    """
    step = abs(step)
    forward = trace_fieldline(field_func, start_position, step=step, **kwargs)
    backward = trace_fieldline(field_func, start_position, step=-step, **kwargs)

    # Combine: backward (reversed, skip start duplicate) + forward
    positions = np.concatenate([backward.positions[::-1], forward.positions[1:]])
    field = np.concatenate([backward.field[::-1], forward.field[1:]])
    mag = np.concatenate([backward.field_magnitude[::-1], forward.field_magnitude[1:]])

    return FieldLineTrace(positions=positions, field=field, field_magnitude=mag)


def saturn_field_wrapper(
    saturn_field,
    time: float,
    coord: str = "KSM",
) -> FieldModel:
    """Create a field_func callable from a SaturnField instance.

    Parameters
    ----------
    saturn_field : SaturnField
        The field model instance.
    time : float
        Time in the model's configured epoch.
    coord : str
        Coordinate system for positions and field ('KSM', 'S3C', etc.).

    Returns
    -------
    callable
        A function f(pos) -> B that trace_fieldline can use.

    Examples
    --------
    >>> from qp.fieldline.kmag_model import SaturnField
    >>> field = SaturnField()
    >>> field_func = saturn_field_wrapper(field, time=284040000.0)
    >>> trace = trace_fieldline(field_func, [10.0, 0.0, 0.0])
    """

    def field_func(position: np.ndarray) -> np.ndarray:
        bx, by, bz = saturn_field.field_cartesian(
            position[0],
            position[1],
            position[2],
            time,
            coord=coord,
        )
        return np.array([bx, by, bz])

    return field_func


def _field_unit(field_func: FieldModel, pos: np.ndarray) -> np.ndarray:
    """Unit vector of field at position, for RK4 intermediate steps."""
    B = field_func(pos)
    B_mag = np.linalg.norm(B)
    return B / B_mag if B_mag > 0 else np.zeros(3)
