"""Coordinate system conversions: Cartesian <-> spherical, rotation matrices,
unit vector utilities.

Consolidates cassinilib/Vector.py and wavesolver/linalg.py coordinate functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def unit_vector(v: ArrayLike) -> np.ndarray:
    """Normalize vector(s) to unit length. Handles arrays of shape (3,) or (N, 3)."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    # Avoid division by zero
    norm = np.where(norm > 0, norm, 1.0)
    return v / norm


def car2sph(xyz: ArrayLike) -> np.ndarray:
    """Cartesian (x, y, z) to spherical (r, theta, phi).

    theta: colatitude [0, pi], phi: azimuth [-pi, pi].
    Input/output shape: (3,) or (N, 3).
    """
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(rho, z)
    phi = np.arctan2(y, x)
    return np.stack([r, theta, phi], axis=-1)


def sph2car(rtp: ArrayLike) -> np.ndarray:
    """Spherical (r, theta, phi) to Cartesian (x, y, z).

    Input/output shape: (3,) or (N, 3).
    """
    rtp = np.asarray(rtp, dtype=float)
    r, theta, phi = rtp[..., 0], rtp[..., 1], rtp[..., 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def rotation_matrix_sph2car(theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """Rotation matrix from spherical (r, theta, phi) to Cartesian (x, y, z) basis.

    For a single point, returns shape (3, 3).
    For N points, returns shape (N, 3, 3).
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    z = np.zeros_like(theta)

    R = np.array(
        [
            [st * cp, ct * cp, -sp],
            [st * sp, ct * sp, cp],
            [ct, -st, z],
        ]
    )
    # Move the matrix indices to the end: shape (..., 3, 3)
    return np.moveaxis(R, [0, 1], [-2, -1])


def rotation_matrix_car2sph(theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """Rotation matrix from Cartesian (x, y, z) to spherical (r, theta, phi) basis.

    Inverse (transpose) of rotation_matrix_sph2car.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    z = np.zeros_like(theta)

    R = np.array(
        [
            [st * cp, st * sp, ct],
            [ct * cp, ct * sp, -st],
            [-sp, cp, z],
        ]
    )
    return np.moveaxis(R, [0, 1], [-2, -1])


def rotate_about_x(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, angle: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate (x, y, z) about the x-axis by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return (
        np.asarray(x),
        c * np.asarray(y) - s * np.asarray(z),
        s * np.asarray(y) + c * np.asarray(z),
    )


def rotate_about_y(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, angle: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate (x, y, z) about the y-axis by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return (
        c * np.asarray(x) + s * np.asarray(z),
        np.asarray(y),
        -s * np.asarray(x) + c * np.asarray(z),
    )


def phi_to_lt(phi: ArrayLike) -> np.ndarray:
    r"""Convert azimuthal angle (radians) to local time (hours).

    Convention: noon (12h) is at $\phi = 0$, midnight (0h) at $\phi = \pi$.

    Replaces ``cassinilib/Plot.py:phi2LT()``.

    Parameters
    ----------
    phi : array_like
        Azimuthal angle in radians.

    Returns
    -------
    ndarray
        Local time in hours [0, 24).
    """
    phi = np.asarray(phi, dtype=float)
    return (phi / np.pi) * 12 + 12


def lt_to_phi(lt: ArrayLike) -> np.ndarray:
    r"""Convert local time (hours) to azimuthal angle (radians).

    Convention: noon (12h) is at $\phi = 0$, midnight (0h) at $\phi = \pi$.

    Replaces ``cassinilib/Plot.py:LT2phi()``.

    Parameters
    ----------
    lt : array_like
        Local time in hours.

    Returns
    -------
    ndarray
        Azimuthal angle in radians.
    """
    lt = np.asarray(lt, dtype=float)
    return lt / 12 * np.pi


def lat_to_colat(lat: ArrayLike) -> np.ndarray:
    r"""Convert latitude (degrees) to colatitude (degrees).

    Parameters
    ----------
    lat : array_like
        Latitude in degrees [-90, 90].

    Returns
    -------
    ndarray
        Colatitude in degrees [0, 180].
    """
    return 90.0 - np.asarray(lat, dtype=float)


def colat_to_lat(colat: ArrayLike) -> np.ndarray:
    r"""Convert colatitude (degrees) to latitude (degrees).

    Parameters
    ----------
    colat : array_like
        Colatitude in degrees [0, 180].

    Returns
    -------
    ndarray
        Latitude in degrees [-90, 90].
    """
    return 90.0 - np.asarray(colat, dtype=float)
