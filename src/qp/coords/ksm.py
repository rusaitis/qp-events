"""Kronocentric Solar Magnetospheric (KSM) coordinate system definitions.

KSM coordinates:
    X: points from Saturn toward the Sun
    Z: X-Z plane contains Saturn's centered magnetic dipole axis
    Y: completes right-handed system

The origin is offset by 0.037 R_S along Saturn's rotational axis
from Saturn's center of mass.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# Saturn's dipole axis offset from center of mass (R_S)
DIPOLE_OFFSET_Z = 0.037

# Saturn's axis tilt in KSM (approximate, for reference)
SATURN_AXIS_TILT_X = np.deg2rad(-26.7)


def magnetic_latitude(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
    """Compute magnetic latitude in KSM coordinates.

    Uses the offset dipole: origin shifted by DIPOLE_OFFSET_Z along z.
    Returns latitude in degrees.
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    # Offset to dipole origin
    z_off = z - DIPOLE_OFFSET_Z
    r = np.sqrt(x**2 + y**2 + z_off**2)
    return np.where(r > 0, np.degrees(np.arcsin(np.clip(z_off / r, -1, 1))), 0.0)


def dipole_invariant_latitude(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
    r"""Dipole invariant latitude (footpoint latitude) for the offset dipole.

    For a dipole field line passing through position $(r, \lambda)$, the
    L-shell is $L = r / \cos^2(\lambda)$ and the invariant latitude is
    $\lambda_{\mathrm{inv}} = \arccos(1 / \sqrt{L})$.

    Parameters
    ----------
    x, y, z : array_like
        Position in KSM coordinates ($R_S$).

    Returns
    -------
    ndarray
        Invariant latitude in degrees (signed by hemisphere).
        NaN for points inside the planet ($L < 1$).
    """
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    z_off = z - DIPOLE_OFFSET_Z
    r = np.sqrt(x**2 + y**2 + z_off**2)

    # Magnetic latitude from offset dipole
    lat_rad = np.where(r > 0, np.arcsin(np.clip(z_off / r, -1, 1)), 0.0)
    cos_lat = np.cos(lat_rad)

    # L-shell: L = r / cos²(λ)
    L = np.where(cos_lat != 0, r / cos_lat**2, np.inf)

    # Invariant latitude: cos²(λ_inv) = 1/L
    inv_lat_rad = np.where(L >= 1.0, np.arccos(np.clip(1.0 / np.sqrt(L), 0, 1)), np.nan)

    # Sign matches hemisphere of spacecraft
    return np.degrees(np.copysign(inv_lat_rad, z_off))


def local_time(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Compute local time (hours) from KSM x, y coordinates.

    Noon (12 LT) is along +x (sunward), midnight (0 LT) along -x.
    """
    x, y = np.asarray(x), np.asarray(y)
    angle = np.arctan2(y, x)  # radians, 0 = sunward
    lt = 12.0 + angle * 12.0 / np.pi
    return lt % 24.0
