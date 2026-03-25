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
    return np.degrees(np.arcsin(z_off / r))


def local_time(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Compute local time (hours) from KSM x, y coordinates.

    Noon (12 LT) is along +x (sunward), midnight (0 LT) along -x.
    """
    x, y = np.asarray(x), np.asarray(y)
    angle = np.arctan2(y, x)  # radians, 0 = sunward
    lt = 12.0 + angle * 12.0 / np.pi
    return lt % 24.0
