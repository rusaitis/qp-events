"""Saturn coordinate system transformations.

Implements the KROT rotation subroutine and KSun solar ephemeris from
the KMAG Fortran model (Khurana 2020). Supports transformations between:

- S3C: System III Cartesian (right-handed, rotating with Saturn)
- KSM: Kronian-Sun-Magnetic
- KSO: Kronian-Sun-Orbital
- DIP: Dipole Cartesian
- DIS: Dipole-Sun
"""

from __future__ import annotations

import math

import numpy as np

from qp.fieldline.kmag_coefficients import (
    CTIME_TO_J2000_OFFSET,
    DIPOLE_MATRIX,
    JTIME1,
    KSUN_AA,
    KSUN_BB,
    KSUN_CC,
    LEAP_SECOND_THRESHOLDS,
    OMEGA_ROT,
    OMEGA_Y,
    OMEGA_Z,
    YEAR_SECONDS,
    YR_SAT,
    ZTHETD,
)

_DEG2RAD = math.pi / 180.0


def epoch_to_j2000(time: float, epoch: str) -> float:
    """Convert time in a given epoch to J2000 seconds.

    Parameters
    ----------
    time : float
        Time value in the specified epoch.
    epoch : str
        One of 'j2000', 'i2000', 'ctime', 'etime'.

    Returns
    -------
    float
        Time in J2000 seconds (TDB).
    """
    epoch = epoch.lower().strip()
    if epoch == "j2000":
        return time
    if epoch == "i2000":
        return time + 32.184
    if epoch == "etime":
        return time
    if epoch == "ctime":
        # Leap second correction
        tcor = float(np.sum(time >= LEAP_SECOND_THRESHOLDS))
        return time + tcor + CTIME_TO_J2000_OFFSET
    raise ValueError(
        f"Unknown epoch: {epoch!r}. Use 'j2000', 'i2000', 'ctime', or 'etime'."
    )


def sun_position(j2000_time: float) -> tuple[float, float, float, float]:
    r"""Compute Sun and Zkso directions in System III coordinates.

    Implements the KSun subroutine from KMAG2012.f.

    Parameters
    ----------
    j2000_time : float
        Time in J2000 seconds.

    Returns
    -------
    stheta : float
        Solar colatitude in S3C (radians). Actually the Fortran computes
        latitude-like angle in degrees then converts.
    sphi : float
        Solar longitude in S3C (radians).
    ztheta : float
        Zkso colatitude in S3C (radians).
    zphi : float
        Zkso longitude in S3C (radians).
    """
    t = j2000_time - JTIME1

    # Build the 11-element basis vector
    omyt = OMEGA_Y * t
    t_over_yr = t / YEAR_SECONDS
    basis = np.array(
        [
            math.cos(omyt),
            math.sin(omyt),
            math.cos(2.0 * omyt),
            math.sin(2.0 * omyt),
            math.cos(3.0 * omyt),
            math.sin(3.0 * omyt),
            math.cos(4.0 * omyt),
            math.sin(4.0 * omyt),
            t_over_yr**2,
            t_over_yr,
            1.0,
        ]
    )

    # Solar latitude (degrees) and non-rotating longitude (degrees)
    stheta_deg = float(np.dot(KSUN_AA, basis))
    fphi_deg = float(np.dot(KSUN_BB, basis))
    zfphi_deg = float(np.dot(KSUN_CC, basis))

    # Rotate to System III
    # Add orbital motion, then subtract Saturn's spin
    fphi_deg = math.fmod(fphi_deg + t / YR_SAT * 360.0, 360.0)
    sphi_deg = math.fmod(fphi_deg - t * OMEGA_ROT, 360.0)
    if sphi_deg < 0.0:
        sphi_deg += 360.0

    # Zkso longitude
    zfphi_deg = math.fmod(zfphi_deg + t / YR_SAT * 360.0 + 180.0, 360.0)
    zphi_deg = math.fmod(zfphi_deg - t * OMEGA_Z, 360.0)
    if zphi_deg < 0.0:
        zphi_deg += 360.0

    stheta = stheta_deg * _DEG2RAD
    sphi = sphi_deg * _DEG2RAD
    ztheta = ZTHETD * _DEG2RAD
    zphi = zphi_deg * _DEG2RAD

    return stheta, sphi, ztheta, zphi


def _build_kso_axes_s3c(
    stheta: float, sphi: float, ztheta: float, zphi: float
) -> np.ndarray:
    """KSO axis unit vectors expressed in S3C. Returns 3x3: rows are X,Y,Z of KSO."""
    # X_KSO = sun direction in S3C
    x_kso = np.array(
        [
            math.cos(stheta) * math.cos(sphi),
            math.cos(stheta) * math.sin(sphi),
            math.sin(stheta),
        ]
    )
    # Z_KSO from Zkso angles
    z_kso = np.array(
        [
            math.cos(ztheta) * math.cos(zphi),
            math.cos(ztheta) * math.sin(zphi),
            math.sin(ztheta),
        ]
    )
    # Y_KSO = Z × X
    y_kso = np.cross(z_kso, x_kso)
    return np.array([x_kso, y_kso, z_kso])


def _build_ksm_axes_s3c(stheta: float, sphi: float) -> np.ndarray:
    """KSM axis unit vectors expressed in S3C. Returns 3x3: rows are X,Y,Z of KSM."""
    dipole_z = DIPOLE_MATRIX[2]  # [0, 0, 1] for Saturn

    # X_KSM = sun direction
    x_ksm = np.array(
        [
            math.cos(stheta) * math.cos(sphi),
            math.cos(stheta) * math.sin(sphi),
            math.sin(stheta),
        ]
    )
    # Y_KSM = dipole_Z × X_KSM (then normalize)
    # Fortran uses this cross-product ordering
    y_ksm = np.cross(dipole_z, x_ksm)
    y_ksm /= np.linalg.norm(y_ksm)
    # Z_KSM = X × Y
    z_ksm = np.cross(x_ksm, y_ksm)
    return np.array([x_ksm, y_ksm, z_ksm])


def _build_dis_axes_s3c(stheta: float, sphi: float) -> np.ndarray:
    """DIS axis unit vectors expressed in S3C. Returns 3x3: rows are X,Y,Z of DIS."""
    dipole_z = DIPOLE_MATRIX[2]  # [0, 0, 1]

    # Z_DIS = dipole direction
    z_dis = dipole_z.copy()

    # X_KSO (sun direction, used as intermediate)
    x_kso = np.array(
        [
            math.cos(stheta) * math.cos(sphi),
            math.cos(stheta) * math.sin(sphi),
            math.sin(stheta),
        ]
    )

    # Y_DIS = Z_DIS × X_KSO (normalized)
    # Fortran uses this cross-product ordering
    y_dis = np.cross(z_dis, x_kso)
    y_dis /= np.linalg.norm(y_dis)

    # X_DIS = Y_DIS × Z_DIS
    x_dis = np.cross(y_dis, z_dis)

    return np.array([x_dis, y_dis, z_dis])


def rotation_matrix_to_s3c(system: str, j2000_time: float) -> np.ndarray:
    """Build 3x3 matrix that rotates a vector FROM `system` TO S3C.

    Parameters
    ----------
    system : str
        One of 'S3C', 'KSM', 'KSO', 'DIP', 'DIS'.
    j2000_time : float
        Time in J2000 seconds.
    """
    system = system.upper()
    if system == "S3C":
        return np.eye(3)

    stheta, sphi, ztheta, zphi = sun_position(j2000_time)

    if system == "KSO":
        # Rows of axes_s3c are KSO axes in S3C coords.
        # axes_s3c takes S3C→KSO, so transpose takes KSO→S3C.
        return _build_kso_axes_s3c(stheta, sphi, ztheta, zphi).T

    if system == "KSM":
        return _build_ksm_axes_s3c(stheta, sphi).T

    if system == "DIP":
        # Transpose of dipole matrix (identity for Saturn)
        return DIPOLE_MATRIX.T

    if system == "DIS":
        # Fortran subtlety: DIS construction gives rows = DIS axes in S3C,
        # and the Fortran uses this matrix directly (NOT transposed) as
        # the "first" matrix when DIS is the FROM system.
        # first(i,j) = dummy(j,i) for KSM/KSO (transpose), but
        # first(i,j) = dummy(i,j) for DIS (no transpose in "from" path).
        #
        # Actually re-reading the Fortran more carefully:
        # For DIS as "from": first(i,j) = dummy(j,i) at line 742 — it IS transposed.
        # For DIS as "to": second(i,j) = dummy(i,j) at line 887 — NOT transposed.
        #
        # The axes matrix rows are DIS axes in S3C. So:
        # axes @ v_s3c = v_dis (takes S3C → DIS)
        # axes.T @ v_dis = v_s3c (takes DIS → S3C)
        return _build_dis_axes_s3c(stheta, sphi).T

    raise ValueError(f"Unknown coordinate system: {system!r}")


def rotation_matrix_from_s3c(system: str, j2000_time: float) -> np.ndarray:
    """Build 3x3 matrix that rotates a vector FROM S3C TO `system`."""
    # The inverse of to_s3c is from_s3c; for rotation matrices, inverse = transpose.
    return rotation_matrix_to_s3c(system, j2000_time).T


def rotate_vector(
    vec: np.ndarray,
    from_sys: str,
    to_sys: str,
    j2000_time: float,
) -> np.ndarray:
    """Transform a 3-vector between Saturn coordinate systems.

    Equivalent to Fortran KROT: from→S3C→to.

    Parameters
    ----------
    vec : ndarray, shape (3,)
        Input vector.
    from_sys, to_sys : str
        Coordinate systems ('S3C', 'KSM', 'KSO', 'DIP', 'DIS').
    j2000_time : float
        Time in J2000 seconds.
    """
    # Step 1: from → S3C
    first = rotation_matrix_to_s3c(from_sys, j2000_time)
    v_s3c = first @ vec
    # Step 2: S3C → to
    second = rotation_matrix_from_s3c(to_sys, j2000_time)
    return second @ v_s3c


def car2sph_field(
    bx: float,
    by: float,
    bz: float,
    theta: float,
    phi: float,
) -> tuple[float, float, float]:
    r"""Cartesian to spherical field components (Arfken convention).

    $$B_r = B_x \sin\theta \cos\phi + B_y \sin\theta \sin\phi + B_z \cos\theta$$
    """
    st, ct = math.sin(theta), math.cos(theta)
    sp, cp = math.sin(phi), math.cos(phi)
    br = bx * st * cp + by * st * sp + bz * ct
    bth = bx * ct * cp + by * ct * sp - bz * st
    bph = -bx * sp + by * cp
    return br, bth, bph


def sph2car_field(
    br: float,
    bth: float,
    bph: float,
    theta: float,
    phi: float,
) -> tuple[float, float, float]:
    """Spherical to Cartesian field components (inverse of car2sph_field)."""
    st, ct = math.sin(theta), math.cos(theta)
    sp, cp = math.sin(phi), math.cos(phi)
    bx = br * st * cp + bth * ct * cp - bph * sp
    by = br * st * sp + bth * ct * sp + bph * cp
    bz = br * ct - bth * st
    return bx, by, bz
