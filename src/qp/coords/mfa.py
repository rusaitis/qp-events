"""Mean-Field-Aligned (MFA) coordinate transformation.

Implements the coordinate system from Rusaitis et al. (Equations 1-3):

    b_parallel = <B> / |<B>|                        (Eq 1)
    b_perp1    = (phi_hat x b_parallel) / |...|     (Eq 2)
    b_perp2    = b_parallel x b_perp1               (Eq 3)

where phi_hat is the azimuthal unit vector in the Kronographic equatorial
plane (z_hat x R_hat), positive in the direction of corotation.

Consolidates three redundant functions from cassinilib/ToMagneticCoords.py
into one vectorized implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from qp.coords.transforms import (
    unit_vector,
    rotation_matrix_sph2car,
)


def to_mfa(
    position: ArrayLike,
    field: ArrayLike,
    background_field: ArrayLike,
    coords: str = "KSM",
) -> np.ndarray:
    """Transform magnetic field perturbations into MFA coordinates.

    Parameters
    ----------
    position : array_like, shape (3,) or (N, 3)
        Spacecraft position vectors.
    field : array_like, shape (3,) or (N, 3)
        Instantaneous magnetic field (with perturbations).
    background_field : array_like, shape (3,) or (N, 3)
        Background (running-averaged) magnetic field <B>.
    coords : str
        Input coordinate system: 'KSM', 'KSO', or 'KRTP'.

    Returns
    -------
    mfa : ndarray, shape (3,) or (N, 3)
        Field components [b_parallel, b_perp1, b_perp2] in MFA coordinates.
    """
    pos = np.asarray(position, dtype=float)
    B = np.asarray(field, dtype=float)
    B_avg = np.asarray(background_field, dtype=float)

    single = pos.ndim == 1
    if single:
        pos = pos[np.newaxis, :]
        B = B[np.newaxis, :]
        B_avg = B_avg[np.newaxis, :]

    # If input is spherical (KRTP), convert field vectors to Cartesian
    if coords.upper() in ("KRTP", "RTN"):
        B, B_avg, pos = _sph_to_car_fields(pos, B, B_avg)

    # Compute azimuthal unit vector: phi_hat = z_hat x R_hat (positive corotation)
    # In Cartesian: phi_hat = (-y, x, 0) / |(-y, x, 0)|
    phi = np.stack([-pos[:, 1], pos[:, 0], np.zeros(len(pos))], axis=-1)
    phi_hat = unit_vector(phi)

    # MFA basis vectors (Eq 1-3)
    b_par_hat = unit_vector(B_avg)  # Eq 1
    b_perp1_hat = unit_vector(np.cross(phi_hat, b_par_hat))  # Eq 2
    b_perp2_hat = np.cross(b_par_hat, b_perp1_hat)  # Eq 3 (already unit)

    # Project field onto MFA basis
    b_par = np.einsum("...i,...i->...", B, b_par_hat)
    b_perp1 = np.einsum("...i,...i->...", B, b_perp1_hat)
    b_perp2 = np.einsum("...i,...i->...", B, b_perp2_hat)

    result = np.stack([b_par, b_perp1, b_perp2], axis=-1)
    return result[0] if single else result


def to_mfa_basis(
    position: ArrayLike,
    background_field: ArrayLike,
    coords: str = "KSM",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the MFA basis vectors without projecting a field.

    Returns (b_par_hat, b_perp1_hat, b_perp2_hat), each shape (3,) or (N, 3).
    """
    pos = np.asarray(position, dtype=float)
    B_avg = np.asarray(background_field, dtype=float)

    single = pos.ndim == 1
    if single:
        pos = pos[np.newaxis, :]
        B_avg = B_avg[np.newaxis, :]

    if coords.upper() in ("KRTP", "RTN"):
        _, B_avg, pos = _sph_to_car_fields(pos, B_avg, B_avg)

    phi = np.stack([-pos[:, 1], pos[:, 0], np.zeros(len(pos))], axis=-1)
    phi_hat = unit_vector(phi)

    b_par_hat = unit_vector(B_avg)
    b_perp1_hat = unit_vector(np.cross(phi_hat, b_par_hat))
    b_perp2_hat = np.cross(b_par_hat, b_perp1_hat)

    if single:
        return b_par_hat[0], b_perp1_hat[0], b_perp2_hat[0]
    return b_par_hat, b_perp1_hat, b_perp2_hat


def _sph_to_car_fields(
    pos_sph: np.ndarray,
    B_sph: np.ndarray,
    B_avg_sph: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical (KRTP) position and field vectors to Cartesian.

    pos_sph columns: [r, theta, phi] where theta=colatitude, phi=azimuth.
    """
    theta = pos_sph[:, 1]
    phi = pos_sph[:, 2]

    # Rotation matrices: (N, 3, 3)
    R = rotation_matrix_sph2car(theta, phi)

    # Rotate field vectors
    B_car = np.einsum("...ij,...j->...i", R, B_sph)
    B_avg_car = np.einsum("...ij,...j->...i", R, B_avg_sph)

    # Convert position to Cartesian
    from qp.coords.transforms import sph2car

    pos_car = sph2car(pos_sph)

    return B_car, B_avg_car, pos_car
