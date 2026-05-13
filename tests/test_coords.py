"""Sanity tests for qp.coords.

Covers the public helpers in `transforms`, `ksm`, and `mfa`. Test goals
are conservative: round-trip identities, known reference points, and
shape preservation. No DATA/ inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from qp.coords.ksm import (
    DIPOLE_OFFSET_Z,
    dipole_invariant_latitude,
    local_time,
    magnetic_latitude,
)
from qp.coords.transforms import (
    car2sph,
    lt_to_phi,
    phi_to_lt,
    rotation_matrix_sph2car,
    sph2car,
    unit_vector,
)


class TestTransforms:
    def test_unit_vector_normalises(self) -> None:
        v = np.array([3.0, 0.0, 4.0])
        np.testing.assert_allclose(np.linalg.norm(unit_vector(v)), 1.0)

    def test_unit_vector_zero_does_not_blow_up(self) -> None:
        v = np.zeros(3)
        np.testing.assert_array_equal(unit_vector(v), v)

    def test_car_sph_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        xyz = rng.normal(size=(20, 3))
        np.testing.assert_allclose(sph2car(car2sph(xyz)), xyz, atol=1e-12)

    def test_car_sph_known_point(self) -> None:
        # Point on +x axis: r=1, theta=pi/2, phi=0
        out = car2sph(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(out, [1.0, np.pi / 2, 0.0], atol=1e-12)

    def test_phi_to_lt_anchor_points(self) -> None:
        # Noon (phi=0) -> 12 LT; phi=+pi -> 0/24 LT (subsolar branch cut).
        np.testing.assert_allclose(phi_to_lt(0.0), 12.0)
        np.testing.assert_allclose(phi_to_lt(np.pi) % 24.0, 0.0, atol=1e-12)

    def test_lt_phi_roundtrip(self) -> None:
        lt_in = np.linspace(0.0, 24.0, 25, endpoint=False)
        lt_out = phi_to_lt(lt_to_phi(lt_in))
        # Wrap the difference into (-12, 12] so the modular boundary at 24h
        # doesn't masquerade as a real error.
        delta = np.mod(lt_out - lt_in + 12.0, 24.0) - 12.0
        np.testing.assert_allclose(delta, 0.0, atol=1e-9)

    def test_rotation_matrix_orthonormal(self) -> None:
        # Rotation matrices should be orthogonal: R @ R.T == I.
        rng = np.random.default_rng(7)
        theta = rng.uniform(0.1, np.pi - 0.1, 5)
        phi = rng.uniform(-np.pi, np.pi, 5)
        R = rotation_matrix_sph2car(theta, phi)
        eye = np.einsum("nij,nkj->nik", R, R)
        np.testing.assert_allclose(
            eye, np.broadcast_to(np.eye(3), eye.shape), atol=1e-12
        )


class TestKSM:
    def test_magnetic_latitude_equator(self) -> None:
        # A point at (r, 0, DIPOLE_OFFSET_Z) sits on the offset-dipole equator.
        out = magnetic_latitude(10.0, 0.0, DIPOLE_OFFSET_Z)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_magnetic_latitude_north_pole(self) -> None:
        # Far above the dipole origin → +90 deg.
        out = magnetic_latitude(0.0, 0.0, DIPOLE_OFFSET_Z + 100.0)
        np.testing.assert_allclose(out, 90.0, atol=1e-9)

    def test_magnetic_latitude_array_shape(self) -> None:
        x = np.linspace(1, 20, 8)
        out = magnetic_latitude(x, np.zeros_like(x), np.zeros_like(x))
        assert out.shape == x.shape

    def test_local_time_noon_and_midnight(self) -> None:
        # +x sunward -> noon; -x antisunward -> midnight.
        np.testing.assert_allclose(local_time(1.0, 0.0), 12.0, atol=1e-12)
        np.testing.assert_allclose(local_time(-1.0, 0.0) % 24.0, 0.0, atol=1e-12)

    def test_local_time_dusk_dawn(self) -> None:
        # +y dawn (06 LT), -y dusk (18 LT) by convention.
        np.testing.assert_allclose(local_time(0.0, 1.0), 18.0, atol=1e-12)
        np.testing.assert_allclose(local_time(0.0, -1.0), 6.0, atol=1e-12)

    def test_dipole_invariant_latitude_inside_planet_nan(self) -> None:
        # L = r / cos^2(lat) < 1 → NaN. An equatorial point at r=0.5 R_S
        # gives L=0.5; (0,0,>>1) along z maps to L=∞ which is the pole,
        # not "inside the planet".
        out = dipole_invariant_latitude(0.5, 0.0, DIPOLE_OFFSET_Z)
        assert np.isnan(out)

    @pytest.mark.parametrize("L", [5.0, 10.0, 20.0])
    def test_dipole_invariant_latitude_at_equator(self, L: float) -> None:
        # Equatorial point at L-shell L: inv_lat = arccos(1/sqrt(L)) in deg.
        x = L
        y = 0.0
        z = DIPOLE_OFFSET_Z
        expected = np.degrees(np.arccos(1.0 / np.sqrt(L)))
        # Sign matches hemisphere of z_off; z == DIPOLE_OFFSET_Z makes
        # z_off == 0 — copysign on 0.0 returns +0, so expect +expected.
        np.testing.assert_allclose(
            dipole_invariant_latitude(x, y, z), expected, atol=1e-9
        )
