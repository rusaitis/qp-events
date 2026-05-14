r"""Tests for qp.coords.mfa — MFA coordinate transform (Eq. 1-3).

The MFA basis at position $(R, 0, 0)$ with $\mathbf{B}_{avg} = B_0 \hat{z}$ is:

    b_par_hat   = (0, 0, 1)   (along B)
    b_perp1_hat = (1, 0, 0)   (radial — phi_hat × b_par_hat)
    b_perp2_hat = (0, 1, 0)   (azimuthal — b_par_hat × b_perp1_hat)

so radial perturbations show up in `b_perp1` and azimuthal ones in `b_perp2`.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from qp.coords.mfa import to_mfa, to_mfa_basis


class TestUniformFieldLimit:
    def test_par_recovers_field_magnitude(self):
        """B aligned with B_avg → all magnitude in b_par, perps zero."""
        pos = np.array([5.0, 0.0, 0.0])
        B = np.array([0.0, 0.0, 7.5])
        B_avg = np.array([0.0, 0.0, 7.5])
        b_par, b_perp1, b_perp2 = to_mfa(pos, B, B_avg)
        assert_allclose(b_par, 7.5, atol=1e-12)
        assert_allclose(b_perp1, 0.0, atol=1e-12)
        assert_allclose(b_perp2, 0.0, atol=1e-12)

    def test_radial_perturbation_goes_to_perp1(self):
        r"""$\delta\mathbf{B} = \delta B \hat{x}$ at $(R,0,0)$ → b_perp1."""
        pos = np.array([5.0, 0.0, 0.0])
        B_avg = np.array([0.0, 0.0, 10.0])
        B = B_avg + np.array([2.0, 0.0, 0.0])
        b_par, b_perp1, b_perp2 = to_mfa(pos, B, B_avg)
        assert_allclose(b_par, 10.0, atol=1e-12)
        assert_allclose(b_perp1, 2.0, atol=1e-12)
        assert_allclose(b_perp2, 0.0, atol=1e-12)

    def test_azimuthal_perturbation_goes_to_perp2(self):
        r"""$\delta\mathbf{B} = \delta B \hat{y} = \delta B \hat{\phi}$ at $(R,0,0)$ → b_perp2."""
        pos = np.array([5.0, 0.0, 0.0])
        B_avg = np.array([0.0, 0.0, 10.0])
        B = B_avg + np.array([0.0, 3.0, 0.0])
        b_par, b_perp1, b_perp2 = to_mfa(pos, B, B_avg)
        assert_allclose(b_par, 10.0, atol=1e-12)
        assert_allclose(b_perp1, 0.0, atol=1e-12)
        assert_allclose(b_perp2, 3.0, atol=1e-12)


class TestBasisProperties:
    def test_basis_orthonormal_at_arbitrary_position(self):
        pos = np.array([4.0, 3.0, 2.0])
        B_avg = np.array([0.5, -0.2, 8.0])
        e_par, e_perp1, e_perp2 = to_mfa_basis(pos, B_avg)
        # unit length
        assert_allclose(np.linalg.norm(e_par), 1.0, atol=1e-12)
        assert_allclose(np.linalg.norm(e_perp1), 1.0, atol=1e-12)
        assert_allclose(np.linalg.norm(e_perp2), 1.0, atol=1e-12)
        # mutually orthogonal
        assert_allclose(np.dot(e_par, e_perp1), 0.0, atol=1e-12)
        assert_allclose(np.dot(e_par, e_perp2), 0.0, atol=1e-12)
        assert_allclose(np.dot(e_perp1, e_perp2), 0.0, atol=1e-12)
        # right-handed: e_par × e_perp1 = e_perp2
        assert_allclose(np.cross(e_par, e_perp1), e_perp2, atol=1e-12)

    def test_perp1_lies_in_meridional_plane(self):
        r"""Since perp1 = phi_hat × b_par_hat and phi_hat is in the equatorial plane,
        perp1 must be perpendicular to phi_hat (lives in the meridional plane).
        """
        pos = np.array([5.0, 5.0, 1.0])
        B_avg = np.array([0.1, 0.1, 9.0])
        _, e_perp1, _ = to_mfa_basis(pos, B_avg)
        # phi_hat at this position
        phi_hat = np.array([-pos[1], pos[0], 0.0])
        phi_hat = phi_hat / np.linalg.norm(phi_hat)
        assert_allclose(np.dot(e_perp1, phi_hat), 0.0, atol=1e-12)


class TestVectorization:
    def test_vectorized_matches_per_row(self):
        rng = np.random.default_rng(7)
        n = 20
        positions = rng.normal(0.0, 5.0, (n, 3))
        # avoid degenerate r=0 cases
        positions[:, 0] += 6.0
        B = rng.normal(0.0, 5.0, (n, 3))
        # ensure non-trivial background field
        B_avg = B + np.array([0.0, 0.0, 10.0])

        batched = to_mfa(positions, B, B_avg)
        per_row = np.stack(
            [to_mfa(positions[i], B[i], B_avg[i]) for i in range(n)],
            axis=0,
        )
        assert_allclose(batched, per_row, atol=1e-12)


class TestKrtpRoundTrip:
    def test_krtp_input_matches_ksm_input(self):
        r"""Feeding the same physical state in KSM and in KRTP coords must yield
        the same MFA components — KRTP is just a rotation of the basis at $\mathbf{r}$.
        """
        # KSM position and field
        pos_ksm = np.array([6.0, 0.0, 0.0])
        B_avg_ksm = np.array([0.0, 0.0, 12.0])
        # a transverse perturbation
        B_ksm = B_avg_ksm + np.array([1.5, -2.5, 0.0])

        # KRTP equivalent: (r, theta, phi)
        from qp.coords.transforms import car2sph, rotation_matrix_sph2car

        rtp = car2sph(pos_ksm)
        R_sph2car = rotation_matrix_sph2car(rtp[1], rtp[2])
        # invert: car -> sph fields
        R_car2sph = np.linalg.inv(R_sph2car)
        B_avg_krtp = R_car2sph @ B_avg_ksm
        B_krtp = R_car2sph @ B_ksm

        mfa_ksm = to_mfa(pos_ksm, B_ksm, B_avg_ksm, coords="KSM")
        mfa_krtp = to_mfa(rtp, B_krtp, B_avg_krtp, coords="KRTP")
        assert_allclose(mfa_ksm, mfa_krtp, atol=1e-10)
