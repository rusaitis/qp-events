r"""Stokes parameters and degree of polarization for two-component waves.

For two complex analytic-signal time series :math:`z_1(t), z_2(t)` —
typically the Morlet-CWT coefficients of the two transverse magnetic
field components at a single frequency — the four Stokes parameters
are

.. math::

    I &= \langle |z_1|^2 + |z_2|^2 \rangle \\
    Q &= \langle |z_1|^2 - |z_2|^2 \rangle \\
    U &= 2 \langle \mathrm{Re}(z_1 z_2^*) \rangle \\
    V &= 2 \langle \mathrm{Im}(z_1 z_2^*) \rangle

and the degree of polarization is

.. math::

    d = \frac{\sqrt{Q^2 + U^2 + V^2}}{I} \in [0, 1].

A coherent monochromatic wave packet has :math:`d = 1` regardless of
whether the polarization is linear (``Q``- or ``U``-dominated),
circular (``V``-dominated), or elliptical. An incoherent superposition
of independent transverse fluctuations has :math:`d \to 0` as the
averaging window grows: the cross-terms :math:`\langle z_1 z_2^* \rangle`
vanish in expectation. Thus a single 0–1 score cleanly separates
genuinely polarized waves from broadband transients, with no need for
separate "linear" and "circular" branches.

References
----------
Born & Wolf, *Principles of Optics*, 7th ed., §1.4.2.

Samson, J. C. (1973), "Description of the polarisation states of vector
processes", *Geophys. J. R. Astr. Soc.* 34, 403.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def stokes_parameters(
    z1: ArrayLike,
    z2: ArrayLike,
) -> tuple[float, float, float, float]:
    r"""Stokes parameters of a two-component analytic signal.

    Parameters
    ----------
    z1, z2 : array_like, complex
        Analytic-signal samples of the two transverse components.
        Typically a slice of a Morlet CWT at a single frequency row,
        but any pair of complex 1-D arrays works.

    Returns
    -------
    I, Q, U, V : float
        Stokes parameters averaged over all input samples. Units are
        the same as :math:`|z|^2`.
    """
    z1 = np.asarray(z1, dtype=complex)
    z2 = np.asarray(z2, dtype=complex)
    if z1.shape != z2.shape:
        raise ValueError(
            f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}"
        )
    p1 = np.abs(z1) ** 2
    p2 = np.abs(z2) ** 2
    cross = z1 * z2.conj()
    return (
        float(np.mean(p1 + p2)),
        float(np.mean(p1 - p2)),
        float(2.0 * np.mean(np.real(cross))),
        float(2.0 * np.mean(np.imag(cross))),
    )


def degree_of_polarization(
    z1: ArrayLike,
    z2: ArrayLike,
) -> float:
    r"""Degree of polarization :math:`d \in [0, 1]`.

    .. math::

        d = \frac{\sqrt{Q^2 + U^2 + V^2}}{I}

    A coherent wave packet has :math:`d \to 1` regardless of
    polarization geometry; incoherent broadband transients have
    :math:`d \to 0`. Returns ``0.0`` if the total power :math:`I` is
    not positive.
    """
    i_, q, u, v = stokes_parameters(z1, z2)
    if i_ <= 0.0:
        return 0.0
    return float(np.sqrt(q * q + u * u + v * v) / i_)


def mva_intermediate_minimum_ratio(
    field: ArrayLike,
) -> float:
    r"""Intermediate-to-minimum eigenvalue ratio from minimum variance analysis.

    Minimum variance analysis (Sonnerup & Cahill 1967; Sonnerup &
    Scheible 1998) diagonalises the field covariance matrix
    :math:`M_{ij} = \langle B_i B_j \rangle - \langle B_i \rangle
    \langle B_j \rangle` and identifies the principal axes of the
    perturbation. The eigenvalue ordering is
    :math:`\lambda_1 \geq \lambda_2 \geq \lambda_3`; the minimum
    variance direction (eigenvector of :math:`\lambda_3`) is the wave
    normal for a planar transverse perturbation.

    The ratio :math:`\lambda_2 / \lambda_3` measures how well-defined
    that minimum-variance direction is. A planar perturbation (e.g.
    an Alfvén wave with :math:`\delta B \perp B_0`) has
    :math:`\lambda_3 \to 0` so the ratio diverges; an isotropic
    transient (FGM step affecting all three axes) has all eigenvalues
    similar and the ratio approaches 1.

    Parameters
    ----------
    field : array_like, shape (n_samples, 3)
        Three-component magnetic-field time series — typically
        ``(b_par, b_perp1, b_perp2)`` in MFA frame.

    Returns
    -------
    ratio : float
        :math:`\lambda_2 / \lambda_3`. Returns ``+inf`` when
        :math:`\lambda_3 \le 0` (degenerate planar perturbation),
        ``0.0`` for an empty or zero-variance input.
    """
    field = np.asarray(field, dtype=float)
    if field.ndim != 2 or field.shape[1] != 3:
        raise ValueError(
            f"field must have shape (n_samples, 3), got {field.shape}"
        )
    if field.shape[0] < 3:
        return 0.0
    cov = np.cov(field, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)  # ascending: [lambda_3, lambda_2, lambda_1]
    if not np.all(np.isfinite(eigvals)) or eigvals[2] <= 0:
        return 0.0
    if eigvals[0] <= 0:
        return float("inf")
    return float(eigvals[1] / eigvals[0])


def mva_major_axis_parallel_fraction(
    field: ArrayLike,
    par_axis: int = 0,
) -> float:
    r"""Squared projection of the MVA major axis onto the background-field axis.

    The maximum-variance eigenvector :math:`\hat e_1` of the
    covariance matrix points along the wave's largest perturbation
    axis. For a transverse Alfvén wave :math:`\delta\mathbf{B} \perp
    \mathbf{B}_0`, so :math:`\hat e_1` lies in the perpendicular
    plane and :math:`(\hat e_1 \cdot \hat b_\parallel)^2 \to 0`. For
    a compressional perturbation :math:`\delta\mathbf{B} \parallel
    \mathbf{B}_0`, so :math:`(\hat e_1 \cdot \hat b_\parallel)^2 \to
    1`. Combined with :math:`\lambda_2/\lambda_3 \geq 5` the pair
    test "planar AND transverse" is the textbook MVA Alfvén
    signature (Sonnerup & Cahill 1967, Sonnerup & Scheible 1998 §8).

    Parameters
    ----------
    field : array_like, shape (n_samples, 3)
        Three-component magnetic-field time series in the MFA frame
        (or equivalent), with the background-field component in
        column ``par_axis``.
    par_axis : int, default 0
        Which column corresponds to :math:`b_\parallel`. The other
        two are taken as transverse.

    Returns
    -------
    fraction : float
        :math:`(\hat e_1 \cdot \hat e_{\text{par}})^2 \in [0, 1]`.
        Returns ``0.0`` for inputs with fewer than three samples or
        zero covariance (no preferred direction).
    """
    field = np.asarray(field, dtype=float)
    if field.ndim != 2 or field.shape[1] != 3:
        raise ValueError(
            f"field must have shape (n_samples, 3), got {field.shape}"
        )
    if not 0 <= par_axis < 3:
        raise ValueError(f"par_axis must be 0, 1, or 2; got {par_axis}")
    if field.shape[0] < 3:
        return 0.0
    cov = np.cov(field, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    if eigvals[2] <= 0:
        return 0.0
    e_max = eigvecs[:, 2]  # eigenvector of the largest eigenvalue
    return float(e_max[par_axis] ** 2)
