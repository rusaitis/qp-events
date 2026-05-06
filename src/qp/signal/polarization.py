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
