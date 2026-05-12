r"""Stokes parameters, ellipticity, inclination, and MVA for transverse waves.

Single source of truth for polarization math used by the detector
(:mod:`qp.events.detector`) and post-hoc figure scripts (Fig 10,
fig11 PPO fold, supplementary diagnostics).

Stokes parameters of an analytic two-component signal
:math:`z_1(t), z_2(t)` — typically the Morlet-CWT coefficients of the
two transverse magnetic-field components at a single frequency — are

.. math::

    I &= \langle |z_1|^2 + |z_2|^2 \rangle \\
    Q &= \langle |z_1|^2 - |z_2|^2 \rangle \\
    U &= 2 \,\mathrm{Re}\langle z_1 z_2^* \rangle \\
    V &= 2 \,\mathrm{Im}\langle z_1 z_2^* \rangle

with the Samson (1973) / Born & Wolf §1.4.2 sign convention: positive
:math:`V` means :math:`z_2` lags :math:`z_1` by :math:`\pi/2` (right-handed
in the :math:`(\hat b_{\perp 1}, \hat b_{\perp 2})` plane viewed along
:math:`\hat b_\parallel`). The degree of polarization

.. math::

    d = \frac{\sqrt{Q^2 + U^2 + V^2}}{I} \in [0, 1]

is 1 for a coherent monochromatic wave packet — linear, circular,
elliptical alike — and tends to 0 for an incoherent superposition.

Derived shape parameters: the ellipticity :math:`e = \tan\chi` with
:math:`\sin 2\chi = V / p`, the inclination
:math:`\psi = \tfrac{1}{2}\mathrm{atan2}(U, Q)`, and the polarized
fraction :math:`p / I`.

References
----------
Born & Wolf, *Principles of Optics*, 7th ed., §1.4.2.
Samson, J. C. (1973), "Description of the polarisation states of vector
processes", *Geophys. J. R. Astr. Soc.* 34, 403.
Sonnerup & Cahill (1967), *JGR* 72, 171. Sonnerup & Scheible (1998),
*ISSI Scientific Reports* SR-001 §8 — MVA helpers below.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import hilbert, windows

from qp.signal.polarization_config import TUKEY_TAPER_ALPHA


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


def stokes_parameters_real(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
) -> tuple[float, float, float, float]:
    r"""Stokes parameters from real time-series via the Hilbert transform.

    Convenience wrapper around :func:`stokes_parameters` that builds the
    analytic signal of each input with ``scipy.signal.hilbert``. Use
    this when you have raw band-pass-filtered time series rather than
    complex CWT coefficients.
    """
    a = hilbert(np.asarray(b_perp1, dtype=float))
    b = hilbert(np.asarray(b_perp2, dtype=float))
    return stokes_parameters(a, b)


def stokes_parameters_tapered(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
    alpha: float = TUKEY_TAPER_ALPHA,
) -> tuple[float, float, float, float]:
    r"""Stokes parameters with a Tukey taper to suppress Hilbert edge effects.

    The Hilbert transform of a short rectangular window leaks energy at
    the boundaries; a Tukey taper smoothly rolls the signal to zero at
    both ends and removes the artifact without distorting the central
    portion. ``alpha=0.25`` tapers the outer 12.5% of each side.
    """
    b1 = np.asarray(b_perp1, dtype=float)
    b2 = np.asarray(b_perp2, dtype=float)
    taper = windows.tukey(len(b1), alpha=alpha)
    a = hilbert(b1 * taper)
    b = hilbert(b2 * taper)
    return stokes_parameters(a, b)


def degree_of_polarization(
    z1: ArrayLike,
    z2: ArrayLike,
) -> float:
    r"""Degree of polarization :math:`d \in [0, 1]`.

    A coherent wave packet has :math:`d \to 1` regardless of
    polarization geometry; incoherent broadband transients have
    :math:`d \to 0`. Returns ``0.0`` if the total power :math:`I` is
    not positive.
    """
    i_, q, u, v = stokes_parameters(z1, z2)
    if i_ <= 0.0:
        return 0.0
    return float(np.sqrt(q * q + u * u + v * v) / i_)


def ellipticity_inclination_from_stokes(
    i_: float,
    q: float,
    u: float,
    v: float,
) -> tuple[float, float, float]:
    r"""Derive (ellipticity, inclination°, polarized fraction) from Stokes.

    Returns
    -------
    ellipticity : float
        Signed minor/major axis ratio :math:`\tan\chi` with
        :math:`\sin 2\chi = V/p`. ``+1`` is right-circular, ``-1``
        left-circular, ``0`` linear.
    inclination_deg : float
        Major-axis tilt from :math:`\hat b_{\perp 1}`,
        :math:`\tfrac{1}{2}\,\mathrm{atan2}(U, Q)`, in degrees.
    polarized_fraction : float
        :math:`p/I \in [0, 1]`, identical to :func:`degree_of_polarization`.
    """
    p = np.sqrt(q * q + u * u + v * v)
    if p <= 0.0 or i_ <= 0.0:
        return 0.0, 0.0, 0.0
    chi = 0.5 * np.arcsin(np.clip(v / p, -1.0, 1.0))
    psi = 0.5 * np.arctan2(u, q)
    return float(np.tan(chi)), float(np.degrees(psi)), float(p / i_)


def ellipticity_inclination(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
) -> tuple[float, float, float]:
    r"""Ellipticity, inclination, polarized fraction from real time-series."""
    return ellipticity_inclination_from_stokes(
        *stokes_parameters_real(b_perp1, b_perp2),
    )


def ellipticity_inclination_tapered(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
    alpha: float = TUKEY_TAPER_ALPHA,
) -> tuple[float, float, float]:
    r"""Ellipticity from real time-series with a Tukey taper.

    Preferred over :func:`ellipticity_inclination` for events spanning
    fewer than ~5 oscillations.
    """
    return ellipticity_inclination_from_stokes(
        *stokes_parameters_tapered(b_perp1, b_perp2, alpha=alpha),
    )


def per_oscillation_ellipticity(
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
    dt: float = 60.0,
    period: float = 3600.0,
    alpha: float = TUKEY_TAPER_ALPHA,
) -> tuple[float, float]:
    r"""Median per-oscillation ellipticity and its inter-quartile spread.

    Splits the event window into individual cycles of the peak period,
    computes Stokes-derived ellipticity for each cycle, and returns the
    median and IQR. Vectorized over cycles — reshape + axis-batched
    Hilbert in two FFT calls — so a 100-cycle event runs in O(N log N).

    Returns
    -------
    median_ellipticity : float
    iqr_ellipticity : float
        Inter-quartile range (75th - 25th percentile).
    """
    b1 = np.asarray(b_perp1, dtype=float)
    b2 = np.asarray(b_perp2, dtype=float)
    samples_per_cycle = int(round(period / dt))
    if samples_per_cycle < 4 or len(b1) < samples_per_cycle:
        e, _, _ = ellipticity_inclination_tapered(b1, b2, alpha=alpha)
        return e, 0.0

    n_cycles = len(b1) // samples_per_cycle
    cut = n_cycles * samples_per_cycle
    cycles_1 = b1[:cut].reshape(n_cycles, samples_per_cycle)
    cycles_2 = b2[:cut].reshape(n_cycles, samples_per_cycle)
    taper = windows.tukey(samples_per_cycle, alpha=alpha)
    a = hilbert(cycles_1 * taper, axis=-1)
    b = hilbert(cycles_2 * taper, axis=-1)
    # Per-cycle Stokes — broadcast over the cycle axis.
    p1 = np.abs(a) ** 2
    p2 = np.abs(b) ** 2
    cross = a * b.conj()
    i_ = (p1 + p2).mean(axis=-1)
    q = (p1 - p2).mean(axis=-1)
    u = 2.0 * cross.real.mean(axis=-1)
    v = 2.0 * cross.imag.mean(axis=-1)
    p = np.sqrt(q * q + u * u + v * v)
    valid = (i_ > 0.0) & (p > 0.0)
    if not np.any(valid):
        e, _, _ = ellipticity_inclination_tapered(b1, b2, alpha=alpha)
        return e, 0.0
    chi = 0.5 * np.arcsin(np.clip(v[valid] / p[valid], -1.0, 1.0))
    ell = np.tan(chi)
    ell = ell[np.isfinite(ell)]
    if ell.size == 0:
        e, _, _ = ellipticity_inclination_tapered(b1, b2, alpha=alpha)
        return e, 0.0
    q25, q50, q75 = np.percentile(ell, [25, 50, 75])
    return float(q50), float(q75 - q25)


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
        raise ValueError(f"field must have shape (n_samples, 3), got {field.shape}")
    if field.shape[0] < 3:
        return 0.0
    cov = np.cov(field, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)  # ascending: [lambda_3, lambda_2, lambda_1]
    if not np.all(np.isfinite(eigvals)) or eigvals[2] <= 0:
        return 0.0
    if eigvals[0] <= 0:
        return float("inf")
    return float(eigvals[1] / eigvals[0])


def mva_minimum_eigenvalue_fraction(
    field: ArrayLike,
) -> float:
    r"""Ratio :math:`\lambda_3 / \lambda_1` from minimum variance analysis.

    Generalises :func:`mva_intermediate_minimum_ratio` to handle all
    polarization geometries with a single criterion. The classic
    "planar perturbation" test is :math:`\lambda_2/\lambda_3 \geq 5`,
    but this fails for purely linear polarization where
    :math:`\lambda_2 \approx \lambda_3 \approx \sigma_{\text{noise}}^2`
    even though the wave is well-defined (just rank-1 instead of
    rank-2).

    The ratio :math:`\lambda_3 / \lambda_1` is small whenever the
    perturbation has rank :math:`\leq 2` in 3-D space:

    - linear pol (rank 1): :math:`\lambda_2 \approx \lambda_3 \to 0`,
      :math:`\lambda_3 / \lambda_1 \to 0`
    - circular pol (rank 2 planar): :math:`\lambda_3 \to 0`,
      :math:`\lambda_3 / \lambda_1 \to 0`
    - elliptical pol (rank 2 planar): same
    - 3-axis FGM step (rank 3): :math:`\lambda_3 \sim \lambda_1`,
      :math:`\lambda_3 / \lambda_1 \sim 0.3` to 1

    A single threshold (typically :math:`\leq 0.2`) cleanly separates
    plasma waves from isotropic transients without requiring a
    polarization branch.

    Parameters
    ----------
    field : array_like, shape (n_samples, 3)
        Three-component magnetic-field time series — typically
        ``(b_par, b_perp1, b_perp2)`` in MFA frame.

    Returns
    -------
    fraction : float
        :math:`\lambda_3 / \lambda_1 \in [0, 1]`. Returns ``0.0`` for
        empty or degenerate inputs and clips above at 1.0 if the
        eigenvalues are nearly equal.
    """
    field = np.asarray(field, dtype=float)
    if field.ndim != 2 or field.shape[1] != 3:
        raise ValueError(f"field must have shape (n_samples, 3), got {field.shape}")
    if field.shape[0] < 3:
        return 0.0
    cov = np.cov(field, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return 0.0
    eigvals = np.linalg.eigvalsh(cov)  # ascending: [lambda_3, lambda_2, lambda_1]
    if eigvals[2] <= 0:
        return 0.0
    return float(max(eigvals[0], 0.0) / eigvals[2])


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
        raise ValueError(f"field must have shape (n_samples, 3), got {field.shape}")
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
