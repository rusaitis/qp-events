r"""Alternative CWT background/threshold estimators for diagnostic comparison.

The canonical round-8 detector uses
:func:`qp.events.threshold.wavelet_sigma_mask` — a per-row MAD on
"background" period rows (those outside every QP band), interpolated in
log-period across QP-band rows on the *same* 36-h segment. With the
octave-tiled QP bands of commit ``c31b27b`` the in-band region spans
4 octaves [10, 160) min, leaving only a 1-octave HF anchor
[5, 10) min (in the aliasing zone) and a ~2.5-octave LF anchor
[160 min, 12 h) (in the red-noise hump) to constrain the threshold —
roughly half the search range is now bridged by a single log-period
slope. This module provides three alternative gates plus a cone-of-influence
helper so the trade-offs can be benchmarked head-to-head before any
behaviour change to the canonical sweep.

Functions
---------
:func:`coi_mask`
    Boolean mask that is ``False`` inside the Morlet cone of influence.
    Drop-in companion to any of the σ-mask functions (apply as
    ``mask &= coi_mask(...)`` after thresholding).
:func:`torrence_compo_chi2_mask`
    Parametric AR(1) red-noise null with closed-form Fourier expectation
    and per-pixel :math:`\chi^2_2` significance (Torrence & Compo 1998
    §4). Calibrated against the in-segment background-row power so the
    wavelet normalisation cancels.
:func:`fdr_chi2_mask`
    Same red-noise null as Torrence-Compo, but with Benjamini-Hochberg
    (dependent-test variant ``fdr_by``) FDR control over the TF cells
    rather than Bonferroni FWER.
:func:`pooled_archive_mask`
    Per-row threshold drawn from a pre-computed multi-segment archive
    of background CWT power keyed on plasma region. Defers the noise
    estimate from a single 36-h segment to thousands of segments;
    addresses the "one segment to estimate itself" failure mode where
    a long sheath excursion inflates the per-row MAD and starves
    detection in the magnetospheric portion of the same segment.

All four mask functions share the signature
``mask(cwt_power, cwt_freq, ...) -> ndarray[bool]`` so they can be
swapped behind the detector's ``threshold_method`` parameter once the
diagnostic results are in.

The ``cwt_power`` input convention matches
:func:`qp.events.threshold.wavelet_sigma_mask`: pass the wavelet
*amplitude* ``np.abs(cwt)`` (not the squared power). The χ²-based
functions internally square this to recover ``|W|^2``, which is the
quantity whose AR(1) noise distribution is ``P_AR1(s)·χ²_2/2``.

References
----------
Torrence, C. & Compo, G. P. (1998), "A practical guide to wavelet
analysis", *Bull. Amer. Meteor. Soc.* **79**, 61-78. §4 gives the
AR(1)+χ² null derivation; §3i defines the cone of influence.

Benjamini, Y. & Yekutieli, D. (2001), "The control of the false
discovery rate in multiple testing under dependency", *Ann. Stat.*
**29**, 1165-1188. Justifies the ``fdr_by`` correction used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from qp.events.bands import freq_to_period
from qp.events.threshold import MAD_TO_SIGMA, _background_row_indices

#: Lag-1 autocorrelation safety clamp. Values outside this band collapse
#: the AR(1) spectrum to near-singular shapes (alpha→±1 is a pure tone
#: or pure anti-correlation) that are unphysical for magnetospheric
#: background noise. The clamp keeps the theoretical spectrum
#: well-behaved without changing the empirical calibration step.
_AR1_CLAMP: float = 0.99


# ---------------------------------------------------------------------- #
# Cone of influence                                                      #
# ---------------------------------------------------------------------- #


def coi_mask(
    cwt_freq: ArrayLike,
    n_time: int,
    dt: float = 60.0,
    omega0: float = 10.0,
) -> NDArray[np.bool_]:
    r"""Boolean mask, ``False`` inside the Morlet cone of influence.

    Torrence & Compo (1998) §3i define the COI as the region within
    one e-folding time of the segment edge, where the wavelet kernel
    overlaps the boundary and the magnitude is biased low. For the
    Morlet wavelet of order :math:`\omega_0`, the COI half-width at
    frequency :math:`f` is

    .. math::

        \tau_{\mathrm{COI}}(f) = \frac{\sqrt{2}\,\omega_0}{2\pi f}.

    With the repository's default :math:`\omega_0 = 10`, this is
    about 2.25 periods from each edge — most QP events satisfying the
    ``min_duration_hours = 2`` requirement clear the COI comfortably,
    but the unmasked region at the lowest frequencies (QP120 row) can
    occupy up to 75 % of a 36-h segment.

    Parameters
    ----------
    cwt_freq : array_like, shape (n_freq,)
        Wavelet frequency axis in Hz.
    n_time : int
        Number of time samples in the CWT.
    dt : float
        Sampling interval in seconds.
    omega0 : float
        Morlet wavelet central-frequency parameter.

    Returns
    -------
    mask : ndarray of bool, shape (n_freq, n_time)
        ``True`` outside the COI, ``False`` inside.
    """
    f = np.asarray(cwt_freq, dtype=float)
    if n_time <= 0:
        return np.zeros((f.size, 0), dtype=bool)
    # COI half-width in samples. Clipped to leave at least one valid
    # cell at the row centre even at frequencies whose COI would cover
    # the whole segment — otherwise an extreme low-freq row contributes
    # nothing and downstream consumers (median, percentile) get NaN.
    t_coi = np.sqrt(2.0) * omega0 / (2.0 * np.pi * f)  # seconds
    n_coi = np.clip(np.ceil(t_coi / dt).astype(int), 0, max(0, (n_time - 1) // 2))
    t_idx = np.arange(n_time)
    return (t_idx[None, :] >= n_coi[:, None]) & (
        t_idx[None, :] < n_time - n_coi[:, None]
    )


# ---------------------------------------------------------------------- #
# AR(1) red-noise null (Torrence & Compo 1998 §4)                        #
# ---------------------------------------------------------------------- #


def ar1_lag1_coefficient(b_t: ArrayLike) -> float:
    r"""Lag-1 autocorrelation coefficient :math:`\alpha_1` of a time series.

    Used to parameterise the AR(1) red-noise null hypothesis. Computed
    as the Pearson correlation between ``b_t[:-1]`` and ``b_t[1:]``
    after mean removal. Clamped to ``[-_AR1_CLAMP, _AR1_CLAMP]`` so the
    theoretical AR(1) Fourier spectrum stays finite.

    Parameters
    ----------
    b_t : array_like
        Real-valued time series — typically a single MFA component.

    Returns
    -------
    alpha1 : float
        Lag-1 autocorrelation coefficient.
    """
    b = np.asarray(b_t, dtype=float).ravel()
    if b.size < 3:
        return 0.0
    b = b - np.mean(b)
    num = float(np.sum(b[:-1] * b[1:]))
    den = float(np.sum(b * b))
    if den <= 0.0:
        return 0.0
    return float(np.clip(num / den, -_AR1_CLAMP, _AR1_CLAMP))


def _ar1_fourier_shape(
    cwt_freq: NDArray[np.floating],
    alpha1: float,
    dt: float,
) -> NDArray[np.floating]:
    r"""Unit-amplitude Fourier shape of an AR(1) process at the given frequencies.

    Torrence & Compo (1998) Eq. (16):

    .. math::

        P_k = \frac{1 - \alpha_1^2}{1 + \alpha_1^2 - 2\alpha_1\cos(2\pi k)}

    with :math:`k = f / f_s` the normalised frequency. Returns the
    shape only; the amplitude is set by per-segment calibration in
    :func:`_calibrate_ar1_to_bg`.
    """
    k = cwt_freq * dt  # f / f_s
    return (1.0 - alpha1 * alpha1) / (
        1.0 + alpha1 * alpha1 - 2.0 * alpha1 * np.cos(2.0 * np.pi * k)
    )


def _calibrate_ar1_to_bg(
    cwt_power_sq: NDArray[np.floating],
    cwt_freq: NDArray[np.floating],
    alpha1: float,
    dt: float,
) -> NDArray[np.floating]:
    r"""Theoretical mean :math:`|W|^2` per CWT row, calibrated to the segment.

    The Morlet CWT in this repo is energy-normalised at each scale
    (``wavelet /= sqrt(s)``), so the absolute amplitude of the
    expected AR(1) wavelet power depends on the convolution
    convention. Rather than derive that constant from first
    principles, we anchor the shape to data: compute the empirical
    median of ``|W|^2`` on background rows (rows the QP signal is
    forbidden from occupying), divide by the median-to-mean factor
    :math:`\ln 2` for the exponential distribution that ``|W|^2 /
    \langle |W|^2 \rangle`` follows under AR(1) noise, and fit a
    single scaling factor so the AR(1) spectrum passes through the
    background-row means.

    The returned array is the per-row mean :math:`\langle |W(s)|^2
    \rangle` under the AR(1) null, with the same shape as
    ``cwt_freq``.
    """
    shape = _ar1_fourier_shape(cwt_freq, alpha1, dt)
    bg_rows = _background_row_indices(cwt_freq)
    if bg_rows.size == 0:
        # Degenerate: no background rows. Anchor to the overall median
        # so the threshold at least has the right order of magnitude.
        overall_mean = float(np.median(cwt_power_sq)) / np.log(2.0)
        return shape * overall_mean / float(np.median(shape))
    bg_medians_sq = np.median(cwt_power_sq[bg_rows], axis=1)
    bg_means_estimate = bg_medians_sq / np.log(2.0)  # exponential bias correction
    # Single-scalar calibration: minimises the median per-row ratio.
    calibration = float(np.median(bg_means_estimate / shape[bg_rows]))
    return shape * calibration


def torrence_compo_chi2_mask(
    cwt_power: ArrayLike,
    cwt_freq: ArrayLike,
    b_t: ArrayLike,
    dt: float = 60.0,
    alpha: float = 0.01,
) -> NDArray[np.bool_]:
    r"""Boolean mask using the Torrence-Compo AR(1)+χ²(2) significance test.

    Per-pixel test of ``|W|^2`` against the analytic AR(1) red-noise
    expectation, calibrated to the segment's background rows. Rejects
    cells with ``|W|^2 > P_AR1(s) · χ²_{2, α}/2``, where
    :math:`\chi^2_{2, \alpha}` is the upper-tail quantile of the
    :math:`\chi^2_2` distribution. No Bonferroni / FDR control —
    caller is expected to apply a multiple-testing correction
    downstream, or use :func:`fdr_chi2_mask` directly.

    Parameters
    ----------
    cwt_power : array_like, shape (n_freq, n_time)
        Wavelet amplitude ``np.abs(cwt)``. Squared internally to
        recover :math:`|W|^2`.
    cwt_freq : array_like, shape (n_freq,)
        Wavelet frequency axis in Hz.
    b_t : array_like
        Real-valued time series — used only to estimate the AR(1)
        lag-1 coefficient. Typically a single MFA component
        (``b_perp1`` or ``b_perp2``).
    dt : float
        Sampling interval in seconds.
    alpha : float
        Per-pixel false-positive probability under the AR(1) null.
        ``0.01`` ≈ 4.6 σ (Gaussian-equivalent) at one test; with
        Bonferroni this maps to the FWER level after correction.

    Returns
    -------
    mask : ndarray of bool, shape (n_freq, n_time)
        ``True`` where ``|W|^2`` exceeds its row's threshold.
    """
    power = np.asarray(cwt_power, dtype=float)
    freq = np.asarray(cwt_freq, dtype=float)
    alpha1 = ar1_lag1_coefficient(b_t)
    p_theoretical = _calibrate_ar1_to_bg(power * power, freq, alpha1, dt)
    chi2_thresh = float(chi2.isf(alpha, df=2)) / 2.0
    thr = p_theoretical * chi2_thresh
    return (power * power) > thr[:, None]


def fdr_chi2_mask(
    cwt_power: ArrayLike,
    cwt_freq: ArrayLike,
    b_t: ArrayLike,
    dt: float = 60.0,
    q: float = 0.01,
    method: str = "fdr_by",
) -> NDArray[np.bool_]:
    r"""Boolean mask using AR(1) χ²(2) p-values + Benjamini-Hochberg FDR.

    Same red-noise null as :func:`torrence_compo_chi2_mask`, but the
    per-pixel p-values are corrected via Benjamini-Hochberg-Yekutieli
    (default ``method="fdr_by"``, valid under arbitrary dependency)
    rather than Bonferroni FWER. FDR control is less conservative
    when many tests have true positives; the trade-off is that the
    *expected* false-discovery rate is bounded by ``q`` instead of
    the FWER.

    Parameters
    ----------
    cwt_power : array_like, shape (n_freq, n_time)
        Wavelet amplitude ``np.abs(cwt)``.
    cwt_freq : array_like, shape (n_freq,)
        Wavelet frequency axis in Hz.
    b_t : array_like
        Real-valued time series for the AR(1) fit.
    dt : float
        Sampling interval in seconds.
    q : float
        FDR target (expected fraction of false discoveries among
        rejected cells).
    method : str
        Multiple-testing correction passed to
        :func:`statsmodels.stats.multitest.multipletests`. Default
        ``"fdr_by"`` (BH under arbitrary dependence). Use
        ``"fdr_bh"`` if you are willing to assume positive
        regression dependency — slightly less conservative.

    Returns
    -------
    mask : ndarray of bool, shape (n_freq, n_time)
    """
    from statsmodels.stats.multitest import multipletests

    power = np.asarray(cwt_power, dtype=float)
    freq = np.asarray(cwt_freq, dtype=float)
    power_sq = power * power
    alpha1 = ar1_lag1_coefficient(b_t)
    p_theoretical = _calibrate_ar1_to_bg(power_sq, freq, alpha1, dt)
    # Pixel statistic: 2|W|^2 / <|W|^2> ~ chi^2_2 under the null.
    stat = 2.0 * power_sq / p_theoretical[:, None]
    pvals = chi2.sf(stat, df=2).ravel()
    reject, *_ = multipletests(pvals, alpha=q, method=method)
    return reject.reshape(power.shape)


# ---------------------------------------------------------------------- #
# Pooled archive across segments                                         #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BGArchive:
    r"""Pre-computed background CWT power statistics, pooled across segments.

    Produced by ``scripts/build_bg_archive.py`` (Phase 2). For each plasma
    region (``magnetosphere``, ``magnetosheath``, ``solar_wind``,
    ``unknown``) and each period in :attr:`periods_sec`, stores the
    cross-segment median of the per-segment background-row median and
    MAD of :math:`|W|`. The pooled-mask gate uses these as the noise
    floor in place of the per-segment estimate.

    Attributes
    ----------
    periods_sec : ndarray, shape (n_periods,)
        Period axis on which medians/MADs are tabulated.
    medians : dict[str, ndarray]
        ``{region: per-period background median of |W|}``.
    mads : dict[str, ndarray]
        ``{region: per-period background MAD of |W|}``.
    n_segments : dict[str, int]
        Number of segments contributing to each region's statistics.
    """

    periods_sec: NDArray[np.floating]
    medians: dict[str, NDArray[np.floating]]
    mads: dict[str, NDArray[np.floating]]
    n_segments: dict[str, int]


def pooled_archive_mask(
    cwt_power: ArrayLike,
    cwt_freq: ArrayLike,
    region: str,
    archive: BGArchive,
    n_sigma: float = 3.0,
) -> NDArray[np.bool_]:
    r"""Boolean mask using a multi-segment archive for the noise floor.

    For each CWT row, the median and MAD come from the archive at the
    matching period (log-period interpolation if the archive grid
    differs from the CWT grid), and the threshold is
    ``median + n_sigma · 1.4826 · MAD``. The result is a per-row
    threshold like :func:`qp.events.threshold.wavelet_sigma_mask`,
    but it does not change with the 36-h segment under analysis.

    Parameters
    ----------
    cwt_power : array_like, shape (n_freq, n_time)
        Wavelet amplitude ``np.abs(cwt)``.
    cwt_freq : array_like, shape (n_freq,)
        Wavelet frequency axis in Hz.
    region : str
        Plasma region label — must be a key in ``archive.medians``
        and ``archive.mads``.
    archive : BGArchive
        Pre-computed background-row statistics from multiple
        segments.
    n_sigma : float
        Threshold multiplier on the robust σ (``1.4826 · MAD``).

    Returns
    -------
    mask : ndarray of bool, shape (n_freq, n_time)
    """
    if region not in archive.medians or region not in archive.mads:
        raise KeyError(
            f"region {region!r} not in archive (known: {sorted(archive.medians)})"
        )
    power = np.asarray(cwt_power, dtype=float)
    freq = np.asarray(cwt_freq, dtype=float)
    log_arch = np.log10(np.asarray(archive.periods_sec, dtype=float))
    log_cwt = np.log10(freq_to_period(freq))
    order = np.argsort(log_arch)
    med = np.interp(log_cwt, log_arch[order], archive.medians[region][order])
    mad = np.interp(log_cwt, log_arch[order], archive.mads[region][order])
    thr = med + n_sigma * MAD_TO_SIGMA * mad
    return power > thr[:, None]
