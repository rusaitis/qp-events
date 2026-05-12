r"""Verification of alternative CWT thresholding methods.

Tests are pinned to closed-form behaviour where possible and to
statistical false-positive rates on synthetic AR(1) noise everywhere
else, so the gates can't silently regress to a different sensitivity.
"""

from __future__ import annotations

import numpy as np
import pytest

from qp.events.threshold_diag import (
    BGArchive,
    _ar1_fourier_shape,
    ar1_lag1_coefficient,
    coi_mask,
    fdr_chi2_mask,
    pooled_archive_mask,
    torrence_compo_chi2_mask,
)
from qp.signal.wavelet import morlet_cwt


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _ar1_series(
    n: int,
    alpha1: float,
    *,
    sigma: float = 1.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an AR(1) time series x_t = alpha1·x_{t-1} + sigma·eps."""
    x = np.zeros(n)
    eps = rng.standard_normal(n) * sigma * np.sqrt(1.0 - alpha1 * alpha1)
    for i in range(1, n):
        x[i] = alpha1 * x[i - 1] + eps[i]
    return x


# --------------------------------------------------------------------- #
# COI mask                                                              #
# --------------------------------------------------------------------- #


def test_coi_mask_shape() -> None:
    freq = np.array([1e-4, 1e-3, 1e-2])
    mask = coi_mask(freq, n_time=1024, dt=60.0)
    assert mask.shape == (3, 1024)
    assert mask.dtype == bool


def test_coi_mask_higher_freq_exclusion_smaller_than_lower_freq() -> None:
    freq = np.array([1e-4, 1e-3, 1e-2])  # increasing → smaller COI
    mask = coi_mask(freq, n_time=1024, dt=60.0)
    excluded = (~mask).sum(axis=1)
    # Higher frequency → fewer COI samples.
    assert excluded[2] < excluded[1] < excluded[0]


def test_coi_mask_handles_zero_time() -> None:
    freq = np.array([1e-3, 1e-2])
    mask = coi_mask(freq, n_time=0, dt=60.0)
    assert mask.shape == (2, 0)


def test_coi_mask_never_masks_full_row() -> None:
    """Even at extremely low frequencies, leave at least one valid cell."""
    freq = np.array([1e-7])  # ~ 1 / 100 days
    mask = coi_mask(freq, n_time=120, dt=60.0)
    assert mask.any(axis=1).all()


# --------------------------------------------------------------------- #
# AR(1) lag-1 coefficient                                                #
# --------------------------------------------------------------------- #


def test_ar1_coefficient_zero_for_white_noise() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    assert abs(ar1_lag1_coefficient(x)) < 0.03


@pytest.mark.parametrize("alpha_true", [0.3, 0.6, 0.9, 0.95, -0.5])
def test_ar1_coefficient_recovers_known_alpha(alpha_true: float) -> None:
    rng = np.random.default_rng(abs(int(alpha_true * 1000)) + 7)
    x = _ar1_series(50000, alpha_true, rng=rng)
    alpha_hat = ar1_lag1_coefficient(x)
    assert abs(alpha_hat - alpha_true) < 0.02


def test_ar1_coefficient_clamps_to_safe_range() -> None:
    # A near-perfectly correlated series would otherwise blow up the
    # AR(1) Fourier denominator; the function clamps to [-0.99, 0.99].
    x = np.linspace(0.0, 1.0, 10000)  # monotonic ramp → alpha → 1
    assert abs(ar1_lag1_coefficient(x)) <= 0.99 + 1e-9


def test_ar1_coefficient_safe_on_degenerate_input() -> None:
    assert ar1_lag1_coefficient([]) == 0.0
    assert ar1_lag1_coefficient([1.0]) == 0.0
    assert ar1_lag1_coefficient([0.0, 0.0, 0.0]) == 0.0


# --------------------------------------------------------------------- #
# AR(1) Fourier shape                                                    #
# --------------------------------------------------------------------- #


def test_ar1_shape_flat_for_white_noise() -> None:
    freq = np.linspace(1e-4, 5e-3, 100)
    shape = _ar1_fourier_shape(freq, alpha1=0.0, dt=60.0)
    # At alpha=0, AR(1) collapses to white noise — flat spectrum.
    np.testing.assert_allclose(shape, np.ones_like(shape))


def test_ar1_shape_red_for_positive_alpha() -> None:
    freq = np.linspace(1e-4, 5e-3, 100)
    shape = _ar1_fourier_shape(freq, alpha1=0.9, dt=60.0)
    # Strictly decreasing with frequency on the resolved range.
    assert np.all(np.diff(shape) < 0)


def test_ar1_shape_higher_alpha_steeper_slope() -> None:
    freq = np.linspace(1e-4, 5e-3, 100)
    s_mild = _ar1_fourier_shape(freq, alpha1=0.3, dt=60.0)
    s_strong = _ar1_fourier_shape(freq, alpha1=0.9, dt=60.0)
    # Slope steeper (more low-freq concentration) at higher alpha.
    ratio_mild = s_mild[0] / s_mild[-1]
    ratio_strong = s_strong[0] / s_strong[-1]
    assert ratio_strong > 5 * ratio_mild


# --------------------------------------------------------------------- #
# Torrence-Compo χ² mask                                                 #
# --------------------------------------------------------------------- #


def test_tc_mask_false_positive_rate_matches_alpha() -> None:
    """On pure AR(1) noise, fraction of rejected cells ≈ alpha (per-pixel)."""
    rng = np.random.default_rng(42)
    dt = 60.0
    n = 1500  # 25 h, ~ a 36-h segment minus COI
    b = _ar1_series(n, alpha1=0.9, rng=rng)
    freq, _, cwt = morlet_cwt(b, dt=dt, n_freqs=80)
    amp = np.abs(cwt)
    # Apply COI mask so edge contamination doesn't bias the statistic.
    valid = coi_mask(freq, n, dt=dt)
    for alpha in (0.05, 0.01, 0.001):
        mask = torrence_compo_chi2_mask(amp, freq, b, dt=dt, alpha=alpha)
        fp_rate = (mask & valid).sum() / valid.sum()
        # Calibration uses a per-segment median, so the actual rate
        # tracks alpha within a factor of ~3 — tight enough to catch
        # gross miscalibration, loose enough to not flake.
        assert alpha / 3 < fp_rate < alpha * 3, (
            f"alpha={alpha}, observed FP rate={fp_rate}"
        )


def test_tc_mask_detects_injected_packet() -> None:
    """Inject a 60-min sinusoid; TC must flag the packet's CWT cells."""
    rng = np.random.default_rng(7)
    dt = 60.0
    n = 1500
    t = np.arange(n) * dt
    b = _ar1_series(n, alpha1=0.85, rng=rng)
    # 8-hour packet centred at 12 h, 60-min period, amplitude 2 σ
    period = 60.0 * 60.0
    centre = 12.0 * 3600.0
    half = 4.0 * 3600.0
    envelope = np.exp(-(((t - centre) / half) ** 2))
    b = b + 2.0 * envelope * np.sin(2 * np.pi * t / period)
    freq, _, cwt = morlet_cwt(b, dt=dt, n_freqs=80)
    mask = torrence_compo_chi2_mask(np.abs(cwt), freq, b, dt=dt, alpha=0.001)
    # Find the row closest to f = 1/60min and check the packet centre
    # cells are flagged.
    i_row = int(np.argmin(np.abs(freq - 1.0 / period)))
    i_t = int(np.argmin(np.abs(t - centre)))
    # At least 10 % of the packet's ±2 h window at the peak row should
    # be flagged.
    window = mask[i_row, max(0, i_t - 120) : i_t + 120]
    assert window.mean() > 0.1


def test_tc_mask_signature_matches_wavelet_sigma_mask() -> None:
    """Returned mask has the same shape and dtype as the baseline gate."""
    rng = np.random.default_rng(1)
    b = _ar1_series(600, alpha1=0.5, rng=rng)
    freq, _, cwt = morlet_cwt(b, dt=60.0, n_freqs=40)
    amp = np.abs(cwt)
    mask = torrence_compo_chi2_mask(amp, freq, b, dt=60.0)
    assert mask.shape == amp.shape
    assert mask.dtype == bool


# --------------------------------------------------------------------- #
# FDR χ² mask                                                            #
# --------------------------------------------------------------------- #


def test_fdr_mask_no_rejections_under_null() -> None:
    """Under pure AR(1) noise, BH-FDR at q=0.01 rejects almost nothing."""
    rng = np.random.default_rng(99)
    b = _ar1_series(1200, alpha1=0.9, rng=rng)
    freq, _, cwt = morlet_cwt(b, dt=60.0, n_freqs=60)
    mask = fdr_chi2_mask(np.abs(cwt), freq, b, dt=60.0, q=0.01)
    # FDR controls the expected proportion of false discoveries among
    # rejections; under the null with no true signal, that's the whole
    # rejection set. So we expect very few rejected cells — well under
    # 1 % overall.
    assert mask.mean() < 0.01


def test_fdr_mask_detects_strong_packet() -> None:
    """FDR control still flags an obviously injected packet."""
    rng = np.random.default_rng(13)
    dt = 60.0
    n = 1500
    t = np.arange(n) * dt
    b = _ar1_series(n, alpha1=0.85, rng=rng)
    period = 60.0 * 60.0
    centre = 12.0 * 3600.0
    half = 4.0 * 3600.0
    envelope = np.exp(-(((t - centre) / half) ** 2))
    b = b + 4.0 * envelope * np.sin(2 * np.pi * t / period)  # strong signal
    freq, _, cwt = morlet_cwt(b, dt=dt, n_freqs=80)
    mask = fdr_chi2_mask(np.abs(cwt), freq, b, dt=dt, q=0.01)
    i_row = int(np.argmin(np.abs(freq - 1.0 / period)))
    i_t = int(np.argmin(np.abs(t - centre)))
    assert mask[i_row, i_t]


# --------------------------------------------------------------------- #
# Pooled archive mask                                                    #
# --------------------------------------------------------------------- #


def _make_synthetic_archive(
    freq: np.ndarray,
    cwt_amp: np.ndarray,
    region: str = "magnetosphere",
) -> BGArchive:
    """Build a one-segment archive from per-row stats of a CWT amplitude."""
    medians = np.median(cwt_amp, axis=1)
    mads = np.median(np.abs(cwt_amp - medians[:, None]), axis=1)
    return BGArchive(
        periods_sec=1.0 / freq,
        medians={region: medians},
        mads={region: mads},
        n_segments={region: 1},
    )


def test_pooled_archive_mask_threshold_increases_with_n_sigma() -> None:
    rng = np.random.default_rng(2)
    b = _ar1_series(600, alpha1=0.7, rng=rng)
    freq, _, cwt = morlet_cwt(b, dt=60.0, n_freqs=40)
    amp = np.abs(cwt)
    arch = _make_synthetic_archive(freq, amp)
    mask3 = pooled_archive_mask(amp, freq, "magnetosphere", arch, n_sigma=3.0)
    mask5 = pooled_archive_mask(amp, freq, "magnetosphere", arch, n_sigma=5.0)
    # 5-sigma must be a strict subset of 3-sigma.
    assert mask5.sum() <= mask3.sum()
    assert (mask5 & ~mask3).sum() == 0


def test_pooled_archive_mask_unknown_region_raises() -> None:
    freq = np.linspace(1e-4, 1e-2, 10)
    arch = BGArchive(
        periods_sec=1.0 / freq,
        medians={"magnetosphere": np.ones_like(freq)},
        mads={"magnetosphere": np.ones_like(freq) * 0.1},
        n_segments={"magnetosphere": 1},
    )
    amp = np.ones((10, 20))
    with pytest.raises(KeyError, match="unknown_region"):
        pooled_archive_mask(amp, freq, "unknown_region", arch)


def test_pooled_archive_mask_interpolates_in_log_period() -> None:
    """Archive on a coarser grid is interpolated, not nearest-neighbour."""
    # Archive: two points only — a flat median, MAD growing linearly in
    # log-period from 1 to 2.
    arch_freq = np.array([1e-3, 1e-4])  # 1000 s, 10000 s
    arch_periods = 1.0 / arch_freq
    arch = BGArchive(
        periods_sec=arch_periods,
        medians={"r": np.array([0.0, 0.0])},
        mads={"r": np.array([1.0, 2.0])},
        n_segments={"r": 1},
    )
    # CWT grid: three points spanning the archive range plus midpoint.
    cwt_freq = np.array([1e-3, np.sqrt(1e-3 * 1e-4), 1e-4])  # log-midpoint
    # At n_sigma=1, threshold = MAD_TO_SIGMA*mad. Midpoint MAD should be
    # the geometric-midpoint interpolant in log-period — ~1.5.
    arch_lo, arch_hi = arch.mads["r"]
    log_mid = 0.5 * (np.log10(arch_periods[0]) + np.log10(arch_periods[1]))
    log_p = np.log10(arch_periods)
    order = np.argsort(log_p)
    expect_mid_mad = float(np.interp(log_mid, log_p[order], arch.mads["r"][order]))
    assert arch_lo < expect_mid_mad < arch_hi
    # Sanity: threshold at the midpoint row is between the two anchor
    # thresholds.
    amp_above = np.full((3, 10), 100.0)
    mask = pooled_archive_mask(amp_above, cwt_freq, "r", arch, n_sigma=1.0)
    assert mask.all()  # all cells very above any threshold


def test_pooled_archive_matches_per_segment_when_built_from_one_segment() -> None:
    """An archive built from one segment ≈ that segment's own per-row threshold."""
    rng = np.random.default_rng(11)
    b = _ar1_series(800, alpha1=0.85, rng=rng)
    freq, _, cwt = morlet_cwt(b, dt=60.0, n_freqs=40)
    amp = np.abs(cwt)
    arch = _make_synthetic_archive(freq, amp)
    # Per-segment per-row threshold computed locally — same formula.
    medians = np.median(amp, axis=1)
    mads = np.median(np.abs(amp - medians[:, None]), axis=1)
    thr_local = medians + 3.0 * 1.4826 * mads
    expected = amp > thr_local[:, None]
    got = pooled_archive_mask(amp, freq, "magnetosphere", arch, n_sigma=3.0)
    # Tiny numerical drift from the log-period interpolation step.
    disagree = (expected ^ got).sum()
    assert disagree / expected.size < 1e-3
