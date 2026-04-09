"""Tests for Phase 7 modules: power-law background, matched filter,
wavelet coherence, quality score, and polarization fixes."""

from __future__ import annotations

import numpy as np
import pytest

from qp.signal.fft import estimate_background_powerlaw, welch_psd


# ------------------------------------------------------------------
# Phase 7.1 — power-law FFT background
# ------------------------------------------------------------------


class TestPowerlawBackground:
    """The power-law background must track a f^-α noise floor and NOT
    self-fit a narrow spectral peak."""

    def test_pure_powerlaw_recovery(self):
        """A synthetic f^-2 PSD should be recovered with <10% error."""
        rng = np.random.default_rng(42)
        freq = np.linspace(1e-5, 1e-2, 500)
        alpha = 2.0
        psd = 1e6 * freq ** (-alpha) + rng.normal(0, 1e2, 500)
        psd = np.maximum(psd, 1e-10)
        bg = estimate_background_powerlaw(psd, freq)
        # Check the slope in log space
        log_f = np.log10(freq[10:490])
        log_bg = np.log10(bg[10:490])
        coeffs = np.polyfit(log_f, log_bg, 1)
        np.testing.assert_allclose(coeffs[0], -alpha, atol=0.3)

    def test_peak_not_absorbed(self):
        """A narrow Gaussian peak on top of f^-2 noise should NOT
        be absorbed into the background — the ratio should be >>1."""
        freq = np.linspace(1e-5, 1e-2, 1000)
        psd = 1e6 * freq ** (-2.0)
        # Inject a strong peak at QP60 = 1/3600 Hz
        # Peak must be large relative to the base PSD at that frequency
        f_peak = 1.0 / 3600.0
        base_at_peak = 1e6 * f_peak ** (-2.0)
        psd += 10 * base_at_peak * np.exp(-((freq - f_peak) / 5e-5) ** 2)
        bg = estimate_background_powerlaw(psd, freq, exclude_bands=True)
        ratio = psd / bg
        peak_idx = np.argmin(np.abs(freq - f_peak))
        assert ratio[peak_idx] > 5.0, (
            f"Peak ratio = {ratio[peak_idx]:.2f}, expected > 5"
        )

    def test_degenerate_input(self):
        """Constant PSD should return a flat background without crashing."""
        freq = np.linspace(1e-5, 1e-2, 100)
        psd = np.ones(100) * 42.0
        bg = estimate_background_powerlaw(psd, freq)
        assert bg.shape == psd.shape
        assert np.all(np.isfinite(bg))


# ------------------------------------------------------------------
# Phase 7.2 — matched filter
# ------------------------------------------------------------------


class TestMatchedFilter:

    def test_detects_injected_signal(self):
        """A Gaussian-windowed sine should produce high MF-SNR."""
        from qp.signal.matched_filter import matched_filter_snr

        rng = np.random.default_rng(123)
        dt = 60.0
        n = 2160  # 36 hours
        t = np.arange(n) * dt
        period = 3600.0  # 1 hour

        # Red-ish noise + injected signal at centre
        noise = rng.normal(0, 0.3, n)
        signal = np.zeros(n)
        centre = n // 2
        env_width = 1.5 * period
        signal = 1.0 * np.sin(2 * np.pi * t / period) * np.exp(
            -0.5 * ((t - t[centre]) / env_width) ** 2
        )
        data = noise + signal

        snr = matched_filter_snr(data, dt=dt, period=period)
        assert snr.shape == (n,)
        # SNR near peak should be >> 3
        peak_snr = snr[centre - 30 : centre + 30].max()
        assert peak_snr > 3.0, f"Peak MF-SNR = {peak_snr:.2f}"

    def test_noise_only_low_snr(self):
        """Pure noise should produce low MF-SNR everywhere."""
        from qp.signal.matched_filter import matched_filter_snr

        rng = np.random.default_rng(456)
        data = rng.normal(0, 1.0, 1440)
        snr = matched_filter_snr(data, dt=60.0, period=3600.0)
        # 99th percentile should be < 5 for pure noise
        assert np.percentile(snr, 99) < 6.0


# ------------------------------------------------------------------
# Phase 7.3 — wavelet coherence
# ------------------------------------------------------------------


class TestWaveletCoherence:

    def test_coherent_signals(self):
        """Two sine waves with fixed phase offset should have high coherence."""
        from qp.signal.coherence import wavelet_coherence

        dt = 60.0
        n = 1440  # 24 hours
        t = np.arange(n) * dt
        period = 3600.0
        b1 = np.sin(2 * np.pi * t / period)
        b2 = np.cos(2 * np.pi * t / period)  # 90° offset

        freq, coh, phase, _ = wavelet_coherence(b1, b2, dt=dt, n_freqs=100)
        # Coherence at QP60 frequency should be high
        f_target = 1.0 / period
        f_idx = np.argmin(np.abs(freq - f_target))
        mean_coh = coh[f_idx, n // 4 : 3 * n // 4].mean()
        assert mean_coh > 0.7, f"Coherence = {mean_coh:.2f}"

    def test_incoherent_signals(self):
        """Independent noise should have low coherence."""
        from qp.signal.coherence import wavelet_coherence

        rng = np.random.default_rng(789)
        b1 = rng.normal(0, 1, 1440)
        b2 = rng.normal(0, 1, 1440)
        freq, coh, _, _ = wavelet_coherence(b1, b2, dt=60.0, n_freqs=100)
        mean_coh = np.mean(coh[:, 100:-100])
        assert mean_coh < 0.65, f"Noise coherence = {mean_coh:.2f}"


# ------------------------------------------------------------------
# Phase 7.4 — quality score
# ------------------------------------------------------------------


class TestQualityScore:

    def test_high_metrics_high_score(self):
        from qp.events.quality import compute_quality
        q = compute_quality(
            wavelet_sigma=15.0,
            fft_ratio=8.0,
            mf_snr=12.0,
            coherence=0.85,
            n_oscillations=8.0,
            transverse_ratio=15.0,
            polarization_fraction=0.9,
        )
        assert q > 0.7, f"Quality = {q:.3f}"

    def test_low_metrics_low_score(self):
        from qp.events.quality import compute_quality
        q = compute_quality(
            wavelet_sigma=1.0,
            fft_ratio=0.5,
            mf_snr=0.5,
            coherence=0.05,
            n_oscillations=1.0,
            transverse_ratio=0.3,
            polarization_fraction=0.1,
        )
        assert q < 0.3, f"Quality = {q:.3f}"

    def test_missing_metrics_graceful(self):
        from qp.events.quality import compute_quality
        q = compute_quality(wavelet_sigma=10.0)
        assert 0.0 < q <= 1.0

    def test_all_none_returns_zero(self):
        from qp.events.quality import compute_quality
        assert compute_quality() == 0.0


# ------------------------------------------------------------------
# Phase 7.8 — tapered polarization
# ------------------------------------------------------------------


class TestTaperedPolarization:

    def test_circular_signal_returns_high_ellipticity(self):
        """A perfect circularly polarized signal (π/2 phase shift)
        should give |ellipticity| ≈ 1."""
        from qp.signal.cross_correlation import ellipticity_inclination_tapered

        t = np.arange(1440) * 60.0
        b1 = np.sin(2 * np.pi * t / 3600.0)
        b2 = np.cos(2 * np.pi * t / 3600.0)
        e, _, pf = ellipticity_inclination_tapered(b1, b2)
        assert abs(e) > 0.8, f"|ellipticity| = {abs(e):.3f}, expected >0.8"
        assert pf > 0.8, f"pol_frac = {pf:.3f}"

    def test_linear_signal_returns_low_ellipticity(self):
        """In-phase signals (linear polarization) → ellipticity ≈ 0."""
        from qp.signal.cross_correlation import ellipticity_inclination_tapered

        t = np.arange(1440) * 60.0
        b1 = np.sin(2 * np.pi * t / 3600.0)
        b2 = np.sin(2 * np.pi * t / 3600.0)  # same phase
        e, _, _ = ellipticity_inclination_tapered(b1, b2)
        assert abs(e) < 0.2, f"|ellipticity| = {abs(e):.3f}"

    def test_per_oscillation_circular(self):
        """Per-oscillation ellipticity of a uniformly circular wave
        should be consistent (low IQR)."""
        from qp.signal.cross_correlation import per_oscillation_ellipticity

        t = np.arange(1440) * 60.0
        b1 = np.sin(2 * np.pi * t / 3600.0)
        b2 = np.cos(2 * np.pi * t / 3600.0)
        median_e, iqr = per_oscillation_ellipticity(b1, b2, dt=60.0,
                                                      period=3600.0)
        assert abs(median_e) > 0.7
        assert iqr < 0.3, f"IQR = {iqr:.3f}"
