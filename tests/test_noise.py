"""Tests for colored/power-law noise generation."""

import numpy as np

from qp.signal.noise import (
    colored_noise_3component,
    magnetospheric_background,
    power_law_noise,
)


class TestPowerLawNoise:
    """Verify PSD slope and basic properties of power_law_noise."""

    def test_white_noise_flat_psd(self):
        """alpha=0 should produce roughly flat PSD."""
        noise = power_law_noise(8192, dt=60.0, alpha=0.0, sigma=1.0, seed=42)
        psd = np.abs(np.fft.rfft(noise)) ** 2
        # Ratio of high-freq to low-freq power should be near 1 for flat PSD
        n = len(psd)
        low = np.mean(psd[n // 10 : n // 5])
        high = np.mean(psd[3 * n // 5 : 4 * n // 5])
        ratio = high / low
        assert 0.2 < ratio < 5.0, f"White noise PSD ratio {ratio} not flat"

    def test_red_noise_slope(self):
        """alpha=2 should produce PSD with slope near -2."""
        noise = power_law_noise(16384, dt=60.0, alpha=2.0, sigma=1.0, seed=42)
        freqs = np.fft.rfftfreq(len(noise), d=60.0)
        psd = np.abs(np.fft.rfft(noise)) ** 2

        # Fit slope in log-log space (exclude DC and Nyquist neighborhood)
        mask = (freqs > freqs[5]) & (freqs < freqs[-5])
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask])
        slope = np.polyfit(log_f, log_p, 1)[0]

        # Should be near -2 (within ±0.5 for finite-length realization)
        assert -2.8 < slope < -1.2, f"Red noise slope {slope} not near -2"

    def test_rms_matches_sigma(self):
        """Output RMS should match requested sigma."""
        for sigma in [0.1, 1.0, 5.0]:
            noise = power_law_noise(4096, dt=60.0, alpha=1.0, sigma=sigma, seed=7)
            rms = np.sqrt(np.mean(noise**2))
            np.testing.assert_allclose(rms, sigma, rtol=0.05)

    def test_seed_reproducibility(self):
        """Same seed → identical output."""
        a = power_law_noise(1000, seed=123)
        b = power_law_noise(1000, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self):
        """Different seeds → different output."""
        a = power_law_noise(1000, seed=1)
        b = power_law_noise(1000, seed=2)
        assert not np.allclose(a, b)

    def test_zero_mean(self):
        """DC component is zeroed; output should be near zero mean."""
        # For steep spectra (α≥1.5), oversampling + central extraction
        # suppresses but doesn't eliminate low-frequency content
        noise = power_law_noise(8192, alpha=1.0, sigma=1.0, seed=42)
        assert abs(np.mean(noise)) < 0.1


class TestColoredNoise3Component:
    def test_shape(self):
        result = colored_noise_3component(1000, dt=60.0, seed=10)
        assert result.shape == (1000, 3)

    def test_components_independent(self):
        """Three components should be weakly correlated."""
        result = colored_noise_3component(8192, dt=60.0, seed=10)
        corr01 = np.corrcoef(result[:, 0], result[:, 1])[0, 1]
        corr02 = np.corrcoef(result[:, 0], result[:, 2])[0, 1]
        assert abs(corr01) < 0.15
        assert abs(corr02) < 0.15


class TestMagnetosphericBackground:
    def test_shape(self):
        bg = magnetospheric_background(1440, dt=60.0, seed=42)
        assert bg.shape == (1440, 3)

    def test_mean_field_in_bpar(self):
        """B_par should have the mean field; B_perp should be near zero mean."""
        bg = magnetospheric_background(14400, dt=60.0, seed=42, b_mean=5.0)
        assert abs(np.mean(bg[:, 0]) - 5.0) < 3.0  # mean ± trend + noise
        assert abs(np.mean(bg[:, 1])) < 1.0
        assert abs(np.mean(bg[:, 2])) < 1.0

    def test_ppo_present(self):
        """PPO should produce power near 1/10.7h frequency."""
        bg = magnetospheric_background(
            14400, dt=60.0, seed=42, ppo_amplitude=2.0, noise_sigma=0.01
        )
        freqs = np.fft.rfftfreq(14400, d=60.0)
        psd = np.abs(np.fft.rfft(bg[:, 1])) ** 2
        ppo_freq = 1.0 / (10.7 * 3600)
        ppo_idx = np.argmin(np.abs(freqs - ppo_freq))
        # PPO peak should be prominent
        neighborhood = psd[max(0, ppo_idx - 3) : ppo_idx + 4]
        assert np.max(neighborhood) > 10 * np.median(psd)

    def test_seed_reproducibility(self):
        a = magnetospheric_background(1000, seed=99)
        b = magnetospheric_background(1000, seed=99)
        np.testing.assert_array_equal(a, b)


class TestHeavyTailedAndRegimeSwitching:
    """Tests for round-3 noise hardening additions."""

    def test_heavy_tail_kurtosis(self):
        """Student-t df=5 should produce heavier tails than Gaussian."""
        from scipy import stats

        gauss = power_law_noise(8192, alpha=0.0, sigma=1.0, seed=1)
        heavy = power_law_noise(8192, alpha=0.0, sigma=1.0, seed=1, tail_df=5.0)
        # Excess kurtosis: Gaussian ≈ 0; Student-t df=5 has population
        # excess kurtosis = 6 / (df - 4) = 6. Sample estimate is noisy
        # but should be clearly elevated vs Gaussian.
        assert stats.kurtosis(heavy) > stats.kurtosis(gauss) + 0.5

    def test_heavy_tail_variance_match(self):
        """tail_df-renormalised noise still hits requested sigma."""
        x = power_law_noise(16384, alpha=1.2, sigma=0.3, seed=7, tail_df=5.0)
        np.testing.assert_allclose(np.std(x), 0.3, rtol=0.05)

    def test_regime_switching_shape_and_variance(self):
        from qp.signal.noise import regime_switching_noise

        n = 86400 // 60 * 2  # 2 days at 1-min cadence
        x = regime_switching_noise(
            n, dt=60.0, alpha_range=(1.0, 1.7),
            segment_hours_range=(6.0, 12.0), sigma=0.05, seed=11,
        )
        assert x.shape == (n,)
        np.testing.assert_allclose(np.std(x), 0.05, rtol=0.05)

    def test_regime_switching_seed_reproducibility(self):
        from qp.signal.noise import regime_switching_noise

        a = regime_switching_noise(2880, sigma=1.0, seed=22)
        b = regime_switching_noise(2880, sigma=1.0, seed=22)
        np.testing.assert_array_equal(a, b)

    def test_realistic_ppo_broadens_line(self):
        """Realistic PPO smears the line vs. the clean dual sinusoid."""
        from qp.signal.noise import inject_ppo

        # 30 days at 1-min cadence — enough Δf resolution to resolve
        # the bandwidth difference between clean and realistic PPO.
        n = 30 * 24 * 60
        t = np.arange(n) * 60.0

        clean = np.zeros((n, 3))
        inject_ppo(clean, t, amplitude=1.0, seed=33, realistic=False)
        rough = np.zeros((n, 3))
        inject_ppo(rough, t, amplitude=1.0, seed=33, realistic=True)

        psd_c = np.abs(np.fft.rfft(clean[:, 1])) ** 2
        psd_r = np.abs(np.fft.rfft(rough[:, 1])) ** 2
        freqs = np.fft.rfftfreq(n, d=60.0)
        # Window: ±30 % around the geometric mean of the two PPO
        # systems, generous enough to contain drift sidebands.
        center = 1.0 / (10.7 * 3600.0)
        mask = (freqs > 0.7 * center) & (freqs < 1.3 * center)

        # Spectral entropy: lower for the clean dual-tone than for the
        # broadened realistic PPO.
        def _spread(p: np.ndarray) -> float:
            p = p / p.sum()
            return float(-np.sum(p[p > 0] * np.log(p[p > 0])))

        assert _spread(psd_r[mask]) > _spread(psd_c[mask])
