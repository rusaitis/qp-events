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
        noise = power_law_noise(8192, alpha=1.5, sigma=1.0, seed=42)
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
