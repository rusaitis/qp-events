"""Tests for qp.analysis.spectral_fitting — power-law fits and spectral slopes."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.analysis.spectral_fitting import (
    PowerLawResult,
    bin_power_spectra,
    power_law_fit,
    spectral_slopes,
)


@pytest.fixture
def synthetic_power_law():
    """Synthetic f^{-2} spectrum for testing slope recovery."""
    freq = np.logspace(-4, -2, 500)  # 0.1 mHz to 10 mHz
    psd = 1e4 * freq**-2  # clean power law with slope -2
    return freq, psd


@pytest.fixture
def noisy_power_law():
    """Synthetic f^{-1.5} spectrum with multiplicative noise."""
    rng = np.random.default_rng(42)
    freq = np.logspace(-4, -2, 500)
    psd = 1e3 * freq**-1.5 * rng.lognormal(0, 0.1, 500)
    return freq, psd


class TestPowerLawFit:
    def test_recovers_slope_clean(self, synthetic_power_law):
        """Should recover slope = -2 from a clean f^{-2} spectrum."""
        freq, psd = synthetic_power_law
        result = power_law_fit(freq, psd)
        assert_allclose(result.slope, -2.0, atol=0.05)

    def test_recovers_slope_noisy(self, noisy_power_law):
        """Should recover slope ~ -1.5 from noisy data within 10%."""
        freq, psd = noisy_power_law
        result = power_law_fit(freq, psd)
        assert_allclose(result.slope, -1.5, atol=0.15)

    def test_returns_powerlaw_result(self, synthetic_power_law):
        freq, psd = synthetic_power_law
        result = power_law_fit(freq, psd)
        assert isinstance(result, PowerLawResult)
        assert result.freq_fit is None  # return_fit=False by default
        assert result.psd_fit is None

    def test_return_fit_curve(self, synthetic_power_law):
        freq, psd = synthetic_power_law
        result = power_law_fit(freq, psd, return_fit=True)
        assert result.freq_fit is not None
        assert result.psd_fit is not None
        assert len(result.freq_fit) > 0
        assert len(result.psd_fit) == len(result.freq_fit)

    def test_freq_range_subset(self, synthetic_power_law):
        """Fitting a subset of frequencies should still work."""
        freq, psd = synthetic_power_law
        result = power_law_fit(freq, psd, freq_range=(1e-4, 5e-3))
        assert_allclose(result.slope, -2.0, atol=0.05)

    def test_quadratic_degree(self, synthetic_power_law):
        """Higher-degree fit should have 3 coefficients."""
        freq, psd = synthetic_power_law
        result = power_law_fit(freq, psd, degree=2)
        assert len(result.coefficients) == 3

    def test_fit_range_different_from_freq_range(self, synthetic_power_law):
        """Fit over one range, output curve over another."""
        freq, psd = synthetic_power_law
        result = power_law_fit(
            freq,
            psd,
            freq_range=(1e-4, 5e-3),
            return_fit=True,
            fit_range=(5e-4, 1e-3),
        )
        assert result.freq_fit is not None
        assert np.all(result.freq_fit >= 5e-4)
        assert np.all(result.freq_fit <= 1e-3)


class TestSpectralSlopes:
    def test_default_bands(self, synthetic_power_law):
        """Default bands should return 'low' and 'high' slopes."""
        freq, psd = synthetic_power_law
        slopes = spectral_slopes(freq, psd)
        assert "low" in slopes
        assert "high" in slopes

    def test_slope_values(self, synthetic_power_law):
        """Both bands of a clean f^{-2} spectrum should yield slope ~ -2."""
        freq, psd = synthetic_power_law
        slopes = spectral_slopes(freq, psd)
        for name, slope in slopes.items():
            assert_allclose(slope, -2.0, atol=0.1, err_msg=f"Band {name}")

    def test_custom_bands(self, synthetic_power_law):
        freq, psd = synthetic_power_law
        bands = {"full": (freq[1], freq[-1])}
        slopes = spectral_slopes(freq, psd, bands=bands)
        assert "full" in slopes
        assert_allclose(slopes["full"], -2.0, atol=0.05)

    def test_different_slopes_per_band(self):
        """Piecewise spectrum should yield different slopes per band."""
        freq = np.logspace(-4, -2, 1000)
        breakpoint = 1e-3  # Hz
        psd = np.where(
            freq < breakpoint,
            1e4 * freq**-1.0,  # slope = -1 below breakpoint
            1e4 * breakpoint**-1.0 * (freq / breakpoint) ** -3.0,  # slope = -3 above
        )
        slopes = spectral_slopes(
            freq,
            psd,
            bands={
                "low": (1e-4, 8e-4),
                "high": (2e-3, 8e-3),
            },
        )
        assert slopes["low"] > slopes["high"]  # -1 > -3
        assert_allclose(slopes["low"], -1.0, atol=0.2)
        assert_allclose(slopes["high"], -3.0, atol=0.3)


class TestBinPowerSpectra:
    def test_basic(self):
        psd = np.arange(100, dtype=float)
        centers = [10, 50, 90]
        powers = bin_power_spectra(psd, centers, half_width=2)
        assert len(powers) == 3
        # Median of [8, 9, 10, 11, 12] = 10
        assert_allclose(powers[0], 10.0)

    def test_edge_clamping(self):
        """Bins near edges should be clamped to valid range."""
        psd = np.arange(20, dtype=float)
        powers = bin_power_spectra(psd, [1, 18], half_width=5)
        assert len(powers) == 2
        # Should not raise IndexError

    def test_single_bin(self):
        psd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        powers = bin_power_spectra(psd, [2], half_width=1)
        # Median of [2, 3, 4] = 3.0 (indices 1, 2, 3)
        assert_allclose(powers[0], 3.0, atol=0.5)

    def test_wide_bin(self):
        psd = np.ones(100)
        powers = bin_power_spectra(psd, [50], half_width=20)
        assert_allclose(powers[0], 1.0)
