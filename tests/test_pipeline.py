"""Tests for qp.signal.pipeline — spectral analysis pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.signal.pipeline import SpectralResult, analyze_segment


@pytest.fixture
def sine_60min():
    """A pure 60-minute sinusoid sampled at 1-min intervals for 36 hours."""
    dt = 60.0  # seconds
    n_hours = 36
    n = int(n_hours * 3600 / dt)  # 2160 samples
    t = np.arange(n) * dt
    period = 60 * 60  # 60 minutes in seconds
    signal = 2.0 * np.sin(2 * np.pi * t / period)
    return signal, dt, period


@pytest.fixture
def noisy_signal():
    """White noise with no dominant frequency."""
    rng = np.random.default_rng(42)
    n = 2000
    return rng.normal(0, 1, n), 60.0


class TestAnalyzeSegment:
    """Tests for the main pipeline function."""

    def test_returns_spectral_result(self, sine_60min):
        signal, dt, _ = sine_60min
        result = analyze_segment(signal, dt=dt)
        assert isinstance(result, SpectralResult)

    def test_freq_array_positive(self, sine_60min):
        signal, dt, _ = sine_60min
        result = analyze_segment(signal, dt=dt)
        assert np.all(result.freq >= 0)

    def test_psd_positive(self, sine_60min):
        signal, dt, _ = sine_60min
        result = analyze_segment(signal, dt=dt)
        assert np.all(result.psd >= 0)

    def test_peak_at_signal_frequency(self, sine_60min):
        """PSD peak should be near the injected 60-min period."""
        signal, dt, period = sine_60min
        result = analyze_segment(signal, dt=dt)

        expected_freq = 1.0 / period  # Hz
        peak_idx = np.argmax(result.psd[1:]) + 1  # skip DC
        peak_freq = result.freq[peak_idx]

        # Peak should be within 10% of expected frequency
        assert_allclose(peak_freq, expected_freq, rtol=0.1)

    def test_power_ratio_above_one_at_peak(self, sine_60min):
        """Power ratio should exceed 1 at the signal frequency."""
        signal, dt, period = sine_60min
        result = analyze_segment(signal, dt=dt)

        expected_freq = 1.0 / period
        freq_idx = np.argmin(np.abs(result.freq - expected_freq))
        assert result.power_ratio[freq_idx] > 1.0

    def test_background_below_peak(self, sine_60min):
        """Background estimate should be below the PSD peak."""
        signal, dt, period = sine_60min
        result = analyze_segment(signal, dt=dt)

        expected_freq = 1.0 / period
        freq_idx = np.argmin(np.abs(result.freq - expected_freq))
        assert result.background[freq_idx] < result.psd[freq_idx]

    def test_detrended_output(self, sine_60min):
        signal, dt, _ = sine_60min
        result = analyze_segment(signal, dt=dt)
        assert result.detrended.shape == signal.shape
        assert result.trend.shape == signal.shape

    def test_detrend_removes_mean(self, sine_60min):
        """Detrended signal should have near-zero mean."""
        signal, dt, _ = sine_60min
        # Add a large DC offset
        signal_with_offset = signal + 100.0
        result = analyze_segment(signal_with_offset, dt=dt)
        assert abs(np.mean(result.detrended)) < 1.0

    def test_noise_power_ratio_near_one(self, noisy_signal):
        """White noise should have power ratio ~1 everywhere."""
        signal, dt = noisy_signal
        result = analyze_segment(signal, dt=dt)
        # Median power ratio of noise should be near 1 (within 50%)
        median_ratio = np.median(result.power_ratio[1:])  # skip DC
        assert 0.3 < median_ratio < 3.0


class TestSpectrogramOption:
    """Tests for the optional spectrogram computation."""

    def test_spectrogram_disabled_by_default(self, sine_60min):
        signal, dt, _ = sine_60min
        result = analyze_segment(signal, dt=dt)
        assert result.spectrogram_result is None

    def test_spectrogram_enabled(self, sine_60min):
        signal, dt, _ = sine_60min
        result = analyze_segment(signal, dt=dt, include_spectrogram=True)
        assert result.spectrogram_result is not None
        freq_s, time_s, sxx = result.spectrogram_result
        assert freq_s.ndim == 1
        assert time_s.ndim == 1
        assert sxx.ndim == 2


class TestCWTOption:
    """Tests for the optional CWT computation."""

    def test_cwt_disabled_by_default(self, sine_60min):
        signal, dt, _ = sine_60min
        result = analyze_segment(signal, dt=dt)
        assert result.cwt_result is None

    def test_cwt_enabled(self, sine_60min):
        signal, dt, _ = sine_60min
        result = analyze_segment(
            signal,
            dt=dt,
            include_cwt=True,
            cwt_n_freqs=50,
        )
        assert result.cwt_result is not None
        freq_c, time_c, power_c = result.cwt_result
        assert freq_c.ndim == 1
        assert time_c.ndim == 1
        assert power_c.ndim == 2
        assert power_c.shape == (50, len(signal))


class TestSpectralResultDataclass:
    """Tests for the SpectralResult frozen dataclass."""

    def test_frozen(self, sine_60min):
        signal, dt, _ = sine_60min
        result = analyze_segment(signal, dt=dt)
        with pytest.raises(AttributeError):
            result.freq = np.array([1, 2, 3])
