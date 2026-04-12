"""Tests for qp.signal.morphology waveform morphology analysis."""

from __future__ import annotations

import numpy as np
import pytest

from qp.signal.morphology import (
    amplitude_growth_rate,
    band_envelope,
    envelope_rise_fall,
    freq_drift_rate,
    harmonic_ratio,
    inter_cycle_coherence,
    instantaneous_frequency,
)

DT = 60.0  # 1-minute samples
FS = 1.0 / DT
PERIOD_SEC = 3600.0  # 60-min wave
N = 720  # 12 hours of samples


def _sine(n: int = N, period: float = PERIOD_SEC, dt: float = DT,
          phase: float = 0.0, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n) * dt
    return amplitude * np.sin(2 * np.pi * t / period + phase)


def _sawtooth(n: int = N, period: float = PERIOD_SEC, dt: float = DT) -> np.ndarray:
    from scipy.signal import sawtooth as sp_sawtooth
    t = np.arange(n) * dt
    return sp_sawtooth(2 * np.pi * t / period)


def _gaussian_envelope_sine(n: int = N, period: float = PERIOD_SEC,
                              dt: float = DT) -> np.ndarray:
    t = np.arange(n) * dt
    centre = t[n // 2]
    width = n * dt / 4.0
    envelope = np.exp(-0.5 * ((t - centre) / width) ** 2)
    return envelope * np.sin(2 * np.pi * t / period)


def _chirped_sine(n: int = N, period_start: float = PERIOD_SEC,
                   chirp_rate_hz_s: float = 1e-7, dt: float = DT) -> np.ndarray:
    """Linearly chirped sine: f(t) = f0 + α*t."""
    t = np.arange(n) * dt
    f0 = 1.0 / period_start
    phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate_hz_s * t ** 2)
    return np.sin(phase)


# ---------------------------------------------------------------------------
# band_envelope
# ---------------------------------------------------------------------------

class TestBandEnvelope:
    LOW_HZ = 1.0 / (80 * 60)
    HIGH_HZ = 1.0 / (45 * 60)

    def test_envelope_nonnegative(self):
        x = _sine()
        env = band_envelope(x, DT, self.LOW_HZ, self.HIGH_HZ)
        assert np.all(env >= 0)

    def test_envelope_same_length(self):
        x = _sine()
        env = band_envelope(x, DT, self.LOW_HZ, self.HIGH_HZ)
        assert len(env) == len(x)

    def test_envelope_peak_near_amplitude(self):
        x = _sine(amplitude=2.0)
        env = band_envelope(x, DT, self.LOW_HZ, self.HIGH_HZ)
        # Allow generous tolerance due to filter transients
        sl = slice(N // 4, 3 * N // 4)
        assert np.mean(env[sl]) == pytest.approx(2.0, abs=0.3)


# ---------------------------------------------------------------------------
# instantaneous_frequency
# ---------------------------------------------------------------------------

class TestInstantaneousFrequency:
    def test_pure_sine_returns_correct_frequency(self):
        x = _sine()
        # Band-pass first to isolate the frequency
        from qp.signal.morphology import _bandpass
        low = 1.0 / (80 * 60)
        high = 1.0 / (45 * 60)
        xf = _bandpass(x, low, high, FS)
        f_inst = instantaneous_frequency(xf, DT)
        # Middle half avoids edge transients
        sl = slice(N // 4, 3 * N // 4)
        f_expected = 1.0 / PERIOD_SEC
        assert np.median(f_inst[sl]) == pytest.approx(f_expected, rel=0.05)

    def test_same_length_as_input(self):
        x = _sine()
        f = instantaneous_frequency(x, DT)
        assert len(f) == len(x)


# ---------------------------------------------------------------------------
# harmonic_ratio
# ---------------------------------------------------------------------------

class TestHarmonicRatio:
    def test_pure_sine_low_harmonic(self):
        x = _sine()
        ratio = harmonic_ratio(x, DT, PERIOD_SEC)
        # Pure sine should have very low harmonic content
        assert ratio < 0.05

    def test_sawtooth_higher_harmonic(self):
        x = _sawtooth()
        ratio = harmonic_ratio(x, DT, PERIOD_SEC)
        # Sawtooth has significant 2nd harmonic (≈1/4 power)
        assert ratio > 0.10

    def test_zero_period_guard(self):
        x = _sine()
        # Should not crash for any valid period
        ratio = harmonic_ratio(x, DT, 60.0)  # 1-minute period (just outside Nyquist range)
        assert np.isfinite(ratio)


# ---------------------------------------------------------------------------
# envelope_rise_fall
# ---------------------------------------------------------------------------

class TestEnvelopeRiseFall:
    def test_gaussian_envelope_symmetric(self):
        x = _gaussian_envelope_sine()
        from qp.signal.morphology import band_envelope
        env = band_envelope(x, DT, 1.0 / (80 * 60), 1.0 / (45 * 60))
        result = envelope_rise_fall(env, DT)
        assert result is not None
        _, _, ratio = result
        # Gaussian envelope is symmetric → ratio ≈ 1
        assert 0.3 < ratio < 3.0, f"ratio={ratio:.2f} expected ~1 for symmetric envelope"

    def test_sharp_onset_ratio_lt_one(self):
        """An envelope that rises fast and falls slowly has ratio < 1."""
        n = N
        env = np.zeros(n)
        # Fast rise in first 30 samples, slow decay over remaining 690
        env[:30] = np.linspace(0, 1, 30)
        env[30:] = np.linspace(1, 0, n - 30)
        result = envelope_rise_fall(env, DT)
        assert result is not None
        _, _, ratio = result
        # Fast rise, slow fall → rise_time < fall_time → ratio < 1
        assert ratio < 1.0

    def test_returns_none_for_flat_envelope(self):
        env = np.ones(100)
        assert envelope_rise_fall(env, DT) is None

    def test_returns_none_for_short_input(self):
        assert envelope_rise_fall(np.array([1.0, 2.0, 1.0]), DT) is None


# ---------------------------------------------------------------------------
# amplitude_growth_rate
# ---------------------------------------------------------------------------

class TestAmplitudeGrowthRate:
    def test_growing_sine_positive_slope(self):
        # Use exponential growth (1→e^2 ≈ 7.4) so signal clearly grows each period
        n = N
        t = np.arange(n) * DT
        amp = np.exp(2.0 * t / t[-1])
        x = amp * np.sin(2 * np.pi * t / PERIOD_SEC)
        from qp.signal.morphology import band_envelope
        env = band_envelope(x, DT, 1.0 / (80 * 60), 1.0 / (45 * 60))
        # Middle 2/3 avoids edge transients
        mid = slice(N // 6, 5 * N // 6)
        rate = amplitude_growth_rate(env[mid], DT, PERIOD_SEC)
        assert rate > 0, f"Expected positive growth, got {rate:.3f} dB/period"

    def test_decaying_sine_negative_slope(self):
        # Use exponential decay (3:1 amplitude drop) over 12 h to ensure
        # the per-cycle envelope trend is clearly negative
        n = N
        t = np.arange(n) * DT
        amp = 3.0 * np.exp(-t / (t[-1] / 2.0))  # 3→~0.4 over full window
        x = amp * np.sin(2 * np.pi * t / PERIOD_SEC)
        from qp.signal.morphology import band_envelope
        env = band_envelope(x, DT, 1.0 / (80 * 60), 1.0 / (45 * 60))
        # Use middle 2/3 to avoid filter edge effects
        mid = slice(N // 6, 5 * N // 6)
        env_mid = env[mid]
        rate = amplitude_growth_rate(env_mid, DT, PERIOD_SEC)
        assert rate < 0, f"Expected negative growth, got {rate:.3f} dB/period"

    def test_steady_sine_near_zero(self):
        x = _sine()
        from qp.signal.morphology import band_envelope
        env = band_envelope(x, DT, 1.0 / (80 * 60), 1.0 / (45 * 60))
        rate = amplitude_growth_rate(env, DT, PERIOD_SEC)
        assert abs(rate) < 2.0, f"Steady sine should have ~0 growth rate, got {rate:.3f}"

    def test_too_short_returns_zero(self):
        env = np.array([1.0, 1.0, 1.0])
        assert amplitude_growth_rate(env, DT, PERIOD_SEC) == 0.0


# ---------------------------------------------------------------------------
# inter_cycle_coherence
# ---------------------------------------------------------------------------

class TestInterCycleCoherence:
    def test_pure_sine_high_coherence(self):
        x = _sine()
        coh = inter_cycle_coherence(x, DT, PERIOD_SEC)
        assert coh > 0.8, f"Pure sine should have high coherence, got {coh:.3f}"

    def test_white_noise_lower_coherence(self):
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(N)
        coh = inter_cycle_coherence(noise, DT, PERIOD_SEC)
        assert coh < 0.5, f"White noise should have low coherence, got {coh:.3f}"

    def test_result_in_valid_range(self):
        x = _sine() + 0.5 * np.random.default_rng(0).standard_normal(N)
        coh = inter_cycle_coherence(x, DT, PERIOD_SEC)
        assert -1.0 <= coh <= 1.0


# ---------------------------------------------------------------------------
# freq_drift_rate
# ---------------------------------------------------------------------------

class TestFreqDriftRate:
    LOW = 1.0 / (80 * 60)
    HIGH = 1.0 / (45 * 60)

    def test_pure_sine_near_zero_drift(self):
        x = _sine()
        alpha = freq_drift_rate(x, DT, self.LOW, self.HIGH)
        # Pure sine → negligible drift relative to band width
        band_hz = self.HIGH - self.LOW
        assert abs(alpha) < band_hz / (N * DT), (
            f"Drift {alpha:.2e} Hz/s should be << band width / duration"
        )

    def test_chirped_sine_nonzero_drift(self):
        # Build a signal that sweeps from the low edge to the high edge of
        # the QP60 band so the upward chirp is unambiguous.
        n = N
        t = np.arange(n) * DT
        duration = t[-1]
        f_start = self.LOW * 1.05   # just inside lower edge
        f_end = self.HIGH * 0.95     # just inside upper edge
        chirp_rate = (f_end - f_start) / duration  # known positive Hz/s
        phase = 2 * np.pi * (f_start * t + 0.5 * chirp_rate * t ** 2)
        x = np.sin(phase)
        alpha = freq_drift_rate(x, DT, self.LOW, self.HIGH)
        # Signal sweeps upward → freq_drift should be positive
        assert alpha > 0, f"Expected positive chirp rate, got {alpha:.3e}"

    def test_finite_for_short_input(self):
        x = np.ones(4)
        alpha = freq_drift_rate(x, DT, self.LOW, self.HIGH)
        assert np.isfinite(alpha)
