"""Tests for qp.signal.synthetic — synthetic signal generation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.events.catalog import WaveTemplate
from qp.signal.synthetic import (
    generate_long_signal,
    simulate_multi_component,
    simulate_signal,
)


class TestSimulateSignal:
    """Tests for simulate_signal()."""

    def test_output_shape(self):
        t, y = simulate_signal(n_samples=1000, dt=60.0)
        assert t.shape == (1000,)
        assert y.shape == (1000,)

    def test_time_starts_at_zero(self):
        t, _ = simulate_signal(n_samples=100, dt=60.0)
        assert t[0] == 0.0

    def test_time_spacing(self):
        t, _ = simulate_signal(n_samples=100, dt=30.0)
        dt_actual = np.diff(t)
        assert_allclose(dt_actual, 30.0)

    def test_no_waves_gives_zeros(self):
        _, y = simulate_signal(n_samples=100, dt=60.0, waves=None, noise_sigma=0.0)
        assert_allclose(y, 0.0)

    def test_single_sine_wave(self):
        """A single sine wave should produce the correct peak-to-peak amplitude."""
        wave = WaveTemplate(period=3600.0, amplitude=5.0, phase=0.0)
        t, y = simulate_signal(n_samples=2160, dt=60.0, waves=[wave])
        # Peak-to-peak should be ~2*amplitude = 10
        assert_allclose(np.max(y) - np.min(y), 10.0, atol=0.5)

    def test_injected_wave_frequency(self):
        """FFT of injected wave should peak at the correct frequency."""
        period = 3600.0  # 60 minutes
        wave = WaveTemplate(period=period, amplitude=2.0)
        t, y = simulate_signal(n_samples=2160, dt=60.0, waves=[wave])

        freq = np.fft.rfftfreq(len(y), d=60.0)
        power = np.abs(np.fft.rfft(y)) ** 2
        peak_freq = freq[np.argmax(power[1:]) + 1]
        expected_freq = 1.0 / period

        assert_allclose(peak_freq, expected_freq, rtol=0.05)

    def test_noise_only(self):
        """Noise-only signal should have zero mean and correct std."""
        _, y = simulate_signal(
            n_samples=10000,
            dt=60.0,
            noise_sigma=1.0,
            seed=42,
        )
        assert abs(np.mean(y)) < 0.1
        assert_allclose(np.std(y), 1.0, atol=0.1)

    def test_seed_reproducibility(self):
        _, y1 = simulate_signal(n_samples=100, dt=60.0, noise_sigma=1.0, seed=42)
        _, y2 = simulate_signal(n_samples=100, dt=60.0, noise_sigma=1.0, seed=42)
        assert_allclose(y1, y2)

    def test_different_seeds_differ(self):
        _, y1 = simulate_signal(n_samples=100, dt=60.0, noise_sigma=1.0, seed=42)
        _, y2 = simulate_signal(n_samples=100, dt=60.0, noise_sigma=1.0, seed=99)
        assert not np.allclose(y1, y2)

    def test_gaussian_envelope(self):
        """Wave with Gaussian decay should have smaller amplitude at edges."""
        wave = WaveTemplate(
            period=600.0,
            amplitude=1.0,
            decay_width=3000.0,
            shift=5000.0,
        )
        t, y = simulate_signal(n_samples=200, dt=60.0, waves=[wave])
        # Amplitude near center (t~5000s) should be larger than at edges
        center_idx = np.argmin(np.abs(t - 5000.0))
        center_amp = np.max(np.abs(y[center_idx - 5 : center_idx + 5]))
        edge_amp = np.max(np.abs(y[:10]))
        assert center_amp > edge_amp

    def test_cutoff(self):
        """Wave with cutoff should be zero outside the window."""
        wave = WaveTemplate(
            period=600.0,
            amplitude=1.0,
            cutoff=(3000.0, 6000.0),
        )
        t, y = simulate_signal(n_samples=200, dt=60.0, waves=[wave])
        # Before cutoff start should be zero
        before = y[t < 2900]
        assert_allclose(before, 0.0)
        # After cutoff end should be zero
        after = y[t > 6100]
        assert_allclose(after, 0.0)

    def test_sawtooth_waveform(self):
        wave = WaveTemplate(period=600.0, amplitude=1.0, waveform="sawtooth")
        t, y = simulate_signal(n_samples=100, dt=60.0, waves=[wave])
        assert np.max(np.abs(y)) > 0

    def test_square_waveform(self):
        wave = WaveTemplate(period=600.0, amplitude=1.0, waveform="square")
        t, y = simulate_signal(n_samples=100, dt=60.0, waves=[wave])
        assert np.max(np.abs(y)) > 0

    def test_unknown_waveform_raises(self):
        wave = WaveTemplate(period=600.0, waveform="chirp")
        with pytest.raises(ValueError, match="Unknown waveform"):
            simulate_signal(n_samples=100, dt=60.0, waves=[wave])


class TestSimulateMultiComponent:
    """Tests for simulate_multi_component()."""

    def test_output_shape(self):
        t, fields = simulate_multi_component(n_samples=500, dt=60.0)
        assert t.shape == (500,)
        assert fields.shape == (500, 4)

    def test_btot_column(self):
        """Fourth column should be the magnitude of the first three."""
        wave = WaveTemplate(period=3600.0, amplitude=1.0)
        t, fields = simulate_multi_component(
            n_samples=500,
            dt=60.0,
            waves=[wave],
        )
        expected_btot = np.linalg.norm(fields[:, :3], axis=1)
        assert_allclose(fields[:, 3], expected_btot)

    def test_phase_offsets_applied(self):
        """Different components should have different phases."""
        wave = WaveTemplate(period=3600.0, amplitude=1.0, phase=0.0)
        _, fields = simulate_multi_component(
            n_samples=500,
            dt=60.0,
            waves=[wave],
            phase_offsets=(0.0, np.pi / 4, np.pi / 2),
        )
        # Components should not be identical
        assert not np.allclose(fields[:, 0], fields[:, 1])
        assert not np.allclose(fields[:, 1], fields[:, 2])

    def test_no_waves_no_noise_is_zero(self):
        _, fields = simulate_multi_component(
            n_samples=100,
            dt=60.0,
            noise_sigma=0.0,
        )
        assert_allclose(fields[:, :3], 0.0)


class TestGenerateLongSignal:
    """Tests for generate_long_signal()."""

    def test_output_length(self):
        t, y = generate_long_signal(duration_days=1.0, dt=60.0)
        expected_n = int(1.0 * 86400 / 60)
        assert len(t) == expected_n
        assert len(y) == expected_n

    def test_signal_not_zero(self):
        _, y = generate_long_signal(duration_days=1.0, dt=60.0)
        assert np.std(y) > 0

    def test_reproducible(self):
        _, y1 = generate_long_signal(duration_days=1.0, dt=60.0, seed=42)
        _, y2 = generate_long_signal(duration_days=1.0, dt=60.0, seed=42)
        assert_allclose(y1, y2)


class TestRound3Hardening:
    """Round-3 envelope and harmonic-model additions."""

    def test_lognormal_envelope_skewed(self):
        """Log-normal envelope decays slower on the right tail than left."""
        from qp.events.catalog import WaveTemplate
        from qp.signal.synthetic import _generate_waveform

        t = np.arange(2160) * 60.0  # 36 h
        wave = WaveTemplate(
            period=3600.0, amplitude=1.0, decay_width=2 * 3600.0,
            shift=18 * 3600.0, envelope_shape="lognormal",
        )
        y = _generate_waveform(t, wave)
        # Energy left of peak vs right of peak
        peak_idx = int(np.argmax(np.abs(y)))
        left_energy = float(np.sum(y[:peak_idx] ** 2))
        right_energy = float(np.sum(y[peak_idx:] ** 2))
        assert right_energy > 1.5 * left_energy

    def test_rayleigh_envelope_zero_at_left_edge(self):
        from qp.events.catalog import WaveTemplate
        from qp.signal.synthetic import _generate_waveform

        t = np.arange(2160) * 60.0
        wave = WaveTemplate(
            period=3600.0, amplitude=1.0, decay_width=2 * 3600.0,
            shift=18 * 3600.0, envelope_shape="rayleigh",
        )
        y = _generate_waveform(t, wave)
        # Rayleigh envelope = 0 at t = -decay_width = 16h ⇒ index 960
        assert abs(y[960]) < 1e-6

    def test_sawtooth_truncated_phase_locked(self):
        """Sawtooth-truncated harmonics share phase with the fundamental."""
        from qp.events.catalog import WaveTemplate
        from qp.signal.synthetic import _generate_waveform

        t = np.arange(2160) * 60.0
        # Two waves with same parameters and seed should be identical;
        # crucially, repeating them must NOT produce random phase
        # variation in the harmonic content.
        rng = np.random.default_rng(7)
        wave = WaveTemplate(
            period=3600.0, amplitude=1.0,
            harmonic_content=0.5,
            harmonic_model="sawtooth_truncated",
        )
        y1 = _generate_waveform(t, wave, rng=rng)
        rng = np.random.default_rng(99)  # different RNG
        y2 = _generate_waveform(t, wave, rng=rng)
        # Phase-locked harmonics → output is deterministic regardless
        # of the RNG (it does not read from it).
        np.testing.assert_array_equal(y1, y2)

    def test_linear_2f_random_phase(self):
        """Linear-2f harmonic model uses a random phase per call."""
        from qp.events.catalog import WaveTemplate
        from qp.signal.synthetic import _generate_waveform

        t = np.arange(2160) * 60.0
        wave = WaveTemplate(
            period=3600.0, amplitude=1.0,
            harmonic_content=0.5, harmonic_model="linear_2f",
        )
        y1 = _generate_waveform(t, wave, rng=np.random.default_rng(1))
        y2 = _generate_waveform(t, wave, rng=np.random.default_rng(2))
        # Different RNGs → different waveforms (random harmonic phase).
        assert not np.allclose(y1, y2)
