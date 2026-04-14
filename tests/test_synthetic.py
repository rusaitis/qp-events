"""Tests for qp.signal.synthetic — synthetic signal generation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.events.catalog import WaveTemplate
from qp.signal.synthetic import simulate_signal, simulate_wave_physics


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


class TestSimulateWavePhysics:
    """Tests for simulate_wave_physics() — 3-component field generator."""

    @staticmethod
    def _bare_sine() -> WaveTemplate:
        # Continuous sine: no envelope, no jitter, no harmonics — so the
        # phase relationship between B_perp1 and B_perp2 is exact.
        return WaveTemplate(
            period=3600.0,
            amplitude=1.0,
            harmonic_content=0.0,
            amplitude_jitter=0.0,
        )

    def test_circular_polarization_phase_lag(self):
        """B_perp2 leads B_perp1 by π/2 (i.e. perp2 = cos when perp1 = sin)."""
        wave = self._bare_sine()
        _, fields = simulate_wave_physics(
            n_samples=2160, dt=60.0, waves=[wave],
            mode="alfvenic", polarization="circular",
            par_leakage=0.0, seed=0,
        )
        b_perp1, b_perp2 = fields[:, 1], fields[:, 2]
        # Inner product over many cycles: <sin·sin> ≈ 0.5, <sin·cos> ≈ 0.
        n = len(b_perp1)
        assert abs(np.dot(b_perp1, b_perp2) / n) < 0.02
        # Auto-power equal to within sampling noise.
        p1 = np.dot(b_perp1, b_perp1) / n
        p2 = np.dot(b_perp2, b_perp2) / n
        assert_allclose(p1, p2, rtol=0.02)

    def test_linear_polarization_zero_perp2(self):
        """B_perp2 is identically zero under linear polarization."""
        wave = self._bare_sine()
        _, fields = simulate_wave_physics(
            n_samples=1000, dt=60.0, waves=[wave],
            mode="alfvenic", polarization="linear",
            par_leakage=0.0, seed=0,
        )
        assert_allclose(fields[:, 2], 0.0)
        assert np.max(np.abs(fields[:, 1])) > 0.5

    def test_elliptical_negative_is_left_handed(self):
        """ellipticity=-1 produces the negative of the circular B_perp2."""
        wave = self._bare_sine()
        _, fields_r = simulate_wave_physics(
            n_samples=1000, dt=60.0, waves=[wave],
            mode="alfvenic", polarization="circular",
            par_leakage=0.0, seed=0,
        )
        _, fields_l = simulate_wave_physics(
            n_samples=1000, dt=60.0, waves=[wave],
            mode="alfvenic", polarization="elliptical", ellipticity=-1.0,
            par_leakage=0.0, seed=0,
        )
        # B_perp1 identical, B_perp2 sign-flipped.
        assert_allclose(fields_l[:, 1], fields_r[:, 1])
        assert_allclose(fields_l[:, 2], -fields_r[:, 2])

    def test_compressional_concentrates_power_in_bpar(self):
        """Compressional mode places ≥90% of perturbation power in B_par."""
        wave = self._bare_sine()
        _, fields = simulate_wave_physics(
            n_samples=2160, dt=60.0, waves=[wave],
            mode="compressional", polarization="circular",
            par_leakage=0.05, seed=0,
        )
        p_par = float(np.var(fields[:, 0]))
        p_perp = float(np.var(fields[:, 1]) + np.var(fields[:, 2]))
        # leakage=0.05 ⇒ perp power ≈ 2·leakage²·par power ≈ 0.5%
        assert p_par / (p_par + p_perp) > 0.90

    def test_par_leakage_scalar_is_deterministic(self):
        """A scalar par_leakage applies the same fraction to every event."""
        wave = self._bare_sine()
        _, fields = simulate_wave_physics(
            n_samples=2160, dt=60.0, waves=[wave],
            mode="alfvenic", polarization="linear",
            par_leakage=0.05, seed=0,
        )
        # Linear ⇒ B_perp1 = w1, B_par = 0.05·w1 ⇒ peak ratio = 0.05.
        peak_perp = float(np.max(np.abs(fields[:, 1])))
        peak_par = float(np.max(np.abs(fields[:, 0])))
        assert_allclose(peak_par / peak_perp, 0.05, rtol=1e-6)
