"""Tests for physics-aware wave synthesis (simulate_wave_physics)."""

import numpy as np

from qp.events.catalog import WaveTemplate
from qp.signal.synthetic import simulate_wave_physics
from qp.signal.cross_correlation import ellipticity_inclination_tapered


def _make_wave(**kwargs) -> WaveTemplate:
    """Convenience: QP60 sine with 4h Gaussian envelope centered at 12h."""
    defaults = dict(
        period=3600.0, amplitude=2.0, shift=12 * 3600, decay_width=4 * 3600
    )
    defaults.update(kwargs)
    return WaveTemplate(**defaults)


def _transverse_ratio(fields: np.ndarray) -> float:
    """RMS transverse / RMS parallel power ratio."""
    b_par, b_p1, b_p2 = fields[:, 0], fields[:, 1], fields[:, 2]
    par_power = np.mean(b_par**2)
    perp_power = np.mean(b_p1**2 + b_p2**2)
    return perp_power / par_power if par_power > 0 else float("inf")


class TestAlfvenicMode:
    """Alfvénic waves should be transverse-dominated."""

    def test_high_transverse_ratio(self):
        wave = _make_wave()
        _, fields = simulate_wave_physics(
            2160, 60.0, [wave], mode="alfvenic", polarization="circular", seed=42
        )
        ratio = _transverse_ratio(fields)
        assert ratio > 3.0, f"Alfvénic transverse ratio {ratio:.1f} too low"

    def test_bpar_small(self):
        wave = _make_wave()
        _, fields = simulate_wave_physics(
            2160, 60.0, [wave], mode="alfvenic", par_leakage=0.05, seed=42
        )
        rms_par = np.sqrt(np.mean(fields[:, 0] ** 2))
        rms_perp = np.sqrt(np.mean(fields[:, 1] ** 2 + fields[:, 2] ** 2))
        assert rms_par < 0.2 * rms_perp


class TestCompressionalMode:
    """Compressional waves should be parallel-dominated."""

    def test_low_transverse_ratio(self):
        wave = _make_wave()
        _, fields = simulate_wave_physics(
            2160, 60.0, [wave], mode="compressional", par_leakage=0.05, seed=42
        )
        ratio = _transverse_ratio(fields)
        assert ratio < 0.3, f"Compressional transverse ratio {ratio:.1f} too high"


class TestPolarization:
    """Verify circular, linear, and elliptical polarization."""

    def _event_window(self, fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract the central event window where signal is strong."""
        # Use the middle third where the Gaussian envelope is strongest
        n = len(fields)
        lo, hi = n // 3, 2 * n // 3
        return fields[lo:hi, 1], fields[lo:hi, 2]

    def test_circular_high_ellipticity(self):
        wave = _make_wave()
        _, fields = simulate_wave_physics(
            2160, 60.0, [wave], polarization="circular", seed=42
        )
        p1, p2 = self._event_window(fields)
        e, _, _ = ellipticity_inclination_tapered(p1, p2)
        assert abs(e) > 0.7, f"Circular pol ellipticity {e:.2f} too low"

    def test_linear_low_ellipticity(self):
        wave = _make_wave()
        _, fields = simulate_wave_physics(
            2160, 60.0, [wave], polarization="linear", seed=42
        )
        p1, p2 = self._event_window(fields)
        e, _, _ = ellipticity_inclination_tapered(p1, p2)
        assert abs(e) < 0.2, f"Linear pol ellipticity {e:.2f} too high"

    def test_elliptical_intermediate(self):
        wave = _make_wave()
        _, fields = simulate_wave_physics(
            2160, 60.0, [wave],
            polarization="elliptical", ellipticity=0.5, seed=42,
        )
        p1, p2 = self._event_window(fields)
        e, _, _ = ellipticity_inclination_tapered(p1, p2)
        assert 0.2 < abs(e) < 0.8, f"Elliptical pol ellipticity {e:.2f} out of range"


class TestChirp:
    """Verify frequency drift is present when chirp_rate is set."""

    def test_chirped_vs_unchirped_different(self):
        """A chirped waveform should produce a different signal than unchirped."""
        wave_chirp = _make_wave(chirp_rate=5e-8, decay_width=6 * 3600)
        wave_still = _make_wave(chirp_rate=0.0, decay_width=6 * 3600)
        _, f_chirp = simulate_wave_physics(
            4320, 60.0, [wave_chirp], polarization="linear", seed=42
        )
        _, f_still = simulate_wave_physics(
            4320, 60.0, [wave_still], polarization="linear", seed=42
        )
        # The signals should be different due to chirp
        residual = np.max(np.abs(f_chirp[:, 1] - f_still[:, 1]))
        assert residual > 0.1 * np.max(np.abs(f_chirp[:, 1])), (
            f"Chirp should produce a measurably different signal (residual={residual:.4f})"
        )


class TestWaveformShapes:
    """Verify different waveform shapes produce distinct harmonic content."""

    def test_sawtooth_has_harmonics(self):
        # width=1.0 is a pure ascending sawtooth (rich in even harmonics)
        wave = WaveTemplate(
            period=3600.0, amplitude=2.0, waveform="sawtooth",
            sawtooth_width=1.0,
        )
        _, fields = simulate_wave_physics(
            720, 60.0, [wave], polarization="linear", seed=42
        )
        from qp.signal.morphology import harmonic_ratio

        hr = harmonic_ratio(fields[:, 1], dt=60.0, period_sec=3600.0)
        assert hr > 0.05, f"Sawtooth harmonic ratio {hr:.3f} too low"

    def test_sine_low_harmonics(self):
        wave = WaveTemplate(period=3600.0, amplitude=2.0, waveform="sine")
        _, fields = simulate_wave_physics(
            720, 60.0, [wave], polarization="linear", seed=42
        )
        from qp.signal.morphology import harmonic_ratio

        hr = harmonic_ratio(fields[:, 1], dt=60.0, period_sec=3600.0)
        assert hr < 0.15, f"Sine harmonic ratio {hr:.3f} too high"


class TestAsymmetricEnvelope:
    """Verify envelope asymmetry produces different rise/fall times."""

    def test_fast_rise_slow_fall(self):
        wave = _make_wave(asymmetry=0.2)
        _, fields = simulate_wave_physics(
            4320, 60.0, [wave], polarization="linear", seed=42
        )
        from qp.signal.morphology import band_envelope, envelope_rise_fall

        env = band_envelope(fields[:, 1], dt=60.0, low_hz=1/4800, high_hz=1/2400)
        result = envelope_rise_fall(env, dt=60.0)
        if result is not None:
            rise, fall, ratio = result
            assert ratio < 0.8, f"Rise/fall ratio {ratio:.2f} should be < 0.8"

    def test_symmetric_envelope(self):
        wave = _make_wave(asymmetry=0.5)
        _, fields = simulate_wave_physics(
            4320, 60.0, [wave], polarization="linear", seed=42
        )
        from qp.signal.morphology import band_envelope, envelope_rise_fall

        env = band_envelope(fields[:, 1], dt=60.0, low_hz=1/4800, high_hz=1/2400)
        result = envelope_rise_fall(env, dt=60.0)
        if result is not None:
            rise, fall, ratio = result
            assert 0.5 < ratio < 2.0, f"Symmetric ratio {ratio:.2f} out of range"


class TestSeedReproducibility:
    def test_same_seed_same_output(self):
        wave = _make_wave()
        _, a = simulate_wave_physics(1000, 60.0, [wave], seed=99)
        _, b = simulate_wave_physics(1000, 60.0, [wave], seed=99)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_output(self):
        wave = _make_wave(amplitude_jitter=0.3)
        _, a = simulate_wave_physics(1000, 60.0, [wave], seed=1)
        _, b = simulate_wave_physics(1000, 60.0, [wave], seed=2)
        assert not np.allclose(a, b)
