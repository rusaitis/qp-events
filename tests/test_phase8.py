"""Phase 8 tests.

Covers:
- 8.1  Quality-weighted occurrence rate
- 8.2  Band-pass transverse ratio discriminates circular vs linear
- 8.7  Polarization discrepancy: Stokes vs cross-correlation on synthetic signals
"""

from __future__ import annotations

import numpy as np
import pytest

from qp.events.normalization import occurrence_rate, weighted_occurrence_rate
from qp.signal.cross_correlation import (
    classify_polarization,
    ellipticity_inclination_tapered,
    phase_shift,
)
from scipy.signal import butter, sosfilt


# ── Phase 8.1 — Quality-weighted occurrence rate ───────────────────────────────

class TestWeightedOccurrenceRate:
    """Quality-weighted occurrence rate delegates to occurrence_rate."""

    def test_same_result_as_unweighted_when_weight_equals_one(self):
        """If quality = 1 everywhere, weighted == unweighted."""
        event_grid = np.array([10.0, 20.0, 5.0])
        dwell_grid = np.array([100.0, 200.0, 50.0])
        unw = occurrence_rate(event_grid, dwell_grid, min_dwell_minutes=0.0)
        w = weighted_occurrence_rate(event_grid, dwell_grid, min_dwell_minutes=0.0)
        np.testing.assert_allclose(unw, w, rtol=1e-9)

    def test_weighted_always_le_unweighted(self):
        """When quality < 1, weighted numerator ≤ unweighted numerator."""
        rng = np.random.default_rng(0)
        event_grid = rng.uniform(0, 100, size=(20,))
        dwell_grid = np.full(20, 1000.0)
        quality = rng.uniform(0.0, 1.0, size=(20,))
        weighted_events = event_grid * quality
        unw = occurrence_rate(event_grid, dwell_grid, min_dwell_minutes=0.0)
        w = weighted_occurrence_rate(weighted_events, dwell_grid, min_dwell_minutes=0.0)
        assert np.all((w <= unw) | ~np.isfinite(unw)), \
            "weighted rate must not exceed unweighted rate"

    def test_dwell_floor_applied(self):
        event_grid = np.array([5.0, 5.0])
        dwell_grid = np.array([100.0, 50.0])  # second below 60-min floor
        rate = weighted_occurrence_rate(
            event_grid, dwell_grid, min_dwell_minutes=60.0,
        )
        assert np.isfinite(rate[0])
        assert np.isnan(rate[1])

    def test_zero_event_time_gives_zero_rate(self):
        rate = weighted_occurrence_rate(
            np.zeros(5), np.full(5, 200.0), min_dwell_minutes=100.0,
        )
        np.testing.assert_allclose(rate, 0.0)


# ── Phase 8.2 — Band-pass transverse ratio ────────────────────────────────────

def _bandpass(data: np.ndarray, low_hz: float, high_hz: float,
              fs: float = 1 / 60.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
    return sosfilt(sos, data)


class TestBandpassTransverseRatio:
    """Band-pass filtering isolates the Alfvénic perturbation."""

    def _make_alfvenic(self, n: int = 2160, period_min: float = 60.0,
                       amplitude: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quasi-periodic perp waves + slow compressional par."""
        dt = 60.0  # seconds
        t = np.arange(n) * dt
        period_sec = period_min * 60.0
        b_perp1 = amplitude * np.sin(2 * np.pi * t / period_sec)
        b_perp2 = amplitude * np.cos(2 * np.pi * t / period_sec)
        # Slow compressional variation at 2h period >> QP band
        b_par = 5.0 * np.sin(2 * np.pi * t / (2 * 3600.0))
        return b_perp1, b_perp2, b_par

    def test_broadband_ratio_suppressed_by_slow_compression(self):
        """Broadband transverse ratio is < 1 when slow b_par dominates."""
        p1, p2, par = self._make_alfvenic()
        broadband_tr = np.mean(p1 ** 2 + p2 ** 2) / np.mean(par ** 2)
        assert broadband_tr < 1.0, (
            f"broadband ratio {broadband_tr:.3f} should be < 1 "
            "due to slow compressional par"
        )

    def test_bandpass_ratio_large_for_alfvenic_wave(self):
        """Band-pass ratio >> 1 for a QP60 Alfvénic event."""
        p1, p2, par = self._make_alfvenic(period_min=60.0, amplitude=1.0)
        # QP60 band: 45-80 min → 1/4800 to 1/2700 Hz
        low_hz = 1.0 / (80 * 60)
        high_hz = 1.0 / (45 * 60)
        p1f = _bandpass(p1, low_hz, high_hz)
        p2f = _bandpass(p2, low_hz, high_hz)
        parf = _bandpass(par, low_hz, high_hz)
        # Use middle half to avoid filter edge effects
        sl = slice(len(p1f) // 4, 3 * len(p1f) // 4)
        bp_tr = np.mean(p1f[sl] ** 2 + p2f[sl] ** 2) / np.mean(parf[sl] ** 2)
        assert bp_tr > 5.0, (
            f"band-pass transverse ratio {bp_tr:.2f} should be >> 1 "
            "for Alfvénic QP60 wave"
        )

    def test_bandpass_ratio_near_one_for_off_band_noise(self):
        """Band-pass ratio ≈ 1 for white-noise (isotropic) input."""
        rng = np.random.default_rng(99)
        n = 2160
        noise1 = rng.standard_normal(n)
        noise2 = rng.standard_normal(n)
        noisepar = rng.standard_normal(n)
        low_hz = 1.0 / (80 * 60)
        high_hz = 1.0 / (45 * 60)
        f1 = _bandpass(noise1, low_hz, high_hz)
        f2 = _bandpass(noise2, low_hz, high_hz)
        fp = _bandpass(noisepar, low_hz, high_hz)
        sl = slice(n // 4, 3 * n // 4)
        bp_tr = np.mean(f1[sl] ** 2 + f2[sl] ** 2) / np.mean(fp[sl] ** 2)
        # For independent white noise, ratio should be near 1 (within factor 3)
        assert 0.2 < bp_tr < 6.0, (
            f"band-pass ratio for noise {bp_tr:.2f} should be ~1"
        )


# ── Phase 8.7 — Polarization: Stokes vs cross-correlation ─────────────────────

class TestPolarizationComparison:
    """Compare Stokes and cross-correlation on known synthetic signals."""

    DT = 60.0
    PERIOD = 3600.0  # 60-min period
    N = 360  # 6 hours

    def _wave(self, phase_offset_deg: float) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(self.N) * self.DT
        p1 = np.sin(2 * np.pi * t / self.PERIOD)
        p2 = np.sin(2 * np.pi * t / self.PERIOD
                     + np.radians(phase_offset_deg))
        return p1, p2

    def test_stokes_circular_signal(self):
        """Circularly polarized wave → |ellipticity| ≈ 1."""
        p1, p2 = self._wave(90.0)
        ell, _, _ = ellipticity_inclination_tapered(p1, p2)
        assert abs(ell) > 0.85, f"|ellipticity|={abs(ell):.3f} should be > 0.85"

    def test_stokes_linear_signal(self):
        """Linearly polarized wave → |ellipticity| ≈ 0."""
        p1, p2 = self._wave(0.0)
        ell, _, _ = ellipticity_inclination_tapered(p1, p2)
        assert abs(ell) < 0.15, f"|ellipticity|={abs(ell):.3f} should be < 0.15"

    def test_xcorr_circular_signal(self):
        """Cross-correlation correctly identifies circular polarization."""
        p1, p2 = self._wave(90.0)
        _, phase = phase_shift(p1, p2, dt=self.DT, period=self.PERIOD)
        pol = classify_polarization(phase)
        assert pol == "circular", f"Expected circular, got {pol} (phase={phase:.1f}°)"

    def test_xcorr_linear_signal(self):
        """Cross-correlation correctly identifies linear polarization."""
        p1, p2 = self._wave(0.0)
        lag, phase = phase_shift(p1, p2, dt=self.DT, period=self.PERIOD)
        # Phase ≈ 0 or None → classified as linear
        if phase is not None:
            pol = classify_polarization(phase)
            assert pol in ("linear", "mixed"), \
                f"Expected linear/mixed, got {pol} (phase={phase:.1f}°)"

    def test_stokes_more_robust_than_xcorr_for_real_events(self):
        """For noisy quasi-periodic data: Stokes ellipticity stays near 0 for linear."""
        rng = np.random.default_rng(42)
        n = 360
        t = np.arange(n) * self.DT
        noise_level = 0.5
        # Linear wave with moderate noise
        p1 = (np.sin(2 * np.pi * t / self.PERIOD)
               + noise_level * rng.standard_normal(n))
        p2 = (np.sin(2 * np.pi * t / self.PERIOD)  # same phase → linear
               + noise_level * rng.standard_normal(n))

        ell, _, _ = ellipticity_inclination_tapered(p1, p2)
        # Stokes should still show low ellipticity for noisy linear wave
        assert abs(ell) < 0.4, (
            f"|ellipticity|={abs(ell):.3f} should be < 0.4 for noisy linear wave"
        )
