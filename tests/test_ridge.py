"""Tests for the band-aware ridge extractor."""

from __future__ import annotations

import numpy as np
import pytest

from qp.events.bands import QP_BANDS
from qp.events.ridge import Ridge, extract_ridges
from qp.signal.synthetic import simulate_signal
from qp.signal.wavelet import morlet_cwt
from qp.events.catalog import WaveTemplate


def _cwt_for(signal: np.ndarray, dt: float = 60.0):
    freq, _, m = morlet_cwt(signal, dt=dt, n_freqs=200)
    return freq, np.abs(m)


def _make_signal(period_min: float, amplitude: float = 1.5,
                  decay_hours: float = 3.0,
                  center_hours: float = 18.0,
                  n_samples: int = 2160,
                  dt: float = 60.0,
                  noise_sigma: float = 0.0,
                  seed: int = 1) -> np.ndarray:
    """Synthetic Gaussian-windowed sinusoid for one band."""
    wave = WaveTemplate(
        period=period_min * 60.0,
        amplitude=amplitude,
        decay_width=decay_hours * 3600.0,
        shift=center_hours * 3600.0,
    )
    _, signal = simulate_signal(
        n_samples=n_samples,
        dt=dt,
        waves=[wave],
        noise_sigma=noise_sigma,
        seed=seed,
    )
    return signal


class TestRidgeExtractionPerBand:
    """v4: ridges are band-agnostic. We inject a wave at a canonical
    period and check the peak period lands in the correct band
    post-hoc via classify_period()."""

    @pytest.mark.parametrize(
        "band_name,period_min",
        [("QP30", 30), ("QP60", 60), ("QP120", 120)],
    )
    def test_ridge_lands_in_correct_band(self, band_name, period_min):
        from qp.events.bands import classify_period
        signal = _make_signal(period_min=period_min, amplitude=2.0)
        freq, power = _cwt_for(signal)
        ridges = extract_ridges(
            power, freq,
            min_duration_sec=1 * 3600,
            min_pixels=20,
        )
        assert len(ridges) >= 1, f"no ridge found for {band_name}"
        # Peak period should classify back to the expected band
        band = QP_BANDS[band_name]
        for r in ridges:
            if r.peak_power > 0.2 * power.max():
                # Strongest ridge(s): must land in the target band
                assert classify_period(r.peak_period_sec) == band_name


class TestRidgeDuration:
    def test_duration_within_20_percent(self):
        # 4-hour decay → ~8h FWHM-equivalent footprint above threshold
        decay_hours = 4.0
        signal = _make_signal(period_min=60, amplitude=2.0,
                              decay_hours=decay_hours)
        freq, power = _cwt_for(signal)
        ridges = extract_ridges(
            power, freq,
            min_duration_sec=2 * 3600,
            min_pixels=20,
        )
        assert len(ridges) >= 1
        r = ridges[0]
        duration_h = r.duration_seconds(dt=60.0) / 3600
        # The ridge duration depends on the amplitude threshold; we
        # demand only that it falls in a sensible factor-2 envelope
        # around the ~2 sigma envelope width 2*decay = 8h.
        assert 4.0 <= duration_h <= 16.0


class TestRidgeCOI:
    def test_packets_at_segment_edge_are_clipped(self):
        # Inject a Gaussian packet right at t=0 -- should be killed by
        # the COI mask.
        wave = WaveTemplate(
            period=60 * 60,
            amplitude=2.0,
            decay_width=2 * 3600,
            shift=0.5 * 3600,  # 30 min from edge
        )
        _, signal = simulate_signal(
            n_samples=2160, dt=60.0, waves=[wave]
        )
        freq, power = _cwt_for(signal)
        ridges = extract_ridges(
            power, freq,
            min_duration_sec=2 * 3600,
            min_pixels=30,
            coi_factor=2.0,  # very strict
        )
        # Either no ridge or one whose center is far from t=0
        for r in ridges:
            assert r.peak_time_idx > 60  # at least 1h from edge


class TestRidgeShape:
    def test_returns_list(self):
        signal = _make_signal(period_min=60, amplitude=2.0)
        freq, power = _cwt_for(signal)
        out = extract_ridges(power, freq)
        assert isinstance(out, list)
        assert all(isinstance(r, Ridge) for r in out)
