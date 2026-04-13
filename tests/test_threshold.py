"""Tests for the Phase 2 detection gate (FFT screen + wavelet σ mask)."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from qp.events.bands import QP_BANDS
from qp.events.catalog import WaveTemplate
from qp.events.detector import detect_with_gate
from qp.events.threshold import (
    GateConfig,
    MAD_TO_SIGMA,
    screen_segment_by_power_ratio,
    screen_spectral_result,
    wavelet_sigma_mask,
)
from qp.signal.pipeline import analyze_segment
from qp.signal.synthetic import simulate_signal
from qp.signal.wavelet import morlet_cwt


@pytest.fixture
def time_axis():
    n = 2160
    dt = 60.0
    t0 = datetime.datetime(2007, 1, 1)
    times = [t0 + datetime.timedelta(seconds=i * dt) for i in range(n)]
    return n, dt, times


def _red_noise(n: int, sigma: float = 0.1, seed: int = 1):
    """First-order AR(1) red noise — closer to magnetospheric background."""
    rng = np.random.default_rng(seed)
    alpha = 0.95
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + e[i]
    return x


def _qp_signal(period_min: float, amplitude: float, n: int, dt: float,
                noise_sigma: float = 0.0, seed: int = 1,
                add_red_noise: bool = False):
    """Gaussian-windowed QP packet on top of an optional red-noise background.

    The red-noise background mimics the f^-2 red spectrum of real
    magnetospheric MAG data, which makes the smoothed background
    estimator work the way it does on real data — i.e. the signal
    sticks well above background instead of being self-fitted away.
    """
    wave = WaveTemplate(
        period=period_min * 60.0,
        amplitude=amplitude,
        decay_width=4 * 3600.0,
        shift=18 * 3600.0,
    )
    _, sig = simulate_signal(
        n_samples=n, dt=dt, waves=[wave],
        noise_sigma=noise_sigma, seed=seed,
    )
    if add_red_noise:
        sig = sig + _red_noise(n, sigma=0.05, seed=seed + 1000)
    return sig


# ----------------------------------------------------------------------
# FFT screen
# ----------------------------------------------------------------------


class TestFFTScreen:
    def test_quiet_segment_does_not_trigger(self, time_axis):
        n, dt, _ = time_axis
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.05, n)
        spec = analyze_segment(
            noise, dt=dt,
            detrend_window_sec=60.0,
            welch_nperseg=12 * 60,
            welch_noverlap=6 * 60,
        )
        for b in ("QP30", "QP60", "QP120"):
            res = screen_spectral_result(spec, b, ratio_threshold=5.0)
            assert not res.triggered, f"{b} false-positive on noise"

    def test_strong_qp60_triggers_qp60(self, time_axis):
        n, dt, _ = time_axis
        sig = _qp_signal(period_min=60, amplitude=2.0, n=n, dt=dt,
                          noise_sigma=0.05, add_red_noise=True)
        spec = analyze_segment(
            sig, dt=dt,
            detrend_window_sec=60.0,
            welch_nperseg=12 * 60,
            welch_noverlap=6 * 60,
        )
        res = screen_spectral_result(spec, "QP60", ratio_threshold=2.5)
        assert res.triggered
        assert 45 <= res.peak_period_min <= 80

    def test_strong_qp60_does_not_trigger_qp120(self, time_axis):
        n, dt, _ = time_axis
        sig = _qp_signal(period_min=60, amplitude=2.0, n=n, dt=dt,
                          add_red_noise=True)
        spec = analyze_segment(
            sig, dt=dt,
            detrend_window_sec=60.0,
            welch_nperseg=12 * 60,
            welch_noverlap=6 * 60,
        )
        res = screen_spectral_result(spec, "QP120", ratio_threshold=2.5)
        assert not res.triggered

    def test_screen_works_on_raw_arrays(self, time_axis):
        n, dt, _ = time_axis
        # Use the same params as test_strong_qp60_triggers_qp60 so the
        # test stays in lock-step with the production-leaning one.
        sig = _qp_signal(period_min=60, amplitude=2.0, n=n, dt=dt,
                          noise_sigma=0.05, add_red_noise=True)
        spec = analyze_segment(
            sig, dt=dt,
            detrend_window_sec=60.0,
            welch_nperseg=12 * 60,
            welch_noverlap=6 * 60,
        )
        res = screen_segment_by_power_ratio(
            spec.freq, spec.power_ratio, "QP60", ratio_threshold=2.5,
        )
        assert res.triggered


# ----------------------------------------------------------------------
# σ mask
# ----------------------------------------------------------------------


class TestSigmaMask:
    def test_mad_to_sigma_constant(self):
        # Gaussian σ ≈ 1.4826 * MAD; sanity check the constant.
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1.0, 100_000)
        mad = float(np.median(np.abs(x - np.median(x))))
        assert abs(MAD_TO_SIGMA * mad - 1.0) < 0.02

    def test_pure_noise_mask_is_sparse(self, time_axis):
        n, dt, _ = time_axis
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.05, n)
        freq, _, cwt = morlet_cwt(noise, dt=dt, n_freqs=200)
        power = np.abs(cwt)
        mask = wavelet_sigma_mask(power, freq, n_sigma=3.0)
        # On pure Gaussian noise, ~0.13 % of cells should exceed 3σ.
        # We're conservative: allow up to 5 % to account for the
        # CWT's auto-correlated cells and use_global_floor.
        frac = mask.mean()
        assert frac < 0.05, f"noise mask too dense: {frac:.3f}"

    def test_signal_mask_covers_packet(self, time_axis):
        n, dt, _ = time_axis
        sig = _qp_signal(period_min=60, amplitude=2.0, n=n, dt=dt,
                          noise_sigma=0.05)
        freq, _, cwt = morlet_cwt(sig, dt=dt, n_freqs=200)
        power = np.abs(cwt)
        mask = wavelet_sigma_mask(power, freq, n_sigma=3.0)

        # Find QP60 rows
        periods = 1.0 / freq
        qp60 = QP_BANDS["QP60"]
        qp60_rows = np.flatnonzero(
            (periods >= qp60.period_min_sec)
            & (periods < qp60.period_max_sec)
        )
        # In the QP60 band, near t=18h, mask should fire substantially
        center = int(18 * 3600 / dt)
        window = mask[qp60_rows][:, center - 60 : center + 60]
        frac = window.mean()
        assert frac > 0.10, f"signal not flagged in mask: {frac:.3f}"

    def test_mask_shape_matches(self, time_axis):
        n, dt, _ = time_axis
        sig = _qp_signal(period_min=60, amplitude=2.0, n=n, dt=dt)
        freq, _, cwt = morlet_cwt(sig, dt=dt, n_freqs=200)
        power = np.abs(cwt)
        mask = wavelet_sigma_mask(power, freq, n_sigma=3.0)
        assert mask.shape == power.shape
        assert mask.dtype == bool


# ----------------------------------------------------------------------
# Combined gate
# ----------------------------------------------------------------------


class TestDetectWithGate:
    def test_quiet_segment_no_packets(self, time_axis):
        n, dt, times = time_axis
        rng = np.random.default_rng(0)
        b_perp1 = rng.normal(0, 0.05, n)
        b_perp2 = rng.normal(0, 0.05, n)
        packets = detect_with_gate(b_perp1, b_perp2, times, dt=dt)
        assert len(packets) == 0

    def test_qp60_packet_recovered(self, time_axis):
        n, dt, times = time_axis
        b_perp1 = _qp_signal(period_min=60, amplitude=2.0, n=n, dt=dt,
                              noise_sigma=0.05, add_red_noise=True)
        b_perp2 = _qp_signal(period_min=60, amplitude=1.5, n=n, dt=dt,
                              noise_sigma=0.05, seed=2, add_red_noise=True)
        packets = detect_with_gate(
            b_perp1, b_perp2, times, dt=dt,
            gate=GateConfig(n_sigma=3.0),
        )
        qp60 = [p for p in packets if p.band == "QP60"]
        assert len(qp60) >= 1

    def test_strongest_packet_in_correct_band(self, time_axis):
        n, dt, times = time_axis
        b_perp1 = _qp_signal(period_min=60, amplitude=2.0, n=n, dt=dt,
                              noise_sigma=0.05, add_red_noise=True)
        b_perp2 = _qp_signal(period_min=60, amplitude=1.5, n=n, dt=dt,
                              noise_sigma=0.05, seed=2, add_red_noise=True)
        packets = detect_with_gate(
            b_perp1, b_perp2, times, dt=dt,
            gate=GateConfig(n_sigma=3.0),
        )
        if not packets:
            pytest.skip("no packets detected")
        strongest = max(packets, key=lambda p: p.prominence)
        assert strongest.band == "QP60"

    def test_gate_config_frozen(self):
        cfg = GateConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.n_sigma = 5.0  # type: ignore[misc]
