"""Tests for qp.signal.power_ratio — Eq. 4 power-ratio computation (Fig 5)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.signal.fft import estimate_background, welch_psd
from qp.signal.power_ratio import compute_power_ratios

DT = 60.0  # 1-min MAG cadence
N_SAMPLES = 36 * 60  # 36-hour segment, the canonical FFT window


@pytest.fixture
def t_36h() -> np.ndarray:
    return np.arange(N_SAMPLES) * DT


def _white(rng: np.random.Generator, sigma: float = 1.0) -> np.ndarray:
    return rng.normal(0.0, sigma, N_SAMPLES)


class TestComputePowerRatios:
    def test_keys_and_shapes(self):
        rng = np.random.default_rng(0)
        out = compute_power_ratios(
            _white(rng), _white(rng), _white(rng), _white(rng), dt=DT
        )
        expected = {
            "freq",
            "psd_par",
            "psd_perp1",
            "psd_perp2",
            "psd_total",
            "background",
            "r_par",
            "r_perp1",
            "r_perp2",
            "r_total",
        }
        assert set(out) == expected
        n_freq = out["freq"].size
        for key in expected - {"freq"}:
            assert out[key].shape == (n_freq,), f"{key} shape mismatch"

    def test_flat_noise_ratios_near_unity(self):
        """White noise on all components → all ratios cluster around 1 in the QP band.

        The Savitzky-Golay background is iteratively levelled so that ~50% of
        spectral points sit above it within the 30-min–3-h fit window, so the
        median ratio over that window should be close to 1.
        """
        rng = np.random.default_rng(1)
        out = compute_power_ratios(
            _white(rng), _white(rng), _white(rng), _white(rng), dt=DT
        )
        freq = out["freq"]
        # restrict to the QP search band (30 min - 3 h) where the background fit anchors
        band = (freq >= 1.0 / (3.0 * 3600.0)) & (freq < 1.0 / (30.0 * 60.0))
        for key in ("r_par", "r_perp1", "r_perp2", "r_total"):
            median = np.median(out[key][band])
            assert 0.3 < median < 3.0, f"{key} median {median:.2f} not near unity"

    def test_injected_qp60_peaks_in_perp1(self, t_36h):
        """Inject a 60-min sinusoid on b_perp1; ratio should spike there only."""
        rng = np.random.default_rng(2)
        period = 3600.0
        amplitude = 4.0
        wave = amplitude * np.sin(2 * np.pi * t_36h / period)

        b_par = _white(rng, sigma=0.5)
        b_perp1 = wave + _white(rng, sigma=0.5)
        b_perp2 = _white(rng, sigma=0.5)
        b_total = _white(rng, sigma=0.5)

        out = compute_power_ratios(b_par, b_perp1, b_perp2, b_total, dt=DT)
        freq = out["freq"]
        target = 1.0 / period
        idx = int(np.argmin(np.abs(freq - target)))

        # b_perp1 should pop massively above background; b_par should not
        assert out["r_perp1"][idx] > 50.0
        assert out["r_par"][idx] < 5.0
        # peak in r_perp1 must land within one bin of the injected frequency
        peak_idx = int(np.argmax(out["r_perp1"]))
        assert abs(peak_idx - idx) <= 1

    def test_welch_nperseg_passthrough(self):
        """nperseg controls the Welch segment length and thus freq resolution."""
        rng = np.random.default_rng(3)
        b = _white(rng)
        out_full = compute_power_ratios(b, b, b, b, dt=DT, nperseg=N_SAMPLES)
        out_half = compute_power_ratios(b, b, b, b, dt=DT, nperseg=N_SAMPLES // 4)
        # shorter segments → coarser frequency grid (fewer bins)
        assert out_half["freq"].size < out_full["freq"].size

    def test_background_safe_against_zero(self):
        """A pathologically tiny PSD must not produce NaN/inf ratios."""
        rng = np.random.default_rng(4)
        tiny = 1e-20 * _white(rng)
        out = compute_power_ratios(tiny, tiny, tiny, tiny, dt=DT)
        for key in ("r_par", "r_perp1", "r_perp2", "r_total"):
            assert np.all(np.isfinite(out[key])), f"{key} has non-finite values"


class TestEstimateBackground:
    def test_positive_everywhere(self):
        rng = np.random.default_rng(5)
        freq, psd = welch_psd(_white(rng), dt=DT)
        bg = estimate_background(psd, freq)
        assert bg.shape == psd.shape
        assert np.all(bg > 0)
        assert np.all(np.isfinite(bg))

    def test_power_law_tracking(self):
        r"""On a clean $f^{-1}$ PSD the smoothed background should follow it.

        Tests that the Savitzky-Golay fit doesn't introduce systematic
        log-space bias — median(log bg − log psd) ≈ 0 in the QP fit band.
        """
        # construct frequency grid matching a 36-h Welch run
        rng = np.random.default_rng(6)
        freq, _ = welch_psd(_white(rng), dt=DT)
        # synthetic 1/f spectrum, freq>0 only
        nz = freq > 0
        psd = np.where(nz, 1.0 / np.maximum(freq, 1e-10), 1.0)
        bg = estimate_background(psd, freq)
        band = (freq >= 1.0 / (3.0 * 3600.0)) & (freq < 1.0 / (30.0 * 60.0))
        # background should track PSD to within factor of ~3 in log-space
        log_residual = np.log(bg[band]) - np.log(psd[band])
        assert_allclose(np.median(log_residual), 0.0, atol=np.log(3.0))
