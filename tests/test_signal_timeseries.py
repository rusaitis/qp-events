"""Tests for qp.signal.timeseries — resample, detrend, smooth, block average."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.signal.timeseries import (
    block_average,
    detrend,
    detrend_for_fft,
    running_average,
    smooth_savgol,
    uniform_resample,
)


class TestRunningAverage:
    def test_constant_input_returns_constant(self):
        data = np.ones(100) * 7.0
        out = running_average(data, window_samples=11)
        assert_allclose(out, 7.0)

    def test_window_forced_odd(self):
        # passing even window should still work (rounded up internally)
        data = np.arange(50, dtype=float)
        out_even = running_average(data, window_samples=10)
        out_odd = running_average(data, window_samples=11)
        assert_allclose(out_even, out_odd)


class TestDetrend:
    def test_removes_linear_trend(self):
        # 100-sample linear ramp; window 21 ≈ smaller than length
        t = np.arange(100, dtype=float)
        ramp = 0.1 * t + 2.0
        detrended, trend = detrend(ramp, window_samples=21)
        # the running-average trend should track the ramp closely except
        # at the edges; the centre region should detrend to ~0
        assert_allclose(detrended[30:70], 0.0, atol=0.1)
        # trend's middle samples should be close to the input
        assert_allclose(trend[30:70], ramp[30:70], atol=0.1)

    def test_detrend_for_fft_default_window(self):
        # 36 h of constant data → detrended values are all ~zero
        n = 36 * 60
        const = np.full(n, 5.0)
        detrended, trend = detrend_for_fft(const, dt=60.0)
        assert detrended.shape == const.shape
        assert_allclose(detrended, 0.0, atol=1e-10)
        assert_allclose(trend, 5.0)


class TestBlockAverage:
    def test_exact_division(self):
        data = np.arange(12, dtype=float)
        out = block_average(data, block_size=3)
        # blocks: [0,1,2], [3,4,5], [6,7,8], [9,10,11] → [1,4,7,10]
        assert_allclose(out, [1.0, 4.0, 7.0, 10.0])

    def test_partial_last_block_padded_with_nan(self):
        data = np.arange(10, dtype=float)  # 10 elements, block_size 3
        out = block_average(data, block_size=3)
        # blocks: [0,1,2],[3,4,5],[6,7,8],[9,NaN,NaN] → [1, 4, 7, 9]
        assert out.size == 4
        assert_allclose(out, [1.0, 4.0, 7.0, 9.0])

    def test_ignores_nan(self):
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0])
        out = block_average(data, block_size=2)
        # [1, NaN] → 1; [3, NaN] → 3; [5, 6] → 5.5
        assert_allclose(out, [1.0, 3.0, 5.5])


class TestUniformResample:
    def test_already_uniform_passthrough(self):
        t = np.arange(0.0, 600.0, 60.0)
        y = np.sin(2 * np.pi * t / 360.0)
        y_re, t_re = uniform_resample(y, t, dt=60.0)
        # uniform_resample defaults n_samples to int((t[-1]-t[0])/dt), which is
        # one short of len(t); compare on the shared prefix.
        n = t_re.size
        assert_allclose(t_re, t[:n])
        assert_allclose(y_re, y[:n], atol=1e-12)

    def test_handles_nans_zero_method(self):
        t = np.arange(0.0, 600.0, 60.0)
        y = np.ones_like(t)
        y[3] = np.nan
        y_re, _ = uniform_resample(y, t, dt=60.0, nan_method="zero")
        assert not np.any(np.isnan(y_re))


class TestSmoothSavgol:
    def test_zero_response_to_constant(self):
        data = np.full(101, 4.0)
        out = smooth_savgol(data, window_samples=11, order=3)
        assert_allclose(out, 4.0, atol=1e-10)

    def test_too_short_window_raises(self):
        # scipy raises ValueError if window > data length
        with pytest.raises(ValueError):
            smooth_savgol(np.arange(5, dtype=float), window_samples=11, order=3)
