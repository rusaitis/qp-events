"""Tests for qp.analysis.correlation — event property correlation and phase lag."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.analysis.correlation import (
    event_property_correlation,
    sliding_phase_lag,
)


@dataclass
class MockEvent:
    period: float
    amplitude: float
    snr: float | None = None


class TestEventPropertyCorrelation:
    def test_perfect_positive_correlation(self):
        events = [MockEvent(period=i, amplitude=i * 2) for i in range(1, 20)]
        corr, pval = event_property_correlation(events, "period", "amplitude")
        assert corr > 0.99
        assert pval < 0.01

    def test_perfect_negative_correlation(self):
        events = [MockEvent(period=i, amplitude=20 - i) for i in range(1, 20)]
        corr, pval = event_property_correlation(events, "period", "amplitude")
        assert corr < -0.99

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        events = [
            MockEvent(period=rng.uniform(1, 100), amplitude=rng.uniform(1, 100))
            for _ in range(100)
        ]
        corr, _ = event_property_correlation(events, "period", "amplitude")
        assert abs(corr) < 0.3

    def test_pearson_method(self):
        events = [MockEvent(period=i, amplitude=i * 2) for i in range(1, 20)]
        corr, pval = event_property_correlation(
            events,
            "period",
            "amplitude",
            method="pearson",
        )
        assert corr > 0.99

    def test_too_few_events_raises(self):
        events = [MockEvent(period=1, amplitude=2)]
        with pytest.raises(ValueError, match="at least 3"):
            event_property_correlation(events, "period", "amplitude")

    def test_none_values_skipped(self):
        events = [
            MockEvent(period=1, amplitude=2, snr=None),
            MockEvent(period=2, amplitude=4, snr=3.0),
            MockEvent(period=3, amplitude=6, snr=5.0),
            MockEvent(period=4, amplitude=8, snr=7.0),
        ]
        # snr has one None — should skip that event
        corr, _ = event_property_correlation(events, "period", "snr")
        assert corr > 0.9

    def test_unknown_method_raises(self):
        events = [MockEvent(period=i, amplitude=i) for i in range(5)]
        with pytest.raises(ValueError, match="Unknown method"):
            event_property_correlation(events, "period", "amplitude", method="kendall")


class TestSlidingPhaseLag:
    def test_zero_lag_for_identical_signals(self):
        """Identical signals should have ~0 phase lag."""
        n = 500
        t = np.arange(n) * 60.0
        y = np.sin(2 * np.pi * t / 3600)
        lags = sliding_phase_lag(y, y, dt=60.0, window_samples=121)
        assert_allclose(np.median(lags), 0.0, atol=5.0)

    def test_90_degree_lag(self):
        """cos vs sin should give ~90 degree lag."""
        n = 1000
        t = np.arange(n) * 60.0
        period = 3600.0
        y1 = np.sin(2 * np.pi * t / period)
        y2 = np.cos(2 * np.pi * t / period)  # 90 deg ahead
        lags = sliding_phase_lag(y1, y2, dt=60.0, window_samples=121)
        median_lag = np.median(np.abs(lags))
        # Should be near 90 degrees (within 30 deg tolerance for discrete sampling)
        assert 60.0 < median_lag < 120.0

    def test_output_length(self):
        n = 300
        y = np.random.default_rng(42).normal(0, 1, n)
        window = 121
        lags = sliding_phase_lag(y, y, dt=60.0, window_samples=window)
        half_w = window // 2
        expected_len = n - 2 * half_w
        assert len(lags) == expected_len

    def test_bounded_by_max_lag(self):
        """All lag values should be within [-max_lag, max_lag]."""
        n = 500
        rng = np.random.default_rng(42)
        y1 = rng.normal(0, 1, n)
        y2 = rng.normal(0, 1, n)
        max_lag = 180.0
        lags = sliding_phase_lag(y1, y2, dt=60.0, max_lag_degrees=max_lag)
        assert np.all(np.abs(lags) <= max_lag + 1.0)  # small numerical margin
