"""Tests for qp.events.detector — band-agnostic in-segment dedup."""

from __future__ import annotations

import datetime

import pytest

from qp.events.bands import classify_period
from qp.events.catalog import WavePacketPeak
from qp.events.detector import dedup_peaks_by_band, dedup_peaks_by_period


def _peak(t_hours: float, period_min: float) -> WavePacketPeak:
    """Tiny WavePacketPeak factory for dedup tests."""
    t0 = datetime.datetime(2007, 1, 1)
    pt = t0 + datetime.timedelta(hours=t_hours)
    period_sec = period_min * 60.0
    return WavePacketPeak(
        peak_time=pt,
        prominence=1.0,
        date_from=pt - datetime.timedelta(hours=1),
        date_to=pt + datetime.timedelta(hours=1),
        band=classify_period(period_sec),
        period_sec=period_sec,
    )


class TestDedupPeaksByPeriod:
    def test_drops_close_same_period(self):
        peaks = [_peak(0.0, 60), _peak(1.5, 60)]
        kept = dedup_peaks_by_period(peaks, dt_sec=7200.0)
        assert [p.peak_time.hour for p in kept] == [0]

    def test_keeps_far_same_period(self):
        peaks = [_peak(0.0, 60), _peak(3.0, 60)]
        kept = dedup_peaks_by_period(peaks, dt_sec=7200.0)
        assert len(kept) == 2

    def test_interleaved_different_period_does_not_unmask_dup(self):
        """QP60-QP30-QP60 in 1.5 h must drop the second QP60.

        The dedup walks the merged tail backward within ``dt_sec`` so an
        interleaved different-period peak can't break the comparison.
        """
        peaks = [
            _peak(0.0, 60),
            _peak(1.0, 30),
            _peak(1.5, 60),
        ]
        kept = dedup_peaks_by_period(peaks, dt_sec=7200.0)
        periods = [p.period_sec for p in kept]
        assert periods == [3600.0, 1800.0]

    def test_multi_period_preserved(self):
        """Genuine multi-harmonic event: three distinct periods stay."""
        peaks = [
            _peak(0.0, 30),
            _peak(0.1, 60),
            _peak(0.2, 120),
        ]
        kept = dedup_peaks_by_period(peaks, dt_sec=7200.0)
        assert {p.period_sec for p in kept} == {1800.0, 3600.0, 7200.0}

    def test_cross_octave_edge_collapses(self):
        """Two near-period peaks classified into different bands collapse.

        39 min → QP30, 41 min → QP60. Same physical wave; band-keyed
        dedup would let both through, but the period-proximity rule
        catches them.
        """
        peaks = [_peak(0.0, 39), _peak(0.5, 41)]
        kept = dedup_peaks_by_period(peaks, dt_sec=7200.0)
        assert len(kept) == 1

    def test_empty_input(self):
        assert dedup_peaks_by_period([], dt_sec=7200.0) == []

    def test_missing_period_kept_as_is(self):
        """Peaks without period_sec are kept verbatim — no band fallback.

        Production round-8 detections always populate ``period_sec``;
        a missing value indicates a malformed peak that should not be
        silently dropped.
        """
        t0 = datetime.datetime(2007, 1, 1)
        bare = WavePacketPeak(
            peak_time=t0,
            prominence=1.0,
            date_from=t0 - datetime.timedelta(hours=1),
            date_to=t0 + datetime.timedelta(hours=1),
        )
        kept = dedup_peaks_by_period([bare, bare], dt_sec=7200.0)
        assert len(kept) == 2


class TestDeprecatedAlias:
    def test_alias_emits_warning(self):
        peaks = [_peak(0.0, 60), _peak(3.0, 60)]
        with pytest.warns(DeprecationWarning, match="dedup_peaks_by_band"):
            kept = dedup_peaks_by_band(peaks, dt_sec=7200.0)
        assert len(kept) == 2
