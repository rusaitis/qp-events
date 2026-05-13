"""Tests for qp.events.sweep_loader.

Focus: the ownership-window arithmetic that determines which segment
emits a given event. Adjacent 36-h segments stepped by 24 h must tile
the mission without overlap or gap, so a wave packet whose peak falls
in the boundary region is emitted by exactly one segment.
"""

from __future__ import annotations

import datetime

import pytest

from qp.events.sweep_loader import segment_central_window


def _times(t0: datetime.datetime, span_hours: float = 36.0) -> list[datetime.datetime]:
    """Build 1-min-cadence timestamps spanning ``span_hours`` from ``t0``."""
    n = int(span_hours * 60)
    return [t0 + datetime.timedelta(minutes=i) for i in range(n)]


class TestSegmentCentralWindow:
    def test_width_equals_hop(self):
        """Returned window must be exactly hop_hours wide."""
        t0 = datetime.datetime(2007, 1, 1)
        start, end = segment_central_window(_times(t0), 36.0, 24.0)
        assert end - start == datetime.timedelta(hours=24)

    def test_starts_at_half_padding(self):
        """Pad on the left = (span - hop)/2 = 6 h for the canonical archive."""
        t0 = datetime.datetime(2007, 1, 1)
        start, _ = segment_central_window(_times(t0), 36.0, 24.0)
        assert start == t0 + datetime.timedelta(hours=6)

    def test_adjacent_segments_tile_disjointly(self):
        """Window N's end must equal window N+1's start — half-open both sides."""
        t0_a = datetime.datetime(2007, 1, 1)
        t0_b = t0_a + datetime.timedelta(hours=24)  # next segment, hop=24h
        _, end_a = segment_central_window(_times(t0_a), 36.0, 24.0)
        start_b, _ = segment_central_window(_times(t0_b), 36.0, 24.0)
        assert end_a == start_b

    def test_handles_data_gap_hop(self):
        """Non-standard hop (e.g. across a data gap): no overlap, gap allowed."""
        t0_a = datetime.datetime(2007, 1, 1)
        t0_b = t0_a + datetime.timedelta(hours=48)  # one segment skipped
        _, end_a = segment_central_window(_times(t0_a), 36.0, 24.0)
        start_b, _ = segment_central_window(_times(t0_b), 36.0, 24.0)
        # 24-h gap between windows is fine — those hours are genuinely
        # uncovered by any segment.
        assert start_b - end_a == datetime.timedelta(hours=24)

    @pytest.mark.parametrize(
        "span_hours,hop_hours,expected_pad",
        [
            (36.0, 24.0, 6.0),
            (48.0, 24.0, 12.0),
            (36.0, 36.0, 0.0),
            (24.0, 12.0, 6.0),
        ],
    )
    def test_pad_arithmetic(self, span_hours, hop_hours, expected_pad):
        """pad = (span - hop)/2 across alternative archive layouts."""
        t0 = datetime.datetime(2007, 1, 1)
        start, end = segment_central_window(
            _times(t0, span_hours),
            span_hours,
            hop_hours,
        )
        assert start == t0 + datetime.timedelta(hours=expected_pad)
        assert end - start == datetime.timedelta(hours=hop_hours)
