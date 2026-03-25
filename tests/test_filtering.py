"""Tests for qp.analysis.filtering — generic filtering and binning."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.analysis.filtering import (
    bin_to_value,
    filter_by_datetime,
    filter_by_property,
    group_by_bins,
    value_to_bin,
)


@dataclass
class MockEvent:
    """Minimal event-like object for testing."""

    local_time: float
    r_distance: float
    date_from: datetime.datetime


@pytest.fixture
def events():
    """10 mock events spread across local time and radial distance."""
    return [
        MockEvent(
            local_time=i * 2.4,
            r_distance=5.0 + i * 1.5,
            date_from=datetime.datetime(2007, 1, i + 1),
        )
        for i in range(10)
    ]


class TestFilterByProperty:
    def test_filter_by_string_key(self, events):
        result = filter_by_property(events, "local_time", 6.0, 18.0)
        for item in result:
            assert 6.0 <= item.local_time < 18.0

    def test_filter_by_callable(self, events):
        result = filter_by_property(events, lambda e: e.r_distance, 10.0, 15.0)
        for item in result:
            assert 10.0 <= item.r_distance < 15.0

    def test_empty_result(self, events):
        result = filter_by_property(events, "local_time", 100.0, 200.0)
        assert result == []

    def test_all_pass(self, events):
        result = filter_by_property(events, "local_time", 0.0, 100.0)
        assert len(result) == len(events)

    def test_exclusive_upper_bound(self, events):
        # Event with local_time=0.0 exists; max_val=0.0 should exclude it
        result = filter_by_property(events, "local_time", 0.0, 0.0)
        assert result == []


class TestFilterByDatetime:
    def test_basic_range(self, events):
        result = filter_by_datetime(
            events,
            datetime.datetime(2007, 1, 3),
            datetime.datetime(2007, 1, 7),
        )
        for item in result:
            assert (
                datetime.datetime(2007, 1, 3)
                <= item.date_from
                <= datetime.datetime(2007, 1, 7)
            )

    def test_inclusive_bounds(self, events):
        # Exactly matching the boundary should be included
        result = filter_by_datetime(
            events,
            datetime.datetime(2007, 1, 1),
            datetime.datetime(2007, 1, 1),
        )
        assert len(result) == 1

    def test_custom_key(self, events):
        result = filter_by_datetime(
            events,
            datetime.datetime(2007, 1, 5),
            datetime.datetime(2007, 1, 10),
            key=lambda e: e.date_from,
        )
        assert len(result) > 0


class TestValueToBin:
    def test_basic(self):
        assert value_to_bin(5.0, 0.0, 10.0, 10) == 5

    def test_lower_edge(self):
        assert value_to_bin(0.0, 0.0, 10.0, 10) == 0

    def test_upper_edge_clipped(self):
        # Value at exactly max_val should be clipped to last bin
        assert value_to_bin(10.0, 0.0, 10.0, 10) == 9

    def test_below_range_clipped(self):
        assert value_to_bin(-1.0, 0.0, 10.0, 10) == 0

    def test_above_range_clipped(self):
        assert value_to_bin(15.0, 0.0, 10.0, 10) == 9

    def test_array_input(self):
        values = np.array([1.0, 5.0, 9.0])
        result = value_to_bin(values, 0.0, 10.0, 10)
        assert_allclose(result, [1, 5, 9])

    def test_scalar_returns_int(self):
        result = value_to_bin(3.5, 0.0, 10.0, 10)
        assert isinstance(result, int)


class TestBinToValue:
    def test_roundtrip(self):
        """bin_to_value(value_to_bin(x)) should be near x for centered bins."""
        for x in [0.5, 2.5, 5.5, 9.5]:
            idx = value_to_bin(x, 0.0, 10.0, 10)
            center = bin_to_value(idx, 0.0, 10.0, 10)
            assert_allclose(center, x, atol=0.51)

    def test_bin_centers(self):
        # 10 bins from 0 to 10: centers at 0.5, 1.5, ..., 9.5
        centers = [bin_to_value(i, 0.0, 10.0, 10) for i in range(10)]
        expected = [0.5 + i for i in range(10)]
        assert_allclose(centers, expected)

    def test_array_input(self):
        indices = np.array([0, 4, 9])
        result = bin_to_value(indices, 0.0, 10.0, 10)
        assert_allclose(result, [0.5, 4.5, 9.5])


class TestGroupByBins:
    def test_basic_grouping(self, events):
        bins, centers = group_by_bins(events, "r_distance", 5.0, 20.0, 5)
        assert len(bins) == 5
        assert len(centers) == 5
        total = sum(len(b) for b in bins)
        assert total <= len(events)

    def test_all_items_binned(self, events):
        # Use a range that covers all events
        bins, _ = group_by_bins(events, "r_distance", 0.0, 30.0, 30)
        total = sum(len(b) for b in bins)
        assert total == len(events)

    def test_empty_bins(self):
        items = [
            MockEvent(
                local_time=1.0, r_distance=5.0, date_from=datetime.datetime(2007, 1, 1)
            )
        ]
        bins, _ = group_by_bins(items, "r_distance", 0.0, 100.0, 100)
        non_empty = [b for b in bins if len(b) > 0]
        assert len(non_empty) == 1

    def test_callable_key(self, events):
        bins, centers = group_by_bins(
            events,
            lambda e: e.local_time,
            0.0,
            24.0,
            4,
        )
        assert len(bins) == 4
        assert_allclose(centers, [3.0, 9.0, 15.0, 21.0])

    def test_out_of_range_excluded(self):
        items = [
            MockEvent(
                local_time=25.0, r_distance=5.0, date_from=datetime.datetime(2007, 1, 1)
            )
        ]
        bins, _ = group_by_bins(items, "local_time", 0.0, 24.0, 4)
        total = sum(len(b) for b in bins)
        assert total == 0
