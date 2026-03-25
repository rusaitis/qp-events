"""Tests for qp.time_utils — datetime/timestamp conversions."""

from __future__ import annotations

import datetime

import pytest
from numpy.testing import assert_allclose

from qp.time_utils import from_timestamp, parse_datetime, to_timestamp

UTC = datetime.timezone.utc


class TestToTimestamp:
    """Tests for to_timestamp()."""

    def test_epoch_is_zero(self):
        epoch = datetime.datetime(1970, 1, 1, tzinfo=UTC)
        assert to_timestamp(epoch) == 0.0

    def test_known_date(self):
        # 2012-01-02T12:00:00 UTC = 1325505600.0
        dt = datetime.datetime(2012, 1, 2, 12, 0, 0, tzinfo=UTC)
        assert_allclose(to_timestamp(dt), 1325505600.0, atol=1.0)

    def test_naive_datetime_treated_as_utc(self):
        dt_naive = datetime.datetime(2000, 1, 1, 0, 0, 0)
        dt_aware = datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert to_timestamp(dt_naive) == to_timestamp(dt_aware)

    def test_cassini_mission_start(self):
        # Cassini SOI: 2004-06-30
        dt = datetime.datetime(2004, 6, 30, tzinfo=UTC)
        ts = to_timestamp(dt)
        assert ts > 0
        # Roughly 34.5 years after epoch
        assert 1.08e9 < ts < 1.10e9


class TestFromTimestamp:
    """Tests for from_timestamp()."""

    def test_epoch_roundtrip(self):
        dt = from_timestamp(0.0)
        assert dt.year == 1970
        assert dt.month == 1
        assert dt.day == 1
        assert dt.tzinfo is not None

    def test_roundtrip(self):
        original = datetime.datetime(2007, 2, 15, 6, 30, 0, tzinfo=UTC)
        ts = to_timestamp(original)
        recovered = from_timestamp(ts)
        assert abs((recovered - original).total_seconds()) < 1e-3

    @pytest.mark.parametrize("ts", [0.0, 1e9, 1.5e9, -86400.0])
    def test_roundtrip_various(self, ts):
        dt = from_timestamp(ts)
        assert_allclose(to_timestamp(dt), ts, atol=1e-3)

    def test_result_is_utc(self):
        dt = from_timestamp(1e9)
        assert dt.tzinfo is not None
        assert dt.utcoffset() == datetime.timedelta(0)


class TestParseDatetime:
    """Tests for parse_datetime()."""

    def test_iso_format(self):
        dt = parse_datetime("2007-02-15T06:30:00")
        assert dt == datetime.datetime(2007, 2, 15, 6, 30, 0)

    def test_custom_format(self):
        dt = parse_datetime("2007-046T12:00:00", "%Y-%jT%H:%M:%S")
        assert dt.year == 2007
        assert dt.month == 2
        assert dt.day == 15  # DOY 46 = Feb 15

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_datetime("not-a-date")


class TestConsistencyWithOldAPI:
    """Verify we match the old cassinilib/DatetimeFunctions.py behavior."""

    def test_matches_old_date2timestamp(self):
        # Old: (date - datetime(1970,1,1)) / timedelta(seconds=1)
        dt = datetime.datetime(2012, 1, 8, 15, 34, 10)
        old_result = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
        new_result = to_timestamp(dt)
        assert_allclose(new_result, old_result, atol=1e-3)
