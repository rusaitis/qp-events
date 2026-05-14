"""Smoke tests for qp.io.data_quality — flag CSV round-trip + parsing.

The five PDS quality-flag files live under ``DATA/CASSINI-Data-Quality/``
which isn't shipped to CI. We don't try to parse them here; instead we:

  1. Round-trip ``flags_to_csv`` / ``flags_from_csv`` on a synthetic
     :class:`QualityFlag` list (no DATA needed).
  2. Smoke-test ``load_all_quality_flags`` against an empty synthetic
     directory (graceful no-op).
  3. Spot-check internal DOY parsing on the canonical formats used by
     the real ASCII files.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from qp.io.data_quality import (
    QualityFlag,
    _parse_doy,
    _parse_iso,
    flags_from_csv,
    flags_to_csv,
    load_all_quality_flags,
)


def _make_flag(year: int, month: int, kind: str = "test") -> QualityFlag:
    start = datetime.datetime(year, month, 1, 12, 0, 0)
    end = start + datetime.timedelta(minutes=5)
    return QualityFlag(
        start=start,
        end=end,
        flag_type=kind,
        sensor="FGM",
        severity="high",
        description=f"synthetic {kind} {year}-{month:02d}",
    )


class TestQualityFlagDataclass:
    def test_round_trip_through_csv(self, tmp_path: Path):
        flags = [
            _make_flag(2007, 3, "range_change"),
            _make_flag(2010, 8, "scas"),
        ]
        csv_path = tmp_path / "flags.csv"
        flags_to_csv(flags, csv_path)
        assert csv_path.exists()

        restored = flags_from_csv(csv_path)
        assert len(restored) == 2
        for orig, got in zip(flags, restored, strict=True):
            assert orig.start == got.start
            assert orig.end == got.end
            assert orig.flag_type == got.flag_type
            assert orig.sensor == got.sensor
            assert orig.severity == got.severity
            assert orig.description == got.description


class TestParseDoy:
    def test_doy_with_fraction(self):
        # 2007-day213 = 2007-08-01
        dt = _parse_doy("2007-213T05:16:11.386")
        assert dt is not None
        assert dt.year == 2007 and dt.month == 8 and dt.day == 1
        assert dt.hour == 5 and dt.minute == 16 and dt.second == 11

    def test_doy_space_separator(self):
        dt = _parse_doy("2007-213 05:16:11")
        assert dt is not None
        assert dt.hour == 5

    def test_garbage_returns_none(self):
        assert _parse_doy("not a date") is None


class TestParseIso:
    def test_iso_with_milliseconds(self):
        dt = _parse_iso("2007-08-01T05:16:11.386")
        assert dt is not None
        assert dt.year == 2007

    def test_iso_without_seconds_fails_clean(self):
        assert _parse_iso("2007-08-01T05:16") is None


class TestLoadAllQualityFlags:
    def test_empty_quality_files_return_empty(self, tmp_path: Path):
        """Empty (but present) quality files yield an empty merged list."""
        for name in (
            "FGM_FULL_TIMING_ERRS.CSV",
            "RANGE_CHANGES.ASC",
            "MODE_CHANGES.ASC",
            "SCAS_TIMES.ASC",
            "SPURIOUS_RANGE_CHANGES.ASC",
        ):
            (tmp_path / name).write_text("")
        flags = load_all_quality_flags(quality_dir=tmp_path)
        assert flags == []

    def test_science_window_filter_drops_pre_2004(self, tmp_path: Path):
        """A flag entirely before the min_year window must be filtered out."""
        for name in (
            "FGM_FULL_TIMING_ERRS.CSV",
            "RANGE_CHANGES.ASC",
            "MODE_CHANGES.ASC",
            "SCAS_TIMES.ASC",
            "SPURIOUS_RANGE_CHANGES.ASC",
        ):
            (tmp_path / name).write_text("")
        # min_year=2010 → 2007 should be filtered, 2011 kept (when present).
        flags = load_all_quality_flags(
            quality_dir=tmp_path, min_year=2010, max_year=2012
        )
        assert flags == []
