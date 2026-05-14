"""Smoke tests for qp.io.pds.

The PDS readers depend on the local `DATA/CASSINI-DATA/` tree, which we
do NOT want to require in CI. These tests:

 1. Exercise the path-construction logic on a synthetic data root.
 2. Round-trip a tiny synthetic TAB file through `read_timeseries_file`
    and `select_data`.

so we catch regressions in the public API without depending on the real
25 GB Cassini archive.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from qp.io.pds import (
    COLUMNS,
    DATETIME_FMT,
    MISSION_END,
    MISSION_START,
    ColumnDef,
    mag_filepath,
    mag_filepaths_for_range,
    read_timeseries_file,
    select_data,
)


class TestColumnDefAndLayouts:
    def test_column_def_is_frozen(self):
        col = ColumnDef(0, "Time", "min", "str", "Time")
        with pytest.raises((AttributeError, Exception)):
            col.index = 5  # type: ignore[misc]

    def test_known_layouts_present(self):
        assert "KRTP" in COLUMNS
        assert "KSM" in COLUMNS
        assert "KSO" in COLUMNS
        # KRTP has 9 columns (LT field), KSM has 8
        assert len(COLUMNS["KRTP"]) == 9
        assert len(COLUMNS["KSM"]) == 8

    def test_ksm_kso_share_layout(self):
        assert COLUMNS["KSM"] is COLUMNS["KSO"] or COLUMNS["KSM"] == COLUMNS["KSO"]

    def test_mission_window_is_cassini_orbit(self):
        # Sanity: Cassini Saturn-orbit phase 2004-06-30 to 2017-09-13.
        assert MISSION_START.year == 2004
        assert MISSION_END.year == 2017


class TestMagFilepath:
    def test_path_layout(self, tmp_path: Path):
        p = mag_filepath("2007", "KRTP", "1min", data_root=tmp_path)
        # expected layout: <root>/CASSINI-DATA/<dataset>/DATA/<year>/<file>.TAB
        parts = p.relative_to(tmp_path).parts
        assert parts[0] == "CASSINI-DATA"
        assert parts[2] == "DATA"
        assert parts[3] == "2007"
        assert parts[-1] == "2007_FGM_KRTP_1M.TAB"

    def test_1sec_path(self, tmp_path: Path):
        p = mag_filepath("2010", "KSM", "1sec", data_root=tmp_path)
        assert p.name == "2010_FGM_KSM_1S.TAB"

    def test_unknown_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            mag_filepath("2007", "KRTP", "5min")


class TestReadTimeseriesFile:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert read_timeseries_file(tmp_path / "nope.tab") == []

    def test_round_trip_minimal_ksm_file(self, tmp_path: Path):
        path = tmp_path / "small.tab"
        # 3 rows of synthetic KSM data
        lines = [
            "2007-01-02T00:00:00  1.0  2.0  3.0  3.7  -5.0  0.0  0.0",
            "2007-01-02T00:01:00  1.1  2.1  3.1  3.8  -5.1  0.0  0.0",
            "2007-01-02T00:02:00  1.2  2.2  3.2  3.9  -5.2  0.0  0.0",
        ]
        path.write_text("\n".join(lines) + "\n")
        rows = read_timeseries_file(path)
        assert len(rows) == 3
        assert rows[0][0] == "2007-01-02T00:00:00"
        # ASCII tokens, not parsed floats
        assert rows[0][1] == "1.0"

    def test_date_filter(self, tmp_path: Path):
        path = tmp_path / "filtered.tab"
        lines = [
            "2007-01-02T00:00:00  1.0",
            "2007-01-02T00:05:00  2.0",
            "2007-01-02T00:10:00  3.0",
        ]
        path.write_text("\n".join(lines) + "\n")
        rows = read_timeseries_file(
            path,
            date_from=datetime.datetime(2007, 1, 2, 0, 3),
            date_to=datetime.datetime(2007, 1, 2, 0, 8),
        )
        assert len(rows) == 1
        assert rows[0][0] == "2007-01-02T00:05:00"


class TestSelectData:
    def test_select_data_no_files_returns_empty(self, tmp_path: Path):
        out = select_data(
            datetime.datetime(2007, 1, 1),
            datetime.datetime(2007, 1, 2),
            coords="KSM",
            data_root=tmp_path,
        )
        assert out == []

    def test_select_data_string_date_parses(self, tmp_path: Path):
        # Should not raise on string-typed dates
        out = select_data(
            "2007-01-01T00:00:00",
            "2007-01-02T00:00:00",
            coords="KSM",
            datetime_fmt=DATETIME_FMT,
            data_root=tmp_path,
        )
        assert out == []

    def test_filepaths_for_range_year_boundary(self, tmp_path: Path):
        # A range spanning 3 years would request 3 yearly files;
        # since none exist on tmp_path, we get back an empty list.
        paths = mag_filepaths_for_range(
            datetime.datetime(2005, 1, 1),
            datetime.datetime(2007, 1, 1),
            coords="KSM",
            data_root=tmp_path,
        )
        assert paths == []
