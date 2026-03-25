"""Tests for qp.config and qp.constants."""

from __future__ import annotations

import datetime
import math

from numpy.testing import assert_allclose

from qp.config import (
    COLUMN_DEFS,
    DATETIME_FORMATS,
    DEFAULT_COORD,
    CoordSystem,
    REFERENCE_EVENTS,
    SATURN_AXISROT,
    TEST_DATE_RANGE,
    SLS5_FILENAME,
    CROSSING_FILENAME,
)
from qp.constants import (
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    PROTON_MASS,
    SATURN_RADIUS,
    SEC_PER_DAY,
    SEC_PER_HOUR,
    SEC_PER_MIN,
)


class TestCoordSystem:
    """Tests for the CoordSystem enum."""

    def test_values(self):
        assert CoordSystem.KRTP == "KRTP"
        assert CoordSystem.KSM == "KSM"
        assert CoordSystem.KSO == "KSO"

    def test_default(self):
        assert DEFAULT_COORD == CoordSystem.KRTP

    def test_string_comparison(self):
        assert CoordSystem.KSM == "KSM"
        assert "KRTP" == CoordSystem.KRTP


class TestDatetimeFormats:
    """Tests for datetime format strings."""

    def test_iso_format_parses(self):
        dt = datetime.datetime.strptime("2007-02-15T06:30:00", DATETIME_FORMATS[0])
        assert dt.year == 2007

    def test_doy_format_parses(self):
        dt = datetime.datetime.strptime("2007-046T12:00:00", DATETIME_FORMATS[1])
        assert dt.year == 2007
        assert dt.month == 2

    def test_formats_are_tuple(self):
        assert isinstance(DATETIME_FORMATS, tuple)


class TestReferenceEvents:
    """Tests for the reference event list."""

    def test_count(self):
        assert len(REFERENCE_EVENTS) == 37

    def test_all_datetimes(self):
        for event in REFERENCE_EVENTS:
            assert isinstance(event, datetime.datetime)

    def test_sorted_by_date(self):
        # Events should be chronologically sorted
        sorted_events = tuple(sorted(REFERENCE_EVENTS))
        assert REFERENCE_EVENTS == sorted_events

    def test_first_event(self):
        assert REFERENCE_EVENTS[0].year == 2006
        assert REFERENCE_EVENTS[0].month == 9

    def test_last_event(self):
        assert REFERENCE_EVENTS[-1].year == 2013


class TestColumnDefs:
    """Tests for column definition lookups."""

    def test_krtp_exists(self):
        assert "KRTP" in COLUMN_DEFS

    def test_ksm_exists(self):
        assert "KSM" in COLUMN_DEFS

    def test_krtp_has_9_columns(self):
        assert len(COLUMN_DEFS["KRTP"]) == 9

    def test_ksm_has_8_columns(self):
        assert len(COLUMN_DEFS["KSM"]) == 8

    def test_first_column_is_time(self):
        for key in COLUMN_DEFS:
            assert COLUMN_DEFS[key][0].kind == "Time"


class TestDataProductFilenames:
    """Tests for data product filename constants."""

    def test_sls5(self):
        assert SLS5_FILENAME.endswith(".txt")

    def test_crossing(self):
        assert CROSSING_FILENAME.endswith(".txt")


class TestSaturnAxes:
    """Tests for Saturn axis rotation angles."""

    def test_x_tilt(self):
        assert_allclose(SATURN_AXISROT[0], math.radians(-26.7), rtol=1e-10)

    def test_y_tilt(self):
        assert_allclose(SATURN_AXISROT[1], math.radians(12.0), rtol=1e-10)


class TestTestDateRange:
    """Tests for the synthetic test date range."""

    def test_start_before_end(self):
        assert TEST_DATE_RANGE[0] < TEST_DATE_RANGE[1]

    def test_year_2000(self):
        assert TEST_DATE_RANGE[0].year == 2000


class TestConstants:
    """Tests for physical constants."""

    def test_time_conversions(self):
        assert SEC_PER_MIN == 60.0
        assert SEC_PER_HOUR == 3600.0
        assert SEC_PER_DAY == 86400.0

    def test_electron_mass_order(self):
        assert 9e-31 < ELECTRON_MASS < 10e-31

    def test_proton_mass_order(self):
        assert 1.6e-27 < PROTON_MASS < 1.7e-27

    def test_elementary_charge(self):
        assert_allclose(ELEMENTARY_CHARGE, 1.602e-19, rtol=1e-3)

    def test_saturn_radius(self):
        assert_allclose(SATURN_RADIUS, 60268e3, rtol=1e-6)
