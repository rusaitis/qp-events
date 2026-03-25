"""Project-wide configuration: coordinate systems, datetime formats, reference data.

Extracts remaining configuration from ``cassinilib/DataPaths.py`` that wasn't
already covered by ``qp.io.pds`` or ``qp.plotting.style``.
"""

from __future__ import annotations

import datetime
import math
from enum import StrEnum

from qp.io.pds import COLUMNS


# ============================================================================
# Coordinate system enum
# ============================================================================


class CoordSystem(StrEnum):
    """Cassini magnetometer coordinate systems."""

    KRTP = "KRTP"
    KSM = "KSM"
    KSO = "KSO"


DEFAULT_COORD = CoordSystem.KRTP

# ============================================================================
# Datetime formats used across PDS products
# ============================================================================

DATETIME_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%jT%H:%M:%S",
)

DATETIME_FORMAT_ORBITS: str = "%Y-%m-%dT%H:%M:%S.000Z"
DATE_SHORT_FMT: str = "%Y-%m-%d"

# ============================================================================
# Data product filenames (relative to DataProducts/)
# ============================================================================

EVENT_CATALOG_FILE: str = "events_proposal.dat"
SLS5_FILENAME: str = "SLS5_2004-2018.txt"
CROSSING_FILENAME: str = "BS_MP_Crossing.txt"

# ============================================================================
# Test / synthetic data date ranges
# ============================================================================

TEST_DATE_RANGE: tuple[datetime.datetime, datetime.datetime] = (
    datetime.datetime(2000, 1, 1),
    datetime.datetime(2000, 1, 10),
)

TEST_FULL_DATE_RANGE: tuple[datetime.datetime, datetime.datetime] = (
    datetime.datetime(2000, 1, 1),
    datetime.datetime(2000, 3, 15),
)

# ============================================================================
# Saturn axis rotation angles (radians) for 3D visualization
# ============================================================================

SATURN_AXISROT: tuple[float, float] = (
    math.radians(-26.7),  # tilt about X
    math.radians(12.0),  # tilt about Y (speculative)
)

# ============================================================================
# Reference events selected for detailed analysis
# ============================================================================

_REFERENCE_EVENT_STRINGS: tuple[str, ...] = (
    "2006-09-27",
    "2006-11-10",
    "2006-11-23",
    "2006-12-06",
    "2006-12-18",
    "2006-12-21",
    "2006-12-22",
    "2007-02-01",
    "2007-02-02",
    "2007-02-03",
    "2007-02-04",
    "2007-02-05",
    "2007-02-06",
    "2007-02-07",
    "2007-02-08",
    "2007-02-09",
    "2007-02-10",
    "2007-02-11",
    "2007-02-12",
    "2007-02-13",
    "2007-02-14",
    "2007-02-15",
    "2007-02-16",
    "2007-02-17",
    "2007-02-18",
    "2007-02-19",
    "2007-02-20",
    "2007-02-21",
    "2007-02-22",
    "2007-02-23",
    "2007-02-24",
    "2008-02-29",
    "2008-08-10",
    "2008-09-09",
    "2009-02-23",
    "2012-10-18",
    "2013-08-16",
)

REFERENCE_EVENTS: tuple[datetime.datetime, ...] = tuple(
    datetime.datetime.strptime(s, "%Y-%m-%d") for s in _REFERENCE_EVENT_STRINGS
)

# ============================================================================
# Column definitions (re-exported from qp.io.pds for convenience)
# ============================================================================

COLUMN_DEFS = COLUMNS
