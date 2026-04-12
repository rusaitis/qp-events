"""PDS Cassini data file path generation and ASCII reading.

Extracts and modernizes logic from cassinilib/DataPaths.py and
cassinilib/SelectData.py.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import qp


# --- Column descriptor for PDS TAB files ---


@dataclass
class ColumnDef:
    """Describes one column in a PDS TAB file."""

    index: int
    name: str
    units: str
    dtype: str = "float"
    kind: str = ""  # 'Time', 'Field', 'Coord', 'lt'
    dt: float = 60.0  # sampling interval in seconds
    color: str = "black"
    fmt: str = ""  # strptime format for time columns


# --- Standard column layouts for PDS MAG files ---

DATETIME_FMT = "%Y-%m-%dT%H:%M:%S"

# Field component colors — canonical source is qp.plotting.style
from qp.plotting.style import FIELD_COLORS

MAG_COLUMNS_KRTP = [
    ColumnDef(0, "Time", "min", "str", "Time", fmt=DATETIME_FMT),
    ColumnDef(1, "Br", "nT", kind="Field", color=FIELD_COLORS[0]),
    ColumnDef(2, "Bth", "nT", kind="Field", color=FIELD_COLORS[1]),
    ColumnDef(3, "Bphi", "nT", kind="Field", color=FIELD_COLORS[2]),
    ColumnDef(4, "Btot", "nT", kind="Field", color=FIELD_COLORS[3]),
    ColumnDef(5, "r", "R_S", kind="Coord"),
    ColumnDef(6, "th", "deg", kind="Coord"),
    ColumnDef(7, "phi", "deg", kind="Coord"),
    ColumnDef(8, "LT", "h", kind="lt"),
]

MAG_COLUMNS_KSM = [
    ColumnDef(0, "Time", "min", "str", "Time", fmt=DATETIME_FMT),
    ColumnDef(1, "Bx", "nT", kind="Field", color=FIELD_COLORS[0]),
    ColumnDef(2, "By", "nT", kind="Field", color=FIELD_COLORS[1]),
    ColumnDef(3, "Bz", "nT", kind="Field", color=FIELD_COLORS[2]),
    ColumnDef(4, "Btot", "nT", kind="Field", color=FIELD_COLORS[3]),
    ColumnDef(5, "x", "R_S", kind="Coord"),
    ColumnDef(6, "y", "R_S", kind="Coord"),
    ColumnDef(7, "z", "R_S", kind="Coord"),
]

COLUMNS = {
    "KRTP": MAG_COLUMNS_KRTP,
    "KSM": MAG_COLUMNS_KSM,
    "KSO": MAG_COLUMNS_KSM,  # same layout as KSM
}


# --- Cassini mission epoch ---

MISSION_START = datetime.datetime(2004, 6, 30)
MISSION_END = datetime.datetime(2017, 9, 13)


# --- PDS file path generation ---


def mag_filepath(
    year: str = "YYYY",
    coords: str = "KRTP",
    resolution: str = "1min",
    data_root: Path | None = None,
) -> Path:
    """Build path to a PDS MAG TAB file.

    Parameters
    ----------
    year : str
        Year string or 'YYYY' template placeholder.
    coords : str
        Coordinate system: 'KRTP', 'KSM', 'KSO'.
    resolution : str
        '1min' or '1sec'.
    data_root : Path, optional
        Override for DATA_ROOT.
    """
    root = data_root or qp.DATA_ROOT
    cassini = root / "CASSINI-DATA"

    if resolution == "1min":
        dataset = "CO-E_SW_J_S-MAG-4-SUMM-1MINAVG-V2"
        fname = f"{year}_FGM_{coords}_1M.TAB"
    elif resolution == "1sec":
        dataset = "CO-E_SW_J_S-MAG-4-SUMM-1SECAVG-V2"
        fname = f"{year}_FGM_{coords}_1S.TAB"
    else:
        raise ValueError(f"Unknown resolution: {resolution!r}")

    return cassini / dataset / "DATA" / year / fname


def mag_filepaths_for_range(
    date_from: datetime.datetime,
    date_to: datetime.datetime,
    coords: str = "KRTP",
    resolution: str = "1min",
    data_root: Path | None = None,
) -> list[Path]:
    """Generate MAG file paths covering a date range (one file per year)."""
    paths = []
    for year in range(date_from.year, date_to.year + 1):
        p = mag_filepath(str(year), coords, resolution, data_root)
        if p.exists():
            paths.append(p)
    return paths


# --- ASCII data reading ---


def _parse_datetime(s: str, fmt: str = DATETIME_FMT) -> datetime.datetime:
    return datetime.datetime.strptime(s, fmt)


def read_timeseries_file(
    path: Path,
    date_from: datetime.datetime | None = None,
    date_to: datetime.datetime | None = None,
    datetime_fmt: str = DATETIME_FMT,
    date_column: int = 0,
    resolution_sec: float | None = None,
) -> list[list[str]]:
    """Read a PDS TAB file, optionally filtering by date range.

    Returns a list of rows, each row a list of string tokens.
    """
    path = Path(path)
    if not path.exists():
        return []

    # Determine read stride for downsampling
    stride = 1
    if resolution_sec is not None:
        with open(path) as f:
            t0 = _parse_datetime(f.readline().split()[date_column], datetime_fmt)
            t1 = _parse_datetime(f.readline().split()[date_column], datetime_fmt)
            native_dt = (t1 - t0).total_seconds()
            if native_dt > 0:
                stride = max(1, int(resolution_sec // native_dt))

    rows: list[list[str]] = []
    with open(path) as f:
        for n, line in enumerate(f):
            if n % stride != 0:
                continue
            tokens = line.split()
            if date_from is not None and date_to is not None:
                t = _parse_datetime(tokens[date_column], datetime_fmt)
                if t < date_from:
                    continue
                if t > date_to:
                    break
            rows.append(tokens)
    return rows


def select_data(
    date_from: datetime.datetime | str,
    date_to: datetime.datetime | str,
    coords: str = "KRTP",
    resolution: str = "1min",
    datetime_fmt: str = DATETIME_FMT,
    margin_sec: float = 0,
    resolution_sec: float | None = None,
    data_root: Path | None = None,
) -> list[list[str]]:
    """Select MAG data for a datetime interval across yearly files.

    This is the modernized replacement for cassinilib.SelectData.SelectData().
    """
    if isinstance(date_from, str):
        date_from = _parse_datetime(date_from, datetime_fmt)
    if isinstance(date_to, str):
        date_to = _parse_datetime(date_to, datetime_fmt)

    margin = datetime.timedelta(seconds=margin_sec)
    date_from = date_from - margin
    date_to = date_to + margin

    paths = mag_filepaths_for_range(date_from, date_to, coords, resolution, data_root)

    rows: list[list[str]] = []
    for p in paths:
        rows.extend(
            read_timeseries_file(
                p,
                date_from=date_from,
                date_to=date_to,
                datetime_fmt=datetime_fmt,
                resolution_sec=resolution_sec,
            )
        )
    return rows
