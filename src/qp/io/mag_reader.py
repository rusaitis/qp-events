r"""Read and write Cassini MAG data segments.

Replaces ``cassinilib/NewSignal.py:readSignal()`` and ``saveSignal()``
with a structured ``MagSegment`` dataclass instead of a list of
monolithic ``NewSignal`` objects.

Usage
-----
>>> from qp.io.mag_reader import read_mag_segment
>>> seg = read_mag_segment(
...     datetime(2007, 1, 2), datetime(2007, 1, 3), coord="KRTP"
... )
>>> seg.field_names
('Br', 'Bth', 'Bphi', 'Btot')
>>> seg.fields.shape
(1440, 4)
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qp.io.pds import COLUMNS, DATETIME_FMT, select_data
from qp.time_utils import parse_datetime, to_timestamp

log = logging.getLogger(__name__)


@dataclass(slots=True)
class MagSegment:
    r"""A time segment of Cassini MAG data.

    Replaces the old list-of-``NewSignal`` objects returned by
    ``cassinilib.readSignal()``.

    Attributes
    ----------
    time_dt : list[datetime.datetime]
        Timestamps as datetime objects.
    time_unix : ndarray
        POSIX timestamps (float64, seconds since 1970-01-01 UTC).
    fields : ndarray, shape (N, n_fields)
        Magnetic field components (nT). Column order matches ``field_names``.
    coords : ndarray, shape (N, n_coords)
        Spacecraft position. Column order matches ``coord_names``.
        For KRTP: (r, theta, phi) in (R_S, radians, radians).
        For KSM/KSO: (x, y, z) in R_S.
    dt : float
        Sampling interval in seconds.
    coord_system : str
        Coordinate system ('KRTP', 'KSM', 'KSO').
    field_names : tuple[str, ...]
        Names of the field columns.
    coord_names : tuple[str, ...]
        Names of the coordinate columns.
    local_time : ndarray or None
        Local time in hours (KRTP only).
    """

    time_dt: list[datetime.datetime]
    time_unix: np.ndarray
    fields: np.ndarray
    coords: np.ndarray
    dt: float
    coord_system: str
    field_names: tuple[str, ...]
    coord_names: tuple[str, ...]
    local_time: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        return len(self.time_dt)

    @property
    def duration_hours(self) -> float:
        if len(self.time_dt) < 2:
            return 0.0
        return (self.time_dt[-1] - self.time_dt[0]).total_seconds() / 3600


def read_mag_segment(
    date_from: datetime.datetime | str,
    date_to: datetime.datetime | str,
    coord: str = "KRTP",
    dt: float = 60.0,
    margin_sec: float = 0.0,
    data_root: Path | None = None,
) -> MagSegment:
    r"""Read Cassini MAG data for a time interval.

    Parameters
    ----------
    date_from, date_to : datetime or str
        Time interval. Strings are parsed with ISO format.
    coord : str
        Coordinate system: 'KRTP', 'KSM', 'KSO'.
    dt : float
        Desired sampling interval in seconds.
    margin_sec : float
        Extra margin at each end of the interval.
    data_root : Path, optional
        Override for ``qp.DATA_ROOT``.

    Returns
    -------
    MagSegment
        Structured data segment.

    Raises
    ------
    ValueError
        If no data found for the requested interval.
    """
    if isinstance(date_from, str):
        date_from = parse_datetime(date_from)
    if isinstance(date_to, str):
        date_to = parse_datetime(date_to)

    rows = select_data(
        date_from,
        date_to,
        coords=coord,
        margin_sec=margin_sec,
        resolution_sec=dt if dt != 60.0 else None,
        data_root=data_root,
    )

    if not rows:
        raise ValueError(f"No MAG data found for {date_from} to {date_to} in {coord}")

    columns = COLUMNS.get(coord)
    if columns is None:
        raise ValueError(f"Unknown coordinate system: {coord!r}")

    data = np.array(rows)  # shape (N, n_columns), dtype=str

    # Parse time column
    time_col = columns[0]
    time_dt = [
        datetime.datetime.strptime(t, time_col.fmt or DATETIME_FMT)
        for t in data[:, time_col.index]
    ]
    time_unix = np.array([to_timestamp(t) for t in time_dt])

    # Extract field columns
    field_cols = [c for c in columns if c.kind == "Field"]
    field_names = tuple(c.name for c in field_cols)
    fields = np.column_stack([data[:, c.index].astype(float) for c in field_cols])

    # Extract coordinate columns
    coord_cols = [c for c in columns if c.kind == "Coord"]
    coord_names = tuple(c.name for c in coord_cols)
    coords = np.column_stack([data[:, c.index].astype(float) for c in coord_cols])

    # Convert angular coordinates to radians for KRTP
    if coord == "KRTP":
        for i, c in enumerate(coord_cols):
            if c.units == "deg":
                coords[:, i] = np.deg2rad(coords[:, i])

    # Extract local time if available
    lt_cols = [c for c in columns if c.kind == "lt"]
    local_time = None
    if lt_cols:
        local_time = data[:, lt_cols[0].index].astype(float)

    # Compute actual dt from the data
    actual_dt = dt
    if len(time_unix) > 1:
        actual_dt = float(np.median(np.diff(time_unix)))

    return MagSegment(
        time_dt=time_dt,
        time_unix=time_unix,
        fields=fields,
        coords=coords,
        dt=actual_dt,
        coord_system=coord,
        field_names=field_names,
        coord_names=coord_names,
        local_time=local_time,
    )


def save_segment(segment: MagSegment, path: Path) -> None:
    r"""Save a MAG segment to a .npz file.

    Parameters
    ----------
    segment : MagSegment
        Data to save.
    path : Path
        Output file path (should end in .npz).
    """
    path = Path(path)
    np.savez_compressed(
        path,
        time_unix=segment.time_unix,
        fields=segment.fields,
        coords=segment.coords,
        dt=np.array([segment.dt]),
        coord_system=np.array([segment.coord_system]),
        field_names=np.array(segment.field_names),
        coord_names=np.array(segment.coord_names),
        local_time=segment.local_time
        if segment.local_time is not None
        else np.array([]),
    )
    log.info("Saved MAG segment to %s (%d samples)", path, segment.n_samples)


def load_segment(path: Path) -> MagSegment:
    r"""Load a MAG segment from a .npz file.

    Parameters
    ----------
    path : Path
        Input file path.

    Returns
    -------
    MagSegment
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        time_unix = f["time_unix"]
        fields = f["fields"]
        coords = f["coords"]
        dt_val = float(f["dt"][0])
        coord_system = str(f["coord_system"][0])
        field_names = tuple(f["field_names"])
        coord_names = tuple(f["coord_names"])
        lt_arr = f["local_time"]
        local_time = lt_arr if len(lt_arr) > 0 else None

    from qp.time_utils import from_timestamp

    time_dt = [from_timestamp(ts).replace(tzinfo=None) for ts in time_unix]

    return MagSegment(
        time_dt=time_dt,
        time_unix=time_unix,
        fields=fields,
        coords=coords,
        dt=dt_val,
        coord_system=coord_system,
        field_names=field_names,
        coord_names=coord_names,
        local_time=local_time,
    )
