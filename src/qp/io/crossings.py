"""Parse Jackman et al. 2019 boundary crossing list and generate
a time-indexed array of magnetosphere/magnetosheath/solar wind locations.

Modernized from boundary_crossings.py.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import qp

# Location codes
MS = 0  # magnetosphere
SH = 1  # magnetosheath
SW = 2  # solar wind
UNKNOWN = 9


@dataclass
class BoundaryCrossing:
    """Single boundary crossing event."""

    time: datetime.datetime
    crossing_type: str  # 'BS' or 'MP'
    direction: str  # 'I' (inbound) or 'O' (outbound), or extended codes
    x: float  # KSM position (R_S)
    y: float
    z: float


def parse_crossing_list(filepath: Path | None = None) -> list[BoundaryCrossing]:
    """Parse the Jackman et al. 2019 boundary crossing list.

    Default path: DATA/CASSINI-DATA/DataProducts/BS_MP_Crossing.txt
    """
    if filepath is None:
        filepath = qp.DATA_PRODUCTS / "BS_MP_Crossing.txt"
    filepath = Path(filepath)

    crossings = []
    with open(filepath) as f:
        for line in f:
            tokens = line.split()
            if len(tokens) < 9:
                continue
            year, doy, h, m = (
                int(tokens[0]),
                int(tokens[1]),
                int(tokens[2]),
                int(tokens[3]),
            )
            time = datetime.datetime.strptime(
                f"{year} {doy:03d} {h:02d}:{m:02d}", "%Y %j %H:%M"
            )
            crossings.append(
                BoundaryCrossing(
                    time=time,
                    crossing_type=tokens[4],
                    direction=tokens[5],
                    x=float(tokens[6]),
                    y=float(tokens[7]),
                    z=float(tokens[8]),
                )
            )
    return crossings


def _crossing_to_location(crossing_type: str, direction: str) -> int:
    """Map a crossing type+direction to the location code AFTER the crossing."""
    key = crossing_type + direction
    mapping = {
        "BSI": SH,  # crossed bow shock inbound → now in magnetosheath
        "BSO": SW,  # crossed bow shock outbound → now in solar wind
        "MPI": MS,  # crossed magnetopause inbound → now in magnetosphere
        "MPO": SH,  # crossed magnetopause outbound → now in magnetosheath
    }
    if key in mapping:
        return mapping[key]

    # Extended direction codes from Jackman list
    direction_mapping = {
        "E_SW": SW,
        "E_SH": SH,
        "E_SP": MS,
        "E_MP": MS,
    }
    if direction in direction_mapping:
        return direction_mapping[direction]
    if direction.startswith("S"):
        return UNKNOWN
    return UNKNOWN


def build_crossing_timeseries(
    crossings: list[BoundaryCrossing] | None = None,
    interval_sec: int = 3600,
    mission_start: datetime.datetime | None = None,
    mission_end: datetime.datetime | None = None,
) -> np.ndarray:
    """Build a time-indexed array of spacecraft location.

    Returns shape (3, N) object array:
        row 0: datetime timestamps
        row 1: location codes (0=MS, 1=SH, 2=SW, 9=unknown)
        row 2: crossing label at transition points, None elsewhere
    """
    if crossings is None:
        crossings = parse_crossing_list()

    t0 = mission_start or datetime.datetime(2004, 6, 30)
    t1 = mission_end or datetime.datetime(2017, 9, 15)
    delta = datetime.timedelta(seconds=interval_sec)

    # Build time grid
    times = []
    t = t0
    while t <= t1:
        t += delta
        times.append(t)

    n = len(times)
    times_arr = np.array(times, dtype=object)
    locs = np.full(n, UNKNOWN, dtype=object)
    labels = np.full(n, None, dtype=object)

    # Walk through crossings and fill location codes
    prev_idx = 0
    for c in crossings:
        loc = _crossing_to_location(c.crossing_type, c.direction)
        for i in range(prev_idx, n):
            if times_arr[i] >= c.time:
                locs[i:] = loc
                if c.crossing_type in ("BS", "MP"):
                    labels[i] = c.crossing_type + c.direction
                prev_idx = i
                break

    return np.vstack((times_arr, locs, labels))


def crossing_lookup_arrays(
    crossings: list[BoundaryCrossing] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted crossing times and region codes for vectorized lookup.

    Parameters
    ----------
    crossings : list of BoundaryCrossing, optional
        Parsed crossing list. If None, parses the default file.

    Returns
    -------
    times_unix : ndarray of float64
        POSIX timestamps of each boundary crossing, sorted.
    region_codes : ndarray of int
        Region code AFTER each crossing (0=MS, 1=SH, 2=SW, 9=unknown).
    """
    if crossings is None:
        crossings = parse_crossing_list()

    times = np.array([c.time.timestamp() for c in crossings], dtype=np.float64)
    codes = np.array(
        [_crossing_to_location(c.crossing_type, c.direction) for c in crossings],
        dtype=int,
    )
    # Ensure sorted by time
    order = np.argsort(times)
    return times[order], codes[order]


def export_crossings(
    output_path: Path | None = None,
    **kwargs,
) -> np.ndarray:
    """Parse crossings and save to .npy file. Returns the array."""
    arr = build_crossing_timeseries(**kwargs)
    if output_path is None:
        output_path = qp.DATA_PRODUCTS / "CROSSINGS.npy"
    np.save(output_path, arr)
    print(f"Saved crossings to {output_path} — shape {arr.shape}")
    return arr
