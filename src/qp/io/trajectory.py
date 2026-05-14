r"""Mission-wide spacecraft trajectory + Jackman region tagging.

These helpers were originally embedded in
``scripts/compute_dwell_grid.py`` and ``scripts/bin_events_round8.py``;
three downstream scripts imported them via ``sys.path`` hacks. They are
ordinary library logic (PDS read + concatenate + boundary-crossing
lookup), so they live here in :mod:`qp.io`.

Two layers:

- :func:`load_year_positions` reads one year of PDS KSM 1-min MAG data
  and returns positions in $R_S$ plus the corresponding datetimes.
- :func:`load_mission_trajectory` aggregates a range of years into one
  sorted ``(t_unix, x, y, z, btotal)`` tuple.

For region tagging:

- :func:`lookup_region_codes` does the vectorized
  ``np.searchsorted`` against Jackman 2019 boundary crossings.
- :func:`load_region_codes` is the one-shot convenience that loads the
  crossings file and returns codes for a given ``t_unix`` array.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np

from qp.io.crossings import crossing_lookup_arrays, parse_crossing_list
from qp.io.pds import DATETIME_FMT, mag_filepath, read_timeseries_file

__all__ = [
    "load_year_positions",
    "load_mission_trajectory",
    "lookup_region_codes",
    "load_region_codes",
]

log = logging.getLogger(__name__)

_UNIX_EPOCH = datetime.datetime(1970, 1, 1)
#: Region code assigned to samples that predate the first Jackman
#: boundary crossing (the catalogue does not cover them).
UNKNOWN_REGION_CODE: int = 9


def load_year_positions(
    year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[datetime.datetime]]:
    """Read one year of PDS KSM 1-min MAG data and return positions.

    Returns ``(x, y, z, btotal, datetimes)``; positions in $R_S$
    (KSM), ``btotal`` in nT. Returns empty arrays if the year file is
    missing.
    """
    path = mag_filepath(str(year), coords="KSM")
    if not path.exists():
        log.warning("No PDS file for year %d: %s", year, path)
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    log.info("Reading %s ...", path.name)
    rows = read_timeseries_file(path)
    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    data = np.array(rows)
    # KSM columns: 0=Time, 1=Bx, 2=By, 3=Bz, 4=Btot, 5=x, 6=y, 7=z
    times = [datetime.datetime.strptime(t, DATETIME_FMT) for t in data[:, 0]]
    x = data[:, 5].astype(float)
    y = data[:, 6].astype(float)
    z = data[:, 7].astype(float)
    btotal = data[:, 4].astype(float)
    log.info("  → %d samples, %.1f days", len(x), len(x) / 1440)
    return x, y, z, btotal, times


def load_mission_trajectory(
    year_from: int = 2004,
    year_to: int = 2017,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate :func:`load_year_positions` across a year range.

    Returns ``(t_unix, x, y, z, btotal)``, sorted by time. POSIX
    seconds for ``t_unix``, $R_S$ for positions, nT for ``btotal``.
    """
    all_t: list[np.ndarray] = []
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    all_b: list[np.ndarray] = []
    for year in range(year_from, year_to + 1):
        x, y, z, btotal, times = load_year_positions(year)
        if len(x) == 0:
            continue
        t_unix = np.array(
            [(t - _UNIX_EPOCH).total_seconds() for t in times],
            dtype=float,
        )
        all_t.append(t_unix)
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
        all_b.append(btotal)
    if not all_t:
        raise RuntimeError("no trajectory data loaded — check DATA path")
    t = np.concatenate(all_t)
    order = np.argsort(t)
    return (
        t[order],
        np.concatenate(all_x)[order],
        np.concatenate(all_y)[order],
        np.concatenate(all_z)[order],
        np.concatenate(all_b)[order],
    )


def lookup_region_codes(
    sample_timestamps: np.ndarray,
    crossing_times_unix: np.ndarray,
    crossing_codes: np.ndarray,
) -> np.ndarray:
    """Assign a region code (MS/SH/SW) per timestamp via vectorized lookup.

    Samples before the first crossing get :data:`UNKNOWN_REGION_CODE`,
    since Cassini's magnetospheric region is not cataloged for that
    period. ``crossing_times_unix`` must be sorted ascending; a sample at
    exactly a crossing time gets the *previous* region (``np.searchsorted``
    side='left' convention) — at 1-min MAG cadence the off-by-one-minute
    edge case is below the catalog's temporal resolution.
    """
    idx = np.searchsorted(crossing_times_unix, sample_timestamps) - 1
    return np.where(
        (idx >= 0) & (idx < len(crossing_codes)),
        crossing_codes[idx],
        UNKNOWN_REGION_CODE,
    )


def load_region_codes(t_unix: np.ndarray) -> np.ndarray:
    """One-shot region tagging: parse Jackman crossings and look up.

    Convenience for scripts that don't care about the crossing arrays
    themselves.
    """
    crossings = parse_crossing_list()
    cross_unix, cross_codes = crossing_lookup_arrays(crossings)
    return lookup_region_codes(t_unix, cross_unix, cross_codes)
