#!/usr/bin/env python
"""Compute Cassini dwell-time grid from raw PDS MAG data.

Reads yearly PDS KSM 1-min MAG files, converts positions to
(r, magnetic_latitude, local_time), accumulates into a 3D grid,
optionally traces KMAG field lines for invariant latitude,
and saves the result as an xarray zarr store.

Usage
-----
    # Single year (fast, for testing):
    uv run python scripts/compute_dwell_grid.py --year 2007

    # Full mission, no tracing (fast):
    uv run python scripts/compute_dwell_grid.py --no-trace

    # Full mission with KMAG tracing every 2 hours:
    uv run python scripts/compute_dwell_grid.py --trace-every 120
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is on path when running as script
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from qp.dwell.grid import DwellGridConfig, accumulate_with_regions
from qp.dwell.io import save_zarr, to_xarray
from qp.dwell.tracing import compute_invariant_latitudes
from qp.io.crossings import build_crossing_timeseries, parse_crossing_list
from qp.io.pds import DATETIME_FMT, mag_filepath, read_timeseries_file
from qp.time_utils import to_timestamp

log = logging.getLogger(__name__)


def load_year_positions(year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Read one year of PDS KSM 1-min MAG data and extract positions.

    Returns (x, y, z, datetimes) arrays. x/y/z in R_S (KSM).
    """
    path = mag_filepath(str(year), coords="KSM")
    if not path.exists():
        log.warning("No PDS file for year %d: %s", year, path)
        return np.array([]), np.array([]), np.array([]), []

    log.info("Reading %s ...", path.name)
    rows = read_timeseries_file(path)

    if not rows:
        return np.array([]), np.array([]), np.array([]), []

    data = np.array(rows)
    # KSM columns: 0=Time, 1=Bx, 2=By, 3=Bz, 4=Btot, 5=x, 6=y, 7=z
    times = [datetime.datetime.strptime(t, DATETIME_FMT) for t in data[:, 0]]
    x = data[:, 5].astype(float)
    y = data[:, 6].astype(float)
    z = data[:, 7].astype(float)

    log.info("  → %d samples, %.1f days", len(x), len(x) / 1440)
    return x, y, z, times


def lookup_region_codes(
    times: list[datetime.datetime],
    crossing_times: np.ndarray,
    crossing_codes: np.ndarray,
) -> np.ndarray:
    """Assign a region code (MS/SH/SW) to each timestamp."""
    codes = np.full(len(times), 9, dtype=int)  # default: unknown
    for i, t in enumerate(times):
        idx = np.searchsorted(crossing_times, t) - 1
        if 0 <= idx < len(crossing_codes):
            codes[i] = int(crossing_codes[idx])
    return codes


def main():
    parser = argparse.ArgumentParser(description="Compute Cassini dwell-time grid")
    parser.add_argument("--year-from", type=int, default=2004)
    parser.add_argument("--year-to", type=int, default=2017)
    parser.add_argument("--year", type=int, default=None,
                        help="Process a single year (overrides --year-from/--year-to)")
    parser.add_argument("--trace-every", type=int, default=60,
                        help="KMAG trace every N minutes (default: 60 = hourly)")
    parser.add_argument("--no-trace", action="store_true",
                        help="Skip KMAG tracing (spatial grid only)")
    parser.add_argument("--output", type=str, default="Output/dwell_grid.zarr")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.year is not None:
        year_from, year_to = args.year, args.year
    else:
        year_from, year_to = args.year_from, args.year_to

    config = DwellGridConfig()
    log.info("Grid: %s, r=[%.0f, %.0f], lat=[%.0f, %.0f], LT=[%.0f, %.0f]",
             config.shape, *config.r_range, *config.lat_range, *config.lt_range)

    # Load boundary crossings from raw text
    log.info("Loading boundary crossings...")
    crossings = parse_crossing_list()
    crossing_ts = build_crossing_timeseries(crossings)
    crossing_datetimes = np.array(crossing_ts[0])
    crossing_codes = np.array(crossing_ts[1], dtype=int)
    log.info("  → %d hourly crossing entries", len(crossing_datetimes))

    # Accumulators
    grid_total = np.zeros(config.shape)
    grid_ms = np.zeros(config.shape)
    grid_sh = np.zeros(config.shape)
    grid_sw = np.zeros(config.shape)
    grid_unknown = np.zeros(config.shape)

    all_x, all_y, all_z, all_times_unix = [], [], [], []
    total_samples = 0

    for year in range(year_from, year_to + 1):
        x, y, z, times = load_year_positions(year)
        if len(x) == 0:
            continue

        # Region codes for this year
        codes = lookup_region_codes(times, crossing_datetimes, crossing_codes)

        # Accumulate into grids
        grids = accumulate_with_regions(x, y, z, codes, dt_minutes=1.0, config=config)
        grid_total += grids["total"]
        grid_ms += grids.get("ms", np.zeros(config.shape))
        grid_sh += grids.get("sh", np.zeros(config.shape))
        grid_sw += grids.get("sw", np.zeros(config.shape))
        grid_unknown += grids.get("unknown", np.zeros(config.shape))

        total_samples += len(x)

        # Collect for optional tracing
        if not args.no_trace:
            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            all_times_unix.extend([to_timestamp(t) for t in times])

        log.info("Year %d: %d samples accumulated (total: %d)", year, len(x), total_samples)

    # Summary
    total_minutes = float(grid_total.sum())
    total_hours = total_minutes / 60
    expected_hours = (
        datetime.datetime(year_to + 1, 1, 1) - datetime.datetime(year_from, 1, 1)
    ).total_seconds() / 3600
    ms_hours = float(grid_ms.sum()) / 60
    sh_hours = float(grid_sh.sum()) / 60
    sw_hours = float(grid_sw.sum()) / 60

    print(f"\n{'='*60}")
    print(f"Total samples:     {total_samples:,}")
    print(f"Total dwell time:  {total_hours:,.1f} hours ({total_hours/8766:.1f} years)")
    print(f"Expected (approx): {expected_hours:,.0f} hours")
    print(f"  Magnetosphere:   {ms_hours:,.1f} h ({ms_hours/total_hours*100:.1f}%)")
    print(f"  Magnetosheath:   {sh_hours:,.1f} h ({sh_hours/total_hours*100:.1f}%)")
    print(f"  Solar wind:      {sw_hours:,.1f} h ({sw_hours/total_hours*100:.1f}%)")
    print(f"{'='*60}\n")

    # Build xarray Dataset
    grids_dict = {
        "total": grid_total,
        "magnetosphere": grid_ms,
        "magnetosheath": grid_sh,
        "solar_wind": grid_sw,
        "unknown": grid_unknown,
    }

    attrs = {
        "year_from": year_from,
        "year_to": year_to,
        "total_samples": total_samples,
        "total_hours": total_hours,
        "dt_minutes": 1.0,
        "source": "PDS MAG 1-min KSM",
    }

    # KMAG tracing (optional)
    if not args.no_trace and all_x:
        log.info("Starting KMAG field line tracing (every %d min)...", args.trace_every)
        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        z_all = np.concatenate(all_z)
        t_all = np.array(all_times_unix)

        inv_north, inv_south, is_closed = compute_invariant_latitudes(
            x_all, y_all, z_all, t_all,
            trace_every_n=args.trace_every,
        )
        attrs["n_traces"] = len(inv_north)
        attrs["n_closed"] = int(np.sum(is_closed))
        attrs["trace_every_minutes"] = args.trace_every

        # TODO: accumulate inv_lat grids from tracing results

    ds = to_xarray(grids_dict, config, attrs=attrs)
    print(ds)
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_zarr(ds, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
