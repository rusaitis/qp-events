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

    # Full mission, no tracing, compressed float32:
    uv run python scripts/compute_dwell_grid.py --no-trace --compress zstd --float32

    # Custom grid resolution with tracing every 2 hours:
    uv run python scripts/compute_dwell_grid.py --n-r 30 --n-lat 45 --trace-every 120

    # Custom field model parameters:
    uv run python scripts/compute_dwell_grid.py --dp 0.02 --by-imf -0.3 --trace-step 0.05

Field model defaults (dp=0.017 nPa, By=-0.2 nT, Bz=0.1 nT) are nominal
values from Khurana (2020) representing average solar wind conditions at Saturn.
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

from qp.dwell.grid import DwellGridConfig, accumulate_inv_lat_grid, accumulate_with_regions
from qp.dwell.io import ZarrEncoding, save_zarr, to_xarray
from qp.dwell.tracing import TracingConfig, compute_invariant_latitudes
from qp.fieldline.kmag_model import SaturnFieldConfig
from qp.io.crossings import crossing_lookup_arrays, parse_crossing_list
from qp.io.pds import DATETIME_FMT, mag_filepath, read_timeseries_file

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
    sample_timestamps: np.ndarray,
    crossing_times_unix: np.ndarray,
    crossing_codes: np.ndarray,
) -> np.ndarray:
    """Assign a region code (MS/SH/SW) to each timestamp via vectorized lookup.

    Parameters
    ----------
    sample_timestamps : ndarray of float64
        POSIX timestamps for each sample.
    crossing_times_unix : ndarray of float64
        Sorted POSIX timestamps of boundary crossings.
    crossing_codes : ndarray of int
        Region code after each crossing.

    Notes
    -----
    Samples before the first boundary crossing are assigned code 9
    (unknown), since Cassini's magnetospheric region is not cataloged
    for that period.
    """
    idx = np.searchsorted(crossing_times_unix, sample_timestamps) - 1
    codes = np.where(
        (idx >= 0) & (idx < len(crossing_codes)),
        crossing_codes[idx],
        9,  # UNKNOWN
    )
    return codes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Cassini dwell-time grid from raw PDS MAG data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Time range ---
    time_group = parser.add_argument_group("Time range")
    time_group.add_argument("--year-from", type=int, default=2004,
                            help="First year to process")
    time_group.add_argument("--year-to", type=int, default=2017,
                            help="Last year to process")
    time_group.add_argument("--year", type=int, default=None,
                            help="Process a single year (overrides --year-from/--year-to)")

    # --- Grid configuration ---
    grid_group = parser.add_argument_group("Grid configuration")
    grid_group.add_argument("--n-r", type=int, default=100,
                            help="Number of radial bins")
    grid_group.add_argument("--n-lat", type=int, default=180,
                            help="Number of latitude bins")
    grid_group.add_argument("--n-lt", type=int, default=96,
                            help="Number of local time bins")
    grid_group.add_argument("--r-max", type=float, default=100.0,
                            help="Maximum radial distance (R_S)")

    # --- Tracing ---
    trace_group = parser.add_argument_group("KMAG tracing")
    trace_group.add_argument("--trace-every", type=int, default=60,
                             help="Trace every N minutes (60 = hourly)")
    trace_group.add_argument("--trace-step", type=float, default=0.1,
                             help="RK4 step size (R_S)")
    trace_group.add_argument("--trace-max-radius", type=float, default=100.0,
                             help="Outer tracing boundary (R_S)")
    trace_group.add_argument("--trace-min-radius", type=float, default=1.0,
                             help="Inner boundary / planet surface (R_S)")
    trace_group.add_argument("--no-trace", action="store_true",
                             help="Skip KMAG tracing entirely")

    # --- Field model ---
    field_group = parser.add_argument_group("KMAG field model")
    field_group.add_argument("--dp", type=float, default=0.017,
                             help="Solar wind dynamic pressure (nPa)")
    field_group.add_argument("--by-imf", type=float, default=-0.2,
                             help="IMF By component (nT, KSM)")
    field_group.add_argument("--bz-imf", type=float, default=0.1,
                             help="IMF Bz component (nT, KSM)")

    # --- Storage ---
    storage_group = parser.add_argument_group("Storage")
    storage_group.add_argument("--output", type=str, default="Output/dwell_grid.zarr",
                               help="Output zarr path")
    storage_group.add_argument("--compress", type=str, default="zstd",
                               choices=["zstd", "blosc", "none"],
                               help="Zarr compressor")
    storage_group.add_argument("--compress-level", type=int, default=3,
                               help="Compression level (1-9)")
    storage_group.add_argument("--float32", action="store_true",
                               help="Store as float32 instead of float64")

    # --- Misc ---
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def main():
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.year is not None:
        year_from, year_to = args.year, args.year
    else:
        year_from, year_to = args.year_from, args.year_to

    # Build configs from CLI args
    grid_config = DwellGridConfig(
        n_r=args.n_r, n_lat=args.n_lat, n_lt=args.n_lt,
        r_range=(0.0, args.r_max),
    )
    tracing_config = TracingConfig(
        trace_every_n=args.trace_every,
        step=args.trace_step,
        max_radius=args.trace_max_radius,
        min_radius=args.trace_min_radius,
    )
    field_config = SaturnFieldConfig(
        dp=args.dp, by_imf=args.by_imf, bz_imf=args.bz_imf,
    )
    zarr_encoding = ZarrEncoding(
        compressor=args.compress,
        compression_level=args.compress_level,
        dtype="float32" if args.float32 else "float64",
    )

    log.info("Grid: %s, r=[0, %.0f], lat=[-90, 90], LT=[0, 24]",
             grid_config.shape, args.r_max)

    # Load boundary crossings from raw text
    log.info("Loading boundary crossings...")
    crossings = parse_crossing_list()
    crossing_times_unix, crossing_codes = crossing_lookup_arrays(crossings)
    log.info("  → %d boundary crossings", len(crossing_times_unix))

    run_start = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Accumulators — use same names as REGION_CODES in grid.py
    from qp.dwell.grid import REGION_CODES
    grids_accum = {name: np.zeros(grid_config.shape) for name in ["total", *REGION_CODES.values()]}
    inv_lat_shape = (grid_config.n_lat, grid_config.n_lt)
    inv_lat_accum = {name: np.zeros(inv_lat_shape) for name in ["total", *REGION_CODES.values()]}

    all_x, all_y, all_z, all_times_unix = [], [], [], []
    total_samples = 0

    for year in range(year_from, year_to + 1):
        x, y, z, times = load_year_positions(year)
        if len(x) == 0:
            continue

        # Region codes for this year (vectorized lookup on POSIX timestamps)
        sample_unix = np.array([t.timestamp() for t in times], dtype=np.float64)
        codes = lookup_region_codes(sample_unix, crossing_times_unix, crossing_codes)

        # Accumulate into 3D spatial grids
        grids = accumulate_with_regions(x, y, z, codes, dt_minutes=1.0, config=grid_config)
        for name in grids_accum:
            grids_accum[name] += grids.get(name, np.zeros(grid_config.shape))

        # Accumulate into 2D dipole invariant latitude grids (instant, no tracing)
        inv_grids = accumulate_inv_lat_grid(x, y, z, dt_minutes=1.0, region_codes=codes, config=grid_config)
        for name in inv_lat_accum:
            inv_lat_accum[name] += inv_grids.get(name, np.zeros(inv_lat_shape))

        total_samples += len(x)

        # Collect for optional tracing
        if not args.no_trace:
            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            all_times_unix.append(sample_unix)

        log.info("Year %d: %d samples accumulated (total: %d)", year, len(x), total_samples)

    # Summary
    total_minutes = float(grids_accum["total"].sum())
    total_hours = total_minutes / 60
    expected_hours = (
        datetime.datetime(year_to + 1, 1, 1) - datetime.datetime(year_from, 1, 1)
    ).total_seconds() / 3600
    ms_hours = float(grids_accum["magnetosphere"].sum()) / 60
    sh_hours = float(grids_accum["magnetosheath"].sum()) / 60
    sw_hours = float(grids_accum["solar_wind"].sum()) / 60

    print(f"\n{'='*60}")
    print(f"Total samples:     {total_samples:,}")
    print(f"Total dwell time:  {total_hours:,.1f} hours ({total_hours/8766:.1f} years)")
    print(f"Expected (approx): {expected_hours:,.0f} hours")
    if total_hours > 0:
        print(f"  Magnetosphere:   {ms_hours:,.1f} h ({ms_hours/total_hours*100:.1f}%)")
        print(f"  Magnetosheath:   {sh_hours:,.1f} h ({sh_hours/total_hours*100:.1f}%)")
        print(f"  Solar wind:      {sw_hours:,.1f} h ({sw_hours/total_hours*100:.1f}%)")
    inv_hours = float(inv_lat_accum["total"].sum()) / 60
    print(f"Grid:              {grid_config.shape} (r×lat×LT)")
    print(f"Inv lat grid:      {inv_lat_shape} (inv_lat×LT), {inv_hours:,.0f} h mapped")
    print(f"Storage:           {zarr_encoding.compressor}, {zarr_encoding.dtype}")
    print(f"{'='*60}\n")

    attrs = {
        "year_from": year_from,
        "year_to": year_to,
        "total_samples": total_samples,
        "total_hours": total_hours,
        "dt_minutes": 1.0,
        "source": "PDS MAG 1-min KSM",
        "computation_started": run_start,
    }

    # KMAG tracing (optional)
    if not args.no_trace and all_x:
        log.info("Starting KMAG field line tracing...")
        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        z_all = np.concatenate(all_z)
        t_all = np.concatenate(all_times_unix)

        inv_north, inv_south, is_closed = compute_invariant_latitudes(
            x_all, y_all, z_all, t_all,
            config=tracing_config,
            field_config=field_config,
        )
        attrs["n_traces"] = len(inv_north)
        attrs["n_closed"] = int(np.sum(is_closed))

        # TODO: accumulate inv_lat grids from tracing results

    # Prefix inv lat grid names for clarity in the Dataset
    inv_lat_named = {f"dipole_inv_lat_{k}": v for k, v in inv_lat_accum.items()}

    ds = to_xarray(
        grids_accum, grid_config, attrs=attrs,
        tracing_config=tracing_config if not args.no_trace else None,
        field_config=field_config if not args.no_trace else None,
        inv_lat_grids=inv_lat_named,
    )
    print(ds)
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_zarr(ds, output_path, encoding=zarr_encoding)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
