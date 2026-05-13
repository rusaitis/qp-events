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
from pathlib import Path

import numpy as np

# Ensure src/ is on path when running as script
_project_root = Path(__file__).resolve().parent.parent

from qp.cli import (
    add_field_model_args,
    add_tracing_args,
    add_verbosity_arg,
    add_workers_arg,
    add_year_range_args,
)
from qp.constants import J2000_POSIX
from qp.coords.ksm import local_time as compute_lt
from qp.dwell.grid import (
    DwellGridConfig,
    accumulate_inv_lat_grid,
    accumulate_traced_inv_lat_grid,
    accumulate_weak_field_grid,
    accumulate_with_regions,
)
from qp.dwell.io import ZarrEncoding, save_zarr, to_xarray
from qp.dwell.tracing import TracingConfig, compute_invariant_latitudes_parallel
from qp.fieldline.kmag_model import SaturnFieldConfig
from qp.io.crossings import crossing_lookup_arrays, parse_crossing_list
from qp.io.trajectory import load_year_positions, lookup_region_codes

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Cassini dwell-time grid from raw PDS MAG data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Time range ---
    time_group = parser.add_argument_group("Time range")
    add_year_range_args(time_group)
    time_group.add_argument(
        "--year",
        type=int,
        default=None,
        help="Process a single year (overrides --year-from/--year-to)",
    )

    # --- Grid configuration ---
    grid_group = parser.add_argument_group("Grid configuration")
    grid_group.add_argument(
        "--n-r", type=int, default=100, help="Number of radial bins"
    )
    grid_group.add_argument(
        "--n-lat", type=int, default=180, help="Number of latitude bins"
    )
    grid_group.add_argument(
        "--n-lt", type=int, default=96, help="Number of local time bins"
    )
    grid_group.add_argument(
        "--r-max", type=float, default=100.0, help="Maximum radial distance (R_S)"
    )

    # --- Tracing ---
    trace_group = parser.add_argument_group("KMAG tracing")
    trace_group.add_argument(
        "--trace-every",
        type=int,
        default=10,
        help="Trace every N minutes (10 = every 10 min)",
    )
    add_tracing_args(trace_group)
    trace_group.add_argument(
        "--trace-min-radius",
        type=float,
        default=1.0,
        help="Inner boundary / planet surface (R_S)",
    )
    trace_group.add_argument(
        "--trace-max-steps",
        type=int,
        default=20_000,
        help="Max RK4 steps per trace arm (caps pathological traces)",
    )
    add_workers_arg(trace_group)
    trace_group.add_argument(
        "--no-region-filter",
        action="store_true",
        help="Trace all samples (default: magnetosphere only)",
    )
    trace_group.add_argument(
        "--no-trace", action="store_true", help="Skip KMAG tracing entirely"
    )
    trace_group.add_argument(
        "--include-equatorial",
        action="store_true",
        help=(
            "Also accumulate the equatorial-r schema "
            "(kmag_eq_r × LT) alongside the existing schemas. "
            "Free since equatorial apex comes from the same trace."
        ),
    )
    trace_group.add_argument(
        "--equatorial-only",
        action="store_true",
        help=(
            "Produce a sibling zarr with ONLY the equatorial-r "
            "schema (kmag_eq_r × LT, with closed-only variants). "
            "Useful to add the new schema next to an existing dwell "
            "grid without re-running the full pipeline."
        ),
    )

    # --- Field model ---
    field_group = parser.add_argument_group("KMAG field model")
    add_field_model_args(field_group)

    # --- Storage ---
    storage_group = parser.add_argument_group("Storage")
    storage_group.add_argument(
        "--output", type=str, default="Output/dwell_grid.zarr", help="Output zarr path"
    )
    storage_group.add_argument(
        "--compress",
        type=str,
        default="zstd",
        choices=["zstd", "blosc", "none"],
        help="Zarr compressor",
    )
    storage_group.add_argument(
        "--compress-level", type=int, default=3, help="Compression level (1-9)"
    )
    storage_group.add_argument(
        "--float32", action="store_true", help="Store as float32 instead of float64"
    )

    # --- Misc ---
    add_verbosity_arg(parser)

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
        n_r=args.n_r,
        n_lat=args.n_lat,
        n_lt=args.n_lt,
        r_range=(0.0, args.r_max),
    )
    tracing_config = TracingConfig(
        trace_every_n=args.trace_every,
        step=args.trace_step,
        max_radius=args.trace_max_radius,
        min_radius=args.trace_min_radius,
        max_steps=args.trace_max_steps,
        region_filter=None if args.no_region_filter else (0,),
        n_workers=args.workers,
    )
    field_config = SaturnFieldConfig(
        dp=args.dp,
        by_imf=args.by_imf,
        bz_imf=args.bz_imf,
    )
    zarr_encoding = ZarrEncoding(
        compressor=args.compress,
        compression_level=args.compress_level,
        dtype="float32" if args.float32 else "float64",
    )

    log.info(
        "Grid: %s, r=[0, %.0f], lat=[-90, 90], LT=[0, 24]",
        grid_config.shape,
        args.r_max,
    )

    # Load boundary crossings from raw text
    log.info("Loading boundary crossings...")
    crossings = parse_crossing_list()
    crossing_times_unix, crossing_codes = crossing_lookup_arrays(crossings)
    log.info("  → %d boundary crossings", len(crossing_times_unix))

    run_start = datetime.datetime.now(datetime.UTC).isoformat()

    # Accumulators — use same names as REGION_CODES in grid.py
    from qp.dwell.grid import REGION_CODES

    region_names = ["total", *REGION_CODES.values()]

    def _zeros_per_region(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        return {name: np.zeros(shape) for name in region_names}

    grids_accum = _zeros_per_region(grid_config.shape)
    inv_lat_shape = (grid_config.n_lat, grid_config.n_lt)
    inv_lat_accum = _zeros_per_region(inv_lat_shape)
    weak_field_accum = _zeros_per_region(inv_lat_shape)

    all_x, all_y, all_z, all_times_unix, all_codes = [], [], [], [], []
    total_samples = 0

    for year in range(year_from, year_to + 1):
        x, y, z, btotal, times = load_year_positions(year)
        if len(x) == 0:
            continue

        # Region codes for this year (vectorized lookup on POSIX timestamps)
        sample_unix = np.array([t.timestamp() for t in times], dtype=np.float64)
        codes = lookup_region_codes(sample_unix, crossing_times_unix, crossing_codes)

        # Accumulate into 3D spatial grids
        grids = accumulate_with_regions(
            x, y, z, codes, dt_minutes=1.0, config=grid_config
        )
        for name in grids_accum:
            grids_accum[name] += grids.get(name, np.zeros(grid_config.shape))

        # Accumulate into 2D dipole invariant latitude grids (instant, no tracing)
        inv_grids = accumulate_inv_lat_grid(
            x, y, z, dt_minutes=1.0, region_codes=codes, config=grid_config
        )
        for name in inv_lat_accum:
            inv_lat_accum[name] += inv_grids.get(name, np.zeros(inv_lat_shape))

        # Accumulate weak-field (plasma sheet proxy) dwell time
        wf_grids = accumulate_weak_field_grid(
            x,
            y,
            z,
            btotal,
            dt_minutes=1.0,
            b_threshold=2.0,
            region_codes=codes,
            config=grid_config,
        )
        for name in weak_field_accum:
            weak_field_accum[name] += wf_grids.get(name, np.zeros(inv_lat_shape))

        total_samples += len(x)

        # Collect for optional tracing
        if not args.no_trace:
            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            all_times_unix.append(sample_unix)
            all_codes.append(codes)

        log.info(
            "Year %d: %d samples accumulated (total: %d)", year, len(x), total_samples
        )

    # Summary
    total_minutes = float(grids_accum["total"].sum())
    total_hours = total_minutes / 60
    expected_hours = (
        datetime.datetime(year_to + 1, 1, 1) - datetime.datetime(year_from, 1, 1)
    ).total_seconds() / 3600
    ms_hours = float(grids_accum["magnetosphere"].sum()) / 60
    sh_hours = float(grids_accum["magnetosheath"].sum()) / 60
    sw_hours = float(grids_accum["solar_wind"].sum()) / 60

    inv_hours = float(inv_lat_accum["total"].sum()) / 60
    wf_hours = float(weak_field_accum["total"].sum()) / 60
    region_lines = ""
    if total_hours > 0:
        region_lines = (
            f"  Magnetosphere:   {ms_hours:,.1f} h ({ms_hours / total_hours * 100:.1f}%)\n"
            f"  Magnetosheath:   {sh_hours:,.1f} h ({sh_hours / total_hours * 100:.1f}%)\n"
            f"  Solar wind:      {sw_hours:,.1f} h ({sw_hours / total_hours * 100:.1f}%)\n"
        )
    sep = "=" * 60
    print(
        f"\n{sep}\n"
        f"Total samples:     {total_samples:,}\n"
        f"Total dwell time:  {total_hours:,.1f} hours ({total_hours / 8766:.1f} years)\n"
        f"Expected (approx): {expected_hours:,.0f} hours\n"
        f"{region_lines}"
        f"Grid:              {grid_config.shape} (r×lat×LT)\n"
        f"Inv lat grid:      {inv_lat_shape} (inv_lat×LT), {inv_hours:,.0f} h mapped\n"
        f"Weak field (<2nT): {wf_hours:,.0f} h (plasma sheet proxy)\n"
        f"Storage:           {zarr_encoding.compressor}, {zarr_encoding.dtype}\n"
        f"{sep}\n"
    )

    attrs = {
        "year_from": year_from,
        "year_to": year_to,
        "total_samples": total_samples,
        "total_hours": total_hours,
        "dt_minutes": 1.0,
        "source": "PDS MAG 1-min KSM",
        "boundary_crossings_source": "Jackman et al. 2019",
        "time_epoch": "J2000 (POSIX - 946728000.0)",
        "dipole_offset_RS": 0.037,
        "b_threshold_nT": 2.0,
        "conjugate_convention": (
            "KMAG inv lat signed by spacecraft hemisphere: "
            "z >= 0 uses northern footpoint, z < 0 uses southern"
        ),
        "computation_started": run_start,
    }

    # KMAG tracing (optional)
    kmag_inv_lat_named: dict[str, np.ndarray] = {}
    kmag_eq_r_named: dict[str, np.ndarray] = {}
    want_equatorial = args.include_equatorial or args.equatorial_only
    if not args.no_trace and all_x:
        log.info("Starting KMAG field line tracing...")
        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        z_all = np.concatenate(all_z)
        t_all = np.concatenate(all_times_unix)
        codes_all = np.concatenate(all_codes)

        # Convert POSIX timestamps to J2000 seconds for KMAG model
        t_j2000 = t_all - J2000_POSIX

        result = compute_invariant_latitudes_parallel(
            x_all,
            y_all,
            z_all,
            t_j2000,
            config=tracing_config,
            field_config=field_config,
            region_codes=codes_all,
        )
        attrs["n_traces"] = result.n_traces
        attrs["n_closed"] = result.n_closed

        # Subsample local_time, z, and region codes to match tracing indices
        indices = np.arange(0, len(x_all), tracing_config.trace_every_n)
        lt_sub = compute_lt(x_all, y_all)[indices]
        z_sub = z_all[indices]
        codes_sub = codes_all[indices]
        dt_trace = float(tracing_config.trace_every_n)

        # Accumulate KMAG inv lat grids (all field lines)
        kmag_grids = accumulate_traced_inv_lat_grid(
            result.inv_lat_north,
            result.inv_lat_south,
            result.is_closed,
            lt_sub,
            z_sub,
            dt_minutes=dt_trace,
            region_codes=codes_sub,
            config=grid_config,
        )
        for k, v in kmag_grids.items():
            kmag_inv_lat_named[f"kmag_inv_lat_{k}"] = v

        # Accumulate closed-only grids
        kmag_closed = accumulate_traced_inv_lat_grid(
            result.inv_lat_north,
            result.inv_lat_south,
            result.is_closed,
            lt_sub,
            z_sub,
            dt_minutes=dt_trace,
            region_codes=codes_sub,
            closed_only=True,
            config=grid_config,
        )
        for k, v in kmag_closed.items():
            kmag_inv_lat_named[f"kmag_inv_lat_closed_{k}"] = v

        kmag_hours = float(kmag_grids["total"].sum()) / 60
        closed_hours = float(kmag_closed["total"].sum()) / 60
        print(
            f"KMAG inv lat:      {inv_lat_shape}, {kmag_hours:,.0f} h mapped, {closed_hours:,.0f} h closed"
        )

        if want_equatorial:
            from qp.dwell.grid import accumulate_kmag_eq_r_grid

            eq_r_grids = accumulate_kmag_eq_r_grid(
                result.l_equatorial,
                result.is_closed,
                lt_sub,
                dt_minutes=dt_trace,
                region_codes=codes_sub,
                config=grid_config,
            )
            for k, v in eq_r_grids.items():
                kmag_eq_r_named[f"kmag_eq_r_{k}"] = v
            eq_r_closed = accumulate_kmag_eq_r_grid(
                result.l_equatorial,
                result.is_closed,
                lt_sub,
                dt_minutes=dt_trace,
                region_codes=codes_sub,
                closed_only=True,
                config=grid_config,
            )
            for k, v in eq_r_closed.items():
                kmag_eq_r_named[f"kmag_eq_r_closed_{k}"] = v
            eq_total_h = float(eq_r_grids["total"].sum()) / 60
            eq_closed_h = float(eq_r_closed["total"].sum()) / 60
            eq_r_shape = (grid_config.n_r, grid_config.n_lt)
            print(
                f"KMAG eq_r:         {eq_r_shape}, "
                f"{eq_total_h:,.0f} h mapped, {eq_closed_h:,.0f} h closed",
            )

    # In --equatorial-only mode, write a sibling zarr with just the
    # equatorial schemas (and the bare minimum of context). The
    # existing dwell grid stays untouched.
    if args.equatorial_only:
        attrs["title"] = "Cassini KMAG Equatorial-r Sibling Grid"
        attrs["description"] = (
            "Per-region dwell time on a (kmag_eq_r, LT) axis. Each "
            "spacecraft sample is binned by the equatorial apex of "
            "its KMAG-traced field line. To be consumed alongside "
            "the canonical dwell grid via xarray.merge."
        )
        ds = to_xarray(
            {},  # no 3D spatial grids
            grid_config,
            attrs=attrs,
            tracing_config=tracing_config if not args.no_trace else None,
            field_config=field_config if not args.no_trace else None,
            kmag_eq_r_grids=kmag_eq_r_named or None,
        )
        print(ds)
        print()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_zarr(ds, output_path, encoding=zarr_encoding)
        print(f"Saved (equatorial-only) to {output_path}")
        return

    # Prefix inv lat grid names for clarity in the Dataset
    inv_lat_named = {f"dipole_inv_lat_{k}": v for k, v in inv_lat_accum.items()}
    weak_field_named = {f"weak_field_{k}": v for k, v in weak_field_accum.items()}
    # Merge weak-field into dipole inv_lat grids (same dimensions)
    inv_lat_named.update(weak_field_named)

    ds = to_xarray(
        grids_accum,
        grid_config,
        attrs=attrs,
        tracing_config=tracing_config if not args.no_trace else None,
        field_config=field_config if not args.no_trace else None,
        inv_lat_grids=inv_lat_named,
        kmag_inv_lat_grids=kmag_inv_lat_named or None,
        kmag_eq_r_grids=kmag_eq_r_named or None,
    )
    print(ds)
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_zarr(ds, output_path, encoding=zarr_encoding)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
