"""Round-8 event-time grid: bin a parquet event catalogue onto the
full-mirror dwell-grid schema.

Reads ``Output/events_round8.parquet`` (output of
``sweep_events_round8.py``), opens the canonical Cassini KSM
trajectory (the same year-by-year PDS files that feed the dwell grid)
plus the Jackman 2019 boundary crossings, and accumulates per-band
event-time grids on every dwell-grid coord system **except** KMAG-
traced invariant latitude (which requires per-sample field-line
tracing — out of scope for this pass).

Output is a zarr file whose variable names mirror the dwell grid with
a ``<band>_`` prefix, e.g.::

    QP60_total                       (r, mag_lat, LT)        minutes
    QP60_magnetosphere               (r, mag_lat, LT)        minutes
    QP60_dipole_inv_lat_total        (dipole_inv_lat, LT)    minutes
    QP60_weak_field_total            (dipole_inv_lat, LT)    minutes
    total_total                      (band-union)            minutes
    ...

Per-band schema:
- 5 region splits × 3 schemas (3D, dipole_inv_lat, weak_field) = 15 vars
- × (3 paper bands + 1 union, or 5 fine bands + 1 union) = 60 or 90 vars

To compute the occurrence rate for a given band/region/coord::

    import xarray as xr
    ev = xr.open_zarr("Output/event_time_grid_round8.zarr")
    dw = xr.open_zarr("Output/dwell_grid_cassini_saturn.zarr")
    rate = ev["QP60_kmag_inv_lat_closed_total"] / dw["kmag_inv_lat_closed_total"]
    # ^ KMAG variables are NOT populated yet — see --with-kmag-trace below

Usage::

    uv run python scripts/bin_events_round8.py
    uv run python scripts/bin_events_round8.py --bands fine
    uv run python scripts/bin_events_round8.py \\
        --events Output/events_round8.parquet \\
        --output Output/event_time_grid_round8.zarr
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import qp  # noqa: E402
from qp.dwell.grid import (  # noqa: E402
    DwellGridConfig,
    accumulate_inv_lat_grid,
    accumulate_weak_field_grid,
    accumulate_with_regions,
)
from qp.events.bands import QP_BAND_NAMES, get_band  # noqa: E402
from qp.events.binning import full_mirror_grids_to_xarray  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Band schemes — the user can pick "paper" (3 bands) or "fine"
# (5 log-spaced bins covering 15-180 min). At gridding time we
# reclassify each event by its parabolic-interpolated period, so a
# single parquet can be re-binned into either scheme without
# rerunning the detector.
# ---------------------------------------------------------------------


def _paper_bands() -> tuple[list[str], dict[str, tuple[float, float]]]:
    bands = list(QP_BAND_NAMES)
    edges = {b: (get_band(b).period_min_sec / 60.0,
                 get_band(b).period_max_sec / 60.0) for b in bands}
    return bands, edges


def _fine_bands() -> tuple[list[str], dict[str, tuple[float, float]]]:
    """5 log-spaced bands across 15-180 min."""
    edges_min = np.geomspace(15.0, 180.0, num=6)
    bands: list[str] = []
    edges: dict[str, tuple[float, float]] = {}
    for k in range(5):
        name = f"B{k + 1}"
        bands.append(name)
        edges[name] = (float(edges_min[k]), float(edges_min[k + 1]))
    return bands, edges


def _band_lookup(edges: dict[str, tuple[float, float]]):
    """Return a callable mapping period_min → band name (or None)."""
    items = sorted(edges.items(), key=lambda kv: kv[1][0])

    def lookup(period_min: float) -> str | None:
        for name, (lo, hi) in items:
            if lo <= period_min < hi:
                return name
        return None

    return lookup


# ---------------------------------------------------------------------
# Trajectory + region loader (mirrors compute_dwell_grid.py exactly).
# ---------------------------------------------------------------------


def load_mission_trajectory(
    year_from: int = 2004,
    year_to: int = 2017,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the full Cassini KSM trajectory year-by-year.

    Returns ``(t_unix, x, y, z, btotal)``. Identical to
    :func:`scripts.compute_dwell_grid.load_year_positions` aggregated.
    """
    from compute_dwell_grid import load_year_positions

    all_t: list[np.ndarray] = []
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    all_b: list[np.ndarray] = []
    epoch = datetime.datetime(1970, 1, 1)
    for year in range(year_from, year_to + 1):
        x, y, z, btotal, times = load_year_positions(year)
        if len(x) == 0:
            continue
        t_unix = np.array(
            [(t - epoch).total_seconds() for t in times], dtype=float,
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


def load_region_codes(t_unix: np.ndarray) -> np.ndarray:
    """Assign Jackman 2019 region codes to each trajectory sample."""
    from compute_dwell_grid import lookup_region_codes
    from qp.io.crossings import crossing_lookup_arrays, parse_crossing_list

    crossings = parse_crossing_list()
    cross_unix, cross_codes = crossing_lookup_arrays(crossings)
    return lookup_region_codes(t_unix, cross_unix, cross_codes)


# ---------------------------------------------------------------------
# Per-event masking + accumulation.
# ---------------------------------------------------------------------


def _build_band_masks(
    df,
    t_unix: np.ndarray,
    bands: list[str],
    band_lookup,
) -> dict[str, np.ndarray]:
    """For each band, build a per-sample boolean mask.

    A sample is True for band B if it falls inside any event whose
    period maps to band B. The "total" band is the union across bands
    (each minute counts once regardless of how many bands fired).
    """
    n = t_unix.size
    masks: dict[str, np.ndarray] = {b: np.zeros(n, dtype=bool) for b in bands}
    epoch = np.datetime64("1970-01-01T00:00:00")
    n_unmapped = 0
    for r in df.itertuples(index=False):
        period_min = float(r.period_min)
        band = band_lookup(period_min)
        if band is None or band not in masks:
            n_unmapped += 1
            continue
        t_from = float(
            (np.datetime64(r.date_from) - epoch).astype("timedelta64[s]")
            .astype(float),
        )
        t_to = float(
            (np.datetime64(r.date_to) - epoch).astype("timedelta64[s]")
            .astype(float),
        )
        i_lo = int(np.searchsorted(t_unix, t_from, side="left"))
        i_hi = int(np.searchsorted(t_unix, t_to, side="right"))
        masks[band][i_lo:i_hi] = True
    if n_unmapped:
        log.warning("%d events fell outside the band edges", n_unmapped)
    return masks


def _accumulate_per_band(
    masks: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    b_total_nT: np.ndarray,
    region_codes: np.ndarray,
    config: DwellGridConfig,
    b_threshold_nT: float,
) -> dict[str, np.ndarray]:
    """Run the dwell-grid accumulators on each per-band mask."""
    region_names = ("total", "magnetosphere", "magnetosheath", "solar_wind", "unknown")
    grids: dict[str, np.ndarray] = {}
    # union mask
    masks = dict(masks)
    union = np.zeros_like(next(iter(masks.values())))
    for m in masks.values():
        union |= m
    masks["total"] = union

    for band, mask in masks.items():
        if not mask.any():
            for r in region_names:
                grids[f"{band}_{r}"] = np.zeros(config.shape, dtype=np.float64)
                grids[f"{band}_dipole_inv_lat_{r}"] = np.zeros(
                    (config.n_lat, config.n_lt), dtype=np.float64,
                )
                grids[f"{band}_weak_field_{r}"] = np.zeros(
                    (config.n_lat, config.n_lt), dtype=np.float64,
                )
            continue
        log.info("band %s: %d minutes mapped", band, int(mask.sum()))
        r3d = accumulate_with_regions(
            x[mask], y[mask], z[mask], region_codes[mask], 1.0, config,
        )
        r2d = accumulate_inv_lat_grid(
            x[mask], y[mask], z[mask], 1.0, region_codes[mask], config,
        )
        rwf = accumulate_weak_field_grid(
            x[mask], y[mask], z[mask], b_total_nT[mask],
            1.0, b_threshold_nT, region_codes[mask], config,
        )
        for r in region_names:
            grids[f"{band}_{r}"] = r3d[r].astype(np.float64)
            grids[f"{band}_dipole_inv_lat_{r}"] = r2d[r].astype(np.float64)
            grids[f"{band}_weak_field_{r}"] = rwf[r].astype(np.float64)
    return grids


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=qp.OUTPUT_DIR / "events_round8.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=qp.OUTPUT_DIR / "event_time_grid_round8.zarr",
    )
    parser.add_argument(
        "--bands",
        choices=("paper", "fine"),
        default="paper",
        help="paper = 3 QP bands; fine = 5 log-spaced 15-180 min bins",
    )
    parser.add_argument(
        "--year-from", type=int, default=2004,
    )
    parser.add_argument(
        "--year-to", type=int, default=2017,
    )
    parser.add_argument(
        "--b-threshold", type=float, default=2.0,
        help="weak-field threshold in nT (default 2.0; matches dwell grid)",
    )
    parser.add_argument(
        "--with-kmag-trace",
        action="store_true",
        help=(
            "Populate KMAG-traced inv-lat schemas. NOT IMPLEMENTED in "
            "this pass — requires per-sample tracing parity with "
            "compute_dwell_grid.py. Will raise NotImplementedError."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.with_kmag_trace:
        raise NotImplementedError(
            "KMAG-traced inv-lat schemas require a sample-level tracing "
            "cache. Track that work as a follow-up; the slim mirror "
            "(3D + dipole_inv_lat + weak_field) covers Figs 7 and the "
            "dipole-latitude denominators."
        )

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Load events
    import pandas as pd
    log.info("loading events: %s", args.events)
    df = pd.read_parquet(args.events)
    log.info("  %d events", len(df))

    # 2. Load trajectory + regions
    t_load = time.perf_counter()
    log.info("loading mission trajectory %d-%d ...",
             args.year_from, args.year_to)
    t_unix, x, y, z, btotal = load_mission_trajectory(
        args.year_from, args.year_to,
    )
    log.info("  %d samples", t_unix.size)
    region_codes = load_region_codes(t_unix)
    log.info("trajectory loaded in %.1fs", time.perf_counter() - t_load)

    # 3. Build band masks
    if args.bands == "paper":
        bands, edges = _paper_bands()
    else:
        bands, edges = _fine_bands()
    log.info("band scheme: %s = %s", args.bands, edges)
    band_lookup = _band_lookup(edges)
    masks = _build_band_masks(df, t_unix, bands, band_lookup)

    # 4. Accumulate
    config = DwellGridConfig()
    log.info("accumulating per-band grids ...")
    t_acc = time.perf_counter()
    grids = _accumulate_per_band(
        masks, x, y, z, btotal, region_codes, config,
        args.b_threshold,
    )
    log.info("accumulated in %.1fs", time.perf_counter() - t_acc)

    # 5. Write zarr
    band_edges_min = {b: list(edges[b]) for b in bands}
    extra_attrs = {
        "band_scheme": args.bands,
        "band_edges_min": band_edges_min,
        "year_from": args.year_from,
        "year_to": args.year_to,
        "n_events": int(len(df)),
        "n_samples_trajectory": int(t_unix.size),
        "b_threshold_nT": args.b_threshold,
        "events_parquet": str(args.events),
        "kmag_inv_lat_populated": False,
        "time_epoch": "J2000 (POSIX - 946728000.0)",
        "coordinate_system": "KSM",
        "source": "PDS MAG 1-min KSM",
        "boundary_crossings_source": "Jackman et al. 2019",
    }
    ds = full_mirror_grids_to_xarray(
        grids,
        config,
        bands=bands,
        title="QP Event Time Grid (round-8 detector)",
        description=(
            "Per-band cumulative event time on the canonical Cassini "
            "dwell-grid axes. KMAG-traced invariant-latitude schemas "
            "are not populated in this pass."
        ),
        extra_attrs=extra_attrs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        import shutil
        shutil.rmtree(args.output)
    ds.to_zarr(args.output, mode="w", consolidated=False)

    # Self-consistency: total mins in the union "total" 3D == sum of
    # event durations (within the trajectory window).
    total_minutes_in_grid = float(ds["total_total"].sum())
    total_minutes_in_events = float(df["duration_minutes"].sum())
    log.info(
        "total event minutes: in_grid=%.1f  in_events=%.1f  diff=%.1f",
        total_minutes_in_grid,
        total_minutes_in_events,
        abs(total_minutes_in_grid - total_minutes_in_events),
    )
    print(
        f"Wrote {args.output}\n"
        f"  bands: {bands}\n"
        f"  total event minutes (grid): {total_minutes_in_grid:.0f}\n"
        f"  total event minutes (events parquet): {total_minutes_in_events:.0f}"
    )


if __name__ == "__main__":
    main()
