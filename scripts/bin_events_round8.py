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
import logging
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

import qp  # noqa: E402
from qp.cli import add_verbosity_arg, add_workers_arg  # noqa: E402
from qp.dwell.grid import (  # noqa: E402
    DwellGridConfig,
    accumulate_inv_lat_grid_cached,
    accumulate_weak_field_grid_cached,
    accumulate_with_regions_cached,
    precompute_bins,
)
from qp.events.bands import QP_BAND_NAMES, get_band  # noqa: E402
from qp.events.binning import (  # noqa: E402
    build_band_masks,
    full_mirror_grids_to_xarray,
)
from qp.io.trajectory import load_mission_trajectory, load_region_codes  # noqa: E402

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
    edges = {
        b: (get_band(b).period_min_sec / 60.0, get_band(b).period_max_sec / 60.0)
        for b in bands
    }
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
# Per-event masking + accumulation.
# ---------------------------------------------------------------------


def _build_band_masks(
    df,
    t_unix: np.ndarray,
    bands: list[str],
    band_lookup,
) -> dict[str, np.ndarray]:
    """Thin script-side wrapper around ``qp.events.binning.build_band_masks``."""
    masks, n_unmapped = build_band_masks(
        df["date_from"].to_numpy(),
        df["date_to"].to_numpy(),
        df["period_min"].to_numpy(),
        t_unix,
        bands,
        band_lookup,
    )
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
    """Run the dwell-grid accumulators on each per-band mask.

    Computes bin indices once per trajectory via ``precompute_bins`` and
    feeds the same cache to all band×schema accumulations. Saves the
    repeated ``_compute_coords`` + ``_bin_index`` work that was the
    binning hot spot.
    """
    region_names = ("total", "magnetosphere", "magnetosheath", "solar_wind", "unknown")
    grids: dict[str, np.ndarray] = {}
    # union mask
    masks = dict(masks)
    union = np.zeros_like(next(iter(masks.values())))
    for m in masks.values():
        union |= m
    masks["total"] = union

    cache = precompute_bins(x, y, z, config)

    for band, mask in masks.items():
        if not mask.any():
            for r in region_names:
                grids[f"{band}_{r}"] = np.zeros(config.shape, dtype=np.float32)
                grids[f"{band}_dipole_inv_lat_{r}"] = np.zeros(
                    (config.n_lat, config.n_lt),
                    dtype=np.float32,
                )
                grids[f"{band}_weak_field_{r}"] = np.zeros(
                    (config.n_lat, config.n_lt),
                    dtype=np.float32,
                )
            continue
        log.info("band %s: %d minutes mapped", band, int(mask.sum()))
        r3d = accumulate_with_regions_cached(
            cache,
            region_codes,
            1.0,
            mask=mask,
            config=config,
        )
        r2d = accumulate_inv_lat_grid_cached(
            cache,
            region_codes,
            1.0,
            mask=mask,
            config=config,
        )
        rwf = accumulate_weak_field_grid_cached(
            cache,
            b_total_nT,
            1.0,
            b_threshold_nT,
            region_codes=region_codes,
            mask=mask,
            config=config,
        )
        # Accumulators already return float32 (see qp.dwell.grid
        # _accumulate_grid contract). Keep storage at float32 to halve
        # resident memory; the downstream zarr is already float32 too.
        for r in region_names:
            grids[f"{band}_{r}"] = r3d[r]
            grids[f"{band}_dipole_inv_lat_{r}"] = r2d[r]
            grids[f"{band}_weak_field_{r}"] = rwf[r]
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
        "--year-from",
        type=int,
        default=2004,
    )
    parser.add_argument(
        "--year-to",
        type=int,
        default=2017,
    )
    parser.add_argument(
        "--b-threshold",
        type=float,
        default=2.0,
        help="weak-field threshold in nT (default 2.0; matches dwell grid)",
    )
    parser.add_argument(
        "--with-kmag-trace",
        action="store_true",
        help=(
            "Populate KMAG-traced schemas: kmag_inv_lat (footpoint, "
            "for conjugate-latitude axis) AND kmag_eq_r (equatorial "
            "apex, for FLR-resonance axis). One tracing pass over "
            "the union of all event-window minutes; cost scales with "
            "total event minutes, not mission minutes."
        ),
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=1,
        help=(
            "Subsampling cadence for KMAG event tracing (default 1: "
            "full cadence). Higher values speed up tracing but bias "
            "sum(B_kmag_*) below sum(B_total) by silently dropping "
            "short events that fall between traced positions. The "
            "dwell-grid cadence is set separately in "
            "scripts/compute_dwell_grid.py."
        ),
    )
    add_workers_arg(parser, default=8, flag="--trace-workers")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help=(
            "Bin every row in the parquet. Default behaviour drops rows "
            "tagged by qp.events.dedup (cross- and intra-segment "
            "duplicates), if the column is present."
        ),
    )
    parser.add_argument(
        "--max-mva-par-frac",
        type=float,
        default=0.5,
        help=(
            "Drop events with mva_par_frac above this threshold before "
            "binning. Default 0.5 matches the detector gate (no change). "
            "Tighten (e.g. 0.2) for FLR-focused figures to suppress "
            "compressional boundary contamination."
        ),
    )
    add_verbosity_arg(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Load events
    import pandas as pd

    log.info("loading events: %s", args.events)
    df = pd.read_parquet(args.events)
    log.info("  %d events", len(df))
    if "is_duplicate" in df.columns and not args.keep_duplicates:
        n_drop = int(df["is_duplicate"].sum())
        df = df.loc[~df["is_duplicate"]].reset_index(drop=True)
        log.info("  dropped %d duplicate rows (post-hoc dedup)", n_drop)

    if args.max_mva_par_frac < 0.5:
        n_before = len(df)
        df = df.loc[df["mva_par_frac"] <= args.max_mva_par_frac].reset_index(drop=True)
        log.info(
            "  dropped %d events with mva_par_frac > %.3f (FLR-purity cut); %d remain",
            n_before - len(df),
            args.max_mva_par_frac,
            len(df),
        )

    # 2. Load trajectory + regions
    t_load = time.perf_counter()
    log.info("loading mission trajectory %d-%d ...", args.year_from, args.year_to)
    t_unix, x, y, z, btotal = load_mission_trajectory(
        args.year_from,
        args.year_to,
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
        masks,
        x,
        y,
        z,
        btotal,
        region_codes,
        config,
        args.b_threshold,
    )
    log.info("accumulated in %.1fs", time.perf_counter() - t_acc)

    # 5. Optional KMAG tracing — adds kmag_inv_lat + kmag_eq_r schemas
    #    (with closed-only variants) per band.
    kmag_grids: dict[str, np.ndarray] = {}
    kmag_stats: dict[str, int] = {}
    if args.with_kmag_trace:
        from qp.dwell.tracing import SaturnFieldConfig, TracingConfig
        from qp.events.binning import accumulate_kmag_event_grids

        log.info("KMAG tracing event-window samples ...")
        t_kmag = time.perf_counter()
        kmag_grids, kmag_stats = accumulate_kmag_event_grids(
            masks,
            x,
            y,
            z,
            t_unix,
            region_codes,
            trace_every_n=args.trace_every,
            config=config,
            tracing_config=TracingConfig(
                trace_every_n=args.trace_every,
                n_workers=args.trace_workers,
            ),
            field_config=SaturnFieldConfig(),
        )
        log.info(
            "KMAG traced in %.1fs (%d traces, %d closed)",
            time.perf_counter() - t_kmag,
            kmag_stats["n_traces"],
            kmag_stats["n_closed"],
        )

    # 6. Write zarr
    band_edges_min = {b: list(edges[b]) for b in bands}
    extra_attrs = {
        "band_scheme": args.bands,
        "band_edges_min": band_edges_min,
        "year_from": args.year_from,
        "year_to": args.year_to,
        "n_events": int(len(df)),
        "n_samples_trajectory": int(t_unix.size),
        "b_threshold_nT": args.b_threshold,
        "max_mva_par_frac": float(args.max_mva_par_frac),
        "events_parquet": str(args.events),
        "kmag_inv_lat_populated": bool(args.with_kmag_trace),
        "kmag_eq_r_populated": bool(args.with_kmag_trace),
        "time_epoch": "J2000 (POSIX - 946728000.0)",
        "coordinate_system": "KSM",
        "source": "PDS MAG 1-min KSM",
        "boundary_crossings_source": "Jackman et al. 2019",
    }
    if kmag_stats:
        extra_attrs["kmag_n_traces"] = kmag_stats["n_traces"]
        extra_attrs["kmag_n_closed"] = kmag_stats["n_closed"]
        extra_attrs["kmag_n_event_minutes"] = kmag_stats["n_events_traced_min"]
        extra_attrs["kmag_trace_every_n"] = args.trace_every

    ds = full_mirror_grids_to_xarray(
        grids,
        config,
        bands=bands,
        title="QP Event Time Grid (round-8 detector)",
        description=(
            "Per-band cumulative event time on the canonical Cassini "
            "dwell-grid axes. KMAG-traced schemas populated when "
            "--with-kmag-trace was set."
        ),
        extra_attrs=extra_attrs,
    )

    if kmag_grids:
        import xarray as xr

        from qp.events.binning import kmag_event_grids_to_xarray

        ds_kmag = kmag_event_grids_to_xarray(
            kmag_grids,
            config,
            bands=bands,
        )
        ds_kmag.attrs.clear()
        # Merge — shared local_time; kmag side adds kmag_inv_lat,
        # kmag_eq_r dims. Use override on shared coord edges (they're
        # identical by construction).
        ds = xr.merge([ds, ds_kmag], compat="override")
        ds.attrs.update(extra_attrs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        import shutil

        shutil.rmtree(args.output)
    ds.to_zarr(args.output, mode="w", consolidated=False)

    # Self-consistency.
    #
    # The grid counts sample-minutes (closed-interval mask, inclusive of
    # date_to). WaveEvent.duration_minutes counts the open interval
    # date_to - date_from, which under-counts by one sample per event
    # for 1-min data. The expected grid total is therefore
    # sum(duration_minutes) + n_events_in_window.
    total_minutes_in_grid = float(ds["total_total"].sum())
    sum_event_durations = float(df["duration_minutes"].sum())
    expected_grid_total = sum_event_durations + len(df)
    grid_minus_expected = total_minutes_in_grid - expected_grid_total
    log_fn = log.warning if abs(grid_minus_expected) > 60.0 else log.info
    log_fn(
        "self-consistency  grid=%.1f  expected=%.1f  (events_dur=%.1f + n=%d)  "
        "diff=%+.1f",
        total_minutes_in_grid,
        expected_grid_total,
        sum_event_durations,
        len(df),
        grid_minus_expected,
    )

    # If KMAG-traced schemas were populated, surface any drift between
    # the (r, lat, LT) and KMAG inv-lat totals — short events whose only
    # sample-minutes fall between the trace_every_n subsampling steps
    # are missed in the KMAG grid but counted in the 3D grid.
    if "total_kmag_inv_lat_total" in ds:
        kmag_total = float(ds["total_kmag_inv_lat_total"].sum())
        loss_frac = (
            (total_minutes_in_grid - kmag_total) / total_minutes_in_grid
            if total_minutes_in_grid > 0
            else 0.0
        )
        log.info(
            "KMAG schema coverage: total_kmag_inv_lat=%.1f  total=%.1f  "
            "missed=%.1f%% (trace_every=%d)",
            kmag_total,
            total_minutes_in_grid,
            100.0 * loss_frac,
            args.trace_every,
        )
    print(
        f"Wrote {args.output}\n"
        f"  bands: {bands}\n"
        f"  total event minutes (grid): {total_minutes_in_grid:.0f}\n"
        f"  sum(duration_minutes) + n_events: {expected_grid_total:.0f}"
    )


if __name__ == "__main__":
    main()
