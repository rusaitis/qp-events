"""Build per-event sparse spatial footprints from the round-8 catalogue.

For each row in ``Output/events_round8.parquet`` (or a sibling
enriched parquet given via ``--events``), this script walks the
Cassini KSM trajectory inside the event's ``[date_from, date_to]``
window and records the bins visited on three canonical grids:

- ``g3d``           ``(r, mag_lat, LT)``       — 1-min cadence
- ``g_kmag_inv_lat`` ``(kmag_inv_lat, LT)``    — closed lines, ``trace-every-n``-min cadence
- ``g_l_eq``        ``(kmag_eq_r, LT)``        — closed lines, ``trace-every-n``-min cadence

Each grid shares its bin edges with the canonical dwell denominator
(``Output/dwell_grid_cassini_saturn.zarr`` and the sibling
``Output/dwell_grid_kmag_eq_r.zarr``), so an event-time / dwell-time
ratio is well-defined per cell.

Output: ``Output/event_footprints_round8.zarr`` in the CSR layout
defined by :mod:`qp.events.footprints`. Total size for the canonical
2004-event catalogue is a few MB.

Usage::

    uv run python scripts/build_event_footprints.py -v
    uv run python scripts/build_event_footprints.py --workers 8 \\
        --events Output/events_round8_enriched.parquet \\
        --output Output/event_footprints_round8.zarr
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

import qp  # noqa: E402
from qp.cli import (  # noqa: E402
    add_field_model_args,
    add_tracing_args,
    add_verbosity_arg,
    add_workers_arg,
    add_year_range_args,
)
from qp.constants import J2000_POSIX  # noqa: E402
from qp.coords.ksm import local_time, magnetic_latitude  # noqa: E402
from qp.dwell.grid import DwellGridConfig, _bin_index  # noqa: E402
from qp.dwell.tracing import (  # noqa: E402
    TracingConfig,
    compute_invariant_latitudes_parallel,
)
from qp.events.footprints import (  # noqa: E402
    EventFootprints,
    build_sparse_grid,
    grid_shape,
    write_zarr,
)
from qp.fieldline.kmag_model import SaturnFieldConfig  # noqa: E402
from qp.io.trajectory import load_mission_trajectory, load_region_codes  # noqa: E402

log = logging.getLogger(__name__)


def _peak_window_unix(
    df,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse parquet ``date_from`` / ``date_to`` as POSIX seconds."""
    t_from = (
        df["date_from"].astype("datetime64[ns]").astype("int64").to_numpy()
        / 1_000_000_000.0
    )
    t_to = (
        df["date_to"].astype("datetime64[ns]").astype("int64").to_numpy()
        / 1_000_000_000.0
    )
    return t_from, t_to


def _aggregate_per_event(
    sample_lo: np.ndarray,
    sample_hi: np.ndarray,
    flat_idx: np.ndarray,
    valid: np.ndarray,
    weight_per_sample: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """For each event, aggregate the in-window samples to unique-bin counts.

    Returns lists ``(bins_per_event, weights_per_event)``. Each entry is
    1-D, sorted by bin index, ready to be packed by
    :func:`build_sparse_grid`.
    """
    n_events = len(sample_lo)
    bins_out: list[np.ndarray] = [np.empty(0, dtype=np.int32)] * n_events
    weights_out: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * n_events
    for i in range(n_events):
        lo, hi = int(sample_lo[i]), int(sample_hi[i])
        if hi <= lo:
            continue
        sel = valid[lo:hi]
        if not sel.any():
            continue
        idx = flat_idx[lo:hi][sel]
        unique, counts = np.unique(idx, return_counts=True)
        bins_out[i] = unique.astype(np.int32, copy=False)
        weights_out[i] = counts.astype(np.float32) * weight_per_sample
    return bins_out, weights_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=qp.OUTPUT_DIR / "events_round8.parquet",
        help="Input events parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=qp.OUTPUT_DIR / "event_footprints_round8.zarr",
        help="Output zarr path",
    )
    add_year_range_args(parser)
    parser.add_argument(
        "--trace-every-n",
        type=int,
        default=10,
        help="Sub-sample cadence for KMAG tracing (minutes). Matches "
        "the canonical dwell grid (10).",
    )
    add_workers_arg(parser, default=8)
    add_tracing_args(parser)
    add_field_model_args(parser)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="If the events parquet has an ``is_duplicate`` column, "
        "default behaviour drops those rows (matches bin_events_round8).",
    )
    add_verbosity_arg(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ---- load events --------------------------------------------------
    import pandas as pd

    df = pd.read_parquet(args.events)
    if "is_duplicate" in df.columns and not args.keep_duplicates:
        n_drop = int(df["is_duplicate"].sum())
        df = df.loc[~df["is_duplicate"]].reset_index(drop=True)
        log.info("dropped %d duplicate rows (post-hoc dedup)", n_drop)
    n_events = len(df)
    if n_events == 0:
        raise SystemExit("no events to bin — empty parquet")
    log.info("events: %d", n_events)

    # ---- trajectory + region codes ------------------------------------
    t0 = time.perf_counter()
    log.info("loading trajectory %d-%d ...", args.year_from, args.year_to)
    t_unix, x, y, z, _btotal = load_mission_trajectory(
        args.year_from,
        args.year_to,
    )
    region_codes = load_region_codes(t_unix)
    log.info(
        "  %d samples loaded in %.1fs",
        t_unix.size,
        time.perf_counter() - t0,
    )

    # ---- per-event sample windows -------------------------------------
    t_from, t_to = _peak_window_unix(df)
    sample_lo = np.searchsorted(t_unix, t_from, side="left").astype(np.int64)
    sample_hi = np.searchsorted(t_unix, t_to, side="right").astype(np.int64)

    # ---- 3D bin indices for every trajectory sample -------------------
    config = DwellGridConfig()
    mag_lat = magnetic_latitude(x, y, z)
    lt = local_time(x, y)
    r_off = np.sqrt(x * x + y * y + (z - 0.037) ** 2)

    i_r = _bin_index(r_off, *config.r_range, config.n_r)
    i_lat = _bin_index(mag_lat, *config.lat_range, config.n_lat)
    i_lt = _bin_index(lt, *config.lt_range, config.n_lt)
    in_range_3d = (
        (r_off >= config.r_range[0])
        & (r_off < config.r_range[1])
        & (mag_lat >= config.lat_range[0])
        & (mag_lat < config.lat_range[1])
        & (lt >= config.lt_range[0])
        & (lt < config.lt_range[1])
    )
    flat_3d = np.ravel_multi_index((i_r, i_lat, i_lt), config.shape).astype(np.int32)

    # ---- KMAG tracing on union of event-window samples ----------------
    # Build union mask over trajectory (event windows only) — same idea
    # as accumulate_kmag_event_grids.
    union = np.zeros(t_unix.size, dtype=bool)
    for lo, hi in zip(sample_lo, sample_hi):
        union[int(lo) : int(hi)] = True
    n_event_min = int(union.sum())
    keep = np.flatnonzero(union)
    log.info(
        "KMAG tracing: %d event-minutes, ~%d traces at every-%d cadence",
        n_event_min,
        max(1, n_event_min // args.trace_every_n),
        args.trace_every_n,
    )

    if n_event_min == 0:
        raise SystemExit("union event window is empty — nothing to trace")

    t1 = time.perf_counter()
    result = compute_invariant_latitudes_parallel(
        x[keep],
        y[keep],
        z[keep],
        t_unix[keep] - J2000_POSIX,
        config=TracingConfig(
            trace_every_n=args.trace_every_n,
            step=args.trace_step,
            max_radius=args.trace_max_radius,
            region_filter=None,
            n_workers=args.workers,
        ),
        field_config=SaturnFieldConfig(
            dp=args.dp,
            by_imf=args.by_imf,
            bz_imf=args.bz_imf,
        ),
        region_codes=region_codes[keep],
    )
    log.info(
        "traced %d field lines (%d closed) in %.1fs",
        result.n_traces,
        result.n_closed,
        time.perf_counter() - t1,
    )

    # ---- KMAG bin indices ---------------------------------------------
    # Indices into the original trajectory of each *traced* sample.
    sub_idx = np.arange(0, keep.size, args.trace_every_n)
    orig_idx_kmag = keep[sub_idx]
    z_sub = z[orig_idx_kmag]
    lt_sub = lt[orig_idx_kmag]

    inv_n = result.inv_lat_north
    inv_s = result.inv_lat_south
    l_eq = result.l_equatorial
    closed = result.is_closed

    conjugate_lat = np.where(z_sub >= 0.0, inv_n, inv_s)
    i_inv = _bin_index(conjugate_lat, *config.lat_range, config.n_lat)
    i_lt_kmag = _bin_index(lt_sub, *config.lt_range, config.n_lt)
    in_range_inv = (
        np.isfinite(conjugate_lat)
        & closed
        & (conjugate_lat >= config.lat_range[0])
        & (conjugate_lat < config.lat_range[1])
        & (lt_sub >= config.lt_range[0])
        & (lt_sub < config.lt_range[1])
    )
    flat_inv = np.ravel_multi_index(
        (i_inv, i_lt_kmag),
        (config.n_lat, config.n_lt),
    ).astype(np.int32)

    i_leq = _bin_index(l_eq, *config.r_range, config.n_r)
    in_range_leq = (
        np.isfinite(l_eq)
        & closed
        & (l_eq >= config.r_range[0])
        & (l_eq < config.r_range[1])
        & (lt_sub >= config.lt_range[0])
        & (lt_sub < config.lt_range[1])
    )
    flat_leq = np.ravel_multi_index(
        (i_leq, i_lt_kmag),
        (config.n_r, config.n_lt),
    ).astype(np.int32)

    # ---- per-event aggregation ---------------------------------------
    # 3D — direct on the full trajectory
    log.info("aggregating per-event 3D bins ...")
    t2 = time.perf_counter()
    bins_3d, weights_3d = _aggregate_per_event(
        sample_lo,
        sample_hi,
        flat_3d,
        in_range_3d,
        weight_per_sample=1.0,
    )

    # KMAG — index events into the traced-sample axis via searchsorted
    # on orig_idx_kmag (sorted by construction).
    sample_lo_kmag = np.searchsorted(
        orig_idx_kmag,
        sample_lo,
        side="left",
    ).astype(np.int64)
    sample_hi_kmag = np.searchsorted(
        orig_idx_kmag,
        sample_hi,
        side="left",
    ).astype(np.int64)
    dt_trace = float(args.trace_every_n)
    log.info("aggregating per-event KMAG bins ...")
    bins_inv, weights_inv = _aggregate_per_event(
        sample_lo_kmag,
        sample_hi_kmag,
        flat_inv,
        in_range_inv,
        weight_per_sample=dt_trace,
    )
    bins_leq, weights_leq = _aggregate_per_event(
        sample_lo_kmag,
        sample_hi_kmag,
        flat_leq,
        in_range_leq,
        weight_per_sample=dt_trace,
    )
    log.info("aggregated in %.1fs", time.perf_counter() - t2)

    # ---- pack into CSR -----------------------------------------------
    fp = EventFootprints(
        event_ids=df["event_id"].to_numpy(dtype=np.int64),
        grids={
            "g3d": build_sparse_grid(
                bins_3d,
                weights_3d,
                grid_shape("g3d", config),
            ),
            "g_kmag_inv_lat": build_sparse_grid(
                bins_inv,
                weights_inv,
                grid_shape("g_kmag_inv_lat", config),
            ),
            "g_l_eq": build_sparse_grid(
                bins_leq,
                weights_leq,
                grid_shape("g_l_eq", config),
            ),
        },
        config=config,
    )

    # ---- self-consistency report -------------------------------------
    total_3d_min = float(fp.total("g3d").sum())
    total_inv_min = float(fp.total("g_kmag_inv_lat").sum())
    total_leq_min = float(fp.total("g_l_eq").sum())
    total_event_min = float(df["duration_minutes"].sum())
    log.info(
        "footprint sums (min):  3d=%.0f  kmag_inv_lat=%.0f  l_eq=%.0f",
        total_3d_min,
        total_inv_min,
        total_leq_min,
    )
    log.info(
        "  vs duration_minutes sum: %.0f (3d/dur=%.3f)",
        total_event_min,
        total_3d_min / max(total_event_min, 1.0),
    )

    # ---- write zarr ---------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        import shutil

        shutil.rmtree(args.output)
    write_zarr(
        fp,
        str(args.output),
        extra_attrs={
            "source_parquet": str(args.events),
            "year_from": args.year_from,
            "year_to": args.year_to,
            "trace_every_n": args.trace_every_n,
            "tracing": {
                "step": args.trace_step,
                "max_radius": args.trace_max_radius,
                "dp": args.dp,
                "by_imf": args.by_imf,
                "bz_imf": args.bz_imf,
            },
            "n_traces": int(result.n_traces),
            "n_closed": int(result.n_closed),
            "n_event_min": n_event_min,
            "totals_min": {
                "g3d": total_3d_min,
                "g_kmag_inv_lat": total_inv_min,
                "g_l_eq": total_leq_min,
            },
        },
    )
    print(
        f"Wrote {args.output}\n"
        f"  events:           {n_events}\n"
        f"  total event-min:  {total_event_min:.0f} (parquet duration sum)\n"
        f"  total in 3D grid: {total_3d_min:.0f}\n"
        f"  total KMAG inv:   {total_inv_min:.0f}\n"
        f"  total L_eq:       {total_leq_min:.0f}\n"
        f"  closed/traces:    {result.n_closed} / {result.n_traces}"
    )


if __name__ == "__main__":
    main()
