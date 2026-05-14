"""Backfill KMAG peak metadata onto a round-8 events parquet.

For each event in ``Output/events_round8.parquet`` (or a sibling file
provided via ``--input``), this script looks up the spacecraft KSM
position at the ``peak_time`` from the canonical mission trajectory,
runs a single KMAG field-line trace, and writes three new columns:

- ``kmag_inv_lat_peak`` — signed invariant latitude (deg, hemisphere
  follows the spacecraft);
- ``l_eq_peak`` — equatorial-crossing distance (R_S);
- ``is_closed_peak`` — bool; True iff both footpoints reach Saturn's
  surface inside the trace's outer boundary.

Open-line events get NaN for both numeric columns and ``False`` for
``is_closed_peak``. The original parquet is left untouched; the
enriched table is written to ``--output`` and the side-car
``.meta.json`` carries the same attrs plus ``enriched_kmag: true``.

Usage::

    uv run python scripts/enrich_events_peak.py
    uv run python scripts/enrich_events_peak.py --workers 8 -v
    uv run python scripts/enrich_events_peak.py \\
        --input  Output/events_round8.parquet \\
        --output Output/events_round8_enriched.parquet
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
from pathlib import Path  # noqa: F401  (used in argparse type and helpers)

import numpy as np
from _common import setup_logging  # noqa: E402

import qp  # noqa: E402
from qp.cli import (  # noqa: E402
    add_field_model_args,
    add_tracing_args,
    add_verbosity_arg,
    add_workers_arg,
    add_year_range_args,
)
from qp.constants import J2000_POSIX  # noqa: E402
from qp.dwell.tracing import (  # noqa: E402
    TracingConfig,
    compute_invariant_latitudes_parallel,
)
from qp.events.persistence import events_to_parquet, read_events_parquet  # noqa: E402
from qp.fieldline.kmag_model import SaturnFieldConfig  # noqa: E402
from qp.io.trajectory import load_mission_trajectory  # noqa: E402

log = logging.getLogger(__name__)


def _peak_times_unix(df) -> np.ndarray:
    """Parse the parquet ``peak_time`` column as POSIX seconds."""
    return (
        df["peak_time"].astype("datetime64[ns]").astype("int64").to_numpy()
        / 1_000_000_000.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=qp.OUTPUT_DIR / "events_round8.parquet",
        help="Input parquet (default: canonical round-8 catalogue)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=qp.OUTPUT_DIR / "events_round8_enriched.parquet",
        help="Output parquet path",
    )
    add_workers_arg(parser, default=8)
    add_year_range_args(parser)
    add_tracing_args(parser)
    add_field_model_args(parser)
    add_verbosity_arg(parser)
    args = parser.parse_args()

    setup_logging(
        args.verbose,
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ---- read parquet + check whether already enriched ---------------
    log.info("reading events: %s", args.input)
    df, attrs = read_events_parquet(args.input)
    if {"kmag_inv_lat_peak", "l_eq_peak", "is_closed_peak"} <= set(df.columns):
        log.info(
            "input already carries KMAG peak columns; passing through "
            "without recomputing (use --force to override)",
        )
    n_events = len(df)
    log.info("  %d events", n_events)
    if n_events == 0:
        raise SystemExit("no events to enrich — empty parquet")

    # ---- load mission trajectory once --------------------------------
    t0 = time.perf_counter()
    log.info("loading trajectory %d-%d ...", args.year_from, args.year_to)
    t_unix, x, y, z, _btotal = load_mission_trajectory(args.year_from, args.year_to)
    log.info(
        "  %d samples loaded in %.1fs",
        t_unix.size,
        time.perf_counter() - t0,
    )

    # ---- map peak_time → trajectory sample index ---------------------
    peak_unix = _peak_times_unix(df)
    # Nearest-sample lookup (1-min cadence ⇒ peak_time is at-most 30 s off).
    idx = np.searchsorted(t_unix, peak_unix)
    idx = np.clip(idx, 1, t_unix.size - 1)
    left = peak_unix - t_unix[idx - 1]
    right = t_unix[idx] - peak_unix
    idx = np.where(right < left, idx, idx - 1)
    # Sanity: complain if any peak is more than 5 min outside the trajectory.
    dt = np.abs(t_unix[idx] - peak_unix)
    n_far = int((dt > 300.0).sum())
    if n_far:
        log.warning(
            "%d / %d peaks > 5 min from nearest trajectory sample — those "
            "events will get the closest available position",
            n_far,
            n_events,
        )

    x_pk = x[idx].astype(float)
    y_pk = y[idx].astype(float)
    z_pk = z[idx].astype(float)
    t_j2000 = peak_unix - J2000_POSIX

    # ---- batch trace -------------------------------------------------
    log.info(
        "tracing %d field lines (workers=%d, step=%.3f R_S, max_r=%.0f R_S) ...",
        n_events,
        args.workers,
        args.trace_step,
        args.trace_max_radius,
    )
    t1 = time.perf_counter()
    result = compute_invariant_latitudes_parallel(
        x_pk,
        y_pk,
        z_pk,
        t_j2000,
        config=TracingConfig(
            trace_every_n=1,
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
    )
    elapsed = time.perf_counter() - t1
    log.info(
        "traced in %.1fs (%d closed of %d, %.0f traces/s)",
        elapsed,
        result.n_closed,
        result.n_traces,
        n_events / elapsed,
    )

    # ---- sign inv-lat by spacecraft hemisphere -----------------------
    inv_signed = np.where(z_pk >= 0.0, result.inv_lat_north, result.inv_lat_south)
    # Open lines: NaN in both arrays, no need to mask explicitly.

    # ---- write enriched parquet --------------------------------------
    df_out = df.copy()
    df_out["kmag_inv_lat_peak"] = inv_signed.astype(float)
    df_out["l_eq_peak"] = result.l_equatorial.astype(float)
    df_out["is_closed_peak"] = result.is_closed.astype(bool)

    enriched_attrs = dict(attrs)
    enriched_attrs["enriched_kmag"] = True
    enriched_attrs["enrichment"] = {
        "tool": "scripts/enrich_events_peak.py",
        "n_events": n_events,
        "n_closed": int(result.n_closed),
        "tracing": {
            "step": args.trace_step,
            "max_radius": args.trace_max_radius,
            "dp": args.dp,
            "by_imf": args.by_imf,
            "bz_imf": args.bz_imf,
        },
        "year_from": args.year_from,
        "year_to": args.year_to,
        "source_parquet": str(args.input),
        "elapsed_seconds": elapsed,
        "completed": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    # Use events_to_parquet so the side-car JSON layout matches the
    # canonical schema.
    records = df_out.to_dict(orient="records")
    n_written = events_to_parquet(records, args.output, attrs=enriched_attrs)
    log.info(
        "Enriched %d events → %s\n"
        "  closed lines: %d / %d (%.1f%%)\n"
        "  tracing time: %.1fs (%.0f traces/s)",
        n_written,
        args.output,
        result.n_closed,
        n_events,
        100 * result.n_closed / n_events,
        elapsed,
        n_events / elapsed,
    )
    # Optional: write a tiny enrichment-summary JSON next to the parquet.
    summary_path = args.output.with_suffix(args.output.suffix + ".enrich.json")
    summary_path.write_text(json.dumps(enriched_attrs["enrichment"], indent=2))


if __name__ == "__main__":
    main()
