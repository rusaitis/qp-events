"""Post-process the round-8 event catalogue: tag duplicates + co-bands.

Two annotations are written back into ``Output/events_round8.parquet``:

1. ``is_duplicate`` — cross- and intra-segment duplicates flagged by
   :func:`qp.events.dedup.tag_duplicates`. Same-band peaks within 2 h
   and within 10 % of each other in period are collapsed to the
   highest-q_factor representative; the rest are tagged.
2. ``co_bands`` — sorted comma-separated list of *other* bands that
   temporally overlap this row within the same ``segment_idx``,
   produced by :func:`qp.events.cooccurrence.tag_co_bands`. Empty for
   pure-band events. Duplicates are skipped both as siblings and as
   recipients.

A CSV log of the duplicate rows lands in
``Output/diagnostics/round8_duplicates.csv`` for spot-checking against
the webapp inspection UI.

The original parquet is copied to ``events_round8.parquet.bak`` before
the overwrite — the sweep that produced it takes hours, so we keep a
safety copy.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from qp.events.cooccurrence import tag_co_bands
from qp.events.dedup import tag_duplicates

log = logging.getLogger("tag_event_duplicates")

DEFAULT_PARQUET = Path("Output/events_round8.parquet")
DEFAULT_LOG_CSV = Path("Output/diagnostics/round8_duplicates.csv")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--log-csv", type=Path, default=DEFAULT_LOG_CSV)
    parser.add_argument("--dt-sec", type=float, default=7200.0)
    parser.add_argument("--period-rel-tol", type=float, default=0.10)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a .bak copy (default: backup before overwrite).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_parquet(args.parquet)
    log.info("loaded %d events from %s", len(df), args.parquet)

    # 1. Tag duplicates.
    tagged = tag_duplicates(
        df, dt_sec=args.dt_sec, period_rel_tol=args.period_rel_tol
    )
    n_dup = int(tagged["is_duplicate"].sum())
    log.info(
        "tagged %d duplicates (%.1f%%) — %d unique events remain",
        n_dup, 100.0 * n_dup / max(len(tagged), 1), len(tagged) - n_dup,
    )
    dup_breakdown = (
        tagged.groupby("band")["is_duplicate"].agg(["sum", "count"])
        .rename(columns={"sum": "n_dup", "count": "n_total"})
    )
    dup_breakdown["pct_dup"] = (
        100.0 * dup_breakdown["n_dup"] / dup_breakdown["n_total"]
    )
    log.info("per-band duplicate breakdown:\n%s", dup_breakdown.to_string())

    # 2. Tag co-occurring bands (sees the is_duplicate column to exclude dups).
    tagged = tag_co_bands(tagged)
    co_mask = tagged["co_bands"].fillna("").ne("")
    n_co = int(co_mask.sum())
    log.info(
        "tagged %d events with co_bands (%.1f%%)",
        n_co, 100.0 * n_co / max(len(tagged), 1),
    )
    if n_co:
        co_breakdown = (
            tagged.loc[co_mask]
            .groupby(["band", "co_bands"])
            .size()
            .rename("n")
            .reset_index()
            .sort_values("n", ascending=False)
        )
        log.info(
            "(band, co_bands) co-occurrence table:\n%s",
            co_breakdown.to_string(index=False),
        )

    # Persist the dropped-row log (duplicates only).
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    dropped = tagged.loc[tagged["is_duplicate"]].copy()
    dropped.to_csv(args.log_csv, index=False)
    log.info("wrote duplicate log → %s", args.log_csv)

    # Backup + overwrite the parquet.
    if not args.no_backup:
        backup = args.parquet.with_suffix(args.parquet.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(args.parquet, backup)
            log.info("wrote backup → %s", backup)
        else:
            log.info("backup already exists at %s — not overwriting", backup)

    tagged.to_parquet(args.parquet, engine="pyarrow", index=False)
    log.info(
        "rewrote %s with is_duplicate + co_bands columns", args.parquet,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
