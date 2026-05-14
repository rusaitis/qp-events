"""Post-process the round-8 event catalogue: tag duplicates + peers.

Two annotations are written back into ``Output/events_round8.parquet``:

1. ``is_duplicate`` — cross- and intra-segment duplicates flagged by
   :func:`qp.events.dedup.tag_duplicates`. Same-band peaks within 2 h
   and within 10 % of each other in period are collapsed to the
   highest-q_factor representative; the rest are tagged.
2. Peer columns ``peer_event_ids``, ``peer_periods_min``,
   ``peer_overlap_frac`` — list-typed columns from
   :func:`qp.events.peers.tag_peers`. A peer is another non-duplicate
   detection in the same ``segment_idx`` whose window overlaps this
   one by ≥ ``--peer-min-overlap-frac`` of the shorter window
   (default 0.5). Same-band peers are recorded; band membership is
   irrelevant to the criterion. Band-label views ("QP60+QP120 event")
   are derived at analysis time via :func:`qp.events.peers.derive_co_bands`.

A CSV log of the duplicate rows lands in
``Output/diagnostics/round8_duplicates.csv`` for spot-checking against
the webapp inspection UI.

The original parquet is copied to ``events_round8.parquet.bak`` before
the overwrite — the sweep that produced it takes hours, so we keep a
safety copy.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from _common import setup_logging

from qp.events.dedup import tag_duplicates
from qp.events.peers import DEFAULT_MIN_OVERLAP_FRAC, tag_peers
from qp.events.persistence import SCHEMA_VERSION

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
        "--peer-min-overlap-frac",
        type=float,
        default=DEFAULT_MIN_OVERLAP_FRAC,
        help=(
            "Minimum overlap fraction (of the shorter window) required "
            "for two same-segment, non-duplicate detections to be peers. "
            "0.0 reproduces the legacy any-overlap rule."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a .bak copy (default: backup before overwrite).",
    )
    args = parser.parse_args(argv)

    setup_logging()

    df = pd.read_parquet(args.parquet)
    log.info("loaded %d events from %s", len(df), args.parquet)

    # 1. Tag duplicates.
    tagged = tag_duplicates(df, dt_sec=args.dt_sec, period_rel_tol=args.period_rel_tol)
    n_dup = int(tagged["is_duplicate"].sum())
    log.info(
        "tagged %d duplicates (%.1f%%) — %d unique events remain",
        n_dup,
        100.0 * n_dup / max(len(tagged), 1),
        len(tagged) - n_dup,
    )
    dup_breakdown = (
        tagged.groupby("band")["is_duplicate"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_dup", "count": "n_total"})
    )
    dup_breakdown["pct_dup"] = 100.0 * dup_breakdown["n_dup"] / dup_breakdown["n_total"]
    log.info("per-band duplicate breakdown:\n%s", dup_breakdown.to_string())

    # 2. Tag peers (sees is_duplicate to skip dups; same-band peers included).
    tagged = tag_peers(tagged, min_overlap_frac=args.peer_min_overlap_frac)
    peer_counts = tagged["peer_event_ids"].map(len)
    n_with_peer = int((peer_counts > 0).sum())
    log.info(
        "tagged %d events with ≥1 peer (%.1f%%) at min_overlap_frac=%.2f",
        n_with_peer,
        100.0 * n_with_peer / max(len(tagged), 1),
        args.peer_min_overlap_frac,
    )
    if n_with_peer:
        peer_hist = peer_counts[peer_counts > 0].value_counts().sort_index()
        log.info(
            "peer-count distribution (n_peers → n_events):\n%s", peer_hist.to_string()
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
        "rewrote %s with is_duplicate + peer_* columns",
        args.parquet,
    )

    # Update the side-car so the catalogue is self-describing.
    meta_path = args.parquet.with_suffix(args.parquet.suffix + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        attrs = dict(meta.get("attrs", {}))
        attrs["schema_version"] = SCHEMA_VERSION
        attrs["peer_min_overlap_frac"] = float(args.peer_min_overlap_frac)
        meta["attrs"] = attrs
        meta["columns"] = list(tagged.columns)
        meta["n_rows"] = len(tagged)
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        log.info(
            "updated %s with peer_min_overlap_frac=%.3f, schema=%s",
            meta_path,
            args.peer_min_overlap_frac,
            SCHEMA_VERSION,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
