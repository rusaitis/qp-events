#!/usr/bin/env python
"""Parse Cassini MAG data quality files into a single CSV.

Reads the 5 quality files from DATA/CASSINI-Data-Quality/ and writes
a unified quality_flags.csv to DATA/CASSINI-DATA/DataProducts/.

Usage:
    uv run python scripts/build_quality_flags.py [-v]
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from qp import DATA_ROOT
from qp.io.data_quality import flags_to_csv, load_all_quality_flags


def main():
    parser = argparse.ArgumentParser(description="Build quality flags CSV")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "CASSINI-DATA" / "DataProducts" / "quality_flags.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    flags = load_all_quality_flags()
    logging.info("Parsed %d quality flags (2004-2017)", len(flags))

    # Summary
    by_type = Counter(f.flag_type for f in flags)
    by_severity = Counter(f.severity for f in flags)
    by_year = Counter(f.start.year for f in flags)

    lines = [
        "=" * 55,
        "CASSINI MAG DATA QUALITY FLAGS (2004-2017)",
        "=" * 55,
        f"  Total flags: {len(flags)}",
        "",
        "  By type:",
        *(f"    {t:25s} {n:5d}" for t, n in sorted(by_type.items())),
        "",
        "  By severity:",
        *(f"    {s:25s} {n:5d}" for s, n in sorted(by_severity.items())),
        "",
        "  By year:",
        *(
            f"    {y}  {by_year.get(y, 0):5d}  {'█' * (by_year.get(y, 0) // 20)}"
            for y in range(2004, 2018)
        ),
        "",
    ]
    total_sec = sum((f.end - f.start).total_seconds() for f in flags)
    total_hours = total_sec / 3600
    mission_hours = 14 * 365.25 * 24
    lines.append(
        f"  Total flagged time: {total_hours:.1f} hours "
        f"({total_hours / mission_hours * 100:.2f}% of mission)"
    )
    lines.append("=" * 55)
    logging.info("\n%s", "\n".join(lines))

    flags_to_csv(flags, args.output)
    logging.info("Written to: %s", args.output)


if __name__ == "__main__":
    main()
