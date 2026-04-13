#!/usr/bin/env python
"""Run the QP wave detection benchmark suite.

Usage:
    uv run python scripts/run_benchmark.py              # full suite
    uv run python scripts/run_benchmark.py --tier tier1  # one tier
    uv run python scripts/run_benchmark.py --generate    # generate canonical data
    uv run python scripts/run_benchmark.py --scenario tier1_clean_qp60
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="QP event detection benchmark")
    parser.add_argument(
        "--tier", type=str, default=None,
        help="Run only scenarios from this tier (tier1-tier4, decoy)",
    )
    parser.add_argument(
        "--scenario", type=str, nargs="+", default=None,
        help="Run specific scenario(s) by name",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("Output/benchmark"),
        help="Output directory (default: Output/benchmark)",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate canonical datasets (zarr + manifest JSON)",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Regenerate even if canonical data exists",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    from qp.benchmark.runner import (
        generate_canonical_datasets,
        run_benchmark,
        run_tier,
    )

    if args.generate:
        generate_canonical_datasets(
            output_dir=args.output,
            scenario_ids=args.scenario,
        )
        return

    if args.tier:
        suite = run_tier(
            args.tier,
            data_dir=args.output,
            regenerate=args.regenerate,
        )
    elif args.scenario:
        suite = run_benchmark(
            scenario_ids=args.scenario,
            data_dir=args.output,
            regenerate=args.regenerate,
        )
    else:
        suite = run_benchmark(
            data_dir=args.output,
            regenerate=args.regenerate,
        )

    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"  Overall F1:          {suite.overall_f1:.3f}")
    print(f"  Precision:           {suite.overall_precision:.3f}")
    print(f"  Recall:              {suite.overall_recall:.3f}")
    print(f"  Band accuracy:       {suite.band_accuracy:.3f}")
    print(f"  Decoy rejection:     {suite.decoy_rejection_rate:.3f}")
    print(f"  Summary score:       {suite.summary_score:.3f}")
    print()
    for tier, recall in sorted(suite.per_tier_recall.items()):
        print(f"  {tier:20s} recall = {recall:.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
