#!/usr/bin/env python
"""Run the benchmark suite and append a row to the scoreboards.

Usage:
    uv run python scripts/score_pipeline.py
    uv run python scripts/score_pipeline.py --notes "baseline"
    uv run python scripts/score_pipeline.py --tier tier1
    uv run python scripts/score_pipeline.py --status discard --notes "failed attempt"
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SCOREBOARD = Path("scoreboard.tsv")
SCOREBOARD_SHAPE = Path("scoreboard-shape.tsv")
SCOREBOARD_MECHANISM = Path("scoreboard-mechanism.tsv")

DETECTION_HEADER = [
    "commit", "date", "score",
    "f1", "precision", "recall", "band_acc",
    "period_err_pct", "mean_iou", "decoy_rejection", "f1@0.5",
    "tier1_recall", "tier2_recall", "tier3_recall", "tier4_recall",
    "loc", "runtime_sec", "status", "notes",
]

SHAPE_HEADER = [
    "commit", "date", "score",
    "envelope_err", "chirp_err", "harmonic_err",
    "coherence_err", "asymmetry_err",
    "notes",
]

MECHANISM_HEADER = [
    "commit", "date", "score",
    "mode_acc", "propagation_acc", "polarization_acc",
    "transverse_ratio_err",
    "notes",
]


def _git_commit_short() -> str:
    """Current git commit hash (short)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _count_python_loc() -> int:
    """Count Python lines of code in src/ using cloc."""
    try:
        output = subprocess.check_output(
            ["cloc", "src/", "--include-lang=Python", "--quiet", "--csv"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        # CSV format: files,language,blank,comment,code
        for line in output.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 5 and parts[1] == "Python":
                return int(parts[4])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return 0


def _ensure_header(path: Path, header: list[str]) -> None:
    """Create TSV with header if it doesn't exist."""
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)


def _append_row(path: Path, row: list[str]) -> None:
    """Append a TSV row."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark and update scoreboards",
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Notes for the scoreboard row",
    )
    parser.add_argument(
        "--status", type=str, default="keep",
        choices=["keep", "discard", "crash"],
        help="Status of this run (default: keep)",
    )
    parser.add_argument(
        "--tier", type=str, default=None,
        help="Run only one tier (tier1-tier4, decoy)",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Regenerate canonical datasets (default: load from disk)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    from qp.benchmark.runner import run_benchmark, run_tier
    from qp.benchmark.scoring import composite_detection_score

    # Measure wall-clock time
    t_start = time.monotonic()

    # Run benchmark (loads canonical data from disk by default)
    if args.tier:
        suite = run_tier(args.tier, regenerate=args.regenerate)
    else:
        suite = run_benchmark(regenerate=args.regenerate)

    runtime_sec = time.monotonic() - t_start

    # Compute composite score
    score = composite_detection_score(suite)

    commit = _git_commit_short()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    loc = _count_python_loc()

    # Per-tier recalls
    t1 = suite.per_tier_recall.get("tier1", 0.0)
    t2 = suite.per_tier_recall.get("tier2", 0.0)
    t3 = suite.per_tier_recall.get("tier3", 0.0)
    t4 = suite.per_tier_recall.get("tier4", 0.0)

    # Mean period error across all matches
    all_matches = [
        m for s in suite.dataset_scores for m in s.matches
    ]
    mean_period_err = (
        sum(m.period_error_pct for m in all_matches) / len(all_matches)
        if all_matches else 0.0
    )

    # Mean IoU across datasets with matches
    ious = [s.mean_iou for s in suite.dataset_scores if s.mean_iou > 0]
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    f1_05 = suite.overall_f1_at_iou.get(0.5, 0.0)

    # Append to detection scoreboard
    _ensure_header(SCOREBOARD, DETECTION_HEADER)
    _append_row(SCOREBOARD, [
        commit, date,
        f"{score:.4f}",
        f"{suite.overall_f1:.4f}",
        f"{suite.overall_precision:.4f}",
        f"{suite.overall_recall:.4f}",
        f"{suite.band_accuracy:.4f}",
        f"{mean_period_err:.2f}",
        f"{mean_iou:.4f}",
        f"{suite.decoy_rejection_rate:.4f}",
        f"{f1_05:.4f}",
        f"{t1:.4f}", f"{t2:.4f}", f"{t3:.4f}", f"{t4:.4f}",
        str(loc),
        f"{runtime_sec:.1f}",
        args.status,
        args.notes,
    ])

    # Ensure shape and mechanism scoreboards exist (headers only for now)
    _ensure_header(SCOREBOARD_SHAPE, SHAPE_HEADER)
    _ensure_header(SCOREBOARD_MECHANISM, MECHANISM_HEADER)

    # Print summary
    print(f"\n{'=' * 60}")
    print("PIPELINE SCOREBOARD")
    print(f"{'=' * 60}")
    print(f"  Commit:           {commit}")
    print(f"  Date:             {date}")
    print(f"  COMPOSITE SCORE:  {score:.4f}")
    print()
    print(f"  F1:               {suite.overall_f1:.4f}")
    print(f"  Precision:        {suite.overall_precision:.4f}")
    print(f"  Recall:           {suite.overall_recall:.4f}")
    print(f"  Band accuracy:    {suite.band_accuracy:.4f}")
    print(f"  Period error:     {mean_period_err:.2f}%")
    print(f"  Mean IoU:         {mean_iou:.4f}")
    print(f"  Decoy rejection:  {suite.decoy_rejection_rate:.4f}")
    print(f"  F1 @ IoU=0.5:    {f1_05:.4f}")
    print()
    print(f"  Tier 1 recall:    {t1:.4f}")
    print(f"  Tier 2 recall:    {t2:.4f}")
    print(f"  Tier 3 recall:    {t3:.4f}")
    print(f"  Tier 4 recall:    {t4:.4f}")
    print()
    print(f"  Python LOC:       {loc}")
    print(f"  Runtime:          {runtime_sec:.1f}s")
    print(f"  Status:           {args.status}")
    print(f"{'=' * 60}")
    print(f"\nAppended to {SCOREBOARD}")


if __name__ == "__main__":
    main()
