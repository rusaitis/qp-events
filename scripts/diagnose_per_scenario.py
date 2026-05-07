"""D0: per-scenario TP/FP/FN breakdown of the round-6 detector.

Re-uses the same code path as ``run_benchmark`` but exposes per-scenario
:class:`BenchmarkScore` rows so we can identify which decoys leak and
which tier-2 events miss before designing round-7 changes.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

# Import private helpers from the runner — this is a diagnostic, not a
# public API consumer.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qp.benchmark.runner import (  # noqa: E402
    ALL_SCENARIOS,
    BENCHMARK_DIR,
    _detect_events_in_dataset,
    _has_canonical_datasets,
    _load_canonical_dataset,
    score_dataset,
)


def main() -> None:
    scenario_ids = list(ALL_SCENARIOS.keys())
    if not _has_canonical_datasets(BENCHMARK_DIR, scenario_ids):
        print(f"FATAL: canonical datasets missing in {BENCHMARK_DIR}", file=sys.stderr)
        print("Run `uv run python -m qp.benchmark.runner --suite canonical` first.")
        sys.exit(1)

    epoch_sec = datetime.datetime(2000, 1, 1).timestamp()

    print(
        f"{'scenario':36s}  {'TP':>3s} {'FP':>3s} {'FN':>3s} "
        f"{'Pdet':>5s} {'Pdcy':>5s}  {'F1':>5s}  {'IoU':>5s}  notes"
    )
    print("-" * 100)

    tier_totals: dict[str, dict[str, int]] = {}

    for scenario_id in scenario_ids:
        try:
            t, fields, manifest = _load_canonical_dataset(scenario_id, BENCHMARK_DIR)
        except Exception as exc:  # noqa: BLE001
            print(f"{scenario_id:36s}  load-failed: {exc}")
            continue

        detections = _detect_events_in_dataset(t, fields, manifest.dt)
        score = score_dataset(manifest, detections, t0_sec=epoch_sec)

        tier = scenario_id.split("_", 1)[0]
        td = tier_totals.setdefault(
            tier, {"tp": 0, "fp": 0, "fn": 0, "n_detectable": 0, "n_decoy": 0, "n_decoy_det": 0}
        )
        td["tp"] += score.n_true_positives
        td["fp"] += score.n_false_positives
        td["fn"] += score.n_false_negatives
        td["n_detectable"] += score.n_detectable
        td["n_decoy"] += score.n_decoy_events
        td["n_decoy_det"] += score.n_decoy_detected

        notes = []
        if tier == "decoy":
            if score.n_decoy_detected > 0:
                notes.append(f"LEAKS ({score.n_decoy_detected} false-positive)")
            else:
                notes.append("rejected ✓")
        else:
            if score.n_false_negatives > 0:
                notes.append(f"MISSES {score.n_false_negatives}")
            if score.n_false_positives > 0:
                notes.append(f"+{score.n_false_positives} extra FP")
            if not notes:
                notes.append("clean ✓")

        mean_iou = (
            sum(m.iou for m in score.matches) / len(score.matches)
            if score.matches
            else 0.0
        )

        print(
            f"{scenario_id:36s}  "
            f"{score.n_true_positives:3d} {score.n_false_positives:3d} {score.n_false_negatives:3d}  "
            f"{score.n_detectable:>4d}  {score.n_decoy_events:>4d}  "
            f"{score.f1:>5.3f}  {mean_iou:>5.3f}  {'; '.join(notes)}"
        )

    print("-" * 100)
    print("\nPer-tier rollup:")
    for tier in ("tier1", "tier2", "tier3", "tier4", "decoy"):
        td = tier_totals.get(tier)
        if td is None:
            continue
        if tier == "decoy":
            print(
                f"  {tier:6s}: rejected {td['n_decoy'] - td['n_decoy_det']}/{td['n_decoy']} "
                f"(rate {1.0 - td['n_decoy_det']/td['n_decoy']:.3f})"
            )
        else:
            recall = td["tp"] / td["n_detectable"] if td["n_detectable"] else 0.0
            print(
                f"  {tier:6s}: TP={td['tp']}  FP={td['fp']}  FN={td['fn']}  "
                f"recall={recall:.3f}"
            )


if __name__ == "__main__":
    main()
