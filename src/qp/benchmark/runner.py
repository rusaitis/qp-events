"""Benchmark runner: generate → detect → score."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np

from qp.benchmark.generator import generate_benchmark_dataset
from qp.benchmark.manifest import (
    events_to_csv,
    manifest_to_json,
)
from qp.benchmark.scenarios import ALL_SCENARIOS, TIER_SCENARIOS
from qp.benchmark.scoring import BenchmarkScore, SuiteScore, score_dataset, score_suite
from qp.events.catalog import WavePacketPeak

log = logging.getLogger(__name__)

# Base seed — each scenario gets seed = BASE_SEED + index
BASE_SEED = 20260413


def _detect_events_in_dataset(
    t: np.ndarray,
    fields: np.ndarray,
    dt: float = 60.0,
) -> list[WavePacketPeak]:
    """Run the detection pipeline on synthetic 3-component data.

    Uses the same pipeline as the real mission sweep: CWT ridge
    extraction with sigma-mask gating.
    """
    from qp.events.detector import detect_wave_packets_multi

    b_perp1 = fields[:, 1]
    b_perp2 = fields[:, 2]

    # Create datetime timeline (epoch at 2000-01-01 for synthetic)
    epoch = datetime.datetime(2000, 1, 1)
    times = [epoch + datetime.timedelta(seconds=float(s)) for s in t]

    # Detect in both transverse components, merge
    all_peaks: list[WavePacketPeak] = []
    for component in [b_perp1, b_perp2]:
        peaks = detect_wave_packets_multi(
            data=component,
            times=times,
            dt=dt,
            min_duration_hours=2.0,
            min_pixels=50,
        )
        all_peaks.extend(peaks)

    # Deduplicate: merge peaks within 2h of each other in the same band
    all_peaks.sort(key=lambda p: p.peak_time)
    merged: list[WavePacketPeak] = []
    for peak in all_peaks:
        if merged and peak.band == merged[-1].band:
            sep = abs((peak.peak_time - merged[-1].peak_time).total_seconds())
            if sep < 7200:
                continue
        merged.append(peak)

    return merged


def run_benchmark(
    scenario_ids: list[str] | None = None,
    output_dir: Path | None = None,
    save_data: bool = False,
) -> SuiteScore:
    r"""Generate datasets, run detection, and score results.

    Parameters
    ----------
    scenario_ids : list of str, optional
        Which scenarios to run. None = all.
    output_dir : Path, optional
        Where to write results. None = don't persist.
    save_data : bool
        If True, save .npz data files alongside manifests.

    Returns
    -------
    SuiteScore
        Aggregate benchmark results.
    """
    if scenario_ids is None:
        scenario_ids = list(ALL_SCENARIOS.keys())

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[BenchmarkScore] = []
    all_events = []

    epoch_sec = datetime.datetime(2000, 1, 1).timestamp()

    for idx, scenario_id in enumerate(scenario_ids):
        factory = ALL_SCENARIOS.get(scenario_id)
        if factory is None:
            log.warning("Unknown scenario: %s", scenario_id)
            continue

        scenario = factory()
        seed = BASE_SEED + idx
        log.info("Generating %s (seed=%d)...", scenario_id, seed)

        t, fields, manifest = generate_benchmark_dataset(scenario, seed)

        # Detect
        log.info("Detecting events in %s...", scenario_id)
        detections = _detect_events_in_dataset(t, fields, scenario.dt)
        log.info(
            "  %d detections vs %d ground truth (%d detectable)",
            len(detections), manifest.n_events, manifest.n_detectable,
        )

        # Score
        ds_score = score_dataset(manifest, detections, t0_sec=epoch_sec)
        all_scores.append(ds_score)
        all_events.extend(manifest.events)

        log.info(
            "  precision=%.2f recall=%.2f f1=%.2f",
            ds_score.precision, ds_score.recall, ds_score.f1,
        )

        # Persist
        if output_dir is not None:
            manifest_to_json(manifest, output_dir / f"{scenario_id}.json")
            if save_data:
                import zarr

                store = zarr.open(
                    str(output_dir / f"{scenario_id}.zarr"),
                    mode="w",
                )
                store.array("time", t, dtype="float64")
                store.array("fields", fields, dtype="float32")
                store.attrs["dataset_id"] = scenario_id
                store.attrs["seed"] = seed
                store.attrs["dt"] = scenario.dt

    suite = score_suite(all_scores)

    if output_dir is not None:
        events_to_csv(all_events, output_dir / "all_events.csv")
        # Write summary
        summary = {
            "overall_f1": suite.overall_f1,
            "overall_precision": suite.overall_precision,
            "overall_recall": suite.overall_recall,
            "band_accuracy": suite.band_accuracy,
            "decoy_rejection_rate": suite.decoy_rejection_rate,
            "summary_score": suite.summary_score,
            "per_tier_recall": suite.per_tier_recall,
        }
        import json

        (output_dir / "suite_score.json").write_text(
            json.dumps(summary, indent=2)
        )

    return suite


def run_tier(tier: str, **kwargs) -> SuiteScore:
    """Run all scenarios in a single tier."""
    ids = TIER_SCENARIOS.get(tier, [])
    if not ids:
        raise ValueError(f"Unknown tier: {tier!r}. Known: {sorted(TIER_SCENARIOS)}")
    return run_benchmark(scenario_ids=ids, **kwargs)
