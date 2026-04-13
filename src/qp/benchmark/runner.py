"""Benchmark runner: generate → detect → score.

Datasets are generated once and stored as canonical zarr files in
``Output/benchmark/``. Subsequent scoring runs load from disk,
ensuring bit-identical data across runs regardless of NumPy version.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import numpy as np

from qp import OUTPUT_DIR
from qp.benchmark.generator import generate_benchmark_dataset
from qp.benchmark.manifest import (
    DatasetManifest,
    events_to_csv,
    manifest_from_json,
    manifest_to_json,
)
from qp.benchmark.scenarios import ALL_SCENARIOS, TIER_SCENARIOS
from qp.benchmark.scoring import (
    BenchmarkScore,
    SuiteScore,
    score_dataset,
    score_suite,
)
from qp.events.catalog import WavePacketPeak

log = logging.getLogger(__name__)

# Base seed — each scenario gets seed = BASE_SEED + index
BASE_SEED = 20260413

# Default canonical dataset directory
BENCHMARK_DIR = OUTPUT_DIR / "benchmark"


def _detect_events_in_dataset(
    t: np.ndarray,
    fields: np.ndarray,
    dt: float = 60.0,
) -> list[WavePacketPeak]:
    """Run the detection pipeline on synthetic 3-component data."""
    from qp.events.detector import detect_wave_packets_multi
    from qp.events.threshold import wavelet_sigma_mask
    from qp.signal.wavelet import morlet_cwt

    b_perp1 = fields[:, 1].copy()
    b_perp2 = fields[:, 2].copy()

    # Interpolate NaN gaps so CWT doesn't produce garbage
    for arr in [b_perp1, b_perp2]:
        nans = np.isnan(arr)
        if nans.any() and not nans.all():
            good = ~nans
            arr[nans] = np.interp(
                np.flatnonzero(nans),
                np.flatnonzero(good),
                arr[good],
            )

    epoch = datetime.datetime(2000, 1, 1)
    times = [epoch + datetime.timedelta(seconds=float(s)) for s in t]

    # Shared CWT for both components — build sigma masks
    n_freqs = 300
    freq1, _, cwt1 = morlet_cwt(b_perp1, dt=dt, n_freqs=n_freqs)
    freq2, _, cwt2 = morlet_cwt(b_perp2, dt=dt, n_freqs=n_freqs)
    power1 = np.abs(cwt1)
    power2 = np.abs(cwt2)
    mask1 = wavelet_sigma_mask(power1, freq1, n_sigma=3.0)
    mask2 = wavelet_sigma_mask(power2, freq2, n_sigma=3.0)

    all_peaks: list[WavePacketPeak] = []
    for component, freq, power, mask in [
        (b_perp1, freq1, power1, mask1),
        (b_perp2, freq2, power2, mask2),
    ]:
        peaks = detect_wave_packets_multi(
            data=component,
            times=times,
            dt=dt,
            cwt_freq=freq,
            cwt_power=power,
            threshold_mask=mask,
            min_duration_hours=2.0,
            min_pixels=50,
        )
        all_peaks.extend(peaks)

    # Deduplicate: within same band, keep higher-power peak
    all_peaks.sort(key=lambda p: p.peak_time)
    merged: list[WavePacketPeak] = []
    for peak in all_peaks:
        if merged and peak.band == merged[-1].band:
            sep = abs(
                (peak.peak_time - merged[-1].peak_time).total_seconds()
            )
            if sep < 7200:
                # Keep whichever has higher prominence
                if peak.prominence > merged[-1].prominence:
                    merged[-1] = peak
                continue
        merged.append(peak)

    return merged


# ------------------------------------------------------------------
# Canonical dataset generation and loading
# ------------------------------------------------------------------


def generate_canonical_datasets(
    output_dir: Path | None = None,
    scenario_ids: list[str] | None = None,
) -> None:
    """Generate and persist canonical benchmark datasets as zarr.

    This should be run once. Subsequent scoring loads from disk.
    """
    import zarr

    if output_dir is None:
        output_dir = BENCHMARK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if scenario_ids is None:
        scenario_ids = list(ALL_SCENARIOS.keys())

    all_events = []

    for idx, scenario_id in enumerate(scenario_ids):
        factory = ALL_SCENARIOS.get(scenario_id)
        if factory is None:
            log.warning("Unknown scenario: %s", scenario_id)
            continue

        scenario = factory()
        seed = BASE_SEED + idx
        log.info("Generating %s (seed=%d)...", scenario_id, seed)

        t, fields, manifest = generate_benchmark_dataset(scenario, seed)

        # Save zarr
        store = zarr.open(str(output_dir / f"{scenario_id}.zarr"), mode="w")
        store.create_array(name="time", data=t.astype("float64"))
        store.create_array(name="fields", data=fields.astype("float32"))
        store.attrs["dataset_id"] = scenario_id
        store.attrs["seed"] = seed
        store.attrs["dt"] = scenario.dt

        # Save manifest JSON
        manifest_to_json(manifest, output_dir / f"{scenario_id}.json")
        all_events.extend(manifest.events)

    events_to_csv(all_events, output_dir / "all_events.csv")
    log.info(
        "Generated %d canonical datasets in %s",
        len(scenario_ids), output_dir,
    )


def _load_canonical_dataset(
    scenario_id: str,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, DatasetManifest]:
    """Load a pre-generated dataset from zarr + JSON manifest."""
    import zarr

    zarr_path = data_dir / f"{scenario_id}.zarr"
    json_path = data_dir / f"{scenario_id}.json"

    store = zarr.open(str(zarr_path), mode="r")
    t = np.asarray(store["time"])
    fields = np.asarray(store["fields"])
    manifest = manifest_from_json(json_path)

    return t, fields, manifest


def _has_canonical_datasets(
    data_dir: Path,
    scenario_ids: list[str],
) -> bool:
    """Check if all canonical datasets exist."""
    for sid in scenario_ids:
        if not (data_dir / f"{sid}.zarr").exists():
            return False
        if not (data_dir / f"{sid}.json").exists():
            return False
    return True


# ------------------------------------------------------------------
# Benchmark scoring
# ------------------------------------------------------------------


def run_benchmark(
    scenario_ids: list[str] | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    regenerate: bool = False,
) -> SuiteScore:
    r"""Detect and score against canonical benchmark datasets.

    By default, loads pre-generated data from ``Output/benchmark/``.
    Falls back to on-the-fly generation if canonical data is missing.

    Parameters
    ----------
    scenario_ids : list of str, optional
        Which scenarios to run. None = all.
    data_dir : Path, optional
        Where to find canonical zarr/json data. Default: Output/benchmark/.
    output_dir : Path, optional
        Where to write scoring results. None = don't persist.
    regenerate : bool
        If True, regenerate datasets even if canonical data exists.

    Returns
    -------
    SuiteScore
        Aggregate benchmark results.
    """
    if scenario_ids is None:
        scenario_ids = list(ALL_SCENARIOS.keys())

    if data_dir is None:
        data_dir = BENCHMARK_DIR

    # Generate canonical data if missing or forced
    if regenerate or not _has_canonical_datasets(data_dir, scenario_ids):
        log.info("Generating canonical datasets...")
        generate_canonical_datasets(data_dir, scenario_ids)

    use_canonical = _has_canonical_datasets(data_dir, scenario_ids)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[BenchmarkScore] = []
    epoch_sec = datetime.datetime(2000, 1, 1).timestamp()

    for idx, scenario_id in enumerate(scenario_ids):
        # Load or generate
        if use_canonical:
            log.info("Loading %s from disk...", scenario_id)
            t, fields, manifest = _load_canonical_dataset(
                scenario_id, data_dir,
            )
            dt = manifest.dt
        else:
            factory = ALL_SCENARIOS.get(scenario_id)
            if factory is None:
                log.warning("Unknown scenario: %s", scenario_id)
                continue
            scenario = factory()
            seed = BASE_SEED + idx
            log.info("Generating %s (seed=%d)...", scenario_id, seed)
            t, fields, manifest = generate_benchmark_dataset(scenario, seed)
            dt = scenario.dt

        # Detect
        log.info("Detecting events in %s...", scenario_id)
        detections = _detect_events_in_dataset(t, fields, dt)
        log.info(
            "  %d detections vs %d ground truth (%d detectable)",
            len(detections), manifest.n_events, manifest.n_detectable,
        )

        # Score
        ds_score = score_dataset(manifest, detections, t0_sec=epoch_sec)
        all_scores.append(ds_score)

        log.info(
            "  precision=%.2f recall=%.2f f1=%.2f",
            ds_score.precision, ds_score.recall, ds_score.f1,
        )

    suite = score_suite(all_scores)

    if output_dir is not None:
        summary = {
            "overall_f1": suite.overall_f1,
            "overall_precision": suite.overall_precision,
            "overall_recall": suite.overall_recall,
            "band_accuracy": suite.band_accuracy,
            "decoy_rejection_rate": suite.decoy_rejection_rate,
            "summary_score": suite.summary_score,
            "per_tier_recall": suite.per_tier_recall,
        }
        (output_dir / "suite_score.json").write_text(
            json.dumps(summary, indent=2)
        )

    return suite


def run_tier(tier: str, **kwargs) -> SuiteScore:
    """Run all scenarios in a single tier."""
    ids = TIER_SCENARIOS.get(tier, [])
    if not ids:
        raise ValueError(
            f"Unknown tier: {tier!r}. Known: {sorted(TIER_SCENARIOS)}"
        )
    return run_benchmark(scenario_ids=ids, **kwargs)
