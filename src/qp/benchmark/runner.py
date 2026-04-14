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
    from qp.events.detector import detect_wave_packets_multi, filter_detections
    from qp.events.threshold import wavelet_sigma_mask
    from qp.signal.wavelet import morlet_cwt

    b_par = fields[:, 0].copy()
    b_perp1 = fields[:, 1].copy()
    b_perp2 = fields[:, 2].copy()

    # Interpolate NaN gaps so CWT doesn't produce garbage
    for arr in [b_par, b_perp1, b_perp2]:
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

    # CWT both transverse components; restrict to QP-relevant frequencies
    n_freqs = 300
    freq_max = 1.0e-3  # period ≥ ~17 min; covers all QP bands
    freq, _, cwt1 = morlet_cwt(
        b_perp1, dt=dt, n_freqs=n_freqs, freq_max=freq_max,
    )
    _, _, cwt2 = morlet_cwt(
        b_perp2, dt=dt, n_freqs=n_freqs, freq_max=freq_max,
    )
    joint_power = (np.abs(cwt1) + np.abs(cwt2)) / 2.0
    # σ-mask on total transverse power: elevated power in the
    # combined perpendicular field indicates wave activity.
    combined_mask = wavelet_sigma_mask(joint_power, freq, n_sigma=4.5)

    # CWT of parallel component for in-band transverse ratio checks
    _, _, cwt_par = morlet_cwt(
        b_par, dt=dt, n_freqs=n_freqs, freq_max=freq_max,
    )
    power_par = np.abs(cwt_par)

    all_peaks = detect_wave_packets_multi(
        data=b_perp1,
        times=times,
        dt=dt,
        cwt_freq=freq,
        cwt_power=joint_power,
        threshold_mask=combined_mask,
        min_duration_hours=2.0,
        min_pixels=80,
    )

    # Physical post-filters: min oscillations, transverse ratio,
    # spectral concentration, and same-band deduplication.
    return filter_detections(
        all_peaks, t, freq, joint_power, power_par, epoch=epoch,
        spectral_concentration=None,
        min_coherence=0.9,
        cwt_perp1_complex=cwt1,
        cwt_perp2_complex=cwt2,
    )


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


# ------------------------------------------------------------------
# Multi-seed sweep — paper-grade reporting
# ------------------------------------------------------------------


from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeedSweepResult:
    """Aggregate of a multi-seed run.

    The point estimate ``mean`` is the same one a single-seed run
    would report; ``std`` is the standard deviation across seeds.
    Without ``std``, a single number is not paper-defensible —
    referees will (rightly) ask whether the headline depends on a
    lucky seed. Aim for ``std/mean`` < 5 % on the composite.
    """

    composite_mean: float
    composite_std: float
    composite_per_seed: list[float]
    per_metric_mean: dict[str, float]
    per_metric_std: dict[str, float]
    suites: list[SuiteScore]
    seeds: list[int]


def run_benchmark_multi_seed(
    scenario_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    n_seeds: int = 10,
    base_seed_offset: int = 0,
    **kwargs,
) -> SeedSweepResult:
    """Run the suite at multiple seeds and report mean ± std.

    Each seed regenerates the canonical datasets to a fresh per-seed
    output directory (``data_dir / f"seed_{seed:04d}"``) so seeds
    are fully independent. To save disk, leave ``data_dir`` unset
    and the runner will write under the default ``Output/benchmark/``
    namespace; pass an explicit ``data_dir`` per-seed if you want
    parallel runs or want to keep historical seeds.

    Parameters
    ----------
    scenario_ids : list of str, optional
        Which scenarios to run. ``None`` = all.
    seeds : list of int, optional
        Explicit per-seed offsets added to ``BASE_SEED``. Overrides
        ``n_seeds`` and ``base_seed_offset`` when given.
    n_seeds : int
        Number of seeds to run when ``seeds`` is None. Default 10
        (one decade of independent realisations — matches GW /
        biosignal benchmark conventions).
    base_seed_offset : int
        Starting offset added to ``BASE_SEED`` for the first seed
        when ``seeds`` is None. Sweeps run as
        ``BASE_SEED + base_seed_offset + i``.
    **kwargs
        Passed through to :func:`run_benchmark`. Note: the per-seed
        runner forces ``regenerate=True`` so each seed gets fresh
        data; pass ``data_dir`` to control where it lives.
    """
    from qp.benchmark.scoring import composite_detection_score

    if seeds is None:
        seeds = list(range(base_seed_offset, base_seed_offset + n_seeds))

    base_data_dir = kwargs.pop("data_dir", None)

    suites: list[SuiteScore] = []
    composites: list[float] = []
    metric_lists: dict[str, list[float]] = {
        "f1": [],
        "band_macro": [],
        "decoy": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }

    # Temporarily override BASE_SEED so that scenario-index seeding
    # picks up our seed offset. The simpler path is to dispatch with
    # a per-seed data_dir — generate_canonical_datasets uses
    # ``BASE_SEED + idx``, so to vary across seeds we monkey-patch
    # the module global for the duration of each call.
    import qp.benchmark.runner as _runner

    original_base = _runner.BASE_SEED
    try:
        for s in seeds:
            _runner.BASE_SEED = original_base + s
            seed_data_dir = (
                (base_data_dir / f"seed_{s:04d}")
                if base_data_dir is not None else None
            )
            suite = run_benchmark(
                scenario_ids=scenario_ids,
                data_dir=seed_data_dir,
                regenerate=True,
                **kwargs,
            )
            suites.append(suite)
            composites.append(composite_detection_score(suite))
            metric_lists["f1"].append(suite.overall_f1)
            metric_lists["band_macro"].append(suite.band_accuracy_macro)
            metric_lists["decoy"].append(suite.decoy_rejection_rate)
            metric_lists["precision"].append(suite.overall_precision)
            metric_lists["recall"].append(suite.overall_recall)
            mean_iou_seed = float(np.mean([
                s.mean_iou for s in suite.dataset_scores if s.mean_iou > 0
            ])) if any(
                s.mean_iou > 0 for s in suite.dataset_scores
            ) else 0.0
            metric_lists["iou"].append(mean_iou_seed)
    finally:
        _runner.BASE_SEED = original_base

    arr = np.asarray(composites)
    return SeedSweepResult(
        composite_mean=float(arr.mean()),
        composite_std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        composite_per_seed=composites,
        per_metric_mean={
            k: float(np.mean(v)) for k, v in metric_lists.items()
        },
        per_metric_std={
            k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            for k, v in metric_lists.items()
        },
        suites=suites,
        seeds=seeds,
    )
