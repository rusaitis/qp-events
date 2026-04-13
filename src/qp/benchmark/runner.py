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

    # CWT both transverse components, AND their sigma masks (coincidence)
    # Restrict freq_max to QP-relevant range for better low-freq resolution
    n_freqs = 300
    freq_max = 1.0e-3  # period > ~17 min; covers all QP bands
    freq, _, cwt1 = morlet_cwt(
        b_perp1, dt=dt, n_freqs=n_freqs, freq_max=freq_max,
    )
    _, _, cwt2 = morlet_cwt(
        b_perp2, dt=dt, n_freqs=n_freqs, freq_max=freq_max,
    )
    power1 = np.abs(cwt1)
    power2 = np.abs(cwt2)
    mask1 = wavelet_sigma_mask(power1, freq, n_sigma=3.0)
    mask2 = wavelet_sigma_mask(power2, freq, n_sigma=3.0)
    joint_mask = mask1 & mask2
    joint_power = (power1 + power2) / 2.0

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
        threshold_mask=joint_mask,
        min_duration_hours=2.0,
        min_pixels=80,
    )

    # Fallback: if AND mask found nothing, try single-component detection
    # with strict sigma. Catches linearly polarized events.
    if not all_peaks:
        for power, mask in [(power1, mask1), (power2, mask2)]:
            strict_mask = wavelet_sigma_mask(power, freq, n_sigma=4.0)
            single_peaks = detect_wave_packets_multi(
                data=b_perp1,
                times=times,
                dt=dt,
                cwt_freq=freq,
                cwt_power=power,
                threshold_mask=strict_mask,
                min_duration_hours=2.0,
                min_pixels=100,
            )
            all_peaks.extend(single_peaks)

    # Split long detections at power minima. The CWT at long periods
    # (QP120) merges separate wave packets into one giant ridge.
    from scipy.signal import find_peaks as _find_peaks

    split_peaks: list[WavePacketPeak] = []
    for peak in all_peaks:
        dur_h = (peak.date_to - peak.date_from).total_seconds() / 3600
        period_h = (peak.period_sec or 3600) / 3600
        # Only split if duration >> expected wave packet length
        if dur_h > max(8.0, 6 * period_h):
            # Time-marginal power in the detection's band rows
            periods = 1.0 / freq
            band_rows = np.zeros(len(freq), dtype=bool)
            if peak.band:
                from qp.events.bands import get_band
                b = get_band(peak.band)
                band_rows = (
                    (periods >= b.period_min_sec)
                    & (periods < b.period_max_sec)
                )
            if not band_rows.any():
                split_peaks.append(peak)
                continue
            i_from = max(
                0,
                int(
                    (peak.date_from - epoch).total_seconds() / dt
                ),
            )
            i_to = min(
                joint_power.shape[1],
                int(
                    (peak.date_to - epoch).total_seconds() / dt
                ),
            )
            marginal = joint_power[band_rows, i_from:i_to].mean(axis=0)
            # Find peaks in the marginal to identify wave packets
            min_dist = max(int(period_h * 3600 / dt * 2), 30)
            pk_idx, pk_props = _find_peaks(
                marginal, distance=min_dist, prominence=0.1 * marginal.max(),
            )
            if len(pk_idx) > 1:
                # Build sub-peaks from the found peaks
                for j, pi in enumerate(pk_idx):
                    # Boundaries: midpoints between adjacent peaks
                    left = (
                        i_from
                        if j == 0
                        else i_from + (pk_idx[j - 1] + pi) // 2
                    )
                    right = (
                        i_to
                        if j == len(pk_idx) - 1
                        else i_from + (pi + pk_idx[j + 1]) // 2
                    )
                    abs_pi = i_from + pi
                    t_pk = times[min(abs_pi, len(times) - 1)]
                    t_l = times[min(left, len(times) - 1)]
                    t_r = times[min(right, len(times) - 1)]
                    split_peaks.append(WavePacketPeak(
                        peak_time=t_pk,
                        prominence=float(marginal[pi]),
                        date_from=t_l,
                        date_to=t_r,
                        band=peak.band,
                        period_sec=peak.period_sec,
                    ))
            else:
                split_peaks.append(peak)
        else:
            split_peaks.append(peak)
    all_peaks = split_peaks

    # Pre-compute band row masks for spectral concentration checks
    periods = 1.0 / freq
    from qp.events.bands import QP_BANDS
    band_row_masks = {}
    for bname, bobj in QP_BANDS.items():
        band_row_masks[bname] = (
            (periods >= bobj.period_min_sec)
            & (periods < bobj.period_max_sec)
        )

    # Post-filter: transverse ratio + min oscillations + spectral conc.
    t_sec = t - t[0]
    filtered: list[WavePacketPeak] = []
    for peak in all_peaks:
        from_sec = (peak.date_from - epoch).total_seconds()
        to_sec = (peak.date_to - epoch).total_seconds()
        i0 = int(np.searchsorted(t_sec, from_sec))
        i1 = int(np.searchsorted(t_sec, to_sec))
        i1 = min(i1, len(t_sec) - 1)
        if i1 <= i0:
            continue

        # Min oscillations: require >= 2.5 cycles
        if peak.period_sec and peak.period_sec > 0:
            duration_sec = to_sec - from_sec
            n_osc = duration_sec / peak.period_sec
            if n_osc < 2.5:
                continue

        # Transverse ratio: compare in-band CWT power (not time-domain
        # RMS, which is contaminated by out-of-band signals like PPO)
        if peak.band and peak.band in band_row_masks:
            bm = band_row_masks[peak.band]
            perp_bp = float(joint_power[bm, i0:i1].mean())
            par_bp = float(power_par[bm, i0:i1].mean())
            if par_bp > 0 and perp_bp / par_bp < 0.5:
                continue

        # Spectral concentration: reject if other bands have comparable
        # power averaged over the event window (broadband signature).
        if peak.band and peak.band in band_row_masks:
            in_band = band_row_masks[peak.band]
            in_band_power = float(
                joint_power[in_band, i0:i1].mean()
            )
            other_power = []
            for ob, om in band_row_masks.items():
                if ob != peak.band and om.any():
                    other_power.append(
                        float(joint_power[om, i0:i1].mean())
                    )
            if other_power:
                max_other = max(other_power)
                # Reject if another band has >= 60% of this band's power
                if max_other > 0.6 * in_band_power:
                    continue

        filtered.append(peak)

    # Deduplicate: within same band, keep higher-power peak
    filtered.sort(key=lambda p: p.peak_time)
    merged: list[WavePacketPeak] = []
    for peak in filtered:
        if merged and peak.band == merged[-1].band:
            sep = abs(
                (peak.peak_time - merged[-1].peak_time).total_seconds()
            )
            if sep < 10800:  # 3h window
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
