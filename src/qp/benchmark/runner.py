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


#: Round 6 / S1 — minimum Stokes degree of polarization for a detection.
#:
#: A coherent wave packet has d → 1 regardless of polarization
#: geometry (linear, circular, elliptical); broadband transients have
#: d → 0. The 0.7 threshold keeps detections that are at least 70%
#: polarized at the dominant frequency over the packet duration.
MIN_DEGREE_OF_POLARIZATION: float = 0.7

#: Round 6 / R1 — minimum spectral quality factor Q = f0 / FWHM.
#:
#: A coherent wave concentrates its power near a single peak period:
#: the Morlet-CWT FWHM at peak time is set by the wavelet bandwidth
#: (Q ~ omega0 / 2.355 ~ 4.3 for omega0=10). A coherent broadband
#: transient (FGM step, range change) has flat in-band power with
#: FWHM equal to the full band width (Q ~ period_centroid / band_width
#: ~ 3 for all three QP bands). Q >= 3 separates these regimes with a
#: defensible single threshold; Stokes alone cannot reject coherent
#: broadband decoys, but Q can.
MIN_Q_FACTOR: float = 3.0

#: Round 6 / N1 — minimum lambda_2/lambda_3 from MVA on the detection
#: window. Alfvén waves are planar transverse perturbations, so the
#: minimum-variance direction is well-defined (lambda_3 << lambda_2)
#: and the ratio is large. An FGM artefact that affects all three
#: axes (range change, calibration step) has all eigenvalues similar
#: and ratio ~ 1. The textbook threshold for a "well-resolved" wave
#: normal is lambda_2/lambda_3 >= 5 (Sonnerup & Scheible 1998).
MIN_MVA_LAMBDA_RATIO: float = 5.0


#: Round 6 / N2 — family-wise error rate for the whitened CWT mask.
#:
#: FWER per 36-h segment. The actual sigma threshold is derived from
#: the effective number of independent time-frequency cells (see
#: ``_bonferroni_n_sigma``); this constant is the only knob.
SEGMENT_FWER_ALPHA: float = 0.01


def _bonferroni_n_sigma(
    n_time: int,
    dt: float,
    freq: np.ndarray,
    morlet_omega0: float = 10.0,
    alpha: float = SEGMENT_FWER_ALPHA,
) -> float:
    r"""Sigma threshold for FWER control over the CWT search volume.

    The Morlet wavelet has temporal correlation length ~1 period at
    every frequency and frequency-bandwidth :math:`\Delta f / f \approx
    1/\omega_0`. The number of *independent* time-frequency cells
    therefore scales as

    .. math::

        V_{\mathrm{indep}} = \underbrace{\frac{\omega_0}{2\pi}
            \ln\!\left(\frac{f_{\max}}{f_{\min}}\right)}_{n_{f,\,\mathrm{indep}}}
        \;\cdot\;
        \underbrace{n_t\,dt\,\bar f}_{\bar n_{t,\,\mathrm{indep}}}

    not the raw :math:`n_f \times n_t` cell count. For the QP
    benchmark (300 freqs, 36-h segment at 60-s sampling, :math:`\omega_0
    = 10`) :math:`V_{\mathrm{indep}} \approx 1{,}400`, four orders of
    magnitude below the raw 6.5 \times 10^5. Bonferroni then sets the
    per-pixel false-positive probability to :math:`\alpha /
    V_{\mathrm{indep}}` and the corresponding Gaussian quantile is the
    threshold (~4.4 sigma at :math:`\alpha = 0.01`).

    Reviewer pitch: *"The sigma threshold controls the family-wise
    error rate over the effective search volume of the wavelet
    scalogram."*
    """
    from scipy.stats import norm

    freq = np.asarray(freq, dtype=float)
    f_min = float(freq[freq > 0].min())
    f_max = float(freq.max())
    n_freq_indep = (morlet_omega0 / (2.0 * np.pi)) * np.log(f_max / f_min)
    n_time_indep = n_time * dt * float(freq.mean())
    v_indep = max(n_freq_indep * n_time_indep, 1.0)
    return float(norm.isf(alpha / v_indep))


def _detect_events_in_dataset(
    t: np.ndarray,
    fields: np.ndarray,
    dt: float = 60.0,
) -> list[WavePacketPeak]:
    """Run the detection pipeline on synthetic 3-component data."""
    from qp.events.bands import get_band
    from qp.events.detector import detect_wave_packets_multi
    from qp.events.threshold import wavelet_sigma_mask
    from qp.signal.polarization import (
        degree_of_polarization,
        mva_intermediate_minimum_ratio,
    )
    from qp.signal.wavelet import morlet_cwt

    b_perp1 = fields[:, 1]
    b_perp2 = fields[:, 2]

    # Complex CWTs of all three field components. Magnitudes of the
    # transverse pair feed ridge extraction; phases feed the Stokes
    # gate; the b_par row is needed by N1 (MVA on bandpass-filtered
    # data, per Sonnerup & Scheible 1998).
    b_par = fields[:, 0]
    freq, _, cwt_par = morlet_cwt(b_par, dt=dt, n_freqs=300)
    _, _, cwt1 = morlet_cwt(b_perp1, dt=dt, n_freqs=300)
    _, _, cwt2 = morlet_cwt(b_perp2, dt=dt, n_freqs=300)
    power1 = np.abs(cwt1)
    power2 = np.abs(cwt2)

    # N2 + S2: per-frequency MAD whitening + sigma threshold derived
    # from the effective CWT search volume. The mask builds a robust
    # noise floor on background rows (outside all QP bands) and
    # interpolates it into the in-band rows; the threshold is the
    # Bonferroni-corrected Gaussian quantile at FWER alpha = 1%.
    n_sigma = _bonferroni_n_sigma(power1.shape[1], dt, freq)
    mask1 = wavelet_sigma_mask(power1, freq, n_sigma=n_sigma)
    mask2 = wavelet_sigma_mask(power2, freq, n_sigma=n_sigma)

    epoch = datetime.datetime(2000, 1, 1)
    times = [epoch + datetime.timedelta(seconds=float(s)) for s in t]

    all_peaks: list[WavePacketPeak] = []
    for component, power, mask in (
        (b_perp1, power1, mask1),
        (b_perp2, power2, mask2),
    ):
        peaks = detect_wave_packets_multi(
            data=component,
            times=times,
            dt=dt,
            cwt_freq=freq,
            cwt_power=power,
            threshold_mask=mask,
            min_duration_hours=2.0,
            # min_pixels is a salt-and-pepper safety net now that the
            # whitened sigma mask handles noise rejection; the floor is
            # well below any plausible real-wave footprint (>= 3 cycles
            # x >= 1 freq row >> 10 px) but still kills single-pixel
            # blobs below the sigma threshold.
            min_pixels=10,
        )
        all_peaks.extend(peaks)

    # Deduplicate: merge peaks within 2h of each other in the same band
    all_peaks.sort(key=lambda p: p.peak_time)
    merged: list[WavePacketPeak] = []
    for peak in all_peaks:
        if merged and peak.band == merged[-1].band:
            sep = abs(
                (peak.peak_time - merged[-1].peak_time).total_seconds()
            )
            if sep < 7200:
                continue
        merged.append(peak)

    # Per-detection physical gates. Each rejects a different decoy
    # mode:
    #   R1 (Q-factor)   — coherent broadband transients (FGM steps)
    #   N1 (MVA ratio)  — non-planar perturbations (3-axis steps)
    #   S1 (Stokes d)   — incoherent broadband bursts
    n_time = cwt1.shape[1]
    kept: list[WavePacketPeak] = []
    for peak in merged:
        if peak.period_sec is None or peak.period_sec <= 0 or peak.band is None:
            continue
        # R1: spectral narrowness
        q = peak.q_factor
        if q is None or q < MIN_Q_FACTOR:
            continue
        # Time window indices used by N1 and S1
        i_start = max(
            0, int(np.floor((peak.date_from - epoch).total_seconds() / dt))
        )
        i_end = min(
            n_time - 1,
            int(np.ceil((peak.date_to - epoch).total_seconds() / dt)),
        )
        if i_end <= i_start:
            continue
        # N1: minimum variance analysis on the 3-component field,
        # bandpass-filtered to the wave frequency by taking the real
        # part of the CWT at the peak period. This isolates the
        # wave's geometry from broadband alpha=1.2 noise that would
        # otherwise dominate the covariance.
        i_freq_peak = int(np.argmin(np.abs(freq - 1.0 / peak.period_sec)))
        field_bp = np.column_stack([
            np.real(cwt_par[i_freq_peak, i_start : i_end + 1]),
            np.real(cwt1[i_freq_peak, i_start : i_end + 1]),
            np.real(cwt2[i_freq_peak, i_start : i_end + 1]),
        ])
        mva_ratio = mva_intermediate_minimum_ratio(field_bp)
        if mva_ratio < MIN_MVA_LAMBDA_RATIO:
            continue
        # S1: polarization purity over the detection's TF window
        band_obj = get_band(peak.band)
        in_band = (
            (freq >= band_obj.freq_min_hz)
            & (freq < band_obj.freq_max_hz)
        )
        if not in_band.any():
            continue
        c1_window = cwt1[in_band, i_start : i_end + 1]
        c2_window = cwt2[in_band, i_start : i_end + 1]
        d = degree_of_polarization(c1_window.ravel(), c2_window.ravel())
        if d < MIN_DEGREE_OF_POLARIZATION:
            continue
        kept.append(peak)

    return kept


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
