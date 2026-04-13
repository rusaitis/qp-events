"""Benchmark dataset generator.

Composes noise backgrounds and injected wave packets into reproducible
synthetic datasets with full ground-truth manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qp.events.bands import QP_BANDS
from qp.events.catalog import WaveTemplate
from qp.signal.noise import magnetospheric_background
from qp.signal.synthetic import simulate_wave_physics
from qp.benchmark.manifest import DatasetManifest, InjectedEvent


@dataclass(frozen=True, slots=True)
class EventSpec:
    """Specification for a single event to inject."""

    band: str = "QP60"
    period_offset_frac: float = 0.0  # offset from band centroid
    amplitude: float = 1.0  # nT
    center_hours: float = 12.0  # hours from dataset start
    decay_hours: float = 4.0  # Gaussian envelope width in hours
    mode: str = "alfvenic"  # "alfvenic", "compressional", "mixed"
    polarization: str = "circular"  # "circular", "linear", "elliptical"
    propagation: str = "standing"  # "standing", "travelling"
    waveform: str = "sine"
    chirp_rate: float = 0.0  # Hz/s
    ellipticity: float = 1.0  # [-1, 1]
    asymmetry: float = 0.5  # envelope asymmetry
    amplitude_jitter: float = 0.0
    sawtooth_width: float = 0.8
    harmonic_content: float = 0.0
    should_detect: bool = True
    difficulty: str = "easy"
    event_type: str = "qp_wave"
    # For non-band signals (decoys at specific periods)
    period_sec_override: float | None = None


@dataclass
class ScenarioConfig:
    """Configuration for one benchmark dataset."""

    dataset_id: str
    description: str = ""
    duration_days: float = 10.0
    dt: float = 60.0
    noise_alpha: float = 1.2
    noise_sigma: float = 0.05
    background_trend: bool = True
    difficulty_tier: str = "tier1"
    event_specs: list[EventSpec] = field(default_factory=list)


def _resolve_period(spec: EventSpec) -> float:
    """Compute the injection period in seconds from band centroid + offset."""
    if spec.period_sec_override is not None:
        return spec.period_sec_override
    band_obj = QP_BANDS.get(spec.band.upper())
    if band_obj is None:
        raise ValueError(f"Unknown band: {spec.band!r}")
    centroid = band_obj.period_centroid_sec
    return centroid * (1.0 + spec.period_offset_frac)


def generate_benchmark_dataset(
    scenario: ScenarioConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, DatasetManifest]:
    r"""Generate one benchmark dataset with injected events.

    Parameters
    ----------
    scenario : ScenarioConfig
        What to generate.
    seed : int
        Master seed for full reproducibility.

    Returns
    -------
    time : ndarray, shape (n_samples,)
        Time array in seconds.
    fields : ndarray, shape (n_samples, 4)
        Columns [B_par, B_perp1, B_perp2, B_tot].
    manifest : DatasetManifest
        Ground truth with all injected event metadata.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(scenario.duration_days * 86400 / scenario.dt)
    t = np.arange(n_samples) * scenario.dt

    # Background
    if scenario.background_trend:
        bg = magnetospheric_background(
            n_samples, scenario.dt,
            seed=int(rng.integers(0, 2**31)),
            noise_alpha=scenario.noise_alpha,
            noise_sigma=scenario.noise_sigma,
        )
    else:
        from qp.signal.noise import colored_noise_3component

        bg = colored_noise_3component(
            n_samples, scenario.dt,
            alpha=scenario.noise_alpha,
            sigma=scenario.noise_sigma,
            seed=int(rng.integers(0, 2**31)),
        )

    b_par = bg[:, 0].copy()
    b_perp1 = bg[:, 1].copy()
    b_perp2 = bg[:, 2].copy()

    # Inject events
    injected_events: list[InjectedEvent] = []

    for i, spec in enumerate(scenario.event_specs):
        period_sec = _resolve_period(spec)
        center_sec = spec.center_hours * 3600.0
        decay_sec = spec.decay_hours * 3600.0

        wave = WaveTemplate(
            period=period_sec,
            amplitude=spec.amplitude,
            shift=center_sec,
            decay_width=decay_sec,
            waveform=spec.waveform,
            chirp_rate=spec.chirp_rate,
            asymmetry=spec.asymmetry,
            amplitude_jitter=spec.amplitude_jitter,
            sawtooth_width=spec.sawtooth_width,
            harmonic_content=spec.harmonic_content,
        )

        _, wave_fields = simulate_wave_physics(
            n_samples, scenario.dt, [wave],
            mode=spec.mode,
            polarization=spec.polarization,
            ellipticity=spec.ellipticity,
            propagation=spec.propagation,
            seed=int(rng.integers(0, 2**31)),
        )

        b_par += wave_fields[:, 0]
        b_perp1 += wave_fields[:, 1]
        b_perp2 += wave_fields[:, 2]

        # Compute event boundaries (3-sigma envelope width)
        half_width = 3.0 * decay_sec
        start_sec = max(0.0, center_sec - half_width)
        end_sec = min(t[-1], center_sec + half_width)
        duration_sec = end_sec - start_sec
        n_osc = duration_sec / period_sec

        band_label = spec.band if spec.should_detect else None

        injected_events.append(InjectedEvent(
            event_id=f"{scenario.dataset_id}-{i:03d}",
            dataset_id=scenario.dataset_id,
            event_type=spec.event_type,
            should_detect=spec.should_detect,
            band=band_label,
            period_sec=period_sec,
            amplitude_nT=spec.amplitude,
            start_sec=start_sec,
            end_sec=end_sec,
            center_sec=center_sec,
            duration_sec=duration_sec,
            n_oscillations=n_osc,
            polarization=spec.polarization,
            ellipticity=spec.ellipticity,
            wave_mode=spec.mode,
            propagation=spec.propagation,
            chirp_rate=spec.chirp_rate,
            waveform=spec.waveform,
            sawtooth_width=spec.sawtooth_width,
            envelope_asymmetry=spec.asymmetry,
            amplitude_jitter=spec.amplitude_jitter,
            harmonic_content=spec.harmonic_content,
            snr_injected=spec.amplitude / max(scenario.noise_sigma, 1e-30),
            difficulty=spec.difficulty,
        ))

    b_tot = np.sqrt(b_par**2 + b_perp1**2 + b_perp2**2)
    fields = np.column_stack([b_par, b_perp1, b_perp2, b_tot])

    manifest = DatasetManifest(
        dataset_id=scenario.dataset_id,
        description=scenario.description,
        duration_days=scenario.duration_days,
        dt=scenario.dt,
        n_samples=n_samples,
        seed=seed,
        noise_alpha=scenario.noise_alpha,
        noise_sigma=scenario.noise_sigma,
        difficulty_tier=scenario.difficulty_tier,
        events=injected_events,
    )

    return t, fields, manifest
