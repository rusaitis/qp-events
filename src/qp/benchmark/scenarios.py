"""Pre-defined benchmark scenarios at four difficulty tiers plus decoys.

Each function returns a :class:`ScenarioConfig` with deterministic seeds.
Events are spaced to avoid overlap (≥30 h between packet centers for
QP30/QP60, ≥40 h for QP120).

Scenario counts:
    Tier 1 (easy):     5 scenarios
    Tier 2 (moderate): 6 scenarios
    Tier 3 (hard):     12 scenarios (+ gaps, steep noise, PPO beat, plasma sheet)
    Tier 4 (extreme):  11 scenarios (+ heavy gaps, Kolmogorov, non-stationary)
    Decoy:             6 scenarios (+ roll artifacts)
    TOTAL:             40 scenarios
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from qp.benchmark.generator import (
    EventSpec,
    GapSpec,
    NoiseBurstSpec,
    RollArtifactSpec,
    ScenarioConfig,
)
from qp.events.bands import QP_BANDS, classify_period

_MIN = 60.0  # seconds

# Per-scenario RNG base, independent of the generator's noise seed so that
# scenario structure (sampled periods, amplitudes) is reproducible across
# runs. Picked to not collide with BASE_SEED in runner.py.
_SCENARIO_SEED_BASE: int = 0x5CE_A10


# ======================================================================
# Helpers
# ======================================================================

def _centers(
    n_events: int,
    duration_days: float,
    margin_h: float = 12.0,
    *,
    jitter_frac: float = 0.4,
    rng_seed: str | None = None,
) -> list[float]:
    r"""Return ``n_events`` center times (hours) across the dataset.

    Events are evenly spaced with ``margin_h`` buffer from the edges,
    then perturbed by a uniform jitter of ±``jitter_frac·spacing/2`` so
    the pattern is not strictly periodic. The jitter is seeded from
    ``(n_events, duration_days, margin_h)`` by default — deterministic
    across runs — or from the explicit ``rng_seed`` when provided.
    Pass ``jitter_frac=0`` to recover the old exact-grid behaviour.
    """
    total_h = duration_days * 24
    usable = total_h - 2 * margin_h
    spacing = usable / max(n_events, 1)
    base = [margin_h + i * spacing + spacing / 2 for i in range(n_events)]
    if jitter_frac <= 0:
        return base
    key = rng_seed or f"_centers::{n_events}::{duration_days}::{margin_h}"
    rng = _scenario_rng(key)
    amp = jitter_frac * spacing / 2.0
    return [float(c + rng.uniform(-amp, amp)) for c in base]


def _scenario_rng(dataset_id: str) -> np.random.Generator:
    """Deterministic RNG seeded from the scenario ID (stable across runs).

    Uses SHA-1 over the id so the seed is reproducible across Python
    processes — `hash()` would be PYTHONHASHSEED-randomized.
    """
    digest = hashlib.sha1(dataset_id.encode("utf-8")).digest()[:8]
    id_hash = int.from_bytes(digest, "big") & 0xFFFF_FFFF
    seed = _SCENARIO_SEED_BASE ^ id_hash
    return np.random.default_rng(seed)


def _sample_periods_in_band(
    band_name: str, n: int, rng: np.random.Generator,
) -> list[float]:
    r"""Log-uniform sample ``n`` periods (in seconds) across a QP band.

    Avoids pinning events to exact band centroids (30/60/120 min) so the
    detector has to demonstrate band-general detection rather than
    learning the injection points.
    """
    band = QP_BANDS[band_name]
    log_lo = math.log(band.period_min_sec)
    log_hi = math.log(band.period_max_sec)
    return [float(math.exp(x)) for x in rng.uniform(log_lo, log_hi, n)]


def _amplitude_for_target_snr(
    band_name: str,
    target_snr: float,
    noise_alpha: float,
    noise_sigma: float,
    dt: float = 60.0,
) -> float:
    r"""Solve for the amplitude that yields ``target_snr`` in band.

    Uses the analytic power-law integral to estimate the in-band noise
    RMS for a given (alpha, sigma) background and returns the amplitude
    such that ``A / noise_rms_in_band = target_snr``. This neutralises
    the band-width bias: with constant ``noise_sigma``, QP120 has less
    in-band noise than QP30 at α=1.2, so an amplitude-matched test is
    a difficulty-biased test. This helper lets scenarios request equal
    in-band difficulty across bands.
    """
    from qp.benchmark.generator import _in_band_snr

    # The analytic SNR is linear in amplitude, so A_target = target_snr
    # divided by the per-unit-amplitude analytic SNR.
    snr_per_unit = _in_band_snr(
        amplitude=1.0,
        noise_sigma=noise_sigma,
        noise_alpha=noise_alpha,
        period_sec=QP_BANDS[band_name].period_centroid_sec,
        dt=dt,
        band_name=band_name,
    )
    if snr_per_unit <= 0:
        return target_snr  # fallback
    return float(target_snr / snr_per_unit)


def _balanced_band_assignment(n: int, rng: np.random.Generator) -> list[str]:
    r"""Assign ``n`` events to QP30/QP60/QP120 as evenly as possible.

    Guarantees per-band counts differ by at most 1 and shuffles the
    order so the assignment is not correlated with center_hours.
    """
    bands: list[str] = []
    for i in range(n):
        bands.append(("QP30", "QP60", "QP120")[i % 3])
    rng.shuffle(bands)
    return bands


def _band_snr_scale(
    band_name: str,
    noise_alpha: float,
    noise_sigma: float,
    dt: float = 60.0,
    reference_band: str = "QP60",
) -> float:
    r"""Amplitude multiplier to match in-band SNR of ``reference_band``.

    With constant broadband ``noise_sigma``, QP120 carries less in-band
    noise power than QP30 (different bandwidths × different power-law
    density). An event injected at the same amplitude into different
    bands therefore has different in-band SNR. This helper returns a
    multiplier so that ``A * scale`` matches the SNR the amplitude
    would have in ``reference_band``.
    """
    from qp.benchmark.generator import _in_band_snr

    ref = _in_band_snr(
        1.0, noise_sigma, noise_alpha,
        QP_BANDS[reference_band].period_centroid_sec, dt, reference_band,
    )
    this = _in_band_snr(
        1.0, noise_sigma, noise_alpha,
        QP_BANDS[band_name].period_centroid_sec, dt, band_name,
    )
    if this <= 0:
        return 1.0
    return float(ref / this)


def _multiband_specs(
    dataset_id: str,
    centers: list[float],
    *,
    amplitude: float | list[float] = 1.0,
    n_cycles: float = 5.0,
    n_cycles_sigma: float = 0.3,
    difficulty: str = "moderate",
    equalize_band_snr: bool = True,
    noise_alpha: float | None = None,
    noise_sigma: float | None = None,
    amp_jitter_sigma: float = 0.0,
    dt: float = 60.0,
    **extra,
) -> list[EventSpec]:
    r"""Build ``len(centers)`` EventSpecs distributed evenly across bands.

    Each event's period is log-uniform sampled within its assigned
    band. ``decay_hours`` is set to ``n_cycles·P/4`` where
    ``n_cycles`` is drawn per-event from ``LogNormal(ln(n_cycles),
    n_cycles_sigma)`` — real catalog packets show substantial
    duration scatter at fixed period, and deterministically coupling
    duration to period ("5-cycle always") turns envelope width into
    a free label. Pass ``n_cycles_sigma=0`` to recover the old
    deterministic behaviour. ``extra`` is passed through to
    :class:`EventSpec` unchanged.

    Parameters
    ----------
    amplitude : float or list[float]
        Reference amplitude (interpreted as QP60-band amplitude when
        ``equalize_band_snr`` is True).
    n_cycles : float
        Median number of cycles enclosed by the ±2σ envelope window.
    n_cycles_sigma : float
        Log-normal scatter of ``n_cycles`` per event. 0 disables
        jitter. 0.3 (default) ≈ ±30 % fractional scatter — matches
        the observed spread in the v5 catalog.
    equalize_band_snr : bool
        If True (default), scale each event's amplitude so its analytic
        in-band SNR matches what ``amplitude`` would produce at QP60.
        Removes the bandwidth-driven difficulty bias (QP120 easier,
        QP30 harder) when ``noise_sigma`` is constant. Needs
        ``noise_alpha`` and ``noise_sigma``; silently falls back to
        raw ``amplitude`` if either is ``None``.
    amp_jitter_sigma : float
        If > 0, multiplies every amplitude by ``exp(sigma * N(0,1))``
        (log-normal jitter) so the distribution is continuous.
    """
    rng = _scenario_rng(dataset_id)
    bands = _balanced_band_assignment(len(centers), rng)
    raw_amps = (
        list(amplitude)
        if isinstance(amplitude, (list, tuple))
        else [float(amplitude)] * len(centers)
    )
    if equalize_band_snr and noise_alpha is not None and noise_sigma is not None:
        amps = [
            a * _band_snr_scale(b, noise_alpha, noise_sigma, dt)
            for a, b in zip(raw_amps, bands)
        ]
    else:
        amps = raw_amps
    if amp_jitter_sigma > 0:
        jitter = rng.lognormal(mean=0.0, sigma=amp_jitter_sigma, size=len(amps))
        amps = [float(a * j) for a, j in zip(amps, jitter)]
    # Log-normal n_cycles per event. Use ``math.log(n_cycles)`` as
    # the mean so the *median* of the distribution is ``n_cycles``.
    if n_cycles_sigma > 0:
        cycles_draw = rng.lognormal(
            mean=math.log(n_cycles), sigma=n_cycles_sigma, size=len(centers),
        )
    else:
        cycles_draw = np.full(len(centers), n_cycles)
    specs: list[EventSpec] = []
    for c, b, a, n_cyc in zip(centers, bands, amps, cycles_draw):
        p_sec = _sample_periods_in_band(b, 1, rng)[0]
        # ±2σ envelope encloses ~n_cyc full cycles → σ = n·P/4.
        decay_h = float(n_cyc) * p_sec / (4.0 * 3600.0)
        specs.append(EventSpec(
            band=b, period_sec_override=p_sec,
            amplitude=float(a), center_hours=c,
            decay_hours=decay_h, difficulty=difficulty,
            **extra,
        ))
    return specs


# ======================================================================
# TIER 1: Easy (target ≥95% recall)
# ======================================================================

def tier1_clean_qp30() -> ScenarioConfig:
    dataset_id = "tier1_clean_qp30"
    centers = _centers(8, 10)
    rng = _scenario_rng(dataset_id)
    periods = _sample_periods_in_band("QP30", 8, rng)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="8 QP30 sine packets at log-uniform periods in 20–40 min",
        duration_days=10, noise_alpha=0.0, noise_sigma=0.01,
        background_trend=False, difficulty_tier="tier1",
        event_specs=[
            EventSpec(band="QP30", period_sec_override=p,
                      amplitude=2.0, center_hours=c,
                      decay_hours=2.0, difficulty="easy")
            for c, p in zip(centers, periods)
        ],
    )


def tier1_clean_qp60() -> ScenarioConfig:
    dataset_id = "tier1_clean_qp60"
    centers = _centers(8, 10)
    rng = _scenario_rng(dataset_id)
    periods = _sample_periods_in_band("QP60", 8, rng)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="8 QP60 sine packets at log-uniform periods in 45–80 min",
        duration_days=10, noise_alpha=0.0, noise_sigma=0.01,
        background_trend=False, difficulty_tier="tier1",
        event_specs=[
            EventSpec(band="QP60", period_sec_override=p,
                      amplitude=2.0, center_hours=c,
                      decay_hours=4.0, difficulty="easy")
            for c, p in zip(centers, periods)
        ],
    )


def tier1_clean_qp120() -> ScenarioConfig:
    dataset_id = "tier1_clean_qp120"
    centers = _centers(8, 10)
    rng = _scenario_rng(dataset_id)
    periods = _sample_periods_in_band("QP120", 8, rng)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="8 QP120 sine packets at log-uniform periods in 90–150 min",
        duration_days=10, noise_alpha=0.0, noise_sigma=0.01,
        background_trend=False, difficulty_tier="tier1",
        event_specs=[
            EventSpec(band="QP120", period_sec_override=p,
                      amplitude=2.0, center_hours=c,
                      decay_hours=6.0, difficulty="easy")
            for c, p in zip(centers, periods)
        ],
    )


def tier1_mixed_bands() -> ScenarioConfig:
    centers = _centers(12, 15)
    bands = ["QP30", "QP60", "QP120"] * 4
    decays = [2.0, 4.0, 6.0] * 4
    return ScenarioConfig(
        dataset_id="tier1_mixed_bands",
        description="12 events (4 per band), white noise, well-separated",
        duration_days=15, noise_alpha=0.0, noise_sigma=0.01,
        background_trend=False, difficulty_tier="tier1",
        event_specs=[
            EventSpec(band=b, amplitude=2.0, center_hours=c,
                      decay_hours=d, difficulty="easy")
            for b, c, d in zip(bands, centers, decays)
        ],
    )


def tier1_full_spectrum() -> ScenarioConfig:
    """Sweep across the full detectable spectrum (5–180 min)."""
    periods_min = [5, 10, 20, 30, 45, 60, 80, 90, 100, 110, 120, 130, 150, 180]
    centers = _centers(len(periods_min), 15)
    specs = []
    for p, c in zip(periods_min, centers):
        band_name = classify_period(p * _MIN)
        specs.append(EventSpec(
            band=band_name or "QP60",
            period_sec_override=p * _MIN,
            amplitude=3.0, center_hours=c, decay_hours=max(2.0, p / 30),
            should_detect=band_name is not None,
            difficulty="easy",
        ))
    return ScenarioConfig(
        dataset_id="tier1_full_spectrum",
        description="Period sweep 5–180 min, high amplitude, white noise",
        duration_days=15, noise_alpha=0.0, noise_sigma=0.02,
        background_trend=False, difficulty_tier="tier1",
        event_specs=specs,
    )


# ======================================================================
# TIER 2: Moderate (target 70–90% recall)
# ======================================================================

def tier2_colored_noise() -> ScenarioConfig:
    dataset_id = "tier2_colored_noise"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="QP30/60/120 in realistic colored noise (alpha=1.2)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.0, n_cycles=5.0,
            difficulty="moderate",
            noise_alpha=1.2, noise_sigma=0.07,
            amp_jitter_sigma=0.25,
        ),
    )


def tier2_low_amplitude() -> ScenarioConfig:
    dataset_id = "tier2_low_amplitude"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Low-amplitude (0.3 nT) waves across all bands",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=0.3, n_cycles=5.0,
            difficulty="moderate",
            noise_alpha=1.2, noise_sigma=0.07,
            amp_jitter_sigma=0.25,
        ),
    )


def tier2_sawtooth_shapes() -> ScenarioConfig:
    centers = _centers(8, 10)
    shapes = [
        ("sawtooth", 0.2), ("sawtooth", 0.8), ("sawtooth", 0.0), ("sawtooth", 1.0),
        ("square", 0.8), ("sine", 0.8), ("sawtooth", 0.5), ("sine", 0.8),
    ]
    return ScenarioConfig(
        dataset_id="tier2_sawtooth_shapes",
        description="Mixed waveform shapes in realistic noise",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=3.0, waveform=wf, sawtooth_width=sw,
                      difficulty="moderate")
            for c, (wf, sw) in zip(centers, shapes)
        ],
    )


def tier2_linear_pol() -> ScenarioConfig:
    dataset_id = "tier2_linear_pol"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Linear-polarization waves across all bands, realistic noise",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.0, n_cycles=6.0,
            difficulty="moderate",
            polarization="linear", ellipticity=0.0,
            noise_alpha=1.2, noise_sigma=0.07,
            amp_jitter_sigma=0.25,
        ),
    )


def tier2_short_packets() -> ScenarioConfig:
    dataset_id = "tier2_short_packets"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Short packets (~4 oscillations) across all bands",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.0, n_cycles=4.0,
            difficulty="moderate",
            noise_alpha=1.2, noise_sigma=0.07,
            amp_jitter_sigma=0.25,
        ),
    )


def tier2_ppo_background() -> ScenarioConfig:
    dataset_id = "tier2_ppo_background"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Waves across bands with full magnetospheric background",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=True, difficulty_tier="tier2",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.5, n_cycles=5.0,
            difficulty="moderate",
            noise_alpha=1.2, noise_sigma=0.07,
            amp_jitter_sigma=0.25,
        ),
    )


# ======================================================================
# TIER 3: Hard (target 30–70% recall)
# ======================================================================

def tier3_overlapping_bands() -> ScenarioConfig:
    centers = _centers(4, 10)
    specs = []
    for c in centers:
        specs.append(EventSpec(band="QP30", amplitude=1.0, center_hours=c,
                               decay_hours=2.0, difficulty="hard"))
        specs.append(EventSpec(band="QP60", amplitude=1.0, center_hours=c + 0.5,
                               decay_hours=3.0, difficulty="hard"))
    return ScenarioConfig(
        dataset_id="tier3_overlapping_bands",
        description="Simultaneous QP30 + QP60 packets (band separation test)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=specs,
    )


def tier3_frequency_drift() -> ScenarioConfig:
    dataset_id = "tier3_frequency_drift"
    centers = _centers(18, 10)
    # chirp_rate in Hz/s; scaled per-band below via _multiband_specs so
    # each band gets comparable fractional frequency drift.
    rng = _scenario_rng(dataset_id + "_chirp")
    chirps = rng.choice([2e-9, -2e-9, 5e-9, -5e-9, 3e-9, -3e-9], 9).tolist()
    specs = []
    base = _multiband_specs(
        dataset_id, centers, amplitude=1.0, n_cycles=5.0,
        difficulty="hard", asymmetry=0.3, propagation="travelling",
    )
    for s, cr in zip(base, chirps):
        # Re-create with chirp_rate (EventSpec is a dataclass, simpler to rebuild)
        specs.append(EventSpec(
            band=s.band, period_sec_override=s.period_sec_override,
            amplitude=s.amplitude, center_hours=s.center_hours,
            decay_hours=s.decay_hours, difficulty=s.difficulty,
            asymmetry=s.asymmetry, propagation=s.propagation,
            chirp_rate=float(cr),
        ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Chirped waves across all bands (travelling signature)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=specs,
    )


def tier3_asymmetric_envelope() -> ScenarioConfig:
    dataset_id = "tier3_asymmetric_envelope"
    centers = _centers(18, 10)
    rng = _scenario_rng(dataset_id + "_asym")
    asym = rng.choice(
        [0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85], 9,
    ).tolist()
    base = _multiband_specs(
        dataset_id, centers, amplitude=1.0, n_cycles=5.0,
        difficulty="hard",
    )
    specs = [
        EventSpec(
            band=s.band, period_sec_override=s.period_sec_override,
            amplitude=s.amplitude, center_hours=s.center_hours,
            decay_hours=s.decay_hours, difficulty=s.difficulty,
            asymmetry=float(a),
        )
        for s, a in zip(base, asym)
    ]
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Asymmetric envelopes across all bands",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=specs,
    )


def tier3_amplitude_jitter() -> ScenarioConfig:
    dataset_id = "tier3_amplitude_jitter"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="30 % per-cycle amplitude jitter across all bands",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.0, n_cycles=5.0,
            difficulty="hard", amplitude_jitter=0.3,
        ),
    )


def tier3_near_threshold() -> ScenarioConfig:
    dataset_id = "tier3_near_threshold"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="SNR~3 threshold test across all bands (0.15 nT)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=0.15, n_cycles=5.0,
            difficulty="hard",
        ),
    )


def tier3_mixed_waveforms() -> ScenarioConfig:
    dataset_id = "tier3_mixed_waveforms"
    centers = _centers(18, 10)
    rng = _scenario_rng(dataset_id + "_wf")
    waveforms = rng.choice(["sine", "sawtooth", "square"], 9).tolist()
    base = _multiband_specs(
        dataset_id, centers, amplitude=1.0, n_cycles=5.0,
        difficulty="hard",
    )
    specs = [
        EventSpec(
            band=s.band, period_sec_override=s.period_sec_override,
            amplitude=s.amplitude, center_hours=s.center_hours,
            decay_hours=s.decay_hours, difficulty=s.difficulty,
            waveform=str(wf),
        )
        for s, wf in zip(base, waveforms)
    ]
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Mixed sine/sawtooth/square across all bands",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=specs,
    )


def tier3_travelling_waves() -> ScenarioConfig:
    dataset_id = "tier3_travelling_waves"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Travelling Alfvén wave packets across all bands",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.0, n_cycles=5.0,
            difficulty="hard",
            chirp_rate=3e-9, asymmetry=0.3, propagation="travelling",
        ),
    )


# ======================================================================
# TIER 4: Extreme (target <30% recall)
# ======================================================================

def tier4_buried_in_noise() -> ScenarioConfig:
    dataset_id = "tier4_buried_in_noise"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="SNR < 1.5 across all bands (nearly indistinguishable)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=True, difficulty_tier="tier4",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=0.07, n_cycles=5.0,
            difficulty="extreme",
        ),
    )


def tier4_qp120_short() -> ScenarioConfig:
    centers = _centers(6, 10)
    return ScenarioConfig(
        dataset_id="tier4_qp120_short",
        description="QP120 with only ~3 oscillations (6h duration)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP120", amplitude=0.5, center_hours=c,
                      decay_hours=2.0, difficulty="extreme")
            for c in centers
        ],
    )


def tier4_inter_band_period() -> ScenarioConfig:
    """Periods straddling the QP30/QP60 boundary (40-45 min).

    With contiguous geo-mean bands the old 40–45 min "gap" no longer
    exists — 40 min falls in QP30, 43+ min falls in QP60. Kept as a
    test of detection right at the band boundary.
    """
    centers = _centers(6, 10)
    periods = [40, 41, 42, 43, 44, 45]
    return ScenarioConfig(
        dataset_id="tier4_inter_band_period",
        description="Waves straddling the QP30/QP60 boundary (40–45 min)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band=classify_period(p * _MIN) or "QP60",
                      period_sec_override=p * _MIN,
                      amplitude=1.0, center_hours=c, decay_hours=4.0,
                      should_detect=True, difficulty="extreme",
                      event_type="qp_wave_boundary")
            for p, c in zip(periods, centers)
        ],
    )


def tier4_extreme_chirp() -> ScenarioConfig:
    centers = _centers(6, 10)
    return ScenarioConfig(
        dataset_id="tier4_extreme_chirp",
        description="Freq sweeps across band boundary during packet",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                      decay_hours=5.0, chirp_rate=1e-8,
                      propagation="travelling", difficulty="extreme")
            for c in centers
        ],
    )


def tier4_decaying() -> ScenarioConfig:
    """QP120 with strong amplitude decay (−4.2 dB/period from real stats)."""
    centers = _centers(6, 10)
    return ScenarioConfig(
        dataset_id="tier4_decaying",
        description="QP120 with decaying amplitude (-4.2 dB/period)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP120", amplitude=1.0, center_hours=c,
                      decay_hours=3.0, asymmetry=0.8, difficulty="extreme")
            for c in centers
        ],
    )


def tier4_incoherent() -> ScenarioConfig:
    dataset_id = "tier4_incoherent"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="50 % per-cycle jitter across all bands (low coherence)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=0.5, n_cycles=5.0,
            difficulty="extreme", amplitude_jitter=0.5,
        ),
    )


def tier4_harmonic_contamination() -> ScenarioConfig:
    """2f harmonic distortion across fundamentals from all three bands.

    4 fundamentals per band (12 total), periods log-uniform within
    their band. 2f harmonic lands at:
      QP30 (21-42 min)  fundamentals → 2f at 10-21 min (sub_qp30/below)
      QP60 (42-85 min)  fundamentals → 2f at 21-42 min (QP30)
      QP120 (85-170 min) fundamentals → 2f at 42-85 min (QP60)

    The harmonic suppression filter must handle all three cases, not
    just the historical QP60→QP30 test.
    """
    dataset_id = "tier4_harmonic_contamination"
    rng = _scenario_rng(dataset_id)
    centers = _centers(12, 12)
    bands = _balanced_band_assignment(12, rng)
    specs = []
    for c, b in zip(centers, bands):
        p_sec = _sample_periods_in_band(b, 1, rng)[0]
        specs.append(EventSpec(
            band=b, period_sec_override=p_sec,
            amplitude=0.8, center_hours=c,
            decay_hours=5.0 * p_sec / (4.0 * 3600.0),  # ~5 cycles
            harmonic_content=0.4, difficulty="extreme",
        ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="2f harmonic distortion, fundamentals across QP30/60/120",
        duration_days=12, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=specs,
    )


def tier4_elliptical_pol() -> ScenarioConfig:
    dataset_id = "tier4_elliptical_pol"
    centers = _centers(18, 10)
    rng = _scenario_rng(dataset_id + "_ell")
    ellipticities = rng.choice(
        [0.3, -0.3, 0.4, -0.4, 0.5, -0.5], 9,
    ).tolist()
    base = _multiband_specs(
        dataset_id, centers, amplitude=0.8, n_cycles=5.0,
        difficulty="extreme", polarization="elliptical",
    )
    specs = [
        EventSpec(
            band=s.band, period_sec_override=s.period_sec_override,
            amplitude=s.amplitude, center_hours=s.center_hours,
            decay_hours=s.decay_hours, difficulty=s.difficulty,
            polarization=s.polarization, ellipticity=float(e),
        )
        for s, e in zip(base, ellipticities)
    ]
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Elliptical-polarization waves across all bands",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=specs,
    )


# ======================================================================
# DECOY datasets (target: 0% detection)
# ======================================================================

def decoy_compressional() -> ScenarioConfig:
    centers = _centers(6, 10)
    return ScenarioConfig(
        dataset_id="decoy_compressional",
        description="Compressional oscillations at QP60 freq (B_par only)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="decoy",
        event_specs=[
            EventSpec(band="QP60", amplitude=2.0, center_hours=c,
                      decay_hours=4.0, mode="compressional",
                      should_detect=False, difficulty="easy",
                      event_type="decoy_compressional")
            for c in centers
        ],
    )


def decoy_single_pulses() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="decoy_single_pulses",
        description="Isolated 1-cycle magnetic pressure pulses",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="decoy",
        event_specs=[
            EventSpec(band="QP60", amplitude=3.0, center_hours=c,
                      decay_hours=0.5, mode="compressional",
                      should_detect=False, difficulty="easy",
                      event_type="decoy_single_pulse")
            for c in centers
        ],
    )


def decoy_broadband_burst() -> ScenarioConfig:
    """Broadband power bursts — bandpass-filtered Gaussian noise.

    Uses bandlimited_noise_burst() to produce genuine broadband power
    that is spectrally indistinguishable from a continuous noise process
    in a CWT scalogram (unlike discrete sinusoids which produce
    resolvable ridges).
    """
    centers = _centers(4, 10)
    return ScenarioConfig(
        dataset_id="decoy_broadband_burst",
        description="Bandlimited noise bursts (continuous broadband power)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="decoy",
        event_specs=[
            EventSpec(
                band="QP60", amplitude=1.5, center_hours=c,
                decay_hours=2.0, should_detect=False, difficulty="easy",
                event_type="decoy_broadband",
                injection_type="noise_burst",
            )
            for c in centers
        ],
    )


def decoy_step_functions() -> ScenarioConfig:
    centers = _centers(4, 10)
    return ScenarioConfig(
        dataset_id="decoy_step_functions",
        description="Slow step-like B_tot changes (magnetopause crossings)",
        duration_days=10, noise_alpha=0.5, noise_sigma=0.02,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="decoy",
        event_specs=[
            EventSpec(band="QP60", period_sec_override=12 * 3600.0,
                      amplitude=5.0, center_hours=c, decay_hours=6.0,
                      mode="compressional", waveform="square",
                      should_detect=False, difficulty="easy",
                      event_type="decoy_step")
            for c in centers
        ],
    )


def decoy_ppo_only() -> ScenarioConfig:
    return ScenarioConfig(
        dataset_id="decoy_ppo_only",
        description="Only PPO modulation at 10.7h (no QP-band signal)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=True, difficulty_tier="decoy",
        event_specs=[
            EventSpec(band="QP60", period_sec_override=10.7 * 3600.0,
                      amplitude=0.0, center_hours=120.0, decay_hours=100.0,
                      should_detect=False, difficulty="easy",
                      event_type="decoy_ppo")
        ],
    )


# ======================================================================
# NEW: Data gaps, artifacts, non-stationary noise, steep spectra
# ======================================================================


def tier3_gaps_at_onset() -> ScenarioConfig:
    """Data gaps at the rising edge of events across all bands."""
    dataset_id = "tier3_gaps_at_onset"
    centers = _centers(18, 10)
    gaps = [GapSpec(center_hours=c - 1.0, duration_minutes=10) for c in centers]
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Events across bands with 10-min gaps at onset",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.0, n_cycles=5.0,
            difficulty="hard",
        ),
        gaps=gaps,
    )


def tier3_gaps_midpacket() -> ScenarioConfig:
    """Data gaps in the middle of events across all bands."""
    dataset_id = "tier3_gaps_midpacket"
    centers = _centers(18, 10)
    gaps = [GapSpec(center_hours=c, duration_minutes=8) for c in centers]
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Events across bands with 8-min gaps at midpoint",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.0, n_cycles=5.0,
            difficulty="hard",
        ),
        gaps=gaps,
    )


def tier4_heavy_gaps() -> ScenarioConfig:
    """Frequent data gaps (~every 2h, 5–15 min each)."""
    dataset_id = "tier4_heavy_gaps"
    gaps = [
        GapSpec(center_hours=h, duration_minutes=5 + (h % 3) * 5)
        for h in range(12, 228, 2)  # every 2h across 10 days
    ]
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Waves across bands with frequent gaps (every ~2h, 5-15 min)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=0.8, n_cycles=5.0,
            difficulty="extreme",
        ),
        gaps=gaps,
    )


def tier3_steep_noise() -> ScenarioConfig:
    """Steep α = 1.5 noise across all bands (von Papen et al. 2014)."""
    dataset_id = "tier3_steep_noise"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Waves across bands in steep colored noise (alpha=1.5)",
        duration_days=10, noise_alpha=1.5, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.0, n_cycles=5.0,
            difficulty="hard",
        ),
    )


def tier4_kolmogorov_noise() -> ScenarioConfig:
    """Waves across bands in Kolmogorov α ≈ 5/3 ≈ 1.7 noise (Xu et al. 2023)."""
    dataset_id = "tier4_kolmogorov_noise"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Waves across bands in Kolmogorov turbulence (alpha=1.7)",
        duration_days=10, noise_alpha=1.7, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=0.8, n_cycles=5.0,
            difficulty="extreme",
        ),
    )


def tier3_ppo_beat() -> ScenarioConfig:
    """Dual-PPO beat constructively enhancing transverse power."""
    dataset_id = "tier3_ppo_beat"
    centers = _centers(18, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Waves across bands with dual N/S PPO beat modulation",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=True, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.0, n_cycles=5.0,
            difficulty="hard",
        ),
    )


def tier3_plasma_sheet() -> ScenarioConfig:
    """Waves across bands with localized noise bursts (plasma sheet)."""
    dataset_id = "tier3_plasma_sheet"
    centers = _centers(18, 10)
    bursts = [
        NoiseBurstSpec(center_hours=c, duration_hours=3.0,
                       sigma_multiplier=5.0)
        for c in centers[:3]
    ]
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Waves across bands overlapping 5x noise bursts (plasma sheet)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.0, n_cycles=5.0,
            difficulty="hard",
        ),
        noise_bursts=bursts,
    )


def tier4_nonstationary() -> ScenarioConfig:
    """Waves across bands in continuously varying background noise."""
    dataset_id = "tier4_nonstationary"
    centers = _centers(18, 10)
    bursts = [
        NoiseBurstSpec(center_hours=h, duration_hours=5.0,
                       sigma_multiplier=3.0 + (h % 7))
        for h in range(24, 216, 18)
    ]
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Waves across bands in non-stationary noise (varying σ)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=0.8, n_cycles=5.0,
            difficulty="extreme",
        ),
        noise_bursts=bursts,
    )


def decoy_roll_artifacts() -> ScenarioConfig:
    """Spacecraft roll maneuver artifacts (smooth transverse transients)."""
    centers = _centers(6, 10)
    rolls = [
        RollArtifactSpec(center_hours=c, duration_hours=2.0,
                         rotation_deg=15.0 + i * 5)
        for i, c in enumerate(centers)
    ]
    return ScenarioConfig(
        dataset_id="decoy_roll_artifacts",
        description="Spacecraft roll maneuver artifacts",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=True, difficulty_tier="decoy",
        event_specs=[],  # no injected wave events
        roll_artifacts=rolls,
    )


# ======================================================================
# NEW: Multi-band co-occurrence and realistic scenarios
# ======================================================================


def tier2_qp60_qp120_cooccurrence() -> ScenarioConfig:
    """QP60 + QP120 active simultaneously — the critical missing scenario."""
    centers = _centers(6, 10)
    specs = []
    for c in centers:
        specs.append(EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                               decay_hours=2.0, difficulty="moderate"))
        specs.append(EventSpec(band="QP120", amplitude=1.5, center_hours=c,
                               decay_hours=3.5, difficulty="moderate"))
    return ScenarioConfig(
        dataset_id="tier2_qp60_qp120_cooccurrence",
        description="Simultaneous QP60 + QP120 (tests cross-band rejection)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=specs,
    )


def tier2_qp30_qp60_cooccurrence() -> ScenarioConfig:
    """QP30 + QP60 simultaneous, completing band-pair coverage."""
    centers = _centers(6, 10)
    specs = []
    for c in centers:
        specs.append(EventSpec(band="QP30", amplitude=0.8, center_hours=c,
                               decay_hours=1.5, difficulty="moderate"))
        specs.append(EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                               decay_hours=2.0, difficulty="moderate"))
    return ScenarioConfig(
        dataset_id="tier2_qp30_qp60_cooccurrence",
        description="Simultaneous QP30 + QP60 at tier2 difficulty",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=specs,
    )


def tier2_realistic_background() -> ScenarioConfig:
    """QP60 at catalog-median parameters in full magnetospheric background."""
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier2_realistic_background",
        description="QP60 at catalog-median amplitude in realistic background",
        duration_days=10, noise_alpha=1.3, noise_sigma=0.07,
        background_trend=True, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.6, center_hours=c,
                      decay_hours=2.0, difficulty="moderate")
            for c in centers
        ],
    )


def tier2_qp30_in_continuum() -> ScenarioConfig:
    """QP30 in steep red noise — QP120 band has high background power."""
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier2_qp30_in_continuum",
        description="QP30 in steep noise where QP120 band power is high",
        duration_days=10, noise_alpha=1.5, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP30", amplitude=1.6, center_hours=c,
                      decay_hours=1.5, difficulty="moderate")
            for c in centers
        ],
    )


def tier2_multiband_mixed() -> ScenarioConfig:
    """Mix of isolated and co-occurring events in one dataset."""
    centers = _centers(8, 10)
    specs = []
    # 4 isolated QP60
    for c in centers[:4]:
        specs.append(EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                               decay_hours=2.0, difficulty="moderate"))
    # 4 co-occurring QP60 + QP120 pairs
    for c in centers[4:8]:
        specs.append(EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                               decay_hours=2.0, difficulty="moderate"))
        specs.append(EventSpec(band="QP120", amplitude=1.5, center_hours=c,
                               decay_hours=3.5, difficulty="moderate"))
    return ScenarioConfig(
        dataset_id="tier2_multiband_mixed",
        description="4 isolated QP60 + 2 co-occurring QP60/QP120 pairs",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=specs,
    )


def tier3_triple_band() -> ScenarioConfig:
    """All three QP bands active simultaneously."""
    centers = _centers(3, 10)
    specs = []
    for c in centers:
        specs.append(EventSpec(band="QP30", amplitude=1.0, center_hours=c,
                               decay_hours=1.5, difficulty="hard"))
        specs.append(EventSpec(band="QP60", amplitude=1.5, center_hours=c,
                               decay_hours=2.0, difficulty="hard"))
        specs.append(EventSpec(band="QP120", amplitude=2.0, center_hours=c,
                               decay_hours=3.5, difficulty="hard"))
    return ScenarioConfig(
        dataset_id="tier3_triple_band",
        description="QP30 + QP60 + QP120 simultaneous (real amplitude hierarchy)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=specs,
    )


def tier3_qp120_dominant() -> ScenarioConfig:
    """QP60 where QP120 has 3x higher power — the exact failure pattern."""
    centers = _centers(4, 10)
    specs = []
    for c in centers:
        specs.append(EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                               decay_hours=2.0, difficulty="hard"))
        specs.append(EventSpec(band="QP120", amplitude=2.5, center_hours=c,
                               decay_hours=3.5, difficulty="hard"))
    return ScenarioConfig(
        dataset_id="tier3_qp120_dominant",
        description="QP60 with QP120 at 3x amplitude (spectral conc. test)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=specs,
    )


def tier3_asymmetric_cooccurrence() -> ScenarioConfig:
    """One band fading in while another fades out."""
    centers = _centers(4, 10)
    specs = []
    for c in centers:
        specs.append(EventSpec(band="QP120", amplitude=2.0,
                               center_hours=c - 1.0, decay_hours=3.0,
                               difficulty="hard"))
        specs.append(EventSpec(band="QP60", amplitude=1.5,
                               center_hours=c + 1.0, decay_hours=2.0,
                               difficulty="hard"))
    return ScenarioConfig(
        dataset_id="tier3_asymmetric_cooccurrence",
        description="QP120 fading out as QP60 fades in (transition test)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=specs,
    )


def tier3_catalog_realistic() -> ScenarioConfig:
    """Events drawn from v5 catalog statistics with ~10% co-occurrence."""
    centers = _centers(10, 10)
    specs = []
    bands = ["QP30", "QP60", "QP60", "QP60", "QP120", "QP120", "QP60"]
    amp = {"QP30": 1.6, "QP60": 1.6, "QP120": 2.1}
    decay = {"QP30": 1.5, "QP60": 2.0, "QP120": 3.6}
    for i, c in enumerate(centers[:7]):
        b = bands[i]
        specs.append(EventSpec(band=b, amplitude=amp[b], center_hours=c,
                               decay_hours=decay[b], difficulty="hard"))
    # 1 co-occurring pair (~10%)
    c = centers[8]
    specs.append(EventSpec(band="QP60", amplitude=1.6, center_hours=c,
                           decay_hours=2.0, difficulty="hard"))
    specs.append(EventSpec(band="QP120", amplitude=2.1, center_hours=c,
                           decay_hours=3.6, difficulty="hard"))
    return ScenarioConfig(
        dataset_id="tier3_catalog_realistic",
        description="Events from v5 catalog statistics (~10% co-occurrence)",
        duration_days=10, noise_alpha=1.3, noise_sigma=0.07,
        background_trend=True, difficulty_tier="tier3",
        event_specs=specs,
    )


def tier4_weak_qp60_strong_qp120() -> ScenarioConfig:
    """QP60 near threshold with strong QP120 — worst case for spec. conc."""
    centers = _centers(4, 10)
    specs = []
    for c in centers:
        specs.append(EventSpec(band="QP60", amplitude=0.15, center_hours=c,
                               decay_hours=2.0, difficulty="extreme"))
        specs.append(EventSpec(band="QP120", amplitude=2.5, center_hours=c,
                               decay_hours=3.5, difficulty="extreme"))
    return ScenarioConfig(
        dataset_id="tier4_weak_qp60_strong_qp120",
        description="QP60 at threshold with QP120 at 17x amplitude",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=specs,
    )


def tier4_harmonic_pairs() -> ScenarioConfig:
    """Harmonic-strength sweep across all bands.

    Tests the harmonic suppression filter's power-ratio threshold.
    9 events — 3 fundamentals per band × 3 harmonic strengths
    (0.2, 0.4, 0.6). Periods log-uniform within each band so the 2f
    harmonic also samples a range of locations.

    Harmonic-to-fundamental CWT prominence ratio is ~= harmonic_content
    (linear in amplitude). At h=0.2 the harmonic should be suppressed;
    at h=0.6 it's a borderline case; at h=0.4 it's the design target.
    """
    dataset_id = "tier4_harmonic_pairs"
    rng = _scenario_rng(dataset_id)
    centers = _centers(9, 12)
    strengths = rng.choice([0.2, 0.4, 0.6], 9, replace=True).tolist()
    # 3 per band, shuffled
    bands = ["QP30", "QP60", "QP120"] * 3
    rng.shuffle(bands)
    specs = []
    for c, b, h in zip(centers, bands, strengths):
        p_sec = _sample_periods_in_band(b, 1, rng)[0]
        specs.append(EventSpec(
            band=b, period_sec_override=p_sec,
            amplitude=1.0, center_hours=c,
            decay_hours=5.0 * p_sec / (4.0 * 3600.0),  # ~5 cycles
            harmonic_content=float(h),
            difficulty="extreme",
        ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Harmonic-strength sweep (h=0.2/0.4/0.6) across all bands",
        duration_days=12, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=specs,
    )


def tier4_harmonic_noncanonical() -> ScenarioConfig:
    """Harmonic distortion at non-canonical fundamental periods.

    Fundamentals at 45, 50, 70, 100, 145 min — deliberately NOT at
    30/60/120. Tests that the harmonic filter works on fundamental
    periods the paper's canonical bands don't centre on:

      45 min → 2f at 22.5 min (QP30 lower edge)
      50 min → 2f at 25.0 min (QP30)
      70 min → 2f at 35.0 min (QP30)
      100 min → 2f at 50.0 min (QP60)
      145 min → 2f at 72.5 min (QP60)

    If the harmonic suppression is truly period-ratio-based (not
    band-pair-based) it should handle all these.
    """
    dataset_id = "tier4_harmonic_noncanonical"
    rng = _scenario_rng(dataset_id)
    centers = _centers(10, 12)
    # Two events per fundamental period
    fundamentals_min = [45.0, 50.0, 70.0, 100.0, 145.0] * 2
    amps = rng.uniform(0.8, 1.2, 10).tolist()
    specs = []
    for c, p_min, a in zip(centers, fundamentals_min, amps):
        p_sec = p_min * _MIN
        band = classify_period(p_sec) or "QP60"
        specs.append(EventSpec(
            band=band, period_sec_override=p_sec,
            amplitude=float(a), center_hours=c,
            decay_hours=5.0 * p_sec / (4.0 * 3600.0),
            harmonic_content=0.4,
            difficulty="extreme",
        ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Harmonics at non-canonical fundamentals (45/50/70/100/145 min)",
        duration_days=12, noise_alpha=1.0, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=specs,
    )


def decoy_red_noise_qp120() -> ScenarioConfig:
    """Steep red noise that looks like QP120 in CWT — no real events."""
    return ScenarioConfig(
        dataset_id="decoy_red_noise_qp120",
        description="Steep noise (alpha=1.7) mimicking QP120 in CWT",
        duration_days=10, noise_alpha=1.7, noise_sigma=0.08,
        background_trend=False, ppo_amplitude=0.7, difficulty_tier="decoy",
        event_specs=[],
    )


def decoy_broadband_redslope() -> ScenarioConfig:
    """Noise burst spanning QP60+QP120 with red slope."""
    from qp.events.bands import QP_BANDS as _bands
    centers = _centers(4, 10)
    return ScenarioConfig(
        dataset_id="decoy_broadband_redslope",
        description="Broadband red-slope bursts spanning QP60+QP120",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="decoy",
        event_specs=[
            EventSpec(
                band="QP60", amplitude=1.5, center_hours=c,
                decay_hours=3.0, should_detect=False, difficulty="easy",
                event_type="decoy_broadband_red",
                injection_type="noise_burst",
                burst_freq_lo=_bands["QP120"].freq_min_hz,
                burst_freq_hi=_bands["QP60"].freq_max_hz,
                burst_alpha=1.5,
            )
            for c in centers
        ],
    )


def decoy_ppo_harmonic() -> ScenarioConfig:
    """Enhanced PPO whose beat modulation mimics wave packets."""
    return ScenarioConfig(
        dataset_id="decoy_ppo_harmonic",
        description="Strong PPO (1.0 nT) beat modulation mimicking QP events",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=1.0, difficulty_tier="decoy",
        event_specs=[],
    )


# ======================================================================
# v3 hardening — continuous period coverage, short packets, broader decoys
# ======================================================================

# --- Short-packet scenarios (stress-test min_oscillations) ---------------

def _short_packet_decay_hours(period_min: float, n_cycles: float) -> float:
    r"""Gaussian σ that yields ``n_cycles`` full cycles between ±2σ.

    Duration(±2σ) = 4·σ → n_cycles = Duration / P → σ = n_cycles·P/4.
    """
    return n_cycles * (period_min / 60.0) / 4.0


def tier3_short_packets_qp60() -> ScenarioConfig:
    """QP60 packets with 2.5–4.5 cycles spanning detection threshold."""
    dataset_id = "tier3_short_packets_qp60"
    centers = _centers(8, 10)
    rng = _scenario_rng(dataset_id)
    periods = _sample_periods_in_band("QP60", 8, rng)
    # Paired: half above (≥3 cycles should detect), half below (fail osc test)
    cycles_values = [2.5, 2.8, 3.0, 3.2, 3.5, 4.0, 4.2, 4.5]
    amps = rng.uniform(0.8, 1.4, 8).tolist()
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="QP60 packets at 2.5–4.5 cycles (tests osc threshold)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(
                band="QP60", period_sec_override=p,
                amplitude=a, center_hours=c,
                decay_hours=_short_packet_decay_hours(p / _MIN, k),
                difficulty="hard",
            )
            for c, p, k, a in zip(centers, periods, cycles_values, amps)
        ],
    )


def tier3_short_packets_qp120() -> ScenarioConfig:
    """QP120 packets with 2.5–4.5 cycles — the hardest band for cycle count."""
    dataset_id = "tier3_short_packets_qp120"
    centers = _centers(8, 10)
    rng = _scenario_rng(dataset_id)
    periods = _sample_periods_in_band("QP120", 8, rng)
    cycles_values = [2.5, 2.8, 3.0, 3.2, 3.5, 4.0, 4.2, 4.5]
    amps = rng.uniform(1.2, 2.4, 8).tolist()
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="QP120 packets at 2.5–4.5 cycles (tests osc threshold)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(
                band="QP120", period_sec_override=p,
                amplitude=a, center_hours=c,
                decay_hours=_short_packet_decay_hours(p / _MIN, k),
                difficulty="hard",
            )
            for c, p, k, a in zip(centers, periods, cycles_values, amps)
        ],
    )


def tier4_minimal_cycles_multiband() -> ScenarioConfig:
    """2–3 cycle packets across all three bands in α=1.5 colored noise."""
    dataset_id = "tier4_minimal_cycles_multiband"
    rng = _scenario_rng(dataset_id)
    centers = _centers(12, 12)
    bands = ["QP30", "QP60", "QP120"] * 4
    # 2–3 cycle range: most should fail the min_oscillations=3 cutoff
    cycles = rng.uniform(2.0, 3.0, 12)
    amps = rng.uniform(1.0, 2.0, 12)
    specs = []
    for c, b, k, a in zip(centers, bands, cycles, amps):
        p_sec = _sample_periods_in_band(b, 1, rng)[0]
        specs.append(EventSpec(
            band=b, period_sec_override=p_sec,
            amplitude=float(a), center_hours=c,
            decay_hours=_short_packet_decay_hours(p_sec / _MIN, float(k)),
            difficulty="extreme",
        ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="12 packets at 2–3 cycles across QP30/60/120 in α=1.5 noise",
        duration_days=12, noise_alpha=1.5, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=specs,
    )


# --- QP60/QP120-heavy scenarios (rebalance band coverage) ---------------

def tier3_qp60_continuous() -> ScenarioConfig:
    """10 QP60 events log-uniform in 45–80 min, log-normal amplitudes."""
    dataset_id = "tier3_qp60_continuous"
    rng = _scenario_rng(dataset_id)
    centers = _centers(10, 12)
    periods = _sample_periods_in_band("QP60", 10, rng)
    # Log-normal amplitudes matching Cassini catalog spread (~1 nT median)
    amps = rng.lognormal(mean=0.0, sigma=0.35, size=10).tolist()
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="10 QP60 events, continuous periods + log-normal amps",
        duration_days=12, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", period_sec_override=p,
                      amplitude=float(a), center_hours=c,
                      decay_hours=3.0, difficulty="hard")
            for c, p, a in zip(centers, periods, amps)
        ],
    )


def tier3_qp120_continuous() -> ScenarioConfig:
    """10 QP120 events log-uniform in 90–150 min, log-normal amplitudes."""
    dataset_id = "tier3_qp120_continuous"
    rng = _scenario_rng(dataset_id)
    centers = _centers(10, 15)  # Longer to accommodate longer packets
    periods = _sample_periods_in_band("QP120", 10, rng)
    amps = rng.lognormal(mean=0.2, sigma=0.35, size=10).tolist()
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="10 QP120 events, continuous periods + log-normal amps",
        duration_days=15, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP120", period_sec_override=p,
                      amplitude=float(a), center_hours=c,
                      decay_hours=5.0, difficulty="hard")
            for c, p, a in zip(centers, periods, amps)
        ],
    )


def tier4_qp120_weak_long() -> ScenarioConfig:
    """Long-duration low-amplitude QP120 buried in α=1.5 noise."""
    dataset_id = "tier4_qp120_weak_long"
    rng = _scenario_rng(dataset_id)
    centers = _centers(6, 12)
    periods = _sample_periods_in_band("QP120", 6, rng)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="6 weak QP120 packets in α=1.5 colored noise",
        duration_days=12, noise_alpha=1.5, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP120", period_sec_override=p,
                      amplitude=0.5, center_hours=c,
                      decay_hours=5.0, difficulty="extreme")
            for c, p in zip(centers, periods)
        ],
    )


# --- Out-of-band decoys (should_detect=False, test edge-veto) -----------

def tier4_qp60_qp120_boundary() -> ScenarioConfig:
    """Waves at 80–90 min — straddles QP60/QP120 boundary (85 min)."""
    dataset_id = "tier4_qp60_qp120_boundary"
    rng = _scenario_rng(dataset_id)
    centers = _centers(6, 10)
    periods = [80.5, 82.0, 84.0, 86.0, 88.0, 89.5]
    amps = rng.uniform(0.8, 1.4, 6).tolist()
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Waves at 80–90 min (QP60/QP120 boundary region)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band=classify_period(p * _MIN) or "QP60",
                      period_sec_override=p * _MIN,
                      amplitude=float(a), center_hours=c, decay_hours=4.0,
                      should_detect=True, difficulty="extreme",
                      event_type="qp_wave_boundary")
            for p, c, a in zip(periods, centers, amps)
        ],
    )


def tier4_super_qp120() -> ScenarioConfig:
    """Waves at 155–180 min — spans QP120 upper (169.7) + super_qp120."""
    dataset_id = "tier4_super_qp120"
    rng = _scenario_rng(dataset_id)
    centers = _centers(6, 12)
    periods = [155.0, 160.0, 165.0, 170.0, 175.0, 180.0]
    amps = rng.uniform(1.0, 1.8, 6).tolist()
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Waves at 155–180 min (QP120 upper edge + super_qp120)",
        duration_days=12, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band=classify_period(p * _MIN) or "QP120",
                      period_sec_override=p * _MIN,
                      amplitude=float(a), center_hours=c, decay_hours=6.0,
                      should_detect=True, difficulty="extreme",
                      event_type="qp_wave_long_period")
            for p, c, a in zip(periods, centers, amps)
        ],
    )


def tier4_sub_qp30() -> ScenarioConfig:
    """Waves at 15–19 min — sub_qp30 label (inside search window)."""
    dataset_id = "tier4_sub_qp30"
    rng = _scenario_rng(dataset_id)
    centers = _centers(6, 10)
    periods = [15.0, 16.0, 17.0, 18.0, 18.5, 19.0]
    amps = rng.uniform(0.6, 1.2, 6).tolist()
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Waves at 15–19 min (sub_qp30: below QP30, within search)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band=classify_period(p * _MIN) or "QP30",
                      period_sec_override=p * _MIN,
                      amplitude=float(a), center_hours=c, decay_hours=1.0,
                      should_detect=True, difficulty="extreme",
                      event_type="qp_wave_short_period")
            for p, c, a in zip(periods, centers, amps)
        ],
    )


# --- Continuous-spectrum scenario ---------------------------------------

def tier3_continuous_spectrum() -> ScenarioConfig:
    """24 events log-uniform in 20–240 min — natural cluster test.

    About half fall in QP bands (``should_detect=True`` via
    :func:`classify_period`), half outside. A general detector should
    cleanly separate the two populations without band-specific tuning.
    """
    dataset_id = "tier3_continuous_spectrum"
    rng = _scenario_rng(dataset_id)
    centers = _centers(24, 24)  # spread over 24 days
    log_lo = math.log(20 * _MIN)
    log_hi = math.log(240 * _MIN)
    periods = [math.exp(x) for x in rng.uniform(log_lo, log_hi, 24)]
    amps = rng.uniform(1.2, 2.2, 24).tolist()
    # Resample so all periods fall inside the search window [15, 180] min
    # Resampling ensures they're all should_detect=True under the new
    # band-agnostic semantics.
    periods = [min(max(p, 15 * _MIN), 179 * _MIN) for p in periods]
    specs = []
    for c, p, a in zip(centers, periods, amps):
        band_name = classify_period(p) or "QP60"
        decay_h = 6.0 * (p / _MIN) / 60.0 / 4.0 * 2.0  # ≈ 0.75·P_h
        specs.append(EventSpec(
            band=band_name, period_sec_override=p,
            amplitude=float(a), center_hours=c,
            decay_hours=max(1.5, decay_h),
            should_detect=True,
            difficulty="hard",
            event_type="continuous_spectrum",
        ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="24 events log-uniform in 15-180 min (all detectable)",
        duration_days=24, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=specs,
    )


# ======================================================================
# Round-3 hardening scenarios — defensible-paper additions
# ======================================================================

# --- Item 3: Anisotropy sensitivity ---

def tier3_isotropic_noise() -> ScenarioConfig:
    """8 multiband packets in noise where σ_par = σ_perp.

    The default magnetospheric background uses σ_par = σ_perp/2,
    matching von Papen 2014. A detector whose transverse-ratio gate
    rides on this prior would over-reject Alfvénic packets in
    isotropic noise. This scenario exposes that dependency.
    """
    dataset_id = "tier3_isotropic_noise"
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="8 multiband packets in isotropic colored noise (σ_par = σ_perp)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=True, isotropic_noise=True,
        difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.6, n_cycles=5.0,
            difficulty="hard", noise_alpha=1.2, noise_sigma=0.07,
        ),
    )


def tier3_extreme_anisotropy() -> ScenarioConfig:
    """8 multiband packets where σ_par = σ_perp / 4.

    Pushes the parallel noise floor down by a factor 4 from the
    nominal σ_perp/2. The transverse-ratio filter must still pass
    Alfvénic events — a too-loose ratio threshold would let
    compressional false positives through.
    """
    dataset_id = "tier3_extreme_anisotropy"
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="8 multiband packets in extreme-anisotropy noise (σ_par = σ_perp/4)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=True, anisotropy_par_ratio=0.25,
        difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.6, n_cycles=5.0,
            difficulty="hard", noise_alpha=1.2, noise_sigma=0.07,
        ),
    )


# --- Item 5: Non-Gaussian envelope shapes ---

def tier3_lognormal_envelope() -> ScenarioConfig:
    """8 multiband packets with log-normal envelopes (heavy-tailed decay).

    Real packets often grow rapidly out of a noise floor and decay
    slowly; the Gaussian-only training of a detector that trims by
    fixed-power threshold may bias toward packets with symmetric
    decay. This scenario forces the detector to handle skewed
    envelopes without leaning on a Gaussian prior.
    """
    dataset_id = "tier3_lognormal_envelope"
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="8 multiband packets with log-normal envelopes",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=True, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.6, n_cycles=5.0,
            difficulty="hard", noise_alpha=1.2, noise_sigma=0.05,
            envelope_shape="lognormal",
        ),
    )


def tier3_rayleigh_envelope() -> ScenarioConfig:
    """8 multiband packets with Rayleigh-shaped envelopes (sharp rise)."""
    dataset_id = "tier3_rayleigh_envelope"
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="8 multiband packets with Rayleigh envelopes (sharp rise, slow fall)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=True, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.6, n_cycles=5.0,
            difficulty="hard", noise_alpha=1.2, noise_sigma=0.05,
            envelope_shape="rayleigh",
        ),
    )


# --- Item 1: Heavy-tailed and regime-switching noise ---

def tier3_heavy_tail_noise() -> ScenarioConfig:
    """8 multiband packets in Student-t (df=5) heavy-tailed colored noise.

    Real magnetometer residuals show occasional rare-event spikes
    (CME shocks, current-sheet crossings) that the Gaussian
    Timmer–König model under-represents. df=5 gives kurtosis 9 vs
    Gaussian 3 — large excursions are 2–4× more frequent.
    """
    dataset_id = "tier3_heavy_tail_noise"
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="8 multiband packets in heavy-tailed (Student-t df=5) noise",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.06,
        background_trend=True, noise_tail_df=5.0,
        difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.6, n_cycles=5.0,
            difficulty="hard", noise_alpha=1.2, noise_sigma=0.06,
        ),
    )


def tier3_regime_switching_noise() -> ScenarioConfig:
    """10 multiband packets in piecewise-stationary α(t)∈[1.0, 1.7] noise.

    Tests robustness to non-stationary spectral slope on 6–12 h
    blocks — the actual mode of variability in Cassini residuals.
    """
    dataset_id = "tier3_regime_switching_noise"
    centers = _centers(10, 14)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="10 multiband packets in regime-switching α∈[1.0,1.7] noise",
        duration_days=14, noise_alpha=1.3, noise_sigma=0.06,
        background_trend=False,
        noise_regime_switching=True,
        noise_alpha_range=(1.0, 1.7),
        noise_segment_hours_range=(6.0, 12.0),
        ppo_amplitude=0.4, difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.7, n_cycles=5.0,
            difficulty="hard", noise_alpha=1.3, noise_sigma=0.06,
        ),
    )


# --- Item 2: Realistic PPO ---

def tier3_realistic_ppo() -> ScenarioConfig:
    """8 multiband packets atop log-normal-modulated, drifting PPO.

    Real PPO is not a clean dual sinusoid — it has cycle-to-cycle
    amplitude scatter, slow period drift, and occasional phase
    slips. A detector that rejects clean 10.7 h tones may still
    over-reject realistic PPO; this scenario discriminates that case.
    """
    dataset_id = "tier3_realistic_ppo"
    centers = _centers(8, 14)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="8 multiband packets atop log-normal/drifting/phase-slipping PPO",
        duration_days=14, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=True, ppo_amplitude=0.7,
        realistic_ppo=True,
        difficulty_tier="tier3",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.6, n_cycles=5.0,
            difficulty="hard", noise_alpha=1.2, noise_sigma=0.05,
        ),
    )


# --- Item 7: Envelope-correlation gate calibration sweep + true sawtooth harmonic ---

def tier4_envcorr_sweep() -> ScenarioConfig:
    """QP60+QP120 pairs at Δ-centre = {0, 0.25σ, 0.5σ, σ}.

    Calibration scenario: sweeps the time-offset between matched
    co-occurring QP60 and QP120 packets so that the harmonic
    suppressor's envelope-correlation gate is exercised across the
    ambiguity boundary (env_corr ≈ 0.95). Reports the gate as a
    ROC-style sweep, not a single point.
    """
    dataset_id = "tier4_envcorr_sweep"
    rng = _scenario_rng(dataset_id)
    duration_days = 18
    # Four offsets × 3 pairs each, evenly spaced, with σ derived from
    # QP60 5-cycle envelope: σ_60 ≈ 5·60min/4 = 75 min ≈ 1.25 h.
    sigma_h = 1.25
    offsets = [0.0, 0.25, 0.5, 1.0]
    n_pairs_per_offset = 3
    centers = _centers(
        len(offsets) * n_pairs_per_offset, duration_days,
        margin_h=24, jitter_frac=0.2,
    )
    specs: list[EventSpec] = []
    idx = 0
    for off in offsets:
        for _ in range(n_pairs_per_offset):
            c = centers[idx]
            idx += 1
            p60 = float(_sample_periods_in_band("QP60", 1, rng)[0])
            p120 = 2.0 * p60  # exact 2:1 to challenge the gate
            decay_h = 5.0 * p60 / (4.0 * 3600.0)
            # QP60 at c; QP120 at c + off·σ
            specs.append(EventSpec(
                band="QP60", period_sec_override=p60,
                amplitude=1.6, center_hours=c,
                decay_hours=decay_h, difficulty="extreme",
                event_type="envcorr_sweep",
            ))
            specs.append(EventSpec(
                band="QP120", period_sec_override=p120,
                amplitude=1.6, center_hours=c + off * sigma_h,
                decay_hours=decay_h * 2.0,  # QP120 packet roughly 2× longer
                difficulty="extreme",
                event_type="envcorr_sweep",
            ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description=(
            "Calibration: matched QP60+QP120 pairs at Δ=0, 0.25σ, 0.5σ, σ — "
            "exposes env-corr gate as ROC, not a single threshold"
        ),
        duration_days=duration_days, noise_alpha=1.3, noise_sigma=0.05,
        background_trend=True, difficulty_tier="tier4",
        event_specs=specs,
    )


def tier4_phase_locked_sawtooth() -> ScenarioConfig:
    """8 QP60 packets steepened to sawtooth shape via Fourier truncation.

    Uses ``harmonic_model='sawtooth_truncated'`` with
    ``harmonic_content=0.6`` so the second harmonic is phase-locked
    to the fundamental — the *real* steepened-wave behaviour the
    harmonic suppressor is meant to detect, vs. the random-phase
    nuisance harmonic of the existing ``tier4_harmonic_pairs``
    scenario. Detector should keep all QP60 fundamentals and reject
    the QP30 ‘children’ at 2f.
    """
    dataset_id = "tier4_phase_locked_sawtooth"
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="8 QP60 packets with phase-locked sawtooth harmonics (true MHD steepening)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=True, difficulty_tier="tier4",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.8, n_cycles=5.0,
            difficulty="extreme", noise_alpha=1.2, noise_sigma=0.05,
            harmonic_content=0.6,
            harmonic_model="sawtooth_truncated",
        ),
    )


# --- Item 10: Empty-null specificity scenarios ---

def decoy_null_alpha_white() -> ScenarioConfig:
    """10-day pure colored background at α=0.8 with no injections."""
    return ScenarioConfig(
        dataset_id="decoy_null_alpha_white",
        description="Pure α=0.8 background, 10 d, no injections (FP/day measure)",
        duration_days=10, noise_alpha=0.8, noise_sigma=0.06,
        background_trend=False, ppo_amplitude=0.0,
        difficulty_tier="decoy", event_specs=[],
    )


def decoy_null_alpha_pink() -> ScenarioConfig:
    """10-day pure colored background at α=1.2 (Cassini median) with no injections."""
    return ScenarioConfig(
        dataset_id="decoy_null_alpha_pink",
        description="Pure α=1.2 background + PPO, 10 d, no injections",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.06,
        background_trend=True, ppo_amplitude=0.5,
        difficulty_tier="decoy", event_specs=[],
    )


def decoy_null_alpha_steep() -> ScenarioConfig:
    """10-day pure colored background at α=1.7 (steep / Kolmogorov-ish)."""
    return ScenarioConfig(
        dataset_id="decoy_null_alpha_steep",
        description="Pure α=1.7 background, 10 d, no injections (worst-case red noise)",
        duration_days=10, noise_alpha=1.7, noise_sigma=0.06,
        background_trend=False, ppo_amplitude=0.4,
        difficulty_tier="decoy", event_specs=[],
    )


# --- Item 12: Out-of-distribution holdout family ---

def holdout_unseen_regime() -> ScenarioConfig:
    """Held-out: 4-cycle, ellipticity ∈ [-0.3, 0.3], α=0.5, lognormal env.

    Combination of parameters that no tuning scenario exposes the
    detector to. If composite is stable when this scenario is
    included, the detector generalises beyond the calibration set.
    """
    dataset_id = "holdout_unseen_regime"
    rng = _scenario_rng(dataset_id)
    centers = _centers(8, 12)
    bands = _balanced_band_assignment(8, rng)
    specs: list[EventSpec] = []
    for c, b in zip(centers, bands):
        p_sec = float(_sample_periods_in_band(b, 1, rng)[0])
        decay_h = 4.0 * p_sec / (4.0 * 3600.0)  # exactly 4 cycles
        ell = float(rng.uniform(-0.3, 0.3))
        specs.append(EventSpec(
            band=b, period_sec_override=p_sec,
            amplitude=1.7, center_hours=c, decay_hours=decay_h,
            difficulty="hard", ellipticity=ell,
            polarization="elliptical",
            envelope_shape="lognormal",
            event_type="holdout",
        ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Holdout: 4-cycle, near-linear pol, α=0.5 noise, lognormal env",
        duration_days=12, noise_alpha=0.5, noise_sigma=0.06,
        background_trend=False, ppo_amplitude=0.5,
        difficulty_tier="holdout", event_specs=specs,
    )


def holdout_band_edges() -> ScenarioConfig:
    """Held-out: events at QP-band edges with mixed harmonic models."""
    dataset_id = "holdout_band_edges"
    rng = _scenario_rng(dataset_id)
    centers = _centers(9, 14)
    # Three events per band, periods near band edges
    period_picks: list[tuple[str, float]] = []
    for band_name in ("QP30", "QP60", "QP120"):
        b = QP_BANDS[band_name]
        for f in (0.05, 0.5, 0.95):
            p = math.exp(
                math.log(b.period_min_sec) +
                f * (math.log(b.period_max_sec) - math.log(b.period_min_sec))
            )
            period_picks.append((band_name, p))
    rng.shuffle(period_picks)
    specs: list[EventSpec] = []
    for c, (band_name, p_sec) in zip(centers, period_picks):
        decay_h = 5.0 * p_sec / (4.0 * 3600.0)
        # Mix harmonic models — half phase-locked steepening, half random
        harm_model = "sawtooth_truncated" if rng.uniform() < 0.5 else "linear_2f"
        specs.append(EventSpec(
            band=band_name, period_sec_override=p_sec,
            amplitude=1.6, center_hours=c, decay_hours=decay_h,
            difficulty="hard",
            harmonic_content=float(rng.uniform(0.0, 0.4)),
            harmonic_model=harm_model,
            envelope_shape="rayleigh",
            event_type="holdout",
        ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="Holdout: band-edge periods, mixed harmonic models, Rayleigh envelopes",
        duration_days=14, noise_alpha=1.4, noise_sigma=0.06,
        background_trend=True, difficulty_tier="holdout", event_specs=specs,
    )


def holdout_realistic_ppo_packets() -> ScenarioConfig:
    """Held-out: catalog-amplitude packets atop realistic PPO + heavy-tailed noise."""
    dataset_id = "holdout_realistic_ppo_packets"
    centers = _centers(10, 16)
    return ScenarioConfig(
        dataset_id=dataset_id,
        description=(
            "Holdout: catalog-amplitude multiband packets atop realistic PPO + "
            "Student-t (df=6) heavy-tailed noise"
        ),
        duration_days=16, noise_alpha=1.4, noise_sigma=0.05,
        background_trend=True, ppo_amplitude=0.7,
        realistic_ppo=True, noise_tail_df=6.0,
        difficulty_tier="holdout",
        event_specs=_multiband_specs(
            dataset_id, centers, amplitude=1.5, n_cycles=5.0,
            n_cycles_sigma=0.35,
            difficulty="hard", noise_alpha=1.4, noise_sigma=0.05,
        ),
    )


# ======================================================================
# Diagnostic scenarios — reported separately, NOT in composite score
# ======================================================================

def diag_null_pure_noise() -> ScenarioConfig:
    """Pure colored noise + PPO + background trend. Zero wave injections.

    A bias-free detector must produce 0 detections here. Any output
    is a false-positive baseline — the ceiling on catalog purity on
    real Cassini data.
    """
    return ScenarioConfig(
        dataset_id="diag_null_pure_noise",
        description="Pure α=1.2 colored noise + PPO, no waves (0 expected)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=True, ppo_amplitude=0.5, difficulty_tier="diagnostic",
        event_specs=[],
    )


def diag_uniform_period_sweep() -> ScenarioConfig:
    """30 events log-uniform in 15-180 min, all detectable.

    Exposes residual period bias in the detector: the detection-rate
    histogram over period should be roughly flat (weighted by the
    distribution of injected periods). Any clustering at 30/60/120
    min *in the output* would indicate band-locked bias despite the
    band-agnostic architecture.
    """
    dataset_id = "diag_uniform_period_sweep"
    rng = _scenario_rng(dataset_id)
    centers = _centers(30, 30)
    log_lo = math.log(15 * _MIN)
    log_hi = math.log(180 * _MIN)
    periods = [math.exp(x) for x in rng.uniform(log_lo, log_hi, 30)]
    amps = rng.uniform(1.2, 2.2, 30).tolist()
    specs = []
    for c, p, a in zip(centers, periods, amps):
        band_name = classify_period(p) or "QP60"
        # ~6-cycle packets regardless of period
        decay_h = 6.0 * p / (4.0 * 3600.0)
        specs.append(EventSpec(
            band=band_name, period_sec_override=p,
            amplitude=float(a), center_hours=c,
            decay_hours=max(0.5, decay_h),
            should_detect=True, difficulty="hard",
            event_type="diagnostic_uniform",
        ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="30 log-uniform periods in 15-180 min (flat distribution test)",
        duration_days=30, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="diagnostic",
        event_specs=specs,
    )


def diag_non_canonical_clusters() -> ScenarioConfig:
    """Events clustered at 22, 50, 100 min — non-canonical periods.

    Old band-locked detector would miss these architecturally. The
    band-agnostic detector should find them. Their detection is
    the empirical evidence that non-30/60/120 periodicities can be
    recovered if they exist in real data.
    """
    dataset_id = "diag_non_canonical_clusters"
    rng = _scenario_rng(dataset_id)
    # 3 clusters of 8 events each at 22, 50, 100 min
    centers_per_cluster = _centers(8, 30)
    # Offset each cluster in time so they don't overlap
    all_centers = []
    all_periods = []
    for offset_day, cluster_p in enumerate([22.0, 50.0, 100.0]):
        for c in centers_per_cluster:
            all_centers.append(c + offset_day * 30.0 * 24 / 3)
            jitter = rng.uniform(-0.5, 0.5)  # ±0.5 min jitter
            all_periods.append((cluster_p + jitter) * _MIN)
    amps = rng.uniform(1.5, 2.0, len(all_centers)).tolist()
    specs = []
    for c, p, a in zip(all_centers, all_periods, amps):
        band_name = classify_period(p) or "QP60"
        decay_h = 6.0 * p / (4.0 * 3600.0)
        specs.append(EventSpec(
            band=band_name, period_sec_override=p,
            amplitude=float(a), center_hours=c,
            decay_hours=max(0.5, decay_h),
            should_detect=True, difficulty="hard",
            event_type="diagnostic_noncanonical",
        ))
    return ScenarioConfig(
        dataset_id=dataset_id,
        description="24 events clustered at 22, 50, 100 min (non-canonical)",
        duration_days=30, noise_alpha=1.2, noise_sigma=0.07,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="diagnostic",
        event_specs=specs,
    )


# ======================================================================
# Registry
# ======================================================================

ALL_SCENARIOS: dict[str, callable] = {
    # Tier 1
    "tier1_clean_qp30": tier1_clean_qp30,
    "tier1_clean_qp60": tier1_clean_qp60,
    "tier1_clean_qp120": tier1_clean_qp120,
    "tier1_mixed_bands": tier1_mixed_bands,
    "tier1_full_spectrum": tier1_full_spectrum,
    # Tier 2
    "tier2_colored_noise": tier2_colored_noise,
    "tier2_low_amplitude": tier2_low_amplitude,
    "tier2_sawtooth_shapes": tier2_sawtooth_shapes,
    "tier2_linear_pol": tier2_linear_pol,
    "tier2_short_packets": tier2_short_packets,
    "tier2_ppo_background": tier2_ppo_background,
    # Tier 3
    "tier3_overlapping_bands": tier3_overlapping_bands,
    "tier3_frequency_drift": tier3_frequency_drift,
    "tier3_asymmetric_envelope": tier3_asymmetric_envelope,
    "tier3_amplitude_jitter": tier3_amplitude_jitter,
    "tier3_near_threshold": tier3_near_threshold,
    "tier3_mixed_waveforms": tier3_mixed_waveforms,
    "tier3_travelling_waves": tier3_travelling_waves,
    "tier3_gaps_at_onset": tier3_gaps_at_onset,
    "tier3_gaps_midpacket": tier3_gaps_midpacket,
    "tier3_steep_noise": tier3_steep_noise,
    "tier3_ppo_beat": tier3_ppo_beat,
    "tier3_plasma_sheet": tier3_plasma_sheet,
    # Tier 4
    "tier4_buried_in_noise": tier4_buried_in_noise,
    "tier4_qp120_short": tier4_qp120_short,
    "tier4_inter_band_period": tier4_inter_band_period,
    "tier4_extreme_chirp": tier4_extreme_chirp,
    "tier4_decaying": tier4_decaying,
    "tier4_incoherent": tier4_incoherent,
    "tier4_harmonic_contamination": tier4_harmonic_contamination,
    "tier4_elliptical_pol": tier4_elliptical_pol,
    "tier4_heavy_gaps": tier4_heavy_gaps,
    "tier4_kolmogorov_noise": tier4_kolmogorov_noise,
    "tier4_nonstationary": tier4_nonstationary,
    # Decoys
    "decoy_compressional": decoy_compressional,
    "decoy_single_pulses": decoy_single_pulses,
    "decoy_broadband_burst": decoy_broadband_burst,
    "decoy_step_functions": decoy_step_functions,
    "decoy_ppo_only": decoy_ppo_only,
    "decoy_roll_artifacts": decoy_roll_artifacts,
    # New: multi-band co-occurrence scenarios
    "tier2_qp60_qp120_cooccurrence": tier2_qp60_qp120_cooccurrence,
    "tier2_qp30_qp60_cooccurrence": tier2_qp30_qp60_cooccurrence,
    "tier2_realistic_background": tier2_realistic_background,
    "tier2_qp30_in_continuum": tier2_qp30_in_continuum,
    "tier2_multiband_mixed": tier2_multiband_mixed,
    "tier3_triple_band": tier3_triple_band,
    "tier3_qp120_dominant": tier3_qp120_dominant,
    "tier3_asymmetric_cooccurrence": tier3_asymmetric_cooccurrence,
    "tier3_catalog_realistic": tier3_catalog_realistic,
    "tier4_weak_qp60_strong_qp120": tier4_weak_qp60_strong_qp120,
    "tier4_harmonic_pairs": tier4_harmonic_pairs,
    "tier4_harmonic_noncanonical": tier4_harmonic_noncanonical,
    "decoy_red_noise_qp120": decoy_red_noise_qp120,
    "decoy_broadband_redslope": decoy_broadband_redslope,
    "decoy_ppo_harmonic": decoy_ppo_harmonic,
    # v3 hardening — continuous-period QP60/QP120 coverage
    "tier3_qp60_continuous": tier3_qp60_continuous,
    "tier3_qp120_continuous": tier3_qp120_continuous,
    "tier4_qp120_weak_long": tier4_qp120_weak_long,
    # v3 hardening — short-packet (min_oscillations boundary) stress tests
    "tier3_short_packets_qp60": tier3_short_packets_qp60,
    "tier3_short_packets_qp120": tier3_short_packets_qp120,
    "tier4_minimal_cycles_multiband": tier4_minimal_cycles_multiband,
    # v3 hardening — out-of-band decoys (edge-veto stress tests)
    "tier4_qp60_qp120_boundary": tier4_qp60_qp120_boundary,
    "tier4_super_qp120": tier4_super_qp120,
    "tier4_sub_qp30": tier4_sub_qp30,
    # v3 hardening — continuous-spectrum population test
    "tier3_continuous_spectrum": tier3_continuous_spectrum,
    # v4 band-agnostic — diagnostic scenarios reported separately
    "diag_null_pure_noise": diag_null_pure_noise,
    "diag_uniform_period_sweep": diag_uniform_period_sweep,
    "diag_non_canonical_clusters": diag_non_canonical_clusters,
    # v5 hardening (round-3 statistical review) — included in composite
    "tier3_isotropic_noise": tier3_isotropic_noise,
    "tier3_extreme_anisotropy": tier3_extreme_anisotropy,
    "tier3_lognormal_envelope": tier3_lognormal_envelope,
    "tier3_rayleigh_envelope": tier3_rayleigh_envelope,
    "tier3_heavy_tail_noise": tier3_heavy_tail_noise,
    "tier3_regime_switching_noise": tier3_regime_switching_noise,
    "tier3_realistic_ppo": tier3_realistic_ppo,
    "tier4_envcorr_sweep": tier4_envcorr_sweep,
    "tier4_phase_locked_sawtooth": tier4_phase_locked_sawtooth,
    "decoy_null_alpha_white": decoy_null_alpha_white,
    "decoy_null_alpha_pink": decoy_null_alpha_pink,
    "decoy_null_alpha_steep": decoy_null_alpha_steep,
    # v5 holdout — excluded from composite by default (see HOLDOUT_SCENARIOS)
    "holdout_unseen_regime": holdout_unseen_regime,
    "holdout_band_edges": holdout_band_edges,
    "holdout_realistic_ppo_packets": holdout_realistic_ppo_packets,
}

#: Diagnostic scenarios are excluded from the composite score.
#: They characterize the detector on edge cases (null, uniform,
#: non-canonical clusters) and their results appear as per-scenario
#: rows in the breakdown, not as contributions to the headline metric.
DIAGNOSTIC_SCENARIOS: frozenset[str] = frozenset({
    "diag_null_pure_noise",
    "diag_uniform_period_sweep",
    "diag_non_canonical_clusters",
})

#: Holdout scenarios are also excluded from the default composite —
#: they exist to verify generalisation. The intended workflow is:
#: tune the detector on the main 40+ scenario set, then check the
#: holdout suite separately. Including them in the headline number
#: contaminates that test.
HOLDOUT_SCENARIOS: frozenset[str] = frozenset({
    "holdout_unseen_regime",
    "holdout_band_edges",
    "holdout_realistic_ppo_packets",
})

TIER_SCENARIOS: dict[str, list[str]] = {
    "tier1": [k for k in ALL_SCENARIOS if k.startswith("tier1")],
    "tier2": [k for k in ALL_SCENARIOS if k.startswith("tier2")],
    "tier3": [k for k in ALL_SCENARIOS if k.startswith("tier3")],
    "tier4": [k for k in ALL_SCENARIOS if k.startswith("tier4")],
    "decoy": [k for k in ALL_SCENARIOS if k.startswith("decoy")],
    "holdout": sorted(HOLDOUT_SCENARIOS),
    "diagnostic": sorted(DIAGNOSTIC_SCENARIOS),
}
