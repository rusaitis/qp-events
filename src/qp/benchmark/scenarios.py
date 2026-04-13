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

from qp.benchmark.generator import (
    EventSpec,
    GapSpec,
    NoiseBurstSpec,
    RollArtifactSpec,
    ScenarioConfig,
)
from qp.events.bands import classify_period

_MIN = 60.0  # seconds


# ======================================================================
# Helper: evenly space event centers across the dataset
# ======================================================================

def _centers(
    n_events: int, duration_days: float, margin_h: float = 12.0,
) -> list[float]:
    """Return n_events center times (hours) evenly spaced with margin from edges."""
    total_h = duration_days * 24
    usable = total_h - 2 * margin_h
    spacing = usable / max(n_events, 1)
    return [margin_h + i * spacing + spacing / 2 for i in range(n_events)]


# ======================================================================
# TIER 1: Easy (target ≥95% recall)
# ======================================================================

def tier1_clean_qp30() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier1_clean_qp30",
        description="8 clean QP30 sine packets, white noise, circular pol",
        duration_days=10, noise_alpha=0.0, noise_sigma=0.01,
        background_trend=False, difficulty_tier="tier1",
        event_specs=[
            EventSpec(band="QP30", amplitude=2.0, center_hours=c,
                      decay_hours=2.0, difficulty="easy")
            for c in centers
        ],
    )


def tier1_clean_qp60() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier1_clean_qp60",
        description="8 clean QP60 sine packets, white noise, circular pol",
        duration_days=10, noise_alpha=0.0, noise_sigma=0.01,
        background_trend=False, difficulty_tier="tier1",
        event_specs=[
            EventSpec(band="QP60", amplitude=2.0, center_hours=c,
                      decay_hours=4.0, difficulty="easy")
            for c in centers
        ],
    )


def tier1_clean_qp120() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier1_clean_qp120",
        description="8 clean QP120 sine packets, white noise, circular pol",
        duration_days=10, noise_alpha=0.0, noise_sigma=0.01,
        background_trend=False, difficulty_tier="tier1",
        event_specs=[
            EventSpec(band="QP120", amplitude=2.0, center_hours=c,
                      decay_hours=6.0, difficulty="easy")
            for c in centers
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
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier2_colored_noise",
        description="QP60 in realistic colored noise (alpha=1.2)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, difficulty="moderate")
            for c in centers
        ],
    )


def tier2_low_amplitude() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier2_low_amplitude",
        description="QP60 at 0.3 nT in realistic noise",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.3, center_hours=c,
                      decay_hours=4.0, difficulty="moderate")
            for c in centers
        ],
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
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=3.0, waveform=wf, sawtooth_width=sw,
                      difficulty="moderate")
            for c, (wf, sw) in zip(centers, shapes)
        ],
    )


def tier2_linear_pol() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier2_linear_pol",
        description="QP60 with linear polarization in realistic noise",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=3.0, polarization="linear",
                      ellipticity=0.0, difficulty="moderate")
            for c in centers
        ],
    )


def tier2_short_packets() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier2_short_packets",
        description="QP60 with short duration (~3-4 oscillations) in realistic noise",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=1.5, difficulty="moderate")
            for c in centers
        ],
    )


def tier2_ppo_background() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier2_ppo_background",
        description="QP60 with full magnetospheric background (PPO + trend)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=True, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.5, center_hours=c,
                      decay_hours=4.0, difficulty="moderate")
            for c in centers
        ],
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
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=specs,
    )


def tier3_frequency_drift() -> ScenarioConfig:
    centers = _centers(6, 10)
    chirps = [2e-9, -2e-9, 5e-9, -5e-9, 3e-9, -3e-9]
    return ScenarioConfig(
        dataset_id="tier3_frequency_drift",
        description="QP60 with frequency chirp (travelling wave signature)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, chirp_rate=cr, propagation="travelling",
                      asymmetry=0.3, difficulty="hard")
            for c, cr in zip(centers, chirps)
        ],
    )


def tier3_asymmetric_envelope() -> ScenarioConfig:
    centers = _centers(8, 10)
    asym = [0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85]
    return ScenarioConfig(
        dataset_id="tier3_asymmetric_envelope",
        description="QP60 with asymmetric envelopes (fast rise or fast fall)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, asymmetry=a, difficulty="hard")
            for c, a in zip(centers, asym)
        ],
    )


def tier3_amplitude_jitter() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier3_amplitude_jitter",
        description="QP60 with 30% per-cycle amplitude jitter",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, amplitude_jitter=0.3, difficulty="hard")
            for c in centers
        ],
    )


def tier3_near_threshold() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier3_near_threshold",
        description="QP60 at 0.15 nT with colored noise (SNR ~3)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.15, center_hours=c,
                      decay_hours=4.0, difficulty="hard")
            for c in centers
        ],
    )


def tier3_mixed_waveforms() -> ScenarioConfig:
    centers = _centers(8, 10)
    waveforms = ["sine", "sawtooth", "square", "sine",
                 "sawtooth", "sine", "square", "sawtooth"]
    return ScenarioConfig(
        dataset_id="tier3_mixed_waveforms",
        description="Mixed sine/sawtooth/square in same dataset",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, waveform=wf, difficulty="hard")
            for c, wf in zip(centers, waveforms)
        ],
    )


def tier3_travelling_waves() -> ScenarioConfig:
    centers = _centers(6, 10)
    return ScenarioConfig(
        dataset_id="tier3_travelling_waves",
        description="Travelling Alfvén wave packets (chirp + asymmetric envelope)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, chirp_rate=3e-9, asymmetry=0.3,
                      propagation="travelling", difficulty="hard")
            for c in centers
        ],
    )


# ======================================================================
# TIER 4: Extreme (target <30% recall)
# ======================================================================

def tier4_buried_in_noise() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier4_buried_in_noise",
        description="QP60 at SNR < 1.5 (nearly indistinguishable from noise)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=True, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.07, center_hours=c,
                      decay_hours=4.0, difficulty="extreme")
            for c in centers
        ],
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
    """Periods in the gap between QP30 and QP60 bands (40–45 min)."""
    centers = _centers(6, 10)
    periods = [40, 41, 42, 43, 44, 45]
    return ScenarioConfig(
        dataset_id="tier4_inter_band_period",
        description="Periods in the 40–45 min gap between QP30 and QP60",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", period_sec_override=p * _MIN,
                      amplitude=1.0, center_hours=c, decay_hours=4.0,
                      should_detect=False, difficulty="extreme",
                      event_type="qp_wave_inter_band")
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
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier4_incoherent",
        description="QP60 with 50% per-cycle jitter (very low coherence)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.5, center_hours=c,
                      decay_hours=4.0, amplitude_jitter=0.5, difficulty="extreme")
            for c in centers
        ],
    )


def tier4_harmonic_contamination() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier4_harmonic_contamination",
        description="QP60 with strong 2nd harmonic distortion",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                      decay_hours=4.0, harmonic_content=0.4, difficulty="extreme")
            for c in centers
        ],
    )


def tier4_elliptical_pol() -> ScenarioConfig:
    centers = _centers(6, 10)
    ellipticities = [0.3, -0.3, 0.4, -0.4, 0.5, -0.5]
    return ScenarioConfig(
        dataset_id="tier4_elliptical_pol",
        description="QP60 with ambiguous elliptical polarization",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                      decay_hours=4.0, polarization="elliptical",
                      ellipticity=e, difficulty="extreme")
            for c, e in zip(centers, ellipticities)
        ],
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
    """Data gaps at the rising edge of QP60 events."""
    centers = _centers(6, 10)
    # Place 10-min gap 1h before each event center (rising edge)
    gaps = [GapSpec(center_hours=c - 1.0, duration_minutes=10) for c in centers]
    return ScenarioConfig(
        dataset_id="tier3_gaps_at_onset",
        description="QP60 with 10-min gaps at event onset",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, difficulty="hard")
            for c in centers
        ],
        gaps=gaps,
    )


def tier3_gaps_midpacket() -> ScenarioConfig:
    """Data gaps in the middle of QP60 events."""
    centers = _centers(6, 10)
    gaps = [GapSpec(center_hours=c, duration_minutes=8) for c in centers]
    return ScenarioConfig(
        dataset_id="tier3_gaps_midpacket",
        description="QP60 with 8-min gaps at event midpoint",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, difficulty="hard")
            for c in centers
        ],
        gaps=gaps,
    )


def tier4_heavy_gaps() -> ScenarioConfig:
    """Frequent data gaps (~every 2h, 5–15 min each)."""
    gaps = [
        GapSpec(center_hours=h, duration_minutes=5 + (h % 3) * 5)
        for h in range(12, 228, 2)  # every 2h across 10 days
    ]
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier4_heavy_gaps",
        description="QP60 with frequent gaps (every ~2h, 5-15 min)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                      decay_hours=4.0, difficulty="extreme")
            for c in centers
        ],
        gaps=gaps,
    )


def tier3_steep_noise() -> ScenarioConfig:
    """QP60 in steep α = 1.5 noise (von Papen et al. 2014)."""
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier3_steep_noise",
        description="QP60 in steep colored noise (alpha=1.5)",
        duration_days=10, noise_alpha=1.5, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, difficulty="hard")
            for c in centers
        ],
    )


def tier4_kolmogorov_noise() -> ScenarioConfig:
    """QP60 in Kolmogorov α ≈ 5/3 ≈ 1.7 noise (Xu et al. 2023)."""
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier4_kolmogorov_noise",
        description="QP60 in Kolmogorov turbulence (alpha=1.7)",
        duration_days=10, noise_alpha=1.7, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                      decay_hours=4.0, difficulty="extreme")
            for c in centers
        ],
    )


def tier3_ppo_beat() -> ScenarioConfig:
    """QP60 with dual-PPO beat constructively enhancing transverse power."""
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier3_ppo_beat",
        description="QP60 with dual N/S PPO beat modulation",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=True, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, difficulty="hard")
            for c in centers
        ],
    )


def tier3_plasma_sheet() -> ScenarioConfig:
    """QP60 with localized noise bursts (plasma sheet crossings)."""
    centers = _centers(6, 10)
    bursts = [
        NoiseBurstSpec(center_hours=c, duration_hours=3.0,
                       sigma_multiplier=5.0)
        for c in centers[:3]
    ]
    return ScenarioConfig(
        dataset_id="tier3_plasma_sheet",
        description="QP60 overlapping 5x noise bursts (plasma sheet)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, difficulty="hard")
            for c in centers
        ],
        noise_bursts=bursts,
    )


def tier4_nonstationary() -> ScenarioConfig:
    """QP60 with continuously varying background noise level."""
    centers = _centers(8, 10)
    # Many overlapping noise bursts create non-stationary background
    bursts = [
        NoiseBurstSpec(center_hours=h, duration_hours=5.0,
                       sigma_multiplier=3.0 + (h % 7))
        for h in range(24, 216, 18)
    ]
    return ScenarioConfig(
        dataset_id="tier4_nonstationary",
        description="QP60 in non-stationary noise (varying σ)",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.03,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                      decay_hours=4.0, difficulty="extreme")
            for c in centers
        ],
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
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
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
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier2",
        event_specs=specs,
    )


def tier2_realistic_background() -> ScenarioConfig:
    """QP60 at catalog-median parameters in full magnetospheric background."""
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier2_realistic_background",
        description="QP60 at catalog-median amplitude in realistic background",
        duration_days=10, noise_alpha=1.3, noise_sigma=0.05,
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
        duration_days=10, noise_alpha=1.5, noise_sigma=0.05,
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
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
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
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
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
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
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
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
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
        duration_days=10, noise_alpha=1.3, noise_sigma=0.05,
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
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=specs,
    )


def tier4_harmonic_pairs() -> ScenarioConfig:
    """QP60 with harmonic content producing correlated QP120-range power."""
    centers = _centers(6, 10)
    return ScenarioConfig(
        dataset_id="tier4_harmonic_pairs",
        description="QP60 with 50% harmonic leaking into QP120 band",
        duration_days=10, noise_alpha=1.2, noise_sigma=0.05,
        background_trend=False, ppo_amplitude=0.5, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=3.0, harmonic_content=0.5,
                      difficulty="extreme")
            for c in centers
        ],
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
    "decoy_red_noise_qp120": decoy_red_noise_qp120,
    "decoy_broadband_redslope": decoy_broadband_redslope,
    "decoy_ppo_harmonic": decoy_ppo_harmonic,
}

TIER_SCENARIOS: dict[str, list[str]] = {
    "tier1": [k for k in ALL_SCENARIOS if k.startswith("tier1")],
    "tier2": [k for k in ALL_SCENARIOS if k.startswith("tier2")],
    "tier3": [k for k in ALL_SCENARIOS if k.startswith("tier3")],
    "tier4": [k for k in ALL_SCENARIOS if k.startswith("tier4")],
    "decoy": [k for k in ALL_SCENARIOS if k.startswith("decoy")],
}
