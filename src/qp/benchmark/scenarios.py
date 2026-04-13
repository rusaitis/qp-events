"""Pre-defined benchmark scenarios at four difficulty tiers plus decoys.

Each function returns a :class:`ScenarioConfig` with deterministic seeds.
Events are spaced to avoid overlap (≥30 h between packet centers for
QP30/QP60, ≥40 h for QP120).

Scenario counts:
    Tier 1 (easy):     5 scenarios, ~50 detectable events
    Tier 2 (moderate): 6 scenarios, ~48 detectable events
    Tier 3 (hard):     7 scenarios, ~52 detectable events
    Tier 4 (extreme):  8 scenarios, ~54 detectable events
    Decoy:             6 scenarios, ~27 non-detectable events
    TOTAL:             32 scenarios, ~204 detectable + 27 decoy = 231 events
"""

from __future__ import annotations

from qp.benchmark.generator import EventSpec, ScenarioConfig

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
    return ScenarioConfig(
        dataset_id="tier1_full_spectrum",
        description="Period sweep 5–180 min, high amplitude, white noise",
        duration_days=15, noise_alpha=0.0, noise_sigma=0.02,
        background_trend=False, difficulty_tier="tier1",
        event_specs=[
            EventSpec(
                band="QP60",  # not band-matched; using period override
                period_sec_override=p * _MIN,
                amplitude=3.0, center_hours=c, decay_hours=max(2.0, p / 30),
                should_detect=(20 <= p <= 150),  # only QP-band periods detectable
                difficulty="easy",
            )
            for p, c in zip(periods_min, centers)
        ],
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
        background_trend=False, difficulty_tier="tier2",
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
        description="QP60 at 0.3 nT (near detection threshold)",
        duration_days=10, noise_alpha=0.5, noise_sigma=0.02,
        background_trend=False, difficulty_tier="tier2",
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
        description="Mixed waveform shapes: sawtooth (various), square, sine",
        duration_days=10, noise_alpha=0.5, noise_sigma=0.02,
        background_trend=False, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.5, center_hours=c,
                      decay_hours=4.0, waveform=wf, sawtooth_width=sw,
                      difficulty="moderate")
            for c, (wf, sw) in zip(centers, shapes)
        ],
    )


def tier2_linear_pol() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier2_linear_pol",
        description="QP60 with linear polarization",
        duration_days=10, noise_alpha=0.5, noise_sigma=0.02,
        background_trend=False, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.5, center_hours=c,
                      decay_hours=4.0, polarization="linear",
                      ellipticity=0.0, difficulty="moderate")
            for c in centers
        ],
    )


def tier2_short_packets() -> ScenarioConfig:
    centers = _centers(8, 10)
    return ScenarioConfig(
        dataset_id="tier2_short_packets",
        description="QP60 with short duration (~3-4 oscillations)",
        duration_days=10, noise_alpha=0.5, noise_sigma=0.02,
        background_trend=False, difficulty_tier="tier2",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.5, center_hours=c,
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
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, difficulty_tier="tier3",
        event_specs=specs,
    )


def tier3_frequency_drift() -> ScenarioConfig:
    centers = _centers(6, 10)
    chirps = [2e-8, -2e-8, 5e-8, -5e-8, 3e-8, -3e-8]
    return ScenarioConfig(
        dataset_id="tier3_frequency_drift",
        description="QP60 with frequency chirp (travelling wave signature)",
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, difficulty_tier="tier3",
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
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, difficulty_tier="tier3",
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
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, difficulty_tier="tier3",
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
        background_trend=False, difficulty_tier="tier3",
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
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, difficulty_tier="tier3",
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
        duration_days=10, noise_alpha=1.0, noise_sigma=0.03,
        background_trend=False, difficulty_tier="tier3",
        event_specs=[
            EventSpec(band="QP60", amplitude=1.0, center_hours=c,
                      decay_hours=4.0, chirp_rate=3e-8, asymmetry=0.3,
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
        background_trend=False, difficulty_tier="tier4",
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
        background_trend=False, difficulty_tier="tier4",
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
        background_trend=False, difficulty_tier="tier4",
        event_specs=[
            EventSpec(band="QP60", amplitude=0.8, center_hours=c,
                      decay_hours=5.0, chirp_rate=8e-8,
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
        background_trend=False, difficulty_tier="tier4",
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
        background_trend=False, difficulty_tier="tier4",
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
        background_trend=False, difficulty_tier="tier4",
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
        background_trend=False, difficulty_tier="tier4",
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
        background_trend=False, difficulty_tier="decoy",
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
        background_trend=False, difficulty_tier="decoy",
        event_specs=[
            EventSpec(band="QP60", amplitude=3.0, center_hours=c,
                      decay_hours=0.5, mode="compressional",
                      should_detect=False, difficulty="easy",
                      event_type="decoy_single_pulse")
            for c in centers
        ],
    )


def decoy_broadband_burst() -> ScenarioConfig:
    """Broadband power bursts — just windowed colored noise, no spectral peak."""
    centers = _centers(4, 10)
    return ScenarioConfig(
        dataset_id="decoy_broadband_burst",
        description="Broadband noise bursts (no monochromatic content)",
        duration_days=10, noise_alpha=1.5, noise_sigma=0.1,
        background_trend=False, difficulty_tier="decoy",
        event_specs=[
            EventSpec(band="QP60", period_sec_override=3600.0,
                      amplitude=0.0, center_hours=c, decay_hours=2.0,
                      should_detect=False, difficulty="easy",
                      event_type="decoy_broadband")
            for c in centers
        ],
    )


def decoy_step_functions() -> ScenarioConfig:
    centers = _centers(4, 10)
    return ScenarioConfig(
        dataset_id="decoy_step_functions",
        description="Slow step-like B_tot changes (magnetopause crossings)",
        duration_days=10, noise_alpha=0.5, noise_sigma=0.02,
        background_trend=False, difficulty_tier="decoy",
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
    # Tier 4
    "tier4_buried_in_noise": tier4_buried_in_noise,
    "tier4_qp120_short": tier4_qp120_short,
    "tier4_inter_band_period": tier4_inter_band_period,
    "tier4_extreme_chirp": tier4_extreme_chirp,
    "tier4_decaying": tier4_decaying,
    "tier4_incoherent": tier4_incoherent,
    "tier4_harmonic_contamination": tier4_harmonic_contamination,
    "tier4_elliptical_pol": tier4_elliptical_pol,
    # Decoys
    "decoy_compressional": decoy_compressional,
    "decoy_single_pulses": decoy_single_pulses,
    "decoy_broadband_burst": decoy_broadband_burst,
    "decoy_step_functions": decoy_step_functions,
    "decoy_ppo_only": decoy_ppo_only,
}

TIER_SCENARIOS: dict[str, list[str]] = {
    "tier1": [k for k in ALL_SCENARIOS if k.startswith("tier1")],
    "tier2": [k for k in ALL_SCENARIOS if k.startswith("tier2")],
    "tier3": [k for k in ALL_SCENARIOS if k.startswith("tier3")],
    "tier4": [k for k in ALL_SCENARIOS if k.startswith("tier4")],
    "decoy": [k for k in ALL_SCENARIOS if k.startswith("decoy")],
}
