"""Benchmark dataset generator.

Composes noise backgrounds and injected wave packets into reproducible
synthetic datasets with full ground-truth manifests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from qp.events.bands import QP_BANDS
from qp.events.catalog import WaveTemplate
from qp.signal.noise import (
    bandlimited_noise_burst,
    colored_noise_3component,
    inject_ppo,
    magnetospheric_background,
)
from qp.signal.synthetic import simulate_wave_physics
from qp.benchmark.manifest import DatasetManifest, InjectedEvent


@dataclass(frozen=True, slots=True)
class GapSpec:
    """A data gap to insert into the synthetic dataset."""

    center_hours: float
    duration_minutes: float


@dataclass(frozen=True, slots=True)
class NoiseBurstSpec:
    """A localized noise enhancement (e.g., plasma sheet crossing)."""

    center_hours: float
    duration_hours: float
    sigma_multiplier: float  # local noise increase factor


@dataclass(frozen=True, slots=True)
class RollArtifactSpec:
    """A spacecraft roll maneuver artifact."""

    center_hours: float
    duration_hours: float  # typically 1–4h
    rotation_deg: float  # rotation amplitude


@dataclass(frozen=True, slots=True)
class EventSpec:
    """Specification for a single event to inject."""

    band: str = "QP60"
    period_offset_frac: float = 0.0  # offset from band centroid
    amplitude: float = 1.0  # nT
    center_hours: float = 12.0  # hours from dataset start
    decay_hours: float = 4.0  # Gaussian envelope width in hours
    mode: str = "alfvenic"  # "alfvenic", "compressional", "mixed"
    polarization: str = "circular"
    propagation: str = "standing"  # "standing", "travelling"
    waveform: str = "sine"
    chirp_rate: float = 0.0  # Hz/s
    ellipticity: float = 1.0  # [-1, 1]
    asymmetry: float = 0.5  # envelope asymmetry
    amplitude_jitter: float = 0.0
    sawtooth_width: float = 0.8
    harmonic_content: float = 0.0
    envelope_shape: str = "gaussian"  # 'gaussian', 'lognormal', 'rayleigh'
    harmonic_model: str = "linear_2f"  # 'linear_2f' or 'sawtooth_truncated'
    should_detect: bool = True
    difficulty: str = "easy"
    event_type: str = "qp_wave"
    injection_type: str = "wave"  # "wave" or "noise_burst"
    # For non-band signals (decoys at specific periods)
    period_sec_override: float | None = None
    # Override frequency range and spectral slope for noise bursts
    burst_freq_lo: float | None = None
    burst_freq_hi: float | None = None
    burst_alpha: float = 1.0


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
    # PPO amplitude (nT). Applied independently of background_trend.
    # 0.0 = no PPO; 0.5 = typical magnetospheric; 1.0 = strong.
    ppo_amplitude: float = 0.0
    event_specs: list[EventSpec] = field(default_factory=list)
    gaps: list[GapSpec] = field(default_factory=list)
    noise_bursts: list[NoiseBurstSpec] = field(default_factory=list)
    roll_artifacts: list[RollArtifactSpec] = field(default_factory=list)
    # Noise-shape upgrades (off by default for backward compat).
    # ``noise_tail_df``: degrees of freedom for Student-t Fourier
    # innovations (heavy tails). None → Gaussian.
    noise_tail_df: float | None = None
    # ``noise_regime_switching``: if True, replaces the colored
    # background with piecewise-stationary α(t) (range below).
    noise_regime_switching: bool = False
    noise_alpha_range: tuple[float, float] = (1.0, 1.7)
    noise_segment_hours_range: tuple[float, float] = (6.0, 12.0)
    # ``realistic_ppo``: if True, ``inject_ppo`` uses log-normal
    # amplitude envelopes, period drift, and Brownian phase walk.
    realistic_ppo: bool = False
    # ``isotropic_noise``: if True, B_par noise uses the same
    # (alpha, sigma) as B_perp instead of the magnetospheric
    # default (steeper + smaller). ``anisotropy_par_ratio`` overrides
    # the σ_par/σ_perp ratio when set (e.g. 0.25 = extreme anisotropy).
    isotropic_noise: bool = False
    anisotropy_par_ratio: float | None = None


def _resolve_period(spec: EventSpec) -> float:
    """Compute the injection period in seconds from band centroid + offset."""
    if spec.period_sec_override is not None:
        return spec.period_sec_override
    band_obj = QP_BANDS.get(spec.band.upper())
    if band_obj is None:
        raise ValueError(f"Unknown band: {spec.band!r}")
    centroid = band_obj.period_centroid_sec
    return centroid * (1.0 + spec.period_offset_frac)


def _power_law_integral(f_lo: float, f_hi: float, alpha: float) -> float:
    r"""Evaluate $\int_{f_1}^{f_2} f^{-\alpha}\,df$ analytically.

    Linearly blends the log and power branches over ``|α-1| < 0.05``
    so the integral is continuous in α (the ad-hoc switch at 0.01
    produced a visible kink near α=1).
    """
    delta = alpha - 1.0
    if abs(delta) < 1e-9:
        return math.log(f_hi / f_lo)
    if abs(delta) < 0.05:
        # Blend to avoid numerical instability at alpha==1.
        w = abs(delta) / 0.05
        exp = 1.0 - alpha
        power = (f_hi**exp - f_lo**exp) / exp
        logv = math.log(f_hi / f_lo)
        return w * power + (1.0 - w) * logv
    exp = 1.0 - alpha
    return (f_hi**exp - f_lo**exp) / exp


def _empirical_band_rms(
    noise: np.ndarray,
    dt: float,
    f_lo: float,
    f_hi: float,
) -> float:
    r"""Empirical RMS of ``noise`` bandpassed to ``[f_lo, f_hi]``.

    Uses a single-sided power spectrum (rfft) and Parseval normalization,
    so the return value is in the same units as ``noise``. Unlike the
    analytic ``_in_band_snr``, this captures the spectral shape of the
    *realized* noise after sample-RMS renormalization in
    :func:`qp.signal.noise.power_law_noise`.
    """
    n = noise.size
    if n < 4:
        return 0.0
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, d=dt)
    # One-sided PSD (V^2/Hz): |X|^2 / (n * fs); multiply by 2 except DC/Nyquist.
    fs = 1.0 / dt
    psd = (np.abs(spec) ** 2) / (n * fs)
    if n > 1:
        psd[1:-1] *= 2.0
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0
    df = freqs[1] - freqs[0] if len(freqs) > 1 else fs / n
    band_power = float(np.sum(psd[mask]) * df)
    return math.sqrt(max(band_power, 0.0))


def _in_band_snr(
    amplitude: float,
    noise_sigma: float,
    noise_alpha: float,
    period_sec: float,
    dt: float,
    band_name: str | None,
) -> float:
    r"""Compute in-band SNR analytically.

    $$\text{SNR}_{\text{in-band}} = \frac{A}{\sqrt{\int_{f_1}^{f_2}
    C \cdot f^{-\alpha} \, df}}$$

    where $C$ is set so that the total broadband RMS equals ``noise_sigma``.
    """
    if noise_sigma <= 0 or noise_alpha <= 0:
        return amplitude / max(noise_sigma, 1e-30)

    f_nyq = 0.5 / dt
    f_min = 1e-6  # Hz, well below any QP band

    total_integral = _power_law_integral(f_min, f_nyq, noise_alpha)
    c_norm = noise_sigma**2 / total_integral

    # Band edges (use Band properties for canonical lookup)
    if band_name and band_name.upper() in QP_BANDS:
        band = QP_BANDS[band_name.upper()]
        f1 = band.freq_min_hz
        f2 = band.freq_max_hz
    else:
        f0 = 1.0 / period_sec
        f1 = 0.7 * f0
        f2 = 1.3 * f0

    band_integral = _power_law_integral(f1, f2, noise_alpha)
    noise_in_band = math.sqrt(c_norm * band_integral)
    return amplitude / max(noise_in_band, 1e-30)


def _inject_noise_bursts(
    bg: np.ndarray, t: np.ndarray,
    bursts: list[NoiseBurstSpec],
) -> None:
    """Multiply background noise by localized Gaussian gain envelopes."""
    for burst in bursts:
        center = burst.center_hours * 3600.0
        sigma = burst.duration_hours * 3600.0
        gain = 1.0 + (burst.sigma_multiplier - 1.0) * np.exp(
            -0.5 * ((t - center) / sigma) ** 2
        )
        bg *= gain[:, np.newaxis]


def _inject_roll_artifact(
    b_perp1: np.ndarray, b_perp2: np.ndarray,
    t: np.ndarray, spec: RollArtifactSpec,
) -> None:
    r"""Inject a smooth spacecraft roll artifact into transverse components.

    A real roll maneuver is not a perfect unitary rotation: FGM gain and
    zero-level mismatches introduce a small magnitude perturbation in
    the transverse plane (~few percent of field magnitude). Without
    this, the decoy is invisible to any detector operating on power or
    magnitude, making decoy_roll_artifacts a free rejection credit.
    """
    center = spec.center_hours * 3600.0
    half_dur = spec.duration_hours * 3600.0 / 2
    mask = (t >= center - half_dur) & (t <= center + half_dur)
    if not np.any(mask):
        return
    # Raised cosine angle ramp
    local_t = (t[mask] - center) / half_dur  # [-1, 1]
    angle_rad = np.radians(spec.rotation_deg) * 0.5 * (1 + np.cos(np.pi * local_t))
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    p1 = b_perp1[mask].copy()
    p2 = b_perp2[mask].copy()
    # Non-unitary gain/offset: small per-component scaling modulated by
    # the same raised-cosine envelope so |B_perp| changes slightly over
    # the roll interval. Keep the perturbation small (≲ 3 %) — a real
    # detector should still see the DC-like artifact rather than a
    # QP-band signal.
    gain_env = 0.5 * (1 + np.cos(np.pi * local_t))
    gain1 = 1.0 + 0.025 * gain_env
    gain2 = 1.0 - 0.018 * gain_env
    b_perp1[mask] = gain1 * (cos_a * p1 - sin_a * p2)
    b_perp2[mask] = gain2 * (sin_a * p1 + cos_a * p2)


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

    # Background — three orthogonal options:
    #   * ``noise_regime_switching=True`` overrides the spectral
    #     model with piecewise-stationary α(t).
    #   * ``noise_tail_df`` switches the Fourier innovations to
    #     Student-t (heavy-tailed) for both regimes.
    #   * ``isotropic_noise=True`` and ``anisotropy_par_ratio`` tune
    #     the parallel-component anisotropy.
    if scenario.noise_regime_switching:
        from qp.signal.noise import regime_switching_noise

        bg = np.empty((n_samples, 3))
        if scenario.isotropic_noise:
            sig_par = scenario.noise_sigma
        elif scenario.anisotropy_par_ratio is not None:
            sig_par = scenario.noise_sigma * scenario.anisotropy_par_ratio
        else:
            sig_par = scenario.noise_sigma / 2.0
        for i in range(3):
            sigma_i = sig_par if i == 0 else scenario.noise_sigma
            bg[:, i] = regime_switching_noise(
                n_samples, scenario.dt,
                alpha_range=scenario.noise_alpha_range,
                segment_hours_range=scenario.noise_segment_hours_range,
                sigma=sigma_i,
                seed=int(rng.integers(0, 2**31)),
                tail_df=scenario.noise_tail_df,
            )
        if scenario.background_trend:
            bg[:, 0] += 5.0  # mean field in B_par
    elif scenario.background_trend:
        if scenario.isotropic_noise:
            sig_par_arg = scenario.noise_sigma
            alpha_par_arg = scenario.noise_alpha
        elif scenario.anisotropy_par_ratio is not None:
            sig_par_arg = scenario.noise_sigma * scenario.anisotropy_par_ratio
            alpha_par_arg = None  # fall back to magnetospheric default
        else:
            sig_par_arg = None
            alpha_par_arg = None
        bg = magnetospheric_background(
            n_samples, scenario.dt,
            seed=int(rng.integers(0, 2**31)),
            noise_alpha=scenario.noise_alpha,
            noise_sigma=scenario.noise_sigma,
            noise_alpha_par=alpha_par_arg,
            noise_sigma_par=sig_par_arg,
            # Suppress the built-in (sinusoidal) PPO when the scenario
            # requests the realistic model — it is injected separately
            # below so the realistic-PPO knob actually takes effect.
            ppo_amplitude=0.0 if scenario.realistic_ppo else 0.5,
        )
        # If a tail_df is requested, replace the colored components
        # in-place so the trend/PPO additions remain.
        if scenario.noise_tail_df is not None:
            from qp.signal.noise import power_law_noise as _pln
            for i in range(3):
                if scenario.isotropic_noise:
                    a_i = scenario.noise_alpha
                    s_i = scenario.noise_sigma
                elif i == 0:
                    a_i = scenario.noise_alpha + 0.2
                    s_i = (
                        sig_par_arg if sig_par_arg is not None
                        else scenario.noise_sigma / 2.0
                    )
                else:
                    a_i = scenario.noise_alpha
                    s_i = scenario.noise_sigma
                bg[:, i] -= bg[:, i].mean()  # remove the colored part's bias
                bg[:, i] = _pln(
                    n_samples, scenario.dt, alpha=a_i, sigma=s_i,
                    seed=int(rng.integers(0, 2**31)),
                    tail_df=scenario.noise_tail_df,
                )
            if scenario.background_trend:
                bg[:, 0] += 5.0  # restore mean field
    else:
        if scenario.isotropic_noise:
            a_par = scenario.noise_alpha
            s_par = scenario.noise_sigma
        elif scenario.anisotropy_par_ratio is not None:
            a_par = scenario.noise_alpha + 0.2
            s_par = scenario.noise_sigma * scenario.anisotropy_par_ratio
        else:
            a_par = None
            s_par = None
        if scenario.noise_tail_df is None:
            bg = colored_noise_3component(
                n_samples, scenario.dt,
                alpha=scenario.noise_alpha,
                sigma=scenario.noise_sigma,
                seed=int(rng.integers(0, 2**31)),
                alpha_par=a_par,
                sigma_par=s_par,
            )
        else:
            from qp.signal.noise import power_law_noise as _pln
            bg = np.empty((n_samples, 3))
            for i in range(3):
                a_i = (a_par if a_par is not None else scenario.noise_alpha) if i == 0 else scenario.noise_alpha
                s_i = (s_par if s_par is not None else scenario.noise_sigma) if i == 0 else scenario.noise_sigma
                bg[:, i] = _pln(
                    n_samples, scenario.dt, alpha=a_i, sigma=s_i,
                    seed=int(rng.integers(0, 2**31)),
                    tail_df=scenario.noise_tail_df,
                )

    # Standalone PPO (independent of background_trend)
    if scenario.ppo_amplitude > 0:
        inject_ppo(
            bg, t, amplitude=scenario.ppo_amplitude,
            seed=int(rng.integers(0, 2**31)),
            realistic=scenario.realistic_ppo,
        )

    # Non-stationary noise bursts (before event injection)
    if scenario.noise_bursts:
        _inject_noise_bursts(bg, t, scenario.noise_bursts)

    b_par = bg[:, 0].copy()
    b_perp1 = bg[:, 1].copy()
    b_perp2 = bg[:, 2].copy()

    # Snapshot of the transverse-component noise (pre-injection) so we
    # can compute empirical in-band SNR per event. Stored once for the
    # whole dataset — the noise floor is stationary by construction
    # (non-stationary bursts are applied before this snapshot).
    noise_perp1 = b_perp1.copy()
    noise_perp2 = b_perp2.copy()

    # Inject events
    injected_events: list[InjectedEvent] = []

    for i, spec in enumerate(scenario.event_specs):
        period_sec = _resolve_period(spec)
        center_sec = spec.center_hours * 3600.0
        decay_sec = spec.decay_hours * 3600.0

        if spec.injection_type == "noise_burst":
            # Bandlimited noise burst (broadband decoy)
            if spec.burst_freq_lo is not None and spec.burst_freq_hi is not None:
                f_lo, f_hi = spec.burst_freq_lo, spec.burst_freq_hi
            else:
                band_obj = QP_BANDS.get(spec.band.upper())
                if band_obj:
                    f_lo = band_obj.freq_min_hz
                    f_hi = band_obj.freq_max_hz
                else:
                    f0 = 1.0 / period_sec
                    f_lo, f_hi = 0.5 * f0, 2.0 * f0

            burst1 = bandlimited_noise_burst(
                n_samples, scenario.dt,
                center_sec=center_sec, decay_sec=decay_sec,
                freq_lo=f_lo, freq_hi=f_hi,
                amplitude=spec.amplitude,
                alpha=spec.burst_alpha,
                seed=int(rng.integers(0, 2**31)),
            )
            burst2 = bandlimited_noise_burst(
                n_samples, scenario.dt,
                center_sec=center_sec, decay_sec=decay_sec,
                freq_lo=f_lo, freq_hi=f_hi,
                amplitude=spec.amplitude * 0.7,
                alpha=spec.burst_alpha,
                seed=int(rng.integers(0, 2**31)),
            )
            b_perp1 += burst1
            b_perp2 += burst2
        else:
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
                envelope_shape=spec.envelope_shape,
                harmonic_model=spec.harmonic_model,
            )

            _, wave_fields = simulate_wave_physics(
                n_samples, scenario.dt, [wave],
                mode=spec.mode,
                polarization=spec.polarization,
                ellipticity=spec.ellipticity,
                seed=int(rng.integers(0, 2**31)),
            )

            b_par += wave_fields[:, 0]
            b_perp1 += wave_fields[:, 1]
            b_perp2 += wave_fields[:, 2]

        # Compute event boundaries. The primary boundary is ±2σ
        # (95.4 % energy) so that the ground-truth interval aligns
        # with the detector's envelope-trim convention: a Morlet-CWT
        # ridge of a Gaussian packet falls to ≈0.02 of its peak power
        # at ≈±2σ (power envelope = exp(-t²/σ²) ⇒ 0.02 → t≈1.98σ),
        # which is the threshold used in ``filter_detections``. The
        # historical ±3σ boundary systematically over-extended GT
        # beyond the detector's reachable duration, producing a
        # ~5 % IoU underestimate on otherwise-perfect matches.
        if spec.asymmetry != 0.5:
            sigma_left = decay_sec * (0.5 + spec.asymmetry)
            sigma_right = decay_sec * (1.5 - spec.asymmetry)
        else:
            sigma_left = decay_sec
            sigma_right = decay_sec

        start_sec = max(0.0, center_sec - 2.0 * sigma_left)
        end_sec = min(t[-1], center_sec + 2.0 * sigma_right)
        # ``start_2sigma_sec`` / ``end_2sigma_sec`` are kept in the
        # manifest for back-compat and now coincide with the primary
        # boundary. A wider 3σ window remains available via
        # (center ± 3σ) for downstream tooling that needs the full
        # tail; it is not used in scoring.
        start_2s = start_sec
        end_2s = end_sec
        duration_sec = end_sec - start_sec
        n_osc = duration_sec / period_sec

        band_label = spec.band if spec.should_detect else None

        # In-band SNR — analytic estimate.
        snr_bb = spec.amplitude / max(scenario.noise_sigma, 1e-30)
        snr_ib = _in_band_snr(
            spec.amplitude, scenario.noise_sigma, scenario.noise_alpha,
            period_sec, scenario.dt, band_label,
        )
        # Empirical SNR from the realised noise: measure perp1/perp2
        # band RMS over the target band and average. NaN for decoys
        # without an assigned band.
        if band_label and band_label.upper() in QP_BANDS:
            band_obj = QP_BANDS[band_label.upper()]
            f1_emp, f2_emp = band_obj.freq_min_hz, band_obj.freq_max_hz
        else:
            f0 = 1.0 / period_sec
            f1_emp, f2_emp = 0.7 * f0, 1.3 * f0
        rms1 = _empirical_band_rms(noise_perp1, scenario.dt, f1_emp, f2_emp)
        rms2 = _empirical_band_rms(noise_perp2, scenario.dt, f1_emp, f2_emp)
        rms_mean = math.sqrt(max((rms1 * rms1 + rms2 * rms2) / 2.0, 0.0))
        snr_ib_emp = (
            spec.amplitude / rms_mean if rms_mean > 0 else math.nan
        )

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
            snr_injected=snr_bb,
            snr_in_band=snr_ib,
            start_2sigma_sec=start_2s,
            end_2sigma_sec=end_2s,
            difficulty=spec.difficulty,
            snr_in_band_empirical=snr_ib_emp,
            envelope_shape=spec.envelope_shape,
            harmonic_model=spec.harmonic_model,
        ))

    # Roll artifacts (decoy transverse perturbations)
    for roll in scenario.roll_artifacts:
        _inject_roll_artifact(b_perp1, b_perp2, t, roll)

    b_tot = np.sqrt(b_par**2 + b_perp1**2 + b_perp2**2)
    fields = np.column_stack([b_par, b_perp1, b_perp2, b_tot])

    # Data gaps (NaN insertion)
    for gap in scenario.gaps:
        gap_center = gap.center_hours * 3600.0
        gap_half = gap.duration_minutes * 30.0
        i_start = max(0, int((gap_center - gap_half) / scenario.dt))
        i_end = min(n_samples, int((gap_center + gap_half) / scenario.dt))
        fields[i_start:i_end, :] = np.nan

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
