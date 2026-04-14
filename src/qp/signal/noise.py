r"""Colored and power-law noise generation.

Real Cassini MAG data has a power spectral density that follows a power
law $P(f) \propto f^{-\alpha}$ with $\alpha \approx 1.0$–$1.7$ in the
outer magnetosphere (von Papen, Saur & Alexandrova 2014; Xu et al. 2023).
White Gaussian noise ($\alpha = 0$) is unrealistic for benchmarking.

The Timmer–König (1995) method generates noise with a prescribed PSD
shape by drawing independent Gaussian random numbers for the real and
imaginary parts of each Fourier coefficient, scaling by
$\sqrt{P(f)/2}$, and inverse-transforming. For steep spectra
($\alpha \geq 1.5$), we oversample by 10× and extract the central
segment to suppress low-frequency leakage (Kirchner 2005).

References
----------
Timmer, J. & König, M. (1995). *A&A*, 300, 707.
von Papen, M., Saur, J. & Alexandrova, O. (2014). *JGR*, 119, 2797.
Kirchner, J. W. (2005). *Phys. Rev. E*, 71, 066110.
"""

from __future__ import annotations

import math

import numpy as np

from qp.signal.morphology import bandpass


# Threshold above which we oversample to suppress low-frequency leakage
_STEEP_ALPHA_THRESHOLD = 1.5
_OVERSAMPLE_FACTOR = 10


def power_law_noise(
    n_samples: int,
    dt: float = 60.0,
    alpha: float = 1.2,
    sigma: float = 1.0,
    seed: int | None = None,
    tail_df: float | None = None,
) -> np.ndarray:
    r"""Generate noise with PSD $\propto f^{-\alpha}$ (Timmer–König method).

    Parameters
    ----------
    n_samples : int
        Number of time-domain samples.
    dt : float
        Sampling interval in seconds.
    alpha : float
        PSD slope. 0 = white, 1 = pink (1/f), 2 = Brownian (1/f²).
        Real Cassini magnetospheric noise: $\alpha \approx 1.0$–$1.7$.
    sigma : float
        Target RMS amplitude of the output.
    seed : int or None
        Random seed for reproducibility.
    tail_df : float, optional
        If given, draws the Fourier coefficients from a (scaled)
        Student-t distribution with ``tail_df`` degrees of freedom
        instead of Gaussian. ``tail_df=4`` gives heavy-tailed
        innovations with finite variance (kurtosis 3 + 6/(df-4), so
        undefined for df=4 and 9 for df=5 — use df≥5 in practice).
        The output is renormalised to ``sigma`` after transforming.
        Matches the heavy-tailed behaviour of real magnetometer
        residuals around CME shocks and current-sheet crossings.

    Returns
    -------
    ndarray, shape (n_samples,)
        Time-domain noise with the requested spectral shape.
    """
    rng = np.random.default_rng(seed)

    # For steep spectra, oversample and extract central segment
    if alpha >= _STEEP_ALPHA_THRESHOLD:
        n_gen = n_samples * _OVERSAMPLE_FACTOR
    else:
        n_gen = n_samples

    freqs = np.fft.rfftfreq(n_gen, d=dt)

    # Power scaling: P(f) ∝ f^{-alpha}, so amplitude ∝ f^{-alpha/2}
    amplitude = np.ones_like(freqs)
    amplitude[1:] = freqs[1:] ** (-alpha / 2)
    amplitude[0] = 0.0  # zero DC component

    # Timmer–König: independent Gaussian draws for real and imaginary
    # parts, each scaled by sqrt(P(f)/2). Generates Gaussian-marginal
    # colored noise.
    re = rng.standard_normal(len(freqs))
    im = rng.standard_normal(len(freqs))
    spectrum = (re + 1j * im) * amplitude / np.sqrt(2)

    # Nyquist bin must be real for even-length signals
    if n_gen % 2 == 0:
        spectrum[-1] = spectrum[-1].real

    noise = np.fft.irfft(spectrum, n=n_gen)

    # Extract central segment for oversampled steep spectra
    if n_gen > n_samples:
        start = (n_gen - n_samples) // 2
        noise = noise[start : start + n_samples]

    # Heavy-tailed reshaping: keep the spectral shape but reshape
    # the marginal distribution from Gaussian to Student-t. We do
    # this by sorting the Gaussian samples and substituting the
    # rank-equivalent t-quantiles (a copula-style transform that
    # preserves the autocorrelation/PSD shape to leading order
    # while converting the marginal). Without this step, applying
    # Student-t draws in the Fourier domain leaves the time-domain
    # marginal essentially Gaussian by CLT — the test we used to
    # ship would have caught nothing.
    if tail_df is not None:
        from scipy import stats as _stats

        df = max(float(tail_df), 2.1)
        order = np.argsort(noise)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(noise.size)
        # Plotting positions in (0, 1) — Hazen formula, avoids 0 and 1.
        u = (ranks + 0.5) / noise.size
        # Student-t quantiles, standardised to unit variance.
        scale = math.sqrt((df - 2.0) / df)
        noise = _stats.t.ppf(u, df) * scale
        # Restore the requested sample variance after the transform.

    # Normalize to requested sigma
    current_rms = np.sqrt(np.mean(noise**2))
    if current_rms > 0:
        noise *= sigma / current_rms

    return noise


def regime_switching_noise(
    n_samples: int,
    dt: float = 60.0,
    alpha_range: tuple[float, float] = (1.0, 1.7),
    segment_hours_range: tuple[float, float] = (6.0, 12.0),
    sigma: float = 1.0,
    seed: int | None = None,
    tail_df: float | None = None,
) -> np.ndarray:
    r"""Piecewise-stationary 1/f^α noise with time-varying slope.

    Real Cassini backgrounds are not globally stationary: the spectral
    slope drifts between α≈1.0 (quiet intervals) and α≈1.7 (active
    plasma-sheet crossings) on 6–12 h timescales. A detector
    validated on a single fixed α overstates specificity relative
    to the real data. This generator concatenates segments of fixed
    duration in ``segment_hours_range`` at α drawn uniformly from
    ``alpha_range``, cross-fades between segments with a 5-minute
    raised-cosine window to avoid discontinuity artefacts, and
    renormalises the whole trace to ``sigma``.

    Parameters
    ----------
    n_samples, dt, sigma, seed : same as :func:`power_law_noise`
    alpha_range : (alpha_lo, alpha_hi)
        Bounds for the per-segment α.
    segment_hours_range : (lo, hi)
        Bounds for per-segment duration in hours.
    tail_df : float, optional
        Passed through to :func:`power_law_noise` for heavy-tailed
        innovations in every segment.

    Returns
    -------
    ndarray, shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(n_samples)
    cursor = 0
    seg_lo, seg_hi = segment_hours_range
    a_lo, a_hi = alpha_range
    fade_samples = max(1, int(300.0 / dt))  # 5-minute raised cosine
    while cursor < n_samples:
        seg_hours = float(rng.uniform(seg_lo, seg_hi))
        seg_n = min(int(seg_hours * 3600.0 / dt), n_samples - cursor)
        if seg_n <= 1:
            break
        a = float(rng.uniform(a_lo, a_hi))
        seg = power_law_noise(
            seg_n, dt=dt, alpha=a, sigma=sigma,
            seed=int(rng.integers(0, 2**31)),
            tail_df=tail_df,
        )
        if cursor == 0 or fade_samples >= seg_n:
            out[cursor : cursor + seg_n] = seg
        else:
            # Raised-cosine cross-fade to eliminate the step in
            # realised power at the segment boundary.
            k = min(fade_samples, seg_n, cursor)
            ramp = 0.5 * (1 - np.cos(np.linspace(0, math.pi, k)))
            out[cursor : cursor + k] = (
                (1 - ramp) * out[cursor : cursor + k] + ramp * seg[:k]
            )
            out[cursor + k : cursor + seg_n] = seg[k:seg_n]
        cursor += seg_n
    # Renormalise end-to-end so the advertised ``sigma`` is honoured.
    rms = float(np.sqrt(np.mean(out**2)))
    if rms > 0:
        out *= sigma / rms
    return out


def colored_noise_3component(
    n_samples: int,
    dt: float = 60.0,
    alpha: float = 1.2,
    sigma: float = 1.0,
    seed: int | None = None,
    alpha_par: float | None = None,
    sigma_par: float | None = None,
) -> np.ndarray:
    r"""Generate power-law noise for 3 field components.

    Returns shape (n_samples, 3) with columns [B_par, B_perp1, B_perp2].
    B_perp components share ``alpha`` and ``sigma``; B_par can optionally
    use different values (real turbulence has $P_\perp \approx 4 P_\parallel$
    and $\alpha_\parallel > \alpha_\perp$; von Papen et al. 2014).

    Parameters
    ----------
    n_samples, dt, alpha, sigma : same as :func:`power_law_noise`
    seed : int or None
        Base seed; components use seed, seed+1, seed+2.
    alpha_par : float or None
        PSD slope for B_par. Defaults to ``alpha`` (isotropic).
    sigma_par : float or None
        RMS for B_par. Defaults to ``sigma`` (isotropic).
    """
    a_par = alpha_par if alpha_par is not None else alpha
    s_par = sigma_par if sigma_par is not None else sigma

    components = np.empty((n_samples, 3))
    for i in range(3):
        s = seed + i if seed is not None else None
        a = a_par if i == 0 else alpha
        sig = s_par if i == 0 else sigma
        components[:, i] = power_law_noise(n_samples, dt, a, sig, s)
    return components


def magnetospheric_background(
    n_samples: int,
    dt: float = 60.0,
    seed: int | None = None,
    noise_alpha: float = 1.2,
    noise_sigma: float = 0.05,
    noise_alpha_par: float | None = None,
    noise_sigma_par: float | None = None,
    b_mean: float = 5.0,
    slow_trend_amplitude: float = 2.0,
    slow_trend_period_days: float = 5.0,
    ppo_amplitude: float = 0.5,
    ppo_n_period_hours: float = 10.6,
    ppo_s_period_hours: float = 10.8,
) -> np.ndarray:
    r"""Generate a realistic magnetospheric background field.

    Combines:
    1. A mean background field (mostly in B_par)
    2. Power-law colored noise in all 3 components (optionally
       anisotropic: weaker and steeper in B_par)
    3. A slow sinusoidal trend (magnetospheric breathing)
    4. Dual PPO modulation — northern (~10.6 h) and southern (~10.8 h)
       systems with independent random phases, producing beat
       modulation on ~25-day timescales (Andrews et al. 2008, 2010)

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    dt : float
        Sampling interval in seconds.
    noise_alpha : float
        PSD slope for colored noise (B_perp components).
    noise_sigma : float
        RMS of colored noise per B_perp component.
    noise_alpha_par : float or None
        PSD slope for B_par noise. Default: ``noise_alpha + 0.2``.
    noise_sigma_par : float or None
        RMS of B_par noise. Default: ``noise_sigma / 2``.
    b_mean : float
        Mean background field magnitude (nT), placed in B_par.
    slow_trend_amplitude : float
        Amplitude of slow trend (nT).
    slow_trend_period_days : float
        Period of slow trend.
    ppo_amplitude : float
        Amplitude of each PPO system (nT).
    ppo_n_period_hours : float
        Northern PPO period. Default 10.6 h.
    ppo_s_period_hours : float
        Southern PPO period. Default 10.8 h.

    Returns
    -------
    ndarray, shape (n_samples, 3)
        Background field [B_par, B_perp1, B_perp2].
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt

    # Anisotropic noise defaults
    a_par = (
        noise_alpha_par if noise_alpha_par is not None
        else noise_alpha + 0.2
    )
    s_par = (
        noise_sigma_par if noise_sigma_par is not None
        else noise_sigma / 2
    )

    # Colored noise for all 3 components
    noise_seed = int(rng.integers(0, 2**31))
    bg = colored_noise_3component(
        n_samples, dt, noise_alpha, noise_sigma, noise_seed,
        alpha_par=a_par, sigma_par=s_par,
    )

    # Mean field in B_par
    bg[:, 0] += b_mean

    # Slow sinusoidal trend in B_par
    slow_period_sec = slow_trend_period_days * 86400.0
    bg[:, 0] += slow_trend_amplitude * np.sin(
        2 * np.pi * t / slow_period_sec
    )

    # Dual PPO modulation
    if ppo_amplitude > 0:
        inject_ppo(
            bg, t, ppo_amplitude,
            ppo_n_period_hours, ppo_s_period_hours,
            seed=int(rng.integers(0, 2**31)),
        )

    return bg


def inject_ppo(
    bg: np.ndarray,
    t: np.ndarray,
    amplitude: float = 0.5,
    n_period_hours: float = 10.6,
    s_period_hours: float = 10.8,
    seed: int | None = None,
    *,
    realistic: bool = False,
    amp_lognormal_sigma: float = 0.25,
    period_drift_hours: float = 0.1,
    phase_slip_period_days: float = 50.0,
) -> None:
    r"""Add dual N/S PPO modulation to transverse components in-place.

    Northern (~10.6 h) and southern (~10.8 h) PPO systems with
    independent random phases, producing beat modulation on ~25-day
    timescales (Andrews et al. 2008, 2010).

    With ``realistic=True``, three additional features that real PPO
    exhibits but the bare sinusoid does not:

    * **Log-normal amplitude envelope** — the per-cycle amplitude is
      multiplied by ``exp(σ·N(0,1))`` with σ = ``amp_lognormal_sigma``,
      so the magnitude wanders by ~25 % cycle-to-cycle.
    * **Slow period drift** — the PPO period is sinusoidally modulated
      by ±``period_drift_hours`` on the ~50-day system-tracking
      timescale (Provan et al. 2018).
    * **Occasional phase slips** — Brownian phase walk with a
      coherence time of ``phase_slip_period_days`` reproduces the
      occasional re-locking events observed in long PPO records.

    These features push the PPO signal off a clean Fourier delta and
    into a low-Q broadband bump — the regime that actually challenges
    the detector's ability to reject PPO without rejecting QP120.
    """
    rng = np.random.default_rng(seed)
    n_sec = n_period_hours * 3600.0
    s_sec = s_period_hours * 3600.0
    phase_n0 = float(rng.uniform(0, 2 * np.pi))
    phase_s0 = float(rng.uniform(0, 2 * np.pi))

    if not realistic:
        # Original behaviour: pure dual-tone sinusoids.
        for period, phase in [(n_sec, phase_n0), (s_sec, phase_s0)]:
            ppo_arg = 2 * np.pi * t / period + phase
            bg[:, 1] += amplitude * np.sin(ppo_arg)
            bg[:, 2] += 0.3 * amplitude * np.cos(ppo_arg)
        return

    # Realistic regime — same per-component split (B_perp1 full,
    # B_perp2 0.3×) but each PPO system carries a slowly-varying
    # amplitude, period, and phase drift.
    drift_period_sec = phase_slip_period_days * 86400.0
    drift_omega = 2.0 * np.pi / max(drift_period_sec, 1.0)
    for period_sec, phase0 in [(n_sec, phase_n0), (s_sec, phase_s0)]:
        # Per-cycle log-normal amplitude envelope, interpolated to t.
        n_cycles = max(2, int((t[-1] - t[0]) / period_sec) + 2)
        cycle_amps = rng.lognormal(
            mean=0.0, sigma=amp_lognormal_sigma, size=n_cycles,
        )
        cycle_t = t[0] + period_sec * np.arange(n_cycles)
        amp_env = np.interp(t, cycle_t, cycle_amps)

        # Period drift: ±period_drift_hours·3600 over a ~50-day cycle.
        drift_phase = float(rng.uniform(0, 2 * np.pi))
        omega_inst = (2.0 * np.pi / period_sec) * (
            1.0
            - (period_drift_hours * 3600.0 / period_sec)
            * np.sin(drift_omega * t + drift_phase)
        )
        # Brownian phase walk: standard deviation grows like √t with a
        # coherence time of ``phase_slip_period_days``. Increment per
        # sample = sqrt(dt / coherence_sec) · N(0, 1) (in radians).
        dt_local = float(t[1] - t[0]) if len(t) > 1 else 1.0
        slip_inc = math.sqrt(dt_local / drift_period_sec) * rng.standard_normal(t.size)
        slip_phase = np.cumsum(slip_inc)

        # Cumulative instantaneous phase = ∫ω(t)dt + slip + initial.
        inst_phase = np.cumsum(omega_inst) * dt_local + slip_phase + phase0
        bg[:, 1] += amplitude * amp_env * np.sin(inst_phase)
        bg[:, 2] += 0.3 * amplitude * amp_env * np.cos(inst_phase)


def bandlimited_noise_burst(
    n_samples: int,
    dt: float,
    center_sec: float,
    decay_sec: float,
    freq_lo: float,
    freq_hi: float,
    amplitude: float,
    alpha: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    r"""Generate a Gaussian-windowed bandpass-filtered noise burst.

    Produces broadband power between ``freq_lo`` and ``freq_hi`` that
    is spectrally indistinguishable from a continuous noise process in
    a CWT scalogram — unlike discrete sinusoids which produce
    resolvable ridges.

    Parameters
    ----------
    n_samples : int
        Total number of time samples.
    dt : float
        Sampling interval in seconds.
    center_sec : float
        Center of the Gaussian envelope (seconds from t=0).
    decay_sec : float
        Gaussian envelope σ (seconds).
    freq_lo, freq_hi : float
        Bandpass edges in Hz.
    amplitude : float
        Target peak amplitude (nT) of the burst.
    alpha : float
        PSD slope of the underlying noise.
    seed : int or None
        Random seed.

    Returns
    -------
    ndarray, shape (n_samples,)
        Bandpass-filtered, windowed noise burst.
    """
    # Generate colored noise and bandpass filter to target band
    noise = power_law_noise(n_samples, dt, alpha, sigma=1.0, seed=seed)
    fs = 1.0 / dt
    if freq_lo <= 0 or freq_lo >= freq_hi:
        return np.zeros(n_samples)
    noise = bandpass(noise, freq_lo, freq_hi, fs)

    # Gaussian envelope
    t = np.arange(n_samples) * dt
    envelope = np.exp(-0.5 * ((t - center_sec) / decay_sec) ** 2)
    noise *= envelope

    # Normalize by the RMS *inside* the envelope, so the burst's
    # in-band power is controlled by ``amplitude`` regardless of
    # packet duration. Peak-normalization (the old behaviour) scaled
    # with √(log N) and made long bursts spuriously weaker.
    mask = envelope > 0.1 * envelope.max()
    if mask.any():
        rms = float(np.sqrt(np.mean(noise[mask] ** 2)))
        if rms > 0:
            noise *= amplitude / rms

    return noise
