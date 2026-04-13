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

    # Timmer-König: independent Gaussian draws for real and imaginary
    # parts, each scaled by sqrt(P(f)/2)
    spectrum = (
        rng.standard_normal(len(freqs))
        + 1j * rng.standard_normal(len(freqs))
    ) * amplitude / np.sqrt(2)

    # Nyquist bin must be real for even-length signals
    if n_gen % 2 == 0:
        spectrum[-1] = spectrum[-1].real

    noise = np.fft.irfft(spectrum, n=n_gen)

    # Extract central segment for oversampled steep spectra
    if n_gen > n_samples:
        start = (n_gen - n_samples) // 2
        noise = noise[start : start + n_samples]

    # Normalize to requested sigma
    current_rms = np.sqrt(np.mean(noise**2))
    if current_rms > 0:
        noise *= sigma / current_rms

    return noise


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
) -> None:
    r"""Add dual N/S PPO modulation to transverse components in-place.

    Northern (~10.6 h) and southern (~10.8 h) PPO systems with
    independent random phases, producing beat modulation on ~25-day
    timescales (Andrews et al. 2008, 2010).
    """
    rng = np.random.default_rng(seed)
    n_sec = n_period_hours * 3600.0
    s_sec = s_period_hours * 3600.0
    phase_n = rng.uniform(0, 2 * np.pi)
    phase_s = rng.uniform(0, 2 * np.pi)

    # B_perp1 gets full PPO, B_perp2 gets 0.3× (elliptical polarization)
    for period, phase in [(n_sec, phase_n), (s_sec, phase_s)]:
        ppo_arg = 2 * np.pi * t / period + phase
        bg[:, 1] += amplitude * np.sin(ppo_arg)
        bg[:, 2] += 0.3 * amplitude * np.cos(ppo_arg)


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

    # Scale to target amplitude
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise *= amplitude / peak

    return noise
