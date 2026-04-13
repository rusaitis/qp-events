r"""Colored and power-law noise generation.

Real Cassini MAG data has a power spectral density that follows a power
law $P(f) \propto f^{-\alpha}$ with $\alpha \approx 1.0$–$1.5$ in the
outer magnetosphere. White Gaussian noise ($\alpha = 0$) is unrealistic
for benchmarking event detection.

The Timmer–König (1995) method generates noise with a prescribed PSD
shape by drawing complex Gaussian random numbers in Fourier space,
scaling their amplitude by $f^{-\alpha/2}$, and inverse-transforming.

References
----------
Timmer, J. & König, M. (1995). "On generating power law noise."
*Astronomy and Astrophysics*, 300, 707.
"""

from __future__ import annotations

import numpy as np


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
        Real Cassini magnetospheric noise: $\alpha \approx 1.0$–$1.5$.
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

    n_fft = n_samples
    freqs = np.fft.rfftfreq(n_fft, d=dt)

    # Power scaling: P(f) ∝ f^{-alpha}, so amplitude ∝ f^{-alpha/2}
    amplitude = np.ones_like(freqs)
    amplitude[1:] = freqs[1:] ** (-alpha / 2)
    amplitude[0] = 0.0  # zero DC component

    # Random complex spectrum
    phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    magnitudes = rng.standard_normal(size=len(freqs)) * amplitude
    spectrum = magnitudes * np.exp(1j * phases)

    # Nyquist bin must be real for even-length signals
    if n_fft % 2 == 0:
        spectrum[-1] = spectrum[-1].real

    noise = np.fft.irfft(spectrum, n=n_fft)

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
) -> np.ndarray:
    r"""Generate independent power-law noise for 3 field components.

    Returns shape (n_samples, 3) with columns [B_par, B_perp1, B_perp2],
    each an independent realization of power-law noise with the same
    spectral slope and RMS.

    Parameters
    ----------
    n_samples, dt, alpha, sigma : same as :func:`power_law_noise`
    seed : int or None
        Base seed; components use seed, seed+1, seed+2.
    """
    components = np.empty((n_samples, 3))
    for i in range(3):
        s = seed + i if seed is not None else None
        components[:, i] = power_law_noise(n_samples, dt, alpha, sigma, s)
    return components


def magnetospheric_background(
    n_samples: int,
    dt: float = 60.0,
    seed: int | None = None,
    noise_alpha: float = 1.2,
    noise_sigma: float = 0.05,
    b_mean: float = 5.0,
    slow_trend_amplitude: float = 2.0,
    slow_trend_period_days: float = 5.0,
    ppo_amplitude: float = 0.5,
    ppo_period_hours: float = 10.7,
) -> np.ndarray:
    r"""Generate a realistic magnetospheric background field.

    Combines:
    1. A mean background field (mostly in B_par)
    2. Power-law colored noise in all 3 components
    3. A slow sinusoidal trend (magnetospheric breathing)
    4. PPO modulation at Saturn's planetary period (~10.7 h)

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    dt : float
        Sampling interval in seconds.
    noise_alpha : float
        PSD slope for colored noise.
    noise_sigma : float
        RMS of colored noise per component.
    b_mean : float
        Mean background field magnitude (nT), placed in B_par.
    slow_trend_amplitude : float
        Amplitude of slow trend (nT).
    slow_trend_period_days : float
        Period of slow trend.
    ppo_amplitude : float
        Amplitude of PPO modulation (nT).
    ppo_period_hours : float
        PPO period (hours). Default 10.7 h for Saturn.

    Returns
    -------
    ndarray, shape (n_samples, 3)
        Background field [B_par, B_perp1, B_perp2].
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt

    # Colored noise for all 3 components
    noise_seed = int(rng.integers(0, 2**31))
    bg = colored_noise_3component(n_samples, dt, noise_alpha, noise_sigma, noise_seed)

    # Mean field in B_par
    bg[:, 0] += b_mean

    # Slow sinusoidal trend in B_par
    slow_period_sec = slow_trend_period_days * 86400.0
    bg[:, 0] += slow_trend_amplitude * np.sin(2 * np.pi * t / slow_period_sec)

    # PPO modulation (primarily in B_perp1, weaker in B_perp2)
    ppo_period_sec = ppo_period_hours * 3600.0
    ppo_phase = rng.uniform(0, 2 * np.pi)
    bg[:, 1] += ppo_amplitude * np.sin(2 * np.pi * t / ppo_period_sec + ppo_phase)
    bg[:, 2] += 0.3 * ppo_amplitude * np.cos(
        2 * np.pi * t / ppo_period_sec + ppo_phase
    )

    return bg
