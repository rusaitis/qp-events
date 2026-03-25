r"""Synthetic signal generation for testing and validation.

Replaces ``cassinilib/NewSignal.py:simulateSignal()`` and
``generateLongSignal()`` with clean functions that return numpy arrays
instead of lists of ``NewSignal`` objects.

Uses ``WaveTemplate`` from ``qp.events.catalog`` for wave specification.

Usage
-----
>>> from qp.signal.synthetic import simulate_signal
>>> t, y = simulate_signal(n_samples=2160, dt=60.0, waves=[
...     WaveTemplate(period=3600, amplitude=2.0),
... ])
"""

from __future__ import annotations

import numpy as np
from scipy.signal import sawtooth, square

from qp.events.catalog import WaveTemplate


def _generate_waveform(
    t: np.ndarray,
    wave: WaveTemplate,
) -> np.ndarray:
    r"""Generate a single waveform component from a WaveTemplate.

    Parameters
    ----------
    t : ndarray
        Time array in seconds.
    wave : WaveTemplate
        Wave parameters.

    Returns
    -------
    ndarray
        Waveform signal, same length as ``t``.
    """
    f = 1.0 / wave.period
    w = 2.0 * np.pi * f
    time = t - wave.shift

    match wave.waveform:
        case "sine":
            y = np.sin(w * time - wave.phase)
        case "sawtooth":
            y = sawtooth(w * time - wave.phase, width=0.8)
        case "square":
            y = square(w * time - wave.phase, duty=0.2)
        case _:
            raise ValueError(f"Unknown waveform type: {wave.waveform!r}")

    # Gaussian envelope for amplitude decay
    if wave.decay_width is not None:
        envelope = np.exp(-0.5 * (time / wave.decay_width) ** 2)
        y *= envelope

    # Amplitude scaling
    y *= wave.amplitude

    # Time cutoff
    if wave.cutoff is not None:
        start_sec, end_sec = wave.cutoff
        y[t < start_sec] = 0.0
        y[t > end_sec] = 0.0

    return y


def simulate_signal(
    n_samples: int = 2160,
    dt: float = 60.0,
    waves: list[WaveTemplate] | None = None,
    noise_sigma: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate a synthetic time series with specified wave content.

    Replaces ``cassinilib.simulateSignal()`` — returns plain arrays instead
    of a list of ``NewSignal`` objects.

    Parameters
    ----------
    n_samples : int
        Number of time samples (default 2160 = 36 hours at 1-min cadence).
    dt : float
        Sampling interval in seconds.
    waves : list[WaveTemplate], optional
        Wave components to inject. If None, returns zero signal (+ noise).
    noise_sigma : float
        Standard deviation of additive Gaussian noise.
    seed : int, optional
        RNG seed for reproducible noise.

    Returns
    -------
    time : ndarray, shape (n_samples,)
        Time array in seconds from t=0.
    signal : ndarray, shape (n_samples,)
        Synthetic signal.
    """
    t = np.arange(n_samples) * dt
    y = np.zeros(n_samples)

    if waves is not None:
        for wave in waves:
            y += _generate_waveform(t, wave)

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        y += rng.normal(0, noise_sigma, n_samples)

    return t, y


def simulate_multi_component(
    n_samples: int = 2160,
    dt: float = 60.0,
    waves: list[WaveTemplate] | None = None,
    noise_sigma: float = 0.0,
    phase_offsets: tuple[float, float, float] = (0.0, np.pi / 4, np.pi / 2),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate synthetic 3-component field data (B_par, B_perp1, B_perp2).

    Each component gets the same waves but with a phase offset, mimicking
    the circularly polarized QP events observed at Saturn.

    Replaces the multi-component logic in ``cassinilib.simulateSignal()``.

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    dt : float
        Sampling interval in seconds.
    waves : list[WaveTemplate], optional
        Wave components to inject.
    noise_sigma : float
        Standard deviation of additive Gaussian noise per component.
    phase_offsets : tuple[float, float, float]
        Phase offsets for (par, perp1, perp2) components in radians.
        Default (0, pi/4, pi/2) mimics circular polarization.
    seed : int, optional
        RNG seed for reproducible noise.

    Returns
    -------
    time : ndarray, shape (n_samples,)
        Time array in seconds from t=0.
    fields : ndarray, shape (n_samples, 4)
        Columns: [B_par, B_perp1, B_perp2, B_tot].
    """
    from copy import replace

    t = np.arange(n_samples) * dt
    components = np.zeros((n_samples, 3))

    if waves is not None:
        for i, offset in enumerate(phase_offsets):
            for wave in waves:
                shifted = replace(wave, phase=wave.phase + offset)
                components[:, i] += _generate_waveform(t, shifted)

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        components += rng.normal(0, noise_sigma, components.shape)

    b_tot = np.linalg.norm(components, axis=1)
    fields = np.column_stack([components, b_tot])

    return t, fields


def generate_long_signal(
    duration_days: float = 10.0,
    dt: float = 60.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate a long synthetic signal with realistic multi-scale wave content.

    Includes:
    - Long-period background trend (~200 days, large amplitude)
    - PPO modulation (~10.7 hours)
    - Sporadic QP60 wave packets (~60 min, scattered every ~55 hours)
    - Sporadic QP30 wave packets (~30 min, scattered every ~72 hours)
    - Random noise

    This replaces ``cassinilib.generateLongSignal()`` for pipeline testing.

    Parameters
    ----------
    duration_days : float
        Total signal duration in days.
    dt : float
        Sampling interval in seconds.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    time : ndarray
        Time array in seconds from t=0.
    signal : ndarray
        Synthetic signal.
    """
    hour = 3600.0
    mins = 60.0
    n_samples = int(duration_days * 86400 / dt)
    rng = np.random.default_rng(seed)

    # Collect wave packets
    waves: list[WaveTemplate] = []

    # Background trend (long period, large amplitude)
    waves.append(
        WaveTemplate(
            period=200 * 24 * hour,
            amplitude=1000.0,
            phase=0.0,
        )
    )

    # PPO modulation (~10.7 hours)
    waves.append(
        WaveTemplate(
            period=10.7 * hour,
            amplitude=2.0,
            phase=0.0,
        )
    )

    # Sporadic QP60 wave packets (every ~55 hours, Gaussian-enveloped)
    total_sec = duration_days * 86400
    shift = 0.0
    while shift < total_sec:
        shift += 55 * hour + rng.normal(0, 55 * hour / 2)
        if shift >= total_sec:
            break
        waves.append(
            WaveTemplate(
                period=60 * mins + rng.normal(0, 6 * mins),
                amplitude=0.02 + rng.normal(0, 0.01),
                phase=rng.uniform(0, np.pi),
                decay_width=3 * hour,
                shift=shift,
            )
        )

    # Sporadic QP30 wave packets (every ~72 hours)
    shift = 0.0
    while shift < total_sec:
        shift += 72 * hour + rng.normal(0, 36 * hour)
        if shift >= total_sec:
            break
        waves.append(
            WaveTemplate(
                period=30 * mins + rng.normal(0, 3 * mins),
                amplitude=0.02 + rng.normal(0, 0.01),
                phase=rng.uniform(0, np.pi),
                decay_width=1 * hour,
                shift=shift,
            )
        )

    t, y = simulate_signal(
        n_samples=n_samples,
        dt=dt,
        waves=waves,
        noise_sigma=0.01,
        seed=seed,
    )
    return t, y
