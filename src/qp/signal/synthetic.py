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
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Generate a single waveform component from a WaveTemplate.

    Parameters
    ----------
    t : ndarray
        Time array in seconds.
    wave : WaveTemplate
        Wave parameters.
    rng : Generator, optional
        Random generator for amplitude jitter. Required if
        ``wave.amplitude_jitter > 0``.

    Returns
    -------
    ndarray
        Waveform signal, same length as ``t``.
    """
    f = 1.0 / wave.period
    w = 2.0 * np.pi * f
    time = t - wave.shift

    # Phase with optional linear chirp: phi(t) = w*t + pi*chirp*t^2
    phase_arg = w * time - wave.phase + np.pi * wave.chirp_rate * time**2

    match wave.waveform:
        case "sine":
            y = np.sin(phase_arg)
        case "sawtooth":
            y = sawtooth(phase_arg, width=wave.sawtooth_width)
        case "square":
            y = square(phase_arg, duty=0.2)
        case _:
            raise ValueError(f"Unknown waveform type: {wave.waveform!r}")

    # Add 2nd harmonic content
    if wave.harmonic_content > 0:
        y += wave.harmonic_content * np.sin(2 * phase_arg)

    # Envelope (symmetric or asymmetric Gaussian)
    if wave.decay_width is not None:
        if wave.asymmetry != 0.5:
            # asymmetry < 0.5 → fast rise (small sigma_left), slow fall
            # asymmetry > 0.5 → slow rise, fast fall
            sigma_left = wave.decay_width * (0.5 + wave.asymmetry)
            sigma_right = wave.decay_width * (1.5 - wave.asymmetry)
            sigma = np.where(time < 0, sigma_left, sigma_right)
            envelope = np.exp(-0.5 * (time / sigma) ** 2)
        else:
            envelope = np.exp(-0.5 * (time / wave.decay_width) ** 2)
        y *= envelope

    # Per-cycle amplitude jitter
    if wave.amplitude_jitter > 0 and rng is not None:
        samples_per_cycle = max(1, int(round(wave.period / (t[1] - t[0]))))
        n_cycles = len(t) // samples_per_cycle + 1
        jitter = 1.0 + wave.amplitude_jitter * rng.standard_normal(n_cycles)
        jitter_expanded = np.repeat(jitter, samples_per_cycle)[: len(t)]
        y *= jitter_expanded

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
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt
    y = np.zeros(n_samples)

    if waves is not None:
        for wave in waves:
            y += _generate_waveform(t, wave, rng)

    if noise_sigma > 0:
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

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt
    components = np.zeros((n_samples, 3))

    if waves is not None:
        for i, offset in enumerate(phase_offsets):
            for wave in waves:
                shifted = replace(wave, phase=wave.phase + offset)
                components[:, i] += _generate_waveform(t, shifted, rng)

    if noise_sigma > 0:
        components += rng.normal(0, noise_sigma, components.shape)

    b_tot = np.linalg.norm(components, axis=1)
    fields = np.column_stack([components, b_tot])

    return t, fields


def simulate_wave_physics(
    n_samples: int,
    dt: float,
    waves: list[WaveTemplate],
    mode: str = "alfvenic",
    polarization: str = "circular",
    ellipticity: float = 1.0,
    propagation: str = "standing",
    par_leakage: float = 0.05,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate a 3-component field with physically motivated wave modes.

    Unlike :func:`simulate_multi_component` which uses naive phase offsets,
    this function models the physics of Alfvénic vs compressional modes,
    circular vs linear polarization, and standing vs travelling propagation.

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    dt : float
        Sampling interval in seconds.
    waves : list[WaveTemplate]
        Wave packets to inject. The ``amplitude`` sets the peak transverse
        (Alfvénic) or parallel (compressional) perturbation.
    mode : str
        ``"alfvenic"`` — transverse oscillation, small B_par.
        ``"compressional"`` — parallel oscillation, small B_perp.
        ``"mixed"`` — comparable power in all components.
    polarization : str
        ``"circular"`` — B_perp2 leads B_perp1 by 90°.
        ``"linear"`` — B_perp2 = 0 (all power in B_perp1).
        ``"elliptical"`` — controlled by ``ellipticity``.
    ellipticity : float
        Minor/major axis ratio of the polarization ellipse, [-1, 1].
        Only used when ``polarization="elliptical"``.
        +1 = right-circular, -1 = left-circular, 0 = linear.
    propagation : str
        ``"standing"`` — uses waves as-is (chirp_rate=0 expected).
        ``"travelling"`` — uses waves as-is (nonzero chirp_rate expected).
        The distinction is encoded in the WaveTemplate chirp_rate and
        asymmetry fields; this parameter is metadata for the manifest.
    par_leakage : float
        Fraction of transverse amplitude that leaks into B_par (Alfvénic)
        or fraction of parallel amplitude leaking into B_perp (compressional).
        Models imperfect MFA rotation. Default 0.05 (5%).
    seed : int or None
        RNG seed for jitter.

    Returns
    -------
    time : ndarray, shape (n_samples,)
    fields : ndarray, shape (n_samples, 4)
        Columns: [B_par, B_perp1, B_perp2, B_tot].
    """
    from copy import replace as copy_replace

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt
    b_par = np.zeros(n_samples)
    b_perp1 = np.zeros(n_samples)
    b_perp2 = np.zeros(n_samples)

    for wave in waves:
        # Base waveform (B_perp1 axis)
        w1 = _generate_waveform(t, wave, rng)

        # B_perp2 depends on polarization
        match polarization:
            case "circular":
                w2_template = copy_replace(wave, phase=wave.phase - np.pi / 2)
                w2 = _generate_waveform(t, w2_template, rng)
            case "linear":
                w2 = np.zeros_like(w1)
            case "elliptical":
                w2_template = copy_replace(wave, phase=wave.phase - np.pi / 2)
                w2_full = _generate_waveform(t, w2_template, rng)
                w2 = abs(ellipticity) * w2_full
                if ellipticity < 0:
                    w2 = -w2  # left-handed
            case _:
                raise ValueError(f"Unknown polarization: {polarization!r}")

        # Distribute into components based on wave mode
        match mode:
            case "alfvenic":
                b_perp1 += w1
                b_perp2 += w2
                b_par += par_leakage * w1  # small leakage
            case "compressional":
                b_par += w1
                b_perp1 += par_leakage * w1  # small leakage
                b_perp2 += par_leakage * w2 if np.any(w2) else np.zeros_like(w1)
            case "mixed":
                scale = 1.0 / np.sqrt(3)
                b_par += w1 * scale
                b_perp1 += w1 * scale
                b_perp2 += w2 * scale
            case _:
                raise ValueError(f"Unknown wave mode: {mode!r}")

    b_tot = np.sqrt(b_par**2 + b_perp1**2 + b_perp2**2)
    fields = np.column_stack([b_par, b_perp1, b_perp2, b_tot])
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
