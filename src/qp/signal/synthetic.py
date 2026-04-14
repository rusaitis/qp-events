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
from scipy.signal import lfilter, sawtooth, square

from qp.events.catalog import WaveTemplate

# Fraction of Nyquist retained by the sharp-waveform anti-alias filter.
# Set close-to-but-below 1 so the guard band sits in the stop-band
# rolloff rather than the detector's analysis range.
_ANTIALIAS_NYQUIST_FRACTION = 0.8


def _bandlimit(signal: np.ndarray, dt: float, fc: float) -> np.ndarray:
    r"""Zero-phase low-pass via rfft amplitude mask at cutoff ``fc``.

    Sharp non-sine waveforms (sawtooth, square) contain power at all
    harmonics of the fundamental; at dt=60 s these alias across the
    full Nyquist band. We truncate the spectrum above ``fc`` so the
    waveform drops the aliasing tail without biting into the detector
    analysis band — the cutoff is set from the Nyquist frequency, not
    from the wave fundamental, so it is band-agnostic.
    """
    n = signal.size
    if n < 4 or fc <= 0:
        return signal
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.fft.rfft(signal)
    spec[freqs > fc] = 0.0
    return np.fft.irfft(spec, n=n)


def _ou_log_jitter(
    n: int, dt: float, tau: float, sigma: float, rng: np.random.Generator,
) -> np.ndarray:
    r"""Ornstein–Uhlenbeck multiplicative amplitude jitter.

    Returns a length-``n`` positive multiplier with unit mean and
    stationary log-variance ``sigma**2``. The log-amplitude follows a
    discrete OU process with correlation time ``tau``; spectral content
    rolls off above $f \sim 1/(2\pi\tau)$ so the jitter injects no
    broadband power at integer-cycle boundaries (unlike the historical
    per-cycle step multiplier).
    """
    if sigma <= 0 or n == 0:
        return np.ones(n)
    alpha = float(np.exp(-dt / tau))
    drive = np.sqrt(1.0 - alpha * alpha) * rng.standard_normal(n)
    # Warm-start from the stationary distribution so there is no
    # transient at t=0.
    drive[0] = rng.standard_normal()
    x = lfilter([1.0], [1.0, -alpha], drive)
    # Bias-correct the log-normal so E[multiplier] = 1 exactly.
    return np.exp(sigma * x - 0.5 * sigma * sigma)


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

    # Anti-alias non-sine waveforms. Cutoff is tied to the Nyquist
    # frequency of the sample grid (``_ANTIALIAS_NYQUIST_FRACTION`` ×
    # f_Nyq), not to the wave fundamental — this keeps the guard band
    # inside the detector rolloff regardless of injection period and
    # avoids the old ``10 × f0`` rule that landed inside the search
    # band for QP120 and near Nyquist for QP30.
    if wave.waveform in ("sawtooth", "square") and len(t) > 1:
        dt_sample = float(t[1] - t[0])
        f_nyq = 0.5 / dt_sample
        y = _bandlimit(y, dt_sample, fc=_ANTIALIAS_NYQUIST_FRACTION * f_nyq)

    # Harmonic generation. Two models are supported:
    #   * ``"linear_2f"`` (default) — phenomenological nuisance
    #     harmonic at 2f with a random Hilbert-phase lag, scaled
    #     linearly by ``harmonic_content``. Easy to reason about
    #     but does not enforce the phase relationship of nonlinear
    #     wave steepening.
    #   * ``"sawtooth_truncated"`` — partial Fourier series of a
    #     sawtooth wave truncated at 10 harmonics with phases tied
    #     to the fundamental (a_n = (-1)^{n+1}/n). ``harmonic_content``
    #     sets the steepening fraction: 0 → pure sine, 1 → fully
    #     steepened sawtooth shape. Phases are deterministic from
    #     the fundamental, matching real MHD wave steepening.
    if wave.harmonic_content > 0:
        if wave.harmonic_model == "linear_2f":
            if rng is not None:
                harm_phase = float(rng.uniform(-np.pi, np.pi))
            else:
                harm_phase = 0.0
            y += wave.harmonic_content * np.sin(2 * phase_arg + harm_phase)
        elif wave.harmonic_model == "sawtooth_truncated":
            harm_sum = np.zeros_like(y)
            n_harm = 10
            for n in range(2, n_harm + 1):
                # Fourier coefficient of a sawtooth: 2/(πn) (-1)^{n+1}.
                # We feed n=1 from ``y`` already and add n≥2 here, scaled
                # by ``harmonic_content`` so a value of 1 yields the
                # full sawtooth Fourier-series amplitude relative to
                # the fundamental.
                coeff = 2.0 / (np.pi * n) * ((-1.0) ** (n + 1))
                harm_sum += coeff * np.sin(n * phase_arg)
            # Normalise so that ``harmonic_content`` ≈ steepening
            # fraction: a value of 1 reproduces the truncated-sawtooth
            # amplitude that a fully steepened wave would have, given
            # that the fundamental coefficient of a sawtooth is 2/π.
            y += wave.harmonic_content * (np.pi / 2.0) * harm_sum
            # Anti-alias at the same Nyquist-fraction cutoff as above.
            if len(t) > 1:
                dt_sample = float(t[1] - t[0])
                f_nyq = 0.5 / dt_sample
                y = _bandlimit(
                    y, dt_sample, fc=_ANTIALIAS_NYQUIST_FRACTION * f_nyq
                )
        else:
            raise ValueError(
                f"Unknown harmonic_model: {wave.harmonic_model!r}"
            )

    # Envelope. Three shapes are supported, all peak-normalised to 1:
    #   * ``"gaussian"`` — exp(-t²/2σ²), default, with optional
    #     erf-blended left/right asymmetry (smooth at t=0 — the old
    #     ``np.where`` switch had a derivative kink that injected a
    #     broadband spike at the event centre).
    #   * ``"lognormal"`` — fast rise, slow decay with a heavy tail.
    #     Matches packets that grow out of a noise floor and unwind.
    #   * ``"rayleigh"`` — sharp rise, slower fall than Gaussian.
    # ``decay_width`` is the characteristic scale (Gaussian σ for
    # ``gaussian`` / ``rayleigh``, mode-distance for ``lognormal``).
    if wave.decay_width is not None:
        shape = wave.envelope_shape
        if shape == "gaussian":
            if wave.asymmetry != 0.5:
                from scipy.special import erf
                sigma_left = wave.decay_width * (0.5 + wave.asymmetry)
                sigma_right = wave.decay_width * (1.5 - wave.asymmetry)
                tau = 0.05 * min(sigma_left, sigma_right)
                w = 0.5 * (1.0 + erf(time / max(tau, 1e-9)))
                sigma = (1.0 - w) * sigma_left + w * sigma_right
                envelope = np.exp(-0.5 * (time / sigma) ** 2)
            else:
                envelope = np.exp(-0.5 * (time / wave.decay_width) ** 2)
        elif shape == "lognormal":
            # Place the mode at t=0 by setting τ = 1 + t/decay_width
            # and using a shifted log-normal kernel whose mode in τ
            # is at τ=1. The standard log-normal kernel
            # exp(-(log τ - μ)²/(2s²)) has its mode at exp(μ - s²);
            # choose μ = s² so the mode in τ is at 1, i.e. at t = 0.
            # Then the envelope rises sharply for t < 0 and decays
            # with a heavy right tail for t > 0.
            s = 0.7
            mu = s * s
            tau = 1.0 + time / wave.decay_width
            envelope = np.where(
                tau > 0,
                np.exp(
                    -0.5 * ((np.log(np.maximum(tau, 1e-9)) - mu) / s) ** 2
                ),
                0.0,
            )
        elif shape == "rayleigh":
            # Rayleigh-like envelope: peak at t = decay_width, zero
            # at t = -decay_width, monotone decay for t > peak.
            tau = (time + wave.decay_width) / wave.decay_width
            envelope = np.where(
                tau > 0,
                tau * np.exp(-0.5 * (tau**2 - 1.0)),
                0.0,
            )
        else:
            raise ValueError(f"Unknown envelope_shape: {shape!r}")
        y *= envelope

    # Amplitude jitter. Historical versions multiplied by a step
    # function resampled once per cycle, which injects broadband
    # spectral power at cycle boundaries that the detector can see as
    # transient signal. Instead we use an Ornstein–Uhlenbeck-driven
    # log-normal multiplier with correlation time equal to one period
    # — stationary unit mean, log-variance ``amplitude_jitter**2``,
    # smooth across cycle boundaries, and well-defined PSD rolloff.
    if wave.amplitude_jitter > 0 and rng is not None and len(t) > 1:
        dt_sample = float(t[1] - t[0])
        y *= _ou_log_jitter(
            n=len(t),
            dt=dt_sample,
            tau=wave.period,
            sigma=wave.amplitude_jitter,
            rng=rng,
        )

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


def simulate_wave_physics(
    n_samples: int,
    dt: float,
    waves: list[WaveTemplate],
    mode: str = "alfvenic",
    polarization: str = "circular",
    ellipticity: float = 1.0,
    par_leakage: float | tuple[float, float] = (0.0, 0.10),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate a 3-component field with physically motivated wave modes.

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
    par_leakage : float or tuple[float, float]
        MFA cross-axis leakage: fraction of the primary-axis amplitude
        that appears on the secondary axis (B_par for Alfvénic, B_perp
        for compressional). A scalar is applied deterministically to
        every event; a ``(lo, hi)`` tuple draws ``U(lo, hi)`` per event,
        which is the published default ``(0.0, 0.10)`` — it reflects
        MFA rotation uncertainty rather than assuming a single magic
        value. Set to ``0.0`` to switch leakage off.
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

    if isinstance(par_leakage, tuple):
        leakage_lo, leakage_hi = par_leakage
        if leakage_lo < 0 or leakage_hi < leakage_lo:
            raise ValueError(
                f"par_leakage range must satisfy 0 ≤ lo ≤ hi; got {par_leakage!r}"
            )
    else:
        leakage_lo = leakage_hi = float(par_leakage)

    for wave in waves:
        leakage = float(rng.uniform(leakage_lo, leakage_hi)) \
            if leakage_hi > leakage_lo else leakage_lo

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
                b_par += leakage * w1
            case "compressional":
                b_par += w1
                b_perp1 += leakage * w1
                b_perp2 += leakage * w2 if np.any(w2) else np.zeros_like(w1)
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
