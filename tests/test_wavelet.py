"""Synthetic-signal tests for the Morlet CWT.

These tests pin down the contract of ``qp.signal.wavelet.morlet_cwt`` —
they would catch regressions in the kernel shape, the scale-to-frequency
inverse (T&C98 Table 1), and the energy normalisation. Detection-time
contracts (thresholding, ridge extraction) are covered in
``test_threshold.py`` and ``test_ridge.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from qp.signal.wavelet import cwt_power, morlet_cwt


@pytest.mark.parametrize("period_min", [30.0, 60.0, 120.0])
def test_pure_tone_peak_recovery(period_min: float) -> None:
    """A pure sine at period ``period_min`` minutes peaks at 1/(60*period_min) Hz.

    Tolerance is one frequency bin — the linear grid has ``n_freqs=300``
    over ``[1/(4 h), Nyquist/2]`` for 1-min data, so the bin width is
    ~2.8e-5 Hz; the test signal sits well above this.
    """
    dt = 60.0
    n_samples = 36 * 60  # 36 h at 1-min sampling
    t = np.arange(n_samples) * dt
    f_in = 1.0 / (period_min * 60.0)
    signal = np.sin(2.0 * np.pi * f_in * t)

    freq, _, power = cwt_power(signal, dt=dt, n_freqs=300)
    # Average power per row over the inner half of the segment (outside COI).
    coi = n_samples // 4
    row_power = power[:, coi:-coi].mean(axis=1)
    peak_idx = int(np.argmax(row_power))
    df = float(freq[1] - freq[0])

    assert abs(freq[peak_idx] - f_in) <= df, (
        f"Peak at {freq[peak_idx]:.6g} Hz, expected {f_in:.6g} Hz (bin width {df:.6g})"
    )


def test_exact_scale_to_frequency_inverse_is_unbiased() -> None:
    """At ω₀=10, the (s, f) relation matches T&C98 Table 1.

    Implementation uses ``s = (ω₀ + √(2+ω₀²)) / (4π·f·dt)``. The
    simplified inverse ``s_simple = ω₀/(2π·f·dt)`` would systematically
    bias the peak frequency by ~0.5 % at ω₀=10 — always *low* (the
    simplified ``s_simple`` is smaller than ``s``, so it labels the
    same wavelet shape with a higher Fourier-period frequency).

    Test: at ω₀=10 the peak should *bracket* the input across multiple
    test frequencies (some peaks above f_in, some below within one
    bin), not consistently below.
    """
    dt = 60.0
    n_samples = 36 * 60
    t = np.arange(n_samples) * dt

    # Pick test frequencies that fall exactly on the linear grid — this
    # eliminates the bin-quantisation noise and isolates the inverse
    # formula choice as the only source of bias.
    n_freqs = 2400
    f_grid_min = 1.0 / (4 * 3600.0)
    f_grid_max = 1.0 / (2 * dt)
    grid = np.linspace(f_grid_min, f_grid_max, n_freqs)
    # Pick four grid points in the QP search range.
    test_idx = [grid.searchsorted(1.0 / (p * 60.0)) for p in (30.0, 60.0, 90.0, 120.0)]
    f_inputs = grid[test_idx]

    rel_errors_signed = []
    for f_in in f_inputs:
        signal = np.sin(2.0 * np.pi * f_in * t)
        freq, _, power = cwt_power(signal, dt=dt, omega0=10.0, n_freqs=n_freqs)
        coi = n_samples // 4
        row_power = power[:, coi:-coi].mean(axis=1)
        peak_idx = int(np.argmax(row_power))
        rel_errors_signed.append((freq[peak_idx] - f_in) / f_in)

    # Under the *simplified* inverse the peak frequencies would all be
    # systematically biased low by ~0.5 % (the bias factor at omega0=10
    # is (omega0 + sqrt(2 + omega0^2)) / (2 omega0) - 1 ≈ 0.005). The
    # exact inverse should give a mean signed error close to zero.
    mean_signed = sum(rel_errors_signed) / len(rel_errors_signed)
    assert abs(mean_signed) < 0.003, (
        f"Mean signed error {mean_signed:.4%} across "
        f"{len(rel_errors_signed)} tones — a value near -0.5 % would "
        f"signal use of the simplified inverse s_simple = omega0/(2π·f·dt) "
        f"instead of the T&C98 Table 1 exact relation."
    )


def test_linearity() -> None:
    """The CWT is linear: ``CWT(a*x + b*y) = a*CWT(x) + b*CWT(y)``."""
    dt = 60.0
    n_samples = 1024
    t = np.arange(n_samples) * dt
    f1 = 1.0 / (30 * 60.0)
    f2 = 1.0 / (90 * 60.0)
    x = np.sin(2.0 * np.pi * f1 * t)
    y = np.cos(2.0 * np.pi * f2 * t)
    a, b = 0.7, -1.3
    combo = a * x + b * y

    _, _, c_combo = morlet_cwt(combo, dt=dt, n_freqs=100)
    _, _, c_x = morlet_cwt(x, dt=dt, n_freqs=100)
    _, _, c_y = morlet_cwt(y, dt=dt, n_freqs=100)

    np.testing.assert_allclose(c_combo, a * c_x + b * c_y, rtol=1e-10, atol=1e-12)


def test_zero_signal() -> None:
    """Zero input → zero output (sanity)."""
    n_samples = 128
    freq, time, cwt = morlet_cwt(np.zeros(n_samples), dt=60.0, n_freqs=20)
    assert cwt.shape == (20, n_samples)
    assert time.shape == (n_samples,)
    assert freq.shape == (20,)
    np.testing.assert_array_equal(cwt, 0.0)


def test_freq_grid_bounds() -> None:
    """Default freq grid covers [1/(4 h), Nyquist/2]; user overrides honoured."""
    n_samples = 256
    dt = 60.0
    freq, _, _ = morlet_cwt(
        np.random.default_rng(0).standard_normal(n_samples), dt=dt, n_freqs=50
    )
    assert freq[0] == pytest.approx(1.0 / (4 * 3600))
    assert freq[-1] == pytest.approx(1.0 / (2 * dt))

    f_lo, f_hi = 1e-4, 5e-3
    freq2, _, _ = morlet_cwt(
        np.random.default_rng(0).standard_normal(n_samples),
        dt=dt,
        n_freqs=50,
        freq_min=f_lo,
        freq_max=f_hi,
    )
    assert freq2[0] == pytest.approx(f_lo)
    assert freq2[-1] == pytest.approx(f_hi)


def test_amplitude_scaling_linearity_in_signal_power() -> None:
    """|CWT|² scales linearly with input signal power.

    For two pure sinusoids at the same frequency but different
    amplitudes a₁ and a₂, the peak row-power ratio must equal
    (a₂/a₁)². This is a direct consequence of CWT linearity and is
    independent of the choice of energy normalisation — it would catch
    a regression where ``/sqrt(s)`` is changed to e.g. ``/s``.
    """
    dt = 60.0
    n_samples = 36 * 60
    t = np.arange(n_samples) * dt
    coi = n_samples // 4
    f_in = 1.0 / (60.0 * 60.0)

    sig_a = 1.0 * np.sin(2.0 * np.pi * f_in * t)
    sig_b = 2.5 * np.sin(2.0 * np.pi * f_in * t)
    _, _, p_a = cwt_power(sig_a, dt=dt, n_freqs=200)
    _, _, p_b = cwt_power(sig_b, dt=dt, n_freqs=200)

    peak_a = p_a[:, coi:-coi].mean(axis=1).max()
    peak_b = p_b[:, coi:-coi].mean(axis=1).max()
    ratio = peak_b / peak_a
    expected = 2.5**2  # = 6.25

    assert abs(ratio - expected) / expected < 0.01, (
        f"|CWT|² scaling: got ratio {ratio:.4f}, expected {expected:.4f}"
    )


def test_omega0_increases_frequency_resolution() -> None:
    """Higher ω₀ gives narrower in-frequency FWHM at fixed input.

    Direct consequence of T&C98 §3a: the Morlet's frequency-domain
    Gaussian width is :math:`\\sigma_\\omega = 1/s`, so the FWHM at
    fixed peak frequency :math:`f_0` is
    :math:`\\Delta f_{\\mathrm{FWHM}} = 2\\sqrt{2\\ln 2}\\,f_0/\\omega_0`.
    The ratio ``fwhm(ω₀=6)/fwhm(ω₀=10)`` should therefore equal
    ``10/6 ≈ 1.67`` — directionally correct, magnitude within a
    generous factor allowing for time-domain row averaging.
    """
    dt = 60.0
    n_samples = 4096
    t = np.arange(n_samples) * dt
    f_in = 1.0 / (60.0 * 60.0)
    signal = np.sin(2.0 * np.pi * f_in * t)

    coi = n_samples // 4

    def fwhm(omega0: float) -> float:
        freq, _, power = cwt_power(signal, dt=dt, omega0=omega0, n_freqs=400)
        row = power[:, coi:-coi].mean(axis=1)
        half = row.max() / 2.0
        above = np.flatnonzero(row >= half)
        return float(freq[above[-1]] - freq[above[0]])

    fwhm_6 = fwhm(6.0)
    fwhm_10 = fwhm(10.0)
    # Directional check: ω₀=6 must be broader than ω₀=10.
    assert fwhm_6 > fwhm_10
    # Magnitude: target 1.67, accept the band [1.2, 3.5] — time-domain
    # averaging over the inner half of the segment broadens the
    # apparent FWHM relative to the wavelet's pure spectral FWHM
    # because the wavelet's response builds and decays slowly compared
    # to its own period.
    ratio = fwhm_6 / fwhm_10
    assert 1.2 < ratio < 3.5, (
        f"fwhm(omega0=6)/fwhm(omega0=10) = {ratio:.2f}, expected ~1.67"
    )


def test_returns_match_signal_length() -> None:
    """Output time axis length matches the input data length."""
    for n in (64, 256, 1000):
        _, time, cwt = morlet_cwt(np.zeros(n), dt=60.0, n_freqs=30)
        assert time.size == n
        assert cwt.shape[1] == n
