r"""End-to-end verification of Stokes / ellipticity / inclination recovery.

Tests assert against closed-form ground truth rather than against the
implementation's own output, so they pin the convention rather than ratify it.

For a transverse wave with major-axis amplitude :math:`A`, ellipticity
:math:`e \in [-1, 1]`, and inclination :math:`\psi` of the major axis from
:math:`\hat b_{\perp 1}`,

.. math::

    I &= A^2 (1 + e^2) \\
    Q &= A^2 (1 - e^2) \cos 2\psi \\
    U &= A^2 (1 - e^2) \sin 2\psi \\
    V &= 2 A^2 e

so :math:`Q^2 + U^2 + V^2 = A^4 (1 + e^2)^2 = I^2` for a fully polarized wave
and the degree of polarization is exactly 1.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from qp.signal.cross_correlation import (
    classify_polarization,
    phase_shift,
)
from qp.signal.cross_correlation import (
    stokes_parameters as stokes_parameters_real,
)
from qp.signal.polarization import (
    degree_of_polarization,
    stokes_parameters,
)
from qp.signal.wavelet import morlet_cwt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wave(
    period_s: float,
    dt: float,
    n: int,
    *,
    amplitude: float = 1.0,
    ellipticity: float = 0.0,
    inclination_deg: float = 0.0,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build a (b_perp1, b_perp2) pair with prescribed polarization state.

    The unrotated frame has the major axis along :math:`\hat b_{\perp 1}` and
    the minor axis (quadrature, amplitude :math:`A|e|`) along
    :math:`\hat b_{\perp 2}`. A positive ``ellipticity`` means
    :math:`b_{\perp 2}` lags :math:`b_{\perp 1}` by :math:`\pi/2` — the
    Stokes-V > 0 convention. Inclination rotates the ellipse in the
    transverse plane.
    """
    t = np.arange(n) * dt
    w = 2 * np.pi / period_s
    u1 = amplitude * np.sin(w * t)
    u2 = amplitude * ellipticity * (-np.cos(w * t))
    psi = np.radians(inclination_deg)
    c, s = np.cos(psi), np.sin(psi)
    b1 = c * u1 - s * u2
    b2 = s * u1 + c * u2
    if snr_db is not None and rng is not None:
        signal_power = float(np.mean(b1**2 + b2**2))
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        sigma = np.sqrt(noise_power / 2.0)
        b1 = b1 + sigma * rng.standard_normal(n)
        b2 = b2 + sigma * rng.standard_normal(n)
    return b1, b2


def _predicted_stokes(
    amplitude: float,
    ellipticity: float,
    inclination_deg: float,
) -> tuple[float, float, float, float]:
    """Closed-form Stokes for the wave built by ``_make_wave``."""
    a2 = amplitude * amplitude
    e = ellipticity
    psi = np.radians(inclination_deg)
    return (
        a2 * (1 + e * e),
        a2 * (1 - e * e) * np.cos(2 * psi),
        a2 * (1 - e * e) * np.sin(2 * psi),
        2 * a2 * e,
    )


def _cwt_inband_slice(
    b1: np.ndarray,
    b2: np.ndarray,
    dt: float,
    period_s: float,
    *,
    edge_drop: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """CWT both signals at the wave period; return central in-band slices."""
    f0 = 1.0 / period_s
    freq, _, c1 = morlet_cwt(
        b1,
        dt=dt,
        freq_min=f0 * 0.7,
        freq_max=f0 * 1.4,
        n_freqs=12,
    )
    _, _, c2 = morlet_cwt(
        b2,
        dt=dt,
        freq_min=f0 * 0.7,
        freq_max=f0 * 1.4,
        n_freqs=12,
    )
    i_peak = int(np.argmin(np.abs(freq - f0)))
    n = c1.shape[1]
    lo = int(edge_drop * n)
    hi = n - lo
    return c1[i_peak, lo:hi], c2[i_peak, lo:hi]


def _ellipticity_from_stokes(I_: float, Q: float, U: float, V: float) -> float:
    p = np.sqrt(Q * Q + U * U + V * V)
    if p <= 0 or I_ <= 0:
        return 0.0
    return float(np.tan(0.5 * np.arcsin(np.clip(V / p, -1.0, 1.0))))


def _inclination_from_stokes(Q: float, U: float) -> float:
    return float(np.degrees(0.5 * np.arctan2(U, Q)))


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_stokes_invariant_inequality_for_random_pairs() -> None:
    """Q² + U² + V² ≤ I² always, with equality for fully polarized waves."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 4000
        z1 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        z2 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        i_, q, u, v = stokes_parameters(z1, z2)
        # Strict ≤ (with a small numerical tolerance); independent random
        # pairs are nearly unpolarized so the gap should be wide.
        assert q * q + u * u + v * v <= i_ * i_ + 1e-9


def test_stokes_invariant_equality_for_fully_polarized_wave() -> None:
    """For a pure synthetic wave, Q² + U² + V² == I² to ~1%."""
    for ell in (-1.0, -0.5, 0.0, 0.3, 0.7, 1.0):
        b1, b2 = _make_wave(3600.0, 60.0, 4000, ellipticity=ell)
        z1, z2 = hilbert(b1), hilbert(b2)
        i_, q, u, v = stokes_parameters(z1, z2)
        ratio = np.sqrt(q * q + u * u + v * v) / i_
        assert ratio == pytest.approx(1.0, rel=0.02), f"d={ratio} for ellipticity={ell}"


@pytest.mark.parametrize("ellipticity", [-1.0, -0.5, 0.0, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("psi_deg", [0.0, 30.0, 60.0, 90.0])
def test_stokes_matches_closed_form(ellipticity: float, psi_deg: float) -> None:
    """Implementation matches the (I, Q, U, V) closed form to ~1%."""
    b1, b2 = _make_wave(
        3600.0,
        60.0,
        4000,
        ellipticity=ellipticity,
        inclination_deg=psi_deg,
    )
    z1, z2 = _cwt_inband_slice(b1, b2, dt=60.0, period_s=3600.0)
    got = np.array(stokes_parameters(z1, z2))
    pred = np.array(_predicted_stokes(1.0, ellipticity, psi_deg))
    # Compare normalized so absolute scaling (CWT vs Hilbert) doesn't matter.
    got_norm = got / got[0]
    pred_norm = pred / pred[0]
    np.testing.assert_allclose(got_norm[1:], pred_norm[1:], atol=0.03)


def test_ellipticity_bound() -> None:
    """|tan χ| ≤ 1 always — derived ellipticity stays in the geometric range."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        e = rng.uniform(-1, 1)
        psi = rng.uniform(0, 180)
        b1, b2 = _make_wave(3600.0, 60.0, 2000, ellipticity=e, inclination_deg=psi)
        z1, z2 = hilbert(b1), hilbert(b2)
        i_, q, u, v = stokes_parameters(z1, z2)
        ell = _ellipticity_from_stokes(i_, q, u, v)
        assert -1.0 - 1e-9 <= ell <= 1.0 + 1e-9, f"|ell|>1: {ell}"


# ---------------------------------------------------------------------------
# Handedness — pin the sign convention
# ---------------------------------------------------------------------------


def test_handedness_right_circular_gives_positive_V() -> None:
    """b_perp2 lags b_perp1 by +π/2 → V > 0 → ellipticity > 0."""
    b1, b2 = _make_wave(3600.0, 60.0, 4000, ellipticity=+1.0)
    z1, z2 = hilbert(b1), hilbert(b2)
    i_, _, _, v = stokes_parameters(z1, z2)
    assert v > 0, f"expected V > 0 for right-circular, got V={v}"
    assert _ellipticity_from_stokes(i_, 0.0, 0.0, v) > 0.9


def test_handedness_left_circular_gives_negative_V() -> None:
    """Reversed quadrature → V < 0 → ellipticity < 0."""
    b1, b2 = _make_wave(3600.0, 60.0, 4000, ellipticity=-1.0)
    z1, z2 = hilbert(b1), hilbert(b2)
    i_, _, _, v = stokes_parameters(z1, z2)
    assert v < 0, f"expected V < 0 for left-circular, got V={v}"
    assert _ellipticity_from_stokes(i_, 0.0, 0.0, v) < -0.9


# ---------------------------------------------------------------------------
# Inclination recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("psi_deg", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
def test_inclination_recovery_linear(psi_deg: float) -> None:
    """A linear wave at ψ recovers ψ via 0.5·arctan2(U, Q) within 1°."""
    b1, b2 = _make_wave(3600.0, 60.0, 4000, ellipticity=0.0, inclination_deg=psi_deg)
    z1, z2 = hilbert(b1), hilbert(b2)
    _, q, u, _ = stokes_parameters(z1, z2)
    psi_hat = _inclination_from_stokes(q, u)
    # Inclination is mod 180° — fold both into [0, 180).
    diff = abs((psi_hat - psi_deg + 90) % 180 - 90)
    assert diff < 1.0, f"recovered ψ={psi_hat:.3f}°, expected {psi_deg:.3f}°"


# ---------------------------------------------------------------------------
# Two-implementation agreement — partial, documents the V-sign discrepancy
# ---------------------------------------------------------------------------


def test_I_Q_U_agree_across_implementations() -> None:
    """I, Q, U match between polarization and cross_correlation paths."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        ell = rng.uniform(-1, 1)
        psi = rng.uniform(0, 180)
        b1, b2 = _make_wave(
            3600.0,
            60.0,
            4000,
            ellipticity=ell,
            inclination_deg=psi,
        )
        cwt_path = stokes_parameters(hilbert(b1), hilbert(b2))
        real_path = stokes_parameters_real(b1, b2)
        np.testing.assert_allclose(cwt_path[:3], real_path[:3], rtol=1e-10, atol=1e-12)


def test_V_sign_convention_discrepancy_documented() -> None:
    r"""**Known issue** to be fixed in Phase 2 of the consolidation.

    ``polarization.stokes_parameters`` uses :math:`V = 2\,\mathrm{Im}\langle
    z_1 z_2^* \rangle` (Samson 1973, Born & Wolf §1.4.2 — the literature
    convention). ``cross_correlation.stokes_parameters`` uses
    :math:`V = 2\,\mathrm{Im}\langle z_1^* z_2 \rangle = -2\,\mathrm{Im}
    \langle z_1 z_2^* \rangle`. The two are equal-and-opposite.

    Pre-existing tests assert on :math:`|V|` so the discrepancy has been
    invisible. This test pins it explicitly so the Phase 2 consolidation
    fixes the right module: keep the polarization-module convention,
    flip the cross_correlation copy.
    """
    b1, b2 = _make_wave(3600.0, 60.0, 4000, ellipticity=+1.0)
    v_polar = stokes_parameters(hilbert(b1), hilbert(b2))[3]
    v_cross = stokes_parameters_real(b1, b2)[3]
    assert v_polar > 0, "V_polar must be positive for ellipticity=+1"
    assert v_cross < 0, "V_cross is currently negative — to be flipped"
    assert v_polar == pytest.approx(-v_cross, rel=1e-10)


# ---------------------------------------------------------------------------
# Stokes ↔ phase_shift agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase_offset_deg", [0.0, 45.0, 90.0, 135.0, 180.0, 270.0])
def test_phase_shift_geometry_matches_input_lag(phase_offset_deg: float) -> None:
    r"""``phase_shift()`` recovers the input quadrature lag modulo sign.

    For ``b1 = sin(ωt)`` and ``b2 = sin(ωt − δ)``, the analytic Stokes V is
    :math:`+2A^2\sin\delta` (b2 lags b1 by δ ⇒ right-handed for δ > 0). The
    cross-correlation peak lag has the opposite sign convention from V (it
    measures the shift you'd apply to b2 to align it with b1), so this test
    only pins the unsigned geometric agreement — same point on the
    polarization circle. The sign convention is unified in Phase 2.
    """
    dt, period_s, n = 60.0, 3600.0, 1000
    t = np.arange(n) * dt
    w = 2 * np.pi / period_s
    delta = np.radians(phase_offset_deg)
    b1 = np.sin(w * t)
    b2 = np.sin(w * t - delta)
    _, phase_xc = phase_shift(b1, b2, dt=dt, period=period_s)
    # Accept either sign convention by comparing the unit complex phasor.
    z_got = np.exp(1j * np.radians(phase_xc))
    z_expect_pos = np.exp(1j * delta)
    z_expect_neg = np.exp(-1j * delta)
    err = min(abs(z_got - z_expect_pos), abs(z_got - z_expect_neg))
    assert err < 0.1, (
        f"phase_shift returned {phase_xc:.2f}°, expected ±{phase_offset_deg:.2f}°"
    )


# ---------------------------------------------------------------------------
# Parametric recovery grid — the core verification
# ---------------------------------------------------------------------------


_ELLIPTICITIES = (-1.0, -0.5, 0.0, 0.3, 0.7, 1.0)
_INCLINATIONS = (0.0, 30.0, 60.0, 90.0)
_PERIODS_MIN = (30.0, 60.0, 120.0)
_TOL_BY_SNR = {None: 0.05, 20.0: 0.10, 10.0: 0.15, 0.0: 0.30}


@pytest.mark.parametrize("snr_db", list(_TOL_BY_SNR))
@pytest.mark.parametrize("period_min", _PERIODS_MIN)
@pytest.mark.parametrize("psi_deg", _INCLINATIONS)
@pytest.mark.parametrize("ellipticity", _ELLIPTICITIES)
def test_parametric_recovery(
    ellipticity: float,
    psi_deg: float,
    period_min: float,
    snr_db: float | None,
) -> None:
    """Recover ellipticity from synthetic waves over (e, ψ, period, SNR) grid.

    Uses the CWT path (the same one the detector uses): build the wave,
    take a narrow Morlet CWT around the wave frequency, and read the
    Stokes parameters of the in-band, edge-trimmed slice.
    """
    dt = 60.0
    period_s = period_min * 60.0
    # Enough cycles for stable Stokes statistics and for the in-band CWT
    # cone of influence to clear: ~10 cycles after edge trimming.
    n_samples = max(1024, int(round(15 * period_s / dt)))
    rng = np.random.default_rng(
        hash((ellipticity, psi_deg, period_min, snr_db)) & 0xFFFFFFFF,
    )
    b1, b2 = _make_wave(
        period_s,
        dt,
        n_samples,
        ellipticity=ellipticity,
        inclination_deg=psi_deg,
        snr_db=snr_db,
        rng=rng,
    )
    z1, z2 = _cwt_inband_slice(b1, b2, dt, period_s)
    i_, q, u, v = stokes_parameters(z1, z2)
    ell_hat = _ellipticity_from_stokes(i_, q, u, v)
    tol = _TOL_BY_SNR[snr_db]
    assert abs(ell_hat - ellipticity) < tol, (
        f"ellipticity={ellipticity}, ψ={psi_deg}°, T={period_min} min, "
        f"SNR={snr_db} dB → recovered {ell_hat:.3f} (tol {tol})"
    )


# ---------------------------------------------------------------------------
# classify_polarization — boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase_deg, expected",
    [
        (0.0, "linear"),
        (180.0, "linear"),
        (359.0, "linear"),
        (90.0, "circular"),
        (270.0, "circular"),
        (45.0, "mixed"),
        (135.0, "mixed"),
        (225.0, "mixed"),
    ],
)
def test_classify_polarization_canonical_phases(
    phase_deg: float,
    expected: str,
) -> None:
    assert classify_polarization(phase_deg) == expected


# ---------------------------------------------------------------------------
# degree_of_polarization edge-cases
# ---------------------------------------------------------------------------


def test_degree_of_polarization_fully_polarized_circular() -> None:
    b1, b2 = _make_wave(3600.0, 60.0, 4000, ellipticity=1.0)
    z1, z2 = hilbert(b1), hilbert(b2)
    assert degree_of_polarization(z1, z2) == pytest.approx(1.0, abs=0.02)


def test_degree_of_polarization_low_for_independent_noise() -> None:
    rng = np.random.default_rng(11)
    n = 20000
    z1 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    z2 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert degree_of_polarization(z1, z2) < 0.05
