"""Phase 5 validation: cross-check new wavesolver against reference eigenfrequencies.

The reference values are from Rusaitis et al. (2021) — KMAG field model
+ Bagenal density at noon. Our new solver should produce eigenfrequencies
in the same ballpark (within ~50% for the fundamental mode), with the
correct trends (frequency decreases with L-shell, higher modes at higher
frequency).

Exact numerical agreement is not expected because:
- Different ODE integration method (RK4+numba vs scipy LSODA)
- Different field line tracing step size
- Different scale factor computation (analytical dipole approx vs numerical)
- Different density interpolation details
"""

from __future__ import annotations

import numpy as np
import pytest

from qp.fieldline.kmag_model import SaturnField
from qp.wavesolver.density import UniformDensity
from qp.wavesolver.solver import WavesolverConfig, solve_eigenfrequencies

# Reference: toroidal eigenperiods (minutes) at L=20 noon from
# Rusaitis et al. (2021) Figure 2 — the only place in the paper where
# explicit numerical mode periods are reported. KMAG field + Bagenal &
# Delamere (2011) density model. These are the gold-standard reference
# values for cross-validation. (The earlier `_REF_FUNDAMENTAL_MHZ` table
# at L=8/10/15 was digitised from Rusaitis Fig 3 log-scale curves by a
# previous session and turned out to be incorrect — the actual m=1
# values are ~2× lower at those L-shells.)
_REF_FIG2_L20_PERIODS_MIN: dict[int, float] = {
    1: 469.0,
    2: 121.0,
    3: 72.0,
    4: 52.0,
}


class TestDipoleTrends:
    """Verify physical trends with the dipole field model (fast)."""

    def test_frequency_decreases_with_l_shell(self):
        """Fundamental frequency should decrease with increasing L-shell.

        Uses UniformDensity so v_A is physically sensible for a pure dipole
        (Bagenal-Delamere expects KMAG's larger equatorial B and produces
        v_A_eq ~ km/s when paired with the dipole at L>~5).
        """
        freqs = {}
        for L in [6, 10]:
            config = WavesolverConfig(
                l_shell=float(L),
                n_modes=1,
                density_model=UniformDensity(n0=1e7),
            )
            result = solve_eigenfrequencies(config)
            freqs[L] = result.angular_frequencies[0]

        assert freqs[6] > freqs[10], "Fundamental should decrease with L"

    def test_higher_modes_at_higher_frequency(self):
        """Mode 2 should be at higher frequency than mode 1."""
        config = WavesolverConfig(
            l_shell=8.0,
            n_modes=2,
            density_model=UniformDensity(n0=1e7),
        )
        result = solve_eigenfrequencies(config)
        assert result.n_modes >= 2
        assert result.angular_frequencies[1] > result.angular_frequencies[0]

    def test_toroidal_and_poloidal_both_converge(self):
        """Both toroidal and poloidal should find valid eigenfrequencies.

        For axisymmetric fields, toroidal uses h_φ = ρ = r·sin θ while
        poloidal uses h_ψ = 1/(B·ρ) (Singer 1981 thin-flux-tube formulation).
        The two are NOT degenerate in general — they only coincide for very
        specific profiles. This test only checks both solvers converge to a
        positive eigenfrequency.
        """
        tor = solve_eigenfrequencies(
            WavesolverConfig(
                l_shell=8.0,
                n_modes=1,
                component="toroidal",
                density_model=UniformDensity(n0=1e7),
            )
        )
        pol = solve_eigenfrequencies(
            WavesolverConfig(
                l_shell=8.0,
                n_modes=1,
                component="poloidal",
                density_model=UniformDensity(n0=1e7),
            )
        )
        assert tor.n_modes >= 1
        assert pol.n_modes >= 1
        assert tor.angular_frequencies[0] > 0
        assert pol.angular_frequencies[0] > 0


class TestKMAGValidation:
    """Cross-validate KMAG eigenfrequencies against published reference."""

    @pytest.fixture(scope="class")
    def field(self):
        return SaturnField()

    @pytest.mark.slow
    def test_kmag_matches_rusaitis_fig2_at_L20(self, field):
        """Reproduces the explicit Rusaitis (2021) Fig 2 mode ladder at L=20.

        Figure 2 of Rusaitis et al. (2021) plots the first four harmonics
        of a noon-sector field line with equatorial crossing distance
        L = 20 R_S, and gives explicit eigenperiods in the legend:
        m=1 → 469 min, m=2 → 121 min, m=3 → 72 min, m=4 → 52 min.
        These are the only place in the paper where numerical mode
        periods are written down, making them the gold-standard
        reference for cross-validation.

        Our matrix solver matches every mode to within 6 % at L=20 noon
        (see ``docs/notes/wavesolver_reference.md`` for the audit).
        This includes mode 1 — the soft-wall mode that earlier sessions
        had hypothesised Rusaitis was missing. They were not missing it.

        Notes
        -----
        The earlier reference table ``_REF_FUNDAMENTAL_MHZ`` at L=8/10/15
        (= 0.12, 0.088, 0.053 mHz) was digitised from Rusaitis Fig 3
        log-scale curves and turned out to be a factor of ~2 too high.
        The current test is anchored on Fig 2 instead, which gives
        explicit numerical values.
        """
        cfg = WavesolverConfig(
            l_shell=20.0,
            n_modes=4,
            field=field,
            local_time_hours=12.0,
        )
        result = solve_eigenfrequencies(cfg)
        for mode_number, ref_period in _REF_FIG2_L20_PERIODS_MIN.items():
            our_period = float(result.periods_minutes[mode_number - 1])
            rel = abs(our_period - ref_period) / ref_period
            assert rel < 0.10, (
                f"KMAG L=20 noon m={mode_number}: ours {our_period:.1f} min "
                f"vs Rusaitis Fig 2 {ref_period:.1f} min — {rel:.1%} off"
            )

    @pytest.mark.slow
    def test_kmag_high_mode_spacing_converges_to_wkb(self, field):
        """High-mode spacing converges to the WKB-asymptotic value at L=8.

        The Sturm–Liouville ladder for the toroidal wave equation with
        Dirichlet BCs satisfies ω_m → (m − δ) π / ∫(ds/v_A) for large m,
        with a constant phase shift δ ≈ 0.55 at KMAG L=8 noon (soft-wall
        boundary). The spacing ω_m − ω_{m−1} therefore converges to
        π / ∫(ds/v_A) = ω_WKB independent of m and δ. This test pins
        that convergence at the upper end of the ladder.
        """
        cfg = WavesolverConfig(
            l_shell=8.0,
            n_modes=6,
            field=field,
            local_time_hours=12.0,
        )
        result = solve_eigenfrequencies(cfg)
        s = np.asarray(result.arc_length)
        va = np.asarray(result.alfven_velocity)
        omega_wkb = np.pi / np.trapezoid(1.0 / va, s)
        spacings = np.diff(result.angular_frequencies)
        assert abs(spacings[-1] - omega_wkb) / omega_wkb < 0.05, (
            f"High-mode spacing {spacings[-1]:.3e} differs from "
            f"WKB {omega_wkb:.3e} by more than 5 %"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("l_shell", [8.0, 10.0, 15.0])
    def test_kmag_soft_wall_phase_shift_locked(self, field, l_shell):
        """The soft-wall phase shift ω_1 / ω_WKB ≈ 0.45 is locked across L.

        For the Singer (1981) arc-length wave equation with Dirichlet BCs
        at the trace endpoints, the standing-wave ladder satisfies
        :math:`\\omega_m \\approx (m - \\delta) \\pi / \\int(ds/v_A)` with
        a constant phase shift :math:`\\delta \\approx 0.55` set by the
        soft-wall behaviour at the high-:math:`v_A` ionospheric region.
        Mode 1 therefore lands at ~0.45 × the WKB-asymptotic value.

        This regression test pins that ratio so that future changes to
        the v_A profile, BC enforcement, or discretisation cannot silently
        remove or alter the soft-wall mode. See
        ``docs/notes/wavesolver_reference.md`` for the full physical
        discussion.
        """
        cfg = WavesolverConfig(
            l_shell=l_shell,
            n_modes=2,
            field=field,
            local_time_hours=12.0,
        )
        result = solve_eigenfrequencies(cfg)
        s = np.asarray(result.arc_length)
        va = np.asarray(result.alfven_velocity)
        omega_wkb = np.pi / np.trapezoid(1.0 / va, s)
        ratio = result.angular_frequencies[0] / omega_wkb
        assert 0.42 < ratio < 0.48, (
            f"KMAG L={l_shell:.0f} soft-wall ratio ω_1/ω_WKB = {ratio:.3f} "
            f"is outside the expected [0.42, 0.48] band. Either the soft-wall "
            f"mode has shifted or the WKB integral has changed."
        )

    @pytest.mark.slow
    def test_kmag_trend_with_l(self, field):
        """KMAG eigenfrequencies should decrease with L-shell."""
        freqs = {}
        for L in [8, 15]:
            config = WavesolverConfig(
                l_shell=float(L),
                n_modes=1,
                field=field,
                local_time_hours=12.0,
                freq_range=(1e-4, 0.005),
                resolution=100,
            )
            result = solve_eigenfrequencies(config)
            freqs[L] = result.frequencies_mhz[0]

        assert freqs[8] > freqs[15], (
            f"KMAG frequency should decrease: L=8 ({freqs[8]:.4f}) > L=15 ({freqs[15]:.4f})"
        )

    @pytest.mark.slow
    def test_kmag_multiple_modes(self, field):
        """KMAG at L=8 should find at least 3 modes with correct ordering."""
        config = WavesolverConfig(
            l_shell=8.0,
            n_modes=3,
            field=field,
            local_time_hours=12.0,
            freq_range=(1e-4, 0.008),
            resolution=150,
        )
        result = solve_eigenfrequencies(config)
        assert result.n_modes >= 3
        freqs = result.angular_frequencies
        assert np.all(np.diff(freqs) > 0), (
            "Modes should be in ascending frequency order"
        )
