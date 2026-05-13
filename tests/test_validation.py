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

# Reference: fundamental toroidal eigenfrequencies (mHz) from KMAG + Bagenal
# at noon local time, from Rusaitis et al. (2021)
_REF_FUNDAMENTAL_MHZ = {
    8: 0.12,
    10: 0.088,
    15: 0.053,
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
    def test_kmag_wkb_fundamental_at_L8(self, field):
        """WKB-asymptotic fundamental at L=8 matches Rusaitis (2021) 0.12 mHz.

        Both the shooting and matrix backends find an additional
        low-frequency eigenmode at ~0.057 mHz at L=8 because the high-v_A
        regions near the ionospheres behave as soft walls (phase offset
        δ ≈ 0.45 in ω_m = (m − δ) · π / ∫(ds/v_A)). The Phase-1 diagnostic
        (``scripts/diag_ccap_effect.py``) confirmed this is intrinsic to
        the v_A profile, NOT an artefact of the relativistic v_A cap —
        running with raw uncapped v_A, the relativistic density floor,
        or both, yields the same sub-WKB mode 1. The published reference
        aligns with the WKB-asymptotic mode 1, ω_WKB = π / ∫(ds/v_A) —
        equivalent to our high-mode spacing. See
        docs/notes/wavesolver_reference.md for the full discussion.
        """
        cfg = WavesolverConfig(
            l_shell=8.0,
            n_modes=6,
            field=field,
            local_time_hours=12.0,
            freq_range=(1e-6, 1e-2),
            resolution=400,
        )
        result = solve_eigenfrequencies(cfg)
        s = np.asarray(result.arc_length)
        va = np.asarray(result.alfven_velocity)
        phase_integral = np.trapezoid(1.0 / va, s)
        omega_wkb = np.pi / phase_integral
        f_wkb_mhz = omega_wkb / (2.0 * np.pi) * 1e3
        ref = _REF_FUNDAMENTAL_MHZ[8]
        assert ref * 0.7 < f_wkb_mhz < ref * 1.3, (
            f"KMAG L=8 WKB fundamental {f_wkb_mhz:.4f} mHz outside "
            f"[{ref * 0.7:.4f}, {ref * 1.3:.4f}] mHz around reference {ref} mHz"
        )
        # Sanity: the solver's high-mode spacing should converge to ω_WKB.
        spacings = np.diff([m.angular_frequency for m in result.modes])
        assert abs(spacings[-1] - omega_wkb) / omega_wkb < 0.05, (
            f"High-mode spacing {spacings[-1]:.3e} differs from "
            f"WKB {omega_wkb:.3e} by more than 5 %"
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
