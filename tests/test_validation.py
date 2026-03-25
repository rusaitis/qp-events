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
from numpy.testing import assert_allclose

from qp.wavesolver.solver import WavesolverConfig, solve_eigenfrequencies
from qp.fieldline.kmag_model import SaturnField


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
        """Fundamental frequency should decrease with increasing L-shell."""
        freqs = {}
        for L in [6, 10]:
            config = WavesolverConfig(
                l_shell=float(L),
                n_modes=1,
                freq_range=(1e-4, 0.005),
                resolution=100,
            )
            result = solve_eigenfrequencies(config)
            freqs[L] = result.angular_frequencies[0]

        assert freqs[6] > freqs[10], "Fundamental should decrease with L"

    def test_higher_modes_at_higher_frequency(self):
        """Mode 2 should be at higher frequency than mode 1."""
        config = WavesolverConfig(
            l_shell=8.0,
            n_modes=2,
            freq_range=(1e-4, 0.005),
            resolution=100,
        )
        result = solve_eigenfrequencies(config)
        assert result.n_modes >= 2
        assert result.angular_frequencies[1] > result.angular_frequencies[0]

    def test_toroidal_and_poloidal_both_converge(self):
        """Both toroidal and poloidal should find valid eigenfrequencies.

        For a dipole they are nearly degenerate (identical scale factor
        geometry in the analytical approximation), which is physically correct.
        """
        tor = solve_eigenfrequencies(
            WavesolverConfig(
                l_shell=8.0,
                n_modes=1,
                component="toroidal",
                freq_range=(1e-4, 0.005),
                resolution=100,
            )
        )
        pol = solve_eigenfrequencies(
            WavesolverConfig(
                l_shell=8.0,
                n_modes=1,
                component="poloidal",
                freq_range=(1e-4, 0.005),
                resolution=100,
            )
        )
        assert tor.n_modes >= 1
        assert pol.n_modes >= 1
        # For dipole, toroidal ≈ poloidal (nearly degenerate)
        assert_allclose(
            tor.angular_frequencies[0],
            pol.angular_frequencies[0],
            rtol=0.01,
        )


class TestKMAGValidation:
    """Cross-validate KMAG eigenfrequencies against published reference."""

    @pytest.fixture(scope="class")
    def field(self):
        return SaturnField()

    @pytest.mark.slow
    def test_kmag_fundamental_at_L8(self, field):
        """KMAG fundamental at L=8 should be ~0.12 mHz (within 50%)."""
        config = WavesolverConfig(
            l_shell=8.0,
            n_modes=1,
            field=field,
            local_time_hours=12.0,
            freq_range=(1e-4, 0.005),
            resolution=100,
        )
        result = solve_eigenfrequencies(config)
        f_mhz = result.frequencies_mhz[0]
        ref = _REF_FUNDAMENTAL_MHZ[8]
        assert ref * 0.5 < f_mhz < ref * 2.0, (
            f"KMAG L=8 fundamental {f_mhz:.4f} mHz outside "
            f"[{ref * 0.5:.4f}, {ref * 2.0:.4f}] mHz"
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
