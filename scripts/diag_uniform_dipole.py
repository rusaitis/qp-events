"""Phase-1 diagnostic: dipole + uniform-density Sturm-Liouville baseline.

Cross-checks the wavesolver against Singer (1981, Fig 4) and Cummings et al.
(1969, Table I), which both give mode-frequency ratios near 1 : 1.91 : 2.86
for a dipole field line with cold uniform plasma. If our solver reproduces
those ratios at L=6.6 to within ~20%, the solver mechanics are correct and
the previously-reported 1 : 7.5 anomaly was a snippet-side setup error
(most likely a freq_range that clipped the true fundamental).

Run:
    uv run python scripts/diag_uniform_dipole.py

Outputs:
    stdout — mode frequencies, ratios, periods, v_A endpoints
    Output/diagnostics/diag_uniform_dipole_va.png
    Output/diagnostics/diag_uniform_dipole_dlnh.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import atomic_mass as AMU
from scipy.constants import mu_0 as MU0

from qp.wavesolver.density import SATURN_RADIUS, UniformDensity
from qp.wavesolver.solver import WavesolverConfig, solve_eigenfrequencies

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ION_MASS_AMU = 18.0
SATURN_DIPOLE_MOMENT_NT = 380.0  # nT * R_S^3 — must match tracer.SATURN_DIPOLE_MOMENT

SINGER_1981_RATIOS = np.array([1.00, 1.91, 2.86, 3.74])  # L=6.6 Earth, uniform plasma


def n0_for_va_equator(l_shell: float, va_eq_target_ms: float) -> float:
    """Number density that produces a target equatorial Alfvén speed (m^-3).

    For a pure dipole at L, B_eq = M / L^3 (in nT) and v_A = B / sqrt(mu0 * rho).
    """
    B_eq_T = SATURN_DIPOLE_MOMENT_NT * 1e-9 / l_shell**3
    rho = B_eq_T**2 / (MU0 * va_eq_target_ms**2)
    return rho / (ION_MASS_AMU * AMU)


def run_case(l_shell: float, va_eq_target_kms: float, n_modes: int = 6):
    n0 = n0_for_va_equator(l_shell, va_eq_target_kms * 1e3)
    density = UniformDensity(n0=n0, ion_mass_amu=ION_MASS_AMU)
    cfg = WavesolverConfig(
        l_shell=l_shell,
        n_modes=n_modes,
        component="toroidal",
        field=None,  # dipole
        density_model=density,
        freq_range=(1e-6, 1e-1),  # wide — let the solver find the real fundamental
        resolution=400,
        include_eigenfunctions=True,
    )
    result = solve_eigenfrequencies(cfg)

    omegas = result.angular_frequencies
    periods_min = result.periods_minutes
    ratios = omegas / omegas[0] if len(omegas) else np.array([])

    assert result.alfven_velocity is not None and result.arc_length is not None
    va = np.asarray(result.alfven_velocity)
    s_rs = np.asarray(result.arc_length) / SATURN_RADIUS
    eq_idx = int(np.argmin(va))  # equator = v_A minimum (uniform density)

    log.info("=" * 72)
    log.info(
        "dipole L=%.2f  uniform n0=%.3e m^-3  va_eq target=%.0f km/s",
        l_shell,
        n0,
        va_eq_target_kms,
    )
    log.info(
        "v_A: s_min=%.2f km/s   s_eq=%.2f km/s   s_max=%.2f km/s",
        va[0] / 1e3,
        va[eq_idx] / 1e3,
        va[-1] / 1e3,
    )
    log.info(
        "arc length: s_min=%.3f R_S   s_max=%.3f R_S   total=%.3f R_S",
        s_rs[0],
        s_rs[-1],
        s_rs[-1] - s_rs[0],
    )
    log.info(
        "modes found: %d / %d requested",
        len(omegas),
        n_modes,
    )
    log.info("ω (rad/s)         period (min)   m  ratio ω/ω₁")
    for i, m in enumerate(result.modes):
        log.info(
            "  %14.6e  %12.2f   %2d  %6.3f",
            m.angular_frequency,
            periods_min[i],
            m.mode_number,
            ratios[i],
        )
    log.info(
        "Singer-1981 expected ratios (L=6.6 Earth, uniform plasma): %s",
        np.array2string(SINGER_1981_RATIOS, precision=2),
    )

    return result


def save_plots(result, name: str) -> None:
    out = Path("Output/diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    s_rs = np.asarray(result.arc_length) / SATURN_RADIUS
    va = np.asarray(result.alfven_velocity)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_rs, va / 1e3)
    ax.set_yscale("log")
    ax.set_xlabel("arc length s (R_S)")
    ax.set_ylabel("v_A (km/s)")
    ax.set_title(f"v_A(s) — {name}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{name}_va.png", dpi=120)
    plt.close(fig)

    # dlnh1B(s) from the spline embedded in the profile — recompute via solver
    # We have it via result, but EigenResult does not surface it; skip plot and
    # rely on the va plot for the moment. Add later if Phase 1 step 4 fires.
    log.info("wrote %s_va.png", name)


def main() -> None:
    for L, va_kms in [(6.6, 100.0), (8.0, 100.0)]:
        result = run_case(L, va_kms)
        save_plots(result, f"diag_uniform_dipole_L{L:.1f}")


if __name__ == "__main__":
    main()
