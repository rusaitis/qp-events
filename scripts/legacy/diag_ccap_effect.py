"""Phase-1 diagnostic: does the v_A relativistic cap cause the sub-WKB mode 1?

The default `alfven_velocity` in `qp.wavesolver.density` applies a relativistic
post-cap `va ← va / √(1 + (va/c)²)`. This bounds v_A by c in the low-density
ionospheric region, where the Bagenal profile decays exponentially and v_A_raw
diverges. Two competing hypotheses for the observed sub-WKB mode 1 at
KMAG L=8 (0.057 mHz vs WKB π/∫(ds/v_A) = 0.126 mHz):

  (a) The c-cap softens the boundary and admits one extra eigenmode below WKB.
  (b) The soft-wall mode is inherent to *any* v_A that grows large near the
      ionosphere — even without the cap, (ω/v_A)² → 0 near the boundary and
      the wave equation degenerates to y'' + dlnh·y' = 0 (linear), still
      admitting the sub-WKB mode.

This script runs KMAG L=8 noon three ways and prints the mode ladders:

  - Default: current code (cap active, no density floor)
  - No-cap: monkey-patched `alfven_velocity` returns raw B/√(μ₀ n m_i)
  - Density-floor: monkey-patched to apply n_eff = max(n, B²/(μ₀ m_i c²))
    before computing v_A, with no post-cap.

If the no-cap run also gives mode 1 ≈ 0.057 mHz, hypothesis (b) wins and the
plan needs to pivot (a c-cap toggle alone will not reconcile with Rusaitis 2021).
If no-cap and density-floor both give a clean 1:2:3:... ladder anchored at
~0.126 mHz, hypothesis (a) wins and we proceed with Phase 2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from _common import setup_logging
from scipy.constants import mu_0 as MU0
from scipy.constants import speed_of_light as SPEED_OF_LIGHT

from qp.fieldline.kmag_model import SaturnField
from qp.wavesolver import density as density_mod
from qp.wavesolver import field_profile as field_profile_mod
from qp.wavesolver.density import SATURN_RADIUS
from qp.wavesolver.solver import WavesolverConfig, solve_eigenfrequencies

setup_logging()
log = logging.getLogger(__name__)


def _va_raw(B, n, ion_mass_kg):
    """No-cap, no-floor Alfvén speed (allows v_A > c)."""
    n_safe = np.maximum(n, 1e-10)
    va = B / np.sqrt(MU0 * n_safe * ion_mass_kg)
    return np.nan_to_num(va, nan=SPEED_OF_LIGHT, posinf=SPEED_OF_LIGHT * 100)


def _va_density_floor(B, n, ion_mass_kg):
    """B-dependent density floor that bounds v_A ≤ c without a post-cap."""
    n_floor = B**2 / (MU0 * ion_mass_kg * SPEED_OF_LIGHT**2)
    n_eff = np.maximum(n, n_floor)
    n_eff = np.maximum(n_eff, 1e-10)
    return B / np.sqrt(MU0 * n_eff * ion_mass_kg)


def _solve(label: str, l_shell: float = 8.0, n_modes: int = 6):
    cfg = WavesolverConfig(
        l_shell=l_shell,
        n_modes=n_modes,
        field=SaturnField(),
        local_time_hours=12.0,
        freq_range=(1e-6, 1e-2),
        resolution=400,
        include_eigenfunctions=True,
    )
    result = solve_eigenfrequencies(cfg)
    s = np.asarray(result.arc_length)
    va = np.asarray(result.alfven_velocity)
    phase_integral = np.trapezoid(1.0 / va, s)
    omega_wkb = np.pi / phase_integral

    log.info("=" * 72)
    log.info("  %s", label)
    log.info("  v_A: s_min=%.3e  s_eq=%.3e  s_max=%.3e  (m/s)", va[0], va.min(), va[-1])
    log.info(
        "  WKB omega_1 = pi / int(ds/v_A) = %.4e rad/s = %.4f mHz",
        omega_wkb,
        omega_wkb / (2 * np.pi) * 1e3,
    )
    log.info("  mode |   omega       |  freq (mHz) | period (min) | ratio /m1")
    omegas = result.angular_frequencies
    fs = result.frequencies_mhz
    Ts = result.periods_minutes
    for i, m in enumerate(result.modes):
        log.info(
            "    %d  | %.6e | %10.4f  | %10.2f  | %6.3f",
            m.mode_number,
            omegas[i],
            fs[i],
            Ts[i],
            omegas[i] / omegas[0],
        )
    return result, omega_wkb, va, s


def main() -> None:
    out = Path("Output/diagnostics")
    out.mkdir(parents=True, exist_ok=True)

    log.info("KMAG L=8 noon — testing three v_A boundary-region models")

    # (1) Default: current code (c-cap on)
    r_def, wkb_def, va_def, s_def = _solve("(1) default (relativistic cap ON)")

    # (2) Monkey-patch BOTH density and field_profile namespaces (field_profile
    # imports alfven_velocity into its own namespace, so density-only patching
    # does not reach the call site).
    with (
        patch.object(density_mod, "alfven_velocity", _va_raw),
        patch.object(field_profile_mod, "alfven_velocity", _va_raw),
    ):
        r_nocap, wkb_nocap, va_nocap, s_nocap = _solve(
            "(2) no cap (raw v_A, can exceed c)"
        )

    # (3) B-dependent density floor, no post-cap
    with (
        patch.object(density_mod, "alfven_velocity", _va_density_floor),
        patch.object(field_profile_mod, "alfven_velocity", _va_density_floor),
    ):
        r_floor, wkb_floor, va_floor, s_floor = _solve(
            "(3) density floor n>=B^2/(mu0 m_i c^2), no post-cap"
        )

    # Plot v_A(s) overlays
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(s_def / SATURN_RADIUS, va_def / 1e3, label="(1) c-cap", lw=1.8)
    ax.semilogy(
        s_nocap / SATURN_RADIUS, va_nocap / 1e3, "--", label="(2) no cap (raw)", lw=1.2
    )
    ax.semilogy(
        s_floor / SATURN_RADIUS, va_floor / 1e3, ":", label="(3) density floor", lw=1.8
    )
    ax.axhline(SPEED_OF_LIGHT / 1e3, color="0.6", lw=0.5, label="c (km/s)")
    ax.set_xlabel("arc length s (R_S)")
    ax.set_ylabel("v_A (km/s)")
    ax.set_title("KMAG L=8 noon — v_A under three boundary-region models")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "diag_ccap_effect_va.png", dpi=120)
    plt.close(fig)
    log.info("wrote %s/diag_ccap_effect_va.png", out)

    # Plot mode-1 eigenfunctions
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, r in [
        ("(1) c-cap", r_def),
        ("(2) no cap", r_nocap),
        ("(3) density floor", r_floor),
    ]:
        m = r.modes[0]
        y = m.eigenfunction
        if y is not None:
            s = m.arc_length
            ax.plot(np.asarray(s) / SATURN_RADIUS, y / np.max(np.abs(y)), label=label)
    ax.set_xlabel("arc length s (R_S)")
    ax.set_ylabel("y₁(s) / |y₁|_max")
    ax.set_title("KMAG L=8 noon — mode-1 eigenfunction under three v_A models")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "diag_ccap_effect_eigenfunction_m1.png", dpi=120)
    plt.close(fig)
    log.info("wrote %s/diag_ccap_effect_eigenfunction_m1.png", out)

    # Summary verdict
    f1_def = r_def.frequencies_mhz[0]
    f1_nocap = r_nocap.frequencies_mhz[0]
    f1_floor = r_floor.frequencies_mhz[0]
    log.info("=" * 72)
    log.info("VERDICT")
    log.info("  (1) default        mode 1 = %.4f mHz", f1_def)
    log.info("  (2) no cap         mode 1 = %.4f mHz", f1_nocap)
    log.info("  (3) density floor  mode 1 = %.4f mHz", f1_floor)
    log.info("  reference (Rusaitis 2021, L=8) = 0.12 mHz")
    log.info("")
    if abs(f1_nocap - f1_def) / f1_def < 0.05:
        log.info("  → no-cap leaves mode 1 essentially unchanged.")
        log.info("    Soft-wall mode is INTRINSIC to the v_A profile.")
        log.info("    Pivot the plan — c-cap is not the culprit.")
    elif 0.10 < f1_nocap < 0.18:
        log.info("  → no-cap recovers mode 1 ≈ 0.12 mHz (matches reference).")
        log.info("    c-cap IS the source of the factor-of-2.")
        log.info("    Proceed with Phase 2 — add toggleable flags.")
    else:
        log.info("  → no-cap gives mode 1 = %.4f mHz (neither current nor reference).")
        log.info("    Investigate further before committing to Phase 2.", f1_nocap)


if __name__ == "__main__":
    main()
