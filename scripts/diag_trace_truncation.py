"""Phase-3b follow-up: does truncating the trace endpoints kill the soft-wall mode?

Phases 1, 2, and 3b ruled out shooter clipping, density floors, and the
Cummings cos-θ coordinate respectively. The remaining hypothesis: Rusaitis
(2021) effectively imposed the Dirichlet wall *above* the high-v_A
ionospheric region by truncating the trace at some r > 1 R_S. Our trace
runs all the way to r = 1.0 R_S where Bagenal extrapolates density to
~10⁻⁵ cm⁻³ (unphysically low) and v_A → c — exactly the soft-wall regime
that admits the sub-WKB mode.

This script truncates the KMAG L=8 noon trace at progressively higher
``r_min`` thresholds and reruns the matrix solver on the restricted
profile. If a single ``r_min`` brings mode 1 up to ~0.12 mHz (matching
Rusaitis 2021), the endpoint-location hypothesis is confirmed and the
real fix is a physically defensible inner boundary at r ≈ ionosphere
top (~1.5 R_S above E-layer peak) rather than the planetary surface.

We also report the WKB-asymptotic mode 1, ω_WKB = π/∫(ds/v_A) over the
*truncated* arc length. The Rusaitis ladder follows this asymptote;
mode 1 from the truncated matrix solver should converge to it as the
truncation gets stricter.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qp.fieldline.kmag_model import SaturnField
from qp.wavesolver.matrix_solver import find_eigenfrequencies_matrix
from qp.wavesolver.solver import WavesolverConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_REF_RUSAITIS_MHZ_BY_L = {8.0: 0.12, 10.0: 0.088, 15.0: 0.053}


def _truncated_mode1(
    l_shell: float, r_min_rs: float
) -> tuple[float, float, int, float, float]:
    """Solve the matrix problem on a trace truncated to r ≥ r_min_rs.

    Returns
    -------
    mode1_mhz, wkb_mhz, n_kept, frac_kept, va_at_endpoint
    """
    cfg = WavesolverConfig(
        l_shell=l_shell,
        n_modes=2,
        field=SaturnField(),
        local_time_hours=12.0,
        method="matrix",
    )
    # The matrix solver runs on the uniform-s samples. To truncate by r,
    # we need the per-sample radius. Re-trace and rebuild the profile so
    # we keep the native positions array for the r-mask.
    from qp.fieldline.tracer import saturn_field_wrapper, trace_fieldline_bidirectional
    from qp.wavesolver.density import BagenalDelamere
    from qp.wavesolver.field_profile import compute_field_line_profile

    field_func = saturn_field_wrapper(SaturnField(), cfg.time, coord="KSM")
    import math

    phi = math.radians((cfg.local_time_hours - 12.0) * 15.0)
    start = np.array([l_shell * math.cos(phi), l_shell * math.sin(phi), 0.0])
    trace = trace_fieldline_bidirectional(field_func, start, step=cfg.trace_step, max_steps=100000)
    profile = compute_field_line_profile(trace, BagenalDelamere())

    s = profile.s_samples
    va = profile.va_samples
    h1 = profile.h1_samples
    B = profile.B_samples
    assert s is not None and va is not None and h1 is not None and B is not None

    # Per-sample radius: interpolate r(s) from the native trace.
    from scipy.interpolate import CubicSpline

    r_native = np.linalg.norm(profile.positions, axis=1)
    r_of_s = CubicSpline(profile.arc_length, r_native)(s)

    keep = r_of_s >= r_min_rs
    if keep.sum() < 50:
        return float("nan"), float("nan"), int(keep.sum()), float("nan"), float("nan")

    s_t = s[keep]
    va_t = va[keep]
    h1_t = h1[keep]
    B_t = B[keep]
    # Resample onto a uniform sub-grid for the matrix solver.
    s_uniform = np.linspace(s_t[0], s_t[-1], len(s_t))
    va_u = CubicSpline(s_t, va_t)(s_uniform)
    h1_u = CubicSpline(s_t, h1_t)(s_uniform)
    B_u = CubicSpline(s_t, B_t)(s_uniform)

    modes = find_eigenfrequencies_matrix(
        s_uniform, h1_u, B_u, va_u, n_modes=2, include_eigenfunctions=False
    )
    omega1 = modes[0].angular_frequency
    f1_mhz = omega1 / (2 * np.pi) * 1e3

    phase = np.trapezoid(1.0 / va_u, s_uniform)
    omega_wkb = np.pi / phase
    wkb_mhz = omega_wkb / (2 * np.pi) * 1e3

    frac_kept = float(keep.sum() / len(keep))
    va_endpoint = float(va_u[0])  # equal endpoints by symmetry
    return f1_mhz, wkb_mhz, int(keep.sum()), frac_kept, va_endpoint


def main() -> None:
    out = Path("Output/diagnostics")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    log.info("Trace-truncation sweep across L=8, 10, 15 noon")
    log.info("  L  |  r_min (R_S) |  mode 1 (mHz) |  WKB (mHz) |  ref (mHz) | f1/WKB | n_pts | frac")
    log.info("  " + "-" * 96)
    r_mins = [1.00, 1.05, 1.10, 1.20, 1.30, 1.50, 2.00, 3.00]
    for L, ref in _REF_RUSAITIS_MHZ_BY_L.items():
        for r_min in r_mins:
            f1, fwkb, n_kept, frac, va_end = _truncated_mode1(L, r_min)
            ratio = f1 / fwkb if np.isfinite(f1) and np.isfinite(fwkb) and fwkb > 0 else float("nan")
            log.info(
                "  %.0f  |   %.2f       |   %10.6f  |  %.4f    |   %.4f   |  %.3f |  %5d | %.2f",
                L, r_min, f1, fwkb, ref, ratio, n_kept, frac,
            )
            rows.append(
                {
                    "l_shell": L,
                    "r_min_rs": r_min,
                    "mode1_mhz": f1,
                    "wkb_mhz": fwkb,
                    "ref_mhz": ref,
                    "f1_over_wkb": ratio,
                    "n_kept": n_kept,
                    "frac_kept": frac,
                    "va_endpoint_kms": va_end / 1e3,
                }
            )

    csv_path = out / "trace_truncation_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", csv_path)

    # Plot — mode 1 vs r_min for each L
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {8.0: "C0", 10.0: "C1", 15.0: "C2"}
    for L, ref in _REF_RUSAITIS_MHZ_BY_L.items():
        L_rows = [r for r in rows if r["l_shell"] == L and np.isfinite(r["mode1_mhz"])]
        xs = [r["r_min_rs"] for r in L_rows]
        ys = [r["mode1_mhz"] for r in L_rows]
        wkbs = [r["wkb_mhz"] for r in L_rows]
        ax.plot(xs, ys, "o-", color=colors[L], lw=1.6, ms=6,
                label=f"L={L:.0f} solver mode 1")
        ax.plot(xs, wkbs, "s--", color=colors[L], lw=1.0, ms=4, alpha=0.6,
                label=f"L={L:.0f} WKB asymptote")
        ax.axhline(ref, ls=":", color=colors[L], lw=0.8, alpha=0.7,
                   label=f"L={L:.0f} Rusaitis ref ({ref:.3f} mHz)")
    ax.set_xlabel("trace truncation r_min (R_S)")
    ax.set_ylabel("mode 1 (mHz)")
    ax.set_title("KMAG noon — mode 1 vs trace truncation radius")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "trace_truncation_sweep.png", dpi=120)
    plt.close(fig)
    log.info("wrote %s/trace_truncation_sweep.png", out)

    # Verdict
    log.info("=" * 96)
    log.info("VERDICT")
    for L, ref in _REF_RUSAITIS_MHZ_BY_L.items():
        L_rows = [r for r in rows if r["l_shell"] == L and np.isfinite(r["mode1_mhz"])]
        matches = [r for r in L_rows if abs(r["mode1_mhz"] - ref) / ref < 0.15]
        if matches:
            best = min(matches, key=lambda r: abs(r["r_min_rs"] - 1.0))
            log.info("  L=%.0f: r_min=%.2f R_S reproduces Rusaitis %.3f mHz "
                     "(solver=%.4f mHz, WKB=%.4f mHz)",
                     L, best["r_min_rs"], ref, best["mode1_mhz"], best["wkb_mhz"])
        else:
            log.info("  L=%.0f: no r_min in sweep reproduces %.3f mHz", L, ref)


if __name__ == "__main__":
    main()
