"""Phase-1 diagnostic: does the shooter's bracket-scan clip the sub-WKB mode?

The matrix solver finds an additional sub-WKB mode 1 at KMAG L=8 noon
(~0.057 mHz = 3.58e-4 rad/s) below the WKB-asymptotic mode (~0.126 mHz).
The published Rusaitis (2021) reference (0.12 mHz) matches the WKB
asymptotic, not the soft-wall mode. One hypothesis: Rusaitis's solver
used a bracket-scan with a frequency lower bound *above* our sub-WKB
mode, missing it entirely.

This script sweeps ``WavesolverConfig.freq_range[0]`` across six decades
and records which mode the shooter returns first. The matrix solver is
run once as a reference (immune to freq_range clipping). Outcome:

  - Case A (expected): shooter returns 0.057 mHz for every lower-bound
    ≲ 3.5e-4 rad/s and clips above. The clipping threshold matches the
    actual mode-1 frequency — a generic property of any bracket-scan
    solver, not a specific bug.
  - Case B: shooter misses the sub-WKB mode even at very low freq_range[0]
    (e.g., 1e-7). This would indicate a deeper bug in the bracket scan
    (e.g., insufficient resolution to catch the steep sub-WKB sign change)
    and would more strongly support the "Rusaitis missed mode 1" hypothesis.

Either outcome is informative: Case A says the clipping hypothesis is
*possible* but speculative without Rusaitis's actual freq_range. Case B
says it's the most parsimonious explanation.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import setup_logging

from qp.fieldline.kmag_model import SaturnField
from qp.wavesolver.solver import WavesolverConfig, solve_eigenfrequencies

setup_logging()
log = logging.getLogger(__name__)

# Same upper bound throughout (well above the WKB-spacing mode ~1.27e-3).
_FREQ_HIGH = 1e-2
_RESOLUTION = 400
_L_SHELL = 8.0
_REF_RUSAITIS_MHZ = 0.12  # Rusaitis et al. 2021 KMAG L=8 reference


def _solve(
    method: str, *, freq_low: float, n_modes: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Return (omegas, freqs_mHz) for the configured backend."""
    cfg = WavesolverConfig(
        l_shell=_L_SHELL,
        n_modes=n_modes,
        field=SaturnField(),
        local_time_hours=12.0,
        freq_range=(freq_low, _FREQ_HIGH),
        resolution=_RESOLUTION,
        method=method,
    )
    r = solve_eigenfrequencies(cfg)
    return r.angular_frequencies, r.frequencies_mhz


def main() -> None:
    out = Path("Output/diagnostics")
    out.mkdir(parents=True, exist_ok=True)

    # Matrix solver (single run — immune to freq_range)
    _, fs_mat = _solve("matrix", freq_low=1e-7)
    f1_matrix = float(fs_mat[0])
    log.info("=" * 72)
    log.info("Matrix solver baseline (KMAG L=%.0f noon)", _L_SHELL)
    log.info(
        "  mode 1 = %.6f mHz   (Rusaitis 2021 ref = %.3f mHz)",
        f1_matrix,
        _REF_RUSAITIS_MHZ,
    )
    log.info("  full ladder: %s", "  ".join(f"{f:.4f}" for f in fs_mat))

    # Shooter sweep over freq_range[0]
    sweep = [1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3]
    rows: list[dict[str, float]] = []
    log.info("=" * 72)
    log.info(
        "Shooter sweep over freq_range[0]  (KMAG L=%.0f, freq_high=%.0e, resolution=%d)",
        _L_SHELL,
        _FREQ_HIGH,
        _RESOLUTION,
    )
    log.info(
        "  freq_low (rad/s)  |  shooter mode 1 (mHz)  |  Δ vs matrix (mHz)  |  n_modes"
    )
    log.info("  " + "-" * 76)
    for freq_low in sweep:
        try:
            _, fs = _solve("shoot", freq_low=freq_low)
            f1 = float(fs[0]) if len(fs) else float("nan")
            delta = f1 - f1_matrix
            n_m = len(fs)
            log.info("  %16.2e  |  %20.6f  |  %18.6f  |  %d", freq_low, f1, delta, n_m)
            rows.append(
                {
                    "freq_low_rads": freq_low,
                    "shooter_f1_mHz": f1,
                    "delta_vs_matrix_mHz": delta,
                    "n_modes": n_m,
                }
            )
        except Exception as exc:
            log.warning("  shooter failed at freq_low=%.2e: %s", freq_low, exc)
            rows.append(
                {
                    "freq_low_rads": freq_low,
                    "shooter_f1_mHz": float("nan"),
                    "delta_vs_matrix_mHz": float("nan"),
                    "n_modes": 0,
                }
            )

    # CSV
    csv_path = out / "shooter_clipping_l8.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "freq_low_rads",
                "shooter_f1_mHz",
                "delta_vs_matrix_mHz",
                "n_modes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", csv_path)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = [r["freq_low_rads"] for r in rows if np.isfinite(r["shooter_f1_mHz"])]
    ys = [r["shooter_f1_mHz"] for r in rows if np.isfinite(r["shooter_f1_mHz"])]
    ax.semilogx(xs, ys, "o-", label="shooter mode 1", lw=1.6, ms=7)
    ax.axhline(
        f1_matrix,
        ls="--",
        color="C2",
        lw=1.6,
        label=f"matrix mode 1 = {f1_matrix:.4f} mHz",
    )
    ax.axhline(
        _REF_RUSAITIS_MHZ,
        ls=":",
        color="C3",
        lw=1.6,
        label=f"Rusaitis 2021 ref = {_REF_RUSAITIS_MHZ:.3f} mHz",
    )
    ax.set_xlabel("shooter freq_range[0]  (rad/s, log scale)")
    ax.set_ylabel("mode 1  (mHz)")
    ax.set_title(
        f"KMAG L={_L_SHELL:.0f} noon — does the shooter clip the sub-WKB mode?"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "shooter_clipping_l8.png", dpi=120)
    plt.close(fig)
    log.info("wrote %s/shooter_clipping_l8.png", out)

    # Verdict
    log.info("=" * 72)
    log.info("VERDICT")
    valid = [r for r in rows if np.isfinite(r["shooter_f1_mHz"])]
    matches = [r for r in valid if abs(r["delta_vs_matrix_mHz"]) < 0.005]
    clips = [r for r in valid if abs(r["delta_vs_matrix_mHz"]) >= 0.005]
    log.info(
        "  shooter agreed with matrix (Δ<5e-3 mHz) for %d/%d settings",
        len(matches),
        len(valid),
    )
    log.info(
        "  shooter diverged (Δ≥5e-3 mHz) for %d/%d settings", len(clips), len(valid)
    )
    if clips:
        clip_thresh = min(r["freq_low_rads"] for r in clips)
        clip_f1 = next(
            r["shooter_f1_mHz"] for r in clips if r["freq_low_rads"] == clip_thresh
        )
        log.info(
            "  earliest divergence at freq_range[0] = %.2e rad/s "
            "(shooter f1 = %.4f mHz)",
            clip_thresh,
            clip_f1,
        )
        if 0.10 < clip_f1 < 0.14:
            log.info(
                "  ⇒ clipping at this threshold yields ~0.12 mHz "
                "(matches Rusaitis 2021)."
            )
        else:
            log.info(
                "  ⇒ post-clip f1 = %.4f mHz does not match Rusaitis "
                "(0.12 mHz); clipping not the reconciliation.",
                clip_f1,
            )
    else:
        log.info("  shooter agreed across the full sweep — no clipping observed.")
        log.info(
            "  hypothesis (Rusaitis 2021 missed mode 1 via bracket-scan) "
            "is UNLIKELY for any freq_range[0] in [%.0e, %.0e].",
            sweep[0],
            sweep[-1],
        )


if __name__ == "__main__":
    main()
