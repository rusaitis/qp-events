"""Phase-2 diagnostic: does an ionospheric density floor reproduce Rusaitis (2021)?

The matrix solver finds a sub-WKB mode 1 at KMAG L=8 noon (0.057 mHz),
roughly a factor of 2 below the published Rusaitis (2021) reference
(0.12 mHz at L=8, 0.088 mHz at L=10, 0.053 mHz at L=15). The Phase-1
diagnostic ruled out shooter clipping. Hypothesis tested here:

  Bagenal's intrinsic density floor (0.07 cm⁻³, equatorial only) is
  unphysically low for the ionospheric footpoint. Off-equator the
  density decays exponentially with no floor, so v_A → c at the
  trace endpoint — producing the soft-wall behaviour that admits the
  sub-WKB mode. Saturn's actual E-layer peaks at ~10⁴ cm⁻³, ~10⁹×
  higher. A physical floor would harden the Dirichlet boundary and
  push mode 1 up to the WKB-asymptotic value, reproducing Rusaitis.

This script sweeps ``WavesolverConfig.density_floor`` (m⁻³) across
six decades from 0 (current behaviour) up to 10¹⁰ m⁻³ (= 10⁴ cm⁻³,
Saturn E-layer peak). For each (L, floor) pair it records mode 1 from
the matrix solver. If a *single* floor value brings all three reference
L-shells (8, 10, 15) within ±15 % of their published values, the floor
hypothesis is validated and we adopt that floor as the new default.
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

# Reference values from Rusaitis (2021) — tests/test_validation.py:27-31
_REFERENCES_MHZ: dict[float, float] = {8.0: 0.12, 10.0: 0.088, 15.0: 0.053}
_MATCH_TOL = 0.15  # ±15 % tolerance for "reproduces" verdict

# Sweep in m⁻³ — 0 (current), 1e5 (=0.1 cm⁻³), then logarithmically up to
# 1e10 (=10⁴ cm⁻³ = Saturn E-layer peak).
_FLOOR_SWEEP_M3 = [
    0.0,
    1.0e5,
    1.0e6,
    3.0e6,
    1.0e7,
    3.0e7,
    1.0e8,
    3.0e8,
    1.0e9,
    3.0e9,
    1.0e10,
]


def _solve_mode1(l_shell: float, density_floor: float) -> float:
    """Return matrix-solver mode-1 frequency (mHz) at given L and floor."""
    floor: float | None = density_floor if density_floor > 0 else None
    cfg = WavesolverConfig(
        l_shell=l_shell,
        n_modes=2,  # 2 is enough — we only inspect mode 1
        field=SaturnField(),
        local_time_hours=12.0,
        method="matrix",
        density_floor=floor,
    )
    r = solve_eigenfrequencies(cfg)
    return float(r.frequencies_mhz[0])


def main() -> None:
    out = Path("Output/diagnostics")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    log.info("Sweeping density floor across L=%s", list(_REFERENCES_MHZ.keys()))
    log.info(
        "  L  |  density_floor (m⁻³)  | mode 1 (mHz) | ref (mHz) | f1/ref | within ±%.0f%%?",
        _MATCH_TOL * 100,
    )
    log.info("  " + "-" * 80)

    for L, ref in _REFERENCES_MHZ.items():
        for floor in _FLOOR_SWEEP_M3:
            try:
                f1 = _solve_mode1(L, floor)
                ratio = f1 / ref
                within = abs(ratio - 1.0) < _MATCH_TOL
                log.info(
                    "  %.0f  |  %.2e            |  %10.6f  |  %.4f  | %.3f  | %s",
                    L,
                    floor,
                    f1,
                    ref,
                    ratio,
                    "✓" if within else " ",
                )
                rows.append(
                    {
                        "l_shell": L,
                        "density_floor_m3": floor,
                        "mode1_mhz": f1,
                        "reference_mhz": ref,
                        "ratio": ratio,
                        "within_tol": float(within),
                    }
                )
            except Exception as exc:
                log.warning("L=%.0f floor=%.2e: %s", L, floor, exc)
                rows.append(
                    {
                        "l_shell": L,
                        "density_floor_m3": floor,
                        "mode1_mhz": float("nan"),
                        "reference_mhz": ref,
                        "ratio": float("nan"),
                        "within_tol": 0.0,
                    }
                )

    csv_path = out / "density_floor_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "l_shell",
                "density_floor_m3",
                "mode1_mhz",
                "reference_mhz",
                "ratio",
                "within_tol",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", csv_path)

    # Plot: mode 1 vs density_floor for each L-shell
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {8.0: "C0", 10.0: "C1", 15.0: "C2"}
    for L, ref in _REFERENCES_MHZ.items():
        L_rows = [r for r in rows if r["l_shell"] == L and np.isfinite(r["mode1_mhz"])]
        xs = [max(r["density_floor_m3"], 1.0) for r in L_rows]  # log axis safety
        ys = [r["mode1_mhz"] for r in L_rows]
        ax.semilogx(
            xs,
            ys,
            "o-",
            color=colors[L],
            lw=1.6,
            ms=6,
            label=f"L={L:.0f}  (ref = {ref:.3f} mHz)",
        )
        ax.axhline(ref, ls="--", color=colors[L], lw=0.8, alpha=0.6)
        ax.fill_between(
            ax.get_xlim(),
            ref * (1 - _MATCH_TOL),
            ref * (1 + _MATCH_TOL),
            color=colors[L],
            alpha=0.07,
        )
    ax.set_xlabel("density floor  (m⁻³, log scale; left edge = 0)")
    ax.set_ylabel("matrix-solver mode 1  (mHz)")
    ax.set_title("KMAG noon — mode 1 vs ionospheric density floor")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "density_floor_sweep.png", dpi=120)
    plt.close(fig)
    log.info("wrote %s/density_floor_sweep.png", out)

    # Verdict — find the floor that brings all three L's within tolerance
    log.info("=" * 72)
    log.info("VERDICT")
    floors_all_match: list[float] = []
    for floor in _FLOOR_SWEEP_M3:
        floor_rows = [
            r
            for r in rows
            if r["density_floor_m3"] == floor and np.isfinite(r["mode1_mhz"])
        ]
        if len(floor_rows) == len(_REFERENCES_MHZ) and all(
            r["within_tol"] for r in floor_rows
        ):
            floors_all_match.append(floor)
    if floors_all_match:
        log.info("  ✓ a single floor reproduces all three reference L-shells:")
        for floor in floors_all_match:
            cm3 = floor / 1e6
            log.info("      floor = %.2e m⁻³ = %.4f cm⁻³", floor, cm3)
        log.info(
            "  ⇒ adopt the *lowest* such floor as the new default "
            "(less perturbation to the model where it already works)."
        )
    else:
        log.info(
            "  ✗ NO single floor reproduces all three L-shells within ±%.0f %%.",
            _MATCH_TOL * 100,
        )
        log.info("    Either:")
        log.info("      (a) the required floor is L-dependent (unphysical)")
        log.info("      (b) the soft-wall mode is more deeply intrinsic and a")
        log.info("          floor alone is insufficient.")
        log.info("    ⇒ proceed to Phase 3b (Cummings cos-θ sibling solver).")


if __name__ == "__main__":
    main()
