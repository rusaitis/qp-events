"""Figure 11 (round-8) — QP wave events folded by SLS5 PPO phase.

The published Fig 9 shows the 10.7 h modulation indirectly via
wave-train separation times. This figure shows the same modulation
**directly** in PPO phase space, which is what the proposed driver
theory (periodic magnetotail flapping at the PPO period) predicts: a
double-peaked distribution with maxima near phases at which the tail
crosses the magnetic equator.

Reads ``Output/events_round8.parquet`` (round-8 detector, 1881 events)
and bins peak times against ``sls5_phase_n`` and ``sls5_phase_s``
(degrees, already populated by the sweep). For each QP band we plot
two histograms (SLS5N, SLS5S) with the uniform-distribution reference
line and Rayleigh-test p-value annotated.

Output: ``Output/figures/figure11_ppo_phase_fold.png``
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import OUTPUT_DIR, ensure_figures_dir, setup_logging  # noqa: E402

from qp.events.bands import QP_BAND_COLORS, QP_BAND_NAMES  # noqa: E402
from qp.plotting.style import use_paper_style  # noqa: E402

log = logging.getLogger(__name__)

BANDS = QP_BAND_NAMES
BAND_COLORS = QP_BAND_COLORS


def _rayleigh_p(phases_deg: np.ndarray) -> float:
    r"""Rayleigh test for non-uniformity on the circle.

    Returns the p-value under the null hypothesis of a uniform
    distribution on $[0, 2\pi)$. Small p means the distribution has a
    preferred direction — i.e. PPO phase modulation.
    """
    n = phases_deg.size
    if n == 0:
        return float("nan")
    phi = np.deg2rad(phases_deg)
    R = np.hypot(np.cos(phi).sum(), np.sin(phi).sum()) / n
    # Wilkie (1983) approximation, accurate for n >= 10
    z = n * R**2
    p = np.exp(-z) * (1.0 + (2 * z - z**2) / (4 * n))
    return float(p)


def main() -> None:
    setup_logging()

    parquet = OUTPUT_DIR / "events_round8.parquet"
    if not parquet.exists():
        raise SystemExit(f"missing {parquet} — run sweep_events_round8.py first")

    df = pd.read_parquet(parquet)
    log.info("loaded %d events from %s", len(df), parquet.name)
    if "is_duplicate" in df.columns:
        n_dup = int(df["is_duplicate"].sum())
        df = df.loc[~df["is_duplicate"]].reset_index(drop=True)
        log.info("dropped %d duplicate rows (post-hoc dedup)", n_dup)

    by_band = {b: df[df["band"] == b] for b in BANDS}
    bins = np.linspace(0, 360, 25)  # 15 deg bins

    use_paper_style()
    fig, axes = plt.subplots(
        2,
        len(BANDS),
        figsize=(4.3 * len(BANDS), 7),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    for col, band in enumerate(BANDS):
        sub = by_band[band]
        for row, key in enumerate(("sls5_phase_n", "sls5_phase_s")):
            ax = axes[row, col]
            phases = sub[key].dropna().to_numpy()
            if phases.size == 0:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                continue
            counts, edges = np.histogram(phases, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            uniform = phases.size / (bins.size - 1)
            color = BAND_COLORS[band] if row == 0 else "#7fcdcd"
            ax.bar(
                centers,
                counts,
                width=14,
                color=color,
                alpha=0.85,
                edgecolor="black",
                lw=0.3,
            )
            ax.axhline(uniform, color="grey", lw=1, ls="--", label="uniform")
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])
            if row == 1:
                ax.set_xlabel("PPO phase [deg]")
            label = "SLS5N" if row == 0 else "SLS5S"
            p_ray = _rayleigh_p(phases)
            ax.set_title(
                f"{band} — {label}  (n={phases.size}, p$_R$={p_ray:.2g})",
                fontsize=10,
            )
            if col == 0:
                ax.set_ylabel("Event count")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Figure 11 (round-8) — QP wave events vs SLS5 PPO phase",
        fontsize=12,
    )
    out = ensure_figures_dir() / "figure11_ppo_phase_fold.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
