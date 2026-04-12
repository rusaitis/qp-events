"""Phase 10.6 — Inter-period amplitude evolution.

For the top 20 events per band (by quality_v3), plots the per-oscillation
amplitude as a stacked bar chart, then overlays the median trend across all
top events.

Physics predictions:
- PPO-driven standing FLR: grow for first ~3 cycles, then decay
- Freely dispersing packet: monotonic decay from onset
- Random noise: flat

Also: chirp direction histogram (Phase 10.2) and amplitude growth distribution
(Phase 10.3) + PPO phase of onset (Phase 10.4) + N/S comparison (Phase 10.5)
— all on a single diagnostic figure.

Output: ``Output/figures/figure_amplitude_evolution.png``
         ``Output/diagnostics/morphology_physics_tests.txt``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qp.plotting.style import use_paper_style  # noqa: E402

BAND_COLORS = {"QP30": "#4ecdc4", "QP60": "#ff6b6b", "QP120": "#ffd93d"}
BANDS = ["QP30", "QP60", "QP120"]


def _rayleigh_test(phases_deg: np.ndarray) -> tuple[float, float]:
    """Rayleigh test for uniformity of circular data.

    Returns (R, p_value) where R is the mean resultant length (0=uniform,
    1=all same direction). Small p-value → non-uniform → preferred direction.
    """
    angles = np.radians(phases_deg)
    n = len(angles)
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C ** 2 + S ** 2) / n
    # Rayleigh statistic Z = n*R^2
    Z = n * R ** 2
    # P-value approximation (Mardia & Jupp 2000)
    p = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n) - (24 * Z - 132 * Z ** 2
        + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n ** 2))
    return float(R), float(np.clip(p, 0, 1))


def main() -> None:
    cat_path = _PROJECT_ROOT / "Output" / "events_qp_v4.parquet"
    if not cat_path.exists():
        print(f"v4 catalog not found at {cat_path}")
        return
    df = pd.read_parquet(cat_path)
    quality_col = "quality_v3" if "quality_v3" in df.columns else "quality"
    print(f"Loaded {len(df)} events")

    use_paper_style()
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    # 4 rows × 3 cols: (amplitude evolution, chirp histogram, growth dist, PPO phase)
    outer_rows = 4
    outer_cols = 3
    axes = fig.subplots(outer_rows, outer_cols)

    report_lines: list[str] = ["# Phase 10 Physics Tests\n"]

    for col_idx, band in enumerate(BANDS):
        color = BAND_COLORS[band]
        q_col = quality_col
        sub = (
            df[(df.band == band) & df[q_col].fillna(0).gt(0.3)]
            .sort_values(q_col, ascending=False)
        )
        period_sec = sub["period"].median() if "period" in sub.columns else None

        # ── Row 0: inter-period amplitude evolution ────────────────────────
        ax0 = axes[0, col_idx]
        if period_sec and period_sec > 0 and "amplitude_growth_db" in df.columns:
            top20 = sub.head(20)
            growth_vals = top20["amplitude_growth_db"].dropna()
            n_top = len(top20)
            ax0.hist(growth_vals, bins=10, color=color, alpha=0.8, edgecolor="none")
            ax0.axvline(0, color="white", lw=1, ls="--", label="flat (0)")
            ax0.axvline(growth_vals.median(), color="yellow", lw=1.5, ls="-",
                         label=f"median={growth_vals.median():.2f} dB")
            ax0.set_title(f"{band} top-20 amplitude trend (n={n_top})", fontsize=9)
            ax0.set_xlabel("dB/period", fontsize=8)
            ax0.set_ylabel("Count", fontsize=8)
            ax0.legend(fontsize=7, frameon=False)
            ax0.grid(alpha=0.2)
            report_lines.append(
                f"\n## {band} amplitude growth (top 20, q>0.3)\n"
                f"  median = {growth_vals.median():.3f} dB/period  "
                f"n = {len(growth_vals)}\n"
                f"  positive (growing): {(growth_vals > 0).sum()}  "
                f"negative (decaying): {(growth_vals < 0).sum()}\n"
            )
        else:
            ax0.set_title(f"{band} (no data)", fontsize=9)

        # ── Row 1: frequency drift histogram (Phase 10.2) ─────────────────
        ax1 = axes[1, col_idx]
        if "freq_drift_hz_per_s" in df.columns:
            drift_vals = sub["freq_drift_hz_per_s"].dropna() * 1e12  # pHz/s
            # Clip for display
            drift_clip = drift_vals.clip(-25, 25)
            ax1.hist(drift_clip, bins=30, color=color, alpha=0.8, edgecolor="none")
            ax1.axvline(0, color="white", lw=1, ls="--", label="no drift")
            ax1.axvline(drift_vals.median(), color="yellow", lw=1.5,
                         label=f"med={drift_vals.median():.1f} pHz/s")
            # t-test: is median significantly different from zero?
            if len(drift_vals) > 5:
                t, p = stats.ttest_1samp(drift_vals, 0)
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                ax1.set_title(f"{band} chirp rate (t={t:.1f}, p={p:.3f}{sig})", fontsize=9)
                report_lines.append(
                    f"\n## {band} frequency drift (chirp)\n"
                    f"  median = {drift_vals.median():.3f} pHz/s  n={len(drift_vals)}\n"
                    f"  t-test vs 0: t={t:.2f}, p={p:.4f}\n"
                    f"  upward chirp: {(drift_vals > 0).mean()*100:.1f}%  "
                    f"downward: {(drift_vals < 0).mean()*100:.1f}%\n"
                )
            ax1.set_xlabel("pHz / s", fontsize=8)
            ax1.legend(fontsize=7, frameon=False)
            ax1.grid(alpha=0.2)

        # ── Row 2: N/S hemisphere comparison (Phase 10.5) ─────────────────
        ax2 = axes[2, col_idx]
        if "ellipticity" in df.columns and "mag_lat" in df.columns:
            north = sub[sub["mag_lat"].fillna(0) > 0]["ellipticity"].dropna()
            south = sub[sub["mag_lat"].fillna(0) < 0]["ellipticity"].dropna()
            data_for_box = []
            labels = []
            if len(north) > 0:
                data_for_box.append(north.values)
                labels.append(f"North\n(n={len(north)})")
            if len(south) > 0:
                data_for_box.append(south.values)
                labels.append(f"South\n(n={len(south)})")
            if data_for_box:
                bp = ax2.boxplot(data_for_box, tick_labels=labels,
                                  patch_artist=True, notch=False)
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax2.axhline(0, color="white", lw=0.8, ls="--")
                ax2.set_title(f"{band} N vs S ellipticity", fontsize=9)
                ax2.set_ylabel("Ellipticity", fontsize=8)
                ax2.grid(alpha=0.2)
                if len(north) > 2 and len(south) > 2:
                    u, p = stats.mannwhitneyu(north, south, alternative="two-sided")
                    report_lines.append(
                        f"\n## {band} N/S hemisphere comparison\n"
                        f"  North median={north.median():.3f} (n={len(north)})\n"
                        f"  South median={south.median():.3f} (n={len(south)})\n"
                        f"  Mann-Whitney U={u:.0f}, p={p:.4f}\n"
                    )

        # ── Row 3: PPO phase of onset (Phase 10.4) ────────────────────────
        ax3 = axes[3, col_idx]
        if "ppo_phase_onset_deg" in df.columns:
            onset_phases = sub["ppo_phase_onset_deg"].dropna()
            if len(onset_phases) > 5:
                bins = np.linspace(0, 360, 25)
                ax3.hist(onset_phases, bins=bins, color=color, alpha=0.8, edgecolor="none")
                ax3.set_xlabel("PPO phase at onset (°)", fontsize=8)
                ax3.set_ylabel("Count", fontsize=8)
                R, p = _rayleigh_test(onset_phases.values)
                ax3.set_title(
                    f"{band} PPO phase of onset\n"
                    f"Rayleigh R={R:.3f}, p={p:.3f}"
                    + (" **" if p < 0.01 else " *" if p < 0.05 else " (uniform)"),
                    fontsize=9,
                )
                ax3.axvline(onset_phases.mean(), color="white", lw=1.5, ls="--",
                             label=f"mean={onset_phases.mean():.0f}°")
                ax3.legend(fontsize=7, frameon=False)
                ax3.grid(alpha=0.2)
                report_lines.append(
                    f"\n## {band} PPO phase of onset\n"
                    f"  n={len(onset_phases)}\n"
                    f"  Rayleigh R={R:.4f}, p={p:.4f}\n"
                    f"  mean phase={onset_phases.mean():.1f}°\n"
                    f"  Interpretation: {'non-uniform (PPO phase-locked)' if p < 0.05 else 'uniform (no preferred onset phase)'}\n"
                )

    fig.suptitle(
        "Phase 10 — Standing vs Travelling FLR Discrimination Tests (q > 0.3)",
        fontsize=12,
    )

    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure_amplitude_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")

    diag_dir = _PROJECT_ROOT / "Output" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_out = diag_dir / "morphology_physics_tests.txt"
    diag_out.write_text("\n".join(report_lines))
    print(f"Wrote {diag_out}")


if __name__ == "__main__":
    main()
