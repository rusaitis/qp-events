"""Phase 9.4 — Morphology distribution plots.

Violin + strip plots per QP band for:
- rise_fall_ratio
- harmonic_ratio_2f
- amplitude_growth_db
- freq_drift_hz_per_s (scaled to pHz/s)
- inter_cycle_coherence

Physics reference lines mark theoretical predictions for standing vs
travelling modes.

Output: ``Output/figures/figure_morphology_distributions.png``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qp.plotting.style import use_paper_style  # noqa: E402

BAND_COLORS = {"QP30": "#4ecdc4", "QP60": "#ff6b6b", "QP120": "#ffd93d"}
BANDS = ["QP30", "QP60", "QP120"]


def _violin_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    col: str,
    ylabel: str,
    ref_lines: list[tuple[float, str, str]] | None = None,
    scale: float = 1.0,
    quality_col: str = "quality_v3",
    min_quality: float = 0.0,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Draw violin + median marker per band."""
    data_by_band = []
    positions = []
    for i, band in enumerate(BANDS):
        sub = df[(df.band == band) & (df[quality_col].fillna(0) >= min_quality)]
        vals = sub[col].dropna() * scale
        data_by_band.append(vals.values)
        positions.append(i + 1)

    # Violin
    parts = ax.violinplot(
        data_by_band, positions=positions,
        showmedians=True, showextrema=False,
    )
    for i, (pc, band) in enumerate(zip(parts["bodies"], BANDS)):
        pc.set_facecolor(BAND_COLORS[band])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    # Reference lines
    if ref_lines:
        for yval, label, ls in ref_lines:
            ax.axhline(yval * scale, color="white", lw=1.0, ls=ls,
                        alpha=0.7, label=label)
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    ax.set_xticks(positions)
    ax.set_xticklabels(BANDS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2, axis="y")

    # Print medians
    for i, (band, vals) in enumerate(zip(BANDS, data_by_band)):
        if len(vals):
            print(f"  {col} {band}: n={len(vals)}, med={np.median(vals):.4g}")


def main() -> None:
    cat_path = _PROJECT_ROOT / "Output" / "events_qp_v4.parquet"
    if not cat_path.exists():
        print(f"v4 catalog not found: {cat_path}")
        return
    df = pd.read_parquet(cat_path)
    quality_col = "quality_v3" if "quality_v3" in df.columns else "quality"
    print(f"Loaded {len(df)} events")

    use_paper_style()
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), constrained_layout=True)

    print("\nrise_fall_ratio:")
    _violin_panel(
        axes[0], df, "rise_fall_ratio",
        "Rise / Fall time",
        ref_lines=[
            (1.0, "symmetric (=1)", "--"),
        ],
        quality_col=quality_col,
        ylim=(0, 6),
    )
    axes[0].set_title("Envelope asymmetry\n(>1 = slow rise, fast fall)", fontsize=9)

    print("\nharmonic_ratio_2f:")
    _violin_panel(
        axes[1], df, "harmonic_ratio_2f",
        "P(2f) / P(f)",
        ref_lines=[
            (0.0, "pure sine (0)", "--"),
            (0.25, "sawtooth (0.25)", ":"),
        ],
        quality_col=quality_col,
        ylim=(0, 1.5),
    )
    axes[1].set_title("Harmonic content\n(0 = sine, 0.25 = sawtooth)", fontsize=9)

    print("\namplitude_growth_db:")
    _violin_panel(
        axes[2], df, "amplitude_growth_db",
        "dB / period",
        ref_lines=[
            (0.0, "flat (0 dB)", "--"),
        ],
        quality_col=quality_col,
        ylim=(-8, 5),
    )
    axes[2].set_title("Amplitude growth rate\n(>0 = growing, <0 = decaying)", fontsize=9)

    print("\nfreq_drift_hz_per_s (→ pHz/s):")
    _violin_panel(
        axes[3], df, "freq_drift_hz_per_s",
        "pHz / s",
        ref_lines=[
            (0.0, "no drift (0)", "--"),
        ],
        scale=1e12,  # Hz/s → pHz/s
        quality_col=quality_col,
        ylim=(-30, 30),
    )
    axes[3].set_title("Frequency drift (chirp)\n(0 = standing, ≠0 = dispersive)", fontsize=9)

    print("\ninter_cycle_coherence:")
    _violin_panel(
        axes[4], df, "inter_cycle_coherence",
        "Cycle correlation",
        ref_lines=[
            (1.0, "perfect (1)", "--"),
            (0.0, "incoherent (0)", ":"),
        ],
        quality_col=quality_col,
        ylim=(-0.5, 1.1),
    )
    axes[4].set_title("Inter-cycle coherence\n(1 = identical cycles)", fontsize=9)

    # Annotation box with physics interpretation
    fig.text(
        0.01, 0.01,
        "Standing FLR: symmetric envelope (ratio≈1), zero chirp, moderate coherence\n"
        "Travelling packet: asymmetric, nonzero chirp, lower coherence",
        fontsize=8, color="lightgrey", va="bottom",
    )

    fig.suptitle(
        "Waveform Morphology Distributions (Phase 9) — all 1636 events by band",
        fontsize=11,
    )

    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure_morphology_distributions.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
