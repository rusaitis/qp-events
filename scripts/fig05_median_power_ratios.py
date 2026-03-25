"""Figure 5: Median power ratios by local time quadrant (CENTRAL RESULT).

4-panel plot showing median power ratios of MFA field perturbations
vs frequency for four local time sectors: 0±3h, 6±3h, 12±3h, 18±3h.
Quartile shading shows the spread. QP30/QP60/QP120 peaks are visible.

Referee: white dashed line at 50-min period.
"""

import sys
import types
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]

# Register stubs for legacy pickle
_stub_classes = ["SignalSnapshot", "NewSignal", "Interval", "FFT_list", "WaveSignal", "Wave"]
_stub_modules = ["__main__", "data_sweeper", "mag_fft_sweeper", "cassinilib", "cassinilib.NewSignal"]
for mod_path in _stub_modules:
    if mod_path not in sys.modules:
        sys.modules[mod_path] = types.ModuleType(mod_path)
    for cls_name in _stub_classes:
        setattr(sys.modules[mod_path], cls_name, type(cls_name, (), {}))

sys.path.insert(0, str(_project_root / "src"))

import qp
from qp.signal.power_ratio import compute_power_ratios
from qp.plotting.style import (
    FIELD_COLORS_SPECTRA, use_paper_style, style_axes,
    draw_period_lines, plot_segmented, BG_COLOR,
)


# Local time quadrants: (center, half-width, label, panel_label)
LT_QUADRANTS = [
    (0, 3, r"Local Time = 0 h $\pm$ 3h", "a"),
    (6, 3, r"Local Time = 6 h $\pm$ 3h", "b"),
    (12, 3, r"Local Time = 12 h $\pm$ 3h", "c"),
    (18, 3, r"Local Time = 18 h $\pm$ 3h", "d"),
]

LABELS = [r"$B_{\parallel}$", r"$B_{\perp 1}$", r"$B_{\perp 2}$", r"$B_{total}$"]
RATIO_KEYS = ["r_par", "r_perp1", "r_perp2", "r_total"]

# Welch parameters matching the paper
DT = 60.0           # 1-minute resolution
NPERSEG = 12 * 60   # 12-hour window
NOVERLAP = 6 * 60   # 6-hour overlap


def in_lt_range(lt, center, half_width):
    """Check if local time is within center ± half_width, handling wraparound."""
    lo = (center - half_width) % 24
    hi = (center + half_width) % 24
    if lo < hi:
        return lo <= lt < hi
    else:  # wraps around midnight
        return lt >= lo or lt < hi


def compute_all_ratios(data):
    """Compute power ratios for each valid segment, grouped by LT quadrant.

    Returns dict: {quadrant_idx: list of ratio dicts}
    """
    quadrant_ratios = {i: [] for i in range(4)}
    n_total = len(data)
    n_valid = 0
    n_skipped = 0

    for idx, seg in enumerate(data):
        if idx % 500 == 0:
            print(f"  Processing segment {idx}/{n_total}...")

        # Filter: no flag, magnetosphere only
        if seg.flag is not None:
            n_skipped += 1
            continue
        if not hasattr(seg, "info") or not isinstance(seg.info, dict):
            n_skipped += 1
            continue
        location = seg.info.get("location")
        if location != 0:  # 0 = magnetosphere
            n_skipped += 1
            continue
        median_lt = seg.info.get("median_LT")
        if median_lt is None:
            n_skipped += 1
            continue

        # Get field data
        if not hasattr(seg, "FIELDS") or len(seg.FIELDS) < 4:
            n_skipped += 1
            continue

        b_par = seg.FIELDS[0].y
        b_perp1 = seg.FIELDS[1].y
        b_perp2 = seg.FIELDS[2].y
        b_tot = seg.FIELDS[3].y

        if b_par is None or len(b_par) < NPERSEG:
            n_skipped += 1
            continue

        # Trim to central 24h (skip 6h padding on each end)
        # 36h = 2160 pts, central 24h = pts 360:1800
        pad = 6 * 60  # 6 hours in minutes
        if len(b_par) > 2 * pad:
            sl = slice(pad, len(b_par) - pad)
        else:
            sl = slice(None)

        try:
            ratios = compute_power_ratios(
                b_par[sl], b_perp1[sl], b_perp2[sl], b_tot[sl],
                dt=DT, nperseg=NPERSEG, noverlap=NOVERLAP, window="hann",
            )
        except Exception:
            n_skipped += 1
            continue

        # Check for valid output
        if np.any(np.isnan(ratios["r_perp1"])) or np.any(np.isinf(ratios["r_perp1"])):
            n_skipped += 1
            continue

        # Assign to LT quadrants
        for qi, (center, hw, _, _) in enumerate(LT_QUADRANTS):
            if in_lt_range(median_lt, center, hw):
                quadrant_ratios[qi].append(ratios)
                break

        n_valid += 1

    print(f"  Valid: {n_valid}, Skipped: {n_skipped}")
    for qi, (_, _, label, _) in enumerate(LT_QUADRANTS):
        print(f"  {label}: {len(quadrant_ratios[qi])} segments")

    return quadrant_ratios


def compute_statistics(quadrant_ratios):
    """Compute median and quartiles for each quadrant."""
    stats = {}
    for qi in range(4):
        ratio_list = quadrant_ratios[qi]
        if len(ratio_list) == 0:
            stats[qi] = None
            continue

        freq = ratio_list[0]["freq"]

        result = {"freq": freq}
        for key in RATIO_KEYS:
            all_ratios = np.array([r[key] for r in ratio_list])
            result[f"{key}_median"] = np.median(all_ratios, axis=0)
            result[f"{key}_q1"] = np.quantile(all_ratios, 0.25, axis=0)
            result[f"{key}_q3"] = np.quantile(all_ratios, 0.75, axis=0)
        result["n_segments"] = len(ratio_list)
        stats[qi] = result

    return stats


def draw_clock_icon(ax, center_lt, half_width):
    """Draw a small clock dial showing the LT sector."""
    # Position in axes coordinates (top-left)
    inset = ax.inset_axes([0.02, 0.72, 0.15, 0.25])
    inset.set_facecolor(BG_COLOR)

    # Draw circle
    theta = np.linspace(0, 2 * np.pi, 100)
    inset.plot(np.cos(theta), np.sin(theta), color="white", lw=1)

    # Shade the active sector
    lt_start = (center_lt - half_width) * np.pi / 12 - np.pi / 2
    lt_end = (center_lt + half_width) * np.pi / 12 - np.pi / 2
    wedge_theta = np.linspace(lt_start, lt_end, 50)
    wedge_x = np.concatenate([[0], np.cos(wedge_theta), [0]])
    wedge_y = np.concatenate([[0], np.sin(wedge_theta), [0]])
    inset.fill(wedge_x, wedge_y, color="orange", alpha=0.4)

    # Mark noon (top) and midnight (bottom)
    inset.text(0, 1.2, "12", ha="center", va="bottom", fontsize=6, color="white")
    inset.text(0, -1.3, "0", ha="center", va="top", fontsize=6, color="white")
    inset.text(1.2, 0, "6", ha="left", va="center", fontsize=6, color="white")
    inset.text(-1.3, 0, "18", ha="right", va="center", fontsize=6, color="white")
    inset.text(-0.2, -1.6, "Local\nTime", ha="center", va="top", fontsize=5, color="grey")

    inset.set_xlim(-1.8, 1.8)
    inset.set_ylim(-2.0, 1.6)
    inset.set_aspect("equal")
    inset.axis("off")


def main():
    print("Loading MFA 36H data...")
    data = np.load(qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy", allow_pickle=True)
    print(f"Loaded {len(data)} segments")

    print("Computing power ratios for all segments...")
    quadrant_ratios = compute_all_ratios(data)

    print("Computing statistics...")
    stats = compute_statistics(quadrant_ratios)

    # --- Plot ---
    use_paper_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.22, wspace=0.18, left=0.08, right=0.95,
                        top=0.95, bottom=0.07)

    colors = FIELD_COLORS_SPECTRA

    for qi, ax in enumerate(axes.flat):
        s = stats[qi]
        if s is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", color="white")
            continue

        freq = s["freq"]
        n_seg = s["n_segments"]

        # Plot each component: median line + quartile fill
        for ki, (key, label, color) in enumerate(zip(RATIO_KEYS, LABELS, colors)):
            median = s[f"{key}_median"]
            q1 = s[f"{key}_q1"]
            q3 = s[f"{key}_q3"]

            # Segment-based plotting for median
            plot_segmented(ax, freq, median, lw=2.5, color=color, label=label)

            # Quartile shading (full range, no segmentation needed)
            ax.fill_between(
                freq / 1e-3, q1, q3,
                color=color, alpha=0.15,
            )

        # Unity line
        ax.axhline(1.0, ls="--", lw=1, color="grey", alpha=0.4)

        # 50-min dashed line (referee request)
        f_50min = 1.0 / (50 * 60)
        ax.axvline(f_50min / 1e-3, ls="--", lw=2, color="white", alpha=0.7)

        # Period annotations
        draw_period_lines(ax, periods_min=[30, 60, 90, 120, 180],
                          lw=1.5, alpha=0.3, fontsize=10)

        # Axes
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(freq[1] / 1e-3, freq[-1] / 1e-3)
        ax.set_ylim(5e-2, 5e2)
        ax.set_xlabel("Frequency (mHz)", fontsize=12)
        ax.set_ylabel("Power Ratio", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        style_axes(ax, minimal=False)

        # Title
        _, _, title, panel_label = LT_QUADRANTS[qi]
        ax.set_title(title, fontsize=14, color="orange")
        ax.text(0.01, 0.97, panel_label, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="top", color="white")

        # Clock icon
        center_lt, hw = LT_QUADRANTS[qi][0], LT_QUADRANTS[qi][1]
        draw_clock_icon(ax, center_lt, hw)

    # Legend at bottom
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 0.01))

    plt.savefig("output/figure5.png", dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved output/figure5.png")
    plt.close()


if __name__ == "__main__":
    main()
