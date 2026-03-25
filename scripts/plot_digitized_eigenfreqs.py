"""Extract and validate digitized eigenfrequencies from graph digitizer exports.

Reads pixel-coordinate JSON files from the online graph digitizer, converts
to physical (latitude, frequency) values using axis calibration, saves a
clean CSV, and plots for visual comparison with paper/figure6.jpeg.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODE_COLORS = {
    1: "#1f77b4",  # blue
    2: "#ff69b4",  # pink
    3: "#2ca02c",  # green
    4: "#d62728",  # red
    5: "#9467bd",  # purple
    6: "#8c564b",  # brown
}

PAPER_DIR = Path("paper")
OUTPUT_DIR = Path("output")


def extract_calibration_0lt(data: dict) -> tuple[float, float, float, float]:
    """Extract axis calibration from properly set axisSets."""
    axis = data["axisSets"][0]
    x1_px = axis["x1"]["coord"]["xPx"]
    x2_px = axis["x2"]["coord"]["xPx"]
    y1_px = axis["y1"]["coord"]["yPx"]  # y=0.1
    y2_px = axis["y2"]["coord"]["yPx"]  # y=1.0
    return x1_px, x2_px, y1_px, y2_px


def extract_calibration_12lt(data: dict) -> tuple[float, float, float, float]:
    """Extract calibration from first 4 points of m1-p dataset.

    The 12LT file has -999 placeholders in axisSets. The user placed the
    axis calibration as the first 4 points of the m1-p dataset:
    point 1 = x1 (bottom-left), point 2 = x2 (bottom-right),
    point 3 = y1 (0.1 level), point 4 = y2 (1.0 level).
    """
    m1_points = data["datasets"][0]["points"]
    x1_px = m1_points[0]["xPx"]
    x2_px = m1_points[1]["xPx"]
    y1_px = m1_points[2]["yPx"]  # y=0.1
    y2_px = m1_points[3]["yPx"]  # y=1.0
    return x1_px, x2_px, y1_px, y2_px


def px_to_physical(
    xPx: float, yPx: float,
    x1_px: float, x2_px: float, y1_px: float, y2_px: float,
) -> tuple[float, float]:
    """Convert pixel coordinates to (latitude_deg, freq_mhz).

    x-axis: linear from 64° to 76°
    y-axis: log from 0.1 to 1.0 mHz
    """
    lat = 64.0 + (xPx - x1_px) / (x2_px - x1_px) * 12.0
    log_y = math.log10(0.1) + (yPx - y1_px) / (y2_px - y1_px) * (math.log10(1.0) - math.log10(0.1))
    freq = 10.0 ** log_y
    return lat, freq


def extract_panel(json_path: Path, panel_name: str) -> dict[int, list[tuple[float, float]]]:
    """Extract all modes from a JSON export file.

    Returns dict[mode_number] = [(lat, freq_mhz), ...] sorted by latitude.
    """
    with open(json_path) as f:
        data = json.load(f)

    # Calibration
    if panel_name == "midnight":
        x1_px, x2_px, y1_px, y2_px = extract_calibration_0lt(data)
        skip_first_n = {ds["name"]: 0 for ds in data["datasets"]}
    else:
        x1_px, x2_px, y1_px, y2_px = extract_calibration_12lt(data)
        # m1-p has 4 calibration points at the start
        skip_first_n = {"m1-p": 4, "m2-p": 0, "m3-p": 0, "m4-p": 0}

    modes: dict[int, list[tuple[float, float]]] = {}
    for ds in data["datasets"]:
        name = ds["name"]  # e.g. "m1-p"
        mode_num = int(name.split("-")[0][1:])  # "m1-p" -> 1
        skip = skip_first_n.get(name, 0)

        points = []
        for pt in ds["points"][skip:]:
            lat, freq = px_to_physical(pt["xPx"], pt["yPx"], x1_px, x2_px, y1_px, y2_px)
            points.append((lat, freq))
        points.sort(key=lambda p: p[0])
        modes[mode_num] = points

    return modes


def save_csv(noon: dict, midnight: dict, path: Path):
    """Save all digitized points to CSV."""
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["panel", "inv_lat_deg", "mode", "freq_mhz"])
        for panel_name, modes in [("noon", noon), ("midnight", midnight)]:
            for mode_num in sorted(modes):
                for lat, freq in modes[mode_num]:
                    writer.writerow([panel_name, f"{lat:.2f}", mode_num, f"{freq:.5f}"])
    print(f"Saved {path} ({sum(len(v) for v in noon.values()) + sum(len(v) for v in midnight.values())} points)")


def plot_panel(ax, title: str, modes: dict[int, list[tuple[float, float]]]):
    """Plot one panel of eigenfrequency curves."""
    for mode_num in sorted(modes):
        pts = modes[mode_num]
        lats = [p[0] for p in pts]
        freqs = [p[1] for p in pts]
        color = MODE_COLORS.get(mode_num, "#333333")
        ax.plot(lats, freqs, "-o", color=color, lw=2, ms=3,
                label=f"m={mode_num}")

    for period_min, label in [(30, "30 min"), (60, "60 min")]:
        f_mhz = 1000.0 / (period_min * 60)
        ax.axhline(f_mhz, color="white", ls="--", lw=0.8, alpha=0.6)
        ax.text(76.3, f_mhz, label, fontsize=8, va="center", color="0.7")

    ax.set_yscale("log")
    ax.set_ylim(0.005, 2.0)
    ax.set_xlim(63, 77)
    ax.set_xlabel("Invariant Latitude (degrees)")
    ax.set_ylabel("Eigenfrequency (mHz)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)

    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(ax.get_ylim())
    period_ticks = [5, 10, 30, 60, 120, 300]
    period_freqs = [1000.0 / (t * 60) for t in period_ticks]
    ax2.set_yticks(period_freqs)
    ax2.set_yticklabels([f"{t} min" if t < 120 else f"{t // 60} h"
                         for t in period_ticks])
    ax2.set_ylabel("Period")


def main():
    noon = extract_panel(PAPER_DIR / "data-export-12LT.json", "noon")
    midnight = extract_panel(PAPER_DIR / "data-export-0LT.json", "midnight")

    # Print summary
    for name, modes in [("12 LT (noon)", noon), ("0 LT (midnight)", midnight)]:
        print(f"\n=== {name} ===")
        for mode_num in sorted(modes):
            pts = modes[mode_num]
            print(f"  m={mode_num}: {len(pts)} points, "
                  f"lat [{pts[0][0]:.1f}° .. {pts[-1][0]:.1f}°], "
                  f"freq [{pts[-1][1]:.4f} .. {pts[0][1]:.4f}] mHz")

    # Save CSV
    csv_path = OUTPUT_DIR / "published_eigenfrequencies.csv"
    save_csv(noon, midnight, csv_path)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="black")
    for ax in (ax1, ax2):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

    plot_panel(ax1, "Field Lines at 12 LT (noon)", noon)
    plot_panel(ax2, "Field Lines at 0 LT (midnight)", midnight)

    for ax in fig.axes:
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        if ax.get_ylabel():
            ax.yaxis.label.set_color("white")

    fig.suptitle("Digitized from paper/figure6.jpeg", color="white",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = OUTPUT_DIR / "digitized_eigenfreqs.png"
    fig.savefig(out, dpi=200, facecolor="black", bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close()


if __name__ == "__main__":
    main()
