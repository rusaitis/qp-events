"""Figure 9: Wave train separation time distribution.

Probability distribution of time between consecutive QP60 wave packets.
Median separation = ~10.73 h, matching the PPO period.

Reads the event catalog (v5) produced by sweep_events.py instead of
re-detecting events — ensures consistency with the simplified pipeline.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "src"))


def load_qp60_peaks(catalog_path: Path):
    """Load QP60 events from parquet, return sorted peak times in hours."""
    import pandas as pd
    df = pd.read_parquet(catalog_path)
    qp60 = df[df["band"] == "QP60"].copy()
    # Restrict to magnetosphere
    qp60 = qp60[qp60["region"] == "magnetosphere"]
    # Parse peak times (midpoint of date_from/date_to)
    from_dt = pd.to_datetime(qp60["date_from"])
    to_dt = pd.to_datetime(qp60["date_to"])
    peak_dt = from_dt + (to_dt - from_dt) / 2
    peak_dt = peak_dt.sort_values()
    # Compute separations in hours
    seps_sec = peak_dt.diff().dt.total_seconds().dropna().values
    seps_h = seps_sec / 3600.0
    # Filter to reasonable range (skip multi-day gaps)
    seps_h = seps_h[(seps_h > 0) & (seps_h < 36)]
    return seps_h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path,
                        default=_project_root / "Output" / "events_qp_v5.parquet")
    parser.add_argument("--output", type=Path,
                        default=_project_root / "Output" / "figures" / "figure9_separation_times.png")
    args = parser.parse_args()

    print(f"Loading catalog: {args.catalog}")
    seps = load_qp60_peaks(args.catalog)
    print(f"  QP60 separations: {len(seps)}")

    if len(seps) < 10:
        print("Not enough separations for a meaningful histogram!")
        return

    stats = {
        "count": len(seps),
        "median": float(np.median(seps)),
        "mean": float(np.mean(seps)),
        "std": float(np.std(seps)),
    }
    print(f"\nSeparation statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Median: {stats['median']:.2f} h")
    print(f"  Mean: {stats['mean']:.2f} h")
    print(f"  Std: {stats['std']:.2f} h")

    # Histogram
    bin_width = 1.5  # hours
    bins = np.arange(0, 36 + bin_width, bin_width)
    counts, edges = np.histogram(seps, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    pdf = counts / (counts.sum() * bin_width)

    # KDE for smooth curve
    kde = gaussian_kde(seps, bw_method=0.2)
    x_smooth = np.linspace(0, 36, 500)
    y_smooth = kde(x_smooth)

    # --- Plot ---
    plt.style.use("default")
    plt.rcParams.update({"font.size": 16})

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(centers, pdf, width=bin_width * 0.9, color="#f0b87a", alpha=0.7,
           edgecolor="#d4944a", linewidth=0.5)
    ax.plot(x_smooth, y_smooth, color="#555555", lw=2.5)

    # Median line
    ax.axvline(stats["median"], ls="--", lw=2, color="grey", alpha=0.7)
    ax.text(stats["median"] + 0.3, ax.get_ylim()[1] * 0.3,
            f"median = {stats['median']:.2f} h",
            fontsize=14, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel("Separation [h]", fontsize=18)
    ax.set_ylabel("Probability density", fontsize=18)
    ax.set_title(r"Wave Activity Separation in Time ($\tau$)", fontsize=20)
    ax.set_xlim(0, 27)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
