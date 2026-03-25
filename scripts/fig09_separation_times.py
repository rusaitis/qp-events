"""Figure 9: Wave train separation time distribution.

Probability distribution of time between consecutive QP60 wave packets.
Median separation = ~10.73 h, matching the PPO period.

Uses CWT-based event detection on all MFA segments to find wave packet
peaks, then computes separations between consecutive peaks.
"""

import sys
import types
import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]

# Register stubs
_stub_modules = ["__main__", "data_sweeper", "mag_fft_sweeper",
                 "cassinilib", "cassinilib.NewSignal", "cassinilib.PlotFFT"]
for mod_path in _stub_modules:
    if mod_path not in sys.modules:
        sys.modules[mod_path] = types.ModuleType(mod_path)
    for cls_name in ["SignalSnapshot", "NewSignal", "Interval", "FFT_list",
                     "WaveSignal", "Wave"]:
        setattr(sys.modules[mod_path], cls_name, type(cls_name, (), {}))

sys.path.insert(0, str(_project_root / "src"))

import qp
from qp.events.detector import detect_wave_packets
from qp.events.wave_packets import compute_separations, separation_statistics, separation_histogram
from qp.plotting.style import use_paper_style, BG_COLOR


# QP60 band: 50-70 min (60±10 min as stated in paper)
PERIOD_BAND = (50 * 60, 70 * 60)  # seconds


def detect_all_wave_packets(data):
    """Run CWT-based wave packet detection on all valid MFA segments.

    Uses component 1 (b_perp1) which has the strongest QP60 signal.
    """
    all_packets = []
    n_total = len(data)
    n_valid = 0
    prev_peak_time = None

    for idx, seg in enumerate(data):
        if idx % 500 == 0:
            print(f"  Processing {idx}/{n_total}...")

        # Filter
        if seg.flag is not None:
            continue
        if not isinstance(seg.info, dict) or seg.info.get("location") != 0:
            continue
        if not hasattr(seg, "FIELDS") or len(seg.FIELDS) < 3:
            continue

        # Use b_perp1 (component 1) — strongest transverse signal
        b_perp1 = seg.FIELDS[1].y
        times = seg.datetime

        if b_perp1 is None or len(b_perp1) < 720:
            continue
        if times is None or len(times) == 0:
            continue

        # Trim to central 24h
        pad = 6 * 60
        if len(b_perp1) > 2 * pad:
            b_perp1 = b_perp1[pad:-pad]
            times = times[pad:-pad]

        try:
            packets = detect_wave_packets(
                b_perp1, times, dt=60.0,
                period_band=PERIOD_BAND,
                n_period_bins=5,
                min_prominence=0.03,
                min_peak_distance=60,
                min_peak_width=60,
                min_duration_hours=2.0,
                dedup_window_hours=3.0,
                previous_peak_time=prev_peak_time,
            )
        except Exception:
            continue

        if packets:
            all_packets.extend(packets)
            prev_peak_time = packets[-1].peak_time
            n_valid += 1

    print(f"  Processed: {n_valid} segments with detections, {len(all_packets)} total packets")
    return all_packets


def main():
    print("Loading MFA 36H data...")
    data = np.load(qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy", allow_pickle=True)
    print(f"Loaded {len(data)} segments")

    print("Detecting QP60 wave packets...")
    packets = detect_all_wave_packets(data)

    if len(packets) < 2:
        print("Not enough wave packets detected!")
        return

    # Sort by peak time
    packets.sort(key=lambda p: p.peak_time)

    # Compute separations
    seps = compute_separations(packets, max_separation_hours=36.0)
    stats = separation_statistics(seps)

    print(f"\nSeparation statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Median: {stats['median']:.2f} h")
    print(f"  Mean: {stats['mean']:.2f} h")
    print(f"  Std: {stats['std']:.2f} h")

    # Histogram
    bin_width = 1.5  # hours
    centers, counts, pdf = separation_histogram(seps, bin_width_hours=bin_width, max_hours=36.0)

    # KDE for smooth curve
    kde = gaussian_kde(seps, bw_method=0.2)
    x_smooth = np.linspace(0, 36, 500)
    y_smooth = kde(x_smooth)

    # --- Plot ---
    # Figure 9 uses LIGHT background (unlike other figures)
    plt.style.use("default")
    plt.rcParams.update({"font.size": 16})

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram bars
    ax.bar(centers, pdf, width=bin_width * 0.9, color="#f0b87a", alpha=0.7,
           edgecolor="#d4944a", linewidth=0.5)

    # Smooth KDE curve
    ax.plot(x_smooth, y_smooth, color="#555555", lw=2.5)

    # Median line
    ax.axvline(stats["median"], ls="--", lw=2, color="grey", alpha=0.7)
    ax.text(stats["median"] + 0.3, ax.get_ylim()[1] * 0.3,
            f"median sep = {stats['median']:.2f} h",
            fontsize=14, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel("Separation [h]", fontsize=18)
    ax.set_ylabel("Probability density", fontsize=18)
    ax.set_title(r"Wave Activity Separation in Time ($\tau$)", fontsize=20)
    ax.set_xlim(0, 27)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=14)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig("output/figure9.png", dpi=300, bbox_inches="tight")
    print("Saved output/figure9.png")
    plt.close()


if __name__ == "__main__":
    main()
