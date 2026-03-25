"""Figure 4: Power spectral density and power ratios (2007-01-02).

Panel (a): MFA field perturbations showing three wave trains.
Panel (b): Power density spectra with background estimates (dashed).
Panel (c): Power ratios r_i = P(b_i) / P(<B_T>_f), showing QP60 peak.

Referee notes: dark background, ephemeris ticks, visible vertical period lines.
"""

import sys
import types
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]

# Register stubs for legacy pickle deserialization
_stub_classes = ["SignalSnapshot", "NewSignal", "Interval", "FFT_list", "WaveSignal", "Wave"]
_stub_modules = ["__main__", "data_sweeper", "mag_fft_sweeper", "cassinilib", "cassinilib.NewSignal"]
for mod_path in _stub_modules:
    if mod_path not in sys.modules:
        sys.modules[mod_path] = types.ModuleType(mod_path)
    for cls_name in _stub_classes:
        setattr(sys.modules[mod_path], cls_name, type(cls_name, (), {}))

sys.path.insert(0, str(_project_root / "src"))

import qp
from qp.coords.mfa import to_mfa
from qp.signal.timeseries import running_average
from qp.signal.power_ratio import compute_power_ratios
from qp.plotting.style import (
    FIELD_COLORS, FIELD_COLORS_SPECTRA, use_paper_style, style_axes,
    draw_period_lines, plot_segmented, BG_COLOR,
)


TARGET_DATE = datetime.date(2007, 1, 2)
DETREND_WINDOW_MIN = 180  # 3-hour running average for MFA background


def load_segment(target_date):
    """Load the 36H KSM segment for a given date."""
    data = np.load(qp.DATA_PRODUCTS / "Cassini_MAG_KSM_36H.npy", allow_pickle=True)
    for seg in data:
        d = seg.datetime[0]
        if isinstance(d, datetime.datetime) and d.date() == target_date:
            return seg
    raise ValueError(f"No segment found for {target_date}")


def process_segment(seg, target_date):
    """Extract 24h of detrended MFA data from a segment."""
    times = seg.datetime
    fields = {f.name: f.y for f in seg.FIELDS}
    coords = {c.name: c.y for c in seg.COORDS}

    # Trim to 24h
    start_idx = next(i for i, t in enumerate(times) if t.date() == target_date)
    end_idx = min(start_idx + 24 * 60, len(times))
    sl = slice(start_idx, end_idx)

    times_24h = times[sl]
    fields_24h = {k: v[sl] for k, v in fields.items()}
    coords_24h = {k: v[sl] for k, v in coords.items()}

    # Detrend with running average
    dt = 60.0
    window = int(DETREND_WINDOW_MIN * 60 / dt)
    if window % 2 == 0:
        window += 1

    perturbations = {}
    bg_fields = {}
    for name, data in fields_24h.items():
        bg = running_average(data, window)
        perturbations[name] = data - bg
        bg_fields[name] = bg

    # MFA transform
    position = np.column_stack([coords_24h["x"], coords_24h["y"], coords_24h["z"]])
    field = np.column_stack([perturbations["Bx"], perturbations["By"], perturbations["Bz"]])
    background = np.column_stack([bg_fields["Bx"], bg_fields["By"], bg_fields["Bz"]])

    mfa = to_mfa(position, field, background, coords="KSM")
    b_par = mfa[:, 0]
    b_perp1 = mfa[:, 1]
    b_perp2 = mfa[:, 2]
    b_tot = np.sqrt(b_par**2 + b_perp1**2 + b_perp2**2)

    return times_24h, b_par, b_perp1, b_perp2, b_tot, coords_24h


def main():
    print("Loading data...")
    seg = load_segment(TARGET_DATE)
    times, b_par, b_perp1, b_perp2, b_tot, coords = process_segment(seg, TARGET_DATE)
    dt = 60.0  # 1-minute resolution
    N = len(b_par)
    print(f"Segment: {times[0]} to {times[-1]}, {N} points")

    # Welch PSD parameters (paper: 12h window, 6h overlap, Hann)
    nperseg = 12 * 60  # 12 hours in minutes
    noverlap = 6 * 60  # 6 hours overlap

    # Compute power ratios
    ratios = compute_power_ratios(
        b_par, b_perp1, b_perp2, b_tot,
        dt=dt, nperseg=nperseg, noverlap=noverlap, window="hann",
    )
    freq = ratios["freq"]

    # --- Plot ---
    use_paper_style()

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(12, 11),
        gridspec_kw={"height_ratios": [2, 2, 2]},
    )
    fig.subplots_adjust(hspace=0.15, left=0.10, right=0.95, top=0.93, bottom=0.06)

    # ===== Panel (a): MFA perturbations timeseries =====
    labels_mfa = [r"$B_{\parallel}$", r"$B_{\perp 1}$", r"$B_{\perp 2}$", r"$B_{total}$"]
    for comp, label, color in zip([b_par, b_perp1, b_perp2, b_tot], labels_mfa, FIELD_COLORS):
        ax_a.plot(times, comp, color=color, label=label, lw=1.5, alpha=0.8)

    ax_a.set_ylabel("Field Perturbations (nT)", fontsize=14)
    ax_a.axhline(0, ls="--", lw=1, color="grey", alpha=0.3)
    ax_a.legend(loc="upper left", frameon=False, fontsize=12, ncol=4)
    ax_a.set_title(f"Spectral Density of Wave Activity ({TARGET_DATE})", fontsize=18)
    ax_a.text(0.01, 0.95, "a", transform=ax_a.transAxes, fontsize=18, fontweight="bold", va="top")
    style_axes(ax_a)

    # Mark wave train intervals (from original: ~3-8 UT, ~13-18 UT, ~20-24 UT)
    activity = [
        (datetime.datetime(2007, 1, 2, 2, 0), datetime.datetime(2007, 1, 2, 8, 0)),
        (datetime.datetime(2007, 1, 2, 13, 0), datetime.datetime(2007, 1, 2, 18, 0)),
        (datetime.datetime(2007, 1, 2, 20, 0), datetime.datetime(2007, 1, 2, 23, 59)),
    ]
    for t0, t1 in activity:
        ax_a.axvspan(t0, t1, alpha=0.08, color="orange")
    # Separation annotations
    for (_, t1_end), (t2_start, _), label in [
        (activity[0], activity[1], "~ 11 hours"),
        (activity[1], activity[2], "~ 8 hours"),
    ]:
        mid = t1_end + (t2_start - t1_end) / 2
        ypos = ax_a.get_ylim()[0] * 0.7
        ax_a.annotate(
            "", xy=(mdates.date2num(t2_start), ypos),
            xytext=(mdates.date2num(t1_end), ypos),
            arrowprops=dict(arrowstyle="<->", color="orange", lw=1),
        )
        ax_a.text(mid, ypos * 0.8, label, fontsize=7, color="orange", ha="center")

    for lbl, pos in zip(
        ["Strong QP60\nActivity"] * 3,
        [activity[0][0] + (activity[0][1] - activity[0][0]) / 2,
         activity[1][0] + (activity[1][1] - activity[1][0]) / 2,
         activity[2][0] + (activity[2][1] - activity[2][0]) / 2],
    ):
        ax_a.text(pos, ax_a.get_ylim()[0] * 0.95, lbl, fontsize=6, color="orange",
                  ha="center", va="bottom",
                  bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="orange", alpha=0.7, lw=0.5))

    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax_a.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_a.set_xlim(times[0], times[-1])
    ax_a.set_xlabel("Local Time (h)", fontsize=10)

    # ===== Panel (b): Power density =====
    colors_spec = FIELD_COLORS_SPECTRA

    psd_keys = ["psd_par", "psd_perp1", "psd_perp2", "psd_total"]
    for key, label, color in zip(psd_keys, labels_mfa, colors_spec):
        plot_segmented(ax_b, freq, ratios[key], lw=2, marker="o", ms=2,
                       color=color, label=label)

    # Background estimate (dashed white)
    plot_segmented(ax_b, freq, ratios["background"], lw=2, ls="--",
                   color="white", base_alpha=0.3, label="Background")

    ax_b.set_yscale("log")
    ax_b.set_xscale("log")
    ax_b.set_ylabel(r"Power Density (nT$^2$/Hz)", fontsize=14)
    ax_b.legend(loc="upper right", frameon=False, fontsize=10, ncol=3)
    ax_b.text(0.01, 0.95, "b", transform=ax_b.transAxes, fontsize=18, fontweight="bold", va="top")
    ax_b.text(0.01, 0.82, "Power Density", transform=ax_b.transAxes, fontsize=12, color="white")
    style_axes(ax_b, minimal=False)

    draw_period_lines(ax_b)

    # ===== Panel (c): Power ratios =====
    ratio_keys = ["r_par", "r_perp1", "r_perp2", "r_total"]
    for key, label, color in zip(ratio_keys, labels_mfa, colors_spec):
        plot_segmented(ax_c, freq, ratios[key], lw=2, marker="o", ms=2,
                       color=color, label=label)

    ax_c.axhline(1.0, ls="--", lw=2, color="orange", alpha=0.5)
    ax_c.set_yscale("log")
    ax_c.set_xscale("log")
    ax_c.set_ylabel(r"Power Ratio $r_i$", fontsize=14)
    ax_c.set_xlabel("Frequency (mHz)", fontsize=14)
    ax_c.legend(loc="upper right", frameon=False, fontsize=10, ncol=2)
    ax_c.text(0.01, 0.95, "c", transform=ax_c.transAxes, fontsize=18, fontweight="bold", va="top")
    ax_c.text(0.01, 0.82, r"Power Ratio to $B_T$ Background Power",
              transform=ax_c.transAxes, fontsize=12, color="white")
    style_axes(ax_c, minimal=False)

    draw_period_lines(ax_c)

    # Frequency axis formatting for panels b and c
    for ax in [ax_b, ax_c]:
        ax.set_xlim(freq[1] / 1e-3, freq[-1] / 1e-3)
        ax.tick_params(axis="both", labelsize=12)

    plt.savefig("output/figure4.png", dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved output/figure4.png")
    plt.close()


if __name__ == "__main__":
    main()
