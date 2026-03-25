"""Figure 1: QP60 wave activity example (2008-02-29).

Panel (a): Magnetic field perturbations in KSM coordinates.
Panel (b): Same perturbations in field-aligned (MFA) coordinates.
Two intervals of strong QP60 activity marked, separated by ~11 hours.
Spacecraft ephemeris (KSM) at the bottom.

Referee notes: dark background, ephemeris in caption.
"""

import sys
import pickle
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
import types

_project_root = Path(__file__).resolve().parents[1]

# --- Register stub classes BEFORE any project imports ---
# The pickled .npy files reference classes from data_sweeper, cassinilib.NewSignal,
# etc. We register lightweight stubs so numpy.load can deserialize them without
# importing the full (heavy, fragile) old codebase.
_stub_classes = ["SignalSnapshot", "NewSignal", "Interval", "FFT_list", "WaveSignal", "Wave"]
_stub_modules = ["__main__", "data_sweeper", "mag_fft_sweeper", "cassinilib", "cassinilib.NewSignal"]

for mod_path in _stub_modules:
    if mod_path not in sys.modules:
        sys.modules[mod_path] = types.ModuleType(mod_path)
    for cls_name in _stub_classes:
        setattr(sys.modules[mod_path], cls_name, type(cls_name, (), {}))

# Now safe to import qp
sys.path.insert(0, str(_project_root / "src"))

import qp
from qp.coords.mfa import to_mfa
from qp.signal.timeseries import running_average
from qp.plotting.style import (
    FIELD_COLORS, use_paper_style, style_axes, BG_COLOR,
)


def load_legacy_segments(name: str) -> np.ndarray:
    """Load a legacy .npy file containing pickled SignalSnapshot objects."""
    return np.load(qp.DATA_PRODUCTS / name, allow_pickle=True)


def find_segment_by_date(data, target_date: datetime.date):
    """Find the segment starting on a given date."""
    for seg in data:
        d = seg.datetime[0]
        if isinstance(d, datetime.datetime) and d.date() == target_date:
            return seg
    raise ValueError(f"No segment found for {target_date}")


def extract_arrays(seg):
    """Extract field and coordinate arrays from a SignalSnapshot."""
    times = seg.datetime
    fields = {f.name: f.y for f in seg.FIELDS}
    coords = {c.name: c.y for c in seg.COORDS}
    return times, fields, coords


def detrend_fields(fields: dict, window_minutes: int = 180, dt: float = 60.0):
    """Remove running average (3-hour default) to get perturbations."""
    window = int(window_minutes * 60 / dt)
    if window % 2 == 0:
        window += 1
    perturbations = {}
    for name, data in fields.items():
        trend = running_average(data, window)
        perturbations[name] = data - trend
    return perturbations


def compute_mfa(perturbations, coords, fields, window_minutes=180, dt=60.0):
    """Transform KSM perturbations to MFA coordinates."""
    window = int(window_minutes * 60 / dt)
    if window % 2 == 0:
        window += 1

    # Background field from running average
    bg_bx = running_average(fields["Bx"], window)
    bg_by = running_average(fields["By"], window)
    bg_bz = running_average(fields["Bz"], window)

    N = len(perturbations["Bx"])
    position = np.column_stack([coords["x"], coords["y"], coords["z"]])
    field = np.column_stack([perturbations["Bx"], perturbations["By"], perturbations["Bz"]])
    background = np.column_stack([bg_bx, bg_by, bg_bz])

    mfa = to_mfa(position, field, background, coords="KSM")
    b_par, b_perp1, b_perp2 = mfa[:, 0], mfa[:, 1], mfa[:, 2]
    b_tot = np.sqrt(b_par**2 + b_perp1**2 + b_perp2**2)
    return b_par, b_perp1, b_perp2, b_tot


# --- Main ---

def main():
    print("Loading KSM 36H data...")
    data = load_legacy_segments("Cassini_MAG_KSM_36H.npy")
    print(f"Loaded {len(data)} segments")

    # Find 2008-02-29
    target = datetime.date(2008, 2, 29)
    seg = find_segment_by_date(data, target)
    times, fields, coords = extract_arrays(seg)
    print(f"Segment: {times[0]} to {times[-1]}, {len(times)} points")

    # Trim to 24h starting at the target date
    start_idx = 0
    for i, t in enumerate(times):
        if t.date() == target:
            start_idx = i
            break
    end_idx = min(start_idx + 24 * 60, len(times))
    sl = slice(start_idx, end_idx)

    times_24h = times[sl]
    fields_24h = {k: v[sl] for k, v in fields.items()}
    coords_24h = {k: v[sl] for k, v in coords.items()}

    # Detrend to get perturbations
    perturbations = detrend_fields(fields_24h)

    # MFA transform
    b_par, b_perp1, b_perp2, b_tot = compute_mfa(
        perturbations, coords_24h, fields_24h
    )

    # --- Plot ---
    use_paper_style()

    fig, (ax_a, ax_b, ax_eph) = plt.subplots(
        3, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [3, 3, 1]},
    )
    fig.subplots_adjust(hspace=0.08, left=0.08, right=0.95, top=0.93, bottom=0.05)

    # Panel (a): KSM perturbations
    ksm_components = [perturbations["Bx"], perturbations["By"],
                      perturbations["Bz"], perturbations["Btot"]]
    ksm_labels = [r"$B_x$", r"$B_y$", r"$B_z$", r"$B_{total}$"]

    for comp, label, color in zip(ksm_components, ksm_labels, FIELD_COLORS):
        ax_a.plot(times_24h, comp, color=color, label=label, lw=1.5, alpha=0.8)

    ax_a.set_ylabel("Amplitude (nT)", fontsize=14)
    ax_a.axhline(0, ls="--", lw=1, color="grey", alpha=0.3)
    ax_a.legend(loc="upper left", frameon=False, fontsize=12, ncol=4)
    ax_a.set_title(f"QP60 Wave Activity ({target})", fontsize=18)
    ax_a.text(0.01, 0.95, "a", transform=ax_a.transAxes, fontsize=18,
              fontweight="bold", va="top")
    style_axes(ax_a)

    # Panel (b): MFA perturbations
    mfa_components = [b_par, b_perp1, b_perp2, b_tot]
    mfa_labels = [r"$B_{\parallel}$", r"$B_{\perp 1}$", r"$B_{\perp 2}$", r"$B_{total}$"]

    for comp, label, color in zip(mfa_components, mfa_labels, FIELD_COLORS):
        ax_b.plot(times_24h, comp, color=color, label=label, lw=1.5, alpha=0.8)

    ax_b.set_ylabel("Amplitude (nT)", fontsize=14)
    ax_b.axhline(0, ls="--", lw=1, color="grey", alpha=0.3)
    ax_b.legend(loc="upper left", frameon=False, fontsize=12, ncol=4)
    ax_b.text(0.01, 0.95, "b", transform=ax_b.transAxes, fontsize=18,
              fontweight="bold", va="top")
    style_axes(ax_b)

    # Mark Strong QP60 Activity intervals
    activity_intervals = [
        (datetime.datetime(2008, 2, 29, 4, 0), datetime.datetime(2008, 2, 29, 12, 0)),
        (datetime.datetime(2008, 2, 29, 16, 0), datetime.datetime(2008, 2, 29, 22, 0)),
    ]

    # Match y-ranges first so labels position correctly
    ymax = max(
        max(abs(np.min(perturbations[k])), abs(np.max(perturbations[k])))
        for k in ["Bx", "By", "Bz"]
    )
    ymax = max(ymax, max(abs(b_perp1.min()), abs(b_perp1.max()),
                          abs(b_perp2.min()), abs(b_perp2.max()))) * 1.15
    ax_a.set_ylim(-ymax, ymax)
    ax_b.set_ylim(-ymax, ymax)

    for ax in [ax_a, ax_b]:
        for t0, t1 in activity_intervals:
            ax.axvspan(t0, t1, alpha=0.06, color="orange")

        # Activity labels at bottom of each panel
        for t_center in [datetime.datetime(2008, 2, 29, 8, 0),
                         datetime.datetime(2008, 2, 29, 19, 0)]:
            ax.text(
                t_center, -ymax * 0.78,
                "Strong QP60\nActivity",
                fontsize=10, color="orange", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc=BG_COLOR, ec="orange",
                          alpha=0.8, lw=0.8),
            )

    # ~11 hour separation annotation on panel (a)
    ax_a.annotate(
        "", xy=(mdates.date2num(activity_intervals[1][0]), 0),
        xytext=(mdates.date2num(activity_intervals[0][1]), 0),
        arrowprops=dict(arrowstyle="<->", color="orange", lw=2),
    )
    mid_time = activity_intervals[0][1] + (activity_intervals[1][0] - activity_intervals[0][1]) / 2
    ax_a.text(mid_time, 0.02, "~ 11 hours", fontsize=12, color="orange",
              ha="center", va="bottom")

    # Time axis formatting
    for ax in [ax_a, ax_b]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.set_xlim(times_24h[0], times_24h[-1])
        ax.tick_params(axis="both", labelsize=12)
    ax_a.set_xticklabels([])
    ax_b.set_xlabel("Local Time (h)", fontsize=14)

    # Panel (ephemeris): Spacecraft position in KSM
    ax_eph.set_facecolor(BG_COLOR)
    ax_eph.set_yticks([0, 1, 2])
    ax_eph.set_yticklabels(["KSM Z", "KSM Y", "KSM X"], fontsize=10)
    ax_eph.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax_eph.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_eph.set_xlim(times_24h[0], times_24h[-1])

    coord_order = ["z", "y", "x"]  # match original: Z on top, X on bottom
    t_start = times_24h[0]
    delta = datetime.timedelta(hours=2)
    t = t_start
    while t < times_24h[-1]:
        idx = min(int((t - t_start).total_seconds() / 60), len(times_24h) - 1)
        for row, name in enumerate(coord_order):
            val = coords_24h[name][idx]
            ax_eph.text(t, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="grey")
        t += delta

    ax_eph.set_ylim(-0.5, 2.5)
    for spine in ax_eph.spines.values():
        spine.set_visible(False)
    ax_eph.tick_params(which="both", width=0, labelsize=10)
    ax_eph.grid(False)

    plt.savefig("output/figure1.png", dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved output/figure1.png")
    plt.close()


if __name__ == "__main__":
    main()
