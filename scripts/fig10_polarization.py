"""Figure 10: Cross-correlation and polarization analysis.

Two examples of running cross-correlation between b_perp1 and b_perp2:
  Example 1: mostly 90° phase shift (circular polarization, most common)
  Example 2: mostly 180° phase shift (linear polarization)

Referee: full 360° range, no phase-jump artifacts.
"""

import sys
import types

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal as sig

from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]

# Register stubs
for mod_path in ["__main__", "data_sweeper", "mag_fft_sweeper",
                 "cassinilib", "cassinilib.NewSignal", "cassinilib.PlotFFT"]:
    if mod_path not in sys.modules:
        sys.modules[mod_path] = types.ModuleType(mod_path)
    for cls_name in ["SignalSnapshot", "NewSignal", "Interval", "FFT_list",
                     "WaveSignal", "Wave"]:
        setattr(sys.modules[mod_path], cls_name, type(cls_name, (), {}))

sys.path.insert(0, str(_project_root / "src"))

import qp
from qp.plotting.style import use_paper_style, style_axes, BG_COLOR


def running_phase(b_perp1, b_perp2, dt=60.0, half_window=61):
    """Compute running cross-correlation phase between two transverse components.

    Matches the algorithm from mag_fft_sweeper.py lines 802-832.

    Parameters
    ----------
    b_perp1, b_perp2 : array
        Perpendicular field components.
    dt : float
        Sampling interval in seconds.
    half_window : int
        Half-width of sliding window in samples (61 = ~1 hour at 1-min).

    Returns
    -------
    phase_deg : array
        Phase shift in degrees at each time step.
    valid_mask : array of bool
        True where the phase estimate is valid (away from edges).
    """
    N = len(b_perp1)
    phase_deg = np.full(N, np.nan)
    sr = 1.0 / dt

    for ind in range(half_window, N - half_window):
        i0 = ind - half_window
        i1 = ind + half_window
        y1 = b_perp1[i0:i1]
        y2 = b_perp2[i0:i1]
        npts = len(y1)

        ccor = sig.correlate(y2, y1, mode="same", method="direct")
        delay_arr = np.linspace(-npts / sr, npts / sr, npts)
        delay_arr = delay_arr / 3600.0 * 180.0  # Convert seconds to degrees (1h = 180°)

        # Mask lags outside ±180°
        mask_lo = delay_arr < -180
        mask_hi = delay_arr > 180
        ccor[mask_lo] = 0
        ccor[mask_hi] = 0

        phase_deg[ind] = delay_arr[np.argmax(ccor)]

    valid = ~np.isnan(phase_deg)
    return phase_deg, valid


def classify_segment(phase_deg, valid):
    """Classify a segment's dominant polarization."""
    ph = phase_deg[valid]
    if len(ph) < 100:
        return "unknown", 0.0

    # Circular: phase near ±90°
    near_90 = np.sum(np.abs(np.abs(ph) - 90) < 30)
    # Linear (anti-phase): phase near ±180° (wrapping at the boundary)
    near_180 = np.sum(np.abs(ph) > 150)
    frac_90 = near_90 / len(ph)
    frac_180 = near_180 / len(ph)

    if frac_90 > frac_180 and frac_90 > 0.3:
        return "circular", frac_90
    elif frac_180 > frac_90 and frac_180 > 0.3:
        return "linear", frac_180
    else:
        return "mixed", max(frac_90, frac_180)


def find_examples(data, max_scan=2000):
    """Scan MFA segments to find good circular and linear polarization examples."""
    best_circular = None
    best_circular_score = 0
    best_linear = None
    best_linear_score = 0

    n_scanned = 0
    for idx, seg in enumerate(data):
        if n_scanned >= max_scan:
            break

        # Filter
        if seg.flag is not None:
            continue
        if not isinstance(seg.info, dict) or seg.info.get("location") != 0:
            continue
        if not hasattr(seg, "FIELDS") or len(seg.FIELDS) < 3:
            continue

        b_perp1 = seg.FIELDS[1].y
        b_perp2 = seg.FIELDS[2].y

        if b_perp1 is None or len(b_perp1) < 1440:
            continue

        # Check signal strength — skip weak segments
        amp = max(np.std(b_perp1), np.std(b_perp2))
        if amp < 0.03:
            continue

        # Trim to central 24h
        pad = 6 * 60
        bp1 = b_perp1[pad:-pad] if len(b_perp1) > 2 * pad else b_perp1
        bp2 = b_perp2[pad:-pad] if len(b_perp2) > 2 * pad else b_perp2

        phase, valid = running_phase(bp1, bp2)
        pol_type, score = classify_segment(phase, valid)

        n_scanned += 1
        if n_scanned % 200 == 0:
            print(f"  Scanned {n_scanned} segments...")

        if pol_type == "circular" and score > best_circular_score:
            best_circular_score = score
            best_circular = idx
        elif pol_type == "linear" and score > best_linear_score:
            best_linear_score = score
            best_linear = idx

    print(f"  Scanned {n_scanned} segments total")
    print(f"  Best circular: idx={best_circular}, score={best_circular_score:.2f}")
    print(f"  Best linear: idx={best_linear}, score={best_linear_score:.2f}")
    return best_circular, best_linear


def plot_example(axes_ts, axes_ph, seg, title, panel_labels):
    """Plot one example: timeseries + running phase."""
    b_perp1 = seg.FIELDS[1].y
    b_perp2 = seg.FIELDS[2].y
    times = seg.datetime

    # Trim to central 24h
    pad = 6 * 60
    if len(b_perp1) > 2 * pad:
        b_perp1 = b_perp1[pad:-pad]
        b_perp2 = b_perp2[pad:-pad]
        times = times[pad:-pad]

    # Time in hours from start
    t0 = times[0]
    hours = np.array([(t - t0).total_seconds() / 3600 for t in times])

    # Timeseries panel
    ax = axes_ts
    ax.plot(hours, b_perp1, color="#FFB000", label=r"$B_{\perp 1}$", lw=1.2, alpha=0.8)
    ax.plot(hours, b_perp2, color="#fdf33c", label=r"$B_{\perp 2}$", lw=1.2, alpha=0.8)
    ax.set_ylabel("(nT)", fontsize=12)
    ax.axhline(0, ls="--", lw=0.5, color="grey", alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=11, ncol=2)
    ax.set_title(title, fontsize=14, loc="left", color="white")
    ax.text(0.01, 0.92, panel_labels[0], transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")
    ax.set_xlim(0, 24)
    ax.set_xticklabels([])
    style_axes(ax)

    # Phase panel
    ax = axes_ph
    phase, valid = running_phase(b_perp1, b_perp2)
    ax.plot(hours[valid], phase[valid], color="#FFB000", lw=1.2, alpha=0.8)

    # Reference lines
    for deg in [-180, -90, 0, 90, 180]:
        ls = "--" if deg in [-90, 90] else "-" if deg == 0 else ":"
        alpha = 0.5 if deg in [-90, 90] else 0.3
        ax.axhline(deg, ls=ls, lw=0.8, color="grey", alpha=alpha)

    ax.set_ylabel("Phase (deg)", fontsize=12)
    ax.set_xlabel("Time (h)", fontsize=12)
    ax.set_ylim(-200, 200)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlim(0, 24)
    ax.text(0.01, 0.92, panel_labels[1], transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")
    style_axes(ax)


def main():
    print("Loading MFA 36H data...")
    data = np.load(qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy", allow_pickle=True)
    print(f"Loaded {len(data)} segments")

    print("Scanning for circular and linear polarization examples...")
    idx_circ, idx_lin = find_examples(data)

    if idx_circ is None or idx_lin is None:
        print("Could not find both examples!")
        return

    seg_circ = data[idx_circ]
    seg_lin = data[idx_lin]
    date_circ = seg_circ.datetime[0].strftime("%Y-%m-%d")
    date_lin = seg_lin.datetime[0].strftime("%Y-%m-%d")
    print(f"Circular example: idx={idx_circ}, date={date_circ}")
    print(f"Linear example: idx={idx_lin}, date={date_lin}")

    # --- Plot ---
    use_paper_style()

    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(
        4, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [2, 1.5, 2, 1.5]},
    )
    fig.subplots_adjust(hspace=0.12, left=0.08, right=0.95, top=0.95, bottom=0.06)

    plot_example(
        ax_a, ax_b, seg_circ,
        f"EXAMPLE 1 (Mostly 90° polarization) — {date_circ}",
        panel_labels=("a", "b"),
    )
    plot_example(
        ax_c, ax_d, seg_lin,
        f"EXAMPLE 2 (Mostly 180° polarization) — {date_lin}",
        panel_labels=("c", "d"),
    )

    plt.savefig("output/figure10.png", dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved output/figure10.png")
    plt.close()


if __name__ == "__main__":
    main()
