"""Figure 10: Cross-correlation and polarization analysis.

Two examples of running cross-correlation between b_perp1 and b_perp2:
  Example 1: mostly 90° phase shift (circular polarization, most common)
  Example 2: mostly 180° phase shift (linear polarization)

Examples are picked from the round-8 catalogue (``Output/events_round8.parquet``)
by Stokes ellipticity rather than by the cross-correlation classifier — the
Stokes estimator is the round-8 detector's ground truth and is period-aware
by construction. The phase-axis conversion ``deg/sec = 360 / period`` is
threaded through ``running_phase`` so the ±90° / ±180° reference lines
carry their literal physical meaning for any band (QP30 / QP60 / QP120).

Referee: full 360° range, no phase-jump artifacts.
"""

import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sig

_project_root = Path(__file__).resolve().parents[1]

# Register stubs.
# DO NOT REMOVE: stub names match the module paths used when the legacy
# DataProducts/*.npy arrays were pickled. Removing them silently breaks np.load().
for mod_path in [
    "__main__",
    "data_sweeper",
    "mag_fft_sweeper",
    "cassinilib",
    "cassinilib.NewSignal",
    "cassinilib.PlotFFT",
]:
    if mod_path not in sys.modules:
        sys.modules[mod_path] = types.ModuleType(mod_path)
    for cls_name in [
        "SignalSnapshot",
        "NewSignal",
        "Interval",
        "FFT_list",
        "WaveSignal",
        "Wave",
    ]:
        setattr(sys.modules[mod_path], cls_name, type(cls_name, (), {}))


import qp
from qp.plotting.style import style_axes, use_paper_style


def running_phase(b_perp1, b_perp2, dt=60.0, period_sec=3600.0):
    """Sliding cross-correlation phase between two transverse components.

    A ±1-period window slides over the signal; at each sample the
    normalized cross-correlation is computed and the lag of its peak is
    reported as a phase shift in degrees, restricted to ±180° (one
    half-period — beyond that the wrap-around lobes are indistinguishable
    from the central peak).

    Parameters
    ----------
    b_perp1, b_perp2 : array
        Transverse field components (same length).
    dt : float
        Sampling interval in seconds.
    period_sec : float
        Wave period in seconds — sets both the deg/sec conversion
        (``360° per period``) and the sliding-window half-width
        (one period each side).

    Returns
    -------
    phase_deg : array
        Phase shift in degrees at each time step (NaN at edges).
    valid : array of bool
        True where ``phase_deg`` is finite.
    """
    N = len(b_perp1)
    half_window = int(round(period_sec / dt))
    phase_deg = np.full(N, np.nan)
    deg_per_sec = 360.0 / period_sec

    for ind in range(half_window, N - half_window):
        y1 = b_perp1[ind - half_window : ind + half_window]
        y2 = b_perp2[ind - half_window : ind + half_window]
        npts = len(y1)
        ccor = sig.correlate(y2, y1, mode="same", method="direct")
        lags_samples = sig.correlation_lags(npts, npts, mode="same")
        deg = lags_samples * dt * deg_per_sec
        # Restrict to ±180° (one half-period); beyond that the cross-correlation
        # has wrap-around peaks that are indistinguishable from the central one.
        in_range = np.abs(deg) <= 180.0
        if not np.any(in_range):
            continue
        idx = np.where(in_range)[0]
        phase_deg[ind] = deg[idx[np.argmax(ccor[idx])]]

    valid = ~np.isnan(phase_deg)
    return phase_deg, valid


def pick_examples_from_catalogue(
    catalogue_path: Path,
    prefer_band: str = "QP60",
    circ_abs_ell: float = 0.8,
    lin_abs_ell: float = 0.15,
):
    """Pick best circular and linear examples from the round-8 catalogue.

    Selection by Stokes ellipticity (the detector's ground truth) rather
    than by the cross-correlation classifier. Among candidates passing
    the ellipticity cut, pick the one with the highest ``stokes_d``
    (most coherent), preferring ``prefer_band`` if available.

    Returns
    -------
    circ_row, lin_row : pandas.Series
        Catalogue rows for the chosen examples. ``segment_idx`` keys back
        into ``Cassini_MAG_MFA_36H.npy``; ``period_min`` sets the
        phase-axis scale for ``running_phase``.
    """
    df = pd.read_parquet(catalogue_path)

    def best_from(pool: pd.DataFrame) -> pd.Series:
        pref = pool[pool["band"] == prefer_band]
        if len(pref) > 0:
            return pref.nlargest(1, "stokes_d").iloc[0]
        return pool.nlargest(1, "stokes_d").iloc[0]

    circ_pool = df[df["ellipticity"].abs() > circ_abs_ell]
    lin_pool = df[df["ellipticity"].abs() < lin_abs_ell]
    if circ_pool.empty or lin_pool.empty:
        raise RuntimeError(
            f"Catalogue does not contain both circular (|e|>{circ_abs_ell}) "
            f"and linear (|e|<{lin_abs_ell}) examples"
        )
    return best_from(circ_pool), best_from(lin_pool)


def plot_example(axes_ts, axes_ph, seg, period_sec, title, panel_labels):
    """Plot one example: timeseries + period-aware running phase."""
    b_perp1 = seg.FIELDS[1].y
    b_perp2 = seg.FIELDS[2].y
    times = seg.datetime

    # Trim to central 24 h of the 36-h segment (drop 6 h of context each side).
    pad = 6 * 60
    if len(b_perp1) > 2 * pad:
        b_perp1 = b_perp1[pad:-pad]
        b_perp2 = b_perp2[pad:-pad]
        times = times[pad:-pad]

    t0 = times[0]
    hours = np.array([(t - t0).total_seconds() / 3600 for t in times])

    ax = axes_ts
    ax.plot(hours, b_perp1, color="#FFB000", label=r"$B_{\perp 1}$", lw=1.2, alpha=0.8)
    ax.plot(hours, b_perp2, color="#fdf33c", label=r"$B_{\perp 2}$", lw=1.2, alpha=0.8)
    ax.set_ylabel("(nT)", fontsize=12)
    ax.axhline(0, ls="--", lw=0.5, color="grey", alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=11, ncol=2)
    ax.set_title(title, fontsize=14, loc="left", color="white")
    ax.text(
        0.01,
        0.92,
        panel_labels[0],
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    ax.set_xlim(0, 24)
    ax.set_xticklabels([])
    style_axes(ax)

    ax = axes_ph
    phase, valid = running_phase(b_perp1, b_perp2, period_sec=period_sec)
    ax.plot(hours[valid], phase[valid], color="#FFB000", lw=1.2, alpha=0.8)

    for deg in [-180, -90, 0, 90, 180]:
        ls = "--" if deg in [-90, 90] else "-" if deg == 0 else ":"
        alpha = 0.5 if deg in [-90, 90] else 0.3
        ax.axhline(deg, ls=ls, lw=0.8, color="grey", alpha=alpha)

    ax.set_ylabel("Phase (deg)", fontsize=12)
    ax.set_xlabel("Time (h)", fontsize=12)
    ax.set_ylim(-200, 200)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlim(0, 24)
    ax.text(
        0.01,
        0.92,
        panel_labels[1],
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    style_axes(ax)


def main():
    catalogue_path = qp.OUTPUT_DIR / "events_round8.parquet"
    print(f"Picking examples from {catalogue_path}")
    circ_row, lin_row = pick_examples_from_catalogue(catalogue_path)

    print(
        f"  circular: seg {int(circ_row['segment_idx'])} {circ_row['date_from']} "
        f"{circ_row['band']}  T={circ_row['period_min']:.1f} min  "
        f"e={circ_row['ellipticity']:+.3f}  d={circ_row['stokes_d']:.3f}"
    )
    print(
        f"  linear:   seg {int(lin_row['segment_idx'])} {lin_row['date_from']} "
        f"{lin_row['band']}  T={lin_row['period_min']:.1f} min  "
        f"e={lin_row['ellipticity']:+.3f}  d={lin_row['stokes_d']:.3f}"
    )

    print("Loading MFA 36H data...")
    data = np.load(qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy", allow_pickle=True)

    seg_circ = data[int(circ_row["segment_idx"])]
    seg_lin = data[int(lin_row["segment_idx"])]
    date_circ = seg_circ.datetime[0].strftime("%Y-%m-%d")
    date_lin = seg_lin.datetime[0].strftime("%Y-%m-%d")

    use_paper_style()
    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [2, 1.5, 2, 1.5]},
    )
    fig.subplots_adjust(hspace=0.12, left=0.08, right=0.95, top=0.95, bottom=0.06)

    plot_example(
        ax_a,
        ax_b,
        seg_circ,
        period_sec=float(circ_row["period_min"]) * 60.0,
        title=(
            f"EXAMPLE 1 (Circular: |e|={abs(circ_row['ellipticity']):.2f}, "
            f"T={circ_row['period_min']:.1f} min) — {date_circ}"
        ),
        panel_labels=("a", "b"),
    )
    plot_example(
        ax_c,
        ax_d,
        seg_lin,
        period_sec=float(lin_row["period_min"]) * 60.0,
        title=(
            f"EXAMPLE 2 (Linear: |e|={abs(lin_row['ellipticity']):.2f}, "
            f"T={lin_row['period_min']:.1f} min) — {date_lin}"
        ),
        panel_labels=("c", "d"),
    )

    out_path = Path("output/figure10.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
