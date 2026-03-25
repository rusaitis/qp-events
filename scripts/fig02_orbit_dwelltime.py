"""Figure 2: Cassini orbit and dwell time map.

Panel (a): Cassini orbit in LT vs magnetic latitude, colored by MS/SH/SW.
Panel (b): Dwell time heatmap in conjugate latitude vs local time bins.
           Green line = 10 R_S shell, dashed yellow = 20 R_S, dotted = 25 R_S.

Referee: remove spurious tilted dashed line, make green 10 R_S more visible.
"""

import sys
import types
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
from qp.fieldline.tracer import trace_dipole_fieldline_bidirectional, conjugate_latitude


def load_orbit_data():
    """Load orbit positions and locations from KSM segments.

    Uses KSM coordinates to compute magnetic latitude = arcsin(z/r).
    Each segment contributes one point (daily resolution).
    """
    data = np.load(qp.DATA_PRODUCTS / "Cassini_MAG_KSM_36H.npy", allow_pickle=True)
    crossings = np.load(qp.DATA_PRODUCTS / "CROSSINGS.npy", allow_pickle=True)
    crossing_times = crossings[0]
    crossing_locs = crossings[1]

    lts, mlats, locations, flags = [], [], [], []
    for seg in data:
        if not isinstance(seg.info, dict):
            continue

        flag = seg.flag
        # Use midpoint of segment for position
        mid = len(seg.COORDS[0].y) // 2
        x = seg.COORDS[0].y[mid]  # x in KSM
        y = seg.COORDS[1].y[mid]  # y in KSM
        z = seg.COORDS[2].y[mid]  # z in KSM
        r = np.sqrt(x**2 + y**2 + z**2)

        if r < 1:
            continue

        mlat = np.degrees(np.arcsin(z / r))
        lt = (np.degrees(np.arctan2(y, x)) / 15.0 + 12.0) % 24.0

        # Location from crossings data
        seg_time = seg.datetime[mid]
        loc_idx = np.searchsorted(crossing_times, seg_time)
        loc_idx = min(loc_idx, len(crossing_locs) - 1)
        loc = int(crossing_locs[loc_idx])

        lts.append(lt)
        mlats.append(mlat)
        locations.append(loc)
        flags.append(flag)

    return np.array(lts), np.array(mlats), np.array(locations), flags


def load_dwell_time_map():
    """Sum pre-computed surfaceMaps across all years."""
    backup_dir = qp.OUTPUT_DIR / "Backup"
    total_map = None

    for year in range(2004, 2018):
        year_dir = backup_dir / str(year)
        if not year_dir.exists():
            continue
        for fname in os.listdir(year_dir):
            if fname.startswith("surfaceMap_Total") and fname.endswith(".npy"):
                sm = np.load(year_dir / fname)
                if total_map is None:
                    total_map = sm.copy()
                else:
                    total_map += sm
                break

    if total_map is None:
        raise FileNotFoundError("No surfaceMap files found in Output/Backup/")

    # Sum over magnetic latitude axis → 2D (inv_lat, LT)
    dwell_2d = np.sum(total_map, axis=2)
    return dwell_2d


def compute_shell_curves(l_shells=[10, 20, 25]):
    """Compute conjugate latitude vs LT for dipole field line shells.

    Returns dict: {L: (lt_array, lat_north, lat_south)}
    """
    curves = {}
    for L in l_shells:
        # For a dipole, conjugate latitude only depends on L:
        # sin^2(colat) = R_surface / L → colat = arcsin(1/sqrt(L))
        colat = np.degrees(np.arcsin(1.0 / np.sqrt(L)))
        lat = 90.0 - colat  # same for all LT in a dipole
        # Create arrays spanning all local times
        lt = np.linspace(0, 24, 200)
        lat_north = np.full_like(lt, lat)
        lat_south = np.full_like(lt, -lat)
        curves[L] = (lt, lat_north, lat_south)
    return curves


def main():
    print("Loading orbit data from MFA segments...")
    lts, mlats, locations, flags = load_orbit_data()
    print(f"  {len(lts)} data points")

    print("Loading dwell time map...")
    dwell_2d = load_dwell_time_map()
    print(f"  Shape: {dwell_2d.shape}")

    print("Computing field line shell curves...")
    shells = compute_shell_curves()

    # --- Plot ---
    use_paper_style()

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(10, 10),
        gridspec_kw={"height_ratios": [1, 1.5]},
    )
    fig.subplots_adjust(hspace=0.18, left=0.10, right=0.92, top=0.95, bottom=0.06)

    # ===== Panel (a): Orbit scatter =====
    # Plot unflagged data by location
    unflagged = np.array([f is None for f in flags])
    flagged = ~unflagged

    loc_config = [
        (0, "#4488ff", "Magnetosphere"),
        (1, "#f29539", "Magnetosheath"),
        (2, "#e04040", "Solar Wind"),
    ]

    for loc_code, color, label in loc_config:
        mask = (locations == loc_code) & unflagged
        if np.any(mask):
            ax_a.scatter(lts[mask], mlats[mask], c=color, s=4, alpha=0.4,
                         edgecolors="none", label=label, rasterized=True)

    # Plot flagged data as black crosses
    if np.any(flagged):
        ax_a.scatter(lts[flagged], mlats[flagged], c="black", s=10, marker="x",
                     alpha=0.6, label="Flagged Data", linewidths=0.5)

    ax_a.set_xlabel("Local Time (h)", fontsize=13)
    ax_a.set_ylabel("Magnetic Latitude (deg)", fontsize=13)
    ax_a.set_title("Spacecraft Location", fontsize=15)
    ax_a.set_xlim(0, 24)
    ax_a.set_ylim(-70, 70)
    ax_a.legend(loc="lower center", ncol=5, frameon=False, fontsize=9,
                markerscale=3, bbox_to_anchor=(0.5, -0.01))
    ax_a.text(0.01, 0.97, "a", transform=ax_a.transAxes, fontsize=18,
              fontweight="bold", va="top")
    ax_a.tick_params(labelsize=11)
    style_axes(ax_a, minimal=False)

    # ===== Panel (b): Dwell time heatmap =====
    # Axes: inv_lat (-90 to 90, 180 bins), LT (0 to 24, 48 bins)
    lt_edges = np.linspace(0, 24, 49)
    lat_edges = np.linspace(-90, 90, 181)

    # Convert seconds to days
    dwell_days = dwell_2d / (3600 * 24)

    # Mask zero bins
    dwell_masked = np.ma.masked_where(dwell_days == 0, dwell_days)

    im = ax_b.pcolormesh(
        lt_edges, lat_edges, dwell_masked,
        cmap="inferno",
        norm=mcolors.LogNorm(vmin=0.1, vmax=np.max(dwell_days)),
        shading="auto",
        rasterized=True,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_b, pad=0.02, aspect=30)
    cbar.set_label("Cassini Dwell Time (days)", fontsize=12)

    # Overlay field line shells
    shell_styles = {
        10: ("green", "-", 2.5, "10 $R_S$"),
        20: ("#FFD700", "--", 1.5, "20 $R_S$"),
        25: ("#FFD700", ":", 1.5, "25 $R_S$"),
    }
    for L, (lt, lat_n, lat_s) in shells.items():
        color, ls, lw, label = shell_styles[L]
        ax_b.plot(lt, lat_n, color=color, ls=ls, lw=lw, label=label)
        ax_b.plot(lt, lat_s, color=color, ls=ls, lw=lw)

    ax_b.set_xlabel("Local Time (h)", fontsize=13)
    ax_b.set_ylabel("Conjugate Latitude (deg)", fontsize=13)
    ax_b.set_title("Dwell Time", fontsize=15, color="orange")
    ax_b.set_xlim(0, 24)
    ax_b.set_ylim(-90, -65)
    ax_b.invert_yaxis()

    # Create a broken y-axis effect: show 70-90 (North) on top, -90 to -70 (South) on bottom
    # Actually, the original shows conjugate lat from ~70 to 90 (North) and -70 to -90 (South)
    # as two separate bands. Use a single axis with limits showing both polar regions.
    ax_b.set_ylim(-90, 90)

    # Mask the equatorial region (not interesting for conjugate latitude)
    # The data already has most signal at high latitudes (>65°)
    # Crop to show only |lat| > 65° by splitting into two sub-axes

    # Remove the single axis approach and use two sub-panels
    ax_b.remove()

    # Create two sub-axes for North and South
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.75, 0.75],
                          hspace=0.18, left=0.10, right=0.88, top=0.95, bottom=0.06)
    # Keep ax_a in place (already drawn)
    ax_n = fig.add_subplot(gs[1])
    ax_s = fig.add_subplot(gs[2])

    for ax, lat_lo, lat_hi, label in [(ax_n, 65, 90, "North"), (ax_s, -90, -65, "South")]:
        # Select the rows for this latitude range
        lat_idx_lo = int((lat_lo + 90))  # bin index
        lat_idx_hi = int((lat_hi + 90))
        dwell_sub = dwell_masked[lat_idx_lo:lat_idx_hi, :]
        lat_sub_edges = lat_edges[lat_idx_lo:lat_idx_hi + 1]

        im = ax.pcolormesh(
            lt_edges, lat_sub_edges, dwell_sub,
            cmap="inferno",
            norm=mcolors.LogNorm(vmin=0.1, vmax=np.max(dwell_days)),
            shading="auto",
            rasterized=True,
        )

        # Overlay shells
        for L, (lt_sh, lat_n_sh, lat_s_sh) in shells.items():
            color, ls, lw, slabel = shell_styles[L]
            if lat_lo > 0:
                ax.plot(lt_sh, lat_n_sh, color=color, ls=ls, lw=lw,
                        label=slabel if ax is ax_n else None)
            else:
                ax.plot(lt_sh, lat_s_sh, color=color, ls=ls, lw=lw,
                        label=slabel if ax is ax_s else None)

        ax.set_xlim(0, 24)
        ax.set_ylabel("Conjugate Lat (deg)", fontsize=11)
        ax.text(0.5, 0.92, label, transform=ax.transAxes, fontsize=12,
                color="white", ha="center", va="top")
        ax.tick_params(labelsize=11)
        style_axes(ax, minimal=False, grid=False)

    ax_n.set_xticklabels([])
    ax_s.set_xlabel("Local Time (h)", fontsize=13)
    ax_n.set_title("Dwell Time", fontsize=15, color="orange")
    ax_n.text(0.01, 0.92, "b", transform=ax_n.transAxes, fontsize=18,
              fontweight="bold", va="top")
    ax_n.legend(loc="lower right", frameon=False, fontsize=9, ncol=3)

    # Colorbar
    cbar = fig.colorbar(im, ax=[ax_n, ax_s], pad=0.02, aspect=30)
    cbar.set_label("Cassini Dwell Time (days)", fontsize=12)

    style_axes(ax_b, minimal=False, grid=False)

    plt.savefig("output/figure2.png", dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved output/figure2.png")
    plt.close()


if __name__ == "__main__":
    main()
