"""Figure 3: Cassini dwell time along KMAG-traced field lines.

Shows field lines in the noon-midnight meridian (x-z plane), colored
by Cassini dwell time in magnetic latitude bins. Left = midnight (0 LT ± 2h),
right = noon (12 LT ± 2h). Uses realistic KMAG field model.

Referee: dark background.
"""

import sys
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "src"))

import qp
from qp.fieldline.dwell_time import value_to_bin
from qp.plotting.style import use_paper_style, BG_COLOR

# SurfaceMap bins
N_LAT_BINS = 180
N_LT_BINS = 48
N_MLAT_BINS = 36

# Colatitudes to trace (degrees from pole)
COLATS_DEG = [10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50, 55, 60, 65, 70]


def load_total_surface_map():
    """Sum surfaceMaps across all years."""
    backup_dir = qp.OUTPUT_DIR / "Backup"
    total = None
    for year in range(2004, 2018):
        year_dir = backup_dir / str(year)
        if not year_dir.exists():
            continue
        for fname in os.listdir(year_dir):
            if fname.startswith("surfaceMap_Total") and fname.endswith(".npy"):
                sm = np.load(year_dir / fname)
                total = sm.copy() if total is None else total + sm
                break
    return total


def run_kmag_traces(colats_deg, side="noon"):
    """Run KMAG tracer for multiple starting colatitudes.

    side: 'noon' (x > 0) or 'midnight' (x < 0).
    Returns list of (x, z) trace arrays.
    """
    kmag_dir = _project_root / "KMAGhelper"
    output_dir = _project_root / "Output"

    # Clean old trace files
    for f in output_dir.glob("0*.txt"):
        if "eigenmodes" not in f.name:
            f.unlink()

    # Epoch: mid-mission 2009
    j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
    t_ref = datetime.datetime(2009, 1, 1, 0, 0, 0)
    etime = (t_ref - j2000).total_seconds()

    # Write params
    params_fields = ["Epoch", "ETime(sec)", "BY_IMF", "BZ_IMF", "Dp", "IN_COORD",
                     "OUT_COORD", "COMMENT", "TRACE(Y/N)", "CARSPH", "STEP", "FLATTENING"]
    params_values = ["j2000", str(etime), "-0.2", "0.1", "0.017", "KSM",
                     "DIS", "000", "Y", "CAR", "0.01", "0.09796"]
    data = np.vstack([params_fields, params_values])
    np.savetxt(kmag_dir / "kmag_params.txt", data, fmt="%14s", delimiter=" ")

    # Starting positions on the surface
    starts = []
    r = 1.1
    sign = 1 if side == "noon" else -1
    for colat_deg in colats_deg:
        colat = np.radians(colat_deg)
        x = sign * r * np.sin(colat)
        z = r * np.cos(colat)
        starts.append([etime, x, 0.0, z])

    header = ["Etime(sec)", "X(RS)", "Y(RS)", "Z(RS)"]
    data_in = np.vstack([header, starts])
    np.savetxt(kmag_dir / "kmag_input.txt", data_in, fmt="%14s", delimiter=" ")

    # Run KMAG
    import subprocess
    result = subprocess.run(
        [str(_project_root / "fortran" / "KMAG")],
        cwd=str(_project_root),
        capture_output=True, text=True,
    )

    # Read traces
    traces = []
    for fname in sorted(output_dir.glob("0*.txt")):
        if "eigenmodes" in fname.name:
            continue
        try:
            raw = np.loadtxt(fname, skiprows=1, dtype="U")
            raw = np.atleast_2d(raw)
            if raw.shape[0] < 10:
                continue
            x = raw[:, 1].astype(float)
            y = raw[:, 2].astype(float)
            z = raw[:, 3].astype(float)
            r_max = np.max(np.sqrt(x**2 + y**2 + z**2))
            # Skip open field lines (go too far)
            if r_max > 50:
                continue
            traces.append(np.column_stack([x, y, z]))
        except Exception:
            continue

    return traces


def extract_dwell_vs_mlat(surface_map, inv_lat_deg, lt_center, lt_half_width):
    """Extract dwell time vs magnetic latitude for given inv lat and LT range."""
    lat_bin = value_to_bin(inv_lat_deg, -90, 90, N_LAT_BINS)

    lt_lo = (lt_center - lt_half_width) % 24
    lt_hi = (lt_center + lt_half_width) % 24

    if lt_lo < lt_hi:
        lt_bin_lo = value_to_bin(lt_lo, 0, 24, N_LT_BINS)
        lt_bin_hi = value_to_bin(lt_hi, 0, 24, N_LT_BINS)
        dwell = surface_map[lat_bin, lt_bin_lo:lt_bin_hi, :].sum(axis=0)
    else:
        lt_bin_lo = value_to_bin(lt_lo, 0, 24, N_LT_BINS)
        lt_bin_hi = value_to_bin(lt_hi, 0, 24, N_LT_BINS)
        dwell = (surface_map[lat_bin, lt_bin_lo:, :].sum(axis=0) +
                 surface_map[lat_bin, :lt_bin_hi, :].sum(axis=0))

    return dwell / (3600 * 24)  # seconds to days


def map_dwell_to_trace(trace_xyz, dwell_vs_mlat):
    """Map dwell time onto trace points using magnetic latitude."""
    r = np.sqrt(np.sum(trace_xyz**2, axis=1))
    mlat_deg = np.degrees(np.arcsin(trace_xyz[:, 2] / np.clip(r, 1e-10, None)))
    bins = value_to_bin(mlat_deg, -90, 90, N_MLAT_BINS)
    bins = np.clip(np.atleast_1d(bins), 0, N_MLAT_BINS - 1)
    return dwell_vs_mlat[bins]


def colored_line(ax, x, y, c, cmap, norm, lw=3):
    """Plot a line colored by scalar array."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(c[:-1])
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    return lc


def l_shell_to_inv_lat(L):
    """L-shell to invariant latitude (degrees)."""
    if L <= 1:
        return 0.0
    return 90.0 - np.degrees(np.arcsin(1.0 / np.sqrt(L)))


def main():
    print("Loading surface map...")
    surface_map = load_total_surface_map()

    cmap = plt.get_cmap("inferno")
    vmin, vmax = 0.05, 0.9  # days
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    use_paper_style()
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Trace field lines for NOON side
    print("Tracing noon field lines with KMAG...")
    noon_traces = run_kmag_traces(COLATS_DEG, side="noon")
    print(f"  Got {len(noon_traces)} closed traces")

    # Trace field lines for MIDNIGHT side
    print("Tracing midnight field lines with KMAG...")
    midnight_traces = run_kmag_traces(COLATS_DEG, side="midnight")
    print(f"  Got {len(midnight_traces)} closed traces")

    # Plot noon traces (x > 0 side)
    for trace in noon_traces:
        r_eq = np.max(np.sqrt(trace[:, 0]**2 + trace[:, 1]**2 + trace[:, 2]**2))
        inv_lat = l_shell_to_inv_lat(r_eq)
        dwell = extract_dwell_vs_mlat(surface_map, inv_lat, 12, 2)
        dwell_neg = extract_dwell_vs_mlat(surface_map, -inv_lat, 12, 2)
        dwell_avg = (dwell + dwell_neg) / 2
        colors = map_dwell_to_trace(trace, dwell_avg)
        # Project to x-z plane (use abs(x) for noon side)
        x_plot = np.abs(trace[:, 0])
        colored_line(ax, x_plot, trace[:, 2], colors, cmap, norm, lw=2.5)

    # Plot midnight traces (x < 0 side)
    for trace in midnight_traces:
        r_eq = np.max(np.sqrt(trace[:, 0]**2 + trace[:, 1]**2 + trace[:, 2]**2))
        inv_lat = l_shell_to_inv_lat(r_eq)
        dwell = extract_dwell_vs_mlat(surface_map, inv_lat, 0, 2)
        dwell_neg = extract_dwell_vs_mlat(surface_map, -inv_lat, 0, 2)
        dwell_avg = (dwell + dwell_neg) / 2
        colors = map_dwell_to_trace(trace, dwell_avg)
        x_plot = -np.abs(trace[:, 0])
        colored_line(ax, x_plot, trace[:, 2], colors, cmap, norm, lw=2.5)

    # Saturn
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill(np.cos(theta), np.sin(theta), color="#b8a050", alpha=0.6, zorder=5)
    ax.plot(np.cos(theta), np.sin(theta), color="white", lw=0.5, zorder=5)

    ax.set_xlabel(r"$x$ ($R_S$)", fontsize=16)
    ax.set_ylabel(r"$z$ ($R_S$)", fontsize=16)
    ax.set_xlim(-22, 22)
    ax.set_ylim(-18, 18)
    ax.set_aspect("equal")
    ax.tick_params(colors="white", labelsize=13)
    ax.axvline(0, color="grey", ls="--", lw=0.5, alpha=0.3)
    ax.grid(True, color="grey", alpha=0.15, ls="--")

    ax.text(-10, 16, r"0 LT $\pm$ 2h", fontsize=16, color="white", ha="center",
            bbox=dict(fc=BG_COLOR, ec="white", boxstyle="round,pad=0.3", alpha=0.8))
    ax.text(10, 16, r"12 LT $\pm$ 2h", fontsize=16, color="white", ha="center",
            bbox=dict(fc=BG_COLOR, ec="white", boxstyle="round,pad=0.3", alpha=0.8))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_label("Dwell Time (days)", fontsize=14, color="white")
    cbar.ax.tick_params(colors="white")

    plt.savefig("output/figure3.png", dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved output/figure3.png")
    plt.close()


if __name__ == "__main__":
    main()
