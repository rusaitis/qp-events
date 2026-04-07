#!/usr/bin/env python
"""Plot dwell-time maps from the Cassini mission grid.

Produces four core figures:

1. **Invariant latitude vs LT** — side-by-side comparison of the analytical
   dipole invariant latitude and the KMAG-traced invariant latitude. Shows
   where on Saturn's auroral oval Cassini's sampled field lines map to.

2. **Equatorial (L, LT) polar plot** — orbit coverage as a function of
   L-shell and local time, computed by mapping each (r, mag_lat) 3D bin
   to L = r/cos²(mag_lat). This is the view from Saturn's north pole.

3. **Meridional (ρ, z) plot** — side view of the magnetosphere in a fixed
   meridian (summed over LT), showing dwell time in cylindrical-radius
   and height above the equator. Saturn at origin, equator horizontal.

4. **Magnetic latitude vs LT** (general + plasma sheet) — paper Fig 2a
   analog. Two panels: all MS samples, and only the plasma sheet
   (|B| < 2 nT) for SI Fig 2. The plasma sheet panel is computed
   on-the-fly from raw PDS data and cached in Output/cache/.

Usage
-----
    uv run python scripts/plot_dwell_maps.py

    # Custom input / output:
    uv run python scripts/plot_dwell_maps.py \\
        --input Output/dwell_grid_cassini_saturn.zarr \\
        --outdir Output/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from qp.plotting.style import save_figure, use_paper_style

log = logging.getLogger(__name__)


def plot_inv_lat_vs_lt(ds: xr.Dataset, outpath: Path) -> None:
    """Side-by-side heatmap of dwell time in (inv_lat, LT) for dipole vs KMAG."""
    # Extract data (convert minutes -> hours)
    dipole = ds["dipole_inv_lat_magnetosphere"].values / 60.0
    kmag = ds["kmag_inv_lat_closed_magnetosphere"].values / 60.0

    inv_lat = ds["dipole_inv_lat"].values  # degrees
    lt = ds["local_time"].values  # hours

    # Find a common color scale that keeps both panels comparable
    vmax = max(dipole.max(), kmag.max())
    vmin = max(0.05, min(dipole[dipole > 0].min() if (dipole > 0).any() else 0.1,
                          kmag[kmag > 0].min() if (kmag > 0).any() else 0.1))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.set_facecolor(plt.rcParams["figure.facecolor"])

    panels = [
        (axes[0], dipole, "Dipole invariant latitude"),
        (axes[1], kmag, "KMAG traced invariant latitude (closed)"),
    ]

    for ax, data, title in panels:
        im = ax.pcolormesh(lt, inv_lat, data, norm=norm, cmap="inferno", shading="auto")
        ax.set_xlabel("Local time (h)")
        ax.set_title(title)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["00", "06", "12", "18", "24"])
        ax.set_ylim(-90, 90)
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
        # Annotate approximate FLR harmonic L-shell bands
        # QP30 ~L=6 → 65.9°,  QP60 ~L=10 → 71.6°,  QP120 ~L=20 → 77.1°
        for lat_ref, label in [(65.9, "QP30"), (71.6, "QP60"), (77.1, "QP120")]:
            ax.axhline(lat_ref, color="cyan", linewidth=0.5, linestyle="--", alpha=0.4)
            ax.axhline(-lat_ref, color="cyan", linewidth=0.5, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Invariant latitude (°)")

    cbar = fig.colorbar(im, ax=axes, pad=0.02, label="Dwell time (hours)", shrink=0.85)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    fig.suptitle(
        "Cassini dwell time mapped to auroral invariant latitude (magnetosphere only)",
        y=0.98, color="white",
    )
    save_figure(fig, outpath)


def plot_equatorial_L_polar(ds: xr.Dataset, outpath: Path, L_max: float = 40.0) -> None:
    """Polar (L, LT) plot of dwell time from the 3D grid.

    For each (r, mag_lat, LT) bin, compute L = r / cos²(λ_mag) and
    accumulate dwell time into (L, LT) bins. Displayed as a polar heatmap
    with noon at top and dawn on the right (view from Saturn's north pole).
    """
    # Restrict to magnetosphere only — no point plotting where KMAG is invalid
    total = ds["magnetosphere"].values / 60.0  # hours, shape (n_r, n_lat, n_lt)

    r = ds["r"].values  # (n_r,)
    mag_lat = ds["magnetic_latitude"].values  # (n_lat,) degrees
    lt = ds["local_time"].values  # (n_lt,)

    # Compute L for each (r, mag_lat) cell
    r_mesh, lat_mesh = np.meshgrid(r, mag_lat, indexing="ij")
    cos_lat = np.cos(np.radians(lat_mesh))
    # Avoid div-by-zero near the poles (cos_lat → 0) — cap at a tiny value
    cos_lat = np.where(np.abs(cos_lat) < 1e-3, 1e-3, cos_lat)
    L_mesh = r_mesh / cos_lat ** 2  # (n_r, n_lat)

    # Rebin into (L, LT)
    n_L = 80
    L_edges = np.linspace(0, L_max, n_L + 1)
    L_centers = 0.5 * (L_edges[:-1] + L_edges[1:])

    L_flat = L_mesh.flatten()  # (n_r * n_lat,)
    valid_geom = np.isfinite(L_flat) & (L_flat >= 0) & (L_flat < L_max)

    n_lt = len(lt)
    L_lt = np.zeros((n_L, n_lt))
    for i_lt in range(n_lt):
        w = total[:, :, i_lt].flatten()
        L_lt[:, i_lt], _ = np.histogram(
            L_flat[valid_geom], bins=L_edges, weights=w[valid_geom]
        )

    # Polar plot: noon at top, clockwise → dawn on right, midnight at bottom
    fig = plt.figure(figsize=(10, 10))
    fig.set_facecolor(plt.rcParams["figure.facecolor"])
    ax = fig.add_subplot(111, projection="polar", facecolor=plt.rcParams["axes.facecolor"])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Convert LT (hours) to angle so that noon → top, dawn → right, etc.
    theta = (lt / 24.0) * 2.0 * np.pi
    # The theta offset already handles LT=12 → top via set_theta_zero_location.
    # We need theta=0 to correspond to LT=12 though. With zero at 'N' and
    # clockwise direction, theta=0 is at top. So we need LT=12 → theta=0:
    theta = ((lt - 12.0) / 24.0) * 2.0 * np.pi
    # Build mesh for pcolormesh: theta and L edges
    theta_edges = ((np.concatenate([lt - 0.125, [lt[-1] + 0.125]]) - 12.0) / 24.0) * 2.0 * np.pi
    Theta, R = np.meshgrid(theta_edges, L_edges)

    # Mask zeros for a clean log plot
    Z = np.ma.masked_where(L_lt <= 0, L_lt)

    vmin = max(0.1, Z.min() if Z.count() > 0 else 0.1)
    vmax = Z.max() if Z.count() > 0 else 1.0
    im = ax.pcolormesh(
        Theta, R, Z, norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="inferno", shading="auto",
    )

    ax.set_rlim(0, L_max)
    ax.set_rticks([10, 20, 30, 40][: max(1, int(L_max // 10))])
    ax.set_rlabel_position(45)
    ax.tick_params(colors="white")
    ax.grid(color="white", alpha=0.25)

    # LT tick labels
    theta_ticks = np.radians([0, 90, 180, 270])  # noon, dusk (?), midnight, dawn (?)
    # With set_theta_zero_location('N') + direction=-1:
    #   theta=0 → top (noon), theta=π/2 → right (going clockwise 6h = LT=6 = dawn)
    #   theta=π → bottom (midnight), theta=3π/2 → left (LT=18 = dusk)
    ax.set_xticks(theta_ticks)
    ax.set_xticklabels(["12\nnoon", "06\ndawn", "00\nmidnight", "18\ndusk"])

    cbar = fig.colorbar(im, ax=ax, pad=0.12, label="Dwell time (hours)", shrink=0.75)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_title(
        f"Cassini equatorial L-shell dwell time (MS, L ≤ {L_max:.0f} R$_S$)\n"
        f"View from Saturn's north pole",
        color="white", pad=20,
    )

    save_figure(fig, outpath)


def plot_meridional_rho_z(
    ds: xr.Dataset,
    outpath: Path,
    r_max: float = 40.0,
) -> None:
    """Meridional side view (ρ, z) of dwell time summed over LT.

    For each (r, mag_lat) bin, convert to cylindrical (ρ, z):

        ρ = r * cos(lat)   (distance from spin axis)
        z = r * sin(lat)   (height above equator)

    This gives a side view of the magnetosphere with Saturn at the origin,
    equator horizontal, and both hemispheres visible.
    """
    # Sum the 3D MS grid over local time → (r, mag_lat)
    r_lat = ds["magnetosphere"].sum(dim="local_time").values / 60.0  # hours

    r_edges = ds["r_edges"].values  # (n_r + 1,)
    lat_edges = ds["lat_edges"].values  # (n_lat + 1,)

    # Build curvilinear edge meshes for pcolormesh
    r_edge_m, lat_edge_m = np.meshgrid(r_edges, lat_edges, indexing="ij")
    lat_rad = np.radians(lat_edge_m)
    rho = r_edge_m * np.cos(lat_rad)
    z = r_edge_m * np.sin(lat_rad)

    # Mask zero cells for log-scale display
    Z = np.ma.masked_where(r_lat <= 0, r_lat)
    vmin = max(0.1, Z.min() if Z.count() > 0 else 0.1)
    vmax = Z.max() if Z.count() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 10))
    fig.set_facecolor(plt.rcParams["figure.facecolor"])
    ax.set_facecolor(plt.rcParams["axes.facecolor"])

    im = ax.pcolormesh(
        rho, z, Z,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="inferno", shading="auto",
    )

    # Draw Saturn's filled half-disk at origin (only the visible ρ ≥ 0 side)
    saturn = Circle((0, 0), 1.0, facecolor="#e8d5a2", edgecolor="white",
                    linewidth=0.8, zorder=10)
    ax.add_patch(saturn)

    # Reference arcs at 10, 20, 30, 40 R_S (only show the ρ ≥ 0 half)
    theta_arc = np.linspace(-np.pi / 2, np.pi / 2, 64)
    for r_ref in [10, 20, 30, 40]:
        if r_ref > r_max:
            break
        ax.plot(r_ref * np.cos(theta_arc), r_ref * np.sin(theta_arc),
                color="white", linewidth=0.5, linestyle="--", alpha=0.35, zorder=5)
        # Label at the equator (theta=0)
        ax.text(r_ref - 0.5, 0.4, f"{r_ref}",
                color="white", fontsize=9, alpha=0.7, ha="right", zorder=6)

    # Equator guide line
    ax.axhline(0, color="white", linewidth=0.4, alpha=0.3, zorder=4)

    ax.set_xlim(0, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\rho$ (cylindrical radius, R$_S$)")
    ax.set_ylabel(r"$z$ (R$_S$)")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, label="Dwell time (hours)", shrink=0.75)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_title(
        f"Cassini meridional dwell time (MS, summed over local time)\n"
        f"Side view — Saturn at origin, equator horizontal",
        color="white", pad=12,
    )

    save_figure(fig, outpath)


def _compute_weak_field_mag_lat_lt(
    b_threshold: float = 2.0,
    n_lat: int = 180,
    n_lt: int = 96,
    year_from: int = 2004,
    year_to: int = 2017,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram plasma sheet dwell time in (mag_lat, LT) from raw PDS data.

    Reads each year's MAG file, filters to magnetosphere samples with
    |B| < b_threshold, and histograms into a (n_lat, n_lt) grid of minutes.

    Returns (weak_hist, lat_edges, lt_edges). Caches to ``cache_path`` if given.
    """
    import datetime

    from qp.coords.ksm import local_time as compute_lt
    from qp.coords.ksm import magnetic_latitude
    from qp.io.crossings import crossing_lookup_arrays, parse_crossing_list
    from qp.io.pds import DATETIME_FMT, mag_filepath, read_timeseries_file

    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)
    lt_edges = np.linspace(0.0, 24.0, n_lt + 1)

    if cache_path is not None and cache_path.exists():
        log.info("Loading cached weak-field histogram from %s", cache_path)
        cached = np.load(cache_path)
        return cached["weak"], cached["lat_edges"], cached["lt_edges"]

    log.info("Computing weak-field (|B| < %.1f nT) dwell from raw PDS data...", b_threshold)
    crossings = parse_crossing_list()
    crossing_times_unix, crossing_codes = crossing_lookup_arrays(crossings)

    weak_h = np.zeros((n_lat, n_lt))
    for year in range(year_from, year_to + 1):
        path = mag_filepath(str(year), coords="KSM")
        if not path.exists():
            continue
        rows = read_timeseries_file(path)
        if not rows:
            continue
        data = np.array(rows)

        btot = data[:, 4].astype(float)
        x = data[:, 5].astype(float)
        y = data[:, 6].astype(float)
        z = data[:, 7].astype(float)

        # Region code lookup (MS only)
        sample_unix = np.array(
            [
                datetime.datetime.strptime(t, DATETIME_FMT)
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
                for t in data[:, 0]
            ],
            dtype=np.float64,
        )
        idx = np.searchsorted(crossing_times_unix, sample_unix) - 1
        codes = np.where(
            (idx >= 0) & (idx < len(crossing_codes)),
            crossing_codes[np.clip(idx, 0, len(crossing_codes) - 1)],
            9,
        )

        lat = magnetic_latitude(x, y, z)
        lt = compute_lt(x, y)

        mask = (
            (codes == 0)
            & np.isfinite(lat)
            & np.isfinite(lt)
            & (btot > 0)
            & (btot < b_threshold)
        )

        h, _, _ = np.histogram2d(
            lat[mask], lt[mask], bins=[lat_edges, lt_edges]
        )
        weak_h += h  # one sample = one minute
        log.info("  %d: %d plasma-sheet samples", year, int(mask.sum()))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, weak=weak_h, lat_edges=lat_edges, lt_edges=lt_edges)
        log.info("Cached to %s", cache_path)

    return weak_h, lat_edges, lt_edges


def plot_mag_lat_vs_lt(
    ds: xr.Dataset,
    outpath: Path,
    weak_hist: np.ndarray,
    weak_lat_edges: np.ndarray,
    weak_lt_edges: np.ndarray,
) -> None:
    """Side-by-side: general MS dwell vs plasma-sheet dwell in (mag_lat, LT).

    Left panel uses the zarr's 3D grid reduced over r.
    Right panel uses the on-the-fly weak-field histogram.
    """
    # General: sum magnetosphere grid over r → (mag_lat, LT)
    general = ds["magnetosphere"].sum(dim="r").values / 60.0  # hours
    mag_lat = ds["magnetic_latitude"].values
    lt = ds["local_time"].values

    weak_hours = weak_hist / 60.0  # minutes → hours
    # Bin centers for the weak-field grid
    weak_lat_centers = 0.5 * (weak_lat_edges[:-1] + weak_lat_edges[1:])
    weak_lt_centers = 0.5 * (weak_lt_edges[:-1] + weak_lt_edges[1:])

    # Shared log color scale
    vmin = 0.1
    vmax = max(general.max(), weak_hours.max())
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.set_facecolor(plt.rcParams["figure.facecolor"])

    # Panel 1: general MS dwell
    im1 = axes[0].pcolormesh(
        lt, mag_lat, general, norm=norm, cmap="inferno", shading="auto",
    )
    axes[0].set_title("All magnetosphere samples")
    axes[0].set_xlabel("Local time (h)")
    axes[0].set_ylabel("Magnetic latitude (°)")

    # Panel 2: plasma sheet (|B| < 2 nT)
    axes[1].pcolormesh(
        weak_lt_centers, weak_lat_centers, weak_hours,
        norm=norm, cmap="inferno", shading="auto",
    )
    axes[1].set_title(r"Plasma sheet only ($|B| < 2$ nT)")
    axes[1].set_xlabel("Local time (h)")

    for ax in axes:
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["00", "06", "12", "18", "24"])
        ax.set_ylim(-90, 90)
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
        # Guide band at ±30° (plasma sheet criterion orientation range)
        ax.axhline(30, color="cyan", linewidth=0.4, linestyle="--", alpha=0.3)
        ax.axhline(-30, color="cyan", linewidth=0.4, linestyle="--", alpha=0.3)

    cbar = fig.colorbar(im1, ax=axes, pad=0.02, label="Dwell time (hours)", shrink=0.85)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    fig.suptitle(
        "Cassini dwell time in magnetic latitude vs local time",
        y=0.98, color="white",
    )
    save_figure(fig, outpath)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str,
                        default="Output/dwell_grid_cassini_saturn.zarr",
                        help="Path to dwell grid zarr")
    parser.add_argument("--outdir", type=str, default="Output/figures",
                        help="Directory to write figures")
    parser.add_argument("--L-max", type=float, default=40.0,
                        help="Maximum L-shell for equatorial plot")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    use_paper_style()

    log.info("Loading %s", args.input)
    ds = xr.open_zarr(args.input)
    log.info(
        "  grid %dx%dx%d, %s years, %d total samples",
        ds.attrs["n_r"], ds.attrs["n_lat"], ds.attrs["n_lt"],
        f"{ds.attrs['year_from']}-{ds.attrs['year_to']}",
        ds.attrs["total_samples"],
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_inv_lat_vs_lt(ds, outdir / "dwell_inv_lat_vs_lt.png")
    plot_equatorial_L_polar(ds, outdir / "dwell_equatorial_L_polar.png", L_max=args.L_max)
    plot_meridional_rho_z(ds, outdir / "dwell_meridional_rho_z.png", r_max=args.L_max)

    # Weak-field histogram needs a raw-data pass (cached after first run)
    cache_path = Path("Output/cache/weak_field_mag_lat_lt.npz")
    weak_hist, weak_lat_edges, weak_lt_edges = _compute_weak_field_mag_lat_lt(
        n_lat=ds.attrs["n_lat"],
        n_lt=ds.attrs["n_lt"],
        year_from=ds.attrs["year_from"],
        year_to=ds.attrs["year_to"],
        cache_path=cache_path,
    )
    plot_mag_lat_vs_lt(
        ds, outdir / "dwell_mag_lat_vs_lt.png",
        weak_hist, weak_lat_edges, weak_lt_edges,
    )

    print(f"\nSaved figures to {outdir}/")
    print(f"  - dwell_inv_lat_vs_lt.png")
    print(f"  - dwell_equatorial_L_polar.png")
    print(f"  - dwell_meridional_rho_z.png")
    print(f"  - dwell_mag_lat_vs_lt.png")


if __name__ == "__main__":
    main()
