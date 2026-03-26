#!/usr/bin/env python
"""Verification plots for the Cassini dwell-time grid.

Loads the zarr dataset produced by compute_dwell_grid.py and creates:
1. Equatorial slice (|lat| < 5°) — polar heatmap in (r, LT)
2. Noon-midnight meridian (x-z plane) — 2D heatmap
3. Dawn-dusk meridian (y-z plane) — 2D heatmap
4. Orbit trajectory overlay on equatorial slice
5. Total time integral validation

Usage
-----
    uv run python scripts/plot_dwell_slices.py
    uv run python scripts/plot_dwell_slices.py --input Output/dwell_grid.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from qp.dwell.io import load_zarr
from qp.plotting.style import use_paper_style


def equatorial_slice(ds, ax, variable="total"):
    """Plot equatorial cut (|lat| < 5°) as polar heatmap in (r, LT)."""
    lat = ds.coords["magnetic_latitude"].values
    lat_mask = np.abs(lat) < 5.0
    data = ds[variable].values[:, lat_mask, :].sum(axis=1)  # sum over lat

    r = ds.coords["r"].values
    lt = ds.coords["local_time"].values
    theta = lt / 24.0 * 2 * np.pi  # LT → angle (0h = top)

    R, Theta = np.meshgrid(r, theta, indexing="ij")
    pcm = ax.pcolormesh(
        Theta, R, np.log10(data + 1),
        cmap="inferno", shading="auto",
    )
    ax.set_theta_zero_location("S")  # midnight at bottom
    ax.set_theta_direction(-1)  # clockwise (standard LT convention)
    ax.set_rlabel_position(45)
    ax.set_xlabel("")
    ax.set_title("Equatorial slice (|lat| < 5°)", pad=15, fontsize=12)

    # LT labels
    lt_ticks = [0, 6, 12, 18]
    lt_labels = ["0h\n(midnight)", "6h\n(dawn)", "12h\n(noon)", "18h\n(dusk)"]
    ax.set_xticks([t / 24 * 2 * np.pi for t in lt_ticks])
    ax.set_xticklabels(lt_labels, fontsize=8)

    return pcm


def meridian_slice(ds, ax, lt_center, lt_half_width=1.0, variable="total", title=""):
    """Plot a meridional cut (r vs lat) for a given LT range."""
    lt = ds.coords["local_time"].values
    r = ds.coords["r"].values
    lat = ds.coords["magnetic_latitude"].values

    # Handle LT wraparound
    lt_lo = lt_center - lt_half_width
    lt_hi = lt_center + lt_half_width
    if lt_lo < 0:
        lt_mask = (lt >= lt_lo + 24) | (lt <= lt_hi)
    elif lt_hi > 24:
        lt_mask = (lt >= lt_lo) | (lt <= lt_hi - 24)
    else:
        lt_mask = (lt >= lt_lo) & (lt <= lt_hi)

    data = ds[variable].values[:, :, lt_mask].sum(axis=2)  # sum over LT slice

    R, Lat = np.meshgrid(r, lat, indexing="ij")
    pcm = ax.pcolormesh(
        Lat, R, np.log10(data + 1),
        cmap="inferno", shading="auto",
    )
    ax.set_xlabel("Magnetic latitude [°]")
    ax.set_ylabel("r [R_S]")
    ax.set_title(title, fontsize=12)
    return pcm


def main():
    parser = argparse.ArgumentParser(description="Plot dwell-time grid slices")
    parser.add_argument("--input", type=str, default="Output/dwell_grid.zarr")
    parser.add_argument("--output-dir", type=str, default="Output")
    parser.add_argument("--variable", type=str, default="total",
                        choices=["total", "magnetosphere", "magnetosheath", "solar_wind"])
    args = parser.parse_args()

    use_paper_style()
    plt.style.use("dark_background")

    ds = load_zarr(args.input)
    var = args.variable
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Three slices ---
    fig = plt.figure(figsize=(18, 12))
    fig.set_facecolor("#171717")

    # Equatorial
    ax_eq = fig.add_subplot(2, 2, 1, projection="polar")
    pcm = equatorial_slice(ds, ax_eq, variable=var)
    fig.colorbar(pcm, ax=ax_eq, label="log₁₀(minutes + 1)", shrink=0.8)

    # Noon-midnight (x-z plane)
    ax_xz = fig.add_subplot(2, 2, 2)
    pcm2 = meridian_slice(ds, ax_xz, lt_center=12.0, lt_half_width=1.0,
                          variable=var, title="Noon-midnight meridian (x-z)")
    # Also add the midnight side
    lt = ds.coords["local_time"].values
    r = ds.coords["r"].values
    lat_vals = ds.coords["magnetic_latitude"].values
    midnight_mask = (lt <= 1.0) | (lt >= 23.0)
    midnight_data = ds[var].values[:, :, midnight_mask].sum(axis=2)
    R2, Lat2 = np.meshgrid(r, lat_vals, indexing="ij")
    ax_xz.pcolormesh(-Lat2, R2, np.log10(midnight_data + 1),
                     cmap="inferno", shading="auto", alpha=0.5)
    fig.colorbar(pcm2, ax=ax_xz, label="log₁₀(minutes + 1)")

    # Dawn-dusk (y-z plane)
    ax_yz = fig.add_subplot(2, 2, 3)
    pcm3 = meridian_slice(ds, ax_yz, lt_center=6.0, lt_half_width=1.0,
                          variable=var, title="Dawn-dusk meridian (y-z)")
    fig.colorbar(pcm3, ax=ax_yz, label="log₁₀(minutes + 1)")

    # Summary text
    ax_text = fig.add_subplot(2, 2, 4)
    ax_text.axis("off")
    total_hours = float(ds[var].sum()) / 60
    year_from = ds.attrs.get("year_from", "?")
    year_to = ds.attrs.get("year_to", "?")
    text = (
        f"Cassini Dwell Time Grid\n"
        f"{'─' * 30}\n"
        f"Period: {year_from}–{year_to}\n"
        f"Variable: {var}\n"
        f"Total: {total_hours:,.0f} hours ({total_hours/8766:.1f} years)\n"
        f"Grid: {ds.sizes['r']}×{ds.sizes['magnetic_latitude']}×{ds.sizes['local_time']}\n"
        f"r: {float(ds.r.min()):.0f}–{float(ds.r.max()):.0f} R_S\n"
        f"lat: {float(ds.magnetic_latitude.min()):.0f}° to {float(ds.magnetic_latitude.max()):.0f}°\n"
        f"LT: {float(ds.local_time.min()):.1f}–{float(ds.local_time.max()):.1f} h"
    )
    ax_text.text(0.1, 0.5, text, transform=ax_text.transAxes,
                 fontsize=14, family="monospace", verticalalignment="center",
                 color="white")

    fig.suptitle(f"Cassini Dwell Time — {var}", fontsize=16, y=0.98)
    fig.tight_layout()
    outpath = outdir / f"dwell_slices_{var}.png"
    fig.savefig(outpath, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()

    # --- Validation: total time integral ---
    total_min = float(ds["total"].sum())
    total_h = total_min / 60
    n_samples = ds.attrs.get("total_samples", 0)
    expected_h = n_samples / 60  # 1-min samples
    stored_h = ds.attrs.get("total_hours", 0)
    print("\nValidation:")
    print(f"  Grid integral:  {total_h:,.1f} hours")
    print(f"  Stored total:   {stored_h:,.1f} hours (from compute script)")
    print(f"  Raw samples:    {expected_h:,.1f} hours (from {n_samples:,} 1-min samples)")
    if stored_h > 0:
        # Compare grid integral against the stored total (which accounts for out-of-range)
        pct_diff = abs(total_h - stored_h) / stored_h * 100
        print(f"  Grid vs stored: {pct_diff:.2f}%")
        if expected_h > stored_h:
            out_of_range = expected_h - stored_h
            print(f"  Out of range:   {out_of_range:,.1f} hours ({out_of_range/expected_h*100:.1f}%)")
        if pct_diff > 0.1:
            print(f"  WARNING: Grid integral differs from stored total by {pct_diff:.2f}%")
        else:
            print("  ✓ Grid integral matches stored total")


if __name__ == "__main__":
    main()
