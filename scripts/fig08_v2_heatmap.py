"""Phase 7.7 — Figure 8 v2 heatmap with coarser cells, stronger smoothing,
and marginal histograms. Also generates the invariant-latitude view.

Reads ``Output/event_time_grid_v2.zarr`` and optionally
``Output/event_time_inv_lat_v2.npz``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qp.plotting.style import use_paper_style  # noqa: E402

MIN_DWELL_MINUTES = 600.0  # 10 hours


def smooth_2d(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian smooth with NaN-aware weighted averaging."""
    valid = np.isfinite(arr)
    out = np.where(valid, arr, 0.0)
    out = gaussian_filter(out, sigma=sigma)
    weight = gaussian_filter(valid.astype(float), sigma=sigma)
    out = np.where(weight > 0, out / weight, np.nan)
    out[~valid] = np.nan
    return out


def plot_heatmap_with_marginals(
    ratio: np.ndarray,
    lat_centers: np.ndarray,
    lt_centers: np.ndarray,
    band: str,
    ax_main,
    ax_top,
    ax_right,
    vmax: float = 0.3,
):
    """Plot a heatmap with top and right marginal histograms."""
    im = ax_main.pcolormesh(
        lt_centers, lat_centers, ratio,
        shading="auto", cmap="plasma", vmin=0.0, vmax=vmax,
    )
    ax_main.set_xlabel("Local time (h)")
    ax_main.set_xlim(0, 24)
    ax_main.set_xticks([0, 6, 12, 18, 24])
    ax_main.axvline(6, color="white", lw=0.3, ls=":", alpha=0.4)
    ax_main.axvline(12, color="white", lw=0.3, ls=":", alpha=0.4)
    ax_main.axvline(18, color="white", lw=0.3, ls=":", alpha=0.4)

    # Top marginal: mean over latitude for each LT
    lt_marginal = np.nanmean(ratio, axis=0)
    ax_top.bar(lt_centers, lt_marginal, width=lt_centers[1] - lt_centers[0],
               color="#ff6b6b", alpha=0.7, edgecolor="none")
    ax_top.set_xlim(0, 24)
    ax_top.set_title(band, fontsize=13)
    ax_top.tick_params(labelbottom=False)

    # Right marginal: mean over LT for each latitude
    lat_marginal = np.nanmean(ratio, axis=1)
    ax_right.barh(lat_centers, lat_marginal,
                   height=lat_centers[1] - lat_centers[0],
                   color="#4ecdc4", alpha=0.7, edgecolor="none")
    ax_right.set_ylim(lat_centers[0], lat_centers[-1])
    ax_right.tick_params(labelleft=False)

    return im


def main() -> None:
    zarr_path = _PROJECT_ROOT / "Output" / "event_time_grid_v2.zarr"
    if not zarr_path.exists():
        print(f"Grid not found: {zarr_path}")
        return

    ds = xr.open_zarr(zarr_path, consolidated=False)
    lt_centers = ds.local_time.values
    lat_centers = ds.magnetic_latitude.values

    use_paper_style()

    bands = ("QP30", "QP60", "QP120")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, band in zip(axes, bands):
        ev_key = f"event_time_{band}_lt_mag_lat"
        dwell_key = "event_time_dwell_lt_mag_lat"
        if ev_key not in ds or dwell_key not in ds:
            continue

        ev = ds[ev_key].values
        dwell = ds[dwell_key].values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                dwell >= MIN_DWELL_MINUTES,
                ev / dwell,
                np.nan,
            )
        ratio = smooth_2d(ratio, sigma=2.0)
        ratio = np.clip(ratio, 0.0, 0.3)

        im = ax.pcolormesh(
            lt_centers, lat_centers, ratio,
            shading="auto", cmap="plasma", vmin=0.0, vmax=0.3,
        )
        ax.set_title(f"{band}", fontsize=14)
        ax.set_xlabel("Local time (h)")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.axvline(6, color="white", lw=0.4, ls=":", alpha=0.4)
        ax.axvline(12, color="white", lw=0.4, ls=":", alpha=0.4)
        ax.axvline(18, color="white", lw=0.4, ls=":", alpha=0.4)

    axes[0].set_ylabel("Magnetic latitude (KSM, offset dipole)")
    axes[0].set_ylim(-90, 90)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Event time / dwell time")
    fig.suptitle(
        "Figure 8 v2 — QP wave occurrence (5° bins, 10h dwell floor, σ=2)",
        fontsize=13,
    )
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure8_v2_qp_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")

    # Invariant latitude view
    inv_path = _PROJECT_ROOT / "Output" / "event_time_inv_lat_v2.npz"
    if inv_path.exists():
        _plot_inv_lat_heatmap(inv_path)


def _plot_inv_lat_heatmap(npz_path: Path) -> None:
    """Figure 8b — invariant-latitude × LT heatmap."""
    data = np.load(npz_path)
    inv_lat = data["inv_lat_centers"]
    lt = data["lt_centers"]

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, band in zip(axes, ("QP30", "QP60", "QP120")):
        ev = data[f"event_time_{band}"]
        total = data["event_time_total"]
        # Simple normalization: fraction of total event time per cell
        # (since we don't have a dwell grid in inv_lat coords, show
        # raw event time normalized to the maximum for visual comparison)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(ev.sum() > 0, ev / ev.sum() * 100, 0.0)
        norm = smooth_2d(norm, sigma=1.5)

        ax.pcolormesh(lt, inv_lat, norm, shading="auto", cmap="plasma",
                       vmin=0)
        ax.set_title(band, fontsize=13)
        ax.set_xlabel("Local time (h)")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])

    axes[0].set_ylabel("Dipole invariant latitude (°)")
    fig.suptitle(
        "Figure 8b — QP event time in invariant latitude × LT",
        fontsize=13,
    )
    out_dir = npz_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure8b_inv_lat_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
