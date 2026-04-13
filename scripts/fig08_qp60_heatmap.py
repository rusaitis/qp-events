"""Figure 8 — 2-D occurrence-rate heatmap (LT × magnetic latitude).

Reads ``Output/event_time_grid_v1.zarr`` and divides the
``event_time_*_lt_mag_lat`` 2-D pre-aggregations by the matching
dwell pre-aggregation. Produces three panels: QP60 (the published
focus), plus QP30 and QP120 as bonus panels not in the original
figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qp.plotting.style import use_paper_style  # noqa: E402


def smooth_2d(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Light Gaussian smoother for the 2D maps."""
    from scipy.ndimage import gaussian_filter
    valid = np.isfinite(arr)
    out = np.where(valid, arr, 0.0)
    out = gaussian_filter(out, sigma=sigma)
    weight = gaussian_filter(valid.astype(float), sigma=sigma)
    out = np.where(weight > 0, out / weight, np.nan)
    out[~valid] = np.nan
    return out


def main(grid_path: Path | None = None) -> None:
    if grid_path is None:
        grid_path = _PROJECT_ROOT / "Output" / "event_time_grid_v5.zarr"
    ds = xr.open_zarr(grid_path, consolidated=False)
    lt_centers = ds.local_time.values
    lat_centers = ds.magnetic_latitude.values

    use_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                              constrained_layout=True)
    bands = ("QP30", "QP60", "QP120")
    for ax, band in zip(axes, bands):
        ev = ds[f"event_time_{band}_lt_mag_lat"].values
        dwell = ds["event_time_dwell_lt_mag_lat"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                dwell >= 60.0,
                ev / dwell,
                np.nan,
            )
        ratio = smooth_2d(ratio, sigma=1.5)
        ratio = np.clip(ratio, 0.0, 0.5)

        # Reorder so LT runs along x and lat along y
        # event_time_*_lt_mag_lat is (mag_lat, local_time)
        im = ax.pcolormesh(
            lt_centers, lat_centers, ratio,
            shading="auto", cmap="plasma", vmin=0.0, vmax=0.4,
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
        "Figure 8 — Dwell-normalized QP wave occurrence (LT × magnetic latitude)",
        fontsize=14,
    )
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure8_qp_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
