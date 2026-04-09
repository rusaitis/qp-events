"""Phase 8.8 — KMAG invariant-latitude dwell normalization.

Bins events on the KMAG invariant-latitude × LT grid that matches the
``kmag_inv_lat_*`` variables in ``Output/dwell_grid_cassini_saturn.zarr``
and divides by the KMAG dwell time. This produces a properly normalized
occurrence rate in invariant latitude, matching the paper's Fig 8 layout.

The dipole invariant latitude stored per event is used as an approximation
for the KMAG value (they agree to within ~2° at L > 10).

Also compares three approaches side-by-side for Fig 8:
(a) quality-weighted, (b) quality > 0.3 cut, (c) raw unweighted.

Output:
    ``Output/figures/figure8_phase8_inv_lat.png``
    ``Output/figures/figure8_phase8_comparison.png``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qp.plotting.style import use_paper_style  # noqa: E402

MIN_DWELL_MINUTES = 600.0  # 10 h floor
SIGMA = 2.0  # Gaussian smoothing


def _smooth(arr: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    sm = gaussian_filter(filled, sigma=sigma)
    wt = gaussian_filter(valid.astype(float), sigma=sigma)
    out = np.where(wt > 0, sm / wt, np.nan)
    out[~valid] = np.nan
    return out


def _bin_on_kmag_grid(
    df: pd.DataFrame,
    quality_col: str,
    inv_lat_edges: np.ndarray,
    lt_edges: np.ndarray,
    *,
    quality_weighted: bool = False,
    min_quality: float = 0.0,
) -> dict[str, np.ndarray]:
    """Bin events onto the KMAG (inv_lat, LT) grid."""
    n_lat = len(inv_lat_edges) - 1
    n_lt = len(lt_edges) - 1

    bands = ["QP30", "QP60", "QP120"]
    grids: dict[str, np.ndarray] = {b: np.zeros((n_lat, n_lt)) for b in bands}
    grids["total"] = np.zeros((n_lat, n_lt))

    df_f = df.dropna(subset=["dipole_inv_lat", "local_time", "duration_minutes"])
    if min_quality > 0:
        df_f = df_f[df_f[quality_col].fillna(0) >= min_quality]

    for _, row in df_f.iterrows():
        inv_lat = float(row["dipole_inv_lat"])  # signed, degrees
        lt = float(row["local_time"]) % 24.0
        dur = float(row["duration_minutes"])
        band = str(row["band"]) if pd.notna(row["band"]) else None

        weight = 1.0
        if quality_weighted:
            q = row.get(quality_col)
            if pd.notna(q):
                weight = float(q)

        i_lat = int(np.searchsorted(inv_lat_edges, inv_lat, side="right") - 1)
        i_lt = int(np.searchsorted(lt_edges, lt, side="right") - 1)
        if not (0 <= i_lat < n_lat and 0 <= i_lt < n_lt):
            continue

        contribution = dur * weight
        if band in grids:
            grids[band][i_lat, i_lt] += contribution
        grids["total"][i_lat, i_lt] += contribution

    return grids


def main() -> None:
    # ── Load catalog ──────────────────────────────────────────────────────────
    cat_path = _PROJECT_ROOT / "Output" / "events_qp_v3.parquet"
    if not cat_path.exists():
        cat_path = _PROJECT_ROOT / "Output" / "events_qp_v2.parquet"
    df = pd.read_parquet(cat_path)
    quality_col = "quality_v3" if "quality_v3" in df.columns else "quality"
    print(f"Loaded {len(df)} events from {cat_path.name}")

    # ── Load KMAG dwell ───────────────────────────────────────────────────────
    dwell_zarr = _PROJECT_ROOT / "Output" / "dwell_grid_cassini_saturn.zarr"
    ds = xr.open_zarr(dwell_zarr, consolidated=False)
    kmag_inv_lat = ds.coords["kmag_inv_lat"].values  # degrees, shape (180,)
    lt_coords = ds.coords["local_time"].values        # hours, shape (96,)
    # Use magnetosphere-only dwell as denominator
    dwell_kmag = ds["kmag_inv_lat_magnetosphere"].values  # (180, 96) minutes

    # Build grid edges from centers
    dlat = kmag_inv_lat[1] - kmag_inv_lat[0]
    inv_lat_edges = np.append(
        kmag_inv_lat - dlat / 2, kmag_inv_lat[-1] + dlat / 2,
    )
    dlt = lt_coords[1] - lt_coords[0]
    lt_edges = np.append(lt_coords - dlt / 2, lt_coords[-1] + dlt / 2)

    print(f"KMAG grid: inv_lat {kmag_inv_lat.min():.1f}..{kmag_inv_lat.max():.1f}, "
          f"LT {lt_coords.min():.1f}..{lt_coords.max():.1f}")
    print(f"Dwell (mag-only): {dwell_kmag.sum()/60:.0f} h")

    # ── Bin events ────────────────────────────────────────────────────────────
    grids_uw = _bin_on_kmag_grid(
        df, quality_col, inv_lat_edges, lt_edges, quality_weighted=False,
    )
    grids_w = _bin_on_kmag_grid(
        df, quality_col, inv_lat_edges, lt_edges, quality_weighted=True,
    )
    grids_q03 = _bin_on_kmag_grid(
        df, quality_col, inv_lat_edges, lt_edges,
        quality_weighted=False, min_quality=0.3,
    )
    print("Event grids built.")

    def _occurrence(event_grid: np.ndarray) -> np.ndarray:
        """Compute occurrence rate with dwell floor."""
        rate = np.full_like(event_grid, np.nan, dtype=float)
        valid = dwell_kmag >= MIN_DWELL_MINUTES
        rate[valid] = event_grid[valid] / dwell_kmag[valid]
        rate = np.clip(rate, 0.0, 0.5)
        return _smooth(rate, SIGMA)

    # ── Figure: 3-band × 3-approach comparison ────────────────────────────────
    use_paper_style()
    bands = ["QP30", "QP60", "QP120"]
    approaches = [
        ("unweighted", grids_uw),
        ("q > 0.3 cut", grids_q03),
        ("quality-weighted", grids_w),
    ]

    fig, axes = plt.subplots(
        3, 3, figsize=(15, 12), constrained_layout=True,
    )

    for row_idx, (approach_label, grids) in enumerate(approaches):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            rate = _occurrence(grids[band])
            im = ax.pcolormesh(
                lt_coords, kmag_inv_lat, rate,
                shading="auto", cmap="plasma", vmin=0.0, vmax=0.15,
            )
            ax.set_xlabel("Local time (h)" if row_idx == 2 else "")
            ax.set_ylabel("Inv. lat. (°)" if col_idx == 0 else "")
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_ylim(-90, 90)
            ax.axhline(0, color="white", lw=0.4, ls=":", alpha=0.4)
            ax.axvline(6, color="white", lw=0.3, ls=":", alpha=0.3)
            ax.axvline(12, color="white", lw=0.3, ls=":", alpha=0.3)
            ax.axvline(18, color="white", lw=0.3, ls=":", alpha=0.3)
            if row_idx == 0:
                ax.set_title(band, fontsize=12)
            if col_idx == 2:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Event / dwell")
        axes[row_idx, 0].set_ylabel(approach_label, fontsize=10)

    fig.suptitle(
        "Phase 8.8 — QP occurrence in KMAG invariant latitude × LT\n"
        "(KMAG magnetosphere-only dwell, 10h floor, σ=2 smooth)",
        fontsize=11,
    )
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure8_phase8_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")

    # ── Figure: Final Fig 8 (quality-weighted, 3 bands) ──────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, band in zip(axes2, bands):
        rate = _occurrence(grids_q03[band])
        im = ax.pcolormesh(
            lt_coords, kmag_inv_lat, rate,
            shading="auto", cmap="plasma", vmin=0.0, vmax=0.15,
        )
        ax.set_title(band, fontsize=13)
        ax.set_xlabel("Local time (h)")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylim(-90, 90)
        ax.axhline(0, color="white", lw=0.4, ls=":", alpha=0.4)
        for xv in [6, 12, 18]:
            ax.axvline(xv, color="white", lw=0.3, ls=":", alpha=0.3)

    axes2[0].set_ylabel("KMAG invariant latitude (°)")
    cbar2 = fig2.colorbar(im, ax=axes2, fraction=0.02, pad=0.02)
    cbar2.set_label("Event time / dwell time (q > 0.3)")
    fig2.suptitle(
        "Figure 8 Phase 8 — QP occurrence (KMAG inv. lat, q>0.3, 10h dwell floor)",
        fontsize=11,
    )
    out2 = out_dir / "figure8_phase8_inv_lat.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight",
                 facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
