"""Phase 8.9.1 — Final Figure 7 (publication-ready).

Compares three approaches side-by-side for each LT sector:
- Unweighted (all 1636 events)
- Quality > 0.3 hard cut (667 events)
- Quality-weighted (Σ q_i × Δt_i)

Uses the v3 event-time zarr grids from ``scripts/bin_event_time_v3.py``.

Output: ``Output/figures/figure7_phase8_final.png``
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

from qp.events.normalization import collapse_to_latitude, slice_lt_sector  # noqa: E402
from qp.plotting.style import use_paper_style  # noqa: E402

LT_SECTORS = [
    (0, 3, "Midnight (0 ± 3 h)"),
    (6, 3, "Dawn (6 ± 3 h)"),
    (12, 3, "Noon (12 ± 3 h)"),
    (18, 3, "Dusk (18 ± 3 h)"),
]

MIN_DWELL_MINUTES = 600.0


def _bootstrap_band(
    ev_1d: np.ndarray, dw_1d: np.ndarray, n: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(42)
    ratios = np.zeros((n, len(ev_1d)))
    for i in range(n):
        ev_r = rng.poisson(np.maximum(ev_1d, 0)).astype(float)
        ratios[i] = np.where(dw_1d >= MIN_DWELL_MINUTES, ev_r / dw_1d, np.nan)
    return np.nanpercentile(ratios, 16, axis=0), np.nanpercentile(ratios, 84, axis=0)


def _compute_sector_ratios(
    ds: xr.Dataset, lt_centers: np.ndarray, lat_centers: np.ndarray,
    band: str, center: float, hw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ev_var = f"event_time_{band}"
    dw_var = "event_time_dwell"
    if ev_var not in ds or dw_var not in ds:
        return None
    ev_3d = ds[ev_var].values
    dw_3d = ds[dw_var].values
    ev_2d = slice_lt_sector(ev_3d, lt_centers, center, hw)
    dw_2d = slice_lt_sector(dw_3d, lt_centers, center, hw)
    ev_1d = collapse_to_latitude(ev_2d)
    dw_1d = collapse_to_latitude(dw_2d)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dw_1d >= MIN_DWELL_MINUTES, ev_1d / dw_1d, np.nan)
    lo, hi = _bootstrap_band(ev_1d, dw_1d)
    return ratio, lo, hi, dw_1d


def main() -> None:
    # Load v3 zarr grids (unweighted, weighted, q>0.3)
    zarr_paths = {
        "unweighted": _PROJECT_ROOT / "Output" / "event_time_grid_v3_unweighted.zarr",
        "q > 0.3": _PROJECT_ROOT / "Output" / "event_time_grid_v3_q03.zarr",
        "quality-weighted": _PROJECT_ROOT / "Output" / "event_time_grid_v3_weighted.zarr",
    }
    # Fall back to v2 if v3 not available
    v2_path = _PROJECT_ROOT / "Output" / "event_time_grid_v2.zarr"

    datasets: dict[str, xr.Dataset] = {}
    for label, path in zarr_paths.items():
        if path.exists():
            datasets[label] = xr.open_zarr(path, consolidated=False)
        elif v2_path.exists():
            print(f"  {path.name} not found — using v2 zarr")
            datasets[label] = xr.open_zarr(v2_path, consolidated=False)

    if not datasets:
        print("No zarr grids found. Run bin_event_time_v3.py first.")
        return

    ds0 = next(iter(datasets.values()))
    lt_centers = ds0.local_time.values
    lat_centers = ds0.magnetic_latitude.values

    use_paper_style()
    colors_band = {
        "QP30": "#4ecdc4",
        "QP60": "#ff6b6b",
        "QP120": "#ffd93d",
        "total": "white",
    }
    ls_approach = {
        "unweighted": "-",
        "q > 0.3": "--",
        "quality-weighted": ":",
    }

    n_sectors = len(LT_SECTORS)
    fig, axes = plt.subplots(1, n_sectors, figsize=(16, 5), sharey=True,
                              constrained_layout=True)

    for ax, (center, hw, sector_label) in zip(axes, LT_SECTORS):
        for approach_label, ds in datasets.items():
            for band, color in colors_band.items():
                result = _compute_sector_ratios(
                    ds, lt_centers, lat_centers, band, center, hw,
                )
                if result is None:
                    continue
                ratio, lo, hi, dw_1d = result
                lw = 1.5 if approach_label == "q > 0.3" else 1.0
                alpha_fill = 0.12 if approach_label == "q > 0.3" else 0.0
                ax.plot(
                    lat_centers, ratio,
                    color=color,
                    lw=lw,
                    ls=ls_approach.get(approach_label, "-"),
                    alpha=0.85,
                    label=f"{band} ({approach_label})" if band != "total" else None,
                )
                if approach_label == "q > 0.3":
                    ax.fill_between(lat_centers, lo, hi,
                                     color=color, alpha=alpha_fill)

        ax.set_title(sector_label, fontsize=11)
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, 0.35)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.grid(alpha=0.25)
        ax.set_xlabel("Magnetic latitude (KSM, offset dipole)")

    axes[0].set_ylabel("Event time / dwell time")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = (
        [Line2D([0], [0], color=c, lw=2, label=b)
         for b, c in colors_band.items() if b != "total"]
        + [Line2D([0], [0], color="grey", ls=ls, lw=1.5, label=lbl)
           for lbl, ls in ls_approach.items()]
    )
    axes[-1].legend(handles=legend_elements, fontsize=7, frameon=False,
                     loc="upper right")

    fig.suptitle(
        "Figure 7 (Phase 8) — QP occurrence vs magnetic latitude\n"
        "(5° bins, 10h dwell floor, bootstrap 16–84% for q > 0.3)",
        fontsize=11,
    )
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure7_phase8_final.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
