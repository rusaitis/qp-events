"""Phase 7.7 — Figure 7 v2 with 5-degree bins, 10h dwell floor, and
bootstrap confidence bands.

Reads ``Output/event_time_grid_v2.zarr`` (built by bin_event_time_v2.py).
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

from qp.events.normalization import (  # noqa: E402
    collapse_to_latitude,
    slice_lt_sector,
)
from qp.plotting.style import use_paper_style  # noqa: E402


LT_SECTORS = [
    (0, 3, "0 h ± 3 h", "midnight"),
    (6, 3, "6 h ± 3 h", "dawn"),
    (12, 3, "12 h ± 3 h", "noon"),
    (18, 3, "18 h ± 3 h", "dusk"),
]

MIN_DWELL_MINUTES = 600.0  # 10 hours — much more conservative than v1's 60 min


def bootstrap_ratio(
    ev_1d: np.ndarray,
    dw_1d: np.ndarray,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap 16th/84th percentile bands for the occurrence ratio.

    Returns (lo, hi) arrays, same length as ev_1d.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(ev_1d)
    # Each latitude bin's event count follows a Poisson process.
    # Resample by drawing from Poisson with lambda = ev_1d
    ratios = np.zeros((n_boot, n), dtype=float)
    for i in range(n_boot):
        ev_resampled = rng.poisson(np.maximum(ev_1d, 0)).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[i] = np.where(
                dw_1d >= MIN_DWELL_MINUTES,
                ev_resampled / dw_1d,
                np.nan,
            )
    lo = np.nanpercentile(ratios, 16, axis=0)
    hi = np.nanpercentile(ratios, 84, axis=0)
    return lo, hi


def main() -> None:
    zarr_path = _PROJECT_ROOT / "Output" / "event_time_grid_v2.zarr"
    if not zarr_path.exists():
        print(f"Grid not found: {zarr_path}")
        print("Run scripts/bin_event_time_v2.py first.")
        return

    ds = xr.open_zarr(zarr_path, consolidated=False)
    lt_centers = ds.local_time.values
    lat_centers = ds.magnetic_latitude.values

    use_paper_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for ax, (center, hw, label, _) in zip(axes.flat, LT_SECTORS):
        for band, color in (
            ("QP30", "#4ecdc4"),
            ("QP60", "#ff6b6b"),
            ("QP120", "#ffd93d"),
            ("total", "white"),
        ):
            ev_var = f"event_time_{band}"
            if ev_var not in ds:
                continue
            ev_3d = ds[ev_var].values
            dwell_3d = ds["event_time_dwell"].values
            ev_2d = slice_lt_sector(ev_3d, lt_centers, center, hw)
            dw_2d = slice_lt_sector(dwell_3d, lt_centers, center, hw)
            ev_1d = collapse_to_latitude(ev_2d)
            dw_1d = collapse_to_latitude(dw_2d)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(
                    dw_1d >= MIN_DWELL_MINUTES, ev_1d / dw_1d, np.nan
                )

            ax.plot(lat_centers, ratio, color=color, lw=1.6, label=band)

            # Bootstrap bands
            lo, hi = bootstrap_ratio(ev_1d, dw_1d)
            ax.fill_between(lat_centers, lo, hi, color=color, alpha=0.15)

        ax.set_title(label, fontsize=12)
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, 0.5)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Magnetic latitude (KSM, offset dipole)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Event time / dwell time")
    axes[0, 0].legend(loc="upper left", fontsize=10, frameon=False)

    fig.suptitle(
        "Figure 7 v2 — QP wave occurrence (5° bins, 10h dwell floor, "
        "bootstrap 16–84%)",
        fontsize=13,
    )
    fig.tight_layout()
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure7_v2_event_dwell_ratio.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
