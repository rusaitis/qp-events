"""Figure 7 — cumulative QP event time / dwell time vs magnetic latitude
in four local time sectors (0±3 h, 6±3 h, 12±3 h, 18±3 h).

Referee request: use ±3 h LT bins (not the original ±2 h) so the
sectors cover the entire LT range.

Reads ``Output/event_time_grid_v1.zarr`` (built by Phase 4) and
divides ``event_time_*`` by ``event_time_dwell`` per cell.
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
    occurrence_rate,
    slice_lt_sector,
)
from qp.plotting.style import use_paper_style  # noqa: E402


LT_SECTORS = [
    (0, 3, "0 h ± 3 h", "midnight"),
    (6, 3, "6 h ± 3 h", "dawn"),
    (12, 3, "12 h ± 3 h", "noon"),
    (18, 3, "18 h ± 3 h", "dusk"),
]


def main(grid_path: Path | None = None) -> None:
    if grid_path is None:
        grid_path = _PROJECT_ROOT / "Output" / "event_time_grid_v5.zarr"
    ds = xr.open_zarr(grid_path, consolidated=False)
    lt_centers = ds.local_time.values
    lat_centers = ds.magnetic_latitude.values

    use_paper_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for ax, (center, hw, label, _) in zip(axes.flat, LT_SECTORS):
        # Sum over LT inside the sector for each band
        for band, color in (
            ("QP30", "#4ecdc4"),
            ("QP60", "#ff6b6b"),
            ("QP120", "#ffd93d"),
            ("total", "white"),
        ):
            ev_var = f"event_time_{band}"
            ev_3d = ds[ev_var].values
            dwell_3d = ds["event_time_dwell"].values
            ev_2d = slice_lt_sector(ev_3d, lt_centers, center, hw)
            dw_2d = slice_lt_sector(dwell_3d, lt_centers, center, hw)
            ev_1d = collapse_to_latitude(ev_2d)
            dw_1d = collapse_to_latitude(dw_2d)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(dw_1d >= 60.0, ev_1d / dw_1d, np.nan)
            ax.plot(lat_centers, ratio, color=color, lw=1.4, label=band)

        ax.set_title(label, fontsize=12)
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, 0.8)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Magnetic latitude (KSM, offset dipole)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Event time / dwell time")
    axes[0, 0].legend(loc="upper left", fontsize=10, frameon=False)

    fig.suptitle(
        "Figure 7 — Dwell-normalized QP wave occurrence vs magnetic latitude",
        fontsize=14,
    )
    fig.tight_layout()
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure7_event_dwell_ratio.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
