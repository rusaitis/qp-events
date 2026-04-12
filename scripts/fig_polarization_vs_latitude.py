"""Phase 10.1 — Ellipticity vs magnetic latitude.

Bins the Stokes-derived ellipticity by magnetic latitude (5° bins).

Physics predictions:
- Toroidal even-mode standing FLR: polarization nearly uniform in latitude
  (azimuthal mode; linear at all latitudes)
- Poloidal standing FLR: ellipticity sign reverses between hemispheres
- Travelling Alfvén wave: no latitude trend

Output: ``Output/figures/figure_polarization_vs_latitude.png``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qp.plotting.style import use_paper_style  # noqa: E402

BAND_COLORS = {"QP30": "#4ecdc4", "QP60": "#ff6b6b", "QP120": "#ffd93d"}


def main() -> None:
    cat_path = _PROJECT_ROOT / "Output" / "events_qp_v4.parquet"
    if not cat_path.exists():
        cat_path = _PROJECT_ROOT / "Output" / "events_qp_v3.parquet"
    df = pd.read_parquet(cat_path)
    quality_col = "quality_v3" if "quality_v3" in df.columns else "quality"
    print(f"Loaded {len(df)} events")

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, constrained_layout=True)

    lat_edges = np.arange(-90, 91, 5)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    for ax, band in zip(axes, ["QP30", "QP60", "QP120"]):
        sub = df[
            (df.band == band)
            & df[quality_col].fillna(0).gt(0.3)
            & df["ellipticity"].notna()
            & df["mag_lat"].notna()
        ]
        print(f"{band}: {len(sub)} events with ellipticity and mag_lat")

        # Bin by latitude
        bin_idx = np.digitize(sub["mag_lat"].values, lat_edges) - 1
        medians = np.full(len(lat_centers), np.nan)
        iqrs = np.full(len(lat_centers), np.nan)
        ns = np.zeros(len(lat_centers), dtype=int)

        for i in range(len(lat_centers)):
            mask = bin_idx == i
            vals = sub["ellipticity"].values[mask]
            vals = vals[np.isfinite(vals)]
            ns[i] = len(vals)
            if len(vals) >= 3:
                medians[i] = np.median(vals)
                q25, q75 = np.percentile(vals, [25, 75])
                iqrs[i] = (q75 - q25) / 2.0

        valid = ns >= 3
        ax.fill_between(
            lat_centers[valid],
            medians[valid] - iqrs[valid],
            medians[valid] + iqrs[valid],
            color=BAND_COLORS[band], alpha=0.2,
        )
        ax.plot(lat_centers[valid], medians[valid],
                 color=BAND_COLORS[band], lw=2, marker="o", ms=4, label="median ± IQR/2")

        # Reference lines
        ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.6,
                    label="linear (0)")
        ax.axhline(1, color="grey", lw=0.5, ls=":", alpha=0.5, label="|circular|")
        ax.axhline(-1, color="grey", lw=0.5, ls=":", alpha=0.5)
        ax.axvline(0, color="grey", lw=0.5, ls=":", alpha=0.4)

        # Spearman correlation test
        valid_both = sub["mag_lat"].notna() & sub["ellipticity"].notna()
        lats = sub.loc[valid_both, "mag_lat"].values
        ells = sub.loc[valid_both, "ellipticity"].values
        if len(lats) > 5:
            r, pval = stats.spearmanr(lats, ells)
            print(f"  Spearman r={r:.3f}  p={pval:.3f}")
            ax.set_title(
                f"{band} (n={len(sub)})\nSpearman r={r:.2f}, p={pval:.3f}",
                fontsize=10,
            )
        else:
            ax.set_title(f"{band} (n={len(sub)})", fontsize=10)

        ax.set_xlabel("Magnetic latitude (°)", fontsize=10)
        ax.set_xlim(-80, 80)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)

    axes[0].set_ylabel("Stokes ellipticity\n(+1 right-circ, 0 linear, -1 left-circ)", fontsize=9)

    fig.text(
        0.5, 0.01,
        "Standing poloidal FLR: sign reversal expected at equator | "
        "Toroidal/travelling: no trend",
        fontsize=8, ha="center", color="lightgrey",
    )
    fig.suptitle(
        "Phase 10.1 — Ellipticity vs magnetic latitude (q > 0.3 events)",
        fontsize=11,
    )

    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure_polarization_vs_latitude.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
