"""Phase 6.5 — Stokes/ellipticity histograms (Fig 10 v2).

Replaces the published Fig 10's discrete circular/linear/mixed
classification with a continuous ellipticity distribution. Reads
``Output/events_qp_v1.parquet``.

For each band, plots:
- ellipticity histogram (–1 = left circular, 0 = linear, +1 = right circular)
- inclination angle histogram
- polarization fraction histogram
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main() -> None:
    df = pd.read_parquet(_PROJECT_ROOT / "Output" / "events_qp_v1.parquet")
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)

    bins_e = np.linspace(-1, 1, 31)
    bins_i = np.linspace(-90, 90, 31)
    bins_p = np.linspace(0, 1, 26)

    for col, band in enumerate(("QP30", "QP60", "QP120")):
        sub = df[df["band"] == band]
        if len(sub) == 0:
            continue
        e = sub["ellipticity"].dropna().values
        i = sub["inclination_deg"].dropna().values
        p = sub["polarization_fraction"].dropna().values

        ax = axes[0, col]
        ax.hist(e, bins=bins_e, color="#ff6b6b", alpha=0.85)
        ax.set_title(f"{band} ellipticity (n={len(e)})", fontsize=11)
        ax.axvline(0, color="white", lw=0.5, ls="--")
        ax.set_xlim(-1, 1)

        ax = axes[1, col]
        ax.hist(i, bins=bins_i, color="#4ecdc4", alpha=0.85)
        ax.set_title(f"{band} inclination (deg)", fontsize=11)
        ax.set_xlim(-90, 90)

        ax = axes[2, col]
        ax.hist(p, bins=bins_p, color="#ffd93d", alpha=0.85)
        ax.set_title(f"{band} polarization fraction", fontsize=11)
        ax.set_xlim(0, 1)

    fig.suptitle("Figure 10 v2 — Stokes-derived polarization "
                 "(continuous, replaces 3-way classification)",
                 fontsize=13)
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure10_polarization_v2.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
