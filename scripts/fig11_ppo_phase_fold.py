"""Phase 6.6 — PPO phase folding of QP60 wave events.

Bins QP60 event peak times by SLS5N phase (and SLS5S phase) and plots
the resulting histogram. The published Fig 9 shows the 10.7 h
modulation indirectly via wave-train separation times; this figure
shows the same modulation directly in PPO phase space, which is what
the proposed driver theory (periodic magnetotail flapping at the PPO
period) predicts.
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
    print(f"Total events: {len(df)}")

    by_band = {b: df[df["band"] == b] for b in ("QP30", "QP60", "QP120")}
    bins = np.linspace(0, 360, 25)  # 15° bins

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for col, band in enumerate(("QP30", "QP60", "QP120")):
        sub = by_band[band]
        for row, key in enumerate(("ppo_phase_n_deg", "ppo_phase_s_deg")):
            ax = axes[row, col]
            phases = sub[key].dropna().values
            if len(phases) == 0:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center")
                continue
            counts, edges = np.histogram(phases, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            uniform = len(phases) / (len(bins) - 1)
            ax.bar(centers, counts, width=14, color="#ff6b6b" if row == 0 else "#4ecdc4",
                   alpha=0.8, edgecolor="black", lw=0.3)
            ax.axhline(uniform, color="grey", lw=1, ls="--",
                       label="uniform")
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])
            if row == 1:
                ax.set_xlabel("PPO phase (deg)")
            label = "SLS5N" if row == 0 else "SLS5S"
            ax.set_title(f"{band} — {label}", fontsize=11)
            if col == 0:
                ax.set_ylabel("Event count")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9, loc="upper right")

    fig.suptitle("Figure 11 (Phase 6.6) — QP wave events vs SLS5 PPO phase",
                 fontsize=13)
    fig.tight_layout()
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure11_ppo_phase_fold.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
