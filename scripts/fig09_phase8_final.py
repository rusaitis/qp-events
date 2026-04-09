"""Phase 8.9.3 — Final Figure 9: separation-time distribution.

Uses quality-filtered QP60 events (q > 0.3) from the v3 catalog.
Marks the median and the PPO period (10.7 h).

Output: ``Output/figures/figure9_phase8_final.png``
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

from qp.events.wave_packets import compute_separations  # noqa: E402
from qp.plotting.style import use_paper_style  # noqa: E402

PPO_PERIOD_H = 10.73  # Saturn PPO period


def main() -> None:
    cat_path = _PROJECT_ROOT / "Output" / "events_qp_v3.parquet"
    if not cat_path.exists():
        cat_path = _PROJECT_ROOT / "Output" / "events_qp_v2.parquet"
    df = pd.read_parquet(cat_path)
    quality_col = "quality_v3" if "quality_v3" in df.columns else "quality"

    # Quality > 0.3 filter for each band
    df_q = df[df[quality_col].fillna(0) > 0.3].copy()
    print(f"Events with q>0.3: {len(df_q)} of {len(df)}")
    for band in ["QP30", "QP60", "QP120"]:
        b = df_q[df_q.band == band]
        print(f"  {band}: {len(b)} events")

    use_paper_style()
    colors = {"QP30": "#4ecdc4", "QP60": "#ff6b6b", "QP120": "#ffd93d"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, band in zip(axes, ["QP30", "QP60", "QP120"]):
        b = df_q[df_q.band == band].copy()
        b = b.sort_values("date_from")

        if len(b) < 2:
            ax.set_title(f"{band} (too few events)")
            continue

        # Compute separations between consecutive event peaks
        import datetime
        peak_times = [
            datetime.datetime.fromisoformat(str(f))
            + (datetime.datetime.fromisoformat(str(t))
               - datetime.datetime.fromisoformat(str(f))) / 2
            for f, t in zip(b["date_from"], b["date_to"])
        ]

        seps_h = []
        for i in range(1, len(peak_times)):
            dt = (peak_times[i] - peak_times[i - 1]).total_seconds() / 3600.0
            if 0 < dt <= 24.0:  # 24-h cutoff
                seps_h.append(dt)

        if len(seps_h) == 0:
            ax.set_title(f"{band} (no separations)")
            continue

        seps_h = np.array(seps_h)
        median = np.median(seps_h)
        print(f"  {band}: {len(seps_h)} separations, median={median:.2f} h")

        bins = np.linspace(0, 24, 25)
        ax.hist(seps_h, bins=bins, color=colors[band], alpha=0.8,
                 density=True, edgecolor="none")
        ax.axvline(median, color="white", lw=1.5, ls="--",
                    label=f"median={median:.1f} h")
        ax.axvline(PPO_PERIOD_H, color="#ffd93d", lw=1.5, ls=":",
                    label=f"PPO={PPO_PERIOD_H} h")

        ax.set_xlabel("Time between events (h)")
        ax.set_ylabel("Probability density")
        ax.set_title(f"{band} (q>0.3, n={len(seps_h)} seps)", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.set_xlim(0, 24)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Figure 9 (Phase 8) — QP wave packet separation times (q > 0.3, 24h max)",
        fontsize=11,
    )
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure9_phase8_final.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
