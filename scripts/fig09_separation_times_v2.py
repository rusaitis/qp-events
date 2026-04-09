"""Figure 9 v2 — wave train separation distribution from the Phase 3 catalog.

The legacy ``fig09_separation_times.py`` runs the QP60-only detector
in-process. This v2 reads the parquet catalog produced by Phase 3
and computes separations on the QP60 events directly. The published
median is ~10.73 h (the PPO period); we expect a similar value with
the new detector.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qp.events.wave_packets import (  # noqa: E402
    compute_separations,
    separation_histogram,
    separation_statistics,
)
from qp.events.catalog import WavePacketPeak  # noqa: E402


def main() -> None:
    catalog = _PROJECT_ROOT / "Output" / "events_qp_v1.parquet"
    df = pd.read_parquet(catalog)

    df = df[df["band"] == "QP60"].copy()
    df["peak_time"] = (
        pd.to_datetime(df["date_from"]) + (
            pd.to_datetime(df["date_to"]) - pd.to_datetime(df["date_from"])
        ) / 2
    )
    df.sort_values("peak_time", inplace=True)

    print(f"QP60 events in catalog: {len(df)}")

    packets: list[WavePacketPeak] = []
    for row in df.itertuples(index=False):
        packets.append(
            WavePacketPeak(
                peak_time=row.peak_time.to_pydatetime(),
                prominence=row.snr or 0.0,
                date_from=datetime.datetime.fromisoformat(row.date_from),
                date_to=datetime.datetime.fromisoformat(row.date_to),
            )
        )

    # The published median (10.73 h) is the modal separation. With a
    # max-separation cutoff above ~24 h the long tail of cross-orbit
    # pairs biases the median upward. We compute both the full and
    # the cutoff versions.
    seps_full = compute_separations(packets, max_separation_hours=36.0)
    seps_24h = compute_separations(packets, max_separation_hours=24.0)
    seps_18h = compute_separations(packets, max_separation_hours=18.0)
    print(f"  separations (cutoff 36h): n={len(seps_full):3d} "
          f"median={float(np.median(seps_full)):.2f} h")
    print(f"  separations (cutoff 24h): n={len(seps_24h):3d} "
          f"median={float(np.median(seps_24h)):.2f} h")
    print(f"  separations (cutoff 18h): n={len(seps_18h):3d} "
          f"median={float(np.median(seps_18h)):.2f} h "
          f"(paper: 10.73 h)")
    seps = seps_24h
    stats = separation_statistics(seps)

    bin_width = 1.5
    centers, counts, pdf = separation_histogram(
        seps, bin_width_hours=bin_width, max_hours=36.0,
    )

    plt.style.use("default")
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(centers, pdf, width=bin_width * 0.9, color="#f0b87a",
           alpha=0.7, edgecolor="#d4944a", lw=0.5)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(seps, bw_method=0.2)
        x = np.linspace(0, 36, 500)
        ax.plot(x, kde(x), color="#444444", lw=2.0)
    except Exception:
        pass
    ax.axvline(stats["median"], ls="--", color="grey", lw=2)
    ax.text(stats["median"] + 0.3, ax.get_ylim()[1] * 0.6,
            f"median = {stats['median']:.2f} h",
            color="black",
            bbox=dict(boxstyle="round", fc="white", ec="grey"))
    ax.axvline(10.7, ls=":", color="green", lw=1, alpha=0.7,
               label="PPO period (10.7 h)")
    ax.set_xlabel("Separation between wave packets [h]")
    ax.set_ylabel("Probability density")
    ax.set_title("Figure 9 v2 — QP60 wave train separations (Phase 3 catalog)")
    ax.set_xlim(0, 24)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure9_separations_v2.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
