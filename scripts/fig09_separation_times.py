"""Figure 9: Wave train separation time distribution.

Probability distribution of time intervals between consecutive QP60
wave packet peaks. Median separation = ~10.73 h, matching the PPO
modulation of Saturn's magnetotail flapping (Provan et al. 2011).

Input is the canonical round-8 catalogue
(``Output/events_round8.parquet``), filtered to ``band == 'QP60'``
by default. The catalogue is q_factor-gated, Stokes-validated, and
polarization-pure, so it is strictly better input than the pre-round-8
single-band CWT peak finder this script used to call.

A ``--band-agnostic`` flag widens the filter to a peak-period window
in minutes (default 50–70 min, matching the paper's QP60 definition)
to recover any boundary-edge events that were classified into a
neighbouring QP band but represent the same physical population.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import qp
from qp.events.catalog import WavePacketPeak
from qp.events.wave_packets import (
    compute_separations,
    separation_histogram,
    separation_statistics,
)


def _load_qp60_packets(
    parquet_path: Path,
    *,
    band_agnostic: bool,
    period_min_minutes: float,
    period_max_minutes: float,
) -> list[WavePacketPeak]:
    """Load the round-8 catalogue, filter to QP60, return WavePacketPeaks.

    The :class:`WavePacketPeak` API requires ``peak_time``, ``prominence``,
    ``date_from``, and ``date_to``. ``compute_separations`` only consults
    ``peak_time``; ``date_from``/``date_to`` are filled with ±1 h sentinels
    purely to satisfy the constructor.
    """
    df = pd.read_parquet(parquet_path)
    if band_agnostic:
        mask = (df["period_min"] >= period_min_minutes) & (
            df["period_min"] < period_max_minutes
        )
    else:
        mask = df["band"] == "QP60"
    sub = df.loc[mask].copy()
    sub["peak_time"] = pd.to_datetime(sub["peak_time"])

    one_hour = pd.Timedelta(hours=1)
    packets = [
        WavePacketPeak(
            peak_time=row.peak_time.to_pydatetime(),
            prominence=float(row.q_factor),
            date_from=(row.peak_time - one_hour).to_pydatetime(),
            date_to=(row.peak_time + one_hour).to_pydatetime(),
            band=row.band,
            period_sec=float(row.period_min) * 60.0,
        )
        for row in sub.itertuples()
    ]
    packets.sort(key=lambda p: p.peak_time)
    return packets


def _parse_args() -> argparse.Namespace:
    summary = (__doc__ or "").splitlines()[0] if __doc__ else ""
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=qp.OUTPUT_DIR / "events_round8.parquet",
        help="Round-8 catalogue (default: Output/events_round8.parquet).",
    )
    parser.add_argument(
        "--band-agnostic",
        action="store_true",
        help=(
            "Filter on peak period in [period-min, period-max) min instead of "
            "band == 'QP60'. Recovers boundary-edge events classified as "
            "neighbouring bands."
        ),
    )
    parser.add_argument("--period-min", type=float, default=50.0)
    parser.add_argument("--period-max", type=float, default=70.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/figure9.png"),
        help="Output PNG path (default: output/figure9.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    packets = _load_qp60_packets(
        args.parquet,
        band_agnostic=args.band_agnostic,
        period_min_minutes=args.period_min,
        period_max_minutes=args.period_max,
    )
    label = (
        f"period in [{args.period_min:.0f}, {args.period_max:.0f}) min"
        if args.band_agnostic
        else "band == QP60"
    )
    print(f"Loaded {len(packets)} packets ({label}) from {args.parquet}")

    if len(packets) < 2:
        print("Not enough wave packets to compute separations.")
        return

    seps = compute_separations(packets, max_separation_hours=36.0)
    stats = separation_statistics(seps)
    print(
        f"  count={stats['count']}  median={stats['median']:.2f} h  "
        f"mean={stats['mean']:.2f} h  std={stats['std']:.2f} h"
    )

    bin_width = 1.5  # hours
    centers, _, pdf = separation_histogram(
        seps, bin_width_hours=bin_width, max_hours=36.0
    )

    kde = gaussian_kde(seps, bw_method=0.2)
    x_smooth = np.linspace(0, 36, 500)
    y_smooth = kde(x_smooth)

    plt.style.use("default")
    plt.rcParams.update({"font.size": 16})

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        centers,
        pdf,
        width=bin_width * 0.9,
        color="#f0b87a",
        alpha=0.7,
        edgecolor="#d4944a",
        linewidth=0.5,
    )
    ax.plot(x_smooth, y_smooth, color="#555555", lw=2.5)
    ax.axvline(stats["median"], ls="--", lw=2, color="grey", alpha=0.7)
    ax.text(
        stats["median"] + 0.3,
        ax.get_ylim()[1] * 0.3,
        f"median sep = {stats['median']:.2f} h",
        fontsize=14,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )

    ax.set_xlabel("Separation [h]", fontsize=18)
    ax.set_ylabel("Probability density", fontsize=18)
    ax.set_title(r"Wave Activity Separation in Time ($\tau$)", fontsize=20)
    ax.set_xlim(0, 27)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
