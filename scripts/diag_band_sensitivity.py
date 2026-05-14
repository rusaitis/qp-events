r"""Diagnostic: sensitivity of the round-8 detector to QP band edges.

The canonical band scheme leaves 5-minute and 10-minute gaps between the
three QP bands (40–45 min between QP30/QP60; 80–90 min between
QP60/QP120) and rejects everything below 10 min and above 12 h. Section
7A in TASKS.md asks: are these gaps physically motivated, or are we
filtering out real signal that should be reported?

This script re-runs the four-gate detector on a representative subset of
Cassini MFA segments with the band classifier replaced by a **contiguous,
extended** scheme spanning 10–360 min in five touching bins:

    BELOW_QP30   [10,  20) min
    QP30_E       [20,  40) min   (same as canonical QP30)
    QP60_E       [40,  80) min   (extends LEFT through the 40–45 gap)
    QP120_E      [80, 150) min   (extends LEFT through the 80–90 gap)
    ABOVE_QP120 [150, 360) min

Each surviving event records its peak period; we then bin those periods
back into the canonical-band zones, the two gap zones, and the two
"extended" zones to count how much signal each region contains. Results
are saved as:

* ``Output/diagnostics/band_sensitivity_events.csv`` — every event from
  the sweep with its period, gates, and bin label.
* ``Output/figures/diag_p2_band_gap_sensitivity.png`` — period histogram
  with the canonical band edges and gap shading overlaid.

The detector logic is replicated inline (rather than calling
``detect_round8``) so the band scheme can be parameterised without
modifying production code.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from _common import TIMESTAMPED_LOG_FMT, setup_logging

from qp.events.bands import Band
from qp.events.detector import (
    MAX_MVA_PARALLEL_FRACTION,
    MIN_DEGREE_OF_POLARIZATION,
    MIN_Q_FACTOR,
    SEGMENT_FWER_ALPHA,
    bonferroni_n_sigma_for_cwt,
    detect_wave_packets_multi,
)
from qp.events.sweep_loader import (
    load_segments,
    segment_central_window,
    segment_to_payload,
)
from qp.events.threshold import wavelet_sigma_mask
from qp.io import register_legacy_pickle_stubs
from qp.signal.polarization import (
    degree_of_polarization,
    mva_major_axis_parallel_fraction,
)
from qp.signal.wavelet import morlet_cwt

register_legacy_pickle_stubs()

log = logging.getLogger("diag_band_sensitivity")

DT_SEC = 60.0
_MIN = 60.0  # seconds per minute

# Extended, contiguous bands covering 10–360 min in five touching bins.
EXTENDED_BANDS: list[Band] = [
    Band(
        name="BELOW_QP30",
        period_min_sec=10 * _MIN,
        period_max_sec=20 * _MIN,
        period_centroid_sec=14 * _MIN,
    ),
    Band(
        name="QP30_E",
        period_min_sec=20 * _MIN,
        period_max_sec=40 * _MIN,
        period_centroid_sec=30 * _MIN,
    ),
    Band(
        name="QP60_E",
        period_min_sec=40 * _MIN,
        period_max_sec=80 * _MIN,
        period_centroid_sec=60 * _MIN,
    ),
    Band(
        name="QP120_E",
        period_min_sec=80 * _MIN,
        period_max_sec=150 * _MIN,
        period_centroid_sec=120 * _MIN,
    ),
    Band(
        name="ABOVE_QP120",
        period_min_sec=150 * _MIN,
        period_max_sec=360 * _MIN,
        period_centroid_sec=240 * _MIN,
    ),
]


def _zone(period_min: float) -> str:
    """Classify a period into a canonical-vs-extended zone label."""
    if 10 <= period_min < 20:
        return "below_QP30 [10,20)"
    if 20 <= period_min < 40:
        return "QP30 [20,40)"
    if 40 <= period_min < 45:
        return "gap1 [40,45)"
    if 45 <= period_min < 80:
        return "QP60 [45,80)"
    if 80 <= period_min < 90:
        return "gap2 [80,90)"
    if 90 <= period_min < 150:
        return "QP120 [90,150)"
    if 150 <= period_min < 360:
        return "above_QP120 [150,360)"
    return "other"


def detect_extended(
    seg_idx: int,
    times: list[datetime.datetime],
    b_par: np.ndarray,
    b_perp1: np.ndarray,
    b_perp2: np.ndarray,
) -> list[dict]:
    """Re-implementation of ``detect_round8`` with the extended bands.

    Mirrors ``qp.events.detector.detect_round8`` (the four gates: σ-mask,
    Q-factor, MVA-transversality, Stokes-d) but passes ``EXTENDED_BANDS``
    to the ridge extractor so detections inside the canonical gaps are
    not silently discarded.
    """
    n = len(times)
    if n < 18 * 60:
        return []

    # CWT of all three components, identical to detect_round8.
    freq, _, cwt_par = morlet_cwt(b_par, dt=DT_SEC, n_freqs=300)
    _, _, cwt1 = morlet_cwt(b_perp1, dt=DT_SEC, n_freqs=300)
    _, _, cwt2 = morlet_cwt(b_perp2, dt=DT_SEC, n_freqs=300)
    p1 = np.abs(cwt1)
    p2 = np.abs(cwt2)

    n_sigma = bonferroni_n_sigma_for_cwt(
        p1.shape[1],
        DT_SEC,
        freq,
        alpha=SEGMENT_FWER_ALPHA,
    )
    mask1 = wavelet_sigma_mask(p1, freq, n_sigma=n_sigma)
    mask2 = wavelet_sigma_mask(p2, freq, n_sigma=n_sigma)

    # Per-component ridge extraction in the extended band scheme.
    all_peaks = []
    for component, power, mask in (
        (b_perp1, p1, mask1),
        (b_perp2, p2, mask2),
    ):
        peaks = detect_wave_packets_multi(
            data=component,
            times=times,
            dt=DT_SEC,
            bands=EXTENDED_BANDS,
            cwt_freq=freq,
            cwt_power=power,
            threshold_mask=mask,
            min_duration_hours=2.0,
            min_pixels=10,
        )
        all_peaks.extend(peaks)

    # Same-band dedup within 2 h (mirrors detector.py).
    all_peaks.sort(key=lambda p: p.peak_time)
    merged = []
    last_by_band: dict = {}
    for peak in all_peaks:
        prev = last_by_band.get(peak.band)
        if prev is not None:
            if abs((peak.peak_time - prev.peak_time).total_seconds()) < 7200:
                continue
        merged.append(peak)
        last_by_band[peak.band] = peak

    # Build a band-name → Band lookup for the polarization in-band window.
    band_by_name = {b.name: b for b in EXTENDED_BANDS}

    # Per-detection physical gates (mirrors detector.py:732-783).
    epoch = times[0]
    central_start, central_end = segment_central_window(times)
    n_time = cwt1.shape[1]
    out = []
    for peak in merged:
        if peak.period_sec is None or peak.period_sec <= 0 or peak.band is None:
            continue
        if not (central_start <= peak.peak_time < central_end):
            continue
        q = peak.q_factor
        if q is None or q < MIN_Q_FACTOR:
            continue
        i_start = max(
            0,
            int(np.floor((peak.date_from - epoch).total_seconds() / DT_SEC)),
        )
        i_end = min(
            n_time - 1,
            int(np.ceil((peak.date_to - epoch).total_seconds() / DT_SEC)),
        )
        if i_end <= i_start:
            continue
        i_freq_peak = int(np.argmin(np.abs(freq - 1.0 / peak.period_sec)))
        sl = slice(i_start, i_end + 1)
        field_bp = np.column_stack(
            [
                np.real(cwt_par[i_freq_peak, sl]),
                np.real(cwt1[i_freq_peak, sl]),
                np.real(cwt2[i_freq_peak, sl]),
            ]
        )
        par_frac = mva_major_axis_parallel_fraction(field_bp, par_axis=0)
        if par_frac > MAX_MVA_PARALLEL_FRACTION:
            continue
        band_obj = band_by_name.get(peak.band)
        if band_obj is None:
            continue
        in_band = (freq >= band_obj.freq_min_hz) & (freq < band_obj.freq_max_hz)
        if not in_band.any():
            continue
        c1w = cwt1[in_band, sl]
        c2w = cwt2[in_band, sl]
        d = degree_of_polarization(c1w.ravel(), c2w.ravel())
        if d < MIN_DEGREE_OF_POLARIZATION:
            continue

        out.append(
            {
                "segment_idx": seg_idx,
                "peak_time": peak.peak_time.isoformat(),
                "date_from": peak.date_from.isoformat(),
                "date_to": peak.date_to.isoformat(),
                "band_extended": peak.band,
                "period_min": peak.period_sec / 60.0,
                "q_factor": float(q),
                "mva_par_frac": float(par_frac),
                "stokes_d": float(d),
                "zone": _zone(peak.period_sec / 60.0),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stride",
        type=int,
        default=43,
        help="Sample 1 segment per N (default 43 ≈ 100 segments / 4278).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("Output/diagnostics/band_sensitivity_events.csv"),
    )
    p.add_argument(
        "--fig",
        type=Path,
        default=Path("Output/figures/diag_p2_band_gap_sensitivity.png"),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of segments processed (smoke tests).",
    )
    args = p.parse_args(argv)

    setup_logging(fmt=TIMESTAMPED_LOG_FMT)

    arr, keep_idx = load_segments()
    sampled = keep_idx[:: args.stride]
    if args.limit:
        sampled = sampled[: args.limit]
    log.info(
        "processing %d / %d segments (stride %d)",
        len(sampled),
        len(keep_idx),
        args.stride,
    )

    all_events: list[dict] = []
    for k, idx in enumerate(sampled):
        payload = segment_to_payload(idx, arr[idx])
        if payload is None:
            continue
        try:
            events = detect_extended(
                idx,
                payload.times,
                payload.b_par,
                payload.b_perp1,
                payload.b_perp2,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("seg %d raised: %s", idx, exc)
            continue
        all_events.extend(events)
        if (k + 1) % 20 == 0:
            log.info(
                "  %d / %d segments done, %d events so far",
                k + 1,
                len(sampled),
                len(all_events),
            )

    log.info(
        "sweep complete: %d events from %d segments", len(all_events), len(sampled)
    )

    df = pd.DataFrame(all_events)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    log.info("wrote events CSV → %s", args.csv)

    if df.empty:
        log.warning("no events — skipping figure")
        return 0

    # Zone tallies — both the absolute count and the rate per segment.
    zone_counts = df["zone"].value_counts().sort_index()
    log.info("zone counts:\n%s", zone_counts.to_string())

    # Histogram + figure.
    use_paper_style = True
    if use_paper_style:
        try:
            from qp.plotting.style import use_paper_style as _ups

            _ups()
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120, constrained_layout=True)
    # Bin edges aligned with the canonical band edges and gap zones so
    # the histogram visually respects the gap shading. 25 log bins from
    # 10 to 360 min is fine for ~100-event samples.
    bins = np.logspace(np.log10(10), np.log10(360), 30)
    ax.hist(
        df["period_min"],
        bins=bins,
        color="#4dd2ff",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )

    # Canonical band edges in min.
    canonical_edges = {
        "QP30": (20, 40),
        "QP60": (45, 80),
        "QP120": (90, 150),
    }
    band_color = {"QP30": "#80c0ff", "QP60": "#ffb000", "QP120": "#f06090"}
    for name, (lo, hi) in canonical_edges.items():
        ax.axvspan(lo, hi, alpha=0.10, color=band_color[name], zorder=0)
        ax.text(
            (lo * hi) ** 0.5,
            ax.get_ylim()[1] * 0.95,
            name,
            color=band_color[name],
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
    # Gap shading.
    for lo, hi in ((40, 45), (80, 90)):
        ax.axvspan(
            lo, hi, alpha=0.40, facecolor="#444", zorder=0, hatch="//", edgecolor="#888"
        )
    # Above / below shading.
    ax.axvspan(10, 20, alpha=0.20, color="#888", zorder=0)
    ax.axvspan(150, 360, alpha=0.20, color="#888", zorder=0)

    ax.set_xscale("log")
    ax.set_xlim(10, 360)
    ax.set_xlabel("peak period [min]")
    ax.set_ylabel(f"events per bin  (N={len(df)}, {len(sampled)} segments)")
    ax.set_title("Band-edge sensitivity sweep — extended scheme [10–360 min]")
    ax.set_xticks([10, 20, 30, 45, 60, 90, 120, 150, 240, 360])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.15)

    # Inset summary text.
    summary_lines = ["zone               count"]
    for zone in (
        "below_QP30 [10,20)",
        "QP30 [20,40)",
        "gap1 [40,45)",
        "QP60 [45,80)",
        "gap2 [80,90)",
        "QP120 [90,150)",
        "above_QP120 [150,360)",
    ):
        n = int(zone_counts.get(zone, 0))
        summary_lines.append(f"{zone:<20s} {n:>4d}")
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        family="monospace",
        bbox=dict(
            facecolor="white", alpha=0.85, edgecolor="#888", boxstyle="round,pad=0.4"
        ),
    )

    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=120, bbox_inches="tight")
    log.info("wrote figure → %s", args.fig)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
