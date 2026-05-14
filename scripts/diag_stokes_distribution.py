r"""Diagnostic: distribution of Stokes degree-of-polarization across candidates.

The canonical detector applies a hard cut ``d >= MIN_DEGREE_OF_POLARIZATION = 0.7``
on the wavelet-derived Stokes degree-of-polarization (Eq. 6-7 in the paper).
Section 7A in TASKS.md asks: *why 0.7, and what would the catalogue look like
under a different cut?*

This script re-runs the four-gate detector on a representative subset of
Cassini MFA segments with ``min_stokes_d=0`` so every candidate ridge that
passed the other three gates (CWT σ-mask, Q-factor ≥ 3, MVA-parallel
fraction ≤ 0.5) gets its Stokes-d recorded — including rejects. We then
plot the resulting distribution against the canonical threshold and
tabulate the catalogue size that would result from each candidate cutoff.

Outputs:

* ``Output/diagnostics/stokes_candidates.csv`` — every candidate's
  Stokes-d, band, q_factor, mva_par_frac, and period.
* ``Output/figures/diag_p4_stokes_distribution.png`` — histogram with the
  canonical cut marked and a cumulative survival curve.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import TIMESTAMPED_LOG_FMT, setup_logging

from qp.events.bands import QP_BANDS, get_band
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

log = logging.getLogger("diag_stokes_distribution")

DT_SEC = 60.0
CUTOFFS = (0.5, 0.6, 0.7, 0.8, 0.9)


def candidates_from_segment(
    seg_idx: int,
    times: list[datetime.datetime],
    b_par: np.ndarray,
    b_perp1: np.ndarray,
    b_perp2: np.ndarray,
) -> list[dict]:
    """Run the round-8 detector with the Stokes gate disabled.

    Every ridge that passes the σ-mask, Q-factor, and MVA-transversality
    gates is returned with its Stokes-d value recorded — including those
    that would normally be rejected.
    """
    n = len(times)
    if n < 18 * 60:
        return []

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

    all_peaks = []
    for component, power, mask in (
        (b_perp1, p1, mask1),
        (b_perp2, p2, mask2),
    ):
        peaks = detect_wave_packets_multi(
            data=component,
            times=times,
            dt=DT_SEC,
            bands=list(QP_BANDS.values()),
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
        band_obj = get_band(peak.band)
        in_band = (freq >= band_obj.freq_min_hz) & (freq < band_obj.freq_max_hz)
        if not in_band.any():
            continue
        # Stokes gate intentionally omitted — record d for every candidate.
        d = degree_of_polarization(
            cwt1[in_band, sl].ravel(),
            cwt2[in_band, sl].ravel(),
        )
        out.append(
            {
                "segment_idx": seg_idx,
                "peak_time": peak.peak_time.isoformat(),
                "band": peak.band,
                "period_min": peak.period_sec / 60.0,
                "q_factor": float(q),
                "mva_par_frac": float(par_frac),
                "stokes_d": float(d),
                "passes_canonical": bool(d >= MIN_DEGREE_OF_POLARIZATION),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stride", type=int, default=43)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("Output/diagnostics/stokes_candidates.csv"),
    )
    p.add_argument(
        "--fig",
        type=Path,
        default=Path("Output/figures/diag_p4_stokes_distribution.png"),
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

    all_cands: list[dict] = []
    for k, idx in enumerate(sampled):
        payload = segment_to_payload(idx, arr[idx])
        if payload is None:
            continue
        try:
            cands = candidates_from_segment(
                idx,
                payload.times,
                payload.b_par,
                payload.b_perp1,
                payload.b_perp2,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("seg %d raised: %s", idx, exc)
            continue
        all_cands.extend(cands)
        if (k + 1) % 20 == 0:
            log.info(
                "  %d / %d segments done, %d candidates so far",
                k + 1,
                len(sampled),
                len(all_cands),
            )

    log.info(
        "sweep complete: %d candidates from %d segments",
        len(all_cands),
        len(sampled),
    )

    df = pd.DataFrame(all_cands)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    log.info("wrote candidates CSV → %s", args.csv)

    if df.empty:
        log.warning("no candidates — skipping figure")
        return 0

    # Per-cutoff catalogue size.
    log.info(
        "Stokes-d distribution: min=%.3f median=%.3f max=%.3f",
        df.stokes_d.min(),
        df.stokes_d.median(),
        df.stokes_d.max(),
    )
    table_rows = []
    for c in CUTOFFS:
        kept = int((df.stokes_d >= c).sum())
        pct = 100.0 * kept / len(df)
        table_rows.append((c, kept, len(df) - kept, pct))
        log.info(
            "  d ≥ %.2f: %d kept (%.1f%%)  %d rejected", c, kept, pct, len(df) - kept
        )

    # Plot: histogram + cumulative survival curve.
    fig, (ax_hist, ax_surv) = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        dpi=120,
        constrained_layout=True,
    )

    # Histogram of Stokes d, coloured by canonical pass/fail.
    bins = np.linspace(0.0, 1.0, 41)
    ax_hist.hist(
        df.loc[df.stokes_d >= MIN_DEGREE_OF_POLARIZATION, "stokes_d"],
        bins=bins,
        color="#2bd07b",
        alpha=0.85,
        label="pass (d ≥ 0.7)",
        edgecolor="black",
        linewidth=0.4,
    )
    ax_hist.hist(
        df.loc[df.stokes_d < MIN_DEGREE_OF_POLARIZATION, "stokes_d"],
        bins=bins,
        color="#f29539",
        alpha=0.85,
        label="rejected (d < 0.7)",
        edgecolor="black",
        linewidth=0.4,
    )
    ax_hist.axvline(
        MIN_DEGREE_OF_POLARIZATION,
        color="#dddddd",
        linestyle="--",
        linewidth=1.2,
        label=f"canonical cut d = {MIN_DEGREE_OF_POLARIZATION}",
    )
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Stokes degree-of-polarization d")
    ax_hist.set_ylabel(f"candidates per bin  (N={len(df)})")
    ax_hist.set_title("(a) distribution of d across all candidates")
    ax_hist.legend(loc="upper left", fontsize=9)
    ax_hist.grid(True, alpha=0.15)

    # Cumulative survival curve.
    d_axis = np.linspace(0.0, 1.0, 201)
    n_kept = np.array([(df.stokes_d >= c).sum() for c in d_axis], dtype=float)
    ax_surv.plot(d_axis, n_kept, color="#4dd2ff", linewidth=1.8)
    for c, kept, _, pct in table_rows:
        ax_surv.axvline(c, color="#dddddd", linestyle=":", linewidth=0.6, alpha=0.7)
        ax_surv.scatter([c], [kept], s=30, color="#ffb000", zorder=5)
        ax_surv.annotate(
            f"{c:.1f} → {kept}\n({pct:.0f}%)",
            xy=(c, kept),
            xytext=(c + 0.01, kept + 0.4),
            fontsize=8,
            color="#dddddd",
        )
    ax_surv.axvline(
        MIN_DEGREE_OF_POLARIZATION,
        color="#2bd07b",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    ax_surv.set_xlim(0, 1)
    ax_surv.set_ylim(0, max(n_kept) * 1.05)
    ax_surv.set_xlabel("threshold d_min")
    ax_surv.set_ylabel("candidates with d ≥ d_min")
    ax_surv.set_title("(b) catalogue size vs threshold")
    ax_surv.grid(True, alpha=0.15)

    fig.suptitle(
        "Stokes-d cut sensitivity — candidates surviving Q + MVA + σ-mask",
        fontsize=11,
    )
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=120, bbox_inches="tight")
    log.info("wrote figure → %s", args.fig)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
