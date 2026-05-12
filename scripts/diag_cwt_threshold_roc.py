r"""Synthetic-injection ROC sweep over the four CWT threshold methods.

Take a sample of "clean" 36-h MFA segments (no events flagged by the
current detector at default settings), inject a single Morlet-shaped
wave packet of known band, amplitude, and centre time into each, then
run all four threshold gates and measure:

- **Recall** — fraction of cells inside the
  ``(period_band × t_centre ± Δt)`` ground-truth rectangle that the
  method flags. The "true" rectangle is sized to the wave packet's
  half-width.
- **Background rate** — fraction of cells *outside* the rectangle that
  the method flags. A method that flags every cell trivially has
  recall=1 but background rate ≈ method's nominal false-positive
  rate.

Sweep over a log-uniform grid of amplitudes (0.1-5 nT) and over all
four QP bands. One curve per (method, band) is plotted as recall and
background-rate vs amplitude.

Pooled-archive comparison is skipped if ``Output/bg_archive.zarr``
does not exist.

Usage
-----
``uv run python scripts/diag_cwt_threshold_roc.py
  --n-segments 12 --seed 0
  --output Output/figures/cwt_threshold_roc.png``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qp
from qp.events.bands import QP_BAND_COLORS, QP_BANDS, freq_to_period
from qp.events.detector import SEGMENT_FWER_ALPHA, bonferroni_n_sigma_for_cwt
from qp.events.sweep_loader import (
    load_segments,
    region_at_peak_from_info,
    segment_to_payload,
)
from qp.events.threshold import wavelet_sigma_mask
from qp.events.threshold_diag import (
    BGArchive,
    coi_mask,
    fdr_chi2_mask,
    pooled_archive_mask,
    torrence_compo_chi2_mask,
)
from qp.io import legacy_pickle
from qp.plotting.style import use_paper_style
from qp.signal.wavelet import morlet_cwt

log = logging.getLogger("diag.cwt_roc")

_DT_SEC: float = 60.0
_N_FREQS: int = 300
_OMEGA0: float = 10.0
_INJECTION_HALF_WIDTH_HOURS: float = 4.0  # ~ 8-h Gaussian envelope
_AMPLITUDES_NT: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.5, 5.0)

#: Threshold-parameter grids for the like-for-like fixed-FPR sweep.
#: Each method's threshold knob has a different name and units; the
#: grids are chosen so the resulting bg-rate range straddles the
#: canonical 1-2 % FPR target with enough samples to interpolate.
_ALPHA_GRIDS: dict[str, tuple[float, ...]] = {
    # MAD-row: n_sigma. Bonferroni σ for the standard ω₀=10, n_freq=300
    # search volume is ≈ 4.6 — anchor the centre of the grid there.
    "mad_row": (2.5, 3.0, 3.5, 4.0, 4.6, 5.0, 5.5, 6.0, 7.0),
    # T&C χ²(2): per-pixel α. Spans uncorrected 1 % down through
    # Bonferroni-corrected territory. The very small α values matter
    # because below ~2 % bg-rate the curve is dominated by Morlet
    # spectral leakage of the injection — pushing α further changes
    # the threshold but barely moves bg-rate.
    "tc_chi2": (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-14, 1e-20),
    # FDR (BH-Y): target q-value. Same conservatism argument applies.
    "fdr_chi2": (3e-2, 1e-2, 1e-3, 1e-5, 1e-7, 1e-11, 1e-15),
    # Pooled archive: n_sigma against the cross-segment median+MAD.
    # Wider grid because the pooled MAD is tighter than per-segment.
    "pooled": (3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0),
}


# --------------------------------------------------------------------- #
# Injection                                                             #
# --------------------------------------------------------------------- #


def _inject_packet(
    b_perp1: np.ndarray,
    b_perp2: np.ndarray,
    *,
    period_sec: float,
    centre_idx: int,
    half_width_sec: float,
    amplitude: float,
    dt: float = _DT_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Add a Gaussian-windowed circularly-polarised packet to a segment.

    Right-handed (b_perp2 lags b_perp1 by π/2). Returns the contaminated
    components — the originals are not mutated.
    """
    n = len(b_perp1)
    t = (np.arange(n) - centre_idx) * dt
    envelope = np.exp(-((t / half_width_sec) ** 2))
    omega = 2.0 * np.pi / period_sec
    s1 = amplitude * envelope * np.sin(omega * t)
    s2 = amplitude * envelope * (-np.cos(omega * t))
    return b_perp1 + s1, b_perp2 + s2


def _injection_rectangle(
    freq: np.ndarray,
    n_time: int,
    *,
    period_sec: float,
    centre_idx: int,
    half_width_sec: float,
    dt: float = _DT_SEC,
) -> np.ndarray:
    r"""Boolean (n_freq, n_time) rectangle that contains the injected packet.

    Period range = ±10 % of the injection period (wide enough to cover
    Morlet ringing in scale), time range = ±2·σ of the Gaussian
    envelope. Cells outside this rectangle are treated as background
    for the FP-rate calculation.
    """
    periods = freq_to_period(freq)
    in_band = (periods >= period_sec / 1.10) & (periods <= period_sec * 1.10)
    half_width_samples = int(round(2.0 * half_width_sec / dt))
    t_lo = max(0, centre_idx - half_width_samples)
    t_hi = min(n_time, centre_idx + half_width_samples + 1)
    rect = np.zeros((len(freq), n_time), dtype=bool)
    rect[np.ix_(in_band, np.arange(t_lo, t_hi))] = True
    return rect


# --------------------------------------------------------------------- #
# Mask runner (mirrors the comparison script)                           #
# --------------------------------------------------------------------- #


def _run_all_methods_alpha_sweep(
    b_perp1: np.ndarray,
    b_perp2: np.ndarray,
    *,
    archive: BGArchive | None,
    region: str,
) -> dict[tuple[str, float], np.ndarray]:
    """Per-method per-α masks for the like-for-like fixed-FPR ROC.

    Returns a dict keyed by ``(method_name, alpha)`` so the caller can
    sweep both axes uniformly. The CWT is computed once and reused
    across α; mask internals re-run their per-row stat estimation each
    call, which is fast relative to the CWT.
    """
    freq, _, cwt1 = morlet_cwt(
        b_perp1,
        dt=_DT_SEC,
        omega0=_OMEGA0,
        n_freqs=_N_FREQS,
    )
    _, _, cwt2 = morlet_cwt(
        b_perp2,
        dt=_DT_SEC,
        omega0=_OMEGA0,
        n_freqs=_N_FREQS,
    )
    amp1, amp2 = np.abs(cwt1), np.abs(cwt2)
    n_time = amp1.shape[1]
    coi = coi_mask(freq, n_time, dt=_DT_SEC, omega0=_OMEGA0)

    def _join(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        return (m1 | m2) & coi

    out: dict[tuple[str, float], np.ndarray] = {}
    for n_sigma in _ALPHA_GRIDS["mad_row"]:
        out[("mad_row", n_sigma)] = _join(
            wavelet_sigma_mask(amp1, freq, n_sigma=n_sigma),
            wavelet_sigma_mask(amp2, freq, n_sigma=n_sigma),
        )
    for alpha in _ALPHA_GRIDS["tc_chi2"]:
        out[("tc_chi2", alpha)] = _join(
            torrence_compo_chi2_mask(
                amp1,
                freq,
                b_perp1,
                dt=_DT_SEC,
                alpha=alpha,
            ),
            torrence_compo_chi2_mask(
                amp2,
                freq,
                b_perp2,
                dt=_DT_SEC,
                alpha=alpha,
            ),
        )
    for q in _ALPHA_GRIDS["fdr_chi2"]:
        out[("fdr_chi2", q)] = _join(
            fdr_chi2_mask(amp1, freq, b_perp1, dt=_DT_SEC, q=q),
            fdr_chi2_mask(amp2, freq, b_perp2, dt=_DT_SEC, q=q),
        )
    if archive is not None and region in archive.medians:
        for n_sigma in _ALPHA_GRIDS["pooled"]:
            out[("pooled", n_sigma)] = _join(
                pooled_archive_mask(amp1, freq, region, archive, n_sigma=n_sigma),
                pooled_archive_mask(amp2, freq, region, archive, n_sigma=n_sigma),
            )
    return out


def _run_all_methods(
    b_perp1: np.ndarray,
    b_perp2: np.ndarray,
    *,
    archive: BGArchive | None,
    region: str,
) -> dict[str, np.ndarray]:
    """Compute the four masks (or three if no archive). Joined via OR + COI."""
    freq, _, cwt1 = morlet_cwt(
        b_perp1,
        dt=_DT_SEC,
        omega0=_OMEGA0,
        n_freqs=_N_FREQS,
    )
    _, _, cwt2 = morlet_cwt(
        b_perp2,
        dt=_DT_SEC,
        omega0=_OMEGA0,
        n_freqs=_N_FREQS,
    )
    amp1, amp2 = np.abs(cwt1), np.abs(cwt2)
    n_time = amp1.shape[1]
    coi = coi_mask(freq, n_time, dt=_DT_SEC, omega0=_OMEGA0)
    n_sigma = bonferroni_n_sigma_for_cwt(
        n_time,
        _DT_SEC,
        freq,
        alpha=SEGMENT_FWER_ALPHA,
    )

    def _join(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        return (m1 | m2) & coi

    out: dict[str, np.ndarray] = {"freq": freq}
    out["mad_row"] = _join(
        wavelet_sigma_mask(amp1, freq, n_sigma=n_sigma),
        wavelet_sigma_mask(amp2, freq, n_sigma=n_sigma),
    )
    out["tc_chi2"] = _join(
        torrence_compo_chi2_mask(amp1, freq, b_perp1, dt=_DT_SEC),
        torrence_compo_chi2_mask(amp2, freq, b_perp2, dt=_DT_SEC),
    )
    out["fdr_chi2"] = _join(
        fdr_chi2_mask(amp1, freq, b_perp1, dt=_DT_SEC),
        fdr_chi2_mask(amp2, freq, b_perp2, dt=_DT_SEC),
    )
    if archive is not None and region in archive.medians:
        out["pooled"] = _join(
            pooled_archive_mask(amp1, freq, region, archive),
            pooled_archive_mask(amp2, freq, region, archive),
        )
    return out


def _try_load_bg_archive(path: Path) -> BGArchive | None:
    if not path.exists():
        log.info("bg_archive not found at %s — skipping pooled-mask", path)
        return None
    import zarr

    root = zarr.open(str(path), mode="r")
    periods = np.asarray(root["periods_sec"])
    n_segments = {
        r: int(np.asarray(root[f"n_segments/{r}"])) for r in root["n_segments"]
    }
    keep = {r for r, n in n_segments.items() if n > 0}
    medians = {
        r: np.asarray(root[f"medians/{r}"]) for r in root["medians"] if r in keep
    }
    mads = {r: np.asarray(root[f"mads/{r}"]) for r in root["mads"] if r in keep}
    return BGArchive(
        periods_sec=periods,
        medians=medians,
        mads=mads,
        n_segments={r: n for r, n in n_segments.items() if r in keep},
    )


# --------------------------------------------------------------------- #
# Fixed-FPR post-processing                                              #
# --------------------------------------------------------------------- #


def _recall_at_target_fpr(
    bg_rates: np.ndarray,
    recalls: np.ndarray,
    target: float,
) -> float:
    """Interpolate recall at the bg-rate matching ``target``.

    The (bg_rate, recall) curve is monotonic-ish in α, so a 1-D interp
    is enough. If the target is outside the swept range, return the
    closest endpoint and let the caller flag the saturation.
    """
    order = np.argsort(bg_rates)
    br = bg_rates[order]
    rc = recalls[order]
    if target <= br[0]:
        return float(rc[0])
    if target >= br[-1]:
        return float(rc[-1])
    return float(np.interp(target, br, rc))


def _emit_fixed_fpr_summary(df: pd.DataFrame, *, target_fpr: float) -> None:
    """Print the paper-headline table: recall at fixed bg-rate per method × amp."""
    # Aggregate across (segment, band) at each (method, alpha, amplitude).
    agg = df.groupby(["method", "alpha", "amplitude"], as_index=False).agg(
        recall_mean=("recall", "mean"),
        bg_rate_mean=("bg_rate", "mean"),
    )
    methods = sorted(agg["method"].unique())
    amplitudes = sorted(agg["amplitude"].unique())
    rows = []
    for amp in amplitudes:
        row = {"amplitude_nT": amp}
        for m in methods:
            sub = agg[(agg.method == m) & (agg.amplitude == amp)]
            if sub.empty:
                row[m] = np.nan
                continue
            row[m] = _recall_at_target_fpr(
                sub["bg_rate_mean"].to_numpy(),
                sub["recall_mean"].to_numpy(),
                target_fpr,
            )
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("amplitude_nT")
    log.info(
        "Recall at fixed bg-rate = %.3f (averaged over segments × bands):\n%s",
        target_fpr,
        summary.round(3).to_string(),
    )
    # Also a per-method "achievable bg-rate range" sanity report —
    # warns if the swept α grid doesn't bracket the target.
    rng = (
        agg.groupby("method")
        .agg(bg_min=("bg_rate_mean", "min"), bg_max=("bg_rate_mean", "max"))
        .round(4)
    )
    log.info("Swept bg-rate range per method:\n%s", rng.to_string())
    for m in methods:
        if rng.loc[m, "bg_min"] > target_fpr:
            log.warning(
                "method %s never reaches bg_rate=%.3f from below "
                "(min swept = %.4f); recall clamped to most-conservative α",
                m,
                target_fpr,
                rng.loc[m, "bg_min"],
            )
        if rng.loc[m, "bg_max"] < target_fpr:
            log.warning(
                "method %s never reaches bg_rate=%.3f from above "
                "(max swept = %.4f); recall clamped to most-permissive α",
                m,
                target_fpr,
                rng.loc[m, "bg_max"],
            )


def _plot_fixed_fpr_roc(
    df: pd.DataFrame,
    *,
    target_fpr: float,
    out_path: Path,
) -> None:
    """Two panels: (a) recall@FPR vs injection amplitude; (b) raw ROC curve."""
    agg = df.groupby(["method", "alpha", "amplitude"], as_index=False).agg(
        recall_mean=("recall", "mean"),
        bg_rate_mean=("bg_rate", "mean"),
    )
    methods = sorted(agg["method"].unique())
    method_colours = {
        "mad_row": "#ffffff",
        "tc_chi2": "#ffb000",
        "fdr_chi2": "#80c0ff",
        "pooled": "#4ecdc4",
    }
    method_labels = {
        "mad_row": "MAD-row (current)",
        "tc_chi2": "T&C χ²(2)",
        "fdr_chi2": "BH-Y FDR",
        "pooled": "pooled archive",
    }

    use_paper_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel (a): recall at fixed FPR vs injection amplitude.
    for m in methods:
        sub_m = agg[agg.method == m]
        amps = sorted(sub_m["amplitude"].unique())
        recalls = []
        for amp in amps:
            s = sub_m[sub_m.amplitude == amp]
            recalls.append(
                _recall_at_target_fpr(
                    s["bg_rate_mean"].to_numpy(),
                    s["recall_mean"].to_numpy(),
                    target_fpr,
                ),
            )
        ax_a.plot(
            amps,
            recalls,
            "o-",
            color=method_colours.get(m, "#aaaaaa"),
            lw=1.4,
            ms=5,
            label=method_labels.get(m, m),
        )
    ax_a.set_xscale("log")
    ax_a.set_xlabel("injection amplitude (nT)")
    ax_a.set_ylabel("recall (TP rate inside packet)")
    ax_a.set_ylim(-0.03, 1.05)
    ax_a.set_title(f"(a) recall at bg-rate = {target_fpr:.3f}", loc="left")
    ax_a.legend(loc="lower right", frameon=False, fontsize=9)

    # Panel (b): full ROC trace at one representative amplitude.
    amps_unique = sorted(agg["amplitude"].unique())
    # Pick the amplitude closest to 0.3 nT — the QP detection edge.
    amp_pick = min(amps_unique, key=lambda x: abs(x - 0.3))
    for m in methods:
        s = agg[(agg.method == m) & (agg.amplitude == amp_pick)].sort_values(
            "bg_rate_mean",
        )
        ax_b.plot(
            s["bg_rate_mean"],
            s["recall_mean"],
            "o-",
            color=method_colours.get(m, "#aaaaaa"),
            lw=1.4,
            ms=5,
            label=method_labels.get(m, m),
        )
    ax_b.axvline(target_fpr, ls=":", color="white", lw=0.8, alpha=0.7)
    ax_b.text(
        target_fpr * 1.05,
        0.04,
        f"target FPR = {target_fpr:.3f}",
        color="white",
        alpha=0.7,
        fontsize=8,
    )
    ax_b.set_xscale("log")
    ax_b.set_xlabel("bg rate (out-of-rect FP rate)")
    ax_b.set_ylabel("recall")
    ax_b.set_ylim(-0.03, 1.05)
    ax_b.set_title(
        f"(b) ROC at injection amplitude = {amp_pick} nT (QP detection edge)",
        loc="left",
    )
    ax_b.legend(loc="lower right", frameon=False, fontsize=9)

    fig.suptitle(
        "Like-for-like CWT threshold ROC — fixed-FPR comparison",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-segments", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--archive",
        default=str(qp.OUTPUT_DIR / "bg_archive.zarr"),
    )
    parser.add_argument(
        "--output",
        default=str(qp.OUTPUT_DIR / "figures" / "cwt_threshold_roc.png"),
    )
    parser.add_argument(
        "--csv-out",
        default=str(qp.OUTPUT_DIR / "cwt_threshold_roc.csv"),
    )
    parser.add_argument(
        "--alpha-sweep",
        action="store_true",
        help=(
            "run each method at multiple α/σ values so the resulting CSV "
            "spans a recall-vs-bg-rate curve per method. Required for "
            "--fixed-fpr."
        ),
    )
    parser.add_argument(
        "--fixed-fpr",
        type=float,
        default=None,
        help=(
            "target false-positive rate at which to report recall per method. "
            "Implies --alpha-sweep. Typical: 0.02 (matches MAD-row at "
            "Bonferroni σ on a 36-h segment)."
        ),
    )
    args = parser.parse_args()
    if args.fixed_fpr is not None:
        args.alpha_sweep = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    legacy_pickle.register_stubs()
    arr, _ = load_segments()
    archive = _try_load_bg_archive(Path(args.archive))
    methods = ["mad_row", "tc_chi2", "fdr_chi2"]
    if archive is not None:
        methods.append("pooled")

    # Sample valid segments. We don't filter "clean" segments rigorously —
    # the injection signal is large enough at the upper amplitudes to be
    # detectable above any normal MS background, and the FP rate is
    # measured outside the injection rectangle anyway.
    rng = np.random.default_rng(args.seed)
    valid = [i for i, seg in enumerate(arr) if getattr(seg, "flag", None) is None]
    sample = sorted(
        rng.choice(valid, size=min(args.n_segments, len(valid)), replace=False).tolist()
    )
    log.info("sampled %d segments", len(sample))

    half_width = _INJECTION_HALF_WIDTH_HOURS * 3600.0
    rows: list[dict] = []

    for k, seg_idx in enumerate(sample):
        seg = arr[seg_idx]
        payload = segment_to_payload(seg_idx, seg)
        if payload is None:
            continue
        n = len(payload.b_perp1)
        midpoint = payload.times[len(payload.times) // 2]
        region = region_at_peak_from_info(payload.info, midpoint)
        # Random injection centre in the middle 60 % of the segment so
        # the packet clears the COI on both sides comfortably.
        centre_idx = int(rng.integers(int(n * 0.2), int(n * 0.8)))

        for band_name, band in QP_BANDS.items():
            period_sec = band.period_centroid_sec
            # Get the freq grid + rectangle once per band (independent
            # of amplitude).
            freq, _, _ = morlet_cwt(
                payload.b_perp1,
                dt=_DT_SEC,
                omega0=_OMEGA0,
                n_freqs=_N_FREQS,
            )
            rect = _injection_rectangle(
                freq,
                n,
                period_sec=period_sec,
                centre_idx=centre_idx,
                half_width_sec=half_width,
            )
            rect_count = int(rect.sum())
            bg_count = int((~rect).sum())

            for amp_nt in _AMPLITUDES_NT:
                b1, b2 = _inject_packet(
                    payload.b_perp1,
                    payload.b_perp2,
                    period_sec=period_sec,
                    centre_idx=centre_idx,
                    half_width_sec=half_width,
                    amplitude=amp_nt,
                )
                if args.alpha_sweep:
                    sweep = _run_all_methods_alpha_sweep(
                        b1,
                        b2,
                        archive=archive,
                        region=region,
                    )
                    for (m, alpha_val), mask in sweep.items():
                        tp = int((mask & rect).sum())
                        fp = int((mask & ~rect).sum())
                        rows.append(
                            {
                                "seg_idx": seg_idx,
                                "band": band_name,
                                "amplitude": amp_nt,
                                "method": m,
                                "alpha": float(alpha_val),
                                "recall": tp / rect_count if rect_count else 0.0,
                                "bg_rate": fp / bg_count if bg_count else 0.0,
                                "n_in_rect": rect_count,
                                "n_out_rect": bg_count,
                            }
                        )
                else:
                    masks = _run_all_methods(
                        b1,
                        b2,
                        archive=archive,
                        region=region,
                    )
                    for m in methods:
                        if m not in masks:
                            continue
                        mask = masks[m]
                        tp = int((mask & rect).sum())
                        fp = int((mask & ~rect).sum())
                        rows.append(
                            {
                                "seg_idx": seg_idx,
                                "band": band_name,
                                "amplitude": amp_nt,
                                "method": m,
                                "recall": tp / rect_count if rect_count else 0.0,
                                "bg_rate": fp / bg_count if bg_count else 0.0,
                                "n_in_rect": rect_count,
                                "n_out_rect": bg_count,
                            }
                        )

        if (k + 1) % 2 == 0:
            log.info("processed %d / %d segments", k + 1, len(sample))

    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False)
    log.info("wrote %s", args.csv_out)
    if df.empty:
        log.error("no rows generated — aborting plot")
        return

    if args.fixed_fpr is not None:
        _emit_fixed_fpr_summary(df, target_fpr=args.fixed_fpr)
        _plot_fixed_fpr_roc(
            df,
            target_fpr=args.fixed_fpr,
            out_path=Path(args.output).with_name(
                Path(args.output).stem + "_fixed_fpr.png",
            ),
        )
        return

    if args.alpha_sweep:
        log.info(
            "alpha-sweep CSV written; supply --fixed-fpr <p> for the summary "
            "table (skipping the canonical per-band plot since methods now "
            "have multiple α per row)",
        )
        return

    # Aggregate: mean ± SEM across segments per (method, band, amplitude).
    agg = df.groupby(["method", "band", "amplitude"], as_index=False).agg(
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        bg_rate_mean=("bg_rate", "mean"),
    )

    use_paper_style()
    bands = ["QP15", "QP30", "QP60", "QP120"]
    fig, axes = plt.subplots(
        2, len(bands), figsize=(13, 5.6), sharex=True, sharey="row"
    )
    method_colours = {
        "mad_row": "#ffffff",
        "tc_chi2": "#ffb000",
        "fdr_chi2": "#80c0ff",
        "pooled": "#4ecdc4",
    }
    method_labels = {
        "mad_row": "MAD-row (current)",
        "tc_chi2": "T&C χ²(2)",
        "fdr_chi2": "BH-Y FDR",
        "pooled": "pooled archive",
    }
    for j, b in enumerate(bands):
        ax_r = axes[0][j]
        ax_f = axes[1][j]
        for m in methods:
            sub = agg[(agg.band == b) & (agg.method == m)].sort_values("amplitude")
            if sub.empty:
                continue
            ax_r.plot(
                sub.amplitude,
                sub.recall_mean,
                "o-",
                color=method_colours[m],
                lw=1.2,
                ms=4,
                label=method_labels[m],
            )
            ax_r.fill_between(
                sub.amplitude,
                sub.recall_mean - sub.recall_std,
                sub.recall_mean + sub.recall_std,
                color=method_colours[m],
                alpha=0.18,
                edgecolor="none",
            )
            ax_f.plot(
                sub.amplitude,
                sub.bg_rate_mean,
                "o-",
                color=method_colours[m],
                lw=1.2,
                ms=4,
            )
        ax_r.set_xscale("log")
        ax_r.set_ylim(-0.05, 1.05)
        ax_r.set_title(
            f"{b} ({QP_BANDS[b].period_centroid_minutes:.0f} min)",
            fontsize=10,
            color=QP_BAND_COLORS[b],
            loc="left",
        )
        if j == 0:
            ax_r.set_ylabel("recall (in-rect TP rate)")
            ax_f.set_ylabel("bg rate (out-of-rect FP rate)")
        ax_f.set_xscale("log")
        ax_f.set_xlabel("injection amplitude (nT)")
        ax_f.set_yscale("log")
        ax_f.set_ylim(1e-4, 1.0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Synthetic-injection ROC — recall and background rate per method × band",
        y=0.99,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
