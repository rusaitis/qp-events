r"""Head-to-head comparison of CWT background-threshold methods.

Runs the canonical per-row MAD gate
(:func:`qp.events.threshold.wavelet_sigma_mask`) and the three
alternatives in :mod:`qp.events.threshold_diag` on a random sample of
MFA segments. Reports:

1. **Overlay grid** — for a handful of representative segments, plot
   each method's per-row threshold against the 5/50/95-percentile CWT
   amplitude envelope. Lets you eyeball where the methods disagree.
2. **Aggregate counts** — total cells passing each method, broken out
   by QP band, summed over the full sample.
3. **Jaccard agreement matrix** — pairwise overlap of the kept-cell
   sets across methods, averaged over segments.

Pooled-archive comparison is skipped automatically if
``Output/bg_archive.zarr`` does not yet exist (Phase 2 builds it).

Usage
-----
``uv run python scripts/diag_cwt_threshold_comparison.py
  --n-segments 50 --seed 0
  --overlay-out Output/figures/cwt_threshold_comparison.png
  --agreement-out Output/figures/cwt_threshold_agreement.png
  --csv-out Output/cwt_threshold_comparison.csv``
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

log = logging.getLogger("diag.cwt_threshold")

# CWT configuration mirrors detect_round8 so the diagnostic threshold
# curves are directly comparable to the production sweep.
_DT_SEC: float = 60.0
_N_FREQS: int = 300


def _try_load_bg_archive(path: Path) -> BGArchive | None:
    """Return the pooled-archive Zarr if it exists, else ``None``."""
    if not path.exists():
        log.info("bg_archive not found at %s — skipping pooled-mask column", path)
        return None
    try:
        import zarr  # local import; archive is optional

        root = zarr.open(str(path), mode="r")
        periods = np.asarray(root["periods_sec"])
        # n_segments stored as one scalar per region under a group;
        # drop regions with zero contributing segments so callers
        # don't pull all-NaN stats by accident.
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
    except Exception as exc:  # pragma: no cover — defensive only
        log.warning("could not load bg_archive (%s); skipping pooled-mask", exc)
        return None


def _per_segment_masks(
    b_perp1: np.ndarray,
    b_perp2: np.ndarray,
    *,
    archive: BGArchive | None,
    region: str,
) -> dict[str, np.ndarray]:
    """Run every available mask method on one segment's transverse pair."""
    freq, _, cwt1 = morlet_cwt(b_perp1, dt=_DT_SEC, n_freqs=_N_FREQS)
    _, _, cwt2 = morlet_cwt(b_perp2, dt=_DT_SEC, n_freqs=_N_FREQS)
    amp1 = np.abs(cwt1)
    amp2 = np.abs(cwt2)
    n_time = amp1.shape[1]
    coi = coi_mask(freq, n_time, dt=_DT_SEC)

    # Bonferroni σ for the MAD baseline — matches detect_round8.
    n_sigma = bonferroni_n_sigma_for_cwt(
        n_time,
        _DT_SEC,
        freq,
        alpha=SEGMENT_FWER_ALPHA,
    )

    out: dict[str, np.ndarray] = {}
    out["freq"] = freq
    out["coi"] = coi

    # Per-component → OR (matches the detector's two-axis acceptance).
    def _join(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        return (m1 | m2) & coi

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

    # Per-row CWT amplitude envelope for the overlay plot — geometric
    # mean of the two components so a single curve summarises both.
    out["amp_mean"] = np.sqrt(amp1 * amp2 + 1e-30)
    out["n_sigma_bonf"] = float(n_sigma)
    return out


def _sample_segments(
    arr: np.ndarray,
    n: int,
    seed: int,
) -> list[int]:
    """Random sample of valid (non-flagged) segment indices."""
    rng = np.random.default_rng(seed)
    valid = [i for i, seg in enumerate(arr) if getattr(seg, "flag", None) is None]
    if len(valid) <= n:
        return valid
    return sorted(rng.choice(valid, size=n, replace=False).tolist())


# ---------------------------------------------------------------------- #
# Aggregation                                                            #
# ---------------------------------------------------------------------- #


def _count_cells_per_band(
    mask: np.ndarray,
    freq: np.ndarray,
) -> dict[str, int]:
    """Cells passing the mask, broken down by QP band."""
    periods = freq_to_period(freq)
    counts: dict[str, int] = {}
    for name, band in QP_BANDS.items():
        in_band = (periods >= band.period_min_sec) & (periods < band.period_max_sec)
        counts[name] = int(mask[in_band].sum())
    counts["total"] = int(mask.sum())
    return counts


def _jaccard(m1: np.ndarray, m2: np.ndarray) -> float:
    """Jaccard index of two boolean masks; 1.0 when both are empty."""
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    return 1.0 if union == 0 else inter / union


# ---------------------------------------------------------------------- #
# Plots                                                                  #
# ---------------------------------------------------------------------- #


_METHOD_COLOURS: dict[str, str] = {
    "mad_row": "#ffffff",  # baseline → white
    "tc_chi2": "#ffb000",  # amber
    "fdr_chi2": "#80c0ff",  # cool blue
    "pooled": "#4ecdc4",  # teal
}
_METHOD_LABELS: dict[str, str] = {
    "mad_row": "current (MAD-interp)",
    "tc_chi2": "Torrence-Compo AR(1)+χ²",
    "fdr_chi2": "FDR (BH-Y) + AR(1) null",
    "pooled": "pooled archive (region MAD)",
}


def _plot_overlay_grid(
    panels: list[dict],
    out_path: Path,
    *,
    methods: list[str],
) -> None:
    """Per-segment overlays of the threshold curves and CWT envelope."""
    n = len(panels)
    cols = 2
    rows = (n + cols - 1) // cols
    use_paper_style()
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(11, 2.6 * rows + 1),
        squeeze=False,
    )
    for k, panel in enumerate(panels):
        ax = axes[k // cols][k % cols]
        freq = panel["freq"]
        periods_min = freq_to_period(freq) / 60.0
        amp = panel["amp_mean"]
        # 5/50/95 percentile envelope over time on each row.
        p5 = np.percentile(amp, 5, axis=1)
        p50 = np.percentile(amp, 50, axis=1)
        p95 = np.percentile(amp, 95, axis=1)
        ax.fill_between(
            periods_min, p5, p95, color="#888888", alpha=0.25, label="5–95% CWT amp"
        )
        ax.plot(periods_min, p50, color="#aaaaaa", lw=1.2, label="median")

        # Per-row effective threshold = smallest CWT amplitude in the
        # mask. For methods whose nominal threshold is a simple per-row
        # number, this recovers the canonical curve directly.
        for m in methods:
            if m not in panel:
                continue
            mask = panel[m]
            with np.errstate(invalid="ignore"):
                thr = np.where(
                    mask.any(axis=1),
                    np.where(mask, amp, np.inf).min(axis=1),
                    np.nan,
                )
            ax.plot(
                periods_min,
                thr,
                color=_METHOD_COLOURS[m],
                lw=1.0,
                label=_METHOD_LABELS[m],
            )

        # Shade QP bands.
        for name, band in QP_BANDS.items():
            ax.axvspan(
                band.period_min_minutes,
                band.period_max_minutes,
                color=QP_BAND_COLORS[name],
                alpha=0.07,
                zorder=0,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(5, 720)
        ax.set_xlabel("period (min)")
        ax.set_ylabel("|W| (nT)")
        ax.set_title(panel["title"], fontsize=9, loc="left")

    # Hide spare axes and emit a single legend.
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "CWT threshold methods — per-row envelope and effective threshold "
        f"({n} representative segments)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def _plot_aggregate(
    df_counts: pd.DataFrame,
    jaccard: np.ndarray,
    methods: list[str],
    out_path: Path,
) -> None:
    """Two-panel: per-band cell totals + Jaccard agreement heatmap."""
    use_paper_style()
    fig, (ax_bar, ax_jac) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Per-band totals, log y-axis (cells/method can vary by ≥ 1 dex).
    bands = ["QP15", "QP30", "QP60", "QP120", "total"]
    x = np.arange(len(bands))
    width = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        ax_bar.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            [int(df_counts.loc[m, b]) for b in bands],
            width=width,
            color=_METHOD_COLOURS[m],
            edgecolor="black",
            lw=0.4,
            label=_METHOD_LABELS[m],
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bands)
    ax_bar.set_yscale("log")
    ax_bar.set_ylabel("cells passing (summed across sample)")
    ax_bar.set_title("(a) total kept cells per method × band", loc="left")
    ax_bar.legend(loc="upper right", fontsize=8, frameon=False)

    # Jaccard agreement.
    im = ax_jac.imshow(
        jaccard,
        vmin=0.0,
        vmax=1.0,
        cmap="cividis",
        aspect="auto",
    )
    ax_jac.set_xticks(range(len(methods)))
    ax_jac.set_yticks(range(len(methods)))
    ax_jac.set_xticklabels(methods, rotation=20, ha="right")
    ax_jac.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax_jac.text(
                j,
                i,
                f"{jaccard[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if jaccard[i, j] < 0.55 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax_jac, fraction=0.045)
    ax_jac.set_title(
        "(b) Jaccard agreement of kept-cell sets (segment-mean)",
        loc="left",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-segments", type=int, default=50)
    parser.add_argument(
        "--n-overlay",
        type=int,
        default=8,
        help="how many segments to show in the overlay grid",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--archive",
        default=str(qp.OUTPUT_DIR / "bg_archive.zarr"),
        help="path to pooled bg archive (optional)",
    )
    parser.add_argument(
        "--overlay-out",
        default=str(qp.OUTPUT_DIR / "figures" / "cwt_threshold_comparison.png"),
    )
    parser.add_argument(
        "--agreement-out",
        default=str(qp.OUTPUT_DIR / "figures" / "cwt_threshold_agreement.png"),
    )
    parser.add_argument(
        "--csv-out", default=str(qp.OUTPUT_DIR / "cwt_threshold_comparison.csv")
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    legacy_pickle.register_stubs()
    arr, _ = load_segments()
    log.info("loaded %d segments from archive", len(arr))

    sample = _sample_segments(arr, args.n_segments, args.seed)
    archive = _try_load_bg_archive(Path(args.archive))
    methods = ["mad_row", "tc_chi2", "fdr_chi2"]
    if archive is not None:
        methods.append("pooled")

    # Aggregates.
    counts_by_method: dict[str, dict[str, int]] = {
        m: {b: 0 for b in ("QP15", "QP30", "QP60", "QP120", "total")} for m in methods
    }
    jaccard_sum = np.zeros((len(methods), len(methods)))
    jaccard_n = np.zeros((len(methods), len(methods)), dtype=int)
    overlay_panels: list[dict] = []
    rng = np.random.default_rng(args.seed + 1)
    overlay_pick = set(
        rng.choice(
            len(sample), size=min(args.n_overlay, len(sample)), replace=False
        ).tolist()
    )
    rows_csv: list[dict] = []

    for k, seg_idx in enumerate(sample):
        seg = arr[seg_idx]
        payload = segment_to_payload(seg_idx, seg)
        if payload is None:
            continue
        midpoint = payload.times[len(payload.times) // 2]
        region = region_at_peak_from_info(payload.info, midpoint)
        results = _per_segment_masks(
            payload.b_perp1,
            payload.b_perp2,
            archive=archive,
            region=region,
        )
        freq = results["freq"]
        row = {
            "seg_idx": seg_idx,
            "region": region,
            "t0": payload.times[0].isoformat(),
            "n_sigma_bonf": results["n_sigma_bonf"],
        }
        # Some methods are conditional (pooled requires the segment's
        # region to be present in the archive). Skip them gracefully
        # rather than crashing the whole sweep.
        available = [m for m in methods if m in results]
        for m in available:
            counts = _count_cells_per_band(results[m], freq)
            for b, v in counts.items():
                counts_by_method[m][b] += v
                row[f"{m}_{b}"] = v
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if m1 not in results or m2 not in results:
                    continue
                jaccard_sum[i, j] += _jaccard(results[m1], results[m2])
                jaccard_n[i, j] += 1
        rows_csv.append(row)

        if k in overlay_pick:
            results["title"] = (
                f"seg {seg_idx} · {payload.times[0]:%Y-%m-%d %H:%M} · "
                f"{region}  (Bonf σ={results['n_sigma_bonf']:.1f})"
            )
            overlay_panels.append(results)

        if (k + 1) % 5 == 0:
            log.info("processed %d / %d segments", k + 1, len(sample))

    # Save aggregates.
    df_counts = pd.DataFrame(counts_by_method).T
    log.info("aggregate kept-cell counts:\n%s", df_counts.to_string())

    pd.DataFrame(rows_csv).to_csv(args.csv_out, index=False)
    log.info("wrote %s", args.csv_out)

    with np.errstate(invalid="ignore"):
        jaccard = np.where(jaccard_n > 0, jaccard_sum / jaccard_n, np.nan)

    if overlay_panels:
        _plot_overlay_grid(overlay_panels, Path(args.overlay_out), methods=methods)
    _plot_aggregate(df_counts, jaccard, methods, Path(args.agreement_out))


if __name__ == "__main__":
    main()
