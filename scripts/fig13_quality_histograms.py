"""Phase 7.4.5 — Quality-score histograms per band.

Shows the distribution of composite quality scores for each QP band,
revealing the separation between signal and noise populations. Also
shows the individual metric distributions.

Reads ``Output/events_qp_v2.parquet``.
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

from qp.plotting.style import use_paper_style  # noqa: E402


def main() -> None:
    catalog = _PROJECT_ROOT / "Output" / "events_qp_v2.parquet"
    if not catalog.exists():
        print(f"Catalog not found: {catalog}")
        return

    df = pd.read_parquet(catalog)
    use_paper_style()

    # --- Panel 1: Quality score per band ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    bins_q = np.linspace(0, 1, 31)

    for ax, band in zip(axes, ("QP30", "QP60", "QP120")):
        sub = df[df["band"] == band]
        q = sub["quality"].dropna().values
        if len(q) == 0:
            continue
        ax.hist(q, bins=bins_q, color="#ff6b6b", alpha=0.85, edgecolor="none")
        ax.axvline(np.median(q), color="white", lw=1.5, ls="--",
                    label=f"median = {np.median(q):.3f}")
        ax.set_title(f"{band} (n={len(q)})", fontsize=12)
        ax.set_xlabel("Quality score")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9, frameon=False)
    axes[0].set_ylabel("Count")

    fig.suptitle("Phase 7 — Quality score distribution per band", fontsize=13)
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure13_quality_histograms.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")

    # --- Panel 2: Individual metrics ---
    metrics = [
        ("wavelet_sigma", "Wavelet σ"),
        ("fft_screen_ratio", "FFT ratio"),
        ("mf_snr", "MF-SNR"),
        ("coherence", "Coherence"),
        ("transverse_ratio", "Trans. ratio"),
        ("polarization_fraction", "Pol. fraction"),
    ]
    available = [(col, label) for col, label in metrics if col in df.columns]
    if not available:
        return

    n = len(available)
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, (col, label) in zip(axes2.flat, available):
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue
        p5, p95 = np.percentile(vals, [5, 95])
        bins = np.linspace(max(0, p5 - 0.1 * (p95 - p5)),
                            p95 + 0.1 * (p95 - p5), 40)
        ax.hist(vals, bins=bins, color="#4ecdc4", alpha=0.85, edgecolor="none")
        ax.axvline(np.median(vals), color="white", lw=1.5, ls="--")
        ax.set_title(f"{label} (med={np.median(vals):.3f})", fontsize=11)
    for ax in axes2.flat[len(available):]:
        ax.set_visible(False)

    fig2.suptitle("Phase 7 — Individual detection metrics", fontsize=13)
    out2 = out_dir / "figure13b_metric_distributions.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight",
                 facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
