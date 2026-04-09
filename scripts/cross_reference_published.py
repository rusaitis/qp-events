"""Phase 8.6 — Cross-reference detector against published events.

Checks that the paper's example events are in the v3 catalog with
high quality scores:

- 2007-01-02: QP60 example shown in Fig 1 and Fig 4 of the paper.

Reports event_id, quality, all metrics, duration, and amplitude for
all events detected within that date. Also verifies that the power-law
FFT background ratio at the 60-min period is >> 1 (confirming the
pipeline sees the published wave peak).

Output: ``Output/diagnostics/cross_reference_published.txt``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main() -> None:
    cat_path = _PROJECT_ROOT / "Output" / "events_qp_v3.parquet"
    if not cat_path.exists():
        cat_path = _PROJECT_ROOT / "Output" / "events_qp_v2.parquet"
    df = pd.read_parquet(cat_path)

    quality_col = "quality_v3" if "quality_v3" in df.columns else "quality"

    lines: list[str] = ["Phase 8.6 — Cross-reference with published events",
                        "=" * 60, ""]

    # ── 2007-01-02: QP60 example from paper Fig 1 / Fig 4 ──────────────────
    target_date = "2007-01-02"
    mask = (
        pd.to_datetime(df["date_from"]).dt.date.astype(str) == target_date
    ) | (
        pd.to_datetime(df["date_to"]).dt.date.astype(str) == target_date
    )
    near = df[mask].copy()

    lines.append(f"Date: {target_date} (QP60 example from paper Fig 1/4)")
    lines.append(f"Events touching this date: {len(near)}")
    lines.append("")

    if len(near) == 0:
        lines.append("  *** DETECTOR MISSED THIS DATE — investigate! ***")
        lines.append("")
        # Also check ±1 day
        for d in ["2007-01-01", "2007-01-03"]:
            mask2 = (
                pd.to_datetime(df["date_from"]).dt.date.astype(str) == d
            ) | (
                pd.to_datetime(df["date_to"]).dt.date.astype(str) == d
            )
            adj = df[mask2]
            if len(adj) > 0:
                lines.append(f"  Adjacent date {d}: {len(adj)} events found")
                for _, row in adj.iterrows():
                    lines.append(
                        f"    {row.get('event_id','?')}: band={row['band']}, "
                        f"quality={row.get(quality_col,np.nan):.3f}, "
                        f"period={row['period_peak_min']:.1f} min"
                    )
    else:
        metric_cols = [
            "event_id", "band", "date_from", "date_to", "period_peak_min",
            "duration_minutes", "amplitude", "rms_amplitude_perp",
            quality_col, "wavelet_sigma", "mf_snr", "coherence",
            "fft_screen_ratio", "local_fft_ratio", "bp_transverse_ratio",
            "transverse_ratio", "polarization_fraction", "ellipticity",
        ]
        available = [c for c in metric_cols if c in near.columns]
        for _, row in near.sort_values("band").iterrows():
            lines.append(f"  event_id           : {row.get('event_id','?')}")
            lines.append(f"  band               : {row['band']}")
            lines.append(f"  date_from          : {row['date_from']}")
            lines.append(f"  date_to            : {row['date_to']}")
            lines.append(f"  period_peak_min    : {row['period_peak_min']:.2f}")
            lines.append(f"  duration_minutes   : {row['duration_minutes']:.1f}")
            lines.append(f"  amplitude (nT)     : {row['amplitude']:.3f}")
            lines.append(f"  rms_perp (nT)      : {row['rms_amplitude_perp']:.3f}")
            lines.append(f"  {quality_col}       : {row.get(quality_col,np.nan):.3f}")
            lines.append(f"  wavelet_sigma      : {row.get('wavelet_sigma',np.nan):.3f}")
            lines.append(f"  mf_snr             : {row.get('mf_snr',np.nan):.2f}")
            lines.append(f"  coherence          : {row.get('coherence',np.nan):.3f}")
            lines.append(f"  fft_ratio (36h)    : {row.get('fft_screen_ratio',np.nan):.3f}")
            lines.append(f"  fft_ratio (local)  : {row.get('local_fft_ratio',np.nan):.3f}")
            lines.append(f"  bp_transv_ratio    : {row.get('bp_transverse_ratio',np.nan):.2f}")
            lines.append(f"  transv_ratio (bb)  : {row.get('transverse_ratio',np.nan):.4f}")
            lines.append(f"  pol_fraction       : {row.get('polarization_fraction',np.nan):.3f}")
            lines.append(f"  ellipticity        : {row.get('ellipticity',np.nan):.3f}")
            lines.append("")

    # ── Catalog-wide statistics summary ──────────────────────────────────────
    lines.append("")
    lines.append("Catalog-wide quality statistics:")
    for band in ["QP30", "QP60", "QP120", "all"]:
        if band == "all":
            b = df
            label = "all"
        else:
            b = df[df.band == band]
            label = band
        q = b[quality_col].dropna()
        if len(q) == 0:
            continue
        lines.append(
            f"  {label:6s}: n={len(q):4d}, median={q.median():.3f}, "
            f">0.3={( q>0.3).sum():3d} ({( q>0.3).mean()*100:.0f}%), "
            f">0.5={( q>0.5).sum():3d} ({( q>0.5).mean()*100:.0f}%)"
        )

    # ── Detection of missed known events ─────────────────────────────────────
    lines.append("")
    lines.append("Known high-quality events from paper:")
    known = [
        ("2007-01-02", "QP60", "Fig 1 and Fig 4 example"),
        ("2007-01-03", "QP60", "adjacent to Fig 1 example"),
    ]
    for date, band, note in known:
        mask_k = (
            pd.to_datetime(df["date_from"]).dt.date.astype(str) == date
        ) | (
            pd.to_datetime(df["date_to"]).dt.date.astype(str) == date
        )
        found = df[mask_k & (df.band == band)]
        q_str = (
            f"quality={found[quality_col].max():.3f}"
            if len(found) > 0 else "MISSED"
        )
        lines.append(f"  {date} {band:6s} ({note}): n={len(found):2d}, {q_str}")

    out_dir = _PROJECT_ROOT / "Output" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cross_reference_published.txt"
    text = "\n".join(lines)
    out.write_text(text)
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
