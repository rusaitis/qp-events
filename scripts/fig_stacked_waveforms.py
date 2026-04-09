"""Phase 8.5 — Phase-coherent stacked waveforms.

Selects the top events by quality score in each band, extracts a
±3-period window of b_perp1 centred on the CWT peak time, normalizes
each snippet to unit amplitude, aligns at the zero-crossing nearest
to the peak, and computes the mean ± 1σ envelope.

A clear oscillation in the stack (amplitude >> σ) is powerful visual
proof that the detections are phase-coherent real waves rather than
incoherent noise blobs.

Output: ``Output/figures/figure_stacked_waveforms.png``
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _register_pickle_stubs() -> None:
    stub_classes = [
        "SignalSnapshot", "NewSignal", "Interval", "FFT_list",
        "WaveSignal", "Wave",
    ]
    stub_modules = [
        "__main__", "data_sweeper", "mag_fft_sweeper",
        "cassinilib", "cassinilib.NewSignal",
    ]
    for mod_path in stub_modules:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)
        for cls in stub_classes:
            setattr(sys.modules[mod_path], cls, type(cls, (), {}))


_register_pickle_stubs()

import qp  # noqa: E402
from qp.plotting.style import use_paper_style  # noqa: E402

DT = 60.0  # seconds per sample
N_TOP = 50   # events per band
N_PERIODS = 3  # half-window in periods


def _extract_snippet(
    b_perp1: np.ndarray,
    times,
    peak_time,
    period_sec: float,
    n_periods: int = N_PERIODS,
) -> np.ndarray | None:
    """Extract ±n_periods window around peak, zero-phase-aligned."""
    import datetime
    epoch = datetime.datetime(1970, 1, 1)
    t_unix = np.array(
        [(t - epoch).total_seconds() for t in times], dtype=float
    )
    peak_unix = (peak_time - epoch).total_seconds()
    peak_idx = int(np.argmin(np.abs(t_unix - peak_unix)))

    half_n = int(np.ceil(n_periods * period_sec / DT))
    lo = peak_idx - half_n
    hi = peak_idx + half_n + 1
    if lo < 0 or hi > len(b_perp1):
        return None

    snippet = b_perp1[lo:hi].copy()
    if not np.isfinite(snippet).all():
        return None

    # Normalize to unit amplitude
    rms = np.sqrt(np.mean(snippet ** 2))
    if rms < 1e-12:
        return None
    snippet /= rms

    # Align by zero-crossing nearest to the peak (centre of snippet)
    centre = len(snippet) // 2
    sign_centre = np.sign(snippet[centre])
    # Find closest zero-crossing to centre
    zero_cross_candidates = np.where(
        np.diff(np.sign(snippet[:centre + 1]))
    )[0]
    if len(zero_cross_candidates) == 0:
        shift = 0
    else:
        closest = zero_cross_candidates[
            np.argmin(np.abs(zero_cross_candidates - centre))
        ]
        shift = centre - closest
        # Only shift by at most half a period
        max_shift = int(0.5 * period_sec / DT)
        shift = int(np.clip(shift, -max_shift, max_shift))

    if shift != 0:
        if shift > 0:
            snippet = np.pad(snippet[shift:], (0, shift), mode="edge")
        else:
            snippet = np.pad(snippet[:shift], (-shift, 0), mode="edge")

    return snippet


def stack_band(
    df: pd.DataFrame,
    band: str,
    quality_col: str,
    segments: np.ndarray,
    n_top: int = N_TOP,
    target_n_samples: int = 600,  # common grid for all snippets
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Stack top n_top events for one band. Returns (time_periods, mean, std, period_min).

    All snippets are resampled to a common phase grid using linear
    interpolation, so events with slightly different periods can be
    co-added coherently.
    """
    import datetime

    b = df[(df.band == band)].sort_values(quality_col, ascending=False)
    b = b.dropna(subset=[quality_col, "period_peak_min", "segment_id"])
    b = b.head(n_top)

    if len(b) == 0:
        return None

    median_period = float(b["period_peak_min"].median())
    # Target phase grid: −N_PERIODS … +N_PERIODS in target_n_samples steps
    t_common = np.linspace(-N_PERIODS, N_PERIODS, target_n_samples)

    snippets = []
    for _, row in b.iterrows():
        seg_id = int(row["segment_id"])
        if seg_id >= len(segments):
            continue
        seg = segments[seg_id]
        if seg.flag is not None:
            continue

        b_perp1 = np.nan_to_num(np.asarray(seg.FIELDS[1].y, dtype=float))
        times = list(seg.datetime)

        # Use the event's own period for extraction
        period_sec = float(row["period_peak_min"]) * 60.0

        t_from = datetime.datetime.fromisoformat(str(row["date_from"]))
        t_to = datetime.datetime.fromisoformat(str(row["date_to"]))
        peak_time = t_from + (t_to - t_from) / 2

        snip = _extract_snippet(b_perp1, times, peak_time, period_sec)
        if snip is None:
            continue

        # Map snippet to phase coordinates and resample to common grid
        t_snip = np.linspace(-N_PERIODS, N_PERIODS, len(snip))
        snip_resampled = np.interp(t_common, t_snip, snip)
        snippets.append(snip_resampled)

    if len(snippets) < 3:
        return None

    stack = np.array(snippets)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)

    print(f"    {band}: stacked {len(snippets)} snippets")
    return t_common, mean, std, median_period


def main() -> None:
    cat_path = _PROJECT_ROOT / "Output" / "events_qp_v3.parquet"
    if not cat_path.exists():
        cat_path = _PROJECT_ROOT / "Output" / "events_qp_v2.parquet"
    df = pd.read_parquet(cat_path)
    quality_col = "quality_v3" if "quality_v3" in df.columns else "quality"

    print("Loading segments...")
    arr = np.load(
        qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy", allow_pickle=True,
    )

    use_paper_style()
    colors = {"QP30": "#4ecdc4", "QP60": "#ff6b6b", "QP120": "#ffd93d"}
    bands = ["QP30", "QP60", "QP120"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, band in zip(axes, bands):
        print(f"Stacking {band}...")
        result = stack_band(df, band, quality_col, arr, n_top=N_TOP)
        if result is None:
            ax.set_title(f"{band} (insufficient data)")
            continue

        t_phases, mean, std, period_min = result
        c = colors[band]

        ax.fill_between(t_phases, mean - std, mean + std,
                         color=c, alpha=0.25, label=r"$\pm 1\sigma$")
        ax.plot(t_phases, mean, color=c, lw=2, label="mean")
        ax.axhline(0, color="white", lw=0.5, alpha=0.3)
        ax.axvline(0, color="white", lw=0.5, ls=":", alpha=0.4)

        # SNR of stack
        peak_amp = np.max(np.abs(mean))
        noise_at_tail = np.std(mean[:10])
        snr_stack = peak_amp / max(noise_at_tail, 1e-6)

        ax.set_xlabel("Phase (periods from peak)")
        ax.set_ylabel("Normalized amplitude")
        ax.set_title(
            f"{band}  (top {N_TOP}, P={period_min:.0f} min, "
            f"stack SNR={snr_stack:.1f})",
            fontsize=11,
        )
        ax.legend(fontsize=8, frameon=False)
        ax.set_xlim(-N_PERIODS, N_PERIODS)

        print(f"  {band}: stacked {len(mean) // int(period_min)}-sample snippets, "
              f"stack SNR={snr_stack:.1f}")

    fig.suptitle(
        f"Phase 8.5 — Phase-coherent stacked waveforms (top {N_TOP} events per band)",
        fontsize=12,
    )
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure_stacked_waveforms.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
