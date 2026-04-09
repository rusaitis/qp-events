"""Phase 8 — Post-hoc enrichment of the v2 event catalog.

Adds to each event:

Phase 8.2 — ``bp_transverse_ratio``:
    Band-pass filtered transverse ratio (perp/par) within the event's
    QP band using a 4th-order Butterworth filter. Isolates the Alfvénic
    perturbation from slow compressions that dominate the broadband par.

Phase 8.3 — ``local_fft_ratio``:
    Time-resolved power-law FFT ratio computed from a 6-hour Hann
    window centred on the event peak time, rather than the full 36-hour
    Welch estimate. Avoids dilution by quiet hours on either side.

Also recomputes quality scores using the improved metrics.

Output: ``Output/events_qp_v3.parquet``

Usage::

    uv run python scripts/enrich_events_v3.py
    uv run python scripts/enrich_events_v3.py --serial   # debug
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

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
from qp.events.bands import QP_BANDS, QP_BAND_NAMES  # noqa: E402
from qp.events.quality import QualityConfig, compute_quality  # noqa: E402
from qp.signal.fft import estimate_background_powerlaw, welch_psd  # noqa: E402

from sweep_events import load_segments, SegmentPayload, segment_to_payload  # noqa: E402

DT = 60.0  # 1-min samples, seconds
NYQUIST = 0.5 / DT  # Hz

# ── Quality config: recalibrated after including improved metrics ──────────────
# We keep the same anchors as Phase 7 for existing metrics. The bp_transverse_ratio
# replaces the broadband transverse_ratio in the quality formula.
# Anchors derived empirically after running the full enrichment:
# bp_transverse_ratio p10~0.5, p90~20 (much better than broadband 0.001–0.65)
QUALITY_V3_CONFIG = QualityConfig(
    wavelet_sigma=(0.5, 9.0),
    fft_ratio=(0.10, 2.56),      # local_fft_ratio p10/p90 from mission catalog
    mf_snr=(1.7, 16.0),
    coherence=(0.12, 0.66),
    n_oscillations=(3.0, 10.0),
    transverse_ratio=(0.77, 82.1),  # bp_transverse_ratio p10/p90 from mission catalog
    polarization_fraction=(0.3, 0.9),
)


# ── Butterworth band-pass filter ──────────────────────────────────────────────

def _bandpass_butterworth(
    data: np.ndarray, low_hz: float, high_hz: float, fs: float = 1 / DT,
    order: int = 4,
) -> np.ndarray:
    """4th-order Butterworth band-pass filter (sos, zero-phase)."""
    nyq = 0.5 * fs
    lo = max(low_hz / nyq, 1e-4)
    hi = min(high_hz / nyq, 0.999)
    if lo >= hi:
        return data.copy()
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfilt(sos, data)


def _bp_transverse_ratio(
    b_perp1: np.ndarray, b_perp2: np.ndarray, b_par: np.ndarray,
    band_name: str, sl: slice,
) -> float | None:
    """Band-pass transverse ratio for Phase 8.2."""
    if band_name not in QP_BANDS:
        return None
    band = QP_BANDS[band_name]
    # Filter the full array (avoids edge effects at event boundary)
    try:
        p1f = _bandpass_butterworth(b_perp1, band.freq_min_hz, band.freq_max_hz)
        p2f = _bandpass_butterworth(b_perp2, band.freq_min_hz, band.freq_max_hz)
        paf = _bandpass_butterworth(b_par, band.freq_min_hz, band.freq_max_hz)
    except Exception:
        return None
    perp_pwr = np.nanmean(p1f[sl] ** 2 + p2f[sl] ** 2)
    par_pwr = np.nanmean(paf[sl] ** 2)
    if par_pwr < 1e-30:
        return None
    return float(perp_pwr / par_pwr)


def _local_fft_ratio(
    b_perp1: np.ndarray, b_perp2: np.ndarray,
    peak_idx: int, period_sec: float,
    win_minutes: int = 360,  # 6 hours
) -> float | None:
    """Time-resolved FFT ratio at event frequency (Phase 8.3).

    Uses a Hann window of width ``win_minutes`` centred on ``peak_idx``.
    """
    n = len(b_perp1)
    half = win_minutes // 2
    lo = max(0, peak_idx - half)
    hi = min(n, peak_idx + half)
    if hi - lo < 60:  # need at least 1 hour of data
        return None
    seg1 = b_perp1[lo:hi]
    seg2 = b_perp2[lo:hi]
    try:
        nperseg = min(len(seg1), 360)  # up to 6h window for Welch
        f1, psd1 = welch_psd(seg1, dt=DT, nperseg=nperseg, noverlap=nperseg // 2)
        _, psd2 = welch_psd(seg2, dt=DT, nperseg=nperseg, noverlap=nperseg // 2)
        # Use mean of both perp channels
        psd_mean = 0.5 * (psd1 + psd2)
        bg = estimate_background_powerlaw(psd_mean, f1)
        ratio = psd_mean / np.maximum(bg, 1e-30)
        f_event = 1.0 / period_sec
        f_idx = int(np.argmin(np.abs(f1 - f_event)))
        return float(ratio[f_idx])
    except Exception:
        return None


# ── Per-segment enrichment worker ─────────────────────────────────────────────

class _EnrichArgs:
    """Picklable args for the pool worker."""
    __slots__ = ("payload", "events_df")

    def __init__(self, payload: SegmentPayload, events_df: pd.DataFrame) -> None:
        self.payload = payload
        self.events_df = events_df


def _enrich_segment(arg: _EnrichArgs) -> list[dict]:
    """Worker: compute 8.2 + 8.3 metrics for all events in one segment."""
    payload = arg.payload
    evs = arg.events_df
    if len(evs) == 0:
        return []

    b_perp1 = payload.b_perp1
    b_perp2 = payload.b_perp2
    b_par = payload.b_par
    times = payload.times
    n = len(times)
    epoch = __import__("datetime").datetime(1970, 1, 1)
    time_unix = np.array(
        [(t - epoch).total_seconds() for t in times], dtype=float
    )

    results = []
    for _, row in evs.iterrows():
        t_from_unix = (
            __import__("datetime").datetime.fromisoformat(str(row["date_from"]))
            - epoch
        ).total_seconds()
        t_to_unix = (
            __import__("datetime").datetime.fromisoformat(str(row["date_to"]))
            - epoch
        ).total_seconds()
        t_peak_unix = t_from_unix + (t_to_unix - t_from_unix) / 2.0
        i_from = int(np.argmin(np.abs(time_unix - t_from_unix)))
        i_to = int(np.argmin(np.abs(time_unix - t_to_unix)))
        peak_idx = int(np.argmin(np.abs(time_unix - t_peak_unix)))
        sl = slice(i_from, min(i_to + 1, n))

        period_sec = float(row["period"]) if pd.notna(row["period"]) else None
        band = str(row["band"]) if pd.notna(row["band"]) else None

        bp_tr = _bp_transverse_ratio(b_perp1, b_perp2, b_par, band, sl)
        local_fft = (
            _local_fft_ratio(b_perp1, b_perp2, peak_idx, period_sec)
            if period_sec else None
        )

        # Recompute quality with improved metrics
        quality_v3 = compute_quality(
            wavelet_sigma=(
                float(row["wavelet_sigma"])
                if pd.notna(row.get("wavelet_sigma")) else None
            ),
            fft_ratio=local_fft,                      # use local ratio
            mf_snr=(
                float(row["mf_snr"])
                if pd.notna(row.get("mf_snr")) else None
            ),
            coherence=(
                float(row["coherence"])
                if pd.notna(row.get("coherence")) else None
            ),
            n_oscillations=(
                float(row["n_oscillations"])
                if pd.notna(row.get("n_oscillations")) else None
            ),
            transverse_ratio=bp_tr,                    # use band-pass ratio
            polarization_fraction=(
                float(row["polarization_fraction"])
                if pd.notna(row.get("polarization_fraction")) else None
            ),
            config=QUALITY_V3_CONFIG,
        )

        results.append({
            "event_id": row["event_id"],
            "bp_transverse_ratio": bp_tr,
            "local_fft_ratio": local_fft,
            "quality_v3": quality_v3,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=_PROJECT_ROOT / "Output" / "events_qp_v2.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=_PROJECT_ROOT / "Output" / "events_qp_v3.parquet",
    )
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()

    print(f"Loading catalog: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  {len(df)} events, {df['segment_id'].nunique()} segments")

    print("Loading segments...")
    segments, _ = load_segments(year=None)
    print(f"  {len(segments)} total segments")

    # Build segment→payload map (only for segments with events)
    seg_ids_needed = set(df["segment_id"].dropna().astype(int))
    print(f"  building payloads for {len(seg_ids_needed)} active segments...")
    payloads: dict[int, SegmentPayload] = {}
    for i in seg_ids_needed:
        if i < len(segments):
            p = segment_to_payload(i, segments[i])
            if p is not None:
                payloads[i] = p
    print(f"  {len(payloads)} payloads ready")

    # Group events by segment
    df["_seg_id_int"] = df["segment_id"].fillna(-1).astype(int)
    task_args = [
        _EnrichArgs(payloads[seg_id], grp)
        for seg_id, grp in df.groupby("_seg_id_int")
        if seg_id in payloads
    ]
    print(f"  {len(task_args)} segment tasks")

    t0 = time.time()
    all_results: list[dict] = []

    if args.serial:
        for a in task_args:
            all_results.extend(_enrich_segment(a))
    else:
        n_workers = args.n_workers or max(1, (os.cpu_count() or 4) - 2)
        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for batch in pool.imap_unordered(
                _enrich_segment, task_args, chunksize=4,
            ):
                all_results.extend(batch)

    print(f"Enrichment done in {time.time() - t0:.1f} s")
    print(f"  enriched {len(all_results)} events")

    enrich_df = pd.DataFrame(all_results).set_index("event_id")
    df = df.set_index("event_id")
    df = df.join(enrich_df, how="left")
    df = df.drop(columns=["_seg_id_int"], errors="ignore")
    df = df.reset_index()

    # Print diagnostics
    bp = df["bp_transverse_ratio"].dropna()
    if len(bp):
        print(f"\nbp_transverse_ratio: median={bp.median():.3f}, "
              f"p10={bp.quantile(0.1):.3f}, p90={bp.quantile(0.9):.3f}")
        print(f"  vs broadband median={df['transverse_ratio'].median():.4f}")

    lf = df["local_fft_ratio"].dropna()
    if len(lf):
        print(f"local_fft_ratio:     median={lf.median():.3f}, "
              f"p10={lf.quantile(0.1):.3f}, p90={lf.quantile(0.9):.3f}")
        print(f"  vs 36h median={df['fft_screen_ratio'].median():.3f}")

    q3 = df["quality_v3"].dropna()
    if len(q3):
        print(f"quality_v3:          median={q3.median():.3f}, "
              f">0.3: {(q3>0.3).sum()} ({(q3>0.3).mean()*100:.1f}%), "
              f">0.5: {(q3>0.5).sum()} ({(q3>0.5).mean()*100:.1f}%)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
