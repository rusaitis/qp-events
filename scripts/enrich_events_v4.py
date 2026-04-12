"""Phase 9 — Post-hoc morphology enrichment of the v3 event catalog.

Adds waveform shape metrics to each event:

- ``envelope_skewness``   — skewness of the Hilbert amplitude envelope
- ``rise_fall_ratio``     — rise time / fall time (>1 = slow rise, fast fall)
- ``harmonic_ratio_2f``   — P(2f)/P(f); high → non-sinusoidal (sawtooth-like)
- ``amplitude_growth_db`` — dB/period slope of per-oscillation amplitude
- ``freq_drift_hz_per_s`` — linear chirp rate from instantaneous frequency
- ``inter_cycle_coherence`` — mean correlation between successive cycles
- ``ppo_phase_onset_deg`` — SLS5N phase at event *start* (date_from)
- ``waveform_skewness``   — skewness of the raw waveform (asymmetric cycles)

Output: ``Output/events_qp_v4.parquet``

Usage::

    uv run python scripts/enrich_events_v4.py
    uv run python scripts/enrich_events_v4.py --serial   # debug
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import types
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew

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

from qp.events.bands import QP_BANDS  # noqa: E402
from qp.signal.morphology import (  # noqa: E402
    amplitude_growth_rate,
    band_envelope,
    envelope_rise_fall,
    freq_drift_rate,
    harmonic_ratio,
    inter_cycle_coherence,
)

from sweep_events import load_segments, SegmentPayload, segment_to_payload  # noqa: E402

DT = 60.0  # seconds


# ---------------------------------------------------------------------------
# PPO phase interpolation at arbitrary time
# ---------------------------------------------------------------------------

def _ppo_at_time(
    info: dict,
    target_time: datetime.datetime,
    seg_t0: datetime.datetime,
    n_samples: int,
) -> float | None:
    """Interpolate SLS5N phase at ``target_time`` from segment info."""
    arr = info.get("SLS5N")
    if arr is None or len(arr) == 0:
        return None
    elapsed_min = (target_time - seg_t0).total_seconds() / 60.0
    cadence = n_samples / len(arr)
    idx = int(round(elapsed_min / cadence))
    idx = max(0, min(len(arr) - 1, idx))
    try:
        return float(arr[idx])
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-event morphology
# ---------------------------------------------------------------------------

@dataclass
class _MorphArgs:
    payload: SegmentPayload
    events_df: pd.DataFrame


def _morph_segment(arg: _MorphArgs) -> list[dict]:
    """Worker: compute morphology metrics for all events in one segment."""
    payload = arg.payload
    evs = arg.events_df
    if len(evs) == 0:
        return []

    b_perp1 = payload.b_perp1
    times = payload.times
    n = len(times)
    epoch = datetime.datetime(1970, 1, 1)
    time_unix = np.array([(t - epoch).total_seconds() for t in times], dtype=float)
    seg_t0 = times[0]

    results = []
    for _, row in evs.iterrows():
        t_from = datetime.datetime.fromisoformat(str(row["date_from"]))
        t_to = datetime.datetime.fromisoformat(str(row["date_to"]))
        i_from = int(np.argmin(np.abs(time_unix - (t_from - epoch).total_seconds())))
        i_to = int(np.argmin(np.abs(time_unix - (t_to - epoch).total_seconds())))
        sl = slice(i_from, min(i_to + 1, n))

        period_sec = float(row["period"]) if pd.notna(row.get("period")) else None
        band_name = str(row["band"]) if pd.notna(row.get("band")) else None

        rec: dict = {"event_id": row["event_id"]}

        if period_sec is None or period_sec <= 0 or band_name not in QP_BANDS:
            results.append(rec)
            continue

        band = QP_BANDS[band_name]
        low_hz = band.freq_min_hz
        high_hz = band.freq_max_hz
        snippet = b_perp1[sl]

        if len(snippet) < 8:
            results.append(rec)
            continue

        # --- Amplitude envelope ---
        try:
            env = band_envelope(snippet, DT, low_hz, high_hz)
            rec["envelope_skewness"] = float(scipy_skew(env))
            rf = envelope_rise_fall(env, DT)
            if rf is not None:
                rec["rise_fall_ratio"] = rf[2]
            rec["amplitude_growth_db"] = amplitude_growth_rate(env, DT, period_sec)
        except Exception:
            pass

        # --- Harmonic content ---
        try:
            rec["harmonic_ratio_2f"] = harmonic_ratio(snippet, DT, period_sec)
        except Exception:
            pass

        # --- Frequency drift ---
        try:
            rec["freq_drift_hz_per_s"] = freq_drift_rate(snippet, DT, low_hz, high_hz)
        except Exception:
            pass

        # --- Inter-cycle coherence ---
        try:
            rec["inter_cycle_coherence"] = inter_cycle_coherence(snippet, DT, period_sec)
        except Exception:
            pass

        # --- Raw waveform skewness ---
        try:
            rec["waveform_skewness"] = float(scipy_skew(snippet))
        except Exception:
            pass

        # --- PPO phase at event onset ---
        try:
            rec["ppo_phase_onset_deg"] = _ppo_at_time(
                payload.info, t_from, seg_t0, n,
            )
        except Exception:
            pass

        results.append(rec)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=_PROJECT_ROOT / "Output" / "events_qp_v3.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=_PROJECT_ROOT / "Output" / "events_qp_v4.parquet",
    )
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()

    print(f"Loading catalog: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  {len(df)} events across {df['segment_id'].nunique()} segments")

    print("Loading segments...")
    segments, _ = load_segments(year=None)

    seg_ids_needed = set(df["segment_id"].dropna().astype(int))
    print(f"  building payloads for {len(seg_ids_needed)} active segments...")
    payloads: dict[int, SegmentPayload] = {}
    for i in seg_ids_needed:
        if i < len(segments):
            p = segment_to_payload(i, segments[i])
            if p is not None:
                payloads[i] = p
    print(f"  {len(payloads)} payloads ready")

    df["_seg_id_int"] = df["segment_id"].fillna(-1).astype(int)
    tasks = [
        _MorphArgs(payloads[sid], grp)
        for sid, grp in df.groupby("_seg_id_int")
        if sid in payloads
    ]
    print(f"  {len(tasks)} segment tasks")

    t0 = time.time()
    all_results: list[dict] = []

    if args.serial:
        for task in tasks:
            all_results.extend(_morph_segment(task))
    else:
        n_workers = args.n_workers or max(1, (os.cpu_count() or 4) - 2)
        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for batch in pool.imap_unordered(_morph_segment, tasks, chunksize=4):
                all_results.extend(batch)

    elapsed = time.time() - t0
    print(f"Morphology enrichment done in {elapsed:.1f} s ({len(all_results)} events)")

    enrich_df = pd.DataFrame(all_results).set_index("event_id")
    df = df.set_index("event_id")
    df = df.join(enrich_df, how="left")
    df = df.drop(columns=["_seg_id_int"], errors="ignore")
    df = df.reset_index()

    # Diagnostics
    new_cols = [
        "envelope_skewness", "rise_fall_ratio", "harmonic_ratio_2f",
        "amplitude_growth_db", "freq_drift_hz_per_s",
        "inter_cycle_coherence", "ppo_phase_onset_deg", "waveform_skewness",
    ]
    for col in new_cols:
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                print(f"  {col:30s}: n={len(s):4d}  "
                      f"med={s.median():.4g}  "
                      f"p10={s.quantile(0.1):.4g}  "
                      f"p90={s.quantile(0.9):.4g}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nWrote {args.output}")
    print(f"  columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
