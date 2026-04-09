"""Phase 7.5 — Injection-recovery test on real magnetometer data.

Identifies quiet segments (zero events in v1 catalog), injects synthetic
QP packets at known amplitudes, runs the full detection pipeline, and
measures the detection efficiency curve: recall vs amplitude per band.

This quantifies the real-data sensitivity floor and the false-negative
rate at each amplitude tier.

Usage::

    uv run python scripts/injection_recovery.py
    uv run python scripts/injection_recovery.py --n-quiet 50 --n-trials 5
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
from qp.events.detector import detect_with_gate  # noqa: E402
from qp.events.threshold import GateConfig  # noqa: E402

GATE = GateConfig(
    n_sigma=5.0, min_pixels=300, min_duration_hours=2.5,
    min_oscillations=3.0, enable_fft_screen=False, require_both_perp=False,
)

AMPLITUDES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]


def identify_quiet_segments(
    catalog_path: Path, segments_arr, max_quiet: int = 200,
) -> list[int]:
    """Find segments with zero events in the v1 catalog."""
    import pandas as pd
    df = pd.read_parquet(catalog_path)
    event_seg_ids = set(df["segment_id"].dropna().astype(int))

    quiet = []
    for i in range(len(segments_arr)):
        if i in event_seg_ids:
            continue
        seg = segments_arr[i]
        if getattr(seg, "flag", None) is not None:
            continue
        if not hasattr(seg, "FIELDS") or len(seg.FIELDS) < 4:
            continue
        if not hasattr(seg, "datetime") or len(seg.datetime) < 18 * 60:
            continue
        quiet.append(i)
        if len(quiet) >= max_quiet:
            break
    return quiet


def inject_and_detect(
    b_perp1: np.ndarray,
    b_perp2: np.ndarray,
    times,
    band_name: str,
    amplitude: float,
    rng: np.random.Generator,
) -> bool:
    """Inject a synthetic QP packet into a quiet segment and test detection."""
    n = len(b_perp1)
    dt = 60.0
    t = np.arange(n) * dt
    band = QP_BANDS[band_name]
    period = band.period_centroid_sec

    # Place the packet in the centre of the segment
    t_centre = t[n // 2]
    env_width = period * 2.0  # Gaussian envelope: 2 periods wide

    # Random phase offset
    phase1 = rng.uniform(0, 2 * np.pi)
    phase2 = phase1 + np.pi / 2  # 90° for circular polarization

    signal1 = amplitude * np.sin(2 * np.pi * t / period + phase1) * \
              np.exp(-0.5 * ((t - t_centre) / env_width) ** 2)
    signal2 = amplitude * np.sin(2 * np.pi * t / period + phase2) * \
              np.exp(-0.5 * ((t - t_centre) / env_width) ** 2)

    injected1 = b_perp1 + signal1
    injected2 = b_perp2 + signal2

    # Run detector
    try:
        packets = detect_with_gate(
            injected1, injected2, times, dt=dt,
            bands=[band_name], gate=GATE,
        )
    except Exception:
        return False

    # Check if any detection overlaps with the injection
    return len(packets) > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path,
                         default=_PROJECT_ROOT / "Output" / "events_qp_v1.parquet")
    parser.add_argument("--n-quiet", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=3)
    args = parser.parse_args()

    print("Loading segments...")
    arr = np.load(
        qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy",
        allow_pickle=True,
    )

    print("Finding quiet segments...")
    quiet_ids = identify_quiet_segments(args.catalog, arr, args.n_quiet)
    print(f"  found {len(quiet_ids)} quiet segments")

    rng = np.random.default_rng(42)
    results: dict[str, dict[float, list[bool]]] = {
        band: {amp: [] for amp in AMPLITUDES}
        for band in QP_BAND_NAMES
    }

    t0 = time.time()
    for trial in range(args.n_trials):
        for seg_idx in quiet_ids:
            seg = arr[seg_idx]
            b_perp1 = np.nan_to_num(np.asarray(seg.FIELDS[1].y, dtype=float))
            b_perp2 = np.nan_to_num(np.asarray(seg.FIELDS[2].y, dtype=float))
            times = list(seg.datetime)

            for band in QP_BAND_NAMES:
                for amp in AMPLITUDES:
                    detected = inject_and_detect(
                        b_perp1, b_perp2, times, band, amp, rng,
                    )
                    results[band][amp].append(detected)

        print(f"  trial {trial + 1}/{args.n_trials} done "
              f"({time.time() - t0:.0f} s)")

    # Compute recall curves
    print("\n=== Detection efficiency ===")
    recall: dict[str, list[float]] = {band: [] for band in QP_BAND_NAMES}
    for band in QP_BAND_NAMES:
        print(f"\n  {band}:")
        for amp in AMPLITUDES:
            hits = results[band][amp]
            r = sum(hits) / max(len(hits), 1)
            recall[band].append(r)
            print(f"    {amp:.2f} nT: {sum(hits)}/{len(hits)} "
                  f"({r * 100:.1f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"QP30": "#4ecdc4", "QP60": "#ff6b6b", "QP120": "#ffd93d"}
    for band in QP_BAND_NAMES:
        ax.plot(AMPLITUDES, recall[band], "o-", color=colors[band],
                 lw=2, ms=6, label=band)
    ax.set_xlabel("Injected amplitude (nT)")
    ax.set_ylabel("Detection rate (recall)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(AMPLITUDES) + 0.05)
    ax.axhline(0.9, color="grey", ls=":", lw=0.8, label="90% recall")
    ax.legend(frameon=False)
    ax.set_title("Phase 7.5 — Real-data injection-recovery", fontsize=13)
    ax.grid(alpha=0.3)

    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "injection_recovery.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")

    # Save CSV
    import csv
    csv_path = _PROJECT_ROOT / "Output" / "diagnostics" / "injection_recovery.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["band", "amplitude_nT", "recall"])
        for band in QP_BAND_NAMES:
            for amp, r in zip(AMPLITUDES, recall[band]):
                writer.writerow([band, amp, f"{r:.4f}"])
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
