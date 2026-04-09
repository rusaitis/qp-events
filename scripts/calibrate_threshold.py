"""Phase 2.4 — calibrate the QP detection threshold via synthetic injection.

For each QP band and a grid of injection amplitudes:

1. Build N synthetic 36-h segments (Gaussian-windowed packet on top of
   AR(1) red noise).
2. Build N background-only segments (red noise alone).
3. Run :func:`qp.events.detector.detect_with_gate` on both populations.
4. Record the recall (fraction of injected packets recovered in the
   right band) and the false-positive rate (fraction of background
   segments yielding a spurious packet in any band).
5. For each band, find the smallest n_sigma value at which
   ``recall ≥ 0.9`` and ``FPR ≤ 0.01``.

Outputs
-------
- ``Output/diagnostics/threshold_calibration.csv`` — full sweep table
- ``Output/diagnostics/threshold_calibration.png`` — recall/FPR curves
- ``Output/diagnostics/threshold_calibration.md`` — chosen n_sigma per
  band, with rationale

This script is fast (~1 minute on a laptop) and is the canonical
source for the production threshold values used by Phase 3's
mission-wide sweep.
"""

from __future__ import annotations

import csv
import datetime
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qp.events.bands import QP_BANDS
from qp.events.catalog import WaveTemplate
from qp.events.detector import detect_with_gate
from qp.events.threshold import GateConfig
from qp.signal.synthetic import simulate_signal


# ----------------------------------------------------------------------
# Synthetic signal generators
# ----------------------------------------------------------------------


def make_red_noise(n: int, sigma: float = 0.05, alpha: float = 0.95,
                    seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + e[i]
    return x


def make_packet(period_min: float, amplitude: float, n: int, dt: float,
                  decay_hours: float = 4.0,
                  center_hours: float = 18.0,
                  seed: int = 1) -> np.ndarray:
    wave = WaveTemplate(
        period=period_min * 60.0,
        amplitude=amplitude,
        decay_width=decay_hours * 3600.0,
        shift=center_hours * 3600.0,
    )
    _, sig = simulate_signal(
        n_samples=n, dt=dt, waves=[wave], seed=seed,
    )
    return sig


# ----------------------------------------------------------------------
# Calibration sweep
# ----------------------------------------------------------------------


@dataclass
class CalibrationRow:
    band: str
    period_min: float
    amplitude: float
    n_sigma: float
    n_trials: int
    recall: float
    fpr: float
    avg_packets_per_inj: float
    avg_packets_per_bg: float


def run_band_calibration(
    band_name: str,
    period_min: float,
    amplitudes: list[float],
    sigma_values: list[float],
    n_trials: int = 30,
    n_samples: int = 2160,
    dt: float = 60.0,
) -> list[CalibrationRow]:
    rows: list[CalibrationRow] = []
    t0 = datetime.datetime(2007, 1, 1)
    times = [t0 + datetime.timedelta(seconds=i * dt) for i in range(n_samples)]

    for amp in amplitudes:
        for sigma in sigma_values:
            gate = GateConfig(
                fft_ratio_threshold=2.5,
                n_sigma=sigma,
                min_duration_hours=2.5,
                min_pixels=300,
                min_oscillations=3.0,
                enable_fft_screen=False,
            )
            n_recovered = 0
            n_packets_inj = 0
            n_packets_bg = 0
            n_fp_segments = 0
            for trial in range(n_trials):
                # Injection
                noise1 = make_red_noise(n_samples, seed=2 * trial)
                noise2 = make_red_noise(n_samples, seed=2 * trial + 1)
                sig1 = make_packet(period_min, amp, n_samples, dt,
                                    seed=trial) + noise1
                sig2 = make_packet(period_min, 0.7 * amp, n_samples, dt,
                                    seed=trial + 5000) + noise2
                pkts = detect_with_gate(sig1, sig2, times, dt=dt, gate=gate)
                n_packets_inj += len(pkts)
                if any(p.band == band_name for p in pkts):
                    n_recovered += 1

                # Background only
                bg1 = make_red_noise(n_samples, seed=10_000 + 2 * trial)
                bg2 = make_red_noise(n_samples, seed=10_000 + 2 * trial + 1)
                bg_pkts = detect_with_gate(bg1, bg2, times, dt=dt, gate=gate)
                n_packets_bg += len(bg_pkts)
                if len(bg_pkts) > 0:
                    n_fp_segments += 1

            recall = n_recovered / n_trials
            fpr = n_fp_segments / n_trials
            rows.append(
                CalibrationRow(
                    band=band_name,
                    period_min=period_min,
                    amplitude=amp,
                    n_sigma=sigma,
                    n_trials=n_trials,
                    recall=recall,
                    fpr=fpr,
                    avg_packets_per_inj=n_packets_inj / n_trials,
                    avg_packets_per_bg=n_packets_bg / n_trials,
                )
            )
    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def pick_best_sigma(
    rows: list[CalibrationRow],
    band: str,
    target_recall: float = 0.9,
    max_fpr: float = 0.01,
) -> tuple[float, float, float, str] | None:
    """Smallest n_sigma at which recall>=target and fpr<=max_fpr.

    Aggregates over the highest amplitude tier (the calibration is
    really a function of n_sigma; amplitude is just the lever to
    decide what "detectable" means).
    """
    band_rows = [r for r in rows if r.band == band]
    if not band_rows:
        return None
    max_amp = max(r.amplitude for r in band_rows)
    candidates = [r for r in band_rows if r.amplitude == max_amp]
    candidates.sort(key=lambda r: r.n_sigma)
    for r in candidates:
        if r.recall >= target_recall and r.fpr <= max_fpr:
            return r.n_sigma, r.recall, r.fpr, "recall>=0.9 & fpr<=0.01"
    # Fallback: pick highest recall with fpr<=max_fpr; otherwise lowest fpr.
    valid = [r for r in candidates if r.fpr <= max_fpr]
    if valid:
        best = max(valid, key=lambda r: r.recall)
        return best.n_sigma, best.recall, best.fpr, (
            f"fpr<={max_fpr} but recall<{target_recall} (best: "
            f"recall={best.recall:.2f})"
        )
    best = min(candidates, key=lambda r: r.fpr)
    return best.n_sigma, best.recall, best.fpr, (
        f"no n_sigma achieves fpr<={max_fpr}; best fpr={best.fpr:.2f}"
    )


def write_csv(rows: list[CalibrationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "band", "period_min", "amplitude_nT", "n_sigma", "n_trials",
            "recall", "fpr", "avg_packets_inj", "avg_packets_bg",
        ])
        for r in rows:
            writer.writerow([
                r.band, r.period_min, r.amplitude, r.n_sigma, r.n_trials,
                f"{r.recall:.4f}", f"{r.fpr:.4f}",
                f"{r.avg_packets_per_inj:.3f}",
                f"{r.avg_packets_per_bg:.3f}",
            ])


def plot_curves(rows: list[CalibrationRow], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    for col, band in enumerate(("QP30", "QP60", "QP120")):
        ax_r = axes[0, col]
        ax_f = axes[1, col]
        band_rows = [r for r in rows if r.band == band]
        amps = sorted({r.amplitude for r in band_rows})
        for amp in amps:
            xs = sorted({r.n_sigma for r in band_rows if r.amplitude == amp})
            recall = [
                next(r for r in band_rows
                     if r.amplitude == amp and r.n_sigma == s).recall
                for s in xs
            ]
            fpr = [
                next(r for r in band_rows
                     if r.amplitude == amp and r.n_sigma == s).fpr
                for s in xs
            ]
            ax_r.plot(xs, recall, "o-", label=f"{amp:.2f} nT")
            ax_f.plot(xs, fpr, "o-", label=f"{amp:.2f} nT")
        ax_r.set_title(band)
        ax_r.axhline(0.9, ls="--", color="grey", lw=0.7)
        ax_f.axhline(0.01, ls="--", color="grey", lw=0.7)
        ax_r.set_ylabel("recall" if col == 0 else "")
        ax_f.set_ylabel("FPR" if col == 0 else "")
        ax_f.set_xlabel(r"$n_\sigma$")
        ax_r.set_ylim(-0.02, 1.05)
        ax_f.set_ylim(-0.005, 0.6)
        ax_r.grid(alpha=0.3)
        ax_f.grid(alpha=0.3)
        ax_r.legend(fontsize=7, loc="lower left")
    fig.suptitle("Phase 2.4 — threshold calibration sweep "
                 "(red noise + Gaussian QP packet)", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    rows: list[CalibrationRow],
    chosen: dict[str, tuple[float, float, float, str]],
    path: Path,
    duration_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 2.4 — threshold calibration summary")
    lines.append("")
    lines.append(f"Generated: {datetime.datetime.utcnow().isoformat()}Z")
    lines.append(f"Wall time: {duration_sec:.1f} s")
    lines.append("")
    lines.append(
        "Method: for each band, inject a Gaussian-windowed packet "
        "(decay 4 h, centered at 18 h of a 36-h segment) on top of "
        "AR(1) red noise (alpha=0.95, sigma=0.05 nT). Build a "
        "background-only twin from independent noise. Run "
        "`detect_with_gate` on both, recording recall (fraction of "
        "injections recovered in the right band) and FPR (fraction "
        "of background segments yielding any packet)."
    )
    lines.append("")
    lines.append("## Chosen n_sigma per band (target: recall>=0.9, FPR<=0.01)")
    lines.append("")
    lines.append("| Band | Chosen n_sigma | Recall | FPR | Note |")
    lines.append("|------|---------------:|-------:|----:|------|")
    for band in ("QP30", "QP60", "QP120"):
        if band in chosen:
            sigma, recall, fpr, note = chosen[band]
            lines.append(
                f"| {band} | {sigma:.2f} | {recall:.2f} | {fpr:.3f} | {note} |"
            )
        else:
            lines.append(f"| {band} | n/a | n/a | n/a | calibration failed |")
    lines.append("")
    lines.append(
        "These values feed `GateConfig` in the Phase 3 mission sweep "
        "(`scripts/sweep_events.py`)."
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    out_dir = _PROJECT_ROOT / "Output" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    band_amplitudes = {
        "QP30": [0.10, 0.20, 0.40, 0.80],
        "QP60": [0.10, 0.20, 0.40, 0.80],
        "QP120": [0.10, 0.20, 0.40, 0.80],
    }
    sigma_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    rows: list[CalibrationRow] = []
    t_start = time.time()
    for band_name, band in QP_BANDS.items():
        print(f"  Calibrating {band_name} (period={band.period_centroid_minutes} min)...",
              flush=True)
        rows.extend(
            run_band_calibration(
                band_name=band_name,
                period_min=band.period_centroid_minutes,
                amplitudes=band_amplitudes[band_name],
                sigma_values=sigma_values,
                n_trials=20,
            )
        )
    duration = time.time() - t_start
    print(f"Calibration finished in {duration:.1f} s ({len(rows)} rows)")

    chosen = {}
    for band in ("QP30", "QP60", "QP120"):
        c = pick_best_sigma(rows, band)
        if c is not None:
            chosen[band] = c
            print(f"  {band}: n_sigma={c[0]:.2f}, recall={c[1]:.2f}, "
                  f"fpr={c[2]:.3f}  ({c[3]})")

    csv_path = out_dir / "threshold_calibration.csv"
    png_path = out_dir / "threshold_calibration.png"
    md_path = out_dir / "threshold_calibration.md"
    write_csv(rows, csv_path)
    plot_curves(rows, png_path)
    write_summary(rows, chosen, md_path, duration)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
