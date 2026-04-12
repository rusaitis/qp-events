"""Phase 9.3 — Waveform gallery: top 9 events per band.

For each QP band shows a 3×3 grid with:
- Top row: raw b_perp1 (grey) + Hilbert amplitude envelope (colour)
- Middle row: instantaneous frequency vs time
- Bottom row: per-oscillation amplitude bar chart (growing/decaying)

Output: ``Output/figures/figure_waveform_gallery_{band}.png``
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _register_pickle_stubs() -> None:
    stub_classes = ["SignalSnapshot", "NewSignal", "Interval", "FFT_list",
                    "WaveSignal", "Wave"]
    stub_modules = ["__main__", "data_sweeper", "mag_fft_sweeper",
                    "cassinilib", "cassinilib.NewSignal"]
    for mod_path in stub_modules:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)
        for cls in stub_classes:
            setattr(sys.modules[mod_path], cls, type(cls, (), {}))


_register_pickle_stubs()

import datetime  # noqa: E402

from qp.events.bands import QP_BANDS  # noqa: E402
from qp.plotting.style import use_paper_style  # noqa: E402
from qp.signal.morphology import (  # noqa: E402
    amplitude_growth_rate,
    band_envelope,
    instantaneous_frequency,
)

from sweep_events import load_segments, segment_to_payload  # noqa: E402

DT = 60.0
BAND_COLORS = {"QP30": "#4ecdc4", "QP60": "#ff6b6b", "QP120": "#ffd93d"}


def _make_panel(
    ax_wave: plt.Axes,
    ax_freq: plt.Axes,
    ax_amp: plt.Axes,
    b_perp1: np.ndarray,
    times: list[datetime.datetime],
    row: pd.Series,
    color: str,
) -> None:
    """Fill one column's three sub-panels for a single event."""
    epoch = datetime.datetime(1970, 1, 1)
    time_unix = np.array([(t - epoch).total_seconds() for t in times])
    t_from = datetime.datetime.fromisoformat(str(row["date_from"]))
    t_to = datetime.datetime.fromisoformat(str(row["date_to"]))
    i_from = int(np.argmin(np.abs(time_unix - (t_from - epoch).total_seconds())))
    i_to = int(np.argmin(np.abs(time_unix - (t_to - epoch).total_seconds())))
    sl = slice(max(0, i_from - 30), min(len(b_perp1), i_to + 30))

    snippet = b_perp1[sl]
    t_min = (np.arange(len(snippet)) - (i_from - max(0, i_from - 30))) * DT / 60.0

    band_name = str(row["band"])
    if band_name not in QP_BANDS:
        return
    band = QP_BANDS[band_name]
    period_sec = float(row["period"]) if pd.notna(row.get("period")) else band.period_centroid_sec

    env = band_envelope(snippet, DT, band.freq_min_hz, band.freq_max_hz)
    f_inst = instantaneous_frequency(
        band_envelope.__module__ and snippet or snippet, DT
    )
    # Use band-passed signal for instantaneous frequency
    from qp.signal.morphology import _bandpass
    filtered = _bandpass(snippet, band.freq_min_hz, band.freq_max_hz, 1.0 / DT)
    f_inst = instantaneous_frequency(filtered, DT)
    f_inst_mhz = f_inst * 1e6  # μHz for display

    # Panel 1: waveform + envelope
    ax_wave.plot(t_min, snippet, color="0.5", lw=0.5, alpha=0.7)
    ax_wave.plot(t_min, env, color=color, lw=1.2)
    ax_wave.plot(t_min, -env, color=color, lw=1.2, alpha=0.4)
    ax_wave.axvline(0, color="white", lw=0.5, ls=":")
    ax_wave.axvline((i_to - i_from) * DT / 60.0, color="white", lw=0.5, ls=":")
    ax_wave.set_ylabel("B (nT)", fontsize=7)
    ax_wave.set_title(
        f"{row['date_from'][:10]}  n={row.get('n_oscillations', '?'):.1f} cyc",
        fontsize=7,
    )
    ax_wave.tick_params(labelsize=6)

    # Panel 2: instantaneous frequency
    f_center = 1.0 / period_sec * 1e6
    ax_freq.plot(t_min, f_inst_mhz, color=color, lw=0.8, alpha=0.8)
    ax_freq.axhline(f_center, color="white", lw=0.5, ls="--", alpha=0.5)
    ax_freq.set_ylabel("f (μHz)", fontsize=7)
    ax_freq.set_ylim(
        band.freq_min_hz * 1e6 * 0.8,
        band.freq_max_hz * 1e6 * 1.2,
    )
    ax_freq.tick_params(labelsize=6)

    # Panel 3: per-oscillation amplitude
    spc = max(4, int(round(period_sec / DT)))
    n_cycles = len(env) // spc
    cycle_rms = []
    for i in range(n_cycles):
        chunk = env[i * spc:(i + 1) * spc]
        cycle_rms.append(float(np.sqrt(np.mean(chunk ** 2))))
    if cycle_rms:
        x_cycles = np.arange(len(cycle_rms))
        ax_amp.bar(x_cycles, cycle_rms, color=color, alpha=0.8, width=0.8)
        if len(cycle_rms) >= 2:
            slope = float(np.polyfit(x_cycles, np.log10(np.maximum(cycle_rms, 1e-9)), 1)[0])
            growth_db = slope * 20.0
            ax_amp.set_title(f"{growth_db:+.1f} dB/period", fontsize=7)
    ax_amp.set_xlabel("Cycle #", fontsize=7)
    ax_amp.set_ylabel("|env| RMS", fontsize=7)
    ax_amp.tick_params(labelsize=6)


def make_gallery(band_name: str, df: pd.DataFrame, segments, out_dir: Path) -> None:
    quality_col = "quality_v3" if "quality_v3" in df.columns else "quality"
    sub = (
        df[(df.band == band_name) & df[quality_col].notna()]
        .sort_values(quality_col, ascending=False)
        .head(9)
    )
    if len(sub) == 0:
        print(f"  {band_name}: no events")
        return

    color = BAND_COLORS.get(band_name, "white")
    use_paper_style()
    n_events = min(len(sub), 9)
    n_cols = min(n_events, 3)
    n_rows_groups = (n_events + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 8 * n_rows_groups), constrained_layout=True)
    outer_gs = gridspec.GridSpec(n_rows_groups, n_cols, figure=fig)

    event_idx = 0
    for row_grp in range(n_rows_groups):
        for col_grp in range(n_cols):
            if event_idx >= n_events:
                break
            row = sub.iloc[event_idx]
            event_idx += 1

            inner = outer_gs[row_grp, col_grp].subgridspec(3, 1, hspace=0.05)
            ax_wave = fig.add_subplot(inner[0])
            ax_freq = fig.add_subplot(inner[1], sharex=ax_wave)
            ax_amp = fig.add_subplot(inner[2])

            seg_id = int(row["segment_id"]) if pd.notna(row.get("segment_id")) else -1
            if seg_id < 0 or seg_id >= len(segments):
                continue
            payload = segment_to_payload(seg_id, segments[seg_id])
            if payload is None:
                continue

            try:
                _make_panel(ax_wave, ax_freq, ax_amp,
                             payload.b_perp1, payload.times, row, color)
            except Exception as exc:
                ax_wave.set_title(f"error: {exc}", fontsize=6)

    fig.suptitle(
        f"Waveform Gallery — {band_name} (top {n_events} by quality)",
        fontsize=12,
    )
    out = out_dir / f"figure_waveform_gallery_{band_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


def main() -> None:
    cat_path = _PROJECT_ROOT / "Output" / "events_qp_v4.parquet"
    if not cat_path.exists():
        cat_path = _PROJECT_ROOT / "Output" / "events_qp_v3.parquet"
    df = pd.read_parquet(cat_path)
    print(f"Loaded {len(df)} events from {cat_path.name}")

    print("Loading segments...")
    segments, _ = load_segments(year=None)

    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for band in ["QP30", "QP60", "QP120"]:
        print(f"Processing {band}...")
        make_gallery(band, df, segments, out_dir)


if __name__ == "__main__":
    main()
