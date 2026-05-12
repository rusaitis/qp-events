r"""Per-cycle vs window-averaged ellipticity comparison.

For every event in ``Output/events_round8.parquet``, recompute the
median per-cycle Stokes ellipticity from the band-passed MFA
transverse components and plot it against the persisted
``ellipticity`` column (window-averaged Stokes V/I over the in-band
CWT slice).

The persisted column averages the full event window — if the
polarization rotates during the event, the window average can wash
out to ~0. The per-cycle median is the more representative summary of
the instantaneous state, and is the quantity closest in spirit to
Fig 10's running cross-correlation classifier.

Output: ``Output/figures/ellipticity_comparison.png`` plus a CSV with
``event_id, band, ellipticity_window, ellipticity_percycle_median,
percycle_iqr`` for further analysis.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import qp
from qp.events.persistence import read_events_parquet
from qp.events.sweep_loader import segment_to_payload
from qp.io import legacy_pickle
from qp.plotting.style import use_paper_style
from qp.signal.polarization import per_oscillation_ellipticity

log = logging.getLogger("diag.ellipticity")


_DT = 60.0  # seconds per sample
_BAND_COLORS = {
    "QP15": "#7f3fbf",
    "QP30": "#dc267f",
    "QP60": "#ffb000",
    "QP120": "#648fff",
}


def _bandpass(
    x: np.ndarray, period_sec: float, *, half_width_frac: float = 0.4,
) -> np.ndarray:
    """4th-order Butterworth zero-phase bandpass around ``1/period_sec``."""
    f0 = 1.0 / period_sec
    nyq = 0.5 / _DT
    f_lo = max(f0 * (1.0 - half_width_frac), 1e-6)
    f_hi = min(f0 * (1.0 + half_width_frac), 0.95 * nyq)
    if f_hi <= f_lo:
        return np.zeros_like(x)
    sos = butter(4, [f_lo / nyq, f_hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _event_per_cycle(
    seg, event: pd.Series,
) -> tuple[float, float, int]:
    """Return (median ellipticity, IQR, n_cycles_used) for one event."""
    payload = segment_to_payload(int(event["segment_idx"]), seg)
    if payload is None:
        return float("nan"), float("nan"), 0
    times = np.asarray(payload.times)
    t_lo = pd.Timestamp(event["date_from"]).to_pydatetime()
    t_hi = pd.Timestamp(event["date_to"]).to_pydatetime()
    mask = (times >= t_lo) & (times <= t_hi)
    if mask.sum() < 60:
        return float("nan"), float("nan"), 0
    period_sec = event["period_min"] * 60.0
    b1 = _bandpass(payload.b_perp1[mask], period_sec)
    b2 = _bandpass(payload.b_perp2[mask], period_sec)
    if b1.size < int(round(period_sec / _DT)) * 2:
        return float("nan"), float("nan"), 0
    median_e, iqr_e = per_oscillation_ellipticity(
        b1, b2, dt=_DT, period=period_sec,
    )
    n_cycles = b1.size // int(round(period_sec / _DT))
    return float(median_e), float(iqr_e), int(n_cycles)


def _compute_all(df: pd.DataFrame) -> pd.DataFrame:
    legacy_pickle.register_stubs()
    log.info("loading MFA segment archive")
    arr = np.load(
        qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy", allow_pickle=True,
    )
    log.info("processing %d events", len(df))
    out_med = np.full(len(df), np.nan)
    out_iqr = np.full(len(df), np.nan)
    out_n = np.zeros(len(df), dtype=int)
    for i, (_, row) in enumerate(df.iterrows()):
        seg_idx = int(row["segment_idx"])
        if not 0 <= seg_idx < len(arr):
            continue
        m, q, n = _event_per_cycle(arr[seg_idx], row)
        out_med[i] = m
        out_iqr[i] = q
        out_n[i] = n
        if (i + 1) % 100 == 0:
            log.info("%d/%d events processed", i + 1, len(df))
    return df.assign(
        ellipticity_percycle_median=out_med,
        ellipticity_percycle_iqr=out_iqr,
        ellipticity_percycle_n=out_n,
    )


def _plot(df: pd.DataFrame, output_path: Path) -> None:
    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    # Panel A: scatter
    ax = axes[0]
    for band, color in _BAND_COLORS.items():
        sub = df[df["band"] == band]
        if sub.empty:
            continue
        ax.scatter(
            sub["ellipticity"],
            sub["ellipticity_percycle_median"],
            s=10,
            alpha=0.55,
            color=color,
            edgecolors="none",
            label=f"{band}  n={len(sub)}",
        )
    valid = (
        df["ellipticity"].notna()
        & df["ellipticity_percycle_median"].notna()
    )
    r = df.loc[valid, "ellipticity"].corr(
        df.loc[valid, "ellipticity_percycle_median"]
    )
    lim = 1.05
    ax.plot([-lim, lim], [-lim, lim], ls=":", color="white", lw=0.8, alpha=0.7)
    ax.axhline(0, color="white", lw=0.4, alpha=0.3)
    ax.axvline(0, color="white", lw=0.4, alpha=0.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"window-averaged ellipticity")
    ax.set_ylabel(r"per-cycle median ellipticity")
    ax.set_title(f"(a) scatter — Pearson r = {r:.3f}", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # Panel B: residual histogram
    ax = axes[1]
    diff = (df["ellipticity_percycle_median"] - df["ellipticity"]).dropna()
    ax.hist(diff, bins=60, color="#888888", alpha=0.85)
    med = float(np.median(diff))
    p16, p84 = np.percentile(diff, [16, 84])
    ax.axvline(0, ls="--", color="white", lw=0.8, alpha=0.6)
    ax.axvline(med, ls="-", color="#dc267f", lw=1.6)
    ax.set_xlabel(r"per-cycle $-$ window-averaged")
    ax.set_ylabel("events")
    ax.set_title(
        f"(b) Δ — median {med:+.3f},  16/84 [{p16:+.3f}, {p84:+.3f}]",
        loc="left",
    )

    # Panel C: |e| distributions per band
    ax = axes[2]
    bands = ["QP15", "QP30", "QP60", "QP120"]
    bins = np.linspace(0, 1, 31)
    for band in bands:
        sub = df[(df["band"] == band) & df["ellipticity_percycle_median"].notna()]
        if sub.empty:
            continue
        ax.hist(
            sub["ellipticity_percycle_median"].abs(),
            bins=bins, histtype="step", lw=1.4, density=True,
            color=_BAND_COLORS[band], label=band,
        )
    ax.axvline(0.5, ls=":", color="white", lw=0.6, alpha=0.6)
    ax.axvline(0.7, ls=":", color="white", lw=0.6, alpha=0.6)
    ax.set_xlabel(r"|per-cycle ellipticity|")
    ax.set_ylabel("density")
    ax.set_title("(c) |e_pc| distribution by band", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle(
        "Per-cycle vs window-averaged Stokes ellipticity — round-8.1 catalogue",
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    log.info("wrote %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parquet = qp.OUTPUT_DIR / "events_round8.parquet"
    df, attrs = read_events_parquet(parquet)
    log.info(
        "loaded %d events (schema=%s)",
        len(df),
        attrs.get("schema_version", "<none>"),
    )
    if "ellipticity" not in df.columns:
        log.error(
            "parquet has no 'ellipticity' column — re-run "
            "sweep_events_round8.py to populate round8.1 schema"
        )
        sys.exit(2)

    df = _compute_all(df)
    csv_path = qp.OUTPUT_DIR / "ellipticity_comparison.csv"
    cols = [
        "event_id", "band", "period_min",
        "ellipticity", "ellipticity_percycle_median",
        "ellipticity_percycle_iqr", "ellipticity_percycle_n",
    ]
    df[cols].to_csv(csv_path, index=False)
    log.info("wrote %s", csv_path)
    _plot(df, qp.OUTPUT_DIR / "figures" / "ellipticity_comparison.png")


if __name__ == "__main__":
    main()
