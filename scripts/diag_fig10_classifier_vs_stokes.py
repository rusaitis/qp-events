r"""Fig 10's per-event polarization classifier vs the persisted Stokes ellipticity.

For every event in ``Output/events_round8.parquet`` (round-8.1 schema),
slice the b_perp1 / b_perp2 components over the event window, run the
same 1-hour sliding cross-correlation classifier that produces Fig 10
(``scripts/fig10_polarization.py:running_phase``), and report the
fraction of valid windows whose phase lag falls within ±30° of ±90°
(the "circular" criterion in Fig 10's ``classify_segment``). Plot
this fraction against the persisted window-averaged Stokes
``ellipticity`` column.

The classifier is a literal port of fig10's algorithm: raw transverse
components (no bandpass), 122-sample window (≈1 h half-width on
either side), peak of the within-window cross-correlation mapped to
degrees via 1 h ↔ 180°. This is the apples-to-apples test of whether
the paper's "predominantly circular" claim survives when applied
event-by-event and stacked against the Stokes geometry now
persisted in the catalogue.

Output: ``Output/figures/fig10_classifier_vs_stokes.png`` and
``Output/fig10_classifier_vs_stokes.csv`` with per-event
``frac_near_90, frac_near_180, n_valid_windows`` alongside the
persisted Stokes columns.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import setup_logging
from scipy.signal import correlate

import qp
from qp.events.persistence import read_events_parquet
from qp.events.sweep_loader import segment_to_payload
from qp.io import legacy_pickle
from qp.plotting.style import use_paper_style

log = logging.getLogger("diag.fig10_vs_stokes")


_DT = 60.0  # seconds per sample
_HALF_WIN = 61  # samples (≈ 1 h half-width)
_BAND_COLORS = {
    "QP15": "#7f3fbf",
    "QP30": "#dc267f",
    "QP60": "#ffb000",
    "QP120": "#648fff",
}


def _running_phase(
    b_perp1: np.ndarray,
    b_perp2: np.ndarray,
    *,
    half_window: int = _HALF_WIN,
) -> np.ndarray:
    """Port of ``scripts/fig10_polarization.py:running_phase``.

    Returns the phase-shift in degrees at each interior sample (NaN
    elsewhere). Phase is the time-lag of the cross-correlation peak
    of ``b_perp2`` against ``b_perp1`` within a ``2·half_window``
    sample window, mapped 1 h ↔ 180° and clamped to ±180°.
    """
    n = len(b_perp1)
    out = np.full(n, np.nan)
    if n < 2 * half_window + 1:
        return out
    for i in range(half_window, n - half_window):
        y1 = b_perp1[i - half_window : i + half_window]
        y2 = b_perp2[i - half_window : i + half_window]
        npts = len(y1)
        c = correlate(y2, y1, mode="same", method="direct")
        delay_h = np.linspace(-npts * _DT, npts * _DT, npts) / 3600.0
        deg = delay_h * 180.0  # 1 h ↔ 180°
        # Mask lags outside ±180° to suppress wraparound peaks.
        mask = (deg < -180) | (deg > 180)
        c[mask] = 0.0
        out[i] = deg[np.argmax(c)]
    return out


def _classify_fractions(phase_deg: np.ndarray) -> tuple[float, float, int]:
    """Return (frac_near_90, frac_near_180, n_valid_windows)."""
    valid = np.isfinite(phase_deg)
    ph = phase_deg[valid]
    n = int(valid.sum())
    if n < 10:
        return float("nan"), float("nan"), n
    near_90 = np.abs(np.abs(ph) - 90) < 30
    near_180 = np.abs(ph) > 150
    return float(near_90.mean()), float(near_180.mean()), n


def _process_event(seg, event: pd.Series) -> tuple[float, float, int]:
    payload = segment_to_payload(int(event["segment_idx"]), seg)
    if payload is None:
        return float("nan"), float("nan"), 0
    times = np.asarray(payload.times)
    t_lo = pd.Timestamp(event["date_from"]).to_pydatetime()
    t_hi = pd.Timestamp(event["date_to"]).to_pydatetime()
    mask = (times >= t_lo) & (times <= t_hi)
    if mask.sum() < 2 * _HALF_WIN + 10:
        return float("nan"), float("nan"), 0
    ph = _running_phase(payload.b_perp1[mask], payload.b_perp2[mask])
    return _classify_fractions(ph)


def _compute_all(df: pd.DataFrame) -> pd.DataFrame:
    legacy_pickle.register_stubs()
    log.info("loading MFA segment archive")
    arr = np.load(
        qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy",
        allow_pickle=True,
    )
    log.info("processing %d events (this is O(N·W) per event, ~30 s total)", len(df))
    f90 = np.full(len(df), np.nan)
    f180 = np.full(len(df), np.nan)
    n_valid = np.zeros(len(df), dtype=int)
    for i, (_, row) in enumerate(df.iterrows()):
        seg_idx = int(row["segment_idx"])
        if not 0 <= seg_idx < len(arr):
            continue
        a, b, n = _process_event(arr[seg_idx], row)
        f90[i] = a
        f180[i] = b
        n_valid[i] = n
        if (i + 1) % 100 == 0:
            log.info("%d/%d events", i + 1, len(df))
    return df.assign(
        fig10_frac_near_90=f90,
        fig10_frac_near_180=f180,
        fig10_n_valid_windows=n_valid,
    )


def _plot(df: pd.DataFrame) -> None:
    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    sub_all = df[
        df["fig10_frac_near_90"].notna()
        & df["ellipticity"].notna()
        & df["fig10_n_valid_windows"].ge(30)
    ]

    # Panel A: fig10 frac_near_90 vs |Stokes ellipticity|
    ax = axes[0]
    for band, color in _BAND_COLORS.items():
        sub = sub_all[sub_all["band"] == band]
        if sub.empty:
            continue
        ax.scatter(
            sub["ellipticity"].abs(),
            sub["fig10_frac_near_90"],
            s=10,
            alpha=0.55,
            color=color,
            edgecolors="none",
            label=f"{band}  n={len(sub)}",
        )
    r = sub_all["ellipticity"].abs().corr(sub_all["fig10_frac_near_90"])
    ax.axhline(0.3, ls=":", color="white", lw=0.6, alpha=0.6)
    ax.set_xlabel(r"persisted |Stokes ellipticity|")
    ax.set_ylabel(r"fig10 fraction of windows near $\pm 90^\circ$")
    ax.set_title(f"(a) Pearson r = {r:.3f}  (n={len(sub_all)})", loc="left")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    # Panel B: classifier-call agreement
    ax = axes[1]
    # fig10 "circular" gate: frac_90 > frac_180 AND frac_90 > 0.3
    is_fig10_circular = (
        sub_all["fig10_frac_near_90"] > sub_all["fig10_frac_near_180"]
    ) & (sub_all["fig10_frac_near_90"] > 0.3)
    is_fig10_linear = (
        sub_all["fig10_frac_near_180"] > sub_all["fig10_frac_near_90"]
    ) & (sub_all["fig10_frac_near_180"] > 0.3)

    labels = ["fig10 circ.", "fig10 linear", "fig10 mixed"]
    fig10_groups = [
        sub_all[is_fig10_circular],
        sub_all[is_fig10_linear],
        sub_all[~is_fig10_circular & ~is_fig10_linear],
    ]
    stokes_circ = [int((g["ellipticity"].abs() > 0.5).sum()) for g in fig10_groups]
    stokes_lin = [int((g["ellipticity"].abs() < 0.2).sum()) for g in fig10_groups]
    stokes_mid = [
        int(len(g)) - sc - sl
        for g, sc, sl in zip(fig10_groups, stokes_circ, stokes_lin, strict=False)
    ]
    x = np.arange(len(labels))
    width = 0.27
    ax.bar(x - width, stokes_circ, width, color="#dc267f", label="Stokes |e|>0.5")
    ax.bar(x, stokes_mid, width, color="#888888", label="Stokes 0.2≤|e|≤0.5")
    ax.bar(x + width, stokes_lin, width, color="#648fff", label="Stokes |e|<0.2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("events")
    ax.set_title("(b) Fig 10 call vs Stokes |e| bin", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    # Annotate counts on each fig10 group
    for i, g in enumerate(fig10_groups):
        ax.text(
            i,
            max(stokes_circ[i], stokes_mid[i], stokes_lin[i]) * 1.04,
            f"N={len(g)}",
            ha="center",
            color="white",
            fontsize=9,
        )

    # Panel C: frac_near_90 distribution
    ax = axes[2]
    bins = np.linspace(0, 1, 41)
    for band, color in _BAND_COLORS.items():
        sub = sub_all[sub_all["band"] == band]
        if sub.empty:
            continue
        ax.hist(
            sub["fig10_frac_near_90"],
            bins=bins,
            histtype="step",
            lw=1.4,
            density=True,
            color=color,
            label=band,
        )
    ax.axvline(0.3, ls=":", color="white", lw=0.6, alpha=0.6)
    ax.set_xlabel("fig10 frac near ±90°")
    ax.set_ylabel("density")
    ax.set_title("(c) classifier distribution by band", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle(
        "Fig 10 polarization classifier vs persisted Stokes ellipticity",
        y=1.02,
    )
    fig.tight_layout()
    out = qp.OUTPUT_DIR / "figures" / "fig10_classifier_vs_stokes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    log.info("wrote %s", out)


def main() -> None:
    setup_logging(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df, attrs = read_events_parquet(qp.OUTPUT_DIR / "events_round8.parquet")
    log.info(
        "loaded %d events (schema=%s)",
        len(df),
        attrs.get("schema_version", "<none>"),
    )
    df = _compute_all(df)
    csv = qp.OUTPUT_DIR / "fig10_classifier_vs_stokes.csv"
    cols = [
        "event_id",
        "band",
        "period_min",
        "ellipticity",
        "fig10_frac_near_90",
        "fig10_frac_near_180",
        "fig10_n_valid_windows",
    ]
    df[cols].to_csv(csv, index=False)
    log.info("wrote %s", csv)
    _plot(df)


if __name__ == "__main__":
    main()
