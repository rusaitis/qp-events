r"""Per-event wavelet diagnostic panel for the QP web app.

Renders the Morlet-CWT scalogram around an event's peak time with the
Bonferroni σ-mask boundary, detection-peak crosshair, QP band edges and
event window overlaid. This is the "evidence" panel that complements the
Welch PSD: Welch can look flat for narrow, localized wave packets, yet
the CWT shows a sharp ridge — and the σ-mask is what actually triggered
detection in :mod:`qp.events.detector`.

The endpoint returns the panel as a PNG (server-side matplotlib render),
which keeps the client logic trivial — just an ``<img>`` tag — while
reusing the project's paper style. The companion ``wavelet_gates`` call
returns the four gate values for the gate-summary tile.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qp.events.bands import QP_BANDS
from qp.events.detector import (
    MIN_Q_FACTOR,
    SEGMENT_FWER_ALPHA,
    bonferroni_n_sigma_for_cwt,
)
from qp.events.threshold import wavelet_sigma_mask
from qp.signal.polarization_config import (
    MAX_MVA_PARALLEL_FRACTION,
    MIN_DEGREE_OF_POLARIZATION,
)
from qp.signal.wavelet import morlet_cwt
from qp.webapp.loaders import (
    _row_for,
    get_segment_payload,
    load_event_table,
)

log = logging.getLogger(__name__)

DT_SEC: float = 60.0
N_FREQS_PANEL: int = 300  # match detector exactly so per-row σ matches the gate.
DEFAULT_HOURS: float = 12.0


@lru_cache(maxsize=512)
def render_wavelet_png(event_id: int, hours_pad: float = DEFAULT_HOURS) -> bytes | None:
    """Return PNG bytes of the wavelet scalogram for an event.

    The figure shows the b_perp1 CWT power (log color), the σ-mask
    boundary as a dashed contour, the QP band edges as horizontal lines,
    the event window as a vertical band, and the detection peak as a
    crosshair. None if the event is unknown or its window is too short.
    """
    panel = _build_panel(event_id, hours_pad=hours_pad)
    if panel is None:
        return None

    period_min = panel["period_min"]
    times_h = panel["times_h"]
    power = panel["power"]
    mask = panel["mask"]
    peak = panel["peak"]

    # Heatmap data: log of mass-normalised power, periods reversed so high
    # periods sit at the top of the y-axis (standard scalogram convention).
    order = np.argsort(period_min)
    period_min = period_min[order]
    power = power[order, :]
    mask = mask[order, :]

    # Display power *normalised by each row's own noise floor* so colours
    # are comparable across periods (raw |CWT| scales with period for red
    # noise, which would otherwise wash out short-period detail). Per-row
    # median is robust to the ≤14% time-coverage of a typical wave
    # packet, so dividing by it puts the noise floor at log10(1)=0 in
    # every row.
    row_med = np.median(np.maximum(power, 1e-12), axis=1, keepdims=True)
    log_pwr = np.log10(np.maximum(power / row_med, 1e-3))

    # Constant colour scale across all events so the same colour means
    # the same signal-to-noise ratio:
    #   vmin = 0     → at the row's noise floor (deep blue)
    #   vmin..0.75   → sub-detection fluctuations
    #   ~0.75        → σ-mask threshold (n_sigma ≈ 4.57 × MAD_TO_SIGMA ×
    #                  MAD/median ≈ 1.0 in log10 units for ~Gaussian
    #                  noise; the actual contour is drawn from the
    #                  boolean σ-mask)
    #   0.75..1.3    → above-threshold ridges
    #   1.3          → saturation (≈ 20× noise floor)
    vmin = 0.0
    vmax = 1.3

    bg_color = "#1c1c1c"
    fg_color = "#e8e8e8"
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=110, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    # pcolormesh edges (cell-centered axes work too, but edges are exact).
    period_edges = _bin_edges_log(period_min)
    time_edges = _bin_edges_linear(times_h)
    # turbo: perceptually-improved rainbow, much wider colour gamut than
    # magma — high-contrast peaks pop, low-power background stays cool.
    mesh = ax.pcolormesh(
        time_edges,
        period_edges,
        log_pwr,
        cmap="turbo",
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )
    cb = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.04)
    cb.set_label(
        r"$\log_{10}\,(|\mathrm{CWT}| / \tilde{|\mathrm{CWT}|}_\mathrm{row})$",
        fontsize=9,
        color=fg_color,
    )
    cb.ax.tick_params(labelsize=8, colors=fg_color)
    cb.outline.set_edgecolor(fg_color)
    # σ-mask threshold marker on the colorbar. The value 0.75 corresponds
    # to ~5x noise floor in linear |CWT| ratio; the actual contour
    # location is set by the per-row Bonferroni-derived n_sigma.
    cb.ax.axhline(0.75, color="#00ffaa", linestyle="--", linewidth=1.1)
    cb.ax.text(
        1.6,
        0.75,
        "σ-mask",
        color="#00ffaa",
        fontsize=7,
        transform=cb.ax.get_yaxis_transform(),
        va="center",
        ha="left",
    )

    # σ-mask boundary as a contour at the half-level of the boolean mask.
    if mask.any():
        ax.contour(
            times_h,
            period_min,
            mask.astype(float),
            levels=[0.5],
            colors="#00ffaa",
            linewidths=1.2,
            linestyles="--",
        )

    # Event window shading.
    win = panel["window_h"]
    if win is not None:
        ax.axvspan(win[0], win[1], color="white", alpha=0.10, zorder=2)
        for x in win:
            ax.axvline(x, color="white", linewidth=0.8, alpha=0.6, zorder=3)

    # Peak crosshair.
    ax.axvline(0.0, color="white", linewidth=1.0, linestyle=":", zorder=4)
    if peak is not None:
        ax.plot(
            peak["time_h"],
            peak["period_min"],
            marker="+",
            color="white",
            markersize=14,
            mew=2.0,
            zorder=5,
        )

    # QP band edges (dashed gray) + labelled centroid lines.
    xlim_for_label = panel.get("xlim_h") or (times_h[0], times_h[-1])
    x_text = xlim_for_label[1] - 0.04 * (xlim_for_label[1] - xlim_for_label[0])
    for band in QP_BANDS.values():
        for edge in (band.period_min_sec / 60.0, band.period_max_sec / 60.0):
            ax.axhline(
                edge, color="#bbbbbb", linewidth=0.5, linestyle=":", alpha=0.6, zorder=2
            )
        centroid_min = band.period_centroid_sec / 60.0
        ax.axhline(
            centroid_min,
            color="#dddddd",
            linewidth=0.6,
            linestyle="-",
            alpha=0.4,
            zorder=2,
        )
        ax.text(
            x_text,
            centroid_min,
            f"{int(centroid_min)} min",
            color=fg_color,
            fontsize=8,
            alpha=0.95,
            ha="right",
            va="center",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=bg_color,
                edgecolor="none",
                alpha=0.6,
            ),
        )

    ax.set_yscale("log")
    ax.set_ylim(period_min.min(), period_min.max())
    xlim = panel.get("xlim_h") or (times_h[0], times_h[-1])
    ax.set_xlim(xlim)
    ax.set_xlabel("time relative to peak [h]", fontsize=9, color=fg_color)
    ax.set_ylabel("period [min]", fontsize=9, color=fg_color)
    ax.tick_params(labelsize=8, colors=fg_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(fg_color)
    ax.set_title(
        f"event {event_id} · CWT scalogram · σ-mask ({panel['n_sigma']:.2f}σ)",
        fontsize=9,
        color=fg_color,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=bg_color)
    plt.close(fig)
    return buf.getvalue()


@lru_cache(maxsize=512)
def wavelet_gates(
    event_id: int, hours_pad: float = DEFAULT_HOURS
) -> dict[str, Any] | None:
    """Gate-summary JSON for the event's wavelet panel.

    Returns the four gate values from the parquet, the Bonferroni σ
    threshold for this segment, the peak σ at the detected period, and
    the canonical thresholds — enough for the UI to colour each gate
    green / red.
    """
    panel = _build_panel(event_id, hours_pad=hours_pad)
    if panel is None:
        return None

    df = load_event_table()
    row = _row_for(df, event_id)
    if row is None:
        return None

    # Sigma over the event window: re-derive from the per-row noise model
    # built on the full 36-h segment (matches the detector's gate
    # exactly). Reports the MAX σ at the peak's period row within
    # [date_from, date_to] — the detector's ridge is a connected
    # component of the mask, so the recorded peak_time is the ridge
    # centroid, not necessarily the brightest pixel. Computed on each
    # transverse component separately (mirroring the detector's
    # per-axis σ-mask) and reported as the larger of the two.
    s1 = _sigma_at_peak(
        panel["power1"],
        panel["freq"],
        panel["times_h"],
        panel["peak"],
        panel["window_h"],
    )
    s2 = _sigma_at_peak(
        panel["power2"],
        panel["freq"],
        panel["times_h"],
        panel["peak"],
        panel["window_h"],
    )
    sigma_at_peak = (
        None
        if (s1 is None and s2 is None)
        else max(x for x in (s1, s2) if x is not None)
    )

    # Polarization geometry was added in the round-8.1 schema (full Stokes
    # vector + derived ellipticity/inclination/polarized fraction). Legacy
    # round-8 parquet files do not have these columns — surface as None so
    # the frontend can show a "—" placeholder instead of crashing.
    def _opt(col: str) -> float | None:
        if col not in row.index:
            return None
        v = row[col]
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    out = {
        "event_id": int(row.event_id),
        "event_uid": str(row.get("event_uid", "")) if "event_uid" in row else "",
        "band": str(row.band),
        "peak_time": row.peak_time.isoformat(),
        "period_min": float(row.period_min),
        "n_sigma_threshold": float(panel["n_sigma"]),
        "sigma_at_peak": sigma_at_peak,
        "gates": {
            "q_factor": float(row.q_factor),
            "mva_par_frac": float(row.mva_par_frac),
            "stokes_d": float(row.stokes_d),
        },
        "polarization": {
            "ellipticity": _opt("ellipticity"),
            "inclination_deg": _opt("inclination_deg"),
            "polarized_fraction": _opt("polarized_fraction"),
            "stokes_i": _opt("stokes_i"),
            "stokes_q": _opt("stokes_q"),
            "stokes_u": _opt("stokes_u"),
            "stokes_v": _opt("stokes_v"),
        },
        "thresholds": {
            "q_factor_min": MIN_Q_FACTOR,
            "mva_par_frac_max": MAX_MVA_PARALLEL_FRACTION,
            "stokes_d_min": MIN_DEGREE_OF_POLARIZATION,
            "fwer_alpha": SEGMENT_FWER_ALPHA,
        },
    }
    if "is_duplicate" in row.index:
        out["is_duplicate"] = bool(row.is_duplicate)
    return out


# ---------------------------------------------------------------------- #
# Internals                                                              #
# ---------------------------------------------------------------------- #


def _build_panel(event_id: int, *, hours_pad: float) -> dict[str, Any] | None:
    """Compute the scalogram and σ-mask for one event.

    Returns a dict with ``freq``, ``period_min``, ``times_h``, ``power``,
    ``mask``, ``n_sigma``, ``peak``, ``window_h``, plus ``xlim_h`` for
    the display crop. The CWT and σ-mask are evaluated over the **full
    36-h segment** so the noise model matches the detector exactly; the
    panel only shows ±``hours_pad`` h around the peak. The CWT is taken
    of ``b_perp1`` only — that is one of the two components the detector
    σ-mask is applied to, and showing one keeps the panel readable.
    """
    df = load_event_table()
    row = _row_for(df, event_id)
    if row is None:
        return None
    seg_idx = int(row.segment_idx)
    payload = get_segment_payload(seg_idx)
    if payload is None:
        return None

    peak_time = row.peak_time.to_pydatetime()
    times = payload.times
    b_perp1 = np.asarray(payload.b_perp1, dtype=float)
    b_perp2 = np.asarray(payload.b_perp2, dtype=float)
    if len(b_perp1) < 64:
        return None
    # Replace any NaNs with the segment mean so the convolution doesn't
    # propagate them across the panel. Gates were computed on the raw
    # segment; this is just for display robustness.
    for arr in (b_perp1, b_perp2):
        if not np.isfinite(arr).all():
            arr[~np.isfinite(arr)] = np.nanmean(arr)

    freq, _, cwt1 = morlet_cwt(b_perp1, dt=DT_SEC, n_freqs=N_FREQS_PANEL)
    _, _, cwt2 = morlet_cwt(b_perp2, dt=DT_SEC, n_freqs=N_FREQS_PANEL)
    power1 = np.abs(cwt1)
    power2 = np.abs(cwt2)
    # Display the max of the two perpendicular powers — detector fires
    # on whichever component triggers, so the max is what catches the
    # eye. The σ computation downstream operates on each component
    # separately (matching the detector's gate) and returns the larger.
    power = np.maximum(power1, power2)
    n_sigma = bonferroni_n_sigma_for_cwt(
        power.shape[1],
        DT_SEC,
        freq,
        alpha=SEGMENT_FWER_ALPHA,
    )
    mask = wavelet_sigma_mask(power1, freq, n_sigma=n_sigma) | wavelet_sigma_mask(
        power2, freq, n_sigma=n_sigma
    )

    period_min = (1.0 / freq) / 60.0
    times_h = np.array(
        [(t - peak_time).total_seconds() / 3600.0 for t in times],
        dtype=float,
    )
    xlim_h = (-hours_pad, hours_pad)

    peak: dict[str, float] | None = None
    if row.period_min and np.isfinite(row.period_min):
        peak = {
            "time_h": 0.0,  # peak_time is the origin of the relative axis
            "period_min": float(row.period_min),
        }

    window_h: tuple[float, float] | None = None
    if row.date_from is not None and row.date_to is not None:
        d_from = (
            row.date_from.to_pydatetime()
            if hasattr(row.date_from, "to_pydatetime")
            else row.date_from
        )
        d_to = (
            row.date_to.to_pydatetime()
            if hasattr(row.date_to, "to_pydatetime")
            else row.date_to
        )
        window_h = (
            (d_from - peak_time).total_seconds() / 3600.0,
            (d_to - peak_time).total_seconds() / 3600.0,
        )

    return {
        "freq": freq,
        "period_min": period_min,
        "times_h": times_h,
        "power": power,  # max(|cwt1|, |cwt2|) for display
        "power1": power1,  # |cwt(b_perp1)| for per-component σ
        "power2": power2,  # |cwt(b_perp2)| for per-component σ
        "mask": mask,  # union of both σ-masks (detector-equivalent)
        "n_sigma": float(n_sigma),
        "peak": peak,
        "window_h": window_h,
        "xlim_h": xlim_h,
    }


def _sigma_at_peak(
    power: np.ndarray,
    freq: np.ndarray,
    times_h: np.ndarray,
    peak: dict[str, float] | None,
    window_h: tuple[float, float] | None,
) -> float | None:
    """Sigma value of the detection peak relative to the per-row noise model.

    Mirrors :func:`wavelet_sigma_mask` exactly: per-row median + MAD over
    rows outside every QP band, then interpolated in log-period to the
    peak's frequency. Returned in Gaussian-equivalent σ (MAD × 1.4826).
    ``None`` if ``peak`` is missing or the period is out of range.
    """
    from qp.events.bands import freq_to_period
    from qp.events.threshold import MAD_TO_SIGMA, _background_row_indices

    if peak is None:
        return None
    target_period_sec = float(peak["period_min"]) * 60.0
    if target_period_sec <= 0:
        return None
    bg_rows = _background_row_indices(freq)
    if bg_rows.size == 0:
        return None
    # Per-row stats on background rows.
    bg_medians = np.median(power[bg_rows], axis=1)
    bg_mads = np.median(
        np.abs(power[bg_rows] - bg_medians[:, None]),
        axis=1,
    )
    if not np.any(bg_mads > 0):
        return None
    periods_sec = freq_to_period(freq)
    log_p_bg = np.log10(periods_sec[bg_rows])
    order = np.argsort(log_p_bg)
    log_p_target = float(np.log10(target_period_sec))
    med_at_peak = float(
        np.interp(
            log_p_target,
            log_p_bg[order],
            bg_medians[order],
        )
    )
    mad_at_peak = float(
        np.interp(
            log_p_target,
            log_p_bg[order],
            bg_mads[order],
        )
    )
    if mad_at_peak <= 0:
        return None

    i_freq = int(np.argmin(np.abs(periods_sec - target_period_sec)))
    # Restrict the time search to the recorded event window. The
    # detector's ridge is a connected component of the σ-mask spanning
    # [date_from, date_to]; the parquet's ``peak_time`` is the ridge's
    # centroid, not necessarily the brightest pixel. Reporting the MAX
    # σ in the window faithfully answers "how strong is the evidence at
    # this period for this event."
    if window_h is None:
        within = np.ones_like(times_h, dtype=bool)
    else:
        within = (times_h >= window_h[0]) & (times_h <= window_h[1])
    if not within.any():
        return None
    peak_power = float(power[i_freq, within].max())
    return (peak_power - med_at_peak) / (MAD_TO_SIGMA * mad_at_peak)


def _bin_edges_log(centers: np.ndarray) -> np.ndarray:
    """Edges for pcolormesh from log-spaced centers."""
    log_c = np.log10(np.maximum(centers, 1e-12))
    log_d = np.diff(log_c)
    edges = np.concatenate(
        [
            [log_c[0] - log_d[0] / 2],
            log_c[:-1] + log_d / 2,
            [log_c[-1] + log_d[-1] / 2],
        ]
    )
    return 10.0**edges


def _bin_edges_linear(centers: np.ndarray) -> np.ndarray:
    d = np.diff(centers)
    return np.concatenate(
        [
            [centers[0] - d[0] / 2],
            centers[:-1] + d / 2,
            [centers[-1] + d[-1] / 2],
        ]
    )
