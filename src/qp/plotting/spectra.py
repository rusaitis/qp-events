"""Spectral plotting: power spectra, power ratios, spectrograms.

Extracted and cleaned from the monolithic PlotFFT.py.
All functions take plain numpy arrays.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy.typing import ArrayLike

import matplotlib.patches as mpatches

from qp.plotting.style import FIELD_COLORS, FIELD_LABELS_MFA, QP_COLORS


# Key periods to annotate (in minutes)
PERIOD_MARKERS = {
    "QP120": 120,
    "QP60": 60,
    "QP30": 30,
}


def plot_power_density(
    ax: plt.Axes,
    freq: ArrayLike,
    psds: list[ArrayLike],
    background: ArrayLike | None = None,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    ylabel: str = r"PSD [nT$^2$/Hz]",
    mark_periods: bool = True,
) -> None:
    """Plot power spectral density in log-log space.

    Parameters
    ----------
    ax : Axes
    freq : array_like
        Frequency array (Hz).
    psds : list of array_like
        Power spectral densities for each component.
    background : array_like, optional
        Background PSD estimate (plotted as dashed line).
    """
    freq = np.asarray(freq)
    if colors is None:
        colors = FIELD_COLORS
    if labels is None:
        labels = FIELD_LABELS_MFA[: len(psds)]

    period_min = np.where(freq > 0, 1 / (freq * 60), np.inf)

    for psd, label, color in zip(psds, labels, colors):
        ax.loglog(period_min, psd, color=color, label=label, lw=0.8)

    if background is not None:
        ax.loglog(
            period_min,
            background,
            color="white",
            ls="--",
            lw=1,
            alpha=0.7,
            label="Background",
        )

    if mark_periods:
        _draw_period_markers(ax)

    ax.set_xlabel("Period [min]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.invert_xaxis()


def plot_power_ratios(
    ax: plt.Axes,
    freq: ArrayLike,
    ratios: dict[str, ArrayLike],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    ylabel: str = r"Power ratio $r_i$",
    mark_periods: bool = True,
    unity_line: bool = True,
) -> None:
    """Plot power ratios r_i vs period.

    Parameters
    ----------
    ratios : dict
        Must contain 'r_par', 'r_perp1', 'r_perp2', 'r_total' arrays.
    """
    freq = np.asarray(freq)
    if colors is None:
        colors = FIELD_COLORS
    if labels is None:
        labels = FIELD_LABELS_MFA

    period_min = np.where(freq > 0, 1 / (freq * 60), np.inf)

    keys = ["r_par", "r_perp1", "r_perp2", "r_total"]
    for key, label, color in zip(keys, labels, colors):
        if key in ratios:
            ax.semilogy(period_min, ratios[key], color=color, label=label, lw=0.8)

    if unity_line:
        ax.axhline(1.0, ls="--", lw=0.5, color="grey", alpha=0.5)

    if mark_periods:
        _draw_period_markers(ax)

    ax.set_xlabel("Period [min]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.invert_xaxis()


def plot_median_power_ratios(
    ax: plt.Axes,
    freq: ArrayLike,
    median_ratios: dict[str, ArrayLike],
    lower_quartiles: dict[str, ArrayLike] | None = None,
    upper_quartiles: dict[str, ArrayLike] | None = None,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    title: str = "",
    mark_periods: bool = True,
) -> None:
    """Plot median power ratios with quartile shading (Fig 5 style).

    Parameters
    ----------
    median_ratios : dict
        Keys 'r_par', 'r_perp1', 'r_perp2', 'r_total' with median arrays.
    lower_quartiles, upper_quartiles : dict, optional
        Same keys, for shaded interquartile range.
    """
    freq = np.asarray(freq)
    if colors is None:
        colors = FIELD_COLORS
    if labels is None:
        labels = FIELD_LABELS_MFA

    period_min = np.where(freq > 0, 1 / (freq * 60), np.inf)

    keys = ["r_par", "r_perp1", "r_perp2", "r_total"]
    for key, label, color in zip(keys, labels, colors):
        if key not in median_ratios:
            continue
        ax.semilogy(period_min, median_ratios[key], color=color, label=label, lw=1.2)

        if lower_quartiles and upper_quartiles and key in lower_quartiles:
            ax.fill_between(
                period_min,
                lower_quartiles[key],
                upper_quartiles[key],
                color=color,
                alpha=0.15,
            )

    ax.axhline(1.0, ls="--", lw=0.5, color="grey", alpha=0.5)

    # White dashed line at 50 min (referee request)
    ax.axvline(50, ls="--", lw=0.8, color="white", alpha=0.6)

    if mark_periods:
        _draw_period_markers(ax)

    ax.set_xlabel("Period [min]")
    ax.set_ylabel(r"Power ratio $r_i$")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.invert_xaxis()


def plot_spectrogram(
    ax: plt.Axes,
    time: ArrayLike,
    freq: ArrayLike,
    power: ArrayLike,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "inferno",
    ylabel: str = "Period [min]",
    freq_as_period: bool = True,
) -> plt.cm.ScalarMappable:
    """Plot a spectrogram (time-frequency power map).

    Parameters
    ----------
    time : array_like
        Time axis (datetime or seconds).
    freq : array_like
        Frequency axis (Hz).
    power : array_like
        Power values, shape (n_freq, n_time).

    Returns the mappable for colorbar creation.
    """
    freq = np.asarray(freq)
    if freq_as_period:
        y = np.where(freq > 0, 1 / (freq * 60), np.inf)
    else:
        y = freq

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if vmin or vmax else mcolors.LogNorm()
    im = ax.pcolormesh(time, y, power, cmap=cmap, norm=norm, shading="auto")

    ax.set_ylabel(ylabel)
    if freq_as_period:
        ax.set_ylim(5, 200)
        ax.invert_yaxis()

    return im


def annotate_spectral_peaks(
    ax: plt.Axes,
    freq: ArrayLike,
    psd: ArrayLike,
    peak_indices: ArrayLike,
    marker: str = "v",
    color: str = "yellow",
    size: float = 40,
    annotate_period: bool = True,
    fontsize: float = 8,
) -> None:
    """Mark detected spectral peaks on a PSD plot.

    Parameters
    ----------
    ax : Axes
    freq : array_like
        Frequency array (Hz).
    psd : array_like
        Power spectral density.
    peak_indices : array_like
        Indices into freq/psd of the peaks to mark.
    annotate_period : bool
        If True, add text labels showing the period in minutes.
    """
    freq = np.asarray(freq)
    psd = np.asarray(psd)
    peak_indices = np.atleast_1d(peak_indices)

    period_min = np.where(freq > 0, 1 / (freq * 60), np.inf)

    for idx in peak_indices:
        ax.scatter(
            period_min[idx],
            psd[idx],
            marker=marker,
            s=size,
            color=color,
            zorder=5,
        )
        if annotate_period:
            ax.annotate(
                f"{period_min[idx]:.0f} min",
                (period_min[idx], psd[idx]),
                textcoords="offset points",
                xytext=(5, 10),
                fontsize=fontsize,
                color=color,
                alpha=0.9,
            )


def overlay_power_law(
    ax: plt.Axes,
    freq_fit: ArrayLike,
    psd_fit: ArrayLike,
    color: str = "cyan",
    ls: str = "-.",
    lw: float = 1.2,
    alpha: float = 0.8,
    label: str | None = None,
) -> None:
    """Overlay a fitted power-law curve on a PSD plot.

    Parameters
    ----------
    freq_fit, psd_fit : array_like
        Fitted curve arrays from ``power_law_fit(return_fit=True)``.
    """
    freq_fit = np.asarray(freq_fit)
    psd_fit = np.asarray(psd_fit)
    period_min = np.where(freq_fit > 0, 1 / (freq_fit * 60), np.inf)
    ax.plot(period_min, psd_fit, color=color, ls=ls, lw=lw, alpha=alpha, label=label)


def draw_period_rectangles(
    ax: plt.Axes,
    bands: dict[str, tuple[float, float]] | None = None,
    colors: dict[str, str] | None = None,
    alpha: float = 0.12,
    y_span: tuple[float, float] | None = None,
) -> None:
    """Draw shaded rectangles marking QP period bands.

    Replaces ``cassinilib/PlotFFT.py:periodVisual()``.

    Parameters
    ----------
    bands : dict
        Period bands as ``{name: (min_period_min, max_period_min)}``.
        Default: QP30 (25-35 min), QP60 (50-70 min), QP120 (100-140 min).
    colors : dict
        Colors per band name. Default from QP_COLORS.
    """
    if bands is None:
        bands = {
            "QP30": (25, 35),
            "QP60": (50, 70),
            "QP120": (100, 140),
        }
    if colors is None:
        colors = QP_COLORS

    if y_span is None:
        y_lo, y_hi = ax.get_ylim()
    else:
        y_lo, y_hi = y_span

    for name, (p_lo, p_hi) in bands.items():
        color = colors.get(name, "white")
        rect = mpatches.Rectangle(
            (p_lo, y_lo),
            p_hi - p_lo,
            y_hi - y_lo,
            alpha=alpha,
            facecolor=color,
            edgecolor="none",
            label=name,
        )
        ax.add_patch(rect)


def plot_fft_snapshots(
    axes: list[plt.Axes] | np.ndarray,
    freq: ArrayLike,
    psd_list: list[ArrayLike],
    snapshot_labels: list[str] | None = None,
    colors: list[str] | None = None,
    ylabel: str = r"PSD [nT$^2$/Hz]",
    mark_periods: bool = True,
) -> None:
    """Plot multiple FFT snapshots in a row of axes.

    Replaces ``cassinilib/PlotFFT.py:plotFFTSnaps()``.

    Parameters
    ----------
    axes : list of Axes
        One axes per snapshot.
    freq : array_like
        Shared frequency array (Hz).
    psd_list : list of array_like
        One PSD array per snapshot.
    snapshot_labels : list of str, optional
        Title for each snapshot panel.
    colors : list of str, optional
        Color for each snapshot curve.
    """
    freq = np.asarray(freq)
    period_min = np.where(freq > 0, 1 / (freq * 60), np.inf)

    if colors is None:
        default_palette = [
            "#E1DAAE",
            "#FF934F",
            "#CC2D35",
            "#058ED9",
            "#848FA2",
            "#2D3142",
        ]
        colors = default_palette

    for i, (ax, psd) in enumerate(zip(axes, psd_list)):
        color = colors[i % len(colors)]
        label = snapshot_labels[i] if snapshot_labels else None
        ax.loglog(period_min, psd, color=color, lw=0.8, label=label)

        if mark_periods:
            _draw_period_markers(ax)

        ax.set_ylabel(ylabel if i == 0 else "")
        ax.set_xlabel("Period [min]")
        ax.invert_xaxis()

        if label:
            ax.set_title(label, fontsize=10)


# --- Helpers ---


def _draw_period_markers(ax: plt.Axes) -> None:
    """Draw vertical lines at key QP periods."""
    for name, period in PERIOD_MARKERS.items():
        color = QP_COLORS.get(name, "white")
        ax.axvline(period, ls=":", lw=0.8, color=color, alpha=0.6)
        ax.text(
            period,
            ax.get_ylim()[1] * 0.8,
            name,
            fontsize=7,
            color=color,
            ha="center",
            va="top",
            alpha=0.8,
        )
