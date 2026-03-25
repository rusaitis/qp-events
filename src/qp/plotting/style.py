"""Publication figure styling: color palettes, matplotlib defaults.

Matches the original publication styling from mag_fft_sweeper.py and PlotFFT.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


# Publication color palette for field components
# [b_par / Bx / Br, b_perp1 / By / Bth, b_perp2 / Bz / Bphi, Btot]
FIELD_COLORS = ["#DC267F", "#FFB000", "#FE6100", "#648FFF"]

# Spectral plot colors (original used yellow #fdf33c for b_perp2 instead of orange)
FIELD_COLORS_SPECTRA = ["#DC267F", "#FFB000", "#fdf33c", "#648FFF"]

FIELD_LABELS_MFA = [
    r"$B_{\parallel}$",
    r"$B_{\perp 1}$",
    r"$B_{\perp 2}$",
    r"$B_{tot}$",
]
FIELD_LABELS_KSM = [r"$B_x$", r"$B_y$", r"$B_z$", r"$B_{tot}$"]
FIELD_LABELS_KRTP = [r"$B_r$", r"$B_\theta$", r"$B_\phi$", r"$B_{tot}$"]

# Location colors (magnetosphere / magnetosheath / solar wind)
LOC_COLORS = {0: "#12d5ae", 1: "#f29539", 2: "#f26b59", 9: "black"}
LOC_LABELS = {0: "MS", 1: "SH", 2: "SW", 9: "--"}

# QP band colors
QP_COLORS = {"QP30": "grey", "QP60": "#FFB000", "QP120": "#DC267F"}

# Dark background color (dark grey, not pure black — matches original)
BG_COLOR = "#171717"


def use_paper_style(style_path: Path | str | None = None) -> None:
    """Load the paper matplotlib style, then apply dark background."""
    plt.style.use("default")

    if style_path is None:
        candidates = [
            Path(__file__).resolve().parents[3] / "paper.mplstyle",
            Path.cwd() / "paper.mplstyle",
        ]
        for p in candidates:
            if p.exists():
                style_path = p
                break

    if style_path is not None and Path(style_path).exists():
        plt.style.use(str(style_path))

    # Override with publication defaults
    plt.rcParams.update(
        {
            "font.size": 17,
            "text.usetex": False,
        }
    )

    dark_background()


def dark_background() -> None:
    """Apply dark background style matching the original figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": BG_COLOR,
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "legend.facecolor": BG_COLOR,
            "legend.edgecolor": "none",
        }
    )


def style_axes(ax: plt.Axes, grid: bool = True, minimal: bool = True) -> None:
    """Apply consistent axis styling matching the originals."""
    ax.set_facecolor(BG_COLOR)
    if grid:
        ax.grid(True, which="major", color="#CCCCCC", linestyle="--", alpha=0.4)
        ax.grid(True, which="minor", color="#CCCCCC", linestyle=":", alpha=0.2)
    if minimal:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_frame_on(False)


def draw_period_lines(
    ax: plt.Axes,
    periods_min: list[float] | None = None,
    units: float = 1e-3,
    direction: str = "vertical",
    color: str = "white",
    lw: float = 2,
    ls: str = "--",
    alpha: float = 0.5,
    fontsize: float = 12,
    show_labels: bool = True,
) -> None:
    """Draw period annotation lines matching the original PlotFFT.drawPeriodLine()."""
    if periods_min is None:
        periods_min = [5, 15, 30, 45, 60, 90, 120, 180]

    for T_min in periods_min:
        T_sec = T_min * 60
        f = 1.0 / T_sec / units  # frequency in display units (mHz if units=1e-3)

        line_kw = dict(color=color, alpha=alpha, ls=ls, lw=lw)

        if direction == "vertical":
            ax.axvline(x=f, **line_kw)
            if show_labels:
                label = f"{T_min / 60:.0f}h" if T_min >= 90 else f"{T_min:.0f}min"
                ax.text(
                    f,
                    0.97,
                    label,
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                    color=color,
                    alpha=alpha + 0.2,
                    bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"),
                )
        elif direction == "horizontal":
            ax.axhline(y=f, **line_kw)
            if show_labels:
                label = f"{T_min:.0f}min"
                ax.text(
                    0.02,
                    f,
                    label,
                    transform=ax.get_yaxis_transform(),
                    fontsize=fontsize,
                    color=color,
                    alpha=alpha + 0.2,
                )


def plot_segmented(
    ax: plt.Axes,
    freq: np.ndarray,
    data: np.ndarray,
    units: float = 1e-3,
    base_alpha: float = 0.8,
    **kwargs,
) -> None:
    """Plot spectral data with frequency-dependent transparency.

    Makes the critical 30min-6h band fully opaque and fades the rest,
    matching the original PlotFFT.plotSegs() behavior.
    """
    # Frequency band boundaries
    splits = []
    for period_sec in [6 * 3600, 30 * 60, 15 * 60, 5 * 60]:
        idx = np.searchsorted(freq, 1.0 / period_sec)
        splits.append(idx)
    splits = [0] + splits + [len(freq)]

    # Alpha weights: [<6h, 6h-30m, 30m-15m, 15m-5m, >5m]
    alphas = np.array([0.1, 1.0, 0.1, 0.05, 0.03]) * base_alpha

    for i in range(len(splits) - 1):
        sl = slice(splits[i], splits[i + 1])
        kw = kwargs.copy()
        if i != 1 and "label" in kw:
            kw.pop("label")  # only label the critical band
        ax.plot(freq[sl] / units, data[sl], alpha=alphas[i], **kw)


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray | plt.Axes]:
    """Create a figure with consistent styling."""
    if figsize is None:
        figsize = (12, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    fig.set_facecolor(BG_COLOR)
    return fig, axes


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 200,
    transparent: bool = False,
    close: bool = True,
) -> None:
    r"""Save a figure to file with sensible defaults.

    Replaces ``cassinilib/Plot.py:figure_output()``.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    path : str or Path
        Output file path. Format inferred from extension.
    dpi : int
        Resolution in dots per inch.
    transparent : bool
        If True, save with transparent background.
    close : bool
        If True, close the figure after saving to free memory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=dpi,
        transparent=transparent,
        facecolor=fig.get_facecolor() if not transparent else "none",
        bbox_inches="tight",
    )
    log.info("Saved figure to %s", path)

    if close:
        plt.close(fig)


def style_colorbar(
    mappable: plt.cm.ScalarMappable,
    ax: plt.Axes,
    label: str | None = None,
    ticks: list[float] | None = None,
    tick_labels: list[str] | None = None,
    location: str = "right",
    pad: float = 0.02,
) -> plt.colorbar.Colorbar:
    r"""Add and style a colorbar.

    Replaces ``cassinilib/Plot.py:styleColorbar()``.

    Parameters
    ----------
    mappable : ScalarMappable
        The image/contour/pcolormesh to attach the colorbar to.
    ax : Axes
        Axes to attach the colorbar alongside.
    label : str, optional
        Colorbar label.
    ticks : list, optional
        Tick positions.
    tick_labels : list, optional
        Custom tick labels.
    location : str
        Side of the axes: 'right', 'left', 'top', 'bottom'.
    pad : float
        Padding between axes and colorbar.

    Returns
    -------
    Colorbar
    """
    cbar = plt.colorbar(
        mappable,
        ax=ax,
        location=location,
        ticks=ticks,
        pad=pad,
        extend="both",
        extendfrac=0.05,
        fraction=0.046,
    )
    if label is not None:
        cbar.set_label(label, labelpad=-5)
    if tick_labels is not None:
        cbar.ax.set_yticklabels(tick_labels)
    return cbar
