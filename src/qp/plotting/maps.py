"""Dwell time heatmaps and local time / latitude bin maps.

Used for Figures 2b, 7, 8, SI1, SI2.
"""

from __future__ import annotations

import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from numpy.typing import ArrayLike


def plot_lt_lat_heatmap(
    ax: plt.Axes,
    data: ArrayLike,
    lt_edges: ArrayLike,
    lat_edges: ArrayLike,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
    xlabel: str = "Local Time [h]",
    ylabel: str = "Conjugate Latitude [deg]",
    cbar_label: str = "Hours",
) -> plt.cm.ScalarMappable:
    """Plot a 2D heatmap in local time vs latitude bins.

    Parameters
    ----------
    data : array_like, shape (n_lat, n_lt)
        Bin values (e.g., dwell time in hours, or event/dwell ratio).
    lt_edges : array_like
        Local time bin edges.
    lat_edges : array_like
        Latitude bin edges.

    Returns the mappable for colorbar use.
    """
    if log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(lt_edges, lat_edges, data, cmap=cmap, norm=norm, shading="auto")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return im


def plot_marginal_distributions(
    fig: plt.Figure,
    ax_main: plt.Axes,
    data: ArrayLike,
    lt_edges: ArrayLike,
    lat_edges: ArrayLike,
    ax_top: plt.Axes | None = None,
    ax_right: plt.Axes | None = None,
    color: str = "cyan",
) -> None:
    """Add marginal distributions (cumulative sums) to a heatmap.

    Used for Fig 8 style: bar plots at top (sum over latitude)
    and right (sum over local time).
    """
    data = np.asarray(data)
    lt_centers = 0.5 * (np.asarray(lt_edges[:-1]) + np.asarray(lt_edges[1:]))
    lat_centers = 0.5 * (np.asarray(lat_edges[:-1]) + np.asarray(lat_edges[1:]))

    lt_sum = np.nansum(data, axis=0)  # sum over latitudes
    lat_sum = np.nansum(data, axis=1)  # sum over local times

    if ax_top is not None:
        ax_top.bar(lt_centers, lt_sum, width=np.diff(lt_edges), color=color, alpha=0.6)
        ax_top.set_xlim(ax_main.get_xlim())
        ax_top.set_xticklabels([])

    if ax_right is not None:
        ax_right.barh(
            lat_centers, lat_sum, height=np.diff(lat_edges), color=color, alpha=0.6
        )
        ax_right.set_ylim(ax_main.get_ylim())
        ax_right.set_yticklabels([])


def plot_ratio_vs_latitude(
    ax: plt.Axes,
    lat_centers: ArrayLike,
    ratio: ArrayLike,
    color: str = "white",
    label: str = "",
    ylabel: str = "Magnetic Latitude [deg]",
    xlabel: str = "Event / Dwell Time Ratio",
) -> None:
    """Plot event-to-dwell-time ratio vs magnetic latitude (Fig 7 style)."""
    ax.plot(ratio, lat_centers, color=color, lw=1.2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, ls="--", lw=0.5, color="grey", alpha=0.3)


def plot_polar_heatmap(
    ax: plt.Axes,
    data: ArrayLike,
    lat_edges: ArrayLike,
    n_lt_bins: int | None = None,
    hemisphere: str = "north",
    cmap: str = "plasma",
    vmin: float = 0.001,
    vmax: float | None = None,
) -> plt.cm.ScalarMappable:
    r"""Plot a polar heatmap in local-time vs conjugate-latitude.

    Replaces ``cassinilib/Plot.py:plotHeatmapPolarAxis()``.

    Parameters
    ----------
    ax : Axes
        Must be a polar projection axes.
    data : array_like, shape (n_lat, n_lt)
        Bin values.
    lat_edges : array_like
        Latitude bin edges (degrees).
    n_lt_bins : int, optional
        Number of local-time bins. Default: ``data.shape[1]``.
    hemisphere : str
        'north' or 'south'. Controls radial axis direction.
    cmap : str
        Colormap name.
    vmin, vmax : float
        Color range.

    Returns
    -------
    ScalarMappable
        The pcolormesh mappable (for colorbar).
    """
    data = np.asarray(data, dtype=float)
    lat_edges = np.asarray(lat_edges, dtype=float)
    if n_lt_bins is None:
        n_lt_bins = data.shape[1]
    if vmax is None:
        vmax = float(np.nanmax(data))

    lt_edges = np.linspace(0, 2 * np.pi, n_lt_bins + 1)

    cmap_obj = copy.copy(plt.colormaps[cmap])
    cmap_obj.set_under(color="black")

    if hemisphere == "north":
        ax.set_rlim(float(lat_edges[-1]), float(lat_edges[0]))
    else:
        ax.set_rlim(float(lat_edges[0]), float(lat_edges[-1]))
        data = np.flip(data, axis=0)

    im = ax.pcolormesh(lt_edges, lat_edges, data, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_theta_zero_location("W")
    ax.grid(color="white", linestyle="--", linewidth=0.9, alpha=0.2)
    ax.set_facecolor("#171717")
    return im


def draw_latitude_indicator(
    ax: plt.Axes,
    lat_range: tuple[float, float] | None = None,
    lat_ticks: list[float] | None = None,
    color: str = "orange",
    minor_ticks: list[float] | None = None,
) -> None:
    r"""Draw a latitude position indicator diagram.

    Replaces ``cassinilib/PlotFFT.py:latitudeVisual()``.

    Parameters
    ----------
    ax : Axes
        Target axes (will be set to equal aspect, axis off).
    lat_range : tuple, optional
        (min_deg, max_deg) to shade as an arc.
    lat_ticks : list, optional
        Specific latitude values to mark as radial lines.
    minor_ticks : list, optional
        Reference latitude lines (degrees). Default: [30, 60, -30, -60].
    """
    if minor_ticks is None:
        minor_ticks = [30, 60, -30, -60]

    # Half-sphere and equator line
    theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
    ax.plot(0.4 * np.cos(theta), 0.4 * np.sin(theta), color="white", alpha=0.9, lw=1)
    ax.plot((0.4, 1), (0, 0), lw=1, ls="--", alpha=0.5, color="white")
    ax.annotate(
        "",
        xy=(0, 1),
        xytext=(0, -1),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
    )

    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.5, 1.2)
    ax.axis("off")
    ax.text(
        0,
        -1.5,
        "Latitude",
        color="white",
        fontsize=14,
        alpha=0.6,
        ha="center",
        clip_on=False,
    )

    if lat_range is not None:
        _arc_patch(
            ax,
            (0, 0),
            1.0,
            lat_range[0],
            lat_range[1],
            radius_inner=0.5,
            alpha=0.2,
            facecolor="white",
        )

    if lat_ticks is not None:
        for th in lat_ticks:
            th_rad = np.deg2rad(th)
            ax.plot(
                (0.5 * np.cos(th_rad), np.cos(th_rad)),
                (0.5 * np.sin(th_rad), np.sin(th_rad)),
                lw=0.5,
                color=color,
                alpha=0.9,
            )

    for th in minor_ticks:
        th_rad = np.deg2rad(th)
        ax.plot(
            (0.4 * np.cos(th_rad), np.cos(th_rad)),
            (0.4 * np.sin(th_rad), np.sin(th_rad)),
            lw=1,
            color="white",
            alpha=0.3,
            ls="--",
        )


def draw_range_indicator(
    ax: plt.Axes,
    data: ArrayLike | None = None,
    data_range: tuple[float, float] | None = None,
    color: str = "orange",
    label: str | None = None,
) -> None:
    r"""Draw a range indicator bar.

    Replaces ``cassinilib/PlotFFT.py:rangeValueVisual()``.

    Parameters
    ----------
    ax : Axes
    data : array_like, optional
        Values to mark on the bar.
    data_range : tuple, optional
        (min, max) for the bar extent.
    """
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_edgecolor("white")
    ax.set_yticks([])

    if data is not None:
        data = np.asarray(data)
        ax.scatter(data, np.zeros_like(data), color=color, s=10, alpha=0.6, zorder=3)

    if data_range is not None:
        ax.set_xlim(data_range)

    if label is not None:
        ax.set_xlabel(label, color="white", fontsize=12)


def draw_local_time_indicator(
    ax: plt.Axes,
    lt_range: tuple[float, float] | None = None,
    lt_values: list[float] | None = None,
    color: str = "orange",
) -> None:
    r"""Draw a local-time clock diagram.

    Replaces ``cassinilib/PlotFFT.py:localTimeVisual()``.

    Parameters
    ----------
    ax : Axes
        Target axes (set to equal aspect, axis off).
    lt_range : tuple, optional
        (lt_min, lt_max) in hours to shade as an arc.
    lt_values : list, optional
        Specific LT values to mark as radial lines.
    """
    axis_kw = {"lw": 1, "ls": "--", "alpha": 0.4, "color": "white"}

    # Planet: white circle with dark nightside
    _arc_patch(ax, (0, 0), 0.4, 0, 360, facecolor="white", alpha=0.9)
    _arc_patch(ax, (0, 0), 0.4, 90, 270, facecolor="black", alpha=0.9)

    # Cross-hair axes
    ax.plot((0.4, 1), (0, 0), **axis_kw)
    ax.plot((-1, -0.4), (0, 0), **axis_kw)
    ax.plot((0, 0), (0.4, 1), **axis_kw)
    ax.plot((0, 0), (-1, -0.4), **axis_kw)

    if lt_range is not None:
        th1 = _lt_to_degrees(lt_range[0])
        th2 = _lt_to_degrees(lt_range[1])
        _arc_patch(
            ax, (0, 0), 1.0, th1, th2, radius_inner=0.5, facecolor="white", alpha=0.2
        )

    if lt_values is not None:
        for lt in lt_values:
            th = np.deg2rad(_lt_to_degrees(lt))
            ax.plot(
                (0.5 * np.cos(th), np.cos(th)),
                (0.5 * np.sin(th), np.sin(th)),
                lw=0.5,
                color=color,
                alpha=0.6,
            )

    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.3, 1.2)
    ax.axis("off")
    ax.text(
        0,
        -1.3,
        "Local Time",
        color="white",
        fontsize=14,
        alpha=0.6,
        ha="center",
        clip_on=False,
    )


def draw_power_indicator(
    ax: plt.Axes,
    data: ArrayLike | None = None,
    data_range: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    color: str = "orange",
    ylabel: str = r"Median Power ($B_T$, 60m) [$nT^2/Hz$]",
) -> None:
    r"""Draw a vertical power indicator bar (log scale).

    Replaces ``cassinilib/PlotFFT.py:PowerVisual()``.

    Parameters
    ----------
    ax : Axes
    data : array_like, optional
        Individual values (drawn as thin horizontal lines + median bold).
    data_range : tuple, optional
        (min, max) to shade as a rectangle.
    ylim : tuple, optional
        Y-axis limits.
    """
    _style_vertical_indicator(ax, "left")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel, alpha=0.5, fontsize=14)
    ax.set_xlim(0, 1)

    if data_range is not None:
        from matplotlib.patches import Rectangle

        ax.add_patch(
            Rectangle(
                (0, data_range[0]),
                1,
                data_range[1] - data_range[0],
                color="white",
                alpha=0.2,
            )
        )

    if data is not None:
        data = np.asarray(data, dtype=float)
        for val in data:
            ax.axhline(y=val, lw=1, color=color, alpha=0.1)
        ax.axhline(y=float(np.median(data)), lw=3, color=color, alpha=0.8)

    if ylim is not None:
        ax.set_ylim(ylim)


def draw_field_indicator(
    ax: plt.Axes,
    data: ArrayLike | None = None,
    data_range: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    color: str = "orange",
    ylabel: str = r"Median Magnetic Field ($B_T$) [$nT$]",
) -> None:
    r"""Draw a vertical field magnitude indicator bar (log scale).

    Replaces ``cassinilib/PlotFFT.py:FieldVisual()``.

    Parameters
    ----------
    ax : Axes
    data : array_like, optional
        Individual values (drawn as thin lines + bold median).
    data_range : tuple, optional
        (min, max) to shade as a rectangle.
    ylim : tuple, optional
        Y-axis limits.
    """
    _style_vertical_indicator(ax, "left")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel, alpha=0.5, fontsize=14)
    ax.set_xlim(0, 1)

    if data_range is not None:
        from matplotlib.patches import Rectangle

        ax.add_patch(
            Rectangle(
                (0, data_range[0]),
                1,
                data_range[1] - data_range[0],
                color="white",
                alpha=0.2,
            )
        )

    if data is not None:
        data = np.asarray(data, dtype=float)
        for val in data:
            ax.axhline(y=val, lw=1, color=color, alpha=0.1)
        ax.axhline(y=float(np.median(data)), lw=4, color=color, alpha=1.0)

    if ylim is not None:
        ax.set_ylim(ylim)


# --- Internal helpers ---


def _lt_to_degrees(lt: float) -> float:
    """Convert local time (hours) to angular degrees for clock diagram."""
    return (lt - 12) / 24 * 360


def _style_vertical_indicator(ax: plt.Axes, spine_side: str = "left") -> None:
    """Common styling for vertical indicator bars (PowerVisual/FieldVisual)."""
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines[spine_side].set_visible(True)
    ax.spines[spine_side].set_edgecolor("white")
    ax.get_xaxis().set_ticks([])
    ax.tick_params(labelbottom=False)
    ax.grid(True, which="major", color="#CCCCCC", linestyle="--", alpha=0.1, lw=0.5)
    ax.grid(True, which="minor", color="#CCCCCC", linestyle=":", alpha=0.05)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_facecolor("#171717")


def _arc_patch(
    ax: plt.Axes,
    center: tuple[float, float],
    radius: float,
    theta1: float,
    theta2: float,
    radius_inner: float | None = None,
    resolution: int = 50,
    **kwargs,
) -> mpatches.Polygon:
    """Create and add an arc (or annular) patch."""
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    if radius_inner is not None:
        x2 = radius_inner * np.cos(theta) + center[0]
        y2 = radius_inner * np.sin(theta) + center[1]
    else:
        x2 = np.array([center[0]])
        y2 = np.array([center[1]])
    xvals = np.append(x, np.flip(x2))
    yvals = np.append(y, np.flip(y2))
    poly = mpatches.Polygon(np.column_stack([xvals, yvals]), closed=True, **kwargs)
    ax.add_patch(poly)
    return poly
