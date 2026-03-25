"""Time series plotting for magnetic field data.

Extracted and cleaned from cassinilib/PlotTimeseries.py.
Functions take plain arrays, not NewSignal objects.
"""

from __future__ import annotations

import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy.typing import ArrayLike

from qp.plotting.style import FIELD_COLORS, LOC_COLORS, LOC_LABELS


def plot_field_timeseries(
    ax: plt.Axes,
    times: ArrayLike,
    components: list[ArrayLike],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    ylabel: str = "B [nT]",
    zero_line: bool = True,
    hour_interval: int = 2,
    time_fmt: str = "%H:%M",
) -> None:
    """Plot magnetic field components vs time.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    times : array_like
        Datetime array for x-axis.
    components : list of array_like
        Field component arrays, e.g. ``[Bx, By, Bz, Btot]``.
    labels : list of str, optional
        Labels for each component.
    colors : list of str, optional
        Colors for each component. Defaults to FIELD_COLORS.
    """
    if colors is None:
        colors = FIELD_COLORS
    if labels is None:
        labels = [f"comp {i}" for i in range(len(components))]

    for comp, label, color in zip(components, labels, colors):
        ax.plot(times, comp, color=color, label=label, lw=0.8)

    if zero_line:
        ax.axhline(0, ls="--", lw=0.5, color="grey", alpha=0.5)

    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    _format_time_axis(ax, hour_interval, time_fmt)
    _clean_spines(ax)


def plot_ephemeris_bar(
    ax: plt.Axes,
    times: ArrayLike,
    coords: dict[str, ArrayLike],
    location_times: ArrayLike | None = None,
    location_codes: ArrayLike | None = None,
    hour_interval: int = 2,
    fontsize: int = 9,
) -> None:
    """Plot spacecraft ephemeris info bar below a timeseries.

    Parameters
    ----------
    ax : Axes
        Axes for the ephemeris bar (typically a thin subplot).
    times : array_like
        Datetime array.
    coords : dict
        Mapping of coordinate name to value array, e.g. {'X': x, 'Y': y, 'Z': z}.
    location_times : array_like, optional
        Datetime array for magnetosphere location lookup.
    location_codes : array_like, optional
        Location codes (0=MS, 1=SH, 2=SW) aligned with location_times.
    """
    times = np.asarray(times)
    coord_names = list(coords.keys())
    n_rows = len(coord_names) + (1 if location_codes is not None else 0)

    ax.set_yticks(range(n_rows))
    ylabels = (["Loc"] if location_codes is not None else []) + coord_names
    ax.set_yticklabels(ylabels, fontsize=fontsize)
    ax.set_xticklabels([])

    _format_time_axis(ax, hour_interval)

    # Sample at each major tick
    t_start = times[0]
    if isinstance(t_start, datetime.datetime):
        delta = datetime.timedelta(hours=hour_interval)
        t = _round_hour(t_start, hour_interval)
        while t < times[-1]:
            idx = np.searchsorted(times, t)
            if idx >= len(times):
                break
            row = 0
            # Location badge
            if location_codes is not None and location_times is not None:
                loc_idx = np.searchsorted(np.asarray(location_times), t)
                loc_idx = min(loc_idx, len(location_codes) - 1)
                code = int(location_codes[loc_idx])
                color = LOC_COLORS.get(code, "black")
                label = LOC_LABELS.get(code, "--")
                ax.text(
                    t,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="none", lw=0),
                )
                row += 1
            # Coordinate values
            for name in coord_names:
                val = coords[name][idx] if idx < len(coords[name]) else 0
                ax.text(
                    t,
                    row,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="grey",
                )
                row += 1
            t += delta

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.5, n_rows - 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(which="both", width=0)
    ax.grid(False)


def plot_highlight_intervals(
    ax: plt.Axes,
    intervals: list[tuple[datetime.datetime, datetime.datetime]],
    color: str = "orange",
    alpha: float = 0.15,
    label: str | None = None,
) -> None:
    """Shade time intervals (e.g., strong QP60 activity) on a timeseries."""
    for i, (t0, t1) in enumerate(intervals):
        ax.axvspan(t0, t1, alpha=alpha, color=color, label=label if i == 0 else None)


def lookup_mag_region(
    time: datetime.datetime,
    crossing_times: ArrayLike,
    crossing_codes: ArrayLike,
) -> tuple[str, str]:
    """Look up the magnetospheric region at a given time.

    Replaces ``cassinilib/PlotTimeseries.py:magnetospherePosition()``.

    Parameters
    ----------
    time : datetime
        Query time.
    crossing_times : array_like
        Datetime array of boundary crossing times.
    crossing_codes : array_like
        Integer region codes aligned with crossing_times.
        0 = magnetosphere, 1 = magnetosheath, 2 = solar wind.

    Returns
    -------
    label : str
        Region abbreviation ('MS', 'SH', 'SW', or '--').
    color : str
        Hex color for the region.
    """
    crossing_times = np.asarray(crossing_times)
    crossing_codes = np.asarray(crossing_codes)

    idx = np.searchsorted(crossing_times, time)
    idx = min(idx, len(crossing_codes) - 1)
    code = int(crossing_codes[idx])

    label = LOC_LABELS.get(code, "--")
    color = LOC_COLORS.get(code, "black")
    return label, color


def field_range(components: list[ArrayLike]) -> float:
    """Maximum peak-to-peak range across multiple field components.

    Replaces ``cassinilib/PlotTimeseries.py:findSeriesRange()``.

    Parameters
    ----------
    components : list of array_like
        Field component arrays.

    Returns
    -------
    float
        Maximum range (max - min) across all components.
    """
    ranges = []
    for c in components:
        c = np.asarray(c, dtype=float)
        if len(c) > 0:
            ranges.append(float(np.nanmax(c) - np.nanmin(c)))
    return max(ranges) if ranges else 0.0


def field_limits(components: list[ArrayLike]) -> tuple[float, float]:
    """Global min and max across multiple field components.

    Replaces ``cassinilib/PlotTimeseries.py:findSeriesLim()``.

    Parameters
    ----------
    components : list of array_like
        Field component arrays.

    Returns
    -------
    tuple[float, float]
        (global_min, global_max).
    """
    all_min = []
    all_max = []
    for c in components:
        c = np.asarray(c, dtype=float)
        if len(c) > 0:
            all_min.append(float(np.nanmin(c)))
            all_max.append(float(np.nanmax(c)))
    if not all_min:
        return (0.0, 0.0)
    return (min(all_min), max(all_max))


# --- Helpers ---


def _format_time_axis(
    ax: plt.Axes,
    hour_interval: int = 2,
    time_fmt: str = "%H:%M",
) -> None:
    """Set up time-axis formatting."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=max(1, hour_interval // 2)))


def _clean_spines(ax: plt.Axes) -> None:
    """Remove top and right spines for cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _round_hour(t: datetime.datetime, interval: int = 2) -> datetime.datetime:
    """Round datetime up to nearest even hour."""
    h = t.hour
    if t.minute > 0 or t.second > 0:
        h += 1
    h = ((h + interval - 1) // interval) * interval
    if h >= 24:
        h -= 24
        t += datetime.timedelta(days=1)
    return t.replace(hour=h, minute=0, second=0, microsecond=0)
