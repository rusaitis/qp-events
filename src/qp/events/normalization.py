r"""Dwell-normalized occurrence rate of QP wave events.

Phase 5 of the plan. The numerator is the event-time grid produced
by :mod:`qp.events.binning` (one zarr variable per QP band plus a
``"total"`` union grid). The denominator is the **consistency dwell
grid** built alongside the event grid, using the same coordinate
approximations — see the design note in
``src/qp/events/binning.py:accumulate_segment_dwell``.

The helpers here are pure: they take two ``numpy`` arrays and return
a third. The figure scripts wrap these in matplotlib renderings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class OccurrenceConfig:
    """Knobs for occurrence-rate computation."""

    min_dwell_minutes: float = 60.0  # 1 h floor
    clip_max: float = 1.0  # cap ratio at 100 %


DEFAULT_OCCURRENCE: OccurrenceConfig = OccurrenceConfig()


def occurrence_rate(
    event_grid: ArrayLike,
    dwell_grid: ArrayLike,
    *,
    min_dwell_minutes: float = 60.0,
    clip_max: float | None = 1.0,
) -> NDArray[np.floating]:
    r"""Per-cell occurrence rate ``event_time / dwell_time``.

    Parameters
    ----------
    event_grid, dwell_grid : array_like
        Cumulative event and dwell time, **same shape**, **same units**
        (minutes). Values must be non-negative.
    min_dwell_minutes : float, default 60.0
        Cells with less than this much dwell are set to ``NaN`` to
        suppress noisy single-pass cells.
    clip_max : float or None, default 1.0
        Upper clip on the ratio. ``None`` disables clipping. With the
        consistency dwell grid produced by
        :mod:`qp.events.binning`, the unclipped ratio is already
        guaranteed to lie in ``[0, 1]`` by construction; the clip is
        a defensive guard against numerical edge cases.

    Returns
    -------
    rate : ndarray of float
        Same shape as input. ``NaN`` where dwell is below the floor.
    """
    event_grid = np.asarray(event_grid, dtype=float)
    dwell_grid = np.asarray(dwell_grid, dtype=float)
    if event_grid.shape != dwell_grid.shape:
        raise ValueError(
            f"shape mismatch: event {event_grid.shape}, "
            f"dwell {dwell_grid.shape}"
        )
    rate = np.full_like(event_grid, np.nan)
    valid = dwell_grid >= min_dwell_minutes
    rate[valid] = event_grid[valid] / dwell_grid[valid]
    if clip_max is not None:
        np.clip(rate, 0.0, clip_max, out=rate, where=valid)
    return rate


def slice_lt_sector(
    grid: ArrayLike,
    lt_centers: ArrayLike,
    center_h: float,
    half_width_h: float = 3.0,
) -> NDArray[np.floating]:
    r"""Sum a 3D grid along the LT axis inside a ``center ± half_width`` sector.

    Handles wraparound at midnight. Returns shape ``(n_r, n_lat)``.
    """
    grid = np.asarray(grid, dtype=float)
    lt_centers = np.asarray(lt_centers, dtype=float)
    if grid.ndim != 3 or grid.shape[2] != lt_centers.size:
        raise ValueError(
            f"grid shape {grid.shape} incompatible with "
            f"lt_centers shape {lt_centers.shape}"
        )
    lo = (center_h - half_width_h) % 24.0
    hi = (center_h + half_width_h) % 24.0
    if lo < hi:
        mask = (lt_centers >= lo) & (lt_centers < hi)
    else:
        mask = (lt_centers >= lo) | (lt_centers < hi)
    return grid[:, :, mask].sum(axis=2)


def collapse_to_latitude(
    grid_2d_r_lat: ArrayLike,
) -> NDArray[np.floating]:
    r"""Sum a ``(r, mag_lat)`` slice along radius to get a 1D latitude profile."""
    grid_2d_r_lat = np.asarray(grid_2d_r_lat, dtype=float)
    if grid_2d_r_lat.ndim != 2:
        raise ValueError(f"expected 2D, got {grid_2d_r_lat.ndim}D")
    return grid_2d_r_lat.sum(axis=0)
