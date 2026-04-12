"""Dwell time computation and spatial binning.

Computes how long Cassini spent in each (invariant latitude, local time,
magnetic latitude) bin. Used for normalizing event occurrence rates.

Extracted from mission_trace.py and mission_trace_reader.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from qp.analysis.filtering import value_to_bin


def compute_dwell_time_map(
    inv_lat: ArrayLike,
    local_time: ArrayLike,
    mag_lat: ArrayLike,
    dt_seconds: float = 60.0,
    n_lat_bins: int = 180,
    n_lt_bins: int = 48,
    n_mlat_bins: int = 180,
    lat_range: tuple[float, float] = (-90, 90),
    lt_range: tuple[float, float] = (0, 24),
    mlat_range: tuple[float, float] = (-90, 90),
) -> np.ndarray:
    """Accumulate spacecraft dwell time into a 3D bin map.

    Parameters
    ----------
    inv_lat : array_like
        Invariant (conjugate) latitude in degrees for each time step.
    local_time : array_like
        Local time in hours for each time step.
    mag_lat : array_like
        Magnetic latitude in degrees for each time step.
    dt_seconds : float
        Time step between samples (default 60s).
    n_lat_bins, n_lt_bins, n_mlat_bins : int
        Number of bins in each dimension.
    lat_range, lt_range, mlat_range : tuple
        (min, max) for each axis.

    Returns
    -------
    dwell_map : ndarray, shape (n_lat_bins, n_lt_bins, n_mlat_bins)
        Accumulated dwell time in seconds.
    """
    inv_lat = np.asarray(inv_lat, dtype=float)
    local_time = np.asarray(local_time, dtype=float)
    mag_lat = np.asarray(mag_lat, dtype=float)

    dwell_map = np.zeros((n_lat_bins, n_lt_bins, n_mlat_bins))

    lat_idx = value_to_bin(inv_lat, lat_range[0], lat_range[1], n_lat_bins)
    lt_idx = value_to_bin(local_time, lt_range[0], lt_range[1], n_lt_bins)
    mlat_idx = value_to_bin(mag_lat, mlat_range[0], mlat_range[1], n_mlat_bins)

    # Accumulate
    for i in range(len(inv_lat)):
        if (
            0 <= lat_idx[i] < n_lat_bins
            and 0 <= lt_idx[i] < n_lt_bins
            and 0 <= mlat_idx[i] < n_mlat_bins
        ):
            dwell_map[lat_idx[i], lt_idx[i], mlat_idx[i]] += dt_seconds

    return dwell_map


def filter_dwell_map(
    dwell_map: np.ndarray,
    lat_range: tuple[float, float] | None = None,
    lt_range: tuple[float, float] | None = None,
    mlat_range: tuple[float, float] | None = None,
    n_lat_bins: int = 180,
    n_lt_bins: int = 48,
    n_mlat_bins: int = 180,
    full_lat_range: tuple[float, float] = (-90, 90),
    full_lt_range: tuple[float, float] = (0, 24),
    full_mlat_range: tuple[float, float] = (-90, 90),
) -> np.ndarray:
    """Filter a 3D dwell time map to a spatial sub-region.

    Handles local time wraparound (e.g., LT range [-3, 3] = [21, 24] + [0, 3]).

    Returns a copy with zeros outside the selected region.
    """
    result = dwell_map.copy()

    if lat_range is not None:
        i0 = value_to_bin(
            lat_range[0], full_lat_range[0], full_lat_range[1], n_lat_bins
        )
        i1 = (
            value_to_bin(lat_range[1], full_lat_range[0], full_lat_range[1], n_lat_bins)
            + 1
        )
        mask = np.zeros(n_lat_bins, dtype=bool)
        mask[i0:i1] = True
        result[~mask, :, :] = 0

    if lt_range is not None:
        lt0, lt1 = lt_range
        mask = np.zeros(n_lt_bins, dtype=bool)
        if lt0 < 0:
            # Wraparound: e.g., [-3, 3] → [21, 24] + [0, 3]
            j0 = value_to_bin(24 + lt0, full_lt_range[0], full_lt_range[1], n_lt_bins)
            mask[j0:] = True
            j1 = value_to_bin(lt1, full_lt_range[0], full_lt_range[1], n_lt_bins) + 1
            mask[:j1] = True
        elif lt1 > 24:
            j0 = value_to_bin(lt0, full_lt_range[0], full_lt_range[1], n_lt_bins)
            mask[j0:] = True
            j1 = (
                value_to_bin(lt1 - 24, full_lt_range[0], full_lt_range[1], n_lt_bins)
                + 1
            )
            mask[:j1] = True
        else:
            j0 = value_to_bin(lt0, full_lt_range[0], full_lt_range[1], n_lt_bins)
            j1 = value_to_bin(lt1, full_lt_range[0], full_lt_range[1], n_lt_bins) + 1
            mask[j0:j1] = True
        result[:, ~mask, :] = 0

    if mlat_range is not None:
        k0 = value_to_bin(
            mlat_range[0], full_mlat_range[0], full_mlat_range[1], n_mlat_bins
        )
        k1 = (
            value_to_bin(
                mlat_range[1], full_mlat_range[0], full_mlat_range[1], n_mlat_bins
            )
            + 1
        )
        mask = np.zeros(n_mlat_bins, dtype=bool)
        mask[k0:k1] = True
        result[:, :, ~mask] = 0

    return result


def reduce_to_lt_lat(dwell_map: np.ndarray) -> np.ndarray:
    """Reduce 3D dwell map to 2D (invariant_lat, local_time) by summing over mag_lat."""
    return np.sum(dwell_map, axis=2)


def reduce_to_mlat(dwell_map: np.ndarray) -> np.ndarray:
    """Reduce 3D dwell map to 1D magnetic latitude profile."""
    return np.sum(np.sum(dwell_map, axis=0), axis=0)


def reduce_to_lt(dwell_map: np.ndarray) -> np.ndarray:
    """Reduce 3D dwell map to 1D local time profile."""
    return np.sum(np.sum(dwell_map, axis=0), axis=1)


def normalize_event_to_dwell(
    event_map: np.ndarray,
    dwell_map: np.ndarray,
    min_dwell_seconds: float = 3600.0,
) -> np.ndarray:
    """Compute event-to-dwell-time ratio, masking bins with insufficient dwell time."""
    ratio = np.zeros_like(event_map, dtype=float)
    valid = dwell_map > min_dwell_seconds
    ratio[valid] = event_map[valid] / dwell_map[valid]
    return ratio
