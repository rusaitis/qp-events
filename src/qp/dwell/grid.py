r"""Spherical dwell-time grid: accumulate spacecraft time into (r, lat, LT) bins.

Reads Cassini KSM positions, converts to (r, magnetic_latitude, local_time),
and histograms dwell time in minutes into a configurable 3D grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from qp.coords.ksm import local_time, magnetic_latitude

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DwellGridConfig:
    r"""Configuration for the spherical dwell-time grid.

    Parameters
    ----------
    n_r : int
        Number of radial bins.
    n_lat : int
        Number of magnetic latitude bins.
    n_lt : int
        Number of local time bins.
    r_range : tuple[float, float]
        Radial range in $R_S$.
    lat_range : tuple[float, float]
        Magnetic latitude range in degrees.
    lt_range : tuple[float, float]
        Local time range in hours.
    """

    n_r: int = 70
    n_lat: int = 90
    n_lt: int = 48
    r_range: tuple[float, float] = (0.0, 70.0)
    lat_range: tuple[float, float] = (-90.0, 90.0)
    lt_range: tuple[float, float] = (0.0, 24.0)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_r, self.n_lat, self.n_lt)

    @property
    def r_edges(self) -> np.ndarray:
        return np.linspace(self.r_range[0], self.r_range[1], self.n_r + 1)

    @property
    def lat_edges(self) -> np.ndarray:
        return np.linspace(self.lat_range[0], self.lat_range[1], self.n_lat + 1)

    @property
    def lt_edges(self) -> np.ndarray:
        return np.linspace(self.lt_range[0], self.lt_range[1], self.n_lt + 1)

    @property
    def r_centers(self) -> np.ndarray:
        e = self.r_edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def lat_centers(self) -> np.ndarray:
        e = self.lat_edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def lt_centers(self) -> np.ndarray:
        e = self.lt_edges
        return 0.5 * (e[:-1] + e[1:])


def _bin_index(value: np.ndarray, vmin: float, vmax: float, n: int) -> np.ndarray:
    """Map values to bin indices in [0, n), clipped."""
    frac = (value - vmin) / (vmax - vmin)
    idx = np.floor(frac * n).astype(int)
    return np.clip(idx, 0, n - 1)


def accumulate_dwell_time(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    dt_minutes: float = 1.0,
    config: DwellGridConfig | None = None,
) -> np.ndarray:
    r"""Accumulate spacecraft dwell time into a 3D (r, mag_lat, LT) grid.

    Parameters
    ----------
    x, y, z : array_like
        Spacecraft position in KSM coordinates ($R_S$).
    dt_minutes : float
        Time step per sample in minutes (default 1.0 for 1-min MAG data).
    config : DwellGridConfig, optional
        Grid configuration. Uses defaults if not provided.

    Returns
    -------
    ndarray, shape (n_r, n_lat, n_lt)
        Accumulated dwell time in minutes.
    """
    if config is None:
        config = DwellGridConfig()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r = np.sqrt(x**2 + y**2 + z**2)
    lat = magnetic_latitude(x, y, z)
    lt = local_time(x, y)

    i_r = _bin_index(r, *config.r_range, config.n_r)
    i_lat = _bin_index(lat, *config.lat_range, config.n_lat)
    i_lt = _bin_index(lt, *config.lt_range, config.n_lt)

    # Mask out-of-range points
    in_range = (
        (r >= config.r_range[0])
        & (r < config.r_range[1])
        & (lat >= config.lat_range[0])
        & (lat < config.lat_range[1])
    )

    grid = np.zeros(config.shape, dtype=float)
    np.add.at(grid, (i_r[in_range], i_lat[in_range], i_lt[in_range]), dt_minutes)
    return grid


def accumulate_with_regions(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    region_codes: ArrayLike,
    dt_minutes: float = 1.0,
    config: DwellGridConfig | None = None,
) -> dict[str, np.ndarray]:
    r"""Accumulate dwell time with per-region breakdown.

    Parameters
    ----------
    x, y, z : array_like
        Spacecraft position in KSM ($R_S$).
    region_codes : array_like of int
        Location code for each sample: 0=MS, 1=SH, 2=SW, 9=unknown.
    dt_minutes : float
        Time step per sample in minutes.
    config : DwellGridConfig, optional
        Grid configuration.

    Returns
    -------
    dict
        Keys: ``'total'``, ``'ms'``, ``'sh'``, ``'sw'``, ``'unknown'``.
        Values: 3D numpy arrays of shape ``config.shape``.
    """
    if config is None:
        config = DwellGridConfig()

    codes = np.asarray(region_codes, dtype=int)
    total = accumulate_dwell_time(x, y, z, dt_minutes, config)

    result = {"total": total}
    region_map = {0: "ms", 1: "sh", 2: "sw", 9: "unknown"}

    for code, name in region_map.items():
        mask = codes == code
        if np.any(mask):
            xa, ya, za = np.asarray(x)[mask], np.asarray(y)[mask], np.asarray(z)[mask]
            result[name] = accumulate_dwell_time(xa, ya, za, dt_minutes, config)
        else:
            result[name] = np.zeros(config.shape, dtype=float)

    return result
