r"""Spherical dwell-time grid: accumulate spacecraft time into (r, lat, LT) bins.

Reads Cassini KSM positions, converts to (r, magnetic_latitude, local_time),
and histograms dwell time in minutes into a configurable 3D grid.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from qp.coords.ksm import dipole_invariant_latitude, local_time, magnetic_latitude

log = logging.getLogger(__name__)

# Region code → name mapping (matches qp.io.crossings: MS=0, SH=1, SW=2, UNKNOWN=9)
REGION_CODES: dict[int, str] = {
    0: "magnetosphere",
    1: "magnetosheath",
    2: "solar_wind",
    9: "unknown",
}


def region_name(code: int | None) -> str:
    """Map a Jackman boundary-crossing region code to its canonical name."""
    if code is None:
        return "unknown"
    return REGION_CODES.get(int(code), "unknown")


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

    n_r: int = 100
    n_lat: int = 180
    n_lt: int = 96
    r_range: tuple[float, float] = (0.0, 100.0)
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


def _compute_coords(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert KSM Cartesian to (r, magnetic_latitude, local_time)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = magnetic_latitude(x, y, z)
    lt = local_time(x, y)
    return r, lat, lt


def _compute_bins(
    r: np.ndarray,
    lat: np.ndarray,
    lt: np.ndarray,
    config: DwellGridConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute bin indices and in-range mask."""
    i_r = _bin_index(r, *config.r_range, config.n_r)
    i_lat = _bin_index(lat, *config.lat_range, config.n_lat)
    i_lt = _bin_index(lt, *config.lt_range, config.n_lt)

    in_range = (
        (r >= config.r_range[0])
        & (r < config.r_range[1])
        & (lat >= config.lat_range[0])
        & (lat < config.lat_range[1])
        & (lt >= config.lt_range[0])
        & (lt < config.lt_range[1])
    )
    return i_r, i_lat, i_lt, in_range


def _bin_index(
    value: float | np.ndarray,
    vmin: float,
    vmax: float,
    n: int,
) -> np.ndarray:
    """Map values to bin indices in [0, n), clipped.

    Accepts a scalar or ndarray; the return is always an ndarray (0-d
    for scalar input), which indexes numpy arrays correctly.

    Caller contract
    ---------------
    ``value`` MUST be pre-masked. Callers are responsible for filtering
    NaN and out-of-range entries (via ``np.isfinite()`` and range
    comparisons, or via the ``in_range`` mask returned by
    :func:`_compute_bins`) before using the returned indices. This
    function does **not** detect or flag invalid inputs:

    - NaN inputs produce platform-dependent garbage integer values,
      which are then silently clipped into ``[0, n-1]``.
    - Out-of-range finite values are silently clipped to bin 0 or
      ``n-1`` rather than rejected.

    Using the indices unmasked would corrupt the accumulated grid with
    fictitious counts at the boundary bins. Prefer :func:`_compute_bins`
    or :func:`precompute_bins` (the cache variant) when consuming
    indices directly — both return an ``in_range`` mask alongside the
    indices.
    """
    frac = (np.asarray(value, dtype=float) - vmin) / (vmax - vmin)
    with np.errstate(invalid="ignore"):
        idx = np.floor(frac * n).astype(int)
    return np.clip(idx, 0, n - 1)


def _accumulate_grid(
    i_r: np.ndarray,
    i_lat: np.ndarray,
    i_lt: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int, int],
    dt_minutes: float,
) -> np.ndarray:
    """Accumulate dwell time into a 3D grid using bincount.

    Returns float32 minutes. Counts are accumulated as int64 (bincount
    default), multiplied by ``dt_minutes`` in float64 to preserve
    precision for large ``dt_minutes × count`` products, then cast to
    float32. Halves resident memory of the output dict vs float64 with
    no observable change in the downstream zarr (already float32).
    """
    ir, ilat, ilt = i_r[mask], i_lat[mask], i_lt[mask]
    if len(ir) == 0:
        return np.zeros(shape, dtype=np.float32)
    flat = np.ravel_multi_index((ir, ilat, ilt), shape)
    counts = np.bincount(flat, minlength=math.prod(shape)).reshape(shape)
    return (counts.astype(np.float64) * dt_minutes).astype(np.float32)


def _accumulate_grid_2d(
    i_lat: np.ndarray,
    i_lt: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int],
    dt_minutes: float,
) -> np.ndarray:
    """Accumulate dwell time into a 2D (lat, LT) grid using bincount.

    See :func:`_accumulate_grid` for the int64-counts → float64-multiply
    → float32-output precision/storage contract.
    """
    il, ilt = i_lat[mask], i_lt[mask]
    if len(il) == 0:
        return np.zeros(shape, dtype=np.float32)
    flat = np.ravel_multi_index((il, ilt), shape)
    counts = np.bincount(flat, minlength=math.prod(shape)).reshape(shape)
    return (counts.astype(np.float64) * dt_minutes).astype(np.float32)


# ----------------------------------------------------------------------
# Shared coord/bin cache — lets callers compute (r, mag_lat, LT, inv_lat)
# once per trajectory and feed the same cache to multiple accumulators.
# Without this, accumulate_with_regions, accumulate_inv_lat_grid, and
# accumulate_weak_field_grid each re-derive coords + bin indices, which
# dominates the binning cost for the full-mirror schema.
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BinCache:
    """Pre-computed bin indices + in-range masks for a trajectory.

    Holds the integer bin indices and the boolean in-range masks for
    both the 3D ``(r, mag_lat, LT)`` schema (``in_range_3d``) and the
    2D ``(dipole_inv_lat, LT)`` schema (``in_range_2d``, which also
    excludes NaN inv_lat for L<1).

    Use :func:`precompute_bins` to build, then pass to the
    ``*_cached`` accumulator variants.
    """

    i_r: np.ndarray  # (N,) int
    i_lat: np.ndarray  # (N,) int — offset-dipole magnetic latitude
    i_lt: np.ndarray  # (N,) int
    i_inv_lat: np.ndarray  # (N,) int — dipole invariant latitude
    in_range_3d: np.ndarray  # (N,) bool
    in_range_2d: np.ndarray  # (N,) bool (NaN inv_lat excluded)


def precompute_bins(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    config: DwellGridConfig | None = None,
) -> BinCache:
    """Compute bin indices once for a trajectory; reuse across schemas."""
    if config is None:
        config = DwellGridConfig()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r, lat, lt = _compute_coords(x, y, z)
    inv_lat = dipole_invariant_latitude(x, y, z)

    i_r = _bin_index(r, *config.r_range, config.n_r)
    i_lat = _bin_index(lat, *config.lat_range, config.n_lat)
    i_lt = _bin_index(lt, *config.lt_range, config.n_lt)
    i_inv_lat = _bin_index(inv_lat, *config.lat_range, config.n_lat)

    in_range_3d = (
        (r >= config.r_range[0])
        & (r < config.r_range[1])
        & (lat >= config.lat_range[0])
        & (lat < config.lat_range[1])
        & (lt >= config.lt_range[0])
        & (lt < config.lt_range[1])
    )
    in_range_2d = (
        np.isfinite(inv_lat)
        & (inv_lat >= config.lat_range[0])
        & (inv_lat < config.lat_range[1])
        & (lt >= config.lt_range[0])
        & (lt < config.lt_range[1])
    )
    return BinCache(i_r, i_lat, i_lt, i_inv_lat, in_range_3d, in_range_2d)


def _region_masks(
    region_codes: np.ndarray,
    base: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build per-region masks from a base (in-range) mask."""
    out = {"total": base}
    for code, name in REGION_CODES.items():
        out[name] = base & (region_codes == code)
    return out


def accumulate_with_regions_cached(
    cache: BinCache,
    region_codes: ArrayLike,
    dt_minutes: float = 1.0,
    *,
    mask: ArrayLike | None = None,
    config: DwellGridConfig | None = None,
) -> dict[str, np.ndarray]:
    """Cache-aware variant of :func:`accumulate_with_regions`."""
    if config is None:
        config = DwellGridConfig()
    codes = np.asarray(region_codes, dtype=int)
    base = cache.in_range_3d
    if mask is not None:
        base = base & np.asarray(mask, dtype=bool)
    masks = _region_masks(codes, base)
    return {
        name: _accumulate_grid(
            cache.i_r,
            cache.i_lat,
            cache.i_lt,
            m,
            config.shape,
            dt_minutes,
        )
        for name, m in masks.items()
    }


def accumulate_inv_lat_grid_cached(
    cache: BinCache,
    region_codes: ArrayLike | None = None,
    dt_minutes: float = 1.0,
    *,
    mask: ArrayLike | None = None,
    config: DwellGridConfig | None = None,
) -> dict[str, np.ndarray]:
    """Cache-aware variant of :func:`accumulate_inv_lat_grid`."""
    if config is None:
        config = DwellGridConfig()
    shape_2d = (config.n_lat, config.n_lt)
    base = cache.in_range_2d
    if mask is not None:
        base = base & np.asarray(mask, dtype=bool)

    result = {
        "total": _accumulate_grid_2d(
            cache.i_inv_lat,
            cache.i_lt,
            base,
            shape_2d,
            dt_minutes,
        ),
    }
    if region_codes is not None:
        codes = np.asarray(region_codes, dtype=int)
        for code, name in REGION_CODES.items():
            result[name] = _accumulate_grid_2d(
                cache.i_inv_lat,
                cache.i_lt,
                base & (codes == code),
                shape_2d,
                dt_minutes,
            )
    return result


def accumulate_weak_field_grid_cached(
    cache: BinCache,
    btotal: ArrayLike,
    dt_minutes: float = 1.0,
    b_threshold: float = 2.0,
    region_codes: ArrayLike | None = None,
    *,
    mask: ArrayLike | None = None,
    config: DwellGridConfig | None = None,
) -> dict[str, np.ndarray]:
    """Cache-aware variant of :func:`accumulate_weak_field_grid`."""
    if config is None:
        config = DwellGridConfig()
    shape_2d = (config.n_lat, config.n_lt)
    bt = np.asarray(btotal, dtype=float)
    base = cache.in_range_2d & (bt < b_threshold)
    if mask is not None:
        base = base & np.asarray(mask, dtype=bool)

    result = {
        "total": _accumulate_grid_2d(
            cache.i_inv_lat,
            cache.i_lt,
            base,
            shape_2d,
            dt_minutes,
        ),
    }
    if region_codes is not None:
        codes = np.asarray(region_codes, dtype=int)
        for code, name in REGION_CODES.items():
            result[name] = _accumulate_grid_2d(
                cache.i_inv_lat,
                cache.i_lt,
                base & (codes == code),
                shape_2d,
                dt_minutes,
            )
    return result


def accumulate_dwell_time(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    dt_minutes: float = 1.0,
    config: DwellGridConfig | None = None,
    stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    r"""Accumulate spacecraft dwell time into a 3D (r, mag_lat, LT) grid.

    Parameters
    ----------
    x, y, z : array_like
        Spacecraft position in KSM coordinates ($R_S$).
    dt_minutes : float
        Time step per sample in minutes (default 1.0 for 1-min MAG data).
    config : DwellGridConfig, optional
        Grid configuration. Uses defaults if not provided.
    stats : bool
        If True, return ``(grid, info)`` where ``info`` is a dict with
        out-of-range statistics. Useful for tuning grid bounds.

    Returns
    -------
    ndarray, shape (n_r, n_lat, n_lt)
        Accumulated dwell time in minutes. If ``stats=True``, returns
        ``(grid, info)`` instead.
    """
    if config is None:
        config = DwellGridConfig()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r, lat, lt = _compute_coords(x, y, z)
    i_r, i_lat, i_lt, in_range = _compute_bins(r, lat, lt, config)
    grid = _accumulate_grid(i_r, i_lat, i_lt, in_range, config.shape, dt_minutes)

    if stats:
        n_total = len(x)
        n_in = int(in_range.sum())
        info = {
            "n_total": n_total,
            "n_in_range": n_in,
            "n_out_of_range": n_total - n_in,
            "pct_in_range": n_in / n_total * 100 if n_total > 0 else 0.0,
            "r_max_observed": float(r.max()) if len(r) > 0 else 0.0,
            "r_out_high": int((r >= config.r_range[1]).sum()),
            "lat_out": int(
                ((lat < config.lat_range[0]) | (lat >= config.lat_range[1])).sum()
            ),
        }
        return grid, info

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
        Location code for each sample (see ``REGION_CODES``):
        0=magnetosphere, 1=magnetosheath, 2=solar_wind, 9=unknown.
    dt_minutes : float
        Time step per sample in minutes.
    config : DwellGridConfig, optional
        Grid configuration.

    Returns
    -------
    dict
        Keys: ``'total'``, ``'magnetosphere'``, ``'magnetosheath'``,
        ``'solar_wind'``, ``'unknown'``.
        Values: 3D numpy arrays of shape ``config.shape``.
        The ``'total'`` grid includes all in-range samples regardless
        of region code.
    """
    if config is None:
        config = DwellGridConfig()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    codes = np.asarray(region_codes, dtype=int)

    # Compute coordinates and bin indices once
    r, lat, lt = _compute_coords(x, y, z)
    i_r, i_lat, i_lt, in_range = _compute_bins(r, lat, lt, config)

    # Total includes ALL in-range samples regardless of region code
    result = {
        "total": _accumulate_grid(
            i_r,
            i_lat,
            i_lt,
            in_range,
            config.shape,
            dt_minutes,
        ),
    }

    # Per-region grids for known codes
    for code, name in REGION_CODES.items():
        mask = in_range & (codes == code)
        result[name] = _accumulate_grid(
            i_r,
            i_lat,
            i_lt,
            mask,
            config.shape,
            dt_minutes,
        )

    # Warn about unrecognized region codes
    known = np.isin(codes, list(REGION_CODES.keys()))
    n_unrecognized = int((in_range & ~known).sum())
    if n_unrecognized > 0:
        log.warning(
            "%d in-range samples have unrecognized region codes "
            "(included in 'total' but not in any per-region grid)",
            n_unrecognized,
        )

    return result


def accumulate_inv_lat_grid(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    dt_minutes: float = 1.0,
    region_codes: ArrayLike | None = None,
    config: DwellGridConfig | None = None,
) -> dict[str, np.ndarray]:
    r"""Accumulate dwell time in a 2D (dipole_inv_lat, local_time) grid.

    Uses the analytical dipole invariant latitude (footpoint latitude)
    instead of numerical field line tracing. Points inside the planet
    ($L < 1$) are excluded.

    Parameters
    ----------
    x, y, z : array_like
        Spacecraft position in KSM coordinates ($R_S$).
    dt_minutes : float
        Time step per sample in minutes.
    region_codes : array_like of int, optional
        Location codes (0=MS, 1=SH, 2=SW, 9=unknown). If provided,
        per-region grids are returned.
    config : DwellGridConfig, optional
        Uses ``n_lat`` and ``n_lt`` for the grid shape (``n_r`` is ignored).

    Returns
    -------
    dict
        Keys: ``'total'`` (and per-region names if ``region_codes`` given).
        Values: 2D arrays of shape ``(n_lat, n_lt)``.
    """
    if config is None:
        config = DwellGridConfig()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    inv_lat = dipole_invariant_latitude(x, y, z)
    lt = local_time(x, y)

    shape_2d = (config.n_lat, config.n_lt)
    i_lat = _bin_index(inv_lat, *config.lat_range, config.n_lat)
    i_lt = _bin_index(lt, *config.lt_range, config.n_lt)

    # Valid: not NaN (L >= 1) and within lat/lt ranges
    valid = (
        np.isfinite(inv_lat)
        & (inv_lat >= config.lat_range[0])
        & (inv_lat < config.lat_range[1])
        & (lt >= config.lt_range[0])
        & (lt < config.lt_range[1])
    )

    def accum(mask: np.ndarray) -> np.ndarray:
        return _accumulate_grid_2d(i_lat, i_lt, mask, shape_2d, dt_minutes)

    result = {"total": accum(valid)}

    if region_codes is not None:
        codes = np.asarray(region_codes, dtype=int)
        for code, name in REGION_CODES.items():
            result[name] = accum(valid & (codes == code))

    return result


def accumulate_traced_inv_lat_grid(
    inv_lat_north: ArrayLike,
    inv_lat_south: ArrayLike,
    is_closed: ArrayLike,
    local_time: ArrayLike,
    z: ArrayLike,
    dt_minutes: float = 60.0,
    region_codes: ArrayLike | None = None,
    closed_only: bool = False,
    config: DwellGridConfig | None = None,
) -> dict[str, np.ndarray]:
    r"""Accumulate dwell time in a 2D (inv_lat, LT) grid from traced KMAG field lines.

    Uses pre-computed invariant latitudes from ``compute_invariant_latitudes()``
    rather than an analytical dipole formula. The conjugate latitude is chosen
    from the footpoint in the spacecraft's hemisphere (signed by hemisphere),
    matching the convention of ``dipole_invariant_latitude()``.

    Parameters
    ----------
    inv_lat_north, inv_lat_south : array_like
        Northern and southern footpoint latitudes in degrees, from
        ``compute_invariant_latitudes()``. NaN for open field lines.
    is_closed : array_like of bool
        True where the field line is closed.
    local_time : array_like
        Local time in hours at each traced position.
    z : array_like
        KSM $z$-coordinate at each traced position ($R_S$). Used to
        determine spacecraft hemisphere for conjugate latitude sign.
    dt_minutes : float
        Dwell time per sample in minutes. For subsampled tracing this
        equals ``trace_every_n`` (e.g. 60 min for hourly traces).
    region_codes : array_like of int, optional
        Region codes at each traced position. If provided, per-region
        grids are returned.
    closed_only : bool
        If True, only accumulate points on closed field lines.
    config : DwellGridConfig, optional
        Uses ``n_lat`` and ``n_lt`` for grid shape.

    Returns
    -------
    dict
        Keys: ``'total'`` (and per-region names if ``region_codes`` given).
        Values: 2D arrays of shape ``(n_lat, n_lt)``.
    """
    if config is None:
        config = DwellGridConfig()

    inv_n = np.asarray(inv_lat_north, dtype=float)
    inv_s = np.asarray(inv_lat_south, dtype=float)
    closed = np.asarray(is_closed, dtype=bool)
    lt = np.asarray(local_time, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    # Conjugate latitude: footpoint in **spacecraft's** hemisphere
    # (sign of z), not the field-line's. So the inv_lat × LT map shows
    # where Cassini was when the line was traced, mirrored to a single
    # signed footpoint. North/south asymmetries in the resulting grid
    # reflect orbital coverage, not intrinsic field-line asymmetry.
    conjugate_lat = np.where(z_arr >= 0, inv_n, inv_s)

    shape_2d = (config.n_lat, config.n_lt)
    i_lat = _bin_index(conjugate_lat, *config.lat_range, config.n_lat)
    i_lt = _bin_index(lt, *config.lt_range, config.n_lt)

    valid = (
        np.isfinite(conjugate_lat)
        & (conjugate_lat >= config.lat_range[0])
        & (conjugate_lat < config.lat_range[1])
        & (lt >= config.lt_range[0])
        & (lt < config.lt_range[1])
    )
    if closed_only:
        valid &= closed

    def accum(mask: np.ndarray) -> np.ndarray:
        return _accumulate_grid_2d(i_lat, i_lt, mask, shape_2d, dt_minutes)

    result = {"total": accum(valid)}

    if region_codes is not None:
        codes = np.asarray(region_codes, dtype=int)
        for code, name in REGION_CODES.items():
            result[name] = accum(valid & (codes == code))

    return result


def accumulate_weak_field_grid(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    btotal: ArrayLike,
    dt_minutes: float = 1.0,
    b_threshold: float = 2.0,
    region_codes: ArrayLike | None = None,
    config: DwellGridConfig | None = None,
) -> dict[str, np.ndarray]:
    r"""Accumulate dwell time in 2D (dipole_inv_lat, LT) grid for weak-field regions.

    Filters samples where $|B| < $ ``b_threshold`` before accumulating,
    providing a proxy for plasma sheet dwell time (SI Fig 2).

    Uses the analytical dipole invariant latitude (no tracing needed).

    Parameters
    ----------
    x, y, z : array_like
        Spacecraft position in KSM coordinates ($R_S$).
    btotal : array_like
        Total magnetic field magnitude in nT.
    dt_minutes : float
        Time step per sample in minutes.
    b_threshold : float
        Maximum $|B|$ in nT to include (default 2.0).
    region_codes : array_like of int, optional
        Region codes. If provided, per-region grids are returned.
    config : DwellGridConfig, optional
        Uses ``n_lat`` and ``n_lt`` for grid shape.

    Returns
    -------
    dict
        Keys: ``'total'`` (and per-region names if ``region_codes`` given).
        Values: 2D arrays of shape ``(n_lat, n_lt)``.
    """
    if config is None:
        config = DwellGridConfig()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    bt = np.asarray(btotal, dtype=float)

    inv_lat = dipole_invariant_latitude(x, y, z)
    lt = local_time(x, y)

    shape_2d = (config.n_lat, config.n_lt)
    i_lat = _bin_index(inv_lat, *config.lat_range, config.n_lat)
    i_lt = _bin_index(lt, *config.lt_range, config.n_lt)

    valid = (
        np.isfinite(inv_lat)
        & (inv_lat >= config.lat_range[0])
        & (inv_lat < config.lat_range[1])
        & (lt >= config.lt_range[0])
        & (lt < config.lt_range[1])
        & (bt < b_threshold)
    )

    def accum(mask: np.ndarray) -> np.ndarray:
        return _accumulate_grid_2d(i_lat, i_lt, mask, shape_2d, dt_minutes)

    result = {"total": accum(valid)}

    if region_codes is not None:
        codes = np.asarray(region_codes, dtype=int)
        for code, name in REGION_CODES.items():
            result[name] = accum(valid & (codes == code))

    return result


def accumulate_kmag_eq_r_grid(
    l_equatorial: ArrayLike,
    is_closed: ArrayLike,
    local_time: ArrayLike,
    dt_minutes: float = 60.0,
    region_codes: ArrayLike | None = None,
    closed_only: bool = False,
    config: DwellGridConfig | None = None,
) -> dict[str, np.ndarray]:
    r"""Accumulate dwell time on a 2D ``(kmag_eq_r, local_time)`` grid.

    Each spacecraft sample contributes to the bin given by the
    *equatorial apex* of its KMAG-traced field line -- the maximum
    radial distance reached by the field line, which corresponds to
    the equatorial-plane crossing for closed lines. This is the
    natural axis for visualising field-line-resonance occurrence:
    a wave detected at any latitude on a field line maps to the same
    equatorial-r bin as one detected at the apex itself.

    Parameters
    ----------
    l_equatorial : array_like
        Equatorial apex distance ($R_S$), one entry per traced
        position. Aligned to ``compute_invariant_latitudes`` output
        (NaN for open / failed traces).
    is_closed : array_like of bool
        Whether the field line is closed at both footpoints.
    local_time : array_like
        Local time in hours, one entry per traced position.
    dt_minutes : float, default 60.0
        Dwell time per traced sample in minutes (matches
        ``trace_every_n`` x cadence -- e.g. 10 for 10-min subsampling
        of 1-min data).
    region_codes : array_like of int, optional
        If given, also produce per-region grids using the standard
        ``REGION_CODES`` mapping.
    closed_only : bool, default False
        If True, drop open field lines.
    config : DwellGridConfig, optional
        Grid axes. The radial axis ``r_range, n_r`` is reused for the
        ``kmag_eq_r`` dimension so the output bins match the standard
        spacecraft-position radial bins.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``'total'`` and (if ``region_codes`` is given) the
        region-name keys from :data:`REGION_CODES`. Values: 2D arrays
        of shape ``(n_r, n_lt)`` in minutes.
    """
    if config is None:
        config = DwellGridConfig()

    l_eq = np.asarray(l_equatorial, dtype=float)
    closed = np.asarray(is_closed, dtype=bool)
    lt = np.asarray(local_time, dtype=float)

    shape_2d = (config.n_r, config.n_lt)
    i_r = _bin_index(l_eq, *config.r_range, config.n_r)
    i_lt = _bin_index(lt, *config.lt_range, config.n_lt)

    valid = (
        np.isfinite(l_eq)
        & (l_eq >= config.r_range[0])
        & (l_eq < config.r_range[1])
        & (lt >= config.lt_range[0])
        & (lt < config.lt_range[1])
    )
    if closed_only:
        valid &= closed

    def accum(mask: np.ndarray) -> np.ndarray:
        return _accumulate_grid_2d(i_r, i_lt, mask, shape_2d, dt_minutes)

    result = {"total": accum(valid)}

    if region_codes is not None:
        codes = np.asarray(region_codes, dtype=int)
        for code, name in REGION_CODES.items():
            result[name] = accum(valid & (codes == code))

    return result
