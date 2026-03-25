r"""xarray/zarr I/O for dwell-time grids.

Converts 3D numpy arrays + DwellGridConfig into self-describing
xarray Datasets with named dimensions, coordinate arrays, and metadata,
then saves/loads as zarr stores.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from qp.dwell.grid import DwellGridConfig

log = logging.getLogger(__name__)


def to_xarray(
    grids: dict[str, np.ndarray],
    config: DwellGridConfig,
    attrs: dict | None = None,
) -> xr.Dataset:
    r"""Convert dwell-time grids to an xarray Dataset.

    Parameters
    ----------
    grids : dict
        Keys are variable names (e.g. ``'total'``, ``'ms'``, ``'sh'``, ``'sw'``).
        Values are 3D arrays of shape ``config.shape``.
    config : DwellGridConfig
        Grid configuration (provides coordinate arrays).
    attrs : dict, optional
        Global attributes (metadata) to attach to the Dataset.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(r, magnetic_latitude, local_time)``
        and coordinate arrays for bin centers.
    """
    coords = {
        "r": ("r", config.r_centers, {"units": "R_S", "long_name": "Radial distance"}),
        "magnetic_latitude": (
            "magnetic_latitude",
            config.lat_centers,
            {"units": "degrees", "long_name": "Magnetic latitude (KSM offset dipole)"},
        ),
        "local_time": (
            "local_time",
            config.lt_centers,
            {"units": "h", "long_name": "Local time"},
        ),
    }

    dims = ("r", "magnetic_latitude", "local_time")
    data_vars = {}
    for name, arr in grids.items():
        data_vars[name] = (
            dims,
            arr,
            {"units": "min", "long_name": f"Dwell time ({name})"},
        )

    default_attrs = {
        "title": "Cassini Dwell Time Grid",
        "description": "Accumulated spacecraft time in (r, mag_lat, LT) bins",
        "coordinate_system": "KSM",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_r": config.n_r,
        "n_lat": config.n_lat,
        "n_lt": config.n_lt,
        "r_range": list(config.r_range),
        "lat_range": list(config.lat_range),
        "lt_range": list(config.lt_range),
    }
    if attrs:
        default_attrs.update(attrs)

    return xr.Dataset(data_vars, coords=coords, attrs=default_attrs)


def save_zarr(ds: xr.Dataset, path: str | Path) -> None:
    """Save an xarray Dataset to a zarr store.

    Parameters
    ----------
    ds : xr.Dataset
    path : str or Path
        Output directory (will be created/overwritten).
    """
    path = Path(path)
    ds.to_zarr(path, mode="w")
    log.info("Saved zarr store to %s", path)


def load_zarr(path: str | Path) -> xr.Dataset:
    """Load an xarray Dataset from a zarr store.

    Parameters
    ----------
    path : str or Path
        Path to the zarr directory.

    Returns
    -------
    xr.Dataset
    """
    path = Path(path)
    ds = xr.open_zarr(path, decode_timedelta=False)
    log.info("Loaded zarr store from %s", path)
    return ds
