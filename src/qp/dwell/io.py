r"""xarray/zarr I/O for dwell-time grids.

Converts 3D numpy arrays + DwellGridConfig into self-describing
xarray Datasets with named dimensions, coordinate arrays, and metadata,
then saves/loads as zarr stores.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from qp.dwell.grid import DwellGridConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ZarrEncoding:
    """Configuration for zarr storage encoding.

    Parameters
    ----------
    compressor : str
        Compression algorithm: ``'zstd'``, ``'blosc'``, or ``'none'``.
    compression_level : int
        Compression level (1-9, higher = slower but smaller).
    chunks : tuple[int, ...] or None
        Chunk shape for each variable. None = zarr auto-chunking.
    dtype : str
        Data type for storage: ``'float32'`` or ``'float64'``.
    """

    compressor: str = "zstd"
    compression_level: int = 3
    chunks: tuple[int, ...] | None = None
    dtype: str = "float32"


def _make_compressor(encoding: ZarrEncoding):
    """Create a zarr v3 codec from ZarrEncoding config."""
    from zarr.codecs import BloscCodec, ZstdCodec

    match encoding.compressor:
        case "zstd":
            return ZstdCodec(level=encoding.compression_level)
        case "blosc":
            return BloscCodec(
                cname="lz4", clevel=encoding.compression_level,
                typesize=4 if encoding.dtype == "float32" else 8,
            )
        case "none":
            return None
        case _:
            raise ValueError(
                f"Unknown compressor {encoding.compressor!r}. "
                "Use 'zstd', 'blosc', or 'none'."
            )


def to_xarray(
    grids: dict[str, np.ndarray],
    config: DwellGridConfig,
    attrs: dict | None = None,
    tracing_config=None,
    field_config=None,
    inv_lat_grids: dict[str, np.ndarray] | None = None,
) -> xr.Dataset:
    r"""Convert dwell-time grids to an xarray Dataset.

    Parameters
    ----------
    grids : dict
        Keys are variable names (e.g. ``'total'``, ``'magnetosphere'``).
        Values are 3D arrays of shape ``config.shape``.
    config : DwellGridConfig
        Grid configuration (provides coordinate arrays).
    attrs : dict, optional
        Global attributes (metadata) to attach to the Dataset.
    tracing_config : TracingConfig, optional
        If provided, tracing parameters are stored in metadata.
    field_config : SaturnFieldConfig, optional
        If provided, field model parameters are stored in metadata.
    inv_lat_grids : dict, optional
        2D grids of shape ``(n_lat, n_lt)`` keyed by name (e.g.
        ``'dipole_inv_lat_total'``). Uses ``(dipole_inv_lat, local_time)``
        dimensions.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(r, magnetic_latitude, local_time)``
        and optionally ``(dipole_inv_lat, local_time)`` for invariant
        latitude grids.
    """
    coords = {
        # Dimension coordinates (bin centers)
        "r": ("r", config.r_centers, {"units": "R_S", "long_name": "Radial distance (bin center)"}),
        "magnetic_latitude": (
            "magnetic_latitude",
            config.lat_centers,
            {"units": "degrees", "long_name": "Magnetic latitude, KSM offset dipole (bin center)"},
        ),
        "local_time": (
            "local_time",
            config.lt_centers,
            {"units": "h", "long_name": "Local time (bin center)"},
        ),
        # Non-dimension coordinates (bin edges, for exact reconstruction)
        "r_edges": ("r_edge", config.r_edges, {"units": "R_S", "long_name": "Radial bin edges"}),
        "lat_edges": ("lat_edge", config.lat_edges, {"units": "degrees", "long_name": "Latitude bin edges"}),
        "lt_edges": ("lt_edge", config.lt_edges, {"units": "h", "long_name": "Local time bin edges"}),
    }

    dims = ("r", "magnetic_latitude", "local_time")
    data_vars = {}
    for name, arr in grids.items():
        data_vars[name] = (
            dims,
            arr,
            {"units": "min", "long_name": f"Dwell time ({name})"},
        )

    # 2D invariant latitude grids (dipole_inv_lat × local_time)
    if inv_lat_grids:
        inv_lat_dim = "dipole_inv_lat"
        coords[inv_lat_dim] = (
            inv_lat_dim,
            config.lat_centers,  # same binning as magnetic_latitude
            {"units": "degrees", "long_name": "Dipole invariant latitude (bin center)"},
        )
        inv_dims = (inv_lat_dim, "local_time")
        for name, arr in inv_lat_grids.items():
            data_vars[name] = (
                inv_dims,
                arr,
                {"units": "min", "long_name": f"Dwell time by dipole inv lat ({name})"},
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

    if tracing_config is not None:
        default_attrs["trace_every_n"] = tracing_config.trace_every_n
        default_attrs["trace_step_RS"] = tracing_config.step
        default_attrs["trace_max_radius_RS"] = tracing_config.max_radius
        default_attrs["trace_min_radius_RS"] = tracing_config.min_radius
        default_attrs["trace_surface_tolerance_RS"] = tracing_config.surface_tolerance
        default_attrs["trace_max_steps"] = tracing_config.max_steps

    if field_config is not None:
        default_attrs["dp_nPa"] = field_config.dp
        default_attrs["by_imf_nT"] = field_config.by_imf
        default_attrs["bz_imf_nT"] = field_config.bz_imf

    if attrs:
        default_attrs.update(attrs)

    return xr.Dataset(data_vars, coords=coords, attrs=default_attrs)


def save_zarr(
    ds: xr.Dataset,
    path: str | Path,
    encoding: ZarrEncoding | None = None,
) -> None:
    """Save an xarray Dataset to a zarr store.

    Parameters
    ----------
    ds : xr.Dataset
    path : str or Path
        Output directory (will be created/overwritten).
    encoding : ZarrEncoding, optional
        Compression and dtype settings. If None, uses zarr defaults
        (no compression, float64).
    """
    path = Path(path)

    if encoding is not None:
        comp = _make_compressor(encoding)
        enc = {}
        for var in ds.data_vars:
            var_enc: dict = {"dtype": encoding.dtype}
            if comp is not None:
                var_enc["compressors"] = comp
            if encoding.chunks is not None:
                var_enc["chunks"] = encoding.chunks
            enc[var] = var_enc
        ds.to_zarr(path, mode="w", encoding=enc)
        log.info(
            "Saved zarr store to %s (compressor=%s, dtype=%s)",
            path, encoding.compressor, encoding.dtype,
        )
    else:
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
