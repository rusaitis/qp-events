"""Dwell-time accumulation: spherical grids, xarray I/O, KMAG tracing."""

from qp.dwell.grid import (
    DwellGridConfig,
    accumulate_dwell_time,
    accumulate_with_regions,
)
from qp.dwell.io import load_zarr, save_zarr, to_xarray
from qp.dwell.tracing import compute_invariant_latitudes
