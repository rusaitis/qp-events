"""Dwell-time accumulation: spherical grids, xarray I/O, KMAG tracing."""

from qp.dwell.grid import (
    DwellGridConfig,
    accumulate_dwell_time,
    accumulate_inv_lat_grid,
    accumulate_traced_inv_lat_grid,
    accumulate_weak_field_grid,
    accumulate_with_regions,
)
from qp.dwell.io import ZarrEncoding, load_zarr, save_zarr, to_xarray
from qp.dwell.tracing import (
    TracingConfig,
    TracingResult,
    compute_invariant_latitudes,
    compute_invariant_latitudes_parallel,
)
