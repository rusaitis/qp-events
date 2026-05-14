"""Magnetic field line tracing and KMAG interface.

For dwell-time accumulation see :mod:`qp.dwell.grid` (canonical pipeline
behind ``Output/dwell_grid_cassini_saturn.zarr``).
"""

from qp.analysis.filtering import (
    bin_centers,
    bin_edges,
    bin_to_value,
    value_to_bin,
)
from qp.fieldline.kmag_model import (
    SaturnField,
    SaturnFieldConfig,
)
from qp.fieldline.saturn_coords import (
    epoch_to_j2000,
    rotate_vector,
    sun_position,
)
from qp.fieldline.tracer import (
    FieldLineTrace,
    conjugate_latitude,
    dipole_field,
    field_line_to_spherical,
    saturn_field_wrapper,
    trace_dipole_fieldline,
    trace_dipole_fieldline_bidirectional,
    trace_fieldline,
    trace_fieldline_bidirectional,
)
