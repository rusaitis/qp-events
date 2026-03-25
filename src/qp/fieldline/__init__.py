"""Magnetic field line tracing, KMAG interface, and dwell time computation."""

from qp.fieldline.kmag_model import (
    SaturnField,
    SaturnFieldConfig,
)
from qp.fieldline.saturn_coords import (
    rotate_vector,
    sun_position,
    epoch_to_j2000,
)
from qp.fieldline.tracer import (
    FieldLineTrace,
    dipole_field,
    trace_fieldline,
    trace_fieldline_bidirectional,
    saturn_field_wrapper,
    trace_dipole_fieldline,
    trace_dipole_fieldline_bidirectional,
    field_line_to_spherical,
    conjugate_latitude,
)
from qp.fieldline.dwell_time import (
    value_to_bin,
    bin_to_value,
    bin_edges,
    bin_centers,
    compute_dwell_time_map,
    filter_dwell_map,
    reduce_to_lt_lat,
    reduce_to_mlat,
    reduce_to_lt,
    normalize_event_to_dwell,
)
