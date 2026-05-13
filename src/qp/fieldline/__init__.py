"""Magnetic field line tracing, KMAG interface, and dwell time computation."""

from qp.analysis.filtering import (
    bin_centers,
    bin_edges,
    bin_to_value,
    value_to_bin,
)
from qp.fieldline.dwell_time import (
    compute_dwell_time_map,
    filter_dwell_map,
    normalize_event_to_dwell,
    reduce_to_lt,
    reduce_to_lt_lat,
    reduce_to_mlat,
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
