r"""Batch KMAG field line tracing for invariant latitude computation.

Traces field lines from spacecraft positions using the Python KMAG model
and extracts conjugate (invariant) latitude at Saturn's surface.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

from qp.fieldline.kmag_model import SaturnField
from qp.fieldline.tracer import (
    conjugate_latitude,
    saturn_field_wrapper,
    trace_fieldline_bidirectional,
)
log = logging.getLogger(__name__)


def compute_invariant_latitudes(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    times_unix: ArrayLike,
    trace_every_n: int = 60,
    step: float = 0.1,
    max_radius: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute invariant latitude for a sequence of spacecraft positions.

    Traces a KMAG field line from each position (subsampled by
    ``trace_every_n``) and extracts the conjugate latitude where the
    field line intersects Saturn's surface.

    Parameters
    ----------
    x, y, z : array_like
        Spacecraft positions in KSM coordinates ($R_S$).
    times_unix : array_like
        POSIX timestamps for each position (needed for KMAG epoch).
    trace_every_n : int
        Trace every N-th point (default 60 = hourly for 1-min data).
    step : float
        RK4 step size in $R_S$ (default 0.1).
    max_radius : float
        Maximum tracing radius in $R_S$ (default 100).

    Returns
    -------
    inv_lat_north : ndarray
        Invariant latitude of northern footpoint (degrees). NaN if open.
    inv_lat_south : ndarray
        Invariant latitude of southern footpoint (degrees). NaN if open.
    is_closed : ndarray of bool
        True if the field line is closed (both footpoints found).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    times = np.asarray(times_unix, dtype=float)

    n_total = len(x)
    indices = np.arange(0, n_total, trace_every_n)
    n_traces = len(indices)

    inv_lat_north = np.full(n_traces, np.nan)
    inv_lat_south = np.full(n_traces, np.nan)
    is_closed = np.zeros(n_traces, dtype=bool)

    field = SaturnField()

    log.info(
        "Tracing %d field lines (every %d samples from %d total)",
        n_traces, trace_every_n, n_total,
    )

    for count, idx in enumerate(indices):
        pos = np.array([x[idx], y[idx], z[idx]])
        r = np.linalg.norm(pos)

        # Skip points too close to planet or too far out
        if r < 1.5 or r > max_radius:
            continue

        field_func = saturn_field_wrapper(field, time=times[idx], coord="KSM")

        try:
            trace = trace_fieldline_bidirectional(
                field_func, pos, step=step,
                max_radius=max_radius, min_radius=1.0,
            )
        except Exception:
            log.debug("Trace failed at index %d (r=%.1f)", idx, r)
            continue

        # Extract northern footpoint (min colatitude = highest latitude)
        rtp = np.column_stack([
            np.linalg.norm(trace.positions, axis=1),
            np.arctan2(
                np.sqrt(trace.positions[:, 0]**2 + trace.positions[:, 1]**2),
                trace.positions[:, 2],
            ),
        ])
        r_trace = rtp[:, 0]

        # Find footpoints near surface
        near_surface = r_trace < 1.5
        if not np.any(near_surface):
            continue

        # Use conjugate_latitude for the full trace
        lat = conjugate_latitude(trace.positions, surface_radius=1.0)
        if lat is not None:
            # The trace goes from one footpoint to the other
            # First footpoint is at the start, second at the end
            lat_start = conjugate_latitude(
                trace.positions[:len(trace.positions) // 2], surface_radius=1.0
            )
            lat_end = conjugate_latitude(
                trace.positions[len(trace.positions) // 2:], surface_radius=1.0
            )

            if lat_start is not None and lat_end is not None:
                # Assign north/south based on sign
                lats = sorted([lat_start, lat_end])
                inv_lat_south[count] = lats[0]
                inv_lat_north[count] = lats[1]
                is_closed[count] = True

        if (count + 1) % 1000 == 0:
            pct = (count + 1) / n_traces * 100
            n_closed = np.sum(is_closed[:count + 1])
            log.info(
                "Progress: %d/%d (%.1f%%) — %d closed",
                count + 1, n_traces, pct, n_closed,
            )

    n_closed = int(np.sum(is_closed))
    log.info("Tracing complete: %d/%d closed field lines", n_closed, n_traces)
    return inv_lat_north, inv_lat_south, is_closed
