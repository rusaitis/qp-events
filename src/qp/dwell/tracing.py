r"""Batch KMAG field line tracing for invariant latitude computation.

Traces field lines from spacecraft positions using the Python KMAG model
and extracts conjugate (invariant) latitude at Saturn's surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from qp.fieldline.kmag_model import SaturnField, SaturnFieldConfig
from qp.fieldline.tracer import (
    saturn_field_wrapper,
    trace_fieldline_bidirectional,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TracingConfig:
    r"""Configuration for batch field line tracing.

    Parameters
    ----------
    trace_every_n : int
        Trace every N-th point (default 60 = hourly for 1-min data).
    step : float
        RK4 integration step size in $R_S$.
    max_radius : float
        Outer tracing boundary in $R_S$. Field lines extending beyond
        this are treated as open.
    min_radius : float
        Inner tracing boundary in $R_S$ (planet surface).
    surface_tolerance : float
        A trace endpoint is considered "at the surface" if
        $r < $ ``surface_tolerance``. Also used to skip points
        too close to the planet.
    max_steps : int
        Maximum RK4 integration steps per trace arm.
    log_interval : int
        Log progress every N traces.
    """

    trace_every_n: int = 60
    step: float = 0.1
    max_radius: float = 100.0
    min_radius: float = 1.0
    surface_tolerance: float = 1.5
    max_steps: int = 100_000
    log_interval: int = 1000


def compute_invariant_latitudes(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    times_unix: ArrayLike,
    config: TracingConfig | None = None,
    field_config: SaturnFieldConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute invariant latitude for a sequence of spacecraft positions.

    Traces a KMAG field line from each position (subsampled by
    ``config.trace_every_n``) and extracts the conjugate latitude where
    the field line intersects Saturn's surface.

    Parameters
    ----------
    x, y, z : array_like
        Spacecraft positions in KSM coordinates ($R_S$).
    times_unix : array_like
        POSIX timestamps for each position (needed for KMAG epoch).
    config : TracingConfig, optional
        Tracing parameters. Uses defaults if not provided.
    field_config : SaturnFieldConfig, optional
        KMAG field model parameters ($d_p$, IMF $B_y$, $B_z$).
        Uses defaults if not provided.

    Returns
    -------
    inv_lat_north : ndarray
        Invariant latitude of northern footpoint (degrees). NaN if open.
    inv_lat_south : ndarray
        Invariant latitude of southern footpoint (degrees). NaN if open.
    is_closed : ndarray of bool
        True if the field line is closed (both footpoints found).
    """
    if config is None:
        config = TracingConfig()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    times = np.asarray(times_unix, dtype=float)

    n_total = len(x)
    indices = np.arange(0, n_total, config.trace_every_n)
    n_traces = len(indices)

    inv_lat_north = np.full(n_traces, np.nan)
    inv_lat_south = np.full(n_traces, np.nan)
    is_closed = np.zeros(n_traces, dtype=bool)

    field = SaturnField(field_config)

    log.info(
        "Tracing %d field lines (every %d samples from %d total, "
        "step=%.3f R_S, max_r=%.0f R_S)",
        n_traces, config.trace_every_n, n_total,
        config.step, config.max_radius,
    )
    if field_config:
        log.info(
            "  Field model: dp=%.4f nPa, By=%.2f nT, Bz=%.2f nT",
            field_config.dp, field_config.by_imf, field_config.bz_imf,
        )

    for count, idx in enumerate(indices):
        pos = np.array([x[idx], y[idx], z[idx]])
        r = np.linalg.norm(pos)

        # Skip points too close to planet or too far out
        if r < config.surface_tolerance or r > config.max_radius:
            continue

        field_func = saturn_field_wrapper(field, time=times[idx], coord="KSM")

        try:
            trace = trace_fieldline_bidirectional(
                field_func, pos, step=config.step,
                max_radius=config.max_radius,
                min_radius=config.min_radius,
                max_steps=config.max_steps,
            )
        except Exception as exc:
            log.debug("Trace failed at index %d (r=%.1f): %s", idx, r, exc)
            continue

        # Extract footpoints from trace endpoints.
        # trace_fieldline_bidirectional terminates each arm at min_radius,
        # so the first and last positions are near the surface for closed
        # field lines.
        rtp = trace.spherical  # compute once: (N, 3) in (r, theta, phi)
        r_first, r_last = rtp[0, 0], rtp[-1, 0]

        if r_first < config.surface_tolerance and r_last < config.surface_tolerance:
            lat_first = 90.0 - np.degrees(rtp[0, 1])
            lat_last = 90.0 - np.degrees(rtp[-1, 1])
            lats = sorted([lat_first, lat_last])
            inv_lat_south[count] = lats[0]
            inv_lat_north[count] = lats[1]
            is_closed[count] = True

        if (count + 1) % config.log_interval == 0:
            pct = (count + 1) / n_traces * 100
            n_closed = np.sum(is_closed[:count + 1])
            log.info(
                "Progress: %d/%d (%.1f%%) — %d closed",
                count + 1, n_traces, pct, n_closed,
            )

    n_closed = int(np.sum(is_closed))
    log.info("Tracing complete: %d/%d closed field lines", n_closed, n_traces)
    return inv_lat_north, inv_lat_south, is_closed
