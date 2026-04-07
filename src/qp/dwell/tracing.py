r"""Batch KMAG field line tracing for invariant latitude computation.

Traces field lines from spacecraft positions using the Python KMAG model
and extracts conjugate (invariant) latitude at Saturn's surface.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from qp.fieldline.kmag_model import (
    SaturnField,
    SaturnFieldConfig,
    make_kmag_field_func,
)
from qp.fieldline.tracer import trace_fieldline_bidirectional

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TracingResult:
    r"""Results from batch field line tracing.

    Parameters
    ----------
    inv_lat_north : ndarray
        Invariant latitude of northern footpoint (degrees). NaN if open.
    inv_lat_south : ndarray
        Invariant latitude of southern footpoint (degrees). NaN if open.
    is_closed : ndarray of bool
        True if the field line is closed (both footpoints at surface).
    l_equatorial : ndarray
        Maximum radial distance along the field line ($R_S$),
        i.e. the equatorial crossing distance (L-shell proxy).
        NaN if open or trace failed.
    n_traces : int
        Total number of traces attempted.
    n_closed : int
        Number of closed field lines found.
    """

    inv_lat_north: np.ndarray
    inv_lat_south: np.ndarray
    is_closed: np.ndarray
    l_equatorial: np.ndarray
    n_traces: int
    n_closed: int


@dataclass(frozen=True, slots=True)
class TracingConfig:
    r"""Configuration for batch field line tracing.

    Parameters
    ----------
    trace_every_n : int
        Trace every N-th point (default 10 = every 10 min for 1-min data).
    step : float
        RK4 integration step size in $R_S$.
    max_radius : float
        Outer tracing boundary in $R_S$. Field lines extending beyond
        this are treated as open. 60 R_S is enough for the Saturn auroral
        oval — apex > 60 maps past it anyway.
    min_radius : float
        Inner tracing boundary in $R_S$ (planet surface).
    surface_tolerance : float
        A trace endpoint is considered "at the surface" if
        $r < $ ``surface_tolerance``. Also used to skip points
        too close to the planet.
    max_steps : int
        Maximum RK4 integration steps per trace arm. Caps the cost of
        pathological traces (long open tail field lines).
    log_interval : int
        Log progress every N traces.
    region_filter : tuple[int, ...] or None
        If not None, only trace samples whose region code is in this
        tuple. Default ``(0,)`` = magnetosphere only (KMAG is not valid
        in magnetosheath/SW). Pass ``None`` to disable filtering.
    n_workers : int
        Number of multiprocessing workers for the parallel tracer.
        ``1`` (default) uses the serial path.
    chunk_size : int or None
        Round-robin chunk size for parallel tracing. ``None`` = auto.
    """

    trace_every_n: int = 10
    step: float = 0.15
    max_radius: float = 60.0
    min_radius: float = 1.0
    surface_tolerance: float = 1.5
    max_steps: int = 20_000
    log_interval: int = 1000
    region_filter: tuple[int, ...] | None = (0,)
    n_workers: int = 1
    chunk_size: int | None = None


def _interp_footpoint_lat(
    p_inner: np.ndarray, p_outer: np.ndarray, target_r: float = 1.0
) -> float:
    """Linearly extrapolate from p_inner past the surface to find latitude at r=target_r.

    The trace stops at p_inner (the last point above the surface) without taking
    the step that would have crossed it. p_outer is one step further from the
    surface. We extrapolate the chord (p_outer → p_inner) past p_inner until
    r=target_r and return the latitude at that point.

    For step=0.15 R_S, this reduces footpoint latitude error from ~1° to ~0.2°.
    """
    r_in = math.sqrt(p_inner[0] ** 2 + p_inner[1] ** 2 + p_inner[2] ** 2)
    r_out = math.sqrt(p_outer[0] ** 2 + p_outer[1] ** 2 + p_outer[2] ** 2)
    if r_out == r_in:
        # Degenerate (shouldn't happen for a real trace)
        x, y, z = float(p_inner[0]), float(p_inner[1]), float(p_inner[2])
    else:
        # t < 0 because target_r < r_in and (r_out - r_in) > 0
        t = (target_r - r_in) / (r_out - r_in)
        x = float(p_inner[0] + t * (p_outer[0] - p_inner[0]))
        y = float(p_inner[1] + t * (p_outer[1] - p_inner[1]))
        z = float(p_inner[2] + t * (p_outer[2] - p_inner[2]))
    r = math.sqrt(x * x + y * y + z * z)
    if r > 0:
        return math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
    return 0.0


def _trace_one(
    field: SaturnField,
    px: float,
    py: float,
    pz: float,
    t: float,
    config: TracingConfig,
) -> tuple[float, float, bool, float]:
    """Trace one field line and return (inv_lat_north, inv_lat_south, is_closed, l_eq).

    Returns NaN/False if the trace failed or the point is out of range.
    Footpoint latitudes use linear surface interpolation for sub-step accuracy.
    """
    r = math.sqrt(px * px + py * py + pz * pz)
    if r < config.surface_tolerance or r > config.max_radius:
        return float("nan"), float("nan"), False, float("nan")

    field_func = make_kmag_field_func(field, time=t, coord="KSM")
    try:
        trace = trace_fieldline_bidirectional(
            field_func, np.array([px, py, pz]), step=config.step,
            max_radius=config.max_radius,
            min_radius=config.min_radius,
            max_steps=config.max_steps,
        )
    except Exception as exc:
        log.debug("Trace failed at r=%.1f: %s", r, exc)
        return float("nan"), float("nan"), False, float("nan")

    rtp = trace.spherical
    r_first, r_last = rtp[0, 0], rtp[-1, 0]
    l_eq = float(np.max(rtp[:, 0]))

    inv_n = float("nan")
    inv_s = float("nan")
    closed = False
    if (
        r_first < config.surface_tolerance
        and r_last < config.surface_tolerance
        and len(trace.positions) >= 2
    ):
        # Surface interpolation: extrapolate the last chord at each end to r=1
        lat_first = _interp_footpoint_lat(trace.positions[0], trace.positions[1])
        lat_last = _interp_footpoint_lat(trace.positions[-1], trace.positions[-2])
        lats = sorted([lat_first, lat_last])
        inv_s = lats[0]
        inv_n = lats[1]
        closed = True
    return inv_n, inv_s, closed, l_eq


def compute_invariant_latitudes(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    times_unix: ArrayLike,
    config: TracingConfig | None = None,
    field_config: SaturnFieldConfig | None = None,
    region_codes: ArrayLike | None = None,
) -> TracingResult:
    r"""Compute invariant latitude for a sequence of spacecraft positions.

    Traces a KMAG field line from each position (subsampled by
    ``config.trace_every_n``) and extracts the conjugate latitude where
    the field line intersects Saturn's surface.

    Parameters
    ----------
    x, y, z : array_like
        Spacecraft positions in KSM coordinates ($R_S$).
    times_unix : array_like
        Timestamps for each position in the KMAG model's epoch
        (J2000 seconds by default). Caller must convert from POSIX
        if needed: ``t_j2000 = t_posix - 946728000.0``.
    config : TracingConfig, optional
        Tracing parameters. Uses defaults if not provided.
    field_config : SaturnFieldConfig, optional
        KMAG field model parameters ($d_p$, IMF $B_y$, $B_z$).
        Uses defaults if not provided.
    region_codes : array_like of int, optional
        Region code per sample. If given AND ``config.region_filter``
        is not None, only samples with codes in ``region_filter``
        are traced. Skipped slots stay NaN/False in the result.

    Returns
    -------
    TracingResult
        Dataclass with ``inv_lat_north``, ``inv_lat_south``,
        ``is_closed``, ``l_equatorial``, ``n_traces``, ``n_closed``.
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
    l_equatorial = np.full(n_traces, np.nan)

    # Optional region filter: build a mask over the result slots
    if region_codes is not None and config.region_filter is not None:
        codes = np.asarray(region_codes, dtype=int)
        codes_at_indices = codes[indices]
        active_mask = np.isin(codes_at_indices, list(config.region_filter))
        active_slots = np.flatnonzero(active_mask)
    else:
        active_slots = np.arange(n_traces)

    field = SaturnField(field_config)

    log.info(
        "Tracing %d field lines (every %d samples from %d total, "
        "step=%.3f R_S, max_r=%.0f R_S, max_steps=%d) — %d active after region filter",
        n_traces, config.trace_every_n, n_total,
        config.step, config.max_radius, config.max_steps, len(active_slots),
    )
    if field_config:
        log.info(
            "  Field model: dp=%.4f nPa, By=%.2f nT, Bz=%.2f nT",
            field_config.dp, field_config.by_imf, field_config.bz_imf,
        )

    for count, slot in enumerate(active_slots):
        idx = indices[slot]
        inv_n, inv_s, closed, l_eq = _trace_one(
            field, float(x[idx]), float(y[idx]), float(z[idx]), float(times[idx]), config,
        )
        inv_lat_north[slot] = inv_n
        inv_lat_south[slot] = inv_s
        is_closed[slot] = closed
        l_equatorial[slot] = l_eq

        if (count + 1) % config.log_interval == 0:
            pct = (count + 1) / len(active_slots) * 100
            n_cl = int(np.sum(is_closed))
            log.info(
                "Progress: %d/%d (%.1f%%) — %d closed",
                count + 1, len(active_slots), pct, n_cl,
            )

    n_closed = int(np.sum(is_closed))
    log.info("Tracing complete: %d/%d closed field lines", n_closed, n_traces)
    return TracingResult(
        inv_lat_north=inv_lat_north,
        inv_lat_south=inv_lat_south,
        is_closed=is_closed,
        l_equatorial=l_equatorial,
        n_traces=n_traces,
        n_closed=n_closed,
    )


# ----------------------------------------------------------------------
# Parallel implementation
# ----------------------------------------------------------------------

# Module-level field instance for worker processes (set in _worker_init)
_WORKER_FIELD: SaturnField | None = None


def _worker_init(field_config: SaturnFieldConfig | None) -> None:
    """Initialise worker process: build SaturnField and warm up numba JIT."""
    global _WORKER_FIELD
    _WORKER_FIELD = SaturnField(field_config)
    # Force JIT compilation now (~2 s) so the first real trace doesn't pay it
    try:
        _WORKER_FIELD.field_cartesian(10.0, 0.0, 0.0, 0.0, coord="KSM")
    except Exception:
        pass


def _trace_chunk(payload: tuple) -> dict:
    """Worker entry point. Trace a chunk of samples and return results.

    payload = (slots, x_chunk, y_chunk, z_chunk, t_chunk, config)
    Returns dict with keys: slots, inv_lat_north, inv_lat_south, is_closed, l_equatorial.
    """
    slots, x_c, y_c, z_c, t_c, config = payload
    n = len(slots)
    inv_n = np.full(n, np.nan)
    inv_s = np.full(n, np.nan)
    closed = np.zeros(n, dtype=bool)
    l_eq = np.full(n, np.nan)

    field = _WORKER_FIELD
    if field is None:  # safety: shouldn't happen if pool was initialised
        field = SaturnField()

    for i in range(n):
        a, b, c, d = _trace_one(
            field, float(x_c[i]), float(y_c[i]), float(z_c[i]), float(t_c[i]), config,
        )
        inv_n[i] = a
        inv_s[i] = b
        closed[i] = c
        l_eq[i] = d

    return {
        "slots": slots,
        "inv_lat_north": inv_n,
        "inv_lat_south": inv_s,
        "is_closed": closed,
        "l_equatorial": l_eq,
    }


def compute_invariant_latitudes_parallel(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    times_unix: ArrayLike,
    config: TracingConfig | None = None,
    field_config: SaturnFieldConfig | None = None,
    region_codes: ArrayLike | None = None,
    n_workers: int | None = None,
) -> TracingResult:
    r"""Parallel version of :func:`compute_invariant_latitudes`.

    Splits the active samples into round-robin chunks and processes them
    on a multiprocessing ``Pool`` with the ``"spawn"`` start method (safer
    for numba). Each worker creates its own ``SaturnField`` and warms up
    the numba JIT once.

    Returns element-wise identical results to the serial function for
    ``n_workers=1``.

    Parameters
    ----------
    x, y, z, times_unix, config, field_config, region_codes
        Same as :func:`compute_invariant_latitudes`.
    n_workers : int, optional
        Number of worker processes. Overrides ``config.n_workers`` if given.
        Defaults to ``config.n_workers`` (default 1 = serial fallback).
    """
    if config is None:
        config = TracingConfig()
    if n_workers is None:
        n_workers = config.n_workers

    # Fallback to serial path
    if n_workers <= 1:
        return compute_invariant_latitudes(
            x, y, z, times_unix,
            config=config, field_config=field_config, region_codes=region_codes,
        )

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
    l_equatorial = np.full(n_traces, np.nan)

    # Build active slot list (slots into the result arrays)
    if region_codes is not None and config.region_filter is not None:
        codes = np.asarray(region_codes, dtype=int)
        codes_at_indices = codes[indices]
        active_mask = np.isin(codes_at_indices, list(config.region_filter))
        active_slots = np.flatnonzero(active_mask)
    else:
        active_slots = np.arange(n_traces)

    # Also drop slots whose position is out of (surface_tolerance, max_radius).
    # This avoids spawning workers for samples that would be skipped anyway.
    pos_idx = indices[active_slots]
    r_at = np.sqrt(x[pos_idx] ** 2 + y[pos_idx] ** 2 + z[pos_idx] ** 2)
    in_range = (r_at >= config.surface_tolerance) & (r_at <= config.max_radius)
    active_slots = active_slots[in_range]

    n_active = len(active_slots)
    if n_active == 0:
        log.warning("No active traces after region/range filtering")
        return TracingResult(
            inv_lat_north=inv_lat_north,
            inv_lat_south=inv_lat_south,
            is_closed=is_closed,
            l_equatorial=l_equatorial,
            n_traces=n_traces,
            n_closed=0,
        )

    # Round-robin chunking — interleaves fast/slow Cassini regions
    n_chunks = max(1, n_workers * 16)
    if config.chunk_size is not None:
        n_chunks = max(1, n_active // config.chunk_size)
    chunks = [active_slots[i::n_chunks] for i in range(n_chunks)]

    log.info(
        "Parallel tracing: %d active samples, %d workers, %d chunks "
        "(step=%.3f R_S, max_r=%.0f R_S, max_steps=%d)",
        n_active, n_workers, n_chunks,
        config.step, config.max_radius, config.max_steps,
    )
    if field_config:
        log.info(
            "  Field model: dp=%.4f nPa, By=%.2f nT, Bz=%.2f nT",
            field_config.dp, field_config.by_imf, field_config.bz_imf,
        )

    # Build per-chunk payloads up front (small — just slot indices + arrays)
    payloads = []
    for slots in chunks:
        if len(slots) == 0:
            continue
        pi = indices[slots]
        payloads.append((
            slots,
            x[pi].copy(),
            y[pi].copy(),
            z[pi].copy(),
            times[pi].copy(),
            config,
        ))

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(n_workers, initializer=_worker_init, initargs=(field_config,))
    n_done = 0
    try:
        for result in pool.imap_unordered(_trace_chunk, payloads):
            slots = result["slots"]
            inv_lat_north[slots] = result["inv_lat_north"]
            inv_lat_south[slots] = result["inv_lat_south"]
            is_closed[slots] = result["is_closed"]
            l_equatorial[slots] = result["l_equatorial"]
            n_done += len(slots)
            # Log roughly every config.log_interval samples
            if (n_done // config.log_interval) > ((n_done - len(slots)) // config.log_interval):
                pct = n_done / n_active * 100
                n_cl = int(np.sum(is_closed))
                log.info(
                    "Progress: %d/%d (%.1f%%) — %d closed",
                    n_done, n_active, pct, n_cl,
                )
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt — terminating worker pool")
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()

    n_closed = int(np.sum(is_closed))
    log.info("Parallel tracing complete: %d/%d closed field lines", n_closed, n_traces)
    return TracingResult(
        inv_lat_north=inv_lat_north,
        inv_lat_south=inv_lat_south,
        is_closed=is_closed,
        l_equatorial=l_equatorial,
        n_traces=n_traces,
        n_closed=n_closed,
    )
