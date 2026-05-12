r"""KMAG single-trace enrichment for wave-event peaks.

A round-8 detection carries spacecraft ephemeris (``r_distance``,
``mag_lat``, ``local_time``) at the wave-packet peak time, but the
natural axes for field-line-resonance physics are KMAG-traced:

- ``kmag_inv_lat_peak`` — signed invariant latitude (degrees), the
  footpoint latitude of the field line threading the spacecraft's
  position, chosen in the spacecraft's hemisphere.
- ``l_eq_peak`` — equatorial-crossing distance ($R_S$), i.e. the
  apex of the traced field line. Aliased ``L``-shell for closed
  lines.
- ``is_closed_peak`` — whether both footpoints reach Saturn's
  surface inside the trace's outer boundary.

This module wraps :func:`qp.dwell.tracing._trace_one` with the
sign-by-hemisphere convention used by
:func:`qp.dwell.grid.accumulate_traced_inv_lat_grid` and returns a
dict suitable for the round-8 parquet ``extra`` payload. The trace
itself is ~0.3 ms; per-event use adds ~1 s to a 100 s full-mission
sweep, and ~30 s to a one-shot backfill over 2000 events.
"""

from __future__ import annotations

import math

from qp.constants import J2000_POSIX
from qp.dwell.tracing import TracingConfig, _trace_one
from qp.fieldline.kmag_model import SaturnField

__all__ = ["J2000_POSIX", "kmag_peak_columns"]


def kmag_peak_columns(
    x_ksm: float,
    y_ksm: float,
    z_ksm: float,
    t_j2000: float,
    field: SaturnField,
    config: TracingConfig | None = None,
) -> dict[str, float | bool]:
    r"""Trace one field line from a peak position; return enrichment columns.

    Parameters
    ----------
    x_ksm, y_ksm, z_ksm
        Spacecraft position in KSM coordinates ($R_S$).
    t_j2000
        Time at the peak, in J2000 seconds (POSIX minus
        :data:`J2000_POSIX`).
    field
        Pre-built :class:`SaturnField`. Callers building many peaks
        should construct this once outside the loop.
    config
        Tracing parameters. Defaults to :class:`TracingConfig` with
        ``step=0.15``, ``max_radius=60`` — same as the canonical dwell
        grid.

    Returns
    -------
    dict
        Three keys: ``kmag_inv_lat_peak`` (signed degrees;
        ``float('nan')`` if open or trace failed), ``l_eq_peak``
        ($R_S$; ``float('nan')`` if the trace was out of range), and
        ``is_closed_peak`` (bool).
    """
    if config is None:
        config = TracingConfig()
    inv_n, inv_s, closed, l_eq = _trace_one(
        field,
        float(x_ksm),
        float(y_ksm),
        float(z_ksm),
        float(t_j2000),
        config,
    )
    if not closed:
        return {
            "kmag_inv_lat_peak": float("nan"),
            "l_eq_peak": float(l_eq) if math.isfinite(l_eq) else float("nan"),
            "is_closed_peak": False,
        }
    # Sign the invariant latitude by the spacecraft's hemisphere — same
    # convention used by accumulate_traced_inv_lat_grid (and the canonical
    # KMAG dwell grid). Saturn's offset dipole is at z = +0.037 R_S, but
    # the boundary at z = 0 is a negligible asymmetry at typical Cassini
    # distances and keeps the convention consistent.
    inv_signed = inv_n if z_ksm >= 0.0 else inv_s
    return {
        "kmag_inv_lat_peak": float(inv_signed),
        "l_eq_peak": float(l_eq),
        "is_closed_peak": True,
    }
