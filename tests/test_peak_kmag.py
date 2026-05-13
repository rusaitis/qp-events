r"""Tests for :mod:`qp.events.peak_kmag`.

Single-trace enrichment of round-8 wave-event peaks. Trace numerics
are covered in ``test_dwell_grid.py`` and ``test_tracer.py``; here we
exercise only the wrapper:

- closed dayside field line at moderate distance produces a sane
  signed inv-lat and a finite ``l_eq`` greater than the spacecraft
  radius;
- out-of-range positions return NaNs and ``is_closed_peak=False``;
- hemisphere sign convention matches the dwell-grid wrapper
  (``z >= 0`` chooses northern footpoint).
"""

from __future__ import annotations

import math

import pytest

from qp.dwell.tracing import TracingConfig
from qp.events.peak_kmag import J2000_POSIX, kmag_peak_columns
from qp.fieldline.kmag_model import SaturnField

# Mid-mission epoch (2009-01-01 UTC) as J2000 seconds. KMAG is
# weakly time-dependent at low frequency; any year inside the
# Cassini mission works.
_T_J2000_2009 = 1.262304e9 - J2000_POSIX

# Wider boundary so the bidirectional trace can find both footpoints.
_CFG = TracingConfig(step=0.15, max_radius=60.0, max_steps=20_000)


@pytest.fixture(scope="module")
def field() -> SaturnField:
    return SaturnField()


def test_closed_dayside_line(field: SaturnField) -> None:
    out = kmag_peak_columns(8.0, 0.0, 1.0, _T_J2000_2009, field, _CFG)
    assert out["is_closed_peak"] is True
    inv = float(out["kmag_inv_lat_peak"])
    l_eq = float(out["l_eq_peak"])
    assert math.isfinite(inv) and 30.0 <= abs(inv) <= 89.0
    # Apex >= spacecraft radial distance (closed line passes through us).
    assert l_eq >= math.hypot(8.0, 1.0) - 0.5


def test_out_of_range_returns_nans(field: SaturnField) -> None:
    # Inside surface tolerance — should never be traced.
    out = kmag_peak_columns(0.5, 0.0, 0.0, _T_J2000_2009, field, _CFG)
    assert out["is_closed_peak"] is False
    assert math.isnan(float(out["kmag_inv_lat_peak"]))
    assert math.isnan(float(out["l_eq_peak"]))


def test_hemisphere_sign(field: SaturnField) -> None:
    # Closed line just north of the equator → positive inv-lat.
    north = kmag_peak_columns(8.0, 0.0, 1.0, _T_J2000_2009, field, _CFG)
    south = kmag_peak_columns(8.0, 0.0, -1.0, _T_J2000_2009, field, _CFG)
    if north["is_closed_peak"] and south["is_closed_peak"]:
        assert float(north["kmag_inv_lat_peak"]) > 0
        assert float(south["kmag_inv_lat_peak"]) < 0
