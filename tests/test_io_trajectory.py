"""Smoke tests for qp.io.trajectory — region tagging (vectorized lookup).

We don't load any real PDS data here; we exercise the pure-Python
``lookup_region_codes`` against a hand-built crossings array. The two
PDS-dependent functions (``load_year_positions``, ``load_mission_trajectory``)
are imported only as a signature-regression check.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal

from qp.io.trajectory import (
    UNKNOWN_REGION_CODE,
    load_mission_trajectory,
    load_year_positions,
    lookup_region_codes,
)


def test_public_api_present():
    """Regression guard: the public API must keep these symbols."""
    assert callable(load_year_positions)
    assert callable(load_mission_trajectory)
    assert callable(lookup_region_codes)
    assert isinstance(UNKNOWN_REGION_CODE, int)


class TestLookupRegionCodes:
    def test_samples_before_first_crossing_get_unknown(self):
        # crossings at t=10, 20, 30 with region codes 1, 2, 3 after each
        cross_t = np.array([10.0, 20.0, 30.0])
        cross_codes = np.array([1, 2, 3])
        samples = np.array([0.0, 5.0, 9.999])
        codes = lookup_region_codes(samples, cross_t, cross_codes)
        assert_array_equal(codes, np.full(3, UNKNOWN_REGION_CODE))

    def test_samples_between_crossings_get_previous_region(self):
        cross_t = np.array([10.0, 20.0, 30.0])
        cross_codes = np.array([1, 2, 3])
        # np.searchsorted defaults to side='left': a sample EQUAL to a
        # crossing time gets the previous region, since the new region
        # only takes effect strictly after the crossing.
        samples = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
        codes = lookup_region_codes(samples, cross_t, cross_codes)
        assert_array_equal(
            codes,
            np.array([UNKNOWN_REGION_CODE, 1, 1, 2, 2, 3]),
        )

    def test_empty_samples_returns_empty(self):
        cross_t = np.array([10.0, 20.0])
        cross_codes = np.array([1, 2])
        codes = lookup_region_codes(np.array([]), cross_t, cross_codes)
        assert codes.size == 0

    def test_monotonic_assumption(self):
        """``np.searchsorted`` assumes ``crossing_times_unix`` is sorted
        ascending. Validate the contract with a tiny case."""
        cross_t = np.array([0.0, 100.0, 200.0])
        cross_codes = np.array([0, 1, 2])
        # large sample → falls in last region
        codes = lookup_region_codes(np.array([1e6]), cross_t, cross_codes)
        assert_array_equal(codes, np.array([2]))
