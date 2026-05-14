"""Tests for qp.events.normalization — dwell-floor masking & ratio clipping.

The 600-min dwell floor (one PPO recurrence period) gates every cell of
Figs 7, 8, SI 1, SI 2. A regression on `>=` vs `>` or on the constant value
would silently corrupt the published occurrence maps.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.events.normalization import (
    MIN_DWELL_MINUTES_PER_CELL,
    OccurrenceConfig,
    collapse_to_latitude,
    occurrence_rate,
    slice_lt_sector,
    weighted_occurrence_rate,
)


class TestMinDwellInvariant:
    def test_constant_value_is_paper_published(self):
        """600 min ≈ one PPO recurrence period (~10.7 h). This value is
        documented in CLAUDE.md and is load-bearing across Figs 7, 8.
        """
        assert MIN_DWELL_MINUTES_PER_CELL == 600.0


class TestOccurrenceRateFloor:
    def test_canonical_floor_boundary(self):
        r"""At the published 600-min floor, cells at and above 600 must be
        finite; cells strictly below must be NaN.
        """
        event = np.array([60.0, 60.0, 60.0, 60.0])
        dwell = np.array([599.0, 600.0, 601.0, 1200.0])
        rate = occurrence_rate(
            event, dwell, min_dwell_minutes=MIN_DWELL_MINUTES_PER_CELL
        )
        assert np.isnan(rate[0])
        assert np.isfinite(rate[1])
        assert np.isfinite(rate[2])
        assert np.isfinite(rate[3])
        assert_allclose(rate[1], 60.0 / 600.0)
        assert_allclose(rate[2], 60.0 / 601.0)

    def test_below_floor_is_nan(self):
        event = np.array([10.0, 20.0, 30.0])
        dwell = np.array([10.0, 30.0, 59.0])  # all below default 60-min floor
        rate = occurrence_rate(event, dwell, min_dwell_minutes=60.0)
        assert np.all(np.isnan(rate))

    def test_clip_max(self):
        """Defensive guard: even if event > dwell (shouldn't happen for
        the consistency grid), the ratio caps at clip_max."""
        event = np.array([1500.0])
        dwell = np.array([1000.0])
        rate = occurrence_rate(event, dwell, min_dwell_minutes=60.0, clip_max=1.0)
        assert_allclose(rate, 1.0)

    def test_clip_max_none_disables_clipping(self):
        event = np.array([1500.0])
        dwell = np.array([1000.0])
        rate = occurrence_rate(event, dwell, min_dwell_minutes=60.0, clip_max=None)
        assert_allclose(rate, 1.5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            occurrence_rate(np.zeros((2, 3)), np.zeros((3, 2)))

    def test_zero_event_finite_dwell_yields_zero(self):
        event = np.zeros(3)
        dwell = np.array([100.0, 1000.0, 10000.0])
        rate = occurrence_rate(event, dwell, min_dwell_minutes=60.0)
        assert_allclose(rate, 0.0)


class TestWeightedOccurrenceRate:
    def test_weighted_bounded_by_unweighted(self):
        r"""Since $q \in [0, 1]$, $\sum q_i \Delta t_i \leq \sum \Delta t_i$ per cell,
        so weighted ratio ≤ unweighted ratio.
        """
        rng = np.random.default_rng(11)
        unweighted = rng.uniform(50.0, 500.0, 20)
        weights = rng.uniform(0.0, 1.0, 20)
        weighted = weights * unweighted
        dwell = rng.uniform(700.0, 5000.0, 20)
        rate_u = occurrence_rate(unweighted, dwell, min_dwell_minutes=600.0)
        rate_w = weighted_occurrence_rate(weighted, dwell, min_dwell_minutes=600.0)
        finite = np.isfinite(rate_u) & np.isfinite(rate_w)
        assert np.all(rate_w[finite] <= rate_u[finite] + 1e-12)


class TestSliceLtSector:
    def test_no_wraparound_sector(self):
        grid = np.ones((2, 3, 24))  # uniform unit values, 24 LT bins of 1 h
        lt_centers = np.arange(24) + 0.5  # centers at 0.5, 1.5, ..., 23.5
        out = slice_lt_sector(grid, lt_centers, center_h=12.0, half_width_h=3.0)
        # 6 LT bins (9-15) selected, each contributes 1
        assert out.shape == (2, 3)
        assert_allclose(out, 6.0)

    def test_wraparound_midnight(self):
        grid = np.ones((2, 3, 24))
        lt_centers = np.arange(24) + 0.5
        out = slice_lt_sector(grid, lt_centers, center_h=0.0, half_width_h=3.0)
        # bins at 21, 22, 23, 0, 1, 2 → 6 bins
        assert_allclose(out, 6.0)


class TestCollapseToLatitude:
    def test_sums_along_radius(self):
        grid = np.arange(12.0).reshape(3, 4)  # 3 r bins, 4 lat bins
        out = collapse_to_latitude(grid)
        assert out.shape == (4,)
        # column sums
        assert_allclose(out, grid.sum(axis=0))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="expected 2D"):
            collapse_to_latitude(np.zeros((2, 3, 4)))


class TestOccurrenceConfigDefaults:
    def test_dataclass_is_frozen(self):
        cfg = OccurrenceConfig()
        with pytest.raises((AttributeError, Exception)):
            cfg.min_dwell_minutes = 999.0  # type: ignore[misc]
        assert cfg.min_dwell_minutes == 60.0
        assert cfg.clip_max == 1.0
