r"""Tests for :mod:`qp.events.footprints`.

Verifies the CSR layout, zarr round-trip, and filter semantics. The
heavy I/O (loading the full mission trajectory + tracing) is covered
elsewhere; here we use small synthetic per-event bin lists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qp.dwell.grid import DwellGridConfig
from qp.events.footprints import (
    EventFootprints,
    GRID_NAMES,
    SCHEMA_VERSION,
    apply_filter,
    build_sparse_grid,
    grid_shape,
    read_zarr,
    write_zarr,
)


def _synthetic_fp(n_events: int = 4) -> EventFootprints:
    """Build a tiny EventFootprints with all three grids populated."""
    config = DwellGridConfig()
    rng = np.random.default_rng(1)
    grids = {}
    for name in GRID_NAMES:
        shape = grid_shape(name, config)
        n_cells = int(np.prod(shape))
        bins_per_event = [
            rng.integers(0, n_cells, size=int(rng.integers(1, 6))).astype(np.int32)
            for _ in range(n_events)
        ]
        weights_per_event = [np.ones(b.size, dtype=np.float32) for b in bins_per_event]
        grids[name] = build_sparse_grid(bins_per_event, weights_per_event, shape)
    return EventFootprints(
        event_ids=np.arange(n_events, dtype=np.int64),
        grids=grids,
        config=config,
    )


def test_csr_offsets_consistent() -> None:
    fp = _synthetic_fp(5)
    for name, g in fp.grids.items():
        assert g.offsets.size == fp.event_ids.size + 1, name
        assert g.offsets[0] == 0
        assert int(g.offsets[-1]) == g.bin_indices.size
        # Offsets are monotonic non-decreasing.
        assert np.all(np.diff(g.offsets) >= 0)


def test_total_unfiltered_equals_full_sum() -> None:
    fp = _synthetic_fp(6)
    for name, g in fp.grids.items():
        t = fp.total(name)
        assert t.sum() == pytest.approx(g.weights_min.sum())


def test_all_pass_mask_matches_unfiltered() -> None:
    fp = _synthetic_fp(7)
    keep = np.ones(fp.event_ids.size, dtype=bool)
    for name in fp.grids:
        np.testing.assert_array_equal(fp.total(name), fp.total(name, keep))


def test_empty_mask_returns_zero() -> None:
    fp = _synthetic_fp(3)
    drop = np.zeros(fp.event_ids.size, dtype=bool)
    for name in fp.grids:
        out = fp.total(name, drop)
        assert out.sum() == 0.0
        assert out.shape == fp.grids[name].shape


def test_single_event_mask_picks_single_contribution() -> None:
    fp = _synthetic_fp(4)
    for i in range(fp.event_ids.size):
        mask = np.zeros_like(fp.event_ids, dtype=bool)
        mask[i] = True
        for name, g in fp.grids.items():
            start, stop = int(g.offsets[i]), int(g.offsets[i + 1])
            expected = float(g.weights_min[start:stop].sum())
            assert fp.total(name, mask).sum() == pytest.approx(expected)


def test_mask_length_mismatch_raises() -> None:
    fp = _synthetic_fp(3)
    with pytest.raises(ValueError, match="mask length"):
        fp.total("g3d", np.array([True, False], dtype=bool))


def test_apply_filter_empty_is_all_true() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert apply_filter(df, None).tolist() == [True, True, True]
    assert apply_filter(df, "").tolist() == [True, True, True]


def test_apply_filter_returns_bool_mask() -> None:
    df = pd.DataFrame({"q_factor": [1.0, 5.0, 3.0], "band": ["QP30", "QP60", "QP60"]})
    mask = apply_filter(df, "q_factor > 2 and band == 'QP60'")
    assert mask.tolist() == [False, True, True]


def test_apply_filter_non_bool_raises() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="must evaluate to bool"):
        apply_filter(df, "x + 1")


def test_zarr_roundtrip(tmp_path) -> None:
    fp = _synthetic_fp(5)
    path = tmp_path / "fp.zarr"
    write_zarr(fp, str(path), extra_attrs={"note": "test"})

    fp2 = read_zarr(str(path))
    assert fp2.schema_version == SCHEMA_VERSION
    np.testing.assert_array_equal(fp.event_ids, fp2.event_ids)
    assert set(fp.grids) == set(fp2.grids)
    for name in fp.grids:
        g1, g2 = fp.grids[name], fp2.grids[name]
        np.testing.assert_array_equal(g1.offsets, g2.offsets)
        np.testing.assert_array_equal(g1.bin_indices, g2.bin_indices)
        np.testing.assert_array_equal(g1.weights_min, g2.weights_min)
        assert g1.shape == g2.shape
        np.testing.assert_array_equal(fp.total(name), fp2.total(name))


def test_grid_shape_matches_dwell_axes() -> None:
    config = DwellGridConfig()
    assert grid_shape("g3d", config) == (config.n_r, config.n_lat, config.n_lt)
    assert grid_shape("g_kmag_inv_lat", config) == (config.n_lat, config.n_lt)
    assert grid_shape("g_l_eq", config) == (config.n_r, config.n_lt)
    with pytest.raises(ValueError, match="unknown grid"):
        grid_shape("bogus", config)
