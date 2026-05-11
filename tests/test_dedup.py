"""Tests for `qp.events.dedup`.

Verify the post-hoc deduplication logic catches both the cross-segment
and intra-segment leakage modes documented in the module docstring.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qp.events.dedup import (
    DEFAULT_DT_SEC,
    collapse_duplicates,
    tag_duplicates,
)


def _make_row(
    *,
    event_id: int,
    segment_idx: int,
    band: str,
    peak_time: str,
    period_min: float,
    q_factor: float,
) -> dict:
    return {
        "event_id": event_id,
        "segment_idx": segment_idx,
        "band": band,
        "peak_time": peak_time,
        "period_min": period_min,
        "q_factor": q_factor,
    }


def test_empty_dataframe_round_trips() -> None:
    df = pd.DataFrame(
        columns=["event_id", "segment_idx", "band", "peak_time", "period_min", "q_factor"]
    )
    tagged = tag_duplicates(df)
    assert tagged.shape == (0, 7)
    assert "is_duplicate" in tagged.columns
    assert collapse_duplicates(df).shape == (0, 6)


def test_cross_segment_dup_keeps_highest_q() -> None:
    """`0903260044A`/`B`-style: same band, same minute, neighbouring segments."""
    df = pd.DataFrame([
        _make_row(event_id=606, segment_idx=1678, band="QP60",
                  peak_time="2009-03-26 00:44:00", period_min=60.2, q_factor=5.46),
        _make_row(event_id=607, segment_idx=1679, band="QP60",
                  peak_time="2009-03-26 00:44:00", period_min=60.2, q_factor=5.47),
    ])
    tagged = tag_duplicates(df)
    assert tagged.loc[0, "is_duplicate"]
    assert not tagged.loc[1, "is_duplicate"]


def test_intra_segment_dup_through_interleaved_band() -> None:
    """Two QP60s within 2 h with a QP30 between them, all same segment.

    This is the failure mode in detector.py:722-730 where the in-loop dedup
    only compares against `merged[-1]`. The post-hoc pass must catch it.
    """
    df = pd.DataFrame([
        _make_row(event_id=1, segment_idx=42, band="QP60",
                  peak_time="2010-01-01 00:00:00", period_min=60.0, q_factor=8.0),
        _make_row(event_id=2, segment_idx=42, band="QP30",
                  peak_time="2010-01-01 00:30:00", period_min=30.0, q_factor=4.0),
        _make_row(event_id=3, segment_idx=42, band="QP60",
                  peak_time="2010-01-01 01:00:00", period_min=60.0, q_factor=5.0),
    ])
    tagged = tag_duplicates(df)
    assert not tagged.loc[0, "is_duplicate"]  # higher-q QP60 kept
    assert not tagged.loc[1, "is_duplicate"]  # different band, never a dup
    assert tagged.loc[2, "is_duplicate"]


def test_separation_outside_window_is_not_dup() -> None:
    df = pd.DataFrame([
        _make_row(event_id=1, segment_idx=1, band="QP60",
                  peak_time="2010-01-01 00:00:00", period_min=60.0, q_factor=5.0),
        _make_row(event_id=2, segment_idx=1, band="QP60",
                  peak_time="2010-01-01 02:01:00", period_min=60.0, q_factor=5.0),
    ])
    tagged = tag_duplicates(df, dt_sec=DEFAULT_DT_SEC)  # 2h
    assert not tagged["is_duplicate"].any()


def test_different_period_outside_tol_is_not_dup() -> None:
    df = pd.DataFrame([
        _make_row(event_id=1, segment_idx=1, band="QP60",
                  peak_time="2010-01-01 00:00:00", period_min=50.0, q_factor=5.0),
        _make_row(event_id=2, segment_idx=1, band="QP60",
                  peak_time="2010-01-01 00:30:00", period_min=75.0, q_factor=5.0),
    ])
    tagged = tag_duplicates(df, period_rel_tol=0.10)
    assert not tagged["is_duplicate"].any()


def test_different_bands_never_collapse() -> None:
    df = pd.DataFrame([
        _make_row(event_id=1, segment_idx=1, band="QP30",
                  peak_time="2010-01-01 00:00:00", period_min=30.0, q_factor=5.0),
        _make_row(event_id=2, segment_idx=1, band="QP60",
                  peak_time="2010-01-01 00:00:00", period_min=60.0, q_factor=5.0),
        _make_row(event_id=3, segment_idx=1, band="QP120",
                  peak_time="2010-01-01 00:00:00", period_min=120.0, q_factor=5.0),
    ])
    tagged = tag_duplicates(df)
    assert not tagged["is_duplicate"].any()


def test_chain_collapses_to_single_keeper() -> None:
    """Three same-band events daisy-chained inside a 2-h window: keep best q.

    Each consecutive pair is within window, but A and C are >2 h apart.
    The greedy band-rep update is intentional — Cassini sees the same
    wave train across overlapping segments, so daisy-chained matches
    are the same physical packet.
    """
    df = pd.DataFrame([
        _make_row(event_id=1, segment_idx=1, band="QP60",
                  peak_time="2010-01-01 00:00:00", period_min=60.0, q_factor=4.0),
        _make_row(event_id=2, segment_idx=2, band="QP60",
                  peak_time="2010-01-01 01:00:00", period_min=60.0, q_factor=9.0),
        _make_row(event_id=3, segment_idx=3, band="QP60",
                  peak_time="2010-01-01 02:00:00", period_min=60.0, q_factor=6.0),
    ])
    tagged = tag_duplicates(df)
    # The middle row has the highest q_factor and ends up as the survivor.
    assert tagged.loc[tagged["q_factor"] == 9.0, "is_duplicate"].iloc[0] is False or \
        not tagged.loc[tagged["q_factor"] == 9.0, "is_duplicate"].iloc[0]
    assert tagged["is_duplicate"].sum() == 2


def test_collapse_drops_marked_rows() -> None:
    df = pd.DataFrame([
        _make_row(event_id=1, segment_idx=1, band="QP60",
                  peak_time="2010-01-01 00:00:00", period_min=60.0, q_factor=5.0),
        _make_row(event_id=2, segment_idx=2, band="QP60",
                  peak_time="2010-01-01 00:30:00", period_min=60.0, q_factor=8.0),
    ])
    out = collapse_duplicates(df)
    assert len(out) == 1
    assert out.iloc[0]["q_factor"] == 8.0
    assert "is_duplicate" not in out.columns


def test_row_order_is_preserved() -> None:
    df = pd.DataFrame([
        _make_row(event_id=99, segment_idx=99, band="QP120",
                  peak_time="2012-06-15 10:00:00", period_min=120.0, q_factor=4.0),
        _make_row(event_id=11, segment_idx=11, band="QP30",
                  peak_time="2008-01-01 00:00:00", period_min=30.0, q_factor=4.0),
        _make_row(event_id=42, segment_idx=42, band="QP60",
                  peak_time="2010-05-05 05:00:00", period_min=60.0, q_factor=4.0),
    ])
    tagged = tag_duplicates(df)
    assert tagged["event_id"].tolist() == [99, 11, 42]


def test_input_not_mutated() -> None:
    df = pd.DataFrame([
        _make_row(event_id=1, segment_idx=1, band="QP60",
                  peak_time="2010-01-01 00:00:00", period_min=60.0, q_factor=5.0),
    ])
    snapshot = df.copy()
    _ = tag_duplicates(df)
    pd.testing.assert_frame_equal(df, snapshot)


@pytest.mark.parametrize("dt_sec,expected_dup", [(7200, True), (60, False)])
def test_dt_threshold_parameterized(dt_sec: float, expected_dup: bool) -> None:
    df = pd.DataFrame([
        _make_row(event_id=1, segment_idx=1, band="QP60",
                  peak_time="2010-01-01 00:00:00", period_min=60.0, q_factor=5.0),
        _make_row(event_id=2, segment_idx=1, band="QP60",
                  peak_time="2010-01-01 00:30:00", period_min=60.0, q_factor=5.0),
    ])
    tagged = tag_duplicates(df, dt_sec=dt_sec)
    assert tagged["is_duplicate"].any() == expected_dup
