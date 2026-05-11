"""Tests for `qp.events.cooccurrence.tag_co_bands`."""

from __future__ import annotations

import pandas as pd

from qp.events.cooccurrence import tag_co_bands


def _row(
    *,
    event_id: int,
    segment_idx: int,
    band: str,
    date_from: str,
    date_to: str,
    is_duplicate: bool | None = None,
) -> dict:
    r = {
        "event_id": event_id,
        "segment_idx": segment_idx,
        "band": band,
        "date_from": date_from,
        "date_to": date_to,
    }
    if is_duplicate is not None:
        r["is_duplicate"] = is_duplicate
    return r


def test_empty_dataframe() -> None:
    df = pd.DataFrame(
        columns=["event_id", "segment_idx", "band", "date_from", "date_to"]
    )
    out = tag_co_bands(df)
    assert out.shape == (0, 6)
    assert "co_bands" in out.columns


def test_two_overlapping_bands_tag_each_other() -> None:
    df = pd.DataFrame([
        _row(event_id=1, segment_idx=42, band="QP60",
             date_from="2010-01-01 00:00:00", date_to="2010-01-01 04:00:00"),
        _row(event_id=2, segment_idx=42, band="QP120",
             date_from="2010-01-01 01:00:00", date_to="2010-01-01 03:00:00"),
    ])
    out = tag_co_bands(df)
    assert out.loc[0, "co_bands"] == "QP120"
    assert out.loc[1, "co_bands"] == "QP60"


def test_three_bands_all_overlap() -> None:
    df = pd.DataFrame([
        _row(event_id=1, segment_idx=7, band="QP30",
             date_from="2010-01-01 00:00:00", date_to="2010-01-01 04:00:00"),
        _row(event_id=2, segment_idx=7, band="QP60",
             date_from="2010-01-01 01:00:00", date_to="2010-01-01 03:00:00"),
        _row(event_id=3, segment_idx=7, band="QP120",
             date_from="2010-01-01 00:30:00", date_to="2010-01-01 02:30:00"),
    ])
    out = tag_co_bands(df)
    assert out.loc[0, "co_bands"] == "QP120,QP60"
    assert out.loc[1, "co_bands"] == "QP120,QP30"
    assert out.loc[2, "co_bands"] == "QP30,QP60"


def test_non_overlapping_windows_have_no_siblings() -> None:
    df = pd.DataFrame([
        _row(event_id=1, segment_idx=7, band="QP60",
             date_from="2010-01-01 00:00:00", date_to="2010-01-01 02:00:00"),
        _row(event_id=2, segment_idx=7, band="QP120",
             date_from="2010-01-01 04:00:00", date_to="2010-01-01 06:00:00"),
    ])
    out = tag_co_bands(df)
    assert out.loc[0, "co_bands"] == ""
    assert out.loc[1, "co_bands"] == ""


def test_same_band_overlap_does_not_self_tag() -> None:
    df = pd.DataFrame([
        _row(event_id=1, segment_idx=7, band="QP60",
             date_from="2010-01-01 00:00:00", date_to="2010-01-01 02:00:00"),
        _row(event_id=2, segment_idx=7, band="QP60",
             date_from="2010-01-01 00:30:00", date_to="2010-01-01 01:30:00"),
    ])
    out = tag_co_bands(df)
    assert out.loc[0, "co_bands"] == ""
    assert out.loc[1, "co_bands"] == ""


def test_different_segments_never_co_occur() -> None:
    df = pd.DataFrame([
        _row(event_id=1, segment_idx=7, band="QP60",
             date_from="2010-01-01 00:00:00", date_to="2010-01-01 04:00:00"),
        _row(event_id=2, segment_idx=8, band="QP120",
             date_from="2010-01-01 01:00:00", date_to="2010-01-01 03:00:00"),
    ])
    out = tag_co_bands(df)
    assert out.loc[0, "co_bands"] == ""
    assert out.loc[1, "co_bands"] == ""


def test_duplicates_are_skipped_as_siblings_and_get_no_tags() -> None:
    df = pd.DataFrame([
        _row(event_id=1, segment_idx=7, band="QP60", is_duplicate=False,
             date_from="2010-01-01 00:00:00", date_to="2010-01-01 04:00:00"),
        _row(event_id=2, segment_idx=7, band="QP120", is_duplicate=True,
             date_from="2010-01-01 01:00:00", date_to="2010-01-01 03:00:00"),
        _row(event_id=3, segment_idx=7, band="QP30", is_duplicate=False,
             date_from="2010-01-01 00:30:00", date_to="2010-01-01 02:30:00"),
    ])
    out = tag_co_bands(df)
    # row 0 (QP60, kept) sees row 2 (QP30, kept) but NOT row 1 (dup QP120).
    assert out.loc[0, "co_bands"] == "QP30"
    # row 1 is a duplicate — no co_bands assigned to duplicates.
    assert out.loc[1, "co_bands"] == ""
    # row 2 (QP30, kept) sees row 0 (QP60, kept) but NOT row 1 (dup QP120).
    assert out.loc[2, "co_bands"] == "QP60"


def test_overlap_is_half_open() -> None:
    # date_to == other.date_from → no overlap (the windows touch but don't share time)
    df = pd.DataFrame([
        _row(event_id=1, segment_idx=7, band="QP60",
             date_from="2010-01-01 00:00:00", date_to="2010-01-01 02:00:00"),
        _row(event_id=2, segment_idx=7, band="QP120",
             date_from="2010-01-01 02:00:00", date_to="2010-01-01 04:00:00"),
    ])
    out = tag_co_bands(df)
    assert out.loc[0, "co_bands"] == ""
    assert out.loc[1, "co_bands"] == ""


def test_input_not_mutated() -> None:
    df = pd.DataFrame([
        _row(event_id=1, segment_idx=7, band="QP60",
             date_from="2010-01-01 00:00:00", date_to="2010-01-01 02:00:00"),
    ])
    snapshot = df.copy()
    _ = tag_co_bands(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_row_order_preserved() -> None:
    df = pd.DataFrame([
        _row(event_id=99, segment_idx=42, band="QP120",
             date_from="2010-01-01 00:00:00", date_to="2010-01-01 04:00:00"),
        _row(event_id=11, segment_idx=42, band="QP30",
             date_from="2010-01-01 01:00:00", date_to="2010-01-01 03:00:00"),
    ])
    out = tag_co_bands(df)
    assert out["event_id"].tolist() == [99, 11]
