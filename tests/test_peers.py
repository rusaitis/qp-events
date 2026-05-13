"""Tests for ``qp.events.peers.tag_peers`` and ``derive_co_bands``."""

from __future__ import annotations

import pandas as pd
import pytest

from qp.events.peers import (
    DEFAULT_MIN_OVERLAP_FRAC,
    derive_co_bands,
    tag_peers,
)


def _row(
    *,
    event_id: int,
    segment_idx: int,
    period_min: float,
    date_from: str,
    date_to: str,
    is_duplicate: bool | None = None,
    band: str | None = None,
) -> dict:
    r: dict = {
        "event_id": event_id,
        "segment_idx": segment_idx,
        "period_min": period_min,
        "date_from": date_from,
        "date_to": date_to,
    }
    if is_duplicate is not None:
        r["is_duplicate"] = is_duplicate
    if band is not None:
        r["band"] = band
    return r


# ---------------------------------------------------------------------
# tag_peers
# ---------------------------------------------------------------------


def test_default_threshold_is_half() -> None:
    assert DEFAULT_MIN_OVERLAP_FRAC == 0.5


def test_empty_dataframe() -> None:
    df = pd.DataFrame(
        columns=[
            "event_id",
            "segment_idx",
            "period_min",
            "date_from",
            "date_to",
        ]
    )
    out = tag_peers(df)
    assert {"peer_event_ids", "peer_periods_min", "peer_overlap_frac"} <= set(
        out.columns
    )
    assert len(out) == 0


def test_two_overlapping_rows_become_peers() -> None:
    # A: 0–4 h (240 min), B: 1–3 h (120 min) → overlap 120 min,
    # shorter 120 min → frac = 1.0 (passes default 0.5).
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=42,
                period_min=60.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 04:00:00",
            ),
            _row(
                event_id=2,
                segment_idx=42,
                period_min=120.0,
                date_from="2010-01-01 01:00:00",
                date_to="2010-01-01 03:00:00",
            ),
        ]
    )
    out = tag_peers(df)
    assert out.loc[0, "peer_event_ids"] == [2]
    assert out.loc[0, "peer_periods_min"] == [120.0]
    assert out.loc[0, "peer_overlap_frac"] == [pytest.approx(1.0)]
    assert out.loc[1, "peer_event_ids"] == [1]
    assert out.loc[1, "peer_periods_min"] == [60.0]
    assert out.loc[1, "peer_overlap_frac"] == [pytest.approx(1.0)]


def test_same_band_peers_recorded() -> None:
    # Two QP60 peaks in the same segment with overlap > 0.5 of the
    # shorter window. Legacy `co_bands` would have left both empty —
    # `tag_peers` records them.
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=55.0,
                band="QP60",
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 02:00:00",
            ),
            _row(
                event_id=2,
                segment_idx=7,
                period_min=75.0,
                band="QP60",
                date_from="2010-01-01 00:30:00",
                date_to="2010-01-01 01:30:00",
            ),
        ]
    )
    out = tag_peers(df)
    assert out.loc[0, "peer_event_ids"] == [2]
    assert out.loc[1, "peer_event_ids"] == [1]


def test_threshold_below_default_rejects() -> None:
    # Overlap = 29.4 min, shorter window = 60 min → frac = 0.49 < 0.5.
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=30.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 01:00:00",
            ),
            _row(
                event_id=2,
                segment_idx=7,
                period_min=60.0,
                date_from="2010-01-01 00:30:36",
                date_to="2010-01-01 02:00:00",
            ),
        ]
    )
    out = tag_peers(df)
    assert out.loc[0, "peer_event_ids"] == []
    assert out.loc[1, "peer_event_ids"] == []


def test_threshold_above_default_accepts() -> None:
    # Overlap = 30.6 min, shorter window = 60 min → frac = 0.51 ≥ 0.5.
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=30.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 01:00:00",
            ),
            _row(
                event_id=2,
                segment_idx=7,
                period_min=60.0,
                date_from="2010-01-01 00:29:24",
                date_to="2010-01-01 02:00:00",
            ),
        ]
    )
    out = tag_peers(df)
    assert out.loc[0, "peer_event_ids"] == [2]
    assert out.loc[1, "peer_event_ids"] == [1]
    assert out.loc[0, "peer_overlap_frac"][0] == pytest.approx(0.51, abs=0.01)


def test_zero_threshold_recovers_any_overlap() -> None:
    # Sliver overlap of 1 min vs 60 + 120 min windows — accepted at τ=0,
    # rejected at the default.
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=30.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 01:00:00",
            ),
            _row(
                event_id=2,
                segment_idx=7,
                period_min=60.0,
                date_from="2010-01-01 00:59:00",
                date_to="2010-01-01 02:59:00",
            ),
        ]
    )
    out_default = tag_peers(df)
    out_zero = tag_peers(df, min_overlap_frac=0.0)
    assert out_default.loc[0, "peer_event_ids"] == []
    assert out_zero.loc[0, "peer_event_ids"] == [2]


def test_half_open_touching_windows_have_no_overlap() -> None:
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=30.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 02:00:00",
            ),
            _row(
                event_id=2,
                segment_idx=7,
                period_min=60.0,
                date_from="2010-01-01 02:00:00",
                date_to="2010-01-01 04:00:00",
            ),
        ]
    )
    out = tag_peers(df, min_overlap_frac=0.0)
    assert out.loc[0, "peer_event_ids"] == []
    assert out.loc[1, "peer_event_ids"] == []


def test_different_segments_never_peer() -> None:
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=30.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 04:00:00",
            ),
            _row(
                event_id=2,
                segment_idx=8,
                period_min=60.0,
                date_from="2010-01-01 01:00:00",
                date_to="2010-01-01 03:00:00",
            ),
        ]
    )
    out = tag_peers(df)
    assert out.loc[0, "peer_event_ids"] == []
    assert out.loc[1, "peer_event_ids"] == []


def test_duplicates_skipped_both_ways() -> None:
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=60.0,
                is_duplicate=False,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 04:00:00",
            ),
            _row(
                event_id=2,
                segment_idx=7,
                period_min=120.0,
                is_duplicate=True,
                date_from="2010-01-01 01:00:00",
                date_to="2010-01-01 03:00:00",
            ),
            _row(
                event_id=3,
                segment_idx=7,
                period_min=30.0,
                is_duplicate=False,
                date_from="2010-01-01 00:30:00",
                date_to="2010-01-01 02:30:00",
            ),
        ]
    )
    out = tag_peers(df)
    # row 0 sees row 2 (kept) but NOT row 1 (dup).
    assert out.loc[0, "peer_event_ids"] == [3]
    # row 1 is itself a duplicate — no peers recorded.
    assert out.loc[1, "peer_event_ids"] == []
    # row 2 sees row 0 (kept) but NOT row 1 (dup).
    assert out.loc[2, "peer_event_ids"] == [1]


def test_singleton_segment_has_no_peer() -> None:
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=60.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 04:00:00",
            ),
        ]
    )
    out = tag_peers(df)
    assert out.loc[0, "peer_event_ids"] == []
    assert out.loc[0, "peer_periods_min"] == []
    assert out.loc[0, "peer_overlap_frac"] == []


def test_peers_sorted_by_event_id() -> None:
    # Three rows all mutually overlapping; row 0 should list peers
    # sorted ascending regardless of insertion order.
    df = pd.DataFrame(
        [
            _row(
                event_id=10,
                segment_idx=7,
                period_min=30.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 04:00:00",
            ),
            _row(
                event_id=2,
                segment_idx=7,
                period_min=60.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 04:00:00",
            ),
            _row(
                event_id=5,
                segment_idx=7,
                period_min=120.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 04:00:00",
            ),
        ]
    )
    out = tag_peers(df)
    assert out.loc[0, "peer_event_ids"] == [2, 5]
    assert out.loc[0, "peer_periods_min"] == [60.0, 120.0]


def test_input_not_mutated() -> None:
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=60.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 02:00:00",
            ),
        ]
    )
    snapshot = df.copy()
    _ = tag_peers(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_row_order_preserved() -> None:
    df = pd.DataFrame(
        [
            _row(
                event_id=99,
                segment_idx=42,
                period_min=120.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 04:00:00",
            ),
            _row(
                event_id=11,
                segment_idx=42,
                period_min=30.0,
                date_from="2010-01-01 01:00:00",
                date_to="2010-01-01 03:00:00",
            ),
        ]
    )
    out = tag_peers(df)
    assert out["event_id"].tolist() == [99, 11]


def test_invalid_threshold_raises() -> None:
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=60.0,
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 02:00:00",
            ),
        ]
    )
    with pytest.raises(ValueError):
        tag_peers(df, min_overlap_frac=-0.1)
    with pytest.raises(ValueError):
        tag_peers(df, min_overlap_frac=1.5)


def test_missing_required_column_raises() -> None:
    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "segment_idx": 7,
                "date_from": "2010-01-01 00:00:00",
                "date_to": "2010-01-01 02:00:00",
            },
        ]
    )  # missing period_min
    with pytest.raises(ValueError, match="period_min"):
        tag_peers(df)


# ---------------------------------------------------------------------
# derive_co_bands
# ---------------------------------------------------------------------


def test_derive_co_bands_default_classifier() -> None:
    # Periods at 25, 60, 120 min → QP30, QP60, QP120.
    assert derive_co_bands([25.0, 60.0, 120.0]) == "QP120,QP30,QP60"


def test_derive_co_bands_empty_peer_list() -> None:
    assert derive_co_bands([]) == ""


def test_derive_co_bands_excludes_self_band() -> None:
    # Mimics the legacy co_bands view that drops the row's own band.
    assert (
        derive_co_bands([25.0, 60.0, 120.0], exclude_self_band="QP60") == "QP120,QP30"
    )


def test_derive_co_bands_dedup_same_band() -> None:
    # Two peers both at ~60 min → one QP60 label, not two.
    assert derive_co_bands([55.0, 65.0]) == "QP60"


def test_derive_co_bands_period_outside_bands() -> None:
    # 5 min lands in the HF reject guard → no label contributed.
    assert derive_co_bands([5.0, 60.0]) == "QP60"


def test_derive_co_bands_custom_classifier() -> None:
    # Caller-supplied classifier (e.g. paper's even-mode harmonic naming).
    def h_classifier(period_min: float) -> str | None:
        if period_min < 50:
            return "harmonic_high"
        return "harmonic_low"

    assert (
        derive_co_bands([30.0, 90.0], classifier=h_classifier)
        == "harmonic_high,harmonic_low"
    )


def test_derive_co_bands_regression_bridge_to_legacy() -> None:
    # With min_overlap_frac=0.0 + exclude_self_band=row.band + the
    # default classifier, `derive_co_bands` reproduces the old
    # `co_bands` string for any row of the catalogue.
    df = pd.DataFrame(
        [
            _row(
                event_id=1,
                segment_idx=7,
                period_min=55.0,
                band="QP60",
                date_from="2010-01-01 00:00:00",
                date_to="2010-01-01 04:00:00",
            ),
            _row(
                event_id=2,
                segment_idx=7,
                period_min=120.0,
                band="QP120",
                date_from="2010-01-01 01:00:00",
                date_to="2010-01-01 03:00:00",
            ),
            _row(
                event_id=3,
                segment_idx=7,
                period_min=25.0,
                band="QP30",
                date_from="2010-01-01 00:30:00",
                date_to="2010-01-01 02:30:00",
            ),
        ]
    )
    out = tag_peers(df, min_overlap_frac=0.0)
    co_bands_row0 = derive_co_bands(
        out.loc[0, "peer_periods_min"],
        exclude_self_band="QP60",
    )
    assert co_bands_row0 == "QP120,QP30"
