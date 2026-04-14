"""Tests for the canonical QP period bands."""

from __future__ import annotations

import math

import pytest

from qp.events.bands import (
    QP_BANDS,
    QP_BAND_NAMES,
    SEARCH_BAND_EXTENDED,
    classify_period,
    get_band,
    is_in_band,
    is_rejected,
)


class TestBandConstants:
    def test_three_qp_bands(self):
        assert set(QP_BAND_NAMES) == {"QP30", "QP60", "QP120"}

    def test_band_centroids_in_minutes(self):
        assert QP_BANDS["QP30"].period_centroid_minutes == 30
        assert QP_BANDS["QP60"].period_centroid_minutes == 60
        assert QP_BANDS["QP120"].period_centroid_minutes == 120

    def test_bands_dont_overlap(self):
        ordered = sorted(QP_BANDS.values(), key=lambda b: b.period_min_sec)
        for a, b in zip(ordered, ordered[1:]):
            assert a.period_max_sec <= b.period_min_sec, (
                f"{a.name} and {b.name} overlap"
            )

    def test_band_freq_inversion(self):
        for band in QP_BANDS.values():
            assert math.isclose(
                band.freq_min_hz, 1.0 / band.period_max_sec
            )
            assert math.isclose(
                band.freq_max_hz, 1.0 / band.period_min_sec
            )

    def test_band_is_frozen(self):
        b = QP_BANDS["QP60"]
        with pytest.raises((AttributeError, TypeError)):
            b.name = "X"  # type: ignore[misc]


class TestGetBand:
    def test_lookup_by_name(self):
        b = get_band("QP60")
        assert b.name == "QP60"

    def test_lookup_case_insensitive(self):
        assert get_band("qp60").name == "QP60"

    def test_pass_through_band_object(self):
        b = QP_BANDS["QP120"]
        assert get_band(b) is b

    def test_unknown_band_raises(self):
        with pytest.raises(KeyError):
            get_band("QP999")


class TestIsInBand:
    """Bands are contiguous half-octave windows (v4):
    QP30 [21.21, 42.43), QP60 [42.43, 84.85), QP120 [84.85, 169.71) min.
    """
    @pytest.mark.parametrize(
        "period_min,band,expected",
        [
            (30.0, "QP30", True),
            (60.0, "QP60", True),
            (120.0, "QP120", True),
            (50.0, "QP30", False),  # 50 min belongs to QP60
            (42.43, "QP30", False),  # half-open at upper edge
            (42.43, "QP60", True),
            (84.0, "QP60", True),
            (85.0, "QP120", True),
            (21.0, "QP30", False),  # below QP30 lower (21.21)
            (22.0, "QP30", True),
            (169.0, "QP120", True),
            (170.0, "QP120", False),  # above QP120 upper (169.71)
        ],
    )
    def test_membership(self, period_min, band, expected):
        assert is_in_band(period_min * 60.0, band) is expected


class TestRejectionBands:
    def test_hf_guard(self):
        assert is_rejected(5 * 60)
        assert is_rejected(9.99 * 60)
        assert not is_rejected(10.01 * 60)

    def test_lf_guard(self):
        assert is_rejected(13 * 3600)
        assert not is_rejected(11.99 * 3600)


class TestClassifyPeriod:
    def test_each_band(self):
        assert classify_period(30 * 60) == "QP30"
        assert classify_period(60 * 60) == "QP60"
        assert classify_period(120 * 60) == "QP120"

    def test_gap_labels_inside_search(self):
        # Below QP30 but within [15, 180] → sub_qp30
        assert classify_period(17 * 60) == "sub_qp30"
        # Above QP120 but within [15, 180] → super_qp120
        assert classify_period(175 * 60) == "super_qp120"

    def test_outside_search_returns_none(self):
        assert classify_period(5 * 60) is None     # below 15 min
        assert classify_period(200 * 60) is None   # above 180 min
        assert classify_period(15 * 3600) is None  # 15 hours


class TestSearchBand:
    def test_search_band_covers_qp_bands(self):
        for band in QP_BANDS.values():
            assert SEARCH_BAND_EXTENDED.period_min_sec <= band.period_min_sec
            assert SEARCH_BAND_EXTENDED.period_max_sec > band.period_max_sec
