"""Tests for the canonical QP period bands."""

from __future__ import annotations

import math

import pytest

from qp.events.bands import (  # noqa: I001
    QP_BAND_COLORS,
    QP_BAND_NAMES,
    QP_BANDS,
    QP_SEARCH_BAND,
    classify_period,
    get_band,
    is_in_band,
    is_rejected,
)


class TestBandConstants:
    def test_four_qp_bands(self):
        assert set(QP_BAND_NAMES) == {"QP15", "QP30", "QP60", "QP120"}

    def test_band_centroids_in_minutes(self):
        assert QP_BANDS["QP15"].period_centroid_minutes == 15
        assert QP_BANDS["QP30"].period_centroid_minutes == 30
        assert QP_BANDS["QP60"].period_centroid_minutes == 60
        assert QP_BANDS["QP120"].period_centroid_minutes == 120

    def test_bands_tile_contiguously(self):
        """No gaps and no overlaps between adjacent bands."""
        ordered = sorted(QP_BANDS.values(), key=lambda b: b.period_min_sec)
        for a, b in zip(ordered, ordered[1:], strict=False):
            assert a.period_max_sec == b.period_min_sec, (
                f"gap or overlap between {a.name} and {b.name}"
            )

    def test_bands_are_octaves(self):
        """Each band's upper edge equals 2× its lower edge."""
        for band in QP_BANDS.values():
            assert math.isclose(band.period_max_sec, 2.0 * band.period_min_sec), (
                f"{band.name} is not an octave"
            )

    def test_band_freq_inversion(self):
        for band in QP_BANDS.values():
            assert math.isclose(band.freq_min_hz, 1.0 / band.period_max_sec)
            assert math.isclose(band.freq_max_hz, 1.0 / band.period_min_sec)

    def test_band_is_frozen(self):
        b = QP_BANDS["QP60"]
        with pytest.raises((AttributeError, TypeError)):
            b.name = "X"  # type: ignore[misc]


class TestBandColors:
    def test_color_per_band(self):
        assert set(QP_BAND_COLORS) == set(QP_BAND_NAMES)

    def test_colors_are_hex(self):
        for c in QP_BAND_COLORS.values():
            assert c.startswith("#") and len(c) == 7


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
    @pytest.mark.parametrize(
        "period_min,band,expected",
        [
            (15.0, "QP15", True),
            (30.0, "QP30", True),
            (60.0, "QP60", True),
            (120.0, "QP120", True),
            (10.0, "QP15", True),  # closed left edge
            (20.0, "QP15", False),  # half-open right edge → QP30
            (20.0, "QP30", True),
            (40.0, "QP60", True),  # gap removed: 40 now belongs to QP60
            (40.0, "QP30", False),
            (80.0, "QP120", True),
            (80.0, "QP60", False),
            (159.9, "QP120", True),
            (160.0, "QP120", False),
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
        assert classify_period(15 * 60) == "QP15"
        assert classify_period(30 * 60) == "QP30"
        assert classify_period(60 * 60) == "QP60"
        assert classify_period(120 * 60) == "QP120"

    def test_no_gaps_in_qp_range(self):
        """Every period in [10, 160) min must classify to a band."""
        for p_min in (10.5, 19.9, 20.0, 39.9, 40.0, 79.9, 80.0, 159.9):
            assert classify_period(p_min * 60) is not None, (
                f"{p_min} min unexpectedly returned None"
            )

    def test_outside_returns_none(self):
        assert classify_period(5 * 60) is None  # rejected HF
        assert classify_period(15 * 3600) is None  # rejected LF
        assert classify_period(160 * 60) is None  # above QP120


class TestSearchBand:
    def test_search_band_unions_qp_bands(self):
        """``QP_SEARCH_BAND`` is exactly the union of the four QP bands."""
        for band in QP_BANDS.values():
            assert QP_SEARCH_BAND.period_min_sec <= band.period_min_sec
            assert QP_SEARCH_BAND.period_max_sec >= band.period_max_sec
        assert QP_SEARCH_BAND.period_min_sec == 10 * 60
        assert QP_SEARCH_BAND.period_max_sec == 160 * 60
