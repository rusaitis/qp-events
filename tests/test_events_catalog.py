"""Tests for qp.events.catalog — WaveEvent and WavePacketPeak dataclasses.

These are heavily used downstream (persistence, detector, plotting), so a
regression on field set or frozen-ness would ripple widely.
"""

from __future__ import annotations

import dataclasses
import datetime

import pytest

from qp.events.catalog import WaveEvent, WavePacketPeak, WaveTemplate


def _epoch(h: int = 0, m: int = 0) -> datetime.datetime:
    return datetime.datetime(2007, 1, 2, h, m, 0)


class TestWaveEvent:
    def test_construct_with_required_only(self):
        ev = WaveEvent(date_from=_epoch(0), date_to=_epoch(2))
        assert ev.date_from == _epoch(0)
        # All other fields default to None
        assert ev.period is None
        assert ev.band is None

    def test_duration_helpers(self):
        ev = WaveEvent(date_from=_epoch(0), date_to=_epoch(3))
        assert ev.duration_hours == pytest.approx(3.0)
        assert ev.duration_minutes == pytest.approx(180.0)

    def test_period_minutes_handles_none(self):
        ev = WaveEvent(date_from=_epoch(0), date_to=_epoch(1))
        assert ev.period_minutes is None
        ev2 = WaveEvent(date_from=_epoch(0), date_to=_epoch(1), period=3600.0)
        assert ev2.period_minutes == pytest.approx(60.0)

    def test_is_significant(self):
        ev = WaveEvent(date_from=_epoch(0), date_to=_epoch(1), snr=5.0)
        assert ev.is_significant(snr_threshold=3.0)
        assert not ev.is_significant(snr_threshold=10.0)
        # None SNR → not significant
        ev_no_snr = WaveEvent(date_from=_epoch(0), date_to=_epoch(1))
        assert not ev_no_snr.is_significant()

    def test_is_frozen(self):
        ev = WaveEvent(date_from=_epoch(0), date_to=_epoch(1))
        with pytest.raises((AttributeError, Exception)):
            ev.period = 60.0  # type: ignore[misc]

    def test_equality_by_value(self):
        a = WaveEvent(date_from=_epoch(0), date_to=_epoch(1), band="QP60")
        b = WaveEvent(date_from=_epoch(0), date_to=_epoch(1), band="QP60")
        assert a == b

    def test_dataclasses_replace_preserves_unchanged_fields(self):
        ev = WaveEvent(date_from=_epoch(0), date_to=_epoch(1), band="QP60", snr=4.0)
        ev2 = dataclasses.replace(ev, snr=7.0)
        assert ev2.snr == 7.0
        assert ev2.band == "QP60"
        assert ev2.date_from == ev.date_from


class TestWavePacketPeak:
    def test_q_factor_none_if_fwhm_missing(self):
        p = WavePacketPeak(
            peak_time=_epoch(1),
            prominence=1.0,
            date_from=_epoch(0),
            date_to=_epoch(2),
            period_sec=3600.0,
        )
        assert p.q_factor is None

    def test_q_factor_computed(self):
        p = WavePacketPeak(
            peak_time=_epoch(1),
            prominence=1.0,
            date_from=_epoch(0),
            date_to=_epoch(2),
            period_sec=3600.0,
            period_fwhm_sec=600.0,
        )
        # Q = period / fwhm = 3600 / 600 = 6
        assert p.q_factor == pytest.approx(6.0)

    def test_duration_hours(self):
        p = WavePacketPeak(
            peak_time=_epoch(1),
            prominence=1.0,
            date_from=_epoch(0),
            date_to=_epoch(4),
        )
        assert p.duration_hours == pytest.approx(4.0)

    def test_is_frozen(self):
        p = WavePacketPeak(
            peak_time=_epoch(0),
            prominence=1.0,
            date_from=_epoch(0),
            date_to=_epoch(1),
        )
        with pytest.raises((AttributeError, Exception)):
            p.prominence = 2.0  # type: ignore[misc]


class TestWaveTemplate:
    def test_defaults_match_paper(self):
        wt = WaveTemplate()
        # default period 3600 s = 60 min (QP60 anchor)
        assert wt.period == 3600.0
        assert wt.amplitude == 1.0
        assert wt.waveform == "sine"

    def test_replace_for_chirp(self):
        wt = WaveTemplate(period=1800.0, amplitude=3.0)
        wt2 = dataclasses.replace(wt, chirp_rate=1e-6)
        assert wt2.period == 1800.0
        assert wt2.chirp_rate == pytest.approx(1e-6)
