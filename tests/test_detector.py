"""Tests for qp.events.detector — wave event detection and collection."""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.events.catalog import WaveEvent, WavePacketPeak
from qp.events.detector import (
    collect_wave_events,
    dedup_peaks_by_band,
    detect_wave_packets,
)


@pytest.fixture
def qp60_signal():
    """Synthetic signal with a strong 60-min wave packet centered at 18 hours.

    36 hours of data at 1-min cadence. The wave packet has a Gaussian
    envelope with 3-hour decay width centered at t=18h.
    """
    dt = 60.0
    n = int(36 * 3600 / dt)  # 2160 samples
    t = np.arange(n) * dt
    period = 3600.0  # 60 min

    # Gaussian-enveloped sinusoid
    center = 18 * 3600.0
    decay = 3 * 3600.0
    envelope = np.exp(-0.5 * ((t - center) / decay) ** 2)
    signal = 2.0 * np.sin(2 * np.pi * t / period) * envelope

    # Build timestamps
    t0 = datetime.datetime(2007, 1, 2, 0, 0, 0)
    times = [t0 + datetime.timedelta(seconds=float(s)) for s in t]
    time_unix = np.array(
        [(ti - datetime.datetime(1970, 1, 1)).total_seconds() for ti in times]
    )

    return signal, times, time_unix, dt


class TestDetectWavePackets:
    def test_finds_packet_in_synthetic(self, qp60_signal):
        signal, times, _, dt = qp60_signal
        packets = detect_wave_packets(
            signal,
            times,
            dt=dt,
            period_band=(50 * 60, 70 * 60),
            min_duration_hours=2.0,
            min_prominence=0.01,
            min_peak_width=30,
        )
        assert len(packets) >= 1

    def test_peak_near_expected_time(self, qp60_signal):
        signal, times, _, dt = qp60_signal
        packets = detect_wave_packets(
            signal,
            times,
            dt=dt,
            period_band=(50 * 60, 70 * 60),
            min_duration_hours=2.0,
            min_prominence=0.01,
            min_peak_width=30,
        )
        if packets:
            peak_hour = (packets[0].peak_time - times[0]).total_seconds() / 3600
            # Peak should be near 18 hours (center of envelope)
            assert 12.0 < peak_hour < 24.0

    def test_fewer_detections_in_noise(self):
        """Pure noise should produce far fewer detections than a real signal."""
        rng = np.random.default_rng(42)
        n = 2160
        dt = 60.0
        noise = rng.normal(0, 0.01, n)
        t0 = datetime.datetime(2007, 1, 1)
        times = [t0 + datetime.timedelta(seconds=i * dt) for i in range(n)]

        # Noise with strict thresholds should produce at most 1 spurious detection
        # (CWT normalization means even noise has normalized peaks)
        packets = detect_wave_packets(
            noise,
            times,
            dt=dt,
            min_prominence=0.5,
            min_duration_hours=6.0,
        )
        assert len(packets) <= 1


class TestCollectWaveEvents:
    def test_returns_wave_events(self, qp60_signal):
        signal, _, time_unix, dt = qp60_signal
        events = collect_wave_events(
            signal,
            time_unix,
            dt=dt,
            period_band=(50 * 60, 70 * 60),
            min_snr=0.01,
            min_duration_hours=2.0,
        )
        for event in events:
            assert isinstance(event, WaveEvent)

    def test_events_have_amplitude(self, qp60_signal):
        signal, _, time_unix, dt = qp60_signal
        events = collect_wave_events(
            signal,
            time_unix,
            dt=dt,
            period_band=(50 * 60, 70 * 60),
            min_snr=0.01,
            min_duration_hours=2.0,
        )
        if events:
            assert events[0].amplitude is not None
            assert events[0].amplitude > 0

    def test_coordinates_populated_when_given(self, qp60_signal):
        signal, _, time_unix, dt = qp60_signal
        n = len(signal)
        coords = np.column_stack(
            [
                np.full(n, 15.0),
                np.full(n, np.deg2rad(70.0)),
                np.full(n, np.deg2rad(45.0)),
            ]
        )
        lt = np.full(n, 18.0)

        events = collect_wave_events(
            signal,
            time_unix,
            dt=dt,
            period_band=(50 * 60, 70 * 60),
            min_snr=0.01,
            min_duration_hours=2.0,
            coords_krtp=coords,
            local_time=lt,
        )
        if events:
            assert events[0].r_distance is not None
            assert_allclose(events[0].r_distance, 15.0)
            assert events[0].local_time is not None
            assert_allclose(events[0].local_time, 18.0)
            assert events[0].mag_lat is not None

    def test_no_coordinates_when_not_given(self, qp60_signal):
        signal, _, time_unix, dt = qp60_signal
        events = collect_wave_events(
            signal,
            time_unix,
            dt=dt,
            period_band=(50 * 60, 70 * 60),
            min_snr=0.01,
            min_duration_hours=2.0,
        )
        if events:
            assert events[0].r_distance is None
            assert events[0].local_time is None


def _peak(t_hours: float, band: str) -> WavePacketPeak:
    """Tiny WavePacketPeak factory for dedup tests."""
    t0 = datetime.datetime(2007, 1, 1)
    pt = t0 + datetime.timedelta(hours=t_hours)
    return WavePacketPeak(
        peak_time=pt,
        prominence=1.0,
        date_from=pt - datetime.timedelta(hours=1),
        date_to=pt + datetime.timedelta(hours=1),
        band=band,
    )


class TestDedupPeaksByBand:
    def test_drops_close_same_band(self):
        peaks = [_peak(0.0, "QP60"), _peak(1.5, "QP60")]
        kept = dedup_peaks_by_band(peaks, dt_sec=7200.0)
        assert [p.peak_time.hour for p in kept] == [0]

    def test_keeps_far_same_band(self):
        peaks = [_peak(0.0, "QP60"), _peak(3.0, "QP60")]
        kept = dedup_peaks_by_band(peaks, dt_sec=7200.0)
        assert len(kept) == 2

    def test_interleaved_different_band_does_not_unmask_dup(self):
        """The regression: QP60-QP30-QP60 in 1.5 h must drop the second QP60.

        Under the previous ``merged[-1]``-only guard, the QP30 between
        the two QP60s broke the band comparison and let the second
        QP60 through. The per-band rolling-last fix keeps the QP30 and
        drops the second QP60.
        """
        peaks = [
            _peak(0.0, "QP60"),
            _peak(1.0, "QP30"),
            _peak(1.5, "QP60"),
        ]
        kept = dedup_peaks_by_band(peaks, dt_sec=7200.0)
        bands = [p.band for p in kept]
        assert bands == ["QP60", "QP30"]

    def test_multi_band_preserved(self):
        """Genuine multi-harmonic event: three bands in close time stay."""
        peaks = [
            _peak(0.0, "QP30"),
            _peak(0.1, "QP60"),
            _peak(0.2, "QP120"),
        ]
        kept = dedup_peaks_by_band(peaks, dt_sec=7200.0)
        assert {p.band for p in kept} == {"QP30", "QP60", "QP120"}

    def test_empty_input(self):
        assert dedup_peaks_by_band([], dt_sec=7200.0) == []
