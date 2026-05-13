"""Band-agnostic detection: ridges form anywhere in 10–160 min.

Companion tests to ``test_detector_multi.py``. Those tests verify that
injection at a band centroid (30 / 60 / 120 min) lands in that band.
These verify the complementary property the refactor is meant to give
us: a wave whose period sits *near* an octave boundary is still
detected as ONE ridge, with the correct post-hoc band label, instead
of being split into two amputated halves on either side of the band
edge.
"""

from __future__ import annotations

import datetime

import pytest

from qp.events.bands import classify_period
from qp.events.catalog import WaveTemplate, WavePacketPeak
from qp.events.detector import dedup_peaks_by_period, detect_wave_packets_multi
from qp.signal.synthetic import simulate_signal


@pytest.fixture
def time_axis():
    n = 2160
    dt = 60.0
    t0 = datetime.datetime(2007, 1, 1)
    times = [t0 + datetime.timedelta(seconds=i * dt) for i in range(n)]
    return n, dt, times


def _inject(
    period_min: float,
    n: int,
    dt: float,
    amplitude: float = 2.0,
    decay_hours: float = 4.0,
    center_hours: float = 18.0,
):
    wave = WaveTemplate(
        period=period_min * 60.0,
        amplitude=amplitude,
        decay_width=decay_hours * 3600.0,
        shift=center_hours * 3600.0,
    )
    _, signal = simulate_signal(
        n_samples=n, dt=dt, waves=[wave], noise_sigma=0.0, seed=1
    )
    return signal


class TestEdgeAdjacentPeriodsLandInCorrectBand:
    """Periods at 21, 39, 41, 79 min sit just inside their host band.

    The pre-refactor detector handled these fine; we keep these tests
    to confirm no regression and to anchor the boundary-no-split test
    below.
    """

    # Note on the 82-min case (was 81 prior to the S4 scale-to-frequency
    # correction in qp.signal.wavelet): the T&C98 Table 1 exact inverse
    # shifts reported peak periods by ~0.5–1.5 % relative to the legacy
    # simplified inverse. 81 min landed at 79.8 min after the fix and
    # crossed the QP60/QP120 band edge at 80 min; 82 min keeps the test
    # honest about edge-adjacent behaviour with a safer ~2-min margin.
    @pytest.mark.parametrize(
        "period_min,expected_band",
        [(21, "QP30"), (39, "QP30"), (41, "QP60"), (79, "QP60"), (82, "QP120")],
    )
    def test_post_hoc_label_matches_period(self, time_axis, period_min, expected_band):
        n, dt, times = time_axis
        signal = _inject(period_min=period_min, n=n, dt=dt)
        packets = detect_wave_packets_multi(
            signal,
            times,
            dt=dt,
            min_duration_hours=1.5,
            min_pixels=30,
        )
        # At least one packet should be in the expected band.
        in_band = [p for p in packets if p.band == expected_band]
        debug = [
            (p.band, None if p.period_sec is None else round(p.period_sec / 60, 1))
            for p in packets
        ]
        assert len(in_band) >= 1, (
            f"no {expected_band} packet for {period_min}-min injection; got {debug}"
        )


class TestBoundaryPeriodIsOneRidge:
    """A wave sitting *exactly* on a band edge (40 or 80 min) used to be
    cut at the edge and could fail ``min_pixels`` in both halves; with
    band-agnostic detection it is one ridge.
    """

    @pytest.mark.parametrize("period_min", [40, 80])
    def test_one_packet_at_octave_boundary(self, time_axis, period_min):
        n, dt, times = time_axis
        signal = _inject(period_min=period_min, n=n, dt=dt)
        packets = detect_wave_packets_multi(
            signal,
            times,
            dt=dt,
            min_duration_hours=1.5,
            min_pixels=30,
        )
        # Filter to the strongest packet near the injected period.
        # We tolerate a parabolic-interp offset of ~10% (still a half-
        # bin in 300-row log-period CWT grid).
        near = [
            p
            for p in packets
            if p.period_sec is not None
            and abs(p.period_sec - period_min * 60.0) / (period_min * 60.0) < 0.1
        ]
        debug_all = [
            None if p.period_sec is None else round(p.period_sec / 60, 1)
            for p in packets
        ]
        assert len(near) >= 1, (
            f"no packet within 10% of {period_min} min; got {debug_all}"
        )
        # One ridge, not two near-twins on either side of the edge.
        # The half-power splitter may keep both sub-peaks if the
        # injection itself genuinely has two envelope lobes, but for
        # a single Gaussian decay we expect exactly one strong peak.
        strongest = max(near, key=lambda p: p.prominence)
        # Sub-leading near-boundary peaks should be at least 2x weaker.
        others = [p for p in near if p is not strongest]
        if others:
            second = max(others, key=lambda p: p.prominence)
            debug_near = [
                (
                    None if p.period_sec is None else round(p.period_sec / 60, 1),
                    p.prominence,
                )
                for p in near
            ]
            assert second.prominence < 0.5 * strongest.prominence, (
                f"boundary split: two near-equal ridges at {period_min} min: "
                f"{debug_near}"
            )


class TestPeriodHistogramIsSmooth:
    """Inject a sweep of periods across a band edge and confirm the
    detected period distribution does not pile at the edge.
    """

    def test_no_pile_up_at_40min(self, time_axis):
        n, dt, times = time_axis
        detected_periods: list[float] = []
        for p_min in (36, 38, 40, 42, 44):
            signal = _inject(period_min=p_min, n=n, dt=dt)
            packets = detect_wave_packets_multi(
                signal,
                times,
                dt=dt,
                min_duration_hours=1.5,
                min_pixels=30,
            )
            if not packets:
                continue
            best = max(packets, key=lambda p: p.prominence)
            if best.period_sec is not None:
                detected_periods.append(best.period_sec / 60.0)
        # Each injection should be recovered within the parabolic-
        # interpolation tolerance — and there should be no spike of
        # detections clustering at exactly 40 min.
        edge_pile = sum(abs(p - 40.0) < 0.3 for p in detected_periods)
        # At most one detection within 0.3 min of the edge (the 40-min
        # injection itself).
        assert edge_pile <= 1, (
            f"detected periods clustered at the 40-min edge: {detected_periods}"
        )


class TestDedupByPeriodProximity:
    """``dedup_peaks_by_period`` compares periods in log2 space, not
    band-string equality. Distinct-period peaks within the same 2-h
    window survive; near-period peaks collapse.
    """

    def _peak(
        self, t_offset_min: float, period_min: float, band: str | None = None
    ) -> WavePacketPeak:
        t0 = datetime.datetime(2007, 1, 1)
        period_sec = period_min * 60.0
        return WavePacketPeak(
            peak_time=t0 + datetime.timedelta(minutes=t_offset_min),
            prominence=1.0,
            date_from=t0 + datetime.timedelta(minutes=t_offset_min - 60),
            date_to=t0 + datetime.timedelta(minutes=t_offset_min + 60),
            band=band if band is not None else classify_period(period_sec),
            period_sec=period_sec,
            period_fwhm_sec=10 * 60.0,
        )

    def test_distinct_periods_within_window_both_kept(self):
        # 25 min vs 65 min: ~1.4 octaves apart → not duplicates.
        peaks = [
            self._peak(0, 25),
            self._peak(60, 65),  # 1 h later
        ]
        kept = dedup_peaks_by_period(peaks, dt_sec=7200.0)
        assert len(kept) == 2

    def test_near_periods_within_window_collapsed(self):
        # 60 min vs 61 min: ~0.02 octaves apart → duplicates.
        peaks = [
            self._peak(0, 60),
            self._peak(30, 61),
        ]
        kept = dedup_peaks_by_period(peaks, dt_sec=7200.0)
        assert len(kept) == 1

    def test_same_band_far_apart_in_time_both_kept(self):
        # Both QP60, but 3 h apart → not duplicates.
        peaks = [
            self._peak(0, 60),
            self._peak(180, 60),
        ]
        kept = dedup_peaks_by_period(peaks, dt_sec=7200.0)
        assert len(kept) == 2

    def test_cross_band_close_period_still_collapses(self):
        # 39 min (QP30) vs 41 min (QP60): same physical wave straddling
        # an octave edge — the old band-keyed dedup let both through;
        # the new period-keyed dedup correctly merges them.
        peaks = [
            self._peak(0, 39),
            self._peak(30, 41),
        ]
        kept = dedup_peaks_by_period(peaks, dt_sec=7200.0)
        assert len(kept) == 1
