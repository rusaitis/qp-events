"""Tests for ``qp.events.persistence``: tabular event I/O."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from qp.events.catalog import WavePacketPeak
from qp.events.detector import DetectedEvent
from qp.events.persistence import (
    REQUIRED_COLUMNS,
    detection_to_dict,
    event_to_record,
    events_to_parquet,
    read_events_parquet,
)


def _make_detection(
    *,
    band: str = "QP60",
    period_min: float = 60.0,
    duration_h: float = 4.0,
    q: float = 4.5,
    par_frac: float = 0.1,
    d: float = 0.85,
) -> DetectedEvent:
    t0 = datetime.datetime(2008, 6, 15, 0, 0, 0)
    peak = WavePacketPeak(
        peak_time=t0 + datetime.timedelta(hours=duration_h / 2),
        prominence=2.0,
        date_from=t0,
        date_to=t0 + datetime.timedelta(hours=duration_h),
        local_time=12.0,
        r_distance=20.0,
        theta=np.radians(60.0),
        band=band,
        period_sec=period_min * 60.0,
        period_fwhm_sec=(period_min / q) * 60.0,
    )
    return DetectedEvent(
        peak=peak,
        q_factor=q,
        mva_par_frac=par_frac,
        stokes_d=d,
        b_perp1_amp=0.20,
        b_perp2_amp=0.18,
        b_par_amp=0.05,
        stokes_i=1.0,
        stokes_q=0.1,
        stokes_u=0.05,
        stokes_v=d * 1.0,
        ellipticity=0.6,
        inclination_deg=12.0,
        polarized_fraction=d,
    )


class TestEventToRecord:
    def test_required_columns_populated(self) -> None:
        det = _make_detection()
        rec = event_to_record(det, event_id=42, segment_id="seg_00007")
        for col in REQUIRED_COLUMNS:
            assert col in rec, f"missing required column: {col}"
            assert rec[col] is not None

    def test_gate_values_match_detection(self) -> None:
        det = _make_detection(q=5.5, par_frac=0.27, d=0.91)
        rec = event_to_record(det, event_id=0, segment_id="x")
        assert rec["q_factor"] == pytest.approx(5.5)
        assert rec["mva_par_frac"] == pytest.approx(0.27)
        assert rec["stokes_d"] == pytest.approx(0.91)

    def test_iso_timestamps(self) -> None:
        det = _make_detection()
        rec = event_to_record(det, event_id=0, segment_id="x")
        # Round-trip through fromisoformat
        df = datetime.datetime.fromisoformat(rec["date_from"])
        dt = datetime.datetime.fromisoformat(rec["date_to"])
        assert (dt - df).total_seconds() == pytest.approx(
            rec["duration_minutes"] * 60.0,
        )

    def test_extra_fields_merge(self) -> None:
        det = _make_detection()
        rec = event_to_record(
            det,
            event_id=0,
            segment_id="x",
            extra={"region": "magnetosphere", "ksm_x": 12.0},
        )
        assert rec["region"] == "magnetosphere"
        assert rec["ksm_x"] == 12.0

    def test_extra_collision_raises(self) -> None:
        det = _make_detection()
        with pytest.raises(ValueError, match="collide"):
            event_to_record(
                det,
                event_id=0,
                segment_id="x",
                extra={"q_factor": 99.0},
            )

    def test_missing_period_raises(self) -> None:
        peak = WavePacketPeak(
            peak_time=datetime.datetime(2008, 1, 1),
            prominence=1.0,
            date_from=datetime.datetime(2008, 1, 1),
            date_to=datetime.datetime(2008, 1, 1, 4),
            band="QP60",
            period_sec=None,
        )
        det = DetectedEvent(
            peak=peak,
            q_factor=4.0,
            mva_par_frac=0.1,
            stokes_d=0.8,
            b_perp1_amp=0.1,
            b_perp2_amp=0.1,
            b_par_amp=0.05,
            stokes_i=1.0,
            stokes_q=0.1,
            stokes_u=0.0,
            stokes_v=0.8,
            ellipticity=0.5,
            inclination_deg=0.0,
            polarized_fraction=0.8,
        )
        with pytest.raises(ValueError, match="period_sec"):
            event_to_record(det, event_id=0, segment_id="x")


class TestParquetRoundTrip:
    def test_roundtrip_three_events(self, tmp_path: Path) -> None:
        rows = [
            event_to_record(
                _make_detection(band=b, period_min=p),
                event_id=k,
                segment_id=f"seg_{k:03d}",
            )
            for k, (b, p) in enumerate(
                (("QP30", 30.0), ("QP60", 60.0), ("QP120", 120.0))
            )
        ]
        path = tmp_path / "events.parquet"
        n = events_to_parquet(rows, path, attrs={"run": "test"})
        assert n == 3
        assert path.exists()
        df, attrs = read_events_parquet(path)
        assert len(df) == 3
        assert attrs["run"] == "test"
        assert attrs["schema_version"] == "round8.2"
        assert sorted(df["band"].tolist()) == ["QP120", "QP30", "QP60"]
        assert (df["q_factor"] > 0).all()

    def test_meta_sidecar_is_json(self, tmp_path: Path) -> None:
        det = _make_detection()
        rec = event_to_record(det, event_id=0, segment_id="x")
        path = tmp_path / "events.parquet"
        events_to_parquet([rec], path, attrs={"foo": "bar", "n": 1})
        meta = json.loads((path.with_suffix(".parquet.meta.json")).read_text())
        assert meta["attrs"]["foo"] == "bar"
        assert meta["attrs"]["n"] == 1
        assert meta["attrs"]["schema_version"] == "round8.2"
        assert meta["n_rows"] == 1

    def test_empty_records(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.parquet"
        n = events_to_parquet([], path, attrs={})
        assert n == 0


class TestDetectionToDict:
    def test_includes_peak_fields(self) -> None:
        det = _make_detection(band="QP30", period_min=30.0)
        d = detection_to_dict(det)
        assert d["band"] == "QP30"
        assert d["period_sec"] == pytest.approx(1800.0)
        assert d["q_factor"] == pytest.approx(4.5)
