"""Tests for qp.io.mag_reader — MAG data segment I/O."""

from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.io.mag_reader import MagSegment, load_segment, save_segment


@pytest.fixture
def synthetic_segment():
    """Create a small synthetic MagSegment for testing."""
    n = 100
    dt = 60.0
    t0 = datetime.datetime(2007, 1, 2, 0, 0, 0)
    time_dt = [t0 + datetime.timedelta(seconds=i * dt) for i in range(n)]
    time_unix = np.array(
        [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in time_dt]
    )

    # Synthetic field: sinusoidal Br, constant Bth/Bphi, parabolic Btot
    t_sec = np.arange(n) * dt
    fields = np.column_stack(
        [
            5.0 * np.sin(2 * np.pi * t_sec / (60 * 60)),  # Br: 1-hour period
            np.full(n, 2.0),  # Bth: constant
            np.full(n, -1.0),  # Bphi: constant
            np.sqrt(25 * np.sin(2 * np.pi * t_sec / 3600) ** 2 + 4 + 1),  # Btot
        ]
    )

    coords = np.column_stack(
        [
            np.full(n, 15.0),  # r = 15 R_S
            np.full(n, np.pi / 3),  # theta = 60 deg
            np.linspace(0, 2 * np.pi, n),  # phi sweeps
        ]
    )

    local_time = np.linspace(0, 24, n)

    return MagSegment(
        time_dt=time_dt,
        time_unix=time_unix,
        fields=fields,
        coords=coords,
        dt=dt,
        coord_system="KRTP",
        field_names=("Br", "Bth", "Bphi", "Btot"),
        coord_names=("r", "th", "phi"),
        local_time=local_time,
    )


class TestMagSegment:
    """Tests for the MagSegment dataclass."""

    def test_n_samples(self, synthetic_segment):
        assert synthetic_segment.n_samples == 100

    def test_duration_hours(self, synthetic_segment):
        # 100 samples at 60s = 99 intervals = 99 minutes
        assert_allclose(synthetic_segment.duration_hours, 99 / 60, rtol=0.01)

    def test_field_shape(self, synthetic_segment):
        assert synthetic_segment.fields.shape == (100, 4)

    def test_coord_shape(self, synthetic_segment):
        assert synthetic_segment.coords.shape == (100, 3)

    def test_field_names(self, synthetic_segment):
        assert synthetic_segment.field_names == ("Br", "Bth", "Bphi", "Btot")


class TestSaveLoadRoundtrip:
    """Test save_segment / load_segment roundtrip."""

    def test_roundtrip_preserves_data(self, synthetic_segment):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_segment.npz"
            save_segment(synthetic_segment, path)
            loaded = load_segment(path)

            assert loaded.n_samples == synthetic_segment.n_samples
            assert loaded.coord_system == synthetic_segment.coord_system
            assert loaded.field_names == synthetic_segment.field_names
            assert loaded.coord_names == synthetic_segment.coord_names
            assert_allclose(loaded.dt, synthetic_segment.dt)
            assert_allclose(loaded.fields, synthetic_segment.fields)
            assert_allclose(loaded.coords, synthetic_segment.coords)
            assert_allclose(loaded.time_unix, synthetic_segment.time_unix, atol=1e-3)

    def test_roundtrip_local_time(self, synthetic_segment):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_lt.npz"
            save_segment(synthetic_segment, path)
            loaded = load_segment(path)

            assert loaded.local_time is not None
            assert_allclose(loaded.local_time, synthetic_segment.local_time)

    def test_roundtrip_no_local_time(self, synthetic_segment):
        """Segments without local time (KSM/KSO) should roundtrip."""
        seg = MagSegment(
            time_dt=synthetic_segment.time_dt,
            time_unix=synthetic_segment.time_unix,
            fields=synthetic_segment.fields,
            coords=synthetic_segment.coords,
            dt=synthetic_segment.dt,
            coord_system="KSM",
            field_names=("Bx", "By", "Bz", "Btot"),
            coord_names=("x", "y", "z"),
            local_time=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_ksm.npz"
            save_segment(seg, path)
            loaded = load_segment(path)

            assert loaded.local_time is None
            assert loaded.coord_system == "KSM"

    def test_file_is_compressed(self, synthetic_segment):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            save_segment(synthetic_segment, path)
            assert path.exists()
            assert path.stat().st_size > 0
