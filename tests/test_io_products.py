"""Smoke tests for qp.io.products.

The real DataProducts files live outside the repository (25 GB Cassini
archive), so these tests use a temporary directory with a synthetic
.npy file to verify the loader plumbing without touching DATA/.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np

from qp.io.products import load_spacecraft_position


def test_load_spacecraft_position_from_custom_dir(tmp_path: Path) -> None:
    # Build a tiny synthetic CassiniLocation_KSM.npy mimicking the
    # 11-column object schema documented in qp.io.products.
    n = 5
    arr = np.empty((n, 11), dtype=object)
    t0 = datetime.datetime(2007, 1, 1, tzinfo=datetime.UTC)
    for i in range(n):
        arr[i, 0] = t0 + datetime.timedelta(hours=i)
        arr[i, 1:5] = [1.0, 2.0, 3.0, np.sqrt(14.0)]  # B components, |B|
        arr[i, 5:8] = [10.0 + i, 0.0, 0.0]            # x, y, z (R_S)
        arr[i, 8:11] = [0, 1, 0]                      # location flags
    np.save(tmp_path / "CassiniLocation_KSM.npy", arr, allow_pickle=True)

    out = load_spacecraft_position(tmp_path)

    assert out.shape == (n, 11)
    # Column 5 (KSM x) should round-trip exactly.
    np.testing.assert_array_equal(out[:, 5], np.arange(10.0, 10.0 + n))
    # Column 0 should hold datetime objects.
    assert isinstance(out[0, 0], datetime.datetime)
