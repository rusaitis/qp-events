"""Load pre-computed DataProducts .npy files.

These are large numpy arrays produced by data_sweeper.py and
boundary_crossings.py from raw PDS data. For most figure reproduction,
start here rather than reprocessing from raw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import qp


def _load(name: str, products_dir: Path | None = None) -> np.ndarray:
    """Load a .npy file from DataProducts with allow_pickle=True."""
    d = products_dir or qp.DATA_PRODUCTS
    return np.load(d / name, allow_pickle=True)


def load_spacecraft_position(products_dir: Path | None = None) -> np.ndarray:
    """Load Cassini position time series in KSM coordinates.

    Returns shape (N, 11) object array. Columns:
        0: datetime
        1-4: Bx, By, Bz, Btot (nT)  [or field components]
        5-7: x, y, z (R_S) in KSM
        8-10: location flags
    """
    return _load("CassiniLocation_KSM.npy", products_dir)
