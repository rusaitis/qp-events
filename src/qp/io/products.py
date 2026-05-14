"""Load pre-computed DataProducts .npy files.

These are large numpy arrays produced by data_sweeper.py and
boundary_crossings.py from raw PDS data. For most figure reproduction,
start here rather than reprocessing from raw.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

import qp

#: Module paths the legacy DataProducts/*.npy pickles reference. They no
#: longer exist in the tree (removed in the cleanup commit); stub modules
#: are registered so ``numpy.load(..., allow_pickle=True)`` resolves the
#: class references without dragging the old codebase back.
_LEGACY_PICKLE_MODULES: tuple[str, ...] = (
    "__main__",
    "data_sweeper",
    "mag_fft_sweeper",
    "cassinilib",
    "cassinilib.NewSignal",
    "cassinilib.PlotFFT",
)

#: Class names referenced by the legacy pickles. Registered as empty
#: types on each stub module above — unpickling only needs the symbol to
#: resolve, the array data itself is recovered as a numpy ndarray.
_LEGACY_PICKLE_CLASSES: tuple[str, ...] = (
    "SignalSnapshot",
    "NewSignal",
    "Interval",
    "FFT_list",
    "WaveSignal",
    "Wave",
)


def register_legacy_pickle_stubs() -> None:
    """Register empty stub modules + classes for legacy ``.npy`` pickles.

    Must be called **before** ``np.load(..., allow_pickle=True)`` on any
    of the ``DataProducts/Cassini_MAG_*.npy`` arrays — otherwise unpickling
    raises ``ModuleNotFoundError`` for ``data_sweeper`` / ``cassinilib``.
    Idempotent.
    """
    for mod_path in _LEGACY_PICKLE_MODULES:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)
        mod = sys.modules[mod_path]
        for cls_name in _LEGACY_PICKLE_CLASSES:
            if not hasattr(mod, cls_name):
                setattr(mod, cls_name, type(cls_name, (), {}))


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
