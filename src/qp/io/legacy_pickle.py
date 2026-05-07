"""Pickle-stub registration for legacy ``DataProducts/*.npy`` arrays.

The 36-hour MAG segment arrays under
``DATA/CASSINI-DATA/DataProducts/Cassini_MAG_*_36H.npy`` were pickled in
2020-era code paths whose module names no longer exist. Numpy's
``np.load(allow_pickle=True)`` walks the pickle stream, sees references
like ``data_sweeper.NewSignal`` or ``cassinilib.SignalSnapshot``, and
fails with ``ModuleNotFoundError`` unless those module paths resolve to
*something* with the right class names.

:func:`register_stubs` injects empty placeholder modules and classes
into :data:`sys.modules` so the legacy pickles deserialize as opaque
objects whose ``__dict__`` (the COORDS / TIME / FIELDS arrays) is what
the modern code path actually consumes. Call this once at the top of
any script that loads the legacy ``.npy`` files.

DO NOT change the stub names: each one matches a module path that
existed when the pickles were written. Removing one silently breaks
``np.load()`` of those arrays.
"""

from __future__ import annotations

import sys
import types

_STUB_CLASSES: tuple[str, ...] = (
    "SignalSnapshot",
    "NewSignal",
    "Interval",
    "FFT_list",
    "WaveSignal",
    "Wave",
)

_STUB_MODULES: tuple[str, ...] = (
    "__main__",
    "data_sweeper",
    "mag_fft_sweeper",
    "cassinilib",
    "cassinilib.NewSignal",
)


def register_stubs() -> None:
    """Register placeholder modules/classes so legacy MFA pickles load."""
    for mod_path in _STUB_MODULES:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)
        mod = sys.modules[mod_path]
        for cls in _STUB_CLASSES:
            if not hasattr(mod, cls):
                setattr(mod, cls, type(cls, (), {}))
