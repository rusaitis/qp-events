r"""Canonical QP period bands.

The Rusaitis et al. paper identifies three quasi-periodic peaks in
Saturn's outer magnetosphere:

================  ==============  ===============================
Band              Period (min)    Field-line resonance harmonic
================  ==============  ===============================
QP30              ~30             m=6 (6th even)
QP60              ~60             m=4 (4th even)
QP120             ~120            m=2 (2nd even)
================  ==============  ===============================

Band edges are *symmetric in log-period* about the nominal centroid
and were chosen to give clean separation between bands while still
catching the natural spread observed in the FFT power-ratio plots
(Figs 4-5 of the paper).

The :data:`SEARCH_BAND_EXTENDED` covers the diagnostic 5 min - 12 h
range. The :data:`REJECT_BAND_HF` and :data:`REJECT_BAND_LF` constants
mark the Nyquist guard (10 min for 1-min data) and the Welch-window
upper bound (12 h) respectively — used by the threshold module to
mask edge bins of the wavelet scalogram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


_MIN = 60.0  # seconds in a minute


@dataclass(frozen=True, slots=True)
class Band:
    r"""A period band with name, edges (seconds), and a centroid.

    Attributes
    ----------
    name : str
        Short label, e.g. ``"QP60"``.
    period_min_sec : float
        Lower edge in seconds.
    period_max_sec : float
        Upper edge in seconds.
    period_centroid_sec : float
        Nominal centroid in seconds (geometric mean unless overridden).
    """

    name: str
    period_min_sec: float
    period_max_sec: float
    period_centroid_sec: float

    @property
    def freq_min_hz(self) -> float:
        return 1.0 / self.period_max_sec

    @property
    def freq_max_hz(self) -> float:
        return 1.0 / self.period_min_sec

    @property
    def freq_centroid_hz(self) -> float:
        return 1.0 / self.period_centroid_sec

    @property
    def period_min_minutes(self) -> float:
        return self.period_min_sec / _MIN

    @property
    def period_max_minutes(self) -> float:
        return self.period_max_sec / _MIN

    @property
    def period_centroid_minutes(self) -> float:
        return self.period_centroid_sec / _MIN


# ----------------------------------------------------------------------
# Canonical QP detection bands
# ----------------------------------------------------------------------

QP_BANDS: dict[str, Band] = {
    "QP30": Band(
        name="QP30",
        period_min_sec=20 * _MIN,
        period_max_sec=40 * _MIN,
        period_centroid_sec=30 * _MIN,
    ),
    "QP60": Band(
        name="QP60",
        period_min_sec=45 * _MIN,
        period_max_sec=80 * _MIN,
        period_centroid_sec=60 * _MIN,
    ),
    "QP120": Band(
        name="QP120",
        period_min_sec=90 * _MIN,
        period_max_sec=150 * _MIN,
        period_centroid_sec=120 * _MIN,
    ),
}

QP_BAND_NAMES: tuple[str, ...] = tuple(QP_BANDS.keys())


# ----------------------------------------------------------------------
# Diagnostic / guard bands
# ----------------------------------------------------------------------

#: Wide search band for diagnostic CWT plots and the wavelet noise model.
SEARCH_BAND_EXTENDED: Band = Band(
    name="SEARCH",
    period_min_sec=5 * _MIN,
    period_max_sec=12 * 3600.0,
    period_centroid_sec=60 * _MIN,
)

#: High-frequency rejection band: anything below 10 min is < 5x Nyquist
#: at 1-min cadence and is dominated by quantization / aliasing.
REJECT_BAND_HF: Band = Band(
    name="REJECT_HF",
    period_min_sec=0.0,
    period_max_sec=10 * _MIN,
    period_centroid_sec=5 * _MIN,
)

#: Low-frequency rejection band: anything above 12 h exceeds the
#: Welch window length used in the paper (12 h with 6 h overlap).
REJECT_BAND_LF: Band = Band(
    name="REJECT_LF",
    period_min_sec=12 * 3600.0,
    period_max_sec=float("inf"),
    period_centroid_sec=24 * 3600.0,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def freq_to_period(freq: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert frequency array to period, returning inf where freq is zero."""
    return np.where(freq > 0, 1.0 / freq, np.inf)


def get_band(band: str | Band) -> Band:
    r"""Resolve a band identifier to a :class:`Band`.

    Parameters
    ----------
    band : str or Band
        Either a key into :data:`QP_BANDS` (case-insensitive) or an
        already-constructed :class:`Band` instance.

    Returns
    -------
    Band
    """
    if isinstance(band, Band):
        return band
    key = band.upper()
    if key not in QP_BANDS:
        raise KeyError(
            f"Unknown band {band!r}. Known: {sorted(QP_BANDS)}"
        )
    return QP_BANDS[key]


def is_in_band(period_sec: float, band: str | Band) -> bool:
    r"""Check whether a period (in seconds) lies inside a band.

    Half-open ``[min, max)``.
    """
    b = get_band(band)
    return b.period_min_sec <= period_sec < b.period_max_sec


def is_rejected(period_sec: float) -> bool:
    r"""Return True if the period falls in either rejection guard band."""
    return (
        period_sec < REJECT_BAND_HF.period_max_sec
        or period_sec >= REJECT_BAND_LF.period_min_sec
    )


def classify_period(period_sec: float) -> str | None:
    r"""Return the name of the QP band containing this period, or ``None``.

    Used by the ridge extractor and post-hoc band labelling. Returns
    ``None`` if the period falls outside all QP bands or in a rejection
    band.
    """
    if is_rejected(period_sec):
        return None
    for name, band in QP_BANDS.items():
        if band.period_min_sec <= period_sec < band.period_max_sec:
            return name
    return None
