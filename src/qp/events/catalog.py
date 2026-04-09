"""Event and wave packet data structures.

Modernized from cassinilib/Event.py and cassinilib/Wave.py.
Uses dataclasses instead of mutable-default-arg classes.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field


@dataclass
class WaveEvent:
    """A detected quasi-periodic wave event in the magnetometer data.

    Replaces the old Event class from cassinilib/Event.py.

    The legacy fields (``period``, ``amplitude``, ``snr``, ...) are
    preserved so existing tests and figures continue to work. The new
    fields added in Phase 1 (``band``, ``period_peak_min``,
    ``rms_amplitude_perp``, ``region``, polarization metadata, ...) are
    populated by the multi-band sweep in :mod:`qp.events.detector` and
    :mod:`qp.events.threshold`.
    """

    date_from: datetime.datetime
    date_to: datetime.datetime

    # ------------------------------------------------------------------
    # Legacy fields (kept for back-compat)
    # ------------------------------------------------------------------
    period: float | None = None  # dominant period in seconds
    period_std: float | None = None  # period uncertainty
    amplitude: float | None = None  # peak amplitude in nT
    amplitude_std: float | None = None
    snr: float | None = None  # signal-to-noise ratio (or prominence)
    local_time: float | None = None  # hours
    mag_lat: float | None = None  # magnetic latitude (degrees)
    r_distance: float | None = None  # radial distance (R_S)
    l_shell: float | None = None
    coord_ksm: tuple[float, float, float] | None = None
    coord_krtp: tuple[float, float, float] | None = None
    closed_field_line: bool | None = None
    comment: str = ""

    # ------------------------------------------------------------------
    # Phase 1+ fields
    # ------------------------------------------------------------------
    band: str | None = None  # 'QP30', 'QP60', or 'QP120'
    period_peak_min: float | None = None  # CWT-ridge centroid period (min)
    period_fwhm_min: float | None = None  # period FWHM (min)
    rms_amplitude_perp: float | None = None  # RMS of sqrt(perp1^2+perp2^2)
    b_perp1_amp: float | None = None  # peak |b_perp1| in window (nT)
    b_perp2_amp: float | None = None  # peak |b_perp2| in window (nT)
    b_par_amp: float | None = None  # peak |b_par| in window (nT)
    region: str | None = None  # 'magnetosphere', 'magnetosheath', 'solar_wind', 'unknown'
    polarization: str | None = None  # 'circular', 'linear', or 'mixed'
    phase_deg: float | None = None  # b_perp1<->b_perp2 phase shift in deg
    ppo_phase_n_deg: float | None = None  # SLS5N at peak time
    ppo_phase_s_deg: float | None = None  # SLS5S at peak time
    event_id: str | None = None  # globally unique id (segment#-band-idx)
    segment_id: int | None = None  # source segment index
    n_oscillations: int | None = None  # duration / period

    @property
    def duration_hours(self) -> float:
        """Event duration in hours."""
        return (self.date_to - self.date_from).total_seconds() / 3600

    @property
    def duration_minutes(self) -> float:
        """Event duration in minutes."""
        return (self.date_to - self.date_from).total_seconds() / 60

    @property
    def period_minutes(self) -> float | None:
        """Period in minutes."""
        return self.period / 60 if self.period else None

    def is_significant(self, snr_threshold: float = 3.0) -> bool:
        """Check if event exceeds SNR threshold."""
        if self.snr is None:
            return False
        return self.snr > snr_threshold


@dataclass
class WavePacketPeak:
    """A single peak in the wavelet power, representing one wave packet.

    Used for computing wave train separations (Fig 9).
    """

    peak_time: datetime.datetime
    prominence: float  # peak prominence in normalized CWT power
    date_from: datetime.datetime  # wave packet start (left inflection)
    date_to: datetime.datetime  # wave packet end (right inflection)
    local_time: float | None = None
    r_distance: float | None = None
    theta: float | None = None  # colatitude in radians
    sls5_phase: list[float] = field(default_factory=list)
    total_gaps_hours: float = 0.0
    continuous: bool = True  # no data gap before this packet
    band: str | None = None  # which QP band fired (Phase 1+)
    period_sec: float | None = None  # peak period from CWT ridge

    @property
    def duration_hours(self) -> float:
        return (self.date_to - self.date_from).total_seconds() / 3600


@dataclass
class WaveTemplate:
    """Synthetic wave parameters for test signal generation.

    Replaces cassinilib/Wave.py.
    """

    period: float = 3600.0  # seconds
    amplitude: float = 1.0  # nT
    phase: float = 0.0  # radians
    waveform: str = "sine"  # 'sine', 'sawtooth', 'square'
    decay_width: float | None = None  # Gaussian envelope width (seconds)
    shift: float = 0.0  # time offset (seconds)
    cutoff: tuple[float, float] | None = None  # (start, end) in seconds
