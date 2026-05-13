"""Event detection, cataloging, and wave packet analysis."""

from qp.events.catalog import WaveEvent, WavePacketPeak, WaveTemplate
from qp.events.wave_packets import (
    compute_separations,
    separation_histogram,
    separation_statistics,
)

__all__ = [
    "WaveEvent",
    "WavePacketPeak",
    "WaveTemplate",
    "compute_separations",
    "separation_histogram",
    "separation_statistics",
]
