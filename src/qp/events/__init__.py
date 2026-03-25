"""Event detection, cataloging, and wave packet analysis."""

from qp.events.catalog import WaveEvent, WavePacketPeak, WaveTemplate
from qp.events.detector import detect_wave_packets, compute_event_measure
from qp.events.wave_packets import (
    compute_separations,
    separation_statistics,
    separation_histogram,
)
