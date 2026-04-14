"""Ground truth data structures for the synthetic benchmark suite.

Every injected event — whether a detectable QP wave or a decoy that
should be rejected — is recorded as an :class:`InjectedEvent`. A
:class:`DatasetManifest` groups events belonging to one synthetic
dataset together with its generation parameters.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field, fields as dc_fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InjectedEvent:
    """Ground truth metadata for one injected wave packet."""

    event_id: str
    dataset_id: str
    event_type: str  # "qp_wave", "decoy_compressional", "decoy_pulse", etc.
    should_detect: bool
    band: str | None  # "QP30", "QP60", "QP120", or None for decoys
    period_sec: float
    amplitude_nT: float
    start_sec: float
    end_sec: float
    center_sec: float
    duration_sec: float
    n_oscillations: float
    polarization: str  # "circular", "linear", "elliptical"
    ellipticity: float
    wave_mode: str  # "alfvenic", "compressional", "mixed"
    propagation: str  # "standing", "travelling"
    chirp_rate: float
    waveform: str  # "sine", "sawtooth", "square"
    sawtooth_width: float
    envelope_asymmetry: float
    amplitude_jitter: float
    harmonic_content: float
    snr_injected: float  # amplitude / broadband noise RMS
    snr_in_band: float  # amplitude / in-band noise RMS (analytic)
    start_2sigma_sec: float  # ±2σ boundary (95.4% energy)
    end_2sigma_sec: float
    difficulty: str  # "easy", "moderate", "hard", "extreme"
    # Empirical in-band SNR measured from the realised noise array.
    # NaN means not computed (older manifests or back-compat construction).
    snr_in_band_empirical: float = math.nan


@dataclass
class DatasetManifest:
    """Manifest for one benchmark dataset."""

    dataset_id: str
    description: str
    duration_days: float
    dt: float
    n_samples: int
    seed: int
    noise_alpha: float
    noise_sigma: float
    difficulty_tier: str  # "tier1", "tier2", "tier3", "tier4", "decoy"
    events: list[InjectedEvent] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def n_detectable(self) -> int:
        return sum(1 for e in self.events if e.should_detect)

    @property
    def n_decoys(self) -> int:
        return sum(1 for e in self.events if not e.should_detect)


# --- JSON I/O ---


def manifest_to_json(manifest: DatasetManifest, path: Path) -> None:
    """Write a DatasetManifest to JSON."""
    data = asdict(manifest)
    path.write_text(json.dumps(data, indent=2, default=str))


def manifest_from_json(path: Path) -> DatasetManifest:
    """Read a DatasetManifest from JSON."""
    raw = json.loads(path.read_text())
    events = [InjectedEvent(**e) for e in raw.pop("events", [])]
    return DatasetManifest(**raw, events=events)


# --- CSV I/O (flat table of all events across datasets) ---

_EVENT_FIELDS = [f.name for f in dc_fields(InjectedEvent)]


def events_to_csv(events: list[InjectedEvent], path: Path) -> None:
    """Write a flat CSV of InjectedEvents (one row per event)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EVENT_FIELDS)
        writer.writeheader()
        for e in events:
            writer.writerow(asdict(e))


def events_from_csv(path: Path) -> list[InjectedEvent]:
    """Read InjectedEvents from a CSV file."""
    events = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types from strings. Optional fields (those with
            # defaults on InjectedEvent) may be missing in older CSVs.
            typed: dict[str, object] = dict(row)
            typed["should_detect"] = (
                str(row["should_detect"]).lower() in ("true", "1")
            )
            for key in (
                "period_sec", "amplitude_nT", "start_sec", "end_sec",
                "center_sec", "duration_sec", "n_oscillations", "ellipticity",
                "chirp_rate", "sawtooth_width", "envelope_asymmetry",
                "amplitude_jitter", "harmonic_content", "snr_injected",
                "snr_in_band", "start_2sigma_sec", "end_2sigma_sec",
            ):
                typed[key] = float(row[key])
            if "snr_in_band_empirical" in row and row["snr_in_band_empirical"]:
                typed["snr_in_band_empirical"] = float(
                    row["snr_in_band_empirical"]
                )
            events.append(InjectedEvent(**typed))  # type: ignore[arg-type]
    return events
