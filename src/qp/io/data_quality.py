r"""Cassini MAG instrument data quality flags.

Parses the five data quality files from ``DATA/CASSINI-Data-Quality/``
into a unified table of time intervals to flag or exclude from QP wave
analysis.

Files and their impact on 1-min averaged MAG data:

===============================  ========  ==========================================
File                             Priority  Impact
===============================  ========  ==========================================
RANGE_CHANGES.ASC                Critical  Each range change spikes 1–2 min bins
MODE_CHANGES.ASC                 Critical  Data gaps / corrupted intervals
SCAS_TIMES.ASC                   High      Calibration maneuvers corrupt B-field
SPURIOUS_RANGE_CHANGES.ASC       Low       Single corrupted 1-sec vectors
FGM_FULL_TIMING_ERRS.CSV         Negligible Sub-second timing metadata
===============================  ========  ==========================================
"""

from __future__ import annotations

import csv
import datetime
import re
from dataclasses import dataclass
from pathlib import Path

from qp import DATA_ROOT

_QUALITY_DIR = DATA_ROOT / "CASSINI-Data-Quality"


@dataclass(frozen=True, slots=True)
class QualityFlag:
    """One flagged time interval."""

    start: datetime.datetime
    end: datetime.datetime
    flag_type: str  # timing_error, range_change, mode_change, etc.
    sensor: str  # FGM, VHM, ALL
    severity: str  # critical, high, low, negligible
    description: str


# ------------------------------------------------------------------
# Datetime parsing helpers
# ------------------------------------------------------------------

# DOY formats: "2007-213T05:16:11.386", "2007-213 05:16:11", "2007-213T05:16:11"
_DOY_RE = re.compile(
    r"(\d{4})-(\d{3})"       # year-doy
    r"[T ]"                   # separator
    r"(\d{2}):(\d{2}):(\d{2})"  # hh:mm:ss
    r"(?:\.(\d+))?"           # optional fractional seconds
)


def _parse_doy(s: str) -> datetime.datetime | None:
    """Parse a DOY timestamp string, return None on failure."""
    m = _DOY_RE.search(s)
    if m is None:
        return None
    year, doy = int(m.group(1)), int(m.group(2))
    h, mi, sec = int(m.group(3)), int(m.group(4)), int(m.group(5))
    frac = int(m.group(6).ljust(6, "0")[:6]) if m.group(6) else 0
    base = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
    return base.replace(hour=h, minute=mi, second=sec, microsecond=frac)


def _parse_iso(s: str) -> datetime.datetime | None:
    """Parse an ISO timestamp string, return None on failure."""
    m = re.search(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?", s,
    )
    if m is None:
        return None
    try:
        return datetime.datetime.fromisoformat(m.group(0))
    except ValueError:
        return None


# ------------------------------------------------------------------
# Per-file parsers
# ------------------------------------------------------------------


def parse_timing_errors(path: Path | None = None) -> list[QualityFlag]:
    r"""Parse ``FGM_FULL_TIMING_ERRS.CSV``.

    Format: ``filepath,start_iso,end_iso,description``
    """
    if path is None:
        path = _QUALITY_DIR / "FGM_FULL_TIMING_ERRS.CSV"
    flags: list[QualityFlag] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        start = _parse_iso(parts[1])
        end = _parse_iso(parts[2])
        if start is None or end is None:
            continue
        flags.append(QualityFlag(
            start=start, end=end,
            flag_type="timing_error", sensor="FGM",
            severity="negligible",
            description=f"backwards timing: {parts[3].strip()}",
        ))
    return flags


def parse_range_changes(path: Path | None = None) -> list[QualityFlag]:
    r"""Parse ``RANGE_CHANGES.ASC``.

    Most lines: ``YYYY-DOYTHH:MM:SS.sss,  FGM    FGM changed to range N``
    Range changes cause 1–2 min spikes in averaged data.
    """
    if path is None:
        path = _QUALITY_DIR / "RANGE_CHANGES.ASC"
    flags: list[QualityFlag] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("Each", "Time", "---")):
            continue
        ts = _parse_doy(line)
        if ts is None:
            continue
        # Determine sensor
        upper = line.upper()
        if "VHM" in upper and "FGM" in upper:
            sensor = "ALL"
        elif "VHM" in upper:
            sensor = "VHM"
        else:
            sensor = "FGM"
        # Extract range number
        range_match = re.search(r"range\s+(\d)", line, re.IGNORECASE)
        range_num = range_match.group(1) if range_match else "?"
        # Point event → flag ±60s (one 1-min bin)
        flags.append(QualityFlag(
            start=ts,
            end=ts + datetime.timedelta(seconds=60),
            flag_type="range_change", sensor=sensor,
            severity="critical",
            description=f"{sensor} to range {range_num}",
        ))
    return flags


def parse_mode_changes(path: Path | None = None) -> list[QualityFlag]:
    r"""Parse ``MODE_CHANGES.ASC``.

    Free-form text. Each entry starts with a DOY timestamp.
    We extract the timestamp and classify the event from keywords.
    """
    if path is None:
        path = _QUALITY_DIR / "MODE_CHANGES.ASC"
    flags: list[QualityFlag] = []
    text = path.read_text()
    # Split into blocks: each block starts with a timestamp at line start
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _DOY_RE.match(stripped):
            if current:
                blocks.append("\n".join(current))
            current = [stripped]
        elif current:
            current.append(stripped)
    if current:
        blocks.append("\n".join(current))

    for block in blocks:
        lines = block.split("\n")
        ts = _parse_doy(lines[0])
        if ts is None:
            continue
        full_text = " ".join(ln.strip() for ln in lines).lower()

        # Determine sensor
        first_line = lines[0]
        upper_first = first_line.upper()
        if "ALL DATA" in upper_first or "ALL" in upper_first:
            sensor = "ALL"
        elif "VHM" in upper_first and "SHM" in upper_first:
            sensor = "VHM"
        elif "VHM" in upper_first:
            sensor = "VHM"
        elif "FGM" in upper_first:
            sensor = "FGM"
        else:
            sensor = "ALL"

        # Classify severity and estimate duration
        if any(kw in full_text for kw in ("turns off", "turns on", "sick",
                                           "sleep", "muted", "unmuted",
                                           "reset", "no data", "no science")):
            severity = "critical"
            # Check if there's a "to" date for an interval
            to_match = re.search(r"to\s+(\d{4}-\d{3})", block)
            if to_match:
                end = _parse_doy(to_match.group(0).replace("to ", ""))
                if end and end > ts:
                    flags.append(QualityFlag(
                        start=ts, end=end,
                        flag_type="mode_change", sensor=sensor,
                        severity=severity,
                        description=_mode_desc(full_text),
                    ))
                    continue
            # Point event → flag 10 min (conservative for instrument settling)
            flags.append(QualityFlag(
                start=ts,
                end=ts + datetime.timedelta(minutes=10),
                flag_type="mode_change", sensor=sensor,
                severity=severity,
                description=_mode_desc(full_text),
            ))
        elif "scalar/vector" in full_text or "vector/vector" in full_text:
            # VHM/SHM mode switch — brief transient
            flags.append(QualityFlag(
                start=ts,
                end=ts + datetime.timedelta(minutes=2),
                flag_type="mode_change", sensor=sensor,
                severity="high",
                description="mode switch: " + (
                    "scalar/vector" if "scalar/vector" in full_text
                    else "vector/vector"
                ),
            ))
        else:
            # Other mode change
            flags.append(QualityFlag(
                start=ts,
                end=ts + datetime.timedelta(minutes=5),
                flag_type="mode_change", sensor=sensor,
                severity="high",
                description=_mode_desc(full_text),
            ))

    return flags


def _mode_desc(text: str) -> str:
    """Extract a short description from mode change text."""
    for kw, desc in [
        ("turns off", "instrument off"),
        ("turns on", "instrument on"),
        ("sick", "instrument sick"),
        ("sleep", "sleep mode"),
        ("muted", "instrument muted"),
        ("unmuted", "instrument unmuted"),
        ("reset", "instrument reset"),
        ("no data", "no data"),
        ("scalar/vector", "VHM→SHM switch"),
        ("vector/vector", "SHM→VHM switch"),
        ("science packets recommence", "science data resumes"),
        ("maintenance", "instrument maintenance"),
    ]:
        if kw in text:
            return desc
    return "mode change"


def parse_scas_times(path: Path | None = None) -> list[QualityFlag]:
    r"""Parse ``SCAS_TIMES.ASC``.

    Format: timestamp + description lines. Calibrations come in
    commences/ceases pairs.
    """
    if path is None:
        path = _QUALITY_DIR / "SCAS_TIMES.ASC"
    flags: list[QualityFlag] = []
    text = path.read_text()

    # Extract all timestamped events
    events: list[tuple[datetime.datetime, str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        ts = _parse_doy(stripped)
        if ts is None:
            continue
        upper = stripped.upper()
        if "ALL DATA" in upper or "ALL" in upper:
            sensor = "ALL"
        elif "VHM" in upper:
            sensor = "VHM"
        elif "FGM" in upper:
            sensor = "FGM"
        else:
            sensor = "ALL"
        events.append((ts, sensor, stripped.lower()))

    # Pair commences/ceases into intervals
    i = 0
    while i < len(events):
        ts, sensor, text_lower = events[i]
        if "commences" in text_lower:
            # Look for matching "ceases"
            if i + 1 < len(events) and "ceases" in events[i + 1][2]:
                end_ts = events[i + 1][0]
                desc = "SCAS activity" if "scas" in text_lower else "calibration"
                flags.append(QualityFlag(
                    start=ts, end=end_ts,
                    flag_type="calibration", sensor=sensor,
                    severity="high",
                    description=desc,
                ))
                i += 2
                continue
            # No matching ceases — flag 1 hour
            flags.append(QualityFlag(
                start=ts,
                end=ts + datetime.timedelta(hours=1),
                flag_type="calibration", sensor=sensor,
                severity="high",
                description="calibration (no end time)",
            ))
        elif "ceases" in text_lower:
            # Orphaned ceases — skip
            pass
        else:
            # Other event
            flags.append(QualityFlag(
                start=ts,
                end=ts + datetime.timedelta(minutes=5),
                flag_type="calibration", sensor=sensor,
                severity="high",
                description="SCAS event",
            ))
        i += 1

    return flags


def parse_spurious_range_changes(path: Path | None = None) -> list[QualityFlag]:
    r"""Parse ``SPURIOUS_RANGE_CHANGES.ASC``.

    Timestamps of single corrupted data vectors.
    """
    if path is None:
        path = _QUALITY_DIR / "SPURIOUS_RANGE_CHANGES.ASC"
    flags: list[QualityFlag] = []
    for line in path.read_text().splitlines():
        ts = _parse_doy(line.strip())
        if ts is None:
            continue
        flags.append(QualityFlag(
            start=ts,
            end=ts + datetime.timedelta(seconds=60),
            flag_type="spurious_range", sensor="FGM",
            severity="low",
            description="corrupted range vector",
        ))
    return flags


# ------------------------------------------------------------------
# Combined loader
# ------------------------------------------------------------------

_FLAG_CSV_FIELDS = [
    "start_time", "end_time", "flag_type", "sensor", "severity", "description",
]


def load_all_quality_flags(
    quality_dir: Path | None = None,
    min_year: int = 2004,
    max_year: int = 2017,
) -> list[QualityFlag]:
    """Load and merge all quality flags, filtered to the science window."""
    if quality_dir is not None:
        # Override the default directory for all parsers
        d = quality_dir
        parsers = [
            parse_timing_errors(d / "FGM_FULL_TIMING_ERRS.CSV"),
            parse_range_changes(d / "RANGE_CHANGES.ASC"),
            parse_mode_changes(d / "MODE_CHANGES.ASC"),
            parse_scas_times(d / "SCAS_TIMES.ASC"),
            parse_spurious_range_changes(d / "SPURIOUS_RANGE_CHANGES.ASC"),
        ]
        all_flags = [f for group in parsers for f in group]
    else:
        all_flags = (
            parse_timing_errors()
            + parse_range_changes()
            + parse_mode_changes()
            + parse_scas_times()
            + parse_spurious_range_changes()
        )

    # Filter to science window
    t_min = datetime.datetime(min_year, 1, 1)
    t_max = datetime.datetime(max_year + 1, 1, 1)
    filtered = [
        f for f in all_flags
        if f.end >= t_min and f.start < t_max
    ]
    filtered.sort(key=lambda f: f.start)
    return filtered


def flags_to_csv(flags: list[QualityFlag], path: Path) -> None:
    """Write quality flags to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FLAG_CSV_FIELDS)
        writer.writeheader()
        for flag in flags:
            writer.writerow({
                "start_time": flag.start.isoformat(),
                "end_time": flag.end.isoformat(),
                "flag_type": flag.flag_type,
                "sensor": flag.sensor,
                "severity": flag.severity,
                "description": flag.description,
            })


def flags_from_csv(path: Path) -> list[QualityFlag]:
    """Read quality flags from a CSV file."""
    flags: list[QualityFlag] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flags.append(QualityFlag(
                start=datetime.datetime.fromisoformat(row["start_time"]),
                end=datetime.datetime.fromisoformat(row["end_time"]),
                flag_type=row["flag_type"],
                sensor=row["sensor"],
                severity=row["severity"],
                description=row["description"],
            ))
    return flags
