r"""Tabular persistence for round-8 wave-event detections.

Provides a flat row schema (one row per :class:`DetectedEvent`) and
parquet I/O. The output of a full-mission sweep is one parquet file
per run — typically ``Output/events_round8.parquet`` — paired with a
side-car ``.meta.json`` that records the detector configuration, band
edges, and source-data hashes so a reader can verify the table was
produced by the expected pipeline.

The required columns mirror the detector's output one-to-one. Each
row carries the four gate values (``q_factor``, ``mva_par_frac``,
``stokes_d``, plus the bandpass amplitudes) so a downstream analyst
can reproduce the gate decision from the table without rerunning the
detector.

Schema
------
Required (always populated):

- ``event_id`` int          — globally unique event id within the run
- ``segment_id`` str        — id of the 36-h MFA segment the event came from
- ``date_from``, ``date_to`` ISO str — wave-packet window
- ``peak_time`` ISO str     — time of maximum CWT power
- ``duration_minutes`` float — ``date_to - date_from``
- ``band`` str              — QP30 / QP60 / QP120
- ``period_min`` float      — parabolic-interpolated peak period, minutes
- ``period_fwhm_min`` float — peak FWHM, minutes
- ``q_factor`` float        — period / FWHM
- ``mva_par_frac`` float    — :math:`(\hat e_1 \cdot \hat b_\parallel)^2`
- ``stokes_d`` float        — degree of polarization in [0, 1]
- ``stokes_i``, ``stokes_q``, ``stokes_u``, ``stokes_v`` float
                            — full Stokes vector over the in-band TF window
                              (units: nT² of CWT amplitude); ``stokes_d``
                              equals ``sqrt(Q^2+U^2+V^2)/I``
- ``ellipticity`` float     — signed minor/major axis ratio in [-1, 1]
                              (Samson 1973 convention: positive ⇔ V > 0
                              ⇔ b_perp2 lags b_perp1 by +π/2)
- ``inclination_deg`` float — major-axis tilt from :math:`\hat b_{\perp 1}`
                              in degrees, :math:`\tfrac{1}{2}\,\mathrm{atan2}(U,Q)`
- ``polarized_fraction`` float — :math:`p/I \in [0, 1]`, identical to
                                  ``stokes_d``; kept as a distinct column
                                  so consumers don't have to recompute
- ``b_perp1_amp`` float     — RMS of bandpass :math:`b_{\perp 1}`, nT
- ``b_perp2_amp`` float     — RMS of bandpass :math:`b_{\perp 2}`, nT
- ``b_par_amp`` float       — RMS of bandpass :math:`b_\parallel`, nT

Optional (populated when ephemeris/region info is available):

- ``r_distance`` float       — radial distance, R_S
- ``mag_lat`` float          — magnetic latitude, deg
- ``local_time`` float       — magnetic local time, h
- ``ksm_x``, ``ksm_y``, ``ksm_z`` float — KSM position at peak, R_S
- ``region`` str             — magnetosphere / magnetosheath / solar_wind / unknown
- ``sls5_phase_n`` float     — northern SLS5 phase at peak, deg
- ``sls5_phase_s`` float     — southern SLS5 phase at peak, deg
- ``dipole_inv_lat`` float   — analytic dipole invariant latitude, deg

Optional (populated by the peer-tagging post-pass,
:func:`qp.events.peers.tag_peers`; see that module for the overlap
criterion). All three columns are list-typed and aligned by position
— empty lists when the row has no peers, never NaN:

- ``peer_event_ids``   list[int64]   peer ``event_id`` values
- ``peer_periods_min`` list[float64] peer ``period_min`` values, minutes
- ``peer_overlap_frac`` list[float64] overlap fraction in [τ, 1]

Band labels for cobands are *not* stored — they are derived at view
time from ``peer_periods_min`` via :func:`qp.events.peers.derive_co_bands`,
so the catalogue is band-scheme-agnostic.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from qp.events.detector import DetectedEvent

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)


# Required columns — every row must have these populated.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_id",
    "segment_id",
    "date_from",
    "date_to",
    "peak_time",
    "duration_minutes",
    "band",
    "period_min",
    "period_fwhm_min",
    "q_factor",
    "mva_par_frac",
    "stokes_d",
    "stokes_i",
    "stokes_q",
    "stokes_u",
    "stokes_v",
    "ellipticity",
    "inclination_deg",
    "polarized_fraction",
    "b_perp1_amp",
    "b_perp2_amp",
    "b_par_amp",
)

#: Bumped when new required columns are added or the optional-column
#: contract changes. Round-8.1 introduced the full Stokes vector and
#: derived geometry. Round-8.2 adds the optional ``peer_event_ids /
#: peer_periods_min / peer_overlap_frac`` triple from
#: :func:`qp.events.peers.tag_peers`, replacing the band-labelled
#: ``co_bands`` string. Legacy parquet files written with earlier
#: schemas remain readable; callers detect the upgrade via the
#: side-car ``meta.json`` ``schema_version`` key.
SCHEMA_VERSION: str = "round8.2"


def event_to_record(
    detection: DetectedEvent,
    *,
    event_id: int,
    segment_id: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten a :class:`DetectedEvent` into a parquet-ready dict.

    The four gate values are required and raise if missing. Any
    additional ephemeris (``r_distance``, ``mag_lat``, ``local_time``,
    ``ksm_x/y/z``, ``region``, SLS5 phase, dipole inv-lat) should be
    passed via ``extra``; their column types are inferred by parquet.

    Parameters
    ----------
    detection : DetectedEvent
        Output of :func:`qp.events.detector.detect_round8`.
    event_id : int
        Run-unique event id. Caller is responsible for monotonic
        assignment.
    segment_id : str
        Identifier of the source MFA segment (e.g. its zarr path stem).
    extra : mapping, optional
        Additional columns to merge in. Keys may not collide with the
        required columns.
    """
    peak = detection.peak
    if peak.period_sec is None or peak.period_fwhm_sec is None:
        raise ValueError(
            "DetectedEvent.peak must have period_sec and period_fwhm_sec set"
        )

    duration_sec = (peak.date_to - peak.date_from).total_seconds()
    record: dict[str, Any] = {
        "event_id": int(event_id),
        "segment_id": str(segment_id),
        "date_from": peak.date_from.isoformat(),
        "date_to": peak.date_to.isoformat(),
        "peak_time": peak.peak_time.isoformat(),
        "duration_minutes": duration_sec / 60.0,
        "band": peak.band,
        "period_min": peak.period_sec / 60.0,
        "period_fwhm_min": peak.period_fwhm_sec / 60.0,
        "q_factor": float(detection.q_factor),
        "mva_par_frac": float(detection.mva_par_frac),
        "stokes_d": float(detection.stokes_d),
        "stokes_i": float(detection.stokes_i),
        "stokes_q": float(detection.stokes_q),
        "stokes_u": float(detection.stokes_u),
        "stokes_v": float(detection.stokes_v),
        "ellipticity": float(detection.ellipticity),
        "inclination_deg": float(detection.inclination_deg),
        "polarized_fraction": float(detection.polarized_fraction),
        "b_perp1_amp": float(detection.b_perp1_amp),
        "b_perp2_amp": float(detection.b_perp2_amp),
        "b_par_amp": float(detection.b_par_amp),
    }

    if extra is not None:
        collisions = set(extra) & set(REQUIRED_COLUMNS)
        if collisions:
            raise ValueError(
                f"extra fields collide with required columns: {sorted(collisions)}"
            )
        record.update(extra)

    return record


def events_to_parquet(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    attrs: Mapping[str, Any] | None = None,
) -> int:
    """Write event records to parquet with a side-car JSON metadata file.

    Returns the number of rows written. Side-car path is
    ``<path>.meta.json`` and contains ``attrs`` plus the column order.

    Falls back to ``.npy`` (a structured numpy array) only if
    ``pyarrow`` is unavailable, in which case the on-disk file is
    ``<path-with-.parquet-stripped>.npy`` to make the substitution
    obvious. Tests should pin pyarrow to avoid this branch.
    """
    import pandas as pd

    rows = list(records)
    if not rows:
        log.warning("events_to_parquet: no records to write to %s", path)

    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if rows and missing:
        raise ValueError(f"missing required columns: {missing}")
    attrs = {**(attrs or {})}
    attrs.setdefault("schema_version", SCHEMA_VERSION)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, engine="pyarrow", index=False)
    except (ImportError, ValueError) as exc:
        log.warning(
            "pyarrow unavailable (%s); falling back to .npy",
            exc,
        )
        npy_path = path.with_suffix(".npy")
        np.save(npy_path, df.to_records(index=False))
        path = npy_path

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "n_rows": len(rows),
                "columns": list(df.columns),
                "attrs": dict(attrs),
            },
            indent=2,
            default=str,
        )
    )
    return len(rows)


def read_events_parquet(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Round-trip companion to :func:`events_to_parquet`.

    Returns ``(dataframe, attrs)``. ``attrs`` is the side-car JSON's
    ``attrs`` field, or ``{}`` if no side-car exists.
    """
    import pandas as pd

    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        df = pd.DataFrame.from_records(arr)
    else:
        df = pd.read_parquet(path)

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    attrs: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        attrs = meta.get("attrs", {})
    return df, attrs


def detection_to_dict(detection: DetectedEvent) -> dict[str, Any]:
    """Pure-Python view of a DetectedEvent (debugging / logging only)."""
    out = asdict(detection)
    # asdict turns the WavePacketPeak into a nested dict — flatten the
    # times for human-readable diagnostics.
    peak = out.pop("peak")
    out.update(
        {
            "peak_time": peak["peak_time"],
            "date_from": peak["date_from"],
            "date_to": peak["date_to"],
            "band": peak["band"],
            "period_sec": peak["period_sec"],
            "period_fwhm_sec": peak["period_fwhm_sec"],
        }
    )
    return out
