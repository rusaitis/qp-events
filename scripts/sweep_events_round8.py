"""Round-8 mission-wide QP wave-event sweep.

Walks the 36-hour MFA segments (``DataProducts/Cassini_MAG_MFA_36H.npy``),
runs the simplified four-gate detector
(:func:`qp.events.detector.detect_round8`), enriches each detection
with spacecraft ephemeris, region label, and SLS5 phase, and writes a
parquet event catalogue via :func:`qp.events.persistence.events_to_parquet`.

The output parquet has one row per detection with the four gate values
(``q_factor``, ``mva_par_frac``, ``stokes_d``, plus the bandpass-RMS
amplitudes) populated. A side-car ``<output>.meta.json`` records the
detector configuration and the source data hash.

Usage::

    uv run python scripts/sweep_events_round8.py
    uv run python scripts/sweep_events_round8.py --year 2007
    uv run python scripts/sweep_events_round8.py --serial      # debug
    uv run python scripts/sweep_events_round8.py -o Output/events_round8.parquet
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path  # noqa: F401  (used in argparse type and helpers)

import numpy as np
from _common import setup_logging  # noqa: E402

import qp  # noqa: E402
from qp.cli import add_tracing_args, add_verbosity_arg  # noqa: E402
from qp.io import register_legacy_pickle_stubs  # noqa: E402

register_legacy_pickle_stubs()
from qp.dwell.tracing import TracingConfig  # noqa: E402
from qp.events.detector import (  # noqa: E402
    MAX_MVA_PARALLEL_FRACTION,
    MIN_DEGREE_OF_POLARIZATION,
    MIN_Q_FACTOR,
    SEGMENT_FWER_ALPHA,
    DetectedEvent,
    detect_round8,
)
from qp.events.peak_kmag import J2000_POSIX, kmag_peak_columns  # noqa: E402
from qp.events.persistence import event_to_record, events_to_parquet  # noqa: E402
from qp.events.sweep_loader import (  # noqa: E402
    load_segments,
    ppo_at_peak_from_info,
    region_at_peak_from_info,
    segment_central_window,
    segment_to_payload,
)
from qp.events.threshold_diag import BGArchive, load_bg_archive  # noqa: E402
from qp.fieldline.kmag_model import SaturnField, SaturnFieldConfig  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class SegmentDetectionPayload:
    seg_idx: int
    times: list[datetime.datetime]
    b_par: np.ndarray
    b_perp1: np.ndarray
    b_perp2: np.ndarray
    coord_r: np.ndarray | None
    coord_th: np.ndarray | None
    coord_phi: np.ndarray | None
    info: dict
    region_midpoint: str


def _to_payload(seg_idx: int, seg) -> SegmentDetectionPayload | None:
    base = segment_to_payload(seg_idx, seg)
    if base is None:
        return None
    times = list(base.times)
    midpoint = times[len(times) // 2] if times else datetime.datetime(2000, 1, 1)
    info = dict(base.info)
    region = region_at_peak_from_info(info, midpoint)
    return SegmentDetectionPayload(
        seg_idx=base.seg_idx,
        times=times,
        b_par=base.b_par,
        b_perp1=base.b_perp1,
        b_perp2=base.b_perp2,
        coord_r=base.coord_r,
        coord_th=base.coord_th,
        coord_phi=base.coord_phi,
        info=info,
        region_midpoint=region,
    )


def _ksm_from_offset_spherical(
    r_off: float,
    mag_lat_rad: float,
    phi_rad: float,
) -> tuple[float, float, float]:
    """Reconstruct KSM (x, y, z) from segment offset-dipole spherical coords.

    The MFA-36h segments store ``(r, th, phi)`` in the offset-dipole
    frame: ``r`` is sqrt(x² + y² + (z-0.037)²), ``th`` is signed magnetic
    latitude (radians), ``phi`` is KSM longitude (radians; verified
    against the local_time formula). Inverting these gives KSM cartesian
    to within the 0.037 R_S dipole offset — well below the 0.15 R_S
    KMAG trace step at all spacecraft distances.
    """
    z = 0.037 + r_off * math.sin(mag_lat_rad)
    xy = r_off * math.cos(mag_lat_rad)
    return xy * math.cos(phi_rad), xy * math.sin(phi_rad), z


def _enrich_at_peak(
    detection: DetectedEvent,
    payload: SegmentDetectionPayload,
) -> dict:
    """Look up ephemeris / region / SLS5 / KMAG at the detection peak time."""
    peak_time = detection.peak.peak_time
    times = payload.times
    if not times:
        return {}
    seg_t0 = times[0]
    n_samples = len(times)
    elapsed_min = (peak_time - seg_t0).total_seconds() / 60.0
    idx = int(round(elapsed_min))
    idx = max(0, min(n_samples - 1, idx))

    extra: dict = {}
    r_off_val: float | None = None
    th_rad: float | None = None
    phi_rad: float | None = None
    if payload.coord_r is not None and idx < len(payload.coord_r):
        r_off_val = float(payload.coord_r[idx])
        extra["r_distance"] = r_off_val
    if payload.coord_th is not None and idx < len(payload.coord_th):
        # `coord_th` in MFA-36h segments is signed magnetic latitude in
        # radians (range ~ +/-0.5 rad), already converted upstream — not
        # KRTP colatitude. Just convert to degrees.
        th_rad = float(payload.coord_th[idx])
        extra["mag_lat"] = float(np.degrees(th_rad))
    if payload.coord_phi is not None and idx < len(payload.coord_phi):
        # KSM longitude in radians; LT = (phi_deg / 15 + 12) mod 24
        phi_rad = float(payload.coord_phi[idx])
        lt = (np.degrees(phi_rad) / 15.0 + 12.0) % 24.0
        extra["local_time"] = float(lt)

    extra["region"] = region_at_peak_from_info(payload.info, peak_time)
    sls = ppo_at_peak_from_info(payload.info, peak_time, seg_t0, n_samples)
    if sls.get("sls5n") is not None:
        extra["sls5_phase_n"] = sls["sls5n"]
    if sls.get("sls5s") is not None:
        extra["sls5_phase_s"] = sls["sls5s"]

    if (
        _ENRICH_KMAG
        and _SATURN_FIELD is not None
        and r_off_val is not None
        and th_rad is not None
        and phi_rad is not None
    ):
        x, y, z = _ksm_from_offset_spherical(r_off_val, th_rad, phi_rad)
        t_j2000 = peak_time.timestamp() - J2000_POSIX
        try:
            kmag = kmag_peak_columns(x, y, z, t_j2000, _SATURN_FIELD, _TRACING_CFG)
        except Exception as exc:  # noqa: BLE001 — never crash a row for a trace failure
            log.debug(
                "seg %d: KMAG enrichment failed at peak: %s", payload.seg_idx, exc
            )
        else:
            extra.update(kmag)
    return extra


_MAX_AMP_NT: float = 10.0  # set by main(); workers receive it via Pool initializer
_THRESHOLD_METHOD: str = "mad_row"
_APPLY_COI: bool = False
_BG_ARCHIVE: BGArchive | None = None
_ENRICH_KMAG: bool = False
_SATURN_FIELD: SaturnField | None = None
_TRACING_CFG: TracingConfig = TracingConfig()


def _init_worker(
    cap_nT: float,
    threshold_method: str,
    apply_coi: bool,
    bg_archive: BGArchive | None,
    enrich_kmag: bool,
    field_config: SaturnFieldConfig | None,
    tracing_cfg: TracingConfig,
) -> None:
    """Pool initializer: propagate detector config into each worker's module globals."""
    global _MAX_AMP_NT, _THRESHOLD_METHOD, _APPLY_COI, _BG_ARCHIVE
    global _ENRICH_KMAG, _SATURN_FIELD, _TRACING_CFG
    _MAX_AMP_NT = float(cap_nT)
    _THRESHOLD_METHOD = str(threshold_method)
    _APPLY_COI = bool(apply_coi)
    _BG_ARCHIVE = bg_archive
    _ENRICH_KMAG = bool(enrich_kmag)
    _TRACING_CFG = tracing_cfg
    if enrich_kmag:
        _SATURN_FIELD = SaturnField(field_config)
        # Warm up the numba JIT so the first real trace doesn't pay it.
        try:
            _SATURN_FIELD.field_cartesian(10.0, 0.0, 0.0, 0.0, coord="KSM")
        except Exception:  # noqa: BLE001
            pass
    else:
        _SATURN_FIELD = None


def _amp_within_cap(d: DetectedEvent, cap_nT: float) -> bool:
    """Reject events where any wave component exceeds ``cap_nT``.

    Catches the proximal-orbit pathology where the detector picks up the
    rotating ambient dipole field as a "wave" of ~10-20 kT amplitude at
    perikrone (r ~ 1 R_S). Cassini's true wave amplitudes in the QP bands
    are well below 20 nT in the magnetosphere.
    """
    return max(d.b_perp1_amp, d.b_perp2_amp, d.b_par_amp) <= cap_nT


def process_segment(payload: SegmentDetectionPayload) -> list[dict]:
    """Worker entry point: detect events and return parquet-ready rows."""
    times = payload.times
    if len(times) < 18 * 60:
        return []
    info = payload.info or {}
    if info.get("NaN_count", 0) and info["NaN_count"] > 18 * 60:
        return []

    seg_t0 = times[0]
    t_seconds = np.array(
        [(t - seg_t0).total_seconds() for t in times],
        dtype=float,
    )
    fields = np.column_stack([payload.b_par, payload.b_perp1, payload.b_perp2])

    try:
        detections = detect_round8(
            t_seconds,
            fields,
            dt=60.0,
            epoch=seg_t0,
            threshold_method=_THRESHOLD_METHOD,  # type: ignore[arg-type]
            apply_coi_mask=_APPLY_COI,
            bg_archive=_BG_ARCHIVE,
            region=payload.region_midpoint,
        )
    except Exception as exc:  # noqa: BLE001 — worker isolation
        log.warning("seg %d: detector raised: %s", payload.seg_idx, exc)
        return []

    if not detections:
        return []

    central_start, central_end = segment_central_window(times)
    seg_id = f"seg_{payload.seg_idx:05d}"
    rows: list[dict] = []
    n_amp_rejected = 0
    for k, d in enumerate(detections):
        # Only keep detections whose peak falls in the central 24 h.
        if not (central_start <= d.peak.peak_time < central_end):
            continue
        if not _amp_within_cap(d, _MAX_AMP_NT):
            n_amp_rejected += 1
            continue
        extra = _enrich_at_peak(d, payload)
        extra["segment_idx"] = payload.seg_idx
        # event_id will be reassigned globally after merge.
        row = event_to_record(
            d,
            event_id=k,
            segment_id=seg_id,
            extra=extra,
        )
        rows.append(row)
    if n_amp_rejected:
        log.debug(
            "seg %d: rejected %d events exceeding %.0f nT amplitude cap",
            payload.seg_idx,
            n_amp_rejected,
            _MAX_AMP_NT,
        )
    return rows


def _smoke_check(rows: list[dict]) -> None:
    if not rows:
        log.warning("no events detected — sweep produced an empty catalogue")
        return
    df_bands: dict[str, int] = {}
    for r in rows:
        df_bands[r["band"]] = df_bands.get(r["band"], 0) + 1
    log.info(
        "rows=%d  bands=%s  q∈[%.2f,%.2f]  d∈[%.2f,%.2f]  par_frac∈[%.2f,%.2f]",
        len(rows),
        ", ".join(f"{b}={n}" for b, n in sorted(df_bands.items())),
        min(r["q_factor"] for r in rows),
        max(r["q_factor"] for r in rows),
        min(r["stokes_d"] for r in rows),
        max(r["stokes_d"] for r in rows),
        min(r["mva_par_frac"] for r in rows),
        max(r["mva_par_frac"] for r in rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Process a single year only (default: full mission)",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run in-process (no multiprocessing) for debugging",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes (default: 8)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=qp.OUTPUT_DIR / "events_round8.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--max-amp-nT",
        type=float,
        default=10.0,
        help="Reject events with any wave-component amplitude above this "
        "value (default: 10 nT). Enforces the small-amplitude linear-wave "
        "regime (delta-B/B << 1 at Cassini's typical 10-25 R_S range) "
        "and excludes proximal-orbit perikrone passes where the rotating "
        "ambient field looks like a wave.",
    )
    parser.add_argument(
        "--threshold-method",
        choices=("mad_row", "tc_chi2", "fdr_chi2", "pooled"),
        default="tc_chi2",
        help="CWT amplitude gate. 'tc_chi2' (default) — Torrence-Compo "
        "AR(1)+chi2(2) test at the round-8 Bonferroni FWER. 'mad_row' is "
        "the legacy per-row MAD gate. 'fdr_chi2' and 'pooled' are "
        "diagnostic alternatives; 'pooled' requires --bg-archive.",
    )
    parser.add_argument(
        "--coi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Morlet cone-of-influence mask before ridge extraction "
        "(default: on). Pass --no-coi to disable.",
    )
    parser.add_argument(
        "--bg-archive",
        type=Path,
        default=qp.OUTPUT_DIR / "bg_archive.zarr",
        help="Pooled background archive (built by scripts/build_bg_archive.py); "
        "only used when --threshold-method=pooled.",
    )
    parser.add_argument(
        "--enrich-kmag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Per-event KMAG single trace at the peak — populates "
        "kmag_inv_lat_peak / l_eq_peak / is_closed_peak columns "
        "(default: on; pass --no-enrich-kmag to disable). Adds ~1 s "
        "to a 100 s full-mission sweep.",
    )
    add_tracing_args(parser)
    add_verbosity_arg(parser)
    args = parser.parse_args()

    # Propagate to workers (spawn picks up module globals at import time).
    global _MAX_AMP_NT, _THRESHOLD_METHOD, _APPLY_COI, _BG_ARCHIVE
    global _ENRICH_KMAG, _SATURN_FIELD, _TRACING_CFG
    _MAX_AMP_NT = float(args.max_amp_nT)
    _THRESHOLD_METHOD = str(args.threshold_method)
    _APPLY_COI = bool(args.coi)
    _ENRICH_KMAG = bool(args.enrich_kmag)
    _TRACING_CFG = TracingConfig(
        step=args.trace_step,
        max_radius=args.trace_max_radius,
        region_filter=None,
    )
    field_config = SaturnFieldConfig() if args.enrich_kmag else None
    if args.enrich_kmag and args.serial:
        _SATURN_FIELD = SaturnField(field_config)

    setup_logging(
        args.verbose,
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bg_archive: BGArchive | None = None
    if args.threshold_method == "pooled":
        bg_archive = load_bg_archive(args.bg_archive)
        if bg_archive is None:
            parser.error(
                f"--threshold-method=pooled requires {args.bg_archive} to exist; "
                "build it with scripts/build_bg_archive.py first.",
            )
    _BG_ARCHIVE = bg_archive

    t_start = time.perf_counter()
    segments, keep_idx = load_segments(args.year)
    log.info("loaded %d segments (year=%s)", len(keep_idx), args.year)

    payloads: list[SegmentDetectionPayload] = []
    for i, idx in enumerate(keep_idx):
        seg = segments[idx]
        p = _to_payload(int(idx), seg)
        if p is not None:
            payloads.append(p)
        if (i + 1) % 200 == 0:
            log.info("payloads built: %d / %d", i + 1, len(keep_idx))
    log.info("built %d valid payloads", len(payloads))

    rows: list[dict] = []
    if args.serial:
        for p in payloads:
            rows.extend(process_segment(p))
    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(
                args.max_amp_nT,
                args.threshold_method,
                args.coi,
                bg_archive,
                args.enrich_kmag,
                field_config,
                _TRACING_CFG,
            ),
        ) as pool:
            for batch in pool.imap_unordered(process_segment, payloads, chunksize=4):
                rows.extend(batch)

    # Reassign monotonic event_ids in time order.
    rows.sort(key=lambda r: r["peak_time"])
    for k, r in enumerate(rows):
        r["event_id"] = k

    _smoke_check(rows)

    attrs = {
        "run": "round8",
        "year": args.year,
        "n_segments_processed": len(payloads),
        "n_events": len(rows),
        "detector": {
            "name": "detect_round8",
            "fwer_alpha": SEGMENT_FWER_ALPHA,
            "min_q_factor": MIN_Q_FACTOR,
            "min_stokes_d": MIN_DEGREE_OF_POLARIZATION,
            "max_mva_par_frac": MAX_MVA_PARALLEL_FRACTION,
            "max_amp_nT": float(args.max_amp_nT),
            "threshold_method": str(args.threshold_method),
            "apply_coi_mask": bool(args.coi),
            "bg_archive": (
                str(args.bg_archive) if args.threshold_method == "pooled" else None
            ),
            "enrich_kmag": bool(args.enrich_kmag),
            "trace_step": float(args.trace_step),
            "trace_max_radius": float(args.trace_max_radius),
        },
        "elapsed_seconds": time.perf_counter() - t_start,
    }
    n_written = events_to_parquet(rows, args.output, attrs=attrs)
    log.info(
        "wrote %d rows to %s in %.1fs",
        n_written,
        args.output,
        attrs["elapsed_seconds"],
    )
    print(f"Wrote {n_written} events to {args.output}")


if __name__ == "__main__":
    main()
