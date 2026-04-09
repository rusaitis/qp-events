"""Phase 3 — mission-wide QP event sweep.

For each 36-hour MFA segment in
``DATA/CASSINI-DATA/DataProducts/Cassini_MAG_MFA_36H.npy``:

1. Skip if the segment is flagged or has fewer than 18 h of valid samples.
2. Run :func:`qp.events.detector.detect_with_gate` on
   ``b_perp1`` and ``b_perp2`` to get a list of accepted wave packets,
   one per QP band.
3. Restrict each accepted packet to the **central 24 h** of its
   segment (to avoid double-counting at day boundaries).
4. Enrich each packet with spacecraft coordinates, region,
   polarization, PPO phase, and a globally unique ``event_id``.
5. Return a list of :class:`qp.events.catalog.WaveEvent`.

A multiprocessing :class:`multiprocessing.Pool` farms one segment per
worker. The driver collects the results, persists them to
``Output/events_qp_v1.parquet`` (with an ``.npy`` fallback), and writes
a smoke-check summary to
``Output/diagnostics/event_catalog_summary.txt``.

Usage
-----
::

    uv run python scripts/sweep_events.py                # full mission
    uv run python scripts/sweep_events.py --year 2007    # one year only
    uv run python scripts/sweep_events.py --max-segments 50  # smoke test
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import types
from dataclasses import asdict
from multiprocessing import Pool, get_context
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


# Stub legacy classes for unpickling SignalSnapshot objects
def _register_pickle_stubs() -> None:
    stub_classes = [
        "SignalSnapshot", "NewSignal", "Interval", "FFT_list",
        "WaveSignal", "Wave",
    ]
    stub_modules = [
        "__main__", "data_sweeper", "mag_fft_sweeper",
        "cassinilib", "cassinilib.NewSignal",
    ]
    for mod_path in stub_modules:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)
        for cls in stub_classes:
            setattr(sys.modules[mod_path], cls, type(cls, (), {}))


_register_pickle_stubs()


import qp  # noqa: E402
from qp.events.bands import QP_BAND_NAMES  # noqa: E402
from qp.events.catalog import WaveEvent  # noqa: E402
from qp.events.detector import detect_with_gate  # noqa: E402
from qp.events.threshold import GateConfig  # noqa: E402
from qp.signal.cross_correlation import (  # noqa: E402
    classify_polarization,
    phase_shift,
)


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------

#: Production gate (calibrated by scripts/calibrate_threshold.py).
PRODUCTION_GATE: GateConfig = GateConfig(
    n_sigma=5.0,
    min_pixels=300,
    min_duration_hours=2.5,
    min_oscillations=3.0,
    enable_fft_screen=False,
)


def _segment_central_window(
    times: list[datetime.datetime], hours_pad: int = 6,
) -> tuple[datetime.datetime, datetime.datetime]:
    """Return the central 24-hour window of a 36-hour segment."""
    t0 = times[0]
    t_end = times[-1]
    central_start = t0 + datetime.timedelta(hours=hours_pad)
    central_end = t_end - datetime.timedelta(hours=hours_pad - 1)  # inclusive
    return central_start, central_end


def _ppo_at_peak_from_info(
    info: dict,
    peak_time: datetime.datetime,
    seg_t0: datetime.datetime,
    n_samples: int,
) -> dict[str, float | None]:
    """Look up SLS5 N/S phases at the packet peak time from a payload info."""
    out: dict[str, float | None] = {
        "sls5n": None, "sls5s": None,
        "sls5n2": None, "sls5s2": None,
    }
    elapsed_min = (peak_time - seg_t0).total_seconds() / 60.0
    for key in ("SLS5N", "SLS5S", "SLS5N2", "SLS5S2"):
        arr = info.get(key)
        if arr is None or len(arr) == 0:
            continue
        cadence = n_samples / len(arr)
        idx = int(round(elapsed_min / cadence))
        idx = max(0, min(len(arr) - 1, idx))
        try:
            out[key.lower()] = float(arr[idx])
        except (TypeError, ValueError):
            pass
    return out


def _region_at_peak_from_info(
    info: dict, peak_time: datetime.datetime,
) -> str:
    """Translate seg.info['locations'] into a region label at peak_time."""
    locations = info.get("locations")
    flag_times = info.get("flag_times")
    if locations is None or flag_times is None:
        return "unknown"
    times = [
        t for t in flag_times if isinstance(t, datetime.datetime)
    ]
    if not times:
        return "unknown"
    deltas = [abs((t - peak_time).total_seconds()) for t in times]
    idx = int(np.argmin(deltas))
    if idx >= len(locations):
        return "unknown"
    try:
        code = int(locations[idx])
    except (TypeError, ValueError):
        return "unknown"
    return _REGION_CODE_TO_NAME.get(code, "unknown")


_REGION_CODE_TO_NAME = {
    0: "magnetosphere",
    1: "magnetosheath",
    2: "solar_wind",
    9: "unknown",
}


def segment_to_payload(seg_idx: int, seg) -> "SegmentPayload | None":
    """Extract a SegmentPayload from a SignalSnapshot in the parent process.

    This avoids shipping SignalSnapshot objects across the spawn
    boundary, which would require the worker to also register the
    legacy pickle stubs (and re-load every time).
    """
    if getattr(seg, "flag", None) is not None:
        return None
    if not hasattr(seg, "FIELDS") or len(seg.FIELDS) < 4:
        return None
    if not hasattr(seg, "datetime") or len(seg.datetime) == 0:
        return None
    times = list(seg.datetime)
    n_samples = len(times)
    if n_samples < 18 * 60:
        return None
    b_par = np.asarray(seg.FIELDS[0].y, dtype=float)
    b_perp1 = np.asarray(seg.FIELDS[1].y, dtype=float)
    b_perp2 = np.asarray(seg.FIELDS[2].y, dtype=float)
    if (
        len(b_perp1) != n_samples
        or np.isnan(b_perp1).all()
        or np.isnan(b_perp2).all()
    ):
        return None
    coords = {c.name: np.asarray(c.y, dtype=float) for c in seg.COORDS}
    info = getattr(seg, "info", None) or {}
    return SegmentPayload(
        seg_idx=seg_idx,
        times=times,
        b_par=np.nan_to_num(b_par, nan=0.0),
        b_perp1=np.nan_to_num(b_perp1, nan=0.0),
        b_perp2=np.nan_to_num(b_perp2, nan=0.0),
        coord_r=coords.get("r"),
        coord_th=coords.get("th"),
        coord_phi=coords.get("phi"),
        info=_clean_info(info),
    )


def _clean_info(info: dict) -> dict:
    """Strip non-picklable / un-needed fields from seg.info."""
    # We only need a small subset of info downstream:
    keep = ("median_LT", "median_BT", "median_coords", "location",
            "locations", "flag_times",
            "SLS5N", "SLS5S", "SLS5N2", "SLS5S2", "NaN_count", "gaps")
    out: dict = {}
    for k in keep:
        if k in info:
            v = info[k]
            # Convert numpy object arrays of datetimes to plain lists
            if hasattr(v, "tolist"):
                try:
                    out[k] = v.tolist()
                except Exception:
                    out[k] = v
            else:
                out[k] = v
    return out


from dataclasses import dataclass, field  # noqa: E402


@dataclass
class SegmentPayload:
    seg_idx: int
    times: list[datetime.datetime]
    b_par: np.ndarray
    b_perp1: np.ndarray
    b_perp2: np.ndarray
    coord_r: np.ndarray | None
    coord_th: np.ndarray | None
    coord_phi: np.ndarray | None
    info: dict = field(default_factory=dict)


def process_segment(
    args: tuple[SegmentPayload, GateConfig],
) -> list[WaveEvent]:
    """Worker: detect events in one MFA segment and return WaveEvents."""
    payload, gate = args
    seg_idx = payload.seg_idx

    info = payload.info or {}
    n_samples = len(payload.times)
    n_nan = info.get("NaN_count", 0)
    if n_nan and n_nan > 18 * 60:
        return []
    if n_samples < 18 * 60:
        return []

    b_par = payload.b_par
    b_perp1 = payload.b_perp1
    b_perp2 = payload.b_perp2
    times = payload.times

    # Run the gate
    try:
        packets = detect_with_gate(
            b_perp1, b_perp2, times, dt=60.0,
            bands=QP_BAND_NAMES, gate=gate,
        )
    except Exception:
        return []

    if not packets:
        return []

    # Restrict to the central 24h
    central_start, central_end = _segment_central_window(times)
    central = [
        p for p in packets
        if central_start <= p.peak_time < central_end
    ]
    if not central:
        return []

    # Enrich each packet
    events: list[WaveEvent] = []
    coords = {
        "r": payload.coord_r,
        "th": payload.coord_th,
        "phi": payload.coord_phi,
    }
    median_lt = info.get("median_LT")

    # Build a UNIX-time array for fast nearest-index lookup
    epoch = datetime.datetime(1970, 1, 1)
    time_unix = np.array(
        [(t - epoch).total_seconds() for t in times], dtype=float
    )

    for pkt_idx, pkt in enumerate(central):
        peak_unix = (pkt.peak_time - epoch).total_seconds()
        peak_idx = int(np.argmin(np.abs(time_unix - peak_unix)))
        i_from = int(np.argmin(
            np.abs(time_unix - (pkt.date_from - epoch).total_seconds())
        ))
        i_to = int(np.argmin(
            np.abs(time_unix - (pkt.date_to - epoch).total_seconds())
        ))
        sl = slice(i_from, i_to + 1)
        if i_to <= i_from:
            continue

        # Spacecraft coordinates at peak
        r_peak = (
            float(coords["r"][peak_idx])
            if coords["r"] is not None else None
        )
        th_peak = (
            float(coords["th"][peak_idx])
            if coords["th"] is not None else None
        )
        phi_peak = (
            float(coords["phi"][peak_idx])
            if coords["phi"] is not None else None
        )
        if th_peak is not None:
            mag_lat = 90.0 - np.degrees(th_peak)
        else:
            mag_lat = None

        # LT at peak (if median is set, use it; otherwise compute from phi)
        if median_lt is not None:
            lt = float(median_lt)
        elif phi_peak is not None:
            # phi in radians, convert to LT hours (this is approximate;
            # the median_LT in seg.info is more reliable)
            lt = ((np.degrees(phi_peak) / 15.0) + 12.0) % 24.0
        else:
            lt = None

        # Amplitudes in the packet window
        b_perp1_amp = float(np.nanmax(np.abs(b_perp1[sl])))
        b_perp2_amp = float(np.nanmax(np.abs(b_perp2[sl])))
        b_par_amp = float(np.nanmax(np.abs(b_par[sl])))
        rms_perp = float(np.sqrt(
            np.nanmean(b_perp1[sl] ** 2 + b_perp2[sl] ** 2)
        ))
        amplitude = max(b_perp1_amp, b_perp2_amp)

        # Polarization (cross-correlation phase shift)
        try:
            _, phase_deg = phase_shift(
                b_perp1[sl], b_perp2[sl], dt=60.0,
                period=pkt.period_sec or 3600.0,
            )
            polarization = classify_polarization(phase_deg)
        except Exception:
            phase_deg = None
            polarization = None

        # PPO phase
        ppo = _ppo_at_peak_from_info(info, pkt.peak_time, times[0],
                                       n_samples)

        # Region
        region = _region_at_peak_from_info(info, pkt.peak_time)

        # Number of oscillations
        duration_sec = (pkt.date_to - pkt.date_from).total_seconds()
        n_osc = (
            duration_sec / pkt.period_sec if pkt.period_sec else None
        )

        event = WaveEvent(
            date_from=pkt.date_from,
            date_to=pkt.date_to,
            period=pkt.period_sec,
            amplitude=amplitude,
            snr=pkt.prominence,
            local_time=lt,
            mag_lat=mag_lat,
            r_distance=r_peak,
            coord_krtp=(
                (r_peak, th_peak, phi_peak)
                if all(v is not None for v in (r_peak, th_peak, phi_peak))
                else None
            ),
            band=pkt.band,
            period_peak_min=(pkt.period_sec / 60.0) if pkt.period_sec else None,
            rms_amplitude_perp=rms_perp,
            b_perp1_amp=b_perp1_amp,
            b_perp2_amp=b_perp2_amp,
            b_par_amp=b_par_amp,
            region=region,
            polarization=polarization,
            phase_deg=phase_deg,
            ppo_phase_n_deg=ppo.get("sls5n"),
            ppo_phase_s_deg=ppo.get("sls5s"),
            event_id=f"seg{seg_idx:05d}-{pkt.band}-{pkt_idx:02d}",
            segment_id=seg_idx,
            n_oscillations=int(n_osc) if n_osc is not None else None,
        )
        events.append(event)

    return events


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def event_to_dict(ev: WaveEvent) -> dict:
    d = asdict(ev)
    # Convert datetimes to ISO strings for parquet/CSV friendliness
    d["date_from"] = ev.date_from.isoformat()
    d["date_to"] = ev.date_to.isoformat()
    d["duration_minutes"] = ev.duration_minutes
    # Flatten coord_krtp tuple
    if ev.coord_krtp is not None:
        d["coord_r"] = ev.coord_krtp[0]
        d["coord_th"] = ev.coord_krtp[1]
        d["coord_phi"] = ev.coord_krtp[2]
    else:
        d["coord_r"] = None
        d["coord_th"] = None
        d["coord_phi"] = None
    d.pop("coord_krtp", None)
    d.pop("coord_ksm", None)
    return d


def write_catalog(events: list[WaveEvent], out_path: Path) -> Path:
    """Persist a list of WaveEvents to parquet (with .npy fallback)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [event_to_dict(e) for e in events]
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({len(rows)} rows)")
        return out_path
    except Exception as exc:  # pragma: no cover - fallback
        print(f"parquet failed ({exc}); falling back to .npy")
        npy_path = out_path.with_suffix(".npy")
        np.save(npy_path, np.array(rows, dtype=object), allow_pickle=True)
        print(f"Wrote {npy_path}")
        return npy_path


def smoke_check(events: list[WaveEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# QP event catalog smoke checks")
    lines.append("")
    lines.append(
        f"Generated: {datetime.datetime.now(datetime.UTC).isoformat()}"
    )
    lines.append(f"Total events: {len(events)}")
    lines.append("")
    if not events:
        lines.append("(empty)")
        path.write_text("\n".join(lines) + "\n")
        return

    by_band: dict[str, list[WaveEvent]] = {b: [] for b in QP_BAND_NAMES}
    for e in events:
        if e.band in by_band:
            by_band[e.band].append(e)

    lines.append("## Counts by band")
    lines.append("")
    lines.append("| Band | N events | Median dur (h) | Median |B_perp| (nT) |")
    lines.append("|------|---------:|---------------:|---------------------:|")
    for band, evs in by_band.items():
        if not evs:
            lines.append(f"| {band} | 0 | n/a | n/a |")
            continue
        dur_h = np.median([e.duration_hours for e in evs])
        amp = np.median([
            e.amplitude for e in evs if e.amplitude is not None
        ])
        lines.append(
            f"| {band} | {len(evs)} | {dur_h:.2f} | {amp:.3f} |"
        )
    lines.append("")

    lines.append("## Period distribution")
    lines.append("")
    for band, evs in by_band.items():
        if not evs:
            continue
        periods = [e.period_peak_min for e in evs if e.period_peak_min]
        if not periods:
            continue
        q = np.percentile(periods, [10, 50, 90])
        lines.append(
            f"- {band}: median {q[1]:.1f} min "
            f"(10–90 %: {q[0]:.1f}–{q[2]:.1f} min)"
        )
    lines.append("")

    lines.append("## Region fraction")
    lines.append("")
    region_counts: dict[str, int] = {}
    for e in events:
        region_counts[e.region or "unknown"] = (
            region_counts.get(e.region or "unknown", 0) + 1
        )
    for region, count in sorted(region_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {region}: {count} ({count/len(events)*100:.1f} %)")
    lines.append("")

    lines.append("## Polarization fraction")
    lines.append("")
    pol_counts: dict[str, int] = {}
    for e in events:
        pol_counts[e.polarization or "n/a"] = (
            pol_counts.get(e.polarization or "n/a", 0) + 1
        )
    for pol, count in sorted(pol_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {pol}: {count} ({count/len(events)*100:.1f} %)")
    lines.append("")

    lines.append("## Local-time histogram (1-h bins)")
    lines.append("")
    lts = [e.local_time for e in events if e.local_time is not None]
    if lts:
        hist, _ = np.histogram(lts, bins=np.arange(0, 25, 1))
        lines.append("```")
        for h, count in enumerate(hist):
            bar = "#" * int(count * 40 / max(hist.max(), 1))
            lines.append(f"  {h:02d}h | {bar:<40s} {count}")
        lines.append("```")

    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def load_segments(year: int | None = None) -> tuple[np.ndarray, list[int]]:
    """Load all MFA segments and return (array, list of indices to process)."""
    products = qp.DATA_PRODUCTS
    arr = np.load(products / "Cassini_MAG_MFA_36H.npy", allow_pickle=True)
    if year is None:
        keep_idx = list(range(len(arr)))
    else:
        keep_idx = []
        for i, seg in enumerate(arr):
            t0 = seg.datetime[0]
            if isinstance(t0, datetime.datetime) and t0.year == year:
                keep_idx.append(i)
    return arr, keep_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None,
                         help="restrict to a single year")
    parser.add_argument("--max-segments", type=int, default=None,
                         help="cap number of segments processed")
    parser.add_argument("--n-workers", type=int, default=None,
                         help="multiprocessing pool size "
                              "(default: cpu_count - 2)")
    parser.add_argument("--output", type=Path,
                         default=_PROJECT_ROOT / "Output" / "events_qp_v1.parquet")
    parser.add_argument("--summary", type=Path,
                         default=_PROJECT_ROOT / "Output" / "diagnostics" /
                                 "event_catalog_summary.txt")
    parser.add_argument("--serial", action="store_true",
                         help="run in single-process mode for debugging")
    args = parser.parse_args()

    n_workers = args.n_workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"Loading segments...")
    segments, keep_idx = load_segments(args.year)
    if args.max_segments is not None:
        keep_idx = keep_idx[:args.max_segments]
    print(f"  total in archive: {len(segments)}")
    print(f"  to process     : {len(keep_idx)}")
    if args.year is not None:
        print(f"  year filter    : {args.year}")

    print("Extracting payloads...")
    payloads: list[SegmentPayload] = []
    for i in keep_idx:
        p = segment_to_payload(i, segments[i])
        if p is not None:
            payloads.append(p)
    print(f"  payloads ready : {len(payloads)}")
    tasks = [(p, PRODUCTION_GATE) for p in payloads]

    t0 = time.time()
    all_events: list[WaveEvent] = []
    if args.serial:
        for task in tasks:
            all_events.extend(process_segment(task))
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for events in pool.imap_unordered(
                process_segment, tasks, chunksize=4,
            ):
                all_events.extend(events)
    duration = time.time() - t0

    print(f"Done in {duration:.1f} s")
    print(f"  events found    : {len(all_events)}")

    out_path = write_catalog(all_events, args.output)
    smoke_check(all_events, args.summary)
    print(f"  catalog at      : {out_path}")
    print(f"  summary at      : {args.summary}")


if __name__ == "__main__":
    main()
