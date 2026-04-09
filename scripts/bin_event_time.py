"""Phase 4 — bin the QP event catalog onto the dwell-grid coordinates.

Reads ``Output/events_qp_v1.parquet``, scans the MFA segment archive
to recover per-event positions, walks each event minute-by-minute
through its segment, and accumulates minutes per ``(r, mag_lat, LT)``
bin per QP band. Output is a zarr file at
``Output/event_time_grid_v1.zarr`` shaped exactly like
``Output/dwell_grid_cassini_saturn.zarr`` so the two can be divided
side-by-side in Phase 5.

Usage::

    uv run python scripts/bin_event_time.py
    uv run python scripts/bin_event_time.py --strategy peak  # fast fallback
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
import types
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


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
from qp.dwell.grid import DwellGridConfig  # noqa: E402
from qp.events.binning import (  # noqa: E402
    SegmentPositions,
    accumulate_segment_dwell,
    bin_events_peak_position,
    bin_events_walking,
    save_event_time_zarr,
)
from qp.events.catalog import WaveEvent  # noqa: E402


def load_events(parquet_path: Path) -> list[WaveEvent]:
    """Reconstruct WaveEvents from the parquet catalog."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    events: list[WaveEvent] = []
    for row in df.itertuples(index=False):
        coord_krtp = None
        coord_r = getattr(row, "coord_r", None)
        coord_th = getattr(row, "coord_th", None)
        coord_phi = getattr(row, "coord_phi", None)
        if coord_r is not None and coord_th is not None and coord_phi is not None:
            try:
                coord_krtp = (float(coord_r), float(coord_th), float(coord_phi))
            except Exception:
                coord_krtp = None
        events.append(
            WaveEvent(
                date_from=datetime.datetime.fromisoformat(row.date_from),
                date_to=datetime.datetime.fromisoformat(row.date_to),
                period=row.period,
                amplitude=row.amplitude,
                snr=row.snr,
                local_time=row.local_time,
                mag_lat=row.mag_lat,
                r_distance=row.r_distance,
                coord_krtp=coord_krtp,
                band=row.band,
                period_peak_min=row.period_peak_min,
                rms_amplitude_perp=row.rms_amplitude_perp,
                b_perp1_amp=row.b_perp1_amp,
                b_perp2_amp=row.b_perp2_amp,
                b_par_amp=row.b_par_amp,
                region=row.region,
                polarization=row.polarization,
                phase_deg=row.phase_deg,
                ppo_phase_n_deg=row.ppo_phase_n_deg,
                ppo_phase_s_deg=row.ppo_phase_s_deg,
                event_id=row.event_id,
                segment_id=int(row.segment_id) if row.segment_id is not None else None,
                n_oscillations=int(row.n_oscillations) if row.n_oscillations is not None else None,
            )
        )
    return events


def build_segment_positions(
    needed_seg_ids: set[int] | None = None,
) -> dict[int, SegmentPositions]:
    """Load the MFA segment archive and extract minute-cadence positions.

    If ``needed_seg_ids`` is None, return positions for **every** valid
    segment in the archive. Otherwise return only the requested ids.
    """
    print(f"Loading segments archive...")
    arr = np.load(
        qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy",
        allow_pickle=True,
    )
    print(f"  archive size : {len(arr)} segments")
    if needed_seg_ids is None:
        seg_ids: list[int] = list(range(len(arr)))
        print(f"  loading all valid segments...")
    else:
        seg_ids = sorted(needed_seg_ids)
        print(f"  needed by events : {len(seg_ids)}")

    out: dict[int, SegmentPositions] = {}
    epoch = datetime.datetime(1970, 1, 1)
    for seg_idx in seg_ids:
        if seg_idx < 0 or seg_idx >= len(arr):
            continue
        seg = arr[seg_idx]
        if not hasattr(seg, "datetime") or len(seg.datetime) == 0:
            continue
        if not hasattr(seg, "COORDS") or len(seg.COORDS) < 3:
            continue
        if getattr(seg, "flag", None) is not None:
            continue
        coords = {c.name: np.asarray(c.y, dtype=float) for c in seg.COORDS}
        r_arr = coords.get("r")
        th_arr = coords.get("th")
        if r_arr is None or th_arr is None:
            continue
        # The MFA segment's `th` coordinate is **latitude in radians**
        # (not colatitude — verified against info["median_coords"]
        # for several segments). This is the magnetographic latitude
        # in KRTP, not the offset-dipole magnetic latitude that the
        # dwell grid uses. The difference is < 0.25° outside ~3 R_S,
        # well below the 1° bin width.
        mag_lat = np.degrees(th_arr)
        # Local time: the segment file stores median_LT in info, and
        # COORDS phi is body-fixed (KRTP) so it's not directly LT.
        # Use the segment's median_LT for every minute of the segment
        # (LT changes by < 1 h over a 36 h pass for L > 10).
        info = getattr(seg, "info", None) or {}
        median_lt = info.get("median_LT")
        if median_lt is None:
            continue
        lt_arr = np.full(len(seg.datetime), float(median_lt))

        time_unix = np.array(
            [(t - epoch).total_seconds() for t in seg.datetime],
            dtype=float,
        )
        # Mark the central 24 h. The MFA segments are 36 h, t0 at 18:00
        # of the day before the central day, so the central window is
        # t0 + 6 h to t0 + 30 h (i.e. samples 360 to 1799 inclusive at
        # 1-min cadence).
        central_mask = np.zeros(len(time_unix), dtype=bool)
        if len(time_unix) >= 1800:
            central_mask[360:1800] = True
        else:
            # Short segment — use the middle two-thirds
            n = len(time_unix)
            central_mask[n // 6 : 5 * n // 6] = True

        out[seg_idx] = SegmentPositions(
            seg_idx=seg_idx,
            times_unix=time_unix,
            r=r_arr,
            mag_lat=mag_lat,
            local_time=lt_arr,
            central_mask=central_mask,
        )
    print(f"  loaded positions : {len(out)} segments")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path,
                         default=_PROJECT_ROOT / "Output" / "events_qp_v1.parquet")
    parser.add_argument("--output", type=Path,
                         default=_PROJECT_ROOT / "Output" / "event_time_grid_v1.zarr")
    parser.add_argument("--strategy", choices=["walking", "peak"],
                         default="walking")
    args = parser.parse_args()

    print(f"Loading catalog: {args.catalog}")
    events = load_events(args.catalog)
    print(f"  events read   : {len(events)}")

    config = DwellGridConfig()
    print(f"Grid shape : {config.shape}")

    t0 = time.time()
    if args.strategy == "walking":
        # Load positions for ALL valid segments so we can accumulate
        # the consistency dwell grid alongside the events. This costs
        # a few extra seconds of I/O but guarantees event_time
        # ≤ dwell_time per cell by construction (Phase 4 verification).
        positions = build_segment_positions(None)
        grids, stats = bin_events_walking(events, positions, config=config)
        print(f"Accumulating consistency dwell grid...")
        dwell_grid = accumulate_segment_dwell(positions, config=config)
        grids["dwell"] = dwell_grid
    else:
        grids, stats = bin_events_peak_position(events, config=config)
    duration = time.time() - t0
    print(f"Binning done in {duration:.1f} s "
          f"(strategy={args.strategy})")
    print(f"  events binned   : {stats.n_binned}/{stats.n_total} "
          f"({stats.fraction_binned*100:.1f} %)")
    if stats.n_missing_coords:
        print(f"  missing coords  : {stats.n_missing_coords}")
    if stats.n_out_of_range:
        print(f"  out of range    : {stats.n_out_of_range}")

    print()
    print("Per-band totals (minutes / hours):")
    for name in ("QP30", "QP60", "QP120", "total", "dwell"):
        if name not in grids:
            continue
        m = float(grids[name].sum())
        print(f"  {name:6s}: {m:10.0f} min  ({m/60:8.1f} h)")

    if "dwell" in grids:
        # Per-cell sanity: event_time ≤ dwell by construction
        evt_total = grids["total"]
        dwell = grids["dwell"]
        violations = int((evt_total > dwell).sum())
        print(f"  cells where event_total > dwell : {violations} "
              f"(should be 0)")

    print()
    print("Saving zarr...")
    out_path = save_event_time_zarr(
        grids, config, args.output,
        title="QP event-time grid v1",
        description=(
            "Cumulative minutes of detected QP wave activity per "
            "(r, magnetic_latitude, local_time) cell, split by band. "
            "Numerator for the dwell-normalized occurrence-rate maps "
            "in Figs 7/8 of Rusaitis et al."
        ),
        extra_attrs={
            "catalog": str(args.catalog.name),
            "strategy": args.strategy,
            "n_events_binned": stats.n_binned,
            "n_events_total": stats.n_total,
        },
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
