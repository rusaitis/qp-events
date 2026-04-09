"""Phase 7.6/7.7 — v2 event-time binning with quality filter + invariant latitude.

Improvements over v1:
- Quality-score filter (--min-quality, default 0.0)
- 5-degree latitude bins (--n-lat, default 36 for ±90°)
- Builds both (r, mag_lat, LT) and (inv_lat, LT) 2D grids
- Higher dwell floor in occurrence rate (10 h default)

Usage::

    uv run python scripts/bin_event_time_v2.py
    uv run python scripts/bin_event_time_v2.py --min-quality 0.3
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
    bin_events_walking,
    save_event_time_zarr,
)
from qp.events.catalog import WaveEvent  # noqa: E402

# Re-use segment loader from v1
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from bin_event_time import build_segment_positions  # noqa: E402


def load_events_v2(parquet_path: Path, min_quality: float = 0.0) -> list[WaveEvent]:
    """Load v2 catalog with optional quality filter."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if min_quality > 0 and "quality" in df.columns:
        before = len(df)
        df = df[df["quality"] >= min_quality]
        print(f"  quality filter >= {min_quality}: {before} -> {len(df)} events")

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

        # Read quality and dipole_inv_lat if available
        quality = getattr(row, "quality", None)
        inv_lat = getattr(row, "dipole_inv_lat", None)

        ev = WaveEvent(
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
            period_peak_min=getattr(row, "period_peak_min", None),
            rms_amplitude_perp=getattr(row, "rms_amplitude_perp", None),
            b_perp1_amp=getattr(row, "b_perp1_amp", None),
            b_perp2_amp=getattr(row, "b_perp2_amp", None),
            b_par_amp=getattr(row, "b_par_amp", None),
            region=getattr(row, "region", None),
            polarization=getattr(row, "polarization", None),
            phase_deg=getattr(row, "phase_deg", None),
            ppo_phase_n_deg=getattr(row, "ppo_phase_n_deg", None),
            ppo_phase_s_deg=getattr(row, "ppo_phase_s_deg", None),
            event_id=getattr(row, "event_id", None),
            segment_id=int(row.segment_id) if row.segment_id is not None else None,
            n_oscillations=int(row.n_oscillations) if getattr(row, "n_oscillations", None) is not None else None,
            quality=float(quality) if quality is not None and np.isfinite(quality) else None,
            dipole_inv_lat=float(inv_lat) if inv_lat is not None and np.isfinite(inv_lat) else None,
        )
        events.append(ev)
    return events


def bin_inv_lat_lt(
    events: list[WaveEvent],
    n_inv_lat: int = 36,
    n_lt: int = 24,
    inv_lat_range: tuple[float, float] = (0.0, 90.0),
    lt_range: tuple[float, float] = (0.0, 24.0),
) -> dict[str, np.ndarray]:
    """Bin events onto a 2D (invariant_latitude, local_time) grid.

    Uses |dipole_inv_lat| (folded to positive values) to combine
    N and S hemispheres, matching the paper's "conjugate latitude" view.
    """
    from qp.events.bands import QP_BAND_NAMES
    import math

    shape = (n_inv_lat, n_lt)
    grids: dict[str, np.ndarray] = {
        b: np.zeros(shape, dtype=np.float64) for b in QP_BAND_NAMES
    }
    grids["total"] = np.zeros(shape, dtype=np.float64)

    for ev in events:
        inv_lat = ev.dipole_inv_lat
        lt = ev.local_time
        if inv_lat is None or lt is None or ev.band is None:
            continue
        # Fold to positive invariant latitude
        inv_lat_abs = abs(inv_lat)
        if not (inv_lat_range[0] <= inv_lat_abs < inv_lat_range[1]):
            continue
        if not (lt_range[0] <= lt < lt_range[1]):
            continue

        i_lat = int(math.floor(
            (inv_lat_abs - inv_lat_range[0])
            / (inv_lat_range[1] - inv_lat_range[0]) * n_inv_lat
        ))
        i_lat = max(0, min(n_inv_lat - 1, i_lat))
        i_lt = int(math.floor(
            (lt - lt_range[0]) / (lt_range[1] - lt_range[0]) * n_lt
        ))
        i_lt = max(0, min(n_lt - 1, i_lt))

        minutes = ev.duration_minutes
        if ev.band in grids:
            grids[ev.band][i_lat, i_lt] += minutes
        grids["total"][i_lat, i_lt] += minutes

    return grids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path,
                         default=_PROJECT_ROOT / "Output" / "events_qp_v2.parquet")
    parser.add_argument("--output", type=Path,
                         default=_PROJECT_ROOT / "Output" / "event_time_grid_v2.zarr")
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--n-lat", type=int, default=36,
                         help="Number of latitude bins (default 36 = 5-deg bins)")
    parser.add_argument("--strategy", choices=["walking", "peak"],
                         default="walking")
    args = parser.parse_args()

    print(f"Loading catalog: {args.catalog}")
    events = load_events_v2(args.catalog, min_quality=args.min_quality)
    print(f"  events loaded : {len(events)}")

    # Use coarser latitude bins (5° default instead of 1°)
    config = DwellGridConfig(n_lat=args.n_lat)
    print(f"Grid shape : {config.shape}")

    t0 = time.time()
    if args.strategy == "walking":
        positions = build_segment_positions(None)
        grids, stats = bin_events_walking(events, positions, config=config)
        print("Accumulating consistency dwell grid...")
        dwell_grid = accumulate_segment_dwell(positions, config=config)
        grids["dwell"] = dwell_grid
    else:
        from qp.events.binning import bin_events_peak_position
        grids, stats = bin_events_peak_position(events, config=config)
    elapsed = time.time() - t0
    print(f"Binning done in {elapsed:.1f} s")
    print(f"  events binned : {stats.n_binned}/{stats.n_total}")

    # Save 3D grid
    save_event_time_zarr(
        grids, config, args.output,
        title="QP event-time grid v2",
        description=(
            "Phase 7 event-time grid with quality-filtered events, "
            f"5-degree latitude bins. min_quality={args.min_quality}"
        ),
        extra_attrs={
            "catalog": str(args.catalog.name),
            "strategy": args.strategy,
            "min_quality": args.min_quality,
            "n_events_binned": stats.n_binned,
        },
    )
    print(f"Wrote {args.output}")

    # Invariant-latitude 2D grid
    print("Building invariant-latitude × LT grid...")
    inv_grids = bin_inv_lat_lt(events, n_inv_lat=36, n_lt=24)
    inv_out = args.output.parent / "event_time_inv_lat_v2.npz"
    np.savez(
        inv_out,
        **{f"event_time_{k}": v for k, v in inv_grids.items()},
        inv_lat_centers=np.linspace(1.25, 88.75, 36),
        lt_centers=np.linspace(0.5, 23.5, 24),
    )
    print(f"Wrote {inv_out}")

    # Summary
    print("\nPer-band totals (hours):")
    for name in ("QP30", "QP60", "QP120", "total"):
        if name in grids:
            h = grids[name].sum() / 60
            print(f"  {name:6s}: {h:8.1f} h")
    for name in ("QP30", "QP60", "QP120", "total"):
        if name in inv_grids:
            h = inv_grids[name].sum() / 60
            print(f"  {name:6s} (inv_lat): {h:8.1f} h")


if __name__ == "__main__":
    main()
