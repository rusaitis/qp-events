"""Phase 8.1 — Quality-weighted event-time binning.

Bins the v3 event catalog onto the same 3D (r, mag_lat, LT) grid as the
dwell zarr, but uses quality-weighted contributions:

    numerator cell = sum(quality_i * duration_i) in minutes

Three grids are written side by side for comparison:
- ``event_time_grid_v3_unweighted.zarr`` — unweighted (same as v2)
- ``event_time_grid_v3_weighted.zarr``   — quality-weighted (q × Δt)
- ``event_time_grid_v3_q03.zarr``        — hard quality > 0.3 cut, unweighted

Also writes a 2D (inv_lat, LT) grid using ``|dipole_inv_lat|`` folded onto
the positive hemisphere axis, for Fig 8b.

Usage::

    uv run python scripts/bin_event_time_v3.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qp.dwell.grid import DwellGridConfig  # noqa: E402
from qp.events.binning import (  # noqa: E402
    BinningStats,
    SegmentPositions,
    accumulate_segment_dwell,
    bin_events_peak_position,
    grids_to_xarray,
    save_event_time_zarr,
)
from qp.events.bands import QP_BAND_NAMES  # noqa: E402
from qp.events.catalog import WaveEvent  # noqa: E402


def _df_to_events(df: pd.DataFrame, quality_col: str | None = None) -> list[WaveEvent]:
    """Convert a parquet DataFrame to WaveEvent objects."""
    import datetime
    events = []
    for _, row in df.iterrows():
        q = None
        if quality_col and quality_col in df.columns:
            v = row.get(quality_col)
            if pd.notna(v):
                q = float(v)

        # Parse datetimes so duration_minutes property works
        def _parse_dt(v):
            if isinstance(v, datetime.datetime):
                return v
            return datetime.datetime.fromisoformat(str(v))

        ev = WaveEvent(
            date_from=_parse_dt(row["date_from"]),
            date_to=_parse_dt(row["date_to"]),
            band=row.get("band"),
            r_distance=row.get("coord_r"),
            mag_lat=row.get("mag_lat"),
            local_time=row.get("local_time"),
            quality=q,
        )
        events.append(ev)
    return events


def _bin_inv_lat_lt(
    df: pd.DataFrame, quality_col: str, quality_weighted: bool,
    n_inv_lat: int = 36, n_lt: int = 96,
) -> dict[str, np.ndarray]:
    """Bin events on |dipole_inv_lat| × LT 2D grid."""
    inv_lat_edges = np.linspace(0, 90, n_inv_lat + 1)
    lt_edges = np.linspace(0, 24, n_lt + 1)

    grids: dict[str, np.ndarray] = {
        b: np.zeros((n_inv_lat, n_lt), dtype=np.float64)
        for b in QP_BAND_NAMES
    }
    grids["total"] = np.zeros((n_inv_lat, n_lt), dtype=np.float64)

    for _, row in df.iterrows():
        inv_lat = row.get("dipole_inv_lat")
        lt = row.get("local_time")
        band = row.get("band")
        dur = row.get("duration_minutes")
        if not all(pd.notna(v) for v in [inv_lat, lt, dur, band]):
            continue
        abs_inv_lat = abs(float(inv_lat))
        lt_val = float(lt) % 24
        dur_val = float(dur)

        weight = 1.0
        if quality_weighted and quality_col in df.columns:
            q = row.get(quality_col)
            if pd.notna(q):
                weight = float(q)

        i_lat = int(np.searchsorted(inv_lat_edges, abs_inv_lat, side="right") - 1)
        i_lt = int(np.searchsorted(lt_edges, lt_val, side="right") - 1)
        i_lat = max(0, min(n_inv_lat - 1, i_lat))
        i_lt = max(0, min(n_lt - 1, i_lt))

        contribution = dur_val * weight
        if band in grids:
            grids[band][i_lat, i_lt] += contribution
        grids["total"][i_lat, i_lt] += contribution

    return grids


def main() -> None:
    cat_v3 = _PROJECT_ROOT / "Output" / "events_qp_v3.parquet"
    if not cat_v3.exists():
        cat_v3 = _PROJECT_ROOT / "Output" / "events_qp_v2.parquet"
        print(f"Using v2 catalog: {cat_v3}")

    print(f"Loading catalog: {cat_v3}")
    df = pd.read_parquet(cat_v3)
    quality_col = "quality_v3" if "quality_v3" in df.columns else "quality"
    print(f"  {len(df)} events, quality column: {quality_col}")

    config = DwellGridConfig()

    # ── 1. Unweighted grid ────────────────────────────────────────────────────
    print("Binning (unweighted)...")
    events_all = _df_to_events(df, quality_col=quality_col)
    grids_uw, stats_uw = bin_events_peak_position(events_all, config)
    print(f"  binned {stats_uw.n_binned}/{stats_uw.n_total} events")

    out_uw = _PROJECT_ROOT / "Output" / "event_time_grid_v3_unweighted.zarr"
    save_event_time_zarr(grids_uw, config, out_uw,
                          title="QP event time v3 (unweighted)",
                          extra_attrs={"quality_col": quality_col})
    print(f"  wrote {out_uw}")

    # ── 2. Quality-weighted grid ──────────────────────────────────────────────
    print("Binning (quality-weighted)...")
    grids_w, stats_w = bin_events_peak_position(
        events_all, config,
        quality_weighted=True, quality_col="quality",
    )
    out_w = _PROJECT_ROOT / "Output" / "event_time_grid_v3_weighted.zarr"
    save_event_time_zarr(grids_w, config, out_w,
                          title="QP event time v3 (quality-weighted)",
                          extra_attrs={"quality_col": quality_col,
                                       "weighted": True})
    print(f"  wrote {out_w}")

    # ── 3. Hard quality cut grid ──────────────────────────────────────────────
    print("Binning (quality > 0.3 hard cut)...")
    df_q03 = df[df[quality_col].fillna(0) > 0.3]
    print(f"  events with q>0.3: {len(df_q03)}")
    events_q03 = _df_to_events(df_q03, quality_col=quality_col)
    grids_q03, stats_q03 = bin_events_peak_position(events_q03, config)

    out_q03 = _PROJECT_ROOT / "Output" / "event_time_grid_v3_q03.zarr"
    save_event_time_zarr(grids_q03, config, out_q03,
                          title="QP event time v3 (q > 0.3)",
                          extra_attrs={"quality_col": quality_col,
                                       "min_quality": 0.3})
    print(f"  wrote {out_q03}")

    # ── 4. Consistency dwell (same approximations as event binner) ────────────
    # Load the existing v2 zarr dwell for the denominator
    dwell_v2 = _PROJECT_ROOT / "Output" / "event_time_grid_v2.zarr"
    if dwell_v2.exists():
        print(f"  using dwell from {dwell_v2}")
    else:
        print("  dwell zarr not found — normalization will use raw event grid")

    # ── 5. Invariant-latitude 2D grid ─────────────────────────────────────────
    print("Building invariant-latitude grids...")
    inv_uw = _bin_inv_lat_lt(df, quality_col, quality_weighted=False)
    inv_w = _bin_inv_lat_lt(df, quality_col, quality_weighted=True)
    inv_q03 = _bin_inv_lat_lt(
        df[df[quality_col].fillna(0) > 0.3], quality_col, quality_weighted=False,
    )

    n_inv_lat, n_lt = 36, 96
    inv_lat_centers = np.linspace(0, 90, n_inv_lat, endpoint=False) + 90 / (2 * n_inv_lat)
    lt_centers = np.linspace(0, 24, n_lt, endpoint=False) + 12 / n_lt

    npz_out = _PROJECT_ROOT / "Output" / "event_time_inv_lat_v3.npz"
    save_dict = {
        "inv_lat_centers": inv_lat_centers,
        "lt_centers": lt_centers,
        "quality_col": np.array([quality_col]),
    }
    for band in [*QP_BAND_NAMES, "total"]:
        save_dict[f"event_time_{band}"] = inv_uw[band]
        save_dict[f"event_time_{band}_weighted"] = inv_w[band]
        save_dict[f"event_time_{band}_q03"] = inv_q03[band]

    np.savez(npz_out, **save_dict)
    print(f"  wrote {npz_out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nSummary:")
    for name, grids in [("unweighted", grids_uw), ("weighted", grids_w), ("q>0.3", grids_q03)]:
        total = grids["total"].sum() / 60.0  # convert minutes to hours
        for band in QP_BAND_NAMES:
            band_h = grids[band].sum() / 60.0
        print(f"  {name:12s}: total event time = {total:.0f} h, "
              f"QP30={grids['QP30'].sum()/60:.0f}h, "
              f"QP60={grids['QP60'].sum()/60:.0f}h, "
              f"QP120={grids['QP120'].sum()/60:.0f}h")


if __name__ == "__main__":
    main()
