"""Cross-check the round-8 events parquet, slim event-time zarr, and
KMAG dwell sibling. Computes a sample event/dwell ratio per band on
the equatorial-r axis and the dipole inv-lat axis.

Usage::

    uv run python scripts/verify_event_dwell_ratios.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

import qp  # noqa: E402


def _bar(width: int, value: float, vmax: float) -> str:
    if vmax <= 0:
        return ""
    n = int(round(width * value / vmax))
    return "█" * n + "·" * (width - n)


def main() -> None:
    import pandas as pd
    import xarray as xr

    out = qp.OUTPUT_DIR
    parquet = out / "events_round8.parquet"
    ev_zarr = out / "event_time_grid_round8.zarr"
    dw_zarr = out / "dwell_grid_cassini_saturn.zarr"
    eq_zarr = out / "dwell_grid_kmag_eq_r.zarr"

    print(f"Project Output dir: {out}")
    print()

    # ---- parquet
    print(f"[1/4] parquet: {parquet}")
    df = pd.read_parquet(parquet)
    print(f"  rows                 : {len(df)}")
    print(f"  bands                : {dict(df['band'].value_counts())}")
    print(
        f"  date range           : {df['peak_time'].min()} to {df['peak_time'].max()}"
    )
    print(
        f"  duration  (h)        : "
        f"min={df['duration_minutes'].min() / 60:.2f} "
        f"med={df['duration_minutes'].median() / 60:.2f} "
        f"max={df['duration_minutes'].max() / 60:.2f}"
    )
    print(
        f"  Q-factor             : "
        f"min={df['q_factor'].min():.2f} "
        f"med={df['q_factor'].median():.2f} "
        f"max={df['q_factor'].max():.2f}"
    )
    print(
        f"  Stokes d             : "
        f"min={df['stokes_d'].min():.3f} "
        f"med={df['stokes_d'].median():.3f}"
    )
    print(
        f"  MVA |e_max·b_par|^2  : "
        f"min={df['mva_par_frac'].min():.3f} "
        f"med={df['mva_par_frac'].median():.3f} "
        f"max={df['mva_par_frac'].max():.3f}"
    )
    if "region" in df.columns:
        print(f"  regions              : {dict(df['region'].value_counts())}")

    # ---- event-time zarr
    print()
    print(f"[2/4] event-time zarr: {ev_zarr}")
    ev = xr.open_zarr(ev_zarr)
    has_kmag = bool(ev.attrs.get("kmag_inv_lat_populated", False))
    has_eq_r = bool(ev.attrs.get("kmag_eq_r_populated", False))
    print(f"  bands                : {ev.attrs.get('bands')}")
    print(f"  schema               : {ev.attrs.get('schema')}")
    print(f"  kmag_inv_lat?        : {has_kmag}")
    print(f"  kmag_eq_r?           : {has_eq_r}")
    print(f"  total minutes (3D)   : {float(ev['total_total'].sum()):.0f}")
    for b in ev.attrs.get("bands", []):
        print(f"    {b}_total          : {float(ev[f'{b}_total'].sum()):.0f}")

    # ---- canonical dwell zarr
    print()
    print(f"[3/4] canonical dwell: {dw_zarr}")
    if not dw_zarr.exists():
        print(
            "  MISSING — skip ratio against the canonical 3D + dipole_inv_lat schemas"
        )
    else:
        dw = xr.open_zarr(dw_zarr)
        print(f"  total minutes        : {float(dw['total'].sum()):.0f}")
        # Dipole inv-lat ratio for QP60 in the magnetosphere
        if "QP60_dipole_inv_lat_magnetosphere" in ev:
            num = ev["QP60_dipole_inv_lat_magnetosphere"]
            den = dw["dipole_inv_lat_magnetosphere"]
            ratio = (num / den).where(den > 0)
            n_filled = int(ratio.notnull().sum())
            r_max = float(ratio.max(skipna=True))
            print(
                f"  QP60 dipole_inv_lat ratio: max={r_max:.4f}, "
                f"{n_filled} non-empty bins"
            )

    # ---- equatorial-r dwell sibling
    print()
    print(f"[4/4] equatorial-r dwell sibling: {eq_zarr}")
    if not eq_zarr.exists():
        print("  MISSING — skip ratio against the kmag_eq_r schema")
    else:
        eq = xr.open_zarr(eq_zarr)
        print(f"  variables            : {sorted(eq.data_vars)[:6]}…")
        print(
            f"  kmag_eq_r dwell (closed): "
            f"{float(eq['kmag_eq_r_closed_total'].sum()) / 60:.1f} h"
        )

        if has_eq_r and "QP60_kmag_eq_r_closed_total" in ev:
            num = ev["QP60_kmag_eq_r_closed_total"]
            den = eq["kmag_eq_r_closed_total"]
            ratio = (num / den).where(den > 0)
            print("\n  QP60 closed-line equatorial-r vs LT (occurrence rate):")
            print(
                f"  axes: rows = kmag_eq_r ({len(eq.kmag_eq_r)} bins), "
                f"cols = LT ({len(eq.local_time)} bins)"
            )
            agg_lt = ratio.mean("local_time", skipna=True)
            r_max = float(np.nanmax(agg_lt.values))
            print(
                f"  max LT-averaged ratio: {r_max:.4e}  "
                f"(at L = {float(agg_lt.kmag_eq_r[int(np.nanargmax(agg_lt.values))]):.1f} R_S)"
            )
            # ASCII histogram of LT-averaged ratio across L-shells (top 30 R_S)
            l_centers = eq["kmag_eq_r"].values
            mask = (l_centers <= 30.0) & (l_centers >= 4.0)
            vals = agg_lt.values[mask]
            ls = l_centers[mask]
            v_max = float(np.nanmax(vals)) if vals.size else 0.0
            for L, v in zip(ls, vals, strict=False):
                if not np.isfinite(v):
                    continue
                print(f"    L = {L:5.1f} R_S | {_bar(40, v, v_max):s} | {v:.4e}")
        elif "QP60_kmag_eq_r_closed_total" not in ev:
            print(
                "  (KMAG schemas not populated in event-time zarr — "
                "re-run bin_events_round8.py --with-kmag-trace)"
            )


if __name__ == "__main__":
    main()
