"""Filter-aware event-time / dwell-time map renderer.

Given the enriched events parquet, the per-event footprints zarr, and
the canonical dwell denominator, this script:

1. Evaluates an optional pandas-style filter expression against the
   parquet (default: keep everything).
2. Sums the per-event sparse contributions over the kept events on
   one of three grids.
3. Divides by the dwell denominator and renders the ratio as a
   heatmap (or as latitude profiles, for ``--grid 3d --integrate r``).

Examples::

    # Default: KMAG invariant-latitude × LT, all events, all bands.
    uv run python scripts/plot_event_dwell_filtered.py

    # QP60 only, high q-factor, equatorial-r view.
    uv run python scripts/plot_event_dwell_filtered.py \\
        --grid l_eq --filter "band == 'QP60' and q_factor > 4"

    # Closed-line events on KMAG inv-lat axis with custom output.
    uv run python scripts/plot_event_dwell_filtered.py \\
        --grid kmag_inv_lat --filter "is_closed_peak" \\
        --output Output/figures/filtered_closed_kmag.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path  # noqa: F401  (used in argparse type and helpers)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _common import FIGURES_DIR, setup_logging  # noqa: E402

import qp  # noqa: E402
from qp.events.footprints import apply_filter, read_zarr  # noqa: E402
from qp.plotting.style import use_paper_style  # noqa: E402

log = logging.getLogger(__name__)

#: Minimum dwell time per cell to render — below this we treat the
#: ratio as noise-dominated and mask out.
MIN_DWELL_MINUTES_DEFAULT = 600.0

#: Default Gaussian smoothing scale, in cells (~2 deg lat / 0.5 h LT).
SMOOTHING_SIGMA_DEFAULT = 2.0


def _smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a 2D array, ignoring NaN cells (mirrors Fig 8)."""
    if sigma <= 0:
        return arr
    from scipy.ndimage import gaussian_filter

    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    sm = gaussian_filter(filled, sigma=sigma)
    wt = gaussian_filter(valid.astype(float), sigma=sigma)
    out = np.where(wt > 0, sm / wt, np.nan)
    out[~valid] = np.nan
    return out


def _load_denominator(grid: str) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    """Return (denom_minutes, coord_arrays, denom_label) for the chosen grid."""
    import xarray as xr

    if grid in ("kmag_inv_lat", "3d"):
        path = qp.OUTPUT_DIR / "dwell_grid_cassini_saturn.zarr"
    elif grid == "l_eq":
        path = qp.OUTPUT_DIR / "dwell_grid_kmag_eq_r.zarr"
    else:
        raise ValueError(f"unknown grid: {grid!r}")

    if not path.exists():
        raise SystemExit(f"dwell denominator missing: {path}")

    ds = xr.open_zarr(path, consolidated=False)
    if grid == "kmag_inv_lat":
        return (
            ds["kmag_inv_lat_closed_magnetosphere"].values,
            {"lt": ds["local_time"].values, "axis": ds["kmag_inv_lat"].values},
            "kmag_inv_lat_closed_magnetosphere",
        )
    if grid == "l_eq":
        return (
            ds["kmag_eq_r_closed_magnetosphere"].values,
            {"lt": ds["local_time"].values, "axis": ds["kmag_eq_r"].values},
            "kmag_eq_r_closed_magnetosphere",
        )
    # 3D: integrate over r → (mag_lat, LT). Use magnetosphere region only.
    g3d = ds["magnetosphere"].values  # (r, lat, LT)
    g2d = g3d.sum(axis=0)
    return (
        g2d,
        {"lt": ds["local_time"].values, "axis": ds["magnetic_latitude"].values},
        "magnetosphere (∑ over r)",
    )


def _ratio(
    numerator: np.ndarray,
    denom: np.ndarray,
    *,
    min_dwell_min: float,
    sigma: float,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom >= min_dwell_min, numerator / denom, np.nan)
    return _smooth(r, sigma)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=qp.OUTPUT_DIR / "events_round8_enriched.parquet",
    )
    parser.add_argument(
        "--footprints",
        type=Path,
        default=qp.OUTPUT_DIR / "event_footprints_round8.zarr",
    )
    parser.add_argument(
        "--grid",
        choices=("kmag_inv_lat", "l_eq", "3d"),
        default="kmag_inv_lat",
        help="Output grid. '3d' integrates over r → (mag_lat, LT).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional pandas-eval expression over event columns "
        "(q_factor, band, region, l_eq_peak, ...). Default: all events.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: Output/figures/filtered_<grid>_<hash>.png)",
    )
    parser.add_argument(
        "--min-dwell-min",
        type=float,
        default=MIN_DWELL_MINUTES_DEFAULT,
        help="Cells with less than this many dwell-minutes are masked.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SMOOTHING_SIGMA_DEFAULT,
        help="Gaussian smoothing sigma in cells; 0 to disable.",
    )
    parser.add_argument(
        "--vmax-pct",
        type=float,
        default=99.0,
        help="Percentile of finite ratios used as colorbar upper bound.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # ---- load --------------------------------------------------------
    import pandas as pd

    df = pd.read_parquet(args.events)
    log.info("events: %d (%s)", len(df), args.events)
    fp = read_zarr(str(args.footprints))
    log.info(
        "footprints: %d events × %d grids (%s)",
        fp.event_ids.size,
        len(fp.grids),
        args.footprints,
    )
    if not np.array_equal(df["event_id"].to_numpy(dtype=np.int64), fp.event_ids):
        raise SystemExit(
            "event_id mismatch between parquet and footprints — they came "
            "from different sweeps. Rebuild footprints with the matching "
            "parquet.",
        )

    # ---- filter ------------------------------------------------------
    mask = apply_filter(df, args.filter)
    n_keep = int(mask.sum())
    log.info(
        "filter %r → kept %d / %d events (%.1f%%)",
        args.filter or "(all)",
        n_keep,
        len(df),
        100.0 * n_keep / len(df) if len(df) else 0.0,
    )

    # ---- numerator ---------------------------------------------------
    grid_key = {
        "kmag_inv_lat": "g_kmag_inv_lat",
        "l_eq": "g_l_eq",
        "3d": "g3d",
    }[args.grid]
    numerator = fp.total(grid_key, mask)
    if args.grid == "3d":
        # Integrate over r to compare with the dwell denominator.
        numerator = numerator.sum(axis=0)  # (lat, LT)

    denom, coords, denom_label = _load_denominator(args.grid)
    if numerator.shape != denom.shape:
        raise SystemExit(
            f"shape mismatch: numerator {numerator.shape} vs "
            f"denominator {denom.shape} for grid={args.grid}",
        )

    ratio = _ratio(
        numerator,
        denom,
        min_dwell_min=args.min_dwell_min,
        sigma=args.sigma,
    )

    # ---- render ------------------------------------------------------
    use_paper_style()
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)

    lt = coords["lt"]
    axis = coords["axis"]

    finite = ratio[np.isfinite(ratio)]
    vmax = float(np.percentile(finite, args.vmax_pct)) if finite.size else 0.1

    im = ax.pcolormesh(
        lt,
        axis,
        np.ma.masked_invalid(ratio),
        shading="auto",
        cmap="plasma",
        vmin=0.0,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(f"Event time / dwell time  ({denom_label})")

    ax.set_xlabel("Local time [h]")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    if args.grid == "kmag_inv_lat":
        ax.set_ylabel("KMAG invariant latitude [deg]")
        ax.set_ylim(-90, 90)
        ax.axhline(0, color="white", lw=0.4, ls=":", alpha=0.5)
    elif args.grid == "l_eq":
        ax.set_ylabel("KMAG equatorial crossing  $L_{eq}$  [R$_S$]")
        ax.set_ylim(0, 50)
    else:
        ax.set_ylabel("Magnetic latitude  [deg]")
        ax.set_ylim(-90, 90)
        ax.axhline(0, color="white", lw=0.4, ls=":", alpha=0.5)
    for x in (6, 12, 18):
        ax.axvline(x, color="white", lw=0.3, ls=":", alpha=0.35)

    title = f"Event/dwell ratio — grid={args.grid}, n={n_keep} events"
    if args.filter:
        title += f", filter={args.filter!r}"
    ax.set_title(title, fontsize=11)

    out = args.output
    if out is None:
        tag = "all" if not args.filter else f"f{abs(hash(args.filter)) % 10_000_000}"
        out = FIGURES_DIR / f"filtered_{args.grid}_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("wrote %s", out)
    print(f"Wrote {out}  (n_keep={n_keep} / {len(df)})")


if __name__ == "__main__":
    main()
