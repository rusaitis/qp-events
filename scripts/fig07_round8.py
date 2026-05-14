"""Figure 7 (round-8) — QP event/dwell ratio vs magnetic latitude.

Reads the round-8 event-time zarr
(``Output/event_time_grid_round8.zarr``) and divides by the canonical
Cassini dwell grid (``Output/dwell_grid_cassini_saturn.zarr``) per
``(r, magnetic_latitude, local_time)`` cell. For each of four LT sectors
(midnight, dawn, noon, dusk; +/-3 h) we slice both grids, collapse the
radial axis, and plot the event-time / dwell-time ratio vs magnetic
latitude for each QP band.

Round-8 supplies only one detection path (the four-gate detector), so
this figure has one line per band per sector, replacing the
unweighted/q03/quality-weighted comparison of the phase-8 version.

Output: ``Output/figures/figure7_round8.png``
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _common import OUTPUT_DIR, ensure_figures_dir, setup_logging  # noqa: E402

from qp.events.bands import QP_BAND_COLORS, QP_BAND_NAMES  # noqa: E402
from qp.events.normalization import collapse_to_latitude, slice_lt_sector  # noqa: E402
from qp.plotting.style import use_paper_style  # noqa: E402

log = logging.getLogger(__name__)

LT_SECTORS = [
    (0.0, 3.0, "Midnight (0 ± 3 h)"),
    (6.0, 3.0, "Dawn (6 ± 3 h)"),
    (12.0, 3.0, "Noon (12 ± 3 h)"),
    (18.0, 3.0, "Dusk (18 ± 3 h)"),
]

BANDS = list(QP_BAND_NAMES)
BAND_COLORS = QP_BAND_COLORS

#: Floor on summed dwell time per latitude bin. Bins below this floor
#: produce ratio NaN — they have too little Cassini coverage to be
#: trusted.
MIN_DWELL_MINUTES_LATITUDE = 600.0  # 10 hours

#: Latitude rebinning width. Native dwell grid is 1 deg; collapse to 5 deg
#: to recover the latitude trend without wallpapering single-cell noise.
LAT_BIN_WIDTH_DEG = 5.0


def _rebin_latitude(
    arr_1d: np.ndarray,
    lat_centers: np.ndarray,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum a 1D latitude array into wider bins (sum-preserving)."""
    edges = np.arange(-90.0, 90.0 + bin_width / 2, bin_width)
    out = np.zeros(edges.size - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for i in range(out.size):
        mask = (lat_centers >= edges[i]) & (lat_centers < edges[i + 1])
        out[i] = arr_1d[mask].sum()
    return out, centers


def _bootstrap_band(
    ev_1d: np.ndarray,
    dw_1d: np.ndarray,
    n: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson bootstrap of the per-latitude ratio."""
    if rng is None:
        rng = np.random.default_rng(42)
    ratios = np.full((n, ev_1d.size), np.nan)
    valid = dw_1d >= MIN_DWELL_MINUTES_LATITUDE
    for i in range(n):
        ev_r = rng.poisson(np.maximum(ev_1d, 0.0)).astype(float)
        ratios[i, valid] = ev_r[valid] / dw_1d[valid]
    return (
        np.nanpercentile(ratios, 16, axis=0),
        np.nanpercentile(ratios, 84, axis=0),
    )


def _sector_ratio(
    ev_3d: np.ndarray,
    dw_3d: np.ndarray,
    lt_centers: np.ndarray,
    lat_centers: np.ndarray,
    center: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the per-latitude event/dwell ratio inside one LT sector."""
    ev_2d = slice_lt_sector(ev_3d, lt_centers, center, half_width)
    dw_2d = slice_lt_sector(dw_3d, lt_centers, center, half_width)
    ev_1d = collapse_to_latitude(ev_2d)
    dw_1d = collapse_to_latitude(dw_2d)
    ev_b, lat_b = _rebin_latitude(ev_1d, lat_centers, LAT_BIN_WIDTH_DEG)
    dw_b, _ = _rebin_latitude(dw_1d, lat_centers, LAT_BIN_WIDTH_DEG)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            dw_b >= MIN_DWELL_MINUTES_LATITUDE,
            ev_b / dw_b,
            np.nan,
        )
    lo, hi = _bootstrap_band(ev_b, dw_b)
    return ratio, lo, hi, lat_b


def main() -> None:
    setup_logging()
    import xarray as xr

    ev_path = OUTPUT_DIR / "event_time_grid_round8.zarr"
    dw_path = OUTPUT_DIR / "dwell_grid_cassini_saturn.zarr"
    if not ev_path.exists() or not dw_path.exists():
        raise SystemExit(
            f"Missing input(s):\n  {ev_path}\n  {dw_path}\n"
            "Run scripts/sweep_events_round8.py + scripts/bin_events_round8.py "
            "and scripts/compute_dwell_grid.py first."
        )

    ev = xr.open_zarr(ev_path, consolidated=False)
    dw = xr.open_zarr(dw_path, consolidated=False)
    log.info("event-time : %s", ev_path.name)
    log.info("dwell      : %s", dw_path.name)
    log.info("bands      : %s", ev.attrs.get("bands"))

    lt_centers = ev["local_time"].values
    lat_centers = ev["magnetic_latitude"].values

    # Magnetosphere-only denominator: FLR is a closed-field-line phenomenon
    # and the small minority of sheath/SW events would dilute the latitude
    # signature. The supplementary region figure already documents the
    # MS vs sheath split.
    dw_mag = dw["magnetosphere"].values  # (r, lat, LT) minutes

    use_paper_style()
    fig, axes = plt.subplots(
        1,
        len(LT_SECTORS),
        figsize=(16, 4.8),
        sharey=True,
        constrained_layout=True,
    )

    for ax, (center, hw, sector_label) in zip(axes, LT_SECTORS, strict=True):
        for band in BANDS:
            ev_var = f"{band}_magnetosphere"
            if ev_var not in ev:
                log.warning("missing %s in event-time zarr", ev_var)
                continue
            ev_3d = ev[ev_var].values
            ratio, lo, hi, lat_b = _sector_ratio(
                ev_3d,
                dw_mag,
                lt_centers,
                lat_centers,
                center,
                hw,
            )
            color = BAND_COLORS[band]
            ax.plot(lat_b, ratio, color=color, lw=1.7, label=band)
            ax.fill_between(lat_b, lo, hi, color=color, alpha=0.18)

        ax.set_title(sector_label, fontsize=11)
        ax.set_xlim(-90, 90)
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_ylim(0, None)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Event time / dwell time")
    fig.supxlabel("Magnetic latitude (KSM, offset dipole) [deg]", fontsize=11)
    axes[-1].legend(fontsize=10, frameon=False, loc="upper right")

    n_events = ev.attrs.get("n_events", "?")
    fig.suptitle(
        f"Figure 7 (round-8) — QP occurrence vs magnetic latitude  "
        f"(n={n_events} events; magnetosphere-only dwell; "
        f"{int(MIN_DWELL_MINUTES_LATITUDE / 60)} h dwell floor; "
        f"Poisson 16-84% band)",
        fontsize=11,
    )

    out = ensure_figures_dir() / "figure7_round8.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
