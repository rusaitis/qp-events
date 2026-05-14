"""Figure 8 (round-8) — per-band QP occurrence in (KMAG inv. lat, LT).

Reads the round-8 event-time zarr
(``Output/event_time_grid_round8.zarr``) and the canonical Cassini dwell
grid (``Output/dwell_grid_cassini_saturn.zarr``). For each QP band, plots
the event-time / dwell-time ratio on the closed-field-line KMAG
invariant-latitude × local-time grid, smoothed with a Gaussian kernel.

Round-8 supplies a single detection path; this replaces the
phase-8 3 x 3 (band × approach) comparison panel with one cleaner
3-panel row.

Output: ``Output/figures/figure8_round8.png``
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _common import OUTPUT_DIR, ensure_figures_dir, setup_logging  # noqa: E402

from qp.events.bands import QP_BAND_NAMES, get_band  # noqa: E402
from qp.events.normalization import MIN_DWELL_MINUTES_PER_CELL  # noqa: E402
from qp.plotting.maps import smooth_2d_nan_aware  # noqa: E402
from qp.plotting.style import use_paper_style  # noqa: E402

log = logging.getLogger(__name__)

_HARMONIC_HINT = {"QP15": "m=8?", "QP30": "m=6", "QP60": "m=4", "QP120": "m=2"}
BANDS = list(QP_BAND_NAMES)
BAND_LABELS = {
    b: f"{b} (~{int(get_band(b).period_centroid_minutes)} min, "
    f"{_HARMONIC_HINT.get(b, '')})"
    for b in BANDS
}

#: Floor on dwell time per cell. Below this we treat the ratio as
#: noise-dominated and mask. Shared with Fig 7.
MIN_DWELL_MINUTES = MIN_DWELL_MINUTES_PER_CELL

#: Gaussian smoothing scale, in cells. With dlat = 1 deg and dlt = 0.25 h
#: this corresponds to ~2 deg latitude / 0.5 h LT. The smoothing wraps
#: across the midnight seam (LT=0/24): without that, the post-dusk
#: QP30/QP60 peaks near 23-24 LT would not bleed into 0-1 LT and the
#: figure would show an artificial discontinuity at midnight.
SMOOTHING_SIGMA = 2.0


def _smooth(arr: np.ndarray, sigma: float = SMOOTHING_SIGMA) -> np.ndarray:
    """Periodic-LT Gaussian smoothing of an ``(inv_lat, LT)`` grid."""
    return smooth_2d_nan_aware(arr, sigma, periodic_axis=1)


def main() -> None:
    setup_logging()
    import xarray as xr

    ev_path = OUTPUT_DIR / "event_time_grid_round8.zarr"
    dw_path = OUTPUT_DIR / "dwell_grid_cassini_saturn.zarr"
    if not ev_path.exists() or not dw_path.exists():
        raise SystemExit(f"Missing input(s):\n  {ev_path}\n  {dw_path}")

    ev = xr.open_zarr(ev_path, consolidated=False)
    dw = xr.open_zarr(dw_path, consolidated=False)
    log.info("event-time : %s", ev_path.name)
    log.info("dwell      : %s", dw_path.name)

    lt = ev["local_time"].values
    inv_lat = ev["kmag_inv_lat"].values
    # Closed-MS denominator. inv_lat is signed by spacecraft hemisphere
    # (see qp.dwell.grid.accumulate_traced_inv_lat_grid), so the N/S
    # axis here mirrors orbital coverage, not intrinsic asymmetry.
    dw_kmag = dw["kmag_inv_lat_closed_magnetosphere"].values  # (inv_lat, LT)

    log.info("dwell coverage (closed MS): %.0f h", float(dw_kmag.sum()) / 60.0)

    # Compute per-band ratios.
    ratios: dict[str, np.ndarray] = {}
    n_event_min: dict[str, float] = {}
    for band in BANDS:
        ev_var = f"{band}_kmag_inv_lat_closed_magnetosphere"
        if ev_var not in ev:
            log.warning("missing %s — falling back to closed_total", ev_var)
            ev_var = f"{band}_kmag_inv_lat_closed_total"
        ev_2d = ev[ev_var].values  # (inv_lat, LT)
        n_event_min[band] = float(ev_2d.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(dw_kmag >= MIN_DWELL_MINUTES, ev_2d / dw_kmag, np.nan)
        ratios[band] = _smooth(r)

    # Shared color scale across bands so the QP60 vs QP30/QP120 contrast is
    # comparable. p99 of the union of finite ratios sets the upper bound.
    finite = np.concatenate([r[np.isfinite(r)] for r in ratios.values()])
    vmax = float(np.percentile(finite, 99)) if finite.size else 0.1
    vmin = 0.0

    # Cassini's closed-field coverage clusters at |inv_lat| > ~30 deg —
    # the equatorial strip is empty. Split each band into a broken-axis
    # pair so the populated hemispheres dominate the figure.
    use_paper_style()
    fig, axes_grid = plt.subplots(
        2,
        len(BANDS),
        figsize=(5 * len(BANDS), 5.2),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 3], "hspace": 0.06, "wspace": 0.05},
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.06, right=0.92, top=0.9, bottom=0.1, hspace=0.06, wspace=0.05
    )
    axes_north = axes_grid[0]
    axes_south = axes_grid[1]

    images = []
    for ax_n, ax_s, band in zip(axes_north, axes_south, BANDS, strict=True):
        Z = np.ma.masked_invalid(ratios[band])
        for ax in (ax_n, ax_s):
            im = ax.pcolormesh(
                lt,
                inv_lat,
                Z,
                shading="auto",
                cmap="plasma",
                vmin=vmin,
                vmax=vmax,
            )
            images.append(im)
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            for x in (6, 12, 18):
                ax.axvline(x, color="white", lw=0.3, ls=":", alpha=0.35)
        ax_n.set_ylim(40, 90)
        ax_s.set_ylim(-90, -40)
        ax_n.set_title(
            f"{BAND_LABELS[band]}  ({n_event_min[band] / 60:.0f} event-h)",
            fontsize=11,
        )
        ax_s.set_xlabel("Local time [h]")
        # Hide the inner spines + ticks to make the gap read as a break.
        ax_n.spines["bottom"].set_visible(False)
        ax_s.spines["top"].set_visible(False)
        ax_n.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_n.set_yticks([40, 60, 80])
        ax_s.set_yticks([-80, -60, -40])

    # Share y-tick labels only on the leftmost column.
    for ax in list(axes_north[1:]) + list(axes_south[1:]):
        ax.tick_params(axis="y", labelleft=False)

    axes_north[0].set_ylabel("KMAG inv. lat. [deg]")
    axes_south[0].set_ylabel("KMAG inv. lat. [deg]")
    cbar = fig.colorbar(
        images[1], ax=axes_grid.ravel().tolist(), fraction=0.025, pad=0.02
    )
    cbar.set_label("Event time / dwell time  (closed-field MS)")

    n_events = ev.attrs.get("n_events", "?")
    fig.suptitle(
        f"Figure 8 (round-8) — QP occurrence in KMAG inv. lat × LT  "
        f"(n={n_events} events; closed-field MS; "
        f"{int(MIN_DWELL_MINUTES / 60)} h dwell floor; sigma={SMOOTHING_SIGMA} smooth)",
        fontsize=11,
    )

    out = ensure_figures_dir() / "figure8_round8.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
