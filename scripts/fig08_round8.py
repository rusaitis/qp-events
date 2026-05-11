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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from qp.events.bands import QP_BAND_NAMES, get_band  # noqa: E402
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
#: noise-dominated and mask.
MIN_DWELL_MINUTES = 600.0  # 10 hours

#: Gaussian smoothing scale, in cells. With dlat = 1 deg and dlt = 0.25 h
#: this corresponds to ~2 deg latitude / 0.5 h LT.
SMOOTHING_SIGMA = 2.0


def _smooth(arr: np.ndarray, sigma: float = SMOOTHING_SIGMA) -> np.ndarray:
    """Gaussian-smooth a 2D array, ignoring NaN cells."""
    from scipy.ndimage import gaussian_filter

    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    sm = gaussian_filter(filled, sigma=sigma)
    wt = gaussian_filter(valid.astype(float), sigma=sigma)
    out = np.where(wt > 0, sm / wt, np.nan)
    out[~valid] = np.nan
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import xarray as xr

    ev_path = _PROJECT_ROOT / "Output" / "event_time_grid_round8.zarr"
    dw_path = _PROJECT_ROOT / "Output" / "dwell_grid_cassini_saturn.zarr"
    if not ev_path.exists() or not dw_path.exists():
        raise SystemExit(
            f"Missing input(s):\n  {ev_path}\n  {dw_path}"
        )

    ev = xr.open_zarr(ev_path, consolidated=False)
    dw = xr.open_zarr(dw_path, consolidated=False)
    log.info("event-time : %s", ev_path.name)
    log.info("dwell      : %s", dw_path.name)

    lt = ev["local_time"].values
    inv_lat = ev["kmag_inv_lat"].values
    dw_kmag = dw["kmag_inv_lat_closed_magnetosphere"].values  # (inv_lat, LT)

    log.info("dwell coverage (closed MS): %.0f h",
             float(dw_kmag.sum()) / 60.0)

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
    finite = np.concatenate([
        r[np.isfinite(r)] for r in ratios.values()
    ])
    vmax = float(np.percentile(finite, 99)) if finite.size else 0.1
    vmin = 0.0

    use_paper_style()
    fig, axes = plt.subplots(
        1, len(BANDS), figsize=(5 * len(BANDS), 5), sharey=True,
        constrained_layout=True,
    )

    images = []
    for ax, band in zip(axes, BANDS, strict=True):
        Z = np.ma.masked_invalid(ratios[band])
        im = ax.pcolormesh(
            lt, inv_lat, Z, shading="auto",
            cmap="plasma", vmin=vmin, vmax=vmax,
        )
        images.append(im)
        ax.set_title(
            f"{BAND_LABELS[band]}  ({n_event_min[band]/60:.0f} event-h)",
            fontsize=11,
        )
        ax.set_xlabel("Local time [h]")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylim(-90, 90)
        ax.axhline(0, color="white", lw=0.4, ls=":", alpha=0.5)
        for x in (6, 12, 18):
            ax.axvline(x, color="white", lw=0.3, ls=":", alpha=0.35)

    axes[0].set_ylabel("KMAG invariant latitude [deg]")
    cbar = fig.colorbar(images[1], ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Event time / dwell time  (closed-field MS)")

    n_events = ev.attrs.get("n_events", "?")
    fig.suptitle(
        f"Figure 8 (round-8) — QP occurrence in KMAG inv. lat × LT  "
        f"(n={n_events} events; closed-field MS; "
        f"{int(MIN_DWELL_MINUTES/60)} h dwell floor; sigma={SMOOTHING_SIGMA} smooth)",
        fontsize=11,
    )

    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure8_round8.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
