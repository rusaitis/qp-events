"""Supplementary figure: equatorial event/dwell ratio per QP band.

For each round-8 detection, the spacecraft position is traced along the KMAG
field line to its equatorial apex (max-r along the trace). Both the event
time and the spacecraft dwell time are accumulated on the same
``(kmag_eq_r, local_time)`` grid (only closed field lines), and we plot the
ratio per QP band as a polar heatmap viewed from Saturn's north pole. This
is the equatorial-plane signature of where each band's resonance is most
active relative to where Cassini actually spent time.

Outputs (overwriting):
    Output/figures/sup_event_dwell_polar.png

Usage::

    uv run python scripts/sup_event_dwell_polar.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import qp  # noqa: E402
from qp.events.bands import QP_BAND_NAMES, get_band  # noqa: E402
from qp.plotting.style import use_paper_style  # noqa: E402

log = logging.getLogger(__name__)

_HARMONIC_LABEL = {
    "QP15": "m=8? (toroidal)",
    "QP30": "m=6 (toroidal)",
    "QP60": "m=4 (toroidal)",
    "QP120": "m=2 (toroidal)",
}

# Bands to plot (rendered left to right) and their period mid-points for labels.
BANDS = [
    (
        b,
        f"{int(get_band(b).period_centroid_minutes)} min",
        _HARMONIC_LABEL.get(b, ""),
    )
    for b in QP_BAND_NAMES
]


def _polar_pcolormesh(ax, lt_centers: np.ndarray, L_edges: np.ndarray, Z: np.ndarray, *,
                      norm, cmap: str = "inferno") -> object:
    """Polar pcolormesh with noon at top, dawn on right (view from N pole)."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # LT bin width: assume uniform spacing.
    dlt = float(lt_centers[1] - lt_centers[0]) if len(lt_centers) > 1 else 0.25
    lt_edges = np.concatenate([lt_centers - dlt / 2, [lt_centers[-1] + dlt / 2]])
    theta_edges = ((lt_edges - 12.0) / 24.0) * 2.0 * np.pi

    Theta, R = np.meshgrid(theta_edges, L_edges)
    return ax.pcolormesh(Theta, R, Z, norm=norm, cmap=cmap, shading="auto")


def _format_polar(ax, *, L_max: float) -> None:
    ax.set_rlim(0, L_max)
    rticks = [r for r in (10, 20, 30, 40, 50, 60) if r <= L_max]
    ax.set_rticks(rticks)
    # 135 deg keeps the radial labels in the upper-left arc, away from the
    # dawn/dusk LT labels at 90 / 270 deg.
    ax.set_rlabel_position(135)
    ax.grid(color="white", alpha=0.25)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["12", "06", "00", "18"])
    ax.tick_params(axis="x", colors="white", pad=1, labelsize=10)
    ax.tick_params(axis="y", colors="white", labelsize=8)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import xarray as xr

    out_dir = qp.OUTPUT_DIR
    ev_path = out_dir / "event_time_grid_round8.zarr"
    eq_path = out_dir / "dwell_grid_kmag_eq_r.zarr"
    if not ev_path.exists() or not eq_path.exists():
        raise SystemExit(
            f"Required zarrs not found:\n  {ev_path}\n  {eq_path}\n"
            "Run scripts/bin_events_round8.py --with-kmag-trace and "
            "scripts/compute_dwell_grid.py --equatorial-only first."
        )

    ev = xr.open_zarr(ev_path)
    dw = xr.open_zarr(eq_path)
    log.info("event-time zarr : %s", ev_path)
    log.info("dwell sibling   : %s", eq_path)

    if "kmag_eq_r_closed_total" not in dw:
        raise SystemExit("dwell zarr lacks kmag_eq_r_closed_total — re-run compute_dwell_grid.py")
    den = dw["kmag_eq_r_closed_total"]  # (kmag_eq_r, local_time), minutes
    L_native = den["kmag_eq_r"].values
    lt_native = den["local_time"].values

    # Coarsen the LT axis from native 15-min cadence to 1-hour bins. The
    # native L axis (1 R_S per bin) is fine.
    n_lt_native = len(lt_native)
    coarsen_factor = max(1, n_lt_native // 24)
    n_lt_native_kept = (n_lt_native // coarsen_factor) * coarsen_factor
    den_native = den.values[:, :n_lt_native_kept]
    den_arr_full = den_native.reshape(
        den_native.shape[0], n_lt_native_kept // coarsen_factor, coarsen_factor
    ).sum(axis=2)
    lt_centers = lt_native[:n_lt_native_kept].reshape(-1, coarsen_factor).mean(axis=1)

    # L bin edges from native centers.
    dL = float(L_native[1] - L_native[0]) if len(L_native) > 1 else 1.0
    L_edges_full = np.concatenate([L_native - dL / 2, [L_native[-1] + dL / 2]])

    # Visual cutoff — Cassini's closed-field equatorial coverage drops off
    # past ~30 R_S; further out the dwell denominator is small and ratios
    # become noisy.
    L_max = 30.0
    keep_L = L_native <= L_max
    L_centers = L_native[keep_L]
    L_edges = L_edges_full[: keep_L.sum() + 1]
    den_arr = den_arr_full[keep_L]  # (n_L, n_lt_coarse)

    # Mask bins with little dwell. With 1-hour LT bins and 1 R_S L bins,
    # 3 hours is a reasonable floor (the bin saw at least three orbits' worth
    # of closed-field samples).
    min_dwell_minutes = 180.0
    insufficient = den_arr < min_dwell_minutes

    # Compute ratios per band (closed-field only, region-summed total).
    ratios: dict[str, np.ndarray] = {}
    n_events_by_band: dict[str, int] = {}
    for band, *_ in BANDS:
        var = f"{band}_kmag_eq_r_closed_total"
        if var not in ev:
            log.warning("event zarr missing %s — using zeros", var)
            num_arr = np.zeros_like(den_arr)
        else:
            n_full = ev[var].values[:, :n_lt_native_kept]
            num_arr_full = n_full.reshape(
                n_full.shape[0], n_lt_native_kept // coarsen_factor, coarsen_factor
            ).sum(axis=2)
            num_arr = num_arr_full[keep_L]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den_arr > 0, num_arr / den_arr, np.nan)
        r[insufficient] = np.nan
        ratios[band] = r
        # Total event-minutes (across the kept L range and all LT)
        n_events_by_band[band] = int(np.nansum(num_arr))

    # Pick a shared log color range so panels are directly comparable.
    finite = np.concatenate([r[np.isfinite(r) & (r > 0)].ravel() for r in ratios.values()])
    if finite.size == 0:
        raise SystemExit("All ratios are NaN — nothing to plot.")
    vmin = float(np.percentile(finite, 5))
    vmax = float(np.percentile(finite, 99))
    norm = LogNorm(vmin=max(vmin, 1e-4), vmax=vmax)

    use_paper_style()
    n_bands = len(BANDS)
    fig = plt.figure(figsize=(5.3 * n_bands, 6.5))
    fig.set_facecolor(plt.rcParams["figure.facecolor"])
    fig.subplots_adjust(left=0.04, right=0.88, top=0.86, bottom=0.10, wspace=0.25)

    images = []
    for i, (band, period_label, mode_label) in enumerate(BANDS):
        ax = fig.add_subplot(1, n_bands, i + 1, projection="polar",
                             facecolor=plt.rcParams["axes.facecolor"])
        Z = np.ma.masked_invalid(ratios[band])
        im = _polar_pcolormesh(ax, lt_centers, L_edges, Z, norm=norm)
        _format_polar(ax, L_max=L_max)
        ax.set_title(
            f"{band}  (~{period_label})\n{mode_label}",
            color="white", pad=18, fontsize=13,
        )
        images.append(im)

    # Single shared colorbar in the right margin we reserved via subplots_adjust.
    cbar_ax = fig.add_axes((0.90, 0.15, 0.015, 0.65))
    cbar = fig.colorbar(images[1], cax=cbar_ax,
                        label="event-time / dwell-time ratio  (closed field lines)")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # Annotate denominator threshold and L-cutoff at the bottom.
    fig.text(
        0.46, 0.02,
        f"LT in hours (12=noon, 06=dawn, 00=midnight, 18=dusk).  "
        f"L = KMAG-traced equatorial apex r (R$_S$);  "
        f"L $\\leq$ {L_max:.0f}; bins with dwell < {min_dwell_minutes:.0f} min masked.",
        ha="center", color="white", fontsize=10,
    )
    fig.suptitle(
        "Equatorial event/dwell ratio per QP band  "
        "(field-line traced to magnetic equator)",
        color="white", fontsize=14, y=0.98,
    )

    out = qp.OUTPUT_DIR / "figures" / "sup_event_dwell_polar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    log.info("wrote %s", out)

    # Quick numerical summary printed alongside.
    print()
    print(f"Equatorial event/dwell ratio (closed field lines, L <= {L_max} R_S):")
    for band, *_ in BANDS:
        r = ratios[band]
        finite_r = r[np.isfinite(r)]
        if finite_r.size == 0:
            print(f"  {band}: no finite ratios")
            continue
        # LT-averaged ratio vs L (peak)
        with np.errstate(invalid="ignore"):
            agg = np.nanmean(r, axis=1)
        peak_i = int(np.nanargmax(agg))
        print(f"  {band}: peak LT-avg ratio = {agg[peak_i]:.3f} at L = {L_centers[peak_i]:.1f} R_S "
              f"(median ratio = {float(np.nanmedian(finite_r)):.3f})")


if __name__ == "__main__":
    main()
