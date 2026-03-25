"""Figure 6: Eigenfrequencies vs observed QP peaks.

Two panels: noon (12 LT) and midnight (0 LT) showing FLR eigenfrequency
curves (modes 1-6) vs invariant latitude, with observed QP30/QP60/QP120
bands overlaid.

New computed curves use KMAG + Bagenal density via the rewritten wavesolver.
Published reference curves (digitized from the original figure) are overlaid
faintly in the background for comparison.

Referee: rectangle widths must be meaningful (frequency spread of peaks).
"""

import csv
import sys
import types
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]

# Register stubs
for mod_path in ["__main__", "data_sweeper", "mag_fft_sweeper",
                 "cassinilib", "cassinilib.NewSignal"]:
    if mod_path not in sys.modules:
        sys.modules[mod_path] = types.ModuleType(mod_path)
    for cls_name in ["SignalSnapshot", "NewSignal", "Interval", "FFT_list",
                     "WaveSignal", "Wave"]:
        setattr(sys.modules[mod_path], cls_name, type(cls_name, (), {}))

sys.path.insert(0, str(_project_root / "src"))

import qp
from qp.signal.power_ratio import compute_power_ratios

# Mode colors matching the original figure
MODE_COLORS = {
    1: "#1f77b4",  # blue
    2: "#ff69b4",  # pink
    3: "#2ca02c",  # green
    4: "#d62728",  # red
    5: "#9467bd",  # purple
    6: "#8c564b",  # brown
}


def load_published_reference(
    csv_path: Path,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    r"""Load digitized published eigenfrequencies from CSV.

    Returns dict[panel][mode] = (lats_array, freqs_array).
    """
    data: dict[str, dict[int, dict[float, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            panel = row["panel"]
            lat = float(row["inv_lat_deg"])
            mode = int(row["mode"])
            freq = float(row["freq_mhz"])
            data[panel][mode][lat] = freq

    result: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for panel, modes in data.items():
        result[panel] = {}
        for mode, lat_freq in sorted(modes.items()):
            lats = np.array(sorted(lat_freq.keys()))
            freqs = np.array([lat_freq[l] for l in lats])
            result[panel][mode] = (lats, freqs)
    return result


def estimate_observed_bands():
    """Estimate QP peak frequency ranges from the spectral data.

    Uses FWHM of the median power ratio peaks from Figure 5 data.
    The widths represent the frequency range where the peak power
    exceeds 50% of its maximum — a physically meaningful measure.
    """
    # Load MFA data and compute power ratios for a subset to get peak widths
    print("  Estimating QP band widths from MFA data...")
    data = np.load(qp.DATA_PRODUCTS / "Cassini_MAG_MFA_36H.npy", allow_pickle=True)

    # Collect power ratios from magnetospheric segments near midnight
    all_r_perp1 = []
    freq_ref = None
    count = 0
    for seg in data:
        if count >= 500:
            break
        if seg.flag is not None:
            continue
        if not isinstance(seg.info, dict) or seg.info.get("location") != 0:
            continue
        bp1 = seg.FIELDS[1].y
        bp2 = seg.FIELDS[2].y
        bpar = seg.FIELDS[0].y
        btot = seg.FIELDS[3].y
        if bp1 is None or len(bp1) < 1440:
            continue

        pad = 360
        ratios = compute_power_ratios(
            bpar[pad:-pad], bp1[pad:-pad], bp2[pad:-pad], btot[pad:-pad],
            dt=60.0, nperseg=720, noverlap=360, window="hann",
        )
        all_r_perp1.append(ratios["r_perp1"])
        if freq_ref is None:
            freq_ref = ratios["freq"]
        count += 1

    if not all_r_perp1:
        # Fallback: use approximate values from the paper
        return _default_bands()

    median_r = np.median(all_r_perp1, axis=0)
    freq_mhz = freq_ref * 1000

    # Find FWHM for each QP band
    bands = {}
    for name, center_min, search_range_min in [
        ("QP120", 120, (80, 200)),
        ("QP60", 55, (35, 80)),
        ("QP30", 30, (20, 40)),
    ]:
        f_lo = 1000.0 / (search_range_min[1] * 60)
        f_hi = 1000.0 / (search_range_min[0] * 60)
        mask = (freq_mhz >= f_lo) & (freq_mhz <= f_hi)

        if not np.any(mask):
            continue

        peak_val = np.max(median_r[mask])
        half_max = peak_val / 2

        # Find where the ratio exceeds half-max
        above = freq_mhz[mask][median_r[mask] > half_max]
        if len(above) >= 2:
            bands[name] = (above[0], above[-1])  # mHz
        else:
            f_center = 1000.0 / (center_min * 60)
            bands[name] = (f_center * 0.8, f_center * 1.2)

    print(f"  Estimated bands: {bands}")
    return bands


def _default_bands():
    """Fallback band widths from visual estimation of the paper's Figure 5."""
    return {
        "QP30": (0.45, 0.65),   # mHz
        "QP60": (0.22, 0.35),   # mHz
        "QP120": (0.10, 0.18),  # mHz
    }


QP_BAND_COLORS = {"QP30": "grey", "QP60": "#FFD700", "QP120": "#FF69B4"}


def plot_panel(ax, title, eigen_tor, eigen_pol, bands, lat_range=(72, 76),
               inv_lats=None, published_ref=None):
    """Plot one panel of eigenfrequency curves + observed bands."""
    # Overlay published reference curves (subtle grey, behind everything)
    if published_ref is not None:
        for mode in sorted(published_ref.keys()):
            lats, freqs = published_ref[mode]
            ax.plot(lats, freqs, color="grey", lw=1.0, ls="-",
                    alpha=0.35, zorder=1)

    # Plot eigenfrequency curves
    for mode in sorted(eigen_tor.keys()):
        color = MODE_COLORS.get(mode, "#333333")
        lats = inv_lats[:len(eigen_tor[mode])]
        ax.plot(lats, eigen_tor[mode], color=color, lw=2, ls="-",
                label=f"m={mode}" if mode <= 6 else None, zorder=2)
        if mode in eigen_pol and len(eigen_pol[mode]) > 0:
            lats_p = inv_lats[:len(eigen_pol[mode])]
            ax.plot(lats_p, eigen_pol[mode], color=color, lw=1, ls="--",
                    alpha=0.6, zorder=2)

    # Overlay observed QP bands as rectangles
    for name, (f_lo, f_hi) in bands.items():
        color = QP_BAND_COLORS[name]
        rect = mpatches.Rectangle(
            (lat_range[0], f_lo), lat_range[1] - lat_range[0], f_hi - f_lo,
            color=color, alpha=0.4, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            np.mean(lat_range), (f_lo + f_hi) / 2, name,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=color, zorder=4,
        )

    # Formatting
    ax.set_yscale("log")
    ax.set_ylim(0.02, 2)
    ax.set_xlim(63, 77)
    ax.set_xlabel("Invariant Latitude (degrees)", fontsize=13)
    ax.set_ylabel("Eigenfrequency (mHz)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3, ls="--")

    # Secondary x-axis: equatorial crossing distance
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    eq_ticks = [5, 10, 15, 20, 25]
    eq_lats = [90 - np.degrees(np.arcsin(1 / np.sqrt(L))) for L in eq_ticks]
    ax2.set_xticks(eq_lats)
    ax2.set_xticklabels([str(L) for L in eq_ticks])
    ax2.set_xlabel("Equatorial Crossing Distance ($R_S$)", fontsize=11)
    ax2.tick_params(labelsize=10)

    # Secondary y-axis: period in minutes
    ax3 = ax.twinx()
    ax3.set_yscale("log")
    ax3.set_ylim(ax.get_ylim())
    period_ticks_min = [5, 10, 30, 60, 90, 120, 300, 600]
    period_ticks_mhz = [1000 / (t * 60) for t in period_ticks_min]
    ax3.set_yticks(period_ticks_mhz)
    ax3.set_yticklabels([f"{t} min" if t < 120 else f"{t//60} h"
                         for t in period_ticks_min])
    ax3.set_ylabel("Period", fontsize=11)
    ax3.tick_params(labelsize=9)


def compute_eigenfrequencies_dynamic(
    local_time_hours: float = 12.0,
    use_kmag: bool = True,
    n_modes: int = 6,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compute eigenfrequency curves dynamically using the new wavesolver.

    Parameters
    ----------
    local_time_hours : float
        Local time (0=midnight, 12=noon).
    use_kmag : bool
        Use KMAG field model (True) or dipole (False).
    n_modes : int
        Number of modes to compute.

    Returns
    -------
    inv_lats : ndarray
        Invariant latitudes (degrees).
    eigen_tor : dict
        Toroidal eigenfrequencies (mHz) by mode number.
    eigen_pol : dict
        Poloidal eigenfrequencies (mHz) by mode number.
    """
    from qp.wavesolver.solver import WavesolverConfig, solve_for_latitude_range
    from qp.fieldline.kmag_model import SaturnField

    field = SaturnField() if use_kmag else None
    print(f"  Computing eigenfrequencies at {local_time_hours}h LT "
          f"({'KMAG' if use_kmag else 'dipole'})...")

    eigen_tor: dict[int, list[float]] = {m: [] for m in range(1, n_modes + 1)}
    eigen_pol: dict[int, list[float]] = {m: [] for m in range(1, n_modes + 1)}
    inv_lats_list: list[float] = []

    for component, eigen_dict in [("toroidal", eigen_tor), ("poloidal", eigen_pol)]:
        config = WavesolverConfig(
            component=component,
            n_modes=n_modes,
            field=field,
            local_time_hours=local_time_hours,
            freq_range=(1e-4, 0.008),
            resolution=150,
        )
        results = solve_for_latitude_range(config, lat_min=63, lat_max=76, n_fieldlines=25)

        for result in results:
            if component == "toroidal":
                inv_lats_list.append(np.degrees(np.arccos(1.0 / np.sqrt(result.l_shell))))
            for mode in result.modes:
                if mode.mode_number <= n_modes:
                    eigen_dict[mode.mode_number].append(mode.frequency_mhz)

    inv_lats = np.array(inv_lats_list)
    eigen_tor_arr = {m: np.array(v) for m, v in eigen_tor.items() if v}
    eigen_pol_arr = {m: np.array(v) for m, v in eigen_pol.items() if v}

    return inv_lats, eigen_tor_arr, eigen_pol_arr


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Figure 6: Eigenfrequencies")
    parser.add_argument("--compute", action="store_true",
                        help="Compute eigenfrequencies dynamically instead of using lookup tables")
    parser.add_argument("--dipole", action="store_true",
                        help="Use dipole field model instead of KMAG (with --compute)")
    args = parser.parse_args()

    print("Estimating observed QP band widths...")
    bands = estimate_observed_bands()
    if not bands:
        bands = _default_bands()

    # Load published reference curves (digitized from paper/figure6.jpeg)
    ref_csv = Path("output/published_eigenfrequencies.csv")
    published = load_published_reference(ref_csv) if ref_csv.exists() else {}
    ref_noon = published.get("noon")
    ref_midnight = published.get("midnight")

    if not args.compute:
        print("ERROR: --compute is required (no hardcoded lookup tables)")
        print("Usage: uv run python scripts/fig06_eigenfrequencies.py --compute")
        sys.exit(1)

    inv_lats_noon, eigen_tor_noon, eigen_pol_noon = compute_eigenfrequencies_dynamic(
        local_time_hours=12.0, use_kmag=not args.dipole,
    )
    inv_lats_mid, eigen_tor_mid, eigen_pol_mid = compute_eigenfrequencies_dynamic(
        local_time_hours=0.0, use_kmag=not args.dipole,
    )

    # --- Plot ---
    plt.style.use("default")
    plt.rcParams.update({"font.size": 14})

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.90, top=0.88, bottom=0.12)

    plot_panel(ax_a, "Field Lines at 12 LT", eigen_tor_noon, eigen_pol_noon,
               bands, lat_range=(72, 76), inv_lats=inv_lats_noon,
               published_ref=ref_noon)
    ax_a.text(0.02, 0.95, "a", transform=ax_a.transAxes, fontsize=18,
              fontweight="bold", va="top")

    plot_panel(ax_b, "Field Lines at 0 LT", eigen_tor_mid, eigen_pol_mid,
               bands, lat_range=(72, 74), inv_lats=inv_lats_mid,
               published_ref=ref_midnight)
    ax_b.text(0.02, 0.95, "b", transform=ax_b.transAxes, fontsize=18,
              fontweight="bold", va="top")

    # Legend
    handles_model = [plt.Line2D([0], [0], color=MODE_COLORS[m], lw=2, label=f"m={m}")
                     for m in range(1, 7)]
    handles_model.append(
        plt.Line2D([0], [0], color="grey", lw=1.0, alpha=0.35, label="Published")
    )
    handles_obs = [mpatches.Patch(color=QP_BAND_COLORS[n], alpha=0.4, label=n)
                   for n in ["QP30", "QP60", "QP120"]]

    ax_a.legend(handles=handles_model, loc="upper right", title="MODELED",
                frameon=True, fontsize=9, title_fontsize=10)
    ax_b.legend(handles=handles_obs, loc="center right", title="OBSERVED",
                frameon=True, fontsize=9, title_fontsize=10)

    Path("output").mkdir(exist_ok=True)
    suffix = "_computed" if args.compute else ""
    plt.savefig(f"output/figure6{suffix}.png", dpi=300, bbox_inches="tight")
    print(f"Saved output/figure6{suffix}.png")
    plt.close()


if __name__ == "__main__":
    main()
