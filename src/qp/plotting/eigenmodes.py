"""Eigenfrequency plotting for field line resonances (Figure 6 style).

Plots eigenfrequencies vs conjugate latitude with observed QP peaks overlaid.
Extracted from wavesolver/plot.py — only the publication figure functions.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from qp.plotting.style import QP_COLORS


# Mode colors for eigenfrequency curves
MODE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_eigenfrequencies(
    ax: plt.Axes,
    conj_lat: ArrayLike,
    eigenfreqs: dict[int, ArrayLike],
    toroidal: bool = True,
    poloidal: bool = True,
    n_modes: int = 6,
    ylabel: str = "Frequency [mHz]",
    xlabel: str = "Conjugate Latitude [deg]",
) -> None:
    """Plot FLR eigenfrequencies vs conjugate latitude.

    Parameters
    ----------
    conj_lat : array_like
        Conjugate latitude values (degrees).
    eigenfreqs : dict
        {mode_number: freq_array} for each harmonic.
        Toroidal and poloidal can be separate dicts or combined.
    toroidal : bool
        Plot toroidal modes as solid lines.
    poloidal : bool
        Plot poloidal modes as dashed lines.
    """
    conj_lat = np.asarray(conj_lat)

    for mode, freqs in eigenfreqs.items():
        if mode > n_modes:
            continue
        color = MODE_COLORS[mode - 1] if mode <= len(MODE_COLORS) else "white"
        freqs = np.asarray(freqs)
        label = f"m={mode}"

        if toroidal:
            ax.plot(conj_lat, freqs, color=color, ls="-", lw=1.2, label=label)
        if poloidal:
            ax.plot(conj_lat, freqs, color=color, ls="--", lw=0.8, alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def overlay_observed_peaks(
    ax: plt.Axes,
    peak_bands: dict[str, tuple[float, float]],
    lat_range: tuple[float, float] = (72, 76),
) -> None:
    """Overlay observed QP frequency bands as semi-transparent rectangles.

    Parameters
    ----------
    peak_bands : dict
        {name: (freq_low, freq_high)} in mHz.
        e.g. {'QP30': (0.45, 0.65), 'QP60': (0.22, 0.35), 'QP120': (0.10, 0.18)}
    lat_range : tuple
        (lat_min, lat_max) for rectangle extent.

    The widths represent the frequency spread of observed peaks (referee requirement).
    """
    for name, (f_lo, f_hi) in peak_bands.items():
        color = QP_COLORS.get(name, "grey")
        ax.fill_between(
            [lat_range[0], lat_range[1]],
            f_lo,
            f_hi,
            color=color,
            alpha=0.3,
            label=name,
        )
        # Label
        mid_lat = np.mean(lat_range)
        mid_freq = (f_lo + f_hi) / 2
        ax.text(
            mid_lat,
            mid_freq,
            name,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            fontweight="bold",
        )
