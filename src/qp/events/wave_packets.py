"""Wave packet separation time analysis (Figure 9).

Computes the distribution of time intervals between consecutive
wave packet peaks, revealing the ~10.7 hour PPO modulation.

Extracted from cassinilib/PlotFFT.py:calculateEventSeparation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from qp.events.catalog import WavePacketPeak


def compute_separations(
    packets: list[WavePacketPeak],
    max_separation_hours: float = 48.0,
) -> np.ndarray:
    """Compute time separations between consecutive wave packet peaks.

    Parameters
    ----------
    packets : list of WavePacketPeak
        Detected wave packets, assumed sorted by peak_time.
    max_separation_hours : float
        Ignore separations larger than this (likely data gaps).

    Returns
    -------
    separations : ndarray
        Separation times in hours.
    """
    if len(packets) < 2:
        return np.array([])

    # Sort by peak time
    sorted_packets = sorted(packets, key=lambda p: p.peak_time)

    seps = []
    for i in range(1, len(sorted_packets)):
        dt_sec = (
            sorted_packets[i].peak_time - sorted_packets[i - 1].peak_time
        ).total_seconds()
        dt_hours = dt_sec / 3600
        if 0 < dt_hours <= max_separation_hours:
            seps.append(dt_hours)

    return np.array(seps)


def separation_statistics(separations: ArrayLike) -> dict[str, float]:
    """Compute summary statistics of wave train separations.

    Returns dict with median, mean, std, and mode estimate.
    """
    seps = np.asarray(separations)
    if len(seps) == 0:
        return {"median": 0, "mean": 0, "std": 0, "count": 0}

    return {
        "median": float(np.median(seps)),
        "mean": float(np.mean(seps)),
        "std": float(np.std(seps)),
        "count": len(seps),
    }


def separation_histogram(
    separations: ArrayLike,
    bin_width_hours: float = 0.5,
    max_hours: float = 36.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute histogram and probability distribution of separations.

    Returns (bin_centers, counts, probability_density).
    """
    seps = np.asarray(separations)
    bins = np.arange(0, max_hours + bin_width_hours, bin_width_hours)

    counts, edges = np.histogram(seps, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Probability density (normalized so integral = 1)
    total = np.sum(counts) * bin_width_hours
    pdf = counts / total if total > 0 else counts.astype(float)

    return centers, counts, pdf
