"""3D orbit and field line visualization.

Extracted from cassinilib/Plot.py — Saturn sphere, rings, orbit paths,
and field line overlays for figures like Fig 2a and Fig 3.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from numpy.typing import ArrayLike


def draw_sphere(
    ax: plt.Axes,
    center: tuple[float, float, float] = (0, 0, 0),
    radius: float = 1.0,
    color: str = "#e8d282",
    alpha: float = 0.3,
    n_points: int = 30,
) -> None:
    """Draw a sphere (e.g., Saturn) on a 3D axes."""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True)


def draw_ring(
    ax: plt.Axes,
    inner_radius: float = 1.5,
    outer_radius: float = 2.3,
    color: str = "#c8a832",
    alpha: float = 0.2,
    n_points: int = 60,
) -> None:
    """Draw Saturn's rings on a 3D axes."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    r = np.linspace(inner_radius, outer_radius, 5)
    T, R = np.meshgrid(theta, r)
    x = R * np.cos(T)
    y = R * np.sin(T)
    z = np.zeros_like(x)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=False)


def draw_field_line(
    ax: plt.Axes,
    xyz: ArrayLike,
    color: str = "white",
    lw: float = 1.0,
    alpha: float = 0.7,
) -> None:
    """Draw a 3D field line from traced coordinates.

    Parameters
    ----------
    xyz : array_like, shape (N, 3)
        Traced field line coordinates in Cartesian.
    """
    xyz = np.asarray(xyz)
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=lw, alpha=alpha)


def plot_orbit_lt_mlat(
    ax: plt.Axes,
    local_time: ArrayLike,
    mag_lat: ArrayLike,
    location_codes: ArrayLike | None = None,
    xlabel: str = "Local Time [h]",
    ylabel: str = "Magnetic Latitude [deg]",
    s: float = 1.0,
) -> None:
    """Plot Cassini orbit in local time vs magnetic latitude (Fig 2a style).

    Parameters
    ----------
    location_codes : array_like, optional
        0=MS (blue), 1=SH (orange), 2=SW (red). Used for point coloring.
    """
    from qp.plotting.style import LOC_COLORS

    lt = np.asarray(local_time)
    mlat = np.asarray(mag_lat)

    if location_codes is not None:
        codes = np.asarray(location_codes)
        for code, color in LOC_COLORS.items():
            mask = codes == code
            if np.any(mask):
                from qp.plotting.style import LOC_LABELS

                ax.scatter(
                    lt[mask],
                    mlat[mask],
                    c=color,
                    s=s,
                    label=LOC_LABELS.get(code, ""),
                    alpha=0.5,
                    edgecolors="none",
                )
    else:
        ax.scatter(lt, mlat, c="blue", s=s, alpha=0.5, edgecolors="none")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 24)
    ax.legend(loc="upper right", frameon=False, fontsize=9, markerscale=3)
