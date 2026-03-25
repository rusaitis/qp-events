"""Saturn 3D visualization: axes, vectors, reference shells, movie rendering.

Extracted from ``cassinilib/Plot.py`` — 3D helpers not already in
``orbits.py`` (``draw_sphere``, ``draw_ring``, ``draw_field_line``).
"""

from __future__ import annotations

import logging
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d  # noqa: F401

log = logging.getLogger(__name__)


# ============================================================================
# 3D Arrow (used by draw_axes_3d and draw_vector_3d)
# ============================================================================


class Arrow3D(FancyArrowPatch):
    """A 3D arrow patch for matplotlib 3D axes.

    Extracted from ``cassinilib/Plot.py:Arrow3D``.
    """

    def __init__(
        self,
        xs: list[float],
        ys: list[float],
        zs: list[float],
        *args,
        **kwargs,
    ):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


# ============================================================================
# 3D drawing helpers
# ============================================================================


def draw_axes_3d(
    ax: plt.Axes,
    length: float | None = None,
    color: str = "white",
    alpha: float = 0.8,
    lw: float = 2,
    mutation_scale: float = 8,
    labels: tuple[str, str, str] = (r"X ($R_S$)", r"Y ($R_S$)", r"Z ($R_S$)"),
) -> None:
    r"""Draw 3D coordinate axes with arrowheads.

    Replaces ``cassinilib/Plot.py:PlotAxes()``.

    Parameters
    ----------
    ax : Axes
        3D matplotlib axes.
    length : float, optional
        Half-length of each axis. Default: inferred from current limits.
    """
    if length is None:
        length = max(
            abs(ax.get_xlim3d()[1]),
            abs(ax.get_ylim3d()[1]),
            abs(ax.get_zlim3d()[1]),
        )

    kw = dict(
        mutation_scale=mutation_scale, lw=lw, arrowstyle="-|>", color=color, alpha=alpha
    )
    ax.add_artist(Arrow3D([-length, length], [0, 0], [0, 0], **kw))
    ax.add_artist(Arrow3D([0, 0], [-length, length], [0, 0], **kw))
    ax.add_artist(Arrow3D([0, 0], [0, 0], [-length, length], **kw))


def draw_vector_3d(
    ax: plt.Axes,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    color: str = "black",
    lw: float = 2,
    alpha: float = 1.0,
    arrowstyle: str = "-|>",
    label: str | None = None,
    fontsize: float = 14,
) -> None:
    r"""Draw a 3D vector arrow with optional text label.

    Replaces ``cassinilib/Plot.py:addVector3D()``.
    """
    o = np.asarray(origin)
    d = np.asarray(direction)
    tip = o + d

    arrow = Arrow3D(
        [o[0], tip[0]],
        [o[1], tip[1]],
        [o[2], tip[2]],
        mutation_scale=10,
        lw=lw,
        arrowstyle=arrowstyle,
        alpha=alpha,
        color=color,
    )
    ax.add_artist(arrow)

    if label is not None:
        d_unit = d / (np.linalg.norm(d) + 1e-30)
        margin = abs(ax.get_xlim3d()[1] - ax.get_xlim3d()[0]) * 0.02
        label_pos = tip + d_unit * margin
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            label,
            fontsize=fontsize,
            color=color,
            ha="center",
            va="center",
        )


def equalize_3d_axes(ax: plt.Axes, radius: float | None = None) -> None:
    r"""Make 3D axes have equal scale.

    Replaces ``cassinilib/Plot.py:setAxesEqual()`` and ``setAxesRadius()``.

    Parameters
    ----------
    ax : Axes
        3D matplotlib axes.
    radius : float, optional
        Half-width for all axes. Default: computed from current limits.
    """
    if radius is None:
        limits = np.array(
            [
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ]
        )
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    else:
        origin = np.zeros(3)

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


# ============================================================================
# Reference field-line shell data (KMAG traces)
# ============================================================================

# Digitized from KMAG field line traces: (local_time, conjugate_latitude)
# for north and south hemispheres at various L-shells and moon orbits.
REFERENCE_SHELLS: dict[str, dict[str, list[float]]] = {
    "10Rs": {
        "north_lt": [
            0.41,
            1.24,
            2.07,
            2.9,
            3.73,
            4.56,
            5.39,
            6.22,
            7.05,
            7.88,
            8.71,
            9.53,
            10.36,
            11.18,
            12.01,
            12.01,
            12.83,
            13.66,
            14.48,
            15.3,
            16.13,
            16.96,
            17.78,
            18.61,
            19.43,
            20.26,
            21.09,
            21.92,
            22.75,
            23.58,
        ],
        "north_lat": [
            73.08,
            73.11,
            73.16,
            73.09,
            73.15,
            73.21,
            73.13,
            73.12,
            73.14,
            73.28,
            73.27,
            73.24,
            73.21,
            73.32,
            73.3,
            73.35,
            73.34,
            73.34,
            73.2,
            73.21,
            73.21,
            73.2,
            73.19,
            73.2,
            73.16,
            73.13,
            73.09,
            73.06,
            73.19,
            73.05,
        ],
        "south_lt": [
            0.42,
            1.25,
            2.08,
            2.91,
            3.74,
            4.56,
            5.39,
            6.22,
            7.04,
            7.87,
            8.69,
            9.52,
            10.34,
            11.17,
            11.99,
            11.99,
            12.82,
            13.64,
            14.47,
            15.29,
            16.12,
            16.95,
            17.78,
            18.61,
            19.44,
            20.27,
            21.1,
            21.93,
            22.76,
            23.59,
        ],
        "south_lat": [
            -71.31,
            -71.25,
            -71.27,
            -71.26,
            -71.35,
            -71.4,
            -71.42,
            -71.33,
            -71.33,
            -71.48,
            -71.51,
            -71.43,
            -71.41,
            -71.46,
            -71.53,
            -71.44,
            -71.45,
            -71.43,
            -71.48,
            -71.43,
            -71.41,
            -71.4,
            -71.4,
            -71.33,
            -71.29,
            -71.36,
            -71.24,
            -71.36,
            -71.3,
            -71.33,
        ],
        "label": r"10 $R_S$",
        "color": "#84ff3d",
        "ls": "-",
    },
    "20Rs": {
        "north_lt": [
            0.37,
            1.21,
            2.05,
            2.9,
            3.74,
            4.59,
            5.45,
            6.3,
            7.15,
            7.99,
            8.83,
            9.66,
            10.47,
            11.26,
            12.04,
            12.04,
            12.82,
            13.6,
            14.4,
            15.22,
            16.04,
            16.87,
            17.7,
            18.54,
            19.37,
            20.2,
            21.03,
            21.87,
            22.7,
            23.53,
        ],
        "north_lat": [
            75.34,
            75.37,
            75.28,
            75.41,
            75.45,
            75.59,
            75.64,
            75.83,
            75.97,
            76.22,
            76.42,
            76.73,
            76.87,
            76.98,
            77.03,
            77.05,
            77.06,
            76.85,
            76.77,
            76.45,
            76.28,
            76.1,
            75.84,
            75.69,
            75.59,
            75.55,
            75.4,
            75.33,
            75.38,
            75.34,
        ],
        "south_lt": [
            0.46,
            1.3,
            2.13,
            2.96,
            3.8,
            4.63,
            5.46,
            6.29,
            7.13,
            7.96,
            8.78,
            9.6,
            10.39,
            11.18,
            11.96,
            11.96,
            12.74,
            13.53,
            14.34,
            15.17,
            16.01,
            16.85,
            17.7,
            18.56,
            19.41,
            20.26,
            21.1,
            21.95,
            22.79,
            23.63,
        ],
        "south_lat": [
            -73.68,
            -73.76,
            -73.75,
            -73.87,
            -73.98,
            -74.04,
            -74.24,
            -74.37,
            -74.61,
            -74.8,
            -75.04,
            -75.29,
            -75.49,
            -75.58,
            -75.65,
            -75.63,
            -75.64,
            -75.44,
            -75.29,
            -75.05,
            -74.76,
            -74.47,
            -74.32,
            -74.17,
            -73.98,
            -73.91,
            -73.76,
            -73.71,
            -73.79,
            -73.78,
        ],
        "label": r"20 $R_S$",
        "color": "#ffec3d",
        "ls": "--",
    },
    "25Rs": {
        "north_lt": [
            0.34,
            1.19,
            2.04,
            2.9,
            3.76,
            4.63,
            5.5,
            6.39,
            7.29,
            8.19,
            9.1,
            10.01,
            10.87,
            11.57,
            12.1,
            12.1,
            12.61,
            13.27,
            14.09,
            14.97,
            15.85,
            16.74,
            17.61,
            18.47,
            19.32,
            20.16,
            20.99,
            21.83,
            22.66,
            23.5,
        ],
        "north_lat": [
            75.69,
            75.71,
            75.83,
            75.95,
            76.05,
            76.17,
            76.48,
            76.76,
            77.21,
            77.7,
            78.33,
            79.1,
            79.91,
            80.63,
            80.65,
            80.7,
            80.64,
            80.03,
            79.21,
            78.46,
            77.83,
            77.29,
            76.87,
            76.55,
            76.27,
            76.09,
            75.98,
            75.79,
            75.79,
            75.71,
        ],
        "south_lt": [
            0.5,
            1.33,
            2.17,
            3.0,
            3.84,
            4.68,
            5.53,
            6.39,
            7.26,
            8.14,
            9.03,
            9.91,
            10.73,
            11.38,
            11.9,
            11.9,
            12.43,
            13.13,
            13.99,
            14.9,
            15.81,
            16.72,
            17.61,
            18.5,
            19.37,
            20.24,
            21.1,
            21.96,
            22.81,
            23.65,
        ],
        "south_lat": [
            -74.17,
            -74.29,
            -74.28,
            -74.43,
            -74.62,
            -74.81,
            -75.06,
            -75.41,
            -75.92,
            -76.52,
            -77.21,
            -78.07,
            -78.92,
            -79.63,
            -79.71,
            -79.68,
            -79.56,
            -78.83,
            -77.91,
            -77.12,
            -76.42,
            -75.83,
            -75.34,
            -75.0,
            -74.74,
            -74.52,
            -74.39,
            -74.3,
            -74.25,
            -74.19,
        ],
        "label": r"25 $R_S$",
        "color": "#ffb53d",
        "ls": ":",
    },
    "Enceladus": {
        "north_lt": [
            0.41,
            1.24,
            2.07,
            2.9,
            3.72,
            4.55,
            5.38,
            6.21,
            7.04,
            7.86,
            8.69,
            9.52,
            10.34,
            11.18,
            12.0,
            12.0,
            12.83,
            13.65,
            14.48,
            15.31,
            16.14,
            16.96,
            17.79,
            18.62,
            19.45,
            20.27,
            21.1,
            21.93,
            22.76,
            23.58,
        ],
        "north_lat": [
            65.17,
            65.18,
            65.17,
            65.15,
            65.17,
            65.16,
            65.2,
            65.05,
            65.01,
            65.01,
            65.05,
            65.04,
            65.06,
            65.05,
            65.05,
            65.05,
            65.04,
            65.0,
            65.04,
            65.05,
            65.03,
            65.04,
            65.02,
            65.14,
            65.19,
            65.15,
            65.18,
            65.15,
            65.21,
            65.16,
        ],
        "south_lt": [
            0.42,
            1.24,
            2.07,
            2.89,
            3.73,
            4.55,
            5.38,
            6.21,
            7.03,
            7.86,
            8.69,
            9.51,
            10.35,
            11.17,
            12.0,
            12.0,
            12.83,
            13.65,
            14.48,
            15.31,
            16.14,
            16.97,
            17.79,
            18.62,
            19.45,
            20.28,
            21.11,
            21.93,
            22.76,
            23.58,
        ],
        "south_lat": [
            -62.13,
            -62.12,
            -62.1,
            -62.09,
            -62.1,
            -62.07,
            -62.06,
            -62.28,
            -62.24,
            -62.27,
            -62.22,
            -62.24,
            -62.25,
            -62.26,
            -62.26,
            -62.26,
            -62.26,
            -62.25,
            -62.27,
            -62.27,
            -62.25,
            -62.26,
            -62.25,
            -62.12,
            -62.12,
            -62.14,
            -62.09,
            -62.1,
            -62.12,
            -62.13,
        ],
        "label": r"Enceladus (3.94 $R_S$)",
        "color": "#00d48a",
        "ls": "--",
    },
    "Rhea": {
        "north_lt": [
            0.4,
            1.24,
            2.06,
            2.9,
            3.73,
            4.56,
            5.38,
            6.21,
            7.04,
            7.88,
            8.69,
            9.52,
            10.35,
            11.18,
            12.0,
            12.0,
            12.83,
            13.65,
            14.48,
            15.31,
            16.14,
            16.95,
            17.79,
            18.62,
            19.44,
            20.27,
            21.1,
            21.92,
            22.74,
            23.58,
        ],
        "north_lat": [
            72.33,
            72.34,
            72.34,
            72.49,
            72.45,
            72.43,
            72.49,
            72.5,
            72.51,
            72.48,
            72.52,
            72.43,
            72.52,
            72.51,
            72.44,
            72.44,
            72.61,
            72.46,
            72.51,
            72.57,
            72.49,
            72.57,
            72.5,
            72.47,
            72.39,
            72.45,
            72.37,
            72.39,
            72.47,
            72.49,
        ],
        "south_lt": [
            0.42,
            1.25,
            2.08,
            2.91,
            3.74,
            4.56,
            5.39,
            6.21,
            7.04,
            7.86,
            8.69,
            9.52,
            10.34,
            11.16,
            11.99,
            11.99,
            12.82,
            13.65,
            14.47,
            15.3,
            16.12,
            16.96,
            17.79,
            18.61,
            19.45,
            20.27,
            21.1,
            21.93,
            22.76,
            23.59,
        ],
        "south_lat": [
            -70.38,
            -70.52,
            -70.46,
            -70.42,
            -70.49,
            -70.47,
            -70.53,
            -70.54,
            -70.62,
            -70.51,
            -70.6,
            -70.55,
            -70.52,
            -70.67,
            -70.49,
            -70.49,
            -70.51,
            -70.59,
            -70.52,
            -70.56,
            -70.54,
            -70.46,
            -70.52,
            -70.5,
            -70.41,
            -70.5,
            -70.5,
            -70.37,
            -70.41,
            -70.37,
        ],
        "label": r"Rhea (8.75 $R_S$)",
        "color": "white",
        "ls": "--",
    },
}


def draw_reference_shells(
    ax: plt.Axes,
    shells: list[str] | None = None,
    hemisphere: str = "north",
    polar: bool = False,
) -> None:
    r"""Draw reference field-line shell traces on a LT-latitude plot.

    Replaces ``cassinilib/Plot.py:plotMoonOrbits()``.

    Parameters
    ----------
    ax : Axes
        Target axes.
    shells : list of str, optional
        Shell names from ``REFERENCE_SHELLS``. Default: all.
    hemisphere : str
        'north' or 'south'.
    polar : bool
        If True, convert local time to radians for polar projection.
    """
    if shells is None:
        shells = list(REFERENCE_SHELLS.keys())

    for name in shells:
        shell = REFERENCE_SHELLS.get(name)
        if shell is None:
            continue

        if hemisphere == "north":
            lt = np.array(shell["north_lt"])
            lat = np.array(shell["north_lat"])
        else:
            lt = np.array(shell["south_lt"])
            lat = np.array(shell["south_lat"])

        if polar:
            lt = lt * (2 * np.pi / 24)  # hours → radians

        ax.plot(
            lt,
            lat,
            label=shell.get("label", name),
            color=shell.get("color", "white"),
            ls=shell.get("ls", "--"),
            lw=1.5,
            alpha=0.7,
        )


def draw_plane_3d(
    ax: plt.Axes,
    origin: tuple[float, float, float] = (0, 0, 0),
    normal: tuple[float, float, float] = (0, 0, 1),
    xlim: tuple[float, float] = (-10, 10),
    ylim: tuple[float, float] = (-10, 10),
    color: str = "black",
    alpha: float = 0.1,
) -> None:
    r"""Draw a plane in 3D space.

    Replaces ``cassinilib/Plot.py:drawPlane()``.

    Parameters
    ----------
    origin : tuple
        A point on the plane.
    normal : tuple
        Normal vector to the plane.
    xlim, ylim : tuple
        Extent of the mesh grid.
    """
    origin = np.asarray(origin, dtype=float)
    normal = np.asarray(normal, dtype=float)
    d = -origin.dot(normal)
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 20),
        np.linspace(ylim[0], ylim[1], 20),
    )
    if abs(normal[2]) > 1e-12:
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
    else:
        zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color, zorder=-1)


def add_text_3d(
    ax: plt.Axes,
    pos: tuple[float, float, float],
    text: str,
    color: str = "white",
    alpha: float = 1.0,
    box: bool = True,
    fontsize: float = 18,
) -> None:
    r"""Add a text label in 3D space.

    Replaces ``cassinilib/Plot.py:addText3D()``.
    """
    if box:
        bbox = dict(boxstyle="round,pad=0.25", fc=color, lw=1, alpha=0.5)
        ax.text(
            pos[0],
            pos[1],
            pos[2],
            text,
            size=fontsize,
            color="white",
            ha="center",
            va="center",
            bbox=bbox,
            alpha=alpha,
        )
    else:
        ax.text(
            pos[0],
            pos[1],
            pos[2],
            text,
            size=fontsize,
            color=color,
            ha="center",
            va="center",
            alpha=alpha,
        )


def project_points_3d(
    ax: plt.Axes,
    point: tuple[float, float, float],
    color: str = "white",
    lw: float = 1.0,
    alpha: float = 0.7,
    radius_line: bool = False,
) -> None:
    r"""Draw dashed projection lines from a 3D point to the axes planes.

    Replaces ``cassinilib/Plot.py:projectPoints3D()``.
    """
    px, py, pz = float(point[0]), float(point[1]), float(point[2])
    kw = dict(color=color, linestyle="--", lw=lw, alpha=alpha)
    ax.plot([0, px], [py, py], [0, 0], **kw)
    ax.plot([px, px], [0, py], [0, 0], **kw)
    ax.plot([px, px], [py, py], [0, pz], **kw)
    if radius_line:
        ax.plot([px, 0], [py, 0], [0, 0], **kw)


def draw_orbit_circle(
    ax: plt.Axes,
    center: tuple[float, float] = (0, 0),
    radius: float = 4.0,
    color: str = "#ffd000",
    alpha: float = 0.8,
    lw: float = 1.5,
    z: float = 0.0,
) -> None:
    r"""Draw a circular orbit as a 2D patch in 3D space.

    Replaces ``cassinilib/Plot.py:drawOrbit()``.
    """
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import art3d

    p = Circle(center, radius, ls="--", color=color, fill=False, lw=lw, alpha=alpha)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")


def render_movie(
    frame_pattern: str,
    output: str = "movie.mp4",
    fps: int = 30,
    resolution: str = "1920x1080",
) -> None:
    r"""Render a movie from numbered frame images using ffmpeg.

    Replaces ``cassinilib/Plot.py:processMovie2()``.

    Parameters
    ----------
    frame_pattern : str
        Input file pattern (e.g., ``'frames/frame_%03d.png'``).
    output : str
        Output filename.
    fps : int
        Frames per second.
    resolution : str
        Output resolution (e.g., ``'1920x1080'``).
    """
    cmd = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-vcodec",
        "mpeg4",
        "-s:v",
        resolution,
        "-y",
        output,
    ]
    log.info("Rendering movie: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
