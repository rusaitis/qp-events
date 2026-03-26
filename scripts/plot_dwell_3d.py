#!/usr/bin/env python
"""3D dwell-time visualization using PyVista (VTK).

Interactive 3D view of Cassini equatorial dwell time with orbit trajectory,
Saturn, and rings.

Keyboard shortcuts (interactive mode):
    o — toggle orbit transparency (50% / 25%)
    m — toggle dawn-dusk meridian plane

Usage
-----
    # Interactive window:
    uv run python scripts/plot_dwell_3d.py

    # Save screenshot:
    uv run python scripts/plot_dwell_3d.py --screenshot Output/dwell_3d.png

    # Custom view:
    uv run python scripts/plot_dwell_3d.py --r-max 40 --variable magnetosphere
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="PyVista 3D equatorial dwell-time visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default="Output/dwell_grid.zarr")
    parser.add_argument("--variable", type=str, default="total")
    parser.add_argument("--r-max", type=float, default=70.0,
                        help="Max radial distance to display (R_S)")
    parser.add_argument("--screenshot", type=str, default=None,
                        help="Save screenshot to file instead of interactive")
    parser.add_argument("--window-size", type=str, default="1920x1080",
                        help="Window size WxH")
    parser.add_argument("--orbit-radius", type=float, default=0.10,
                        help="Orbit tube radius (R_S)")
    parser.add_argument("--no-orbit", action="store_true")
    args = parser.parse_args()

    import pyvista as pv

    from qp.dwell.io import load_zarr
    from qp.io.products import load_spacecraft_position

    # --- Load dwell grid ---
    ds = load_zarr(args.input)

    lat = ds.coords["magnetic_latitude"].values
    lat_mask = np.abs(lat) < 5.0
    data_raw = ds[args.variable].values[:, lat_mask, :].sum(axis=1)  # (n_r, n_lt)

    r = ds.coords["r"].values
    lt = ds.coords["local_time"].values

    # Upsample 4× with cubic interpolation, then smooth for clean gradients
    # (base grid is already 100×96, so 4× gives 400×384 — smooth enough)
    upsample = 4
    data_log = np.log10(np.clip(data_raw, 0.1, None))
    data_hi = zoom(data_log, upsample, order=3)
    data_hi = gaussian_filter(data_hi, sigma=1.5)

    r_hi = np.linspace(r[0], r[-1], len(r) * upsample)
    lt_hi = np.linspace(lt[0], lt[-1], len(lt) * upsample)

    # Upsample the raw validity mask (no smoothing — binary)
    valid_raw = data_raw >= 1
    valid_hi = zoom(valid_raw.astype(float), upsample, order=0) > 0.5

    # Build structured grid in Cartesian at z=0
    R, LT = np.meshgrid(r_hi, lt_hi, indexing="ij")
    phi = (LT - 12.0) * np.pi / 12.0
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    Z = np.zeros_like(X)

    # Mask: clip to r_max, remove zero-dwell
    r_mesh = np.sqrt(X**2 + Y**2)
    valid = valid_hi & (r_mesh <= args.r_max)

    # PyVista StructuredGrid
    grid = pv.StructuredGrid(
        X[:, :, np.newaxis],
        Y[:, :, np.newaxis],
        Z[:, :, np.newaxis],
    )
    grid.point_data["log10_dwell"] = data_hi.ravel(order="F")
    grid.point_data["valid"] = valid.ravel(order="F").astype(float)

    surface = grid.threshold(value=0.5, scalars="valid")

    # --- Plotter ---
    w, h = [int(x) for x in args.window_size.split("x")]
    off_screen = args.screenshot is not None
    pl = pv.Plotter(window_size=(w, h), off_screen=off_screen)
    pl.set_background("#080810", top="#0a0a1a")  # subtle gradient
    if not off_screen:
        pl.enable_terrain_style()  # constrains rotation to azimuth around Z
    pl.enable_depth_peeling(number_of_peels=8)  # correct transparency compositing

    # Dwell surface — semi-transparent so orbits show through
    dwell_max = data_hi[valid].max() if valid.any() else 4
    pl.add_mesh(
        surface,
        scalars="log10_dwell",
        cmap="inferno",
        clim=[0, dwell_max],
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "log₁₀(dwell time / min)",
            "title_font_size": 20,
            "label_font_size": 18,
            "color": "white",
            "position_x": 0.75,
            "position_y": 0.05,
            "width": 0.22,
            "height": 0.08,
            "vertical": False,
        },
        opacity=0.75,
        smooth_shading=True,
        specular=0.15,
        specular_power=30,
        ambient=0.3,
        diffuse=0.7,
    )

    # Saturn sphere
    saturn = pv.Sphere(radius=1.0, center=(0, 0, 0),
                       theta_resolution=60, phi_resolution=60)
    pl.add_mesh(saturn, color="#e8d282", opacity=0.9,
                smooth_shading=True, specular=0.4, ambient=0.2)

    # Rings — higher resolution
    ring = pv.Disc(center=(0, 0, 0), inner=1.24, outer=2.27,
                   normal=(0, 0, 1), r_res=5, c_res=120)
    pl.add_mesh(ring, color="#c8a832", opacity=0.6)

    # --- Dawn-dusk meridian plane (Y-Z, toggled with 'm') ---
    lat_vals = ds.coords["magnetic_latitude"].values
    lt_vals = ds.coords["local_time"].values
    data_3d = ds[args.variable].values  # (n_r, n_lat, n_lt)

    # Sum dwell time over LT bands: dawn (5-7h) and dusk (17-19h)
    dawn_mask = (lt_vals >= 5) & (lt_vals <= 7)
    dusk_mask = (lt_vals >= 17) & (lt_vals <= 19)
    dawn_data = data_3d[:, :, dawn_mask].sum(axis=2)  # (n_r, n_lat)
    dusk_data = data_3d[:, :, dusk_mask].sum(axis=2)

    # Upsample both halves
    dawn_log = np.log10(np.clip(dawn_data, 0.1, None))
    dusk_log = np.log10(np.clip(dusk_data, 0.1, None))
    dawn_hi = gaussian_filter(zoom(dawn_log, upsample, order=3), sigma=1.5)
    dusk_hi = gaussian_filter(zoom(dusk_log, upsample, order=3), sigma=1.5)
    dawn_valid = zoom((dawn_data >= 1).astype(float), upsample, order=0) > 0.5
    dusk_valid = zoom((dusk_data >= 1).astype(float), upsample, order=0) > 0.5

    lat_hi = np.linspace(lat_vals[0], lat_vals[-1], len(lat_vals) * upsample)

    # Build dusk half (+Y side): y = r*cos(lat), z = r*sin(lat), x = 0
    R_m, LAT_m = np.meshgrid(r_hi, np.radians(lat_hi), indexing="ij")
    Y_dusk = R_m * np.cos(LAT_m)
    Z_dusk = R_m * np.sin(LAT_m)
    X_dusk = np.zeros_like(Y_dusk)

    # Build dawn half (-Y side): mirror y
    Y_dawn = -R_m * np.cos(LAT_m)
    Z_dawn = R_m * np.sin(LAT_m)
    X_dawn = np.zeros_like(Y_dawn)

    r_m = np.sqrt(Y_dusk**2 + Z_dusk**2)
    dusk_mask_r = dusk_valid & (r_m <= args.r_max)
    dawn_mask_r = dawn_valid & (r_m <= args.r_max)

    # Dusk surface
    g_dusk = pv.StructuredGrid(
        X_dusk[:, :, np.newaxis], Y_dusk[:, :, np.newaxis], Z_dusk[:, :, np.newaxis],
    )
    g_dusk.point_data["log10_dwell"] = dusk_hi.ravel(order="F")
    g_dusk.point_data["valid"] = dusk_mask_r.ravel(order="F").astype(float)
    s_dusk = g_dusk.threshold(value=0.5, scalars="valid")

    # Dawn surface
    g_dawn = pv.StructuredGrid(
        X_dawn[:, :, np.newaxis], Y_dawn[:, :, np.newaxis], Z_dawn[:, :, np.newaxis],
    )
    g_dawn.point_data["log10_dwell"] = dawn_hi.ravel(order="F")
    g_dawn.point_data["valid"] = dawn_mask_r.ravel(order="F").astype(float)
    s_dawn = g_dawn.threshold(value=0.5, scalars="valid")

    meridian_actors = []
    for s_half in [s_dusk, s_dawn]:
        actor = pl.add_mesh(
            s_half, scalars="log10_dwell", cmap="inferno",
            clim=[0, dwell_max], opacity=0.7, smooth_shading=True,
            specular=0.15, ambient=0.3, diffuse=0.7,
            show_scalar_bar=False,
        )
        actor.SetVisibility(False)  # hidden by default
        meridian_actors.append(actor)

    # --- Equatorial reference grid at z=0 (every 10 R_S) ---
    spacing = 10
    lim = args.r_max
    grid_color = "#888888"
    ticks = np.arange(-lim, lim + spacing, spacing)
    for v in ticks:
        pl.add_mesh(pv.Line((-lim, v, 0), (lim, v, 0)),
                     color="white", opacity=0.12, line_width=1)
        pl.add_mesh(pv.Line((v, -lim, 0), (v, lim, 0)),
                     color="white", opacity=0.12, line_width=1)

    # Grid tick labels along both edges, every 10 R_S including 0
    label_offset = 3.0  # R_S offset from the grid edge
    for v in ticks:
        label = f"{int(v)}"
        # Labels along X-axis (at y = -lim edge)
        pt_x = np.array([[v, -lim - label_offset, 0]])
        pl.add_point_labels(
            pv.PolyData(pt_x), [label], font_size=36, text_color=grid_color,
            point_size=0, shape=None, render_points_as_spheres=False,
        )
        # Labels along Y-axis (at x = -lim edge)
        pt_y = np.array([[-lim - label_offset, v, 0]])
        pl.add_point_labels(
            pv.PolyData(pt_y), [label], font_size=36, text_color=grid_color,
            point_size=0, shape=None, render_points_as_spheres=False,
        )

    # --- Axis arrows from Saturn ---
    arrow_lens = {
        (1, 0, 0): args.r_max * 1.05,   # X and Y: 50% longer than before
        (0, 1, 0): args.r_max * 1.05,
        (0, 0, 1): args.r_max * 0.7,     # Z: same as before
    }
    for direction, label, color in [
        ((1, 0, 0), "X (Sun)", "#ff6666"),
        ((0, 1, 0), "Y (Dusk)", "#66ff66"),
        ((0, 0, 1), "Z", "#6688ff"),
    ]:
        alen = arrow_lens[direction]
        arrow = pv.Arrow(
            start=(0, 0, 0), direction=direction,
            scale=alen, shaft_radius=0.005, tip_radius=0.015, tip_length=0.06,
        )
        pl.add_mesh(arrow, color=color, opacity=0.7, smooth_shading=True)
        tip = np.array(direction, dtype=float) * alen * 1.05
        pl.add_point_labels(
            pv.PolyData(tip.reshape(1, 3)), [label],
            font_size=28, text_color=color,
            point_size=0, shape=None, render_points_as_spheres=False,
        )

    # --- Cassini orbit colored by time ---
    if not args.no_orbit:
        try:
            import datetime as dt

            pos_data = load_spacecraft_position()
            times = pos_data[:, 0]  # datetime objects
            ox = pos_data[:, 5].astype(float)
            oy = pos_data[:, 6].astype(float)
            oz = pos_data[:, 7].astype(float)
            or_ = np.sqrt(ox**2 + oy**2 + oz**2)

            in_range = or_ <= args.r_max
            times = times[in_range]
            ox, oy, oz = ox[in_range], oy[in_range], oz[in_range]

            # Fine sampling for smooth orbit
            step = 10
            points = np.column_stack([ox[::step], oy[::step], oz[::step]])
            times_sub = times[::step]

            # Convert datetimes to fractional year for color scale
            def to_fractional_year(d):
                year_start = dt.datetime(d.year, 1, 1)
                year_end = dt.datetime(d.year + 1, 1, 1)
                return d.year + (d - year_start).total_seconds() / (year_end - year_start).total_seconds()

            fyear = np.array([to_fractional_year(t) for t in times_sub])

            # Build polyline
            n = len(points)
            cells = np.column_stack([
                np.full(n - 1, 2),
                np.arange(n - 1),
                np.arange(1, n),
            ]).ravel()
            orbit = pv.PolyData(points, lines=cells)
            orbit.point_data["year"] = fyear

            tube = orbit.tube(radius=args.orbit_radius, n_sides=12)
            orbit_actor = pl.add_mesh(
                tube,
                scalars="year",
                cmap="cool",
                clim=[2004, 2017],
                opacity=0.5,
                smooth_shading=True,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Year",
                    "title_font_size": 20,
                    "label_font_size": 18,
                    "color": "white",
                    "position_x": 0.75,
                    "position_y": 0.18,
                    "width": 0.22,
                    "height": 0.08,
                    "vertical": False,
                    "fmt": "%.0f",
                    "n_labels": 4,
                },
            )

            # 'o' toggles orbit between full and half opacity
            orbit_dimmed = [False]
            base_opacity = 0.5

            def toggle_orbit_opacity():
                orbit_dimmed[0] = not orbit_dimmed[0]
                new_op = base_opacity * 0.5 if orbit_dimmed[0] else base_opacity
                orbit_actor.GetProperty().SetOpacity(new_op)
                pl.render()

            pl.add_key_event("o", toggle_orbit_opacity)

        except FileNotFoundError:
            print("  (orbit data not available, skipping)")

    # 'm' toggles dawn-dusk meridian plane
    def toggle_meridian():
        vis = not meridian_actors[0].GetVisibility()
        for a in meridian_actors:
            a.SetVisibility(vis)
        pl.render()

    pl.add_key_event("m", toggle_meridian)

    # Sun marker
    sun_pos = np.array([[args.r_max * 0.9, 0, 0]])
    sun_pt = pv.PolyData(sun_pos)
    pl.add_point_labels(
        sun_pt, ["☉"], font_size=28, text_color="#ffdd44",
        point_size=0, shape=None, render_points_as_spheres=False,
    )

    # Title
    var_label = args.variable.replace("_", " ").title()
    year_from = ds.attrs.get("year_from", "?")
    year_to = ds.attrs.get("year_to", "?")
    total_hours = float(ds[args.variable].sum()) / 60
    pl.add_text(
        f"Cassini Equatorial Dwell Time — {var_label}\n"
        f"{year_from}–{year_to}  ·  {total_hours:,.0f} hours",
        position="upper_left", font_size=12, color="white",
    )

    # Camera — Z-up, rotation around Z-axis
    dist = args.r_max * 3.0
    elev_rad = np.radians(30)
    azim_rad = np.radians(-60)
    cam_x = dist * np.cos(elev_rad) * np.cos(azim_rad)
    cam_y = dist * np.cos(elev_rad) * np.sin(azim_rad)
    cam_z = dist * np.sin(elev_rad)
    pl.camera_position = [
        (cam_x, cam_y, cam_z),  # position
        (0, 0, 0),               # focal point
        (0, 0, 1),               # view-up = Z
    ]

    # Subtle light from the "Sun" direction
    pl.add_light(pv.Light(
        position=(args.r_max * 2, 0, args.r_max),
        focal_point=(0, 0, 0),
        color="white",
        intensity=0.6,
    ))

    if args.screenshot:
        outpath = Path(args.screenshot)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        pl.show(screenshot=str(outpath))
        print(f"Saved: {outpath}")
    else:
        pl.show()


if __name__ == "__main__":
    main()
