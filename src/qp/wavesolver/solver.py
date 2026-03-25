"""Main entry point for eigenfrequency computation.

Wires together field line tracing (Phase 3 of KMAG rewrite),
density models (wavesolver Phase 2), field line profiles (Phase 3),
and the eigensolver (Phase 1) into a single clean API.

Usage
-----
>>> from qp.wavesolver.solver import solve_eigenfrequencies, WavesolverConfig
>>> config = WavesolverConfig(l_shell=15, component="toroidal", n_modes=6)
>>> result = solve_eigenfrequencies(config)
>>> print(result.periods_minutes)
"""

from __future__ import annotations

import logging
import math
from copy import replace
from dataclasses import dataclass

import numpy as np

from qp.fieldline.kmag_model import SaturnField
from qp.fieldline.tracer import (
    dipole_field,
    saturn_field_wrapper,
    trace_fieldline_bidirectional,
)
from qp.wavesolver.density import (
    BagenalDelamere,
    DensityModel,
    PersoonEtAl,
    PowerLawDensity,
    UniformDensity,
)
from qp.wavesolver.eigensolver import find_eigenfrequencies
from qp.wavesolver.field_profile import compute_field_line_profile
from qp.wavesolver.result import EigenResult

_DENSITY_MODELS: dict[str, type] = {
    "bagenal": BagenalDelamere,
    "persoon": PersoonEtAl,
    "power_law": PowerLawDensity,
    "uniform": UniformDensity,
}


@dataclass(frozen=True, slots=True)
class WavesolverConfig:
    """Configuration for FLR eigenfrequency computation.

    Parameters
    ----------
    l_shell : float
        Equatorial crossing distance (R_S). Determines the field line to trace.
    colatitude : float or None
        Alternative to l_shell: starting colatitude in degrees. If given,
        overrides l_shell. The field line is traced from (1.1 R_S, colatitude).
    local_time_hours : float
        Local time in hours (0=midnight, 12=noon). Determines the azimuthal
        angle for the starting position.
    component : str
        Wave polarization: 'toroidal' or 'poloidal'.
    n_modes : int
        Number of eigenmodes to find.
    field : SaturnField or None
        Magnetic field model. None uses an analytical dipole.
    time : float
        Epoch time in J2000 seconds (only used with SaturnField).
    density_model : str or DensityModel
        Either a string key ('bagenal', 'persoon', 'power_law', 'uniform')
        or a DensityModel instance directly.
    ion_mass_amu : float
        Ion mass in atomic mass units (default 18 = water group).
    freq_range : tuple[float, float]
        Angular frequency search range (rad/s).
    resolution : int
        Number of frequency samples for bracket search.
    tolerance : float
        Root-finding convergence tolerance.
    include_eigenfunctions : bool
        Store eigenfunctions in the result.
    trace_step : float
        Field line tracing step size (R_S).
    """

    # Field line geometry
    l_shell: float = 15.0
    colatitude: float | None = None
    local_time_hours: float = 12.0

    # Wave
    component: str = "toroidal"
    n_modes: int = 6

    # Field model
    field: SaturnField | None = None
    time: float = 284040000.0

    # Density
    density_model: str | DensityModel = "bagenal"
    ion_mass_amu: float = 18.0

    # Solver
    freq_range: tuple[float, float] = (1e-5, 0.01)
    resolution: int = 300
    tolerance: float = 1e-7
    include_eigenfunctions: bool = False

    # Tracing
    trace_step: float = 0.03


def solve_eigenfrequencies(config: WavesolverConfig) -> EigenResult:
    """Compute FLR eigenfrequencies for a single field line.

    Steps:
    1. Determine starting position from L-shell or colatitude
    2. Trace field line bidirectionally (dipole or KMAG)
    3. Build field line profile (B, n, vA, h1, h2, splines)
    4. Find eigenfrequencies using bracket search + Brent's method
    5. Package into EigenResult

    Parameters
    ----------
    config : WavesolverConfig
        Solver configuration.

    Returns
    -------
    EigenResult
        Eigenfrequencies, mode numbers, and optional field line profiles.
    """
    # 1. Starting position
    start_pos = _starting_position(config)

    # 2. Trace field line
    if config.field is not None:
        field_func = saturn_field_wrapper(config.field, config.time, coord="KSM")
    else:
        field_func = dipole_field
    trace = trace_fieldline_bidirectional(
        field_func,
        start_pos,
        step=config.trace_step,
        max_steps=100000,
    )

    # 3. Build field line profile
    density = _resolve_density_model(config)
    profile = compute_field_line_profile(trace, density)

    # 4. Select the appropriate dlnh spline and pre-sampled arrays
    if config.component == "toroidal":
        dlnh_spline = profile.dlnh1B_spline
        dlnh_samples = profile.dlnh1B_samples
    else:
        dlnh_spline = profile.dlnh2B_spline
        dlnh_samples = profile.dlnh2B_samples

    # 5. Find eigenfrequencies (numba fast path when samples available)
    modes = find_eigenfrequencies(
        s_span=profile.s_span_meters,
        va_spline=profile.va_spline,
        dlnh_spline=dlnh_spline,
        freq_range=config.freq_range,
        n_modes=config.n_modes,
        resolution=config.resolution,
        tolerance=config.tolerance,
        include_eigenfunctions=config.include_eigenfunctions,
        va_samples=profile.va_samples,
        dlnh_samples=dlnh_samples,
    )

    return EigenResult(
        modes=modes,
        l_shell=profile.l_shell,
        conjugate_latitude=profile.conjugate_latitude,
        component=config.component,
        arc_length=profile.arc_length,
        alfven_velocity=profile.alfven_velocity_profile,
        field_magnitude=profile.field_magnitude,
        density=profile.density,
    )


def solve_for_latitude_range(
    config: WavesolverConfig,
    lat_min: float = 60.0,
    lat_max: float = 80.0,
    n_fieldlines: int = 20,
) -> list[EigenResult]:
    """Solve eigenfrequencies across a range of conjugate latitudes.

    Creates field lines at evenly spaced colatitudes and solves each one.

    Parameters
    ----------
    config : WavesolverConfig
        Base configuration (component, density, field model, etc.).
        The l_shell/colatitude fields are overridden for each field line.
    lat_min, lat_max : float
        Conjugate latitude range (degrees).
    n_fieldlines : int
        Number of field lines to compute.

    Returns
    -------
    list[EigenResult]
        One result per field line, sorted by conjugate latitude.
    """
    log = logging.getLogger(__name__)
    results = []

    if config.field is not None:
        # KMAG: loop over L-shells directly (equatorial start ensures
        # consistent equatorial crossing distance at all local times)
        l_max = 1.0 / math.sin(math.radians(90.0 - lat_min)) ** 2
        l_min = 1.0 / math.sin(math.radians(90.0 - lat_max)) ** 2
        l_shells = np.linspace(l_min, l_max, n_fieldlines)
        for L in l_shells:
            cfg = replace(config, l_shell=float(L), colatitude=None)
            try:
                result = solve_eigenfrequencies(cfg)
                results.append(result)
            except Exception as exc:
                log.warning("Failed at L=%.1f: %s", L, exc)
    else:
        # Dipole: loop over colatitudes (ionospheric start, dipole formula exact)
        colatitudes = 90.0 - np.linspace(lat_min, lat_max, n_fieldlines)
        for colat in colatitudes:
            cfg = replace(config, colatitude=colat)
            try:
                result = solve_eigenfrequencies(cfg)
                results.append(result)
            except Exception as exc:
                log.warning("Failed at colat %.1f°: %s", colat, exc)

    results.sort(key=lambda r: r.conjugate_latitude)
    return results


# ======================================================================
# Internal helpers
# ======================================================================


def _starting_position(config: WavesolverConfig) -> np.ndarray:
    """Compute the starting position for field line tracing.

    For KMAG field models, starts at the equatorial plane at the desired
    L-shell distance. This guarantees the equatorial crossing matches the
    intended L-shell regardless of local time (avoids current-sheet
    stretching artifacts on the nightside).

    For dipole (field=None), starts near the ionosphere using the exact
    dipole relation sin²(θ) = 1/L.

    If colatitude is given explicitly, always starts from the ionosphere
    (for backwards compatibility).
    """
    # Local time → azimuthal angle (0h=midnight → φ=π, 12h=noon → φ=0)
    phi = math.radians((config.local_time_hours - 12.0) * 15.0)

    if config.colatitude is not None:
        # Explicit colatitude: start from ionosphere
        colat_rad = math.radians(config.colatitude)
        r0 = 1.1
        x = r0 * math.sin(colat_rad) * math.cos(phi)
        y = r0 * math.sin(colat_rad) * math.sin(phi)
        z = r0 * math.cos(colat_rad)
        return np.array([x, y, z])

    if config.field is not None:
        # KMAG: start at equatorial plane at distance L
        L = config.l_shell
        x = L * math.cos(phi)
        y = L * math.sin(phi)
        z = 0.0
        return np.array([x, y, z])

    # Dipole: start from ionosphere using exact formula
    sin_colat = 1.0 / math.sqrt(config.l_shell)
    colat_rad = math.asin(sin_colat)
    r0 = 1.1
    x = r0 * math.sin(colat_rad) * math.cos(phi)
    y = r0 * math.sin(colat_rad) * math.sin(phi)
    z = r0 * math.cos(colat_rad)
    return np.array([x, y, z])


def _resolve_density_model(config: WavesolverConfig) -> DensityModel:
    """Resolve the density model from config."""
    if isinstance(config.density_model, str):
        key = config.density_model.lower()
        if key not in _DENSITY_MODELS:
            raise ValueError(
                f"Unknown density model: {config.density_model!r}. "
                f"Choose from: {list(_DENSITY_MODELS.keys())}"
            )
        cls = _DENSITY_MODELS[key]
        return cls(ion_mass_amu=config.ion_mass_amu)
    return config.density_model
