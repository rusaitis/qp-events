"""Field line resonance (FLR) eigenfrequency solver.

Computes standing Alfvén wave eigenfrequencies on magnetospheric field lines
using a shooting method with realistic density and field models.

Usage
-----
>>> from qp.wavesolver import solve_eigenfrequencies, WavesolverConfig
>>> config = WavesolverConfig(l_shell=15, component="toroidal", n_modes=6)
>>> result = solve_eigenfrequencies(config)
>>> print(result.periods_minutes)
"""

from qp.wavesolver.density import (
    BagenalDelamere,
    DensityModel,
    PersoonEtAl,
    PowerLawDensity,
    UniformDensity,
    alfven_velocity,
)
from qp.wavesolver.eigensolver import find_eigenfrequencies
from qp.wavesolver.field_profile import (
    FieldLineProfile,
    compute_field_line_profile,
)
from qp.wavesolver.result import EigenMode, EigenResult
from qp.wavesolver.solver import (
    WavesolverConfig,
    solve_eigenfrequencies,
    solve_for_latitude_range,
)
from qp.wavesolver.wave_equation import (
    boundary_error,
    count_mode_number,
    integrate_wave_equation,
)

__all__ = [
    # Solver API
    "WavesolverConfig",
    "solve_eigenfrequencies",
    "solve_for_latitude_range",
    # Results
    "EigenMode",
    "EigenResult",
    # Density models
    "BagenalDelamere",
    "DensityModel",
    "PersoonEtAl",
    "PowerLawDensity",
    "UniformDensity",
    "alfven_velocity",
    # Field line profile
    "FieldLineProfile",
    "compute_field_line_profile",
    # Low-level eigensolver
    "find_eigenfrequencies",
    "integrate_wave_equation",
    "boundary_error",
    "count_mode_number",
]
