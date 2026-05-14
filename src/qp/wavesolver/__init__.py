"""Field line resonance (FLR) eigenfrequency solver.

Computes standing Alfvén wave eigenfrequencies on magnetospheric field lines
using a shooting method with realistic density and field models.

Import directly from submodules, e.g.::

    from qp.wavesolver.solver import WavesolverConfig, solve_eigenfrequencies
    config = WavesolverConfig(l_shell=15, component="toroidal", n_modes=6)
    result = solve_eigenfrequencies(config)
"""
