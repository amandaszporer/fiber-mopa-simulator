"""
Spectrally-resolved bidirectional ASE solver for the Yb-MOPA simulator.

Public entry points:

    SpectralGrid.from_fiber(r_core, NA)     — build a wavelength grid for a fiber
    AseState.zero(grid)                     — zero ASE state for a stage input
    OpticalState(signal, ase)               — what every component receives/returns
    solve_steady_state(...)                 — Mode A: steady-state BVP
    solve_time_dependent(...)               — Mode B: time-dependent (B1 + B2)

See docs/ase.md for the full physics. Read it before changing solver internals.
"""

from .spectral_grid import SpectralGrid
from .state import AseState, OpticalState
from .solver_steady import solve_steady_state, SteadyResult
from .solver_time import solve_time_dependent

__all__ = [
    "SpectralGrid",
    "AseState",
    "OpticalState",
    "solve_steady_state",
    "SteadyResult",
    "solve_time_dependent",
]
