"""Shared fixtures for ASE solver tests."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project root is importable when running pytest from anywhere.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ase.solver_steady import AmplifierGeometry  # noqa: E402
from ase.spectral_grid import SpectralGrid  # noqa: E402

_SIGMA_A_PUMP = 2.5e-24


def _stage_geom(core_um, clad_um, NA, abs_dB_per_m, length_m):
    r_core = core_um * 1e-6 / 2
    A_core = math.pi * r_core ** 2
    A_clad = math.pi * (clad_um * 1e-6 / 2) ** 2
    gamma_pump = A_core / A_clad
    N_Yb = abs_dB_per_m / (4.343 * _SIGMA_A_PUMP * gamma_pump)
    geom = AmplifierGeometry(
        fiber_length=length_m,
        A_core=A_core,
        A_clad=A_clad,
        N_Yb=N_Yb,
        gamma_pump=gamma_pump,
    )
    grid = SpectralGrid.from_fiber(r_core=r_core, NA=NA)
    return geom, grid, r_core


@pytest.fixture
def stage1_setup():
    """5/130 µm Yb fiber, NA=0.12, 1.65 dB/m cladding abs, 3 m length."""
    return _stage_geom(5, 130, 0.12, 1.65, 3.0)


@pytest.fixture
def stage1_long():
    """Same fiber, 5 m — used for the high-ASE / clamping tests."""
    return _stage_geom(5, 130, 0.12, 1.65, 5.0)


@pytest.fixture
def zero_ase(stage1_setup):
    geom, grid, _ = stage1_setup
    return np.zeros(grid.n_bins)
