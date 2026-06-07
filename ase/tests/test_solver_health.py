"""Unit tests for `ase.solver_health` (Layer 1 diagnostics)."""

import math

import numpy as np
import pytest

from ase.solver_health import (
    SolverHealth,
    compute_solver_health,
)
from ase.solver_steady import solve_steady_state


def test_health_dataclass_is_frozen():
    """SolverHealth must be a frozen dataclass (no accidental mutation)."""
    h = SolverHealth(
        energy_residual_ratio=-0.1,
        ase_conversion_fraction=0.01,
        small_signal_g0L=2.0,
        energy_status="ok",
        regime="amplifier",
    )
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        h.energy_status = "violation"  # type: ignore[misc]


def test_health_bgu_stage1_is_ok(stage1_setup, zero_ase):
    """BGU stage-1 healthy regime: energy_status=ok, regime=amplifier."""
    geom, grid, _ = stage1_setup
    P_pump, P_sig = 0.3, 0.75e-3
    r = solve_steady_state(
        geom, grid, P_pump=P_pump, P_signal_avg=P_sig,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    h = compute_solver_health(r, geom, grid, P_pump, P_sig)
    assert h.energy_status == "ok"
    assert h.regime in ("amplifier", "high_gain")  # depends on g₀·L
    # The signal output stays under the QD ceiling — residual ≤ 0 (with
    # numerical noise margin).
    assert h.energy_residual_ratio < 0.01


def test_health_g0L_high_gain_flagged():
    """A synthetic geometry with g₀·L >> 30 is flagged `high_gain`."""
    from ase.solver_steady import AmplifierGeometry
    from ase.spectral_grid import SpectralGrid

    # 12 m fiber × 1.65 dB/m with a stronger pump gives g₀·L > 30.
    r_core = 2.5e-6
    A_core = math.pi * r_core ** 2
    A_clad = math.pi * (130e-6 / 2) ** 2
    gamma_pump = A_core / A_clad
    N_Yb = 1.65 / (4.343 * 2.5e-24 * gamma_pump)
    geom = AmplifierGeometry(
        fiber_length=12.0, A_core=A_core, A_clad=A_clad,
        N_Yb=N_Yb, gamma_pump=gamma_pump,
    )
    grid = SpectralGrid.from_fiber(r_core=r_core, NA=0.12)

    from ase.solver_steady import SteadyResult
    n_z = 10
    fake = SteadyResult(
        z=np.linspace(0, geom.fiber_length, n_z),
        n2_z=np.zeros(n_z),
        P_pump_z=np.full(n_z, 2.0),     # 2 W pump → near-asymptotic n₂
        P_signal_z=np.zeros(n_z),
        P_ase_fwd_z=np.zeros((n_z, grid.n_bins)),
        P_ase_bwd_z=np.zeros((n_z, grid.n_bins)),
        converged=True,
        iterations=1,
    )
    h = compute_solver_health(fake, geom, grid, P_pump=2.0, P_signal_in=0.0)
    assert h.small_signal_g0L > 30, (
        f"expected high-gain design (>30); got g₀·L = {h.small_signal_g0L:.1f}"
    )
    assert h.regime == "high_gain"


def test_health_zero_pump_zero_signal_is_ok():
    """Edge case: no pump, no signal → residual = 0, status = ok, regime
    = amplifier."""
    from ase.solver_steady import AmplifierGeometry, SteadyResult
    from ase.spectral_grid import SpectralGrid

    r_core = 2.5e-6
    geom = AmplifierGeometry(
        fiber_length=3.0,
        A_core=math.pi * r_core ** 2,
        A_clad=math.pi * (130e-6 / 2) ** 2,
        N_Yb=1.0e26,
        gamma_pump=0.001,
    )
    grid = SpectralGrid.from_fiber(r_core=r_core, NA=0.12)
    n_z = 10
    fake = SteadyResult(
        z=np.linspace(0, geom.fiber_length, n_z),
        n2_z=np.zeros(n_z),
        P_pump_z=np.zeros(n_z),
        P_signal_z=np.zeros(n_z),
        P_ase_fwd_z=np.zeros((n_z, grid.n_bins)),
        P_ase_bwd_z=np.zeros((n_z, grid.n_bins)),
        converged=True,
        iterations=1,
    )
    h = compute_solver_health(fake, geom, grid, P_pump=0.0, P_signal_in=0.0)
    assert h.energy_status == "ok"
    assert h.energy_residual_ratio == 0.0


def test_health_eta_ase_classification_thresholds(stage1_setup, zero_ase):
    """η_ASE classification: amplifier (≤0.1) vs mixed (0.1-0.3) vs sfs
    (>0.3). For BGU stage 1 with strong pump and tiny seed, we expect a
    measurable but small η_ASE (amplifier regime)."""
    geom, grid, _ = stage1_setup
    r = solve_steady_state(
        geom, grid, P_pump=0.3, P_signal_avg=0.75e-3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    h = compute_solver_health(r, geom, grid, P_pump=0.3, P_signal_in=0.75e-3)
    assert h.ase_conversion_fraction < 0.10, (
        f"BGU stage 1 should be in amplifier regime; "
        f"η_ASE = {h.ase_conversion_fraction:.3f}"
    )
    assert h.regime in ("amplifier", "high_gain")
