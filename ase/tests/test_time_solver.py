"""Mode B (time-dependent) sanity tests + Level 5 (B1+B2) validation."""

import math

import numpy as np
import pytest

from ase.solver_steady import solve_steady_state
from ase.solver_time import (
    _V_G,
    _b1_inter_pulse,
    _b2_pulse,
    _gaussian_pulse_shape,
    solve_time_dependent,
)


def test_high_rep_mode_B_matches_mode_A(stage1_setup, zero_ase):
    """At 100 kHz (period 10 µs ≪ τ = 0.84 ms), Mode B's auto path should
    pick "high_rep_quasi_cw" and return the same answer as Mode A. This
    justifies Mode A as the default at the project's nominal rep rate."""
    geom, grid, _ = stage1_setup
    P_pump, P_sig = 0.3, 0.75e-3

    a = solve_steady_state(
        geom, grid, P_pump=P_pump, P_signal_avg=P_sig,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    b = solve_time_dependent(
        geom, grid, P_pump=P_pump, P_signal_avg=P_sig,
        rep_rate=100e3, pulse_duration=8e-9, pulse_energy=P_sig / 100e3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4, mode="auto",
    )

    assert abs(a.signal_out - b.signal_out) / a.signal_out < 1e-9
    assert abs(a.ase_fwd_out.sum() - b.ase_fwd_out.sum()) / a.ase_fwd_out.sum() < 1e-9


def test_low_rep_extracts_more_per_pulse(stage1_setup, zero_ase):
    """At 10 Hz (period 100 ms ≫ τ), inter-pulse recovery is full and a
    pulse extracts much more energy per shot than the same average power
    would deliver at 100 kHz."""
    geom, grid, _ = stage1_setup
    P_sig = 0.75e-3

    high_rep = solve_time_dependent(
        geom, grid, P_pump=0.3, P_signal_avg=P_sig,
        rep_rate=100e3, pulse_duration=8e-9, pulse_energy=P_sig / 100e3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4, mode="auto",
    )
    low_rep = solve_time_dependent(
        geom, grid, P_pump=0.3, P_signal_avg=P_sig,
        rep_rate=10.0, pulse_duration=8e-9, pulse_energy=P_sig / 10.0,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4, mode="auto",
    )
    # Per-pulse energy: signal_out / rep_rate. At 10 Hz this should be
    # orders of magnitude higher than at 100 kHz with the same pump.
    e_per_pulse_hi = high_rep.signal_out / 100e3
    e_per_pulse_lo = low_rep.signal_out / 10.0
    assert e_per_pulse_lo > 100 * e_per_pulse_hi, (
        f"low-rep pulse energy {e_per_pulse_lo*1e6:.1f} µJ vs "
        f"high-rep {e_per_pulse_hi*1e9:.2f} nJ"
    )


# ── Level 5 (B1+B2) validation ─────────────────────────────────────────


def test_b2_cfl_grid_spacing(stage1_setup):
    """B2 must operate at CFL = 1, i.e. Δt = Δz / v_g exactly. This is
    the precondition that makes the upwind advection exact."""
    geom, grid, _ = stage1_setup
    n_z = 200
    dz = geom.fiber_length / (n_z - 1)
    expected_dt = dz / _V_G

    b2 = _b2_pulse(
        geom=geom, grid=grid,
        n2_pre=np.full(n_z, 0.5),
        P_pump_in=0.3,
        P_pump_z_init=np.full(n_z, 0.3),
        pulse_duration=8e-9, pulse_energy=10e-9,
        n_z=n_z,
    )
    measured_dt = b2.t[1] - b2.t[0]
    assert measured_dt == pytest.approx(expected_dt, rel=1e-12)


def test_b1_long_period_matches_pump_only_mode_A(stage1_setup, zero_ase):
    """For period ≫ τ, B1 starting from a depleted n₂ must drive the
    inversion to the same asymptote that Mode A's pump-only solve gives.
    This validates the rate-equation time-stepping in the limit where the
    answer is known analytically."""
    geom, grid, _ = stage1_setup
    n_z = 200
    P_pump = 0.3
    period = 50e-3   # 50 ms, ≈ 60·τ — plenty of time to asymptote

    # Reference: Mode A with no signal.
    ref = solve_steady_state(
        geom, grid, P_pump=P_pump, P_signal_avg=0.0,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4, n_z=n_z,
    )

    # B1 from fully depleted, integrate for a long time.
    b1 = _b1_inter_pulse(
        geom=geom, grid=grid,
        n2_post=np.zeros(n_z),
        P_pump=P_pump, ase_in_fwd=zero_ase,
        period=period, R_in=0.0, R_out=1e-4, n_z=n_z,
    )
    # Within 1 % at the peak — both pathways solve the same physics.
    assert b1.n2_z.max() == pytest.approx(ref.n2_z.max(), rel=0.01)


def test_b2_small_signal_gain_matches_analytic(stage1_setup):
    """Level 5 small-signal benchmark (docs/ase.md §10.3 limit): for a
    vanishing input pulse with no pump, B2's gain at z=L must equal
    exp(g₀·L) within numerical noise. This validates the (z, t)
    propagator's exact-per-cell exponential update.
    """
    from dataclasses import replace as _replace
    geom_base, grid, _ = stage1_setup
    # Short fiber → modest G₀, clean check. (B2 no longer propagates ASE,
    # so no m_pol override is needed to isolate the signal channel.)
    geom = _replace(geom_base, fiber_length=0.5)
    n_z = 200
    n2_uniform = 0.5
    g0 = (grid.gamma_signal * geom.N_Yb
          * (n2_uniform * grid.sigma_e_signal
             - (1 - n2_uniform) * grid.sigma_a_signal))
    G0_analytic = math.exp(g0 * geom.fiber_length)

    # Vanishingly small input → negligible depletion → pure small-signal G.
    b2 = _b2_pulse(
        geom=geom, grid=grid,
        n2_pre=np.full(n_z, n2_uniform),
        P_pump_in=0.0,
        P_pump_z_init=np.zeros(n_z),
        pulse_duration=5e-9, pulse_energy=1e-18,
        n_z=n_z,
    )
    G_measured = b2.pulse_energy_out / b2.pulse_energy_in
    assert G_measured == pytest.approx(G0_analytic, rel=1e-2), (
        f"G_measured={G_measured:.3f}  G0_analytic={G0_analytic:.3f}"
    )


def test_b2_conserves_energy(stage1_setup):
    """Level 5 sanity: the pulse can only extract energy that's actually
    stored in the inversion (docs/ase.md §1.1). With no pump,
    `E_out - E_in` must equal the energy released by the population
    drop, `Δ⟨n₂⟩ · N · A_core · L · h·ν`, within numerical noise.
    """
    from dataclasses import replace as _replace
    geom_base, grid, _ = stage1_setup
    geom = _replace(geom_base, fiber_length=0.5)
    n_z = 200
    n2_uniform = 0.5
    E_sat = (6.626e-34 * grid.nu_signal * grid.A_eff_signal
             / (grid.sigma_e_signal * grid.gamma_signal))

    for E_in in (E_sat * 0.01, E_sat * 0.5, E_sat * 5.0):
        b2 = _b2_pulse(
            geom=geom, grid=grid,
            n2_pre=np.full(n_z, n2_uniform),
            P_pump_in=0.0,
            P_pump_z_init=np.zeros(n_z),
            pulse_duration=5e-9, pulse_energy=float(E_in),
            n_z=n_z,
        )
        extracted = b2.pulse_energy_out - b2.pulse_energy_in
        # Mean over z of the inversion drop, times the dopant volume:
        dn2_mean = n2_uniform - b2.n2_post_z.mean()
        e_per_ion = 6.626e-34 * grid.nu_signal
        released = dn2_mean * geom.N_Yb * geom.A_core * geom.fiber_length * e_per_ion
        # B2 must not extract more than the inversion released, and the two
        # must agree to better than 5% (numerical pulse-tail truncation).
        rel = abs(extracted - released) / max(released, 1e-30)
        assert extracted <= released * 1.05, (
            f"E_in/E_sat={E_in/E_sat:.2f}  extracted={extracted*1e9:.2f} nJ  "
            f"released={released*1e9:.2f} nJ  (B2 over-extracted!)"
        )
        assert rel < 0.05, (
            f"E_in/E_sat={E_in/E_sat:.2f}  extracted={extracted*1e9:.2f} nJ  "
            f"released={released*1e9:.2f} nJ  rel_err={rel:.3f}"
        )


def test_periodic_converges_in_under_25_cycles(stage1_setup, zero_ase):
    """The B1+B2 periodic loop must converge within the default cycles_max
    for a realistic BGU stage-1 input (high rep, weak seed) — this is the
    practical guarantee that `mode="full"` is usable."""
    geom, grid, _ = stage1_setup
    P_sig = 0.75e-3
    res = solve_time_dependent(
        geom, grid, P_pump=1.0, P_signal_avg=P_sig,
        rep_rate=100e3, pulse_duration=8e-9, pulse_energy=P_sig / 100e3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4, mode="full",
    )
    assert res.converged, f"did not converge in {res.iterations} cycles"
    # And it should produce time-resolved arrays.
    assert res.t is not None
    assert res.P_signal_tz is not None
    assert res.P_signal_tz.shape[1] == 200


def test_gaussian_pulse_shape_normalises_to_energy():
    """Helper sanity: pulse-shape generator integrates to the target energy."""
    t = np.linspace(0, 24e-9, 600)
    shape = _gaussian_pulse_shape(8e-9, 7.5e-9, t)
    integrated = float(np.trapezoid(shape, t))
    assert integrated == pytest.approx(7.5e-9, rel=1e-3)
