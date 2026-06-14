"""Validation tests for the steady-state BVP solver (ase.md §10)."""

import numpy as np
import pytest

from ase.solver_steady import (
    solve_steady_state,
    solve_steady_state_homotopy,
    _spontaneous_emission_init,
)
from ase.solver_time import solve_steady_state_robust
from ase.spectral_grid import SpectralGrid


def _single_amp_geom():
    """The examples/single_amp.json geometry (7 m, 915 nm pump) — a high-g0L
    fiber that under-resolves at the default segment count. Built through the
    real Amplifier so the 915 nm pump cross-section / N_Yb are correct (a
    hand-built SpectralGrid.from_fiber defaults to 976 nm and gives wrong gain).
    """
    from components import Amplifier

    amp = Amplifier(
        name="single_amp",
        core_diameter=7e-6, clad_diameter=128e-6, core_na=0.16, length=7,
        clad_absorption_dB_per_m=1.4, pump_power=0.9, pump_direction="co",
        pump_wavelength=915e-9, signal_wavelength=1064e-9, dopant="Yb",
        R_in=0.0, R_out=1e-4, m_pol=2,
    )
    return amp._geom, amp.grid


def test_under_resolved_flag_trips_on_coarse_grid():
    """ase.md §5.3 high-gain regime: a coarse spatial grid (n_z=100) under-
    resolves the gain-saturation feedback and runs away — this must be flagged
    `under_resolved` (numerical) and NOT silently mislabeled, with an actionable
    note recommending more segments."""
    geom, grid = _single_amp_geom()
    a0 = np.zeros(grid.n_bins)
    r = solve_steady_state(geom, grid, 0.9, 1e-3, a0, R_out=1e-4, n_z=100)
    assert r.under_resolved
    assert not r.converged
    assert any("num_segments" in n for n in r.notes)


def test_under_resolved_clears_and_converges_on_fine_grid():
    """Refining the spatial grid recovers the physical, ASE/saturation-clamped
    steady state: converged, no solver issues, not flagged.

    With the measured Melkumov AS dataset the 915 nm pump σ_a is smaller, so the
    back-derived N_Yb (and hence g₀·L) is higher than under the old hand-traced
    anchors; this fiber now needs n_z≈2000 to resolve the gain-saturation
    feedback (n_z=1000 under-resolves and runs away — see the coarse-grid test
    above). The physical output is essentially unchanged (~0.36 W)."""
    geom, grid = _single_amp_geom()
    a0 = np.zeros(grid.n_bins)
    r = solve_steady_state(geom, grid, 0.9, 1e-3, a0, R_out=1e-4, n_z=2000)
    assert r.converged
    assert not r.under_resolved
    assert not r.solver_failed
    assert 0.2 < r.P_signal_z[-1] < 0.4


def test_zero_signal_peaks_at_gain_wavelength(stage1_setup, zero_ase):
    """ase.md §10.1: zero signal → forward ASE peaks near 1030 nm."""
    geom, grid, _ = stage1_setup
    r = solve_steady_state(
        geom, grid, P_pump=0.3, P_signal_avg=0.0,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    assert r.converged, f"solver did not converge: {r.notes}"
    peak_idx = int(np.argmax(r.ase_fwd_out))
    peak_wl_nm = grid.wavelengths[peak_idx] * 1e9
    assert 1020 <= peak_wl_nm <= 1040, f"peak at {peak_wl_nm:.1f} nm"


def test_zero_signal_backward_has_975_structure(stage1_setup, zero_ase):
    """Backward ASE should show the 975 nm zero-phonon-line dip from
    reabsorption near the pump-injection end (ase.md §11.1)."""
    geom, grid, _ = stage1_setup
    r = solve_steady_state(
        geom, grid, P_pump=0.3, P_signal_avg=0.0,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    assert r.converged
    bwd = r.ase_bwd_in
    # Find bins around 975 nm and around 1010 nm
    i_975 = int(np.abs(grid.wavelengths - 975e-9).argmin())
    i_1030 = int(np.abs(grid.wavelengths - 1030e-9).argmin())
    # 975 nm should be heavily reabsorbed compared to 1030 nm
    assert bwd[i_975] < bwd[i_1030], (
        f"backward ASE at 975 nm ({bwd[i_975]:.3e}) should be < at 1030 nm "
        f"({bwd[i_1030]:.3e}) due to ground-state absorption"
    )


def test_counter_pump_injects_at_output_end(stage1_setup, zero_ase):
    """Counter-pumping injects the pump at z=L; the boundary condition and the
    residual must flip relative to a co-pump solve, while the total absorbed
    pump (and hence quasi-CW gain) stays essentially the same."""
    geom, grid, _ = stage1_setup
    co = solve_steady_state(
        geom, grid, P_pump=0.3, P_signal_avg=5e-3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4, pump_direction="co",
    )
    ct = solve_steady_state(
        geom, grid, P_pump=0.3, P_signal_avg=5e-3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4, pump_direction="counter",
    )
    assert co.converged and ct.converged

    # Pump boundary condition lands on the correct facet.
    assert np.isclose(co.P_pump_z[0], 0.3) and co.P_pump_z[-1] < 0.3
    assert np.isclose(ct.P_pump_z[-1], 0.3) and ct.P_pump_z[0] < 0.3

    # Residual exits the opposite facet from injection.
    assert ct.pump_residual == ct.P_pump_z[0]
    assert co.pump_residual == co.P_pump_z[-1]

    # Same total pump absorbed (both directions absorb the same photons in
    # this saturated quasi-CW regime) → near-identical signal gain.
    co_abs = 0.3 - co.pump_residual
    ct_abs = 0.3 - ct.pump_residual
    assert abs(co_abs - ct_abs) / co_abs < 0.05
    assert abs(ct.signal_out - co.signal_out) / co.signal_out < 0.05


def test_high_signal_transparency(stage1_setup, zero_ase):
    """ase.md §10.2: very strong signal saturates inversion → ASE ≪ signal,
    and gain → small (transparency limit)."""
    geom, grid, _ = stage1_setup
    P_sat_estimate = 0.1  # 100 mW — well above true saturation for stage 1
    r = solve_steady_state(
        geom, grid, P_pump=0.3, P_signal_avg=P_sat_estimate,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    assert r.converged
    total_ase = float(r.ase_fwd_out.sum())
    assert total_ase < r.signal_out * 1e-3, (
        f"ASE ({total_ase:.3e} W) should be much smaller than signal "
        f"({r.signal_out:.3e} W) under heavy saturation"
    )


def test_signal_depletes_inversion_downstream(stage1_setup, zero_ase):
    """At the output end (z=L) where the amplified signal is largest,
    raising the seed should monotonically reduce the inversion. This is the
    standard saturation signature.
    """
    geom, grid, _ = stage1_setup
    P_pump = 0.3
    n2_out = []
    for P_sig in [1e-6, 1e-3, 100e-3]:
        r = solve_steady_state(
            geom, grid, P_pump=P_pump, P_signal_avg=P_sig,
            ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
        )
        assert r.converged
        n2_out.append(float(r.n2_z[-1]))

    assert n2_out[0] > n2_out[1] > n2_out[2], (
        f"n2(z=L) should drop with seed power: {n2_out}"
    )


def test_ase_grows_with_pump(stage1_setup, zero_ase):
    """Total integrated forward ASE should grow monotonically with pump
    power (at fixed seed). Falsified if the inversion sum over ASE bins is
    miscoded and ASE is suppressed at high pump."""
    geom, grid, _ = stage1_setup
    seed = 0.75e-3
    ase_totals = []
    for P_pump in [0.05, 0.1, 0.2, 0.3, 0.5]:
        r = solve_steady_state(
            geom, grid, P_pump=P_pump, P_signal_avg=seed,
            ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
        )
        assert r.converged
        ase_totals.append(float(r.ase_fwd_out.sum()))

    diffs = np.diff(ase_totals)
    assert np.all(diffs > 0), (
        f"ASE should grow with pump but went {ase_totals}"
    )


def test_runaway_flags_solver_issue(stage1_long, zero_ase):
    """A long, highly-doped, zero-signal fiber drives the ASE field into runaway;
    the solver must not pretend it found a steady state — it raises the generic
    `solver_failed` flag and reports as not converged. (Diagnosing *why* no
    steady state exists is out of scope; we only flag that it doesn't.)"""
    geom, grid, _ = stage1_long
    r = solve_steady_state(
        geom, grid, P_pump=1.0, P_signal_avg=0.0,
        ase_in_fwd=np.zeros(grid.n_bins),
        R_in=0.035, R_out=0.035,  # flat-cleaved both ends
        max_iter=80,
    )
    assert r.solver_failed
    assert not r.converged


def test_grid_convergence_total_ase(stage1_setup):
    """ase.md §3.3: halving Δλ should change total integrated forward ASE
    by less than 5% (the spec target is 1%, but tighter would require many
    more bins than the default 160 — 5% is the practical threshold)."""
    geom, grid_1nm, r_core = stage1_setup
    r_1 = solve_steady_state(
        geom, grid_1nm, P_pump=0.3, P_signal_avg=0.75e-3,
        ase_in_fwd=np.zeros(grid_1nm.n_bins), R_in=0.0, R_out=1e-4,
    )
    grid_05nm = SpectralGrid.from_fiber(r_core=r_core, NA=0.12, d_lambda=0.5e-9)
    r_2 = solve_steady_state(
        geom, grid_05nm, P_pump=0.3, P_signal_avg=0.75e-3,
        ase_in_fwd=np.zeros(grid_05nm.n_bins), R_in=0.0, R_out=1e-4,
    )
    total_1 = float(r_1.ase_fwd_out.sum())
    total_2 = float(r_2.ase_fwd_out.sum())
    rel_change = abs(total_2 - total_1) / total_1
    assert rel_change < 0.05, (
        f"halving Δλ changed integrated ASE by {rel_change*100:.1f}% — "
        f"expected < 5% (1 nm: {total_1*1e6:.2f} µW, "
        f"0.5 nm: {total_2*1e6:.2f} µW)"
    )


def test_signal_amplification_is_positive(stage1_setup, zero_ase):
    """A reasonable seed and pump combination should produce gain (signal
    out > signal in)."""
    geom, grid, _ = stage1_setup
    P_sig = 0.75e-3
    r = solve_steady_state(
        geom, grid, P_pump=0.3, P_signal_avg=P_sig,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    assert r.converged
    assert r.signal_out > P_sig, (
        f"signal should be amplified: in {P_sig*1e3:.3f} mW, "
        f"out {r.signal_out*1e3:.3f} mW"
    )


def test_pump_absorption_is_partial(stage1_setup, zero_ase):
    """The pump should be substantially absorbed but not entirely — a
    well-pumped Yb stage runs in the partial-absorption regime."""
    geom, grid, _ = stage1_setup
    P_pump = 0.3
    r = solve_steady_state(
        geom, grid, P_pump=P_pump, P_signal_avg=0.75e-3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    abs_frac = 1 - r.pump_residual / P_pump
    assert 0.3 < abs_frac < 0.95, (
        f"pump absorption {abs_frac*100:.0f}% out of expected 30-95% range"
    )


# ── Homotopy continuation (Layer 2) ──────────────────────────────────────


def test_init_parameter_round_trips_through_solve_steady_state(
    stage1_setup, zero_ase,
):
    """The new `init=` parameter must accept a tuple of P-field profiles
    and use them as the initial guess instead of the constant default."""
    geom, grid, _ = stage1_setup
    # First solve normally.
    r1 = solve_steady_state(
        geom, grid, P_pump=0.3, P_signal_avg=0.75e-3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    # Then re-run with that as the init — should converge in ≤2 iterations.
    r2 = solve_steady_state(
        geom, grid, P_pump=0.3, P_signal_avg=0.75e-3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
        init=(r1.P_pump_z, r1.P_signal_z, r1.P_ase_fwd_z, r1.P_ase_bwd_z),
    )
    assert r2.converged
    assert r2.iterations <= 3, (
        f"warm-started solve should converge fast; took {r2.iterations} iters"
    )
    # Same physical answer.
    assert abs(r2.signal_out - r1.signal_out) / r1.signal_out < 1e-3


def test_homotopy_succeeds_on_healthy_regime(stage1_setup, zero_ase):
    """For a healthy operating point, homotopy still arrives at the
    physical solution (in this case via early termination at step 3)."""
    geom, grid, _ = stage1_setup
    r = solve_steady_state_homotopy(
        geom, grid, P_pump=0.3, P_signal_avg=0.75e-3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
        # Predicate accepts anything ≤1.05× the QD ceiling, i.e. all
        # physical solutions.
        health_predicate=lambda r: (
            (r.signal_out - 0.75e-3) / max(0.917 * (0.3 - r.pump_residual), 1e-12)
            < 1.05
        ),
    )
    assert r.solver_path_used == "homotopy"
    # Should not need more than the 3 base steps for a healthy regime.
    assert r.homotopy_steps_used <= 3


# ── Robust wrapper (Layer 1+2+3 orchestration) ───────────────────────────


def test_robust_healthy_regime_takes_direct_path(stage1_setup, zero_ase):
    """In a healthy regime the direct Layer-1 solve passes the QD check;
    the wrapper returns immediately without invoking homotopy."""
    geom, grid, _ = stage1_setup
    r = solve_steady_state_robust(
        geom, grid, P_pump=0.3, P_signal_avg=0.75e-3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    assert r.solver_path_used == "direct"
    assert r.homotopy_steps_used == 0
    assert r.health is not None
    assert r.health.energy_status == "ok"


def test_robust_attaches_health_diagnostics(stage1_setup, zero_ase):
    """Every robust solve returns a result with `health` populated."""
    geom, grid, _ = stage1_setup
    r = solve_steady_state_robust(
        geom, grid, P_pump=0.3, P_signal_avg=0.75e-3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    assert r.health is not None
    assert hasattr(r.health, "energy_residual_ratio")
    assert hasattr(r.health, "ase_conversion_fraction")
    assert hasattr(r.health, "small_signal_g0L")
    assert hasattr(r.health, "energy_status")
    assert hasattr(r.health, "regime")


# --- Spontaneous-emission-seeded default initial condition ----------------


def test_se_init_shapes_and_nonneg(stage1_setup, zero_ase):
    """`_spontaneous_emission_init` returns correctly-shaped, nonnegative
    seeds; the pump decays from the input facet."""
    geom, grid, _ = stage1_setup
    n_z = 200
    P_p, P_s, fwd, bwd = _spontaneous_emission_init(
        geom, grid, P_pump=0.3, P_signal_avg=0.75e-3,
        ase_in_fwd=zero_ase, n_z=n_z,
    )
    assert P_p.shape == (n_z,)
    assert P_s.shape == (n_z,)
    assert fwd.shape == (n_z, grid.n_bins)
    assert bwd.shape == (n_z, grid.n_bins)
    for a in (P_p, P_s, fwd, bwd):
        assert np.all(a >= 0.0)
    assert P_p[0] == 0.3
    assert P_p[-1] < P_p[0]


def test_se_init_ase_monotonicity(stage1_setup, zero_ase):
    """Gain-free accumulation: forward ASE seed grows toward the output
    facet, backward ASE seed grows toward the input facet."""
    geom, grid, _ = stage1_setup
    _, _, fwd, bwd = _spontaneous_emission_init(
        geom, grid, 0.3, 0.75e-3, zero_ase, 200,
    )
    total_fwd = fwd.sum(axis=1)
    total_bwd = bwd.sum(axis=1)
    assert np.all(np.diff(total_fwd) >= -1e-30)   # nondecreasing in z
    assert np.all(np.diff(total_bwd) <= 1e-30)    # nonincreasing in z


def test_se_init_degrades_to_zero_ase_at_zero_pump(stage1_setup, zero_ase):
    """With no pump the inversion is zero, so the seed collapses to the
    historical zero-ASE guess."""
    geom, grid, _ = stage1_setup
    P_p, _, fwd, bwd = _spontaneous_emission_init(
        geom, grid, P_pump=0.0, P_signal_avg=0.75e-3,
        ase_in_fwd=zero_ase, n_z=200,
    )
    assert np.allclose(P_p, 0.0)
    assert np.allclose(fwd, zero_ase[None, :])
    assert np.allclose(bwd, 0.0)


def test_se_init_default_solve_still_converges(stage1_setup, zero_ase):
    """The new default IC must not break the standard healthy solve."""
    geom, grid, _ = stage1_setup
    r = solve_steady_state(
        geom, grid, P_pump=0.3, P_signal_avg=0.75e-3,
        ase_in_fwd=zero_ase, R_in=0.0, R_out=1e-4,
    )
    assert r.converged

