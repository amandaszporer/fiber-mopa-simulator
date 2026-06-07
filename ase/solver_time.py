"""
Time-dependent ASE solver (Level 5).

Implements the runtime-mode dispatcher described in docs/ase.md §4.2/§4.3:

  - "high_rep_quasi_cw": one Mode A solve at the time-averaged signal power.
                         Exact when period ≪ τ; auto-selected at the BGU
                         nominal 100 kHz operating point.
  - "periodic"         : true B1 (inter-pulse rate-equation time-stepping)
                         + B2 (Lax-Wendroff (z, t) pulse PDE), iterated
                         until the pre-pulse n₂(z) profile is stable.
  - "full"             : alias for "periodic" — force the B2 path even at
                         high rep (useful for pulse-shape studies).

`mode="auto"` picks the cheapest branch that's accurate for the input
rep rate.

All branches return a `SteadyResult`-shaped record so `Amplifier` can
switch on `mode=` without restructuring its output. B2 also populates
the optional `t`, `P_signal_tz`, `P_ase_fwd_tz` fields on the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .solver_health import SolverHealth, compute_solver_health
from .solver_steady import (
    AmplifierGeometry,
    SteadyResult,
    solve_pfield_fixed_n2,
    solve_steady_state,
    solve_steady_state_homotopy,
)
from .spectral_grid import SpectralGrid


_H = 6.626e-34
_N_GLASS = 1.45                  # silica-fiber refractive index (matches components.py)
_C = 3e8
_V_G = _C / _N_GLASS             # group velocity in the fiber [m/s]


def _absorption_emission_rates(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump_z: np.ndarray,
    P_signal_z: np.ndarray,
    P_ase_fwd_z: Optional[np.ndarray] = None,
    P_ase_bwd_z: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-z absorption and emission rates summed over the active channels.

    Returns (R_abs, R_em) each shape [n_z], in units of [1/s]. The
    inversion rate equation is then
        dn₂/dt = (1 - n₂)·R_abs - n₂·R_em - n₂/τ.

    The pump and signal channels are always included. The ASE channels
    are summed in only when both `P_ase_fwd_z` and `P_ase_bwd_z` are
    supplied — B1 (inter-pulse) passes them; B2 (the ns pulse step)
    omits them, because ASE is negligible on the pulse timescale (see
    `_b2_pulse` and docs/ase.md §4.2).

    This is the time-dependent counterpart of `_compute_n2_local` in
    `solver_steady.py`, which solves the same balance for steady state.
    """
    A_dope = geom.A_core
    inv_h_A = 1.0 / (_H * A_dope)

    pump_flux = P_pump_z * inv_h_A / grid.nu_pump
    R_abs = geom.gamma_pump * grid.sigma_a_pump * pump_flux
    R_em = geom.gamma_pump * grid.sigma_e_pump * pump_flux

    sig_flux = P_signal_z * inv_h_A / grid.nu_signal
    R_abs = R_abs + grid.gamma_signal * grid.sigma_a_signal * sig_flux
    R_em = R_em + grid.gamma_signal * grid.sigma_e_signal * sig_flux

    if P_ase_fwd_z is not None and P_ase_bwd_z is not None:
        inv_nu = 1.0 / grid.frequencies
        ase_total = P_ase_fwd_z + P_ase_bwd_z
        ase_flux = ase_total * (inv_h_A * inv_nu)[None, :]
        R_abs = R_abs + np.sum(grid.gamma * grid.sigma_a * ase_flux, axis=1)
        R_em = R_em + np.sum(grid.gamma * grid.sigma_e * ase_flux, axis=1)

    return R_abs, R_em


@dataclass
class _B1Result:
    """Inter-pulse recovery state at the moment the next pulse arrives."""
    n2_z: np.ndarray
    P_pump_z: np.ndarray
    P_ase_fwd_z: np.ndarray
    P_ase_bwd_z: np.ndarray
    steps: int
    notes: list[str] = field(default_factory=list)


def _b1_inter_pulse(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    n2_post: np.ndarray,
    P_pump: float,
    ase_in_fwd: np.ndarray,
    period: float,
    R_in: float,
    R_out: float,
    n_z: int,
    dt_init: float = 1e-6,
    dt_max: float = 1e-5,
    dt_min: float = 1e-9,
    max_steps: int = 20000,
    p_solve_drift_tol: float = 0.05,
) -> _B1Result:
    """Inter-pulse rate-equation time-stepping (docs/ase.md §4.2 B1).

    Starts from the post-pulse depleted `n2_post(z)` and evolves under
    pump + ASE (no signal) for duration `period`. Returns the pre-pulse
    state — `n2(z)` and the P-field profiles at t = period — for B2 to
    consume as its initial condition.

    Operator splitting: the optical fields equilibrate on the ~30 ns transit
    time, which is far faster than the n₂ evolution (τ = 0.84 ms). We
    therefore re-solve the spatial P-field only when `n₂` has drifted by
    `p_solve_drift_tol` from the last solve; the n₂ rate equation is
    integrated with a small adaptive Euler step in between.
    """
    notes: list[str] = []
    n2_z = np.asarray(n2_post, dtype=float).copy()
    np.clip(n2_z, 0.0, 1.0, out=n2_z)

    # Initial P-field solve and rate-coefficient evaluation.
    def _resolve_p():
        _, p_pump, _, p_fwd, p_bwd = solve_pfield_fixed_n2(
            geom, grid, n2_z,
            P_pump_in=P_pump, P_signal_in=0.0,
            ase_in_fwd=ase_in_fwd,
            R_in=R_in, R_out=R_out, n_z=n_z,
        )
        r_abs, r_em = _absorption_emission_rates(
            geom, grid, p_pump, np.zeros(n_z), p_fwd, p_bwd,
        )
        return p_pump, p_fwd, p_bwd, r_abs, r_em

    P_pump_z, P_ase_fwd_z, P_ase_bwd_z, R_abs, R_em = _resolve_p()
    n2_at_psolve = n2_z.copy()

    t = 0.0
    dt = dt_init
    step = 0
    while t < period and step < max_steps:
        dt_step = min(dt, period - t)
        dn2_dt = (1.0 - n2_z) * R_abs - n2_z * R_em - n2_z / geom.tau
        delta = dt_step * dn2_dt
        max_abs_delta = float(np.max(np.abs(delta)))

        # Reject step if the change exceeds 1% absolute (n₂ ∈ [0, 1]).
        if max_abs_delta > 0.01 and dt_step > dt_min:
            dt = max(dt * 0.5, dt_min)
            continue

        n2_z = n2_z + delta
        np.clip(n2_z, 0.0, 1.0, out=n2_z)
        t += dt_step
        step += 1

        # Re-solve P-field if n₂ has drifted enough that R_abs/R_em are stale.
        rel_drift = float(np.max(np.abs(n2_z - n2_at_psolve)
                                  / np.maximum(n2_at_psolve, 1e-3)))
        if rel_drift > p_solve_drift_tol:
            P_pump_z, P_ase_fwd_z, P_ase_bwd_z, R_abs, R_em = _resolve_p()
            n2_at_psolve = n2_z.copy()

        # Grow dt toward the 1% target.
        if max_abs_delta > 0:
            dt = dt * min(2.0, max(0.5, 0.005 / max(max_abs_delta, 1e-15)))
        dt = min(max(dt, dt_min), dt_max)

    if step >= max_steps:
        notes.append(f"B1 hit max_steps={max_steps}; n2 evolution truncated.")

    # Final P-field at the converged n₂ — this is what B2 will see.
    P_pump_z, P_ase_fwd_z, P_ase_bwd_z, _, _ = _resolve_p()

    return _B1Result(
        n2_z=n2_z,
        P_pump_z=P_pump_z,
        P_ase_fwd_z=P_ase_fwd_z,
        P_ase_bwd_z=P_ase_bwd_z,
        steps=step,
        notes=notes,
    )


def _gaussian_pulse_shape_offset(
    pulse_duration: float,
    pulse_energy: float,
    t_axis: np.ndarray,
    t_center: float,
) -> np.ndarray:
    """Gaussian pulse centred at `t_center` and normalised to `pulse_energy`.

    FWHM = `pulse_duration` so peak power matches the SHAPE_FACTOR = 0.94
    convention used elsewhere in the project (components.py:59).
    """
    sigma_t = pulse_duration / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    shape = np.exp(-0.5 * ((t_axis - t_center) / sigma_t) ** 2)
    integral = float(np.trapezoid(shape, t_axis))
    if integral <= 0:
        return np.zeros_like(t_axis)
    return shape * (pulse_energy / integral)


def _gaussian_pulse_shape(
    pulse_duration: float,
    pulse_energy: float,
    t_axis: np.ndarray,
) -> np.ndarray:
    """Centred Gaussian pulse (in the middle of `t_axis`) normalised to
    `pulse_energy`. Convenience wrapper used by the test suite."""
    t_center = 0.5 * (t_axis[0] + t_axis[-1])
    return _gaussian_pulse_shape_offset(
        pulse_duration, pulse_energy, t_axis, t_center,
    )


@dataclass
class _B2Result:
    """Outputs of one B2 pulse propagation.

    B2 propagates the signal pulse and the inversion only — ASE is
    negligible on the ns pulse timescale and is handled exclusively by
    B1 (see `_b2_pulse`). Hence no ASE fields here.
    """
    t: np.ndarray                          # [n_t]
    z: np.ndarray                          # [n_z]
    n2_post_z: np.ndarray                  # [n_z]
    P_signal_tz: np.ndarray                # [n_t, n_z]
    pulse_energy_out: float
    pulse_energy_in: float
    notes: list[str] = field(default_factory=list)


def _b2_pulse(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    n2_pre: np.ndarray,
    P_pump_in: float,
    P_pump_z_init: np.ndarray,
    pulse_duration: float,
    pulse_energy: float,
    n_z: int,
    pulse_window_factor: float = 6.0,
) -> _B2Result:
    """Time-dependent Lax-Wendroff pulse solver (docs/ase.md §4.2 B2).

    Part 2 of the two-part MOPA model: the signal pulse passes through the
    amplifier and extracts energy from the inversion built up by B1.

    **ASE is neglected here.** ASE accumulates on the µs–ms inter-pulse
    timescale; over a 4–8 ns pulse it has no time to build up meaningfully.
    All ASE accounting therefore happens in B1 (`_b1_inter_pulse`). B2
    propagates only two channels — the pump (CW, stays on through the
    pulse) and the signal pulse — coupled to the inversion n₂(z, t).

    Channels are advected at v_g = c / n_glass. CFL is set to exactly 1
    (Δt = Δz / v_g), which turns the upwind difference into the method of
    characteristics for the propagating part; the gain is applied via the
    exact per-cell exponential `exp((g − α_bg)·Δz)`.

    Initial condition: n₂(z) = n2_pre (set by B1), pump profile = the
    quasi-CW profile that existed at the moment the pulse arrived (also
    from B1). The signal channel starts at zero everywhere; the pulse
    enters via the z=0 boundary condition over the time window.

    `pulse_window_factor` controls how many pulse-durations of tail to
    simulate. 6× FWHM (3σ either side, ≈ ±5σ) keeps Gaussian truncation
    error below 0.1% of the integrated energy.
    """
    notes: list[str] = []
    L = geom.fiber_length
    dz = L / (n_z - 1)
    z = np.linspace(0.0, L, n_z)
    dt = dz / _V_G

    # Window must cover (a) the pulse's temporal extent at z=0 and (b) the
    # fiber transit time, otherwise the trailing edge never reaches z=L.
    transit_time = L / _V_G
    pulse_window = pulse_window_factor * pulse_duration + transit_time
    n_t = int(math.ceil(pulse_window / dt)) + 1
    t = np.linspace(0.0, (n_t - 1) * dt, n_t)
    # Centre the input pulse early in the window so it has room to traverse.
    pulse_center = 0.5 * pulse_window_factor * pulse_duration
    signal_in_t = _gaussian_pulse_shape_offset(
        pulse_duration, pulse_energy, t, pulse_center,
    )
    pulse_energy_in = float(np.trapezoid(signal_in_t, t))

    # State arrays — pump (CW) and signal pulse only.
    n2_z = np.asarray(n2_pre, dtype=float).copy()
    np.clip(n2_z, 0.0, 1.0, out=n2_z)
    P_pump_z = np.asarray(P_pump_z_init, dtype=float).copy()
    P_signal_z = np.zeros(n_z)

    # Per-step record of the signal pulse (for the pulse-shape plot).
    P_signal_tz = np.zeros((n_t, n_z))
    P_signal_tz[0, 0] = signal_in_t[0]
    # The first row records the initial (pulse-not-yet-arrived) state.

    alpha_bg = geom.alpha_bg
    N = geom.N_Yb
    inv_tau = 1.0 / geom.tau

    # Pre-vectorise: per-channel gain coefficients used at every step.
    g_pump_coef_e = geom.gamma_pump * N * grid.sigma_e_pump
    g_pump_coef_a = geom.gamma_pump * N * grid.sigma_a_pump
    g_sig_coef_e = grid.gamma_signal * N * grid.sigma_e_signal
    g_sig_coef_a = grid.gamma_signal * N * grid.sigma_a_signal

    def _propagate(g_net, P_up):
        """Exact one-cell solution to dP/dz = g_net·P along a CFL=1
        characteristic. `g_net = g - α_bg`. Returns P after traversing
        one cell of length Δz from upstream amplitude P_up.
        """
        # exp(g_net·dz) is the per-cell amplification factor.
        return P_up * (1.0 + np.expm1(g_net * dz))

    for k in range(1, n_t):
        one_minus = 1.0 - n2_z
        # Per-channel gain coefficient at current n₂(z).
        g_pump = g_pump_coef_e * n2_z - g_pump_coef_a * one_minus           # [n_z]
        g_sig = g_sig_coef_e * n2_z - g_sig_coef_a * one_minus              # [n_z]

        # At CFL=1 the characteristic from (z_{i-1}, t) to (z_i, t+Δt) carries
        # amplitude P_old[i-1]. The medium it traverses has gain coefficient
        # g(z_i) (destination value). The exact exponential `exp(g_net·dz)`
        # is what makes the small-signal limit recover the analytic
        # Frantz-Nodvik gain G₀ = exp(g₀·L) (docs/ase.md §10.3).
        P_pump_new = np.empty(n_z)
        P_pump_new[1:] = _propagate(g_pump[1:] - alpha_bg, P_pump_z[:-1])
        P_pump_new[0] = P_pump_in

        P_signal_new = np.empty(n_z)
        P_signal_new[1:] = _propagate(g_sig[1:] - alpha_bg, P_signal_z[:-1])
        P_signal_new[0] = signal_in_t[k]

        # Floor at zero (numerical safety).
        np.maximum(P_pump_new, 0.0, out=P_pump_new)
        np.maximum(P_signal_new, 0.0, out=P_signal_new)

        # n₂ rate-equation step at the *previous* P-field (operator splitting:
        # advect first, then react). ASE is omitted from the rate balance —
        # over the ns pulse it cannot drain the inversion. dt is the CFL
        # Δt = 36–72 ps, far below τ = 0.84 ms, so forward Euler is stable.
        R_abs, R_em = _absorption_emission_rates(
            geom, grid, P_pump_z, P_signal_z,
        )
        dn2 = dt * ((1.0 - n2_z) * R_abs - n2_z * R_em - n2_z * inv_tau)
        n2_z = n2_z + dn2
        np.clip(n2_z, 0.0, 1.0, out=n2_z)

        # Commit the new P-field state.
        P_pump_z = P_pump_new
        P_signal_z = P_signal_new

        # Record the signal pulse profile.
        P_signal_tz[k, :] = P_signal_z

    pulse_energy_out = float(np.trapezoid(P_signal_tz[:, -1], t))

    return _B2Result(
        t=t,
        z=z,
        n2_post_z=n2_z,
        P_signal_tz=P_signal_tz,
        pulse_energy_out=pulse_energy_out,
        pulse_energy_in=pulse_energy_in,
        notes=notes,
    )


def solve_time_dependent(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump: float,
    P_signal_avg: float,
    rep_rate: float,
    pulse_duration: float,
    pulse_energy: float,
    ase_in_fwd: np.ndarray,
    R_in: float = 0.0,
    R_out: float = 1e-4,
    mode: str = "auto",
    n_z: int = 200,
    cycles_max: int = 25,
    cycles_tol: float = 1e-3,
) -> SteadyResult:
    """Run a time-dependent solve and return a steady-state-shaped result.

    See module docstring for the meaning of each mode.
    """
    period = 1.0 / rep_rate if rep_rate > 0 else float("inf")
    tau = geom.tau

    if mode == "auto":
        if period < 0.1 * tau:
            mode = "high_rep_quasi_cw"
        else:
            mode = "periodic"

    if mode == "high_rep_quasi_cw":
        # High-rep delegates to Mode A — use the layered robust wrapper so
        # we get the energy-conservation diagnostics + homotopy fallback
        # automatically, with no path-different behaviour vs `mode="steady"`.
        return solve_steady_state_robust(
            geom=geom, grid=grid,
            P_pump=P_pump, P_signal_avg=P_signal_avg,
            ase_in_fwd=ase_in_fwd, R_in=R_in, R_out=R_out, n_z=n_z,
        )

    # "full" forces the B1+B2 path even at high rep — useful for pulse-shape
    # studies; "low_rep_recovery" kept as a back-compat alias for one-cycle
    # periodic (sufficient when period ≫ τ).
    if mode in ("periodic", "full", "low_rep_recovery"):
        eff_cycles_max = 1 if mode == "low_rep_recovery" else cycles_max
        return _periodic_steady_state(
            geom=geom, grid=grid,
            P_pump=P_pump, P_signal_avg=P_signal_avg,
            rep_rate=rep_rate,
            pulse_duration=pulse_duration,
            pulse_energy=pulse_energy,
            ase_in_fwd=ase_in_fwd,
            R_in=R_in, R_out=R_out, n_z=n_z,
            cycles_max=eff_cycles_max, cycles_tol=cycles_tol,
        )

    raise ValueError(f"Unknown time-dependent mode: {mode!r}")


def _periodic_steady_state(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump: float,
    P_signal_avg: float,
    rep_rate: float,
    pulse_duration: float,
    pulse_energy: float,
    ase_in_fwd: np.ndarray,
    R_in: float,
    R_out: float,
    n_z: int,
    cycles_max: int,
    cycles_tol: float,
) -> SteadyResult:
    """B1 recovery + B2 pulse, iterated to periodic steady state (§4.3).

    On each cycle:
      1. **B1** — rate-equation time-stepping over the inter-pulse interval
         starting from the post-pulse depleted `n₂(z)`. Returns the
         pre-pulse `n₂(z)` and recovered ASE.
      2. **B2** — Lax-Wendroff (z, t) pulse PDE seeded by the B1 output.
         Returns the post-pulse `n₂(z)`, the time-resolved pulse at z=L,
         and the ASE state after the pulse.
      3. Convergence: stop when the pre-pulse `n₂(z)` is stable between
         consecutive cycles to `cycles_tol`.

    The returned `SteadyResult` carries:
      - `P_signal_z` linearly interpolated from cycle-averaged input to
        output power (consistent with how downstream report code expects a
        z-profile);
      - `n2_z` = post-pulse depleted profile;
      - `P_ase_fwd_z` / `P_ase_bwd_z` from the recovered (pre-pulse) state;
      - time-resolved `t` / `P_signal_tz` / `P_ase_fwd_tz` from the last
        B2 cycle (Optional fields on SteadyResult).
    """
    notes: list[str] = []
    period = 1.0 / rep_rate

    # Warm start. Use the homotopy continuation (Layer 2 of the robust
    # cascade, but not the full robust wrapper — that would recurse into
    # this function via Layer 3). Homotopy by itself only calls
    # `solve_steady_state`, so no recursion risk. The benefit: when the
    # raw Mode A would find an unphysical fixed point near the parasitic
    # edge, homotopy walks past it via pump-only → +signal → +ASE
    # continuation, giving B1+B2 a physical starting n₂(z) profile.
    # The existing NaN-guard on B1+B2 below is kept as a final backstop
    # for genuinely past-parasitic configs.
    from .solver_health import compute_solver_health
    warmup = solve_steady_state_homotopy(
        geom=geom, grid=grid,
        P_pump=P_pump, P_signal_avg=P_signal_avg,
        ase_in_fwd=ase_in_fwd, R_in=R_in, R_out=R_out, n_z=n_z,
        health_predicate=lambda r: compute_solver_health(
            r, geom, grid, P_pump, P_signal_avg
        ).energy_status != "violation",
    )

    # If the system is past parasitic-lasing threshold, no physical periodic
    # steady state exists. Mode A's runaway clamp gives a sensible (if
    # unphysical) report; B1+B2 will diverge to NaN under fp64 overflow.
    # Return the warm-start verbatim with a clarifying note.
    if warmup.parasitic_lasing or not warmup.converged:
        warmup.notes.append(
            "B1+B2 skipped: system past parasitic-lasing threshold "
            "(or Mode A did not converge). No physical periodic steady state."
        )
        return warmup

    n2_post = warmup.n2_z.copy()
    last_b1: Optional[_B1Result] = None
    last_b2: Optional[_B2Result] = None
    n2_pre_prev: Optional[np.ndarray] = None
    converged_periodic = False

    cycle = 0
    for cycle in range(cycles_max):
        b1 = _b1_inter_pulse(
            geom=geom, grid=grid,
            n2_post=n2_post,
            P_pump=P_pump,
            ase_in_fwd=ase_in_fwd,
            period=period,
            R_in=R_in, R_out=R_out, n_z=n_z,
        )
        notes.extend(b1.notes)

        # B1's spatial sub-solve has no internal runaway clamp (the operator
        # split makes it cheaper than Mode A's iterated solver). When the
        # system is near parasitic but Mode A barely converged, B1 can still
        # overflow during the rate-equation step. Detect and bail to the
        # Mode A warmup (clamped but finite).
        if not (np.all(np.isfinite(b1.n2_z))
                and np.all(np.isfinite(b1.P_ase_fwd_z))):
            warmup.notes.append(
                f"B1 numerical blowup in cycle {cycle + 1}: system is on the "
                f"parasitic-lasing edge — Mode A barely converged but B1's "
                f"high-n₂ regime overflows. Returning Mode A warmup result."
            )
            return warmup

        b2 = _b2_pulse(
            geom=geom, grid=grid,
            n2_pre=b1.n2_z,
            P_pump_in=P_pump,
            P_pump_z_init=b1.P_pump_z,
            pulse_duration=pulse_duration,
            pulse_energy=pulse_energy,
            n_z=n_z,
        )
        notes.extend(b2.notes)

        if not np.all(np.isfinite(b2.n2_post_z)):
            warmup.notes.append(
                f"B2 numerical blowup in cycle {cycle + 1}: pulse propagation "
                f"overflowed (system on parasitic edge). Returning Mode A "
                f"warmup result."
            )
            return warmup

        n2_post = b2.n2_post_z
        last_b1 = b1
        last_b2 = b2

        if n2_pre_prev is not None:
            rel = float(np.max(np.abs(b1.n2_z - n2_pre_prev)
                                / np.maximum(b1.n2_z, 1e-15)))
            if rel < cycles_tol:
                converged_periodic = True
                break
        n2_pre_prev = b1.n2_z.copy()

    if last_b1 is None or last_b2 is None:
        raise RuntimeError("periodic loop exited without producing a cycle")

    # Cycle-averaged signal output for the SteadyResult contract: the average
    # signal power at z=L is (pulse energy delivered per shot) × rep_rate.
    P_signal_out_avg = last_b2.pulse_energy_out * rep_rate

    z = np.linspace(0.0, geom.fiber_length, n_z)
    P_signal_z_avg = np.linspace(P_signal_avg, P_signal_out_avg, n_z)

    parasitic_lasing, parasitic_dB = _check_parasitic_b2(
        geom, grid, last_b1.n2_z, z, R_in, R_out,
    )

    notes.append(
        f"periodic B1+B2 {'converged' if converged_periodic else 'capped'} "
        f"after {cycle + 1} cycles; pulse extracted "
        f"{(last_b2.pulse_energy_out - last_b2.pulse_energy_in) * 1e9:.2f} nJ "
        f"(B1 steps in last cycle: {last_b1.steps})"
    )

    # ASE is a B1-only product (B2 neglects it on the ns pulse timescale):
    # the stage's forward/backward ASE spectra come from the inter-pulse
    # recovery, while the time-resolved signal pulse comes from B2.
    return SteadyResult(
        z=z,
        n2_z=last_b2.n2_post_z,
        P_pump_z=last_b1.P_pump_z,
        P_signal_z=P_signal_z_avg,
        P_ase_fwd_z=last_b1.P_ase_fwd_z,
        P_ase_bwd_z=last_b1.P_ase_bwd_z,
        converged=converged_periodic,
        iterations=cycle + 1,
        parasitic_lasing=parasitic_lasing,
        parasitic_gain_max_dB=parasitic_dB,
        notes=notes,
        t=last_b2.t,
        P_signal_tz=last_b2.P_signal_tz,
    )


def _check_parasitic_b2(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    n2_z: np.ndarray,
    z: np.ndarray,
    R_in: float,
    R_out: float,
) -> tuple[bool, float]:
    """Round-trip-gain parasitic check on the pre-pulse n₂(z), reusing the
    same formula as `solver_steady._check_parasitic_lasing`. Inlined here so
    we don't need to import a private helper across modules."""
    if R_in <= 0 or R_out <= 0:
        return False, 0.0
    one_minus = 1.0 - n2_z
    g_lam_z = (
        grid.gamma[None, :] * geom.N_Yb
        * (n2_z[:, None] * grid.sigma_e[None, :]
           - one_minus[:, None] * grid.sigma_a[None, :])
    )
    g_lam_z = g_lam_z - geom.alpha_bg
    int_g_lam = np.trapezoid(g_lam_z, z, axis=0)
    G_lam = np.exp(int_g_lam)
    round_trip = G_lam ** 2 * R_in * R_out
    max_round_trip = float(round_trip.max())
    max_g_dB = float(10 * np.log10(max(max_round_trip, 1e-300)))
    return max_round_trip >= 1.0, max_g_dB


# ── Layered robust solver (Layers 1+2+3) ────────────────────────────────
#
# Top-level entry point for any caller that wants energy-conservative
# results. See `docs/ase.md` Part II §13b for the design rationale and
# literature backing.
#
#   Layer 1: direct `solve_steady_state` + health classification.
#   Layer 2: if `energy_status == "violation"`, retry with the Ren et al.
#            2015 homotopy continuation (pump-only → +signal → +ASE →
#            parameter continuation → Xu et al. 2014 linear-gain
#            multistart). Picks the first sub-step that passes the
#            energy check.
#   Layer 3: if homotopy still violates, dispatch to the time-marching
#            B1+B2 cycle (a fully-time-dependent IVP, unique attractor
#            by construction — cannot land on an energy-violating fixed
#            point). The result is the authoritative steady state.
#   "All-failed" path: B1+B2 itself bails out (parasitic guard triggers)
#            means the system has no physical steady state. Return Mode
#            A's clamped result and surface `solver_path_used="all_failed"`
#            + the violation flag so the report decoration shows it.
#
# References inline in the implementation.


def solve_steady_state_robust(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump: float,
    P_signal_avg: float,
    ase_in_fwd: np.ndarray,
    R_in: float = 0.0,
    R_out: float = 1e-4,
    n_z: int = 200,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> SteadyResult:
    """Production entry point for steady-state amplifier solves.

    Wraps the bare iterative-shooting BVP with a three-layer cascade:
    direct solve + health check, then homotopy continuation (Ren et al.
    2015) on violation, then a time-marching B1+B2 arbiter (PyFiberAmp
    `DynamicSimulation` pattern) on continued violation.

    Returns a `SteadyResult` whose `health` field carries the
    energy-residual / η_ASE / g₀·L diagnostics and whose
    `solver_path_used` field records which layer produced the answer
    (`"direct"`, `"homotopy"`, `"time_marching_arbiter"`, or
    `"all_failed"`).
    """
    # ── Layer 1: direct solve + health classification ─────────────────
    r1 = solve_steady_state(
        geom=geom, grid=grid,
        P_pump=P_pump, P_signal_avg=P_signal_avg,
        ase_in_fwd=ase_in_fwd,
        R_in=R_in, R_out=R_out, n_z=n_z, tol=tol, max_iter=max_iter,
    )
    h1 = compute_solver_health(r1, geom, grid, P_pump, P_signal_avg)
    if h1.energy_status != "violation":
        return _attach(r1, h1, "direct", 0)

    # ── Layer 2: homotopy continuation ────────────────────────────────
    def _passes(r: SteadyResult) -> bool:
        h = compute_solver_health(r, geom, grid, P_pump, P_signal_avg)
        return h.energy_status != "violation"

    r2 = solve_steady_state_homotopy(
        geom=geom, grid=grid,
        P_pump=P_pump, P_signal_avg=P_signal_avg,
        ase_in_fwd=ase_in_fwd,
        R_in=R_in, R_out=R_out, n_z=n_z, tol=tol, max_iter=max_iter,
        health_predicate=_passes,
    )
    h2 = compute_solver_health(r2, geom, grid, P_pump, P_signal_avg)
    if h2.energy_status != "violation":
        return _attach(r2, h2, "homotopy", r2.homotopy_steps_used)

    # ── Layer 3: time-marching arbiter ────────────────────────────────
    # Convert the steady-state input to a high-rep pulsed signal that B1+B2
    # can run. At rep_rate ≫ 1/τ, the inter-pulse Δn₂ is negligible and the
    # pulse-train behaves as quasi-CW — but the time-marching solver is
    # energy-conservative by construction (cannot land on the wrong fixed
    # point regardless of the operating point).
    rep_rate = 1e6
    pulse_duration = 1.0 / (rep_rate * 10)
    pulse_energy = P_signal_avg / rep_rate

    r3 = solve_time_dependent(
        geom=geom, grid=grid,
        P_pump=P_pump, P_signal_avg=P_signal_avg,
        rep_rate=rep_rate,
        pulse_duration=pulse_duration,
        pulse_energy=pulse_energy,
        ase_in_fwd=ase_in_fwd,
        R_in=R_in, R_out=R_out, n_z=n_z,
        mode="full",
    )
    h3 = compute_solver_health(r3, geom, grid, P_pump, P_signal_avg)

    if r3.parasitic_lasing or not r3.converged:
        # B1+B2 itself bailed out — no physical steady state exists. The
        # warmup Mode A result (returned by `_periodic_steady_state` via
        # the parasitic guard) is the cleanest "system is past parasitic"
        # report we can produce. Mark `all_failed` so callers know.
        r3.notes.append(
            "Robust solver: all three layers indicate no physical steady "
            "state. System is past parasitic-lasing threshold; "
            "values reported are Mode A's runaway-clamped fallback."
        )
        return _attach(r3, h3, "all_failed", r2.homotopy_steps_used)

    return _attach(r3, h3, "time_marching_arbiter", r2.homotopy_steps_used)


def _attach(
    result: SteadyResult,
    health: SolverHealth,
    path: str,
    homotopy_steps: int,
) -> SteadyResult:
    """Stamp a `SteadyResult` with health + path metadata."""
    from dataclasses import replace as _replace
    return _replace(
        result,
        health=health,
        solver_path_used=path,
        homotopy_steps_used=homotopy_steps,
    )
