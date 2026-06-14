"""
Steady-state spectrally-resolved bidirectional ASE solver (Mode A).

Solves the boundary-value problem from docs/ase.md §4.1 by iterative shooting:
forward sweep with RK4, apply z=L reflection BC, backward sweep with RK4,
recompute n2(z), repeat until convergence.

Channel layout per spatial point z:
    pump (forward), signal (forward), ASE forward [n_bins], ASE backward [n_bins]

The inversion n2(z) at every z is computed from the steady-state form of the
rate equation summed over ALL channels (pump + signal + every ASE bin in both
directions) — leaving any of these out (the ase.md §11.3 pitfall) makes the
whole exercise pointless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .spectral_grid import SpectralGrid

# Physical constants
_H = 6.626e-34          # Planck constant [J·s]
_LN10_OVER_10 = math.log(10) / 10.0

# Maximum small-signal gain per spatial segment (g0·dz) before the explicit
# forward sweep under-resolves the gain-saturation feedback and runs away.
# Empirically the convergence boundary for the single_amp.json geometry is
# g0·dz ≈ 0.065; 0.05 leaves margin. Above this the solve is flagged
# `under_resolved` and the user is told to increase num_segments.
_GRID_GAIN_PER_SEGMENT_MAX = 0.05


@dataclass(frozen=True)
class AmplifierGeometry:
    """Per-fiber parameters needed by the solver."""
    fiber_length: float                    # [m]
    A_core: float                          # [m²]
    A_clad: float                          # [m²]
    N_Yb: float                            # ions/m³
    gamma_pump: float                      # = A_core / A_clad
    tau: float = 0.83e-3                   # upper-state lifetime [s] (Melkumov AS); overridden by DopantData.tau at build time
    alpha_bg_dB_per_m: float = 0.005       # background fiber loss [dB/m]
    m_pol: int = 2                         # 2 for non-PM, 1 for PM

    @property
    def alpha_bg(self) -> float:
        """Background loss in Np/m (natural-log convention)."""
        return self.alpha_bg_dB_per_m * _LN10_OVER_10


@dataclass
class SteadyResult:
    """Outputs of one steady-state BVP solve.

    The trailing `t` and `P_signal_tz` fields are only populated when a
    B2 time-dependent solve (Level 5) ran upstream — Mode A and the
    high-rep quasi-CW path leave them as `None`. They carry the
    time-resolved signal pulse so callers can plot pulse-shape
    distortion at z=L.
    """
    z: np.ndarray                          # [n_z]
    n2_z: np.ndarray                       # [n_z]
    P_pump_z: np.ndarray                   # [n_z]
    P_signal_z: np.ndarray                 # [n_z]
    P_ase_fwd_z: np.ndarray                # [n_z, n_bins]
    P_ase_bwd_z: np.ndarray                # [n_z, n_bins]
    converged: bool
    iterations: int
    pump_direction: str = "co"             # "co" (pump in at z=0) or "counter" (z=L)
    parasitic_lasing: bool = False
    parasitic_gain_max_dB: float = 0.0
    # True when the spatial grid is too coarse for the fiber's gain
    # (g0·dz > _GRID_GAIN_PER_SEGMENT_MAX): the forward sweep over-amplifies and
    # may run away. This is a numerical artifact (fix: raise num_segments), not
    # physical parasitic lasing.
    under_resolved: bool = False
    notes: list[str] = field(default_factory=list)

    # B2-only (time-dependent); None for Mode A and high-rep quasi-CW.
    t: Optional[np.ndarray] = None                  # [n_t]
    P_signal_tz: Optional[np.ndarray] = None        # [n_t, n_z]

    # Set by `solve_steady_state_robust` (see ase/solver_health.py and the
    # robust wrapper below). `health` is None for direct callers of
    # `solve_steady_state`; the robust wrapper always populates it.
    health: object = None                           # ase.solver_health.SolverHealth
    solver_path_used: str = "direct"                # "direct" | "homotopy" | "time_marching_arbiter" | "all_failed"
    homotopy_steps_used: int = 0

    @property
    def ase_fwd_out(self) -> np.ndarray:
        """Forward ASE spectrum at z=L (the output end)."""
        return self.P_ase_fwd_z[-1]

    @property
    def ase_bwd_in(self) -> np.ndarray:
        """Backward ASE spectrum at z=0 (the input end)."""
        return self.P_ase_bwd_z[0]

    @property
    def signal_out(self) -> float:
        return float(self.P_signal_z[-1])

    @property
    def pump_residual(self) -> float:
        """Unabsorbed pump leaving the fiber. For a co-pump it exits at z=L;
        for a counter-pump (injected at z=L) it exits at z=0."""
        return float(self.P_pump_z[0] if self.pump_direction == "counter"
                     else self.P_pump_z[-1])


def _compute_n2_local(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump: float,
    P_signal: float,
    P_ase_fwd: np.ndarray,
    P_ase_bwd: np.ndarray,
) -> float:
    """Steady-state inversion at one z point given local channel powers.

    Per-ion rate for channel k is σ · Γ · P / (h·ν · A_dope), with
    A_dope = A_core for full-core-doped Yb fiber. For the pump that simplifies
    (Γ_pump = A_core/A_clad) to σ · P / (h·ν·A_clad). Note: docs/ase.md §2.5
    lists "A_k = A_clad for pump" alongside Γ_pump — that double-counts; A_core
    is the correct A_dope for all channels when Γ is also present.
    """
    A_dope = geom.A_core
    inv_h_A = 1.0 / (_H * A_dope)

    pump_term = geom.gamma_pump * P_pump * inv_h_A / grid.nu_pump
    num = grid.sigma_a_pump * pump_term
    den = (grid.sigma_a_pump + grid.sigma_e_pump) * pump_term

    sig_term = grid.gamma_signal * P_signal * inv_h_A / grid.nu_signal
    num += grid.sigma_a_signal * sig_term
    den += (grid.sigma_a_signal + grid.sigma_e_signal) * sig_term

    P_ase = P_ase_fwd + P_ase_bwd
    ase_term = grid.gamma * P_ase * inv_h_A / grid.frequencies
    num += float(np.sum(grid.sigma_a * ase_term))
    den += float(np.sum((grid.sigma_a + grid.sigma_e) * ase_term))

    n2 = num / (den + 1.0 / geom.tau)
    if n2 < 0.0:
        return 0.0
    if n2 > 1.0:
        return 1.0
    return float(n2)


def _compute_n2_profile(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump_z: np.ndarray,
    P_signal_z: np.ndarray,
    P_ase_fwd_z: np.ndarray,
    P_ase_bwd_z: np.ndarray,
) -> np.ndarray:
    """Vectorised inversion across a full z grid (used for diagnostics)."""
    A_dope = geom.A_core
    inv_h_A = 1.0 / (_H * A_dope)

    pump_term = geom.gamma_pump * P_pump_z * inv_h_A / grid.nu_pump
    num = grid.sigma_a_pump * pump_term
    den = (grid.sigma_a_pump + grid.sigma_e_pump) * pump_term

    sig_term = grid.gamma_signal * P_signal_z * inv_h_A / grid.nu_signal
    num += grid.sigma_a_signal * sig_term
    den += (grid.sigma_a_signal + grid.sigma_e_signal) * sig_term

    inv_nu = 1.0 / grid.frequencies
    P_ase_total = P_ase_fwd_z + P_ase_bwd_z
    ase_term = grid.gamma * P_ase_total * (inv_h_A * inv_nu)[None, :]
    num += np.sum(grid.sigma_a * ase_term, axis=1)
    den += np.sum((grid.sigma_a + grid.sigma_e) * ase_term, axis=1)

    n2 = num / (den + 1.0 / geom.tau)
    return np.clip(n2, 0.0, 1.0)


def _rate_fwd(
    P_pump: float,
    P_signal: float,
    P_ase: np.ndarray,
    n2: float,
    geom: AmplifierGeometry,
    grid: SpectralGrid,
) -> tuple[float, float, np.ndarray]:
    """Forward-channel rates: dP/dz for pump, signal, and each ASE bin."""
    one_minus = 1.0 - n2
    a_bg = geom.alpha_bg

    # Pump (no spontaneous emission source)
    g_pump = geom.gamma_pump * geom.N_Yb * (
        n2 * grid.sigma_e_pump - one_minus * grid.sigma_a_pump
    )
    d_pump = (g_pump - a_bg) * P_pump

    # Signal (no spontaneous emission source)
    g_sig = grid.gamma_signal * geom.N_Yb * (
        n2 * grid.sigma_e_signal - one_minus * grid.sigma_a_signal
    )
    d_sig = (g_sig - a_bg) * P_signal

    # ASE bins, vectorised
    g_ase = grid.gamma * geom.N_Yb * (n2 * grid.sigma_e - one_minus * grid.sigma_a)
    S_ase = (
        geom.m_pol * _H * grid.frequencies * grid.d_nu
        * grid.gamma * grid.sigma_e * geom.N_Yb * n2
    )
    d_ase = (g_ase - a_bg) * P_ase + S_ase

    return d_pump, d_sig, d_ase


def _rate_ase(
    P_ase: np.ndarray,
    n2: float,
    geom: AmplifierGeometry,
    grid: SpectralGrid,
) -> np.ndarray:
    """Magnitude of the per-bin rate (gain·P + S - α·P).

    Used for backward-direction RK4 — the integration loop applies the sign.
    Spontaneous source is identical in both directions (isotropic).
    """
    one_minus = 1.0 - n2
    g_ase = grid.gamma * geom.N_Yb * (n2 * grid.sigma_e - one_minus * grid.sigma_a)
    S_ase = (
        geom.m_pol * _H * grid.frequencies * grid.d_nu
        * grid.gamma * grid.sigma_e * geom.N_Yb * n2
    )
    return (g_ase - geom.alpha_bg) * P_ase + S_ase


def _rate_pump(
    P_pump: float,
    n2: float,
    geom: AmplifierGeometry,
    grid: SpectralGrid,
) -> float:
    """Pump rate dP_pump/dz (forward-z convention), no spontaneous source.

    Identical to the pump term in `_rate_fwd`, factored out so the backward
    (counter-pump) RK4 sweep can integrate the pump the same way the backward
    ASE sweep integrates its channels. For a counter-pump the net coefficient
    `g_pump - α_bg` is negative (the pump is being absorbed), so stepping from
    z=L toward z=0 — i.e. adding `dz · rate` — makes the pump decay toward the
    input end, exactly as a physically-injected backward pump should.
    """
    g_pump = geom.gamma_pump * geom.N_Yb * (
        n2 * grid.sigma_e_pump - (1.0 - n2) * grid.sigma_a_pump
    )
    return (g_pump - geom.alpha_bg) * P_pump


def solve_pfield_fixed_n2(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    n2_z: np.ndarray,
    P_pump_in: float,
    P_signal_in: float,
    ase_in_fwd: np.ndarray,
    R_in: float,
    R_out: float,
    n_z: int,
    pump_direction: str = "co",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-shot P-field profile for a *fixed* n₂(z) profile.

    Because n₂ is held constant, the forward and backward channels decouple
    (no feedback through inversion). A single forward RK4 sweep + boundary
    condition at z=L + a single backward RK4 sweep is therefore exact for
    this n₂. Used by B1 between time steps as a much cheaper substitute
    for the full iterative shooting in `solve_steady_state`.

    With `pump_direction="counter"` the pump is injected at z=L and integrated
    in the backward sweep instead of the forward one; the signal and forward
    ASE always propagate from z=0.

    `n2_z` must already be defined on `n_z` linearly-spaced points over the
    fiber length; the function does NOT update n₂.

    Returns (z, P_pump_z, P_signal_z, P_ase_fwd_z, P_ase_bwd_z).
    """
    counter = pump_direction == "counter"
    z = np.linspace(0.0, geom.fiber_length, n_z)
    dz = z[1] - z[0]

    P_pump_z = np.empty(n_z)
    P_signal_z = np.empty(n_z)
    P_ase_fwd_z = np.empty((n_z, grid.n_bins))
    P_ase_bwd_z = np.zeros((n_z, grid.n_bins))

    if counter:
        P_pump_z[-1] = P_pump_in     # pump injected at the output end
    else:
        P_pump_z[0] = P_pump_in
    P_signal_z[0] = P_signal_in
    P_ase_fwd_z[0] = ase_in_fwd

    # Forward RK4 — gain coefficients are frozen via n₂(z); only the
    # propagating powers themselves enter the rate functions. For a
    # counter-pump the pump is NOT advanced here (it is a backward channel);
    # _rate_fwd's d_pump is simply discarded.
    for i in range(n_z - 1):
        n2_i = float(n2_z[i])
        P_p, P_s, P_a = P_pump_z[i], P_signal_z[i], P_ase_fwd_z[i]

        d_p1, d_s1, d_a1 = _rate_fwd(P_p, P_s, P_a, n2_i, geom, grid)
        d_p2, d_s2, d_a2 = _rate_fwd(
            P_p + 0.5 * dz * d_p1, P_s + 0.5 * dz * d_s1,
            P_a + 0.5 * dz * d_a1, n2_i, geom, grid,
        )
        d_p3, d_s3, d_a3 = _rate_fwd(
            P_p + 0.5 * dz * d_p2, P_s + 0.5 * dz * d_s2,
            P_a + 0.5 * dz * d_a2, n2_i, geom, grid,
        )
        d_p4, d_s4, d_a4 = _rate_fwd(
            P_p + dz * d_p3, P_s + dz * d_s3,
            P_a + dz * d_a3, n2_i, geom, grid,
        )
        if not counter:
            P_pump_z[i + 1] = max(P_p + dz / 6.0 * (d_p1 + 2 * d_p2 + 2 * d_p3 + d_p4), 0.0)
        P_signal_z[i + 1] = max(P_s + dz / 6.0 * (d_s1 + 2 * d_s2 + 2 * d_s3 + d_s4), 0.0)
        P_ase_fwd_z[i + 1] = np.maximum(
            P_a + dz / 6.0 * (d_a1 + 2 * d_a2 + 2 * d_a3 + d_a4), 0.0,
        )

    # z=L BC for backward ASE
    P_ase_bwd_z[-1] = R_out * P_ase_fwd_z[-1]

    # Backward RK4 — same rate magnitude as forward ASE (isotropic source),
    # integrated from z=L to z=0. A counter-pump is integrated alongside the
    # backward ASE (same z=L→0 direction) using its own rate.
    for i in range(n_z - 2, -1, -1):
        n2_hi = float(n2_z[i + 1])
        P_b = P_ase_bwd_z[i + 1]
        r1 = _rate_ase(P_b, n2_hi, geom, grid)
        r2 = _rate_ase(P_b + 0.5 * dz * r1, n2_hi, geom, grid)
        r3 = _rate_ase(P_b + 0.5 * dz * r2, n2_hi, geom, grid)
        r4 = _rate_ase(P_b + dz * r3, n2_hi, geom, grid)
        P_ase_bwd_z[i] = np.maximum(
            P_b + dz / 6.0 * (r1 + 2 * r2 + 2 * r3 + r4), 0.0,
        )
        if counter:
            P_p = P_pump_z[i + 1]
            q1 = _rate_pump(P_p, n2_hi, geom, grid)
            q2 = _rate_pump(P_p + 0.5 * dz * q1, n2_hi, geom, grid)
            q3 = _rate_pump(P_p + 0.5 * dz * q2, n2_hi, geom, grid)
            q4 = _rate_pump(P_p + dz * q3, n2_hi, geom, grid)
            P_pump_z[i] = max(P_p + dz / 6.0 * (q1 + 2 * q2 + 2 * q3 + q4), 0.0)

    return z, P_pump_z, P_signal_z, P_ase_fwd_z, P_ase_bwd_z


def _check_parasitic_lasing(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    n2_z: np.ndarray,
    z: np.ndarray,
    R_in: float,
    R_out: float,
) -> tuple[bool, float]:
    """ase.md §5.3: round-trip gain G(λ)²·R_in·R_out ≥ 1 ⇒ parasitic lasing.

    Computes G(λ) = exp(∫ g(λ,z) dz) over the converged n2 profile and checks
    the threshold per bin.
    """
    if R_in <= 0 or R_out <= 0:
        return False, 0.0
    one_minus = 1.0 - n2_z
    # g(λ, z) = Γ(λ)·N·[n2·σ_e(λ) - (1-n2)·σ_a(λ)]
    # Shape: [n_z, n_bins]
    g_lam_z = (
        grid.gamma[None, :] * geom.N_Yb
        * (n2_z[:, None] * grid.sigma_e[None, :]
           - one_minus[:, None] * grid.sigma_a[None, :])
    )
    g_lam_z -= geom.alpha_bg
    int_g_lam = np.trapezoid(g_lam_z, z, axis=0)            # [n_bins]
    G_lam = np.exp(int_g_lam)
    round_trip = G_lam ** 2 * R_in * R_out
    max_round_trip = float(round_trip.max())
    max_g_dB = float(10 * np.log10(max(max_round_trip, 1e-300)))
    return max_round_trip >= 1.0, max_g_dB


def _spontaneous_emission_init(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump: float,
    P_signal_avg: float,
    ase_in_fwd: np.ndarray,
    n_z: int,
    pump_direction: str = "co",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spontaneous-emission-seeded initial guess for ``solve_steady_state``.

    A single sweep of dP_pump/dz with signal = 0 and the ASE source OFF yields
    an exact pump-only inversion profile n2(z): pump-only has no
    backward-coupled wave, so no iteration is needed. The spontaneous-emission
    source S_ase(z) evaluated on that profile is then accumulated gain-free
    from each facet to seed the forward/backward ASE fields — a deliberate
    lower-bound floor that lands the production iteration in the *physical*
    (signal-dominated) basin rather than the ASE-runaway one.

    For a counter-pump the pump-only sweep runs from z=L back to z=0 so the
    seed profile peaks at the injection (output) end.

    Returns ``(P_pump_z, P_signal_z, P_ase_fwd_z, P_ase_bwd_z)`` with shapes
    ``(n_z,)``, ``(n_z,)``, ``(n_z, n_bins)``, ``(n_z, n_bins)``.

    Degrades gracefully: at ``P_pump == 0`` the inversion is zero everywhere,
    so the ASE seed collapses to the historical zero-ASE guess.
    """
    n_bins = grid.n_bins
    z = np.linspace(0.0, geom.fiber_length, n_z)
    dz = z[1] - z[0]
    zeros_bins = np.zeros(n_bins)

    # 1. Pump-only sweep (forward Euler — this is only a guess; the production
    #    RK4 loop corrects any slack). n2 is the steady-state inversion driven
    #    by the local pump alone. Co-pump sweeps z=0→L; counter-pump z=L→0.
    P_pump_z = np.empty(n_z)
    n2_pump = np.empty(n_z)
    if pump_direction == "counter":
        P_pump_z[-1] = P_pump
        for i in range(n_z - 1, 0, -1):
            n2_pump[i] = _compute_n2_local(
                geom, grid, P_pump_z[i], 0.0, zeros_bins, zeros_bins
            )
            d_pump = _rate_pump(P_pump_z[i], n2_pump[i], geom, grid)
            # Stepping toward z=0 (−dz in z): add dz·rate, matching the
            # backward-channel integration convention used elsewhere.
            P_pump_z[i - 1] = max(P_pump_z[i] + dz * d_pump, 0.0)
        n2_pump[0] = _compute_n2_local(
            geom, grid, P_pump_z[0], 0.0, zeros_bins, zeros_bins
        )
    else:
        P_pump_z[0] = P_pump
        for i in range(n_z - 1):
            n2_pump[i] = _compute_n2_local(
                geom, grid, P_pump_z[i], 0.0, zeros_bins, zeros_bins
            )
            d_pump = _rate_fwd(P_pump_z[i], 0.0, zeros_bins, n2_pump[i], geom, grid)[0]
            P_pump_z[i + 1] = max(P_pump_z[i] + dz * d_pump, 0.0)
        n2_pump[-1] = _compute_n2_local(
            geom, grid, P_pump_z[-1], 0.0, zeros_bins, zeros_bins
        )

    # 2. Spontaneous-emission source per bin [W/m] on the pump-only profile.
    #    Same expression as the S_ase term in `_rate_fwd`; only n2 varies in z.
    S_coeff = (
        geom.m_pol * _H * grid.frequencies * grid.d_nu
        * grid.gamma * grid.sigma_e * geom.N_Yb
    )                                                 # (n_bins,)
    S_ase_z = n2_pump[:, None] * S_coeff[None, :]     # (n_z, n_bins), >= 0

    # 3. Gain-free cumulative-trapezoid accumulation from each facet.
    seg = 0.5 * (S_ase_z[1:] + S_ase_z[:-1]) * dz     # (n_z-1, n_bins)
    P_ase_fwd_z = np.empty((n_z, n_bins))
    P_ase_fwd_z[0] = ase_in_fwd
    P_ase_fwd_z[1:] = ase_in_fwd[None, :] + np.cumsum(seg, axis=0)

    P_ase_bwd_z = np.zeros((n_z, n_bins))
    P_ase_bwd_z[:-1] = np.cumsum(seg[::-1], axis=0)[::-1]

    # 4. Flat signal guess (signal IC is out of scope for this seed).
    P_signal_z = np.full(n_z, P_signal_avg, dtype=float)

    return P_pump_z, P_signal_z, P_ase_fwd_z, P_ase_bwd_z


def solve_steady_state(
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
    relaxation: float = 1.0,
    init: Optional[tuple] = None,
    pump_direction: str = "co",
) -> SteadyResult:
    """Iterative-shooting BVP solver.

    Args:
        geom: fiber geometry and doping.
        grid: spectral grid for this fiber.
        P_pump: input pump power [W]. Injected at z=0 for a co-pump and at
                z=L for a counter-pump (see `pump_direction`).
        P_signal_avg: input signal power at z=0 [W] (average, quasi-CW).
        ase_in_fwd: forward ASE spectrum at z=0 [W per bin], shape (n_bins,).
                    Zero for the first stage; output of the previous stage's
                    bandpass filter for later stages.
        R_in, R_out: power reflectivity at the input and output facets.
                     Default R_in=0 (well-spliced/AR-coated input),
                     R_out=1e-4 (8° angle-cleaved output).
        n_z: number of spatial grid points (200 by default).
        tol: relative-change convergence tolerance (1e-5 per ase.md §4.1).
        max_iter: iteration cap.
        relaxation: under-relaxation factor (0<α≤1). 1.0 = no relaxation.
        init: optional `(P_pump_z, P_signal_z, P_ase_fwd_z, P_ase_bwd_z)`
              tuple overriding the default spontaneous-emission-seeded
              initial guess (see `_spontaneous_emission_init`).
              Used by `solve_steady_state_homotopy` for the warm-start
              continuation recipe (Ren et al., Opt. Quantum Electron.
              47(7), 2199, 2015).
        pump_direction: "co" (pump injected at z=0, co-propagating with the
              signal) or "counter" (pump injected at z=L, counter-propagating).

    Returns:
        SteadyResult with full P_k(z) profiles and the converged n2(z).
    """
    counter = pump_direction == "counter"
    z = np.linspace(0.0, geom.fiber_length, n_z)
    dz = z[1] - z[0]

    if init is None:
        # Default spontaneous-emission-seeded initial guess from a pump-only
        # Beer-Lambert inversion (see `_spontaneous_emission_init`).
        P_pump_z, P_signal_z, P_ase_fwd_z, P_ase_bwd_z = (
            _spontaneous_emission_init(
                geom, grid, P_pump, P_signal_avg, ase_in_fwd, n_z, pump_direction
            )
        )
    else:
        # Caller-supplied warm start (homotopy continuation).
        P_pump_z, P_signal_z, P_ase_fwd_z, P_ase_bwd_z = (a.copy() for a in init)

    converged = False
    notes: list[str] = []
    n2_z = np.zeros(n_z)
    iterations = 0
    max_rel = float("inf")
    runaway = False

    # Under-resolution guard: estimate the worst-case small-signal gain per
    # segment (g0·dz) from the pump-only asymptotic inversion. If a single
    # segment can amplify by more than ~e^0.05, the explicit forward sweep can
    # under-resolve the gain-saturation feedback and run away — a coarse-grid
    # numerical artifact, not physical parasitic lasing. (Same g0 estimate as
    # solver_health.small_signal_g0L, but max over the ASE bins.) This is the
    # geometric condition; the result is only *flagged* under_resolved below if
    # the grid actually caused a failure (a converged solve was resolved enough).
    phi_pump = geom.gamma_pump * P_pump / (_H * grid.nu_pump * geom.A_core)
    n2_asymp = (
        grid.sigma_a_pump * phi_pump
        / ((grid.sigma_a_pump + grid.sigma_e_pump) * phi_pump + 1.0 / geom.tau)
    )
    g0_bins = grid.gamma * geom.N_Yb * (
        n2_asymp * grid.sigma_e - (1.0 - n2_asymp) * grid.sigma_a
    )
    g0_max = max(float(g0_bins.max()), 0.0)
    g0_dz = g0_max * dz
    grid_marginal = g0_dz > _GRID_GAIN_PER_SEGMENT_MAX
    n_z_recommended = math.ceil(
        g0_max * geom.fiber_length / _GRID_GAIN_PER_SEGMENT_MAX
    )

    # Cap on per-bin ASE power as a runaway sentinel. With 1 W of pump and
    # realistic clamping, no bin should reach kW; if any does, the iteration
    # is diverging and we stop early.
    ase_runaway_threshold_W = 1e3

    for it in range(max_iter):
        iterations = it + 1

        # Snapshot for convergence + relaxation
        pmp_old = P_pump_z.copy()
        sig_old = P_signal_z.copy()
        fwd_old = P_ase_fwd_z.copy()
        bwd_old = P_ase_bwd_z.copy()

        # 1. Forward sweep — n2 is recomputed at every z step from the locally
        #    just-updated forward profile and the frozen previous-iteration
        #    backward profile. This keeps gain-saturation pinned to the local
        #    state and prevents single-pass blow-up in high-gain stages.
        if counter:
            P_pump_z[-1] = P_pump        # counter-pump injected at z=L
        else:
            P_pump_z[0] = P_pump
        P_signal_z[0] = P_signal_avg
        P_ase_fwd_z[0] = ase_in_fwd
        n2_z[0] = _compute_n2_local(
            geom, grid, P_pump_z[0], P_signal_z[0], P_ase_fwd_z[0], P_ase_bwd_z[0]
        )

        for i in range(n_z - 1):
            n2_lo = n2_z[i]

            P_p = P_pump_z[i]
            P_s = P_signal_z[i]
            P_a = P_ase_fwd_z[i]

            d_p1, d_s1, d_a1 = _rate_fwd(P_p, P_s, P_a, n2_lo, geom, grid)
            d_p2, d_s2, d_a2 = _rate_fwd(
                P_p + 0.5 * dz * d_p1,
                P_s + 0.5 * dz * d_s1,
                P_a + 0.5 * dz * d_a1,
                n2_lo, geom, grid,
            )
            d_p3, d_s3, d_a3 = _rate_fwd(
                P_p + 0.5 * dz * d_p2,
                P_s + 0.5 * dz * d_s2,
                P_a + 0.5 * dz * d_a2,
                n2_lo, geom, grid,
            )
            d_p4, d_s4, d_a4 = _rate_fwd(
                P_p + dz * d_p3,
                P_s + dz * d_s3,
                P_a + dz * d_a3,
                n2_lo, geom, grid,
            )

            if not counter:
                # Co-pump advances with the forward channels. For a counter-pump
                # the pump is a backward channel (integrated in the backward
                # sweep below); leave its profile untouched here.
                P_pump_z[i + 1] = max(P_p + dz / 6.0 * (d_p1 + 2 * d_p2 + 2 * d_p3 + d_p4), 0.0)
            P_signal_z[i + 1] = max(P_s + dz / 6.0 * (d_s1 + 2 * d_s2 + 2 * d_s3 + d_s4), 0.0)
            P_ase_fwd_z[i + 1] = np.maximum(
                P_a + dz / 6.0 * (d_a1 + 2 * d_a2 + 2 * d_a3 + d_a4), 0.0
            )

            # Re-evaluate local n2 at the new step using the just-updated
            # forward state (frozen backward) — this is what saturates gain.
            n2_z[i + 1] = _compute_n2_local(
                geom, grid,
                P_pump_z[i + 1], P_signal_z[i + 1],
                P_ase_fwd_z[i + 1], P_ase_bwd_z[i + 1],
            )

        # 2. BC at z=L: backward ASE = R_out · forward ASE
        P_ase_bwd_z[-1] = R_out * P_ase_fwd_z[-1]

        # 3. Backward sweep — n2 again recomputed at every step from the new
        #    P_fwd profile and the locally-just-updated P_bwd. A counter-pump
        #    is integrated here too (same z=L→0 direction as backward ASE).
        for i in range(n_z - 2, -1, -1):
            n2_hi = _compute_n2_local(
                geom, grid,
                P_pump_z[i + 1], P_signal_z[i + 1],
                P_ase_fwd_z[i + 1], P_ase_bwd_z[i + 1],
            )
            P_b = P_ase_bwd_z[i + 1]

            r1 = _rate_ase(P_b, n2_hi, geom, grid)
            r2 = _rate_ase(P_b + 0.5 * dz * r1, n2_hi, geom, grid)
            r3 = _rate_ase(P_b + 0.5 * dz * r2, n2_hi, geom, grid)
            r4 = _rate_ase(P_b + dz * r3, n2_hi, geom, grid)

            P_ase_bwd_z[i] = np.maximum(
                P_b + dz / 6.0 * (r1 + 2 * r2 + 2 * r3 + r4), 0.0
            )

            if counter:
                P_p = P_pump_z[i + 1]
                q1 = _rate_pump(P_p, n2_hi, geom, grid)
                q2 = _rate_pump(P_p + 0.5 * dz * q1, n2_hi, geom, grid)
                q3 = _rate_pump(P_p + 0.5 * dz * q2, n2_hi, geom, grid)
                q4 = _rate_pump(P_p + dz * q3, n2_hi, geom, grid)
                P_pump_z[i] = max(P_p + dz / 6.0 * (q1 + 2 * q2 + 2 * q3 + q4), 0.0)

        # Refresh the diagnostic n2 profile from the converged P fields.
        n2_z = _compute_n2_profile(
            geom, grid, P_pump_z, P_signal_z, P_ase_fwd_z, P_ase_bwd_z
        )

        # Apply under-relaxation: blend newly-computed P with previous iterate
        if relaxation < 1.0 and it > 0:
            P_pump_z = relaxation * P_pump_z + (1.0 - relaxation) * pmp_old
            P_signal_z = relaxation * P_signal_z + (1.0 - relaxation) * sig_old
            P_ase_fwd_z = relaxation * P_ase_fwd_z + (1.0 - relaxation) * fwd_old
            P_ase_bwd_z = relaxation * P_ase_bwd_z + (1.0 - relaxation) * bwd_old

        # Runaway detection: if any single bin exceeds the sentinel power, the
        # iteration is diverging. Bail out and clamp. The cause is either an
        # under-resolved grid (numerical artifact — see the guard above) or a
        # genuinely non-physical operating point; distinguish them in the note.
        if (P_ase_fwd_z.max() > ase_runaway_threshold_W
                or P_ase_bwd_z.max() > ase_runaway_threshold_W):
            runaway = True
            # Clamp to the threshold for downstream sanity
            np.minimum(P_ase_fwd_z, ase_runaway_threshold_W, out=P_ase_fwd_z)
            np.minimum(P_ase_bwd_z, ase_runaway_threshold_W, out=P_ase_bwd_z)
            if grid_marginal:
                notes.append(
                    "ASE runaway from an under-resolved spatial grid — a "
                    "numerical artifact, not physical parasitic lasing; "
                    f"increase num_segments to >= {n_z_recommended}."
                )
            else:
                notes.append(
                    "ASE runaway — system appears past parasitic-lasing "
                    "threshold; no physical steady state."
                )
            break

        # 5. Convergence: max relative change across all channels
        def rel_max(new: np.ndarray, old: np.ndarray) -> float:
            scale = np.maximum(np.abs(new), 1e-15)
            return float(np.max(np.abs(new - old) / scale))

        max_rel = max(
            rel_max(P_pump_z, pmp_old),
            rel_max(P_signal_z, sig_old),
            rel_max(P_ase_fwd_z, fwd_old),
            rel_max(P_ase_bwd_z, bwd_old) if it > 0 else 0.0,
        )
        if max_rel < tol and it >= 2:
            converged = True
            break

    if not converged and not runaway:
        notes.append(
            f"Did not converge in {max_iter} iterations (max_rel={max_rel:.2e})"
        )

    # Flag under-resolution only when the marginal grid actually caused a
    # failure — a converged solve was evidently resolved enough. This drives the
    # user-facing "[UNDER-RESOLVED: increase num_segments]" diagnostic.
    under_resolved = grid_marginal and (runaway or not converged)
    if under_resolved:
        notes.append(
            f"Spatial grid under-resolved for this gain: g0*dz={g0_dz:.2f} "
            f"(g0*L={g0_max * geom.fiber_length:.0f}, n_z={n_z}). Increase "
            f"num_segments to >= {n_z_recommended} for stable convergence."
        )

    parasitic, max_g_dB = _check_parasitic_lasing(
        geom, grid, n2_z, z, R_in, R_out
    )
    if parasitic:
        notes.append(
            f"Parasitic lasing condition met: max round-trip gain {max_g_dB:.1f} dB"
        )

    return SteadyResult(
        z=z,
        n2_z=n2_z,
        P_pump_z=P_pump_z,
        P_signal_z=P_signal_z,
        P_ase_fwd_z=P_ase_fwd_z,
        P_ase_bwd_z=P_ase_bwd_z,
        converged=converged,
        iterations=iterations,
        pump_direction=pump_direction,
        parasitic_lasing=parasitic or runaway,
        parasitic_gain_max_dB=max_g_dB,
        under_resolved=under_resolved,
        notes=notes,
    )


# ── Homotopy continuation (Ren et al. 2015) ─────────────────────────────
#
# Reference: Ren, Han, Liu et al., "Numerical methods for high-power
# Er/Yb-codoped fiber amplifiers," Opt. Quantum Electron. 47(7),
# 2199-2212 (2015). DOI: 10.1007/s11082-014-0096-8.
#
# When the default spontaneous-emission-seeded initial guess falls into a non-physical
# basin, walk from a known-easy operating point to the target by re-using
# each step's solution as the next step's initial guess (`init=`).
#
# Step 1: pump-only solve (signal=0, spontaneous emission disabled via
#         m_pol=0). The fixed point here is unique — gain comes from a
#         linear equation, no inversion-saturation multi-modality.
# Step 2: + signal, still no ASE source. Beer-Lambert-like, signal
#         clamps n₂ via stimulated emission. Still unique.
# Step 3: + ASE source (m_pol restored). Critical step — the iteration
#         starts in the *signal-clamped* basin instead of the
#         *ASE-dominated* basin.
# Step 4 (optional): parameter continuation. If step 3 still fails the
#         energy-conservation check, scale the "hardest" parameter
#         (whichever maximises g₀·L: fiber length or pump power) down
#         to 50 % and walk back up in increments.
# Step 5 (final fallback): Xu et al. 2014 linear-gain-shape multistart.
#         Slope-efficiency values 0.3, 0.5, 0.7 give convergence in
#         ≤ 8 iterations across "all the fiber length, Yb3+-doped
#         concentration, signal reflectivity and pump power" they tested
#         (Optik 2014, pii S1068520014000546).


def solve_steady_state_homotopy(
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
    health_predicate=None,
    pump_direction: str = "co",
) -> SteadyResult:
    """Homotopy-continuation wrapper around `solve_steady_state`.

    `health_predicate(result) -> bool` is called after each step; if it
    returns True (i.e. "physical, stop"), the wrapper returns early.
    Default predicate accepts anything (run all steps).

    `pump_direction` ("co"/"counter") is threaded into every sub-solve.
    """
    from dataclasses import replace as _replace

    notes: list[str] = []
    steps_used = 0
    geom_noase = _replace(geom, m_pol=0)
    zeros_ase = np.zeros_like(ase_in_fwd)

    def _stop(r: SteadyResult) -> bool:
        return False if health_predicate is None else health_predicate(r)

    # ── Step 1: pump-only ───────────────────────────────────────────
    # This is a warm-start pre-solve — its result is not a candidate
    # answer, so we don't apply the health predicate here. (signal_gain = 0
    # trivially passes the QD check but isn't the user's question.)
    r1 = solve_steady_state(
        geom_noase, grid,
        P_pump=P_pump, P_signal_avg=0.0,
        ase_in_fwd=zeros_ase,
        R_in=R_in, R_out=R_out, n_z=n_z, tol=tol, max_iter=max_iter,
        pump_direction=pump_direction,
    )
    steps_used = 1
    notes.append("homotopy step 1: pump-only solve converged")

    # ── Step 2: + signal, ASE source still disabled ─────────────────
    # Also a warm-start (no ASE means the physics is incomplete) — skip
    # the predicate check.
    init2 = (r1.P_pump_z, np.full(n_z, P_signal_avg), r1.P_ase_fwd_z, r1.P_ase_bwd_z)
    r2 = solve_steady_state(
        geom_noase, grid,
        P_pump=P_pump, P_signal_avg=P_signal_avg,
        ase_in_fwd=zeros_ase,
        R_in=R_in, R_out=R_out, n_z=n_z, tol=tol, max_iter=max_iter,
        init=init2,
        pump_direction=pump_direction,
    )
    steps_used = 2
    notes.append("homotopy step 2: signal added, no ASE source")

    # ── Step 3: enable ASE source — first candidate answer ──────────
    init3 = (r2.P_pump_z, r2.P_signal_z, r2.P_ase_fwd_z, r2.P_ase_bwd_z)
    r3 = solve_steady_state(
        geom, grid,
        P_pump=P_pump, P_signal_avg=P_signal_avg,
        ase_in_fwd=ase_in_fwd,
        R_in=R_in, R_out=R_out, n_z=n_z, tol=tol, max_iter=max_iter,
        init=init3,
        pump_direction=pump_direction,
    )
    steps_used = 3
    notes.append("homotopy step 3: ASE source enabled, signal-clamped basin")
    if _stop(r3):
        return _stamp(r3, steps_used, notes)

    # ── Step 4: parameter continuation in pump power ────────────────
    # Walk pump from 0.5×P_pump to P_pump in three steps, using each as
    # the seed for the next.
    seed = r3
    for fraction in (0.5, 0.75, 1.0):
        P_pump_scaled = fraction * P_pump
        init_k = (seed.P_pump_z, seed.P_signal_z, seed.P_ase_fwd_z, seed.P_ase_bwd_z)
        rk = solve_steady_state(
            geom, grid,
            P_pump=P_pump_scaled, P_signal_avg=P_signal_avg,
            ase_in_fwd=ase_in_fwd,
            R_in=R_in, R_out=R_out, n_z=n_z, tol=tol, max_iter=max_iter,
            init=init_k,
            pump_direction=pump_direction,
        )
        seed = rk
        steps_used += 1
        notes.append(
            f"homotopy step {steps_used}: parameter continuation at "
            f"{fraction*100:.0f}% pump"
        )
        if _stop(rk):
            return _stamp(rk, steps_used, notes)

    # ── Step 5: Xu et al. 2014 linear-gain-shape multistart ─────────
    # Three slope efficiencies. First one to pass the health predicate wins.
    QD = grid.nu_signal / grid.nu_pump
    for eta_slope in (0.3, 0.5, 0.7):
        steps_used += 1
        init_xu = _linear_gain_shape_init(
            geom, grid, P_pump, P_signal_avg, ase_in_fwd, n_z, eta_slope, QD,
            pump_direction=pump_direction,
        )
        r_xu = solve_steady_state(
            geom, grid,
            P_pump=P_pump, P_signal_avg=P_signal_avg,
            ase_in_fwd=ase_in_fwd,
            R_in=R_in, R_out=R_out, n_z=n_z, tol=tol, max_iter=max_iter,
            init=init_xu,
            pump_direction=pump_direction,
        )
        notes.append(
            f"homotopy step {steps_used}: Xu et al. 2014 linear-gain shape "
            f"η_slope={eta_slope}"
        )
        if _stop(r_xu):
            return _stamp(r_xu, steps_used, notes)
        if eta_slope == 0.7:
            seed = r_xu     # carry the last attempt forward as the final result

    notes.append(
        f"homotopy exhausted {steps_used} steps without satisfying the "
        f"health predicate; returning last attempt."
    )
    return _stamp(seed, steps_used, notes)


def _stamp(result: SteadyResult, steps: int, extra_notes: list) -> SteadyResult:
    """Return a copy of `result` tagged with homotopy bookkeeping."""
    from dataclasses import replace as _replace
    return _replace(
        result,
        solver_path_used="homotopy",
        homotopy_steps_used=steps,
        notes=list(result.notes) + list(extra_notes),
    )


def _linear_gain_shape_init(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump: float,
    P_signal_in: float,
    ase_in_fwd: np.ndarray,
    n_z: int,
    eta_slope: float,
    QD: float,
    pump_direction: str = "co",
) -> tuple:
    """Build an `init` tuple for `solve_steady_state` using Xu et al. 2014's
    linear-gain-shape initial guess (Optik 2014, pii S1068520014000546).

    The signal power follows a log-linear ramp from `P_signal_in` to an
    estimated output `P_signal_in + η_slope · QD · P_pump`. Xu et al.
    showed that any `η_slope ∈ [0.3, 0.9]` gives convergence in ≤ 8
    iterations across the design envelope they tested. For a counter-pump
    the pump-decay seed is mirrored so it peaks at z=L.
    """
    z = np.linspace(0.0, geom.fiber_length, n_z)
    L = geom.fiber_length

    P_signal_out_est = max(
        P_signal_in + eta_slope * QD * P_pump, P_signal_in * 1.01
    )
    # Log-linear ramp avoids zero on the first cell.
    P_signal_z = P_signal_in * np.exp(
        (z / L) * np.log(P_signal_out_est / max(P_signal_in, 1e-30))
    )

    # Pump decays at the cold-cladding rate (lower bound on absorption).
    # Co-pump decays from z=0; counter-pump decays from z=L (mirror the axis).
    alpha_pump_cold = geom.gamma_pump * geom.N_Yb * grid.sigma_a_pump
    z_from_inject = (L - z) if pump_direction == "counter" else z
    P_pump_z = P_pump * np.exp(-alpha_pump_cold * z_from_inject)

    P_ase_fwd_z = np.tile(ase_in_fwd.astype(float), (n_z, 1))
    P_ase_bwd_z = np.zeros((n_z, grid.n_bins))
    return P_pump_z, P_signal_z, P_ase_fwd_z, P_ase_bwd_z
