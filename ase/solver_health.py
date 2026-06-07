"""
Post-hoc diagnostics for steady-state BVP results.

The Giles-Desurvire iterative-shooting BVP in `solver_steady.py` enforces
*per-ion photon balance* locally at every z (via the rate equation
`n2 = R_abs / (R_abs + R_em + 1/τ)`) but does **not** by itself guarantee
*global energy conservation*. Pump photons (h·ν_pump) carry more energy
than the signal/ASE photons (h·ν_signal) they spawn — the difference is
phonons (quantum-defect heat). When channel coupling is mild, the
iteration flows to a unique physical fixed point and the energy budget
balances automatically. Near the parasitic-lasing edge, the same BVP
admits multiple self-consistent fixed points, and the shooter can lock
onto one where every local equation is satisfied to tolerance while the
global output exceeds the QD ceiling by factors of 2–10×.

This module computes three diagnostics that classify a converged
`SteadyResult` into amplifier health and operating-regime categories.
The thresholds come from published amplifier-modelling practice, not
from invention:

  - **Energy-conservation residual** (Ren, Han, Liu et al., *Opt. Quantum
    Electron.* 47(7), 2199–2212, 2015, DOI 10.1007/s11082-014-0096-8;
    Paschotta RP Photonics tutorials).
  - **ASE conversion fraction η_ASE** (Wang & Clarkson, *Opt. Lett.* 31,
    3116 (2006) document the SFS regime physically; Dong, *Front. Phys.*
    13, 1539099 (2025) gives operational thresholds).
  - **Small-signal `g₀·L`** (Furuse et al., PubMed 23736565, Yb:YAG
    thin-disk: ASE transitions from spontaneous to inversion-draining
    around g(0)·l_ASE ≈ 3 — the same rule of thumb is used for fibre).

The 1 %/5 % thresholds on the energy residual are engineering judgement
(no published canonical value, see `docs/ase.md` Part II §13b):
1 % is ~10× the RK4 + iteration noise floor (~10⁻³),
5 % is comfortably below the smallest documented BVP-multi-root
violation (Paschotta tutorials note ~2× as the smallest practical
violation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver_steady import AmplifierGeometry, SteadyResult
from .spectral_grid import SpectralGrid


_H = 6.626e-34


# Published thresholds, with citations in docstrings of the consumers.
_ENERGY_WARNING_THRESHOLD = 0.01
_ENERGY_VIOLATION_THRESHOLD = 0.05
_ETA_ASE_MIXED_THRESHOLD = 0.10
_ETA_ASE_SFS_THRESHOLD = 0.30
# Furuse et al. (PubMed 23736565) calibrated g(0)·l ≈ 3 on Yb:YAG
# thin-disk geometry. Real Yb fibre amplifiers operate routinely at
# g₀·L ∈ [10, 30] because the doped core is much smaller than the
# pumped volume (Γ_signal mismatch). We flag *very* high small-signal
# gain — `g₀·L > 30` — as a "high-gain design" diagnostic: not
# necessarily wrong, just worth being aware of.
_G0L_HIGH_GAIN_THRESHOLD = 30.0


@dataclass(frozen=True)
class SolverHealth:
    """Three orthogonal diagnostics for a converged steady-state result.

    All three are derived purely from the result fields — no extra solve
    is required. Cheap to compute (~10 µs for a 200×160 grid).
    """

    # Δ / (QD · pump_absorbed), where Δ = (signal_gain + ASE_out + bg_loss)
    # − QD · pump_absorbed. ≤ 0 means consumed < delivered (perfectly fine,
    # the excess goes to numerical noise / bg_loss already counted). > 0
    # means consumed > delivered — physically impossible past noise floor.
    energy_residual_ratio: float

    # ASE_total / pump_absorbed. Indicates how much of the absorbed pump
    # ends up as broadband ASE rather than coherent signal.
    ase_conversion_fraction: float

    # Small-signal gain coefficient × fibre length, evaluated at the
    # pump-only-equilibrium inversion. Tells you whether the unsaturated
    # gain is so large that ASE will dominate even with a strong seed.
    small_signal_g0L: float

    # Classification — see thresholds in this module's body.
    energy_status: str   # "ok" | "warning" | "violation"
    regime: str          # "amplifier" | "mixed" | "sfs" | "high_gain"

    @property
    def healthy(self) -> bool:
        """True iff the converged BVP is energy-conservative and the
        operating point is the amplifier regime — what most callers want."""
        return self.energy_status == "ok" and self.regime in ("amplifier", "mixed")


def compute_solver_health(
    result: SteadyResult,
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump: float,
    P_signal_in: float,
) -> SolverHealth:
    """Run the three diagnostics on a converged `SteadyResult`.

    Args:
        result: output of `solve_steady_state` (or its wrappers).
        geom: same `AmplifierGeometry` the solve used.
        grid: same `SpectralGrid` the solve used.
        P_pump: input pump power at z=0 [W].
        P_signal_in: input signal power at z=0 [W].

    Returns a `SolverHealth` snapshot.
    """
    energy_residual_ratio = _energy_residual_ratio(
        result, geom, grid, P_pump, P_signal_in,
    )
    ase_conversion_fraction = _ase_conversion_fraction(
        result, P_pump,
    )
    small_signal_g0L = _small_signal_g0L(
        geom, grid, P_pump,
    )

    if energy_residual_ratio > _ENERGY_VIOLATION_THRESHOLD:
        energy_status = "violation"
    elif energy_residual_ratio > _ENERGY_WARNING_THRESHOLD:
        energy_status = "warning"
    else:
        energy_status = "ok"

    if ase_conversion_fraction > _ETA_ASE_SFS_THRESHOLD:
        regime = "sfs"
    elif ase_conversion_fraction > _ETA_ASE_MIXED_THRESHOLD:
        regime = "mixed"
    elif small_signal_g0L > _G0L_HIGH_GAIN_THRESHOLD:
        regime = "high_gain"
    else:
        regime = "amplifier"

    return SolverHealth(
        energy_residual_ratio=energy_residual_ratio,
        ase_conversion_fraction=ase_conversion_fraction,
        small_signal_g0L=small_signal_g0L,
        energy_status=energy_status,
        regime=regime,
    )


# ── Individual diagnostics ───────────────────────────────────────────


def _energy_residual_ratio(
    result: SteadyResult,
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump: float,
    P_signal_in: float,
) -> float:
    """Global energy-conservation residual.

    Steady-state photon-balance derivation (Ren et al. 2015, §2;
    Paschotta tutorials): the pump deposits `QD · pump_absorbed` of
    optical energy into the gain medium, where QD = ν_signal / ν_pump.
    That energy is taken up by signal amplification, by ASE in both
    directions, and by background-loss heat as light propagates:

        delivered = QD · (P_pump_in − P_pump_out)
        consumed  = signal_gain + ASE_out_total + bg_loss
        residual  = consumed − delivered

    A physical solution gives `|residual| / delivered` near zero (limited
    by the RK4 + iteration noise floor ≈ 10⁻³). Solutions trapped in a
    non-physical fixed point can violate this by factors of 2–10×.
    """
    QD = grid.nu_signal / grid.nu_pump
    pump_absorbed = max(P_pump - result.pump_residual, 0.0)
    if pump_absorbed <= 0.0:
        # Pump didn't deliver anything; any signal gain is unphysical.
        signal_gain = max(result.signal_out - P_signal_in, 0.0)
        if signal_gain > 0:
            return float("inf")
        return 0.0

    signal_gain = result.signal_out - P_signal_in
    ase_fwd_out = float(result.ase_fwd_out.sum())
    ase_bwd_out = float(result.ase_bwd_in.sum())   # exits at z=0
    bg_loss = _background_loss(result, geom)

    delivered = QD * pump_absorbed
    consumed = signal_gain + ase_fwd_out + ase_bwd_out + bg_loss
    return (consumed - delivered) / delivered


def _background_loss(result: SteadyResult, geom: AmplifierGeometry) -> float:
    """Integrated optical power lost to fibre background absorption [W].

    `α_bg` (in Np/m) acts on every channel: `dP_k/dz` has a `-α_bg · P_k`
    term. The total optical power removed per unit length is therefore
    `α_bg · Σ P_k`, integrated over z. This is irreversible heat — it
    doesn't appear in any output power but must be accounted for in the
    energy balance.
    """
    alpha = geom.alpha_bg
    if alpha == 0.0:
        return 0.0
    P_pump_z = result.P_pump_z
    P_signal_z = result.P_signal_z
    # ASE arrays are [n_z, n_bins] — sum bins first.
    P_ase_fwd_z = result.P_ase_fwd_z.sum(axis=1)
    P_ase_bwd_z = result.P_ase_bwd_z.sum(axis=1)
    total_z = P_pump_z + P_signal_z + P_ase_fwd_z + P_ase_bwd_z
    return alpha * float(np.trapezoid(total_z, result.z))


def _ase_conversion_fraction(result: SteadyResult, P_pump: float) -> float:
    """`η_ASE = (forward + backward ASE out) / pump_absorbed`.

    Wang & Clarkson, Opt. Lett. 31, 3116 (2006) demonstrated a 110 W
    Yb-fibre superfluorescent source operating at 68 % slope efficiency
    where η_ASE → 1. Dong, Front. Phys. 13, 1539099 (2025) uses η_ASE-
    based thresholds operationally for pulsed amplifiers. We classify
    > 0.3 as the SFS regime.
    """
    pump_absorbed = max(P_pump - result.pump_residual, 0.0)
    if pump_absorbed <= 0.0:
        return 0.0
    ase_total = float(result.ase_fwd_out.sum()) + float(result.ase_bwd_in.sum())
    return ase_total / pump_absorbed


def _small_signal_g0L(
    geom: AmplifierGeometry,
    grid: SpectralGrid,
    P_pump: float,
) -> float:
    """Small-signal `g₀·L` evaluated at the pump-only inversion asymptote.

    The pump-only steady-state inversion (no signal, no ASE) satisfies
    `(1-n2)·R_abs_pump = n2·(R_em_pump + 1/τ)`. The resulting `n2_asymp`
    is the highest inversion the medium can reach at the given pump
    power. With that inversion, the small-signal gain at the signal
    wavelength is `g₀ = Γ_signal · N · (n2·σ_e_signal − (1−n2)·σ_a_signal)`,
    and `g₀·L` is the unsaturated logarithmic gain over the fibre length.

    Furuse et al. (PubMed 23736565) found in Yb:YAG thin-disk experiments
    that ASE transitions from spontaneous emission to dominant inversion-
    draining around `g(0)·l_ASE ≈ 3`. Fibre amplifiers operate routinely
    at `g₀·L ∈ [10, 30]` because the doped core covers only a fraction of
    the mode (Γ_signal mismatch); we flag the design as "high_gain" past
    `g₀·L = 30`.
    """
    A_dope = geom.A_core
    inv_h_A = 1.0 / (_H * A_dope)
    pump_term = geom.gamma_pump * P_pump * inv_h_A / grid.nu_pump
    R_abs_pump = grid.sigma_a_pump * pump_term
    R_em_pump = grid.sigma_e_pump * pump_term

    denom = R_abs_pump + R_em_pump + 1.0 / geom.tau
    if denom <= 0.0:
        return 0.0
    n2_asymp = R_abs_pump / denom

    g0 = grid.gamma_signal * geom.N_Yb * (
        n2_asymp * grid.sigma_e_signal
        - (1.0 - n2_asymp) * grid.sigma_a_signal
    )
    return g0 * geom.fiber_length


__all__ = [
    "SolverHealth",
    "compute_solver_health",
]
