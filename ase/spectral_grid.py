"""
Spectral grid for ASE bins.

Holds wavelength bins, frequencies, bin widths in Hz, cross-section arrays,
overlap factors and effective areas — all wavelength-dependent and all
pre-computed once for a given fiber geometry. A grid is created per amplifier
because Γ(λ) and A_eff(λ) depend on the core radius and NA of that fiber.

The signal at 1064 nm is treated as a separate channel (its parameters are
exposed via SpectralGrid.signal_*) and is NOT one of the ASE bins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .dopants import DopantData, get_dopant

# Physical constants
_C = 3e8                  # speed of light [m/s]
_LAM_PUMP = 976e-9        # pump wavelength [m]
_LAM_SIGNAL = 1064e-9     # signal wavelength [m]


def _marcuse_w_over_a(V: float) -> float:
    """Mode radius / core radius from the Marcuse approximation.

    Valid in the single-mode regime V < 2.405; for V > 2.405 we return the
    fundamental-mode value 0.65, which is what the existing Amplifier class
    already uses (components.py:176-180).
    """
    if V < 2.405:
        return 0.65 + 1.619 / V ** 1.5 + 2.879 / V ** 6
    return 0.65


@dataclass(frozen=True)
class SpectralGrid:
    """All wavelength-dependent quantities needed by the BVP solver."""

    lambda_min: float                  # [m]
    lambda_max: float                  # [m]
    d_lambda: float                    # [m]
    r_core: float                      # [m]
    NA: float

    # Derived (filled by from_fiber)
    wavelengths: np.ndarray = field(default_factory=lambda: np.empty(0))
    frequencies: np.ndarray = field(default_factory=lambda: np.empty(0))
    d_nu: np.ndarray = field(default_factory=lambda: np.empty(0))
    sigma_a: np.ndarray = field(default_factory=lambda: np.empty(0))
    sigma_e: np.ndarray = field(default_factory=lambda: np.empty(0))
    gamma: np.ndarray = field(default_factory=lambda: np.empty(0))
    A_eff: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Signal channel scalars (1064 nm)
    sigma_a_signal: float = 0.0
    sigma_e_signal: float = 0.0
    gamma_signal: float = 0.0
    A_eff_signal: float = 0.0
    nu_signal: float = _C / _LAM_SIGNAL

    # Pump channel scalars (976 nm by default) — pump is cladding-pumped, A_pump = A_clad
    sigma_a_pump: float = 0.0
    sigma_e_pump: float = 0.0
    nu_pump: float = _C / _LAM_PUMP

    # Dopant + wavelength configuration (kept for diagnostics and to thread τ through)
    dopant: Optional[DopantData] = None
    pump_wavelength: float = _LAM_PUMP
    signal_wavelength: float = _LAM_SIGNAL

    @property
    def n_bins(self) -> int:
        return self.wavelengths.size

    @classmethod
    def from_fiber(
        cls,
        r_core: float,
        NA: float,
        lambda_min: float = 970e-9,
        lambda_max: float = 1130e-9,
        d_lambda: float = 1e-9,
        *,
        dopant: Optional[DopantData] = None,
        pump_wavelength: float = _LAM_PUMP,
        signal_wavelength: float = _LAM_SIGNAL,
    ) -> "SpectralGrid":
        """Build a grid for a fiber with the given core radius and NA.

        The optional `dopant` parameter (defaults to the registry's "Yb")
        sources the cross-section spectra and ties this grid to a τ value.
        `pump_wavelength` and `signal_wavelength` parameterise the two
        special channels — change them for non-1064 nm signals or non-976 nm
        pumps without changing the dopant.

        NOTE on 915 nm pumping: the measured Melkumov AS dataset now covers
        848-1180 nm, so the pump/signal scalar channels (sigma_a_at /
        sigma_e_at) resolve a 915 nm pump correctly. The *ASE bin grid*,
        however, still defaults to [970, 1130] nm — deliberately, because
        changing lambda_min changes n_bins and therefore every ASE result. To
        fully resolve the ASE band for a 915-nm-pumped config, lambda_min should
        drop to ~900-950 nm; that is left to the caller as an explicit choice.
        """
        if dopant is None:
            dopant = get_dopant("Yb")

        # Bin centers: λ_min + (i + 0.5)·Δλ (ase.md §3.1)
        n_bins = int(round((lambda_max - lambda_min) / d_lambda))
        i = np.arange(n_bins)
        wavelengths = lambda_min + (i + 0.5) * d_lambda
        frequencies = _C / wavelengths
        d_nu = _C * d_lambda / wavelengths ** 2

        # ASE-bin cross-sections interpolated from the dopant
        sigma_a, sigma_e = dopant.interpolate_to(wavelengths)

        # Per-bin overlap and effective area via Marcuse with V(λ)
        V_per_bin = (2 * math.pi / wavelengths) * r_core * NA
        w_per_bin = np.array([_marcuse_w_over_a(v) * r_core for v in V_per_bin])
        gamma = 1.0 - np.exp(-2.0 * (r_core / w_per_bin) ** 2)
        A_eff = math.pi * w_per_bin ** 2

        # Signal channel — same Marcuse formula at the chosen signal wavelength
        V_sig = (2 * math.pi / signal_wavelength) * r_core * NA
        w_sig = _marcuse_w_over_a(V_sig) * r_core
        gamma_signal = 1.0 - math.exp(-2.0 * (r_core / w_sig) ** 2)
        A_eff_signal = math.pi * w_sig ** 2
        sa_sig = dopant.sigma_a_at(signal_wavelength)
        se_sig = dopant.sigma_e_at(signal_wavelength)

        # Pump channel — cladding-pumped, σ at the pump wavelength.
        sa_pmp = dopant.sigma_a_at(pump_wavelength)
        se_pmp = dopant.sigma_e_at(pump_wavelength)

        return cls(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            d_lambda=d_lambda,
            r_core=r_core,
            NA=NA,
            wavelengths=wavelengths,
            frequencies=frequencies,
            d_nu=d_nu,
            sigma_a=sigma_a,
            sigma_e=sigma_e,
            gamma=gamma,
            A_eff=A_eff,
            sigma_a_signal=sa_sig,
            sigma_e_signal=se_sig,
            gamma_signal=gamma_signal,
            A_eff_signal=A_eff_signal,
            nu_signal=_C / signal_wavelength,
            sigma_a_pump=sa_pmp,
            sigma_e_pump=se_pmp,
            nu_pump=_C / pump_wavelength,
            dopant=dopant,
            pump_wavelength=pump_wavelength,
            signal_wavelength=signal_wavelength,
        )

    def refine(self, d_lambda: float) -> "SpectralGrid":
        """Return a new grid at a finer Δλ — used for the §3.3 convergence check."""
        return SpectralGrid.from_fiber(
            r_core=self.r_core,
            NA=self.NA,
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max,
            d_lambda=d_lambda,
            dopant=self.dopant,
            pump_wavelength=self.pump_wavelength,
            signal_wavelength=self.signal_wavelength,
        )
