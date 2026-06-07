"""
AseState and OpticalState: the propagation containers.

OpticalState is what flows through every component's propagate() method.
Signal carries the coherent pulse; AseState carries the spectrally-resolved
incoherent noise. A component that doesn't care about ASE just leaves
state.ase untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .spectral_grid import SpectralGrid


@dataclass(frozen=True)
class AseState:
    """Spectrally-resolved bidirectional ASE state.

    fwd_spectrum is the forward-going ASE power per bin at the *output* end of
    whichever fiber produced this state (or at the *input* of a passive
    component). bwd_spectrum is the backward-going ASE power per bin at the
    *input* end of the producing stage — kept as a diagnostic and for
    parasitic-lasing checks; isolators block it from reaching upstream.
    """

    spectral_grid: SpectralGrid
    fwd_spectrum: np.ndarray            # shape (n_bins,), [W per bin]
    bwd_spectrum: np.ndarray            # shape (n_bins,), [W per bin]
    n2_profile: Optional[np.ndarray] = None   # diagnostic; from producing stage
    z_grid: Optional[np.ndarray] = None       # diagnostic; from producing stage
    parasitic_lasing: bool = False
    # True when the producing stage's spatial grid was too coarse for its gain
    # (a numerical artifact distinct from physical parasitic lasing — fix by
    # increasing num_segments). Takes display priority over parasitic_lasing.
    under_resolved: bool = False

    @classmethod
    def zero(cls, grid: SpectralGrid) -> "AseState":
        z = np.zeros(grid.n_bins)
        return cls(spectral_grid=grid, fwd_spectrum=z.copy(), bwd_spectrum=z.copy())

    def total_fwd(self) -> float:
        return float(self.fwd_spectrum.sum())

    def total_bwd(self) -> float:
        return float(self.bwd_spectrum.sum())

    def peak_fwd_wavelength(self) -> float:
        """Wavelength [m] of the bin with the largest forward ASE."""
        if self.fwd_spectrum.sum() == 0:
            return float("nan")
        return float(self.spectral_grid.wavelengths[int(np.argmax(self.fwd_spectrum))])

    def with_flat_loss(self, loss_dB: float) -> "AseState":
        """Apply uniform attenuation to both spectra."""
        factor = 10 ** (-loss_dB / 10)
        return replace(
            self,
            fwd_spectrum=self.fwd_spectrum * factor,
            bwd_spectrum=self.bwd_spectrum * factor,
        )

    def with_spectral_transfer(
        self,
        T_fwd: np.ndarray,
        T_bwd: Optional[np.ndarray] = None,
    ) -> "AseState":
        """Multiply forward (and optionally backward) spectrum by a per-bin
        transmission. Used by BandpassFilter and lossy components.
        """
        T_bwd = T_fwd if T_bwd is None else T_bwd
        return replace(
            self,
            fwd_spectrum=self.fwd_spectrum * T_fwd,
            bwd_spectrum=self.bwd_spectrum * T_bwd,
        )

    def with_blocked_backward(self) -> "AseState":
        """Drop the backward spectrum (perfect-isolator behaviour)."""
        return replace(self, bwd_spectrum=np.zeros_like(self.bwd_spectrum))

    def with_attenuated_backward(self, isolation_dB: float) -> "AseState":
        """Attenuate the backward spectrum by the isolator's reverse isolation.

        At the default 30 dB this leaves 1e-3 of the backward ASE — effectively
        zero for downstream propagation, but lets a real isolator's finite
        isolation be modelled when needed.
        """
        factor = 10 ** (-isolation_dB / 10)
        return replace(self, bwd_spectrum=self.bwd_spectrum * factor)


@dataclass(frozen=True)
class OpticalState:
    """The full optical state passed through the component chain."""

    signal: "Signal"            # forward reference; defined in components.py
    ase: Optional[AseState] = None

    def with_signal(self, new_signal) -> "OpticalState":
        return replace(self, signal=new_signal)

    def with_ase(self, new_ase: Optional[AseState]) -> "OpticalState":
        return replace(self, ase=new_ase)
