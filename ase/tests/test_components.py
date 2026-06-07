"""Component-level tests: BandpassFilter, OpticalState contract."""

import numpy as np

from ase.spectral_grid import SpectralGrid
from ase.state import AseState, OpticalState
from components import (
    Amplifier,
    BandpassFilter,
    Isolator,
    Signal,
    make_seed_state,
)


def _seed_state_with_uniform_ase(grid: SpectralGrid, fwd_W: float = 1e-6) -> OpticalState:
    fwd = np.full(grid.n_bins, fwd_W)
    bwd = np.full(grid.n_bins, fwd_W)
    ase = AseState(spectral_grid=grid, fwd_spectrum=fwd, bwd_spectrum=bwd)
    sig = Signal(
        average_power=1e-3, peak_power=1.0, pulse_energy=1e-8,
        rep_rate=100e3, pulse_duration=8e-9, linewidth=10e9,
        wavelength=1064e-9, mfd=5e-6,
    )
    return OpticalState(signal=sig, ase=ase)


def test_bandpass_filter_rejects_off_band():
    """A 1064 nm / 2 nm FWHM filter should reject 1030 nm by ~30+ dB while
    passing 1064 nm at the peak loss only."""
    grid = SpectralGrid.from_fiber(r_core=2.5e-6, NA=0.12)
    state = _seed_state_with_uniform_ase(grid, fwd_W=1e-6)

    bpf = BandpassFilter(
        name="BPF", center_wavelength=1064e-9, fwhm=2e-9,
        insertion_loss_dB=0.5, rejection_dB=40.0,
    )
    out = bpf.propagate(state)

    i_1030 = int(np.abs(grid.wavelengths - 1030e-9).argmin())
    i_1064 = int(np.abs(grid.wavelengths - 1064e-9).argmin())
    rejection_1030 = 10 * np.log10(
        state.ase.fwd_spectrum[i_1030] / max(out.ase.fwd_spectrum[i_1030], 1e-30)
    )
    insertion_1064 = 10 * np.log10(
        state.ase.fwd_spectrum[i_1064] / max(out.ase.fwd_spectrum[i_1064], 1e-30)
    )
    assert rejection_1030 > 20, (
        f"expected ≥20 dB rejection at 1030 nm, got {rejection_1030:.1f} dB"
    )
    # The ASE bin nearest 1064 nm is at the bin center (1064.5 nm), 0.5 nm
    # off the filter center. With FWHM=2 nm that's a Gaussian factor ~0.84,
    # so the in-band attenuation is ~peak_loss + 0.7 dB. Allow up to 1.5 dB.
    assert insertion_1064 < 1.5, (
        f"in-band insertion loss too high: {insertion_1064:.2f} dB"
    )


def test_isolator_attenuates_backward_ase():
    """A default isolator (30 dB isolation) drops backward ASE by 1e-3."""
    grid = SpectralGrid.from_fiber(r_core=2.5e-6, NA=0.12)
    state = _seed_state_with_uniform_ase(grid, fwd_W=1e-6)
    iso = Isolator(name="Iso", insertion_loss_dB=0.5, isolation_dB=30.0)
    out = iso.propagate(state)
    expected_bwd = state.ase.total_bwd() * 10 ** (-3.0)
    assert abs(out.ase.total_bwd() - expected_bwd) / expected_bwd < 1e-9
    factor = 10 ** (-0.5 / 10)
    expected_fwd = state.ase.total_fwd() * factor
    assert abs(out.ase.total_fwd() - expected_fwd) / expected_fwd < 1e-9


def test_amp_propagates_with_optical_state(stage1_setup):
    """The Amplifier should accept an OpticalState (with ASE) and return one."""
    _, grid, _ = stage1_setup
    state = make_seed_state(grid)
    amp = Amplifier.stage1(pump_power=0.3, fiber_length=3.0)
    out = amp.propagate(state)
    assert isinstance(out, OpticalState)
    assert out.ase is not None
    assert out.ase.fwd_spectrum.shape == (grid.n_bins,)
    assert out.signal.average_power > state.signal.average_power
