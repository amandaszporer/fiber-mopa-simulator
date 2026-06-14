"""
Yb-doped fiber MOPA simulator — component definitions.

Every component inherits from `Component`, declares its parameter schema in
`parameters()`, and implements `propagate(state: OpticalState) -> OpticalState`.
The `to_dict()` / `from_dict()` pair serialises a configured component to and
from a JSON-friendly dict, so a chain of components can be saved as a
`SystemConfig` (see `framework.py`).

`Signal` carries the coherent pulse state. `AseState` (in `ase/state.py`)
carries the spectrally-resolved bidirectional ASE. The two are bundled by
`OpticalState`. Components that don't care about ASE leave it untouched.

The Amplifier delegates physics to the BVP solver in `ase/`. See
`docs/ase.md` for the physics and `ase/solver_steady.py` for the numerics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np

from ase.dopants import DopantData, get_dopant
from ase.solver_steady import AmplifierGeometry
from ase.solver_time import solve_steady_state_robust, solve_time_dependent
from ase.spectral_grid import SpectralGrid
from ase.state import AseState, OpticalState

# ── Physical constants ────────────────────────────────────────────────
h       = 6.626e-34        # Planck constant [J·s]
c       = 3e8              # Speed of light [m/s]
lam_pump   = 976e-9        # Default pump wavelength [m]
lam_signal = 1064e-9       # Default signal wavelength [m]
nu_pump    = c / lam_pump
nu_signal  = c / lam_signal

# Legacy Yb scalars (kept so any external code still importing them keeps working).
# The BVP solver reads λ-dependent values from `ase/data/cross_sections_yb.csv`.
sigma_a_pump   = 2.5e-24
sigma_e_pump   = 2.5e-24
sigma_a_signal = 0.01e-24
sigma_e_signal = 0.35e-24
tau = 0.83e-3  # Melkumov AS; runtime value comes from DopantData.tau (see dopants.py)

# Nonlinear coefficients
g_B_intrinsic = 3e-11
dnu_B         = 35e6
T_phonon      = 10e-9
g_R           = 1e-13
n_glass       = 1.45

# Pulse shape factor (Gaussian)
SHAPE_FACTOR = 0.94


# ── Signal ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Signal:
    """Coherent-pulse optical state. Scalar quantities only."""
    average_power: float    # [W]
    peak_power: float       # [W]
    pulse_energy: float     # [J]
    rep_rate: float         # [Hz]
    pulse_duration: float   # [s]
    linewidth: float        # [Hz]
    wavelength: float       # [m]
    mfd: float              # [m]

    def scaled(self, loss_dB: float) -> "Signal":
        """New Signal with coherent powers/energy reduced by loss_dB."""
        factor = 10 ** (-loss_dB / 10)
        return replace(
            self,
            average_power=self.average_power * factor,
            peak_power=self.peak_power * factor,
            pulse_energy=self.pulse_energy * factor,
        )


def make_seed() -> Signal:
    """Default seed signal — 0.75 mW @ 100 kHz, 8 ns, 10 GHz, 1064 nm, 5 µm MFD."""
    avg = 0.75e-3
    rep = 100e3
    dur = 8e-9
    energy = avg / rep
    peak = energy / (dur * SHAPE_FACTOR)
    return Signal(
        average_power=avg,
        peak_power=peak,
        pulse_energy=energy,
        rep_rate=rep,
        pulse_duration=dur,
        linewidth=10e9,
        wavelength=1064e-9,
        mfd=5.0e-6,
    )


def make_seed_state(grid: Optional[SpectralGrid] = None) -> OpticalState:
    """Default seed wrapped in an OpticalState with zero ASE."""
    sig = make_seed()
    ase = AseState.zero(grid) if grid is not None else None
    return OpticalState(signal=sig, ase=ase)


# ── Parameter schema and component base class ───────────────────────────

@dataclass(frozen=True)
class Param:
    """Parameter metadata for a component.

    Used by `Component.from_dict` for validation, by future GUIs to render
    input forms, and by `to_dict` to know which attributes to serialise.
    `min`/`max` are inclusive; `choices` (if set) overrides them. Setting
    `unit` is purely advisory.
    """
    type: type
    unit: str = ""
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[tuple] = None
    description: str = ""

    def validate(self, name: str, value: Any) -> Any:
        """Coerce + validate a value. Returns the coerced value or raises."""
        if value is None:
            if self.default is None:
                raise ValueError(f"Parameter {name!r} is required (no default)")
            value = self.default
        if self.type is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, self.type):
            raise ValueError(
                f"Parameter {name!r}: expected {self.type.__name__}, "
                f"got {type(value).__name__} ({value!r})"
            )
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"Parameter {name!r} = {value!r} not in allowed choices {self.choices}"
            )
        if self.min is not None and value < self.min:
            raise ValueError(
                f"Parameter {name!r} = {value} below minimum {self.min}"
            )
        if self.max is not None and value > self.max:
            raise ValueError(
                f"Parameter {name!r} = {value} above maximum {self.max}"
            )
        return value


COMPONENT_REGISTRY: Dict[str, type] = {}


def register_component(cls: type) -> type:
    """Class decorator — adds a component to COMPONENT_REGISTRY by class name."""
    COMPONENT_REGISTRY[cls.__name__] = cls
    return cls


class Component:
    """Base class for all optical components.

    Subclasses must:
      1. Override `parameters()` (classmethod) to declare a dict of `Param`s.
      2. Override `propagate(state)`.
      3. Use the @register_component decorator.

    The base `__init__` reads each declared parameter from kwargs, validates it
    against the declared schema (type + range/choices), and stores it as an
    attribute. Out-of-range values raise `ValueError`.
    """

    name: str
    info: Dict[str, Any]

    def __init__(self, name: str, **params: Any) -> None:
        self.name = name
        self.info = {}
        param_meta = self.parameters()
        consumed: set = set()
        for key, meta in param_meta.items():
            value = params.get(key, meta.default)
            value = meta.validate(key, value)
            setattr(self, key, value)
            consumed.add(key)
        leftover = set(params) - consumed
        if leftover:
            raise ValueError(
                f"{type(self).__name__}: unexpected parameter(s) {sorted(leftover)}; "
                f"valid: {sorted(param_meta)}"
            )

    def propagate(self, state: OpticalState) -> OpticalState:
        raise NotImplementedError

    @classmethod
    def parameters(cls) -> Dict[str, Param]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict including the type tag."""
        d: Dict[str, Any] = {"type": type(self).__name__, "name": self.name}
        for key in self.parameters():
            d[key] = getattr(self, key)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Component":
        """Reconstruct a component from a dict produced by `to_dict`.

        If `cls` is `Component` itself, dispatches via COMPONENT_REGISTRY.
        Otherwise the dict's `type` (if present) must match `cls.__name__`.
        """
        d = dict(d)
        type_tag = d.pop("type", None)
        if cls is Component:
            if type_tag is None:
                raise ValueError("from_dict on Component requires a 'type' field")
            if type_tag not in COMPONENT_REGISTRY:
                raise ValueError(
                    f"Unknown component type {type_tag!r}; "
                    f"known: {sorted(COMPONENT_REGISTRY)}"
                )
            return COMPONENT_REGISTRY[type_tag].from_dict({"type": type_tag, **d})
        if type_tag is not None and type_tag != cls.__name__:
            raise ValueError(
                f"Cannot build {cls.__name__} from dict tagged {type_tag!r}"
            )
        return cls(**d)


def component_from_dict(d: Dict[str, Any]) -> Component:
    """Dispatch from a dict's `type` field to the right Component subclass."""
    return Component.from_dict(d)


def _flat_attenuate_ase(state: OpticalState, loss_dB: float) -> Optional[AseState]:
    return state.ase.with_flat_loss(loss_dB) if state.ase is not None else None


# ── Passive components ───────────────────────────────────────────────

@register_component
class Isolator(Component):
    """Forward-pass insertion loss; attenuates backward ASE by isolation_dB."""

    @classmethod
    def parameters(cls) -> Dict[str, Param]:
        return {
            "insertion_loss_dB": Param(
                type=float, unit="dB", min=0.0, max=5.0, default=1.0,
                description="Forward signal insertion loss",
            ),
            "isolation_dB": Param(
                type=float, unit="dB", min=10.0, max=60.0, default=30.0,
                description="Backward attenuation (signal + ASE blocked from "
                            "propagating upstream)",
            ),
        }

    def propagate(self, state: OpticalState) -> OpticalState:
        new_signal = state.signal.scaled(self.insertion_loss_dB)
        new_ase = state.ase
        if new_ase is not None:
            # Forward sees insertion_loss_dB; backward sees isolation_dB
            # (the datasheet quote for total backward attenuation).
            factor_fwd = 10 ** (-self.insertion_loss_dB / 10)
            factor_bwd = 10 ** (-self.isolation_dB / 10)
            new_ase = replace(
                new_ase,
                fwd_spectrum=new_ase.fwd_spectrum * factor_fwd,
                bwd_spectrum=new_ase.bwd_spectrum * factor_bwd,
            )
        self.info = {
            "insertion_loss_dB": self.insertion_loss_dB,
            "isolation_dB": self.isolation_dB,
        }
        return OpticalState(signal=new_signal, ase=new_ase)


@register_component
class PumpCombiner(Component):
    """Signal-path insertion loss. The pump injection itself is modelled as an
    Amplifier parameter (`pump_power`) — this component captures only the
    signal-path attenuation introduced by the combiner."""

    @classmethod
    def parameters(cls) -> Dict[str, Param]:
        return {
            "insertion_loss_dB": Param(
                type=float, unit="dB", min=0.0, max=3.0, default=0.3,
                description="Signal-path insertion loss",
            ),
        }

    def propagate(self, state: OpticalState) -> OpticalState:
        new_signal = state.signal.scaled(self.insertion_loss_dB)
        new_ase = _flat_attenuate_ase(state, self.insertion_loss_dB)
        self.info = {"insertion_loss_dB": self.insertion_loss_dB}
        return OpticalState(signal=new_signal, ase=new_ase)


@register_component
class Circulator(Component):
    """One port-pair pass of a 3-port optical circulator.

    A single component instance represents ONE direction through the
    circulator — either port 1 → port 2 (with the 1→2 datasheet loss) OR
    port 2 → port 3 (with the 2→3 datasheet loss). In the BGU MOPA each
    physical circulator is used in *double-pass* mode with an FBG as the
    reflector, so the JSON places two `Circulator` entries — one before
    the BPF/FBG and one after — to model both passes faithfully.

    Backward-going ASE is attenuated by the same insertion loss; in the
    real device port 2→1 (or 3→2) is blocked by ≥25 dB intrinsic
    isolation, but by the time bwd ASE reaches the circulator it has
    already been quenched by the upstream isolator, so this minor
    asymmetry is below the noise floor.
    """

    @classmethod
    def parameters(cls) -> Dict[str, Param]:
        return {
            "insertion_loss_dB": Param(
                type=float, unit="dB", min=0.0, max=3.0, default=0.8,
                description="Forward signal insertion loss for this port-pair "
                            "pass (1→2 OR 2→3)",
            ),
        }

    def propagate(self, state: OpticalState) -> OpticalState:
        new_signal = state.signal.scaled(self.insertion_loss_dB)
        new_ase = _flat_attenuate_ase(state, self.insertion_loss_dB)
        self.info = {"insertion_loss_dB": self.insertion_loss_dB}
        return OpticalState(signal=new_signal, ase=new_ase)


@register_component
class ModeFieldAdapter(Component):
    """Bridges the size mismatch between fibers of different core diameters.

    Applies a flat insertion loss and updates the signal MFD (which then
    determines A_eff for nonlinear thresholds in the next amplifier).
    """

    @classmethod
    def parameters(cls) -> Dict[str, Param]:
        return {
            "insertion_loss_dB": Param(
                type=float, unit="dB", min=0.0, max=3.0, default=0.3,
                description="Forward signal insertion loss",
            ),
            "output_mfd": Param(
                type=float, unit="m", min=1e-6, max=50e-6, default=10e-6,
                description="Mode field diameter at the output port",
            ),
        }

    def propagate(self, state: OpticalState) -> OpticalState:
        new_signal = state.signal.scaled(self.insertion_loss_dB)
        new_signal = replace(new_signal, mfd=self.output_mfd)
        new_ase = _flat_attenuate_ase(state, self.insertion_loss_dB)
        self.info = {
            "insertion_loss_dB": self.insertion_loss_dB,
            "mfd_in": state.signal.mfd * 1e6,
            "mfd_out": self.output_mfd * 1e6,
        }
        return OpticalState(signal=new_signal, ase=new_ase)


@register_component
class BandpassFilter(Component):
    """Inter-stage spectral filter — Gaussian transmission with a rejection floor.

    T(λ) = max(T_peak · exp(-4·ln2·((λ - λ_c)/FWHM)²), 10^(-rejection_dB/10))

    The floor models real interference filters, which have a sidelobe level
    rather than infinite rejection. Default 40 dB rejection floor matches a
    typical commercial bandpass.
    """

    @classmethod
    def parameters(cls) -> Dict[str, Param]:
        return {
            "center_wavelength": Param(
                type=float, unit="m", min=900e-9, max=2100e-9, default=1064e-9,
                description="Filter center wavelength",
            ),
            "fwhm": Param(
                type=float, unit="m", min=0.1e-9, max=50e-9, default=2e-9,
                description="Full width at half maximum",
            ),
            "insertion_loss_dB": Param(
                type=float, unit="dB", min=0.0, max=3.0, default=0.5,
                description="Peak insertion loss (at center)",
            ),
            "rejection_dB": Param(
                type=float, unit="dB", min=20.0, max=60.0, default=40.0,
                description="Rejection floor for far-detuned light",
            ),
        }

    def _transfer(self, wavelengths: np.ndarray) -> np.ndarray:
        T_peak = 10 ** (-self.insertion_loss_dB / 10)
        gauss = T_peak * np.exp(
            -4 * math.log(2)
            * ((wavelengths - self.center_wavelength) / self.fwhm) ** 2
        )
        floor = 10 ** (-self.rejection_dB / 10)
        return np.maximum(gauss, floor)

    def propagate(self, state: OpticalState) -> OpticalState:
        new_signal = state.signal.scaled(self.insertion_loss_dB)
        new_ase = state.ase
        if new_ase is not None:
            T = self._transfer(new_ase.spectral_grid.wavelengths)
            new_ase = new_ase.with_spectral_transfer(T)
        self.info = {
            "center_wl_nm": self.center_wavelength * 1e9,
            "fwhm_nm": self.fwhm * 1e9,
            "insertion_loss_dB": self.insertion_loss_dB,
            "rejection_dB": self.rejection_dB,
        }
        return OpticalState(signal=new_signal, ase=new_ase)


@register_component
class FusionSplice(Component):
    """A fusion splice between two fibers. A small flat loss; nothing else."""

    @classmethod
    def parameters(cls) -> Dict[str, Param]:
        return {
            "insertion_loss_dB": Param(
                type=float, unit="dB", min=0.0, max=1.0, default=0.03,
                description="Splice insertion loss",
            ),
        }

    def propagate(self, state: OpticalState) -> OpticalState:
        new_signal = state.signal.scaled(self.insertion_loss_dB)
        new_ase = _flat_attenuate_ase(state, self.insertion_loss_dB)
        self.info = {"insertion_loss_dB": self.insertion_loss_dB}
        return OpticalState(signal=new_signal, ase=new_ase)


@register_component
class PassiveFiber(Component):
    """A length of undoped fiber. Total loss = length · loss_dB_per_m, applied
    flat to signal and ASE. Used for delay lines, connector pigtails, etc."""

    @classmethod
    def parameters(cls) -> Dict[str, Param]:
        return {
            "length": Param(
                type=float, unit="m", min=0.01, max=100.0, default=1.0,
                description="Fiber length",
            ),
            "loss_dB_per_m": Param(
                type=float, unit="dB/m", min=0.0, max=1.0, default=0.005,
                description="Background propagation loss",
            ),
        }

    @property
    def total_loss_dB(self) -> float:
        return self.length * self.loss_dB_per_m

    def propagate(self, state: OpticalState) -> OpticalState:
        loss = self.total_loss_dB
        new_signal = state.signal.scaled(loss)
        new_ase = _flat_attenuate_ase(state, loss)
        self.info = {
            "length_m": self.length,
            "loss_dB_per_m": self.loss_dB_per_m,
            "total_loss_dB": loss,
        }
        return OpticalState(signal=new_signal, ase=new_ase)


# ── Amplifier ─────────────────────────────────────────────────────────

@register_component
class Amplifier(Component):
    """Active fiber amplifier — delegates physics to the BVP solver in `ase/`.

    The Amplifier carries the fiber geometry, dopant identity, pump
    configuration, and solver knobs. On `propagate(state)` it builds (or
    reuses) a SpectralGrid for its core/NA/dopant, runs the BVP solver,
    populates `self.info` with diagnostics, and returns an OpticalState
    with the updated signal and a fresh AseState.
    """

    MAX_GAIN_DB = 40.0

    @classmethod
    def parameters(cls) -> Dict[str, Param]:
        return {
            "core_diameter": Param(
                type=float, unit="m", min=1e-6, max=100e-6, default=5e-6,
                description="Doped-core diameter",
            ),
            "clad_diameter": Param(
                type=float, unit="m", min=50e-6, max=500e-6, default=130e-6,
                description="Inner cladding diameter (where pump propagates)",
            ),
            "core_na": Param(
                type=float, min=0.01, max=0.5, default=0.12,
                description="Core numerical aperture",
            ),
            "length": Param(
                type=float, unit="m", min=0.1, max=20.0, default=3.0,
                description="Fiber length",
            ),
            "clad_absorption_dB_per_m": Param(
                type=float, unit="dB/m", min=0.1, max=20.0, default=1.65,
                description="Cladding absorption at the pump wavelength "
                            "(used to derive N_dopant)",
            ),
            "pump_power": Param(
                type=float, unit="W", min=0.0, max=500.0, default=1.0,
                description="Input pump power",
            ),
            "pump_direction": Param(
                type=str, choices=("co", "counter"), default="co",
                description="Pump direction relative to the signal",
            ),
            "pump_wavelength": Param(
                type=float, unit="m", min=700e-9, max=2100e-9, default=976e-9,
                description="Pump diode wavelength",
            ),
            "signal_wavelength": Param(
                type=float, unit="m", min=900e-9, max=2200e-9, default=1064e-9,
                description="Operating signal wavelength",
            ),
            "dopant": Param(
                type=str, default="Yb",
                description="Dopant key in DOPANT_REGISTRY (e.g. 'Yb')",
            ),
            "num_segments": Param(
                type=int, min=10, max=1000, default=200,
                description="Number of z-grid points for the BVP solve",
            ),
            "R_in": Param(
                type=float, min=0.0, max=1.0, default=0.0,
                description="Power reflectivity at the input facet",
            ),
            "R_out": Param(
                type=float, min=0.0, max=1.0, default=1e-4,
                description="Power reflectivity at the output facet "
                            "(default = 8° angle-cleaved)",
            ),
            "m_pol": Param(
                type=int, choices=(1, 2), default=2,
                description="Polarization mode count: 1 for PM, 2 otherwise",
            ),
            "alpha_bg_dB_per_m": Param(
                type=float, unit="dB/m", min=0.0, max=1.0, default=0.005,
                description="Background fiber loss outside the gain band",
            ),
        }

    def __init__(self, name: str, **params: Any) -> None:
        super().__init__(name, **params)

        # Geometry derived from declared parameters
        r_core = self.core_diameter / 2
        r_clad = self.clad_diameter / 2
        self.A_core = math.pi * r_core ** 2
        self.A_clad = math.pi * r_clad ** 2

        # V-number at the signal wavelength
        self.V = (2 * math.pi / self.signal_wavelength) * r_core * self.core_na

        # MFD (Marcuse for V<2.405, fundamental-mode cap above)
        if self.V < 2.405:
            w_over_a = 0.65 + 1.619 / self.V ** 1.5 + 2.879 / self.V ** 6
            self.fiber_mfd = 2 * w_over_a * r_core
        else:
            self.fiber_mfd = 2 * 0.65 * r_core
        self.A_eff = math.pi * (self.fiber_mfd / 2) ** 2

        w = self.fiber_mfd / 2
        self.gamma_signal = 1 - math.exp(-2 * (r_core / w) ** 2)
        self.gamma_pump = self.A_core / self.A_clad

        # Dopant + doping concentration (derived from cladding absorption)
        try:
            self._dopant: DopantData = get_dopant(self.dopant)
        except KeyError as e:
            raise ValueError(str(e)) from e
        sa_pump_local = self._dopant.sigma_a_at(self.pump_wavelength)
        self.N_dopant = (
            self.clad_absorption_dB_per_m
            / (4.343 * sa_pump_local * self.gamma_pump)
        )
        # Legacy attribute name retained for downstream reporting
        self.N_Yb = self.N_dopant

        self._geom = AmplifierGeometry(
            fiber_length=self.length,
            A_core=self.A_core,
            A_clad=self.A_clad,
            N_Yb=self.N_dopant,
            gamma_pump=self.gamma_pump,
            tau=self._dopant.tau,
            alpha_bg_dB_per_m=self.alpha_bg_dB_per_m,
            m_pol=self.m_pol,
        )
        self._grid: Optional[SpectralGrid] = None

    @property
    def grid(self) -> SpectralGrid:
        if self._grid is None:
            self._grid = SpectralGrid.from_fiber(
                r_core=self.core_diameter / 2,
                NA=self.core_na,
                dopant=self._dopant,
                pump_wavelength=self.pump_wavelength,
                signal_wavelength=self.signal_wavelength,
            )
        return self._grid

    # Legacy attribute aliases used by downstream report code
    @property
    def core_diameter_um(self) -> float:
        return self.core_diameter * 1e6

    @property
    def fiber_length(self) -> float:
        # Backwards-compat for old tests/report code that referenced fiber_length
        return self.length

    @property
    def na(self) -> float:
        # Backwards-compat alias
        return self.core_na

    @property
    def n_segments(self) -> int:
        return self.num_segments

    @property
    def cladding_abs_dB_per_m(self) -> float:
        return self.clad_absorption_dB_per_m

    # ── Pre-built fiber presets ──────────────────────────────────────

    @classmethod
    def yb_5_130(cls, name: str = "Amp", *, pump_power: float = 1.0,
                  length: float = 3.0) -> "Amplifier":
        """Nufern-style Yb 5/130 µm single-mode pre-amp fiber."""
        return cls(
            name=name,
            core_diameter=5e-6, clad_diameter=130e-6, core_na=0.12,
            length=length, clad_absorption_dB_per_m=1.65,
            pump_power=pump_power, dopant="Yb",
        )

    @classmethod
    def yb_10_125(cls, name: str = "Amp", *, pump_power: float = 9.0,
                   length: float = 2.0) -> "Amplifier":
        """Nufern-style Yb 10/125 µm few-mode mid-stage fiber."""
        return cls(
            name=name,
            core_diameter=10e-6, clad_diameter=125e-6, core_na=0.075,
            length=length, clad_absorption_dB_per_m=2.5,
            pump_power=pump_power, dopant="Yb",
        )

    @classmethod
    def yb_30_250(cls, name: str = "Amp", *, pump_power: float = 100.0,
                   length: float = 1.2) -> "Amplifier":
        """Nufern-style Yb 30/250 µm large-mode-area power-stage fiber."""
        return cls(
            name=name,
            core_diameter=30e-6, clad_diameter=220e-6, core_na=0.06,
            length=length, clad_absorption_dB_per_m=5.0,
            pump_power=pump_power, dopant="Yb",
        )

    # Backwards-compat: stage1/2/3 used by older tests and lab-mode code.
    @classmethod
    def stage1(cls, pump_power: float = 1.0, fiber_length: float = 3.0) -> "Amplifier":
        return cls.yb_5_130(name="Amp-Stage1", pump_power=pump_power, length=fiber_length)

    @classmethod
    def stage2(cls, pump_power: float = 9.0, fiber_length: float = 2.0) -> "Amplifier":
        return cls.yb_10_125(name="Amp-Stage2", pump_power=pump_power, length=fiber_length)

    @classmethod
    def stage3(cls, pump_power: float = 100.0, fiber_length: float = 1.2) -> "Amplifier":
        return cls.yb_30_250(name="Amp-Stage3", pump_power=pump_power, length=fiber_length)

    # ── Propagation ───────────────────────────────────────────────────

    def propagate(self, state: OpticalState, mode: str = "time-dependent") -> OpticalState:
        """Run the BVP solver and return an updated OpticalState.

        mode="time-dependent" → Mode B with auto-dispatch (DEFAULT). At high
                                rep (period < 0.1·τ) this delegates to Mode A
                                and is bit-identical to it. At lower rep
                                rates it runs the Level 5 B1+B2 cycle. Always
                                picks the most physically accurate path.
        mode="steady"         → Force Mode A (steady-state BVP) regardless
                                of rep rate. Useful when you want to compare
                                against quasi-CW or for backwards-compat.
        mode="full"           → Force the B1+B2 path (Level 5) even at high
                                rep. Enables pulse-shape diagnostics; same
                                average-power answer as Mode A at 100 kHz
                                but ~10× slower.
        """
        signal = state.signal
        signal = replace(signal, mfd=self.fiber_mfd)

        grid = self.grid
        ase_in_fwd = (
            state.ase.fwd_spectrum
            if state.ase is not None and state.ase.spectral_grid.n_bins == grid.n_bins
            else np.zeros(grid.n_bins)
        )

        if mode == "steady":
            result = solve_steady_state_robust(
                geom=self._geom, grid=grid,
                P_pump=self.pump_power,
                P_signal_avg=signal.average_power,
                ase_in_fwd=ase_in_fwd,
                R_in=self.R_in, R_out=self.R_out,
                n_z=self.num_segments,
                pump_direction=self.pump_direction,
            )
        elif mode in ("time-dependent", "full"):
            result = solve_time_dependent(
                geom=self._geom, grid=grid,
                P_pump=self.pump_power,
                P_signal_avg=signal.average_power,
                rep_rate=signal.rep_rate,
                pulse_duration=signal.pulse_duration,
                pulse_energy=signal.pulse_energy,
                ase_in_fwd=ase_in_fwd,
                R_in=self.R_in, R_out=self.R_out,
                n_z=self.num_segments,
                mode="full" if mode == "full" else "auto",
                pump_direction=self.pump_direction,
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        new_ase = AseState(
            spectral_grid=grid,
            fwd_spectrum=result.ase_fwd_out.copy(),
            bwd_spectrum=result.ase_bwd_in.copy(),
            n2_profile=result.n2_z.copy(),
            z_grid=result.z.copy(),
            solver_failed=result.solver_failed,
            under_resolved=result.under_resolved,
        )

        gain_linear = (
            result.signal_out / signal.average_power
            if signal.average_power > 0 else 1.0
        )
        gain_capped = gain_linear > 10 ** (self.MAX_GAIN_DB / 10)
        gain_dB = 10 * math.log10(max(gain_linear, 1e-12)) if gain_linear > 0 else -999.0
        new_peak = signal.peak_power * gain_linear
        new_energy = result.signal_out / signal.rep_rate if signal.rep_rate > 0 else 0.0

        # Effective length from the actual gain profile
        if gain_linear > 1:
            g_per_m = math.log(gain_linear) / self.length
            L_eff_fiber = (
                (1 - math.exp(-g_per_m * self.length)) / g_per_m
                if g_per_m > 0 else self.length
            )
        else:
            L_eff_fiber = self.length

        L_pulse = c * signal.pulse_duration / (2 * n_glass)
        L_eff = min(L_eff_fiber, L_pulse)
        P_peak_out = new_peak

        # SBS
        g_B_eff = g_B_intrinsic / (1 + signal.linewidth / dnu_B)
        P_th_SBS = 21 * 1 * self.A_eff / (g_B_eff * L_eff)
        if signal.pulse_duration < 5 * T_phonon:
            r_p = signal.pulse_duration / T_phonon
            transient_factor = r_p / (1 - math.exp(-r_p))
            P_th_SBS *= transient_factor
        sbs_ratio = P_peak_out / P_th_SBS

        # SRS
        P_th_SRS = 16 * self.A_eff / (g_R * L_eff)
        srs_ratio = P_peak_out / P_th_SRS

        # ASE summary
        total_ase_fwd = float(new_ase.total_fwd())
        total_ase_bwd = float(new_ase.total_bwd())
        sig_out = result.signal_out
        ase_ratio_dB = (
            10 * math.log10(total_ase_fwd / sig_out)
            if total_ase_fwd > 0 and sig_out > 0 else -100.0
        )

        peak_fwd_idx = int(np.argmax(result.ase_fwd_out)) if result.ase_fwd_out.sum() > 0 else 0
        peak_bwd_idx = int(np.argmax(result.ase_bwd_in)) if result.ase_bwd_in.sum() > 0 else 0
        P_ase_z = result.P_ase_fwd_z.sum(axis=1)

        self.info = {
            "gain_linear": gain_linear,
            "gain_dB": gain_dB,
            "gain_capped": gain_capped,
            "P_signal_in": signal.average_power,
            "P_signal_out": result.signal_out,
            "P_pump_residual": result.pump_residual,
            "pump_absorption_pct": (
                (1 - result.pump_residual / self.pump_power) * 100
                if self.pump_power > 0 else 0.0
            ),
            "peak_power_out": P_peak_out,
            "sbs_threshold": P_th_SBS,
            "sbs_ratio": sbs_ratio,
            "sbs_safe": sbs_ratio < 1.0,
            "srs_threshold": P_th_SRS,
            "srs_ratio": srs_ratio,
            "srs_safe": srs_ratio < 1.0,
            "ase_power_out": total_ase_fwd,
            "ase_power_bwd": total_ase_bwd,
            "ase_ratio_dB": ase_ratio_dB,
            "ase_safe": ase_ratio_dB < -20,
            "peak_ase_fwd_wavelength_nm": float(grid.wavelengths[peak_fwd_idx] * 1e9),
            "peak_ase_bwd_wavelength_nm": float(grid.wavelengths[peak_bwd_idx] * 1e9),
            "solver_iterations": result.iterations,
            "solver_converged": result.converged,
            "solver_failed": result.solver_failed,
            "under_resolved": result.under_resolved,
            "solver_notes": list(result.notes),
            "solver_path_used": result.solver_path_used,
            "homotopy_steps_used": result.homotopy_steps_used,
            # Energy/regime diagnostics — see ase/solver_health.py. None when
            # the result comes from a path that bypasses the robust wrapper.
            "energy_residual_ratio":
                result.health.energy_residual_ratio if result.health else 0.0,
            "ase_conversion_fraction":
                result.health.ase_conversion_fraction if result.health else 0.0,
            "small_signal_g0L":
                result.health.small_signal_g0L if result.health else 0.0,
            "energy_status":
                result.health.energy_status if result.health else "ok",
            "regime":
                result.health.regime if result.health else "amplifier",
            "fiber_mfd_um": self.fiber_mfd * 1e6,
            "A_eff_um2": self.A_eff * 1e12,
            "N_Yb": self.N_dopant,
            "N_dopant": self.N_dopant,
            "V_number": self.V,
            "gamma_signal": self.gamma_signal,
            "z": result.z,
            "P_signal_z": result.P_signal_z,
            "P_pump_z": result.P_pump_z,
            "P_ase_z": P_ase_z,
            "n2_z": result.n2_z,
            "wavelengths_nm": grid.wavelengths * 1e9,
            "ase_fwd_spectrum_W": result.ase_fwd_out,
            "ase_bwd_spectrum_W": result.ase_bwd_in,
            "ase_fwd_spectrum_z": result.P_ase_fwd_z,
            "ase_bwd_spectrum_z": result.P_ase_bwd_z,
        }

        # B2 (Level 5) only — Mode A and high-rep quasi-CW leave these as None.
        if result.t is not None and result.P_signal_tz is not None:
            self.info["t_ns"] = result.t * 1e9
            self.info["P_signal_tz_W"] = result.P_signal_tz
            self.info["P_signal_out_t_W"] = result.P_signal_tz[:, -1]
            self.info["P_signal_in_t_W"] = result.P_signal_tz[:, 0]

        new_signal = replace(
            signal,
            average_power=result.signal_out,
            peak_power=new_peak,
            pulse_energy=new_energy,
        )
        return OpticalState(signal=new_signal, ase=new_ase)


__all__ = [
    "Signal",
    "Param",
    "Component",
    "COMPONENT_REGISTRY",
    "register_component",
    "component_from_dict",
    "Isolator",
    "PumpCombiner",
    "Circulator",
    "ModeFieldAdapter",
    "BandpassFilter",
    "FusionSplice",
    "PassiveFiber",
    "Amplifier",
    "make_seed",
    "make_seed_state",
]
