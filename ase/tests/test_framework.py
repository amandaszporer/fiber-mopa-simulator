"""Tests for the Component / SystemConfig / Simulator refactor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ase.spectral_grid import SpectralGrid
from ase.state import AseState, OpticalState
from components import (
    Amplifier,
    BandpassFilter,
    Circulator,
    Component,
    COMPONENT_REGISTRY,
    FusionSplice,
    Isolator,
    ModeFieldAdapter,
    Param,
    PassiveFiber,
    PumpCombiner,
    Signal,
    component_from_dict,
)
from framework import (
    Simulator,
    SystemConfig,
    signal_from_dict,
    signal_to_dict,
)

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "bgu_3stage_mopa.json"


# ── Component registry & metadata ────────────────────────────────────

def test_every_component_is_registered():
    expected = {
        "Isolator", "PumpCombiner", "Circulator", "ModeFieldAdapter",
        "BandpassFilter", "FusionSplice", "PassiveFiber", "Amplifier",
    }
    assert expected.issubset(set(COMPONENT_REGISTRY))


def test_parameters_metadata_returns_param_objects():
    for cls in COMPONENT_REGISTRY.values():
        meta = cls.parameters()
        assert isinstance(meta, dict)
        for key, p in meta.items():
            assert isinstance(p, Param), f"{cls.__name__}.{key} not a Param"


# ── Validation ───────────────────────────────────────────────────────

def test_param_validation_rejects_below_min():
    with pytest.raises(ValueError, match="below minimum"):
        Isolator(name="Iso", insertion_loss_dB=-1.0)


def test_param_validation_rejects_above_max():
    with pytest.raises(ValueError, match="above maximum"):
        FusionSplice(name="FS", insertion_loss_dB=100.0)


def test_param_validation_rejects_bad_type():
    with pytest.raises(ValueError, match="expected"):
        ModeFieldAdapter(name="MFA", insertion_loss_dB="zero")


def test_param_validation_rejects_bad_choice():
    with pytest.raises(ValueError, match="not in allowed choices"):
        Amplifier(name="A", pump_direction="sideways")


def test_param_validation_rejects_unknown_param():
    with pytest.raises(ValueError, match="unexpected parameter"):
        Isolator(name="Iso", insertion_loss_dB=0.5, foo="bar")


def test_unknown_dopant_raises():
    with pytest.raises(ValueError, match="Unknown dopant"):
        Amplifier(name="A", dopant="Vibranium")


# ── to_dict / from_dict round-trip ───────────────────────────────────

def _all_default_components() -> list[Component]:
    return [
        Isolator(name="ISO"),
        PumpCombiner(name="PC"),
        Circulator(name="CIRC"),
        ModeFieldAdapter(name="MFA"),
        BandpassFilter(name="BPF"),
        FusionSplice(name="FS"),
        PassiveFiber(name="PF"),
        Amplifier(name="AMP"),
    ]


def test_to_dict_round_trip_per_component():
    for c in _all_default_components():
        d = c.to_dict()
        assert d["type"] == type(c).__name__
        assert d["name"] == c.name
        c2 = component_from_dict(d)
        assert type(c2) is type(c)
        for key in c.parameters():
            assert getattr(c, key) == getattr(c2, key), (
                f"{type(c).__name__}.{key} not preserved"
            )


def test_from_dict_through_base_class_dispatches():
    iso = Isolator(name="X", insertion_loss_dB=0.5, isolation_dB=25.0)
    rebuilt = Component.from_dict(iso.to_dict())
    assert isinstance(rebuilt, Isolator)
    assert rebuilt.insertion_loss_dB == 0.5


# ── Flat-loss components apply uniformly to signal and ASE ───────────

def _seed_with_uniform_ase(grid: SpectralGrid, level: float = 1e-6) -> OpticalState:
    fwd = np.full(grid.n_bins, level)
    bwd = np.full(grid.n_bins, level * 0.5)
    sig = Signal(
        average_power=1e-3, peak_power=1.0, pulse_energy=1e-8,
        rep_rate=100e3, pulse_duration=8e-9, linewidth=10e9,
        wavelength=1064e-9, mfd=5e-6,
    )
    return OpticalState(signal=sig, ase=AseState(spectral_grid=grid, fwd_spectrum=fwd, bwd_spectrum=bwd))


def test_pumpcombiner_flat_loss_on_signal_and_ase():
    grid = SpectralGrid.from_fiber(r_core=2.5e-6, NA=0.12)
    state = _seed_with_uniform_ase(grid)
    pc = PumpCombiner(name="PC", insertion_loss_dB=1.0)
    out = pc.propagate(state)
    factor = 10 ** (-1.0 / 10)
    assert abs(out.signal.average_power - state.signal.average_power * factor) < 1e-15
    assert np.allclose(out.ase.fwd_spectrum, state.ase.fwd_spectrum * factor)
    assert np.allclose(out.ase.bwd_spectrum, state.ase.bwd_spectrum * factor)


def test_fusion_splice_loss_math():
    grid = SpectralGrid.from_fiber(r_core=2.5e-6, NA=0.12)
    state = _seed_with_uniform_ase(grid)
    splice = FusionSplice(name="FS", insertion_loss_dB=0.05)
    out = splice.propagate(state)
    factor = 10 ** (-0.05 / 10)
    assert abs(out.signal.average_power - state.signal.average_power * factor) < 1e-15
    assert np.allclose(out.ase.fwd_spectrum, state.ase.fwd_spectrum * factor)


def test_passive_fiber_loss_scales_with_length():
    grid = SpectralGrid.from_fiber(r_core=2.5e-6, NA=0.12)
    state = _seed_with_uniform_ase(grid)
    fiber = PassiveFiber(name="PF", length=10.0, loss_dB_per_m=0.01)
    out = fiber.propagate(state)
    expected_loss_dB = 10.0 * 0.01
    factor = 10 ** (-expected_loss_dB / 10)
    assert abs(out.signal.average_power - state.signal.average_power * factor) < 1e-15
    assert fiber.total_loss_dB == pytest.approx(0.1)


def test_bandpass_filter_rejection_floor():
    """Far off-band the filter should hit its rejection_dB floor (not the
    bare Gaussian which would crush to zero)."""
    grid = SpectralGrid.from_fiber(r_core=2.5e-6, NA=0.12)
    state = _seed_with_uniform_ase(grid)
    bpf = BandpassFilter(
        name="BPF", center_wavelength=1064e-9, fwhm=2e-9,
        insertion_loss_dB=0.5, rejection_dB=40.0,
    )
    out = bpf.propagate(state)
    # 970 nm is 94 nm off from center — at 47 FWHMs, the bare Gaussian is
    # underflow-zero. The floor at 1e-4 relative attenuation (40 dB) should
    # keep the bin alive.
    floor = 10 ** (-40.0 / 10)
    assert out.ase.fwd_spectrum[0] >= state.ase.fwd_spectrum[0] * floor * 0.99


# ── SystemConfig save/load round-trip ────────────────────────────────

def test_signal_dict_round_trip():
    sig = Signal(
        average_power=1e-3, peak_power=1.0, pulse_energy=1e-8,
        rep_rate=100e3, pulse_duration=8e-9, linewidth=10e9,
        wavelength=1064e-9, mfd=5e-6,
    )
    sig2 = signal_from_dict(signal_to_dict(sig))
    for f in ("average_power", "rep_rate", "pulse_duration",
              "linewidth", "wavelength", "mfd"):
        assert getattr(sig, f) == getattr(sig2, f), f


def test_systemconfig_save_load_round_trip(tmp_path: Path):
    cfg = SystemConfig(
        name="test",
        description="round-trip test",
        seed=signal_to_dict(Signal(
            average_power=1e-3, peak_power=1.0, pulse_energy=1e-8,
            rep_rate=100e3, pulse_duration=8e-9, linewidth=10e9,
            wavelength=1064e-9, mfd=5e-6,
        )),
        components=[
            Isolator(name="Iso", insertion_loss_dB=0.5, isolation_dB=30.0).to_dict(),
            PumpCombiner(name="PC").to_dict(),
            Amplifier.yb_5_130(name="A1", pump_power=0.3, length=3.0).to_dict(),
        ],
        metadata={"author": "test", "v": 1},
    )
    path = tmp_path / "cfg.json"
    cfg.save(path)
    cfg2 = SystemConfig.load(path)
    assert cfg.name == cfg2.name
    assert cfg.components == cfg2.components
    assert cfg.metadata == cfg2.metadata


def test_save_emits_valid_json(tmp_path: Path):
    cfg = SystemConfig(
        name="x", components=[Isolator(name="I").to_dict()],
        seed=signal_to_dict(Signal(
            average_power=1e-3, peak_power=1.0, pulse_energy=1e-8,
            rep_rate=100e3, pulse_duration=8e-9, linewidth=10e9,
            wavelength=1064e-9, mfd=5e-6,
        )),
    )
    path = tmp_path / "x.json"
    cfg.save(path)
    parsed = json.loads(path.read_text())
    assert parsed["name"] == "x"
    assert parsed["components"][0]["type"] == "Isolator"


# ── Simulator end-to-end ─────────────────────────────────────────────

def test_simulator_runs_bgu_example():
    cfg = SystemConfig.load(EXAMPLE_PATH)
    sim = Simulator.from_config(cfg)
    state = sim.run()
    assert state.signal.average_power > 30.0   # baseline ~40 W
    assert state.signal.peak_power > 40e3      # ~53 kW
    assert state.ase is not None
    assert state.ase.total_fwd() > 0
    # Should have a result entry per component
    assert len(sim.results) == len(cfg.components)


def test_bgu_example_double_pass_circulator_topology():
    """The BGU chain models each physical circulator as a (1→2) entry, an
    FBG, and a (2→3) entry — so the double-pass loss is captured. This
    test pins that topology in case someone refactors the JSON."""
    cfg = SystemConfig.load(EXAMPLE_PATH)
    sim = Simulator.from_config(cfg)
    circ_names = [c.name for c in sim.components if isinstance(c, Circulator)]
    assert circ_names == [
        "CIRC-1 (1->2)", "CIRC-1 (2->3)",
        "CIRC-2 (1->2)", "CIRC-2 (2->3)",
    ]


def test_bgu_amps_all_take_direct_solver_path():
    """Healthy BGU regime: every amplifier resolves via the Layer-1
    direct solve. Multistart / homotopy / time-marching never invoked."""
    cfg = SystemConfig.load(EXAMPLE_PATH)
    sim = Simulator.from_config(cfg)
    sim.run()
    for amp in sim.amplifiers:
        assert amp.info["solver_path_used"] == "direct", (
            f"{amp.name} took non-direct path: "
            f"{amp.info['solver_path_used']}"
        )
        assert amp.info["energy_status"] == "ok", (
            f"{amp.name} flagged {amp.info['energy_status']}"
        )


def test_bgu_report_has_no_health_tag_decoration():
    """When all amps are healthy, the report should NOT print the
    `[E:.. R:.. path=..]` decoration (clean output is the user contract)."""
    cfg = SystemConfig.load(EXAMPLE_PATH)
    sim = Simulator.from_config(cfg)
    sim.run()
    report = sim.report()
    assert "[E:" not in report, (
        "healthy BGU run should not show energy/regime tags"
    )


def test_simulator_report_renders():
    cfg = SystemConfig.load(EXAMPLE_PATH)
    sim = Simulator.from_config(cfg)
    sim.run()
    report = sim.report()
    assert "V&V Requirements Compliance" in report
    assert "Output:" in report
    assert "ASE AMP-1" in report or "ASE AMP-3" in report


def test_round_trip_via_from_simulator(tmp_path: Path):
    """Simulator -> SystemConfig -> Simulator should reproduce the same numbers."""
    cfg = SystemConfig.load(EXAMPLE_PATH)
    sim_a = Simulator.from_config(cfg)
    sim_a.run()
    cfg_b = SystemConfig.from_simulator(sim_a, name="round-trip", description="")
    # requirements round-trip too: BGU config carries a non-empty block.
    assert cfg_b.requirements == cfg.requirements
    path = tmp_path / "rt.json"
    cfg_b.save(path)
    cfg_loaded = SystemConfig.load(path)
    sim_b = Simulator.from_config(cfg_loaded)
    sim_b.run()
    assert abs(sim_a.final_state.signal.average_power
               - sim_b.final_state.signal.average_power) < 1e-6


def test_simulator_without_requirements_skips_compliance():
    """A config with no requirements block must not print a V&V table."""
    sim = Simulator(
        components=[Amplifier.yb_5_130("AMP", pump_power=1.0, length=2.0)],
        seed=Signal(
            average_power=1e-3, peak_power=125, pulse_energy=10e-9,
            rep_rate=100e3, pulse_duration=8e-9, linewidth=10e9,
            wavelength=1064e-9, mfd=5e-6,
        ),
    )
    sim.run()
    report = sim.report()
    assert "V&V Requirements Compliance" not in report
    assert sim.check_requirements() == []


def test_unknown_amplifier_requirement_key_raises():
    """Per-stage keying under `amplifier` (or any typo) must fail loud."""
    amp = Amplifier.yb_5_130("AMP", pump_power=1.0, length=2.0)
    sim = Simulator(
        components=[amp],
        seed=Signal(
            average_power=1e-3, peak_power=125, pulse_energy=10e-9,
            rep_rate=100e3, pulse_duration=8e-9, linewidth=10e9,
            wavelength=1064e-9, mfd=5e-6,
        ),
        requirements={"amplifier": {"AMP-1": {"ase_ratio_dB_max": -20}}},
    )
    sim.run()
    with pytest.raises(ValueError, match="Unknown amplifier requirement key"):
        sim.check_requirements()
