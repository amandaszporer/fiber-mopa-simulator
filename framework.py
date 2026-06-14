"""
SystemConfig + Simulator — the orchestration layer above `components.py`.

A `SystemConfig` is a JSON-serialisable description of a complete MOPA chain:
seed parameters, an ordered list of components, and metadata. It can be saved
to disk, loaded later, and instantiated into a runnable `Simulator` with
`Simulator.from_config(cfg)`. The `Simulator` propagates the seed `Signal`
through the chain, stores per-stage `StageResult`s, and produces a text
`report()` summarising power, gain, ASE, and nonlinear margins at every step.

The format of saved JSON files matches the example in the project's task spec:
SI units, one component per dict entry with a `type` field that maps into
`COMPONENT_REGISTRY`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ase.state import OpticalState
from components import (
    SHAPE_FACTOR,
    Amplifier,
    Component,
    Signal,
    component_from_dict,
)

# Speed of light is needed for the spectral-width calc in the V&V table.
_C = 3e8


# ── Signal helpers ───────────────────────────────────────────────────

_SIGNAL_FIELDS_REQUIRED = {
    "average_power", "rep_rate", "pulse_duration", "linewidth", "wavelength",
}
_SIGNAL_FIELDS_OPTIONAL = {"peak_power", "pulse_energy", "mfd"}
_SIGNAL_FIELDS_ALL = _SIGNAL_FIELDS_REQUIRED | _SIGNAL_FIELDS_OPTIONAL


def signal_to_dict(sig: Signal) -> dict:
    """Serialise a Signal to the seed-dict shape used in SystemConfig.

    Only the independent quantities are kept; peak_power and pulse_energy
    are derived from average_power, rep_rate, and pulse_duration on rebuild
    (so the JSON stays minimal and edits to one field reliably propagate).
    """
    return {
        "average_power": sig.average_power,
        "rep_rate": sig.rep_rate,
        "pulse_duration": sig.pulse_duration,
        "linewidth": sig.linewidth,
        "wavelength": sig.wavelength,
        "mfd": sig.mfd,
    }


def signal_from_dict(d: dict) -> Signal:
    """Reconstruct a Signal from a seed dict.

    `peak_power`, `pulse_energy`, `mfd` may be omitted — they're derived
    (peak/energy from avg/rep_rate/pulse_duration; mfd defaults to 5 µm,
    which is then reset by the first amplifier's fiber MFD anyway).
    """
    extra = set(d) - _SIGNAL_FIELDS_ALL
    if extra:
        raise ValueError(
            f"Unknown seed field(s): {sorted(extra)}. Allowed: {sorted(_SIGNAL_FIELDS_ALL)}"
        )
    missing = _SIGNAL_FIELDS_REQUIRED - set(d)
    if missing:
        raise ValueError(f"Seed missing required field(s): {sorted(missing)}")

    avg = float(d["average_power"])
    rep = float(d["rep_rate"])
    dur = float(d["pulse_duration"])
    energy = d.get("pulse_energy", avg / rep if rep > 0 else 0.0)
    peak = d.get("peak_power", energy / (dur * SHAPE_FACTOR) if dur > 0 else 0.0)
    return Signal(
        average_power=avg,
        peak_power=float(peak),
        pulse_energy=float(energy),
        rep_rate=rep,
        pulse_duration=dur,
        linewidth=float(d["linewidth"]),
        wavelength=float(d["wavelength"]),
        mfd=float(d.get("mfd", 5e-6)),
    )


# ── SystemConfig ─────────────────────────────────────────────────────

@dataclass
class SystemConfig:
    """A complete saveable system description.

    `components` is a list of dicts (one per component) with at minimum a
    `type` field naming a class in COMPONENT_REGISTRY and a `name` field. The
    remaining entries are the component's declared parameters.

    `requirements` is an optional dict of V&V acceptance criteria. When
    populated, `Simulator` evaluates each criterion and produces a
    pass/fail compliance table at the end of the report. When empty, the
    simulator just reports the propagated output (no pass/fail).
    """

    name: str
    description: str = ""
    seed: dict = field(default_factory=dict)
    components: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    requirements: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "seed": dict(self.seed),
            "components": [dict(c) for c in self.components],
            "metadata": dict(self.metadata),
            "requirements": dict(self.requirements),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SystemConfig":
        return cls(
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            seed=dict(d.get("seed", {})),
            components=[dict(c) for c in d.get("components", [])],
            metadata=dict(d.get("metadata", {})),
            requirements=dict(d.get("requirements", {})),
        )

    def save(self, path: str | Path) -> None:
        """Write the config to a JSON file (UTF-8, indent=2)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=_json_default)

    @classmethod
    def load(cls, path: str | Path) -> "SystemConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def build(self) -> tuple[list[Component], Signal]:
        """Instantiate the component chain and seed Signal."""
        seed = signal_from_dict(self.seed)
        components = [component_from_dict(c) for c in self.components]
        return components, seed

    @classmethod
    def from_simulator(
        cls,
        sim: "Simulator",
        *,
        name: str = "",
        description: str = "",
        metadata: Optional[dict] = None,
        requirements: Optional[dict] = None,
    ) -> "SystemConfig":
        """Round-trip helper: build a SystemConfig that reproduces `sim`."""
        return cls(
            name=name,
            description=description,
            seed=signal_to_dict(sim.seed),
            components=[c.to_dict() for c in sim.components],
            metadata=metadata or {},
            requirements=requirements if requirements is not None else dict(sim.requirements),
        )


def _json_default(obj: Any) -> Any:
    """Fall-back encoder for numpy scalars (avoid surprises on save)."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
    except ImportError:
        pass
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


# ── Simulator ────────────────────────────────────────────────────────

@dataclass
class StageResult:
    """One entry in `Simulator.results` — input/output state at one component."""
    component: Component
    state_in: OpticalState
    state_out: OpticalState


class Simulator:
    """Runs a configured chain and exposes per-stage results plus a report."""

    components: list[Component]
    seed: Signal
    results: list[StageResult]
    final_state: Optional[OpticalState]
    requirements: dict

    def __init__(
        self,
        components: list[Component],
        seed: Signal,
        requirements: Optional[dict] = None,
    ) -> None:
        self.components = list(components)
        self.seed = seed
        self.results = []
        self.final_state = None
        self.requirements = dict(requirements) if requirements else {}

    @classmethod
    def from_config(cls, config: SystemConfig) -> "Simulator":
        components, seed = config.build()
        return cls(components, seed, requirements=config.requirements)

    @property
    def amplifiers(self) -> list[Amplifier]:
        """Convenience: only the Amplifier stages, in order."""
        return [c for c in self.components if isinstance(c, Amplifier)]

    def run(self, mode: str = "time-dependent") -> OpticalState:
        """Propagate the seed through every component. `mode` is forwarded to
        each Amplifier.

        Default is `"time-dependent"` (auto-dispatch): bit-identical to
        Mode A at high rep, full Level 5 B1+B2 at lower rep rates.
        Pass `"steady"` to force Mode A, or `"full"` to force B1+B2."""
        state = OpticalState(signal=self.seed, ase=None)
        # If the first amplifier exists, seed a zero AseState on its grid so
        # passive components downstream can attenuate it cleanly.
        first_amp = next((c for c in self.components if isinstance(c, Amplifier)), None)
        if first_amp is not None:
            from ase.state import AseState
            state = OpticalState(signal=self.seed, ase=AseState.zero(first_amp.grid))

        self.results = []
        for comp in self.components:
            state_in = state
            if isinstance(comp, Amplifier):
                state = comp.propagate(state, mode=mode)
            else:
                state = comp.propagate(state)
            self.results.append(StageResult(comp, state_in, state))
        self.final_state = state
        return state

    def report(self, *, mode_label: str = "steady") -> str:
        """Text report mirroring the legacy `simulate.py` output."""
        if self.final_state is None:
            raise RuntimeError("Call run() before report().")

        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("  Yb-Doped Fiber MOPA Simulator")
        lines.append(f"  Solver mode: {mode_label}")
        lines.append("=" * 72)
        lines.append("")

        sig0 = self.seed
        lines.append(
            f"  Seed: {_fmt_power(sig0.average_power)} avg, "
            f"{_fmt_peak(sig0.peak_power)} peak, "
            f"{_fmt_energy(sig0.pulse_energy)}, "
            f"{_fmt_lw(sig0.linewidth)} linewidth"
        )
        lines.append("")

        for r in self.results:
            sig = r.state_out.signal
            lines.append(f"  [{r.component.name}]")
            lines.append(
                f"    Avg: {_fmt_power(sig.average_power):>12s}   "
                f"Peak: {_fmt_peak(sig.peak_power):>12s}   "
                f"Energy: {_fmt_energy(sig.pulse_energy):>10s}   "
                f"LW: {_fmt_lw(sig.linewidth):>10s}"
            )
            if r.state_out.ase is not None:
                tot = r.state_out.ase.total_fwd()
                peak_wl = r.state_out.ase.peak_fwd_wavelength()
                peak_str = (
                    f"{peak_wl*1e9:.1f} nm" if not math.isnan(peak_wl) else "—"
                )
                if r.state_out.ase.under_resolved:
                    tag = "   [UNDER-RESOLVED: increase num_segments]"
                elif r.state_out.ase.solver_failed:
                    tag = "   [SOLVER ISSUE: no stable steady state]"
                else:
                    tag = ""
                lines.append(
                    f"    ASE fwd: {_fmt_power(tot):>12s}  peak {peak_str}{tag}"
                )
            if isinstance(r.component, Amplifier):
                info = r.component.info
                lines.append(
                    f"    Gain: {info['gain_dB']:.1f} dB  "
                    f"({info['gain_linear']:.1f}x)"
                    f"{'  [CAPPED]' if info['gain_capped'] else ''}"
                    f"{_health_tag(info)}"
                )
                lines.append(
                    f"    Pump absorbed: {info['pump_absorption_pct']:.0f}%  "
                    f"(residual {info['P_pump_residual']*1e3:.1f} mW)"
                )
                lines.append(
                    f"    SBS: {info['sbs_ratio']:.3f}  [{_flag(info['sbs_safe'])}]   "
                    f"SRS: {info['srs_ratio']:.4f}  [{_flag(info['srs_safe'])}]"
                )
                lines.append(
                    f"    ASE: {info['ase_ratio_dB']:.1f} dB  "
                    f"[{_flag(info['ase_safe'])}]   "
                    f"solver: {info['solver_iterations']} iters"
                    f" ({'conv' if info['solver_converged'] else 'NOT CONV'})"
                )
            lines.append("")

        # Footer + V&V
        lines.append("=" * 72)
        sig = self.final_state.signal
        lines.append(
            f"  Output: {_fmt_power(sig.average_power)} avg, "
            f"{_fmt_peak(sig.peak_power)} peak, "
            f"{_fmt_energy(sig.pulse_energy)}"
        )
        if self.final_state.ase is not None:
            lines.append(
                f"  Output ASE (forward): "
                f"{_fmt_power(self.final_state.ase.total_fwd())} integrated"
            )
        lines.append("=" * 72)
        lines.append("")

        for line in self._compliance_table():
            lines.append(f"  {line}")

        return "\n".join(lines)

    # ── V&V compliance ──────────────────────────────────────────────

    def check_requirements(self) -> list[tuple[str, str, str, bool]]:
        """Evaluate the criteria in `self.requirements` against the final state.

        Returns one ``(parameter, actual_str, criterion_str, passed)`` tuple
        per criterion present. Empty `requirements` → empty list (and the
        report omits the compliance table entirely).

        Schema — every top-level key is optional. All values are in SI units
        to stay consistent with the rest of the JSON config:

        ``wavelength``      ``{"target": float, "tolerance": float}`` [m]
        ``spectral_width``  ``{"max": float}`` [m]
        ``rep_rate``        ``{"min": float, "max": float}`` [Hz]
        ``pulse_duration``  ``{"min": float, "max": float}`` [s]
        ``avg_power``       ``{"min": float, "max": float}`` [W]
        ``peak_power``      ``{"min": float, "max": float}`` [W]
        ``amplifier``       per-amp gates (uniform across every Amplifier);
                            see _amp_requirements.
        """
        if self.final_state is None:
            raise RuntimeError("Call run() before check_requirements().")

        results: list[tuple[str, str, str, bool]] = []
        if not self.requirements:
            return results

        signal = self.final_state.signal
        req = self.requirements

        if "wavelength" in req:
            spec = req["wavelength"]
            target_m = float(spec["target"])
            tol_m = float(spec["tolerance"])
            wl_nm = signal.wavelength * 1e9
            results.append((
                "Wavelength", f"{wl_nm:.1f} nm",
                f"{target_m*1e9:g} +/- {tol_m*1e9:g} nm",
                abs(signal.wavelength - target_m) < tol_m,
            ))

        if "spectral_width" in req:
            dlam_m = (signal.wavelength ** 2 / _C) * signal.linewidth
            results.append(_fmt_range_row(
                "Spectral width", f"{dlam_m*1e9:.4f} nm", "nm",
                dlam_m, req["spectral_width"],
                display_scale=1e9,
            ))

        if "rep_rate" in req:
            rep_kHz = signal.rep_rate / 1e3
            actual_str = (
                f"{rep_kHz:.0f} kHz" if rep_kHz >= 1
                else f"{signal.rep_rate:.0f} Hz"
            )
            results.append(_fmt_range_row(
                "Rep rate", actual_str, "kHz",
                signal.rep_rate, req["rep_rate"],
                display_scale=1e-3,
            ))

        if "pulse_duration" in req:
            results.append(_fmt_range_row(
                "Pulse width", f"{signal.pulse_duration*1e9:.0f} ns", "ns",
                signal.pulse_duration, req["pulse_duration"],
                display_scale=1e9,
            ))

        if "avg_power" in req:
            results.append(_fmt_range_row(
                "Avg power", f"{signal.average_power:.1f} W", "W",
                signal.average_power, req["avg_power"],
            ))

        if "peak_power" in req:
            results.append(_fmt_range_row(
                "Peak power", f"{signal.peak_power/1e3:.1f} kW", "kW",
                signal.peak_power, req["peak_power"],
                display_scale=1e-3,
            ))

        amp_req = req.get("amplifier") or {}
        if amp_req:
            _validate_amp_requirement_keys(amp_req)
            for amp in self.amplifiers:
                results.extend(self._amp_requirements(amp, amp_req))

        return results

    @staticmethod
    def _amp_requirements(
        amp: Amplifier, amp_req: dict,
    ) -> list[tuple[str, str, str, bool]]:
        info = amp.info
        label = amp.name or "amp"
        rows: list[tuple[str, str, str, bool]] = []

        if "ase_ratio_dB_max" in amp_req:
            ceiling = float(amp_req["ase_ratio_dB_max"])
            rows.append((
                f"ASE {label}", f"{info['ase_ratio_dB']:.1f} dB",
                f"< {ceiling:g} dB", info["ase_ratio_dB"] < ceiling,
            ))
        if "sbs_ratio_max" in amp_req:
            ceiling = float(amp_req["sbs_ratio_max"])
            rows.append((
                f"SBS {label}", f"{info['sbs_ratio']:.3f}",
                f"< {ceiling:g}", info["sbs_ratio"] < ceiling,
            ))
        if "srs_ratio_max" in amp_req:
            ceiling = float(amp_req["srs_ratio_max"])
            rows.append((
                f"SRS {label}", f"{info['srs_ratio']:.4f}",
                f"< {ceiling:g}", info["srs_ratio"] < ceiling,
            ))
        if amp_req.get("solver_stable"):
            ok = info["solver_converged"] and not info["solver_failed"]
            rows.append((
                f"Solver {label}",
                "ok" if ok else "ISSUE",
                "ok", ok,
            ))
        return rows

    def _compliance_table(self) -> list[str]:
        results = self.check_requirements()
        if not results:
            return []
        all_pass = all(passed for _, _, _, passed in results)
        lines: list[str] = []
        lines.append("V&V Requirements Compliance")
        lines.append("=" * 60)
        lines.append(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
        lines.append("")
        lines.append(f"  {'Parameter':<20s} {'Actual':<16s} {'Criterion':<20s} Result")
        lines.append(f"  {'-'*20} {'-'*16} {'-'*20} {'-'*6}")
        for param, actual, criterion, passed in results:
            tag = "PASS" if passed else "FAIL"
            lines.append(f"  {param:<20s} {actual:<16s} {criterion:<20s} {tag}")
        lines.append("=" * 60)
        return lines


# ── Formatting helpers used by report() ──────────────────────────────

def _fmt_power(p: float) -> str:
    if p >= 1: return f"{p:.2f} W"
    if p >= 1e-3: return f"{p*1e3:.2f} mW"
    return f"{p*1e6:.2f} uW"


def _fmt_peak(p: float) -> str:
    if p >= 1e3: return f"{p/1e3:.2f} kW"
    return f"{p:.2f} W"


def _fmt_energy(e: float) -> str:
    if e >= 1e-3: return f"{e*1e3:.2f} mJ"
    if e >= 1e-6: return f"{e*1e6:.2f} uJ"
    return f"{e*1e9:.2f} nJ"


def _fmt_lw(lw: float) -> str:
    if lw >= 1e9: return f"{lw/1e9:.2f} GHz"
    if lw >= 1e6: return f"{lw/1e6:.1f} MHz"
    return f"{lw:.0f} Hz"


def _flag(safe: bool) -> str:
    return "SAFE" if safe else "DANGER"


def _health_tag(info: dict) -> str:
    """Compact `[E:.. R:.. path=..]` decoration for an amplifier's gain line.

    Three signals at a glance:
      • Energy status (OK / WARNING / VIOLATION) plus the consumed/delivered
        ratio in the warning + violation cases.
      • Operating regime (AMPLIFIER / MIXED / SFS / HIGH_GAIN).
      • Which solver layer produced the result (direct / homotopy /
        time_marching_arbiter / all_failed); homotopy carries the step count.

    Returns "" for healthy direct results so the report stays uncluttered.
    """
    status = info.get("energy_status", "ok")
    regime = info.get("regime", "amplifier")
    path = info.get("solver_path_used", "direct")
    homotopy_steps = info.get("homotopy_steps_used", 0)
    ratio = info.get("energy_residual_ratio", 0.0)

    # Suppress the tag in the common, healthy case to keep the report tidy.
    if status == "ok" and regime in ("amplifier", "mixed") and path == "direct":
        return ""

    e_part = "OK"
    if status == "warning":
        e_part = f"WARNING ×{1 + ratio:.2f}"
    elif status == "violation":
        e_part = f"VIOLATION ×{1 + ratio:.2f}"

    path_part = path
    if path == "homotopy":
        path_part = f"homotopy({homotopy_steps})"

    return f"  [E:{e_part}  R:{regime.upper()}  path={path_part}]"


_AMP_REQ_KEYS = frozenset({
    "ase_ratio_dB_max",
    "sbs_ratio_max",
    "srs_ratio_max",
    "solver_stable",
})


def _validate_amp_requirement_keys(amp_req: dict) -> None:
    """Fail loud if an `amplifier` requirements dict has unknown keys.

    The block is applied uniformly to every Amplifier — per-stage keying
    (e.g. `{"AMP-1": {...}}`) is not supported and would silently produce no
    rows. Raise rather than let that slip past.
    """
    unknown = set(amp_req) - _AMP_REQ_KEYS
    if unknown:
        raise ValueError(
            f"Unknown amplifier requirement key(s): {sorted(unknown)}. "
            f"Allowed: {sorted(_AMP_REQ_KEYS)}. (Per-stage keying is not "
            f"supported; the block applies uniformly to every Amplifier.)"
        )


def _fmt_range_row(
    param: str,
    actual_str: str,
    display_unit: str,
    actual: float,
    spec: dict,
    *,
    display_scale: float = 1.0,
) -> tuple[str, str, str, bool]:
    """Build a compliance row for a {min, max} (either bound optional) spec.

    `display_scale` is applied to the criterion bounds for human-readable
    units (e.g. rep_rate stored in Hz, displayed in kHz → 1e-3).
    """
    lo = spec.get("min")
    hi = spec.get("max")
    passed = (lo is None or actual >= lo) and (hi is None or actual <= hi)

    lo_d = None if lo is None else lo * display_scale
    hi_d = None if hi is None else hi * display_scale
    if lo_d is not None and hi_d is not None:
        criterion = f"{lo_d:g} - {hi_d:g} {display_unit}"
    elif lo_d is not None:
        criterion = f">= {lo_d:g} {display_unit}"
    elif hi_d is not None:
        criterion = f"< {hi_d:g} {display_unit}"
    else:
        criterion = "(no bound)"
    return param, actual_str, criterion, passed


__all__ = [
    "SystemConfig",
    "Simulator",
    "StageResult",
    "signal_to_dict",
    "signal_from_dict",
]
