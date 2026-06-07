"""
Yb-doped fiber MOPA simulator — CLI driver.

Loads a system definition from JSON, runs it through the framework's
`Simulator`, prints the report, and optionally writes the standard 4 PNG
plots. The default config is `examples/bgu_3stage_mopa.json`; pass
`--config path/to/your.json` to run a different system.

Examples:
    python simulate.py
    python simulate.py --plots
    python simulate.py --time-dependent --plots
    python simulate.py --config examples/my_system.json
    python simulate.py --lab     # interactive lab-continuation mode
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import replace
from pathlib import Path

from ase.state import OpticalState
from components import SHAPE_FACTOR, Amplifier, Signal
from framework import Simulator, SystemConfig

DEFAULT_CONFIG = Path(__file__).resolve().parent / "examples" / "bgu_3stage_mopa.json"
REPORT_ROOT = Path(__file__).resolve().parent / "report"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
    p.add_argument("--config", "-c", type=Path, default=DEFAULT_CONFIG,
                   help=f"Path to the SystemConfig JSON (default: {DEFAULT_CONFIG.name}).")
    p.add_argument("--steady", action="store_true",
                   help="Force Mode A (steady-state BVP) regardless of rep rate. "
                        "By default the simulator auto-dispatches Mode B (which "
                        "delegates to Mode A at high rep and uses the Level 5 "
                        "B1+B2 cycle at lower rep).")
    p.add_argument("--time-dependent", action="store_true",
                   help="(Default behaviour, kept for backwards compatibility.) "
                        "Use Mode B with auto-dispatch.")
    p.add_argument("--force-b2", action="store_true",
                   help="Force the B1+B2 (Level 5) path even at high rep — useful "
                        "for studying pulse-shape distortion. Overrides --steady.")
    p.add_argument("--plots", action="store_true",
                   help="Render the 4 standard PNG plots (matplotlib required).")
    p.add_argument("--lab", action="store_true",
                   help="Interactive lab-continuation mode: enter measured values "
                        "after a real amplifier stage and continue the simulation.")
    p.add_argument("--output-dir", "-o", type=Path, default=None,
                   help="Where to write summary.txt and (with --plots) the figures. "
                        "Default: report/<timestamp>_full/")
    return p.parse_args(argv)


def _output_dir(args: argparse.Namespace, suffix: str) -> Path:
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = REPORT_ROOT / f"{timestamp}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ── Lab continuation mode ─────────────────────────────────────────────

def _prompt_float(label: str, units: str, default: float | None = None) -> float:
    prompt_str = (f"  {label} [{units}] (Enter = {default}): "
                  if default is not None
                  else f"  {label} [{units}]: ")
    while True:
        raw = input(prompt_str).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = float(raw)
            if val <= 0:
                print("    Please enter a positive number.")
                continue
            return val
        except ValueError:
            print("    Invalid input. Please enter a number.")


def run_lab_mode(args: argparse.Namespace) -> tuple[Simulator, int]:
    """Interactive: prompt for measurements, then continue from that stage onward.

    Returns (sim, completed_stage_number). The sim's `results` only contain the
    stages run after the measurement.
    """
    print()
    print("=" * 60)
    print("  Lab Continuation Mode")
    print("=" * 60)
    print()

    # Pick stage
    while True:
        raw = input("Which amplifier stage did you just complete? (1/2): ").strip()
        if raw in ("1", "2"):
            stage_number = int(raw)
            break
        print("  Please enter 1 or 2.")

    print()
    print(f"Enter measured values after Stage {stage_number}:")
    avg_mW = _prompt_float("Average power", "mW")
    dur_ns = _prompt_float("Pulse duration", "ns", default=8.0)
    lw_GHz = _prompt_float("Linewidth", "GHz", default=10.0)
    rep_kHz = _prompt_float("Rep rate", "kHz", default=100.0)
    avg = avg_mW * 1e-3
    dur = dur_ns * 1e-9
    lw = lw_GHz * 1e9
    rep = rep_kHz * 1e3
    energy = avg / rep
    peak = energy / (dur * SHAPE_FACTOR)

    measured_signal = Signal(
        average_power=avg, peak_power=peak, pulse_energy=energy,
        rep_rate=rep, pulse_duration=dur, linewidth=lw,
        wavelength=1064e-9, mfd=5e-6,
    )

    # Pump overrides
    print()
    print("Pump powers for remaining stages:")
    pump_overrides: dict[int, float] = {}
    default_pumps = {2: 9.0, 3: 100.0}
    for s in range(stage_number + 1, 4):
        p = _prompt_float(f"Stage {s} pump power", "W", default=default_pumps[s])
        pump_overrides[s] = p

    print()
    print(f"Constructed signal: {avg*1e3:.2f} mW avg, {peak/1e3:.2f} kW peak, "
          f"{energy*1e9:.2f} nJ, {lw/1e9:.2f} GHz")

    # Load the saved config and run through the just-completed stage to get the
    # theoretical ASE spectrum entering the next stage.
    cfg = SystemConfig.load(args.config)
    sim_full = Simulator.from_config(cfg)
    lab_mode = (
        "full" if args.force_b2
        else "steady" if args.steady
        else "time-dependent"
    )
    sim_full.run(mode=lab_mode)

    # Find the index of the stage_number-th amplifier
    amp_indices = [i for i, c in enumerate(sim_full.components)
                   if isinstance(c, Amplifier)]
    completed_idx = amp_indices[stage_number - 1]
    handoff_state = sim_full.results[completed_idx].state_out

    # Replace the simulated signal with the measured one but keep the spectral ASE
    completed_amp = sim_full.components[completed_idx]
    measured_signal = replace(measured_signal, mfd=completed_amp.fiber_mfd)
    state = OpticalState(signal=measured_signal, ase=handoff_state.ase)

    # Slice off the completed prefix; clone the remaining components from cfg
    # and apply the pump overrides.
    cfg2 = SystemConfig.load(args.config)
    remaining_dicts = cfg2.components[completed_idx + 1:]
    for d in remaining_dicts:
        if d.get("type") == "Amplifier":
            stage = next((s for s in pump_overrides
                          if d.get("name", "").endswith(str(s))), None)
            if stage is not None:
                d["pump_power"] = pump_overrides[stage]

    # Build a new simulator from just the remaining components, but seed it
    # by injecting our measured state directly.
    from components import component_from_dict
    remaining_components = [component_from_dict(d) for d in remaining_dicts]

    sim = Simulator(remaining_components, measured_signal)
    # Manually run with our injected state (bypassing run()'s seeding logic)
    sim.results = []
    cur = state
    mode = (
        "full" if args.force_b2
        else "steady" if args.steady
        else "time-dependent"
    )
    for c in remaining_components:
        state_in = cur
        if isinstance(c, Amplifier):
            cur = c.propagate(cur, mode=mode)
        else:
            cur = c.propagate(cur)
        from framework import StageResult
        sim.results.append(StageResult(c, state_in, cur))
    sim.final_state = cur

    return sim, stage_number


# ── Main ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.force_b2:
        mode = "full"
    elif args.steady:
        mode = "steady"
    else:
        # Default and --time-dependent both resolve to auto-dispatch.
        mode = "time-dependent"

    if args.lab:
        sim, stage_number = run_lab_mode(args)
        out_dir = _output_dir(args, f"lab_stage{stage_number}")
        report_text = sim.report(mode_label=mode + " (lab)")
    else:
        cfg = SystemConfig.load(args.config)
        print(f"Loaded {cfg.name} ({len(cfg.components)} components)")
        sim = Simulator.from_config(cfg)
        sim.run(mode=mode)
        out_dir = _output_dir(args, "full")
        report_text = sim.report(mode_label=mode)

    print(report_text)
    summary = out_dir / "summary.txt"
    summary.write_text(report_text)
    print(f"\nSummary saved to {summary}")

    if args.plots:
        from utils import plotting
        paths = plotting.plot_all(sim, out_dir)
        print(f"Plots saved to {out_dir}/ ({len(paths)} files)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
