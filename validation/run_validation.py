"""
Validation harness: replicate published Yb-MOPA experiments in our simulator
and check that the predicted outputs are within the published tolerances.

Currently validates against:

    Świderski et al., Optica Applicata XXXVIII(4), 669-676 (2008).
    Single-stage Nufern 20/400 µm Yb LMA, 978 nm pump, 1063.91 nm diode seed,
    100 kHz rep rate, 11.4 W launched pump, multiple seed pulse widths.

Pass criterion (per the validation roadmap doc):
    avg-power agreement within 15% at the final stage.

Run from project root:

    .venv/bin/python validation/run_validation.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from framework import Simulator, SystemConfig

HERE = Path(__file__).resolve().parent
SWIDERSKI_CONFIG = HERE / "swiderski_2008.json"

PASS_TOLERANCE = 0.15


@dataclass(frozen=True)
class TestPoint:
    """One published data row to replicate."""
    label: str
    pulse_duration_ns: float
    seed_avg_mW: float          # diode-side seed avg power, before launch losses
    expected_output_W: float    # paper's reported avg output power
    expected_gain_dB: float     # paper's reported end-to-end amplification


SWIDERSKI_100KHZ_TESTS = [
    TestPoint(label="11 ns",  pulse_duration_ns=11.0,  seed_avg_mW=1.5, expected_output_W=2.04, expected_gain_dB=31.3),
    TestPoint(label="30 ns",  pulse_duration_ns=30.0,  seed_avg_mW=2.7, expected_output_W=2.44, expected_gain_dB=29.6),
    TestPoint(label="50 ns",  pulse_duration_ns=50.0,  seed_avg_mW=4.0, expected_output_W=2.60, expected_gain_dB=28.1),
    TestPoint(label="100 ns", pulse_duration_ns=100.0, seed_avg_mW=7.5, expected_output_W=2.97, expected_gain_dB=25.9),
]


@dataclass(frozen=True)
class RunResult:
    p_out_W: float
    gain_dB: float
    energy_balance_W: float        # signal_out + ASE + pump_residual; must be <= pump_in + signal_in
    pump_in_W: float
    pump_residual_W: float
    converged: bool
    solver_failed: bool
    iterations: int


def run_one(cfg_template: SystemConfig, tp: TestPoint) -> RunResult:
    """Override seed avg power + pulse duration, run sim, return diagnostics."""
    import math
    cfg = SystemConfig.from_dict(cfg_template.to_dict())
    cfg.seed["average_power"] = tp.seed_avg_mW * 1e-3
    cfg.seed["pulse_duration"] = tp.pulse_duration_ns * 1e-9

    sim = Simulator.from_config(cfg)
    sim.run(mode="steady")
    assert sim.final_state is not None  # set by run()

    amp = sim.amplifiers[0]
    info = amp.info
    p_out = sim.final_state.signal.average_power
    p_seed_diode = tp.seed_avg_mW * 1e-3
    gain_dB = 10 * math.log10(p_out / p_seed_diode) if p_out > 0 else float("-inf")
    energy_out = (
        info["P_signal_out"] + info["ase_power_out"]
        + info["ase_power_bwd"] + info["P_pump_residual"]
    )
    return RunResult(
        p_out_W=p_out,
        gain_dB=gain_dB,
        energy_balance_W=energy_out,
        pump_in_W=amp.pump_power,
        pump_residual_W=info["P_pump_residual"],
        converged=info["solver_converged"],
        solver_failed=info["solver_failed"],
        iterations=info["solver_iterations"],
    )


def fmt_pct(actual: float, expected: float) -> str:
    if expected == 0:
        return "n/a"
    delta = (actual - expected) / expected
    return f"{delta * 100:+.1f}%"


def classify(r: RunResult, tp: TestPoint) -> str:
    """Categorise the simulator's result against the published value."""
    if r.solver_failed:
        return "SOLVER FAIL (no stable steady state — results unreliable)"
    if not r.converged:
        return "SOLVER FAIL (did not converge)"
    energy_in = r.pump_in_W + tp.seed_avg_mW * 1e-3
    if r.energy_balance_W > energy_in * 1.01:
        return f"SOLVER FAIL (energy balance violated: {r.energy_balance_W:.2e} W out vs {energy_in:.2f} W in)"
    delta = abs(r.p_out_W - tp.expected_output_W) / tp.expected_output_W
    if delta <= PASS_TOLERANCE:
        return "PASS"
    return f"FAIL (Δ = {delta * 100:+.1f}%)"


def run_swiderski() -> bool:
    """Run all Świderski 100 kHz test points. Returns True if all pass."""
    print("=" * 88)
    print("  Validation: Świderski et al., Optica Applicata 38(4), 669 (2008)")
    print("  Setup: single-stage Nufern Yb LMA 20/400 µm, 978 nm pump @ 11.4 W")
    print("  Tests: 100 kHz rep rate, varying seed pulse width / avg power")
    print(f"  Pass tolerance: ±{PASS_TOLERANCE * 100:.0f}% on output avg power, plus solver convergence + energy balance")
    print("=" * 88)

    cfg_template = SystemConfig.load(SWIDERSKI_CONFIG)

    print()
    print(f"  {'Test':<8s} {'P_seed':>9s} {'P_out (paper)':>15s} {'P_out (sim)':>14s} {'iters':>6s} {'Result':>52s}")
    print(f"  {'-' * 8} {'-' * 9} {'-' * 15} {'-' * 14} {'-' * 6} {'-' * 52}")

    all_pass = True
    for tp in SWIDERSKI_100KHZ_TESTS:
        r = run_one(cfg_template, tp)
        verdict = classify(r, tp)
        all_pass = all_pass and verdict == "PASS"
        # Format the simulator output: scientific if absurdly large
        p_out_str = (
            f"{r.p_out_W:>10.2f} W" if r.p_out_W < 1e6 else f"{r.p_out_W:>10.2e} W"
        )
        print(
            f"  {tp.label:<8s} {tp.seed_avg_mW:>6.2f} mW "
            f"{tp.expected_output_W:>11.2f} W  "
            f"{p_out_str}  "
            f"{r.iterations:>5d}  {verdict:>52s}"
        )

    print()
    print("=" * 88)
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        print()
        print("  NOTE: 'SOLVER FAIL' results are not validation failures of the underlying")
        print("  physics — they indicate the iterative-shooting BVP solver in")
        print("  ase/solver_steady.py cannot find a physical steady state at the")
        print("  deep-unsaturated regime Świderski 2008 operates in (P_seed << P_sat ≈")
        print("  80 mW for this fiber, with 11.4 W pump). See validation/README.md.")
    print("=" * 88)
    return all_pass


def main() -> int:
    ok = run_swiderski()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
