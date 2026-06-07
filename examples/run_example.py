"""
Example: load a saved MOPA system, run it, tweak a parameter, rerun, save.

Run from the project root:

    .venv/bin/python examples/run_example.py

This script demonstrates the full save/load/edit cycle. It does not import
matplotlib; for plots, see `utils/plotting.py` and pass `--plots` to
`simulate.py` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python examples/run_example.py` from anywhere by adding the project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from framework import Simulator, SystemConfig  # noqa: E402

# Resolve paths relative to this script so it works from anywhere.
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "bgu_3stage_mopa.json"
MODIFIED = HERE / "bgu_3stage_mopa_modified.json"


def find_component(cfg: SystemConfig, name: str) -> dict:
    """Return the dict for the component with the given `name`. Raises KeyError."""
    for c in cfg.components:
        if c.get("name") == name:
            return c
    raise KeyError(f"No component named {name!r} in this config")


def main() -> None:
    # 1. Load the saved system
    cfg = SystemConfig.load(ORIGINAL)
    print(f"Loaded {cfg.name}")
    print(f"  ({len(cfg.components)} components, "
          f"version {cfg.metadata.get('version', '?')})")

    # 2. Run the baseline simulation
    sim = Simulator.from_config(cfg)
    sim.run()
    out = sim.final_state.signal
    print(f"\nBaseline output: {out.average_power:.2f} W avg, "
          f"{out.peak_power/1e3:.1f} kW peak, "
          f"{out.pulse_energy*1e6:.2f} uJ/pulse")
    print(f"Baseline forward ASE: {sim.final_state.ase.total_fwd()*1e3:.2f} mW")

    # 3. Tweak a parameter — bump stage 2 pump power by 50%
    amp2 = find_component(cfg, "AMP-2")
    old_pump = amp2["pump_power"]
    amp2["pump_power"] = old_pump * 1.5
    print(f"\nTweaked AMP-2 pump_power: {old_pump} W -> {amp2['pump_power']} W")

    # 4. Rerun with the modified config
    sim2 = Simulator.from_config(cfg)
    sim2.run()
    out2 = sim2.final_state.signal
    print(f"\nNew output: {out2.average_power:.2f} W avg, "
          f"{out2.peak_power/1e3:.1f} kW peak, "
          f"{out2.pulse_energy*1e6:.2f} uJ/pulse")
    print(f"New forward ASE: {sim2.final_state.ase.total_fwd()*1e3:.2f} mW")

    delta_W = out2.average_power - out.average_power
    print(f"\nDelta avg power: {delta_W:+.2f} W "
          f"({delta_W/out.average_power*100:+.1f}%)")

    # 5. Save the modified config to a new file (no overwrite of the original)
    cfg.metadata = {**cfg.metadata,
                    "version": "1.1-modified",
                    "modified_note": "AMP-2 pump_power scaled 1.5x"}
    cfg.save(MODIFIED)
    print(f"\nSaved modified config to {MODIFIED.relative_to(Path.cwd())}")
    print(f"  Original ({ORIGINAL.stat().st_size} bytes) "
          f"and modified ({MODIFIED.stat().st_size} bytes) "
          f"diff only in AMP-2 pump_power and metadata.")


if __name__ == "__main__":
    main()
