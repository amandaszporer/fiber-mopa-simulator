"""One-shot helper to populate `pyfiberamp_canonical.json` with results
from a local PyFiberAmp install. Requires `pip install pyfiberamp`
(GPL-3.0). Not run on CI; not imported by anything else in the project.

Usage:
    pip install pyfiberamp
    .venv/bin/python validation/generate_pyfiberamp_reference.py

This writes the PyFiberAmp solution back into the reference JSON so that
`test_pyfiberamp_agreement.py` can run instead of skipping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    try:
        from pyfiberamp.fibers import YbDopedDoubleCladFiber  # type: ignore
        from pyfiberamp.steady_state import SteadyStateSimulation  # type: ignore
    except ImportError:
        print(
            "PyFiberAmp is not installed. Run `pip install pyfiberamp` "
            "first (GPL-3.0)."
        )
        return 1

    ref_path = Path(__file__).parent / "pyfiberamp_canonical.json"
    with ref_path.open() as f:
        ref = json.load(f)

    cfg = ref["config"]

    fiber = YbDopedDoubleCladFiber(
        length=cfg["fiber_length_m"],
        core_radius=cfg["core_diameter_m"] / 2,
        core_na=cfg["core_na"],
        background_loss=0,
        ion_number_density=None,  # let PyFiberAmp pick from cladding-abs
        background_loss_dB_per_m_pump=0,
        background_loss_dB_per_m_signal=0,
        ratio_of_core_and_cladding_diameters=(
            cfg["core_diameter_m"] / cfg["clad_diameter_m"]
        ),
        cladding_pump_absorption_dB_per_m=cfg["clad_absorption_dB_per_m"],
    )

    sim = SteadyStateSimulation()
    sim.fiber = fiber
    sim.add_cw_signal(
        wl=cfg["signal_wavelength_m"],
        power=cfg["signal_power_W"],
    )
    sim.add_forward_pump(
        wl=cfg["pump_wavelength_m"],
        power=cfg["pump_power_W"],
    )
    sim.add_ase(
        wl_start=cfg["lambda_min_m"],
        wl_end=cfg["lambda_max_m"],
        n_bins=cfg["n_ase_bins"],
    )
    sim.run()

    res = sim.steady_state_result   # PyFiberAmp's result object
    ref["expected_outputs"]["P_signal_out_W"] = float(
        res.powers.forward_signal_power[-1]
    )
    ref["expected_outputs"]["P_pump_residual_W"] = float(
        res.powers.forward_pump_power[-1]
    )
    ref["expected_outputs"]["ASE_fwd_total_W"] = float(
        res.powers.forward_ase_power[-1].sum()
    )
    ref["expected_outputs"]["ASE_bwd_total_W"] = float(
        res.powers.backward_ase_power[0].sum()
    )
    ref["expected_outputs"]["small_signal_gain_dB"] = None  # tutorial value

    with ref_path.open("w") as f:
        json.dump(ref, f, indent=2)

    print(f"Wrote PyFiberAmp reference outputs to {ref_path}")
    print(
        f"  P_signal_out = {ref['expected_outputs']['P_signal_out_W']*1e3:.3f} mW"
    )
    print(
        f"  P_pump_resid = {ref['expected_outputs']['P_pump_residual_W']*1e3:.3f} mW"
    )
    print(
        f"  ASE_fwd      = {ref['expected_outputs']['ASE_fwd_total_W']*1e6:.3f} µW"
    )
    print(
        f"  ASE_bwd      = {ref['expected_outputs']['ASE_bwd_total_W']*1e6:.3f} µW"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
