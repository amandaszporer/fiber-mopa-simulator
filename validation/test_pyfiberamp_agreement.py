"""External cross-validation of our solver against PyFiberAmp.

PyFiberAmp (Rissanen, github.com/Jomiri/pyfiberamp, GPL-3.0) is a
production-quality fiber-amplifier simulator that wraps
`scipy.integrate.solve_bvp` (4th-order collocation, Kierzenka-Shampine
2001). Running their canonical Yb-amplifier example with our
`solve_steady_state_robust` and comparing the two outputs is the
strongest external sanity check available — agreement to within a few
percent demonstrates that our iterative-shooting BVP arrives at the
same physical solution as a different solver family on the same
problem.

This test is **conditionally enabled**:
  - Reference outputs live in `validation/pyfiberamp_canonical.json`.
  - When the reference's `expected_outputs` fields are null (no local
    PyFiberAmp run available), the test skips with a clear message.
  - To populate the reference, install PyFiberAmp
    (`pip install pyfiberamp`) and run
    `validation/generate_pyfiberamp_reference.py` (one-shot).

The agreement threshold is **5 %** (per the JSON file's
`comparison_tolerance_pct`). Tighter agreement would require aligning
σ-data tables and ASE-bin grid centroids exactly with PyFiberAmp's
internal choices — out of scope for a sanity benchmark.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ase.solver_steady import AmplifierGeometry  # noqa: E402
from ase.solver_time import solve_steady_state_robust  # noqa: E402
from ase.spectral_grid import SpectralGrid  # noqa: E402


REFERENCE_PATH = Path(__file__).parent / "pyfiberamp_canonical.json"


def _load_reference() -> dict:
    with REFERENCE_PATH.open() as f:
        return json.load(f)


def _reference_is_populated(ref: dict) -> bool:
    return all(
        ref["expected_outputs"][k] is not None
        for k in (
            "P_signal_out_W",
            "P_pump_residual_W",
            "ASE_fwd_total_W",
            "ASE_bwd_total_W",
        )
    )


def test_pyfiberamp_canonical_agreement():
    """Run our solver on PyFiberAmp's canonical Yb amplifier and compare."""
    ref = _load_reference()
    if not _reference_is_populated(ref):
        pytest.skip(
            "PyFiberAmp reference outputs not yet generated. Install "
            "pyfiberamp and run validation/generate_pyfiberamp_reference.py "
            "to enable this test."
        )

    cfg = ref["config"]
    tolerance_pct = ref["comparison_tolerance_pct"]
    expected = ref["expected_outputs"]

    # Build geometry matching the reference config.
    r_core = cfg["core_diameter_m"] / 2
    A_core = math.pi * r_core ** 2
    A_clad = math.pi * (cfg["clad_diameter_m"] / 2) ** 2
    gamma_pump = A_core / A_clad
    # Same N_Yb derivation as components.py.
    sigma_a_pump_at_976 = 2.5e-24
    N_Yb = cfg["clad_absorption_dB_per_m"] / (
        4.343 * sigma_a_pump_at_976 * gamma_pump
    )
    geom = AmplifierGeometry(
        fiber_length=cfg["fiber_length_m"],
        A_core=A_core,
        A_clad=A_clad,
        N_Yb=N_Yb,
        gamma_pump=gamma_pump,
    )
    grid = SpectralGrid.from_fiber(
        r_core=r_core,
        NA=cfg["core_na"],
        signal_wavelength=cfg["signal_wavelength_m"],
        pump_wavelength=cfg["pump_wavelength_m"],
    )

    result = solve_steady_state_robust(
        geom=geom, grid=grid,
        P_pump=cfg["pump_power_W"],
        P_signal_avg=cfg["signal_power_W"],
        ase_in_fwd=np.zeros(grid.n_bins),
        R_in=0.0, R_out=1e-4,
    )

    # Compare the four key outputs.
    checks = {
        "P_signal_out_W": result.signal_out,
        "P_pump_residual_W": result.pump_residual,
        "ASE_fwd_total_W": float(result.ase_fwd_out.sum()),
        "ASE_bwd_total_W": float(result.ase_bwd_in.sum()),
    }
    failures = []
    for key, measured in checks.items():
        expected_val = expected[key]
        if expected_val is None or expected_val == 0:
            continue
        rel_err = abs(measured - expected_val) / abs(expected_val)
        if rel_err * 100 > tolerance_pct:
            failures.append(
                f"{key}: ours={measured:.6g}  pyfiberamp={expected_val:.6g}  "
                f"rel_err={rel_err*100:.2f}% (tol={tolerance_pct:.1f}%)"
            )
    assert not failures, "PyFiberAmp disagreement:\n  " + "\n  ".join(failures)
