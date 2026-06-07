"""SI <-> human-readable unit conversion for GUI input widgets.

The engine stores everything in SI, but component spec sheets quote nm / ns /
kHz / GHz / um. A `Param.unit` string alone is ambiguous — `core_diameter` and
`fwhm` are both ``"m"`` yet live at different scales — so this module keeps a
per-parameter-name display table. Anything not in the table is shown in its SI
unit unchanged.
"""
from __future__ import annotations

# name -> (display unit label, scale).  display_value = si_value * scale
_DISPLAY: dict[str, tuple[str, float]] = {
    # wavelengths
    "wavelength": ("nm", 1e9),
    "signal_wavelength": ("nm", 1e9),
    "pump_wavelength": ("nm", 1e9),
    "center_wavelength": ("nm", 1e9),
    "fwhm": ("nm", 1e9),
    # transverse dimensions
    "core_diameter": ("µm", 1e6),
    "clad_diameter": ("µm", 1e6),
    "mfd": ("µm", 1e6),
    "output_mfd": ("µm", 1e6),
    # time / rate / linewidth
    "pulse_duration": ("ns", 1e9),
    "rep_rate": ("kHz", 1e-3),
    "linewidth": ("GHz", 1e-9),
    # seed power / energy
    "average_power": ("mW", 1e3),
    "peak_power": ("kW", 1e-3),
    "pulse_energy": ("nJ", 1e9),
}


def display_spec(name: str, si_unit: str = "") -> tuple[str, float]:
    """Return ``(display_unit_label, scale)`` for a parameter / field name.

    Falls back to the supplied SI unit and a scale of 1.0 for names that have
    no friendly conversion.
    """
    if name in _DISPLAY:
        return _DISPLAY[name]
    return (si_unit, 1.0)


def to_display(name: str, si_value: float, si_unit: str = "") -> float:
    """Convert an SI value to its display value."""
    return si_value * display_spec(name, si_unit)[1]


def to_si(name: str, display_value: float, si_unit: str = "") -> float:
    """Convert a display value back to SI."""
    return display_value / display_spec(name, si_unit)[1]
