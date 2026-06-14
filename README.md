# Yb-Doped Fiber MOPA Simulator

Simulation of a 3-stage Ytterbium-doped fiber Master Oscillator Power Amplifier (MOPA) system for generating high-power pulsed laser light at 1064 nm.

Built for BGU engineering project **p-2026-158**.

## Target Specifications

| Parameter       | Range              |
|-----------------|--------------------|
| Average power   | 22.5 -- 70 W       |
| Peak power      | 15 -- 50 kW        |
| Pulse duration  | 4 -- 8 ns          |
| Wavelength      | 1064 nm            |
| Rep rate        | 10 Hz -- 100 kHz   |


## Quick Start

**Prerequisites:** Python 3.10+ with `numpy`, `scipy`, and `matplotlib` (the BVP solver and spectral arrays need numpy/scipy; matplotlib renders the report plots). Install via:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
python simulate.py                                     # default config, Mode B auto-dispatch (Mode A at high rep, Level 5 B1+B2 at low rep)
python simulate.py --plots                             # also write the diagnostic PNGs
python simulate.py --steady                            # force Mode A (steady-state BVP) regardless of rep rate
python simulate.py --force-b2 --plots                  # force B1+B2 even at high rep — adds pulse_shape.png
python simulate.py --config examples/my_system.json    # run a different system file
python simulate.py -o some/dir                         # choose where summary.txt + figures go
python simulate.py --lab                               # interactive lab continuation mode
python examples/run_example.py                         # round-trip load -> tweak -> rerun -> save demo
python -m pytest ase/tests/ -v                         # ASE solver validation suite
python -m pytest validation/ -v                        # PyFiberAmp / Świderski cross-validation suite
```

The default config is `examples/bgu_3stage_mopa.json`. The terminal prints the report; without `-o`, output is written to a timestamped subdirectory under `report/` (`summary.txt` always; with `--plots`, five PNGs: power evolution, amplifier details, ASE spectra, BPF spectra, nonlinear margins — plus `pulse_shape.png` when `--force-b2` is used).

### Graphical interface

A Streamlit front-end wraps the same engine (no engine code is duplicated — it calls `framework.Simulator` / `SystemConfig`, the `components.py` registry, and `utils/plotting.py`). Launch from the project root:

```bash
streamlit run app.py
```

Three pages: **Home** (overview), **Builder** (assemble/edit a system visually), and **Run & Results** (run a config and view the plots). Requires the GUI dependencies in `requirements.txt` (`streamlit`, `streamlit-sortables`).

In lab mode (`--lab`), enter measured values after completing a real amplifier stage. The simulator continues from that point, letting you compare predicted vs measured performance and plan pump settings for remaining stages. The forward ASE spectrum at the handoff point is taken from a fresh theoretical run through the completed stages, since ASE cannot be measured directly.


## Project Structure

```
components.py                Signal, passive components, BandpassFilter, Amplifier (delegates ASE physics to ase/)
framework.py                 SystemConfig (JSON load/save) + Simulator (runs the chain, builds the report + V&V)
simulate.py                  CLI driver: loads a SystemConfig, runs it, writes summary.txt and (with --plots) figures
app.py                       Streamlit GUI entry point (streamlit run app.py) — thin front-end over the engine
gui/                         Streamlit pages + widgets (home, builder, results, state, units)
utils/plotting.py            Matplotlib report figures (power evolution, amplifier details, ASE/BPF spectra, NL margins, pulse shape)
examples/                    Saved system configs (JSON) + run_example.py round-trip demo
ase/                         Spectrally-resolved bidirectional ASE solver package
ase/data/                    Yb cross-section CSVs (measured Melkumov aluminosilicate dataset + raw pm² provenance)
ase/tests/                   pytest validation suite (python -m pytest ase/tests/ -v)
validation/                  External cross-validation (PyFiberAmp, Świderski 2008) + its own pytest suite + README
docs/physics_spec.md              Physics reference (gain modeling, cross-sections, fiber params, NL effects)
docs/computation_walkthrough.md   Step-by-step guide to gain & nonlinear calculations (no programming knowledge required)
docs/ase.md                       ASE documentation — specification (Part I) + walkthrough (Part II), one file
docs/CHANGELOG.md                 Dated record of notable changes (this repo is the cross-session source of truth)
report/                      Generated outputs in timestamped subdirectories (git-ignored)
```


## How It Works

### Signal Propagation

`Signal` is a frozen dataclass carrying the coherent pulse state: average power, peak power, pulse energy, rep rate, pulse duration, linewidth, wavelength, mode field diameter (MFD). `AseState` (in `ase/state.py`) carries the spectrally-resolved ASE — forward and backward spectra over 160 wavelength bins plus the converged inversion profile. Both are bundled by `OpticalState`, which is what flows through every component's `propagate(state) -> state` method. Components are immutable: each call returns a new `OpticalState`, never mutates the input.

For the physics behind the ASE state and the bidirectional BVP solver that produces it, read `docs/ase.md` (companion to `docs/computation_walkthrough.md`).

### Passive Components

- **Isolator** -- applies insertion loss via `signal.scaled(loss_dB)`
- **PumpCombiner** -- applies insertion loss (signal path only; pump coupling is handled by the Amplifier)
- **Circulator** -- applies insertion loss
- **ModeFieldAdapter** -- applies insertion loss *and* updates the MFD for the next fiber

### BandpassFilter

A Gaussian inter-stage filter (default: 1064 nm center, 2 nm FWHM, 0.5 dB peak loss). Applies a wavelength-dependent transfer function to the ASE spectrum and a flat insertion loss to the signal. The default config inserts one between stages 1↔2 (`BPF-12`) and 2↔3 (`BPF-23`) to suppress the 1030 nm ASE peak before the next gain stage.

### Amplifier

The core physics engine. Each amplifier stage:

1. **BVP gain solver** — iterative-shooting solve of the spatial propagation equations across 1 pump + 1 signal + 160 forward ASE + 160 backward ASE channels, all coupled to a self-consistent inversion equation. See `docs/ase.md` for details.
2. **ASE tracking** — forward and backward ASE spectra are produced as part of the BVP solve, on a 160-bin wavelength grid (970–1130 nm).
3. **Solver-health flag** — if the BVP iteration diverges (the ASE/signal fields run away) the solver clamps the result and flags the stage as having no stable steady state, rather than reporting untrustworthy numbers.
4. **Nonlinear threshold checks** — SBS and SRS thresholds evaluated against the BVP-derived output peak power.
5. **Mode A / Mode B** — `propagate(state, mode="time-dependent")` is the default: Mode B with auto-dispatch. At high rep (period < 0.1·τ, e.g. the BGU 100 kHz operating point) it delegates to Mode A and returns bit-identical numbers. At lower rep it runs the **Level 5** cycle — B1 (inter-pulse rate-equation time-stepping) + B2 (Lax-Wendroff (z, t) pulse PDE), iterated to periodic steady state. Pass `mode="steady"` (or `simulate.py --steady`) to force Mode A regardless of rep rate. Pass `mode="full"` (or `simulate.py --force-b2`) to force the B1+B2 path even at high rep — useful for pulse-shape distortion studies; this also enables the `pulse_shape.png` plot.

### Default 3-Stage Chain

Defined by `examples/bgu_3stage_mopa.json`:

```
Seed -> ISO-LP -> PC-1 -> AMP-1
     -> ISO-HP-1 -> CIRC-1 (1->2) -> BPF-12 (FBG) -> CIRC-1 (2->3) -> MFA-5/10 -> PC-2 -> AMP-2
     -> ISO-HP-2 -> CIRC-2 (1->2) -> BPF-23 (FBG) -> CIRC-2 (2->3) -> MFA-10/30 -> PC-3 -> AMP-3
```

Each circulator is entered twice (once per port-pair pass) so the chain mirrors
the physical double-pass FBG topology — see `docs/CHANGELOG.md` (2026-05-17 #5).


## Editing Parameters

The system is JSON-driven through `framework.SystemConfig`. The simplest workflow is to edit the JSON directly. For repeated tweak/run cycles, use the round-trip pattern in `examples/run_example.py` (load → mutate the dict → re-instantiate → save).

All values are SI. The `// unit` annotations in the examples below are for documentation only — strip them before saving, since real JSON does not allow comments.

### Changing Seed Parameters

Edit the `seed` block in `examples/bgu_3stage_mopa.json` (or your own copy). All values are SI:

```json
"seed": {
  "average_power": 0.00075,    // W
  "rep_rate": 100000.0,        // Hz
  "pulse_duration": 8e-09,     // s
  "linewidth": 10000000000.0,  // Hz (10 GHz)
  "wavelength": 1.064e-06,     // m
  "mfd": 5e-06                 // m  (gets overwritten by the first amplifier's fiber MFD)
}
```

`peak_power` and `pulse_energy` are derived from `average_power`, `rep_rate`, and `pulse_duration` — you don't need to (and shouldn't) set them manually.

### Changing Pump Power or Fiber Length

Find the amplifier component by `name` and edit its fields:

```json
{
  "type": "Amplifier",
  "name": "AMP-1",
  "pump_power": 1.0,    // W   <-- change this
  "length": 3.0,        // m   <-- and/or this
  ...
}
```

Or do it programmatically (see `examples/run_example.py`):

```python
from framework import Simulator, SystemConfig
cfg = SystemConfig.load("examples/bgu_3stage_mopa.json")
amp = next(c for c in cfg.components if c["name"] == "AMP-2")
amp["pump_power"] *= 1.5
sim = Simulator.from_config(cfg)
sim.run()
```

### Swapping a Passive Component

Edit the dict in `components`. Field names match the `Param` schema in each component class:

```json
{ "type": "Circulator",       "name": "CIRC-1",   "insertion_loss_dB": 0.5 },                       // insertion_loss_dB: dB
{ "type": "ModeFieldAdapter", "name": "MFA-5/15", "insertion_loss_dB": 0.3, "output_mfd": 1.5e-05 } // insertion_loss_dB: dB, output_mfd: m
```

To insert a brand-new component, add a new dict at the desired position in the `components` list.

### Changing Fiber Specifications

For a different gain fiber, set the amplifier dict's geometry/dopant fields directly:

```json
{
  "type": "Amplifier",
  "name": "AMP-2",
  "core_diameter": 1e-05,            // m
  "clad_diameter": 1.25e-04,         // m
  "core_na": 0.075,                  // numerical aperture (dimensionless)
  "length": 2.0,                     // m
  "clad_absorption_dB_per_m": 2.5,   // dB/m (pump absorption spec from manufacturer)
  "pump_power": 9.0,                 // W
  "pump_direction": "co",            // "co" (co-propagating) or "counter" (counter-propagating) pump
  "pump_wavelength": 9.76e-07,       // m
  "signal_wavelength": 1.064e-06,    // m
  "dopant": "Yb",                    // dopant species (string)
  "num_segments": 200,               // spatial grid points (count, dimensionless)
  "R_in": 0.0,                       // input-facet power reflectivity (fraction, 0–1)
  "R_out": 1e-04,                    // output-facet power reflectivity (fraction, 0–1)
  "m_pol": 2,                        // polarization modes (1 = PM, 2 = non-PM)
  "alpha_bg_dB_per_m": 0.005         // dB/m (background fiber loss)
}
```

**Important:** The Yb doping concentration (`N_Yb`) is automatically derived from `clad_absorption_dB_per_m` — never hardcode it. Use the cladding absorption value from the fiber manufacturer's datasheet.

`Amplifier.yb_5_130()`, `yb_10_125()`, `yb_30_250()` (and the legacy `stage1/2/3()` aliases) in `components.py` are convenience constructors for common Nufern-style fibers; they're useful from Python code but the saved JSON expands every parameter explicitly.


## Adding a New Amplifier Stage

Add the inter-stage components and the amplifier dict to the `components` list in your config JSON, in chain order:

```json
{ "type": "Isolator",         "name": "ISO-HP-3",  "insertion_loss_dB": 0.5, "isolation_dB": 30.0 }, // insertion_loss_dB, isolation_dB: dB
{ "type": "ModeFieldAdapter", "name": "MFA-30/40", "insertion_loss_dB": 0.5, "output_mfd": 4e-05 }, // insertion_loss_dB: dB, output_mfd: m
{ "type": "PumpCombiner",     "name": "PC-4",      "insertion_loss_dB": 0.5 },                       // insertion_loss_dB: dB
{
  "type": "Amplifier",
  "name": "AMP-4",
  "core_diameter": 4e-05,            // m
  "clad_diameter": 2.5e-04,          // m
  "core_na": 0.06,                   // numerical aperture (dimensionless)
  "length": 1.0,                     // m
  "clad_absorption_dB_per_m": 5.0,   // dB/m
  "pump_power": 150.0,               // W
  "pump_direction": "co",            // "co" (co-propagating) or "counter" (counter-propagating) pump
  "pump_wavelength": 9.76e-07,       // m
  "signal_wavelength": 1.064e-06,    // m
  "dopant": "Yb",                    // dopant species (string)
  "num_segments": 200,               // spatial grid points (count, dimensionless)
  "R_in": 0.0,                       // input-facet power reflectivity (fraction, 0–1)
  "R_out": 1e-04,                    // output-facet power reflectivity (fraction, 0–1)
  "m_pol": 2,                        // polarization modes (1 = PM, 2 = non-PM)
  "alpha_bg_dB_per_m": 0.005         // dB/m (background fiber loss)
}
```

No other changes needed: V&V checks iterate over every `Amplifier` in the chain, and the plots include every stage found in the system.


## Understanding the Outputs

### Terminal Output

Each component prints a one-line summary. The fields (values shown are
placeholders — actual numbers depend on your configuration):

```
[AMP-3]
  Avg:     <W>   Peak:    <kW>   Energy:  <uJ>   LW:  <GHz>
  ASE fwd: <mW>  peak <nm>
  Gain: <dB>  (<x>)
  Pump absorbed: <%>  (residual <mW>)
  SBS: <ratio>  [SAFE/DANGER]   SRS: <ratio>  [SAFE/DANGER]
  ASE: <dB>  [SAFE/DANGER]   solver: <n> iters (conv)
```

Amplifiers that take a non-trivial solver path also show an `[E:.. R:.. path=..]`
decoration on the gain line (energy-status / regime / solver path — see
`docs/ase.md` Part II §13b). At the end, a **V&V compliance table** shows whether
each parameter meets the project specifications (OVERALL PASS/FAIL).

### report/summary.txt

Contains:
- V&V compliance table (OVERALL PASS/FAIL + per-parameter rows)
- Component-by-component table: average power, peak power, pulse energy, linewidth at each point
- Per-amplifier details: fiber specs, derived properties (MFD, A_eff, V-number, N_Yb, overlap), gain, pump absorption, all nonlinear ratios, ASE level, linewidth evolution

### report/power_evolution.png

Dual-axis semilog plot showing **average power** (left axis, blue) and **peak power** (right axis, red) at each component. Shows exponential gain in amplifier stages and step losses at passive components.

### report/amplifier_details.png

One subplot per amplifier stage. Semilog plot of **signal power** (blue), **pump power** (green), **forward ASE integrated** (orange dashed) and **backward ASE integrated** (red dotted) vs position along the fiber. Shows pump depletion, signal growth, and how the bidirectional ASE distributes itself.

### report/ase_spectra.png

One subplot per amplifier stage. Forward (orange) and backward (red dashed) ASE spectra at the fiber endpoints, plotted as spectral power density (dBm/nm) vs wavelength. Reference lines mark the signal (1064 nm), Yb gain peak (1030 nm), and pump (976 nm).

### report/bpf_spectra.png

One subplot per inter-stage bandpass filter. Shows the filter transfer function over wavelength together with the forward ASE spectrum before and after the filter, illustrating how the 1030 nm ASE peak is suppressed before the next gain stage.

### report/nonlinear_margins.png

Grouped bar chart of SBS (blue) and SRS (orange) ratios per stage, with a red dashed threshold line at 1.0. Below the line = safe.


## Key Physics Notes

- **SBS depends on linewidth.** Broader seed linewidth raises the SBS threshold. The default 10 GHz seed linewidth keeps SBS safely below threshold at full power.
- **SAFE/DANGER flags:**
  - SBS and SRS ratios < 1.0 = SAFE (peak power below threshold)
  - ASE/signal < -20 dB = SAFE
- **The simulation omits splice and coupling losses**, so it slightly overestimates output power compared to a real system.
- **N_Yb is derived, not hardcoded.** Doping concentration is calculated from the manufacturer's cladding absorption specification, which accounts for the actual overlap geometry.
- **Cross-sections are measured data.** σ_a(λ)/σ_e(λ) come from the measured Yb aluminosilicate dataset of Melkumov et al. (arXiv:1502.02885), stored in `ase/data/cross_sections_yb.csv`. Only the spectral *shape* matters physically — the absolute scale cancels under the N_Yb back-derivation. See `docs/ase.md` Part I §3.2 and `docs/CHANGELOG.md` for provenance.

For the complete physics reference (all equations, cross-sections, and derivations), see `docs/physics_spec.md`.
