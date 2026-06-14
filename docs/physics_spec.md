# Physics Specification — Yb-Doped Fiber MOPA Simulator

This document contains all the optical engineering context needed to implement and extend the simulator. It covers the laser physics, fiber parameters, gain modeling, nonlinear effects, component specifications, and system requirements. **ASE physics lives in its own document — see `ase.md`.**

> **Numbers policy.** This document contains physics constants, equations, and clearly-labelled *example* fiber/seed inputs — all system-independent. It deliberately does **not** quote simulation *outputs* (gains, output powers, ASE levels, nonlinear ratios); those depend on the specific system. Run `python simulate.py` to see live values for a given configuration.

---

## 1. System Overview

A MOPA (Master Oscillator Power Amplifier) takes a weak seed laser pulse and amplifies it through multiple fiber amplifier stages to reach high power. Our system has three stages:

```
Seed → [Isolator] → [Pump Combiner] → [Yb Fiber Amp 1 (5/130 µm)] →
     → [Isolator] → [Circulator+FBG] → [Mode Field Adapter] →
     → [Pump Combiner] → [Yb Fiber Amp 2 (10/125 µm)] →
     → [Isolator] → [Circulator+FBG] → [Mode Field Adapter] →
     → [Pump Combiner] → [Yb Fiber Amp 3 (30/250 µm)] → Output
```

Each stage uses a progressively larger fiber core to handle increasing power while managing nonlinear effects. The fiber core sizes (5, 10, 30 µm) are standard in the industry for this purpose.

### Why three stages?

The total gain needed is ~50 dB (seed at ~0.5 mW → output at ~30 W). A single amplifier with 50 dB gain would have uncontrollable ASE (amplified spontaneous emission). Splitting across stages with isolators between them prevents backward ASE from one stage from being amplified by the previous stage.

---

## 2. Seed Laser

The seed defines the initial signal that enters the amplifier chain.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Average power | 0.75 mW | Typical for a semiconductor seed |
| Wavelength | 1064 nm | Yb emission peak |
| Rep rate | 10 Hz – 100 kHz | Adjustable; 100 kHz is the primary design point |
| Pulse duration | 4 – 8 ns FWHM | Nanosecond regime |
| Linewidth | 10 GHz | Broadened intentionally for SBS suppression |
| Pulse shape | Gaussian | Shape factor k = 0.94 for FWHM→energy conversion |

### Derived quantities from seed parameters

```
pulse_energy = average_power / rep_rate
peak_power = pulse_energy / (pulse_duration × shape_factor)
```

For a Gaussian pulse, shape_factor = 0.94 (ratio of FWHM to equivalent square pulse).

Example: 0.75 mW at 100 kHz, 8 ns:
- Energy = 0.75e-3 / 100e3 = 7.5 nJ
- Peak = 7.5e-9 / (8e-9 × 0.94) = 1.0 W

---

## 3. Ytterbium-Doped Fiber Physics

### 3.1 Energy Levels

Yb³⁺ is a quasi-two-level system with two manifolds:
- ²F₇/₂ (ground state, 4 Stark sublevels)
- ²F₅/₂ (excited state, 3 Stark sublevels)

The pump (976 nm) excites ions from ground to excited state. The signal (1064 nm) stimulates emission from excited back to ground. Because both pump and signal transitions share the same two manifolds, signal photons can also be *absorbed* by ground-state ions (unlike a true four-level system like Nd:YAG). This is why it's called "quasi-two-level" — you need a minimum pump power just to reach transparency.

### 3.2 Cross-Sections

These are the probabilities of absorption/emission per ion per unit intensity. All values in m². They are sampled from the measured aluminosilicate (AS) table of Melkumov et al. (arXiv:1502.02885, Appendix 2; see `ase.md` Part I §3.2 and §12 for full provenance — the table is a non-peer-reviewed preprint, with a peer-reviewed journal version that omits the table). The full λ-dependent spectra live in `ase/data/cross_sections_yb.csv`; the scalars below are the values at the pump and signal wavelengths.

| Symbol | Value | Transition |
|--------|-------|------------|
| σ_a_pump | 2.69 × 10⁻²⁴ | Pump absorption at 976 nm |
| σ_e_pump | 2.97 × 10⁻²⁴ | Pump emission at 976 nm |
| σ_a_signal | 0.0046 × 10⁻²⁴ | Signal absorption at 1064 nm |
| σ_e_signal | 0.30 × 10⁻²⁴ | Signal emission at 1064 nm |

The signal emission cross-section (0.30 × 10⁻²⁴) is ~65× larger than signal absorption (0.0046 × 10⁻²⁴), which is why 1064 nm is an efficient lasing wavelength for Yb.

At the 976 nm zero line σ_e is slightly above σ_a (σ_e/σ_a ≈ 1.10), the McCumber-consistent ratio for the lowest-sublevel transitions of each manifold. (Only the spectral *shape* enters the model — the absolute scale cancels because N_Yb is back-derived from the measured cladding absorption, §4.4.)

### 3.3 Upper-State Lifetime

τ = 0.83 ms (830 µs)   ← Melkumov AS value (was 0.84 ms under the old anchors)

This is the spontaneous decay time of the excited state. It sets the energy storage time — at rep rates much higher than 1/τ ≈ 1.2 kHz, the inversion is continuously replenished between pulses. At low rep rates, the inversion can build up to near full inversion between pulses, extracting more energy per pulse.

### 3.4 Steady-State Inversion

At any point along the fiber, the fraction of ions in the excited state (n₂) is determined by the local pump and signal intensities:

```
n₂ = (W_p + W_s) × τ / (1 + (W_p + W_ep + W_s + W_es) × τ)
```

Where the transition rates are:
```
W_p  = σ_a_pump × I_pump / (h × ν_pump)     — pump absorption rate
W_ep = σ_e_pump × I_pump / (h × ν_pump)     — pump emission rate
W_s  = σ_a_signal × I_signal / (h × ν_signal) — signal absorption rate
W_es = σ_e_signal × I_signal / (h × ν_signal) — signal emission rate
```

I_pump and I_signal are the local intensities [W/m²]. For double-clad fibers:
```
I_pump = P_pump / A_clad    (pump fills the entire cladding)
I_signal = P_signal / A_core  (signal is guided in the core)
```

### 3.5 Gain Coefficient

The local gain coefficient for the signal [1/m] is:

```
g_signal = Γ_signal × N_Yb × (n₂ × σ_e_signal - (1 - n₂) × σ_a_signal)
```

Where:
- Γ_signal is the signal-core overlap factor
- N_Yb is the Yb³⁺ doping concentration [ions/m³]
- The term (n₂ × σ_e - (1-n₂) × σ_a) gives net gain when n₂ is above the transparency inversion

The transparency inversion level is:
```
n₂_transparency = σ_a_signal / (σ_a_signal + σ_e_signal) ≈ 0.0046 / (0.0046 + 0.30) ≈ 0.015
```

This is very low (~1.5%), meaning Yb at 1064 nm is nearly four-level — you need barely any pump to start amplifying.

### 3.6 Pump Depletion

Similarly, the pump evolves as:
```
g_pump = Γ_pump × N_Yb × (n₂ × σ_e_pump - (1 - n₂) × σ_a_pump)
```

For double-clad fibers, the pump overlap factor is:
```
Γ_pump = A_core / A_clad
```

This is small (e.g., 0.0015 for 5/130 fiber) because the pump fills the much larger cladding. This is why double-clad fibers need longer lengths or higher doping — the pump is only absorbed where it overlaps with the doped core.

---

## 4. Fiber Specifications

### 4.1 Geometry and Modes

Each fiber is characterized by core diameter, cladding diameter, and core numerical aperture (NA).

The V-number determines whether the fiber is single-mode:
```
V = (2π / λ) × r_core × NA
```

If V < 2.405, the fiber supports only the fundamental mode (single-mode).
If V > 2.405, it's multi-mode but can be operated quasi-single-mode by coiling to strip higher-order modes.

### 4.2 Mode Field Diameter (MFD)

For single-mode fibers (V < 2.405), use the Marcuse approximation:
```
w/a = 0.65 + 1.619 / V^1.5 + 2.879 / V^6
MFD = 2 × w
```
where a = core radius and w = mode radius.

For multi-mode fibers (V > 2.405) operating on the fundamental mode with coil-based HOM suppression:
```
w = 0.65 × r_core
MFD = 2 × w
```

The effective area is:
```
A_eff = π × (MFD/2)²
```

### 4.3 Overlap Factor

Signal overlap with the core:
```
Γ_signal = 1 - exp(-2 × (r_core / w)²)
```

This is typically 0.6–0.99 depending on V-number.

### 4.4 Doping Concentration

**Critical implementation detail**: Never hardcode doping concentration. Derive it from the manufacturer's cladding absorption specification, which is the only reliable number on the datasheet.

The relationship is:
```
α_clad [dB/m] = 4.343 × σ_a_pump × N_Yb × (A_core / A_clad)
```

Therefore:
```
N_Yb = α_clad / (4.343 × σ_a_pump × A_core / A_clad)
```

This ensures the gain model is self-consistent with the manufacturer's specs.

### 4.5 Project Fiber Inventory

**These are representative round-number examples — the actual JSON config carries the measured per-fiber datasheet values.** Example-fiber convenience constructors live in `components.py` → `Amplifier.yb_5_130()`, `yb_10_125()`, `yb_30_250()` (with legacy `stage1()`/`stage2()`/`stage3()` aliases); the saved JSON expands every parameter explicitly.

| Fiber | Core/Clad [µm] | NA | Clad abs [dB/m] | Type |
|-------|-----------------|-----|------------------|------|
| Stage 1 | 5 / 130 | 0.12 | ~1.6 | Single-mode |
| Stage 2 | 10 / 125 | ~0.08 | ~5 | Single-mode |
| Stage 3 | 30 / 250 | ~0.06 | ~7 | Multi-mode (LMA) |

The geometry above is what you *set*. Everything else — MFD, A_eff, V-number,
Γ_signal, Γ_pump, N_Yb (derived from clad absorption, §4.4), E_sat, P_sat, the
Kerr coefficient γ — is **computed by the simulator** from these inputs using the
formulas in §4.1–§4.6. Because they are deterministic functions of the fiber
spec, they are not reproduced here as a table (they would go stale whenever a
spec changes); the per-amplifier `summary.txt` prints them for whatever fiber you
actually run.

### 4.6 Saturation Parameters

These set the scale of gain saturation:
```
E_sat = h × ν_signal × A_eff / (σ_e_signal + σ_a_signal)
P_sat = E_sat / τ
```

When signal power approaches P_sat (CW) or pulse energy approaches E_sat (pulsed), the gain compresses.

---

## 5. Gain Modeling

### 5.1 Spatial Propagation (Recommended)

Divide the fiber into N segments (typically 100). At each segment:

1. Compute local intensities from current P_signal and P_pump
2. Compute local inversion n₂ from the steady-state equation (Section 3.4)
3. Compute local gain g_signal and pump depletion g_pump (Sections 3.5–3.6)
4. Update powers: P_signal *= exp(g_signal × dz), P_pump *= exp(g_pump × dz)

This captures non-uniform pump depletion along the fiber and naturally handles gain saturation.

### 5.2 Key Gain Modeling Considerations

- **Average power vs peak power**: The gain model propagates *average* power through the fiber. The steady-state inversion at 100 kHz rep rate (period = 10 µs << τ = 840 µs) is essentially set by the average signal power, not the peak. The gain experienced by each pulse equals the gain computed from average powers. Peak power is then derived by multiplying the average power gain by the input peak power.

- **ASE clamping**: In a real amplifier, ASE grows exponentially with gain and eventually clamps the inversion. The simulator's spectrally-resolved BVP solver in the `ase/` package models this directly — backward ASE feeds into the inversion equation summed over all 322 channels (160 forward + 160 backward ASE bins plus pump and signal), so gain saturation by ASE depletion is built in. See `ase.md` for the full spectrally-resolved model. The cross-section scalars listed in §3.2 of this document are the pump (976 nm) and signal (1064 nm) values; the full λ-dependent spectra used by the BVP solver live in `ase/data/cross_sections_yb.csv`.

- **Pump direction**: We model co-propagating pump (pump and signal travel in the same direction). Counter-propagating pump gives higher output power but is harder to model (requires iterative boundary value solver). The spatial model handles co-propagating straightforwardly.

---

## 6. Passive Component Specifications

These components only affect power (insertion loss) and, for mode field adapters, the beam size. They do not affect wavelength, linewidth, or pulse shape.

### 6.1 Insertion Loss Values

**These are typical literature estimates — update with actual purchased component datasheet values.** Loss values are set in `simulate.py` → `build_default_system()` (marked with TODO comments).

| Component | Loss [dB] | Source | Function |
|-----------|-----------|--------|----------|
| PM Isolator (low-power) | 1.0 | estimate | Blocks backward light, protects seed |
| PM Isolator (high-power) | 0.5 | estimate | Inter-stage backward ASE protection |
| Mode Field Adapter 5→10 | 0.3 | estimate | Adiabatic taper between fiber sizes |
| Mode Field Adapter 10→30 | 0.5 | estimate | Larger MFD mismatch = higher loss |
| Pump Combiner (signal path) | 0.3–0.5 | estimate | Injects pump light, signal passes through |
| Circulator (port 1→2) | 0.8 | estimate | Enables backward reflection monitoring via FBG |
| Bandpass Filter | 0.5 | estimate | Rejects out-of-band ASE |
| Fusion Splice (matched) | 0.03 | estimate | Typical single-mode splice |

### 6.2 Why Each Component Exists

- **Isolators** between stages prevent backward ASE and reflections from reaching upstream amplifiers (which could destabilize or damage them).
- **Circulators + FBGs** (fiber Bragg gratings) monitor backward-propagating power. The FBG reflects backward light at 1064 nm into port 3 of the circulator for monitoring. If SBS or back-reflections spike, the system can shut down.
- **Mode field adapters** bridge the MFD mismatch between fiber stages. Without them, the splice loss between a 5 µm and 10 µm core fiber would be several dB.
- **Pump combiners** couple 976 nm pump diode light into the fiber cladding while allowing the 1064 nm signal to pass through the core with minimal loss.

### 6.3 Implementation

Every passive component applies the same transformation:
```python
output = input.scaled(loss_dB)  # reduces power/energy by loss_dB
```

ModeFieldAdapter additionally updates the MFD:
```python
output.mfd = new_mfd  # so subsequent A_eff calculations use the right value
```

No passive component changes wavelength, linewidth, or pulse duration.

---

## 7. Nonlinear Effects

Nonlinear effects arise from the interaction of high-intensity light with the glass fiber. They are the primary limitation on achievable peak power in fiber lasers.

### 7.1 Stimulated Brillouin Scattering (SBS)

**What it is**: The signal creates acoustic waves (phonons) in the glass via electrostriction. These acoustic waves act as a moving Bragg grating that reflects the signal backward. Above a threshold power, this process runs away — most of the signal is reflected backward, potentially damaging upstream components.

**Threshold formula**:
```
P_th_SBS = 21 × K × A_eff / (g_B_eff × L_eff)
```

Where:
- K = 1 for PM fiber (polarization maintained), K = 2 for non-PM
- A_eff = effective mode area [m²]
- g_B_eff = effective Brillouin gain coefficient [m/W]
- L_eff = effective interaction length [m]
- The factor 21 comes from the critical SBS gain (≈ 21 Nepers for silica fibers)

**Effective Brillouin gain** includes linewidth broadening:
```
g_B_eff = g_B / (1 + Δν_signal / Δν_B)
```

Where:
- g_B = 3 × 10⁻¹¹ m/W (intrinsic Brillouin gain of silica)
- Δν_B = 35 MHz (Brillouin linewidth)
- Δν_signal = signal linewidth

**This is the most important equation for SBS mitigation.** When Δν_signal >> Δν_B, the threshold scales linearly with signal linewidth. A 10 GHz seed linewidth gives a factor of ~286× improvement over a 100 MHz seed.

**Effective length**:
For short pulses, the interaction length is limited by the pulse spatial extent:
```
L_pulse = c × τ_pulse / (2 × n_glass)
L_eff = min(fiber_effective_length, L_pulse)
```

The factor of 2 accounts for the backward-propagating Stokes wave meeting the forward-propagating signal.

**Transient suppression**:
For pulses shorter than ~5× the phonon lifetime (T_phonon = 10 ns), the acoustic wave doesn't have time to build up fully:
```
if τ_pulse < 5 × T_phonon:
    r = τ_pulse / T_phonon
    transient_factor = 1 / (r / (1 - exp(-r)))
else:
    transient_factor = 1.0

P_th_SBS *= transient_factor  (divide threshold by this; it's < 1)
```

For 8 ns pulses with 10 ns phonon lifetime, the transient factor provides ~20% threshold reduction (marginal benefit).

**Brillouin parameters**:
- g_B = 3 × 10⁻¹¹ m/W
- Δν_B = 35 MHz
- ν_B_shift = 16 GHz (frequency shift of Stokes wave)
- T_phonon = 10 ns

### 7.2 Stimulated Raman Scattering (SRS)

**What it is**: The signal excites molecular vibrations in the glass, transferring energy to a red-shifted Stokes wave at ~1116 nm (13.2 THz shift for silica). Unlike SBS, SRS propagates forward and has a very broad gain bandwidth (~10 THz), so linewidth broadening does NOT help.

**Threshold formula**:
```
P_th_SRS = 16 × A_eff / (g_R × L_eff)
```

Where:
- g_R = 1 × 10⁻¹³ m/W (Raman gain coefficient)
- The factor 16 is the critical Raman gain for silica
- L_eff = effective fiber length

SRS threshold is linewidth-independent and only depends on fiber geometry and length. Short fibers with large cores have the highest thresholds.

**Stokes wavelength**: 1116.3 nm (for 1064 nm pump)

### 7.3 Self-Phase Modulation (SPM)

**What it is**: The Kerr effect (intensity-dependent refractive index) causes the signal to accumulate a nonlinear phase shift, broadening its spectrum without energy transfer to other waves.

**B-integral** (accumulated nonlinear phase):
```
B = γ × P_peak × L_eff
```

Where the nonlinear coefficient is:
```
γ = 2π × n₂ / (λ × A_eff)
```

And n₂ = 2.6 × 10⁻²⁰ m²/W (Kerr nonlinear index of silica).

**SPM-induced linewidth broadening**:
```
T₀ = τ_pulse / 1.665        (Gaussian half-1/e duration from FWHM)
Δν_SPM = 0.86 × B / (π × T₀)  (approximate spectral broadening)
```

**Linewidth update** (quadrature addition):
```
Δν_new = sqrt(Δν_old² + Δν_SPM²)
```

This broadened linewidth feeds back into the SBS threshold calculation for subsequent stages — a beneficial effect.

**Severity thresholds**:
- B < 1 rad: SAFE (minimal spectral distortion)
- 1 < B < π rad: CAUTION (noticeable broadening)
- B > π rad: DANGER (severe spectral distortion, potential pulse breakup)

For this system, B-integrals of 10–30 rad are typical. This is high but acceptable because the broad seed linewidth (10 GHz) means the relative spectral broadening is small, and the downstream application (SRS to yellow) doesn't require narrow linewidth.

### 7.4 Nonlinear Effect Summary and Flagging

After each amplifier stage, compute all three effects and flag them:

```
SBS: ratio = P_peak / P_th_SBS
     ✅ if ratio < 1.0     (safe)
     ❌ if ratio ≥ 1.0     (exceeded — signal will be reflected)

SRS: ratio = P_peak / P_th_SRS
     ✅ if ratio < 1.0     (safe)
     ❌ if ratio ≥ 1.0     (exceeded — energy transfer to Stokes)

SPM: B = γ × P_peak × L_eff
     ✅ if B < π           (acceptable broadening)
     ⚠️ if B ≥ π           (significant broadening, not necessarily fatal)
```

**The nonlinear check uses peak power**, not average power, because nonlinear effects scale with instantaneous intensity.

---

## 8. Wavelength and Frequency Effects

### 8.1 What Doesn't Change the Wavelength

None of the following shift the center wavelength:
- Passive components (isolators, combiners, circulators, splices, MFAs)
- Yb fiber amplification (stimulated emission preserves photon frequency exactly)
- SPM (broadens the spectrum symmetrically around the center frequency)

The signal stays at 1064 nm throughout the amplifier chain.

### 8.2 What Changes the Linewidth

- **SPM** broadens the linewidth (see Section 7.3). This is a real physical effect.
- **Passive components** do not affect linewidth.
- **Amplification** itself does not change linewidth (stimulated emission preserves the spectral profile). ASE adds background noise across a broad spectrum but doesn't broaden the signal peak.

### 8.3 Effects That Would Shift the Wavelength (Not Modeled)

These would only occur in downstream conversion stages, not in the MOPA itself:
- Second Harmonic Generation (SHG): 1064 nm → 532 nm (green)
- Stimulated Raman Scattering conversion: 1064 nm → 1116 nm → 1178 nm
- SHG of Raman-shifted: 1178 nm → 589 nm (yellow — the project's end goal)

---

## 9. System Requirements

### 9.1 Final Output Requirements

From the project report (Section 7, V&V table):

| Parameter | Min | Max |
|-----------|-----|-----|
| Average power | 22.5 W | 70 W |
| Peak power | 15 kW | 50 kW |
| Pulse duration | 4 ns | 8 ns |
| Wavelength | 1064 nm | 1064 nm |
| Rep rate | 10 Hz | 100 kHz |
| SBS ratio | — | < 1.0 per stage |
| SRS ratio | — | < 1.0 per stage |
| ASE | — | < −20 dB below signal |

### 9.2 Feasibility Constraint

The average power and peak power targets are linked through:
```
P_avg = P_peak × f_rep × τ × k
```

Both targets can only be simultaneously met for certain (rep_rate, pulse_duration) combinations. Analysis shows the feasible region is:
- 80–100 kHz rep rate
- 6–8 ns pulse duration
- Requires ~10 GHz seed linewidth for SBS suppression

The optimal point is 100 kHz / 8 ns / 10 GHz seed, giving output around 25–35 W average, 30–45 kW peak.

### 9.3 Optimized Component Parameters

From parametric optimization with the Spatial gain model:

| Component | Parameter | Optimized Range |
|-----------|-----------|-----------------|
| Seed linewidth | Δν | 10 GHz |
| Stage 1 pump | P_pump1 | 0.3 – 1.0 W |
| Stage 1 fiber | L₁ | 2.0 – 4.0 m |
| Stage 2 pump | P_pump2 | 3.0 – 9.0 W |
| Stage 2 fiber | L₂ | 1.5 – 2.0 m |
| Stage 3 pump | P_pump3 | 50 – 100 W |
| Stage 3 fiber | L₃ | 1.0 – 1.5 m |

---

## 10. Physical Constants Reference

All in SI units.

```
h       = 6.626 × 10⁻³⁴   J·s       Planck constant
c       = 3 × 10⁸          m/s       Speed of light
λ_pump  = 976 × 10⁻⁹       m         Pump wavelength
λ_signal= 1064 × 10⁻⁹      m         Signal wavelength
ν_pump  = c / λ_pump        Hz        Pump frequency
ν_signal= c / λ_signal      Hz        Signal frequency

σ_a_pump  = 2.69 × 10⁻²⁴   m²        Pump absorption cross-section (Melkumov AS, 976 nm)
σ_e_pump  = 2.97 × 10⁻²⁴   m²        Pump emission cross-section (Melkumov AS, 976 nm)
σ_a_signal= 0.0046 × 10⁻²⁴ m²        Signal absorption cross-section (Melkumov AS, 1064 nm)
σ_e_signal= 0.30 × 10⁻²⁴   m²        Signal emission cross-section (Melkumov AS, 1064 nm)
τ         = 0.83 × 10⁻³     s         Upper-state lifetime (Melkumov AS)

g_B       = 3 × 10⁻¹¹       m/W       Brillouin gain coefficient
Δν_B      = 35 × 10⁶        Hz        Brillouin linewidth
T_phonon  = 10 × 10⁻⁹       s         Phonon lifetime
g_R       = 1 × 10⁻¹³       m/W       Raman gain coefficient
n₂        = 2.6 × 10⁻²⁰     m²/W      Kerr nonlinear index
n_glass   = 1.45             —         Linear refractive index of silica
```

---

## 11. Implementation Notes

### 11.1 Signal State

The Signal object must carry at minimum:
- average_power [W]
- peak_power [W]
- pulse_energy [J]
- rep_rate [Hz]
- pulse_duration [s]
- linewidth [Hz]
- wavelength [m]
- mfd [m] (mode field diameter — determines A_eff for NL calculations)

Always keep power, energy, and peak power consistent after any modification.

### 11.2 Gain Model Propagation

```python
for each segment dz:
    compute local n₂ from pump and signal intensities
    g = overlap × N_Yb × (n₂ × σ_e - (1-n₂) × σ_a)
    P_signal *= exp(g × dz)
    P_pump *= exp(g_pump × dz)   # g_pump is negative (absorption)
```

Use average power for propagation. Multiply gain by peak power afterward:
```
new_peak = old_peak × (P_signal_out / P_signal_in)
```

### 11.3 Doping Derivation

Never hardcode N_Yb. Always derive:
```
pump_overlap = A_core / A_clad
N_Yb = α_clad_dBm / (4.343 × σ_a_pump × pump_overlap)
```

### 11.4 Mode Field Diameter Handling

The signal's MFD must be updated whenever it enters a new fiber (either through a ModeFieldAdapter or at the start of an Amplifier). The Amplifier should set the signal's MFD to match its fiber's MFD before computing gain.

### 11.5 Nonlinear Checks Use Peak Power

All nonlinear thresholds (SBS, SRS, SPM) compare against **peak power**, not average power. The nonlinear interaction happens during the pulse duration, not averaged over the repetition period.

### 11.6 SPM Linewidth Feedback

After computing SPM broadening in one amplifier, update the signal's linewidth before it enters the next stage. This broadened linewidth will increase the SBS threshold in subsequent stages — this is a real and important physical effect that significantly helps system feasibility.
