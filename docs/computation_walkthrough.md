# Computation Walkthrough — How the Simulator Calculates Power and Signal Parameters

This document explains, step by step, how the simulator computes the signal as it
propagates: power, gain, and nonlinear-effect thresholds. It follows a laser pulse
from the seed through the amplifier chain, showing every equation. No programming
knowledge is required.

ASE (Amplified Spontaneous Emission) has its own dedicated document — see
`ase.md`. This walkthrough covers gain, nonlinear effects, and the system chain,
and points to `ase.md` where the two interact.

> **Numbers policy.** This walkthrough shows equations and a few *worked
> examples* using clearly-labelled example inputs (an example seed, an example
> 5/130 fiber) to illustrate the formulas. It deliberately does **not** quote
> end-to-end simulation *results* (per-stage gains, output powers, ASE levels,
> nonlinear ratios), because those depend on the specific system you simulate.
> To see live numbers for any configuration, run `python simulate.py` — it prints
> a component-by-component table and a V&V compliance table.

---

## 1. The Signal — What Numbers Describe a Laser Pulse

At every point in the system, the laser pulse is described by these parameters:

| Parameter           | Symbol | Units | What it means                                            |
| ------------------- | ------ | ----- | -------------------------------------------------------- |
| Average power       | P_avg  | W     | Power averaged over time (what a slow detector reads)    |
| Peak power          | P_peak | W     | Power during the pulse itself (much higher than average) |
| Pulse energy        | E      | J     | Energy in a single pulse                                 |
| Repetition rate     | f_rep  | Hz    | How many pulses per second                               |
| Pulse duration      | tau    | s     | How long each pulse lasts (FWHM)                         |
| Linewidth           | dnu    | Hz    | Spectral width of the laser (how "pure" the color is)    |
| Wavelength          | lambda | m     | The color of light (1064 nm = infrared)                  |
| Mode field diameter | MFD    | m     | How wide the light beam is inside the fiber              |

ASE is tracked separately as a spectrally-resolved bidirectional state, not a
single number — see `ase.md`.

These parameters are linked by three fundamental relationships:

```
E = P_avg / f_rep

P_peak = E / (tau x k)

where k = 0.94 (shape factor for a Gaussian pulse)
```

**Why average vs peak matters:** a weak-average pulse train can have a very high
instantaneous peak. Nonlinear effects (SBS, SRS) depend on **peak** power; the
amplifier gain model uses **average** power. Keeping these two consistent after
every operation is essential.

---

## 2. Starting Point — The Seed Laser

The simulation begins by creating the seed signal. The default example seed:

```
P_avg = 0.75 mW = 0.00075 W
f_rep = 100 kHz = 100,000 Hz
tau   = 8 ns = 8 x 10^-9 s
dnu   = 10 GHz = 10 x 10^9 Hz
lambda = 1064 nm
MFD   = 5 um (matched to the first amplifier fiber)
```

The derived values follow from the relationships in §1:

```
E      = P_avg / f_rep = 0.00075 / 100,000 = 7.5 x 10^-9 J = 7.5 nJ
P_peak = E / (tau x 0.94) = 7.5e-9 / (8e-9 x 0.94) ~ 1.0 W
```

So the example seed is a very weak signal: 0.75 mW average, about 1 W peak. (These
are example seed inputs; change them in the config JSON and the derived values
follow.)

---

## 3. Passive Components — Simple Power Loss

Four types of passive components sit between amplifier stages. They all do the
same basic thing: reduce the signal power by a fixed amount (in decibels). None of
them change the wavelength, linewidth, or pulse shape.

### How decibel loss works

A loss of L dB reduces all power values by a fixed fraction:

```
fraction_remaining = 10^(-L / 10)
P_out = P_in x fraction_remaining
```

This applies equally to average power, peak power, and pulse energy. ASE is
attenuated bin-by-bin in the spectral state (the same flat factor for plain
insertion-loss components, a wavelength-dependent factor for bandpass filters —
see `ase.md`).

| Loss (dB) | Fraction remaining | Meaning     |
| --------- | ------------------ | ----------- |
| 0.3 dB    | 93.3%              | Loses 6.7%  |
| 0.5 dB    | 89.1%              | Loses 10.9% |
| 0.8 dB    | 83.2%              | Loses 16.8% |
| 1.0 dB    | 79.4%              | Loses 20.6% |
| 3.0 dB    | 50.0%              | Loses half  |

### The four passive component types

> **Note:** loss values are component-datasheet inputs you set in the config JSON
> (or via the factory defaults). The values below are typical ranges, not results.

| Component                | Typical loss | Purpose                                                                                          |
| ------------------------ | ------------ | ------------------------------------------------------------------------------------------------ |
| Isolator                 | 0.5–1.0 dB   | Blocks backward-traveling light (protects upstream components from reflections and backward ASE) |
| Pump combiner            | 0.3–0.5 dB   | Injects 976 nm pump light into the cladding; the signal passes through with some loss            |
| Circulator               | ~0.8 dB / pass | Routes backward reflections to a monitor port for safety (one port-pair pass per entry)        |
| Mode field adapter (MFA) | 0.3–0.5 dB   | Bridges the size mismatch between fibers of different core diameters                              |

The MFA additionally updates the mode field diameter (MFD) to match the next
fiber. For example, an MFA between a 5 µm and a 10 µm stage changes the MFD from
5 µm to 10 µm so subsequent A_eff calculations use the right value.

### Worked example: applying a loss

Starting from any input power, an Isolator with 1.0 dB loss gives:

```
fraction = 10^(-1.0/10) = 0.7943
P_avg_out  = P_avg_in  x 0.7943
P_peak_out = P_peak_in x 0.7943
E_out      = E_in      x 0.7943      (equivalently P_avg_out / f_rep)
```

Every passive component is just this multiply. The signal entering the first
amplifier has already lost the combined dB of every passive before it.

---

## 4. Amplifier Stages — The Core Physics

This is where the signal gets amplified. Each amplifier is a piece of
Ytterbium-doped fiber pumped by a 976 nm laser diode. The computation involves
several steps done in sequence.

### 4.1 Fiber geometry calculations (done once per amplifier)

Before any signal propagation, the simulator computes the fiber's optical
properties from its physical specifications. The worked numbers below are for an
**example 5/130 µm fiber (NA = 0.12)** — they illustrate the formulas; a different
fiber gives different numbers.

**Core and cladding areas:**

```
A_core = pi x (core_diameter / 2)^2 = pi x (2.5e-6)^2 = 1.96e-11 m^2
A_clad = pi x (clad_diameter / 2)^2 = pi x (65e-6)^2  = 1.33e-8  m^2
```

**V-number** (determines if the fiber is single-mode):

```
V = (2 x pi / lambda) x r_core x NA
  = (2 x pi / 1064e-9) x 2.5e-6 x 0.12 = 1.77
```

Since V < 2.405, this example fiber is single-mode.

**Mode field diameter (MFD):** for single-mode fibers (V < 2.405), the Marcuse
approximation gives how wide the light spreads in the core:

```
w/a = 0.65 + 1.619 / V^1.5 + 2.879 / V^6
    = 0.65 + 1.619 / 1.77^1.5 + 2.879 / 1.77^6 = 1.43
MFD = 2 x (w/a) x r_core = 2 x 1.43 x 2.5 um = 7.16 um
```

For multi-mode fibers (V > 2.405, like a 30 µm core), a simpler approximation is
used: `MFD = 2 x 0.65 x r_core`.

**Effective mode area:**

```
A_eff = pi x (MFD / 2)^2 = pi x (3.58e-6)^2 = 4.0e-11 m^2 = 40 um^2
```

This is the area that matters for nonlinear effects — smaller area means higher
intensity for the same power.

**Signal overlap factor** (what fraction of the signal mode overlaps the doped
core):

```
Gamma_signal = 1 - exp(-2 x (r_core / w)^2)
             = 1 - exp(-2 x (2.5 / 3.58)^2) = 0.62
```

So only ~62% of the signal light overlaps with the gain region for this fiber; the
rest travels in the undoped part and isn't amplified.

**Pump overlap factor:**

```
Gamma_pump = A_core / A_clad = 1.96e-11 / 1.33e-8 = 0.0015
```

Very small — the pump fills the whole cladding but only ~0.15% of it overlaps the
doped core at any cross-section. This is inherent to double-clad fiber design and
is why cladding pumping needs lots of pump power.

**Doping concentration** (derived from the manufacturer's cladding absorption
spec — never hardcoded):

```
N_Yb = alpha_clad / (4.343 x sigma_a_pump x Gamma_pump)
```

Where `alpha_clad` is the cladding absorption [dB/m] from the datasheet, 4.343 =
10/ln(10) converts dB↔Nepers, and `sigma_a_pump` is the measured Yb:AS absorption
cross-section at 976 nm = 2.69 x 10^-24 m^2 (Melkumov et al., arXiv:1502.02885;
see `ase.md` Part I §3.2). For the example fiber with alpha_clad ≈ 1.65 dB/m:

```
N_Yb = 1.65 / (4.343 x 2.69e-24 x 0.0015) ~ 9.5e25 ions/m^3
```

a physically reasonable doping concentration for Yb-doped silica that lands close
to the published Liekki Yb1200 concentration (~9e25) — a consistency check on the
measured cross-section scale.

**Why the absolute cross-section scale doesn't matter.** N_Yb is back-derived so
that `N_Yb x sigma_a_pump` reproduces the measured cladding absorption. If a
different dataset reported a different absolute sigma_a(976), N_Yb would shift
inversely and every gain/absorption quantity (which depends on the product N_Yb x
sigma) would be unchanged. Only the spectral *shape* sigma(lambda)/sigma(976)
physically matters.

### 4.2 Segment-by-segment propagation (the gain loop)

The fiber is divided into small segments (200 by default). The solver walks
through the fiber, updating the signal power, pump power, and the
spectrally-resolved ASE arrays at each step:

```
dz = fiber_length / num_segments
```

At each segment the solver computes the inversion locally and steps every channel
forward. The same gain coefficient and inversion equation below apply per-segment;
the spectrally-resolved BVP wraps a forward sweep and a backward sweep around this
loop and iterates to convergence (the full algorithm — including the ASE bins — is
in `ase.md`). The gain-relevant steps:

#### Step A: Compute local intensities

```
I_pump   = P_pump / A_clad        (pump fills the whole cladding)
I_signal = P_signal / A_core      (signal mode lives in the core)
```

ASE is not lumped into the signal intensity — it enters the inversion equation as
320 separate per-bin terms (forward + backward, 160 bins each). See `ase.md`.

#### Step B: Compute transition rates

These describe how fast ions are pumped up and stimulated down:

```
W_p  = sigma_a_pump   x I_pump   / (h x nu_pump)      pump absorption rate
W_ep = sigma_e_pump   x I_pump   / (h x nu_pump)      pump stimulated emission rate
W_s  = sigma_a_signal x I_signal / (h x nu_signal)    signal absorption rate
W_es = sigma_e_signal x I_signal / (h x nu_signal)    signal stimulated emission rate
```

where `h = 6.626e-34 J·s` and `nu = c / lambda`. The factor `(h x nu)` converts
intensity [W/m²] to photon flux [photons/m²/s].

**Cross-section values used** (measured Yb:AS, Melkumov et al.; see `ase.md`):

| Symbol         | Value (m²)    | Process                                                  |
| -------------- | ------------- | -------------------------------------------------------- |
| sigma_a_pump   | 2.69 x 10⁻²⁴  | Pump photon absorbed, ion goes up (976 nm)               |
| sigma_e_pump   | 2.97 x 10⁻²⁴  | Pump photon triggers emission, ion comes down (976 nm)   |
| sigma_a_signal | 0.0046 x 10⁻²⁴ | Signal photon absorbed — very small (1064 nm)           |
| sigma_e_signal | 0.30 x 10⁻²⁴  | Signal photon triggers emission — ~65× larger (1064 nm)  |

#### Step C: Compute the inversion fraction

The inversion fraction n2 is the fraction of Yb ions in the excited state at this
point. It determines whether the fiber amplifies or absorbs:

```
n2 = (W_p + W_s) x tau / (1 + (W_p + W_ep + W_s + W_es) x tau)
```

where `tau = 0.83 ms` is the upper-state lifetime (measured Melkumov AS value).

**Interpretation:**
- The numerator (W_p + W_s) represents excitation pushing ions up.
- The denominator includes every process that moves ions up or down.
- At high pump intensity, n2 approaches a maximum set by the cross-section ratio.
- A strong signal pulls ions down by stimulated emission (W_es), reducing n2 —
  that is gain saturation.

The transparency threshold (where gain just equals absorption):

```
n2_transparency = sigma_a_signal / (sigma_a_signal + sigma_e_signal)
                = 0.0046 / (0.0046 + 0.30) ~ 0.015
```

Any n2 above ~1.5% means the fiber amplifies at 1064 nm — very low, which is why
Yb at 1064 nm is an efficient laser system.

#### Step D: Compute gain and update powers

The local gain coefficient [per meter] for the signal:

```
g_signal = Gamma_signal x N_Yb x (n2 x sigma_e_signal - (1 - n2) x sigma_a_signal)
```

This is the net gain: stimulated emission minus reabsorption, weighted by overlap
and ion density. The pump gain coefficient (negative = absorption):

```
g_pump = Gamma_pump x N_Yb x (n2 x sigma_e_pump - (1 - n2) x sigma_a_pump)
```

Each segment multiplies the powers by an exponential factor:

```
P_signal_new = P_signal x exp(g_signal x dz)
P_pump_new   = P_pump   x exp(g_pump   x dz)
```

Since g_signal is positive (amplification) and g_pump is negative (absorption),
the signal grows and the pump shrinks at each step.

#### Step E: ASE at this segment

ASE is not a single scalar. At each segment the solver evaluates a per-bin
spontaneous-emission source and steps 160 forward + 160 backward bins, all coupled
back into the inversion. The full equation set and the boundary-value solve are in
`ase.md`. The takeaway for the gain calculation here: **every one of the 322
channels (pump + signal + 160 forward + 160 backward ASE bins) appears in the
inversion equation**, so backward ASE in particular eats inversion that a
pump-and-signal-only model would miss — which is why the spectrally-resolved model
produces lower n2 (and lower gain) than a naïve scalar model in high-gain stages.

#### Solver-health flag

Instead of a hard gain cap, gain saturates naturally through ASE depletion of the
inversion. If the iteration nevertheless diverges (the ASE/signal fields run
away), the solver clamps the result and raises a generic `solver_failed` flag
(see `ase.md`) so the stage is reported as having no stable steady state rather
than producing untrustworthy numbers.

### 4.3 After the propagation loop — computing output values

**Gain:**

```
gain_linear = P_signal_out / P_signal_in
gain_dB     = 10 x log10(gain_linear)
```

**Peak power** scales by the same gain ratio as average power:

```
P_peak_out = P_peak_in x gain_linear
```

This works because the gain model uses average power (which sets the steady-state
inversion at high rep rate), and every pulse sees that same gain — so the peak
multiplier equals the average multiplier.

**Output pulse energy and average power:** `E_out = P_avg_out / f_rep`. The
repetition rate and pulse duration do not change through amplifiers or passive
components — only power and energy change.

**Pump absorption:**

```
pump_absorption_% = (1 - P_pump_residual / P_pump_input) x 100
```

**ASE ratio:**

```
total_ase_fwd = sum over bins of P_ASE_fwd(lambda)
ASE_ratio_dB  = 10 x log10(total_ase_fwd / P_signal_out)
```

This measures noise relative to signal; the acceptance threshold is < -20 dB (ASE
at least 100× below the signal). The spectral model also reports the peak ASE
wavelength and a solver-health flag. See `ase.md`.

---

## 5. Nonlinear Effect Thresholds

After each amplifier, the simulator checks whether the output **peak** power
exceeds the thresholds for destructive nonlinear effects. These checks use peak
power (not average) because nonlinear effects depend on instantaneous intensity.

### 5.1 Effective interaction length

Nonlinear effects accumulate over the fiber length, but not uniformly — the signal
is weaker at the input and stronger at the output. The effective length accounts
for this:

```
g_per_m     = ln(gain_linear) / fiber_length
L_eff_fiber = (1 - exp(-g_per_m x fiber_length)) / g_per_m
```

For short pulses, the interaction is also limited by how far the pulse extends
spatially:

```
L_pulse = c x tau / (2 x n_glass)
```

where `n_glass = 1.45` and the factor of 2 accounts for the counter-propagating
nature of SBS. The actual effective length used is the smaller of the two:

```
L_eff = min(L_eff_fiber, L_pulse)
```

(For an 8 ns pulse, `L_pulse = (3e8 x 8e-9) / (2 x 1.45) ≈ 0.83 m`.)

### 5.2 Stimulated Brillouin Scattering (SBS)

SBS is the most dangerous nonlinear effect for this system. High-intensity light
creates acoustic waves that act like a mirror, reflecting the signal backward.
Above threshold, most of the signal is reflected and can damage upstream
components.

**Effective Brillouin gain** (reduced by signal linewidth):

```
g_B_eff = g_B / (1 + dnu_signal / dnu_B)
```

where `g_B = 3e-11 m/W` (intrinsic Brillouin gain of silica) and `dnu_B = 35 MHz`
(Brillouin linewidth). For a 10 GHz example linewidth:

```
g_B_eff = 3e-11 / (1 + 10e9 / 35e6) = 3e-11 / 286.7 = 1.05e-13 m/W
```

**This is the key insight of the system design.** A 10 GHz seed linewidth reduces
the effective Brillouin gain by ~287× compared to a narrow-linewidth source.
Without this broadening, SBS would make the system impossible.

**SBS threshold:**

```
P_th_SBS = 21 x A_eff / (g_B_eff x L_eff)
```

The factor 21 is the critical SBS gain for silica fibers (~21 Nepers).

**Transient correction for short pulses:** when the pulse is shorter than ~5× the
phonon lifetime (T_phonon = 10 ns), the acoustic wave doesn't fully build up,
raising the threshold:

```
if tau < 5 x T_phonon:
    r = tau / T_phonon
    correction = r / (1 - exp(-r))
    P_th_SBS = P_th_SBS x correction
```

(For 8 ns pulses, r = 0.8 → correction ≈ 1.45, so the threshold is ~45% higher
than the steady-state value.)

**SBS ratio:** `SBS_ratio = P_peak_out / P_th_SBS`. Below 1.0 is SAFE; at or above
1.0 the signal will be reflected backward.

### 5.3 Stimulated Raman Scattering (SRS)

SRS transfers energy from the signal to a red-shifted wave at ~1116 nm. Unlike
SBS, it propagates forward and has a very broad bandwidth, so linewidth broadening
does NOT help.

**SRS threshold:**

```
P_th_SRS = 16 x A_eff / (g_R x L_eff)
```

where `g_R = 1e-13 m/W` (Raman gain coefficient) and 16 is the critical Raman gain
factor. **SRS ratio:** `SRS_ratio = P_peak_out / P_th_SRS`; same interpretation
(< 1.0 is SAFE). SRS generally has a higher threshold than SBS for this system.

### 5.4 Self-Phase Modulation (SPM)

The Kerr effect broadens the signal spectrum without transferring energy. The
accumulated nonlinear phase (B-integral):

```
B = gamma x P_peak x L_eff,   gamma = 2 x pi x n2_kerr / (lambda x A_eff)
```

with `n2_kerr = 2.6e-20 m²/W`. SPM broadens the linewidth, which is then fed back
into the SBS calculation for subsequent stages (a beneficial effect):

```
T0       = tau / 1.665
dnu_SPM  = 0.86 x B / (pi x T0)
dnu_new  = sqrt(dnu_old^2 + dnu_SPM^2)
```

Severity: B < 1 rad SAFE; 1 < B < π CAUTION; B > π DANGER (potential pulse
breakup).

---

## 6. The System Chain — Component by Component

The full default 3-stage MOPA system is a series of components, each transforming
the signal in turn. In chain order (from `examples/bgu_3stage_mopa.json`):

```
Seed
  -> Isolator (low-power)           protect seed from backward light
  -> Pump combiner 1                inject 976 nm pump for stage 1
  -> Amplifier 1                    5/130-type fiber  (gain)
  -> Isolator (high-power) 1        block backward ASE from stage 2
  -> Circulator 1 (port 1->2)       route toward the FBG
  -> Bandpass filter (FBG) 1->2     suppress the 1030 nm ASE peak
  -> Circulator 1 (port 2->3)       route toward stage 2 (double-pass topology)
  -> Mode field adapter 5/10        adapt beam to the larger core
  -> Pump combiner 2                inject pump for stage 2
  -> Amplifier 2                    10/125-type fiber (gain)
  -> Isolator (high-power) 2
  -> Circulator 2 (port 1->2)
  -> Bandpass filter (FBG) 2->3
  -> Circulator 2 (port 2->3)
  -> Mode field adapter 10/30       adapt beam to the 30 µm core
  -> Pump combiner 3                inject pump for stage 3
  -> Amplifier 3                    30/250-type fiber (gain) -> Output
```

A few structural points (the actual dB/gain values depend on the run — see
`summary.txt`):

- **Each circulator appears twice** — once for the 1→2 port pass and once for the
  2→3 port pass — because the BGU lab uses each circulator in double-pass mode
  around its FBG. The forward signal therefore traverses both port pairs, so both
  insertion losses are counted (see `CHANGELOG.md`, 2026-05-17 entry #5).
- **Each MFA updates the MFD** to match the next fiber's mode, so downstream A_eff
  (and the nonlinear thresholds) use the right area.
- **Each amplifier uses a progressively larger core** (5 → 10 → 30 µm) so the
  growing power stays below the nonlinear thresholds (larger A_eff → higher
  P_th_SBS and P_th_SRS).
- **The bandpass filters crush the 1030 nm ASE peak** before the next gain stage,
  so downstream amplifiers re-amplify only near-signal ASE (see `ase.md`).

Total amplifier gain is offset by the passive losses between stages — the price of
inter-stage protection (isolators and circulators). To see the realized gains,
losses, and powers for a given configuration, run the simulator.

---

## 7. V&V Acceptance Criteria

The simulator checks the final output against the project's verification and
validation requirements. The criteria (carried in the config's `requirements`
block) are:

| Parameter      | Acceptance criterion |
| -------------- | -------------------- |
| Wavelength     | 1064 ± 0.2 nm        |
| Spectral width | < 0.05 nm            |
| Rep rate       | 10 Hz – 100 kHz      |
| Pulse width    | 4 – 8 ns             |
| Average power  | 22 – 70 W            |
| Peak power     | ≥ 15 kW              |

Per-stage checks (must all pass): ASE < -20 dB, SBS ratio < 1.0, SRS ratio < 1.0,
solver reached a stable steady state.

The spectral-width criterion compares against the linewidth converted from Hz to
nm:

```
delta_lambda = (lambda^2 / c) x delta_nu
             = (1064e-9)^2 / 3e8 x 10e9 = 0.0377 nm   (for a 10 GHz linewidth)
```

Only requirement keys present in the config are evaluated; a config with no
`requirements` block produces no compliance table (see `CHANGELOG.md`, 2026-05-17
entry #2). The actual pass/fail for a given run is printed by `simulate.py`.

---

## 8. Lab Continuation Mode

When running with `--lab`, the simulator starts from a measured signal instead of
the seed. The user provides which stage was just completed, the measured average
power, the pulse parameters (with seed defaults), and pump powers for the
remaining stages. The simulator then:

1. Derives peak power and pulse energy from the measured average power
   (`E = P_avg / f_rep`, `P_peak = E / (tau x 0.94)`).
2. Estimates the ASE state at the handoff by running the theoretical model through
   the completed stages (ASE cannot be measured directly in the lab).
3. Sets the MFD to match the completed amplifier's fiber.
4. Propagates through only the remaining components.

This lets you compare predicted vs measured performance and adjust pump settings
for the next stages based on real measurements.

---

## Summary of All Equations

### Signal relationships
```
E = P_avg / f_rep
P_peak = E / (tau x 0.94)
```

### Passive component loss
```
P_out = P_in x 10^(-loss_dB / 10)
```

### Fiber geometry
```
V = (2 x pi / lambda) x r_core x NA
A_eff = pi x (MFD/2)^2
Gamma_signal = 1 - exp(-2 x (r_core / w)^2)
Gamma_pump = A_core / A_clad
N_Yb = alpha_clad / (4.343 x sigma_a_pump x Gamma_pump)
```

### Inversion and gain (per segment)
```
n2 = (W_p + W_s) x tau / (1 + (W_p + W_ep + W_s + W_es) x tau)
g_signal = Gamma_signal x N_Yb x (n2 x sigma_e_signal - (1 - n2) x sigma_a_signal)
P_signal_new = P_signal x exp(g_signal x dz)
```

### Nonlinear thresholds
```
g_B_eff   = g_B / (1 + dnu_signal / dnu_B)
P_th_SBS  = 21 x A_eff / (g_B_eff x L_eff)
P_th_SRS  = 16 x A_eff / (g_R x L_eff)
B         = gamma x P_peak x L_eff
L_eff     = min(L_eff_fiber, c x tau / (2 x n_glass))
ratio     = P_peak / P_th        (must be < 1.0 for SBS and SRS)
```

For the full ASE equation set and the bidirectional BVP solver, see `ase.md`.
