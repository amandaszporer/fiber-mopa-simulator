# The Pulse-Shape Plot (`pulse_shape.png`)

How the simulator generates the time-resolved pulse-profile figure, the physics
behind the data it shows, the exact code path that produces it, and the
literature that backs the method.

This is the only **time-resolved** figure the simulator emits. The other four
standard plots (`power_evolution`, `amplifier_details`, `nonlinear_margins`,
`ase_spectra`) are spatial or spectral; `pulse_shape.png` is the one that shows
the signal as a function of *time*.

---

## 1. What the plot shows

For every amplifier that ran the **Level-5 B2** time-dependent pulse stage, the
figure overlays two curves on a *power [W] vs. time [ns]* axis:

| Curve | Meaning | Source array |
|-------|---------|--------------|
| **Input @ z = 0** (blue) | the seed pulse entering the active fiber | `P_signal_tz[:, 0]` |
| **Output @ z = L** (red) | the amplified pulse leaving the active fiber | `P_signal_tz[:, -1]` |

One subplot is stacked per B2 amplifier. The title tags each as
`<name> pulse profile (Level 5 B2)`, and the legend reports each curve's peak
power. Rendered by `plot_pulse_shape()` in `utils/plotting.py:298`.

### How to read it

- **Amplitude:** the red curve is the blue curve multiplied by the (large) gain —
  this is the single-pass pulse amplification.
- **Reshaping / leading-edge steepening:** the output pulse is generally **not** a
  scaled copy of the input. The pulse front arrives first and sees the full,
  pump-built inversion, so it is amplified the most. As the pulse passes it
  *depletes* the inversion (stimulated emission), so the trailing edge sees less
  gain. The result is a pulse whose **peak shifts toward the leading edge** and
  whose shape is skewed forward — the classic signature of **gain saturation**
  during pulse extraction. The stronger the saturation (more stored energy
  extracted per pulse), the more pronounced the asymmetry.
- **Energy:** the area under each curve is the pulse energy; the difference is the
  energy the pulse extracted from the inversion
  (`pulse_energy_out − pulse_energy_in`, reported in the solver notes).

---

## 2. The physics behind the data

### 2.1 The two-part MOPA decomposition

A pulsed fiber MOPA operates on two widely separated timescales:

- **Inter-pulse (µs–ms):** the pump rebuilds the inversion `n₂(z)` and ASE grows.
  Handled by sub-regime **B1** (`_b1_inter_pulse`). **All ASE accounting lives
  here.**
- **Pulse (ns):** the signal pulse sweeps through and extracts the stored energy.
  Handled by sub-regime **B2** (`_b2_pulse`). **ASE is neglected** — over a 4–8 ns
  pulse it has no time to build up or drain the inversion.

The pulse-shape plot is purely a **B2 product**: it is the time-resolved signal
pulse from the last converged B2 cycle. (Spec: `docs/ase.md` §4.2; the standard
decomposition follows **Wang & Po 2003**, ref. 3 below.)

### 2.2 The B2 equations

During the pulse, B2 advances the signal and pump channels (only) coupled to the
inversion (`docs/ase.md` §4.2, B2):

```
(1/v_g) ∂P_k/∂t + ∂P_k/∂z = (g_k − α_bg) · P_k      (k ∈ {pump, signal})
        ∂n₂/∂t            = R_abs − R_em − n₂/τ
```

with the local gain coefficient

```
g_k(z,t) = Γ_k · N · [ n₂(z,t)·σ_e(λ_k) − (1 − n₂(z,t))·σ_a(λ_k) ]
```

and the boundary condition that injects the pulse:

```
P_signal(z = 0, t) = input_pulse_shape(t)   (Gaussian, FWHM = pulse_duration)
P_pump(z = 0, t)   = P_pump                 (CW — pump stays on through the pulse)
```

The Gaussian seed is built by `_gaussian_pulse_shape_offset()`
(`ase/solver_time.py:201`), normalised so its integral equals the shot energy and
its FWHM equals the configured pulse duration (consistent with the project's
`SHAPE_FACTOR = 0.94` peak-power convention, `components.py:59`).

### 2.3 Numerical method: Lax–Wendroff at CFL = 1 → method of characteristics

The grid is

```
Δz = L / (N_z − 1)
Δt = Δz / v_g          ← CFL = 1 (exactly one cell per time step)
```

where `v_g = c / n_glass ≈ 2.07 × 10⁸ m/s` is the group velocity. Choosing
**CFL = 1** is the key trick: the advection step becomes the **method of
characteristics** — the amplitude at `(z_i, t+Δt)` is exactly the upstream
amplitude at `(z_{i-1}, t)` — and the gain over the cell is applied with the
*exact* per-cell exponential

```
P_out = P_up · exp((g − α_bg)·Δz)
```

implemented as `P_up * (1 + expm1(g_net·dz))` for numerical accuracy at small
arguments (`ase/solver_time.py:328`, `_propagate`). This exactness is what makes
the small-signal limit reproduce the analytic Frantz–Nodvik / `G₀ = exp(g₀·L)`
gain (see validation, §4 below). The inversion is then advanced one step with
forward Euler using operator splitting (advect first, react second,
`ase/solver_time.py:365–374`); `Δt ≈ 36–72 ps ≪ τ = 0.83 ms`, so the explicit
step is stable.

The per-step signal profile across the whole fiber is recorded into
`P_signal_tz[k, :]` (`ase/solver_time.py:381`); the plot reads the first and last
columns of this `[n_t, n_z]` array.

### 2.4 Time window

The simulated window must cover both the pulse's temporal extent **and** the
fiber transit time, or the trailing edge never reaches `z = L`:

```
pulse_window = pulse_window_factor · pulse_duration + L / v_g
```

with `pulse_window_factor = 6` (±~3σ of the Gaussian, truncation error
< 0.1 % of the integrated energy; `ase/solver_time.py:283`).

---

## 3. The data pipeline (code path)

```
solve_time_dependent(mode="auto")                 ase/solver_time.py:396
  └─ period > 0.1·τ  → "periodic"  → _periodic_steady_state()   :458
        ├─ warm-start steady solve (homotopy)                    :507
        │     ⚠ if it does NOT converge → return early,          :521
        │        B1+B2 SKIPPED, no pulse data produced
        ├─ B1  _b1_inter_pulse()  → pre-pulse n₂(z)              :537
        └─ B2  _b2_pulse()        → P_signal_tz [n_t, n_z]       :563
                                     t, pulse_energy_in/out
                                     (carried on SteadyResult)
Amplifier.propagate()                              components.py:703
  └─ if result.t is not None and result.P_signal_tz is not None: :877
        info["t_ns"]            = result.t * 1e9                  :878
        info["P_signal_tz_W"]   = result.P_signal_tz             :879
        info["P_signal_out_t_W"]= result.P_signal_tz[:, -1]      :880
        info["P_signal_in_t_W"] = result.P_signal_tz[:, 0]       :881
plot_pulse_shape(sim, out_dir)                     utils/plotting.py:298
  └─ b2_amps = [a for a in amps
                if a.info.get("P_signal_out_t_W") is not None]   :310
     if not b2_amps: return (no file written)                    :312
     else: one stacked subplot per amp → pulse_shape.png         :316–333
```

The figure is requested unconditionally by `plot_all()`
(`utils/plotting.py:346`), but `plot_pulse_shape` **writes a PNG only if at least
one amplifier carries `P_signal_out_t_W`**. Otherwise it returns the path without
creating a file (mirroring `plot_bpf_spectra`'s no-op when there are no BPFs).

---

## 4. When the plot appears — and why it can be absent

The plot exists **iff the B2 stage actually ran to completion**. Two gates
control that:

1. **Rep-rate dispatch** (`solve_time_dependent`, `ase/solver_time.py:421`). With
   `mode="auto"`:
   - `period < 0.1·τ` (high rep) → `high_rep_quasi_cw` → delegates to the steady
     Mode A solver, which produces **no** time-resolved pulse → **no plot**. This
     is the normal BGU 100 kHz case (period 10 µs ≪ 0.1·τ = 83 µs).
   - `period ≥ 0.1·τ` (lower rep) → `periodic` → runs B1+B2 → **plot produced**.
   - `mode="full"` (CLI `--force-b2`) forces B1+B2 even at high rep.

2. **Warm-start convergence** (`_periodic_steady_state`, `ase/solver_time.py:521`).
   Even on the periodic branch, B1+B2 is **skipped** if the warm-start steady
   solve fails or does not converge:

   ```python
   if warmup.solver_failed or not warmup.converged:
       warmup.notes.append("B1+B2 skipped: the warm-start solve failed or did
                            not converge. No usable periodic steady state.")
       return warmup            # ← B2 never runs ⇒ no P_signal_tz ⇒ no plot
   ```

> **Worked example (this project's `single_amp` at 5 kHz).** Period = 200 µs
> > 0.1·τ = 83 µs, so the periodic branch is selected at *both* resolutions. At
> `num_segments = 1000` the fiber is spatially under-resolved for the high
> small-signal gain, the warm-start steady solve does **not** converge, B1+B2 is
> skipped (gate 2), and only 4 PNGs are written — **no** `pulse_shape.png`. At
> `num_segments = 4000` the warm-start converges in ~8 iterations, B1+B2 runs, B2
> emits the time-resolved pulse, and `pulse_shape.png` appears (5 PNGs). The plot
> showing up is therefore an independent confirmation that the full Level-5
> pipeline completed — not just that the `UNDER-RESOLVED` warning cleared.

---

## 5. Sources

The method and its references are documented in `docs/ase.md` (§4.2, §10.3,
§12). The ones specifically underpinning the pulse-shape computation:

1. **Wang, Y. & Po, H. (2003)** — "Dynamic characteristics of double-clad fiber
   amplifiers for high-power pulse amplification," *J. Lightwave Technol.*
   **21**(10), 2262. The time-dependent equation set and the B1 (inter-pulse ASE
   recovery) / B2 (ns pulse extraction) two-part decomposition.
   (`docs/ase.md` §12, ref. 3.)

2. **Frantz, L. M. & Nodvik, J. S. (1963)** — "Theory of pulse propagation in a
   laser amplifier," *J. Appl. Phys.* **34**(8), 2346. The analytic
   pulse-extraction / gain-saturation model that the saturated, forward-skewed
   output pulse reproduces; used as the small-signal validation limit
   (`docs/ase.md` §10.3, ref. 14).

3. **Giles, C. R. & Desurvire, E. (1991)** — "Modeling erbium-doped fiber
   amplifiers," *J. Lightwave Technol.* **9**(2), 271. The underlying gain /
   rate-equation formulation (`g_k`, `R_abs`, `R_em`), adapted to Yb
   (`docs/ase.md` §2–§4, ref. 1).

4. **Agrawal, G. P.** — *Nonlinear Fiber Optics* / *Applications of Nonlinear
   Fiber Optics*. Standard text for the `(1/v_g) ∂P/∂t + ∂P/∂z` transport form
   and CFL-stable finite-difference advection of optical pulses.

5. **Method of characteristics / Lax–Wendroff at CFL = 1** — standard
   computational-physics result that an upwind/Lax–Wendroff advection step at
   Courant number 1 carries the characteristic exactly; here combined with the
   exact per-cell gain exponential `exp((g−α_bg)·Δz)`. See *Numerical Recipes*
   Ch. 17 (`docs/ase.md` §12, ref. 15) for the finite-difference background.

6. **PyFiberAmp** (Rissanen, GPL-3.0, github.com/Jomiri/pyfiberamp) — the
   time-marching steady-state pattern and a cross-validation reference for the
   pulsed solver (`docs/ase.md` §12, ref. 11).

7. **Melkumov et al. (2004)** — arXiv:1502.02885 (FORC Preprint No. 5). The
   measured Yb aluminosilicate cross-sections `σ_a(λ)`, `σ_e(λ)` that set the gain
   the pulse experiences (`ase/data/cross_sections_yb.csv`; `docs/ase.md` §12,
   ref. 2).

---

*Generated as part of the single-amp under-resolution investigation. Code line
numbers refer to the repository state at the time of writing; re-check against
`utils/plotting.py`, `ase/solver_time.py`, and `components.py` if they drift.*
