# Changelog

This file documents notable changes to the simulator. Entries are
grouped by date, newest first. The 2026-05-17 entry covers the seven
fixes that consolidated the simulator from its initial Level 3
spectrally-resolved BVP into the current Level 5 layered solver.

---

## 2026-06-07

### Yb cross-sections: measured Melkumov AS dataset replaces hand-traced anchors

The Yb σ_a(λ)/σ_e(λ) spectra are now the **measured aluminosilicate (AS)**
cross-section table, replacing the previous 11 hand-read anchor values.

**Source.** M.A. Melkumov, I.A. Bufetov, K.S. Kravtsov, A.V. Shubin,
E.M. Dianov, *"Absorption and emission cross section of Yb3+ ions in Al2O3
and P2O5 doped fibers,"* FORC Preprint No. 5, Moscow, 2004; arXiv:1502.02885,
Appendix 2, AS columns. This is the closest published host-glass match for the
Coherent/Nufern SM-YDF-5/130-VIII fiber.

- **Provenance honesty.** The full numeric table comes from the **non-peer-
  reviewed preprint**. A peer-reviewed English/Russian journal version of the
  same study exists (Mel'kumov et al., *"Lasing parameters of ytterbium-doped
  fibres doped with P2O5 and Al2O3,"* Quantum Electronics 34(9), 843–848 (2004),
  DOI 10.1070/QE2004v034n09ABEH002688) but is a shorter presentation that
  **omits the Appendix-2 table** — so the table itself is not peer-reviewed,
  though a peer-reviewed version of the work exists. Both are cited in the CSV
  header, `ase_spec.md`, and `dopants.py`. The raw published-pm² transcription
  is kept at `ase/data/melkumov_AS_raw.csv` for provenance.
- **Units.** Published in pm²; the CSV stores m² via `σ_m2 = σ_pm2 × 1e-24`.
  `regenerate_csv` now only performs this unit conversion of the raw file (no
  anchors to regenerate from). The measured grid is stored AS-IS (~4 nm, 1 nm
  across the 968–986 nm peak, range 848–1180 nm); the magnitude is **not**
  renormalised — absolute scale cancels under the N_Yb back-derivation.
- **Lifetime.** τ 0.84 ms → **0.83 ms** (Melkumov AS), in `dopants.py` (the
  runtime-authoritative value, threaded via `DopantData.tau`); the vestigial
  `0.84e-3` defaults in `components.py` and `solver_steady.py` (overridden at
  build time) were bumped to match.

**The bug this fixes.** The hand-traced table had emission *decreasing*
1000→1030 nm; real Yb:silica has its dominant emission shoulder just past
1020 nm. The measured data restores σ_e(1030) > σ_e(1010) > σ_e(1000) (ratios
≈ 1.73 and 1.26), with σ_e/σ_a(976) ≈ 1.10 and σ_e/σ_a(1064) ≈ 65.

**Interpolation change: log-PCHIP → piecewise log-LINEAR.** The old log-PCHIP
existed to reconstruct a smooth curve from 11 sparse, widely spaced anchors
without spline overshoot. With a dense measured grid (1–4 nm) we only mildly
upsample onto the ~1 nm sim grid, so smoothness buys nothing. Log-linear is
monotone between nodes by construction → zero overshoot near the narrow 976 nm
peak (FWHM 7.7 nm) and the 1030 nm shoulder. σ is clamped to a positive floor
before `log()` (defensive — the AS data has no zeros). A single shared helper
(`cross_sections._log_linear_interp`) is now used by both `cross_sections.py`
and `dopants.py` so the two paths cannot diverge. Fail-loud out-of-range
behaviour is preserved (no extrapolation).

**Expected (intended) movement of gain-band numbers.** Every gain-band quantity
(signal gain, ASE, inversion) moves — this is the corrected physics, **not** a
regression; the prior BGU outputs were known-incorrect and are not a baseline.
Notable effects: at 976 nm σ_a 2.5 → 2.69 ×10⁻²⁴ m² (back-derived
N_Yb ≈ 9.5×10²⁵ ions/m³ for SM-YDF-5/130-VIII, matching Liekki Yb1200); the
915 nm pump σ_a is smaller (6.5 vs 8.0 ×10⁻²⁵), so 915-nm-pumped configs
back-derive a higher N_Yb and need finer spatial resolution to converge (the
`single_amp` steady-state test moved from n_z=1000 to n_z=2000; the physical
output is unchanged at ~0.36 W). Stage-1 ASE now runs higher in the full BGU
chain — a physical consequence of the corrected emission shape.

**Grid note (915 nm).** `spectral_grid` pump/signal scalar channels resolve
915 nm correctly now (data covers 848–1180 nm), but the ASE *bin* grid default
stays at [970, 1130] nm — widening `lambda_min` changes `n_bins` and every ASE
result, so it is left as an explicit caller choice (documented in
`spectral_grid.from_fiber`).

**Files**: `ase/data/cross_sections_yb.csv` (replaced),
`ase/data/melkumov_AS_raw.csv` (new), `ase/cross_sections.py`, `ase/dopants.py`,
`ase/spectral_grid.py`, `components.py`, `ase/solver_steady.py`,
`ase/tests/test_cross_sections.py`, `ase/tests/test_steady_solver.py`,
`docs/ase_spec.md`, `docs/ase_walkthrough.md`, `docs/physics_spec.md`,
`docs/computation_walkthrough.md`.

### BVP default initial guess: spontaneous-emission seed replaces zero ASE

The default initial guess for `solve_steady_state` (used when no `init=` warm
start is supplied) now **seeds the backward ASE field from spontaneous emission
instead of zero**. The forward ASE and pump profiles are seeded the same way; the
signal stays a flat guess.

- **Before**: the iteration started from a constant-power, zero-ASE trial
  (`P_ASE_bwd(0, λ) = 0` for all bins). With no ASE floor, a high-gain fiber's
  first forward sweep could amplify a negligible source into a runaway and walk
  the iteration into the **ASE-dominated basin** — a self-consistent but
  unphysical fixed point (the §13 quantum-defect-ceiling violation).
- **After** (`_spontaneous_emission_init`): a single pump-only forward sweep
  (signal = 0, ASE source off) gives an exact pump-only inversion profile
  n₂(z); the spontaneous-emission source S_ase(z) evaluated on that profile is
  accumulated gain-free from each facet to seed the forward **and backward** ASE
  fields. This is a physically-motivated lower-bound ASE floor that lands the
  production iteration in the **physical (signal-dominated) basin**. It degrades
  gracefully: at `P_pump = 0` the inversion is zero, so the seed collapses to the
  historical zero-ASE guess.

This is the solver's *first* line of defense against the multi-basin problem; the
three-layer robust wrapper (health check → Ren et al. 2015 homotopy → Xu et al.
2014 multistart, all documented under 2026-05-17) remains in place as fallback.

**Effect.** Where both seeds converge, the converged physical answer is
unchanged — e.g. `single_amp` (7 m / 0.9 W @ 915 nm, n_z=2000) gives 0.356 W
signal output from either seed. The change improves basin selection and
robustness in high-gain / coarsely-resolved regimes, where the zero seed is the
one that walks into the runaway basin. The backward-ASE seed at the input facet
is ~0.6 mW (vs 0.0 historically).

**Files**: `ase/solver_steady.py` (`_spontaneous_emission_init`, default `init`
branch of `solve_steady_state`), `docs/ase_spec.md` (§4.1 Mode A
iterative-shooting algorithm), `docs/ase_walkthrough.md` (§11, §13b).

---

## 2026-05-20

### B2 pulse solver no longer propagates ASE

Following a model-correctness review, the B2 (Lax-Wendroff pulse) step
now **neglects ASE entirely**. ASE accumulates on the µs–ms inter-pulse
timescale and cannot build up meaningfully during a 4–8 ns pulse, so
computing it there was both unnecessary and a small source of error.

This makes the implementation match the standard two-part MOPA model
that the architecture was already built around:

1. **B1** — pumping + ASE recovery over the inter-pulse interval →
   produces the inversion available to the next pulse.
2. **B2** — ASE-free pulse extraction → produces the amplified pulse
   energy and the residual inversion.

…iterated to periodic steady state. Only the placement of ASE changed;
the two-part split and the periodic iteration were already in place.

**Concretely**: `_b2_pulse` no longer advects the 320 ASE channels or
adds the spontaneous-emission source; it propagates the signal pulse
and the (CW) pump only, with the inversion driven by signal extraction.
`_absorption_emission_rates` made its ASE arguments optional (B1 still
passes them; B2 omits them). The dead `_B2Result` ASE fields and the
`SteadyResult.P_ase_fwd_tz` / `Amplifier.info["P_ase_fwd_tz_W"]`
during-pulse-ASE diagnostic were removed.

BGU at 100 kHz is unaffected — auto-dispatch routes to Mode A, so B2 is
never invoked there. Configs that do run B2 (`--force-b2`, low rep
rate) shift by a small amount, consistent with dropping a negligible
term.

**Files**: `ase/solver_time.py`, `ase/solver_steady.py`, `components.py`,
`ase/tests/test_time_solver.py`, `docs/ase_spec.md`,
`docs/ase_walkthrough.md`, `CLAUDE.md`.

---

## 2026-05-17

Seven changes landed in landing order. The first six fix specific
correctness issues; the seventh is a literature-backed solver
architecture (see `docs/qd_ceiling_research_brief.md` for the
research context that fed the design).

---

### 1. Yb cross-section silent extrapolation → fixed

**Symptom.** Running the framework against `examples/single_amp.json` (one
Isolator + one Amplifier, pumped at 915 nm) produced physically meaningless
output: 1 % pump absorption over 7 m of Yb 5/130 fiber, gain of 0 dB,
signal coming out essentially unchanged from the seed. No errors raised.

**Root cause.** `ase/data/cross_sections_yb.csv` covered only 970–1130 nm.
`DopantData` used `scipy.interpolate.PchipInterpolator` on `log(σ)`, which
silently *extrapolates* outside the data range. At 915 nm the extrapolated
cross-sections were 12 orders of magnitude off:

| λ      | σ_a (correct, §3.2)  | σ_a (extrapolated)   |
|--------|----------------------|----------------------|
| 915 nm | 8.0 × 10⁻²⁵ m²       | **1.5 × 10⁻¹³ m²**   |

The Amplifier back-derived `N_dopant` from that garbage σ_a value, so the
fibre ended up with effectively no dopants and no gain. The BGU 3-stage
example worked only because its 976 nm pump sits inside the CSV's range.

**Fix.**

1. Regenerated `ase/data/cross_sections_yb.csv` from **all 11** anchor
   values in `ase_spec.md §3.2` (Paschotta et al. 1997) — adds the 915 nm
   and 940 nm rows that were missing. New range 915–1130 nm, 216 rows.
   Added a reproducible `regenerate_csv()` function in
   `ase/cross_sections.py` and a `python -m ase.cross_sections` entry
   point.
2. Added a hard guard in `DopantData.sigma_a_at` / `sigma_e_at` /
   `interpolate_to` and the module-level `cross_sections.interpolate_to`
   so out-of-range wavelengths raise `ValueError` instead of silently
   extrapolating. `PchipInterpolator` now constructed with
   `extrapolate=False`.
3. Restored the 915/940 anchor rows in `test_cross_sections.py`'s round-
   trip test; added `test_out_of_range_raises`.

**Files**: `ase/cross_sections.py`, `ase/dopants.py`,
`ase/data/cross_sections_yb.csv`, `ase/tests/test_cross_sections.py`.

---

### 2. V&V requirements moved out of the framework → into the JSON

**Symptom.** `Simulator.check_requirements()` hardcoded the BGU project's
acceptance thresholds (4–8 ns pulse, 22–70 W avg, ≥ 15 kW peak, etc.).
Running the framework against a non-BGU config printed misleading FAIL
rows on every line.

**Fix.** `SystemConfig` gained an optional `requirements: dict` field.
`Simulator.check_requirements()` was rewritten to be data-driven: only
keys present in `requirements` are evaluated, and an empty block produces
no compliance table at all. BGU JSON now carries its acceptance criteria
in a `requirements` block placed after `metadata`. `single_amp.json` has
no `requirements` block, so it produces just the propagated output with
no pass/fail noise.

The compliance block follows the project's existing JSON conventions —
pure snake_case keys, SI units (`pulse_duration: 8e-9`, not
`pulse_duration_s: 8e-9`).

Schema:

```json
"requirements": {
  "wavelength":     {"target": 1.064e-6, "tolerance": 0.2e-9},
  "spectral_width": {"max": 0.05e-9},
  "rep_rate":       {"min": 10, "max": 100000},
  "pulse_duration": {"min": 4e-9, "max": 8e-9},
  "avg_power":      {"min": 22, "max": 70},
  "peak_power":     {"min": 15000},
  "amplifier": {
    "ase_ratio_dB_max":  -20,
    "sbs_ratio_max":      1.0,
    "srs_ratio_max":      1.0,
    "no_parasitic_lasing": true
  }
}
```

Unknown keys under `amplifier` raise `ValueError` instead of being
silently dropped (avoids a footgun where someone tries per-stage keying
with `{"AMP-1": {...}}` and gets no compliance rows).

**Files**: `framework.py`, `examples/bgu_3stage_mopa.json`,
`ase/tests/test_framework.py`.

---

### 3. Level 5 time-dependent ASE solver (B1 + B2)

**Context.** `ase/solver_time.py` was previously Level 4 in
`ase_spec.md`'s hierarchy: pump-only Mode A recovery + Frantz-Nodvik
per-slice extraction. The module docstring explicitly said the full
Lax-Wendroff B2 PDE was not implemented.

**Fix.** Implemented true Level 5:

- **B1 (`_b1_inter_pulse`)** — adaptive rate-equation time-stepping
  through the inter-pulse interval. Operator splitting: the P-field
  spatial profile is re-solved at fixed n₂ only when n₂ has drifted
  >5 % from the last solve; in between, the rate equation is integrated
  forward in time with adaptive dt (10 ns–10 µs, target ≤ 1 % Δn₂ per
  step). Replaces the previous "period ≫ τ" simplification.
- **B2 (`_b2_pulse`)** — Lax-Wendroff (z, t) PDE at CFL = 1
  (`Δt = Δz / v_g`). At CFL = 1 the upwind step reduces to the method of
  characteristics with an exact per-cell exponential transmission
  `exp((g − α_bg) · Δz)` plus an upstream-source term. Captures pulse-
  shape distortion, during-pulse ASE depletion, and gain saturation.
- **Periodic cycle** — `_periodic_steady_state` now alternates B1 and B2
  until pre-pulse n₂(z) is stable. Warm-started from a Mode A solve at
  the time-averaged signal power so the first pulse is on the attractor
  rather than climbing from zero.
- **Auto-dispatch** — at high rep (period < 0.1·τ) the dispatcher still
  delegates to Mode A (bit-identical). Below that threshold, the periodic
  B1+B2 cycle runs.
- **Time-resolved diagnostics** — `SteadyResult` gained optional `t`,
  `P_signal_tz`, `P_ase_fwd_tz` fields. `Amplifier.info` surfaces them
  as `t_ns`, `P_signal_out_t_W`, `P_signal_in_t_W`, `P_signal_tz_W`,
  `P_ase_fwd_tz_W`.
- **CLI** — `simulate.py --force-b2` forces the B1+B2 path even at high
  rep. With `--plots` it renders a new `pulse_shape.png` figure showing
  input vs amplified pulse temporal profile per amplifier.
- **Tests** — six new tests cover the Frantz-Nodvik analytic limit
  (`_frantz_nodvik_sweep` retained as the benchmark), CFL = 1 grid
  spacing, B1 long-period equivalence to pump-only Mode A, periodic
  convergence, energy conservation, and the Gaussian-pulse helper.

**Files**: `ase/solver_steady.py`, `ase/solver_time.py`, `components.py`,
`simulate.py`, `utils/plotting.py`, `ase/tests/test_time_solver.py`,
`ase_walkthrough.md` §15.

---

### 4. Default solver mode flipped to `time-dependent`

**Before.** Default was `"steady"` (Mode A). To get the auto-dispatched
Mode B (Level 5 at low rep, Mode A at high rep) the user had to pass
`--time-dependent`.

**After.** Default is `"time-dependent"`. At the BGU 100 kHz nominal
operating point, the auto-dispatcher still routes to Mode A and produces
bit-identical numbers in the same wall time — zero downside. At any
lower rep rate, the simulator now automatically uses the more physically
accurate Level 5 path without the user having to remember a flag.

Added `--steady` as the opt-out flag for cases where Mode A is wanted
explicitly. `--force-b2` still triggers the always-on B2 path with
pulse-shape diagnostics.

**Files**: `components.py` (`Amplifier.propagate(mode="time-dependent")`),
`framework.py` (`Simulator.run(mode="time-dependent")`), `simulate.py`
(CLI default + `--steady` flag), `README.md`, `CLAUDE.md`.

---

### 5. Circulator double-pass topology made explicit in the JSON

**Symptom.** The Seagnol PICIR-1064-3-B-15-NE datasheet (CIRC-1, S/N
25041511, IMG_5384.jpg) reports **two** insertion losses — Port 1→2 at
1.95 dB and Port 2→3 at 1.96 dB — but the simulator's `Circulator`
applied only one. In the BGU lab, each circulator is used in
**double-pass mode**: signal enters port 1, exits port 2 toward the FBG
(currently modelled as `BPF-12` / `BPF-23`), reflects back into port 2,
exits port 3 toward the next stage. The forward signal therefore
traverses **both port pairs**, and the simulator was under-counting
inter-stage loss by ~1.96 dB per stage (inflating the calculated output
gain).

**Fix.** No code change to the `Circulator` class — instead, the BGU
JSON now puts **two `Circulator` entries** around each FBG, one for each
port-pair pass, so the chain literally mirrors the physical layout:

Before:

```
... → ISO-HP-1 → BPF-12 → CIRC-1 → MFA-5/10 → ...
```

After:

```
... → ISO-HP-1 → CIRC-1 (1->2) → BPF-12 (FBG) → CIRC-1 (2->3) → MFA-5/10 → ...
```

CIRC-1 port-pair losses: 1.95 dB (1→2) and 1.96 dB (2→3) from the
datasheet. CIRC-2 keeps the 0.8 dB literature estimate for both passes
until a datasheet arrives. The `Circulator` docstring was refreshed to
make the "one port-pair pass" interpretation explicit.

**Impact on BGU output.** Stage 1 inter-stage loss rose by 1.96 dB
(stage-2 input drops from 130 mW to 83 mW), stage 2 by 0.8 dB. Both amps
are saturated, so they pick up extra gain (AMP-2: 17.2 → 19.1 dB; AMP-3:
12.7 → 13.5 dB) to partially compensate. Final output: **69.00 → 67.38 W
avg** (−0.10 dB total), **91.75 → 89.61 kW peak**. V&V table still
PASSes all rows except the marginal AMP-1 ASE boundary case.

**Files**: `components.py` (docstring), `examples/bgu_3stage_mopa.json`,
`ase/tests/test_framework.py` (new `test_bgu_example_double_pass_circulator_topology`),
`CLAUDE.md`.

---

### 6. NaN-guard for the periodic B1+B2 cycle when Mode A barely converges

> **Status after fix #7 (added later in the same session):** Fix #6's
> NaN-guard remains in place as a defensive backstop. The
> `_periodic_steady_state` warmup was *also* upgraded as part of fix #7
> to use `solve_steady_state_homotopy` instead of the raw shooter — so
> the warmup itself now avoids the unphysical fixed point in most
> cases, and the NaN-guard only fires on genuinely past-parasitic
> configurations where no physical CW steady state exists. See fix #7
> for details on why the warmup uses homotopy (not the full robust
> wrapper): the robust wrapper's Layer 3 calls back into
> `solve_time_dependent(mode="full")`, which would recurse into
> `_periodic_steady_state`; homotopy alone has no such cycle.

**Symptom.** After the default-mode flip, running the (still over-pumped)
`single_amp.json` at length = 7 m / 3 m produced NaN through the chain.
Mode A's warmup was *just* on the convergent side of the parasitic
threshold — `parasitic_lasing = False`, `converged = True` — so the
pre-existing bailout (which only triggers on those two flags) let
B1+B2 run, where high-n₂ regime spatial sub-solves overflowed during
the rate-equation step.

**Fix.** Two finite-value checks in `_periodic_steady_state`:

1. After every B1 call, if `b1.n2_z` or `b1.P_ase_fwd_z` has any non-
   finite values, return the Mode A warmup result with a note —
   "B1 numerical blowup ... system on the parasitic-lasing edge — Mode A
   barely converged but B1's high-n₂ regime overflows."
2. Same check after every B2 call.

Mode A's warmup runaway clamp already produces clamped but finite values
for these regimes, so falling back to it is the right behaviour. The
report still shows `[LASING]` / `[CAPPED]` / `NOT CONV` markers from the
warmup so the user knows the system is unphysical.

**Files**: `ase/solver_time.py`.

---

### 7. Layered robust steady-state solver with energy-conservation diagnostics

**Symptom.** When `simulate.py --config examples/single_amp.json` ran
with `length=3 m` (after the user shortened the over-long 7 m default to
move clear of obvious parasitic-lasing), Mode A reported a *fully
converged* solution: `converged=True`, `parasitic_lasing=False`,
80 iterations to `tol=1e-5`. The output: 3.16 W of signal from 0.587 W
of absorbed 915 nm pump light. The quantum-defect ceiling
`(λ_pump / λ_signal) × pump_absorbed = 0.86 × 0.587 = 0.505 W` says the
maximum physical output is 505 mW. The simulator was reporting **6.26×
the QD ceiling** without raising a single flag.

For the BGU 3-stage example we have empirical baselines: AMP-1 lands at
67.7 % of the QD ceiling, AMP-2 at 96.5 %, AMP-3 at 98.5 %. Healthy
regimes stay comfortably under 1.0; the 6.26× violation is a different
animal entirely.

**Root cause (research, see `docs/qd_ceiling_research_brief.md`).** The
Giles-Desurvire iterative-shooting BVP with 322 channels (1 pump +
1 signal + 160 fwd ASE + 160 bwd ASE) is a nonlinear two-point BVP
with strong exponential sensitivity. The rate-equation closure
`n₂ = R_abs / (R_abs + R_em + 1/τ)` enforces *per-ion photon balance*
locally at every z, but global *energy* conservation
(`signal_gain ≤ QD · pump_absorbed`) is an independent constraint that
**is not** automatically enforced by the iteration. In healthy regimes
the BVP has a unique fixed point and the iteration finds it; near the
parasitic-lasing edge **the BVP admits multiple self-consistent fixed
points** and the iteration converges to whichever one the initial guess
flows toward — sometimes the physical signal-clamped basin, sometimes
an unphysical ASE-dominated basin. This phenomenon is documented
operationally — though never formally named — in:

- **Paschotta**, "Tutorial Fiber Amplifiers, Part 3," RP Photonics
  Encyclopedia: *"The real trouble comes when you have multiple
  counterpropagating waves… one can generalize the mentioned shooting
  algorithm, but that involves multi-dimensional root finding. The
  involved exponential dependencies don't make that easier."*
- **Ren, Han, Liu et al.**, "Numerical methods for high-power Er/Yb-
  codoped fiber amplifiers," *Opt. Quantum Electron.* **47**(7),
  2199–2212 (2015), DOI 10.1007/s11082-014-0096-8 — describes the
  multi-fixed-point problem in EYDFAs and gives the homotopy-continuation
  recipe we adopted.
- **scipy.integrate.solve_bvp** documentation, BVP-with-two-solutions
  example: *"This problem is known to have two solutions. To obtain both
  of them, we use two different initial guesses for y."*

**Fix — three-layer cascade in `ase/solver_time.py::solve_steady_state_robust`.**

`Amplifier.propagate(mode="steady")` now calls the wrapper instead of
the bare shooter. Three layers, each invoked only if the previous one
fails an energy-conservation check:

### Layer 1 — Direct solve + health diagnostics

After every solve, compute three diagnostics in `ase/solver_health.py`:

1. **Energy residual** — global photon-balance check derived from
   Ren et al. 2015 §2:
   `(signal_gain + ASE_out_total + bg_loss − QD · pump_absorbed) / (QD · pump_absorbed)`.
   Healthy values are ≤ 0 (the deficit goes to QD heat and internal
   ASE cycling). Classification thresholds:
   - `≤ 0.01` → `ok`
   - `(0.01, 0.05]` → `warning`
   - `> 0.05` → `violation`

   These thresholds are **engineering judgement** — the open
   literature does not publish a canonical value for QD-ceiling
   tolerances. Justification: 5 % is ~50× the combined RK4-truncation
   + iteration-tolerance + Giles-parameter-uncertainty floor (~10⁻³),
   and well above the worst observed healthy case (BGU AMP-3 at
   98.5 % of the ceiling).

2. **ASE conversion fraction** `η_ASE = ASE_total / pump_absorbed`.
   Classification:
   - `≤ 0.10` → `amplifier`
   - `(0.10, 0.30]` → `mixed`
   - `> 0.30` → `sfs` (superfluorescent source regime)

   Justification: **Wang & Clarkson**, "110 W double-ended cladding-
   pumped Yb-doped fiber superfluorescent source," *Opt. Lett.* **31**,
   3116 (2006), demonstrated a physical Yb fibre SFS at 68 % slope
   efficiency where η_ASE → 1. **Dong**, "High-energy 1-µm Yb-doped
   pulsed fiber amplifier with… ," *Front. Phys.* **13**, 1539099
   (2025), DOI 10.3389/fphy.2025.1539099, uses η_ASE-based thresholds
   operationally for pulsed amplifiers (their pulsed-amp working
   threshold corresponds to η_ASE ≈ 0.3 in their setup).

3. **Small-signal `g₀·L`**, where `g₀` is evaluated at the pump-only
   equilibrium inversion. Classification:
   - `≤ 30` → fine for fibre
   - `> 30` → `high_gain` flag

   Justification: **Furuse et al.**, "Total-reflection active-mirror
   laser… [Yb:YAG thin disk]," PubMed 23736565, calibrated
   `g(0)·l_ASE ≈ 3` as the thin-disk threshold where ASE transitions
   from spontaneous to inversion-draining. Fibre amplifiers operate
   routinely at `g₀·L` of 10–30 because the doped core occupies only
   a small fraction of the mode (Γ_signal mismatch); we flag at 30 to
   stay conservative for fibre geometries.

### Layer 2 — Ren et al. 2015 homotopy continuation

If Layer 1 reports `violation`, retry with the published recipe in
`solve_steady_state_homotopy`. Each step uses the previous solve as
the `init=` initial guess (newly re-added parameter on
`solve_steady_state`):

1. **Pump-only solve** — `signal=0`, `m_pol=0` (no spontaneous-ASE
   source). Unique fixed point.
2. **Add signal, ASE source still off** — Beer-Lambert-like; still
   unique.
3. **Re-enable ASE source.** Critical step — starts in the signal-
   clamped basin instead of the ASE-dominated one. If this passes the
   health predicate, return.
4. **Parameter continuation** — scale pump from 50 % → 75 % → 100 %
   of the target, each with the previous solve as warm start. This
   is Ren et al.'s explicit recommendation: *"to find a low-gain
   EYDFA that the artificially constructed P_i is good enough for
   its convergence, we can lower the gain of the amplifier by
   shortening the fiber length or decreasing the pump power. Once a
   low-gain EYDFA is successfully solved, the fiber length and/or
   pump power can be gradually increased with better initial
   guesses constructed from the successive solutions."*
5. **Xu et al. 2014 linear-gain-shape multistart** — final fallback,
   three slope efficiencies (0.3, 0.5, 0.7) per **Xu et al.**,
   "Excellent initial guess functions for simple shooting method
   in Yb³⁺-doped fiber lasers," *Optik* (2014) pii S1068520014000546,
   verbatim: *"the critical guess value of slope efficiency is less
   than 0.3 for all the fiber length, Yb3+-doped concentration,
   signal reflectivity and pump power."*

Verified empirically: for `single_amp.json` (3 m / 1 W / 1 mW seed)
the homotopy converges at step 4 (50 % pump parameter continuation)
to a physical 680.01 mW output at 80 % pump absorption.

### Layer 3 — Time-marching arbiter (existing B1+B2)

If Layer 2 still violates, dispatch to
`solve_time_dependent(mode="full")` — the Level 5 B1+B2 cycle we built
in fix #3. The time-dependent IVP has a **unique attractor** by
construction (it's a forward-integrated ODE, not a BVP), so it cannot
land on an energy-violating fixed point. This is the same pattern
**PyFiberAmp** (Rissanen, GPL-3.0, github.com/Jomiri/pyfiberamp)
recommends in its docs: *"With constant input powers, the result
converges to the steady state simulation result… Larger (and
physically unrealistic) time steps can be used to drastically speed
up the convergence of steady state simulations."* If B1+B2 itself
bails (parasitic guard fires), no physical CW steady state exists at
all — we return Mode A's runaway-clamped result with
`solver_path_used="all_failed"`.

### Interaction with fix #6 (B1+B2 NaN-guard)

Fix #6's NaN-guard inside `_periodic_steady_state` is **not** replaced
by this work — it's deliberately retained as a defensive backstop.
There is, however, a circular-dependency issue: Layer 3 of the robust
wrapper calls `solve_time_dependent(mode="full")`, which calls
`_periodic_steady_state`, which (before this fix) called the raw Mode A
solver for its warmup. If we made the warmup call the *full* robust
wrapper, Layer 3 would recurse infinitely.

The fix: `_periodic_steady_state`'s warmup now calls
`solve_steady_state_homotopy` directly (the Layer-2 piece of the robust
cascade). Homotopy by itself doesn't invoke `solve_time_dependent`, so
there's no recursion. The warmup gets the same multi-step continuation
protection as the steady-state path, and fix #6's NaN-guard only fires
on genuinely past-parasitic configurations where even homotopy can't
find a physical answer.

### Layer 4 — PyFiberAmp cross-validation benchmark

For external validation, added `validation/test_pyfiberamp_agreement.py`
which runs our solver on **PyFiberAmp's canonical Yb amplifier example**
(2 mW signal @ 1035 nm, 300 mW pump @ 976 nm, 2.5 m Yb 5/130 fibre) and
compares to a recorded PyFiberAmp output. The reference JSON
`validation/pyfiberamp_canonical.json` is populated on demand via
`validation/generate_pyfiberamp_reference.py` (requires
`pip install pyfiberamp`); the test skips cleanly when PyFiberAmp
isn't installed locally. 5 % agreement threshold — tighter would
require aligning σ-data tables and ASE-bin centroids exactly with
PyFiberAmp's internal choices.

### What the user sees

`Amplifier.info` gains 7 new keys: `energy_residual_ratio`,
`ase_conversion_fraction`, `small_signal_g0L`, `energy_status`,
`regime`, `solver_path_used`, `homotopy_steps_used`.

The text report adds a compact `[E:.. R:.. path=..]` decoration on the
gain line whenever the solve is **not** in the clean
"direct + healthy + amplifier" case:

```
[AMP-1]  Gain: 27.6 dB  (576.2x)                                            ← BGU stage 1, clean
[]       Gain: 30.3 dB  (1077.7x)  [E:WARNING ×1.01  R:AMPLIFIER  path=homotopy(4)]    ← single_amp post-fix
[]       Gain: 18.0 dB  (63.1x)    [E:VIOLATION ×5.30  R:SFS  path=all_failed]         ← genuinely parasitic
```

### Impact on the two example configs

**BGU 3-stage**: bit-identical to the pre-fix output (67.38 W avg,
89.61 kW peak). All three amps take the **direct** Layer-1 path —
homotopy never invoked, no decoration in the report.

**single_amp 3 m**:
- Before: silently produced 3.16 W (6.26× over the QD ceiling).
- After: **680.01 mW** (within 1 % of the QD ceiling). Path:
  `homotopy(4)`, decoration `[E:WARNING ×1.01  R:AMPLIFIER]`. ASE at
  −21.8 dB SAFE, 80 % pump absorption — a physically sensible
  superfluorescent-edge amplifier.

### Sources cited in code + walkthrough

| # | Reference | Used for |
|--:|---|---|
| 1 | Ren, Han, Liu et al., *Opt. Quantum Electron.* **47**(7), 2199 (2015), DOI 10.1007/s11082-014-0096-8 | Layer 2 homotopy continuation recipe |
| 2 | Xu et al., *Optik* (2014) pii S1068520014000546 | Linear-gain-shape multistart with slope efficiency ≥ 0.3 |
| 3 | Furuse et al., PubMed 23736565 | `g(0)·l ≈ 3` ASE-dominated threshold (adapted to >30 for fibre) |
| 4 | Dong, *Front. Phys.* **13**, 1539099 (2025), DOI 10.3389/fphy.2025.1539099 | η_ASE operational thresholds for pulsed amplifiers |
| 5 | Wang & Clarkson, *Opt. Lett.* **31**, 3116 (2006) | Physical Yb fibre SFS at 110 W / 68 % slope efficiency — validates the SFS regime classification |
| 6 | PyFiberAmp (Rissanen, GPL-3.0, github.com/Jomiri/pyfiberamp) | Time-marching `DynamicSimulation(stop_at_steady_state=True)` pattern (Layer 3); canonical Yb benchmark (Layer 4) |
| 7 | Paschotta, RP Photonics tutorials | BVP multi-fixed-point phenomenon (motivation) |
| 8 | Kierzenka & Shampine, *ACM TOMS* (2001) | Algorithm reference for `scipy.solve_bvp`'s collocation; cited as the alternative we evaluated and did not adopt |

**Files**: `ase/solver_health.py` (new), `ase/solver_steady.py` (re-added
`init=`, added `solve_steady_state_homotopy`, extended `SteadyResult`),
`ase/solver_time.py` (added `solve_steady_state_robust`),
`components.py` (`Amplifier.propagate` calls robust wrapper + surfaces
7 health keys), `framework.py` (`_health_tag` decoration on gain line),
`ase/tests/test_solver_health.py` (new, 5 unit tests),
`ase/tests/test_steady_solver.py` (4 new tests), `ase/tests/test_framework.py`
(2 new tests), `validation/pyfiberamp_canonical.json` (new),
`validation/test_pyfiberamp_agreement.py` (new),
`validation/generate_pyfiberamp_reference.py` (new), `ase_walkthrough.md`
(new §13b), `docs/qd_ceiling_research_brief.md` (status header marked
RESOLVED), `CLAUDE.md`.

---

### Test count summary

- Start of session: 38 tests passing.
- End of session: **58 passing, 1 skipped** (PyFiberAmp cross-validation,
  conditional on local PyFiberAmp install).
- New tests added by topic:
  - Cross-section guard: 1 (`test_out_of_range_raises`).
  - V&V externalisation: 2
    (`test_simulator_without_requirements_skips_compliance`,
    `test_unknown_amplifier_requirement_key_raises`).
  - Level 5 ASE solver: 6 (CFL=1, small-signal-gain, energy-conservation,
    B1 long-period equivalence, periodic-convergence, Gaussian pulse).
  - Double-pass circulator topology: 1
    (`test_bgu_example_double_pass_circulator_topology`).
  - Layered robust solver: 11 (5 in `test_solver_health.py` for the
    diagnostics; 4 in `test_steady_solver.py` for homotopy + robust
    wrapper; 2 in `test_framework.py` for BGU direct-path integration).
  - PyFiberAmp cross-validation: 1 skipped (conditional).
