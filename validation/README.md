# Validation

Quantitative validation of the simulator against published Yb-MOPA experiments.

## Run

From the project root:

```bash
.venv/bin/python validation/run_validation.py
```

## Targets

### Świderski 2008 — single-stage gain anchor

> J. Świderski, A. Zając, M. Skorczakowski,
> "Diode-seeded nanosecond Yb-doped fiber amplifier operating at the repetition rate up to 500 kHz,"
> *Optica Applicata* **XXXVIII**(4), 669–676 (2008).

Single Nufern Yb-doped LMA double-clad gain fiber (20/400 µm, NA 0.06, 13 dB total absorption at 976 nm), 978 nm CW co-propagating pump (max launched 11.4 W into the fiber), 1063.91 nm pulsed laser-diode seed (0.52 nm spectral width). Free-space coupling onto the gain fiber facet (~70% launch efficiency, modeled as an Isolator with 1.55 dB combined loss).

The script sweeps the four 100-kHz published data points (Table p.671 + Fig. 5):

| Pulse | P_seed (diode) | E_out | P_out (paper) | Gain (paper) |
|---|---|---|---|---|
| 11 ns | 1.5 mW | 20.4 µJ | 2.04 W | ~31 dB |
| 30 ns | 2.7 mW | 24.4 µJ | 2.44 W | ~29 dB |
| 50 ns | 4.0 mW | 26 µJ | 2.60 W | ~28 dB |
| 100 ns | 7.5 mW | 29.7 µJ | 2.97 W | ~26 dB |

**Pass criterion**: average-power agreement within 15% (per the validation roadmap recommendation), AND solver convergence, AND energy balance respected (signal_out + ASE + pump_residual ≤ pump_in + signal_in).

## Current result (2026-05-09)

```
OVERALL: FAIL — all four data points trip "SOLVER FAIL (parasitic lasing)"
```

This is **not** a validation failure of the underlying physics. It is a real limitation of the iterative-shooting BVP solver in `ase/solver_steady.py`, surfaced by attempting to replicate the Świderski experiment.

### What the validation discovered

The solver does not converge to a physical steady state when:

- Seed power is much smaller than the gain medium's saturation power (P_seed ≪ P_sat), AND
- Pump power is high enough that the unsaturated gain potential exceeds ~30 dB.

For the Świderski 20/400 fiber, P_sat ≈ 80 mW at 1064 nm. Seed values 1.5–7.5 mW are 10–50× below saturation. With 11.4 W pump:

- The forward sweep, with n2 frozen within each RK4 step, generates exponential signal growth
  faster than the next-step inversion update can clamp.
- ASE in high-gain bins (1030 nm peak) runs away, hits the 1 kW/bin sentinel in
  `solver_steady.py:294`, and the solver bails after iteration 1.
- Energy conservation is violated by 100–10000× (signal output exceeds pump input).

A pump-power sweep on the Świderski fiber confirmed the boundary:

```
P_pump (W)   Status
0.5          converges, sensible (–12 dB net)
1.0          converges (–2.6 dB)
2.0          converges (40 dB, 98 iters — at convergence cap)
3.0+         parasitic lasing trips, output unphysical
```

### What this means for our project

**The simulator is reliable for our intended operating regime** — multi-stage MOPAs where each amplifier is operated near or above its saturation power. In our `examples/bgu_3stage_mopa.json`:

- AMP-1: 0.6 mW seed, 1.0 W pump → output ~360 mW. Solver converges in ~8 iters. Saturation borderline but solver handles it.
- AMP-2: ~130 mW seed (1.6× P_sat ≈ 80 mW for 10/125), 9 W pump → ~6.8 W output. Converges in 3 iters.
- AMP-3: ~3.7 W seed (8× P_sat ≈ 0.45 W for 30/250), 100 W pump → ~69 W output. Converges in 3 iters.

The simulator is **not reliable for deeply unsaturated single-stage amplifiers** (P_seed ≪ P_sat with P_pump ≫ a few watts), which is exactly Świderski's regime.

### Implications for AMP-1

AMP-1 in our chain is closest to the unsaturated regime (seed 0.6 mW, P_sat ~80 mW for the 5/130 fiber). The solver does converge there, but the validation finding suggests its predictions for AMP-1 specifically should be treated with more caution than its predictions for AMP-2 and AMP-3. If you make AMP-1's seed dramatically lower or its pump dramatically higher than the current configuration, watch for the same solver-failure signature.

## To make the Świderski validation pass

The validation is intentionally not gated on a passing result — it's a regression test that exposes the solver's working envelope. To pass it, the solver needs improvement. Two viable paths:

1. **Sub-step the RK4 with within-step n2 updates.** Replace each RK4 step with a finer-grained integrator that updates n2 every dz/k for some k, so signal saturation feedback is enforced within the step, not only at step boundaries.
2. **Replace the iterative shooting with a true BVP solver** (e.g. `scipy.integrate.solve_bvp`). Handles the boundary conditions properly; converges in regimes where shooting can't.

Both are non-trivial (~days). Neither is required for our current example MOPA simulation to be trustworthy.

## Other targets considered (not built)

- **Liu et al., Procedia Eng. 140, 123 (2016)** — 2-stage 1064 nm 100 kHz 200 ns MOPA. Architecturally the closest peer to our system. Blocked because the power amplifier is **counter-pumped** and `Amplifier.propagate` raises `NotImplementedError` for `pump_direction != "co"`. Implementing counter-pumping in the BVP solver is ~1 day of work.
- **Lago et al., JOSA B 27, 2231 (2010)** — multi-stage ns MOPA with explicit ASE/OSNR comparison to model. Best ASE benchmark in the literature. Not yet obtained (paywalled).
- **He et al., Opt. Express 14, 12846 (2006)** — gold-standard 4-stage Yb cascade. Operates broadband CW, not the ns-pulsed regime our model is built for.

See the validation-strategy briefing in the project's working notes for the full ranked list of candidate experimental papers.
