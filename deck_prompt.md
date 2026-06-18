# Claude Design Prompt — Yb-Doped Fiber MOPA Simulation Results Deck

Paste everything below the line into Claude (design / slides artifact). It already
contains every number and the exact file path of every figure to embed. **All
plot paths are absolute on this machine** — drag the referenced PNGs in, or keep
the paths as captions so the slides can be assembled with the real images.

> Project: Yb-doped fiber MOPA simulator (BGU engineering project p-2026-158).
> Pulsed 1064 nm Master Oscillator Power Amplifier. These are **simulation
> results** from the corrected high-resolution (4000-segment) solver. **Ignore
> verification/validation entirely — this deck is about the results we got.**

---

## ROLE & GOAL

You are a presentation designer. Build a clean, technical results deck (16:9)
for a fiber-laser engineering audience. Visual style: dark-on-light, one accent
color, generous whitespace, large readable plots, minimal text per slide
(headline + 3–5 bullet key points + the figure). Use a monospace font for
numbers. No verification/validation content — focus on the physics results.

Each figure referenced below lives at an absolute path under
`/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/`. Embed the named PNG
on its slide; if you cannot load it, place a labelled image placeholder that
prints the full path so it can be dropped in later.

---

## DECK STRUCTURE

### Slide 1 — Title
- **Yb-Doped Fiber MOPA — Simulation Results**
- Subtitle: 3-stage 1064 nm pulsed amplifier chain + single-stage pump study
- Footnote: spectrally-resolved bidirectional ASE solver, 4000 z-segments

### Slide 2 — Executive summary (results headline)
Key points:
- **BGU 3-stage chain delivers 62.5 W average, 83.2 kW peak** at 100 kHz / 8 ns,
  1064 nm — within target spec.
- Single-stage pre-amp study: output scales **25.5 → 28.3 dB gain** as pump
  rises 0.7 → 1.0 W; **ASE worsens with pump** (−15.8 → −12.5 dB).
- Co- vs counter-pumping the BGU chain: **negligible difference** at this
  operating point (62.54 vs 62.55 W).
- Nonlinear thresholds (SBS, SRS) **safe everywhere**; ASE is the dominant
  limiter, concentrated in the high-gain first stage.

---

## PART A — BGU 3-STAGE MOPA (the system architecture)

Shared operating point: seed 750 µW avg / 7.5 nJ / 100 kHz / 8 ns / 10 GHz
linewidth → output **62.5 W avg, 83.2 kW peak, 625 µJ**. Per-stage gains
**26.8 / 19.6 / 13.3 dB**. Output forward ASE 16.3 mW.

### Slide 3 — BGU (co-pump): ASE spectrum of the architecture
- **Figure:** `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/bgu_example/ase_spectra.png`
  (3 stacked panels: forward ASE = orange, backward ASE = red dashed, per stage)
- Key points:
  - Stage 1: gain **26.8 dB**, ASE **−14.5 dB (dominant — high-gain pre-amp)**,
    57% pump absorbed.
  - Stage 2: gain **19.6 dB**, ASE **−32.2 dB**, SBS 0.33 / SRS 0.60 — safe.
  - Stage 3 (power amp): gain **13.3 dB**, ASE **−35.8 dB**, SBS 0.39 / SRS 0.70.
  - In-band ASE peak at 1064 nm in stages 2–3 = filtered ASE reamplified by the
    next stage (not the signal).

### Slide 4 — BGU (co-pump): power & gain evolution
- **Figure:** `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/bgu_example/power_evolution.png`
- Key points: avg & peak power climbing across the chain; integrated forward ASE
  per component; final 62.5 W / 83.2 kW.
- (Optional supporting figure on same or next slide:)
  `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/bgu_example/amplifier_details.png`

### Slide 5 — BGU (counter-pump): ASE spectrum of the architecture
- **Figure:** `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/bgu_example_counter_pump/ase_spectra.png`
- Key points: per-stage gains identical to co-pump (26.8 / 19.6 / 13.3 dB);
  output 62.55 W / 83.18 kW; ASE −14.5 / −32.1 / −35.8 dB.

### Slide 6 — COMPARISON: BGU co-pump vs counter-pump
Two-column layout. Left = co-pump, right = counter-pump.
- Embed both spectra side by side:
  - `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/bgu_example/ase_spectra.png`
  - `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/bgu_example_counter_pump/ase_spectra.png`
- Comparison table:

  | Metric | Co-pump | Counter-pump |
  |---|---|---|
  | Output avg power | 62.54 W | 62.55 W |
  | Output peak power | 83.16 kW | 83.18 kW |
  | Output pulse energy | 625.4 µJ | 625.5 µJ |
  | Stage gains | 26.8 / 19.6 / 13.3 dB | 26.8 / 19.6 / 13.3 dB |
  | Stage ASE | −14.5 / −32.2 / −35.8 dB | −14.5 / −32.1 / −35.8 dB |
  | Output fwd ASE | 16.31 mW | 16.35 mW |

- Takeaway: **pump direction makes no practical difference** at this operating
  point — the inversion is set by total absorbed pump, and absorption is near-
  complete either way.

### Slide 7 — BGU pulse shape (time-resolved, forced B1+B2)
- **Figure:** `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/bgu_example_forceb2/pulse_shape.png`
  (3 panels: input blue vs amplified output red, per stage)
- Key points:
  - Per-stage peak pulse power: **0.74 → 363 W**, **83 → 7,478 W**,
    **3,420 → 71,873 W**.
  - Output 61.9 W / 82.4 kW (B1+B2 path; matches quasi-CW average to <1%).
  - Counter-pump pulse shape available at
    `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/bgu_example_counter_pump_forceb2/pulse_shape.png`.

---

## PART B — SINGLE-STAGE PUMP-POWER STUDY

Shared setup: 7 m Yb fiber, seed 1 mW avg / 200 nJ / 5 kHz / 300 GHz linewidth,
co-pump at 915 nm. Pump swept 0.7 → 1.0 W. ~89% pump absorbed in all cases;
solver converged in 8–10 iterations.

### Slides 8–11 — One spectrum slide per pump power
For each pump power, a slide titled "Single-amp — pump = X W" with:
- **Figure (spectrum):**
  - 0.7 W → `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/single_amp_pump_0p7W/ase_spectra.png`
  - 0.8 W → `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/single_amp_pump_0p8W/ase_spectra.png`
  - 0.9 W → `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/single_amp_pump_0p9W/ase_spectra.png`
  - 1.0 W → `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/single_amp_pump_1p0W/ase_spectra.png`
- Key points (use the row from the table below): gain, output avg/peak, output
  energy, ASE dB, integrated ASE, SBS/SRS.

### Slide 12 — COMPARISON: single-amp vs pump power
Headline: **"More pump → more gain, but more ASE."**
- Results table:

  | Pump | Gain | Out avg | Out peak | Out energy | ASE | Fwd ASE | SBS | SRS |
  |---|---|---|---|---|---|---|---|---|
  | 0.7 W | 25.5 dB (356×) | 200.0 mW | 212.7 W | 40.0 µJ | −15.8 dB | 5.22 mW | 0.003 | 0.097 |
  | 0.8 W | 26.7 dB (465×) | 261.4 mW | 278.1 W | 52.3 µJ | −14.9 dB | 8.53 mW | 0.003 | 0.122 |
  | 0.9 W | 27.6 dB (574×) | 322.5 mW | 343.1 W | 64.5 µJ | −13.7 dB | 13.62 mW | 0.004 | 0.145 |
  | 1.0 W | 28.3 dB (679×) | 381.9 mW | 406.3 W | 76.4 µJ | −12.5 dB | 21.28 mW | 0.004 | 0.167 |

- Build 2 small trend charts from this table (Design should generate these as
  native charts):
  1. **Gain & output power vs pump** (both rise, near-linear).
  2. **ASE ratio (dB) & integrated forward ASE vs pump** (ASE grows — fwd ASE
     **4×** from 0.7 → 1.0 W while gain rises only ~3 dB).
- Takeaways:
  - Output power scales cleanly with pump (200 → 382 mW).
  - **ASE is the cost:** the ASE-to-signal ratio climbs from −15.8 to −12.5 dB and
    integrated ASE quadruples — the high-gain weak-seed stage breeds ASE.
  - SBS/SRS stay far below threshold throughout (< 0.2) — nonlinearity is not the
    limiter here; ASE is.

### Slide 13 — Single-amp pulse shape (optional, time-resolved)
- Any one of:
  `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/single_amp_pump_1p0W/pulse_shape.png`
  (input vs amplified output pulse, Level-5 B2). Use to show pulse amplification &
  gain-saturation reshaping.

---

## Slide 14 — Closing / takeaways
- BGU 3-stage chain meets the 1064 nm / high-peak-power target (62.5 W, 83.2 kW).
- Pump direction (co vs counter) is interchangeable at this operating point.
- ASE — not SBS/SRS — is the dominant limiter, concentrated in high-gain front-end
  stages and growing with pump.

---

## APPENDIX — all available figures per run (absolute paths)

Base directory: `/Users/amanda/Documents/Uni/fiber-mopa-simulator/report/`

| Run | Folder | Figures present |
|---|---|---|
| BGU co-pump | `bgu_example/` | ase_spectra, power_evolution, amplifier_details, nonlinear_margins, bpf_spectra |
| BGU counter-pump | `bgu_example_counter_pump/` | (same 5) |
| BGU co-pump (B1+B2) | `bgu_example_forceb2/` | (same 5) + **pulse_shape** |
| BGU counter (B1+B2) | `bgu_example_counter_pump_forceb2/` | (same 5) + **pulse_shape** |
| Single-amp 0.7 W | `single_amp_pump_0p7W/` | ase_spectra, power_evolution, amplifier_details, nonlinear_margins, **pulse_shape** |
| Single-amp 0.8 W | `single_amp_pump_0p8W/` | (same 5) |
| Single-amp 0.9 W | `single_amp_pump_0p9W/` | (same 5) |
| Single-amp 1.0 W | `single_amp_pump_1p0W/` | (same 5) |

Each folder also contains `summary.txt` with the full per-stage numbers if more
detail is needed for a slide.
