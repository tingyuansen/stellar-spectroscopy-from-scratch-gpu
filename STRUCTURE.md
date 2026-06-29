# STRUCTURE — the organizing map of *Stellar Spectroscopy from Scratch*

This is the book's directory bible: the coherent, holistic view of the four-part arc, what each
lecture builds, how the two halves of "end to end" fit together, and where the documented
boundaries lie. It sits beside the per-lecture notebooks and the README's "How to read this book"
section.

Each lecture's computation is a clean, depth-batched `torch` implementation, written as a
pedagogical reduction of the production `kgpu` engine. It runs on MPS/CUDA when available, with a
CPU fallback, and ends with a **comparison cell** validating the computed result against shipped
reference data to the documented float floor. The roadmap — which lectures are fully torch-native
now and which still have named integration boundaries — is in **[PLAN.md](PLAN.md)**.

The README has the prose; this file has the map.

---

## The one idea: a stellar-atmosphere code is two halves

A working stellar-atmosphere code does two things, and this book builds both — each *end to end*,
each benchmarked to [pykurucz](https://arxiv.org/abs/2603.11693) (a pure-Python ATLAS12 + SYNTHE) at
its documented floor.

```
            stellar parameters                         model atmosphere
          (Teff, log g, [M/H])                         (T, P, rho vs depth)
                   |                                            |
                   v                                            v
        +----------------------+                    +------------------------+
        |   ATMOSPHERE HALF    |  --- structure --> |    SYNTHESIS HALF      |
        |  parameters -> T,P,  |                    |  structure -> spectrum |
        |  rho structure       |                    |  (EOS, opacity,        |
        |  (Part IV scaffold;  |                    |   transfer)            |
        |   Part VI finale)    |                    |   (Part V, Lecture 14) |
        +----------------------+                    +------------------------+
                                                                 |
                                                                 v
                                                          emergent spectrum
```

- **The spectrum half** — *atmosphere in, spectrum out* — is **Lecture 14** (Part V). Given a model
  atmosphere, it computes the entire opacity and all the radiative transfer from scratch and
  reproduces the emergent spectrum across four stars spanning the HR diagram.
- **The atmosphere half** — *parameters in, model atmosphere out* — is **Part VI (Lectures 15–16)**,
  the book's finale. It switches on the millions of spectral lines, builds the per-iteration
  equation of state from scratch, and runs the convergence engine until it lands on the *real,
  line-blanketed* Sun.

Chain the two and a star's few numbers become its converged structure and, from that, its spectrum:
**the complete from-scratch Sun.**

A note on "end to end": the phrase appears twice in the book, and the two uses are deliberately
distinct. Lecture 14 is the spectrum end to end: it computes opacity and transfer from scratch on
the atmosphere it is given. Three of its four atmospheres are warm-started because their grey starts
fail; the Sun now consumes the Part-VI line-blanketed solar state (base `RHOX = 12.1439331`, not the
old continuum-only `10.5357` bundle). Part VI is the atmosphere end to end (the genuinely
line-blanketed model, with the last borrowed intermediate removed). Neither half claims the other's
achievement.

---

## The four-part arc, lecture by lecture

Each lecture is **self-contained**: it imports `torch`, `numpy`, `matplotlib`, and `pathlib`, loads
its own `reference/*.npz`, runs top to bottom on the selected device, and ends by benchmarking its
computed arrays to the shipped reference (the comparison cell). The "builds" column is what the lecture
*constructs*; the "leans on" column is the light conceptual thread to earlier work (a signpost, not a
prerequisite you must chase).

### Part I — Foundations & Microphysics  *(atmosphere treated as given)*

| L | Builds | Leans on |
|---|--------|----------|
| 1 | Overview, units, the Planck function, optical depth, a grey model atmosphere from `(Teff, log g)` | — |
| 2 | The equation of state: Saha–Boltzmann ionization, partition functions, electron density, the per-ion `PFSAHA` core | L1 (the atmosphere it runs on) |
| 3 | Continuous opacity (`KAPP`): H⁻, H, He, metal bound-free/free-free, Rayleigh & Thomson scattering | L2 (the populations opacity is built on) |

### Part II — Line Opacity  *(atmosphere still given)*

| L | Builds | Leans on |
|---|--------|----------|
| 4 | A single line: oscillator strength, the Boltzmann population, the Voigt profile | L2 (populations) |
| 5 | The full atomic line list (all Z + He), the log-λ grid, the cutoff, the wing-accumulation kernel (`ASYNTH`) | L4 (the single-line profile) |
| 6 | Hydrogen lines: the linear Stark effect, the `HPROF4` engine, the Hβ wing | L4–5 (the line machinery) |

### Part III — Radiative Transfer  *(atmosphere still given)*

| L | Builds | Leans on |
|---|--------|----------|
| 7 | The transfer equation, the formal solution, Eddington–Barbier, the assembled spectrum | L1–6 (the opacity to carry) |
| 8 | The `JOSH` moment solver with scattering (production transfer) | L7 (the formal solution) |

### Part IV — Building the Atmosphere  *(stop taking the atmosphere as given)*

| L | Builds | Leans on |
|---|--------|----------|
| 9 | Hydrostatic equilibrium: `TTAUP`, the pressure/density structure from the grey start | L1 (the grey start) |
| 10 | Radiative equilibrium: flux constancy, the Avrett–Krook/`TCORR` correction, the Rosseland mean, the radiation-pressure moment | L8 (JOSH flux), L9 (structure) |
| 11 | Mixing-length convection, overshoot, the EOS derivatives, and the convergence loop → the **converged *continuum-only* model** | L2, L8, L10; **finished in Part VI** |

### Part V — Cool Stars & the Spectrum End to End

| L | Builds | Leans on |
|---|--------|----------|
| 12 | Molecular equilibrium & the TiO bands of a cool dwarf | L2 (EOS), L4–5 (line machinery) |
| 13 | The coupled `NMOLEC` Newton chemistry and the molecular continuum (CH, OH, H₂-CIA) | L12 (the molecular bands it completes) |
| 14 | **The spectrum, end to end** — the lean synthesiser assembled from L2/L3/L4–6/L7–8/L12 and run across the HR diagram, every spectrum from scratch | all of L1–13 |

### Part VI — The Line-Blanketed Atmosphere (the finale)

| L | Builds | Leans on |
|---|--------|----------|
| 15 | **Line blanketing**: the predicted line list, `SELECTLINES`, the `LINOP1` wing-walk deposit kernel, the line-blanketed Rosseland mean; the **unchanged Lecture-11 engine** rerun with the blanket on, reaching the real Sun's `sun.npz` | L11 (the engine), L4 (the Voigt profile); **consumes L16's state** |
| 16 | **The full per-iteration equation of state from scratch**: the `POPSALL` species slots, Doppler widths, the `TABCONT` cutoff table + far-UV metal bound-free forest, the molecular slots, the `EDENS` heat capacity carrying the ionization energy — the *last borrowed intermediate*, removed; the line-blanketed convergence then runs with **no pykurucz in the computed path** | L2 (the Saha ladder it extends), L15 (the loop it feeds) |

---

## The genuine dependency thread (read both directions)

The arc is cumulative but the heavy dependencies are few and specific. The ones that matter:

- **L2's equation of state** is the spine. L3, L4–6, L12 all build opacity *on its populations*;
  L11's convection finite-differences it; **L16 extends it** to the full multi-element `POPSALL`
  state.
- **L16's per-iteration state is consumed by L15's deposit and by L11/L14's convergence.** This is
  the one place the *later* lecture supplies what the *earlier* ones quietly assumed: L11 and L14
  read that state from a reference; L16 builds it from scratch and hands it back to L15's loop. That
  closes the book's last borrowed intermediate.
- **L11's converged model is continuum-only by design.** Part VI switches the blanket on and turns
  it into the real Sun. L14 now consumes that Part-VI line-blanketed solar state for the Sun while
  retaining documented emulator warm-starts for the hot dwarf, giant, and M dwarf.
- **L14 uses everything from L1–13** — it is the assembly point of the synthesis half, not a new
  engine.

---

## The documented boundaries (out of scope, by design)

These are named in the lectures, not hidden. They do not enter the benchmarks.

- **Full optical bandwidth.** Lecture 14 synthesises short, physically representative windows per
  star (a Balmer line, a metal forest, a pressure-sensitive triplet, a band head), not the whole
  4000–9000 Å band. Doing the full band in pure NumPy is an *engineering* problem (compiled kernels,
  parallelism); the physics is all present.
- **NLTE.** Every population is Saha–Boltzmann (LTE). Relaxing it — the coupled statistical-
  equilibrium solution — is the real physics frontier beyond this book. The architecture is the right
  scaffold: same opacity engines, same transfer, with the population step replaced.
- **1D, plane-parallel, time-independent geometry.** No granulation, spots, rotation, or
  variability — the standard 1D LTE model-atmosphere picture, which is also what the production code
  does, so reproducing it is the correct target.
- **Warm-started atmosphere *structures* in Lecture 14.** Three of the four stars (hot dwarf, giant,
  M dwarf) do not converge from a grey start and are warm-started from the production emulator,
  documented per star. The Sun's structure is the Part-VI line-blanketed solar state.

---

## Files

```
content/        executed notebooks (.ipynb) + rendered fragments (.html) — the chapters
_pipeline/      build_lecture*_gpu.py (assemble each GPU notebook), build.py (execute + render)
reference/      benchmark data (*.npz)
resources/      figures and schematics
assets/         render.js / style.css / book-data.js — the reader (parts + lecture manifest)
PLAN.md         the GPU-substitution roadmap (per-lecture pattern; lectures now vs. deferred)
README.md       overview + the lecture list + "How to read this book"
STRUCTURE.md    this file — the organizing map
index.html      table of contents      reader.html  the lecture reader
```

The source of truth for each GPU notebook is `_pipeline/build_lecture*_gpu.py` (it assembles the
`.ipynb`); `build.py` executes it (on MPS/CUDA if present, else CPU) and renders it.
`assets/book-data.js` is the reader's manifest of parts and lectures and is kept in sync with the
README and `build.py`. The lecture-by-lecture substitution status lives in `PLAN.md`.
