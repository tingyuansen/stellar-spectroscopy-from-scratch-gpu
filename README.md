# Stellar Spectroscopy from Scratch

A self-contained, GPU-native textbook that rebuilds the Kurucz ATLAS12/SYNTHE stellar-spectrum
pipeline from first principles in readable `torch`: model atmosphere, equation of state, opacity,
radiative transfer, and finally synthetic spectra. Each lecture is written as a small, executable
notebook that runs on Apple **MPS** or **CUDA** when available, with a CPU fallback for the
high-precision reference path.

The standard is not "looks plausible"; every lecture computes a result and validates it against
shipped reference data at the documented float floor. Those references are produced by the
independent NumPy/pykurucz validation chain and are used here only as parity targets. The taught path
does not import pykurucz and does not import the production `kgpu` package.

A stellar-atmosphere code has **two halves**, and the book now exposes both. The **spectrum half**
(Part V, Lecture 14) takes a model atmosphere and assembles the whole opacity-and-transfer stack
into a lean synthesiser, run across the HR diagram (a hot dwarf, the Sun, a giant, an M dwarf). The
**atmosphere half** (Part VI, Lectures 15–16) switches on the line blanket and rebuilds much of the
per-iteration state, including the exact line-deposit teaching window and EOS-derived species
state. The remaining boundary is explicit: L14 still consumes loaded atmosphere/EOS state, L15
still consumes some line-blanketing fixtures, and L16 is still partly helper-backed. The book is
therefore honest and parity-tested, but not yet a fully closed stellar-parameters-to-spectrum
torch capstone.

The notebooks are **self-contained**: each imports `torch`, `numpy`, `matplotlib`, and `pathlib`
and loads small reference data files shipped beside it (`reference/*.npz`). They never import
pykurucz, and they never import the production `kgpu` engine — the reference values are precomputed
once and travel with the book, so a reader needs only `torch` + `numpy` to run, and to *validate*,
every result.

*Yuan-Sen Ting — Max Planck Institute for Astronomy & The Ohio State University. Written in
collaboration with Claude Opus 4.8 under the author's supervision; schematics generated with
Gemini 3 Pro.*

## The discipline

The code is a **pedagogical reduction** of the production torch/MPS engine (`kgpu`): the same
formulas, constants, numerical steps, and output, but stripped of production hardening a lecture does
not need (custom kernels, residency bookkeeping, caching, defensive guards, CLI plumbing). It is
**plain, readable `torch`** in bite-size cells, vectorized over depth so each tensor op processes all
atmospheric layers at once.

Each lecture ends with a **comparison cell**: it runs the taught computation, loads the shipped
reference values, and reports the maximum relative deviation. The bar is the **documented float
floor** — for the fp32 GPU path against the fp64 reference, a relative difference at the level of
single-precision round-off (typically a few x 10^-6 for the equation of state and the opacities,
tighter where the computation is a single reduction). On a CPU fallback the same code runs in fp64
and recovers machine precision. The deviation is **quantified, not hidden**.

## The lectures

**Part I — Foundations**
1. **Overview & a First Model Atmosphere** — the pipeline, units, the Planck function, optical depth, and a grey model atmosphere from `(Teff, log g)`.
2. **The Equation of State** — Saha–Boltzmann ionization, partition functions, the electron density, and the full per-ion **PFSAHA** ionization core (bit-exact).

**Part II — Opacity**
3. **Continuous Opacity** — H⁻, H, He and metal bound-free/free-free, Rayleigh and Thomson scattering: the KAPP engine reproduced from the atomic cross-section tables (machine precision through the photosphere).
4. **Line Opacity I: A Single Line** — oscillator strength, the Boltzmann population, and the Voigt profile (bit-exact).
5. **Line Opacity II: The Line List** — the full atomic line list (all Z + helium), the log-λ grid, the cutoff, the wing-accumulation kernel, and the autoionizing/merged-continuum line types (machine precision).
6. **Hydrogen Lines: Stark Broadening** — the linear Stark effect and the HPROF4 engine; the Hβ wing reaching into the window (machine precision).

**Part III — Radiative transfer**
7. **Radiative Transfer & the Emergent Spectrum** — the transfer equation, the formal solution, Eddington–Barbier, and the assembled spectrum.
8. **The JOSH Solver: Production Radiative Transfer** — the moment method with scattering, reproduced to machine precision.

**Part IV — Building the atmosphere**
9. **Hydrostatic Equilibrium & Temperature Structure** — the grey start and hydrostatic integration (TTAUP), bit-exact.
10. **Radiative Equilibrium & Temperature Correction** — flux constancy, the Avrett–Krook/TCORR correction, the Rosseland mean, and the radiation-pressure moment.
11. **Convection & the Converged Atmosphere** — mixing-length convection, overshoot, the EOS derivatives, and the converged *continuum-only* model (the line-blanketed finish is Part VI).

**Part V — Cool stars & the spectrum end to end**
12. **Molecular Equilibrium & Molecular Bands** — dissociation equilibrium and the TiO band opacity of a cool dwarf (machine precision).
13. **Molecular Chemistry: the Coupled Equilibrium and Continuous Opacity** — the coupled NMOLEC Newton solver and the molecular continuum (CH, OH, H₂ collision-induced absorption), from scratch.
14. **A Spectrum from Stellar Parameters, End to End** — the **synthesis half**: the lean “kurucz” assembled from every component (EOS, continuum, atomic/hydrogen/helium lines, molecular bands, JOSH transfer) and run across the HR diagram, computing every spectrum from scratch given an atmosphere. *Atmosphere in, spectrum out.*

**Part VI — The line-blanketed atmosphere (the finale)**
The other half of “end to end”: the model atmosphere itself, built genuinely from scratch.
15. **Line Blanketing: the True Model Atmosphere** — the predicted line list and `SELECTLINES`, the `LINOP1` wing-walk deposit kernel (the asymmetric sub-pixel walk, the full Voigt, the cutoff reach) reproduced in a teaching window, the line-blanketed Rosseland mean, and the convergence-core precision audit. It still consumes loaded atmosphere/EOS/window/continuum/full-grid blanket fixtures, which are named in the lecture.
16. **The Full Equation of State: Species Slots & the Convective Heat Capacity** — the per-iteration state the line deposit and continuum consume: the multi-element `POPSALL`/`NELECT` species slots, Doppler widths, van-der-Waals perturber number, continuum-cutoff bridge, molecular slots, and `EDENS` convective heat-capacity inputs. It recomputes this state for a loaded atmosphere fixture and documents the remaining helper-backed boundaries.

## How to read this book

The full structural map — the four-part arc, what each lecture builds and depends on, the two
halves of "end to end," and the documented out-of-scope boundaries — is in
**[STRUCTURE.md](STRUCTURE.md)**, the book's organizing map. In brief:

- **Read it front to back.** The arc is cumulative: Parts I–III build the microphysics (foundations,
  opacity, transfer) treating the atmosphere as given; Part IV builds the atmosphere's structure;
  Part V adds cool-star chemistry and then assembles the **spectrum end to end**; Part VI is the
  **finale** — the line-blanketed atmosphere from scratch.
- **Each lecture is self-contained and stands alone.** Every notebook imports `torch`, `numpy`,
  `matplotlib`, and `pathlib`, loads its own `reference/*.npz`, runs top to bottom on the GPU, and
  ends by benchmarking its arrays to the shipped reference. You can open any one
  lecture and understand what it achieves without flipping back; the cross-references are light
  signposts, not prerequisites you must chase.
- **The two halves of "end to end."** A stellar-atmosphere code is two halves. **Lecture 14** is the
  *spectrum* half — atmosphere in, spectrum out — assembled and run from scratch across the HR
  diagram. **Part VI (Lectures 15–16)** is the *atmosphere* half — parameters in, line-blanketed
  model atmosphere out — the book's true finale. Lecture 11's converged model is the *continuum-only*
  scaffold; Part VI switches the blanket on and builds the per-iteration equation of state that
  closes the last borrowed intermediate. Lecture 14 now consumes that Part-VI line-blanketed solar
  state for the Sun, while the hot dwarf, giant, and M dwarf remain documented emulator warm-starts.
- **The boundaries are named, not hidden.** Out of scope, by design: full optical bandwidth (an
  engineering problem of compiled kernels and parallelism — the physics is all here), and NLTE
  statistical equilibrium (the real physics frontier). The geometry is 1D plane-parallel throughout,
  matching the production code's own picture.

## Viewing

The rendered book is a static site. Serve the directory and open `index.html`:

```bash
python3 -m http.server 8899
# then open http://127.0.0.1:8899/   (or reader.html?ch=N for a lecture)
```

The rendered `content/*.html` already embed all executed outputs and figures, so the site
displays without running anything.

## Building from source

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt               # numpy + matplotlib + torch + nbformat/nbconvert
npm install                                   # marked / katex / highlight.js for rendering
python _pipeline/build_lecture2_gpu.py         # assemble one notebook (content/Lecture2.ipynb)
python _pipeline/build.py 2                     # execute it (MPS if available, else CPU) + render
# or rebuild all:  python _pipeline/build.py 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

The notebooks select the device automatically: Apple **MPS** or **CUDA** if present, otherwise
**CPU** (where the same code runs in fp64 and recovers machine precision). The `reference/*.npz`
data files are shipped references — the gold standard each lecture validates against. They were
generated once by the offline validation pipeline; this book never regenerates them and never imports
pykurucz or the production `kgpu` engine in the taught path.

Read **[PASSDOWN.md](PASSDOWN.md)** first for the current handoff state, especially the L14
line-blanketed Sun refresh and the no-`kgpu`/no-`pykurucz` self-containment rule. See
**[PLAN.md](PLAN.md)** for the GPU-substitution roadmap. For L14-L16 reference-bundle edits, also
read **[reference/MANIFEST_L14_L16.md](reference/MANIFEST_L14_L16.md)** so loaded computed state
and comparison-only targets stay clearly separated.

## Layout

```
content/        executed notebooks (.ipynb) + rendered fragments (.html) — the chapters
_pipeline/      build_lecture*_gpu.py (assemble), build.py (execute+render), gen_schematics.py
reference/      shipped benchmark data (*.npz)
resources/      figures and schematics
assets/         render.js / style.css / book-data.js — the reader
PASSDOWN.md     current handoff: relationship to kgpu, L14 state, checks, delegation policy
PLAN.md         the GPU-substitution roadmap (per-lecture pattern; now vs. deferred)
index.html      table of contents      reader.html  the lecture reader
```

## Credits

The book is a pedagogical reduction of the production `kgpu` torch/MPS engine; the physics follows
Kurucz's ATLAS12 & SYNTHE and is validated through [Kim & Ting (2026), pykurucz](https://arxiv.org/abs/2603.11693).
Written in collaboration with Claude Opus 4.8 under the author's supervision; schematics by Gemini 3
Pro (Nano Banana). Dedicated to the memory of Robert L. Kurucz (1944–2025).
