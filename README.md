# Stellar Spectroscopy from Scratch — GPU Edition

The **torch/MPS vectorized companion** to
[*Stellar Spectroscopy from Scratch*](https://github.com/tingyuansen/stellar-spectroscopy-from-scratch)
(the NumPy edition). The same build-from-scratch course — model atmosphere, equation of state,
opacity, and radiative transfer, all the way to a synthetic stellar spectrum — but each lecture's
NumPy code is swapped for a clean, pedagogical `torch` version that is **vectorized over the depth
axis and runs on the GPU** (Apple **MPS** or **CUDA**, with a CPU fallback that stays the
high-precision reference). The physics, the constants, and the lecture arc are identical to the
NumPy edition; only the array library and the device change.

This edition keeps the same goal as its twin — reproduce a real stellar spectrum from first
principles — and adds a second discipline on top. The NumPy edition is benchmarked to
[**pykurucz**](https://arxiv.org/abs/2603.11693), a pure-Python implementation of Kurucz's ATLAS12
and SYNTHE, at its documented floor. Here, **each lecture additionally validates the GPU result
against the NumPy edition's reference** with a per-lecture comparison cell that reports the maximum
relative deviation, asserting parity to the documented float floor (the fp32 GPU path against the
fp64 NumPy result). That gives two things at once: an **independent, per-part check** of the GPU
code, and a **cleaner GPU-native textbook**.

A stellar-atmosphere code has **two halves**, and the book builds both. The **spectrum half**
(Part V, Lecture 14) takes a model atmosphere and assembles the whole opacity-and-transfer stack
into a lean synthesiser, run across the HR diagram (a hot dwarf, the Sun, a giant, an M dwarf). The
**atmosphere half** (Part VI, Lectures 15–16) — the book's finale — switches on the millions of
spectral lines and builds the per-iteration equation of state from scratch, so the line-blanketed
convergence reaches the *real* Sun's model atmosphere. Chain the two and a star's parameters
$(T_{\rm eff}, \log g, [{\rm M/H}])$ become its converged line-blanketed structure and, from it,
its emergent spectrum: the complete from-scratch Sun — now on the GPU.

The notebooks are **self-contained**: each imports `torch`, `numpy`, `matplotlib`, and `pathlib`
and loads small reference data files shipped beside it (`reference/*.npz`, identical to the NumPy
edition's). They never import pykurucz, and they never import the production `kgpu` engine — the
reference values are precomputed once and travel with the book, so a reader needs only `torch` +
`numpy` to run, and to *validate*, every result.

*Yuan-Sen Ting — Max Planck Institute for Astronomy & The Ohio State University. Written in
collaboration with Claude Opus 4.8 under the author's supervision; schematics generated with
Gemini 3 Pro.*

## The discipline

The GPU code is a **pedagogical reduction** of the production torch/MPS engine (`kgpu`), exactly as
the NumPy edition is a pedagogical reduction of pykurucz: the same formulas, the same constants, the
same numerical steps, and the same output — but stripped of the hardening a production code needs
and a lecture does not (custom kernels, residency bookkeeping, caching, defensive guards, CLI
plumbing). It is **plain, readable `torch`** in bite-size cells, vectorized over depth so each
tensor op processes all atmospheric layers at once. The physics and the numbers are identical to the
NumPy edition; only the packaging — and the device — changes.

Each lecture ends with a **comparison cell**: it runs the GPU computation and loads the NumPy
edition's shipped reference values, then reports the maximum relative deviation. The bar is the
**documented float floor** — for the fp32 GPU path against the fp64 NumPy reference, a relative
difference at the level of single-precision round-off (typically a few ×10⁻⁶ for the equation of
state and the opacities, tighter where the computation is a single reduction). On a CPU fallback the
same code runs in fp64 and recovers machine precision. The deviation is **quantified, not hidden**,
and the NumPy edition — itself the gold standard here — is never modified.

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
15. **Line Blanketing: the True Model Atmosphere** — the predicted line list and `SELECTLINES`, the `LINOP1` wing-walk deposit kernel (the asymmetric sub-pixel walk, the full Voigt, the cutoff reach) reproduced bit-exact, the line-blanketed Rosseland mean, and one iteration of the Lecture-11 convergence engine — unchanged, with the blanket switched on — reaching the **real Sun's** model atmosphere (`sun.npz`).
16. **The Full Equation of State: Species Slots & the Convective Heat Capacity** — the per-iteration state the line deposit and the continuum actually consume (the last borrowed intermediate), now built from scratch: the multi-element `POPSALL`/`NELECT` species slots (the flat 1006-slot population layout, the Doppler widths, the van-der-Waals perturber number), the `TABCONT` continuum-cutoff table and its far-UV metal bound-free forest, the molecular deposit slots, and the `EDENS` convective heat capacity carrying the **ionization energy** (the partial-ionization adiabat that keeps the deep base from over-heating). With this, the full line-blanketed convergence runs end to end with **zero pykurucz in the computed path**, descending onto the real Sun's model atmosphere (`sun.npz`) to a temperature median of 7.7 × 10⁻⁴. *Parameters in, atmosphere out — the complete from-scratch Sun.*

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
  ends by benchmarking its GPU arrays to the NumPy edition's shipped reference. You can open any one
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
data files are the **NumPy edition's** shipped references, copied here unchanged — they are the
gold standard each GPU lecture validates against. They were generated once (in the NumPy edition) by
the only script that imports pykurucz; this edition never regenerates them and never imports
pykurucz or the production `kgpu` engine.

Read **[PASSDOWN.md](PASSDOWN.md)** first for the current handoff state, especially the L14
line-blanketed Sun refresh and the no-`kgpu`/no-`pykurucz` self-containment rule. See
**[PLAN.md](PLAN.md)** for the GPU-substitution roadmap. For L14-L16 reference-bundle edits, also
read **[reference/MANIFEST_L14_L16.md](reference/MANIFEST_L14_L16.md)** so loaded computed state
and comparison-only targets stay clearly separated.

## Layout

```
content/        executed notebooks (.ipynb) + rendered fragments (.html) — the chapters
_pipeline/      build_lecture*_gpu.py (assemble), build.py (execute+render), gen_schematics.py
reference/      shipped benchmark data (*.npz) — copied from the NumPy edition (the gold standard)
resources/      figures (Gemini schematics — synced from the NumPy edition)
assets/         render.js / style.css / book-data.js — the reader
PASSDOWN.md     current handoff: relationship to kgpu, L14 state, checks, delegation policy
PLAN.md         the GPU-substitution roadmap (per-lecture pattern; now vs. deferred)
index.html      table of contents      reader.html  the lecture reader
```

## Credits

The GPU port is a pedagogical reduction of the production `kgpu` torch/MPS engine; the physics and
the benchmark references follow the [NumPy edition](https://github.com/tingyuansen/stellar-spectroscopy-from-scratch)
and Kurucz's ATLAS12 & SYNTHE via [Kim & Ting (2026), pykurucz](https://arxiv.org/abs/2603.11693).
Written in collaboration with Claude Opus 4.8 under the author's supervision; schematics by Gemini 3
Pro (Nano Banana). Dedicated to the memory of Robert L. Kurucz (1944–2025).
