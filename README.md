# Stellar Spectroscopy from Scratch

A build-from-scratch course that reconstructs a synthetic stellar spectrum from first
principles — model atmosphere, equation of state, opacity, and radiative transfer — in short,
readable NumPy. Every step is **benchmarked to machine precision** against
[**pykurucz**](https://arxiv.org/abs/2603.11693), a pure-Python implementation of Kurucz's
ATLAS12 and SYNTHE — the equation of state, every opacity source, the model-atmosphere build, and the radiative transfer — then assembled into a lean end-to-end synthesiser and run across the HR diagram (a hot dwarf, the Sun, a giant, an M dwarf) in parity with pykurucz end to end.

The notebooks are **self-contained**: each imports only `numpy`, `matplotlib`, and `pathlib` and loads
small reference data files shipped beside it (`reference/*.npz`). They never import pykurucz —
the reference values are precomputed once and travel with the book, so a reader needs only
NumPy to run and verify every result.

*Yuan-Sen Ting — Max Planck Institute for Astronomy & The Ohio State University. Written in
collaboration with Claude Opus 4.8 under the author's supervision; schematics generated with
Gemini 3 Pro.*

## The discipline

The from-scratch code is a **pedagogical reduction** of pykurucz: the same formulas, the same
constants, the same numerical steps, and the same output — but stripped of the hardening a
production code needs and a lecture does not (Numba `@jit`/`fastmath`, caching, defensive
guards, CLI plumbing). Where a plain Python loop would be too slow for a notebook cell, the
code is **vectorized in NumPy**, never accelerated with a compiler. The physics and the
numbers are identical; only the packaging changes.

Each lecture ends by comparing its from-scratch arrays to the shipped reference values and
reporting the maximum deviation. The bar is **machine precision** — a relative difference at
the level of floating-point round-off (≈1e-12…1e-15); the only tolerated residual is the last
bit of a single-precision iteration the production code itself carries, and it is quantified
rather than hidden. pykurucz is the gold standard and is never modified.

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
11. **Convection & the Converged Atmosphere** — mixing-length convection, overshoot, the EOS derivatives, and the converged model.

**Part V — Cool stars and the whole pipeline**
12. **Molecular Equilibrium & Molecular Bands** — dissociation equilibrium and the TiO band opacity of a cool dwarf (machine precision).
13. **Molecular Chemistry: the Coupled Equilibrium and Continuous Opacity** — the coupled NMOLEC Newton solver and the molecular continuum (CH, OH, H₂ collision-induced absorption), from scratch.
14. **The Capstone: A Spectrum from Stellar Parameters, End to End** — the lean “kurucz” assembled from every component and run across the HR diagram, in parity with pykurucz end to end.

**Part VI — The true line-blanketed atmosphere**
15. **Line Blanketing: the True Model Atmosphere** — the predicted line list and `SELECTLINES`, the `LINOP1` wing-walk deposit kernel (the asymmetric sub-pixel walk, the full Voigt, the cutoff reach) reproduced bit-exact, the line-blanketed Rosseland mean, and one iteration of the Lecture-11 convergence engine — unchanged, with the blanket switched on — reaching the **real Sun's** model atmosphere (`sun.npz`).
16. **The Full Equation of State: Species Slots & the Convective Heat Capacity** — the per-iteration state the line deposit and the continuum actually consume, now built from scratch: the multi-element `POPSALL`/`NELECT` species slots (the flat 1006-slot population layout, the Doppler widths, the van-der-Waals perturber number), the `TABCONT` continuum-cutoff table and its far-UV metal bound-free forest, the molecular deposit slots, and the `EDENS` convective heat capacity carrying the **ionization energy** (the partial-ionization adiabat that keeps the deep base from over-heating). With this, the full line-blanketed convergence runs end to end with **zero pykurucz in the computed path**, descending from a grey start onto the real Sun's model atmosphere (`sun.npz`) to a temperature median of 7.7 × 10⁻⁴.

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
pip install -r requirements.txt
npm install                                   # marked / katex / highlight.js for rendering
python _pipeline/build_lecture4.py            # assemble one notebook (content/Lecture4.ipynb)
python _pipeline/build.py 4                    # execute it + render to content/Lecture4.html
# or rebuild all:  python _pipeline/build.py 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

The `reference/*.npz` data files were generated once by `_pipeline/make_references.py` (the
only script that imports pykurucz) and by the verifier scripts `_pipeline/verify_*.py`, which
reproduce each production engine from scratch and confirm machine precision.

## Layout

```
content/        executed notebooks (.ipynb) + rendered fragments (.html) — the chapters
_pipeline/      build_lecture*.py (assemble), build.py (execute+render), verify_*.py, gen_schematics.py
reference/      shipped benchmark data (*.npz) — the book is self-contained
resources/      figures (Gemini schematics)
assets/         render.js / style.css / book-data.js — the reader
index.html      table of contents      reader.html  the lecture reader
```

## Credits

Physics and benchmark code after Kurucz's ATLAS12 & SYNTHE, via
[Kim & Ting (2026), pykurucz](https://arxiv.org/abs/2603.11693). Written in collaboration with
Claude Opus 4.8 under the author's supervision; schematics by Gemini 3 Pro (Nano Banana).
Dedicated to the memory of Robert L. Kurucz (1944–2025).
