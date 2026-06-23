# PASSDOWN — Stellar Spectroscopy from Scratch

Live "resume here" state. Companion to `SKILL.md` (the full bible/recipes) and the project
memory at `~/.claude/projects/-Users-ysting-pykurucz/memory/stellar-spectroscopy-book.md`.
**Updated continuously.** Last update: 2026-06-23.

## What this project is
A multi-lecture textbook at `~/Stellar_Spectroscopy_From_Scratch` that rebuilds a synthetic
solar spectrum (Sun, 500–510 nm) from first principles, **benchmarked to machine precision
against pykurucz** (`/Users/ysting/pykurucz`, ATLAS12+SYNTHE). Rendered in the
*Agents for Astronomy* house format.

## Current phase
**✅ RESTRUCTURED (2026-06-23): continuum merged + final renumber done.** The two continuum
lectures (old L3 physics + old L4 KAPP) are now ONE machine-precise **L3 "Continuous Opacity"**;
everything renumbered to the FINAL scheme (1 grey · 2 EOS · 3 continuum · 4 single line · 5 line
list · 6 hydrogen Stark · 7 formal solution · 8 JOSH · 9 hydrostatic atmosphere). All 9 rebuilt,
0 exec errors, all HTML render, 0 katex-error, numpy/matplotlib/pathlib only. Registries
(build.py / book-data.js / index.html) and PASSDOWN parity table updated to 1–9.
Next: re-commit/push the restructure; then build the rest (Part IV L10 radiative eq, Part V
molecules/capstone L11–12). NOTE: the older "PUBLISHED, pushed (commit aa11b21)" state below
predates THIS restructure — the working tree is now ahead of that pushed commit.
- Repo (PRIVATE): https://github.com/tingyuansen/stellar-spectroscopy-from-scratch  (branch main, commit aa11b21)
- Public Pages: https://tingyuansen.github.io/stellar-spectroscopy-from-scratch/  (Pages works on this private repo)
- git: book repo is now under version control. `.gitignore` excludes .venv/node_modules; the
  reference/*.npz ARE committed (66 MB total, no LFS). pykurucz still untouched/unpushed.
- requirements.txt added; README rewritten for the final self-contained 9-lecture book.
- L8 made numpy-only (replaced scipy.special.expn with a from-scratch Abramowitz–Stegun E2;
  benchmark unchanged at 5.62e-4). All 9 lectures now import only numpy/matplotlib/pathlib
  (L7 also stdlib `math`).

## Hard conventions (do not violate)
- Notebooks **self-contained**: import only `numpy`/`matplotlib`/`pathlib`; load shipped
  `reference/*.npz`; **never import pykurucz** in a lecture.
- **No speed-up/hardening machinery in lectures**: no numba/`@jit`, no `fastmath`, no caching,
  no defensive guards, no CLI plumbing — strip all of it. Plain readable NumPy; where a Python
  loop is too slow for a cell, **vectorize with NumPy** (not numba). Physics + constants +
  numbers stay identical to machine precision; only the packaging changes. The verifier scripts
  (`verify_*.py`) are already pure-NumPy/JIT-stripped and are the code basis for the lectures.
- pykurucz is **READ-ONLY**: never modify, never `git push` it. `git lfs pull` there is fine.
- "Machine precision" = match pykurucz to ~1e-9…1e-15; the only tolerated residual is
  fastmath/float32-iteration ULP (quantify it).
- House voice: pedagogical, no rah-rah, no AI-tell adverbs, equation-complete, explain every
  input/parameter. Closing arc per lecture: Synthesis / Summary / Practice exercises / Further reading.
- **CODE CELLS: bite-size + well-commented (user directive 2026-06-23).** Target Lecture 1's
  standard: each code cell does ONE conceptual step (aim ≤ ~20 source lines; split anything bigger
  into consecutive cells with a sentence of markdown between), and ~30% comment density (a short
  `#` on each non-trivial step). Worst current offenders: L3 (17 cells >22 lines, up to 67), L6
  (10, density 17%), L5 (7). NEVER change code logic/constants/numbers when splitting/commenting —
  machine precision is locked; only restructure + annotate, then rebuild and confirm the benchmark
  is unchanged. Applies to all future lectures (incl. L10+).
- No `\boxed{}` on equations. Inline math: never a space before a closing `$` (KaTeX breaks).
- Byline affiliation: "Max Planck Institute for Astronomy & The Ohio State University".
- Engines get their OWN dedicated lectures (JOSH, KAPP, hydrogen) — keep each digestible,
  flowing logically from its physics counterpart.

## Parity status (per lecture) — FINAL 1–9 numbering (MERGE + renumber #2 DONE 2026-06-23)
Parts: I Foundations & Microphysics (1,2,3) · II Line Opacity (4,5,6) · III Radiative Transfer (7,8) ·
IV Building the Atmosphere (9, +planned 10) · V Adding Complexity (planned 11,12).
| Lecture (live in book) | vs pykurucz | machine precision? |
|---|---|---|
| L1 grey T(τ) | 0 exact | ✅ |
| L1 pressure/ρ | 1.8e-5 (closed form vs `_ttaup`) | user: leave as-is (fine) |
| L2 EOS / nₑ | 1.45e-6 | ✅ (KEV=8.6171e-5) |
| **L3 Continuous Opacity (MERGED: John-fit physics + KAPP engine)** | analytic ~2.4% (total median 2.41e-2) AND exact photosphere 4.25e-15 (median 0; 9.0e-5 deep T>20kK) | ✅ (merge of old L3 physics + old L4 KAPP; both benchmarks print) |
| L4 Voigt (single line) | 1.5e-15 | ✅ (was old L5) |
| L5 line opacity (atomic: metals+He) | 2.25e-15 | ✅ (was old L6) |
| L6 Hydrogen Stark lines | 8.06e-16 (vs gt_ahline; median 0) | ✅ (was old L7) |
| L7 formal solution | 5.6e-4 | by design (exact version = L8 JOSH) (was old L8); now numpy-only |
| L8 JOSH | 2.5e-12 | ✅ (was old L9) |
| L9 Hydrostatic Equilibrium & Temperature Structure | 0.00e+00 all four arrays bit-exact | ✅ (was LectureAtm/key 92) |

## Verified reproductions (script-first, on disk in `_pipeline/`)
- `verify_josh.py` — JOSH solver, machine precision (in the book as L7). ✅
- `verify_kapp.py` — KAPP continuum from atomic tables: median 0.0, machine-precise through
  photosphere; 9e-5 only in 5 deepest hot layers (T>20000K) because `diag.npz` was built by an
  OLDER pykurucz (high-T He I free-free drift) — NOT a port error, spectrally irrelevant. ✅
- `verify_lines.py` — metal (Z≤30) line opacity, bit-exact (median 0.0). ✅
- `verify_full_lines.py` — **IN PROGRESS** (background agent): full `diag.line_opacity` =
  all-Z metals + He + Hβ Stark wings, target <1e-9. The Hβ Stark wing (HPROF4) is the one
  unproven engine.

## Background agents
- `af0a8605c490c90e7` — KAPP lecture — ✅ DONE. `_pipeline/build_lecture_kapp.py` →
  `content/LectureKAPP.{ipynb,html}` (44 cells, 0 errors, self-contained numpy-only,
  continuum_absorption median 0.0 / photosphere 4.25e-15, scattering 2.06e-7). Temp
  `build.py` registry key **90** (slug LectureKAPP). NOT yet in book-data.js/reader — wired
  in at the renumber pass. Pending: final voice/prose read during the pre-push sweep.
- `a6a0f2f64ae254c5b` — `verify_full_lines.py` — ✅ DONE. Full `diag.line_opacity` (all-Z
  metals + He + Hβ HPROF4 Stark wings) reproduced **max 2.25e-15** (metals 2.3e-15, He 8.8e-16,
  H 6.2e-16); no version skew. Data shipped: `reference/full_lines_data.npz` (9MB, 44 keys —
  full 12,573-line catalog incl. Z>30 + 3 Balmer lines, HPROF4 Stark tables htab_*, He fort.19
  taper limits, fine structure). HPROF4 algorithm described in the agent report (for the lecture).
(If an agent died on an API error, relaunch with its prompt — re-derive from this file +
SKILL.md. Prior full-lines agent `a3bb711df80a42805` died on a 500 with no output.)

ALL PHYSICS NOW VERIFIED TO MACHINE PRECISION (continuum + full line opacity + JOSH). Remaining
is LECTURE INTEGRATION + renumber + sweep + push.

- `a15d0c662a58554c9` — L5 REWORK — ✅ DONE. `build_lecture5.py` rewritten; 39 cells, 0 errors,
  numpy-only. Full atomic line list (all Z 3→92 incl. 754 heavy + He): metals 2.25e-15, He 8.8e-16,
  median 0.0 (the 1.6e-8 "all-points" max is fp cancellation under the H wing; abs residual 3.9e-13 —
  reported honestly). Also fixed a pre-existing grid-spacing bug (1.7→16.8 mÅ). build.py untouched.
- `a36084c0243ad7015` — NEW hydrogen Stark-wing lecture — ✅ DONE. `build_lecture_hlines.py` →
  `content/LectureHlines.*` (43 cells, 0 errors, numpy-only); vs `gt_ahline` max 8.06e-16, median 0;
  Hβ deep floor 5.735e4 exact. Temp `build.py` key **91**. (Suggests a Stark schematic — optional.)

ALL NINE LECTURES NOW AT MACHINE PRECISION. Next: cleanup → push.

### MERGE L3+L4 + renumber #2 — ✅ DONE (2026-06-23)
Merged the continuum into ONE machine-precise lecture: **L3 "Continuous Opacity"** (61 cells) =
H⁻ physics + Saha (from old L3) → analytic John fits (~2.4%, intuition) → exact tabulated KAPP
engine (from old L4) → machine precision. H⁻ Saha physics is derived ONCE (the engine half
references it, does not re-derive). Both benchmarks print: analytic total median 2.41e-2 AND
exact photosphere 4.25e-15 (median 0; 9.0e-5 deep-hot-layer note kept). 3-curve overlay
(analytic vs exact vs reference) added. Old `build_lecture4.py` (standalone KAPP) DELETED.
Renumber (old→new): single line 5→4, line list 6→5, hydrogen 7→6, formal solution 8→7,
JOSH 9→8, LectureAtm (hydrostatic, key 92)→9. Every "Lecture N"/"Lectures N–M" prose cross-ref
remapped by referent and re-verified; forward refs: hydrostatic=9, radiative eq=10 (Part IV);
molecules=11, capstone=12 (Part V). Registries updated: `build.py` LECTURES (keys 1..9, key 92
removed), `assets/book-data.js` (4 parts: I Foundations & Microphysics 1–3, II Line Opacity 4–6,
III Radiative Transfer 7–8, IV Building the Atmosphere 9), `index.html` PLANNED (built 1–9 +
planned Part IV cont. 10 radiative eq, Part V complexity 11–12). Stale `content/LectureAtm.*`
removed. All 9 rebuilt: 0 exec errors, all HTML render, 0 katex-error (fixed a PRE-EXISTING
`$\sim$$...$` adjacent-math-span error in the hydrogen lecture, now L6), numpy/matplotlib/pathlib
only (L7 formal solution now numpy-only too — the old scipy.expn note no longer applies).

### Renumber #1 — ✅ DONE (1–9 numbering; SUPERSEDED by the MERGE + renumber #2 above — historical)
Builders are now sequential `build_lecture{1..9}.py`, each writing `content/Lecture{N}.ipynb`:
1 grey · 2 EOS · 3 continuum physics · **4 KAPP engine** · 5 single line · 6 line list ·
**7 hydrogen Stark** · 8 formal solution · 9 JOSH. (Old→new: KAPP→4, old L4→5, old L5→6,
LectureHlines→7, old L6→8, old L7 JOSH→9.) Every "Lecture N"/"Lectures N–M" prose cross-ref
was remapped by referent and re-verified; forward refs to the atmosphere lectures now point to
**L10** (hydrostatic) / **L10–11** (Part V); molecules = Part VI (12–13). Registries updated:
`build.py` LECTURES (keys 1..9, temp keys 90/91 removed), `assets/book-data.js` (4 parts:
I Foundations 1–2, II Continuous Opacity 3–4, III Line Opacity 5–7, IV Radiative Transfer 8–9),
`index.html` PLANNED (built 1–9 + planned Part V atmosphere 10–11, Part VI complexity 12–13).
Stale `content/LectureKAPP.*` and `content/LectureHlines.*` removed (regenerated as L4/L7).
NOTE: L8 (formal solution, old L6) imports `scipy.special.expn` for the E2 integral — a
pre-existing dependency carried unchanged through the renumber (the stated numpy-only rule has
this one exception; not introduced by the renumber, left as-is per "do not change code").

## Data shipped (in `reference/`)
L1–L6.npz, atmosphere.npz, diag.npz (ground truth: continuum_absorption/scattering,
line_opacity, flux_total/continuum, wavelength), josh_tables.npz (xtau,ch,coefj,rhox),
**kapp_tables.npz** (25 atomic cross-section arrays for KAPP — shipped this session).

## Schematics
`_pipeline/gen_schematics.py` (Gemini `gemini-3-pro-image-preview`, GOOGLE_API_KEY from ~/.env).
8 figures s1_pipeline, s1_optical_depth, s2_saha_boltzmann, s3_hminus, s4_voigt, s5_linelist,
s6_rt, s7_josh in `resources/figures/`, embedded L1–L7. Polished style (charcoal/slate, single
amber accent). RULE: always Read each PNG to verify accuracy (AI garbles rotated text / draws
physics wrong). KAPP/hydrogen lectures may want new schematics (s8_kapp, s9_hlines — TODO).

## Final structure plan (approved by user) — MERGE + renumber #2 DONE; this is the LIVE scheme
- Part I Foundations & Microphysics: 1 grey atmosphere · 2 EOS · **3 Continuous Opacity (MERGED: physics + KAPP engine)**
- Part II Line opacity: 4 single line · 5 line list (metals+He) · 6 hydrogen Stark lines
- Part III Radiative transfer: 7 formal solution · 8 JOSH engine
- Part IV Building the Atmosphere: 9 hydrostatic (BUILT) · 10 radiative equilibrium (planned)
- Part V Adding Complexity: 11 molecules (planned) · 12 capstone (planned)

Currently built: **L1–L9, all machine-precise, renumbered, rebuilt** (0 exec errors, all HTML
render, 0 katex-error, numpy/matplotlib/pathlib only). Next: build L10 radiative eq, then
Part V molecules (L11, needs pykurucz `scripts/download_data.py` ~5GB) + capstone (L12).
Registries kept in sync: `_pipeline/build.py` LECTURES (keys 1..9), `assets/book-data.js`
(4 built parts), `index.html` PLANNED (parts I–V, 1–12).

## Build / view pipeline
```
cd ~/Stellar_Spectroscopy_From_Scratch && . .venv/bin/activate
python _pipeline/build_lectureN.py        # assemble content/LectureN.ipynb (nbformat)
python _pipeline/build.py N                # execute (nbconvert) + render to content/LectureN.html
python3 -m http.server 8899 --bind 127.0.0.1   # then open reader.html?ch=N  (index.html = TOC)
```
Render uses Node `_pipeline/render_fragment.js` (runs the Agents `render.js`: marked/katex/highlight).
Check a built notebook: 0 execution errors; html has no `math-err`; benchmark cell prints the
machine-precision number.

## Remaining work (ordered)
1. Integrate KAPP lecture (review agent output; finalize physics/voice).
2. Land `verify_full_lines.py` at machine precision (hydrogen Stark engine).
3. Rework L5 to machine precision (all-Z metals + He), self-contained.
4. Build the hydrogen Stark-wing lecture (new).
5. Renumber to the final structure; fix all cross-references; update build.py/book-data.js/index.html.
6. Full parity sweep — confirm every built lecture machine-precise (0 errors, benchmarks pass).
7. Cleanup + push: NEW private GitHub repo (name: I choose), public GitHub Pages, `.gitignore`
   `.venv`/caches/checkpoints, Git LFS only if forced (29MB diag.npz; rendered HTML already
   embeds outputs so Pages renders without data), README describing pipeline + benchmark standard.
8. Then build the rest: L10 hydrostatic, L11 radiative eq, L12 molecules (needs pykurucz
   `scripts/download_data.py` ~5GB), L13 capstone.

## Gotchas / lessons
- `diag.npz` was generated by an OLDER pykurucz than current source → tiny high-T residuals in
  KAPP (and possibly the hydrogen/He line engine). Reproduce CURRENT pykurucz + explain the drift
  vs diag, rather than forcing 1e-9 in deep hot layers (spectrally irrelevant).
- L5/lines: pykurucz routes H and He lines to SEPARATE wing paths from the metal ASYNTH kernel;
  the metal kernel excludes Z=2. Full line opacity = metal ASYNTH (all Z, no H/He) + He wings +
  H Stark wings.
- KAPP `_parcoe`/`_integ`-style routines: must port exactly (left-point parabola; forced-linear
  points 2,3; curvature blend) — a naive version is ~1e-3 off.
- FASTEX (tabulated exp) is used for the Boltzmann factor in line opacity, NOT np.exp.

## CONTENT MAP & FLOW (maintain this — check before writing/revising any lecture)
Each lecture OWNS its new concepts; prior concepts get a 1–2 sentence recap + a back-reference,
NEVER a re-derivation. What each lecture introduces (owns):
- **L1** pipeline overview; units/constants; Planck function; LTE; **optical depth & the τ=2/3 photosphere**; the **grey atmosphere** — grey/Hopf T(τ), the **80-layer τ grid**; **hydrostatic equilibrium** intro + the closed-form P=gτ.
- **L2** Boltzmann; partition functions; **Saha**; Debye pressure-ionization; charge conservation → nₑ.
- **L3** continuum opacity; H⁻ (Saha balance→John fits); Rayleigh/Thomson; stimulated emission; then the **exact KAPP engine** (tables, edge grid, MAP1/Karsas/COULFF/Gavrila, 3-pt interp).
- **L4** line strength (log gf); lower-level Boltzmann population; **Voigt profile** (Doppler core + Lorentz wings); damping.
- **L5** line-list anatomy; log-λ grid; exact population (population_per_ion); Voigt Harris tables; cutoff; **wing-accumulation kernel**; metals all-Z + He.
- **L6** hydrogen **Stark broadening**; Holtsmark microfield; HPROF4 (sofbeta, 3-piece profile).
- **L7** transfer equation; **formal solution**; E₂ kernel; Eddington–Barbier; the scattering approximation.
- **L8** moment equations; Eddington closure; the **Λ operator (COEFJ)**; parabolic integ/MAP1; scattering iteration; CH flux.
- **L9** the **exact hydrostatic integrator** (`_ttaup` predictor–corrector in log-pressure, κ≡1 cold start). ← its ONLY new content.

REDUNDANCY FLAGS to fix in the revision pass:
- **L1 ↔ L9 (user-reported):** L9 currently RE-COVERS L1's grey/Hopf T(τ), the 80-layer τ grid, and the hydrostatic-equilibrium intro. TRIM: L9 should recap those in 1–2 sentences + "(Lecture 1)" back-ref and own ONLY the exact `_ttaup` integrator (why P=gτ was approximate → predictor–corrector → bit-exact P/ρ).
- Watch **optical depth** (defined in L1) recurring in L3/L7/L8 — back-ref L1, don't redefine.
- Watch **Saha** (L2) reused in L3 (H⁻) — fine, but back-ref L2.
- Watch L3 internal: the merged physics-half and engine-half both touch H⁻/Rayleigh/Thomson — H⁻ Saha is derived once (good); ensure the engine half references it, not re-derives.
RULE: the revision pass (bite-size + comments + critic prose) ALSO trims redundancy and fixes flow against this map. Update this map whenever a lecture is added/changed.
