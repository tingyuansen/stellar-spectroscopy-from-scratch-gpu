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
**Strict-parity rework: get EVERY built lecture to machine precision → then push to a new
private GitHub repo + public GitHub Pages → then build the rest (atmosphere/molecules/capstone).**
(User directive: machine precision for all built lectures first, so nothing is silently missing.)

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
- No `\boxed{}` on equations. Inline math: never a space before a closing `$` (KaTeX breaks).
- Byline affiliation: "Max Planck Institute for Astronomy & The Ohio State University".
- Engines get their OWN dedicated lectures (JOSH, KAPP, hydrogen) — keep each digestible,
  flowing logically from its physics counterpart.

## Parity status (per lecture) — FINAL 1–9 numbering (renumber DONE)
| Lecture (live in book) | vs pykurucz | machine precision? |
|---|---|---|
| L1 grey T(τ) | 0 exact | ✅ |
| L1 pressure/ρ | 1.8e-5 (closed form vs `_ttaup`) | user: leave as-is (fine) |
| L2 EOS / nₑ | 1.45e-6 | ✅ (KEV=8.6171e-5) |
| L3 continuum (John-fit physics) | ~2.4% | intuition lecture; exact engine = L4 KAPP ✅ |
| L4 KAPP continuum engine | photosphere 4.25e-15 | ✅ (was LectureKAPP/key 90) |
| L5 Voigt (single line) | 1.5e-15 | ✅ (was old L4) |
| L6 line opacity (atomic: metals+He) | 2.25e-15 | ✅ (was old L5) |
| L7 Hydrogen Stark lines | 8.06e-16 (vs gt_ahline; median 0) | ✅ (was LectureHlines/key 91) |
| L8 formal solution | 5.6e-4 | by design (exact version = L9 JOSH) (was old L6) |
| L9 JOSH | 2.5e-12 | ✅ (was old L7) |

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

### Renumber — ✅ DONE (final 1–9 numbering live)
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

## Final structure plan (approved by user)
- Part I Foundations: 1 grey atmosphere · 2 EOS
- Part II Continuous opacity: 3 continuum physics · **4 KAPP engine** (new)
- Part III Line opacity: 5 single line · 6 line list (metals) · **7 hydrogen Stark lines** (new)
- Part IV Radiative transfer: 8 formal solution · 9 JOSH engine
- Part V Atmosphere: 10 hydrostatic · 11 radiative equilibrium
- Part VI Complexity: 12 molecules · 13 capstone

Currently built: L1–L7 (JOSH=7). New engine lectures are built under descriptive slugs
(LectureKAPP, LectureHlines); **renumber to the above as the LAST step before push** —
grep every "Lecture N" cross-reference so none break. Registries to update on renumber:
`_pipeline/build.py` LECTURES, `assets/book-data.js`, `index.html` PLANNED.

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
