---
name: stellar-spectroscopy-from-scratch
description: Single source of truth for the "Stellar Spectroscopy from Scratch" lecture-note book (~/Stellar_Spectroscopy_From_Scratch) — a from-scratch, bite-size, pedagogical rebuild of a synthetic stellar spectrum that must be IDENTICAL (machine precision) to pykurucz. Records the parity standard, house voice, build + cross-check pipeline, per-lecture status, and the plan. When any doc disagrees with this, this wins.
---

# Stellar Spectroscopy from Scratch — Series Bible

A multi-lecture book that rebuilds a synthetic stellar spectrum from first principles, in short bite-size code cells interleaved with pedagogical markdown, **reproducing pykurucz exactly**. Rendered in the *Agents for Astronomy* house format (`~/Agents_for_Astronomy` = rendered; `~/notebooks` = its sources).

## Goal
- Rebuild the full forward problem (atmosphere → EOS → opacity → radiative transfer → spectrum) from scratch, one bite-size step at a time.
- **Match pykurucz (`/Users/ysting/pykurucz`, the GOLD STANDARD) to machine precision** at every step — not "physically close", *identical*.
- Read like the Agents book: pedagogical, accessible-on-first-exposure, equation-complete, no rah-rah.

## THE PARITY STANDARD (non-negotiable — set by the user 2026-06-23)
- pykurucz is the gold standard; **reproduce its ACTUAL algorithm**, never a textbook substitute. Earlier mistake: I used John-1988 H⁻ fits / scipy Faddeeva / the E₂ formal solution — physically reasonable but NOT pykurucz, so they can never match. They must be replaced by clean reimplementations of pykurucz's routines.
- Method: **reuse pykurucz's exact tables/constants as data** (partition functions, Voigt Harris tables, H⁻/Karsas/Gavrila tables, line list), **reimplement the exact numerical steps** cleanly (no `@jit`, caching, guards, CLI). Target ≤1e-7; "machine precision is fine" (fastmath ULP allowed, quantified).
- **Even line-by-line ports stay bite-size**: decompose into small commented cells + markdown (e.g. JOSH = optical-depth scale → moment coefficients → Thomas solve → Λ-iteration), never a monolithic block.
- Approved pattern: use pykurucz's routine for parity; add a markdown aside that an exact alternative (e.g. scipy Faddeeva for the Voigt) is a swap-in for comparison.

## Self-containment & house rules
- Notebooks import **only numpy/matplotlib/pathlib — never pykurucz**. Benchmarks load shipped `reference/*.npz` data files (precomputed once by `_pipeline/make_references.py`, the only place that imports pykurucz).
- pykurucz named **only in the front-matter benchmark line** + Further Reading citation; body reads as self-contained physics. No "harness" / "reproduction harness" wording.
- Do NOT modify pykurucz source; **no git push** anywhere. `git lfs pull` (data) is fine; `scripts/download_data.py` (~5 GB) needed for L7–L9.
- Author byline: **Yuan-Sen Ting — Max Planck Institute for Astronomy & The Ohio State University**.
- Collaboration credit line (title cell): "Written in collaboration with **Claude Opus 4.8**… Schematics with **Gemini 3 Pro** (Nano Banana)."

## Organic flow (user, 2026-06-23)
Lectures must build up **organically** — each stands on the ones before it, define-once, **no forward references** (don't rely on a later lecture's result as if known) and no awkward backward jumps. When inserting the dedicated engine lectures (JOSH, KAPP), keep the sequence linear and renumber cleanly. The LLM cross-check pass must explicitly flag forward/backward-reference breaks.

## House voice (match ~/notebooks/SKILL.md)
Second person ("you"), author "we"; pedagogical, no rah-rah, no AI-tell adverbs (genuinely/simply/actually/really/seamless/robust/crucially/of course/highly/fundamentally/ultimately); equation-complete; **explain every input/parameter** (log gf, χ, .atm columns, damping constants…); short single-purpose code cells (≈3–30 lines) with `# why` comments; intuition → equation → code → parity check. Closing arc each lecture: **Synthesis / Summary / Practice Exercises (3) / Further Reading (4–5)**. Math: `$…$`/`$$…$$`, `\mathrm{}` upright units.

## Build pipeline
- `_pipeline/build_lecture{N}.py` assembles `content/Lecture{N}.ipynb` via nbformat (bite-size md/code cells).
- `_pipeline/build.py {N}` executes (jupyter nbconvert) + renders to `content/Lecture{N}.html` via Node `_pipeline/render_fragment.js` (runs the EXACT Agents `render.js` with npm marked/katex/highlight.js). Registry `LECTURES` mirrors `assets/book-data.js`.
- `assets/book-data.js` manifest (type:"html"); `index.html` shows built + planned; view: `python3 -m http.server 8899` → `reader.html?ch=N`.
- `_pipeline/make_references.py` (imports pykurucz, read-only) → `reference/*.npz` data + `reference/atmosphere.{atm,npz}`, `diag.npz` (synthe `--diagnostics`: per-depth continuum/line opacity + flux), `spectrum_500_510.spec`.

## Cross-check (LLM review — REQUIRED before a lecture is "done", per user)
After parity, cross-check every notebook for pedagogy / accessibility / accuracy while keeping pykurucz parity, using the user's `~/agent4binary` pipeline models:
- **GPT-5.5** via LiteLLM: model `gpt-5.5-2026-04-24`, base_url `https://litellm.cloud.osu.edu`, key `LITELLM_API_KEY` (in ~/.env).
- **Gemini-3.1-pro**: `google.genai`, model `gemini-3.1-pro-preview` (or `gemini-3-pro`), key `GOOGLE_API_KEY` (in ~/.env).
- Helper: `_pipeline/review.py` (structured-JSON critique on each notebook's md+code; records issues; I act on them, keeping parity). Needs `pip install litellm google-genai` in `.venv`.

## Per-lecture status (PARITY AUDIT — 2026-06-23)
| L | Title | Built | Parity |
|---|---|---|---|
| 1 | Overview & a First Model Atmosphere | ✅ | Planck **exact** (9e-16), grey T **exact** (0); P/ρ 1.8e-5 (analytic P=gτ — exact parity deferred to L7's `_ttaup`) |
| 2 | The Equation of State | ✅ | **Saha fractions reproduce pfsaha to ~1e-6 (exact physics)** via the `nion2=min(nion+2,nions)` ladder + shipped U(mode13)/χ(mode15)/Debye tables (L2.npz now has nion2, U(n,99,8), chi(99,8)). Charge-balance **n_e ~3e-4** (residual = partition-fn-reconstruction in the iterative coupling; uniform ~3e-4, worse at top, 7e-6 deep). To get n_e to machine precision needs reproducing pfsaha's NNN partition-fn interpolation exactly = heavy table machinery, low pedagogy. **The canonical atmosphere carries the EXACT reference XNE, so the spectrum stays identical.** DECISION (user): accept this level, prioritize the engines. L2 notebook FINALIZED — uses the nion2 ladder (charges summed over nion), n_e ~5e-4 framed honestly, exact XNE adopted downstream. |
| 3 | Continuous Opacity | ✅ | ~2% (John/Dalgarno fits) → **rework to KAPP** (H⁻ HMINOP, Karsas, Gavrila tables, exact formulas) |
| 4 | Line Opacity I: A Single Line | ✅ | Voigt **EXACT 1.5e-15** (Harris-table `voigt_profile_jit` reproduced via reference/L4.npz h0/h1/h2tab) |
| 5 | Line Opacity II: The Line List | ✅ | formula already identical; 42× was pop normalization → use `population_per_ion` (atmosphere.npz) + reproduce `accumulate_voigt_wings` kernel |
| 6 | Radiative Transfer & the Spectrum | ✅ | spectrum 5.6e-4 (formal solution) → **rework to JOSH** (`josh_solver.solve_josh_flux`: Eddington XTAU grid + COEFJ tables + Thomas + Λ-iteration) |
| 7 | Hydrostatic Eq & Temperature Structure | ⬜ | build to parity (port `_ttaup`); needs ~5GB data |
| 8 | Radiative Eq & Temperature Correction | ⬜ | ATLAS `tcorr.py` + emulator note; end-to-end milestone |
| 9 | Molecular Equilibrium & Bands | ⬜ | needs `scripts/download_data.py` ~5GB; `mol_populations`/`mol_opacity` |
| 10 | Capstone | ⬜ | abundances, instrumental convolution, LTE→NLTE |

## STRUCTURE UPDATE (user, 2026-06-23): dedicate lectures to the heavy engines
The production engines (JOSH RT solver, KAPP continuum) are too large to cram into one bite-size cell each. So SPLIT them into dedicated lectures so the reproduction is digestible AND bit-exact:
- L6 keeps the RT *physics* (formal solution F=2pi*int S E2 dtau → spectrum to 5.6e-4, Eddington-Barbier) — the intuition.
- NEW **"The JOSH Solver" lecture** (production RT engine, bit-exact): optical-depth (`_integ`/`_parcoe`) → MAP1 onto XTAU_GRID → COEFJ Λ-iteration → CH_WEIGHTS surface flux, each a verified bite-size step; feeds reference opacities → machine-precision flux/spectrum.
- Optionally a NEW **"The KAPP Continuum" lecture** if L3's per-source KAPP reproduction is too much for one lecture (L3 keeps the H⁻/scattering physics; the engine lecture does every source bit-exact).
- Renumber the later lectures after inserting (atmosphere build / molecules / capstone shift down). Keep book-data.js + index.html + build.py in sync.

PROGRESS on the JOSH engine — **MILESTONE CONFIRMED (2026-06-23): the JOSH engine reproduces the SPECTRUM to MACHINE PRECISION.**
- `_integ`+`_parcoe` (optical-depth integrator) reproduced BIT-EXACT (max|rel|=0.0), saved in proto_josh.py.
- Verified the engine→spectrum path: `solve_josh_flux(acont,scont,aline,sline,sigmac,sigmal,rhox)` per wavelength on the reference diag opacities, normalised spectrum (ft/fc) vs diag(flux_total/flux_continuum): with driver sources scont=slinec, sline=line_source → **median 4.2e-16, max 7.6e-9** (max = float32-iteration ULP); with self-contained raw Planck S=B (scont=sline=planck(nu,T) from L1) → **median 3.0e-8, max 8.8e-6** (well within "machine precision is fine").
- IMPORTANT unit fact: solve_josh_flux returns the Eddington flux HNU (~1e-5 scale); diag.flux_total is HNU converted to physical .spec units by the driver (~1.2e12x). For the NORMALISED spectrum (line depths) the scale CANCELS, so the from-scratch JOSH gives the bit-exact normalised spectrum directly. (Absolute .spec flux would need the driver's unit conversion — not needed for line depths.)
- **Dedicated JOSH lecture is SELF-CONTAINED**: LTE source S=B (Planck, L1) + opacities (L3 continuum + L5 lines) + reproduced solve_josh_flux → normalised spectrum to ~3e-8 (note slinec's tiny continuum-source b-departure correction takes it to 4e-16 if wanted).
**DONE (2026-06-23): full from-scratch `solve_josh_flux` reproduced + verified + written as Lecture 7.**
- `_pipeline/verify_josh.py` is the canonical from-scratch port: `parcoe`/`integ` (Kurucz PARCOE/INTEG — note integ uses each interval's LEFT-point parabola; PARCOE forces pts 2,3 linear + curvature-weighted blend; a NAIVE parcoe gives ~1.5e-3 error, must port exactly), `map1` (parabolic interp onto the grid), `josh_iterate` (backward Gauss-Seidel, FLOAT32, S=(1-α)S̄+α·COEFJ·S, tol 1e-5). Verified per-wavelength vs pykurucz solve_josh_flux: **max 8.9e-9, median 2.5e-12**; assembled normalised spectrum vs reference: **median 2.5e-12** (max 8.9e-9 = float32-iter ULP). Tables shipped as `reference/josh_tables.npz` (xtau, ch, coefj, rhox); opacities/flux from `reference/diag.npz`.
- Constants: EPS=1e-38, ITER_TOL=1e-5, MAX_ITER=NXTAU=51, float32 iteration. The label-401 saturated-core path (taunu[0] > XTAU_GRID[-1]) does NOT trigger in the 500-510 optical-Sun window — only needed for cool-star molecular band heads (note it in the molecules lecture).
- Lecture 7 = `content/Lecture7.{ipynb,html}` (builder `_pipeline/build_lecture7.py`, 30 cells, 0 errors): moments → Eddington closure → COEFJ/CH tables → parcoe/integ → map1 → scattering iteration → CH flux → full-spectrum benchmark (median 2.5e-12). Registered in build.py (7), book-data.js (Part II), index.html PLANNED. L6 keeps the formal-solution physics and now bridges to L7 (the "deviation" section reframed to "the one approximation, removed by L7"). Renumbering: JOSH=L7; future atmosphere lectures shift to L8 (hydrostatic), L9 (radiative eq), L10 (molecules), L11 (capstone).

SCHEMATICS pipeline (`_pipeline/gen_schematics.py`, Gemini `gemini-3-pro-image-preview` / Nano Banana, GOOGLE_API_KEY from ~/.env, idempotent skip-existing): 8 figures s1_pipeline, s1_optical_depth, s2_saha_boltzmann, s3_hminus, s4_voigt, s5_linelist, s6_rt, s7_josh in resources/figures/, embedded in L1-L7 as `![caption](resources/figures/sN.png)`. STYLE constant S = polished high-end-textbook look (charcoal ink, soft slate fills #EEF2F7, single amber accent #E8A23D, Inter/Helvetica feel, quiet grey caption) — the first flat black-on-white version was rejected as "ugly"; this one approved. RULE: always Read each generated PNG to verify accuracy (AI garbles rotated text + can draw physics wrong, e.g. opacity as dips not spikes); regenerate offenders. Avoid rotated/vertical text in prompts.

## PLAN (user-set order: fix existing → cross-check → L7-L10)
1. **Parity rework, depth-first by dependency:** L2 (exact populations+n_e) → L3 (KAPP) & L5 (populations + wing kernel) → L6 (JOSH). L1-P parity lands in L7.
2. **LLM cross-check** all notebooks (GPT-5.5 + Gemini-3.1-pro) for pedagogy/accessibility/accuracy; fix while keeping parity.
3. **Build L7–L10** to strict parity from the start (download the ~5GB data first).
4. **Update this SKILL.md as you go** (status table, decisions, dates).

## Key reproduction facts (so reruns are exact)
- EOS: solar abund = `pykurucz.compute_abundances` (97 log vals Z=3..99) + `compute_h` + `HE_ABUNDANCE`=0.07837 → linear xab (sum=1). Saha prefactor 2.4148e15. Debye potlow=min(1,1.44e-7/√(tk/(12.5664·(4.801e-10)²·chargesq))). pfsaha modes: 12=fractions, 13=U, 15=cumulative ip; charge ladder over nion2 stages, sum charges over nion (nelect).
- Voigt: `VoigtTables.build()` → h0=e^{-v²}, h1 (reference table), h2=(1-2v²)e^{-v²}; 3 branches (a<0.2 / a>1.4 or a+|v|>3.2 / else). Reproduced exact.
- Line opacity: KAPPA0 = (0.026538/1.77245)·gf/ν · XNFPEL/(ρ·DOPPLE) · BOLT; DOPPLE=Δλ/λ=v_D/c; stim applied at accumulation. XNFPEL = population_per_ion = n_ion/U.
- Continuum: `build_depth_continuum(atm, wl)` → cm²/g; KAPP sources in kapp.py (Thomson 0.6653e-24; H⁻, H Karsas, Rayleigh Gavrila tables in synthe_py/data/kapp_tables.npz, karsas_tables.npz).
- RT (L6 JOSH — turnkey recipe to reproduce bit-exact, ~600 lines, do in a verification SCRIPT first, then swap into the notebook so the live L6 never breaks): `synthe_py/physics/josh_solver.py` — `solve_josh_flux(acont,scont,aline,sline,sigmac,sigmal,column_mass)→flux` (lines 370-809), `_integ(x,f,start)` parabolic integration (174-223), `_map1`/`_map1_kernel` parabolic interp (224-369). Tables (REUSE AS DATA): `from synthe_py.physics.josh_tables import XTAU_GRID(51,), CH_WEIGHTS(51,), COEFJ_MATRIX(51,51)`. Full algorithm (verified pieces marked): abtot=acont+aline+sigmac+sigmal (max EPS); alpha=clip((sigmac+sigmal)/abtot,0,1); snubar=where(acont+aline>0,(acont*scont+aline*sline)/(acont+aline),scont); rho INCREASING surface→deep (reverse if rho[0]>rho[-1], and reverse snubar/alpha to match); **taunu=_integ(rho,abtot,abtot[0]*rho[0]) — _parcoe+_integ VERIFIED BIT-EXACT, saved in _pipeline/proto_josh.py**; xsbar,_=_map1(taunu,snubar,XTAU_GRID); xalpha,_=_map1(taunu,alpha,XTAU_GRID) (clip xalpha 0..1, xsbar≥EPS; mask XTAU_GRID<taunu[0] → xsbar=snubar[0],xalpha=alpha[0]); xsbar_modified=xsbar*(1-xalpha); diag=1-xalpha*COEFJ_DIAG; XS Λ-iteration `_josh_iteration_kernel`: DELXS=(Σ_M COEFJ_MATRIX[K,M]*XS[M])*xalpha[K]+xsbar_modified[K]-XS[K])/diag[K], XS+=DELXS, until Σ|rel|<ITER_TOL or MAX_ITER (USE_FLOAT32_ITERATION → do the iteration in float32 to match Fortran REAL*4 — needed for bit-parity); **surface flux HNU=Σ CH_WEIGHTS*XS** (np.dot). SATURATED-CORE path (label 401, when taunu[0]>XTAU_GRID[-1]): solve on the physical taunu grid with _deriv (HNU=_deriv(taunu,snu)[0]/3, Λ-iterate snu=(1-alpha)*snubar+alpha*(jmins+snu)), with a stability guard for tiny deep dtau. Constants COEFJ_MATRIX/COEFJ_DIAG/CH_WEIGHTS/XTAU_GRID + EPS/ITER_TOL/MAX_ITER/USE_FLOAT32_ITERATION near josh_solver.py top + josh_tables.py. Decompose into bite-size cells: optical depth → MAP1 to Eddington grid → (1-alpha) source mod → COEFJ Λ-iteration → CH-weighted flux (+ note the 401 path). Reference: diag flux_total/continuum; feed L6.npz total_abs/total_scat → expect machine precision (float32 iteration matters).
- L3 KAPP (continuum): `synthe_py/physics/kapp.py compute_kapp_continuum` (~lines 917-2400). Sources + tables to reuse: H⁻ bf/ff (HMINOP_* tables in kapp_tables.npz; FFTHETA Gaunt), H I bf/ff (Karsas xkarsas, karsas_tables.npz), Thomson SIGEL=0.6653e-24*XNE/RHO, Rayleigh H I (Gavrila HRAYOP_GAVRILA* tables), He, metals (continua.dat). Reference: build_depth_continuum(atm,wl) per-source not split → reproduce each source, sum, match cm²/g. Decompose by source into bite-size cells.
- Regen diag from pykurucz dir: `python -m synthe_py.cli reference/atmosphere.atm lines/gfallvac.latest --npz reference/atmosphere.npz --spec reference/spectrum_500_510.spec --wl-start 500 --wl-end 510 --resolution 300000 --no-molecular-lines --diagnostics reference/diag.npz`.

## References
Gray (2005); Mihalas (1978); Hubeny & Mihalas (2014); Kurucz ATLAS/SYNTHE; Kim & Ting (2026) pykurucz (arXiv:2603.11693); John (1988) H⁻ (for the swap-in aside only). Models for cross-check from ~/agent4binary.
