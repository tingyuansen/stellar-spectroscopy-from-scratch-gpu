# Payne Zero Textbook — Live Passdown

**If you are a new agent, read this file first, then `BIBLE.md` (quality
standard), `PLAN.md` (architecture and backlog), `COVERAGE.md` (scientific
ledger), and `design/global_chapter_contracts.md` (chapter boundaries and
notation).** Those five documents plus `design/pedagogical_flow_rubric.md` are
sufficient to continue without any prior conversation.

## Document ownership

| File | Owns |
|---|---|
| `README.md` | Public entry: what the book is, how to build and read it |
| `BIBLE.md` | Quality and pedagogical standard — the bar prose must clear |
| `PLAN.md` | 15-chapter architecture, the six whole-book passes, backlog |
| `COVERAGE.md` | Scientific completeness ledger against Payne Zero |
| `PASSDOWN.md` | **This file** — live state and the next concrete actions |

Do not duplicate live state into `PLAN.md`; it belongs here.

## What this book is

A from-scratch stellar spectroscopy textbook for a final-year undergraduate or
first-year graduate student, anchored on the working Payne Zero implementation.
Scope is **atmosphere + synthesis + the atmosphere emulator. Fitting is out of
scope.** The reader must be able to rebuild Payne Zero's atmosphere and
synthesis pathways from the book's own code and data, ending with output
identical to Payne Zero.

Non-negotiable style rules live in `BIBLE.md`. The short form: explain all the
physics, assume only basic maths and physics, bite-size code cells, no large
code blocks inside Markdown, no source-file tours, professional one-panel plots,
schematics in the payne-zero-website aesthetic, every chapter closes with a
summary and a next-chapter link, no exercises, self-contained rather than
constantly deferring to Payne Zero, and Payne Zero's own notation throughout.

## Read-only external authority

Never modify either tree.

- `/Users/ysting/payne-zero` — pinned commit
  `9c44001feae40b85146630499e6f8a5fed42e5af`
- `/Users/ysting/Source_Files_Not_For_Review` — the paper; `main.tex` SHA-256
  `e11507b9150550b246f6664debf22e540aa92d8261eb40daabb594da91bd8e0d`
- `/Users/ysting/payne-zero-website` — schematic generation code and aesthetic;
  regenerate original schematics from the same `.py` approach rather than
  copying website assets.

## Cadence

**End-to-end first.** Keep a complete, runnable, rendered 15-chapter book after
every revision wave. Do not perfect one chapter behind acceptance ceremony while
later chapters stay thin. Scientific discrepancies block the affected claim;
metadata ceremony does not block unrelated chapters.

The live reader is `http://127.0.0.1:8765/reader.html`.

## Verified state — 2026-07-31

- `scripts/verify_pinned_source_fragments.py` **passes**: all chapter-stage
  fragments match Payne Zero `9c44001`. 58 staged modules.
- 711 tests collect. A per-file sweep of all 55 test files accounts for every
  one of them: **709 passed, 2 skipped, 0 failed** after this session's fixes.
  The 2026-07-30 ledger entry of "9 failed, 26 errors" is superseded — those
  were repaired by later waves. The only failures the sweep found were the 10
  this session introduced and then fixed (8 summary-heading assertions plus 2
  molecular-lane anchors). The sweep was re-run after all the prose work below
  and still reports 0 failed.
- `python scripts/check_section_references.py` passes: all 228 section headings
  are defined and every `§N.M` cross-reference resolves.
- `tests/test_symbol_coverage.py` is **slow, not hung**: 42 tests in 799 s
  (13 min) at 100 % CPU. It is the sole reason a plain `python -m pytest`
  appears to stall. Give it its own generous budget and run the other 54 files
  per-file when you need a fast signal.
- All 15 chapter sources, notebooks, HTML fragments, and registry entries exist;
  the reader publishes Chapters 1–15.
- All 15 generated notebooks match their canonical Python cell sources.
- Chapters 11 → 12 → 13 pass an exact live-state atmosphere handoff.
- The exact solar physical atmosphere converges on pass four; deep-layer
  relative temperature change `4.778548e-4 < 5e-4`.
- Three exact solar atmosphere runs — staged, staged repeat, and pinned
  read-only oracle — produce identical 27 arrays and byte-identical source
  archives, SHA-256
  `14e552717e0bbf5eb263deec043c16d3ef796708b5328d822f8a2e2e06fb1fbc`.
- Their four physical spectrum arrays are bitwise-identical; timing-free payload
  SHA-256 `5e2b65add5326a9bfa0442216b8198b225305bc1ed0a05325858b34b2f345f27`.
- The render pipeline was verified after relocation (see Cleanup below): a fresh
  render of `content/Chapter01.ipynb` is byte-faithful to the committed HTML.

Not yet evaluated, and must stay visibly absent rather than assumed:

- flux-error and hydrostatic-residual acceptance thresholds for the retained
  solar product;
- a separately retained standard-optical-grid acceptance record;
- full physical trajectories for the hot dwarf, low-gravity giant, and cool
  molecule-rich capstone requests.

## Completed 2026-07-31 — repository cleanup

(Book-content work from the same date is under *Completed 2026-07-31* below.)

Removed, all recoverable via `git checkout HEAD~1 -- <path>` — every one was
verified unreferenced by live code before deletion:

- `_pipeline/` — 72 legacy build/verify scripts from the superseded
  Lecture-based book. The single live file, `render_fragment.js`, moved to
  `scripts/render_fragment.js`; `scripts/build_book.py` updated.
- `content/Lecture1–16.{ipynb,html}` — superseded by `Chapter01–15`.
- `reference/` (369 MB) and `resources/` — legacy data and figures for the old
  book.
- Committed `.pytest_cache` / `.ruff_cache`; `.gitignore` simplified.

Working tree is ~310 MB, down from ~690 MB. Root now holds only the five
Markdown documents, the reader entry points, and the live source trees.

## Per-chapter design documents — corrected picture

An earlier reading of this gap was wrong and is corrected here. `design/` holds
**two alternative formats from two eras**, not one complete set and one
incomplete set:

| | Ch1 | Ch2–4 | Ch5–6 | Ch7–15 |
|---|---|---|---|---|
| `first_pass_contract` | — | — | — | ✅ |
| `causal_outline` | — | ✅ | ✅ | — |
| `exact_source_contract` | — | ✅ | ✅ | — |
| `acceptance` | ✅ *(added 2026-07-31)* | ✅ | ✅ *(added 2026-07-31)* | ✅ *(added 2026-07-31)* |

Chapter 7's `first_pass_contract` is 1,249 lines and already covers what
`causal_outline` covers — canonical placement and boundaries, the chapter's
single question, reader promise and prerequisites, a notation and decision
ledger, and per-movement section detail — **plus** a visible code-cell ledger
and a figure/schematic contract that the older format lacks. Regenerating
`causal_outline` for Ch7–15 would duplicate existing material in an older
format, which is exactly the metadata ceremony the Cadence section warns
against. Do not do it without a specific reason.

**Remaining real gaps, in priority order:**

1. **Chapter 1's two audits predate the text they describe.**
   `chapter01_rewrite_audit.md` and `chapter01_flow_audit.md` returned
   conditional verdicts against an earlier draft. Their headline findings
   measurably no longer apply — the rewrite audit counted 151 lines of source
   across five Markdown cells, and the count is now zero — but "appears
   addressed" is not a fresh ACCEPT. Re-run them, or mark them historical.
2. **Exact-source depth.** `first_pass_contract` is lighter here than
   `exact_source_contract` was (Ch7 names the pinned packages 7 times against
   Ch3's 19). This is descriptive rather than load-bearing, because
   `verify_pinned_source_fragments.py` already enforces exactness mechanically
   for every chapter — so treat it as low priority.
3. **No independent pedagogy audit exists for Ch5–15.** Chapters 2–4 carry one.
   Every Ch7–14 acceptance record states this openly in its *Not evaluated*
   section. Closing it means running a genuine audit, not writing a document.

## Known pedagogical defects — re-verified 2026-07-31

The 2026-07-30 whole-book audit is **partly stale**; several of its findings had
already been repaired by later waves. Always re-measure before acting on an
audit note. Current measured state, per generated notebook:

| ch | prose words | visible cells | code lines | cells over 30-line *target* | figures |
|---:|---:|---:|---:|---:|---:|
| 1 | 4991 | 15 | 252 | 1 | 5 |
| 2 | 4516 | 21 | 453 | 3 | 2 |
| 3 | 6058 | 21 | 585 | 10 | 3 |
| 4 | 5668 | 17 | 480 | 10 | 2 |
| 5 | 5342 | 16 | 451 | 6 | 6 |
| 6 | 3358 | 15 | 339 | 3 | 4 |
| 7 | 3576 | 18 | 372 | 1 | 6 |
| 8 | 3893 | 17 | 375 | 3 | 4 |
| 9 | 3250 | 18 | 449 | 4 | 5 |
| 10 | 4618 | 18 | 296 | 1 | 2 |
| 11 | 3172 | 19 | 196 | 0 | 2 |
| 12 | 3361 | 21 | 248 | 0 | 2 |
| 13 | 3009 | 15 | 259 | 0 | 4 |
| 14 | 2758 | 20 | 267 | 0 | 6 |
| 15 | 3293 | 16 | 274 | 1 | 2 |

Regenerate this table after substantive edits: count markdown words, visible
code cells, nonblank code lines, and `image/png` outputs per
`content/ChapterNN.ipynb`. Note the fourth column counts cells over the 10–30
line *target*, **not** violations — `BIBLE.md:219` puts the soft ceiling at 60
and the hard ceiling at 80, and nothing in the book exceeds either.

**Verified fixed 2026-07-31 — do not re-report.** Each was checked against the
current source, not assumed:

- Ch12 title mismatch — `chapter_12.py:8` and `registry.py:130` now agree.
- Ch10→Ch11 "converged" contradiction — Ch11 now opens "Chapter 10 deliberately
  accepted a supplied, schema-valid atmosphere … It did not claim that the
  supplied structure was converged."
- Ch1 naming late API concepts early — no occurrence of `Payne Zero`,
  `InitializedAtmosphere`, or `Chapter 11–14` remains in `chapter_01.py`.
- Ch2 teaching Ch14–15 material early — no `centidex`, `InitializedAtmosphere`,
  97-slot, 81-abundance, or 25-array/schema-v4 material remains in
  `chapter_02.py`. Density also fell from 5,490→4,516 words, 31→21 cells,
  734→453 lines.
- Ch15 "has no plot or schematic" — false. §15.9 already carries a one-panel
  four-regime normalized-flux comparison, and §15.2 embeds the
  `ch15-two-workflow-gates-v1` schematic.
- Ch12/Ch15 thinness — partly resolved: 1,228→3,259 and 966→2,518 words.
- Closing-heading drift — fixed this session, see *Completed* below.

**Resolved scope question (author decision, 2026-07-31):** equivalent width,
saturation, and the curve of growth **are in scope**, even though Payne Zero does
not compute them, because they explain the spectra the book synthesizes. Now
taught in Ch9 §9.14. Precedent: physics needed to *understand* the output may be
taught as a clearly-labelled controlled limit, provided the production path still
computes the real thing.

**Open — all measured, not inherited:**

1. **Density imbalance — much reduced, not gone.** The spread was 1,846–6,058
   prose words; it is now 2,333–6,058 after the atmosphere-arc pass. The
   remaining outliers are Ch3 (6,058) at the top and Ch13 (2,333), Ch9 (2,456),
   and Ch14 (2,492) at the bottom. **Ch9 is now the worst prose-to-code ratio in
   the book** — 2,456 words carrying 427 code lines — and is a better target
   than raw word count suggests. Do not change the 15-chapter count.
2. **Cell length — a mild target overshoot, not a violation.** `BIBLE.md:219`
   sets the target at 10–30 lines with a **soft ceiling of 60 and a hard ceiling
   of 80**. Measured across all 264 visible code cells: 43 exceed the 30-line
   target, but **zero exceed either ceiling**. The largest cell in the book is
   46 lines (Ch8); Ch3 and Ch4's ten "oversize" cells each sit at 31–35. Earlier
   audits reported this as a bite-size failure by counting against the target
   rather than the ceiling. Treat it as low-priority polish: split a cell only
   where it genuinely carries two ideas, not to satisfy a line count.
3. **Figure starvation — closed.** Every chapter now has at least two figures.
   Ch11, Ch12, and Ch15 each gained one during the 2026-07-31 wave.
   Ch15's capstone figure also spans only a 0.2 nm window (498.95–499.15 nm),
   narrow for a four-regime comparison — consider a wider or second panel
   showing where the regimes visibly diverge, now that §15.10 explains what to
   look for.
4. **Display-math inconsistency.** Ch1 uses `$$`; Ch5/9/11/12/13 use `\[ \]`.
   Both render under KaTeX, but the book should pick one — `\[ \]` is the
   majority convention.
5. **Ch12 is the only chapter using `## Act` / `### N.N` nesting**; every other
   chapter is flat `## N.N`. This is part of why it reads differently from its
   neighbours. Consider flattening.

**Method note.** Five of the eight findings in the 2026-07-30 audit were
already repaired by the time they were acted on. Re-measure before editing;
prefer a grep or a notebook measurement over trusting any inherited defect
list, including this one.

## Completed 2026-07-31 — book content

- **Repository cleanup** — see *Completed 2026-07-31 — repository cleanup*
  above.
- **Chapter summary headings normalized.** Chapters 7–15 used
  `### Chapter summary`, violating the `## N.N Chapter summary` requirement at
  `design/global_chapter_contracts.md:751-764`. All nine now carry correct
  numbered headings (7.18, 8.18, 9.15, 10.12, 11.13, 12.24, 13.10, 14.8, 15.17).
  The eight `tests/test_chapterNN_runtime.py` assertions that encoded the *old*
  wrong convention were updated to match the contract. All nine chapters were
  rebuilt so stored notebooks carry no drift.
- **Chapter 12 convection physics derived, not asserted.** Added §12.19
  (Schwarzschild criterion, derived from the buoyancy argument via the parcel
  displacement), §12.20 (adiabatic gradient from the first law,
  \(\nabla_{\rm ad}=(\gamma-1)/\gamma=2/5\) for a monatomic ideal gas, why
  partial ionization lowers it, and the pressure scale height from Chapter 1's
  hydrostatic balance), and §12.21 (temperature excess, buoyant velocity, and the
  \((\nabla-\nabla_{\rm ad})^{3/2}\) enthalpy flux). This also supplies the
  causal link that was missing between the perturbed-EOS machinery of
  §§12.14–12.18 and convection: those derivatives exist precisely to give
  \(c_P\) and \(\nabla_{\rm ad}\). Ch12 prose 2,493 → 3,259 words; the chapter
  rebuilds with zero execution errors.
- **Chapter 13 temperature correction derived, not listed.** §13.2 previously
  introduced the three correction terms as a bullet list of implementation
  field names. Added §13.2 (deep layers: the Eddington closure and
  \(dJ/d\tau_{\rm R}=3H\) give
  \(\Delta T=\tfrac{3\pi}{4\sigma T^3}\int_0^{\tau}[H_\star-H]\,d\tau'\), so the
  flux term is necessarily an *integral* — one bad layer misplaces every layer
  beneath it) and §13.3 (shallow layers: the diffusion argument fails, radiative
  equilibrium \(\int\kappa_\nu(J_\nu-S_\nu)d\nu=0\) is what survives, and
  keeping only the diagonal \(\Lambda_{dd}\) of the lambda operator yields the
  local Newton step — hence why the outermost layer needs a third, separate
  boundary repair, and why all three are linearizations that the §13.5
  safeguards must protect). Old §§13.2–13.9 renumbered to 13.4–13.11, summary to
  13.12, with the matching test assertion updated. Ch13 prose 1,846 → 2,333
  words; rebuilds with zero execution errors; 20 chapter tests pass.
- **Chapter 14 PCA explained, not asserted.** §14.2 previously stated that PCA
  keeps 160 of 480 coordinates without saying what PCA does — unacceptable when
  `BIBLE.md` assumes no prior machine-learning knowledge. Added a new §14.2
  deriving it: the 480 numbers are not independent because the preceding
  thirteen chapters *are* the constraints among them (grey temperature law,
  hydrostatic balance, EOS), so a real atmosphere lies on a thin surface;
  covariance eigenvectors ordered by variance give the best linear \(k\)-summary,
  and the network only predicts the coefficients. The section closes on the
  caveat that carries the chapter: discarded directions are negligible *for the
  training distribution only*, so an out-of-distribution request returns a
  confident, smooth, wrong profile with no internal warning — which is what makes
  the §14.8 closure requirement mandatory rather than procedural. Old §§14.2–14.8
  renumbered to 14.3–14.9 with the matching test assertion updated. Ch14 prose
  2,009 → 2,492 words; zero execution errors; 46 tests pass.
- **Chapter 11 backwarming derived.** §11.12 previously disposed of line
  blanketing in one sentence — "photons blocked in those frequencies must escape
  elsewhere" — and the word *backwarming* did not appear in the chapter at all.
  Added a new §11.12 that runs the flux constraint through: every layer must
  still carry \(\sigma_{\rm SB}T_{\rm eff}^4\); the harmonic Rosseland mean is
  dominated by transparent windows, which lines close, so \(\kappa_R\) rises;
  at fixed flux the diffusion relation then forces a steeper \(|dT/dr|\), so
  every deep layer is hotter than its grey twin. Includes the companion surface
  cooling, and closes by naming blanketing as the reason Chapter 13's correction
  must exist — the blanketed opacity changes the flux the next pass measures, and
  that gap *is* \(\delta_H\). Old §11.12 became §11.13, summary 11.14, test
  assertion updated. Ch11 prose 2,282 → 2,732 words; zero execution errors;
  11 tests pass.
- **Chapter 15 capstone made physical.** §15.3 asserted one sentence per regime
  and H\(^-\) — the dominant optical continuum opacity in solar-type and cooler
  stars — appeared nowhere in the chapter. Added §15.10, a physical reading of
  the four-star figure: the solar continuum as H\(^-\), needing neutral H *and*
  metal-donated electrons (Chapter 3 Saha); the hot dwarf ionizing hydrogen and
  so destroying H\(^-\)'s partner while thinning the neutral-metal forest; the
  giant's \(P=gm\) (§1.10) lowering electron density and narrowing collisional
  damping wings (Chapter 7), which *derives* the luminosity-class diagnostic
  rather than asserting it; and the cool dwarf's TiO/water bands making the
  continuum notional and driving convection through §12.19's criterion. Old
  §§15.10–15.16 renumbered to 15.11–15.17, summary 15.18, test updated. Ch15
  prose 2,518 → 2,993 words; zero execution errors; 17 tests pass.
- **Chapter 12 Schwarzschild figure added.** The checkpoint already exposed
  `logarithmic_gradient` and `adiabatic_gradient`, so §12.19 now evaluates the
  criterion visually instead of asserting it. On the six-layer teaching fixture
  it correctly reports 0 unstable layers of 6, which the following prose
  explains rather than hides. Ch12 figures 1 → 2, visible cells 20 → 21 (the
  `test_..._twenty_cell_spine` assertion was a snapshot, not a contract limit;
  the contract's >18 is a review trigger). Ch12 is now the densest chapter at 21
  visible cells — watch it before adding more.
- **Chapter 9 scattering physics derived.** The chapter said "scattering" 46
  times without once stating the full source function or naming a
  thermalization depth. Added §9.2: combining the two fates of a removed photon
  gives \(S_\nu=(1-\alpha_\nu)S_{\nu,\rm th}+\alpha_\nu J_\nu\); the
  \(J_\nu\) term is what makes the problem nonlocal and is *why* §9.7 must
  iterate; the random-walk argument gives
  \(\tau_{\rm th}\sim(1-\alpha_\nu)^{-1/2}\) and the
  \(\sqrt{1-\alpha_\nu}\,S_{\rm th}\) surface law, which is why a
  scattering-dominated line is darker than pure absorption predicts and why
  §9.11 needs a separate diffusion route when the surface saturates. Uses the
  book's own \(\alpha_\nu\) rather than introducing \(\epsilon\). Old
  §§9.2–9.14 renumbered to 9.3–9.15, summary 9.16, test updated. Ch9 prose
  2,456 → 2,769 words; zero execution errors; 16 tests pass.
- **Chapter 6 Lorentzian derived.** §6.6 listed three damping mechanisms and
  summed their rates into one \(\gamma\) without justifying either the
  Lorentzian shape or the summation, and §6.7 then asserted a "Lorentzian
  damping branch". Added the classical damped-oscillator argument: a finite
  lifetime makes \(E(t)\propto e^{-\gamma t/2}e^{-2\pi i\nu_l t}\), whose
  Fourier transform is a Lorentzian of FWHM \(\gamma/2\pi\); the quantum
  \(\Delta E\,\Delta t\) route gives the same width; and the
  \((\nu-\nu_l)^{-2}\) tail is why the damping branch always wins far from
  line centre however small \(\gamma\) is. A second short passage explains why
  three distinct mechanisms may be *added* — each independently terminates the
  same wave train, so the rates sum and one Lorentzian describes the result.
  Ch6 prose 3,138 → 3,358 words; zero execution errors; 21 tests pass.
- **Chapter 7 Inglis–Teller exponent derived.** §7.15 stated
  \(n_{\rm IT}=1600\,n_e^{-2/15}\) with the \(2/15\) unexplained, which reads
  as a fitted constant. Added the two-length comparison it comes from: level
  spacing falls as \(n^{-3}\); hydrogen's linear Stark effect splits a level by
  \(n^2E\); the microfield from mean separation \(d\sim n_e^{-1/3}\) is
  \(E\propto n_e^{2/3}\); setting splitting equal to spacing gives
  \(n^5\sim n_e^{-2/3}\), so \(n\propto n_e^{-2/15}\). The exponent is
  \(2/3\) divided by \(5\), not a fit. Ch7 prose 3,405 → 3,576 words; zero
  execution errors; 15 tests pass.
- **Chapter 9 equivalent width and curve of growth added** (author-approved
  scope extension — see the resolved scope question below). This could not go in
  Chapter 6: §6.8 states plainly that line opacity "is still not a flux dip", and
  equivalent width needs *emergent* flux. New §9.14 therefore sits right after the
  first synthesized normalized-flux window. It defines
  \(W_\lambda=\int(1-F_\lambda/F_{\lambda,c})d\lambda\), notes that measuring
  area makes it immune to instrument smearing, measures it on the book's own
  window (1.206401e-2 nm), then derives all three regimes from one integral in a
  Schuster–Schwarzschild slab — a controlled limit in the spirit of Chapter 1's
  grey atmosphere — and states explicitly that the slab explains the answer
  rather than producing it. Verified numerically: \(W/(\sqrt\pi\tau_0)\to1.0\)
  for \(\tau_0\le0.1\) (linear); \(W\) grows only 3.4→9.4 while \(\tau_0\)
  goes \(10^1\to10^3\) (saturated); \(W/\sqrt{\tau_0}\approx0.26\) for
  \(\tau_0\ge10^4\) (damping). Old §§9.14–9.16 renumbered to 9.15–9.17; test
  summary heading and visible-cell count (17 → 18) updated. `continuous_voigt_h`
  is now imported from `book.chapter06_runtime` — the first cross-chapter use of
  that helper; the offset grid is dense through the core and logarithmic across
  the wings because the reference Voigt convolves per sample. Ch9 prose
  2,769 → 3,250 words, figures 4 → 5; zero execution errors; 43 tests pass.

### Implementation-narrative hunt — Ch11/13/14 (2026-07-31)

A second pass over the three chapters that had already had a physics pass,
looking specifically for *implementation* decisions stated without reasons.
Five found and closed; all are prose-only, so no cell counts changed.

- **Ch13 §13.5** listed eight ordered safeguards and asserted that "reordering
  these operations changes the trajectory" without saying why any of them
  exists. Each now names the failure it prevents — surface-suppressed convective
  noise, a non-monotonic \(\tau_{\rm R}\) grid, the weakest linearization
  proposing the largest jump, NaN propagating through the next pass's EOS — and a
  closing paragraph explains why the *order* is load-bearing: bound before
  combining so each term's failure stays visible, damp after combining so
  under-relaxation acts on the total, and let the monotonic walk have the last
  word.
- **Ch13 §13.8** gave the convergence norms as formulas only. Now explains why
  the deep slice drops the outer 39 layers (the §13.3 region, which would
  dominate a max-norm and never settle) and the inner 5 (pinned by the boundary
  condition, so they carry no convergence information); why the norm is a
  maximum rather than an RMS; and why convergence must be *consecutive* — a
  damped oscillation crosses through a small step on its way between sides, so
  one quiet pass proves nothing.
- **Ch13 §13.9** now explains why the chunk reduction is *ordered*: private
  accumulators avoid locks, and fixing the reduction order is what makes a
  bitwise comparison against the pinned oracle meaningful rather than
  scheduler-dependent. Cross-references §12.9 instead of rederiving it.
- **Ch14 §14.5** now says why the in-memory route deliberately discards
  precision it holds: an unquantized in-memory path would let the same star
  enter the solver in two different states depending on invocation. Framed as
  the same rule as Chapter 13's terminal \(Q\), applied at the near end of the
  loop — a lossy format operation may happen, but only at one declared place,
  never inside the fixed-point map.
- **Ch14 §14.7** now explains why the element responses are *summed*: a mixture
  is a set, so the prediction must be order-invariant, and addition has that
  symmetry built in. A concatenating or sequential design would have to spend
  capacity learning an invariance we already know exactly.
- **Ch11 §11.9** now explains why selection uses 344 sparse reference
  coordinates rather than all 30,000 frequencies (the full test would cost what
  the deposit costs), why the filter is deliberately conservative (a dropped
  line silently removes opacity; a kept one costs only arithmetic), and what the
  duplicated entry plus \(2^{30}\) sentinel buys — a branchless inner scan, in
  the same family as the Chapter 2 `prange` work.

Ch11 2,732 → 2,935; Ch13 2,333 → 3,009; Ch14 2,492 → 2,758. Zero execution
errors; 70 tests pass.

## Immediate sequence

1. Continue deepening the thin chapters (open defect 1). Ch9 (2,456 → 2,769),
   Ch11 (2,282 → 2,732), Ch13 (1,846 → 2,333), Ch14 (2,009 → 2,492), and Ch15
   (2,518 → 2,993) are done, as are Ch6 (3,138 → 3,358) and Ch7
   (3,405 → 3,576) — the whole atmosphere arc, transfer, and the line chapters.
   The implementation-narrative hunt over Ch11/13/14 is **done** (see Completed).
   **The thin end is now Ch9 excepted — Ch14 (2,758), Ch11 (2,935), and Ch15
   (2,993)** — but no chapter is below 2,700 and the spread is much tighter than
   it was. Further deepening should be driven by a specific unexplained claim,
   not by word count.

   The pattern that has now worked five times: find where a chapter *names* a
   quantity the implementation computes, and derive it instead. Worked examples
   — Ch9 §9.2 (scattering source function), Ch11 §11.12 (backwarming),
   Ch12 §§12.19–12.21 (Schwarzschild, adiabatic gradient, enthalpy flux),
   Ch13 §§13.2–13.3 (temperature correction), Ch15 §15.10 (why the four regimes
   differ). Each also closed a causal gap between neighbouring chapters, which
   matters more than the word count.
2. Give Ch11 and Ch15 a second figure each (open defect 3) — they are the only
   chapters left with one. Ch11's would naturally show lines raising the
   Rosseland mean, but `BlanketingCheckpoint` does not currently expose the
   layer temperature needed for the \(\partial B_\nu/\partial T\) weighting;
   add that field rather than plotting an unweighted proxy.
3. All 15 chapters now have an acceptance record. The remaining design work is
   judgement, not filing: decide whether to re-run Chapter 1's stale audits and
   whether the absent independent pedagogy audit for Ch5–15 is worth
   commissioning. Do *not* regenerate `causal_outline` for Ch7–15; see the
   corrected picture above for why.
4. Settle the display-math and heading-nesting inconsistencies (defects 4, 5)
   in a single mechanical pass once prose work has settled.

After each step: rebuild the affected chapters with `scripts/build_book.py`,
re-run `verify_pinned_source_fragments.py`, run the affected tests, and update
this file's Verified state, Completed, and Immediate sequence so the next agent
can resume cold.

## Traps that have already cost time

- **Governance anchors are verbatim.** `tests/test_global_molecular_lane_language.py`
  requires exact text blocks in `PASSDOWN.md`, `PLAN.md`, and `BIBLE.md`
  (anchors `passdown-resolved-lane-boundary`, `plan-resolved-lane-boundary`,
  `bible-active-selectors`). Paraphrasing them fails the suite. Edit those
  paragraphs only by copying the required string out of the test.
- **Editing `book/chapters/*.py` creates notebook drift** until you rebuild.
  Always run `scripts/build_book.py <n>` for every chapter you touch.
- **Some `tests/test_chapterNN_runtime.py` assertions encode chapter prose**
  (exact headings, visible code-cell counts). Changing structure means updating
  the matching assertion — check whether the test or the chapter is wrong
  against `design/global_chapter_contracts.md` before assuming the test is right.
- **Renumbering a chapter silently breaks cross-references.** Inserting a
  section shifts every later number, and `§13.5`-style references elsewhere in
  the book keep pointing at the old slot — invisible in the rendered output.
  Run `python scripts/check_section_references.py` after any renumber; it is
  cheap and covers all 227 sections.
- **Two published Chapter 6 artifacts require filesystem mode `0600`, which
  git cannot record.** A fresh clone — or any `git checkout` that rewrites the
  working tree, such as a branch merge — leaves them at the umask default and
  6 publication-authority tests fail with `PublicationStateError: ... not the
  exact single-link mode-0600 artifact`. The contents are fine; only the
  permission bit is lost. Fix with `python scripts/restore_publication_modes.py`.
  This bit me after merging to `main` and cost real time to diagnose.
- **`python -m pytest`, never bare `pytest`** — the latter omits the repo root
  from `sys.path`.
- **The full suite can stall.** Run per-file with a watchdog if `python -m pytest`
  appears to hang at 0% CPU; that isolates the offending file instead of losing
  the whole run.

## Commands

Build and execute selected chapters:

```bash
python scripts/build_book.py 12 15
```

Build the full reader:

```bash
python scripts/build_book.py
```

Verify exact staged source against pinned Payne Zero:

```bash
python scripts/verify_pinned_source_fragments.py
```

Verify every `§N.M` / `Section N.M` cross-reference still resolves — run this
after any chapter renumbering:

```bash
python scripts/check_section_references.py
```

Run the tests — use `python -m pytest`, not bare `pytest`, so the repository
root is on `sys.path`:

```bash
python -m pytest -q
```

Serve the reader locally:

```bash
python -m http.server 8765
```

Jupyter execution needs permission to bind local kernel ports in restricted
environments. Never replace execution with an unexecuted render.

## Role boundaries that must remain visible

1. **Resolved boundary:** H2O compiler parity is retained and verified, while
   the pinned standard synthesis pipeline continues to compile text bands plus
   TiO rather than H2O. This synthesis boundary does not apply to the standard
   atmosphere water selection/deposition path.

- static inputs, computed fixtures, and golden comparison outputs are different
  data roles;
- initializer predictions are starting proposals, never converged atmosphere
  products;
- schema validity proves an interface contract, not physical closure;
- synthesis proves the consequence of a supplied atmosphere, not that
  atmosphere's convergence;
- `r_grid` and `resolution` are interface-specific names and are never silently
  merged;
- standard synthesis omits the separate H2O compiler path, even though Chapter 8
  implements and verifies H2O compiler parity — see the resolved boundary above;
- the direct-abundance initializer remains experimental and requires exact
  physical closure;
- fixed-thread reproducibility is the strict atmosphere target; regrouped
  reductions may move final bits;
- timings are performance observations, not physical identity.

## Integration rule

Subagents may take bounded, nonoverlapping chapters. The primary agent owns
final integration and must recheck global notation, prerequisites, single
ownership of repeated concepts, neighbouring chapter transitions, exact runtime
claims, and rendered output. No subagent draft is accepted solely because its
local tests pass. Preserve unrelated user changes; never clean the tree with
destructive Git commands.
