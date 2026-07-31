# Chapter 6 causal-outline independent re-audit

Status: fresh independent re-audit  
Audited: 2026-07-30  
Disposition: **ACCEPT after final hash/search closure; all three former P0
findings and all targeted P1 corrections are closed**

This is a pedagogical and neighbor-contract rejection, not a scientific,
runtime, or coverage rejection. The repaired outline now has the right causal
architecture. It should become authorable after the finite corrections in
Section 4; none requires another structural rewrite.

No outline, chapter, runtime, test, data, source, paper, or external Payne Zero
file was changed by this re-audit. This report is the only file added.

## 1. Exact review snapshot

The exact revised outline audited here is:

```text
design/chapter06_causal_outline.md
SHA-256 698b3846d6e096037ae7c331b9a4dd1dff38edb3ef5332960b10c649be4ec2e0
```

The governing files were read at these identities:

| authority | SHA-256 |
| --- | --- |
| `BIBLE.md` | `1433c2d3d18dd7397f8a739765c7f8e4c36f4b79e41c2809f596fa7fe3bf59b0` |
| `PLAN.md` | `bf2da67593c8b23c02c82846ef2d55388cce2383c57860719cb5601209342539` |
| `PASSDOWN.md` | `08a07167ace224963eea6220449ba13c3a21b2d626d8bbac03ed0878be5adbeb` |
| `design/global_chapter_contracts.md` | `51ed5e93cdd1787ec355155573bfa20bbc7320d376fc3afa16ff3485438b370c` |
| `design/pedagogical_flow_rubric.md` | `4626b53dd9953df120398fd8762504b9cb1a429b8b0e6df4a1e5a049a433f35d` |
| prior `design/chapter06_pedagogy_neighbor_audit.md` | `60aebd622941a23b555d8c12bdd1411dd4479ee26a7fbbcc7e07d6fc1f38a33c` |

## 2. Former P0 closure

### P0.1 — Transparent physical line before FASTEX and Harris: **CLOSED**

The revised causal order is now correct:

```text
integrated strength
→ Doppler core
→ damping
→ normalized continuous Voigt profile
→ direct-exponential one-depth line
→ FASTEX
→ Harris
→ synthesis shortcut/cutoff
→ all-depth and production products
```

The decisive repair is Section 6.8 at lines 519–550. It explicitly assembles a
one-depth line with a direct excitation exponential and a continuous Voigt
profile before Section 6.9 opens FASTEX and Section 6.10 opens Harris. Section
6.9 even asks the earned numerical question: which table rule replaces the
already-evaluated exponential without changing a cutoff decision.

The visible plan preserves the same order in cells 7–10. Production machinery
no longer interrupts the strength-and-profile derivation.

One wording cleanup remains below: cell 7 should say “continuous mathematical
Voigt reference,” not “continuous/full profile,” because “full Harris” is used
later for a different numerical object. This does not reopen the structural
P0.

### P0.2 — Conceptual \(n_l\) versus executable transition factor: **CLOSED**

Section 6.2 now gives the honest conceptual relation

\[
n_l=\frac{n_{s,r}}{U_{s,r}}g_l e^{-E_l/kT},
\]

then states immediately that the selected record does not supply \(g_l\) and
\(f_{lu}\) separately. It correctly identifies the executable factor as

\[
\frac{n_{s,r}}{U_{s,r}}(gf)_l e^{-E_l/kT}
=n_l f_{lu},
\]

not an independently recovered \(n_l\) and not \(n_l(gf)\).

The destination, cell 2, accepted runtime names, and summary all preserve that
distinction:

- `excitation_weighted_partition_normalized_population_cm3`;
- `gf_weighted_excitation_factor_cm3`;
- no `lower_level_population_cm3` claim;
- no second multiplication by \(g_l\).

The remaining cm\(^{-1}\) notation finding concerns the stored coordinate, not
the \(n_l\) semantics.

### P0.3 — Coherent fifteen-cell learner journey: **CLOSED**

The visible plan has exactly fifteen checkpoints and a sixteen-cell hard
ceiling. Each checkpoint asks one intelligible question. In particular:

- FASTEX, Harris, and the synthesis shortcut are separate cells 8, 9, and 10;
- atmosphere and synthesis parity are separate cells 12 and 13;
- cell 14 is one compact analytic-to-production ledger, not a backend matrix;
- exhaustive seams, loop caps, constructor support, reach cases, and hardware
  matrices remain in tests or verification ledgers;
- no visible cell claims a one-line `prange` speedup.

This is now a learner journey rather than a compressed verification suite. The
stale phrase “third or fourth visible cell” at line 327 should nevertheless be
made “third visible cell” so the prose agrees with the frozen inventory.

## 3. Requirements that now pass

| gate | result | evidence |
| --- | --- | --- |
| Central causal flow | **PASS** | One unexplained narrow opacity excess leads to bound levels, strength, broadening, a normalized profile, one depth, all depths, and only then production convergence. |
| Senior-undergraduate physical altitude | **PASS with the terminology correction in Section 4.4** | Bound-bound, statistical weight, oscillator strength, microturbulence, damping causes, and convolution are introduced in physical language before implementation detail. |
| Continuum/line distinction | **PASS with one caption wording correction** | The prose distinguishes continuum extinction from line absorption, says equal units do not equate the processes, and never calls opacity a flux dip. |
| Conceptual versus sampled normalization | **PASS** | \(\int H\,du=\sqrt{\pi}\), \(\int\phi_\nu\,d\nu=1\), the signed frequency/wavelength relation, and the no-Jacobian sampled-\(\kappa_\nu\) convention are explicit. |
| No detached exercises | **PASS** | The outline explicitly forbids an exercise section; perturbations and limits occur beside the claims they test. |
| Bite-sized code | **PASS** | Fifteen cells have one purpose each, normally 10–30 lines; long kernels remain canonical package calls rather than Markdown dumps. |
| Original schematics | **PASS** | Exactly four owned compositions are specified: opening extinction/line tension, two levels, convolution, and one immutable record through layers. They reuse only the website aesthetic and generation pattern. |
| Quantitative plots | **PASS** | Exactly four one-claim figures are specified: Doppler, damping, one-depth line over continuum, and all-depth heatmap. Harris remains a table; there is no decorative fifth plot. |
| Professional plot contract | **PASS** | One panel, physical axes/units, named palette, serif math typography, inward ticks, restrained guides, numerical interpretation, and render inspection are required. |
| Chapter 5 → 6 ownership | **PASS** | Chapter 6 consumes the supplied continuum, state, normalized population, fractional Doppler support, collision proxy, and stimulated factor without rederiving Chapters 3–5. |
| Chapter 6 → 7 ownership | **PASS** | Chapter 6 owns one ordinary line through cutoff/deposit primitives; catalog decoding, selection, forests, races, reduction, routing, and special profiles remain Chapter 7. |
| Minimal backward/forward references | **PASS** | Reader-facing prose mostly says “supplied”; the only substantive forward dependency is the atomic forest at the close. The detailed neighbor audit is an authoring contract, not reader narrative. |
| Source-tour/API risk | **PASS** | Fixed-width parsing, hashes, source locations, support-key plumbing, loop matrices, and optional routes are hidden or ledger-only. Exact calls appear late as production boundaries, after the physical object exists. |
| Limitations | **PASS** | The opening declares 1D, static, plane-parallel, LTE, isolated ordinary type-0 Fe I, no blends/magnetism/NLTE/special profile, and opacity rather than intensity or flux. Local numerical limitations appear where needed. |
| Summary and next link | **PASS** | The seven-statement summary is earned, keeps atmosphere and synthesis outputs separate, makes no cross-lane equality or flux claim, and links directly to `/reader.html?ch=7`. |
| Payne Zero restraint | **PASS** | Exact implementation names appear where they carry interface or parity information; the project name is not paragraph-level branding and no legacy KGPU framing remains. |

## 4. Actionable findings that still block immediate authoring

### 4.1 P1 — Cell 1 consumes cm\(^{-1}\) term values before their coordinate is explained

Lines 153–182 begin correctly with conceptual energies
\(h\nu_l=E_u-E_l\), but the executable cell then uses two source columns whose
stored values are spectroscopic term values in cm\(^{-1}\). The outline calls
them merely “the two energy columns” and does not give the executable conversion
before asking the reader to reproduce the wavelength.

Later, Section 6.2 correctly distinguishes
\(\tilde E_l=E_l/(hc)\) from \(E_l\). The current notation reveal map and
schematic 3 now preserve that distinction correctly. The remaining defect is
strictly chronological: the first executable wavelength cell consumes the
source term values before that coordinate or its conversion has been explained.

Required repair:

1. In Section 6.1, keep \(E_l,E_u\) for conceptual energies.
2. Before cell 1, state that the source columns used by the code are term
   wavenumbers
   \(\tilde E_l=E_l/(hc)\) and \(\tilde E_u=E_u/(hc)\), both in
   cm\(^{-1}\).
3. Give the actual executable bridge
   \[
   \lambda_l[{\rm nm}]
   =\frac{10^7}{|\tilde E_u-\tilde E_l|[{\rm cm}^{-1}]}.
   \]
4. Keep the already-correct reveal-map entry and Boltzmann cancellation
   \(\tilde E_l(hc/kT)=E_l/kT\).

This is the main rejection reason because the first code cell otherwise violates
the book rule that a unit and notation be taught before code consumes it.

### 4.2 P1 — The opening read contract does not yet provide all units, axes, dtypes, and devices before the early cells consume them

The supplied-input table at lines 116–125 is useful, but its “exact name /
axes” column omits the axes for `fractional_doppler_widths`, even though cell 4
uses it. The table also has no unit or dtype/device column. The first explicit
CPU NumPy `float64` teaching-state declaration does not appear until Section
6.12, after most of the physical cells.

Required repair:

- give `fractional_doppler_widths` its teaching-state axes `(D,6,139)` before
  cell 4;
- state the early teaching-state dtype/device once, or add a compact
  dtype/device column;
- attach units before consumption:
  normalized populations and perturber densities in cm\(^{-3}\), mass density
  in g cm\(^{-3}\), \(hc/kT\) in cm, fractional Doppler width and stimulated
  factor dimensionless, and continuum opacity in cm\(^2\) g\(^{-1}\);
- identify the stimulated factor's applicable depth/wavelength axes rather than
  calling it only a “construction”;
- retain the packed atmosphere counterpart as an opaque, late production
  fixture rather than expanding the opening into a second storage lesson.

The same cleanup should change “third or fourth visible cell” at line 327 to
“third visible cell.” The count is coherent; this is a contract-alignment fix.

### 4.3 P1 — Immediate interpretation is guaranteed for plots, but not yet for every table- or parity-producing cell

The outline is exemplary about interpreting cells 1, 2, 4, 5, 7, 11, 13, and
15. Its global plot rule also requires the paragraph after every plot to quote
actual values.

The same explicit next-paragraph contract is not yet present for every
non-plot output. Cells 3, 8, 9, 10, 12, and 14 specify useful ratios, tables, or
parity rows, but several sections stop at what to print rather than what the
actual printed values allow the reader to conclude.

Required repair:

- cell 3: read the measured 2, 2, and 1/2 ratios as linear \(gf\)/population
  scaling and inverse density scaling;
- cell 8: state which representative values are unchanged, quantized, or
  lane-dependent at the domain boundary;
- cell 9: interpret the measured continuous-versus-Harris difference without
  promoting it into a new physical effect;
- cell 10: interpret the actual center, near-wing, far-wing, reach, and float32
  values as deposition arithmetic;
- cell 12: state what its parity row establishes and what pre-stimulated output
  still cannot be compared directly with;
- cell 14: read the measured float32 and gross/net differences, including when
  an unavailable backend is not evidence.

One short authoring lock in each relevant section is enough. No new cell or
figure is needed.

### 4.4 P1 — Three phrases still create avoidable physical or terminology ambiguity

These are local edits, but they matter at the declared prerequisite floor.

1. The schematic ordinate “mass extinction / absorption (conceptual)” at line
   75 uses a slash after the prose has carefully separated the processes.
   Label the smooth curve “continuum mass extinction” and the added curve “line
   mass absorption,” both in cm\(^2\) g\(^{-1}\).
2. Lines 529 and 866 call the analytic reference “continuous/full Voigt” or
   “direct exponential/full profile,” while Section 6.11 later says “full
   Harris evaluator.” Use “continuous mathematical Voigt reference” for cell 7
   so “full” cannot collapse analytic physics and a production evaluator.
3. Define LTE at its first opening use in one plain sentence, before saying
   “NLTE,” and gloss “loss of phase coherence” as loss of a well-defined
   oscillation phase. If “ordinary type-0” remains in the opening, explain that
   “ordinary” means the standard isolated Voigt/Harris route; otherwise delay
   the type number to the production boundary.

## 5. Publication-gated output note

The outline correctly says the atmosphere and synthesis output dimensions
remain provisional until their separate artifacts pass publication. That
honesty is not a rejection finding. It is a later freeze condition:

- publish the atmosphere axes before naming its NumPy `float32`,
  gross/pre-stimulated slab as final;
- publish the synthesis axes/device before naming its Torch `float32`
  gross/net slabs as final;
- preserve lane-specific oracle claims and never replace them with cross-lane
  slab equality.

## 6. Final disposition

**REJECT for immediate prose authoring at outline hash
`698b3846d6e096037ae7c331b9a4dd1dff38edb3ef5332960b10c649be4ec2e0`.**

The former rejection's three structural P0s are genuinely resolved. The
chapter now has the correct physical-first sequence, honest \(n_l\) semantics,
and a credible fifteen-cell build. It also passes the exercise, visual,
neighbor-ownership, reference, source-tour, summary, and next-link gates.

Acceptance now requires only:

1. placing the term-wavenumber-to-wavelength bridge before cell 1;
2. completing the early read/shape/unit/dtype/device contract;
3. requiring actual-value interpretation after every remaining output cell;
4. removing the three local opacity/terminology ambiguities above.

After those edits, a re-hash plus a short targeted verification of Sections
6.1–6.2, the supplied-input table, cells 3/8/9/10/12/14, and the opening labels
should be sufficient. No second causal redesign is warranted.

## Targeted correction verification

Verified: 2026-07-30  
Scope: Sections 6.1–6.2, the supplied-input table, cells
3/8/9/10/12/14, opening labels and definitions, and correction-induced
consistency only  
Latest outline audited:
`design/chapter06_causal_outline.md`  
Latest outline SHA-256:
`099b33f1a8ca7836548fdde3007c29ff300a2f5713511916220651e52d3c9d2f`  
Audit SHA-256 before this appended verification:
`862f4ddab4b121cf1da32883ef0471d62b545973e704a11c813fa9ed86cc48d2`

This section supersedes the earlier disposition only for the latest outline
hash above. The historical findings remain in place so the correction path is
auditable.

### Targeted results

| prior finding | latest result | exact verification |
| --- | --- | --- |
| 4.1 term-wavenumber bridge before cell 1 | **CLOSED** | Lines 176–188 now distinguish conceptual energies from stored term wavenumbers, state cm\(^{-1}\), give \(\lambda_l[{\rm nm}]=10^7/|\tilde E_u-\tilde E_l|\), explain the conversion factor, and place all of this before the source values and first cell. Section 6.2 retains the correct \(\tilde E_l(hc/kT)=E_l/kT\) cancellation. |
| 4.2 early axes/unit/dtype/device contract | **CLOSED** | Lines 122–138 now give axes, units, and CPU NumPy `float64` ownership for every early teaching input; `fractional_doppler_widths` is `(D,6,139)`, the stimulated factor is dimensionless `(D,W)`, \(D/W\) and depth direction are defined, and packed atmosphere state remains opaque until the late production cell. Line 351 now says “third visible cell.” |
| 4.3 actual-value interpretation for cells 3/8/9/10/12/14 | **CLOSED** | Lines 351–356 interpret the three measured strength ratios; 595–599 interpret FASTEX ordinary/half-step/domain values; 613–618 identify Harris residuals as approximation error; 629–638 interpret shortcut/reach/float32 values; 720–723 state what atmosphere parity does and does not establish; and 788–796 require the measured float32, one-factor stimulation, and unavailable-backend reading. |
| 4.4 opening label and terminology corrections | **PARTIALLY CLOSED** | Lines 75–90 now separate continuum mass extinction from line mass absorption, define LTE before NLTE, define “ordinary,” delay the type number, and retain the opacity-not-flux boundary. Lines 455–456 gloss phase coherence. Section 6.8 and visible cell 7 now say “continuous mathematical Voigt reference.” One occurrence remains, described below. |

### Remaining acceptance blocker

The exact notation reveal map at line 886 still names the cell-7 object:

```text
local direct-exponential/full-profile result
```

This is the same ambiguity that the correction successfully removed from
Section 6.8 and the visible-cell inventory. “Full profile” can still be read as
the later full Harris evaluator rather than the analytic continuous reference.
Replace only that phrase with:

```text
local direct-exponential/continuous-Voigt reference; no Jacobian
```

No equation, cell, figure, section order, or chapter ownership needs to change.

### Targeted consistency result

No new scientific, unit, axis, dtype/device, neighbor-ownership, cell-count,
exercise, source-tour, or opening-definition inconsistency was introduced in
the inspected regions. The correction also preserves:

- conceptual \(E_l\) versus stored \(\tilde E_l\);
- conceptual \(n_l\) versus executable \(n_l f_{lu}\);
- the transparent one-depth line before FASTEX and Harris;
- the fifteen-cell sequence;
- separate atmosphere and synthesis stimulation/parity lifecycles.

### Latest disposition

**REJECT for authoring at outline SHA-256
`099b33f1a8ca7836548fdde3007c29ff300a2f5713511916220651e52d3c9d2f`,
solely because the notation-map wording residual above means not every recorded
P1 is closed.**

After that one phrase is corrected, a final hash check and search for
`full-profile`/`continuous/full` is sufficient for acceptance; another
pedagogical or neighbor audit is not required.

### Final hash/search closure

Verified: 2026-07-30  
Final outline SHA-256:
`1b66df5d548f2854f83289fcf9de5109058f1482a7b64aadaff3505d1f57e019`

The final notation-map entry is:

```text
local direct-exponential/continuous-Voigt reference; no Jacobian
```

Searches of the final outline for `full-profile` and `continuous/full` return
no matches. No other outline content was re-audited or changed in this closure.

**ACCEPT for prose authoring at outline SHA-256
`1b66df5d548f2854f83289fcf9de5109058f1482a7b64aadaff3505d1f57e019`.**
