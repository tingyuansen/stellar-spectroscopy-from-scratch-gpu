# Chapter 6 pedagogy and neighbor audit

Status: fresh pre-authoring audit  
Audited: 2026-07-30  
Disposition: **REJECT for authoring readiness until the three P0 findings
below are resolved**

No chapter, outline, runtime, test, data, source, paper, or external Payne Zero
file was changed by this audit. This report is the only file added.

## 1. Scope and review snapshot

This audit asks whether `design/chapter06_causal_outline.md` can be turned
directly into a self-contained chapter for a final-year undergraduate or
first-year graduate student who knows basic mathematics, physics, and Python,
but not spectroscopy, statistical mechanics, radiative transfer, NumPy/Numba
internals, Torch, or GPU programming.

The governing snapshot was:

| authority | SHA-256 |
| --- | --- |
| `design/chapter06_causal_outline.md` | `e1ff327bcf6c2588708e7291b5af395b8cc4e23d665773cf75829b49068b65a3` |
| `BIBLE.md` | `1433c2d3d18dd7397f8a739765c7f8e4c36f4b79e41c2809f596fa7fe3bf59b0` |
| `PLAN.md` | `bf2da67593c8b23c02c82846ef2d55388cce2383c57860719cb5601209342539` |
| `design/pedagogical_flow_rubric.md` | `4626b53dd9953df120398fd8762504b9cb1a429b8b0e6df4a1e5a049a433f35d` |
| `design/global_chapter_contracts.md` | `51ed5e93cdd1787ec355155573bfa20bbc7320d376fc3afa16ff3485438b370c` |
| accepted `book/chapters/chapter_05.py` | `68ec2c7f9afc078d0b88c31a886516edcf10214978a790f687019104342ed96e` |
| `design/chapter05_final_acceptance.md` | `d94b81b561ce97d52a0f1d129da8d2acb91a8f89841bf3bc235e63b66f266d08` |
| `design/part3_part4_synthesis_brief.md` | `9906ae384aa19e49abe714e2728b9a6156192da1c9bdced3cb47708d230d14b4` |
| accepted `book/chapter06_runtime.py` | `c011c542b72539c94537bd71def1ea0b63f1b69c1014107edd0c23821083a0fe` |
| `design/chapter06_runtime_core_audit.md` | `83366bed79293bb8d02ccdf67b54d0350dcc638888e6678d9dedc7b1d58f3313` |

The accepted Chapter 5 ending, its final acceptance report, the complete
Chapter 6 and Chapter 7 contracts, and the accepted Chapter 6 runtime evidence
were read as one continuous handoff. The projected atmosphere fixture/oracle
and synthesis comparison artifact were still separate in-progress gates at
the audited `PLAN.md` snapshot. Nothing in this report treats those products
as published goldens.

## 2. Executive verdict

The outline has an excellent physical spine:

```text
energy separation
→ excitation-weighted transition strength
→ Doppler core
→ damping wings
→ normalized Voigt profile
→ one depth
→ all depths
→ exact atmosphere and synthesis products
```

Its opening question, one-line scope, ordinary-versus-special boundary,
continuous-profile-versus-deposit distinction, lane-specific stimulation
lifecycle, visual discipline, absence of detached exercises, and causal link
to Chapter 7 are all strong.

It is nevertheless **not ready to author verbatim**.

1. FASTEX interrupts the physics before integrated opacity, broadening, or the
   exact mathematical profile has been built. This contradicts both the
   pedagogical rubric and the detailed Chapter 6 contract, which require exact
   physics before production approximation.
2. The destination, cell 2 question, and frozen summary promise an actual
   lower-level absorber count. The accepted runtime deliberately removed that
   false semantic claim: the record supplies \(gf\), not \(g_l\) separately,
   so the executable result is an excitation-weighted
   partition-normalized population and then a \(gf\)-weighted factor, not
   \(n_l\).
3. The fourteen-cell plan fits only by turning several cells into compact test
   suites. Harris authorities plus shortcut, six unrelated route comparisons,
   constructor support keys, cutoff decisions, backend policy, and parity
   cannot all remain reader-visible while preserving one conceptual purpose
   per cell.

These are outline-order and presentation-boundary defects, not missing
science. All accepted runtime evidence can fit in **fifteen visible cells**
provided exhaustive seams, provenance ledgers, constructor plumbing, and
backend matrices remain in tests or compact verification records rather than
becoming the main lesson.

## 3. What should be preserved

The following decisions already meet the governing standard.

1. The chapter opens with a narrow opacity excess that the accepted Chapter 5
   continuum cannot explain. It explicitly says this is not flux.
2. Two bound levels appear before oscillator strength or profile machinery.
3. \(gf\) is identified honestly as the catalog convention, and the outline
   explicitly prevents a second statistical-weight factor.
4. Integrated strength is separated from profile shape.
5. Thermal/microturbulent broadening and three damping causes enter before the
   Voigt profile.
6. The normalized continuous profile is distinguished from values of
   \(\kappa_\nu\) sampled on a wavelength grid; no wavelength Jacobian is
   invented.
7. The atmosphere and synthesis lanes remain distinct in Harris authority,
   FASTEX domain behavior, stimulation timing, dtype, device, grid, cutoff,
   and oracle.
8. One ordinary Fe I transition is the only physical line. No hydrogen,
   helium, autoionizing, molecular, or many-line derivation enters the chapter.
9. Chapter 7 starts from the genuinely new problem of decoding, selecting,
   routing, and safely accumulating many heterogeneous records.
10. There is no exercise section. Useful perturbations are placed beside the
    claims they test.
11. The three declared schematics have genuine explanatory jobs and use the
    website only as an aesthetic/process reference.
12. The quantitative visual rules are professional: one panel, one claim,
    physical units, named colors, and immediate numerical interpretation.
13. The next-chapter heading, causal paragraph, and `/reader.html?ch=7` link
    are already strong.
14. Payne Zero naming is restrained. The proposed reader voice mostly says
    “atmosphere product,” “synthesis product,” or the physical object rather
    than repeating the project name as branding.

## 4. P0 findings — resolve before prose authoring

### P0.1 — FASTEX arrives before the reader has a line for it to accelerate

The outline introduces FASTEX in Section 6.3, immediately after the
Boltzmann-factor ingredients and before:

- the integrated-opacity equation;
- stimulated emission in the line context;
- Doppler broadening;
- damping;
- the normalized Voigt profile;
- one transparent line-opacity calculation.

The question “which approximation is the exact production rule?” is therefore
asked before the exact physical calculation exists in the reader's mind.
“Millions of lines” also imports Chapter 7's scale motivation too early. The
result would be a table-lookup lesson interrupting the causal movement from
absorber to line strength.

This conflicts with:

- the rubric's “production complexity after the transparent core” rule;
- the Chapter 6 brief's order “Voigt profile, then Harris and FASTEX”;
- the accepted runtime's explicit distinction between the analytic
  direct-exponential/full-profile construction and exact production
  FASTEX/ordinary-metal shortcuts.

Required repair:

```text
energy difference
→ conceptual lower-level population and exact gf-weighted factor
→ integrated opacity and one reused stimulation factor
→ Doppler and damping
→ normalized continuous Voigt profile
→ one direct-exponential/full-profile line at one depth
→ FASTEX and Harris as production approximations
→ exact deposited all-depth products
```

The direct calculation should make a prediction and produce a trustworthy
one-depth line before either lookup table is opened. FASTEX can then answer a
question the reader has earned: how does the production route replace the
repeated direct exponential without silently changing cutoff decisions?

### P0.2 — The outline promises a lower-level population that the accepted
runtime correctly does not compute

The outline currently promises “the number of absorbers available in one
lower bound level,” asks cell 2 “How many absorbers reach the lower level?”,
and says in the frozen summary that the “lower-level absorber count” comes
from the normalized ion population, Boltzmann factor, and \(gf\).

Those statements conflate two different quantities.

Conceptually,

\[
n_l=\frac{n_{s,r}}{U_{s,r}}g_l e^{-E_l/kT}.
\]

The selected source record, however, supplies \(gf=g_lf_{lu}\), not \(g_l\)
and \(f_{lu}\) separately. Its executable production factor is

\[
\frac{n_{s,r}}{U_{s,r}}(gf)e^{-E_l/kT}
  =n_lf_{lu}.
\]

The accepted runtime therefore exposes:

- `excitation_weighted_partition_normalized_population_cm3`;
- `gf_weighted_excitation_factor_cm3`;

and intentionally removed the false field
`lower_level_population_cm3`.

Required repair:

- keep \(n_l\) as the conceptual equation that explains where statistical
  weight would enter;
- immediately say that this record does not let the code isolate \(n_l\);
- change the destination to “the excitation-weighted population factor needed
  by one transition”;
- change cell 2's question to “Which fraction of the ion-stage population is
  excitation-weighted, and where has \(g_l\) gone in the source convention?”;
- print the accepted runtime field names without calling either one an actual
  lower-level population;
- rewrite summary statements 2–3 so \(gf\)-weighted strength, not an
  independently computed absorber count, is the executable output.

This repair is mandatory for notation continuity with the accepted runtime
and for the book-wide distinction among ion-stage population,
partition-normalized population, bound-level population, and line-strength
factor.

### P0.3 — The nominal fourteen-cell plan is a compressed verification suite,
not yet a bite-sized learner journey

The outline declares fourteen cells, but three checkpoints violate its own
one-purpose rule:

- checkpoint 8 combines two table authorities, multiple Harris seams, and the
  separate low-damping synthesis shortcut;
- checkpoint 12 combines state indices, support metadata, two cutoff stages,
  dtype/device ownership, output contract, and golden parity;
- checkpoint 13 combines a float32 cast, stimulation identity, loop/batched
  equality, CPU/CUDA comparison, CPU/MPS comparison, and the atmosphere
  downstream-stimulation identity.

The atmosphere section additionally asks the reader to hold asymmetric
red/blue loop bounds, first-below-threshold retention, an 8-layer gate,
80-depth fixture provenance, `njit` route identity, dtype, and two parity
views in one visible cell. The synthesis section exposes five constructor
support keys, including empty helium metadata, before Chapter 7 has taught the
full catalog and routing structure.

Required repair:

1. Freeze a **reader-visible evidence budget** separately from the exhaustive
   verification budget.
2. Use fifteen visible cells, as specified in Section 6 below.
3. Keep exhaustive one-sided FASTEX/Harris seams, all constructor support
   keys, hashes, halfword/packing details, every reach boundary, and full
   backend matrices in tests and generated ledgers.
4. Show only representative values that change the learner's current mental
   model.
5. Let each production-lane cell answer one question:
   “Does this exact route preserve the line we just built?”
6. Report strict parity in one compact row after the local calculation;
   never make the golden an input or a tutorial object.

All accepted runtime/evidence requirements remain covered under this repair;
they simply do not all become printed notebook output.

## 5. P1 findings — correct in the revised outline

### P1.1 — The opening conflates line absorption with continuum extinction

The opening asks for an ordinate in “mass absorption” while plotting
“absorption-plus-scattering.” The accepted Chapter 5 close calls
`continuum_absorption + continuum_scattering` a derived **extinction** scale.
The line output, by contrast, is a mass absorption coefficient.

Required repair:

- label the smooth curve “continuum extinction
  \(\kappa_\nu^{\rm abs}+\kappa_\nu^{\rm sca}\)
  [cm\(^{2}\) g\(^{-1}\)]” or the exact already-established
  `continuum_opacity`;
- label the narrow added quantity “line mass absorption”;
- state at the cutoff comparison that equal units make the comparison useful,
  but do not make scattering an absorption process;
- do not call the sum a third independently calculated opacity process.

This is a neighbor-handoff correction, not a request to re-teach Chapter 5.

### P1.2 — Spectroscopic wavenumber notation and several physical bridges are
still implicit

The outline writes \(E_l/kT\), then code consumes
`lower_excitation_cm * hc_over_kt`. For the selected record,
`lower_excitation_cm` is a spectroscopic wavenumber in cm\(^{-1}\), while
`hc_over_kt` has units cm. A novice should not have to infer why their product
is dimensionless.

Required additions, each kept to one compact causal bridge:

- distinguish energy \(E_l\) from the stored wavenumber
  \(\tilde E_l=E_l/(hc)\), or explicitly declare that the chapter uses
  \(E_l\) in cm\(^{-1}\) from that point onward;
- show
  \(\tilde E_l[{\rm cm}^{-1}]\,(hc/kT)[{\rm cm}]=E_l/kT\);
- before the Gaussian, state the small-velocity Doppler relation
  \(\Delta\nu/\nu\simeq v_{\parallel}/c\) and explain that one component of a
  thermal velocity distribution is Gaussian;
- before adding damping rates, name the raw units:
  \(\gamma_{\rm rad}\) in s\(^{-1}\) and the Stark/van der Waals coefficients
  in cm\(^3\) s\(^{-1}\);
- explain that independent broadening rates add before division by
  \(4\pi\Delta\nu_D\);
- state
  \(\int H(a,u)\,du=\sqrt{\pi}\) and
  \(\int\phi_\nu\,d\nu=1\) beside the normalization check;
- state that positive frequency offset is negative wavelength offset locally,
  or use \(|u|\) wherever the production helper actually consumes an absolute
  offset.

These sentences supply the “concrete limit → math → implementation” bridge
without becoming a statistical-mechanics detour.

### P1.3 — Source provenance appears too early in the physical story

Section 6.1 currently interrupts the two-level picture with raw row `873702`,
four activity counts, the hot-dwarf cutoff result, unfinished packing/oracle
conditions, and golden terminology before the first wavelength cell.

The row identity is important data honesty, but its activity result answers a
question that belongs near the final four-regime evidence. It also reveals
production cutoff behavior before line strength has been defined.

Required repair:

- introduce the row as one checksum-bound ordinary Fe I teaching record in a
  short cell contract/caption;
- keep its source-row ID and static-input role visible;
- move `3/6, 6/6, 6/6, 6/6` to the final regime cell, where the reader can
  explain it physically;
- keep packing conversion, candidate/frozen status, builder hashes, and oracle
  authorization in the data note/manifest and verification ledger;
- do not mention “golden” before the reader-built result exists.

### P1.4 — The production-lane sections risk becoming API documentation

Exact function and field names are properly delayed until Movement III, but
the planned level of constructor and loop detail is too high for the main
causal sequence.

Reader-visible production convergence should show:

- the exact entry call;
- physical inputs and output;
- axes, unit, dtype, and device;
- gross/net stimulation ownership;
- one representative cutoff/reach fact;
- one lane-specific parity result;
- for the atmosphere lane, the useful fact that one line uses the serial
  cached `njit` route and therefore makes no parallel-speed claim.

Keep outside the main narrative:

- fixed-width and halfword conversion;
- the five support-key constructor table, especially empty helium metadata;
- source line-number citations;
- all asymmetric loop bounds and every controlled maximum-reach case;
- full manifest/hash ledgers;
- the optional detailed-transition type-0 route;
- a generic Numba or GPU tutorial already owned by Chapter 2.

The progressive package remains available to a reader building the complete
system. The chapter should explain the exact public boundary without reading
like generated API reference.

### P1.5 — The Chapter 3–5 recap budget is larger than the outline claims

The main flow currently names prior chapters in the opening, bound-free
contrast, partition-sum paragraph, stimulated-emission paragraph, Doppler
paragraph, continuum comparison, atmosphere threshold, and final handoff.
Each reference is individually short, but together they create a
page-turning rhythm inconsistent with the “at most one short backward recap”
rule.

Required repair:

- retain one compact opening reads/writes contract;
- describe its entries thereafter as “supplied,” without chapter numbers;
- keep one sentence that reuses the already-derived stimulated factor;
- keep one sentence that the stored fractional Doppler width is an input, not
  rederived state;
- use the accepted Chapter 5 continuum directly without repeating its process
  list or grid construction;
- reserve the only explicit forward chapter reference for the close.

The opening contract should use the six-depth synthesis teaching state first.
The opaque 80-depth atmosphere integration fixture should appear only when
the atmosphere production lane is reached, not as a second packed-state
schema beside the opening physics.

### P1.6 — One-line cutoff ownership is valid, but several Chapter 7 details
must remain deferred

The Chapter 6/7 boundary is mostly correct. Chapter 6 does own the physical
one-line cutoff and deposit primitive that Chapter 7 will reuse. It does not
own:

- full-record decoding and corrections;
- context/window selection;
- category routing;
- sparse many-line center deposits;
- chunking, scatter-add, private buffers, races, or reduction order;
- helium metadata as a teaching topic;
- special profiles.

The Chapter 6 text may call a verified one-record catalog object at the exact
production boundary, but it must treat it as the already-derived
representation of the one teaching transition. It must not explain general
catalog construction, and its special-branch placeholders should remain
hidden setup. The current explicit deferrals are good and should be retained.

### P1.7 — Limitations should be declared at the opening and repeated only
where they matter

The ordinary-line assumptions do not appear until the integrated-opacity
section. Move a compact controlled-case declaration next to the opening:

- one-dimensional, static, plane-parallel supplied atmosphere;
- LTE populations and reverse process;
- one isolated ordinary type-0 Fe I transition;
- no blends, magnetic splitting, NLTE departure, or special profile;
- opacity only, not emergent intensity or flux.

Then attach local limitations at the exact point they matter:

- the wavelength Doppler conversion is a narrow-line approximation;
- analytic normalization applies to the continuous untruncated profile;
- a cutoff deposit does not preserve the analytic area exactly;
- atmosphere and synthesis slabs are not cross-lane parity targets;
- unavailable CUDA/MPS measurements are “unavailable,” not passing evidence.

This is more useful than collecting caveats at the end.

### P1.8 — The frozen summary collapses two exact outputs and repeats the
lower-population error

The seven summary statements answer the opening physical question well, and
the next link is causal. However:

- statement 2 repeats the false independently computed lower-level count;
- statement 7 collapses the atmosphere and synthesis products into one
  generic slab;
- the close does not yet state each lane's actual accepted axes, dtype,
  device, and gross/net stimulation state.

Required repair after the pending artifacts are accepted:

- name the atmosphere NumPy `float32` pre-stimulated output with its published
  axes;
- name the synthesis Torch `float32` gross/net output with its published axes
  and selected device;
- state that each matches its own oracle under its own policy;
- state explicitly that neither is emergent flux and that cross-lane slab
  equality is not claimed;
- preserve the current heading
  `### Next: From one trustworthy line to an atomic forest` and direct link.

Do not freeze placeholder dimensions before the in-progress artifacts have
passed their separate publication gates.

## 6. A feasible fifteen-cell learner journey

The following sequence covers every accepted runtime/evidence element without
turning the chapter into API documentation. “Strict evidence” means the full
matrix remains mandatory in tests/ledgers; the visible cell shows only the
representative result needed by the narrative.

| cell | learner question | visible evidence | strict evidence retained outside the cell |
| ---: | --- | --- | --- |
| 1 | Which photon matches the two selected bound levels? | derived versus stored wavelength, units, source-row identity | all raw fields, manifest and conversion hashes |
| 2 | What population factor is available, and where is \(g_l\)? | normalized population, dimensionless excitation exponent, direct exponential, accepted honest factor names | all depths/regimes and defensive-copy checks |
| 3 | What fixes the integrated area? | \(gf\), normalized-population, density, and stimulated-factor scaling ratios | full dimensional matrix and constants audit |
| 4 | How does supplied motion redistribute that area? | one Doppler-width perturbation, Gaussian profile, area check, one plot | Chapter 3 mass table and full temperature/mass/microturbulence suite |
| 5 | Which physical perturber changes which damping term? | three-term ledger plus one electron and one neutral perturbation | all-depth damping and exact stored-literal tests |
| 6 | Why is the profile a Voigt convolution? | three damping limits, \(H\) and \(\phi_\nu\) normalization, one plot | refined-domain normalization matrix |
| 7 | What does the exact transparent line look like at one depth? | direct exponential plus continuous/full profile over continuum, factor ledger, one plot | complete analytic checkpoint arrays |
| 8 | How does FASTEX replace the direct exponential? | one ordinary value, one half-step, one domain-boundary comparison | scalar/compiled/Torch, nonfinite, negative, 1001, f64/f32 seam matrix |
| 9 | How does Harris replace the continuous profile? | one representative \(a\), normalization/approximation error, compact table or residual | both lane authorities and every one-sided Harris seam |
| 10 | What additional shortcut and cutoff does the synthesis deposit use? | center/near/far values, one cutoff/reach boundary, float32 center effect | all 24 center reconstructions and maximum-reach cases |
| 11 | How does one immutable record change through depth? | dense `(D,W)` slab, axes, units, one heatmap | finite/nonnegative/all-depth state-causality suite |
| 12 | Does the exact atmosphere route preserve the one-line result? | one serial-`njit` call, output/stimulation contract, one parity row | 80-depth oracle, asymmetric loop, 8-layer, dtype, empty-path, and detailed-route tests |
| 13 | Does the exact synthesis route preserve it on the selected device? | exact call, indices/counts, output contract, one parity row | support-key schema, cutoff masks, loop/batched arrays, reach arrays |
| 14 | What changed when the transparent calculation became production arithmetic? | one compact analytic→production error and gross→net identity table | full float32, loop/batched, CPU/CUDA/MPS and lane-lifecycle matrix |
| 15 | Does the same construction behave honestly in four stellar regimes? | active depths, peak opacity, representative width/damping, exact status | all four complete slabs and lane-specific artifact comparisons |

This sequence remains below the sixteen-cell hard ceiling. It also restores
the required rhythm:

```text
physical need
→ intuition
→ exact mathematics
→ small executable result
→ interpreted output
→ production convergence
→ honest limitation
```

Cells 8–10 are three distinct numerical questions and must not be compressed
into one “Harris/FASTEX internals” cell. Cells 12–13 are two distinct products
and must not be collapsed into false cross-lane parity. Cell 14 is one
controlled approximation ledger; hardware and exhaustive branch status may
appear as concise rows, not as six new mini-lessons.

## 7. Visual and pacing audit

### 7.1 Schematics

The three declared schematics are justified:

1. two levels and one resonant photon;
2. two broadening causes combined by convolution;
3. one immutable record passing through changing atmospheric layers.

There is one inventory inconsistency: the opening also instructs the author to
“show” a smooth background with a conceptual narrow bump, but this object is
not listed in either the schematic or quantitative-plot plan. Resolve it
explicitly in one of two ways:

- add a fourth, very small conceptual opening schematic with its own prompt,
  provenance, caption guard, and alt text; or
- fold that comparison into schematic 1 without implying a measured line
  shape.

Do not render the conceptual bump with quantitative axes that make invented
numbers look like data. Four schematics remain within the book's normal
two-to-four range.

### 7.2 Quantitative figures

Use four quantitative figures:

1. Doppler redistribution;
2. damping redistribution;
3. one-depth line over continuum;
4. all-depth slab.

Make the Harris comparison a compact numerical table unless a residual plot
reveals a physically meaningful spatial pattern. This resolves the outline's
“optional fifth plot” ambiguity and prevents a tiny implementation residual
from receiving the same visual weight as the line physics.

The all-depth heatmap is justified only if it is more legible than six
profiles. Its depth axis must name `column_mass` with units when available,
otherwise “supplied depth index” with outermost/innermost annotation. A
logarithmic color normalization must be stated and must handle exact zeros
honestly.

Each plot plan already requires one claim, physical units, professional
styling, and an immediate paragraph quoting measured values. Preserve that
discipline. No plot should be a dashboard, a parity overlay with invisible
differences, or a decorative repetition of its schematic.

## 8. Neighbor ownership audit

| topic | owner | Chapter 6 disposition |
| --- | --- | --- |
| actual versus partition-normalized ion populations | Chapter 3 | consume one exact population slot; do not reopen Saha or partition sums |
| thermal/microturbulent support construction | Chapter 3 | show its new consequence for a profile; do not repeat packed/public storage or mass-table teaching |
| molecular equilibrium and H2 abundance | Chapter 4 | consume only `collision_density_proxy`; no molecule solve or line record |
| \(n\sigma/\rho\), continuum components, scattering, and stimulated-factor derivation | Chapter 5 | reuse mass-opacity meaning, `continuum_opacity`, and one factor; do not repeat processes, grid construction, or limiting plot |
| one ordinary line's strength, width, damping, Voigt/Harris/FASTEX behavior, cutoff, and deposit | Chapter 6 | derive once and close both one-line products |
| catalog decode/correction/window selection, many-line deposits, races, routing, and special atomic profiles | Chapter 7 | defer completely except for the causal close |
| molecular source formats and molecular line deposition | Chapter 8 | absent |
| transfer and emergent flux | later transfer chapter | state “not yet a spectrum”; do not infer a line depth in flux |

The accepted Chapter 5 close already hands forward exactly the missing
bound-bound problem. Chapter 6 should begin from that tension without
re-explaining why the continuum exists. The planned Chapter 7 opening already
has its own genuinely new stakes—an impossible dense `(D,L,W)` allocation and
heterogeneous routing—so Chapter 6 should not pre-teach catalog scale,
scatter-add, races, or special branch metadata.

## 9. Explicit risk register

### Terseness and hidden-prerequisite risk

- stored wavenumber versus energy is not yet explained;
- a Gaussian thermal line-of-sight distribution is asserted rather than
  bridged from the Doppler relation;
- damping-rate units and why rates add are implicit;
- continuous \(H\) versus normalized \(\phi_\nu\) needs one explicit area
  identity;
- the lower-level-population wording is currently wrong, not merely terse.

### Large-source-dump and API-documentation risk

- constructor support keys and empty helium metadata;
- atmosphere loop bounds and 8-layer gate;
- source line-number citations;
- manifest/oracle language before a reader-built result;
- six independent comparisons in the planned final technical cell.

These belong in progressive source, tests, generated ledgers, or concise
post-physics route notes.

### Redundancy risk

- a multi-temperature/mass/microturbulence lesson can repeat Chapter 3;
- a stimulated-factor limit plot would repeat Chapter 5;
- packed population or fixed-width conversion would repeat or pre-empt later
  schema/catalog chapters;
- general catalog scale motivation before the exact line is complete
  pre-empts Chapter 7.

### Notation and semantics risk

- “mass absorption” versus continuum extinction;
- \(E_l\) in energy units versus cm\(^{-1}\);
- conceptual \(n_l\) versus the executable \(n_lf_{lu}\);
- signed \(u\) versus an absolute-offset production helper;
- gross atmosphere versus net synthesis line slabs.

### Forward/back-reference burden

The current design names earlier chapters more often than its own declared
budget. A single opening input contract plus local “supplied” language is
enough. The final Chapter 7 link is appropriate and should remain the only
substantive forward reference.

### Over-branding and legacy framing

No material problem is present. Preserve the current restraint. The reader
chapter should mention Payne Zero only when identifying the pinned
implementation/parity destination, not as the subject of every paragraph.
No legacy KGPU framing is present.

### Detached-exercise residue

No exercise, homework, challenge, or postponed “try it yourself” structure is
planned. The perturbations are properly embedded in the main text. **PASS.**

## 10. Authoring acceptance gate

Chapter 6 becomes ready for prose authoring when all of the following are
true.

- FASTEX and Harris follow the transparent direct-exponential/continuous-profile
  line rather than interrupting the physical derivation.
- The destination, cell 2, notation map, and summary no longer claim that the
  selected record computes \(n_l\) separately.
- A frozen fifteen-cell reader-visible plan separates teaching evidence from
  exhaustive verification evidence.
- The opening distinguishes continuum extinction from line absorption.
- Stored cm\(^{-1}\) excitation, Doppler connection, damping units, profile
  normalization, and offset convention are explained at undergraduate
  altitude.
- Record activity counts appear with the four-regime result, not before the
  line-strength derivation.
- Atmosphere and synthesis production sections retain exact calls and output
  contracts without exposing irrelevant constructor plumbing.
- The opening states the controlled ordinary-LTE limitations.
- The summary names the two accepted lane outputs separately after their
  artifacts are actually published.
- The implicit opening visual is either owned as a fourth schematic or removed.
- Harris uses a table unless a residual plot passes a genuine one-claim visual
  test.
- Chapter 3–5 references are reduced to one compact handoff and local supplied
  inputs.
- Catalog decoding, special profiles, sparse many-line accumulation, and race
  reasoning remain in Chapter 7.

## 11. Final disposition

**REJECT for immediate authoring from the present outline.**

This is a narrow pedagogical rejection, not a scientific or runtime
rejection. The ordinary-line physics, exact numerical boundaries, accepted
runtime, and Chapter 5→6→7 ownership are strong. Resolving P0.1–P0.3 and
applying the P1 edits will produce a coherent fifteen-cell chapter in which a
reader first understands why a line has an area and shape, then watches that
transparent construction converge to the two exact production lanes.
