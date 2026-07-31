# Chapter 5 independent pedagogy audit

Status: bounded pre-draft audit; no chapter, runtime, source, or data files were
changed  
Audited on: 2026-07-30  
Primary contract:
`design/chapter05_exact_source_contract.md`  
Pinned implementation:
`9c44001feae40b85146630499e6f8a5fed42e5af`

## Audit question

Can the present Chapter 5 plan become a self-contained, exact, and readable
chapter for a final-year undergraduate or first-year graduate student without
turning the complete continuum implementation into a source tour?

The answer is **not yet**. The scientific inventory is unusually strong, and
the four-lane boundary is now exact, but the authoring documents still encode
two incompatible opening orders, several cells with more than one conceptual
job, and more planned figures than the declared six-plot limit. If those
conflicts are not resolved before prose is written, the likely result is a
technically complete but terse process catalogue.

This audit treats these as binding:

- the reader has basic mathematics, physics, and ordinary Python, but no
  assumed spectroscopy, statistical mechanics, Numba, Torch, or GPU knowledge;
- Chapter 3 owns actual versus partition-normalized populations;
- Chapter 4 owns molecular equilibrium and closes with two distinct consumer
  handoffs;
- Chapter 5 owns continuum physics, its two product grids, and their exact
  implementations;
- Chapter 6 owns the first bound-bound line;
- there are no exercises or postponed "try it yourself" sections;
- useful variations are predictions, compact calculations, and immediate
  interpretations inside the main text;
- exact implementation names arrive only when a physically understood object
  reaches its implementation boundary.

## What is already sound

The current plan should preserve the following decisions.

1. One physical spine,
   \(\kappa_\nu=n_{\rm absorber}\sigma_\nu/\rho\), organizes the chapter.
2. Atmosphere and synthesis are distinct exact consumers, not two backends of
   one invented continuum algorithm.
3. The atmosphere line-reference calculation is a subroute, not a fifth lane.
4. CH, OH, and H2 collision-induced absorption are atmosphere-only product
   terms; they are not added to standard synthesis for apparent symmetry.
5. The stored `molecular_hydrogen_population` is not a continuum input in
   either product lane.
6. Absorption, scattering, and the atmosphere thermal-source construction
   remain separate.
7. Long exact kernels live in the progressive package. Markdown does not paste
   source files, and visible code calls small checkpoints or canonical
   functions.
8. The proposed schematics are original textbook compositions with the
   website aesthetic used only as a style reference.
9. The summary already returns to the opening physical question and makes one
   bound-bound transition the causal next step.
10. No detached exercise section is planned.

## Prioritized findings

### P0 — Resolve before Chapter 5 prose

#### P0.1 — The opening code order has two incompatible authorities

`design/chapter05_exact_source_contract.md` currently puts a
manifest/table preflight in visible cell 1. The causal outline instead puts
the dimensional \(n\sigma/\rho\) experiment first and delays table identity
until the implementation boundary. Only the latter satisfies the book's
"experience before abstraction" and "physical need before production name"
rules.

A manifest cell must still occur before the first table-driven opacity result,
but it must not be the chapter's first intellectual event. The binding order
should be:

```text
opening observation
→ microscopic-to-mass-opacity calculation
→ net-absorption idea
→ just-in-time static-table preflight
→ first table-driven absorber
```

This is not permission to use undeclared data. The preflight moves only from
cell 1 to the first moment a table is needed.

#### P0.2 — The 27-field-to-18-field synthesis transition is a hidden interface

Chapter 4 closes by displaying the complete 27-field schema-v4 handoff.
Chapter 5's exact standard synthesis route consumes an 18-field trimmed
pipeline view. The causal outline names both facts but does not yet prescribe
the one sentence and one executable projection check that reconcile them.

Without that bridge, a careful reader can reasonably infer one of two false
claims: either Chapter 4's 27-field contract was wrong, or Chapter 5 has
invented a second synthesis state. The chapter must state explicitly:

- the 27-field schema-v4 mapping remains the upstream synthesis atmosphere;
- the standard continuum call receives the exact 18 fields that its pipeline
  projects from that mapping;
- this trimmed synthesis view is not the packed 18-field
  `ContinuumAtmosphereState`;
- neither 18-field consumer view is reconstructed from the other.

The visible check should report the two exact consumer projections and their
field shapes. It should not repeat Chapter 4's complete 27-field table.

#### P0.3 — The first physical absorber still relies on unexplained state

H-minus is a good first complete absorber, but "implicit H-minus population
factor" is too terse for the declared audience. H-minus population is not a
field that Chapter 4 handed forward. The exact route forms its factor locally
from temperature, the normalized neutral-hydrogen base, and electron density;
the atmosphere route can also carry hydrogen and H-minus departure factors.

Before the H-minus result appears, the prose must answer:

1. why an H-minus absorber can be present without a stored H-minus population;
2. which supplied quantities determine its local factor;
3. that `hydrogen_partition_normalized_ion_stage_populations[:, 0]` is
   \(n_{\rm H\,I}/U_{\rm H\,I}\), not actual H I;
4. what a departure coefficient multiplies and that the value one is the
   reference LTE case;
5. why bound-free and free-free use different table conventions and therefore
   do not receive an indiscriminate extra stimulated-emission factor.

This explanation should be a short causal bridge, not a second Saha
derivation. Chapter 3 remains the owner of ionization equilibrium.

#### P0.4 — Several proposed visible cells violate the one-purpose cell gate

The present checkpoint descriptions compress too many scientific claims into
single cells. The largest risks are:

- the molecular cell, which currently asks for molecule-enabled/disabled
  comparison, stored-H2 invariance, neutral-H sensitivity, several temperature
  seams, a frequency seam, all-warm/mixed-column behavior, reversed table
  weights, a route matrix, and a plot;
- the exact-object cell, which currently combines manifest hashes, two state
  views, table residency, host/device placement, and a standalone dtype
  exception;
- the final synthesis cell, which combines a standard route call trace,
  product slabs, component parity, diagnostic behavior, and extension
  behavior.

These are test plans, not bite-sized teaching cells. Keep every exact seam in
the test suite, but show only the seam that changes the reader's current
mental model. The main-text molecular cell should demonstrate:

- CH/OH versus locally reconstructed H2 ownership;
- one temperature-boundary comparison;
- stored-H2 invariance in the synthesis product;
- one interpreted cool-state plot.

The remaining exact nodes and global-gate combinations remain mandatory
verification, but do not all need printed notebook output.

### P1 — Correct during first narrative draft

#### P1.1 — The figure inventory says six plots but the prose currently asks for nine

The causal outline asks for:

1. the stimulated-emission curve;
2. the H-minus edge;
3. the solar/hot H/He comparison;
4. cool molecular continua;
5. the scattering budget;
6. an ordered component-budget plot;
7. atmosphere-grid coverage;
8. edge reconstruction;
9. a final total/component plot.

Its formal visual plan later declares only six. This conflict should be
resolved before implementation. Retain the six plots in the formal visual
plan. Use compact numerical tables for:

- the five atmosphere grid regimes;
- the ordered component budget and final four-regime parity.

The final parity result is better evidence as exact/tolerance numbers than as
two nearly overplotted curves. The combined two-grid schematic already
supplies the qualitative grid comparison.

#### P1.2 — The H/He and metal passages risk becoming a process parade

Sections 5.6 and 5.7 currently name many families in rapid succession. A
reader can execute a helper without understanding why the next helper is
needed. The prose should use a repeated three-sentence rhythm:

1. identify the missing physical partner or threshold;
2. predict which stellar regime makes the term important;
3. call the exact named component and interpret the change in the budget.

Only one representative neutral-metal bound-free term needs a transparent
level-weight construction. Only one ionic free-free perturbation needs to
demonstrate the actual charge-square population. The remaining exact terms
belong in the named component budget with concise physical ownership, not
miniature source walkthroughs.

#### P1.3 — Implementation tables are clustered into a likely source tour

The current plan places the full IFOP ownership table, four-lane table, static
bundle list, constant tiers, complete field contract, host/device rules, two
product calls, and two secondary lane calls within Sections 5.10 and
5.16–5.20. That sequence is exact but reads like documentation after the
physics story ends.

Use only two reader-facing implementation tables:

1. a **consumer projection** table: atmosphere 18-field adapter versus
   synthesis 18-field trimmed projection, grouped by physical role;
2. a **route identity** table: atmosphere product, synthesis product, sampled
   diagnostic, sampled extension, with grid, dtype/device, output, and
   non-claim.

The full 20-entry runner vector remains visible with the atmosphere product
call because it changes the executed route. Optional IFOP 19, inactive
`ContinuumLevelTables`, loader-only fields, failure classes, and unused public
helpers remain test/coverage facts. They do not advance the main causal
narrative.

#### P1.4 — Chapter 2 acceleration material is currently being re-taught

Chapter 2 owns the meanings of `njit`, `cache=True`, `nogil=True`,
`parallel=True`, `prange`, Torch device priority, cold versus warm timing, and
host/device boundaries. Chapter 5 should apply those ideas to one real
workload, not define all of them again.

The Chapter 5 recap should be limited to:

- one sentence recalling cold compile versus warm execution;
- the exact independence proof: one frequency worker reads immutable state and
  writes one complete, disjoint depth column;
- one serial/parallel parity result and one machine-labelled timing;
- why ordered assembly and interpolation remain outside the worker loop.

Likewise, the synthesis section should explain which *specific* bracket
choices stay in host float64 and which arrays stay on the resolved device. It
should not repeat the generic Torch tutorial or device priority lesson.

#### P1.5 — Chapters 3 and 4 are being recapped beyond the minimum needed

Chapter 3 already derived \(n/U\); Chapter 4 has just displayed both complete
consumer handoffs and the CH/OH slot meanings. Chapter 5 needs one short
backward recap:

> Bound-state sums read a population divided by its partition function;
> free-particle and charge-square terms read actual densities.

Then the metal perturbation makes that distinction executable. Do not derive
partition functions, repeat the Saha relation, repeat molecular mass action,
or reproduce Chapter 4's 18- and 27-field tables.

The H2 material is new only where it describes a **continuum consumer policy**.
Its local reconstruction, temperature gates, and stored-field non-use belong
here; the molecular equilibrium that supplied other molecule populations does
not.

#### P1.6 — The Chapter 6 boundary needs a tighter vocabulary lock

The opening currently asks what happens "if no spectral line is centred
here." Because the reader has not yet built a line, open instead with the
observable:

> Between the narrow dips, why is the gas not transparent?

The term bound-bound line can then appear only when the continuum's limitation
has been demonstrated.

Chapter 5 owns the continuum stimulated-emission factor. Chapter 6 may state
that the same LTE balance appears in its line-strength convention, but it must
not derive the factor again. Chapter 5 may build and name the
`assemble_continuum_line_selection_threshold` output, but it must stop before
oscillator strength, the selection inequality, catalog routing, or general
line-selection logic.

#### P1.7 — Source-function terminology needs a one-paragraph altitude reset

The plan introduces
\(\kappa_\nu^{\rm abs}\), \(\kappa_\nu^{\rm sca}\), the thermal numerator
\(N_\nu\), and \(S_\nu^{\rm cont}\) in quick succession. Chapter 1 introduced
the source function, but a student should not have to recover its unit and
meaning from memory.

Before the first source output:

- recall that \(B_\nu(T)\) is the LTE intensity scale per frequency;
- say that multiplying an absorptive opacity by its source contribution forms
  a numerator with opacity times intensity units;
- say that division by total absorption returns a source function with the
  units of \(B_\nu\);
- state plainly that scattering is extinction but is not added to this
  thermal numerator.

This is a recap of an owned concept, not a transfer derivation. Transfer and
the scattering source remain Chapter 9.

#### P1.8 — Exact table and fit detail must follow a physical prediction

The outline sometimes presents values—the H-minus stored `1e-18` convention,
the reversed H2-CIA weights, the Rayleigh caps, and the `1.0000001` edge
offsets—before stating what physical or numerical question each resolves.
Each needs the same order:

```text
predict the inactive/active side or limiting trend
→ state the exact pinned convention
→ compute the boundary
→ interpret whether the result matches the prediction
```

Otherwise these memorable implementation seams will crowd out the underlying
physics.

### P2 — Polish and rendered-review requirements

#### P2.1 — The summary needs the exact output contract

The six proposed summary statements are physically good, but the global close
template also requires the exact objects now available. Add, without new
terminology:

- atmosphere `continuum_absorption`, `continuum_scattering`, and
  `continuum_source`, each CPU NumPy float64 `(D,30000)`;
- the atmosphere `(D,344)` float32 line-selection threshold;
- synthesis `continuum_absorption` and `continuum_scattering`, each `(D,W)` in
  the resolved device work dtype;
- derived synthesis
  `continuum_opacity = continuum_absorption + continuum_scattering`.

Then state that no line opacity, emergent flux, or spectrum has yet been
computed.

#### P2.2 — The final navigation link must be literal and causal

The final source should use:

```text
### Next: give one transition a strength and shape

[Chapter 6: One Spectral Line](/reader.html?ch=6)
```

The surrounding sentence should explain that a smooth continuum cannot create
the narrow dips, not merely announce the chapter title.

#### P2.3 — The four original schematics need explicit placement rules

Place each schematic before the equation or code whose structure it prepares:

1. cross section to mass opacity before \(n\sigma/\rho\);
2. absorption versus scattering before the source-numerator distinction;
3. `prange` column ownership before the parallel kernel;
4. two-grid/edge-triplet composition before edge indexing and interpolation.

Each caption must call the image conceptual and provide a complete linear
reading. In particular, the `prange` caption must say that each worker owns a
complete matching depth column, and the edge caption must say that an exact
internal edge is assigned to the next red-wavelength interval.

#### P2.4 — Every quantitative plot needs a post-output sentence template

The paragraph immediately after each plot should report:

1. the actual trend visible in the rendered result;
2. the physical reason for it;
3. the limit or route boundary the plot does **not** establish.

All curves should carry physical units, exact quantity names in captions,
declared depth/regime, and stable process colours. No figure should use an
arbitrary normalization unless the text explains why only a ratio is being
tested.

#### P2.5 — Payne Zero naming is currently restrained enough, but exact names
must not leak earlier

The plan generally treats Payne Zero as destination and oracle rather than
paragraph subject. Preserve that restraint. In reader prose, use "the
atmosphere product," "the synthesis product," or the physical process.
Use the product name only for:

- the pinned parity result;
- an exact source or state name whose identity matters;
- a route choice that would otherwise be ambiguous.

Private helper names such as `_hminus_bf_scalar` should not appear in prose.

## Explicit risk register

### Terse-prose risks

| Risk | Why a student may be lost | Required correction |
| --- | --- | --- |
| rapid H/He list | six physical families appear before a reader can predict any one of them | use one missing-physics question and one regime prediction per family group |
| "implicit H-minus factor" | no H-minus population was supplied by Chapter 4 | explain the exact local factor and the role of normalized H I and electrons |
| "Gaunt factor" as a label | the student may mistake it for a population or fitted opacity | define it as a dimensionless quantum correction before the first free-free table |
| source numerator | four opacity/source symbols arrive before their units are stabilized | give the one-paragraph altitude reset in P1.7 |
| compact/hot/lukewarm metals | source categories can sound like new physics classes | organize by neutral bound-free versus ionic bound-free/free-free, then map exact groups |
| one-sided edge sampling | wavelength and frequency directions reverse | say "red means larger wavelength/lower frequency" before the offset formula |

### Hidden-prerequisite risks

| Hidden prerequisite | Owner already available | Minimum local bridge |
| --- | --- | --- |
| \(B_\nu\), frequency/wavelength conversion | Chapter 1 | one sentence and units |
| actual versus \(n/U\) populations | Chapter 3 | one definition and the metal ownership perturbation |
| CH/OH populations | Chapter 4 | state they are supplied normalized slots; do not rerun chemistry |
| local H2 reconstruction | new Chapter 5 consumer policy | show its inputs and prove stored-H2 non-use |
| departure coefficients | exact atmosphere state | define the multiplicative meaning and unit reference case before use |
| Numba/Torch vocabulary | Chapter 2 | apply independence, timing, and placement; do not reteach syntax |
| three-point interpolation | basic algebra, but not assumed as a named method | derive the three basis functions from node reproduction and sum-to-one |

### Source-tour risks

The following belong in tests, the coverage ledger, or one compact
implementation table rather than a paragraph sequence:

- all 46 public continuum symbols;
- exception classes and loader failure branches;
- inactive `ContinuumLevelTables`;
- loader-required but unconsumed Gavrila/Peach fields;
- the full optional IFOP-19 interpolation story;
- private helper names;
- a field-by-field repeat of Chapter 4's 27-field schema;
- a constant inventory unrelated to a currently evaluated formula;
- separate mini-tours of sampled diagnostic and sampled extension helpers.

Completeness is preserved by symbol disposition and tests. Reader visibility
is reserved for objects that change the causal calculation.

### Redundancy locks with Chapters 3, 4, and 6

| Boundary | Chapter 5 may say/do | Chapter 5 must not do |
| --- | --- | --- |
| Chapter 3 | recall \(n/U\); perturb actual and normalized views independently | derive Boltzmann, partition sums, Saha, or electron closure |
| Chapter 4 | consume the two closed views; explain CH/OH aliases and continuum-local H2 policies | solve molecular equilibrium, repeat catalog construction, or repeat full handoff tables |
| Chapter 6 | hand forward continuum slabs and the reference threshold; say narrow features remain | derive oscillator strength, line stimulated-emission balance again, Doppler/damping/Voigt physics, or catalog selection |

## Proposed causal section, cell, and figure order

The following order preserves all required physics while reducing the current
21 numbered subsections to 14 reader-facing sections. It keeps 14 substantial
visible cells, four schematics, and six one-claim plots. Focused tests retain
the denser seam matrix.

| Section | Causal need and exact boundary | Visible cell | Figure or table |
| --- | --- | ---: | --- |
| opening | Return to the Chapter 1 observable: between narrow dips, why is the gas not transparent? State that Chapter 4 supplies particles but not interaction strengths. | — | reuse only the already-established opening spectrum; no golden |
| 5.1 One particle to one gram | Define cross section, number absorption coefficient, and mass opacity; predict linear population and inverse-density scaling. | 1. unit and scaling check, followed by immediate interpretation | schematic 1: cross section → mass opacity |
| 5.2 Net absorption and redirection | Define bound-free/free-free, Gaunt factor, stimulated-emission limits, absorption, scattering, and thermal numerator. | 2. stable stimulated factor plus low/high limits | plot 1: stimulated factor; schematic 2 before the absorption/scattering split |
| 5.3 H-minus, the first complete absorber | Explain the locally inferred H-minus factor, normalized H I/electron ownership, table units, and threshold. Run the manifest/table preflight immediately before using the table. | 3. manifest/table identity; 4. H-minus edge and ordered bound-free/free-free result | plot 2: H-minus edge |
| 5.4 Light-particle background | Grow one H I level into the H I/H II, H2-plus, He-minus, He I, and He II/III budget; compare physical regimes rather than enumerate helpers. | 5. named solar/hot H/He budget | plot 3: one-panel regime switch |
| 5.5 Metals fill the missing budget | Use one neutral photoionization example, then one actual-versus-normalized perturbation for ionic free-free; add remaining exact named families in ordered budget form. | 6. population-owner perturbation and named metal partial sums | compact component table, no new plot |
| 5.6 Molecular continua have exact consumer boundaries | Explain CH/OH photodissociation and H2 CIA once; show CH/OH normalized aliases, continuum-local H2, stored-H2 non-use, and one meaningful temperature seam. | 7. atmosphere-only boundary and H2 ownership check | plot 4: cool CH/OH/CIA; compact route matrix |
| 5.7 Scattering remains separate | Derive Thomson scaling, predict Rayleigh blueward trend, show exact caps, then assemble the named absorption/scattering budget and runner vector. | 8. Thomson/Rayleigh limit and separate-slab check | plot 5: scattering budget; compact IFOP ownership table |
| 5.8 Direct atmosphere sampling | Motivate the five 30,000-point grids from spectral coverage; verify boundary starts, weights, and the 343/344-point line-reference subroute without teaching line selection. | 9. grid/weight/reference-threshold checkpoint | numerical grid table, no plot |
| 5.9 Why frequency columns parallelize | Apply Chapter 2's Numba ideas to disjoint complete depth columns; distinguish parity, cold compile, warm run, and thread scaling. | 10. Python/serial-`njit`/`prange` parity and labelled timing | schematic 3: independent frequency-column ownership |
| 5.10 Synthesis samples only used edge intervals | Define exact red-side assignment, construct triplets, derive node-reproducing basis functions, then interpolate floored positive opacity in log space. | 11. interval assignment, basis, and reconstruction check | schematic 4: two grids and one enlarged interval; plot 6: one reconstructed interval |
| 5.11 Bind the two exact consumer projections | Reconcile Chapter 4's 27-field synthesis schema with its exact 18-field continuum projection; contrast it with the separate 18-field atmosphere adapter; apply exact host/device dtype placement. | 12. grouped projection/shape/device contract | one consumer-projection table |
| 5.12 Close the atmosphere product | Pass the explicit runner-default flags, preserve component/source order, compute `(D,30000)` slabs and `(D,344)` threshold, then open goldens. | 13. named partial sums, full atmosphere route, four-regime parity table | parity table, no plot |
| 5.13 Close the synthesis product and label alternatives | Run standard `continuum(...)` with edge triplets, Coulomb layout `False`, and no `FrequencyInvariants`; form derived `continuum_opacity`; then identify diagnostic and extension lanes without promoting them. | 14. standard route parity plus compact call-trace/route-identity output | four-lane identity/parity table, no plot |
| 5.14 Chapter summary | Answer the opening question, list exact outputs and non-claims, then link causally to Chapter 6. | — | direct `/reader.html?ch=6` link |

## Cell-output discipline for the proposed order

Each cell must have:

1. a preceding prediction and complete reads/writes/shape/unit/dtype/device
   contract;
2. no more than one live question;
3. canonical code or a transparent law, never pasted source;
4. visible output;
5. an immediate paragraph that reads the actual number, boolean, table, or
   curve;
6. one statement of what the output does not prove.

The main text need not print every threshold node or failure injection.
Mandatory exact checks that would make a cell compound remain in focused tests
and the verification ledger.

## Acceptance gate for the first executable draft

Do not accept the chapter until all answers are yes.

- Does the first implementation object appear only after the physical need for
  it?
- Is the static table preflight just in time, before H-minus consumes it but
  after \(n\sigma/\rho\) is understood?
- Can a student explain why H-minus exists without a stored H-minus field?
- Is the 27-field synthesis handoff visibly projected, not contradicted or
  replaced?
- Does every process family enter because the preceding budget is incomplete?
- Are actual, normalized, and continuum-local molecule populations never
  interchanged?
- Are six and only six quantitative plots present, each with one claim and an
  immediate interpretation?
- Are all four original schematics placed before the structure they explain?
- Does the Numba section apply Chapter 2 instead of repeating it?
- Does the source narrative expose only the exact symbols needed by the live
  calculation?
- Does Chapter 5 stop before bound-bound strength, broadening, profiles, and
  catalog selection?
- Does the summary list exact arrays, axes, units, dtype/device, and the
  derived `continuum_opacity` without introducing new material?
- Does the final link make one line—not a repository module—the necessary next
  object?

