# Global coverage zoom-out after Chapter 6 evidence

## 1. Scope and verdict

This is an audit of the book architecture and its evidence, not a chapter draft
and not an implementation change. It compares the complete non-fitting
atmosphere, initializer/emulator, and synthesis surfaces of the pinned Payne Zero
checkout with:

- `BIBLE.md`;
- `PLAN.md`;
- `COVERAGE.md`;
- `design/global_chapter_contracts.md`;
- the accepted Chapters 1–5;
- the accepted Chapter 6 causal and exact-source evidence; and
- the generated public-symbol inventories.

The pinned source checkout was read only. Its `HEAD` is
`9c44001feae40b85146630499e6f8a5fed42e5af`, exactly the commit required by
`PLAN.md:12-20`.

**Verdict: REJECT the present documents as a proof of complete global coverage.**

This is a narrow rejection. The fifteen-chapter causal architecture is sound,
Chapters 1–6 form a coherent construction, the optimization and hardware story
tracks the working implementation, and no wholesale rechaptering is indicated.
Three P0 contract repairs are needed before the plan can honestly claim that
every non-redundant routine and branch has a correct owner:

1. replace the universal `r_grid` spelling rule with an exact interface-by-
   interface name matrix;
2. give the atmosphere and synthesis molecular-line branches one explicit
   two-lane ownership/status matrix; and
3. review branch-sensitive symbols in the generated inventory instead of
   treating their module-level inherited assignments as semantic proof.

Once those three repairs are made, the architecture should be re-audited rather
than redesigned.

## 2. Evidence frozen for this audit

| Evidence | Identity used here |
| --- | --- |
| Payne Zero source | commit `9c44001feae40b85146630499e6f8a5fed42e5af` |
| `BIBLE.md` | SHA-256 `1433c2d3d18dd7397f8a739765c7f8e4c36f4b79e41c2809f596fa7fe3bf59b0` |
| `PLAN.md` | SHA-256 `4b0a5449b6a184ba2646fa3a9104f0217508893830236a1b7dca6d04fe18d286` |
| `COVERAGE.md` | SHA-256 `c7fe1c5685b87a0c098c377c7368ee1c1ef23f81ae7880887373530ccb84a2c7` |
| global chapter contracts | SHA-256 `51ed5e93cdd1787ec355155573bfa20bbc7320d376fc3afa16ff3485438b370c` |
| Chapter 6 causal outline | SHA-256 `1b66df5d548f2854f83289fcf9de5109058f1482a7b64aadaff3505d1f57e019` |
| Chapter 6 exact-source contract | SHA-256 `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |
| raw public-symbol inventory | SHA-256 `010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf` |
| symbol-to-chapter ledger | SHA-256 `589322e42fcbfdb46d2603419cf1ce6bdcab913b13f6859204138faa7baee8b4` |

The source inventory contains all 38 Python modules in
`payne_zero_atmosphere` and all 20 in `payne_zero_synthesis`. The generated
coverage ledger contains 1,501 records across public exports, functions,
classes, constructors, methods, annotated fields, and named module data. Of
those records, 1,467 (97.7%) still have `mapping_precision=module_default`; only
34 (2.3%) have a reviewed symbol override. The inventory therefore proves
visibility, but not yet correct semantic placement for every mixed-
responsibility module.

Spectrum fitting, line-list calibration, and implementation of instrument
operators are excluded, as required by `BIBLE.md:124-127`. This audit retains
only the small spectral-operator boundary needed to state the flux and
normalization order.

## 3. What already passes

### 3.1 The global causal construction is correct

The dependency order is physically and pedagogically defensible:

1. radiation, optical depth, hydrostatic balance, and a controlled grey
   atmosphere;
2. numerical meaning, data contracts, Numba, Torch, and honest timing;
3. atomic populations and electron closure;
4. coupled molecular equilibrium;
5. continuous absorption and scattering;
6. one ordinary spectral line;
7. atomic forests and special profiles;
8. molecular bands and source compilation;
9. formal transfer with scattering;
10. complete device-resident synthesis from a supplied atmosphere;
11. construction of a blanketed atmosphere pass;
12. radiative reductions, thermodynamics, and convection;
13. correction and full physical iteration;
14. learned initializers as starting states, never physical closure; and
15. exploratory and independently verified end-to-end workflows.

In particular, teaching synthesis before the physical atmosphere solver is not
a circular dependency. Chapter 10 consumes an explicitly supplied structured
atmosphere integration fixture; Chapters 11–13 later construct the physically
closed product. This follows the forward-reference contract in
`design/global_chapter_contracts.md:697-710`.

Placing the full physical solver before the learned initializer is also
correct. It prevents a neural initializer from being mistaken for an
atmosphere solution and makes Chapter 14's mandatory-closure rule intelligible.

### 3.2 Chapters 1–6 have a coherent local flow

The accepted Chapters 1–5 each end in a named summary and a direct causal link
to the next chapter:

- Chapter 1: `book/chapters/chapter_01.py:1266-1295`;
- Chapter 2: `book/chapters/chapter_02.py:1633-1666`;
- Chapter 3: `book/chapters/chapter_03.py:1723-1760`;
- Chapter 4: `book/chapters/chapter_04.py:1474-1504`; and
- Chapter 5: `book/chapters/chapter_05.py:1555-1592`.

The accepted Chapter 6 outline freezes the same close:
`design/chapter06_causal_outline.md:825-867` specifies a seven-point summary,
the single missing dependency, and `/reader.html?ch=7`.

Chapter 6 is especially well bounded. It teaches one line from physical need,
then reveals exact production names. It does not smuggle catalog compilation,
line-forest selection, `prange`, special hydrogen/helium profiles, molecular
bands, or emergent flux into the chapter. Its one-record atmosphere path is
honestly serial cached `njit`; bulk selection, private buffers, and parallel
reduction are deferred to Chapter 7
(`design/chapter06_causal_outline.md:699-727`). This is the right balance
between a readable construction and exact source behavior.

### 3.3 The optimization and hardware story is source-faithful

The high-level split in `PLAN.md:48-74` and
`design/global_chapter_contracts.md:102-120` matches the pinned source:

- the atmosphere is NumPy/Numba on multicore CPU;
- the initializer performs Torch inference but produces a starting structure;
- synthesis is PyTorch on CUDA, MPS, or CPU;
- synthesis device priority is CUDA, then MPS, then CPU
  (`payne_zero_synthesis/device.py:13-22`);
- default work precision is float32 on MPS and float64 on CUDA/CPU, and an MPS
  float64 request is rejected (`device.py:36-56`);
- line-opacity accumulation remains float32 on every synthesis backend;
- atmosphere `prange` is real and appears in line selection, EOS depth batches,
  continuum frequency kernels, line-opacity private chunks, and transfer
  private chunks; and
- the only synthesis-side Numba use is the serial cached molecular compiler
  (`source_catalog_molecular_compiler.py:431`).

The documents correctly prohibit an invented synthesis `prange` lesson and a
nonexistent `torch.compile` path (`BIBLE.md:348-357`,
`design/global_chapter_contracts.md:397-404`). The planned progression—readable
loop, vector form where honest, serial `njit`, then `parallel=True`/`prange`
only for independent work—fits the requested undergraduate/early-graduate
level.

### 3.4 No forbidden framing or detached work was found

The current book sources and design documents contain no active `kgpu`
narrative. Historical boundary vocabulary is confined by `BIBLE.md:313-316`;
it is not used to make the reader tour an earlier project.

No accepted chapter contains a detached exercise or homework section. Useful
predictions, limiting cases, and debugging checks are in the causal main text,
consistent with `PLAN.md:43-46` and `BIBLE.md:200-208`.

The product name is used mainly at exact interfaces, provenance statements,
and parity checks. The narrative remains a from-scratch construction rather
than repeated product attribution.

## 4. Global stage-by-stage coverage

The following table records the independent result of comparing the source
surface with the fifteen chapter contracts. “Pass” means the non-redundant
physics has a sensible primary owner. “Repair” means the architecture is still
appropriate, but the evidence or exact disposition is incomplete.

| Chapter | Non-redundant source responsibility | Audit result |
| --- | --- | --- |
| 1 | scope, Planck radiation, flux/intensity, optical depth, grey structure, hydrostatic intuition | Pass. Controlled model and exact later boundaries are distinguished. |
| 2 | units/axes/data roles; NumPy/Numba/Torch; cache timing; schema; device contract; exact integration variants | Pass in prose and implementation; ledger ownership is stale for several exact routines. |
| 3 | partition functions, ground corrections, Saha, full/fixed-\(n_e\) population states, packed layouts, atomic internal energy | Pass in accepted chapter; ledger statuses and some inherited locations need synchronization. |
| 4 | atmosphere and synthesis molecular catalogs, mass action, continuation, molecular state/energy | Pass and already has the first narrow symbol overrides. |
| 5 | distinct atmosphere and synthesis continuum schemas; all active processes; absorption/scattering; interpolation | Pass. It correctly excludes formal transfer and production Rosseland averaging from this chapter. |
| 6 | one ordinary atomic line, widths, damping, Voigt evaluation, cutoff, one-record deposits in both lanes | Pass at causal/evidence level. The chapter avoids false cross-lane slab parity. |
| 7 | atomic catalog decode/correction/routes, line selection, ordinary forests, H/He/autoionizing/merged-series branches, thread-safe deposits | Architecturally complete. Must explicitly hand molecular source-family branches to Chapter 8. |
| 8 | molecular source formats/compilation, populations, chunks, synthesis deposits, feature-status boundary | Repair. It specifies the synthesis lane well but does not yet give the atmosphere molecular selection/deposition branches an exact home. |
| 9 | ordered optical-depth integration, thermal/scattering source, source iteration, total/continuum transfer, flux conversion | Pass. Earlier chapters may teach exact component routines without rederiving transfer here. |
| 10 | window invariants, star state, caches, context/crop, device/dtype islands, complete supplied-atmosphere synthesis | Pass in architecture. Optional `source_path`, `keep_slabs`, and `spectral_operator` require explicit boundary dispositions. |
| 11 | exact atmosphere seed/config/setup, quantization/remap, 30,000-frequency blanketing state, select-once/reuse | Pass if Chapter 8 owns family-specific molecular behavior and Chapter 11 only composes it. |
| 12 | frequency reductions, Rosseland mean/depth, radiative support/heating, EOS derivatives, convection | Pass. This is the correct production owner for the Rosseland and convection implementations. |
| 13 | correction, remap, iteration order, convergence, publication, cache/thread/failure behavior | Pass in architecture. Unsupported-branch claims must be tied to actual guards rather than broad prose. |
| 14 | five-label, CNO8, direct-abundance initializers; transforms, network/PCA decode, support, retries, provenance, mandatory closure | Pass. No synthesis or fitting dependency is required to validate the initializer. |
| 15 | exploratory versus converged workflows, four regimes, final independent gates, reproducible public interfaces | Pass. It composes prior objects and does not re-inline the engine. |

All 58 modules have at least one module-level location. No complete physics
domain is absent. The remaining problems are exact names, branch ownership,
and the difference between an inherited module assignment and a reviewed
symbol disposition.

## 5. P0 findings

### P0.1 — `r_grid` and `resolution` are incorrectly collapsed

`BIBLE.md:232-235` requires exact public argument spelling, but
`BIBLE.md:255,307-308` and
`design/global_chapter_contracts.md:78-79` then impose `r_grid` as if it were
the one exact spelling everywhere. That is not the pinned interface.

The source has four distinct boundaries:

| Boundary | Exact pinned spelling |
| --- | --- |
| mathematical intrinsic grid density | \(R_{\rm grid}\) |
| `synthesize_from_labels` | canonical `r_grid`; `resolution` also accepted and conflict-checked |
| public archive/in-memory `synthesize` | `resolution` only |
| `SynthesisPipeline`, engine helpers, `Grid`, molecular compilers, cache records | `resolution` |
| CLI | `--r-grid` and `--resolution` aliases stored as `resolution` |

Evidence:

- `_resolved_r_grid(*, r_grid, resolution)` checks double specification and
  resolves the label API (`payne_zero_synthesis/api.py:221-237`);
- `synthesize(..., resolution=20000.0, ...)` has no `r_grid` parameter
  (`api.py:442-452`);
- `synthesize_from_labels(..., r_grid=None, resolution=None, ...)` exposes both
  (`api.py:732-760`);
- `SynthesisPipeline.__init__(..., resolution=20000.0, ...)` uses
  `resolution` (`pipeline.py:962-977`); and
- the molecular compilers also use `resolution`.

The current universal rule can force Chapter 10 to invent a parameter name at
the very point where the book promises exact source spelling. Replace it with
the table above in the Bible, global contracts, coverage ledger, and relevant
chapter contracts. The physics should still be explained as intrinsic
\(\lambda/\Delta\lambda\) grid density, never as instrumental resolving power.
This is a contract correction, not permission to rename the source.

### P0.2 — Atmosphere molecular-line branches lack exact ownership and status

The pinned standard atmosphere workflow actively supplies molecular line
catalogs:

- the high-level solve builds `AtmosphereInput` with `source_line_paths()`
  (`payne_zero_atmosphere/cli.py:414-424`);
- that standard source set includes diatomic, TiO, and water arrays
  (`source_catalogs.py:142-166`);
- `_generate_standard_selected_lines` forwards atomic, diatomic, TiO, water,
  and optional H3+ paths (`runner.py:599-646`); and
- `generate_selected_lines` has distinct diatomic, TiO, water, and H3+ paths
  (`line_selection.py:1101-1161`).

The current Chapter 8 contract, by contrast, describes the synthesis compiler
and synthesis runtime boundary: text bands and TiO run, whereas the synthesis
H2O compiler is not wired into the standard synthesis pipeline
(`design/global_chapter_contracts.md:391-414`). Chapter 11 says only
“first-pass catalog selection and later object reuse”
(`global_chapter_contracts.md:492-511`). This leaves enough ambiguity to
misstate the working atmosphere water path as the unwired synthesis H2O path.

The coverage boundary “raw molecular selectors unsupported by pinned exact
runner” (`COVERAGE.md:91-100`) is also too broad. The actual high-level runner
uses the converted source arrays above. If “raw selector” means a different
unported fixed-format input, the ledger must name that exact input and guard.
It must not label the active converted diatomic/TiO/water selector path
unsupported.

There is one additional source inconsistency that the textbook must resolve
from runtime behavior: `source_line_paths` says its keys include H3+ at
`source_catalogs.py:143-148`, but the returned mapping at lines 152-166 contains
no H3+ entry. The selection function accepts an explicitly supplied H3+ file,
so H3+ is an opt-in/nonstandard atmosphere branch, not part of the default
source set and not simply nonexistent.

Freeze one two-lane matrix before drafting Chapters 7–11:

| Family/path | Atmosphere lane | Synthesis lane | Primary teaching owner |
| --- | --- | --- | --- |
| ordinary atomic source selection | standard | standard atomic catalog route | Ch. 7 |
| diatomic molecular source | standard converted atmosphere source | text-band runtime | Ch. 8 |
| TiO | standard converted atmosphere source | standard runtime | Ch. 8 |
| H2O/water | standard converted atmosphere source | compiler exists; standard synthesis runtime omits it | Ch. 8 |
| H3+ | explicit opt-in atmosphere path; absent from default source mapping | no standard runtime path identified | Ch. 8 status table / App. B data detail |
| selected/detailed object reuse during a physical pass | composed after family-specific teaching | not applicable | Ch. 11 |

Chapter 7 may teach the general selector/deposit machinery once. Chapter 8
must teach the molecular family differences and the two product-lane status
matrix. Chapter 11 should consume the resulting selected catalogs without
rederiving them. This preserves both completeness and non-redundancy.

### P0.3 — The symbol inventory is exhaustive but not yet a semantic coverage proof

The generated inventory and rebuild test are valuable. They ensure that a
module or public object cannot disappear silently. However,
`scripts/build_symbol_coverage.py:2-12` explicitly says that
`mapping_precision=module_default` is only a first expansion to be refined
during chapter construction.

At the audited hashes, 1,467 of 1,501 records still inherit one module-wide
owner, gate, responsibility, and status. That is unsafe for mixed modules. The
problem is visible in concrete already-taught objects:

- `payne_zero_synthesis.radiative_transfer.planck_bnu` is inherited as
  Chapter 9/planned, although accepted Chapter 1 teaches the exact routine
  (`book/chapters/chapter_01.py:307-364`);
- `payne_zero_synthesis.radiative_transfer.integrate_optical_depth` is inherited
  as Chapter 9/planned, although the global exact spine and accepted Chapter 2
  teach it (`global_chapter_contracts.md:198-203`,
  `book/chapters/chapter_02.py:135-186`);
- atmosphere `integrate_on_depth_grid` and
  `accumulate_transfer_range_parallel` omit their Chapter 2 ownership even
  though the global contract assigns them there;
- `ground_partition_value` remains Chapter 3/planned even though accepted
  Chapter 3 derives and uses it
  (`book/chapters/chapter_03.py:519-540`);
- `rosseland_mean_step` inherits Chapter 5 even though the accepted Chapter 5
  contract explicitly excludes the production Rosseland mean
  (`design/chapter05_exact_source_contract.md:26`) and Chapter 12 owns it; and
- `SynthesisPipeline.__init__` and `.run` inherit Chapter 7 despite containing
  the complete device pipeline and branch options chiefly owned by Chapter 10.

The required repair is not 1,501 handcrafted prose entries. It is a targeted
semantic review of every mixed-responsibility module and every optional,
unsupported, compatibility, diagnostic, or test-only branch. Each such symbol
must receive:

- its actual primary chapter or appendix;
- one of `taught`, `composed`, `plumbing-only`, `compatibility-only`,
  `diagnostic-only`, or `unsupported`;
- the exact source spelling where reader-visible;
- a gate appropriate to that role; and
- a status synchronized with accepted chapters.

The inherited default may remain for genuinely homogeneous private plumbing,
but it cannot be cited as proof that a branch-sensitive public routine has been
pedagogically placed.

## 6. Required branch dispositions

These branches need explicit placement during the P0.3 review. The placement
below avoids fitting content, source archaeology, and repeated lessons.

### Reader-visible main text

- standard atmosphere diatomic/TiO/water selection and deposition status;
- synthesis text-band and TiO compilation/runtime;
- synthesis H2O compiler-only boundary;
- LTE line source rebuilt with `planck_bnu` and zero standard line scattering;
- exact CPU/Numba and CUDA/MPS/CPU behavior where the relevant kernel is first
  accelerated;
- default caches, context samples, crop-before-host-transfer, and float32
  accumulation;
- unsupported turbulent-pressure and HLINOP failure behavior; and
- the distinction between initializer success, structural convergence, and
  independent physical/spectral acceptance.

### Compact boundary only

- `spectral_operator`: one Chapter 10 or Appendix C/D box stating that it
  operates on total and continuum wavelength-density flux together, on device,
  before normalization; no instrument or fitting implementation
  (`pipeline.py:1480-1509`);
- `source_path`: comparison/reference-source input only; standard LTE rebuilds
  the Planck line source and uses zero line scattering
  (`pipeline.py:1142-1159,1431-1448`);
- `keep_slabs=True`: diagnostic host-return option, not the standard
  device-resident workflow (`pipeline.py:1294-1304,1524-1543`); and
- H3+: opt-in/nonstandard atmosphere source status, with data provenance in an
  appendix.

### Test/ledger only

- `wing_mode="loop"` as the readable/parity alternative to the standard
  batched wing path;
- `host_accumulator` as a restricted metal-only unstimulated diagnostic path;
- failure injection, full tolerance matrices, branch call traces, and
  exhaustive catalog hashes;
- unavailable hardware recorded as unavailable rather than passed; and
- exact compatibility round trips and alternate schema loaders.

### Appendix only

- API/CLI spelling and complete public-symbol index;
- data installation, environment variables, cache paths, checksums, and
  licensing;
- fixed-column and older schema compatibility behavior;
- full tolerance profiles and reproducibility metadata; and
- boundary source codes whose spelling must remain exact at file interfaces.

The compatibility appendix should state present input behavior without turning
the main narrative into a history of earlier implementations.

## 7. P1 findings

### P1.1 — Synchronize the human and generated ledgers with accepted Chapters 1–5

Several statuses and module rows lag behind accepted content:

- synthesis `radiative_transfer.py` needs Chapter 1 for `planck_bnu` and
  Chapter 2 for `integrate_optical_depth`, while Chapter 9 retains the formal
  transfer ownership;
- atmosphere `radiative_transfer.py` and `transfer_kernels.py` need Chapter 2
  for the exact integration/parallel spine;
- `ground_partition_table.py` and the atomic-EOS end-to-end gate should reflect
  accepted Chapter 3 evidence rather than remain wholly `planned`;
- `pipeline.py` needs a narrow Chapter 3/4 supporting assignment for its
  structured-column builder, not a blanket movement of the complete pipeline;
- `rosseland_mean.py` should be Chapter 12 production ownership, with Chapter 1
  only the controlled concept and no Chapter 5 implementation claim; and
- `microturbulence.py` should distinguish Chapter 3/6 use of the physical width
  from Chapter 11 ownership of `standard_microturbulence`.

These are ledger corrections, not reasons to repeat material in later
chapters.

### P1.2 — Make unsupported-runner claims evidence-specific

`_require_supported_run_setup` actually guards:

- fewer than one iteration;
- the turbulent-pressure branch; and
- opacity flag 13, HLINOP
  (`payne_zero_atmosphere/runner.py:2003-2020`).

The runner docstring also names NLTE and raw molecular selectors
(`runner.py:1604-1613`), but the cited guard does not independently reject every
such spelling. Every “fails loudly” statement in `BIBLE.md:494-507` must point
to the exact validation or configuration path that enforces it. If a branch is
merely absent from the standard route rather than explicitly rejected, say
that instead.

### P1.3 — Freeze density before Chapters 7–14 are drafted

Fifteen navigation chapters remain feasible, but fifteen uniform 90-minute
meetings do not. Accepted Chapter 2 already has 31 visible code cells and is
deliberately two lesson movements
(`design/chapter02_acceptance.md:70-84`). That is an honest pacing decision,
not a defect.

The existing global density rule is appropriate: use web anchors/movements by
default; split only after a chapter exceeds 18 substantial visible code cells,
cannot fit a 90-minute lecture/lab, or cannot keep independent parity suites
understandable (`global_chapter_contracts.md:801-816`).

Apply that measurement prospectively:

- Chapter 7 is the clean optional sixteenth-chapter split point: ordinary
  forests versus H/He/special profiles;
- Chapter 11 should have separate seed/setup and blanketing-state routes;
- Chapter 12 should separate radiative reductions from thermodynamics/
  convection;
- Chapter 13 should separate one correction/pass from iteration/convergence;
  and
- Chapter 14 should separate decoder mechanics from support/provenance and
  mandatory closure.

Do not delete physics to preserve the number fifteen. Conversely, do not split
only because a source module list looks long. Measure the rendered causal
lesson.

### P1.4 — Keep synthesis boundaries out of fitting

The optional spectral operator is already identified as interface-only in
`COVERAGE.md:127-140`. Give it one exact owner and gate. It should establish
that total and continuum flux are transformed together and normalized
afterward. No fitting objective, optimizer, likelihood, survey model, or
line-list calibration belongs in the main chapters.

## 8. P2 findings

1. Chapter 2's closing material contains a broad “Chapters 11–14” forward
   pointer before its direct Chapter 3 handoff. Replace broad range pointers
   with the single deferred object and its eventual owner where practical.
2. Chapter 4 announces Chapter 5 shortly before its formal summary/next link.
   One causal bridge at the close is enough.
3. When Chapter 6 is built, retain its current 15-cell causal scale and resist
   expanding hidden manifest/oracle mechanics into visible source tours.
4. Generate the complete public API/CLI index from the symbol ledger as an
   appendix. Do not duplicate that inventory as prose inside physics chapters.
5. After each chapter wave, update statuses immediately so “accepted” chapters
   do not coexist with `planned` exact routines for multiple waves.

These are editorial or evidence-maintenance improvements. None requires a new
chapter.

## 9. Fifteen-chapter feasibility decision

**The fifteen-chapter architecture is feasible and should remain the default.**

It succeeds because its chapter boundaries follow dependency changes rather
than package modules:

- Chapters 1–5 establish one reusable thermochemical/continuum foundation;
- Chapters 6–10 construct the spectrum from one line through complete transfer;
- Chapters 11–13 construct the physical atmosphere pass and iteration;
- Chapter 14 explains why a fast learned start is not a solution; and
- Chapter 15 composes and independently accepts the workflows.

There is no missing conceptual stage that presently demands a sixteenth
chapter. The one credible split is already documented at Chapter 7 and should
be triggered by measured draft density, not by anticipation. Chapters 11–14
are better handled as multiple routes/acts under stable navigation unless the
rendered density gate fails.

The architecture is also nonredundant if the ownership table is enforced:

- derive shared physics once;
- reveal exact names at the earned production boundary;
- let later chapters compose the object;
- keep exhaustive parity detail in tests/ledgers;
- keep compatibility, installation, and full API spelling in appendices; and
- give each chapter one summary and one causal next link.

## 10. Repair order and re-audit gate

The shortest safe path to an acceptable global plan is:

1. correct the `r_grid`/`resolution` interface matrix globally;
2. freeze the two-lane molecular family/status matrix and Chapter 7/8/11
   ownership;
3. apply reviewed overrides to mixed/branch-sensitive symbols, including all
   optional and unsupported branches listed above;
4. synchronize accepted Chapter 1–5 statuses and module rows;
5. add exact evidence for every unsupported-runner claim;
6. measure the planned visible density/routes for Chapters 7–14; and
7. rerun this global zoom-out before Chapter 7 construction is accepted.

The re-audit should pass only if:

- every branch-sensitive public symbol has a reviewed semantic disposition;
- no public interface is taught under an invented spelling;
- atmosphere and synthesis H2O status cannot be confused;
- optional/test/compatibility code cannot leak into the main causal narrative;
- all accepted chapters and ledger statuses agree; and
- the fifteen chapter summaries and next links form one dependency chain.

## 11. Final decision

**REJECT — global completeness proof at the audited hashes.**

- **P0:** Correct the exact `r_grid`/`resolution` interface matrix.
- **P0:** Freeze the atmosphere/synthesis molecular-line ownership and feature-
  status matrix, including standard water and opt-in H3+ behavior.
- **P0:** Semantically review mixed-module and branch-sensitive public symbols;
  module-default inheritance is visibility evidence, not complete ownership
  proof.
- **P1:** Synchronize ledger ownership/status with accepted Chapters 1–5.
- **P1:** Tie unsupported-branch language to exact guards and distinguish
  rejection from absence in the standard route.
- **P1:** Measure and freeze route/cell density for Chapters 7–14 while keeping
  fifteen navigation chapters as the default.
- **P1:** Confine the spectral-operator interface to one flux-order boundary;
  fitting remains out of scope.
- **P2:** Remove redundant/broad forward pointers, generate API detail in an
  appendix, and update ledger status after every accepted wave.

**ACCEPT — the fifteen-chapter causal architecture and the Chapter 1–6 flow,
subject to the P0 repairs above.**
