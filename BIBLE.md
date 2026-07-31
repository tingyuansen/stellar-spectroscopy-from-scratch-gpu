# Stellar Spectroscopy from Scratch — Book Bible

This file is the non-negotiable standard for the book. `PLAN.md` defines the
architecture and work order. `COVERAGE.md` maps Payne Zero into the book. `PASSDOWN.md` records
live status.

## Mission

The book teaches a reader to rebuild the non-redundant atmosphere, atmosphere-initializer, and
spectral-synthesis calculation used by Payne Zero from physical principles and readable code.
The result is a final-year undergraduate / first-year graduate textbook, an executable
implementation, and a reproducible validation record.

The working Payne Zero source and paper are read-only evidence. They are never runtime
dependencies of the taught path and are never modified:

- Payne Zero source: `/Users/ysting/payne-zero`, pinned in `PLAN.md`;
- paper source: `/Users/ysting/Source_Files_Not_For_Review`, pinned in `PLAN.md`.

The book may copy provenance-clear code, tables, schemas, manifests, checkpoints, and prepared
teaching subsets into this repository. Every copy must record its source identity and checksum.
The reader-facing calculation must run from this repository alone. The pinned source and paper are
development oracles, never prerequisites for reading a chapter or executing its taught code.

## Audience and Prerequisite Floor

Assume only:

- high-school algebra and trigonometry;
- introductory calculus, including derivatives and one-dimensional integrals;
- basic vectors and matrices;
- introductory mechanics, energy, waves, and the ideal-gas idea;
- ordinary Python syntax: variables, functions, arrays, loops, and plots.

Do not assume prior knowledge of:

- astronomy or stellar spectra;
- quantum mechanics beyond what the chapter develops;
- statistical mechanics;
- atomic or molecular spectroscopy;
- radiative transfer;
- stellar atmospheres;
- numerical root finding or differential equations in code;
- NumPy vectorization;
- Numba, threads, race conditions, or reductions;
- PyTorch, GPUs, CUDA, or Apple Metal;
- machine-learning models or principal-component analysis.

When an advanced idea first appears, build it in four layers:

1. the physical question in ordinary language;
2. a concrete example or limiting case;
3. the mathematical statement with every symbol defined;
4. the smallest readable implementation and a check.

Terms such as statistical weight, partition function, ionization stage, optical depth, source
function, opacity, Jacobian, fixed point, cache, kernel, thread, and scatter-add are never treated as
self-explanatory.

## Narrative Viewpoint

The reader is building a modern stellar-atmosphere and spectrum-synthesis code from first
principles. Payne Zero is the destination and numerical reference, not the grammatical subject of
every paragraph.

- Begin with the physical question, derive the needed quantity, and build the smallest honest
  implementation.
- Introduce the exact production name when the reader's object becomes a source interface, stored
  field, serialized key, or validated kernel.
- In surrounding prose, prefer the physical noun—“the atmosphere solver,” “the transfer
  integral,” or “the line catalog”—to repeated product-name attribution.
- End a construction stage by checking that the reader-built result matches the pinned
  implementation.
- Never let exact-name fidelity turn the chapter into API documentation or product promotion.

Payne Zero must still be named when the distinction carries information: when adopting an exact
public name, explaining a production numerical choice, declaring that a component will be built in
a later chapter, identifying an integration fixture, or reporting a parity result. The rule is not
to hide the destination; it is to make the reader earn the destination by constructing it.

## Self-Contained Reader Contract

“From scratch” means the book supplies or derives every dependency needed by its teaching path:

- physical concepts and notation are introduced before use;
- small exact source modules, schemas, and invariant tables live inside this repository;
- teaching catalog subsets and integration fixtures carry provenance, units, shapes, and hashes;
- optional full catalogs have an explicit installation and checksum path;
- notebook setup never imports from `/Users/ysting/payne-zero` or the paper tree;
- golden Payne Zero results are comparison targets loaded only after the reader's computation;
- every chapter ends with a summary and a causal bridge to the next chapter.

A chapter may rely on earlier chapters because this is a sequential course. It may not rely on an
unexplained external document, hidden calculation, private source checkout, or undeclared data
product.

## Scientific Scope

The supported physical model is:

- one-dimensional;
- static;
- plane parallel;
- local thermodynamic equilibrium;
- opacity-sampled and line blanketed;
- hydrostatic, with turbulent pressure disabled;
- convective where the mixing-length calculation activates;
- coupled to direct spectral synthesis.

The book covers:

- atomic and molecular thermochemical equilibrium;
- continuous absorption and scattering;
- atomic, hydrogen, helium, autoionizing, merged-series, and molecular line opacity;
- radiative transfer with continuum scattering;
- Rosseland means, radiative pressure, radiative acceleration, convection, and temperature
  correction;
- atmosphere remapping, fixed-column quantization, convergence, diagnostics, and failure modes;
- the five-label, eight-label CNO, and direct-abundance atmosphere initializers;
- the structured atmosphere interface;
- GPU spectral synthesis and multicore CPU atmosphere iteration;
- data preparation, manifests, caching, prewarming, precision, and reproducibility.

Spectrum fitting, line-list calibration, and survey-specific instrument modeling are out of scope
except for a short boundary explanation when it clarifies the flux contract. NLTE, three-dimensional
structure, spherical geometry, winds, and turbulent-pressure/HLINOP branches are explicit horizons,
not silently simplified features.

## Actual Hardware Architecture

The textbook follows Payne Zero's working production architecture.

### Atmosphere

The physical atmosphere iteration is a NumPy/Numba calculation on multicore CPUs.

- NumPy expresses whole-array algebra.
- `@njit(cache=True, nogil=True)` compiles numerical loops and releases the GIL.
- `parallel=True` and `prange` are used only for independent work.
- Ordered state transitions, recurrences, depth continuation, and fixed-order reductions remain
  ordered.
- Cold compilation, warm cached execution, and thread scaling are measured separately.

The atmosphere continuum follows the same architecture but parallelizes over
independent frequency columns. Its product grid is a direct 30,000-point,
effective-temperature-dependent sampling grid in CPU `float64`; it never uses
the synthesis edge-triplet interpolation.

The progressive physical spine is now an exact typed state chain:

```text
ModelAtmosphere / RunSetup
→ AtmospherePopulationState
→ OpacityState
→ TransferAccumulation
→ IterationFinalization
→ IterationRemap
→ AtmosphereRunResult
```

Chapter 11 owns population and opacity preparation, Chapter 12 consumes that
exact state for transfer/finalization, and Chapter 13 consumes the live
finalization for correction and complete-state remapping. The canonical
`run_atmosphere_model` composes the same boundaries for repeated passes.
Terminal fixed-column quantization remains outside the interior iteration
orbit, and structural convergence remains distinct from scientific
acceptance.

### Synthesis

Spectral synthesis is a PyTorch calculation on CUDA, Apple Metal, or CPU.

- Default device priority is CUDA, then MPS, then CPU.
- Default work precision is float64 on CUDA/CPU and float32 on MPS.
- Line accumulation is intentionally float32 on every backend.
- Large depth–wavelength tensors and sparse deposits stay on device.
- Discrete regime and bracket choices may stay in host float64 when they define parity-sensitive
  branches.
- Completed spectra cross the host boundary once.

### Initializers

The learned initializers use Torch to predict a starting structure. They do not establish physical
validity.

- An initialized atmosphere is always labeled unconverged.
- A retained physical atmosphere must pass the iterative atmosphere solver.
- Direct-abundance initialization is taught with its experimental and mandatory-closure status
  intact.

The two public workflows are implemented as distinct typed routes. A compact
schema fixture may verify the atmosphere-to-synthesis boundary, but it cannot
stand in for either an initialized-label run or a converged physical workflow.
Fresh execution of those public routes requires the complete manifest-owned
synthesis runtime table bundle; a checked-in compact spectrum is not evidence
that every required runtime table is installed.

## Lecture 1 Establishes the Voice

Lecture 1 defines the tone and pacing of the whole book.

Every chapter must:

1. open with one physical question, tension, or observable;
2. state assumptions before using them;
3. define a term at first use in plain language;
4. develop an important equation as physical need → equation → interpretation;
5. state units, sign convention, axis order, shape, dtype, and device before code consumes them;
6. use code cells with one conceptual purpose;
7. use the exact Payne Zero implementation name whenever that object exists, and state its units,
   shape, and mathematical symbol alongside the name;
8. explain why work is independent, why a loop remains, why a dtype changes, or why a hardware
   target is appropriate;
9. include a physical sanity check such as conservation, limiting behavior, monotonicity, scale, or
   sign;
10. close with what was computed, the exact output contract, a concise summary, and the unresolved
    physical or numerical question that motivates the next chapter.

The target code-cell length is 10–30 lines. Sixty lines is a soft ceiling and eighty lines is a hard
ceiling. A longer exact kernel must be split into named conceptual stages without changing its
numerical order.

Avoid:

- opening with a syllabus, benchmark, or marketing claim;
- unexplained arrays or table fields;
- several dense code cells without a prose bridge;
- end-of-chapter exercise sets that postpone useful reasoning instead of teaching it in place;
- implementation archaeology in the main narrative;
- calling a loaded computed field “from scratch”;
- claiming that a numerically plausible answer is verified.

Any worthwhile parameter variation, limiting case, or debugging question belongs in the main
causal sequence as a prediction, a compact executable comparison, and an immediate interpretation.
If it does not help the next dependency, omit it.

## Sequential Book, Progressive Code

This is a sequential textbook, not a collection of unrelated standalone notebooks.

- Each concept is derived once.
- A chapter may use code already built in an earlier chapter.
- A chapter begins with a compact “reads / writes / shape / units / dtype / device” contract.
- At most one short backward recap should be needed.
- Every chapter ends with one explicit forward bridge: what is now trustworthy, what remains
  impossible, and why the next chapter is the necessary next dependency.
- The progressive textbook package is the canonical implementation.
- Chapter sources display and execute the same canonical functions; they do not maintain a second
  handwritten copy.
- Capstones compose the progressive package and never re-inline the full engine.

Generated notebooks and HTML are build products. They are not hand-edited sources.

## Naming and Notation

Depth always runs from the outermost layer to the innermost layer. Column mass increases with array
index.

The book teaches Payne Zero's vocabulary; it does not invent a parallel cleaned-up API. Public
function names, argument names, dataclass fields, serialized keys, array names, axis order,
defaults, branch labels, and output fields must match the pinned source exactly. Units are taught
in the surrounding contract rather than added to or removed from a public name.

Important exact names include:

| Physical meaning | Payne Zero name | Unit or convention |
| --- | --- | --- |
| effective temperature | `effective_temperature` | K |
| base-10 surface gravity | `log_surface_gravity` | \(g\) in cm s\(^{-2}\) |
| bulk metallicity | `metallicity` | dex |
| alpha enhancement | `alpha_enhancement` | dex |
| microturbulent label | `microturbulence_km_s` | km s\(^{-1}\) |
| atmosphere temperature | `temperature` | K, shape `(depth,)` |
| column mass | `column_mass` | g cm\(^{-2}\), shape `(depth,)` |
| gas pressure | `gas_pressure` | dyn cm\(^{-2}\), shape `(depth,)` |
| electron number density | `electron_density` | cm\(^{-3}\), shape `(depth,)` |
| mass density | `mass_density` | g cm\(^{-3}\), shape `(depth,)` |
| Rosseland opacity | `rosseland_opacity` | cm\(^{2}\) g\(^{-1}\), shape `(depth,)` |
| radiative acceleration | `radiative_acceleration` | cm s\(^{-2}\), shape `(depth,)` |
| atmosphere microturbulence | `microturbulence` | cm s\(^{-1}\), scalar or `(depth,)` |
| standard depth coordinate | `standard_rosseland_optical_depth` | dimensionless, `(depth,)` |
| intrinsic synthesis-grid density | boundary-specific `r_grid` or `resolution` | dimensionless \(\lambda/\Delta\lambda\) sampling; see the exact matrix below |
| synthesis interval | `wavelength_start_nm`, `wavelength_end_nm` | nm |
| returned wavelength | `Spectrum.wavelength_nm` | nm |
| returned total flux | `Spectrum.flux_total` | surface \(F_\lambda\) per nm |
| returned continuum flux | `Spectrum.flux_continuum` | surface \(F_\lambda\) per nm |
| returned normalized flux | `Spectrum.normalized_flux` | `flux_total / flux_continuum` |
| initialization status | `atmosphere_converged` | Boolean |
| physical-closure status | `atmosphere_closure_required` | Boolean |

The native schema-v4 array keys must always be quoted exactly. In particular, the book must not
shorten or blur:

- `partition_normalized_populations`;
- `ion_stage_populations`;
- `fractional_doppler_widths`;
- `hydrogen_partition_normalized_ion_stage_populations`;
- `carbon_partition_normalized_ion_stage_populations`;
- the species-specific neutral-population fields;
- the signed continuum-edge frequency and wavelength fields.

The stored `molecular_hydrogen_population` field is not a continuum input in
either product lane. Atmosphere H2 CIA/Rayleigh and synthesis H2 Rayleigh each
reconstruct an H2 population through a distinct local policy. CH, OH, and H2
collision-induced absorption belong to the atmosphere continuum only.

The continuum handoff is not one universal schema. The atmosphere product
consumes an 18-field adapter built from the solved `ModelAtmosphere` and
`AtmosphereRuntimeState`, including packed `(depth,1006)` population views,
hydrogen departure coefficients, and `ch_population`/`oh_population` aliases
of partition-normalized slots 845/847. The synthesis product separately
consumes the 27-field schema-v4 mapping. Never reconstruct one view from the
other or call the CH/OH aliases generic actual-molecule densities.

The intrinsic-grid quantity has one mathematical meaning but several exact
source spellings. Chapters must preserve the spelling at the boundary they are
actually teaching:

| Boundary | Exact spelling | Required behavior |
| --- | --- | --- |
| mathematics and explanatory prose | \(R_{\rm grid}\) | \(R_{\rm grid}=\lambda/\Delta\lambda\) for adjacent intrinsic model samples |
| `synthesize_from_labels` | canonical `r_grid`; alias `resolution` | either name is accepted; unequal double specification raises an error |
| public archive/in-memory `synthesize` | `resolution` only | do not invent an `r_grid` keyword |
| `SynthesisPipeline`, `Grid`, engine/window helpers, molecular compilers, and cache records | `resolution` | retain the pinned internal field and argument name |
| synthesis and prewarm CLIs | `--r-grid` and `--resolution`, stored as `resolution` | the two flags are aliases for the same intrinsic-grid value |

Neither source spelling means instrumental resolving power. An instrument or
line-spread operator is a separate downstream operation.

The two line-production lanes also have different molecular source coverage.
This matrix is the global status and ownership contract; later chapters may
expand a cell, but may not silently broaden it:

| Line family or lifecycle | Atmosphere lane | Synthesis lane | Primary book ownership |
| --- | --- | --- | --- |
| ordinary atomic | The standard high-level solve supplies predicted, observed, and high-excitation sources; they are selected into the common `SelectedLineCatalog` and deposited at runtime. | The atomic catalog is always compiled and deposited; `molecular_lines` does not disable it. | Ch. 7 teaches atomic routes and the common selected-record/deposit machinery; Ch. 11 composes atmosphere selection. |
| diatomic / text bands | `diatomic_lines_path` is in the standard high-level source set; its family-specific strength correction runs before common selected-line deposition. | Manifest-ordered text bands are compiled and deposited when `molecular_lines=True`, which is the public default; `False` is the opt-out. | Ch. 8 owns the family format, selection/compiler rules, and molecular status; Ch. 11 only composes the atmosphere path. |
| TiO | `titanium_oxide_lines_path` is in the standard high-level source set and is selected and deposited at runtime. | The Schwenke compiler is invoked by the standard molecular build and the result is deposited when `molecular_lines=True`. | Ch. 8, with Ch. 11 composition in the atmosphere lane. |
| water / H\(_2\)O | `water_lines_path` is in the standard high-level source set; the dedicated water selector feeds the common runtime deposit. | `compile_h2o_partridge` exists, but the standard `_compile_molecular` path does not invoke it; this is compiler-only, not standard runtime or public opt-in support. | Ch. 8 states and tests the cross-lane difference; Ch. 11 composes the active atmosphere path. |
| H\(_3^+\) | `AtmosphereInput.h3plus_lines_path` and the selector/runtime path exist, but `source_line_paths()` does not supply a default file; it is opt-in by an explicit path. | A species-mass entry alone does not make a feature: no standard source compiler or pipeline wiring supplies H\(_3^+\). | Ch. 8 owns the exact optional/absent boundary; Ch. 11 may compose the atmosphere option without reteaching it. |
| selected/detailed-object reuse | `run_atmosphere_model` generates or loads the common selected catalog and, when enabled, loads the detailed-transition catalog on the first opacity pass, then reuses those objects on later iterations; the selected object includes every admitted family above. | Atomic and enabled molecular window invariants are reusable cache objects keyed by window, `resolution`, device, dtype, feature flag, and source identity. | Ch. 11 owns atmosphere select-once/reuse. Chs. 7–8 own what the selected objects contain; Ch. 10 owns synthesis invariant-cache composition. |

A local pedagogical variable is allowed only when it is genuinely intermediate and not a renamed
production object. At first use, show the relationship

> mathematical symbol → physical meaning → exact Payne Zero name → unit → shape.

If the corresponding source object appears later, the chapter must converge to the exact source
name before presenting a reusable function. A teaching-only simplification must be called a
controlled limit and must never be wrapped in a class or API that looks like an alternative Payne
Zero interface.

The book must distinguish:

- elemental number abundance;
- actual ion-stage population;
- population divided by partition function;
- bound-level population;
- molecular population;
- number-density and mass-opacity quantities.

Use \(R_{\rm grid}\) for the mathematical intrinsic sampling density and the
boundary-specific source spelling in the matrix above. Never regularize
`resolution` to `r_grid` or the reverse merely for stylistic consistency. This
quantity is not instrumental resolving power.

Every flux statement must say whether it refers to `H_nu`, `F_nu`, or `F_lambda`, and whether the
factor `4π` and frequency-to-wavelength Jacobian have been applied.

Historical source-boundary names such as `PFSAHA`, `POPSALL`, `JOSH`, `LINOP1`, `RHOX`, and `XNFP`
appear only where the working Payne Zero code or an external fixed-column format still uses them.
They are explained as boundary vocabulary, not used to narrate a legacy project. Serialized fields
and source-catalog codes remain unchanged at file boundaries.

## Constants and Numerical Conventions

Do not “clean up” constants by replacing all literals with the newest CODATA value. Payne Zero
intentionally contains:

- exact physical constants;
- rounded reference constants required for numerical parity.

The chapter and code must identify which tier is used and why.

Numerical ordering is part of the method when it affects:

- line-selection threshold decisions;
- profile branch selection;
- interpolation brackets;
- fixed-column quantization;
- float32 line deposition;
- thread-private accumulation and reduction;
- atmosphere correction and remapping.

`fastmath` is not enabled where a one-ulp change can alter a discrete physical decision.

## Teaching Numba and GPU Methods

Optimization is taught at the first natural use, not as unexplained decoration.

Where applicable, show:

1. a clear scalar Python statement of the algorithm;
2. a vectorized NumPy or Torch form;
3. a serial `njit` form;
4. a parallel `prange` form when iterations are independent;
5. a parity check;
6. cold and warm timing;
7. thread or device scaling;
8. the reason further parallelism would be unsafe or unhelpful.

Do not manufacture a `prange` example inside synthesis: its real Numba use is the serial molecular
source compiler. Do not manufacture a GPU atmosphere path: the physical atmosphere calculation is
CPU-oriented.

## Visual Pedagogy

Schematics are part of the explanation, not decoration. The official Payne Zero website at
`/Users/ysting/payne-zero-website` is the read-only visual reference.

Use a schematic when a reader first meets:

- a physical geometry or flow of radiation;
- a coupled feedback loop;
- several opacity sources merging into one quantity;
- a branching algorithm or hardware/data movement pattern;
- an interface connecting three or more stages.

The shared visual language is:

- hand-drawn, scientist-sketched objects on pure white;
- muted slate blue, deep navy, warm grey, and pale beige;
- slightly varied line weight but precise composition;
- short, legible labels and generous white space;
- landscape layout unless the concept strongly requires another shape;
- no decorative filler, photorealism, gradients, drop shadows, logos, watermarks, or invented
  numerical values.

Quantitative figures take their publication finish from the Payne Zero paper while remaining
simpler than paper figures:

- prefer one panel and one physical claim;
- build a complicated comparison across successive figures instead of presenting a dashboard;
- use the shared color-safe black/slate, blue, orange, green, magenta, and warm-grey palette;
- keep reference curves dark, model curves colored, and guides quiet;
- use consistent serif mathematical typography, line weights, inward ticks, and white background;
- label every axis with a physical quantity and unit;
- choose limits and annotations to expose the physical relationship;
- never rely on positional Matplotlib colors or unreviewed defaults;
- inspect every rendered figure for clipping, overlap, legibility, contrast, and excess whitespace.

Conceptual schematics and data-valued figures have different evidentiary roles:

- a schematic must say in its caption that it is conceptual when sizes or curves are not measured;
- a numerical plot must be generated by the canonical chapter code from declared data;
- a generated illustration never serves as a parity result or quantitative claim.

Every generated schematic has an owned prompt/specification, source or generation provenance,
alt text, caption, and file hash. The official-site figures are **style references**, not
reader-facing textbook assets. Each chapter receives an original schematic whose objects, labels,
and composition are chosen for that chapter's teaching claim, using the prompt architecture in
`scripts/textbook_schematic_specs.py`. A chapter's visual review checks scientific meaning, label
accuracy, readability at notebook width, light/dark reader behavior, and placement before the
equations or code it prepares.

## Data Honesty

Every numerical file belongs to exactly one class:

1. **Static physical input** — constants, tables, source records, schemas, or checkpoint weights.
2. **Teaching subset** — a documented, checksum-bound slice of a larger static source catalog.
3. **Integration fixture** — a computed upstream state intentionally supplied so one downstream
   stage can be studied in isolation.
4. **Golden output** — a comparison target loaded only after the textbook computation.

These classes live in separate directories and must not share an opaque `.npz`.

Every file records:

- source path or generating command;
- Payne Zero commit;
- source-data checksums;
- byte size and SHA-256;
- array names, shapes, dtypes, and units;
- role in the book;
- whether regeneration needs the optional full catalog bundle.

The approximately 6.8–6.9 GB full source catalog is not silently duplicated. The normal book ships
small invariant tables, manifests, checkpoints when practical, and explicit teaching subsets. A
checksum-verified optional full-data path enables complete atmosphere and full-bandwidth capstones.

## Verification Ladder

Rendered chapters show physical checks and a concise parity result. Strict arrays and tolerances
live in tests and a generated verification ledger.

The ladder is:

1. dimensional and analytic limiting checks;
2. table/schema identity;
3. scalar-to-optimized kernel parity;
4. stage-interface parity;
5. one complete atmosphere-pass parity;
6. fixed-point trajectory and structural convergence;
7. flux, hydrostatic, optical-depth, and spectrum acceptance;
8. same-atmosphere synthesis parity;
9. initialized-label synthesis parity;
10. independently converged atmosphere-plus-spectrum parity;
11. cold-cache/warm-cache equality;
12. CPU/CUDA/MPS and fixed-thread backend comparisons;
13. warm performance benchmarks.

Golden metadata includes:

- Payne Zero commit and paper hash;
- data and checkpoint checksums;
- stellar labels and all configuration;
- Python, NumPy, Numba, and Torch versions;
- device, dtype, and thread count;
- exact or tolerance-based comparison policy.

Do not promise cross-thread bit identity where private accumulators change the grouping. Do not use
“machine precision” as a universal phrase. Tolerances are stage-, dtype-, backend-, and thread-policy
specific.

The four spectrum outputs are all checked:

- `wavelength_nm`;
- total `F_lambda`;
- continuum `F_lambda`;
- normalized flux.

## Chapter Review Gates

No chapter is accepted directly from a subagent.

Each draft passes:

1. its own physics, code, data, and parity contract;
2. a neighbor review with the preceding and following chapter;
3. a dependency and coverage-ledger review;
4. a redundancy and notation review;
5. a rendered visual and pacing review;
6. a whole-book zoom-out after its chapter wave.

After every wave, pause new chapter writing until the global plan and coverage ledger have been
updated.

## Honest Boundaries

The book must explicitly track:

- H2O source compilation exists, but the pinned standard synthesis pipeline does not currently
  invoke it;
- the direct-abundance initializer is experimental and always requires exact physical closure for
  an accepted atmosphere;
- atmosphere convergence is a structural stopping test, not by itself proof of flux closure or
  spectral accuracy;
- molecular equilibrium remains depth sequential because one layer initializes the next;
- line-opacity and transfer results may vary by a few ulp when thread regrouping changes;
- cool, high-gravity molecular atmospheres can leave the useful fixed-point basin;
- turbulent-pressure and HLINOP branches have explicit unsupported-run guards, while NLTE remains
  outside the scientific scope;
- converted atomic, diatomic, TiO, and water source selectors are active atmosphere paths and must
  not be described as unsupported “raw molecular selectors”; atmosphere H\(_3^+\) is instead an
  explicit-path opt-in boundary.

Unknowns are documented and tested. They are not converted into reassuring prose.

## Completion Standard

The project is complete only when:

- every non-redundant Payne Zero atmosphere/synthesis module and public routine is mapped and
  covered;
- every chapter builds and executes;
- every verification layer passes or has a precise, evidence-backed limitation;
- the two public workflows and four stellar regimes are demonstrated;
- the local web reader is coherent and current;
- navigation, chapter numbering, source ownership, and generated artifacts agree;
- no checked-in notebook contains an environmental failure;
- no duplicated builder, mixed-role fixture, filename inversion, or unowned generated artifact
  remains active;
- whole-book physics, flow, redundancy, notation, accessibility, provenance, and reproducibility
  audits are green.
