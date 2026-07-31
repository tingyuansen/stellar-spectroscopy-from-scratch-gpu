# Global Chapter, Notation, and Handoff Contract

This document is the central integration contract for the fifteen-chapter book. It reconciles:

- `design/part1_foundations_brief.md`;
- `design/part3_part4_synthesis_brief.md`;
- `design/part5_part6_atmosphere_brief.md`;
- `BIBLE.md`, `PLAN.md`, `COVERAGE.md`, and
  `design/pedagogical_flow_rubric.md`.

The pinned implementation identity is Payne Zero commit
`9c44001feae40b85146630499e6f8a5fed42e5af`. The pinned source and paper are
read-only development oracles, not reader dependencies.

## Authority and superseded draft policies

When a detailed brief conflicts with this document, this document governs the
reader-facing book. In particular:

- there are no detached exercise sections;
- useful variations, limiting cases, and debugging questions are taught in the
  main causal sequence, with a prediction, a small computation, and immediate
  interpretation;
- official website figures are style references only;
- every reader-facing schematic is an original textbook composition with an
  owned prompt, provenance record, checksum, alt text, and scientific audit;
- “Payne Zero” is named at exact interfaces, numerical choices, deferred-feature
  boundaries, and parity checks, not used as paragraph-level branding;
- the reader builds the calculation from physical need; production names arrive
  when the constructed object reaches the exact implementation boundary.

The detailed briefs remain the full field/API/source inventories. This document
owns chapter order, primary concept ownership, cross-chapter interfaces, causal
handoffs, and whole-book redundancy policy.

## One narrative spine

The book answers one question:

> Given a physical description of a star, how can we compute the flux that
> leaves its surface at every wavelength and know that the answer came from the
> intended atmosphere, physics, data, and numerical algorithm?

The reader earns the answer in this order:

```text
observed spectral structure
    → depth-dependent atmosphere
    → trustworthy numerical and data contracts
    → atomic and molecular thermochemical state
    → continuum and line opacity
    → radiative transfer
    → native spectrum from a supplied atmosphere
    → physical atmosphere iteration
    → learned starting states with mandatory closure
    → verified labels-to-spectrum workflow
```

Each chapter opens because the previous chapter still cannot compute one
necessary object. Each chapter closes by naming the exact object now available,
the claim it cannot yet support, and the single missing dependency that opens
the next chapter.

## Global notation, array, and numerical contracts

### Physical coordinates and units

- Depth runs outermost to innermost. Array index and `column_mass` increase
  inward.
- Atmosphere work uses CGS units unless an exact public argument states
  otherwise.
- Wavelength-facing public synthesis arguments and outputs use nanometres.
- `effective_temperature` is in K.
- `log_surface_gravity` means
  \(\log_{10}(g/\mathrm{cm\,s^{-2}})\).
- `microturbulence_km_s` is a public label in km s\(^{-1}\);
  `microturbulence` inside an atmosphere is in cm s\(^{-1}\).
- \(R_{\rm grid}=\lambda/\Delta\lambda\) denotes the intrinsic logarithmic
  wavelength sampling in mathematics. It is not instrumental resolving power.

Exact source spelling depends on the boundary and must not be regularized:

| Boundary | Exact pinned spelling | Contract |
| --- | --- | --- |
| `synthesize_from_labels` | canonical `r_grid`; alias `resolution` | either may be supplied; unequal double specification is rejected |
| public archive/in-memory `synthesize` | `resolution` only | no `r_grid` keyword exists |
| `SynthesisPipeline`, `Grid`, engine/window helpers, molecular compilers, and cache records | `resolution` | preserve the internal argument/field/key name |
| synthesis and prewarm CLIs | `--r-grid` and `--resolution`, stored as `resolution` | both flags mean intrinsic model-grid density |

An instrumental line-spread or resolving-power operator is a distinct,
downstream interface.

### Radiation

Every flux statement identifies its quantity:

- `H_nu`: Eddington flux;
- \(F_\nu=4\pi H_\nu\): surface flux per frequency;
- \(F_\lambda\): surface flux per wavelength after the Jacobian.

`Spectrum.flux_total` and `Spectrum.flux_continuum` are \(F_\lambda\) per nm.
`Spectrum.normalized_flux` is their dimensionless ratio.

### State representations

- The standard atmosphere grid has 80 layers:
  `10**(-6.875 + 0.125*np.arange(80))`.
- Atmosphere atomic state uses packed `(depth, 1006)` columns.
- Synthesis population state uses `(depth, 6, 139)` cubes.
- Actual ion-stage populations and populations divided by partition functions
  are separate arrays with separate meanings.
- Atmosphere opacity slabs are generally depth-major `(depth, frequency)`.
- Synthesis transfer tensors are generally `(wavelength, depth)`.
- `planck_bnu` specifically returns `(depth, wavelength)`.
- No chapter may invent one “universal” axis order to hide these exact
  differences.

### Backend and precision

- Physical atmosphere iteration: CPU NumPy `float64` plus Numba.
- `@njit(cache=True, nogil=True)` and `prange` are introduced only where the
  exact implementation has independent work.
- Molecular depth continuation, transfer depth recurrences, correction order,
  and fixed-order reductions remain ordered.
- Synthesis device priority: CUDA, then MPS, then CPU.
- Synthesis work dtype: `float64` on CUDA/CPU, `float32` on MPS.
- Shared synthesis line accumulation and scattering source iteration use
  `float32` on every backend.
- Completed spectral arrays cross to host only at the final construction
  boundary and become NumPy `float64`.
- There is no `torch.compile` path in the pinned implementation.

### Numerical identity

Values alone do not define parity. These are part of the algorithm:

- data/checkpoint/catalog identity;
- axis and traversal order;
- dtype and device;
- discrete bracket, selection, and profile decisions;
- float32 deposition order;
- fixed thread grouping and reduction order;
- fixed-column format/parse boundaries;
- correction, remap, and carried-state order;
- cold, warm, and fresh-process cache state.

## Data-role contract

Every file used by a chapter has exactly one role:

| Role | Meaning | May influence the reader-built result? |
| --- | --- | --- |
| `data/static` | immutable physical tables, schemas, checkpoints | yes |
| `data/subsets` | checksum-bound small source-catalog slices | yes, for the declared teaching case |
| `data/fixtures` | explicit computed upstream state used to isolate a lesson | yes, but never called “built here” |
| `data/golden/payne_zero` | comparison-only pinned outputs | no, until after the reader computation |

Every asset records source identity, checksum, shape, units, dtype, and role.
Full source catalogs remain optional, explicitly installed, and
checksum-verified. Compact pedagogy must not be described as full-catalog
parity.

## Fifteen chapter contracts

### Chapter 1 — From Starlight to a First Grey Atmosphere

**Question.** Why does a stellar spectrum require a depth-dependent
temperature and pressure structure rather than one emitting surface?

**Primary ownership.** Observable and forward problem; atmosphere versus
synthesis; plane-parallel/static geometry; photon energy; intensity and flux;
Planck radiation and its limits; extinction, optical depth, LTE source,
constant-source slab, column mass; angular moments; grey radiative equilibrium;
Rosseland weighting; hydrostatic balance; the standard 80-layer grid.

**Produces.** CPU NumPy `float64[80]`
`standard_rosseland_optical_depth`, `temperature`, `column_mass`, and
`gas_pressure`, plus the exact `planck_bnu` checkpoint. These form a controlled
grey scaffold, not a `ModelAtmosphere`.

**Checks in the main text.** Photon-to-thermal energy scale; Rayleigh–Jeans and
Wien predictions; optically thin/thick slab limits; two-window harmonic mean;
\(T(\tau_{\rm Ross}=2/3)=T_{\rm eff}\); inward monotonicity; unit-opacity
hydrostatic identities and radiation-pressure scale.

**Original schematic.** Depth-dependent forward problem; wavelength-dependent
formation depth.

**Close and handoff.** The scaffold explains why layers exist but cannot yet
be trusted as a large numerical calculation. Chapter 2 must make depth
integration, axes, data identity, and acceleration explicit before chemistry
multiplies the state.

### Chapter 2 — From Equations to Fast, Trustworthy Kernels and Explicit Data

**Question.** How can the same depth integral remain scientifically identical
when expressed as readable loops, NumPy arrays, compiled CPU work, parallel
reductions, and Torch device tensors?

**Primary ownership.** Reads/writes/shape/unit/dtype/device contracts;
broadcasting, strides, and ordered scans; `njit`, `cache=True`, `nogil=True`,
`parallel=True`, and `prange` as distinct choices; Torch devices and host
boundaries; cold/warm/fresh-cache timing; the first logarithmic-to-linear
abundance conversion; element/isotope/ion/molecule distinctions; the
static/fixture/golden data roles; manifests and checksums; the difference
between actual and partition-normalized population arrays; and the validation
ladder from analytic limit through local parity to a pinned comparison.

**Exact numerical spine.**

- `payne_zero_atmosphere.radiative_transfer.parabolic_coefficients`;
- `integrate_on_depth_grid` and its compiled equivalents;
- `accumulate_transfer_range_parallel` with chunk-private state and fixed-order
  reduction;
- `payne_zero_synthesis.radiative_transfer.integrate_optical_depth` with
  `(wavelength, depth)` and `cumsum(dim=1)`.

These are compared as genuinely distinct implementations. No invented wrapper
pretends that their layouts or dependency graphs are identical.

**Produces.** An auditable scalar depth recurrence; exact NumPy, compiled
Numba, fixed-order parallel, and Torch comparisons; declared timing and
tolerance policies; one manifest-bound regression chain; and two named but
still physically uncomputed population representations.

**Course pacing.** Chapter 2 moves from the smallest depth recurrence through
batching, compilation, safe frequency parallelism, device placement, tolerance,
and timing. Only then does it attach the result to a mixture, exact input
bytes, and population meanings. Full storage schemas and public product types
remain deferred until the reader has built the physical quantities they store.

**Checks in the main text.** Analytic integral; scalar/array/compiled/parallel/
Torch agreement; cold versus warm timing; manifest and checksum identity;
pinned-output comparison; and actual-versus-partition-normalized
representation identity.

**Original schematic.** Independent wavelength work versus ordered depth;
acceleration ladder; CPU atmosphere/device synthesis split; actual versus
partition-normalized population.

**Close and handoff.** The numerical containers are trustworthy, but their
population fields are still empty. Chapter 3 must determine how a temperature,
pressure, and mixture divide matter among levels, ions, and electrons.

### Chapter 3 — Atoms, Ions, and Electrons

**Question.** Given temperature, pressure, and composition at one depth, how
many particles occupy each excitation and ionization state, and how is the
electron density made self-consistent?

**Primary ownership.** Atomic levels and statistical weights; Boltzmann
factors; partition functions and stable level sums; Saha ionization; occupation
and pressure-ionization corrections; charge conservation and damped electron
closure; mass and perturber densities; Doppler support; atomic internal energy;
the 1006-slot atmosphere layout and `(depth, 6, 139)` synthesis layout; actual
versus partition-normalized populations.

**Produces.** Complete atomic states for the atmosphere and synthesis
representations, including a full electron-closure route and the explicitly
fixed-\(n_e\) synthesis route.

**Checks in the main text.** Two-level Boltzmann ratio; cool-neutral and
hot-ionized Saha limits; charge residual; population conservation; partition
identity; packed-slot/cube mapping; depth-parallel atomic EOS parity.

**Original schematic.** Level ladder → partition sum → ion ladder → charge
closure; the two exact population layouts.

**Close and handoff.** Atomic conservation closes only if atoms remain
unbound. In cool layers, molecules consume several elements simultaneously, so
Chapter 4 must solve a coupled rather than element-by-element equilibrium.

### Chapter 4 — Molecules and Coupled Equilibrium

**Question.** How can mass action and elemental conservation determine many
molecular populations without producing negative or path-dependent chemistry?

**Primary ownership.** Molecular formation constants; coupled conservation
residual and Jacobian with density-space unknowns; atmosphere unscaled
`solve`/`lstsq` updates with absolute-value recovery and a shared ordered
floor; synthesis density-column-scaled updates with its distinct positive
floor; exact sign-flip damping in both; synthesis-only log evaluation of molecular products;
pressure-scaled ordered depth continuation in both backends, with the
atmosphere specific-energy saved-row override taught at its lifecycle boundary; atmosphere
170-species and synthesis 190-species catalogs; Numba/LAPACK atmosphere solve
versus one-depth-at-a-time Torch `jacrev` synthesis solve; synthesis `vmap`
only for final post-depth-loop molecular-density evaluation; molecular
raw and partition-normalized molecular populations, the synthetic
axis-index-5 public lane, the exact synthesis cold-chain/abundance-floor and
hard-coded molecular-mass policies, structured-builder reuse, and molecular
internal energy, including the public-true versus lower-level-false tracking
default boundary.

**Produces.** Checked warm and cool molecular states that reuse Chapter 3's
atomic definitions and layouts while recomputing the coupled numerical atomic,
electron, nuclei, mass, and molecular state.

**Checks in the main text.** One-reaction mass-action limit; elemental
conservation; residual decrease; exact damping and positivity-floor branches;
warm dissociation and cool formation trends; raw-versus-normalized gate
behavior; ordered depth-continuation trajectory; deliberate
reset-versus-continuation comparison; energy-mode saved-row override;
molecule-enabled full/fixed/disabled-route claim table; global H2 bridge
selection; explicit direct-solver dtype; exact structured-builder edge-grid
dependency and molecular-array reuse.

**Original schematic.** Coupled element budget network; sequential depth
continuation in both backends; residual → Jacobian → density update, with the
synthesis log-product evaluation and post-depth-loop `vmap` placed at their
actual boundaries, including iteration exhaustion.

**Close and handoff.** We now know which absorbers and scatterers exist, but not
how strongly they interact with light. Chapter 5 converts these populations
into continuous absorption and scattering.

### Chapter 5 — Continuous Opacity and Scattering

**Question.** Which processes create the smooth background against which
spectral lines are seen, and how are their differently tabulated cross-sections
assembled on an exact wavelength grid?

**Primary ownership.** H\(^{-}\), H I, H\(_2^+\), He\(^{-}\), He I, He II;
C/Mg/Al/Si/Fe and light/hot metal continua; H/He/H\(_2\) Rayleigh and electron
scattering; stimulated emission; number-to-mass opacity; edge triplets,
interpolation, and component budgets. CH/OH/H\(_2\) collision-induced terms
are identified as atmosphere-only where that is the pinned boundary, not
claimed as standard synthesis terms.

**Produces.** Nonnegative continuum absorption and scattering slabs in the
exact atmosphere and synthesis layouts. `FrequencyInvariants` may be inspected
as an available object but is not passed by the standard synthesis pipeline.

**Checks in the main text.** Per-process limits; non-negativity; interpolation
at source points; number-to-mass units; component and total slab parity.

**Original schematic.** Population → cross-section → mass opacity; absorption
versus redirection; edge-triplet interpolation.

**Close and handoff.** The continuum explains broad escape windows but not the
narrow forest in the opening spectrum. Chapter 6 builds one line carefully
enough that thousands of lines can later reuse the same physics.

### Chapter 6 — One Spectral Line

**Question.** How do a bound transition, thermal motion, microturbulence, and
collisions turn one rest wavelength into a normalized absorption profile?

**Primary ownership.** Bound-level population; oscillator strength;
application of the Chapter 5 stimulated-emission factor to line strength;
thermal and microturbulent Doppler width; natural, Stark,
and van der Waals damping; Voigt/Harris/FASTEX profile; one-depth and
all-depth opacity.

**Produces.** One checked line-opacity slab and the exact ordinary-profile
building blocks used by Chapters 7–8.

**Checks in the main text.** Profile normalization; weak/strong damping limits;
mass/temperature/microturbulence trends; one-line component parity; float32
deposit effect measured rather than hidden.

**Original schematic.** Rest transition broadened into Gaussian core and
Lorentzian wings.

**Close and handoff.** A correct line profile does not say which catalog
records belong in a window or how overlapping contributions are deposited
without races. Chapter 7 builds the atomic forest and its non-ordinary
branches.

### Chapter 7 — Atomic Line Forests and Special Profiles

**Question.** How can millions of heterogeneous atomic records be decoded,
selected, routed, and accumulated without losing source semantics or changing
borderline decisions?

**Primary ownership.** Catalog decoding and corrections; exact selection
inequality; host-float64 discrete choices; context and line mapping; narrow and
wide deposits; scatter-add/chunk/private-reduction behavior; ordinary type-0
and type-3 records through the ordinary LTE branch; type-1 autoionizing lines;
type-2 COR parsed-but-unwired boundary; helium families; hydrogen/deuterium
fine structure and broadening; series merging and pseudo-continuum. In the
atmosphere lane this chapter also owns the family-independent
`SelectedLineCatalog` representation and common deposit kernel. It hands the
diatomic, TiO, water, and H3+ source-specific selection rules to Chapter 8.

`do_metal=True` also invokes the autoionizing accumulator. This coupled branch
behavior must be shown, not silently regularized. H/D Lyman support limits are
stated exactly.

**Produces.** Checked ordinary and special atomic line slabs and source-faithful
catalog routes.

**Checks in the main text.** Hand-selected record; catalog order/hash; scalar
versus chunk deposits; no-race reduction; special-profile limits; branch call
trace.

**Original schematic.** Catalog routing tree; sparse center/wing deposits;
ordinary versus special-profile paths.

**Density gate.** Keep one chapter with two visible movements by default.
Split into Chapters 7 and 8 only if the draft exceeds 18 substantial visible
code cells, cannot fit one 90-minute lecture/lab, or cannot retain separate
ordinary-forest and H/He/special-profile parity suites without rushing. A split
renumbers later chapters but loses no content.

**Close and handoff.** Atomic catalogs are no longer the whole opacity source:
cool spectra contain molecular bands with different source formats and
compilation constraints. Chapter 8 owns those formats and their runtime
boundary.

### Chapter 8 — Molecular Bands and Source Compilation

**Question.** How can large, differently encoded molecular line sources become
the same checked opacity contribution without pretending every available
compiler is wired into production?

**Primary ownership.** The exact two-lane family boundary: atmosphere
diatomic/TiO/water/H3+ source readers, family corrections, and selectors;
synthesis text-band, TiO, and H\(_2\)O formats; manifest-ordered compilation;
scalar and serial cached `njit` compiler; streaming and pair chunking;
synthesis molecular center/wing deposits; molecule-population mapping.

**Exact boundary.** The standard high-level atmosphere workflow supplies
diatomic, TiO, and water source arrays. Its H\(_3^+\) selector/deposit path is
runtime-capable only when an explicit `h3plus_lines_path` is supplied; the
standard `source_line_paths()` set supplies no H\(_3^+\) file. In synthesis,
text bands and TiO run by default when `molecular_lines=True`; the H\(_2\)O
compiler exists, but the pinned standard pipeline does not invoke it. The
synthesis source compiler and pipeline do not supply H\(_3^+\); a generic
species-mass entry does not establish runtime support. The serial compiler is
not rewritten with `prange`; there is no `torch.compile` path.

**Produces.** Checked molecular opacity slabs, compiled-source cache products,
and a machine-readable feature-status matrix.

**Checks in the main text.** Scalar/compiled catalog equality; source identity;
chunk invariance; text/TiO parity; explicit test that standard runtime omits
H\(_2\)O.

**Original schematic.** Three source formats through a common compiled record;
compiler-only versus runtime-enabled lanes.

**Frozen two-lane family and lifecycle matrix.**

| Family or lifecycle | Atmosphere lane | Synthesis lane | Ownership after this chapter |
| --- | --- | --- | --- |
| ordinary atomic | standard predicted/observed/high-excitation sources; common selected runtime deposit | standard runtime regardless of `molecular_lines` | Ch. 7 source routes/common deposit; Ch. 11 composition |
| diatomic / text bands | standard `diatomic_lines_path`; family correction then common selected deposit | manifest text compiler and runtime deposit when default `molecular_lines=True` | Ch. 8 family behavior; Ch. 11 atmosphere composition |
| TiO | standard `titanium_oxide_lines_path`; selected runtime deposit | Schwenke compiler and runtime deposit when default `molecular_lines=True` | Ch. 8; Ch. 11 atmosphere composition |
| water / H\(_2\)O | standard `water_lines_path`; dedicated selector then common selected runtime deposit | `compile_h2o_partridge` is compiler-only; `_compile_molecular` omits it | Ch. 8 boundary; Ch. 11 active atmosphere composition |
| H\(_3^+\) | explicit-path opt-in selector/runtime; no default file | no standard source compiler or pipeline wiring | Ch. 8 boundary; optional Ch. 11 composition |
| selected/detailed-object reuse | first-pass common selected-catalog generation/load and detailed-catalog load, then the same objects on later iterations | reusable atomic and enabled-molecular `WindowInvariants` keyed by exact physical inputs | Ch. 11 owns atmosphere reuse; Ch. 10 owns synthesis cache composition |

**Close and handoff.** We can now assemble extinction and emissivity at every
depth and wavelength, but still do not know what reaches the surface. Chapter 9
solves the ordered transfer problem.

### Chapter 9 — Radiative Transfer with Scattering

**Question.** Given depth-dependent extinction, thermal emission, and
scattering, what total and continuum flux actually emerge?

**Primary ownership.** Variable-depth optical-depth integration; formal
absorption-only solution; parabolic integration; Eddington–Barbier and
saturation limits; 51-point transfer grid; thermal/scattering source;
backward Gauss–Seidel iteration; fixed `DEFAULT_SWEEPS=8`; saturated-core
fallback; stacked total/continuum solution; `H_nu` → \(F_\nu\) →
\(F_\lambda\) conversion.

**Produces.** Native total and continuum `H_nu`, public \(F_\lambda\) semantics,
and normalized flux, still on the selected device.

**Checks in the main text.** Constant-source and zero-scattering limits;
hand-computed Gauss–Seidel update; positivity through eight sweeps; scalar row
versus batch; saturated/non-saturated paths; total equals continuum when line
opacity is zero; factor-\(4\pi\) and Jacobian tests.

**Original schematic.** Ordered depth sweep; thermal plus scattered source;
total/continuum stacked solve.

**Close and handoff.** The transfer operator is correct for prepared arrays,
but a usable spectrum needs reusable window state, exact component order,
bounded memory, cache identity, and one controlled host crossing. Chapter 10
composes that architecture.

### Chapter 10 — GPU Synthesis from a Structured Atmosphere

**Question.** How can one structured atmosphere and reusable window data remain
on CUDA, MPS, or CPU long enough to produce a complete native spectrum with
the pinned precision and cache behavior?

**Primary ownership.** `WindowInvariants`; star-dependent synthesis state;
device/dtype policy; precision islands; `WINDOW_CONTEXT_SAMPLES=16`;
`METAL_CHUNK=40000`, `CHUNK_LINES=500000`, `PAIR_CHUNK=200000`; process and
persistent caches; prewarm identity; exact pipeline order; device-side crop;
one final host-copy block; exact `SpectrumResult` and `Spectrum`.

**Exact composition order.**

```text
continuum absorption/scattering
→ shared float32 line slab
→ ordinary/PRD-routed LTE metal + autoionizing
→ helium
→ hydrogen/deuterium
→ text-band + TiO molecular
→ LTE line source and zero standard line scattering
→ stacked total/continuum transfer
→ context crop
→ one final host copy
→ public F_lambda per nm
```

**Produces.** A validated schema-v4/column builder path and exact host-float64
`Spectrum` from a supplied structured atmosphere.

**Checks in the main text.** Cache-key changes; cold/warm identity; corrupt
cache rebuild; no star-dependent invariant retention; no dense `(D,L,W)`
allocation; no early spectral host crossing; four-regime same-atmosphere
parity for wavelength, total, continuum, and normalized flux.

**Original schematic.** Window invariants versus one-star state; precision and
host/device map; context-compute-crop path.

**Close and handoff.** Synthesis is now verified for a supplied atmosphere, but
the book has not yet constructed a physically closed one. Chapter 11 starts
with a validated seed and builds the blanketed opacity state needed by one
physical atmosphere pass.

### Chapter 11 — Starting and Blanketing an Atmosphere

**Question.** What exact state can safely enter a physical atmosphere pass, and
how does its composition become the 30,000-frequency blanketing opacity?

**Primary ownership.** `ModelAtmosphere`, `AtmosphereConfig`, `RunSetup`;
validation; exact standard grid helper; hydrostatic next-pass pressure;
microturbulence initialization; fixed-column format/parse as quantization;
scalar remap behavior; `AtmospherePopulationState`; `OpacityState`;
30,000-point sampling grid and weights; continuum reference threshold;
first-pass catalog selection and later object reuse.

The standard family set composed here is ordinary atomic plus diatomic, TiO,
and water. H\(_3^+\) enters only when the caller supplied its explicit path.
Chapter 11 must demonstrate that the combined `SelectedLineCatalog` is created
or loaded on the first opacity pass and reused on later iterations. It does not
rederive Chapter 7's common deposit or Chapter 8's family-specific selectors.

**Produces.** A validated pass setup, current population state, complete
depth-major opacity-sampling state, and reusable selected/detailed catalog
handles. It does not correct or converge the atmosphere.

**Checks in the main text.** Bad-seed gallery; hydrostatic residual;
quantization idempotence; constant/monotone remap; grid/weight identity;
continuum/line components; unchanged catalog object identity with changed
atmosphere-dependent opacity.

**Original schematic.** Seed validation and quantization boundary feeding
population, continuum, selection, and blanketed opacity.

**Close and handoff.** The pass now contains opacity but not the integrated
radiative force, pressure, heating, or convective response. Chapter 12 reduces
the frequency field into those physical columns.

### Chapter 12 — Radiation, Thermodynamics, and Convection

**Question.** Which frequency-integrated radiation quantities and EOS
derivatives determine radiative support and the flux that convection must
carry?

**Primary ownership.** `TransferAccumulation`; per-pass reset versus persistent
correction/lookup state; chunk-private frequency accumulation and fixed-order
reduction; Rosseland opacity/depth; `RadiativePressureState`; heating and
lambda accumulators; surface radiation-pressure constant; four full EOS
perturbations; snapshot/restore; `ConvectionFiniteDifferenceSamples`;
`ConvectionResult`; disabled-convection diagnostics.

`finalize_transfer_state` is one exact combined source routine. Chapter 12
explains its radiation/convection prefix and returns the exact
`IterationFinalization`; it does not invent a partial-finalization class.

**Produces.** Named integrated radiation, force/pressure, thermodynamic, and
convection records, with the correction result present but deferred to Chapter
13 for explanation.

**Checks in the main text.** One-frequency scalar versus parallel
contribution; Rosseland and force identities; one/many chunk agreement;
reduction-order last-bit demonstration; exception-safe snapshot restoration;
finite-difference convergence; atomic and molecular convection branches.

**Original schematic.** Frequency fan into private accumulators and fixed
reduction; four perturbed EOS states into convection.

**Close and handoff.** The solver can measure imbalance and material response,
but those numbers do not themselves create a safe new structure. Chapter 13
orders the correction, remap, carried state, and stopping logic.

### Chapter 13 — Correction and the Full Numba Iteration

**Question.** How do flux/heating imbalance and convection become a stable
temperature/column update, and when may repeated passes be called structurally
converged?

**Primary ownership.** Mode-3 `apply_temperature_correction`; exact clamp,
damping, smoothing, monotonicity, and pressure-response order;
`TemperatureCorrectionResult`; complete `IterationRemap`; the full
`run_atmosphere_model` state machine; persistent/carry/reset classification;
convergence norms, minimum passes, consecutive count; Numba cache/prewarm;
terminal quantization; converged-only schema-v4 product with rebuilt final
populations.

**Non-negotiable interior edge.**

```text
correction → complete remap → next pass
```

There is no fixed-column quantization between passes. Seed and terminal product
boundaries are the only normal quantization boundaries.

**Produces.** Terminal quantized `AtmosphereRunResult`, structural convergence
status, exact diagnostics, optional debug state, and a schema-v4 product only
when structurally converged.

**Checks in the main text.** Each correction term; clamp/damping branches;
positive monotone corrected state; no interior quantizer; two-pass call trace;
catalog reuse; carried correction/lookup state; one terminal round trip;
convergence counter sequences; unconverged product suppression; stale-state
sentinel proving final population rebuild.

**Original schematic.** Exact 15-step pass orbit, with carried and reset state
visually distinct and a terminal double-line quantization/product gate.

**Close and handoff.** An explicit seed can now reach a declared structural
fixed point, but some labels begin outside a reliable basin. Chapter 14 learns
where to start without changing which physical solver decides the answer.

### Chapter 14 — Learned Initializers and Mandatory Physical Closure

**Question.** Can a learned prediction place the solver inside its convergence
basin for ordinary, CNO-varied, and direct-abundance mixtures without being
mistaken for the fixed point?

**Primary ownership.** Fixed-point basin and restart trajectories; six exact
profile transforms; `(80,6)` → 480 C-order coordinates → 160 PCA
coefficients; Torch `float32` network inference and one CPU NumPy `float64`
decode boundary; ordinary five-label and CNO8 routing; support projection only
for initializer queries; deterministic candidates; direct 81/84/97 abundance
layouts; 0.01-dex lattice; set encoder; sentinel inheritance; immutable mixture
and deck hashes; experimental direct safety object; mandatory exact closure.

**Produces.** Exact deterministic initializer-label tuples, exact
`(ModelAtmosphere, deck_text)` ordinary/CNO warm starts, or the exact
`DirectAbundanceOptimizerSurrogate`/deck boundary. A physical direct result
exists only through `run_direct_abundance_atmosphere`.

**Checks in the main text.** Six transform round trips; PCA independent of
network; reshape sentinel; deterministic inference/candidate order; requested
labels unchanged after initializer-only projection; CNO routing; direct
lattice/sentinels/hash; one-element 0.01-dex perturbation; exact restart and
closure.

**Original schematic.** Several encoders feeding one shared profile decoder,
then a predicted point crossing the quantization gate into the Chapter 13
physical orbit.

**Close and handoff.** We can now propose honest starting atmospheres and solve
them physically, but users still need to choose between a fast exploratory
route and a verified product route. Chapter 15 makes that distinction
executable and audits the final spectrum.

### Chapter 15 — From Stellar Labels to a Verified Spectrum

**Question.** What exact sequence turns stellar labels into either an honestly
marked exploratory spectrum or a physically converged, independently accepted
spectrum?

**Primary ownership.** Workflow choice and composition only:
`initialize_atmosphere_from_labels`, `synthesize_from_labels`,
`InitializedAtmosphere`, `LabelSpectrum`, `ForwardTimings`;
`solve_structured_atmosphere` → validated schema-v4 `Path` → `synthesize` →
`Spectrum`; four-regime acceptance; limitations and reproducibility model
card.

**Two honest lanes.**

```text
exploratory:
labels → initialized atmosphere (converged=False, closure_required=True)
       → fixed-ne bridge → Chapter 10 synthesis → LabelSpectrum

verified:
labels → initialized seed(s) → Chapter 13 exact solve
       → terminal quantization → rebuilt schema v4
       → Chapter 10 synthesis → Spectrum + editorial acceptance table
```

The acceptance table is not a new public class. Structural, flux,
hydrostatic, optical-depth, schema, trajectory, and four spectral-output gates
remain independent.

**Produces.** Exact exploratory objects and compact schema/synthesis evidence
for hot, solar, giant, and cool molecule-rich requests. The retained full
physical product is the solar case: an exact converged 80-layer atmosphere,
validated schema-v4 arrays, and a narrow `Spectrum`. The other three requests
remain compact integration evidence until their full physical trajectories
are run and accepted.

**Checks in the main text.** Immutable exploratory safety flags; converged-only
promotion; wavelength identity before flux comparison; normalized-flux
identity; independent injected gate failures; exact solar structural
convergence; and three-way staged/repeat/pinned array parity. Atmosphere
archives are byte-identical and the four physical spectrum arrays are
bitwise-identical; timings are excluded. Flux-error, hydrostatic-residual, and
separately retained standard-optical-grid acceptance remain explicitly
unevaluated.

**Original schematic.** Fast exploratory lane versus lower verified lane,
with missing-catalog and failed-convergence branches unable to enter the
verified product boundary.

**Close.** Return to the opening spectrum and enumerate exactly what the reader
can now compute, verify, reproduce, and deliberately not claim. Link to
appendices for data installation, complete API/CLI spelling, tolerance
profiles, and reproducibility—not to a sixteenth physics chapter.

## Primary ownership for duplication-prone concepts

| Concept | Derived/taught once | Later treatment |
| --- | --- | --- |
| Planck radiation and spectral Jacobian | Ch. 1 | invoked; flux conversion completed in Ch. 9 |
| Numba/Torch syntax and honest timing | Ch. 2 | only real kernel-specific consequences later |
| abundance notation and data roles | Ch. 2 | consumed exactly |
| actual vs partition-normalized populations | Ch. 3 | validated/mapped, never redefined |
| electron closure | Ch. 3 | Ch. 4 couples molecules; Ch. 10 fixed-\(n_e\) only |
| molecular equilibrium | Ch. 4 | opacity chapters consume populations |
| continuum process physics | Ch. 5 | Ch. 10/11 assemble, not rederive |
| ordinary line profile | Ch. 6 | Ch. 7/8 deposit it |
| atomic routes and family-independent selected-record deposition | Ch. 7 | Ch. 8 adds molecular family selectors; Ch. 11 owns only composition/reuse |
| molecular family selection and source compilation | Ch. 8 | Ch. 11 composes the active atmosphere families; Ch. 10 composes synthesis runtime/cache |
| formal transfer/scattering | Ch. 9 | Ch. 12 owns integrated accumulator lifecycle |
| synthesis caches/device composition | Ch. 10 | Ch. 15 calls once |
| fixed-column quantization operator | Ch. 11 | Ch. 13/14 decide exact boundaries |
| convection | Ch. 12 | Ch. 13 consumes |
| correction/remap/convergence | Ch. 13 | Ch. 15 independently accepts |
| PCA/profile decoder and initializer support | Ch. 14 | Ch. 15 routes only |
| exploratory vs verified workflow | Ch. 15 | nowhere earlier as a full workflow |

## Forward-reference policy

A forward reference is allowed only when all four conditions hold:

1. the present chapter needs the output but cannot yet derive it;
2. the object is named an **integration fixture** or deferred component;
3. its reads, writes, shape, unit, dtype, device, and provenance are stated;
4. the chapter that later owns it is named once.

The current chapter may not inspect the future implementation, teach its API,
or imply that the fixture was built locally. Learned-origin seeds in Chapter
11 are simply supplied `ModelAtmosphere` fixtures. Chapter 10's supplied
structured atmosphere does not imply atmosphere convergence. Chapter 14 does
not require synthesis to validate an initializer.

## Chapter close template

Every chapter ends with:

1. `## N.N Chapter summary`;
2. the opening question answered at the level actually computed;
3. exact outputs now available;
4. important missing claim or dependency;
5. `### Next: <causal need>`;
6. one direct `/reader.html?ch=N+1` link.

The summary contains no new terminology, API, field, or scientific claim.
Chapter 15 replaces the next-chapter link with a concise “what you can now
build” close and appendix links.

## Original schematic and plot gate

Each schematic must:

- solve a genuine spatial, state-flow, hierarchy, or branching problem;
- be generated from a textbook-owned prompt/specification;
- use the restrained website-inspired scientific-notebook aesthetic without
  copying a website figure;
- have short labels, an informative caption, alt text, and a linear prose
  reading;
- pass a scientific-content review before publication.

Each quantitative plot must:

- make one primary physical or numerical claim;
- usually use one panel;
- use the shared professional paper-inspired typography and palette;
- label units and exact quantity;
- avoid ornamental legends, dense multi-panel dashboards, or default styling;
- be interpreted from its actual values in the next paragraph.

## Verification ladder and four-regime matrix

The book's tests grow in this order:

1. analytic identities and limiting cases;
2. exact small-kernel source/AST parity;
3. shape/unit/dtype/device and schema validation;
4. component and stage parity;
5. one complete atmosphere pass;
6. multi-pass carried-state trajectory;
7. structural convergence and terminal quantization;
8. same-atmosphere synthesis;
9. initialized-label exploratory synthesis;
10. independently converged labels-to-spectrum synthesis;
11. cache, backend, and thread-profile reproducibility.

The high-level ladder is run for:

- hot, mostly atomic atmosphere;
- solar-like atmosphere;
- low-gravity giant;
- cool molecule-rich atmosphere.

At every spectral gate compare all four:

- `wavelength_nm`;
- `flux_total`;
- `flux_continuum`;
- `normalized_flux`.

Exact equality is claimed only where measured. CPU/CUDA, MPS float32, cache,
and alternate-thread comparisons use separately measured, machine-readable
tolerance profiles.

## Draft and integration cadence

No chapter is accepted after one writing pass. Each wave follows:

1. source/API/coverage inventory;
2. causal outline and chapter contract;
3. canonical code and focused tests;
4. first executable narrative;
5. paragraph and code-output audit;
6. exact-name, axis, unit, dtype, device, and data-role audit;
7. neighboring-chapter continuity and redundancy audit;
8. rendered plot/schematic/pacing review;
9. parity and failure-injection pass;
10. whole-book summaries/dependency/coverage zoom-out.

Subagents may draft or audit bounded sections, but the central integrator owns
this contract, shared notation, chapter handoffs, acceptance, and deletion of
redundancy.

## Density and chapter-count decision

The target remains fifteen chapters. Internal movements and web anchors are
preferred to a split. A sixteenth chapter is justified only when one of these
measured conditions holds:

- a chapter exceeds 18 substantial visible code cells after honest splitting;
- it cannot fit one 90-minute lecture/lab without skipping a required
  derivation or interpretation;
- two independent parity suites cannot remain visible and understandable;
- the rendered chapter becomes too long for a reader to retain its central
  question.

The cleanest optional split is Chapter 7 between ordinary atomic forests and
H/He/special profiles. Chapters 11–14 may use multiple web routes under one
chapter navigation item. No split or merge changes the coverage ledger.
