# Design brief for Chapters 11–15: the physical atmosphere and complete workflow

> **Integration note.** `design/global_chapter_contracts.md` governs the final
> reader-facing sequence. It supersedes any implied end-of-chapter exercise
> block or reuse of official website figures: useful variations are taught
> inline, and every textbook schematic is an original composition.

## Scope

This brief defines the last five reader-facing chapters of the fifteen-chapter course. It preserves all atmosphere and initializer material from internal completeness units 21–29, but those unit numbers are editorial tags only; they must not appear as additional chapters in the reader-facing book.

Implementation identity is pinned to source revision
`9c44001feae40b85146630499e6f8a5fed42e5af`. Exact names and behavior below
refer to that revision and the accompanying paper source. A later source
change requires rerunning the notation/API audit rather than silently updating
the prose.

| Reader-facing chapter | Internal completeness coverage |
|---|---|
| 11. Starting and Blanketing an Atmosphere | seed/hydrostatic/quantization/remap unit; opacity-sampling/blanketing unit |
| 12. Radiation, Thermodynamics, and Convection | integrated-transfer/Rosseland/radiative-force unit; internal-energy/convection unit |
| 13. Correction and the Full Numba Iteration | correction/remap unit; full iteration/convergence/cache/failure unit |
| 14. Learned Initializers and Mandatory Physical Closure | fixed-point/PCA ordinary initializer unit; direct-abundance unit |
| 15. From Stellar Labels to a Verified Spectrum | final workflow/capstone unit |

The reader has completed Chapters 1–10 and therefore already has:

- radiative quantities, the Planck function, optical depth, the grey atmosphere, the standard 80-layer Rosseland grid, hydrostatic balance, Rosseland means, and fixed-point vocabulary;
- scalar, NumPy, Numba `njit`, `prange`, Torch, caching, dtype, shape, and device vocabulary;
- abundance parsing, immutable data manifests, atomic and molecular equilibrium, packed atmosphere populations, and specific internal energy;
- continuous opacity, line opacity, catalogs, conservative line selection, sparse deposits, special atomic profiles, and molecular bands;
- formal transfer with scattering;
- schema-v4 atmosphere files and a complete CPU/CUDA/MPS spectrum calculation from a supplied structured atmosphere.

Chapters 11–15 must use those components rather than rederive them. Their purpose is to show how the components form a nonlinear CPU atmosphere solver, how learned models place a seed inside its convergence basin, and how a converged atmosphere is handed to the existing synthesis pipeline.

The governing teaching rule is:

> Each major section begins with a physical question, introduces one typed state transition, shows the smallest code that makes the transition real, and ends with a physical check plus a parity gate.

The governing numerical rule is:

> Values alone do not define the algorithm. Depth/frequency order, dtype, update order, carried state, catalog reuse, reduction order, remapping, and quantization boundaries are part of the method.

All physical-atmosphere arrays in Chapters 11–13 are CPU NumPy `float64` unless stated otherwise. Depth always runs outermost to innermost, but its exact axis is artifact-specific: opacity slabs are depth-major `(L, Nnu)`, whereas transfer precomputes such as `planck_all` are frequency-major `(Nnu, L)`. The production grid has `L = 80`,

\[
\tau_{\mathrm R,i}=10^{-6.875+0.125i},\qquad i=0,\ldots,79.
\]

Torch is used in Chapter 14 for initializer inference, normally on CPU in `float32`. Decoded coefficients cross to CPU NumPy exactly once, after which PCA inversion, physical transforms, seed validation, the exact atmosphere solve, and final quantization use NumPy `float64`. Chapter 15 hands the converged schema-v4 state to the device policy already established in Chapter 10.

## Non-negotiable atmosphere state machine

Every explanation, code path, schematic, trace, test, and capstone must agree with this order:

1. Validate and resolve one initial `ModelAtmosphere`.
2. Initialize loop-carried state:
   - no previous remap;
   - no previous Rosseland lookup table;
   - the seed surface-radiation-pressure constant;
   - one persistent temperature-correction state;
   - no selected-line catalog;
   - no detailed-transition catalog;
   - the seed molecular thermal-energy reference.
3. For physical pass \(n=1,\ldots,N\):
   1. On pass 1, copy the validated seed. On later passes:
      - consume the preceding pass's **unquantized remapped** atmosphere;
      - if pressure iteration is enabled, recompute gas pressure hydrostatically using the preceding remapped integrated radiation pressure and turbulent pressure;
      - copy this state into the new pass.
   2. Build a pass-local setup with the carried surface-radiation-pressure constant.
   3. Recompute electron/molecular closure, populations, specific internal energy, Doppler widths, and selection-strength factors.
   4. Recompute continuum opacity and the 30,000-point opacity-sampling slab.
   5. On the first pass, select/load the enabled line catalogs. On subsequent passes, reuse the exact catalog objects and recompute only their atmosphere-dependent opacity.
   6. Reset per-pass radiative accumulators while preserving the explicitly persistent correction history and Rosseland lookup.
   7. Accumulate transfer in frequency chunks with private arrays and fixed-order reduction.
   8. Finalize Rosseland opacity and Rosseland optical depth.
   9. Finalize radiative acceleration, radiation pressure, and the new surface-radiation-pressure constant.
   10. Ingest current \((T,P_{\rm gas},\kappa_{\rm R})\) into the correction lookup.
   11. When convection is enabled, compute four perturbed EOS states and evaluate convection. Otherwise, compute the disabled-convection diagnostics without pretending to take the production branch.
   12. Apply temperature and column-mass correction on the current/native Rosseland grid.
   13. Remap the complete corrected state to the standard Rosseland grid.
   14. Carry the Rosseland lookup and surface-radiation-pressure constant.
   15. Compare the current pass input temperature with the remapped output, update the consecutive structural-convergence state, and record flux diagnostics.
4. Stop when the declared structural criterion has passed after the minimum iteration count for the required number of consecutive passes, or when the pass budget is exhausted.
5. Apply fixed-column format/parse quantization **once** to the terminal remapped atmosphere.
6. Return the terminal atmosphere, `converged` status, iteration count, and diagnostics.
7. Write a schema-v4 physical product only when structural convergence is true. Rebuild synthesis populations from the final quantized columns; never serialize the stale live population state from before the final correction/remap.

There are only two normal fixed-column quantization boundaries:

- an initialized warm start is format/parsed in memory before its first exact pass;
- the terminal remapped atmosphere is format/parsed before it becomes the returned atmosphere.

There is no format/parse round trip between physical passes. The interior edge is:

```text
correction → complete remap → next pass
```

not:

```text
correction → remap → quantize → next pass
```

## Exact notation and API identity

The textbook must reproduce the pinned implementation, not introduce a cleaned-up parallel API. Exact source class, function, argument, field, constant, branch, and output names are typeset as code. A mathematical symbol may be used for a derivation only when it is mapped immediately to the exact stored name.

### Physical notation mapped to stored fields

| Paper/source notation | Exact stored name | Shape, units, or convention |
|---|---|---|
| \(m\) | `ModelAtmosphere.column_mass` | `(L,)`, g cm\(^{-2}\), outer-to-inner and strictly increasing |
| \(T\) | `ModelAtmosphere.temperature` | `(L,)`, K |
| \(P_{\rm gas}\) | `ModelAtmosphere.gas_pressure` and `AtmosphereRuntimeState.gas_pressure` | `(L,)`, dyn cm\(^{-2}\) |
| \(n_e\) | `electron_density` | `(L,)`, cm\(^{-3}\) |
| \(\kappa_{\rm R}\) | `rosseland_opacity` | `(L,)`, cm\(^2\) g\(^{-1}\) |
| \(g_{\rm rad}\) | `radiative_acceleration` | `(L,)`, cm s\(^{-2}\) |
| \(\xi\) | `microturbulence` | `(L,)`, cm s\(^{-1}\); public label arguments use `microturbulence_km_s` |
| \(\tau_{\rm R}\) | `standard_rosseland_optical_depth` or `rosseland_optical_depth` | `(L,)`; source uses both standard/remapped and current/native grids |
| \(\chi_\nu\) | current total mass extinction assembled inside transfer | atmosphere opacity slabs are `(L, Nnu)`; depth is axis 0 |
| \(F_\nu\) | source/paper physical flux | transfer code accumulates the Eddington flux \(H_\nu=F_\nu/(4\pi)\) in fields named `monochromatic_eddington_flux` and `integrated_eddington_flux` |
| \(F_{\rm rad}\) | paper physical radiative flux | source target is `target_integrated_eddington_flux = 5.6697e-5 / 12.5664 * effective_temperature**4` |
| \(F_{\rm conv}\) | paper physical convective flux | source field `convective_flux`; correction compares it in the same integrated-Eddington-flux convention used by the solver |
| \(P_{\rm rad}\) | paper radiation pressure | source distinguishes `integrated_radiation_pressure`, `absolute_radiation_pressure`, and scalar `surface_radiation_pressure_constant` |
| \(\mathbf{x}^{(n+1)}=\mathcal G(\mathbf{x}^{(n)})\) | one complete pass of `run_atmosphere_model` | augmented state includes remap, lookup, previous correction, catalog handles, surface constant, and controls |
| \(\epsilon_T\) | `deep_layer_relative_temperature_change` | maximum over zero-based `[39:L-5]`, denominator is the new/remapped temperature |

The paper's supported hydrostatic form is

\[
P_{\rm gas}(m)+P_{\rm rad}(m)=gm+P_0.
\]

The exact implementation stores the surface piece separately. `integrate_hydrostatic_pressure` computes

```text
surface_gravity_cgs * column_mass
- integrated_radiation_pressure
- turbulent_pressure
- pressure_constant
```

and floors nonpositive values with a warning. In the production loop, `pressure_constant` is left at its default `0.0`; `surface_radiation_pressure_constant` is instead carried in `RunSetup` and used by the total-pressure/convection/correction path. The book must keep `integrated_radiation_pressure`, `absolute_radiation_pressure`, and `surface_radiation_pressure_constant` distinct.

The atmosphere opacity arrays are depth-major:

- `OpacityState.continuum_absorption`, `continuum_scattering`, and `continuum_source`: `float64[L, 30_000]`;
- `OpacityState.continuum_line_selection_threshold`: `float32[L, 344]`; active continuum computations use `Nref <= 343`, then the exact builder fills the complete 344-column table and duplicates column 342 into 343;
- `OpacityState.continuum_reference_wavelength_nm`: `float64[344]`;
- `OpacityState.wavelength_bin_edges`: the exact field name for the packed continuum wavelength indices, `int64[344]`;
- `LineOpacityState.line_mass_absorption_coefficient`: allocated as `float64[L, 30_000]`, then explicitly converted to contiguous `float32` at the transfer-kernel boundary;
- `AtmosphereRuntimeState.elemental_abundances_by_layer`: `(L, 99)`;
- `ion_stage_populations_by_packed_slot` and `partition_normalized_populations_by_packed_slot`: `(L, 1006)`;
- `fractional_doppler_widths` and `partition_normalized_population_over_mass_density_and_fractional_doppler_width`: `(L, 1006)`;
- `hydrogen_departure_coefficients`: `(L, 6)`;
- initializer coordinates: `(80, 6)`, flattened to 480 in C order.

### Generalized draft names that are forbidden in canonical code

These names appeared in an earlier planning draft. They are not present in the pinned implementation and must not become textbook package APIs.

| Removed generalized name | Exact implementation identity |
|---|---|
| `ValidatedIterationSeed` | `RunSetup` whose `.atmosphere` is a validated `ModelAtmosphere`; produced by `resolve_run_setup` |
| `BlanketedIterationInput` | no wrapper; use `OpacityState` |
| `OpacitySamplingState` | `OpacityState` |
| `RadiativePassState` | no dataclass; before finalization use `TransferAccumulation`, after the exact combined call use `IterationFinalization` |
| `RadiativeThermodynamicPass` | no dataclass; use `IterationFinalization` and its exact fields |
| `ThermodynamicSnapshot` | local explanatory list only; `compute_convection_finite_difference_samples` performs the snapshot/restore internally |
| `FrequencyContribution` / `PrivateTransferAccumulators` | private arrays inside `accumulate_transfer_range_parallel`; not public records |
| `DecodedAtmosphereSeed` | ordinary path returns `tuple[ModelAtmosphere, str]` from `emulator_warm_start_model` |
| `InitializerCandidateSet` | `tuple[dict[str, float] | None, ...]` from `deterministic_initializer_labels` |
| `DirectAbundanceInitializedState` | public safety object is `DirectAbundanceOptimizerSurrogate`; public physical route is `run_direct_abundance_atmosphere` |
| `SurrogateSpectrum` | `LabelSpectrum`; its `atmosphere_converged=False` and `atmosphere_closure_required=True` are fixed fields |
| `VerifiedStellarSpectrum` | no public wrapper; exact verified outputs are the `Path` from `solve_structured_atmosphere`, validation names from `validate_atmosphere_npz`, and `Spectrum` from `synthesize` |
| `VerificationReport` / `WorkflowPlan` / `TrialReport` | editorial tables assembled in the chapter, not runtime package types |
| `quantize_atmosphere_columns` | `parse_atmosphere_deck(format_atmosphere_deck(model), source="<roundtrip>")`; learned starts use `format_warm_start_deck` then `parse_atmosphere_deck` |
| `remap_depth_column` | `remap_to_grid` |
| `save_converged_schema_v4` | `save_product_structured_atmosphere` |
| `prewarm_atmosphere_kernels` | `prewarm` |

Pedagogical labels such as “seed boundary” or “radiative phase” may appear in prose/diagrams, but must be marked as local labels and immediately point to the exact object/function above.

### Chapter-by-chapter API audit

| Chapter | Exact source entry points taught | Exact returned types/products | Audit rule |
|---|---|---|---|
| 11 | `parse_atmosphere_deck`, `format_atmosphere_deck`, `resolve_run_setup`, `integrate_hydrostatic_pressure`, `prepare_population_state`, `prepare_opacity_state` | `ModelAtmosphere`, `RunSetup`, `AtmospherePopulationState`, `OpacityState` | no seed or opacity wrapper |
| 12 | `accumulate_transfer_state`, `finalize_transfer_state`, `compute_convection_finite_difference_samples`, `compute_convection`, `compute_disabled_convection_diagnostics` | `TransferAccumulation`, `IterationFinalization`, `ConvectionFiniteDifferenceSamples`, `ConvectionResult`, `DisabledConvectionDiagnostics` | no partial finalized-radiation dataclass |
| 13 | `apply_temperature_correction`, `remap_finalized_iteration_state`, `finalize_remapped_iteration`, `run_atmosphere_model`, `save_product_structured_atmosphere`, `prewarm` | `TemperatureCorrectionResult`, `IterationRemap`, `AtmosphereRunResult`, schema-v4 NPZ `Path`, prewarm `dict` | `converged` remains structural only |
| 14 | `deterministic_initializer_labels`, `emulator_warm_start_model`, `complete_direct_abundance_vector`, `build_direct_abundance_optimizer_surrogate`, `direct_abundance_warm_start_deck`, `run_direct_abundance_atmosphere` | label tuple, `(ModelAtmosphere, deck_text)`, `float64[97]`, `DirectAbundanceOptimizerSurrogate`, `str`, converged `AtmosphereRunResult` | decoded direct `ModelAtmosphere` stays private outside optimizer safety object |
| 15 | `initialize_atmosphere_from_labels`, `synthesize_from_labels`, `InitializedAtmosphere.save_npz`, `solve_structured_atmosphere`, `validate_atmosphere_npz`, `load_atmosphere_product_metadata`, `load_atmosphere_npz`, `synthesize` | `InitializedAtmosphere`, `LabelSpectrum`, initializer-marked name tuple, product `Path`, validated name tuple, optional metadata, atmosphere mapping, `Spectrum` | verification summary is editorial, not an invented package type |

## Exact artifact graph

```text
Chapter 11
ModelAtmosphere ──resolve_run_setup──► RunSetup
      │
      └──prepare_population_state──► AtmospherePopulationState
                                      │
                                      └──prepare_opacity_state──► OpacityState
                                                                      │
                                                                      ▼
Chapter 12
accumulate_transfer_state ──► TransferAccumulation
                                      │
                                      └──finalize_transfer_state──► IterationFinalization
                                                                      │
                                                                      ▼
Chapter 13
remap_finalized_iteration_state ──► IterationRemap
                                      │
                                      └──finalize_remapped_iteration──► AtmosphereRunResult
                                                                      │
                                                   save_product_structured_atmosphere
                                                                      ▼
                                                        schema-v4 NPZ `Path`
                                                                      ▲
                                                                      │ exact restart
Chapter 14
emulator_warm_start_model ────────────┘
run_direct_abundance_atmosphere ──────► AtmosphereRunResult
build_direct_abundance_optimizer_surrogate ──► DirectAbundanceOptimizerSurrogate
                                                                      │
                                                                      ▼
Chapter 15
initialize_atmosphere_from_labels ──► InitializedAtmosphere
synthesize_from_labels ─────────────► LabelSpectrum
solve_structured_atmosphere ────────► Path ──synthesize──► Spectrum
```

Chapter 13's exact solver accepts `AtmosphereConfig`, whose `AtmosphereInput.initial_atmosphere` is one explicit `ModelAtmosphere`. Chapter 14 supplies that model through the exact initializer functions. This keeps the physical solver independent of the learned model and removes a forward dependency from Chapter 13.

## Source-routine and field placement

A source file is not a chapter boundary. The book should place routines where their physical input becomes available and where their output first matters.

| Routine/state family | Expository owner | Later use |
|---|---|---|
| `ModelAtmosphere`, validation, run-control resolution | Ch. 11, Act I | all later chapters |
| Standard Rosseland-grid helper | Ch. 1 derives; Ch. 11 makes exact helper | Ch. 13 remap; Ch. 14 decode |
| Hydrostatic update with radiation/turbulent pressure | Ch. 1 derives; Ch. 11 implements pass boundary | Ch. 13 calls after pass 1 |
| Fixed-column format/parse operator | Ch. 11 | Ch. 13 terminal output; Ch. 14 seeds |
| General remap behavior | Ch. 9 establishes interpolation; Ch. 11 states carried-column contract | Ch. 13 performs complete remap |
| Atmosphere population wrapper | Ch. 11, Act II, composing Ch. 3–4 microphysics | Ch. 12 perturbations; Ch. 13 |
| Sampling grid, continuum threshold, catalog lifecycle | Ch. 11, Act II | Ch. 12 |
| Selection mathematics/sparse deposition | Ch. 7 | Ch. 11 teaches select-once/reuse |
| Formal transfer/scattering kernel | Ch. 9 | Ch. 12 teaches integrated accumulator lifecycle |
| Rosseland modes and radiative-pressure state | Ch. 12, Act I | Ch. 13 correction |
| Specific-energy formulae | Ch. 3–4 | Ch. 12, Act II uses four recomputed states |
| Finite-difference thermodynamics/convection | Ch. 12, Act II | Ch. 13 consumes |
| Temperature/column correction and full remap | Ch. 13, Act I | Ch. 13 loop |
| Pass loop, convergence, diagnostics, caches/prewarm | Ch. 13, Act II | Ch. 14–15 call |
| Structural convergence | Ch. 13 | Ch. 15 reports; never broaden meaning |
| Six transforms, PCA, ordinary/CNO initializer | Ch. 14, Acts I–II | Ch. 14 direct decoder; Ch. 15 |
| Direct abundance set encoder and safety policy | Ch. 14, Act III | Ch. 15 |
| Schema meanings and structured synthesis bridge | Ch. 2/10 | Ch. 13 validates/writes; Ch. 15 consumes |
| GPU synthesis implementation | Ch. 10 | Ch. 15 calls; never re-inlines |
| Flux/hydrostatic/schema/spectrum acceptance | Ch. 15 | capstone |

The critical ownership rules are:

- Chapter 11 teaches remap mechanics and the list of carried state conceptually; Chapter 13 alone implements the full post-correction remap.
- Chapter 11 teaches quantization as a numerical operator; Chapter 13 decides when terminal quantization happens.
- Chapter 7 owns the selection inequality; Chapter 11 owns first-pass selection and cross-pass object reuse.
- Chapter 9 owns formal transfer; Chapter 12 owns atmosphere-wide integrated accumulators.
- Chapter 12 owns convection; Chapter 13 consumes it without rederiving it.
- Chapter 13 owns structural convergence; Chapter 15 owns broader physical and spectral acceptance.
- Chapter 14 defines one shared PCA/profile decoder. Direct abundances add a new encoder, not a second atmosphere representation.
- Chapter 15 composes only; it contains no opacity, transfer, correction, PCA, or synthesis kernels.

## Pinned source-module coverage

Chapter authors must cite the exact module that owns each canonical excerpt.
This is a coverage map, not a proposed reorganization:

| Exact pinned module | Reader-facing owner and treatment |
|---|---|
| `payne_zero_atmosphere/config.py` | Ch. 11: exact input/output/config dataclasses and defaults |
| `payne_zero_atmosphere/atmosphere_io.py` | Ch. 11: `ModelAtmosphere`, fixed-column parse/format, metadata and abundance conventions |
| `payne_zero_atmosphere/run_setup.py`, `microturbulence.py` | Ch. 11: validation, standard grid, microturbulence initialization/prescription, and resolved run controls |
| `payne_zero_atmosphere/hydrostatic.py` | Ch. 11 derives and tests the next-pass gas-pressure update |
| `payne_zero_atmosphere/runner.py` | Ch. 11 owns population/opacity preparation; Ch. 12 owns transfer finalization/convection prefix; Ch. 13 owns remap and the exact loop |
| `payne_zero_atmosphere/continuum_opacity.py` | Ch. 5 owns kernels; Ch. 11 owns atmosphere-grid assembly and selection-reference state |
| `payne_zero_atmosphere/line_selection.py`, `line_catalog.py`, `line_opacity.py` | Chs. 6–8 own physics/kernels; Ch. 11 owns exact catalog lifecycle and atmosphere-state composition |
| `payne_zero_atmosphere/radiative_transfer.py` | Ch. 9 owns formal transfer/remap primitives; Chs. 11–13 import the exact functions without cloning them |
| `payne_zero_atmosphere/transfer_kernels.py` | Ch. 12: compiled frequency-range kernels, chunk-private arrays, and fixed reduction |
| `payne_zero_atmosphere/rosseland_mean.py`, `radiative_pressure.py` | Ch. 12: named reset/accumulate/finalize states and physical interpretation |
| `payne_zero_atmosphere/convection.py` | Ch. 12: finite-difference records, ideal-gas diagnostic path, and production mixing-length calculation |
| `payne_zero_atmosphere/temperature_correction.py` | Ch. 12 introduces persistent accumulators; Ch. 13 owns mode-3 correction and column-mass response |
| `payne_zero_atmosphere/convergence.py` | Ch. 13: exact structural norms and limit predicate |
| `payne_zero_atmosphere/_numba_cache.py`, `prewarm.py` | Ch. 13 plus operations appendix: cache resolution and representative branch compilation |
| `payne_zero_atmosphere/synthesis_bridge.py` | Ch. 13: converged-only population rebuild and schema-v4 handoff |
| `payne_zero_atmosphere/warm_start.py` | Ch. 14: ordinary/CNO routing, support, Torch/PCA decode, deck boundary, and deterministic candidates |
| `payne_zero_atmosphere/direct_abundance.py` | Ch. 14: 81/84/97 layouts, lattice, set encoder, immutable optimizer surrogate, and mandatory exact closure |
| `payne_zero_atmosphere/cli.py` | Ch. 15: exact `solve_structured_atmosphere` composition; complete CLI spelling remains in the appendix |
| `payne_zero_synthesis/api.py` | Ch. 15: label-driven exploratory objects and the exact synthesis entry points |
| `payne_zero_synthesis/atmosphere.py` | Chs. 2/10 own schema details; Ch. 15 validates/loads exact products and optional product metadata |
| `payne_zero_atmosphere/data_files.py`, `source_catalogs.py`, `install_runtime_data.py`, package `__init__.py`/`__main__.py` files | data/API/CLI appendices; chapters link to them at the point of first runtime dependency |

The lower-level EOS, population, Doppler, continuum, profile, molecular, and
high-resolution synthesis modules are already owned by Chapters 3–10. In
Chapters 11–15 they appear only as exact imported calls and state fields,
preventing both omission and redundant re-teaching.

## Website-aesthetic schematic plan

The online edition should feel like a modern observatory control notebook: precise, calm, and data-led rather than decorative. The atmosphere chapters are dense, so visual consistency is a cognitive tool.

### Visual system

- **Canvas:** warm off-white for prose and deep blue-black for code/interactive panels.
- **Accent colors with fixed meaning:**
  - violet: labels and initializer-only state;
  - blue: radiation and transfer;
  - amber: matter/EOS/opacity;
  - red-orange: correction and warnings;
  - green: validated output or passed gate;
  - grey: cached/carried state.
- **Typography:** a readable serif or humanist face for physics prose, a compact sans face for labels, and a high-legibility monospace face for code and shapes.
- **Array badges:** every code/output card may show compact badges such as `CPU`, `float64`, `(80, 30_000)`, `depth-major`, `outer → inner`.
- **State edges:** solid arrows mean recomputation, grey looped arrows mean carried state, dashed arrows mean optional/debug output, and double borders mark quantization/product gates.
- **Accessibility:** color is never the only discriminator; every edge has a label, diagrams have linear-text equivalents, and motion can be disabled.
- **Print fallback:** all schematics must remain meaningful in greyscale through line style and labels.

### Page rhythm

Each chapter should alternate:

1. physical question and one compact hero schematic;
2. a worked physical example;
3. derivation cards;
4. 10–30-line canonical code cells;
5. a state inspector showing names/shapes/units;
6. a physical plot;
7. a parity-gate panel;
8. an end-of-chapter consumes/produces card.

Long tables belong in collapsible “field atlas” panels on the website and remain expanded in downloadable/print editions. Failure cases use amber/red callouts with the exact failing contract, never vague “gotcha” boxes.

### Chapter hero schematics

**Chapter 11 — “Seed to blanketed pass input.”**

- Left: a vertical 80-layer atmosphere column with nine aligned tracks.
- Middle top: validation → optional seed quantization → hydrostatic boundary.
- Middle bottom: EOS/populations → continuum threshold → line selection.
- Right: a frequency-by-depth opacity heatmap.
- A grey loop from catalog objects back to the opacity builder says “reuse membership; recompute opacity.”

**Chapter 12 — “30,000 frequencies become a few physical columns.”**

- A wide frequency fan enters private chunk boxes.
- Fixed-order arrows reduce to named Rosseland, flux, heating, lambda, and radiative-force columns.
- Below, four perturbed EOS cards (`T+`, `T−`, `P+`, `P−`) feed a convection card.
- The exact cards are `TransferAccumulation`, then the radiation/convection prefix of `finalize_transfer_state`; the latter returns `IterationFinalization`, whose correction fields are opened in Chapter 13.

**Chapter 13 — “The exact atmosphere orbit.”**

- A circular pass loop with the 15 exact steps from the governing state machine.
- Carried edges for Rosseland lookup, previous correction, radiation-pressure constant, and catalogs are drawn inside the orbit.
- The correction/remap edge is enlarged.
- The only terminal arrow crosses a double-line fixed-column quantization gate to `AtmosphereRunResult`.
- A separate red/green fork shows “unconverged: debug only” versus “converged: rebuild schema v4.”

**Chapter 14 — “Prediction aims at a basin, not at acceptance.”**

- Labels enter either the five/CNO feature path or the direct set encoder.
- Both meet at the shared 160-coefficient → PCA → six-profile decoder.
- A violet initialized point appears near a green fixed point inside a contour basin.
- An arrow through fixed-column quantization enters the Chapter 13 orbit.
- Direct abundance has a visible mixture-hash tether from public coordinates to the exact solver.

**Chapter 15 — “Two honest workflows.”**

- A top fast lane ends at `LabelSpectrum`, with `atmosphere_converged=False` and `atmosphere_closure_required=True`.
- A lower verified lane crosses exact convergence, product rebuild, synthesis, and independent acceptance gates.
- Four vertical case cards—hot, solar, giant, cool molecule-rich—share the same metrics.

### Interactive diagrams worth building

- Chapter 11: toggle continuum-only/line-blanketed opacity and inspect one layer/frequency.
- Chapter 12: change chunk count and see identical physical sums within the declared tolerance; toggle ideal-gas versus full-EOS convection derivatives.
- Chapter 13: step through passes and inspect carried versus reset fields; quantization can be toggled only at the terminal boundary.
- Chapter 14: move a requested point relative to the support region and display requested versus initializer-only projected labels; adjust one direct abundance by 0.01 dex and show the exact changed mixture slot/hash.
- Chapter 15: acceptance dashboard where structural, flux, hydrostatic, schema, and spectral gates can fail independently.

Interactions must use compact checked fixtures. They are explanatory views of canonical code, not alternative implementations.

### Density-driven split flags

The fifteen-chapter spine should remain intact, but three chapters need visible internal acts and web anchors:

- **Chapter 11: high density.** Use `11A Seed and numerical boundaries` and `11B Opacity sampling and blanketing`. If the draft exceeds roughly 70 rendered pages or 18 substantial code cells, publish them as two web routes under one Chapter 11 navigation item.
- **Chapter 12: high density.** Use `12A Integrated radiation` and `12B Thermodynamics and convection`. The exact seam is the returned `TransferAccumulation`; Act B then follows the radiation/convection prefix of `finalize_transfer_state`.
- **Chapter 13: very high density.** Use `13A Correction and remap` and `13B Full iteration and operations`. The exact seam is `IterationRemap`.
- **Chapter 14: very high density.** Use `14A Fixed points and common decoder`, `14B Five/CNO initializer`, and `14C Direct abundances and closure`. If a print split is later authorized, the clean seam is between ordinary/CNO and direct abundance; until then they remain acts of Chapter 14.
- **Chapter 15: normal capstone density** if it imports all prior artifacts rather than reimplementing them.

These are layout warnings, not permission to omit material or create hidden dependencies.

---

# Chapter 11 — Starting and Blanketing an Atmosphere

## Chapter question

What exact state can safely enter a physical atmosphere pass, and how is its composition converted into the sampled continuum-plus-line opacity that drives blanketing?

The chapter should open with two apparently smooth 80-layer seeds, one invalid because column mass turns over at a single layer, then compare continuum-only and line-blanketed opacity for the valid seed. This joins the chapter's two acts around one idea: the transfer solver is only as trustworthy as both its depth state and its opacity state.

## Consumes

- Chapter 1 grey seed, standard depth grid, hydrostatic balance, Rosseland definitions, and physical assumptions.
- Chapter 2 data/schema records, Numba basics, dtype/shape conventions, and fixed-ne bridge vocabulary.
- Chapters 3–4 atomic/molecular state, `(L, 99)` abundance fractions, `(L, 1006)` packed atomic populations, and molecular state.
- Chapter 5 complete continuum functions.
- Chapters 6–8 line widths, catalogs, selection, sparse accumulation, special profiles, and molecular lines.
- Chapter 9 depth remapping/interpolation and transfer grid conventions.

A supplied learned-origin seed may be used as a second fixture, but it is treated strictly as a `ModelAtmosphere`. Its generation is not explained until Chapter 14.

## Produces

### Act I artifact: exact `ModelAtmosphere`, `AtmosphereConfig`, and `RunSetup`

There is no seed-wrapper dataclass. The exact validated artifact is the `RunSetup`
returned by `resolve_run_setup(config)`, and its seed is
`RunSetup.atmosphere: ModelAtmosphere`.

`ModelAtmosphere` contains nine aligned CPU `float64[L]` fields:

| Field | Meaning/unit/constraint |
|---|---|
| `column_mass` | \(m\), g cm\(^{-2}\), positive and strictly increasing |
| `temperature` | \(T\), K, positive |
| `gas_pressure` | \(P_{\rm gas}\), dyn cm\(^{-2}\), positive |
| `electron_density` | \(n_e\), cm\(^{-3}\), positive |
| `rosseland_opacity` | \(\kappa_{\rm R}\), cm\(^2\) g\(^{-1}\), positive |
| `radiative_acceleration` | \(g_{\rm rad}\), cm s\(^{-2}\), finite |
| `microturbulence` | \(\xi\), cm s\(^{-1}\), nonnegative |
| `convective_flux` | solver integrated-flux convention, finite; physically introduced in Ch. 12 |
| `convective_velocity` | cm s\(^{-1}\), finite; physically introduced in Ch. 12 |

Metadata carries effective temperature, \(\log g\), opacity flags, pressure-iteration choice, iteration label, and the surface-radiation-pressure constant. Fixed-column abundances are keyed by atomic number 1–99; H/He keep their linear convention and metals their logarithmic convention as already defined in Chapter 2.

`AtmosphereConfig` contains exact `AtmosphereInput` and `AtmosphereOutput`
dataclasses. Its source defaults are:

```text
iterations=1
enable_molecules=False
enable_convection=True
enable_convergence_stop=False
minimum_iterations_before_convergence=3
required_consecutive_converged_iterations=1
maximum_deep_layer_relative_temperature_change=5.0e-4
maximum_all_layer_relative_temperature_change=None
molecular_convection_thermal_tracks_perturbation=True
```

`AtmosphereInput.initial_atmosphere` is required. Its optional paths have the exact
names `molecules_path`, `selected_line_catalog_path`,
`detailed_line_catalog_path`, `predicted_atomic_lines_path`,
`observed_atomic_lines_path`, `high_excitation_lines_path`,
`diatomic_lines_path`, `titanium_oxide_lines_path`, `water_lines_path`, and
`h3plus_lines_path`. `AtmosphereOutput` has
`structured_atmosphere_path`, `diagnostics_path`, and `debug_state_path`;
`diagnostics_path` is declared but is not currently written.

`RunSetup` carries the resolved controls and exact fields
`atmosphere`, `iterations`, `enable_convergence_stop`,
`minimum_iterations_before_convergence`,
`required_consecutive_converged_iterations`,
`maximum_deep_layer_relative_temperature_change`,
`maximum_all_layer_relative_temperature_change`, `surface_gravity_cgs`,
`opacity_flags`, `molecules_enabled`, `pressure_iteration_enabled`,
`convection`, `turbulence`, `surface_radiation_pressure_constant`,
`effective_temperature`, `log_surface_gravity`, and
`standard_rosseland_optical_depth`.

`resolve_run_setup` fixes the supported convection settings to
`enabled=config.enable_convection`, `mixing_length=1.25`,
`overshoot_weight=0.0`, and `zero_top_layer_count=0`. It constructs
`TurbulenceSettings(enabled=False, density_coefficient=0.0,
density_power=0.0, sound_speed_fraction=0.0,
constant_velocity_km_s=0.0)`. The later convection call interprets
`zero_top_layer_count <= 0` as 36 for its endpoint diagnostic.

`DEFAULT_OPACITY_FLAGS` is exactly
`[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]`.
In `prepare_opacity_state`, zero-based `line_flags[14]` enables selected-line
opacity and `line_flags[16]` enables detailed-transition opacity.

### Act II artifact: exact `AtmospherePopulationState` and `OpacityState`

`AtmospherePopulationState` has exact fields `setup`, `runtime_state`,
`fractional_doppler_widths`,
`partition_normalized_population_over_mass_density_and_fractional_doppler_width`,
`temperature_iteration_cache`, and optional `molecular_state`.
Its `AtmosphereRuntimeState` field atlas must preserve the source names
`gas_pressure`, `electron_density`, `total_nuclei_number_density`,
`mass_density`, `charge_square_density`, `elemental_abundances_by_layer`,
`mean_nuclear_mass_amu`, `ion_stage_populations_by_packed_slot`,
`partition_normalized_populations_by_packed_slot`,
`specific_internal_energy`, `major_isotope_mass_amu`, optional
`fractional_doppler_widths`,
`partition_normalized_population_over_mass_density_and_fractional_doppler_width`,
`hydrogen_departure_coefficients`, `metal_departure_coefficients`, and
`geometric_depth_below_surface_km`.

```text
OpacityState
    population_state: AtmospherePopulationState
    continuum_atmosphere: ContinuumAtmosphereState
    opacity_wavelength_grid_nm: float64[Nnu]
    opacity_frequency_hz: float64[Nnu]
    frequency_weights: float64[Nnu]
    active_continuum_indices: integer[Nref]
    active_continuum_frequency_hz: float64[Nref]
    continuum_absorption: float64[L, Nnu]
    continuum_scattering: float64[L, Nnu]
    continuum_source: float64[L, Nnu]
    continuum_line_selection_threshold: float32[L, 344]
    continuum_reference_wavelength_nm: float64[344]
    wavelength_bin_edges: int64[344]
    line_opacity: LineOpacityState
    rosseland_table: RosselandOpacityTable
    selected_line_catalog: SelectedLineCatalog | None
    transition_line_catalog: LineTransitionCatalog | None
```

`Nnu = 30_000` for the production atmosphere grid. `Nref` is the dynamic
active subset of the first 343 continuum-reference columns; the returned
threshold table always has 344 columns. The catalog handles are part of the
returned state so Chapter 13 can preserve object identity across passes.

`RosselandOpacityTable` has the exact fields
`normalized_log_temperature`, `normalized_log_pressure`,
`log10_rosseland_opacity`, `entry_count`, `log_temperature_origin`,
`log_pressure_origin`, `log_temperature_span`, and `log_pressure_span`.
Chapter 13 carries this object through
`TemperatureCorrectionState.rosseland_opacity_table`.

## Prose and derivation arc

### Act I — Seed and numerical boundaries

1. **A structure is more than \(T(\tau)\).** Contrast the grey profile with the nine-column state required by the exact pass.
2. **The exact standard coordinate.** Materialize
   \[
   \tau_{\mathrm R,i}=10^{-6.875+0.125i}
   \]
   as one tested helper.
3. **Validation before compilation.** Check shape, dtype-convertibility, finiteness, signs, monotonic column mass, and compatible layer counts in Python before any compiled physics call.
4. **Hydrostatic pressure with carried non-gas support.** Starting from \(dP_{\rm tot}/dm=g\), identify
   \(P_{\rm tot}=P_{\rm gas}+P_{\rm rad}+P_{\rm turb}\). The next pass solves gas pressure using the preceding remapped radiation/turbulence columns.
5. **Microturbulence as resolved seed state.** Fill the standard prescription only when no positive input value is present; do not evolve it as a correction variable.
6. **Fixed-column quantization as an operator.** Define \(Q(x)\) by in-memory format/parse, compare \(x\) and \(Q(x)\), and test idempotence. It is a numerical boundary, not cosmetic I/O.
7. **Remap contract.** Demonstrate one constant and one monotone scalar column, endpoint diagnostics, and the rule that every carried field must share one target coordinate. Defer the complete corrected-state remap to Chapter 13.

### Act II — Sampling and blanketing

8. **Why atmosphere sampling differs from final synthesis.** The 30,000-point grid represents radiative equilibrium/blanketing, not a final high-resolution spectrum.
   Show the exact `build_opacity_sampling_grid` branches: start index 1,
   replaced by 3577 below 30,000 K, 7027 below 13,000 K, 9599 below 7,250 K,
   and 11601 below 4,500 K. The exact wavelength rule is
   `10**(1 + 0.0001*(one_based_index + start_index - 1))` nm, followed by the
   source's endpoint/interior frequency-weight formulas.
9. **Compose the population wrapper.** Reuse the Chapter 3–4 EOS, molecular, specific-energy, Doppler, and line-strength functions.
10. **Continuum state at all sampled frequencies.** Allocate absorption, scattering, and source slabs only after declaring the exact depth-major `(L, Nnu)` order and memory cost.
11. **Compact continuum reference for selection.** Build the reference columns and interpolate the selection threshold to line positions.
12. **Discrete membership, continuous opacity.** Select catalog membership once on the first pass. Recompute the selected transitions' opacity every pass because populations and widths change.
13. **Selected versus detailed transitions.** Preserve separate provenance while adding both to the same line-opacity state.
14. **Resident catalogs, bounded chunks, fixed reduction.** Connect earlier `prange` kernels to production memory and deterministic deposition.
15. **Back-warming.** Compare depth/frequency opacity budgets and explain qualitatively how line blocking changes the correction direction without claiming a corrected atmosphere yet.

## Bite-size canonical code artifacts

### Act I

1. `standard_rosseland_optical_depth_grid(layers: int) -> np.ndarray`
2. `validate_atmosphere_seed(atmosphere: ModelAtmosphere) -> None`
3. `surface_gravity_from_atmosphere(atmosphere: ModelAtmosphere) -> float`
4. `surface_radiation_pressure_constant_from_atmosphere(atmosphere: ModelAtmosphere) -> float`
5. `opacity_flags_from_atmosphere(atmosphere: ModelAtmosphere) -> list[int]`
6. `initialize_microturbulence(atmosphere, *, effective_temperature, log_surface_gravity, standard_rosseland_optical_depth) -> None`
7. `integrate_hydrostatic_pressure(atmosphere, *, surface_gravity_cgs, integrated_radiation_pressure, turbulent_pressure, pressure_constant=0.0) -> np.ndarray`
8. `format_atmosphere_deck(model: ModelAtmosphere) -> str`
9. `parse_atmosphere_deck(deck_text: str, *, source="<in-memory deck>") -> ModelAtmosphere`
10. `remap_to_grid(source_grid, source_values, target_grid) -> tuple[np.ndarray, int]` using the exact Chapter 9 function
11. `resolve_run_setup(config: AtmosphereConfig) -> RunSetup`

The fixed-column cell must literally call
`parse_atmosphere_deck(format_atmosphere_deck(model), source="<roundtrip>")`; it must
not wrap that boundary in a differently named package API.

### Act II

12. `prepare_population_state(config, *, temperature_iteration_index=1, setup=None, molecular_thermal_energy_erg=None) -> AtmospherePopulationState`
13. `build_opacity_sampling_grid(effective_temperature: float) -> tuple[np.ndarray, np.ndarray]`, returning wavelength and frequency weights; `prepare_opacity_state` constructs `opacity_frequency_hz`
14. `build_continuum_atmosphere_state(atmosphere, state) -> ContinuumAtmosphereState`
15. `active_continuum_reference_frequencies(effective_temperature: float) -> tuple[np.ndarray, np.ndarray]`
16. `assemble_continuum_line_selection_threshold(*, effective_temperature, temperature_k, active_continuum_absorption, active_continuum_scattering) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
17. `allocate_line_opacity_state(*, layer_count, wavelength_count) -> LineOpacityState`
18. `generate_selected_lines(*, partition_normalized_population_over_mass_density_and_fractional_doppler_width, continuum_line_selection_threshold, packed_continuum_wavelengths, hc_over_kt, selected_lines_output=None, predicted_atomic_lines_path=None, observed_atomic_lines_path=None, high_excitation_lines_path=None, diatomic_lines_path=None, titanium_oxide_lines_path=None, water_lines_path=None, h3plus_lines_path=None)`, called by the exact private adapter `_generate_standard_selected_lines`
19. `accumulate_selected_line_opacity(*, selected_lines, opacity_wavelength_grid_nm, wavelength_bin_edges, continuum_line_selection_threshold, temperature, hc_over_kt, electron_density, ion_stage_populations_by_packed_slot, partition_normalized_population_over_mass_density_and_fractional_doppler_width, fractional_doppler_widths, wavelength_start_index=1, wavelength_stop_index=None) -> LineOpacityState`
20. `read_line_transition_catalog(path) -> LineTransitionCatalog`
21. `accumulate_transition_line_opacity(*, transition_lines, opacity_wavelength_grid_nm, wavelength_bin_edges, continuum_line_selection_threshold, temperature, hc_over_kt, electron_density, ion_stage_populations_by_packed_slot, partition_normalized_population_over_mass_density_and_fractional_doppler_width, fractional_doppler_widths, partition_normalized_populations_by_packed_slot=None, mass_density=None, base_line_mass_absorption_coefficient=None, wavelength_start_index=1, wavelength_stop_index=None) -> LineOpacityState`
22. `prepare_opacity_state(config, *, population_state=None, temperature_iteration_index=1, rosseland_table=None, selected_line_catalog=None, transition_line_catalog=None) -> OpacityState`

If the chapter density becomes too high, cells 1–11 and 12–22 become the two
web routes identified above; their exact artifacts do not change.

## Exact state ordering

### Seed boundary

For a hand-built seed:

1. construct nine columns and metadata;
2. apply the fixed-column round trip only if this input is intentionally at a seed quantization boundary;
3. validate;
4. resolve gravity, flags, pressure choice, and standard grid;
5. fill microturbulence only if the profile contains no positive value;
6. return the setup.

For pass \(n>1\), later used by Chapter 13:

1. consume the prior unquantized `IterationRemap`;
2. recompute gas pressure with its remapped radiation/turbulent support when enabled;
3. copy all other remapped columns;
4. substitute the carried surface-radiation-pressure constant;
5. form the pass-local setup.

### Population and opacity

1. allocate runtime state from the current atmosphere;
2. update charge-square density;
3. when molecules are enabled, consume/load the molecular catalog and initialize molecular state;
4. when pressure iteration is enabled, solve electron/molecular closure;
5. populate all required species and specific internal energy;
6. update fractional Doppler widths and normalized strength factors;
7. build the continuum atmosphere state;
8. use the previous Rosseland table or a correctly shaped empty first-pass table;
9. build the 30,000-point grid and weights;
10. compute full continuum absorption/scattering/source;
11. extract compact reference columns and selection threshold;
12. allocate a fresh line-opacity state for the current atmosphere;
13. if selection is enabled and no selected catalog exists, create/read it;
14. accumulate selected-line opacity;
15. if detailed opacity is enabled and no detailed catalog exists, load it;
16. accumulate detailed-transition opacity;
17. return state plus both reusable catalog handles.

Selecting before populations/widths exist, regenerating selection every pass, or retaining a previous pass's line-opacity slab are all wrong.

## Data

- A tiny 80-layer grey/checked seed and one supplied learned-origin seed fixture.
- A fixed-column text fixture for numerical round-trip tests.
- Compact atomic/molecular catalog subsets with production record schema and bin behavior.
- Continuum-only operation requires no full line catalog.
- Full-catalog parity is conditional on optional installation data.
- Checksums, licenses, full inventory, and installation commands remain in the data appendix.

## Checks shown

- Exact grid endpoints and 0.125 spacing in \(\log_{10}\tau_{\rm R}\).
- All nine columns are finite CPU `float64[80]`, outer-to-inner.
- Positive/monotone required fields and bad-seed failure gallery.
- Hydrostatic residual with and without radiation-pressure support.
- `Q(Q(x)) == Q(x)` at the parsed field/metadata boundary.
- Constant/monotone remap and endpoint behavior.
- Frequency/wavelength order and weight checks.
- Nonnegative continuum absorption/scattering and finite slabs.
- One-line selection by hand.
- Selected catalog object identity remains unchanged on a second call, while opacity changes under a temperature perturbation.
- Continuum-only, selected-only, detailed-only, and combined branches.
- Chunk-size invariance within declared floating-point tolerance.

## Strict parity gates

- Standard 80-point grid exact in `float64`.
- Fixed-column round trip exact against pinned seed fixtures.
- Hydrostatic pressure component parity for hot, solar, giant, and cool cases.
- Population state parity before opacity, including molecular-enabled cases.
- 30,000-point grid/weights parity.
- Continuum and line slab component parity.
- First-pass selected catalog membership/order/hash parity.
- Later-pass catalog identity reuse and no repeated file load/selection.
- Full-catalog runs record manifest/catalog hashes.
- Cross-thread reductions may differ by the documented few ulp but never change discrete membership or branch choice.

## Failure modes and boundaries

- Nonmonotone depth, NaNs, nonpositive required fields, and incompatible layer count fail before compilation.
- A warm-start quantization round trip can change the early trajectory; this is expected and tested.
- Evolving turbulent pressure is unsupported; nonzero unsupported settings fail loudly.
- Production line opacity expects the validated 80-layer layout.
- Missing enabled catalog/shard/checksum is a hard error, not a continuum-only fallback.
- A detailed-line flag without data is a hard error.
- Full resident catalogs may exceed constrained memory; subset pedagogy is not full-data parity.
- Line selection is a host-side discrete choice and is not differentiable.
- The standard atmosphere route supports converted atomic, diatomic, TiO, and
  water selection. Atmosphere H3+ is a separate explicit-path opt-in because
  `source_line_paths()` supplies no default H3+ file.
- `_require_supported_run_setup` rejects the exact turbulent-pressure and
  HLINOP branches; it has no blanket raw-molecular-selector guard. Only the
  synthesis H2O compiler is compiler-only and omitted from the standard
  synthesis runtime; atmosphere water selection is active.
- A validated, blanketed pass input is not a converged atmosphere.

## End contract

Chapter 11 consumes stellar labels/mixture plus a seed and produces a validated pass setup, current population state, complete opacity-sampling state, and reusable catalog handles. It does **not** solve transfer, correct the structure, decide convergence, generate learned seeds, or write a physical product.

---

# Chapter 12 — Radiation, Thermodynamics, and Convection

## Chapter question

Which frequency-integrated radiation quantities and equation-of-state derivatives are required to decide how much flux is radiative and how much can be carried by convection?

Open by passing one sampled frequency through all layers, then show how 30,000 frequencies collapse into a small set of named columns. Below that, compare ideal-gas and full-EOS convection derivatives for a partially ionized or molecule-rich layer.

Use the paper's notation before mapping to exact accumulator fields:

\[
\frac{1}{\kappa_{\rm R}}=
\frac{\int_0^\infty \chi_\nu^{-1}
(\partial B_\nu/\partial T)\,\mathrm d\nu}
{\int_0^\infty(\partial B_\nu/\partial T)\,\mathrm d\nu},
\qquad
\mathrm d\tau_{\rm R}=\kappa_{\rm R}\,\mathrm dm,
\]

\[
g_{\rm rad}(m)=\frac1c\int_0^\infty
\chi_\nu(m)F_\nu(m)\,\mathrm d\nu,
\]

\[
F_{\rm rad}(m)+F_{\rm conv}(m)=\sigma_{\rm SB}T_{\rm eff}^4,
\qquad
\epsilon_F(m)=
\frac{F_{\rm rad}(m)+F_{\rm conv}(m)}
{\sigma_{\rm SB}T_{\rm eff}^4}-1.
\]

Then state that the source integrates Eddington flux \(H=F/(4\pi)\), using
the exact `target_integrated_eddington_flux` named in the notation audit.

## Consumes

- `OpacityState` and `AtmospherePopulationState` from Chapter 11.
- Formal transfer, scattering, moment definitions, boundary conventions, and transfer tables from Chapter 9.
- Rosseland weighting from Chapters 1 and 5.
- Atomic/molecular populations and specific internal energy from Chapters 3–4.
- Numba private-state/reduction concepts from Chapter 2.

## Produces

### Act I: radiation records

`TransferAccumulation`:

```text
opacity_state: OpacityState
frequency_start_index: int
frequency_stop_index: int
rosseland_accumulator: float64[L]
radiative_pressure_state: RadiativePressureState
temperature_correction_state: TemperatureCorrectionState
```

`RadiativePressureState`:

- `integrated_eddington_flux: float64[L]`;
- `radiation_energy_density: float64[L]`;
- `radiative_acceleration: float64[L]`;
- `integrated_radiation_pressure: float64[L]`;
- `absolute_radiation_pressure: float64[L]`;
- `surface_radiation_pressure_constant: float`.

`TemperatureCorrectionState`:

- `mean_intensity_minus_source_integral: float64[L]`;
- `absorption_heating_derivative: float64[L]`;
- `diagonal_lambda_accumulator: float64[L]`;
- `integrated_eddington_flux: float64[L]`;
- `previous_temperature_correction: float64[L]`;
- `rosseland_opacity_table: RosselandOpacityTable`.

The transfer boundary deliberately mixes axis orders and dtypes:
`planck_all` and `stimulated_all` are `float64[Nnu, L]`, while continuum and
line slabs are passed depth-major `(L, Nnu)`. The line slab and transfer
operator tables are explicitly made contiguous `float32`; the nine
thread-private accumulator families comprise eight
`float64[chunk_count, L]` arrays plus one
`float64[chunk_count, 1]` surface term, all reduced in ascending chunk order.

The exact implementation has no intermediate finalized-radiation dataclass.
`finalize_transfer_state` performs Rosseland finalization, radiation-pressure
finalization, lookup ingestion, optional finite-difference convection, and
the exact Chapter 13 `apply_temperature_correction` call with `mode=3` in one
source routine. It returns
the exact `IterationFinalization`:

```text
transfer_accumulation: TransferAccumulation
rosseland_opacity: float64[L]
rosseland_optical_depth: float64[L]
radiative_pressure_state: RadiativePressureState
temperature_correction_result: TemperatureCorrectionResult
convection_result: ConvectionResult | None
convection_finite_difference_samples: ConvectionFiniteDifferenceSamples | None
```

Chapter 12 explains the radiation and convection prefix of this exact routine;
Chapter 13 opens its `temperature_correction_result`. The code cell may call
the full exact function and keep the correction fields collapsed in the state
inspector. It must not create a replacement partial-finalization API.

### Act II: thermodynamic/convection records

`ConvectionFiniteDifferenceSamples` contains eight `float64[L]` arrays:

- \(e(1.001T,P)\), \(e(0.999T,P)\);
- \(e(T,1.001P)\), \(e(T,0.999P)\);
- the corresponding four mass-density columns.

The specific-energy samples include the radiation-energy term used by the production branch. With molecular thermal tracking enabled, the molecular thermal-energy column follows each temperature perturbation.

`ConvectionResult`:

```text
geometric_depth_below_surface_km: float64[L]
logarithmic_temperature_pressure_gradient: float64[L]
heat_capacity: float64[L]
log_density_temperature_derivative_at_constant_total_pressure: float64[L]
sound_speed: float64[L]
adiabatic_gradient: float64[L]
pressure_scale_height: float64[L]
convective_flux: float64[L]
convective_velocity: float64[L]
raw_convective_flux: float64[L]
overshoot_convective_flux: float64[L]
```

A separate `DisabledConvectionDiagnostics` contains exact fields
`convective_flux` and `convective_velocity`. It is not production
finite-difference convection.

## Prose and derivation arc

### Act I — Integrated radiation

1. Trace total opacity, source, mean intensity, Eddington flux, surface moment, Rosseland weight, heating residual, and force contribution for one frequency.
2. Introduce the reset/accumulate/finalize mode lifecycle.
3. Connect the Rosseland harmonic accumulator to \(\kappa_{\rm R}\) and \(d\tau_{\rm R}=\kappa_{\rm R}dm\).
4. Derive opacity-weighted flux as radiative force per mass and show how the surface moment becomes the pressure integration constant.
5. Distinguish integrated flux error from local \(J-S\)-weighted heating imbalance.
6. Explain private frequency-chunk arrays and fixed-order reduction.
7. Vectorize Planck/stimulated-emission columns in validated frequency order.
8. Separate reset fields from persistent `previous_temperature_correction` and Rosseland lookup.

### Act II — Thermodynamics and convection

9. Summarize, by reference, the translational, excitation, ionization, dissociation, molecular, and radiation energy contributions.
10. Derive the central differences for \(\pm0.1\%\) perturbations and the factor of 500.
11. Explain why each perturbation requires full electron/molecular closure.
12. Add the \(3P_{\rm rad}/\rho\) energy term and optical-depth dilution behavior.
13. Derive heat capacity, density derivative, sound speed, and adiabatic gradient from the samples.
14. Build mixing-length scale height, superadiabatic gradient, radiative leakage, velocity, and flux.
15. Distinguish raw local flux, overshoot contribution, surface suppression, and returned flux.
16. Treat snapshot/restore as part of the algorithm.
17. Mark the parallelism boundary: layerwise algebra can compile, but the four mutations and molecular continuation are ordered.

## Bite-size canonical code artifacts

### Act I

1. private scalar check `_planck_source_and_stimulated_emission(*, frequency_hz, h_over_kt) -> tuple[np.ndarray, np.ndarray]`, immediately paired with the exact vectorized block in `accumulate_transfer_state`
2. `initialize_radiative_pressure_state(layer_count: int) -> RadiativePressureState`
3. `initialize_temperature_correction_state(layer_count: int) -> TemperatureCorrectionState`
4. `rosseland_mean_step(rosseland_accumulator, *, mode, frequency_weight, planck_source, frequency_hz, h_over_kt, temperature_k, stimulated_emission, total_opacity, frequency_count, column_mass) -> tuple[np.ndarray, np.ndarray]`
5. `accumulate_radiative_pressure(state, *, mode, frequency_weight, total_opacity, monochromatic_eddington_flux, mean_intensity, surface_second_moment, target_integrated_eddington_flux, column_mass) -> None`
6. `apply_temperature_correction(state, *, mode, frequency_weight, column_mass, total_opacity, monochromatic_eddington_flux, mean_intensity_minus_source, monochromatic_optical_depth, planck_source, frequency_hz, h_over_kt, temperature_k, stimulated_emission, scattering_fraction, target_integrated_eddington_flux, effective_temperature, frequency_count, rosseland_optical_depth=None, rosseland_opacity=None, iteration_index=1, convection_enabled=False, convective_flux=None, previous_convective_flux=None, logarithmic_temperature_pressure_gradient=None, adiabatic_gradient=None, pressure_scale_height=None, total_pressure=None, mass_density=None, log_density_temperature_derivative_at_constant_total_pressure=None, heat_capacity=None, mixing_length=1.0, smooth_start_layer=0, smooth_stop_layer=0, smooth_left_weight=0.3, smooth_center_weight=0.4, smooth_right_weight=0.3, integrated_radiation_pressure=None, turbulent_pressure=None, surface_gravity_cgs=1.0e4, standard_log_tau_step=0.125, standard_log_tau_start=-6.875) -> TemperatureCorrectionResult | None`, restricted here to exact mode-1 reset and mode-2 accumulation
7. `transfer_chunk_count() -> int`
8. `accumulate_transfer_range_parallel`, with exact positional order shown
   verbatim in the code cell:

   ```text
   chunk_count, range_start, range_stop,
   frequency_hz, frequency_weights, planck_all, stimulated_all,
   continuum_absorption_slab, continuum_scattering_slab,
   continuum_source_slab, line_mass_absorption_coefficient_slab,
   column_mass, h_over_kt, temperature, transfer_grid,
   mean_intensity_operator, eddington_flux_operator, second_moment_weights,
   target_integrated_eddington_flux, effective_temperature, frequency_count,
   rosseland_accumulator, radiation_energy_density,
   integrated_eddington_flux, radiative_acceleration,
   surface_radiation_pressure_constant,
   temperature_correction_heating_derivative,
   temperature_correction_mean_intensity_minus_source_integral,
   temperature_correction_integrated_eddington_flux,
   temperature_correction_diagonal_lambda
   ```
9. `accumulate_transfer_state(opacity_state, *, frequency_start_index=0, frequency_stop_index=None, temperature_correction_state=None) -> TransferAccumulation`
10. `ingest_temperature_correction_rosseland_table(state, *, temperature_k, gas_pressure, rosseland_opacity) -> None`

### Act II

11. `compute_convection_finite_difference_samples(*, atmosphere, runtime_state, absolute_radiation_pressure, rosseland_optical_depth, temperature_iteration_seed, temperature_iteration_cache, molecules_enabled=False, molecular_state=None, molecular_thermal_energy_tracks_perturbation=False) -> ConvectionFiniteDifferenceSamples`
12. `integrate_geometric_depth_below_surface_km(*, column_mass, mass_density) -> np.ndarray`
13. `compute_convection(*, rosseland_table, column_mass, rosseland_optical_depth, temperature_k, gas_pressure, mass_density, rosseland_opacity, microturbulence, absolute_radiation_pressure, total_pressure, surface_gravity_cgs, target_integrated_eddington_flux, mixing_length=1.0, overshoot_weight=1.0, convection_enabled=True, zero_top_layer_count=36, specific_internal_energy_plus_temperature=None, specific_internal_energy_minus_temperature=None, specific_internal_energy_plus_pressure=None, specific_internal_energy_minus_pressure=None, density_plus_temperature=None, density_minus_temperature=None, density_plus_pressure=None, density_minus_pressure=None) -> ConvectionResult`
14. `compute_disabled_convection_diagnostics(*, column_mass, rosseland_optical_depth, temperature_k, gas_pressure, mass_density, rosseland_opacity, absolute_radiation_pressure, total_pressure, surface_gravity_cgs, target_integrated_eddington_flux, mixing_length, rosseland_table, overshoot_weight=1.0, zero_top_layer_count=36) -> DisabledConvectionDiagnostics`
15. `finalize_transfer_state(transfer_accumulation, *, iteration_index=1, temperature_iteration_seed=None, convection_enabled=False, convective_flux=None, previous_convective_flux=None, logarithmic_temperature_pressure_gradient=None, adiabatic_gradient=None, pressure_scale_height=None, total_pressure=None, log_density_temperature_derivative_at_constant_total_pressure=None, heat_capacity=None, mixing_length=1.0, integrated_radiation_pressure=None, turbulent_pressure=None, molecular_convection_thermal_tracks_perturbation=False) -> IterationFinalization`

The scalar one-frequency view and the snapshot inventory are notebook-local
explanations only. They introduce no package classes or function names.

## Exact state ordering

### Transfer

1. receive complete opacity state;
2. reset current-pass Rosseland, radiative-pressure, and correction accumulators;
3. preserve prior correction and Rosseland lookup;
4. prepare Planck/stimulated-emission columns in opacity-grid order;
5. partition the declared `[start, stop)` frequency interval deterministically;
6. within each chunk, solve frequencies in order and accumulate private Rosseland, pressure, flux, heating, derivative, and lambda arrays;
7. reduce private chunks in fixed chunk order;
8. return `TransferAccumulation`;
9. in the prefix of `finalize_transfer_state`, finalize \(\kappa_{\rm R}\) and integrate \(\tau_{\rm R}\);
10. finalize radiation energy density, acceleration, integrated/absolute pressure, and surface constant;
11. ingest current \((T,P_{\rm gas},\kappa_{\rm R})\) into the persistent lookup;
12. proceed through the exact convection and mode-3 correction calls before returning `IterationFinalization`; Chapter 13 explains that correction rather than inventing an earlier return.

### Four EOS perturbations

1. snapshot temperature, gas pressure, electron density, total nuclei density, mass density, actual/normalized packed populations, specific internal energy, iteration cache, and all molecular population/equation/seed/thermal-mode fields;
2. compute dilution \(1-e^{-\tau_{\rm R}}\);
3. set \(T=1.001T_0\), recompute closure with a distinct deterministic iteration index, save \(e_{T+},\rho_{T+}\);
4. set \(T=0.999T_0\), recompute, save \(e_{T-},\rho_{T-}\);
5. restore \(T=T_0\), set \(P=1.001P_0\), recompute, save \(e_{P+},\rho_{P+}\);
6. set \(P=0.999P_0\), recompute, save \(e_{P-},\rho_{P-}\);
7. in `finally`, restore every central field, cache entry, molecular seed, and mode;
8. evaluate convection from the restored central state and eight saved arrays;
9. return without mutating the central state.

## Data

- Chapter 11 compact/full opacity fixtures and transfer tables.
- Full 30,000-frequency state for normal pass parity; a smaller slice only for first explanatory cells.
- Compact molecular-equilibrium data for the cool derivative example.
- No new large bundle.

## Checks shown

- One-frequency scalar terms match the parallel kernel contribution.
- Grey/constant-source limiting behavior.
- \(\tau_{\rm R}\) is monotone inward.
- Differentiated integrated radiation pressure recovers radiative acceleration within discretization tolerance.
- Mode reset zeroes only per-pass fields.
- One/many chunks and thread counts agree within the declared envelope.
- Deliberately reversing reduction order illustrates last-bit sensitivity without becoming canonical.
- Snapshot/restore equality, including forced-exception restoration.
- Finite differences converge when the perturbation is halved on a smooth toy EOS.
- Full-EOS and ideal-gas derivatives differ intelligibly in ionized/molecular layers.
- Geometric depth starts at zero and increases inward.
- Stable layers carry no positive convective flux under the production convention.
- Top-layer suppression and overshoot switches have isolated checks.

## Strict parity gates

- Every radiative accumulator is gated independently for hot, solar, giant, and cool fixtures.
- Frequency interval/order exact.
- Same-thread deterministic; cross-thread few-ulp envelope explicit.
- All eight EOS perturbation arrays exact within component tolerance.
- Restored central state equal to pre-perturbation state.
- Heat capacity, density derivative, sound speed, adiabatic gradient, scale height, raw/overshoot/final flux, and velocity gated separately.
- Production convection parity requires finite-difference samples; ideal-gas fallback cannot pass.
- Atomic-only and molecule-enabled branches both covered.

## Failure modes and boundaries

- Negative/nonfinite total opacity fails with frequency/layer context.
- Excessive threads/chunks can make private arrays dominate memory.
- Resetting correction history or lookup can yield plausible values while breaking trajectory parity.
- Frequency parallelism does not make the depth-recursive transfer solve parallel.
- A molecular Newton failure in a perturbation aborts after restoration; it does not fall back to ideal gas.
- Omitting molecular thermal tracking can bias cool-star derivatives.
- Mixing-length convection is a declared local one-dimensional prescription, not hydrodynamics.
- Turbulent pressure remains unsupported.
- This chapter produces radiative and convection state, not a corrected atmosphere.

## End contract

Chapter 12 consumes one exact `OpacityState` and explains the radiation and
convection portions of the exact `accumulate_transfer_state` /
`finalize_transfer_state` path. Because `finalize_transfer_state` is a combined
source routine, its returned `IterationFinalization` already contains
`temperature_correction_result`; Chapter 12 does not invent a partial return.
Chapter 13 explains that exact result and the remap.

---

# Chapter 13 — Correction and the Full Numba Iteration

## Chapter question

How do radiative and convective imbalance become a stable structural update, and how are the chapter components ordered, cached, and tested until a declared fixed point is reached?

Open with a formal correction direction that reduces one residual but creates an invalid temperature/grid when applied without safeguards. Then show the exact pass orbit, making carried state as visible as recomputed state.

The chapter's orbit uses the paper's exact schematic notation:

\[
\begin{aligned}
\mathbf{x}^{(n)}
&\longrightarrow
\left(P_{\rm gas},n_e,\{n_{s,r}\},\{n_{\rm mol}\}\right)^{(n)}
\longrightarrow \chi_\nu^{(n)}\\
&\longrightarrow
\left(J_\nu,F_{\rm rad},g_{\rm rad},\kappa_{\rm R},\tau_{\rm R}\right)^{(n)}
\longrightarrow \left(F_{\rm conv},\Delta T\right)^{(n)}\\
&\longrightarrow (T,m)^{(n+1)}
\xrightarrow{\rm remap}\mathbf{x}^{(n+1)},
\qquad
T^{(n+1)}=T^{(n)}+\Delta T^{(n)} .
\end{aligned}
\]

## Consumes

- Validated seed, population/opacity state, and reusable catalog handles from Chapter 11.
- Radiative, correction-accumulator, and convection state from Chapter 12.
- Schema-v4 builder and synthesis population bridge from Chapters 2 and 10.
- Numba/cache concepts from Chapter 2.

## Produces

### Act I records

`TemperatureCorrectionResult` on the **current/native** Rosseland grid:

```text
temperature: float64[L]
flux_error_percent: float64[L]
flux_derivative: float64[L]
flux_temperature_derivative: float64[L]
lambda_temperature_derivative: float64[L]
surface_temperature_derivative: float64[L]
temperature_correction: float64[L]
flux_ratio: float64[L]
convective_flux: float64[L]
column_mass: float64[L]
column_mass_correction: float64[L]
```

`IterationRemap`:

```text
finalization: IterationFinalization
atmosphere: ModelAtmosphere
standard_rosseland_optical_depth: float64[L]
integrated_radiation_pressure: float64[L]
turbulent_pressure: float64[L]
```

`IterationFinalization` retains transfer, \(\kappa_{\rm R}\), native \(\tau_{\rm R}\), radiative-pressure state, correction result, and optional convection samples/result for trajectory/debug tests.

### Act II records

`AtmosphereRunResult`:

```text
atmosphere: ModelAtmosphere       # terminal fixed-column-quantized columns
iterations_completed: int
converged: bool                   # structural criterion only
diagnostics: dict[str, typed values/tables]
```

The exact base diagnostics written by `run_atmosphere_model` are:

- `supported_branch`, `layer_count`, `frequency_count`,
  `frequency_start_index`, and `frequency_stop_index`;
- `line_selection_enabled`, `detailed_line_enabled`, `molecules_enabled`, and
  `convection_enabled`;
- `deep_layer_relative_temperature_change`,
  `all_layer_relative_temperature_change`,
  `median_absolute_flux_error_percent`,
  `p95_absolute_flux_error_percent`, and
  `maximum_absolute_flux_error_percent`;
- the configured deep/all-layer limits, `enable_convergence_stop`,
  `minimum_iterations_before_convergence`,
  `required_consecutive_converged_iterations`, and
  `consecutive_converged_iterations`;
- `setup_seconds`, `total_seconds`, and exact per-pass dictionaries in
  `iteration_timings`.

Optional product/debug work adds the exact keys
`structured_atmosphere_path`, `structured_atmosphere_source`,
`structured_atmosphere_warning`, `structured_atmosphere_seconds`,
`diagnostic_runtime_structured_atmosphere_path`,
`diagnostic_runtime_structured_population_source`, and warning variants when
those paths are requested. Catalog/source fingerprints and trial identity are
important editorial provenance, but they are not claimed as current
`AtmosphereRunResult.diagnostics` fields unless a later exact implementation
adds them.

The exact `supported_branch` values are composed as
`one_iteration_{molecule_label}_line_opacity`,
`fixed_iteration_{molecule_label}_line_opacity`, or
`fixed_iteration_{molecule_label}_no_lines`, where `molecule_label` is
`molecules` or `no_molecules`.

The schema-v4 product helper writes only when `converged=True` and records that populations were rebuilt from final fixed-column-quantized columns.

## Prose and derivation arc

### Act I — Correction and complete remap

1. Define target integrated Eddington flux and radiative-plus-convective imbalance.
2. Explain convective-flux endpoint handling and the exact smoothing stencil.
3. Derive the column-correction coefficient and integrating factor.
4. Derive the optical-depth correction and its \([-\tau/3,+\tau/3]\) clamp.
5. Separate three temperature contributions: flux/integral, lambda diagonal, and surface.
6. Explain \(T_{\rm eff}/25\) clipping, previous-step damping, finite replacement, optional local smoothing, and the 1 K inward monotonicity rule in exact order.
7. Convert the old/new total-pressure response into a column-mass correction.
8. Perform the complete remap and enumerate every field once.
9. Explicitly show why correction/remap does not quantize.

### Act II — Exact Numba iteration

10. Draw loop-carried versus pass-local state.
11. Assemble one pass from Chapters 11–13 in source-faithful order.
12. Explain why hydrostatic update starts at pass 2.
13. Distinguish per-pass caches, process caches, and persistent Numba cache artifacts.
    `configure_numba_cache` honors an existing `NUMBA_CACHE_DIR`, otherwise
    `PAYNE_ZERO_NUMBA_CACHE_DIR`, otherwise the exact
    `default_numba_cache_dir`; it also updates live `numba.config.CACHE_DIR`
    when Numba is already imported.
14. Inventory real parallel axes without re-teaching decorators:
    - depth-parallel atomic population/EOS;
    - frequency-parallel continuum;
    - line-catalog chunks;
    - frequency-chunk transfer;
    - sequential molecular continuation and fixed-order reductions.
15. Define deep structural convergence over zero-based `[39:L-5]` on the 80-layer grid:
    \[
    \epsilon_{\rm deep}=\max_i\frac{|T_i^{\rm out}-T_i^{\rm in}|}{|T_i^{\rm out}|}.
    \]
    Smaller grids use all layers.
16. Define the optional symmetric all-layer norm.
    The runner calls
    `max_normalized_column_delta(before, after, floor=1.0, symmetric=True)`, so its
    denominator is
    \(\max(|T_i^{\rm in}|,|T_i^{\rm out}|,1\,{\rm K})\).
17. Explain minimum iterations and consecutive qualifying passes.
18. Separate structural convergence from flux and later scientific acceptance.
19. Apply terminal quantization once.
20. Rebuild product populations after terminal quantization.
21. Prewarm representative branches and record source/runtime/data fingerprints.

## Bite-size canonical code artifacts

### Act I

1. `apply_temperature_correction(state, *, mode, frequency_weight, column_mass, total_opacity, monochromatic_eddington_flux, mean_intensity_minus_source, monochromatic_optical_depth, planck_source, frequency_hz, h_over_kt, temperature_k, stimulated_emission, scattering_fraction, target_integrated_eddington_flux, effective_temperature, frequency_count, rosseland_optical_depth=None, rosseland_opacity=None, iteration_index=1, convection_enabled=False, convective_flux=None, previous_convective_flux=None, logarithmic_temperature_pressure_gradient=None, adiabatic_gradient=None, pressure_scale_height=None, total_pressure=None, mass_density=None, log_density_temperature_derivative_at_constant_total_pressure=None, heat_capacity=None, mixing_length=1.0, smooth_start_layer=0, smooth_stop_layer=0, smooth_left_weight=0.3, smooth_center_weight=0.4, smooth_right_weight=0.3, integrated_radiation_pressure=None, turbulent_pressure=None, surface_gravity_cgs=1.0e4, standard_log_tau_step=0.125, standard_log_tau_start=-6.875) -> TemperatureCorrectionResult | None`
2. the exact private `_pressure_on_standard_depth_grid` block, shown only
   in place inside `apply_temperature_correction(mode=3)` and not promoted to
   a textbook API
3. `finalize_transfer_state` with the exact Chapter 12 signature, revisited
   for its mode-3 correction output
4. `remap_finalized_iteration_state(finalization, *, convective_flux=None, convective_velocity=None, turbulent_pressure=None, completed_iterations=None, standard_log_tau_step=0.125, standard_log_tau_start=-6.875) -> IterationRemap`

The flux, lambda, surface, damping, monotonicity, and column-mass pieces are
short annotated excerpts inside the exact `apply_temperature_correction`; they
must not be factored into newly named textbook package functions.

### Act II

5. exact private `_copy_iteration_atmosphere(atmosphere, *, gas_pressure=None) -> ModelAtmosphere`
6. `deep_layer_relative_temperature_change(before, after) -> float`
7. `max_normalized_column_delta(before, after, *, floor=1.0e-300, symmetric=False) -> float`
8. `temperature_changes_within_limits(*, deep_layer_change, all_layer_change, maximum_deep_layer_change, maximum_all_layer_change) -> bool`
9. `finalize_remapped_iteration(remapped_iteration, *, iterations_completed, converged=False, diagnostics=None) -> AtmosphereRunResult`
10. `run_atmosphere_model(config: AtmosphereConfig) -> AtmosphereRunResult`
11. exact private `_write_debug_state_npz(path, *, remapped_iteration, opacity_state, iterations_completed) -> None`
12. `prepare_structured_handoff_population_state(config, *, temperature_iteration_index=1, setup=None, molecular_thermal_energy_erg=None) -> AtmospherePopulationState`, used only for the optional runtime diagnostic handoff
13. `save_product_structured_atmosphere(atmosphere, output_npz, *, source_catalog_root=None, molecular_lines=True, device="cpu", dtype="float64") -> Path`, the product route that calls synthesis `build_structured_atmosphere` from final columns
14. `configure_numba_cache() -> Path`
15. `prewarm(*, out_dir: Path, force: bool=False) -> dict`

The exact source owns the consecutive counter and pass-local preparation
inline inside `run_atmosphere_model`; the textbook annotates those blocks
without inventing `prepare_iteration_input`, `run_one_atmosphere_pass`, or
`update_consecutive_convergence_state` APIs.

## Exact correction ordering

1. Differentiate temperature, logarithmic gradient, and Rosseland opacity with respect to column mass.
2. Copy current convective flux, zero top boundary layers, and apply the fixed smoothing stencil.
3. Form radiative-heating derivatives and convection-sensitive correction coefficients.
4. Integrate to obtain the integrating factor.
5. Form normalized integrated flux error.
6. Integrate optical-depth correction and clamp to \([-\tau/3,+\tau/3]\).
7. Form the flux-temperature term.
8. Form the lambda-diagonal term with shallow/radiative activation and neighbor taper.
9. Clip local terms to \(T_{\rm eff}/25\).
10. Compute surface term and adjust it with steps sampled at \(\tau=0.1\) and \(2\).
11. Sum the three terms.
12. Apply previous-step sign/magnitude damping in the exact branch order and update `previous_temperature_correction`.
13. Apply correction, replace nonfinite new values by old values, and enforce \(T\ge1\) K.
14. Apply configured local smoothing.
15. Walk from the inner boundary outward and enforce the 1 K inward rise.
16. For the column-mass response, form the exact separate array
    `temperature_plus_correction = temperature + temperature_correction`;
    this uses the damped raw correction, not the later smoothed/monotonic
    `new_temperature` returned in `TemperatureCorrectionResult.temperature`.
17. Remap old `temperature` and `temperature_plus_correction` to the standard
    grid and solve their old/new total-pressure columns.
18. Remap fractional pressure change to the native grid and correct column
    mass.

## Exact complete-remap ordering

1. construct standard \(\tau_{\rm R}\);
2. use native \(\tau_{\rm R}\) as source coordinate;
3. remap corrected column mass and temperature;
4. remap current gas pressure and electron density;
5. remap finalized Rosseland opacity;
6. remap integrated radiation pressure;
7. remap microturbulence and turbulent pressure;
8. remap radiative acceleration;
9. copy supplied convection flux, replace interior `[1:-1]` with the correction's smoothed convective flux, then remap;
10. remap convective velocity;
11. update surface-radiation-pressure and iteration metadata;
12. return `IterationRemap`.

No quantizer appears in this list.

## Exact loop ordering

Initialization:

1. resolve/validate setup and reject unsupported branches;
2. set previous remap and prior Rosseland table to `None`;
3. read seed surface-radiation-pressure constant;
4. copy seed thermal energy as molecular reference;
5. initialize persistent correction state;
6. set selected/detailed catalogs to `None`;
7. initialize iteration, consecutive-convergence, and timing state.

Each pass:

1. update the exact counter as
   `iteration_itemp += iteration_index` (giving 1, 3, 6, ...);
2. choose seed copy or hydrostatically updated previous remap;
3. build pass-local setup with carried surface radiation pressure;
4. prepare population state;
5. prepare opacity state with previous lookup/catalogs;
6. carry returned catalog handles;
7. accumulate transfer;
8. call `finalize_transfer_state` with
   `temperature_iteration_seed=iteration_itemp * 10`, finalize radiation,
   ingest lookup, compute convection, and apply correction;
9. when convection is disabled, compute only disabled-convection diagnostics;
10. remap complete state;
11. carry lookup and new surface-radiation-pressure constant;
12. compare pass input temperature with remapped output;
13. record flux-error distribution;
14. update consecutive counter only when minimum-pass eligibility is met;
15. append timings;
16. break when required consecutive count is reached.

Terminal:

1. require at least one completed remap/opacity/transfer state;
2. assemble diagnostics;
3. fixed-column format/parse terminal remapped atmosphere exactly once;
4. return `AtmosphereRunResult`;
5. optionally write debug state;
6. if `config.outputs.structured_atmosphere_path` is not `None`, unlink any
   existing target before product gating;
7. if unconverged, do not write schema v4 and record
   `structured_atmosphere_warning`;
8. if converged, rebuild populations from terminal quantized fields and write
   schema v4, recording
   `structured_atmosphere_source="final_fixed_column_quantized_arrays"`.

## Data

- Analytic correction fixture.
- One-pass and multi-pass hot, solar, giant, and cool molecule-rich goldens.
- Compact data for smoke tests; optional full catalogs for full-physics trajectory gates.
- `prewarm` exact `REPRESENTATIVE_BRANCHES`:
  - `hot`: 9000 K, \(\log g=4.0\), `[M/H]=0.0`,
    `[alpha/M]=0.0`, 2 km s\(^{-1}\), molecules on;
  - `sun`: 5777 K, 4.44, 0.0, 0.0, 2 km s\(^{-1}\), molecules on;
  - `giant`: 4500 K, 2.0, -0.5, +0.2, 2 km s\(^{-1}\), molecules on;
  - `sun_atomic_only`: solar labels, molecules off, selected/detailed line
    flags forced off, and `numba_threads=1`.
  Each completes exactly one pass and is recorded in the prewarm manifest.

## Checks shown

- Plot each of the three correction terms separately.
- Verify clamps/damping branch behavior and finite/positive/monotone temperatures.
- Corrected/remapped column mass remains positive/monotone.
- Constant remap and all-nine-column coordinate alignment.
- Assert no quantizer call during correction/remap.
- Two-pass call trace proves exact stage order and hydrostatic update only on pass 2.
- Catalog creation/load counters equal one.
- Correction history/lookup persist; pass accumulators reset.
- Exactly one terminal format/parse call.
- Deep-slice edge test and minimum/consecutive Boolean-sequence test.
- Unconverged result writes no product.
- A stale-runtime sentinel proves product populations are rebuilt.
- Python, first `njit`, warm `njit`, and `prange` timings are reported separately for one existing representative kernel.
- Same-process and fresh-process cache reuse are distinguished.

## Strict parity ladder

1. Correction intermediate arrays: flux error, derivatives, three terms, final correction, flux ratio, smoothed convection, column correction.
2. Complete one-pass corrected/remapped states across four regimes.
3. Exact one-pass stage order.
4. Multi-pass trajectories and carried state.
5. Stopping pass, norms, and consecutive counter.
6. Terminal fixed-column-quantized state and metadata.
7. Converged schema keys/shapes/dtypes/arrays and population rebuild.
8. Warm-cache and declared thread-count envelope.

Keep low- and high-level defaults distinct. `AtmosphereConfig` has the source
defaults listed in Chapter 11. The exact high-level
`solve_structured_atmosphere` defaults are `iterations_per_trial=15`,
`max_trials=2`, `initializer_seed=20260713`, and
`initializer_jitter_scale=0.01`; it constructs a config with molecules,
convection, and convergence stopping enabled, minimum 3 iterations, one
required consecutive pass, and deep threshold `0.0005`.

## Failure modes and boundaries

- A small correction or structural change can coexist with large flux error.
- Clamps/damping can slow convergence; removing them changes the method.
- Nonmonotone native/remapped coordinates are hard failures.
- A nonfinite correction replacement is diagnostic evidence and cannot pass strict physical acceptance.
- Exhausting the pass budget returns `converged=False`.
- Structural convergence does not imply flux closure or spectral parity.
- Cool, high-gravity, molecule-rich states can leave the basin.
- Fixed thread count is required for strictest trajectory reproducibility; alternate counts may differ by a few ulp.
- Missing enabled data/unsupported physics fails before product promotion.
- `_require_supported_run_setup` rejects
  `RunSetup.turbulence.enabled=True` and zero-based
  `opacity_flags[13] == 1` (`HLINOP` hydrogen wings).
- NLTE remains outside the scientific scope, but it is not evidence for an
  invented raw-molecular-selector guard. Converted atomic, diatomic, TiO, and
  water selection is active in the standard atmosphere route, while atmosphere
  H3+ is an explicit-path opt-in.
- Only the exact turbulent-pressure and HLINOP guards above are described as
  failing loudly. The separate synthesis H2O compiler remains unwired in the
  standard synthesis runtime; this does not disable atmosphere water opacity.
- A declared diagnostics path with no implemented file product must not be documented as producing one.

## End contract

Chapter 13 produces a terminal quantized `AtmosphereRunResult`, complete diagnostics, optional debug state, and a converged-only schema-v4 atmosphere. `converged` means the declared structural fixed-point criterion only. The chapter accepts explicit seeds and does not know how learned candidates are generated.

---

# Chapter 14 — Learned Initializers and Mandatory Physical Closure

## Chapter question

Can a learned model place an atmosphere inside the physical solver's convergence basin—including unusual abundance mixtures—without being mistaken for the physical fixed point?

Begin with

\[
\mathbf{x}^{(n+1)}=\mathcal{G}\!\left(\mathbf{x}^{(n)}\right),
\qquad
\mathbf{x}_\star=\mathcal{G}\!\left(\mathbf{x}_\star\right).
\]

This is the paper's notation: the requested \(T_{\rm eff}\), \(\log g\),
\(\xi\), and \(\{[\mathrm{X}/\mathrm{H}]\}\) are fixed while
\(\mathbf{x}^{(n)}\) is the complete carried state. The initializer chooses
\(\mathbf{x}^{(0)}\); it does not prove a fixed point.

Near the fixed point,

\[
\mathbf e^{(n+1)}\simeq
\mathbf J_{\mathcal G}(\mathbf x_\star)\mathbf e^{(n)},\qquad
\mathcal R(\mathbf x)=\mathcal G(\mathbf x)-\mathbf x.
\]

The chapter uses the paper's statement that local convergence requires the
spectral radius of \(\mathbf J_{\mathcal G}\) below unity.

## Consumes

- Chapter 11 seed contract and quantization operator.
- Chapter 13 exact one-seed solver, trajectory diagnostics, and product gate.
- Chapter 2 abundance/provenance records and Torch/dtype conventions.
- Standard 80-layer grid.

No spectrum calculation is needed to validate the initializer itself.

## Produces

### Exact common representation

There is no decoded-seed dataclass. `AtmosphereInitializer.predict` returns
`dict[str, np.ndarray]` with exact keys from `INITIALIZER_OUTPUT_FIELDS`:
`column_mass`, `temperature`, `gas_pressure`, `electron_density`,
`rosseland_opacity`, and `radiative_acceleration`. Each value is
`float64[80]`.

The exact transformed fields in `INITIALIZER_COORDINATE_FIELDS`, in stored
column order, are:

```text
log10_column_mass_increment
log10_temperature_relative_to_grey
log10_gas_pressure
log10_electron_density
log10_rosseland_opacity
asinh_radiative_acceleration
```

They form `(80, 6)` and flatten to 480 coordinates in C order. The retained
PCA representation has 160 coefficients. Torch inference produces the
standardized coefficients in `float32`; coefficient de-standardization, PCA
matrix multiplication, coordinate de-standardization, `reshape(80, 6)`, and
physical decoding run in NumPy `float64`.

Use the paper's exact transform notation:

\[
\mathbf{u}_j=\left(
\log_{10}\Delta m_j,\,
\log_{10}\frac{T_j}{T_{{\rm grey},j}},\,
\log_{10}P_{{\rm gas},j},\,
\log_{10}n_{e,j},\,
\log_{10}\kappa_{{\rm R},j},\,
\operatorname{asinh}\frac{g_{{\rm rad},j}}{s_g}
\right),
\]

with \(\Delta m_1=m_1\),
\(\Delta m_j=m_j-m_{j-1}\) for \(j>1\), and

\[
T_{\rm grey}(\tau_{\rm R})=
T_{\rm eff}\left[\frac34\left(\tau_{\rm R}+\frac23\right)\right]^{1/4}.
\]

The source inverse is equally explicit:

```text
column_mass = cumsum(10**clip(u[:, 0], -30, 30))
temperature = grey_temperature * 10**clip(u[:, 1], -3, 3)
gas_pressure, electron_density, rosseland_opacity
    = 10**clip(u[:, 2:5], -30, 30)
radiative_acceleration
    = acceleration_scale * sinh(clip(u[:, 5], -20, 20))
```

For standardized network output \(\widehat{\mathbf z}\), use the paper's
decoder notation

\[
\widehat{\mathbf c}
=\overline{\mathbf c}+\mathbf{s}_c\odot\widehat{\mathbf z},
\qquad
\operatorname{vec}(\widehat{\mathbf u})
=\overline{\mathbf u}
+\mathbf{s}_u\odot\mathbf{B}_K^{\mathsf T}\widehat{\mathbf c},
\qquad
\mathbf p^{(0)}=\mathcal D(\widehat{\mathbf u}),
\]

while the code expression is
`standardized_coordinates = coefficients @ pca["basis"]`; the transpose
placement depends on how the stored basis is oriented. The text must show
both and state that the checkpoint array shape decides the code orientation.

### Ordinary/CNO artifact

`deterministic_initializer_labels` returns the exact
`tuple[dict[str, float] | None, ...]`. Entry 0 is `None` when the requested
labels already lie inside support; otherwise it is the projected initializer
label dictionary. Later entries are deterministic nearby initializer labels.
For each entry, `emulator_warm_start_model` returns
`tuple[ModelAtmosphere, str]`: the in-memory parsed atmosphere and its exact
fixed-column `deck_text`.

The exact five/eight feature semantics must be read from the release manifest and named in the prose. The family router chooses the smallest family that explicitly represents the requested mixture; it cannot silently discard C/N/O offsets.

The exact constants are:

```text
FIVE_LABEL_CHECKPOINT_FEATURE_FIELDS =
    temperature_ratio_5040_k_over_temperature
    log10_surface_gravity_cgs
    metallicity
    alpha_enhancement
    microturbulence_km_s

CNO8_CHECKPOINT_FEATURE_FIELDS =
    all five fields above
    carbon_enhancement
    nitrogen_enhancement
    oxygen_enhancement
```

### Direct-abundance artifact

Public direct input contains 81 manifest-ordered \([X/H]\) coordinates. Its internal 84 features are:

- \(5040/T_{\rm eff}\);
- \(\log g\);
- microturbulence;
- \([\mathrm{Fe/H}]\);
- 80 non-Fe \([X/\mathrm{Fe}]\) offsets.

The exact solver mixture has 97 active slots; 16 non-public slots inherit iron through explicit sentinels.

The exact low-level `complete_direct_abundance_vector` requires the complete
81-coordinate mapping and returns the authoritative, quantized
`float64[97]` solver/synthesis mixture. The 16
`DIRECT_XH_SENTINEL_ATOMIC_NUMBERS` inherit the quantized iron value.

The only public decoded-structure object is
`DirectAbundanceOptimizerSurrogate`, with exact fields:

```text
optimizer_atmosphere: ModelAtmosphere
effective_temperature: float
log_surface_gravity: float
microturbulence_km_s: float
realized_abundance_vector: float64[97]       # immutable
realized_mixture_sha256: str
deck_sha256: str
surrogate_identity_sha256: str
checkpoint: DirectAbundanceCheckpointProvenance
role: "experimental_direct_xh_optimizer_surrogate"
exact_closure_required: True
is_final_atmosphere: False
```

`DirectAbundanceCheckpointProvenance` has exact fields `path`, `sha256`,
`manifest_path`, `manifest_sha256`, and `release_gate_passed`.
`direct_abundance_warm_start_deck` exposes only a `str`; it deliberately does
not return the parsed atmosphere. `run_direct_abundance_atmosphere` returns
`AtmosphereRunResult` only after the exact solve converges and the realized
97-slot mixture still matches; otherwise it raises.

A high-level CLI/API boundary may fill unspecified public coordinates with
`[Fe/H]`, but the exact low-level initializer still receives all 81.
`DIRECT_XH_SUPPORT` is exactly
`effective_temperature=(4000.0, 10500.0)`,
`log_surface_gravity=(0.7, 5.3)`,
`microturbulence_km_s=(0.5, 4.0)`,
`iron_abundance_relative_to_hydrogen=(-2.5, 0.5)`, and
`element_abundance_relative_to_iron=(-0.5, 0.5)`.
`DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX=0.01`. All three public experimental
entry points default their explicit opt-in flag to `False`, and the retained
checkpoint's `release_gate_passed` is false.

## Prose and derivation arc

### Act I — Fixed points and the shared profile decoder

1. Define basin of attraction, restart trajectory, fixed point, and failed restart using Chapter 13's concrete map.
2. For each of six physical profiles, state units/constraints, forward transform, standardized distribution, inverse transform, and round-trip check.
3. Flatten in declared profile/layer order, standardize, project 480 values to 160 PCA coefficients, reconstruct, reshape, and invert transforms.
4. Explain the explicit Torch `float32` → CPU NumPy → `float64` boundary.
5. State the paper's decoded-profile training objective without pretending it
   runs at inference:
   \[
   \mathcal L=\mathcal L_{\rm prof}
   +\lambda_\nabla\mathcal L_\nabla
   +\lambda_\tau\mathcal L_\tau
   +\lambda_{\rm hse}\mathcal L_{\rm hse},
   \]
   with physical consistency checked against
   \(d\tau_{\rm R}/dm=\kappa_{\rm R}\) and
   \(dP_{\rm gas}/dm\simeq g-g_{\rm rad}\). Five-label/CNO use
   \(\lambda_\nabla=0.1\); direct abundance uses 0.2; all use
   \(\lambda_\tau=\lambda_{\rm hse}=0.05\).
6. Establish validation by exact restart rather than profile loss alone.

### Act II — Five-label and CNO initializers

7. Standardize manifest-ordered features and pass them through the compact SiLU MLP.
8. Route five-label versus CNO family without silent information loss.
9. Treat support projection as an initializer-only query; retain original labels/mixture for exact closure.
10. Generate a deterministic ordered set of nearby initializer queries.
11. Decode and fixed-column-quantize every seed before its first exact pass.
12. Compare profile error, one-pass residual, pass count, trajectory, terminal state, and failure rate as different metrics.

### Act III — Direct abundances and safety

13. Derive iron baseline plus 80 \([X/\mathrm{Fe}]\) coordinates.
14. Make sparse high-level completion explicit and low-level completeness strict.
15. Apply the 0.01-dex lattice before model evaluation and mixture hashing.
16. Build the exact element-aware set encoder: element identity plus linear/quadratic abundance response, latent mapping, and summed element representation.
17. Reuse the common 160-coefficient PCA/profile decoder.
18. Reject unsupported direct inputs; never project to a lower-dimensional family that erases the requested pattern.
19. Tether the 81-vector to the 97-slot exact mixture with immutable hashes and sentinels.
20. Explain why initialized spectra are useful for proposal/optimization but initialized atmospheres are not physical products.
21. Run one exact closure trial with the exact same quantized mixture.

## Bite-size canonical code artifacts

### Common representation

1. exact constants `INITIALIZER_COORDINATE_FIELDS`, `INITIALIZER_OUTPUT_FIELDS`, and `INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH`
2. `AtmosphereInitializer.predict(*, effective_temperature, log_surface_gravity, metallicity, alpha_enhancement, microturbulence_km_s, carbon_enhancement=None, nitrogen_enhancement=None, oxygen_enhancement=None) -> dict[str, np.ndarray]`, annotated in short consecutive excerpts for feature standardization, Torch inference, PCA decoding, `reshape(80, 6)`, and the six exact inverse transforms
3. `AtmosphereInitializer.predict_layer_table(**labels) -> np.ndarray`
4. `atmosphere_prediction_to_layer_table(prediction, *, microturbulence_km_s=2.0) -> np.ndarray`

### Ordinary/CNO

5. `AtmosphereInitializer(checkpoint, *, device)`, the exact wrapper whose `.model` is the SiLU `torch.nn.Sequential`
6. `select_warm_start_family(*, carbon_enhancement=None, nitrogen_enhancement=None, oxygen_enhancement=None, absolute_abundance_offsets=None) -> str`
7. `resolve_cno8_labels(*, metallicity, alpha_enhancement, carbon_enhancement=None, nitrogen_enhancement=None, oxygen_enhancement=None, absolute_abundance_offsets=None) -> dict[str, float]`
8. `cno8_absolute_abundance_offsets(*, metallicity, cno_labels) -> dict[int, float]`
9. `load_atmosphere_initializer(*, checkpoint_path=None, device="cpu") -> AtmosphereInitializer`
10. `deterministic_initializer_labels(*, effective_temperature, log_surface_gravity, metallicity=0.0, alpha_enhancement=0.0, microturbulence_km_s=2.0, carbon_enhancement=None, nitrogen_enhancement=None, oxygen_enhancement=None, absolute_abundance_offsets=None, max_trials=1, seed=20260713, jitter_scale=0.01, checkpoint_path=None, device="cpu") -> tuple[dict[str, float] | None, ...]`
11. `format_warm_start_deck(*, effective_temperature, log_surface_gravity, layer_table, metallicity=0.0, alpha_enhancement=0.0, absolute_abundance_offsets=None, title=None) -> str`
12. `emulator_warm_start_model(*, effective_temperature, log_surface_gravity, metallicity=0.0, alpha_enhancement=0.0, microturbulence_km_s=2.0, carbon_enhancement=None, nitrogen_enhancement=None, oxygen_enhancement=None, absolute_abundance_offsets=None, device="cpu", five_label_path=None, cno8_path=None, initializer_label=None, title=None) -> tuple[ModelAtmosphere, str]`

### Direct abundance

13. exact constants `DIRECT_XH_ATOMIC_NUMBERS`, `DIRECT_XH_SENTINEL_ATOMIC_NUMBERS`, `DIRECT_XH_FEATURE_FIELDS`, `DIRECT_XH_SUPPORT`, and `DIRECT_XH_ABUNDANCE_QUANTIZATION_DEX`
14. `complete_direct_abundance_vector(abundance_by_atomic_number) -> np.ndarray`
15. `retained_direct_abundance_mixture(*, iron_abundance_relative_to_hydrogen, retained_abundance_relative_to_iron_by_atomic_number) -> tuple[dict[int, float], np.ndarray]`
16. `direct_abundance_mixture_sha256(abundance_vector) -> str`
17. `DirectAbundanceInitializer(checkpoint, *, device, provenance)` and `DirectAbundanceInitializer.predict(*, effective_temperature, log_surface_gravity, microturbulence_km_s, abundance_by_atomic_number) -> dict[str, np.ndarray]`
18. `load_direct_abundance_initializer(*, enable_experimental=False, checkpoint_path=None, manifest_path=None, device="cpu") -> DirectAbundanceInitializer`
19. `build_direct_abundance_optimizer_surrogate(*, effective_temperature, log_surface_gravity, microturbulence_km_s, abundance_by_atomic_number, enable_experimental_optimizer_surrogate=False, checkpoint_path=None, manifest_path=None, device="cpu") -> DirectAbundanceOptimizerSurrogate`
20. `DirectAbundanceOptimizerSurrogate.provenance() -> dict[str, object]`
21. `direct_abundance_warm_start_deck(*, effective_temperature, log_surface_gravity, microturbulence_km_s, abundance_by_atomic_number, enable_experimental=False, checkpoint_path=None, manifest_path=None, device="cpu") -> str`
22. `run_direct_abundance_atmosphere(*, exact_config, effective_temperature, log_surface_gravity, microturbulence_km_s, abundance_by_atomic_number, enable_experimental=False, checkpoint_path=None, manifest_path=None, device="cpu") -> AtmosphereRunResult`

The element-aware set encoder is shown inside the exact private
`_set_encoded_model` used by `DirectAbundanceInitializer`; the book does not
introduce an independently named public encoder class.

## Exact ordinary/CNO ordering

1. parse requested labels/mixture;
2. choose compatible family;
3. load/checksum manifest, network, PCA, scaling, transforms, and support;
4. construct features in manifest order;
5. project only initializer labels when outside supported seed space; record requested/projected values;
6. standardize features;
7. run Torch `float32` SiLU inference;
8. move 160 coefficients once to CPU NumPy;
9. inverse PCA to 480 values;
10. reshape in manifest-declared order;
11. invert six transforms;
12. construct `ModelAtmosphere` plus metadata;
13. fixed-column format/parse once;
14. validate and retain the matching `deck_text`; `ModelAtmosphere` has no
    `converged` field and is described as a starting atmosphere.

Deterministic restart:

1. create an ordered candidate list using the public seed/count/jitter policy;
2. decode/quantize candidate;
3. run Chapter 13 with original requested physical labels/mixture;
4. retain full diagnostics;
5. stop at first structural convergence or exhaust candidates;
6. never average seeds or select by an unverified spectrum.

The public default can use two candidate trials with the pinned pseudorandom seed and 0.01 initializer-space jitter; values belong in one config table and machine-readable tests.

## Exact direct-abundance ordering

1. parse labels and abundance input;
2. only at high-level boundary, fill omitted public abundances with iron and record them;
3. at low-level boundary, require complete 81-vector in manifest order;
4. quantize to 0.01-dex lattice with declared rounding;
5. form iron baseline and 80 offsets;
6. build 97-slot mixture and apply 16 explicit sentinels;
7. for the optimizer route, compute `realized_mixture_sha256`, `deck_sha256`,
   and `surrogate_identity_sha256`;
8. validate `DIRECT_XH_SUPPORT` and reject if outside;
9. run the exact `_set_encoded_model` and coefficient decoder in Torch
   `float32`;
10. cross once to CPU, inverse PCA, `reshape(80, 6)`, and apply the same six
    transforms;
11. construct and deck-quantize through `_decode_direct_abundance_start`;
12. expose either the provenance-marked deck string or the immutable
    `DirectAbundanceOptimizerSurrogate`;
13. for a physical result, call `run_direct_abundance_atmosphere` with the same
    complete 81-coordinate mapping and exact `AtmosphereConfig`;
14. if closure fails, that public route raises and returns no
    `AtmosphereRunResult`.

## Data

- Immutable ordinary/CNO/direct bundles: weights, PCA basis/mean/scale, feature scaling/order, six transforms, support, architecture/family/release metadata, hashes.
- Small checked input/coefficient/decoded-seed fixtures.
- Direct element order/identities, sentinel policy, lattice rule, and experimental flag.
- Golden physical restart trajectories.
- Training corpora are not runtime dependencies.

## Checks shown

- Six transform round trips.
- PCA encode/decode independent of network.
- Sentinel coefficient lands in expected profile/layer after reshape.
- Deterministic Torch inference.
- Finite/positive/monotone quantized seed.
- Requested labels remain unchanged in exact config when initializer query is projected.
- Deterministic candidate order across processes.
- Family routing uses CNO model when required.
- Sparse completion reports inherited values.
- Low-level direct vector rejects missing/duplicate/nonfinite/unknown entries.
- Half-step lattice behavior exact.
- One-element 0.01-dex change affects correct set member/mixture slot/hash.
- Set encoding invariant to record order only when identity-value pairs stay together.
- Sixteen sentinels inherit iron exactly.
- Initialized states cannot set `converged=True` or write product schema.

## Strict parity gates

- Manifest/checkpoint/PCA hashes exact.
- Feature order/scaling, network output, PCA reconstruction, reshape, inverse transforms independently pinned.
- Quantized five/CNO/direct seeds match goldens.
- Ordinary family routing and support projection match release policy.
- Restart trajectory and terminal exact state satisfy Chapter 13 gates.
- Direct 81/84/97 layouts, sentinels, lattice, set encoder, and hashes match manifest.
- Unsupported direct points reject rather than project/fallback.
- Direct exact closure uses one trial and identical mixture hash.
- Only the exact solver's converged terminal state can become a physical product.

## Failure modes and boundaries

- Smooth-looking decoded profiles can lie outside the exact convergence basin.
- Initializer-only projection is not permission to solve a different star.
- Asset/hash/shape/order mismatch is a hard error.
- Cool high-gravity molecular retries may all fail.
- Direct model remains experimental even if its exact closure succeeds.
- Sparse convenience input is not the low-level contract.
- Unsupported direct mixtures cannot fall back to a family that discards their pattern.
- Good initialized spectral agreement does not prove hydrostatic or radiative-convective closure.
- `run_direct_abundance_atmosphere` requires
  `exact_config.enable_convergence_stop=True` and
  `exact_config.iterations >=
  minimum_iterations_before_convergence +
  required_consecutive_converged_iterations - 1`.

## End contract

Chapter 14 produces the exact ordinary/CNO
`tuple[dict[str, float] | None, ...]`, exact
`tuple[ModelAtmosphere, str]` warm starts, or the exact direct
`DirectAbundanceOptimizerSurrogate`/deck boundary. Physical direct output is
available only through `run_direct_abundance_atmosphere`. The physical solver
remains authoritative.

---

# Chapter 15 — From Stellar Labels to a Verified Spectrum

## Chapter question

What sequence of typed transformations and independent checks turns stellar labels into either a fast exploratory spectrum or a verified physical-atmosphere spectrum?

The two workflows must be visually and programmatically distinct:

```text
exploratory:
labels/mixture → initialized, unconverged atmosphere → supported in-memory bridge
               → Chapter 10 synthesis → LabelSpectrum

verified:
labels/mixture → initialized seed(s) → Chapter 13 exact solve
               → terminal quantization → rebuilt schema-v4 atmosphere
               → Chapter 10 synthesis → Spectrum + editorial acceptance table
```

## Consumes

- Chapter 10 synthesis entry point and outputs.
- Chapter 13 exact solver, diagnostics, and converged product writer.
- Chapter 14 ordinary/CNO candidate generation and direct-abundance state/closure.
- Chapter 2 schema validator, abundance records, and provenance.

## Produces

The exact exploratory objects are `ForwardTimings`, `InitializedAtmosphere`,
and `LabelSpectrum`.

```text
ForwardTimings
    initializer_seconds: float
    population_bridge_seconds: float
    synthesis_seconds: float
    total_seconds: float

InitializedAtmosphere
    structured_atmosphere: dict[str, np.ndarray]
    initializer_family: str
    labels: dict[str, object]
    provenance: dict[str, object]
    timings: ForwardTimings
    atmosphere_converged: False
    atmosphere_closure_required: True

LabelSpectrum(Spectrum)
    wavelength_nm: np.ndarray
    flux_total: np.ndarray
    flux_continuum: np.ndarray
    normalized_flux: np.ndarray
    seconds: float
    initializer_family: str
    labels: dict[str, object]
    provenance: dict[str, object]
    timings: ForwardTimings
    initialized_atmosphere: InitializedAtmosphere
    atmosphere_converged: False
    atmosphere_closure_required: True
```

The exact converged high-level route does not return a new wrapper dataclass.
`solve_structured_atmosphere` returns the `Path` of
`payne_zero_structured_atmosphere.npz`; `validate_atmosphere_npz` returns a
`tuple[str, ...]` of validated array names; `synthesize` returns exact
`Spectrum`:

```text
Spectrum
    wavelength_nm: np.ndarray
    flux_total: np.ndarray
    flux_continuum: np.ndarray
    normalized_flux: np.ndarray
    seconds: float
```

`flux_total` and `flux_continuum` are surface spectral flux densities per
nanometer; `normalized_flux` is dimensionless. These exact names replace
generic “total flux”/“continuum flux” variables in code.

The chapter's acceptance table is an editorial/test artifact, not a claimed
public class. It reports independent statuses for:

- initializer support and asset integrity;
- structural fixed-point convergence;
- flux-error acceptance;
- hydrostatic residual acceptance;
- optical-depth/grid acceptance;
- schema-v4 validation;
- trajectory parity when a golden exists;
- wavelength, total-flux, continuum-flux, and normalized-flux parity;
- backend/cache/thread/data identities;
- supported-physics and limitation flags.

The exact exploratory types carry immutable false/true closure flags.
`InitializedAtmosphere.save_npz(path) -> tuple[str, ...]` may persist the
bridge with `atmosphere_product_role="learned_initializer_prediction"` and
the same safety flags; this is an explicitly unconverged initializer artifact,
not physical-product promotion. `LabelSpectrum` inherits
`Spectrum.save_npz(path) -> None`.
The exact physical API prevents product promotion differently:
`run_atmosphere_model` does not write `structured_atmosphere_path` when
`converged=False`, and `solve_structured_atmosphere` raises after exhausting
trials instead of returning a path.
`load_atmosphere_product_metadata` may return `None` for a valid converged
schema-v4 product without the optional product-metadata extension; the chapter
must not misclassify that as schema failure.

## Prose and workflow arc

1. Choose exploratory versus physical workflow by the scientific question.
2. Route initializer:
   - ordinary five-label;
   - CNO-aware;
   - supported experimental direct abundance;
   - explicit/grey seed when learned assets are unavailable/inappropriate.
3. Expose every projection, filled abundance, retry, disabled branch, and missing optional-data condition in provenance.
4. For physical workflow, call Chapter 13; do not inline its loop.
5. For ordinary/CNO, try deterministic candidates in order. For direct abundance, run one exact trial.
6. Build schema v4 only from converged terminal quantized columns and recomputed populations.
7. Call Chapter 10 synthesis once with declared backend/device/dtype.
8. Evaluate independent atmosphere and spectrum gates.
9. Run the same report for hot, solar, low-gravity giant, and cool molecule-rich cases.
10. Separate initializer latency, atmosphere cold/warm time, synthesis cold/warm time, and data/cache availability.
11. End with a compact model card: LTE, one-dimensional static atmosphere, local mixing-length convection, molecular convergence limits, thread sensitivity, catalog coverage, and water-line status.

## Bite-size canonical code artifacts

1. `initialize_atmosphere_from_labels(*, effective_temperature, log_surface_gravity, metallicity=0.0, fe_over_h=None, alpha_enhancement=0.0, microturbulence_km_s=2.0, c_over_m=None, n_over_m=None, o_over_m=None, x_over_h=None, initializer_family="auto", molecular_lines=True, device="auto", dtype="auto", five_label_path=None, cno8_path=None) -> InitializedAtmosphere`
2. `synthesize_from_labels(*, effective_temperature, log_surface_gravity, metallicity=0.0, fe_over_h=None, alpha_enhancement=0.0, microturbulence_km_s=2.0, c_over_m=None, n_over_m=None, o_over_m=None, x_over_h=None, initializer_family="auto", wavelength_start_nm=400.0, wavelength_end_nm=900.0, r_grid=None, resolution=None, molecular_lines=True, device="auto", dtype="auto", spectral_operator=None, five_label_path=None, cno8_path=None) -> LabelSpectrum`
3. `InitializedAtmosphere.save_npz(path) -> tuple[str, ...]`
4. `solve_structured_atmosphere(*, effective_temperature, log_surface_gravity, out_dir, metallicity=0.0, alpha_enhancement=0.0, microturbulence_km_s=2.0, c_over_m=None, n_over_m=None, o_over_m=None, carbon_over_m=None, nitrogen_over_m=None, oxygen_over_m=None, carbon_enhancement=None, nitrogen_enhancement=None, oxygen_enhancement=None, absolute_abundance_offsets=None, initializer="auto", abundance_by_atomic_number=None, iterations_per_trial=15, max_trials=2, initializer_seed=20260713, initializer_jitter_scale=0.01, **element_over_h) -> Path`
5. `validate_atmosphere_npz(path) -> tuple[str, ...]`
6. `load_atmosphere_product_metadata(path) -> dict[str, object] | None`
7. `load_atmosphere_npz(path) -> dict[str, np.ndarray]`
8. `synthesize(atmosphere_npz, *, wavelength_start_nm=400.0, wavelength_end_nm=900.0, resolution=20000.0, molecular_lines=True, device=None, dtype=None, spectral_operator=None) -> Spectrum`
9. `Spectrum.save_npz(path) -> None`, inherited unchanged by `LabelSpectrum`
10. Four short case cells differing only in exact labels/mixture/expected branch metadata.

Hydrostatic, flux, trajectory, and golden-spectrum comparisons are short
chapter/test calculations over exact returned fields; they do not become a
second public workflow API.

Environment variables, cache paths, installation commands, and the complete CLI option table remain in the appendices.

## Exact exploratory ordering

1. call `initialize_atmosphere_from_labels` directly, or let
   `synthesize_from_labels` call it;
2. normalize aliases and choose `initializer_family` exactly as implemented;
3. for direct `x_over_h`, require/fill `fe_over_h` at the high-level boundary
   and route through `build_direct_abundance_optimizer_surrogate`;
4. obtain the fixed-column-quantized warm start and build the exact
   population-bridged `structured_atmosphere`;
5. return `InitializedAtmosphere` with fixed
   `atmosphere_converged=False` and
   `atmosphere_closure_required=True`;
6. synthesize the in-memory state;
7. return `LabelSpectrum` with the same fixed safety flags and exact
   `ForwardTimings`.

## Exact verified ordinary/CNO ordering

1. call the exact public `solve_structured_atmosphere` signature above with `initializer="auto"`;
2. it calls `deterministic_initializer_labels` with exact high-level defaults;
3. for each returned `initializer_label`, it calls
   `emulator_warm_start_model`, creates exact `AtmosphereConfig`, and calls
   `run_atmosphere_model`;
4. each trial stages its structured NPZ in a temporary directory;
5. an unconverged trial is discarded and the next starts;
6. the first converged trial's staged
   `payne_zero_structured_atmosphere.npz` is atomically promoted;
7. exhaustion raises `RuntimeError` with total exact iterations and final deep
   change; it returns no failed-product path;
8. call `validate_atmosphere_npz` and `load_atmosphere_product_metadata`;
9. call `synthesize` to obtain `Spectrum`;
10. assemble the editorial acceptance table from exact low-level diagnostics
    and golden checks; do not claim `solve_structured_atmosphere` returns that
    table.

## Exact verified direct ordering

1. call the exact public `solve_structured_atmosphere` signature above with `initializer="direct-abundance"` and
   either `abundance_by_atomic_number` or the exact `**element_over_h` names;
2. the high-level `_complete_direct_abundances` requires iron and fills
   unspecified public elements from it;
3. `complete_direct_abundance_vector` validates/quantizes the 97-slot mixture;
4. the high-level trial tuple is exactly `(None,)`, so one warm start and one
   exact trial are used;
5. on convergence, validate and synthesize the promoted product as above;
6. retain direct experimental checkpoint/provenance status in the editorial
   report even though the returned physical product itself passed exact
   closure.

## Data

- Four compact requests with exact labels/mixtures.
- Full atmosphere data/catalogs for publication-grade parity when installed.
- Golden trajectories and terminal schema-v4 states.
- Golden wavelength, total, continuum, and normalized flux arrays.
- Backend/cache reference metadata and tolerance profile.
- Every golden identifies source revision, data manifest, thread policy, dtype/device, and catalog hashes.

## Checks shown

- Fixed safety fields on `InitializedAtmosphere` and `LabelSpectrum`, plus
  converged-only file promotion on the physical path.
- Product columns equal terminal quantized fields.
- Product populations are recomputed after quantization.
- Wavelength arrays match before flux comparison.
- `normalized_flux == flux_total / flux_continuum` within the declared
  convention.
- Injected errors show structural, flux, hydrostatic, optical-depth, schema, and spectral failures independently.
- Cold/warm cache runs preserve physical output.
- CPU/CUDA/MPS synthesis comparisons follow Chapter 10 policy.
- Fixed-thread atmosphere reruns meet strictest reproducibility; alternate counts use declared envelope.
- Four regime reports name enabled/disabled molecules, convection, line paths, and initializer family.

## Acceptance matrix

| Gate | Exploratory surrogate | Verified physical product |
|---|---:|---:|
| Input/schema/manifest valid | required | required |
| Initializer support/hash valid | required | required when initializer used |
| Seed finite/positive/monotone | required | required |
| Structural convergence | not required | required |
| Declared flux-error threshold | reported | required |
| Hydrostatic residual | reported when available | required |
| Standard optical-depth grid/monotonicity | required for supported bridge | required |
| Schema-v4 validation | initializer-marked bridge/file only; never a physical-product claim | required |
| Finite, wavelength-aligned spectrum | required | required |
| Golden spectral parity | contextual | required for reference cases |
| Provenance/limitations | required | required |

Numerical tolerances live in one machine-readable profile and the tolerance appendix. The chapter names metrics and interprets them without scattering literals through prose.

## Failure modes and boundaries

- Ordinary retries may all fail.
- Direct initializer can reject unsupported input before exact work.
- Structural convergence can pass while flux or spectrum acceptance fails.
- Unconverged physical-solver state can remain a debug artifact but cannot
  become a physical atmosphere product; a separately saved
  `InitializedAtmosphere` remains explicitly initializer-marked and carries
  its immutable closure-required flag.
- Missing optional full data prevents a full-data parity claim but not compact pedagogy.
- The standard atmosphere workflow includes converted water-line selection and
  opacity deposition; atmosphere H3+ remains an explicit-path opt-in. The
  separate synthesis H2O compiler is verified as a compiler but omitted from
  the standard synthesis runtime.
- Fixed-thread runs are strict reproducibility target; alternate threads can move last bits.
- Cool molecule-rich cases remain most fragile.
- LTE, one-dimensional static geometry, local mixing-length convection, and unsupported pressure/NLTE branches are scientific limits, not just software tasks.

## End contract

Chapter 15 produces either the exact exploratory `InitializedAtmosphere` /
`LabelSpectrum` objects or the exact verified sequence `Path` → validated
schema-v4 arrays/metadata → `Spectrum`, accompanied by an editorial acceptance
table. It adds no new physics kernel or public wrapper type.

---

## Global redundancy and forward-reference audit

### Quantization/remap

- Chapter 11 defines/tests fixed-column quantization and scalar remap behavior.
- Chapter 14 calls quantization once per decoded seed.
- Chapter 13 performs the complete corrected-state remap and terminal quantization.
- Chapter 13 explicitly asserts no quantization during correction/remap or between passes.
- Chapter 15 only verifies the product boundary.

### Line selection

- Chapter 7 alone derives selection/deposition.
- Chapter 11 alone teaches first-pass membership and later object reuse.
- Chapter 13 only carries handles and checks counters.
- Chapters 14–15 report hashes/flags only.

### Transfer

- Chapter 9 alone derives formal transfer/scattering.
- Chapter 12 alone names/resets/accumulates/finalizes atmosphere integrals.
- Chapter 13 consumes correction accumulators without recreating transfer.

### Convection/correction

- Chapters 3–4 own internal-energy microphysics.
- Chapter 12 owns four perturbed EOS solves and mixing-length results.
- Chapter 13 owns correction and complete remap.
- The production source may expose a combined finalizer, but the book retains the physically ordered teaching split: finalized radiation → perturbed thermodynamics/convection → correction → remap.

### Convergence/acceptance

- Chapter 13's `converged` means structural fixed point under configured norms/counters.
- Flux statistics are recorded but do not silently become the default stopping rule.
- Chapter 15 owns flux/hydrostatic/optical/schema/spectral acceptance.
- No CLI, figure, table, or dataclass may broaden `converged` to mean all gates passed.

### Fixed points/PCA/direct abundance

- Chapter 14 introduces fixed-point basin only after Chapter 13 defines the exact map.
- One shared six-profile/480-to-160 PCA decoder serves ordinary, CNO, and direct encoders.
- Direct inputs outside support reject; they never project/fallback to a family that erases mixture detail.
- Chapter 15 routes and calls only.

### Numba/caches

- Chapter 2 teaches syntax and basic timing.
- Earlier physics chapters introduce meaningful depth/frequency/line kernels.
- Chapter 13 revisits only actual atmosphere orchestration: independence, private state, reduction order, sequential molecule work, compile signatures, prewarm, and fingerprints.
- Operational environment/cache-directory details stay in the appendix.

### Forward references

- Chapter 11 may show a supplied learned-origin seed but treats it solely as `ModelAtmosphere`; it does not depend on Chapter 14.
- Chapter 12 depends only on Chapters 1–11.
- Chapter 13 accepts explicit seeds and depends only on Chapters 1–12.
- Chapter 14 depends on Chapter 13's completed exact solver.
- Chapter 15 is the only chapter depending on all prior material.

## Verification ladder

1. Seed: shapes, units, finiteness, monotonicity, metadata, quantization, hydrostatic residual.
2. Population/opacity: EOS, continuum slab, selected membership, catalog identity, line slabs.
3. Radiation: one-frequency terms, chunk accumulators, Rosseland, force/pressure, correction accumulators.
4. Convection: perturbations, restoration, derivatives, flux/velocity.
5. Correction/remap: every correction term, pressure response, complete standard-grid state.
6. One exact pass: call order and all intermediate outputs.
7. Trajectory: carried state, reuse, per-pass fields/counters.
8. Structural convergence: norms, eligibility, consecutive count, terminal pass.
9. Terminal/product: one terminal quantization, converged-only schema, population rebuild.
10. Initializer: assets, features, support, network, PCA, transforms, quantized seed, exact restart.
11. Direct mixture: completion, lattice, set encoder, slots/hashes, one-trial exact closure.
12. Scientific acceptance: flux, hydrostatic, optical depth, schema, spectrum, backend/cache/thread profile.

Each chapter runs the levels it introduces plus a compact regression from prior levels. Chapter 15 runs the complete ladder for all four regimes.

## Editorial checklist

- Open every act with its physical question.
- State units, shapes, axes, dtype, and device before code.
- Keep most code cells 10–30 lines; split orchestration from kernels.
- Name every mutable field as reset, recomputed, reused, or carried.
- Put exact numerical ordering beside the code enforcing it.
- Use a compact fixture before a full-data run.
- Include one physical sanity view and one strict parity gate per act.
- End with exact consumes/produces and honest failure boundaries.
- Grow one canonical package by imports; never re-inline a capstone.
- Keep the presentation current and self-contained, with no migration narrative.

At the end of Chapter 13, a reader can build and inspect a physical CPU atmosphere solver from an explicit seed. At the end of Chapter 14, the reader can generate modern initialized seeds without confusing them with physical closure. At the end of Chapter 15, the reader can choose the right workflow and obtain either an honestly labeled exploratory spectrum or a fully verified physical-atmosphere spectrum.
