# Chapter 11 first-pass contract — Starting and Blanketing an Atmosphere

Status: authoritative complete-spine-first reader contract; no implementation or publication authority  
Pinned Payne Zero commit: `9c44001feae40b85146630499e6f8a5fed42e5af`  
Audience: final-year undergraduate / first-year graduate student  
Canonical title: **Starting and Blanketing an Atmosphere**

## 0. Canonical placement and ownership

Chapter 10 established a complete synthesis boundary for a supplied,
validated schema-v4 atmosphere:

```text
supplied schema-v4 atmosphere
    -> Chapter 10 device synthesis
    -> native wavelength, total F_lambda, continuum F_lambda,
       normalized flux, and elapsed seconds
```

Chapter 11 begins the inverse responsibility. It does not accept a supplied
atmosphere merely because it can make a plausible spectrum. It asks which
state can safely enter one physical atmosphere pass and how that state becomes
the depth-by-frequency opacity field that blocks and redirects radiation:

```text
supplied fixed-column seed
    -> exact ModelAtmosphere parse/format boundary
    -> narrow validation and run-control resolution
    -> exact 80-layer production coordinate
    -> current EOS/population/Doppler state
    -> direct 30,000-frequency atmosphere sampling grid
    -> continuum absorption/scattering/source
    -> first-pass selected/detailed catalog membership
    -> current line-opacity slab
    -> OpacityState for one physical pass
```

This chapter owns:

- the exact `ModelAtmosphere`, `AtmosphereInput`, `AtmosphereOutput`,
  `AtmosphereConfig`, and `RunSetup` interfaces needed before iteration;
- the distinction between source validation, fixed-column quantization, and
  the source's few narrow normalization actions;
- the exact standard Rosseland coordinate used by the production 80-layer
  route;
- the pass-1 structural boundary and the next-pass hydrostatic gas-pressure
  update;
- the standard microturbulence initialization rule;
- fixed-column format/parse quantization as a numerical operator;
- scalar `remap_to_grid` behavior and the all-fields-on-one-coordinate
  contract, while deferring the complete corrected-state remap;
- `AtmosphereRuntimeState` and `AtmospherePopulationState` composition from
  Chapters 3–4;
- the exact effective-temperature-dependent 30,000-point atmosphere
  wavelength grid, its descending frequency coordinate, and its quadrature
  weights;
- the Chapter 5 `ContinuumAtmosphereState` adapter and full atmosphere
  continuum assembly on that grid;
- the 344-column continuum line-selection threshold and its packed-coordinate
  semantics;
- first-pass selected-line generation or loading, optional detailed-transition
  loading, and later-pass object reuse;
- the exact standard atmosphere line-family composition: predicted, observed,
  and high-excitation atomic sources, diatomic sources, TiO, and water, with
  H3+ available only from an explicit path;
- the complete pre-transfer `OpacityState` and its memory/dtype/axis contract;
- one compact, exact, CPU `float64`/Numba blanketing demonstration.

It does **not** own or execute:

- atmosphere-frequency transfer accumulation;
- Rosseland finalization or native Rosseland optical-depth integration;
- radiative pressure, radiative acceleration, surface moment, heating, or
  lambda-diagonal reductions;
- full atomic/molecular specific-internal-energy recomputation for convection;
- finite-difference thermodynamics or mixing-length convection;
- temperature or column-mass correction;
- the complete corrected-state remap;
- convergence, consecutive-pass stopping, terminal quantization, or product
  promotion;
- learned seed generation;
- schema-v4 physical-product writing;
- a new synthesis workflow.

Those boundaries are exact:

```text
Chapter 11 output: OpacityState
    |
    v
Chapter 12: transfer-frequency reductions, radiation, thermodynamics,
            and convection
    |
    v
Chapter 13: correction, complete remap, repetition, structural convergence,
            terminal quantization, and converged-only schema v4
    |
    v
Chapter 10 unchanged: synthesize the final schema-v4 atmosphere
```

Chapter 11 may mention `save_product_structured_atmosphere(...)` only to mark
the deferred Chapter 13 handoff. It must not call that function, serialize a
live `AtmosphereRuntimeState`, or present `OpacityState` as a schema-v4
atmosphere.

There are no detached exercises. Bad-seed cases, quantization, grid-threshold
boundaries, branch switches, catalog reuse, and thread sensitivity are
predictions and checks in the causal narrative.

### 0.1 Complete-spine-first acceptance

The first executable Chapter 11 pass is chapter-complete when it:

1. parses and hash-checks one supplied 80-layer seed;
2. demonstrates the exact failure surface of `validate_atmosphere_seed`;
3. resolves an explicit, supported `AtmosphereConfig`;
4. shows the standard depth grid, narrow microturbulence fill rule,
   next-pass-only hydrostatic helper, fixed-column quantization, and scalar
   remap contract;
5. prepares an exact molecule-enabled `AtmospherePopulationState`;
6. builds one exact 30,000-point continuum-plus-line `OpacityState` from
   compact manifest-owned sources;
7. proves selected-catalog object reuse while recomputing
   atmosphere-dependent opacity for a changed seed, and validates the
   detailed-transition record interface with IFOP(17) inactive;
8. renders the required original schematics and one-claim plots;
9. passes fixed-thread source, state, grid, component, and compact pinned
   parity gates;
10. closes the Chapter 10 prerequisite and Chapter 12 handoff without
    executing future physics.

It does not wait for the optional 6.8 GB full-catalog run, four-regime
trajectory goldens, alternate-thread tolerance authorization, or final
schema-v4 promotion. A wrong interface, wrong axis, wrong population meaning,
wrong grid formula, false catalog-family claim, failed 30,000-point state, or
leakage of Chapter 12/13 work still blocks the first pass.

## 1. The chapter's single question

Open with two apparently smooth 80-layer seeds. They differ only at one depth:
one has strictly increasing `column_mass`; the other turns over by a small
amount. Show both beside continuum-only and line-blanketed opacity for the
valid seed, then ask:

> What exact state can safely enter one physical atmosphere pass, and how does
> its composition become the 30,000-frequency opacity field that produces
> line blanketing?

Make four predictions before running the exact code:

1. a smooth-looking seed with one nonmonotone depth point must fail before any
   compiled physics runs;
2. a fixed-column format/parse round trip may alter the numerical seed but a
   second round trip must not alter the already quantized result;
3. two atmosphere states may reuse the same selected catalog object, but
   their line-opacity values must change when temperature, populations, or
   widths change;
4. adding line opacity can show where radiation will be blocked, but it cannot
   yet establish a temperature correction, back-warming trajectory, or
   converged atmosphere.

The causal sequence is:

```text
nine aligned seed columns + explicit metadata and abundances
    -> parse as ModelAtmosphere
    -> validate exact source conditions
    -> require the chapter's 80-layer/full-mixture teaching fixture
    -> resolve gravity, flags, controls, standard depth coordinate
    -> fill microturbulence only under the exact all-zero condition
    -> optionally cross the declared seed quantization boundary
    -> prepare current populations and Doppler/strength state
    -> build exact atmosphere sampling wavelength/frequency/weight arrays
    -> build continuum adapter and continuum slabs
    -> build the 344-column selection reference
    -> create or load catalog membership once
    -> recompute selected-line opacity; validate the inactive detailed interface
    -> return OpacityState and stop before transfer
```

## 2. Reader promise, assumptions, and honest scope

By the end of the chapter, the reader should be able to:

- distinguish a deck-compatible `ModelAtmosphere`, an internal
  `AtmosphereRuntimeState`, an `AtmospherePopulationState`, an `OpacityState`,
  and a schema-v4 synthesis atmosphere;
- state the units and constraints of all nine aligned `ModelAtmosphere`
  columns;
- explain what `validate_atmosphere_seed` rejects and what it deliberately does
  not repair;
- explain exactly which configuration counts are clamped, which thresholds
  raise, and when microturbulence is filled in-place;
- explain why pass 1 copies the seed and why the hydrostatic gas-pressure
  update first appears at pass 2;
- distinguish `integrated_radiation_pressure`,
  `surface_radiation_pressure_constant`, and the hydrostatic helper's
  `pressure_constant`;
- define fixed-column quantization \(Q\) as
  `parse_atmosphere_deck(format_atmosphere_deck(model), source="<roundtrip>")`;
- explain why \(Q(x)\neq x\) can be expected while \(Q(Q(x))=Q(x)\) is a
  required fixed-point check;
- reproduce both exact atmosphere grids without confusing either with the
  Chapter 10 synthesis grid;
- explain why atmosphere wavelength increases while atmosphere frequency
  decreases;
- interpret `continuum_absorption`, `continuum_scattering`,
  `continuum_source`, and `line_mass_absorption_coefficient` without
  relabeling them as flux;
- explain the 343 active-reference columns plus duplicated column 343 in the
  344-column selection table;
- explain why the exact field `wavelength_bin_edges` contains packed
  wavelength indices rather than physical bin-edge wavelengths;
- distinguish catalog membership from atmosphere-dependent opacity values;
- state the exact family order and the separate selected/detailed provenance;
- identify the exact Numba boundaries that are present and the Python/NumPy
  orchestration that remains uncompiled;
- explain why the returned `OpacityState` is the correct Chapter 12 input but
  is not a converged or synthesis-ready atmosphere.

The supported teaching path is:

- one-dimensional, static, plane-parallel LTE;
- CPU NumPy `float64` high-level atmosphere state;
- exact Numba kernels where the pinned source uses them;
- 80 outer-to-inner depth layers;
- molecule-enabled population closure;
- continuum and selected-line opacity enabled on the normal executable path;
- the detailed-transition catalog schema validated with IFOP(17) inactive;
- compact checksum-bound line sources for the normal notebook;
- a fixed declared Numba thread policy for strict comparison.

The normal notebook does not:

- use `/Users/ysting/payne-zero` at runtime;
- load the 6.8 GB full source-catalog tree;
- call `run_atmosphere_model`;
- call `accumulate_transfer_state` or `finalize_transfer_state`;
- generate a learned seed;
- write a schema-v4 physical product;
- call Chapter 10 synthesis as evidence of atmosphere closure.

## 3. Authoritative source and brief reconciliation

This section corrects statements that are too broad, stale, or ambiguous in
the earlier atmosphere brief and in source docstrings. The implementation and
reader text must follow these narrower statements.

### 3.1 Validation is not a general sanitizer

`validate_atmosphere_seed(atmosphere)`:

- converts each field to a temporary NumPy `float64` view for checking;
- requires all nine aligned fields to have shape `(atmosphere.layers,)`;
- requires all nine fields to be finite;
- requires positive `column_mass`, `temperature`, `gas_pressure`,
  `electron_density`, and `rosseland_opacity`;
- requires strictly increasing `column_mass`;
- requires nonnegative `microturbulence`.

It does not:

- sort depth;
- replace NaNs;
- floor negative seed fields;
- require `radiative_acceleration`, `convective_flux`, or
  `convective_velocity` to have a particular sign;
- require an input dtype to already be `float64`;
- require exactly 80 layers;
- require a complete 99-element abundance block;
- validate the physical consistency of pressure, electron density, opacity,
  or flux;
- prove hydrostatic or radiative equilibrium.

The chapter may use the ordinary-language phrase “validate and narrowly
normalize the supplied seed.” It must not say that the exact source repairs an
arbitrary seed.

### 3.2 The source's narrow normalization actions are exact and visible

`resolve_run_setup`:

- clamps `iterations` to at least one;
- clamps `minimum_iterations_before_convergence` and
  `required_consecutive_converged_iterations` to at least one;
- raises for nonpositive deep/all-layer convergence limits;
- defaults missing `effective_temperature` to 5778 K;
- defaults missing `log_surface_gravity` to 4.44;
- defaults missing/malformed opacity flags to `DEFAULT_OPACITY_FLAGS`;
- defaults missing pressure-iteration metadata to enabled;
- fixes mixing length to 1.25, overshoot weight to 0.0, and turbulence to
  disabled;
- fills `microturbulence` only if **no** element is positive.

The microturbulence fill mutates the supplied mutable `ModelAtmosphere`
in-place even though `AtmosphereConfig` and `AtmosphereInput` are frozen
dataclasses. A partly zero profile with any positive entry is not filled.

The canonical fixture must carry explicit effective temperature, gravity,
opacity flags, pressure control, and all 99 abundance values. The notebook
shows the source defaults as behavior, but does not silently rely on them for
the main scientific state.

### 3.3 The 80-layer production gate is downstream of seed validation

`standard_rosseland_optical_depth_grid(layers)` accepts any integer layer
count. `validate_atmosphere_seed` also accepts any positive common layer
count. The exact selected- and detailed-line opacity routines later require
`layer_count == 80` and raise otherwise.

Therefore:

- the chapter's canonical line-blanketed path explicitly requires 80 layers
  before line opacity;
- a smaller grid is allowed only for isolated seed/remap demonstrations;
- the reader text must not attribute the 80-layer failure to
  `validate_atmosphere_seed`.

### 3.4 Pass 1 does not recompute hydrostatic pressure

In the exact runner:

- pass 1 copies the validated seed;
- pass 2 and later consume the previous **unquantized remapped** atmosphere;
- only then, when pressure iteration is enabled, is gas pressure recomputed
  with `integrate_hydrostatic_pressure`.

Chapter 11 teaches the hydrostatic helper with a controlled previous-support
fixture and analytic limits. It does not call the helper as though it creates
the pass-1 seed.

The exact helper computes

\[
P_{\rm gas}(m)
= gm-P_{\rm rad,int}(m)-P_{\rm turb}(m)-P_{\rm constant}.
\]

The production loop leaves `pressure_constant=0.0`.
`RunSetup.surface_radiation_pressure_constant` is a separate carried scalar
used by total-pressure, convection, and correction logic. It is not passed as
the hydrostatic helper's `pressure_constant`.

If the computed gas pressure is nonpositive, the exact helper warns and
floors those entries. It does not raise. The chapter must show that branch as
a diagnostic failure of the proposed support state, not describe it as a
clean hydrostatic solution.

### 3.5 Turbulent pressure is unsupported even though the helper accepts an array

`integrate_hydrostatic_pressure` subtracts the supplied
`turbulent_pressure` array. The exact supported runner nevertheless constructs
`TurbulenceSettings(enabled=False, ...)`, and `_require_supported_run_setup`
rejects an enabled turbulent-pressure branch.

The canonical Chapter 11 hydrostatic demonstration therefore supplies an
explicit zero turbulent-pressure column. It may show the algebraic argument
position, but it does not claim a supported evolving turbulent-pressure
model.

### 3.6 The standard depth grid is a coordinate, not a solved opacity integral

The exact helper returns

\[
\texttt{standard_rosseland_optical_depth}[i]
=10^{-6.875+0.125i}.
\]

For 80 layers the first value is
`1.333521432163324e-07` and the final value is exactly `1000.0` under the
pinned NumPy `float64` expression. This standard coordinate is used for seed
microturbulence and later remapping. It is not the current/native
`rosseland_optical_depth` that Chapter 12 obtains by integrating the current
Rosseland opacity.

### 3.7 The atmosphere sampling grid is not the synthesis grid

`build_opacity_sampling_grid(effective_temperature)` always returns exactly
30,000 direct host NumPy `float64` samples. Its start index is:

| temperature condition | exact `start_index` |
| --- | ---: |
| \(T_{\rm eff}\ge 30000\) K | 1 |
| \(13000\le T_{\rm eff}<30000\) K | 3577 |
| \(7250\le T_{\rm eff}<13000\) K | 7027 |
| \(4500\le T_{\rm eff}<7250\) K | 9599 |
| \(T_{\rm eff}<4500\) K | 11601 |

The comparisons are strict `<`. Exact threshold values remain in the warmer
branch.

\[
\lambda_i =
10^{1+0.0001(i+\texttt{start_index}-1)}\ {\rm nm},
\qquad i=1,\ldots,30000.
\]

The chapter must preserve:

- increasing `opacity_wavelength_grid_nm`;
- decreasing `opacity_frequency_hz = 2.99792458e17 / wavelength_nm`;
- the exact asymmetric endpoint/interior frequency-weight formulas;
- positive `frequency_weights`;
- no synthesis edge-triplet interpolation;
- no `resolution` or \(R_{\rm grid}\) argument.

The function is `lru_cache(maxsize=16)` and returns mutable NumPy arrays.
Repeated calls at the same effective temperature may return the same array
objects. Chapter code treats them as read-only and copies before any
experiment. It must not call the cached arrays immutable.

### 3.8 The continuum reference is a selection table, not another spectrum

The exact continuum reference has 343 physical source columns plus one
sentinel/duplicate column:

- `continuum_reference_wavelength_nm[:343]` contains the reference
  wavelengths;
- column 343 duplicates column 342;
- `wavelength_bin_edges[:343]` contains packed logarithmic wavelength
  indices;
- `wavelength_bin_edges[343] == 2**30`;
- the returned selection threshold is `float32[D,344]`;
- active columns are those whose reference wavelength lies above the current
  atmosphere grid's first wavelength;
- inactive columns receive the exact large source threshold rather than a
  computed continuum value;
- active thresholds are
  \(10^{-3}(\kappa_\nu^{\rm abs}+\kappa_\nu^{\rm sca})\) divided by the
  stimulated-emission factor.

The exact public field remains `wavelength_bin_edges`, despite holding packed
indices. The book explains the historical name and does not rename it.

### 3.9 `LineOpacityState` dtype is branch-dependent before transfer

`allocate_line_opacity_state(...)` initially allocates a `float64[D,Nnu]`
zero slab. The selected-line compiled path returns a `float32[D,Nnu]` slab.
The detailed-transition path also works in and returns `float32`, casting a
supplied base slab when necessary. A continuum-only state can therefore
retain the initial `float64` zero slab, while selected/detailed states normally
hold `float32`.

The Chapter 12 transfer boundary explicitly converts the line slab to
contiguous `float32`. Chapter 11 must test the actual branch dtype rather than
claim that `OpacityState.line_opacity` has one unconditional dtype.

### 3.10 Population preparation does not yet own convection energy

`prepare_population_state` builds runtime density, charge-square density,
atomic/molecular closure, packed populations, fractional Doppler widths, and
the normalized population-over-density-and-width strength state. It does not
make the standalone `compute_atomic_specific_internal_energy(...)` call used
inside Chapter 12's perturbed thermodynamic states.

`AtmosphereRuntimeState.specific_internal_energy` exists in the record, but
Chapter 11 does not claim that the complete convection-ready internal-energy
column has been finalized. Full energy recomputation remains Chapter 12.

### 3.11 Catalog reuse is carried by the caller, not hidden in the builder

`prepare_opacity_state` has explicit optional arguments
`selected_line_catalog` and `transition_line_catalog`. On the first call:

- it generates selected lines from existing raw sources or loads the supplied
  preselected catalog;
- it loads the detailed-transition catalog when its flag is enabled.

The exact runner stores the two returned objects and passes them into the next
call. `prepare_opacity_state` itself has no global cross-pass catalog cache.
On later calls the same catalog objects are reused, but a fresh line-opacity
slab is accumulated from the new populations and widths.

The normal compact Chapter 11 path keeps IFOP(17) inactive because cold LLVM
compilation of `_accumulate_transition_line_opacity_parallel` is pathological
for an executable reader pass even with one detailed record. It validates the
manifest-owned `LineTransitionCatalog` schema separately and exercises the
exact selected-line full-grid state. Its identity gate is therefore:

```python
second.selected_line_catalog is first.selected_line_catalog
first.transition_line_catalog is None
second.transition_line_catalog is None
```

alongside a demonstrated change in atmosphere-dependent opacity.

### 3.12 Standard family composition and low-level omission behavior differ

The high-level standard workflow obtains:

- predicted atomic shards;
- observed atomic lines;
- high-excitation lines;
- diatomic lines;
- TiO lines;
- water lines;
- detailed transitions.

`source_line_paths()` does **not** return an H3+ path. H3+ is an explicit
`AtmosphereInput.h3plus_lines_path` opt-in.

Two pinned source docstrings are stale:

- `source_line_paths()` says its returned keys include H3+, but its mapping
  does not;
- `run_atmosphere_model` says raw molecular selectors are off, but the exact
  guard rejects only turbulent pressure and zero-based opacity flag 13
  (`HLINOP`), and the standard diatomic/TiO/water selector paths are active.

The low-level selection adapter requires at least one existing raw source when
no preselected catalog is supplied. Individual `None` or nonexistent raw
family paths are skipped if another source exists. The full high-level
`source_line_paths()` route is stricter because it `_require`s every standard
member. The chapter distinguishes these behaviors and explicitly hash-checks
its compact source view; it does not claim that the low-level runner
automatically verifies every source checksum.

### 3.13 `enable_molecules` and molecular line paths are different controls

`AtmosphereConfig.enable_molecules` controls molecular equilibrium in
`prepare_population_state`. The diatomic, TiO, water, and H3+ path fields
control which raw line families are offered to selection. A path alone does
not construct a trustworthy molecular population.

The canonical line-blanketed run sets `enable_molecules=True` and supplies the
atmosphere molecular-equilibrium catalog. Atom-only and molecule-disabled
branches are separate checks, not the primary state.

### 3.14 A live atmosphere state is not schema v4

`ModelAtmosphere` has nine aligned physical columns plus metadata and the
fixed-column abundance mapping. `AtmosphereRuntimeState` adds the packed
atmosphere `(D,1006)` state. `OpacityState` adds depth-frequency opacity and
catalog handles.

None is the Chapter 10 schema-v4 mapping.

The physical product route later calls
`save_product_structured_atmosphere(...)` only after Chapter 13 has:

1. corrected and completely remapped the structure;
2. satisfied the structural stopping criterion;
3. crossed terminal fixed-column quantization once;
4. rebuilt synthesis populations from the final quantized columns.

The live Chapter 11 population arrays must not be serialized as the accepted
schema-v4 physical product.

### 3.15 “One pass” ends before transfer in this chapter

The phrase “the state needed for one pass” means the exact pre-transfer
`OpacityState`. A complete physical pass would continue through Chapter 12
transfer/finalization and Chapter 13 correction/remap. Chapter 11 does not
call the full runner with `iterations=1`, because that would execute future
physics while hiding it behind orchestration.

## 4. Exact notation, axes, units, and coordinates

Let:

- \(D=80\): production depth layers, outermost to innermost;
- \(N_\nu=30000\): atmosphere opacity-sampling frequencies;
- \(N_{\rm ref}\le343\): active continuum-reference columns;
- \(N_{\rm sel}\): selected compact line records;
- \(N_{\rm det}\): detailed-transition records.

All Chapter 11 physical arrays are on CPU. High-level arrays are NumPy
`float64` unless the exact field below states otherwise.

### 4.1 Seed and runtime state

| symbol | physical meaning | exact name | shape, unit, and rule |
| --- | --- | --- | --- |
| \(m\) | column mass | `column_mass` | `(D,)`, g cm\(^{-2}\), positive and strictly increasing inward |
| \(T\) | temperature | `temperature` | `(D,)`, K, positive |
| \(P_{\rm gas}\) | gas pressure | `gas_pressure` | `(D,)`, dyn cm\(^{-2}\), positive |
| \(n_e\) | electron density | `electron_density` | `(D,)`, cm\(^{-3}\), positive |
| \(\kappa_{\rm R,seed}\) | seed/current Rosseland opacity column | `rosseland_opacity` | `(D,)`, cm\(^2\) g\(^{-1}\), positive |
| \(g_{\rm rad,seed}\) | seed radiative acceleration column | `radiative_acceleration` | `(D,)`, cm s\(^{-2}\), finite |
| \(\xi\) | microturbulent velocity | `microturbulence` | `(D,)`, cm s\(^{-1}\), nonnegative |
| \(H_{\rm conv}\) | stored convective flux in solver convention | `convective_flux` | `(D,)`, finite; physical ownership deferred to Ch. 12 |
| \(v_{\rm conv}\) | convective velocity | `convective_velocity` | `(D,)`, cm s\(^{-1}\), finite; deferred to Ch. 12 |
| \(\tau_{\rm R,std}\) | standard remap coordinate | `standard_rosseland_optical_depth` | `(D,)`, dimensionless |
| \(\rho\) | mass density | `AtmosphereRuntimeState.mass_density` | `(D,)`, g cm\(^{-3}\) |
| \(A_Z\) | linear elemental number fraction | `elemental_abundances_by_layer` | `(D,99)`, dimensionless |
| \(n_{s,r}\) | actual packed stage population | `ion_stage_populations_by_packed_slot` | `(D,1006)`, cm\(^{-3}\) |
| \(n_{s,r}/U_{s,r}\) | partition-normalized packed population | `partition_normalized_populations_by_packed_slot` | `(D,1006)`, cm\(^{-3}\) under the source convention |
| \(\Delta\nu_D/\nu\) | fractional Doppler width | `fractional_doppler_widths` | `(D,1006)`, dimensionless |

The fixed-column abundance mapping uses the exact mixed deck convention:

- atomic numbers 1–2: linear number fractions;
- atomic numbers 3–99: base-10 logarithms of number fractions.

The canonical seed carries all 99 entries. The exact runtime builder otherwise
fills missing metal entries with `1.0e-30`, while the later product bridge
requires a complete block. The chapter treats that difference as an explicit
boundary, not a convenient default.

### 4.2 Frequency and opacity state

| symbol | physical meaning | exact name | shape, unit, and dtype |
| --- | --- | --- | --- |
| \(\lambda_i\) | atmosphere sampling wavelength | `opacity_wavelength_grid_nm` | `(30000,)`, nm, increasing `float64` |
| \(\nu_i\) | atmosphere sampling frequency | `opacity_frequency_hz` | `(30000,)`, Hz, decreasing `float64` |
| \(w_i\) | frequency quadrature weight | `frequency_weights` | `(30000,)`, Hz, positive `float64` |
| \(\kappa_{\nu,c}\) | continuum absorption per mass | `continuum_absorption` | `(D,30000)`, cm\(^2\) g\(^{-1}\), `float64` |
| \(\sigma_{\nu,c}\) | continuum scattering per mass | `continuum_scattering` | `(D,30000)`, cm\(^2\) g\(^{-1}\), `float64` |
| \(S_{\nu,c}\) | absorption-weighted continuum source | `continuum_source` | `(D,30000)`, erg s\(^{-1}\) cm\(^{-2}\) sr\(^{-1}\) Hz\(^{-1}\), `float64` |
| \(q_{\ell,\rm ref}\) | line-selection continuum threshold | `continuum_line_selection_threshold` | `(D,344)`, cm\(^2\) g\(^{-1}\) under the exact stimulated-emission convention, `float32` |
| \(\lambda_{\rm ref}\) | reference wavelengths | `continuum_reference_wavelength_nm` | `(344,)`, nm, `float64`, final value duplicated |
| \(p_{\rm ref}\) | packed logarithmic wavelength indices | `wavelength_bin_edges` | `(344,)`, integer code, `int64`; final value `2**30` |
| \(\kappa_{\nu,\ell}\) | selected/detailed line absorption per mass | `line_opacity.line_mass_absorption_coefficient` | `(D,30000)`, cm\(^2\) g\(^{-1}\), branch-dependent `float64`/`float32` |

`continuum_source` is an intensity/source-function quantity, not
`H_nu`, \(F_\nu\), or \(F_\lambda\). No Chapter 11 field is a public spectrum.

### 4.3 Axis and memory contract

All atmosphere opacity slabs are depth-major:

```text
(depth, frequency) == (80, 30000)
```

For \(D=80\):

- one `float64[D,30000]` slab is 19,200,000 bytes;
- the three continuum slabs are 57,600,000 bytes before array overhead;
- one `float32[D,30000]` line slab is 9,600,000 bytes;
- one initial `float64[D,30000]` zero line slab is 19,200,000 bytes;
- the `float32[D,344]` threshold is 110,080 bytes.

The memory lesson reports actual simultaneous liveness. It does not add every
possible array as though all temporaries coexist, and it never allocates a
dense `(D, lines, Nnu)` tensor.

## 5. Exact interfaces and field order

### 5.1 `ModelAtmosphere` and fixed-column I/O

`ModelAtmosphere` fields, in order:

```text
column_mass
temperature
gas_pressure
electron_density
rosseland_opacity
radiative_acceleration
microturbulence
convective_flux
convective_velocity
metadata
fixed_column_abundance_values
```

Exact interfaces:

```python
parse_atmosphere_deck(
    deck_text: str,
    *,
    source: str = "<in-memory deck>",
) -> ModelAtmosphere

format_atmosphere_deck(model: ModelAtmosphere) -> str

remap_to_grid(
    source_grid: np.ndarray,
    source_values: np.ndarray,
    target_grid: np.ndarray,
) -> tuple[np.ndarray, int]
```

The chapter writes no deck to disk during the scientific path. The exact
quantization expression is literal:

```python
quantized = parse_atmosphere_deck(
    format_atmosphere_deck(model),
    source="<roundtrip>",
)
```

The row formatting that defines \(Q\) is:

- column mass: `14.8E`;
- temperature: `8.1f`;
- each remaining physical column: `10.3E`.

### 5.2 Configuration and resolved setup

`AtmosphereInput` fields, in order:

```text
initial_atmosphere
molecules_path
selected_line_catalog_path
detailed_line_catalog_path
predicted_atomic_lines_path
observed_atomic_lines_path
high_excitation_lines_path
diatomic_lines_path
titanium_oxide_lines_path
water_lines_path
h3plus_lines_path
```

`AtmosphereOutput` fields, in order:

```text
structured_atmosphere_path
diagnostics_path
debug_state_path
```

All output paths are `None` in Chapter 11.

`AtmosphereConfig` fields and exact defaults:

```text
inputs
outputs
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

`RunSetup` fields, in order:

```text
atmosphere
iterations
enable_convergence_stop
minimum_iterations_before_convergence
required_consecutive_converged_iterations
maximum_deep_layer_relative_temperature_change
maximum_all_layer_relative_temperature_change
surface_gravity_cgs
opacity_flags
molecules_enabled
pressure_iteration_enabled
convection
turbulence
surface_radiation_pressure_constant
effective_temperature
log_surface_gravity
standard_rosseland_optical_depth
```

Exact functions:

```python
surface_gravity_from_atmosphere(atmosphere: ModelAtmosphere) -> float

surface_radiation_pressure_constant_from_atmosphere(
    atmosphere: ModelAtmosphere,
) -> float

opacity_flags_from_atmosphere(atmosphere: ModelAtmosphere) -> list[int]

standard_rosseland_optical_depth_grid(layers: int) -> np.ndarray

initialize_microturbulence(
    atmosphere: ModelAtmosphere,
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    standard_rosseland_optical_depth: np.ndarray,
) -> None

validate_atmosphere_seed(atmosphere: ModelAtmosphere) -> None

resolve_run_setup(config: AtmosphereConfig) -> RunSetup

integrate_hydrostatic_pressure(
    atmosphere: ModelAtmosphere,
    *,
    surface_gravity_cgs: float,
    integrated_radiation_pressure: np.ndarray,
    turbulent_pressure: np.ndarray,
    pressure_constant: float = 0.0,
) -> np.ndarray
```

`DEFAULT_OPACITY_FLAGS` is exactly:

```text
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
```

Zero-based flag 14 enables selected-line opacity. Zero-based flag 16 enables
detailed-transition opacity. Zero-based flag 13 is the unsupported HLINOP
branch.

### 5.3 Population state

`AtmosphereRuntimeState` fields, in order:

```text
gas_pressure
electron_density
total_nuclei_number_density
mass_density
charge_square_density
elemental_abundances_by_layer
mean_nuclear_mass_amu
ion_stage_populations_by_packed_slot
partition_normalized_populations_by_packed_slot
specific_internal_energy
major_isotope_mass_amu
fractional_doppler_widths
partition_normalized_population_over_mass_density_and_fractional_doppler_width
hydrogen_departure_coefficients
metal_departure_coefficients
geometric_depth_below_surface_km
```

`AtmospherePopulationState` fields, in order:

```text
setup
runtime_state
fractional_doppler_widths
partition_normalized_population_over_mass_density_and_fractional_doppler_width
temperature_iteration_cache
molecular_state
```

Exact entry point:

```python
prepare_population_state(
    config: AtmosphereConfig,
    *,
    temperature_iteration_index: int = 1,
    setup: RunSetup | None = None,
    molecular_thermal_energy_erg: np.ndarray | None = None,
) -> AtmospherePopulationState
```

Chapter 11 reuses the Chapter 3–4 EOS/molecular functions and Chapter 6
Doppler/strength functions. It does not display or fork their implementations.

### 5.4 Continuum and line records

`ContinuumAtmosphereState` has the exact 18 fields:

```text
temperature
mass_density
electron_density
gas_pressure
hydrogen_partition_normalized_ion_stage_populations
hydrogen_neutral_population
hydrogen_ionized_population
helium_neutral_population
helium_singly_ionized_population
helium_neutral_partition_normalized_population
helium_singly_ionized_partition_normalized_population
elemental_abundances_by_layer
hydrogen_departure_coefficients
microturbulence
ion_stage_populations_by_packed_slot
partition_normalized_populations_by_packed_slot
ch_population
oh_population
```

`ch_population` and `oh_population` are exact aliases of
partition-normalized packed slots 845 and 847. They are not renamed as generic
actual molecule densities.

`RosselandOpacityTable` fields, in order:

```text
normalized_log_temperature
normalized_log_pressure
log10_rosseland_opacity
entry_count
log_temperature_origin
log_pressure_origin
log_temperature_span
log_pressure_span
```

The first Chapter 11 opacity state uses an empty table with capacity
`D * 60` and `entry_count == 0`. Chapter 12/13 own ingestion and persistence.

`SelectedLineCatalog` fields, in order:

```text
packed_wavelength_index
packed_species_slot
lower_excitation_index
log_strength_index
radiative_damping_index
stark_damping_index
van_der_waals_damping_index
```

`LineTransitionCatalog` fields, in order:

```text
vacuum_wavelength_nm
lower_excitation_cm
oscillator_strength
lower_hydrogen_level
upper_hydrogen_level
packed_species_slot
line_type
hydrogen_continuum_selector_index
continuum_species_slot
radiative_damping
stark_damping
van_der_waals_damping
packed_wavelength_index
line_limit
```

`LineOpacityState` fields, in order:

```text
line_mass_absorption_coefficient
selected_line_count
```

Exact continuum/grid entry points:

```python
build_opacity_sampling_grid(
    effective_temperature: float,
) -> tuple[np.ndarray, np.ndarray]

build_continuum_atmosphere_state(
    atmosphere: ModelAtmosphere,
    state: AtmosphereRuntimeState,
) -> ContinuumAtmosphereState

active_continuum_reference_frequencies(
    effective_temperature: float,
) -> tuple[np.ndarray, np.ndarray]

assemble_continuum_line_selection_threshold(
    *,
    effective_temperature: float,
    temperature_k: np.ndarray,
    active_continuum_absorption: np.ndarray,
    active_continuum_scattering: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]

compute_continuum_opacity_columns(
    atmosphere: ContinuumAtmosphereState,
    frequency_hz: np.ndarray,
    *,
    opacity_flags: list[int] | tuple[int, ...] | None = None,
    rosseland_table: RosselandOpacityTable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Exact catalog/line entry points:

```python
generate_selected_lines(
    *,
    partition_normalized_population_over_mass_density_and_fractional_doppler_width,
    continuum_line_selection_threshold,
    packed_continuum_wavelengths,
    hc_over_kt,
    selected_lines_output=None,
    predicted_atomic_lines_path=None,
    observed_atomic_lines_path=None,
    high_excitation_lines_path=None,
    diatomic_lines_path=None,
    titanium_oxide_lines_path=None,
    water_lines_path=None,
    h3plus_lines_path=None,
)

allocate_line_opacity_state(
    *,
    layer_count: int,
    wavelength_count: int,
) -> LineOpacityState

accumulate_selected_line_opacity(
    *,
    selected_lines,
    opacity_wavelength_grid_nm,
    wavelength_bin_edges,
    continuum_line_selection_threshold,
    temperature,
    hc_over_kt,
    electron_density,
    ion_stage_populations_by_packed_slot,
    partition_normalized_population_over_mass_density_and_fractional_doppler_width,
    fractional_doppler_widths,
    wavelength_start_index=1,
    wavelength_stop_index=None,
) -> LineOpacityState

read_line_transition_catalog(path: Path | str) -> LineTransitionCatalog

accumulate_transition_line_opacity(
    *,
    transition_lines,
    opacity_wavelength_grid_nm,
    wavelength_bin_edges,
    continuum_line_selection_threshold,
    temperature,
    hc_over_kt,
    electron_density,
    ion_stage_populations_by_packed_slot,
    partition_normalized_population_over_mass_density_and_fractional_doppler_width,
    fractional_doppler_widths,
    partition_normalized_populations_by_packed_slot=None,
    mass_density=None,
    base_line_mass_absorption_coefficient=None,
    wavelength_start_index=1,
    wavelength_stop_index=None,
) -> LineOpacityState
```

### 5.5 Complete pre-transfer state

`OpacityState` fields, in order:

```text
population_state
continuum_atmosphere
opacity_wavelength_grid_nm
opacity_frequency_hz
frequency_weights
active_continuum_indices
active_continuum_frequency_hz
continuum_absorption
continuum_scattering
continuum_source
continuum_line_selection_threshold
continuum_reference_wavelength_nm
wavelength_bin_edges
line_opacity
rosseland_table
selected_line_catalog
transition_line_catalog
```

Exact entry point:

```python
prepare_opacity_state(
    config: AtmosphereConfig,
    *,
    population_state: AtmospherePopulationState | None = None,
    temperature_iteration_index: int = 1,
    rosseland_table: RosselandOpacityTable | None = None,
    selected_line_catalog: SelectedLineCatalog | None = None,
    transition_line_catalog: LineTransitionCatalog | None = None,
) -> OpacityState
```

No wrapper named `ValidatedIterationSeed`, `BlanketedIterationInput`, or
`OpacitySamplingState` is introduced.

## 6. Exact state ordering

### 6.1 Supplied-seed boundary

For the canonical supplied fixed-column seed:

1. verify fixture path, role, bytes, and SHA-256 before import;
2. parse the deck into `ModelAtmosphere`;
3. verify explicit effective temperature, gravity, pressure flag, opacity
   flags, and complete 99-element abundance block as chapter fixture
   requirements;
4. call `validate_atmosphere_seed`;
5. require 80 layers for the canonical line-enabled route;
6. construct `AtmosphereConfig` with all output paths `None`;
7. call `resolve_run_setup`;
8. verify the exact supported branch settings;
9. if this supplied seed is declared to cross the seed quantization boundary,
   format/parse once and resolve again from the quantized model;
10. retain both pre- and post-quantization states for the numerical-delta
    ledger.

The chapter does not apply quantization unconditionally to every arbitrary
input. It states whether the supplied fixture is pre-quantized and whether the
demonstration intentionally crosses \(Q\).

### 6.2 Next-pass hydrostatic demonstration

Using declared previous-support integration-fixture columns:

1. require support arrays to match `(80,)` and be finite in chapter code;
2. set `turbulent_pressure = np.zeros(80, dtype=np.float64)`;
3. evaluate the zero-radiation analytic limit \(P_{\rm gas}=gm\);
4. evaluate the supplied integrated-radiation-pressure case;
5. calculate the hydrostatic residual from the exact returned column;
6. deliberately trigger the warning/floor branch in a separate controlled
   case;
7. state that the exact runner first calls this helper on pass 2.

The support fixture is not called a Chapter 11 radiation calculation.

### 6.3 Population state

1. begin from the resolved `RunSetup`;
2. call `build_runtime_state`;
3. update charge-square density;
4. when molecules are enabled, resolve/load the atmosphere molecular
   equilibrium catalog and initialize molecular state;
5. when pressure iteration is enabled, run molecular seed closure where
   required and `populate_all_species`;
6. compute fractional Doppler widths and the exact
   partition-normalized-population-over-density-and-width field;
7. return `AtmospherePopulationState`;
8. verify actual and partition-normalized packed states remain distinct;
9. stop short of claiming a completed convection internal-energy state.

The canonical path requires `pressure_iteration_enabled=True`. The exact
source's pressure-disabled behavior is characterized in tests, but is not used
as the primary blanketing state.

### 6.4 Continuum and selection reference

1. build the 18-field `ContinuumAtmosphereState`;
2. create the empty first-iteration `RosselandOpacityTable`;
3. build the cached 30,000-point wavelength and weight arrays;
4. derive decreasing `opacity_frequency_hz`;
5. compute full continuum absorption, scattering, and source;
6. determine the active subset of the first 343 continuum references;
7. compute continuum absorption/scattering on those active frequencies;
8. assemble the complete 344-column `float32` selection threshold;
9. verify the duplicate/sentinel final column and packed-index semantics;
10. allocate a fresh current-pass line-opacity state.

### 6.5 Selected and detailed line opacity

When zero-based flag 14 is enabled:

1. if a `SelectedLineCatalog` object was supplied, reuse it;
2. otherwise, if `selected_line_catalog_path` is supplied, read it;
3. otherwise generate one catalog from the existing raw source paths;
4. concatenate admitted family groups in exact source order:
   predicted atomic, observed atomic, high excitation, diatomic, TiO, water,
   then explicit H3+;
5. accumulate selected-line opacity into a new slab.

When zero-based flag 16 is enabled:

1. if a `LineTransitionCatalog` object was supplied, reuse it;
2. otherwise require and read `detailed_line_catalog_path`;
3. accumulate detailed opacity on top of the current selected-line slab.

The normal executable configuration sets flag 16 to zero. It reads the
compact detailed fixture only to validate `LineTransitionCatalog` field
order, dtype, count, and provenance, so the returned
`OpacityState.transition_line_catalog` is `None`. The exact detailed source
and runner branch remain staged for a later opt-in compile/evidence pass.

No selected catalog is regenerated merely because the atmosphere changed.
No previous pass's line-opacity slab is retained.

### 6.6 Reuse demonstration

The chapter's second state:

1. copies the first `ModelAtmosphere`;
2. applies a small, declared, physically valid temperature perturbation while
   retaining the same coordinate and mixture;
3. resolves a fresh setup and population state;
4. calls `prepare_opacity_state` with the first state's selected catalog;
5. asserts Python object identity for that selected catalog;
6. asserts no reselection occurred and the detailed branch remained inactive;
7. demonstrates a nonzero change in line opacity;
8. reports same membership/order/count and changed continuous values.

This is a lifecycle test, not an atmosphere iteration or convergence claim.

## 7. Real Numba boundaries

The chapter must not decorate high-level orchestration merely to make it look
compiled. The actual boundaries are:

| work | exact execution boundary | independence/order |
| --- | --- | --- |
| deck parse/format, validation, config resolution, gravity, hydrostatic update | Python/NumPy | ordered metadata and depth-column work; no `njit` |
| standard microturbulence table interpolation | Python/NumPy | ordered piecewise remap; no `njit` |
| runtime-state allocation | Python/NumPy | array construction |
| atomic EOS | reused exact Chapter 3 Numba kernels | depth-parallel where the exact source uses `prange` |
| molecular equilibrium | reused Chapter 4 compiled subkernels plus Python/LAPACK orchestration | depth continuation remains ordered |
| opacity grid and reference-table assembly | Python/NumPy plus `lru_cache` | exact array order |
| continuum component kernels | exact serial/parallel Numba kernels inside Chapter 5 functions | independent frequency columns use `prange`; top-level component summation remains ordered Python/NumPy |
| standard line keep test | `_selection_mask_compiled(..., parallel=True)` | independent raw line records; `fastmath` off |
| selected-row materialization | `_gather_selected_rows(..., parallel=True)` | independent output rows; source order retained |
| water keep test | `_water_selection_mask_compiled(..., parallel=True)` | independent water records; source order retained |
| selected line opacity | serial compiled kernel or `_accumulate_selected_line_opacity_parallel` | private line-chunk `float32` slabs; fixed ascending chunk reduction |
| detailed transition opacity | compiled normal/special kernels and `_accumulate_transition_line_opacity_parallel` where active | private line-chunk `float32` slabs; fixed ascending chunk reduction |

The continuum module's production compiled kernels include:

- `_planck_frequency_exact_kernel`;
- `_coulomb_freefree_gaunt_kernel`;
- `_linear_interpolate_kernel_serial`;
- `_linear_interpolate_kernel_parallel`;
- `_iron_neutral_branch_kernel`;
- `_helium_low_level_grid_kernel`;
- `_karzas_latter_cross_section_grid_kernel`;
- `_lukewarm_metal_absorption_kernel`.

The reader does not rederive those Chapter 5 kernels. The chapter names them
to explain why the 30,000-frequency state is multicore CPU work.

Required numerical statements:

- frequency columns are independent inside continuum component kernels;
- line-selection keep decisions are independent but must preserve input order;
- line-opacity chunks use private buffers to avoid races;
- private buffers are reduced in fixed chunk order;
- changing thread/chunk grouping may change a few last bits of line opacity;
- discrete selected membership must not change across supported thread counts;
- molecule depth continuation is not parallelized;
- no transfer depth recurrence occurs in Chapter 11.

Strict first-pass parity uses a fixed declared Numba thread count. A separate
subprocess check at another count records the measured tolerance and identical
catalog membership; it does not promise bit identity.

## 8. Two reader movements

### Movement 11A — Which seed is safe to start from?

Question:

> Which properties belong to the physical seed, which changes are exact
> numerical boundaries, and which support columns are not yet available on
> pass 1?

Required arc:

1. load the valid and one-point-bad seed views;
2. map all nine fields to units and constraints;
3. fail the bad seed before compilation;
4. materialize the exact 80-layer standard coordinate;
5. resolve explicit metadata, controls, gravity, flags, and microturbulence;
6. show the all-zero versus partially populated microturbulence behavior;
7. derive and check the next-pass hydrostatic helper without pretending it is
   called on pass 1;
8. cross and measure the fixed-column quantization boundary;
9. prove quantization idempotence;
10. demonstrate constant and monotone scalar remap behavior;
11. state the all-fields-shared-coordinate contract;
12. distinguish the seed/internal state from schema v4.

Movement 11A ends with a validated, explicitly supported `RunSetup`. It does
not yet contain EOS populations or opacity.

### Movement 11B — How does the mixture become blanketing opacity?

Question:

> How do one current atmosphere state and a stable catalog membership become
> continuum-plus-line opacity at every sampled frequency?

Required arc:

1. prepare molecule-enabled current populations and Doppler/strength state;
2. build and inspect the exact 30,000-point grid and threshold branches;
3. build the exact 18-field continuum adapter;
4. allocate/compute the three depth-major continuum slabs;
5. build the active reference subset and complete 344-column threshold;
6. trace standard family source order and provenance;
7. generate/load selected membership once;
8. load and validate the separate detailed catalog interface without
   activating IFOP(17);
9. accumulate selected opacity on the full grid;
10. inspect the complete `OpacityState`;
11. rebuild the continuous state under a small temperature perturbation while
    reusing the same catalog objects;
12. plot continuum-only versus blanketed extinction;
13. stop at the pre-transfer Chapter 12 boundary.

## 9. Visible code-cell ledger

Target: exactly 18 substantial visible cells. Setup/import cells do not count.
Most cells remain 10–30 lines; 60 lines is a soft ceiling and 80 a hard
ceiling. Long exact functions live in the progressive package.

| cell | causal purpose | required visible result |
| ---: | --- | --- |
| 1 | parse the checksum-bound seed and declare all nine columns | `(80,)`/unit/dtype table, explicit metadata, complete abundance count |
| 2 | construct the one-point bad-seed gallery and call exact validation | named failures; valid seed accepted before compiled physics |
| 3 | build the standard Rosseland coordinate | exact endpoints, 0.125 log spacing, inward order |
| 4 | resolve `AtmosphereConfig` and `RunSetup` | exact flags, gravity, controls, default/explicit metadata ledger |
| 5 | compare all-zero and partly positive microturbulence seeds | exact in-place fill rule and no partial fill |
| 6 | evaluate next-pass hydrostatic limits and warning/floor case | residual, support decomposition, pass-2 ownership |
| 7 | apply literal fixed-column format/parse once and twice | field deltas and exact idempotence |
| 8 | call Chapter 9 `remap_to_grid` for constant and monotone columns | remap values, returned final source-interval index, no complete-state remap |
| 9 | prepare molecule-enabled `AtmospherePopulationState` | exact packed shapes, actual/normalized distinction, finite density state |
| 10 | build 30,000-point grids at the seed and exact threshold sentinels | start index, wavelength/frequency order, weights, strict `<` table |
| 11 | build `ContinuumAtmosphereState` | exact 18 fields and CH/OH slot aliases |
| 12 | compute full continuum absorption/scattering/source | `(80,30000)`, units, finite/nonnegative opacity, memory ledger |
| 13 | assemble the active reference and 344-column threshold | active count, `float32`, duplicate 342→343, final packed sentinel |
| 14 | generate the combined selected catalog from compact raw sources | family counts/order/hash; no H3+ in the standard set |
| 15 | validate detailed-transition records and accumulate selected opacity | inactive detailed flag, branch dtype, counts, nonnegative finite full-grid slab |
| 16 | call `prepare_opacity_state` for the complete compact state | exact `OpacityState` fields, component shapes, empty first-pass Rosseland table |
| 17 | perturb temperature and reuse the selected catalog object | `is` identity, zero reselection, inactive detailed branch, changed line opacity |
| 18 | compare continuum-only and blanketed extinction and print handoff | one-panel blanketing plot, compact parity ledger, explicit Chapter 12 boundary |

A chapter runtime helper may:

- configure local data/catalog/cache roots before import;
- verify hashes;
- construct the two supplied seed views;
- count selection/load calls;
- collect typed checkpoints;
- format tables and plotting arrays.

It may not:

- implement a second EOS, continuum, selector, line-opacity builder, or
  atmosphere runner;
- hide future transfer/correction work;
- invent a public seed/opacity wrapper;
- read the external Payne Zero checkout.

## 10. Original schematics and one-panel plot plan

### 10.1 Original conceptual schematics

Create three original textbook assets through
`scripts/textbook_schematic_specs.py`. Use the shared white/slate/navy/beige
scientific-notebook language without copying an official-site composition.

1. **`ch11-seed-gates-to-pass-state-v1`**  
   A nine-track outer-to-inner seed column enters validation. One branch
   crosses an explicitly labelled fixed-column quantization gate. Pass 1
   copies the seed; a separate grey return arrow shows that hydrostatic
   pressure first consumes previous radiation support on pass 2. The diagram
   ends at `RunSetup`, not at a converged atmosphere.

2. **`ch11-two-atmosphere-grids-v1`**  
   Left: the fixed 80-layer \(\tau_{\rm R,std}\) coordinate. Right: the direct
   30,000-sample wavelength grid with the five strict effective-temperature
   start branches and a reversed frequency arrow. A small crossed-out
   synthesis edge-triplet makes the lane distinction clear. It is conceptual
   and contains no invented numerical opacity.

3. **`ch11-select-once-recompute-opacity-v1`**  
   Population/continuum state and raw atomic/diatomic/TiO/water sources enter
   first-pass selection and separate detailed-catalog loading. Two atmosphere
   cards reuse the same grey catalog-object handles but produce different
   amber line-opacity slabs. The lower edge stops at `OpacityState`; a blue
   arrow labelled “Chapter 12 transfer reductions” exits the frame.

Each asset requires:

- an owned prompt/specification;
- generator/source provenance;
- exact alt text and caption;
- native-size and notebook-width review;
- scientific label review;
- SHA-256 registration.

Each caption says “conceptual.” No schematic serves as a parity result.

### 10.2 Required professional one-panel plots

Use `book.plot_style`, white background, inward ticks, exact units, and one
interpreted claim per panel.

1. **Fixed-column quantization is small but not zero.**  
   A one-panel horizontal bar chart shows the maximum normalized change for
   each of the nine columns after the first \(Q\). A second marker at zero
   shows the exact \(Q(Q(x))-Q(x)\) result. The caption identifies the deck
   format and does not call the first delta an error correction.

2. **Effective temperature changes wavelength coverage, not sample count.**  
   A one-panel log-wavelength plot shows the exact coverage intervals for the
   five start-index branches, all with 30,000 samples. Exact threshold
   temperatures are annotated on the warmer side of each strict `<`
   boundary. This is generated from `build_opacity_sampling_grid`, not drawn
   schematically.

3. **Lines convert smooth continuum extinction into a blanketing field.**  
   A one-panel depth-versus-wavelength heat map shows
   \[
   \log_{10}
   \frac{\kappa_{\nu,c}+\sigma_{\nu,c}+\kappa_{\nu,\ell}}
        {\kappa_{\nu,c}+\sigma_{\nu,c}}
   \]
   for the compact canonical state, with a declared positive denominator
   floor used only for plotting. The depth axis is
   \(\log_{10}\tau_{\rm R,std}\); wavelength is in nm. The caption says this
   shows blocking/blanketing opacity, not a computed temperature response.

Catalog identity, branch status, family counts, source hashes, and Numba
thread comparisons are tables, not ornamental plots.

## 11. Pinned source requirements

### 11.1 Primary exact source identities

| pinned file | SHA-256 | Chapter 11 responsibility |
| --- | --- | --- |
| `payne_zero_atmosphere/atmosphere_io.py` | `95c4d2cab230f6925e9404639ecb05b25af8c0c85755ac1ca70d760156a8683e` | `ModelAtmosphere`, parse/format quantization |
| `payne_zero_atmosphere/config.py` | `51e19846fb81c832ae57334faf3da2c1e4fc2ef9edf6e08467ef7296e4640b45` | input/output/config fields and defaults |
| `payne_zero_atmosphere/run_setup.py` | `de7cf08b936585dbcfa2e572c026fafa3f10282a99c27b834b62db0f3f2888c9` | validation, exact standard grid, controls, microturbulence |
| `payne_zero_atmosphere/hydrostatic.py` | `f59f7b807152b74f1cf85ed208c612454aa82f62369f5d7baebe3d1a46740fef` | next-pass gas-pressure helper |
| `payne_zero_atmosphere/microturbulence.py` | `3692062f1d6877e745ed84bba4fc2fdf04c60a7c52bc27856fc696416a0283cb` | standard profile and exact table/remap policy |
| `payne_zero_atmosphere/runtime_state.py` | `fae240ec00f6f89d7c2a7ef721ce6e6539be234e523291fd6e8a096d731430e8` | initial density/mixture/packed state |
| `payne_zero_atmosphere/runner.py` | `05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7` | exact population and opacity state definitions/composition only |
| `payne_zero_atmosphere/continuum_opacity.py` | `1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81` | 18-field adapter, 30,000 grid, continuum/reference state |
| `payne_zero_atmosphere/line_catalog.py` | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` | selected/detailed exact records and readers |
| `payne_zero_atmosphere/line_selection.py` | `b2c62fdf5e1fe43f33022184bfeff88985b13331354e3c745c7dab3a6b634fef` | exact family selection and source order |
| `payne_zero_atmosphere/line_opacity.py` | `d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6` | selected/detailed current-pass line slab |
| `payne_zero_atmosphere/source_catalogs.py` | `a9ea21735c9d4964b785d76c89c9fc976a30ed75f8b6f9d4f7c6aaa4e77dae36` | standard full-source paths/checksum tooling |
| `payne_zero_atmosphere/radiative_transfer.py` | `df8970ca629487537a7c4849278eab5d755b527002d8fc58360c9264a3aa45db` | Chapter 9 scalar remap reused here |
| `payne_zero_atmosphere/synthesis_bridge.py` | `142a960b5e710823754b02766803b3c1dd8c48c9945fdfabe560b4ee7e1acb50` | downstream-only schema-v4 boundary audit |

Reused exact dependency identities:

| pinned file | SHA-256 |
| --- | --- |
| `payne_zero_atmosphere/equation_of_state.py` | `719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2` |
| `payne_zero_atmosphere/molecular_equilibrium.py` | `4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86` |
| `payne_zero_atmosphere/doppler.py` | `e118a78bf5250ef5e1f77d652c9e78fbb7b92acf5c069f717faed7a3b3ea98f0` |
| `payne_zero_atmosphere/specific_internal_energy.py` | `de06ba732ce1333d111a52223e39f5b4f80eece8cfc4ff2f30de9739e16d7ec5` |

### 11.2 Progressive source staging boundary

The local progressive package already contains byte-identical copies of most
Chapter 11 dependencies. Chapter 11 implementation must:

1. add byte-identical `src/payne_zero_atmosphere/hydrostatic.py`;
2. extend progressive `src/payne_zero_atmosphere/runner.py` with AST-exact
   definitions:
   - `OpacityState`;
   - `_empty_first_iteration_rosseland_table`;
   - `_existing_optional_path`;
   - `_generate_standard_selected_lines`;
   - `prepare_opacity_state`;
3. preserve the already exact `AtmospherePopulationState` and
   `prepare_population_state`;
4. add only the exact imports required by those definitions;
5. not stage `TransferAccumulation`, `IterationFinalization`,
   `IterationRemap`, `accumulate_transfer_state`, `finalize_transfer_state`,
   remap-finalization, convergence, or the full runner loop until their owning
   chapters;
6. update `src/PAYNE_ZERO_SOURCE_MANIFEST.json` and
   `scripts/verify_pinned_source_fragments.py`;
7. verify the new full-file and AST-exact source surfaces against the pinned
   commit.

This progressive boundary keeps the Chapter 11 module importable without
pulling in incomplete Chapter 12/13 modules.

## 12. Static tables and chapter data

### 12.1 Atmosphere tables

The local repository already contains these exact static inputs:

| local path | SHA-256 |
| --- | --- |
| `data/static/atmosphere_tables/continuum_opacity_tables.npz` | `6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3` |
| `data/static/atmosphere_tables/hydrogen_line_profile_tables.npz` | `607b686c4e1ca41bd1eba759b05cb44db2a192ec2a008554e92f4b13d8e10fbf` |
| `data/static/atmosphere_tables/ionization_potential_tables.npz` | `82a2e82f2015da02c3d2bce77ca5337aa2b9c4e23d8d6219da07895896ca8a50` |
| `data/static/atmosphere_tables/iron_group_partition_tables.npz` | `137629dea64eca46f77ea3656c18305ade912a468d7eb27029544c0106cc3296` |
| `data/static/atmosphere_tables/isotope_tables.npz` | `53c8d315fb53f1e051dc2752b028fc270d7c17a2c1042279c04ffcb750aef5c6` |
| `data/static/atmosphere_tables/karzas_latter_tables.npz` | `23805dc17c47af45b8ae63b2e278e1fb6c584a01c87d1eb3c31306e4555e6d15` |
| `data/static/atmosphere_tables/line_opacity_tables.npz` | `89f486122cb8939b23dc5423145a46d88a77df8daf57a1def35055b7b8205f16` |
| `data/static/atmosphere_tables/molecular_equilibrium_tables.npz` | `1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe` |
| `data/static/atmosphere_tables/packed_level_metadata.npz` | `de5f17b6a9eaec1d1b07e96fd02ff014279cd8eaa9f976fefde0e2a153961bc3` |
| `data/static/atmosphere_tables/special_partition_tables.npz` | `7d737524aacda1cc2281e5b18ff49f240ca34665dbe6c96d4dd0f39db4aedd22` |
| `data/static/source_catalogs/lines/molecular_equilibrium_atmosphere.npz` | `971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644` |

One exact runtime dependency is not yet staged and is required before
Chapter 11 implementation:

| required local path | pinned source path | SHA-256 |
| --- | --- | --- |
| `data/static/atmosphere_tables/continuum_level_tables.npz` | `/Users/ysting/payne-zero/source_data_files/atmosphere_tables/continuum_level_tables.npz` | `35a6839be4ff3dd824206c7a6b851b987132313374ede7ea5441f9d0bd69888f` |

It is a 26-member, approximately 9.8 KiB static physical input containing the
H/He/C/Mg/Al/Si/K/Ca level arrays, `element_block_offsets`, and
`partition_interpolation_scale`. It must be copied with exact member
shape/dtype/unit metadata and registered in `data/MANIFEST.json`.

`radiative_transfer_tables.npz` is already staged for Chapter 9 but is not a
Chapter 11 scientific input. Importing the progressive runner must not be
confused with loading or executing transfer tables.

### 12.2 Seed and support fixtures

Required new integration fixtures:

1. **Canonical fixed-column seed deck**  
   One 80-layer supplied solar-like seed with:
   - all nine columns;
   - explicit effective temperature and gravity;
   - explicit pressure and opacity controls;
   - complete 99-element fixed-column abundances;
   - source/generator identity;
   - declaration of whether it has already crossed fixed-column
     quantization.

2. **Previous-support columns for the hydrostatic lesson**  
   `integrated_radiation_pressure` and zero
   `turbulent_pressure`, each `(80,)`, CPU `float64`, explicitly labelled as
   an integration fixture generated upstream of Chapter 11. The fixture
   isolates the pass-2 pressure boundary; it is not a reader-built Chapter 12
   result.

Invalid seeds are derived in memory from the canonical seed. They are not
stored as separate opaque files.

An optional learned-origin seed may be supplied only as a second
`ModelAtmosphere` integration fixture. Its generation, checkpoint, support,
and role are not explained until Chapter 14.

### 12.3 Compact atmosphere line-source view

The normal notebook requires compact source-compatible atmosphere records,
not the synthesis compiled catalog:

- a small predicted/ordinary atomic raw-source subset in the exact atmosphere
  selector format;
- the existing atmosphere diatomic subset;
- the existing TiO raw-source subset;
- the existing water raw-source subset;
- a compact detailed-transition NPZ with all exact
  `LineTransitionCatalog` fields;
- the molecular-equilibrium atmosphere catalog.

The current reusable molecular subset hashes are:

| local input | SHA-256 |
| --- | --- |
| `data/subsets/chapter08_atmosphere_diatomic_subset.npy` | `ebc43f107b8e046cf6494eb587a4234fa791e2d85b4a05817cd1935f969e45db` |
| `data/subsets/chapter08_tio_subset.npy` | `204c2aa286b173c7a8125e7aa67139155522f7594acb44eabc1adac11bb6ab13` |
| `data/subsets/chapter08_h2o_subset.npy` | `26bf33be3859dcd3ed601f820a88f7bf66fe7fc132a69ada0ad2aefba2b43409` |

The Chapter 7 synthesis atomic subset is not automatically the exact raw
atmosphere selector format. Chapter 11 must prepare and manifest a dedicated
atmosphere raw-source subset or use a separately provenance-bound exact
conversion. It must not rename the synthesis subset and assume compatibility.

The Chapter 8 H3+ probe at
`data/fixtures/chapter08_h3plus_path_probe.npy` is used only for the explicit
opt-in boundary check. It is not included in the canonical standard family
set.

Every compact source records:

- exact upstream path/record indices or generating command;
- pinned commit;
- source archive SHA-256;
- local bytes/SHA-256;
- array/record schema;
- family;
- units and packed-field convention;
- order;
- role as a teaching subset.

### 12.4 Full optional source data

The exact full source tree is approximately 6.8 GB. Its checksum manifest is:

```text
/Users/ysting/payne-zero/source_data_files/source_catalogs/CHECKSUMS.sha256
SHA-256 61d3233b528a2b98e2f360b6ac795a8a948cf004c4588cbbe530ed3475ca029d
```

The normal notebook never reads that path. A later optional full-data gate
uses the repository's checksum-verified installation path and records:

- all predicted atomic shards;
- observed and high-excitation atomic sources;
- diatomic, TiO, and water sources;
- detailed transitions;
- molecular equilibrium;
- exact checksum manifest and total bytes.

Subset results and full-catalog parity are always labelled separately.

### 12.5 Goldens and comparison roles

Comparison-only Chapter 11 goldens live under
`data/golden/payne_zero/` and are opened only after the reader computation.

The complete-spine-first golden set contains:

- parsed and once-quantized seed columns/metadata;
- `RunSetup` scalar/control fields;
- hydrostatic helper output for the supplied support fixture;
- standard depth grid;
- 30,000 wavelengths, frequencies, and weights or exact independently
  reconstructable hashes plus sentinel values;
- population-state probes and full-array hashes;
- continuum slab probes/full-array hashes;
- active reference indices and complete 344-column threshold;
- selected/detailed catalog field arrays and order hashes;
- selected, detailed, and combined line-opacity probes/full-array hashes;
- complete `OpacityState` member shape/dtype/hash ledger;
- second-state catalog identity/reuse counters and line-opacity delta probes.

At least one fixed-thread compact oracle capture must retain complete arrays
or a lossless separately reconstructable representation sufficient for exact
component comparison. Scalar summaries alone cannot establish slab parity.

Every golden records:

- pinned commit;
- all table/source/fixture hashes;
- Python, NumPy, Numba, operating system, and architecture;
- `NUMBA_NUM_THREADS`;
- temperature iteration index;
- exact flags and paths;
- fixed-column quantization status;
- exact/tolerance comparison policy.

## 13. Runtime configuration boundary

The chapter runtime configures paths before any
`payne_zero_atmosphere` import:

```text
PAYNE_ZERO_DATA_ROOT
PAYNE_ZERO_SOURCE_CATALOG_ROOT
PAYNE_ZERO_NUMBA_CACHE_DIR
NUMBA_CACHE_DIR
NUMBA_NUM_THREADS
```

The normal chapter uses:

- a repository-owned data-root view containing `atmosphere_tables/`;
- a disposable or manifest-owned compact source-catalog view;
- a disposable/persistent chapter cache outside static data;
- a fixed declared thread count.

`book.chapter11_runtime.configure_chapter11_runtime()` must:

1. reject any already imported `payne_zero_*` module that resolved outside
   the repository `src/` tree;
2. verify every input hash before scientific import;
3. set all data/source/cache/thread controls;
4. put the repository `src/` directory first on `sys.path`;
5. remain safe when called before every public checkpoint;
6. support a fresh-process regression in which Chapter 10 synthesis is
   imported/called before Chapter 11 atmosphere preparation;
7. never mutate repository static inputs or the external source tree.

Because atmosphere table paths and some source-root constants are resolved at
module import, tests that change roots or thread counts use fresh subprocesses.

## 14. Exact tests and parity gates

### 14.1 Source, import, and API gates

- pinned commit and every source hash in Section 11 match;
- `hydrostatic.py` is byte-identical;
- every new progressive runner definition is AST-identical;
- dataclass field order matches Section 5;
- exact signatures/defaults match Section 5;
- source verifier passes without the external source becoming a runtime
  dependency;
- no Chapter 12/13 definition is staged merely to satisfy imports;
- fresh process resolves every `payne_zero_*` module under repository `src/`;
- Chapter 10 then Chapter 11 and Chapter 11 then Chapter 10 import/call orders
  both use the intended local data roots;
- chapter code never reads `/Users/ysting/payne-zero`.

### 14.2 Seed and setup gates

- all nine canonical columns have shape `(80,)`;
- parsed columns are NumPy `float64`;
- required positive/finite/monotone rules pass;
- isolated NaN, nonpositive required field, nonmonotone column mass, negative
  microturbulence, and mismatched column length each fail with the exact
  source exception family/message anchor;
- finite negative radiative acceleration is source-characterized as allowed;
- smaller common layer counts pass the seed validator but fail the separate
  canonical 80-layer teaching gate;
- canonical metadata is explicit and does not use silent Teff/logg/flag
  defaults;
- canonical abundance block contains atomic numbers 1–99;
- source behavior with a missing metal entry is characterized as a
  `1.0e-30` runtime fill, not used in the canonical state;
- exact config count clamping and limit rejection pass;
- all-zero microturbulence fills; partly positive microturbulence remains
  unchanged;
- the in-place mutation is tested explicitly;
- exact default opacity flags and flag indices pass.

### 14.3 Depth, quantization, hydrostatic, and remap gates

- standard 80-point grid is exact in `float64`;
- endpoint and log-spacing checks match Section 3.6;
- fixed-column row formatting matches the exact source;
- \(Q(Q(x))\) equals \(Q(x)\) at every parsed field and relevant metadata
  boundary;
- first-round field deltas match pinned seed checkpoints;
- zero-support hydrostatic output equals \(gm\);
- supplied radiation-support output matches the exact helper;
- warning/floor branch emits `RuntimeWarning` and positive floors;
- scalar support arrays are preflight shape-checked by chapter code and that
  stronger check is not attributed to the source helper;
- constant remap remains constant on the in-range target;
- monotone remap and returned final source-interval index match Chapter 9;
- no complete corrected-state remap or quantizer-between-passes code exists.

### 14.4 Atmosphere sampling-grid gates

For threshold sentinels `30000`, just below `30000`, `13000`, just below
`13000`, `7250`, just below `7250`, `4500`, and just below `4500`:

- exact `start_index` branch is inferred from exact endpoint values;
- count is always 30,000;
- wavelength formula is exact;
- wavelength is strictly increasing;
- frequency is strictly decreasing;
- frequency conversion uses `2.99792458e17`;
- first/interior/last weight formulas match independently;
- weights are finite and positive;
- exact threshold values use the warmer branch;
- repeated equal-temperature calls source-characterize cached object identity;
- chapter experiments copy before mutation;
- no `resolution`, synthesis context, or edge-triplet route enters.

### 14.5 Population-state gates

- `AtmospherePopulationState` exact fields/shapes/dtypes pass;
- runtime state uses `(80,99)` abundances and `(80,1006)` packed arrays;
- actual and partition-normalized arrays are separately nonnegative and not
  aliased;
- supplied/mutated electron state follows the exact pressure-iteration route;
- molecule-enabled and atom-only branches are both covered;
- canonical molecule-enabled state loads the exact atmosphere molecular
  catalog;
- ordered molecular continuation matches Chapter 4 regression checkpoints;
- fractional Doppler widths and normalized strength state match Chapter 6
  checkpoints;
- `ch_population` and `oh_population` later map to normalized slots 845/847;
- full convection-ready specific internal energy is not asserted here.

### 14.6 Continuum and reference gates

- `ContinuumAtmosphereState` has the exact 18 fields;
- every field has the exact shape/meaning inherited from Chapter 5;
- full continuum arrays are `(80,30000)` `float64`;
- absorption and scattering are finite and nonnegative for valid canonical
  input;
- continuum source is finite and retains the `B_nu` source-function unit;
- component order and opacity flags match Chapter 5;
- active reference indices/frequencies match exact grid cutoff;
- threshold is `(80,344)` `float32`;
- active threshold formula independently matches;
- inactive threshold formula independently matches;
- reference column 343 duplicates 342;
- packed index 343 equals `2**30`;
- exact `wavelength_bin_edges` name is preserved;
- empty first-pass Rosseland table has capacity `80*60` and `entry_count=0`.

### 14.7 Catalog and line-opacity gates

- compact raw source formats are exact and hash-bound;
- selected family concatenation order is exact;
- standard combined set includes atomic, diatomic, TiO, and water;
- H3+ is absent unless its explicit path is supplied;
- explicit H3+ probe changes only the opt-in branch;
- selected catalog field dtypes/order/count/hash match the pinned oracle;
- detailed catalog field dtypes/order/count/hash match the pinned oracle;
- flag 14 off skips selected selection/deposition;
- flag 16 off skips detailed load/deposition;
- selected flag with neither preselected nor any existing raw source raises;
- detailed flag without a detailed catalog path/object raises;
- low-level omission of one raw family while another exists is
  source-characterized as a skip;
- high-level standard source resolution requires its full family set;
- selected accumulation requires exactly 80 layers;
- initial no-line slab is `float64`;
- selected slabs are `float32`, finite, and nonnegative;
- standard line opacity matches the relevant Chapter 6–8 component
  checkpoints;
- no dense `(D, lines, Nnu)` allocation occurs.

### 14.8 Complete-state and reuse gates

- `prepare_opacity_state` field order and member shapes match Section 5.5;
- complete canonical `OpacityState` matches the compact pinned oracle;
- all three continuum slabs, threshold, line slab, grids, and catalog handles
  are present;
- first call invokes selection exactly once and leaves the detailed handle
  `None`;
- second call supplied with the selected object invokes no reselection;
- the second-call selected catalog identity is an exact `is` match;
- selected membership arrays/order/count remain unchanged;
- temperature perturbation changes populations/widths and at least one
  line-opacity value;
- the old line slab is not reused or mutated;
- the first state's arrays remain unchanged after the second build;
- cached grid arrays are not mutated;
- no transfer, correction, remap-finalization, convergence, quantization, or
  product writer call occurs.

### 14.9 Numba, cache, thread, and memory gates

- fixed thread count is recorded before Numba import;
- exact compiled selectors and line-opacity kernels are present;
- `fastmath` remains off on discrete selection kernels;
- fixed-thread repeated results meet the measured exact/tolerance policy;
- alternate-thread selected membership is identical;
- alternate-thread line opacity uses a measured few-ulp envelope;
- no claim of cross-thread bit identity is made without evidence;
- cold and warm Numba cache runs are scientifically equal under one thread
  policy;
- cache root is disposable/chapter-owned;
- static input and external source snapshots remain unchanged;
- actual live memory is reported and stays within the declared compact
  envelope;
- no `(80, line_count, 30000)` allocation is made.

### 14.10 Schema-v4 and Chapter 10 boundary gates

- `ModelAtmosphere`, `AtmosphereRuntimeState`, `AtmospherePopulationState`,
  and `OpacityState` are explicitly rejected by the Chapter 10 schema
  validator when mispresented as schema-v4 mappings;
- no `save_product_structured_atmosphere` call occurs;
- no live runtime-state product is written;
- no `structured_atmosphere_path`, diagnostics file, or debug product is
  claimed;
- the chapter handoff table states that Chapter 13 later rebuilds schema v4
  from terminal fixed-column-quantized columns;
- Chapter 10's exact five-field `Spectrum` interface remains unchanged;
- no synthesis call is used as the Chapter 11 parity oracle.

### 14.11 First-pass versus later evidence

Mandatory now:

- one canonical 80-layer molecule-enabled compact state;
- exact 30,000-point grid and full continuum slabs;
- selected compact line opacity on the full grid;
- detailed-transition schema/provenance validation with IFOP(17) inactive;
- fixed-thread compact oracle parity;
- catalog object reuse;
- exact schematics/plots;
- source/data/governance tests.

Deferred to the later evidence pass:

- optional 6.8 GB full-catalog membership and slab parity;
- hot/solar/giant/cool full-data component matrix;
- broad alternate-thread tolerance authorization;
- opt-in detailed-transition kernel compilation and deposition parity;
- complete one-pass transfer/correction parity;
- multi-pass trajectory and convergence;
- terminal schema-v4 product;
- Chapter 10 spectrum from the physically converged atmosphere.

## 15. Unsupported branches, failures, and honest boundaries

The chapter must state:

- NaNs, nonpositive required seed fields, nonmonotone column mass, negative
  microturbulence, and mismatched seed-column lengths fail before compiled
  physics;
- the exact source does not generally repair those failures;
- source validation alone does not require 80 layers or complete abundances;
- canonical line blanketing does;
- partial-zero microturbulence is not automatically filled;
- the hydrostatic helper warns/floors nonpositive output rather than raising;
- pass 1 does not use that helper;
- turbulent-pressure evolution is unsupported;
- zero-based opacity flag 13 (`HLINOP`) is unsupported by the exact runner;
- NLTE/departure-coefficient solving remains outside the scientific scope;
- the standard converted atomic, diatomic, TiO, and water selector paths are
  active and must not be described as unsupported raw selectors;
- atmosphere H3+ is explicit-path opt-in and absent from
  `source_line_paths()`;
- the separate synthesis H2O compiler omission does not disable atmosphere
  water opacity;
- missing detailed data under enabled flag 16 is a hard error;
- the low-level raw selection adapter may skip an individually missing family
  if another source exists, so the chapter's own manifest preflight is
  required for a claimed family set;
- line selection is a discrete, nondifferentiable host/Numba decision;
- selected and detailed opacity require 80 layers;
- full source catalogs may exceed constrained memory/storage;
- compact subset pedagogy is not full-catalog parity;
- cached sampling-grid arrays are mutable and must be treated as read-only;
- fixed-thread grouping is part of strict parity;
- a few last bits may change across thread grouping while membership must not;
- an `OpacityState` is not a radiation solution;
- line blocking is not yet a calculated back-warming correction;
- a validated/blanketed state is not structurally converged;
- no physical schema-v4 product exists at the end of this chapter.

`AtmosphereOutput.diagnostics_path` is declared but unwritten in the pinned
source. Chapter 11 sets every output path to `None` and claims no file product.

## 16. Redundancy and deferral audit

Chapter 11 must not:

- rederive Saha, electron closure, or molecular Newton solves;
- rederive continuum processes;
- rederive ordinary/special/molecular line physics or selection inequalities;
- rederive transfer;
- call the full atmosphere runner;
- call a one-iteration runner as a shortcut around Chapters 12–13;
- introduce generalized seed or opacity wrapper names;
- call the standard Rosseland coordinate a computed current optical depth;
- call the atmosphere 30,000 grid a synthesis grid;
- add `resolution` or \(R_{\rm grid}\) to atmosphere sampling;
- rename packed population meanings;
- call CH/OH continuum aliases actual molecular densities;
- claim unconditional `float64` or `float32` line-slab dtype before transfer;
- claim all missing low-level source paths fail automatically;
- include H3+ in the default high-level source set;
- apply fixed-column quantization between physical passes;
- implement the complete remap;
- compute radiation, convection, correction, convergence, or learned starts;
- serialize a live runtime state as an accepted physical product;
- call a supplied or learned-origin seed converged;
- call Chapter 10 synthesis proof of atmosphere closure;
- require the external source checkout or full catalogs at reader runtime;
- add an exercise section.

## 17. Chapter summary and causal handoff

End with `## 11.N Chapter summary`. It introduces no new object or claim.

The summary must state:

1. a usable seed is nine aligned outer-to-inner columns plus explicit metadata
   and mixture, not merely \(T(\tau)\);
2. the exact validator rejects finite/shape/sign/monotonicity failures but is
   not a general sanitizer;
3. setup normalization is narrow, and all-zero microturbulence fill mutates
   the seed in-place;
4. pass 1 copies the seed; the hydrostatic gas-pressure update first consumes
   previous radiation support on pass 2;
5. fixed-column format/parse is a deliberate numerical boundary whose
   idempotence is tested;
6. the production coordinate has 80 standard Rosseland layers, while the
   direct atmosphere opacity grid has 30,000 effective-temperature-dependent
   samples;
7. the current state recomputes EOS populations, molecular closure, Doppler
   widths, continuum, and line opacity;
8. selected catalog membership is created once and reused while
   atmosphere-dependent opacity is recomputed; the detailed catalog interface
   is validated separately with its accumulation flag inactive;
9. the standard atmosphere family set contains atomic, diatomic, TiO, and
   water sources; H3+ is explicit-path opt-in;
10. the returned `OpacityState` is a depth-major CPU pre-transfer state, not
    schema v4, not a spectrum, and not a converged atmosphere.

State the exact output now available:

```text
validated supported RunSetup
    + AtmospherePopulationState
    + opacity_wavelength_grid_nm[30000]
    + opacity_frequency_hz[30000]
    + frequency_weights[30000]
    + continuum_absorption[80,30000]
    + continuum_scattering[80,30000]
    + continuum_source[80,30000]
    + continuum_line_selection_threshold[80,344]
    + line_opacity.line_mass_absorption_coefficient[80,30000]
    + reusable selected_line_catalog
    + transition_line_catalog = None on the normal IFOP(17)-off path
    -> OpacityState
```

The unresolved problem is reduction, not more opacity. The chapter now knows
how difficult every sampled frequency is to escape, but it has not integrated
the transfer field into Rosseland opacity, radiative flux, force, pressure,
heating, or material response.

Close with:

### Next: reduce 30,000 frequencies into physical forces and fluxes

> The atmosphere now carries a complete continuum-plus-line blanketing state,
> but opacity alone cannot say whether the seed transports the required flux
> or how much support radiation and convection provide. [Chapter
> 12](/reader.html?ch=12) sends each frequency through the transfer solver,
> reduces fixed-order private accumulators into Rosseland, flux, pressure,
> heating, and radiative-force columns, and computes the thermodynamic and
> convective response needed before Chapter 13 can correct the structure.
