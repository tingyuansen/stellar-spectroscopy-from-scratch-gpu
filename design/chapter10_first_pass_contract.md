# Chapter 10 first-pass contract — GPU Synthesis from a Structured Atmosphere

Status: bounded reader-facing design; no implementation or publication authority  
Pinned Payne Zero commit: `9c44001feae40b85146630499e6f8a5fed42e5af`  
Audience: final-year undergraduate / first-year graduate student  
Canonical title: **GPU Synthesis from a Structured Atmosphere**

## 0. Canonical placement and ownership

Chapter 10 completes Part IV by composing the physical kernels already earned
by the reader:

```text
Chapters 3–4: fixed-electron-density thermochemical state
Chapters 5–8: continuum and every standard synthesis line family
Chapter 9: total/continuum transfer and H_nu -> F_lambda semantics
                         |
                         v
Chapter 10: reusable window state + one-star orchestration
                         |
                         v
          exact native Spectrum from a supplied atmosphere
                         |
                         v
Chapter 11: begin constructing a physically closed atmosphere
```

This chapter owns:

- the exact CUDA/MPS/CPU runtime policy;
- `WindowInvariants`, its in-process cache, and the persistent cache products
  used to build it;
- the distinction between window-invariant and atmosphere-dependent state;
- the 16-sample context construction and exact device-side crop;
- the complete component call order;
- bounded metal and molecular work;
- the standard LTE source and line-scattering policy;
- the final spectral device-to-host construction stage;
- exact `SpectrumResult` and public `Spectrum` records;
- cold, persistent-warm, and process-warm timing;
- same-atmosphere synthesis parity for four stellar regimes.

It does **not** rederive:

- electron closure or molecular equilibrium from Chapters 3–4;
- any continuum process from Chapter 5;
- line strength, profile, routing, or deposition from Chapters 6–8;
- optical-depth integration, source iteration, saturated-core transfer, or
  the \(H_\nu\rightarrow F_\lambda\) Jacobian from Chapter 9.

The chapter invokes those canonical functions and inspects their interfaces.
It does not paste them into a new monolithic notebook function.

The supplied schema-v4 atmospheres are integration fixtures. Successful
synthesis proves the synthesis stage for those supplied states. It does not
prove that any fixture is structurally converged, in flux balance,
hydrostatically accepted, or a successful output of the later physical
atmosphere iteration.

There are no detached exercises. Chunk variations, cache failures, dtype
comparisons, and limiting cases belong in the causal text beside the claim
they test.

### 0.1 Complete-spine-first acceptance

The first executable pass is accepted at chapter level when the compact,
self-contained CPU-float64 path runs for all four supplied regimes, produces
the exact public outputs, passes same-backend cold/warm and pinned compact
parity, renders the required figures, and closes the Chapter 9 and Chapter 11
handoffs.

It does not wait for an optional full-catalog run or unavailable CUDA/MPS
hardware. Those cases remain explicit later evidence gates. A wrong interface,
wrong array meaning, early spectral host conversion, failed CPU result, or
false cache/backend claim still blocks the affected first-pass text.

## 1. The chapter's single question

Open with two synthesis calls over the same compact wavelength window. The
first call starts with an empty process cache and must decode or load catalogs,
construct the context grid, upload tables, and build invariant tensors. The
second call changes the supplied atmosphere but preserves the window, source
files, device, dtype, molecular flag, and chunk policy.

Ask:

> Which work belongs to the wavelength window, which work belongs to this
> star, and how can the calculation preserve its exact numerical policy until
> it returns one native spectrum?

Make three predictions before timing or code:

1. the two stars may reuse the same `WindowInvariants` object;
2. the hydrogen series-merging state must still change because it depends on
   the new star's `electron_density`;
3. warm and cold-cache runs may take different amounts of time, but under one
   fixed backend and grouping policy they must return the same four public
   spectral arrays.

The causal construction is:

```text
requested window + source/table identity + runtime policy
    -> exact requested grid
    -> 16 blue + 16 red native context samples
    -> reusable WindowInvariants

one validated structured atmosphere
    -> per-star host depth state
    -> star-specific hydrogen merge state
    -> kernel-specific device uploads

invariants + star state
    -> continuum absorption/scattering
    -> shared float32 line slab
    -> metal/autoionizing + helium + hydrogen + molecular deposits
    -> LTE line source + zero standard line scattering
    -> stacked total/continuum transfer
    -> device-side interior crop
    -> final host-result construction stage
    -> public F_lambda per nm and normalized flux
```

The opening timing is motivation, not evidence of speed. Scientific equality
gates must pass before any performance interpretation.

## 2. Reader promise, assumptions, and honest scope

By the end of the chapter, the reader should be able to:

- classify every important pipeline object as immutable window state,
  atmosphere-dependent state, device work, host control state, a cache
  product, or a public result;
- explain why wavelength rows permit broad tensor work while hydrogen walks,
  transfer depth sweeps, and some irregular profile decisions remain ordered;
- state the exact default device order and dtype policy;
- explain why line accumulation and scattering iteration remain float32 even
  when the surrounding CPU/CUDA work dtype is float64;
- reproduce the exact `WindowInvariants` key and field contract;
- explain why `metal_chunk`, molecular line chunks, and surviving-pair chunks
  bound different temporary tensors;
- distinguish a process-resident invariant bundle, a persistent derived
  catalog, and a prewarm manifest;
- explain why the requested grid is preserved bitwise while context is grown
  outward from its endpoints;
- identify permitted small host control boundaries without claiming that the
  entire star state or every control decision remains permanently on device;
- identify the last stage in which spectral tensors become public host
  float64 arrays;
- call the exact structured-atmosphere synthesis interface and interpret all
  five `Spectrum` fields;
- distinguish cold-source, persistent-warm, and process-warm timing;
- state exactly what four-regime same-atmosphere parity proves.

Define at first use:

- an **invariant** as data that does not change when the atmosphere changes
  while the window and its complete identity remain fixed;
- a **cache key** as the tuple used to decide whether an earlier derived object
  may be reused;
- a **process cache** as Python memory that disappears when the process ends;
- a **persistent cache** as a deletable derived file outside the scientific
  source tree;
- **prewarming** as deliberately constructing and inventorying persistent
  derived products before a timed scientific call;
- a **precision island** as a deliberately narrower or wider arithmetic region
  inside a calculation with another work dtype;
- a **host synchronization** as a point where device work must complete before
  Python or NumPy can inspect a value;
- a **native spectrum** as the model's intrinsic wavelength grid before an
  instrument response, observed-pixel projection, or fit.

The standard path remains one-dimensional, static, plane-parallel, LTE
synthesis from one supplied atmosphere. It has no:

- GPU atmosphere iteration;
- `torch.compile` route;
- synthesis `prange` kernel;
- public multi-atmosphere batch;
- implicit H\(_2\)O synthesis wiring;
- H\(_3^+\) synthesis source;
- NLTE population solve or PRD transfer solver;
- instrument model, observed-grid response, continuum fit, or spectrum fit.

`spectral_operator` exists in the pinned interface, but this chapter passes
`None` and returns a native spectrum. The interface receives total and
continuum wavelength-density tensors together on device before their ratio and
before the final host stage. It is a compact boundary note, not an invitation
to implement an instrument.

`source_path` is an internal comparison/reference-source input. The standard
path uses `source_path=None`, rebuilds the LTE Planck line source, and sets
standard line scattering to zero.

`keep_slabs=True` is a diagnostic host-return option. The standard public
workflow uses `keep_slabs=False`; its four optional slab fields are `None`.

## 3. Source/brief reconciliation and known limitations

The following corrections are authoritative for the first pass. They must be
stated in the implementation notes and must prevent stronger reader-facing
claims.

### 3.1 “Resident star state” is only partly literal

The detailed synthesis brief says that one star's state remains on device.
The pinned `SynthesisPipeline.__init__` instead states and implements:

> Per-depth state stays on host until each kernel uploads the arrays it needs.

`column_mass` and `temperature` receive persistent device tensors in the
pipeline object. Most population, density, temperature-helper, and
microturbulence arrays remain host NumPy float64 fields. `continuum.build_pops`
uploads its exact device dictionary and also keeps host float64 temperature
arrays for discrete Peach/Si II choices. Atomic, hydrogen, and molecular
kernels similarly convert the state fields they consume.

Therefore the chapter may claim that large opacity and transfer slabs and
window invariant tensors remain device resident. It may not claim that every
star-dependent column is uploaded once and retained as one all-device state.

### 3.2 “One host copy” means one final result-construction stage

The pinned result block calls `tensor.detach().to("cpu").numpy().astype(np.float64)`
separately for total \(H_\nu\), continuum \(H_\nu\), and normalized flux.
With `keep_slabs=True`, it makes four additional slab conversions. The
requested `wavelength_nm` array is already host NumPy float64.

The exact claim is:

- no full spectral opacity or flux tensor becomes public host output before
  the final result-construction stage;
- each requested result tensor is converted once inside that stage;
- default `keep_slabs=False` returns only the three converted spectral values
  plus the already-host wavelength coordinate.

Do not describe this as one literal device copy or one packed transfer.

### 3.3 Small host control boundaries are real

The pinned kernels contain small or discrete host reads, including host
float64 continuum brackets, atomic active-line lists, a hydrogen live-line
mask, Python Boolean decisions, and scalar `.item()` values controlling
bounded wing loops. These may synchronize a device. They do not transfer the
completed spectrum or a full depth-wavelength slab.

The chapter's host-boundary trace must distinguish:

- allowed host source parsing, identity, bracket, mask, and scalar-control
  work;
- forbidden early public spectral-array conversion;
- the required final result conversions.

A source grep asserting “no `.cpu()` before the end” would be false and is not
an acceptable test.

### 3.4 In-memory mappings are trusted more than archive inputs

`load_atmosphere_npz(...)` validates archive arrays and upgrades documented
legacy inputs. Public `synthesize(...)` loads and validates a path, but a
caller-supplied `Mapping` is copied and passed directly to the engine.
`build_structured_atmosphere(...)` constructs the native mapping but does not
itself call the complete archive validator before returning.

The chapter must:

- show archive validation as the exact public checked path;
- run the Chapter 2 schema validator explicitly on an in-memory mapping before
  synthesis;
- use `save_structured_atmosphere(...)` followed by
  `load_atmosphere_npz(...)` when demonstrating a fully checked round trip.

It must not state that every public in-memory call automatically performs the
same validation as archive loading.

### 3.5 Cache safety claims require narrower wording

The intended rule is that caches accelerate a source-defined calculation and
are deletable. The exact pinned checks do not justify the blanket statement
that every semantically corrupt or identity-mismatched cache rebuilds:

- the in-process window key fingerprints files by resolved path, size, and
  modification time, not by content hash;
- the atomic persistent-cache filename is derived from source stat and grid
  identity, but `LineCatalog.from_npz(...)` does not compare the stored
  metadata with the expected key; a malformed archive rebuilds, while a
  well-formed wrong archive placed at the expected name may be accepted;
- the compiled molecular cache does compare its stored `cache_identity` and
  rebuilds on parse, field, JSON, or identity failure;
- an explicitly injected `window_invariants` bundle is checked only against
  the first seven key entries—window, `resolution`, molecular flag, device,
  dtype, and `metal_chunk`—not the trailing context/file identities;
- a reused prewarm manifest checks the exact required artifact paths, sizes,
  and SHA-256 hashes, but a forced rebuild still relies on each underlying
  loader's validation behavior.

The first-pass chapter demonstrates malformed-cache recovery, molecular
identity recovery, and prewarm-manifest hash rejection. It also contains a
source-characterization test for the well-formed atomic-cache limitation and
labels that case as a pinned limitation, not a successful safety gate.

If the progressive package later hardens the atomic cache or injected-bundle
validation, that is an explicit reliability improvement. It must be
source-difference documented and output-parity checked; it must not be
misreported as the pinned behavior.

### 3.6 Timing fields are narrower than a profiler

Only `_build_window_invariants(...)` populates
`WindowInvariants.build_profile`. The timing hooks inside
`SynthesisPipeline.__init__` and `.run()` are no-op hooks and return no
per-stage run profile.

`Spectrum.seconds` is the elapsed time returned by
`synthesize_structured_atmosphere(...)`: it begins before pipeline
construction and ends after `SynthesisPipeline.run(...)`, including the final
`SpectrumResult` host construction. It excludes the later public `_wrap(...)`
conversion from \(H_\nu\) to \(F_\lambda\).

The chapter may add external timing around a call, but it must label that
measurement as notebook instrumentation, not a new production field.
`ForwardTimings` belongs to the Chapter 15 label workflow.

### 3.7 Prewarm has a fixed policy

Pinned `prewarm(...)` always builds:

- `molecular_lines=True`;
- `runtime_device=torch.device("cpu")`;
- `work_dtype=torch.float64`;
- one persistent atomic window catalog;
- one persistent compiled molecular window catalog.

It does not persist a CUDA/MPS `WindowInvariants` object. Its `identity`
mapping contains `schema_version` in addition to the identity fields listed
in the older brief.

### 3.8 The optional boundaries exist even though the chapter does not use them

The exact source contains:

- `spectral_operator` on `run`, `synthesize_structured_atmosphere`, and public
  `synthesize`;
- `Spectrum.save_npz(...)`;
- diagnostic `keep_slabs=True`;
- reference `source_path`.

The native first-pass calculation passes the first three options as
`None`/`False` as appropriate and does not teach spectrum serialization.
Omitting these branches from the scientific path is not evidence that the
interfaces do not exist.

### 3.9 Superseded exercise wording

The detailed Chapters 5–10 audit ends with “suggested exercises.” The global
chapter contract supersedes it. Chapter 10 has no exercise section; every
useful variation is executed and interpreted inline.

## 4. Notation and exact coordinate contract

Let:

- \(D\): physical depth layers, outermost to innermost;
- \(W\): exact requested public wavelength samples;
- \(C=16\): context samples on each side;
- \(W_{\rm synth}=W+2C=W+32\): internal synthesis samples;
- \(L_{\rm a}\): selected atomic records;
- \(L_{\rm m}\): compiled molecular records;
- \(R_{\rm grid}=\lambda/\Delta\lambda\): intrinsic logarithmic grid density.

At source interfaces in this chapter the exact argument and field name is
`resolution`, not `r_grid`. It is not instrumental resolving power.

For `atomic_lines.Grid`,

\[
q = 1 + \frac{1}{\texttt{resolution}},
\qquad
\lambda_{i+1}=q\lambda_i .
\]

The requested grid is built once in host NumPy float64. Blue context is
generated by repeated division by \(q\) from its first element, red context by
repeated multiplication from its last element. The internal context `Grid`
uses `nextafter`-expanded bounds for catalog metadata, but its own `.build()`
is not the synthesis array. The invariant

```python
np.array_equal(
    synthesis_wavelength_nm[output_slice],
    wavelength_nm,
)
```

must hold.

Use the following quantities without renaming:

| symbol | meaning | exact name | shape and unit |
| --- | --- | --- | --- |
| \(\lambda\) | requested native wavelength | `wavelength_nm` | `(W,)`, nm, host float64 |
| \(\lambda_{\rm synth}\) | context-expanded wavelength | `synthesis_wavelength_nm` | `(W+32,)`, nm, host float64 |
| \(m\) | column mass | `column_mass` | `(D,)`, g cm\(^{-2}\) |
| \(T\) | temperature | `temperature` | `(D,)`, K |
| \(\rho\) | mass density | `mass_density` | `(D,)`, g cm\(^{-3}\) |
| \(n_e\) | electron density | `electron_density` | `(D,)`, cm\(^{-3}\) |
| \(\xi\) | microturbulence | `microturbulence` | `(D,)`, cm s\(^{-1}\) |
| \(K_{\lambda,\rm c}\) | continuum absorption per mass | `continuum_absorption` | `(D,W_{\rm synth})`, cm\(^2\) g\(^{-1}\) |
| \(\Sigma_{\lambda,\rm c}\) | continuum scattering per mass | `continuum_scattering` | `(D,W_{\rm synth})`, cm\(^2\) g\(^{-1}\) |
| \(K_{\lambda,\rm l}\) | total line absorption per mass | `line_mass_absorption_coefficient` | `(D,W_{\rm synth})`, cm\(^2\) g\(^{-1}\), float32 accumulation |
| \(S_{\nu,\rm l}\) | LTE line source | `line_source` | `(D,W_{\rm synth})`, \(B_\nu\) unit |
| \(H_{\nu,\rm total}\) | total Eddington flux | `eddington_flux_total_per_frequency` | `(W,)` after crop |
| \(H_{\nu,\rm cont}\) | continuum Eddington flux | `eddington_flux_continuum_per_frequency` | `(W,)` after crop |
| \(F_{\lambda,\rm total}\) | public surface flux density | `Spectrum.flux_total` | `(W,)`, erg s\(^{-1}\) cm\(^{-2}\) nm\(^{-1}\) |
| \(F_{\lambda,\rm cont}\) | public continuum flux density | `Spectrum.flux_continuum` | `(W,)`, same unit |
| \(f_{\rm norm}\) | native total/continuum ratio | `Spectrum.normalized_flux` | `(W,)`, dimensionless |

The exact conversion, invoked rather than rederived, is

\[
F_\lambda =
4\pi H_\nu\frac{c_{\rm nm\,s^{-1}}}{\lambda_{\rm nm}^2}.
\]

## 5. Structured-atmosphere input contract

### 5.1 Archive and mapping inputs

The canonical static schema version is 4. `load_atmosphere_npz(...)` accepts
documented legacy versions 1–3 and reconstructs
`ion_stage_populations` for pre-v3 inputs, but the main chapter fixture is
native schema v4 and needs no compatibility upgrade.

The required physical array names are:

```text
temperature
gas_pressure
electron_density
mass_density
column_mass
partition_normalized_populations
ion_stage_populations
fractional_doppler_widths
hydrogen_neutral_population
helium_neutral_population
helium_singly_ionized_population
molecular_hydrogen_population
hydrogen_partition_normalized_ion_stage_populations
carbon_partition_normalized_ion_stage_populations
magnesium_neutral_partition_normalized_population
aluminum_neutral_partition_normalized_population
silicon_neutral_partition_normalized_population
iron_neutral_partition_normalized_population
hc_over_kt
microturbulence
elemental_abundances
signed_continuum_edge_frequency_hz
continuum_edge_wavelength_nm
continuum_edge_midpoint_wavelength_nm
continuum_edge_interval_width_squared_over_two_nm2
```

`atmosphere_schema_version` is the typed schema marker.
`hydrogen_ionized_population` is a supported extra field constructed by the
column builder and consumed when present.

Shape locks:

- the scalar depth columns are `(D,)`, with \(D\ge2\);
- `partition_normalized_populations`, `ion_stage_populations`, and
  `fractional_doppler_widths` are `(D,6,139)`;
- the hydrogen and carbon normalized stage arrays are `(D,2)`;
- `elemental_abundances` is one-dimensional with at least 99 positive linear
  number fractions;
- signed edge frequency and edge wavelength share `(E,)`, \(E\ge2\);
- edge midpoint and half squared interval width share `(E-1,)`.

All arrays are numeric and finite. Temperature, gas pressure, electron
density, mass density, column mass, and `hc_over_kt` are strictly positive.
Column mass is strictly increasing inward. Microturbulence and populations are
nonnegative.

### 5.2 Exact column-builder boundary

The public builder is:

```python
build_structured_atmosphere(
    *,
    temperature,
    column_mass,
    gas_pressure,
    electron_density,
    elemental_abundances,
    mean_nuclear_mass_amu=None,
    microturbulence=None,
    mass_density=None,
    molecular_lines=True,
    device=None,
    dtype=None,
    eos_tolerance=1.0e-5,
)
```

It uses the supplied `electron_density` as the fixed-\(n_e\) bridge state; it
does not rerun charge closure. It computes mean nuclear mass when omitted,
constructs atomic and optional molecular populations, Doppler fractions,
thermodynamic helpers, and continuum edge fields, and returns the native
mapping.

Chapter 10 invokes the Chapter 3–4 fixed-\(n_e\) implementation and shows one
sentinel proving the supplied electron density is preserved. It does not
rederive the EOS or molecular Newton solve.

The exact save boundary is:

```python
save_structured_atmosphere(atmosphere, path) -> tuple[str, ...]
```

The returned names are the required public fields. The chapter demonstrates
one builder → Chapter 2 validation → save → load → array identity round trip.
It does not time this as a production synthesis stage.

## 6. Exact runtime and public interfaces

### 6.1 Device policy

```python
resolve_runtime(
    requested_device: torch.device | str | None = None,
    requested_dtype: torch.dtype | None = None,
) -> tuple[torch.device, torch.dtype]
```

Default priority is CUDA, then MPS, then CPU. Default work dtype is float64 on
CUDA/CPU and float32 on MPS. Public callers may explicitly request float32 on
CUDA or CPU. MPS float64 raises `ValueError`.

The public string layer accepts `None`/`"auto"`, device names `"cpu"`,
`"cuda"`, and `"mps"`, and dtypes `"float32"` and `"float64"`.

### 6.2 `WindowInvariants`

The exact dataclass fields, in order, are:

```text
key
device
dtype
molecular_lines
metal_chunk
grid_obj
synthesis_wavelength_nm
wavelength_nm
output_slice
n_synthesis_wl
n_wl
n_atomic
atomic_kernel_catalog
has_metal
has_helium
has_hydrogen
continuum_tables
transfer_tables
metal_invariant_chunks
helium_invariants
hydrogen_invariants_template
molecular_invariants
n_molecular
build_profile
```

`WindowInvariants.n_wl` is the requested count \(W\);
`n_synthesis_wl` is \(W+32\). Inside `SynthesisPipeline`, the historical
attribute `self.n_wl` is assigned `bundle.n_synthesis_wl`. The chapter must
not infer its meaning from the short name.

The hydrogen invariant is a template built with a one-depth placeholder
electron density. A per-pipeline `dataclasses.replace(...)` installs
`merge_wavenumber_by_depth` computed from the actual atmosphere.

The cache-aware accessor is:

```python
window_invariants_for(
    *,
    wl_start_nm,
    wl_end_nm,
    resolution,
    molecular_lines,
    runtime_device,
    work_dtype,
    tables_path=.../"line_profile_tables.npz",
    transfer_tables_path=.../"transfer_tables.npz",
    continuum_tables_path=.../"continuum_tables.npz",
    metal_chunk=None,
) -> WindowInvariants
```

The process-cache controls are:

```python
window_invariant_cache_enabled() -> bool
clear_window_invariant_cache() -> None
```

### 6.3 Pipeline boundary

```python
SynthesisPipeline(
    atmosphere,
    source_path=None,
    wl_start_nm=400.0,
    wl_end_nm=900.0,
    resolution=20000.0,
    tables_path=.../"line_profile_tables.npz",
    transfer_tables_path=.../"transfer_tables.npz",
    continuum_tables_path=.../"continuum_tables.npz",
    molecular_lines=True,
    device=None,
    dtype=None,
    metal_chunk=None,
    window_invariants=None,
)
```

and:

```python
SynthesisPipeline.run(
    keep_slabs=False,
    spectral_operator=None,
) -> SpectrumResult
```

`SpectrumResult` fields, in order, are:

```text
wavelength_nm
eddington_flux_total_per_frequency
eddington_flux_continuum_per_frequency
normalized_flux
continuum_absorption
continuum_scattering
line_mass_absorption_coefficient
line_source
spectral_operator_seconds
spectral_operator_name
```

The four slab fields are `None` under the standard default
`keep_slabs=False`. `SpectrumResult` arrays are host NumPy float64 after the
final result-construction stage.

The engine wrapper is:

```python
synthesize_structured_atmosphere(
    atmosphere,
    *,
    wavelength_start_nm=400.0,
    wavelength_end_nm=900.0,
    resolution=20000.0,
    molecular_lines=True,
    device=None,
    dtype=None,
    spectral_operator=None,
    window_invariants=None,
) -> tuple[SpectrumResult, float]
```

The public boundary is:

```python
synthesize(
    atmosphere_npz,
    *,
    wavelength_start_nm=400.0,
    wavelength_end_nm=900.0,
    resolution=20000.0,
    molecular_lines=True,
    device=None,
    dtype=None,
    spectral_operator=None,
) -> Spectrum
```

Public `synthesize` has `resolution` only. Do not add an `r_grid` alias.

`Spectrum` has exactly five fields:

```text
wavelength_nm
flux_total
flux_continuum
normalized_flux
seconds
```

The three spectral arrays and wavelength are host NumPy float64.
`normalized_flux` is the native total-to-continuum ratio returned by the
transfer solve. `seconds` has the narrower timing scope in Section 3.6.

## 7. Window-invariant versus star-dependent state

### 7.1 Reusable window state

The chapter must classify the following as invariant for one complete key:

- requested and context grid metadata;
- requested and context host wavelength arrays and `output_slice`;
- parsed atomic catalog and kernel mapping;
- atomic line-type flags and counts;
- line-profile tables;
- device-resident continuum tables;
- metal invariant chunks;
- helium invariants;
- hydrogen invariant **template**;
- compiled molecular catalog and device invariants when enabled;
- transfer tables and their derived diagonal;
- original `build_profile`.

`atomic_kernel_catalog` is a mixed host mapping: per-record NumPy arrays and
static tables used to construct device invariants. The word “resident” must
not be applied indiscriminately to every `WindowInvariants` field.

### 7.2 Per-star state

The following must be rebuilt or rebound for every atmosphere:

- archive/mapping view;
- `temperature`, `column_mass`, `mass_density`, `electron_density`,
  `microturbulence`, and `hc_over_kt`;
- continuum population view, including `ion_stage_populations`;
- `partition_normalized_populations` and `fractional_doppler_widths`;
- `collision_density_proxy`;
- star-specific hydrogen `merge_wavenumber_by_depth`;
- kernel-specific uploaded state tensors;
- continuum absorption/scattering;
- the shared line slab and every deposited contribution;
- LTE `line_source` and zero `line_scattering`;
- total/continuum transfer and final spectrum.

The collision proxy is exactly:

\[
\left(n_{\rm H\,I}+0.42n_{\rm He\,I}+0.85n_{\rm H_2}\right)
\left(\frac{T}{10^4\,{\rm K}}\right)^{0.3}.
\]

The chapter may display the exact keys from `continuum.build_pops(...)` once,
as an interface table:

```text
temperature
mass_density
electron_density
hydrogen_partition_normalized_ion_stage_populations
hydrogen_neutral_population
hydrogen_ionized_population
helium_neutral_partition_normalized_population
helium_singly_ionized_partition_normalized_population
helium_doubly_ionized_partition_normalized_population
helium_neutral_population
helium_singly_ionized_population
carbon_partition_normalized_ion_stage_populations
magnesium_neutral_partition_normalized_population
aluminum_neutral_partition_normalized_population
silicon_neutral_partition_normalized_population
iron_neutral_partition_normalized_population
nitrogen_neutral_partition_normalized_population
oxygen_neutral_partition_normalized_population
magnesium_singly_ionized_partition_normalized_population
silicon_singly_ionized_partition_normalized_population
calcium_singly_ionized_partition_normalized_population
hot_metal_populations
charge_square_population_sum
_temperature_host
_natural_log_temperature_host
```

This table is a composition bridge to Chapter 5, not a second derivation of
continuum population ownership.

### 7.3 Non-retention gate

No actual star-dependent tensor may be stored in a reusable invariant bundle.
The test must:

1. build one bundle;
2. construct two pipelines with different `electron_density`;
3. assert the bundle object and hydrogen template are unchanged;
4. assert the two pipeline `hydrogen_invariants` are separate objects;
5. assert their `merge_wavenumber_by_depth` values change as predicted;
6. mutate a copy of one atmosphere field and prove no invariant field changes.

A generic “no field has a depth-sized dimension” test is insufficient because
an invariant table can coincidentally have the same length as \(D\).

## 8. Exact cache and prewarm contract

### 8.1 Process cache key

`_window_invariant_key(...)` returns, in order:

```text
float(wl_start_nm)
float(wl_end_nm)
float(resolution)
bool(molecular_lines)
str(runtime_device)
str(work_dtype)
int(metal_chunk)
("symmetric_native_context", WINDOW_CONTEXT_SAMPLES)
tuple(input_files)
```

`input_files` contains `(resolved_path, size, mtime_ns)` for:

- `source_catalogs/lines/atomic_source_lines_parsed.npz`;
- line-profile tables;
- transfer tables;
- continuum tables;
- and, when `molecular_lines=True`, the molecular `manifest.json`,
  `molecular_band_lines.npz`, and `titanium_oxide_lines.npy`.

The key contains no atmosphere field. Identical accessor requests return the
same Python object when
`PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE != "1"`.

Changing any key member through the accessor causes a miss. Disabling the
cache or clearing it changes object identity but not values.

### 8.2 Persistent atomic and molecular layers

The atomic catalog cache lives under:

```text
PACKAGE_CACHE_ROOT / "atomic_lines" / "atomic_lines_<key>.npz"
```

Its key contains schema, logic version, source path/stat identity, context
window, `resolution`, sort, and isotope-correction policy.

The compiled molecular cache identity has exact keys:

```text
schema
start_wavelength_nm
end_wavelength_nm
resolution
use_energy_level_wavelengths
compiler
files
```

Its path is:

```text
<molecular compiled cache dir> /
    "compiled_molecules_<blake2b-identity>.npz"
```

The standard compiler concatenates manifest-ordered text-band and TiO arrays.
It does not invoke `compile_h2o_partridge`.

The compiler source cache and compiled-window cache are distinct. Chapter 8
already taught their contents; Chapter 10 teaches their lifecycle in the
complete window build.

### 8.3 Prewarm mapping

The exact signature is:

```python
prewarm(
    *,
    wavelength_start_nm,
    wavelength_end_nm,
    resolution,
    force=False,
) -> dict
```

The top-level keys on a completed fresh build are:

```text
schema_version
status
created_utc
identity
cache_root
prewarm_seconds
prewarm_seconds_this_call
reused
cache_inventory_before
cache_inventory
required_window_artifacts
window
```

`identity` contains:

```text
schema_version
cache_root
wavelength_start_nm
wavelength_end_nm
resolution
molecular_lines
system
machine
python
torch
source_fingerprint
window_artifact_paths
```

`window` contains:

```text
wavelength_count
atomic_line_count
molecular_line_count
build_profile
```

The two required artifact records are `atomic_catalog` and
`compiled_molecular_catalog`, each with resolved path, byte length, and
SHA-256.

The `build_profile` keys are:

```text
init.grid
init.atomic_catalog
init.atomic_catalog_mapping
init.component_flags
init.continuum_tables
init.metal_invariants
init.he_invariants
init.hydrogen_invariants_template
init.molecular_compile          # only when molecular_lines is enabled
init.molecular_catalog          # only when molecular_lines is enabled
init.molecular_invariants       # only when molecular_lines is enabled
init.radiative_transfer_tables
```

These are construction timings, not physics values and not a run-stage
profile.

### 8.4 Exact path and environment controls

Record, but do not turn into a long CLI tutorial:

- `PAYNE_ZERO_DATA_ROOT`;
- `PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT`;
- fallback `PAYNE_ZERO_SOURCE_CATALOG_ROOT`;
- `PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE`;
- `PAYNE_ZERO_SYNTHESIS_CACHE_DIR`;
- `PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1`;
- `PAYNE_ZERO_SYNTHESIS_MOLECULAR_COMPILED_CACHE`;
- `PAYNE_ZERO_SYNTHESIS_MOLECULAR_COMPILED_CACHE_DIR`;
- `PAYNE_ZERO_SYNTHESIS_REBUILD_MOLECULAR_COMPILED_CACHE`;
- `PAYNE_ZERO_SYNTHESIS_DISABLE_MOLECULAR_SOURCE_CACHE=1`;
- `PAYNE_ZERO_SYNTHESIS_MOLECULAR_SOURCE_CACHE_DIR`;
- `PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES`.

`SYNTHESIS_TABLE_DIR` and `PACKAGE_CACHE_ROOT` are resolved into module-level
constants during import. The chapter setup must therefore set data/cache
environment controls before importing the progressive synthesis modules.
Fresh-root cache tests use subprocesses rather than changing those variables
after import. The source-catalog root is resolved by a function at use time,
but the chapter should still configure all roots together before import.

The prewarm CLI accepts `--r-grid` and `--resolution` as aliases stored in
`resolution`. This does not add `r_grid` to public `synthesize(...)`.

## 9. Device, dtype, and host-boundary map

The chapter's central diagram and table must distinguish policy from observed
implementation:

| stage | location and dtype | reason |
| --- | --- | --- |
| archive I/O, schema validation, source parsing, identities, geometric grid | host NumPy, primarily float64/int | exact file semantics and discrete choices |
| atomic/molecular catalog cache I/O | host arrays/files | derived source products |
| continuum/transfer tables | selected device, work dtype | repeated broad algebra |
| invariant indices | selected device, int64 | gather/scatter contract |
| selected invariant strengths/damping | mixed float32/work dtype as defined by Chapters 7–8 | pinned precision and memory policy |
| most star depth columns in pipeline storage | host NumPy float64 | exact pinned on-demand upload architecture |
| `column_mass`, persistent temperature tensor | selected device, work dtype | transfer and Planck source |
| continuum algebra | selected device, work dtype, with host float64 bracket state | precise algebra plus discrete regime choices |
| shared line slab | selected device, float32 | one validated accumulation policy on every backend |
| atomic/hydrogen/molecular evaluation | selected device, mixed work dtype/float32; indices int64 | profile precision plus bounded deposition |
| scattering source iteration | selected device, float32 | Chapter 9 validated island |
| final surface weighting | selected device, work dtype | float64 on standard CPU/CUDA, float32 on MPS |
| context crop | selected device | context never becomes hidden output |
| result construction | host NumPy float64 | exact `SpectrumResult` contract |
| public \(H_\nu\rightarrow F_\lambda\) wrap | host NumPy float64 | exact public flux contract |

Default standard-policy matrix:

| backend | default work dtype | line accumulator | scattering iteration | public arrays |
| --- | --- | --- | --- | --- |
| CPU | float64 | float32 | float32 | NumPy float64 |
| CUDA | float64 | float32 | float32 | NumPy float64 |
| MPS | float32 | float32 | float32 | NumPy float64 |

Do not simulate CUDA or MPS. If unavailable, mark that backend unavailable.
Do not claim cross-backend equality until the measured tolerance record exists.

The host-boundary instrumentation must log:

- event name;
- tensor shape;
- source and destination device;
- source and destination dtype;
- whether the event is source/control work, kernel input upload, or result
  construction.

The gate forbids an early host conversion of a full `(D,W_synth)` opacity or
source slab or a `(W_synth,)` flux tensor. It permits documented host control
values and the host wavelength coordinate.

## 10. Bounded memory, chunking, and reduction policy

### 10.1 Three independent bounds

The exact defaults are:

```text
SynthesisPipeline.METAL_CHUNK = 40_000
molecular_lines.CHUNK_LINES = 500_000
molecular_lines.PAIR_CHUNK = 200_000
```

They bound different objects:

- `METAL_CHUNK` partitions selected type 0, 1, and 3 atomic records before
  invariant precomputation and line-opacity evaluation;
- `CHUNK_LINES` bounds the molecular dense cutoff tensors with shape roughly
  `(D, lines_in_chunk)` and can be overridden by
  `PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES`;
- `PAIR_CHUNK` bounds the surviving flattened depth-line pairs passed to
  wider near/far-wing work and is fixed in the pinned source.

`metal_chunk` is part of the invariant key because it changes the invariant
chunk structure and addition grouping. `CHUNK_LINES` is a runtime deposition
control, not a `WindowInvariants` key field.

### 10.2 Required memory explanation

At minimum, the shared float32 line slab costs:

\[
M_{\rm line}=4D(W+32)\ {\rm bytes}.
\]

Each live work-dtype depth-wavelength slab costs:

\[
M_{\rm work}=b_{\rm work}D(W+32)\ {\rm bytes},
\]

where \(b_{\rm work}=8\) for standard CPU/CUDA and 4 for MPS. Continuum
absorption, continuum scattering, their sum, source/scattering arrays,
work-dtype casts, transfer rows, and returned component tensors have different
overlapping lifetimes. The notebook must report an allocation trace or
stage-wise upper estimate from actual liveness; it must not present
`M_line + 2*M_work` as the complete peak.

For each chunk family, report:

- maximum records or pairs admitted;
- largest temporary shape observed;
- dtype and estimated bytes;
- backend;
- whether Python scalar/Boolean control forced synchronization.

The architecture must never allocate a dense `(D,L,W_synth)` tensor. A test
allocation spy rejects any tensor with three axes matching depth, line count,
and wavelength count in any order.

### 10.3 Reduction and parity consequences

All line families deposit into float32 accumulators. `index_put_(...,
accumulate=True)`, chunk boundaries, and repeated slab addition define
floating-point grouping. Therefore:

- the standard chunk policy is part of strict pinned full-pipeline parity;
- a fixed policy must repeat deterministically under the measured backend
  conditions before a timing is accepted;
- alternate safe `metal_chunk` or molecular chunk sizes are an inline
  sensitivity experiment and use a measured fp32 tolerance;
- do not promise bit identity across chunk sizes, devices, Torch versions, or
  nondeterministic accelerator scatter implementations;
- `PAIR_CHUNK` is not exposed as a new public control merely to create an
  exercise.

## 11. Exact complete-pipeline order

The visible composition must call the canonical pipeline in this order:

```text
1. resolve device and work dtype
2. obtain or build WindowInvariants
3. load/adapt one structured atmosphere
4. replace the hydrogen template's merge_wavenumber_by_depth
5. upload column_mass and temperature; retain the exact host star view
6. continuum absorption and scattering on synthesis_wavelength_nm
7. allocate one float32 line_mass_absorption_coefficient slab
8. add metal mask line types 0, 1, and 3 in metal chunks
9. add helium line types -3, -4, and -6
10. add hydrogen/deuterium line types -1 and -2
11. add text-band + TiO molecular opacity when molecular_lines=True
12. build LTE planck_bnu line_source and work-dtype zero line_scattering
13. call stacked total/continuum solve_spectrum
14. crop total H_nu, continuum H_nu, and normalized_flux with output_slice
15. leave spectral_operator=None for the native path
16. construct host-float64 SpectrumResult
17. wrap total/continuum H_nu as public F_lambda per nm
```

Line type 3 travels through the ordinary LTE metal accumulation mask. This is
not a PRD transfer implementation. Type 1 autoionizing opacity is included in
the same metal-chunk loop. Standard H\(_2\)O is absent because
`_compile_molecular(...)` concatenates only text-band and TiO compiler outputs.
`molecular_lines=False` disables that molecular compile/deposit only; it does
not disable the atomic, helium, or hydrogen catalog paths.

The chapter should expose stage shapes and dtype/device checks, not duplicate
the physical implementation. One compact diagnostic run may use
`keep_slabs=True` to inspect component budgets, but the public result and
timing run must use `keep_slabs=False`.

## 12. Two reader movements

### Movement 10A — What can be reused?

Question:

> Which objects belong to the window, and which must change when the star
> changes?

Required arc:

1. compare a cold first call with a second atmosphere over the same window;
2. resolve the runtime policy explicitly;
3. construct the requested grid and grow context without moving its interior;
4. build and inspect `WindowInvariants`;
5. classify host arrays, device tensors, and the hydrogen template;
6. show one exact cache key;
7. prove a process-cache hit by Python object identity;
8. perturb every key family one at a time and show misses;
9. change only the atmosphere and prove an invariant hit plus a changed
   hydrogen merge state;
10. inspect memory bounds and allowed host control boundaries;
11. prewarm persistent products and distinguish persistent-warm from
    process-warm timing;
12. inject the supported cache failures and state the atomic-cache limitation.

Movement 10A ends with a reusable invariant bundle and an exact per-star state
boundary. It does not yet claim a complete spectrum.

### Movement 10B — One native spectrum from a supplied atmosphere

Question:

> Can the already-built physical stages be composed once, in their exact
> order, to return the four checked public spectral arrays?

Required arc:

1. validate one schema-v4 archive;
2. demonstrate the fixed-\(n_e\) column builder and round trip once;
3. reuse the Movement 10A invariant bundle;
4. run continuum and inspect only its aggregate absorption/scattering slabs;
5. allocate the shared float32 line slab and call metal, helium, hydrogen, and
   molecular deposits in order;
6. build the LTE line source and zero standard line scattering;
7. invoke Chapter 9 transfer without rederiving it;
8. crop on device and audit the final host-result stage;
9. call public `synthesize(...)` and inspect exact `Spectrum` fields;
10. run hot, solar, low-gravity giant, and cool molecule-rich fixtures through
    the same code;
11. compare all four public arrays with pinned same-atmosphere goldens;
12. compare cold and warm results before interpreting timing;
13. measure CUDA and MPS only where present and under separate tolerance
    profiles.

The movement ends at native same-atmosphere synthesis. It does not initialize
labels or accept an atmosphere physically.

## 13. Visible code-cell ledger

Target: 18 substantial visible cells. Setup-only imports and short markdown
tables do not count. Each substantial cell should normally remain 10–30 lines,
with 60 lines a soft ceiling and 80 a hard ceiling.

| cell | causal purpose | required visible result |
| ---: | --- | --- |
| 1 | load two checksum-bound schema-v4 fixtures and state the predictions | exact reads/shapes/units plus different `electron_density` |
| 2 | resolve CPU/CUDA/MPS defaults and reject invalid MPS float64 | selected device/dtype policy table |
| 3 | build requested and 16+16 context grids | `W`, `W+32`, `output_slice`, bitwise interior identity |
| 4 | construct one cold `WindowInvariants` bundle | exact field/count table and `build_profile` |
| 5 | classify invariant, host-star, device-star, and output state | no star-dependent invariant retention |
| 6 | exercise process hit, disable, clear, and key perturbations | object identity and value-identity ledger |
| 7 | run persistent prewarm and controlled failure injections | manifest keys, artifact hashes, supported rebuilds, known atomic limitation |
| 8 | replace the hydrogen template for two atmospheres | shared template plus changed `merge_wavenumber_by_depth` |
| 9 | trace dtype/device/host events and estimate live memory | event table and no `(D,L,W)` allocation |
| 10 | run continuum on the context grid | absorption/scattering shape, unit, dtype, device checks |
| 11 | allocate float32 slab and add metal then helium | component order, slab dtype, bounded metal chunks |
| 12 | add hydrogen and enabled molecular opacity | star-specific hydrogen state, molecular chunk/pair maxima |
| 13 | build LTE source, zero line scattering, and run transfer | total/continuum \(H_\nu\), finite ratio, exact order trace |
| 14 | crop on device and construct `SpectrumResult` | requested length, three one-time spectral result conversions, optional slabs `None` |
| 15 | call public `synthesize(...)` and check flux semantics | exact five fields, host float64 arrays, ratio identity |
| 16 | build from columns, validate, save, reload | fixed-\(n_e\) sentinel and complete round-trip identity |
| 17 | run cold-source, persistent-warm, and process-warm timing | equality before an environment-labelled timing figure |
| 18 | run the four same-atmosphere regimes and backend gates | four-output parity table and native normalized-flux figure |

Long exact functions live in the progressive package. A chapter runtime helper
may prepare fixtures, instrument allocation/host events, and format ledgers.
It may not become a second synthesis pipeline or hide component order.

## 14. Original schematic and quantitative-figure plan

### 14.1 Original conceptual schematics

Create three original assets through `scripts/textbook_schematic_specs.py`.
Use the shared hand-sketched white/slate/navy/beige language without copying a
website composition.

1. **`ch10-window-versus-star-state-v1`**  
   A left stream of window, catalogs, tables, device, dtype, and chunk policy
   enters one reusable `WindowInvariants` box. Two different structured
   atmospheres enter separate star-state boxes. The same hydrogen template
   branches into two different merge-state leaves. Caption it as conceptual.

2. **`ch10-host-device-precision-map-v1`**  
   Host source parsing and float64 discrete controls at the top, selected
   device work in the center, a clearly marked float32 line/scattering island,
   device-side crop, and the final three spectral result transfers to host
   float64. Small control-return arrows must be visually different from full
   spectral-array arrows. Do not say that every star field is permanently
   resident.

3. **`ch10-context-compute-crop-v1`**  
   A requested native wavelength strip bracketed by 16 blue and 16 red context
   samples, all opacity/transfer work over the expanded strip, and
   `output_slice` removing context on device. It must communicate bitwise
   preservation of the interior, not an instrumental convolution.

Every asset requires an owned prompt, generation/source provenance, alt text,
caption, native-size and notebook-width review, and SHA-256. A schematic is
not parity evidence.

### 14.2 Professional one-panel plots

Use `book.plot_style`, white background, inward ticks, declared units, and one
interpreted claim per panel.

Required:

1. **Cold-source, persistent-warm, and process-warm wall time.**  
   One compact environment-labelled point/bar figure. Equality gates appear
   immediately before it. The caption states that timing is not a portable
   physical result and separates `Spectrum.seconds` from external total time.

2. **Four supplied atmospheres, one native window.**  
   Synthesize the canonical teaching-subset window 468.6–656.6 nm at
   `resolution=20000.0`, then plot `Spectrum.normalized_flux` over its
   molecule-rich 498.95–499.15 nm excerpt. Use four restrained regime curves
   only if all remain legible; otherwise show hot/solar first and giant/cool
   in the immediately following figure while preserving one claim per panel.
   The claim is that the same code and invariant window produce different
   spectra because the star-dependent state changes.

3. **Optional only if it adds information: measured live bytes versus the
   allocation estimate.**  
   Use several \(W\) values at fixed \(D\) and standard chunks. The claim is
   linear depth-wavelength slab scaling, not a universal GPU-memory benchmark.
   Omit it if reliable allocator measurements are unavailable.

Cache identities, backend errors, and four-output parity are tables, not
ornamental residual plots.

## 15. Self-contained source and data requirements

### 15.1 Pinned source identities audited for this contract

| pinned file | SHA-256 | Chapter 10 responsibility |
| --- | --- | --- |
| `payne_zero_synthesis/device.py` | `22e769ebed60ad3a0f2060264247e469a99afd20ec5cadb69a01b6e5fa82ea3c` | runtime device/dtype policy |
| `payne_zero_synthesis/paths.py` | `2bca3284eb1765449ab3fc87439eb603e3941213d9c1205c71aee3fd1ad30b5d` | data/source/cache roots |
| `payne_zero_synthesis/atmosphere.py` | `06b79770e4d9472093655022d53ee7fddf7cc6727206f34c0f60c57151e2cf9b` | schema load, validation, compatibility |
| `payne_zero_synthesis/pipeline.py` | `465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22` | invariants, cache, state, order, crop, result |
| `payne_zero_synthesis/prewarm.py` | `d8831e4dd342a979d261325185460b564db78c9b4dc7e908e0cb9cb9c8e5ca86` | persistent artifacts and manifest |
| `payne_zero_synthesis/synthesis.py` | `590e430b6582fbcf601a52b721d8f65073432903773a99238073a1d821fe0d0c` | builder and timed engine boundary |
| `payne_zero_synthesis/api.py` | `77718303c1e0052a520ece7fab277b3b1922c21d09b35a288596592d03310940` | public builders, flux wrap, `Spectrum` |
| `payne_zero_synthesis/atomic_lines.py` | `0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0` | grid and atomic persistent cache behavior |
| `payne_zero_synthesis/molecular_lines.py` | `14c9d07e431fa73e6d6938e9db2d11c6688e52348234e0aac37cc76e8be3dc32` | molecular resident invariants, chunk/pair bounds |
| `payne_zero_synthesis/continuum.py` | `ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b` | exact star-population adapter and host brackets |
| `payne_zero_synthesis/line_opacity.py` | `639b95c3812f1a7d227b797fa89a4d6ef9725d5f0e1284f3d49cf86844278275` | float32 atomic/helium deposition boundary |
| `payne_zero_synthesis/hydrogen_lines.py` | `81ab3ee2ca9ecd1994ddde8f01e09535c5b74f7beec5afe98a3c63b44677dcca` | invariant template and star merge state |
| `payne_zero_synthesis/radiative_transfer.py` | `52e0d1a0c4a2713294ce1b43130c5d900e54c4cf1f8b2b05058fc2d6831ff62b` | Chapter 9 transfer invoked here |
| `payne_zero_synthesis/source_catalog_molecular_compiler.py` | `b3e64b36f76228f9602490a927d7443dd701a36098573365da33e008436f633a` | persistent compiler-cache identity and standard family boundary |

The Chapter 10 source-verification gate must cover complete definitions
introduced here, not merely module presence. Exact definitions should be
byte-identical or AST-verified where the progressive module shell differs.

### 15.2 Already staged static inputs

The repository already contains and must manifest-validate:

| local static input | SHA-256 |
| --- | --- |
| `data/static/schemas/atmosphere_schema.json` | `2ba8d637e613be12ff43ce319a752616323f0341ea69f8e2391c3c244939777a` |
| `data/static/synthesis_tables/atomic_masses.npz` | `d4739fef7e03964aea5a7b2604f9585fd9095c26c58f5b7d5d040aaafeb5d117` |
| `data/static/synthesis_tables/continuum_edge_grid.npz` | `11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e` |
| `data/static/synthesis_tables/continuum_tables.npz` | `406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d` |
| `data/static/synthesis_tables/line_profile_tables.npz` | `87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4` |
| `data/static/synthesis_tables/partition_saha_tables.npz` | `83e7708b0ca989caca05532ea701318d7962c3a054e48b5529d69780fc6c1f70` |
| `data/static/synthesis_tables/transfer_tables.npz` | `64f75de9af02697c0b97b7bbf919f6fed9d646622f18859eb5f66ff66e7f7a7b` |

These are static inputs, never goldens.

### 15.3 Teaching source-catalog view

The normal chapter must run without the multi-gigabyte optional catalog tree.
It should consume the already staged Chapter 7 atomic and Chapter 8
text-band/TiO subsets through a chapter-owned, checksum-bound **source-root
view** with the filenames expected by `runtime_paths.source_catalog_path(...)`.

The view may be assembled in a disposable directory from manifest-owned
inputs. It must not duplicate scientific arrays into another opaque checked-in
archive. It needs:

```text
lines/atomic_source_lines_parsed.npz
molecules/manifest.json
molecules/molecular_band_lines.npz
molecules/titanium_oxide_lines.npy
CHECKSUMS.sha256
```

The existing source subset identities include:

- Chapter 7 atomic subset:
  `d797e747d7f557d172505bbe546c0d025dbc2c7a4e0cce831a8bdbec94573e23`;
- Chapter 8 molecular text subset:
  `c264db732dcab4f29cb29be2395f3f6af4af28749706cf4f967205bf4e3feea5`;
- Chapter 8 TiO subset:
  `204c2aa286b173c7a8125e7aa67139155522f7594acb44eabc1adac11bb6ab13`;
- Chapter 8 molecular provenance:
  `9794448f59b0e6d9ac5382ed21022d5a960193d9f947bebbfda9d6749524164d`.

The compact result is explicitly teaching-subset synthesis. Optional
full-catalog same-atmosphere parity is a separate integration gate.

### 15.4 Four structured-atmosphere fixtures and goldens

Stage one input-only schema-v4 fixture for each:

- hot, mostly atomic dwarf;
- solar-like dwarf;
- low-gravity giant;
- cool molecule-rich atmosphere.

Each fixture records source/generating command, upstream chapter boundary,
array names, shapes, dtypes, units, stellar-regime labels, atmosphere identity,
commit, and SHA-256. It contains no Payne Zero spectrum output.

Store comparison-only Chapter 10 goldens separately under
`data/golden/payne_zero/`. For every regime and declared window they contain:

- `wavelength_nm`;
- total \(H_\nu\) checkpoint where stage parity requires it;
- continuum \(H_\nu\) checkpoint where stage parity requires it;
- public `flux_total`;
- public `flux_continuum`;
- `normalized_flux`;
- catalog/table/window/cache/backend metadata.

Timing reports are environment-labelled audit products, not fixture members or
physics goldens.

The normal notebook uses 468.6–656.6 nm at `resolution=20000.0`. With the
declared teaching subsets this one window admits the ordinary/molecular
features near 499 nm, the retained He anchors, the autoionizing anchor, and
the H-alpha anchors needed to exercise the complete family call sequence and
star-dependent hydrogen merge state. Quantitative interpretation may zoom to
498.95–499.15 nm, but the computed result and parity record retain the full
declared teaching window. A full-catalog gate may use an additional declared
window, but it must not replace or silently enlarge the normal executable
chapter.

The reader runtime never imports or reads `/Users/ysting/payne-zero`.

## 16. Exact test and parity gates

### 16.1 Source, API, schema, and naming gates

- pinned commit and every source hash in Section 15.1 match;
- exact function signatures and dataclass field order match Sections 5–6;
- public `synthesize` accepts `resolution` and has no `r_grid` keyword;
- public output `Spectrum` has exactly five fields;
- `SpectrumResult` optional slabs are `None` for `keep_slabs=False`;
- schema-v4 archive validation checks every required field, shape,
  finiteness, sign rule, abundance length, and increasing `column_mass`;
- in-memory mapping validation is called explicitly by chapter code;
- fixed-\(n_e\) builder preserves the supplied electron-density array;
- build/save/load round trip preserves canonical arrays and schema version;
- `source_path=None`, `spectral_operator=None`, and `keep_slabs=False` are
  asserted for the standard reader path;
- `torch.compile` is absent from the pipeline path;
- no public multi-atmosphere batch callable is invented.

### 16.2 Grid and context gates

- `Grid.ratio == 1 + 1/resolution`;
- requested grid is host float64, strictly increasing, and inside declared
  bounds under the exact end tolerance;
- internal count is `W + 32`;
- `output_slice == slice(16, 16 + W)`;
- context interior is bitwise identical to the requested array;
- cache key contains `("symmetric_native_context", 16)`;
- changing a requested boundary or `resolution` changes the key;
- output `wavelength_nm` is bitwise identical to the requested grid;
- no context sample appears in any public or optional returned slab.

### 16.3 Invariant and star-state gates

- exact `WindowInvariants` fields and semantics match Section 6.2;
- identical accessor calls return the same object when enabled;
- disabled-cache calls and clear/rebuild calls return different objects with
  value-identical invariant products;
- every accessor key family causes a miss when changed;
- table/catalog identity perturbations use disposable copies, never source
  mutation;
- two stars reuse the same bundle;
- hydrogen template is unchanged and per-star merge state differs;
- no star-dependent state is retained by mutation and object-graph checks;
- `build_pops(...)` has the exact key set and host/device split;
- `collision_density_proxy` matches its scalar formula;
- component-presence flags and counts match the selected teaching catalogs.

### 16.4 Persistent cache and prewarm gates

- fresh prewarm has the exact top-level, identity, window, and artifact fields;
- prewarm always records CPU float64 and `molecular_lines=True`;
- second valid call reports `reused=True` and
  `prewarm_seconds_this_call == 0.0`;
- `force=True` performs the build path;
- missing required artifact invalidates reuse;
- artifact byte/size/hash mismatch invalidates reuse;
- malformed atomic cache falls back to source parse;
- malformed compiled molecular cache falls back to source compilation;
- wrong molecular `cache_identity` falls back to source compilation;
- cold-built and valid-cache-loaded catalogs are array-identical;
- process-cold/persistent-warm and process-warm spectra are array-identical
  under one fixed backend/grouping policy;
- a source-characterization test records that a well-formed wrong atomic cache
  at the expected path is not guaranteed to be rejected by the pinned loader;
- a source-characterization test records that explicit invariant injection
  checks only the seven-element key head;
- cache tests use disposable roots and leave repository data unchanged.

### 16.5 Device, dtype, and host-boundary gates

- default priority is CUDA → MPS → CPU;
- default CPU/CUDA work dtype is float64;
- default MPS work dtype is float32;
- explicit public CPU/CUDA float32 is accepted;
- MPS float64 raises;
- invariant device tensors and transfer tables are on the resolved device;
- index tensors are int64;
- shared line slab is float32 on every backend;
- scattering source iteration is float32 on every backend;
- final surface weighting follows work dtype;
- documented host control arrays/scalars are classified rather than forbidden;
- no full opacity/source slab or flux vector crosses to host before result
  construction;
- default result construction converts total \(H_\nu\), continuum \(H_\nu\),
  and normalized flux exactly once each;
- public arrays are NumPy float64;
- unavailable backends are reported as unavailable, never passed by
  simulation.

### 16.6 Memory, chunk, order, and feature gates

- `METAL_CHUNK == 40000`, `CHUNK_LINES == 500000`, and
  `PAIR_CHUNK == 200000`;
- positive molecular chunk environment overrides are honored and invalid
  values fall back as pinned;
- allocation trace contains no dense `(D,L,W_synth)` tensor;
- largest observed line/pair temporaries respect declared chunk bounds;
- exact component call trace follows Section 11;
- one shared float32 line slab receives every enabled family;
- metal types 0, 1, and 3 run through the metal loop;
- helium precedes hydrogen; hydrogen precedes molecular;
- standard `_compile_molecular` includes text bands and TiO and omits H\(_2\)O;
- `molecular_lines=False` leaves the atomic/helium/hydrogen paths active;
- standard line source equals `planck_bnu(...)`;
- standard line scattering is zero;
- alternate safe chunk sizes are compared under a measured fp32 tolerance,
  not asserted bit-equal;
- strict pinned parity uses the standard chunk policy.

### 16.7 Stage and final scientific gates

For every regime:

- continuum absorption and scattering match Chapter 5 checkpoints;
- atomic/autoionizing, helium, hydrogen, and molecular component slabs match
  Chapters 7–8 before summation;
- all active stage outputs have exact shape/device/dtype and are finite;
- line opacity is nonnegative;
- total and continuum transfer match Chapter 9;
- zero line opacity gives total equal to continuum and normalized flux one;
- total and continuum \(H_\nu\) remain distinct from public \(F_\lambda\);
- `_surface_flux_per_wavelength_nm(...)` matches the factor \(4\pi\) and
  Jacobian independently;
- public `normalized_flux` equals `flux_total / flux_continuum` within the
  declared backend policy;
- requested wavelength is checked before any flux comparison;
- compare all four public arrays:
  `wavelength_nm`, `flux_total`, `flux_continuum`, and `normalized_flux`;
- compare cold-source, persistent-warm, and process-warm outputs;
- source subset and optional full-catalog results are labelled separately.

### 16.8 Backend tolerance gate

CPU float64 is mandatory for the first executable chapter. CUDA float64 and
MPS float32 run only when hardware is available.

For each available backend, record:

- hardware and operating-system identity;
- Python, NumPy, Torch, and driver/runtime versions;
- device and work dtype;
- line/scattering precision islands;
- window, catalogs, table hashes, atmosphere hash, cache state, and chunks;
- per-stage maximum absolute and relative error;
- final four-output error;
- tolerance selected from measured evidence before acceptance.

CPU/CUDA stage comparisons and MPS float32 comparisons use separate tolerance
profiles. No phrase such as “machine precision” substitutes for those
profiles.

## 17. Timing contract

Report three distinct states:

1. **cold source build** — fresh process, empty disposable persistent caches;
2. **persistent warm** — fresh process, valid prewarmed persistent artifacts,
   empty process invariant cache;
3. **process warm** — repeated call in one process returning the same invariant
   object.

For each timing record:

- equality gate status;
- device/dtype;
- window and counts;
- chunk policy;
- cache hit/reuse state;
- `WindowInvariants.build_profile` where a build occurred;
- prewarm `prewarm_seconds` and `prewarm_seconds_this_call`;
- `Spectrum.seconds`;
- optional externally measured total wall time, clearly labelled.

Do not compare:

- a cold compile/build on one backend with a warm call on another;
- a compact subset with a full catalog as if only hardware changed;
- MPS float32 with CPU/CUDA float64 as if precision were constant;
- a prewarm run with a steady-state spectrum call as one throughput number.

The first-pass text may report observed local timings. It must not generalize
them into a portable performance promise.

## 18. Redundancy and deferral audit

Chapter 10 must not:

- rederive Saha, molecular equilibrium, continuum, line profiles, source
  compilation, or transfer;
- call a schema fixture a physically accepted atmosphere;
- rename `resolution` to `r_grid` at this chapter's public or internal
  interfaces;
- call \(H_\nu\) `flux_total` before public conversion;
- describe every star field as permanently device resident;
- hide small host control synchronizations;
- say that three result tensor conversions are one literal memory copy;
- claim all semantically wrong atomic caches are rejected;
- invent a per-stage pipeline timing return;
- include `ForwardTimings`;
- add H\(_2\)O to the standard synthesis pipeline;
- call type-3 LTE routing a PRD transfer solution;
- expose `PAIR_CHUNK` as a new public option;
- invent a public multi-star batch;
- apply `spectral_operator` in the native parity path;
- teach spectrum saving, CLI workflows, learned labels, or atmosphere
  convergence;
- require the external Payne Zero checkout or full catalogs for the normal
  notebook;
- add an end-of-chapter exercise section.

Chapter 11 may consume Chapter 10's public synthesis boundary only as a
forward reference. It must not move synthesis orchestration into the
atmosphere pass.

## 19. Chapter summary and causal Chapter 11 handoff

End with `## 10.N Chapter summary`. It contains no new interface, term, or
claim.

The summary must state:

1. a wavelength window owns reusable grids, catalogs, tables, and invariant
   tensors;
2. a supplied atmosphere owns densities, populations, Doppler widths,
   collision state, and the hydrogen merge boundary;
3. the exact default policy is CUDA then MPS then CPU, with standard float64
   work on CUDA/CPU, float32 work on MPS, and float32 line/scattering islands;
4. context is computed on both sides and cropped on device without changing
   requested wavelength samples;
5. metal, helium, hydrogen, and text-band/TiO molecular opacity enter one
   float32 line slab in fixed order and bounded chunks;
6. total and continuum transfer remain device calculations until the final
   result-construction stage;
7. public `Spectrum` contains host float64 wavelength, total
   \(F_\lambda\), continuum \(F_\lambda\), normalized flux, and elapsed
   seconds;
8. cold and warm cache states may change time but not the fixed-policy
   scientific result;
9. same-atmosphere parity validates synthesis for the supplied atmosphere and
   does not accept that atmosphere physically.

State the exact output now available:

```text
validated supplied schema-v4 atmosphere
    + exact window/source/table/cache/runtime identity
    -> Spectrum.wavelength_nm
    -> Spectrum.flux_total          # F_lambda per nm
    -> Spectrum.flux_continuum      # F_lambda per nm
    -> Spectrum.normalized_flux
    -> Spectrum.seconds
```

The unresolved problem is physical closure. Chapter 10 can synthesize any
valid supplied atmosphere, but it has not constructed the 30,000-frequency
blanketed opacity state or advanced one atmosphere pass.

Close with:

### Next: make the supplied atmosphere physical

> A validated atmosphere can now produce a complete native spectrum without
> changing the synthesis boundary. The missing task is to construct that
> atmosphere rather than merely supply it: validate a seed, impose the exact
> grid and hydrostatic state, and assemble the 30,000-frequency blanketing
> opacity used by one physical pass. [Chapter 11](/reader.html?ch=11) begins
> that CPU atmosphere calculation while preserving the Chapter 10 synthesis
> interface unchanged.
