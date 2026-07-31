# Chapter 13 first-pass contract — Correction and the Full Numba Iteration

Status: authoritative complete-spine-first design; no implementation or publication authority  
Pinned Payne Zero commit: `9c44001feae40b85146630499e6f8a5fed42e5af`  
Audience: final-year undergraduate / first-year graduate student  
Canonical title: **Correction and the Full Numba Iteration**

## 0. Canonical placement and ownership

Chapter 13 closes the physical atmosphere loop that Chapters 11 and 12 leave
open:

```text
Chapter 11 (pending): validated explicit seed
                      + populations, opacity, and reusable catalog handles
                                  |
                                  v
Chapter 12 (pending): frequency-integrated radiation
                      + Rosseland state + material response + convection
                                  |
                                  v
Chapter 13: safe temperature/column correction
            -> complete common-grid remap
            -> carried-state iteration
            -> structural stopping
            -> one terminal quantization
                                  |
                                  v
        terminal AtmosphereRunResult from one explicit seed
                                  |
                                  v
Chapter 14: place difficult labels inside the exact solver's basin
```

The chapter owns:

- mode-3 `apply_temperature_correction`;
- the exact ordering of its flux, lambda-diagonal, and surface temperature
  terms;
- correction clamps, previous-step damping, finite replacement, optional
  smoothing, and the one-kelvin inward-rise rule;
- the old/new pressure response used to correct column mass;
- `TemperatureCorrectionResult`;
- the complete native-to-standard `IterationRemap`;
- the exact full-pass order in `run_atmosphere_model`;
- the distinction among pass-local, reset-in-place, loop-carried,
  process-cached, persistent-compiled, diagnostic, and terminal-product state;
- the pass-2 hydrostatic/radiation-pressure lifecycle;
- selected and detailed line-catalog object reuse across passes;
- structural convergence norms, minimum-pass eligibility, and consecutive
  qualifying passes;
- fixed-chunk `prange` orchestration and fixed-order reduction as they affect a
  complete atmosphere run;
- `configure_numba_cache`, atmosphere `prewarm`, and honest cold/compiled/warm
  timing;
- exactly one terminal fixed-column format/parse operation;
- `AtmosphereRunResult`, its base diagnostics, its optional debug outputs, and
  the converged-only schema-v4 product gate;
- the supported and rejected controls of the exact one-seed runner.

It does **not** rederive or reimplement:

- formal radiative transfer, scattering iteration, or the meaning of
  \(J_\nu\) and \(H_\nu\), owned by Chapter 9;
- ionization, molecular equilibrium, partition functions, or specific
  internal energy, owned by Chapters 3 and 4;
- continuum or line-opacity physics, line selection, or line deposition,
  owned by Chapters 5–8 and composed for atmosphere work in Chapter 11;
- Rosseland, radiative-force, heating, or lambda-accumulator lifecycle, owned
  by Chapter 12;
- four perturbed EOS solves, their snapshot/restore discipline, or the
  mixing-length convection derivation, owned by Chapter 12;
- learned initialization, restart ordering, support projection, or the
  fixed-point basin, owned by Chapter 14;
- flux, hydrostatic, optical, schema, or spectral *scientific acceptance*,
  owned by Chapter 15.

The exact source routine `finalize_transfer_state` crosses the Chapter 12/13
teaching seam. Chapter 12 owns and explains its radiation/convection prefix;
Chapter 13 opens and explains its exact `temperature_correction_result`.
Neither chapter may invent a partial-finalization dataclass or a replacement
source API.

There are no detached exercises. Every useful variation—turning a clamp off in
an analytic fixture, changing a Boolean convergence sequence, changing a
thread count, or spying on a quantizer—appears where it answers the current
physical or numerical question and is interpreted immediately.

### 0.1 Complete-spine-first acceptance

The first executable pass is accepted at chapter level when a reader can:

1. apply the exact correction to a compact analytic state and inspect every
   returned term;
2. remap one corrected state without quantization;
3. run at least two compact passes and observe the exact state lifecycle;
4. reproduce the structural norms and consecutive-counter stopping pass;
5. prove that fixed-column quantization occurs once, after the loop;
6. obtain an exact terminal `AtmosphereRunResult` for compact hot, solar,
   giant, and cool molecule-rich fixtures;
7. prove that an unconverged solve cannot write the schema-v4 product;
8. prove that a converged product rebuilds populations from the terminal
   quantized columns;
9. distinguish Python, first-compiled, process-warm, and prewarmed execution;
10. render every required original schematic and one-claim plot; and
11. close the exact Chapter 12 input and Chapter 14 output handoffs.

This gate does not wait for every optional full catalog, every machine's
thread-count envelope, or a successful cool molecular convergence trajectory.
Those remain explicit evidence gates. A wrong pass order, stale product
population, interior quantization, broadened meaning of `converged`, hidden
external import, or false data/cache claim blocks the first pass.

## 1. The chapter's single question

Open with a compact atmosphere whose unconstrained formal correction reduces
the flux residual but produces one nonfinite layer, a local temperature
reversal, and a distorted depth coordinate. Do not begin with the full runner
or with performance.

Ask:

> How can radiative and convective imbalance become a safe new atmospheric
> structure, and when may repeating that update be called structurally
> converged?

Before showing safeguards, ask the reader to predict which of these statements
is sufficient:

1. the flux residual became smaller;
2. the temperature correction became small;
3. the corrected temperature and column coordinate remained physical; or
4. several complete passes satisfied a declared structural norm.

The opening calculation should show why no single statement replaces the
others. It then motivates the exact orbit:

\[
\begin{aligned}
\mathbf{x}^{(n)}
&\longrightarrow
\left(P_{\rm gas},n_e,\{n_{s,r}\},\{n_{\rm mol}\}\right)^{(n)}
\longrightarrow \chi_\nu^{(n)}\\
&\longrightarrow
\left(J_\nu,H_{\rm rad},g_{\rm rad},\kappa_{\rm R},\tau_{\rm R}\right)^{(n)}
\longrightarrow \left(H_{\rm conv},\Delta T\right)^{(n)}\\
&\longrightarrow (T,m)^{(n+1)}
\xrightarrow{\rm remap}\mathbf{x}^{(n+1)},
\qquad
T^{(n+1)}=T^{(n)}+\Delta T^{(n)} .
\end{aligned}
\]

Use \(H_{\rm rad}\) and \(H_{\rm conv}\) in the derivation because the exact
target is the integrated Eddington flux,

\[
H_\star=\frac{\sigma_{\rm SB}}{4\pi}T_{\rm eff}^{4},
\]

implemented as `5.6697e-5 / 12.5664 * effective_temperature**4`. When the
reader-facing prose uses \(F_{\rm rad}\) or \(F_{\rm conv}\) for familiarity,
it must state whether the quantity is physical flux \(F=4\pi H\) or the
runner's Eddington-flux-scaled internal quantity. Code and array ledgers retain
the exact executable names.

The causal answer is:

```text
integrated radiation + material response
    -> three temperature-correction terms
    -> bounded and history-aware raw correction
    -> finite, smoothed, inward-rising returned temperature
    -> old/new pressure response
    -> corrected column mass
    -> complete common-grid remap
    -> next unquantized pass
    -> structural norm + minimum/consecutive state machine
    -> one terminal fixed-column quantization
    -> optional converged-only schema-v4 product
```

## 2. Reader promise, assumptions, and honest scope

By the end of the chapter, the reader should be able to:

- explain why flux balance supplies a correction direction but not a complete
  validity or stopping rule;
- identify the flux/integral, lambda-diagonal, and surface temperature terms;
- state the exact order of every correction safeguard and explain why changing
  the order changes the algorithm;
- explain why the column-mass response uses the damped raw correction rather
  than the later smoothed/monotonic returned temperature;
- enumerate every field in a complete remap and explain why all carried
  columns must share one depth coordinate;
- explain why hydrostatic pressure is updated from pass 2 onward;
- classify every important object by lifetime;
- predict which catalogs may be reused and which opacity/population arrays
  must be recomputed;
- reproduce the exact deep and optional all-layer structural norms;
- simulate the minimum-pass and consecutive-pass counter on paper;
- explain why a small structural norm may coexist with a large flux error;
- distinguish a fixed iteration budget from convergence stopping;
- identify the one normal terminal quantization boundary;
- state exactly what the debug snapshot, diagnostic structured handoff, and
  product structured atmosphere do and do not represent;
- explain why `prange` work needs private accumulators and a deterministic
  reduction order;
- distinguish compilation, process-cache reuse, persistent Numba artifacts,
  and prewarm validation;
- run the supported CPU-float64 one-seed wrapper and interpret every base
  diagnostic.

Assume the book's earlier calculus, arrays, opacity, transfer, EOS, and
convection chapters. Define these terms at first use:

- a **correction direction** as a proposed change intended to reduce a
  residual, before physical safeguards are applied;
- a **native grid** as the current pass's Rosseland-depth coordinate;
- a **standard grid** as the fixed
  \(10^{-6.875+0.125i}\) Rosseland-optical-depth coordinate;
- a **remap** as interpolation of every carried field onto one target
  coordinate;
- **loop-carried state** as state whose new value is consumed by the next
  pass;
- a **pass-local value** as state recomputed from the current pass input;
- a **reset-in-place accumulator** as a persistent object whose accumulation
  arrays are zeroed at the start of each pass;
- a **fixed point** as a state changed by less than declared structural norms
  under one complete physical pass;
- a **qualifying pass** as an eligible pass whose configured structural tests
  all pass;
- **terminal quantization** as the single format/parse operation after the
  physical loop;
- a **thread-count envelope** as the tested tolerance allowed when reduction
  grouping changes with the number of chunks;
- **prewarming** as compiling and inventorying representative branches before
  a timed scientific call.

The supported atmosphere remains one-dimensional, static, plane-parallel,
LTE, CPU, and float64 at its public state boundary. Numba is a hard runtime
dependency for the production transfer kernel. This is not a GPU atmosphere
chapter. GPU/Metal/CUDA synthesis remains the Chapter 10 lane.

The chapter accepts one explicit `ModelAtmosphere` seed. It does not mention a
learned initializer until the final handoff, and it never presents the
initializer as part of physical closure.

## 3. Pending Chapter 11 and 12 prerequisite contracts

The Chapter 11 and Chapter 12 first-pass contracts are pending. Chapter 13
must therefore bind to the following required artifacts from the global
brief. Drafting Chapter 13 may not silently fill a missing predecessor with a
new API.

### 3.1 Required Chapter 11 artifacts

Chapter 11 must provide, with exact fields, meanings, units, shapes, and
validation:

- `ModelAtmosphere`, including its derived thermal-energy helpers, metadata,
  abundance values, and outer-to-inner depth direction;
- `AtmosphereInput`, `AtmosphereOutput`, and `AtmosphereConfig`;
- `RunSetup` from `resolve_run_setup`;
- seed validation, opacity-flag interpretation, gravity and surface-radiation
  metadata, standard Rosseland grid, and missing-profile microturbulence fill;
- the fixed-column format/parse operator \(Q\), including idempotence and the
  fact that it is a numerical boundary;
- `AtmospherePopulationState` from `prepare_population_state`;
- `OpacityState` from `prepare_opacity_state`;
- the exact source-data routes and the distinction between a preselected
  catalog and source-catalog line selection;
- `SelectedLineCatalog | None` and `LineTransitionCatalog | None` as returned
  handles whose object identity can be carried across passes;
- the rule that catalog membership/decoding may be reused while populations,
  line strengths, and opacity slabs are recomputed for each atmosphere state;
- scalar `remap_to_grid` behavior and coordinate assumptions, while deferring
  the complete corrected-state remap to this chapter.

Chapter 13 may import and call those objects. It must not rederive species
populations, opacity sampling, selection thresholds, or line deposition.

### 3.2 Required Chapter 12 artifacts

Chapter 12 must provide:

- `TransferAccumulation`, including
  `opacity_state`, `frequency_start_index`, `frequency_stop_index`,
  `rosseland_accumulator`, `radiative_pressure_state`, and
  `temperature_correction_state`;
- `TemperatureCorrectionState` lifecycle through transfer modes 1 and 2:
  the four per-pass frequency accumulators, `previous_temperature_correction`,
  and `rosseland_opacity_table`;
- the exact pass reset and fixed-order frequency-chunk accumulation;
- mode-3 Rosseland opacity/depth and `RadiativePressureState`;
- `ConvectionFiniteDifferenceSamples` and `ConvectionResult`;
- exception-safe restoration after all four EOS perturbations;
- the exact combined `finalize_transfer_state` interface and its
  `IterationFinalization` output.

The exact `IterationFinalization` consumed here is:

| field | required meaning |
| --- | --- |
| `transfer_accumulation` | the exact Chapter 12 accumulated pass state |
| `rosseland_opacity` | current native-grid \(\kappa_{\rm R}\), float64 `(L,)` |
| `rosseland_optical_depth` | current native-grid \(\tau_{\rm R}\), float64 `(L,)` |
| `radiative_pressure_state` | finalized energy density, acceleration, pressure, and surface constant |
| `temperature_correction_result` | the mode-3 result whose construction is owned here |
| `convection_result` | current optional `ConvectionResult` |
| `convection_finite_difference_samples` | current optional four-perturbation record |

Chapter 12 may call the whole source `finalize_transfer_state`, collapse the
correction fields in its inspector, and hand the exact record forward. It must
not create `FinalizedRadiation`, `RadiationOnlyFinalization`, or any other
partial substitute.

### 3.3 Fail-closed dependency rule

If either pending chapter changes an exact field name, dtype, axis, or
lifetime, Chapter 13 must be reconciled before implementation. If a predecessor
omits an artifact above, the affected Chapter 13 cell remains blocked rather
than rebuilding the omitted physics locally.

## 4. Notation, coordinates, records, and interfaces

Depth index zero is the outermost layer. Index increases inward. Column mass
and Rosseland optical depth must increase inward.

### 4.1 Physical and executable notation

| symbol | physical meaning | exact executable name | unit / shape |
| --- | --- | --- | --- |
| \(m\) | column mass | `column_mass` | g cm\(^{-2}\), float64 `(L,)` |
| \(T\) | temperature | `temperature` / `temperature_k` | K, float64 `(L,)` |
| \(P_{\rm gas}\) | gas pressure | `gas_pressure` | dyn cm\(^{-2}\), float64 `(L,)` |
| \(n_e\) | electron number density | `electron_density` | cm\(^{-3}\), float64 `(L,)` |
| \(\kappa_{\rm R}\) | Rosseland mean mass opacity | `rosseland_opacity` | cm\(^2\) g\(^{-1}\), float64 `(L,)` |
| \(\tau_{\rm R}\) | Rosseland optical depth | `rosseland_optical_depth` | dimensionless, float64 `(L,)` |
| \(H_\star\) | target integrated Eddington flux | `target_integrated_eddington_flux` | internal flux scale, scalar |
| \(H_{\rm rad}\) | integrated radiative Eddington flux | `integrated_eddington_flux` | same scale as \(H_\star\), `(L,)` |
| \(H_{\rm conv}\) | runner's convective flux contribution | `convective_flux` | same scale as \(H_\star\), `(L,)` |
| \(\delta_H\) | total flux error in percent | `flux_error_percent` | percent, `(L,)` |
| \(\Delta T_F\) | flux/integral term | `flux_temperature_derivative` | K, `(L,)` |
| \(\Delta T_\Lambda\) | lambda-diagonal term | `lambda_temperature_derivative` | K, `(L,)` |
| \(\Delta T_s\) | surface term | `surface_temperature_derivative` | K, `(L,)` |
| \(\Delta T\) | damped raw correction | `temperature_correction` | K, `(L,)` |
| \(\Delta m\) | column correction | `column_mass_correction` | g cm\(^{-2}\), `(L,)` |
| \(P_{\rm rad}^{\rm int}\) | radiation pressure integrated from acceleration | `integrated_radiation_pressure` | dyn cm\(^{-2}\), `(L,)` |
| \(P_{\rm turb}\) | turbulent pressure column | `turbulent_pressure` | dyn cm\(^{-2}\), `(L,)`; zero in the supported runner |
| \(\epsilon_{\rm deep}\) | deep structural temperature norm | `deep_layer_relative_temperature_change` | dimensionless scalar |
| \(\epsilon_{\rm all}\) | symmetric all-layer norm | `all_layer_relative_temperature_change` | dimensionless scalar |

The historical exact names ending in `_derivative` are retained even when the
array is used as a temperature step. Prose may explain the mathematical role
but may not silently rename the executable field.

### 4.2 `TemperatureCorrectionResult`

The exact mode-3 result is on the current/native Rosseland grid:

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

`temperature` is the finite-replaced, lower-bounded, optionally smoothed, and
one-kelvin-inward-rise result. `temperature_correction` is the earlier damped
raw correction. They are deliberately not guaranteed to satisfy
`temperature == temperature_k + temperature_correction` after later
safeguards.

### 4.3 `IterationRemap`

The exact complete remap is:

```text
finalization: IterationFinalization
atmosphere: ModelAtmosphere
standard_rosseland_optical_depth: float64[L]
integrated_radiation_pressure: float64[L]
turbulent_pressure: float64[L]
```

`finalization` retains the native-grid record for diagnostics. `atmosphere`
holds the next pass's unquantized standard-grid columns.

### 4.4 `AtmosphereRunResult`

The terminal public result is:

```text
atmosphere: ModelAtmosphere
iterations_completed: int
converged: bool
diagnostics: dict[str, typed values/tables]
```

`atmosphere` has passed through the one terminal fixed-column format/parse
operation. `converged` means only that enabled structural stopping reached the
required consecutive count. It does not mean flux, hydrostatic, optical,
schema, or spectral acceptance.

### 4.5 Exact visible interfaces

The chapter displays and executes these exact interfaces:

1. `apply_temperature_correction(state, *, mode, frequency_weight, column_mass, total_opacity, monochromatic_eddington_flux, mean_intensity_minus_source, monochromatic_optical_depth, planck_source, frequency_hz, h_over_kt, temperature_k, stimulated_emission, scattering_fraction, target_integrated_eddington_flux, effective_temperature, frequency_count, rosseland_optical_depth=None, rosseland_opacity=None, iteration_index=1, convection_enabled=False, convective_flux=None, previous_convective_flux=None, logarithmic_temperature_pressure_gradient=None, adiabatic_gradient=None, pressure_scale_height=None, total_pressure=None, mass_density=None, log_density_temperature_derivative_at_constant_total_pressure=None, heat_capacity=None, mixing_length=1.0, smooth_start_layer=0, smooth_stop_layer=0, smooth_left_weight=0.3, smooth_center_weight=0.4, smooth_right_weight=0.3, integrated_radiation_pressure=None, turbulent_pressure=None, surface_gravity_cgs=1.0e4, standard_log_tau_step=0.125, standard_log_tau_start=-6.875) -> TemperatureCorrectionResult | None`;
2. the private `_pressure_on_standard_depth_grid` block, displayed only in its
   mode-3 context and never promoted to a textbook API;
3. the exact Chapter 12 `finalize_transfer_state`, revisited only at its
   correction-result boundary;
4. `remap_finalized_iteration_state(finalization, *, convective_flux=None, convective_velocity=None, turbulent_pressure=None, completed_iterations=None, standard_log_tau_step=0.125, standard_log_tau_start=-6.875) -> IterationRemap`;
5. private `_copy_iteration_atmosphere(atmosphere, *, gas_pressure=None) -> ModelAtmosphere`;
6. `deep_layer_relative_temperature_change(before, after) -> float`;
7. `max_normalized_column_delta(before, after, *, floor=1.0e-300, symmetric=False) -> float`;
8. `temperature_changes_within_limits(*, deep_layer_change, all_layer_change, maximum_deep_layer_change, maximum_all_layer_change) -> bool`;
9. `finalize_remapped_iteration(remapped_iteration, *, iterations_completed, converged=False, diagnostics=None) -> AtmosphereRunResult`;
10. `run_atmosphere_model(config: AtmosphereConfig) -> AtmosphereRunResult`;
11. private `_write_debug_state_npz(path, *, remapped_iteration, opacity_state, iterations_completed) -> None`;
12. `prepare_structured_handoff_population_state(config, *, temperature_iteration_index=1, setup=None, molecular_thermal_energy_erg=None) -> AtmospherePopulationState`, diagnostic route only;
13. `save_product_structured_atmosphere(atmosphere, output_npz, *, source_catalog_root=None, molecular_lines=True, device="cpu", dtype="float64") -> Path`;
14. `configure_numba_cache() -> Path`;
15. `prewarm(*, out_dir: Path, force: bool=False) -> dict`.

The consecutive counter and pass preparation remain inline in
`run_atmosphere_model`. Do not invent `prepare_iteration_input`,
`run_one_atmosphere_pass`, `apply_correction_safeguards`, or
`update_consecutive_convergence_state`.

The long exact correction is shown as ordered, executable slices of the same
canonical progressive-package function. Each visible slice has one conceptual
purpose and a target length of 10–30 lines. The notebook must not place a
large source listing in Markdown, tell the reader to inspect an external
Payne Zero file, or maintain a second handwritten implementation.

## 5. State-lifetime ledger

The lifetime classification appears before the first full pass. It is not
left for readers to infer from variable scope.

| state | created | updated/reset | consumed | lifetime |
| --- | --- | --- | --- | --- |
| resolved top-level `setup` | before loop | immutable except pass-local `replace` | every pass and diagnostics | whole solve |
| `remapped` | `None` before loop | replaced after each correction/remap | next pass; terminal work | loop-carried |
| `previous_rosseland_table` | `None` before loop | assigned the persistent correction lookup after each pass | next opacity preparation | loop-carried handle |
| `previous_surface_radiation_pressure_constant` | seed metadata | replaced by current finalized surface constant | next pass-local setup | loop-carried scalar |
| `molecular_thermal_energy_reference` | copy of seed thermal energy | never evolved | every molecular population pass | whole solve, fixed reference |
| `TemperatureCorrectionState` object | before loop | mode 1 resets four accumulators; mode 2 accumulates; mode 3 updates history/lookup | transfer and correction | persistent object with mixed field lifetimes |
| four correction accumulators | inside persistent state | zeroed at every mode-1 start | current mode-3 correction | reset in place each pass |
| `previous_temperature_correction` | zeros before loop | mode 3 overwrites each layer | next pass damping | loop-carried history |
| `rosseland_opacity_table` | empty before loop | current `(T, Pgas, kappaR)` ingested after radiation finalization | pressure response, convection, next opacity pass | loop-carried lookup |
| selected catalog handle | `None` before loop | created/read on first enabled pass, then same object returned | every later line-opacity pass | loop-carried identity |
| detailed catalog handle | `None` before loop | read on first enabled pass, then same object returned | every later detailed-opacity pass | loop-carried identity |
| pass atmosphere | seed copy or prior-remap copy | optional hydrostatic gas pressure before population work | current pass only | pass-local |
| pass `RunSetup` | `replace(setup, ...)` | none | current pass | pass-local |
| populations, widths, line strengths | population phase | recomputed from pass atmosphere | current opacity/finalization | pass-local |
| continuum/line opacity slabs | opacity phase | recomputed | current transfer | pass-local |
| Rosseland and radiative accumulators | transfer start | reset, accumulated, finalized | current finalization/remap | pass-local |
| convection perturbation/result | finalization | recomputed or absent | current correction/remap/debug | pass-local |
| `IterationFinalization` | after transfer | none | correction inspection/remap/debug | pass-local retained in current remap |
| `iteration_timing` | pass start | filled stage by stage | appended once | per-pass record |
| convergence count | zero before loop | incremented or reset after eligible norm check | stopping decision | loop-carried integer |
| module table caches | first loader call | reused in process | later passes/calls | process-local cache |
| Numba `.nbi`/`.nbc` artifacts | first compile or prewarm | validated/reused by matching runtime/source | later processes | persistent derived cache |
| terminal quantized atmosphere | after loop | never fed back into same loop | public result/product builder | terminal only |

The chapter must make one subtle object rule visible:

```text
TemperatureCorrectionState persists
    ├── reset each pass: integrated H, J-S, heating derivative, lambda diagonal
    └── carried: previous correction, Rosseland opacity lookup
```

Reusing a catalog handle does not reuse an old opacity slab. Preserving a
correction-state object does not preserve its per-pass frequency sums.

## 6. Act I — From imbalance to a safe correction

### 6.1 Start from the residual

The exact percent flux error is

\[
\delta_H(m)=100\,
\frac{H_{\rm rad}(m)+H_{\rm conv}(m)-H_\star}{H_\star}.
\]

Predict the sign before computing a correction: excess outward flux needs a
locally cooling response; deficient flux needs a warming response, subject to
nonlocal transfer and surface behavior. The code must then show why the
correction cannot be just a signed multiple of \(\delta_H\).

### 6.2 Exact correction order

The visible narrative and canonical function must retain this order:

1. Differentiate temperature, logarithmic temperature-pressure gradient, and
   Rosseland opacity with respect to column mass.
2. Copy the current convective flux, zero the first two layers, and apply the
   fixed \(0.25,0.5,0.25\) smoothing stencil, including its inner endpoint
   rule.
3. Form the radiative-heating derivative, convection-sensitive derivative,
   and column-correction coefficient.
4. Integrate that coefficient to obtain the integrating factor.
5. Form the normalized integrated flux error.
6. Integrate the optical-depth correction and clamp it layer by layer to
   \([-\tau_{\rm R}/3,+\tau_{\rm R}/3]\).
7. Form `flux_temperature_derivative`.
8. Form `lambda_temperature_derivative`; it is active only where the
   convective ratio is below \(10^{-5}\) and \(\tau_{\rm R}<1\), and an
   inactive layer halves up to five shallower neighboring values.
9. Clip the local lambda contribution and the later surface step to
   \(\pm T_{\rm eff}/25\).
10. Compute the surface term and adjust it using the integrated step sampled
    at \(\tau_{\rm R}=0.1\) and \(2\).
11. Sum flux, lambda, and surface terms.
12. Apply previous-step sign/magnitude damping in exact branch order, then
    update `previous_temperature_correction`.
13. Apply the raw correction, replace nonfinite proposed temperatures with the
    old temperature, and enforce \(T\ge1\) K.
14. Apply configured local smoothing.
15. Walk from the inner boundary outward and enforce at least a 1 K
    temperature rise inward.
16. Separately form
    `temperature_plus_correction = temperature + temperature_correction`.
    This is the damped raw correction, not the smoothed/monotonic returned
    temperature.
17. Remap old and raw-corrected temperatures to the standard grid and solve
    the old/new total-pressure response there.
18. Remap the fractional total-pressure change to the native grid and apply
    it to column mass.

Every intermediate plot or print follows this same order. A pedagogical
refactor may split display regions but may not change floating-point grouping,
mutate an intermediate earlier, or replace a historical exact branch with a
cleaner-looking formula.

### 6.3 Previous-step damping, exactly

Damping is skipped:

- on iteration 1;
- wherever convection is enabled and `flux_ratio > 0`; or
- wherever convection is enabled and the one-based depth is at least
  `layer_count / 3`.

Otherwise:

- same sign and a larger-magnitude previous correction multiplies the current
  correction by `1.25`;
- a sign reversal multiplies the current correction by `0.5`.

The same-sign branch is evaluated before the sign-reversal branch. The
possibly changed value is then stored in
`state.previous_temperature_correction`.

The main text uses three tiny layer fixtures—first pass, same-sign shrinkage,
and sign reversal—to make the state transition observable. It does not turn
this into an end-of-chapter exercise.

### 6.4 The pressure response is not the returned temperature

The private `_pressure_on_standard_depth_grid` iteratively evaluates the
carried `RosselandOpacityTable` for a supplied standard-grid temperature and
solves total and gas pressure while accounting for integrated radiation and
turbulent pressure. It is displayed only where the old/new pressure ratio is
formed.

The column response is:

\[
r_P(\tau)=\frac{P_{\rm tot,new}(\tau)}
                 {P_{\rm tot,old}(\tau)}-1,
\qquad
\Delta m(m)=m\,{\cal R}_{\tau\rightarrow m}[r_P],
\qquad
m_{\rm corrected}=m+\Delta m .
\]

The chapter must explicitly contrast the two temperature arrays:

| use | exact array |
| --- | --- |
| returned thermal structure | safeguarded `new_temperature` |
| old/new pressure response | raw `temperature + temperature_correction` |

Replacing the second row with `new_temperature` changes the algorithm and is a
parity failure.

### 6.5 Result-level physical checks

Immediately after the analytic fixture:

- all eleven returned fields have shape `(L,)` and dtype float64;
- every finite input fixture returns finite `flux_error_percent` and the three
  displayed correction terms;
- returned temperature is at least 1 K and obeys the inward-rise rule;
- corrected column mass is positive and strictly increasing for accepted
  fixtures;
- the optical correction never exceeds \(\tau_{\rm R}/3\) in magnitude;
- the three-term sum equals the pre-damping correction at the exact checkpoint;
- the stored previous correction equals the post-damping raw correction;
- nonfinite replacement is recorded as failure evidence, not celebrated as
  physical acceptance.

## 7. Complete remap: one coordinate before one more pass

`remap_finalized_iteration_state` executes this exact order:

1. construct
   `10**(standard_log_tau_start + arange(L)*standard_log_tau_step)`;
2. use the current native `rosseland_optical_depth` as the source coordinate;
3. remap corrected column mass and safeguarded temperature;
4. remap the current runtime gas pressure and electron density;
5. remap finalized Rosseland opacity;
6. remap integrated radiation pressure;
7. remap input microturbulence and supplied/zero turbulent pressure;
8. remap finalized radiative acceleration;
9. copy supplied convection flux, replace its interior `[1:-1]` with the
   correction's smoothed convective flux, and remap it;
10. remap convective velocity;
11. copy metadata and update `surface_radiation_pressure_line`; when supplied,
    update the completed-iteration `begin_line`;
12. return `IterationRemap`.

Two details require direct interpretation:

- the pressure solve inside temperature correction determines a **column-mass
  response**; its gas-pressure solution is not installed as the remapped
  atmosphere's gas pressure;
- the next pass optionally performs hydrostatic gas-pressure integration from
  the complete prior remap and prior radiation/turbulent support.

No formatter, parser, fixed-column deck, schema writer, or quantizer appears in
the correction/remap path:

```text
correction -> complete remap -> next pass
```

The normal quantization boundaries are the already-resolved input seed and the
terminal public result. A spy test must fail if format/parse is called between
passes.

The remap check uses one constant column, one monotone analytic column, and one
full nine-column state. It reports endpoint behavior and proves that every
carried physical column has length `L` on the same target grid.

## 8. Act II — The exact full iteration

### 8.1 Initialization

`run_atmosphere_model` initializes in this order:

1. resolve and validate setup;
2. reject unsupported branches;
3. set prior `IterationRemap` and prior Rosseland-table handle to `None`;
4. read the seed surface-radiation-pressure constant;
5. copy the seed thermal-energy column as the fixed molecular thermal
   reference;
6. initialize persistent `TemperatureCorrectionState`;
7. set selected and detailed catalog handles to `None`;
8. initialize pass, timing, convergence, and consecutive-count state.

### 8.2 One pass, exactly

For pass index `iteration_index = 1, 2, ...`:

1. update `iteration_itemp += iteration_index`, producing `1, 3, 6, ...`;
2. on pass 1, copy the seed; on later passes, start from the prior unquantized
   remap;
3. on later passes, when pressure iteration is enabled, replace gas pressure
   with `integrate_hydrostatic_pressure` using the prior remapped integrated
   radiation and turbulent pressure;
4. create pass-local setup with the carried surface-radiation-pressure
   constant;
5. recompute populations, molecular state, internal energy, Doppler widths,
   and line-strength factors;
6. recompute continuum and line opacity while passing the prior Rosseland
   lookup and reusable catalog handles;
7. replace the stored catalog handles with the handles returned by the current
   `OpacityState`;
8. reset and accumulate transfer/Rosseland/radiative/correction accumulators;
9. call exact `finalize_transfer_state` with
   `temperature_iteration_seed=iteration_itemp * 10`; it finalizes Rosseland
   and radiation state, ingests the lookup, computes optional convection, and
   applies the owned mode-3 correction;
10. when convection is disabled, compute disabled-convection diagnostics only
    for the output/remap columns;
11. complete the remap;
12. carry the Rosseland lookup and new surface-radiation-pressure constant;
13. compare pass-input temperature with remapped-output temperature;
14. record structural norms and the current flux-error distribution;
15. update or reset the consecutive count after minimum-pass eligibility;
16. append the timing record and break only when the required consecutive
    count is reached.

There is no hidden one-pass wrapper. The notebook first traces two passes with
small spy objects, then displays the same inline blocks in the canonical
runner.

### 8.3 Hydrostatic and radiation-pressure lifecycle

Pass 1 has no earlier computed radiative support, so it uses the seed gas
pressure. From pass 2 onward the exact supported update is

\[
P_{\rm gas}(m)=g\,m-P_{\rm rad}^{\rm int}(m)-P_{\rm turb}(m)-P_0,
\]

with runner `pressure_constant=0`. Nonpositive values are floored to
\(\max(10^{-6}gm,10^{-30})\) and issue a `RuntimeWarning`. The chapter must
show that warning as diagnostic evidence. It may not call a floored trajectory
strictly accepted without the later Chapter 15 gate.

Radiative finalization also produces a new surface radiation-pressure
constant. That scalar is carried into the next pass-local setup and written to
the remapped metadata as `PRADK ...`.

### 8.4 Catalog reuse

On the first enabled pass:

- a supplied `selected_line_catalog_path` is read, or source catalogs generate
  a selected catalog in memory;
- a supplied detailed-transition path is read if detailed opacity is enabled.

On later passes, the exact catalog objects are passed back into
`prepare_opacity_state`. Their membership and decoded records are reused, but
line opacity is recalculated from the new temperature, electron density,
populations, and widths.

The chapter shows object identity and load/generation counters equal to one.
It must not claim a selected-line disk cache: standard source-catalog
selection is generated fresh once per solve and retained in memory.

### 8.5 Terminal lifecycle

After the loop:

1. require at least one remap, opacity state, and transfer state;
2. assemble base diagnostics from the last pass and the timing list;
3. call `finalize_remapped_iteration`;
4. update `begin_line`;
5. perform exactly one in-memory fixed-column `format_atmosphere_deck` /
   `parse_atmosphere_deck` round trip;
6. return the resulting quantized atmosphere in `AtmosphereRunResult`;
7. optionally write the debug snapshot and adjacent diagnostic handoff;
8. optionally gate the schema-v4 product on structural convergence.

Terminal quantization occurs even when `converged=False`. Product promotion
does not.

## 9. Structural convergence state machine

### 9.1 Deep norm

For the standard 80-layer state, the exact zero-based slice is `[39:L-5]`:

\[
\epsilon_{\rm deep}
=
\max_{i=39,\ldots,L-6}
\frac{|T_i^{\rm out}-T_i^{\rm in}|}{|T_i^{\rm out}|}.
\]

The denominator is the new/remapped temperature. Smaller arrays use all
layers. Shape mismatch or empty input returns `nan`; a nonfinite selected
temperature returns `inf`.

The phrase “layers 39 through layers minus 6” is ambiguous without indexing.
The chapter always gives the zero-based Python slice.

### 9.2 Optional all-layer norm

The runner calls:

```python
max_normalized_column_delta(
    before_temperature,
    after_temperature,
    floor=1.0,
    symmetric=True,
)
```

Therefore

\[
\epsilon_{\rm all}
=
\max_i
\frac{|T_i^{\rm out}-T_i^{\rm in}|}
{\max(|T_i^{\rm in}|,|T_i^{\rm out}|,1~{\rm K})}.
\]

The public helper's general defaults remain `floor=1.0e-300` and
`symmetric=False`; the runner override must not be mistaken for the helper
default.

### 9.3 Strict limits and consecutive passes

`temperature_changes_within_limits` uses strict `<`, not `<=`.
`maximum_all_layer_relative_temperature_change=None` disables the all-layer
test.

A pass increments the counter only if:

- convergence stopping is enabled;
- `iteration_index >= minimum_iterations_before_convergence`; and
- all configured structural limits pass.

Every other pass resets the counter to zero. Reaching
`required_consecutive_converged_iterations` sets `converged=True` and breaks.

If `enable_convergence_stop=False`, the run completes its fixed budget and
returns `converged=False`, even if the recorded structural norm is tiny. If
the enabled budget is exhausted before the counter threshold, it also returns
`False`.

The Boolean-sequence cell predicts and then verifies at least:

```text
minimum=3, required=2, qualifies=[T,T,T,T] -> stop at pass 4
minimum=2, required=2, qualifies=[F,T,F,T,T] -> stop at pass 5
minimum=1, required=1, stopping disabled -> never converged
```

### 9.4 Structural is not scientific acceptance

Flux-error statistics are recorded, not used in the default stopping rule. A
small \(\epsilon_{\rm deep}\) may coexist with a large
`maximum_absolute_flux_error_percent`.

The chapter's language is restricted to:

- “structurally converged under the configured norm and counter”; or
- “did not reach the configured structural stopping criterion.”

It must not shorten this to “physically valid,” “fully converged,” “accepted,”
or “ready for science.” Chapter 15 owns independent acceptance gates.

## 10. Public controls and supported branches

### 10.1 Exact configuration surface

`AtmosphereInput` contains:

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

Every path after `initial_atmosphere` defaults to `None`.

`AtmosphereOutput` contains:

```text
structured_atmosphere_path = None
diagnostics_path = None
debug_state_path = None
```

`AtmosphereConfig` contains these exact top-level controls and defaults:

| control | default |
| --- | ---: |
| `iterations` | `1` |
| `enable_molecules` | `False` |
| `enable_convection` | `True` |
| `enable_convergence_stop` | `False` |
| `minimum_iterations_before_convergence` | `3` |
| `required_consecutive_converged_iterations` | `1` |
| `maximum_deep_layer_relative_temperature_change` | `5.0e-4` |
| `maximum_all_layer_relative_temperature_change` | `None` |
| `molecular_convection_thermal_tracks_perturbation` | `True` |

`resolve_run_setup` normalizes iteration and count values to at least one,
requires positive configured thresholds, fills microturbulence only when the
input has no positive value, and resolves:

- `pressure_iteration_enabled=True` unless metadata overrides it;
- mixing length `1.25`;
- overshoot weight `0.0`;
- configured convection top-layer count `0`, which becomes 36 in the current
  convection call;
- turbulence disabled with all coefficients zero.

These fixed resolved values are implementation facts, not extra public
controls.

### 10.2 Exact diagnostic branch labels

The exact `supported_branch` values are:

```text
one_iteration_{molecule_label}_line_opacity
fixed_iteration_{molecule_label}_line_opacity
fixed_iteration_{molecule_label}_no_lines
```

where `molecule_label` is `molecules` or `no_molecules`.

Line opacity is enabled when zero-based `opacity_flags[14]` or `[16]` is true.
The “one iteration” label depends on configured `setup.iterations == 1`, not
on an early-stopped `iterations_completed` value.

### 10.3 Supported and unsupported matrix

| branch/control | exact status |
| --- | --- |
| fixed continuum-only pass(es) | supported |
| standard selected line opacity | supported with preselected or at least one supported source catalog |
| detailed line opacity | supported with `detailed_line_catalog_path` |
| atmospheric atomic/diatomic/TiO/water selection | supported when paths/flags enable it |
| atmospheric H3+ | explicit-path opt-in via `h3plus_lines_path`; not supplied by the default source-path helper |
| molecular equilibrium | supported when enabled and an equilibrium catalog resolves |
| convection on/off | supported |
| pressure iteration on/off through seed metadata | supported |
| turbulent-pressure branch | rejected by `_require_supported_run_setup` |
| zero-based `opacity_flags[13] == 1` (`HLINOP` wings) | rejected before iteration |
| NLTE populations, PRD, time dependence, spherical geometry | outside scope; no invented config switches |
| raw molecular-selector guard | do not invent one |

Missing data for an enabled branch fails before product promotion. The
separate standard synthesis omission of H2O does not disable atmospheric water
line opacity.

### 10.4 High-level defaults are not runner defaults

The later high-level `solve_structured_atmosphere` policy has
`iterations_per_trial=15`, `max_trials=2`, `initializer_seed=20260713`, and
`initializer_jitter_scale=0.01`, and constructs a more enabled config.
Those are Chapter 14 orchestration defaults. They appear here only in a
boundary note and must not replace the exact low-level defaults above.

## 11. Numba, `njit`, `prange`, chunks, and caches

Chapter 2 has already introduced Python, `njit`, `prange`, compilation
signatures, and basic timing. Chapter 13 revisits only the choices that control
the full atmosphere trajectory.

### 11.1 Actual independence map

| work | real parallel axis / ordering |
| --- | --- |
| atomic population/EOS | independent depth layers use depth `prange` |
| continuum opacity | independent frequency columns use frequency parallelism |
| source-catalog selection/deposition | fixed catalog chunks with private work |
| transfer accumulation | contiguous frequency chunks in `prange` |
| molecular equilibrium continuation | sequential where the previous continuation state is required |
| shared accumulation | private chunk arrays followed by fixed-order reduction |
| atmosphere passes | sequential; pass \(n+1\) consumes remapped pass \(n\) |

The chapter may not draw parallel atmosphere passes or describe all population
work as serial.

### 11.2 Transfer fixed-chunk behavior

`transfer_chunk_count()` returns the current Numba thread count, falling back
to the logical CPU count. The runner uses:

```text
chunk_count = min(transfer_chunk_count(), max(1, stop - start))
bounds[c] = start + ((stop - start) * c) // chunk_count
```

`accumulate_transfer_range_parallel` allocates private arrays per chunk,
executes each contiguous frequency slice with `numba.prange(chunk_count)`,
then adds chunks to the shared float64 accumulators in increasing chunk order.

For a fixed chunk count, frequency membership and reduction order are
deterministic. Changing thread count changes grouping and may change the last
few bits. Strict trajectory parity therefore fixes the thread count; a
separate measured tolerance envelope is required before comparing alternate
counts.

### 11.3 Timing without a false speed claim

One already-existing representative kernel is timed four ways:

1. small pure-Python reference;
2. first `njit` call, including compile time;
3. warm serial `njit`;
4. warm fixed-thread `prange`.

The arrays are compared before timing is interpreted. This demonstration
teaches where compilation cost and parallel work enter; it is not a benchmark
claim for the complete solver.

For the full runner, report:

- cold source/process call;
- a repeated call in the same process;
- a fresh process after prewarm;
- exact machine, Numba, llvmlite/LLVM, CPU feature, thread-count, source, and
  data-inventory context.

### 11.4 Cache-directory precedence

`configure_numba_cache`:

1. honors an existing `NUMBA_CACHE_DIR`;
2. otherwise honors `PAYNE_ZERO_NUMBA_CACHE_DIR`;
3. otherwise uses `default_numba_cache_dir`;
4. sets `NUMBA_CACHE_DIR` when it selected a fallback; and
5. updates live `numba.config.CACHE_DIR` when Numba is already imported.

The default is repository-local
`.cache/payne-zero/numba-atmosphere` in a source checkout with
`pyproject.toml`, otherwise the user's XDG cache location.

Cache files are deletable derived products. They are not physical source data,
golden outputs, or evidence that a new runtime/source combination has been
validated.

### 11.5 Exact prewarm contract

`prewarm` uses schema version 3 and runs exactly one pass for:

| name | \(T_{\rm eff}\) | \(\log g\) | `[M/H]` | `[alpha/M]` | microturbulence | molecules |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `hot` | 9000 K | 4.0 | 0.0 | 0.0 | 2 km s\(^{-1}\) | on |
| `sun` | 5777 K | 4.44 | 0.0 | 0.0 | 2 km s\(^{-1}\) | on |
| `giant` | 4500 K | 2.0 | -0.5 | +0.2 | 2 km s\(^{-1}\) | on |
| `sun_atomic_only` | 5777 K | 4.44 | 0.0 | 0.0 | 2 km s\(^{-1}\) | off |

`sun_atomic_only` forces selected and detailed line flags off and uses one
Numba thread.

The prewarm manifest records:

- source-code SHA-256 over atmosphere-package `.py` files;
- system, machine, Python, Numba, llvmlite, LLVM, host CPU, and Numba CPU
  settings;
- cache directory and artifact inventory;
- representative branch records and exactly one iteration per branch;
- required kernel/specialization completeness;
- fresh-process write-free verification;
- required source-catalog paths, sizes, modification times, and NPY payload
  validation;
- thread and logical CPU counts.

The prewarm source-catalog inventory is not itself a content-hash manifest.
Scientific catalog identity comes from the separately verified
`source_catalogs/CHECKSUMS.sha256`. The chapter must not call path/size/mtime a
cryptographic data fingerprint.

## 12. Diagnostics and files actually written

### 12.1 Base `AtmosphereRunResult.diagnostics`

Every completed run records:

- `supported_branch`;
- `layer_count`, `frequency_count`, `frequency_start_index`,
  `frequency_stop_index`;
- `line_selection_enabled`, `detailed_line_enabled`, `molecules_enabled`,
  `convection_enabled`;
- `deep_layer_relative_temperature_change`,
  `all_layer_relative_temperature_change`;
- `median_absolute_flux_error_percent`,
  `p95_absolute_flux_error_percent`,
  `maximum_absolute_flux_error_percent`;
- `maximum_deep_layer_relative_temperature_change`,
  `maximum_all_layer_relative_temperature_change`;
- `enable_convergence_stop`,
  `minimum_iterations_before_convergence`,
  `required_consecutive_converged_iterations`,
  `consecutive_converged_iterations`;
- `setup_seconds`, `total_seconds`, and `iteration_timings`.

Each per-pass timing dictionary contains:

```text
iteration
prepare_iteration_seconds
population_seconds
opacity_seconds
transfer_seconds
finalization_seconds
remap_seconds
deep_layer_relative_temperature_change
all_layer_relative_temperature_change
median_absolute_flux_error_percent
p95_absolute_flux_error_percent
maximum_absolute_flux_error_percent
convergence_seconds
total_seconds
```

Catalog hashes, source hashes, trial identity, hydrostatic residuals, and
scientific acceptance are not current base diagnostic keys.

### 12.2 `debug_state_path`

When requested, `_write_debug_state_npz` writes a schema-version-4 debug
snapshot containing these unconditional arrays:

```text
debug_schema_version
column_mass, temperature, thermal_energy_erg
gas_pressure, electron_density, microturbulence
total_nuclei_number_density, mass_density, charge_square_density
specific_internal_energy, mean_nuclear_mass_amu
elemental_abundances_by_layer
ion_stage_populations_by_packed_slot
partition_normalized_populations_by_packed_slot
rosseland_opacity, radiative_acceleration
iterations_completed, molecules_enabled
standard_rosseland_optical_depth
integrated_radiation_pressure
integrated_eddington_flux
mean_intensity_minus_source_integral
absorption_heating_derivative
diagonal_lambda_accumulator
flux_error_percent, flux_derivative
flux_temperature_derivative
lambda_temperature_derivative
temperature_correction
convective_flux, convective_velocity
```

It conditionally adds:

- `major_isotope_mass_amu`;
- `fractional_doppler_widths`;
- `partition_normalized_population_over_mass_density_and_fractional_doppler_width`;
- convection gradient, heat capacity, density derivative, adiabatic gradient,
  and pressure scale height;
- eight finite-difference energy/density sample arrays; and
- molecular catalog/equation/population arrays.

This file is a triage snapshot, not the release schema-v4 atmosphere. It mixes
the last **unquantized remapped atmosphere** with the last opacity pass's
runtime state and retained finalization arrays. That distinction must appear
beside the first debug write.

The writer creates the parent directory and overwrites the NPZ target through
`np.savez`.

### 12.3 Adjacent runtime structured diagnostic

Requesting `debug_state_path` also targets:

```text
payne_zero_runtime_state_structured_diagnostic.npz
```

in the same directory. The target is unlinked before the attempt.

- Normally it begins with the last opacity iteration's population state and
  records population source `last_opacity_iteration`.
- If structurally converged, it attempts
  `prepare_structured_handoff_population_state` on the final **unquantized
  remap** and records `final_remapped_atmosphere`.
- Failure is recorded in
  `diagnostic_runtime_structured_atmosphere_warning`.
- Success adds
  `diagnostic_runtime_structured_atmosphere_path` and
  `diagnostic_runtime_structured_population_source`.

This remains a diagnostic route. It does not replace the product builder.

### 12.4 Converged-only product

When `structured_atmosphere_path` is requested, the exact runner first unlinks
an existing target. Then:

- if `converged=False`, it writes no product and records
  `structured_atmosphere_warning`;
- if `converged=True`, it passes `result.atmosphere`—the terminal
  fixed-column-quantized columns—to `save_product_structured_atmosphere`;
- that helper calls the synthesis package's `build_structured_atmosphere`,
  which rebuilds populations from those final columns;
- success records `structured_atmosphere_path` and
  `structured_atmosphere_source="final_fixed_column_quantized_arrays"`;
- caught data/import/runtime/value failures become
  `structured_atmosphere_warning`;
- the attempted product path always adds `structured_atmosphere_seconds`.

The stale-state sentinel must prove that the product did not export populations
from the last unquantized opacity state.

### 12.5 Declared but unwritten output

`AtmosphereOutput.diagnostics_path` is declared but unused by the pinned
runner. The chapter must not promise, create, or show a diagnostics file at
that path. Base diagnostics live in the returned result.

## 13. Source, table, and data requirements

### 13.1 Authoritative source oracle

The following pinned Git blobs define the Chapter 13 source oracle:

| source | pinned Git blob | role |
| --- | --- | --- |
| `temperature_correction.py` | `82adfb17273392c44aa05582d64576cc60a96d28` | correction state, pressure response, exact mode-3 order |
| `runner.py` | `74cf63548f0b93eeff253174af50a43e460021c1` | records, remap, loop, diagnostics, output gates |
| `convergence.py` | `149628e5f6610ce9d7372f33b6ddd456b908b35c` | exact structural norms |
| `config.py` | `b8aaba69dc219a501da57fed51136a2769fb93c1` | public input/output/config fields |
| `run_setup.py` | `48864e3d1f35657391d51b50e212ec936f40f18b` | resolved controls and seed validation |
| `hydrostatic.py` | `e86836bd6cde279bf258521bff0dae5ddac6ef8b` | pass-2 gas-pressure update |
| `radiative_pressure.py` | `740df844dc3d16108c056657755984bf6ccefc86` | carried pressure meanings |
| `radiative_transfer.py` | `01bcf4ebe6bebf6133fe68d45df085025595fe77` | remap and transfer-table boundary |
| `transfer_kernels.py` | `920eb3b6870545110349ba6c89439b7a41c391f9` | fixed chunks, `prange`, fixed reduction |
| `_numba_cache.py` | `f11084adc9f5b38543123e649622214ae9137a1f` | cache precedence |
| `prewarm.py` | `47f87eedcd183558aa179d76deb68e7d8f9a907d` | representative branches and manifest |
| `source_catalogs.py` | `cdbca89b511866dc68f5deb1dcdfc40c5f645342` | catalog resolution/inventory/checksums |
| `synthesis_bridge.py` | `fe535efe9bfcef0bd3d5fd6e55cbbec21c480c34` | diagnostic and product handoffs |
| `atmosphere_io.py` | `bb991c1b0345dd677c65606ede69bf9f5185f608` | fixed-column terminal quantization |
| `continuum_opacity.py` | `6c805f73644c10a79a37f210849d6bda9649d31e` | carried Rosseland lookup |

These paths and hashes are editorial evidence. The reader-facing notebook runs
the copied progressive textbook package and never imports from
`/Users/ysting/payne-zero` or asks the reader to inspect that checkout.

### 13.2 Packed atmosphere tables inherited by the complete runner

The self-contained textbook data manifest must classify and hash the required
packed physics inputs:

```text
continuum_level_tables.npz
continuum_opacity_tables.npz
hydrogen_line_profile_tables.npz
ionization_potential_tables.npz
iron_group_partition_tables.npz
isotope_tables.npz
karzas_latter_tables.npz
line_opacity_tables.npz
molecular_equilibrium_tables.npz
packed_level_metadata.npz
radiative_transfer_tables.npz
special_partition_tables.npz
```

Chapter 13 does not re-explain these tables. It inherits their exact key,
dtype, shape, role, and hash contracts from Chapters 3–12. The transfer table
specifically supplies float64 `transfer_optical_depth_grid` and float32
`mean_intensity_operator`, `eddington_flux_operator`,
`surface_eddington_flux_weights`, and `second_moment_weights`.

The schema-v4 product additionally consumes the Chapter 2/10 synthesis
catalogs and `synthesis_tables/continuum_edge_grid.npz` through the exact
synthesis builder. These must already be copied, manifested, and
provenance-bound.

### 13.3 Full source-catalog route

The standard full atmosphere route inventories:

- every contiguous `predicted_atomic_lines_partN.npy` shard;
- `observed_atomic_lines.npy`;
- `high_excitation_lines.npy`;
- `diatomic_lines.npy`;
- `titanium_oxide_lines.npy`;
- `water_lines.npy`;
- `detailed_transition_lines.npz`; and
- `molecular_equilibrium_atmosphere.npz`.

`CHECKSUMS.sha256` is the scientific identity manifest. NPY files also pass
header/payload-completeness validation. A preselected selected-line catalog is
an alternative input. Atmospheric H3+ requires an explicit path and is not
silently inferred from the default source path map.

### 13.4 Teaching fixtures and evidence roles

Required local artifacts are:

| artifact | role |
| --- | --- |
| analytic correction fixture | derive/check terms and branches without a full atmosphere run |
| compact two-pass fixture | state-lifetime, ordering, quantization-spy, and convergence tests |
| compact hot/solar/giant/cool fixtures | complete-spine CPU-float64 integration |
| compact source/table subsets | executable teaching data, with provenance and hashes |
| pinned Payne Zero golden arrays | comparison targets loaded only after the reader computes |
| optional full catalogs | full-physics trajectory and prewarm evidence |

No computed golden field may be loaded as an input to the reader's correction
or runner.

## 14. Visible cell ledger

Every visible code cell has one conceptual purpose, preceding reads/writes
prose, a predicted result, visible output unless silence is the assertion, and
immediate interpretation. Two code cells never touch without a prose bridge.
Target length is 10–30 lines; 60 is a soft ceiling and 80 a hard ceiling.

### 14.1 Act I cells — correction and remap

| cell | purpose | canonical artifact / visible result |
| --- | --- | --- |
| 13.01 | build the deliberately unsafe analytic proposal | one invalid layer and one reversal, then the chapter question |
| 13.02 | compute \(H_\star\) and percent imbalance | units/sign prediction and numeric scale |
| 13.03 | inspect mode-1 reset versus carried fields | accumulator sums zero; history/lookup unchanged |
| 13.04 | smooth current convective flux | first two layers zero and exact stencil values |
| 13.05 | form integrating factor and clamped optical correction | printed unclamped/clamped extrema |
| 13.06 | form flux/integral term | one `(L,)` float64 term |
| 13.07 | form lambda-diagonal term | activation/taper branch made visible |
| 13.08 | form and adjust surface term | values sampled at \(\tau=0.1,2\) |
| 13.09 | sum and clip the three terms | pre-damping equality checkpoint |
| 13.10 | exercise exact damping branches | first/same-sign/sign-flip table |
| 13.11 | apply finite, 1 K, smoothing, and inward-rise safeguards | returned temperature checks |
| 13.12 | run old/new standard-grid pressure response | old/new total pressure and ratio |
| 13.13 | form corrected column mass | positivity/monotonicity and raw-vs-safeguarded sentinel |
| 13.14 | construct exact `TemperatureCorrectionResult` | field/shape/dtype inspector |
| 13.15 | remap constant and monotone scalar columns | endpoint and interpolation errors |
| 13.16 | execute complete `remap_finalized_iteration_state` | aligned field ledger, metadata, no quantizer calls |

Cells 13.04–13.14 display ordered slices of the same canonical
`apply_temperature_correction`, not newly named production helpers. Small
fixture calculations may use local variables, but no second algorithm enters
the progressive package.

### 14.2 Act II cells — loop, convergence, and operations

| cell | purpose | canonical artifact / visible result |
| --- | --- | --- |
| 13.17 | classify state lifetimes | generated carried/reset/pass-local table |
| 13.18 | trace initialization | exact creation order |
| 13.19 | trace two complete passes | numbered stage log; hydrostatic only on pass 2 |
| 13.20 | prove catalog reuse | object IDs and create/read counters equal one |
| 13.21 | prove accumulator reset/history carry | sentinel values across two passes |
| 13.22 | compute deep and all-layer norms | hand-checkable arrays and exact denominators |
| 13.23 | run minimum/consecutive Boolean sequences | exact stopping pass for each sequence |
| 13.24 | compare structural norm with flux error | separate values; no broadened convergence |
| 13.25 | spy on terminal quantization | zero interior calls, one terminal call |
| 13.26 | inspect `AtmosphereRunResult` and base diagnostics | exact key and per-pass-timing sets |
| 13.27 | request an unconverged product | no target file; warning key present |
| 13.28 | inject stale runtime populations into a converged fixture | product rebuild sentinel passes |
| 13.29 | write and inspect debug state | exact mandatory/conditional keys; unquantized warning |
| 13.30 | show fixed-chunk bounds and private reduction | chunks cover each frequency once in order |
| 13.31 | compare Python/first-`njit`/warm-`njit`/`prange` | equality first, timing second |
| 13.32 | demonstrate cache precedence | resolved path for three isolated environments |
| 13.33 | inspect a prewarm manifest fixture | runtime/source/data/artifact roles separated |
| 13.34 | run compact hot/solar/giant/cool one-pass parity | one concise regime table |
| 13.35 | run compact multi-pass trajectories | norms, count, status, and failure boundary |

The full 30,000-frequency/full-catalog runs may be separate slow evidence
targets. The chapter's normal read path remains runnable on compact data.

## 15. Original schematic and one-panel plot plan

All conceptual schematics are original to the textbook. Their generation code
and prompts/specifications live in the textbook repository. They follow the
website's restrained scientist-sketched aesthetic: white or warm-paper
background, slate blue/deep navy for the main path, soft charcoal text, warm
grey rules, pale beige state regions, one restrained orange warning accent,
slightly varied line weight, generous white space, no gradients, shadows,
logos, or decorative filler.

Data-valued plots use the professional paper-inspired style:

```text
paper  #f7f6f2
ink    #171a20
navy   #334d72
orange #b94d2a
muted  #737373
rule   #c9c6bd
```

They use a sans-serif family with Avenir Next / Helvetica Neue / DejaVu Sans
fallbacks, no top/right spines, restrained grid lines, explicit units,
color-blind-safe line styles, and no legend frame. Each plot makes one claim,
uses one panel, and is interpreted in the following paragraph.

| visual | one claim | required construction and audit |
| --- | --- | --- |
| opening correction failure | a smaller residual does not guarantee a valid structure | one panel versus \(\log_{10}\tau_{\rm R}\); old \(T\), unsafe proposal, and invalid marker; no production result implied |
| three temperature terms | flux, lambda, and surface terms dominate different depth regions | three separate one-panel plots with identical axes and restrained shared styling; each plots its one term and zero line |
| safeguard sequence | ordering changes the proposed temperature | one panel showing raw, post-finite/smoothing, and final inward-rise curves; annotate only the active safeguard |
| pressure response | a temperature proposal moves the column coordinate through total pressure | one panel of \(\Delta m/m\) versus \(\log_{10}\tau_{\rm R}\), with zero line |
| exact pass orbit schematic | one pass is ordered and state crosses the boundary selectively | landscape original schematic with the exact 15 numbered steps; pale pass-local lane, slate carried-state lane, reset glyphs, and a return arrow from unquantized `IterationRemap` |
| terminal gate schematic | remap is inside the loop; quantization/product are outside | small original schematic with a double-line terminal boundary, one \(Q\), then divergent result/debug/product paths |
| structural trajectory | stopping depends on declared structural norms and eligible consecutive passes | one panel of deep/all-layer norms versus pass with strict thresholds and eligible-pass markers |
| flux diagnostic | structural stopping does not enforce flux closure | separate one panel of median/p95/max absolute flux error versus pass |
| fixed chunks schematic | parallel chunks are private; reduction is ordered | original fan from contiguous frequency slices to private accumulator rows, then left-to-right reduction |
| compilation timing | first-call compile cost is not warm kernel cost | one panel or compact table; if plotted, log-time bars with equality gate stated above it |

The 15-step orbit labels, in order, are:

1. seed copy, or prior remap with optional hydrostatic gas pressure;
2. pass-local setup with the carried surface constant;
3. populations, molecular state, internal energy, widths, and strengths;
4. continuum and line opacity;
5. first-pass catalog construction/read or later catalog-handle reuse;
6. accumulator reset while correction history and lookup persist;
7. transfer accumulation and fixed-order reduction;
8. Rosseland finalization;
9. radiation-pressure finalization and new surface constant;
10. current \((T,P_{\rm gas},\kappa_{\rm R})\) lookup ingest;
11. four EOS perturbations and convection when enabled;
12. temperature/column correction;
13. disabled-convection diagnostics when needed, then complete remap;
14. carried lookup and surface constant;
15. structural/flux diagnostics, consecutive counter, and return or exit.

The schematic may group steps spatially but may not merge their order or put
quantization on the return arrow. Step 13 must make the disabled branch
source-faithful: disabled-convection diagnostics occur after the zero-convection
correction and provide only the convection columns used by the remap.

## 16. Tests, parity, convergence, and publication gates

### 16.1 Local correction gates

- Mode 1 resets exactly four accumulator arrays and preserves correction
  history/lookup.
- Mode 3 rejects missing Rosseland depth or opacity.
- Flux error, its derivative, and all three temperature terms match pinned
  arrays.
- Optical correction respects \(\pm\tau/3\).
- Local term clipping and all damping branches match.
- Nonfinite replacement, 1 K floor, local smoothing, and inward-rise order
  match.
- Pressure response uses the raw damped correction sentinel.
- All eleven result arrays match shapes/dtypes and pinned values.
- Corrected column mass is positive and monotone in accepted fixtures.

### 16.2 Remap and state-lifecycle gates

- Constant remap is constant and monotone fixture error is within the declared
  interpolation tolerance.
- All complete-remap fields share the standard coordinate.
- Gas/electron/Rosseland/radiation/microturbulence/turbulence/acceleration and
  convection fields are each remapped exactly once.
- Convection interior replacement is `[1:-1]`.
- No quantizer is called during correction, remap, or between passes.
- Two-pass call trace matches the exact source order.
- Hydrostatic integration is absent on pass 1 and present on pass 2 when
  enabled.
- Catalog generation/read counters equal one and object identity persists.
- Correction history and Rosseland lookup persist; pass accumulators reset.
- Molecular thermal-energy reference remains the seed copy.

### 16.3 Convergence and terminal gates

- Deep-slice boundaries are correct for 80 layers and fall back to all layers
  for smaller arrays.
- Deep denominator is `abs(after)`.
- Runner all-layer denominator is
  `max(abs(before), abs(after), 1 K)`.
- Limits use strict `<`.
- Minimum-pass and consecutive-count Boolean sequences stop on the predicted
  pass.
- Fixed-budget runs with stopping disabled remain `converged=False`.
- Budget exhaustion remains `False`.
- Flux statistics never enter the structural Boolean.
- Exactly one terminal format/parse call occurs for every completed run.
- Terminal metadata and quantized arrays match the pinned result.

### 16.4 Diagnostic and product gates

- Base diagnostic keys are exact; no editorial provenance key is invented.
- Per-pass timing dictionaries have the exact stage and metric keys.
- Debug NPZ mandatory and conditional arrays match exact dtypes/shapes.
- Debug columns are explicitly identified as unquantized-remap diagnostic
  state.
- Adjacent diagnostic population-source key follows the actual branch.
- `diagnostics_path` produces no file and is documented as unused.
- An existing requested product target is removed before gating, preventing a
  stale product from surviving an unconverged run.
- Unconverged solve writes no schema-v4 product and records the exact warning.
- Converged solve records
  `structured_atmosphere_source="final_fixed_column_quantized_arrays"`.
- A stale-runtime sentinel proves product populations are rebuilt through the
  synthesis builder from quantized terminal columns.
- Schema keys, shapes, dtypes, and arrays match the Chapter 2/10 contract.

### 16.5 Parallel/cache gates

- Every frequency belongs to exactly one contiguous transfer chunk.
- Private accumulators are reduced in increasing chunk order.
- Serial and one-chunk compiled results match at the strict gate.
- Fixed-thread repeated trajectories satisfy the strict tolerance.
- Alternate thread counts use a measured, declared ulp/tolerance envelope.
- First compiled and warm compiled results agree before timing is compared.
- Cache precedence works before and after Numba import.
- Prewarm runs all four exact branch names, one pass each.
- Required `.nbi`/`.nbc` kernels and specializations are present.
- Fresh-process representative calls do not change the stable cache inventory.
- Source-catalog content checksums are verified separately from prewarm's
  path/size/mtime inventory.

### 16.6 Strict parity ladder

Parity is promoted in this order:

1. correction intermediate arrays;
2. complete correction result;
3. complete one-pass corrected/remapped state in four regimes;
4. exact one-pass stage order;
5. multi-pass trajectories and carried/reset state;
6. stopping pass, norms, and consecutive count;
7. terminal fixed-column-quantized arrays and metadata;
8. converged schema-v4 fields and final-population rebuild;
9. same-process and fresh-process cache behavior;
10. fixed and alternate thread-count envelopes.

An earlier failed rung blocks claims based on later rungs. Optional full-data
parity may remain pending without hiding which compact rung has passed.

## 17. Failure modes and honest boundaries

- A small structural change can coexist with a large flux error.
- Clamps and damping can slow convergence; removing them defines a different
  method.
- Nonmonotone native or remapped coordinates are hard failures for an accepted
  trajectory.
- A nonfinite proposal that needed replacement is diagnostic evidence and
  cannot pass strict physical acceptance merely because the returned array is
  finite.
- Hydrostatic pressure flooring emits a warning and cannot be silently treated
  as exact closure.
- An exhausted pass budget returns a quantized terminal result with
  `converged=False`.
- Structural convergence does not imply flux, hydrostatic, optical, schema, or
  spectral acceptance.
- Fixed thread count is required for strictest trajectory reproducibility.
- Missing enabled data and explicitly unsupported branches fail before product
  promotion.
- Turbulence and HLINOP are the exact loud runner guards; do not broaden this
  list with an invented molecular or NLTE guard.
- `diagnostics_path` is not an implemented product.
- Debug and diagnostic structured files are not substitutes for the converged
  product.

### 17.1 Cool molecule-rich boundary

The cool, high-gravity, molecule-rich fixture is mandatory even when it does
not converge. It must show:

- molecular equilibrium and molecular opacity are genuinely active;
- pass state remains finite until any recorded failure;
- the structural norms, flux statistics, warnings, and terminal status remain
  visible;
- no schema-v4 product is promoted when structural stopping fails; and
- failure is described as leaving or failing to reach the one-seed solver's
  convergence basin, not as evidence that molecules are unsupported.

The chapter does not tune, jitter, project, or replace this seed. That is the
precise motivation for Chapter 14.

## 18. Narrative and redundancy audit

The final prose follows two acts:

### Act I — Correction and complete remap

1. unsafe formal proposal;
2. total flux residual;
3. three physically distinct temperature terms;
4. exact safeguard order;
5. pressure-mediated column response;
6. complete common-grid remap;
7. proof of no interior quantization.

### Act II — Repetition, stopping, and operations

1. lifetime classification;
2. exact one-pass order;
3. pass-2 hydrostatic support;
4. catalog and lookup reuse;
5. structural norms and counter;
6. terminal quantization;
7. diagnostic versus product outputs;
8. fixed-chunk parallelism and cache/prewarm;
9. four-regime integration and honest failure boundary.

Before publication, audit:

- no transfer equation or scattering iteration is rederived;
- no EOS perturbation or convection formula is rederived;
- no opacity catalog is decoded again;
- no fixed-column operator is taught again beyond its terminal role;
- no Chapter 14 initializer appears in the solver;
- every exact name follows physical motivation rather than leading it;
- every paragraph prepares a question, equation, code cell, result, or honest
  limitation;
- every figure has one claim and immediate interpretation;
- every code cell executes the same canonical progressive-package function;
- no external source checkout is needed at runtime;
- no exercises section exists;
- the closing summary introduces no new concept.

## 19. Required closing and Chapter 14 handoff

Return to the opening invalid proposal. State that the reader has now built:

- an exact, safeguarded temperature/column correction;
- a complete unquantized standard-grid remap;
- the carried-state one-seed atmosphere loop;
- a declared structural stopping machine;
- a terminal quantized `AtmosphereRunResult`;
- exact diagnostics and optional triage files; and
- a converged-only product whose populations are rebuilt from final quantized
  columns.

The summary must also state what is not yet earned: a good starting state for
every requested stellar label, and independent scientific acceptance beyond
the structural criterion.

The final navigation link uses this causal bridge:

> We can now decide whether one explicit atmospheric state lies inside the
> exact physical solver's basin. Cool and molecule-rich labels show that the
> missing problem is no longer the correction rule—it is where to start.
> [Chapter 14: Learned Initializers and Mandatory Physical Closure](../chapter14/)
> constructs candidate seeds, but every candidate must return to the unchanged
> Chapter 13 solver for the answer.

Chapter 14 consumes:

- the exact one-seed `run_atmosphere_model` map;
- complete per-pass trajectory diagnostics;
- the strict meaning of structural `converged`;
- terminal fixed-column quantization;
- the converged-only schema-v4 product gate; and
- the rule that no initializer prediction is itself a fixed point.

Chapter 14 may order and test seeds. It may not change the physical pass,
average terminal atmospheres, bypass product gating, or broaden
`converged`.
