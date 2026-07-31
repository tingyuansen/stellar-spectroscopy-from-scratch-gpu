# Chapter 12 first-pass contract — Radiation, Thermodynamics, and Convection

Status: authoritative bounded reader-facing design; no implementation or
publication authority  
Pinned Payne Zero commit: `9c44001feae40b85146630499e6f8a5fed42e5af`  
Audience: final-year undergraduate / first-year graduate student  
Canonical title: **Radiation, Thermodynamics, and Convection**

## 0. Canonical placement and ownership

Chapter 12 is the material-response chapter of the atmosphere sequence:

```text
Chapter 9:
    one-frequency optical depth, source iteration, J_nu, H_nu, J_nu-S_nu,
    and the surface second moment
                              |
                              v
Chapter 11:
    one validated current atmosphere
    + populations/EOS state
    + complete depth-major 30,000-frequency opacity state
                              |
                              v
Chapter 12:
    ordered frequency accumulation
    + Rosseland opacity/depth
    + radiation energy, force, and pressure
    + heating/lambda columns
    + EOS perturbation samples
    + convection diagnostics/result
                              |
                              v
Chapter 13:
    mode-3 temperature/column correction
    + remap
    + carried-state iteration
    + convergence and terminal product
```

This chapter owns:

- the exact `TransferAccumulation` state;
- the distinction between current-pass reset fields and persistent correction
  history/opacity lookup state;
- vectorized Planck/stimulated-emission preparation on the opacity grid;
- the exact contiguous frequency partition used by the Numba transfer path;
- nine chunk-private accumulator families and ascending-chunk reduction;
- the mathematical meanings of all eight depth accumulators and the surface
  scalar;
- Rosseland-mean opacity finalization and Rosseland optical-depth integration;
- the exact radiative-energy, acceleration, pressure, flux-cap, and surface
  pressure-constant finalization;
- the frequency-integrated heating, flux, and diagonal-lambda accumulators;
- the persistent `RosselandOpacityTable` ingest required by convection and the
  next opacity pass;
- the four ordered temperature/pressure EOS perturbations;
- the exact state snapshot and the exact, imperfect restore behavior of the
  pinned source;
- `ConvectionFiniteDifferenceSamples`;
- the thermodynamic derivatives computed from those samples;
- the local mixing-length, radiative-leakage, overshoot, top-suppression, and
  disabled-diagnostic branches;
- `ConvectionResult` and `DisabledConvectionDiagnostics`;
- the radiation/convection prefix and exact return boundary of
  `finalize_transfer_state`.

It does **not**:

- rederive the single-frequency formal transfer solve, scattering source
  iteration, diffusion continuation, or boundary moments from Chapter 9;
- reconstruct continuum or line opacity from Chapter 11;
- rederive atomic or molecular population physics from Chapters 3–4;
- teach the mode-3 temperature correction, column-mass correction, remap, or
  iteration lifecycle;
- claim that a partial frequency interval is a physically finalized
  atmosphere;
- invent an intermediate `FinalizedRadiation`, `RadiationResult`, or partial
  finalizer;
- correct or converge an atmosphere.

The exact source routine `finalize_transfer_state(...)` combines Rosseland and
radiative-pressure finalization, lookup ingestion, optional convection, and
the Chapter 13 mode-3 correction call. Chapter 12 must call that routine and
return its exact `IterationFinalization`. The correction object is present in
the inspector but collapsed and labelled “opened in Chapter 13.”

There are no detached exercises. Chunk changes, invalid input probes,
finite-difference limiting cases, and overshoot switches appear next to the
claim they test.

### 0.1 Complete-spine-first acceptance

The first executable pass is accepted at chapter level when:

1. a Chapter 11 `OpacityState` enters unchanged;
2. the CPU/NumPy-float64 plus intentional float32-transfer-table path
   accumulates a declared frequency interval;
3. all nine private accumulator families reduce in exact ascending chunk
   order;
4. the complete 30,000-frequency path runs at least one normal solar case;
5. Rosseland, radiation-pressure, and persistent-table finalization run through
   the exact combined finalizer;
6. one atomic-only and one molecule-enabled perturbation/convection branch
   return the exact structured records;
7. the exact source limitations in Section 15 are visible and tested rather
   than hidden;
8. the required schematics and one-panel plots render;
9. the Chapter 11 prerequisite and Chapter 13 handoff are executable.

Four-regime full-grid parity remains a publication gate. Compact hot, solar,
giant, and cool slices belong in the normal notebook so every regime is
visible without making each explanatory cell a full production run.

A working radiative spine does not erase a failed thermodynamic claim. In
particular, the pinned atomic-only energy-sampling limitation blocks any text
that calls that branch a full atomic-EOS heat-capacity calculation.

## 1. The chapter's single question

Open with one sampled frequency. Chapter 9 supplies its physical-depth
\(J_\nu\), \(H_\nu\), \(J_\nu-S_\nu\), total opacity, optical depth, and surface
second moment. Show nine arrows leaving that one frequency, then replace the
single arrow set by the complete opacity-sampling grid.

Ask:

> Which frequency-integrated radiation quantities and EOS derivatives
> determine radiative support and the flux that convection can carry?

Make four predictions before code:

1. one frequency cannot define a Rosseland mean, radiative support, or flux
   balance unless the deliberately special one-frequency test branch is
   invoked;
2. independent frequency chunks may run concurrently, but one frequency's
   depth recurrence and the final chunk reduction remain ordered;
3. a material derivative requires a closed perturbed EOS state, not a
   post-hoc perturbation of density alone;
4. an atmosphere may be locally convectively unstable even though this
   chapter has not yet applied a temperature correction.

The causal construction is:

```text
OpacityState from Chapter 11
    -> B_nu and stimulated-emission columns in opacity-grid order
    -> Chapter 9 one-frequency moments
    -> eight private depth accumulators + one private surface scalar
    -> ascending-chunk reduction
    -> TransferAccumulation
    -> kappa_R and tau_R
    -> radiation energy, acceleration, and pressure
    -> append (T, P_gas, kappa_R) to persistent lookup
    -> four sequential EOS perturbations with restore
    -> thermodynamic derivatives
    -> local/overshoot/suppressed convection columns
    -> exact IterationFinalization
    -> opaque TemperatureCorrectionResult for Chapter 13
```

## 2. Reader promise, assumptions, and honest scope

By the end of the chapter, the reader should be able to:

- distinguish \(J_\nu\), \(H_\nu\), and \(F_\nu=4\pi H_\nu\);
- explain why the source stores radiative and convective flux on an integrated
  Eddington-flux scale;
- write the Rosseland harmonic weighting used by the exact accumulator;
- map every single-frequency moment to every integrated destination;
- explain the reset/accumulate/finalize mode convention without assuming that
  all three modes are called as Python helpers in production;
- identify which fields survive from one pass to the next;
- state the exact `(frequency, layer)` versus `(layer, frequency)` axis
  boundary;
- explain why the line slab and transfer operators are float32 while
  accumulators are float64;
- predict the exact frequency bounds for every chunk;
- explain fixed-policy repeatability and why a different chunk count may
  change low bits;
- derive radiation energy density, acceleration, and pressure in cgs units;
- distinguish raw opacity-weighted radiative acceleration from the
  flux-capped acceleration stored after finalization;
- distinguish integrated flux error, local \(J_\nu-S_\nu\) heating imbalance,
  the opacity-gradient term, and the diagonal-lambda term;
- derive the factor `500.0` in the \(\pm0.1\%\) central differences;
- inventory every field the pinned perturbation routine snapshots and every
  field it actually restores;
- derive the heat-capacity, density-response, sound-speed, and adiabatic
  columns used by `compute_convection`;
- explain stable-layer rejection, local mixing-length leakage, overshoot
  averaging, and top-layer suppression;
- distinguish production finite-difference convection from the one-step
  ideal-gas disabled diagnostic;
- identify the exact point where Chapter 12 stops and Chapter 13 begins.

The standard atmosphere scope remains:

- one-dimensional;
- static;
- plane-parallel;
- LTE;
- CPU NumPy float64 outside declared float32 transfer islands;
- opacity sampled on the Chapter 11 30,000-point grid;
- hydrostatic total pressure with turbulent pressure disabled;
- local mixing-length convection only where the exact prescription activates.

It is not:

- three-dimensional radiation hydrodynamics;
- time-dependent convection;
- non-local turbulent transport;
- NLTE or PRD transfer;
- spherical or expanding-atmosphere transfer;
- a GPU atmosphere iteration;
- a proof that mixing-length theory is unique or complete.

## 3. Exact notation, scale, axes, units, and ordering

Let:

- \(L\) be the number of physical depth layers;
- \(N_\nu\) be the complete opacity-grid count, normally 30,000;
- \(i\) be a zero-based opacity-grid index;
- \(\ell\) be a zero-based depth index, ordered surface to interior;
- \(m_\ell\) be column mass in g cm\(^{-2}\), strictly increasing inward;
- \(\nu_i\) be frequency in Hz;
- \(w_i\) be the positive frequency quadrature weight in Hz;
- \(T_\ell\) be temperature in K;
- \(\chi_{i\ell}\) be total mass extinction in cm\(^2\) g\(^{-1}\);
- \(B_{i\ell}\), \(J_{i\ell}\), and \(H_{i\ell}\) be per-frequency intensity
  quantities in the source's cgs \(B_\nu\) scale;
- \(K_i(0)\) be the one-frequency surface second moment;
- \(\omega_{i\ell}\) be the continuum-scattering fraction;
- \(\tau_{i\ell}\) be monochromatic optical depth;
- \(\kappa_{{\rm R},\ell}\) be Rosseland opacity in cm\(^2\) g\(^{-1}\);
- \(\tau_{{\rm R},\ell}\) be Rosseland optical depth;
- \(c=2.99792458\times10^{10}\) cm s\(^{-1}\).

The Chapter 11 wavelength coordinate increases with index, so
`opacity_frequency_hz` decreases with index. The kernel nevertheless traverses
the stored index interval in ascending index order. “Frequency order” in this
chapter always means stored opacity-grid order, not numerically increasing
\(\nu\).

The exact target stored by the source is:

\[
H_\star =
\frac{5.6697\times10^{-5}}{12.5664}T_{\rm eff}^{4}.
\]

It is an integrated Eddington flux. Thus:

\[
F_{\rm rad}=4\pi H_{\rm rad},\qquad
F_{\rm target}=4\pi H_\star\simeq\sigma_{\rm SB}T_{\rm eff}^{4}.
\]

The source's `convective_flux` is added directly to
`integrated_eddington_flux`; it therefore uses the same \(H\)-scale even
though its field name says “flux.” The exact balance written in source units
is:

\[
H_{\rm rad,\ell}+H_{\rm conv,\ell}=H_\star.
\]

If physical flux is shown, multiply both stored terms by \(4\pi\). Never add a
stored `convective_flux` directly to a physical \(F_{\rm rad}\).

### 3.1 Array boundary

| object | exact shape | dtype in production call | ordering |
| --- | ---: | --- | --- |
| `opacity_frequency_hz` | `(Nnu,)` | float64 contiguous | stored opacity-grid order |
| `frequency_weights` | `(Nnu,)` | float64 contiguous | same |
| `planck_all` | `(Nnu,L)` | float64 | frequency major |
| `stimulated_all` | `(Nnu,L)` | float64 | frequency major |
| `continuum_absorption` | `(L,Nnu)` | float64 contiguous | depth major |
| `continuum_scattering` | `(L,Nnu)` | float64 contiguous | depth major |
| `continuum_source` | `(L,Nnu)` | float64 contiguous | depth major |
| gross line opacity | `(L,Nnu)` | float32 contiguous at call | depth major |
| `column_mass`, `h_over_kt`, `temperature` | `(L,)` | float64 contiguous | surface inward |
| transfer optical-depth grid | `(51,)` | float64 contiguous | Chapter 9 order |
| mean/flux operators | `(51,51)` | float32 contiguous | Chapter 9 convention |
| second-moment weights | `(51,)` | float32 contiguous | Chapter 9 convention |
| every depth accumulator | `(L,)` | float64 | surface inward |

The runner allocates full `(Nnu,L)` Planck and stimulated arrays even for a
restricted diagnostic interval, fills only `[start:stop]`, and leaves the
unvisited rows at zero/one. The compiled loop reads only `[start,stop)`.

### 3.2 Planck and stimulated-emission columns

For frequency \(\nu\), with source field `h_over_kt = h/(kT)`,

\[
x_{\ell}=\nu\,\frac{h}{kT_\ell},\qquad
s_\ell=\max(1-e^{-x_\ell},10^{-300}),
\]

\[
B_{\nu,\ell}
=1.47439\times10^{-2}
\left(\frac{\nu}{10^{15}}\right)^3
\frac{e^{-x_\ell}}{s_\ell}.
\]

The private scalar helper
`_planck_source_and_stimulated_emission(...)` exists for one-frequency
validation. Production `accumulate_transfer_state(...)` evaluates the same
expressions as one vectorized NumPy block over `[start,stop)`. The cell must
compare them element for element before using the vectorized result.

## 4. Exact Chapter 11 prerequisite

Chapter 12 accepts one `OpacityState` produced by Chapter 11. It does not
construct or repair it.

The exact record is:

```text
OpacityState
    population_state: AtmospherePopulationState
    continuum_atmosphere: ContinuumAtmosphereState
    opacity_wavelength_grid_nm: np.ndarray
    opacity_frequency_hz: np.ndarray
    frequency_weights: np.ndarray
    active_continuum_indices: np.ndarray
    active_continuum_frequency_hz: np.ndarray
    continuum_absorption: np.ndarray
    continuum_scattering: np.ndarray
    continuum_source: np.ndarray
    continuum_line_selection_threshold: np.ndarray
    continuum_reference_wavelength_nm: np.ndarray
    wavelength_bin_edges: np.ndarray
    line_opacity: LineOpacityState
    rosseland_table: RosselandOpacityTable
    selected_line_catalog: SelectedLineCatalog | None
    transition_line_catalog: LineTransitionCatalog | None
```

Its nested population record is:

```text
AtmospherePopulationState
    setup: RunSetup
    runtime_state: AtmosphereRuntimeState
    fractional_doppler_widths: np.ndarray
    partition_normalized_population_over_mass_density_and_fractional_doppler_width:
        np.ndarray
    temperature_iteration_cache: dict[str, int]
    molecular_state: MolecularEquilibriumState | None
```

Chapter 12 reads, directly or through the exact source calls:

- `setup.atmosphere.column_mass`;
- `setup.atmosphere.temperature`;
- `setup.atmosphere.h_over_kt`;
- `setup.effective_temperature`;
- `setup.surface_gravity_cgs`;
- the carried `setup.surface_radiation_pressure_constant`;
- `setup.convection`;
- `setup.molecules_enabled`;
- runtime gas pressure, density, electron/nuclei densities, packed populations,
  specific energy, and population cache;
- the molecular state when enabled;
- the complete frequency coordinate and weights;
- the three continuum slabs;
- `line_opacity.line_mass_absorption_coefficient`;
- the prior Rosseland table passed through the opacity state.

Before transfer, the chapter-local validation layer requires:

- all shapes and axes in Section 3.1;
- finite frequency, weight, atmosphere, continuum, and line arrays;
- positive frequency and weights;
- strictly increasing finite column mass;
- positive finite temperature and mass density;
- nonnegative continuum scattering and gross line opacity;
- finite continuum source;
- a finite total extinction after stimulation for every visited
  `(layer,frequency)` cell.

This validation is a fail-closed book boundary, not a claim about the pinned
compiled kernel. The pinned kernel itself floors the summed total opacity with
`max(total, 1e-300)` and does not raise a frequency/layer-context error for a
negative total. The Part V/VI brief's statement that the exact source performs
that contextual failure is incorrect.

If the independently reconciled Chapter 11 contract changes any field name,
axis, dtype, or persistence statement above, Chapter 12 does not guess. The
two contracts must be reconciled before implementation.

## 5. Chapter 9 is the sole single-frequency transfer authority

Chapter 12 calls or instruments Chapter 9's exact one-frequency moment path.
It does not reproduce the transfer derivation.

For each visited frequency, the compiled accumulation kernel:

1. copies one depth column of the three continuum slabs;
2. converts the gross float32 line column to float64 and multiplies it by the
   float64 stimulated-emission column exactly once;
3. passes the stimulated line opacity and LTE Planck line source to
   `_transfer_moments_compiled(...)`;
4. receives:
   `optical_depth`, `source`, `monochromatic_eddington_flux`,
   `mean_intensity`, `mean_intensity_minus_source`, `total_opacity`,
   `scattering_fraction`, and `surface_second_moment`;
5. deposits only the integrated contributions defined in this chapter.

The Chapter 9 contract remains authoritative for:

- \(d\tau_\nu=\chi_\nu\,dm\);
- thermal plus scattered source construction;
- fixed 51-point source iteration;
- physical-to-transfer-grid remap;
- deep diffusion continuation;
- \(J_\nu\), \(H_\nu\), and \(K_\nu(0)\);
- float32 operator/source-iteration islands;
- every ordered depth recurrence.

### 5.1 Exact negative-flux guard

After the Chapter 9 moment call, the accumulation kernel scans
`monochromatic_eddington_flux`. If any layer is negative, it floors **every
layer** of:

```text
monochromatic_eddington_flux
mean_intensity
source
```

at `1e-99`. It does not change `mean_intensity_minus_source`. Therefore, after
that guard, the local work arrays need not satisfy
`mean_intensity == source + mean_intensity_minus_source`.

The normal fixture should avoid this branch when teaching the moment identity.
A separate controlled cell must activate it and show which integrated
destinations receive the floored \(J_\nu/H_\nu\) and which receive the
unchanged \(J_\nu-S_\nu\). It must not silently carry Chapter 9's pre-guard
identity into the post-guard accumulation.

## 6. Exact public records and interfaces

### 6.1 `TransferAccumulation`

```text
TransferAccumulation
    opacity_state: OpacityState
    frequency_start_index: int
    frequency_stop_index: int
    rosseland_accumulator: np.ndarray
    radiative_pressure_state: RadiativePressureState
    temperature_correction_state: TemperatureCorrectionState
```

All array fields created by the normal route are CPU float64. The start is
clamped to at least zero. The stop is clamped to `[start,Nnu]`. The interval is
half open.

The exact signature is:

```python
accumulate_transfer_state(
    opacity_state: OpacityState,
    *,
    frequency_start_index: int = 0,
    frequency_stop_index: int | None = None,
    temperature_correction_state: TemperatureCorrectionState | None = None,
) -> TransferAccumulation
```

This function resets every current-pass accumulator on every call. Passing an
existing `temperature_correction_state` preserves its history/table fields but
does **not** append a second interval to its four current-pass arrays. Thus two
successive calls over adjacent intervals are not a public compositional
replacement for one full call.

Restricted intervals are for contribution inspection and chunk tests.
Finalizing one as if it covered all \(N_\nu\) produces an incomplete physical
state while still using the total-grid `frequency_count`. The normal physical
route uses `[0,Nnu)`.

### 6.2 `RadiativePressureState`

```text
RadiativePressureState
    integrated_eddington_flux: float64[L]
    radiation_energy_density: float64[L]
    radiative_acceleration: float64[L]
    integrated_radiation_pressure: float64[L]
    absolute_radiation_pressure: float64[L]
    surface_radiation_pressure_constant: float
```

Exact interfaces:

```python
initialize_radiative_pressure_state(
    layer_count: int,
) -> RadiativePressureState

accumulate_radiative_pressure(
    state: RadiativePressureState,
    *,
    mode: int,
    frequency_weight: float,
    total_opacity: np.ndarray,
    monochromatic_eddington_flux: np.ndarray,
    mean_intensity: np.ndarray,
    surface_second_moment: float,
    target_integrated_eddington_flux: float,
    column_mass: np.ndarray,
) -> None
```

Initialization zeros all six fields. Mode 1 zeros only:

```text
integrated_eddington_flux
radiation_energy_density
radiative_acceleration
surface_radiation_pressure_constant
```

It does not clear the two finalized pressure arrays. The normal accumulator
creates a fresh state before mode 1, so all six happen to be zero there. A
reset demonstration must distinguish initialization behavior from mode-1
behavior.

### 6.3 `TemperatureCorrectionState`

```text
TemperatureCorrectionState
    mean_intensity_minus_source_integral: float64[L]
    absorption_heating_derivative: float64[L]
    diagonal_lambda_accumulator: float64[L]
    integrated_eddington_flux: float64[L]
    previous_temperature_correction: float64[L]
    rosseland_opacity_table: RosselandOpacityTable
```

Exact interfaces used here:

```python
initialize_temperature_correction_state(
    layer_count: int,
) -> TemperatureCorrectionState

ingest_temperature_correction_rosseland_table(
    state: TemperatureCorrectionState,
    *,
    temperature_k: np.ndarray,
    gas_pressure: np.ndarray,
    rosseland_opacity: np.ndarray,
) -> None
```

`apply_temperature_correction(...)` is shown with its exact public signature
in the API inspector:

```python
apply_temperature_correction(
    state: TemperatureCorrectionState,
    *,
    mode: int,
    frequency_weight: float,
    column_mass: np.ndarray,
    total_opacity: np.ndarray,
    monochromatic_eddington_flux: np.ndarray,
    mean_intensity_minus_source: np.ndarray,
    monochromatic_optical_depth: np.ndarray,
    planck_source: np.ndarray,
    frequency_hz: float,
    h_over_kt: np.ndarray,
    temperature_k: np.ndarray,
    stimulated_emission: np.ndarray,
    scattering_fraction: np.ndarray,
    target_integrated_eddington_flux: float,
    effective_temperature: float,
    frequency_count: int,
    rosseland_optical_depth: np.ndarray | None = None,
    rosseland_opacity: np.ndarray | None = None,
    iteration_index: int = 1,
    convection_enabled: bool | int = False,
    convective_flux: np.ndarray | None = None,
    previous_convective_flux: np.ndarray | None = None,
    logarithmic_temperature_pressure_gradient: np.ndarray | None = None,
    adiabatic_gradient: np.ndarray | None = None,
    pressure_scale_height: np.ndarray | None = None,
    total_pressure: np.ndarray | None = None,
    mass_density: np.ndarray | None = None,
    log_density_temperature_derivative_at_constant_total_pressure:
        np.ndarray | None = None,
    heat_capacity: np.ndarray | None = None,
    mixing_length: float = 1.0,
    smooth_start_layer: int = 0,
    smooth_stop_layer: int = 0,
    smooth_left_weight: float = 0.3,
    smooth_center_weight: float = 0.4,
    smooth_right_weight: float = 0.3,
    integrated_radiation_pressure: np.ndarray | None = None,
    turbulent_pressure: np.ndarray | None = None,
    surface_gravity_cgs: float = 1.0e4,
    standard_log_tau_step: float = 0.125,
    standard_log_tau_start: float = -6.875,
) -> TemperatureCorrectionResult | None
```

Chapter 12 discusses only:

- mode 1: reset the four current-pass accumulator arrays;
- the mode-2 contribution formulas duplicated inline in the compiled transfer
  kernel;
- the fact that mode 3 is invoked by the exact combined finalizer.

It does not derive or unpack mode 3.

Mode 1 preserves:

```text
previous_temperature_correction
rosseland_opacity_table and its entries/origins/spans
```

### 6.4 `IterationFinalization`

```text
IterationFinalization
    transfer_accumulation: TransferAccumulation
    rosseland_opacity: float64[L]
    rosseland_optical_depth: float64[L]
    radiative_pressure_state: RadiativePressureState
    temperature_correction_result: TemperatureCorrectionResult
    convection_result: ConvectionResult | None
    convection_finite_difference_samples:
        ConvectionFiniteDifferenceSamples | None
```

The exact signature is:

```python
finalize_transfer_state(
    transfer_accumulation: TransferAccumulation,
    *,
    iteration_index: int = 1,
    temperature_iteration_seed: int | None = None,
    convection_enabled: bool | int = False,
    convective_flux: np.ndarray | None = None,
    previous_convective_flux: np.ndarray | None = None,
    logarithmic_temperature_pressure_gradient: np.ndarray | None = None,
    adiabatic_gradient: np.ndarray | None = None,
    pressure_scale_height: np.ndarray | None = None,
    total_pressure: np.ndarray | None = None,
    log_density_temperature_derivative_at_constant_total_pressure:
        np.ndarray | None = None,
    heat_capacity: np.ndarray | None = None,
    mixing_length: float = 1.0,
    integrated_radiation_pressure: np.ndarray | None = None,
    turbulent_pressure: np.ndarray | None = None,
    molecular_convection_thermal_tracks_perturbation: bool = False,
) -> IterationFinalization
```

The returned radiative-pressure state is the same mutable object held by the
transfer accumulation. Rosseland mode 3 mutates the float64 Rosseland
accumulator in place into the finalized opacity, so
`transfer_accumulation.rosseland_accumulator` aliases the finalized
`rosseland_opacity` on the normal path.

Finalization is single-use. Calling it twice would reapply conversion/scaling
to already finalized mutable state and would rerun correction-state mutations.
No cell may imply idempotence.

### 6.5 Low-level accumulation interfaces

Freeze these exact readable/reference interfaces:

```python
_planck_source_and_stimulated_emission(
    *,
    frequency_hz: float,
    h_over_kt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]

rosseland_mean_step(
    rosseland_accumulator: np.ndarray,
    *,
    mode: int,
    frequency_weight: float,
    planck_source: np.ndarray,
    frequency_hz: float,
    h_over_kt: np.ndarray,
    temperature_k: np.ndarray,
    stimulated_emission: np.ndarray,
    total_opacity: np.ndarray,
    frequency_count: int,
    column_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]

transfer_chunk_count() -> int
```

The exact positional order of `accumulate_transfer_range_parallel` is:

```text
chunk_count,
range_start,
range_stop,
frequency_hz,
frequency_weights,
planck_all,
stimulated_all,
continuum_absorption_slab,
continuum_scattering_slab,
continuum_source_slab,
line_mass_absorption_coefficient_slab,
column_mass,
h_over_kt,
temperature,
transfer_grid,
mean_intensity_operator,
eddington_flux_operator,
second_moment_weights,
target_integrated_eddington_flux,
effective_temperature,
frequency_count,
rosseland_accumulator,
radiation_energy_density,
integrated_eddington_flux,
radiative_acceleration,
surface_radiation_pressure_constant,
temperature_correction_heating_derivative,
temperature_correction_mean_intensity_minus_source_integral,
temperature_correction_integrated_eddington_flux,
temperature_correction_diagonal_lambda
```

`accumulate_transfer_range_compiled` has the same order after removing the
leading `chunk_count`. These low-level functions mutate the nine supplied
outputs and return `None`.

## 7. Reset, accumulate, finalize, and persistence

Use one explicit state table:

| field | new run | start of each pass | per frequency | finalization | next pass |
| --- | --- | --- | --- | --- | --- |
| Rosseland accumulator | zero | zero | add harmonic numerator | overwritten by \(\kappa_R\) | new accumulator |
| integrated \(J\) | zero | zero | add | multiply by \(4\pi/c\) in place | new state |
| integrated \(H\) | zero | zero | add | retained on \(H\)-scale | new state |
| raw \(\int\chi H\,d\nu\) | zero | zero | add | convert/cap to \(g_{\rm rad}\) | new state |
| surface \(\int K(0)\,d\nu\) | zero | zero | add | convert/cap to pressure | carried scalar through setup |
| four correction accumulators | zero | zero | add | consumed by mode 3 | reset next pass |
| previous temperature correction | zero | preserve | untouched | updated by mode 3 | carry |
| Rosseland lookup table | empty | preserve | untouched | append current column | carry and feed next opacity pass |
| selected/detailed catalogs | Chapter 11 | preserve identity | untouched | untouched | Chapter 11 reuses |

The normal runner initializes one `TemperatureCorrectionState` before its
iteration loop and passes the same object into every
`accumulate_transfer_state(...)`. After each finalization it passes that
state's Rosseland table to the next Chapter 11 opacity construction and carries
the finalized surface radiation-pressure constant through the next
`RunSetup`.

On iteration one, the opacity state and correction state can begin with
separate empty table objects. After finalization, the correction state's table
becomes the carried table used by subsequent opacity states.

## 8. One-frequency contribution ledger

The production compiled range kernel duplicates the mode-2 arithmetic inline.
It does **not** call the Python `rosseland_mean_step(mode=2)`,
`accumulate_radiative_pressure(mode=2)`, or
`apply_temperature_correction(mode=2)` helpers per frequency. The chapter uses
those helpers only as readable scalar references and gates them against the
compiled one-frequency contribution.

For one frequency \(i\), define:

\[
B'_{i\ell}
=\frac{B_{i\ell}\nu_i(h/kT_\ell)}
{T_\ell s_{i\ell}}.
\]

For a true `frequency_count == 1` test, the exact Rosseland branch substitutes:

\[
B'_{i\ell}=4\frac{5.6697\times10^{-5}}{3.14159}T_\ell^3.
\]

The eight depth deposits are:

| destination | one-frequency deposit |
| --- | --- |
| `rosseland_accumulator` | \(w_i B'_{i\ell}/\chi_{i\ell}\) |
| `radiation_energy_density` before finalization | \(w_iJ_{i\ell}\) |
| `integrated_eddington_flux` | \(w_iH_{i\ell}\) |
| `radiative_acceleration` before finalization | \(w_i\chi_{i\ell}H_{i\ell}\) |
| `absorption_heating_derivative` | \(w_iH_{i\ell}\chi_{i\ell}^{-1}d\chi_{i\ell}/dm\) |
| `mean_intensity_minus_source_integral` | \(w_i\chi_{i\ell}(J_{i\ell}-S_{i\ell})\) |
| correction `integrated_eddington_flux` | \(w_iH_{i\ell}\) |
| `diagonal_lambda_accumulator` | exact term in Section 8.1 |

The surface deposit is:

\[
K_{\rm surf}^{\rm accum}\mathrel{+}=w_iK_i(0).
\]

The two integrated-Eddington-flux arrays receive the same expression in the
same frequency loop and must agree under the fixed policy.

The field name `absorption_heating_derivative` is historical. Its exact
compiled expression differentiates the **total** opacity returned by transfer,
not a separately supplied absorption-only opacity.

### 8.1 Exact diagonal-lambda contribution

Let:

\[
\Delta\tau_\ell =
\begin{cases}
\max(\tau_{i,\ell+1}-\tau_{i,\ell},10^{-10}),
&\ell<L-1,\\
10^{-10},&\ell=L-1.
\end{cases}
\]

The source constructs a carried pair of interval terms. For
\(\Delta\tau\le0.01\):

\[
a(\Delta\tau)=
\frac{(0.922784335098467-\ln\Delta\tau)\Delta\tau}{4}
+\frac{\Delta\tau^2}{12}
-\frac{\Delta\tau^3}{96}
+\frac{\Delta\tau^4}{720}.
\]

For larger steps:

\[
a(\Delta\tau)
=\frac12
\frac{\Delta\tau+E_3(\Delta\tau)-1/2}{\Delta\tau}.
\]

The exact branch sets \(E_3=0\) for \(\Delta\tau\ge10\). It also forces
\(E_3=0\) when `effective_temperature <= 4250.0` and
the literal nested predicate is `0.005 < depth_step < 0.02`. Because that
predicate is evaluated only in the `depth_step > 0.01` branch, its effective
range is `0.01 < depth_step < 0.02`. Preserve both the nesting and the literal
source conditions in parity tests.

At layer \(\ell\), the diagonal approximation is the previous interval term
plus the newly computed next term. The deposit is:

\[
w_i\chi_{i\ell}
\frac{\Lambda_{i\ell}^{\rm diag}-1}
{1-\omega_{i\ell}\Lambda_{i\ell}^{\rm diag}}
(1-\omega_{i\ell})B'_{i\ell},
\]

with the exact `1e-300` denominator floors. For
`frequency_count == 1`, the lambda branch substitutes:

\[
B'_{i\ell}=\frac{16H_\star}{T_\ell}.
\]

Chapter 12 identifies this response accumulator. Chapter 13 explains how mode
3 combines it with the other correction terms.

## 9. Numba boundary, private memory, and reduction order

### 9.1 Exact chunk policy

`transfer_chunk_count()` returns:

```text
max(1, numba.get_num_threads())
```

with a defensive CPU-count fallback. The runner then chooses:

```text
chunk_count = min(transfer_chunk_count(), max(1, stop - start))
```

For span \(S=stop-start\), bounds are:

\[
b_c=start+\left\lfloor\frac{Sc}{C}\right\rfloor,
\qquad c=0,\ldots,C.
\]

Chunk \(c\) owns `[b_c,b_(c+1))`. The direct low-level function permits more
chunks than frequencies and skips empty chunks, but the runner caps them.

### 9.2 Exact private families

The parallel function allocates:

```text
rosseland_accumulator_by_chunk                 float64[C,L]
radiation_energy_density_by_chunk              float64[C,L]
integrated_eddington_flux_by_chunk              float64[C,L]
radiative_acceleration_by_chunk                 float64[C,L]
heating_derivative_by_chunk                     float64[C,L]
mean_intensity_minus_source_by_chunk            float64[C,L]
temperature_correction_integrated_eddington_flux_by_chunk
                                                 float64[C,L]
diagonal_lambda_by_chunk                        float64[C,L]
surface_radiation_pressure_by_chunk             float64[C,1]
```

This is eight private depth-array families plus one private scalar family.
Private accumulator storage alone is:

\[
8C(8L)+8C = 64CL+8C\ {\rm bytes},
\]

excluding moment work arrays and the full Planck/stimulation blocks.

### 9.3 Legal `prange`

The only `prange` in transfer accumulation is:

```text
prange(chunk_count)
```

Within a chunk:

- frequency indices are serial and ordered;
- optical-depth integration is serial in depth;
- source sweeps are serial in their exact reverse-grid order;
- deep diffusion derivatives/updates are serial;
- lambda interval recurrence is serial in depth.

After `prange`, the source performs an ordinary:

```text
for c in range(chunk_count):
    for layer in range(layer_count):
        shared += private[c, layer]
    surface += private[c, 0]
```

The reduction order is therefore ascending chunk index, then ascending layer
inside each chunk deposit.

The private buffers and final accumulator additions are float64. The source
module's prose calls the regrouping a “float32 reduction,” but that description
is inaccurate. Float32 operator/source work affects each frequency's moment;
the cross-frequency private and final sums are float64. Chunk-count
sensitivity comes from regrouping float64 additions of those contributions.

### 9.4 Other Chapter 12 parallel boundaries

The four perturbation states themselves are evaluated sequentially:

```text
T+ -> T- -> restore T and P+ -> P- -> finally restore
```

In the atomic EOS branch, each call to `iterate_electron_density(...)` may use
an internal Numba `prange(layer_count)` because layers are independent for
that closure.

The molecular equilibrium solve propagates a warm-start equation-density
vector from one layer to the next and remains depth ordered.

`compute_convection(...)` in the pinned source is a Python/NumPy routine with a
serial layer loop and a per-layer iterative leakage loop. The brief's statement
that this layer algebra “can compile” is an optimization possibility, not the
implemented boundary, and must not be written as current behavior.

### 9.5 Repeatability claims

Require:

- repeated runs with one fixed chunk/thread policy are bit-identical for the
  controlled fixture;
- serial and one-chunk compiled accumulation are bit-identical where the same
  route/order is used;
- one and many chunks agree within a measured CPU-float64 grouping envelope;
- a deliberately reversed private-buffer reduction may differ in low bits and
  is never canonical;
- no cross-thread or cross-chunk-count bit-identity claim.

`NUMBA_NUM_THREADS` must be set before Numba runtime initialization in
subprocess-based thread-policy tests.

## 10. Rosseland opacity and optical depth

The Rosseland relation is:

\[
\frac{1}{\kappa_{{\rm R},\ell}}
=
\frac{
\int \chi_{\nu,\ell}^{-1}
(\partial B_{\nu,\ell}/\partial T)\,d\nu
}{
\int(\partial B_{\nu,\ell}/\partial T)\,d\nu
}.
\]

The accumulated denominator in source naming is actually the quadrature
approximation to the numerator of \(1/\kappa_R\):

\[
R_\ell=\sum_i
w_i\frac{B'_{i\ell}}{\chi_{i\ell}}.
\]

Mode 3 mutates it into:

\[
\kappa_{{\rm R},\ell}
=
\frac{
4(5.6697\times10^{-5}/3.14159)T_\ell^3
}{
\max(R_\ell,10^{-300})
}.
\]

Rosseland optical depth is then the exact Chapter 9 parabolic depth integral:

\[
\tau_{{\rm R},0}=\kappa_{{\rm R},0}m_0,\qquad
\tau_{\rm R}(m)=\tau_{{\rm R},0}
+\int_{m_0}^{m}\kappa_{\rm R}(m')\,dm'.
\]

The chapter must:

- compare the compiled contributions with `rosseland_mean_step(mode=2)`;
- compare mode 3 with a direct quadrature on a controlled smooth fixture;
- show that constant opacity returns that opacity within the exact quadrature
  and constant conventions;
- require positive finite \(\kappa_R\);
- require nondecreasing \(\tau_R\) for valid positive opacity/column mass;
- keep the special `frequency_count == 1` branch labelled as a test convention,
  not a physical single-point quadrature.

## 11. Radiation energy, acceleration, and pressure

Before mode 3:

\[
J_\ell^{\rm int}=\sum_iw_iJ_{i\ell},\qquad
H_\ell^{\rm int}=\sum_iw_iH_{i\ell},
\]

\[
G_\ell^{\rm raw}=\sum_iw_i\chi_{i\ell}H_{i\ell},\qquad
K_0^{\rm int}=\sum_iw_iK_i(0).
\]

Mode 3 applies:

\[
u_{{\rm rad},\ell}=\frac{4\pi}{c}J_\ell^{\rm int},
\qquad
g_{{\rm rad},\ell}^{\rm raw}
=\frac{4\pi}{c}G_\ell^{\rm raw},
\]

using the exact source constant `12.5664 / 2.99792458e10`.

The source then forms:

\[
r_\ell=\frac{H_\ell^{\rm int}}{H_\star}.
\]

Where \(r_\ell>1\), it replaces:

\[
g_{{\rm rad},\ell}
=g_{{\rm rad},\ell}^{\rm raw}/r_\ell.
\]

Thus finalized `radiative_acceleration` is not always the unmodified physical
\((1/c)\int\chi_\nu F_\nu\,d\nu\). It is the exact source's capped value.

The surface constant first becomes:

\[
P_{{\rm rad},0}^{\rm raw}=\frac{4\pi}{c}K_0^{\rm int}.
\]

If \(\max_\ell r_\ell>1\), it is divided by that maximum ratio.

The finalized pressure columns are:

\[
P_{{\rm rad},0}^{\rm int}=g_{{\rm rad},0}m_0,
\]

\[
P_{{\rm rad},\ell}^{\rm int}
=P_{{\rm rad},0}^{\rm int}
+\int_{m_0}^{m_\ell}g_{\rm rad}(m')\,dm',
\]

\[
P_{{\rm rad},\ell}^{\rm abs}
=P_{{\rm rad},\ell}^{\rm int}+P_{{\rm rad},0}.
\]

Checks:

- independently gate \(J^{\rm int}\), \(H^{\rm int}\), raw \(G\), and surface
  \(K\) before finalization;
- gate the conversion constant;
- use a sub-target fixture to test the uncapped identity;
- use a super-target controlled fixture to isolate the layerwise and surface
  caps;
- differentiate/invert the exact parabolic pressure integral only within a
  declared discretization tolerance;
- compare against the **capped** finalized acceleration, not raw \(G\);
- require pressure units and monotone behavior only when acceleration and
  column mass make that consequence valid.

The finalized surface constant is carried to the next iteration's setup. The
current convection call, however, constructs its hydrostatic total pressure
with the **incoming carried** `setup.surface_radiation_pressure_constant`, not
the just-finalized current surface constant.

## 12. Persistent Rosseland lookup

`RosselandOpacityTable` is:

```text
normalized_log_temperature: float64[capacity]
normalized_log_pressure: float64[capacity]
log10_rosseland_opacity: float64[capacity]
entry_count: int
log_temperature_origin: float
log_pressure_origin: float
log_temperature_span: float
log_pressure_span: float
```

Normal capacity is:

```text
max(1, layer_count * 60)
```

At the first ingest, the table fixes its origins and spans from the first and
last current depth points. Every ingest then appends each layer in depth order
until capacity is reached. A full table silently stops accepting later rows.

`finalize_transfer_state(...)` ingests the current:

```text
(temperature, runtime_state.gas_pressure, rosseland_opacity)
```

before convection. Convection therefore evaluates a lookup containing the
current column as well as any carried prior columns.

The chapter must show:

- mode-1 correction reset does not change table identity, entries, origins, or
  spans;
- one finalization increases `entry_count` by \(L\) while capacity remains;
- the next Chapter 11 opacity state receives the same carried table object;
- the empty table evaluates to `1.0` in the pinned kernel;
- lookup behavior at incomplete quadrants and full capacity is pinned by
  tests, not inferred as a general smooth interpolation guarantee.

The nearest-quadrant interpolation mechanics may be inspected briefly because
convection consumes them. Continuum use of the same table remains Chapter 11
ownership.

## 13. Four EOS perturbation samples

### 13.1 Exact record and interface

```text
ConvectionFiniteDifferenceSamples
    specific_internal_energy_plus_temperature: float64[L]
    specific_internal_energy_minus_temperature: float64[L]
    specific_internal_energy_plus_pressure: float64[L]
    specific_internal_energy_minus_pressure: float64[L]
    density_plus_temperature: float64[L]
    density_minus_temperature: float64[L]
    density_plus_pressure: float64[L]
    density_minus_pressure: float64[L]
```

```python
compute_convection_finite_difference_samples(
    *,
    atmosphere: ModelAtmosphere,
    runtime_state: AtmosphereRuntimeState,
    absolute_radiation_pressure: np.ndarray,
    rosseland_optical_depth: np.ndarray,
    temperature_iteration_seed: int,
    temperature_iteration_cache: dict[str, int],
    molecules_enabled: bool = False,
    molecular_state: MolecularEquilibriumState | None = None,
    molecular_thermal_energy_tracks_perturbation: bool = False,
) -> ConvectionFiniteDifferenceSamples
```

`molecules_enabled=True` without a molecular state raises `ValueError`.

### 13.2 Exact ordered mutations

Define:

\[
D_\ell=1-e^{-\tau_{{\rm R},\ell}}.
\]

The exact order is:

1. snapshot the central fields in Section 13.3;
2. if molecular state exists, record its arrays/mode; when molecules are
   enabled, save the previous-equation-density seed and enable molecular
   specific-energy mode;
3. set \(T=1.001T_0\);
4. recompute pressure-iteration closure with `seed + 1`;
5. save \(\rho_{T+}\) and
   \[
   e_{T+}=e_{\rm runtime}
   +\frac{3P_{\rm rad}^{\rm abs}}{\rho_{T+}}
   [1+D(1.001^4-1)];
   \]
6. set \(T=0.999T_0\), recompute with `seed + 2`, and save the analogous
   \(e_{T-},\rho_{T-}\);
7. restore \(T=T_0\), set \(P_{\rm gas}=1.001P_0\), recompute with `seed + 3`,
   and save:
   \[
   e_{P+}=e_{\rm runtime}
   +3P_{\rm rad}^{\rm abs}/\rho_{P+};
   \]
8. set \(P_{\rm gas}=0.999P_0\), recompute with `seed + 4`, and save
   \(e_{P-},\rho_{P-}\);
9. execute the exact `finally` restoration;
10. return copied sample arrays.

When molecular thermal tracking is true, each recomputation first copies the
perturbed `atmosphere.thermal_energy_erg` into
`molecular_state.thermal_energy_erg`. When false, the molecular thermal-energy
column remains its carried reference during the perturbations.

The four closure calls are ordered and use distinct cache keys. They must not
be placed in `prange` or launched concurrently against one mutable state.

### 13.3 Exact snapshot and restore ledger

The source snapshots and restores:

```text
atmosphere.temperature
runtime_state.gas_pressure
runtime_state.electron_density
runtime_state.total_nuclei_number_density
runtime_state.mass_density
runtime_state.ion_stage_populations_by_packed_slot
runtime_state.partition_normalized_populations_by_packed_slot
temperature_iteration_cache contents

when molecular_state exists:
    molecular_populations
    partition_normalized_molecular_populations
    molecular_equation_densities
    previous_molecular_equation_densities
    thermal_energy_erg
    specific_internal_energy_mode_enabled
```

It also snapshots `runtime_state.specific_internal_energy`, but restoration is
conditional:

- if any original entry is nonzero, it restores the copied array;
- if all original entries are zero, it computes and installs atomic specific
  internal energy at the restored central state.

The source does **not** snapshot or restore
`runtime_state.charge_square_density`, although atomic electron-density
closure mutates it.

Consequently the exact routine does not, in general, return a byte-identical
central runtime state:

- `charge_square_density` can retain the final pressure-perturbation value;
- an initially all-zero `specific_internal_energy` becomes a computed atomic
  central energy column.

The brief's claims “restore every central field” and “return without mutating
the central state” are false for the pinned implementation. The notebook must
show a before/after field ledger and mark these two deltas. It may not repair
them invisibly in a teaching wrapper while claiming exact-source behavior.

Forced-exception tests must verify restoration of every field covered by the
actual `finally`, and separately report the uncovered
`charge_square_density`. A future upstream repair requires a new pinned source
hash and contract reconciliation.

### 13.4 Atomic and molecular energy truth

The molecular branch enables its specific-internal-energy mode before each
perturbed molecular solve. That solve updates
`runtime_state.specific_internal_energy` using the molecular energy kernel, so
the saved molecular energy samples follow the perturbed closure.

The atomic-only branch calls `populate_species(code=0.0, population_mode=1)`
for each perturbation. That path recomputes electron density, nuclei density,
mass density, charge-square density, and ion-stage populations, but it does
**not** call `compute_atomic_specific_internal_energy(...)` at each perturbed
state. With the normal atomic runtime's initially zero energy column, the four
saved energy samples contain the added radiation term but no perturbed atomic
material-energy term. The atomic energy helper is called only during final
restoration of an all-zero central column.

Therefore:

- density finite differences are full atomic pressure-iteration samples;
- molecular energy samples follow the implemented molecular energy mode;
- pinned atomic-only energy derivatives are **not** full atomic-EOS energy
  derivatives;
- no text, caption, or table may label the current atomic-only heat capacity
  as full-EOS atomic heat capacity;
- exact parity with the pinned atomic branch and scientific completeness are
  separate gates.

This is a source limitation, not a Chapter 13 issue.

## 14. Thermodynamic derivatives

Let \(\epsilon=0.001\). From the eight samples, the exact source forms:

\[
e_T=\frac{e_{T+}-e_{T-}}{2\epsilon T}
=500\frac{e_{T+}-e_{T-}}{T},
\]

\[
\rho_T=500\frac{\rho_{T+}-\rho_{T-}}{T},
\quad
e_P=500\frac{e_{P+}-e_{P-}}{P_{\rm gas}},
\quad
\rho_P=500\frac{\rho_{P+}-\rho_{P-}}{P_{\rm gas}}.
\]

The current radiation-pressure response is:

\[
P_T^{\rm tot}
=\frac{4P_{\rm rad}^{\rm abs}}{T}D,\qquad
P_P^{\rm tot}=1.
\]

With signed floors where the source uses `_signed_floor`, define:

\[
c_V=e_T-e_P\frac{\rho_T}{\rho_P},
\]

\[
c_P^{\rm src}
=e_T-e_PP_T^{\rm tot}
-\frac{P_{\rm tot}}{\rho^2}
(\rho_T-\rho_PP_T^{\rm tot}),
\]

\[
\chi_{T,P_{\rm tot}}
=\frac{T}{\rho}(\rho_T-\rho_PP_T^{\rm tot}),
\]

\[
c_s=
\sqrt{
\max\left(
\frac{c_P^{\rm src}}{c_V}\frac{1}{\rho_P},
0
\right)
},
\]

\[
\nabla_{\rm ad}
=-\frac{P_{\rm tot}}{\rho T}
\frac{\chi_{T,P_{\rm tot}}}{c_P^{\rm src}}.
\]

The returned source field names are:

```text
heat_capacity                              -> c_P^src
log_density_temperature_derivative_at_constant_total_pressure
                                           -> chi_T,Ptot
sound_speed                                -> c_s
adiabatic_gradient                         -> nabla_ad
```

The actual atmosphere gradient and pressure scale height are:

\[
\nabla=
\frac{P_{\rm tot}}{Tg}\frac{dT}{dm},
\qquad
H_P=\frac{P_{\rm tot}}{\rho g}.
\]

`differentiate_on_depth_grid(...)` is the exact Chapter 9/source numerical
derivative and is invoked rather than replaced.

If all eight finite-difference arrays are supplied,
`compute_convection(...)` uses them. If even one is missing, the pinned source
silently uses its ideal-gas fallback for **all** four primitive derivatives:

\[
e_T=\tfrac32 P/(\rho T),\quad
\rho_T=-\rho/T,\quad
e_P=0,\quad
\rho_P=\rho/P.
\]

The complete production path always supplies all eight arrays when automatic
convection is enabled. The chapter wrapper must validate all-or-none before
calling the exact function so an accidental partial set fails closed. The
ideal-gas route remains visible only as a controlled comparison and as the
basis of disabled-convection diagnostics.

## 15. Exact convection implementation

### 15.1 Records and interfaces

```text
ConvectionResult
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

DisabledConvectionDiagnostics
    convective_flux: float64[L]
    convective_velocity: float64[L]
```

Exact interfaces:

```python
integrate_geometric_depth_below_surface_km(
    *,
    column_mass: np.ndarray,
    mass_density: np.ndarray,
) -> np.ndarray
```

```python
compute_convection(
    *,
    rosseland_table: RosselandOpacityTable,
    column_mass: np.ndarray,
    rosseland_optical_depth: np.ndarray,
    temperature_k: np.ndarray,
    gas_pressure: np.ndarray,
    mass_density: np.ndarray,
    rosseland_opacity: np.ndarray,
    microturbulence: np.ndarray,
    absolute_radiation_pressure: np.ndarray,
    total_pressure: np.ndarray,
    surface_gravity_cgs: float,
    target_integrated_eddington_flux: float,
    mixing_length: float = 1.0,
    overshoot_weight: float = 1.0,
    convection_enabled: bool | int = True,
    zero_top_layer_count: int = 36,
    specific_internal_energy_plus_temperature: np.ndarray | None = None,
    specific_internal_energy_minus_temperature: np.ndarray | None = None,
    specific_internal_energy_plus_pressure: np.ndarray | None = None,
    specific_internal_energy_minus_pressure: np.ndarray | None = None,
    density_plus_temperature: np.ndarray | None = None,
    density_minus_temperature: np.ndarray | None = None,
    density_plus_pressure: np.ndarray | None = None,
    density_minus_pressure: np.ndarray | None = None,
) -> ConvectionResult
```

```python
compute_disabled_convection_diagnostics(
    *,
    column_mass: np.ndarray,
    rosseland_optical_depth: np.ndarray,
    temperature_k: np.ndarray,
    gas_pressure: np.ndarray,
    mass_density: np.ndarray,
    rosseland_opacity: np.ndarray,
    absolute_radiation_pressure: np.ndarray,
    total_pressure: np.ndarray,
    surface_gravity_cgs: float,
    target_integrated_eddington_flux: float,
    mixing_length: float,
    rosseland_table: RosselandOpacityTable,
    overshoot_weight: float = 1.0,
    zero_top_layer_count: int = 36,
) -> DisabledConvectionDiagnostics
```

`microturbulence` is converted to float64 and then unused by the pinned
convection calculation. Do not claim it contributes turbulent pressure or
mixing-length velocity.

### 15.2 Total pressure used by automatic convection

When `finalize_transfer_state(...)` computes convection automatically, it
constructs:

\[
P_{{\rm tot},\ell}
=g\,m_\ell
+P_{{\rm rad},0}^{\rm carried}
+P_{{\rm turb},\ell}.
\]

The supported runner supplies \(P_{\rm turb}=0\). The surface radiation term
is the incoming setup value carried from the prior pass or seed metadata, not
the just-finalized current value.

The source setup fixes normal convection settings to:

```text
enabled = config.enable_convection
mixing_length = 1.25
overshoot_weight = 0.0
zero_top_layer_count = 0       # sentinel; finalizer substitutes 36
```

Thus the standard product path has no overshoot contribution and suppresses
the first 36 returned convective-flux layers. Nonzero overshoot is a supported
diagnostic branch, not the normal setup.

### 15.3 Geometric depth

\[
z_0=0,\qquad
z(m)=\int_{m_0}^{m}\frac{10^{-5}}{\rho(m')}\,dm'
\]

returns kilometers below the surface. It uses the exact parabolic depth
integrator and the `1e-300` density floor.

### 15.4 Local instability and coefficients

The local superadiabatic excess is:

\[
\Delta\nabla=\nabla-\nabla_{\rm ad}.
\]

Raw local convection is skipped when:

- `mixing_length == 0.0`;
- `layer_index < 3`;
- \(\Delta\nabla<0\);
- the velocity coefficient below is zero.

For an eligible layer:

\[
A_v=\frac{\alpha}{2}
\sqrt{\max\left(
-\frac12\frac{P_{\rm tot}}{\rho}\chi_{T,P_{\rm tot}},
0\right)},
\]

\[
A_F=
\frac12\rho c_P^{\rm src}T\frac{\alpha}{12.5664}.
\]

Here \(\alpha\) is the exact `mixing_length` value. The division by `12.5664`
is one reason the returned convective flux is on the integrated-\(H\) scale.

The source evaluates the carried Rosseland table at the central
\((T,P_{\rm gas})\), then at \(T\pm\Delta T\). It combines the plus/minus ratios
harmonically and multiplies by the current direct \(\kappa_R\) to obtain the
convective-element opacity.

For exact parity, retain:

```text
enabled local solve:  at most 30 leakage iterations
disabled diagnostic: exactly one local iteration
temperature-step cap: 0.15 * T
step damping:          0.7 * new + 0.3 * previous
early stop:            new step within previous +/- 0.5 K
series stop:           term <= 1e-6
```

The chapter may name intermediate variables `optical_thickness`, `d_factor`,
`ratio_squared`, and `delta` to mirror the source. It must preserve the exact
branch:

- a power-series solution when `ratio_squared < 0.5`;
- the closed expression
  \((1-\sqrt{1-r^2})/r^2\) otherwise;
- source signed floors;
- nonnegative velocity square root and raw flux clamp.

The exact outputs before overshoot are:

\[
v_{\rm conv}=A_v\sqrt{\delta},
\qquad
H_{\rm conv}^{\rm raw}=\max(A_Fv_{\rm conv}\delta,0).
\]

### 15.5 Overshoot and top suppression

The source copies local flux into `raw_convective_flux` before any later
operation.

For positive `overshoot_weight`, define:

\[
w_{\rm over}
=\operatorname{clip}\left(
\max_\ell\frac{H_{{\rm conv},\ell}^{\rm raw}}{H_\star},
0,1\right)
\times{\tt overshoot\_weight}.
\]

The half-window in km is the minimum of:

- \(0.5H_P10^{-5}w_{\rm over}\);
- distance to the deepest geometric point;
- distance to the surface point.

The source integrates raw convective flux over geometric depth, then evaluates
a symmetric integral difference. It computes this overshoot column only from:

```text
max(L // 2 - 1, 0) through L - 2
```

and returns:

\[
H_{\rm conv}=\max(H_{\rm conv}^{\rm raw},H_{\rm conv}^{\rm over}).
\]

Finally, it sets the first `zero_top_layer_count` entries of only the returned
`convective_flux` to zero. It does not zero the stored raw flux, overshoot
column, or velocity.

The standard `zero_top_layer_count=0` setup value is converted by the finalizer
to 36, not interpreted literally as “suppress none.”

### 15.6 Disabled diagnostics

When automatic convection is disabled, `IterationFinalization` contains:

```text
convection_result = None
convection_finite_difference_samples = None
```

After finalization, the runner separately calls
`compute_disabled_convection_diagnostics(...)`. That function invokes
`compute_convection(...)` with:

- all finite-difference arrays absent;
- the ideal-gas fallback;
- `convection_enabled=False`;
- one local leakage iteration.

It can return nonzero flux/velocity diagnostics. “Disabled” does not mean the
diagnostic evaluator returns zero. The resulting two arrays are later passed
to the Chapter 13 remap path; they are not inserted into
`IterationFinalization` as a `ConvectionResult`.

## 16. Exact combined finalization order and Chapter 13 boundary

The radiation/convection prefix of `finalize_transfer_state(...)` is:

1. obtain the Chapter 11 setup, atmosphere, and runtime state;
2. run Rosseland mode 3, mutating the accumulator into
   `rosseland_opacity` and integrating `rosseland_optical_depth`;
3. run radiative-pressure mode 3, converting/capping energy, acceleration,
   surface constant, and pressure;
4. append current \((T,P_{\rm gas},\kappa_R)\) to the persistent correction
   lookup;
5. if `int(convection_enabled) == 1` and `convective_flux is None`, compute all
   eight finite-difference arrays;
6. construct automatic-convection total pressure from \(gm\), the carried
   surface radiation-pressure constant, and supplied/zero turbulent pressure;
7. compute exact `ConvectionResult`;
8. copy its returned fields into the arguments required by correction mode 3;
9. call exact `apply_temperature_correction(..., mode=3, ...)`;
10. require a non-`None` correction result;
11. return exact `IterationFinalization`.

If a caller supplies `convective_flux`, automatic finite differences and
`ConvectionResult` construction are skipped even when convection is enabled.
The exact full runner does not take that override in its normal path.

Chapter 12's state inspector may reveal only the correction record's field
names:

```text
temperature
flux_error_percent
flux_derivative
flux_temperature_derivative
lambda_temperature_derivative
surface_temperature_derivative
temperature_correction
flux_ratio
convective_flux
column_mass
column_mass_correction
```

It must not graph, derive, interpret, clamp, smooth, or apply those fields.
Those are Chapter 13's first causal task.

## 17. Reader-facing movement and visible cell ledger

Use two acts and 20 substantial cells. Cells may contain short helper
inspectors, but no giant copy of the production transfer or correction kernel.

### Act I — Thirty thousand frequencies become physical columns

1. **Chapter 11 handoff validator.** Display the exact state identity, axes,
   dtypes, frequency direction, visited interval, and table hashes.
2. **One Chapter 9 frequency.** Call the existing moment checkpoint and show
   the eight depth outputs plus surface moment without rederiving transfer.
3. **One-frequency fan.** Compute the nine scalar deposits and label units.
4. **Planck vectorization.** Compare the scalar helper with the exact broadcast
   block in stored grid order.
5. **Mode-1 ledger.** Prove exact radiative/correction resets and preservation
   of prior correction/table state.
6. **Inline-versus-helper gate.** Compare one compiled frequency with the
   three readable mode-2 helper contributions.
7. **Chunk map and memory.** Display exact bounds and private-buffer bytes for
   one, two, and current-thread chunk counts.
8. **Parallel accumulation.** Compare one chunk, fixed many chunks, and a
   repeated fixed-many run.
9. **Reduction-order probe.** Reverse only the private-buffer reduction in a
   notebook-local demonstration and report the last-bit delta.
10. **Rosseland finalization.** Show direct harmonic quadrature,
    \(\kappa_R\), and monotone \(\tau_R\).
11. **Radiation finalization.** Show \(u_{\rm rad}\), raw/capped
    \(g_{\rm rad}\), surface constant, and integrated/absolute pressure.
12. **Heating/lambda distinction.** One layer/frequency ledger separates
    integrated \(H\), \(\chi(J-S)\), opacity-gradient, and lambda terms.
13. **Persistent lookup.** Show entry-count growth, object identity, and the
    next-pass handoff.

### Act II — A restored material state decides convection

14. **Perturbation transaction.** Display ordered `T+`, `T-`, `P+`, `P-`
    closure calls and deterministic cache indices.
15. **Restore audit.** Compare every central/runtime/molecular field before,
    after success, and after a forced exception; expose the two exact source
    deltas.
16. **Atomic truth audit.** Show that atomic density changes while the
    perturbed runtime material-energy column is not recomputed; compare with a
    smooth toy/full helper evaluation without relabelling production output.
17. **Molecular perturbation.** On the compact cool fixture, show thermal
    tracking, molecular warm-start restoration, and all eight returned arrays.
18. **Thermodynamic derivatives.** Derive the factor 500, compare a smooth toy
    finite difference at 0.1% and 0.05%, then inspect source fields.
19. **Convection branches.** Show stable rejection, local raw flux, optional
    nonzero overshoot, standard zero overshoot, and 36-layer suppression in
    separate controlled calls.
20. **Exact finalizer/handoff.** Call `finalize_transfer_state`, inspect exact
    `IterationFinalization`, collapse `temperature_correction_result`, and show
    the separate disabled-diagnostics branch.

The 30,000-frequency run should be one cached, named checkpoint reused by
later Act I cells. Do not rerun it merely to draw each plot.

## 18. Original schematic and one-panel plot plan

### 18.1 Original schematic A — frequency fan and ordered funnel

Create an original vector schematic:

```text
one OpacityState
      |
      +--> contiguous frequency chunk 0 --> 8 [L] + 1 scalar
      +--> contiguous frequency chunk 1 --> 8 [L] + 1 scalar
      +--> ...
      +--> contiguous frequency chunk C-1 --> 8 [L] + 1 scalar
                                              |
                              ascending chunk reduction
                                              |
                                  TransferAccumulation
```

Each chunk lane must show:

```text
frequency loop: ordered
depth transfer: ordered
private writes: isolated
```

Use a visually different border for persistent correction history/table,
which bypasses the per-pass zeroing gate.

Alt text must explain that parallelism exists across private contiguous
frequency chunks and stops before the fixed-order reduction.

### 18.2 Original schematic B — four-state EOS transaction

Create an original vector schematic with one central state surrounded by:

```text
1.001 T, P
0.999 T, P
T, 1.001 P
T, 0.999 P
```

Arrows pass sequentially through atomic or molecular closure and deposit
\((e,\rho)\) samples. A `finally` loop returns to the central state. Two amber
annotations identify:

- `charge_square_density` not restored;
- all-zero central specific energy recomputed during restore.

The right side maps eight samples to
\(c_P^{src},\chi_T,c_s,\nabla_{\rm ad}\), then to raw/overshoot/suppressed
convective flux.

### 18.3 Required one-panel plots

Every plot has one panel and one claim:

1. **Cumulative Rosseland denominator versus stored frequency index.** One
   selected layer; annotate that index increases while frequency decreases.
2. **Rosseland opacity versus Rosseland optical depth.** One atmosphere; show
   inward order and monotonic depth.
3. **Integrated radiation pressure and reconstructed acceleration.** Compare
   only the capped finalized acceleration under a declared derivative
   tolerance.
4. **EOS finite-difference convergence.** One smooth toy layer; show error at
   0.1% and 0.05%, not production atomic completeness.
5. **Convective flux variants versus geometric depth.** Raw, optional
   overshoot, and returned top-suppressed values; one cool controlled case.

Optional:

- one single-layer cumulative \(H_\nu\) contribution plot;
- one molecular-versus-ideal-gas adiabatic-gradient plot, if the molecular
  compact fixture is available.

No multi-panel dashboard. Captions state regime, layer, interval/full-grid
status, chunk policy, dtype, and whether values are source parity or a toy
derivation.

Every generated visual records prompt/source description, generation method,
alt text, caption, and SHA-256 in the chapter asset manifest. No external
Payne Zero figure is copied.

## 19. Source, table, fixture, and golden requirements

### 19.1 Read-only development oracle

The external checkout is a development oracle only:

```text
/Users/ysting/payne-zero
commit 9c44001feae40b85146630499e6f8a5fed42e5af
```

Normal chapter execution must not import from or read that checkout.

Exact audited source hashes:

| source | SHA-256 |
| --- | --- |
| `payne_zero_atmosphere/runner.py` | `05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7` |
| `payne_zero_atmosphere/transfer_kernels.py` | `50e759a085e6aefdb7819a3dbe3ef5e83405834f4b07e0a4de2f3c0e7354d3b9` |
| `payne_zero_atmosphere/rosseland_mean.py` | `91071248fd903e05322b7163d37566e9f894daefc1d7ba018d4850d362f1fc86` |
| `payne_zero_atmosphere/radiative_pressure.py` | `c61a256892282d9a0d6cb19714ea5ce6135f1b6f7f573e5761d216d552f321ec` |
| `payne_zero_atmosphere/temperature_correction.py` | `67728389ba857511979d0f82ea59f0bf41ee635b8151ae26673dace02b195d21` |
| `payne_zero_atmosphere/convection.py` | `9099af3ce97123a88cfee554cefb55b2b47a52085e3cb6cda19e6869e0fef9fd` |
| `payne_zero_atmosphere/radiative_transfer.py` | `df8970ca629487537a7c4849278eab5d755b527002d8fc58360c9264a3aa45db` |
| `payne_zero_atmosphere/specific_internal_energy.py` | `de06ba732ce1333d111a52223e39f5b4f80eece8cfc4ff2f30de9739e16d7ec5` |
| `payne_zero_atmosphere/equation_of_state.py` | `719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2` |
| `payne_zero_atmosphere/molecular_equilibrium.py` | `4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86` |
| `payne_zero_atmosphere/runtime_state.py` | `fae240ec00f6f89d7c2a7ef721ce6e6539be234e523291fd6e8a096d731430e8` |
| `payne_zero_atmosphere/continuum_opacity.py` | `1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81` |
| `payne_zero_atmosphere/run_setup.py` | `de7cf08b936585dbcfa2e572c026fafa3f10282a99c27b834b62db0f3f2888c9` |
| `payne_zero_atmosphere/atmosphere_io.py` | `95c4d2cab230f6925e9404639ecb05b25af8c0c85755ac1ca70d760156a8683e` |

The implementation manifest must reject a different source hash until its
numerical behavior and this contract are reconciled.

### 19.2 Reused static tables

No new large scientific source bundle is introduced. Reuse:

| repository table | role | SHA-256 |
| --- | --- | --- |
| `data/static/atmosphere_tables/radiative_transfer_tables.npz` | Chapter 9 transfer grid/operators/moment weights | `d69fcad9e22dd8dd42634e5720df717f0298849d98ee2cd93236009e22391e56` |
| `data/static/atmosphere_tables/ionization_potential_tables.npz` | atomic specific energy/Saha | `82a2e82f2015da02c3d2bce77ca5337aa2b9c4e23d8d6219da07895896ca8a50` |
| `data/static/atmosphere_tables/iron_group_partition_tables.npz` | atomic Saha/energy | `137629dea64eca46f77ea3656c18305ade912a468d7eb27029544c0106cc3296` |
| `data/static/atmosphere_tables/packed_level_metadata.npz` | atomic Saha/energy | `de5f17b6a9eaec1d1b07e96fd02ff014279cd8eaa9f976fefde0e2a153961bc3` |
| `data/static/atmosphere_tables/special_partition_tables.npz` | atomic Saha/energy | `7d737524aacda1cc2281e5b18ff49f240ca34665dbe6c96d4dd0f39db4aedd22` |
| `data/static/atmosphere_tables/molecular_equilibrium_tables.npz` | molecular energy/equilibrium | `1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe` |
| `data/static/source_catalogs/lines/molecular_equilibrium_atmosphere.npz` | compact/normal molecular catalog | `971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644` |

Load transfer tables through the exact Chapter 9 loader. Load EOS/molecular
tables through the Chapters 3–4 staged modules. Do not duplicate table arrays
inside the notebook or reconstruct operators from prose.

### 19.3 Input fixtures

Reuse the independently accepted Chapter 11 builders/fixtures for:

- hot atomic;
- solar atomic;
- low-gravity giant atomic;
- cool molecule-enabled.

For each, preserve:

- seed identity and quantization status;
- opacity-grid identity and complete ordering;
- continuum/line component provenance;
- selected/detailed catalog identity;
- exact `OpacityState` and nested population-state fields.

The normal explanatory fixture may provide a small **contiguous** interval from
each complete state. It must retain original global frequency indices and
weights. A hand-picked reordered list cannot test the production accumulator.

The normal physical pass uses the full 30,000-point state. Do not serialize a
second giant copy merely for Chapter 12 if Chapter 11 can rebuild the accepted
state.

### 19.4 Golden outputs

Offline oracle generation may write compact test-only output archives such as:

```text
data/golden/payne_zero/chapter12/
    hot_atomic_radiation_cpu_float64.npz
    solar_atomic_radiation_cpu_float64.npz
    giant_atomic_radiation_cpu_float64.npz
    cool_molecular_radiation_convection_cpu_float64.npz
```

Each golden manifest records:

- pinned commit and every source hash above that affected generation;
- input fixture/state manifest hash;
- table/catalog hashes;
- full or compact frequency interval and global indices;
- layer/frequency counts;
- NumPy/Numba/Python versions;
- Numba thread/chunk policy;
- cold/warm compilation status;
- every output field name, shape, dtype, and byte hash;
- whether the atomic energy limitation was present.

Reader-facing runtime never opens a golden. Tests compare staged output to
goldens. Fixtures contain inputs; goldens contain oracle outputs; static tables
remain scientific source data.

## 20. Required tests and parity gates

### 20.1 Contract and interface gates

- exact dataclass field names and order for all records in Sections 6, 13, and
  15;
- exact function names, keyword-only boundaries, defaults, and return types;
- `IterationFinalization`, not an invented partial record;
- correction result present but not interpreted;
- no source import from the external checkout;
- exact source/table hashes in the implementation manifest.

### 20.2 Chapter 11/9 handoff gates

- `OpacityState` identity and nested `AtmospherePopulationState` accepted
  without reconstruction;
- exact axis, dtype, contiguity, and stored frequency order;
- line gross opacity converted to contiguous float32 at the transfer call;
- stimulated emission applied exactly once;
- Chapter 9 one-frequency moments match its existing checkpoint;
- post-negative-flux-guard behavior tested separately;
- formal transfer code is invoked, not duplicated.

### 20.3 Reset and persistence gates

- Rosseland mode 1 zeros only its accumulator;
- fresh radiative initialization zeros all six fields;
- radiative mode 1 leaves finalized pressure arrays unchanged;
- correction mode 1 zeros exactly four arrays;
- prior temperature correction remains byte-identical through mode 1;
- table object, entries, count, origins, and spans survive mode 1;
- finalization appends exactly \(L\) rows when capacity permits;
- carried table and surface-constant identities/values reach the next pass.

### 20.4 Accumulator and reduction gates

- scalar Planck/stimulation equals vectorized block;
- all nine one-frequency deposits match the compiled one-frequency path;
- the two integrated-\(H\) destinations agree;
- exact `[start,stop)` semantics, including empty and clamped intervals;
- partial calls reset rather than append;
- exact chunk bounds for awkward spans;
- no private-buffer aliasing across chunks;
- fixed one-chunk repeatability;
- fixed many-chunk repeatability;
- one/many agreement within a predeclared measured envelope;
- reversed-order last-bit demonstration remains noncanonical;
- private-memory formula and observed allocation agree;
- no `prange` in one-frequency depth recurrences or final reduction.

### 20.5 Rosseland and radiation gates

- every Rosseland deposit independently checked;
- special one-frequency branch isolated;
- direct harmonic relation checked on a controlled grid;
- positive finite \(\kappa_R\);
- monotone \(\tau_R\) for valid inputs;
- radiation energy conversion \(4\pi/c\);
- integrated \(H\) remains on \(H\)-scale;
- raw acceleration integral checked before cap;
- layerwise super-target cap checked;
- maximum-ratio surface cap checked;
- integrated and absolute pressure checked;
- pressure derivative/integral identity uses capped acceleration;
- finalized aliasing and single-use mutation checked.

### 20.6 Heating/lambda gates

- opacity derivative is with respect to column mass;
- historical “absorption” field uses total opacity exactly;
- \(\chi(J-S)\) integral independently checked;
- correction integrated \(H\) independently checked;
- small/large depth-step lambda branches;
- `effective_temperature <= 4250.0` special branch;
- last-layer `1e-10` step;
- one-frequency Planck-derivative substitution;
- compiled exponential-integral path versus readable helper within a measured
  float64 tolerance.

### 20.7 Perturbation and restoration gates

- exact mutation order and seed indices;
- atomic and molecular closure call traces;
- eight sample shapes/dtypes/finiteness;
- radiation energy/dilution expressions;
- molecular thermal tracking on and off;
- molecular equation-density warm-start save/restore;
- success-path equality for every actually restored field;
- forced-exception equality for every actually restored field;
- expected `charge_square_density` delta recorded;
- expected all-zero-specific-energy restoration behavior recorded;
- no false byte-identical central-state assertion;
- atomic-only energy non-recomputation sentinel;
- molecular energy recomputation sentinel.

### 20.8 Thermodynamic and convection gates

- factor-500 central differences;
- 0.1%/0.05% convergence on a smooth toy EOS;
- all eight arrays required for production finite-difference parity;
- partial finite-difference input rejected by the chapter boundary;
- ideal-gas fallback isolated;
- \(c_V\), source heat capacity, density response, sound speed, and
  \(\nabla_{\rm ad}\) gated separately;
- geometric depth begins at zero and increases inward for positive density;
- first three layers have no raw local convection;
- stable layers have no positive raw local flux;
- mixing-length-zero branch;
- exact 30-versus-one leakage iteration policy;
- temperature-step cap, damping, and early stop;
- raw flux and velocity;
- optional nonzero overshoot window/integral;
- standard `overshoot_weight == 0.0`;
- raw/overshoot/final arrays remain distinct;
- 36-layer standard suppression affects only returned final flux;
- `microturbulence` non-effect demonstrated;
- automatic total pressure uses incoming surface constant;
- disabled diagnostic may be nonzero and is not a `ConvectionResult`.

### 20.9 Four-regime and full-grid gates

For hot, solar, giant, and cool fixtures, independently gate:

- Rosseland accumulator/opacity/depth;
- integrated \(J\) and \(H\);
- raw/final radiative acceleration;
- integrated/absolute/surface radiation pressure;
- all four correction accumulator arrays;
- lookup ingest;
- all eight finite-difference arrays where automatic convection runs;
- every thermodynamic field;
- raw, overshoot, final convective flux and velocity.

The cool case must run molecules with thermal tracking enabled. At least one
controlled call runs tracking disabled to expose its effect.

Exact atomic-branch parity is reported with the source limitation label. It
cannot satisfy a “full atomic material-energy derivative” scientific gate
until the source behavior changes.

## 21. Failure modes, unsupported branches, and source/brief corrections

### 21.1 Fail closed in the chapter boundary

Raise with field/frequency/layer context for:

- nonfinite or invalid Chapter 11 state;
- shape/axis mismatch;
- nonmonotone column mass;
- nonpositive frequency or weight;
- invalid visited total opacity before entering the compiled floor;
- partial finite-difference sample sets;
- molecule-enabled perturbations without a molecular state;
- nonfinite returned integrated/thermodynamic/convection state;
- an attempt to treat a partial interval as the normal physical finalization;
- a second call to the same mutable finalization object in chapter helpers.

These guards make the reader path safer; they do not rewrite what the pinned
kernel does on invalid direct input.

### 21.2 Exact unsupported physics

- turbulent-pressure runner branch;
- NLTE populations or transfer;
- HLINOP hydrogen wings;
- spherical geometry;
- winds or velocity-field transfer;
- 3D/time-dependent convection;
- non-local hydrodynamic overshoot;
- GPU atmosphere iteration.

Microturbulence in the input atmosphere remains part of opacity construction,
but it is unused by `compute_convection`.

### 21.3 Corrections to the Part V/VI brief

The chapter must incorporate these corrections:

1. The exact compiled transfer kernel does not raise the brief's contextual
   negative-opacity error; it floors total opacity. The book adds an explicit
   preflight guard and labels it as such.
2. Cross-chunk accumulator reductions are float64, not float32.
3. Production mode-2 accumulation is duplicated inline in the compiled range
   kernel; the three Python mode-2 helpers are scalar/reference gates.
4. Radiative mode 1 does not clear finalized pressure arrays.
5. Partial `accumulate_transfer_state` calls reset and cannot be composed by
   repeated public calls.
6. The negative-flux guard changes \(J/H/S\) but not \(J-S\).
7. Finalization mutates/aliases its input state and is not idempotent.
8. The finalized radiative acceleration and surface constant can be flux
   capped.
9. Stored convective flux uses the integrated-\(H\) scale, not physical
   \(F=4\pi H\).
10. Standard setup has `mixing_length=1.25`, `overshoot_weight=0.0`, and a
    zero-top sentinel that becomes 36.
11. Current automatic convection total pressure uses the incoming carried
    surface radiation-pressure constant.
12. `compute_convection` is not currently Numba compiled.
13. The perturbation restore is not complete:
    `charge_square_density` is omitted and all-zero specific energy is
    recomputed.
14. The atomic-only perturbation path does not recompute atomic material
    specific energy at each perturbed state.
15. Disabled convection diagnostics perform a one-step ideal-gas convection
    evaluation and may be nonzero.

### 21.4 Current blockers

No blocker prevents writing the contract or implementing the radiative
accumulation spine.

The following block unqualified scientific/publication claims:

- the pinned atomic-only energy samples are not full atomic-EOS perturbation
  energies;
- the pinned perturbation helper does not restore every mutable runtime field;
- Chapter 11's separately reconciled contract must match the exact prerequisite
  in Section 4 before cross-chapter implementation is accepted;
- tolerance envelopes for full-grid chunk-count changes must be measured on
  the staged implementation rather than guessed here.

These are not reasons to move correction logic into Chapter 12 or to hide the
exact source behavior.

## 22. Redundancy and deferral audit

Chapter 12 must not:

- paste or rederive Chapter 9's transfer solver;
- call \(H_\nu\) physical flux without the \(4\pi\) conversion;
- call the stored convective term physical \(F_{\rm conv}\);
- rebuild Chapter 11 opacity or line catalogs;
- rederive atomic/molecular equilibrium;
- claim the atomic-only energy perturbations are full EOS;
- claim exact central-state restoration;
- describe float64 private sums as float32 reductions;
- insert `prange` into depth transfer, molecular continuation, perturbation
  ordering, or final reduction;
- promise bit identity across chunk counts;
- treat a compact frequency slice as a physically complete atmosphere;
- call finalization twice;
- invent an intermediate finalized-radiation record;
- unpack `TemperatureCorrectionResult`;
- teach temperature damping, smoothing, column correction, remap, convergence,
  or terminal quantization;
- imply standard overshoot is enabled;
- interpret unused microturbulence as turbulent pressure;
- turn disabled diagnostics into production finite-difference convection;
- require the external source checkout in the notebook;
- add an end-of-chapter exercise section.

Chapter 13 owns:

- every mode-3 correction formula;
- correction clamp, damping, and smoothing;
- column-mass correction;
- remap to the standard Rosseland grid;
- next-pass atmosphere construction;
- complete carried/reset state orbit;
- convergence counters and stopping;
- Numba cache/prewarm policy for the full iteration;
- terminal fixed-column quantization and product gating.

## 23. Chapter summary and causal Chapter 13 handoff

End with `## 12.N Chapter summary`. It introduces no new function, field, or
claim.

The summary must state:

1. Chapter 11 supplies one complete depth-major opacity state;
2. Chapter 9 remains the authority for one-frequency moments;
3. Planck/stimulation arrays are frequency major while opacity slabs remain
   depth major;
4. contiguous frequency chunks own eight private float64 depth arrays and one
   private surface scalar;
5. only chunk work is parallel and the final reduction is ascending and
   ordered;
6. Rosseland, radiation-energy, force, pressure, heating, flux, and lambda
   columns have distinct meanings;
7. correction history and the Rosseland lookup persist while current-pass
   accumulators reset;
8. four EOS perturbations are sequential transactions whose exact pinned
   restore has documented limitations;
9. production molecular convection consumes eight finite-difference arrays,
   while the pinned atomic-only energy sampling is incomplete;
10. local raw convection, optional overshoot, returned top suppression, and
    disabled diagnostics are different arrays/branches;
11. the exact combined finalizer returns `IterationFinalization`, including an
    opaque correction result;
12. no corrected atmosphere exists yet.

State the exact output now available:

```text
one Chapter 11 OpacityState
    -> TransferAccumulation
    -> rosseland_opacity
    -> rosseland_optical_depth
    -> finalized RadiativePressureState
    -> persistent RosselandOpacityTable update
    -> ConvectionFiniteDifferenceSamples | None
    -> ConvectionResult | None
    -> exact IterationFinalization
         temperature_correction_result present but unopened
```

Close with:

### Next: turn imbalance into a safe new atmosphere

> The frequency field has collapsed into opacity, radiative support, heating
> response, and a declared convective contribution. Those columns diagnose
> what must change; they do not yet define a safe updated structure.
> [Chapter 13](/reader.html?ch=13) opens the exact correction result, remaps the
> complete state, carries only the permitted history into the next pass, and
> decides when the repeated orbit may stop.
