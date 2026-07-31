# Part I Foundations: Detailed Design Brief for Chapters 1–2

> **Integration note.** `design/global_chapter_contracts.md` governs the final
> reader-facing sequence and supersedes this brief's draft exercise sections
> and any suggestion to reuse official website figures. Useful variations now
> belong in the main causal narrative, and all textbook schematics are original.

Status: chapter-design authority for Part I of the fifteen-chapter course.

Audience: final-year undergraduates and first-year graduate students who know basic algebra, one-dimensional calculus, introductory mechanics, energy, waves, the ideal-gas idea, ordinary Python syntax, arrays, loops, and plots. No astronomy, spectroscopy, radiative transfer, statistical mechanics, NumPy vectorization, Numba, PyTorch, or GPU experience is assumed.

This is a fresh, self-contained textbook sequence. The physics is developed from first principles and the executable calculation grows with it. Payne Zero supplies the target architecture, field meanings, and eventual verification standard; the reader encounters a coherent model, not a history of how the software came to exist.

## 1. Part I purpose and architecture

Part I has two substantial chapters:

1. **From Starlight to a First Grey Atmosphere**
2. **From Equations to Fast, Trustworthy Kernels and Explicit Data**

The first chapter asks what must be calculated and builds the simplest physically interpretable atmosphere. The second asks how to preserve that meaning while accelerating the calculation and handing its results to later physics without ambiguity.

### 1.1 Part I dependency spine

| Chapter | Consumes | Produces for the rest of the book |
| --- | --- | --- |
| 1 | Basic waves, energy, calculus, ideal gas, Python functions and plots | Forward-model vocabulary; exact radiation constants; stable Planck kernels; opacity, optical-depth, source-function, and depth conventions; a constant-source slab; the standard 80-layer grid; an explicitly approximate grey atmosphere seed |
| 2 | Chapter 1 equations, named arrays, and 80-layer workload | Unit/shape/dtype/device discipline; exact NumPy, compiled CPU, chunk-parallel transfer, and Torch parabolic-integration realizations; parity/timing evidence; exact constant tiers; exact abundance representations; source-catalog checksum verification; schema-v4 NPZ loading and validation |

Part I does **not** produce a physically converged atmosphere or a synthetic spectrum. It produces the trustworthy foundation on which Chapters 3–15 build them.

### 1.2 The exact pedagogical rhythm

Every major section follows the rhythm established by Lecture 1:

1. **Physical question or obstacle.**
2. **Assumptions stated before use.**
3. **New term in plain language.**
4. **Equation motivated by the need.**
5. **Every symbol, unit, sign, and axis defined.**
6. **Concrete example or limiting case.**
7. **One-purpose canonical code cell.**
8. **Visible output: a short table, diagram, or plot.**
9. **Prose that reads the output physically.**
10. **A check and an honest boundary.**

This rhythm operates inside each chapter. Consolidating four planning units into Chapter 1 must not compress them into four dense survey sections. A reader should never meet more than two new abstractions before seeing a numerical or visual consequence.

Code cells should normally be 10–30 lines. Sixty lines is a soft ceiling and eighty lines is a hard ceiling. Long kernels are split into scientifically named stages without changing operation order.

### 1.3 Exact-code authority

The canonical implementation is the pinned Payne Zero source itself:

```text
payne_zero_atmosphere/
payne_zero_synthesis/
source_data_files/
```

The book may copy the pinned source files into this repository so the textbook is self-contained,
but it must preserve the Payne Zero package paths, public names, internal names used in an exact
walkthrough, signatures, array layouts, dtype transitions, constants, and operation order. It must
not place cleaned-up wrappers in a second package or replace exact names with a parallel teaching
API.

A derivation-only scalar expression is permitted when it helps establish the physics. It must be
labeled **derivation-only**, remain chapter-local, and never be presented as a callable Payne Zero
interface or as the function later chapters compose.

Tests and data roles remain physically separate exactly as they are in the pinned implementation:
static files live under `source_data_files/`; generated caches live outside that tree; fixtures and
goldens are verification artifacts, not physical source tables.

### 1.4 Shared conventions established here

- Depth index `0` is the outermost layer.
- Depth index increases inward.
- `column_mass` and optical depth increase inward.
- Wavelength arrays increase with index unless a function explicitly documents otherwise.
- A frequency array computed at those same sample positions decreases as wavelength increases.
- Depth vectors have shape `(D,)`; wavelength vectors have shape `(W,)`.
- Axis order is function-specific and is never normalized by a wrapper:
  `planck_bnu` returns `[depth, wavelength]`, while synthesis
  `integrate_optical_depth` consumes and returns `[wavelength, depth]`.
- Population cubes have shape `(D, 6, 139)` with axes `[depth, ion_stage_slot, species_slot]`.
- Foundational host calculations use NumPy `float64`.
- Internal physical units are CGS; public wavelengths are nm.
- Exact source names are retained even when the unit is not in the identifier; the surrounding
  table or docstring states the unit instead of appending a new suffix.

### 1.5 Hard exact-name and notation contract

These rules override any desire to make an interface look more uniform:

1. Executable prose and code use the exact module, class, function, argument, field, constant, and
   environment-variable names in the pinned Payne Zero source.
2. Mathematical prose uses the paper's notation. Local teaching symbols are allowed only when
   declared as temporary derivation notation; they do not rename a code field.
3. Private functions may be shown when they are the real kernel, but they retain the leading
   underscore and are identified as implementation details rather than public API.
4. A chapter may split an exact function into consecutive display excerpts, but it may not reorder
   operations or replace the function by an idealized implementation and call it Payne Zero.
5. Shape tables are attached to individual functions. There is no global transpose convention.
6. Exact public outputs retain concise field names; units belong in the surrounding contract rather
   than being appended to a new field name.

#### Chapter 1 exact names

| Role | Paper notation | Exact Payne Zero name |
| --- | --- | --- |
| Stellar labels | \(T_{\rm eff},\log g,\xi,[{\rm M}/{\rm H}],[\alpha/{\rm M}]\) | `effective_temperature`, `log_surface_gravity`, `microturbulence_km_s`, `metallicity`, `alpha_enhancement` |
| Independent abundance labels | \([{\rm Fe}/{\rm H}],\{[{\rm X}/{\rm H}]\},[{\rm C}/{\rm M}],[{\rm N}/{\rm M}],[{\rm O}/{\rm M}]\) | `fe_over_h`, `x_over_h`, `c_over_m`, `n_over_m`, `o_over_m` |
| Label-driven API | — | `payne_zero_synthesis.api.initialize_atmosphere_from_labels`, `synthesize_from_labels` |
| Structured-atmosphere API | — | `payne_zero_synthesis.api.build_structured_atmosphere`, `save_structured_atmosphere`, `synthesize` |
| Spectral-window arguments | — | `wavelength_start_nm`, `wavelength_end_nm`; `synthesize` uses `resolution`, while `synthesize_from_labels` uses `r_grid` and accepts `resolution` as its alias |
| Public products | \(F_\lambda,F_{\lambda,{\rm c}},f_{\rm norm}\) | `Spectrum`, `InitializedAtmosphere`, `LabelSpectrum`, `ForwardTimings` |
| Public spectrum fields | \(F_\lambda,F_{\lambda,{\rm c}},f_{\rm norm}\) | `wavelength_nm`, `flux_total`, `flux_continuum`, `normalized_flux`, `seconds` |
| Timing fields | — | `initializer_seconds`, `population_bridge_seconds`, `synthesis_seconds`, `total_seconds` |
| Initialized-atmosphere fields | — | `structured_atmosphere`, `initializer_family`, `labels`, `provenance`, `timings`, `atmosphere_converged`, `atmosphere_closure_required` |
| Additional label-spectrum fields | — | `initializer_family`, `labels`, `provenance`, `timings`, `initialized_atmosphere`, `atmosphere_converged`, `atmosphere_closure_required` |
| Atmosphere state | \(m,T,P_{\rm gas},n_e,\kappa_{\rm R},g_{\rm rad},\xi\) | `ModelAtmosphere.column_mass`, `.temperature`, `.gas_pressure`, `.electron_density`, `.rosseland_opacity`, `.radiative_acceleration`, `.microturbulence` |
| Planck kernel | \(B_\nu(T)\) | `payne_zero_synthesis.radiative_transfer.planck_bnu(wavelength_nm, temperature)` |
| Standard grid | \(\tau_{\rm R}\) | `payne_zero_atmosphere.run_setup.standard_rosseland_optical_depth_grid(layers)` |
| Hydrostatic update | \(P_{\rm gas}+P_{\rm rad}=gm+P_0\) | `payne_zero_atmosphere.hydrostatic.integrate_hydrostatic_pressure`, `update_total_pressure` |

#### Chapter 2 exact names

| Role | Exact Payne Zero name |
| --- | --- |
| NumPy depth integration | `payne_zero_atmosphere.radiative_transfer.parabolic_coefficients`, `integrate_on_depth_grid` |
| Compiled CPU depth integration | `payne_zero_atmosphere.transfer_kernels._parabolic_coefficients_compiled`, `_integrate_on_depth_grid_compiled` |
| CPU parallel transfer | `payne_zero_atmosphere.transfer_kernels.transfer_chunk_count`, `accumulate_transfer_range_compiled`, `accumulate_transfer_range_parallel` |
| Torch batched depth integration | `payne_zero_synthesis.radiative_transfer._parabolic_interval_coefficients`, `integrate_optical_depth` |
| Device/dtype policy | `payne_zero_synthesis.device.device`, `resolve_runtime`, `ACCUMULATION_DTYPE`, `DEFAULT_DTYPE`, `REFERENCE_DTYPE` |
| Standard mixture | `payne_zero_atmosphere.warm_start.compute_metal_log_number_abundances`, `compute_hydrogen_fraction`, `ALPHA_ELEMENT_ATOMIC_NUMBERS` |
| Direct abundance mixture | `payne_zero_atmosphere.direct_abundance.complete_direct_abundance_vector`, `retained_direct_abundance_mixture` |
| Deck-to-linear abundance boundary | `payne_zero_atmosphere.atmosphere_io.linear_elemental_abundances` |
| Runtime/source data paths | `payne_zero_atmosphere.data_files.data_root`, `atmosphere_table_dir`, `atmosphere_table_path`, `atmosphere_emulator_dir`; `payne_zero_synthesis.paths.data_root`, `source_catalog_root`, `source_catalog_path` |
| Source-catalog checksums | `payne_zero_atmosphere.source_catalogs.source_catalog_root`, `load_source_catalog_checksums`, `verify_source_catalog_checksums`, `source_line_paths`, `atmosphere_source_catalog_paths` |
| Schema and archive boundary | `payne_zero_synthesis.atmosphere.ATMOSPHERE_SCHEMA_VERSION`, `REQUIRED_ATMOSPHERE_ARRAYS`, `load_atmosphere_npz`, `load_atmosphere_product_metadata`, `validate_atmosphere_npz` |

#### Paper-notation fence for both chapters

- Depth uses \(m\) and \(\tau_{\rm R}\), with
  \(d\tau_{\rm R}=\kappa_{\rm R}\,dm\).
- Continuum absorption, continuum scattering, and line absorption are
  \(\kappa_{\nu,{\rm c}}\), \(\sigma_{\nu,{\rm c}}\), and
  \(\kappa_{\nu,l}\); the total mass extinction is
  \(\chi_\nu=\kappa_{\nu,{\rm c}}+\sigma_{\nu,{\rm c}}+\sum_l\kappa_{\nu,l}\).
  These symbols are never reassigned to volume coefficients.
- Transfer uses \(\tau_\nu,\epsilon_\nu,\mu,B_\nu,J_\nu,S_\nu,I_\nu\), with
  \(S_\nu=\epsilon_\nu B_\nu+(1-\epsilon_\nu)J_\nu\).
- Atmosphere state uses \(T,\rho,P_{\rm gas},n_e,\kappa_{\rm R},g_{\rm rad},\xi\);
  the supported pressure relation is \(P_{\rm gas}+P_{\rm rad}=gm+P_0\).
- Composition and populations retain
  \(A(X),[X/H],[X/Fe],[X/M],n_{s,r},U_{s,r},n_{\rm mol},n_{\rm pert}\);
  code fields distinguish actual ion-stage populations from \(n/U\).
- Emergent quantities retain \(H_\nu,H_\lambda,F_\nu,F_\lambda,F_{\lambda,{\rm c}}\), and
  \(f_{\rm norm}\), including \(F_\lambda=4\pi H_\nu c/\lambda^2\). A temporary teaching symbol
  is declared locally and cannot replace one of these symbols in a code contract.

---

## 2. Chapter 1 — From Starlight to a First Grey Atmosphere

### 2.1 Opening physical question

**What chain of physics connects the light we observe to the temperature and pressure of the layers that emitted it?**

Open with a hand-sketched stellar spectrum: a broad continuum crossed by narrow and broad absorption lines. Ask why a star described by a few labels produces millions of wavelength-dependent values, and why one effective temperature cannot be the temperature of every emitting layer.

Do not open with imports, a syllabus, a benchmark, or the grey-atmosphere equation.

### 2.2 Chapter contract

| Item | Contract |
| --- | --- |
| Reads | Basic wave relations, exponential functions, one-dimensional integrals and derivatives, force balance, ideal-gas idea |
| Writes | Exact Payne Zero endpoint records; exact/pinned constants; \(B_\nu\) and the \(B_\nu\leftrightarrow B_\lambda\) derivation; optical-depth and constant-source demonstrations; standard 80-layer arrays |
| Principal scalar inputs | \(T_{\rm eff}\), \(\log g\), wavelength range, resolving power, illustrative \(\kappa_R\), \(P_0\) / `pressure_constant` |
| Principal arrays | `wavelength_nm (W,)`; `planck_bnu` output `[depth,wavelength]`; synthesis optical depth `[wavelength,depth]`; atmosphere columns `(80,)` |
| Units | K, nm, Hz, erg, cm, s, sr, g cm\(^{-2}\), cm\(^2\) g\(^{-1}\), dyn cm\(^{-2}\) |
| Dtype/device | Chapter-local derivations use NumPy `float64` on CPU; exact `planck_bnu` uses the supplied Torch dtype/device; backend mechanics are deferred to Chapter 2 |
| Final product | Exact named arrays plus a `ModelAtmosphere` interface preview; initializer products use the real `InitializedAtmosphere` safety fields |
| Honest boundary | No EOS, realistic opacity, line blanketing, convection, iterative closure, or synthetic spectrum |

### 2.3 Pace and internal structure

The chapter should be drafted as four connected acts with short section checkpoints:

| Act | Physical movement | Approximate share |
| --- | --- | --- |
| I. Read the light | Observable → labels → atmosphere versus synthesis → assumptions | 15% |
| II. Give thermal light a scale | Coordinates → intensity/flux → Planck function → limiting behavior | 25% |
| III. Let photons escape | Extinction → optical depth → source function → slab → photosphere | 25% |
| IV. Build the first atmosphere | Flux conservation → moments → grey/Hopf law → Rosseland depth → hydrostatic pressure → 80 layers | 35% |

Each act ends with a “what we can now calculate” box. The boxes replace a long opening roadmap and prevent the consolidated chapter from feeling breathless.

### 2.4 Prerequisite concepts introduced from scratch

#### Stellar and modeling language

- spectrum;
- stellar label;
- effective temperature;
- surface gravity and \(\log g\);
- composition label;
- microturbulence;
- model atmosphere;
- spectral synthesis;
- learned initializer;
- forward model;
- physical closure and convergence.

#### Radiation language

- wavelength, frequency, and photon energy;
- solid angle and steradian;
- specific intensity, flux, and luminosity;
- spectral density;
- blackbody;
- local thermodynamic equilibrium;
- Planck function;
- Rayleigh–Jeans and Wien limits.

#### Transfer language

- absorption, emission, scattering, and extinction;
- cross-section, volume extinction, mass extinction, and mean free path;
- optical depth;
- emissivity and source function;
- column mass;
- direction cosine;
- optically thin and optically thick;
- photosphere.

#### Atmosphere language

- radiative equilibrium and flux conservation;
- angular moments \(J,H,K\);
- closure and Eddington closure;
- grey opacity;
- boundary correction and Hopf function;
- Rosseland mean and Rosseland optical depth;
- hydrostatic equilibrium;
- gas, radiation, and total pressure;
- seed atmosphere.

None of these terms should appear in a heading or caption before its plain-language definition.

### 2.5 Detailed derivation and narrative arc

#### Act I — Read the light

##### Scene 1: The observable

Define a spectrum as radiant energy distributed with wavelength. Introduce:

\[
F_\lambda,
\qquad
F_{\lambda,{\rm c}},
\qquad
f_{\rm norm}(\lambda)
=
\frac{F_\lambda}
{F_{\lambda,{\rm c}}}.
\]

All three output arrays have shape `(W,)`. The public `Spectrum` fields are exactly
`flux_total`, `flux_continuum`, and `normalized_flux`; `wavelength_nm` and `seconds`
complete the record. Total and continuum flux densities use
\(\mathrm{erg\,s^{-1}\,cm^{-2}\,nm^{-1}}\); normalized flux is dimensionless.

Use a three-point hand-built example before introducing a full plotted spectrum.

**Output:** a labeled spectrum and a three-row flux table.

**Check:** recompute normalized flux from the two physical flux arrays.

##### Scene 2: The labels do not contain the spectrum

Define effective temperature by

\[
F_{\rm bol}=\sigma T_{\rm eff}^{4},
\]

and surface gravity by

\[
\log g
=
\log_{10}\!\left[
\frac{g}{\mathrm{cm\,s^{-2}}}
\right].
\]

Explain that \(T_{\rm eff}\) is a bolometric flux label, not the temperature of every depth. Explain composition and microturbulence only to the level needed to type the inputs.

Show the causal chain:

\[
\text{labels}
\rightarrow
\text{temperature/pressure/electron structure}
\rightarrow
\text{atomic and molecular populations}
\rightarrow
\text{opacity and source}
\rightarrow
\text{radiative transfer}
\rightarrow
\text{spectrum}.
\]

Give three coupling examples:

- an abundance can change the electron supply;
- carbon and oxygen can change molecule formation;
- many weak lines can alter atmospheric temperature through blanketing.

**Output:** one compact `synthesize_from_labels` call using the exact keywords
`effective_temperature`, `log_surface_gravity`, `metallicity`, `alpha_enhancement`,
`microturbulence_km_s`, `wavelength_start_nm`, `wavelength_end_nm`, and `r_grid`.
Keep those exact keyword arguments visible; do not replace them with a book-defined label or
request record.

**Check:** use the real public boundary to reject nonfinite labels, reversed wavelength bounds, and
nonpositive `r_grid`; do not invent a parallel validator with renamed arguments.

##### Scene 3: Atmosphere and synthesis are different calculations

Define a model atmosphere as depth-dependent thermodynamic structure and populations. Define synthesis as opacity plus transfer on a wavelength grid.

Show the two exact public workflows:

1. a physically converged structured atmosphere is passed to
   `payne_zero_synthesis.api.synthesize`;
2. labels enter `initialize_atmosphere_from_labels` or `synthesize_from_labels`, returning
   `InitializedAtmosphere` or `LabelSpectrum` with `atmosphere_converged=False` and
   `atmosphere_closure_required=True`.

Show the architecture once:

- atmosphere: NumPy/Numba on multicore CPU;
- initializer: Torch;
- synthesis: Torch on CUDA, MPS, or CPU.

The section does not teach backend syntax. It explains why “use the GPU” is not a universal instruction.

##### Scene 4: State the model assumptions

Introduce before use:

- one dimensional;
- plane parallel;
- static;
- LTE;
- hydrostatic;
- opacity sampled and line blanketed in the final model.

Explicit horizons: NLTE, spherical geometry, three-dimensional flows, winds, and turbulent pressure.

Define LTE carefully: local material populations are described by the local temperature. The radiation field need not be blackbody everywhere.

**Act I checkpoint:** we know the endpoints and assumptions, but no layer yet has a temperature or pressure.

#### Act II — Give thermal light a scale

##### Scene 5: Wavelength, frequency, and photon energy

Derive

\[
c=\lambda\nu,
\qquad
E=h\nu=\frac{hc}{\lambda}.
\]

Calculate photon energy at 400, 500, and 900 nm. Point out that increasing wavelength corresponds to decreasing frequency.

**Code:** one chapter-local arithmetic cell that evaluates the two displayed equations. It exports
no newly named conversion function.

**Check:** wavelength–frequency round trip.

##### Scene 6: Intensity is not flux

Use a ray crossing a small surface patch. Define solid angle \(d\Omega\), direction cosine \(\mu=\cos\theta\), and specific intensity \(I_\nu\).

Then define flux:

\[
F_\nu
=
\int_{4\pi}
I_\nu(\hat{\mathbf n})\,\mu\,d\Omega.
\]

For constant outward intensity over a hemisphere, derive \(F_\nu=\pi I_\nu\). State that this is a special angular field, not a general atmosphere identity.

**Output:** a diagram and one numerical angular quadrature.

**Check:** inward and outward isotropic hemispheres cancel to zero net flux.

##### Scene 7: Assemble the Planck function

Introduce, without assumed statistical mechanics:

1. electromagnetic modes per frequency interval;
2. photon energy \(h\nu\);
3. the thermal occupation factor \(1/[\exp(h\nu/kT)-1]\).

Assemble

\[
B_\nu(T)
=
\frac{2h\nu^3}{c^2}
\frac{1}{\exp(h\nu/kT)-1},
\]

with units

\[
\mathrm{erg\,s^{-1}\,cm^{-2}\,sr^{-1}\,Hz^{-1}}.
\]

Use a boxed “assumptions in this derivation” note. Do not turn the section into a full statistical-mechanics chapter.

The executable kernel is the exact
`payne_zero_synthesis.radiative_transfer.planck_bnu(wavelength_nm, temperature)`.
It accepts wavelength in nm, constructs `frequency_hz`, and returns \(B_\nu\) on
`[depth,wavelength]`. Preserve its exact variable names, the pinned
`PLANCK_PREFACTOR = 1.47439e-2` literal, and algebra; do not recompute that literal from a
different constant tier inside the canonical kernel.

##### Scene 8: Change spectral coordinates with a Jacobian

Start from equal energy in corresponding intervals:

\[
B_\lambda\,|d\lambda|
=
B_\nu\,|d\nu|.
\]

Since \(|d\nu/d\lambda|=c/\lambda^2\),

\[
B_\lambda
=
B_\nu\frac{c}{\lambda^2}
=
\frac{2hc^2}{\lambda^5}
\frac{1}{\exp(hc/\lambda kT)-1}.
\]

Convert “per cm” to “per nm” in a named, tested step.

Explain why \(B_\nu\) and \(B_\lambda\) peak at different coordinates: they are different densities.

##### Scene 9: Read the limits before stabilizing the code

Let \(x=h\nu/kT\).

For \(x\ll1\):

\[
\exp(x)-1\approx x,
\]

giving the Rayleigh–Jeans limit.

For \(x\gg1\):

\[
\exp(x)-1\approx\exp(x),
\]

giving the Wien limit.

Only then compare three evaluations:

- the naive \(1/[\exp(x)-1]\) expression;
- `expm1` as a general small-\(x\) technique;
- Payne Zero's exact `boltzmann_factor = torch.exp(-x)` followed by
  `boltzmann_factor / (1.0 - boltzmann_factor)`.

The comparison explains why the decaying form behaves well at large \(x\), but the canonical
result and all parity checks use the pinned Payne Zero expression rather than substituting
`expm1`.

##### Scene 10: Temperature experiment

Evaluate 3000, 5800, and 10,000 K on one increasing wavelength grid.

**Plots:**

- \(B_\lambda\) versus wavelength;
- a normalized companion plot revealing the peak shift;
- relative error of the Rayleigh–Jeans and Wien limits versus \(x\).

**Checks:**

- positivity and finite output;
- \(B_\lambda=B_\nu c/\lambda^2\) with the correct per-nm factor;
- numerical \(\pi\int B_\lambda\,d\lambda\approx\sigma T^4\), with finite-grid truncation error reported.
- exact equality with `planck_bnu` for the pinned tensor inputs, device, and dtype used by the
  chapter.

**Act II checkpoint:** we can calculate the local thermal emission scale, but not how much escapes.

#### Act III — Let photons escape

##### Scene 11: Count interaction opportunities

Start with particle number density \(n\), microscopic cross-section \(a_\nu\), and distance
\(ds\). Use \(\alpha_\nu\equiv n a_\nu\) only as temporary derivation notation for the volume
extinction coefficient, so \(\sigma_{\nu,{\rm c}}\) remains available for the paper's continuum
scattering opacity:

\[
d\tau_\nu
=
n a_\nu ds
=
\alpha_\nu ds.
\]

Then introduce the exact Payne Zero mass-opacity notation:

\[
\chi_\nu
=
\kappa_{\nu,{\rm c}}
+\sigma_{\nu,{\rm c}}
+\sum_l\kappa_{\nu,l},
\qquad
\alpha_\nu=\rho\chi_\nu,
\qquad
\ell_\nu=\frac{1}{\rho\chi_\nu}.
\]

State units:

- microscopic \(a_\nu\): cm\(^2\);
- temporary volume coefficient \(\alpha_\nu\): cm\(^{-1}\);
- \(\kappa_{\nu,{\rm c}},\sigma_{\nu,{\rm c}},\kappa_{\nu,l},\chi_\nu\):
  cm\(^2\) g\(^{-1}\);
- \(\ell_\nu\): cm;
- \(\tau_\nu\): dimensionless.

Do not use \(\chi_\nu\) for a volume coefficient: in the paper and Payne Zero transfer
contract it is the combined mass extinction used in \(d\tau_\nu=\chi_\nu\,dm\).

##### Scene 12: Pure attenuation

From

\[
\frac{dI_\nu}{ds}
=
-\rho\chi_\nu I_\nu,
\]

derive

\[
I_\nu
=
I_{\nu,0}e^{-\tau_\nu}.
\]

Calculate transmitted fractions at \(\tau=0.01,1,10\).

**Check:** the \(\tau=0\) and \(\tau\to\infty\) limits.

##### Scene 13: Emission and the source function

Write

\[
\frac{dI_\nu}{ds}
=
-\rho\chi_\nu I_\nu+j_\nu,
\qquad
S_\nu
\equiv
\frac{j_\nu}{\rho\chi_\nu}.
\]

Choose geometric height \(z\) increasing outward, optical depth increasing inward, and outward rays with \(\mu>0\):

\[
\mu\frac{dI_\nu}{d\tau_\nu}
=
I_\nu-S_\nu.
\]

For the exact Payne Zero separation of thermal absorption and continuum scattering, define:

\[
\epsilon_\nu
=
\frac{\kappa_{\nu,{\rm c}}+\sum_l\kappa_{\nu,l}}{\chi_\nu},
\qquad
S_\nu
=
\epsilon_\nu B_\nu(T)
+(1-\epsilon_\nu)J_\nu.
\]

In the pure-thermal limit \(\epsilon_\nu=1\), Kirchhoff's law gives \(S_\nu=B_\nu(T)\).
This is the only Part I derivation of that limit. The exact scattering source depends on
\(J_\nu\).

##### Scene 14: Solve a constant-source slab

For constant \(S_\nu\) across a slab of optical thickness \(\Delta\tau_\nu\):

\[
I_{\nu,\rm out}
=
I_{\nu,\rm in}e^{-\Delta\tau_\nu/\mu}
+
S_\nu
\left(1-e^{-\Delta\tau_\nu/\mu}\right).
\]

Read both limits:

- thin: incident light is remembered and local emission is a small correction;
- thick: the incident boundary is forgotten and \(I_{\rm out}\to S\).

The scalar evaluation is explicitly derivation-only. It may use `-np.expm1(-x)` to teach a
general numerical technique, but it remains an unnamed chapter cell rather than a Payne Zero API.

**Output:** transparent-slab plot for several incident/source ratios.

**Checks:** zero thickness, zero source, thick limit, and slanted-ray factor.

##### Scene 15: Replace geometric distance with column mass

Define column mass inward from the surface:

\[
dm=-\rho\,dz,
\qquad
d\tau_\nu=\chi_\nu\,dm.
\]

Show the exact Rosseland-mode call site:
`integrate_on_depth_grid(column_mass, accumulator,
surface_value=accumulator[0] * column_mass[0])`, where `accumulator` has just been converted into
the Rosseland mass opacity. State that the function integrates parabolic intervals. Chapter 2
opens that exact algorithm and its compiled and batched forms. A one-frequency
\(\chi_\nu(m)\) example may be evaluated as unnamed derivation data, but it must not acquire
code-like names that imply a second Payne Zero interface.

For synthesis, name the distinct exact boundary now:
`payne_zero_synthesis.radiative_transfer.integrate_optical_depth(column_mass,
extinction, surface_tau)` consumes `extinction[wavelength,depth]` and
`surface_tau[wavelength]`. Do not transpose this into a book-wide `(D,W)` convention.

**Check:** constant-opacity result \(\tau_\nu-\tau_{\nu,0}=\chi_\nu(m-m_0)\).

##### Scene 16: A wavelength-dependent photosphere

Use two toy opacity curves. High opacity reaches \(\tau_\nu\sim1\) higher; low opacity samples deeper.

Say “photosphere around optical depth of order unity,” not “the photosphere is one geometric layer.”

**Act III checkpoint:** we know which depths can contribute, but we still need the temperature and pressure of those depths.

#### Act IV — Build the first atmosphere

##### Scene 17: Flux conservation

Return to

\[
F=\sigma T_{\rm eff}^{4}.
\]

Define radiative equilibrium as no net local radiative heating or cooling in the equilibrium model. In the grey radiative model, the same bolometric flux passes through each layer.

##### Scene 18: Compress the angular field into moments

Define:

\[
J
=
\frac12\int_{-1}^{1}I(\mu)\,d\mu,
\]

\[
H
=
\frac12\int_{-1}^{1}\mu I(\mu)\,d\mu,
\]

\[
K
=
\frac12\int_{-1}^{1}\mu^2 I(\mu)\,d\mu.
\]

Interpret:

- \(J\): mean intensity;
- \(H\): directed flux information, with \(F=4\pi H\);
- \(K\): radiation-pressure angular moment.

Define closure as an additional relation needed because moment equations create more unknown moments. Introduce Eddington closure \(K=J/3\).

##### Scene 19: Derive the grey temperature law

Use the grey moment equations, radiative equilibrium, and Eddington closure to obtain:

\[
\frac{dT^4}{d\tau_R}
=
\frac34T_{\rm eff}^{4}.
\]

Integrate:

\[
T^4(\tau_R)
=
\frac34T_{\rm eff}^{4}
\left[\tau_R+q(\tau_R)\right].
\]

Payne Zero's actual grey initializer baseline uses \(q=2/3\), giving:

\[
T(\tau_R=2/3)=T_{\rm eff}.
\]

Define \(q(\tau_{\rm R})\) as the general Hopf-function concept, but do not implement a fitted
correction in Chapter 1: no such fit is used by the pinned Payne Zero baseline. All executable
grey-temperature cells use the exact \(2/3\) value appearing in `warm_start.py` and
`direct_abundance.py`.

##### Scene 20: Why Rosseland depth

Return to the paper's exact depth relation
\(d\tau_{\rm R}=\kappa_{\rm R}\,dm\). In deep layers, radiation preferentially
diffuses through low-opacity windows, motivating:

\[
\frac{1}{\kappa_R}
=
\frac{
\int_0^\infty
\kappa_\nu^{-1}
\left(\partial B_\nu/\partial T\right)d\nu
}{
\int_0^\infty
\left(\partial B_\nu/\partial T\right)d\nu
}.
\]

Use a two-bin hand example to show why the mean is harmonic and sensitive to transparent windows.

Do not implement the production Rosseland integral here; Chapter 12 owns it.

##### Scene 21: Create the standard 80-layer grid

Define:

\[
\log_{10}\tau_R(d)
=
-6.875+0.125d,
\qquad
d=0,\ldots,79.
\]

Explain why logarithmic spacing resolves both the optically thin surface and deep diffusion layers.

**Check:** 80 elements, exact endpoint formula, constant 0.125 dex spacing, strict inward increase.

##### Scene 22: Support the layers against gravity

With \(z\) increasing outward:

\[
\frac{d(P_{\rm gas}+P_{\rm rad})}{dz}
=
-\rho g.
\]

Using \(dm=-\rho dz\):

\[
\frac{d(P_{\rm gas}+P_{\rm rad})}{dm}=g.
\]

For the first seed, explicitly declare a constant illustrative Rosseland opacity:

\[
m=\frac{\tau_R}{\kappa_R}.
\]

This is a normalization that attaches a mass and pressure scale. It is not a measured solar opacity.

##### Scene 23: Separate gas and radiation pressure

Introduce:

\[
P_{\rm rad}
=
\frac{aT^4}{3}
=
\frac{4\sigma T^4}{3c},
\]

under the isotropic blackbody/diffusion approximation.

Use the exact paper and supported-branch convention:

\[
P_{\rm gas}(m)+P_{\rm rad}(m)
=
gm+P_0.
\]

Connect this notation directly to
`integrate_hydrostatic_pressure(atmosphere, *, surface_gravity_cgs=...,
integrated_radiation_pressure=..., turbulent_pressure=..., pressure_constant=...)` and
`update_total_pressure(column_mass, *, surface_gravity_cgs=...,
pressure_zero_point=...)`.
For the supported branch, turbulent pressure is disabled. Do not rename \(P_0\) to a new
public argument; retain the exact `pressure_constant` and `pressure_zero_point` names at their
respective function boundaries.

Connect to:

\[
P_{\rm gas}=n_{\rm particles}kT.
\]

Stop there. Electron density, ionization, molecule formation, and mass density require the EOS in Chapters 3–4. Do not insert a hidden mean molecular weight.

##### Scene 24: Read the 80-layer atmosphere

Show:

- temperature versus \(\log_{10}\tau_R\);
- gas, radiation, and total pressure versus column mass;
- a compact table for the outermost, \(\tau_R\sim1\), and deepest layers.

Interpret the trends in prose.

Run checks:

- temperature increases inward for the Eddington seed;
- \(T(2/3)=T_{\rm eff}\);
- \(d(P_{\rm gas}+P_{\rm rad})/dm\approx g\);
- pressure and temperature are finite and positive;
- \(P_{\rm gas}+P_{\rm rad}=gm+P_0\) under the declared convention.

##### Scene 25: Name what is still missing

The derivation-only grey arrays have no:

- self-consistent electron density;
- ion-stage or molecular populations;
- realistic continuum or line opacity;
- line blanketing;
- convection;
- radiative acceleration;
- iterative temperature and column-mass correction;
- convergence claim.

End with an exact output contract, not a claim that an atmosphere has been solved.

### 2.6 Canonical code artifacts

The chapter displays and executes the exact pinned definitions; it adds no public wrapper module.

#### `payne_zero_synthesis.api`

- `Spectrum`
- `ForwardTimings`
- `InitializedAtmosphere`
- `LabelSpectrum`
- `build_structured_atmosphere`
- `save_structured_atmosphere`
- `synthesize`
- `initialize_atmosphere_from_labels`
- `synthesize_from_labels`

Only the first four class definitions and compact public signatures need to be rendered in Chapter
1. Long function bodies are deferred to the workflow chapters.

#### `payne_zero_synthesis.constants`

Use the exact constant names:

- `LIGHT_SPEED_CM_PER_S`
- `LIGHT_SPEED_NM_PER_S`
- `LIGHT_SPEED_ANGSTROM_PER_S`
- `PLANCK_ERG_SECOND`
- `BOLTZMANN_ERG_PER_K`
- `BOLTZMANN_EV_PER_K`
- `ATOMIC_MASS_GRAM`
- `NATURAL_LOG_10`

Chapter 2 introduces the `REFERENCE_*` names and explains why both tiers remain.

#### `payne_zero_synthesis.radiative_transfer`

Display the exact `planck_bnu` body. The frequency/wavelength conversion and \(B_\lambda\)
calculation that precede it in the derivation are chapter-local algebra, not newly exported
functions.

#### `payne_zero_atmosphere.atmosphere_io`

Introduce the exact `ModelAtmosphere` fields:

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

The grey calculation does not construct a fake `ModelAtmosphere`, because it has not produced
positive `electron_density` or `rosseland_opacity` and has not earned a convergence claim.

#### `payne_zero_atmosphere.run_setup` and `payne_zero_atmosphere.hydrostatic`

Use:

- `standard_rosseland_optical_depth_grid`
- `integrate_hydrostatic_pressure`
- `update_total_pressure`

The grey \(T(\tau_{\rm R})\) expression is shown exactly as it appears in the initializer decoding
code, but no new grey-temperature or seed API is invented.

#### Transfer demonstration

The constant-source slab is explicitly derivation-only. Optical-depth code calls the real
`payne_zero_atmosphere.radiative_transfer.integrate_on_depth_grid`; Chapter 2 then opens its exact
parabolic implementation and the synthesis batched counterpart.

### 2.7 Shapes, units, dtype, and device

#### Radiation and spectrum arrays

| Name | Shape | Unit | Ordering | Dtype/device |
| --- | --- | --- | --- | --- |
| `Spectrum.wavelength_nm` | `(W,)` | nm | increasing | NumPy `float64`, host |
| `planck_bnu` argument `wavelength_nm` | `(W,)` | nm | increasing | Torch runtime dtype/device |
| local `frequency_hz` inside `planck_bnu` | `(W,)` | Hz | decreasing at corresponding wavelength samples | same Torch dtype/device |
| `planck_bnu` argument `temperature` | `(D,)` | K | outer to inner | same Torch dtype/device |
| return of `planck_bnu` | `(D,W)` | erg s\(^{-1}\) cm\(^{-2}\) sr\(^{-1}\) Hz\(^{-1}\) | `[depth,wavelength]` | same Torch dtype/device |
| `flux_total`, `flux_continuum` | `(W,)` | erg s\(^{-1}\) cm\(^{-2}\) nm\(^{-1}\) | increasing wavelength | NumPy `float64`, host |
| `normalized_flux` | `(W,)` | dimensionless | increasing wavelength | NumPy `float64`, host |

#### Transfer arrays

| Name | Shape | Unit | Convention | Dtype/device |
| --- | --- | --- | --- | --- |
| CPU `grid` / `values` in the Rosseland call (`column_mass` / `accumulator`) | `(D,)` / `(D,)` | g cm\(^{-2}\) / cm\(^2\) g\(^{-1}\) | outer to inner | NumPy `float64`, CPU |
| CPU `integrate_on_depth_grid` return | `(D,)` | dimensionless | includes `surface_value` at index 0 | NumPy `float64`, CPU |
| synthesis `column_mass` | `(D,)` | g cm\(^{-2}\) | outer to inner | Torch runtime dtype/device |
| synthesis `extinction` | `(W,D)` | cm\(^2\) g\(^{-1}\) | `[wavelength,depth]` | Torch runtime dtype/device |
| synthesis `surface_tau` | `(W,)` | dimensionless | top half-cell seed | Torch runtime dtype/device |
| synthesis optical-depth return | `(W,D)` | dimensionless | `[wavelength,depth]` | Torch runtime dtype/device |
| \(I_\nu,S_\nu,\mu\) slab values | scalar or matching derivation arrays | spectral intensity / dimensionless | derivation-only | chapter-local |

#### Grey atmosphere arrays

| Name | Shape | Unit | Convention | Dtype/device |
| --- | --- | --- | --- | --- |
| `standard_rosseland_optical_depth` | `(80,)` | dimensionless | increases inward | NumPy `float64`, CPU |
| `temperature` | `(80,)` | K | outer to inner | NumPy `float64`, CPU |
| `column_mass` | `(80,)` | g cm\(^{-2}\) | increases inward | NumPy `float64`, CPU |
| `gas_pressure` | `(80,)` | dyn cm\(^{-2}\) | exact atmosphere field name | NumPy `float64`, CPU |
| `integrated_radiation_pressure` | `(80,)` | dyn cm\(^{-2}\) | exact hydrostatic argument | NumPy `float64`, CPU |
| `surface_gravity_cgs` | scalar | cm s\(^{-2}\) | \(10^{\log g}\) | Python float |
| `pressure_constant` / \(P_0\) | scalar | dyn cm\(^{-2}\) | exact code argument / paper symbol | Python float |

### 2.8 Chapter-level checks

The chapter’s final verification cell should summarize, without hiding individual earlier checks:

1. exact public API products have the pinned fields, shapes, and safety metadata;
2. wavelength/frequency conversion round-trips;
3. Planck coordinate conversion is correct;
4. Planck limits and broad-band integral behave as expected;
5. slab solution passes zero/thin/thick limits;
6. optical depth increases inward;
7. the 80-layer grid has exact declared spacing;
8. grey temperature obeys its boundary and monotonicity checks;
9. hydrostatic pressure has the correct discrete gradient;
10. the real `InitializedAtmosphere` preview carries
    `atmosphere_converged=False` and `atmosphere_closure_required=True`, while the chapter-local
    grey arrays are not serialized as an atmosphere product.

This is a physical and numerical sanity suite, not Payne Zero parity and not proof of convergence.

### 2.9 Exercises

#### Conceptual

1. Explain why \(T_{\rm eff}\) is not a layer temperature.
2. Trace how an abundance change can affect another element’s line.
3. Explain why LTE does not require \(I_\nu=B_\nu\) everywhere.
4. Explain why the photosphere is wavelength dependent.
5. Explain what the Eddington closure contributes that the moment equations alone do not.

#### Derivation

6. Derive \(B_\lambda\) from \(B_\nu\) using the Jacobian.
7. Derive the Rayleigh–Jeans and Wien limits.
8. Derive the constant-source slab solution.
9. Derive the slanted-ray factor \(\Delta\tau/\mu\).
10. Derive \(d(P_{\rm gas}+P_{\rm rad})/dm=g\) and its integrated
    \(P_{\rm gas}+P_{\rm rad}=gm+P_0\) form.

#### Computation

11. Map where the naive Planck exponential loses accuracy or overflows.
12. Find optical depths removing 90%, 99%, and 99.9% of an incident ray.
13. Compare arithmetic and harmonic means for two opacity bins.
14. Vary \(\log g\) and predict the pressure response before running the code.
15. Compare a general Hopf \(q(\tau_{\rm R})\) on paper with the pinned Payne Zero
    \(q=2/3\) baseline; keep the executable path unchanged.
16. Compare the exact Rosseland-mode seed
    `surface_value=accumulator[0] * column_mass[0]` with an incorrectly zeroed top optical-depth
    seed.

### 2.10 Forward-reference and redundancy hazards

- Chapter 1 defines abundance labels only as inputs. Chapter 2 owns the standard-pattern log
  abundances, the direct `(97,)` dex vector, and the deck-decoded `(99,)` linear array.
- It shows the hardware map once but does not teach Numba, Torch, CUDA, or MPS. Chapter 2 owns those mechanics.
- It derives \(S=B\) once. Chapter 9 later reuses it within full scattering transfer.
- It defines opacity and optical depth but does not catalogue real continuum or line sources. Chapters 5–8 own them.
- It solves one constant-source slab, not a variable-source or scattering atmosphere. Chapter 9 owns full transfer.
- It motivates the Rosseland mean with a two-bin example. Chapter 12 owns the production integral.
- It evaluates derivation-only grey arrays and the exact Payne Zero standard grid; it does not
  create a new public seed product. Chapters 11–13 own production hydrostatics, blanketing,
  correction, and closure.
- It must not save those grey arrays as a schema-v4 physical atmosphere; required EOS and
  population fields do not exist yet.
- The CPU/initializer/GPU map must stay to one figure and one paragraph.

---

## 3. Chapter 2 — From Equations to Fast, Trustworthy Kernels and Explicit Data

### 3.1 Opening physical question

**How can we make a scientific equation fast and reusable without changing its meaning or losing track of the data that define it?**

Open by timing a chapter-local harness that calls the exact readable
`integrate_on_depth_grid` over many independent toy frequency columns on the 80-layer grid. The
harness is not a new library interface; it exposes why Payne Zero also has a compiled enclosing
transfer loop and a distinct batched Torch synthesis integral. Then show two same-shaped population
arrays with different meanings. Speed and explicitness become two sides of the same requirement:
preserve the scientific contract.

Do not open with a list of libraries or a benchmark bar chart.

### 3.2 Chapter contract

| Item | Contract |
| --- | --- |
| Reads | Chapter 1 optical-depth relation, exact depth convention, exact constants, grey-array scale |
| Writes | Exact-source annotations; parity/timing evidence; exact abundance vectors; source-catalog verification result; schema-v4 validation result |
| Kernel example | Payne Zero's parabolic depth integration in its NumPy, compiled CPU, and batched Torch forms |
| Core shapes | CPU one-frequency depth arrays `(D,)`; synthesis `[wavelength,depth]` arrays `(W,D)`; direct solver abundance vector `(97,)`; deck/builder linear abundance array `(99,)` and schema length at least 99; populations `(D,6,139)` |
| Runtime policy | Atmosphere NumPy/Numba CPU; synthesis Torch CUDA/MPS/CPU; initializer Torch |
| Dtype policy | CPU/CUDA work float64; MPS work float32; synthesis line accumulation float32 |
| Honest boundary | Agreement between implementations is not physical convergence; schema validity is not physical correctness; checksum equality is only file identity |

### 3.3 Pace and internal structure

The chapter should have five acts:

| Act | Question | Approximate share |
| --- | --- | --- |
| I. Freeze meaning | What exactly enters and leaves a kernel? | 15% |
| II. Accelerate without changing dependencies | Which work is vectorized, compiled, parallel, or device-resident? | 35% |
| III. Measure trustworthiness | What does parity mean, and what did timing include? | 15% |
| IV. Make composition and data explicit | Which physical objects and files do arrays represent? | 15% |
| V. Seal the atmosphere–synthesis boundary | What exact arrays must cross the interface? | 20% |

The title is broad, but the chapter has one spine: **a scientific result is trustworthy only when its equation, units, axes, numerical policy, data identity, and output meaning are all explicit.**

### 3.4 Prerequisite concepts introduced from scratch

#### Array and numerical language

- axis, dimension, shape, index order, stride, and contiguous memory;
- broadcasting;
- vectorization;
- kernel;
- recurrence, independent iteration, reduction, and reduction order;
- dtype, precision, overflow, underflow, and rounding;
- numerical parity, absolute error, relative error, and tolerance.

#### Numba and concurrency language

- just-in-time compilation;
- `njit`;
- compilation cache;
- Python global interpreter lock;
- `nogil`;
- thread;
- `parallel=True`;
- `prange`;
- race condition;
- fixed thread count and cross-thread reduction differences.

#### Torch and device language

- tensor;
- host and device;
- CPU, CUDA, and Apple Metal/MPS;
- device transfer;
- asynchronous execution and synchronization;
- device-resident invariant;
- one final host transfer.

#### Composition and data language

- element, atomic number, atom, isotope, ion, ionization stage, molecule, and species;
- number density and number fraction;
- dex, \(A(X)\), \([X/H]\), \([X/Fe]\), \([M/H]\), and \([\alpha/M]\);
- reference mixture;
- source catalog, invariant table, generated subset/cache, fixture, and golden;
- schema, manifest, checksum, SHA-256, and provenance;
- actual and partition-normalized population;
- fixed-electron-density bridge.

Every term receives a small example before it appears in the full kernel or schema.

### 3.5 Detailed derivation and narrative arc

#### Act I — Freeze meaning

##### Scene 1: Write the contract before optimizing

Reuse, without rederiving:

\[
\tau_\nu(m)
=
\tau_{\nu,0}
+
\int_{m_0}^{m}\chi_\nu(m')\,dm'.
\]

Write the two exact contracts side by side:

| Function | Inputs | Output |
| --- | --- | --- |
| `payne_zero_atmosphere.radiative_transfer.integrate_on_depth_grid` | `grid (D,)`, `values (D,)`, keyword-only `surface_value` | one `(D,)` integral in NumPy float64 |
| `payne_zero_synthesis.radiative_transfer.integrate_optical_depth` | `column_mass (D,)`, `extinction (W,D)`, `surface_tau (W,)` | `(W,D)` optical depth on the input Torch device/dtype |

Annotate dependencies:

- every frequency/wavelength row is independent of every other row;
- every depth step depends on the previous depth step.

The two implementations share the physical integral but deliberately do not share one cleaned-up
signature or axis order.

##### Scene 2: Units, axes, and names are executable safeguards

Show three bugs that can all produce numerically plausible arrays:

1. using meters where cm are expected;
2. passing `[depth,wavelength]` where `integrate_optical_depth` requires
   `[wavelength,depth]`;
3. reversing depth so column mass decreases with index.

Use the exact existing validation boundaries rather than adding a renamed validator. Explain that
exact parameter and field names do not perform dimensional algebra, but the surrounding unit/shape
contract makes mistakes visible.

**Output:** a small “contract failures” table with the rejected condition and clear error.

#### Act II — Accelerate without changing dependencies

##### Scene 3: The readable exact NumPy path

Open the exact bodies of:

- `parabolic_coefficients(values, grid)`;
- `integrate_on_depth_grid(grid, values, surface_value=...)`.

Use `D=4` values that can be checked by hand. Explain the constant, linear, and quadratic interval
coefficients, the deliberately linearized near-surface intervals, and the ordered integral
recurrence. This public NumPy helper is the readable executable truth; do not replace it with a
trapezoidal teaching function.

##### Scene 4: Array algebra and broadcasting in the exact batched path

Open the exact Torch `_parabolic_interval_coefficients(values, depth_grid)` and show its actual
shapes:

```text
values                         (W, D)
depth_grid                     (D,)
constant, linear, quadratic    (W, D)
interval_width                 (D-1,)
interval_tau                   (W, D-1)
optical_depth                  (W, D)
```

Explain broadcasting where the same depth-grid factors multiply every wavelength row. The exact
`torch.cumsum(interval_tau, dim=1)` is ordered along the depth axis.

##### Scene 5: Strides and memory access

Inspect both exact layouts:

- atmosphere opacity slabs commonly arrive `[depth,frequency]` and the compiled CPU loop extracts
  one depth column per `frequency_index`;
- synthesis transfer stores batch rows as `[wavelength,depth]`, so its depth axis is contiguous.

Show two real layout operations where they actually occur: the atmosphere runner passes its slabs
through `np.ascontiguousarray` before `accumulate_transfer_range_parallel`, while
`solve_scattering_source` temporarily uses `.transpose(0, 1).contiguous()` so its sequential depth
loop is depth-major and transposes back on return. Do not invent a general-purpose bridge between
the two packages or imply that one layout should replace the other.

##### Scene 6: Compile the loop with `njit`

Show the exact compiled definitions inside `payne_zero_atmosphere.transfer_kernels`:

1. `_njit = numba.njit(cache=True, nogil=True)`;
2. `_parabolic_coefficients_compiled`;
3. `_integrate_on_depth_grid_compiled`.

Explain each option:

- `njit` compiles numerical Python to machine code;
- `cache=True` stores reusable compilation artifacts but does not make every first call free;
- `nogil=True` releases Python’s interpreter lock during compiled work; it does not itself make the loop parallel;
- `parallel=True` enables Numba's parallel transformation;
- `prange` marks the exact outer chunk loop whose iterations own private accumulators.

The compiled depth integrator remains serial. Do not present it as a parallel public
optical-depth API.

##### Scene 7: When parallel work is too small

Open `transfer_chunk_count` and the exact structure of
`accumulate_transfer_range_parallel`:

- obtain the Numba-thread count from `transfer_chunk_count()` and set
  `chunk_count = min(transfer_chunk_count(), max(1, stop - start))` at the runner call site;
- split `[range_start,range_stop)` into contiguous frequency chunks;
- allocate one private accumulator set per chunk;
- `numba.prange` over chunks;
- run `accumulate_transfer_range_compiled` inside each chunk;
- reduce chunk buffers in chunk order.

Benchmark this exact path, not a synthetic `prange` optical-depth function. Explain that each
frequency's work is independent while regrouping the final floating-point reduction can change a
few ulps. Fixed chunk/thread count and cross-thread comparisons are different gates.

##### Scene 8: Express the scan in Torch

Introduce a tensor as an array with dtype and device. Execute the exact
`integrate_optical_depth(column_mass, extinction, surface_tau)` with
`extinction[wavelength,depth]`. Its `torch.cumsum(..., dim=1)` retains the exact synthesis axis
order.

Teach the runtime policy:

- default device preference: CUDA, then MPS, then CPU;
- default work dtype: float64 on CUDA and CPU, float32 on MPS;
- line opacity accumulation later remains float32 on every synthesis backend;
- large synthesis tensors stay device resident;
- completed spectra cross to host once;
- discrete bracket/regime decisions may remain host float64 where they define parity-sensitive branches.

The exact synthesis function runs on available devices for comparison. This does not move the
production atmosphere iteration to GPU.

##### Scene 9: Why the architecture is split

Use mathematical structure, not marketing:

- atmosphere iteration has ordered state updates, depth continuation, and many modest CPU kernels;
- synthesis has large depth–wavelength fields and many independent wavelength/line operations;
- learned initializers are tensor models.

Use the existing CPU/GPU official schematic as the visual anchor.

**Act II checkpoint:** one physical depth integral now has its actual Payne Zero NumPy,
compiled-CPU, chunk-parallel transfer, and batched-Torch realizations. Their purposeful signature
and layout differences remain visible.

#### Act III — Measure trustworthiness

##### Scene 10: Define parity claims

Test:

1. a hand-checkable exactly representable case;
2. a smooth physical case;
3. a dynamic-range stress case.

For every exact candidate/reference comparison report:

- maximum absolute error;
- maximum relative error;
- worst index;
- reference and candidate dtype;
- device;
- tolerance policy;
- monotonicity and shape result.

Exact equality is appropriate only when operation order and representation make it meaningful. Float32 and float64 use separately declared tolerances.

These values form a chapter result table or test assertion record, not a new runtime product:
Payne Zero's callable boundaries and return values stay unchanged.

##### Scene 11: Benchmark cold and warm honestly

Separate:

- imports and setup;
- first Numba compilation;
- warm same-process Numba calls;
- fresh-process cache reload;
- device allocation;
- first device kernel;
- synchronized warm device calls;
- host-to-device transfer;
- device-to-host transfer;
- end-to-end time.

Use `time.perf_counter`, multiple repeats, and medians. Synchronize CUDA/MPS before stopping the clock.

Never time an asynchronous launch alone and label it GPU runtime.

##### Scene 12: Exact and reference constant tiers

Chapter 1 used the synthesis package's exact constants for new physical derivations. Show the
actual, deliberately stage-specific names:

- `payne_zero_synthesis.constants` uses unsuffixed exact names including
  `PLANCK_ERG_SECOND`, `BOLTZMANN_ERG_PER_K`, `BOLTZMANN_EV_PER_K`, and
  `ATOMIC_MASS_GRAM`, alongside the formula-specific
  `REFERENCE_PLANCK_ERG_SECOND`, `REFERENCE_BOLTZMANN_ERG_PER_K`,
  `REFERENCE_BOLTZMANN_EV_PER_K`, `REFERENCE_ATOMIC_MASS_GRAM`,
  `REFERENCE_WAVENUMBER_PER_EV`, `REFERENCE_SAHA_COEFFICIENT`, and
  `REFERENCE_NATURAL_LOG_10`;
- `payne_zero_atmosphere.constants` uses names including
  `PLANCK_ERG_SECOND_EXACT` and `BOLTZMANN_ERG_PER_K_EXACT`, alongside names including
  `PLANCK_ERG_SECOND_REFERENCE` and `BOLTZMANN_ERG_PER_K_REFERENCE`;
- source-conversion conventions retain their exact distinct names, especially
  `payne_zero_synthesis.constants.LINE_CATALOG_WAVENUMBER_PER_EV` and
  `payne_zero_atmosphere.constants.WAVENUMBER_PER_EV_REFERENCE`; the
  synthesis-specific `HYDROGEN_PROFILE_ATOMIC_MASS_GRAM` also stays separate from the general and
  reference atomic-mass names.

The suffix policy therefore differs between the two packages. Do not summarize it as a fictional
book-wide `*_EXACT` interface, cross-import the packages, or replace one literal merely because it
represents the same physical quantity.

**Act III checkpoint:** speed is now reported together with numerical agreement, dtype, device, and timing boundary.

#### Act IV — Make composition and data explicit

##### Scene 13: Distinguish element, isotope, ion, and molecule

Use:

- hydrogen as an element;
- protium and deuterium as isotopes;
- H I and H II as ionization stages;
- H\(_2\) as a molecule.

Explain that a numerical species slot is an indexing convention. It must not be guessed from array position.

##### Scene 14: Derive abundance notation

Define:

\[
A(X)
=
\log_{10}\left(\frac{n_X}{n_H}\right)+12,
\]

\[
[X/H]
=
\log_{10}\left(\frac{n_X}{n_H}\right)_\star
-
\log_{10}\left(\frac{n_X}{n_H}\right)_\mathrm{ref},
\]

\[
[X/Fe]=[X/H]-[Fe/H].
\]

Define:

- \([M/H]\) as a declared global metal-pattern coordinate, not one element;
- \([\alpha/M]\) as the Payne Zero offset for exactly O, Ne, Mg, Si, S, Ca, and Ti, recorded by
  `ALPHA_ELEMENT_ATOMIC_NUMBERS = (8, 10, 12, 14, 16, 20, 22)`;
- dex as a factor-of-ten logarithmic interval.

Then follow the two exact abundance paths without merging their representations:

1. `compute_metal_log_number_abundances(metallicity, alpha_enhancement,
   absolute_abundance_offsets)` returns a `(97,)` array of log10 number abundances for
   \(Z=3,\ldots,99\).
   \([M/H]\) shifts all metals, \([\alpha/M]\) shifts the exact seven-element set, and an entry in
   `absolute_abundance_offsets` is an absolute element bracket offset that overrides the scalar
   pattern for that element. `compute_hydrogen_fraction` completes H after the fixed He abundance
   and metals are assigned.
2. `complete_direct_abundance_vector(abundance_by_atomic_number)` requires all 81 public
   \([X/H]\) labels, fills the 16 solver slots without a public solar reference from \([Fe/H]\),
   quantizes to 0.01 dex, and returns the authoritative `(97,)` solver vector.
   `retained_direct_abundance_mixture` maps a retained \([Fe/H]\), \([X/Fe]\) coordinate set to the
   complete public mapping and realized vector. At the higher
   `initialize_atmosphere_from_labels` boundary, `x_over_h` may be sparse: unspecified public
   elements inherit `fe_over_h` before the complete 81-label mapping reaches this function.
   `metallicity` and `alpha_enhancement` do not apply in that direct-abundance mode.

At the external deck boundary, `linear_elemental_abundances(model)` decodes the fixed-column
abundance block into 99 linear values indexed by `Z-1`: H and He are already stored linearly,
while metals are decoded as \(10^{\text{stored value}}\). It does not divide by the sum or silently
renormalize the deck. The public structured-atmosphere builder documents `(99,)`; the schema
validator accepts a positive one-dimensional `elemental_abundances` array with at least 99 entries.

##### Scene 15: Four data roles

Separate:

1. **Static physical source data:** full line catalogs and atomic/molecular source tables.
2. **Invariant numerical tables:** converted coefficients and grids that are independent of a star or wavelength window.
3. **Generated runtime subsets/caches:** rebuildable window-selected or compiled products tied to source identity.
4. **Verification data:** small fixtures and trusted goldens, stored separately from physical inputs.

Fixtures and goldens are distinguished within the fourth category:

- fixture: intelligible input designed to exercise a branch;
- golden: expected output tied to exact code/data identity.

A golden is never used as a physical input table.

##### Scene 16: Manifest and checksum

Open the exact source-catalog identity path:

- `payne_zero_atmosphere.source_catalogs.source_catalog_root()` locates the staged catalog tree;
- `load_source_catalog_checksums()` parses the committed `CHECKSUMS.sha256` mapping;
- `source_line_paths()` and `atmosphere_source_catalog_paths()` resolve the files required by the
  corresponding run;
- `verify_source_catalog_checksums(root=..., checksum_path=...)` hashes the staged files and
  returns the exact verification dictionary, including `status`, `root`, `checksum_manifest`,
  `checksum_manifest_sha256`, `file_count`, `total_bytes`, and `files`.

Each returned file record contains `path`, `bytes`, and `sha256`. Explain that NPY/NPZ preserve
dense numeric dtype and shape, while the checksum file binds relative paths to exact bytes. A
checksum establishes identity, not scientific correctness. The internal `_sha256` helper remains
private; do not export it under a book-defined name.

Demonstrate failure after changing one byte in a small staged fixture. Multi-gigabyte source
verification is an explicit command, not an automatic rendered cell.

**Act IV checkpoint:** composition and every physical data input now have explicit representation and identity.

#### Act V — Seal the atmosphere–synthesis boundary

##### Scene 17: Why a schema is a scientific interface

Show two same-shaped arrays:

\[
n_{d,i,Z}
\quad\text{and}\quad
\frac{n_{d,i,Z}}{U_{d,i,Z}}.
\]

The first is actual ion-stage number density. The second is partition-normalized population. Both may have shape `(D,6,139)` and density-like units, but they are not interchangeable.

Use the concrete example:

\[
n=10^{12}\ \mathrm{cm^{-3}},
\quad
U=4,
\quad
n/U=2.5\times10^{11}\ \mathrm{cm^{-3}}.
\]

State:

- `partition_normalized_populations` is not a bound-level population;
- `ion_stage_populations` is required for charge-weighted free-free opacity;
- the later LTE bound-level calculation multiplies \(n/U\) by a statistical weight and Boltzmann factor.

##### Scene 18: The fixed-\(n_e\) bridge

The converged atmosphere solver has already established an electron-density column. When assembling the synthesis-shaped populations, the bridge evaluates populations at that supplied \(n_e\). It must not silently run a different full charge-balance solve and change the atmosphere.

Chapter 3 derives the atomic population machinery. Chapter 2 establishes only the interface promise:

- full charge solve: determines \(n_e\);
- fixed-\(n_e\) bridge: preserves supplied \(n_e\) while filling synthesis fields.

##### Scene 19: Validate representation

`validate_atmosphere_npz(path)` calls the exact NPZ loader/validator and returns
`REQUIRED_ATMOSPHERE_ARRAYS`. The validation path checks:

- schema version;
- required canonical names;
- symbolic shapes and common depth count;
- numeric dtype and finiteness;
- positivity or nonnegativity where required;
- strictly increasing column mass;
- outer-to-inner ordering;
- presence and common `(D,6,139)` shape of the separately named actual and
  partition-normalized arrays;
- compatible continuum-edge lengths.

For an archive declaring schema version 4, storage uses the canonical names above. The exact loader
also has internal read compatibility for older supported schema versions; Chapter 2 neither writes
those forms nor teaches their aliases.

`load_atmosphere_product_metadata(path)` separately reads the all-or-none typed product-metadata
extension, checks that the typed values agree with `atmosphere_metadata_json`, and returns the exact
metadata dictionary or `None`. An initialized product identifies its role as
`learned_initializer_prediction` and carries `atmosphere_converged=False` and
`atmosphere_closure_required=True`. `validate_atmosphere_npz` accepts only `path`; downstream code
must interpret these explicit product fields at the boundary where convergence is required.

It does not prove:

- charge conservation;
- radiative equilibrium;
- hydrostatic closure;
- convergence;
- agreement with Payne Zero.

##### Scene 20: Read the validation ladder

End with four distinct claims:

1. **Checksum identity:** these are the named bytes.
2. **Schema validity:** these arrays have the required names, shapes, units, ordering, and metadata.
3. **Numerical parity:** another implementation agrees within a declared policy.
4. **Physical acceptance:** conservation, closure, convergence, and regime checks pass.

No one claim substitutes for another.

### 3.6 Canonical code artifacts

The chapter renders and annotates exact excerpts from the pinned modules. It adds no
`acceleration.py`, parity/timing dataclass, abundance-conversion wrapper, manifest wrapper, or
mapping validator.

#### `payne_zero_atmosphere.radiative_transfer`

- `parabolic_coefficients(values, grid)`
- `integrate_on_depth_grid(grid, values, *, surface_value)`

These are the readable NumPy one-depth-column definitions.

#### `payne_zero_atmosphere.transfer_kernels`

- `_njit = numba.njit(cache=True, nogil=True)`
- `_parabolic_coefficients_compiled(value, coordinate)`
- `_integrate_on_depth_grid_compiled(coordinate, value, surface_value)`
- `transfer_chunk_count()`
- `accumulate_transfer_range_compiled(...)`
- `accumulate_transfer_range_parallel(...)`

The two leading-underscore functions are private compiled implementation details. The public
parallel transfer path owns per-chunk accumulators and reduces chunks in deterministic chunk order;
it is not exposed as a parallel version of `integrate_on_depth_grid`.

#### `payne_zero_synthesis.radiative_transfer` and `payne_zero_synthesis.device`

- `_parabolic_interval_coefficients(values, depth_grid)`
- `integrate_optical_depth(column_mass, extinction, surface_tau)`
- `device()`
- `resolve_runtime(requested_device=None, requested_dtype=None)`
- `ACCUMULATION_DTYPE`, `DEFAULT_DTYPE`, `REFERENCE_DTYPE`

The private coefficient helper and public batched integral retain their exact Torch layout and
runtime dtype/device.

#### Exact constant modules

Show the exact tier names in `payne_zero_synthesis.constants` and
`payne_zero_atmosphere.constants`. The packages remain separately pinned and are never hidden
behind a global compatibility mode.

#### Exact abundance modules

Use:

- `payne_zero_atmosphere.warm_start.compute_metal_log_number_abundances`;
- `payne_zero_atmosphere.warm_start.compute_hydrogen_fraction`;
- `payne_zero_atmosphere.warm_start.ALPHA_ELEMENT_ATOMIC_NUMBERS`;
- `payne_zero_atmosphere.direct_abundance.complete_direct_abundance_vector`;
- `payne_zero_atmosphere.direct_abundance.retained_direct_abundance_mixture`;
- `payne_zero_atmosphere.atmosphere_io.linear_elemental_abundances`.

Small logarithm calculations used to derive bracket notation remain unnamed, chapter-local
arithmetic, not new exported conversion functions.

#### Exact source-catalog boundary

Use:

- `payne_zero_atmosphere.data_files.data_root`;
- `payne_zero_atmosphere.data_files.atmosphere_table_dir`;
- `payne_zero_atmosphere.data_files.atmosphere_table_path`;
- `payne_zero_atmosphere.data_files.atmosphere_emulator_dir`;
- `payne_zero_synthesis.paths.data_root`;
- `payne_zero_synthesis.paths.source_catalog_root`;
- `payne_zero_synthesis.paths.source_catalog_path`;
- `payne_zero_atmosphere.source_catalogs.source_catalog_root`;
- `payne_zero_atmosphere.source_catalogs.load_source_catalog_checksums`;
- `payne_zero_atmosphere.source_catalogs.verify_source_catalog_checksums`;
- `payne_zero_atmosphere.source_catalogs.source_line_paths`;
- `payne_zero_atmosphere.source_catalogs.atmosphere_source_catalog_paths`.

The committed identity file is `source_data_files/source_catalogs/CHECKSUMS.sha256`.
`source_data_files/runtime_data_manifest.json` is a separate runtime data manifest; do not
conflate it with the source-catalog checksum parser.

#### Exact structured-atmosphere boundary

Use:

- `payne_zero_synthesis.atmosphere.ATMOSPHERE_SCHEMA_VERSION`;
- `REQUIRED_ATMOSPHERE_ARRAYS`;
- `load_atmosphere_npz(path)`;
- `load_atmosphere_product_metadata(path)`;
- `validate_atmosphere_npz(path)`.

The declarative schema file is
`payne_zero_synthesis/atmosphere_schema.json`. A tiny NPZ fixture may be used to teach validation,
but it is labeled **synthetic schema data, not a physical atmosphere**.

### 3.7 Backend and performance contract

| Exact implementation | Shapes/layout | Dtype/device | Independent work | Ordered work |
| --- | --- | --- | --- | --- |
| `radiative_transfer.integrate_on_depth_grid` | `grid (D,)`, `values (D,)` → `(D,)` | NumPy float64, CPU | none inside one call | parabolic depth integral |
| `transfer_kernels._integrate_on_depth_grid_compiled` | `(D,)`, `(D,)` → `(D,)` | Numba float64, CPU | none inside one call | parabolic depth integral |
| `transfer_kernels.accumulate_transfer_range_compiled` | atmosphere slabs commonly `[depth,frequency]` | Numba, CPU | frequency work inside one assigned range | each depth integral and local accumulation |
| `transfer_kernels.accumulate_transfer_range_parallel` | same transfer slabs plus per-chunk buffers | Numba multicore CPU | contiguous frequency chunks via `prange` | each chunk; final reduction in chunk order |
| `synthesis.radiative_transfer.integrate_optical_depth` | `(D,)`, `(W,D)`, `(W,)` → `(W,D)` | Torch runtime dtype/device | wavelength rows/elements | `cumsum` along depth axis 1 |

CUDA and CPU synthesis default to float64 work, while MPS uses float32; line-opacity accumulation
later uses `ACCUMULATION_DTYPE = torch.float32`. The chapter must show that `nogil=True`,
`parallel=True`, and `prange` answer different questions. `prange` belongs to the exact chunk loop,
not to a fabricated wavelength-parallel optical-depth API.

### 3.8 Exact schema-v4 array inventory

The complete contract appears once, grouped by role. `D` is depth count and `Q` is continuum-edge count.

#### Thermodynamic depth columns

| Field | Shape | Unit | Meaning/physical owner |
| --- | --- | --- | --- |
| `temperature` | `(D,)` | K | Grey seed in Ch. 1; converged structure in Ch. 11–13 |
| `gas_pressure` | `(D,)` | dyn cm\(^{-2}\) | Hydrostatic gas pressure |
| `electron_density` | `(D,)` | cm\(^{-3}\) | Atomic/molecular closure in Ch. 3–4 |
| `mass_density` | `(D,)` | g cm\(^{-3}\) | EOS result |
| `column_mass` | `(D,)` | g cm\(^{-2}\) | Strictly increases inward |
| `microturbulence` | `(D,)` | cm s\(^{-1}\) | Later line-width state |
| `hc_over_kt` | `(D,)` | cm | Thermodynamic helper |

#### Population and width fields

| Field | Shape | Unit | Meaning/physical owner |
| --- | --- | --- | --- |
| `partition_normalized_populations` | `(D,6,139)` | cm\(^{-3}\) per partition function | \(n/U\), Ch. 3–4 |
| `ion_stage_populations` | `(D,6,139)` | cm\(^{-3}\) | Actual ion-stage number density |
| `fractional_doppler_widths` | `(D,6,139)` | dimensionless \(v/c\) | Ch. 3 and 6 |
| `hydrogen_neutral_population` | `(D,)` | cm\(^{-3}\) | Ch. 3–4 |
| `helium_neutral_population` | `(D,)` | cm\(^{-3}\) | Ch. 3–4 |
| `helium_singly_ionized_population` | `(D,)` | cm\(^{-3}\) | Ch. 3–4 |
| `molecular_hydrogen_population` | `(D,)` | cm\(^{-3}\) | Ch. 4 |
| `hydrogen_partition_normalized_ion_stage_populations` | `(D,2)` | cm\(^{-3}\) per partition function | Ch. 3 |
| `carbon_partition_normalized_ion_stage_populations` | `(D,2)` | cm\(^{-3}\) per partition function | Ch. 3 |
| `magnesium_neutral_partition_normalized_population` | `(D,)` | cm\(^{-3}\) per partition function | Ch. 3 |
| `aluminum_neutral_partition_normalized_population` | `(D,)` | cm\(^{-3}\) per partition function | Ch. 3 |
| `silicon_neutral_partition_normalized_population` | `(D,)` | cm\(^{-3}\) per partition function | Ch. 3 |
| `iron_neutral_partition_normalized_population` | `(D,)` | cm\(^{-3}\) per partition function | Ch. 3 |

#### Composition and invariant continuum-edge fields

| Field | Shape | Unit | Meaning/physical owner |
| --- | --- | --- | --- |
| `elemental_abundances` | `(n_element,)`, `n_element >= 99` | positive linear number abundance | Ch. 2; public builder documents exactly 99 |
| `signed_continuum_edge_frequency_hz` | `(Q,)` | Hz | Sign is interpolation metadata; magnitude is physical frequency |
| `continuum_edge_wavelength_nm` | `(Q,)` | nm | Ch. 5 |
| `continuum_edge_midpoint_wavelength_nm` | `(Q-1,)` | nm | Ch. 5 |
| `continuum_edge_interval_width_squared_over_two_nm2` | `(Q-1,)` | nm\(^2\) | Ch. 5 |

The archive also carries:

```text
atmosphere_schema_version = 4
```

as a one-element `int32` array.

Optional typed product metadata may declare:

- `atmosphere_product_metadata_schema`;
- `atmosphere_product_role`;
- `atmosphere_converged`;
- `atmosphere_closure_required`;
- `initializer_family`;
- `atmosphere_metadata_json`, whose object carries the matching role flags plus labels,
  provenance, and timings.

An initializer product written by `InitializedAtmosphere` is self-identifying with the exact role
`learned_initializer_prediction`. Historical and physically converged schema-v4 products may have
no metadata extension; `load_atmosphere_product_metadata` then returns `None`.

### 3.9 Schema shapes, dtype, and device

| Object | Shape | Unit | Dtype/device |
| --- | --- | --- | --- |
| `elemental_abundances` | `(n_element,)`, at least 99; builder path `(99,)` | positive linear number abundance | NumPy numeric host array |
| thermodynamic columns | `(D,)` | field-specific CGS | NumPy `float64`, host archive |
| population cubes | `(D,6,139)` | field-specific | NumPy `float64`, host archive |
| continuum-edge arrays | `(Q,)` or `(Q-1,)` | field-specific | NumPy `float64`, host archive |
| `atmosphere_schema_version` | `(1,)` | integer version | NumPy `int32`, host archive |
| source-catalog checksum result | dictionary with per-file records | bytes and SHA-256 identity | Python/host |

The host archive is backend-neutral. Synthesis uploads validated arrays later.

### 3.10 Checks and plots

#### Performance and numerical plots

- runtime versus workload size for the exact NumPy depth integrator, private compiled depth
  integrator, chunk-parallel transfer path, and available Torch synthesis devices, with unlike
  workloads clearly separated;
- speedup versus CPU thread count at fixed workload;
- cold versus warm timing grouped by compilation, transfer, and synchronized execution;
- maximum absolute and relative difference by backend and dtype.

#### Data and schema visuals

- periodic-table-style maps from atomic number to the deck `(99,)` index and direct solver `(97,)`
  index, with their different meanings explicit;
- abundance notation ladder from number ratio to \(A(X)\) to bracket notation;
- four-role data matrix;
- exploded structured-atmosphere schema by thermodynamics, populations, composition, and invariant edges;
- actual versus partition-normalized population comparison.

#### Required checks

- agreement of each exact implementation with a hand-checkable parabolic-integral case after any
  required explicit layout conversion;
- tolerance-based agreement on realistic float64;
- separately declared float32 tolerance for MPS;
- each exact function satisfies its own shape and axis contract;
- monotonic optical depth;
- no input mutation;
- fixed-thread repeatability report;
- cold and warm numerical identity;
- `[X/Fe] == [X/H] - [Fe/H]`;
- standard-pattern override precedence and the exact seven-element alpha set;
- direct abundance input has all 81 public labels, produces `(97,)`, and obeys 0.01-dex
  quantization;
- deck decoding yields 99 positive linear entries without implicit renormalization;
- checksum failure after changing one byte in a fixture;
- all schema fields present, finite, and shape-compatible;
- strictly increasing column mass;
- actual and partition-normalized meanings are tested with a fixture whose values satisfy the
  declared \(n\) and \(n/U\) relationship; schema validation itself checks names, shape,
  finiteness, and nonnegativity, not that physical relationship;
- `Q`/`Q-1` continuum-edge consistency;
- initialized product metadata reads back as `learned_initializer_prediction`, unconverged, and
  closure-required.

### 3.11 Exercises

#### Arrays and acceleration

1. Label the axes and units in each intermediate of `integrate_on_depth_grid` and
   `integrate_optical_depth`.
2. Draw the dependency graph and explain why `prange` owns contiguous frequency chunks while each
   depth integration stays ordered.
3. Create a deliberate scalar-reduction race in a noncanonical exercise and explain the result.
4. Compare float32 and float64 error for increments spanning many orders of magnitude.
5. Measure whether a transpose helps after including the copy.
6. Demonstrate why unsynchronized GPU timing is misleading.
7. Explain why `nogil=True` is not a synonym for `parallel=True`.
8. Measure the crossover where the exact chunk-parallel transfer path begins to help, or document
   why it does not on the available machine.

#### Abundances and data

9. Convert supplied \(A(X)\) values into number ratios.
10. Given `[O/Fe]` and `[Fe/H]`, compute `[O/H]`.
11. Use `compute_metal_log_number_abundances` to apply \([M/H]\), \([\alpha/M]\), and one absolute
    element override; separately inspect a complete direct-abundance `(97,)` vector.
12. Classify twelve example files into source, invariant, runtime subset/cache, fixture, or golden.
13. Write a small `CHECKSUMS.sha256` fixture and verify it through
    `verify_source_catalog_checksums`.
14. Explain why a checksum match cannot certify oscillator-strength accuracy.

#### Schema

15. Repair NPZ fixtures with reversed depth, wrong population shape, a missing actual-population
    cube, and an inconsistent edge length.
16. Explain why \(n/U\) is not a bound-level population.
17. Explain why fixed-\(n_e\) bridge construction must not silently rerun charge closure.
18. Explain why a schema-valid initialized atmosphere with explicit closure-required metadata may
    still be physically unacceptable.

### 3.12 Forward-reference and redundancy hazards

- Do not rederive optical depth. Chapter 2 optimizes the Chapter 1 contract.
- Do not turn the chapter into a survey of disconnected microbenchmarks. The same physical
  parabolic depth integral connects the exact implementations, but their real signatures and
  enclosing workloads remain distinct.
- Do not manufacture a Numba synthesis architecture or a GPU atmosphere architecture.
- Do not claim `prange` speedup without a measured workload and overhead.
- Do not call numerical parity physical verification.
- Do not call a checksum scientific validation.
- Do not derive partition functions, Saha ionization, charge closure, or molecular equilibrium. Chapters 3–4 own them.
- Do not teach continuum-edge interpolation. Chapter 5 owns it.
- Do not populate a random schema example and call it an atmosphere.
- Do not list old field aliases or implementation history. The canonical schema has one readable vocabulary.

---

## 4. Extensive schematic plan

### 4.1 Visual authority

All new schematics should match the established official assets:

- `assets/schematics/official/payne-zero-pipeline-v2.png`
- `assets/schematics/official/atmosphere-layers-v3.png`
- `assets/schematics/official/cpu-gpu-acceleration-v2.png`

These three images define the visual language:

- white or near-white paper background;
- hand-drawn navy outlines and arrows;
- cool slate-blue upper layers;
- warm beige deeper layers and secondary traces;
- wavy photon marks;
- lightly irregular, human line work;
- short lower-case labels;
- generous whitespace;
- one scientific claim per schematic.

New figures should reuse the atmosphere block, spectrum trace, wavy photon, arrow, and label styles so the book feels like one visual system.

### 4.2 Palette and material rules

Use the website tokens as the UI/color authority:

| Role | Website token | Figure use |
| --- | --- | --- |
| Paper | `--paper: oklch(0.992 0.004 85)` | Main background |
| Secondary paper | `--paper-2: oklch(0.975 0.005 85)` | Very light panel or layer fill |
| Navy ink | `--ink: oklch(0.255 0.012 264)` | Outlines, axes, principal arrows, labels |
| Slate ink | `--muted: oklch(0.55 0.012 264)` | Secondary annotations, hidden rays |
| Faint rule | `--rule: oklch(0.905 0.006 264)` | Grids and guides |
| Cosmic blue | `--accent: oklch(0.50 0.135 256)` | One active path or selected object |
| Pale blue | `--accent-soft: oklch(0.95 0.03 256)` | Upper atmosphere or selection halo |
| Warm beige | derived from `--ember-soft: oklch(0.95 0.035 62)` | Deep layers, thermal source, comparison trace |
| Warm accent | `--ember: oklch(0.58 0.13 52)` | Sparse warning or thermal highlight |

Constraints:

- no dark background for scientific schematics;
- no neon colors;
- no glossy 3D rendering;
- no photorealistic stars or hardware;
- no rainbow spectrum unless wavelength identity itself is the point;
- no more than navy, slate, beige, and one accent in a single figure;
- use color redundantly with shape, line style, or label so meaning survives grayscale.

### 4.3 Drawing grammar

#### Lines and arrows

- principal outlines: 2–3 px navy, subtly irregular;
- secondary guides: 1–1.5 px slate at reduced opacity;
- arrows: broad hand-drawn curve, open or simple triangular head;
- photon paths: navy/slate sine-like strokes;
- uncertain or deferred process: dashed slate, never a red alarm.

#### Objects

- star: simple disk with sparse wavy rays;
- atmosphere: reusable stacked cuboid with cool top and warm lower layers;
- spectrum: navy absorption trace with a slightly offset beige continuum/reference trace;
- CPU: sketched chip with a few cores, not a branded processor;
- GPU: sketched card or many-lane grid, not a product logo;
- data: paper sheet, folder, table grid, or archive box;
- manifest: small checklist page plus fingerprint/hash marks;
- population cube: stacked translucent slices with three labeled axes.

#### Labels

- two to five words;
- sentence case or lower case, matching existing assets;
- no paragraphs inside a figure;
- equations are typeset by the page beside or below the art unless one short equation is the subject;
- labels use navy; secondary numeric annotations use slate;
- handwritten figure labels should visually match the official assets, while body captions remain IBM Plex Sans.

#### Composition

- one dominant left-to-right or top-to-bottom reading path;
- at least 20% blank area around the principal object;
- maximum five labeled nodes in one schematic;
- complex systems become a sequence of two or three figures, not a dense all-in-one poster;
- recurring object placement is stable: surface/top is visually high, deeper/inward is down, wavelength increases left to right.

### 4.4 Format and delivery specifications

- master canvas: 1536 × 1024 px for standard 3:2 figures, matching the official assets;
- pipeline canvas: approximately 1600 × 960 px for wide 5:3 figures;
- produce a high-resolution PNG for the current reader and preserve an editable source or prompt specification;
- safe text area: 8% inset from every edge;
- test at a 720 px rendered width and on a narrow mobile viewport;
- smallest handwritten label should remain legible at 720 px;
- plots produced by code should use the website serif/sans typography but adopt the same navy/slate/beige palette;
- captions carry detail that would clutter the drawing;
- every figure has concise alt text that states the scientific relationship, not merely the objects.

### 4.5 Reusable motif library

Create these motifs once and reuse them:

1. `stellar-label-notebook` — small spiral notebook with star and label marks.
2. `layered-atmosphere` — the official cool-to-warm cuboid.
3. `photon-wave` — one wavy ray with optional arrowhead.
4. `spectrum-trace` — navy absorption trace plus beige continuum/reference.
5. `depth-arrow` — vertical arrow labeled “inward”.
6. `wavelength-arrow` — horizontal axis labeled “wavelength”.
7. `cpu-chip` — four-to-eight visible cores.
8. `gpu-lanes` — many parallel wavelength lanes.
9. `array-grid` — hand-drawn matrix with named axes.
10. `data-sheet` — source table or catalog sheet.
11. `manifest-page` — path, bytes, hash represented as three short lines.
12. `validation-badge` — outlined check, never used for unverified physical claims.
13. `closure-loop` — circular arrow around the atmosphere block.
14. `warning-ribbon` — muted beige/ember “closure required” marker.

Motifs should be stored as independent transparent-background assets or reusable layers so later chapters inherit the same objects.

### 4.6 Chapter 1 schematic inventory

#### Ch1-F01 — The observable and the question

- **Placement:** chapter opening, before definitions.
- **Message:** starlight is a wavelength-dependent observable whose structure needs explanation.
- **Composition:** small star disk at left; wavy ray; spectrum at right; large white center gap carrying the question.
- **Labels:** `starlight`, `wavelength`, `what shaped it?`
- **Colors:** navy trace, beige continuum shadow, slate grid.
- **Alt text:** “Light from a star becomes a spectrum with broad continuum shape and absorption lines.”

#### Ch1-F02 — Labels are not a spectrum

- **Placement:** after \(T_{\rm eff}\) and \(\log g\).
- **Message:** labels enter a chain of physical states before flux emerges.
- **Composition:** label notebook → layered atmosphere → opacity/source disk → spectrum.
- **Labels:** `stellar labels`, `atmosphere`, `opacity + transfer`, `spectrum`.
- **Reuse:** simplified top row of the official pipeline asset.
- **Accuracy gate:** no direct labels-to-spectrum arrow bypassing atmosphere physics.

#### Ch1-F03 — Two workflows and product roles

- **Placement:** atmosphere versus synthesis section.
- **Message:** an initialized atmosphere and a converged atmosphere are different product roles.
- **Composition:** label notebook → initialized block; one direct exploratory path to synthesis; one downward closure loop to converged block and then synthesis.
- **Labels:** `learned initializer`, `closure required`, `converged`, `synthesis`.
- **Color rule:** amber/beige warning ribbon on initialized path; navy/blue check only on converged path.
- **Reuse:** official pipeline asset with shortened labels.
- **Caption gate:** the initialized archive role is exactly `learned_initializer_prediction`, with
  `atmosphere_converged=false` and `atmosphere_closure_required=true`.

#### Ch1-F04 — Plane-parallel layers

- **Placement:** simplifying assumptions.
- **Message:** a one-dimensional atmosphere varies only inward.
- **Composition:** layered cuboid, upward emergent waves, vertical inward arrow.
- **Labels:** `surface`, `depth increases inward`, `emergent light`.
- **Reuse:** official atmosphere-layers asset.
- **Accuracy gate:** top is outermost; warm/deep layers are lower.

#### Ch1-F05 — Wavelength, frequency, and photon energy

- **Placement:** first radiation section.
- **Message:** longer waves have lower frequency and photon energy.
- **Composition:** three wave sketches of increasing wavelength; decreasing tick annotations for \(\nu\) and \(E\).
- **Labels:** `longer λ`, `lower ν`, `lower E`.
- **Avoid:** a rainbow bar; it adds decoration without explaining the inverse relations.

#### Ch1-F06 — Intensity versus flux

- **Placement:** after solid angle.
- **Message:** intensity follows one direction; flux sums projected directional contributions.
- **Composition:** small surface patch; cone of directions; one highlighted ray; hemisphere bracket.
- **Labels:** `one ray: I`, `project by μ`, `sum: F`.
- **Accuracy gate:** grazing rays have small projected contribution.

#### Ch1-F07 — Building the Planck spectrum

- **Placement:** Planck derivation.
- **Message:** mode count, photon energy, and thermal occupation multiply to make \(B_\nu\).
- **Composition:** three small hand-drawn cards joined by multiplication dots, then a thermal curve.
- **Labels:** `modes`, `energy hν`, `occupation`, `Planck spectrum`.
- **Equation:** only \(B_\nu\) beneath the art, typeset by the page.

#### Ch1-F08 — Spectral density needs a Jacobian

- **Placement:** \(B_\nu\) to \(B_\lambda\).
- **Message:** equal physical energy occupies unequal coordinate intervals.
- **Composition:** one shaded interval on a frequency axis paired by arrows to a differently sized wavelength interval.
- **Labels:** `same energy`, `different interval`, `Jacobian`.
- **Accuracy gate:** do not imply the two peak locations map directly.

#### Ch1-F09 — Thermal limits

- **Placement:** Rayleigh–Jeans/Wien discussion.
- **Message:** long-wave and short-wave approximations occupy different sides of the Planck curve.
- **Composition:** one navy Planck curve; slate long-wave tangent; beige short-wave exponential; sparse brackets.
- **Labels:** `Rayleigh–Jeans`, `Planck`, `Wien`.
- **Plot/art boundary:** this can be a code-generated plot styled to look consistent rather than a freehand illustration.

#### Ch1-F10 — Mean free path and attenuation

- **Placement:** extinction introduction.
- **Message:** more interaction targets shorten the distance a photon travels.
- **Composition:** two small material panels with sparse and dense dots; wavy photon paths of long and short extent.
- **Labels:** `long mean free path`, `short mean free path`.
- **Accuracy gate:** do not equate scattering with destruction; label the combined event as interaction/extinction.

#### Ch1-F11 — Constant-source slab

- **Placement:** formal slab solution.
- **Message:** emergent intensity blends attenuated incident light and locally emitted light.
- **Composition:** horizontal slab; incident ray entering from below/left; small beige emission waves inside; output ray at top/right.
- **Labels:** `incident`, `local source`, `emergent`.
- **Equation:** two color-matched terms beneath the figure.
- **Accuracy gate:** arrow directions must match the chosen outward ray and boundary convention.

#### Ch1-F12 — Wavelength-dependent photosphere

- **Placement:** after column-mass optical depth.
- **Message:** high-opacity and low-opacity wavelengths escape from different depths.
- **Composition:** layered atmosphere with two colored wavy rays beginning at different layers.
- **Labels:** `high opacity`, `low opacity`, `τν ≈ 1`.
- **Accuracy gate:** high opacity emerges from the shallower of the two origins.

#### Ch1-F13 — Angular moments

- **Placement:** grey-atmosphere moment section.
- **Message:** \(J,H,K\) retain different angular information.
- **Composition:** three small circular ray fields: symmetric, outward-weighted, and angle-squared-weighted.
- **Labels:** `mean J`, `flux H`, `pressure K`.
- **Avoid:** dense integral notation inside the illustration.

#### Ch1-F14 — Grey temperature law

- **Placement:** after the derivation.
- **Message:** the atmosphere warms inward, with \(T=T_{\rm eff}\) at \(\tau_R=2/3\) for the Eddington boundary.
- **Composition:** navy curve \(T/T_{\rm eff}\) versus \(\log\tau_R\); beige dot and callout at \(2/3\).
- **Labels:** `surface`, `τR = 2/3`, `deeper`.
- **Accuracy gate:** use log optical depth on the horizontal axis and state it.

#### Ch1-F15 — Why the Rosseland mean is harmonic

- **Placement:** Rosseland motivation.
- **Message:** radiative diffusion favors transparent windows.
- **Composition:** wall with several channels; one broad open beige channel carries most wavy rays while dark/slate blocked channels carry few.
- **Labels:** `opaque`, `window`, `flux finds the window`.
- **Avoid:** implying photons literally choose paths; caption states this is a transport analogy.

#### Ch1-F16 — Hydrostatic balance

- **Placement:** pressure derivation.
- **Message:** gravity pulls a layer inward while the pressure gradient supports it.
- **Composition:** one atmosphere slab with downward gravity arrow and upward pressure-force arrow; column-mass bracket.
- **Labels:** `gravity`, `pressure support`, `dm`.
- **Accuracy gate:** force directions and outward/inward convention must agree with equations.
- **Caption equation:** \(P_{\rm gas}+P_{\rm rad}=gm+P_0\) for the supported branch, with turbulent
  pressure disabled.

#### Ch1-F17 — The first 80-layer atmosphere

- **Placement:** culminating result.
- **Message:** the chapter has built a depth-indexed seed with temperature and pressure, not a converged physical atmosphere.
- **Composition:** layered block with three small side profiles \(T\), \(P_{\rm gas}\), \(m\); amber `seed` ribbon.
- **Labels:** `80 layers`, `temperature`, `pressure`, `closure required`.
- **Reuse:** atmosphere motif from official assets.

#### Ch1-F18 — What exists and what remains

- **Placement:** final boundary box.
- **Message:** separate completed foundation from missing closure.
- **Composition:** two widely separated columns with six icons maximum.
- **Labels left:** `Planck`, `optical depth`, `slab`, `grey seed`.
- **Labels right:** `EOS`, `real opacity`, `blanketing`, `closure`.
- **Color:** completed items navy/blue; deferred items slate outline, not crossed out.

### 4.7 Chapter 2 schematic inventory

#### Ch2-F01 — A kernel is an equation with a contract

- **Placement:** opening.
- **Message:** equation, units, axes, and invariants travel together.
- **Composition:** equation card in center surrounded by four small tags.
- **Labels:** `equation`, `units`, `shape`, `checks`.
- **Whitespace:** tags separated; no software logos.

#### Ch2-F02 — Named axes and broadcasting

- **Placement:** first array section.
- **Message:** one depth-grid interval vector is applied across all wavelength rows.
- **Composition:** horizontal `(D-1,)` strip aligned with the depth axis of a `(W,D-1)` grid;
  arrows show broadcasting by alignment, not literal data copying.
- **Labels:** `depth`, `wavelength`, `broadcast`.
- **Accuracy gate:** in synthesis transfer wavelength is axis 0 and depth is axis 1.

#### Ch2-F03 — Independent wavelengths, ordered depth

- **Placement:** dependency analysis.
- **Message:** each depth integral is ordered; CPU parallelism assigns contiguous frequency chunks
  private accumulators.
- **Composition:** a `[depth,frequency]` CPU slab divided into vertical frequency chunks, with
  ordered depth arrows inside every frequency column and a separate chunk-buffer reduction row.
- **Labels:** `ordered depth`, `frequency chunks`, `private sums`, `chunk-order reduce`.
- **Color:** one chunk highlighted beige, all others slate/blue.

#### Ch2-F04 — The acceleration ladder

- **Placement:** transition from scalar to backends.
- **Message:** exact implementations preserve the physical parabolic integral while retaining
  distinct production layouts and workloads.
- **Composition:** four sparse stepping stones with small shape tags.
- **Labels:** `NumPy (D)`, `njit (D)`, `chunk prange`, `Torch (W,D)`.
- **Arrow:** one continuous navy hand-drawn path.
- **Avoid:** speed claims in the figure; timing comes later.

#### Ch2-F05 — What each Numba option changes

- **Placement:** Numba section.
- **Message:** compilation, GIL release, and parallelism are distinct.
- **Composition:** three small switches/cards.
- **Labels:** `njit: compile`, `nogil: release`, `prange: chunks`.
- **Caption:** `cache=True` is a storage/reload policy, shown as a small archive icon.

#### Ch2-F06 — CPU atmosphere, device synthesis

- **Placement:** Torch and architecture section.
- **Message:** ordered atmosphere passes stay on multicore CPU; wavelengths run in parallel in synthesis.
- **Composition:** reuse official CPU/GPU acceleration asset.
- **Labels:** preserve `ordered passes` and `wavelengths in parallel`.
- **Accuracy gate:** do not add a GPU arrow to the atmosphere loop.

#### Ch2-F07 — Host and device boundary

- **Placement:** device-transfer section.
- **Message:** upload invariants and star state deliberately; return completed spectra once.
- **Composition:** host page/data at left, device grid at center, spectrum at right; one incoming and one outgoing broad arrow.
- **Labels:** `host`, `device`, `one return`.
- **Accuracy gate:** no repeated ping-pong arrows.

#### Ch2-F08 — Cold and warm timing

- **Placement:** benchmarking.
- **Message:** compilation/setup and steady execution are different quantities.
- **Composition:** two stopwatches; cold one includes a small compile gear, warm one only a kernel arrow.
- **Labels:** `cold`, `warm`, `synchronize`.
- **Plot companion:** code-generated stacked timing bars in matching colors.

#### Ch2-F09 — Parity is a gate, not a feeling

- **Placement:** parity section.
- **Message:** compare value, tolerance, worst index, dtype, and device.
- **Composition:** scalar reference sheet and candidate grid entering a five-slot inspection frame.
- **Labels:** `value`, `tolerance`, `worst index`, `dtype`, `device`.
- **Avoid:** a single green check with no conditions.

#### Ch2-F10 — Abundance notation ladder

- **Placement:** abundance derivation.
- **Message:** physical number ratios, logarithmic notations, the direct solver vector, and the
  deck-decoded linear array are distinct representations.
- **Composition:** a three-card notation ladder above two separate output branches.
- **Labels:** `nX/nH`, `A(X)`, `[X/H]`, `97-slot dex`, `99 linear`.
- **Color:** beige for logarithmic labels; navy for the two distinct exact outputs.

#### Ch2-F11 — Element, isotope, ion, molecule

- **Placement:** species distinctions.
- **Message:** related physical objects are not interchangeable.
- **Composition:** hydrogen nucleus motif in four cards.
- **Labels:** `element H`, `isotope D`, `ion H II`, `molecule H₂`.
- **Accuracy gate:** H I means neutral hydrogen; H II means singly ionized.

#### Ch2-F12 — Four data roles

- **Placement:** data taxonomy.
- **Message:** source data, invariant tables, generated caches, and verification artifacts have different lifecycles.
- **Composition:** four separated desk objects.
- **Labels:** `source`, `invariant`, `cache`, `verify`.
- **Secondary labels:** under verify, tiny `fixture` and `golden`.
- **Arrows:** source → invariant/cache; no arrow from golden into physics.

#### Ch2-F13 — Manifest and checksum

- **Placement:** checksum section.
- **Message:** a manifest binds a semantic role to exact bytes.
- **Composition:** data sheet → fingerprint/hash → manifest checklist.
- **Labels:** `path`, `bytes`, `sha256`, `role`.
- **Accuracy gate:** caption says “identity, not correctness.”

#### Ch2-F14 — Exploded structured atmosphere

- **Placement:** schema introduction.
- **Message:** the interface groups depth structure, populations, composition, and invariant edges.
- **Composition:** central atmosphere cuboid with four exploded side trays.
- **Labels:** `thermodynamics`, `populations`, `abundances`, `edge grid`.
- **Whitespace:** large gaps between trays; detailed field names stay in the page table.

#### Ch2-F15 — Actual versus partition-normalized populations

- **Placement:** population distinction.
- **Message:** same shape does not mean same quantity.
- **Composition:** two equal-size cubes; left filled with `n`; right filled with `n/U`; a division-by-\(U\) card between them.
- **Labels:** `actual density`, `divide by U`, `partition-normalized`.
- **Warning:** a small muted `not bound levels` note.

#### Ch2-F16 — Fixed-\(n_e\) bridge

- **Placement:** bridge section.
- **Message:** preserve the atmosphere’s electron density while filling synthesis fields.
- **Composition:** atmosphere block with a locked `ne` column crossing a bridge into the schema cube.
- **Labels:** `supplied ne`, `hold fixed`, `fill populations`.
- **Accuracy gate:** no closure-loop arrow on the bridge path.

#### Ch2-F17 — Four validation claims

- **Placement:** chapter conclusion.
- **Message:** identity, representation, numerical agreement, and physical acceptance are different gates.
- **Composition:** four stepping stones with increasing scientific strength.
- **Labels:** `checksum`, `schema`, `parity`, `physical closure`.
- **Color:** first three slate/navy; final gate outlined and marked `later`, since Part I has not achieved it.

### 4.8 Plot style for code-generated figures

Code-generated plots should visually belong beside the hand-drawn schematics:

- white/`--paper` background;
- navy primary curve;
- slate secondary curve;
- warm beige comparison or approximation;
- pale slate grid at low contrast;
- no boxed legend when direct labels fit;
- top and right spines removed where scientifically harmless;
- axis labels use quantity plus unit;
- line width 2–2.5 px;
- sparse markers at physically meaningful reference points;
- captions explain the check and do not merely restate axes.

Use colorblind-safe line styles in addition to color. The deep navy and beige traces should also differ by solid/dashed or width.

### 4.9 Schematic production workflow and gates

For every schematic:

1. write a one-sentence scientific claim;
2. list no more than five in-figure labels;
3. select reusable motifs;
4. draft a monochrome thumbnail to verify reading order;
5. apply the official palette;
6. review physics conventions independently of aesthetics;
7. test at website render width and mobile width;
8. write alt text;
9. check that the caption carries any caveat removed from the drawing;
10. record the editable source or generation prompt.

Scientific review checklist:

- axes and arrows follow the book’s sign conventions;
- outermost layer is visually at top;
- wavelength increases left to right;
- opacity and photosphere relationships are correct;
- initialized and converged products are visibly distinct;
- CPU/GPU responsibilities match the actual architecture;
- no schema-valid object is depicted as physically converged without the closure gate;
- no figure uses color as its only semantic channel.

---

## 5. Global sequence and redundancy audit

### 5.1 Recommendation: retain the two-chapter Part I

The consolidation is coherent because each chapter has one governing question:

- Chapter 1: what physical structure can emit and transmit the observed light?
- Chapter 2: how do we preserve that structure’s meaning while making the calculation fast and portable?

The complete foundation sequence remains visible as internal acts, so coverage is retained without six separate reader-facing openings and recaps.

### 5.2 Recommendation: protect Chapter 1 with act-level checkpoints

Chapter 1 is the highest density risk in Part I. Its four acts should not be compressed further. The required safeguards are:

- an output or plot every one to two conceptual scenes;
- a short checkpoint after labels/assumptions, Planck radiation, and slab transfer;
- exact Payne Zero source excerpts for radiation and transfer, plus clearly chapter-local grey
  derivation cells;
- one culminating grey-atmosphere table rather than several competing capstones.

If a rendered density review shows that a prepared student cannot complete the chapter in two normal course meetings plus exercises, the permitted sixteen-chapter option should split after the Planck/transfer material. Content must not be deleted to preserve the count.

### 5.3 Recommendation: keep the grey atmosphere intentionally incomplete

Chapter 1 may motivate Rosseland opacity and hydrostatic balance, but it must not absorb:

- production Rosseland integration;
- EOS closure;
- realistic opacity sampling;
- line blanketing;
- convection;
- radiative acceleration;
- iterative temperature correction.

Those are later chapter owners. Chapter 1 therefore returns only explicitly named derivation
arrays and a closure-required warning; it does not define either a seed-product class or a solution
class.

### 5.4 Recommendation: unify Chapter 2 under “trustworthy contracts”

Acceleration, abundance notation, manifests, and schema can look unrelated if presented as a tool survey. The narrative must repeatedly return to one rule:

> A numerical value is usable only when its physical meaning, units, axes, precision, data identity, and validation claim are explicit.

This rule makes the chapter coherent and prepares every later chapter to state its own kernel/data contract.

### 5.5 Recommendation: compare the exact parabolic-integration realizations

The NumPy one-column integral, private compiled one-column integral, chunk-parallel CPU transfer,
and batched Torch synthesis integral form a coherent comparison because they contain:

- a familiar physical equation;
- exact, explicitly different depth/frequency/wavelength layouts;
- broadcasting;
- ordered recurrence;
- independent CPU frequency chunks suitable for the production `prange` loop;
- a natural Torch scan;
- monotonicity and analytic checks;
- realistic `D=80` and large-`W` scaling.

A purely elementwise Planck benchmark would not teach reduction order. A book-defined uniform
wrapper would hide the architecture that the chapter is meant to explain.

### 5.6 Recommendation: divide constant teaching by role and package

- Chapter 1 owns the unsuffixed exact synthesis constants used in its derivations.
- Chapter 2 shows the exact, nonuniform naming in both constant modules: unsuffixed exact plus
  `REFERENCE_*` in synthesis; `*_EXACT` plus `*_REFERENCE` in atmosphere; and distinct
  source-conversion names.

This prevents a compatibility discussion from interrupting the first Planck derivation and prevents Chapter 2 from rederiving radiation physics.

### 5.7 Recommendation: show the complete schema now, derive its physics later

The atmosphere–synthesis interface must be complete, so Chapter 2 shows every field, shape, and unit. It does not derive future fields.

The ownership rule is:

- Chapter 2: name, shape, unit, ordering, validation, and one-sentence semantic distinction;
- Chapters 3–8: physical derivation and canonical production;
- Chapters 10 and 15: integrated consumption and parity.

Synthetic schema examples must be labeled as representation tests.

### 5.8 Single-owner map for duplication-prone concepts

| Concept | Primary derivation | Later use |
| --- | --- | --- |
| \(T_{\rm eff}\), \(\log g\), atmosphere/synthesis distinction | Ch. 1 | Reuse without redefining |
| \(B_\nu\), \(B_\lambda\), Jacobian, limits | Ch. 1 | Import radiation kernels |
| \(S=j/\chi\), LTE \(S=B\), optical depth, column-mass sign | Ch. 1 | Reuse in full transfer |
| Grey law, moments, analytic hydrostatic seed | Ch. 1 | Compare with physical atmosphere |
| Axes, broadcasting, Numba, `prange`, device/dtype, timing | Ch. 2 | Short first-use reminders only |
| Abundance notation and exact representation paths | Ch. 2 | Consume the standard log pattern, direct `(97,)` dex vector, or deck `(99,)` linear array at the documented boundary |
| Data roles, manifests, checksum semantics | Ch. 2 | Apply to chapter assets |
| Schema-v4 inventory and actual versus normalized populations | Ch. 2 | Populate and consume |
| Partition functions, Boltzmann and Saha physics | Ch. 3 | Ch. 2 names \(n/U\) only |
| Molecular equilibrium | Ch. 4 | Ch. 2 names molecular fields only |
| Real continuum opacity and edge interpolation | Ch. 5 | Ch. 1 uses toy opacity |
| Full radiative transfer with scattering | Ch. 9 | Ch. 1 solves constant source only |
| Complete GPU synthesis | Ch. 10 | Ch. 2 teaches one device kernel |
| Production atmosphere closure | Ch. 11–13 | Ch. 1 seed remains approximate |
| Initialized versus converged workflow | Ch. 14–15 | Ch. 1 defines role; Ch. 2 validates metadata |

### 5.9 Minimal forward-reference policy

Chapter 1 has one closing signpost: the next chapter will make its arrays fast and explicit.

Chapter 2 has one closing signpost: Chapter 3 will fill the schema’s atomic population fields from statistical physics.

Within tables, a compact “physical owner” column may name later chapter numbers. Main prose should not repeatedly say “as we will see.”

---

## 6. Part I completion gate

Part I is ready for drafting only when:

- both chapters open with the stated physical question;
- Chapter 1 retains the Lecture 1 rhythm inside all four acts;
- no advanced term is treated as self-explanatory;
- every equation defines symbols, units, signs, and assumptions;
- every code cell has one conceptual purpose;
- every array is introduced with axes, shape, units, dtype, and device;
- the Planck kernel passes coordinate, limit, and broad-band checks;
- the slab passes zero/thin/thick checks;
- the 80-layer grey arrays carry an explicit prose/visual nonconvergence warning and are not
  serialized as a schema-v4 atmosphere product;
- Chapter 2 compares the exact NumPy, private compiled, chunk-parallel CPU, and batched Torch
  realizations of the parabolic depth integral without normalizing their interfaces;
- `prange` is used only over the exact independent contiguous frequency chunks;
- cold, warm, transfer, and synchronized device timing are separated;
- exact and parity-pinned constants are visibly distinct;
- abundance labels cannot be confused with the direct `(97,)` dex vector or deck-decoded `(99,)`
  linear array;
- source, invariant, cache, fixture, and golden roles remain separate;
- schema teaching keeps actual and partition-normalized populations in their exact separate fields
  and states that the validator does not prove their physical relationship;
- checksum identity, schema validity, numerical parity, and physical closure are taught as distinct claims;
- every schematic follows the official white/navy/slate/beige hand-drawn system and passes a physics review;
- the final chapter boundary says clearly that no physically converged atmosphere or verified spectrum exists yet.

The Part I output is a disciplined foundation: the reader can explain starlight, evaluate thermal radiation, calculate photon escape through a slab, build and inspect an 80-layer grey seed, accelerate a known recurrence across the actual hardware architecture, convert abundances explicitly, identify every data dependency, and validate the exact structured-atmosphere interface that the remaining thirteen chapters will fill with physical detail.
