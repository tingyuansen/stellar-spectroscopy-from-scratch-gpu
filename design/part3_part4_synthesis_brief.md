# Design Brief: Chapters 5–10 — Opacity, Transfer, and Structured-Atmosphere Synthesis

> **Integration note.** `design/global_chapter_contracts.md` governs the final
> reader-facing sequence and supersedes this brief's draft exercise section.
> Useful variations belong in the main causal narrative; all textbook
> schematics are original compositions. For Chapter 5 specifically,
> `design/chapter05_exact_source_contract.md` is the reconciled authority for
> the atmosphere product, synthesis product, sampled diagnostic, and sampled
> extension lanes. This older part-wide brief must not be used to blend those
> four implementations into one continuum route.

## Purpose and source boundary

This brief turns the lossless opacity/transfer/synthesis unit map into the reader-facing Chapters
5–10 construction contract. It covers the first complete spectrum calculation in the book:
continuous opacity, every supported line-opacity
branch, radiative transfer, the synthesis device architecture, and composition from a schema-v4
structured atmosphere.

The scientific and numerical source of truth is the pinned Payne Zero atmosphere and synthesis tree at commit
`9c44001feae40b85146630499e6f8a5fed42e5af`. That tree is read only. The textbook implementation,
teaching subsets, fixtures, and golden outputs belong in this repository and must carry source hashes.

The reader entering Chapter 5 knows only Chapters 1–4. In particular, the reader already knows:

- from Chapter 1: the atmosphere/synthesis distinction; wavelength, frequency, `B_nu`, `H_nu`,
  `F_nu`, and `F_lambda`; absorption, scattering, mass opacity, source function, column mass, optical
  depth, LTE, angular moments, and a first grey atmosphere;
- from Chapter 2: scalar Python, NumPy, serial `njit`, justified `prange`, Torch tensors, named axes,
  devices, dtype policy, cold/warm timing, abundance notation, data roles, manifests, schema-v4 depth
  ordering, units, and actual versus partition-normalized populations;
- from Chapter 3: atomic levels, Saha–Boltzmann populations, electron closure, the fixed-electron-
  density synthesis bridge, mass density, perturber density, microturbulence, and fractional Doppler
  width;
- from Chapter 4: the distinct 170-record atmosphere and 190-record synthesis
  molecular routes, consumer-specific H2 policies, molecular populations, and
  the complete schema-v4 synthesis handoff.

The reader does **not** yet know continuum cross sections, oscillator strengths, line profiles,
catalog routing, sparse deposits, operator-based scattering transfer, or how a broad spectrum is
organized on a GPU. Every one of those ideas must therefore begin with the physical question, a
limiting case, fully defined mathematics, and a small checked implementation.

## Part-wide contracts

### Axis and state notation

Upper-case symbols in this table are mathematical axis labels used in prose and diagrams. They are
not source identifiers and must never be substituted for the pinned argument or field names.

| Symbol | Meaning | Typical size |
| --- | --- | ---: |
| `D` | atmosphere depth, outermost to innermost | 80 |
| `W` | internal synthesis wavelength samples, including context | window dependent |
| `W_req` | exact requested/public wavelength samples after `output_slice` | `W - 32` without an operator |
| `F` | sampled continuum frequencies | at most three per used edge interval |
| `L` | atomic or molecular catalog lines in one chunk | window/chunk dependent |
| `P` | surviving depth–line pairs | chunk dependent |
| `T` | fixed transfer optical-depth samples | 51 |

Use `n_depth`, `n_wl`, `n_synthesis_wl`, `n_wavelengths`, `n_frequencies`, and `n_lines` when code
needs a dimension name. The mathematical symbol \(R_{\rm grid}\) denotes the intrinsic logarithmic
sampling density; the exact pinned constructor/API argument is `resolution`. `Grid` stores
`start_wavelength_nm`, `end_wavelength_nm`, and `resolution`.

The principal runtime arrays retain the exact pinned identifiers:

| Exact pinned identifier | Shape / axis order | Units | Residence and dtype |
| --- | ---: | --- | --- |
| `synthesis_wavelength_nm` | `(W,)` | nm | host grid identity; tensor copies are made on the device |
| `wavelength_nm` | `(W_req,)` | nm | exact requested/public host grid |
| `continuum_absorption` | `(D, W)` | cm² g⁻¹ | device `work_dtype` |
| `continuum_scattering` | `(D, W)` | cm² g⁻¹ | device `work_dtype` |
| `line_mass_absorption_coefficient` | `(D, W)` | cm² g⁻¹ | device `ACCUMULATION_DTYPE` |
| `line_source` | `(D, W)` | spectral radiance per Hz | device `work_dtype` |
| `planck_source` | `(D, W)` | spectral radiance per Hz | device `work_dtype` |
| `extinction` passed to `integrate_optical_depth` | `(W, D)` | cm² g⁻¹ | device |
| `thermal_source_grid`, `scattering_fraction_grid` | `(W, T)` | source units, dimensionless | device float32 in the iterative solve |
| `eddington_flux_total_per_frequency` | `(W,)` | Eddington flux per Hz | device until result construction |
| `eddington_flux_continuum_per_frequency` | `(W,)` | Eddington flux per Hz | device until result construction |
| `normalized_flux` | `(W_req,)` after crop | dimensionless | public host float64 |
| public `flux_total`, `flux_continuum` | `(W_req,)` | surface flux per nm | public host float64 |

`resolve_runtime(requested_device=None, requested_dtype=None)` chooses `REFERENCE_DTYPE`
(`torch.float64`) on CPU/CUDA and `DEFAULT_DTYPE` (`torch.float32`) on MPS. The shared line
accumulator is always `ACCUMULATION_DTYPE` (`torch.float32`). Do not call the resolved dtype
`DEFAULT_DTYPE`: those names have different exact meanings in the source. Integer gather/scatter
indices are int64 unless a source-format field is explicitly stored more narrowly.

### Notation and exact-API audit

The following table is a drafting gate. A displayed implementation may use a mathematical symbol in
an equation, but every callable, keyword, dataclass field, archive key, and returned name must match
this ledger exactly. A pedagogical scalar derivation that has no pinned callable is labeled
“inline derivation/check cell (no new API symbol)” rather than given a cleaned-up function name.

| Concept | Mathematical notation | Exact pinned name(s) | Shape, default, or branch contract |
| --- | --- | --- | --- |
| intrinsic wavelength sampling | \(R_{\rm grid}\) | `resolution` | public default `20000.0`; `Grid(start_wavelength_nm, end_wavelength_nm, resolution)` |
| public wavelength bounds | \(\lambda_{\min},\lambda_{\max}\) | `wavelength_start_nm`, `wavelength_end_nm` | public defaults `400.0`, `900.0`; internal pipeline names are `wl_start_nm`, `wl_end_nm` |
| continuum frequency samples | \(\nu_f\) | `frequencies_hz` | `(F,)`; singular scalar helpers use `frequency_hz` |
| absorption/scattering mass opacity | \(\kappa_\nu^{\rm abs},\sigma_\nu\) | `continuum_absorption`, `continuum_scattering` | `(D,W)` before transfer transpose |
| line mass opacity/source | \(\kappa_\nu^{\rm line},S_\nu^{\rm line}\) | `line_mass_absorption_coefficient`, `line_source` | `(D,W)`; opacity accumulator is float32 |
| Planck source | \(B_\nu(T)\) | `planck_source`; constructor `planck_bnu(wavelength_nm, temperature)` | `(D,W)` |
| transfer inputs | \(\chi_\nu,\tau_\nu\) | `extinction`, `column_mass`, `surface_tau` | `extinction` `(W,D)`, `column_mass` `(D,)`, `surface_tau` `(W,)` |
| transfer-grid state | \(S_\nu,\epsilon_\nu\) | `thermal_source_grid`, `scattering_fraction_grid` | `(W,51)`; `DEFAULT_SWEEPS = 8` |
| internal flux | \(H_\nu\) | `eddington_flux_total_per_frequency`, `eddington_flux_continuum_per_frequency` | `(W_req,)` fields of `SpectrumResult` after device-side crop |
| public flux | \(F_\lambda=4\pi H_\nu c_{\rm nm\,s^{-1}}/\lambda_{\rm nm}^2\) | `flux_total`, `flux_continuum` | `(W_req,)`, per nm, fields of `Spectrum` |
| optional slabs | — | `continuum_absorption`, `continuum_scattering`, `line_mass_absorption_coefficient`, `line_source` | `(D,W_req)` `SpectrumResult` fields only when `SynthesisPipeline.run(keep_slabs=True, spectral_operator=None)` retains them; `keep_slabs=False` is the default |
| atomic route code | — | `line_type` | `0` ordinary; `1` AUT; `2` COR parsed but unwired; `3` PRD routed through ordinary LTE metal; `-1` H I; `-2` D I; `-3` He I; `-4` He-3 I; `-6` He II |
| context width | — | `WINDOW_CONTEXT_SAMPLES` | `16` samples on each side |
| window counts | — | `WindowInvariants.n_wl`, `WindowInvariants.n_synthesis_wl`; internal `SynthesisPipeline.n_wl` | invariant fields mean requested and context-inclusive counts respectively; `SynthesisPipeline.n_wl` is assigned `n_synthesis_wl` for the forward pass |
| live line cutoff | — | `LINE_CENTER_CUTOFF_RATIO` | `1e-3` times local `continuum_opacity` |
| metal chunk | — | `SynthesisPipeline.METAL_CHUNK`; `metal_chunk` | default `40000`; `metal_chunk=None` selects it |
| molecular chunks | — | `CHUNK_LINES`, `PAIR_CHUNK`, `chunk_lines` | `500000`, `200000`, and `None`; environment override is `PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES` |
| molecular feature switch | — | `molecular_lines` | public default `True`; text bands and TiO are standard, H₂O is compiler-only |
| standard molecular compiler calls | — | `use_energy_level_wavelengths`, `include_predicted_lines`, `use_vacuum_wavelengths` | pipeline passes `use_energy_level_wavelengths=True` for text, leaves `include_predicted_lines=False`, and passes `use_vacuum_wavelengths=True` for TiO; it does not call `compile_h2o_partridge` |
| stimulated-emission switch | \(1-e^{-h\nu/kT}\) | `apply_stim` | default `True` in atomic, hydrogen, and molecular accumulators |
| atomic branch switches | — | `do_metal`, `do_helium`, `wing_mode` | defaults `True`, `True`, and `'batched'` in `accumulate_atomic`; `do_metal=True` runs both `_accumulate_metal` and `_accumulate_autoionizing`, while `do_helium=True` runs `_accumulate_helium` |
| transfer controls | — | `FLOAT32_POSITIVE_FLOOR`, `DEFAULT_SWEEPS`, `MAX_ITER`, `ITER_TOL` | `1e-38`, `8`, `51`, and `1e-5`; the fixed scattering solve uses `DEFAULT_SWEEPS`, while `MAX_ITER`/`ITER_TOL` belong to `_saturated_core_flux` |
| runtime choice | — | `device`, `dtype`; internal `runtime_device`, `work_dtype` | omitted dtype resolves to float64 on CPU/CUDA and float32 on MPS; requesting float64 on MPS raises `ValueError` |
| default device | — | `DEVICE`, `device()` | CUDA if available, otherwise MPS if available, otherwise CPU |

The schema-v4 archive keys are likewise fixed:

Here `D = n_depth >= 2`; every one-dimensional depth field has the same outer-to-inner length.

| Exact archive key | Shape / axis order | Units or role |
| --- | ---: | --- |
| `atmosphere_schema_version` | `(1,)` integer | schema `4` when written by `save_structured_atmosphere(...)` |
| `temperature` | `(D,)` | K |
| `gas_pressure` | `(D,)` | dyne cm⁻² |
| `electron_density` | `(D,)` | cm⁻³ |
| `mass_density` | `(D,)` | g cm⁻³ |
| `column_mass` | `(D,)` | g cm⁻² |
| `partition_normalized_populations` | `(D,6,139)` | cm⁻³ per partition function |
| `ion_stage_populations` | `(D,6,139)` | cm⁻³ |
| `fractional_doppler_widths` | `(D,6,139)` | dimensionless \(v/c\) |
| `hydrogen_neutral_population`, `helium_neutral_population`, `helium_singly_ionized_population`, `molecular_hydrogen_population` | `(D,)` each | cm⁻³ |
| `hydrogen_partition_normalized_ion_stage_populations`, `carbon_partition_normalized_ion_stage_populations` | `(D,2)` each | cm⁻³ per partition function |
| `magnesium_neutral_partition_normalized_population`, `aluminum_neutral_partition_normalized_population`, `silicon_neutral_partition_normalized_population`, `iron_neutral_partition_normalized_population` | `(D,)` each | cm⁻³ per partition function |
| `hc_over_kt` | `(D,)` | cm |
| `microturbulence` | `(D,)` | cm s⁻¹ |
| `elemental_abundances` | `(n_element,)`, `n_element >= 99` in schema validation | relative number fractions; the public column builder documents exactly `(99,)` for atomic numbers 1–99 |
| `signed_continuum_edge_frequency_hz`, `continuum_edge_wavelength_nm` | `(n_edge,)` each, `n_edge >= 2` | signed Hz; nm |
| `continuum_edge_midpoint_wavelength_nm`, `continuum_edge_interval_width_squared_over_two_nm2` | `(n_edge-1,)` each | nm; nm² |

An initialized-atmosphere product may additionally carry the exact all-or-none metadata fields
`atmosphere_product_metadata_schema`, `atmosphere_product_role`, `atmosphere_converged`,
`atmosphere_closure_required`, `initializer_family`, and `atmosphere_metadata_json`. These are
validated product metadata, not replacements for the physical arrays.

`build_pops(atmosphere, device=None, dtype=None)` returns a dictionary, not a replacement dataclass.
Its exact keys are:

| Exact `build_pops` key | Shape |
| --- | ---: |
| `temperature`, `mass_density`, `electron_density` | `(D,)` each |
| `hydrogen_partition_normalized_ion_stage_populations` | `(D,2)` |
| `hydrogen_neutral_population`, `hydrogen_ionized_population` | `(D,)` each |
| `helium_neutral_partition_normalized_population`, `helium_singly_ionized_partition_normalized_population`, `helium_doubly_ionized_partition_normalized_population` | `(D,)` each |
| `helium_neutral_population`, `helium_singly_ionized_population` | `(D,)` each |
| `carbon_partition_normalized_ion_stage_populations` | `(D,2)` |
| `magnesium_neutral_partition_normalized_population`, `aluminum_neutral_partition_normalized_population`, `silicon_neutral_partition_normalized_population`, `iron_neutral_partition_normalized_population` | `(D,)` each |
| `nitrogen_neutral_partition_normalized_population`, `oxygen_neutral_partition_normalized_population`, `magnesium_singly_ionized_partition_normalized_population`, `silicon_singly_ionized_partition_normalized_population`, `calcium_singly_ionized_partition_normalized_population` | `(D,)` each |
| `hot_metal_populations` | `(D,21)` |
| `charge_square_population_sum` | `(D,5)` |
| `_temperature_host`, `_natural_log_temperature_host` | `(D,)` host float64 |

The three line accumulators receive exact dictionary keys rather than chapter-specific state
dataclasses:

| Mapping | Exact keys and shapes |
| --- | --- |
| atomic `state` for `accumulate_atomic` | `partition_normalized_populations (D,6,139)`, `fractional_doppler_widths (D,6,139)`, `mass_density (D,)`, `electron_density (D,)`, `temperature (D,)`, `hc_over_kt (D,)`, `collision_density_proxy (D,)`, `continuum_opacity (D,W)`, `helium_core_weight_grid=None`, `helium_tail_weight_grid=None` in the standard pipeline |
| hydrogen `state` for `accumulate_hydrogen` | `temperature (D,)`, `electron_density (D,)`, `mass_density (D,)`, `hc_over_kt (D,)`, `helium_neutral_population (D,)`, `molecular_hydrogen_population (D,)`, `hydrogen_partition_normalized_ion_stage_populations (D,2)`, `microturbulence (D,)`, `hydrogen_neutral_partition_normalized_population (D,)`, `hydrogen_fractional_doppler_width (D,)`, `continuum_opacity (D,W)` |
| molecular `state` for `accumulate_molecular` | `partition_normalized_populations (D,6,139)`, `mass_density (D,)`, `electron_density (D,)`, `temperature (D,)`, `hc_over_kt (D,)`, `microturbulence (D,)`, `collision_density_proxy (D,)`, `continuum_opacity (D,W)` |

The exact public structured-synthesis surface is:

| Callable/dataclass | Exact signature or fields |
| --- | --- |
| `build_structured_atmosphere` | keyword-only `temperature`, `column_mass`, `gas_pressure`, `electron_density`, `elemental_abundances`, `mean_nuclear_mass_amu=None`, `microturbulence=None`, `mass_density=None`, `molecular_lines=True`, `device=None`, `dtype=None`, `eos_tolerance=1e-5`; returns the schema-v4 `dict[str, np.ndarray]` |
| `save_structured_atmosphere` | `(atmosphere, path)`; returns the validated field-name `tuple[str, ...]` |
| `synthesize` | `(atmosphere_npz, *, wavelength_start_nm=400.0, wavelength_end_nm=900.0, resolution=20000.0, molecular_lines=True, device=None, dtype=None, spectral_operator=None)`; returns `Spectrum` |
| `Spectrum` | `wavelength_nm`, `flux_total`, `flux_continuum`, `normalized_flux`, `seconds` |
| `SpectrumResult` | `wavelength_nm`, `eddington_flux_total_per_frequency`, `eddington_flux_continuum_per_frequency`, `normalized_flux`, `continuum_absorption`, `continuum_scattering`, `line_mass_absorption_coefficient`, `line_source`, `spectral_operator_seconds=0.0`, `spectral_operator_name=None` |

Static source-file fields must also remain literal:

| Source product | Exact stored fields and shapes |
| --- | --- |
| `continuum_edge_grid.npz` | float64 `signed_continuum_edge_frequency_hz (341,)`, `continuum_edge_wavelength_nm (341,)`, `continuum_edge_wavenumber_cm (341,)`, `continuum_edge_sample_frequency_hz (1020,)`, `continuum_edge_midpoint_wavelength_nm (340,)`, `continuum_edge_interval_width_squared_over_two_nm2 (340,)`; uint8 `__meta__ (236,)` |
| `continuum_tables.npz`, H/H⁻ and Rayleigh | float64 `karzas_latter_log10_frequency_hz (29,15)`, `karzas_latter_total_log10_cross_section_cm2 (29,15)`, `karzas_latter_angular_log10_cross_section_cm2 (6,6,29)`, `karzas_latter_high_level_energy_offset_rydberg (29,)`, `hminus_boundfree_wavelength_nm (85,)`, `hminus_boundfree_cross_section_cm2 (85,)`, `hminus_freefree_inverse_wavelength_grid (22,)`, `hminus_freefree_theta_grid (11,)`, `hminus_freefree_short_wavelength_table (11,11)`, `hminus_freefree_long_wavelength_table (11,11)`, `hydrogen_rayleigh_gavrila_main_table (74,)`, `hydrogen_rayleigh_gavrila_ab_table (27,)`, `hydrogen_rayleigh_gavrila_bc_table (24,)`, `hydrogen_rayleigh_gavrila_cd_table (22,)`, `hydrogen_rayleigh_gavrila_lyman_continuum_table (64,)`, `hydrogen_rayleigh_gavrila_lyman_frequency_ratio_grid (64,)`, `hydrogen_neutral_level_energy_cm (6,)`, and `hydrogen_neutral_level_statistical_weight (6,)` |
| `continuum_tables.npz`, free-free/metals | float64 `coulomb_freefree_charge_log_offset (6,)`, `coulomb_freefree_gaunt_table (12,11)`, `hot_metal_boundfree_transition_table (60,7)`, `silicon_singly_ionized_peach_cross_section_table (14,6)`, `silicon_singly_ionized_peach_threshold_frequencies_hz (7,)`, `silicon_singly_ionized_peach_natural_log_frequency_grid (9,)`, and `silicon_singly_ionized_peach_natural_log_temperature_grid (6,)` |
| `line_profile_tables.npz` | float64 `hydrogen_continuum_edges (15,)`, `radiative_damping_sums (96,)`, `impact_electron_density_thresholds_cm3 (2,2)`, `stark_knm_table (4,3)`, `stark_probability_table (7,5,15)`, `stark_wing_correction_c (5,7)`, `stark_wing_correction_d (5,7)`, `stark_pressure_grid (5,)`, `stark_beta_grid (15,)`, and `harris_profile_h0_table`, `harris_profile_h1_table`, `harris_profile_h2_table` `(2001,)` each |
| `transfer_tables.npz` | float64 `transfer_optical_depth_grid (51,)`, `mean_intensity_operator (51,51)`, `reference_column_mass (80,)`, `surface_eddington_flux_weights (51,)`; uint8 `__meta__ (208,)`. `mean_intensity_diagonal (51,)` is derived at load time and is not a stored source field. |
| `atomic_masses.npz` | float64 `atomic_mass_amu (99,)` |
| `ionization_potential_lookup.npz` | float64 `ionization_potential_cm (999,)` |
| `atomic_source_lines_parsed.npz` | per-record float64 `stored_wavelength_nm`, `raw_log_oscillator_strength`, `species_code`, `first_energy_column_cm`, `second_energy_column_cm`, `radiative_damping_log`, `stark_damping_log`, `van_der_waals_damping_log`, `primary_isotope_log_correction`, `secondary_isotope_log_correction`, `isotope_shift_units`; int64 `lower_principal_quantum_number`, `upper_principal_quantum_number`, `primary_isotope_number`, `line_size`; byte-string `energy_shift_field` and `line_category_tag` |
| molecular `manifest.json` | top-level `description`, `format`, `packed_binary_sources`, `text_sources`; packed-source records use `file`, `provenance`; text-source records use `provenance`, `line_count`, `band` |
| `molecular_band_lines.npz` | under each exact manifest band prefix: float64 `stored_wavelength_nm`, `log_oscillator_strength`, `first_energy_cm`, `second_energy_cm`, `radiative_damping_log_scaled`; int64 `source_code`, `isotope_index`; bool `upper_label_is_ground_state` |
| `titanium_oxide_lines.npy` | packed record fields `wavelength_code` int32, `isotope_species_code` int16, `lower_energy_code` int16, `log_oscillator_strength_code` int16, `radiative_damping_code` int16, `stark_damping_code` int16, `van_der_waals_damping_code` int16 |
| `water_lines.npy` | packed record fields `wavelength_code` int32, `signed_lower_energy_code` int16, `signed_log_oscillator_strength_code` int16; compiler-only in the pinned standard runtime |
| molecular compiler return mapping | `center_index_1based` int32, `classical_line_strength` float32, `species_code` int16, `lower_excitation_cm` float32, `radiative_damping` float32, `stark_damping` float32, `van_der_waals_damping` float32, `margin_class` int16; all have shape `(n_compiled_lines,)` |

`SynthesisPipeline._compile_molecular` adds scalar `log_grid_ratio` and `grid_origin_index` to the
concatenated text-plus-TiO mapping. `margin_class` is an exact compiler-return field but not a
`MolecularLineCatalog` dataclass field; `build_catalog_from_arrays(arrays)` consumes the catalog
fields and derives `center_index`, `species_population_column`, `wavelength_nm`, and
`unique_species_codes`.

### One canonical implementation

Each chapter adds small functions to the progressive textbook package. A chapter displays and calls
those exact functions; it does not keep a notebook-only copy. Later chapters compose earlier
functions:

```text
Ch. 5 continuum ─┐
Ch. 6–8 lines ───┼─> Ch. 9 transfer ─> Ch. 10 device pipeline + spectrum
Ch. 3–4 state ───┘
```

Chapter 10 is therefore an orchestration chapter, not another implementation of continuum, profiles,
catalog parsing, or transfer.

### Four-regime fixture matrix

All stage and integration tests draw from four checksum-bound schema-v4 atmospheres. Exact labels,
mixtures, array hashes, and selected windows belong in the data manifest; prose uses physical regime
names rather than relying on filenames.

| Regime ID | Physical purpose | Branches it must stress |
| --- | --- | --- |
| `hot_dwarf` | hot, relatively ionized atmosphere | H I wings, He/He II when present, ion/electron continuum, weak H⁻ |
| `solar_dwarf` | warm high-gravity reference | H⁻ continuum, ordinary neutral/ionized metal forest, stable normalization |
| `low_gravity_giant` | lower pressure and density | pressure-broadening response, CNO-sensitive molecules, transfer depth mapping |
| `cool_molecule_rich` | cool high-opacity regime | coupled molecular populations, text molecular bands, TiO, crowded line deposits |

Every chapter must state which regimes exercise an active branch and which provide a physically
correct near-zero check. A near-zero branch is evidence only when the input population or threshold
predicts it; it is not silently omitted from the test.

Teaching windows are chosen by querying the pinned catalogs and recording the selected record counts
and branch codes. At minimum the subset collection must contain:

- a Balmer window with resolved and merged-series behavior;
- a helium or He II window if a supported record exists in the selected hot-star range;
- an ordinary metal forest with at least one neutral and one ionized species;
- a type-1 autoionizing record;
- a type-3 record demonstrating its LTE metal routing;
- text molecular-band records and TiO records;
- continuum-only samples on both sides of at least one important edge.

The full-data gate repeats the same tests against the optional checksum-verified catalog root.

### Verification policy

Notebook-visible checks are physical and compact. Strict comparisons live in tests and use goldens
only after the textbook computation has completed. Each golden records source commit, source-data
hashes, fixture hash, window, `resolution`, device, dtype, Torch version, and tolerance policy.

The chapter wave is not complete until it has:

1. analytic or dimensional checks;
2. scalar-to-optimized parity where both implementations exist;
3. component parity before sums hide errors;
4. stage-interface parity at every chapter boundary;
5. four-regime coverage;
6. CPU/CUDA/MPS comparisons where hardware is available;
7. cold-cache/warm-cache equality;
8. a neighbor and no-redundancy review.

### Honest implementation boundaries

These statements must appear explicitly and consistently:

- The standard synthesis pipeline compiles text molecular bands plus TiO. The H₂O compiler and source
  catalog exist, but H₂O is not wired into the pinned standard runtime.
- Type-3 `PRD` catalog records are routed through the ordinary LTE metal-line kernel. The synthesis
  package does not implement a partial-redistribution transfer solver.
- Type-2 `COR` records are decoded and retained by the catalog representation but are not included by
  the standard synthesis pipeline's metal, helium, or hydrogen masks.
- The hydrogen synthesis engine supports neutral H/D records with lower principal quantum number
  `n_lower >= 2`; it raises for Lyman records rather than silently approximating them.
- `FrequencyInvariants` and a fully batched sampled-continuum path exist, but the pinned
  `SynthesisPipeline.run` calls the edge-triplet `continuum(...)` path without those invariants.
- Host float64 work is intentional for parity-sensitive brackets, interpolation regimes, source
  decoding, and small per-depth state. “GPU synthesis” does not mean every scalar decision is moved
  to the device.
- Synthesis uses no `torch.compile` path. The book must not add or benchmark one as if it were part of
  the pinned calculation.
- The real Numba use in synthesis is the serial cached `njit` molecular text compiler. There is no
  manufactured `prange` synthesis kernel.
- The public synthesis call evaluates one atmosphere at a time. Wavelengths, depths, lines, and
  surviving pairs are the internal parallel axes; a public star-batch API is not implied.
- The optional spectral-operator interface is a boundary to downstream resampling/broadening. It is
  not part of the native structured-atmosphere spectrum taught here.

### Density decision: Chapter 7 should split if the course may contain sixteen chapters

Chapter 7 is the only combined chapter in this brief that is genuinely too dense for the book's
normal pacing. It contains two independent physical questions, two large source modules, two
different data/table contracts, and two separately meaningful verified outputs:

1. catalog decoding, route masks, context selection, sparse ordinary-metal deposits, chunks, and
   cache identity;
2. hydrogen fine structure/Stark/series merging, helium isotope/ion profiles, and autoionizing
   profiles.

The fifteen-chapter version below preserves all of this as Movements 7A and 7B with a mandatory
mid-chapter checkpoint. The preferred sixteen-chapter version splits after the verified ordinary
metal forest:

- **Chapter 7 — Atomic Catalogs and Sparse Line Forests**
- **Chapter 8 — Hydrogen, Helium, and Non-Voigt Profiles**

The present Chapters 8–15 then shift by one. Split automatically if the canonical draft exceeds
either 18 core code cells, 55 rendered pages at the site measure, or one 90-minute lecture plus
laboratory while still requiring both parity suites. Do not solve density by deleting a route,
profile family, boundary, or test.

### Schematic plan and visual language

Every schematic uses the official reader aesthetic rather than a dark dashboard or photorealistic
render:

- white or near-white paper (`--paper`);
- hand-sketched scientific objects with slightly irregular 1.5–2 px strokes;
- main ink in muted slate/navy (`--ink`, `--ink-soft`, `--accent-deep`);
- pale blue and warm beige fills (`--accent-soft`, `--ember-soft`);
- one warm accent only for thresholds, warnings, or photons;
- short labels in IBM Plex Sans, equations in the text font, and generous whitespace;
- no gradients, 3-D effects, dense legends, or decorative star fields.

Planned assets:

| Chapter | Schematic | Composition and short labels |
| ---: | --- | --- |
| 5 | **Continuum process garden** | A hand-drawn H atom, H⁻, electron, He atom, metal edge, and three outgoing waves. Labels: “bound–free”, “free–free”, “scatter”, “edge samples”. A beige strip below shows three sample points and a smooth interpolated curve. |
| 6 | **Anatomy of one line** | Energy-level pair at left, velocity bell and damped wing at center, one absorption notch at right. Labels: `gf`, “Doppler”, “collisions”, “Voigt”. |
| 7 | **Catalog funnel to sparse pixels** | A loose stack of record cards flows through “decode”, “route”, and “keep” sieves into a sparse depth–wavelength grid. A second panel shows an H atom in an electric microfield and a merging series. |
| 8 | **Molecule to forest** | A simple diatomic sketch, rotational ladder, manifest cards, then many fine lines deposited into a grid. A small status card reads “text ✓”, “TiO ✓”, “H₂O compiler only”. |
| 9 | **From layers to escaping flux** | Horizontal atmosphere layers, curved optical-depth ruler, arrows mapping to a 51-point column, and an outward flux arrow. Labels: `τν`, `Sν`, `J`, `H`, “8 sweeps”. |
| 10 | **Resident GPU synthesis map** | White-page pipeline with a beige “window invariants” tray and pale-blue “star state” tray feeding continuum, line, and transfer blocks drawn as tensor grids. One arrow crosses to “host arrays” at the end. |

The Part III opener uses the Chapters 5–8 objects on one wide white canvas; the Part IV opener connects
the Chapter 9 atmosphere layers to the Chapter 10 tensor grid. Each asset must remain legible at the
reader's 46-rem article width and have alt text that states the physical relation rather than colors
or decoration.

## Narrow public-symbol and internal-routine placement

The module-level coverage ledger is intentionally broad. The following narrower placement prevents
modules from being “covered” in several chapters without clear ownership.

| Source object | Primary chapter | Exact treatment |
| --- | ---: | --- |
| `continuum.ContinuumTables` and table-field decoding | 5A | define once; metal fields activate in Movement 5B |
| `continuum.build_pops`, `pops_from_population_state` | 5A | continuum input adapter and actual/normalized population audit |
| `_hminus_bf_scalar`, `_hminus_opacity`, `_hminus_opacity_grid`, `_hydrogen_opacity`, `_hydrogen_opacity_grid`, `_helium_opacity`, `_helium_opacity_grid`, and the light-absorption part of `_minor_terms` | 5A | physical derivation and component kernels |
| `_carbon_neutral_boundfree_opacity`, `_magnesium_neutral_boundfree_opacity`, `_silicon_neutral_boundfree_opacity`, `_aluminum_neutral_boundfree_opacity`, `_iron_neutral_boundfree_opacity`, `_hot_metal_and_silicon_singly_ionized_opacity`, `_light_element_opacity_grid`, and `_hot_metal_opacity_grid` | 5B | complete metal continuum |
| `_scattering_opacity`, `_scattering_opacity_grid`, and the scattering part of `_minor_terms` | 5B | Rayleigh/Thomson assembly |
| `build_edge_sample_frequencies`, `compute_sampled_continuum`, `continuum` | 5B | production edge sampling and interpolation |
| `FrequencyInvariants`, `build_frequency_invariants` | 5B boundary, 10A architecture note | implemented extension, verified separately, not presented as the standard pipeline call |
| `line_opacity.highp_dtype`, Harris interpolation/center profile, `fast_ex` | 6 | one-line mathematics and numerical primitives |
| `atomic_lines.Grid`, `LineCatalog`, `_build_records`, `_line_window_mask`, `_assemble_catalog`, `parse_catalog`, and `load_catalog` | 7A | atomic catalog ownership |
| `line_opacity.AtomicInvariants` ordinary-metal fields, grid indices, scatter helpers, metal wing walkers | 7A | many-line ordinary branch |
| `line_opacity._accumulate_metal` and the metal portion of `accumulate_atomic` | 7A | production metal forest |
| `line_opacity._accumulate_autoionizing`, `_accumulate_helium` and their invariant fields | 7B | special atomic profiles |
| `hydrogen_lines._hydrogen_level_energy_cm`, `_fine_structure`, `_stark_quasi_static_profile`, `_hydrogen_state`, `merge_wavenumber_by_depth`, `precompute_invariants`, and `accumulate_hydrogen` | 7B | dedicated H/D engine |
| `source_catalog_molecular_compiler._parse_text_line`, `_get_compiled_band_kernel`, `compile_molecular_text`, `compile_tio_schwenke`, and `compile_h2o_partridge` | 8 | molecular source compilation |
| `molecular_lines.MolecularLineCatalog`, `MolecularLineInvariants`, `build_catalog_from_arrays`, `precompute_invariants`, `species_population_doppler_ratio`, `_accumulate_chunk`, `_near_wing`, `_far_wing`, `_scatter_add_flat`, and `accumulate_molecular` | 8 | molecular line opacity |
| `radiative_transfer.planck_bnu`, parabolic interval coefficients, `integrate_optical_depth` | 9A | variable-depth absorption-only transfer |
| `TransferTables`, `source_and_alpha`, transfer-grid interpolation, scattering solve, saturated fallback, `solve_spectrum` | 9B | full scattering transfer and output flux semantics |
| `api._surface_flux_per_wavelength_nm` | 9B | derive and test `H_nu -> F_lambda`; public dataclass use waits until Ch. 10 |
| `device.resolve_runtime` policy as used by synthesis | 10A | architecture, not a second device tutorial |
| `pipeline._window_grid_contract`, `WindowInvariants`, invariant key/build/cache helpers | 10A | reusable window state |
| `pipeline.SynthesisPipeline.__init__` star-dependent state | 10A | device/host state map |
| `prewarm.prewarm` and artifact identity helpers | 10A | cache preparation and provenance |
| `pipeline.SynthesisPipeline.run` | 10B | compose already implemented stages |
| `atmosphere.load_atmosphere_npz`, metadata loader, validator | 10B | structured input boundary |
| `pipeline.build_structured_atmosphere_from_columns`, `compute_doppler_per_ion`, and `_standard_microturbulence_column` | 10B | in-memory structured builder; thermochemistry is reused from Ch. 3–4 |
| `synthesis.compute_mean_nuclear_mass_amu`, builder wrapper, `synthesize_structured_atmosphere` | 10B | engine boundary |
| `api.Spectrum`, `build_structured_atmosphere`, `save_structured_atmosphere`, `synthesize` | 10B | public structured-atmosphere workflow |
| `ForwardTimings`, `InitializedAtmosphere`, `LabelSpectrum`, label initialization/synthesis | 14–15 | not pulled into the structured-atmosphere capstone |
| `pipeline._apply_spectral_operator_in_wavelength_density` | appendix boundary | native-to-observed operator interface only; no fitter or instrument implementation |

Field-level ownership is equally strict:

| Source dataclass | Exact field placement |
| --- | --- |
| `equation_of_state.PopulationState` | Reused from Chapters 3–4 with exact fields `electron_density`, `total_nuclei_number_density`, `mass_density`, `partition_normalized_populations`, `ion_stage_populations`, `hydrogen_neutral_population`, `hydrogen_ionized_population`, `hydrogen_partition_normalized_ion_stage_populations`, `helium_neutral_population`, `helium_singly_ionized_population`, `carbon_partition_normalized_ion_stage_populations`, `magnesium_neutral_partition_normalized_population`, `aluminum_neutral_partition_normalized_population`, `silicon_neutral_partition_normalized_population`, `iron_neutral_partition_normalized_population`, `eos`, `molecular_populations`, and `molecular_equation_densities`. |
| `continuum.ContinuumTables` | Movement 5A introduces `arrays`, `device`, `dtype`, `hydrogen_neutral_level_energy_ev`, `hydrogen_neutral_level_statistical_weight`, `hminus_freefree_log_table`, `hminus_freefree_log_wavelength_grid`, and `hminus_freefree_temperature_count`; Movement 5B activates `coulomb_freefree_gaunt_table_device` and the metal/scattering keys within `arrays`. |
| `continuum.FrequencyInvariants` | Movement 5B documents `frequencies_hz`, `coulomb_table_energy_first`, `natural_log_frequency`, `hminus_freefree_rows`, `hminus_boundfree_cross_section_cm2`, `silicon_singly_ionized_peach_frequency_rows`, `silicon_singly_ionized_peach_natural_log_temperature_grid`, `rayleigh_factor`, `hydrogen_high_level_photoionization_cross_sections`, `hydrogen_low_level_photoionization_cross_sections`, `hydrogen_ground_level_photoionization_cross_section`, `hydrogen_tail_edge`, `neutral_helium_low_level_photoionization_cross_sections`, `neutral_helium_high_level_photoionization_cross_sections`, `nitrogen_edge_cross_sections`, `oxygen_911_cross_section`, `calcium_edge_cross_sections`, `magnesium_ionized_cross_section_rows`, `carbon_boundfree_cross_section_rows`, `carbon_freefree_prefactor`, `carbon_freefree_threshold`, `magnesium_boundfree_cross_section_rows`, `magnesium_freefree_prefactor`, `magnesium_freefree_threshold`, `silicon_boundfree_cross_section_rows`, `silicon_freefree_prefactor`, `silicon_freefree_threshold`, `aluminum_boundfree_cross_section`, `iron_boundfree_cross_section_rows`, and `_tensor_cache`. This object is not passed by the standard pipeline. |
| `atomic_lines.Grid` | Movement 7A owns `start_wavelength_nm`, `end_wavelength_nm`, and `resolution`. |
| `atomic_lines.LineCatalog` | Movement 7A owns `wavelength_nm`, `index_wavelength_nm`, `oscillator_strength`, `log_oscillator_strength`, `lower_excitation_cm`, `radiative_damping`, `stark_damping`, `van_der_waals_damping`, `raw_radiative_damping_log`, `raw_stark_damping_log`, `raw_van_der_waals_damping_log`, `ion_stage`, `atomic_number`, `species_code`, `line_size`, `line_type`, `lower_principal_quantum_number`, `upper_principal_quantum_number`, `classical_line_strength`, `type_segments`, `grid`, and `sort`. |
| `line_opacity.AtomicInvariants` | Movement 7A owns `wavelength_grid`, `n_wavelengths`, `grid_resolution`, `metal_catalog_index`, `metal_classical_strength`, `metal_lower_excitation_cm`, `metal_radiative_damping`, `metal_stark_damping`, `metal_van_der_waals_damping`, `metal_wavelength_nm`, `metal_population_ion_stage_index`, `metal_population_element_index`, `metal_center_index`, `metal_wing_index`, `metal_center_clamped`, `metal_wing_clamped`, `harris_profile_h0_table`, `harris_profile_h1_table`, `harris_profile_h2_table`, `exponential_integer_table`, and `exponential_fraction_table`. Movement 7B owns `auto_catalog_index`, `auto_oscillator_strength`, `auto_lower_excitation_cm`, `auto_radiative_damping`, `auto_stark_damping`, `auto_van_der_waals_damping`, `auto_wavelength_nm`, `auto_population_ion_stage_index`, `auto_population_element_index`, `auto_center_index`, `auto_center_clamped`, `helium_classical_strength`, `helium_lower_excitation_cm`, `helium_radiative_damping`, `helium_stark_damping`, `helium_van_der_waals_damping`, `helium_wavelength_nm`, `helium_population_ion_stage_index`, `helium_population_element_index`, `helium_center_index`, `helium_line_type`, and `helium_cutoff`. |
| `hydrogen_lines.HydrogenProfileTables` | Movement 7B owns `radiative_damping_sums`, `impact_electron_density_thresholds_cm3`, `stark_knm_table`, `stark_probability_table`, `stark_wing_correction_c`, `stark_wing_correction_d`, `stark_pressure_grid`, and `stark_beta_grid`. |
| `hydrogen_lines.HydrogenLine` | Movement 7B owns `n_lower`, `n_upper`, `line_wavelength_nm`, `center_index`, `classical_strength`, `lower_excitation_cm`, `simple`, `series_limit_wavenumber_cm`, `shifted_series_limit_wavelength_nm`, `component_frequency_offsets`, `component_weights`, `component_count`, `red_neighbor_wavelength_nm`, `far_red_neighbor_wavelength_nm`, `blue_neighbor_wavelength_nm`, `far_blue_neighbor_wavelength_nm`, `red_dominance_cutoff_nm`, and `blue_dominance_cutoff_nm`. |
| `hydrogen_lines.HydrogenMergedCont` | Movement 7B owns `n_lower`, `series_limit_wavelength_nm`, `merged_continuum_strength`, `lower_excitation_cm`, and `last_resolved_upper_level`. |
| `hydrogen_lines.HydrogenInvariants` | Movement 7B owns `wavelength_grid`, `wavelength_grid_host`, `wavelength_count`, `lines`, `merged`, `tables`, `merge_wavenumber_by_depth`, `component_map`, `exponential_integer_table`, and `exponential_fraction_table`. Movement 10A alone discusses why `merge_wavenumber_by_depth` is replaced per star while the rest is cached as a template. |
| `molecular_lines.MolecularLineCatalog` | Chapter 8 owns `center_index_1based`, `classical_line_strength`, `species_code`, `lower_excitation_cm`, `radiative_damping`, `stark_damping`, `van_der_waals_damping`, `center_index`, `species_population_column`, `wavelength_nm`, `log_grid_ratio`, `grid_origin_index`, and `unique_species_codes`. |
| `molecular_lines.MolecularLineInvariants` | Chapter 8 owns `wavelength_grid`, `n_wavelengths`, `local_resolving_power`, `classical_line_strength`, `lower_excitation_cm`, `radiative_damping`, `stark_damping`, `van_der_waals_damping`, `center_index`, `line_species_index`, `line_wavelength_nm`, `species_code`, `species_population_column`, `species_mass_amu`, `harris_profile_h0_table`, `harris_profile_h1_table`, and `harris_profile_h2_table`. |
| `radiative_transfer.TransferTables` | Movement 9B owns `transfer_optical_depth_grid`, `mean_intensity_operator`, `mean_intensity_diagonal`, and `surface_eddington_flux_weights`. |
| `pipeline.WindowInvariants` | Movement 10A owns `key`, `device`, `dtype`, `molecular_lines`, `metal_chunk`, `grid_obj`, `synthesis_wavelength_nm`, `wavelength_nm`, `output_slice`, `n_synthesis_wl`, `n_wl`, `n_atomic`, `atomic_kernel_catalog`, `has_metal`, `has_helium`, `has_hydrogen`, `continuum_tables`, `transfer_tables`, `metal_invariant_chunks`, `helium_invariants`, `hydrogen_invariants_template`, `molecular_invariants`, `n_molecular`, and `build_profile`. |
| `pipeline.SpectrumResult` | Movement 10B owns `wavelength_nm`, `eddington_flux_total_per_frequency`, `eddington_flux_continuum_per_frequency`, `normalized_flux`, `continuum_absorption`, `continuum_scattering`, `line_mass_absorption_coefficient`, `line_source`, `spectral_operator_seconds`, and `spectral_operator_name`. The optional operator fields remain boundary metadata. |
| `api.Spectrum` | Movement 10B owns public `wavelength_nm`, `flux_total`, `flux_continuum`, `normalized_flux`, and `seconds`. Initializer metadata dataclasses remain in Chapters 14–15. |
| `api.ForwardTimings` | Deferred to Chapters 14–15 with exact fields `initializer_seconds`, `population_bridge_seconds`, `synthesis_seconds`, and `total_seconds`. |
| `api.InitializedAtmosphere` | Deferred to Chapters 14–15 with exact fields `structured_atmosphere`, `initializer_family`, `labels`, `provenance`, `timings`, `atmosphere_converged=False`, and `atmosphere_closure_required=True`. |
| `api.LabelSpectrum` | Deferred to Chapters 14–15; in addition to inherited `Spectrum` fields it owns `initializer_family`, `labels`, `provenance`, `timings`, `initialized_atmosphere`, `atmosphere_converged=False`, and `atmosphere_closure_required=True`. |

`_atomic_catalog_for_kernels(...)` preserves the `LineCatalog` names in a mapping and adds exact
profile-table keys. `_slice_atomic_catalog(...)` additionally supplies `helium_line_type` and
`helium_line_center_cutoff_ratio` for the helium invariant build; these are kernel-mapping keys, not
`LineCatalog` fields.

Source anchors for this placement include:

- continuum population roles and host temperature brackets:
  `/Users/ysting/payne-zero/payne_zero_synthesis/continuum.py:1692`;
- actual edge-triplet driver:
  `/Users/ysting/payne-zero/payne_zero_synthesis/continuum.py:5464` and
  `/Users/ysting/payne-zero/payne_zero_synthesis/continuum.py:5515`;
- atomic type routing:
  `/Users/ysting/payne-zero/payne_zero_synthesis/line_opacity.py:342`;
- hydrogen branch boundary:
  `/Users/ysting/payne-zero/payne_zero_synthesis/hydrogen_lines.py:1054`;
- molecular runtime composition:
  `/Users/ysting/payne-zero/payne_zero_synthesis/pipeline.py:1169`;
- transfer implementation:
  `/Users/ysting/payne-zero/payne_zero_synthesis/radiative_transfer.py:97` through
  `/Users/ysting/payne-zero/payne_zero_synthesis/radiative_transfer.py:756`;
- invariant construction and full run:
  `/Users/ysting/payne-zero/payne_zero_synthesis/pipeline.py:1554` and
  `/Users/ysting/payne-zero/payne_zero_synthesis/pipeline.py:1294`.

## Data placement across Chapters 5–10

| Data product | First owning chapter | Role |
| --- | ---: | --- |
| `synthesis_tables/continuum_tables.npz` | 5 | static cross sections, Gaunt factors, continuum interpolation tables |
| `synthesis_tables/continuum_edge_grid.npz` | 5 | static signed edges, midpoint data, edge sample definition |
| `synthesis_tables/ionization_potential_lookup.npz` | 7 | static lookup used while decoding/defaulting atomic records |
| `synthesis_tables/line_profile_tables.npz` | 6 | static Harris/FASTEX, H Stark, radiative-sum, and continuum-edge tables |
| `synthesis_tables/transfer_tables.npz` | 9 | static 51-point grid, lambda operator, diagonal, and surface weights |
| `synthesis_tables/atomic_masses.npz` | earlier Ch. 3; reused Ch. 8/10 | static species masses and Doppler construction |
| `synthesis_tables/partition_saha_inputs.npz` | earlier Ch. 3; reused Ch. 10 | fixed-ne population bridge |
| `source_catalogs/lines/atomic_source_lines_parsed.npz` | 7 | full static decoded atomic source |
| molecular manifest and `molecular_band_lines.npz` | 8 | manifest-ordered text-band sources |
| `titanium_oxide_lines.npy` | 8 | packed TiO source |
| `water_lines.npy` or manifest-resolved H₂O source | 8 | compiler test only unless runtime policy changes |
| four schema-v4 atmosphere fixtures | 5 onward | integration fixtures, never static inputs or goldens |
| per-component and final spectra from the pinned source | matching chapter | comparison-only goldens |

The normal book uses small checksum-bound catalog slices. Full catalog compilation and broad-spectrum
gates require the optional source root. Cache files are accelerations derived from static inputs; they
are neither fixtures nor golden outputs.

---

## Chapter 5 — Continuous Opacity and Scattering

### Chapter question

**Which smooth absorption and scattering processes set the background from which spectral lines
emerge, and how can sharp continuum edges be represented efficiently on the synthesis grid?**

The chapter has two movements but one final artifact: complete continuum absorption and scattering
slabs. Movement 5A establishes H/He/light-ion absorption; Movement 5B completes metals, scattering,
atmosphere-only molecular-continuum boundaries, and edge interpolation.

### Movement 5A question

**How can a spectrum have wavelength-dependent opacity even where no discrete spectral line is
centered?**

Open with a solar-like continuum whose brightness changes smoothly across wavelength, then remove
individual line records. The remaining opacity motivates bound-free and free-free processes.

### Prerequisites

- Ch. 1: photon energy, `B_nu`, absorption, mass opacity, source function, and LTE.
- Ch. 2: schema fields, units, arrays, device/dtype conventions, and data roles.
- Ch. 3: bound states, normalized populations, ionization, electron density, and mass density.
- Ch. 4: molecular populations as available state, including a stored H2
  field that neither continuum product consumes; no molecular-equilibrium
  derivation is repeated.

### Reads / writes contract

Reads:

- schema fields `temperature`, `mass_density`, and `electron_density`: `(D,)`;
- `hydrogen_neutral_population`, `hydrogen_partition_normalized_ion_stage_populations`,
  `helium_neutral_population`, and `helium_singly_ionized_population`;
- sampled `frequencies_hz`: `(F,)`;
- `ContinuumTables`, normally built with
  `ContinuumTables.from_npz(path=_DEFAULT_CONTINUUM_TABLES, device=device,
  dtype=work_dtype)`. The omitted CPU dtype is demonstrated separately because
  it resolves to float32 rather than the standard pipeline's float64.

Writes:

- exact `compute_sampled_continuum(...)` outputs `continuum_absorption`,
  `continuum_scattering`, and `continuum_source`, each `(D,F)`;
- inline component-budget diagnostics (no serialized field or new public API).

### Derivation and prose arc

1. **Absorption without a line.** Use a threshold cartoon: a bound electron absorbs any photon above
   a threshold, while a free electron can exchange energy in the Coulomb field of an ion.
2. **Cross section to mass opacity.** Derive
   `kappa_nu = n_absorber * sigma_nu / rho`, defining every factor and its units.
3. **Stimulated emission.** Derive the LTE correction
   `1 - exp(-h nu / kT)` from upward minus downward transitions. Check its low- and high-frequency
   limits before it enters any opacity function.
4. **H⁻ as a continuum absorber.** Explain why neutral hydrogen plus a spare electron is important in
   warm stellar photospheres. Separate bound-free and free-free terms; connect the H⁻ population
   factor to the Saha machinery already learned without rederiving the Saha equation.
5. **Neutral and ionized hydrogen.** Introduce thresholded bound-free sums over explicit levels,
   high-level tail treatment, Coulomb free-free Gaunt factors, and the difference between actual H II
   density and normalized H I populations.
6. **H₂⁺ and He⁻.** Introduce these as additional smooth absorption terms. Keep their polynomial or
   tabulated fits subordinate to the physical process.
7. **He I and He II.** Build neutral-helium threshold families, ionized-helium hydrogenic scaling,
   and free-free contributions. Show why hot and cool atmospheres activate different terms.
8. **A physical budget.** Plot each component and the ordered sum at selected depths for the hot and
   solar fixtures. A process must be visible by name before it disappears into the aggregate.

### Bite-size source-faithful code artifacts

Each item is one 10–30 line conceptual cell or a compact slice of an exact pinned object:

1. Inline stimulated-emission and number-cross-section-to-mass-opacity derivations/checks (no new API
   symbols).
2. `ContinuumTables.from_npz(path=_DEFAULT_CONTINUUM_TABLES, device=None, dtype=None)`.
3. `ContinuumTables.from_dict(arrays, device=None, dtype=None)`.
4. `build_pops(atmosphere, device=None, dtype=None)`.
5. `pops_from_population_state(population_state, temperature, mass_density, device=None, dtype=None)`.
6. `_hminus_bf_scalar(continuum_tables, frequency_hz)`.
7. `_hminus_opacity(...)`, followed by `_hminus_opacity_grid(...)`.
8. `_hydrogen_opacity(...)`, followed by `_hydrogen_opacity_grid(...)`.
9. The H₂⁺/He⁻ absorption portion of `_minor_terms(...)`.
10. `_helium_opacity(...)`, followed by `_helium_opacity_grid(...)`.
11. `compute_sampled_continuum(continuum_tables, frequencies_hz, pops,
    frequency_invariants=None)`.

Underscore-prefixed names are presented as pinned internal routines, not public promises. The exact
source anchors are `continuum.build_pops` at line 1692, `_hminus_opacity` at line 1936,
`_hydrogen_opacity` at line 2081, `_minor_terms` at line 3538, and `_helium_opacity` at line 4144.

### Data

- `continuum_tables.npz`, copied as static physical input with array-level metadata.
- A tiny threshold table extracted from the same bundle for plots.
- The hot and solar structured-atmosphere fixtures.
- Golden component arrays at a small set of frequencies, stored separately from input tables.

### Parity and physical gates

1. Units reduce to cm² g⁻¹.
2. Stimulated-emission factor lies in `[0, 1]`, tends to zero as `h nu / kT -> 0`, and tends to one
   at large `h nu / kT`.
3. Every absorption component is finite and non-negative.
4. A thresholded bound-free term is zero on its inactive side.
5. Doubling absorber population doubles its isolated opacity.
6. Dividing by twice the mass density halves number opacity per gram when the other state is fixed.
7. H⁻ dominates an explicitly chosen solar-continuum diagnostic point; the test records the point
   rather than making a universal wavelength claim.
8. The hot fixture weakens H⁻ and activates the expected ion/free-electron terms.
9. Scalar and device-grid helpers agree under a dtype-specific tolerance.
10. Named component arrays and their ordered sum match the pinned source for all four regimes; cool
    and giant tests include valid near-zero and low-temperature branches.

### Device and dtype behavior

- Sampled synthesis `(D, F)` arithmetic is Torch on the selected device.
- The atmosphere product remains direct NumPy/Numba CPU float64 work on its
  own 30,000-point grid.
- Table and threshold arrays begin as host float64 static data.
- H⁻ temperature-grid interpolation retains host float64 bracket behavior where the pinned scalar
  path does so; the chosen values return to the device.
- CUDA/CPU use float64 work precision; MPS uses float32.
- No line accumulator exists yet, so Chapter 5 must not pre-emptively cast continuum slabs to
  float32.
- The chapter may compare scalar and batched component helpers, but it must not claim that
  `FrequencyInvariants` is used by the standard pipeline.

### No-redundancy and forward-reference audit

- Do not rederive Saha, partition functions, H₂ equilibrium, or Planck radiation.
- Do not introduce Rayleigh/Thomson scattering or metal continua beyond a one-sentence boundary; they
  belong to Movement 5B.
- Do not introduce bound-bound oscillator strengths; those begin in Chapter 6.
- End with a precise output: named H/He/light-ion absorption components on sampled frequencies, not
  yet the complete continuum on the synthesis grid.

### Movement 5B — Metals, Molecular Continua, Scattering, and Interpolation

### Movement 5B question

**How do many smooth processes, sharp continuum edges, and scattering become two stable opacity slabs
on an arbitrary synthesis wavelength grid?**

Start with three spectra computed at increasingly dense wavelength grids. The physical edge remains
sharp, while blindly evaluating every continuum formula at every spectral pixel wastes work. This
sets up the edge-triplet method.

### Prerequisites

- All Movement 5A continuum concepts and component interfaces.
- Ch. 3–4 actual versus normalized population roles.
- Ch. 2 host/device boundaries and interpolation examples.

### Reads / writes contract

Reads:

- the exact dictionary returned by `build_pops(...)`;
- the schema fields `partition_normalized_populations` and `ion_stage_populations`, each
  `(D,6,139)`, plus the named neutral-metal population fields;
- `hydrogen_neutral_population`, `helium_neutral_population`, and
  `electron_density`. The stored schema
  `molecular_hydrogen_population` is carried by the broader state but is not a
  continuum-product input;
- `signed_continuum_edge_frequency_hz`, `continuum_edge_wavelength_nm`,
  `continuum_edge_midpoint_wavelength_nm`, and
  `continuum_edge_interval_width_squared_over_two_nm2`;
- requested `wavelength_grid_nm`: `(W,)`.

Writes:

- standard synthesis `continuum_absorption` and
  `continuum_scattering`: `(D,W)`;
- atmosphere `continuum_absorption`, `continuum_scattering`, and
  absorption-weighted `continuum_source`: `(D,30000)`;
- atmosphere line-reference threshold: `(D,344)` float32;
- optional diagnostic component budgets on `(D, F)`;
- inline edge-usage diagnostics with used indices and `sample_frequencies_hz` (no new API symbol).

### Derivation and prose arc

1. **Why metals matter without metal lines.** Introduce neutral-metal photoionization and ion
   free-free absorption. Explain that “metal continuum” is a family of processes, not a single
   constant.
2. **Named metal groups.** Treat C I, Mg I, Al I, Si I, Fe I, the N/O/Mg II/Ca II light-ion group,
   the Si II Peach table, and the hot-metal charge ladder. Each gets a one-paragraph physical purpose
   and a plotted regime where it matters.
3. **Population roles.** Demonstrate why bound-state terms read partition-normalized populations,
   while the free-free charge-squared sum reads actual ion-stage populations. The source constructs
   the latter from selected elements and charge states in `continuum.build_pops`.
4. **Molecular-continuum boundary.** Teach CH, OH, and H₂ collision-induced absorption once as
   physical continuum processes because the atmosphere opacity engine needs them. Label the
   implementation boundary accurately: the pinned synthesis `continuum(...)` composite does not add
   those three atmosphere-only terms. H₂⁺ absorption and H₂ Rayleigh scattering are separate and are
   present in synthesis.
5. **Absorption versus scattering.** Derive Thomson scattering from free electrons and Rayleigh
   scattering from induced dipoles of H I, He I, and H₂. Re-emphasize that scattering contributes to
   extinction but not directly to the LTE thermal numerator.
6. **An edge-aware grid.** Draw two neighboring continuum edges and select the slightly offset left
   sample, midpoint sample, and slightly offset right sample used by
   `build_edge_sample_frequencies`.
7. **Log-parabolic interpolation.** Derive the three Lagrange basis functions in wavelength and
   explain why interpolation is applied to `log10(kappa)` with a positive floor.
8. **Only used edges.** Compute `searchsorted` edge indices for the requested window, evaluate three
   samples only for unique used intervals, then deposit interpolated values onto `(D, W)`.
9. **Backend contrast.** The atmosphere calculation evaluates its own 30,000-point opacity-sampling
   grid and includes CH/OH/CIA; synthesis uses the requested-window edge-triplet path. Similar physics
   does not make the grids or composites interchangeable.

### Bite-size source-faithful code artifacts

1. Inline charge-squared ion-population and scattering-limit checks (no new API symbols).
2. `_carbon_neutral_boundfree_opacity(...)`, `_magnesium_neutral_boundfree_opacity(...)`,
   `_silicon_neutral_boundfree_opacity(...)`, `_aluminum_neutral_boundfree_opacity(...)`, and
   `_iron_neutral_boundfree_opacity(...)`, introduced in physically grouped slices.
3. `_hot_metal_and_silicon_singly_ionized_opacity(...)`, `_light_element_opacity_grid(...)`, and
   `_hot_metal_opacity_grid(...)`.
4. `_scattering_opacity(...)`, followed by `_scattering_opacity_grid(...)`.
5. `compute_molecular_continuum_opacity_columns(atmosphere, frequency_hz, *,
   continuum_tables=None, molecular_tables=None)` from the atmosphere package, explicitly labeled
   outside the standard synthesis `continuum(...)` composite.
6. `build_frequency_invariants(continuum_tables, frequencies_hz,
   coulomb_table_energy_first=False)` as a separately verified extension.
7. `build_edge_sample_frequencies(signed_edge_frequency_hz,
   continuum_edge_wavelength_nm)`.
8. `compute_sampled_continuum(continuum_tables, frequencies_hz, pops,
   frequency_invariants=None)`.
9. Inline three-point log-opacity interpolation derivation/check (no new API symbol).
10. `continuum(wavelength_grid_nm, atmosphere, continuum_tables, pops=None)`.

Source ownership is:

- synthesis metal/scattering/component assembly:
  `/Users/ysting/payne-zero/payne_zero_synthesis/continuum.py:2298`,
  `:2424`, `:2588`, `:2806`, `:2917`, `:3106`, `:3538`, `:4144`, `:4563`, and `:4711`;
- edge sampling and interpolation:
  `/Users/ysting/payne-zero/payne_zero_synthesis/continuum.py:5464` and `:5515`;
- atmosphere-only CH/OH/CIA boundary:
  `/Users/ysting/payne-zero/payne_zero_atmosphere/continuum_opacity.py:4961`,
  `:5035`, `:5103`, and `:5188`.

### Data

- `continuum_tables.npz`.
- `continuum_edge_grid.npz`, including 341 edges and the corresponding three-sample structure.
- Four structured-atmosphere fixtures.
- Tiny tabular excerpts for Si II Peach, Coulomb Gaunt, and Rayleigh diagnostic plots.
- Golden per-component sampled arrays and final absorption/scattering slabs.

### Parity and physical gates

1. Charge-squared sum uses actual populations and matches a hand calculation for one depth.
2. Bound-state metal opacity changes when normalized population changes but not when only the unused
   actual-population view is perturbed.
3. Thomson scattering is wavelength independent for fixed `n_e/rho`.
4. Rayleigh scattering has the expected strong blueward increase in an isolated test.
5. Absorption and scattering remain separate and non-negative.
6. All requested wavelengths map to exactly one edge interval.
7. The three interpolation basis functions sum to one.
8. Interpolation reproduces all three edge samples and remains positive after exponentiation.
9. Unused edge intervals trigger no physical opacity evaluation.
10. Textbook and pinned-source sampled components, interpolated absorption, and interpolated
    scattering agree in the four regimes.
11. An explicit feature test confirms CH/OH/H₂ CIA are present in the atmosphere composite and absent
    from the pinned synthesis composite; this is not treated as a parity failure.
12. `FrequencyInvariants` scalar/batched helper parity is tested separately, while a call-trace test
    confirms the standard pipeline does not pass it to `continuum(...)`.

### Device and dtype behavior

- Population and opacity grids are device tensors in work dtype.
- Edge lookup, unique interval selection, signed-frequency construction, and Si II temperature
  brackets remain host float64.
- Interpolation basis values are computed from host float64 wavelength metadata, then transferred as
  small device tensors.
- The loops over used edge intervals are bounded by the 340-interval table, not by millions of
  spectral pixels.
- `FrequencyInvariants` may materialize a full `(D, F)` batched path in tests, but it is described as
  an available helper rather than the standard `SynthesisPipeline` route.

### No-redundancy and forward-reference audit

- Do not repeat H/He derivations from Movement 5A; call the named component functions.
- Do not teach line wings as “continuum”; line opacity begins in Chapter 6.
- Do not teach Rosseland means here beyond a short distinction; their atmosphere role is taught later.
- Do not imply that CH/OH/CIA are synthesized by the standard runtime.
- The chapter ends with the two complete `(D, W)` continuum slabs required by line selection and
  transfer.

---

## Chapter 6 — One Spectral Line

### Governing question

**Why does one atomic transition remove light from a narrow wavelength interval, and what controls
the shape and area of that absorption feature?**

Begin with a single isolated line whose depth, width, and wings change one physical control at a time.

### Prerequisites

- Ch. 1 photon energy and LTE.
- Ch. 3 excitation populations, thermal/microturbulent Doppler width, and perturber densities.
- Ch. 5 stimulated emission, mass-opacity conversion, and complete continuum opacity for a
  line-strength cutoff.

### Reads / writes contract

Reads:

- a one-record slice of `LineCatalog`: `wavelength_nm`, `oscillator_strength`,
  `log_oscillator_strength`, `lower_excitation_cm`, `atomic_number`, `ion_stage`,
  `radiative_damping`, `stark_damping`, and `van_der_waals_damping`;
- exact state keys `partition_normalized_populations`, `fractional_doppler_widths`, `mass_density`,
  `electron_density`, `collision_density_proxy`, `temperature`, and `hc_over_kt`;
- `(W,)` exact `wavelength_grid_nm` and `(D,W)` `continuum_opacity`, where
  Chapter 5 hands forward the derived sum
  `continuum_absorption + continuum_scattering`.

Writes:

- one line's `line_mass_absorption_coefficient`: `(D,W)`, float32 accumulator;
- diagnostic `(D,)` amplitude, damping ratio, center opacity, and reach.

### Derivation and prose arc

1. **A bound-bound transition.** Define lower and upper energy levels and oscillator strength as a
   measure of transition probability.
2. **Integrated strength.** Connect oscillator strength, lower-level Boltzmann population, and
   stimulated emission to opacity per gram.
3. **Doppler broadening.** Combine thermal velocity and microturbulence already built in Chapter 3.
   Check the mass and temperature scalings.
4. **Damping.** Introduce radiative decay, electron Stark collisions, and neutral/perturber van der
   Waals collisions as additive rates before division by Doppler width.
5. **Voigt profile.** Build Gaussian-core and Lorentz-wing intuition, define normalized coordinate and
   damping ratio, and verify profile-area normalization on a sufficiently wide diagnostic grid.
6. **Harris and FASTEX.** Explain table interpolation and the fast exponential as numerical
   approximations after the exact mathematical profile is understood.
7. **One line through all depths.** Compute amplitude and damping for all layers at once, then deposit
   the profile onto the wavelength grid.
8. **A physically motivated cutoff.** Compare line amplitude to local continuum opacity. State that
   this is a numerical selection rule whose order and dtype affect discrete decisions.

### Bite-size source-faithful code artifacts

1. One-record `LineCatalog` slice; no replacement record dataclass.
2. Inline line-population, amplitude, damping-ratio, Gaussian, and Lorentz derivations/checks (no new
   API symbols).
3. `highp_dtype(runtime_device)`.
4. `interpolate_harris_profile(doppler_offset, damping_ratio, harris_profile_h0_table,
   harris_profile_h1_table, harris_profile_h2_table)`.
5. `harris_profile_at_line_center(damping_ratio, harris_profile_h0_table,
   harris_profile_h1_table, harris_profile_h2_table)`.
6. `fast_ex(exponent_argument, exponential_integer_table, exponential_fraction_table)`.
7. `precompute_invariants(catalog, wavelength_grid_nm, runtime_device=None)` for the one-record
   kernel mapping.
8. The amplitude, damping, center-cutoff, and deposit stages of `_accumulate_metal(...)`.
9. `accumulate_atomic(invariants, state, do_metal=True, do_helium=False, apply_stim=True,
   wing_mode='batched', output_line_mass_absorption_coefficient=None, host_accumulator=None)`.

The source primitives and formula stages are in
`/Users/ysting/payne-zero/payne_zero_synthesis/line_opacity.py`, with `_accumulate_metal` beginning at
line 1450. Underscore-prefixed routines are identified as pinned internals.

### Data

- Harris profile and exponential tables from `line_profile_tables.npz`.
- A one-record static teaching subset copied from the atomic catalog with its original record identity.
- Four atmosphere fixtures, but only the selected species/ion population columns.
- A golden one-line slab and intermediate depth vectors.

### Parity and physical gates

1. Profile values are finite and non-negative.
2. Gaussian and Lorentz limiting cases behave correctly.
3. Numerical profile area is stable as the diagnostic grid is widened/refined.
4. Doppler width grows with temperature and microturbulence and shrinks with species mass.
5. Increasing each damping rate strengthens the appropriate wings.
6. Line amplitude is linear in oscillator strength and normalized lower-stage population.
7. Stimulated emission weakens net absorption and never reverses its sign in the supported LTE case.
8. The source FASTEX/Harris tables match a direct reference within their documented approximation
   error.
9. The canonical one-line slab matches the pinned metal kernel for a catalog containing only that
   line in each regime. Hot-star disappearance of a neutral line is checked against its population,
   not mistaken for a missing deposit.

### Device and dtype behavior

- Profile and thermodynamic evaluation use `highp_dtype`: float64 on CPU/CUDA, float32 on MPS.
- The output slab is float32 on every backend.
- Catalog constants may originate as host float64; indices are int64 on device.
- The one-line chapter keeps the implementation transparent and does not introduce chunking or
  giant-catalog host loops.

### No-redundancy and forward-reference audit

- Do not rederive Saha–Boltzmann state or continuum opacity.
- Derive the ordinary Voigt line once here. Chapters 7 and 8 reuse these primitives.
- Mention only that some lines are non-Voigt; their physics belongs to Movement 7B.
- End with exactly one ordinary-line slab and the reusable line-profile primitives.

---

## Chapter 7 — Atomic Line Forests and Special Profiles

### Chapter question

**How do millions of catalog records become a sparse ordinary-metal forest while hydrogen, helium,
and autoionizing records retain the special physics their profiles require?**

Movement 7A ends with a verified ordinary metal forest. Movement 7B begins from that checkpoint and
adds every supported special atomic branch. If the course expands to sixteen chapters, the chapter
boundary belongs exactly between these movements.

### Movement 7A question

**How can millions of possible transitions be turned into the small set of depth–line deposits that
actually affect one wavelength window?**

Open with the impossibility of allocating `(D, L, W)` for a full catalog, then count how many lines
and depth–line pairs survive each filter.

### Prerequisites

- Ch. 2 array layout, caching, scatter-add definition, device timing, source manifests, and checksums.
- Ch. 6 complete ordinary-line formula and cutoff.

### Reads / writes contract

Reads:

- fixed-width decoded atomic source arrays on host;
- `wl_start_nm`, `wl_end_nm`, and `resolution` at the internal pipeline boundary;
- line profile tables;
- Chapter 6 star-dependent depth state and Chapter 5 continuum.

Writes:

- `LineCatalog` structure-of-arrays for the context-expanded window;
- branch masks and counts;
- `AtomicInvariants` chunks on device;
- one float32 `(D, W)` ordinary-metal line forest.

### Derivation and prose arc

1. **A catalog record is a measurement/model statement.** Decode wavelength, energies, oscillator
   strength, species/isotope, damping fields, line-size routing nibble, and category tag.
2. **Corrected physical fields.** Apply isotope/log-`gf` corrections, derive wavelength from shifted
   energies when available, compute lower excitation, and fill missing damping defaults.
3. **Normalized damping convention.** Explain conversion to
   `gamma_linear / (4 pi nu)` and why raw damping logs are retained for special branches.
4. **Logarithmic intrinsic grid.** Construct constant-ratio samples and 16 context samples on each
   side. Define mathematical \(R_{\rm grid}\) as intrinsic sampling density, not instrument
   resolution, while keeping the exact code argument `resolution`.
5. **Conservative window filtering.** Use line-size-dependent margins, a broader hydrogen margin, and
   exact context bounds. This is source/catalog filtering, not the live depth-dependent opacity
   cutoff.
6. **Routing matrix.** Show all source category codes before any kernel is called:

   | Code | Meaning in catalog | Standard synthesis treatment |
   | ---: | --- | --- |
   | `0` | ordinary atomic line | ordinary LTE metal kernel |
   | `1` | `AUT` | Movement 7B autoionizing kernel |
   | `2` | `COR` | parsed/retained, not selected by standard pipeline masks |
   | `3` | `PRD` | ordinary LTE metal kernel; no PRD transfer |
   | `-1` | H I | Movement 7B hydrogen engine |
   | `-2` | D I | Movement 7B hydrogen engine |
   | `-3` | He I | Movement 7B helium kernel |
   | `-4` | He-3 I | Movement 7B helium kernel with isotope scaling |
   | `-6` | He II | Movement 7B helium kernel |

7. **Invariant versus star-dependent selection.** Catalog decoding, centers, routing, and profile
   tables are window invariants. The Chapter 6 amplitude/continuum test is repeated for every depth
   and line at run time.
8. **Sparse center deposits.** Flatten depth and wavelength indices and use scatter-add rather than
   forming a dense line cube.
9. **Wing reach.** Split narrow lines into batched reach tiers and retain a bounded per-wide-line
   walk where work is irregular. Explain why a short loop over genuinely broad survivors can be
   preferable to padding every line to maximum reach.
10. **Chunking.** Build metal invariant chunks, accumulate them into the same slab, and preserve sum
    order. Introduce the cache as a derived acceleration whose identity includes sources, grid, and
    logic version.
11. **Atmosphere-selection boundary.** State that the later atmosphere opacity-sampling solver uses
    an additional resident-catalog, `prange` keep test. Do not implement or time it here; this chapter
    owns synthesis catalog decoding and live cutoffs. The conceptual distinction is now established
    and need not be re-explained later.

### Bite-size source-faithful code artifacts

1. `Grid(start_wavelength_nm, end_wavelength_nm, resolution)`.
2. Exact `Grid.ratio`, `Grid.log_spacing`, and `Grid.build()` behavior.
3. `LineCatalog` with its exact source fields and no replacement alias.
4. `LineCatalog.to_npz(path)`, `LineCatalog.from_npz(path)`, and
   `LineCatalog.to_torch(device=None, float_dtype=None, int_dtype=None)`.
5. `_build_records(raw, apply_iso_corr=True)`.
6. `_line_window_mask(catalog_records, start_wavelength_nm, end_wavelength_nm)`.
7. `_assemble_catalog(catalog_records, window_mask, grid_obj, wavelength_grid, sort)`.
8. `parse_catalog(grid, catalog_path=None, sort='catalog', apply_iso_corr=True)`.
9. `load_catalog(window, grid, catalog_path=None, cache_dir=None, sort='catalog',
   apply_iso_corr=True, rebuild=False)`.
10. `_window_grid_contract(wl_start_nm, wl_end_nm, resolution)`.
11. `nearest_grid_indices(wavelength_grid_nm, line_wavelength_nm)` and
   `nearest_grid_indices_raw(wavelength_grid_nm, line_wavelength_nm,
   origin_wavelength_nm)`.
12. `precompute_invariants(catalog, wavelength_grid_nm, runtime_device=None)`.
13. `harris_wing_walk_profile(doppler_offset, damping_ratio, invariants,
    use_low_damping_branch)`.
14. `_scatter_add_rows(...)`, `_wing_walk_narrow_batched(...)`, and `_wing_walk_metal(...)`.
15. `_accumulate_metal(...)`; the chapter calls `accumulate_atomic(...)` with `do_metal=True` and
    `do_helium=False` only on a type-0/type-3 teaching slice. State explicitly that the production
    `do_metal=True` branch also invokes `_accumulate_autoionizing` when type-1 invariant fields are
    present.

The exact source is:

- `atomic_lines.Grid`, `LineCatalog`, `_build_records`, `_line_window_mask`, `_assemble_catalog`,
  `parse_catalog`, and `load_catalog` at
  `/Users/ysting/payne-zero/payne_zero_synthesis/atomic_lines.py:168`,
  `:393`, `:723`, `:903`, `:934`, `:1059`, and `:1110`;
- ordinary masks, invariant fields, scatter helpers, and `_accumulate_metal` at
  `/Users/ysting/payne-zero/payne_zero_synthesis/line_opacity.py:342`, `:775`, and `:1450`;
- exact context-grid contract at
  `/Users/ysting/payne-zero/payne_zero_synthesis/pipeline.py:843`.

### Data

- A checksum-bound fixed-width/decoded atomic teaching slice containing every route code available in
  the chosen windows.
- Full `atomic_source_lines_parsed.npz` for optional full-data runs.
- `ionization_potential_lookup.npz`.
- Harris/FASTEX arrays from `line_profile_tables.npz`.
- Catalog NPZ `__meta__` with exact keys `schema`, `logic_version`, `start_wavelength_nm`,
  `end_wavelength_nm`, `resolution`, `sort`, and `type_segments`. The derived cache-key payload
  separately uses `schema`, `logic_version`, `source`, `size`, `mtime_ns`,
  `start_wavelength_nm`, `end_wavelength_nm`, `resolution`, `sort`, and `iso_corr`; record count is
  the array length, not an invented metadata field.

### Parity and physical gates

1. Field-by-field decode parity for selected source records.
2. Isotope correction, energy-derived wavelength, and each damping-default branch match the pinned
   source.
3. Route codes and per-route counts match exactly.
4. A test proves type 3 enters the ordinary-metal mask.
5. A test proves type 2 is parsed but absent from all standard run masks.
6. Requested interior wavelength values are bitwise identical before and after context expansion.
7. Window selection counts and selected source identities match.
8. Sparse center deposits equal a dense teaching oracle for a tiny catalog.
9. Batched narrow and loop wing modes agree for overlapping supported cases.
10. Chunk-size changes stay within the explicit fp32 reduction tolerance; fixed chunk policy is
    checked exactly where feasible.
11. The ordinary-metal slab and line counts match the pinned source for all four regimes.
12. Cold and warm catalog-cache results are identical; corrupt or identity-mismatched caches rebuild
    from sources.

### Device and dtype behavior

- Parsing, corrections, masks, and cache identity use host NumPy, primarily float64 and integer arrays.
- `AtomicInvariants` move only selected, preprocessed structure-of-arrays fields to the device.
- Work/profile precision follows `highp_dtype`; accumulation remains float32.
- Gather/scatter indices are int64.
- Narrow wings are batched; genuinely wide lines retain bounded host-controlled iteration and may
  require small synchronization points. The chapter names each synchronization instead of claiming a
  fully branchless kernel.
- No Numba is added to synthesis catalog accumulation. The actual serial `njit` compiler appears in
  Chapter 8.

### No-redundancy and forward-reference audit

- Ordinary-line physics comes only from Chapter 6.
- Special branch formulas are not previewed; only their route ownership is fixed.
- The atmosphere `prange` selector is not implemented here.
- Cache architecture is introduced only enough to make catalog products reproducible; process and
  persistent cache composition is completed in Movement 10A.
- End with a verified ordinary metal forest and a complete branch routing table.

### Movement 7B — Hydrogen, Helium, and Non-Voigt Profiles

### Movement 7B question

**Why do the strongest hydrogen, helium, and autoionizing features refuse to behave like ordinary
isolated Voigt lines?**

Begin with a Balmer wing and an autoionizing asymmetric feature placed beside an ordinary line. The
different shapes demand different physics, not extra Voigt tuning.

### Prerequisites

- Ch. 6 line strength, Doppler width, damping, and Voigt profile.
- Movement 7A catalog fields, route codes, context windows, and sparse deposits.
- Ch. 3 electron and perturber densities.

### Reads / writes contract

Reads:

- routed H/D, He I/He-3 I/He II, and autoionizing catalog records;
- electron density, temperature, H/He normalized populations, mass density, Doppler widths, and
  collision proxy;
- wavelength grid and continuum opacity;
- hydrogen/line profile tables.

Writes:

- additive float32 `(D, W)` slabs for autoionizing, helium, and hydrogen branches;
- exact `HydrogenLine`, `HydrogenMergedCont`, and `HydrogenInvariants` records plus
  `merge_wavenumber_by_depth`: `(D,)`;
- route-specific diagnostics: fine-structure components, Stark state, neighbor/merge weights, and
  active reach.

### Derivation and prose arc

1. **Hydrogen's repeated series.** Introduce principal quantum number, Balmer/Paschen/Brackett series,
   and fine-structure components. Do not assume prior atomic-series terminology.
2. **Electric microfields.** Build quasi-static Stark broadening from nearby charged perturbers, then
   add electron-impact broadening. Explain why density changes the shape, not only the amplitude.
3. **Fine-structure mixture.** Sum component offsets and weights while preserving integrated
   strength.
4. **Neighbor dominance and series merging.** Show how adjacent upper levels crowd toward a series
   limit. Derive the density-dependent Inglis–Teller merge level and explain pseudo-continuum records
   with `n_upper == 99`.
5. **Resolved versus merged records.** Apply active windows, neighbor tapers, merged-continuum
   strength, and stimulated emission.
6. **Hydrogen boundary.** Neutral H/D with `n_lower >= 2` are supported. A Lyman record raises
   `NotImplementedError`. NLTE departure coefficients and the unsupported alternate atmosphere
   hydrogen-wing branch are not silently substituted.
7. **Helium families.** Explain He I, He-3 isotope scaling, He II, and per-depth continuum-merge
   tapers. State which metadata the optical catalog lacks and which default zero tapers the standard
   pipeline therefore supplies.
8. **Autoionization.** Introduce a discrete state coupled to a continuum and the resulting
   Shore–Fano-like asymmetric profile. Explain why raw damping logs are reconstructed separately from
   normalized ordinary-line damping.
9. **One additive slab.** Accumulate the three special branches into the same float32 line slab only
   after each has a separate diagnostic and parity gate.

### Bite-size source-faithful code artifacts

1. `_hydrogen_level_energy_cm(nn)`.
2. `_fine_structure(lower_level, upper_level)`.
3. `_stark_quasi_static_profile(beta, pressure_parameter, lower_level, upper_level,
   stark_probability_table, stark_wing_correction_c_table, stark_wing_correction_d_table,
   stark_pressure_grid, stark_beta_grid)`.
4. `_hydrogen_state(temperature, electron_density, helium_neutral_population,
   molecular_hydrogen_population, hydrogen_partition_normalized_ion_stage_populations,
   microturbulence, dtype, compute_device)`.
5. `merge_wavenumber_by_depth(electron_density)`.
6. Exact `HydrogenLine`, `HydrogenMergedCont`, and `HydrogenInvariants` dataclasses.
7. `precompute_invariants(catalog, wavelength_grid_nm, electron_density, compute_device=None)`.
8. `accumulate_hydrogen(invariants, state, apply_stim=True)`, with resolved and merged stages exposed
   as readable slices rather than renamed functions.
9. `_accumulate_helium(...)`.
10. `_accumulate_autoionizing(...)`.
11. `accumulate_atomic(...)` with the exact `do_metal`, `do_helium`, and `apply_stim` branch switches;
    there is no separate public autoionizing switch because `_accumulate_autoionizing` is inside the
    `do_metal` branch.

Exact source placement:

- all hydrogen level/profile/state/invariant/merge/deposit routines in
  `/Users/ysting/payne-zero/payne_zero_synthesis/hydrogen_lines.py`, especially
  `_fine_structure` at line 186, `_stark_quasi_static_profile` at line 255,
  `_hydrogen_state` at line 771, `merge_wavenumber_by_depth` at line 1036,
  `precompute_invariants` at line 1054, and `accumulate_hydrogen` at line 1610;
- `_accumulate_autoionizing` and `_accumulate_helium` in
  `/Users/ysting/payne-zero/payne_zero_synthesis/line_opacity.py:1289` and `:1713`.

### Data

- Hydrogen Stark/probability grids, radiative sums, Harris tables, and hydrogen continuum edges from
  `line_profile_tables.npz`.
- Catalog slices with H, D, He I, He-3 I, He II, type-1 autoionizing, and merged `n_upper == 99`
  records where present.
- Hot and Balmer-focused fixture windows, plus the giant/solar/cool controls.
- Branch-separated golden slabs and intermediate merge/profile arrays.

### Parity and physical gates

1. Fine-structure weights are non-negative and sum to the required transition weight.
2. Hydrogen profile remains finite, non-negative, and symmetric where the selected physical
   approximation predicts symmetry.
3. Higher electron density changes Stark width and lowers the merge level in the expected direction.
4. Resolved-series strength transfers smoothly into the pseudo-continuum under the source taper.
5. `n_upper == 99` records route only to merged-continuum handling.
6. An `n_lower == 1` teaching record fails loudly with the documented boundary.
7. He-3 scaling and He II population-slot selection match source invariants.
8. Autoionizing center/asymmetry and integrated contribution match the pinned branch.
9. Each branch slab matches independently before the sum.
10. Four-regime tests include: active H/He hot-star behavior, solar Balmer behavior, pressure-sensitive
    giant behavior, and a cool-star low-population/continuum-crowding check.
11. Type-3 PRD and type-2 COR status tests from Movement 7A remain green; this movement must not
    accidentally reroute them.

### Device and dtype behavior

- Per-depth hydrogen state and discrete profile regimes use small host float64 arrays where the source
  does.
- Profile tables and large wavelength/depth evaluations live on the selected device in high
  precision.
- All deposits enter the shared float32 slab.
- Wavelengths/depths are vectorized; the small number of broad hydrogen/special records and
  directional walks retain explicit bounded loops.
- Hydrogen window invariants are cached as a template, but `merge_wavenumber_by_depth` is rebuilt for
  every star from its electron density.

### No-redundancy and forward-reference audit

- Do not reteach ordinary Voigt mathematics.
- Keep H/He continuum processes in Chapter 5 distinct from H/He bound-bound lines here.
- Do not imply a PRD transfer solver.
- Do not introduce molecular opacity.
- End with all atomic/special/hydrogen branches complete and branch-separated parity evidence.

---

## Chapter 8 — Molecular Bands and Source Compilation

### Governing question

**How can tens of millions of molecular transitions be compiled and deposited without losing the
connection between a molecule's equilibrium abundance and its spectral bands?**

Open with the cool fixture: atomic lines alone leave broad forests missing. Then show one molecular
population feeding many transitions.

### Prerequisites

- Ch. 4 molecular equilibrium, 190-species catalog, and synthetic stage-5 population packing.
- Ch. 6 reusable Voigt/excitation/damping primitives.
- Ch. 7 catalog grids, sparse deposits, and cache identity.
- Ch. 2 scalar Python versus cached serial `njit`.

### Reads / writes contract

Reads:

- molecular equilibrium stage-5 normalized population: `partition_normalized_populations[:, 5, :]`;
- species codes and masses;
- manifest-ordered text-band arrays, packed TiO source, and packed H₂O source;
- temperature, mass density, electron density, microturbulence, `hc/kT`, collision proxy, continuum
  slab, and synthesis grid.

Writes:

- compiled molecular line structure-of-arrays;
- `MolecularLineCatalog` and device `MolecularLineInvariants`;
- float32 `(D, W)` molecular opacity;
- compiler/runtime feature-status matrix.

### Derivation and prose arc

1. **Chemistry versus opacity.** Reuse Chapter 4's molecule abundance; do not solve chemistry again.
   Explain why one molecule can have rotational-vibrational-electronic forests.
2. **Three source families.** Introduce manifest-ordered text bands, packed TiO, and packed H₂O as
   distinct source formats that produce one common compiled field contract.
3. **Text parser.** Decode fixed fields, dispatch band-specific quantum-number logic, derive
   wavelength from energy levels when requested, and form damping/strength fields.
4. **Readable scalar compiler.** Implement one record at a time as the parity oracle.
5. **Real Numba optimization.** Compile the same record loop with serial
   `@numba.njit(cache=True)`. Explain why the pinned compiler does not use `prange`: preserving exact
   ordered record output and avoiding a manufactured parallel example matter more than cosmetic
   parallelism.
6. **Packed compiler paths.** Use memory mapping and table decoding for TiO and H₂O without pretending
   they share the text parser.
7. **Feature matrix.**

   | Family | Compiler exists | Standard pipeline compiles it | Standard opacity accumulates it |
   | --- | --- | --- | --- |
   | manifest text bands | yes | yes | yes |
   | TiO | yes | yes | yes |
   | H₂O | yes | no | no |

8. **Common line invariants.** Map species code to stage-5 population column and species mass. Compute
   molecular Doppler widths from temperature, microturbulence, and mass.
9. **Giant-catalog streaming.** Process approximately `CHUNK_LINES = 500000` lines at a time, apply the
   live continuum cutoff, flatten surviving `(depth, line)` pairs, and process
   `PAIR_CHUNK = 200000` pairs at a time.
10. **Center, near wing, far wing.** Reuse Chapter 6 profile primitives, deposit centers, walk near
    wings in bounded offset blocks, then use the inverse-square far-wing approximation until the
    opacity floor or grid boundary stops the pair.
11. **Cache as acceleration.** The standard compiled-window cache `__meta__` contains exact keys
    `cache_identity`, `start_wavelength_nm`, `end_wavelength_nm`, `resolution`, `log_grid_ratio`,
    `grid_origin_index`, and `n_lines`. The later `MolecularLineCatalog.to_npz(...)` cache uses
    `schema`, `logic_version`, `log_grid_ratio`, and `grid_origin_index`. A cache miss may require the
    multi-gigabyte source tree; a valid portable compiled cache does not change physics.

### Bite-size source-faithful code artifacts

1. Exact compiled array keys consumed by `build_catalog_from_arrays(arrays)`; no replacement
   dataclass.
2. `_parse_text_line(line)` and `_iterate_parsed_band(arrays, band)`.
3. `_get_compiled_band_kernel()`, whose returned kernel is the serial cached-`njit` path.
4. `compile_molecular_text(band_catalog_path, band_names, start_wavelength_nm,
   end_wavelength_nm, resolution, use_energy_level_wavelengths=False,
   include_predicted_lines=False)`.
5. `compile_tio_schwenke(bin_path, start_wavelength_nm, end_wavelength_nm, resolution,
   use_vacuum_wavelengths=False)`.
6. `compile_h2o_partridge(bin_path, start_wavelength_nm, end_wavelength_nm, resolution,
   use_vacuum_wavelengths=False)`.
7. The literal compiler/runtime feature table above (no invented feature-matrix callable).
8. Exact `MolecularLineCatalog`, built by `build_catalog_from_arrays(arrays)`.
9. `MolecularLineCatalog.to_npz(path)`, `MolecularLineCatalog.from_npz(path)`, and
   `MolecularLineCatalog.from_mapping(mapping)`.
10. `load_catalog(source_path, cache_dir=None, rebuild=False)`.
11. Exact `MolecularLineInvariants`, built by `precompute_invariants(catalog,
   wavelength_grid_nm, harris_profile_h0_table, harris_profile_h1_table,
   harris_profile_h2_table, runtime_device=None)`.
12. `species_population_doppler_ratio(invariants, state)`.
13. `molecular_chunk_lines(default=CHUNK_LINES)`.
14. `_accumulate_chunk(...)`, `_near_wing(...)`, `_far_wing(...)`, and
    `_scatter_add_flat(...)`.
15. `accumulate_molecular(invariants, state, apply_stim=True, chunk_lines=None)`.

Exact source placement:

- compiler cache/parser/dispatch and serial Numba kernel:
  `/Users/ysting/payne-zero/payne_zero_synthesis/source_catalog_molecular_compiler.py:34`,
  `:291`, `:373`, `:424`, and `:527`;
- TiO and H₂O compiler paths:
  `/Users/ysting/payne-zero/payne_zero_synthesis/source_catalog_molecular_compiler.py:817` and `:960`;
- runtime catalog/population/chunk/deposit code:
  `/Users/ysting/payne-zero/payne_zero_synthesis/molecular_lines.py:144`,
  `:332`, `:435`, `:495`, `:613`, `:806`, `:901`, and `:990`;
- standard text-plus-TiO composition:
  `/Users/ysting/payne-zero/payne_zero_synthesis/pipeline.py:1169`.

### Data

- `source_catalogs/molecules/manifest.json`.
- A small slice from each text band required by the manifest.
- Small packed TiO and H₂O teaching slices with record-range provenance.
- Optional full `molecular_band_lines.npz`, TiO, and H₂O arrays.
- Molecular species masses and `line_profile_tables.npz`.
- Cool and low-gravity fixtures, plus hot/solar near-zero controls.
- Scalar compiler outputs as generated test values; source goldens remain separate.

### Parity and physical gates

1. Every compiled array has identical length and documented dtype.
2. Scalar and serial-`njit` text compilers are exactly equal record by record.
3. Cold compilation and warm cached execution are timed separately.
4. Text-band manifest order and per-band counts match exactly.
5. TiO and H₂O packed teaching slices match the pinned compilers.
6. The standard runtime feature test proves H₂O is not concatenated into
   `SynthesisPipeline._compile_molecular`.
7. Species code maps to the exact expected stage-5 population column.
8. Doppler width obeys temperature, mass, and microturbulence limits.
9. Sparse chunked deposits match a tiny dense oracle.
10. Different safe chunk sizes remain within the explicit fp32 reduction tolerance; the standard
    chunk policy is used for strict full parity.
11. Text-only, TiO-only, and combined slabs match separately in the cool fixture.
12. All four regimes are tested: cool and giant active forests, solar moderate molecular behavior,
    hot-star physically small populations.
13. An H₂O compiler test passes while the complete standard-spectrum golden remains H₂O-free.

### Device and dtype behavior

- Source parsing, scalar compilation, Numba compilation, cache I/O, and memory mapping are CPU/host
  work.
- There is no `prange` compiler path.
- Compiled invariants move to the selected Torch device in work/high precision.
- Population/damping/profile algebra remains device-resident; the shared output is float32.
- Line chunks and surviving pair chunks bound memory. Small host scalar reads that control bounded
  loops are named and profiled.
- No `torch.compile` path is added.

### No-redundancy and forward-reference audit

- Do not repeat molecular equilibrium or Newton solving.
- Do not repeat the Voigt derivation or generic scatter-add explanation.
- Teach source compilation only here; Chapter 10 may discuss cache residence but not parse records
  again.
- The H₂O boundary must appear in the feature matrix and tests, not only prose.
- End with complete standard-runtime molecular opacity, explicitly text bands plus TiO.

---

## Chapter 9 — Radiative Transfer with Scattering

### Chapter question

**How do depth-dependent absorption, emission, and scattering combine to produce the total and
continuum flux that escape from a stellar atmosphere?**

Movement 9A builds and verifies the variable-depth absorption-only formal solution. Movement 9B adds
the fixed-grid moment operator, scattering iteration, saturated-core fallback, and flux conversion.

### Movement 9A question

**Given opacity and thermal emission at every depth, how much radiation escapes from the surface?**

Begin with two atmospheres having the same temperature profile but different opacity. The emergent
light comes from different depths, turning Chapter 1's slab intuition into a variable-depth solver.

### Prerequisites

- Ch. 1 `B_nu`, transfer equation, optical depth, source function, constant-source formal solution,
  column mass, and introductory angular moments.
- Ch. 5–8 complete absorption and scattering inputs, though scattering is switched off in this
  chapter.

### Reads / writes contract

Reads:

- `column_mass`: `(D,)`, g cm⁻²;
- `extinction`: `(W,D)`, cm² g⁻¹;
- `depth_values`: `(W,D)` when remapping explanatory source rows;
- `surface_tau`: `(W,)`.

Writes:

- exact `integrate_optical_depth(...)` return `optical_depth`: `(W,D)`;
- inline absorption-only emergent intensity/flux diagnostics `(W,)`;
- inline contribution-function diagnostics used only for explanation.

### Derivation and prose arc

1. **Transfer through one differential layer.** Return to the physical transfer equation, now with
   depth-varying extinction and source.
2. **Column mass as the integration coordinate.** Derive `d tau_nu = kappa_nu dm` with signs and
   outer-to-inner ordering explicit.
3. **Parabolic opacity between layers.** Derive constant, linear, and quadratic interval
   coefficients and their analytic integral. Compare against trapezoidal integration on a curved
   test profile.
4. **Cumulative optical depth.** Use one `cumsum` after independent interval integrals. Explain why
   interval work vectorizes but the mathematical prefix relation remains ordered.
5. **Formal absorption-only solution.** Build emergent intensity for a small set of rays, then angular
   flux. Use constant-source and isothermal limiting cases.
6. **Eddington–Barbier.** Derive it as an interpretation/approximation, not the production algorithm.
7. **Contribution function.** Plot which layers contribute at continuum and line-center wavelengths.
8. **Saturation.** Explain when the surface already lies at very large monochromatic optical depth
   and why a fixed transfer grid needs explicit handling.
9. **Wavelength independence.** Once atmosphere and opacity are fixed, wavelength rows do not couple
   in absorption-only transfer. This is the physical reason for the batch axis used in Movement 9B.

### Bite-size source-faithful code artifacts

1. `planck_bnu(wavelength_nm, temperature)`.
2. `_parabolic_interval_coefficients(values, depth_grid)`.
3. `integrate_optical_depth(column_mass, extinction, surface_tau)`.
4. Inline formal-intensity, absorption-only flux, contribution-function, and saturated-surface checks
   (no new API symbols).

The source-aligned numerical pieces are `radiative_transfer.planck_bnu`,
`_parabolic_interval_coefficients`, and `integrate_optical_depth` at
`/Users/ysting/payne-zero/payne_zero_synthesis/radiative_transfer.py:97`, `:146`, and `:246`.

### Data

- Four structured-atmosphere fixtures.
- Continuum-only and one-line opacity slabs computed by Chapters 5–7.
- Analytic slab inputs generated in tests, not stored as opaque fixtures.
- Pinned optical-depth and absorption-only diagnostic goldens for selected rows.

### Parity and physical gates

1. Constant extinction gives the analytic linear optical-depth profile.
2. Parabolic integration is exact for a quadratic test opacity.
3. Optical depth is finite, non-negative, and non-decreasing inward.
4. Zero-opacity and optically thick limiting cases recover the expected formal behavior.
5. Constant-source slab matches the Chapter 1 analytic solution.
6. Increasing opacity shifts the contribution function outward.
7. Line-center formation is outward of nearby continuum in a selected active line.
8. Batched rows equal a scalar wavelength loop.
9. Source optical-depth arrays match for selected wavelengths in all four regimes.

### Device and dtype behavior

- Opacity arrives as `(D, W)` and is transposed contiguously to `(W, D)` for wavelength batching.
- Parabolic coefficient and interval work use the selected work/high precision.
- The only ordered operation is the depth prefix sum, implemented by `torch.cumsum`.
- No fixed 51-point operator or scattering iteration is introduced yet.
- Do not cast the source iteration to fp32 because source iteration belongs to Movement 9B.

### No-redundancy and forward-reference audit

- Chapter 1 owns constant-slab intuition; Movement 9A owns the variable-depth numerical method.
- Chapter 1 introduced moments physically; detailed scattering moments wait for Movement 9B.
- Do not repeat opacity physics.
- End with a verified absorption-only transfer result and optical-depth map, not the public spectrum.

### Movement 9B — Scattering and the Moment Solver

### Movement 9B question

**How does radiation escape when photons can be redirected many times before they are absorbed or
leave the atmosphere?**

Open by comparing an atmosphere with the same total extinction but different absorption/scattering
fractions. Extinction alone cannot determine the thermal source.

### Prerequisites

- Ch. 1 source function, scattering distinction, and physical meaning of `J`, `H`, and `K`.
- Movement 9A column-mass optical depth and wavelength batching.
- Ch. 5 separate absorption/scattering slabs.

### Reads / writes contract

Reads:

- `continuum_absorption`, `continuum_scattering`, `line_mass_absorption_coefficient`, and
  `line_scattering`: `(D,W)`; standard LTE synthesis sets `line_scattering` to zero;
- `planck_source`: `(D,W)`;
- `column_mass`: `(D,)`;
- `TransferTables.transfer_optical_depth_grid`: `(51,)`;
- `mean_intensity_operator`: `(51,51)`, `mean_intensity_diagonal`: `(51,)`, and
  `surface_eddington_flux_weights`: `(51,)`.

Writes:

- `total_extinction`, `scattering_fraction`, and `thermal_source`: `(W,D)` after transpose;
- `thermal_source_grid` and `scattering_fraction_grid`: `(W,51)`;
- the `solve_scattering_source(...)` returned source: `(W,51)`;
- `eddington_flux_total_per_frequency`, `eddington_flux_continuum_per_frequency`, and
  `normalized_flux`: `(W,)`;
- public-unit conversion diagnostics.

### Derivation and prose arc

1. **Moments as angular summaries.** Define `J`, `H`, and `K` from intensity and state the role each
   plays.
2. **Thermal and scattering source.** Derive total extinction, scattering fraction, and opacity-weighted
   thermal source. Then write `S = (1 - epsilon) J + epsilon B` with the chapter's exact epsilon
   convention.
3. **Why a fixed optical-depth grid.** Different wavelengths have different physical-depth optical
   depths. Map each row onto the common 51-point grid so one precomputed operator can be reused.
4. **Parabolic remap.** Reuse the interpolation philosophy of Movement 9A, but explain discrete
   `searchsorted` brackets, edge stencils, curvature blending, and detached stencil selection.
5. **Lambda operator.** Interpret the `51 x 51` matrix as a linear map from source to mean intensity.
   Use a tiny matrix example before loading the production table.
6. **Backward Gauss–Seidel.** Derive one update. Explain why deeper updated points are immediately
   reused, making the depth loop sequential while all wavelength rows remain parallel.
7. **Eight sweeps and fp32.** Preserve the validated fixed sweep count and arithmetic. Do not replace
   it with an unmeasured convergence criterion.
8. **Surface flux.** Apply surface Eddington-flux weights in high precision.
9. **Saturated-core fallback.** Explain the detection and the derivative-based fallback when the
   physical surface lies below the deepest fixed-grid point.
10. **Total and continuum together.** Stack total and continuum rows into one call so both follow the
    same operator launch pattern, then form normalized flux.
11. **Flux semantics.** Derive `F_nu = 4 pi H_nu` and
    `F_lambda(per nm) = 4 pi H_nu * c_nm_s / lambda_nm^2`. Verify the Jacobian and units explicitly.

### Bite-size source-faithful code artifacts

1. `TransferTables.from_npz(transfer_tables_path, device=None, dtype=DEFAULT_DTYPE)`.
2. Exact `TransferTables.n_grid` and `TransferTables.to(device, dtype=None)` behavior.
3. `source_and_alpha(continuum_absorption, continuum_source,
   line_mass_absorption_coefficient, line_source, continuum_scattering, line_scattering)`.
4. `_interpolate_to_transfer_grid(optical_depth, depth_values, transfer_depth_grid)`.
5. Inline one-depth Gauss–Seidel update derivation/check (no new API symbol).
6. `solve_scattering_source(thermal_source_grid, scattering_fraction_grid, lambda_operator,
   lambda_operator_diagonal, sweeps=DEFAULT_SWEEPS)`.
7. `_saturated_core_flux(optical_depth, thermal_source, scattering_fraction)`.
8. `_solve_flux_rows(continuum_absorption, continuum_source,
   line_mass_absorption_coefficient, line_source, continuum_scattering, line_scattering,
   column_mass, tables, sweeps=DEFAULT_SWEEPS, assert_no_saturated_core=True)`.
9. `solve_spectrum(continuum_absorption, continuum_scattering,
   line_mass_absorption_coefficient, line_scattering, planck_source, column_mass, tables,
   sweeps=DEFAULT_SWEEPS, assert_no_saturated_core=None)`.
10. `_surface_flux_per_wavelength_nm(wavelength_nm, eddington_flux_per_frequency)`.

Exact source placement:

- `TransferTables`, `source_and_alpha`, transfer-grid interpolation, source iteration, saturated
  fallback, and stacked solve in
  `/Users/ysting/payne-zero/payne_zero_synthesis/radiative_transfer.py:30`,
  `:112`, `:283`, `:415`, `:522`, `:586`, and `:677`;
- public flux conversion in
  `/Users/ysting/payne-zero/payne_zero_synthesis/api.py:312`.

### Data

- `transfer_tables.npz`: 51 optical depths, `51 x 51` mean-intensity operator, 51 diagonal entries
  derived from it, and 51 surface weights.
- Four atmosphere fixtures and their complete opacity slabs.
- Small analytic/tiny-matrix tests generated in code.
- Pinned transfer-grid, source, `H_nu`, and public-flux goldens for selected wavelengths.

### Parity and physical gates

1. Scattering fraction remains in `[0, 1]`; extinction has the positive floor.
2. Zero scattering reduces to the Movement 9A absorption-only result within the method difference
   documented by the test.
3. Purely thermal and scattering-dominated limiting cases behave physically.
4. Transfer-grid interpolation reproduces values at exact source points and remains finite at
   boundaries.
5. One hand-computed Gauss–Seidel update matches code.
6. The source remains positive through all eight sweeps.
7. Batched wavelengths match a scalar row solve.
8. Saturated and non-saturated paths are tested separately.
9. Stacked total/continuum solve matches two independent calls.
10. With zero line opacity, total and continuum flux are equal and normalized flux is one.
11. `H_nu`, `F_nu`, `F_lambda`, and normalized flux conversions pass unit/Jacobian tests.
12. Total and continuum `H_nu` and public `F_lambda` match pinned results for all four regimes.

### Device and dtype behavior

- Wavelength is the batch axis; depth remains the ordered Gauss–Seidel axis.
- Transfer tables reside on the selected device.
- Scattering source iteration is intentionally float32 on all backends.
- Final surface-weight multiplication uses float64 on CPU/CUDA and float32 on MPS.
- Discrete `searchsorted` stencil selection is non-differentiable; curvature weighting is detached as
  in the source to keep the remaining graph finite.
- No host spectrum is produced here; Chapter 10 owns the final host crossing.

### No-redundancy and forward-reference audit

- Do not repeat the Movement 9A optical-depth derivation.
- Do not rederive Planck radiation or continuum/line opacity.
- Flux conversion is derived once here and merely invoked in Chapter 10.
- Do not introduce instrument convolution or observed-grid operators.
- End with checked native total, continuum, and normalized flux on device.

---

## Chapter 10 — GPU Synthesis from a Structured Atmosphere

### Chapter question

**How can reusable window data and one star's structured atmosphere remain on a CUDA, MPS, or CPU
device long enough to produce a complete, verified native spectrum with only one final host-copy
stage?**

Movement 10A builds the invariant/star-state architecture and cache contract. Movement 10B composes
the physical kernels and public structured-atmosphere interface without reimplementing them.

### Movement 10A question

**Which parts of a spectrum calculation can be reused for every star in one wavelength window, and
which parts must be rebuilt from the atmosphere?**

Open with two runs over the same window: the first decodes catalogs and builds invariant tables; the
second changes only the atmosphere. The time breakdown motivates the architecture.

### Prerequisites

- Ch. 2 device, kernel, cache, cold/warm, and scatter-add foundations.
- Ch. 5–9 every physical stage.
- Ch. 2 data identities and manifest roles.

### Reads / writes contract

Reads:

- internal `wl_start_nm`, `wl_end_nm`, `resolution`, `molecular_lines`, `runtime_device`, and
  `work_dtype`;
- immutable source identities and table paths;
- one structured atmosphere's star-dependent columns.

Writes:

- a `WindowInvariants` bundle;
- a star-dependent pipeline state;
- exact initialization `build_profile`, overall/prewarm timing fields, and inline memory diagnostics;
- cache/prewarm manifest;
- no public spectrum yet.

### Derivation and prose arc

1. **Parallelism follows the data graph.** Identify independent wavelength rows, independent line
   records until deposition, and independent depth–line pairs. Contrast them with ordered hydrogen
   walks and transfer depth sweeps.
2. **Window invariants.** Build and diagram:
   - context-expanded wavelength grids and output slice;
   - atomic catalog and kernel mapping;
   - continuum and transfer tables;
   - metal invariant chunks;
   - helium invariants;
   - hydrogen invariant template;
   - molecular compiled catalog and invariants;
   - component-presence flags and counts.
3. **Star-dependent state.** Build and diagram:
   - `temperature`, `mass_density`, `electron_density`, `column_mass`, and `microturbulence`;
   - `partition_normalized_populations`, `ion_stage_populations`, and
     `fractional_doppler_widths`;
   - the exact dictionary returned by `build_pops(...)`;
   - internal `collision_density_proxy` and schema field `hc_over_kt`;
   - `merge_wavenumber_by_depth` from this star's `electron_density`;
   - `line_source` from `planck_bnu(...)`.
4. **Precision islands.** Present one explicit map:
   - host float64: source decoding, identities, edge indices, table brackets, discrete regimes, small
     depth state;
   - device work dtype: continuum/profile/transfer algebra;
   - device float32: shared line slab and scattering source iteration;
   - device int64: indices;
   - host float64: final public arrays after the final host-copy stage.
5. **Sparse work and bounded memory.** Explain `SynthesisPipeline.METAL_CHUNK = 40000`,
   `CHUNK_LINES = 500000`, `PAIR_CHUNK = 200000`, narrow/wide wing paths, and why no dense
   `(D,L,W)` allocation exists.
6. **Context then crop.** Build `WINDOW_CONTEXT_SAMPLES = 16` samples per side; compute on
   `synthesis_wavelength_nm`; preserve `wavelength_nm` exactly; crop with `output_slice` on device.
7. **Cache layers.** Separate process cache, persistent derived catalog/molecular caches, and prewarm
   artifact inventory. Every key includes the physical/source identity required to make reuse safe.
8. **Failure-safe cache behavior.** Missing, corrupt, or identity-mismatched cache products rebuild;
   cache files never become scientific truth.
9. **One final host-copy stage.** Keep opacity and flux arrays on device through crop. Convert each
   requested result tensor to host float64 only inside the final result-construction block.
10. **Cold/warm profiling.** Report the exact initialization `build_profile`, overall synthesis
    `seconds`, prewarm timing fields, cache hit/miss, device, and dtype. The pinned
    `SynthesisPipeline.run` contains internal timing hooks but does not return a per-stage run profile;
    do not invent one as a production output. Do not mix compilation/build time with steady-state
    throughput.
11. **Explicit non-features.** No `torch.compile`; no GPU atmosphere iteration; no `prange` synthesis
    kernel; no public multi-atmosphere batch; no implicit H₂O wiring.

### Bite-size source-faithful code artifacts

1. `_window_invariant_key(wl_start_nm, wl_end_nm, resolution, molecular_lines,
   runtime_device, work_dtype, tables_path, transfer_tables_path, continuum_tables_path,
   metal_chunk)`.
2. `_window_grid_contract(wl_start_nm, wl_end_nm, resolution)`.
3. Exact `WindowInvariants` fields with no renamed wrapper class or extra convenience method.
4. `_build_window_invariants(*, wl_start_nm, wl_end_nm, resolution, molecular_lines,
   runtime_device, work_dtype, tables_path, transfer_tables_path, continuum_tables_path,
   metal_chunk, key)`.
5. `window_invariants_for(*, wl_start_nm, wl_end_nm, resolution, molecular_lines,
   runtime_device, work_dtype,
   tables_path=_SYNTHESIS_TABLE_DIR/'line_profile_tables.npz',
   transfer_tables_path=_SYNTHESIS_TABLE_DIR/'transfer_tables.npz',
   continuum_tables_path=_SYNTHESIS_TABLE_DIR/'continuum_tables.npz', metal_chunk=None)`.
6. `window_invariant_cache_enabled()` and `clear_window_invariant_cache()`.
7. `SynthesisPipeline(atmosphere, source_path=None, wl_start_nm=400.0, wl_end_nm=900.0,
   resolution=20000.0, tables_path=_SYNTHESIS_TABLE_DIR/'line_profile_tables.npz',
   transfer_tables_path=_SYNTHESIS_TABLE_DIR/'transfer_tables.npz',
   continuum_tables_path=_SYNTHESIS_TABLE_DIR/'continuum_tables.npz',
   molecular_lines=True, device=None, dtype=None,
   metal_chunk=None, window_invariants=None)`.
8. `prewarm(*, wavelength_start_nm, wavelength_end_nm, resolution, force=False)`, returning the
   emitted manifest mapping; the pinned prewarm build uses `runtime_device=torch.device("cpu")` and
   `work_dtype=torch.float64`.
9. Inline memory-estimate diagnostics plus reporting of exact `build_profile` and overall timing
   fields (no new API symbols or fabricated per-stage run output).

The prewarm mapping uses exact top-level keys `schema_version`, `status`, `created_utc`, `identity`,
`cache_root`, `prewarm_seconds`, `prewarm_seconds_this_call`, `reused`,
`cache_inventory_before`, `cache_inventory`, `required_window_artifacts`, and `window`; `window`
contains `wavelength_count`, `atomic_line_count`, `molecular_line_count`, and `build_profile`.
`identity` contains `cache_root`, `wavelength_start_nm`, `wavelength_end_nm`, `resolution`,
`molecular_lines`, `system`, `machine`, `python`, `torch`, `source_fingerprint`, and
`window_artifact_paths`.
The `build_profile` keys are `init.grid`, `init.atomic_catalog`, `init.atomic_catalog_mapping`,
`init.component_flags`, `init.continuum_tables`, `init.metal_invariants`, `init.he_invariants`,
`init.hydrogen_invariants_template`, `init.molecular_compile`, `init.molecular_catalog`,
`init.molecular_invariants`, and `init.radiative_transfer_tables` when those timed sections run.

Exact path/cache controls are part of the reproducibility contract:

- `PAYNE_ZERO_DATA_ROOT`, `PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT`, and
  `PAYNE_ZERO_SOURCE_CATALOG_ROOT` select physical data/source roots;
- `PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE` overrides the atomic-mass table;
- `PAYNE_ZERO_SYNTHESIS_CACHE_DIR` selects the general derived-cache root;
- `PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1` disables the in-process window cache;
- `PAYNE_ZERO_SYNTHESIS_MOLECULAR_COMPILED_CACHE` defaults enabled,
  `PAYNE_ZERO_SYNTHESIS_MOLECULAR_COMPILED_CACHE_DIR` relocates it, and
  `PAYNE_ZERO_SYNTHESIS_REBUILD_MOLECULAR_COMPILED_CACHE` forces its rebuild;
- `PAYNE_ZERO_SYNTHESIS_DISABLE_MOLECULAR_SOURCE_CACHE=1` disables the compiler source cache, while
  `PAYNE_ZERO_SYNTHESIS_MOLECULAR_SOURCE_CACHE_DIR` relocates it.

The compiled-molecular cache identity uses exact keys `schema`, `start_wavelength_nm`,
`end_wavelength_nm`, `resolution`, `use_energy_level_wavelengths`, `compiler`, and `files`.

Source placement:

- device policy:
  `/Users/ysting/payne-zero/payne_zero_synthesis/device.py:1`;
- `WindowInvariants`, cache controls/key, context grid, and builder:
  `/Users/ysting/payne-zero/payne_zero_synthesis/pipeline.py:641`,
  `:694`, `:800`, `:843`, and `:1554`;
- star-dependent `SynthesisPipeline.__init__` state:
  `/Users/ysting/payne-zero/payne_zero_synthesis/pipeline.py:962`;
- molecular cache contract:
  `/Users/ysting/payne-zero/payne_zero_synthesis/pipeline.py:704` and `:1169`;
- prewarm artifacts and source identity:
  `/Users/ysting/payne-zero/payne_zero_synthesis/prewarm.py:31` through `:236`.

### Data

- All static table and teaching-catalog products from Chapters 5–9.
- Cache manifests and prewarm inventories, stored as derived metadata.
- Four structured atmospheres for star-state comparisons.
- Timing results kept as environment-labeled reports, never as physics goldens.

### Parity and architecture gates

1. Identical window/source/device/dtype requests return the same in-process invariant object when cache
   policy allows.
2. Any key-defining change causes a cache miss.
3. Cold and warm invariant products are array-identical.
4. Corrupt/mismatched persistent caches rebuild and then match the cold source build.
5. Requested interior wavelengths are bitwise identical with and without context construction.
6. Context crop produces the same requested outputs as the pinned pipeline.
7. Hydrogen merge state changes with atmosphere electron density while the invariant template does
   not.
8. No star-dependent tensor is retained in a reusable window invariant.
9. Memory remains within the formula implied by `(D, W)`, chunk sizes, and invariant arrays; tests
   reject accidental dense `(D, L, W)` allocations.
10. No spectral tensor crosses to host before the final host-copy stage; that stage converts each
    requested result array exactly once.
11. CPU float64 and CUDA float64 are compared stage by stage; MPS float32 uses a separately measured
    tolerance. Float32 line accumulation has one policy across backends.
12. Full stage outputs match for the four regimes in cold and warm runs.
13. A source inspection/call-trace test records that `torch.compile` is absent and
    `FrequencyInvariants` is not passed by the standard pipeline.

### Device and dtype behavior

This is the chapter's central artifact: a table and diagram must show every host/device transition and
every dtype cast. Explanations must say why the transition exists:

- host parsing and discrete choices preserve source conventions and avoid repeated device sync;
- high-precision device work protects profile/continuum/flux arithmetic;
- float32 deposits bound memory and preserve validated accumulation behavior;
- ordered loops remain where later values depend on earlier values or work is too irregular to pad;
- a GPU accelerates broad tensor work without erasing control-flow dependencies.

### No-redundancy and forward-reference audit

- No physical opacity or transfer equation is rederived.
- Cache internals appear here once; Movement 10B simply uses them.
- CLI, label initialization, and learned models remain outside this chapter.
- Performance claims are made only after equality gates and include cold/warm/device context.
- End with a reusable invariant bundle and star state, not an alternate full pipeline.

### Movement 10B — A Spectrum from a Structured Atmosphere

### Movement 10B question

**Can the state and kernels built so far reproduce the complete native spectrum of a supplied physical
atmosphere, with every interface and unit checked?**

Open with a schema-v4 archive and end with the four public arrays. The chapter's intellectual work is
composition, contracts, and verification.

### Prerequisites

- Ch. 2 schema-v4 validator and data roles.
- Ch. 3–4 fixed-ne population bridge and complete synthesis state.
- Ch. 5–8 complete continuum and line opacities.
- Ch. 9 transfer and flux semantics.
- Movement 10A invariant/star-state architecture and caches.

### Reads / writes contract

Reads:

- either a validated schema-v4 mapping/archive or the four required physical depth columns plus
  abundance/microturbulence inputs;
- public `wavelength_start_nm`, `wavelength_end_nm`, `resolution`, `device`, `dtype`, and
  `molecular_lines`;
- static tables, catalog roots, and cache policy.

Writes:

- host-float64 `SpectrumResult` after the single final device-to-host construction block; its four
  slab fields are `None` under the default `keep_slabs=False`;
- public `Spectrum`:
  - `wavelength_nm`;
  - `flux_total` as `F_lambda` per nm;
  - `flux_continuum` as `F_lambda` per nm;
  - `normalized_flux`;
  - elapsed seconds;
- optional validated structured-atmosphere archive when the public builder path is used;
- four-regime verification report.

### Derivation and prose arc

1. **Validate before compute.** Load version/metadata, apply only documented schema compatibility,
   reconstruct actual populations only when that documented input path requires it, then enforce shapes,
   finiteness, non-negativity, positive increasing column mass, and abundance length.
2. **Two structured inputs, one state.**
   - Existing schema-v4 archive/mapping enters directly.
   - In-memory physical columns enter the public builder, which computes mean nuclear mass, uses the
     fixed saved electron density, solves populations/molecules, constructs Doppler widths and
     continuum edge arrays, then validates the complete mapping.
3. **Fixed-ne bridge.** Re-state only the contract: the supplied `electron_density` is held fixed.
   Call the Chapter 3 implementation; do not rederive or rerun full charge closure.
4. **Build/reuse window invariants.** Request the Movement 10A bundle.
5. **Build star state.** Adapt schema fields to device tensors and replace the hydrogen template's
   merge state from this atmosphere.
6. **Run in exact order.**

   ```text
   continuum absorption/scattering
   -> allocate shared float32 line slab
   -> ordinary + PRD-routed LTE metal and autoionizing lines
   -> helium lines
   -> hydrogen/deuterium lines
   -> text-band + TiO molecular lines
   -> LTE Planck line source and zero standard line scattering
   -> stacked total/continuum transfer
   -> device-side context crop
   -> one final host-copy stage
   -> H_nu to public F_lambda per nm
   ```

7. **Inspect before wrap.** Plot process budgets and total/continuum `H_nu` for one compact window.
   Confirm normalized flux is computed from native total/continuum branches.
8. **Public record.** Convert `SpectrumResult` with `_surface_flux_per_wavelength_nm(...)` and wrap
   the exact five `Spectrum` fields. When the builder workflow is demonstrated, serialize and reload
   the structured atmosphere with `save_structured_atmosphere(...)`; do not invent a spectrum
   serializer.
9. **Four regimes.** Run the same chapter code—without branch-specific reimplementation—on hot,
   solar, low-gravity giant, and cool molecule-rich atmospheres.
10. **Scientific meaning.** Explain what same-atmosphere synthesis parity proves and what it does not:
    it validates the synthesis stage for the supplied atmosphere; it does not prove that an
    initialized or iterated atmosphere is physically accepted. Those workflows are completed later.

### Bite-size source-faithful code artifacts

1. `load_atmosphere_product_metadata(path)`, `load_atmosphere_npz(path)`, and
   `validate_atmosphere_npz(path)`.
2. `compute_mean_nuclear_mass_amu(elemental_abundances, atomic_masses=None)`.
3. `solve_population_state_at_electron_density(temperature, gas_pressure,
   elemental_abundances, *, tables, electron_density, mean_nuclear_mass_amu=None,
   mass_density=None, molecules=False, molecules_path=None)`.
4. `load_atomic_masses(path=None)`, `compute_doppler_per_ion(temperature, microturbulence,
   atomic_masses)`, and `_standard_microturbulence_column(microturbulence, n_depths, *,
   default_microturbulence_cm_s=200000.0)`.
5. Internal `pipeline.build_structured_atmosphere_from_columns(*, temperature, column_mass,
   gas_pressure, electron_density, elemental_abundances, mean_nuclear_mass_amu,
   microturbulence=None, eos_tables, electron_density_seed=None, tol=1e-5,
   atomic_masses=None, mass_density=None, molecular_species_codes=None, molecules_path=None)`.
6. Wrapper `synthesis.build_structured_atmosphere_from_columns(*, temperature, column_mass, gas_pressure,
   electron_density, elemental_abundances, mean_nuclear_mass_amu=None, microturbulence=None,
   mass_density=None, device=None, dtype=None, molecular_lines=True, eos_tolerance=1e-5)` from
   the engine module, returning the schema-v4 mapping.
7. Public `build_structured_atmosphere(*, temperature, column_mass, gas_pressure,
   electron_density, elemental_abundances, mean_nuclear_mass_amu=None, microturbulence=None,
   mass_density=None, molecular_lines=True, device=None, dtype=None, eos_tolerance=1e-5)`.
8. `save_structured_atmosphere(atmosphere, path)`.
9. `synthesize_structured_atmosphere(atmosphere, *, wavelength_start_nm=400.0,
   wavelength_end_nm=900.0, resolution=20000.0, molecular_lines=True, device=None,
   dtype=None, spectral_operator=None, window_invariants=None)`, returning
   `(SpectrumResult, seconds)`.
10. `SynthesisPipeline.run(keep_slabs=False, spectral_operator=None)`, returning `SpectrumResult`.
11. `_surface_flux_per_wavelength_nm(wavelength_nm, eddington_flux_per_frequency)`.
12. Public `synthesize(atmosphere_npz, *, wavelength_start_nm=400.0,
   wavelength_end_nm=900.0, resolution=20000.0, molecular_lines=True, device=None,
   dtype=None, spectral_operator=None)`.
13. Exact `SpectrumResult` and `Spectrum` dataclasses.
14. The four-regime same-atmosphere gate as a test module, not an invented public callable.

The capstone must call these exact functions. The visible composition cell calls
`synthesize_structured_atmosphere(...)` or `synthesize(...)`; it does not introduce a parallel
textbook-only synthesis entry point.

Exact source placement:

- schema load/metadata/validation:
  `/Users/ysting/payne-zero/payne_zero_synthesis/atmosphere.py:140`, `:257`, `:388`, and `:548`;
- structured builder:
  `/Users/ysting/payne-zero/payne_zero_synthesis/pipeline.py:221`, `:245`, and `:308`;
- engine boundary:
  `/Users/ysting/payne-zero/payne_zero_synthesis/synthesis.py:24`, `:39`, and `:90`;
- exact component order and final transfer:
  `/Users/ysting/payne-zero/payne_zero_synthesis/pipeline.py:1294`;
- public structured workflow and output:
  `/Users/ysting/payne-zero/payne_zero_synthesis/api.py:46`, `:344`, `:393`, and `:442`.

### Data

- Machine-readable atmosphere schema copied as static input.
- Four checksum-bound schema-v4 integration fixtures.
- Small catalog subsets sufficient for normal chapter execution.
- Optional full source root for complete-window gates.
- Per-stage and final pinned goldens stored under comparison-only paths.
- Verification metadata recording atmosphere identity, source commit, tables/catalogs, window,
  `resolution`, device, dtype, and cache state.

### Parity and physical gates

1. Schema field names, shapes, units, dtypes, finiteness, and monotone column mass pass.
2. Actual ion population equals normalized population times partition function where the schema
   identity applies.
3. Builder and archive paths produce the same structured mapping when given identical physical
   columns and fixed electron density.
4. Fixed-ne bridge preserves the supplied electron density exactly.
5. Every stage output has expected shape/device/dtype and no NaN/Inf.
6. Branch-separated continuum, metal, autoionizing, helium, hydrogen, and molecular slabs match
   chapter gates before summation.
7. H₂O remains absent from the standard full result unless the global runtime policy is deliberately
   changed and separately reviewed.
8. Total and continuum transfer plus normalized flux match Chapter 9.
9. Requested wavelength interior is exact after crop.
10. The final host-copy stage produces float64 public arrays with no earlier spectral-array transfer.
11. `save_structured_atmosphere(...)` round-trip preserves the structured-atmosphere arrays and
    schema/product metadata.
12. For each of the four regimes, compare:
    - `wavelength_nm`;
    - total `F_lambda` per nm;
    - continuum `F_lambda` per nm;
    - normalized flux.
13. Run each regime cold and warm; results are equal under the cache policy.
14. Run CPU, CUDA, and MPS where available using measured backend-specific tolerances.
15. Confirm public output is native synthesis only; no optional spectral operator, observed-grid
    response, fit, or instrument model is applied.

### Device and dtype behavior

- Archive validation and structured construction begin on host NumPy arrays.
- Invariants and star state move to the resolved device according to Movement 10A.
- Continuum uses work dtype; the shared line slab is float32; scattering iteration is float32; final
  flux weighting uses backend high precision.
- Context crop happens on device.
- The completed arrays cross to host only in the final result-construction stage and are wrapped as
  float64 public NumPy arrays.
- Use only the timing outputs that exist here: `WindowInvariants.build_profile`, prewarm-manifest
  timings, and `Spectrum.seconds`/the `seconds` returned with `SpectrumResult`. Any builder, bridge,
  or serialization timing shown in an inline experiment is labeled external instrumentation, not a
  new structured-synthesis return field. `ForwardTimings` remains deferred to the label workflow.

### No-redundancy and forward-reference audit

- No physics kernel is reimplemented in this chapter.
- The fixed-ne bridge is invoked, not rederived.
- Label-driven initialization, convergence, direct abundance, and CLI workflow details remain in
  Chapters 14–15 and the API appendix.
- Spectrum fitting and instrument response remain outside scope.
- The chapter ends with same-atmosphere synthesis parity only. It does not claim full label-to-
  converged-atmosphere workflow completion.

## Chapters 5–10 integration and acceptance audit

Before Chapters 5–10 move from design to drafting, the central integrator must confirm:

### Dependency closure

- Chapter 5 reads only state fields defined by Chapters 1–4 and completes the continuum contract
  needed by Chapters 6–10.
- Chapter 6 owns the ordinary line derivation used by Chapters 7 and 8.
- Movement 7A owns atomic source decoding, routing, grid selection, and generic sparse accumulation.
- Movement 7B completes all non-molecular special line branches.
- Chapter 8 consumes, but does not recompute, Chapter 4 chemistry.
- Movement 9A owns variable-depth optical-depth integration.
- Movement 9B owns scattering, fixed-grid transfer, and flux conversion.
- Movement 10A owns invariant/star-state/caching architecture.
- Movement 10B composes every prior artifact exactly once.

### Branch closure

The generated coverage report must contain explicit rows for:

- H⁻ bound-free and free-free;
- H I bound-free and free-free;
- H₂⁺ and He⁻;
- He I and He II continuum;
- C/Mg/Al/Si/Fe and light/hot metal continuum;
- H/He/H₂ Rayleigh and electron scattering;
- atmosphere-only CH/OH/H₂ CIA boundary;
- ordinary type-0 lines;
- type-3 PRD records through the ordinary LTE branch;
- type-1 autoionizing lines;
- type-2 COR parsed-but-unwired boundary;
- He I, He-3 I, and He II lines;
- H I and D I resolved lines;
- hydrogen merged pseudo-continuum;
- text molecular bands;
- TiO;
- H₂O compiler-only boundary;
- absorption-only transfer;
- scattering transfer;
- saturated-core fallback;
- total and continuum stacked solution;
- native `H_nu -> F_lambda` conversion.

No row may be closed merely because its parent module appears in prose.

### Code-size and pedagogy gate

- Each displayed cell has one conceptual purpose.
- Target length is 10–30 lines; 60 is the soft ceiling and 80 the hard ceiling.
- A long exact kernel is split into named stages without changing numerical order.
- Every new term is defined before code uses it.
- Every chapter opens with a physical observable/tension and ends with an output contract, remaining
  boundary, and exercises.

### Suggested exercises without new production branches

Exercises may vary physical controls, compare limiting cases, change safe chunk sizes, or inspect
component budgets. They must not ask the reader to:

- invent a `torch.compile` path;
- parallelize an ordered molecular-depth or transfer-depth recurrence;
- add H₂O to the standard pipeline without a reviewed feature decision;
- reinterpret type-3 records as a completed PRD solver;
- silently process type-2 COR records;
- build an instrument model or spectrum fitter;
- replace parity-pinned constants or reduction order in the name of cleanup.

### Final handoff from Chapter 10

The progressive package now accepts a complete structured atmosphere and produces a verified native
spectrum. Later chapters may construct a physical atmosphere or initialize one from labels, but they
must call the same Chapter 10 synthesis boundary. This is the only forward signpost needed at the end
of Part IV.
