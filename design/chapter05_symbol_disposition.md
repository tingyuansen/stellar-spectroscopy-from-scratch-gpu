# Chapter 5 continuum symbol disposition

Status: binding authoring and coverage control, not reader-facing prose  
Source boundary:
`src/payne_zero_atmosphere/continuum_opacity.py` and
`src/payne_zero_synthesis/continuum.py`

This inventory assigns every public top-level continuum class and function to
an exact execution lane, a physical owner, and one reader disposition. Its
purpose is to prevent complete source coverage from turning the chapter into
an API tour.

## Lane vocabulary

Only four execution lanes exist:

1. **atmosphere product** — direct CPU/Numba continuum evaluation on the
   30,000-point grid;
2. **synthesis product** — the standard edge-triplet
   `SynthesisPipeline.run()` route;
3. **sampled diagnostic** — caller-frequency evaluation without
   `FrequencyInvariants`;
4. **sampled extension** — caller-frequency evaluation with explicit
   `FrequencyInvariants`.

The atmosphere 343/344-point line-reference calculation is a **subroute of the
atmosphere product**, not a fifth lane. The default-off IFOP-19 Rosseland
surrogate is likewise an optional atmosphere-product subroute.

## Reader-disposition vocabulary

- **visible** — the exact symbol is introduced after its physical role is
  earned and is called or inspected in a reader-visible checkpoint.
- **progressive-only** — the exact implementation lives in the canonical
  progressive package; the chapter teaches its physics and named output but
  does not give the symbol a dedicated visible call.
- **test-only** — the exact branch is exercised for coverage and parity but is
  kept out of the main causal calculation.
- **inactive** — the public source object has no consumer in the audited
  Chapter 5 routes and is explicitly identified as inactive.
- **deferred** — the symbol's physical construction belongs to a later chapter;
  Chapter 5 may state only its present interface boundary.

No public symbol is omitted merely because its disposition is not visible.

## Atmosphere continuum public surface

All symbols in this section come from
`payne_zero_atmosphere.continuum_opacity`.

### Classes

| Exact symbol | Lane | Physical owner | Reader disposition | Rationale |
| --- | --- | --- | --- | --- |
| `ContinuumOpacityTableError` | atmosphere product | static-input validation | test-only | Missing archives and malformed required fields are failure-contract tests, not continuum physics. |
| `ContinuumOpacityTables` | atmosphere product | atmosphere cross sections, fits, and molecular-continuum tables | visible | The table object is inspected at the implementation boundary so stored units, shapes, and consumer roles are explicit. |
| `KarzasLatterTables` | atmosphere product | hydrogenic bound-free cross sections | visible | H I and He I/II cannot be built honestly without identifying the exact hydrogenic table object they consume. |
| `ContinuumLevelTables` | atmosphere product | unconsumed atomic-level archive surface | inactive | No audited atmosphere or synthesis continuum route calls this object, so it must not become an active dependency. |
| `MolecularEquilibriumTables` | atmosphere product | continuum-local H2 reconstruction | visible | The distinct 200-point atmosphere H2 partition table is central to the consumer-specific H2 policy. |
| `RosselandOpacityTable` | atmosphere product — optional IFOP-19 subroute | default-off Rosseland surrogate lookup | test-only | Chapter 5 pins the optional source behavior without presenting the surrogate as the standard continuum or as a Rosseland-mean derivation. |
| `ContinuumAtmosphereState` | atmosphere product | exact atmosphere continuum reads contract | visible | This object makes the packed atmosphere state, departure coefficients, and normalized-versus-actual population roles explicit. |

### Table and state adapters

| Exact symbol | Lane | Physical owner | Reader disposition | Rationale |
| --- | --- | --- | --- | --- |
| `load_continuum_opacity_tables` | atmosphere product | provenance-bound atmosphere continuum input | visible | The manifest preflight should load this exact archive before any component or golden output is evaluated. |
| `load_karzas_latter_tables` | atmosphere product | provenance-bound hydrogenic bound-free input | visible | A short table-identity checkpoint connects the H/He derivation to its exact static data. |
| `load_continuum_level_tables` | atmosphere product | unconsumed atomic-level archive surface | inactive | The loader remains source-covered but must not stage or imply an active table with no traced consumer. |
| `load_molecular_equilibrium_tables` | atmosphere product | provenance-bound H2 reconstruction input | visible | The H2 consumer-policy checkpoint needs the exact table identity and 200-point shape. |
| `build_continuum_atmosphere_state` | atmosphere product | `ModelAtmosphere`/`RuntimeState` to continuum-state adapter | visible | This is the exact boundary that proves the atmosphere lane does not consume the synthesis schema-v4 view alone. |

### Atmosphere line-reference subroute

| Exact symbol | Lane | Physical owner | Reader disposition | Rationale |
| --- | --- | --- | --- | --- |
| `build_continuum_reference_wavelength_grid` | atmosphere product — line-reference subroute | 343 physical reference wavelengths plus duplicate/sentinel | visible | The exact grid is required to explain the later line-strength threshold without teaching line selection early. |
| `active_continuum_reference_frequencies` | atmosphere product — line-reference subroute | effective-temperature-dependent active reference subset | visible | The five exact active counts and strict short-wavelength cutoff are pedagogically meaningful boundary checks. |
| `assemble_continuum_line_selection_threshold` | atmosphere product — line-reference subroute | `(D,344)` float32 continuum threshold | visible | This is the exact Chapter 5 output later consumed by atmosphere line selection. |

### H2 and light-particle components

| Exact symbol | Lane | Physical owner | Reader disposition | Rationale |
| --- | --- | --- | --- | --- |
| `compute_molecular_hydrogen_population` | atmosphere product | table-based continuum-local H2 density | visible | The chapter must execute the 19900/20000 K policy and distinguish this density from both Chapter 4 and schema-v4 H2. |
| `compute_hydrogen_opacity_columns` | atmosphere product | H I bound-free/tail, H II free-free, and H source numerator | visible | One explicit H-level construction grows into this exact component and exposes normalized versus actual population ownership. |
| `compute_helium_neutral_opacity_columns` | atmosphere product | He I bound-free, tail, and free-free | progressive-only | The solar/hot checkpoint reports its named output while the long level and autoionization implementation stays canonical. |
| `compute_hminus_opacity_columns` | atmosphere product | H-minus bound-free and free-free absorption | visible | H-minus is the first complete table-driven absorber and earns a direct threshold checkpoint. |
| `compute_molecular_hydrogen_ion_opacity_columns` | atmosphere product | H2-plus absorption | progressive-only | Its physical distinction from H2 CIA is taught, but the fitted polynomial branch belongs in the progressive component budget. |
| `compute_heminus_opacity_columns` | atmosphere product | He-minus free-free-like absorption | progressive-only | The fitted term is named and checked in the H/He budget without receiving a standalone implementation cell. |
| `compute_helium_ionized_opacity_columns` | atmosphere product | He II bound-free/tail and He III free-free | progressive-only | Its hot-regime behavior is visible through named outputs while the exact shell ladder remains in the canonical package. |
| `compute_light_element_continuum_columns` | atmosphere product | ordered H/He/light-particle absorption and source budget | visible | This exact aggregate closes the first physical movement without pretending to be the complete continuum. |

### Metal and molecular-continuum components

| Exact symbol | Lane | Physical owner | Reader disposition | Rationale |
| --- | --- | --- | --- | --- |
| `compute_carbon_neutral_opacity_columns` | atmosphere product | full C I continuum | progressive-only | The near-UV metal budget exposes its named contribution while the 25-level implementation remains canonical. |
| `compute_magnesium_neutral_opacity_columns` | atmosphere product | full Mg I continuum | progressive-only | The named component is retained for ordered-sum parity without a separate source-sized reader cell. |
| `compute_silicon_neutral_opacity_columns` | atmosphere product | full Si I continuum | progressive-only | The reader learns the bound-free family and exact population owner; the 33-level kernel remains packaged. |
| `compute_lukewarm_metal_opacity_columns` | atmosphere product | N I/O I/C II/Mg II/Si II/Ca II continuum group | progressive-only | The group and its population roles are visible in the budget, but its independent-frequency kernel is too large for direct display. |
| `compute_molecular_continuum_opacity_columns` | atmosphere product | CH/OH photodissociation and H2-H2/H2-He CIA | visible | The atmosphere-only feature boundary and its temperature/frequency seams require a direct exact checkpoint. |
| `compute_hot_metal_opacity_columns` | atmosphere product | hot-metal bound-free and charge-square free-free | progressive-only | The actual-versus-normalized perturbation and running one-percent acceptance rule are shown through named checkpoint outputs. |
| `compute_aluminum_neutral_opacity_columns` | atmosphere product | Al I continuum | progressive-only | The component is source-covered and retained in the IFOP-9 sum without expanding the narrative into another metal derivation. |
| `compute_iron_neutral_opacity_columns` | atmosphere product | Fe I threshold forest | progressive-only | Its exact 48-branch contribution remains named and parity-checked while the branch kernel stays in the package. |

### Optional Rosseland surrogate

| Exact symbol | Lane | Physical owner | Reader disposition | Rationale |
| --- | --- | --- | --- | --- |
| `create_rosseland_opacity_table` | atmosphere product — optional IFOP-19 subroute | empty surrogate-table construction | test-only | The helper is required for source coverage but does not advance the standard continuum argument. |
| `ingest_rosseland_opacity_table` | atmosphere product — optional IFOP-19 subroute | surrogate table population | test-only | A compact branch test preserves exact ingestion order without teaching it as the later physical Rosseland calculation. |
| `evaluate_rosseland_opacity` | atmosphere product — optional IFOP-19 subroute | nearest-quadrant surrogate interpolation | test-only | Its exact interpolation and empty-table behavior are pinned outside the main source lesson. |
| `compute_rosseland_continuum_opacity_columns` | atmosphere product — optional IFOP-19 subroute | frequency-independent surrogate absorption/source columns | test-only | The optional output is tested separately because its bolometric-like source convention is not the standard \(B_\nu\) continuum source. |

### Atmosphere product assembly

| Exact symbol | Lane | Physical owner | Reader disposition | Rationale |
| --- | --- | --- | --- | --- |
| `compute_continuum_opacity_columns` | atmosphere product | exact IFOP-controlled absorption, scattering, and source assembly | visible | This is the atmosphere parity destination and must be called only after every component and ordering rule is earned. |
| `compute_continuum_scattering_columns` | atmosphere product | Thomson plus H I/He I/H2 Rayleigh scattering | visible | A direct call keeps redirection separate from absorption and the thermal numerator. |
| `build_opacity_sampling_grid` | atmosphere product | effective-temperature-dependent 30,000-point wavelength grid and frequency weights | visible | The five start regimes and endpoint-weight rules define where the exact atmosphere product is evaluated. |

## Synthesis continuum public surface

All symbols in this section come from `payne_zero_synthesis.continuum`. A
shared entry lists every lane that genuinely consumes it; this does not create
additional lanes.

| Exact symbol | Lane | Physical owner | Reader disposition | Rationale |
| --- | --- | --- | --- | --- |
| `ContinuumTables` | synthesis product + sampled diagnostic + sampled extension | synthesis cross sections, Gaunt factors, Rayleigh tables, and device table views | visible | The exact table object and omitted-dtype exception are required at the host/device implementation boundary. |
| `FrequencyInvariants` | sampled extension | materialized frequency-only continuum state | visible | Its exact name and nonstandard status must be inspectable so it is never mistaken for a standard-pipeline cache. |
| `build_pops` | synthesis product + sampled diagnostic + sampled extension | structured-atmosphere to continuum population dictionary | visible | This adapter exposes the trimmed-view H II fallback and the normalized-versus-actual population ownership. |
| `pops_from_population_state` | sampled diagnostic + sampled extension | direct EOS-state to continuum population adapter | progressive-only | It is a valid exact adapter for isolated sampled studies but is not called by the standard structured-synthesis pipeline. |
| `build_frequency_invariants` | sampled extension | precomputed full frequency grids and alternate materialized helpers | visible | The extension checkpoint must construct it explicitly and record `coulomb_table_energy_first=True`. |
| `build_edge_sample_frequencies` | synthesis product | one-sided left/midpoint/right samples for all 340 edge intervals | visible | The edge schematic and bitwise packaged-vector check directly earn this exact function. |
| `compute_sampled_continuum` | sampled diagnostic + sampled extension | caller-frequency absorption, scattering, and LTE \(B_\nu\) source | visible | One call without invariants and one domain-bounded call with invariants prove the two secondary lanes are distinct. |
| `continuum` | synthesis product | used-interval sampling and log-parabolic reconstruction on `(D,W)` | visible | This is the standard synthesis parity destination reached by `SynthesisPipeline.run()`. |

## Coverage totals and acceptance rule

The inventory contains:

- 7 atmosphere classes and 31 atmosphere functions;
- 2 synthesis classes and 6 synthesis functions;
- 46 public symbols in total.

Chapter 5 symbol coverage is accepted only when:

1. every **visible** symbol appears after its physical meaning is earned;
2. every **progressive-only** symbol has a focused unit/component test and a
   named role in the ordered component budget;
3. every **test-only** symbol has a branch or failure-contract test without
   entering the standard continuum narrative;
4. every **inactive** symbol has a no-consumer trace and is absent from the
   active data/runtime bundle;
5. no line-reference or IFOP-19 helper is promoted into a fifth execution
   lane;
6. no atmosphere symbol is made to consume the synthesis schema-v4 view when
   the exact atmosphere adapter requires the packed runtime state.
