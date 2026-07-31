# Chapter 6 synthesis fixture, oracle, and publication plan

Status: design-only, implementation-ready; no oracle or golden is yet frozen  
Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

Scientific authorities:

- `design/chapter06_exact_source_contract.md`;
- `design/chapter06_causal_outline.md`;
- `design/chapter06_fe_record_candidate_audit.md`;
- staged byte-identical synthesis `atomic_lines.py` and `line_opacity.py`;
- the accepted Chapter 5 continuum-state fixture.

This plan owns only the six-depth synthesis lane. The separate 80-depth
atmosphere packing, fixture, oracle, and publication route are not inferred
from it. No atmosphere–synthesis slab identity is claimed.

## 1. Central decision: compose inputs instead of making a mixed fixture

Chapter 6 does **not** need a second archive containing copied Chapter 5
atmosphere state. Its exact synthesis input is a composition of three
independently owned artifacts:

| role | canonical artifact | identity |
| --- | --- | --- |
| upstream computed state | `data/fixtures/chapter05_continuum_states.npz` | SHA-256 `ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae` |
| one static source record | `data/subsets/chapter06_fe_i_source_row_873702.npz` | deterministic candidate SHA-256 `bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04` |
| invariant line tables | `data/static/synthesis_tables/line_profile_tables.npz` | SHA-256 `87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4` |

The Chapter 5 archive remains input-only. The Chapter 6 subset remains a
static one-record slice. The prospective golden remains comparison-only.
These roles must never be combined into one NPZ.

The subset identity above is accepted for oracle implementation only after
its builder, manifest record, 27-key schema, and exact source-row proof pass
independent review. Until then it remains a candidate identity. It contains:

- the exact 17 source fields, each shape `(1,)`, with source dtypes preserved;
- ten scalar provenance members;
- no derived wavelength, damping, invariant, opacity, or golden value.

The ten provenance members are
`builder_command`, `payne_zero_commit`, `source_archive_bytes`,
`source_archive_relative_path`, `source_archive_row_count`,
`source_archive_sha256`, `source_field_count`, `source_row_index`,
`subset_role`, and `subset_schema_version`.

There is therefore no new `data/fixtures/chapter06_*.npz` for the synthesis
lane. The phrase **synthesis fixture** below means the validated runtime view
formed from the existing Chapter 5 fixture, the one-record subset, and
deterministically constructed Chapter 6 arrays. It is a typed composition,
not another stored data role.

## 2. Frozen physical record and visibility criterion

The one physical record is zero-based source row `873702`, an ordinary Fe I
transition with:

| field | exact value |
| --- | ---: |
| `wavelength_nm` | `499.03411946178176` nm |
| `oscillator_strength` | `0.1380384264602885` (\(gf\), not bare \(f\)) |
| `lower_excitation_cm` | `33507.123` cm\(^{-1}\) |
| `atomic_number` | `26` |
| `ion_stage` | `1` |
| `line_type` | `0` |
| `radiative_damping` | `3.909296919359072e-08` |
| `stark_damping` | `4.700008650504819e-21` cm\(^3\) |
| `van_der_waals_damping` | `4.093536407068315e-24` cm\(^3\) |

All isotope corrections and energy shifts are zero, and all three damping
values are explicit. No default damping formula is entered.

The objective visibility requirement is:

> The exact production center gate must be active in at least one supplied
> depth in each of the four declared regimes.

It is not “active in every layer.” The accepted high-resolution activity masks
are:

```text
hot_dwarf            1 1 1 0 0 0
solar_dwarf          1 1 1 1 1 1
low_gravity_giant    1 1 1 1 1 1
cool_molecule_rich   1 1 1 1 1 1
```

These masks are physics evidence, not hand-selected plotting masks. The three
inactive hot-dwarf depths expose the exact pre- and post-FASTEX cutoff rather
than invalidating the line.

## 3. Exact four-regime state view

Regime order is immutable:

1. `hot_dwarf`;
2. `solar_dwarf`;
3. `low_gravity_giant`;
4. `cool_molecule_rich`.

Each regime has six layers in outermost-to-innermost order. From the Chapter 5
fixture, the line worker reads only these schema-v4 fields:

| exact field | axes | host dtype | unit |
| --- | --- | --- | --- |
| `temperature` | `(depth=6,)` | NumPy `float64` | K |
| `mass_density` | `(depth=6,)` | NumPy `float64` | g cm\(^{-3}\) |
| `electron_density` | `(depth=6,)` | NumPy `float64` | cm\(^{-3}\) |
| `hc_over_kt` | `(depth=6,)` | NumPy `float64` | cm |
| `hydrogen_neutral_population` | `(depth=6,)` | NumPy `float64` | cm\(^{-3}\) |
| `helium_neutral_population` | `(depth=6,)` | NumPy `float64` | cm\(^{-3}\) |
| `molecular_hydrogen_population` | `(depth=6,)` | NumPy `float64` | cm\(^{-3}\) |
| `partition_normalized_populations` | `(depth=6, ion_stage=6, species=139)` | NumPy `float64` | cm\(^{-3}\) per partition function |
| `fractional_doppler_widths` | `(depth=6, ion_stage=6, species=139)` | NumPy `float64` | dimensionless \(\Delta v_D/c\) |

Every read uses the exact fixture key
`{regime}__synthesis__{field}`. The fixture's complete 217-member schema,
payload digest, positive-domain checks, and regime order are validated before
selecting this view. The worker does not copy these arrays into a new fixture,
repair them, or recompute the equation of state.

The exact neutral-collision proxy is constructed in NumPy `float64`:

```python
collision_density_proxy = (
    hydrogen_neutral_population
    + 0.42 * helium_neutral_population
    + 0.85 * molecular_hydrogen_population
) * (temperature / 1.0e4) ** 0.3
```

It has axes `(depth=6,)` and unit cm\(^{-3}\). This use of the stored H2
population is the exact line-damping pipeline policy. It does not contradict
Chapter 5's separate statement that the standard continuum route locally
reconstructs H2 for its own terms.

The eight-key mapping passed to `accumulate_atomic` is:

| key | axes before device upload | exact upload policy on CPU/CUDA | exact upload policy on MPS |
| --- | --- | --- | --- |
| `partition_normalized_populations` | `(6,6,139)` | work `float64` | work `float32` |
| `fractional_doppler_widths` | `(6,6,139)` | work `float64` | work `float32` |
| `mass_density` | `(6,)` | work `float64` | work `float32` |
| `electron_density` | `(6,)` | `float32` | `float32` |
| `temperature` | `(6,)` | work `float64` | work `float32` |
| `hc_over_kt` | `(6,)` | work `float64` | work `float32` |
| `collision_density_proxy` | `(6,)` | `float32` | `float32` |
| `continuum_opacity` | `(6,6000)` | `float32` | `float32` |

No helium merge-weight field is supplied because the exact call sets
`do_helium=False`.

## 4. Continuum is recomputed before any line golden opens

The Chapter 5 golden is never an input. For each regime, the worker constructs
the exact standard 18-field continuum view:

```text
temperature
mass_density
electron_density
hydrogen_partition_normalized_ion_stage_populations
hydrogen_neutral_population
helium_neutral_population
helium_singly_ionized_population
carbon_partition_normalized_ion_stage_populations
magnesium_neutral_partition_normalized_population
aluminum_neutral_partition_normalized_population
silicon_neutral_partition_normalized_population
iron_neutral_partition_normalized_population
partition_normalized_populations
ion_stage_populations
signed_continuum_edge_frequency_hz
continuum_edge_wavelength_nm
continuum_edge_midpoint_wavelength_nm
continuum_edge_interval_width_squared_over_two_nm2
```

It then calls the exact standard edge-triplet synthesis continuum:

```python
continuum_absorption, continuum_scattering = continuum(
    wavelength_grid_nm,
    continuum_atmosphere,
    continuum_tables,
)
continuum_opacity = continuum_absorption + continuum_scattering
```

Before entering the line kernel:

- absorption and scattering each have `(depth=6, wavelength=6000)` axes;
- CPU work and captured host arrays are `float64`;
- values must be finite and nonnegative;
- the sum is formed in exact source order;
- `accumulate_atomic` performs the required conversion of the cutoff slab to
  `torch.float32`.

The ephemeral raw capture retains the absorption, scattering, float64 total,
and exact float32-fenced total. The final line golden does not publish those
full continuum slabs because they are recomputable Chapter 5 products and
would turn the line golden into a second continuum golden. It retains only
the center/wing cutoff ledger, continuum digests, and source/table identities.

## 5. Exact wavelength grids

### 5.1 Canonical teaching and oracle grid

The canonical grid is built by the exact synthesis `Grid`:

```python
Grid(
    start_wavelength_nm=495.0,
    end_wavelength_nm=505.0,
    resolution=300000.0,
).build()
```

Its frozen contract is:

| property | value |
| --- | ---: |
| axis | `(wavelength=6000,)` |
| dtype | NumPy `float64` |
| unit | nm |
| first sample | `495.0009387906341` |
| last sample | `504.9989209057178` |
| exact line center/wing-anchor index | `2434` |
| center grid sample | `499.0333758196059` nm |

The line is more than 2,400 pixels from either boundary, while the maximum
audited reach is 163 samples. No wing is clipped by the declared window.

The grid is constructed, then checked against these values. It is not loaded
from a fixture or golden. The final golden may own one copy of this axis
because it is a comparison coordinate, but no per-regime copies are allowed.

### 5.2 Coarse-grid stability probe

The same fixed 495–505 nm interval is also built with exact
`Grid.resolution=20000`. The
result has 400 samples and maps the line center to
`499.04420746429804 nm`. It is a robustness probe, not the chapter's profile
grid.

The raw oracle runs the complete four-regime line route on this grid and
requires the same activity masks as the 6000-point grid. The final golden
stores only:

- grid count, first/last sample, center index and center wavelength;
- a SHA-256 of the complete coarse grid;
- the `(regime=4, depth=6)` boolean activity mask;
- peak/reach summary values.

It does not store a second four-regime coarse opacity slab.

## 6. One-record mapping into `precompute_invariants`

The one-row subset is transformed in memory by the exact `_build_records`
path. The derived mapping passed to `precompute_invariants` contains exactly
one physical record and these per-line arrays:

| exact key | axes | host dtype | unit/value |
| --- | --- | --- | --- |
| `line_type` | `(line=1,)` | `int64` | `0` |
| `atomic_number` | `(line=1,)` | `int64` | `26` |
| `ion_stage` | `(line=1,)` | `int64` | `1` |
| `wavelength_nm` | `(line=1,)` | `float64` | nm |
| `index_wavelength_nm` | `(line=1,)` | `float64` | nm |
| `oscillator_strength` | `(line=1,)` | `float64` | dimensionless \(gf\) |
| `lower_excitation_cm` | `(line=1,)` | `float64` | cm\(^{-1}\) |
| `radiative_damping` | `(line=1,)` | `float64` | normalized by \(12.5664\nu_l\) |
| `stark_damping` | `(line=1,)` | `float64` | cm\(^3\), normalized by \(12.5664\nu_l\) |
| `van_der_waals_damping` | `(line=1,)` | `float64` | cm\(^3\), normalized by \(12.5664\nu_l\) |
| `raw_radiative_damping_log` | `(line=1,)` | `float64` | source \(\log_{10}\gamma\) |
| `raw_stark_damping_log` | `(line=1,)` | `float64` | source \(\log_{10}\gamma\) |
| `raw_van_der_waals_damping_log` | `(line=1,)` | `float64` | source \(\log_{10}\gamma\) |

The mapping also contains all five support entries required by the general
constructor:

| exact key | axes/dtype | exact content |
| --- | --- | --- |
| `helium_line_type` | `(helium_line=0,) int64` | explicitly empty |
| `helium_line_center_cutoff_ratio` | scalar NumPy `float64` | `1e-3` |
| `harris_profile_h0_table` | `(doppler_offset=2001,) float64` | staged synthesis H0 authority |
| `harris_profile_h1_table` | `(doppler_offset=2001,) float64` | staged synthesis H1 authority |
| `harris_profile_h2_table` | `(doppler_offset=2001,) float64` | staged synthesis H2 authority |

The raw source subset is the authority for the first thirteen values. The
staged line-profile archive is the authority for the three Harris arrays.
Neither set is copied into an input fixture.

The mapping must have exactly one physical line. “One-record mapping” never
means omitting support entries from the production constructor.

## 7. Exact invariant output, including empty routes

On CPU/CUDA, `precompute_invariants` produces work `float64`; on MPS it
produces work `float32`. The following ordinary invariants must have length
one:

| invariant | axes | dtype on CPU |
| --- | --- | --- |
| `metal_catalog_index` | `(metal_line=1,)` | Torch `int64` |
| `metal_classical_strength` | `(metal_line=1,)` | Torch `float32` |
| `metal_lower_excitation_cm` | `(metal_line=1,)` | Torch `float64` |
| `metal_radiative_damping` | `(metal_line=1,)` | Torch `float32` |
| `metal_stark_damping` | `(metal_line=1,)` | Torch `float32` |
| `metal_van_der_waals_damping` | `(metal_line=1,)` | Torch `float32` |
| `metal_wavelength_nm` | `(metal_line=1,)` | Torch `float64` |
| `metal_population_ion_stage_index` | `(metal_line=1,)` | Torch `int64`, value `0` |
| `metal_population_element_index` | `(metal_line=1,)` | Torch `int64`, value `25` |
| `metal_center_index` | `(metal_line=1,)` | Torch `int64`, value `2434` |
| `metal_wing_index` | `(metal_line=1,)` | Torch `int64`, value `2434` |
| `metal_center_clamped` | `(metal_line=1,)` | Torch `int64`, value `2434` |
| `metal_wing_clamped` | `(metal_line=1,)` | Torch `int64`, value `2434` |

Every autoionizing invariant must exist and have length zero:

```text
auto_catalog_index
auto_oscillator_strength
auto_lower_excitation_cm
auto_radiative_damping
auto_stark_damping
auto_van_der_waals_damping
auto_wavelength_nm
auto_population_ion_stage_index
auto_population_element_index
auto_center_index
auto_center_clamped
```

Their dtypes follow the exact dataclass construction: indices are `int64`;
oscillator and damping arrays are `float32`; excitation and wavelength use the
backend work dtype.

Every helium invariant must likewise exist and have length zero:

```text
helium_classical_strength
helium_lower_excitation_cm
helium_radiative_damping
helium_stark_damping
helium_van_der_waals_damping
helium_wavelength_nm
helium_population_ion_stage_index
helium_population_element_index
helium_center_index
helium_line_type
```

Indices and `helium_line_type` are `int64`; strength/damping arrays are
`float32`; excitation and wavelength use the work dtype. The scalar
`helium_cutoff` remains `1e-3`.

The three Harris tables and two FASTEX tables each have their exact
`(2001,)` and `(1001,)` axes. They are verified against staged authorities but
not duplicated into the final golden. The final artifact records their hashes
and the empty invariant arrays because the empties are part of the exact
general-constructor contract.

## 8. Oracle calls and line-opacity lifecycle

For each regime and grid, the exact canonical calls are:

```python
invariants = precompute_invariants(
    one_record_mapping,
    wavelength_grid_nm,
    runtime_device=torch.device("cpu"),
)

gross_batched = accumulate_atomic(
    invariants,
    state,
    do_metal=True,
    do_helium=False,
    apply_stim=False,
    wing_mode="batched",
)

net_batched = accumulate_atomic(
    invariants,
    state,
    do_metal=True,
    do_helium=False,
    apply_stim=True,
    wing_mode="batched",
)

gross_loop = accumulate_atomic(
    invariants,
    state,
    do_metal=True,
    do_helium=False,
    apply_stim=False,
    wing_mode="loop",
)

net_loop = accumulate_atomic(
    invariants,
    state,
    do_metal=True,
    do_helium=False,
    apply_stim=True,
    wing_mode="loop",
)
```

No output buffer or host accumulator is supplied.

All four outputs have axes `(depth=6, wavelength=6000)`, unit
cm\(^2\) g\(^{-1}\), device CPU, and Torch `float32`. The output contract is
not weakened to float64 merely because the surrounding work is float64.

The worker must show:

1. `gross_batched` is the pre-stimulated line deposit;
2. `net_batched` equals one exact wavelength-dependent float32 stimulated
   multiplication of the gross slab;
3. stimulation is not present in the center cutoff decision;
4. the empty auto route runs as a no-op within `do_metal=True`;
5. the helium route is not invoked;
6. CPU loop and batched slabs are bitwise equal for both gross and net routes.

The accepted candidate audit measured zero loop-versus-batched difference in
all four regimes. The canonical fixture therefore requires bitwise equality,
not a tolerance. A future nonzero difference blocks publication and triggers
a source/order audit.

## 9. Factor and branch ledger

The in-memory oracle records, for every `(regime=4, depth=6)` pair:

| member family | dtype | unit/convention |
| --- | --- | --- |
| selected normalized Fe I population | `float64` | cm\(^{-3}\) per partition function |
| fractional Doppler width | `float64` | dimensionless |
| Doppler width | `float64` | nm |
| mass density | `float64` | g cm\(^{-3}\) |
| pre-excitation strength | CPU work `float64` | cm\(^2\) g\(^{-1}\), pre-profile scale |
| lower excitation exponent | `float64` | dimensionless |
| FASTEX weight | `float64` | dimensionless |
| post-FASTEX line amplitude | `float64` | cm\(^2\) g\(^{-1}\), pre-profile scale |
| center continuum cutoff | CPU work `float64` from float32 input | cm\(^2\) g\(^{-1}\) |
| pre/post cutoff masks | boolean | exact branch decision |
| radiative damping term | `float64` | dimensionless |
| Stark damping term | `float64` | dimensionless |
| van der Waals damping term | `float64` | dimensionless |
| total damping and `damping_ratio` | `float64` | dimensionless |
| center profile | `float64` | dimensionless Harris value |
| gross/net center opacity | `float32` | cm\(^2\) g\(^{-1}\) |
| stimulated center factor | `float32` | dimensionless |
| exact wing reach | `int64` | pixel count from center |
| nonzero output count | `int64` | count |

The three damping terms must independently reconstruct total damping. The
pre- and post-cutoff masks must reconstruct the accepted activity masks. For
each active layer the nonzero count must be `2 * reach + 1` on the unclipped
canonical grid.

The worker also records scalar invariant uploads:

- host and float32 `classical_line_strength`;
- host and float32 normalized damping values;
- center and wing indices;
- work, accumulation, cutoff, and stimulation dtypes.

This is enough for the chapter to expose the numerical fences without
publishing a general catalog or a special-profile trace.

## 10. Scientific worker contract

Prospective path:
`scripts/chapter06_synthesis_oracle_worker.py`.

The worker:

- is an in-memory scientific observer;
- accepts only the canonical repository fixture and subset paths;
- reads no Chapter 6 golden;
- has no output-path, serialization, or publication option;
- imports the pinned Payne Zero checkout only after all pre-import checks;
- returns object-free NumPy arrays plus scalar/string metadata;
- prints only a compact JSON summary from its CLI;
- records `meta__golden_publication_performed=False`.

Before import it verifies:

1. the exact read-only source root and pinned commit;
2. hashes of every directly or transitively loaded synthesis Python module;
3. exact loaded-module-set equality after every route;
4. the staged/upstream byte identity of `atomic_lines.py`, `line_opacity.py`,
   `continuum.py`, `constants.py`, `device.py`, and required dependencies;
5. the exact Chapter 5 fixture path, SHA-256, 217-member schema, regime order,
   and physical payload digest;
6. the exact Chapter 6 subset path, SHA-256, 27-member schema, source row,
   raw source-archive identity, and source dtypes;
7. byte equality of staged/upstream continuum, edge-grid, and line-profile
   tables;
8. canonical CPU Torch work `float64` and accumulation `float32`;
9. one thread for Torch, BLAS, OpenMP, NumExpr, OpenBLAS, Accelerate, and
   Numba;
10. `LC_ALL=C`, `TZ=UTC`, `PYTHONHASHSEED=0`,
    `PYTHONNOUSERSITE=1`, and `PYTHONDONTWRITEBYTECODE=1`;
11. an external, empty, nonsymlink cache directory.

The exact loaded-source count and sorted manifest digest are implementation
outputs. They must be frozen only after an independent import audit; this plan
does not invent them.

### Raw capture contents

The ephemeral raw capture retains everything needed to audit construction:

- complete fixture/subset/source/table/process identities;
- the exact grid and coarse-grid probe;
- the 18-entry one-record mapping;
- all ordinary, auto, and helium invariant fields;
- all selected state arrays and the derived collision proxy;
- continuum absorption, scattering, float64 total, and float32 line-input
  total;
- full stimulated-emission factor;
- the factor/branch ledger;
- gross/net loop and batched slabs;
- activity, reach, nonzero-count, finite, and nonnegative checks;
- post-route loaded-source verification.

It computes a physical-payload fingerprint over every nonidentity array and a
full-capture fingerprint that also binds source, data, environment, worker,
and contract identities. Only the self-referential fingerprint fields are
excluded.

The raw capture's schema version, exact key count, schema digest, two
fingerprints, and worker hash are not guessed here. They are frozen after two
independent fresh-process candidates agree and a reviewer accepts the field
inventory.

## 11. Final comparison-only golden

Prospective publication directory:

`data/golden/payne_zero/chapter06/synthesis`

It contains exactly one file:

`chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz`

The lane-specific subdirectory permits a later independently reviewed
atmosphere publication without weakening this publisher's no-overwrite rule.

The final artifact owns:

- one copy of the canonical 6000-point wavelength axis;
- regime names and depth indices;
- the derived 13-field one-line mapping and five required support identities,
  with Harris arrays represented by exact static-file/member hashes;
- all length-one ordinary invariant fields;
- every exact length-zero auto and helium invariant field;
- the comparison-only derived portion of the `(4,6)` factor/branch ledger:
  excitation exponent/weight, pre/post-FASTEX strengths, cutoff values and
  masks, three damping contributions, total damping, damping ratio, center
  profile, gross/net center opacity, stimulated center factor, reach, and
  nonzero count;
- one stacked gross slab `(regime=4, depth=6, wavelength=6000)` `float32`;
- one stacked net slab with the same axes/dtype;
- activity masks, reach, nonzero counts, continuum center/wing samples, and
  stimulated-factor checks;
- compact coarse-grid stability evidence;
- source, fixture, subset, table, environment, worker, contract, schema, and
  fingerprint metadata.

It does **not** own:

- any copied Chapter 5 fixture array;
- the raw selected population, fractional width, mass density, temperature,
  electron density, or collision proxy retained by the ephemeral audit
  ledger;
- a full continuum slab;
- a second copy of the Harris tables;
- duplicate loop and batched slabs;
- a second coarse-grid line slab;
- any atmosphere selected-line input or output;
- any hydrogen, helium, autoionizing, molecular, or many-line result;
- the raw source catalog;
- an ephemeral raw-capture archive.

Loop arrays are discarded only after bitwise equality with batched arrays is
proved. Continuum and stimulation arrays are discarded only after their
digests, center/wing samples, and exact reconstructions pass.

The final archive is an uncompressed deterministic NPZ with lexical member
order, fixed ZIP metadata, and no object dtype. A 4 MiB hard size ceiling is
appropriate: the two required float32 slabs total about 1.1 MiB. Exceeding
the ceiling indicates duplicated fixture, continuum, route, or table data.

The final member set, key count, schema digest, byte size, and SHA-256 are
frozen only after independent candidate review.

## 12. Golden use rules

The golden is a comparison target only.

- The notebook first loads the manifest-bound subset and fixture.
- It derives the one-record mapping and recomputes the Chapter 5 continuum.
- It builds the dense teaching result and exact production result.
- It completes local dimensional, profile, cutoff, dtype, and stimulation
  checks.
- Only then may it open the compact Chapter 6 synthesis golden.

The notebook never uses the golden to choose:

- the line, wavelength window, grid resolution, active depth, regime, plot
  limits, cutoff, wing mode, dtype, device, or tolerance;
- a state value, continuum value, line invariant, or stimulated factor;
- whether a branch should be active.

Reader code never imports the pinned external checkout. It never opens a raw
capture or an atmosphere integration artifact.

Strict tests may use every final member. The main text reports compact parity
results and interpretable ledgers rather than printing a source dump.

## 13. CPU, CUDA, and MPS comparison strategy

The canonical golden is CPU-only:

- work dtype `float64`;
- line strengths and normalized damping uploads `float32`;
- continuum cutoff, electron density, and collision proxy `float32`;
- accumulation and stimulated multiplication `float32`.

CPU fresh-process A/B captures must be byte-identical. CPU loop and batched
routes must be bitwise identical for this fixture.

CUDA and MPS are backend comparisons, not alternate authorities:

| backend | work dtype | accumulation | required structural checks |
| --- | --- | --- | --- |
| CUDA | `float64` | `float32` | exact mapping, indices, activity mask, finite/nonnegative output, matching stimulation lifecycle |
| MPS | `float32` | `float32` | exact mapping and indices, declared float32 work, finite/nonnegative output, matching stimulation lifecycle |

For every available backend, tests record separately for gross and net:

- maximum absolute error;
- maximum relative error over the union of nonzero cells;
- maximum float32 ULP distance where sign and finiteness permit it;
- zero-pattern disagreement count;
- activity-mask disagreement count;
- reach disagreement count;
- center-opacity and integrated dense-profile diagnostics.

No phrase such as “machine precision” is allowed. No one tolerance is shared
between CUDA and MPS.

Tolerance freezing is a two-stage gate:

1. an implementation candidate reports the complete elementwise metrics from
   at least two repeated runs on each available backend, with device name,
   architecture, Torch version, driver/runtime version, and work dtype;
2. an independent review freezes explicit `atol` and `rtol` per backend and
   per gross/net quantity before chapter acceptance.

The reviewed tolerance must be no smaller than observed repeat variability
and no broader than needed to cover the accepted comparison. Structural
branch masks are assessed separately and are not excused by a numeric
tolerance. If hardware is unavailable, the result is recorded as
`unavailable`; it is not silently passed and does not block publication of
the canonical CPU golden. It remains an open backend gate for final
whole-book acceptance.

## 14. Publisher lifecycle

Prospective publisher:
`scripts/build_chapter06_synthesis_golden.py`.

Prospective scientific acceptance:
`design/chapter06_synthesis_oracle_acceptance.md`.

Prospective detached publication authorization:
`design/chapter06_synthesis_publication_acceptance.json`.

### 14.1 Two fresh captures

The publisher creates unrelated temporary roots `capture-a` and `capture-b`.
Each child receives:

- an absent or empty nonsymlink cache path;
- the frozen one-thread environment;
- canonical CPU Torch `float64`;
- its own internal deterministic raw NPZ;
- its own independently assembled final candidate.

The child imports the scientific worker, calls its in-memory result function,
validates the accepted raw schema/fingerprints, and serializes only to the
publisher's private temporary path. The worker itself never receives a path.

The publisher requires:

1. raw A/B byte identity;
2. independent final A/B assembly;
3. final A/B byte identity;
4. complete raw-to-final ownership coverage;
5. bitwise loop/batched equality before route deduplication;
6. exact reconstruction of net from gross and the stimulated factor before
   stimulated-factor deduplication;
7. final schema, science, role, size, and no-duplication validation.

A second-capture failure, byte mismatch, schema drift, or science failure
prevents candidate formation and leaves the destination untouched.

Populated-cache and symlink-cache invocations must fail before importing
Payne Zero. No retry silently replaces them.

### 14.2 Independent worker and candidate review

The scientific worker acceptance record binds:

- two accepted fresh-cache summaries;
- the exact command and environment;
- worker, source-contract, fixture, subset, and table identities;
- raw schema version/count/digest and both fingerprints;
- negative cache results;
- independent scientific disposition.

The candidate artifact then receives a separate review that:

- classifies every raw member as final, derived/digest-only, or intentionally
  ephemeral;
- reconstructs both final slabs from the accepted raw capture;
- verifies no input-state or continuum duplication;
- verifies all units, axes, dtypes, and ownership notes;
- freezes the final artifact schema and bytes.

Neither review alone authorizes publication.

### 14.3 Cycle-free detached authorization

The strict JSON authorization record binds:

- exact publisher path and reviewed SHA-256;
- this design/contract path and reviewed SHA-256;
- oracle-worker acceptance path and SHA-256;
- deterministic NPZ writer path and SHA-256;
- exact final candidate filename, path, SHA-256, byte size, archive kind,
  schema version, key count, and schema digest;
- exact prepublication `data/MANIFEST.json` SHA-256, schema version, pinned
  commit, and proof that the synthesis golden entry is absent.

The authorization record is not hashed into the publisher or archive. This
avoids a publisher/record/artifact hash cycle.

The record parser rejects duplicate keys, unknown fields, missing fields,
wrong types, malformed hashes, nonfinite JSON values, non-UTF-8 bytes,
relative/noncanonical paths, symlinks at any path component, and nonregular
files.

`--verify-only` performs both captures, both assemblies, and all checks but
cannot publish. `--publish` remains disabled while the detached record is
absent or mismatched. There is no force, replace, repair, destination, or
alternate-root option.

### 14.4 Atomic first publication

The direct publication function independently repeats authorization,
publisher, manifest, candidate hash/size, and complete semantic validation.
The top-level driver cannot be used as a substitute for this boundary.

The publisher stages the single exact file inside a temporary
`synthesis` directory under the canonical Chapter 6 parent, fsyncs the file
and directory, validates again, and atomically installs the directory.

If `synthesis` already exists:

- it must contain exactly the expected file;
- the file must be byte-identical to the accepted candidate;
- it must pass complete validation;
- the result is an identical no-op.

Any different existing byte or member fails without modification. Partial
publication, per-file overwrite, merge, or repair is forbidden.

The manifest is synchronized in a separate reviewed step after successful
publication. The publisher never grants itself authority by rewriting the
manifest.

## 15. Manifest and provenance contract

The subset, staged tables, existing fixture, and final golden have separate
manifest entries.

The final golden entry records:

- role `golden`;
- exact path, SHA-256, byte size, builder, archive kind, and scope;
- pinned Payne Zero commit;
- fixture, subset, source-archive, source-code, continuum-table, edge-grid,
  and line-profile-table identities;
- worker, exact-source contract, this plan, worker acceptance, publisher, and
  deterministic-writer identities;
- raw and final schema identities and physical/full fingerprints;
- CPU, Python, NumPy, Torch, operating-system, architecture, and one-thread
  controls;
- every member's shape, dtype, axes, unit/convention, and ownership.

Required unit vocabulary includes:

| quantity | manifest unit/convention |
| --- | --- |
| wavelength | `nm` |
| frequency | `Hz` |
| lower excitation | `cm^-1` |
| ordinary number density | `cm^-3` |
| partition-normalized population | `cm^-3 per partition function` |
| mass density | `g cm^-3` |
| fractional Doppler width | `dimensionless delta_v_D_over_c` |
| normalized Stark/van der Waals damping | `cm^3 normalized by 12.5664*frequency_hz` |
| normalized radiative damping | `dimensionless normalized by 12.5664*frequency_hz` |
| line mass absorption | `cm^2 g^-1 sampled kappa_nu(c/lambda_i), not per nm` |
| cutoff | `cm^2 g^-1 with embedded 1e-3 center ratio` |
| profile/FASTEX/stimulation factors | `dimensionless` |
| indices/counts/masks | `zero-based index`, `count`, or `boolean` |
| hashes/fingerprints | `SHA-256 hexadecimal identity` |

Residuals retain the unit of the subtracted quantity. Empty arrays retain
their physical field's unit and exact zero-length axis; they are not called
unitless merely because they contain no value.

## 16. Acceptance gates

### A. Input roles and identity

- The exact Chapter 5 fixture and Chapter 6 subset are canonical regular
  files with accepted hashes and exhaustive manifests.
- The subset has exactly one source row, 17 raw fields, ten provenance fields,
  and no derived or output value.
- The worker opens no Chapter 5 or Chapter 6 golden before computation.
- No input array is copied into a new fixture or final golden.

### B. Record and mapping

- `_build_records` reproduces the exact wavelength, \(gf\), excitation,
  species/stage/type, and three normalized damping fields.
- No isotope, energy-shift, or default-damping branch changes the record.
- The mapping contains one physical line plus all five support entries.
- `oscillator_strength` is treated as \(gf\), with no second statistical
  weight.

### C. Grid and continuum

- The canonical grid has the exact 6000-point identity and center indices.
- The line's maximum reach does not approach a boundary.
- The standard 18-field continuum view is exact.
- Absorption and scattering are recomputed, finite, and nonnegative.
- The float64 continuum sum and float32 line-input fence are both recorded.
- The coarse-grid probe preserves the four activity masks.

### D. Invariants and empty routes

- Metal count is one, with population indices `(ion_stage=0, element=25)`.
- Auto and helium invariant arrays all exist with exact zero lengths/dtypes.
- H0/H1/H2 and FASTEX tables match their own staged authorities.
- No hydrogen, helium, autoionizing, molecular, or special-profile kernel
  contributes.

### E. Physics and deposition

- Factor-by-factor strength, FASTEX, Doppler, and damping ledgers reconstruct
  the center result.
- With the continuum and every other kernel input held fixed,
  electron-density and collision-proxy perturbation tests move only their
  respective damping terms.
- Pre/post cutoff masks reproduce the accepted activity masks.
- Gross and net slabs are `(4,6,6000)` float32, finite, and nonnegative.
- Stimulation is applied exactly once and after deposition.
- CPU loop and batched outputs are bitwise equal.
- Active-layer reach and first-below-cutoff behavior are exact.

### F. Determinism and raw coverage

- Two fresh CPU captures have identical raw bytes and fingerprints.
- Two independent final assemblies have identical bytes.
- Populated and symlink cache probes fail closed.
- Every raw member has a reviewed disposition.
- The final artifact has no object dtype and stays below 4 MiB.

### G. Backend comparison

- CPU is the only golden authority.
- Available CUDA and MPS runs report separate work dtypes and error metrics.
- Branch masks and reach are checked separately from numeric tolerance.
- Explicit backend tolerances are frozen only after measurement and
  independent review.
- Unavailable hardware is reported honestly.

### H. Publication and manifest

- Worker acceptance, candidate acceptance, and detached authorization are
  distinct reviewed records.
- `--verify-only` cannot write a golden.
- Direct publication repeats every authorization and candidate check.
- First publication is atomic and no-overwrite.
- Identical existing bytes are a validated no-op; different bytes remain
  untouched.
- The postpublication manifest owns every archive member with exact
  shape/dtype/axes/unit/ownership metadata.

### I. Reader separation

- Reader-built continuum and line calculations finish before opening the
  golden.
- The golden never supplies state, tolerances, branch choices, or plot design.
- The notebook uses only the compact synthesis golden; exhaustive raw capture
  remains unpublished.
- The atmosphere lane is compared only with its own future oracle.

## 17. Findings from the contract audit

The exact source contract and causal outline agree on the essential
synthesis-lane design:

1. the accepted Chapter 5 fixture already supplies every required physical
   state field, so duplicating it would violate data-role ownership;
2. the one-row source subset must remain static input and be transformed
   transparently into the 18-entry production mapping;
3. empty helium metadata is a required constructor input, while empty
   autoionizing and helium invariant fields are required observed outputs;
4. the exact cutoff uses recomputed Chapter 5 continuum cast to float32;
5. line deposition and stimulation remain float32 even under CPU/CUDA
   float64 work;
6. gross and net products must remain distinct because stimulation is applied
   after deposition;
7. loop/batched equality is exact for this accepted fixture and should be
   deduplicated only after proof;
8. CPU is the portable comparison authority, while CUDA and MPS require
   separately measured backend policies;
9. one compact synthesis golden is sufficient; publishing copied state,
   continuum slabs, Harris tables, or duplicate route slabs would add no
   scientific information;
10. the golden remains blocked until the subset, worker, raw schema,
    candidate bytes, detached authorization, and manifest all pass their own
    reviews.
