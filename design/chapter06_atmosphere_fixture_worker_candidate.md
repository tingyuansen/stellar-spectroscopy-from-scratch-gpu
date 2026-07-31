# Chapter 6 atmosphere fixture-worker candidate

Status: twice-repaired implementation candidate for safe-sequence steps 4–6
only; the repair re-audit closed P0-1 but found one residual P0-2 package-
initializer boundary, that boundary is repaired here, another independent
re-review is pending, and no fixture, oracle, golden, manifest entry, or
publication is authorized

Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

## Scope and separation

This pass implemented the read-only scientific worker:

```text
scripts/chapter06_atmosphere_fixture_worker.py
```

It executes the exact structured-solar fixed-electron-density population
handoff and line-disabled continuum preparation, converts the independently
accepted Fe I row, and constructs the proposed nineteen fixture arrays in
memory. It returns two deliberately separate objects:

1. `fixture_arrays`: exactly nineteen detached object-free NumPy arrays;
2. `ephemeral_evidence`: full upstream arrays, reconstructed projections,
   line slabs, source/read-set identities, configuration evidence, and
   fingerprints.

The worker has no output-path parameter or serialization/publication
function. Its CLI prints one JSON summary of the in-memory result. It did not
write `data/fixtures`, `data/golden`, or `data/MANIFEST.json`.

The accepted converter's predecessor hashes and candidate status were also
removed from `design/chapter06_atmosphere_converter_candidate.md`. That record
now names the accepted repaired converter, tests, and independent audit.

## Candidate implementation identities

| file | SHA-256 |
| --- | --- |
| `scripts/chapter06_atmosphere_fixture_worker.py` | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| `tests/test_chapter06_atmosphere_fixture_worker.py` | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| fixture-worker repair re-audit (REJECT) | `00025449d4d39358c99a4ac7e3da76fdcdbcaecfb1f208ebfaa2f507fbcb3fca` |
| first fixture-worker independent audit (REJECT) | `4f82540b4dad05c37b4c3eee6aa5400d490a2b1ca69130634879def0cdae95f7` |
| reconciled converter candidate record | `acae959e9b01f986a553e9806fa7f60c6bb770ce3f356fab11c2d7509b63d03a` |
| accepted converter | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| accepted converter tests | `254d796b7ab761ca806c372d0bcdd935067ff1a89b2acfebcfa3007fe3f549dc` |
| converter independent audit | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |

The fixture-capture schema version is `1`. The converter version is `1`.

## Pre-import authority boundary

At executable entry, before NumPy can initialize a thread pool, the worker
compares the inherited environment with the complete twelve-control capture
contract. It never sets, substitutes, or normalizes those controls. The trusted
parent launcher must supply the canonical values; a missing or changed value
fails immediately.

Before importing any pinned atmosphere module, the worker then:

1. resolves `.git/HEAD` and loose/packed refs without invoking a Git checkout,
   filter, or source-tree write;
2. requires the exact pinned commit;
3. hashes all 35 Python sources that the pinned package will load;
4. verifies the structured source, molecular catalog, and nine atmosphere
   tables by exact path, byte count, and SHA-256;
5. verifies the fixture plan and exact-source contract;
6. verifies the independently accepted converter, its tests, and its audit;
7. derives and hash-gates the complete three-source staged execution boundary:
   the implicitly executed package initializer plus both explicit atmosphere
   submodule imports, rejecting every symlink and requiring each staged and
   pinned path to match its separately accepted identity;
8. requires a caller-created, empty, nonsymlink Numba cache outside the Payne
   Zero, textbook, and paper trees;
9. rechecks all three staged dependencies immediately before starting the
   accepted converter in an isolated no-file child process, then requires
   its exact runtime three-source manifest, seven fields, member hashes,
   four-word payload, and non-authoritative observed-row declaration.

Only after these checks does the worker put the read-only pinned Payne Zero
root first on `sys.path` and import the atmosphere package. The post-route
loaded-source manifest must equal the same 35-entry preverified manifest.

The complete staged converter dependency closure bound into the full-capture
evidence is:

| executed dependency | accepted staged SHA-256 | accepted pinned SHA-256 |
| --- | --- | --- |
| `src/payne_zero_atmosphere/__init__.py` | `db7e0860918d9532e0728fa4b498343fd7cb728b2edb29a64ead3784c5882999` | `dbb7734cab4f3e98b9b88d1f1b5ec27afd02fd7fd003ba7931b43d3750049d61` |
| `src/payne_zero_atmosphere/line_catalog.py` | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` |
| `src/payne_zero_atmosphere/population_layout.py` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` |

The initializer is intentionally a one-line staged package boundary and is not
byte-identical to the full pinned package initializer. Both identities are
therefore pinned and checked independently. The two functional submodules
remain additionally required to be byte-identical to their pinned copies.

## Exact physical route

The worker reads only these physical columns from the pinned structured solar
source:

```text
temperature             (80,) float64
column_mass              (80,) float64
gas_pressure             (80,) float64
electron_density         (80,) float64
microturbulence          (80,) float64
elemental_abundances     (99,) float64
```

It constructs the mixed atmosphere abundance boundary with linear H/He and
`log10` metals, then requires an exact 99-element round trip through
`linear_elemental_abundances`.

The controlled configuration is:

```text
effective_temperature          5778.0 K
log_surface_gravity            4.44
temperature_iteration_index    1
pressure_iteration_enabled     false
enable_molecules               true
enable_convection              false
opacity flags 14 and 16        zero based, disabled
all continuum flags            unchanged
all AtmosphereOutput paths     None
```

The executed call chain is:

```text
prepare_structured_handoff_population_state
  -> prepare_opacity_state with both line flags disabled
  -> accepted one-record SelectedLineCatalog
  -> accumulate_selected_line_opacity with full arrays
  -> reconstruct three projected public-call arrays
  -> accumulate_selected_line_opacity with projections
```

The line-disabled opacity preparation returns an empty float64 line slab and
selected count zero. The separately controlled one-record call returns one
selected line and an `(80,30000)` float32 line slab.

## Exact nineteen-member fixture candidate

| fixture member | shape/dtype | C-byte SHA-256 |
| --- | --- | --- |
| `packed_wavelength_index` | `(1,) int32` | `c423f4ac4a3825c6fad5336a1e15c0038ab17087f930916ad550ad45b4990dfc` |
| `packed_species_slot` | `(1,) int16` | `c2126161f7488b7d198ea310da9a8694786a18a2317eea5e30379ee118d34743` |
| `lower_excitation_index` | `(1,) int16` | `9b5bf0b74e2e212b57bcb2b9f2712eaab4c6169595f4e82767793f4534365648` |
| `log_strength_index` | `(1,) int16` | `3fe821d54660a0c51b42d19a42571e208791b3c5ff6bc0cac16b6553a226515a` |
| `radiative_damping_index` | `(1,) int16` | `5cdf3b75730a5b45c3f24da2c1030103143981191d2844aa7374e948b9abaeea` |
| `stark_damping_index` | `(1,) int16` | `91d82e7b89ae43b29bb673bb416697d005e093d0011c9d7af501630c6a502141` |
| `van_der_waals_damping_index` | `(1,) int16` | `c0346f09a0362e8e9d29c01a9fc7c292a7395b524a90319bb921608b9fdb1b60` |
| `temperature` | `(80,) float64` | `7b76410e731a6772cb6de12f923aea8a26e345778dbab67db3e8938278ce270f` |
| `hc_over_kt` | `(80,) float64` | `fdbd300b88163af886fe2905a883fa319c32abcc89db1099e4b0df7965d8174e` |
| `electron_density` | `(80,) float64` | `1e36bdf5aac263da2eac448f1b4c91e43e48a48afc133c8a8ec5e15735c85c18` |
| `actual_population_slot_indices` | `(3,) int16` | `eff6cc5c731be5128ac078be458469978b6ac7c823665abd47098485291a5af2` |
| `actual_population_slot_values` | `(80,3) float64` | `ae17c6b96dc9eb824051707d83eca2bbdb22483af4c143b18d832326292e86ce` |
| `line_population_slot_zero_based` | scalar `int16` | `8d6dd9f330421e48e73e97c0819d55a9965bac90b1f2f36057b4c906d5791f05` |
| `partition_normalized_population_over_mass_density_and_fractional_doppler_width_at_line_slot` | `(80,) float64` | `8625e8d942892d142384d51fc0625701374f79dfa471a78ff593d88479b3056f` |
| `fractional_doppler_widths_at_line_slot` | `(80,) float64` | `f8018929e269db3e6b9a572d36b528bc0577a3a93bcc25166b277824ffb2e785` |
| `opacity_wavelength_grid_nm` | `(30000,) float64` | `8944f1dd701ba27f50d37a16e48ab9375e1bef1b444ed405b671ba91fde8132b` |
| `wavelength_bin_edges` | `(344,) int64` | `ae50e9c0bafdcbfc39e242fd5029bf53b9e712d43008c6cb78a4493b89272cd5` |
| `continuum_line_selection_threshold` | `(80,344) float32` | `76cb7ef18554149b61b97e32f2a69bbcdba3eb8b7e6b7da8ba3c69b8775b7293` |
| `effective_temperature` | scalar `float64` | `0ae804feb5a3896c0d30fd25ee9853a10ee08fed330969967ce16a4d07a7329b` |

The fixture schema digest is:

```text
f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698
```

No raw-subset field, full packed state, validator-only adapter field, or line
output appears among these nineteen members.

## Projection sufficiency

The full upstream arrays have exact shapes `(80,1006)`, dtype `float64`, and
hashes:

| full upstream array | C-byte SHA-256 |
| --- | --- |
| ion-stage populations | `53ff8546517e744862abf16c490f1a3b4bcd761d681f5fde0cfb8ad62e0812b6` |
| population / density / fractional-width support | `706c8db98522bacb0c428279a29b270ec53772174207cceadd27e701947030c0` |
| fractional Doppler widths | `23c57f545bdfd74af3dd62ef57b8f347570953ea50c4e19289a9aa9715beb868` |

Reconstructing only actual-population slots `[0,2,840]` and line-support/width
slot `350` gives:

| projected `(80,1006)` array | C-byte SHA-256 |
| --- | --- |
| ion-stage populations | `30fa30338c773905b57023bdd6b102c205569207346227128f4e1ce585500166` |
| population / density / fractional-width support | `60db3e4a9c56e347e764b46a7d34e3c33bc1e2256dd44041d0621c1953a63524` |
| fractional Doppler widths | `d93c56e13576a3a6dce3071026ca9479cd6a94de1518d68514ed1866c2656606` |

The full and projected calls both return selected count one, identical shape
and dtype, and bitwise-identical slabs. Their shared dense C-byte SHA-256 is:

```text
43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58
```

The output remains ephemeral evidence and is not a fixture member or golden.

## Validator-only placeholder non-ownership

The baseline adapter uses:

```text
rosseland_opacity       ones(80)
radiative_acceleration  zeros(80)
convective_flux         zeros(80)
convective_velocity     zeros(80)
```

An independent second route replaces all four with finite, nonconstant
profiles, including signed acceleration and velocity. Every one of the
nineteen fixture members remains bitwise equal, and the separately accumulated
line slab remains bitwise equal with the same hash. This closes the
placeholder-field non-ownership requirement for the fixed-state route.

## Complete dynamic data read set

The worker instruments every `numpy.load` reached during package import,
source adaptation, both fixed-state routes, continuum construction, and line
accumulation. The exact unique read set is:

```text
examples/data/sun_structured_atmosphere.npz
source_data_files/atmosphere_tables/continuum_opacity_tables.npz
source_data_files/atmosphere_tables/ionization_potential_tables.npz
source_data_files/atmosphere_tables/iron_group_partition_tables.npz
source_data_files/atmosphere_tables/isotope_tables.npz
source_data_files/atmosphere_tables/karzas_latter_tables.npz
source_data_files/atmosphere_tables/line_opacity_tables.npz
source_data_files/atmosphere_tables/molecular_equilibrium_tables.npz
source_data_files/atmosphere_tables/packed_level_metadata.npz
source_data_files/atmosphere_tables/special_partition_tables.npz
source_data_files/source_catalogs/lines/molecular_equilibrium_atmosphere.npz
```

Every path is below the pinned read-only root and has a preverified identity.
The post-route Python source manifest contains exactly the 35 preverified
modules. `continuum_level_tables.npz` is absent, as required for the inactive
Chapter 5 route. `observed_atomic_lines.npy` is also absent and was not
required or read.

Observed atmosphere row `780108` remains only the accepted converter's
`non_authoritative_packing_corroboration` label. It is not a worker input and
does not enter either physical fingerprint.

## Two fresh-process reproduction

Two distinct subprocesses were run with two distinct caller-created empty
external Numba caches and the exact deterministic controls. Both returned:

```text
fixture member count          19
ephemeral evidence count      89
loaded pinned Python sources  35
fixture schema digest
  f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698
fixture payload fingerprint
  f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663
full capture schema digest
  cdf470038e67301b4c19b0691e672cd97df3233a3decf1b88e32ce3ac0dc1371
full capture fingerprint
  a25875097c7084ffe2577de65c2913d8775f29613823d9e6e0ab0d9db4644654
```

The two processes also agreed on the complete nineteen-member schema, every
member hash, all six full/projected source hashes, line-output hash, read set,
projection decisions, placeholder decisions, the three accepted staged and
pinned converter dependency identities, the exact three-source child runtime
manifest, and no-write declarations.

The physical payload fingerprint binds the nineteen arrays, structured source
identity, accepted converter/version, three full-array hashes, three projected
hashes, selected count, line-output identity, and the projection/placeholder
decisions. The full fingerprint additionally binds all source/data/process
identities—including the accepted staged/pinned dependency pairs and actual
child loaded-source manifest—versions, full ephemeral arrays and slabs,
read-set evidence, configuration, and worker identity. Neither fingerprint
records the temporary cache path.

## Focused verification

Executed:

```text
ruff format \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py

ruff check \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py

PYTHONPATH=src:. pytest -q \
  tests/test_chapter06_atmosphere_fixture_worker.py
```

Result:

```text
Ruff: clean
12 passed in 30.91 s
```

The focused suite launches both fresh processes, compares all three requested
capture identities plus the fixture schema digest, verifies all nineteen
member schemas and hashes, verifies full/projection/placeholder parity,
requires the exact read set and 35-source manifest, snapshots every canonical
data file before and after, and statically excludes serialization and
publication calls. Seven additional fresh executable launches mutate each
thread control separately and require a nonzero exit before any Payne Zero
import. Three comment-only, AST-equivalent staged-source mutations prove that
the initializer and both submodules fail their exact-identity gates before the
converter child can start. The suite derives the complete execution closure
from explicit imports plus every implicitly executed package initializer,
recursively scans that closure for further staged imports, and requires it to
equal the child process's exact runtime loaded-source manifest. Separate
symlink mutations prove that every one of the three sources, including
`__init__.py`, fails before child startup.

## Remaining gates

This repaired worker and report are candidates, not self-acceptance. Both the
historical audit and its appended repair re-audit remain immutable rejection
records; neither was edited. Before any canonical fixture can exist:

1. an independent fixture re-review must first confirm P0-1 remains closed and
   the complete P0-2 initializer/submodule boundary is now closed, then audit
   the exact structured source, fixed-\(n_e\) route, nineteen-member ownership,
   full/projection parity, placeholder non-ownership, dynamic read closure,
   and fingerprints;
2. a separately reviewed deterministic fixture publisher must be implemented
   with detached authorization and atomic no-replace behavior;
3. the candidate fixture bytes must be assembled and independently audited;
4. only then may the accepted fixture be published and registered in
   `data/MANIFEST.json`.

The atmosphere oracle, serial seam suite, sparse golden, and golden publisher
remain later safe-sequence steps. Nothing in this candidate report authorizes
those products.
