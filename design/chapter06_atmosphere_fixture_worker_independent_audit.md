# Chapter 6 atmosphere-fixture-worker independent audit

Status: independent candidate review  
Disposition: **FINAL ACCEPT for in-memory safe-sequence steps 4–6; both
historical P0 findings are closed, while serialization and publication remain
unauthorized**  
Audited: 2026-07-30

## 1. Scope and immutable review snapshot

This audit reviewed, without editing implementation, tests, source data, the
pinned Payne Zero checkout, or the paper tree:

| reviewed file | lines | SHA-256 |
| --- | ---: | --- |
| `design/chapter06_atmosphere_fixture_oracle_plan.md` | 1175 | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| `design/chapter06_exact_source_contract.md` | 1371 | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |
| `scripts/chapter06_atmosphere_line_converter.py` | 619 | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| `scripts/chapter06_atmosphere_fixture_worker.py` | 1479 | `38acaf8137293f25047d6a3ce15ad7b6559227a5bded268ae7b445ef9c4b5a76` |
| `tests/test_chapter06_atmosphere_fixture_worker.py` | 302 | `ea37e5e999c8fefe73906520112db3c8fa11760ea2a7c05303b3d7a25b5bf778` |
| `design/chapter06_atmosphere_fixture_worker_candidate.md` | 305 | `63e0582ddadca86abb1d476815452d328f9cc439be90598452bcccfa02ce381e` |
| `design/chapter06_atmosphere_converter_candidate.md` | 192 | `acae959e9b01f986a553e9806fa7f60c6bb770ce3f356fab11c2d7509b63d03a` |
| `design/chapter06_atmosphere_converter_independent_audit.md` | 300 | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |
| `tests/test_chapter06_atmosphere_line_converter.py` | 585 | `254d796b7ab761ca806c372d0bcdd935067ff1a89b2acfebcfa3007fe3f549dc` |

The independently accepted converter audit authorizes only safe-sequence steps
1--3. Its exact converter, test, and audit hashes are correctly reconciled in
the current converter candidate and fixture worker.

The canonical raw subset remains:

```text
data/subsets/chapter06_fe_i_source_row_873702.npz
bytes    8665
SHA-256 bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04
```

Its manifest, builder, and accepted converter dependencies were unchanged:

```text
data/MANIFEST.json
  d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a
scripts/build_chapter06_fe_record_subset.py
  25bcf4662740155e8b08615b9522f3f4517e1a5ddc4627c68686620ccfff4d6c
```

## 2. Executive verdict

The scientific construction is numerically sound at this snapshot. Independent
execution reproduced the exact nineteen-member schema, fixed-electron-density
route, dynamic data read set, full-versus-projection bitwise parity,
placeholder-field nonownership, line-output hash, and four capture identities.
The worker also has no serializer, output-path, manifest, golden, or
publication surface.

The candidate nevertheless fails two source/process authority requirements:

1. the executable entry point silently replaces seven caller-supplied thread
   controls before validating them, so changed process controls do not fail;
2. the accepted converter child imports two staged atmosphere modules whose
   current bytes are correct, but the worker does not verify their accepted
   local hashes before that import.

The first defect was reproduced in a fresh process and is independently
dispositive. The second is a direct unclosed imported-dependency boundary.
Therefore the worker is **REJECTED for safe-sequence steps 4--6**. No fixture,
oracle, golden, manifest change, or publication is authorized.

## 3. Blocking findings

### P0-1 — Changed thread controls are normalized and accepted: FAIL

The plan requires every declared process control and explicitly requires a
changed process control to fail closed. The candidate also states that the
worker “requires every process control.”

The executable does something different. Before importing NumPy and before
`require_environment()` runs, lines 48--49 execute:

```python
if __name__ == "__main__":
    os.environ.update(THREAD_ENVIRONMENT)
```

That overwrites these seven caller inputs:

```text
MKL_DYNAMIC
MKL_NUM_THREADS
NUMBA_NUM_THREADS
NUMEXPR_NUM_THREADS
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
VECLIB_MAXIMUM_THREADS
```

`require_environment()` later observes only the substituted values. The full
capture therefore records the worker's normalized environment rather than the
actual process request.

An independent fresh-process mutation set:

```text
OMP_NUM_THREADS=7
NUMBA_NUM_THREADS=3
```

while keeping the other controls exact and using a new empty external cache.
The expected outcome was a nonzero exit before physics import. The actual
outcome was:

```text
return code                    0
stderr                         empty
worker accepted mutation       true
fixture payload fingerprint
  f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663
full capture fingerprint
  97072a0b3bdabf65fb2c4cf3012fb6633c60a4b2a2a43e6fff5a38967f73d71e
```

Both fingerprints equal the nominal candidate identities. Consequently the
full fingerprint cannot distinguish an authorized launch from this changed
process request.

The focused environment test does not exercise the executable boundary. It
calls `require_environment()` directly with an empty mocked mapping and checks
only one occupied-cache mutation. The two nominal subprocesses supply already
correct controls, so they do not expose the overwrite.

Required repair:

1. reject, rather than overwrite, a missing or changed thread control;
2. perform that rejection in the pre-NumPy bootstrap path;
3. add fresh-process mutation tests for each of the seven thread controls and
   retain the existing tests for the other controls and cache policy;
4. require nonzero exit and no Payne Zero import for every mutation;
5. recapture twice after the worker/test hashes change.

### P0-2 — The converter child's staged imports are not runtime hash-gated: FAIL

The accepted converter imports:

```text
src/payne_zero_atmosphere/line_catalog.py
src/payne_zero_atmosphere/population_layout.py
```

before performing the conversion. Their accepted and current hashes are:

```text
line_catalog.py
  2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92
population_layout.py
  36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0
```

Independent byte comparisons showed that each staged file is currently
identical to its pinned external counterpart. Import-origin isolation also
works: preloading the staged package in the fixture-worker process caused
`load_pinned_modules()` to reject it with `FixtureIdentityError`.

Current parity is not the same as a capture gate. The
`fixed_local_files` mapping in `verify_preimport_identities()` hashes the plan,
source contract, converter, converter tests, and converter audit, but it does
not hash these two staged imported modules. The 35-entry frozen manifest hashes
the external Payne Zero copies used later by the main fixture route, not the
staged copies used by the converter child. The child checks only that its
imports resolve below `LOCAL_SOURCE_ROOT`.

A byte-only mutation such as a comment change would therefore change an
accepted imported-source identity without changing the exact seven values or
four-word payload, and the present worker would accept it. This violates the
pre-import exact-source boundary even though the current files are correct.

Required repair:

1. pin and verify both staged file hashes before starting the converter child;
2. reject symlinks and confirm each staged file remains byte-identical to the
   corresponding already-verified pinned source;
3. add one mutation test per dependency that proves rejection occurs before
   the child import;
4. include these dependency identities in full-capture evidence.

## 4. Scientific and structural checks that pass

### 4.1 Pinned commit, source, and data: PASS

Independent HEAD resolution returned:

```text
9c44001feae40b85146630499e6f8a5fed42e5af
```

The worker's no-filter resolver, all 35 frozen external Python hashes, and all
11 declared data identities passed before the main physics import. Independent
spot hashes included:

```text
runner.py
  05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7
equation_of_state.py
  719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2
molecular_equilibrium.py
  4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86
doppler.py
  e118a78bf5250ef5e1f77d652c9e78fbb7b92acf5c069f717faed7a3b3ea98f0
continuum_opacity.py
  1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81
line_opacity.py
  d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6
sun_structured_atmosphere.npz
  d686ea7107d60bf1707607e3d6377d283fb3eb7115c170ac2aeef54fbaa6abdb
molecular_equilibrium_atmosphere.npz
  971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644
```

### 4.2 Converter result and non-authoritative witness: PASS

The accepted child reproduced conversion version one, all seven exact
shape/dtype/value/hash contracts, and native payload:

```text
[[12425352, 1370295734, 1628847268, 581578398]]
shape/dtype  (1,4) int32
SHA-256
  1769c9ad8d33e847a099bd6d50df85a2f478f98d554b2fc39db8121ba93158d2
```

Observed row `780108` remains labeled only
`non_authoritative_packing_corroboration`. Neither the converter nor the
fixture route reads `observed_atomic_lines.npy`, and no cross-catalog genealogy
or atmosphere/synthesis slab equality is claimed.

### 4.3 Exact fixed-electron-density route: PASS

The executed scientific chain matches the plan:

```text
ModelAtmosphere adapter
  -> prepare_structured_handoff_population_state(
       temperature_iteration_index=1
     )
  -> prepare_opacity_state(
       population_state=population,
       temperature_iteration_index=1
     )
  -> one accepted SelectedLineCatalog
  -> accumulate_selected_line_opacity with full arrays
  -> reconstruct exact projections
  -> accumulate_selected_line_opacity with projected arrays
```

The adapter uses exactly the six structured source arrays, the mixed H/He
linear and metal-logarithmic abundance boundary, and a bitwise 99-element
round trip. The setup is 80 depths, `5778.0 K`, `log(g)=4.44`, fixed electron
density, molecules enabled, convection disabled, and zero-based line flags 14
and 16 disabled. All three output paths in `AtmosphereOutput` are `None`.

The line-disabled opacity preparation has selected count zero and an empty line
slab. The separately controlled selected-line call has selected count one and
returns `(80,30000)` `float32`.

### 4.4 Exact nineteen-array schema: PASS

The candidate contains exactly the nineteen proposed members, no object dtype,
no raw-row field, no full packed state, no validator-only placeholder, and no
line output. Its schema digest is:

```text
f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698
```

Independent execution reproduced every frozen member hash:

| member | SHA-256 |
| --- | --- |
| `packed_wavelength_index` | `c423f4ac4a3825c6fad5336a1e15c0038ab17087f930916ad550ad45b4990dfc` |
| `packed_species_slot` | `c2126161f7488b7d198ea310da9a8694786a18a2317eea5e30379ee118d34743` |
| `lower_excitation_index` | `9b5bf0b74e2e212b57bcb2b9f2712eaab4c6169595f4e82767793f4534365648` |
| `log_strength_index` | `3fe821d54660a0c51b42d19a42571e208791b3c5ff6bc0cac16b6553a226515a` |
| `radiative_damping_index` | `5cdf3b75730a5b45c3f24da2c1030103143981191d2844aa7374e948b9abaeea` |
| `stark_damping_index` | `91d82e7b89ae43b29bb673bb416697d005e093d0011c9d7af501630c6a502141` |
| `van_der_waals_damping_index` | `c0346f09a0362e8e9d29c01a9fc7c292a7395b524a90319bb921608b9fdb1b60` |
| `temperature` | `7b76410e731a6772cb6de12f923aea8a26e345778dbab67db3e8938278ce270f` |
| `hc_over_kt` | `fdbd300b88163af886fe2905a883fa319c32abcc89db1099e4b0df7965d8174e` |
| `electron_density` | `1e36bdf5aac263da2eac448f1b4c91e43e48a48afc133c8a8ec5e15735c85c18` |
| `actual_population_slot_indices` | `eff6cc5c731be5128ac078be458469978b6ac7c823665abd47098485291a5af2` |
| `actual_population_slot_values` | `ae17c6b96dc9eb824051707d83eca2bbdb22483af4c143b18d832326292e86ce` |
| `line_population_slot_zero_based` | `8d6dd9f330421e48e73e97c0819d55a9965bac90b1f2f36057b4c906d5791f05` |
| `partition_normalized_population_over_mass_density_and_fractional_doppler_width_at_line_slot` | `8625e8d942892d142384d51fc0625701374f79dfa471a78ff593d88479b3056f` |
| `fractional_doppler_widths_at_line_slot` | `f8018929e269db3e6b9a572d36b528bc0577a3a93bcc25166b277824ffb2e785` |
| `opacity_wavelength_grid_nm` | `8944f1dd701ba27f50d37a16e48ab9375e1bef1b444ed405b671ba91fde8132b` |
| `wavelength_bin_edges` | `ae50e9c0bafdcbfc39e242fd5029bf53b9e712d43008c6cb78a4493b89272cd5` |
| `continuum_line_selection_threshold` | `76cb7ef18554149b61b97e32f2a69bbcdba3eb8b7e6b7da8ba3c69b8775b7293` |
| `effective_temperature` | `0ae804feb5a3896c0d30fd25ee9853a10ee08fed330969967ce16a4d07a7329b` |

The projection descriptors are exactly `[0,2,840]` and Fe I slot `350`; the
grid is strictly increasing and the final packed bin sentinel is `2**30`.

### 4.5 Full-versus-projection bitwise parity: PASS

The three full `(80,1006)` `float64` hashes are:

```text
ion-stage populations
  53ff8546517e744862abf16c490f1a3b4bcd761d681f5fde0cfb8ad62e0812b6
population/density/fractional-width support
  706c8db98522bacb0c428279a29b270ec53772174207cceadd27e701947030c0
fractional Doppler widths
  23c57f545bdfd74af3dd62ef57b8f347570953ea50c4e19289a9aa9715beb868
```

The exact projected reconstruction hashes are:

```text
ion-stage populations
  30fa30338c773905b57023bdd6b102c205569207346227128f4e1ce585500166
population/density/fractional-width support
  60db3e4a9c56e347e764b46a7d34e3c33bc1e2256dd44041d0621c1953a63524
fractional Doppler widths
  d93c56e13576a3a6dce3071026ca9479cd6a94de1518d68514ed1866c2656606
```

Both calls return selected count one, the same `(80,30000)` `float32`
contract, and `np.array_equal(full_output, projected_output)`. Their shared
dense C-byte hash is:

```text
43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58
```

The line slab remains ephemeral evidence rather than a fixture member or
golden.

### 4.6 Validator-only placeholder nonownership: PASS

The second route jointly replaces all four placeholders with finite,
nonconstant profiles, including signed acceleration and velocity. It
reproduces every one of the nineteen fixture member bytes and the exact line
slab. The evidence correctly records:

```text
placeholder fixture equality  true
placeholder line equality     true
```

### 4.7 Dynamic data read closure: PASS

In addition to the worker's `numpy.load` tracker, this audit installed an
independent Python open-event hook after authority/converter checks and before
the physical route. The exact pinned non-code data-open set was:

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

This equals the declared 11-entry dynamic read set. All paths are below the
pinned root and prehashed. Neither `continuum_level_tables.npz` nor
`observed_atomic_lines.npy` was opened. The post-route loaded Python manifest
contains the exact 35 preverified external sources.

### 4.8 Deterministic numerical capture: PASS, subject to P0-1

The focused suite launches two nominal fresh processes. This audit launched an
additional fresh process with an independent data-open hook. All nominal
captures agreed on:

```text
fixture members                 19
ephemeral evidence members      87
loaded pinned Python sources    35
fixture schema digest
  f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698
fixture payload fingerprint
  f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663
full capture schema digest
  396ccad382f23dd4a0eae6ac16d8cc0a9fbb55163f5403e26aedf9662efc6f51
full capture fingerprint
  97072a0b3bdabf65fb2c4cf3012fb6633c60a4b2a2a43e6fff5a38967f73d71e
```

These identities correctly reproduce the numerical snapshot, but the full
fingerprint is not acceptable as process authority until P0-1 is repaired.

### 4.9 No serializer or publication authority: PASS

The worker:

- accepts no CLI arguments or output path;
- returns detached in-memory arrays plus detached evidence;
- prints only one JSON summary;
- contains no `save`, `savez`, `savez_compressed`, `tofile`, `write_bytes`, or
  `write_text` call;
- contains no canonical fixture, golden, or manifest destination;
- reports all publication/write booleans as false;
- leaves the complete textbook `data/` snapshot unchanged across nominal
  captures.

The worker therefore has no fixture serializer, publisher, manifest writer, or
golden reader/writer authority.

## 5. Verification executed

```text
python -m pytest -q tests/test_chapter06_atmosphere_fixture_worker.py
10 passed in 28.59s

ruff check \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py
All checks passed!

ruff format --check \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py
2 files already formatted
```

Additional independent probes covered:

- exact pinned HEAD and critical source/data hashes;
- staged-versus-external byte parity for the converter's two atmosphere
  dependencies;
- rejection of an already-imported local atmosphere package;
- independent open-event reconstruction of the complete physical data read
  set;
- a third nominal fresh in-memory capture and fingerprint recomputation;
- a fresh executable process with mutated `OMP_NUM_THREADS` and
  `NUMBA_NUM_THREADS`;
- static absence of serialization and publication calls.

## 6. Re-audit boundary

The next candidate must have new worker and test hashes. It must repair both
blocking authority boundaries, rerun two nominal fresh captures with distinct
empty external caches, and add adversarial fresh-process tests that show:

```text
one changed thread control             -> reject before physics import
one changed staged converter dependency -> reject before converter import
```

The nineteen arrays and current scientific hashes may remain unchanged if the
repair is authority-only, but the full-capture identities must be recomputed
because the worker, tests, and evidence schema will change. No publication
work should proceed from this rejected snapshot.

## Repair re-audit

Re-audited: 2026-07-30  
Disposition: **REJECT; one of the two historical P0s is closed**

This section preserves the historical rejection above and evaluates only the
repaired authority boundary requested for the latest candidate. No worker,
test, candidate, source, data, external Payne Zero, paper, fixture, oracle,
golden, manifest, or publication file was edited by this re-audit.

### R1. Exact repair snapshot

| reviewed object | lines | SHA-256 |
| --- | ---: | --- |
| `scripts/chapter06_atmosphere_fixture_worker.py` | 1561 | `8e566fcbb899ac54b7af265a3f5907c0510808dc3dc17dd2165ec9170f1d8170` |
| `tests/test_chapter06_atmosphere_fixture_worker.py` | 464 | `57ca742142119f76c4706a8c6c1fb090b2a2bcf166e8869162f5849bd2b8ab63` |
| `design/chapter06_atmosphere_fixture_worker_candidate.md` | 332 | `908d8701ebf008da1933d0aad24307db870300c662fd389f5e54a13ae270f390` |
| historical independent audit before this append | 455 | `4f82540b4dad05c37b4c3eee6aa5400d490a2b1ca69130634879def0cdae95f7` |
| staged `line_catalog.py` | 345 | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` |
| staged `population_layout.py` | 204 | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` |
| staged package `__init__.py` | 1 | `db7e0860918d9532e0728fa4b498343fd7cb728b2edb29a64ead3784c5882999` |
| pinned package `__init__.py` | 158 | `dbb7734cab4f3e98b9b88d1f1b5ec27afd02fd7fd003ba7931b43d3750049d61` |

The worker, tests, and candidate identities agree with the repaired candidate
record. The staged `line_catalog.py` and `population_layout.py` bytes are still
exactly equal to their pinned counterparts. The package initializer is listed
because the converter child executes it as part of importing either submodule;
it is not currently listed by the worker.

### R2. P0-1 — Inherited thread controls: **CLOSED**

The overwrite that caused the historical rejection is gone.

The latest executable defines exactly these seven thread controls:

```text
MKL_DYNAMIC
MKL_NUM_THREADS
NUMBA_NUM_THREADS
NUMEXPR_NUM_THREADS
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
VECLIB_MAXIMUM_THREADS
```

At lines 54–70, `_require_bootstrap_environment()` compares every member of
the twelve-control `CAPTURE_ENVIRONMENT` with the inherited value and raises
on either absence or difference. The call occurs at executable startup before
the NumPy import at line 72. The worker contains no `os.environ.update` and
does not assign any of the seven thread controls. Its later environment
assignment is confined to the verified pinned data root and cannot normalize a
thread request.

The focused test at lines 198–230:

- derives its mutation set from the exact seven-member
  `THREAD_ENVIRONMENT`;
- launches a fresh executable process for every member;
- requires a nonzero exit and empty stdout;
- requires the changed control name in stderr;
- checks the pre-import source order;
- rejects any observed `payne_zero_atmosphere` import.

Source order makes the failure pre-NumPy as well as pre-Payne-Zero. The direct
environment test separately rejects wholly missing controls, every changed
non-thread capture control, and a nonempty cache. The repaired worker never
substitutes the expected values for caller values.

**Conclusion:** the exact historical P0-1 failure is repaired.

### R3. P0-2 — Direct staged files are gated, but dependency discovery is incomplete: **FAIL**

#### What is repaired correctly

For the two declared staged files,
`verify_staged_converter_dependencies()` now performs the required gate:

1. the expected hash must agree with the frozen pinned Python manifest;
2. both staged and pinned paths must be regular nonsymlink files;
3. the pinned hash must equal the expected hash;
4. the staged hash must equal the expected hash;
5. the staged and pinned bytes must be equal.

This runs in the full pre-import identity pass at line 549 and again at line
615 immediately before the converter subprocess call at line 653. Thus neither
declared file can drift between the initial authority pass and child startup.

The focused test mutates each declared dependency independently with an
AST-equivalent comment-only byte change and asserts that `subprocess.run` was
not called. It also proves that a byte-identical symlink is rejected before
child startup. The accepted identities enter `identity__names` and
`identity__sha256_or_commit`, and therefore the recomputed full-capture schema
and fingerprint.

Those repairs close the literal missing gates identified for
`line_catalog.py` and `population_layout.py`.

#### Remaining imported-source authority gap

The worker and test equate “converter dependency” with explicit
`ImportFrom` targets. That is not the complete Python import dependency set.

The accepted converter executes:

```python
from payne_zero_atmosphere.line_catalog import SelectedLineCatalog
from payne_zero_atmosphere.population_layout import atomic_population_slot_start
```

Before either submodule can load, Python imports and executes the parent
package:

```text
src/payne_zero_atmosphere/__init__.py
```

An independent import-origin probe returned:

```text
/Users/ysting/stellar-spectroscopy-from-scratch-gpu/src/payne_zero_atmosphere/__init__.py
/Users/ysting/stellar-spectroscopy-from-scratch-gpu/src/payne_zero_atmosphere/line_catalog.py
```

The executed staged initializer is not a harmless path abstraction: arbitrary
Python placed there would run before either named submodule. Its current
SHA-256 is
`db7e0860918d9532e0728fa4b498343fd7cb728b2edb29a64ead3784c5882999`.
It intentionally differs from the pinned package initializer, whose SHA-256 is
`dbb7734cab4f3e98b9b88d1f1b5ec27afd02fd7fd003ba7931b43d3750049d61`.

`STAGED_CONVERTER_DEPENDENCIES` contains only the two submodules. The discovery
test at lines 159–170 extracts only explicit `ImportFrom.module` values and
therefore cannot discover the implicitly executed parent initializer.
Consequently:

- no regular-file or nonsymlink gate covers the staged initializer;
- no accepted local hash covers it;
- no mutation test proves it fails before child startup;
- no pre-import identity row records it;
- the full-capture fingerprint binds the two declared submodules but not all
  staged Python executed by the converter child.

An output-preserving comment mutation of this file would already escape the
claimed complete identity, while an executable mutation could change child
behavior before the direct imports. Exact converted output checks reduce the
chance of an unnoticed numerical change but do not replace source authority:
the historical finding explicitly requires imported-source identity to fail
before child import.

Required repair:

1. include every executed staged package initializer in dependency discovery;
2. give the staged initializer an explicit accepted local identity and
   nonsymlink/regular-file gate, or restructure the child import so it is
   provably not executed;
3. bind that identity into the same pre-import and full-capture evidence;
4. add an AST-equivalent byte-mutation test and symlink test that both fail
   before `subprocess.run`;
5. derive the expected closure from explicit imports **plus their package
   initializers**, rather than only the two AST module strings;
6. recapture twice after the worker/test/evidence identities change.

**Conclusion:** dependency discovery is not complete, mutations do not fail
closed for every staged source executed by the child, and the full evidence
does not yet bind that complete set. Historical P0-2 remains open.

### R4. Nominal physics and deterministic evidence: **PASS**

The repaired authority code did not change the proposed physical payload.
The focused suite launches two nominal fresh processes with distinct empty
external caches. This re-audit also launched a third fresh process. All
reproduced:

```text
fixture member count          19
ephemeral evidence count      87
loaded pinned Python sources  35
fixture schema digest
  f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698
fixture payload fingerprint
  f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663
full capture schema digest
  a96bb585009378fab26c47c97ff2f6476b21ad697c2b2e1f3b3c4a10eb8b045e
full capture fingerprint
  cc69414e43cce0217b5937bead7cb5cd8ea56d54f94da43faf54859fbbf43db5
line-output SHA-256
  43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58
```

The third capture reproduced every frozen nineteen-member shape, dtype, and
hash; all six full/projected hashes; bitwise full/projection line equality;
placeholder fixture and line equality; the eleven-file dynamic read set; and
the two declared staged-submodule hashes. The fixture payload fingerprint is
unchanged from the historical scientific pass. The full fingerprint changed
as expected because it now binds the repaired worker and the two direct staged
identities.

This numerical success does not waive the missing initializer identity.

### R5. Serializer and publication boundary: **PASS**

The worker still:

- accepts no path or publication CLI option;
- returns detached in-memory arrays and evidence;
- prints only one JSON summary;
- contains no NumPy save, file serialization, `write_bytes`, or `write_text`
  call;
- contains no fixture, golden, or manifest destination;
- reports capture scope incomplete and every publication/golden/manifest-write
  boolean false.

The focused test snapshots canonical `data/` identities before and after both
nominal captures and observes no change. No serializer or publisher exists.

### R6. Focused verification evidence

Executed against the exact repair snapshot:

```text
PYTHONPATH=src:. pytest -q \
  tests/test_chapter06_atmosphere_fixture_worker.py

12 passed in 29.86s
```

```text
ruff check \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py

All checks passed!
```

```text
ruff format --check \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py

2 files already formatted
```

The passing suite is valid evidence for the seven control mutations, the two
declared staged-file mutations, the nineteen-member payload, two nominal
captures, dynamic reads, and no-write behavior. It cannot establish complete
staged dependency closure because its discovery assertion omits the parent
package initializer by construction.

### R7. Repair disposition

**REJECT safe-sequence steps 4–6 at worker SHA-256
`8e566fcbb899ac54b7af265a3f5907c0510808dc3dc17dd2165ec9170f1d8170`.**

P0-1 is closed. P0-2 is partially repaired for the two named submodules but is
not closed for the complete executed staged-source set. Therefore the top
disposition cannot become ACCEPT, and no fixture, oracle, golden, manifest
change, serializer, publisher, or publication is authorized.

## Final closure re-audit

Re-audited: 2026-07-30  
Disposition: **FINAL ACCEPT the in-memory atmosphere fixture worker for
safe-sequence steps 4–6**

This section preserves both historical rejection analyses above. It evaluates
the twice-repaired worker against every prior P0, the complete staged converter
execution closure, the nominal nineteen-array science, and the no-publication
boundary.

No worker, test, source, data, pinned Payne Zero file, paper file, fixture,
oracle, golden, manifest, serializer, publisher, or publication artifact was
edited by this re-audit. Apart from the required top-disposition update, this
section is the only content appended.

### F1. Exact accepted review snapshot

| reviewed object | lines before this append | SHA-256 |
| --- | ---: | --- |
| `scripts/chapter06_atmosphere_fixture_worker.py` | 1634 | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| `tests/test_chapter06_atmosphere_fixture_worker.py` | 520 | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| `design/chapter06_atmosphere_fixture_worker_candidate.md` | 349 | `da53e4846ee91f814c437994fff604ae91ed94a4da4db7856f8f47bb61cf72dc` |
| this independent audit before the final append | 714 | `00025449d4d39358c99a4ac7e3da76fdcdbcaecfb1f208ebfaa2f507fbcb3fca` |
| atmosphere fixture/oracle plan | 1175 | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| exact Chapter 6 source contract | 1371 | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |
| independently accepted converter | 619 | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| accepted converter tests | 585 | `254d796b7ab761ca806c372d0bcdd935067ff1a89b2acfebcfa3007fe3f549dc` |
| converter independent acceptance | 300 | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |

The pinned commit remains:

```text
9c44001feae40b85146630499e6f8a5fed42e5af
```

All fixed worker constants agree with the independently accepted converter,
converter tests, converter audit, plan, contract, source, data, and commit
identities.

### F2. P0-1 — inherited process controls: **CLOSED**

The worker no longer normalizes any caller-supplied thread control. At
executable entry, `_require_bootstrap_environment()` compares all twelve
capture controls before the NumPy import. It rejects absence or disagreement;
there is no `os.environ.update` and no assignment to any of the seven thread
controls.

An independent fresh-process loop changed each inherited control:

```text
MKL_DYNAMIC
MKL_NUM_THREADS
NUMBA_NUM_THREADS
NUMEXPR_NUM_THREADS
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
VECLIB_MAXIMUM_THREADS
```

All seven processes:

- exited with return code `1`;
- wrote no stdout;
- named the changed control in stderr;
- showed no `payne_zero_atmosphere` import in import-time evidence.

The later environment boundary separately requires every non-thread capture
control and an existing, empty, external, nonsymlink Numba cache. The two
nominal review captures used distinct caller-created cache directories.

The historical overwrite and indistinguishable-fingerprint failure cannot
recur at this snapshot.

### F3. P0-2 — complete staged converter execution closure: **CLOSED**

#### Complete recursively discovered source set

Independent recursive AST discovery began at the accepted converter, followed
every staged `Import` and `ImportFrom`, inserted every implicitly executed
package initializer, and recursively scanned the resulting staged sources.
It produced exactly:

```text
payne_zero_atmosphere/__init__.py
payne_zero_atmosphere/line_catalog.py
payne_zero_atmosphere/population_layout.py
```

This set equals:

1. `STAGED_CONVERTER_DEPENDENCIES`;
2. `PINNED_CONVERTER_DEPENDENCY_HASHES`;
3. the converter child's actual loaded staged-source manifest.

No additional staged package source was statically reachable or dynamically
loaded.

#### Correct local-versus-pinned authority

| executed staged source | staged SHA-256 | pinned SHA-256 | required relation |
| --- | --- | --- | --- |
| `payne_zero_atmosphere/__init__.py` | `db7e0860918d9532e0728fa4b498343fd7cb728b2edb29a64ead3784c5882999` | `dbb7734cab4f3e98b9b88d1f1b5ec27afd02fd7fd003ba7931b43d3750049d61` | separately pinned local initializer; intentional difference |
| `payne_zero_atmosphere/line_catalog.py` | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` | `2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92` | exact byte parity |
| `payne_zero_atmosphere/population_layout.py` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` | exact byte parity |

The repaired worker does not make the false claim that the one-line
progressive initializer equals the full pinned package initializer. Instead,
it independently hash-gates both initializer identities. The two functional
submodules must additionally remain byte-identical to their pinned
authorities.

For all three sources, both staged and pinned leaves must be regular
nonsymlink files with their exact accepted hashes. The complete gate runs
during `verify_preimport_identities()` and again immediately before converter
child startup. Therefore source drift fails before converter import or
conversion.

#### Runtime loaded-source equality

The converter child reports every loaded module whose name begins with
`payne_zero_atmosphere`, requires every reported path to remain under the
staged source root, and returns each file's actual SHA-256. The parent requires
exact dictionary equality with the three-entry staged manifest above.

The independently observed child manifest was:

```text
payne_zero_atmosphere/__init__.py
  db7e0860918d9532e0728fa4b498343fd7cb728b2edb29a64ead3784c5882999
payne_zero_atmosphere/line_catalog.py
  2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92
payne_zero_atmosphere/population_layout.py
  36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0
```

The exact runtime manifest, seven converted fields, four-word payload, and
non-authoritative observed-row declaration are all required before the
fixture worker accepts the converter result.

#### Adversarial mutation closure

An independent output-preserving comment mutation was applied separately to
each of the three staged sources. Each mutation retained an identical AST but
changed the file bytes. All three were rejected by the exact staged hash gate,
and mocked `subprocess.run` was never called.

Each source was also replaced separately with a byte-identical symlink. All
three symlinks were rejected before child startup.

The focused suite covers the same six mutations and derives the expected
source closure recursively rather than hard-coding only the converter's two
explicit imports.

The complete historical P0-2 gap is closed.

### F4. Full-evidence identity and fingerprint binding: **PASS**

The capture's `identity__names` and `identity__sha256_or_commit` arrays contain
both the staged and pinned identity for all three converter dependencies. The
separate child-loaded path/hash arrays contain the exact runtime three-source
manifest. These arrays enter the full combined mapping.

Independent reconstruction from the detached fixture and evidence mappings
returned:

```text
full capture schema digest, stored and recomputed
  cdf470038e67301b4c19b0691e672cd97df3233a3decf1b88e32ce3ac0dc1371

full capture fingerprint, stored and recomputed
  a25875097c7084ffe2577de65c2913d8775f29613823d9e6e0ab0d9db4644654
```

The full fingerprint also binds:

- the worker, plan, contract, converter, tests, audit, commit, all 35 pinned
  loaded Python sources, and all 11 dynamic data identities;
- the exact inherited process controls and external-cache policy;
- the structured source members and configuration;
- every full/projected/placeholder array and line slab;
- the dynamic read set;
- all lifecycle and no-write declarations.

The physical payload remains separately identified so source/process
authority changes cannot be mistaken for scientific changes.

### F5. Nominal nineteen-array science: **PASS**

Two fresh executable processes with distinct empty external caches, plus one
independent in-process capture, reproduced:

| capture property | exact accepted value |
| --- | --- |
| fixture members | `19` |
| evidence members | `89` |
| loaded pinned Python sources | `35` |
| fixture schema digest | `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698` |
| physical payload fingerprint | `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663` |
| full capture schema digest | `cdf470038e67301b4c19b0691e672cd97df3233a3decf1b88e32ce3ac0dc1371` |
| full capture fingerprint | `a25875097c7084ffe2577de65c2913d8775f29613823d9e6e0ab0d9db4644654` |
| selected-line output SHA-256 | `43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58` |

Every one of the nineteen member shapes, dtypes, and C-byte hashes reproduced
the table in Section 4.4. No raw source field, full packed-state array,
validator-only placeholder, or line output entered the fixture mapping.

The exact full-array identities remained:

```text
ion-stage populations
  53ff8546517e744862abf16c490f1a3b4bcd761d681f5fde0cfb8ad62e0812b6
population/density/fractional-width support
  706c8db98522bacb0c428279a29b270ec53772174207cceadd27e701947030c0
fractional Doppler widths
  23c57f545bdfd74af3dd62ef57b8f347570953ea50c4e19289a9aa9715beb868
```

The exact projected reconstruction identities remained:

```text
ion-stage populations
  30fa30338c773905b57023bdd6b102c205569207346227128f4e1ce585500166
population/density/fractional-width support
  60db3e4a9c56e347e764b46a7d34e3c33bc1e2256dd44041d0621c1953a63524
fractional Doppler widths
  d93c56e13576a3a6dce3071026ca9479cd6a94de1518d68514ed1866c2656606
```

Both selected-line calls return count one and bitwise-identical
`(80,30000) float32` line slabs. The perturbed validator-only route reproduces
all nineteen fixture members and the line slab bitwise.

The exact eleven-file dynamic data read set is unchanged, fully prehashed, and
contains neither `continuum_level_tables.npz` nor
`observed_atomic_lines.npy`. The latter remains only
`non_authoritative_packing_corroboration`.

### F6. Determinism and authority boundary: **PASS**

The two distinct-cache fresh-process summaries agreed on:

- all four capture identities;
- all nineteen member schemas and hashes;
- all full and projected array identities;
- the line-output identity;
- complete staged/pinned/child converter source manifests;
- full/projected and placeholder parity;
- the 11-file dynamic read set and 35-source pinned manifest;
- all no-write and no-publication declarations.

The worker still:

- accepts no CLI argument or output path;
- returns only detached in-memory fixture arrays and ephemeral evidence;
- prints only one JSON summary;
- contains no fixture serializer, NumPy save call, file writer, manifest
  writer, golden reader/writer, publisher, overwrite, repair, or alternate
  destination;
- leaves all canonical `data/` bytes unchanged;
- reports capture scope incomplete and every fixture/golden/manifest-write
  boolean false.

This acceptance is scientific-worker acceptance only. It does not create or
authorize a fixture artifact.

### F7. Verification executed

Focused worker suite:

```text
PYTHONPATH=src:. python -m pytest -q \
  tests/test_chapter06_atmosphere_fixture_worker.py

12 passed in 33.50s
```

Accepted converter plus fixture-worker suites:

```text
PYTHONPATH=src:. python -m pytest -q \
  tests/test_chapter06_atmosphere_line_converter.py \
  tests/test_chapter06_atmosphere_fixture_worker.py

27 passed in 39.95s
```

Repository-wide suite:

```text
PYTHONPATH=src:. python -m pytest -q

396 passed, 1 skipped in 90.74s
```

Targeted static checks:

```text
ruff check \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py
All checks passed!

ruff format --check \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py
2 files already formatted

python -m py_compile \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py
pass
```

### F8. Final acceptance boundary

**FINAL ACCEPT the repaired Chapter 6 atmosphere fixture worker at SHA-256
`21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531`
with focused-test SHA-256
`611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42`.**

This acceptance freezes the in-memory safe-sequence steps 4–6 boundary:

- exact staged converter execution closure and source authority;
- exact structured fixed-electron-density route;
- the nineteen-member fixture science;
- full-versus-projected and placeholder nonownership proofs;
- dynamic source-read closure;
- the four capture identities and no-write lifecycle.

It does **not** authorize:

1. fixture serialization or deterministic candidate bytes;
2. a fixture publisher or canonical fixture file;
3. a manifest entry or manifest mutation;
4. atmosphere oracle or seam-suite acceptance;
5. a sparse golden or golden publisher;
6. detached authorization or any publication.

Those remain separate independently reviewed gates.
