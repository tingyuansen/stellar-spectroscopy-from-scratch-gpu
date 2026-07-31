# Chapter 5 oracle publication contract

Status: **publisher implementation contract; publication remains blocked**  
Prospective publisher: `scripts/build_chapter05_payne_zero_goldens.py`  
Scientific worker: `scripts/chapter05_oracle_worker.py`  
Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

## 1. Purpose and hard boundary

Chapter 5 needs two different comparison products:

1. a small reader golden containing only interpretable state, threshold,
   component, edge-interpolation, and final-parity slices;
2. a full integration artifact containing the four direct atmosphere products
   and the exhaustive boundary, ownership, trace, and counterfactual evidence.

These products must not be collapsed into one large notebook dependency. The
reader must never load the integration artifact. The integration tests load
both artifacts because the integration artifact references, rather than
duplicates, arrays owned by the reader golden.

The 1,161-key worker result is an **ephemeral raw capture**. It is generated
twice, validated twice, and never published verbatim. The publisher makes a
lossless semantic projection from that capture into the two reviewed
artifacts. This distinction permits full scientific coverage without storing
the same 30,000-point grid many times.

This contract does not authorize publication. Publication remains blocked
until all of the following exist and pass independent review:

- a detached Chapter 5 oracle acceptance record;
- the publisher implementation;
- publisher structural and scientific tests;
- a second independent publisher review;
- exact final artifact schema digests and byte hashes;
- a separate detached publication-acceptance JSON record binding the reviewed
  publisher, candidates, and prepublication manifest.

The scientific worker remains an in-memory observer. It must not gain a
golden path, an output option, or knowledge of the final archive layouts.

## 2. Accepted raw-capture identity

The independent v2 oracle re-audit accepted the following tuple after two
independent fresh-cache runs. A future publisher must freeze every value as a
literal constant and reject any mismatch before assembly.

| identity | accepted value |
| --- | --- |
| worker SHA-256 | `429252d5fefd2b911ce4321578820aa67b505d5fe37b174cb647d4b6177d7389` |
| capture-contract SHA-256 | `4198f76419f102efbb3468b5f2ed7ddca7ff6af776ffc566cfa4058b6164fdaf` |
| capture schema version | `2` |
| capture key count | `1161` |
| capture schema digest | `652c110dc79a6f6dfca6893bee35416289675b4920a5d0dcfe6b2cb262dacf3d` |
| fixture archive SHA-256 | `ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae` |
| fixture payload digest | `4bcf0bbd8d61e58334c4c7ef6caaaf9ca47e6fb4536ad0098d5a541d540ec048` |
| physical-payload fingerprint | `d223351fa2c51dc24a1b01896da9ab9a82fc475f4082c47fde34734d8dc03343` |
| full-capture fingerprint | `3d2c131711e1c0dc6aa088892193bb24d41a76d005bc20dd1c42d3e84f66e656` |

The accepted exact-source contract currently has SHA-256
`ec1c84519a898a454a408780bd6b36fa723fc7bade83a98e11eede1343bf2956`.
It is a publisher input even though it is not part of the worker's
self-fingerprint. The deterministic NPZ writer currently has SHA-256
`f4886766524d79623e648d28ab9d24215da42f8bf1f859f69381c546f9c96e49`.

The detached acceptance record should be created at
`design/chapter05_oracle_worker_acceptance.md`. It must state the two accepted
fresh-cache summaries, the exact commands and environment, the populated-cache
failure, the symlink-cache failure, and the independent audit disposition.
Its SHA-256 is intentionally not invented here. A publisher with a missing,
empty, placeholder, or unreviewed acceptance-record hash must fail.

The future publisher also pins the final SHA-256 of this contract. Editing
this contract therefore invalidates the publisher until that input hash is
reviewed and refreshed.

## 3. Final artifact names and roles

The final directory is:

`data/golden/payne_zero/chapter05`

It contains exactly two files:

| file | archive kind | role |
| --- | --- | --- |
| `chapter05_continuum_reader_cpu_float64.npz` | `continuum_reader` | comparison-only reader golden |
| `chapter05_continuum_integration_cpu_float64.npz` | `continuum_integration` | exhaustive CPU-float64 integration evidence |

No raw capture, cache, temporary archive, log, report, duplicate grid, or
third "constants" archive may remain in this directory.

Both archives are uncompressed deterministic NPZ files written by
`scripts/deterministic_npz.py`. Members are lexically ordered, use the frozen
ZIP timestamp and permissions, and forbid object dtype.

## 4. Common archive metadata

The following scalar or short-vector metadata is deliberately duplicated in
both artifacts. This is identity duplication, not scientific-array
duplication:

- `meta__archive_kind`;
- `meta__archive_schema_version`;
- `meta__payne_zero_commit`;
- `meta__worker_sha256`;
- `meta__capture_contract_sha256`;
- `meta__exact_source_contract_sha256`;
- `meta__publisher_contract_sha256`;
- `meta__publisher_sha256`;
- `meta__deterministic_npz_sha256`;
- `meta__oracle_acceptance_sha256`;
- `meta__fixture_sha256`;
- `meta__fixture_payload_digest`;
- `meta__raw_capture_schema_version`;
- `meta__raw_capture_key_count`;
- `meta__raw_capture_schema_digest`;
- `meta__accepted_physical_payload_fingerprint`;
- `meta__accepted_full_capture_fingerprint`;
- `meta__loaded_pinned_python_source_count`;
- `meta__loaded_pinned_python_manifest_digest`;
- `meta__cpu_only = True`;
- `meta__work_dtype = "float64"`;
- `meta__regime_names`, in the accepted order
  `hot_dwarf, solar_dwarf, low_gravity_giant, cool_molecule_rich`;
- the exact one-thread process-control names and values.

The integration artifact additionally stores:

- `meta__reader_archive_sha256`;
- `meta__reader_archive_schema_digest`;
- `meta__logical_raw_capture_coverage_complete = True`.

The reader archive must not store an integration-archive hash. This one-way
reference prevents a hash cycle: the reader is assembled first, and the
integration artifact binds the exact reader bytes.

Both archives retain the exact 52-entry loaded-source name/hash manifest and
the five active upstream static-table name/hash pairs as compact string
vectors. Those vectors are allowed common metadata because either archive
must be rejectable without trusting a neighboring file.

No archive stores an absolute cache path, a temporary path, a wall-clock time,
or a hostname. The canonical source root may be recorded as the declared
read-only root, but it is not treated as a portable scientific input.

## 5. Artifact A: compact reader golden

### 5.1 Scientific purpose

The reader golden answers only questions that appear in the teaching flow:

- which state view a term consumed;
- what happens on each side of a physical threshold;
- how named absorption and scattering terms add;
- how the line-selection threshold differs from raw opacity;
- how the synthesis edge triplet and parabolic basis reconstruct the product;
- whether the four declared lanes reproduce the accepted outputs.

It does not contain any `(6,30000)` product, any 30,000-point sampling grid,
or the exhaustive counterfactual and trace record.

### 5.2 Common axes stored once

The reader archive owns one copy of each compact axis:

- regime name and zero-based regime index, length 4;
- depth index, length 6;
- the nine diagnostic threshold-family names;
- the 27 nextafter threshold frequencies, family indices, and sides;
- the 344 atmosphere reference wavelengths and packed wavelength indices;
- the synthesis requested wavelength vector;
- used edge indices, selected sample indices/frequencies/sides;
- the complete 1,020-entry packaged edge-sample frequency vector;
- signed edge frequencies, edge wavelengths, midpoints, and interval-width
  factors;
- left, center, and right parabolic bases and their sum;
- the twelve-wavelength supported extension axis.

All edge geometry is required to be bit-identical across the four raw regime
captures before it is coalesced. The reader stores it once. Per-regime copies
are forbidden.

### 5.3 Interpretable state slice

The fixture remains the input-state authority. The reader golden stores only
the following exact `build_pops` outputs, stacked in regime order, because
they make population ownership visible:

- `temperature`;
- `mass_density`;
- `electron_density`;
- `hydrogen_partition_normalized_ion_stage_populations`;
- the derived `hydrogen_ionized_population` fallback;
- `helium_neutral_partition_normalized_population`;
- `helium_singly_ionized_partition_normalized_population`;
- `hot_metal_populations`;
- `charge_square_population_sum`.

These values come from each regime's
`synthesis__extension__input__pops__...` capture. The archive labels them
`reader_state__...`, preserving exact units and shapes. It also stores the
ordered 25-field `build_pops` inventory, but the remaining full state arrays
belong to the integration artifact. The slice is comparison output and must
never replace the fixture as an input to a local calculation.

### 5.4 Atmosphere component bank

Instead of publishing dozens of unrelated member names, the publisher stacks
the compact `(6,27)` arrays in frozen source order:

- 14 named absorption components;
- the corresponding 14 component source functions;
- electron, H I Rayleigh, He I Rayleigh, and H2 Rayleigh scattering;
- CH, OH, and H2-CIA molecular subcomponents;
- ordered absorption, scattering, and source-numerator sums;
- final compact absorption, scattering, and source;
- elementwise absorption, scattering, and source residuals.

The archive stores a component-name vector adjacent to every component bank.
The publisher validates each bank against the unstacked raw members before
discarding those raw copies.

### 5.5 Line-reference threshold

The reader archive owns:

- the four `(6,344)` float32 threshold tables;
- active counts in regime order;
- a `(4,338)` padded active-index and active-frequency bank;
- `(4,6,338)` active absorption, scattering, and source banks;
- a boolean validity mask for the padded columns.

Padding is zero, never NaN. The validity mask is authoritative. Unpadding
must reproduce every variable-length raw member bit for bit. The final
threshold column must duplicate column 342, and the final packed index must be
the `2**30` sentinel.

### 5.6 Synthesis edge and component bank

The reader archive owns, stacked in regime order:

- standard-route absorption and scattering `(4,6,36)`;
- selected pre-interpolation absorption and scattering;
- independent reconstructed absorption and scattering;
- interpolation residuals;
- the named standard-route component banks;
- the seven isolated `_minor_terms` case banks and their residuals;
- sampled-diagnostic absorption, scattering, source, components, and
  residuals `(4,6,27)`;
- finite sampled-extension absorption, scattering, and source `(4,6,12)`;
- the standard, diagnostic, and extension layout/invariants flags.

The reader does not publish the full extension `pops` and
`FrequencyInvariants` payload, unused-edge trace arrays, or population
counterfactuals. Those belong to the integration artifact.

### 5.7 Small pedagogical seam slices

Only the slices used directly in the chapter are reader-owned:

- the H-minus nextafter threshold triplet;
- the CH and OH lower/upper nextafter triplets at 8999 and 9000 K;
- the H2-CIA nextafter triplet around 20,000 cm\(^{-1}\);
- the three IFOP 4/13 scattering columns on the compact diagnostic axis;
- the reversed H2-CIA lower/upper interpolation weights;
- the standard-route H-II fallback and stored-H2 non-use result flags.

The integration artifact points to these members through string aliases. It
must not contain second physical copies.

### 5.8 Reader size budget

The expected uncompressed NPZ size is approximately 0.5–2.5 MiB. A hard
4 MiB ceiling is a structural test. Crossing it indicates that a full grid,
full product, exhaustive trace, or duplicated component bank leaked into the
reader artifact.

## 6. Artifact B: full integration evidence

### 6.1 Four direct atmosphere products

The integration artifact owns exactly three product banks:

- `atmosphere_product__absorption`, shape `(4,6,30000)`, float64;
- `atmosphere_product__scattering`, shape `(4,6,30000)`, float64;
- `atmosphere_product__source`, shape `(4,6,30000)`, float64.

The leading axis uses the accepted regime order. Separate per-regime product
members are forbidden after stacking.

### 6.2 One five-class grid bank

The eight boundary temperatures exercise five distinct atmosphere sampling
policies. The four physical regimes reuse members of those policies. The
integration artifact therefore owns:

- `grid_bank__wavelength_nm`, shape `(5,30000)`, float64;
- `grid_bank__frequency_weight_hz`, shape `(5,30000)`, float64;
- the five policy labels;
- `sampling_boundary__grid_bank_index`, shape `(8,)`;
- `atmosphere_product__grid_bank_index`, shape `(4,)`;
- the eight boundary temperatures and active line-reference counts.

The publisher derives frequency as the pinned exact light speed in nm s\(^{-1}\)
divided by wavelength and requires bit equality with every raw
`product_frequency_hz` member. It stores the exact light-speed scalar and a
digest of each derived frequency row; it does not store a second frequency
grid.

Before coalescing, every raw boundary wavelength/weight row and every
per-regime sampling wavelength/weight row must match its selected bank row
bit for bit. The following are forbidden:

- the raw `(8,30000)` boundary wavelength matrix;
- the raw `(8,30000)` boundary weight matrix;
- four per-regime wavelength grids;
- four per-regime weight grids;
- four per-regime frequency grids.

This representation retains every exact grid value while removing redundant
copies.

### 6.3 Exhaustive seam and ownership evidence

The integration artifact owns every scientific raw member not assigned to the
reader archive, including:

- all H2 population-policy boundary inputs, bit patterns, clamped
  temperatures, equilibrium constants, and populations;
- complete CH/OH boundary cross-section grids;
- complete H2-CIA inputs, table indices, fractions, weights, coefficients,
  masks, and outputs;
- all-warm versus mixed-column molecular-entry evidence;
- molecule-disabled CH/OH and continuum-local H2 ownership evidence;
- owner-level H2 CIA and Rayleigh behavior at 19999, 20000, and
  `nextafter(20000,+inf)` K;
- complete IFOP 19 surrogate inputs, flags, table state, opacity, scattering,
  and bolometric-like source;
- all component activation names, masks, nonzero counts, and maxima;
- complete standard-route call traces, unused-edge masks and indices;
- rich versus trimmed H-II, 1.5-times H-II perturbation, stored schema-H2
  perturbation, and signed-edge counterfactuals;
- every non-reader `build_pops` output;
- every numeric `FrequencyInvariants` field in the finite supported extension;
- exact ordered field-name inventories and route flags;
- post-lane loaded-source verification and all non-reader reproducibility
  metadata.

Reader-owned IFOP 4/13 compact arrays, component banks, threshold arrays, edge
geometry, and selected state arrays are represented only by
`alias__...__reader_member` strings plus `meta__reader_archive_sha256`.

### 6.4 Complete logical coverage of the raw capture

The integration artifact stores three parallel length-1161 string vectors:

- `inventory__raw_member_name`;
- `inventory__disposition`;
- `inventory__published_member`.

Each sorted raw member has exactly one disposition:

- `reader`;
- `integration`;
- `coalesced_grid_bank`;
- `reader_alias`;
- `common_identity_metadata`;
- `publisher_control_metadata`.

No scientific raw member may be dropped. For stacked or banked arrays,
`inventory__published_member` names the bank and row/slice convention. A
validator reconstructs the logical raw mapping from the two artifacts and
requires the accepted key set, dtypes, shapes, schema digest, physical
fingerprint, and full fingerprint.

Thus "full integration" means exhaustive logical coverage, not a literal
copy of the redundant 1,161-key temporary NPZ.

### 6.5 Integration size budget

The three product banks contribute about 16.5 MiB. The five-class wavelength
and weight bank contributes about 2.3 MiB. Exhaustive compact seam evidence,
inventories, and full extension state should keep the deterministic
uncompressed artifact near 20–26 MiB. A 32 MiB hard ceiling is a structural
test. Exceeding it normally means raw grids or regime-invariant edge geometry
were duplicated.

## 7. Exact ownership and deduplication rules

Scientific ownership is one-way:

| information | sole owner |
| --- | --- |
| compact diagnostic axes and components | reader |
| selected interpretable `build_pops` state | reader |
| line-reference threshold and active values | reader |
| synthesis edge geometry and interpolation proof | reader |
| standard, diagnostic, and extension compact parity outputs | reader |
| four direct `(6,30000)` atmosphere products | integration |
| five-class 30,000-point grid bank | integration |
| exhaustive boundary/counterfactual/trace evidence | integration |
| full non-reader extension state and invariants | integration |
| fixture inputs | input fixture, never either golden |
| identity metadata | intentionally duplicated in both |

The publisher must maintain an explicit source-member-to-owner table. Prefix
matching alone is insufficient where one prefix contains both reader and
integration fields.

Allowed duplication is limited to short identity metadata, regime names, and
the reader-member alias strings. Any numeric scientific array copied into
both archives is a test failure.

Regime-invariant raw arrays are required to compare bitwise before
coalescing. Near equality is not enough. If a future source makes one of
those arrays regime-dependent, publication fails and this design must be
reviewed rather than silently storing extra copies.

## 8. Fail-closed publisher inputs

Before spawning a worker, the publisher verifies:

1. the exact pinned Payne Zero root and commit;
2. the accepted worker, capture contract, exact-source contract, publisher
   contract, deterministic writer, and oracle-worker acceptance-record hashes;
3. the exact 52-file upstream Python manifest;
4. the two staged continuum source hashes:
   - atmosphere `1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81`;
   - synthesis `ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b`;
5. the canonical fixture path, archive hash, complete 217-field schema, and
   payload digest;
6. byte equality of each staged static table with its pinned upstream file:
   - atmosphere continuum tables
     `6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3`;
   - atmosphere Karzas-Latter tables
     `23805dc17c47af45b8ae63b2e278e1fb6c584a01c87d1eb3c31306e4555e6d15`;
   - atmosphere molecular-equilibrium tables
     `1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe`;
   - synthesis continuum tables
     `406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d`;
   - synthesis edge grid
     `11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e`.

The worker repeats the upstream source, loaded-module, table, fixture, and
environment checks inside each child. Publisher checks do not replace worker
checks.

A Git HEAD match is never sufficient source identity. Edited or redirected
files fail through byte hashes and exact loaded-module-set verification.

## 9. Deterministic two-capture protocol

### 9.1 Child isolation

The publisher creates two unrelated temporary roots, `capture-a` and
`capture-b`. Each root receives:

- a unique Numba cache path that is absent or an empty nonsymlink at child
  start;
- its own temporary raw archive;
- its own assembled reader and integration artifacts.

The publisher child establishes, before NumPy, Numba, Torch, or Payne Zero is
imported:

- `LC_ALL=C`;
- `TZ=UTC`;
- `PYTHONHASHSEED=0`;
- `PYTHONNOUSERSITE=1`;
- `PYTHONDONTWRITEBYTECODE=1`;
- `MKL_DYNAMIC=FALSE`;
- all Numba, Torch, BLAS, OpenMP, NumExpr, OpenBLAS, and Accelerate thread
  counts equal to one;
- canonical `PAYNE_ZERO_DATA_ROOT`;
- CPU Torch float64.

The child imports the scientific worker, calls `build_oracle_results`, checks
the accepted tuple, and writes a temporary deterministic raw NPZ. This hidden
publisher-internal mode is the only serializer for worker results. It is not
a worker CLI option and cannot name a final golden path.

### 9.2 Raw agreement

For each raw capture, the publisher requires:

- `meta__capture_scope_complete=True`;
- `meta__golden_publication_performed=False`;
- exactly 1,161 object-free members;
- the accepted schema version, key count, and schema digest;
- the accepted fixture and payload identities;
- the accepted physical and full fingerprints;
- exact worker and capture-contract hashes;
- exact post-lane source-manifest verification.

The two deterministic raw NPZ files must then be byte-identical. Fingerprint
agreement without byte agreement is insufficient.

### 9.3 Independent assembly agreement

Each raw capture is independently projected into a reader/integration pair.
Both pairs pass complete final validators. The two reader files must be
byte-identical, and the two integration files must be byte-identical.

Only after raw and final byte agreement may the first pair become a
publication candidate. Any failure destroys temporary products and leaves the
destination untouched.

### 9.4 Negative cache evidence

Publisher tests repeat the worker's negative probes:

- a populated cache directory fails before Payne Zero import;
- a symlink cache path fails before Payne Zero import.

No retry may silently replace either path with a fresh cache.

## 10. Detached authorization and atomic first publication

### 10.1 Cycle-free publication-acceptance record

Publication authorization is a strict JSON record at the stable canonical
path:

`design/chapter05_publication_acceptance.json`

This record is distinct from
`design/chapter05_oracle_worker_acceptance.md`. The oracle record accepts the
scientific capture route and is pinned inside the publisher. The publication
record accepts one later publisher implementation and one exact candidate
pair. It is deliberately **not** hashed by the publisher and its identity is
not embedded in either archive. This breaks the otherwise unavoidable cycle
in which:

1. the record would pin the publisher hash;
2. the publisher hash would change when its record hash changed;
3. the artifact hashes would change because they embed the publisher hash;
4. the record would then need new artifact and publisher hashes.

The publication record itself must be an absolute-canonical regular file when
opened through the repository path. The publisher rejects a missing path,
relative or noncanonical path, symlink at the final component or in a parent,
directory, device, socket, duplicate JSON key, non-UTF-8 input, non-finite JSON
constant, unknown field, missing field, or wrong field type. Opening uses the
platform no-follow flag when available and verifies file identity before and
after reading.

The top-level schema version is 1 and has exactly these fields:

- `schema_version`;
- `record_kind`, exactly `chapter05_publication_acceptance`;
- `publisher`;
- `publisher_contract`;
- `manifest`;
- `artifacts`.

`publisher` pins the exact repository-relative builder path and the reviewed
SHA-256 of the stable publisher. `publisher_contract` similarly pins this
contract's path and SHA-256. Both digests are exactly 64 lowercase hexadecimal
characters and must match the canonical current files. The contract digest
must also match the publisher's already frozen contract input.

`manifest` has exactly:

- canonical repository-relative path `data/MANIFEST.json`;
- `prepublication_sha256`;
- manifest schema version and pinned Payne Zero commit;
- the ordered two final golden paths;
- `chapter05_entries_present=false`.

The publisher reads `data/MANIFEST.json` with the same canonical
regular-file/no-symlink rules, checks its exact prepublication byte hash,
schema version, and commit, and independently proves that it contains no path
under `data/golden/payne_zero/chapter05/`. Thus the record authorizes a first
artifact publication from one reviewed manifest state; it does not pretend
that the later manifest edit has already happened.

`artifacts` contains exactly `reader` and `integration`. Each object pins:

- exact artifact filename and final repository-relative path;
- exact SHA-256 and byte size;
- archive kind and archive schema version;
- accepted key count and schema digest.

The record is valid only when those schema identities equal the constants
accepted by the publisher and each byte size is positive and below its
artifact limit. A syntactically valid record with the wrong publisher,
contract, manifest, schema, filename, path, hash, or size leaves publication
disabled.

The record does not exist during publisher development or verify-only review.
Its absence is the normal fail-closed state and keeps CLI `--publish`
unavailable.

### 10.2 Candidate binding and direct-API reauthorization

After two raw captures and two final assemblies compare byte for byte, the
publisher hashes and sizes the first candidate pair and compares both exact
identities with the detached record before calling the publication function.
Schema agreement alone is insufficient.

`publish_verified_directory` is a second independent authorization boundary.
It reparses the detached record, rechecks the current publisher and all static
inputs, runs the non-injectable complete final-directory validator, and
recomputes both candidate hashes and sizes. It repeats record, publisher,
manifest, and candidate validation on the staged copy immediately before the
atomic rename. The existing-destination and race paths revalidate the
destination against the same record before accepting an identical no-op.
Calling the function directly therefore cannot bypass the checks performed by
the double-capture driver. Publication mode also rejects an injected
replacement publication callable; dependency injection remains available only
to verify-only tests that cannot write a golden.

No destination parent or staging directory is created before the initial
record, publisher, manifest, semantic, hash, and size checks have all passed.

### 10.3 Atomic no-overwrite policy

The publisher stages the exact two-file directory under the destination
parent, validates it again, fsyncs both files and the staged directory, and
atomically renames the staged `chapter05` directory into place.

If the destination does not exist, this is the only allowed publication.

If the destination already exists:

- it must be a directory containing exactly the two expected names;
- both files must be byte-identical to the new verified candidate;
- both existing files must independently pass validation;
- the result is `identical-existing`, with no write.

If any existing byte, member, filename, or metadata differs, the publisher
raises and leaves the existing directory untouched. There is no `--force`,
`--replace`, per-file overwrite, repair, merge, or partial-publication path.

`--verify-only` performs both raw captures, both assemblies, and all
comparisons but never calls the publisher.

## 11. Manifest contract and units

`data/MANIFEST.json` receives exactly two `role: "golden"` records, one for
each final file. Each record contains:

- relative path, SHA-256, byte size, builder path, scope, and archive kind;
- all publisher, worker, contract, acceptance, fixture, schema, and
  fingerprint identities;
- reader-archive SHA-256 in the integration record;
- every archive member's exact shape, dtype, axes, unit/convention, and
  ownership note.

The manifest is a reviewed repository edit after successful deterministic
publication; the publisher does not rewrite it opportunistically. The
detached publication record binds the exact **prepublication** manifest bytes
and requires the two Chapter 5 entries to be absent at publication time. A
later reviewed manifest edit adds and binds the records to the actual files;
it is a separate repository change, not authority that the publisher may
manufacture for itself.

The unit vocabulary is fixed:

| quantity family | manifest unit |
| --- | --- |
| wavelength | `nm` |
| frequency and frequency weight | `Hz` |
| wavenumber | `cm^-1` |
| temperature | `K` |
| mass density | `g cm^-3` |
| ordinary number density | `cm^-3` |
| partition-normalized population | `cm^-3 per partition function` |
| microscopic cross section | `cm^2` |
| CH/OH cross section times partition | `cm^2 times partition-function convention` |
| absorption or scattering mass opacity | `cm^2 g^-1` |
| continuum source | `erg s^-1 cm^-2 sr^-1 Hz^-1` |
| line-selection threshold | `cm^2 g^-1 with embedded 1e-3 and stimulated-emission division` |
| edge interval factor | `nm^2` |
| parabolic basis, fractions, weights | `dimensionless` |
| packed wavelength coordinate | `integer logarithmic-grid coordinate` |
| counts, indices, flags, masks | `count`, `zero-based index`, or `boolean` |
| SHA-256 and fingerprints | `SHA-256 hexadecimal identity` |
| schema and archive versions | `schema version` |
| member-name and regime inventories | `ordered identifier` |

Residuals carry the unit of the quantity they subtract. A value must not be
called dimensionless merely because it is close to zero.

## 12. Required publisher tests

### 12.1 Static identity tests

- Pin every accepted identity in Section 2.
- Reject a one-byte mutation of the worker, either contract, deterministic
  writer, fixture, staged source, staged table, acceptance record, or upstream
  loaded source.
- Reject a wrong commit, redirected module, added loaded module, missing
  loaded module, or noncanonical fixture path.
- Reject a placeholder acceptance hash.

### 12.2 Raw-capture tests

- Require the exact 1,161-key schema and accepted schema digest.
- Require both accepted fingerprints and the fixture payload digest.
- Reject member-name, value, shape, dtype, order, or metadata drift.
- Prove the two child caches are distinct and fresh.
- Prove raw byte mismatch prevents assembly and publication.
- Prove second-capture failure prevents publication.
- Exercise populated-cache and symlink-cache failures.

### 12.3 Reader schema and science tests

- Freeze the final reader member set, key count, schema digest, shapes, and
  dtypes after publisher review.
- Forbid object dtype, nonfinite physical values, full 30,000-point arrays,
  and size above 4 MiB.
- Reconstruct every stacked raw component exactly.
- Recheck ordered component sums and elementwise residual gates.
- Recheck the line-reference active counts, zero padding, sentinel, duplicate
  final column, and float32 dtype.
- Recheck basis partition of unity and exact interpolation reconstruction.
- Recheck the H-II fallback, stored-H2 non-use, lane flags, and selected seam
  slices.
- Require reader calculations to finish before this golden is opened.

### 12.4 Integration schema and science tests

- Freeze the final integration member set, key count, schema digest, shapes,
  and dtypes after publisher review.
- Require the three `(4,6,30000)` float64 product banks.
- Require finite nonnegative absorption/scattering and finite source.
- Require all six layers active in every physical-regime product.
- Reconstruct every raw sampling grid from the five-row bank bit for bit.
- Reconstruct product frequencies from the stored wavelength rows and exact
  light speed bit for bit.
- Forbid raw boundary matrices and per-regime grid copies.
- Recheck every molecular, H2-owner, IFOP, activation, trace, H-II, stored-H2,
  signed-edge, minor-term, source, and interpolation seam.
- Require all 25 `build_pops` fields and every numeric
  `FrequencyInvariants` field across the reader/integration pair.
- Reconstruct the logical 1,161-key inventory and accepted schema digest.
- Recompute both accepted raw fingerprints from the reconstructed logical
  capture.
- Require integration size at or below 32 MiB.

### 12.5 Deduplication tests

- Require exact disjoint ownership for every scientific raw member.
- Allow duplicate identity metadata only from the explicit common list.
- Require all regime-invariant edge arrays to have one physical copy.
- Require all 30,000-point grid values to occur only in the five-row bank.
- Require all full products to occur only in the integration artifact.
- Require the integration reader hash and every reader alias to resolve.
- Reject an unowned raw member, an alias cycle, or an alias to a missing
  reader member.

### 12.6 Determinism and publication tests

- Compare raw A/B files byte for byte.
- Compare reader A/B and integration A/B byte for byte.
- Verify `--verify-only` never publishes.
- Keep `--publish` disabled while the canonical detached publication record is
  absent.
- Reject unknown or missing record fields, duplicate keys, malformed SHA-256,
  wrong publisher or contract identity, wrong candidate hash or size, wrong
  accepted schema, and wrong prepublication manifest identity.
- Reject a symlink, noncanonical path, or nonregular file for the publication
  record or manifest.
- Prove the post-double-capture driver compares both exact candidate hashes and
  sizes with the record before calling the publisher.
- Prove the direct publication API independently revalidates the record,
  publisher, manifest, complete candidate semantics, hashes, and sizes.
- Prove every failed authorization or candidate check creates no destination
  parent, stage, or artifact.
- Verify atomic first publication.
- Verify an identical existing directory is a no-op.
- Verify a different existing directory remains byte-for-byte untouched.
- Verify staging validation failure leaves no destination.
- Verify no CLI force/replace/repair option exists.

### 12.7 Manifest tests

- Require exactly two Chapter 5 golden records and exact artifact bytes.
- Require every member's shape, dtype, axes, and unit/convention.
- Require the integration record to bind the reader SHA-256.
- Reject ambiguous `opacity`, `source`, population, threshold, or
  cross-section units.
- Reject a manifest member absent from the archive or an archive member absent
  from the manifest.

## 13. Reader/runtime separation

The notebook computes its candidate arrays from staged source, staged static
data, and the Chapter 4 fixture. Only after those calculations and local
invariants pass may it open
`chapter05_continuum_reader_cpu_float64.npz`.

The notebook never imports a golden, never uses a golden state as input, and
never opens the integration artifact. The integration artifact belongs to
tests and whole-route verification only.

No golden value may choose a plot limit, threshold family, depth, regime,
component order, edge interval, tolerance, or code branch. All such choices
are frozen by source/teaching contracts before comparison.

## 14. Publication gate

The accepted scientific worker is necessary but not sufficient. The first
publication may proceed only when the separate canonical
`design/chapter05_publication_acceptance.json` record, reviewed publisher,
publisher contract, final-schema constants, exact candidate bytes, complete
tests, deterministic double capture, and the reviewed prepublication manifest
all agree.

`publication_gate_ready()` is not a string-length or placeholder check. It
parses and fully validates the strict record and every identity it binds. The
CLI and the direct publication API call the throwing form of the same gate;
the direct API then repeats candidate and manifest validation around staging.
The record's hash is absent from the publisher and artifacts by design.

Until then:

- the publication-acceptance JSON record remains absent;
- no Chapter 5 golden file is created;
- no manifest golden entry is added;
- no reader code assumes an artifact exists;
- no candidate fingerprint is described as a published result.
