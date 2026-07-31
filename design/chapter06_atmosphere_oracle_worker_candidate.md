# Chapter 6 atmosphere comparison-oracle worker candidate

Date: 2026-07-30  
Status: **candidate complete; independent oracle review required**  
Authority granted: **none**

## 1. Candidate boundary

This candidate implements only the read-only scientific worker requested by
`design/chapter06_atmosphere_fixture_oracle_plan.md`:

```text
scripts/chapter06_atmosphere_oracle_worker.py
tests/test_chapter06_atmosphere_oracle_worker.py
```

It begins from the already-published canonical input fixture:

```text
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
363,050 bytes
SHA-256 1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
```

The worker:

1. validates process, source, table, manifest-entry, raw-subset, converter,
   and fixture authority before constructing a selected catalog;
2. reconstructs the three `(80,1006)` public-call arrays only from the
   fixture projections;
3. executes the standard one-record selected-line public call;
4. proves the serial route while making the parallel wrapper raise if called;
5. preserves the returned pre-stimulated float32 slab;
6. constructs and independently checks the float64 stimulated-emission view;
7. returns 134 detached, object-free, C-contiguous in-memory evidence arrays;
8. prints only a compact JSON summary when executed as a script.

It has no output path, serializer, fixture builder, golden reader, writer,
publisher, manifest mutation, overwrite, repair, or alternate destination.
The focused suite statically rejects `--output`, `--publish`, and `--golden`
and checks that the worker contains no NumPy save call or filesystem
write/replace primitive.

This report is a candidate record, not an independent audit, detached
authorization, artifact acceptance, writer design, or publication permission.

## 2. Exact candidate snapshot

After formatting and the final two fresh captures:

| object | SHA-256 |
| --- | --- |
| scientific worker | `6abe225c64cca66a7af56d710b8f2c3d3555114699c47e5211389d5c3d845e5c` |
| focused tests | `0270cb68edc284af20e928be8879d63e816c6f65b4cdd2a287397a274555d499` |
| oracle plan | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| exact source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |
| causal outline | `1b66df5d548f2854f83289fcf9de5109058f1482a7b64aadaff3505d1f57e019` |
| accepted runtime audit | `83366bed79293bb8d02ccdf67b54d0350dcc638888e6678d9dedc7b1d58f3313` |
| accepted fixture worker | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| accepted fixture-worker audit | `336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e` |
| accepted converter | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| accepted converter audit | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |
| fixture publication authorization | `a756fe395bbe9d598fcc4748e7b604920e615a4de9a2c2dabca5942e8a50b9eb` |
| fixture publication-record review | `07518979af4d60d8cbd2321ea6c976a52ba04de53c2fc528af51888a6c42f37b` |

The candidate verifies the converter's literal `CONVERSION_VERSION == 1`
without importing it. It separately verifies the canonical raw subset:

```text
data/subsets/chapter06_fe_i_source_row_873702.npz
8,665 bytes
row 873702
schema version 1
SHA-256 bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04
builder SHA-256 25bcf4662740155e8b08615b9522f3f4517e1a5ddc4627c68686620ccfff4d6c
```

Neither the fixture nor its manifest entry embeds the raw-subset identity or
conversion version. The worker therefore binds them through the unique raw
subset manifest entry and the exact accepted converter/fixture-publication
chain rather than inventing embedded provenance.

## 3. Source and dynamic-read closure

Importing one `payne_zero_atmosphere` submodule executes the package
initializer. The worker consequently verifies the complete accepted 35-file
pinned Python import manifest before import and requires the loaded manifest
to equal it exactly afterward. It also separately verifies all sixteen
critical code identities named in Section 3.1 of the oracle plan, including
the synthesis `atomic_lines.py` raw-row interpretation source.

The final fresh captures recorded exactly three NPZ inputs:

```text
repository:data/fixtures/chapter06_atmosphere_one_line_inputs.npz

pin:source_data_files/atmosphere_tables/line_opacity_tables.npz
  1,359 bytes
  SHA-256 89f486122cb8939b23dc5423145a46d88a77df8daf57a1def35055b7b8205f16

pin:source_data_files/atmosphere_tables/molecular_equilibrium_tables.npz
  1,935 bytes
  SHA-256 1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe
```

The second atmosphere table is an easy-to-miss eager import owned by
`molecular_equilibrium.py`; it supplies `(99,) float64 atomic_mass_amu` and
`(200,) float64 h2_partition_function`. The worker verifies it explicitly
rather than suppressing or laundering the eager read.

No hydrogen profile archive, continuum archive, full source catalog, observed
packed-line catalog, `continuum_level_tables.npz`, or golden was opened.

## 4. Canonical fixture validation

The loader accepts only the exact canonical nonsymlink path. It verifies:

- regular-file role, exact byte count, and file SHA-256;
- one unique fixture manifest entry found by path, never by entry index;
- exact nineteen-member set;
- every shape, dtype, and C-byte SHA-256;
- scientific schema digest
  `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698`;
- accepted fixture-capture payload fingerprint
  `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663`;
- population projection descriptor `[0,2,840]`;
- Fe I line-support/width slot `350`;
- effective-temperature configuration `5778.0 K`;
- strictly increasing finite 30,000-point wavelength grid;
- final packed-bin sentinel `2**30`;
- absence of nonfinite fixture values.

The whole append-only manifest hash and the entry's ordinal position are not
scientific identities. The worker binds the target entry's compact ordered
digest:

```text
cfaf118c11c76a7b97198cb7fe7e3c0f78863f5eee71ea04bbe78c223d3653af
```

This is the already-accepted realized fixture-entry digest and remains stable
when an unrelated later entry is appended.

## 5. Exact scientific result

Both final fresh captures reproduced:

| property | exact result |
| --- | --- |
| reconstructed selected wavelength | `499.03410878793585 nm` |
| first red atmosphere-grid index | `7383` zero based |
| continuum-reference column | `169` zero based |
| pre-stimulated slab | `(80,30000) float32` |
| downstream stimulated view | `(80,30000) float64` |
| selected line count | `1` |
| nonzero values | `240` |
| nonzero values per depth | exactly `3` at all 80 depths |
| physical gate depths | one based `8,16,...,80` |
| active physical gates | all ten |
| peak pre-stimulated opacity | `0.3731258809566498 cm^2 g^-1` |

Exact identities:

```text
pre-stimulated dense
  43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58

post-stimulated dense
  a695164eea303f76f249b1a81ce179426d16e9047cefa079b09a9ff3cfae35ab

sparse C-order index
  9ff0b9443c843eb582ef7d69a1d46ffab6690d938cd213215e6affda680a40f4

sparse pre-stimulated values
  8ad01776bee089b24f3190f45e1bcaee106e7438b7c804c1c263bb78c8fac040

sparse post-stimulated values
  669e5c9928edb9ed7f05b1711b2661e1d4fdfd5b91b06508c94b78745fe996eb
```

The worker retains the plan's exact proposed nine golden projections in its
larger in-memory evidence mapping:

```text
dense_shape
nonzero_flat_index
pre_stimulated_nonzero_value
post_stimulated_nonzero_value
selected_line_count
gate_depth_1based
gate_active
dense_pre_stimulated_sha256
dense_post_stimulated_sha256
```

It does not serialize or publish them.

## 6. Stimulation lifecycle

The worker independently constructs

```text
frequency_hz = 2.99792458e17 / opacity_wavelength_grid_nm
h_over_kt = 6.6256e-27 / (temperature * 1.38054e-16)
stimulated = max(1 - exp(-frequency_hz * h_over_kt), 1e-300)
post = pre.astype(float64) * stimulated
```

It calls pinned
`runner._planck_source_and_stimulated_emission` for every one of the 30,000
wavelength columns and requires bitwise equality with the broadcast
stimulated array.

The pre/post support is exactly identical. The post array is exactly one
float64 multiplication of the preserved float32 pre array by the positive
stimulated factor.

The design phrase “applying the stimulated factor twice fails” is not a
literal production exception: plain NumPy multiplication has no lifecycle
state and succeeds a second time. The enforceable worker evidence is:

1. the accepted post view equals exactly one application;
2. a twice-multiplied array is detectably unequal to the accepted post view;
3. the worker emits only the once-stimulated view;
4. no golden is opened.

The candidate does not invent a stateful production guard merely to make the
design phrase literal.

## 7. Serial route and algorithmic seams

The worker combines source-AST and executed evidence:

- public wrapper contains both guarded parallel and serial dispatches;
- selected line count is exactly one;
- the parallel wrapper is temporarily replaced by a raising sentinel;
- the standard public call still succeeds;
- the compiled route calls FASTEX before its first wing deposit;
- the compiled accumulator calls the wing helper;
- the wing helper calls the compiled Voigt evaluator.

The controlled seam ledger passes:

| seam | evidence |
| --- | --- |
| depth count | 80 succeeds; 79 and 81 raise the exact public `ValueError` |
| fixture axes | malformed 79-row fixture member is rejected by the worker before the kernel |
| center ownership | exact center at test-grid index 200; first red index 201; first blue index 200 |
| point caps | red `201..301` = 101; blue `200..101` = 100; no 100 or 302 |
| first below, red | below-cutoff index 204 retained; next red index absent |
| first below, blue | below-cutoff index 197 retained; next blue index absent |
| wing equality | exact contribution/threshold equality reaches both caps |
| wing nextafter | threshold immediately above equality retains only indices 200 and 201 |
| 8-layer gates | interior depth 10 skipped without gates 8/16 and deposited with either |
| first block | interior depth 1 skipped without gate 8 and deposited with gate 8 |
| final block | interior depth 75 skipped without gates 72/80 and deposited with either |
| center ordering | survival mask `[false,true,false,true]` for pre-below, pre-equal, post-below, post-equal |
| empty allocation | float64 |
| nonempty public line slab | float32 |
| once-stimulated view | float64 |
| selected record | one int32 wavelength plus six int16 fields |
| sanitized compiled inputs | threshold, electrons, support, `hc/kT`, and width are float32 |
| retained float64 inputs | actual populations and wavelength grid |
| synthetic unowned columns | no output change |
| owned actual slots | controlled perturbations of 0, 2, and 840 each change the neutral-collision route |
| owned support/width slot | controlled perturbation of slot 350 changes the line |

The canonical fixture no longer contains the full upstream `(80,1006)`
arrays, so this projections-only worker cannot regenerate full-versus-projected
parity. It instead:

1. binds the exact accepted fixture worker and independent audit that executed
   that parity proof before fixture publication;
2. reconstructs only the accepted projections;
3. executes new synthetic unowned-column and owned-column seam tests.

This is the only interpretation consistent with both plan Section 8.2
(projection-only oracle) and Section 10.7 (full-array parity). The candidate
does not falsely label reconstructed zeros as the lost full upstream state.

## 8. Fresh-process determinism

Two final top-level processes used distinct, initially empty external cache
directories:

```text
/tmp/chapter06-atmosphere-oracle-final-a.XRWt7z
/tmp/chapter06-atmosphere-oracle-final-b.rrrcr6
```

Each cache populated six regular Numba cache files. Neither path, PID, inode,
or origin token enters the capture mapping.

The two 134-member summaries were identical:

```text
capture schema digest
  fe8cf99e2fd47e481a0e413dcf3578f9aee2399c1c00a98289458a0a4b65d428

oracle payload fingerprint
  ba1eae3759b38e326c5104c4131adbf5fd7ac83aad0ee2b4e2d4e072011b00d5

full capture fingerprint
  00d489f5d7d97eb875820423bb2f71ac4d081fc026f82185e5613bee404e018d
```

Both fingerprints hash sorted names, `dtype.str`, shape as contiguous int64
bytes, and contiguous member bytes. The oracle payload excludes only
`meta__*` and `identity__*`. The full fingerprint includes source, table,
fixture-entry, environment, version, command, raw-subset, conversion, and
semantic-conflict evidence. Both exclude the three self-referential
fingerprint fields. Mapping insertion order cannot change either identity.

The focused suite also launches two fresh children and verifies identical
complete summaries, nonempty equal cache inventories, and an exact before/after
hash snapshot of every file under canonical `data/`.

## 9. Published manifest semantic contradictions

The canonical fixture bytes, member schemas, member hashes, and scientific
output are correct. Three already-published manifest labels contradict the
pinned source and plan. The worker verifies their exact current text as
historical publication evidence but sets
`identity__manifest_semantic_labels_authoritative = false`. Calling
`verify_manifest_bindings(treat_array_labels_as_authority=True)` fails closed.

### 9.1 Actual population values

Exact manifest field:

```text
entries[path=data/fixtures/chapter06_atmosphere_one_line_inputs.npz]
  .arrays.actual_population_slot_values.unit
```

Published value:

```text
cm^-3 per partition function
```

Source/fixture evidence:

- the fixture projection descriptor is exactly `[0,2,840]`;
- the accepted construction reads
  `ion_stage_populations_by_packed_slot`;
- the public kernel argument is
  `ion_stage_populations_by_packed_slot`;
- the selected-line wrapper forms neutral collision density directly from
  slots 0, 2, and 840.

These are actual populations in `cm^-3`, not partition-normalized populations.
The worker uses them only under the source-owned actual-population semantics.

### 9.2 Packed species code

Exact manifest field:

```text
entries[path=data/fixtures/chapter06_atmosphere_one_line_inputs.npz]
  .arrays.packed_species_slot.unit
```

Published value:

```text
zero-based packed species slot
```

Source/fixture evidence:

```text
stored int16 code                 3510
abs(code) // 10                  351
kernel array column              350 zero based
```

The stored value is an ordinary packed species code whose decoded population
slot is one based. It is not itself a zero-based slot.

### 9.3 Wavelength bin edges

Exact manifest field:

```text
entries[path=data/fixtures/chapter06_atmosphere_one_line_inputs.npz]
  .arrays.wavelength_bin_edges.unit
```

Published value:

```text
zero-based opacity-wavelength boundary index
```

Source/fixture evidence:

- the array dtype is int64;
- its values are packed wavelength codes;
- the final value is the `2**30` sentinel;
- the compiled kernel compares `packed_wavelength` directly against each
  `bin_edges` value.

The values are not zero-based indices into the 30,000-point grid.

No worker, test, report, source, fixture, or manifest byte was changed to
normalize these contradictions. A repair, if desired, requires a separate
authorized manifest-correction lifecycle.

## 10. Verification

Focused suite:

```text
PYTHONPATH=src:. python -m pytest -q \
  tests/test_chapter06_atmosphere_oracle_worker.py

10 passed in 9.43s
```

Static checks after formatting:

```text
ruff check \
  scripts/chapter06_atmosphere_oracle_worker.py \
  tests/test_chapter06_atmosphere_oracle_worker.py

All checks passed!

ruff format --check \
  scripts/chapter06_atmosphere_oracle_worker.py \
  tests/test_chapter06_atmosphere_oracle_worker.py

2 files already formatted
```

Fresh-process outputs reproduced all exact physical hashes and fingerprints
listed above. The final canonical identities remained:

```text
data/MANIFEST.json
  b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44

data/fixtures/chapter06_atmosphere_one_line_inputs.npz
  1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff

data/subsets/chapter06_fe_i_source_row_873702.npz
  bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04
```

The test-owned complete `data/` snapshot was bitwise identical before and
after both child captures.

## 11. Candidate disposition

The worker is ready for an independent read-only oracle audit at:

```text
worker
  scripts/chapter06_atmosphere_oracle_worker.py
  6abe225c64cca66a7af56d710b8f2c3d3555114699c47e5211389d5c3d845e5c

tests
  tests/test_chapter06_atmosphere_oracle_worker.py
  0270cb68edc284af20e928be8879d63e816c6f65b4cdd2a287397a274555d499
```

Requested independent review should verify:

1. complete preimport authority and 35-module loaded-source closure;
2. the eager molecular-table read;
3. fixture and raw-subset manifest joins;
4. exact dense, stimulated, sparse, gate, and seam evidence;
5. mapping-order-independent fingerprints;
6. no-write/no-golden lifecycle;
7. the projections-only inheritance boundary;
8. all three manifest semantic contradictions and their fail-closed
   non-authority treatment.

No later golden writer or publisher should consume this candidate until that
independent review accepts exact worker and test bytes. This candidate creates
no golden bytes and grants no publication authority.
