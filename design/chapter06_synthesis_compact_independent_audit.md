# Chapter 6 synthesis compact-assembler independent audit

Status: independent in-memory candidate review  
Audited: 2026-07-30  
Disposition: **ACCEPT the compact assembler for candidate-identity freezing
and the next separately reviewed writer/acceptance gate; serialization and
publication remain unauthorized**

No implementation, test, data, manifest, source, paper, external Payne Zero,
candidate archive, raw archive, or golden file was changed or created by this
audit. This report is the only file added.

## 1. Scope and immutable review snapshot

The following files were read in full:

- `design/chapter06_synthesis_fixture_oracle_plan.md`;
- `scripts/chapter06_synthesis_oracle_worker.py`;
- `scripts/chapter06_synthesis_compact_assembler.py`;
- `tests/test_chapter06_synthesis_compact_assembler.py`;
- `design/chapter06_synthesis_compact_candidate.md`.

The reviewed identities are:

| artifact | SHA-256 |
| --- | --- |
| synthesis fixture/oracle plan | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| accepted scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` |
| compact focused tests | `53111433aa3082a58be5ec8b3da1a961f330eac1bb805ac26d1fc6625487e42d` |
| compact candidate report | `41daee0a8b4239fd60a2e67b028afa1a0197ac3ae040a30f5dfd8795234b3550` |
| exact Chapter 6 source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |
| final worker acceptance | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |

The pinned Payne Zero commit remained:

```text
9c44001feae40b85146630499e6f8a5fed42e5af
```

The accepted raw boundary independently reconfirmed by the assembler is:

| property | accepted value |
| --- | --- |
| raw members | `754` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical-payload fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full-capture fingerprint | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` |
| accepted scope | `accepted exhaustive Chapter 6 CPU one-line synthesis capture` |

## 2. Executive verdict

**ACCEPT.**

The assembler is a pure in-memory projection of the already accepted
754-member scientific observation. It does not weaken the worker boundary or
turn itself into a publisher. Independent execution reproduced:

| compact-candidate property | value |
| --- | --- |
| mapping members | `213` |
| array bytes | `1,235,275` |
| schema version | `1` |
| schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| payload fingerprint | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` |
| raw-ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| publication authorized | `False` |
| golden publication performed | `False` |

The candidate is lexical, object-free, C-contiguous, detached from the raw
mapping, and about 29.5% of the 4 MiB array-byte ceiling. Its only dense
`(regime=4, depth=6, wavelength=6000)` members are the required gross and net
`float32` line-opacity slabs.

The complete raw observation is covered exactly once by the ownership ledger:

| disposition | count |
| --- | ---: |
| `final` | `250` |
| `derived_digest_only` | `123` |
| `intentionally_ephemeral` | `381` |
| total | `754` |

No P0 or P1 defect was found in ownership, scientific reduction,
determinism, mutation isolation, identity binding, or authority separation.

## 3. Exact raw ownership

### 3.1 Complete and unique classification: PASS

The assembler rejects any raw count, schema, scope, worker identity,
fingerprint, or science drift before projection. `_complete_ownership`
requires:

- every accepted raw name to appear exactly once;
- no ownership name absent from the accepted raw mapping;
- lexical and unique ownership order;
- one of exactly three declared dispositions;
- every `final` target root to exist in the compact mapping.

Independent checks reproduced:

```text
ownership entries:                 754
unique lexical names:              754
final targets resolving:           all
ownership digest recomputation:    exact
```

The ownership tuple binds each raw name, disposition, target, and reason. A
removed raw member, an extra raw member, or a shortened ownership tuple fails
closed.

### 3.2 Section 11 projection: PASS

The compact mapping retains the exact comparison-owned material:

- one regime axis, one depth axis, and one canonical 6000-point wavelength
  axis;
- the 13 derived physical line fields;
- required empty-helium constructor inputs;
- exact Harris and FASTEX member identities rather than table copies;
- all one-element ordinary invariants;
- all exact zero-length autoionizing and helium invariant fields with their
  source dtypes and physical units;
- the canonical factor/branch ledger;
- one gross and one net canonical line slab;
- center/wing continuum samples and continuum digests;
- exact stimulation reconstruction evidence;
- compact coarse-grid geometry, activity, peak, reach, and nonzero-count
  evidence;
- source, fixture, subset, table, environment, worker, schema, and ownership
  identities.

The following raw families remain deliberately ephemeral:

- all selected Chapter 5 `continuum_state` and `line_state` arrays;
- selected population, width, density, collision, and detailed perturbation
  intermediates not owned by the comparison artifact;
- complete upstream field-name inventories already owned by their input
  artifacts;
- worker-local absolute paths and verified lifecycle flags.

The classification agrees with the fixture/oracle plan. It does not
reclassify an upstream computed state as a Chapter 6 output.

## 4. No-copy and no-forbidden-array audit

### 4.1 Upstream state is absent: PASS

No candidate member has a forbidden state shape `(6,6,139)`. No
`continuum_state` or `line_state` raw name survives. The complete Chapter 5
fixture remains an independently owned input, represented only by its
identity, payload digest, and relative path.

An independent all-pairs memory check found:

```text
candidate/raw shared-memory pairs: 0
```

The builder copies every accepted value into C-contiguous NumPy storage and
copies again when finishing the lexical mapping. Mutating the raw mapping
after assembly leaves the candidate unchanged.

### 4.2 Continuum, table, route, and coarse duplicates are absent: PASS

The candidate contains:

- no full continuum absorption, scattering, total, or float32 cutoff slab;
- no array of length `1001` or `2001`, hence no FASTEX or Harris numerical
  table;
- no loop-mode opacity slab;
- no `(6,400)` or `(4,6,400)` coarse slab;
- no coarse 400-point wavelength axis;
- no raw source-catalog array;
- no atmosphere, hydrogen, helium-line, autoionizing, molecular, forest, or
  many-line output.

Independent shape inventory:

```text
forbidden state/coarse/table shapes: 0
dense (4,6,6000) members:
  opacity__gross_float32
  opacity__net_float32
```

Full continuum arrays are discarded only after
`absorption + scattering == total`, `total.astype(float32) == line_input`,
and center/wing sample reconstruction. Full stimulation factors are discarded
only after both loop and batched net slabs reconstruct exactly. Full support
tables are replaced by dtype/shape/byte member hashes only after catalog and
both invariant-grid copies agree.

### 4.3 Equal-byte semantic fields are not forbidden payload duplication: PASS

The audit explicitly examined all equal-byte candidate groups. Small fields
can have equal bytes while owning different required semantics:

- mapping fields versus transformed invariant fields;
- center versus wing indices for this one centered record;
- canonical versus coarse activity evidence;
- pre-route versus post-route source-manifest checks;
- staged versus pinned source identities;
- exact empty-route fields.

The only equal nonidentity numerical comparison arrays are the 24-value
center and wing continuum samples. They are equal because the accepted record
maps both center and wing anchors to index `2434`, but the two labels preserve
the independently checked center-cutoff and wing-cutoff roles required by the
plan. They are 96 bytes each and do not duplicate their source continuum
slab. This is semantic checkpoint ownership, not an undeclared copied input
or output product.

## 5. Scientific reduction and reconstruction

### 5.1 Route deduplication: PASS

For canonical and coarse grids in all four regimes, the assembler requires:

```text
gross_batched == gross_loop       bitwise
net_batched   == net_loop         bitwise
net == multiply(gross, complete_stimulated_factor, dtype=float32)
```

Only the canonical batched gross/net slabs survive. Coarse batched slabs are
reduced to per-regime peaks only after equality and stimulation checks; loop
slabs never survive.

Mutation of a loop slab or full stimulated factor failed with the exact
deduplication/reconstruction error rather than being absorbed by a tolerance.

### 5.2 Continuum ownership and float32 fence: PASS

For every regime and grid, the assembler independently reconstructs:

```text
continuum_total_float64
    = continuum_absorption_float64 + continuum_scattering_float64

continuum_line_input_float32
    = continuum_total_float64.astype(float32)
```

The center and wing samples must equal the corresponding columns of the
float32 cutoff slab. A one-ULP absorption mutation fails reconstruction.

The final mapping retains four component hashes per regime/grid and only the
small cutoff samples. It does not become a second Chapter 5 continuum golden.

### 5.3 Catalog, invariant, and support deduplication: PASS

The 13 `record__` and `catalog__` fields must be byte-identical before the
single `mapping__` representation is retained. Canonical and coarse
invariants must agree wherever the field is grid-independent. Canonical,
coarse, and catalog Harris arrays must agree before their numerical arrays are
discarded. FASTEX arrays must agree between grids before their hashes are
retained.

A one-ULP Harris mutation fails this boundary. The exact zero-length auto and
helium fields remain present because emptiness, dtype, and axis are part of
the production constructor contract, not because they carry a physical
special-line contribution.

## 6. Schema and payload determinism

### 6.1 Fresh-process stability: PASS

Two unrelated fresh-cache processes reproduced the same:

- 213-member count;
- 1,235,275 array bytes;
- schema digest;
- payload fingerprint;
- ownership digest and counts;
- gross/net shapes;
- false publication flags.

The payload fingerprint covers every lexical member's name, dtype, shape, and
contiguous bytes except its self-referential fingerprint field. The schema
digest separately covers every name, dtype, and shape. Building twice from
one accepted raw mapping also produced equal keys, schema objects, ownership
tuples, and every array.

### 6.2 Schema descriptions: PASS

Every one of the 213 members has exactly one `CompactMemberSpec` with:

- exact lexical name;
- NumPy dtype string;
- exact shape;
- ordered axes matching rank;
- physical unit or convention;
- role `axis` or `comparison-only`.

All values are C-contiguous and object-free. Removing a schema entry fails
coverage validation.

### 6.3 Mutation detection and isolation: PASS

The focused fresh-process harness rejected all eleven declared mutations:

1. removed raw member;
2. extra raw member;
3. unaccepted raw scope;
4. raw physical-payload mutation;
5. stimulated-factor reconstruction break;
6. loop/batched equality break;
7. continuum reconstruction break;
8. Harris deduplication break;
9. compact payload mutation;
10. compact schema mutation;
11. compact ownership mutation.

The raw physical and full fingerprints are recomputed, not merely read from
metadata. Candidate payload, schema, and ownership digests are likewise
recomputed. Publication flags receive independent semantic false checks in
addition to fingerprint coverage.

## 7. Source and worker identity binding

### 7.1 Accepted worker boundary: PASS

Before accepting any raw mapping, the assembler hashes the current worker and
requires:

```text
36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68
```

It then requires the raw worker metadata, accepted scope, schema digest,
physical fingerprint, and full fingerprint to match the independently
accepted values. Re-running the worker science validator is also mandatory.

Independent comparison showed that current and captured identities match for:

| bound authority | SHA-256 |
| --- | --- |
| scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| fixture/oracle plan | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| exact source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |
| compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` |

The assembler identity is written into the candidate metadata and is frozen
by this independent review. The test identity is frozen in Section 1.

### 7.2 Staged/upstream source parity: PASS

Independent byte comparisons reproduced:

| staged source | SHA-256 |
| --- | --- |
| `atomic_lines.py` | `0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0` |
| `constants.py` | `ed58004196790f9fb4a2871044c9cd36bf7bc42046a923f9314f7b8ea7456798` |
| `continuum.py` | `ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b` |
| `device.py` | `22e769ebed60ad3a0f2060264247e469a99afd20ec5cadb69a01b6e5fa82ea3c` |
| `line_opacity.py` | `639b95c3812f1a7d227b797fa89a4d6ef9725d5f0e1284f3d49cf86844278275` |

The staged files were byte-identical to the corresponding files in the
pinned checkout.

The static tables were also byte-identical:

| staged table | SHA-256 |
| --- | --- |
| `continuum_tables.npz` | `406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d` |
| `continuum_edge_grid.npz` | `11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e` |
| `line_profile_tables.npz` | `87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4` |

The worker identity-only route reconfirmed the pinned commit, the 217-member
Chapter 5 fixture, and the 27-member Chapter 6 subset at their accepted
hashes.

## 8. Serialization and publication authority

### 8.1 Assembler API boundary: PASS

The only assembler input is:

```python
assemble_compact_candidate(raw)
```

The module exposes:

- no command-line `main`;
- no argument parser;
- no output or destination path;
- no NPZ or other serializer;
- no manifest mutation;
- no authorization parser;
- no publication function;
- no alternate-root or replacement option.

Static source inspection found no file-writing call. The assembler reads only
its own and the accepted worker's bytes for identity checks and returns an
in-memory `CompactAssembly`.

### 8.2 Lifecycle flags: PASS

The worker raw capture is accepted only when:

```text
golden read performed:          False
golden publication performed:  False
```

The compact candidate independently records:

```text
publication authorized:        False
golden publication performed:  False
```

The candidate report accurately states that no raw archive, candidate NPZ,
golden, manifest mutation, detached authorization, or publication occurred.
Neither the candidate report nor this audit grants publication authority.

## 9. Executed verification

Observed during this independent review:

```text
python -m pytest -q tests/test_chapter06_synthesis_compact_assembler.py
4 passed in 4.85s

python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_compact_assembler.py
13 passed in 8.02s

PYTHONPATH=src:. python -m pytest -q
386 passed, 1 skipped in 64.90s

ruff check scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
All checks passed!

ruff format --check scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
2 files already formatted

python -m py_compile scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
pass
```

The repository-wide count is larger than the earlier candidate-report count
because additional tests have since entered the active repository. The
focused immutable compact hashes and results remain unchanged.

## 10. Acceptance scope and remaining gates

This audit **accepts and freezes for the next gate**:

```text
assembler SHA-256:
  62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a

focused-test SHA-256:
  53111433aa3082a58be5ec8b3da1a961f330eac1bb805ac26d1fc6625487e42d

compact member count:
  213

compact array bytes:
  1,235,275

compact schema digest:
  911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde

compact payload fingerprint:
  ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9

raw ownership digest:
  5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675

raw ownership counts:
  final=250
  derived_digest_only=123
  intentionally_ephemeral=381
```

This acceptance does **not** authorize:

1. any raw-capture serialization;
2. any final-candidate serialization;
3. a deterministic NPZ writer;
4. a candidate or golden file;
5. manifest mutation;
6. detached authorization;
7. atomic publication;
8. overwrite, repair, merge, or alternate destinations;
9. CUDA or MPS tolerance claims.

The next implementation may consume only the exact accepted assembler/test
hashes and compact identities above. It must independently prove deterministic
raw and final A/B bytes, enforce the archive-byte ceiling, validate the
candidate from its serialized form, and remain blocked from publication until
the separate detached authorization gate is present and exact.

## 11. Final disposition

**ACCEPT the completed Chapter 6 synthesis compact assembler and its
213-member in-memory comparison candidate.**

The acceptance is limited to pure in-memory assembly and the exact identities
recorded here. The candidate has complete raw ownership, no prohibited copied
state or numerical-table payload, deterministic schema and values, mutation
and no-copy protection, exact worker/source binding, and no serialization or
publication authority.
