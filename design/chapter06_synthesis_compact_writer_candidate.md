# Chapter 6 synthesis deterministic compact-writer candidate

Status: implementation candidate for the deterministic in-memory writer gate;
independent writer and serialized-candidate review are pending, and no golden,
manifest entry, detached authorization, publisher, or publication is
authorized

Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

This revision repairs both blocking findings in
`design/chapter06_synthesis_compact_writer_independent_audit.md`, SHA-256
`e2a3607849289c3a4e0c67272d6a9583d35991f4df758f6526e8619fd8f891ab`:
the public supplied-raw bypass is removed, and fresh capture A/B now use
distinct disposable cache roots with explicit origin evidence.

## Scope

This pass implemented:

```text
scripts/chapter06_synthesis_compact_writer.py
tests/test_chapter06_synthesis_compact_writer.py
```

The writer consumes the independently accepted synthesis worker and compact
assembler. Its sole public production builder has no arguments and always
owns two raw observations constructed in separate fresh Python child
processes. It then:

1. verifies the exact accepted worker/assembler review snapshot;
2. requires raw A/B canonical bytes and every raw dtype, shape, and C byte to
   agree;
3. independently assembles and validates compact A and compact B;
4. requires identical compact schemas, payloads, and 754-entry ownership
   ledgers;
5. serializes each 213-member mapping independently;
6. deserializes each archive and requires exact member dtype, shape, C bytes,
   semantic validation, and canonical re-encoding;
7. requires final archive A and archive B to be byte-identical;
8. returns one immutable byte string plus a JSON-safe summary.

The module has no caller-supplied raw-observation bypass, output or destination
path, command-line entry point, artifact-file writer, manifest mutation,
authorization parser, publisher, canonical archive, overwrite, rename,
repair, or alternate-root operation. It creates only two empty disposable
Numba cache directories as process-control state and removes them before
returning. No raw or final NPZ file was created during this pass.

## Candidate implementation identities

| file | SHA-256 |
| --- | --- |
| deterministic compact writer | `3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb` |
| focused writer tests | `9601f8717a29ef51f32a62fd0a73c4d3db4b41c4f6ac08374be6149df4030bfc` |

The writer binds these eight previously accepted or frozen inputs before
capture or serialization:

| bound input | SHA-256 |
| --- | --- |
| synthesis fixture/oracle plan | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| accepted scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| accepted worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| scientific-worker independent acceptance | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |
| accepted compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` |
| accepted compact-assembler tests | `53111433aa3082a58be5ec8b3da1a961f330eac1bb805ac26d1fc6625487e42d` |
| compact candidate record | `41daee0a8b4239fd60a2e67b028afa1a0197ac3ae040a30f5dfd8795234b3550` |
| compact-assembler independent acceptance | `a0530cd08d5f0ddcc96b51fdaab4520aa89e62ca65850cf855ad4ede22251a33` |

Every path must be a regular nonsymlink file. The imported worker and assembler
paths must equal the bound canonical repository paths. One changed expected
identity fails before raw capture or compact serialization.

## Two independent raw observations

The accepted worker cannot execute twice in one Python process because its
one-thread Torch inter-op initialization is deliberately one-shot. The
writer's construction route therefore launches two separate read-only child
processes. Each child:

- receives its own parent-created empty nonsymlink `NUMBA_CACHE_DIR`;
- uses a cache root external to the textbook, pinned Payne Zero, paper, and
  data/source trees;
- inherits the already-required deterministic worker environment;
- builds one accepted 754-member observation in memory;
- echoes a parent-generated one-use origin token, child PID, and cache-root
  identity in a framed evidence header;
- emits one transient deterministic object-free mapping through its stdout
  pipe;
- receives no output path and creates no raw file.

The parent verifies both cache roots are still empty and nonsymlink after each
child exits. It rejects equal origin tokens, child PIDs, or cache-root
identities before comparing any scientific value. Both directories are
removed before the builder returns. The parent immediately validates and
discards the two transient raw byte strings after assembly; they are never
returned as an artifact.

The public production signature is exactly:

```python
build_deterministic_compact_archive()
```

There is no supplied-mapping overload. Private raw/value helpers cannot create
an accepted result; the production builder alone returns
`DeterministicCompactArchive`, after checking the child-owned origin evidence.

Independent A/B results:

| raw property | accepted candidate value |
| --- | --- |
| raw members | `754` |
| raw schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full fingerprint | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` |
| complete raw mapping digest | `e2a3dd9670fd9fb5ed34f5131c303687d72ff2d3e762d9224c702f1dffc4a775` |
| transient raw NPZ bytes | `8,689,108` |
| transient raw NPZ SHA-256 | `5ff93594781de6fc12a3a2a0adf0ebcc03c81f6141b8648e709ad306686b93a3` |
| raw A/B archive equality | bitwise exact |
| raw A/B array equality | exact dtype, shape, and C bytes |
| child origin tokens | two distinct SHA-256-bound tokens |
| child processes | two distinct positive PIDs |
| cache roots | two distinct external nonsymlink directories |
| cache contents | empty before and after both captures |
| cache lifecycle | both roots disposed before return |

The transient exhaustive raw transport is intentionally larger than 4 MiB.
The 4 MiB publication ceiling belongs to the compact final candidate, not to
the unpublished 754-member audit observation. The raw bytes remain
pipe-local, are neither returned nor written, and grant no raw-capture
artifact or publication authority.

### Process-independence evidence

The returned summary binds, outside the deterministic scientific archive:

```text
capture A/B origin-token SHA-256
capture A/B child PID
capture A/B cache-root SHA-256
distinct-origin decision
distinct-child-process decision
distinct-cache-root decision
external/nonsymlink cache decisions
empty-before/empty-after decisions
cache-disposal decision
SHA-256 of the complete process-evidence mapping
```

Origin tokens, PIDs, and cache paths are intentionally absent from the 213
scientific arrays and final NPZ bytes. They differ between top-level runs,
while the raw scientific transport and compact archive remain byte-identical.
This separates process provenance from scientific determinism.

## Accepted compact input boundary

Both independently assembled candidates reproduce:

| compact property | value |
| --- | --- |
| members | `213` |
| array bytes | `1,235,275` |
| schema version | `1` |
| schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| payload fingerprint | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` |
| raw ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| ownership counts | `final=250`, `derived_digest_only=123`, `intentionally_ephemeral=381` |

Before writing bytes, each assembly passes the accepted assembler's complete
scientific, schema, size, role, and ownership validator. The writer then
independently requires:

```text
schema A == schema B
ownership A == ownership B
member names A == member names B in lexical order
every dtype, shape, and C byte A == B
```

An A/B schema, ownership, member-set, or payload disagreement fails without
returning candidate bytes.

## Canonical NPZ encoding

The final byte string is a standard uncompressed NPZ with:

| ZIP/NPY property | fixed value |
| --- | --- |
| member count | `213` |
| member order | lexical compact-array name order |
| member suffix | `.npy` |
| compression | `ZIP_STORED` |
| ZIP date/time | `1980-01-01 00:00:00` |
| ZIP creator system | Unix, code `3` |
| ZIP create/extract version | `20` / `20` |
| ZIP flags | `0` |
| internal attributes | `0` |
| external attributes | regular file mode `0600`, fixed integer `0o100600 << 16` |
| member extra/comment | empty |
| archive comment | empty |
| NPY version | `2.0` |
| pickle | forbidden |
| object dtype | forbidden |
| array storage | C-contiguous |
| Zip64 | disabled |

Internal `/` separators already present in a small number of accepted
identity-member names are retained. Absolute names, backslashes, empty/dot
path components, duplicate names, directories, and ambiguous base names ending
in `.npy` are rejected.

The exact in-memory final candidate is:

```text
bytes
  1,294,865
SHA-256
  b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850
```

This is 30.9% of the 4 MiB final archive ceiling. Final A and final B are
byte-identical, and unrelated top-level fresh runs reproduce this same byte
count and SHA-256.

## Serialized-form validation

The reader validates the byte string as an archive, not merely as a mapping:

1. immutable nonempty `bytes` below the 4 MiB ceiling;
2. empty archive comment;
3. exact lexical and unique 213-member set;
4. safe member names and exact fixed ZIP metadata;
5. stored size equals uncompressed size;
6. canonical NPY 2.0 bytes with `allow_pickle=False`;
7. object-free, C-contiguous arrays;
8. exact dtype, shape, and C bytes against the independently assembled
   candidate;
9. complete reconstructed `CompactAssembly` semantic validation;
10. canonical re-encoding equals the entire input byte string.

The final whole-byte equality rejects otherwise readable alternate encodings,
including changed member order, timestamps, permissions, headers, comments,
extra fields, compression, trailing bytes, and archive mutations.

## Adversarial verification

The focused suite covers:

- one changed accepted dependency identity;
- the zero-argument-only public production signature;
- rejection of caller-supplied raw arguments;
- reuse of the same capture object;
- a deep-copied raw mapping with the same origin evidence;
- reuse of one child PID under a changed token/cache;
- distinct child evidence sharing one cache root;
- occupied and symlink cache roots before child launch;
- a cache root inside a forbidden source/data tree;
- equal cache roots at the A/B topology boundary;
- exact distinct-origin/cache claims and evidence digests on two unrelated
  top-level runs;
- raw A/B member or value disagreement;
- compact A/B schema disagreement;
- compact A/B ownership disagreement;
- compact A/B payload disagreement;
- object dtype at the serialization boundary;
- truncated archive;
- mutated member bytes and CRC failure;
- reverse/nonlexical member order;
- changed ZIP timestamp;
- exact fixed metadata on all 213 members;
- canonical NPY 2.0 member headers;
- complete serialized-form reconstruction;
- two unrelated top-level A/B builds;
- complete textbook `data/` byte snapshots before and after;
- empty child Numba caches before and after capture;
- removal of both disposable child cache roots before return;
- static absence of artifact-file and publication calls.

Executed:

```text
ruff check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
All checks passed!

ruff format --check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
2 files already formatted

python -m pytest -q tests/test_chapter06_synthesis_compact_writer.py
6 passed in 15.41s

python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_writer.py
19 passed in 24.72s

PYTHONPATH=src:. python -m pytest -q
396 passed, 1 skipped in 86.70s
```

## Authority boundary and remaining gates

Every returned summary records:

```text
publication_authorized         false
golden_publication_performed   false
manifest_mutation_performed    false
artifact_file_write_performed  false
```

It separately records that two disposable cache directories were created,
verified empty, and removed. Those directories are required process-control
state, not raw/final artifacts and not publication destinations.

This candidate report does not grant publication authority. Before any
canonical synthesis golden can exist, the exact writer, tests, candidate
report, final bytes, fixed archive metadata, deserialized schema, and
scientific content require independent review. A separate deterministic
publisher, candidate-artifact acceptance, cycle-free detached authorization,
atomic no-overwrite installation, and later manifest synchronization remain
unimplemented and unauthorized.
