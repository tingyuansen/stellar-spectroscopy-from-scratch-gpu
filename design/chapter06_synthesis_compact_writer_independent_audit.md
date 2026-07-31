# Chapter 6 synthesis deterministic compact-writer independent audit

Status: independent writer and serialized-candidate review  
Audited: 2026-07-30  
Disposition: **REJECT for freezing or publication-gate use until raw A/B
independence is made non-bypassable**

The deterministic archive encoding, compact semantic validation, and default
fresh-child execution all reproduced exactly. The rejection is narrower but
blocking: the public builder accepts caller-supplied `raw_a` and `raw_b`, and
accepts the *same raw mapping object* as both observations. It can therefore
return the accepted candidate bytes and claim A/B success without conducting
two independent observations.

No implementation, test, source, data, manifest, external Payne Zero,
candidate archive, raw archive, golden, authorization, or publication was
changed or created by this review. This report is the only repository file
added.

## 1. Reviewed snapshot

The following required files were read in full:

- `design/chapter06_synthesis_fixture_oracle_plan.md`;
- `design/chapter06_synthesis_worker_independent_audit.md`;
- `design/chapter06_synthesis_compact_candidate.md`;
- `design/chapter06_synthesis_compact_independent_audit.md`;
- `scripts/chapter06_synthesis_oracle_worker.py`;
- `scripts/chapter06_synthesis_compact_assembler.py`;
- `scripts/chapter06_synthesis_compact_writer.py`;
- `tests/test_chapter06_synthesis_compact_writer.py`;
- `design/chapter06_synthesis_compact_writer_candidate.md`.

The exact review identities were:

| artifact | SHA-256 |
| --- | --- |
| deterministic compact writer | `0dc2f250b23ddc226d8769102dca3050b97df96abe3a32c2468d7aa063437e70` |
| focused writer tests | `f43eee8c6894179dc3e862ff42219b3421f87818a3a8bf906a41b80e2e385980` |
| writer candidate report | `2c80f799f27c375ca7bc7569225df8585d46bd655a370308bcb3f8379e4dd7b9` |
| synthesis fixture/oracle plan | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| accepted scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| accepted worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| scientific-worker independent acceptance | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |
| accepted compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` |
| accepted compact-assembler tests | `53111433aa3082a58be5ec8b3da1a961f330eac1bb805ac26d1fc6625487e42d` |
| compact candidate report | `41daee0a8b4239fd60a2e67b028afa1a0197ac3ae040a30f5dfd8795234b3550` |
| compact-assembler independent acceptance | `a0530cd08d5f0ddcc96b51fdaab4520aa89e62ca65850cf855ad4ede22251a33` |
| exact Chapter 6 source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |

The writer's eight-entry accepted-identity gate reproduced all eight hashes
embedded in its implementation. The imported worker and assembler also
resolved to the bound canonical repository paths.

The pinned Payne Zero commit remained:

```text
9c44001feae40b85146630499e6f8a5fed42e5af
```

## 2. Blocking finding

### P0 — the public archive builder does not require two fresh observations

The public function is:

```python
build_deterministic_compact_archive(raw_a=None, raw_b=None)
```

When both arguments are absent, it launches two child processes. When both
are supplied, however, it serializes and compares the supplied mappings
without establishing how or when either mapping was observed.

The independent adversarial probe supplied one accepted mapping as both
arguments:

```python
raw, _ = writer._capture_raw_in_fresh_child()
result = writer.build_deterministic_compact_archive(
    raw_a=raw,
    raw_b=raw,
)
```

This was accepted. The exact observed result was:

```text
raw_a is raw_b:                 True
builder returned bytes:        True
summary raw_a_b_bitwise_equal: True
archive bytes:                 1,294,865
archive SHA-256:
  b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850
```

The accepted archive identity is therefore reachable from only one scientific
observation. Passing two shallow aliases, two deep copies of one capture, or
two separately decoded copies of one raw byte string is equally incapable of
proving fresh-process independence. Rejecting only `raw_a is raw_b` or only
shared NumPy buffers would not close the provenance defect.

This violates the explicit gate that raw A and raw B be genuinely independent
fresh captures before their equality is used as determinism evidence. It also
means a future publisher could select the supplied-pair path and bypass the
two-capture proof while receiving indistinguishable accepted bytes.

#### Required closure

The publication-capable construction boundary must own the observations:

1. remove caller-supplied `raw_a` and `raw_b` from
   `build_deterministic_compact_archive`, or separate that supplied-mapping
   operation into a clearly non-accepting diagnostic helper that cannot
   return an accepted archive;
2. make the accepted builder always launch two fresh child processes itself;
3. add a focused test proving the accepted public builder has no
   supplied-observation bypass;
4. retain the existing full raw-byte, member, dtype, shape, C-byte, compact
   schema, payload, and ownership comparisons after the two owned captures.

Freshness cannot be inferred from equal in-memory values. It must follow from
the construction topology.

### P1 — the two inner children inherit one cache pathname

`_capture_raw_in_fresh_child` copies `os.environ` but does not replace
`NUMBA_CACHE_DIR`. Both calls made by one top-level build therefore receive
the same cache pathname.

The current route remains scientifically clean because:

- each worker requires that directory to be empty at process start;
- the accepted route leaves it empty;
- any first-child cache population would make the second child fail.

Nevertheless, the two captures do not receive the unrelated cache roots
specified by the A/B lifecycle design. This does not explain any byte
difference—the directory remained empty in every review run—but it weakens
the intended process-isolation proof.

The repaired accepted builder should assign a distinct external, empty,
nonsymlink cache directory to each child and verify both remain empty. These
directories are disposable process-control state, not raw or final artifacts.

## 3. Independently reproduced raw and compact identities

Two unrelated top-level review processes, each started with its own external
empty cache root, reproduced:

| property | exact value |
| --- | --- |
| raw members | `754` |
| raw deterministic transport bytes | `8,689,108` |
| raw transport SHA-256 | `5ff93594781de6fc12a3a2a0adf0ebcc03c81f6141b8648e709ad306686b93a3` |
| complete raw mapping digest | `e2a3dd9670fd9fb5ed34f5131c303687d72ff2d3e762d9224c702f1dffc4a775` |
| raw schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| raw physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| raw full fingerprint | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` |
| compact members | `213` |
| compact array bytes | `1,235,275` |
| compact schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| compact payload fingerprint | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` |
| raw ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |

The ownership counts also reproduced exactly:

```text
final                     250
derived_digest_only       123
intentionally_ephemeral   381
total                     754
```

On the default child-process route, the two returned raw dictionaries were
distinct objects, all 754 corresponding arrays had zero shared-memory pairs,
and their deterministic raw transports were byte-identical. This confirms
that the default route itself has separate process memory; it does not close
the optional supplied-pair bypass.

Each raw observation was independently passed through the accepted compact
assembler. Compact A and B agreed on:

- the complete 213-member lexical set;
- every schema description;
- all 754 ownership entries;
- every dtype, shape, and C byte;
- all accepted compact identities and ownership counts.

## 4. Deterministic archive encoding

The independently reproduced final candidate was:

```text
archive bytes:
  1,294,865

archive SHA-256:
  b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850
```

This is below the 4 MiB compact archive ceiling. Archive A and B were
byte-identical, and the two unrelated top-level runs reproduced the same
whole-file identity.

All 213 members had:

| property | required and observed value |
| --- | --- |
| order | lexical member-name order |
| member uniqueness | exact |
| compression | `ZIP_STORED` |
| ZIP date/time | `1980-01-01 00:00:00` |
| creator system | Unix code `3` |
| create/extract version | `20` / `20` |
| flag bits | `0` |
| internal attributes | `0` |
| external attributes | regular file mode `0600`, exact `0o100600 << 16` |
| member extra/comment | empty |
| archive comment | empty |
| NPY version | `2.0` |
| object dtype | forbidden and absent |
| array order | C-contiguous |
| Zip64 | disabled |

No ambient timestamp or permission entered the bytes. The fixed permissions
are encoded explicitly and checked after decoding.

## 5. Decode, canonicality, and semantic validation

The serialized-form boundary passed all of the following independent checks:

1. immutable, nonempty bytes below the exact size ceiling;
2. lexical, unique, safe member names;
3. exact fixed ZIP metadata for every member;
4. exact NPY 2.0 prefix and `allow_pickle=False`;
5. object-free, C-contiguous decoded arrays;
6. exact dtype, shape, and C bytes against an independently assembled compact
   mapping;
7. reconstruction of a complete `CompactAssembly`;
8. the accepted compact scientific, role, schema, payload, size, and
   ownership validator;
9. complete re-encoding to the identical input byte string.

The whole-archive re-encoding check closes alternate readable encodings, not
merely malformed archives. Independent probes rejected:

- a canonical archive containing a one-ULP compact payload mutation;
- a canonical archive containing a compact shape mutation;
- changed ZIP permissions;
- NPY version 1.0 for one otherwise valid member;
- `ZIP_DEFLATED` instead of `ZIP_STORED`;
- a nonempty archive comment;
- a changed ZIP timestamp;
- reverse/nonlexical member order;
- a member-byte corruption;
- a truncated archive;
- object dtype at the writer boundary.

The valid-CRC payload and shape mutations reached the comparison/semantic
boundary and were rejected there. Thus corruption rejection does not rely
only on CRC failures.

## 6. Authority and filesystem boundary

Static and dynamic review found no:

- output or destination path;
- `open`, NumPy save, `Path.write_*`, rename, replace, or publication call;
- command-line entry point;
- golden or manifest path;
- authorization parser;
- publisher, overwrite, repair, merge, or alternate-root operation.

The raw NPZ bytes exist only as child stdout transport, and the compact NPZ is
returned only as immutable in-memory bytes. Canonical textbook `data/` bytes
were unchanged across the focused fresh-process tests. Every returned summary
records false for publication authorization, golden publication, manifest
mutation, and filesystem archive writing.

This positive boundary does not grant serialization acceptance while the P0
freshness bypass remains.

## 7. Executed verification

Focused writer suite:

```text
python -m pytest -q tests/test_chapter06_synthesis_compact_writer.py
5 passed in 14.87s
```

Combined accepted-worker, compact-assembler, and writer suites:

```text
python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_writer.py
18 passed in 21.64s
```

Repository-wide suite:

```text
PYTHONPATH=src:. python -m pytest -q
393 passed, 1 skipped in 81.51s
```

Targeted static checks:

```text
ruff check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
All checks passed!

ruff format --check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
2 files already formatted

python -m py_compile \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
pass
```

The passing suite does not contradict the rejection. The current focused
tests verify equality and corruption handling but do not test that the
accepted public builder cannot consume one observation twice.

## 8. Final disposition

**REJECT the current deterministic compact writer as an accepted A/B gate.**

The following components are independently confirmed and should be preserved:

- all accepted source, worker, assembler, schema, payload, and ownership
  bindings;
- the exact raw transport and final candidate byte identities;
- lexical object-free NPY-v2/ZIP_STORED serialization;
- fixed ZIP headers with no ambient timestamps or permissions;
- strict canonical decode/re-encode equality;
- full dtype/shape/C-byte and compact semantic validation;
- fail-closed corruption handling;
- the absence of a filesystem publication surface.

Acceptance remains blocked until the public accepted builder always owns two
fresh child observations, cannot accept supplied observations as a substitute,
and gives those children unrelated disposable cache roots. No deterministic
candidate acceptance, detached authorization, publisher, manifest mutation,
or golden publication is authorized by this report.

---

## Repair re-audit — 2026-07-30

Status: independent read-only re-audit of the repaired writer  
Disposition: **ACCEPT the repaired deterministic in-memory compact writer for
the next separately reviewed acceptance/publisher gate; publication remains
unauthorized**

This section supersedes the rejection above only for the exact repaired
snapshot recorded here. The original rejection remains intact as the audit
trail for the removed supplied-observation bypass and shared-cache topology.
No implementation, test, candidate, data, manifest, external Payne Zero,
paper, raw archive, compact archive, golden, authorization, or publication
artifact was changed or created by this re-audit. This appended section is the
only repository mutation.

### R1. Exact repaired snapshot

The three artifacts assigned for re-audit matched their expected identities
before any execution:

| repaired artifact | reviewed SHA-256 |
| --- | --- |
| deterministic compact writer | `3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb` |
| focused writer tests | `9601f8717a29ef51f32a62fd0a73c4d3db4b41c4f6ac08374be6149df4030bfc` |
| repaired candidate report | `540cf57126df93ee34d02c3da446a6ef109b93a8b17d60514439e12a8f63fc71` |
| original independent rejection, before this append | `e2a3607849289c3a4e0c67272d6a9583d35991f4df758f6526e8619fd8f891ab` |

The repaired writer's complete eight-file accepted-input gate also
reproduced:

| bound input | SHA-256 |
| --- | --- |
| synthesis fixture/oracle plan | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| accepted scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| accepted worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| scientific-worker independent acceptance | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |
| accepted compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` |
| compact-assembler tests | `53111433aa3082a58be5ec8b3da1a961f330eac1bb805ac26d1fc6625487e42d` |
| compact candidate report | `41daee0a8b4239fd60a2e67b028afa1a0197ac3ae040a30f5dfd8795234b3550` |
| compact-assembler independent acceptance | `a0530cd08d5f0ddcc96b51fdaab4520aa89e62ca65850cf855ad4ede22251a33` |

The pinned Payne Zero checkout still resolved to commit
`9c44001feae40b85146630499e6f8a5fed42e5af`. The accepted scientific
worker continues to bind its frozen complete loaded-source manifest and
pinned data root; the repaired writer did not weaken that boundary.

### R2. P0 closure: observations are construction-owned

**PASS.** The accepted production signature is now exactly:

```python
build_deterministic_compact_archive()
```

It has no parameters, no overload, and no alternate public archive builder.
Calling it with the former `raw_a, raw_b` arguments raises `TypeError`. Its
body always:

1. verifies all eight accepted identities;
2. creates two distinct disposable cache directories;
3. invokes `_capture_raw_in_fresh_child` once for A and once for B;
4. requires independent origin evidence;
5. only then compares raw science and assembles the two compact candidates.

The repaired route therefore cannot receive one caller-owned mapping twice,
two aliases, two deep copies, or two decodings of one caller-owned transport.
Private comparison/serialization helpers return mappings or bytes, not an
accepted `DeterministicCompactArchive`; the zero-argument production builder
is the only archive-returning construction route.

The origin gate was adversarially re-exercised for:

- the same `_FreshRawCapture` object in both positions;
- a copied raw mapping retaining the same origin;
- distinct tokens/caches claiming one child PID;
- distinct tokens/PIDs claiming one cache root.

All four rejected with their specific topology errors. Static call ordering
and the focused test's patched-capture path confirm that these rejections
occur before compact assembly can consume the raw mapping.

On two unrelated top-level builds, each of which owned two inner children,
the returned evidence showed, within every build:

- two different parent-generated origin-token hashes;
- two different positive child PIDs;
- two different cache-root hashes;
- external, nonsymlink, empty-before, empty-after, and disposed cache
  decisions all true.

Across both top-level builds, all four token hashes and all four cache-root
hashes were distinct. The nondeterministic process evidence was excluded from
the scientific archive, while both top-level builds reproduced identical raw
and compact scientific bytes.

### R3. P1 closure: unrelated cache lifecycles

**PASS.** The repaired builder no longer lets its two inner children inherit
one `NUMBA_CACHE_DIR`. It validates the operating-system temporary parent as
external to the textbook repository, pinned Payne Zero checkout, and paper
tree, then creates separate `chapter06-synthesis-capture-a-*` and
`chapter06-synthesis-capture-b-*` directories.

Before either child launch, each directory must be:

- an existing directory;
- not a symlink;
- external to every forbidden source/data root;
- empty;
- distinct after resolution.

The child environment explicitly replaces `NUMBA_CACHE_DIR` with its assigned
root. The child echoes a hash of that resolved root; the parent requires it to
match the assigned root. The accepted scientific worker independently
requires the cache to be empty, external, and nonsymlink before importing
pinned Payne Zero. After each child exits, the parent revalidates that the
same root is still empty and nonsymlink. After both captures, both temporary
directories must be absent before assembly proceeds.

Focused adversarial checks rejected:

- equal A/B cache roots;
- an occupied cache;
- a symlink cache;
- a cache below a forbidden root.

The occupied and symlink probes also asserted that `subprocess.run` was never
called, so those policy failures precede child launch and scientific
execution. The forbidden-root validation is the first operation inside the
child-capture helper and likewise precedes launch. Live A/B builds confirmed
distinct root hashes, empty-before/after evidence, and disposal.

### R4. Pinned execution and fail-closed ordering

**PASS.** The outer builder verifies its accepted worker/assembler snapshot
before creating or invoking either scientific capture. Each child again calls
that identity gate before calling the scientific worker. The worker then
verifies the exact pinned commit, frozen pinned Python source identities,
staged-source parity, source archive, fixture, subset, and static tables.
`PAYNE_ZERO_DATA_ROOT` is either absent and set to the pin or already equal to
the pin; a different configured root is rejected.

The deterministic thread, locale, hash, timezone, no-user-site, and
no-bytecode controls remain the accepted worker's fresh-process precondition.
The repaired writer owns the two cache paths but deliberately does not
silently repair those other controls. In the reviewed route they are supplied
to each fresh top-level process and are revalidated by each inner worker
before pinned synthesis modules are imported. This preserves, rather than
bypasses, the accepted worker boundary.

### R5. Scientific and byte identities

The repaired route independently reproduced the previously reviewed
scientific boundary:

| property | reproduced value |
| --- | --- |
| raw members | `754` |
| raw deterministic transport bytes | `8,689,108` |
| raw transport SHA-256 | `5ff93594781de6fc12a3a2a0adf0ebcc03c81f6141b8648e709ad306686b93a3` |
| complete raw mapping digest | `e2a3dd9670fd9fb5ed34f5131c303687d72ff2d3e762d9224c702f1dffc4a775` |
| raw schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| raw physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| raw full fingerprint | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` |
| compact members | `213` |
| compact array bytes | `1,235,275` |
| compact schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| compact payload fingerprint | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` |
| raw ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| final archive bytes | `1,294,865` |
| final archive SHA-256 | `b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850` |

Raw A/B transports were byte-identical; all raw member dtypes, shapes, and C
bytes were equal; the two compact assemblies had identical schemas, complete
ownership ledgers, member sets, dtypes, shapes, and C bytes; and final archive
A and archive B were byte-identical.

### R6. Archive canonicality and corruption handling

**PASS.** All 213 archive members were independently rechecked as unique and
lexically ordered, with:

- `ZIP_STORED`;
- fixed `1980-01-01 00:00:00` timestamps;
- creator system `3`;
- create/extract version `20`;
- flag bits, volume, and internal attributes zero;
- exact regular-file mode `0600` in external attributes;
- no per-member extra field or comment;
- no archive comment;
- NPY 2.0 payloads only;
- no object dtype and C-contiguous decoded arrays.

The decoder reconstructs every array, compares exact dtype/shape/C bytes
against the independently assembled candidate, reruns the accepted compact
semantic validator, and requires complete byte-for-byte re-encoding.

The repaired focused suite again rejected object dtype, a truncated archive,
member-byte corruption, nonlexical order, changed ZIP timestamp, raw A/B
payload disagreement, and compact schema/ownership/payload disagreements.
The broader unchanged serialized-form logic reviewed in the original audit
continues to reject alternate valid-CRC payloads or shapes, compression,
permissions, NPY versions, comments, and other noncanonical encodings.

### R7. Filesystem and authority boundary

**PASS.** Static inspection and execution found no output/destination
argument, command-line entry point, `open`, NumPy save operation,
`Path.write_*`, rename, replace, golden path, manifest mutation,
authorization parser, publisher, overwrite, or alternate-root publication
surface.

The only filesystem mutation owned by this writer is creation and disposal of
the two external empty cache directories as process-control state. Raw bytes
travel only through child stdout, and the final compact archive exists only
as returned immutable `bytes`. The live two-build test took a complete
size-and-SHA snapshot of every textbook `data/` file before and after and
found exact equality. The pinned checkout commit and every source/data
identity consumed by the accepted worker reproduced. The paper tree is not a
scientific input and is used only as a forbidden cache root.

Every result still records false for publication authorization, golden
publication, manifest mutation, and artifact-file writing.

### R8. Verification executed

Exact artifact and accepted-input hashing:

```text
sha256sum <writer> <tests> <candidate> <prior-audit> <eight-bound-inputs>
all twelve identities matched the values in R1
```

Focused repaired writer suite:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_chapter06_synthesis_compact_writer.py
6 passed in 17.00s
```

Combined accepted worker/assembler/writer chain:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_writer.py
19 passed in 30.44s
```

Targeted topology and corruption tests:

```text
/Users/ysting/anaconda3/bin/python -m pytest -q \
  ...::test_origin_gate_rejects_alias_copy_child_reuse_and_shared_cache \
  ...::test_cache_policy_rejects_occupied_symlink_shared_and_forbidden_roots \
  ...::test_adversarial_archive_and_pair_mutations_fail_closed
3 passed in 4.23s
```

Targeted live determinism and metadata tests:

```text
/Users/ysting/anaconda3/bin/python -m pytest -q \
  ...::test_two_fresh_top_level_builds_reproduce_raw_and_final_bytes \
  ...::test_archive_has_exact_lexical_members_and_fixed_zip_metadata
2 passed in 14.06s
```

Static checks:

```text
/Users/ysting/anaconda3/bin/python -m ruff check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
All checks passed!

/Users/ysting/anaconda3/bin/python -m ruff format --check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
2 files already formatted
```

A repository-wide suite was intentionally not duplicated during this
read-only re-audit because other Chapter 6 agents were concurrently changing
the shared worktree. Acceptance relies on the exact 19-test dependency chain
above, not on the candidate report's earlier repository-wide count.

### R9. Residual scope and final disposition

No residual defect was found in the repaired A/B topology, cache isolation,
pinned-worker boundary, raw/compact equality, canonical archive encoding,
corruption rejection, source immutability, or filesystem authority separation.

The following are deliberate scope limits, not acceptance exceptions:

- the accepted worker's full deterministic environment remains a
  fresh-process precondition that a later publisher must establish;
- origin tokens, PIDs, and cache-root hashes are provenance evidence for the
  fixed reviewed construction, not a security boundary against monkeypatching
  private Python internals;
- the temporary cache directories are disposable control state, not candidate
  artifacts;
- no canonical fixture, golden, manifest record, detached authorization, or
  publisher exists or is authorized by this review.

**ACCEPT** the exact repaired writer
`3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb`
as the deterministic in-memory compact synthesis writer and as an input to
the next separately designed and independently audited gate. This acceptance
does not authorize writing or publishing any archive.
