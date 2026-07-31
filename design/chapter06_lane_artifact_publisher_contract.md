# Chapter 6 two-lane artifact publication contract

Status: **forward-rebound design candidate with both deterministic writers
independently accepted; both lane candidate-byte acceptances remain
UNRESOLVED; NOT AUTHORIZATION**

This contract defines the publication boundary that may be implemented only
after both lane-specific deterministic in-memory archive writers and both
contract-bound candidate-byte gates have passed independent review. It
creates no candidate file, publisher, authorization record, fixture, golden,
or manifest mutation.

This revision responds to the immutable independent rejection
`design/chapter06_lane_artifact_publisher_contract_independent_audit.md`,
SHA-256
`4327c43e7bf58e776dd65a2939409c2b44a923c3b13a1bc01d56dc343c8f50bc`.
It closes the reported indirect hash cycle, exact manifest-byte ambiguity,
nested-directory durability gap, path/trust-object ambiguity, cache-policy
overreach, incomplete `data/` snapshot vocabulary, and crash-left temporary
state. The rejection remains the audit trail for the original design.

Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`.

## Repair and change ledger

| rejected finding | repaired contract |
| --- | --- |
| P0 indirect authorization/manifest/review hash cycle | explicit forward-only `A -> R -> E -> N -> Z` graph and late-bound entry placeholders |
| P1-1 undefined manifest bytes | preserve current unsorted order, append only, fixed object order and exact JSON encoder/delete-last proof |
| P1-2 nested-directory durability | revalidate and `fsync` every new directory and its immediate parent before descent |
| P1-3 path and trust-anchor ambiguity | exact trust-object paths/schema plus separate repository-relative identity and absolute host-access types |
| P1-4 over-rigid atmosphere cache rule | lane-specific post-child policy frozen only by each accepted writer audit |
| P2-1 incomplete `data/` snapshot | separate directory, regular-file, manifest-role, and closed nonmanifest-support inventories |
| P2-2 crash-left temporaries | inert inode-bound quarantine and separately reviewed exact cleanup disposition |
| accepted atmosphere writer boundary | exact writer/test/candidate/audit, nineteen-member archive bytes, external capture-schema semantics, NPY version, and lane-honest cache inventory frozen below |
| stale atmosphere candidate-byte history | historical record `79b20c…` bound contract `9ee002…`; it is non-authoritative after this contract change and both forward byte gates remain unresolved |
| accepted repaired synthesis writer boundary | exact repaired plan/worker/assembler/writer chain, four provenance-only archive changes, and live 213-member archive identity frozen below |

## 1. Decision and role boundary

Chapter 6 has two deliberately asymmetric first publications:

| lane | canonical destination | manifest role | exact purpose |
| --- | --- | --- | --- |
| atmosphere | `data/fixtures/chapter06_atmosphere_one_line_inputs.npz` | `fixture` | input-only, 80-depth, nineteen-member integration fixture for isolating the selected-line atmosphere kernel |
| synthesis | `data/golden/payne_zero/chapter06/synthesis/chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz` | `golden` | 213-member, comparison-only synthesis oracle opened only after the reader-built result exists |

There is no stored Chapter 6 synthesis fixture. The synthesis input remains
the typed composition of the Chapter 5 continuum-state fixture, the Chapter 6
one-row teaching subset, static line-profile tables, and constructed arrays.
Calling the synthesis comparison archive a fixture would mix roles.

The raw Fe I row remains a `subset`; Harris/FASTEX and other tables remain
`static`; neither may be copied into either artifact. The atmosphere fixture
contains no line output. The synthesis golden contains no copied input state
or atmosphere-lane member. One archive may never be used to reconstruct the
other.

The later atmosphere comparison golden
`data/golden/payne_zero/chapter06/chapter06_atmosphere_one_line_cpu.npz`
is explicitly outside this publication phase. It may be captured and
published only after the atmosphere fixture is both installed and registered,
through its own oracle, byte audit, publisher audit, detached authorization,
and no-replace operation.

This is the concrete Chapter 6 application of `BIBLE.md`'s four data roles,
`PLAN.md`'s open two-lane publication gate, and `PASSDOWN.md`'s requirement
that serialization remain unauthorized after in-memory acceptance. It retains
the independently useful Chapter 5 controls—cycle-free detached
authorization, exact-existing no-op, no-replace installation, and
postpublication audit—but does not reuse Chapter 5's paired-golden directory
transaction. Chapter 6 has different roles and a real fixture-before-oracle
dependency, so it uses two one-entry manifest transitions in a fixed order.

## 2. Frozen writers and unresolved downstream gates

### 2.1 Accepted synthesis input

The publication design consumes this exact independently accepted forward
chain:

| accepted synthesis object | SHA-256 |
| --- | --- |
| `design/chapter06_synthesis_fixture_oracle_plan.md` | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| `design/chapter06_synthesis_plan_rebind_candidate.md` | `dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93` |
| `design/chapter06_synthesis_plan_rebind_independent_audit.md` | `9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7` |
| `scripts/chapter06_synthesis_oracle_worker.py` | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| `tests/test_chapter06_synthesis_oracle_worker.py` | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| `design/chapter06_synthesis_worker_independent_audit.md` | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |
| `scripts/chapter06_synthesis_compact_assembler.py` | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| `tests/test_chapter06_synthesis_compact_assembler.py` | `25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153` |
| `design/chapter06_synthesis_compact_rebind_candidate.md` | `54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c` |
| `design/chapter06_synthesis_compact_rebind_independent_audit.md` | `739854db2b5c4c0c0fe5e9db71d8a52958ce401ded7e7a80a8ab90e15172ddcb` |
| `scripts/chapter06_synthesis_compact_writer.py` | `57aa7147afee4a7366cb2a075715d3607fa20507c23c07ec978b0698368ae47b` |
| `tests/test_chapter06_synthesis_compact_writer.py` | `7c41a74f9d2e38a23d988c990af4040ac262a8066cb3cd9feae4e29f0bdc0a4e` |
| `design/chapter06_synthesis_writer_rebind_candidate.md` | `6ab1f346a409b0302550a0923c35b71a84d6b2899f2c356070c8d76aa8145e5a` |
| `design/chapter06_synthesis_writer_rebind_independent_audit.md` | `467fdc810f14302dba80f0dd18ba34239dfedb7579b48280899f6f9b6e3b3653` |

The accepted raw boundary within that chain is:

| raw property | exact value |
| --- | --- |
| members | `754` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |

The accepted in-memory synthesis candidate is:

| property | exact value |
| --- | --- |
| archive bytes | `1,294,865` |
| archive SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| members | `213` |
| compact schema version | `1` |
| compact schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| compact payload fingerprint | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| raw ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |

The internal value
`meta__archive_kind = "synthesis_one_line_comparison_candidate"` describes
the accepted byte-lifecycle snapshot. It does not grant the file a data role.
Only a later detached authorization and manifest entry may give those exact
bytes the repository role `golden`. An independent artifact reviewer must
explicitly accept that distinction; the publisher may not rewrite the member
to make it appear canonical.

The planned synthesis candidate-byte path is already occupied by this
historical rejection:

| historical non-authoritative object | SHA-256 |
| --- | --- |
| `design/chapter06_synthesis_candidate_byte_acceptance.md` | `474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1` |

That record rejected the historical writer against the repaired plan. It
accepts no repaired synthesis bytes and is not an accepted input here. The
synthesis candidate-byte gate remains **UNRESOLVED** until a new independent
re-audit is appended at the same planned trust-object path, preserving the
rejection history and binding the exact independently accepted final bytes of
this contract.

### 2.2 Accepted atmosphere writer and archive

The atmosphere scientific precursor is accepted:

| item | SHA-256 |
| --- | --- |
| atmosphere fixture/oracle plan | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| accepted in-memory fixture worker | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| worker tests | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| final worker audit | `336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e` |
| accepted converter | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| converter audit | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |

Its accepted scientific mapping has nineteen members, schema digest
`f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698`,
and payload fingerprint
`f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663`.

The deterministic serialization gate is now independently accepted:

| accepted writer-side object | SHA-256 |
| --- | --- |
| `scripts/chapter06_atmosphere_fixture_writer.py` | `0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538` |
| `tests/test_chapter06_atmosphere_fixture_writer.py` | `c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a` |
| `design/chapter06_atmosphere_fixture_writer_candidate.md` | `e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee` |
| `design/chapter06_atmosphere_fixture_writer_independent_audit.md` | `b946dcef0beeacf49a3da9ac036e21af7cd7b44d15092639cd3be744fb42f0f9` |

The accepted in-memory atmosphere candidate is:

| property | exact value |
| --- | --- |
| scientific members | `19` |
| scientific array bytes | `357,984` |
| archive bytes | `363,050` |
| archive SHA-256 | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |
| scientific fixture schema digest | `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698` |
| scientific payload fingerprint | `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663` |
| NPY member format | version `2.0` for every member |

The existing atmosphere candidate-byte record is historical evidence only:

| historical non-authoritative object | SHA-256 |
| --- | --- |
| `design/chapter06_atmosphere_fixture_byte_acceptance.md` | `79b20c065f5f1b588c6796d6413a1b2aa5e1e5a60056968fc7913cd961e12fc6` |

That record bound the historical publisher contract at SHA-256
`9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774`
and explicitly invalidates itself when the publisher contract changes. It is
therefore **STALE**, not an accepted upstream lock for this rebound contract.
The atmosphere candidate-byte gate remains **UNRESOLVED** until a new
independent re-audit is appended at the same planned trust-object path
against the exact independently accepted final bytes of this contract.

Versioning is deliberately external and must not be rewritten as an archive
member. The accepted scientific worker's complete transient capture has
`CAPTURE_SCHEMA_VERSION = 1`. The final archive contains exactly the nineteen
scientific fixture arrays and **no embedded archive-schema-version or
capture-schema-version member**. Future authorization and manifest metadata
therefore bind:

```text
fixture_capture_schema_version          1
archive_contains_embedded_schema_version false
npy_member_format_version               "2.0"
scientific_fixture_schema_digest
  f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698
```

They must not invent `ATMOSPHERE_ARCHIVE_SCHEMA_VERSION`,
`meta__archive_schema_version`, or any twentieth member. The capture schema
version describes the accepted worker/evidence envelope; NPY 2.0 describes
the encoding of each of the nineteen final members.

Neither lane's writer-level candidate facts close its candidate-byte gate.
Both lane candidate-byte gates remain unresolved, and no publisher
implementation may begin until both have appended new independent re-audits
against the exact independently accepted final contract.

Any change to either accepted writer, its audit, this contract after
independent acceptance, or a bound upstream identity invalidates both later
candidate-byte re-audits and every publisher, authorization, and publication
record. Any later change to an accepted candidate-byte record invalidates all
publisher, authorization, and publication records that bind it.

## 3. Trust objects and authority separation

For each lane, keep these seven objects distinct:

1. **Candidate bytes.** Returned in memory by the zero-argument accepted
   writer; they have no canonical path or role.
2. **Independent byte acceptance.** A read-only review freezes exact bytes,
   size, member set, archive metadata, schema, science, ownership, and source
   snapshot. It is not authorization.
3. **Publisher implementation acceptance.** A separate review accepts only
   the fixed-path, no-replace, fail-closed publisher and its tests.
4. **Detached publication authorization.** A strict JSON record binds one
   accepted byte string to one publisher and one prepublication manifest.
5. **Canonical file installation.** A no-replace operation makes the exact
   file name visible; it does not by itself make the file readable by the
   textbook.
6. **Canonical registration.** A narrowly scoped atomic manifest update
   assigns the file its one allowed data role. Readers reject an unregistered
   file.
7. **Postpublication audit.** A later read-only review freezes the realized
   authorization, record review, artifact, and postpublication manifest
   hashes. It cannot feed a hash backward into any earlier object.

No one object implies any later object. In particular, a valid-looking JSON
record that has not passed independent record review is not authority.

### 3.1 Exact planned trust-object paths

These POSIX repository-relative literals are the complete planned control
surface:

| object | atmosphere fixture lane | synthesis comparison lane |
| --- | --- | --- |
| deterministic writer | `scripts/chapter06_atmosphere_fixture_writer.py` | `scripts/chapter06_synthesis_compact_writer.py` |
| writer tests | `tests/test_chapter06_atmosphere_fixture_writer.py` | `tests/test_chapter06_synthesis_compact_writer.py` |
| writer candidate record | `design/chapter06_atmosphere_fixture_writer_candidate.md` | `design/chapter06_synthesis_writer_rebind_candidate.md` |
| writer acceptance | `design/chapter06_atmosphere_fixture_writer_independent_audit.md` | `design/chapter06_synthesis_writer_rebind_independent_audit.md` |
| candidate-byte acceptance | `design/chapter06_atmosphere_fixture_byte_acceptance.md` | `design/chapter06_synthesis_candidate_byte_acceptance.md` |
| publisher | `scripts/build_chapter06_atmosphere_fixture.py` | `scripts/build_chapter06_synthesis_golden.py` |
| publisher tests | `tests/test_chapter06_atmosphere_fixture_publisher.py` | `tests/test_chapter06_synthesis_golden_publisher.py` |
| publisher implementation acceptance | `design/chapter06_atmosphere_fixture_publisher_independent_audit.md` | `design/chapter06_synthesis_golden_publisher_independent_audit.md` |
| detached authorization | `design/chapter06_atmosphere_fixture_publication_acceptance.json` | `design/chapter06_synthesis_publication_acceptance.json` |
| authorization-record review | `design/chapter06_atmosphere_fixture_publication_record_review.json` | `design/chapter06_synthesis_publication_record_review.json` |
| postpublication audit | `design/chapter06_atmosphere_fixture_postpublication_audit.md` | `design/chapter06_synthesis_postpublication_audit.md` |

The deterministic-writer, writer-test, writer-candidate, and
writer-acceptance rows are accepted for both lanes at the exact identities in
Sections 2.1 and 2.2. Both candidate-byte rows remain **UNRESOLVED**. Their
current files are the stale historical atmosphere acceptance `79b20c…` and
the historical synthesis rejection `474e318…`; neither is authoritative for
this contract. After this final contract is independently accepted, each lane
must append a new independent re-audit at its existing candidate-byte path
that binds the exact final contract SHA-256. This contract deliberately
contains neither future candidate-byte acceptance hash.

Every publisher, publisher-acceptance, detached-authorization, record-review,
and postpublication path/hash pair remains **UNRESOLVED** until that exact
object exists and passes its named independent gate. The current contract is
the shared publisher contract at
`design/chapter06_lane_artifact_publisher_contract.md`; each later publisher,
publisher acceptance, authorization, and authorization review binds its
independently reviewed final SHA-256 together with the final applicable
candidate-byte acceptance SHA-256.

The authorization-record review is strict duplicate-free UTF-8 JSON, not
free-form prose. Its exact schema is:

```text
schema_version                    integer 1
record_kind                       exact lane-specific review kind
authorization_path                exact allowlisted repository-relative path
authorization_sha256              lowercase 64-hex
candidate_byte_acceptance_path    exact allowlisted repository-relative path
candidate_byte_acceptance_sha256  lowercase 64-hex
publisher_acceptance_path         exact allowlisted repository-relative path
publisher_acceptance_sha256       lowercase 64-hex
manifest_entry_template_sha256    lowercase 64-hex
disposition                       exact string "ACCEPT"
```

Keys occur in exactly that order. Unknown, missing, duplicate, reordered, or
wrong-typed fields fail. The review contains the authorization hash; the
authorization does not contain the review hash.

The exact `record_kind` values are
`chapter06_atmosphere_fixture_publication_record_review` and
`chapter06_synthesis_publication_record_review`, respectively.

## 4. Required order

No lane may publish until both deterministic writer acceptances are final.
That writer-only common gate is now closed at the exact identities in
Sections 2.1 and 2.2. Both candidate-byte gates are unresolved. This forward
contract rebind first requires its own independent acceptance and grants no
downstream authority. Publisher implementation may begin only after both
later byte re-audits accept their exact candidates against that final
contract. The still-required sequence is:

1. Independently audit and accept or reject the exact final bytes of this
   rebound contract. The final contract contains neither future
   candidate-byte acceptance hash.
2. Rebuild the atmosphere candidate in two unrelated top-level runs under its
   accepted writer topology. Independently repeat every byte, schema, science,
   role, cache, adversarial, and immutability check and append a new re-audit
   at `design/chapter06_atmosphere_fixture_byte_acceptance.md` that binds the
   exact accepted final contract while preserving the stale `79b20c…`
   history.
3. Rebuild the synthesis candidate in two unrelated top-level runs. Each run
   must retain the repaired writer's internally owned A/B fresh-child,
   distinct-cache, source-identity, semantic, and byte-equality checks.
   Independently repeat every byte, schema, science, role, ownership,
   encoding, adversarial, and immutability check and append a new re-audit at
   `design/chapter06_synthesis_candidate_byte_acceptance.md` that binds the
   exact accepted final contract while preserving the `474e318…` rejection.
4. Only after both appended candidate-byte re-audits are independently final,
   implement the two fixed-destination publishers and their dry-run modes.
   Publication remains disabled because no detached record exists.
5. Independently audit publisher identity, path confinement, no-replace
   primitive, manifest delta, cleanup, fsync, race, and failure behavior.
   Each publisher and publisher acceptance binds the final contract and its
   lane's final candidate-byte acceptance.
6. Create and independently review the atmosphere-fixture authorization
   against manifest snapshot \(M_0\), binding the final contract and final
   atmosphere candidate-byte acceptance.
7. Run the authorized atmosphere dry-run twice; require the same deterministic
   report and candidate identity.
8. Install the atmosphere fixture without replacement, register only its
   fixture entry, and independently audit resulting manifest \(M_1\).
9. Only then create and review the synthesis authorization against exact
   manifest \(M_1\), binding the final contract and final synthesis
   candidate-byte acceptance.
10. Run the authorized synthesis dry-run twice, install the synthesis golden
   without replacement, register only its golden entry, and independently
   audit resulting manifest \(M_2\).
11. Only after \(M_2\) may the separate atmosphere oracle/golden lifecycle
    begin.

This fixed order prevents two simultaneously prepared authorization records
from claiming the same prepublication manifest. Reversing the order requires
a reviewed contract revision; a publisher may not improvise.

### 4.1 Acyclic hash dependency

For one lane, let:

```text
D = independently accepted final publisher contract
C = accepted candidate-byte review
P = accepted publisher implementation review
M = exact prepublication manifest bytes
T = intended manifest-entry template with two literal late-bound placeholders
A = detached authorization bytes
R = authorization-record-review bytes
E = realized manifest entry
N = realized postpublication manifest bytes
Z = postpublication audit
```

The only allowed dependency graph is:

```text
D ───────────────────> C
accepted writers/sources ──> C

(publisher implementation, D, C) ──> P
(D, C, P, M, T) ────────────> A

A ──> R
(T, SHA256(A), SHA256(R)) ──> E
(M, E) ──> N
(A, R, E, N, artifact bytes) ──> Z
```

`D` contains the two candidate-byte trust-object paths and their required
review rules, but neither future `SHA256(C)`. Each final `C` binds `D`; only
the later `P`, `A`, and `R` objects bind both final hashes. Thus the contract
can be finalized before either byte re-audit without a `D <-> C` cycle.

Neither `A` nor `R` contains `SHA256(E)`, `SHA256(N)`, or `SHA256(Z)`.
`A` contains the exact placeholder-bearing `T` and its digest. `R` contains
`SHA256(A)` and the same template digest, but not its own hash. Only after
both are final may the publisher substitute their hashes into `T`, yielding
`E` and then `N`. The postpublication audit alone freezes the realized entry
and manifest hashes. No realized hash is copied back into an earlier object.

The two exact placeholder string values are:

```text
__LATE_BOUND_AUTHORIZATION_SHA256__
__LATE_BOUND_RECORD_REVIEW_SHA256__
```

They may occur exactly once each, only as the complete values of
`publication_acceptance_sha256` and
`publication_record_review_sha256`. Substitution changes values only; it may
not add, delete, reorder, or reinterpret a field.

## 5. Detached authorization schema

The planned canonical records remain:

```text
design/chapter06_atmosphere_fixture_publication_acceptance.json
design/chapter06_synthesis_publication_acceptance.json
```

Each duplicate-free UTF-8 JSON record must bind, by exact allowlisted POSIX
repository-relative identity and lowercase 64-character SHA-256:

- record schema version, record kind, lane, and exact manifest role;
- pinned Payne Zero commit;
- exact writer, writer tests, writer candidate record, and final writer
  acceptance;
- exact candidate-byte acceptance record;
- exact publisher, publisher tests, publisher contract, and publisher
  implementation acceptance;
- exact artifact path, filename, byte size, archive SHA-256, member count,
  archive kind, lane-applicable external or embedded schema contract, NPY
  member version, schema digest, and payload/ownership fingerprints;
- exact prepublication `data/MANIFEST.json` path, SHA-256, schema version,
  ordered entry-path digest, and proof that the destination entry is absent;
- the exact ordered, placeholder-bearing manifest-entry template `T` and its
  digest;
- exact accepted source/data snapshot digest and every lane-specific upstream
  file identity;
- the destination-parent policy and no-replace primitive accepted for the
  host platform.

The atmosphere record must use role `fixture`; bind all nineteen members,
capture-schema version `1` as external provenance, no embedded archive schema
member, NPY member version `2.0`, and the exact identities in Section 2.2.
The synthesis record must use role `golden`, comparison-only use, all 213
members, and the exact synthesis identities in Section 2.1.

The authorization contains no self-hash, review hash, realized-entry hash, or
postpublication-manifest hash. The independent JSON review named in Section
3.1 freezes the authorization SHA-256. Once both files are final, the
publisher computes their hashes and realizes the two template placeholders.
The subsequent manifest entry records those now-known hashes. Its exact bytes
and the postpublication manifest SHA-256 are frozen only in the
postpublication audit. Any authorization or review edit invalidates the
realized entry and requires a new record review before publication.

The strict parser rejects duplicate or unknown keys, missing fields, wrong
types, nonfinite JSON values, malformed hashes, wrong roles, stale manifest
identities, unresolved locks, and extra destinations.

Identity paths inside authorization/review/manifest JSON are exact
allowlisted POSIX repository-relative literals from Sections 1 and 3.1. They
must be nonempty NFC ASCII subsets using `/`, contain no leading `/`, `.`,
`..`, empty component, backslash, alternate separator, percent escape, or
Unicode/case alias. They are identifiers, not caller-selected filesystem
locations. On-disk access always starts from the fixed canonical host root
`/Users/ysting/stellar-spectroscopy-from-scratch-gpu` and walks the literal
components with directory-relative no-follow operations. The resulting
absolute host path is an access path and is never substituted into a manifest
identity field. Caller-supplied relative or absolute paths, roots, and
destinations are rejected because the publisher exposes none. External pinned
Payne Zero paths are separately typed absolute source identities and may not
appear in repository-manifest `path` fields.

There is no force, replace, repair, merge, destination, alternate-root, or
partial-lane option.

## 6. Source and data immutability snapshots

Before candidate construction and again immediately before canonical writes,
the publisher records:

- the exact repository root and this contract's reviewed SHA-256;
- the pinned Payne Zero commit, clean identity of every declared loaded source,
  and each lane's complete dynamic data-read manifest;
- all bound local scripts, tests, design acceptances, static files, subset
  files, and upstream fixture files by regular-file/no-symlink state, size,
  and SHA-256;
- `data/MANIFEST.json` bytes, schema, ordered entry paths, device, and inode;
- the forbidden Payne Zero and paper roots as read-only/no-write boundaries.

The `data/` snapshot is four separate closed inventories:

1. **Directories:** every recursive directory component by repository-relative
   path, type, nonsymlink decision, device, inode, owner, group, and mode.
2. **Regular files:** every recursive regular file by path, nonsymlink
   decision, device, inode, owner, group, mode, byte size, and SHA-256.
   Symlinks, sockets, devices, and other special nodes are forbidden.
3. **Manifest-backed files:** an exact join between inventory (2) and every
   manifest entry, including role, entry position, entry digest, bytes, and
   file hash.
4. **Nonmanifest support:** the closed normal-state allowlist contains only
   `data/README.md`, currently SHA-256
   `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b`.
   An exact unregistered target is permitted only in the explicit
   recovery state; crash-left staging names are quarantine states under
   Section 11, not silently added support files.

After each dry run, candidate build, failed publication, and successful
publication, repeat the snapshot. A dry run allows no repository delta. A
successful atmosphere phase allows only its exact new artifact and one-entry
manifest delta. A successful synthesis phase additionally allows creation of
only the two exact directory components named in Section 8, with their
accepted directory metadata and durability evidence.

Cache policy is lane-specific and is inherited from the accepted deterministic
writer rather than invented here:

- both lanes begin with two distinct, existing, empty, external, nonsymlink
  roots and record pre/post path, device, inode, type, and inventory evidence;
- the accepted synthesis writer additionally requires each root to remain
  empty after its child exits;
- the accepted atmosphere writer requires each initially empty child root to
  contain after capture exactly one nonsymlink package directory, eighteen
  nonsymlink `.nbi` Numba index files, and eighteen nonsymlink `.nbc` Numba
  compiled files: 37 entries total, zero symlinks, and zero non-Numba files
  for each of A and B;
- every writer-owned cache root is disposed before that accepted writer
  returns, whether or not the child populated it;
- cache paths, contents, PIDs, and random origins never enter scientific
  candidate bytes.

Unexpected source or data change is a hard stop. It is never incorporated by
refreshing a hash during the same invocation.

## 7. Reproducible dry-run contract

Each publisher exposes a verification-only mode that:

- resolves only the canonical repository and destination;
- verifies all accepted identities and, in authorized mode, the exact
  detached record plus the exact named record-review schema, its binding to
  the authorization hash, and its computed review hash;
- invokes the accepted zero-argument writer twice at top level;
- requires identical complete bytes and accepted summaries;
- decodes the result as untrusted ZIP/NPY, revalidates every semantic and
  ownership rule, and canonically re-encodes it byte for byte;
- verifies destination and manifest preconditions without creating a parent,
  temporary file, lock file, candidate file, or manifest file;
- emits deterministic JSON containing identities and decisions but excluding
  timestamps, PIDs, hostnames, cache paths, and random tokens.

Two independent authorized dry runs must produce byte-identical deterministic
reports. Nondeterministic child-origin evidence remains in ephemeral logs and
is reduced only to required true/false topology decisions in that report.
Verification-only code has no reachable call to the publication primitive.

## 8. Fixed destinations and path safety

The only artifact destinations are the two paths in Section 1. Temporary
staging names may exist only inside the destination's canonical filesystem
and only after authorization. The only publisher-created directories allowed
are the missing literal components
`data/golden/payne_zero/chapter06` and
`data/golden/payne_zero/chapter06/synthesis`.

Starting at the fixed repository root, every existing path component is
checked with no-follow semantics. The repository root, `data`, role directory,
destination parent, manifest, authorization record, reviews, candidate
dependencies, and any existing target must be nonsymlink. Parents must be
directories; files must be regular; resolved and lexical paths must agree.
A relocated publisher, alternate checkout, mount escape, `..`, hard-linked
staging input, or case/Unicode alias is rejected.

Missing allowlisted directories are created one component at a time only
after authorization. For each component, the publisher:

1. retains and revalidates the immediate parent descriptor;
2. creates the one literal name exclusively;
3. opens the new directory with directory/no-follow semantics and records its
   device, inode, owner, group, and mode;
4. `fsync`s the new directory;
5. `fsync`s its immediate parent so the new name is durable;
6. reopens the name without following links and requires the same
   device/inode and accepted metadata before descending into it.

`chapter06` must complete all six steps before `synthesis` is attempted.
Failure of either child or parent `fsync` aborts before artifact staging.
Empty directories created by a handled failed attempt may be removed only if
their device/inode still matches the invocation record; each removal is
followed by `fsync` of its immediate parent. No recursive cleanup is
permitted. A crash-left directory is handled as an incomplete state under
Section 11 rather than inferred to be invocation-owned.

The target state is exactly one of:

- absent;
- a regular, nonsymlink, single-link file whose complete bytes equal the
  authorized artifact.

The second case is an idempotent recovery/no-op, never an overwrite. Any
other existing target is a hard failure.

The strict lifecycle recognizes only three complete states: authorization-
bound prepublication manifest with absent target, the same manifest with an
exact unregistered target awaiting registration, or the uniquely recomputed
postpublication manifest `N` from `(M, T, SHA256(A), SHA256(R))` with its
exact registered target. The third state is a validation-only no-op. No stale
record is allowed to bless a different manifest merely because the target
bytes happen to match. Orphan temporary names and crash-left new directories
are explicit incomplete states, not a fourth complete state.

## 9. Atomic no-replace file installation

After all gates pass:

1. Hold the reviewed exclusive repository-data publication lock for the
   entire file-plus-manifest phase. The lock attaches to the stable canonical
   `data` directory, not to a manifest inode that will be replaced. If this
   platform cannot provide the independently tested lock, fail before writes.
2. Recheck every retained allowlisted ancestor and destination-parent
   device/inode, directory-durability decisions, target state, source
   snapshot, candidate bytes, authorization/review bytes, template digest,
   and manifest \(M_i\).
3. Open a hidden staging file in the destination parent with
   create-exclusive and no-follow semantics, mode `0600`. Never stage through
   a caller path.
4. Write all bytes, reject short writes, `fsync` the file, read it back through
   the open descriptor, and rerun whole-byte and semantic validation.
5. Install it with a reviewed atomic create-if-absent primitive on the same
   filesystem. A hard-link-from-staging operation is acceptable where tested:
   link creation is the authoritative no-replace decision, and `EEXIST`
   triggers exact-existing validation rather than replacement. A tested
   `RENAME_NOREPLACE` equivalent is also acceptable. Plain `rename`,
   `replace`, pre-delete, and check-then-overwrite are forbidden.
6. `fsync` the destination parent, then remove only the invocation-owned
   staging name after verifying its inode and `fsync` the parent again.
7. Open the final path with no-follow semantics; require the expected
   device/inode, one link, size, SHA-256, and complete archive validation.

The no-replace syscall, not an earlier absence check, decides races. A target
that appears between checks is never overwritten.

## 10. Narrow manifest registration

The manifest update is a separate durability step under the same data lock:

1. Read exact \(M_i\) bytes through a retained no-follow descriptor. Parse
   with an order-preserving duplicate-key-rejecting JSON decoder; require
   top-level key order exactly
   `schema_version`, `payne_zero_commit`, `entries`; verify the
   authorization-bound SHA-256, schema version, commit, unique entry paths,
   exact existing entry sequence, and destination absence.
2. Require exact round-trip equality under this byte encoder:

   ```text
   UTF-8(
     json.dumps(
       object,
       indent=2,
       ensure_ascii=False,
       allow_nan=False,
       sort_keys=False,
       separators=(",", ": "),
     )
     + "\n"
   )
   ```

   The current manifest is intentionally **not path-sorted**. Its existing
   entry sequence and every existing object-key sequence are preserved
   exactly; they are never called canonical, sorted, or normalized.
   The current 37-entry manifest at SHA-256
   `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a`
   has been checked to round-trip under this encoder; each future
   authorization repeats the check against its own exact prepublication
   bytes.
3. The authorization embeds `T` as an order-preserving JSON object. Its
   template digest is:

   ```text
   SHA256(
     UTF-8(
       json.dumps(
         T,
         ensure_ascii=False,
         allow_nan=False,
         sort_keys=False,
         separators=(",", ":"),
       )
     )
   )
   ```

   The template's literal object-key order is normative and is frozen by both
   candidate-byte acceptance and authorization review. The `arrays` object
   uses lexical archive-member order. Every array record uses exact key order
   `shape`, `dtype`, `unit`, `sha256`, `axes`, `ownership`. No parser or
   serializer may sort or reconstruct these objects from an unordered map.
4. Realize `E` by an ordered deep copy of `T`, replacing only the two complete
   placeholder values from Section 4.1 with `SHA256(A)` and `SHA256(R)`.
   Require a structural diff of exactly two scalar value replacements and
   recompute the template digest before substitution.
5. `E` records role, path, bytes, artifact SHA-256, builder/publisher, writer
   and all acceptance hashes, the now-known authorization and record-review
   hashes, upstream identities, archive/schema/payload identities,
   reproducibility environment, and every member's shape, dtype, axes,
   unit/convention, ownership, and C-byte SHA-256.
6. Construct \(N=M_{i+1}\) by preserving the complete parsed \(M_i\) tree and
   appending `E` as the final `entries` element. **Append is the sole insertion
   rule.** Serialize with the encoder in step 2. Removing that last entry from
   parsed `N` and reserializing must reproduce the exact original \(M_i\)
   bytes and SHA-256. Parsed deep comparison must independently show that all
   pre-existing values, key orders, and entry positions are unchanged.
7. Validate `N` against the canonical filesystem and repository role tests.
   The atmosphere entry is `fixture`; the synthesis entry is `golden`. No
   entry may claim `static`, `subset`, both roles, or another lane's
   ownership.
8. Write exact `N` bytes to a create-exclusive, nonsymlink temporary file
   beside `data/MANIFEST.json`; reject short writes, `fsync`, descriptor-read
   back, reparse, and compare exact intended bytes.
9. Immediately before the one allowed atomic replacement of the manifest,
   recheck the locked manifest device/inode/SHA, canonical artifact bytes,
   `A`, `R`, `T`, and recomputed `N`. Atomically replace only
   `data/MANIFEST.json`, then `fsync` `data`.
10. Reopen both artifact and manifest without following symlinks and run the
    complete manifest and lane-specific validation in a fresh process. The
    postpublication audit records the realized entry digest, `SHA256(N)`,
    authorization hash, review hash, and artifact hash; none is fed backward
    into `A` or `R`.

This is narrowly scoped replacement of the single manifest file, not
permission to edit an existing entry or artifact. A manifest entry is never
installed before its exact artifact.

## 11. Failure, recovery, and partial publication

- Before the no-replace install: remove only an invocation-owned staging file;
  canonical state remains unchanged.
- After artifact install but before manifest replacement: do not delete or
  rewrite the artifact. It is an inert, unregistered exact candidate. Resume
  only after rechecking the same authorization and unchanged \(M_i\); if
  \(M_i\) changed, issue and review a new authorization.
- After manifest replacement: never roll back by deleting the artifact or
  restoring an old manifest. Preserve evidence, fail closed, and require an
  independent audit or a new narrowly reviewed forward repair.
- An empty allowlisted directory left by a crash carries no data role. A
  nonempty unexpected directory, nonexact target, duplicate manifest entry,
  registered missing file, or entry/file mismatch is a hard stop.
- A later invocation may adopt an exact allowlisted crash-left directory only
  after proving its permitted contents, nonsymlink type, owner/group/mode,
  device/inode stability, and then `fsync`ing both the directory and immediate
  parent. It does not infer prior invocation ownership or delete the
  directory. Unexpected contents remain a hard stop.
- A crash-left hidden artifact stage or manifest temporary is an inert
  **quarantine object**. The next publisher invocation inventories its exact
  repository-relative name, parent device/inode, object device/inode, owner,
  group, mode, link count, size, and SHA-256, then hard-stops. A filename
  prefix, apparent NPZ/JSON validity, or equality with candidate bytes never
  proves ownership; the object is never decoded as an input, promoted,
  renamed to a target, or used as manifest bytes.
- Quarantine cleanup is outside both publishers. It requires a separately
  reviewed, duplicate-free JSON record at
  `design/chapter06_publication_quarantine_cleanup_acceptance.json` binding
  each exact name and all inventory fields above, the classification
  `artifact_stage` or `manifest_temporary`, and disposition `unlink_only`.
  Cleanup revalidates every field with no-follow access, unlinks only that
  inode, and `fsync`s the immediate parent. Any mismatch rejects all cleanup;
  no glob, prefix deletion, directory recursion, target deletion, or manifest
  replacement is allowed. This contract neither creates nor authorizes such a
  record.
- Failure to `fsync`, close, read back, release the data lock, or clean the
  exact staging name is reported as failure. Success is not claimed from
  process exit alone.

Cross-lane atomicity is neither possible nor claimed. At every intermediate
state, readers use only files with exact valid manifest entries. The fixed
phase order and post-phase audit turn a crash into either unchanged state or
an inert exact file, not a mixed-role readable artifact.

## 12. TOCTOU analysis

Read-only path checks alone are insufficient: a parent, target, manifest, or
authorization file can change after validation. The implementation must
therefore combine:

- directory-relative, no-follow opens;
- retained directory descriptors and device/inode comparison;
- a stable data-directory publication lock;
- exact source, manifest, authorization, and candidate hashes rechecked
  immediately before each mutation;
- an atomic no-replace artifact syscall;
- an atomic manifest replacement only after full intended-delta validation;
- fresh no-follow readback after both operations.

These controls close cooperative publisher races and ordinary filesystem
substitution. They do not pretend that hashes are a security signature or
that repository code can defeat a hostile process with unrestricted access
to the same account. Such external mutation is detected when possible and
causes failure; it is not part of the scientific reproducibility claim.

## 13. Required adversarial matrix

| family | required mutation | required result |
| --- | --- | --- |
| common writer gate | change either accepted writer/test/candidate/audit identity from Sections 2.1–2.2 | both publishers reject before any repository write |
| identity | change one writer, test, audit, contract, source, subset, fixture, table, or commit byte | reject before candidate construction or staging |
| determinism | same capture supplied twice, reused origin/cache/PID, or A/B byte disagreement | reject; no candidate accepted |
| archive | truncate, corrupt, add/remove/rename a member, invent an atmosphere schema-version member, alter dtype/shape, use object dtype, alternate valid ZIP metadata, compression, order, permissions, comment, trailing bytes, or noncanonical/wrong-version NPY | reject whole-byte or semantic validation |
| role | add atmosphere output to the fixture; copy synthesis input state; label fixture as golden; label synthesis golden as fixture; mix static/subset members | reject candidate or manifest delta |
| hash graph | put realized-entry, postmanifest, review-self, or postpublication-audit hash into authorization/review | schema rejects the cycle-forming field before publication |
| authorization | missing/stale record, duplicate/unknown/reordered JSON key, malformed hash, wrong role/path/size/schema, mutated named byte/publisher/review object, or changed pre-manifest | reject before parent or staging creation |
| path identity | replace an allowlisted POSIX repository-relative identity with an absolute path, `.`, `..`, backslash, Unicode/case alias, or caller root | reject identity; never reinterpret it as an access path |
| path access | relocate publisher, use alternate root/destination, symlink any component or target, use a nonregular target, cross-device stage, or target with multiple hard links | reject without modifying canonical bytes |
| target race | create target after absence check but before install | no-replace syscall loses safely; exact file becomes no-op, nonexact file is failure |
| parent race | swap/recreate destination parent during staging | device/inode recheck rejects; no final install |
| directory durability | fail `fsync` of newly created `chapter06`, its `payne_zero` parent, newly created `synthesis`, or its `chapter06` parent | abort before staging; never report durable directory creation |
| manifest race | mutate or replace manifest after authorization or while candidate is staged | pre-replace identity check rejects; installed file, if any, remains inert |
| manifest order | sort existing entries, insert rather than append, reorder an existing/new object key, change indentation/separators/escaping/newline, or fail delete-last reconstruction | exact byte-policy validation rejects |
| concurrent lanes | start both phases against \(M_0\) | stable data lock plus phase/manifest binding permits only atmosphere; synthesis record becomes stale |
| I/O | short write, readback mismatch, `fsync` failure, no-replace failure, manifest-temp failure, or manifest-replace failure | never report success; apply Section 11 recovery state |
| crash point | terminate before stage, after stage, after file install, or after manifest replace | respectively unchanged, cleaned/inert stage, inert exact file, or forward-audit state; never overwrite |
| quarantine | leave an artifact stage or manifest temporary, spoof its prefix/content, or change its inode before reviewed cleanup | next publisher hard-stops; object is never promoted; cleanup rejects any inode-bound mismatch |
| idempotence | rerun with exact registered artifact and current postpublication manifest | validation-only no-op; no file or manifest rewrite |
| partial state | exact unregistered artifact; nonexact unregistered artifact; registered missing artifact | recover only the first under an exact current authorization; reject the others |
| cache | start from a populated/shared/symlink root, violate synthesis empty-after, or change atmosphere's per-child `1 directory + 18 .nbi + 18 .nbc`, nonsymlink-only inventory/disposal | reject under the lane writer's accepted cache contract; dispose owned roots |
| immutability | modify any non-target `data/` file, directory metadata, `data/README.md`, manifest role join, or declared source during dry-run/publication | snapshot mismatch and failure; never refresh in place |

Tests must also prove that every negative case leaves all pre-existing
canonical file bytes unchanged.

## 14. Acceptance checklist

### Before either publication

- [x] Both exact zero-argument deterministic writers and tests are accepted.
- [x] Both final writer-audit hashes replace all writer/archive locks.
- [x] Two unrelated top-level builds per lane reproduce exact candidate bytes.
- [ ] The exact final rebound contract is independently accepted without
      embedding either future candidate-byte acceptance hash.
- [ ] A new atmosphere byte re-audit at the existing path binds that exact
      final contract and accepts schema, science, role, encoding, size, and
      hash while preserving stale `79b20c…` history.
- [ ] A new synthesis byte re-audit at the existing path binds that exact
      final contract and accepts schema, science, role, ownership, encoding,
      size, and repaired archive hash while preserving rejection `474e318…`.
- [ ] Both publisher implementations and adversarial suites are independently
      accepted only after both candidate-byte gates close.
- [ ] Source/data snapshots are complete and unchanged.
- [ ] Every trust-object path is the exact Section 3.1 literal and every
      required future hash is resolved by its independent gate.
- [ ] Independent review proves the Section 4.1 dependency graph has no edge
      from a realized manifest/postpublication object back into authorization.
- [x] Each lane-specific cache postcondition exactly matches its accepted
      writer audit; atmosphere's reviewed populated-cache inventory is not
      inferred from synthesis's empty-after result.

### Atmosphere fixture phase

- [ ] Candidate-byte re-acceptance freezes the exact nineteen-member,
      363,050-byte archive at SHA-256 `1b727671…`.
- [ ] Authorization binds exact \(M_0\), role `fixture`, nineteen members, and
      the one allowlisted destination.
- [ ] It records external capture-schema version `1`, no embedded archive
      schema member, and NPY member version `2.0` without creating a twentieth
      member.
- [ ] Authorization record and record hash are independently accepted.
- [ ] Placeholder template `T`, append-only insertion, ordered JSON encoding,
      and delete-last exact reconstruction pass.
- [ ] Two authorized dry runs are identical and write nothing.
- [ ] Atomic no-replace install and one-entry manifest transition
      \(M_0 \rightarrow M_1\) succeed.
- [ ] Fresh postpublication audit accepts file, member metadata, provenance,
      role, and exhaustive manifest.

### Synthesis comparison phase

- [ ] Candidate-byte re-acceptance freezes the repaired 213-member,
      1,294,865-byte archive at SHA-256 `a4b1ffa4…`.
- [ ] Authorization is created only after accepted \(M_1\).
- [ ] It binds the exact 213-member, 1,294,865-byte synthesis candidate and
      role `golden`.
- [ ] Two authorized dry runs are identical and write nothing.
- [ ] Each newly created nested directory and its immediate parent have
      successful identity revalidation and `fsync` evidence.
- [ ] Atomic no-replace install and one-entry manifest transition
      \(M_1 \rightarrow M_2\) succeed.
- [ ] Fresh postpublication audit accepts comparison-only use and proves no
      atmosphere/input-state ownership leaked into the archive.
- [ ] No crash-left stage, manifest temporary, unreviewed support file, or
      unresolved quarantine object remains.

### Before atmosphere golden work

- [ ] Both \(M_1\) and \(M_2\) transitions have immutable acceptance records.
- [ ] The notebook loader rejects any unregistered or wrong-role artifact.
- [ ] A new atmosphere oracle/golden contract binds the exact registered
      fixture; this contract is not reused as its authorization.

## 15. Internal contradiction pass

The repaired design was reread as one lifecycle with these results:

- The independently accepted final contract contains neither future
  candidate-byte acceptance hash; both later byte re-audits bind that
  contract, and only still-later publisher/authorization objects bind both.
- `A -> R -> E -> N -> Z` is strictly forward; neither `A` nor `R` binds a
  later or self-dependent hash.
- The authorization binds exact prepublication bytes and placeholder template,
  while only `Z` freezes realized postpublication bytes.
- The live unsorted manifest is preserved byte-for-byte and extended only by
  one final entry under one explicit JSON encoder.
- Repository-relative identity strings and canonical absolute host access
  paths have different schemas and cannot substitute for each other.
- The normal `data/` inventory, exact unregistered-target recovery, newly
  created directory delta, and quarantine states are disjoint.
- Atmosphere inherits the accepted 37-entry populated-cache inventory and
  synthesis inherits its accepted empty-after rule; neither lane borrows the
  other's cache postcondition.
- Atmosphere's version `1` labels the external transient capture contract,
  while NPY `2.0` labels member encoding; the nineteen-member archive gains no
  invented embedded schema field.
- Exact-existing no-op recomputes `E` and `N` from `M`, `T`, `A`, and `R`; it
  does not rely on a cycle-forming postmanifest hash in authorization.
- The stale atmosphere candidate-byte record and historical synthesis
  rejection are non-authoritative history. Both future candidate-byte
  re-audits remain read-only trust objects; neither implies a publisher,
  authorization, cleanup record, canonical fixture, golden, or manifest
  mutation.

No remaining internal dependency requires bytes to contain their own hash or
the hash of an object that depends on them.

## 16. Disposition

**NOT AUTHORIZATION.** Both repaired writer identities and in-memory
candidate facts are accepted inputs to two still-required candidate-byte
re-audits, not permission to write them. The final contract must be
independently accepted first; both byte gates must then accept that exact
contract before any publisher implementation may begin. No publisher,
detached record, canonical fixture, golden, or manifest edit may be created
or executed on the authority of this document.
