# Chapter 6 two-lane publisher-contract independent audit

Status: independent read-only design review  
Audited: 2026-07-30  
Disposition: **REJECT as a publisher-implementation prerequisite until the
indirect authorization/manifest hash cycle and the P1 contract gaps below are
closed**

The contract gets the scientific role split, two-lane order, no-replace file
installation, partial-state principle, and non-authorization boundary mostly
right. The blocking defect is at the detached authorization boundary. The
authorization is required to contain the exact postpublication manifest hash,
while the postpublication manifest is required to contain the hash of that
same authorization and of its later review. This is an indirect self-reference
despite the statement that the authorization contains no self-hash.

This review created only this report. It did not edit the contract, a writer,
publisher, test, source file, data file, manifest, acceptance record, fixture,
golden, or either read-only external source tree.

## 1. Reviewed snapshot

The assigned contract matched the expected identity before and after review:

| reviewed object | SHA-256 |
| --- | --- |
| `design/chapter06_lane_artifact_publisher_contract.md` | `5c4fda0977625fbe403af6118ff5bc07bc9d7f97eedccd32663f76f9c047ec98` |
| `BIBLE.md` | `1433c2d3d18dd7397f8a739765c7f8e4c36f4b79e41c2809f596fa7fe3bf59b0` |
| `PLAN.md` | `4b0a5449b6a184ba2646fa3a9104f0217508893830236a1b7dca6d04fe18d286` |
| `PASSDOWN.md` | `930fc89b72c7d442fa1892b55ec69cac2cfa169dacd51342fdc86dfe08c3c440` |
| `data/MANIFEST.json` | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |

The live manifest has schema version 1, the pinned Payne Zero commit, 37
unique entries, and exactly the four allowed roles:
`static`, `subset`, `fixture`, and `golden`. Each role currently agrees with
its physical role directory. The atmosphere fixture, synthesis golden, and
later atmosphere golden destinations are all absent, and none has a manifest
entry. Both planned detached Chapter 6 authorization records are absent.

### 1.1 Chapter 5 publication precedent

The following exact Chapter 5 publication objects were inspected:

| object | SHA-256 |
| --- | --- |
| publisher contract | `8a58866b315b2cdd1ca769436aa0ccdbed3b08e4ef180bd309179d6cc7e04c08` |
| publisher | `6e5ed476cbcd64b1599c9f369bf5ed746d9f561f9bd77c13db8b9ec1c1cdad81` |
| publisher tests | `074975d36e8aa57f8173eebfe2e24eec6df52cff0a7fc8619ec4139d32144a43` |
| publisher implementation acceptance | `d2315fc2c22b5767ac4df2a3bfd50fd15c84fb5a77ec5c525a46cd26820604f1` |
| candidate acceptance | `35a7c5c28bab264703fa2606bf4bd806288bf217147778ee6b5d71e991721be8` |
| detached publication record | `80bcfea8e6a817ac1a279b136a247cdd482af1b94af312e94dfa564ad40cafaf` |
| detached-record review | `cba62ced80f7e4da18273278d7e10499f1b22204d7faf51e29ff019afd860dc7` |
| manifest synchronizer | `668ca1e18162ce1b84a880380abdae3c4e3797741f0fda2be4f61447c555e26c` |
| Chapter 5 manifest tests | `8808c1e6fccb7ecdcee370b0c1edf477f5d46da5bd08ceeeec31c89394072396` |
| repository manifest tests | `e15ed8328c05e34f73eefa4f7dce7d2f2cb5052294a84b4a7f5eabec74c2fa1c` |

Chapter 5's authorization binds the exact prepublication manifest and accepted
candidate bytes. It does **not** bind a postpublication manifest hash that
contains the authorization's own hash. Only after the record is final does
the manifest entry record
`publication_acceptance_sha256 = SHA256(record)`. That ordering is
constructible and is the relevant cycle-free precedent.

### 1.2 Accepted Chapter 6 synthesis boundary

All synthesis identities frozen by the contract reproduced:

| object | SHA-256 |
| --- | --- |
| synthesis fixture/oracle plan | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| scientific-worker acceptance | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |
| compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` |
| compact-assembler acceptance | `a0530cd08d5f0ddcc96b51fdaab4520aa89e62ca65850cf855ad4ede22251a33` |
| deterministic compact writer | `3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb` |
| compact-writer tests | `9601f8717a29ef51f32a62fd0a73c4d3db4b41c4f6ac08374be6149df4030bfc` |
| repaired writer candidate record | `540cf57126df93ee34d02c3da446a6ef109b93a8b17d60514439e12a8f63fc71` |
| complete compact-writer audit | `b888b49226e8ca6407c8226a3c021efb88fa100623fef27dd62e9beba43f2535` |

The accepted audit freezes 213 members, 1,294,865 archive bytes, archive
SHA-256
`b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850`,
compact schema digest
`911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde`,
compact payload fingerprint
`ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9`,
and raw ownership digest
`5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675`.
The contract transcribes these values correctly.

The distinction between internal
`meta__archive_kind = "synthesis_one_line_comparison_candidate"` and the
later external manifest role `golden` is scientifically honest. The contract
correctly forbids the publisher from rewriting the accepted member merely to
change its lifecycle wording.

### 1.3 Accepted Chapter 6 atmosphere precursor

The atmosphere archive writer was treated as unresolved, as required. The
accepted precursor identities reproduced:

| object | SHA-256 |
| --- | --- |
| fixture/oracle plan | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| in-memory scientific worker | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| worker tests | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| final worker audit | `336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e` |
| staged converter | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| converter acceptance | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |

The accepted scientific mapping has nineteen members, schema digest
`f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698`,
and payload fingerprint
`f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663`.
Those mapping identities do not establish canonical archive bytes, and the
contract correctly leaves the archive writer, acceptance, size, hash, schema
version, and serialized schema digest unresolved. No current atmosphere
archive-writer byte was treated as accepted evidence in this review.

## 2. Blocking finding

### P0 — the authorization and postpublication manifest form an indirect hash cycle

The authorization schema requires the record to contain:

1. the exact proposed one-entry manifest object;
2. that object's digest; and
3. the deterministically computed postpublication manifest SHA-256.

The later manifest entry is required to contain:

1. the exact authorization-record SHA-256; and
2. the authorization-record-review SHA-256.

Let:

```text
A = final authorization-record bytes
R = later independent record-review bytes
M = final postpublication manifest bytes
```

The contract therefore requires:

```text
A contains SHA256(M)
M contains SHA256(A)
M contains SHA256(R)
R is created only after A is final and freezes SHA256(A)
```

The first two requirements alone make `A` and `M` mutually recursive. Adding
`SHA256(R)` makes the chain longer, not cycle-free:

```text
A -> SHA256(M) -> SHA256(R) -> SHA256(A)
```

Saying that `A` has no literal self-hash does not close an indirect
self-reference through `M`. Constructing these exact bytes would require an
unavailable SHA-256 fixed point. Omitting the future hashes from the proposed
entry would avoid the cycle only by violating the contract's requirement that
the record bind the exact proposed entry and exact postpublication manifest.

This defect blocks:

- creation of either final authorization record;
- independent reproduction of either promised postpublication manifest hash;
- the exact registered-state no-op in Section 8;
- validation of the record-review hash in authorized dry-run mode;
- both \(M_0 \rightarrow M_1\) and \(M_1 \rightarrow M_2\) transitions.

#### Required closure

Adopt one explicit acyclic authority order. The Chapter 5 pattern is a valid
starting point:

1. the authorization binds the exact prepublication manifest, candidate,
   role, destination, and a deterministic manifest-entry template that does
   not contain future record/review hashes;
2. finalize and independently review the authorization;
3. derive the one-entry manifest delta using the now-known authorization hash
   and, only if a separately specified trust model truly needs it, the
   now-known review hash;
4. do **not** place the resulting postpublication manifest hash back into the
   already-final authorization;
5. freeze the resulting manifest in the postpublication audit.

Another acyclic design is acceptable, but the revised contract must show the
dependency graph explicitly and demonstrate that every hash is known before
the bytes containing it are finalized. It must also name and define the
record-review object if the publisher is expected to parse it as authority.

## 3. P1 findings

### P1-1 — exact manifest byte construction is not defined against the current manifest

The contract requires an exact postpublication manifest SHA-256 and says the
new entry is added in “canonical path order.” The live manifest entry list is
not lexically path-sorted: its first divergence from lexical order is at
index 1. The accepted synchronizer preserves existing order, appends new
entries, and serializes with:

```python
json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
```

The contract neither adopts that rule nor defines another complete canonical
encoding. “Deterministic UTF-8 JSON” does not determine:

- whether the new entry is appended or inserted;
- whether all 37 existing entries retain their exact sequence;
- object-key order, including the exhaustive member records;
- indentation, separators, escaping, and terminal newline;
- whether pre-existing JSON bytes may be normalized even when values compare
  equal.

Deep value comparison does not prevent reordering existing entries or object
keys. A self-consistent authorization could therefore bless a broad manifest
rewrite while claiming a one-entry semantic delta.

The revised contract must freeze one byte algorithm: preserve the exact
existing entry sequence and key order, define the exact insertion point,
define the exact serializer, and prove that deletion of the one new entry from
the proposed document reconstructs the accepted prepublication document at
both the parsed and byte-policy levels. If the existing synchronizer is not
the publisher, its behavior must not be assumed implicitly.

### P1-2 — nested allowlisted directory creation lacks a durability protocol

The synthesis phase may create:

```text
data/golden/payne_zero/chapter06
data/golden/payne_zero/chapter06/synthesis
```

Section 8 requires exclusive creation and identity revalidation, but no
`fsync` of each new directory and the directory in which each new name was
created. Later `fsync(destination_parent)` makes the artifact entry inside
`synthesis` durable, and `fsync(data)` makes the manifest replacement durable;
neither recursively guarantees durability of the new `chapter06` entry in
`payne_zero` or the new `synthesis` entry in `chapter06`.

A successful report could therefore durably register a manifest while the
new nested directory chain has not met the contract's own crash-durability
standard. Require, after every allowlisted `mkdir`, identity revalidation plus
`fsync` of the created directory and its immediate parent before proceeding.
Add failures for each ancestor `fsync` to the adversarial matrix.

### P1-3 — path representation and review trust anchors are underspecified

The planned record names and all current repository identity conventions use
canonical repository-relative strings such as
`design/chapter06_synthesis_publication_acceptance.json`. Section 5 then says
the strict parser rejects “relative ... paths” without distinguishing:

- a fixed repository-relative identity stored in JSON; from
- a caller-supplied relative filesystem path used for access.

Literal application rejects the planned records and the Chapter 5-compatible
representation; permissive application leaves the alias rule undefined.

The contract also requires exact publisher tests, publisher implementation
acceptance, candidate-byte acceptance, and record-review hashes without
naming the two publisher paths, their test paths, the two byte-acceptance
paths, or the two record-review paths and schemas. The atmosphere unresolved
block also omits an explicit writer-candidate-record path/hash even though the
authorization schema requires it.

The revision must enumerate these canonical objects. On-disk access should be
through fixed absolute canonical paths derived from the fixed repository
root, while JSON identity fields should use exact allowlisted POSIX
repository-relative literals with explicit rejection of `.`, `..`, alternate
separators, Unicode aliases, or caller-chosen roots.

### P1-4 — external Numba-cache lifecycle is stated too rigidly for the accepted atmosphere route

The accepted atmosphere worker requires a distinct, existing, empty,
external, nonsymlink `NUMBA_CACHE_DIR` at process start. Its compiled Numba
route may legitimately populate that disposable directory during the
capture. Section 6 instead requires every cache root to be “empty afterward”
and removed before success, without defining whether “afterward” means after
the child exits, after reviewed inventory/cleanup, or after the top-level
writer returns.

Because the atmosphere archive writer is deliberately unresolved, the
contract must not silently pre-accept one cache behavior. Freeze a
lane-specific rule after its writer audit:

- both lanes start with distinct empty external roots;
- all cache paths and pre/post identities are checked;
- synthesis may require post-child emptiness if that is its accepted result;
- the atmosphere lane may allow a reviewed compiler-cache inventory;
- every owned cache root is disposed before the accepted writer returns;
- neither cache content nor cache path enters candidate science bytes.

This preserves deterministic provenance without forcing a valid cached Numba
execution to masquerade as a no-cache execution.

## 4. P2 findings

### P2-1 — the `data/` snapshot vocabulary does not cover the actual tree

The contract asks for “every existing `data/` path by role,
regular-file/no-symlink state, size, and SHA-256.” The tree contains
directories and the intentionally unmanifested regular file
`data/README.md`; neither has one of the four numerical-data roles, and
directories are not regular files with file hashes.

Specify separate inventories for:

- all directory components with type, symlink state, device/inode, owner, and
  mode;
- all regular files with size and SHA-256;
- manifest-backed files with their exact role and entry identity;
- a closed allowlist of non-manifest support files such as `data/README.md`.

Also state that successful synthesis publication may add the two explicitly
allowlisted parent directories in addition to the artifact and one manifest
entry; otherwise the “only artifact plus manifest delta” statement is
literally inconsistent with Section 8.

### P2-2 — crash-left temporary names need an explicit next-invocation disposition

Handled failures can remove an invocation-owned staging file, but a process
terminated after staging cannot perform cleanup. The contract correctly
forbids broad or recursive repair, yet its three “complete states” omit an
orphan hidden artifact stage or manifest temporary file.

Define these as inert but incomplete states. A later publisher should either
hard-stop for an independent cleanup decision or use a separately reviewed
exact-name/inode recovery record; it must not infer ownership from a filename
prefix. The adversarial test should prove that a crash-left stage never
becomes a candidate, target, or manifest input.

## 5. Controls that passed design review

Subject to the findings above, the following design decisions are sound and
should be retained.

### 5.1 Scientific roles and ownership

- The atmosphere product is an input-only nineteen-member integration
  `fixture`, not a golden.
- The synthesis product is a 213-member comparison-only `golden`, not a
  fixture.
- The raw Fe I record remains a `subset`.
- Harris, FASTEX, and other invariant tables remain `static`.
- The atmosphere fixture contains no line output.
- The synthesis golden contains no copied Chapter 5 input state, no full
  continuum slab, and no atmosphere-lane member.
- The later atmosphere output golden is explicitly outside this phase and
  depends on the registered fixture through a new lifecycle.

This agrees with `BIBLE.md`, `PLAN.md`, `PASSDOWN.md`, both lane plans, and the
accepted compact-ownership audit.

### 5.2 Phase order and partial-state semantics

The fixed order is correct:

```text
accepted atmosphere bytes
  -> atmosphere authorization against M0
  -> fixture install and registration
  -> independent M1 audit
  -> synthesis authorization against exact M1
  -> synthesis install and registration
  -> independent M2 audit
  -> later atmosphere oracle/golden lifecycle
```

This prevents both lanes from using the same stale prepublication manifest.
It also preserves the causal fixture-before-atmosphere-oracle dependency.

The artifact-before-manifest order is correct. An exact unregistered artifact
is inert and recoverable only under the same authorization and unchanged
prepublication manifest. A nonexact target, registered missing target, or
manifest mismatch is a hard stop. After manifest replacement, forward audit
rather than destructive rollback is the safe rule.

### 5.3 Fixed target allowlists and no-overwrite behavior

The two artifact destinations are exact and role-correct. Only the two missing
synthesis parent components may be created. No force, replace, repair, merge,
destination, alternate-root, or partial-lane option is allowed.

The target-state rule is strong: absent or an exact regular nonsymlink
single-link file. The atomic create-if-absent syscall, rather than an earlier
absence check, decides races. `link` from an invocation-owned same-filesystem
stage or a reviewed `RENAME_NOREPLACE` primitive can satisfy this; plain
rename/replace, pre-delete, and check-then-overwrite cannot. Exact-existing is
an idempotent validation path, not permission to rewrite.

### 5.4 Symlink, parent, and TOCTOU defenses

The required combination is appropriate:

- fixed canonical root and destinations;
- no-follow, directory-relative access;
- retained directory descriptors;
- device/inode comparisons;
- nonsymlink regular-file checks;
- a stable data-directory lock across file plus manifest phases;
- rehash of source, candidate, authorization, and manifest immediately before
  each mutation;
- atomic no-replace artifact installation;
- atomic replacement of only the manifest;
- fresh no-follow readback.

The contract honestly limits this to cooperative races and ordinary
substitution rather than claiming cryptographic protection from a hostile
same-account process.

### 5.5 Artifact write, readback, and manifest ordering

The proposed file sequence is safe in principle:

```text
create-exclusive hidden stage
  -> complete write
  -> reject short write
  -> fsync file
  -> descriptor readback and semantic validation
  -> atomic no-replace install
  -> fsync destination parent
  -> remove only identity-matched stage
  -> fsync destination parent
  -> no-follow final readback
```

The manifest is correctly installed only after the exact artifact and under
the same lock. It is prepared in a create-exclusive sibling temporary file,
fsynced, read back, reparsed, and compared before the one allowed manifest
replacement. No existing artifact or manifest entry may be edited in place.

### 5.6 Reproducible dry run

The dry-run design correctly requires:

- two unrelated top-level writer calls, each retaining its own accepted
  internal A/B proof;
- byte-identical candidates and accepted summaries;
- untrusted ZIP/NPY decode, complete semantic/ownership validation, and
  canonical byte-for-byte re-encoding;
- no canonical parent, stage, lock, candidate, artifact, or manifest file;
- deterministic JSON with nondeterministic origin evidence reduced to
  topology booleans;
- no reachable mutation path in verification-only execution.

This is stronger than a single candidate hash check and should be preserved
after the authority cycle is repaired.

### 5.7 Adversarial matrix

The matrix covers the essential families: unresolved common gate, bound
identity drift, capture alias/reuse, archive canonicality, role leakage,
authorization mutation, path substitution, target/parent/manifest races,
concurrent lanes, I/O failures, crash points, idempotence, partial states, and
source/data drift. It also correctly requires all negative cases to preserve
every pre-existing canonical byte.

Add the missing ancestor-directory `fsync` failures, manifest byte-order
mutations, review-object mutations under a named schema, and crash-left
temporary-name cases when repairing the P1/P2 findings.

## 6. Non-authorization check

**PASS.** The contract repeatedly and unambiguously says:

- design only;
- no writer completion inferred from the accepted atmosphere mapping;
- no candidate path or role granted by in-memory bytes;
- no publisher or detached record exists;
- no fixture, golden, or manifest mutation is authorized;
- every unresolved atmosphere archive lock is a hard failure.

The current repository state agrees: neither canonical artifact, neither
authorization record, nor any Chapter 6 lane-artifact manifest entry exists.
The document does not accidentally authorize publication and does not
conflate fixture and golden roles.

## 7. Final disposition

**REJECT** contract SHA-256
`5c4fda0977625fbe403af6118ff5bc07bc9d7f97eedccd32663f76f9c047ec98`
as a prerequisite for implementing or accepting either publisher.

Retain the role split, fixed phase order, exact target allowlists, atomic
create-if-absent installation, shared-lock artifact/manifest sequence,
partial-state discipline, dry-run topology, and adversarial coverage. Before
implementation:

1. remove the indirect authorization/manifest/review hash cycle;
2. freeze exact manifest insertion and byte serialization;
3. add durability for every newly created directory component;
4. enumerate canonical publisher, byte-review, and record-review objects and
   clarify path representations;
5. make cache and snapshot/recovery semantics exact.

The unresolved atmosphere serialized identity remains an independent hard
gate even after these contract defects are repaired. This audit is not
authorization to create a publisher, authorization record, fixture, golden,
or manifest edit.

---

## Repair re-audit — 2026-07-30

Status: independent read-only re-audit of the repaired design  
Disposition: **ACCEPT the exact repaired contract as the design prerequisite
for the later, still separately gated publisher lifecycle; this is not
implementation or publication authorization**

This section supersedes the rejection above only for the exact repaired
contract recorded here. The original rejection remains immutable evidence of
the indirect hash cycle and six narrower design defects in the first
revision. This re-audit appended only this section. It did not edit the
contract, any writer, publisher, test, source, data file, manifest,
authorization, review, fixture, golden, cleanup record, or external
read-only tree.

### R1. Exact repaired snapshot

The assigned contract and immutable pre-append rejection matched their
expected identities before review:

| object | reviewed SHA-256 |
| --- | --- |
| repaired publisher contract | `f424c47b3ff9f14702c19b6b0444410d04e67bac9037bb9b01e97be8f03cdbe7` |
| this independent audit before this append | `4327c43e7bf58e776dd65a2939409c2b44a923c3b13a1bc01d56dc343c8f50bc` |
| `BIBLE.md` | `65387d009b732446252b5392afcbaa7a12fcb2c6db6083d9cf8eed0b6b1b36ac` |
| `PLAN.md` | `77f76043d97a494c373aa1d4fe9133bda1d98041017c4c892881aee41cc3014e` |
| `PASSDOWN.md` | `3ea9ea4ba9467062c0b0aae559ab7d90a777c4d29761b16e5ee066a8a4cba4e8` |
| `data/MANIFEST.json` | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |

The accepted synthesis identities transcribed in Section 2.1 remain the same
ones independently checked in the original audit:

```text
writer
  3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb
writer tests
  9601f8717a29ef51f32a62fd0a73c4d3db4b41c4f6ac08374be6149df4030bfc
writer candidate record
  540cf57126df93ee34d02c3da446a6ef109b93a8b17d60514439e12a8f63fc71
complete writer audit
  b888b49226e8ca6407c8226a3c021efb88fa100623fef27dd62e9beba43f2535
archive bytes
  1,294,865
archive SHA-256
  b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850
```

The accepted atmosphere scientific-worker, worker-test, final-audit,
converter, and converter-audit identities also remain:

```text
21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531
611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42
336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e
4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb
60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75
```

No current atmosphere archive-writer byte was treated as accepted. At review
time candidate files existed at the three named writer-side paths, but the
independent writer-acceptance path was absent and every contract hash/byte
lock for that archive remained `UNRESOLVED`. Existence at a planned path is
not authority.

### R2. P0 closure — the new hash dependency graph is acyclic

**CLOSED.** The repaired contract removes the postpublication manifest hash
from authorization `A`. It also removes the review hash, realized-entry hash,
and postpublication-audit hash from both `A` and review `R`.

The accepted graph is now:

```text
accepted writers/sources -> C
accepted publisher/contract -> P
(C, P, pre-manifest M, placeholder template T) -> A
A -> R
(T, SHA256(A), SHA256(R)) -> realized entry E
(M, E) -> post-manifest N
(A, R, E, N, artifact) -> postpublication audit Z
```

An independent topological-sort probe used the complete dependency set,
including the writer inputs. It returned:

```text
W, P, M, T, C, A, R, E, N, Z
```

with every node consumed and no residual positive in-degree. None of the
forbidden edges

```text
E -> A   N -> A   Z -> A
E -> R   N -> R   Z -> R   R -> R
```

exists in the repaired design.

The two placeholders are fixed literal complete values and may each occur
exactly once:

```text
__LATE_BOUND_AUTHORIZATION_SHA256__
__LATE_BOUND_RECORD_REVIEW_SHA256__
```

`A` freezes `T`; `R` freezes `SHA256(A)` and the same template digest; only
then are the two known hashes substituted to form `E`. `N` and `Z` are
strictly later objects. The contract explicitly forbids copying any realized
hash back into `A` or `R`.

An independent in-memory construction using the live manifest and a minimal
ordered two-placeholder template demonstrated constructibility rather than
merely inspecting arrows:

```text
template digest
  8c587743572211822fef11b7e2eb85034aab3cea802c9b3bc44c436e61b6e5b8
authorization hash
  022c06f39993d6a228d5276f75559d3806a683bfbdc93e4f6980bea30b839cad
review hash
  e61fb6c480bcdc7cdf9a0158fbf060faa84c28b374f41b787377a71c2981917d
realized dummy post-manifest hash
  e2975b0a507655faf20ca9cd6720e8c1f3d7120cfc6044ea0e18de069a82d953
```

Both placeholders disappeared from the realized entry, `A` did not contain
the later `N` hash, `R` did not contain its own hash, and deleting the final
dummy entry reproduced the exact original manifest bytes. No hash fixed point
or guessed future value was needed.

### R3. P1-1 closure — manifest order, insertion, and bytes are exact

**CLOSED.** The repaired contract no longer calls the manifest path-sorted.
It freezes all of the following:

- order-preserving, duplicate-rejecting JSON parsing;
- exact top-level key order
  `schema_version`, `payne_zero_commit`, `entries`;
- the complete existing entry sequence and every existing object-key
  sequence;
- the exact serializer:

  ```python
  json.dumps(
      object,
      indent=2,
      ensure_ascii=False,
      allow_nan=False,
      sort_keys=False,
      separators=(",", ": "),
  ) + "\n"
  ```

- one insertion rule: append `E` as the final `entries` element;
- a compact, separately specified serializer for the ordered template digest;
- lexical archive-member order and exact per-member metadata key order;
- structural substitution of exactly the two placeholder scalar values;
- parsed deep equality of every existing entry position, key order, and
  value;
- delete-last plus reserialization equality with the exact prepublication
  bytes and SHA-256.

Independent execution against the current manifest reproduced:

```text
schema version                         1
entries                               37
unique entry paths                    37
path sorted                           False
top-level key order                   exact
duplicate-free parse                  pass
exact stated-encoder round trip       True
round-trip SHA-256
  d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a
```

The repaired rule therefore preserves the intentionally unsorted live
manifest. A sorted, inserted, reindented, reescaped, reordered, or
newline-changed document is explicitly an adversarial failure, not an
alternative canonicalization.

The exact-entry template and the full repository role tests remain mandatory
later gates. They must include every field required by the live manifest
schema, including the NPZ format and role-specific provenance; validation is
not satisfied merely by reproducing the byte encoder.

### R4. P1-2 closure — nested-directory creation is durable

**CLOSED.** For each missing allowlisted synthesis component, the repaired
contract now requires:

1. a retained and revalidated immediate-parent descriptor;
2. exclusive creation of one literal component;
3. no-follow opening and device/inode/owner/group/mode capture;
4. `fsync` of the new directory;
5. `fsync` of its immediate parent;
6. no-follow reopen and identity/metadata equality before descent.

`chapter06` must complete all six steps before `synthesis` begins. A failure
of the new-directory or parent `fsync` aborts before artifact staging.
Identity-matched handled cleanup is itself followed by parent `fsync`; crash
state is not inferred to be owned.

The adversarial matrix separately names failures of:

```text
fsync(new chapter06)
fsync(payne_zero)
fsync(new synthesis)
fsync(chapter06)
```

This closes the original gap in which later `fsync(synthesis)` and
`fsync(data)` could not establish durability of the two intermediate
directory names.

The later publisher implementation review must also prove that the manifest
temporary inherits the accepted manifest owner/group/mode before replacement;
that follows from the contract's exact one-entry-only manifest delta and
post-write metadata snapshot rather than granting permission for a metadata
rewrite.

### R5. P1-3 closure — identity types and all trust paths are explicit

**CLOSED.** Section 3.1 now enumerates, separately for both lanes:

- writer;
- writer tests;
- writer candidate record;
- writer acceptance;
- candidate-byte acceptance;
- publisher;
- publisher tests;
- publisher implementation acceptance;
- detached authorization;
- authorization-record review;
- postpublication audit.

It additionally names the shared contract and the one quarantine-cleanup
record. The atmosphere unresolved block now includes the previously missing
writer-candidate path and hash.

The record review is no longer unspecified prose. It is strict
duplicate-free UTF-8 JSON with nine exact ordered keys, exact lane-specific
`record_kind`, exact `ACCEPT` disposition, and bindings to the authorization,
candidate-byte acceptance, publisher acceptance, and template digest. It
contains the authorization hash but not its own hash.

The repaired path type boundary is unambiguous:

- JSON identity values are exact allowlisted ASCII POSIX
  repository-relative literals;
- `.` and `..` path components, empty components, aliases, backslashes,
  leading slash, case/Unicode variants, escapes, and caller roots are
  rejected;
- filesystem access starts only from the fixed absolute canonical host root
  and walks the allowlisted literal components with no-follow,
  directory-relative operations;
- the absolute host path is never substituted into a repository identity
  field;
- external pinned source paths are a distinct absolute source-identity type.

The explicit allowlist resolves the wording around dot characters:
extensions such as `.py` are part of exact accepted component names; only
standalone `.` or `..` alias components are forbidden.

### R6. P1-4 closure — cache lifecycle is lane-specific

**CLOSED.** Both lanes still require distinct, initially empty, external,
nonsymlink roots and complete pre/post path/inode/inventory evidence.
Thereafter:

- the already accepted synthesis writer owns its accepted empty-after-child
  rule;
- the unresolved atmosphere writer audit must freeze either post-child
  emptiness or an exact compiler-cache inventory and cleanup policy;
- every writer-owned root must be disposed before its accepted writer returns;
- cache paths, contents, PIDs, and origins never enter candidate science
  bytes.

The contract therefore preserves the common freshness boundary without
inventing an atmosphere post-child result before that writer has passed
independent review. The adversarial matrix now defers atmosphere cache
acceptance to that exact future audit rather than borrowing the synthesis
policy.

### R7. P2-1 closure — the actual `data/` tree has a complete vocabulary

**CLOSED.** The repaired snapshot separates:

1. every directory with path, type, no-symlink decision, device, inode,
   owner, group, and mode;
2. every regular file with the same identity metadata plus size and SHA-256;
3. an exact manifest-entry join with role, entry position/digest, bytes, and
   file hash;
4. a closed normal-state nonmanifest support allowlist.

The fourth inventory contains only:

```text
data/README.md
SHA-256
  1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b
```

Independent hashing reproduced that value. Symlinks, sockets, devices, and
other special nodes are forbidden. An exact unregistered target and a
quarantine temporary are separate lifecycle states, not silently added
support files.

The successful-delta rules now distinguish the lanes: atmosphere adds only
its artifact plus one manifest entry; synthesis may additionally add exactly
the two named durable directory components. This removes the original
contradiction between “artifact plus manifest only” and necessary synthesis
parent creation.

### R8. P2-2 closure — crash-left temporaries are quarantined, never inferred

**CLOSED.** A crash-left artifact stage or manifest temporary is now an inert
quarantine object. The next publisher:

- inventories its exact path, parent and object device/inode, metadata, link
  count, size, and hash;
- hard-stops;
- never decodes it as input;
- never treats a filename prefix, valid archive/JSON appearance, or candidate
  byte equality as ownership;
- never promotes, renames, or installs it.

Cleanup is outside both publishers and requires a separate duplicate-free
record at the one named path. That record binds each exact object and allows
only inode-revalidated `unlink_only` followed by immediate-parent `fsync`.
It permits no glob, prefix deletion, recursion, target deletion, or manifest
replacement. The contract neither creates nor authorizes the cleanup record.

Crash-left allowlisted directories are also incomplete states. An exact
empty directory may be adopted only after full metadata/content proof and
directory-plus-parent `fsync`; it is never inferred to belong to the prior
invocation or recursively deleted.

### R9. Role, ordering, no-replace, TOCTOU, and dry-run regression

**PASS.** The repair preserves every control accepted in Section 5 of the
original review:

- atmosphere remains a nineteen-member input-only `fixture`;
- synthesis remains a 213-member comparison-only `golden`;
- raw Fe I remains `subset`; Harris/FASTEX remain `static`;
- neither product may copy another role's state or output;
- the later atmosphere golden remains a separate post-\(M_2\) lifecycle;
- atmosphere publication and independent \(M_1\) audit precede synthesis
  authorization against exact \(M_1\);
- the two artifact paths and two optional synthesis directory components are
  the only targets;
- canonical installation remains atomic create-if-absent, never overwrite;
- exact-existing is validation/recovery, not replacement;
- the artifact is installed before its manifest entry under the same stable
  data lock;
- every mutation is preceded by retained-descriptor, device/inode, source,
  candidate, authorization/review, and manifest revalidation;
- failed post-install/pre-manifest state is an inert unregistered exact file;
- post-manifest failure is forward-audit state, never destructive rollback;
- verification-only mode performs two top-level accepted builds and complete
  decode/semantic/re-encode checks while exposing no reachable publication
  primitive and creating no repository path.

The expanded adversarial matrix now adds the repaired hash-graph,
identity/access-path separation, directory durability, exact manifest byte
policy, quarantine, cache, and nonmanifest-support cases. Every negative case
must preserve all pre-existing canonical file bytes.

### R10. Repository state and non-authorization

The live manifest remained exact at:

```text
d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a
```

It still has 37 unique entries and no entry for:

```text
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
data/golden/payne_zero/chapter06/synthesis/
  chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz
data/golden/payne_zero/chapter06/
  chapter06_atmosphere_one_line_cpu.npz
```

All three canonical artifacts were absent. Both detached authorization files,
both authorization-record reviews, both candidate-byte acceptances, both
publishers, both publisher test files, both publisher acceptances, both
postpublication audits, and the quarantine-cleanup record were absent at
review time.

The three named atmosphere writer candidate-side files existed with current
hashes:

```text
scripts/chapter06_atmosphere_fixture_writer.py
  0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538
tests/test_chapter06_atmosphere_fixture_writer.py
  c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a
design/chapter06_atmosphere_fixture_writer_candidate.md
  3947c17c47ecd9a56ac85ad44c4564264b99813fd3467d2cc6872a3d76dac8b0
```

Those hashes are deliberately **not** copied into the contract locks or
accepted here. The independent writer-acceptance path was absent. Candidate
existence remains scientifically and operationally powerless.

The repaired contract still begins and ends with **NOT AUTHORIZATION**,
requires both writer acceptances before either publisher may proceed, treats
every unresolved or candidate atmosphere value as a hard failure, and
creates no force/replace/repair/destination/alternate-root surface.

### R11. Residual unresolved gates

No P0, P1, or P2 design defect remains from the original rejection. The
following are intentional unresolved gates, not acceptance exceptions:

```text
ATMOSPHERE_DETERMINISTIC_WRITER_SHA256
ATMOSPHERE_WRITER_TEST_SHA256
ATMOSPHERE_WRITER_CANDIDATE_RECORD_SHA256
ATMOSPHERE_WRITER_ACCEPTANCE_SHA256
ATMOSPHERE_ARCHIVE_BYTES
ATMOSPHERE_ARCHIVE_SHA256
ATMOSPHERE_ARCHIVE_SCHEMA_VERSION
ATMOSPHERE_ARCHIVE_SCHEMA_DIGEST
```

Every future candidate-byte acceptance, publisher, publisher test,
publisher acceptance, detached authorization, authorization-record review,
and postpublication audit path/hash also remains unresolved until its own
independent gate. The manifest template `T` does not yet exist as authority.

Implementation review must concretely prove the contract rather than relying
on prose, including:

- exact no-follow descriptor operations and platform lock/no-replace calls;
- manifest-temporary owner/group/mode preservation;
- exact entry fields required by `tests/test_data_manifest.py` and the
  lane-specific exhaustive role tests;
- all directory and manifest `fsync` failures;
- source/data snapshot and quarantine transitions;
- dry-run reachability and zero repository delta.

These later obligations do not reopen the repaired design or authorize any
write.

### R12. Final repair disposition

**ACCEPT** repaired contract SHA-256
`f424c47b3ff9f14702c19b6b0444410d04e67bac9037bb9b01e97be8f03cdbe7`
as the independently reviewed design prerequisite for the future Chapter 6
two-lane publishers.

Acceptance means only that the role model, authority graph, exact
manifest-byte transition, durability, path types, cache boundary, snapshot,
quarantine, no-replace, TOCTOU, recovery, and dry-run contracts are coherent
and implementable. It does **not**:

- accept the current atmosphere writer candidate or serialized bytes;
- authorize implementation before the common writer gate closes;
- accept either future publisher or test suite;
- create or accept a detached authorization or its review;
- authorize creation of a fixture, golden, directory, cleanup record, or
  manifest entry;
- permit any write to Payne Zero or the paper tree.

Publication remains blocked by every unresolved gate in R11 and by the
contract's later independent byte, implementation, authorization,
postpublication, and manifest-role reviews.

---

## Accepted-atmosphere lock re-audit — 2026-07-30

Status: independent read-only review of only the accepted-atmosphere lock
revision
Disposition: **ACCEPT as a design prerequisite only; NOT AUTHORIZATION**

This third audit is deliberately narrow. It reviews the substitution of the
independently accepted atmosphere writer and archive facts into the already
accepted two-lane publisher design. It does not reopen, manufacture, or accept
any downstream candidate-byte, publisher, authorization, canonical-data, or
manifest object.

### A1. Reviewed identities

The exact reviewed contract is:

```text
design/chapter06_lane_artifact_publisher_contract.md
  9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774
```

The complete preceding audit prefix had SHA-256:

```text
df710f8d0895a70dc7852f5094b9ff58c0feefb547f76e6e0de95fd433e9291d
```

The four atmosphere writer-side identities transcribed by the contract match
the current files byte for byte:

| accepted writer-side object | SHA-256 |
| --- | --- |
| `scripts/chapter06_atmosphere_fixture_writer.py` | `0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538` |
| `tests/test_chapter06_atmosphere_fixture_writer.py` | `c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a` |
| `design/chapter06_atmosphere_fixture_writer_candidate.md` | `e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee` |
| `design/chapter06_atmosphere_fixture_writer_independent_audit.md` | `b946dcef0beeacf49a3da9ac036e21af7cd7b44d15092639cd3be744fb42f0f9` |

The last object independently accepts the first three at those identities. No
stale candidate identity from an earlier review remains in the revised
contract.

### A2. Accepted atmosphere bytes and version meanings

A fresh top-level invocation of the accepted zero-argument writer was run
under the contract's deterministic environment. The result was held in
memory, decoded independently, and compared with the accepted audit. It
reproduced all exact locked values:

| property | independently reproduced value |
| --- | --- |
| scientific member count | `19` |
| scientific array bytes | `357,984` |
| archive bytes | `363,050` |
| archive SHA-256 | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |
| scientific fixture schema digest | `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698` |
| scientific payload fingerprint | `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663` |
| NPY member format | version `2.0` for every member |

The decoded member names were exactly the accepted nineteen-member
`FIXTURE_SCHEMA`; no additional member contained `schema_version`.

The contract preserves the required distinction between the transient
scientific-capture envelope and the final archive:

- the accepted complete transient capture uses external
  `CAPTURE_SCHEMA_VERSION = 1`;
- the final archive contains only the nineteen scientific fixture arrays;
- it contains no embedded archive-schema-version or capture-schema-version
  member;
- NPY version `2.0` describes every member's encoding.

Therefore the future metadata vocabulary
`fixture_capture_schema_version = 1`,
`archive_contains_embedded_schema_version = false`, and
`npy_member_format_version = "2.0"` is exact. The revision correctly forbids
inventing `ATMOSPHERE_ARCHIVE_SCHEMA_VERSION`,
`meta__archive_schema_version`, or a twentieth archive member.

### A3. Exact cache boundary

The independent run also reproduced the accepted writer's complete cache
contract. Each of the two isolated child captures started with its own empty,
external, nonsymlink cache root. After each child, its exact inventory was:

```text
1 nonsymlink package directory
18 .nbi regular files
18 .nbc regular files
37 total entries
0 symlinks
0 non-Numba files
```

Both writer-owned cache roots were disposed before return. No cache path or
cache content entered the returned archive. This remains distinct from the
synthesis lane's accepted empty-after-child cache postcondition; the contract
does not infer one lane's cache behavior from the other.

### A4. Downstream authority remains absent

The live `data/MANIFEST.json` remained byte-exact at:

```text
d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a
```

The independent build reported
`artifact_file_write_performed = false`,
`manifest_mutation_performed = false`, and
`publication_authorized = false`. A complete pre/post data-file snapshot was
unchanged.

At review time, every exact Section 3.1 path from candidate-byte acceptance
downward was absent for both lanes: both candidate-byte acceptances, both
publishers, both publisher tests, both publisher implementation acceptances,
both detached authorizations, both authorization-record reviews, and both
postpublication audits. The named quarantine-cleanup acceptance was also
absent.

The three planned canonical artifacts were absent and had no manifest entry:

```text
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
data/golden/payne_zero/chapter06/synthesis/
  chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz
data/golden/payne_zero/chapter06/
  chapter06_atmosphere_one_line_cpu.npz
```

Thus accepting the atmosphere writer and in-memory candidate bytes closes
only the common writer gate. Candidate-byte acceptance and every later
publisher, publisher-test, publisher-acceptance, authorization,
authorization-review, file-installation, manifest-registration, and
postpublication-audit gate remain **UNRESOLVED**. Accepted bytes are evidence,
not publication authority.

### A5. Regression check of the accepted publisher design

The atmosphere lock revision does not weaken the prior accepted design:

- the dependency graph remains strictly forward
  `writers/sources -> C/P/M/T -> A -> R -> E -> N -> Z`, with no realized or
  postpublication hash copied backward;
- the exact unsorted-manifest round trip, append-only one-entry transition,
  placeholder realization, and delete-last reconstruction rules are
  unchanged;
- directory and parent `fsync`, no-follow path typing, fixed destinations,
  atomic no-replace installation, source/data snapshots, cache isolation,
  recovery states, and dry-run nonmutation remain required;
- crash-left stages, manifest temporaries, and new directories remain
  quarantine states requiring inode-bound reviewed cleanup rather than
  promotion or recursive deletion;
- the revised adversarial matrix now rejects any changed atmosphere
  writer-side identity, invented schema member, wrong NPY version, or cache
  inventory/disposal mismatch before a repository write.

No P0, P1, or P2 design defect was found in this lock-only revision.

### A6. Final lock-revision disposition

**ACCEPT** contract SHA-256
`9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774`
as the independently re-audited design prerequisite for later Chapter 6
two-lane publisher work.

**NOT AUTHORIZATION.** This disposition accepts only the coherent
substitution of the already independently accepted atmosphere writer and
in-memory archive locks into the previously accepted design. It does not
accept the still-absent candidate-byte records or publishers, create an
authorization, write a canonical fixture or golden, alter the manifest, or
permit any write to Payne Zero or the paper tree.
