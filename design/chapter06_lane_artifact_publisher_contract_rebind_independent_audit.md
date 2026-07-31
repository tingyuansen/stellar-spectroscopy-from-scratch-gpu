# Chapter 6 publisher-contract forward rebind — independent audit

Date: 2026-07-30  
Scope: read-only contract-rebind audit  
Disposition: **REJECT**

## 1. Decision

**REJECT** the forward-rebound publisher contract at SHA-256
`d52d86512f6a576dc7ba167a4e6f7436368ad8552b13cd6dbd4ce7ed93cfd076`
as a prerequisite for either publisher.

The repaired synthesis locks, role separation, local
`A -> R -> E -> N -> Z` publication graph, manifest-byte policy, path
confinement, durability, cache, snapshot, recovery, and quarantine rules all
pass this design audit. The blocking defect is the atmosphere candidate-byte
carry-forward:

1. atmosphere byte acceptance
   `79b20c065f5f1b588c6796d6413a1b2aa5e1e5a60056968fc7913cd961e12fc6`
   binds the exact historical contract
   `9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774`;
2. that acceptance explicitly says that **any** publisher-contract change
   invalidates the candidate-byte acceptance; and
3. the rebound contract is a different byte string but nevertheless embeds
   `79b20c…` and declares that gate accepted.

An immutable downstream record cannot waive its own exact-input condition
merely because a later contract calls it an upstream input. The scientific
atmosphere bytes remain reproducible, but the old acceptance is stale under
its own stated scope.

This is not publication authorization. I created no publisher, detached
authorization, authorization review, artifact, directory, manifest entry, or
canonical data file.

## 2. Exact candidate inputs

The two assigned files were regular nonsymlink files and matched their
requested identities:

| reviewed object | SHA-256 |
| --- | --- |
| `design/chapter06_lane_artifact_publisher_contract.md` | `d52d86512f6a576dc7ba167a4e6f7436368ad8552b13cd6dbd4ce7ed93cfd076` |
| `design/chapter06_lane_artifact_publisher_contract_rebind_candidate.md` | `516d358a5a993f369fa8113bfcafea5fb3e7156287ba80f1e85bba783e3e2513` |

The candidate report correctly identifies the previous accepted contract as
historical only:

```text
historical contract
  9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774

historical complete contract audit
  7f2517ad0abcca312dcf22785e483fe033519264d5513b16ebe6fa580d4521fd
```

Neither historical object accepts the current contract bytes.

## 3. Exact bound files

### 3.1 Repaired synthesis chain

All fourteen synthesis path/hash bindings in the contract matched live
regular nonsymlink files:

| bound object | independently recomputed SHA-256 |
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

### 3.2 Atmosphere chain

All atmosphere precursor, writer, and byte-review identities also matched
live regular nonsymlink files:

| bound object | independently recomputed SHA-256 |
| --- | --- |
| `design/chapter06_atmosphere_fixture_oracle_plan.md` | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| `scripts/chapter06_atmosphere_fixture_worker.py` | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| `tests/test_chapter06_atmosphere_fixture_worker.py` | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| `design/chapter06_atmosphere_fixture_worker_independent_audit.md` | `336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e` |
| `scripts/chapter06_atmosphere_line_converter.py` | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| `design/chapter06_atmosphere_converter_independent_audit.md` | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |
| `scripts/chapter06_atmosphere_fixture_writer.py` | `0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538` |
| `tests/test_chapter06_atmosphere_fixture_writer.py` | `c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a` |
| `design/chapter06_atmosphere_fixture_writer_candidate.md` | `e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee` |
| `design/chapter06_atmosphere_fixture_writer_independent_audit.md` | `b946dcef0beeacf49a3da9ac036e21af7cd7b44d15092639cd3be744fb42f0f9` |
| `design/chapter06_atmosphere_fixture_byte_acceptance.md` | `79b20c065f5f1b588c6796d6413a1b2aa5e1e5a60056968fc7913cd961e12fc6` |

The byte-acceptance file exists and has the expected hash. Its existence and
byte identity do not make its disposition current after a bound input
changed.

## 4. Independently reproduced scientific and archive locks

### 4.1 Synthesis

A live zero-argument build independently reproduced:

| property | exact result |
| --- | --- |
| raw members | `754` |
| raw mapping digest | `09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c` |
| raw schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| raw physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| raw full fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |
| compact members | `213` |
| compact array bytes | `1,235,275` |
| compact schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| compact payload fingerprint | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| raw-ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| archive bytes | `1,294,865` |
| archive SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |

The complete ownership partition was `250 final`, `123
derived_digest_only`, and `381 intentionally_ephemeral`. A/B raw mappings,
raw transport bytes, independently assembled compact mappings, and final
archive bytes were equal.

The builder reported false for publication authorization, artifact writing,
and manifest mutation.

### 4.2 Atmosphere

A separate live zero-argument build independently reproduced:

| property | exact result |
| --- | --- |
| fixture members | `19` |
| scientific array bytes | `357,984` |
| fixture schema digest | `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698` |
| payload fingerprint | `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663` |
| complete transport bytes | `33,147,771` |
| complete transport SHA-256 | `0523ed254a78edaa07480bc30f23082f535e885f46d867de10f92cba1acd5b16` |
| archive bytes | `363,050` |
| archive SHA-256 | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |
| NPY member version | `2.0` |

The two child PIDs, origins, and cache roots were distinct. Each initially
empty external nonsymlink cache contained the accepted 37-entry inventory
after its child: one package directory, eighteen `.nbi` files, and eighteen
`.nbc` files. Both roots were disposed before return. Fixture, evidence,
transport, and final A/B equality passed.

The builder reported false for fixture publication, golden publication,
artifact writing, manifest mutation, and publication authorization.

These successful builds show that the rejection is not based on a changed
scientific candidate. Exact byte reproducibility cannot override an explicit
trust-object invalidation rule.

## 5. Stale-history scan

The rebound contract contains zero occurrences of each historical synthesis
hash:

- plan `d6e6ae1f…`;
- raw full fingerprint `33d1dec1…`;
- assembler/tests `62b7aac3…` / `53111433…`;
- compact candidate/audit `41daee0a…` / `a0530cd0…`;
- compact payload `ce5d1c1d…`;
- writer/tests `3da5191d…` / `9601f871…`;
- writer candidate/audit `540cf571…` / `b888b492…`; and
- archive `b92e44a1…`.

It likewise contains neither historical writer-candidate path
`design/chapter06_synthesis_compact_writer_candidate.md` nor historical
writer-audit path
`design/chapter06_synthesis_compact_writer_independent_audit.md`.

Each of those fourteen historical values or paths occurs exactly once in the
candidate report, only in tables and prose explicitly labelled
non-authoritative history. Each of the seventeen requested repaired synthesis
values occurs exactly once in the rebound contract.

This portion of the rebind passes.

## 6. Trust-object paths and role asymmetry

The contract contains exactly eleven planned trust-object rows for each lane.
Every path is an ASCII, POSIX, repository-relative identity with no leading
slash, empty component, `.`, `..`, backslash, alternate separator, or
case/Unicode alias.

The rows are:

| trust object | atmosphere fixture lane | synthesis comparison lane |
| --- | --- | --- |
| deterministic writer | `scripts/chapter06_atmosphere_fixture_writer.py` | `scripts/chapter06_synthesis_compact_writer.py` |
| writer tests | `tests/test_chapter06_atmosphere_fixture_writer.py` | `tests/test_chapter06_synthesis_compact_writer.py` |
| writer candidate | `design/chapter06_atmosphere_fixture_writer_candidate.md` | `design/chapter06_synthesis_writer_rebind_candidate.md` |
| writer acceptance | `design/chapter06_atmosphere_fixture_writer_independent_audit.md` | `design/chapter06_synthesis_writer_rebind_independent_audit.md` |
| candidate-byte acceptance | `design/chapter06_atmosphere_fixture_byte_acceptance.md` | `design/chapter06_synthesis_candidate_byte_acceptance.md` |
| publisher | `scripts/build_chapter06_atmosphere_fixture.py` | `scripts/build_chapter06_synthesis_golden.py` |
| publisher tests | `tests/test_chapter06_atmosphere_fixture_publisher.py` | `tests/test_chapter06_synthesis_golden_publisher.py` |
| publisher acceptance | `design/chapter06_atmosphere_fixture_publisher_independent_audit.md` | `design/chapter06_synthesis_golden_publisher_independent_audit.md` |
| detached authorization | `design/chapter06_atmosphere_fixture_publication_acceptance.json` | `design/chapter06_synthesis_publication_acceptance.json` |
| authorization review | `design/chapter06_atmosphere_fixture_publication_record_review.json` | `design/chapter06_synthesis_publication_record_review.json` |
| postpublication audit | `design/chapter06_atmosphere_fixture_postpublication_audit.md` | `design/chapter06_synthesis_postpublication_audit.md` |

Only the two candidate-byte paths presently exist below writer acceptance.
Every publisher, publisher test, publisher acceptance, authorization,
authorization review, and postpublication path is absent.

The role split is exact and coherent:

| lane | eventual role | exact purpose |
| --- | --- | --- |
| atmosphere | `fixture` | nineteen-member, input-only, 80-depth integration state with no line output |
| synthesis | `golden` | 213-member comparison-only oracle opened only after the reader-built result exists |

The contract excludes a stored synthesis fixture, copied atmosphere/input
state in the synthesis golden, output in the atmosphere fixture, static or
subset reclassification, and reconstruction of one lane from the other. It
also keeps the later atmosphere comparison golden outside this phase.

This portion passes.

## 7. Blocking atmosphere carry-forward and global graph

### 7.1 Exact stale binding

The atmosphere acceptance's Section 2.3 freezes:

```text
publisher contract
  9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774

publisher-contract audit
  7f2517ad0abcca312dcf22785e483fe033519264d5513b16ebe6fa580d4521fd
```

Its residual-scope rule then states that a change to the publisher contract
invalidates the candidate-byte acceptance and requires the gate to repeat.
That language has no exception for a synthesis-only edit, a byte-preserving
atmosphere result, or a source-faithful forward rebind.

The candidate's proposed argument—treat the immutable old acceptance as an
upstream object because atmosphere science and serialization did not
change—would silently weaken the exact-input rule after the fact. That is not
permitted. The accepted record froze the shared two-lane contract, not a
semantic projection of only its atmosphere paragraphs.

### 7.2 Local graph passes; global trust closure fails

The contract's planned publication graph is locally acyclic. Independent
topological sorting consumed every node. One valid order was:

```text
contract, manifest, sources, template, writers,
publisher review, candidate-byte review,
authorization, authorization review,
realized entry, postpublication manifest, postpublication audit
```

The five explicit forward edges
`A -> R`, `A -> E`, `R -> E`, `E -> N`, and `N -> Z` are present. Neither
`A` nor `R` binds `E`, `N`, `Z`, or a self-hash. The late-bound strings

```text
__LATE_BOUND_AUTHORIZATION_SHA256__
__LATE_BOUND_RECORD_REVIEW_SHA256__
```

occur exactly once each and only as the two complete placeholder values
specified by the contract.

The omitted byte-acceptance/contract relation is the blocker. With immutable
nodes:

```text
D0 = old contract 9ee002…
C0 = atmosphere acceptance 79b20c… that depends on D0
D1 = rebound contract d52d86… that embeds and claims C0

D0 -> C0 -> D1
```

That chain has no literal hash cycle because `D0` and `D1` are different, but
`C0` explicitly ceases to be accepted when `D1` replaces `D0`. It is a stale
trust claim.

Attempting to repair it by rerunning acceptance while retaining an acceptance
hash inside the final contract would create the actual recursion:

```text
final contract D1 -> new acceptance C1
new acceptance SHA256(C1) -> bytes of D1
```

Neither object can be finalized first. The local `A -> R -> E -> N -> Z`
repair therefore remains sound, but the complete trust graph is not yet
acyclic.

## 8. Synthesis historical rejection

The current synthesis candidate-byte path is a regular nonsymlink file:

```text
design/chapter06_synthesis_candidate_byte_acceptance.md
  474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1
```

Its disposition is a historical **REJECT** of the old writer against the
repaired plan. It accepts no synthesis bytes.

The rebind candidate report represents this honestly and explicitly. The
contract itself calls the row `UNRESOLVED`, which is directionally
fail-closed, but does not disclose that the planned path is already occupied
by a rejection. Its wording that an unresolved object must “exist and pass”
is incomplete because an object exists but did not pass.

The repaired contract must state the exact historical rejection and its
non-authoritative status. A later successful byte review must preserve that
failure history—by a clearly append-only superseding review or another
explicitly reviewed immutable path—and must never silently rewrite or
reinterpret `474e318…` as acceptance.

## 9. Manifest-byte and path/durability rules

Independent strict parsing of `data/MANIFEST.json` reproduced:

| property | exact result |
| --- | --- |
| bytes | `1,087,741` |
| SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| top-level key order | `schema_version`, `payne_zero_commit`, `entries` |
| entries | `37`, unique |
| path order | intentionally not sorted |
| ordered-path digest | `d63aa6dfd9209f21172c6ccf721cffc895835b8ee15a4ed160c5bebe0851b1aa` |
| roles | `6 fixture`, `12 golden`, `18 static`, `1 subset` |

The live bytes round-tripped exactly under the contract's order-preserving,
duplicate-rejecting, `sort_keys=False`, two-space-indented encoder with one
final newline.

The append-only rule preserves every existing entry and object-key position;
delete-last reconstruction must recover exact prepublication bytes. Template
member and field order is normative, and only the two complete placeholder
values may change. These rules remove normalization ambiguity.

The repository-relative identity versus absolute host-access distinction is
also exact. Access begins from the fixed canonical root
`/Users/ysting/stellar-spectroscopy-from-scratch-gpu`, uses directory-relative
no-follow walking and retained device/inode evidence, and accepts no
caller-selected root or destination.

For the nested synthesis destination, the contract requires each new
component to be created exclusively, opened no-follow, recorded, `fsync`ed,
followed by immediate-parent `fsync`, and reopened at the same device/inode.
`chapter06` must complete before `synthesis`. Failure aborts before staging.
Handled cleanup is inode-bound, nonrecursive, and followed by parent
`fsync`. This closes the directory-durability boundary at design level.

The atomic file path is create-if-absent on one filesystem, never plain
rename/replace or pre-delete. Target, parent, manifest, candidate,
authorization, and snapshot identities are rechecked immediately before
mutation. These mechanics pass this audit.

## 10. Snapshots, cache, recovery, and quarantine

The contract specifies four closed `data/` inventories: directories, regular
files, manifest joins, and the sole normal nonmanifest support file
`data/README.md`. The latter reproduced SHA-256
`1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b`.

Before and after both live writer builds:

| snapshot | exact result |
| --- | --- |
| `data/` directories | `12` |
| `data/` regular files | `39` |
| regular-file bytes | `30,046,405` |
| aggregate SHA-256 | `288bbe4c6bcbd20da8390f99fb6cc45e07ee6eed24197371d3520aed39f7d004` |
| regular-file symlinks | `0` |
| manifest before/after | exact |

The pinned Payne Zero checkout remained at commit
`9c44001feae40b85146630499e6f8a5fed42e5af`.

The lane-specific cache rules are exact and match the live writers:

- synthesis requires distinct, initially empty, external nonsymlink roots
  that remain empty after each child;
- atmosphere requires the exact 37-entry populated inventory after each
  child; and
- every writer-owned root is disposed before return.

The recovery states are disjoint and fail closed: absent target, exact
unregistered target under the same authorization and unchanged manifest, or
exact registered target with uniquely recomputed postmanifest bytes.
Nonexact, registered-missing, unexpected-directory, and changed-manifest
states reject.

Crash-left artifact stages and manifest temporaries are inert quarantine
objects. They are inventoried by exact name and inode metadata, never decoded
or promoted, and force the publisher to stop. Cleanup is a separate,
duplicate-free, inode-bound `unlink_only` authorization with no glob,
recursive removal, target deletion, or manifest replacement. No cleanup
record exists.

These design boundaries pass.

## 11. Repository state and no-authorization proof

All three planned canonical artifacts were absent and had zero manifest
entries:

```text
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
data/golden/payne_zero/chapter06/synthesis/
  chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz
data/golden/payne_zero/chapter06/
  chapter06_atmosphere_one_line_cpu.npz
```

Both publisher paths, both publisher tests, both publisher acceptances, both
detached authorizations, both authorization reviews, both postpublication
audits, and the quarantine-cleanup acceptance were absent.

The contract and candidate repeatedly state `NOT AUTHORIZATION`. Source,
data, manifest, artifact, and external-tree bytes were not mutated by this
audit.

## 12. Required forward repair

The next contract candidate must:

1. retain the accepted writer/source/archive facts but remove the atmosphere
   candidate-byte hash `79b20c…` from authoritative contract locks;
2. leave **both** lane candidate-byte hashes unresolved in the final contract;
3. state that the existing atmosphere record `79b20c…` is historical and
   stale because it binds contract `9ee002…`;
4. state that the existing synthesis record `474e318…` is a historical
   rejection, not an absent or accepted gate;
5. retain only the exact planned candidate-byte paths and the requirements
   those future reviews must satisfy;
6. independently accept the final hash of that contract without embedding
   either future acceptance hash;
7. rerun both lane byte reviews against that exact final contract, preserving
   the immutable history of the old atmosphere acceptance and synthesis
   rejection; and
8. let later publisher-implementation reviews, detached authorizations, and
   authorization-record reviews bind both the final contract SHA-256 and the
   appropriate newly accepted candidate-byte SHA-256.

The safe dependency order is:

```text
final contract and its audit
  -> new atmosphere and synthesis candidate-byte reviews
  -> publisher implementations and independent reviews
  -> detached authorization and record review
  -> realized entry and manifest
  -> postpublication audit
```

This leaves the valid `A -> R -> E -> N -> Z` mechanism intact while removing
the contract/candidate-acceptance recursion.

## 13. Verification summary

Independent checks completed:

- both assigned candidate hashes;
- all nineteen explicit path/hash table rows;
- all six additional atmosphere precursor locks;
- all synthesis raw, compact, ownership, and final-archive locks;
- a live synthesis zero-argument build;
- all atmosphere mapping, transport, cache, and archive locks;
- a live atmosphere zero-argument build;
- stale-hash and stale-path literal counts;
- exact eleven-row-per-lane trust path inventory;
- local graph topological sorting and global trust-edge analysis;
- placeholder counts;
- strict manifest decode, order, role, and byte round trip;
- source/data/manifest post-build immutability;
- future-object and canonical-artifact absence;
- candidate-input whitespace and `git diff --check`; and
- pinned Payne Zero commit identity.

## 14. Final disposition

**REJECT** the current contract rebind as a design prerequisite.

The rejection is limited to its stale atmosphere byte-acceptance authority
and the resulting incomplete global trust graph. It does not reject the
repaired synthesis writer chain, either reproducible archive, the lane-role
split, or the contract's publication mechanics.

No publisher implementation, authorization record, canonical fixture,
golden, directory, manifest change, or external write follows from this
audit.

<!-- BEGIN PRESERVED FORWARD-CYCLE REPAIR RE-AUDIT -->

---

# Chapter 6 publisher-contract forward-cycle repair — independent re-audit

Date: 2026-07-30  
Scope: independent read-only design-prerequisite re-audit  
Disposition: **ACCEPT**

This delimited append preserves the complete rejection above. The first
23,654 bytes and 506 lines remain byte-identical to the rejected audit at
SHA-256
`33493d0abf0a61ed6a7926f4a49ea6b932960b290a3ddbfa9384f7e7a6f7450d`.
The decision below applies only to the later forward-cycle repair and does not
reinterpret the rejected `d52d8651…` contract.

## R1. Decision

**ACCEPT** the repaired publisher contract at SHA-256
`a663369c3851d89468a41436b8faddeba9d3dcbeba79a7254037734f4a5b3666`
as a Chapter 6 design prerequisite only.

The prior blocker is closed:

1. the old atmosphere byte review `79b20c…` is explicitly stale because it
   binds contract `9ee002…`;
2. the synthesis byte record `474e318…` is explicitly a historical rejection;
3. neither record is an accepted input to the repaired contract;
4. both future candidate-byte acceptance hashes are absent and unresolved;
5. the final contract is frozen before either future byte re-audit;
6. each future `C` binds the final contract `D`, while `D` contains only the
   trust-object path and review rules for `C`; and
7. both future `C` gates must close before either publisher implementation may
   begin.

This acceptance does not accept either future candidate-byte review, a
publisher implementation, publisher tests, publisher review, detached
authorization, authorization-record review, canonical artifact, manifest
transition, cleanup record, or postpublication audit. It is not publication
authorization.

## R2. Exact reviewed inputs

The two assigned repair inputs were regular nonsymlink files and independently
matched:

| reviewed object | bytes | lines | SHA-256 |
| --- | ---: | ---: | --- |
| `design/chapter06_lane_artifact_publisher_contract.md` | `54,997` | `955` | `a663369c3851d89468a41436b8faddeba9d3dcbeba79a7254037734f4a5b3666` |
| `design/chapter06_lane_artifact_publisher_contract_rebind_candidate.md` | `24,211` | `571` | `2fc24a7161916bfbc709e261d74c34be3bc754d15d967377f6c620fac7d478d4` |

The repair report explicitly declares Sections 1–11 to be immutable history
of the rejected first rebind. Its Section 12 is the current candidate record
and accurately describes the repaired contract. The earlier contradictory
carry-forward wording is therefore preserved rejection history, not a current
gate assertion.

Both input files ended in one newline. Neither was modified by this audit.

## R3. All live path/hash locks

### R3.1 Twenty contract rows

All twenty path/hash rows in the repaired contract matched live regular
nonsymlink files. The first eighteen rows are authoritative accepted
writer-chain inputs. The final two exist only as explicitly historical
candidate-byte records.

| role | live object | independently recomputed SHA-256 |
| --- | --- | --- |
| synthesis phase-1 plan | `design/chapter06_synthesis_fixture_oracle_plan.md` | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| synthesis phase-1 candidate | `design/chapter06_synthesis_plan_rebind_candidate.md` | `dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93` |
| synthesis phase-1 audit | `design/chapter06_synthesis_plan_rebind_independent_audit.md` | `9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7` |
| synthesis worker | `scripts/chapter06_synthesis_oracle_worker.py` | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| synthesis worker tests | `tests/test_chapter06_synthesis_oracle_worker.py` | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| synthesis worker audit | `design/chapter06_synthesis_worker_independent_audit.md` | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |
| synthesis phase-2 assembler | `scripts/chapter06_synthesis_compact_assembler.py` | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| synthesis assembler tests | `tests/test_chapter06_synthesis_compact_assembler.py` | `25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153` |
| synthesis phase-2 candidate | `design/chapter06_synthesis_compact_rebind_candidate.md` | `54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c` |
| synthesis phase-2 audit | `design/chapter06_synthesis_compact_rebind_independent_audit.md` | `739854db2b5c4c0c0fe5e9db71d8a52958ce401ded7e7a80a8ab90e15172ddcb` |
| synthesis phase-3 writer | `scripts/chapter06_synthesis_compact_writer.py` | `57aa7147afee4a7366cb2a075715d3607fa20507c23c07ec978b0698368ae47b` |
| synthesis writer tests | `tests/test_chapter06_synthesis_compact_writer.py` | `7c41a74f9d2e38a23d988c990af4040ac262a8066cb3cd9feae4e29f0bdc0a4e` |
| synthesis phase-3 candidate | `design/chapter06_synthesis_writer_rebind_candidate.md` | `6ab1f346a409b0302550a0923c35b71a84d6b2899f2c356070c8d76aa8145e5a` |
| synthesis phase-3 audit | `design/chapter06_synthesis_writer_rebind_independent_audit.md` | `467fdc810f14302dba80f0dd18ba34239dfedb7579b48280899f6f9b6e3b3653` |
| atmosphere writer | `scripts/chapter06_atmosphere_fixture_writer.py` | `0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538` |
| atmosphere writer tests | `tests/test_chapter06_atmosphere_fixture_writer.py` | `c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a` |
| atmosphere writer candidate | `design/chapter06_atmosphere_fixture_writer_candidate.md` | `e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee` |
| atmosphere writer audit | `design/chapter06_atmosphere_fixture_writer_independent_audit.md` | `b946dcef0beeacf49a3da9ac036e21af7cd7b44d15092639cd3be744fb42f0f9` |
| stale atmosphere byte review | `design/chapter06_atmosphere_fixture_byte_acceptance.md` | `79b20c065f5f1b588c6796d6413a1b2aa5e1e5a60056968fc7913cd961e12fc6` |
| rejected synthesis byte review | `design/chapter06_synthesis_candidate_byte_acceptance.md` | `474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1` |

### R3.2 Atmosphere scientific precursors

The six additional atmosphere scientific locks also matched:

| live object | independently recomputed SHA-256 |
| --- | --- |
| `design/chapter06_atmosphere_fixture_oracle_plan.md` | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| `scripts/chapter06_atmosphere_fixture_worker.py` | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| `tests/test_chapter06_atmosphere_fixture_worker.py` | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| `design/chapter06_atmosphere_fixture_worker_independent_audit.md` | `336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e` |
| `scripts/chapter06_atmosphere_line_converter.py` | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| `design/chapter06_atmosphere_converter_independent_audit.md` | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |

## R4. Independently reproduced archive boundaries

### R4.1 Synthesis

Independent controlled zero-argument builds, including two fresh A/B child
captures per build, reproduced:

| property | exact result |
| --- | --- |
| raw members A/B | `754` / `754` |
| raw bitwise equality | `true` |
| raw mapping digest | `09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c` |
| raw transport bytes | `8,689,108` |
| raw transport SHA-256 | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` |
| raw schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| raw physical fingerprint | `51371e5c0db1fae7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| raw full fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |
| ownership partition | `250 final`, `123 derived_digest_only`, `381 intentionally_ephemeral` |
| raw ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| compact members | `213` |
| compact array bytes | `1,235,275` |
| compact schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| compact payload fingerprint | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| final archive bytes | `1,294,865` |
| final archive SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| final A/B byte equality | `true` |
| NPY member version | `2.0` |

The repaired four-member provenance change is therefore reflected in the
accepted archive hash while the physical and remaining compact payload locks
stay exact. Capture PIDs, origin tokens, and external cache roots were
distinct. Both caches were initially and finally empty, nonsymlink, external,
distinct, and disposed.

The builder returned `false` for publication authorization, golden
publication, manifest mutation, and artifact-file writing.

### R4.2 Atmosphere

An independent controlled zero-argument build with fresh A/B children
reproduced:

| property | exact result |
| --- | --- |
| scientific members A/B | `19` / `19` |
| scientific array bytes | `357,984` |
| fixture mapping digest | `f533e3e327c879b1d367a89822bcd1847b15a73ba3d234a9c577c81997f75e0a` |
| fixture schema digest | `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698` |
| payload fingerprint | `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663` |
| evidence members A/B | `89` / `89` |
| evidence mapping digest | `f5e49f9f7f49c4604c08a53def65a9ee3b6ddff37376130eb274646ee375af4f` |
| complete transport bytes | `33,147,771` |
| complete transport SHA-256 | `0523ed254a78edaa07480bc30f23082f535e885f46d867de10f92cba1acd5b16` |
| final archive bytes | `363,050` |
| final archive SHA-256 | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |
| fixture/evidence/transport/archive A/B equality | `true` |
| NPY member version | `2.0` |

The child PIDs, origins, and cache roots were distinct. Each initially empty,
external, nonsymlink cache contained exactly 37 entries after its child:
one package directory, eighteen `.nbi` files, and eighteen `.nbc` files.
Both roots were disposed before return.

The builder returned `false` for publication authorization, fixture
publication, golden publication, manifest mutation, and artifact-file
writing.

## R5. Stale history and unresolved future gates

The repaired contract contains exactly one full occurrence of each:

```text
stale atmosphere review
  79b20c065f5f1b588c6796d6413a1b2aa5e1e5a60056968fc7913cd961e12fc6

historical synthesis rejection
  474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1

old publisher contract
  9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774
```

Their only occurrences are in rows and prose explicitly labelled
`historical non-authoritative`, `STALE`, or rejected. The contract states
that neither is authoritative or accepted input.

Direct inspection of the full `79b20c…` file confirms that it binds
`9ee002…` and says any publisher-contract change invalidates its acceptance.
Direct inspection of `474e318…` confirms disposition `REJECT` and that it
accepts no repaired synthesis bytes. The repair now honors both records
without rewriting either.

All seventeen repaired synthesis values occur exactly once in the contract.
All twelve old synthesis hashes and both old writer-candidate/audit paths
occur zero times in the contract and once in the candidate report's explicitly
non-authoritative history.

Both candidate-byte paths remain fixed, but their future accepted hashes are
absent. The contract says `UNRESOLVED` at the global status, each lane
boundary, the trust-object table, the order gate, and the checklist. No
publisher path/hash is treated as resolved.

## R6. Dependency graph

The normative graph is:

```text
D -> C
accepted writers/sources -> C

(publisher implementation, D, C) -> P
(D, C, P, M, T) -> A

A -> R
(T, SHA256(A), SHA256(R)) -> E
(M, E) -> N
(A, R, E, N, artifact bytes) -> Z
```

An independent topological sort consumed every node. One valid order was:

```text
D, publisher implementation, M, sources, T, writers,
C, P, A, R, E, N, Z
```

There is no `C -> D` edge. `D` contains no future `SHA256(C)`. There is no
contract/byte-review recursion and no edge from `P`, `A`, `R`, `E`, `N`, or
`Z` back into `C` or `D`.

The two exact late-bound strings

```text
__LATE_BOUND_AUTHORIZATION_SHA256__
__LATE_BOUND_RECORD_REVIEW_SHA256__
```

occur exactly once each and only as the complete template values of
`publication_acceptance_sha256` and
`publication_record_review_sha256`. `A` binds the placeholder-bearing
template and its digest. `R` binds `SHA256(A)` and the same template digest,
but neither a review self-hash nor any realized-state hash. `E` is realized
only after both are final. Neither `A` nor `R` contains `SHA256(E)`,
`SHA256(N)`, or `SHA256(Z)`.

The established `A -> R -> E -> N -> Z` late-binding chain is therefore
strictly forward. Both future `C` reviews must accept the exact final `D`, and
both must be final before either publisher implementation may begin.

## R7. Trust paths, lane roles, and destinations

The contract contains exactly eleven planned trust-object rows for each lane.
Every identity is an ASCII POSIX repository-relative path with no leading
slash, empty component, `.`, `..`, backslash, percent escape, Unicode/case
alias, or caller-selected root.

The first four rows per lane exist at their accepted exact hashes. The two
candidate-byte paths exist only as the stale atmosphere history and rejected
synthesis history. Every publisher, publisher-test, publisher-acceptance,
detached-authorization, authorization-review, and postpublication path is
absent.

The exact role and destination split is:

| lane | eventual role | canonical destination |
| --- | --- | --- |
| atmosphere | `fixture` | `data/fixtures/chapter06_atmosphere_one_line_inputs.npz` |
| synthesis | `golden` | `data/golden/payne_zero/chapter06/synthesis/chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz` |

The atmosphere archive is input-only and contains no line output. The
synthesis archive is comparison-only and contains no copied input or
atmosphere-lane state. There is no stored synthesis fixture. Static tables and
the one-row subset retain their roles. The later atmosphere comparison golden
remains outside this lifecycle.

Repository identity strings remain distinct from absolute host access paths.
Access begins only at
`/Users/ysting/stellar-spectroscopy-from-scratch-gpu`, walks fixed literal
components with no-follow directory-relative operations, retains
device/inode evidence, and rejects alternate roots or destinations. External
Payne Zero paths are separately typed source identities and cannot become
manifest paths.

## R8. Manifest, durability, and publication mechanics

Strict duplicate-rejecting parsing of the live manifest reproduced:

| property | exact result |
| --- | --- |
| path | `data/MANIFEST.json` |
| bytes | `1,087,741` |
| SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| top-level order | `schema_version`, `payne_zero_commit`, `entries` |
| entries | `37`, all unique |
| ordered-path digest | `d63aa6dfd9209f21172c6ccf721cffc895835b8ee15a4ed160c5bebe0851b1aa` |
| path order | intentionally unsorted |
| roles | `6 fixture`, `12 golden`, `18 static`, `1 subset` |

The exact bytes round-tripped under the specified `sort_keys=False`,
two-space-indented, UTF-8, finite-value encoder with one final newline.
Appending a probe as the final entry and then deleting that last entry under
the same parser/encoder recovered the exact original bytes and SHA-256.

The contract freezes existing entry and object-key order, lexical archive
member order in `T`, exact per-array key order, append-only insertion, and a
structural diff of exactly the two placeholder values. It does not call the
live manifest canonical or path-sorted.

The path and durability design is complete:

- new allowlisted directories are created one literal component at a time
  after authorization;
- each new directory is opened no-follow, recorded, `fsync`ed, followed by
  immediate-parent `fsync`, then reopened at the same device/inode;
- `chapter06` must complete before `synthesis`;
- handled directory cleanup is inode-bound, nonrecursive, and followed by
  parent `fsync`;
- the data lock attaches to the stable canonical `data` directory;
- staging is create-exclusive, no-follow, same-filesystem, descriptor-read
  back, and file-`fsync`ed;
- installation uses an independently tested atomic create-if-absent primitive,
  never plain rename, replacement, pre-delete, or check-then-overwrite;
- the no-replace syscall decides target races;
- the destination parent is `fsync`ed after installation and after
  inode-verified stage removal; and
- the manifest temporary is create-exclusive, read back, parsed, `fsync`ed,
  and replaced only after immediate identity rechecks, followed by `data`
  directory `fsync` and fresh-process validation.

The exact-existing state is a validation-only no-op. An exact unregistered
artifact is recoverable only under the same authorization and unchanged
prepublication manifest. Nonexact, registered-missing, changed-manifest, or
unexpected states fail closed.

## R9. Snapshots, cache, quarantine, and TOCTOU

The four closed `data/` inventories reproduced:

| inventory fact | exact result |
| --- | --- |
| recursive directories | `12` |
| recursive regular files | `39` |
| regular-file bytes | `30,046,405` |
| symlink/special nodes | `0` |
| exact manifest/file joins | `37` |
| normal nonmanifest support | only `data/README.md` |
| `data/README.md` SHA-256 | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |

Every manifest path joined to one regular nonsymlink file with matching byte
size and SHA-256. No suspicious hidden staging or manifest-temporary file was
present.

The lane cache policies remain deliberately asymmetric and match the live
writers: synthesis requires empty-after disposable roots, while atmosphere
requires exactly one package directory plus eighteen `.nbi` and eighteen
`.nbc` files per child before disposal. Paths, PIDs, origins, and cache
contents do not enter scientific candidate bytes.

Crash-left stages and manifest temporaries are inert inode-bound quarantine
objects. They hard-stop the publisher and are never decoded, promoted, or
used as inputs. Cleanup requires a separate duplicate-free JSON authorization
that binds exact name and inode metadata with `unlink_only`; no cleanup record
exists. Glob deletion, prefix trust, recursive cleanup, target deletion, and
manifest replacement are forbidden.

The TOCTOU design combines retained no-follow directory descriptors, stable
device/inode checks, the data-directory lock, immediate source/manifest/
authorization/candidate hash rechecks, atomic no-replace installation, atomic
manifest replacement, and fresh no-follow readback. The contract accurately
limits these controls to cooperative races rather than claiming a hostile-
account security boundary.

## R10. No-publication state and verification

The pinned Payne Zero checkout remained at commit
`9c44001feae40b85146630499e6f8a5fed42e5af`.

All three planned canonical artifacts were absent and had zero manifest
entries:

```text
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
data/golden/payne_zero/chapter06/synthesis/
  chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz
data/golden/payne_zero/chapter06/
  chapter06_atmosphere_one_line_cpu.npz
```

Both publisher implementations, both publisher tests, both publisher
acceptances, both detached authorizations, both authorization reviews, both
postpublication audits, and the quarantine-cleanup record were absent.

Independent checks included:

- exact hashes, regular-file types, and nonsymlink state for all twenty
  contract rows and six atmosphere precursors;
- all seventeen repaired synthesis literal counts and all historical
  synthesis hash/path absence checks;
- full stale/rejected history inspection and exact counts;
- live synthesis and atmosphere zero-argument builds;
- the complete synthesis and atmosphere writer suites: `20 passed in
  45.44s`;
- exact eleven-row-per-lane path extraction and path-syntax checks;
- graph topological sorting and explicit backward-edge rejection;
- exact placeholder counts;
- strict manifest parsing, order, role counts, path digest, byte round trip,
  append/delete-last reconstruction, and filesystem join;
- closed data inventory and canonical/future-object absence;
- pinned source commit identity;
- input and output newline checks; and
- scoped `git diff --check`.

The live writers, tests, and this audit wrote no publisher, artifact, fixture,
golden, manifest, authorization, or external source. The only authorized
repository mutation is this preserved audit append.

## R11. Residual gates and final disposition

The design prerequisite is accepted, but the lifecycle is intentionally still
closed:

1. append and independently accept a new atmosphere candidate-byte re-audit
   against exact contract `a663369c…`, preserving `79b20c…`;
2. append and independently accept a new synthesis candidate-byte re-audit
   against exact contract `a663369c…`, preserving `474e318…`;
3. require both re-audits to be final before either publisher implementation;
4. independently review each fixed-path publisher and adversarial suite;
5. create and independently review detached authorization against the exact
   phase-specific prepublication manifest; and
6. only then permit no-replace installation and append-only registration,
   atmosphere first and synthesis second.

**Final disposition: ACCEPT the repaired contract as a design prerequisite
only. NOT CANDIDATE-BYTE ACCEPTANCE. NOT PUBLISHER ACCEPTANCE. NOT
AUTHORIZATION. NO DATA OR MANIFEST WRITE IS GRANTED.**

<!-- END PRESERVED FORWARD-CYCLE REPAIR RE-AUDIT -->

<!-- BEGIN PHYSICAL-FINGERPRINT TYPO-REPAIR INDEPENDENT RE-AUDIT 2026-07-30 -->

# Chapter 6 publisher-contract physical-fingerprint repair re-audit

Status: **ACCEPT AS A DESIGN PREREQUISITE ONLY — NOT CANDIDATE-BYTE
ACCEPTANCE, PUBLISHER ACCEPTANCE, OR AUTHORIZATION**

This is an append-only independent review of the one-field transcription
repair. The preceding `45,732` bytes and `949` lines remain an immutable
prefix with SHA-256
`c4a4ca58d94ec71ec509238046afcb127e189ba0be98a96c9929488958a1c286`.
Nothing in the prior rejection and forward-cycle-repair history was edited or
reinterpreted.

## T1. Decision

**ACCEPT** contract
`design/chapter06_lane_artifact_publisher_contract.md`, SHA-256
`3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b`,
as the shared two-lane publication-design prerequisite only.

The transition from contract `a663369c…` is exactly the deletion of one
erroneous `e` from its 65-character synthesis raw physical fingerprint. The
corrected 64-character value matches the accepted assembler, writer audit,
prior contract audit, synthesis-byte rejection evidence, and a new live
zero-argument synthesis build. There is no second byte or semantic change.

Both candidate-byte gates remain unresolved. This acceptance grants no
publisher implementation, detached authorization, fixture, golden, data
write, manifest entry, cleanup record, or publication permission.

## T2. Exact bound inputs and preserved histories

| bound object | exact SHA-256 |
| --- | --- |
| corrected publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| appended typo-repair candidate report | `72d6c2b2dd27d499db68c3c3fb71260a50e6dacf9decdc3b1843d431c2cf7f3b` |
| previous publisher contract | `a663369c3851d89468a41436b8faddeba9d3dcbeba79a7254037734f4a5b3666` |
| preserved prior contract audit prefix | `c4a4ca58d94ec71ec509238046afcb127e189ba0be98a96c9929488958a1c286` |
| current atmosphere candidate-byte history | `cd9be49b2436fc34a33275c54369c549ad6be40aeba2f713bdd033f028156669` |
| current synthesis candidate-byte rejection | `df3714892cf60bb54d22743dcc444157246b8392a3ea907c039af1c196528c55` |

The candidate report preserves its Sections 1–12 prefix exactly: `24,211`
bytes and `571` lines, SHA-256
`2fc24a7161916bfbc709e261d74c34be3bc754d15d967377f6c620fac7d478d4`.
Its Section 13 append is therefore reviewable as a forward repair rather than
a replacement of the earlier design history.

The current atmosphere file preserves the exact `28,077`-byte historical
prefix at SHA-256
`79b20c065f5f1b588c6796d6413a1b2aa5e1e5a60056968fc7913cd961e12fc6`.
Its later acceptance bound contract `a663369c…` and is stale after this
contract repair.

The current synthesis file preserves the exact `15,792`-byte historical
prefix at SHA-256
`474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1`.
Its later section rejected contract `a663369c…` because of the fingerprint
defect repaired here; it does not become an acceptance automatically.

All six bound inputs were regular nonsymlink files at the stated identities
before and after the audit.

## T3. Exact old-to-new byte proof

The wrong and corrected literals are:

```text
wrong (65 characters)
51371e5c0db1fae7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc

right (64 characters)
51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc
```

Independent binary comparison established:

| property | exact result |
| --- | --- |
| old contract bytes | `54,997` |
| old contract lines | `955` |
| reconstructed old SHA-256 | `a663369c3851d89468a41436b8faddeba9d3dcbeba79a7254037734f4a5b3666` |
| corrected contract bytes | `54,996` |
| corrected contract lines | `955` |
| corrected contract SHA-256 | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| non-equal binary opcodes | one deletion |
| deleted old byte | ASCII `e` |
| zero-based byte offset | `7,175` |
| source location | line `106`, column `41` |

Inserting that one `e` into the corrected bytes reconstructs the exact old
contract hash. Bytes before offset `7,175` are equal; old bytes after the
deleted `e` equal new bytes from that offset to EOF. Newline count is
unchanged. The wrong literal occurs zero times in the corrected contract and
the right literal occurs exactly once.

This proves the repair is one deletion, not an unconstrained semantic diff or
silent lock refresh.

## T4. Hash syntax and live lock validation

A scan of every maximal lowercase hexadecimal run of at least 32 characters
in the corrected contract found:

| literal class | count | result |
| --- | ---: | --- |
| lowercase 64-hex hashes/fingerprints | `43` | all exact syntax |
| pinned Git commit | `1` | exact lowercase 40-hex |
| any other long-hex length | `0` | none |

The unique 40-hex literal is the pinned Payne Zero commit
`9c44001feae40b85146630499e6f8a5fed42e5af`, which matched the live checkout.
There is no remaining 65-hex or other malformed accepted hash-like literal.

I independently hashed all 24 active accepted writer-chain files:

- fourteen synthesis plan/worker/assembler/writer candidate-and-audit locks;
- the atmosphere plan, worker, worker tests, and worker audit;
- the atmosphere converter and converter audit; and
- the atmosphere writer, writer tests, candidate, and writer audit.

Every file was regular, nonsymlink, and equal to its exact contract lock. The
two historical candidate-byte table rows intentionally do not equal their
current full files: each row freezes the explicitly non-authoritative prefix
above. Their current full hashes are the stale/rejected histories bound in
Section T2, not future authoritative `C` hashes.

## T5. Independent live synthesis reproduction

A controlled fresh top-level process called
`build_deterministic_compact_archive()` with zero arguments. The accepted
writer internally owned two fresh child captures with distinct origin tokens,
PIDs, and external cache roots. Both cache roots were nonsymlink, empty before
and after capture, and disposed before return.

| live property | independently reproduced result |
| --- | --- |
| raw members | `754` |
| raw transport bytes | `8,689,108` |
| raw transport SHA-256 | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` |
| raw schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| raw physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| raw physical-fingerprint length | `64` |
| raw full fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |
| final archive bytes | `1,294,865` |
| final archive SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |

Raw A and B were exact by member set, dtype, shape, C bytes, mapping digest,
and complete transport bytes. The live 64-character physical fingerprint is
also the exact constant consumed by the compact assembler and recorded by the
writer rebind audit. No candidate or artifact file was written.

## T6. Candidate-byte gates and dependency graph

The corrected contract keeps the only cycle-free direction:

```text
D -> C
(D, C, publisher implementation) -> P
(D, C, P, M, T) -> A
A -> R
(T, SHA256(A), SHA256(R)) -> E
(M, E) -> N
(A, R, E, N, artifact bytes) -> Z
```

There is no `C -> D` edge and no future `SHA256(C)` in `D`. The contract
contains only each candidate-byte path, the stale atmosphere prefix
`79b20c…`, and the historical synthesis prefix `474e318…`, each full prefix
hash occurring once. The current full atmosphere hash `cd9be49b…` and
synthesis hash `df371489…` do not occur in the contract.

Both candidate-byte rows are explicitly **UNRESOLVED**:

1. rerun atmosphere against corrected contract `3a064f82…`, preserving stale
   history `cd9be49b…`;
2. rerun synthesis against corrected contract `3a064f82…`, preserving
   rejection `df371489…`;
3. require both final `C` acceptances before either publisher implementation.

The later graph remains strictly forward. Neither `A` nor `R` contains a
self, realized-entry, postmanifest, or postpublication-audit hash. The exact
placeholders
`__LATE_BOUND_AUTHORIZATION_SHA256__` and
`__LATE_BOUND_RECORD_REVIEW_SHA256__` each occur once. Only their complete
values are substituted to realize `E`; only the postpublication audit freezes
the realized entry and manifest.

## T7. Paths, roles, and absent control objects

The Section 3.1 table still contains exactly eleven POSIX
repository-relative trust-object paths per lane. All 22 are unique, ASCII,
slash-separated, nonabsolute, and free of empty, `.`, `..`, backslash,
percent-escape, case/Unicode alias, or caller-root interpretation.

The ten existing writer/candidate-byte paths are regular nonsymlink files.
All twelve future publisher, publisher-test, publisher-audit,
detached-authorization, authorization-review, and postpublication-audit paths
are absent. The separately named shared contract path is exact and is not a
twelfth per-lane row.

The role boundary remains:

| lane | exact destination | manifest role |
| --- | --- | --- |
| atmosphere | `data/fixtures/chapter06_atmosphere_one_line_inputs.npz` | `fixture` |
| synthesis | `data/golden/payne_zero/chapter06/synthesis/chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz` | `golden` |

The atmosphere archive remains input-only and has no line output. The
synthesis archive remains comparison-only and contains no copied input state.
Static tables and the Fe I subset retain roles `static` and `subset`; no lane
may relabel or copy them.

Repository-relative identity and canonical absolute host access remain
separate types. Access starts only from the fixed repository root and uses
component-wise no-follow operations; no publisher API accepts an alternate
root or destination.

## T8. Manifest byte policy and closed data snapshot

Strict duplicate-rejecting parsing of the live manifest reproduced:

| property | exact result |
| --- | --- |
| path | `data/MANIFEST.json` |
| bytes | `1,087,741` |
| SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| top-level order | `schema_version`, `payne_zero_commit`, `entries` |
| entries | `37`, all unique |
| path order | intentionally unsorted |
| ordered-path digest | `d63aa6dfd9209f21172c6ccf721cffc895835b8ee15a4ed160c5bebe0851b1aa` |
| roles | `6 fixture`, `12 golden`, `18 static`, `1 subset` |

The exact bytes round-tripped under the contract's UTF-8,
`sort_keys=False`, two-space-indented, finite-value encoder with one final
newline. Appending a probe as the final entry and deleting that last entry
reproduced the exact original bytes and hash. Every manifest path joined one
regular nonsymlink file with matching bytes and SHA-256.

The complete `data/` inventory remained:

| inventory | exact result |
| --- | --- |
| recursive directories | `12` |
| recursive regular files | `39` |
| regular-file bytes | `30,046,405` |
| symlink/special nodes | `0` |
| exact manifest/file joins | `37` |
| normal nonmanifest support | only `data/README.md` |
| `data/README.md` SHA-256 | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |

The atmosphere fixture, synthesis golden, and later atmosphere golden
destinations were absent and had no manifest entries.

## T9. Publication mechanics and failure design

Independent rereading confirmed that the single-byte repair leaves all
publication mechanics unchanged:

- nested directories are created only after authorization, one literal
  component at a time, with retained no-follow descriptors, identity
  revalidation, and `fsync` of each new directory and its immediate parent;
- staging is create-exclusive, same-filesystem, mode `0600`, descriptor-read
  back, byte/semantic validated, and file-`fsync`ed;
- installation uses an independently tested atomic create-if-absent
  primitive; plain rename, replacement, pre-delete, and check-then-overwrite
  are forbidden;
- the data-directory lock, device/inode rechecks, no-replace syscall, parent
  `fsync`, and fresh no-follow readback close cooperative races;
- the manifest preserves the existing unsorted entry and object-key order,
  appends exactly one final entry, changes only the two placeholder values,
  proves delete-last reconstruction, and atomically replaces only the
  manifest after complete intended-delta validation;
- synthesis requires empty-after disposable caches, while atmosphere requires
  its accepted one-package-directory plus 18 `.nbi` and 18 `.nbc` files per
  child before owned disposal;
- exact-existing is validation-only; exact unregistered recovery requires the
  same authorization and unchanged prepublication manifest; nonexact,
  registered-missing, entry/file-mismatch, and changed-manifest states fail
  closed;
- crash-left stages and manifest temporaries are inert inode-bound quarantine
  objects, never decoded or promoted; cleanup requires the separately
  reviewed exact `unlink_only` record and forbids glob, prefix, recursive,
  target, or manifest deletion; and
- every dry run, failure, and successful phase repeats the four closed
  source/data inventories and permits only its exact declared delta.

The fixed phase order remains atmosphere \(M_0 \rightarrow M_1\), then
synthesis \(M_1 \rightarrow M_2\). Cross-lane atomicity is not claimed;
readers consume only exact registered artifacts.

## T10. Immutability and no-publication state

The 28 non-audit bound local files—24 active live locks plus the corrected
contract, repair candidate, atmosphere history, and synthesis history—had:

```text
regular files  28
bytes          835,475
aggregate      9583c1915682fa677a017a3922d35bc3f0a881d7f0b8032240871084a37fc5e9
```

The complete 39-file `data/` aggregate remained
`288bbe4c6bcbd20da8390f99fb6cc45e07ee6eed24197371d3520aed39f7d004`.
Each aggregate binds sorted resolved-path length and bytes, file-byte length,
and binary file SHA-256 with little-endian 64-bit lengths.

The Payne Zero checkout remained at commit
`9c44001feae40b85146630499e6f8a5fed42e5af`. The live writer's own identity
gate revalidated its frozen pinned source manifest, atomic source archive,
static tables, subset, and fixture before producing the raw fingerprint.

All canonical Chapter 6 artifacts, both publishers, both publisher tests,
both publisher acceptances, both detached authorizations, both record
reviews, both postpublication audits, and the quarantine-cleanup record
remained absent. The manifest was not modified. No publisher, authorization,
artifact, fixture, golden, data, paper, Payne Zero, or external source write
occurred. The only repository mutation authorized by this audit is this
delimited append.

## T11. Required next gates

The accepted design remains deliberately closed:

1. append a fresh independent atmosphere candidate-byte re-audit against
   exact contract `3a064f82…` and this final contract-audit hash, preserving
   stale file `cd9be49b…`;
2. append a fresh independent synthesis candidate-byte re-audit against the
   same exact contract and final audit, preserving rejection `df371489…`;
3. require both exact byte gates to ACCEPT before authoring either publisher;
4. independently accept each fixed-path publisher and adversarial suite;
5. create and independently review each phase-specific detached
   authorization against its exact prepublication manifest;
6. only then permit no-replace installation and append-only manifest
   registration, atmosphere first and synthesis second.

Any later contract edit invalidates both new byte gates. Any later byte-record
edit invalidates every downstream publisher, authorization, and publication
record that binds it.

## T12. Final disposition

**ACCEPT corrected publisher contract
`3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b`
as a design prerequisite only.**

This accepts the exact one-byte physical-fingerprint transcription repair and
the otherwise unchanged two-lane design. It does not accept either
candidate-byte record, any publisher, detached authorization, canonical
artifact, cleanup action, data role, or manifest mutation.

<!-- END PHYSICAL-FINGERPRINT TYPO-REPAIR INDEPENDENT RE-AUDIT 2026-07-30 -->
