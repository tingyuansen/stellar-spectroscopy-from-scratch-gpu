# Chapter 6 atmosphere-fixture publisher independent audit

Date: 2026-07-30  
Reviewer role: independent adversarial review; no implementation repair  
Disposition: **REJECT the exact publisher candidate**

## 1. Review boundary

This audit reviews the fixed-path atmosphere-fixture publisher against:

- `design/chapter06_lane_artifact_publisher_contract.md`;
- `design/chapter06_atmosphere_fixture_byte_acceptance.md`; and
- the exact publisher, focused tests, and candidate report identified below.

The review was read-only except for this audit record. It did not modify the
publisher candidate, its tests, either contract/acceptance object, detached
authority, canonical data, Payne Zero, or the paper sources. It did not create
the canonical fixture, a publication lock, a staging object, an authorization
record, or a manifest replacement.

This rejection is narrow. It does not withdraw the accepted atmosphere
scientific worker or deterministic writer, and it does not dispute the exact
nineteen-array candidate bytes. It says that this publisher is not yet a safe
implementation of the accepted cross-lane publication protocol.

## 2. Exact reviewed snapshot

Every assigned object was a regular, nonsymlink, single-link file. Exact
SHA-256 identities were:

| reviewed object | SHA-256 |
| --- | --- |
| atmosphere publisher | `a90c4f0324bf87fada2565ff26de6f60a0f4bc3d8728afcab5de5f7937925b48` |
| publisher focused tests | `a947c971385dda96bff6aabf729a92f5567a94c9e7229950f2fe0ce66adacd08` |
| publisher candidate report | `3af044c38108f42bc224248cadb84ec28c45ac8fd21bd7bcfb3e034a059a1c1d` |
| shared publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| atmosphere byte acceptance | `8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71` |
| prepublication manifest | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| data README | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |

The same hashes were re-read after all probes. The canonical destination
remained absent, the canonical manifest remained unchanged, and the proposed
external lock path remained absent.

## 3. Findings

| severity | findings |
| --- | ---: |
| P0 | 1 |
| P1 | 1 |
| P2 | 0 |

Both findings are publication blockers. The P0 finding independently requires
rejection.

### P0-1: the atmosphere lane locks a different object from the shared data lock

The shared contract, Section 9 step 1, requires the exclusive publication lock
to attach to the stable canonical `data` directory for the complete
artifact-plus-manifest phase. Section 12 repeats the stable data-directory
lock requirement, and the required concurrent-lanes case relies on both lanes
contending for that one lock.

The atmosphere publisher instead freezes:

```text
/private/tmp/stellar-spectroscopy-from-scratch-gpu.chapter06-data-publication.lock
```

`_exclusive_data_lock()` creates or opens that external regular file and
applies `flock` to it. It only compares the canonical `data` inode before and
after the critical section; it never locks that inode. The candidate report
explicitly describes this as a fixed lock file outside `data`.

This is not an equivalent implementation. The synthesis publisher and the
shared contract use an exclusive lock on the canonical `data` directory.
Consequently, an atmosphere process and a synthesis process can both believe
they own the shared file-plus-manifest phase.

An independent two-process probe held `_exclusive_data_lock()` and attempted
two nonblocking exclusive locks from child processes:

```text
canonical_data_directory_lock_while_candidate_lock_held=ACQUIRED
same_external_lock_while_candidate_lock_held=BLOCKED
```

The candidate's focused lock test proves only that two users of the same
external lock file exclude one another. It does not test the normative lock
object and cannot detect this cross-lane failure.

The external `O_CREAT` also introduces a persistent write outside the two
canonical mutation targets. More importantly, there is no mutual exclusion
with a contract-conforming lane, so both lanes can start from the same
manifest and enter conflicting artifact/manifest phases.

Required repair: open the canonical `data` directory through the strict
component-wise no-follow path, verify its retained descriptor against the
named canonical inode, and hold `flock(LOCK_EX)` on that descriptor throughout
the complete artifact-plus-manifest phase. Both publishers must use that same
lock object. The external lock file should not be created.

### P1-1: mutation durability reopens pathnames instead of fsyncing retained parents

The implementation retains useful directory descriptors for the two atomic
mutations, but it does not use those descriptors for the immediately required
directory durability operation:

- `_atomic_install_no_replace()` creates the destination hard link through
  its retained destination-parent descriptor, then calls
  `_fsync_directory(parent)`. That helper opens `parent` again by pathname.
- `_replace_manifest_atomically()` replaces `MANIFEST.json` through its
  retained `data` descriptor, then calls
  `_fsync_directory(paths.data_root)`, which again reopens by pathname.

If the named parent is exchanged after the link or replace but before the
pathname reopen, the implementation fsyncs the replacement directory rather
than the directory that actually received the mutation. Its later readback or
inode comparison detects an error, but detection does not retroactively make
the already-mutated parent durable.

Two independent isolated-tree probes forced exactly these transitions. For
the artifact link:

```text
linked_parent_inode=60566062
path_inode_fsynced=60566066
same_inode=False
post_mutation_detection=IdentityError
```

For the manifest replacement:

```text
replaced_parent_inode=60566140
path_inode_fsynced=60566143
same_inode=False
post_mutation_detection=IdentityError
```

In both cases, the candidate failed closed at the API level only after
fsyncing the wrong directory inode. This violates the retained-descriptor and
durability requirements in Sections 9, 10, and 12. The focused
`test_retained_parent_identity_detects_directory_swap` calls only the
standalone comparison helper; it does not drive a swap through either real
mutation sequence and therefore misses the gap.

Required repair: fsync the same retained directory descriptor used for each
link, unlink, and replace. Revalidate the descriptor against the named
canonical directory immediately before each mutation, perform the mutation
relative to that descriptor, and call `os.fsync(retained_fd)` without
re-resolving the pathname. The manifest's pre-replace identity check and
post-replace readback should likewise be tied to that retained `data`
descriptor.

## 4. Gates that did pass

The rejection should not obscure the substantial correct work in this
candidate:

- The production root, destination, manifest, and source identities are
  fixed; the public APIs accept no caller root, destination, force, replace,
  or repair parameter.
- Missing acceptance, authorization, and authorization-review objects stop
  both `--authorized-dry-run` and `--publish` before candidate construction,
  lock creation, parent creation, or staging.
- Repository reads use component-wise directory-relative no-follow opens and
  reject symlinks, aliases, nonregular files, and unexpected hard links.
- Strict JSON rejects duplicate keys and nonfinite values. Authorization,
  review, identity, array-record, manifest, and template key orders are
  closed and checked.
- The real candidate path ran two unrelated accepted top-level writers. The
  archives were byte-identical, the child/cache topology was accepted, and
  the cache roots were disposed.
- The exact candidate is 363,050 bytes at SHA-256
  `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff`.
  It has exactly nineteen lexically ordered members and 357,984 scientific
  array bytes. Decode, scientific validation, and canonical re-encode
  reproduced the complete archive exactly.
- Staging is create-exclusive/no-follow, rejects a short write, fsyncs and
  reads back the file, and validates the stage inode.
- Artifact installation uses hard-link no-replace semantics; `EEXIST`
  preserves and validates the winner. The focused tests cover exact and
  nonexact races plus multi-link rejection.
- The manifest encoder preserves the existing unsorted byte policy. The new
  entry is append-only, its two late authority hashes are narrowly realized,
  and delete-last reconstruction must recover exact prepublication bytes.
- Quarantine, exact-unregistered recovery, manifest-failure recovery, and
  registered-state idempotence are represented in the focused suite.
- No target, manifest, authorization, external source, or data README byte
  changed during candidate verification.

These properties may be retained in a repaired candidate, but they do not
compensate for a disjoint cross-lane lock or fsyncing the wrong parent inode.

## 5. Executed verification

| check | result |
| --- | --- |
| `python -m pytest -q tests/test_chapter06_atmosphere_fixture_publisher.py` | `30 passed, 1 skipped` |
| atmosphere worker and writer focused suites | `26 passed` |
| publisher and test `py_compile` | passed |
| real `--verify-only` CLI | passed; two top-level writers, four fresh scientific children, exact nineteen-member canonical archive, zero repository delta |
| real `--authorized-dry-run` CLI without authority | exit 2 before candidate/lock/stage |
| real `--publish` CLI without authority | exit 2 before candidate/lock/stage |
| independent cross-lock child-process probe | failed contract: canonical data lock was concurrently acquirable |
| independent artifact-parent swap at durability point | failed contract: replacement pathname inode was fsynced |
| independent manifest-parent swap at durability point | failed contract: replacement pathname inode was fsynced |
| post-probe trust/data hashes and destination state | unchanged; destination and external lock absent |

The skipped focused test is the opt-in live reconstruction. The equivalent
real `--verify-only` CLI was run directly and completed successfully.

## 6. Decision

**REJECT** exact publisher
`a90c4f0324bf87fada2565ff26de6f60a0f4bc3d8728afcab5de5f7937925b48`.

No publisher acceptance, detached publication authorization, authorization
review, canonical artifact, or manifest mutation should be issued from this
candidate. A repaired publisher requires a new exact identity, focused tests
that exercise the normative cross-lane lock and the real parent-swap mutation
sequences, a new candidate report, and a fresh independent audit.

## 7. Re-audit of the repaired publisher bytes

Date: 2026-07-30  
Reviewer role: independent adversarial repair re-audit; no implementation repair  
Disposition: **REJECT the exact repaired publisher candidate**

Sections 1–6 above are preserved verbatim as the immutable rejection of
publisher `a90c4f03…`, tests `a947c971…`, and report `3af044c3…`. Their
pre-append audit SHA-256 was
`c46e76a4dacb7a86fd2224ea1c23e95cc2cbca2fe8be7606ae93f660dc11082b`.
This section is a fresh decision about different exact bytes; it neither
rewrites the former evidence nor inherits a disposition from it.

### 7.1 Exact repaired review boundary

Every assigned object was again a regular, nonsymlink, single-link file.
The exact repaired identities were:

| reviewed object | SHA-256 |
| --- | --- |
| atmosphere publisher | `d9cb38e5eacb3dbad66560dad63f13f77ff16f5aa58030b5c53f75a702913dfa` |
| publisher focused tests | `0653f399a3a1c7b5d1600a5bdbe9c9997e6210f0148f6833a973cffdc94a987a` |
| repaired candidate report | `c55b8fd3653e6d89ecea5f1bcbb96b6b0a010452cf123216a02a6320f07de820` |
| shared publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| atmosphere candidate-byte acceptance | `8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71` |
| prepublication manifest | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| data README | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |

The implementation, tests, candidate report, contract, candidate-byte
acceptance, manifest, and README retained those exact hashes after the
re-audit. The production destination, detached authorization,
authorization-record review, postpublication audit, and quarantine-cleanup
authorization remained absent. No production stage or manifest temporary was
created.

### 7.2 Former blockers are repaired

The repaired bytes do close both findings in Section 3:

- `_exclusive_data_lock()` opens the canonical `data` directory by
  component, retains that exact descriptor, and applies `flock(LOCK_EX)` to
  it. The synthesis publisher at exact implementation
  `2a345fd389d2c8f8a97ebef8b4d418a1c81f5b49b94dd94e595ac95cb2e0479e`
  opens and locks the same canonical directory object. In an independent
  process probe, synthesis blocked while atmosphere held the lock and
  acquired it only after atmosphere released it.
- Artifact link, stage unlink, and manifest replacement now use retained
  directory descriptors. Their immediate durability operation calls
  `os.fsync()` on that retained descriptor before rebinding it to the
  canonical name. Forced swaps after the real artifact link and real manifest
  replacement recorded the original mutated inode in each `fsync`, never the
  replacement pathname inode; both invocations then rejected the detached
  canonical name.

Those repairs are necessary and effective. They do not, however, close the
new blocker below.

### 7.3 Repaired-byte finding

| severity | findings |
| --- | ---: |
| P0 | 1 |
| P1 | 0 |
| P2 | 0 |

#### P0-R1: the final mutations re-resolve a staged name after its last inode check

Both mutation sequences validate an invocation-owned staged inode and then
use its name in a later syscall:

- `_atomic_install_no_replace()` closes the descriptor used to create the
  artifact stage. It calls `_validate_stage_at()`, performs retained-directory
  checks, and then calls `os.link(stage_name, target_name, ...)`. The source
  name is resolved again inside `link`.
- `_replace_manifest_atomically()` similarly calls
  `_validate_stage_at()`—including a second call after the final source/data
  snapshot—revalidates `data`, and then calls
  `os.replace(temporary_name, "MANIFEST.json", ...)`. The source name is
  resolved again inside `replace`.

The stage identity check and the mutation are therefore not one
inode-bound operation. A substitution in this last interval is not rejected
before mutation.

Two isolated-tree probes injected substitution at the entry to the real
mutation syscall, after every production pre-mutation validation had
completed:

```text
artifact substitution at link boundary
  final target created: yes
  final target bytes: b"foreign-stage-at-link-boundary"
  later result: IdentityError

manifest-temporary substitution at replace boundary
  original manifest bytes preserved: no
  replacement bytes: b'{"foreign":true}\n'
  later result: ManifestError
```

The later readbacks correctly detect both wrong results, but detection is too
late. The artifact case leaves a foreign unregistered target rather than
rejecting before target creation. More seriously, the manifest case destroys
the pre-existing canonical manifest bytes before reporting failure.

This violates the contract's retained-inode and immediate-pre-mutation
TOCTOU requirements in Sections 10 and 12 and its Section 13 requirement that
every negative case leave all pre-existing canonical file bytes unchanged.
It also directly contradicts the repaired candidate report's stage-
substitution claim that artifact substitution is rejected before target
creation and manifest-temporary substitution preserves the original
manifest.

The focused tests do not cover this interval. Both substitution tests replace
the name in a wrapper around `_create_exclusive_stage()`, so the first later
`_validate_stage_at()` catches the change. They never replace the name after
the last validation but before the real `link` or `replace`.

Required repair: bind each mutation's source to the retained, validated stage
inode at the authoritative syscall boundary, or use another reviewed
transaction design that cannot consume a different name-resolved inode.
Focused tests must substitute at that final boundary and prove that no target
is created and exact original manifest bytes remain. Post-mutation detection
alone is insufficient for manifest safety.

### 7.4 Other repaired gates independently reproduced

The remaining requested shared-transaction matrix passed:

- create-exclusive/no-follow staging, exact `0600` mode independent of umask,
  one exact write, file `fsync`, descriptor readback, same-device checks,
  exact inode/mode/size/hash/byte validation, and identity-bound cleanup;
- early artifact-stage and manifest-temporary substitution rejected before
  target/replacement, while the foreign substituted objects remained
  untouched for quarantine;
- no-replace target races, exact-existing behavior, multilink rejection,
  manifest races, append-only encoding, late-hash realization, manifest
  failure recovery, exact-unregistered recovery, and idempotent registered
  no-op behavior;
- exact unregistered-target normalization reproduced the absent-target
  aggregate digest, while an unexpected empty directory failed the closed
  directory inventory;
- the array hash was exactly SHA-256 over contiguous C-order bytes only. For
  little-endian float64 `[1.0, 2.0]`, the independently computed and returned
  value was
  `dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae`;
- all nineteen members have exact nonempty scientific unit/convention
  strings, dimension-matched axes, lexical template order, and six-key array
  records;
- source and normalized closed-data snapshots are repeated after artifact
  installation and again after manifest staging. Injected source or nontarget
  data changes stopped before manifest replacement and cleaned only the
  invocation-owned temporary;
- a separate interpreter ran the complete internal registered-state validator
  while a parent retained the canonical data-directory lock; that child also
  independently observed the lock as blocked;
- verification-only has no call edge to lock, stage, link, unlink, or manifest
  replacement; absent authority makes both authorized modes stop before
  candidate construction, lock, parent, or stage creation; and
- the public APIs and CLI expose no root, destination, force, replace, repair,
  merge, cleanup, or partial-lane option.

### 7.5 Executed repair verification

| check | result |
| --- | --- |
| repaired publisher focused suite | `47 passed, 1 skipped` |
| atmosphere converter, worker, and writer suites | `41 passed` |
| combined default atmosphere chain | `88 passed, 1 skipped` |
| independent atmosphere-versus-synthesis process lock | synthesis blocked, then acquired only after atmosphere release |
| independent forced artifact-parent swap | exact mutated retained inode fsynced; replacement inode not fsynced; rejected |
| independent forced data-parent swap after manifest replacement | exact mutated retained inode fsynced; replacement inode not fsynced; rejected |
| early artifact-stage / manifest-temporary substitution | rejected before canonical mutation; foreign name preserved |
| final-boundary artifact-stage substitution | **failed contract: foreign target created before rejection** |
| final-boundary manifest-temporary substitution | **failed contract: pre-existing manifest bytes replaced before rejection** |
| exact recovery normalization / closed directory inventory | passed / unexpected empty directory rejected |
| complete fresh registered validation in separate process under retained lock | passed; child observed lock blocked |
| real production `--verify-only` | passed; two top-level writers, four scientific children, exact archive, zero canonical data delta |
| real `--authorized-dry-run` and `--publish` without authority | each exited 2 before candidate/lock/parent/stage; zero canonical data delta |
| Ruff check, Ruff format check, and in-memory AST parse | passed |

The live reconstruction returned exactly:

```text
archive bytes                  363050
archive SHA-256                1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
member count                   19
scientific array bytes         357984
top-level writer invocations   2
data snapshot SHA-256          5be325c0200cf711ce13daa3dc96a47bdd8780c8de1640efe23827196de5b84a
manifest SHA-256               d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a
destination                    absent
canonical data delta           zero
```

The default skip was only the opt-in live reconstruction; the equivalent real
CLI reconstruction above was executed directly.

### 7.6 Repaired-byte decision

**REJECT** exact repaired publisher
`d9cb38e5eacb3dbad66560dad63f13f77ff16f5aa58030b5c53f75a702913dfa`.

Do not create a detached publication authorization or authorization-record
review from these bytes, and do not publish the canonical fixture. The
canonical manifest and README remain unchanged, the destination remains
absent, and the former rejection history remains binding only on its former
bytes. A subsequent repair requires a new publisher identity, final-boundary
substitution tests, an updated candidate report, and another exact-byte
independent re-audit appended after this section.

## 8. Scope-aware re-audit under the threat-model adjudication

Date: 2026-07-30  
Reviewer role: independent hostile review within the frozen Section 12 scope  
Disposition: **REJECT the exact scope-corrected publisher candidate**

Section 7 remains the immutable prior decision. Its technically accurate
syscall-entry traces are preserved rather than erased. The design adjudication
below changes their acceptance classification, not their observed results and
not the historical audit bytes. This section therefore performs a fresh
review of the exact implementation under the adjudicated scope and does not
inherit either acceptance or rejection automatically.

### 8.1 Exact scope-aware review boundary

Every reviewed object was a regular, nonsymlink, single-link file:

| reviewed object | SHA-256 |
| --- | --- |
| atmosphere publisher | `d9cb38e5eacb3dbad66560dad63f13f77ff16f5aa58030b5c53f75a702913dfa` |
| publisher focused tests | `0653f399a3a1c7b5d1600a5bdbe9c9997e6210f0148f6833a973cffdc94a987a` |
| scope-corrected candidate report | `ef98c7c5fc197e5bf1ec0e47d4acc9238fa860e6db19329918a021c865a09372` |
| threat-model adjudication | `4ed79dd4ff622abbeb6106a58b1b7e8c05a10c2ff87cfeb0dd6ee06e24e8e9fc` |
| complete audit history before this append | `df58318960008ebdab127d9d54c3a88b2b295307a2a2f6c7c07496510ede14d6` |
| shared publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| atmosphere candidate-byte acceptance | `8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71` |
| prepublication manifest | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| data README | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |

The implementation and tests are the same bytes reviewed in Section 7. The
candidate report is a different, report-only object, and the adjudication is
read-only design evidence with no publication authority.

The detached authorization, authorization-record review, canonical
atmosphere fixture, postpublication audit, and quarantine-cleanup
authorization remained absent throughout. No production stage, temporary,
external lock file, artifact, or manifest entry was created.

### 8.2 Adjudication of Section 7's syscall-entry trace

The Section 7 trace remains a correct demonstration that Darwin's
name-resolved `link` and `replace` syscalls have an interval after the final
userspace check. Under exact adjudication `4ed79dd4…`, arbitrary same-process
code injected at syscall entry—or an unrestricted same-account process that
ignores the canonical `data` lock and wins that same interval—is outside the
contract's scientific reproducibility threat boundary. That trace is
therefore **not** counted as an acceptance blocker in this section.

The implementation still must, and does, retain the in-scope baseline around
that acknowledged interval:

1. both cooperative lanes lock the same canonical `data` directory inode;
2. the stage is create-exclusive/no-follow, mode-fixed, fsynced, read back,
   and immediately rebound by exact name/inode/mode/link/size/hash/bytes;
3. artifact installation is atomic create-if-absent and manifest replacement
   is atomic;
4. the exact retained mutated directory descriptor is fsynced;
5. that descriptor is rebound to the canonical pathname and detached-parent
   swaps reject; and
6. exact postread plus complete fresh validation remain under the common
   lock.

The corrected candidate report is honest about this limitation. Its
ordinary-substitution claim is explicitly limited to changes visible before
the immediate boundary check. Section 11 then states that it does not claim
to prevent, reject-before-mutation, or preserve canonical bytes against
arbitrary same-process or unrestricted same-account substitution inside
`os.link` or `os.replace`. It neither claims a descriptor-bound Darwin
manifest replacement nor silently changes atomic replacement into in-place
writing.

### 8.3 New in-scope finding

| severity | findings |
| --- | ---: |
| P0 | 1 |
| P1 | 0 |
| P2 | 0 |

#### P0-S1: detached authority is not rechecked after manifest staging

The adjudication's own contract summary requires the old manifest, artifact,
authority objects, template, source/data state, and intended manifest to be
rechecked immediately before replacement. This is an ordinary
pre-replacement identity requirement, not the excluded syscall-entry actor.

The publisher rechecks the authorization, record review, and exact target in
`publish()` before it calls `_replace_manifest_atomically()`. That helper then
performs materially later work:

1. create, write, mode-fix, fsync, and read back the manifest temporary;
2. fsync and rebind retained `data`;
3. reread the old manifest;
4. revalidate the temporary;
5. recompute the complete accepted-source snapshot;
6. recompute the normalized closed-data snapshot, including the exact
   unregistered target; and
7. revalidate the temporary and retained `data` again.

It does not receive or reread the authorization and review identities after
that work. `_verify_accepted_source_identities()` deliberately covers the
accepted source chain and optional publisher acceptance, not the detached
authorization or its review. `_snapshot_data()` covers `data/`, not those
two `design/` records. The later `_registered_state_matches()` uses the
already loaded in-memory `Authority`, so it also cannot observe a changed
named record.

An independent isolated-tree probe used the real stage creation, real
source/data resnapshot, real atomic replacement, and strict synthetic
authorization/review. Immediately after the real manifest stage was created,
the probe changed only the named authorization bytes by adding JSON
whitespace. The parsed authorization remained structurally valid, but its
SHA-256 no longer matched the independently reviewed record. This is the same
pre-syscall boundary used by the accepted source/data mutation regressions;
no code was injected inside `os.replace`.

With the complete internal validator forced for the isolated final gate, the
exact result was:

```text
fresh complete result
  AuthorizationError: authorization-record review binding changed

pre-existing manifest changed before rejection
  true

artifact registered before rejection
  true
```

Without replacing the isolated test helper's intentionally shallow fresh
gate, `publish()` returned `installed-and-registered`; a subsequent direct
complete internal validation produced the same authorization-review
rejection. Production's fresh child would therefore detect the stale
authority, but only after the canonical manifest had already been replaced.
Fresh validation is valuable defense in depth; it is not a substitute for
the required immediate authority check before the one allowed manifest
mutation.

A second probe changed both named records coherently after staging: it added
JSON whitespace to `A`, updated `R.authorization_sha256` to the new exact
authorization hash, and rewrote `R`. Publication again replaced the old
manifest; complete validation then rejected
`registered state entry differs from realized E` because the installed entry
still contained the old late-bound `SHA256(A)` and `SHA256(R)`. This proves
that the missing checks cover both authority objects and their template
realization, not merely malformed JSON.

The adjacent artifact and intended-manifest boundaries did pass. Changing
the target bytes after manifest-stage creation made the real closed-data
snapshot reject before replacement, preserve the old manifest, and clean the
owned temporary. The exact intended `N` bytes are likewise rebound after the
final source/data snapshot. Those passing checks narrow this finding to the
named authorization/review identities and the late hashes derived from them.

The focused suite covers a stale authorization before staging and covers
source and non-target-data changes after manifest staging. It has no
regression that changes the detached authorization or record review after
manifest-stage creation. Consequently, all 47 default focused tests pass
while this in-scope boundary remains open.

Required repair: pass the exact loaded authorization and review identities
into the manifest transition and no-follow reread both named files after the
final source/data snapshot and immediately before replacement. Require exact
hash equality, then revalidate the exact intended temporary and retained
`data` boundary before the atomic syscall. Add separate post-staging
authorization and review mutation regressions; each must reject before
replacement, preserve exact old manifest bytes, and remove only the
identity-bound invocation-owned temporary.

### 8.4 In-scope former repairs and contract matrix

Apart from P0-S1, the complete requested matrix passed again:

- the canonical `data` inode is the sole lock object. A real synthesis
  publisher process blocked while atmosphere held it and acquired it only
  after release; injected lock failure preceded every write;
- forced directory swaps after the real artifact link and real manifest
  replacement proved that `fsync` reached the exact retained mutated inode,
  never the replacement pathname inode, and canonical rebind rejected before
  helper success;
- ordinary stage and manifest-temporary substitutions visible before the
  immediate boundary check rejected before target/replacement, preserved the
  old manifest, and left foreign names untouched for quarantine;
- create-exclusive/no-follow staging, one exact write, file fsync,
  descriptor readback, same-device proof, exact `0600` artifact mode
  independent of umask, inherited exact manifest mode, identity-bound
  cleanup, and quarantine hard-stop behavior passed;
- no-replace exact/nonexact/multilink target races, append-last manifest
  encoding, two-field late-hash realization, manifest race rejection,
  partial failure, exact-unregistered recovery, and registered idempotence
  passed;
- exact recovery normalized only the independently fixed target from the
  aggregate; nonexact target bytes and an unexpected empty directory
  rejected the closed lifecycle inventory;
- array hashes are raw contiguous C-order bytes only. Float64 `[1.0, 2.0]`
  independently produced
  `dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae`;
- all nineteen array records retain exact lexical order, six-key schema,
  nonempty scientific units/conventions, and dimension-matched axes;
- source and normalized data are resnapshotted after artifact installation
  and after manifest staging. Source or non-target-data changes at either
  boundary reject before replacement and clean the invocation-owned
  temporary;
- a separate interpreter ran the complete registered-state validator while a
  parent retained the data-directory lock, and the child independently
  observed a competing lock attempt as blocked;
- verification-only is authorization-independent and cannot call lock,
  staging, link, unlink, or manifest-replacement helpers; and
- the fixed Python and CLI surfaces expose no caller root, destination,
  force, replace, repair, merge, cleanup, or partial-lane option.

### 8.5 Executed scope-aware verification

| check | exact result |
| --- | --- |
| atmosphere publisher focused suite | `47 passed, 1 skipped` |
| converter, worker, and writer suites | `41 passed` |
| combined default atmosphere chain | `88 passed, 1 skipped` |
| independent atmosphere-versus-synthesis lock process | blocked, then acquired after release |
| independent retained-inode parent swaps | exact mutated inode fsynced; canonical rebind rejected |
| ordinary pre-boundary stage/temporary substitution | rejected before canonical mutation |
| recovery normalization / closed directory inventory | passed / rogue empty directory rejected |
| complete fresh registered validation in a separate process under retained lock | passed |
| post-staging target-byte mutation | rejected before replacement; old manifest preserved |
| post-staging detached-authorization mutation | **failed contract: manifest replaced before stale review rejection** |
| coherent post-staging authorization/review rebind | **failed contract: manifest replaced before realized-entry rejection** |
| real production `--verify-only` | passed; exact live archive and zero canonical data delta |
| real `--authorized-dry-run` and `--publish` without authority | each exited 2 before candidate/lock/parent/stage; zero delta |
| Ruff check, Ruff format check, and in-memory AST parse | passed |

The default skip was only the opt-in live reconstruction. The equivalent real
CLI was executed directly and returned:

```text
archive bytes                  363050
archive SHA-256                1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
member count                   19
scientific array bytes         357984
data snapshot SHA-256          5be325c0200cf711ce13daa3dc96a47bdd8780c8de1640efe23827196de5b84a
manifest SHA-256               d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a
destination                    absent
canonical data delta           zero
```

### 8.6 Scope-aware decision

**REJECT** exact publisher
`d9cb38e5eacb3dbad66560dad63f13f77ff16f5aa58030b5c53f75a702913dfa`,
focused tests
`0653f399a3a1c7b5d1600a5bdbe9c9997e6210f0148f6833a973cffdc94a987a`,
and corrected report
`ef98c7c5fc197e5bf1ec0e47d4acc9238fa860e6db19329918a021c865a09372`.

The syscall-entry wrapper is not the reason for this decision. It has been
explicitly adjudicated out of scope, and the report now states that limitation
honestly. P0-S1 is a distinct, finite, pre-syscall authority race expressly
inside the frozen transaction: the publisher can replace the manifest after
the detached authorization and review binding have changed and discover the
failure only in postvalidation.

Do not create detached publication authorization or review records from
these publisher bytes, and do not publish the canonical fixture. The
production manifest and README remain unchanged, the destination remains
absent, no quarantine name exists, and all external scientific sources
remain untouched. A repair requires a new implementation and test identity,
an updated candidate report, and another append-only exact-byte independent
review.

## 9. Final exact-byte re-audit of the late-authority repair

Date: 2026-07-30

Reviewer role: independent hostile final QA; no implementation repair

Disposition: **ACCEPT the exact final publisher candidate within the frozen
Section 12 threat boundary**

Sections 1–8 remain immutable audit history. In particular, Section 8 remains
the exact rejection of publisher `d9cb38e5…`, tests `0653f399…`, and report
`ef98c7c5…` for the finite P0-S1 authority gap. This section reviews different
implementation, test, and report bytes. It does not convert the design
adjudication into authority and does not inherit acceptance from any earlier
gate.

### 9.1 Exact final review boundary

Every reviewed object was a regular, nonsymlink, single-link file:

| reviewed object | SHA-256 |
| --- | --- |
| final atmosphere publisher | `5de0169339b8130d45c8e610684de67c2d6f3c6c62c5298610d0f705ebe0b363` |
| final publisher focused tests | `00c09576b00420bfaa72f058a1b70fab1dedb4e85e04326ca83c7bac0b6ca876` |
| final publisher candidate report | `f5f9ae5508c447938b3262cc5a34fd6901fe9519d6e408318ba6370e0e94cfaf` |
| complete prior audit history | `b77de7e2f543ace58f054d08f5ad0f596f0e80decd4a08b5f63ffa0f4680585c` |
| threat-model adjudication | `4ed79dd4ff622abbeb6106a58b1b7e8c05a10c2ff87cfeb0dd6ee06e24e8e9fc` |
| shared publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| atmosphere candidate-byte acceptance | `8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71` |
| prepublication manifest | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| data README | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |

The detached publisher acceptance, publication authorization, independent
authorization-record review, canonical atmosphere fixture, postpublication
audit, and quarantine-cleanup authorization remained absent. This review
created none of them.

### 9.2 P0-S1 is closed at the ordinary pre-syscall boundary

The repaired `_replace_manifest_atomically()` now performs the complete
finite boundary in this order:

1. create-exclusive/no-follow manifest staging, exact write, mode, file
   `fsync`, descriptor readback, retained-directory `fsync`, old-`M` identity,
   stage identity/bytes, and retained-`data` rebind;
2. freshly reconstruct the accepted-source snapshot and normalized closed-data
   snapshot after staging;
3. call `_revalidate_manifest_transition_authority()` with the prepared
   `Authority`, `Candidate`, exact old `M`, intended `N`, and the fresh
   source/data snapshots;
4. no-follow reload and strictly validate the publisher acceptance, detached
   authorization, and review, including exact schemas, key order, all
   bindings, hashes, source/data identities, and equality with the prepared
   in-memory authority;
5. no-follow reread the exact authorization and review bytes/metadata a second
   time after that validation;
6. recheck the template digest, exact one-link artifact and bytes, and a fresh
   reconstruction of intended `N` from old `M` plus the reloaded authority;
7. revalidate the exact intended temporary and retained canonical `data`
   immediately before the atomic replacement; and
8. perform the one retained-directory-relative `os.replace`, `fsync` that
   exact directory descriptor, canonical-name rebind, exact readback, and
   complete registered-state validation.

This closes the finite P0-S1 interval identified in Section 8. The named
authority is no longer represented only by an earlier in-memory object while
manifest staging and source/data resnapshot proceed.

I reran the six-case boundary regression directly. Each mutation occurs after
the real final normalized data snapshot has returned:

| post-stage ordinary mutation | exact result |
| --- | --- |
| authorization-only JSON whitespace | rejected; zero replacement calls |
| coherent authorization plus review rebind | rejected; zero replacement calls |
| review-only JSON whitespace | rejected; zero replacement calls |
| target-byte change | rejected; zero replacement calls |
| coherent authorization-template and review rebind | rejected; zero replacement calls |
| intended-`N` stage-byte change | rejected; zero replacement calls |

All six preserved exact prepublication `M`, removed the identity-bound
invocation-owned manifest temporary, and did not delete or roll back a named
foreign mutation. When the target itself was not the injected mutation, the
exact artifact remained inert and unregistered as required. The six cases
plus the positive authority integration returned:

```text
7 passed in 0.50s
```

The positive path used strict synthetic `A` and `R`, the real common-directory
lock, real hard-link installation, the real final authority rebind, the real
retained-descriptor replacement, and the complete internal registered-state
validator. Instrumenting the actual `os.replace` boundary observed exactly
one replacement on first publication. The second authorized invocation was
an inode- and byte-preserving registered no-op and did not increase that
count.

### 9.3 Deterministic science and unchanged production repository

The focused publisher suite returned:

```text
53 passed, 1 skipped in 1.11s
```

The complete atmosphere converter, scientific worker, deterministic writer,
and publisher chain returned:

```text
94 passed, 1 skipped in 208.40s
```

The only default skip was the deliberately opt-in live reconstruction. I ran
that exact case separately:

```text
1 passed in 155.39s
```

It invoked two unrelated accepted top-level writers, each with two fresh
scientific children and distinct disposable Numba cache roots. The two
complete archives were byte-identical. Untrusted decode, scientific
validation, canonical member re-encoding, and canonical archive re-encoding
retained:

```text
archive bytes                  363050
archive SHA-256                1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
member count                   19
scientific array bytes         357984
source snapshot SHA-256        7d853d574f8aacce4eb04237df9509facd743ae54c1ca791ef1e7fd5d65ead02
data snapshot SHA-256          5be325c0200cf711ce13daa3dc96a47bdd8780c8de1640efe23827196de5b84a
```

Before and after the full and live runs:

- `data/MANIFEST.json` remained byte-identical at `d8f30e25…`;
- `data/README.md` remained byte-identical at `1a102874…`;
- the complete regular-file inventory under `data/` retained independent
  aggregate digest
  `fa84ba6c3e705d65cb304723279bd0954ad09418e72f285e0049e67ca824c11b`;
- `data/fixtures/chapter06_atmosphere_one_line_inputs.npz` remained absent;
- no artifact stage, manifest temporary, or external lock file existed; and
- the Payne Zero checkout remained clean at pinned commit
  `9c44001feae40b85146630499e6f8a5fed42e5af`.

Real production `--authorized-dry-run` and `--publish` each exited 2 before
candidate construction, lock acquisition, parent creation, or staging
because detached `A` and `R` are absent. Ruff check, Ruff format check, and
bytecode-disabled compilation all passed.

### 9.4 Retained transaction and scientific gates

The exact final focused suite and source trace retain the earlier accepted
properties:

- zero-argument fixed production APIs and CLI; no caller root, destination,
  force, replace, repair, merge, cleanup, or partial-lane surface;
- strict component-wise no-follow repository access, exact regular/single-link
  identities, strict duplicate-free finite JSON, closed key orders, and
  fixed authority paths;
- the canonical `data` directory inode as the shared atmosphere/synthesis
  lock, held through artifact plus manifest validation; lock failure before
  every write;
- create-exclusive no-follow staging, exact `0600` artifact mode independent
  of umask, inherited manifest mode, exact write, file `fsync`, descriptor
  readback, same-device proof, exact inode/link/mode/size/hash/byte checks, and
  identity-bound cleanup;
- atomic no-replace artifact installation, exact/nonexact/multilink target
  race behavior, retained mutated-parent `fsync`, canonical rebind, and no
  destructive rollback;
- exact append-last manifest encoding, two-field late-hash realization,
  delete-last recovery of byte-exact `M`, atomic retained-descriptor
  replacement, exact-directory durability, and postread;
- closed source/data snapshots before mutation, after artifact installation,
  and after manifest staging; source, non-target data, target, manifest,
  template, authority, review, and intended-`N` changes reject before
  replacement at their required boundaries;
- quarantine hard-stop, exact-unregistered recovery, manifest-failure inert
  target, complete registered idempotence, and no cleanup API;
- exact nineteen-member lexical schema, dimension-matched axes, nonempty
  scientific units/conventions, C-contiguous raw-byte array hashes, finite
  values, and full scientific invariants; and
- complete fresh registered-state validation under the still-held common
  lock.

No retained gate regressed in the final repair.

### 9.5 Threat boundary and final decision

The report states the frozen threat boundary accurately. In scope are
cooperative publishers and ordinary substitutions covered by the common
canonical-`data` lock, immediate retained-descriptor/name/byte validation,
atomic namespace syscalls, exact-directory `fsync`, canonical rebind, and
complete postvalidation. The six repaired late-authority cases are ordinary
pre-syscall mutations and now reject before replacement.

Out of scope remains unrestricted same-account or same-process code executing
arbitrary substitution literally inside `os.link` or `os.replace` after the
last userspace check. The report does not claim to prevent that actor or
preserve canonical bytes against it; later validation detects its effects
when possible. It does not claim a nonexistent descriptor-bound Darwin
manifest replacement and does not make the supported platform unusable by
adding such a gate. The adjudication remains design evidence only and grants
no authority.

Final finding counts for these exact bytes are:

| severity | findings |
| --- | ---: |
| P0 | 0 |
| P1 | 0 |
| P2 | 0 |

**ACCEPT** exact publisher
`5de0169339b8130d45c8e610684de67c2d6f3c6c62c5298610d0f705ebe0b363`,
focused tests
`00c09576b00420bfaa72f058a1b70fab1dedb4e85e04326ca83c7bac0b6ca876`,
and candidate report
`f5f9ae5508c447938b3262cc5a34fd6901fe9519d6e408318ba6370e0e94cfaf`
as the final atmosphere-fixture publisher implementation under the exact
frozen contract and adjudicated threat scope.

This acceptance is exact-object evidence only. It does not itself create a
detached publication authorization, approve a future authorization record,
publish the canonical fixture, mutate `data/MANIFEST.json`, authorize
quarantine cleanup, or constitute postpublication acceptance. Those remain
separate later lifecycle objects and actions.
