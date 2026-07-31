# Chapter 6 synthesis golden publisher repaired candidate

Status: **REPAIRED IMPLEMENTATION CANDIDATE — NOT INDEPENDENT ACCEPTANCE,
AUTHORIZATION, PUBLICATION, OR MANIFEST REGISTRATION**

Date: 2026-07-30

This record presents a new exact publisher candidate after the independent
audit at SHA-256
`422e0aa3b3fb04f0fc8c6d77d25a592a52a459c707ac7847c9da115a5c409a24`
rejected the preceding snapshot. It records repairs and adversarial evidence
for a fresh independent review. It grants no publication authority.

The exact identities and results in Sections 1–8 are preserved as the first
repair history. Section 9 onward supersedes them with the bounded functional
repair made after the complete appended re-audit and separate threat-model
adjudication.

## 1. Exact repaired candidate

| object | lines | bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `scripts/build_chapter06_synthesis_golden.py` | `3,566` | `126,898` | `2a345fd389d2c8f8a97ebef8b4d418a1c81f5b49b94dd94e595ac95cb2e0479e` |
| `tests/test_chapter06_synthesis_golden_publisher.py` | `2,049` | `80,405` | `a314d85569201c659c91308c9121e1f1762f4af72ee87f2ba84f93a8e198b7a1` |

This report intentionally does not embed its own hash. Its complete identity
must be computed after the report is final.

The repaired publisher retains the exact accepted contract and scientific
candidate boundary:

| accepted input | SHA-256 |
| --- | --- |
| lane-artifact publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| contract independent acceptance | `fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc` |
| synthesis candidate-byte acceptance | `434088cff95ed60d65dc6c9749d18c2e74e45d114787c03728ed7ae9cf0bd9c9` |
| atmosphere candidate-byte acceptance | `8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71` |
| synthesis writer | `57aa7147afee4a7366cb2a075715d3607fa20507c23c07ec978b0698368ae47b` |
| synthesis writer tests | `7c41a74f9d2e38a23d988c990af4040ac262a8066cb3cd9feae4e29f0bdc0a4e` |
| synthesis writer candidate | `6ab1f346a409b0302550a0923c35b71a84d6b2899f2c356070c8d76aa8145e5a` |
| synthesis writer acceptance | `467fdc810f14302dba80f0dd18ba34239dfedb7579b48280899f6f9b6e3b3653` |

Every phase-1 and phase-2 plan, worker, test, assembler, candidate, and audit
identity remains frozen in `FIXED_UPSTREAM_IDENTITIES`.

The accepted scientific candidate is unchanged:

```text
bytes       1,294,865
SHA-256     a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955
members     213
role        comparison-only synthesis golden
```

## 2. Fixed interface and authority boundary

The sole destination remains:

```text
data/golden/payne_zero/chapter06/synthesis/
chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz
```

There are three distinct fixed-path public modes:

```text
--verify-only   authorization-independent candidate reconstruction
--dry-run       complete authorized verification without mutation
--publish       complete authorized no-replace publication
```

The hidden `--internal-validate-published` route is callable only by the
publisher's fresh-process postpublication validator. No mode accepts a root,
destination, output, force, replace, repair, merge, cleanup, or partial-lane
argument. The Python entry points remain zero-argument.

`verify_only()` has no call edge to authority loading, the data lock,
directory creation, staging, hard-link installation, manifest replacement, or
`publish()`. It:

1. verifies the fixed accepted chain and complete source identity;
2. takes the closed data snapshot;
3. runs the accepted zero-argument writer twice;
4. requires identical complete bytes and reduced deterministic summaries;
5. decodes, semantically validates, and canonically re-encodes all 213
   members;
6. verifies the fixed target state; and
7. repeats source/upstream/data checks and requires zero delta.

Authorized dry-run and publication additionally require the exact detached
authorization and its separate exact JSON review. Missing, duplicate,
unknown, reordered, nonfinite, stale, cyclic, wrong-role, or wrong-lane
records reject. The authorization binds exact `M1`, the atmosphere-first
fixture state, source/data identity, the exact candidate, and one ordered
213-member append template.

## 3. Closure of the three P0 findings

### 3.1 Manifest member hashes now mean raw logical C bytes

The manifest array field `sha256` is now exactly:

```python
hashlib.sha256(
    np.ascontiguousarray(array).tobytes(order="C")
).hexdigest()
```

It no longer includes dtype or shape bytes. Dtype and shape remain separate
exact fields and remain covered by the compact schema digest.

The regression independently checks:

```text
float64 [1.0, 2.0] C-byte SHA-256
dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae
```

The test computes that value directly with `hashlib`, not through the
publisher helper.

### 3.2 Artifact stage is retained and rebound immediately before link

`_create_stage` now:

- opens the stage `O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW`;
- calls `fchmod(0600)` independently of umask;
- requires exact type, filesystem, mode, link count, and size;
- writes exactly once and rejects short progress;
- fsyncs and reads back through the same retained descriptor;
- binds the retained descriptor to the no-follow stage name; and
- returns that still-open descriptor through the install operation.

The narrow `_link_validated_stage_no_replace` primitive immediately
revalidates descriptor/name device, inode, link count, mode, size, and exact
bytes, then calls `os.link`. After a successful link, the publisher requires
both names and the retained descriptor to identify the expected two-link
inode before any staging-name cleanup. It never deletes a canonical target
after installation; a genuine post-link mismatch is preserved as evidence and
fails closed.

The hostile regression substitutes the stage name immediately before this
narrow primitive. The retained/name mismatch rejects before `os.link`; the
canonical target remains absent and the foreign name remains quarantined as
external evidence.

### 3.3 Manifest temporary is retained and rebound immediately before replace

The manifest temporary remains open from exact creation through the atomic
replace. It is explicitly set to the reviewed current-manifest mode,
independent of umask. The narrow
`_replace_validated_manifest_stage` primitive immediately requires:

- retained descriptor/name device and inode equality;
- one link;
- exact reviewed mode and byte count;
- descriptor-read exact intended `M2` bytes; and
- retained canonical `data` directory identity.

Only then does it call descriptor-relative `os.replace`. After replacement,
the manifest name must identify the retained temporary inode with exact bytes
and mode before success is possible.

The hostile regression substitutes the temporary name immediately before the
narrow primitive. Validation rejects before replacement, exact `M1` remains
unchanged, and the foreign temporary remains quarantined. There is no
post-replace rollback path: after any completed replacement, evidence is
preserved and failure requires forward audit as required by the contract.

## 4. Closure of the P1 state and lifecycle findings

### 4.1 Reachable exact-unregistered recovery

The data aggregate now represents stable prepublication data. It excludes
only objects whose identities are independently bound elsewhere:

- `data/MANIFEST.json`, bound by exact manifest hash and ordered path digest;
- the synthesis destination and its manifest-file join; and
- the two literal allowlisted synthesis directories.

All of those objects remain fully inventoried and validated. This narrowly
normalized aggregate makes these three lifecycle views share one authority
digest:

1. exact `M1` with absent target and directories;
2. exact `M1` with the two accepted directories and exact unregistered
   target; and
3. registered `M2` after delete-last reconstructs exact `M1`.

The source/data digest always adds the exact reconstructed `M1` hash and
ordered entry-path digest. No other file, directory, manifest entry, metadata,
or source delta is normalized.

The nonmocked isolated recovery regression binds a digest before the
directories/target exist, creates only the exact two directories and exact
single-link mode-`0600` target, and passes the same authority boundary.
Changing the target bytes then rejects.

### 4.2 Closed directory inventory

The normal prepublication directory set is now a frozen thirteen-path set.
Only these state-specific sets are accepted:

```text
base thirteen directories
base + data/golden/payne_zero/chapter06
base + chapter06 + chapter06/synthesis
```

The two additional directories must be mode `0755` and retain their parent's
owner and group. Any other missing or extra directory, including an empty
rogue directory, rejects. The existing retained-descriptor creation/adoption,
child fsync, parent fsync, reopen, metadata, contents, and reverse-cleanup
checks remain.

### 4.3 Immediate source and non-target-data recheck

Immediately before the manifest replacement primitive, while the canonical
data lock remains held, the publisher now:

1. rechecks every fixed upstream identity;
2. reruns the complete synthesis source identity gate;
3. takes the closed intermediate data snapshot with the exact unregistered
   target;
4. requires exact current `M1` hash and ordered entry digest; and
5. recomputes the authorization-bound source/data digest.

Separate hostile regressions change the accepted source view and the
non-target data aggregate. Both reject before manifest replacement.

### 4.4 Complete fresh-process validation remains under the data lock

The parent retains the canonical `data` directory flock while a new Python
process executes `_registered_validation_report`. The fresh process runs the
complete authorized fixed-root preparation again, including:

- trust identities and detached authority/review;
- two accepted top-level candidate builds;
- full 213-member untrusted decode, semantic validation, and canonical
  re-encode;
- reconstructed exact atmosphere-first `M1`;
- unique realized synthesis entry and exact append-last `M2`;
- complete source and closed data state;
- exact single-link mode-`0600` artifact; and
- exact manifest/artifact/role/member-metadata join.

Its deterministic JSON binds artifact, manifest, entry, authority, review,
source/data, closed-data aggregate, member-metadata, atmosphere phase, role,
and completion decision. The parent requires the exact report before
releasing the lock.

### 4.5 Handled stage failures and exact modes

Write, fsync, descriptor-readback, named-inode, size, and mode checks are now
inside one identity-bound cleanup region. A handled readback failure removes
only the invocation-owned inode and fsyncs its retained parent. A changed or
foreign name is never deleted.

Artifact mode is exactly `0600` even under umask `0777`. Manifest temporary
mode is explicitly set to and verified against the current reviewed manifest
mode, also under umask `0777`.

### 4.6 Exact scientific units and axes

Template validation rejects empty or whitespace-only units, axes, and
ownership. Authorized candidate validation additionally freezes the complete
lexical 213-member mapping of member name to exact Payne Zero unit and axes at
SHA-256:

```text
39d319b577a59ea06de475a14da9fe68d5fcced24119964a4e3e322226b2b78a
```

That digest was independently reproduced from the accepted compact
assembler's `CompactMemberSpec` sequence. Generic unit labels, wrong axis
names, or any one-member semantic change reject even when rank and types
remain valid.

## 5. Preserved contract properties

The repaired candidate retains:

- fixed canonical paths and no caller filesystem authority;
- strict duplicate-free, order-sensitive JSON;
- component-wise directory-relative no-follow reads;
- the canonical `data` directory as the one cross-lane flock object;
- two unrelated accepted top-level writer builds;
- exact 213-member ZIP/NPY metadata and canonical byte re-encoding;
- comparison-only ownership and absence of copied atmosphere input state;
- create-only allowlisted nested directories with retained-dirfd durability;
- hard-link no-replace installation with `EEXIST` exact-winner handling;
- append-last manifest construction and exact delete-last reconstruction;
- quarantine hard stops without inferred cleanup;
- preservation of an exact inert file after artifact-install/manifest
  failure; and
- registered validation-only idempotence without artifact or manifest
  rewrite.

There is no target rollback and no manifest rollback after their respective
atomic mutation boundaries.

## 6. Adversarial verification

Focused repaired publisher suite:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/test_chapter06_synthesis_golden_publisher.py

51 passed in 22.32s
```

Complete synthesis chain:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_golden_publisher.py

71 passed in 232.56s
```

The repaired hostile matrix includes:

- independent raw-C-byte member hash meaning;
- stage substitution immediately before no-replace;
- manifest-temporary substitution immediately before replacement;
- exact-unregistered normalized recovery plus nonexact rejection;
- arbitrary empty-directory rejection;
- source and non-target-data changes at the manifest boundary;
- fresh complete validation while the lock is observably held;
- handled readback cleanup;
- artifact and manifest modes under restrictive umask;
- empty and incorrect scientific units/axes;
- authorization-independent verify-only reachability and mutation
  unreachability;
- all preceding strict JSON, path, symlink, hard-link, ZIP/NPY, writer
  determinism, short-write, fsync, directory, quarantine, manifest-delta,
  target-race, recovery, and idempotence cases.

Compilation, lint, and formatting checks pass:

```text
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  scripts/build_chapter06_synthesis_golden.py \
  tests/test_chapter06_synthesis_golden_publisher.py

python -m ruff check \
  scripts/build_chapter06_synthesis_golden.py \
  tests/test_chapter06_synthesis_golden_publisher.py

python -m ruff format --check \
  scripts/build_chapter06_synthesis_golden.py \
  tests/test_chapter06_synthesis_golden_publisher.py
```

## 7. Real CLI and zero-delta evidence

Two consecutive real fixed-path commands completed successfully:

```text
PYTHONDONTWRITEBYTECODE=1 \
python scripts/build_chapter06_synthesis_golden.py --verify-only
```

The deterministic reports were identical and recorded:

```text
decision                 VERIFIED_CANDIDATE_ONLY_NOT_AUTHORIZED
authorization_checked    false
publication_performed    false
artifact_sha256          a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955
artifact_bytes           1294865
archive_member_count     213
target_state             absent
atmosphere_phase_ready   false
repository_delta         zero
```

The false atmosphere prerequisite is the correct current state: candidate
verification is independent, but authorized synthesis publication remains
blocked until atmosphere `M1` exists.

Real `--dry-run` and `--publish` both return exit status 2 before candidate,
lock, parent, or stage creation because detached authorization is absent:

```json
{"reason":"design/chapter06_synthesis_publication_acceptance.json is missing, symlinked, or unreadable","status":"REJECT"}
```

Before and after all tests and real CLI runs:

| canonical property | exact unchanged result |
| --- | --- |
| closed stable data aggregate | `56fb4fb6be3a0a6733b0427ddc3947cb2deee41cf307dfb1bb073c93822ad910` |
| data directories | `13` |
| data regular files | `39` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| `data/README.md` SHA-256 | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |
| synthesis destination | absent |
| synthesis artifact/manifest quarantine | absent |
| detached synthesis authorization/review | absent |

The previous independent audit exists only as a `REJECT` record. No accepted
publisher audit, detached authorization, authorization review, atmosphere
`M1`, synthesis artifact, manifest entry, or postpublication audit was
created.

## 8. Disposition

This new exact implementation and test pair is submitted for fresh
independent publisher review. The reviewer should rerun the hostile
descriptor/name substitution boundaries, normalized recovery state,
state-specific directory inventory, immediate source/data check, and complete
fresh-process-under-lock validation rather than relying on this report.

Until a later independent audit accepts these new exact bytes and a separate
detached authorization plus exact JSON review bind the then-current
atmosphere `M1`, both authorized modes must continue to reject. This candidate
report is neither acceptance nor authority.

## 9. Bounded functional repair after the appended re-audit

The complete first-repair audit was final at:

```text
7790c4b7366a05456d3c5a947e6ad3b3890c16c07e7681e5acc84e86c409a15c
```

It preserved the earlier findings, accepted the first repair of most of them,
and found one deterministic P0: the immediate source/data snapshot treated
the publisher's own live manifest temporary as crash-left quarantine. It also
required each mutated retained directory to be rebound to its canonical path
after mutation and durability before a helper could report success.

The separate design adjudication is:

```text
design/chapter06_publisher_threat_model_adjudication.md
4ed79dd4ff622abbeb6106a58b1b7e8c05a10c2ff87cfeb0dd6ee06e24e8e9fc
```

It rejects arbitrary same-process code injection literally inside
`os.link`/`os.replace` as an acceptance-blocking interpretation of the frozen
Section 12 threat boundary. It retains immediate validation, retained
descriptors, the common lock, atomic syscalls, durability, canonical rebind,
and postvalidation. This repair follows that boundary: it neither promises to
defeat arbitrary code already executing as the repository owner nor makes
Darwin publication depend on a nonexistent descriptor-bound manifest
replacement primitive.

## 10. Exact current candidate

| object | lines | bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `scripts/build_chapter06_synthesis_golden.py` | `3,804` | `135,798` | `a9cc41f6b0862aa6857b0fdc060c6b6e22b6a2bd094c5cf1b9bc9b04463f6469` |
| `tests/test_chapter06_synthesis_golden_publisher.py` | `2,439` | `96,902` | `2d7fef267bbf3e71cd4267d7cdb8ddd1c0f1047b92cf212fef909fdce15e00b3` |

The shared contract, contract acceptance, candidate-byte acceptances, writer
chain, accepted archive bytes, destination, role, and manifest policy remain
exactly those recorded in Sections 1–2. This report intentionally contains no
self-hash.

## 11. Closure of the remaining functional findings

### 11.1 The immediate snapshot admits only the invocation-owned temporary

The new immutable `OwnedManifestTemporary` evidence carries:

- the exact canonical stage path;
- the still-open stage descriptor;
- the still-open canonical `data` parent descriptor;
- the retained post-write `stat` identity;
- exact intended bytes and their SHA-256; and
- the reviewed manifest mode.

`_validate_owned_manifest_temporary` requires all of the following before the
closed snapshot may treat that file as this invocation's transient:

- the path is one literal child of canonical `data` with the exact manifest
  prefix;
- the retained parent descriptor still reopens as canonical `data`;
- retained descriptor, no-follow name, and creation record have the same
  device and inode;
- all are regular, single-link files with the same owner, group, exact mode,
  and exact intended size;
- the declared intended SHA-256 equals the hash of the intended bytes;
- descriptor readback and a separate no-follow named read both equal the
  complete intended bytes and hash; and
- the complete quarantine inventory contains exactly that one `FileFact`.

The live temporary remains explicitly present in the snapshot's regular-file
inventory and allowed nonmanifest set. Only that exact path is excluded from
the stable authority aggregate. It is rebound and reread again after the
complete data-tree walk. A second temporary, artifact stage, substituted
inode, changed name, link, mode, size, hash, bytes, parent, or descriptor is
never normalized and remains a quarantine hard stop.

`_replace_manifest` constructs this evidence directly from its own retained
stage and passes it only to the immediate source/data recheck. It cannot be
supplied through the public interface.

### 11.2 A real authorized manifest transition reaches replacement

The focused suite now builds an isolated but complete `M1` data tree with:

- one registered atmosphere fixture;
- exact closed README support;
- the two allowlisted synthesis directories;
- the exact unregistered synthesis target;
- exact authority-bound source/data digest;
- real detached authorization/review bytes at their fixed paths; and
- a non-`None` `Authority`.

It calls the real `_replace_manifest`, real `_snapshot_data`, real temporary
creation and validation, and real descriptor-relative `os.replace`. No
snapshot or replacement helper is mocked. The successful regression reaches
the atomic replacement exactly once, installs exact intended manifest bytes,
preserves the artifact, and leaves no temporary.

Three variants use the same real path after staging:

1. changing the accepted source file changes the recomputed source digest and
   rejects before replacement;
2. changing `data/README.md` breaks the closed non-target inventory and
   rejects before replacement; and
3. adding a second manifest-prefixed temporary rejects as quarantine,
   preserves exact old manifest bytes, removes only the invocation-owned
   temporary, and leaves the foreign object untouched.

These regressions directly close the integration gap identified in the
appended re-audit; they do not test the snapshot in isolation through a
mocked return value.

### 11.3 Artifact parent is rebound after link durability and cleanup

After the hard-link decision, `_install_no_replace` now:

1. `fsync`s the exact retained destination-parent descriptor;
2. reopens the canonical destination-parent path and requires the same
   directory device/inode;
3. removes only the identity-bound staging name and `fsync`s its retained
   parent;
4. rebinds the retained parent to the canonical path again;
5. reads the final target through the retained directory; and
6. requires the canonical target pathname to be the same regular inode with
   exact bytes, link count, size, and accepted archive hash.

A regression performs the real hard link, then moves the mutated parent and
creates a replacement at the canonical path before returning from the
wrapped syscall. The exact old directory is still flushed, but the immediate
canonical rebind rejects inside `_install_no_replace`. The exact linked file
and its staging link remain in the moved directory as evidence; the publisher
does not roll them back or delete post-install state.

### 11.4 Manifest parent and canonical name are rebound after replacement

After the real manifest replacement, `_replace_manifest` now:

1. marks the replacement complete before any later check;
2. `fsync`s the exact retained `data` descriptor;
3. reopens canonical `data` and requires the same directory device/inode;
4. requires the manifest name through that retained descriptor to be the
   retained temporary inode with exact mode, size, and bytes; and
5. independently requires the canonical manifest pathname to be that same
   regular inode before helper success.

A regression performs the real `os.replace`, moves the mutated `data`
directory, and creates a replacement canonical directory before the syscall
wrapper returns. The exact mutated directory is flushed and the immediate
canonical rebind rejects. The intended new manifest and exact artifact remain
in the moved directory. There is no rollback, old-manifest restoration, or
deletion after the completed replacement.

These post-mutation wrappers model the contract's explicitly enumerated
parent/name rebind requirement. The implementation makes no stronger claim
that it can contain arbitrary code executing inside the syscall entry itself.

## 12. Current verification evidence

Focused publisher suite:

```text
57 passed in 9.08s
```

Complete synthesis worker/assembler/writer/publisher chain:

```text
77 passed in 64.22s
```

Python compilation, Ruff lint, Ruff format check, and Git whitespace check
all pass for the exact script and test bytes in Section 10.

The real fixed-root `--verify-only` command rebuilt the candidate twice and
returned:

```text
decision                 VERIFIED_CANDIDATE_ONLY_NOT_AUTHORIZED
candidate_built_twice    true
candidate_bytes_equal    true
canonical_reencode_equal true
artifact_sha256          a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955
artifact_bytes           1294865
archive_member_count     213
target_state             absent
publication_performed    false
repository_delta         zero
```

Real `--dry-run` and `--publish` each returned exit status `2` before
candidate, lock, parent, or stage creation because authorization and review
remain absent.

Before and after focused tests, full-chain tests, live reconstruction, and
both negative commands, the exact canonical state remained:

| canonical property | exact unchanged result |
| --- | --- |
| closed stable data aggregate | `56fb4fb6be3a0a6733b0427ddc3947cb2deee41cf307dfb1bb073c93822ad910` |
| data directories | `13` |
| data regular files | `39` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| `data/README.md` SHA-256 | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |
| synthesis destination | absent |
| synthesis authorization/review | absent |
| synthesis quarantine objects | absent |

## 13. Current disposition

**REPAIRED CANDIDATE ONLY — NOT SELF-ACCEPTANCE OR AUTHORITY.**

The deterministic live manifest reachability blocker and both finite
post-mutation canonical-rebind gaps are closed by new exact implementation
and test bytes. The accepted scientific archive remains unchanged.

A new independent adversarial review must bind the exact Section 10 hashes,
rerun the real non-`None`-authority transition and mutation variants, confirm
the exact owned-temporary exclusion, exercise both post-mutation parent
swaps, and repeat the full chain and canonical zero-delta proof. Only that
review may accept or reject this publisher. No detached authorization,
record review, canonical artifact, manifest entry, quarantine cleanup, or
postpublication record may be created from this candidate report alone.
