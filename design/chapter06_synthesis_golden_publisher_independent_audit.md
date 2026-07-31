# Chapter 6 synthesis golden-publisher independent audit

Status: independent publication-boundary review  
Audited: 2026-07-30  
Disposition: **REJECT**

The candidate has a strong fixed-path, authorization, deterministic-build,
canonical-archive, and no-replace foundation. Its focused tests and the full
synthesis construction chain pass. It is nevertheless unsafe to accept as a
canonical publisher.

Three independent defects are individually blocking:

1. manifest member hashes are not the contract's C-byte SHA-256 values;
2. replacing the artifact-stage pathname after validation can install
   attacker-selected bytes at the canonical destination before the publisher
   rejects;
3. replacing the manifest-temporary pathname after validation can overwrite
   `data/MANIFEST.json` with attacker-selected bytes before the publisher
   rejects.

Additional defects make the promised exact-unregistered recovery state
unreachable, leave the closed directory inventory open, weaken immediate
pre-mutation identity checks and fresh-process validation, and leave
invocation-owned temporary files after handled failures.

No publisher implementation, focused test, candidate report, authorization,
record review, canonical data file, manifest, external Payne Zero source,
paper source, or golden artifact was modified by this review. This report is
the only repository file added.

## 1. Frozen review snapshot

The requested candidate identities reproduced exactly:

| reviewed artifact | SHA-256 |
| --- | --- |
| synthesis publisher candidate | `0dcc9e097e386dd1c8774ff722dbf7f78cac62db029e19b94b80018f12b11732` |
| focused publisher tests | `c242cdaaa3d864aefa61f5bc025571baca882e50e608141e3970a0c4f69800ac` |
| publisher candidate report | `974b65234b2e0a577ec623ed2cc173b318aadadbe8c8e7f08510f3237e70ce14` |
| final lane-artifact publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| final synthesis candidate-byte acceptance | `434088cff95ed60d65dc6c9749d18c2e74e45d114787c03728ed7ae9cf0bd9c9` |

The candidate script is 2,968 lines and 106,187 bytes. The focused test file
is 1,393 lines and 55,539 bytes. The candidate report is 323 lines and 14,292
bytes.

The exact accepted scientific candidate remains:

```text
archive bytes:
  1,294,865

archive SHA-256:
  b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850

archive members:
  213
```

This audit does not reopen that candidate-byte decision. It reviews whether
the publisher safely and exactly moves those already accepted bytes across
the canonical publication boundary.

## 2. Disposition summary

| priority | finding | hostile result |
| --- | --- | --- |
| P0 | per-member digest has the wrong meaning | publisher digest differs from SHA-256 of C bytes |
| P0 | artifact-stage pathname substitution | foreign bytes appear at the canonical target before rejection |
| P0 | manifest-temporary pathname substitution | foreign bytes replace the canonical manifest before rejection |
| P1 | exact-unregistered crash recovery is unreachable | recovery snapshot differs from the authorization-bound prepublication snapshot |
| P1 | `data/` directory inventory is not closed | an unexpected empty directory is accepted |
| P1 | source/data are not rechecked at the manifest mutation boundary | `_replace_manifest` has no source or closed-data snapshot validation |
| P1 | fresh-process validation is incomplete and occurs after lock release | child only parses generic JSON and prints two hashes |
| P1 | handled stage-readback failures leak invocation-owned temporaries | injected readback failure leaves a hidden stage |
| P1 | artifact stage mode `0600` is not enforced | restrictive ambient umask produced mode `000` and a leaked stage |
| P1 | array units and axis labels are not semantically validated | empty unit and empty axis label are accepted |
| P1 | contract verification-only mode is not exposed independently of authorization | both public modes enter `_prepare_authorized` |

The first three findings are sufficient for rejection. The remaining findings
are recorded so a repair can close the publication boundary once, rather than
serially discovering blockers.

## 3. P0 findings

### 3.1 Manifest member hashes are not C-byte SHA-256 values

The contract requires each realized array record to contain the SHA-256 of
that member's C-order array bytes. The candidate's `_member_sha256` instead
hashes this concatenation:

```text
dtype.str bytes
+ shape encoded as int64 bytes
+ C-order array bytes
```

That is a useful typed-array fingerprint, but it is not the required C-byte
SHA-256. For:

```python
np.asarray([1.0, 2.0], dtype=np.float64)
```

the independent probe produced:

```text
contract C-byte SHA-256:
  dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae

publisher _member_sha256:
  eb9553124d3bb34d643a8742ed938ef0e358f74ef6bfdc5cf723c3103940f440

equal:
  false
```

`_validate_candidate_semantics` requires the template record to carry the
second digest. The focused test helper `_array_record` also calls the
publisher's `_member_sha256`, so the tests reproduce the implementation's
incorrect definition instead of independently checking the contract.

This would write misleading scientific provenance into all 213 manifest
records. The repair must compute:

```python
hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()
```

for the `sha256` field. If a dtype-and-shape-inclusive fingerprint is also
desired, it needs a separately named field and a reviewed contract change; it
cannot replace the specified C-byte digest.

### 3.2 Artifact-stage inode substitution can install foreign bytes

`_create_stage` records and validates the created stage inode. After that
validation, `_install_no_replace` links the *current object found at the stage
pathname* without first requiring its device/inode and bytes to equal the
retained stage identity and accepted candidate.

An isolated hostile harness performed these operations:

1. created a valid stage containing accepted test bytes;
2. retained the original `stage_details`;
3. unlinked the stage name and created a new regular file with the same name
   containing `foreign-race-bytes`;
4. called `_install_no_replace` with the retained original identity.

The exact result was:

```json
{
  "raised": "PublicationIOError",
  "stage_exists": true,
  "target_bytes": "foreign-race-bytes",
  "target_exists": true,
  "target_link_count": 2
}
```

The publisher eventually rejects because `_unlink_owned_stage` notices that
the stage inode changed. That check is too late: `os.link` has already made
the substituted inode the canonical target. The negative case therefore
mutates canonical bytes, directly contradicting the required no-replace race
and unchanged-on-rejection properties.

The repair must retain an open validated stage descriptor through install
where the platform primitive permits it, or immediately before `os.link`
re-stat the no-follow stage name relative to the retained parent descriptor,
require the original device/inode/link count/mode/size, and read/hash/fully
validate its bytes again. The implementation must have a hostile test that
substitutes the name precisely between initial readback and link.

### 3.3 Manifest-temporary inode substitution can overwrite the manifest

`_replace_manifest` initially validates the manifest temporary and its open
descriptor. It then rechecks the current manifest, artifact,
authorization/review, template, and intended manifest. Immediately before
`os.replace`, however, it does not rebind the temporary pathname to the
retained `stage_details`.

An isolated hostile harness wrapped the `os.replace` boundary. The wrapper
removed the validated temporary, created a different regular file at the same
name containing:

```json
{"foreign":true}
```

and then invoked the real atomic replacement. The exact result was:

```json
{
  "intended_preserved": false,
  "manifest_bytes": "{\"foreign\":true}\n",
  "original_preserved": false,
  "raised": "PublicationIOError"
}
```

Post-replacement readback correctly notices the mismatch, but only after the
canonical manifest has been overwritten. This is a canonical-data corruption
path on a handled rejection.

The repair must re-stat and re-read the no-follow temporary name immediately
before replacement, require the retained device/inode/link/mode/size and exact
intended bytes, and keep the data-directory descriptor and lock throughout.
A hostile test must replace the temporary at that exact boundary and prove
that the original canonical manifest bytes remain unchanged.

## 4. P1 state-machine and inventory findings

### 4.1 The exact-unregistered recovery state cannot pass its authorization

The candidate report promises recovery from:

```text
M1 manifest + exact unregistered synthesis target
```

The implementation allowlists that target as a nonmanifest file when calling
`_snapshot_data`. It still includes the target and any newly created
destination directories in the snapshot's aggregate digest. Later,
`_validate_prepublication_state` requires that digest to equal the
authorization's prepublication source/data snapshot digest.

An authorization created in the required prepublication state necessarily
describes an absent target and absent synthesis destination directories. A
crash after artifact installation but before manifest replacement necessarily
adds those objects. The same authorization can therefore never match both
states.

An independent minimal-state harness reproduced:

```text
authorization-bound prepublication digest:
  d6d0543e5f1dccbf11dc1e3d05e7aa50ffda26ebd65af467fb4e20a25050831d

exact-unregistered recovery digest:
  7429d105c8ebb84d6f8ecdc2c6b19e2203b8126ad11fabcd48b1e00fd11515ad

equal:
  false

new directories:
  4

new files:
  1
```

The focused recovery tests mock the authorized preparation/snapshot boundary,
so they do not exercise this actual digest comparison.

The repair needs an explicit, reviewed recovery snapshot normalization or a
separate exact recovery digest bound by the authorization. Only the exact
accepted target, exact newly permitted directory components, and their exact
metadata may be normalized; every other difference must remain a hard stop.
A full nonmocked recovery test must start from the bound prepublication state,
simulate the crash point, and finish registration with the same authority.

### 4.2 The closed directory inventory is open

`_snapshot_data` inventories directories, but it never compares their path set
against a closed permitted set. It strictly rejects unexpected regular files,
not unexpected empty directories.

In a temporary canonical-data harness, adding:

```text
data/rogue-empty-directory/
```

was accepted. The observed inventory was:

```json
{
  "accepted": true,
  "inventory": [
    "data",
    "data/rogue-empty-directory"
  ]
}
```

The contract explicitly defines `data/` as four closed inventories, beginning
with directories. A fixed expected directory set is therefore required for
normal M1, exact-unregistered recovery, and postpublication M2 states. The
allowed synthesis directories must be state-specific; arbitrary empty
directories cannot be incorporated into the digest and accepted.

## 5. P1 mutation-boundary and lifecycle findings

### 5.1 Source and non-target data are not rechecked immediately before manifest replacement

The publisher performs substantial identity work in `_prepare_authorized`,
including a second call while holding the data-directory lock. It then creates
and installs the artifact and proceeds into `_replace_manifest`.

At the manifest replacement boundary, `_replace_manifest` rechecks:

- current manifest bytes and inode;
- final artifact bytes;
- authorization and record-review bytes;
- template digest;
- intended manifest parse/serialization;
- retained data-directory identity.

It does not rerun `_verify_source_identity`,
`_verify_fixed_upstream_identities`, or `_snapshot_data`. A static inspection
of `_replace_manifest.__code__.co_names` confirmed that none of those three
checks is reachable there.

The contract requires accepted sources and the closed data snapshot both
before construction and immediately before canonical writes. The lock
serializes cooperating publishers; it does not prevent another process from
changing a source or a non-target data object.

The repair must introduce an immediate pre-manifest-mutation comparison
against the authorization-bound source identities and the exact permitted
intermediate data state. A hostile test should change an accepted source and
a separate non-target data file after artifact installation but before
manifest replacement, and prove the manifest remains unchanged.

### 5.2 Fresh-process validation is incomplete and runs after lock release

The fresh child in `_fresh_validate_expected_final`:

1. reads the artifact;
2. reads the manifest;
3. runs only generic `_parse_manifest`;
4. checks target link count;
5. prints the artifact and manifest whole-file hashes.

It does not load or validate the detached authorization/review, reconstruct
the exact realized entry, validate the manifest/file join, call
`_validate_candidate_semantics`, validate the registered lifecycle state,
recheck the atmosphere M1 prerequisite, or verify accepted source identities.
The parent only compares two printed hashes to its expected hashes.

Moreover, `publish` exits `_exclusive_data_lock` before launching this child.
The fresh validation therefore neither satisfies the contract's complete
lane-specific validation nor observes a state protected by the publication
lock.

The repair must make the fresh process execute the complete registered-state
validator from the fixed canonical root and keep the publication lock until
that validation has succeeded. Its success report should be deterministic
and should bind the exact artifact, manifest, realized entry, authority,
review, source/data state, role, member metadata, and lane prerequisite.

### 5.3 Handled readback failures leave invocation-owned stages

`_create_stage` cleans up failures during write and `fsync`. It closes the
descriptor and performs named readback outside that cleanup block. If the
readback helper raises, the invocation-owned stage remains.

An injected `_read_regular_nofollow` failure produced:

```json
{
  "leftovers": [
    ".chapter06-synthesis-artifact-stage-<token>"
  ],
  "raised": "PublisherIdentityError"
}
```

This is a handled exception, not a process crash. It should remove only the
invocation-owned stage after confirming its retained inode. The current
behavior converts an ordinary I/O/identity rejection into a quarantine state
that blocks later publication.

The readback, identity comparison, and second descriptor open need to be
inside the same identity-bound cleanup region as write and `fsync`.

### 5.4 Exact artifact-stage mode `0600` is not enforced

The artifact stage is opened with mode `0600`, but `open(..., mode)` is
filtered by the ambient process umask. The implementation verifies file type,
link count, and filesystem, but not exact mode and does not `fchmod` to the
contract's fixed mode.

With ambient umask `0o777`, the hostile harness observed:

```json
{
  "leftovers": [
    ".chapter06-synthesis-artifact-stage-<token>"
  ],
  "modes": [
    "0o0"
  ],
  "raised": "PublisherIdentityError"
}
```

This also reaches the handled-readback leak above. The repair must set and
verify exact `0600` through the open descriptor independently of umask. The
manifest temporary should likewise explicitly reproduce and verify the
reviewed manifest mode.

## 6. P1 manifest-semantic and interface findings

### 6.1 Empty units and axis labels are accepted

`_validate_template_shape` requires `unit` and each axis label only to be
instances of `str`. `_validate_candidate_semantics` checks axis rank but
neither scientific label contents nor exact per-member semantics.

Using the focused suite's valid 213-member template, the independent probe set
one unit and one axis label to the empty string. Validation still passed:

```json
{
  "empty_axis_member": "member_000",
  "empty_scientific_metadata_accepted": true,
  "empty_unit_member": "member_000"
}
```

The test helper currently assigns every array the generic unit
`accepted compact-member unit` and generic labels such as `axis_0`. This does
not prove the exact scientific unit/convention and axes required by the
manifest contract.

The accepted template must carry a frozen exact 213-member scientific metadata
mapping derived from the Payne Zero notation already used by the synthesis
implementation. The publisher must compare each name to that mapping, not
merely check types and rank. Dimensionless scalars should use one explicit
reviewed convention rather than an empty string.

### 6.2 No authorization-independent verification-only branch exists

The contract distinguishes verification-only behavior generally from the
additional record checks performed “in authorized mode.” The candidate
exposes only:

```text
--dry-run
--publish
```

`dry_run` immediately enters `_prepare_authorized`, and
`_prepare_authorized` always loads the detached authorization/review before
building the candidate. There is no public verification-only route that
validates the accepted identities, builds twice, decodes/re-encodes, and
checks safe preconditions while authority is intentionally absent.

At this reviewed repository state, both direct commands safely rejected:

```text
python scripts/build_chapter06_synthesis_golden.py --dry-run
python scripts/build_chapter06_synthesis_golden.py --publish
```

with exit status `2` and:

```json
{
  "reason": "design/chapter06_synthesis_publication_acceptance.json is missing, symlinked, or unreadable",
  "status": "REJECT"
}
```

That is safe default behavior, but it is not the contract's complete
authorization-independent verification path. The repair should expose one
fixed-path, mutation-unreachable verification mode, then layer exact
authorization/review checks on the authorized dry-run and publish modes.

## 7. What independently passed

The rejection should not obscure the substantial good work in the candidate.
The following reproduced:

```text
focused publisher suite:
  39 passed in 4.56 s

complete synthesis worker/assembler/writer/publisher chain:
  59 passed in 29.85 s
```

The complete-chain command included:

- `tests/test_chapter06_synthesis_oracle_worker.py`;
- `tests/test_chapter06_synthesis_compact_assembler.py`;
- `tests/test_chapter06_synthesis_compact_writer.py`;
- `tests/test_chapter06_synthesis_golden_publisher.py`.

The exact candidate script and tests also passed:

- Python byte compilation;
- Ruff lint;
- Ruff formatting check.

Code reading and the passing cases confirmed:

- fixed canonical repository and destination constants;
- no caller-selected root, destination, force, repair, or replacement option;
- strict duplicate-key and exact-key JSON parsing;
- binding of a detached authorization to a separate review;
- deterministic two-build candidate comparison;
- canonical ZIP/NPY decode and byte-for-byte re-encode validation;
- archive-level size, member, schema, payload, role, and ownership checks;
- stable data-directory locking;
- create-exclusive hidden staging;
- hard-link-based no-replace installation in the normal case;
- manifest append construction and delete-last reconstruction;
- manifest temporary write, `fsync`, and normal-case atomic replacement;
- rejection of ordinary destination races, symlinks, alternate paths,
  malformed records, stale identities, and quarantine objects.

These properties are worth preserving. The findings above concern gaps at
the exact adversarial boundaries where “checked earlier” is not equivalent to
“the object mutated now is the checked object.”

## 8. Canonical no-mutation evidence

After focused tests, full-chain tests, direct command rejection, and all
isolated hostile probes, the live canonical state remained:

| property | exact result |
| --- | --- |
| `data/` directories, including `data` | `13` |
| `data/` regular files | `39` |
| `data/` regular-file bytes | `30,046,405` |
| closed snapshot aggregate | `3b303db7d0ebfa65dfbb4d8f608dff0aa5d9381cf45f67e95e1c58c58525bda2` |
| `data/MANIFEST.json` bytes | `1,087,741` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| `data/README.md` SHA-256 | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |
| synthesis golden destination | absent |
| synthesis manifest entry | absent |
| synthesis authorization | absent |
| synthesis authorization review | absent |
| crash-left synthesis stage/temporary | absent |

All destructive hostile tests ran only in isolated temporary directories with
patched fixed constants. None used the canonical `data/` tree as a mutation
target.

The requested candidate script, test, and candidate-report hashes remained
exactly unchanged after the review.

## 9. Required acceptance closure

A repaired publisher should not be independently accepted until all of the
following are true:

1. every manifest member `sha256` is independently proven to be the exact
   SHA-256 of C-order array bytes only;
2. artifact stage substitution between validation and no-replace install is
   rejected before the destination is created;
3. manifest temporary substitution between validation and replacement is
   rejected while preserving the original manifest bytes;
4. the exact-unregistered recovery state passes a real, nonmocked test with
   the same reviewed authority, while every nonexact delta rejects;
5. the directory inventory is closed for each lifecycle state;
6. accepted sources and the exact permitted intermediate data state are
   rechecked immediately before manifest mutation;
7. the publication lock remains held through a fresh-process *complete*
   registered-state validation;
8. every handled stage failure removes only its identity-bound
   invocation-owned temporary;
9. artifact mode `0600` and the reviewed manifest-temporary mode are enforced
   independently of umask;
10. all 213 units/conventions and axes are fixed scientific semantics, not
    merely strings of the correct rank;
11. the authorization-independent verification-only contract is reachable
    and has no publication primitive;
12. focused hostile tests exercise every boundary above, followed by the
    complete synthesis chain and an independent no-mutation inventory.

After those repairs, the publisher implementation, tests, and candidate report
will have new hashes and will require a new independent review. This rejected
snapshot must not be named in a detached publication authorization.

## 10. Final decision

**REJECT**

The accepted synthesis candidate bytes remain scientifically and
serially accepted. The reviewed publisher does not yet provide the required
canonical mutation safety or exact manifest semantics. In particular, the
artifact and manifest inode-substitution probes demonstrate real
mutate-before-reject behavior, not a documentation-only gap.

<!-- BEGIN REPAIRED SYNTHESIS PUBLISHER RE-AUDIT 2026-07-30 -->

# Re-audit of the repaired Chapter 6 synthesis publisher

**Review date:** 2026-07-30  
**Review mode:** independent hostile repair re-audit  
**Disposition:** **REJECT**

This section is append-only. It preserves the original rejected review above
as the historical record and evaluates the exact repaired candidate named
below. No publisher, test, candidate-report, authority, canonical data, or
accepted external byte was changed during this re-audit.

## 11. Exact repaired snapshot reviewed

| reviewed object | exact SHA-256 |
| --- | --- |
| `scripts/build_chapter06_synthesis_golden.py` | `2a345fd389d2c8f8a97ebef8b4d418a1c81f5b49b94dd94e595ac95cb2e0479e` |
| `tests/test_chapter06_synthesis_golden_publisher.py` | `a314d85569201c659c91308c9121e1f1762f4af72ee87f2ba84f93a8e198b7a1` |
| `design/chapter06_synthesis_golden_publisher_candidate.md` | `89d38d06ff3868289207f9626f068038b1f523cb2c8b456a4e11ed340957d378` |
| original independent audit, before this append | `422e0aa3b3fb04f0fc8c6d77d25a592a52a459c707ac7847c9da115a5c409a24` |

The exact synthesis candidate remains:

| property | exact accepted value |
| --- | --- |
| archive SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| archive bytes | `1,294,865` |
| archive members | `213` |

The repaired implementation closes most of the original findings. It does
not, however, provide a reachable authorized first-publication path. A new
nonmocked integration probe proves that the invocation-owned manifest
temporary is rejected by the publisher's own quarantine inventory before it
can be promoted. Two additional post-mutation parent-swap probes show that
the correct retained directory inode is flushed, but the helpers can report
local success after the canonical pathname has been rebound elsewhere.

## 12. Resolution of the original findings

| original finding | repaired status | independent evidence |
| --- | --- | --- |
| manifest member hashes included typed metadata | **closed** | `_member_sha256` now hashes only contiguous C-order array bytes; the focused suite independently checks the exact byte digest |
| artifact stage could be substituted between validation and installation | **closed** | the retained stage descriptor, name, inode, mode, link count, size, and bytes are rebound immediately before the no-replace hard link; substitution rejects before destination creation |
| manifest temporary could be substituted between validation and replacement | **closed** | the retained manifest-stage identity and bytes are rebound immediately before `os.replace`; substitution preserves the old manifest |
| exact-unregistered recovery was impossible or normalized incorrectly | **closed** | the exact two-directory-plus-target intermediate passes a real isolated recovery test with the same authority digest; a nonexact target rejects |
| directory inventory was not closed | **closed** | only the exact base, base-plus-Chapter-6, and base-plus-synthesis lifecycle inventories are admitted; an arbitrary empty directory rejects |
| sources and non-target data were not checked immediately before manifest mutation | **implemented but not integrated** | the recheck exists, but the new blocker in Section 13 makes every non-noop authorized call fail inside that recheck |
| final validation was not a complete fresh-process validation under the publication lock | **closed structurally** | the registered validator calls the complete authorized preparation path, rebuilds twice, and is invoked in a fresh process while the lock remains held |
| handled stage failures did not reliably clean the invocation-owned stage | **closed** | cleanup is identity-bound and the hostile handled-failure cases pass |
| exact stage mode depended on umask | **closed** | retained-descriptor `fchmod` and readback enforce artifact mode `0600`; restrictive-umask tests pass |
| units, conventions, and axes were not fixed scientific semantics | **closed** | all 213 members are checked against the exact expected metadata digest and two actual assembler schemas |
| verification-only behavior was only nominal | **closed** | direct authorization-independent verification is reachable, deterministic, and has no publication primitive |

“Implemented but not integrated” is intentionally not an acceptance. A
security check that makes the intended transition unreachable does not
satisfy the publication contract.

## 13. New blocking finding: the publisher quarantines its own manifest stage

**Priority:** P0 — deterministic functional publication blocker

The authorized non-noop path has the following unavoidable sequence:

1. `_replace_manifest` creates a regular temporary in canonical `data/` using
   `MANIFEST_TEMP_PREFIX`;
2. while that invocation-owned temporary still exists, `_replace_manifest`
   calls `_revalidate_source_data_before_manifest`;
3. that recheck calls `_snapshot_data`;
4. `_snapshot_data` first calls `_inventory_quarantine(DATA_PATH)`;
5. `_inventory_quarantine` classifies every regular file beginning with
   `MANIFEST_TEMP_PREFIX` as quarantined;
6. `_snapshot_data` therefore raises before
   `_replace_validated_manifest_stage` is reachable.

This is not a theoretical name collision. The temporary created by step 1 is
the exact object that necessarily triggers step 5. An isolated nonmocked call
to the real `_replace_manifest(..., authority=authority)` produced:

```text
raised=PublicationStateError
message=crash-left publication temporary is quarantined; separately review design/chapter06_publication_quarantine_cleanup_acceptance.json
manifest_preserved=true
temporary_count=0
```

The probe used an isolated temporary repository, a valid before/after
manifest, the expected candidate artifact, exact patched authority/review
bytes, and the real `_replace_manifest` and `_snapshot_data`. Only unrelated
fixed-source verification was isolated. The original manifest was preserved
and the invocation-owned temporary was cleaned, so the failure is fail-safe;
it is nevertheless a complete reachability failure for every authorized
first registration.

The focused tests do not expose this integration failure:

- the successful and substitution manifest-replacement tests pass
  `authority=None`, which skips the source/data revalidation;
- the immediate source/data recheck test invokes
  `_revalidate_source_data_before_manifest` separately and mocks
  `_snapshot_data`, so no real manifest temporary exists during inventory.

The candidate report describes the immediate re-snapshot as repaired, but
does not account for its collision with the invocation-owned stage.

## 14. New race finding: no canonical pathname rebind after mutation and fsync

**Priority:** P1 — post-mutation parent/name identity is not closed

The repaired code correctly retains directory descriptors and flushes the
directory inode that was actually mutated. That is an important improvement.
It does not then prove that this retained inode is still the directory named
by the canonical pathname after the mutation and `fsync`.

### 14.1 Artifact installation

An isolated wrapper performed the real no-replace link, then renamed the
destination parent and created a replacement directory at the canonical
pathname before control returned to `_install_no_replace`. The helper
reported:

```text
raised=NONE
result=installed
old_mutated_inode_fsynced=true
replacement_inode_fsynced=false
canonical_target_exists=false
moved_target_exists=true
```

The correct old directory inode was flushed, and no pathname-reopened inode
was substituted for that durability operation. The defect is instead that
the helper performs its final readback through the retained descriptor and
returns `installed` without re-binding that descriptor to the current
canonical parent name. A later canonical artifact check would reject before
manifest replacement, so this probe does not demonstrate manifest
corruption. It does demonstrate that the mutation helper's local success
contract and canonical parent-race closure are false.

The current test swaps the destination parent *before* the no-replace link and
correctly proves pre-mutation rejection. It does not swap the parent after the
link and before the helper's durability/readback boundary.

### 14.2 Manifest replacement

A second isolated wrapper performed the real `os.replace`, then renamed
canonical `data/` and created a replacement `data/` before returning. With
`authority=None` solely to isolate the replacement/durability boundary, the
helper reported:

```text
raised=NONE
old_mutated_inode_fsynced=true
replacement_inode_fsynced=false
canonical_manifest_exists=false
moved_manifest_is_intended=true
```

Again, the correct mutated inode was flushed. The retained-descriptor
readback validated the moved manifest and the helper returned normally
without proving that the retained `data` descriptor was still named by
canonical `data/`. Fresh complete validation would later reject, but the
manifest mutation helper itself did not close the post-replace parent race.
The existing retained-data-descriptor test covers only the normal case.

## 15. Passing evidence retained by this review

The rejection above is narrow and evidence-based; it does not erase the
substantial repaired surface:

| check | exact result |
| --- | --- |
| focused repaired synthesis publisher suite | `51 passed in 16.01s` |
| complete synthesis worker/assembler/writer/publisher chain | `71 passed in 78.49s` |
| Python byte compilation | pass |
| Ruff lint | pass |
| Ruff formatting check | pass |
| two direct verification-only calls in one process | pass; byte-identical reports |
| direct `--dry-run` without authority | reject, exit `2`, no mutation |
| direct `--publish` without authority | reject, exit `2`, no mutation |
| shared canonical `data/` lock probed from a second process | contending lock blocked |

Both verification-only calls reconstructed exactly:

| property | exact result |
| --- | --- |
| archive SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| archive bytes | `1,294,865` |
| archive members | `213` |
| target state | absent |
| canonical data aggregate | `56fb4fb6be3a0a6733b0427ddc3947cb2deee41cf307dfb1bb073c93822ad910` |
| canonical manifest SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |

The direct mutating commands rejected with the same structured reason:

```json
{"reason":"design/chapter06_synthesis_publication_acceptance.json is missing, symlinked, or unreadable","status":"REJECT"}
```

## 16. Canonical zero-delta evidence after hostile probes

| property | exact result |
| --- | --- |
| `data/` directories, including `data` | `13` |
| `data/` regular files | `39` |
| canonical data aggregate | `56fb4fb6be3a0a6733b0427ddc3947cb2deee41cf307dfb1bb073c93822ad910` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| `data/README.md` SHA-256 | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |
| synthesis destination | absent |
| synthesis manifest entry | absent |
| synthesis authorization | absent |
| synthesis authorization review | absent |
| crash-left synthesis stage/temporary | absent |

All mutating hostile probes used isolated temporary roots. The three requested
candidate hashes in Section 11 remained exact throughout the review.

## 17. Required closure before another independent review

1. Make the immediate source/data re-snapshot compatible with exactly one
   invocation-owned manifest temporary. Any allowance must be bound to the
   retained descriptor, exact name, device/inode, link count, mode, size,
   intended bytes, and SHA-256. It must never be a blanket prefix exception.
   An equally safe ordering may re-snapshot before stage creation, provided a
   final closed intermediate-state validator remains immediately bound to the
   replacement primitive.
2. Add a nonmocked integration regression with non-`None` authority that
   creates the real manifest temporary, performs the real canonical snapshot,
   and reaches replacement. Preserve hostile cases proving that source or
   non-target data changes after staging still reject before replacement.
3. After each artifact link/unlink plus directory `fsync`, re-bind the retained
   destination-parent descriptor to the current canonical parent pathname and
   revalidate the final canonical target name/inode before returning success.
4. After manifest replacement plus directory `fsync`, re-bind the retained
   `data` descriptor to the current canonical `data/` pathname and revalidate
   canonical `data/MANIFEST.json` before returning.
5. Add post-mutation parent-swap regressions for both helpers. The helpers
   must reject within their own call; a later fresh validator is useful
   defense in depth, not a substitute for closing their local mutation
   boundary.
6. Re-run the focused and complete synthesis chains, direct verification-only
   reconstruction, shared-lock contention probe, all negative publication
   commands, and independent canonical zero-delta inventory.
7. Freeze the new implementation, test, and candidate-report hashes and
   request a new independent hostile review. This rejected snapshot must not
   self-authorize or be named in a detached publication authority.

## 18. Re-audit decision

**REJECT**

The repaired publisher now has exact C-byte member hashes, strong
retained-stage binding, closed lifecycle inventories, fixed scientific
metadata, deterministic verification-only behavior, and a complete
fresh-process registered validator. Acceptance is still impossible because
the authorized first-publication path deterministically quarantines its own
manifest temporary. The missing post-mutation canonical parent/name rebinds
also leave two mutation helpers able to report success after their retained
directories have been detached from the canonical path.

The synthesis candidate bytes remain accepted and unchanged. This decision
rejects only the repaired publication mechanism reviewed here.

<!-- END REPAIRED SYNTHESIS PUBLISHER RE-AUDIT 2026-07-30 -->

<!-- BEGIN FINAL SYNTHESIS PUBLISHER REPAIR QA 2026-07-30 -->

## 19. Final independent repair QA

Date: 2026-07-30  
Reviewer role: independent ordinary local-software QA under the frozen
Section 12 threat boundary  
Disposition: **ACCEPT the exact repaired publisher candidate**

This section is append-only. Sections 1–18 remain the immutable rejection
history for earlier publisher bytes. This decision reviews a new exact
implementation, focused suite, and candidate report. It accepts no detached
authorization, authorization review, canonical artifact, manifest mutation,
cleanup, or postpublication record.

### 19.1 Exact review boundary

Every assigned object was a regular, nonsymlink, single-link file. The exact
reviewed identities were:

| reviewed object | SHA-256 |
| --- | --- |
| `scripts/build_chapter06_synthesis_golden.py` | `a9cc41f6b0862aa6857b0fdc060c6b6e22b6a2bd094c5cf1b9bc9b04463f6469` |
| `tests/test_chapter06_synthesis_golden_publisher.py` | `2d7fef267bbf3e71cd4267d7cdb8ddd1c0f1047b92cf212fef909fdce15e00b3` |
| `design/chapter06_synthesis_golden_publisher_candidate.md` | `d548e460c76ad1ef488da7f5fd1c0765df5e40b8da2ef69941614d0b54c69a53` |
| complete synthesis publisher audit before this append | `7790c4b7366a05456d3c5a947e6ad3b3890c16c07e7681e5acc84e86c409a15c` |
| `design/chapter06_publisher_threat_model_adjudication.md` | `4ed79dd4ff622abbeb6106a58b1b7e8c05a10c2ff87cfeb0dd6ee06e24e8e9fc` |
| shared publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| synthesis candidate-byte acceptance | `434088cff95ed60d65dc6c9749d18c2e74e45d114787c03728ed7ae9cf0bd9c9` |

The accepted synthesis candidate bytes remain independently unchanged:

| property | exact value |
| --- | --- |
| archive SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| archive bytes | `1,294,865` |
| archive members | `213` |
| canonical destination | `data/golden/payne_zero/chapter06/synthesis/chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz` |
| repository role | `golden` |

No reviewed candidate identity changed during QA.

### 19.2 Closure of the deterministic manifest-temporary blocker

The P0 in Section 13 is closed. The immediate source/data snapshot no longer
classifies its own invocation's live manifest temporary as unrelated crash
debris.

The repair is deliberately narrower than a filename-prefix exception.
`OwnedManifestTemporary` carries the exact stage path, still-open stage
descriptor, still-open canonical `data` descriptor, retained creation
metadata, intended bytes, intended SHA-256, and manifest mode.
`_validate_owned_manifest_temporary` requires:

- one literal manifest-temporary child name under canonical `data`;
- retained-parent rebind to the current canonical `data` inode;
- exact device/inode agreement among the creation record, retained stage
  descriptor, and no-follow named entry;
- regular, single-link type with exact owner, group, mode, and size;
- exact intended-hash agreement; and
- complete intended bytes through both the retained descriptor and a
  separate no-follow named read.

The quarantine inventory must equal exactly the one resulting `FileFact`.
The live temporary remains present in the closed regular-file inventory and
allowed nonmanifest set; only that identity-bound path is omitted from the
stable authorization aggregate. It is revalidated again after the complete
tree walk. A missing, substituted, altered, or second temporary is never
normalized.

The focused suite's real integration harness constructs an isolated complete
M1 state with a registered atmosphere fixture, closed README support, exact
unregistered synthesis target, real detached record bytes, an authority-bound
source/data digest, and a non-`None` `Authority`. It then calls the real
`_replace_manifest`, `_snapshot_data`, temporary creation/validation, and
descriptor-relative `os.replace`; neither the snapshot nor replacement helper
is mocked.

The independently selected integration matrix passed:

```text
authorized non-None authority transition reaches real replacement
source changed after staging rejects before replacement
non-target README changed after staging rejects before replacement
second manifest-prefixed temporary rejects before replacement
post-link destination-parent swap rejects within installation helper
post-replace data-parent swap rejects within manifest helper

6 passed in 0.23s
```

The success case reached `os.replace` exactly once, installed exact intended
manifest bytes, preserved the exact artifact, and left no temporary. Every
pre-replacement mutation case made zero replacement calls and preserved exact
old manifest bytes. The second-temporary case removed only the
invocation-owned temporary and left the foreign object untouched for
quarantine.

### 19.3 Closure of both post-mutation canonical-rebind findings

The artifact helper now `fsync`s the retained destination-parent descriptor
after the no-replace decision, reopens the canonical parent and requires the
same directory inode, removes only the identity-bound stage, `fsync`s and
rebinds again, then requires the retained-directory target and canonical
pathname to identify the same exact single-link artifact before success.

The post-link hostile regression performs the real link, moves the mutated
parent, and creates a replacement canonical directory before the wrapped
syscall returns. The helper flushes the old mutated directory and rejects at
its immediate canonical rebind. The installed target and stage link remain
in the moved directory as evidence; the helper does not delete a canonical
target or claim success.

The manifest helper marks replacement complete immediately after the real
`os.replace`, `fsync`s the retained `data` descriptor, rebinds it to canonical
`data`, proves the retained temporary descriptor and retained-directory
manifest name are the exact new inode with exact mode, size, and bytes, and
then independently binds canonical `data/MANIFEST.json` to that inode.

The post-replace hostile regression performs the real replacement, moves the
mutated `data` directory, and creates a replacement canonical directory. The
helper flushes the actual mutated directory and rejects its canonical rebind.
The intended new manifest and exact artifact remain in the moved directory.
There is no old-manifest restoration, canonical-target deletion, or false
helper success.

### 19.4 Former finding matrix

All earlier findings remain closed in the exact accepted bytes:

| finding or gate | final result |
| --- | --- |
| member hashes accidentally included dtype/shape metadata | closed; SHA-256 is over contiguous C-order array bytes only |
| artifact-stage substitution before the immediate boundary | closed; rejects before target creation |
| manifest-temporary substitution before the immediate boundary | closed; rejects while preserving old manifest bytes |
| exact-unregistered recovery normalization | closed; exact state shares the reviewed authority digest, nonexact state rejects |
| incomplete directory inventory | closed for base, Chapter-6, and synthesis lifecycle states |
| immediate source/non-target-data recheck | closed, including the real owned-temporary integration path |
| complete fresh-process validation under the common lock | closed |
| handled invocation-stage cleanup | closed and inode-bound |
| exact `0600` artifact mode under restrictive umask | closed |
| units, conventions, axes, and lexical 213-member metadata | closed by exact reviewed semantics/digest |
| authorization-independent verification-only path | closed; no publication call edge |
| shared atmosphere/synthesis lock object | closed; both use canonical `data` |
| retained mutation-parent durability | closed; exact retained descriptors are fsynced |
| post-mutation canonical parent/name rebinds | closed by the two finite hostile regressions above |

No new in-scope P0, P1, or P2 finding was identified.

### 19.5 Executed QA

| check | independent result |
| --- | --- |
| exact publisher focused suite | `57 passed in 64.37s` |
| complete synthesis worker/assembler/writer/publisher chain | `77 passed in 122.89s` |
| selected real-authority and post-mutation repair matrix | `6 passed in 0.23s` |
| Python byte compilation | pass |
| Ruff lint | pass |
| Ruff formatting check | pass; both files already formatted |
| scoped Git whitespace check | pass |
| fixed-root live `--verify-only` | pass; two builds, exact bytes, zero repository delta |
| fixed-root `--dry-run` without authority | reject, exit `2`, before mutation |
| fixed-root `--publish` without authority | reject, exit `2`, before mutation |

The live fixed-root result was:

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

Both authorization-bearing command modes returned the same structured,
fail-closed reason:

```json
{"reason":"design/chapter06_synthesis_publication_acceptance.json is missing, symlinked, or unreadable","status":"REJECT"}
```

### 19.6 Canonical zero-delta proof

The same closed snapshot was observed before and after focused tests,
complete-chain tests, selected hostile tests, live reconstruction, and both
negative authorization commands:

| canonical property | exact unchanged result |
| --- | --- |
| stable data aggregate | `56fb4fb6be3a0a6733b0427ddc3947cb2deee41cf307dfb1bb073c93822ad910` |
| data directories, including `data` | `13` |
| data regular files | `39` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| `data/README.md` SHA-256 | `1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b` |
| synthesis destination | absent |
| synthesis manifest entry | absent |
| detached synthesis authorization | absent |
| detached synthesis authorization review | absent |
| synthesis artifact/manifest quarantine | absent |

All mutation tests used disposable isolated roots. No canonical data,
manifest, authority, contract, governing record, Payne Zero source, paper
source, or accepted external byte was modified.

### 19.7 Threat-boundary statement

This acceptance applies the exact design adjudication
`4ed79dd4ff622abbeb6106a58b1b7e8c05a10c2ff87cfeb0dd6ee06e24e8e9fc`.
The reviewed scientific transaction covers cooperative publishers and
ordinary filesystem substitutions through the shared canonical-`data` lock,
immediate descriptor/name/byte checks, atomic no-replace link and manifest
replacement, retained-directory durability, canonical rebind, and complete
postvalidation.

It does not claim that Python repository code forms a security boundary
against arbitrary same-process code injected literally inside `os.link` or
`os.replace`, or against an unrestricted same-account process that ignores
the lock and wins the interval after the last production check. The earlier
syscall-entry trace remains valid evidence of that name-resolved-system-call
limitation, but the frozen Section 12 threat model excludes that actor from
the scientific reproducibility claim. This decision neither reopens that
criterion nor misstates the Darwin manifest primitive as descriptor-bound.

### 19.8 Final decision

**ACCEPT** exact publisher
`a9cc41f6b0862aa6857b0fdc060c6b6e22b6a2bd094c5cf1b9bc9b04463f6469`,
focused tests
`2d7fef267bbf3e71cd4267d7cdb8ddd1c0f1047b92cf212fef909fdce15e00b3`,
and candidate report
`d548e460c76ad1ef488da7f5fd1c0765df5e40b8da2ef69941614d0b54c69a53`
as the Chapter 6 synthesis golden publisher implementation under the
adjudicated ordinary local-software threat boundary.

This is implementation acceptance only. Publication remains blocked until
the required prior atmosphere M1 state exists and a separate detached
synthesis authorization plus independent exact-record review bind these
accepted publisher bytes, the then-current manifest, the accepted candidate
bytes, and the complete source/data snapshot. This audit itself grants no
permission to create those records or mutate the canonical repository.

<!-- END FINAL SYNTHESIS PUBLISHER REPAIR QA 2026-07-30 -->
