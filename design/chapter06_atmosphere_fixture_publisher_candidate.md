# Chapter 6 atmosphere-fixture fixed-path publisher repair candidate

Date: 2026-07-30

Implementation role: publisher author; not independent reviewer

Disposition: **IMPLEMENTATION CANDIDATE ONLY — NOT PUBLISHER ACCEPTANCE,
NOT AUTHORIZATION, NO PUBLICATION**

## 1. Candidate boundary

This report accompanies exactly two implementation objects:

| candidate object | SHA-256 |
| --- | --- |
| `scripts/build_chapter06_atmosphere_fixture.py` | `5de0169339b8130d45c8e610684de67c2d6f3c6c62c5298610d0f705ebe0b363` |
| `tests/test_chapter06_atmosphere_fixture_publisher.py` | `00c09576b00420bfaa72f058a1b70fab1dedb4e85e04326ca83c7bac0b6ca876` |

This repaired snapshot supersedes the audited implementation
`d9cb38e5…`/`0653f399…`. It responds narrowly to the scope-aware independent
audit at SHA-256
`b77de7e2f543ace58f054d08f5ad0f596f0e80decd4a08b5f63ffa0f4680585c`.
That audit remains an immutable rejection of the former bytes and is not
treated as acceptance of this repair. The implementation also preserves the
finite retained-directory and common-lock repairs assessed in the earlier
audit history.

The implementation binds:

| accepted input | exact SHA-256 |
| --- | --- |
| final two-lane publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| final contract independent audit | `fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc` |
| final atmosphere candidate-byte acceptance | `8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71` |
| final synthesis candidate-byte acceptance (common two-writer gate) | `434088cff95ed60d65dc6c9749d18c2e74e45d114787c03728ed7ae9cf0bd9c9` |
| accepted atmosphere writer | `0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538` |
| accepted writer tests | `c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a` |
| accepted writer candidate | `e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee` |
| accepted writer audit | `b946dcef0beeacf49a3da9ac036e21af7cd7b44d15092639cd3be744fb42f0f9` |

The sole production repository and destination are:

```text
repository
  /Users/ysting/stellar-spectroscopy-from-scratch-gpu

destination
  data/fixtures/chapter06_atmosphere_one_line_inputs.npz

role
  fixture
```

The three public operations are all zero-argument:

```text
verify_only()
authorized_dry_run()
publish()
```

There is no root, destination, output, force, replace, repair, merge, cleanup,
or partial-lane argument. The CLI exposes only `--verify-only`,
`--authorized-dry-run`, and `--publish`; its hidden fresh-process validator
also accepts no path.

## 2. Present fail-closed state

The earlier publisher audit exists with disposition `REJECT`. The detached
authorization, authorization-record review, artifact, and postpublication
objects remain absent:

```text
design/chapter06_atmosphere_fixture_publisher_independent_audit.md
  present; SHA-256 b77de7e2…; immutable history rejects the former bytes for
  the finite late-authority boundary P0-S1; that rejection is not rewritten
  by this report

design/chapter06_atmosphere_fixture_publication_acceptance.json
design/chapter06_atmosphere_fixture_publication_record_review.json
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
design/chapter06_atmosphere_fixture_postpublication_audit.md
```

There is no external lock file. The sole lock object is the already existing
canonical `data` directory inode, shared with the synthesis lane. `publish()`
checks the future control objects before candidate construction, lock
acquisition, parent creation, staging, or any repository mutation. A
canonical invocation therefore returns:

```text
AuthorizationError
  publication is disabled before candidate, lock, parent, or stage creation;
  absent: detached publication authorization,
          authorization-record review
```

The production atmosphere lane creates no directory: its exact parent
`data/fixtures` must already be a nonsymlink directory. The contract's
allowlisted nested-directory creation belongs only to the later synthesis
lane.

## 3. Verification-only construction

`verify_only()` has no call edge to the lock, stage, hard-link, unlink, or
manifest-replacement helpers. It:

1. no-follow reads and hashes the complete local common-writer and atmosphere
   trust chain;
2. reruns the accepted worker's complete deep gate over the pinned commit,
   all frozen Python sources, staged converter dependencies, and all eleven
   dynamic data inputs;
3. records the closed `data/` directory, regular-file, manifest-role, and
   nonmanifest-support inventories;
4. starts two unrelated top-level CPython processes;
5. lets each accepted writer own its two fresh scientific children and two
   distinct external cache roots;
6. requires complete top-level archive and stable-summary equality;
7. treats the returned ZIP/NPY as untrusted;
8. requires the exact 19-member allowlist, shape, dtype, C-byte hash,
   C-contiguity, finite floating values, science invariants, and schema;
9. canonically re-encodes every NPY member and the complete archive; and
10. repeats the source and data snapshots and requires exact equality.

The accepted physical payload fingerprint is not redefined. It covers both
the nineteen stored arrays and transient `payload__` evidence. Each accepted
writer recomputes that exact fingerprint before return; the publisher checks
the writer summary and independently checks all stored array and archive
bytes. It does not invent a different archive-only quantity with the same
name.

### 3.1 Exact final dry-run report

The final live run returned:

```json
{
  "report_schema_version": 1,
  "report_kind": "chapter06_atmosphere_fixture_publisher_dry_run",
  "mode": "candidate-verification-only",
  "repository": "/Users/ysting/stellar-spectroscopy-from-scratch-gpu",
  "destination": "data/fixtures/chapter06_atmosphere_one_line_inputs.npz",
  "manifest_role": "fixture",
  "publication_authorized": false,
  "publication_performed": false,
  "lock_created": false,
  "stage_created": false,
  "manifest_mutated": false,
  "top_level_writer_invocations": 2,
  "top_level_archives_byte_identical": true,
  "accepted_writer_topology": {
    "fresh_child_processes": true,
    "distinct_origins": true,
    "distinct_external_cache_roots": true,
    "cache_roots_empty_before": true,
    "cache_entries_after_each_child": 37,
    "cache_package_directories_after_each_child": 1,
    "cache_nbi_files_after_each_child": 18,
    "cache_nbc_files_after_each_child": 18,
    "cache_symlinks_after_each_child": 0,
    "cache_other_files_after_each_child": 0,
    "cache_roots_disposed_before_return": true
  },
  "archive": {
    "bytes": 363050,
    "sha256": "1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff",
    "member_count": 19,
    "scientific_array_bytes": 357984,
    "archive_kind": "atmosphere_one_line_input_fixture",
    "fixture_capture_schema_version": 1,
    "archive_contains_embedded_schema_version": false,
    "npy_member_format_version": "2.0",
    "scientific_fixture_schema_digest": "f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698",
    "scientific_payload_fingerprint": "f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663",
    "canonical_decode_validate_reencode": true
  },
  "trust_identities": {
    "publisher_contract": {
      "path": "design/chapter06_lane_artifact_publisher_contract.md",
      "sha256": "3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b"
    },
    "publisher_contract_audit": {
      "path": "design/chapter06_lane_artifact_publisher_contract_rebind_independent_audit.md",
      "sha256": "fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc"
    },
    "candidate_byte_acceptance": {
      "path": "design/chapter06_atmosphere_fixture_byte_acceptance.md",
      "sha256": "8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71"
    },
    "common_synthesis_candidate_byte_acceptance": {
      "path": "design/chapter06_synthesis_candidate_byte_acceptance.md",
      "sha256": "434088cff95ed60d65dc6c9749d18c2e74e45d114787c03728ed7ae9cf0bd9c9"
    },
    "writer": {
      "path": "scripts/chapter06_atmosphere_fixture_writer.py",
      "sha256": "0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538"
    },
    "writer_tests": {
      "path": "tests/test_chapter06_atmosphere_fixture_writer.py",
      "sha256": "c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a"
    },
    "writer_candidate": {
      "path": "design/chapter06_atmosphere_fixture_writer_candidate.md",
      "sha256": "e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee"
    },
    "writer_acceptance": {
      "path": "design/chapter06_atmosphere_fixture_writer_independent_audit.md",
      "sha256": "b946dcef0beeacf49a3da9ac036e21af7cd7b44d15092639cd3be744fb42f0f9"
    },
    "publisher": {
      "path": "scripts/build_chapter06_atmosphere_fixture.py",
      "sha256": "5de0169339b8130d45c8e610684de67c2d6f3c6c62c5298610d0f705ebe0b363"
    },
    "publisher_tests": {
      "path": "tests/test_chapter06_atmosphere_fixture_publisher.py",
      "sha256": "00c09576b00420bfaa72f058a1b70fab1dedb4e85e04326ca83c7bac0b6ca876"
    }
  },
  "source_snapshot_sha256": "7d853d574f8aacce4eb04237df9509facd743ae54c1ca791ef1e7fd5d65ead02",
  "data_snapshot_sha256": "5be325c0200cf711ce13daa3dc96a47bdd8780c8de1640efe23827196de5b84a",
  "manifest": {
    "path": "data/MANIFEST.json",
    "sha256": "d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a",
    "ordered_entry_path_sha256": "d63aa6dfd9209f21172c6ccf721cffc895835b8ee15a4ed160c5bebe0851b1aa",
    "encoding": "preserve-unsorted-append-only-indent-2"
  },
  "destination_state": "absent",
  "authorization_state": "absent-and-required-before-publication",
  "decision": "VERIFIED_CANDIDATE_ONLY_NOT_AUTHORIZED"
}
```

The nondeterministic origins, PIDs, cache paths, and process-evidence hashes
are checked inside each build but reduced to Boolean/count decisions in the
report.

## 4. Strict future authority boundary

The detached authorization is duplicate-free UTF-8 JSON. The implementation
defines and requires this exact top-level key order:

```text
schema_version
record_kind
lane
manifest_role
payne_zero_commit
writer
writer_tests
writer_candidate
writer_acceptance
candidate_byte_acceptance
publisher
publisher_tests
publisher_contract
publisher_contract_audit
publisher_acceptance
artifact
manifest
manifest_entry_template
manifest_entry_template_sha256
source_snapshot_sha256
data_snapshot_sha256
destination_parent_policy
no_replace_primitive
```

Every named identity object has exactly `path`, `sha256`. The artifact object
has exactly:

```text
path
filename
bytes
sha256
member_count
archive_kind
fixture_capture_schema_version
archive_contains_embedded_schema_version
npy_member_format_version
scientific_fixture_schema_digest
scientific_payload_fingerprint
```

The prepublication-manifest object has exactly:

```text
path
prepublication_sha256
schema_version
payne_zero_commit
ordered_entry_path_sha256
destination_entry_absent
```

The separate review uses the contract's exact order and lane-specific kind:

```text
schema_version
record_kind
authorization_path
authorization_sha256
candidate_byte_acceptance_path
candidate_byte_acceptance_sha256
publisher_acceptance_path
publisher_acceptance_sha256
manifest_entry_template_sha256
disposition
```

Unknown, missing, duplicate, reordered, wrong-typed, malformed-hash, stale,
or cycle-forming fields fail. The authorization has no self hash, review
hash, realized-entry hash, postmanifest hash, or postpublication-audit hash.

## 5. Manifest template and exact bytes

The publisher preserves the live manifest's existing unsorted sequence and
all existing object-key orders. It uses exactly:

```python
(json.dumps(
    manifest,
    indent=2,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=False,
    separators=(",", ": "),
) + "\n").encode("utf-8")
```

The current prepublication locks are:

```text
entries
  37

manifest bytes
  1,087,741

manifest SHA-256
  d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a

ordered entry-path SHA-256
  d63aa6dfd9209f21172c6ccf721cffc895835b8ee15a4ed160c5bebe0851b1aa
```

The future authorization embeds the exact ordered entry template `T`.
`arrays` uses lexical archive-member order. Every array record has exact key
order:

```text
shape, dtype, unit, sha256, axes, ownership
```

The only late values are:

```text
publication_acceptance_sha256
  __LATE_BOUND_AUTHORIZATION_SHA256__

publication_record_review_sha256
  __LATE_BOUND_RECORD_REVIEW_SHA256__
```

Each literal occurs once and only as the complete value of its named field.
Realization permits exactly those two scalar replacements. The intended
manifest is `M` with `E` appended last. Removing the last entry and
reserializing must reproduce exact `M` bytes and hash.

The final template digest is intentionally unresolved here because `T` must
bind the still-future publisher-acceptance SHA-256. It is computed and checked
only once that independent object exists, then frozen by `A` and `R`. This
candidate does not substitute a placeholder for an acceptance object the
contract requires to be final.

## 6. Authorized file and manifest transition

Only after all authority gates pass:

1. `data` is opened component by component from the fixed repository root
   with directory-relative no-follow semantics; that exact retained directory
   descriptor is verified against the canonical named inode and held with
   exclusive `flock` for the complete transition;
2. lock failure rejects before any stage or target write; both atmosphere and
   synthesis contend on this same stable canonical directory object;
3. the retained `data` descriptor/name identity and all
   source/data/authority snapshots are rechecked;
4. any hidden artifact-stage or manifest-temporary prefix is inventoried and
   hard-stops as quarantine;
5. `data/fixtures` is likewise opened and retained by its exact descriptor;
   an artifact stage is created relative to it with
   `O_EXCL | O_NOFOLLOW`, `fchmod(0600)`, and same-device proof;
6. one exact write is required—short writes are rejected—then file `fsync`,
   descriptor readback, and candidate validation run; the named stage's
   retained inode, one-link state, exact mode, size, SHA-256, and bytes are
   rechecked immediately before install;
7. `linkat`/`os.link` from the retained parent descriptor is the atomic
   create-if-absent decision; no target is renamed, replaced, deleted, or
   pre-unlinked;
8. the exact retained parent descriptor—not a reopened pathname—is
   `fsync`ed and rebound to its canonical name; the invocation-owned stage
   inode is rechecked and unlinked through that descriptor, which is
   `fsync`ed and rebound again;
9. the final target must be the installed stage inode when newly linked, then
   one-link, exact, and fully valid after stage cleanup;
10. source, nontarget data, authority, manifest inode/hash, and candidate are
   rechecked;
11. exact `N` is written relative to the retained locked `data` descriptor to
    a create-exclusive no-follow temporary beside
    `MANIFEST.json`, `fsync`ed, descriptor-read back, reparsed, and compared;
12. after staging, the exact accepted source snapshot and the complete closed
    data snapshot are taken again; only the identity-bound invocation
    temporary and exact unregistered target are normalized from the aggregate;
13. both detached authority records are no-follow reopened and strictly
    revalidated against their exact schema, bindings, key order, bytes,
    hashes, metadata, and the prepared in-memory authority; a second exact
    no-follow byte/identity reread closes the validation work itself;
14. the exact artifact bytes, one-link target, template digest, and
    reconstructed intended `N` are revalidated against the prepared candidate
    and temporary;
15. the named manifest temporary's inode, one-link state, exact inherited
    mode, size, SHA-256, and bytes plus the retained `data` name/inode are
    rechecked at the replacement boundary;
16. only `MANIFEST.json` is atomically replaced through the retained locked
    descriptor, preserving its mode; that same descriptor is `fsync`ed and
    rebound to its canonical name; and
17. while the data lock remains held, a fresh production interpreter
    no-follow reopens the artifact and manifest and validates the complete
    source/data/authority/entry/role/member registered state.

An existing exact, one-link unregistered target is the only recoverable
partial state. An exact registered `N` is an inode-preserving validation-only
no-op. A nonexact target, multilink target, registered-missing file, duplicate
entry, stale `M`, changed parent, or entry/file mismatch fails closed.

If the artifact link succeeds but manifest registration fails, the exact file
is left inert and unregistered. The publisher never rolls back by deleting
it. A later invocation may register it only under the same valid
authorization and reconstructed exact `M`.

Crash-left hidden names are never decoded, adopted, promoted, or deleted.
There is no cleanup API. Cleanup remains a distinct future reviewed
`unlink_only` lifecycle.

## 7. Adversarial matrix and results

All production mutations were tested only in disposable roots reached by
monkeypatching the private runtime-path provider. Public production APIs
remain path-free.

| family | exercised result |
| --- | --- |
| surface/path | zero-argument APIs; no root/destination/force/replace/repair options; absolute, `..`, backslash, percent, Unicode, and empty-component identities rejected |
| JSON | duplicate keys, nonfinite values, unknown/missing/reordered fields, and strict review/authorization schemas rejected |
| source/data | live manifest exact round trip, 37-entry unsorted sequence, exact path digest, no-follow symlink rejection, closed support and directory inventories, and unexpected empty-directory rejection |
| recovery snapshot | exact unregistered target is normalized to the authorization-bound absent-target aggregate; a nonexact target remains a hard stop |
| candidate/template | lexical array records, exact six-key array schema, independent C-byte-only hashes, exact nonempty scientific units/axes, two and only two late placeholders, and exact two-field realization |
| absent/stale authority | canonical publication stops before candidate/lock/stage; mutated artifact binding stops before target/stage |
| verification-only | mutation helpers replaced with fail-on-call sentinels; two reports remained byte-identical |
| I/O | injected short write, file-`fsync`, and descriptor-readback failures rejected; invocation-owned stage removed; ambient umask `0777` still produced exact `0600` |
| quarantine | crash-left apparent candidate hard-stopped and remained byte-identical |
| target race | nonexact winner preserved and rejected; exact one-link winner became no-op; exact multilink winner rejected |
| ordinary stage substitution | replacing the artifact-stage inode before the immediate retained-fd/name/byte boundary check rejected before target creation; replacing the manifest-temporary inode before that check rejected with original manifest bytes intact |
| parent race | forced rename/recreation after artifact link proved the exact mutated retained parent inode—not the replacement pathname inode—was `fsync`ed before rejection |
| manifest race | changed manifest inode/hash rejected; forced `data` rename/recreation after replacement proved the exact mutated retained inode was `fsync`ed; raced pre-existing bytes were preserved when replacement had not begun |
| pre-replace resnapshot | source or non-target data change after artifact install rejected; a data change after manifest staging also rejected before manifest replacement and cleaned the owned temporary |
| final authority boundary | post-snapshot authorization whitespace, coherent authorization+review rebinding, review-only whitespace, coherent template rebinding, target-byte mutation, and intended-`N` temporary mutation all rejected with zero manifest-replacement calls; exact old `M` survived, foreign mutation survived, and only the invocation-owned temporary was cleaned |
| lock | a second process locking canonical `data` directly could not acquire it while the atmosphere holder retained it; injected lock failure occurred before all writes |
| full transition | strict synthetic `A`/`R`, shared-directory lock, artifact hard-link install, exact append, late-hash realization, final authority rebind, exactly one real retained-descriptor manifest replacement, and complete internal fresh validation succeeded in an isolated tree |
| fresh validation | a child proved the canonical data lock remained held when fresh validation began; the complete internal registered-state validator passed |
| idempotence | second authorized run preserved artifact and manifest inodes and bytes and reran fresh validation |
| partial recovery | injected manifest failure left exact inert target and unchanged `M`; next authorized run registered that same file without replacement |
| live science | two unrelated accepted top-level writers, four fresh children, exact bytes/schema/cache decisions, and zero repository delta |

Focused default result:

```text
53 passed, 1 skipped in 0.50s
```

The skipped case is only the explicitly selected expensive live
reconstruction. The exact live test was then selected by itself:

```text
CHAPTER06_RUN_LIVE_PUBLISHER_TEST=1 \
PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q \
  tests/test_chapter06_atmosphere_fixture_publisher.py::\
test_live_verify_only_rebuilds_four_fresh_children_without_repo_delta

1 passed in 172.57s
```

The full atmosphere converter/worker/writer/publisher chain passed:

```text
94 passed, 1 skipped in 138.31s
```

The skip was again only the already-separately-run live reconstruction.
The live `verify_only()` regression used exact source and complete data
snapshots on both sides. It reproduced the report in Section 3.1 with source
snapshot `7d853d57…`, data snapshot `5be325c0…`,
manifest `d8f30e25…`, absent destination, and `zero_delta=true`.

Ruff check passed both implementation and test files. Ruff format check found
both already formatted. `py_compile` passed with bytecode writing disabled.
Scoped `git diff --check` passed.

## 8. Implementation corrections and independent-audit repair

Three candidate defects were found and corrected before freezing the hashes
in the former candidate:

1. one copied character in the synthesis compact-audit SHA-256 was corrected
   against the live file;
2. direct script execution was given a repository-root import bootstrap while
   canonical-path validation remains mandatory; and
3. an incorrect attempt to recompute the physical payload fingerprint from
   only the stored arrays was removed because that accepted fingerprint also
   covers deliberately ephemeral evidence.

The final live run passed only after these corrections.

The subsequent independent audit at `c46e76a4…` correctly rejected those
former bytes for two publication-boundary defects:

1. atmosphere used an external regular-file lock while the contract and
   synthesis lane lock canonical `data`, so the lanes did not contend; and
2. artifact and manifest mutations were performed through retained
   descriptors but directory durability reopened parent pathnames, permitting
   a forced parent swap to redirect `fsync`.

The `d9cb38e5…`/`0653f399…` snapshot removed the external lock completely,
locked the stable canonical `data` directory descriptor, and
retained/fsynced/rebound the exact directory descriptors used for every
artifact link, stage unlink, and manifest replace. It also imported every
applicable shared-risk lesson from the
synthesis publisher audit: exact stage/temp identity and bytes, C-byte member
hashes, closed directory inventory, exact recovery normalization, immediate
post-staging source/data resnapshot, complete fresh validation under lock,
handled-stage cleanup, fixed modes, exact units/axes, and a separate
authorization-independent verification path.

The scope-aware independent audit at `b77de7e2…` accepted those finite
repairs but correctly found P0-S1: the detached authorization and its review
were checked before manifest staging, not after the materially later
source/data resnapshot. Ordinary post-snapshot authorization whitespace and
a coherent authorization+review rebind could therefore replace `M` before
fresh validation rejected the registered state.

This candidate passes the exact prepared `Authority` and `Candidate` into the
manifest transition. After the final normalized source/data snapshot it
strictly reloads both detached records through component-wise no-follow paths,
rechecks their complete schemas and bindings, requires exact equality with
the prepared authority, repeats the exact named-file identity/byte reads,
revalidates the template digest and one-link artifact, reconstructs exact
intended `N`, and finally revalidates the intended temporary and retained
`data` immediately before `os.replace`. Six ordinary pre-syscall regressions
close the authority, review, template, target, and intended-`N` cases without
injecting code inside the syscall.

This section is an author report, not an independent acceptance decision.

## 9. No-publication proof

After the final live run:

```text
data/MANIFEST.json SHA-256
  d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a

data/README.md SHA-256
  1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b

canonical atmosphere fixture
  absent

manifest entries at destination
  0

publisher independent audit
  present only as rejection b77de7e2… of former candidate bytes

publisher acceptance for repaired bytes
  absent

detached authorization
  absent

authorization-record review
  absent

postpublication audit
  absent

external lock file
  not defined and not created

shared lock object
  existing canonical data directory; no new inode or byte

artifact stages / manifest temporaries
  0
```

The Payne Zero and paper trees were read-only inputs. This candidate wrote
neither. Runtime scientific writes were confined to accepted writer-owned
external cache roots, all disposed before return.

## 10. Required next gate

A new scope-aware independent review must audit the repaired exact script,
tests, and this report. The complete immutable audit history at SHA-256
`b77de7e2f543ace58f054d08f5ad0f596f0e80decd4a08b5f63ffa0f4680585c`
remains a rejection; the new review must not inherit acceptance from the
design adjudication below. That review must decide whether the shared
canonical-data lock, strict authorization schema, final detached-authority
rebind, no-follow path walk, source/data snapshots, no-replace hard-link
primitive, retained parent descriptors, exact-inode fsync ordering, manifest
byte policy, quarantine behavior, partial recovery, fresh-process validation
under lock, and expanded adversarial coverage satisfy contract `3a064f82…`.

Only a later accepted publisher audit may permit creation of a detached
authorization. That authorization must then be independently reviewed before
either authorized dry run or canonical write. This candidate is not that
review and grants no authority.

## 11. Adjudicated threat boundary

The design evidence governing this clarification is:

| object | SHA-256 |
| --- | --- |
| `design/chapter06_publisher_threat_model_adjudication.md` | `4ed79dd4ff622abbeb6106a58b1b7e8c05a10c2ff87cfeb0dd6ee06e24e8e9fc` |

That adjudication is design evidence only. It grants no publisher acceptance,
detached authorization, authorization review, publication, cleanup, or
postpublication authority. The implementation change in this candidate is
the finite P0-S1 repair described in Section 8; the arbitrary code-inside-
syscall limitation remains exactly the adjudicated out-of-scope boundary.

The frozen Section 12 boundary covers cooperative publishers and ordinary
filesystem substitutions through the common canonical-`data` lock, immediate
retained-descriptor/name/byte validation, the atomic namespace syscall,
`fsync` of the retained mutated directory descriptor, canonical-name rebind,
and complete postvalidation. The in-scope claims above remain intact:
cooperating lanes serialize on the same lock; substitutions visible before
the immediate boundary check reject before canonical mutation; target
creation is decided atomically without replacement; manifest replacement is
atomic; and retained-directory durability plus canonical rebind and fresh
readback detect the contract's enumerated parent, target, manifest, source,
and data races.

The boundary does not claim to prevent unrestricted same-account or
same-process code from executing arbitrary substitution literally inside
`os.link` or `os.replace`, after the last production check but before the
real syscall resolves its source name. Such a probe accurately demonstrates
the limitation of a name-resolved syscall, but it is outside the scientific
reproducibility claim. The publisher's post-mutation rebind and byte/state
validation detect the wrong result afterward when possible; this report does
not claim that this excluded actor is rejected before mutation or that
pre-existing canonical bytes are preserved against it.

This clarification does not introduce fail-closed platform gating that would
make the supported Darwin manifest transaction unusable, and it does not
pretend that Darwin offers a descriptor-bound atomic replacement of an
existing manifest. The scoped transaction continues to require atomic
create-if-absent artifact installation and atomic manifest replacement,
together with all immediate checks, retained descriptors, durability, and
postvalidation described above. A fresh independent audit must still decide
whether the exact frozen implementation satisfies that bounded contract.
