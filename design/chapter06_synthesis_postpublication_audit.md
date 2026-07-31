# Chapter 6 synthesis-oracle postpublication audit

Date: 2026-07-30

Reviewer role: independent read-only postpublication reviewer

Disposition: **ACCEPT the exact realized synthesis golden and M1-to-M2
manifest transition**

## 1. Review boundary

This audit inspected the realized repository state afresh. It did not rely on
the publication command's report and did not invoke `--publish`, `--dry-run`,
the public verification-only mode, a cleanup surface, or any mutation helper.

The review independently checked:

- the realized synthesis artifact and all 213 archive members;
- the detached authorization and its independent record review;
- the exact manifest-entry template and its two late substitutions;
- the complete M2 manifest/file join and append-last transition;
- the exact in-memory delete-last reconstruction of M1;
- the closed source/data inventory and normalized M1/M2 binding;
- the publisher's hidden complete registered-state validator in a fresh
  process; and
- the complete data file and directory state before and after that validator.

The only repository object written by this review is this report:

```text
design/chapter06_synthesis_postpublication_audit.md
```

No manifest, artifact, authority record, publisher, test, plan, notebook,
source tree, accepted external checkout, or paper source was changed.

## 2. Exact realized trust chain

Fresh byte reads produced the following SHA-256 identities:

| reviewed object | exact SHA-256 |
| --- | --- |
| two-lane publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| synthesis candidate-byte acceptance | `434088cff95ed60d65dc6c9749d18c2e74e45d114787c03728ed7ae9cf0bd9c9` |
| accepted synthesis publisher | `a9cc41f6b0862aa6857b0fdc060c6b6e22b6a2bd094c5cf1b9bc9b04463f6469` |
| accepted publisher tests | `2d7fef267bbf3e71cd4267d7cdb8ddd1c0f1047b92cf212fef909fdce15e00b3` |
| accepted publisher candidate report | `d548e460c76ad1ef488da7f5fd1c0765df5e40b8da2ef69941614d0b54c69a53` |
| publisher implementation acceptance | `fac18abc0b86cee71663506a100aed35bf4c24620c6e602792a1f21dd3c188cd` |
| accepted atmosphere postpublication audit | `34cbfd97b58df14ecc561a30dc3ce2e25cab552572030ec19488ee4d73486124` |
| synthesis authorization \(A\) | `ff46aa082226ae29ad7bc488a6e15f92596bfd005cc144077b1d477a117c0200` |
| synthesis authorization review \(R\) | `e2f3902bc0b59ba2dda0264a477b28aa6f9c43898d538514b2e7dded94627aa1` |
| realized synthesis artifact | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| realized manifest M2 | `b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44` |

Each control object was read through its fixed repository identity. The
accepted publisher's complete fixed-upstream gate passed again before the
registered-state validation.

## 3. Authorization and review graph

Both authority objects parsed as duplicate-free, finite UTF-8 JSON with the
exact parser-defined key order:

| property | authorization \(A\) | review \(R\) |
| --- | ---: | ---: |
| bytes | 81,829 | 800 |
| top-level keys | 13 | 10 |
| SHA-256 | `ff46aa082226ae29ad7bc488a6e15f92596bfd005cc144077b1d477a117c0200` | `e2f3902bc0b59ba2dda0264a477b28aa6f9c43898d538514b2e7dded94627aa1` |

The authorization contains:

- 21 ordered path/hash identity records;
- the exact 20-key comparison-only artifact contract;
- the exact 10-key prepublication-M1 contract;
- one 36-key manifest-entry template;
- 213 lexical six-key array records; and
- the exact source/data, destination-parent, and no-replace policies.

Every identity path was safe and every bound file rehashed to the value in
`A`. The publisher, tests, publisher acceptance, writer chain, contract
chain, and both lane byte acceptances all matched.

The review has disposition `ACCEPT` and binds:

```text
authorization_sha256
  ff46aa082226ae29ad7bc488a6e15f92596bfd005cc144077b1d477a117c0200

candidate_byte_acceptance_sha256
  434088cff95ed60d65dc6c9749d18c2e74e45d114787c03728ed7ae9cf0bd9c9

publisher_acceptance_sha256
  fac18abc0b86cee71663506a100aed35bf4c24620c6e602792a1f21dd3c188cd

manifest_entry_template_sha256
  2b168dede934551740917665999eb97a18563efeafb7cbf6ce592942ccef21de
```

The template contains exactly two late values, each once and only in its
complete named field:

```text
publication_acceptance_sha256
  __LATE_BOUND_AUTHORIZATION_SHA256__

publication_record_review_sha256
  __LATE_BOUND_RECORD_REVIEW_SHA256__
```

Neither `A` nor `R` contains a realized-entry digest, M2 hash,
postpublication-audit hash, review self-hash, or another backward edge. The
authority graph is acyclic.

## 4. Canonical synthesis artifact

The realized file is:

```text
path
  data/golden/payne_zero/chapter06/synthesis/
  chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz

type
  regular, nonsymlink

device / inode
  16777229 / 60652130

mode / link count
  0600 / 1

bytes
  1,294,865

SHA-256
  a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955
```

An independent `zipfile`/NumPy decoder required all of the following for
every member:

- exactly 213 unique lexical names equal to the authorization template;
- one `.npy` member per name and no directory member;
- `ZIP_STORED` with equal compressed and uncompressed sizes;
- fixed timestamp `(1980, 1, 1, 0, 0, 0)`;
- Unix creator `3`, create/extract version `20`, zero flags, volume and
  internal attributes;
- regular-file mode `0600` in the external attributes;
- empty archive/member comments and extras;
- NPY version 2.0, no trailing bytes, no object dtype, and C-contiguous
  storage;
- exact shape, dtype, rank-matched axes, scientific unit, and comparison-only
  ownership; and
- SHA-256 of contiguous C-order array bytes only.

All 213 records passed. Their aggregate evidence is:

| evidence | exact value |
| --- | --- |
| scientific array bytes | 1,235,275 |
| compact schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| compact payload fingerprint | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| member metadata digest | `39d319b577a59ea06de475a14da9fe68d5fcced24119964a4e3e322226b2b78a` |
| distinct scientific unit/convention strings | 35 |
| distinct axis labels | 8 |

The rank distribution is 116 scalars, 74 rank-one arrays, 21 rank-two
arrays, and 2 rank-three arrays. The member-prefix distribution is:

| prefix | members | prefix | members |
| --- | ---: | --- | ---: |
| `activity` | 1 | `axis` | 3 |
| `coarse` | 15 | `continuum` | 10 |
| `grid` | 3 | `identity` | 27 |
| `invariant` | 37 | `ledger` | 17 |
| `mapping` | 15 | `meta` | 71 |
| `opacity` | 2 | `stimulation` | 6 |
| `support` | 6 |  |  |

The accepted semantic validator independently recomputed the schema and
payload fingerprints and rejected copied atmosphere/input ownership. A fresh
NPY stream was regenerated for every array and a fresh canonical ZIP was
built from those streams. The resulting 1,294,865 bytes were exactly equal
to the realized artifact.

## 5. Exact M1-to-M2 manifest transition

M2 is a regular, nonsymlink, single-link mode-`0644` file:

```text
bytes
  1,177,564

entries
  39

SHA-256
  b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44

synthesis ordered entry-path digest
  d655d31f5a77beea280e6ebe2c5632b2dc356d4655b9a51ddd235575ad05efae
```

Strict duplicate-rejecting parsing found the exact top-level order
`schema_version`, `payne_zero_commit`, `entries`. Re-encoding with the
unsorted, two-space-indented UTF-8 manifest encoder reproduced every M2 byte.
All 39 paths are unique.

The synthesis destination occurs exactly once and is the final entry. Its
role is exactly `golden`; its path, byte count, and artifact SHA-256 match the
realized file. The manifest contains:

| role | entries |
| --- | ---: |
| `static` | 18 |
| `subset` | 1 |
| `fixture` | 7 |
| `golden` | 13 |

Realizing the authorization template changed exactly two scalar values:

```text
publication_acceptance_sha256
  ff46aa082226ae29ad7bc488a6e15f92596bfd005cc144077b1d477a117c0200

publication_record_review_sha256
  e2f3902bc0b59ba2dda0264a477b28aa6f9c43898d538514b2e7dded94627aa1
```

The final M2 entry is an order-preserving deep-equality match to that realized
template. Its compact ordered digest is:

```text
89779d743b8836f1db7ae5d59e6006b6b6992792befbf195b88f6ad3871180e7
```

Deleting only the final entry in memory and re-encoding gives:

```text
reconstructed M1 bytes
  1,097,670

reconstructed M1 entries
  38

reconstructed M1 SHA-256
  b86959d1bf34607b121d9ac336a7443d966c82f8b55599e45f885ca961f815d9

reconstructed M1 ordered entry-path digest
  6c48e8e946702fc962bd86572828b971beb3a70a3fdfca079cf6709735c44f68
```

Those hashes exactly equal the prepublication identities in `A`. Appending
the realized entry to the reconstructed M1 and re-encoding reproduced every
M2 byte. Thus the complete manifest delta is one append-last entry; no
earlier entry, nested value, or key order changed. This inverse check was
performed entirely in memory and did not rewrite either manifest.

## 6. Complete filesystem and source/data join

All 39 manifest entries were joined independently to their named files.
Every file was a regular, nonsymlink, single-link object with exact manifest
byte count and full-file SHA-256.

The closed M2 data inventory is:

| inventory class | count |
| --- | ---: |
| directories | 15 |
| regular files | 41 |
| manifest-backed regular files | 39 |
| closed nonmanifest support files | 1 |
| manifest file | 1 |
| symlinks or special nodes | 0 |
| quarantine temporaries | 0 |

Relative to accepted atmosphere M1, the only allowlisted directory additions
are:

```text
data/golden/payne_zero/chapter06
data/golden/payne_zero/chapter06/synthesis
```

Both are mode `0755`, owned by the same user and group as their accepted
parents. The only new payload is the exact registered synthesis artifact;
the only other byte change is M1 to M2.

The publisher's normalized stable-data algorithm excludes precisely the
realized synthesis artifact, the two allowlisted destination directories,
and the manifest's own separately bound bytes. It returned:

```text
stable data aggregate SHA-256
  182ff7ee00994225f920014883580e3a90576eb00270e8f46bd7a0ca9b1bf15d

accepted source identity count
  27

M1-normalized source/data snapshot SHA-256
  9d802e5dd96d9dbbb7e71c9a4e769e2959860ca5b2330f7796376dad2cd05ce2
```

The combined digest was recomputed from the current source inventory, current
closed M2 data inventory, reconstructed M1 bytes, and reconstructed M1
ordered-path digest. It exactly equals `A.source_data_snapshot_sha256`.
Therefore no unreviewed source or stable-data delta is hidden by the intended
M1-to-M2 transition.

The quarantine inventory was empty. No artifact-stage, manifest-temporary,
atmosphere temporary, symlink, or special node was present anywhere under
`data/`.

## 7. Fresh registered-state validation and zero-delta proof

The accepted publisher's hidden read-only registered validator was invoked in
a fresh process:

```text
python scripts/build_chapter06_synthesis_golden.py \
  --internal-validate-published
```

It exited `0`, emitted no stderr, rebuilt and checked the candidate through
the complete accepted authority/candidate path, and returned exactly:

```json
{
  "artifact_sha256": "a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955",
  "atmosphere_phase": "M1_registered_fixture",
  "authorization_sha256": "ff46aa082226ae29ad7bc488a6e15f92596bfd005cc144077b1d477a117c0200",
  "complete_validation": true,
  "data_aggregate_sha256": "182ff7ee00994225f920014883580e3a90576eb00270e8f46bd7a0ca9b1bf15d",
  "lane": "synthesis",
  "manifest_sha256": "b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44",
  "member_metadata_digest": "39d319b577a59ea06de475a14da9fe68d5fcced24119964a4e3e322226b2b78a",
  "realized_entry_sha256": "89779d743b8836f1db7ae5d59e6006b6b6992792befbf195b88f6ad3871180e7",
  "record_review_sha256": "e2f3902bc0b59ba2dda0264a477b28aa6f9c43898d538514b2e7dded94627aa1",
  "role": "golden",
  "schema_version": 1,
  "source_data_snapshot_sha256": "9d802e5dd96d9dbbb7e71c9a4e769e2959860ca5b2330f7796376dad2cd05ce2",
  "state": "exact_registered"
}
```

Complete snapshots taken immediately before and after that fresh process
proved:

```text
data file inode/mode/link/bytes/hash delta
  false

data directory inode/mode delta
  false

authorization/review hash delta
  false
```

The validator therefore acted as a read-only registered-state check. It did
not replace, rewrite, normalize, repair, or clean any canonical object.

## 8. Findings

| severity | count |
| --- | ---: |
| P0 | 0 |
| P1 | 0 |
| P2 | 0 |

No discrepancy was found in artifact identity, archive encoding, member
metadata, C-byte hashes, semantic fingerprints, authority/review binding,
manifest realization, append-only reconstruction, file join, directory
policy, source/data binding, quarantine state, registered validation, or
zero-delta behavior.

## 9. Decision

**ACCEPT** the exact realized Chapter 6 synthesis comparison golden:

```text
artifact
  a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955

authorization A
  ff46aa082226ae29ad7bc488a6e15f92596bfd005cc144077b1d477a117c0200

authorization review R
  e2f3902bc0b59ba2dda0264a477b28aa6f9c43898d538514b2e7dded94627aa1

realized final-entry digest
  89779d743b8836f1db7ae5d59e6006b6b6992792befbf195b88f6ad3871180e7

postpublication manifest M2
  b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44
```

This acceptance is limited to the exact realized comparison artifact,
authority graph, and M1-to-M2 transition reviewed above. It grants no
authority to rewrite the artifact or manifest, alter either authority record,
clean a future quarantine object, or treat the comparison golden as
reader-built state.
