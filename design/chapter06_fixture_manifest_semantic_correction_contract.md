# Chapter 6 fixture-manifest semantic-correction contract

Status: design candidate; no manifest mutation is authorized

## 1. Why a correction is required

The canonical Chapter 6 atmosphere fixture bytes are unchanged and remain
scientifically useful. During the first downstream oracle construction,
source-derived checks found three manifest descriptions that contradict the
meaning of those bytes.

The problem is metadata, not array shape, dtype, hash, or numerical payload.
It must nevertheless be repaired before the chapter can treat the manifest as
a self-contained semantic authority.

Historical realized state:

```text
fixture artifact SHA-256
  1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff

synthesis artifact SHA-256
  a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955

current manifest M2 SHA-256
  b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44

current entry count
  39
```

The original atmosphere authorization, its review, and the atmosphere and
synthesis postpublication audits remain immutable historical records. They
prove the exact M0→M1→M2 publication sequence that occurred. They are not
silently rewritten to pretend the semantic error was never published.

## 2. Source-derived correction set

Only the following three leaf values may change inside the existing entry
whose path is
`data/fixtures/chapter06_atmosphere_one_line_inputs.npz`.

| manifest leaf | M2 text | corrected text | source-derived meaning |
| --- | --- | --- | --- |
| `arrays.actual_population_slot_values.unit` | `cm^-3 per partition function` | `cm^-3 actual number density` | the values are projected directly from `ion_stage_populations_by_packed_slot` / `full_actual`, not from the partition-normalized line-support array |
| `arrays.packed_species_slot.unit` | `zero-based packed species slot` | `packed code: 10 times the one-based population slot` | the stored value is `3510`; the kernel decodes `abs(code) // 10 == 351`, while the corresponding zero-based Fe I population slot is separately stored as `350` |
| `arrays.wavelength_bin_edges.unit` | `zero-based opacity-wavelength boundary index` | `packed logarithmic wavelength code; final value is the 2^30 sentinel` | values begin near 4.4 million and end at `2**30`; they are code-space bin boundaries consumed by the packed selection search, not indices into the 30,000-point opacity grid |

These exact sources establish the meanings:

- `design/chapter06_atmosphere_fixture_oracle_plan.md`, especially the
  accepted fixture schema and reconstruction in Sections 5.2–5.3;
- `scripts/chapter06_atmosphere_fixture_worker.py`, where
  `actual_population_slot_values = full_actual[:, [0, 2, 840]]`;
- `scripts/chapter06_atmosphere_line_converter.py`, where
  `packed_species_slot = population_slot_one_based * 10`;
- staged `src/payne_zero_atmosphere/continuum_opacity.py`, where the first 343
  `wavelength_bin_edges` are constructed as rounded logarithmic wavelength
  codes and the final edge is assigned `2**30`;
- staged `src/payne_zero_atmosphere/line_opacity.py`, where the public selected
  line route consumes packed wavelength-code boundaries and decodes the
  packed species code.

No array name, shape, dtype, axes, ownership, C-byte hash, archive hash, role,
path, provenance identity, or other entry text may change.

## 3. Fixed authority graph

The manifest correction must be authorized by a detached JSON record and
independently reviewed before any write. The proposed identities are:

```text
design/chapter06_fixture_manifest_semantic_correction_authorization.json
design/chapter06_fixture_manifest_semantic_correction_record_review.json
scripts/apply_chapter06_fixture_manifest_semantic_correction.py
tests/test_chapter06_fixture_manifest_semantic_correction.py
design/chapter06_fixture_manifest_semantic_correction_candidate.md
design/chapter06_fixture_manifest_semantic_correction_independent_audit.md
design/chapter06_fixture_manifest_semantic_correction_postaudit.md
```

The authorization's source identity list must contain the following existing
objects with these exact hashes:

| authority | SHA-256 |
| --- | --- |
| `design/chapter06_atmosphere_fixture_oracle_plan.md` | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| `scripts/chapter06_atmosphere_fixture_worker.py` | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| `scripts/chapter06_atmosphere_line_converter.py` | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| `src/payne_zero_atmosphere/continuum_opacity.py` | `1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81` |
| `src/payne_zero_atmosphere/line_opacity.py` | `d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6` |
| `src/payne_zero_atmosphere/population_layout.py` | `36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0` |
| atmosphere publication authorization | `a756fe395bbe9d598fcc4748e7b604920e615a4de9a2c2dabca5942e8a50b9eb` |
| atmosphere authorization review | `07518979af4d60d8cbd2321ea6c976a52ba04de53c2fc528af51888a6c42f37b` |
| atmosphere postpublication audit | `34cbfd97b58df14ecc561a30dc3ce2e25cab552572030ec19488ee4d73486124` |
| synthesis publication authorization | `ff46aa082226ae29ad7bc488a6e15f92596bfd005cc144077b1d477a117c0200` |
| synthesis authorization review | `e2f3902bc0b59ba2dda0264a477b28aa6f9c43898d538514b2e7dded94627aa1` |
| synthesis postpublication audit | `eea7a14dfe437f90680f3c0b17e4f8d0cf2af38f04dde2e9e06345b654610c9b` |

The correction authorization must additionally bind by exact path, bytes, and
SHA-256:

- this final repaired correction contract and a later independent **ACCEPT**
  re-audit of that exact hash;
- the correction tool, focused tests, candidate report, and later independent
  implementation **ACCEPT** audit;
- the pinned Payne Zero commit and byte-identical pinned counterparts of the
  three staged atmosphere source files.

No file outside this closed list may supply a semantic or mutation rule.

### 3.1 Historical manifest frames

The authorization must embed these exact historical facts:

| frame | bytes | entries | SHA-256 | ordered path digest |
| --- | ---: | ---: | --- | --- |
| M0 | `1,087,741` | 37 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` | `d63aa6dfd9209f21172c6ccf721cffc895835b8ee15a4ed160c5bebe0851b1aa` |
| M1 | `1,097,670` | 38 | `b86959d1bf34607b121d9ac336a7443d966c82f8b55599e45f885ca961f815d9` | `6c48e8e946702fc962bd86572828b971beb3a70a3fdfca079cf6709735c44f68` |
| M2 | `1,177,564` | 39 | `b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44` | `d655d31f5a77beea280e6ebe2c5632b2dc356d4655b9a51ddd235575ad05efae` |

It must also bind:

```text
M1 atmosphere-entry digest
  cfaf118c11c76a7b97198cb7fe7e3c0f78863f5eee71ea04bbe78c223d3653af

atmosphere entry-template digest
  9bbf7ac236d17331543addc489aee421dbe8c4731f5a87009bf4e083f8e78c08

synthesis entry-template digest
  2b168dede934551740917665999eb97a18563efeafb7cbf6ce592942ccef21de

atmosphere source snapshot
  d1ae2f87d2bb4156c006fc72c963bf806f2253d630e8735ef5ab0f557cfc9101

atmosphere prepublication data snapshot
  5be325c0200cf711ce13daa3dc96a47bdd8780c8de1640efe23827196de5b84a

synthesis M1 source/data snapshot
  9d802e5dd96d9dbbb7e71c9a4e769e2959860ca5b2330f7796376dad2cd05ce2
```

### 3.2 Historical and correction-only digest vocabularies

The correction must preserve the historical algorithms that produced the
accepted M0, M1, and M2 identities. They are not redefined under a convenient
new encoder:

- the M0 ordered-path digest is
  `scripts.build_chapter06_atmosphere_fixture._ordered_path_digest`: compact
  JSON for the ordered path list, with no trailing LF;
- the M1 atmosphere-entry digest is
  `scripts.build_chapter06_atmosphere_fixture._entry_digest`: compact
  insertion-ordered JSON for the entry object, with no trailing LF;
- the atmosphere entry-template digest is over
  `scripts.build_chapter06_atmosphere_fixture._template_encode`, again with no
  trailing LF;
- the atmosphere source and prepublication-data snapshot digests are the
  exact outputs of
  `scripts.build_chapter06_atmosphere_fixture._verify_accepted_source_identities`
  and `scripts.build_chapter06_atmosphere_fixture._snapshot_data`,
  respectively, using the exact historical record order and no normalization;
- the M1 and M2 ordered-path digests are
  `scripts.build_chapter06_synthesis_golden._ordered_path_digest`: for each
  path in manifest order, hash its UTF-8 byte length as an unsigned
  eight-byte little-endian integer followed by the UTF-8 path bytes;
- the synthesis entry and entry-template digests are over
  `scripts.build_chapter06_synthesis_golden._entry_digest` and
  `scripts.build_chapter06_synthesis_golden._template_bytes`,
  respectively, both compact insertion-ordered JSON with no trailing LF;
- the synthesis M1 source/data digest is the exact output of
  `scripts.build_chapter06_synthesis_golden._source_data_snapshot_digest`
  over its exact historical source identities and
  length-prefixed `_snapshot_data` aggregate.

Consequently, the historical M1 atmosphere-entry digest is
`cfaf118c11c76a7b97198cb7fe7e3c0f78863f5eee71ea04bbe78c223d3653af`;
adding an LF would instead produce
`c1543600331ef5dbfa266eb1b6866b47a584eb2384ac5517a0c8f6135516d4fe`
and must be rejected. The M2 synthesis-style ordered-path digest is
`d655d31f5a77beea280e6ebe2c5632b2dc356d4655b9a51ddd235575ad05efae`.
The correction-aware validator must invoke or byte-for-byte reproduce these
named historical algorithms on the appropriate reconstructed frame. It may
not normalize them into the correction-only vocabulary below.

Only the new `unchanged_entry_leaf_digest` uses the following
type-preserving ordered leaf vocabulary.
Traverse dictionaries in their JSON insertion order and arrays by increasing
zero-based index. Emit one record for every terminal value:

```json
{"pointer":"/JSON/Pointer/path","json_type":"string","value":"literal"}
```

`json_type` is exactly one of `null`, `boolean`, `integer`, `number`, or
`string`; booleans are never integers. JSON Pointer escapes `~` as `~0` and
`/` as `~1`. Encode the complete record list with
`json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`
followed by one LF, then hash its UTF-8 bytes.
The `value` member retains the corresponding JSON primitive without string
coercion: `null` carries `null`, `boolean` carries a Boolean, `integer` carries
a non-Boolean integer, `number` carries a finite non-integer JSON number, and
`string` carries a string. A mismatch between `json_type` and the primitive
type is invalid.

`unchanged_entry_leaf_digest` traverses the M2 fixture entry but omits exactly
the three corrected leaf pointers. The M3 computation repeats the traversal
after omitting those same pointers and the entire newly appended
`/semantic_correction` subtree; the two digests must be equal. Container order
and topology are independently bound by the exact reversible M2/M3 encodings.

The resulting M2 value has 142 leaf records and is exactly
`1e0b981519035cf534fc7b98ff19447c768de3243d57f4debd93f918d5aa80b7`.
This literal is independently recomputed; it is not a late-bound placeholder.

The names `correction_source_inventory_sha256` and
`correction_data_inventory_sha256` refer to new correction-only aggregates.
Their record lists use compact insertion-ordered JSON plus one LF. These names
must never be used for any historical snapshot, entry, template, or
ordered-path digest.

The pre-correction data inventory contains every real directory and regular
file recursively under `data`, sorted by NFC-normalized POSIX relative-path
UTF-8 bytes. Each record contains exact path, kind, mode, link count, byte
count, and SHA-256 for files; symlinks and other kinds are rejected. The
manifest record is exact M2.

The correction source inventory is closed over these exact prerequisite
paths:

```text
design/chapter06_atmosphere_fixture_oracle_plan.md
scripts/chapter06_atmosphere_fixture_worker.py
scripts/chapter06_atmosphere_line_converter.py
src/payne_zero_atmosphere/continuum_opacity.py
src/payne_zero_atmosphere/line_opacity.py
src/payne_zero_atmosphere/population_layout.py
design/chapter06_atmosphere_fixture_publication_acceptance.json
design/chapter06_atmosphere_fixture_publication_record_review.json
design/chapter06_atmosphere_fixture_postpublication_audit.md
design/chapter06_synthesis_publication_acceptance.json
design/chapter06_synthesis_publication_record_review.json
design/chapter06_synthesis_postpublication_audit.md
design/chapter06_fixture_manifest_semantic_correction_contract.md
design/chapter06_fixture_manifest_semantic_correction_contract_independent_audit.md
scripts/apply_chapter06_fixture_manifest_semantic_correction.py
tests/test_chapter06_fixture_manifest_semantic_correction.py
design/chapter06_fixture_manifest_semantic_correction_candidate.md
design/chapter06_fixture_manifest_semantic_correction_independent_audit.md
```

No glob or directory walk may enlarge this source scope. In particular, the
future correction authorization, its future review, and the future
post-correction audit are excluded: including them would make the
preauthorization inventory self-referential or future-referential. The
records are sorted by NFC-normalized POSIX path UTF-8 bytes and contain exact
path, kind, mode, link count, byte count, and SHA-256. Each correction-only
inventory aggregate is the SHA-256 of its compact JSON-plus-LF encoding.
Independent review and postaudit must recompute the records rather than trust
tool-reported digests.

### 3.3 Exact two-placeholder closure

The authorization cannot contain its own final SHA-256 or the later review
SHA-256. It therefore binds a complete ordered
`semantic_correction_template` containing exactly these two literal values,
each occurring once:

```text
__LATE_BOUND_CORRECTION_AUTHORIZATION_SHA256__
__LATE_BOUND_CORRECTION_REVIEW_SHA256__
```

The authorization stores the SHA-256 of the compact JSON-plus-LF template.
The review record has this exact key order, spelling, and scalar type schema;
all values are JSON integers or strings as shown, and no additional key is
allowed:

```json
{
  "schema_version": 1,
  "record_kind": "chapter06_fixture_manifest_semantic_correction_record_review",
  "authorization_path": "design/chapter06_fixture_manifest_semantic_correction_authorization.json",
  "authorization_sha256": "<64 lowercase hexadecimal characters>",
  "contract_path": "design/chapter06_fixture_manifest_semantic_correction_contract.md",
  "contract_sha256": "<exact accepted repaired-contract SHA-256>",
  "contract_acceptance_path": "design/chapter06_fixture_manifest_semantic_correction_contract_independent_audit.md",
  "contract_acceptance_sha256": "<exact ACCEPT audit SHA-256>",
  "implementation_acceptance_path": "design/chapter06_fixture_manifest_semantic_correction_independent_audit.md",
  "implementation_acceptance_sha256": "<exact ACCEPT audit SHA-256>",
  "semantic_correction_template_sha256": "<exact template SHA-256>",
  "predecessor_manifest_sha256": "b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44",
  "disposition": "ACCEPT"
}
```

The correction tool must verify the review's exact key order and values. M3 is
constructed only by substituting the authorization SHA-256 for the first
complete placeholder value and the review SHA-256 for the second. It must
prove exactly two scalar substitutions and no residual placeholder text.

The independently reviewed record is a prerequisite, not a self-authorizing
write.

## 4. Realized fixture entry

The corrected fixture entry must retain every existing field and add exactly
one final top-level field:

```json
"semantic_correction": {
  "schema_version": 1,
  "record_kind": "manifest-semantic-correction",
  "authorization_path": "design/chapter06_fixture_manifest_semantic_correction_authorization.json",
  "authorization_sha256": "__LATE_BOUND_CORRECTION_AUTHORIZATION_SHA256__",
  "record_review_path": "design/chapter06_fixture_manifest_semantic_correction_record_review.json",
  "record_review_sha256": "__LATE_BOUND_CORRECTION_REVIEW_SHA256__",
  "previous_manifest_sha256": "b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44",
  "corrected_leaf_count": 3,
  "unchanged_entry_leaf_digest": "1e0b981519035cf534fc7b98ff19447c768de3243d57f4debd93f918d5aa80b7",
  "reason": "Correct three unit/convention descriptions; artifact bytes are unchanged."
}
```

This appended field makes the correction discoverable from the manifest
itself. It does not replace or erase the original publication identities.

## 5. Atomic mutation boundary

The correction tool has one repository, one manifest, one target entry, and
one exact M2 predecessor. It has no caller-selected path, force, repair,
merge, delete, or general JSON-edit option.

Verification mode must be structurally unable to reach a write primitive.
Authorized correction mode must:

1. obtain the same repository-scoped common lock used by the Chapter 6
   publishers;
2. open and identity-check the canonical manifest and both Chapter 6
   artifacts;
3. require exact M2 bytes before staging;
4. construct M3 in memory by changing only the three authorized leaves and
   appending the exact `semantic_correction` object;
5. reparse M3 and prove that every nonauthorized leaf is unchanged;
6. prove all 39 paths, roles, hashes, array schemas, and file joins;
7. revalidate the complete data and source inventories immediately before the
   replacement syscall;
8. use a same-directory, mode-0600 owned temporary and the accepted atomic
   retained-descriptor/fsync/rebind discipline;
9. refuse to overwrite any state other than exact M2;
10. fsync the retained manifest directory;
11. re-open the canonical path and validate exact M3;
12. leave no temporary or quarantine debris.

If the exact already-corrected M3 is present, validation is a zero-write
idempotent success. Any other state fails closed.

On this case-insensitive filesystem, `data/MANIFEST.json` is the sole directory
entry and `data/manifest.json` resolves to that same entry and inode. The tool
must prove this case-alias topology before the write and prove afterward that
both spellings resolve to the one newly installed canonical entry. It must not
misdescribe the aliases as two hard links, create a second case-variant
directory entry, or split the manifest identities.

## 6. Historical publication validation after M3

The old fixed-path publishers bind historical predecessor states. Their
accepted publication records must not be modified merely to make their
registered-state command accept M3.

Instead, the correction-aware current-state validator must:

1. validate M3, the exact two-substitution template realization, and the
   detached correction authorization/review/contract/implementation graph;
2. reconstruct exact M2 in memory by removing the correction field and
   reversing the three leaves;
3. prove reconstructed M2 has SHA-256
   `b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44`;
4. validate the historical synthesis authorization and review with their
   exact schemas, hashes, template digest, and late substitutions against
   reconstructed M2's final synthesis entry;
5. delete that synthesis entry to reconstruct exact M1, then prove the
   synthesis authorization's M1 manifest hash, ordered path digest,
   source/data snapshot, atmosphere artifact hash, and old atmosphere-entry
   digest `cfaf118c...` against that reconstructed M1 frame;
6. validate the historical atmosphere authorization and review with their
   exact schemas, hashes, template digest, and late substitutions against
   reconstructed M1's final atmosphere entry;
7. delete that atmosphere entry to reconstruct exact M0, then prove the
   atmosphere authorization's M0 manifest hash, ordered path digest, source
   snapshot, and prepublication data snapshot against that reconstructed M0
   frame;
8. revalidate both artifact archives and all 232 combined Chapter 6 members;
9. distinguish historical publication validity from current semantic
   validity in its report.

The unchanged synthesis entry and artifact remain historically authorized in
the M1→M2 frame. Their M1-bound fields are not reinterpreted as current M3
fields. The atmosphere-first condition was a publication-order dependency,
not a claim that the two final opacity slabs share grids or bytes. Every new
downstream atmosphere authorization must bind current corrected M3.

Thus the correction preserves the audit trail while giving future chapters
one truthful current manifest.

## 7. Post-correction acceptance

An independent post-correction audit must prove:

- exactly three semantic leaf changes plus one correction object;
- identical 39-entry order and roles;
- identical fixture and synthesis artifact inode/mode/link/bytes/hash;
- identical array shapes, dtypes, axes, ownership, and C-byte hashes;
- exact reverse reconstruction of M2, M1, and M0;
- a current manifest/data/source aggregate bound to M3;
- correction-aware current validation with zero write;
- no temp, quarantine, or unmanifested data;
- no change in the pinned Payne Zero or paper trees.

Only after that audit accepts may the atmosphere oracle authority and later
golden publication bind M3.

## 8. Explicit non-authority

This design candidate authorizes no manifest mutation. It also does not
authorize:

- rewriting either published NPZ;
- editing the historical atmosphere or synthesis authorization/review JSON;
- weakening a publisher's exact predecessor checks;
- treating the semantic mismatch as a numerical discrepancy;
- publishing the atmosphere golden;
- opening a golden before the reader-built calculation.
