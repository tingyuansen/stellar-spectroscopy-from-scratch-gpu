# Chapter 6 fixture-manifest semantic-correction contract: independent audit

Date: 2026-07-30

Reviewer role: independent read-only contract reviewer

Reviewed contract SHA-256:
`488049628c5a79adca0c89ee0bff25704c542fec1923151083270e07462a21a6`

Disposition: **REJECT as mutation authority; the scientific correction set is
accepted, but the authority and historical-validation contracts are not yet
closed**

## 1. Review boundary and method

This audit reviewed only
`design/chapter06_fixture_manifest_semantic_correction_contract.md`. It did
not edit the manifest, either published NPZ, a historical authorization or
review, a publisher, the pinned Payne Zero checkout, or the paper tree.

I independently re-derived the proposed changes from:

- the exact published atmosphere fixture bytes;
- current manifest M2;
- the accepted atmosphere fixture/oracle plan;
- the fixture worker and line converter;
- the staged atmosphere `continuum_opacity.py`, `line_opacity.py`, and
  `population_layout.py`, compared byte-for-byte with the pinned checkout at
  commit `9c44001feae40b85146630499e6f8a5fed42e5af`;
- the atmosphere and synthesis authorization/review records; and
- the two accepted postpublication audits.

I also parsed and re-encoded M2 independently, reconstructed M1 and M0 in
memory, decoded the 19-member fixture with `allow_pickle=False`, and inspected
the manifest directory by exact directory-entry spelling, inode, and link
count.

## 2. Findings

### P0 — The detached correction authority is under-specified and has an unresolved late-binding cycle

Section 3 does not yet require the authorization to bind the semantic
authority that justifies the mutation. Its mandatory identity set omits:

- this correction contract and its independent acceptance;
- the accepted oracle plan, fixture worker, converter, and the critical
  staged/pinned source identities that establish the three meanings;
- the historical atmosphere and synthesis authorizations, authorization
  reviews, and postpublication audits; and
- a complete authorization-bound pre-correction **source** inventory and
  aggregate. Section 5 requires a fresh source-inventory check immediately
  before replacement, but Section 3 binds only a pre-correction data
  inventory, so a conforming implementation has no contractually required
  source-state authority against which to perform that check.

The tenth authorization requirement is also not realizable literally. The
final `semantic_correction` object contains the authorization's own SHA-256
and the later review SHA-256, so the authorization cannot bind that exact
final object. The contract shows late-bound values but does not define the
accepted publication-style closure:

1. two exact literal placeholders, each occurring once and only in its named
   complete value;
2. a compact ordered correction-object or realized-entry template digest;
3. a review that binds the authorization, contract acceptance,
   implementation acceptance, and that template digest; and
4. a realized-M3 check proving that exactly the two named scalar
   substitutions, and no other substitution, instantiate the template.

Without those requirements, the authorization/review pair does not
cryptographically close the exact object installed in M3. This is a mutation
authority blocker.

Required correction: define the complete fixed identity graph and source
snapshot in the authorization schema, and replace the impossible “exact
correction-record field” requirement with the exact two-placeholder template
and review closure above.

### P0 — Historical validation does not yet preserve the exact synthesis-to-atmosphere binding

The proposed inverse manifest sequence is correct, but Section 6 validates
only the reconstructed manifest hashes and the two archive payloads. That is
not yet the historical authorization chain.

The synthesis authorization binds all of the following to historical M1:

```text
M1 SHA-256
  b86959d1bf34607b121d9ac336a7443d966c82f8b55599e45f885ca961f815d9

M1 atmosphere-entry digest
  cfaf118c11c76a7b97198cb7fe7e3c0f78863f5eee71ea04bbe78c223d3653af

atmosphere artifact SHA-256
  1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
```

M3 necessarily has a different current atmosphere-entry digest. A
correction-aware validator must therefore check the synthesis authorization's
old atmosphere-entry digest against the **reconstructed M1 entry**, not
against M3, and must check its M1 manifest and source/data snapshot in that
same historical frame. It must then check the atmosphere authorization
against reconstructed M0. The current checklist does not require either
authorization/template/review validation, and it does not state this
frame-specific downstream synthesis rule.

Required correction: bind the exact M1 and M0 bytes, SHA-256 values, ordered
path digests, historical authorization/review identities, and template
digests; require the current-state validator to validate the synthesis
M1/old-atmosphere-entry binding before applying the atmosphere M1-to-M0
inverse; and state explicitly that:

- the unchanged synthesis entry and artifact remain historically authorized;
- their M1-bound fields are not reinterpreted as M3-current fields;
- the atmosphere-first synthesis condition was a publication-order
  dependency, not a claimed shared scientific slab; and
- every new downstream atmosphere authority binds M3, as Section 7 already
  requires.

This closes the history without rewriting old records or weakening their
predecessor checks.

### P1 — Two digest and inventory vocabularies are not independently reproducible

The contract requires an `unchanged_entry_leaf_digest` and complete
data/source aggregates but does not define their canonical record vocabulary,
ordering, JSON scalar typing, leaf-path escaping, or encoder. In particular,
“every nonauthorized leaf” and “digest over all unchanged fixture-entry
leaves” can produce different correct-looking hashes depending on whether
containers, ordered object positions, or only terminal scalar values are
included.

Required correction: define one type-preserving ordered leaf-record
vocabulary and exact UTF-8 encoder, or bind an already accepted named
algorithm by exact code identity. Define the data/source inventory
canonicalization with the same precision. The post-correction audit must
recompute these values independently rather than merely compare values
reported by the correction tool.

### P1 — The wavelength-boundary source citation stops at the consumer

The proposed wavelength correction is scientifically correct, but the exact
source list names only `line_opacity.py`, which consumes the codes.
`src/payne_zero_atmosphere/continuum_opacity.py` is the direct constructor:
its first 343 values are
`int64(log(wavelength_nm) / log(1 + 1/2_000_000) + 0.5)` and its final value
is assigned `2**30`. The staged file and pinned file are byte-identical at
SHA-256
`1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81`.

Required correction: cite and authority-bind this constructor as well as the
`line_opacity.py` consumer. This makes the phrase “final value is the
`2^30` sentinel” directly source-derived rather than dependent only on the
design plan.

### P2 — None

## 3. Scientific re-derivation

The fixture is exactly 363,050 bytes at SHA-256
`1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff`.
Its 19 manifest array records join exactly to its decoded members.

The three proposed old descriptions are the only contradictions among those
19 records:

1. `actual_population_slot_values` is copied by the worker from
   `full_actual[:, [0, 2, 840]]`, where `full_actual` is
   `ion_stage_populations_by_packed_slot`. It is distinct from the
   separately projected partition-normalized line-support array. The accepted
   plan gives its unit as cm\(^{-3}\). Therefore `cm^-3 actual number density`
   is precise and `cm^-3 per partition function` is wrong.
2. The converter computes `population_slot_one_based = 351` and then
   `packed_species_slot = population_slot_one_based * 10`. The artifact
   stores `3510`; the selected-line consumer decodes
   `abs(packed_species_slot) // 10`, while the fixture separately stores the
   zero-based Fe I array column `350`. Therefore the proposed replacement is
   exact for this ordinary positive Fe I record.
3. `wavelength_bin_edges` has values from `4,414,351` through
   `1,073,741,824`, whereas the direct opacity grid has only 30,000 samples.
   The source constructs logarithmic wavelength codes and assigns the final
   `2**30` sentinel; the line kernel compares each packed line wavelength
   directly with these values. Therefore they are code-space boundaries, not
   zero-based opacity-grid indices.

The other 16 records are sometimes terser than the plan, but none contradicts
its producer, consumer, value, unit, or convention. The exact replacement
texts in Section 2 are scientifically acceptable. Adding the correction
object is provenance, not a fourth correction to an existing leaf.

## 4. Manifest history and alias topology

Independent unsorted two-space JSON re-encoding reproduced current M2
byte-for-byte:

| state | bytes | entries | SHA-256 |
| --- | ---: | ---: | --- |
| M2 | 1,177,564 | 39 | `b9bc60c1d030529b1e2da568245c7da7f4ab147f79fdb9d6a26a8ed730eb9e44` |
| M1, delete synthesis last | 1,097,670 | 38 | `b86959d1bf34607b121d9ac336a7443d966c82f8b55599e45f885ca961f815d9` |
| M0, then delete atmosphere last | 1,087,741 | 37 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |

The M2 fixture-entry digest is
`cfaf118c11c76a7b97198cb7fe7e3c0f78863f5eee71ea04bbe78c223d3653af`.
Changing only the three nested strings and appending one final top-level
correction object is exactly reversible to those bytes and therefore can
preserve the historical chain once the P0 validation requirements above are
added.

The revised case-alias statement is accurate. The directory contains one
entry spelled `MANIFEST.json`. `MANIFEST.json` and `manifest.json` resolve to
the same inode, but `st_nlink == 1`; this is case-insensitive path resolution,
not a second hard link. Section 5 now correctly requires preservation of the
single canonical entry and both path resolutions. A conforming implementation
should prove this by raw directory-entry enumeration plus `lstat`/`fstat`,
not by `samefile` alone.

The retained-descriptor, common-lock, immediate pre-syscall validation,
same-directory atomic replacement, directory `fsync`, canonical rebind, and
zero-write M3 validation requirements are otherwise an adequate atomic
mutation boundary.

## 5. Final judgment

The scientific diagnosis and the exact three replacement texts are accepted.
The M2-to-M3 edit shape, inverse reconstruction, case-alias treatment, and
atomic replacement design are also sound in principle.

The contract is nevertheless **REJECTED as publication authority** until the
late-bound authorization is cryptographically closed and the validator is
required to revalidate the exact historical synthesis/atmosphere authority
frames. No manifest mutation should proceed under the reviewed contract hash.

## 6. Re-audit of repaired contract

Re-audit date: 2026-07-30

Repaired contract SHA-256 reviewed:
`855a29a571a1805ca1b9496648c91a122fe23c654e7698191e7d31142811bc43`

Re-audit disposition: **REJECT; the prior scientific, authority-graph,
historical-frame, downstream-binding, and alias findings are substantively
repaired, but the new canonical-digest rule contradicts the exact historical
digests it is required to validate**

### 6.1 Prior findings that are repaired

The repaired contract now correctly:

- authority-binds the oracle plan, worker, converter, direct
  `continuum_opacity.py` constructor, `line_opacity.py` consumer,
  `population_layout.py`, their pinned byte-identical counterparts, both
  publication authorization/review pairs, both postpublication audits, the
  repaired contract and its acceptance, and the future correction
  implementation chain;
- states exact M0, M1, and M2 byte sizes, entry counts, and manifest hashes;
- requires complete pre-correction data and source inventories and
  independent recomputation;
- defines a type-tagged JSON-Pointer leaf vocabulary for the new unchanged
  leaf digest;
- resolves the authorization self-reference with exactly two one-use literal
  placeholders and a review-bound template digest;
- checks the synthesis authorization's old atmosphere-entry digest in the
  reconstructed M1 frame rather than against current M3;
- validates the atmosphere authorization in reconstructed M1 and its
  predecessor facts in reconstructed M0;
- distinguishes historical publication validity from M3-current semantic
  validity and correctly describes the atmosphere-first condition as
  publication ordering rather than shared-slab science;
- cites the direct packed-wavelength constructor and its final `2**30`
  assignment; and
- preserves the exact one-directory-entry, case-insensitive alias rule
  without inventing a second hard link.

The two-placeholder graph no longer has an inherent self-hash cycle:
implementation acceptance precedes authorization, review binds the completed
authorization and template, and only M3 binds the later review hash.

### 6.2 P0 — The universal digest vocabulary makes the exact historical frames impossible

Section 3.2 says that the ordered path digest and fixture-entry digest use
compact JSON **plus one LF**. The exact historical values in Section 3.1 were
created by two different accepted publishers and do not use that vocabulary.
Fresh recomputation gives:

#### Atmosphere publisher vocabulary

The atmosphere publisher hashes compact unsorted JSON with **no trailing LF**
for both entries and ordered path lists.

```text
M0 37-path digest, compact JSON without LF
  d63aa6dfd9209f21172c6ccf721cffc895835b8ee15a4ed160c5bebe0851b1aa

same path list with the new required LF
  480b24c4d65dba91d27b57092927cadf00dbb02ded08e1be752ebd27ccf4fa08

historical M1 atmosphere-entry digest, compact JSON without LF
  cfaf118c11c76a7b97198cb7fe7e3c0f78863f5eee71ea04bbe78c223d3653af

same entry with the new required LF
  c1543600331ef5dbfa266eb1b6866b47a584eb2384ac5517a0c8f6135516d4fe
```

Thus the repaired contract cannot both apply Section 3.2 and prove its bound
M0 path and M1 atmosphere-entry values.

#### Synthesis publisher vocabulary

The synthesis publisher's ordered path digest is not JSON. For every path it
hashes an eight-byte little-endian UTF-8 byte length followed by those path
bytes.

```text
M1 38-path synthesis digest, accepted length-prefix vocabulary
  6c48e8e946702fc962bd86572828b971beb3a70a3fdfca079cf6709735c44f68

same list under compact JSON plus LF
  0e68dc6cf6b92b35245079d2725ee01450925732170538b69809f5ea19f43097

M2 39-path synthesis digest, accepted length-prefix vocabulary
  d655d31f5a77beea280e6ebe2c5632b2dc356d4655b9a51ddd235575ad05efae

same list under compact JSON plus LF
  7a1cbb7a24c357f4d2dfca8ca74b1c95cc4c627875f8a8914c70d793ee0e031b
```

Section 3.1 leaves the M2 ordered-path value as
“authorization-computed,” but Section 6 must validate the accepted synthesis
M1-to-M2 frame. Computing it with the new Section 3.2 rule would silently
replace the accepted historical vocabulary and fail to prove the synthesis
authorization's path digest.

Required correction:

1. preserve three explicitly named digest namespaces:
   `historical_atmosphere_compact_json_no_lf`,
   `historical_synthesis_length_prefixed_utf8`, and a separately named
   `correction_compact_json_plus_lf`;
2. state that historical template and realized-entry digests retain the
   exact no-LF publisher vocabulary;
3. bind M2's historical synthesis ordered-path digest explicitly as
   `d655d31f5a77beea280e6ebe2c5632b2dc356d4655b9a51ddd235575ad05efae`;
4. require M0/M1 checks under the atmosphere vocabulary and M1/M2 checks
   under the synthesis vocabulary; and
5. use the new plus-LF vocabulary only for newly introduced correction
   template, leaf, inventory, and current-state digests with distinct field
   names.

This is a P0 because no implementation can satisfy both the stated algorithm
and the exact historical hashes.

### 6.3 P1 — The source-inventory boundary can still be read as self-referential

Section 3.2 says the source inventory contains “every fixed authority path in
Section 3.” Section 3 also lists the new correction authorization, its later
review, and the future postaudit as proposed identities. If “every” includes
those records, the authorization would need to inventory itself and future
objects, recreating an impossible self/future-hash cycle.

The likely intent is the finite preauthorization set: the twelve existing
rows in the Section 3 table, the repaired contract and this ACCEPT re-audit,
the tool, tests, candidate report, implementation ACCEPT audit, and three
pinned source counterparts. State that list exhaustively and explicitly
exclude the correction authorization itself, its later review, M3, and the
future postaudit from the authorization's prestate source inventory. Those
later identities are closed by the review, M3 correction object, and
postaudit in their causal order.

### 6.4 P1 — The new review and leaf schemas need literal field/type closure

The review is described as an “exact closed schema,” but Section 3.3 gives
semantic items rather than the literal ordered key tuple and JSON types.
Likewise, the leaf example shows `"value":"literal"` although the vocabulary
is described as type-preserving. The implementation should not be free to
stringify integers, numbers, booleans, or nulls.

Required correction: enumerate the review record's exact key names, insertion
order, and JSON types. State explicitly that each leaf record's `value` is
the original JSON scalar of the type named by `json_type`; only a string leaf
has a string value. These are deterministic-schema clarifications rather than
new scientific rules.

### 6.5 Re-audit judgment

The repaired contract closes all prior scientific and publication-history
substance, and its two-placeholder causality is sound. The digest-namespace
collision is nevertheless a hard implementation contradiction, so the
contract remains **REJECTED** at SHA-256
`855a29a571a1805ca1b9496648c91a122fe23c654e7698191e7d31142811bc43`.
No manifest mutation should proceed under this hash.
