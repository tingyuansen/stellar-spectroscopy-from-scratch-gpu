# Chapter 6 atmosphere-fixture postpublication audit

Date: 2026-07-30  
Reviewer role: independent read-only postpublication reviewer  
Disposition: **ACCEPT the exact realized fixture and manifest transition**

## 1. Review boundary

This audit reviews the already realized Chapter 6 atmosphere-fixture
publication. It did not invoke `publish()`, `--publish`, an authorized dry
run, the deterministic scientific writer, or any cleanup surface. It did not
rewrite the fixture, manifest, detached authorization, authorization review,
publisher, tests, accepted audits, Payne Zero checkout, or paper sources.

The only repository object created by this review is this report:

```text
design/chapter06_atmosphere_fixture_postpublication_audit.md
```

The review independently checked the exact artifact, the complete
authorization graph, the append-only manifest transition, every archive
member, the complete filesystem/manifest join, the closed source and data
inventories, and the publisher's read-only registered-state validator.

## 2. Exact realized trust chain

Every reviewed control object was a regular, nonsymlink, single-link file and
matched its exact expected SHA-256:

| reviewed object | SHA-256 |
| --- | --- |
| publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| final contract audit | `fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc` |
| atmosphere candidate-byte acceptance | `8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71` |
| final publisher | `5de0169339b8130d45c8e610684de67c2d6f3c6c62c5298610d0f705ebe0b363` |
| final publisher tests | `00c09576b00420bfaa72f058a1b70fab1dedb4e85e04326ca83c7bac0b6ca876` |
| final publisher candidate report | `f5f9ae5508c447938b3262cc5a34fd6901fe9519d6e408318ba6370e0e94cfaf` |
| threat-model adjudication, design evidence only | `4ed79dd4ff622abbeb6106a58b1b7e8c05a10c2ff87cfeb0dd6ee06e24e8e9fc` |
| final publisher acceptance | `516109c364f6cdd8f74240c3f34ccbad34655690c60d2b1fc7e58917959b154c` |
| detached publication authorization \(A\) | `a756fe395bbe9d598fcc4748e7b604920e615a4de9a2c2dabca5942e8a50b9eb` |
| authorization-record review \(R\) | `07518979af4d60d8cbd2321ea6c976a52ba04de53c2fc528af51888a6c42f37b` |
| realized artifact | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |
| realized manifest \(M_1\) | `b86959d1bf34607b121d9ac336a7443d966c82f8b55599e45f885ca961f815d9` |

`A` is 12,977 bytes and `R` is 819 bytes. Both parsed as duplicate-free,
finite UTF-8 JSON with the exact publisher-defined key order and types. `R`
has disposition `ACCEPT` and binds the exact authorization, candidate-byte
acceptance, publisher acceptance, and manifest-template digest. Every
path/hash object in `A` was rehashed from disk and matched.

The authorization's source snapshot, prepublication data snapshot, and
template digest remain:

```text
source_snapshot_sha256
  d1ae2f87d2bb4156c006fc72c963bf806f2253d630e8735ef5ab0f557cfc9101

prepublication_data_snapshot_sha256
  5be325c0200cf711ce13daa3dc96a47bdd8780c8de1640efe23827196de5b84a

manifest_entry_template_sha256
  9bbf7ac236d17331543addc489aee421dbe8c4731f5a87009bf4e083f8e78c08
```

The graph is forward-only: neither `A` nor `R` contains the realized-entry
digest, \(M_1\) hash, this audit's hash, or another later object. `A` contains
only the two contract-defined late placeholders inside its manifest template.

## 3. Canonical artifact

The canonical fixture is:

```text
path
  data/fixtures/chapter06_atmosphere_one_line_inputs.npz

file type
  regular, nonsymlink

device / inode
  16777229 / 60619663

mode / link count
  0600 / 1

bytes
  363,050

SHA-256
  1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
```

An independent ZIP/NPY decoder required exactly nineteen unique lexical
`.npy` members, `ZIP_STORED`, equal stored and uncompressed sizes, the fixed
1980 timestamp, Unix creator `3`, create/extract version `20`, zero flags,
volume and internal attributes, exact regular-file mode `0600` in external
attributes, empty member extras/comments and archive comment, no directory
member, and NPY version 2.0. Every NPY stream ended exactly at its array,
used no object dtype, and decoded C-contiguously with `allow_pickle=False`.

The nineteen member records independently matched the worker schema, accepted
C-byte hashes, authorization template, units, axes, and ownership:

| member | shape | dtype | unit | axes | C-byte SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `actual_population_slot_indices` | `(3,)` | `int16` | zero-based packed population slot | `population_projection` | `eff6cc5c731be5128ac078be458469978b6ac7c823665abd47098485291a5af2` |
| `actual_population_slot_values` | `(80, 3)` | `float64` | cm^-3 per partition function | `depth`, `population_projection` | `ae17c6b96dc9eb824051707d83eca2bbdb22483af4c143b18d832326292e86ce` |
| `continuum_line_selection_threshold` | `(80, 344)` | `float32` | cm^2 g^-1 | `depth`, `wavelength_bin` | `76cb7ef18554149b61b97e32f2a69bbcdba3eb8b7e6b7da8ba3c69b8775b7293` |
| `effective_temperature` | scalar | `float64` | K | none | `0ae804feb5a3896c0d30fd25ee9853a10ee08fed330969967ce16a4d07a7329b` |
| `electron_density` | `(80,)` | `float64` | cm^-3 | `depth` | `1e36bdf5aac263da2eac448f1b4c91e43e48a48afc133c8a8ec5e15735c85c18` |
| `fractional_doppler_widths_at_line_slot` | `(80,)` | `float64` | dimensionless Delta-lambda/lambda | `depth` | `f8018929e269db3e6b9a572d36b528bc0577a3a93bcc25166b277824ffb2e785` |
| `hc_over_kt` | `(80,)` | `float64` | cm | `depth` | `fdbd300b88163af886fe2905a883fa319c32abcc89db1099e4b0df7965d8174e` |
| `line_population_slot_zero_based` | scalar | `int16` | zero-based packed population slot | none | `8d6dd9f330421e48e73e97c0819d55a9965bac90b1f2f36057b4c906d5791f05` |
| `log_strength_index` | `(1,)` | `int16` | packed line-table integer | `selected_line` | `3fe821d54660a0c51b42d19a42571e208791b3c5ff6bc0cac16b6553a226515a` |
| `lower_excitation_index` | `(1,)` | `int16` | packed line-table integer | `selected_line` | `9b5bf0b74e2e212b57bcb2b9f2712eaab4c6169595f4e82767793f4534365648` |
| `opacity_wavelength_grid_nm` | `(30000,)` | `float64` | nm | `opacity_wavelength` | `8944f1dd701ba27f50d37a16e48ab9375e1bef1b444ed405b671ba91fde8132b` |
| `packed_species_slot` | `(1,)` | `int16` | zero-based packed species slot | `selected_line` | `c2126161f7488b7d198ea310da9a8694786a18a2317eea5e30379ee118d34743` |
| `packed_wavelength_index` | `(1,)` | `int32` | packed logarithmic wavelength integer | `selected_line` | `c423f4ac4a3825c6fad5336a1e15c0038ab17087f930916ad550ad45b4990dfc` |
| `partition_normalized_population_over_mass_density_and_fractional_doppler_width_at_line_slot` | `(80,)` | `float64` | g^-1 | `depth` | `8625e8d942892d142384d51fc0625701374f79dfa471a78ff593d88479b3056f` |
| `radiative_damping_index` | `(1,)` | `int16` | packed line-table integer | `selected_line` | `5cdf3b75730a5b45c3f24da2c1030103143981191d2844aa7374e948b9abaeea` |
| `stark_damping_index` | `(1,)` | `int16` | packed line-table integer | `selected_line` | `91d82e7b89ae43b29bb673bb416697d005e093d0011c9d7af501630c6a502141` |
| `temperature` | `(80,)` | `float64` | K | `depth` | `7b76410e731a6772cb6de12f923aea8a26e345778dbab67db3e8938278ce270f` |
| `van_der_waals_damping_index` | `(1,)` | `int16` | packed line-table integer | `selected_line` | `c0346f09a0362e8e9d29c01a9fc7c292a7395b524a90319bb921608b9fdb1b60` |
| `wavelength_bin_edges` | `(344,)` | `int64` | zero-based opacity-wavelength boundary index | `wavelength_bin_edge` | `ae50e9c0bafdcbfc39e242fd5029bf53b9e712d43008c6cb78a4493b89272cd5` |

The decoded arrays contain exactly 357,984 scientific bytes. All floating
values are finite; population slots are `[0, 2, 840]`; the selected line's
zero-based population slot is `350`; the final wavelength-bin sentinel is
\(2^{30}\); and the 30,000-point opacity grid is strictly increasing.

I independently regenerated every NPY stream and the complete ZIP container.
The regenerated 363,050-byte string was byte-for-byte equal to the canonical
artifact and retained SHA-256 `1b727671…`.

## 4. Exact append-only manifest transition

The realized manifest is a regular, nonsymlink, single-link mode-`0644` file:

```text
path
  data/MANIFEST.json

device / inode
  16777229 / 60619666

bytes
  1,097,670

entries
  38

SHA-256
  b86959d1bf34607b121d9ac336a7443d966c82f8b55599e45f885ca961f815d9

ordered entry-path SHA-256
  301fc6e7f28944e6c84b9b3970fe672498eebdba456790c802b324df5ebbf0f2
```

The strict duplicate-rejecting parser found top-level key order exactly
`schema_version`, `payne_zero_commit`, `entries`. Exact reserialization under
the contract's unsorted, two-space-indented UTF-8 encoder reproduced every
\(M_1\) byte.

The destination occurs exactly once and is the final entry. Removing only
that last object and reserializing produces:

```text
reconstructed M0 bytes
  1,087,741

reconstructed M0 entries
  37

reconstructed M0 SHA-256
  d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a

reconstructed M0 ordered entry-path SHA-256
  d63aa6dfd9209f21172c6ccf721cffc895835b8ee15a4ed160c5bebe0851b1aa
```

Those are the exact \(M_0\) byte and path identities bound by `A`. The
delete-last operation does not visit or reconstruct any earlier entry, so all
first 37 entry positions, object and nested-object key orders, and values are
preserved. Their complete reserialized tree reproduces the accepted \(M_0\)
hash exactly.

The intended template `T` has exactly 32 top-level keys and nineteen lexical
six-key array records. Compact ordered encoding gives exact template digest
`9bbf7ac2…`. It contains exactly one complete
`__LATE_BOUND_AUTHORIZATION_SHA256__` value and one complete
`__LATE_BOUND_RECORD_REVIEW_SHA256__` value in their two named fields.

The final entry is an order-preserving deep equality match to `T` after
exactly these two scalar substitutions:

```text
publication_acceptance_sha256
  a756fe395bbe9d598fcc4748e7b604920e615a4de9a2c2dabca5942e8a50b9eb

publication_record_review_sha256
  07518979af4d60d8cbd2321ea6c976a52ba04de53c2fc528af51888a6c42f37b
```

No placeholder remains in the realized entry. No other key, key order,
nested record, value, role, or scientific field changed. The realized entry
digest, defined by the contract's compact ordered JSON encoding, is:

```text
cfaf118c11c76a7b97198cb7fe7e3c0f78863f5eee71ea04bbe78c223d3653af
```

Its path, role, bytes, and SHA-256 join exactly to the canonical regular
single-link fixture.

## 5. Complete filesystem, role, and inventory join

Every one of the 38 unique manifest paths exists as a regular, nonsymlink,
single-link file whose complete bytes and SHA-256 equal its entry. Every path
is a safe repository-relative child of `data/`. Directory placement agrees
with role:

| role | entries |
| --- | ---: |
| `static` | 18 |
| `subset` | 1 |
| `fixture` | 7 |
| `golden` | 12 |

The complete closed postpublication `data/` tree contains:

| inventory class | count |
| --- | ---: |
| directories | 13 |
| regular files | 40 |
| manifest-backed regular files | 38 |
| closed nonmanifest support files | 1 |
| manifest file | 1 |
| symlinks or special nodes | 0 |

The thirteen directory identities are exactly the accepted prepublication
directory set. The only files not manifest-backed are
`data/MANIFEST.json` and the one accepted support file `data/README.md`.
The README remains at SHA-256
`1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b`.

The publisher's full postpublication snapshot vocabulary independently
returned:

```text
directories / files / manifest-backed / support
  13 / 40 / 38 / 1

postpublication data snapshot SHA-256
  d88db7094c55c28d6af5f10c9a532bec1cb6d7551fe8dbf5a30078af97d4f886

manifest SHA-256
  b86959d1bf34607b121d9ac336a7443d966c82f8b55599e45f885ca961f815d9

ordered entry-path SHA-256
  301fc6e7f28944e6c84b9b3970fe672498eebdba456790c802b324df5ebbf0f2
```

As an additional implementation-independent inventory, I sorted records by
path and encoded directory records as `path,type,mode` and regular-file
records as `path,type,mode,links,bytes,sha256`, using compact ordered UTF-8
JSON. Its 53-record postpublication digest is:

```text
e2db48ff35c82a3fa095a7c16edfacc41180fcee2fe1dcb0f9e7067b470d49a0
```

Removing the one target record and substituting the exact reconstructed
\(M_0\) bytes/hash for the manifest gives a 52-record reconstructed-prestate
digest:

```text
5c497fdb6d5ab29561794329fff39b49b27c889656571448fb8b8b82dd422a02
```

A recordwise diff between those content/name inventories contains exactly:

```text
data/MANIFEST.json
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
```

All first 37 manifest-backed file joins, the README, and the closed directory
set are unchanged. The live source snapshot also re-derived exactly the
authorization-bound `d1ae2f87…` identity: 32 accepted source records, current
publisher, tests and publisher acceptance, plus the complete accepted
deep-source gate, 36 records total.

No name beginning `.chapter06-atmosphere-stage-` or
`.chapter06-atmosphere-manifest-` exists. The publisher's quarantine scan
passed. There is no external lock file, unexpected support object, extra
directory, unregistered artifact, missing registered file, or role mismatch.

## 6. Read-only registered-state and idempotence validation

Static control-flow inspection found the registered-state branch inside the
common canonical-`data` lock before either mutation call. On a complete
registered match it calls `_fresh_postpublication_validation(paths)` and
returns exactly:

```text
identical-registered-no-op
```

The artifact-install and manifest-replacement calls are structurally later
and are unreachable from that true branch. The hidden internal validator has
no call to `publish`, candidate construction, stage creation, artifact
installation, or manifest replacement.

I then ran only the fresh read-only validation surface in a new interpreter:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /Users/ysting/anaconda3/bin/python \
  scripts/build_chapter06_atmosphere_fixture.py \
  --internal-validate-published
```

It exited `0` with no error. It freshly rechecked the canonical repository,
accepted source and complete data snapshots, artifact decoding and canonical
bytes, strict `A` and `R`, template realization, delete-last \(M_0\), final
entry, and exact registered target. The command did not acquire a publication
mode or reconstruct scientific children.

Before and after this fresh process, the exact hashes of \(A\), \(R\),
publisher acceptance, publisher, publisher tests, artifact, \(M_1\), and
README were identical. Recomputed source and postpublication data snapshot
digests remained `d1ae2f87…` and `d88db709…`, and the quarantine scan remained
empty.

## 7. Findings and disposition

| severity | findings |
| --- | ---: |
| P0 | 0 |
| P1 | 0 |
| P2 | 0 |

**ACCEPT** the exact realized atmosphere-fixture publication:

```text
artifact
  data/fixtures/chapter06_atmosphere_one_line_inputs.npz
  363,050 bytes
  1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff

authorization A
  a756fe395bbe9d598fcc4748e7b604920e615a4de9a2c2dabca5942e8a50b9eb

record review R
  07518979af4d60d8cbd2321ea6c976a52ba04de53c2fc528af51888a6c42f37b

realized entry digest
  cfaf118c11c76a7b97198cb7fe7e3c0f78863f5eee71ea04bbe78c223d3653af

postpublication manifest M1
  b86959d1bf34607b121d9ac336a7443d966c82f8b55599e45f885ca961f815d9
```

This accepts only the exact fixture registration reviewed above. It grants no
authority to rewrite the artifact or manifest, clean a quarantine object,
publish the synthesis golden, create the later atmosphere comparison golden,
or modify Payne Zero or the paper sources.
