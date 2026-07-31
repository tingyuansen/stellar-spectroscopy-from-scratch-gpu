# Chapter 6 synthesis scientific-worker plan-rebind candidate

Status: **CANDIDATE ONLY — independent worker rebind audit required**

Scope: phase 1 of the forward repair cascade required by
`design/chapter06_synthesis_candidate_byte_acceptance.md`

Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

This phase edited no worker, worker test, compact assembler, deterministic
writer, prior candidate/audit, data file, manifest, contract, or external
source. No accepted constant was refreshed. This report is the sole
repository file created by this phase.

## 1. Disposition

Two independently launched fresh controlled processes reproduced the exact
existing scientific worker
`scripts/chapter06_synthesis_oracle_worker.py` at SHA-256
`36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68`
against the repaired plan at SHA-256
`413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856`.

The two complete 754-member mappings and their canonical deterministic raw
transports were byte-identical. The new live identities are:

| property | measured candidate value |
| --- | --- |
| member count | `754` |
| array bytes | `8,451,402` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical-payload fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full-capture fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |
| complete mapping digest | `09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c` |
| deterministic raw transport bytes | `8,689,108` |
| deterministic raw transport SHA-256 | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` |

The member inventory, array-byte total, schema digest, and physical
fingerprint are unchanged from the old accepted snapshot. Exactly two
metadata members changed. Every one of the 670 scientific members has the
same dtype, shape, and C bytes as the old accepted snapshot.

This is a measured rebind candidate, not an acceptance. An independent review
must repeat the captures and decide whether these exact new identities may
replace the former worker acceptance boundary.

## 2. Why the plan identity changed

The forward-cascade trigger is the independent rejection:

| evidence | SHA-256 |
| --- | --- |
| `design/chapter06_synthesis_candidate_byte_acceptance.md` | `474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1` |
| `design/global_interface_molecular_repair_independent_audit.md` | `048a9d970e7a56fd48e2b0cfaa5632b968875f1f741f4d32c2dc84ae6c5d3e6c` |
| repaired synthesis fixture/oracle plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |

The rejection records the accepted global-interface repair candidate at
SHA-256
`6f9176bb8cc11e675fc2f441bd4465cb4f366bb95a3ee17bda1d3d23417261fd`.
The independent global audit explicitly accepts the repaired plan at the
exact live hash above. Its bounded Chapter 6 change replaces source-like
lowercase `r_grid` prose with the actual `Grid.resolution` interface or the
mathematical \(R_{\rm grid}\). It changes no grid value, fixture, source
record, numerical policy, or kernel.

During this phase, the live global repair candidate received an unrelated
concurrent P1 append after the phase-start hash was recorded. It is not a
scientific worker input and is excluded from the scientific source
immutability set below. This report binds the already accepted independent
audit, the exact repaired plan, and the cascade rejection; it does not treat a
later unreviewed live revision of that candidate as authority.

The worker reads the plan hash into `meta__design_sha256`, and its full
fingerprint covers that member. Consequently a plan-byte change must change
the full fingerprint even when every physical/scientific byte is unchanged.

## 3. Exact old-to-new delta

The historical accepted mapping was reconstructed in memory from the live
mapping by restoring only the former `meta__design_sha256` and former
`meta__full_capture_fingerprint`. This was not accepted on assumption:

1. the worker's unchanged full-fingerprint algorithm recomputed exactly
   `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b`;
2. the unchanged physical-fingerprint algorithm recomputed exactly
   `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc`;
3. canonical serialization reproduced the already accepted historical raw
   transport exactly: `8,689,108` bytes and SHA-256
   `5ff93594781de6fc12a3a2a0adf0ebcc03c81f6141b8648e709ad306686b93a3`;
4. the reconstructed complete mapping digest reproduced
   `e2a3dd9670fd9fb5ed34f5131c303687d72ff2d3e762d9224c702f1dffc4a775`.

Matching the complete historical canonical transport is the control that
turns the old/new comparison into a byte measurement rather than a
prose-equivalence inference.

Exactly these two members differ:

| member | dtype / shape | old value | new value |
| --- | --- | --- | --- |
| `meta__design_sha256` | `<U64` / scalar | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| `meta__full_capture_fingerprint` | `<U64` / scalar | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |

Their exact member hashes are:

| member | old C-byte SHA-256 | new C-byte SHA-256 | old canonical NPY SHA-256 | new canonical NPY SHA-256 |
| --- | --- | --- | --- | --- |
| `meta__design_sha256` | `061c11922d36f56e8504fefc343706dfa9d44d2f431d8a2e04e7a7a7a245192c` | `96d7b7dfabd27de62e184d60e3f5d708e39b81aa7513a9a4da784fe0032db5df` | `1dc120ad1be432a56efb84deb9de95c928490b2d459b7eba0bc730d2033dd804` | `3fdd6d84f49c499a872c0f9ef998faf6178ce5f8c1da2833f0ef3c1b27133775` |
| `meta__full_capture_fingerprint` | `fb3259cc9c8f0fa9dfa3654666589f44f4c9c61124a3285d360a645204db4c12` | `19468e78702fa6eefc61243087357689673a299b5dd5925c70aa36fc6d043f41` | `13b7fc0a745a2971e127203a50f21cafc6148bd7dfc52ce2c5445f6ce2b9345f` | `e9af37e046329e0132353587b7f1d5553d105500958bc7b2cc1f7f669448ebeb` |

All 752 other members are byte-identical. The two canonical raw transports
have the same length and differ at 135 byte positions, including the two NPY
payloads and their ZIP checksums. Their whole-transport SHA-256 values are:

| snapshot | bytes | raw SHA-256 |
| --- | ---: | --- |
| old accepted | `8,689,108` | `5ff93594781de6fc12a3a2a0adf0ebcc03c81f6141b8648e709ad306686b93a3` |
| repaired-plan candidate | `8,689,108` | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` |

## 4. Scientific-byte proof

The physical fingerprint excludes all `meta__` and `identity__` members.
Both the historical reconstruction and both live captures independently
produce:

```text
51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc
```

The stronger direct comparison selected all 670 members whose names do not
begin with `meta__` or `identity__` and required exact equality of name,
dtype string, shape, and contiguous C bytes. It passed for all 670 members,
covering `8,433,496` scientific array bytes. Their mapping digest, with names,
dtypes, shapes, and C bytes, is unchanged at:

```text
c05be60a58b5db26544dca8fe7f53e510d0b7d30b70e801b59337c1e41a405b0
```

The complete measured family inventory remains:

| family | members |
| --- | ---: |
| `meta__` | `57` |
| `identity__` | `27` |
| `fixture_payload__` | `2` |
| `subset_payload__` | `2` |
| `record__` | `14` |
| `catalog__` | `18` |
| `grid__` | `8` |
| `invariant__` | `86` |
| each of the four regime families | `135` |

All 754 arrays are C-contiguous and object-free. The schema digest remains
unchanged because both changed values retain their existing names, `<U64`
dtypes, and scalar shapes.

The two live summaries also retained:

- exact canonical and coarse activity masks
  `111000`, `111111`, `111111`, `111111`;
- canonical active reach `5` through `163`;
- eight full stimulated-emission-factor members with extrema
  `0.5894814729690552` and `0.9999690055847168`;
- exact gross-to-net reconstruction for loop and batched routes;
- zero loop/batched absolute difference;
- 17 loaded pinned Python sources; and
- false golden-read and golden-publication flags.

## 5. Fresh-process and environment evidence

Both captures used `/Users/ysting/anaconda3/bin/python` and the exact controls:

```text
MKL_DYNAMIC=FALSE
MKL_NUM_THREADS=1
NUMBA_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
LC_ALL=C
PYTHONHASHSEED=0
PYTHONDONTWRITEBYTECODE=1
PYTHONNOUSERSITE=1
TZ=UTC
PAYNE_ZERO_DATA_ROOT=/Users/ysting/payne-zero/source_data_files
PYTHONPATH=src:.
```

The parent created two distinct empty external nonsymlink cache directories.
Each child received its own one-use origin token and emitted an evidence header
followed by its canonical raw bytes. The parent verified the token, child PID,
cache-root identity, raw byte count/hash, exact A/B array parity, post-capture
cache emptiness, and cache disposal.

| process evidence | capture A | capture B |
| --- | --- | --- |
| child PID | `36949` | `36980` |
| origin-token SHA-256 | `2fe984a71d0f64db82b76d689c39cdece9c962973e198b38cc42b392fc8e2dc1` | `74c35f4c4358de8985cae20aea9a460cc4d9537d67b54fbd1390532c344fdca3` |
| cache-root SHA-256 | `d279dbf12783994c7b67d5107e1f26996be99a7f3ffec720d5b0daa132909116` | `d8f72cd8c2325a0edabe592083d4bc2144cd2a2a12df57da06d1b7c2a548e965` |
| raw bytes | `8,689,108` | `8,689,108` |
| raw SHA-256 | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` |
| cache empty before/after | yes | yes |
| cache removed before return | yes | yes |

The origin tokens, PIDs, and cache paths are process evidence only. They are
not members of the scientific mapping or deterministic transport.

Observed platform:

| field | value |
| --- | --- |
| Python | `3.13.9` |
| NumPy | `2.3.5` |
| Numba | `0.62.1` |
| Torch | `2.11.0` |
| platform | `macOS-26.5.2-arm64-arm-64bit-Mach-O` |
| machine | `arm64` |
| byte order | `little` |

## 6. Read-set and no-write evidence

A Python audit hook was installed before importing the worker in each measured
child. It snapshotted `open` events immediately after
`build_oracle_results()` returned and before importing the existing
deterministic transport codec.

The two captures observed the same bounded source/data read-path set. A
separate fresh-cache repeat canonicalized its sorted path list and measured:

| read-set property | value |
| --- | --- |
| unique in-scope open paths/probes | `53` |
| existing regular files | `35` |
| absent import probes | `18` |
| `.py` paths | `23` |
| `.pyc` paths/probes | `18` |
| `.npz` paths | `9` |
| `.md` paths | `2` |
| sorted read-set SHA-256 | `475fc8d5f7d68af5994bcf701fa75e89ae924c9fc4c1b3b5459dfb83a21cd174` |
| in-scope write-mode open paths | `0` |

The existing reads comprise the worker, plan, exact source contract, five
staged executed sources, the loaded pinned source set, source archive,
fixture, subset, and staged/upstream static tables. Missing `.pyc` probes and
the editable-finder path probe are recorded rather than misrepresented as
scientific files. The worker independently required exact equality with its
17-file loaded pinned Python manifest before and after the numerical route.

No golden was read. Both mappings record:

```text
meta__golden_read_performed                 false
meta__golden_publication_performed          false
```

The worker signature and CLI still expose no output, destination, serialize,
publish, or golden option. The intended synthesis destination remained absent
and `data/MANIFEST.json` retained no synthesis-golden entry. No raw transport
or candidate NPZ was written; raw bytes remained in subprocess pipes and
parent memory.

## 7. Source and data immutability

The before/after scientific-source set contains exactly:

- the unchanged worker, worker tests, repaired plan, and exact source
  contract;
- the five staged executed sources;
- all 17 frozen pinned Python sources;
- the full source archive; and
- the three upstream static synthesis tables.

The complete textbook `data/` tree was independently snapshotted. The
snapshots bracketed a repeated focused worker suite with its own two fresh
scientific processes and were repeated again after the audit-hook capture.

| snapshot | exact before/after value |
| --- | --- |
| scientific source regular files | `30` |
| scientific source bytes | `259,414,135` |
| scientific source snapshot digest | `a945cc441f271e533664ba3cd263c74e8abaa3a0af664d048d3bb6623eba32f0` |
| scientific source symlinks | `0` |
| `data/` directories | `12` |
| `data/` regular files | `39` |
| `data/` regular-file bytes | `30,046,405` |
| complete `data/` snapshot digest | `288bbe4c6bcbd20da8390f99fb6cc45e07ee6eed24197371d3520aed39f7d004` |
| `data/MANIFEST.json` bytes | `1,087,741` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| source before/after | exact |
| data before/after | exact |
| manifest before/after | exact |

Each tree digest hashes the sorted resolved path length and path bytes,
followed by the file-byte length and binary SHA-256 for every member. This
binds both inventory and content rather than only concatenating file hashes.

## 8. Commands and verification

The exact child scientific operation was:

```python
from scripts import chapter06_synthesis_oracle_worker as worker

results = worker.build_oracle_results()
summary = worker.summarize(results)
```

Each child ran as:

```text
/Users/ysting/anaconda3/bin/python -c <framed in-memory capture source>
```

under the complete environment in Section 5. After the audit hook snapshotted
the scientific read set, the child used the unchanged private raw-transport
codec from
`scripts/chapter06_synthesis_compact_writer.py`
SHA-256
`3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb`:

```python
raw_bytes = writer._serialize_ephemeral_raw_mapping(results)
```

No writer accepted-identity gate, compact assembly, final serialization, or
publication function was invoked. The codec is bound here only so the
transient transport can be compared with the former canonical raw transport.

Focused worker suite, executed twice during this phase:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q tests/test_chapter06_synthesis_oracle_worker.py

9 passed in 24.96s
9 passed in 12.57s
```

Static checks:

```text
python -m ruff check \
  scripts/chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_oracle_worker.py
All checks passed!

python -m ruff format --check \
  scripts/chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_oracle_worker.py
2 files already formatted

git diff --check -- \
  scripts/chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_oracle_worker.py \
  design/chapter06_synthesis_fixture_oracle_plan.md \
  design/chapter06_exact_source_contract.md
clean
```

Exact unchanged phase boundary:

| object | SHA-256 |
| --- | --- |
| worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| repaired plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| exact source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |
| compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` |
| deterministic compact writer | `3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb` |

## 9. Authority boundary and next gate

This report does not modify or authorize:

- `ACCEPTED_CAPTURE_KEY_COUNT`;
- `ACCEPTED_CAPTURE_SCHEMA_DIGEST`;
- any former accepted full fingerprint or plan lock;
- compact ownership, assembly, or deterministic final bytes;
- a publisher contract or detached authorization;
- any raw/candidate/golden file; or
- manifest registration or publication.

The next permitted step is an independent worker rebind audit of this exact
candidate report, worker, tests, repaired plan, two-capture evidence, raw
transport, member delta, read set, and source/data immutability proof.

Final disposition: **CANDIDATE ONLY**.
