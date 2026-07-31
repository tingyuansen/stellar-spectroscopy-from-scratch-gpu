# Chapter 6 synthesis compact-assembler rebind candidate

Status: **CANDIDATE ONLY — independent compact-rebind audit required**

Scope: phase 2 of the forward repair cascade after the accepted
scientific-worker plan rebind

Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

Authorized implementation surface:

```text
scripts/chapter06_synthesis_compact_assembler.py
tests/test_chapter06_synthesis_compact_assembler.py
design/chapter06_synthesis_compact_rebind_candidate.md
```

No raw worker/test/plan, historical candidate/audit, compact writer/test/report,
publisher contract, byte-acceptance record, data file, manifest, or external
source was edited by this phase. No writer or publication repair was started.

## 1. Candidate disposition

The compact assembler now accepts only the repaired-plan raw boundary
independently authorized by:

| authority | SHA-256 |
| --- | --- |
| worker plan-rebind candidate | `dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93` |
| worker plan-rebind independent audit | `9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7` |
| repaired synthesis fixture/oracle plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| unchanged scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |

The exact accepted raw values are:

| raw property | value |
| --- | --- |
| members | `754` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical-payload fingerprint | `51371e5c0db1fae7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full-capture fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |
| design SHA-256 | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| scope | `accepted exhaustive Chapter 6 CPU one-line synthesis capture` |

Two unrelated fresh controlled processes passed that boundary and produced
identical compact mappings. The scientific projection and complete ownership
ledger are unchanged, but the compact payload fingerprint is necessarily new
because the compact mapping deliberately retains three changed provenance
values: the repaired plan hash, repaired-plan raw full fingerprint, and
repaired assembler hash.

Measured live compact result:

| compact property | candidate value |
| --- | --- |
| members | `213` |
| array bytes | `1,235,275` |
| schema version | `1` |
| schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| payload fingerprint | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| complete mapping digest | `d46bd8c7096ffcbd3d4247f673930c49a5c513f9408909c55cbcba2440484ab8` |
| raw-ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| publication authorized | `False` |
| golden publication performed | `False` |

This report does not accept those values. It presents them for an independent
compact-rebind audit.

## 2. Minimal explicit implementation repair

The former assembler accepted the old raw full fingerprint:

```text
33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b
```

The repaired assembler changes that one raw scientific-acceptance constant to
the independently authorized live value:

```text
8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893
```

It also makes the provenance refresh explicit rather than silently replacing
a fingerprint:

- the repaired plan, phase-1 candidate, and phase-1 independent audit have
  canonical repository paths and exact SHA-256 constants;
- all three must be regular nonsymlink files at those identities;
- the worker's `DESIGN_PATH` must resolve to the bound plan path;
- live raw `meta__design_sha256` must equal the accepted repaired plan hash;
- the unchanged worker bytes, 754-member count, raw schema, physical
  fingerprint, scope, lifecycle flags, recomputed physical/full fingerprints,
  and complete science checks remain required.

No raw schema, ownership rule, reduction function, compact schema, output
member, serializer, destination, or publication surface was added or removed.

Exact implementation identities:

| object | SHA-256 |
| --- | --- |
| repaired compact assembler | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| repaired focused tests | `25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153` |

## 3. Two fresh live raw captures

Both children used `/Users/ysting/anaconda3/bin/python` with:

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

Each child received a different parent-created, existing, empty, external,
nonsymlink `NUMBA_CACHE_DIR` and a one-use origin token. Each child executed:

```python
raw = worker.build_oracle_results()
assembly = compact.assemble_compact_candidate(raw)
```

Process evidence:

| evidence | capture A | capture B |
| --- | --- | --- |
| child PID | `43256` | `43300` |
| origin-token SHA-256 | `6f87d54eb025235fdcaaf3725ad8a7a41567ef06fcfd2047418d0caeba8966d9` | `fd972306f9bd10ca2e60837e77a3374d09c6305dd295bc5b6f80b7891664a7ed` |
| cache-root SHA-256 | `d2484958017bb2fb921db9c496b77e247f8418456a15aa7349d4be9df229c581` | `0d2f110cb1cf57c2dcd536a42ac80ee741ac04bbe236a9d462fb258c5b2ae2d5` |
| cache empty before/after | yes | yes |
| cache disposed | yes | yes |

The two processes independently agreed on every deterministic result recorded
in Sections 1, 4, and 5. Their raw observations had the exact accepted count,
schema, physical fingerprint, and new full fingerprint. Their compact
member-hash manifests, schema descriptions, ownership tuples, summaries, and
complete mapping digests were identical.

No raw or compact NPZ was written. All comparison material remained in child
memory or subprocess pipes.

## 4. Independently reconstructed historical control

The historical raw capture was reconstructed in memory by restoring only:

```text
meta__design_sha256
  d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565

meta__full_capture_fingerprint
  33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b
```

Before projection, the audit harness independently required its old
754-member count, unchanged schema and physical fingerprint, recomputed old
full fingerprint, accepted scope, lifecycle flags, and complete worker
science validation. The repaired public assembler then rejected it exactly:

```text
accepted raw identity changed: meta__full_capture_fingerprint
```

For comparison only, the audit harness applied the unchanged private
raw-to-compact projection functions after those historical checks, restored
the historical assembler identity, and recomputed the compact payload
fingerprint. This reference projection was never returned by the repaired
public acceptance boundary.

The historical control reproduced:

| historical compact control | reproduced value |
| --- | --- |
| members | `213` |
| array bytes | `1,235,275` |
| schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| payload fingerprint | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` |
| complete mapping digest | `5067397b128cbbcf76402830d9daac6f683b994e610a6e52815a469870388f63` |
| raw-ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |

Reproducing the old independently accepted compact payload fingerprint is the
control that makes the old/live member comparison byte-based rather than an
inference from the intended prose-only plan repair.

## 5. Exact compact delta

All 213 names, dtypes, shapes, and schema descriptions are identical between
the reconstructed historical compact mapping and both live compact mappings.
The raw ownership tuples are also identical entry for entry.

Exactly four scalar `<U64` members have different C bytes:

| member | old value | new value |
| --- | --- | --- |
| `meta__assembler_sha256` | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| `meta__compact_payload_fingerprint` | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| `meta__design_sha256` | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| `meta__full_capture_fingerprint` | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |

Exact member identities, where `member_sha256` covers dtype, shape, and C
bytes:

| member | old member SHA-256 | new member SHA-256 | old C-byte SHA-256 | new C-byte SHA-256 |
| --- | --- | --- | --- | --- |
| `meta__assembler_sha256` | `d1861f234759f05481e899150e4cfe1aaef0e9fdf039a10c0095da8d6eb60090` | `2132b8568ca161776c9f952ebab6febb8a0e45cece75fde998b18773866d64a8` | `edd144fa87b37ec396b7d67c4c76c9c4246555c09d13fcf86a11f3e9776ea6ea` | `9fda687099cece241f20b14ab2e09192c5aaf6b91e33d20896596627995b529a` |
| `meta__compact_payload_fingerprint` | `baf3ae7a59281aa3ea15f6ab351c9cc5ee66cfe6f47d304e600b94c414faf1d6` | `060b1fd14670e50608f26a6e36091a8da4005adf86f78cb6b227b94cefbf741d` | `f8b696459767ecb885ee758a2ef97f15e82216565364d45b5f7cbcbbb913b3cb` | `3bf4e2b12b08c797fef190f1d3f635a3cc7ae0c580eada0312605e892dee80f9` |
| `meta__design_sha256` | `12f346e9976a9304192f78a800b603fdd3dbc14ae94e621979dd7f2c533ef1f4` | `262f5e4395fededf7431d704ccc9fec6ccc058a5092ef9ef164c284cd4f40cc6` | `061c11922d36f56e8504fefc343706dfa9d44d2f431d8a2e04e7a7a7a245192c` | `96d7b7dfabd27de62e184d60e3f5d708e39b81aa7513a9a4da784fe0032db5df` |
| `meta__full_capture_fingerprint` | `6503de2407c4d6fdb08c60149c4604aea8b883b51f89c3ffd61332cde9486ad5` | `8532b4511b8889430e4cc47f8162637531a36f20c8a7ca58807b2f86417c4ac0` | `fb3259cc9c8f0fa9dfa3654666589f44f4c9c61124a3285d360a645204db4c12` | `19468e78702fa6eefc61243087357689673a299b5dd5925c70aa36fc6d043f41` |

The remaining 209 compact members are exactly equal in name, dtype, shape,
and C bytes. Thus:

- member count `213` is unchanged;
- total array bytes `1,235,275` are unchanged;
- schema digest `911d06d9…` is unchanged;
- all scientific arrays, axes, reductions, checks, and source identities not
  named above are unchanged;
- ownership digest `5594db24…` and all 754 ownership entries are unchanged;
- payload fingerprint `ce5d1c1d…` is **not** unchanged, because it correctly
  binds the three updated provenance values.

It would be incorrect to carry the historical payload fingerprint forward.

## 6. Ownership, reduction, and authority gates

The full ownership result remains:

| disposition | count |
| --- | ---: |
| `final` | `250` |
| `derived_digest_only` | `123` |
| `intentionally_ephemeral` | `381` |
| total | `754` |

The tuple remains lexical and unique, every final target resolves, and its
complete digest remains
`5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675`.

The existing tests continue to cover:

- removed and added raw members;
- incomplete or changed ownership;
- raw physical-payload mutation;
- stimulated-factor/net reconstruction;
- loop/batched equality;
- continuum sum and float32 fence reconstruction;
- Harris table deduplication;
- compact payload, schema-description, and ownership mutation;
- detached candidate storage and absence of forbidden copied slabs/tables;
- false publication and golden-publication flags.

This repair adds exact failure evidence for:

- a changed accepted plan-rebind audit identity;
- the historical raw full-fingerprint boundary;
- an arbitrary wrong live full fingerprint;
- the historical plan SHA paired with the live fingerprint; and
- an arbitrary wrong physical fingerprint.

The module still exposes no command-line entry point, serializer, output path,
destination, manifest mutation, authorization parser, or publication
function.

## 7. Source, data, and no-publication boundary

The bound source snapshot contains:

- worker, worker tests, repaired plan, and exact source contract;
- accepted phase-1 candidate and independent audit;
- repaired compact assembler and tests;
- five staged executed synthesis modules;
- all 17 frozen pinned Python sources;
- the full atomic source archive; and
- the three pinned upstream static synthesis tables.

The baseline after the authorized implementation edit is:

| snapshot | value |
| --- | --- |
| bound source regular files | `34` |
| bound source bytes | `259,521,581` |
| bound source digest | `838b31ecae0b3e586fd9a0d14f2b4e6e25e1f0d5595354a485d31a65db8095b4` |
| bound source symlinks | `0` |
| `data/` directories | `12` |
| `data/` regular files | `39` |
| `data/` regular-file bytes | `30,046,405` |
| complete `data/` digest | `288bbe4c6bcbd20da8390f99fb6cc45e07ee6eed24197371d3520aed39f7d004` |
| `data/MANIFEST.json` bytes | `1,087,741` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |

Each aggregate digest binds sorted resolved path length/path bytes, file-byte
length, and binary SHA-256. The final post-verification snapshot reproduced
every value in the table exactly.

The prospective synthesis destination remains absent and the manifest has no
synthesis-golden entry. No raw archive, compact archive, fixture, golden,
authorization, destination directory, or manifest mutation was created.

## 8. Verification

Focused repaired assembler suite:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q tests/test_chapter06_synthesis_compact_assembler.py

5 passed in 21.24s
```

Combined unchanged-worker and repaired-assembler suites:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_compact_assembler.py

14 passed in 24.21s
```

Static checks:

```text
python -m ruff check \
  scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
All checks passed!

python -m ruff format --check \
  scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
2 files already formatted

git diff --check -- \
  scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
clean
```

Unchanged prohibited boundaries:

| object | SHA-256 |
| --- | --- |
| raw worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| raw worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| repaired plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| historical compact candidate | `41daee0a8b4239fd60a2e67b028afa1a0197ac3ae040a30f5dfd8795234b3550` |
| historical compact audit | `a0530cd08d5f0ddcc96b51fdaab4520aa89e62ca65850cf855ad4ede22251a33` |
| compact writer | `3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb` |
| compact writer tests | `9601f8717a29ef51f32a62fd0a73c4d3db4b41c4f6ac08374be6149df4030bfc` |
| compact writer candidate | `540cf57126df93ee34d02c3da446a6ef109b93a8b17d60514439e12a8f63fc71` |
| publisher contract | `9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774` |
| byte-acceptance rejection | `474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1` |

## 9. Remaining gate

This candidate does not authorize updating the writer's accepted assembler
identity, plan identity, raw transport/mapping identity, compact payload
fingerprint, or expected final bytes. It does not authorize changing the
publisher contract, detached authorization, candidate-byte acceptance,
`data/MANIFEST.json`, or any golden.

The next permitted action is an independent read-only compact-rebind audit of
the exact implementation, tests, report, two fresh captures, historical
control, four-member compact delta, complete ownership ledger, and
source/data/no-publication evidence.

Final disposition: **CANDIDATE ONLY**.
