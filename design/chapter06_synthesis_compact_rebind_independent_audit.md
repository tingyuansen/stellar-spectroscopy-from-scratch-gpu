# Chapter 6 synthesis compact-assembler rebind independent audit

Status: independent phase-2 read-only audit  
Audited: 2026-07-30  
Disposition: **ACCEPT the exact compact-assembler rebind as authority for a
phase-3 deterministic-writer repair candidate and its separate independent
audit only**

No writer, publisher contract, byte-acceptance record, authorization, data,
manifest, fixture, golden, or publication is accepted or authorized here.
No assigned input or downstream boundary was edited. This report is the only
repository file created by the audit.

## 1. Exact assigned snapshot

All assigned phase-2 objects matched before execution:

| object | SHA-256 |
| --- | --- |
| repaired compact assembler | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| repaired compact-assembler tests | `25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153` |
| compact-rebind candidate report | `54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c` |
| phase-1 plan-rebind candidate | `dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93` |
| phase-1 independent acceptance | `9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7` |
| repaired synthesis plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| unchanged scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| unchanged worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |

The pinned Payne Zero checkout remained:

```text
9c44001feae40b85146630499e6f8a5fed42e5af
```

## 2. Exact authority and fail-closed ordering

**PASS.** Before copying or projecting a raw mapping, the repaired public
assembler now requires:

1. the worker's `DESIGN_PATH` to resolve to the canonical repaired plan;
2. that plan, the phase-1 candidate, and the phase-1 independent audit to be
   regular nonsymlink files at their exact accepted hashes;
3. the exact unchanged worker hash;
4. exactly 754 object-free, detached raw members;
5. the exact accepted raw schema;
6. exact raw schema, physical fingerprint, full fingerprint, worker hash,
   design hash, capture scope, and accepted-scope scalars;
7. false golden-read and golden-publication flags;
8. recomputation of the physical and full fingerprints; and
9. the worker's complete cross-regime scientific validator.

Only after that boundary passes does `assemble_compact_candidate` construct
the compact mapping and ownership ledger.

The accepted raw boundary is:

| property | exact value |
| --- | --- |
| members | `754` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical-payload fingerprint | `51371e5c0db1fae7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full-capture fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |
| repaired design hash | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| scope | `accepted exhaustive Chapter 6 CPU one-line synthesis capture` |

Independent live probes rejected:

| adversarial raw boundary | exact failure |
| --- | --- |
| historical design plus historical full fingerprint | `accepted raw identity changed: meta__full_capture_fingerprint` |
| arbitrary wrong live full fingerprint | `accepted raw identity changed: meta__full_capture_fingerprint` |
| historical design paired with live full fingerprint | `accepted raw identity changed: meta__design_sha256` |
| arbitrary wrong physical fingerprint | `accepted raw identity changed: meta__physical_payload_fingerprint` |

The focused suite additionally rejected changed phase-1 authority bytes,
removed/added raw members, unaccepted scope, and scientific payload
mutations. The historical boundary cannot enter the projection path merely
because its physical fingerprint remains valid.

## 3. Two unrelated live captures

Two new child processes used
`/Users/ysting/anaconda3/bin/python` under the complete worker controls:

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

Each received a distinct parent-created, existing, empty, external,
nonsymlink cache directory. Both caches remained empty and were disposed
before the audit returned.

| process evidence | capture A | capture B |
| --- | --- | --- |
| child PID | `45346` | `45350` |
| cache-root SHA-256 | `0eb32a9cb21c41cb8ea86bf860d550df2b50fc2ed678ea510c8fd27298e99364` | `1e4508396ae598ab3f343d2c4a2890e115d9a17cc73046955e382881a793a87d` |
| cache empty before/after | yes | yes |
| cache disposed | yes | yes |

Each child independently executed:

```python
raw = worker.build_oracle_results()
assembly = compact.assemble_compact_candidate(raw)
compact.validate_compact_candidate(assembly)
```

After excluding only PID and cache-path evidence, the two complete audit
summaries were identical. Their per-member hash-manifest digest was
`e7abaf1c49c02514c19713632ce5741bcc2d4022259726c24ca31383e67cbd60`;
their complete schema-description manifest digest was
`418f10eff8c300416af38e753db98b86695f1e46dac33ca7e8b2d92ba21f9fb6`.

## 4. Live compact result

Both processes reproduced:

| compact property | exact live value |
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

The complete mapping digest binds each lexical member-name length and name,
dtype, shape, and contiguous C bytes. It is distinct from the payload
fingerprint, whose self-fingerprint field is deliberately excluded.

All 213 arrays were object-free, C-contiguous, and in lexical order. Every
schema description matched its array's name, dtype, shape, and axis rank;
all units were nonempty. The only declared roles were `axis` and
`comparison-only`.

## 5. Independently reconstructed historical projection

The historical reference was reconstructed in memory by restoring only the
four formerly accepted scalar values in a detached copy of the live compact
mapping:

```text
meta__assembler_sha256
  62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a

meta__compact_payload_fingerprint
  ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9

meta__design_sha256
  d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565

meta__full_capture_fingerprint
  33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b
```

This is evidence-backed rather than an assumed field substitution: the
detached reference independently reproduced every old compact control:

| historical compact property | reproduced value |
| --- | --- |
| members | `213` |
| array bytes | `1,235,275` |
| schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| payload fingerprint | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` |
| complete mapping digest | `5067397b128cbbcf76402830d9daac6f683b994e610a6e52815a469870388f63` |
| raw-ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |

The old payload fingerprint is the previously independently accepted compact
identity. Reproducing it controls the historical projection as a complete
byte mapping, not merely as a prose expectation.

## 6. Exact old/live delta

The exact differing-member set is:

```text
meta__assembler_sha256
meta__compact_payload_fingerprint
meta__design_sha256
meta__full_capture_fingerprint
```

All four are scalar `<U64` arrays in both mappings. The remaining 209 members
are identical in name, dtype, shape, and contiguous C bytes. All 213 member
names and schema descriptions remain unchanged.

The new payload fingerprint is required rather than incidental. Its
calculation excludes only `meta__compact_payload_fingerprint` itself and
therefore binds the changed assembler, plan, and raw full-capture provenance
values. Carrying `ce5d1c1d…` forward would falsely describe the live compact
mapping.

No scientific axis, opacity slab, catalog field, invariant, factor/branch
ledger value, continuum digest/sample, activity mask, coarse-grid evidence,
source/table identity, or ownership entry changed.

## 7. Complete ownership and no-copy role

The raw-ownership tuple contains exactly one lexical, unique entry for every
one of the 754 live raw names and no extra name:

| disposition | count |
| --- | ---: |
| `final` | `250` |
| `derived_digest_only` | `123` |
| `intentionally_ephemeral` | `381` |
| total | `754` |

Every `final` target resolves to an existing compact member. The exact tuple
digest remains
`5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675`.

An all-pairs NumPy storage audit found zero shared-memory pairs between the
213 compact arrays and 754 raw arrays. Mutating the raw mapping after
assembly leaves the compact output unchanged.

The role inventory independently found:

- no `(6,6,139)` copied state;
- no full `(6,400)` or `(4,6,400)` coarse slab;
- no numerical Harris/FASTEX table of length 1001 or 2001;
- no continuum absorption, scattering, total, or float32 input slab;
- no loop gross/net slab;
- no full stimulated-emission-factor field;
- no atmosphere member or raw source row;
- exactly two `(4,6,6000)` dense arrays:
  `opacity__gross_float32` and `opacity__net_float32`.

Continuum terms survive only as center/wing samples and full-array hashes.
Harris tables survive only as member identities. No compact array other than
the required gross and net slabs exceeds 100 kB.

The focused adversarial suite also reconfirmed exact gross/net stimulation
reconstruction, loop/batched deduplication, continuum float64-sum/float32
fence reconstruction, Harris deduplication, payload/schema/ownership mutation
rejection, and detached storage.

## 8. No publication or filesystem authority

**PASS.** The module exposes one in-memory raw-to-compact construction
argument and no:

- command-line entry point or parser;
- output/destination path;
- serializer or file writer;
- golden or manifest path;
- authorization parser;
- publisher, overwrite, rename, replace, or alternate-root operation.

The returned mapping records false for publication authorization and golden
publication. The audit created no raw or compact NPZ. Temporary cache
directories were disposable process-control state outside all source/data
trees.

## 9. Source, data, and downstream immutability

The bound scientific-source snapshot contained the unchanged worker/tests,
repaired plan and source contract, both phase-1 authority records, repaired
assembler/tests, five staged executed sources, 17 pinned Python sources, the
full atomic source archive, and three pinned upstream static tables.

Each aggregate digest binds sorted resolved path length/path bytes, file-byte
length, and binary SHA-256 using little-endian 64-bit lengths.

| snapshot | exact before/after result |
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
| source before/after | exact |
| data before/after | exact |
| manifest before/after | exact |

The prospective synthesis publication directory remained absent.

The prohibited downstream boundary also remained exact:

| unchanged downstream object | SHA-256 |
| --- | --- |
| deterministic compact writer | `3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb` |
| writer tests | `9601f8717a29ef51f32a62fd0a73c4d3db4b41c4f6ac08374be6149df4030bfc` |
| writer candidate report | `540cf57126df93ee34d02c3da446a6ef109b93a8b17d60514439e12a8f63fc71` |
| lane-artifact publisher contract | `9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774` |
| byte-acceptance rejection | `474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1` |

The writer still binds the historical assembler/plan/raw/compact identities.
That expected failure is proof that phase 2 did not self-authorize phase 3.

## 10. Commands and results

Exact input and downstream hashing:

```text
sha256sum \
  scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py \
  design/chapter06_synthesis_compact_rebind_candidate.md \
  design/chapter06_synthesis_plan_rebind_candidate.md \
  design/chapter06_synthesis_plan_rebind_independent_audit.md \
  design/chapter06_synthesis_fixture_oracle_plan.md \
  scripts/chapter06_synthesis_oracle_worker.py \
  <unchanged downstream boundary> \
  data/MANIFEST.json

all matched the values in Sections 1 and 9
```

The two independent live/historical/ownership captures used:

```text
/Users/ysting/anaconda3/bin/python -c \
  <in-memory raw capture, assembler, historical reference, and role audit>
```

under the environment in Section 3. They produced the exact results in
Sections 2–9.

Focused assembler suite:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_chapter06_synthesis_compact_assembler.py

5 passed in 12.05s
```

Combined worker and assembler suites:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_compact_assembler.py

14 passed in 20.40s
```

Static checks:

```text
/Users/ysting/anaconda3/bin/python -m ruff check \
  scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
All checks passed!

/Users/ysting/anaconda3/bin/python -m ruff format --check \
  scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
2 files already formatted

git diff --check -- \
  scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
clean
```

## 11. Findings by priority

### P0

No P0 defect. The repaired assembler binds the exact accepted phase-1
authority, rejects historical and mismatched raw boundaries before
projection, and reproduces an unchanged scientific compact projection.

### P1

No P1 defect. The four-member provenance delta, new payload/mapping
identities, exhaustive ownership, storage detachment, no-copy role, source
and data immutability, and no-publication surface all reproduced.

### P2

No phase-2 P2 defect. Exact live identities are frozen by the candidate and
this independent audit rather than silently added to unchanged worker tests.
The next writer repair must cite this phase-2 evidence explicitly.

## 12. Final authority

**ACCEPT** repaired assembler
`583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8`
and phase-2 candidate
`54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c`
as the compact-rebind boundary for a separately implemented and audited
phase-3 deterministic-writer repair.

Phase 3 may bind:

- repaired assembler
  `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8`;
- live raw full fingerprint
  `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893`;
- live compact payload fingerprint
  `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b`;
- live compact mapping digest
  `d46bd8c7096ffcbd3d4247f673930c49a5c513f9408909c55cbcba2440484ab8`;
- unchanged compact schema
  `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde`;
  and
- unchanged ownership digest
  `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675`
  with counts `250/123/381`.

This acceptance does not authorize any writer change by itself, candidate
archive, final byte identity, publisher change, authorization, manifest
entry, golden, or publication. All require their own forward-cascade gates.
