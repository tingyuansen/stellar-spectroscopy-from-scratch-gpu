# Chapter 6 synthesis scientific-worker plan-rebind independent audit

Status: independent phase-1 read-only rebind audit  
Audited: 2026-07-30  
Disposition: **ACCEPT the exact repaired-plan scientific-worker rebind as
authority for the next independently reviewed compact-assembler repair gate
only**

This audit does not accept a repaired assembler or writer, refresh any
downstream constant, construct or publish a compact candidate, create a
fixture or golden, alter `data/MANIFEST.json`, or authorize publication.
`design/chapter06_synthesis_plan_rebind_candidate.md` remained a candidate
until this independent review.

No worker, test, plan, assembler, writer, prior audit, source, data, manifest,
external Payne Zero, or paper file was changed. This report is the only
repository file created by the audit.

## 1. Exact review snapshot

The assigned phase-1 inputs matched their expected identities before
scientific execution:

| reviewed object | SHA-256 |
| --- | --- |
| plan-rebind candidate | `dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93` |
| unchanged scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| unchanged worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| repaired synthesis fixture/oracle plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| forward-cascade trigger rejection | `474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1` |
| accepted global plan-repair audit | `048a9d970e7a56fd48e2b0cfaa5632b968875f1f741f4d32c2dc84ae6c5d3e6c` |
| exact Chapter 6 source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |

The pinned Payne Zero checkout remained:

```text
9c44001feae40b85146630499e6f8a5fed42e5af
```

The trigger report correctly rejected the old writer boundary rather than
silently carrying old candidate bytes across the plan change. The accepted
global repair audit independently binds the repaired plan at the exact live
hash above.

## 2. Repaired plan and exact interface

**PASS.** The bounded plan repair is both accepted and source-exact.

The pinned synthesis grid is:

```python
@dataclass(frozen=True)
class Grid:
    start_wavelength_nm: float
    end_wavelength_nm: float
    resolution: float
```

Its ratio is `1.0 + 1.0 / self.resolution`. The repaired plan now uses:

- the exact constructor keyword `resolution=300000.0` for the canonical
  6000-point grid;
- the exact attribute spelling `Grid.resolution=20000` for the coarse
  400-point probe; and
- mathematical \(R_{\rm grid}\), rather than source-like lowercase
  `r_grid`, when referring to the dimensionless grid density as a quantity.

A literal search found no lowercase `r_grid` in the repaired plan. The
spelling therefore neither invents a nonexistent `Grid.r_grid` attribute nor
confuses this intrinsic logarithmic-grid density with instrumental resolving
power.

The global repair evidence identifies this as a prose/interface correction.
It changes no wavelength bound, grid value, fixture, source record, table,
dtype, branch, kernel, cutoff, or numerical policy.

## 3. Independent fresh-process construction

Two new scientific captures were launched with
`/Users/ysting/anaconda3/bin/python`. Each process received the exact accepted
controls:

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

The parent created a different existing, empty, external, nonsymlink
`NUMBA_CACHE_DIR` for each child. Both remained empty after execution and
were absent after their temporary-directory contexts closed.

| process evidence | capture A | capture B |
| --- | --- | --- |
| child PID | `40649` | `40656` |
| cache-root SHA-256 | `ad478b8019fdf77810fd13fbda7186ad2cf511dffde508f6f07fb1bb34fca349` | `87737a639f863a141e5421362bec1167338689626730f44f5db0f782cf839a53` |
| cache empty before/after | yes | yes |
| cache disposed | yes | yes |

Each child called only:

```python
from scripts import chapter06_synthesis_oracle_worker as worker

results = worker.build_oracle_results()
```

The stale compact assembler, accepted zero-argument writer, and any
publication path were not invoked. After the worker returned, an in-memory
canonical NPY-v2/ZIP-stored codec transported the raw mapping through the
subprocess pipe for comparison. No raw NPZ was written to disk. A separately
implemented canonical encoder reproduced the same whole-transport identity;
the existing private raw codec was used only as a cross-check.

## 4. Reproduced live identities

The two complete mappings were identical member for member, including name,
dtype string, shape, and contiguous C bytes. Their canonical raw transports
were byte-identical.

| property | independently reproduced value |
| --- | --- |
| member count | `754` |
| total array bytes | `8,451,402` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical-payload fingerprint | `51371e5c0db1fae7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full-capture fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |
| complete mapping digest | `09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c` |
| canonical raw transport bytes | `8,689,108` |
| canonical raw transport SHA-256 | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` |

The complete mapping digest uses the accepted raw-boundary encoding of the
member-name length, member name, dtype, shape, and contiguous bytes. The
worker's older `array_mapping_digest` is a different digest convention and
is not substituted for this value.

Both captures retained:

- exact canonical and coarse activity masks
  `111000`, `111111`, `111111`, `111111`;
- canonical active reach from `5` through `163`;
- eight complete stimulated-emission-factor members;
- exact net reconstruction for loop and batched routes;
- zero loop/batched maximum absolute difference;
- all 17 frozen loaded pinned Python sources;
- `meta__golden_read_performed=False`; and
- `meta__golden_publication_performed=False`.

All 754 arrays were object-free and C-contiguous.

## 5. Historical reconstruction and exact delta

The historical mapping was reconstructed in memory by restoring only:

```text
meta__design_sha256
  d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565

meta__full_capture_fingerprint
  33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b
```

This reconstruction was not accepted merely because those were the expected
names. It independently reproduced every former whole-object identity:

| historical control | reproduced value |
| --- | --- |
| full-capture fingerprint | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` |
| physical-payload fingerprint | `51371e5c0db1fae7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| complete mapping digest | `e2a3dd9670fd9fb5ed34f5131c303687d72ff2d3e762d9224c702f1dffc4a775` |
| canonical raw transport bytes | `8,689,108` |
| canonical raw transport SHA-256 | `5ff93594781de6fc12a3a2a0adf0ebcc03c81f6141b8648e709ad306686b93a3` |

The exact old/new differing-member set was:

```text
meta__design_sha256
meta__full_capture_fingerprint
```

Both retain scalar shape and `<U64` dtype. All other `752` members have
identical names, dtypes, shapes, and C bytes. The two equal-length canonical
transports differ at exactly `135` byte positions, all explained by the two
changed NPY payloads and their ZIP checksums.

The direct scientific subset consists of all names outside `meta__` and
`identity__`:

| direct scientific proof | result |
| --- | --- |
| scientific members | `670` |
| scientific array bytes | `8,433,496` |
| direct old/new dtype-shape-C-byte equality | all `670` pass |
| scientific mapping digest | `c05be60a58b5db26544dca8fe7f53e510d0b7d30b70e801b59337c1e41a405b0` |

Thus the unchanged physical fingerprint is supported by a complete direct
comparison, not used as a substitute for one.

## 6. Read set and no-write boundary

A Python audit hook was installed before importing the worker in each child.
The scientific read set was frozen immediately after
`build_oracle_results()` returned and before the transport codec was
imported.

Both children produced the identical sorted in-scope path set:

| read-set property | independent result |
| --- | --- |
| unique repository/pinned/paper paths or probes | `53` |
| existing regular files | `35` |
| absent import probes | `18` |
| sorted read-set SHA-256 | `475fc8d5f7d68af5994bcf701fa75e89ae924c9fc4c1b3b5459dfb83a21cd174` |
| in-scope write-mode paths | `0` |

The existing set covers the exact worker/plan/contract, staged and pinned
loaded source, source archive, fixture, subset, and staged/upstream static
tables. The absent paths are import probes, primarily `.pyc` candidates;
they are recorded rather than misrepresented as reads of existing
scientific data. The paper tree supplied no scientific input.

The worker has no output, destination, serialize, publish, or golden
parameter. Its CLI rejects output/publication arguments. The focused suite
reconfirmed the canonical input-only fixture/subset boundary and rejection
of golden, alternate, symlink, occupied-cache, and symlink-cache paths.

## 7. Source, data, and manifest immutability

The before/after scientific-source snapshot included exactly:

- the worker, worker tests, repaired plan, and exact source contract;
- five staged executed synthesis modules;
- all 17 frozen pinned Python sources;
- the full atomic source archive; and
- the three pinned upstream static synthesis tables.

Each aggregate digest encodes each sorted resolved path length/path, file
length, and binary SHA-256 using little-endian 64-bit lengths.

| snapshot | exact before/after value |
| --- | --- |
| scientific-source regular files | `30` |
| scientific-source bytes | `259,414,135` |
| scientific-source aggregate SHA-256 | `a945cc441f271e533664ba3cd263c74e8abaa3a0af664d048d3bb6623eba32f0` |
| scientific-source symlinks | `0` |
| `data/` directories | `12` |
| `data/` regular files | `39` |
| `data/` regular-file bytes | `30,046,405` |
| complete `data/` aggregate SHA-256 | `288bbe4c6bcbd20da8390f99fb6cc45e07ee6eed24197371d3520aed39f7d004` |
| `data/MANIFEST.json` bytes | `1,087,741` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| source before/after | exact |
| data before/after | exact |
| manifest before/after | exact |

The prospective synthesis destination remained absent:

```text
data/golden/payne_zero/chapter06/synthesis
```

No raw transport, candidate NPZ, fixture, golden, authorization, destination
directory, or manifest entry was created.

## 8. Downstream non-repair proof

Phase 1 did not silently refresh the downstream acceptance chain:

| deliberately unchanged downstream object | SHA-256 |
| --- | --- |
| compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` |
| compact-assembler tests | `53111433aa3082a58be5ec8b3da1a961f330eac1bb805ac26d1fc6625487e42d` |
| compact candidate report | `41daee0a8b4239fd60a2e67b028afa1a0197ac3ae040a30f5dfd8795234b3550` |
| compact-assembler independent audit | `a0530cd08d5f0ddcc96b51fdaab4520aa89e62ca65850cf855ad4ede22251a33` |
| deterministic compact writer | `3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb` |
| writer tests | `9601f8717a29ef51f32a62fd0a73c4d3db4b41c4f6ac08374be6149df4030bfc` |
| writer candidate report | `540cf57126df93ee34d02c3da446a6ef109b93a8b17d60514439e12a8f63fc71` |
| writer independent acceptance | `b888b49226e8ca6407c8226a3c021efb88fa100623fef27dd62e9beba43f2535` |

The live assembler still requires the historical full fingerprint
`33d1dec…`; the live writer still binds the historical plan hash
`d6e6ae…`, historical raw transport `5ff935…`, and historical raw mapping
digest `e2a3dd…`. Those stale constants are expected fail-closed evidence that
the cascade has not been bypassed. They are not accepted for refresh by this
phase.

## 9. Verification executed

Exact identity pass:

```text
sha256sum \
  design/chapter06_synthesis_plan_rebind_candidate.md \
  scripts/chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_oracle_worker.py \
  design/chapter06_synthesis_fixture_oracle_plan.md \
  design/chapter06_synthesis_candidate_byte_acceptance.md \
  design/global_interface_molecular_repair_independent_audit.md \
  design/chapter06_exact_source_contract.md \
  <unchanged downstream chain> \
  data/MANIFEST.json

all matched the values recorded in Sections 1, 7, and 8
```

The two principal scientific captures used:

```text
/Users/ysting/anaconda3/bin/python -c <audit-hook in-memory worker capture>
```

under the full environment in Section 3, with distinct temporary caches.
They produced the exact values in Sections 4–7. An additional unrelated
two-process repeat reproduced the live and reconstructed historical raw
transport identities, the two-member delta, all 752 unchanged members, all
670 scientific members, and 135 changed transport positions.

Focused worker suite:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py

9 passed in 8.42s
```

Static checks:

```text
/Users/ysting/anaconda3/bin/python -m ruff check \
  scripts/chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_oracle_worker.py
All checks passed!

/Users/ysting/anaconda3/bin/python -m ruff format --check \
  scripts/chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_oracle_worker.py
2 files already formatted
```

Exact-interface probes:

```text
rg -F 'r_grid' design/chapter06_synthesis_fixture_oracle_plan.md
# no matches

rg -F 'Grid.resolution' design/chapter06_synthesis_fixture_oracle_plan.md
243:`Grid.resolution=20000`. The

rg -F 'resolution=300000.0' \
  design/chapter06_synthesis_fixture_oracle_plan.md
217:    resolution=300000.0,
```

## 10. Findings by priority

### P0

No P0 defect. The repaired plan is independently accepted at the exact live
hash, uses the pinned interface, and changes no scientific byte. Both fresh
captures and the exact historical-transport control close the scientific
equivalence question.

### P1

No P1 defect. Fresh-process controls, complete member equality, read-set
identity, no-write evidence, source/data immutability, and downstream
non-repair all reproduced.

### P2

No phase-1 P2 defect. The plan retains its design-era lifecycle header; this
is historical planning prose, not an interface or scientific input claim.
Changing it during this gate would create another plan hash and restart the
cascade. Current lifecycle authority is intentionally carried by the
candidate and independent audit records.

## 11. Final authority

**ACCEPT** the exact plan-rebind candidate
`dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93`
as the independent scientific-worker rebind for repaired plan
`413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856`.

This acceptance authorizes only the next compact-assembler repair candidate
to bind:

- the repaired plan identity;
- raw transport
  `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e`;
- complete mapping digest
  `09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c`;
- unchanged schema
  `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178`;
- unchanged physical fingerprint
  `51371e5c0db1fae7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc`;
  and
- full fingerprint
  `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893`.

The assembler repair itself requires a separate candidate and independent
audit. This report does not authorize changing or accepting the deterministic
writer, any publisher, detached authorization, candidate/final NPZ,
`data/MANIFEST.json`, golden, or publication.
