# Chapter 6 synthesis candidate-byte independent acceptance

Status: independent read-only candidate-byte gate

Audited: 2026-07-30

Disposition: **REJECT — the accepted zero-argument writer is no longer
reproducible against the live accepted source snapshot**

The failure is deliberate and correctly fail-closed. The exact accepted
writer binds the synthesis fixture/oracle plan at SHA-256
`d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565`.
That plan was subsequently repaired and independently accepted at SHA-256
`413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856`.
Two unrelated top-level calls to the exact zero-argument writer both rejected
this identity drift before starting either internally owned A/B scientific
capture.

Therefore this review does **not** accept or freeze the formerly observed
1,294,865-byte archive. It does not implement or accept a writer or publisher,
authorize publication, create a candidate file, write under `data/`, or alter
the manifest. This report is the only repository file created by the review.

## 1. Exact reviewed snapshot

All six assigned trust objects matched their requested identities:

| reviewed object | SHA-256 |
| --- | --- |
| `scripts/chapter06_synthesis_compact_writer.py` | `3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb` |
| `tests/test_chapter06_synthesis_compact_writer.py` | `9601f8717a29ef51f32a62fd0a73c4d3db4b41c4f6ac08374be6149df4030bfc` |
| `design/chapter06_synthesis_compact_writer_candidate.md` | `540cf57126df93ee34d02c3da446a6ef109b93a8b17d60514439e12a8f63fc71` |
| `design/chapter06_synthesis_compact_writer_independent_audit.md` | `b888b49226e8ca6407c8226a3c021efb88fa100623fef27dd62e9beba43f2535` |
| `design/chapter06_lane_artifact_publisher_contract.md` | `9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774` |
| `design/chapter06_lane_artifact_publisher_contract_independent_audit.md` | `7f2517ad0abcca312dcf22785e483fe033519264d5513b16ebe6fa580d4521fd` |

The pinned Payne Zero checkout remained read-only at commit:

```text
9c44001feae40b85146630499e6f8a5fed42e5af
```

The exact trust-object path used for this report is the path frozen by the
accepted publisher contract:

```text
design/chapter06_synthesis_candidate_byte_acceptance.md
```

## 2. Blocking identity drift

The writer's `ACCEPTED_FILE_IDENTITIES["fixture_oracle_plan"]` requires:

```text
design/chapter06_synthesis_fixture_oracle_plan.md
  expected:
  d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565

  live:
  413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856
```

The live plan is not unexplained local damage. Its current identity is
explicitly recorded by the later global interface/molecular repair:

| repair evidence | current SHA-256 |
| --- | --- |
| `design/global_interface_molecular_repair_candidate.md` | `6f9176bb8cc11e675fc2f441bd4465cb4f366bb95a3ee17bda1d3d23417261fd` |
| `design/global_interface_molecular_repair_independent_audit.md` | `048a9d970e7a56fd48e2b0cfaa5632b968875f1f741f4d32c2dc84ae6c5d3e6c` |
| repaired synthesis fixture/oracle plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |

That audit accepts the plan repair at the exact live hash and identifies the
change as replacing source-like lowercase `r_grid` prose with the actual
`Grid.resolution` interface or mathematical \(R_{\rm grid}\). Even if that
repair is expected to leave numerical physics unchanged, this byte gate may
not infer equivalence. The scientific worker places the plan SHA-256 in
`meta__design_sha256`; its full-capture fingerprint therefore binds the plan
bytes. The accepted compact assembler in turn requires the former exact
full-capture fingerprint. The identity change necessarily requires the
deliberate upstream repair/re-audit cascade described in Section 8.

The exact fail-closed exception in both top-level attempts was:

```text
accepted identity changed for fixture_oracle_plan:
413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856;
expected d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565
```

Both complete stderr streams had SHA-256
`69aeb4af7f516d7a9feb6675655e9d9a5c0baf6c41a7bd6ab7063240d0ac1022`.
Both processes returned nonzero, emitted zero stdout bytes, and started zero
scientific capture children.

## 3. Two unrelated top-level attempts

The contract requires two unrelated top-level zero-argument builds, with each
successful build retaining two writer-owned fresh A/B children. Two unrelated
top-level processes were started under the exact deterministic controls:

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

Each top-level process received a different existing, empty, external,
nonsymlink disposable cache root. Each cache remained empty and was disposed
after failure. The two results were:

| decision | attempt A | attempt B |
| --- | ---: | ---: |
| called exact zero-argument writer | yes | yes |
| rejected expected plan identity drift | yes | yes |
| stdout bytes | `0` | `0` |
| scientific A/B children started | `0` | `0` |
| candidate bytes returned | no | no |
| outer cache empty after failure | yes | yes |

The failure occurs in `verify_accepted_identities()` before the writer creates
its two internal cache roots or calls the scientific worker. Consequently it
would be false to claim two completed A/B builds, archive equality, or
candidate-byte reproduction.

Observed execution platform, recorded only as review context, was:

```text
Python:      3.13.9
executable:  /Users/ysting/anaconda3/bin/python
NumPy:       2.3.5
platform:    macOS-26.5.2-arm64-arm-64bit-Mach-O
machine:     arm64
byte order:  little
```

No PID, random origin token, or cache pathname is part of an accepted
scientific identity. No such inner-process evidence exists in this rejected
run because the source gate stopped before capture.

## 4. Former candidate facts are not accepted here

The assigned writer records the following former candidate:

| property | formerly observed value |
| --- | --- |
| archive bytes | `1,294,865` |
| archive SHA-256 | `b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850` |
| archive members | `213` |
| compact array bytes | `1,235,275` |
| compact schema version | `1` |
| compact schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| compact payload fingerprint | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` |
| raw ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| internal archive-kind wording | `synthesis_one_line_comparison_candidate` |

These values remain valid evidence for the old accepted source snapshot. They
are **not independently reproduced or accepted for the live repaired plan by
this report**. In particular:

- the old archive hash is not silently carried across the plan identity
  change;
- the internal archive-kind phrase remains lifecycle wording, not a repository
  role;
- no role `golden` exists until a later detached authorization and manifest
  registration;
- no typed Chapter 5 fixture, Chapter 6 subset, or static table is
  reclassified or copied merely because the former bytes existed.

## 5. Member, science, role, and adversarial gate disposition

Because no candidate byte string passed the source-identity gate, this review
did not pretend to decode or accept one. The required checks remain
**unexecuted for the live repaired snapshot**:

- exact 213-member set, lexical order, uniqueness, dtype, shape, C-byte hash,
  and 1,235,275 array-byte total;
- complete 754-entry ownership ledger and
  `final`/`derived_digest_only`/`intentionally_ephemeral` counts;
- comparison-only scientific role, absence of copied input atmosphere/state,
  absence of static/subset role leakage, and absence of atmosphere-lane
  reconstruction;
- `ZIP_STORED`, fixed metadata, NPY 2.0, no object dtype, safe member names,
  unique canonical decode/re-encode identity, no trailing bytes, and the
  four-MiB bound;
- rejection of removed, added, or renamed members; dtype, shape, or C-byte
  mutation; ownership mutation; corruption; alternate valid compression,
  metadata, ordering, permissions, NPY encoding, comments, path-like names,
  trailing bytes, empty input, and oversized input.

Historical writer and assembler audits exercised those cases for the former
snapshot. They cannot substitute for candidate-byte acceptance after an
accepted upstream identity changed.

## 6. Source, data, and manifest immutability

A read-only harness captured the relevant local trust files, all 17 pinned
Payne Zero synthesis Python files, the source archive, the three pinned static
tables, the complete textbook `data/` tree, and `data/MANIFEST.json` before
the two top-level attempts and again afterward.

| snapshot | exact result |
| --- | --- |
| bound local source/trust files | `17` files |
| pinned source/data files | `21` files |
| combined source snapshot digest | `7d3034c7c78ac94da52d42c209e138bdcae1f39efdac9cb7a6a02f88c040b385` |
| `data/` directories | `12` |
| `data/` regular files | `39` |
| `data/` regular-file bytes | `30,046,405` |
| complete `data/` snapshot digest | `900d6d2ae07f60f7a9347d0ee0e3cc761054da673dab4365cee02450ee156a62` |
| `data/MANIFEST.json` bytes | `1,087,741` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |
| source before/after | exact |
| `data/` before/after | exact |
| manifest before/after | exact |

The intended synthesis destination remained absent before and after:

```text
data/golden/payne_zero/chapter06/synthesis/
  chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz
```

It also had no manifest entry. No publisher, authorization, candidate archive,
raw archive, temporary repository file, canonical directory, fixture, golden,
or manifest mutation was created.

The external Payne Zero and paper trees were read-only boundaries. The paper
tree was not used as a scientific input.

## 7. Verification executed

Exact reviewed-object hashes:

```text
shasum -a 256 \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py \
  design/chapter06_synthesis_compact_writer_candidate.md \
  design/chapter06_synthesis_compact_writer_independent_audit.md \
  design/chapter06_lane_artifact_publisher_contract.md \
  design/chapter06_lane_artifact_publisher_contract_independent_audit.md

all six matched the assigned identities
```

Two independent zero-argument attempts were run under the environment in
Section 3. Both produced the exact same fail-closed plan-identity error before
scientific capture, with no source or data delta.

Focused writer suite:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_chapter06_synthesis_compact_writer.py

5 failed, 1 passed in 0.41s
```

All five failures descend from the same live-plan identity mismatch. The
passing test is the cache-policy rejection test, which does not need an
accepted scientific build. This result is expected evidence of the stale
acceptance boundary, not a reason to bypass it.

Static checks:

```text
/Users/ysting/anaconda3/bin/python -m ruff check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
All checks passed!

/Users/ysting/anaconda3/bin/python -m ruff format --check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
2 files already formatted

git diff --check -- \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py \
  design/chapter06_synthesis_fixture_oracle_plan.md \
  design/chapter06_lane_artifact_publisher_contract.md
clean
```

## 8. Downstream affected identities and required closure

The following exact accepted objects bind or transitively depend on the former
plan identity and therefore cannot be carried forward without deliberate
review:

| affected boundary | current object SHA-256 | reason |
| --- | --- | --- |
| synthesis scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` | emits live `meta__design_sha256`; full-capture fingerprint changes |
| worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` | freeze the former accepted raw capture |
| worker candidate record | `182291c5d4862ee7462460517ff00e7fe04dec117cdd485b2b2534b245a60d06` | records the former plan identity |
| worker independent audit | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` | accepts the former plan/raw snapshot |
| compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` | requires the former raw full-capture fingerprint |
| assembler tests | `53111433aa3082a58be5ec8b3da1a961f330eac1bb805ac26d1fc6625487e42d` | freeze the former compact result |
| compact candidate record | `41daee0a8b4239fd60a2e67b028afa1a0197ac3ae040a30f5dfd8795234b3550` | records the former plan and compact identities |
| compact independent audit | `a0530cd08d5f0ddcc96b51fdaab4520aa89e62ca65850cf855ad4ede22251a33` | accepts the former compact snapshot |
| deterministic compact writer | `3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb` | directly rejects the live plan |
| writer tests | `9601f8717a29ef51f32a62fd0a73c4d3db4b41c4f6ac08374be6149df4030bfc` | five tests now fail at that gate |
| writer candidate record | `540cf57126df93ee34d02c3da446a6ef109b93a8b17d60514439e12a8f63fc71` | records the former source and archive snapshot |
| writer independent audit | `b888b49226e8ca6407c8226a3c021efb88fa100623fef27dd62e9beba43f2535` | accepts the former writer snapshot |
| two-lane publisher contract | `9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774` | transcribes the former plan and archive locks |
| publisher-contract audit | `7f2517ad0abcca312dcf22785e483fe033519264d5513b16ebe6fa580d4521fd` | accepts that now-stale contract snapshot |

Required closure is forward-only:

1. reproduce the scientific worker against the accepted repaired plan and
   independently freeze its new complete raw identities while proving whether
   the physical fingerprint is unchanged;
2. update and independently re-audit the compact assembler boundary, including
   the full ownership ledger and no-copy role proof;
3. update the zero-argument writer's exact accepted identities, reproduce two
   internally independent A/B builds, and independently re-accept writer
   implementation and candidate facts;
4. revise and independently re-audit the publisher contract against those new
   exact identities;
5. append a new independent repair re-audit to this exact trust-object path,
   rerunning every byte, member, role, ownership, encoding, adversarial, and
   immutability check listed in Section 5.

No step may silently refresh a hash, reuse the old byte acceptance, or infer
scientific equivalence from a prose-only intention.

## 9. Final disposition

**REJECT** synthesis candidate-byte acceptance at the current live source
snapshot.

This is not a rejection of the repaired `Grid.resolution` notation or of the
former candidate's historical evidence. It is a rejection of crossing an
exact provenance boundary with a stale accepted-input hash.

No candidate byte string is accepted by this report. No publisher
implementation, detached authorization, canonical file, manifest role, or
permission to write `data/` follows from it.

<!-- BEGIN REPAIRED SYNTHESIS CANDIDATE-BYTE RE-AUDIT 2026-07-30 -->

# Chapter 6 repaired synthesis candidate-byte re-audit

Status: **REJECT — exact candidate bytes reproduce, but the final rebound
contract contains a contradictory raw fingerprint**

This section is an append-only independent re-audit. It preserves the complete
historical rejection above as an immutable prefix of `15,792` bytes and `349`
lines with SHA-256
`474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1`.
It neither edits nor supersedes that history.

## 1. Decision

**REJECT** the repaired synthesis candidate bytes against the exact final
publisher contract.

The repaired zero-argument writer is deterministic. Two unrelated top-level
builds, each owning two fresh child captures, reproduced the exact same raw
transport, compact mapping, and final archive. The focused suite passed, all
thirty additional hostile mutations failed closed, and the complete bound
source and `data/` snapshots were unchanged.

Acceptance nevertheless cannot cross the exact contract boundary. The final
contract records the raw physical fingerprint as a `65`-hex-character value
with an extra `e`; the accepted synthesis chain, the actual fresh builds, the
writer rebind audit, and the final contract audit all record the valid
`64`-hex-character value. Exact candidate bytes cannot satisfy both values.

This is a contract rejection, not a rejection of the candidate's deterministic
archive identity or scientific reproduction.

## 2. Exact bound inputs

| object | SHA-256 |
| --- | --- |
| final publisher contract | `a663369c3851d89468a41436b8faddeba9d3dcbeba79a7254037734f4a5b3666` |
| final publisher-contract audit | `c4a4ca58d94ec71ec509238046afcb127e189ba0be98a96c9929488958a1c286` |
| synthesis fixture/oracle plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| phase-1 rebind candidate | `dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93` |
| phase-1 independent audit | `9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7` |
| scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| worker independent audit | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |
| compact assembler | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| assembler tests | `25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153` |
| phase-2 rebind candidate | `54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c` |
| phase-2 independent audit | `739854db2b5c4c0c0fe5e9db71d8a52958ce401ded7e7a80a8ab90e15172ddcb` |
| repaired deterministic writer | `57aa7147afee4a7366cb2a075715d3607fa20507c23c07ec978b0698368ae47b` |
| writer tests | `7c41a74f9d2e38a23d988c990af4040ac262a8066cb3cd9feae4e29f0bdc0a4e` |
| phase-3 candidate | `6ab1f346a409b0302550a0923c35b71a84d6b2899f2c356070c8d76aa8145e5a` |
| writer independent audit | `467fdc810f14302dba80f0dd18ba34239dfedb7579b48280899f6f9b6e3b3653` |

Every object was a regular nonsymlink file with the stated identity before and
after the gate.

## 3. Fatal contract contradiction

| source | recorded raw physical fingerprint | length |
| --- | --- | --- |
| final contract, Section 2.1 | `51371e5c0db1fae7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` | `65` |
| final contract audit | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` | `64` |
| writer independent audit | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` | `64` |
| four fresh raw observations | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` | `64` |

The first divergence is at zero-based character index `14`: the contract has
inserted an extra `e` after the prefix `51371e5c0db1fa`. The contract audit
binds the exact contract file hash but silently transcribes the correct
64-character value instead of detecting the contract's 65-character value.
Therefore the contract and its acceptance audit disagree on an exact
scientific identity.

The exact-file lock does not permit this re-audit to interpret the defect as a
harmless typo. Silently normalizing the value would be the same forbidden
hash refresh that this trust chain is designed to prevent.

## 4. Independent build topology

The audit launched two unrelated top-level Python processes concurrently.
Each called `build_deterministic_compact_archive()` with zero arguments. Each
writer invocation created and disposed its own fresh capture A and B children.
The parent compared the two returned archive byte strings directly.

Top-level evidence:

| build | PID | origin-token SHA-256 | outer-cache-root SHA-256 |
| --- | ---: | --- | --- |
| A | `58483` | `4fa6b04fc9a2985c7c3668f25f1cea6c957f425c5fe8f125be54c977fd4a0dcf` | `e90b9551392642af9927dab02b989fc61ec93e5976bc81de2577903f8d85a3d1` |
| B | `58484` | `c05892500f247296b0b72e105daf5890c2ac385389a5f5cc3b868d896d356240` | `f39b8041cc9209bae60d8300f84738dc89a957d9c28238be0f35957824ef2bff` |

Inner capture evidence:

| top | capture | child PID | origin-token SHA-256 | cache-root SHA-256 |
| --- | --- | ---: | --- | --- |
| A | A | `58486` | `fb4c98564afe3051694dbca11109244be1d1924e97f1559ee73c0f813c34f6db` | `29f78e7af4eba1d3109e09b3d8fca19d530f11e538bf971cac1c95cf3d257ed6` |
| A | B | `58495` | `5ee3819e42a4b7cb8ab9d0a0c2d8f38bd9dd8a1f613488b94be152ebeb357d80` | `c873a0a31f5292a15a4012318b1f2d5f8a254d3555c64012336557b9ad467ced` |
| B | A | `58485` | `a4b6f271b84678fc983acc312bb01ff3e1e4292686b46fa9d3e9275d5483f9b8` | `6bb987ab2bbcd1070dcaad974c5aae9394decd85a6b19d8eeb3cd9849cfd67ad` |
| B | B | `58496` | `c87ceee9b0162c85302cd416125a0cd77daa463ff0f12538d65f48990cebc821` | `81c838b8b0ff3467f44d6fe130e1fe2d803be67ceda3434c6947d6d7ac1c7e56` |

The four inner origins, four child PIDs, and four cache-root identities were
pairwise distinct. Every cache was an external nonsymlink directory, empty
before capture, empty after capture, and absent after owned disposal. Both
outer caches were likewise empty before and after their top-level build and
absent after disposal.

## 5. Reproduced raw, compact, and final bytes

| boundary | independently reproduced value |
| --- | --- |
| raw members | `754` |
| raw transport bytes | `8,689,108` |
| raw transport SHA-256 | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` |
| raw mapping digest | `09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c` |
| raw schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| raw physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| raw full fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |
| compact members | `213` |
| compact array bytes | `1,235,275` |
| compact schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| compact payload fingerprint | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| raw ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| ownership counts | `final=250`, `derived_digest_only=123`, `intentionally_ephemeral=381` |
| final archive bytes | `1,294,865` |
| final archive SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |

Within each top-level build, raw A and B were equal by member set, dtype,
shape, C bytes, mapping digest, and complete transport bytes. Compact A and B
were equal by schema, exhaustive ownership, dtype, shape, and C bytes. Their
two serialized archives were exactly equal. The final archives returned by
unrelated top-level builds A and B were also byte-for-byte equal.

## 6. Archive, ownership, and role matrix

The accepted candidate archive had:

- exactly `213` unique lexically ordered `.npy` members;
- NPY version `2.0` for every member;
- `ZIP_STORED` encoding with the exact fixed timestamp, host/version fields,
  permissions, flags, empty member extra/comment fields, and empty archive
  comment;
- no object arrays and C-contiguous storage for every loaded array;
- a deserialize/serialize round trip exactly equal to the original
  `1,294,865` bytes;
- `meta__archive_kind =
  "synthesis_one_line_comparison_candidate"`;
- false publication and golden-publication flags;
- a complete `754`-entry raw ownership boundary with the exact digest and
  disposition counts above;
- only the two permitted dense `(4, 6, 6000)` opacity slabs;
- no duplicated continuum/line-state slab, coarse input slab, numerical
  table, or intentionally ephemeral loop state.

These checks establish the comparison-only candidate lifecycle. They do not
assign the bytes the repository role `golden`; that role remains unavailable
without a consistent contract, a later detached authorization, publication,
manifest registration, and postpublication audit.

## 7. Adversarial matrix

The unchanged focused writer suite rejected the historical compact payload,
truncation, payload corruption, object dtype, nonlexical ordering, changed
ZIP metadata, raw A/B disagreement, compact schema/ownership/payload
disagreement, reused capture objects/origins/PIDs/caches, occupied caches,
symlink caches, and forbidden-root caches.

An additional independent in-memory matrix exercised thirty exact mutations.
Every case rejected:

| family | rejected mutations |
| --- | --- |
| members | add, remove, rename, invent atmosphere schema member |
| dtype/shape/C bytes | dtype change, shape change, payload-bit change, object mapping, object NPY, non-C mapping |
| ownership | ownership digest change, final-count change |
| role | false publication flag, wrong archive kind, copied input-state member |
| corruption/encoding | truncation, CRC corruption, noncanonical NPY, wrong NPY version |
| ZIP policy | alternate compression, nonlexical order, date change, permission change, member comment, archive comment |
| path | unsafe serializer path, unsafe archive member path |
| envelope | trailing bytes, deserializer size-ceiling breach, serializer size-ceiling breach |

Canonical-but-scientifically-mutated archives were rejected by exact member,
dtype, shape, C-byte, and whole-candidate comparison. Noncanonical archives
were rejected by the serializer/deserializer boundary itself. All probes
remained in memory and wrote no candidate file.

## 8. Source, data, manifest, and destination immutability

Each aggregate hashes sorted resolved-path length and bytes, file-byte length,
and binary file SHA-256 with little-endian 64-bit lengths.

| snapshot | exact before/after result |
| --- | --- |
| bound source regular files | `38` |
| bound source bytes | `259,609,761` |
| bound source aggregate SHA-256 | `a65da87dfce85eedd3da9b310fbc0b9eb960102cc7decc523b15f2406618550e` |
| bound source symlinks | `0` |
| `data/` directories | `12` |
| `data/` regular files | `39` |
| `data/` regular-file bytes | `30,046,405` |
| complete `data/` aggregate SHA-256 | `288bbe4c6bcbd20da8390f99fb6cc45e07ee6eed24197371d3520aed39f7d004` |
| `data/` symlink files | `0` |
| `data/MANIFEST.json` bytes | `1,087,741` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |

The source census includes the worker/tests, repaired plan and exact source
contract, both phase-1 records, assembler/tests, both phase-2 records,
writer/tests, five staged executed synthesis modules, all 17 frozen pinned
Python sources, the full atomic source archive, and three pinned upstream
static synthesis tables.

The prospective synthesis destination remained absent. The manifest retained
no Chapter 6 synthesis-golden entry. The contract, contract audit, writer,
tests, phase-3 candidate, phase-3 audit, all phase-1/2 inputs, and the
historical `15,792`-byte prefix retained their exact hashes throughout all
build, forensic, and adversarial checks.

## 9. Verification result

Focused phase-3 suite:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q tests/test_chapter06_synthesis_compact_writer.py

6 passed in 35.66s
```

Independent hostile archive/mapping matrix:

```text
30 rejected; 0 unexpectedly accepted
```

No publisher, authorization record, candidate file, golden, fixture,
manifest, pinned checkout, paper, or external static source was created or
modified.

## 10. Required forward repair

Closure must remain forward-only:

1. repair the final publisher contract so its synthesis raw physical
   fingerprint is exactly the accepted 64-character value;
2. independently re-audit the new exact contract and explicitly verify the
   length and equality of every frozen digest/fingerprint;
3. rerun this synthesis candidate-byte gate against those new exact hashes,
   preserving both rejection sections above;
4. rerun the separate atmosphere candidate-byte gate against the same final
   contract snapshot;
5. begin publisher implementation only after both lane byte gates append
   exact ACCEPT decisions.

## 11. Final disposition

**REJECT** synthesis candidate-byte acceptance against final contract
`a663369c3851d89468a41436b8faddeba9d3dcbeba79a7254037734f4a5b3666`.

No candidate bytes are accepted by this report. The reproducible archive
identity
`a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955`
is evidence for the next repaired gate, not authorization for publication.
No publisher implementation, detached authorization, canonical file,
manifest role, or permission to write `data/` follows.

<!-- END REPAIRED SYNTHESIS CANDIDATE-BYTE RE-AUDIT 2026-07-30 -->

<!-- BEGIN CORRECTED-CONTRACT SYNTHESIS CANDIDATE-BYTE RE-AUDIT 2026-07-30 -->

# Chapter 6 corrected-contract synthesis candidate-byte re-audit

Status: **ACCEPT EXACT CANDIDATE BYTES ONLY — NOT PUBLISHER ACCEPTANCE OR
AUTHORIZATION**

This is an append-only independent rerun against the corrected final
publisher contract. The complete preceding `29,440` bytes and `615` lines
remain an immutable history prefix with SHA-256
`df3714892cf60bb54d22743dcc444157246b8392a3ea907c039af1c196528c55`.
That prefix includes the original `474e318…` rejection and the later
contract-fingerprint rejection; neither was edited or silently converted
into an acceptance.

## C1. Decision

**ACCEPT** the exact `1,294,865`-byte synthesis candidate at SHA-256
`a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955`
against:

- corrected final contract
  `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b`;
  and
- final contract audit
  `fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc`.

The corrected contract now contains the same valid 64-character raw physical
fingerprint reproduced by all four fresh observations. The sole blocker in
the preceding rejection is closed.

This decision accepts only the exact in-memory candidate bytes, schema,
science, ownership, encoding, and comparison-only role boundary. It grants no
publisher implementation, detached authorization, canonical file, manifest
entry, golden role, cleanup record, or permission to write `data/`.

## C2. Exact accepted chain

| accepted object | SHA-256 |
| --- | --- |
| corrected publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| corrected contract independent audit | `fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc` |
| synthesis fixture/oracle plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| phase-1 rebind candidate | `dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93` |
| phase-1 independent audit | `9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7` |
| scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| worker independent audit | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |
| compact assembler | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| assembler tests | `25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153` |
| phase-2 rebind candidate | `54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c` |
| phase-2 independent audit | `739854db2b5c4c0c0fe5e9db71d8a52958ce401ded7e7a80a8ab90e15172ddcb` |
| repaired deterministic writer | `57aa7147afee4a7366cb2a075715d3607fa20507c23c07ec978b0698368ae47b` |
| writer tests | `7c41a74f9d2e38a23d988c990af4040ac262a8066cb3cd9feae4e29f0bdc0a4e` |
| phase-3 rebind candidate | `6ab1f346a409b0302550a0923c35b71a84d6b2899f2c356070c8d76aa8145e5a` |
| writer independent audit | `467fdc810f14302dba80f0dd18ba34239dfedb7579b48280899f6f9b6e3b3653` |

Every object was a regular nonsymlink file with the exact stated identity
before and after construction, forensics, tests, and adversarial probes.

## C3. Independent process topology

Two unrelated top-level Python processes ran concurrently. Each called the
accepted `build_deterministic_compact_archive()` with zero arguments and
internally owned fresh child captures A and B.

Top-level evidence:

| build | PID | origin-token SHA-256 | outer-cache-root SHA-256 |
| --- | ---: | --- | --- |
| top A | `63334` | `1ad4c1dbda0d9d1b738d5182dfdb575d83bc966b54ea121133ee3bed9a968d33` | `c4b84aab9f4da79888994fd4e256606f1c6a61d9158574fd7166cd476b074e4a` |
| top B | `63335` | `2a5208b72f5e9598fabbe2b09db4bdd83b3364960aa145fea13627346f0143ab` | `5d122d0775c36c5e667908a699b109b492063bb1a06820c7c30f99a667272830` |

Inner capture evidence:

| top | capture | PID | origin-token SHA-256 | cache-root SHA-256 |
| --- | --- | ---: | --- | --- |
| A | A | `63337` | `e3abe078ee59b72c1a616c3dd70f4fbbf594aaa953075a0001fd3397208c1132` | `373d4304e4f6a63426f4cb747c86c114106f18298075ebcec34e0b50f2a6211c` |
| A | B | `63359` | `11220236cc32d173522faceec35d0e4f0f7c38f9eea6b8c3fc631a2dce18499b` | `fe9d0aec55009b659649227ce7e9dd1bd291a1d3611206c5ece2ea7adffac76e` |
| B | A | `63336` | `2ac3aba93cad7d71b5e3a96dd64e7ffa1eb310114943d9c78a3e4e6ac7c6e16b` | `ab0a1d93d0a8299016abb9d02c0b5882fbeb759a69e462827f03489dbe9f50d5` |
| B | B | `63358` | `0e0fe7dc8c7d17eb41b020201a17a0a1041188170783159a439ae54d7758eb90` | `4ce986a54623c2a300ccbe4db108c32c9451ae32915bb8703948ae8a582334e4` |

The four child PIDs, four origin identities, and four cache identities were
pairwise distinct. Each cache was an external nonsymlink directory, empty
before capture, empty after capture, and absent after owned disposal. The two
outer caches satisfied the same empty-before, empty-after, and disposal
checks.

## C4. Raw and compact equality

Each of the four fresh raw observations reproduced:

| raw property | exact result |
| --- | --- |
| members | `754` |
| transport bytes | `8,689,108` |
| transport SHA-256 | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` |
| mapping digest | `09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| physical-fingerprint length | `64` |
| full fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |

Within each top build, raw A and B were equal by lexical member set, dtype,
shape, C bytes, complete mapping digest, and the entire serialized transport.
The corrected contract contains the exact 64-character physical fingerprint
above.

Each independently assembled compact mapping reproduced:

| compact property | exact result |
| --- | --- |
| members | `213` |
| array bytes | `1,235,275` |
| schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| payload fingerprint | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| raw-ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| ownership `final` | `250` |
| ownership `derived_digest_only` | `123` |
| ownership `intentionally_ephemeral` | `381` |

Compact A and B were equal by member set, schema descriptions, exhaustive raw
ownership, dtype, shape, and C bytes. The ownership ledger covered all 754
raw members exactly once.

## C5. Role and no-copy boundary

The archive records
`meta__archive_kind = "synthesis_one_line_comparison_candidate"`,
`meta__publication_authorized = false`,
`meta__golden_publication_performed = false`, and complete raw ownership.

Independent inspection reconfirmed:

- no ephemeral continuum-state, line-state, loop-output, or stimulated-factor
  member entered the compact archive;
- no copied `(6, 6, 139)`, `(6, 400)`, or `(4, 6, 400)` input/coarse slab
  entered it;
- no `1001`- or `2001`-sample numerical table entered it;
- the only dense `(4, 6, 6000)` arrays are
  `opacity__gross_float32` and `opacity__net_float32`; and
- no atmosphere-lane member, fixture role, static role, or subset role was
  assigned to these bytes.

The bytes are comparison-only candidate evidence. This acceptance does not
assign repository role `golden`; only a later independently accepted
publisher, detached authorization, no-replace installation, and manifest
entry can do that.

## C6. Canonical final archive

Both A/B archives inside each build and the returned archives from unrelated
top builds A and B were byte-for-byte equal:

| archive property | exact result |
| --- | --- |
| bytes | `1,294,865` |
| SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| members | `213` |
| member order | unique lexical names |
| NPY version | `2.0` for every member |
| compression | `ZIP_STORED` |
| archive comment | empty |

Every ZIP timestamp, create/extract version, host system, flag, volume,
permission, extra field, member comment, and stored-size field matched the
accepted fixed metadata. Every loaded array was non-object and C-contiguous.
Deserializing and canonically reserializing all 213 arrays reproduced the
exact original `1,294,865` bytes.

## C7. Adversarial verification

The unchanged focused writer suite passed all six tests. It reconfirmed exact
identity/no-publication surfaces, two fresh top-level repetitions, archive
metadata, A/B disagreement rejection, capture-origin/cache policy, and
fail-closed corruption handling.

The independent expanded matrix rejected all 30 mutations:

| family | rejected probes |
| --- | --- |
| member set | add, remove, rename, invented atmosphere schema member |
| dtype/shape/C bytes | dtype, shape, payload byte, object mapping, object NPY, non-C mapping |
| ownership | ownership digest, final ownership count |
| role | false publication, wrong archive kind, copied input-state member |
| corruption/NPY | truncation, CRC corruption, noncanonical NPY, wrong NPY version |
| ZIP | alternate compression, order, timestamp, permissions, member comment, archive comment |
| path | unsafe mapping path, unsafe archive path |
| envelope | trailing bytes, serializer size ceiling, deserializer size ceiling |

Canonical-but-mutated archives failed exact member/dtype/shape/C-byte and
whole-candidate comparison. Noncanonical archives failed at the untrusted
ZIP/NPY boundary. No hostile probe wrote a file.

Verification output:

```text
python -m pytest -q tests/test_chapter06_synthesis_compact_writer.py
6 passed in 17.41s

expanded in-memory hostile matrix
30 rejected; 0 unexpectedly accepted
```

## C8. Source, data, contract, and manifest immutability

Each aggregate hashes sorted resolved-path length and bytes, file-byte length,
and binary file SHA-256 using little-endian 64-bit lengths.

| snapshot | exact before/after result |
| --- | --- |
| bound source regular files | `38` |
| bound source bytes | `259,609,761` |
| bound source aggregate SHA-256 | `a65da87dfce85eedd3da9b310fbc0b9eb960102cc7decc523b15f2406618550e` |
| bound source symlinks | `0` |
| `data/` directories | `12` |
| `data/` regular files | `39` |
| `data/` regular-file bytes | `30,046,405` |
| complete `data/` aggregate SHA-256 | `288bbe4c6bcbd20da8390f99fb6cc45e07ee6eed24197371d3520aed39f7d004` |
| `data/` symlinks/special files | `0` |
| `data/MANIFEST.json` bytes | `1,087,741` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |

The source census includes the worker/tests, repaired plan and exact source
contract, phase-1 records, assembler/tests, phase-2 records, writer/tests,
five staged executed synthesis modules, all 17 frozen pinned Python sources,
the full atomic source archive, and three pinned upstream static synthesis
tables.

The corrected contract, its final audit, all phase-1/2/3 files, writer/tests,
manifest, data files, Payne Zero sources, paper files, and external tables
remained unchanged. The prospective synthesis destination remained absent.

## C9. Residual unresolved boundary

This report closes only the synthesis candidate-byte gate. Future synthesis
objects remain unresolved and absent:

```text
scripts/build_chapter06_synthesis_golden.py
tests/test_chapter06_synthesis_golden_publisher.py
design/chapter06_synthesis_golden_publisher_independent_audit.md
design/chapter06_synthesis_publication_acceptance.json
design/chapter06_synthesis_publication_record_review.json
design/chapter06_synthesis_postpublication_audit.md
```

The accepted byte record contains no future publisher, publisher-audit,
authorization, review, realized-entry, postmanifest, or postpublication hash.
Those hashes must be created only by their later forward gates. The
atmosphere candidate-byte gate must also accept the same corrected contract
before either publisher implementation may begin.

Any later edit to the corrected contract, contract audit, writer chain, or
this candidate-byte record invalidates every downstream object that binds it.

## C10. Final disposition

**ACCEPT exact synthesis candidate bytes
`a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955`
against corrected contract
`3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b`
and final contract audit
`fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc`.**

This is candidate-byte acceptance only. It is not publisher acceptance,
authorization, canonical installation, manifest registration, or permission
to write `data/`.

<!-- END CORRECTED-CONTRACT SYNTHESIS CANDIDATE-BYTE RE-AUDIT 2026-07-30 -->
