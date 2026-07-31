# Chapter 6 synthesis deterministic-writer rebind candidate

Status: **CANDIDATE ONLY — independent writer-rebind audit required**

Scope: phase 3 of the forward repair cascade after the independently accepted
scientific-worker plan rebind and compact-assembler rebind

Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

Authorized implementation surface:

```text
scripts/chapter06_synthesis_compact_writer.py
tests/test_chapter06_synthesis_compact_writer.py
design/chapter06_synthesis_writer_rebind_candidate.md
```

No worker, plan, assembler, prior candidate/audit, publisher contract,
candidate-byte acceptance record, data file, manifest, fixture, golden, or
external Payne Zero source was edited. No publisher or data repair was
started.

## 1. Candidate disposition

The deterministic writer now accepts only the phase-2 compact boundary
independently authorized by:

| authority | SHA-256 |
| --- | --- |
| compact-rebind candidate | `54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c` |
| compact-rebind independent audit | `739854db2b5c4c0c0fe5e9db71d8a52958ce401ded7e7a80a8ab90e15172ddcb` |
| repaired compact assembler | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| repaired compact-assembler tests | `25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153` |

The upstream repaired plan and its independent authority are bound at the
same fail-closed boundary:

| upstream authority | SHA-256 |
| --- | --- |
| repaired synthesis fixture/oracle plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| plan-rebind candidate | `dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93` |
| plan-rebind independent audit | `9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7` |

The unchanged worker boundary remains exact:

| unchanged worker object | SHA-256 |
| --- | --- |
| scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| worker independent audit | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |

All ten paths must be regular nonsymlink files at those exact identities.
The imported worker and assembler must resolve to their bound repository
paths.

The resulting in-memory canonical archive candidate is:

| archive property | exact candidate value |
| --- | --- |
| members | `213` |
| bytes | `1,294,865` |
| SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| member order | lexical |
| NPY version | `2.0` |
| compression | `ZIP_STORED` |
| fixed ZIP date/time | `1980-01-01 00:00:00` |
| publication authorized | `False` |
| golden publication performed | `False` |
| manifest mutation performed | `False` |
| artifact-file write performed | `False` |

This report presents those bytes for independent review. It does not accept
or authorize them.

## 2. Minimal implementation repair

The writer's serializer, decoder, archive metadata, origin gate, fresh-child
capture route, A/B comparisons, public API, and no-publication boundary are
unchanged.

The repair is limited to:

1. replacing historical plan/assembler/compact-review identities with the
   repaired phase-1 and phase-2 identities;
2. adding both new candidate/audit pairs to the exact identity gate; and
3. replacing the historical compact payload expectation
   `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9`
   with the independently accepted live value
   `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b`.

Exact candidate implementation identities:

| object | SHA-256 |
| --- | --- |
| repaired deterministic compact writer | `57aa7147afee4a7366cb2a075715d3607fa20507c23c07ec978b0698368ae47b` |
| repaired focused writer tests | `7c41a74f9d2e38a23d988c990af4040ac262a8066cb3cd9feae4e29f0bdc0a4e` |

The public production signature remains exactly:

```python
build_deterministic_compact_archive()
```

It has no parameters. Supplying raw mappings raises `TypeError`; there is no
accepted supplied-observation route.

## 3. Fresh-process construction topology

Two unrelated top-level controlled Python processes each called the
zero-argument builder. Each top-level build created its own two fresh
raw-capture child processes. The four inner captures had four distinct
one-use origin tokens, process IDs, and external cache roots.

All processes used:

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

Each inner capture received a parent-created, existing, empty, external,
nonsymlink `NUMBA_CACHE_DIR`. Every cache remained empty, was distinct from
all other caches, and was disposed before its builder returned.

Top-level evidence:

| evidence | build A | build B |
| --- | --- | --- |
| process PID | `47184` | `47200` |
| origin-token SHA-256 | `1bb9ee23ff5914aba7e46e8e645ad0cdbc6e65ec3fcc82c506fe852d7c1acebb` | `a0587e39e60a8b76fd9c39893445f0632e9280c71333d55053bfce3c8bd67939` |
| outer cache-root SHA-256 | `99ee27ac5b59d13a81682ea90e20067e200cdfb881f0402d1152773ae5ef0cab` | `ffd7166a39267731c741d647a2b00beec6ddec8f03088142c2d2df46c89711b0` |

Inner child evidence:

| build/capture | child PID | origin-token SHA-256 | cache-root SHA-256 |
| --- | ---: | --- | --- |
| A/A | `47185` | `34a9137649729b34760f6bfd220a75254fda8183988f9b316515c0403a91a7ac` | `5b087fc8f955bf46c3122a908bb0cd4b4ee7bdb5f9957fd400d488e5cecc236f` |
| A/B | `47196` | `8dd0f13cd11f9faa74cd252bcbe221930ea1cbf4da0a57842ec104d119fb1023` | `6a2cdfe746d84c82ffc5f9b2df1928a5e28d2c7c2285d1368da4414eb07ea050` |
| B/A | `47201` | `eca37e37ebafeb16126c010adce0f4de68e6b0597c87a5fd0339f2d5391b66ac` | `2c7d81b089e9a6cfb37356f3e2d21f12f204e5d90eb4dab396ad0d4eca4b9d39` |
| B/B | `47212` | `1103dec57bb88e84a3347c08f71949bc2091e2e425afdfff21a955926207b5cc` | `7d66d9b66c998952c4ccab287ba136bce387d740202c810deab5146f83afdd22` |

Build A's complete process-evidence mapping digest was
`138cdebc3fa01744068e11ced803a6e2bc53a9447519a9a5b5876038379b81f1`.
Process evidence is deliberately excluded from the scientific archive, so it
changes between top-level builds while the raw and final bytes remain exact.

Both top-level builds passed every topology decision:

- distinct inner origins, PIDs, and caches;
- external, nonsymlink, empty-before, empty-after, disposed caches;
- raw A/B transport and mapping equality;
- compact A/B schema, payload, and ownership equality; and
- final A/B whole-archive equality.

## 4. Raw and compact boundaries

Each of the four inner captures reproduced:

| raw property | exact live value |
| --- | --- |
| members | `754` |
| transport bytes | `8,689,108` |
| transport SHA-256 | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` |
| complete mapping digest | `09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical fingerprint | `51371e5c0db1fae7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |

Each independently assembled compact mapping reproduced:

| compact property | exact live value |
| --- | --- |
| members | `213` |
| array bytes | `1,235,275` |
| schema version | `1` |
| schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| payload fingerprint | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| raw-ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |

The complete ownership result is unchanged:

| disposition | count |
| --- | ---: |
| `final` | `250` |
| `derived_digest_only` | `123` |
| `intentionally_ephemeral` | `381` |
| total | `754` |

The ownership tuple remains lexical and unique, and every final target
resolves.

## 5. Canonical archive and serialized-form checks

Both unrelated top-level builds and both independently serialized A/B
assemblies reproduced:

```text
bytes
  1,294,865
SHA-256
  a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955
```

All 213 entries retain the canonical historical format:

| ZIP/NPY property | exact value |
| --- | --- |
| member names | lexical, unique, safe, `.npy` suffixed |
| compression | `ZIP_STORED` |
| ZIP date/time | `1980-01-01 00:00:00` |
| creator system | Unix code `3` |
| create/extract version | `20` / `20` |
| flag bits | `0` |
| internal attributes | `0` |
| external attributes | regular file mode `0600`, `0o100600 << 16` |
| member extra/comment | empty |
| archive comment | empty |
| NPY version | `2.0` |
| object dtype | forbidden and absent |
| decoded storage | C-contiguous |
| Zip64 | disabled |

The reader decoded every member with `allow_pickle=False`, reconstructed and
validated the full compact assembly, compared every dtype, shape, and C byte,
and re-encoded to the identical entire input byte string.

## 6. Independently reconstructed historical control

The historical archive was reconstructed in memory from a detached decoded
copy of the live archive by restoring exactly four scalar `<U64` values:

| member | historical value | live value |
| --- | --- | --- |
| `meta__assembler_sha256` | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| `meta__compact_payload_fingerprint` | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| `meta__design_sha256` | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| `meta__full_capture_fingerprint` | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |

Canonical serialization independently reproduced the previously accepted
historical archive:

```text
bytes
  1,294,865
SHA-256
  b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850
```

This controls the comparison against the actual old bytes rather than
inferring a delta from the intended provenance change.

## 7. Exact historical-to-live archive delta

The historical and live archives have the same total length, lexical
213-member set, member names, ZIP offsets, NPY lengths, dtypes, and shapes.
Exactly four member payloads differ. The remaining 209 NPY payloads are
byte-identical, including every scientific array and unchanged provenance
member.

| changed member | NPY bytes | changed NPY positions | historical CRC-32 | live CRC-32 |
| --- | ---: | ---: | --- | --- |
| `meta__assembler_sha256.npy` | `384` | `60` | `357230db` | `43f0c0bb` |
| `meta__compact_payload_fingerprint.npy` | `384` | `56` | `ef9303ec` | `f069b241` |
| `meta__design_sha256.npy` | `384` | `59` | `3382fe33` | `6c8d87d6` |
| `meta__full_capture_fingerprint.npy` | `384` | `60` | `5eed617b` | `e9a7bd0a` |

Exact changed NPY identities:

| member | historical NPY SHA-256 | live NPY SHA-256 |
| --- | --- | --- |
| `meta__assembler_sha256.npy` | `4af41cd5b478100bd2313f0d71ccee6cc57ac93dfae6b845ba96259f1a0cc9f2` | `e800911ee414139a9b535b7aaf7a38d4ee51935a92f0464b5af1d9bc4069e49c` |
| `meta__compact_payload_fingerprint.npy` | `80fa5d912f433250b44ceb198c466fe2415735a44b52944ce4ce88d707a0541e` | `9f903b89ce83a47fc7b8896626eb91f359cb4a3713dd73965e73b06b1ae68f3f` |
| `meta__design_sha256.npy` | `1dc120ad1be432a56efb84deb9de95c928490b2d459b7eba0bc730d2033dd804` | `3fdd6d84f49c499a872c0f9ef998faf6178ce5f8c1da2833f0ef3c1b27133775` |
| `meta__full_capture_fingerprint.npy` | `13b7fc0a745a2971e127203a50f21cafc6148bd7dfc52ce2c5445f6ce2b9345f` | `e9af37e046329e0132353587b7f1d5553d105500958bc7b2cc1f7f669448ebeb` |

Across the complete 1,294,865-byte archives, exactly `267` byte positions
differ. Those positions are fully explained by the four changed NPY scalar
payloads and their corresponding ZIP CRC fields. No member was added,
removed, resized, reordered, recompressed, or assigned different ZIP
metadata.

## 8. Fail-closed and adversarial coverage

The focused suite retains the prior topology, serialization, pair-equality,
decode, canonicality, and no-publication tests. This rebind adds explicit
rejection of:

- the historical plan hash;
- the historical assembler hash;
- drift in the phase-1 independent audit;
- drift in the phase-2 independent audit; and
- a real live repaired assembly checked against the historical compact
  payload fingerprint.

The suite also requires:

- ten exact accepted file identities;
- the zero-argument public builder and rejection of supplied raw arguments;
- two fresh capture children with distinct origins, PIDs, and cache roots;
- four distinct inner origins and caches across two unrelated top builds;
- exact raw A/B transport and mapping agreement;
- exact compact A/B schema, ownership, and payload agreement;
- canonical lexical 213-member NPY 2.0/ZIP archive bytes;
- rejection of object dtype, truncation, corruption, nonlexical ordering,
  changed ZIP metadata, and A/B raw/schema/ownership/payload disagreement;
- absence of output, golden, manifest, and publication surfaces; and
- exact `data/` tree equality before and after construction.

## 9. Source, data, and no-publication boundary

The bound source snapshot contains:

- worker, worker tests, repaired plan, and exact source contract;
- phase-1 candidate and independent audit;
- repaired compact assembler and tests;
- phase-2 compact-rebind candidate and independent audit;
- repaired deterministic writer and tests;
- five staged executed synthesis modules;
- all 17 frozen pinned Python sources;
- the full atomic source archive; and
- the three pinned upstream static synthesis tables.

The baseline after the authorized implementation edit is:

| snapshot | exact before/after value |
| --- | --- |
| bound source regular files | `38` |
| bound source bytes | `259,609,761` |
| bound source digest | `a65da87dfce85eedd3da9b310fbc0b9eb960102cc7decc523b15f2406618550e` |
| bound source symlinks | `0` |
| `data/` directories | `12` |
| `data/` regular files | `39` |
| `data/` regular-file bytes | `30,046,405` |
| complete `data/` digest | `288bbe4c6bcbd20da8390f99fb6cc45e07ee6eed24197371d3520aed39f7d004` |
| `data/MANIFEST.json` bytes | `1,087,741` |
| `data/MANIFEST.json` SHA-256 | `d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a` |

Each aggregate digest binds sorted resolved path length/path bytes, file-byte
length, and binary SHA-256 using little-endian 64-bit lengths.

The prospective synthesis destination remains absent and the manifest has no
synthesis-golden entry. No raw archive, compact archive, fixture, golden,
authorization, destination directory, or manifest mutation was created.

## 10. Verification

Focused repaired writer suite:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q tests/test_chapter06_synthesis_compact_writer.py

6 passed in 27.66s
```

Combined unchanged-worker, repaired-assembler, and repaired-writer suites:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_writer.py

20 passed in 136.02s (0:02:16)
```

Static checks:

```text
python -m ruff check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
All checks passed!

python -m ruff format --check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
2 files already formatted

git diff --check -- \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py \
  design/chapter06_synthesis_writer_rebind_candidate.md
clean
```

Unchanged prohibited downstream objects:

| object | SHA-256 |
| --- | --- |
| historical writer candidate | `540cf57126df93ee34d02c3da446a6ef109b93a8b17d60514439e12a8f63fc71` |
| historical writer independent audit | `b888b49226e8ca6407c8226a3c021efb88fa100623fef27dd62e9beba43f2535` |
| lane-artifact publisher contract | `9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774` |
| candidate-byte acceptance rejection | `474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1` |

## 11. Remaining gate

This candidate does not authorize a writer acceptance, detached byte
acceptance, publisher change, candidate file, canonical golden,
`data/MANIFEST.json` change, or publication.

The next permitted action is an independent read-only writer-rebind audit of
the exact writer, tests, report, ten-file identity gate, two unrelated
top-level builds, four inner fresh captures, accepted raw/compact boundaries,
canonical archive, reconstructed historical control, four-member/267-byte
delta, source/data immutability, and no-publication evidence.

Final disposition: **CANDIDATE ONLY**.
