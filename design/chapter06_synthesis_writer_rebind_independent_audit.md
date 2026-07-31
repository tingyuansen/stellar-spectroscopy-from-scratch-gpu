# Chapter 6 synthesis deterministic-writer rebind — independent audit

Date: 2026-07-30  
Scope: candidate-byte audit only

## 1. Disposition

**ACCEPT for the candidate-byte-audit phase only.**

The exact phase-3 writer candidate is accepted as a deterministic,
zero-argument, in-memory construction of the Chapter 6 synthesis candidate
bytes. Independent execution reproduced the accepted raw observation,
compact mapping, canonical archive, historical control, and exact
historical-to-live delta.

This decision does **not** accept or authorize:

- a detached candidate file;
- a canonical golden;
- any writer or publisher contract change;
- any `data/MANIFEST.json` change;
- any destination creation, overwrite, rename, or publication; or
- any write to the Payne Zero or paper trees.

No such action was performed by this audit.

## 2. Exact reviewed inputs

The three phase-3 candidate objects were regular nonsymlink files with these
exact identities:

| object | SHA-256 |
| --- | --- |
| `scripts/chapter06_synthesis_compact_writer.py` | `57aa7147afee4a7366cb2a075715d3607fa20507c23c07ec978b0698368ae47b` |
| `tests/test_chapter06_synthesis_compact_writer.py` | `7c41a74f9d2e38a23d988c990af4040ac262a8066cb3cd9feae4e29f0bdc0a4e` |
| `design/chapter06_synthesis_writer_rebind_candidate.md` | `6ab1f346a409b0302550a0923c35b71a84d6b2899f2c356070c8d76aa8145e5a` |

The writer's complete ten-file acceptance gate resolved to the canonical
repository paths and reproduced every expected identity:

| accepted dependency | SHA-256 |
| --- | --- |
| repaired fixture-oracle plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| synthesis oracle worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| synthesis oracle-worker tests | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| synthesis worker independent audit | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |
| phase-1 plan-rebind candidate | `dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93` |
| phase-1 plan-rebind independent audit | `9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7` |
| phase-2 compact assembler | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| phase-2 compact-assembler tests | `25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153` |
| phase-2 compact-rebind candidate | `54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c` |
| phase-2 compact-rebind independent audit | `739854db2b5c4c0c0fe5e9db71d8a52958ce401ded7e7a80a8ab90e15172ddcb` |

The imported assembler and worker paths were also the same canonical files
named by the gate. Replacing a gate value with a historical plan or
assembler identity, or changing either accepted independent-audit identity,
failed closed.

## 3. Public boundary and construction topology

Inspection and execution confirmed that the production entry point remains:

```python
build_deterministic_compact_archive()
```

Its signature has no parameters. Supplying raw mappings raises `TypeError`.
The module exposes no command-line entry point, output path, filesystem
writer, manifest mutation, authorization parser, publisher, overwrite, or
publication operation.

I ran an independent supervisor, separate from the focused test helper. It
started two unrelated controlled top-level Python processes. Each top-level
process invoked the zero-argument builder, which started two fresh
raw-capture children.

| build/capture | PID | origin-token SHA-256 | cache-root SHA-256 |
| --- | ---: | --- | --- |
| top A | `51560` | `3e5610ad0c922796753841ea59d42cddf4671202362a3f2397d14bcf1430fe41` | `5bef8308092ffcddca0e26721e145cc575dc8c6f6ce776ebbd7a0be7279dfc45` |
| A / child A | `51561` | `656dbbcd8f7d03eebc64d53e9391c1b59cb9b14184d0815d1684a3ae90493f0d` | `197f9466fd2639bf7ccfef58ffa93a750b35aafe99baddcd86f65b7634e5bf17` |
| A / child B | `51567` | `df53b1147aade39a7bb51e54ceab28b9d36fc94af2f4c8676597b18b3b41cdd6` | `d02ca941a6da477cbdf46d215c9bf5f5435105d245c5102a6456cbb12158fc68` |
| top B | `51575` | `668f3e862efe231be23e57d8359618b2c49f6c12ed0b2b6651a1e1678c1a8770` | `8bf5342d47e1817b3aa0cffaf7e328d4dc84b3d9ff7559bf0b2d58bc4b4dd19d` |
| B / child A | `51576` | `04d29e8d7eb013be9cc56016e043c3e38221fc8f7aefd1935a1b27ed7cf1ac20` | `a252a992ae3c7ae43562ed00cdafac3452dead72f00d9c0f5df42239f399e6b1` |
| B / child B | `51586` | `0ef395a568b5af1622ba4c74136be0d73082982eb31e4673f7dbcf09df18c106` | `f95aa105e2fb40107956911b8f3cf26352343e8c3c69181d787f29c503ae41d0` |

Thus the two top-level PIDs and all four child PIDs were distinct; the two
top-level tokens and all four child tokens were distinct; and the two outer
and all four inner cache-root identities were distinct.

Every cache was:

- outside the repository, pinned Payne Zero checkout, and paper tree;
- an existing nonsymlink directory before use;
- empty before and after its associated build or capture; and
- disposed after the process boundary completed.

The two inner process-evidence mappings independently rehashed to their
reported identities:

| top-level build | process-evidence SHA-256 |
| --- | --- |
| A | `de672f2195fdc907a1e544d841cc6232f35255fa800d2af083839230fc518077` |
| B | `dd1a6273dc7654c60dc6cf18696c55553c805f51dc7c0f4c9a36926b03ff15a6` |

Process evidence is intentionally fresh. After excluding only that
nondeterministic evidence mapping and its digest, the complete JSON-safe
summaries from top A and top B were equal. The independently computed archive
forensics were equal without exclusions.

## 4. Raw and compact evidence

Each of the four fresh child observations reproduced the same accepted raw
result:

| raw property | independent result |
| --- | --- |
| members | `754` |
| transport bytes | `8,689,108` |
| transport SHA-256 | `e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e` |
| complete mapping digest | `09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |

Within each top-level build, raw A and B were equal by member set, dtype,
shape, contiguous bytes, complete mapping digest, and entire serialized
transport bytes.

The two independently assembled compact mappings in each build reproduced:

| compact property | independent result |
| --- | --- |
| members | `213` |
| array bytes | `1,235,275` |
| schema version | `1` |
| schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| payload fingerprint | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| raw-ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |

The complete 754-member ownership partition remained:

| disposition | count |
| --- | ---: |
| `final` | `250` |
| `derived_digest_only` | `123` |
| `intentionally_ephemeral` | `381` |

Compact A and B were equal by schema descriptions, complete ownership tuple,
member order, dtype, shape, and contiguous payload bytes.

## 5. Canonical archive

Both top-level builds, and both A/B serializations inside each build,
reproduced the same immutable byte string:

| archive property | independent result |
| --- | --- |
| bytes | `1,294,865` |
| SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| members | `213` |
| decoded array bytes | `1,235,275` |

Independent ZIP/NPY inspection confirmed:

- lexical, unique, safe `.npy` member names;
- `ZIP_STORED` with stored and uncompressed sizes equal;
- fixed date/time `1980-01-01 00:00:00`;
- Unix creator system `3`, create/extract versions `20`/`20`;
- zero flag bits, volume, and internal attributes;
- external attributes `0o100600 << 16`;
- empty member extras, member comments, and archive comment;
- NPY version `2.0` for every member;
- no object dtype and C-contiguous decoded storage;
- no Zip64 record; and
- byte-identical decode/re-encode of the whole archive.

The decoder used `allow_pickle=False`. It validated each NPY representation,
the full compact assembly, and the unique canonical archive encoding.

## 6. Reconstructed historical control and exact delta

I decoded a detached live mapping, replaced only these four scalar `<U64`
provenance values with their historical values, and passed that mapping
through the candidate's canonical serializer:

| member | historical value |
| --- | --- |
| `meta__assembler_sha256` | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` |
| `meta__compact_payload_fingerprint` | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` |
| `meta__design_sha256` | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| `meta__full_capture_fingerprint` | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` |

The result was exactly `1,294,865` bytes with SHA-256
`b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850`.
It retained the same 213 names, order, member lengths, and local-header
offsets as the live archive.

Exactly four NPY payloads differed:

| member | positions changed | historical CRC-32 | live CRC-32 |
| --- | ---: | --- | --- |
| `meta__assembler_sha256.npy` | `60` | `357230db` | `43f0c0bb` |
| `meta__compact_payload_fingerprint.npy` | `56` | `ef9303ec` | `f069b241` |
| `meta__design_sha256.npy` | `59` | `3382fe33` | `6c8d87d6` |
| `meta__full_capture_fingerprint.npy` | `60` | `5eed617b` | `e9a7bd0a` |

Each changed NPY payload was 384 bytes. Its exact identities were:

| member | historical NPY SHA-256 | live NPY SHA-256 |
| --- | --- | --- |
| `meta__assembler_sha256.npy` | `4af41cd5b478100bd2313f0d71ccee6cc57ac93dfae6b845ba96259f1a0cc9f2` | `e800911ee414139a9b535b7aaf7a38d4ee51935a92f0464b5af1d9bc4069e49c` |
| `meta__compact_payload_fingerprint.npy` | `80fa5d912f433250b44ceb198c466fe2415735a44b52944ce4ce88d707a0541e` | `9f903b89ce83a47fc7b8896626eb91f359cb4a3713dd73965e73b06b1ae68f3f` |
| `meta__design_sha256.npy` | `1dc120ad1be432a56efb84deb9de95c928490b2d459b7eba0bc730d2033dd804` | `3fdd6d84f49c499a872c0f9ef998faf6178ce5f8c1da2833f0ef3c1b27133775` |
| `meta__full_capture_fingerprint.npy` | `13b7fc0a745a2971e127203a50f21cafc6148bd7dfc52ce2c5445f6ce2b9345f` | `e9af37e046329e0132353587b7f1d5553d105500958bc7b2cc1f7f669448ebeb` |

All other `209` member payloads were byte-identical. Across the two complete
archives, exactly `267` byte positions differed. The four NPY value changes
and their ZIP CRC fields fully explain the complete delta; no scientific
array, member name, size, order, compression choice, offset, or other
metadata changed.

## 7. Fail-closed probes

Independent adversarial probes rejected all of the following:

- object-dtype serialization;
- non-C-contiguous serialization;
- unsafe traversal-like member names;
- a truncated archive;
- a payload bit flip;
- nonlexical member order;
- changed ZIP date metadata;
- raw A/B payload disagreement;
- compact A/B schema disagreement;
- compact A/B ownership disagreement; and
- compact A/B payload disagreement.

The focused suite additionally reconfirmed rejection of the historical
compact payload identity, reused capture objects, copied observations with a
reused origin, reused child processes, shared cache roots, occupied caches,
symlink caches, and caches under forbidden source/data roots.

## 8. Source, data, and manifest immutability

Before and after all execution, mutation, and static checks, I independently
recomputed the same aggregate snapshots. Each aggregate hashes sorted
resolved-path length and bytes, file-byte length, and binary file SHA-256,
using little-endian 64-bit lengths.

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

The 38-file source set covers the worker/tests, repaired plan and exact source
contract, both phase-1 records, assembler/tests, both phase-2 records,
writer/tests, five staged executed synthesis modules, 17 frozen pinned
Python sources, the full atomic source archive, and three pinned upstream
static synthesis tables.

The prospective destination
`data/golden/payne_zero/chapter06/synthesis` remained absent. The manifest
contained neither a synthesis-golden entry nor a Chapter 6 synthesis
destination reference.

The three reviewed candidate files also retained the exact hashes recorded
in Section 2 after verification.

## 9. Verification results

Focused phase-3 suite:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q tests/test_chapter06_synthesis_compact_writer.py

6 passed in 20.24s
```

Combined accepted worker/assembler/writer chain:

```text
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_writer.py

20 passed in 72.68s
```

Static verification:

```text
python -m ruff check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
All checks passed!

python -m ruff format --check \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py
2 files already formatted

python -c "py_compile.compile(...)"  # both files; disposable cfile paths
compiled 2 files in disposable tempdir

git diff --check -- \
  scripts/chapter06_synthesis_compact_writer.py \
  tests/test_chapter06_synthesis_compact_writer.py \
  design/chapter06_synthesis_writer_rebind_candidate.md
clean
```

A separate trailing-whitespace scan of those three candidate inputs was also
clean. The interpreter was `/Users/ysting/anaconda3/bin/python`, Python
`3.13.9`.

## 10. Gate boundary

The deterministic writer rebind has passed its independent candidate-byte
audit. The accepted conclusion is limited to the in-memory candidate bytes
and the exact evidence recorded above.

Any detached byte acceptance, output-path introduction, publisher change,
golden creation, manifest mutation, or publication remains a separate,
unauthorized downstream phase.
