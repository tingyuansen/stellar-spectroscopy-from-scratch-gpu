# Chapter 6 atmosphere-fixture writer independent audit

Date: 2026-07-30  
Reviewer role: independent adversarial review; no implementation repair  
Disposition: **ACCEPT the exact deterministic in-memory writer for the next
separately reviewed gate; publication remains unauthorized**

## 1. Review boundary

This audit reviews only the deterministic serialization gate after the
independently accepted Chapter 6 atmosphere scientific worker. It asks whether
the writer:

1. owns two genuinely fresh scientific observations;
2. validates the complete accepted nineteen-array fixture and all 89 ephemeral
   evidence arrays from each observation;
3. produces one unique, bounded, canonical nineteen-member NPZ byte string;
4. leaves source and canonical data unchanged; and
5. has no artifact-publication or authorization authority.

It does **not** review or authorize a canonical fixture path, publisher,
manifest mutation, golden, atmosphere oracle, seam suite, or notebook result.
No such artifact was created during this review.

## 2. Exact reviewed snapshot

All seven assigned or inherited gate files were regular nonsymlink files at
review time. Their SHA-256 identities matched exactly:

| reviewed object | SHA-256 |
| --- | --- |
| atmosphere deterministic writer | `0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538` |
| writer focused tests | `c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a` |
| writer candidate record | `e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee` |
| accepted scientific worker | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| accepted scientific-worker tests | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| accepted worker candidate record | `da53e4846ee91f814c437994fff604ae91ed94a4da4db7856f8f47bb61cf72dc` |
| accepted worker independent audit | `336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e` |

`verify_accepted_identities()` independently returned the exact four accepted
worker-chain hashes. The imported worker resolved to:

```text
/Users/ysting/stellar-spectroscopy-from-scratch-gpu/
  scripts/chapter06_atmosphere_fixture_worker.py
```

The writer binds these four accepted inputs before launching either scientific
child. The scientific-worker acceptance itself remains scoped exactly as
recorded by the accepted worker audit; this writer does not broaden it.

## 3. Findings

| severity | findings |
| --- | ---: |
| P0 | 0 |
| P1 | 0 |
| P2 | 0 |

No blocking or nonblocking implementation defect was found in the assigned
snapshot.

## 4. Construction-owned A/B observations

### 4.1 Production surface: pass

The sole accepted production constructor is:

```python
build_deterministic_atmosphere_fixture_archive()
```

Its signature has zero parameters. It cannot receive a caller-provided
mapping, capture, byte string, process token, cache root, environment,
destination, or path. Passing a mapping raises `TypeError`.

Independent AST inspection found only these non-private top-level callables:

- `sha256_bytes`;
- `verify_accepted_identities`; and
- `build_deterministic_atmosphere_fixture_archive`.

The other non-private names are the exception and immutable result classes.
There is no second production builder and no private assembly helper that
accepts caller-supplied A/B captures. The private mapping serializer is a
canonical encoding primitive; it neither grants accepted semantics nor
returns the accepted result/summary and is not a production bypass.

The public builder itself:

1. checks inherited process controls;
2. verifies the accepted worker chain;
3. creates two unrelated disposable cache directories;
4. launches one child for A and another child for B;
5. validates both complete transports and their independent origin evidence;
6. serializes A and B independently; and
7. requires exact final byte equality.

### 4.2 Process controls and fail-closed order: pass

Before the first scientific child launch, the parent rejects a conflicting
inherited value for every one of the twelve accepted controls:

```text
MKL_DYNAMIC
MKL_NUM_THREADS
NUMBA_NUM_THREADS
NUMEXPR_NUM_THREADS
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
VECLIB_MAXIMUM_THREADS
LC_ALL
PYTHONHASHSEED
PYTHONDONTWRITEBYTECODE
PYTHONNOUSERSITE
TZ
```

It also rejects any inherited `PAYNE_ZERO_DATA_ROOT`. Missing accepted controls
are completed with their exact frozen values; conflicting values are not
normalized or silently overwritten. The focused adversarial test patches the
child-launch call and proves that all thirteen conflicts fail before that call.

Each child repeats the twelve-control comparison as the first executable
bootstrap block, before importing the writer, NumPy, Numba, or Payne Zero.
The parent-supplied environment cannot contain `PAYNE_ZERO_DATA_ROOT` because
the parent rejects it before constructing either child environment.

### 4.3 Distinct child origin evidence: pass

Each child receives a parent-generated one-use origin token and its own
parent-created cache path. The returned frame binds:

- the exact origin token;
- a positive child PID;
- the SHA-256 identity of the assigned resolved cache path;
- the complete transport byte count; and
- the complete transport SHA-256.

The A/B gate rejects one capture object, one token, one PID, one cache-root
identity, incomplete cache evidence, a changed transport hash, or unequal
complete transports.

For an additional check not supplied by the candidate suite, I instrumented
the `subprocess.run` boundary with `Popen`, recorded the operating-system PID,
and compared it with the PID reported inside each accepted child frame:

```text
capture A actual/reported PID
  27338 / 27338

capture B actual/reported PID
  27428 / 27428
```

Thus the two reported PIDs were not merely distinct strings; they matched two
distinct processes observed independently at the parent launch boundary.

### 4.4 Cache isolation and atmosphere-lane inventory: pass

This atmosphere route legitimately compiles Numba kernels. Requiring the cache
to remain empty after capture would therefore be scientifically false. The
correct gate is:

- an existing, empty, nonsymlink directory before each child;
- distinct A/B resolved paths;
- external to the textbook, pinned Payne Zero, and paper roots;
- no inherited prior compiler artifacts;
- inspection after the child; and
- complete disposal before the result returns.

The implementation enforces the first four properties and records the
post-capture entry count. The tests reject an occupied cache, symlink cache,
forbidden-root cache, shared A/B path, or incomplete cache evidence.

I independently inspected both cache trees after capture but before their
temporary-directory contexts exited. A and B had the same lane-honest
inventory:

| inventory property | A | B |
| --- | ---: | ---: |
| package directories | 1 | 1 |
| `.nbi` Numba index files | 18 | 18 |
| `.nbc` Numba compiled files | 18 | 18 |
| all entries | 37 | 37 |
| symlinks | 0 | 0 |
| non-Numba files | 0 | 0 |

The eighteen compiled/index pairs corresponded to these actually exercised
kernels:

- continuum opacity: Coulomb free-free Gaunt, helium low-level grid, neutral
  iron branch, Karzas-Latter grid, parallel and serial interpolation,
  lukewarm-metal absorption, and exact Planck frequency;
- equation of state: Saha output formatting, iron-group partition function,
  batched and single-depth Saha partition, and special partition function;
- line opacity: selected-line opacity and selected-line wings; and
- molecular equilibrium: equilibrium constants, Newton matrix, and Newton
  update.

The two cache paths were distinct and external. Both no longer existed when
the public builder returned. Compiler-cache bytes and paths remained
disposable process state and did not enter the scientific archive.

## 5. Complete worker-output validation

Each child returns a transient canonical transport containing:

```text
fixture arrays
  19

ephemeral evidence arrays
  89

total arrays
  108
```

The writer validates both captures independently before comparing them. The
validation binds:

- the exact fixture member set, member shapes, dtypes, C-byte hashes,
  projection descriptors, sentinel, monotonic grid, and finiteness through the
  accepted worker validator;
- exact object-free C-contiguous evidence membership through the accepted full
  schema digest;
- every evidence value through the accepted full fingerprint;
- full/projected/placeholder parity;
- selected-line output parity and exact line-output identity;
- all twelve process-control evidence values;
- accepted worker and source identities;
- the no-write/no-publication lifecycle declarations; and
- independent recomputation of the four accepted capture identities.

The independently reproduced identities were:

| identity | exact result |
| --- | --- |
| fixture schema digest | `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698` |
| physical payload fingerprint | `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663` |
| full capture schema digest | `cdf470038e67301b4c19b0691e672cd97df3233a3decf1b88e32ce3ac0dc1371` |
| full capture fingerprint | `a25875097c7084ffe2577de65c2913d8775f29613823d9e6e0ab0d9db4644654` |
| fixture mapping digest | `f533e3e327c879b1d367a89822bcd1847b15a73ba3d234a9c577c81997f75e0a` |
| evidence mapping digest | `f5e49f9f7f49c4604c08a53def65a9ee3b6ddff37376130eb274646ee375af4f` |
| selected-line output SHA-256 | `43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58` |

The complete A and B transports were byte-identical:

```text
transport bytes
  33,147,771

transport SHA-256
  0523ed254a78edaa07480bc30f23082f535e885f46d867de10f92cba1acd5b16
```

The tests additionally reject a one-ULP fixture change and a mutation to a
nonfixture evidence member. Only after the 19+89 validation and whole-transport
equality succeed can final serialization occur.

## 6. Exact nineteen-array archive

Only the nineteen accepted fixture arrays enter the final NPZ. Process tokens,
PIDs, cache paths/counts, transport identities, worker evidence, source
identities, and the 89 ephemeral evidence arrays remain outside it.

The final candidate reproduced exactly:

| final property | exact result |
| --- | --- |
| scientific members | `19` |
| scientific array bytes | `357,984` |
| archive bytes | `363,050` |
| archive SHA-256 | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |
| A/B final archive equality | byte-identical |

Independent decoding recovered the exact lexical `FIXTURE_SCHEMA` member set.
Every decoded shape, dtype, and C-byte hash matched the accepted worker table.
The sum of decoded array `nbytes` was `357,984`, and independent re-encoding
reproduced the complete original byte string.

## 7. Canonical byte representation

### 7.1 ZIP and NPY metadata: pass

Independent `zipfile`/NumPy inspection confirmed:

| property | exact result |
| --- | --- |
| archive bound | at most `1,048,576` bytes |
| member count | `19` |
| member order | lexical and unique |
| suffix | `.npy` |
| compression | `ZIP_STORED` |
| compressed size | equal to stored size for every member |
| ZIP timestamp | `1980-01-01 00:00:00` |
| creator system | Unix, `3` |
| create/extract version | `20` / `20` |
| flag bits | `0` |
| internal attributes | `0` |
| external attributes | regular file `0600`, exact `0o100600 << 16` |
| member extra/comment | empty |
| archive comment | empty |
| NPY version | `2.0` for all members |
| object dtype | absent/forbidden |
| array layout | C-contiguous |
| Zip64 | disabled |

The transient 108-member transport is separately bounded to 128 MiB. Both
formats use `ZIP_STORED`, so compressed-byte bounds are also decoded-payload
bounds; there is no decompression-bomb ambiguity.

### 7.2 Canonical decoder and adversarial encodings: pass

The decoder requires immutable nonempty `bytes`, checks all ZIP metadata before
accepting members, uses `allow_pickle=False`, requires NPY 2.0 and C-contiguous
object-free arrays, then re-encodes the entire mapping and requires complete
whole-byte equality.

In addition to the focused suite, I independently built a small canonical
archive and verified rejection of fifteen mutated byte representations:

1. trailing archive bytes;
2. archive truncation;
3. valid-structure CRC/data corruption;
4. reversed member order;
5. changed timestamp;
6. changed file permissions;
7. a nonempty ZIP extra field;
8. `ZIP_DEFLATED` compression;
9. NPY version 1.0;
10. trailing bytes inside an otherwise readable NPY member;
11. a duplicate member;
12. a parent-path member;
13. an absolute-path member;
14. a nonempty archive comment; and
15. an object-dtype NPY member.

The same independent check rejected six unsafe serializer base names
(`""`, slash, backslash, `.`, `..`, and an ambiguous pre-suffixed `.npy`
name), plus mutable `bytearray`, empty bytes, and over-ceiling bytes. The
candidate suite separately covers alternate valid-CRC scientific payload and
shape changes and requires semantic rejection by the accepted worker schema.

Therefore a merely readable NPZ is insufficient: accepted bytes must be the
unique representation of the exact accepted science.

## 8. Immutability and authority boundary

### 8.1 Source and data immutability: pass

I took size/SHA-256 snapshots immediately before and after the independently
instrumented public build for 92 labeled accepted/source/data entries:

- the four accepted worker-chain files;
- all 35 frozen pinned Payne Zero Python sources;
- all eleven accepted pinned dynamic data inputs;
- all three staged converter dependencies; and
- every regular file currently under the textbook `data/` tree.

The before/after snapshots were identical. No canonical textbook data file had
a new modification time from the audit runs. The only filesystem writes were
inside the two external disposable Numba-cache directories described above,
and both directories were removed before return.

### 8.2 No publisher or write authority: pass

Source and AST inspection found:

- no output or destination parameter;
- no command-line parser or `__main__` entry point;
- no canonical fixture, golden, or manifest path;
- no authorization parser or token;
- no publisher, overwrite, rename, repair, or alternate-root operation;
- no NumPy save/savez/tofile or `Path.write_*` call; and
- no artifact mutation.

The ZIP writer writes to `BytesIO`, not a path. The public result contains
immutable archive `bytes` and a JSON-safe review summary. Its lifecycle fields
remain:

```text
publication_authorized
  false

fixture_publication_performed
  false

golden_publication_performed
  false

manifest_mutation_performed
  false

artifact_file_write_performed
  false
```

## 9. Verification executed

Focused writer suite:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_chapter06_atmosphere_fixture_writer.py

14 passed in 47.03s
```

Accepted converter/worker/writer chain, with all twelve controls supplied
explicitly:

```text
env \
  MKL_DYNAMIC=FALSE \
  MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  LC_ALL=C \
  PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  TZ=UTC \
  PYTHONPATH=src:. \
  /Users/ysting/anaconda3/bin/python -m pytest -q \
    tests/test_chapter06_atmosphere_line_converter.py \
    tests/test_chapter06_atmosphere_fixture_worker.py \
    tests/test_chapter06_atmosphere_fixture_writer.py

41 passed in 95.02s
```

Static verification:

```text
/Users/ysting/anaconda3/bin/python -m ruff check \
  scripts/chapter06_atmosphere_fixture_writer.py \
  tests/test_chapter06_atmosphere_fixture_writer.py

All checks passed!

/Users/ysting/anaconda3/bin/python -m ruff format --check \
  scripts/chapter06_atmosphere_fixture_writer.py \
  tests/test_chapter06_atmosphere_fixture_writer.py

2 files already formatted

/Users/ysting/anaconda3/bin/python -m py_compile \
  scripts/chapter06_atmosphere_fixture_writer.py \
  tests/test_chapter06_atmosphere_fixture_writer.py

pass

git diff --check -- \
  scripts/chapter06_atmosphere_fixture_writer.py \
  tests/test_chapter06_atmosphere_fixture_writer.py \
  design/chapter06_atmosphere_fixture_writer_candidate.md

pass
```

Additional independent runtime probes:

```text
instrumented public build
  two actual PIDs matched the two framed child PIDs
  A/B cache inventory: 37 entries each
  cache disposal: pass
  92-entry source/data snapshot: unchanged
  archive: 363,050 bytes
  archive SHA-256:
    1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
  decode/re-encode whole-byte identity: pass

independent canonical corruption matrix
  canonical control: pass
  mutated byte representations rejected: 15/15
  unsafe serializer names rejected: 6/6
  mutable/empty/oversize boundary inputs rejected: 3/3
```

I did not duplicate the repository-wide suite because other agents were
actively changing the shared tree during this audit. The directly relevant
accepted scientific chain was stable at its assigned hashes and passed in
full. This report makes no independent repository-wide regression claim.

## 10. Residual scope

The following are deliberate boundaries, not defects:

1. The cache is empty before each atmosphere child but nonempty afterward
   because this lane compiles eighteen Numba kernels. Its contents are
   disposable compiler state, not scientific archive members.
2. Origin tokens, PIDs, cache paths, and their summary digest vary by build.
   They prove process separation and remain outside deterministic scientific
   bytes.
3. Acceptance is tied to the exact worker/source/data/platform snapshot. It is
   not a portability claim.
4. The 33.1-MiB complete transport is ephemeral audit evidence, not a
   publication artifact.
5. No canonical fixture path, publisher, detached authorization, manifest
   entry, golden, atmosphere oracle, seam suite, or notebook publication has
   been reviewed or authorized here.

## 11. Final disposition

**ACCEPT** the exact deterministic in-memory atmosphere-fixture writer:

```text
scripts/chapter06_atmosphere_fixture_writer.py
SHA-256
0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538
```

with focused tests:

```text
tests/test_chapter06_atmosphere_fixture_writer.py
SHA-256
c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a
```

and in-memory candidate bytes:

```text
bytes
363,050

SHA-256
1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
```

This acceptance authorizes only progression to the next separately designed
and independently reviewed gate. **Publication remains unauthorized.**
