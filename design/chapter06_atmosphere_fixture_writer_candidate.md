# Chapter 6 deterministic atmosphere-fixture writer candidate

Status: implementation candidate for independent review  
Date: 2026-07-30  
Disposition: **CANDIDATE — NOT ACCEPTANCE, AUTHORIZATION, OR PUBLICATION**

## Scope

This pass implements only the in-memory serialization gate after the
independently accepted Chapter 6 atmosphere scientific worker. It creates:

```text
scripts/chapter06_atmosphere_fixture_writer.py
tests/test_chapter06_atmosphere_fixture_writer.py
design/chapter06_atmosphere_fixture_writer_candidate.md
```

The writer owns two fresh scientific captures, validates the complete accepted
worker output, and returns:

1. one immutable canonical NPZ byte string containing exactly the nineteen
   accepted fixture arrays; and
2. a JSON-safe review summary containing separate process evidence.

It has no output path, command-line interface, filesystem artifact writer,
publisher, authorization parser, canonical fixture, golden, manifest mutation,
overwrite, rename, repair, or alternate-root operation. No fixture, golden,
manifest, or canonical data file was created or changed.

## Candidate implementation identities

| object | SHA-256 |
| --- | --- |
| deterministic atmosphere writer | `0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538` |
| focused writer tests | `c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a` |

The writer binds these exact independently accepted inputs before either
scientific child is launched:

| accepted input | SHA-256 |
| --- | --- |
| scientific worker | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| scientific-worker tests | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| worker candidate record | `da53e4846ee91f814c437994fff604ae91ed94a4da4db7856f8f47bb61cf72dc` |
| worker independent final acceptance | `336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e` |

Every bound path must be a regular nonsymlink file. The imported worker must
resolve to the bound canonical repository path. A changed expected identity
fails before child capture.

## Non-bypassable fresh A/B topology

The only public production builder is exactly:

```python
build_deterministic_atmosphere_fixture_archive()
```

It accepts no mapping, capture, cache, environment, destination, or path
argument. The production builder therefore cannot be asked to treat the same
mapping, copied mapping, decoded byte string, process, token, or cache as two
fresh observations.

The builder creates two separate `TemporaryDirectory` cache roots as caller
state for children A and B. Each root must be:

- an existing directory created empty immediately before its child;
- a real nonsymlink path;
- external to the textbook, pinned Payne Zero, and paper trees;
- distinct from the other child's root.

The atmosphere kernels legitimately compile into their private Numba caches.
Both candidate runs produced 37 directory/file entries after their respective
captures. The gate therefore proves **empty before and isolated during use**,
not the scientifically false condition that a compiling atmosphere run leaves
the cache empty. Both complete cache trees are deleted when their enclosing
temporary-directory contexts exit, and disposal is verified before candidate
bytes are returned.

The writer rejects an inherited value that conflicts with any of the accepted
worker's twelve process controls. It fills controls that are absent, but never
silently replaces a conflicting inherited request. The child repeats the full
twelve-control check before importing the writer, NumPy, Numba, or Payne Zero.
An inherited `PAYNE_ZERO_DATA_ROOT` is also rejected rather than removed or
overwritten; the accepted worker owns explicit, hash-gated data paths.
Each child then receives the accepted environment, its own cache root, and a
parent-generated one-use origin token.

Each child reports, in a framed header outside scientific bytes:

```text
origin token
child PID
cache-root identity
complete-capture byte count
complete-capture SHA-256
```

The parent binds the token it generated, the cache path it created, the actual
child PID, and the exact returned transport. It rejects equal capture objects,
origin tokens, child PIDs, or cache-root identities.

## Complete accepted worker-output validation

The child returns one transient canonical transport containing:

```text
19 fixture arrays
89 ephemeral evidence arrays
108 total arrays
```

The process header is not part of this transport. The final fixture archive is
not part of the process header.

Before the fixture can be serialized, each capture independently passes:

1. exact nineteen-member names, shapes, dtypes, C-byte hashes, projection
   descriptors, wavelength-grid monotonicity, sentinel, and finiteness through
   the accepted worker validator;
2. exact 89-member evidence count and object-free C-contiguous arrays;
3. exact full/projected/placeholder and selected-line parity decisions;
4. exact no-write and no-publication lifecycle declarations;
5. the complete accepted twelve-control environment evidence;
6. exact worker and selected-line-output identities;
7. independent recomputation of all four accepted capture identities.

The required identities are:

| accepted capture identity | exact value |
| --- | --- |
| fixture schema digest | `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698` |
| physical payload fingerprint | `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663` |
| full capture schema digest | `cdf470038e67301b4c19b0691e672cd97df3233a3decf1b88e32ce3ac0dc1371` |
| full capture fingerprint | `a25875097c7084ffe2577de65c2913d8775f29613823d9e6e0ab0d9db4644654` |
| selected-line output SHA-256 | `43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58` |

Only after both captures pass independently does the parent require:

```text
complete transport A == complete transport B, byte for byte
fixture names/dtypes/shapes/C bytes A == B
all 89 evidence names/dtypes/shapes/C bytes A == B
```

The reproduced transient capture is:

| complete transient capture property | exact value |
| --- | --- |
| bytes | `33,147,771` |
| SHA-256 | `0523ed254a78edaa07480bc30f23082f535e885f46d867de10f92cba1acd5b16` |
| fixture mapping digest | `f533e3e327c879b1d367a89822bcd1847b15a73ba3d234a9c577c81997f75e0a` |
| evidence mapping digest | `f5e49f9f7f49c4604c08a53def65a9ee3b6ddff37376130eb274646ee375af4f` |
| A/B whole-transport equality | exact |

This transient transport is pipe-local, is not returned by the public builder,
and is never written as an artifact.

## Canonical nineteen-member NPZ encoding

Only the nineteen fixture arrays enter the final scientific bytes. Their
lexical member names are the exact accepted worker names; process tokens, PIDs,
cache identities, cache counts, worker evidence, fingerprints, and source
identities do not enter the archive.

| ZIP/NPY property | fixed value |
| --- | --- |
| member count | `19` |
| member order | lexical base-name order |
| member suffix | `.npy` |
| compression | `ZIP_STORED` |
| ZIP date/time | `1980-01-01 00:00:00` |
| creator system | Unix, code `3` |
| create/extract version | `20` / `20` |
| flag bits | `0` |
| internal attributes | `0` |
| external attributes | regular file mode `0600`, exact `0o100600 << 16` |
| member extra/comment | empty |
| archive comment | empty |
| NPY version | `2.0` |
| pickle | forbidden |
| object dtype | forbidden |
| array storage | C-contiguous |
| Zip64 | disabled |

Duplicate, empty, absolute, slash-containing, backslash-containing,
dot-component, parent-component, directory, and ambiguous `.npy` base names
are rejected. The final archive is bounded to one MiB.

The exact in-memory candidate is:

```text
array payload bytes
  357,984

archive bytes
  363,050

archive SHA-256
  1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
```

Archive A and archive B are byte-identical.

## Serialized-form validation

The reader validates the byte representation rather than accepting any
readable NPZ:

1. immutable, nonempty `bytes` below the one-MiB bound;
2. empty archive comment;
3. exact lexical and unique nineteen-member set;
4. safe non-path-like member names;
5. exact fixed ZIP metadata and stored sizes;
6. canonical NPY 2.0 bytes with `allow_pickle=False`;
7. object-free C-contiguous decoded arrays;
8. exact accepted fixture schema and every frozen member hash;
9. exact dtype, shape, and C bytes against its accepted child capture;
10. canonical re-encoding equal to the complete original byte string.

The whole-byte re-encoding requirement rejects otherwise readable alternative
timestamps, permissions, compression, order, comments, headers, extra fields,
or trailing bytes.

## Adversarial verification

The focused suite covers:

- a zero-argument-only public production surface;
- a changed accepted source identity before child capture;
- all twelve inherited environment conflicts before child launch;
- an inherited `PAYNE_ZERO_DATA_ROOT` before child launch;
- completion of absent environment controls to the exact accepted values;
- reuse of one capture object;
- a deep-copied mapping/transport carrying one origin token;
- reuse of one child PID;
- sharing one cache-root identity;
- incomplete empty-before cache evidence;
- an occupied cache before launch;
- a symlink cache;
- a cache below a forbidden source tree;
- equal A/B cache paths;
- a one-ULP fixture mutation;
- a complete-evidence mutation;
- object dtype and oversize rejection;
- a valid-CRC payload mutation and shape mutation;
- reverse/nonlexical order;
- changed timestamp and permissions;
- `ZIP_DEFLATED` compression;
- NPY version 1.0;
- a parent-path member;
- a duplicate member;
- a nonempty archive comment;
- truncated and corrupted archive bytes;
- exact decode/re-encode identity;
- process-evidence exclusion from scientific bytes;
- source and canonical-data immutability;
- absence of artifact-writer, publisher, authorization, manifest, golden, and
  output-path surfaces.

Focused execution:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_chapter06_atmosphere_fixture_writer.py

14 passed in 35.63s
```

Accepted atmosphere chain:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_chapter06_atmosphere_line_converter.py \
  tests/test_chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_writer.py

41 passed in 87.89s
```

Repository-wide regression:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q

410 passed, 1 skipped in 166.75s
```

Static checks:

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
```

All runtime commands supplied the exact accepted worker controls:

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
```

## Limitations and next gate

This candidate deliberately does not establish:

1. independent acceptance of this writer, tests, or candidate bytes;
2. authority to create a canonical fixture file;
3. authority to mutate `data/MANIFEST.json`;
4. a fixture publisher or detached authorization contract;
5. an atmosphere oracle, seam-suite acceptance, sparse golden, or notebook
   publication;
6. portability to a different accepted worker/source/data/platform snapshot.

The 33.1-MiB full-capture transport is intentionally transient audit evidence,
not a publication candidate. The 363,050-byte scientific archive remains only
an in-memory candidate. An independent reviewer must reproduce the source
gates, fresh A/B topology, complete evidence identities, canonical encoding,
malformed-archive rejection, and exact whole-byte identity before any later
publication gate is designed or authorized.

**CANDIDATE ONLY. THIS RECORD DOES NOT ACCEPT OR AUTHORIZE PUBLICATION.**
