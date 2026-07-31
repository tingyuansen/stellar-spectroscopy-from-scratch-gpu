# Chapter 6 atmosphere-fixture candidate-byte acceptance

Date: 2026-07-30  
Reviewer role: independent candidate-byte reviewer; no implementation or
publication authority  
Disposition: **ACCEPT the exact in-memory candidate bytes only; NOT
AUTHORIZATION**

## 1. Decision boundary

This review is the contract-required, read-only candidate-byte gate for the
Chapter 6 **atmosphere** lane. It accepts one exact byte string as a candidate
for a later, separately implemented and independently reviewed publisher:

```text
archive bytes
  363,050

archive SHA-256
  1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
```

The accepted candidate has no canonical path or repository data role. This
review does not:

- re-accept or modify the scientific worker or deterministic writer;
- implement or accept a publisher;
- create a detached authorization or authorization review;
- create `data/fixtures/chapter06_atmosphere_one_line_inputs.npz`;
- add a manifest entry;
- accept an atmosphere output, comparison golden, oracle, seam suite, or
  notebook result; or
- permit a write to the pinned Payne Zero or paper trees.

The exact trust-object path is the path frozen by the accepted publisher
contract:

```text
design/chapter06_atmosphere_fixture_byte_acceptance.md
```

This report is evidence for a future gate. It grants no artifact write.

## 2. Frozen input identities

### 2.1 Deterministic writer

The four exact writer-side objects frozen by the publisher contract were
regular nonsymlink files and matched byte for byte:

| reviewed object | SHA-256 |
| --- | --- |
| `scripts/chapter06_atmosphere_fixture_writer.py` | `0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538` |
| `tests/test_chapter06_atmosphere_fixture_writer.py` | `c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a` |
| `design/chapter06_atmosphere_fixture_writer_candidate.md` | `e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee` |
| `design/chapter06_atmosphere_fixture_writer_independent_audit.md` | `b946dcef0beeacf49a3da9ac036e21af7cd7b44d15092639cd3be744fb42f0f9` |

The independent writer audit accepts exactly the first three objects at those
identities and limits progression to this separately reviewed byte gate.

### 2.2 Accepted scientific chain

The writer's complete accepted worker chain also matched:

| reviewed object | SHA-256 |
| --- | --- |
| `design/chapter06_atmosphere_fixture_oracle_plan.md` | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| `scripts/chapter06_atmosphere_line_converter.py` | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| `tests/test_chapter06_atmosphere_line_converter.py` | `254d796b7ab761ca806c372d0bcdd935067ff1a89b2acfebcfa3007fe3f549dc` |
| `design/chapter06_atmosphere_converter_candidate.md` | `acae959e9b01f986a553e9806fa7f60c6bb770ce3f356fab11c2d7509b63d03a` |
| `design/chapter06_atmosphere_converter_independent_audit.md` | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |
| `scripts/chapter06_atmosphere_fixture_worker.py` | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| `tests/test_chapter06_atmosphere_fixture_worker.py` | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| `design/chapter06_atmosphere_fixture_worker_candidate.md` | `da53e4846ee91f814c437994fff604ae91ed94a4da4db7856f8f47bb61cf72dc` |
| `design/chapter06_atmosphere_fixture_worker_independent_audit.md` | `336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e` |

The worker acceptance is final only for its in-memory safe-sequence boundary.
The converter acceptance is likewise limited to its recorded safe-sequence
steps. This byte review does not broaden either disposition.

### 2.3 Publisher design

The shared two-lane publisher design and its complete audit matched:

| reviewed object | SHA-256 |
| --- | --- |
| `design/chapter06_lane_artifact_publisher_contract.md` | `9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774` |
| `design/chapter06_lane_artifact_publisher_contract_independent_audit.md` | `7f2517ad0abcca312dcf22785e483fe033519264d5513b16ebe6fa580d4521fd` |

The contract explicitly leaves this byte acceptance, the publisher,
publisher tests, publisher acceptance, detached authorization, authorization
review, canonical file, manifest registration, and postpublication audit as
separate trust objects. Accepting the contract is not permission to skip any
of them.

## 3. Two unrelated top-level constructions

I invoked the sole accepted production surface twice from two unrelated
top-level CPython interpreter processes:

```python
build_deterministic_atmosphere_fixture_archive()
```

Its inspected signature was exactly:

```text
() -> DeterministicAtmosphereFixtureArchive
```

It accepts no caller mapping, captured state, byte string, cache path, output
path, destination, manifest, or authorization. Each invocation independently
owned:

1. one fresh child capture A;
2. a distinct fresh child capture B;
3. two distinct, initially empty, external cache roots;
4. complete validation of both 19-array fixture mappings;
5. complete validation of both 89-array worker-evidence mappings;
6. direct A/B transport-byte equality;
7. direct A/B final-archive-byte equality; and
8. disposal of both owned cache roots before return.

The two accepted top-level results were:

| property | top-level run 1 | top-level run 2 |
| --- | ---: | ---: |
| final members | 19 | 19 |
| scientific array bytes | 357,984 | 357,984 |
| archive bytes | 363,050 | 363,050 |
| archive SHA-256 | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |
| fixture A/B bitwise equality | pass | pass |
| evidence A/B bitwise equality | pass | pass |
| complete transport A/B byte equality | pass | pass |
| final archive A/B byte equality | pass | pass |
| distinct origins | pass | pass |
| distinct child processes | pass | pass |
| distinct cache roots | pass | pass |
| both cache roots disposed | pass | pass |

The final candidate identity was therefore reproduced from four fresh
scientific children across the two accepted top-level runs. No child, origin
token, PID, or cache root was reused.

One intermediate audit-harness invocation was excluded from this table after
the accepted writer had returned because the harness queried an obsolete
manifest key name (`files` rather than `entries`). That was a reviewer-harness
error, not a writer, science, byte, or mutation failure. The clean second
top-level run above repeated all required checks with the live manifest
schema.

## 4. Exact execution environment and cache topology

The accepted runs used:

| property | exact review value |
| --- | --- |
| interpreter | `/Users/ysting/anaconda3/bin/python` |
| Python | CPython 3.13.9 |
| NumPy | 2.3.5 |
| Numba | 0.62.1 |
| platform | macOS 26.5.2, arm64 |
| resolved temporary root | `/var/folders/zg/v3tx69hd27z2jysr851rlt340000gn/T` |
| temporary-root decision | `tempfile.gettempdir()`, validated external to textbook, Payne Zero, and paper roots |
| top-level relation | two unrelated interpreter processes |
| child relation within each run | two sequential, distinct subprocesses |
| cache relation within each run | distinct writer-owned temporary directories |
| cache initial state | existing, empty, nonsymlink directory |
| cache final state | inspected, then completely disposed before return |

Every top-level run explicitly supplied the complete accepted controls:

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

`PAYNE_ZERO_DATA_ROOT` was absent. The writer independently rejects a
conflicting inherited value for any accepted control and rejects any
inherited `PAYNE_ZERO_DATA_ROOT` before child launch.

I wrapped only the accepted child boundary for observation; the wrapper
delegated to the exact accepted implementation and did not replace a capture,
mapping, validator, serializer, or result. The complete post-child inventory
was identical for all four accepted children:

| cache property | run 1 A | run 1 B | run 2 A | run 2 B |
| --- | ---: | ---: | ---: | ---: |
| package directories | 1 | 1 | 1 | 1 |
| `.nbi` regular files | 18 | 18 | 18 | 18 |
| `.nbc` regular files | 18 | 18 | 18 | 18 |
| total entries | 37 | 37 | 37 | 37 |
| symlinks | 0 | 0 | 0 | 0 |
| other files | 0 | 0 | 0 | 0 |
| root absent after return | yes | yes | yes | yes |

This is the atmosphere lane's accepted cache behavior. It is not replaced by
the synthesis lane's empty-after-child rule.

The origin tokens, child PIDs, cache paths, cache-path hashes, and
process-evidence digest correctly differed between constructions. They prove
freshness but are deliberately excluded from scientific identity and from the
nineteen archive members. No nondeterministic process value was frozen as a
candidate-byte requirement.

## 5. Complete worker-evidence gate

Before final serialization, each child produced a transient canonical
transport containing:

```text
scientific fixture arrays
  19

ephemeral evidence arrays
  89

complete arrays
  108
```

Both top-level runs reproduced:

| identity | exact value |
| --- | --- |
| complete transport bytes | `33,147,771` |
| complete transport SHA-256 | `0523ed254a78edaa07480bc30f23082f535e885f46d867de10f92cba1acd5b16` |
| fixture mapping digest | `f533e3e327c879b1d367a89822bcd1847b15a73ba3d234a9c577c81997f75e0a` |
| evidence mapping digest | `f5e49f9f7f49c4604c08a53def65a9ee3b6ddff37376130eb274646ee375af4f` |
| fixture schema digest | `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698` |
| physical payload fingerprint | `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663` |
| full-capture schema digest | `cdf470038e67301b4c19b0691e672cd97df3233a3decf1b88e32ce3ac0dc1371` |
| full-capture fingerprint | `a25875097c7084ffe2577de65c2913d8775f29613823d9e6e0ab0d9db4644654` |
| selected-line output SHA-256 | `43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58` |

The writer validated both complete transports independently. The bound
evidence includes:

- exact accepted worker and source identities;
- every fixture member's name, dtype, shape, C-byte hash, and physical
  boundary;
- the complete dynamic read set;
- exact twelve-control environment evidence;
- full/projected line-output equality;
- placeholder/full fixture equality;
- exact line-output identity;
- complete schema and fingerprint recomputation; and
- false no-write/no-publication declarations for fixture, golden, and
  manifest operations.

Only after both complete 108-array transports were canonical, valid, and
byte-identical did the writer serialize the nineteen fixture members.

The 33.1-MiB transport, the selected-line output, the full packed arrays, the
three projected arrays, the placeholder arrays, and all process evidence are
audit state. None is an archive member.

## 6. Exact nineteen-member scientific payload

Independent decoding with `zipfile` and NumPy recovered this exact lexical
member set. Hashes are over each decoded array's C-order data bytes:

| lexical member | shape | dtype | array bytes | C-byte SHA-256 |
| --- | --- | --- | ---: | --- |
| `actual_population_slot_indices` | `(3,)` | `<i2` | 6 | `eff6cc5c731be5128ac078be458469978b6ac7c823665abd47098485291a5af2` |
| `actual_population_slot_values` | `(80, 3)` | `<f8` | 1,920 | `ae17c6b96dc9eb824051707d83eca2bbdb22483af4c143b18d832326292e86ce` |
| `continuum_line_selection_threshold` | `(80, 344)` | `<f4` | 110,080 | `76cb7ef18554149b61b97e32f2a69bbcdba3eb8b7e6b7da8ba3c69b8775b7293` |
| `effective_temperature` | scalar | `<f8` | 8 | `0ae804feb5a3896c0d30fd25ee9853a10ee08fed330969967ce16a4d07a7329b` |
| `electron_density` | `(80,)` | `<f8` | 640 | `1e36bdf5aac263da2eac448f1b4c91e43e48a48afc133c8a8ec5e15735c85c18` |
| `fractional_doppler_widths_at_line_slot` | `(80,)` | `<f8` | 640 | `f8018929e269db3e6b9a572d36b528bc0577a3a93bcc25166b277824ffb2e785` |
| `hc_over_kt` | `(80,)` | `<f8` | 640 | `fdbd300b88163af886fe2905a883fa319c32abcc89db1099e4b0df7965d8174e` |
| `line_population_slot_zero_based` | scalar | `<i2` | 2 | `8d6dd9f330421e48e73e97c0819d55a9965bac90b1f2f36057b4c906d5791f05` |
| `log_strength_index` | `(1,)` | `<i2` | 2 | `3fe821d54660a0c51b42d19a42571e208791b3c5ff6bc0cac16b6553a226515a` |
| `lower_excitation_index` | `(1,)` | `<i2` | 2 | `9b5bf0b74e2e212b57bcb2b9f2712eaab4c6169595f4e82767793f4534365648` |
| `opacity_wavelength_grid_nm` | `(30,000,)` | `<f8` | 240,000 | `8944f1dd701ba27f50d37a16e48ab9375e1bef1b444ed405b671ba91fde8132b` |
| `packed_species_slot` | `(1,)` | `<i2` | 2 | `c2126161f7488b7d198ea310da9a8694786a18a2317eea5e30379ee118d34743` |
| `packed_wavelength_index` | `(1,)` | `<i4` | 4 | `c423f4ac4a3825c6fad5336a1e15c0038ab17087f930916ad550ad45b4990dfc` |
| `partition_normalized_population_over_mass_density_and_fractional_doppler_width_at_line_slot` | `(80,)` | `<f8` | 640 | `8625e8d942892d142384d51fc0625701374f79dfa471a78ff593d88479b3056f` |
| `radiative_damping_index` | `(1,)` | `<i2` | 2 | `5cdf3b75730a5b45c3f24da2c1030103143981191d2844aa7374e948b9abaeea` |
| `stark_damping_index` | `(1,)` | `<i2` | 2 | `91d82e7b89ae43b29bb673bb416697d005e093d0011c9d7af501630c6a502141` |
| `temperature` | `(80,)` | `<f8` | 640 | `7b76410e731a6772cb6de12f923aea8a26e345778dbab67db3e8938278ce270f` |
| `van_der_waals_damping_index` | `(1,)` | `<i2` | 2 | `c0346f09a0362e8e9d29c01a9fc7c292a7395b524a90319bb921608b9fdb1b60` |
| `wavelength_bin_edges` | `(344,)` | `<i8` | 2,752 | `ae50e9c0bafdcbfc39e242fd5029bf53b9e712d43008c6cb78a4493b89272cd5` |

The sum is exactly `357,984` scientific array bytes.

Independent semantic checks also required:

- actual population slots exactly `[0, 2, 840]`;
- selected Fe I population slot exactly `350`, zero based;
- final wavelength-bin sentinel exactly \(2^{30}=1,073,741,824\);
- strictly increasing 30,000-point opacity wavelength grid;
- finite values in every floating-point member;
- exact 80-depth axes where specified; and
- exact accepted per-member hashes, so every other fixed scientific bound is
  inherited without a lossy tolerance.

## 7. Role and ownership audit

The archive is an **input-only 80-depth integration fixture**. It supplies the
smallest accepted state needed to isolate the selected-line atmosphere kernel.

It contains:

- one packed selected-line record;
- the required 80-depth thermodynamic and population projections;
- the wavelength grid and bin edges;
- the continuum line-selection threshold; and
- the effective temperature input.

It does not contain:

- the `(80, 30000)` selected-line opacity output;
- any full `(80, 1006)` packed atmosphere array;
- any projected-call output;
- any placeholder array;
- any ephemeral worker or process evidence;
- a source-row file or raw Chapter 6 subset copy;
- a Harris, FASTEX, or other static-table copy;
- a synthesis input or output;
- a golden or comparison member; or
- a member from which another Chapter 6 lane artifact is reconstructed.

The exact nineteen-member allowlist makes those exclusions closed rather than
name-fragment heuristics. The few packed integer fields are fixture-owned
converted kernel inputs, not a copied raw subset archive.

The accepted worker's transient capture envelope uses:

```text
CAPTURE_SCHEMA_VERSION = 1
```

That version is external metadata. The final archive contains no
`schema_version`, capture-schema, archive-schema, or other twentieth member.
The scientific schema digest is an identity, not an embedded version field.
Future authorization metadata must therefore use:

```text
fixture_capture_schema_version           1
archive_contains_embedded_schema_version false
npy_member_format_version                "2.0"
```

No future gate may invent `ATMOSPHERE_ARCHIVE_SCHEMA_VERSION`,
`meta__archive_schema_version`, or a twentieth archive member.

## 8. Canonical byte representation

Independent ZIP/NPY inspection froze:

| representation property | exact accepted value |
| --- | --- |
| archive type | NPZ / ZIP |
| archive size bound | at most 1,048,576 bytes |
| actual archive bytes | 363,050 |
| member count | 19 |
| member order | lexical, unique |
| member suffix | `.npy` |
| compression | `ZIP_STORED` |
| compressed versus stored size | equal for every member |
| ZIP timestamp | `1980-01-01 00:00:00` |
| creator system | Unix, `3` |
| create version | `20` |
| extract version | `20` |
| flag bits | `0` |
| internal attributes | `0` |
| external attributes | regular file `0600`, exact `0o100600 << 16` |
| member extra/comment | empty |
| archive comment | empty |
| Zip64 | disabled |
| NPY member version | `2.0` for all 19 members |
| object dtype | forbidden and absent |
| decoded layout | C-contiguous |
| pickle | disabled |

Every NPY stream ended exactly after its array; no member carried trailing
bytes. An independent serializer reconstructed all nineteen NPY members and
the complete ZIP container. The re-encoded byte string was exactly equal to
the candidate.

The transient 108-member transport was separately below its 128-MiB bound.
Neither transport bytes nor their bound define the final fixture's data role.

## 9. Independent adversarial byte audit

An independent decoder/validator, separate from the writer's accepted decoder,
accepted the canonical candidate and rejected all twelve constructed mutation
classes:

1. appended trailing archive bytes;
2. one-byte corruption;
3. reversed member order;
4. an alternate valid ZIP timestamp;
5. alternate `ZIP_DEFLATED` compression;
6. alternate valid NPY 1.0 member encodings;
7. a parent-path member name;
8. a nonempty archive comment;
9. a valid-CRC one-ULP scientific payload change;
10. a valid-CRC shape change;
11. a valid-CRC dtype change; and
12. a valid-CRC renamed member.

The accepted writer decoder independently rejected the same 12/12 candidates.
Thus rejection was not inferred merely from an expected final SHA-256.

The focused accepted suite additionally exercised:

- missing, duplicate, nonlexical, path-like, empty, dotted, slash, backslash,
  and already-suffixed names;
- truncation and malformed ZIP structures;
- changed permissions, comments, metadata, and compression;
- object dtype, mutable byte strings, empty bytes, and over-ceiling bytes;
- valid-CRC payload and shape changes;
- one-ULP fixture mutation;
- mutation of a nonfixture worker-evidence member;
- one reused capture object, origin, PID, or cache identity;
- occupied, symlink, shared, and forbidden-root caches;
- every inherited process-control conflict;
- an inherited `PAYNE_ZERO_DATA_ROOT`;
- changed accepted source identity before child launch; and
- source/data immutability plus absence of publisher surfaces.

Member names, dtypes, shapes, C-byte hashes, archive and transport bounds,
corruption, alternate-valid encodings, path names, trailing bytes, and the
complete worker-evidence gate are therefore independently covered.

## 10. Source, data, and manifest immutability

For each accepted top-level run I took a before/after snapshot and required
exact equality.

The normalized source closure comprised 59 regular nonsymlink objects:

- the ten writer/worker/contract control objects in the runtime trust closure;
- all 35 frozen pinned Payne Zero Python sources;
- all 11 accepted pinned dynamic data inputs; and
- all three staged converter dependencies.

The normalized snapshot records absolute path, type, mode, byte count, and
file SHA-256 in sorted canonical JSON. Its final digest was:

```text
source entries
  59

source snapshot SHA-256
  e24b8200aa452e7799e0b248bcd179ce79f2261d212b4aec185e1ad5d0f174df
```

The separately listed converter/plan provenance objects in Section 2.2 also
matched their accepted hashes after execution.

The normalized `data/` inventory records every descendant path, object type,
mode, file byte count, and regular-file SHA-256:

```text
data inventory entries
  51

data snapshot SHA-256
  b521bbb6c1e03cc52d7a888f536bc7b86fd62018d65e65f6e3ba477ff27fe78d

data/MANIFEST.json SHA-256
  d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a
```

The live manifest remained at 37 entries:

| role | entries |
| --- | ---: |
| `fixture` | 6 |
| `golden` | 12 |
| `static` | 18 |
| `subset` | 1 |

Before and after both accepted runs:

```text
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
  absent

matching manifest entries
  0
```

The accepted builder summary was false for every authority or mutation field:

```text
publication_authorized
fixture_publication_performed
golden_publication_performed
manifest_mutation_performed
artifact_file_write_performed
```

Independent source and AST inspection found:

- no command-line or `__main__` entry point;
- no output or destination parameter;
- no canonical fixture, golden, or manifest path literal;
- no authorization parser;
- no publisher function;
- no `save`, `savez`, `savez_compressed`, `tofile`, `write_bytes`,
  `write_text`, `rename`, or `replace` call; and
- only in-memory ZIP writes to `BytesIO`.

In the two accepted candidate constructions, the only runtime writes were the
four external writer-owned Numba cache trees. All were inspected and disposed
before their top-level builders returned. The excluded harness attempt and
the focused tests likewise confined their compiler/test state to disposable
temporary roots. No source, canonical data, manifest, Payne Zero, or paper
byte was changed.

At final review, all later atmosphere-lane objects remained absent:

```text
scripts/build_chapter06_atmosphere_fixture.py
tests/test_chapter06_atmosphere_fixture_publisher.py
design/chapter06_atmosphere_fixture_publisher_independent_audit.md
design/chapter06_atmosphere_fixture_publication_acceptance.json
design/chapter06_atmosphere_fixture_publication_record_review.json
design/chapter06_atmosphere_fixture_postpublication_audit.md
```

## 11. Verification commands and results

Both top-level builds used this exact outer invocation with independent audit
assertions supplied on standard input:

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
  /Users/ysting/anaconda3/bin/python -
```

The two audit bodies independently called the zero-argument production
builder, observed the cache boundary, decoded every member, recomputed every
array and schema identity, checked role exclusions and physical bounds,
re-encoded the candidate, compared source/data/manifest snapshots, and
required absence of the canonical target and manifest entry. The first body
also constructed the independent twelve-case adversarial matrix. The bodies
were deliberately not persisted as repository scripts because this gate was
authorized to create only this report.

The full relevant accepted chain passed:

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

41 passed in 148.64s
```

Static checks:

```text
/Users/ysting/anaconda3/bin/python -m ruff check \
  scripts/chapter06_atmosphere_fixture_writer.py \
  tests/test_chapter06_atmosphere_fixture_writer.py \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py

All checks passed!

/Users/ysting/anaconda3/bin/python -m ruff format --check \
  scripts/chapter06_atmosphere_fixture_writer.py \
  tests/test_chapter06_atmosphere_fixture_writer.py \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py

4 files already formatted

PYTHONDONTWRITEBYTECODE=1 \
  /Users/ysting/anaconda3/bin/python -m py_compile \
    scripts/chapter06_atmosphere_fixture_writer.py \
    tests/test_chapter06_atmosphere_fixture_writer.py \
    scripts/chapter06_atmosphere_fixture_worker.py \
    tests/test_chapter06_atmosphere_fixture_worker.py

pass

git diff --check -- \
  scripts/chapter06_atmosphere_fixture_writer.py \
  tests/test_chapter06_atmosphere_fixture_writer.py \
  design/chapter06_atmosphere_fixture_writer_candidate.md \
  design/chapter06_atmosphere_fixture_writer_independent_audit.md \
  scripts/chapter06_atmosphere_fixture_worker.py \
  tests/test_chapter06_atmosphere_fixture_worker.py \
  design/chapter06_atmosphere_fixture_worker_candidate.md \
  design/chapter06_atmosphere_fixture_worker_independent_audit.md \
  design/chapter06_lane_artifact_publisher_contract.md \
  design/chapter06_lane_artifact_publisher_contract_independent_audit.md

pass
```

The four explicit `py_compile` cache files created by that static check were
removed immediately afterward; pre-existing ignored cache files owned by
other concurrent work were not touched. At the final audit state, this report
is the only repository object created by this review.

I did not run the repository-wide suite because other agents were actively
changing unrelated textbook files in the shared worktree. The complete
converter/worker/writer chain relevant to this byte gate passed at the exact
reviewed identities.

## 12. Residual scope

These are intentional limits, not defects:

1. Candidate acceptance is bound to the exact writer, worker/source/data
   closure, interpreter environment, and publisher contract reviewed above.
   It is not a cross-platform portability claim.
2. The cache inventory is expected to be populated after each atmosphere
   child because eighteen Numba kernels compile. Cache contents remain
   disposable process state.
3. Capture tokens, PIDs, cache paths, and process-evidence digests vary. They
   establish independent observations and never enter scientific identity.
4. The 33,147,771-byte complete transport is transient evidence, not a fixture
   or golden.
5. The selected-line output is validated evidence only. A later atmosphere
   comparison golden requires its own oracle, byte acceptance, publisher,
   detached authorization, registration, and postpublication audit after the
   input fixture is canonically installed.
6. This record accepts no publisher implementation or authorization schema
   instance and cannot be used as either.

Any change to the accepted writer, writer tests, writer candidate, writer
audit, bound worker chain, publisher contract, scientific source/data closure,
nineteen-member payload, or canonical encoding invalidates this candidate-byte
acceptance. A changed candidate must repeat this gate; no hash may be silently
refreshed.

## 13. Final disposition

**ACCEPT** exactly this in-memory atmosphere-lane candidate:

```text
role if later and separately authorized
  fixture

scientific purpose
  input-only 80-depth selected-line atmosphere integration fixture

members
  19

scientific array bytes
  357,984

archive bytes
  363,050

archive SHA-256
  1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff

scientific fixture schema digest
  f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698

physical payload fingerprint
  f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663

external capture schema version
  1

embedded schema-version member
  absent

NPY member version
  2.0

ZIP compression
  ZIP_STORED
```

**NOT WRITER ACCEPTANCE. NOT PUBLISHER ACCEPTANCE. NOT PUBLICATION
AUTHORIZATION. NO DATA WRITE IS GRANTED.**

<!-- BEGIN DELIMITED FINAL-CONTRACT REBIND BYTE RE-AUDIT: 2026-07-30 -->

---

# Chapter 6 atmosphere-fixture final-contract byte re-audit

Date: 2026-07-30  
Reviewer role: independent candidate-byte reviewer; no implementation,
publisher, or publication authority  
Disposition: **ACCEPT the exact in-memory candidate bytes only**

## R1. Decision and preserved history

**ACCEPT** the exact 363,050-byte atmosphere fixture candidate at SHA-256

```text
1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
```

against the independently accepted final publisher contract:

```text
design/chapter06_lane_artifact_publisher_contract.md
a663369c3851d89468a41436b8faddeba9d3dcbeba79a7254037734f4a5b3666
```

and its final independent contract audit:

```text
design/chapter06_lane_artifact_publisher_contract_rebind_independent_audit.md
c4a4ca58d94ec71ec509238046afcb127e189ba0be98a96c9929488958a1c286
```

This delimited append preserves the preceding 28,077 bytes and 716 lines
byte-for-byte at historical SHA-256
`79b20c065f5f1b588c6796d6413a1b2aa5e1e5a60056968fc7913cd961e12fc6`.
That historical acceptance remains stale because it binds publisher contract
`9ee0029f…`. The present decision does not reinterpret it; it repeats the byte
gate against exact final contract `a663369c…`.

This is candidate-byte acceptance only. It does not accept or create a
publisher, publisher tests, publisher review, detached authorization,
authorization review, canonical fixture, manifest entry, postpublication
audit, atmosphere comparison golden, or cleanup record. Every future
publisher and authorization hash remains unresolved.

## R2. Exact rebound inputs

All rebound inputs were regular nonsymlink files and independently matched:

| reviewed object | SHA-256 |
| --- | --- |
| deterministic atmosphere writer | `0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538` |
| focused writer tests | `c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a` |
| writer candidate record | `e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee` |
| accepted writer audit | `b946dcef0beeacf49a3da9ac036e21af7cd7b44d15092639cd3be744fb42f0f9` |
| final publisher contract | `a663369c3851d89468a41436b8faddeba9d3dcbeba79a7254037734f4a5b3666` |
| final contract audit | `c4a4ca58d94ec71ec509238046afcb127e189ba0be98a96c9929488958a1c286` |

The accepted atmosphere scientific chain also remained exact:

| accepted precursor | SHA-256 |
| --- | --- |
| atmosphere fixture/oracle plan | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| atmosphere scientific worker | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| worker tests | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| worker candidate record | `da53e4846ee91f814c437994fff604ae91ed94a4da4db7856f8f47bb61cf72dc` |
| worker independent audit | `336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e` |
| accepted line converter | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| converter tests | `254d796b7ab761ca806c372d0bcdd935067ff1a89b2acfebcfa3007fe3f549dc` |
| converter candidate record | `acae959e9b01f986a553e9806fa7f60c6bb770ce3f356fab11c2d7509b63d03a` |
| converter independent audit | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |
| Chapter 6 exact source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |

The pinned Payne Zero checkout remained at
`9c44001feae40b85146630499e6f8a5fed42e5af`.

## R3. Two unrelated top-level constructions

Two unrelated top-level CPython processes each invoked exactly:

```python
build_deterministic_atmosphere_fixture_archive()
```

with no arguments. Each top-level process retained the accepted writer's own
fresh child A and fresh child B. Across the two constructions:

- all four child PIDs were distinct;
- all four one-use origin tokens were distinct;
- all four cache-root identities were distinct;
- every cache root was an existing, initially empty, external, nonsymlink
  directory;
- both children in each run reproduced exact 19-member fixture and 89-member
  evidence mappings;
- both complete 108-member transports in each run were byte-identical;
- archive A and archive B in each run were byte-identical; and
- all four cache roots were absent before their top-level builders returned.

Each child cache was inspected after capture and before disposal. Every one
had exactly:

| post-child cache object | count |
| --- | ---: |
| nonsymlink package directories | 1 |
| nonsymlink regular `.nbi` files | 18 |
| nonsymlink regular `.nbc` files | 18 |
| all entries | 37 |
| symlinks | 0 |
| other files | 0 |

This is the atmosphere lane's accepted populated-cache contract; no
synthesis empty-after rule was substituted.

The two unrelated top-level candidates were equal as complete byte strings:

| reproduced property | run 1 | run 2 |
| --- | ---: | ---: |
| fixture arrays | 19 | 19 |
| evidence arrays | 89 | 89 |
| complete transport bytes | 33,147,771 | 33,147,771 |
| scientific array bytes | 357,984 | 357,984 |
| final archive members | 19 | 19 |
| final archive bytes | 363,050 | 363,050 |
| final archive SHA-256 | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |

The shared scientific identities were:

| identity | exact value |
| --- | --- |
| fixture mapping digest | `f533e3e327c879b1d367a89822bcd1847b15a73ba3d234a9c577c81997f75e0a` |
| evidence mapping digest | `f5e49f9f7f49c4604c08a53def65a9ee3b6ddff37376130eb274646ee375af4f` |
| complete transport SHA-256 | `0523ed254a78edaa07480bc30f23082f535e885f46d867de10f92cba1acd5b16` |
| fixture schema digest | `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698` |
| physical payload fingerprint | `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663` |
| full-capture schema digest | `cdf470038e67301b4c19b0691e672cd97df3233a3decf1b88e32ce3ac0dc1371` |
| full-capture fingerprint | `a25875097c7084ffe2577de65c2913d8775f29613823d9e6e0ab0d9db4644654` |

Origin, PID, cache-path, and process-evidence values differed as required and
remain outside candidate identity.

## R4. Independent scientific and role audit

The nineteen decoded members exactly matched the accepted lexical
`FIXTURE_SCHEMA`. Every member's shape, dtype, C-byte hash, and C-contiguous
layout matched the accepted worker. Their decoded `nbytes` sum was exactly
357,984.

The independent checks additionally reproduced:

- actual population slots `[0, 2, 840]`;
- selected Fe I population slot `350`, zero based;
- wavelength-bin sentinel \(2^{30}\);
- a strictly increasing 30,000-point opacity wavelength grid;
- finite values in every floating-point member; and
- exact schema and payload identities above.

The exact-member allowlist proves that the fixture remains input-only. It
contains no selected-line opacity output, full packed atmosphere state,
projected or placeholder output, transient worker evidence, process evidence,
raw subset copy, static table copy, synthesis state, comparison result, or
golden member.

`CAPTURE_SCHEMA_VERSION = 1` remains external provenance. There is no
capture-schema, archive-schema, `schema_version`, or other twentieth member.
All future metadata must continue to state:

```text
fixture_capture_schema_version           1
archive_contains_embedded_schema_version false
npy_member_format_version                "2.0"
```

## R5. Independent canonical-byte audit

An independent decoder, separate from the accepted writer decoder, required:

- immutable nonempty bytes below the one-MiB ceiling;
- exactly nineteen lexical, unique, safe `.npy` members;
- `ZIP_STORED`, fixed 1980 timestamp, Unix creator `3`, create/extract version
  `20`, flag bits and internal attributes zero, regular-file mode `0600`,
  empty extra fields and comments, no directories, no Zip64, and stored sizes;
- canonical NPY version 2.0 with `allow_pickle=False`;
- object-free C-contiguous arrays with exact shape, dtype, and C-byte hash;
  and
- whole-archive canonical re-encoding equal to the original byte string.

It accepted the candidate and independently re-encoded all 19 NPY members and
the complete ZIP container byte-for-byte.

Without consulting the accepted final archive hash as a rejection shortcut,
the independent validator rejected all 20 constructed mutation classes:

1. trailing archive bytes;
2. truncation;
3. byte corruption;
4. reversed member order;
5. removed member;
6. invented embedded schema-version member;
7. added atmosphere output member;
8. renamed member;
9. duplicate member;
10. changed timestamp;
11. changed permissions;
12. nonempty ZIP extra field;
13. alternate compression;
14. nonempty archive comment;
15. NPY version 1.0;
16. object-dtype NPY;
17. parent-path member;
18. valid-CRC one-ULP payload mutation;
19. valid-CRC shape mutation; and
20. valid-CRC dtype mutation.

The canonical candidate was the sole accepted control.

## R6. Immutability and no-publication proof

Each top-level run took independent before/after snapshots over 66 exact
control/source/data inputs, including the final contract and contract audit,
accepted writer chain, plan/worker/converter chain, all frozen pinned Python
sources, all pinned dynamic inputs, and all staged converter dependencies.
The snapshots were unchanged. The normalized source snapshot digest was:

```text
d5eb357a5dcba3f2f29ad4fd71ed6b6756ed9a9bd84e2ac823379b4c14c81455
```

The closed `data/` snapshot contained 51 descendants: 12 directories and 39
regular files, with no symlink or special node. It remained unchanged through
both builds and the accepted-chain test run. Under the audit's
path/type/mode/device/inode/size/hash encoding, its digest was:

```text
119f07eff726f7b77d344ac97168a9658258dcae2e7c4413718d4a868afa0ea0
```

`data/MANIFEST.json` remained unchanged at 37 entries and SHA-256
`d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a`.
The candidate destination remained absent with zero matching manifest
entries:

```text
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
```

The accepted builder summary was false for publication authorization,
fixture publication, golden publication, manifest mutation, and artifact-file
writing. All planned atmosphere publisher, publisher-test,
publisher-acceptance, detached-authorization, authorization-review, and
postpublication-audit paths remained absent.

No publisher, data, manifest, pinned source, paper, or external-source object
was written. The only runtime writes were inside four builder-owned external
Numba-cache roots, all of which were inspected and disposed before return.

## R7. Verification

The exact converter/worker/writer chain passed under all twelve accepted
process controls:

```text
41 passed in 116.88s
```

Ruff check passed for all six converter/worker/writer implementation and test
files. Ruff format check reported all six already formatted. `py_compile`
passed with bytecode writing disabled. Scoped `git diff --check` passed.

Final identity and no-publication checks again found:

- final contract `a663369c…` and audit `c4a4ca58…` unchanged;
- writer, tests, candidate, and writer audit unchanged;
- plan, worker, worker tests/audit, converter, converter tests/audit, and
  source contract unchanged;
- manifest still `d8f30e25…`;
- canonical fixture absent and unregistered; and
- every future publisher/authorization/postpublication object absent.

## R8. Final disposition

**ACCEPT exactly these candidate bytes under the final rebound contract:**

```text
eventual role, only if later separately authorized
  fixture

scientific purpose
  input-only 80-depth selected-line atmosphere integration fixture

members
  19

scientific array bytes
  357,984

archive bytes
  363,050

archive SHA-256
  1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff

scientific fixture schema digest
  f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698

physical payload fingerprint
  f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663

external capture schema version
  1

embedded schema-version member
  absent

NPY member version
  2.0

ZIP compression
  ZIP_STORED
```

**NOT WRITER ACCEPTANCE. NOT PUBLISHER ACCEPTANCE. NOT AUTHORIZATION. NO
FIXTURE, GOLDEN, DATA, MANIFEST, PINNED-SOURCE, PAPER, OR EXTERNAL WRITE IS
GRANTED. ALL FUTURE PUBLISHER AND AUTHORIZATION HASHES REMAIN UNRESOLVED.**

<!-- END DELIMITED FINAL-CONTRACT REBIND BYTE RE-AUDIT: 2026-07-30 -->

<!-- BEGIN DELIMITED PHYSICAL-FINGERPRINT CONTRACT ATMOSPHERE BYTE RE-AUDIT: 2026-07-30 -->

# Chapter 6 atmosphere candidate-byte physical-fingerprint repair re-audit

Date: 2026-07-30  
Reviewer role: independent candidate-byte reviewer; no implementation,
publisher, authorization, or publication authority  
Disposition: **ACCEPT the exact in-memory candidate bytes only**

## F1. Decision and preserved stale history

**ACCEPT** the exact 363,050-byte atmosphere fixture candidate at SHA-256

```text
1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff
```

against the exact corrected final publisher contract

```text
design/chapter06_lane_artifact_publisher_contract.md
3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b
```

and its final independent contract audit

```text
design/chapter06_lane_artifact_publisher_contract_rebind_independent_audit.md
fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc
```

This delimited append preserves the preceding 40,693 bytes and 1,043 lines
byte-for-byte at historical SHA-256
`cd9be49b2436fc34a33275c54369c549ad6be40aeba2f713bdd033f028156669`.
That history remains review evidence, but its latest acceptance section binds
the superseded contract `a663369c…` and is stale after the one-byte physical
fingerprint repair. Its still-earlier 28,077-byte prefix at
`79b20c065f5f1b588c6796d6413a1b2aa5e1e5a60056968fc7913cd961e12fc6`
binds contract `9ee0029f…` and is also historical only.

Neither stale contract hash is treated as authoritative here. The sole
contract authority for this byte decision is the exact pair `3a064f82…` and
`fe48eb57…` above.

This is candidate-byte acceptance only. It does not accept or create a
publisher, publisher tests, publisher review, detached authorization,
authorization review, canonical fixture, manifest entry, postpublication
audit, comparison golden, or cleanup record. Every future publisher,
authorization, and postpublication hash remains unresolved.

## F2. Exact rebound inputs

All rebound objects were regular nonsymlink files and independently matched:

| reviewed object | SHA-256 |
| --- | --- |
| corrected final publisher contract | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| final contract audit | `fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc` |
| deterministic atmosphere writer | `0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538` |
| focused writer tests | `c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a` |
| writer candidate record | `e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee` |
| accepted writer audit | `b946dcef0beeacf49a3da9ac036e21af7cd7b44d15092639cd3be744fb42f0f9` |

The accepted atmosphere scientific chain also remained exact:

| accepted precursor | SHA-256 |
| --- | --- |
| atmosphere fixture/oracle plan | `cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97` |
| atmosphere scientific worker | `21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531` |
| worker tests | `611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42` |
| worker candidate record | `da53e4846ee91f814c437994fff604ae91ed94a4da4db7856f8f47bb61cf72dc` |
| worker independent audit | `336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e` |
| accepted line converter | `4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb` |
| converter tests | `254d796b7ab761ca806c372d0bcdd935067ff1a89b2acfebcfa3007fe3f549dc` |
| converter candidate record | `acae959e9b01f986a553e9806fa7f60c6bb770ce3f356fab11c2d7509b63d03a` |
| converter independent audit | `60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75` |
| Chapter 6 exact source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |

The pinned Payne Zero checkout remained at
`9c44001feae40b85146630499e6f8a5fed42e5af`.

## F3. Two unrelated top-level constructions and four fresh children

Two unrelated top-level CPython interpreter processes each invoked exactly:

```python
build_deterministic_atmosphere_fixture_archive()
```

with zero arguments. Their parent PIDs were 63,325 and 63,431. Each accepted
writer invocation internally owned a fresh child A and fresh child B. The four
children were:

| construction | child | PID | origin-token SHA-256 | cache-path SHA-256 |
| --- | --- | ---: | --- | --- |
| run 1 | A | 63,326 | `36527edc879679e5d6afeb7b91b7a48fa6b8ffd4d78aeaeaba7d12203c611cd6` | `46b19d94863b405599131c1b660be21fec969024aca132f42c205ac3a1485359` |
| run 1 | B | 63,356 | `b3d1e7e76666c8e6531fb34c988e4dd1839d729afbd3d19e9d4653889bb65233` | `5ffa4f453c905f78190b85c6660ba3436d99abc80c59c1a27b4c230dba9fb41f` |
| run 2 | A | 63,432 | `e94dea0dcdd92aab200402006f8c1c3d5a220ced3464f84ffa62a4d519f8d787` | `6a1487c1e37f04338e6e136de044a87c7ddfb42dc670d9b637ac3b7770b4ab09` |
| run 2 | B | 63,441 | `bd21e9a440bb91befa7ff144799868d3aff93e92d8c5462ba4d9b5fe73b6228f` | `ba0f84cde97389fadb7c8568a9f4dda5d090eee594d8f26325eb54ed94840d1c` |

All four origin tokens, child PIDs, cache paths, and cache identities were
distinct. Every cache root was external to the textbook, Payne Zero, and
paper trees; existing, empty, and nonsymlink before launch; and absent before
its top-level builder returned.

Each child cache was independently inventoried after capture and before
disposal:

| post-child cache object | run 1 A | run 1 B | run 2 A | run 2 B |
| --- | ---: | ---: | ---: | ---: |
| nonsymlink package directories | 1 | 1 | 1 | 1 |
| nonsymlink regular `.nbi` files | 18 | 18 | 18 | 18 |
| nonsymlink regular `.nbc` files | 18 | 18 | 18 | 18 |
| all entries | 37 | 37 | 37 | 37 |
| symlinks | 0 | 0 | 0 | 0 |
| other files | 0 | 0 | 0 | 0 |
| root absent after return | yes | yes | yes | yes |

This is the accepted atmosphere populated-cache contract. The synthesis
lane's empty-after-child rule was not substituted.

Each child reproduced exactly 19 fixture arrays and 89 ephemeral-evidence
arrays. Within each top-level build, A/B mappings were bitwise equal and the
complete 108-array transport byte strings were identical. The two top-level
candidate archive byte strings were also identical:

| reproduced property | run 1 | run 2 |
| --- | ---: | ---: |
| fixture arrays per child | 19 | 19 |
| evidence arrays per child | 89 | 89 |
| complete transport bytes | 33,147,771 | 33,147,771 |
| complete transport SHA-256 | `0523ed254a78edaa07480bc30f23082f535e885f46d867de10f92cba1acd5b16` | `0523ed254a78edaa07480bc30f23082f535e885f46d867de10f92cba1acd5b16` |
| scientific array bytes | 357,984 | 357,984 |
| final archive members | 19 | 19 |
| final archive bytes | 363,050 | 363,050 |
| final archive SHA-256 | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |

The shared scientific identities were:

| identity | exact value |
| --- | --- |
| fixture mapping digest | `f533e3e327c879b1d367a89822bcd1847b15a73ba3d234a9c577c81997f75e0a` |
| evidence mapping digest | `f5e49f9f7f49c4604c08a53def65a9ee3b6ddff37376130eb274646ee375af4f` |
| fixture schema digest | `f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698` |
| physical payload fingerprint | `f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663` |
| full-capture schema digest | `cdf470038e67301b4c19b0691e672cd97df3233a3decf1b88e32ce3ac0dc1371` |
| full-capture fingerprint | `a25875097c7084ffe2577de65c2913d8775f29613823d9e6e0ab0d9db4644654` |
| selected-line output SHA-256 | `43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58` |

The two process-evidence digests intentionally differed
(`8630f1aa5f797c8623b9e58d7b358ef909c12c0ec33d4288ca4ac7000f10a9af`
and
`097a6855c4cdfa07220d461625e137c8eda7e38e5bcb1179159499c7d35abc40`).
Origin, PID, and cache evidence remains outside scientific identity and the
nineteen archive members.

## F4. Scientific, schema, and role locks

The decoded archive member set equals the accepted lexical `FIXTURE_SCHEMA`
exactly. Every member matched its exact accepted shape, dtype, C-byte hash,
C-contiguous layout, and finite-value requirement. Their decoded `nbytes`
sum was exactly 357,984.

The independent scientific checks reproduced:

- actual population slots `[0, 2, 840]`;
- selected Fe I population slot `350`, zero based;
- final wavelength-bin sentinel \(2^{30}\);
- a strictly increasing 30,000-point opacity wavelength grid;
- one selected line and the accepted seven compact selected-line fields;
- 80-depth temperature, \(hc/kT\), electron-density, population, and Doppler
  columns; and
- the exact continuum threshold shape `(80, 344)` and all member hashes.

The archive remains input-only. The exact allowlist contains no line-opacity
output, full packed atmosphere state, projected or placeholder output,
transient worker/process evidence, cache or origin evidence, raw subset copy,
static table copy, synthesis state, comparison result, or golden member.

`CAPTURE_SCHEMA_VERSION = 1` remains external transient-capture provenance.
There is no `schema_version`, capture-schema, archive-schema, or other
twentieth archive member. Future metadata must continue to distinguish:

```text
fixture_capture_schema_version           1
archive_contains_embedded_schema_version false
npy_member_format_version                "2.0"
```

The eventual role, only after a separate publisher and authorization, remains
`fixture`. No static, subset, synthesis-golden, or atmosphere-output role is
accepted by this byte gate.

## F5. Independent canonical-byte and mutation audit

An independent decoder, separate from the writer decoder, required:

- immutable nonempty bytes below the one-MiB ceiling;
- exactly nineteen lexical, unique, safe `.npy` member names;
- `ZIP_STORED`, fixed 1980 timestamp, Unix creator `3`, create/extract version
  `20`, zero flags and internal attributes, regular-file mode `0600`, empty
  member extras/comments and archive comment, no directories, no Zip64, and
  stored sizes;
- canonical NPY version 2.0 with `allow_pickle=False`;
- object-free C-contiguous arrays with exact shape, dtype, and C-byte hash;
  and
- complete canonical re-encoding equal to the original byte string.

It independently re-encoded every NPY member and the complete ZIP container
byte-for-byte.

Without using the accepted final archive hash as a rejection shortcut, the
validator rejected all twenty actual mutation classes:

1. trailing archive bytes;
2. truncation;
3. byte corruption;
4. reversed member order;
5. removed member;
6. invented embedded schema-version member;
7. added atmosphere output member;
8. renamed member;
9. duplicate member;
10. changed timestamp;
11. changed permissions;
12. nonempty ZIP extra field;
13. alternate compression;
14. nonempty archive comment;
15. NPY version 1.0;
16. object-dtype NPY;
17. parent-path member;
18. valid-CRC one-ULP payload mutation;
19. valid-CRC `(80,)` to `(40, 2)` shape mutation; and
20. valid-CRC dtype mutation.

One initial harness attempt at case 19 used `reshape(-1)`, which preserved the
already one-dimensional shape and was therefore correctly excluded as a
no-op, not reported as a rejection. The corrected `(80,)` to `(40, 2)`
temperature mutation was then constructed and rejected on the exact schema
lock. The canonical candidate was the sole accepted control.

## F6. Immutability and no-publication proof

Each accepted top-level run took before/after snapshots of the corrected
contract and final audit, the full accepted writer/worker/converter chain,
`data/MANIFEST.json`, and the complete closed `data/` tree. Both runs found
every snapshot unchanged.

The final `data/` inventory remained:

| property | exact value |
| --- | ---: |
| descendants | 51 |
| directories | 12 |
| regular files | 39 |
| symlinks | 0 |
| special nodes | 0 |

`data/MANIFEST.json` remained at 37 entries and SHA-256
`d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a`.
`data/README.md` remained
`1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b`.
The candidate destination remained absent and had zero matching manifest
entries:

```text
data/fixtures/chapter06_atmosphere_one_line_inputs.npz
```

The builder summaries remained false for publication authorization, fixture
publication, golden publication, manifest mutation, and artifact-file
writing. The planned atmosphere publisher, publisher test, publisher
acceptance, detached authorization, authorization review, and
postpublication-audit paths all remained absent.

No publisher, fixture, golden, data, manifest, Payne Zero, paper, or external
source object was written. Runtime scientific writes were confined to
builder-owned external Numba cache roots, all inspected and disposed before
return. The only trust-object change is this delimited audit append.

## F7. Verification

The exact converter/worker/writer chain passed under all twelve accepted
process controls:

```text
41 passed in 70.22s
```

Ruff check passed for all six converter/worker/writer implementation and test
files. Ruff format check reported all six already formatted. `py_compile`
passed with bytecode writing disabled. Scoped `git diff --check` passed.

Final identity and no-publication checks again found:

- corrected contract `3a064f82…` and final audit `fe48eb57…` unchanged;
- writer, tests, candidate, and writer audit unchanged;
- plan, worker, worker tests/candidate/audit, converter, converter
  tests/candidate/audit, and source contract unchanged;
- manifest still `d8f30e25…`;
- canonical fixture absent and unregistered; and
- every future publisher/authorization/postpublication object absent.

## F8. Final disposition

**ACCEPT exactly these candidate bytes under the corrected
physical-fingerprint contract:**

```text
eventual role, only if later separately authorized
  fixture

scientific purpose
  input-only 80-depth selected-line atmosphere integration fixture

members
  19

scientific array bytes
  357,984

archive bytes
  363,050

archive SHA-256
  1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff

scientific fixture schema digest
  f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698

physical payload fingerprint
  f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663

external capture schema version
  1

embedded schema-version member
  absent

NPY member version
  2.0

ZIP compression
  ZIP_STORED
```

**NOT WRITER ACCEPTANCE. NOT PUBLISHER ACCEPTANCE. NOT AUTHORIZATION. NO
FIXTURE, GOLDEN, DATA, MANIFEST, PINNED-SOURCE, PAPER, OR EXTERNAL WRITE IS
GRANTED. ALL FUTURE PUBLISHER, AUTHORIZATION, AND POSTPUBLICATION HASHES
REMAIN UNRESOLVED.**

Any future publisher-contract change invalidates this exact byte acceptance.

<!-- END DELIMITED PHYSICAL-FINGERPRINT CONTRACT ATMOSPHERE BYTE RE-AUDIT: 2026-07-30 -->
