# Chapter 6 synthesis compact comparison candidate

Status: **in-memory candidate complete; independent candidate review pending**  
Scope: synthesis-lane raw-to-comparison projection only  
Audited worker commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

This report describes:

- `scripts/chapter06_synthesis_compact_assembler.py`;
- `tests/test_chapter06_synthesis_compact_assembler.py`.

It does not authorize or perform serialization, file creation, manifest
mutation, detached authorization, or publication. No NPZ, raw-capture
archive, candidate file, or golden file was created.

## 1. Candidate disposition

**ACCEPT for independent compact-candidate review.**

The assembler consumes only the already accepted 754-member in-memory result
from `scripts/chapter06_synthesis_oracle_worker.py`. It first verifies the
frozen raw count, schema, scope, worker hash, physical fingerprint, full
fingerprint, lifecycle flags, and complete worker science checks. It then:

1. proves every loop slab is bit-identical to its batched slab;
2. proves every net slab is exactly one `float32` multiplication of gross by
   the complete stimulated-emission factor;
3. proves continuum absorption plus scattering reconstructs the `float64`
   total and its `float32` line-input fence;
4. proves center and wing continuum samples come from the exact cutoff slab;
5. proves catalog, canonical invariant, coarse invariant, Harris, FASTEX, and
   wavelength duplicates before discarding them;
6. assigns every raw member exactly one reviewed disposition;
7. returns a detached lexical object-free mapping, an exhaustive schema
   description, and the raw ownership ledger.

The candidate remains deliberately unfrozen. Its member set and identities
must receive independent review before a deterministic writer or publisher
may consume it.

## 2. Exact implementation identities

| artifact | SHA-256 |
| --- | --- |
| compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` |
| compact focused tests | `53111433aa3082a58be5ec8b3da1a961f330eac1bb805ac26d1fc6625487e42d` |
| accepted scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| synthesis fixture/oracle plan | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| final worker audit | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |

The accepted raw boundary is:

| property | value |
| --- | --- |
| key count | `754` |
| schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| full fingerprint | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` |
| scope complete | `True` |

## 3. Proposed compact mapping

Two independent fresh processes produced exactly:

| candidate property | value |
| --- | --- |
| mapping members | `213` |
| array bytes | `1,235,275` |
| schema version | `1` |
| schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| payload fingerprint | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` |
| raw ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| publication authorized | `False` |
| golden publication performed | `False` |

The mapping is about 29.5% of the 4 MiB ceiling. Its family inventory is:

| family | members | role |
| --- | ---: | --- |
| `axis__` | 3 | one regime axis, one depth axis, one canonical wavelength axis |
| `mapping__` | 15 | 13 physical line fields plus two empty-helium support fields |
| `support__` | 6 | three Harris member hashes, their archive identity, and two FASTEX hashes |
| `invariant__` | 37 | 13 ordinary fields, 11 empty auto fields, ten empty helium fields, and three scalars |
| `ledger__` | 17 | comparison-only factor and branch values |
| `activity__` | 1 | canonical post-cutoff mask |
| `continuum__` | 10 | center/wing samples and canonical/coarse component digests |
| `opacity__` | 2 | the only dense gross and net slabs |
| `coarse__` | 15 | compact grid, mask, peak, reach, and count evidence |
| `stimulation__` | 6 | canonical/coarse factor hashes and exact reconstruction flags |
| `grid__` | 3 | canonical grid identity and mapped indices |
| `identity__` | 27 | pinned source, source-archive, and static-table identities |
| `meta__` | 71 | accepted raw, environment, schema, ownership, and lifecycle metadata |

Every proposed member has a separate `CompactMemberSpec` recording:

- exact name;
- NumPy dtype string;
- exact shape;
- ordered axes;
- physical unit or convention;
- role `axis` or `comparison-only`.

The candidate mapping itself contains only NumPy arrays and scalar NumPy
strings/numbers. No object dtype is permitted.

## 4. Exact Section 11 ownership

The proposed mapping owns:

- one `(6000,) float64` canonical wavelength axis;
- four regime names and six depth indices;
- the exact 13-field one-line mapping;
- empty `helium_line_type`, the exact helium cutoff, and H0/H1/H2 member
  identities instead of table copies;
- all 13 one-element ordinary invariant fields;
- all 11 exact zero-length autoionizing fields;
- all ten exact zero-length helium fields;
- the canonical invariant count, grid resolution, and helium cutoff;
- the 17 retained `(4,6)` factor/branch ledger arrays;
- canonical continuum center and wing samples;
- one `(4,6,6000) float32` gross slab;
- one `(4,6,6000) float32` net slab;
- canonical activity, reach, and nonzero-count evidence;
- coarse grid geometry and identity, activity, gross/net peaks, and active
  reach/count extrema;
- canonical and coarse continuum-component digests;
- canonical and coarse stimulated-factor digests and exact batched/loop
  reconstruction flags;
- source, fixture, subset, static-table, environment, worker, raw-schema,
  fingerprint, and logical-ownership metadata.

The proposed mapping does not own:

- any Chapter 5 fixture or selected state array;
- any `continuum_state` or `line_state` member;
- a full continuum slab;
- a full stimulated-emission slab;
- a Harris or FASTEX numerical table;
- loop duplicates;
- a coarse opacity slab or coarse wavelength axis;
- raw source-catalog arrays;
- any atmosphere, hydrogen, helium-line, autoionizing, molecular, forest, or
  many-line product.

The validator rejects input-like shapes `(6,6,139)`, full coarse shapes
`(6,400)` and `(4,6,400)`, and table lengths 1001 or 2001. It also requires
that the only `(4,6,6000)` arrays are:

```text
opacity__gross_float32
opacity__net_float32
```

## 5. Exhaustive raw ownership

All 754 raw members have one and only one disposition:

| disposition | count | meaning |
| --- | ---: | --- |
| `final` | 250 | copied, stacked, or reduced into a named comparison member |
| `derived_digest_only` | 123 | discarded only after equality, reconstruction, or digest evidence |
| `intentionally_ephemeral` | 381 | upstream input state or detailed audit intermediate deliberately excluded |

The ownership tuple is lexical and unique. Every `final` entry resolves to an
existing comparison member. The digest binds raw name, disposition, target,
and reason for all 754 entries.

Important reductions include:

- 16 loop slabs → exact equality proofs against the retained batched routes;
- eight complete stimulated factors → eight hashes plus exact reconstruction
  flags and retained center factors;
- 32 continuum component slabs → component hashes after sum/fence/sample
  reconstruction;
- two Harris table copies per grid plus catalog copies → three exact member
  hashes;
- two FASTEX table copies per grid → two exact member hashes;
- four coarse gross and four coarse net batched slabs → per-regime peaks;
- coarse activity/reach/count matrices → one mask and per-regime active
  extrema;
- 100 upstream continuum/line-state arrays → explicitly ephemeral, never
  copied.

## 6. Mutation and failure evidence

The fresh-process mutation harness proves rejection of:

1. a removed raw member;
2. an extra raw member;
3. an unaccepted raw scope;
4. a raw physical-payload mutation;
5. a stimulated-factor mutation that breaks net reconstruction;
6. a loop-slab mutation that breaks batched/loop equality;
7. a continuum mutation that breaks absorption-plus-scattering reconstruction;
8. a Harris mutation that breaks catalog/invariant/table deduplication;
9. a compact payload mutation;
10. a compact schema-description mutation;
11. a compact ownership-ledger mutation.

Assembling twice from one accepted raw mapping produces equal lexical keys,
schemas, ownership tuples, and every candidate array. Mutating the raw mapping
after assembly does not change the detached candidate.

## 7. Verification status

Completed:

```text
python -m pytest -q tests/test_chapter06_synthesis_compact_assembler.py
4 passed in 4.74s
```

Combined accepted-worker and compact suites:

```text
python -m pytest -q \
  tests/test_chapter06_synthesis_oracle_worker.py \
  tests/test_chapter06_synthesis_compact_assembler.py
13 passed in 8.03s
```

Repository-wide documented suite:

```text
PYTHONPATH=src:. python -m pytest -q
376 passed, 1 skipped in 34.74s
```

Targeted checks:

```text
ruff check scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
All checks passed!

ruff format --check scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
2 files already formatted

python -m py_compile scripts/chapter06_synthesis_compact_assembler.py \
  tests/test_chapter06_synthesis_compact_assembler.py
pass
```

Two additional unrelated fresh-cache processes returned identical candidate
summaries and the identities in Section 3. Their supplied cache directories
remained empty.

## 8. Remaining gates

The following remain explicitly unauthorized and unimplemented:

1. independent review of the 213-member candidate schema, every member spec,
   and the 754-member ownership classification;
2. freezing candidate count, schema digest, payload fingerprint, ownership
   digest, assembler hash, and test hash;
3. independent deterministic raw and final A/B byte serialization;
4. raw-to-final reconstruction from independently accepted captures;
5. deterministic NPZ writing;
6. a separate compact-candidate acceptance record;
7. detached publication authorization;
8. atomic first publication;
9. exhaustive manifest registration;
10. CUDA/MPS measurement and separately reviewed tolerances.

No later gate may infer authority from this candidate report.
