# Chapter 6 synthesis scientific-worker candidate

Status: **scientific worker candidate complete; independent acceptance pending**  
Scope: six-depth synthesis lane only  
Pinned Payne Zero commit:
`9c44001feae40b85146630499e6f8a5fed42e5af`

This report reviews
`scripts/chapter06_synthesis_oracle_worker.py`. It does not authorize a raw
capture, publisher, manifest entry, or golden artifact.

## 1. Candidate disposition

**ACCEPT for independent worker review.**

The worker is an in-memory observer with no output-path, serialization,
publication, or golden-read interface. It executes the canonical one-line
ordinary Fe I synthesis route in two grid configurations and four six-depth
regimes. Every returned value is a detached, object-free NumPy array.

The worker deliberately leaves
`ACCEPTED_CAPTURE_KEY_COUNT` and
`ACCEPTED_CAPTURE_SCHEMA_DIGEST` unset. Therefore
`meta__capture_scope_complete` remains false even though the candidate
capture is internally exhaustive. Those constants may be frozen only after
an independent reviewer accepts this inventory and repeats the fresh-process
comparison.

## 2. Exact implementation identity

| artifact | SHA-256 |
| --- | --- |
| scientific worker | `e127521d61ff185cfc12cfc28b9a13639b96345e86a1b44453460e2f849d9281` |
| focused tests | `536d3bbd3b99ba558021feb33b7b06b0365417490def8e9b56eddc792f18e452` |
| synthesis fixture/oracle plan | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| exact source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |

The candidate inputs were:

| role | path | SHA-256 |
| --- | --- | --- |
| input-only state fixture | `data/fixtures/chapter05_continuum_states.npz` | `ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae` |
| immutable raw line subset | `data/subsets/chapter06_fe_i_source_row_873702.npz` | `bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04` |
| synthesis continuum tables | `data/static/synthesis_tables/continuum_tables.npz` | `406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d` |
| synthesis edge geometry | `data/static/synthesis_tables/continuum_edge_grid.npz` | `11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e` |
| synthesis line-profile tables | `data/static/synthesis_tables/line_profile_tables.npz` | `87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4` |

The worker also verifies:

- the external checkout's exact Git commit;
- all 17 transitively loaded synthesis Python files and their hashes;
- byte identity for the five staged files actually executed by this route:
  `atomic_lines.py`, `constants.py`, `continuum.py`, `device.py`, and
  `line_opacity.py`;
- staged/upstream byte identity for all three static table archives;
- the full source archive's size and
  `4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7`
  hash;
- the fixture's 217-member schema and
  `4bcf0bbd8d61e58334c4c7ef6caaaf9ca47e6fb4536ad0098d5a541d540ec048`
  physical-payload digest;
- the subset's 27-member schema, 17 exact raw dtypes, row index, and source
  provenance.

All file, process, and cache checks occur before the pinned package is
imported.

The 258 MB full archive is required only by this external-reference oracle as
fail-closed provenance evidence. It is not a reader/runtime dependency:
ordinary chapter code and notebook tests use only the canonical 8,665-byte
subset.

## 3. Candidate result inventory

The in-memory mapping has **746 keys** and **7,837,002 array bytes**. Its
candidate identities are:

| identity | value |
| --- | --- |
| schema version | `1` |
| schema digest | `529ce2377adde54696f46177387cf38b804ac1164e322498c4db1a0cea86c688` |
| physical-payload fingerprint | `805b2a1b8922d14ff7fa4164ead302203c6910fe254b9bfbe6421ca837b14fe4` |
| full-capture fingerprint | `955514c5eea78706661089cce1880ff3fb843d9e08da1a591197c8aed5a16c8a` |

Logical family counts are:

| prefix | members | content |
| --- | ---: | --- |
| `meta__` | 57 | process, source-set, contract, dtype, and scope identities |
| `identity__` | 27 | pinned code, source archive, and table hashes |
| `fixture_payload__` | 2 | exact field inventory and payload digest |
| `subset_payload__` | 2 | exact field inventory and payload digest |
| `record__` | 14 | 13 derived per-line fields plus their names |
| `catalog__` | 18 | exact 13-field line mapping plus five support entries |
| `grid__` | 8 | two wavelength axes, hashes, and center/wing indices |
| `invariant__` | 86 | all 43 `AtomicInvariants` fields on both grids |
| each regime prefix | 133 | state views, two continuum routes, four line calls per grid, and ledgers |

Representative contracts are:

| member | shape | dtype |
| --- | --- | --- |
| `grid__canonical__wavelength_nm` | `(6000,)` | `float64` |
| `catalog__helium_line_type` | `(0,)` | `int64` |
| `invariant__canonical__metal_classical_strength` | `(1,)` | `float32` |
| `invariant__canonical__auto_catalog_index` | `(0,)` | `int64` |
| `invariant__canonical__helium_line_type` | `(0,)` | `int64` |
| `{regime}__canonical__continuum_total_float64` | `(6,6000)` | `float64` |
| `{regime}__canonical__continuum_line_input_float32` | `(6,6000)` | `float32` |
| `{regime}__canonical__gross_batched_float32` | `(6,6000)` | `float32` |
| `{regime}__canonical__net_batched_float32` | `(6,6000)` | `float32` |
| `{regime}__canonical__ledger__wing_reach` | `(6,)` | `int64` |

The raw candidate intentionally retains full continuum, loop, batched, and
coarse-grid arrays because it is an audit result, not the future compact
comparison-only golden.

## 4. Scientific observations

The canonical grid is the exact 495–505 nm,
`R_grid = 300000` grid:

- 6000 `float64` samples;
- mapped center and wing-anchor index `2434`;
- mapped center wavelength `499.0333758196059` nm.

The robustness grid uses the same interval at `R_grid = 20000`:

- 400 `float64` samples;
- mapped center and wing-anchor index `162`;
- mapped center wavelength `499.04420746429804` nm.

Both grids reproduce the same depth activity:

```text
hot_dwarf            1 1 1 0 0 0
solar_dwarf          1 1 1 1 1 1
low_gravity_giant    1 1 1 1 1 1
cool_molecule_rich   1 1 1 1 1 1
```

Across active canonical layers, exact reach spans 5–163 samples. The maximum
gross and net canonical cell opacities in this candidate are respectively
`4.732271671295166` and `4.676202774047852` cm\(^2\) g\(^{-1}\).

For every regime and both grids:

- continuum absorption and scattering are recomputed from the exact
  18-field Chapter 5 view in CPU Torch `float64`;
- the line cutoff receives the exact `float32` conversion;
- ordinary invariant count is one;
- all 11 `auto_*` arrays and all ten helium arrays have their exact
  zero-length dtypes;
- gross and net slabs are finite, nonnegative CPU Torch `float32`;
- net equals one exact wavelength-dependent `float32` multiplication of
  gross;
- loop and batched gross slabs are bitwise equal;
- loop and batched net slabs are bitwise equal;
- activity, center opacity, reach, and nonzero-cell counts reconstruct one
  another;
- the first-below-cutoff reach convention is retained;
- isolated electron-density and neutral-collision perturbations leave the
  nonowned damping components unchanged.

The largest measured loop/batched absolute difference is exactly `0.0`.

## 5. Fresh-process determinism

The focused integration test launched two unrelated empty-cache processes
with:

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

Both candidates returned identical summaries, schema digests, physical
fingerprints, and full fingerprints. The unrelated cache pathname is not
stored in the result; only the verified cache policy is recorded.

Measured host:

| field | value |
| --- | --- |
| platform | `macOS-26.5.2-arm64-arm-64bit-Mach-O` |
| architecture | `arm64` |
| Python | `3.13.9` |
| NumPy | `2.3.5` |
| Numba | `0.62.1` |
| Torch | `2.11.0` |

## 6. Negative-boundary evidence

Focused tests prove:

- a Chapter 5 or Chapter 6 golden path is rejected before `np.load`;
- the worker API exposes no output, destination, publish, golden, or
  serialization parameter;
- CLI `--output` and `--publish` are unrecognized;
- populated and symlinked cache paths fail closed;
- alternate fixture and subset paths are rejected;
- object-dtype results are rejected;
- scalar metadata remain zero-dimensional after detachment;
- the worker reports both
  `meta__golden_read_performed=False` and
  `meta__golden_publication_performed=False`.

No `.npz`, raw capture, golden, or external file was created or modified by
the worker.

## 7. Verification status

Completed:

- focused worker suite: **8 passed**;
- repository-wide documented suite: **356 passed, 1 skipped**;
- pinned-source verifier: **pass**;
- targeted Ruff check and format check: **pass**;
- Python compilation: **pass**;
- targeted diff check: **pass**.

The first repository-wide command omitted the documented `PYTHONPATH=src:.`
and therefore collected the external checkout directly; five modules failed
during collection at the external Numba cache decorator. This was an invalid
test invocation, not a worker result.

After the separately owned `book/chapter06_runtime.py` repair settled, the
correct repository-wide command completed with 356 passing tests and one
intentional skip.

A full-tree Ruff audit was also attempted. It remains outside the worker
boundary because the pre-existing tree contains one unrelated unused import
in `scripts/chapter03_oracle_worker.py` and 48 unrelated files that do not
match the current Ruff formatter. Both newly added Python files pass Ruff
check and format verification without exclusions.

## 8. Remaining gates

The following remain intentionally open:

1. an independent reviewer must repeat the fresh-process captures and accept
   the 746-member schema;
2. only that review may freeze the accepted key count and schema digest in the
   worker;
3. CUDA and MPS observations and their separately measured tolerances remain
   future backend work;
4. the compact reader artifact, raw-to-final ownership review, deterministic
   writer, detached authorization, publisher, and manifest entry remain
   unimplemented;
5. no golden may be published from this candidate report.

There is no scientific ambiguity in the accepted CPU record, grids, state
view, empty routes, cutoff lifecycle, damping ledger, reach, stimulation, or
loop/batched result. Publication authority is the only deliberate boundary
still withheld.
