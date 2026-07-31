# Chapter 2 Data-Role and Parity Harness Plan

Status: design only. This plan does not authorize edits to the canonical
chapter, progressive source packages, the pinned Payne Zero checkout, or the
paper tree.

Authority: `BIBLE.md`, `COVERAGE.md`, and
`design/global_chapter_contracts.md`. Where the older Part I brief proposes
official-site figure reuse, a detached exercise set, or a combined
"verification data" category, the current Bible and global contract
supersede it: all schematics are original, useful checks appear in the main
causal text, and static input, teaching subset, integration fixture, and
golden output are four physically separate roles.

Pinned implementation:

```text
Payne Zero commit
9c44001feae40b85146630499e6f8a5fed42e5af
```

## 1. The chapter's governing claim

Chapter 2 asks how one depth integral can remain scientifically identifiable
when it is expressed as a readable NumPy loop, a compiled CPU kernel, a
chunk-parallel transfer calculation, and a Torch device scan.

The data and parity harness must reinforce one sentence:

> A number is usable only when its physical meaning, units, axes, numerical
> policy, input identity, and validation claim are explicit.

This chapter owns representation and evidence, not the later physics that
fills the arrays. It may name every schema-v4 field and show the distinction
between actual and partition-normalized populations, but it must not derive
partition functions, Saha ionization, electron closure, molecular
equilibrium, continuum edges, or the integrated radiation moments. Those
belong to Chapters 3–5 and 12.

The harness therefore has three deliberately narrow jobs:

1. preserve the exact contracts of the depth-integration implementations;
2. make every local numerical asset's role and byte identity auditable;
3. prove that the self-contained textbook result can be compared with a
   pinned-source oracle without importing the external source at reader or
   test runtime.

## 2. Current repository finding

There is no active `data/` hierarchy yet. The repository instead contains a
legacy `reference/` directory with 53 files, including 41 NPZ archives,
occupying about 369 MiB (`377772` KiB). Only the Lecture 14–16 portion has a
human-written role manifest. Several archives mix static inputs, computed
upstream state, and comparison targets. The old `_pipeline/` scripts still
read and sometimes generate these files directly.

That legacy collection is evidence to audit, not a foundation to rename in
bulk. A filename such as `*_ref.npz`, `diag.npz`, or `L6.npz` does not reveal
whether an array is an input, fixture, or answer. Moving such a file under
`data/golden/` would preserve the ambiguity under a cleaner directory name.

The new Chapter 2 path should begin with new, minimal assets whose roles are
known at construction. Legacy files may be deleted only after every active
reader/build/test dependency has been replaced and their unique scientific
content has either been regenerated with provenance or explicitly retired.

## 3. Target data layout

Use the exact four roles in the global contract:

```text
data/
  MANIFEST.json
  static/
    schemas/
      atmosphere_schema_v4.json
  subsets/
    ch02_catalog_identity/
      observed_atomic_lines_first_records.npy
      CHECKSUMS.sha256
  fixtures/
    ch02/
      depth_integral_inputs.npz
      transfer_accumulation_inputs.npz
      synthetic_schema_v4.npz
  golden/
    payne_zero/
      ch02_depth_integral_outputs.npz
      ch02_transfer_accumulation_outputs.npz
```

This is a target, not a requirement to create an asset before it has a real
consumer. In particular, omit `transfer_accumulation_inputs.npz` and its
golden until the exact parallel accumulator can be exercised without copying
an unreadable block into the chapter. Empty placeholder files and speculative
fixtures are forbidden.

### 3.1 `data/static`

Static assets are immutable physical tables, declarative schemas, or exact
small checkpoint inputs. For Chapter 2 the only necessary static asset is the
exact schema-v4 JSON copied from the pinned
`payne_zero_synthesis/atmosphere_schema.json`.

It is a byte-identity checkpoint and a human-readable field inventory. The
textbook's executable validator remains the exact Python implementation;
reading the JSON does not silently create a second validator.

### 3.2 `data/subsets`

A teaching subset is a deterministic, order-preserving slice of a larger
static source catalog. It is not a fixture and it is not a result.

The Chapter 2 subset exists only to make parent identity, selection, and
checksum failure concrete. A suitable candidate is a very small leading slice
of `observed_atomic_lines.npy`, stored with its original structured dtype and
record order. Chapter 2 treats the records as catalog bytes; it does not teach
line-field physics, which remains Chapter 7's responsibility.

Its manifest entry must state:

- parent source-relative path;
- parent SHA-256 from `source_catalogs/CHECKSUMS.sha256`;
- exact selection rule such as `slice(0, N, 1)`;
- order-preservation statement;
- subset dtype, shape, byte size, and SHA-256;
- the generator command and pinned commit;
- `requires_optional_full_catalog_for_regeneration: true`.

The checked-in subset itself is sufficient for the chapter. Only regeneration
needs the optional 6.8 GiB source-catalog tree.

`CHECKSUMS.sha256` contains the subset's relative path and digest in the exact
format consumed by `load_source_catalog_checksums`. The chapter corrupts a
temporary copy, never the committed file.

### 3.3 `data/fixtures`

An integration fixture is supplied input. It may influence the
reader-computed result, so its upstream boundary must be visible before the
file is opened.

- `depth_integral_inputs.npz` contains only a small nonuniform depth grid,
  positive depth-dependent values, batched wavelength rows, and surface
  values used by the exact integration contracts.
- `transfer_accumulation_inputs.npz`, if retained, contains the minimum
  complete prepared state needed to compare the exact serial and
  chunk-parallel accumulator. Its manifest names Chapter 12 as the physical
  owner of the radiation/operator state. Chapter 2 uses it only to expose
  independent frequency chunks, private buffers, and fixed-order reduction.
- `synthetic_schema_v4.npz` is explicitly a representation-test fixture, not
  a physical atmosphere and not an initializer prediction.

No fixture contains an expected result. No fixture is called "ground truth."

### 3.4 `data/golden/payne_zero`

A golden is output from the pinned implementation. It is a comparison target
and must never supply an input to the reader-built calculation.

The Chapter 2 goldens contain output arrays only. Their manifest records the
fixture hash and deterministic case identifier that produced them. The
chapter follows this visible order:

```text
load declared fixture
→ compute with textbook implementation
→ run analytic and representation checks
→ only then open the golden
→ compare under the declared policy
```

Do not put input grids, constants, operator tables, or a reusable atmosphere
inside a golden archive "for convenience." That makes it too easy for an
answer to leak into the computed path.

## 4. Minimal root manifest schema

Use one root `data/MANIFEST.json` rather than a second undocumented sidecar
format for every file. The manifest itself is trusted through the repository
commit and is not listed inside itself; its SHA-256 is reported in the
verification ledger.

The smallest adequate top-level form is:

```json
{
  "schema": 1,
  "payne_zero_commit": "9c44001feae40b85146630499e6f8a5fed42e5af",
  "paper_main_tex_sha256": "e11507b9150550b246f6664debf22e540aa92d8261eb40daabb594da91bd8e0d",
  "paper_pdf_sha256": "5c3585794d31ac649eac13851d3a5a038d67a3f213b3218715f520e65a81b8ab",
  "file_count": 0,
  "total_bytes": 0,
  "files": []
}
```

Every file record requires:

```json
{
  "id": "stable-kebab-case-id",
  "path": "fixtures/ch02/example.npz",
  "role": "fixture",
  "media_type": "application/x-npz",
  "bytes": 1234,
  "sha256": "64 lowercase hexadecimal characters",
  "book_use": ["chapter_02"],
  "source": {
    "kind": "book_generated",
    "paths": ["source-relative/or/repository-relative/path"],
    "sha256": ["corresponding source digest"]
  },
  "generation": {
    "command": "one reproducible repository-relative command",
    "generator": "scripts/...py",
    "requires_optional_full_catalog": false
  },
  "arrays": [
    {
      "name": "column_mass",
      "shape": [4],
      "dtype": "float64",
      "unit": "g cm^-2",
      "axes": ["depth_outer_to_inner"]
    }
  ]
}
```

The four legal role strings are exactly:

```text
static
subset
fixture
golden
```

Directory and declared role must agree. The common record is intentionally
small; role-specific information is carried in one conditional object:

| Role | Required conditional object |
| --- | --- |
| `static` | `identity`: copy mode and pinned source-relative path |
| `subset` | `selection`: parent path/hash, deterministic selector, order policy |
| `fixture` | `upstream_boundary`: synthetic or computed, physical owner, may influence result |
| `golden` | `oracle`: comparison-only flag, input asset hashes, exact/tolerance policy, environment |

For NPY/NPZ, `arrays` is nonempty and lists every stored array. For JSON or
plain text it is an empty list, and `media_type` plus `source` describe the
content. Units use exact physical strings or `dimensionless`; never use an
empty string for a numeric quantity.

The verifier checks:

- schema version and required keys;
- safe normalized relative paths, with no absolute path or `..`;
- unique `id` and `path`;
- exact role/directory agreement;
- exact file count and byte total;
- 64-character lowercase SHA-256;
- actual byte size and digest;
- NPY/NPZ key set, shape, and dtype against `arrays`;
- no object dtype and `allow_pickle=False`;
- required role-specific metadata;
- every golden has `comparison_only: true`;
- every subset points to a parent checksum and selection;
- every fixture names its upstream boundary;
- no asset appears in more than one role.

The manifest records identity; it does not certify the physics represented by
the bytes. An NPZ array does not carry a physical-unit type, so verification
can require the exact declared unit string from the pinned schema/contract but
cannot infer from the bytes that a pressure is truly in dyn cm\(^{-2}\).

## 5. Checksum path taught in the chapter

Keep two different identities visibly separate:

1. `data/MANIFEST.json` is the textbook asset inventory.
2. `CHECKSUMS.sha256` is the exact source-catalog digest mapping parsed by
   `payne_zero_atmosphere.source_catalogs.load_source_catalog_checksums`.

Do not call them both "the manifest" in prose.

The bite-size chapter sequence is:

1. display one subset path, byte count, and expected SHA-256;
2. hash it in 8 MiB blocks to explain why large files need not be loaded into
   memory;
3. call the exact source-catalog checksum loader/verifier on the small subset
   tree;
4. copy the tree to a temporary directory and change one byte;
5. predict, observe, and interpret the checksum mismatch;
6. restore nothing because the committed asset was never touched.

The negative tests also cover a malformed digest, a duplicate path, an
absolute/path-traversal member, and a missing member. These are generated in a
temporary directory rather than checked in as four bad manifests.

The full 6.8 GiB catalog verification remains an explicit appendix/install
command. It must not run while rendering Chapter 2.

## 6. Compact schema-v4 fixture

Use one compact archive with:

```text
D = 4 depth layers
Q = 3 continuum-edge entries
dtype = float64 for physical arrays
atmosphere_schema_version = int32[1] with value 4
```

Four layers are enough to expose outer-to-inner ordering, a monotonic column
mass, and a nontrivial population depth axis while keeping the three
`(D,6,139)` cubes small.

The archive includes exactly the 25 names in
`REQUIRED_ATMOSPHERE_ARRAYS`, plus the optional typed product-metadata
extension only if Chapter 2 actually demonstrates
`load_atmosphere_product_metadata`.

Required construction rules:

- `temperature`, `gas_pressure`, `electron_density`, `mass_density`,
  `column_mass`, and `hc_over_kt` are positive.
- `column_mass` is strictly increasing with index.
- `microturbulence` and all population/width fields are nonnegative.
- `elemental_abundances` is positive with exact shape `(99,)`.
- `ion_stage_populations`,
  `partition_normalized_populations`, and
  `fractional_doppler_widths` have exact shape `(4,6,139)`.
- The two named H/C arrays have exact shape `(4,2)`.
- The four continuum edge arrays have shapes `(3,)`, `(3,)`, `(2,)`, and
  `(2,)`; signed frequencies are nonzero and wavelength-like fields are
  positive.
- Selected sentinel slots use a declared synthetic partition function, for
  example \(n=10^{12}\) and \(U=4\), so the test can independently assert
  \(n/U=2.5\times10^{11}\).
- Actual and partition-normalized cubes are numerically different. Equal
  all-zero placeholder cubes would pass shape validation but fail the
  chapter's semantic purpose.
- The sentinel relation is tested outside the production validator. The exact
  validator is not modified to pretend it can establish Saha/Boltzmann
  physics.

If metadata is included, it is complete and internally agreeing:

```text
atmosphere_product_metadata_schema = 1
atmosphere_product_role = "learned_initializer_prediction"
atmosphere_converged = False
atmosphere_closure_required = True
```

The caption and manifest still call the archive synthetic representation-test
data. The exact role string demonstrates product safety metadata; it does not
claim that a network generated these numerical arrays.

All corrupt-schema cases are made in temporary files by changing one feature
at a time:

- missing required canonical field;
- schema-4 archive containing a legacy alias;
- reversed or repeated `column_mass`;
- wrong `(D,6,139)` population shape;
- nonfinite or negative value in a constrained field;
- inconsistent `Q`/`Q-1` edge length;
- incomplete typed metadata;
- typed metadata disagreeing with `atmosphere_metadata_json`.

This avoids a directory of nearly identical invalid NPZ files.

## 7. Pinned-source oracle generation

Oracle generation is a development workflow, never a reader dependency.

### 7.1 Isolation

Create a development-only command such as:

```text
python scripts/oracles/generate_chapter02_oracles.py \
  --source-root /path/to/pinned/payne-zero \
  --output-dir /temporary/staging/directory
```

The generator:

1. resolves the supplied source root read-only;
2. checks `git rev-parse HEAD` against the pinned commit;
3. hashes every source module and data file used;
4. validates the optional full-data manifest only when a requested oracle
   needs it;
5. launches an isolated subprocess whose import path points to the pinned
   checkout, avoiding collision with the textbook's same-named progressive
   packages;
6. reads inputs from committed textbook fixtures or recreates a deterministic
   analytic case from a versioned case specification;
7. writes outputs to a temporary staging directory, never directly over a
   committed golden;
8. validates names, shapes, dtypes, finiteness, and expected monotonicity;
9. writes source-relative provenance and environment metadata;
10. exits without touching the external checkout.

No absolute user path, username, hostname, transient cache path, or timestamp
is stored in the committed golden. Store source-relative paths, commit, source
hashes, Python/NumPy/Numba/Torch versions, dtype, device class, and fixed
thread/chunk policy.

Goldens are serialized with sorted array keys and `allow_pickle=False`.
Byte-level reproducibility is checked under the recorded generator
environment; array identity remains a separate check so a future ZIP/NumPy
container change cannot be mistaken for changed physics.

Promotion from staging is explicit and reviewable:

```text
generate → inspect staging manifest → compare with current golden
→ promote selected files → regenerate root manifest → run all gates
```

There is no `--force` mode that silently overwrites accepted goldens.

### 7.2 Oracle contents

`ch02_depth_integral_outputs.npz` should contain only:

- the NumPy `integrate_on_depth_grid` outputs for named deterministic cases;
- the private compiled integral outputs for the same cases;
- the Torch `integrate_optical_depth` CPU-float64 output in its exact
  `(wavelength, depth)` layout.

Input identity is held by the golden manifest's `oracle.input_assets`, not by
duplicated input arrays in the golden.

If the parallel transfer gate is retained,
`ch02_transfer_accumulation_outputs.npz` contains each named final accumulator
from:

- exact serial range accumulation;
- exact parallel accumulation at one fixed chunk count.

It does not contain the prepared operator/input state. That remains the
fixture.

### 7.3 Runtime-independence gate

The ordinary build and test commands run with only repository `src/` on
`PYTHONPATH`. They must not import from the oracle checkout or read the paper
tree.

Add a test that:

- imports the exact progressive modules;
- asserts each imported module's resolved `__file__` is inside this
  repository;
- runs the Chapter 2 CPU parity suite from a temporary working directory;
- removes external source paths from `PYTHONPATH`;
- verifies the result using only `data/fixtures` and
  `data/golden/payne_zero`.

Development oracle scripts may mention `--source-root`; notebooks, canonical
source modules, tests, manifests, and reader code may not require a
machine-specific source path.

## 8. Parity and correctness test matrix

Keep deterministic correctness, optional hardware comparison, and performance
measurement in separate lanes.

### 8.1 Fast deterministic tests

Suggested files:

```text
tests/test_data_manifest.py
tests/test_chapter02_schema_v4.py
tests/test_chapter02_depth_integrals.py
tests/test_chapter02_source_identity.py
tests/test_chapter02_self_contained.py
```

Required gates:

| Gate | Case | Claim |
| --- | --- | --- |
| analytic constant | nonuniform grid, constant positive value | integral equals surface seed plus \(c(x-x_0)\) |
| analytic linear | nonuniform grid, exactly representable coefficients | interval integration matches the antiderivative |
| small hand case | `D=4` | coefficients and cumulative values can be inspected in the text |
| stress case | positive values spanning several decades | finite, monotonic optical depth under declared dtype |
| NumPy/compiled | identical float64 inputs | exact-source/local and scalar/compiled policy recorded |
| NumPy/Torch CPU | explicit row/layout conversion only | same physical integral, distinct real interfaces |
| shape/axis | `(D,)` and `(W,D)` | exact output shape and depth axis |
| input mutation | all implementations | inputs remain byte-identical |
| stored source identity | selected local AST definitions | local definitions match committed AST fingerprints |
| schema identity | exact JSON and required tuple | field inventory is complete and canonical |
| schema positive path | compact synthetic archive | exact loader and validator accept it |
| schema negative path | one mutation per temporary file | precise boundary failure |
| manifest positive path | all committed assets | role, size, hash, and array metadata agree |
| manifest negative path | temporary corruption | byte identity fails loudly |
| golden order | chapter cell/source inspection | golden is opened only after candidate result exists |

Do not choose a universal `1e-12` tolerance. Use:

- `array_equal`/zero tolerance for cases whose operation order and
  representation have been measured to be bit-identical;
- a dtype- and implementation-specific policy for changed reduction order;
- separately measured float32 and float64 profiles;
- the worst index, maximum absolute error, maximum relative error, dtype, and
  device in every nonexact report.

The tolerance profile is locked only after source-oracle generation measures
the actual cases. A failing exact gate triggers investigation or an explicit
baseline review; it is not repaired by casually widening a tolerance.

The ordinary source-identity test is self-contained: it compares local
definitions with committed fingerprints. A development-only extension of
`scripts/verify_pinned_source_fragments.py` checks those fingerprints and
definitions against the external pinned checkout when that checkout is
available. Do not make the ordinary unit suite reach outside the repository.

### 8.2 Parallel accumulator test

The exact `accumulate_transfer_range_parallel` gate is useful only if it stays
small and honest:

- load the prepared-state fixture and name every array's shape/dtype/device;
- do not derive its radiation physics;
- run the serial exact function over the same frequency range;
- run parallel with `chunk_count=1` and require the measured exact policy;
- run a fixed `chunk_count>1` and compare under a measured regrouping policy;
- repeat the fixed chunk count and test repeatability;
- change chunk count and report the possible last-bit difference rather than
  promising cross-thread bit identity;
- verify the final reduction follows increasing chunk index.

If the fixture or call requires a long source dump in the main text, keep the
full gate in tests and show only a short call trace plus a private-buffer
schematic. Chapter 12 remains the owner of the accumulators' physics.

### 8.3 Optional backend tests

Run CPU Torch always. CUDA and MPS tests:

- skip explicitly when unavailable;
- use the exact runtime resolver and its dtype policy;
- compare CUDA/CPU float64 separately from MPS float32;
- synchronize before reading results or stopping a timer;
- assert shape, monotonicity, finiteness, and no unexpected host transfer;
- report device name, Torch version, dtype, and comparison policy.

Unavailable hardware is not represented by a fabricated result or a silent
CPU fallback.

### 8.4 Cold, warm, and fresh-process tests

Correctness gates:

- first compiled call and subsequent same-process call return identical
  values under the measured exact policy;
- a second fresh process loading a valid Numba cache returns the same values;
- cached and uncached Torch executions return the same values;
- candidate inputs remain unchanged.

Timing protocol:

- imports/setup are separate;
- first Numba compilation is separate;
- warm same-process calls use multiple repeats and median;
- fresh-process cache reload is separate from kernel execution;
- Torch allocation, host-to-device transfer, first device call, synchronized
  warm calls, and device-to-host transfer are separate;
- CUDA/MPS synchronization occurs before each stop time;
- benchmark records include versions, device, dtype, thread count, chunk
  count, workload shape, repeats, and cache state.

Use a temporary `NUMBA_CACHE_DIR` and keep caches outside the repository.
Performance tests assert valid measurements and numerical identity, not
"warm must be N times faster." Speed thresholds are hardware- and load-flaky.

The benchmark JSON and rendered plots belong under a generated verification
artifact directory, not any of the four `data/` roles, because timing is not
an input or a golden scientific array.

## 9. Main-text outputs and professional plots

Each plot makes one claim and is interpreted immediately from its actual
values.

### Plot 1 — Ordered depth integration

- x: `column_mass` in g cm\(^{-2}\), log scale if the declared case spans
  decades;
- y: optical depth, dimensionless;
- dark reference curve and one blue dashed candidate curve;
- annotate the surface seed and maximum measured difference;
- no claim of physical atmosphere closure.

This plot answers whether the distinct implementations preserve the same
cumulative integral.

### Plot 2 — Workload scaling

- x: number of independent wavelength rows \(W\);
- y: synchronized median time in ms;
- compare only implementations for which the plotted workload is explicitly
  the same depth-integral family;
- direct-label lines; use distinct line styles as well as color;
- identify CPU/Torch device and dtype in the subtitle or caption.

Do not place the full atmosphere transfer accumulator on the same curve as a
one-column integrator without stating its extra work.

### Plot 3 — Cold versus warm boundary

- one horizontal grouped-bar panel;
- categories such as `compile/setup`, `transfer`, and `synchronized execute`;
- no stacked bar that hides which phases overlap;
- actual machine and cache state in the caption;
- bars are a local measurement, not a universal speed promise.

### Plot 4 — Numerical discrepancy

- one logarithmic y-axis for maximum absolute error;
- categories are explicit backend/dtype cases;
- mark exact-zero cases separately rather than replacing zero with an
  unexplained floor;
- show the declared tolerance as a quiet horizontal guide only if it is
  measured and machine-readable.

All use `book.plot_style`: white background, paper-inspired serif math,
inward ticks, navy/slate primary colors, orange only for a caution or
float32 case, units on every axis, no positional Matplotlib color defaults,
and no dashboard-like multi-panel composition. Inspect the actual render for
clipping, label overlap, legend redundancy, and dark-reader behavior.

## 10. Original schematic specifications

These should be added later to `scripts/textbook_schematic_specs.py` using the
existing `FigureSpec`/`STYLE` architecture. They are original textbook
compositions; official website images are style references only.

### `ch02-independent-depth-and-frequency`

Scientific claim: depth remains ordered inside each frequency calculation,
while contiguous frequency chunks can be assigned independent private
accumulators and reduced in a fixed order.

Composition:

- one landscape `[depth, frequency]` slab;
- outermost depth at top, inward arrows down each highlighted frequency
  column;
- three contiguous frequency chunks separated by thin slate dividers;
- a private sum tray below each chunk;
- one left-to-right numbered reduction row.

Labels, at most five:

```text
outer layer
ordered depth
frequency chunks
private sums
fixed-order reduce
```

Do not draw threads writing into one shared accumulator.

### `ch02-acceleration-with-real-layouts`

Scientific claim: the same physical integral appears in distinct production
layouts; acceleration does not erase contracts.

Composition:

- four stepping stones connected by one navy path;
- `NumPy (D)`, `njit (D)`, `chunk prange`, and `Torch (W,D)`;
- a quiet CPU bracket under the first three and a device bracket under Torch;
- a small ordered-depth arrow on every stone.

Avoid speed numbers, software logos, and any GPU atmosphere arrow.

### `ch02-four-data-roles`

Scientific claim: static input, teaching subset, integration fixture, and
golden output have different provenance and information-flow permissions.

Composition:

- four separated desk cards in one row;
- computation receives arrows from static/subset/fixture;
- candidate result flows to a comparison gate;
- golden reaches only that comparison gate, never the computation;
- each card has a small fingerprint/hash mark.

Labels:

```text
static
subset
fixture
golden
compare
```

Caption must say that a checksum establishes identity, not correctness.

### `ch02-two-population-cubes`

Scientific claim: equal shape does not imply equal physical meaning.

Composition:

- two equal-size cubes labelled `actual n` and `partition-normalized n/U`;
- a small `divide by U` card between them;
- the same `(D,6,139)` shape tag below each;
- one muted warning `not bound levels`;
- no energy-level derivation.

The alt text must state that the right cube is an ion-stage population
divided by its partition function, not an individual bound-level population.

## 11. Chapter flow using the harness

The data mechanics should not become a detached appendix inside Chapter 2.
Use this causal order:

1. begin with the already-needed Chapter 1 depth integral;
2. freeze its reads/writes/units/axes/dtype/device contract;
3. predict an analytic result;
4. run the readable exact NumPy implementation and interpret it;
5. expose ordered depth versus independent wavelength/frequency work;
6. introduce compiled, parallel, and Torch realizations only where that
   dependency structure permits them;
7. compare values before timing;
8. separate cold/warm/fresh-cache measurements;
9. ask what input data made the parity claim reproducible;
10. establish the four roles and checksum identity;
11. ask what exact atmosphere arrays must cross into synthesis;
12. validate the compact schema fixture while explaining what validation
    cannot prove;
13. close with checksum identity, schema validity, numerical parity, and
    physical acceptance as four different claims.

There is no exercise section. Unit mistakes, reversed axes, one-byte
corruption, float32 differences, and schema failures appear at their causal
point with an immediate prediction and interpretation.

## 12. Cleanup and integration risks

### P0 — Must block Chapter 2 acceptance

- A golden is read before the reader computation or is used to supply an
  input.
- One NPZ mixes any two of static/subset/fixture/golden roles.
- The schema fixture is described as a physical atmosphere.
- A schema-valid product is called converged or correct.
- Runtime imports reach `/Users/ysting/payne-zero` or the paper tree.
- The exact NumPy, compiled, parallel, and Torch interfaces are hidden behind
  an invented common public wrapper.
- `prange` is applied to the ordered depth recurrence.
- asynchronous device launch time is reported as completed runtime.
- official-site artwork is copied into the chapter.

### P1 — Must resolve before the data migration is called clean

- The 369 MiB `reference/` directory remains an active dependency of the new
  Chapter 2 path.
- The old `_pipeline/make_*_reference.py` scripts can overwrite accepted new
  goldens directly.
- Legacy bundles with mixed roles are moved without array-by-array audit.
- `allow_pickle=True` enters the new data path. Eighteen old `_pipeline`
  scripts currently contain it; they must not be used as templates.
- The root manifest and per-subset `CHECKSUMS.sha256` duplicate a digest and
  drift. Regenerate both in one reviewed command and test consistency.
- NPZ keys are written in nondeterministic order or object arrays are stored.
- Numba caches or Python bytecode appear under `src/`, `data/`, or the book
  directory.
- committed timing artifacts are mistaken for portable performance facts.
- MPS is asked to run the CPU/CUDA float64 policy, or unavailable accelerators
  silently fall back to CPU.
- full-catalog verification runs during ordinary chapter rendering.

### P2 — Whole-repository cleanup after replacement

- Retire old `content/Lecture*.ipynb/html` products only after the new reader
  no longer indexes them.
- Remove obsolete `_pipeline/` builders and legacy reference assets only
  after a dependency search and scientific-content audit.
- Revisit the `.gitignore` comment that explicitly preserves
  `reference/*.npz`; change it only when the legacy directory is actually
  retired.
- Keep development-source identity manifests separate from reader data.
  Absolute local paths may be accepted by an oracle command-line option but
  should not be serialized into reader-facing provenance.

Do not perform these deletions speculatively. The current worktree already
contains broad in-progress changes, and the old reader may still depend on
legacy outputs.

## 13. Acceptance checklist

Chapter 2's data/parity layer is ready for integration only when:

- `data/` contains only assets with one declared role;
- root manifest verification passes from a clean process;
- every file's hash, byte count, array keys, shapes, dtypes, units, axes, and
  source identity are recorded;
- the schema JSON is exact pinned content;
- the synthetic schema-v4 fixture passes the exact loader/validator and all
  one-mutation negative tests fail for the intended reason;
- actual and partition-normalized population sentinels are distinct and their
  relationship is tested outside the validator;
- candidate integrals are computed before any golden is opened;
- pinned-source CPU goldens regenerate in an isolated staging directory;
- ordinary tests pass with no external checkout on `PYTHONPATH`;
- exact source/AST gates cover every displayed exact Chapter 2 kernel;
- fixed-thread/chunk repeatability and cross-thread regrouping are reported
  as different claims;
- CUDA/MPS results are either measured under their own policies or explicitly
  skipped;
- cold, warm, fresh-cache, transfer, and synchronized device timing are
  separated;
- no performance threshold is used as a correctness gate;
- plots are one-claim, professionally styled, and interpreted from their
  rendered values;
- schematic prompts are textbook-owned, scientifically audited, and recorded
  in the schematic manifest;
- Chapter 2 derives no Chapter 3–5 or Chapter 12 physics;
- the final summary states that population fields are now named and validated
  but still physically empty, making Chapter 3's atomic population problem
  the necessary next dependency.
