# Chapter 2 exactness audit

Audit date: 2026-07-30

Authoritative chapter audited: `book/chapters/chapter_02.py`

Executed artifact inspected: `content/Chapter02.ipynb`

Pinned oracle: `/Users/ysting/payne-zero` at
`9c44001feae40b85146630499e6f8a5fed42e5af`

The canonical chapter is authoritative. The notebook is known to await the next
build after the fixed-\(n_e\) bridge and displayed-dtype edits; that expected
staleness is not counted as a finding below.

## Release verdict

- **P0: none.**
- **P1: six findings.** One committed fixture violates its declared unit by
  \(10^7\); the chapter does not actually compare its transfer result with the
  pinned golden it imports; the data manifest is materially less complete than
  the prose says; two prose statements disagree with executed output; one
  direct-abundance sentence contradicts the exact function; and metadata
  enforcement is attributed to the wrong validation boundary.
- **P2: three findings.** Chunk-private state is described as worker-private;
  the parity table combines maxima that can occur at different indices; and
  hardware/timing interpretations are hard-coded to one render.

The exact local source copies themselves pass the pin gate. The chapter-level
tests also pass, including the test that compares all nine local serial and
two-chunk outputs with the pinned goldens. The important distinction is that
the latter comparison currently exists only in a unit test, not in the
pedagogical execution path.

## P1 findings

### P1-01 — `hc_over_kt` in the schema fixture is \(10^7\) too large for its declared unit

**Anchors**

- `scripts/build_chapter02_fixture.py:89`
- `data/fixtures/chapter02_schema_v4_minimal.npz`, field `hc_over_kt`
- `data/static/schemas/atmosphere_schema.json`, field `hc_over_kt`
- `book/chapters/chapter_02.py:1293-1335`

The schema declares `hc_over_kt` in cm. The builder uses

```python
1.438776877e7 / temperature
```

which is \(hc/kT\) in **nm**, not cm. At 4800 K the committed fixture stores
`2997.451827...`; the exact cgs expression

```python
(6.62607015e-27 * 2.99792458e10) / (1.380649e-16 * 4800)
```

is `2.997451828e-4 cm`. The ratio is \(9.999999996\times10^6\).

Calling the archive “synthetic” does not make a declared unit optional. This
fixture is explicitly used to teach that a valid interface must bind meaning
and units, so the discrepancy undercuts the chapter's central contract even
though the current structural validator cannot detect it.

**Required correction**

Use approximately `1.438776877 / temperature` for cm, rebuild the deterministic
NPZ, update its manifest hash, and add a fixture test that checks the formula
against `temperature`. Re-execute every cell whose printed checksum changes.

### P1-02 — the chapter never compares its transfer outputs with the pinned golden

**Anchors**

- import of `TRANSFER_GOLDEN_PATH`: `book/chapters/chapter_02.py:69-73`
- executed transfer comparison: `book/chapters/chapter_02.py:534-562`
- claim that the golden exists for “the later comparison”:
  `book/chapters/chapter_02.py:1150-1154`
- hidden exact test: `tests/test_chapter02_exact_names.py:97-106`

The chapter runs the local byte-copied serial and parallel kernels and compares
them with each other and with a repeated local run. `TRANSFER_GOLDEN_PATH` is
imported but never loaded. The prose then says the golden “exists for the later
comparison,” but no later comparison occurs.

The unit test correctly performs exact comparisons of all nine serial and all
nine fixed-two-chunk arrays against the pinned Payne Zero golden. That proves
the assets are currently sound, but it does not teach the reader how the
external-oracle boundary differs from local self-consistency.

**Required correction**

After computing the reader result, load the golden, verify its embedded commit,
fixture hash, and transfer-table hash, and compare all nine serial and nine
parallel arrays. Keep the local serial-versus-parallel error analysis as a
separate question.

### P1-03 — `data/MANIFEST.json` does not record the complete shapes, units, and provenance claimed in the text

**Anchors**

- prose contract: `book/chapters/chapter_02.py:1127-1132`
- `data/MANIFEST.json`

The chapter says, “A manifest records role, path, shape, unit, provenance, and
checksum.” The current manifest records checksums for every entry, but its
array inventories are partial:

| artifact | arrays/keys in file | arrays described in manifest |
| --- | ---: | ---: |
| schema-v4 fixture | 26 | 4 |
| transfer input fixture | 13 | 5 |
| transfer tables | 5 | 4 |
| golden transfer archive | 21 | 0 |

The schema JSON entry also has no array inventory in the manifest, and the two
locally built fixtures do not name their builder/provenance explicitly. The
checksum gate therefore proves whole-file byte identity, but the manifest does
not independently make every axis, dtype, unit, and provenance claim explicit.
The current tests check allowed role, directory, uniqueness, and hash only.

**Required correction**

Either make the manifest exhaustive and test every NPZ key/shape/dtype/unit
against it, or narrow the prose to say that this version records byte identity
plus selected pedagogical fields. For a chapter whose central claim includes
axes and units, the exhaustive option is the coherent one.

### P1-04 — two quantitative prose statements disagree with the executed notebook

**Anchors**

- parity loop: `book/chapters/chapter_02.py:696-760`
- parity prose: `book/chapters/chapter_02.py:765`
- timing prose: `book/chapters/chapter_02.py:868-874`
- executed cells 15 and 16 of `content/Chapter02.ipynb`

There are three cases and three candidates, so the executed table has **nine**
result rows, not “twelve rows.” All nine pass.

At 1280 points the executed compiled median is
`1.76250032e-05 s = 17.625 microseconds`, not “below ten microseconds.” The
public median is `3.632291e-03 s`, so the qualitative speed comparison is still
true.

**Required correction**

Change twelve to nine. Derive the timing interpretation from executed values
or use a looser accurate phrase such as “tens of microseconds in this render.”

### P1-05 — the direct-abundance completion sentence contradicts the exact function

**Anchors**

- representation table and surrounding prose:
  `book/chapters/chapter_02.py:996-1009`
- contradictory sentence: `book/chapters/chapter_02.py:1033-1041`
- exact implementation:
  `src/payne_zero_atmosphere/direct_abundance.py`,
  `complete_direct_abundance_vector`

The exact function requires all 81 public finite-reference labels and then
itself initializes the complete 97-slot vector with `[Fe/H]`, overwriting the
81 supplied slots before quantization. Thus the remaining 16 solver slots
inherit the realized iron value **inside**
`complete_direct_abundance_vector`.

The sentence

> completion with `[Fe/H]` occurs at a higher initializer boundary, not inside
> `complete_direct_abundance_vector`

is only defensible if “completion” means filling **missing public labels** in a
retained-coordinate workflow. As written, it contradicts both the preceding
table and the exact source.

**Required correction**

State the two boundaries separately: a sparse subset of the 81 public labels
must be expanded to a complete 81-label mapping by a higher retained-coordinate
boundary; the exact function then fills the 16 non-public solver slots with the
quantized iron abundance.

### P1-06 — all-or-none metadata is not enforced by `validate_atmosphere_npz`

**Anchors**

- schema validation demonstration: `book/chapters/chapter_02.py:1338-1402`
- metadata claim: `book/chapters/chapter_02.py:1413-1418`
- exact boundaries in `src/payne_zero_synthesis/atmosphere.py`:
  `validate_atmosphere_npz` and `load_atmosphere_product_metadata`

`validate_atmosphere_npz` calls `load_atmosphere_npz`, which ignores the six
metadata fields. An archive containing only `atmosphere_product_role` plus all
25 physical arrays still returns 25 validated names. The separate
`load_atmosphere_product_metadata` function rejects the same archive as an
incomplete metadata extension.

Consequently, “optional, all-or-none metadata extension” is true only at the
metadata-loader boundary, and “prevent ... from being mislabeled” is too strong
unless a workflow actually calls that loader and acts on the flags.

**Required correction**

Name and demonstrate `load_atmosphere_product_metadata`. Explain that physical
array validation and product-role validation are separate calls, and state
whether the later synthesis workflow enforces the role automatically or leaves
that check to the caller.

## P2 findings

### P2-01 — private transfer state belongs to chunks, not necessarily workers

**Anchor:** `book/chapters/chapter_02.py:506-525`

The implementation allocates nine buffers per `chunk_count` entry and runs
`prange(chunk_count)`. A runtime thread may execute more than one chunk, and the
chapter fixture explicitly requests two chunks without asserting that the
Numba thread count is two. Replace “Each worker owns” with “Each chunk owns.”
The safety argument then matches the exact allocation.

### P2-02 — the parity table's two maxima can occur at different depths

**Anchors:** `book/chapters/chapter_02.py:724-760` and
`book/chapters/chapter_02.py:765-770`

The table prints `max(abs)`, `max(rel)`, and one `worst` index selected from
absolute error. In the stress `torch32` case, maximum absolute error occurs at
index 79 and has relative error \(4.98\times10^{-8}\), while maximum relative
error \(3.24\times10^{-7}\) occurs at index 43. The numbers are both correct,
but one “worst” column invites the reader to associate them with the same
layer, and “relative difference at the worst scale” is ambiguous.

Print separate `worst_abs` and `worst_rel` indices, or print the relative error
at the maximum-absolute-error index and label it that way.

### P2-03 — current-backend and timing prose is brittle under a fresh build

**Anchors**

- backend prose: `book/chapters/chapter_02.py:657-663`
- timing prose: `book/chapters/chapter_02.py:868-874`

The current executed artifact genuinely used MPS, but a self-contained reader
rebuilt on CPU-only or CUDA hardware will print different rows while retaining
the hard-coded MPS interpretation. Timing prose has the same issue. Prefer a
small code-produced interpretation or wording that points the reader to the
executed table without asserting a particular installed accelerator or
threshold.

## Categories with no findings

### Exact source identity — no findings

- Every byte-identical local module matches the pinned working tree and the
  `HEAD` Git object.
- Every progressive-module definition named in
  `src/PAYNE_ZERO_SOURCE_MANIFEST.json` matches the pinned AST.
- The current source hashes match the manifest.
- The copied transfer table and schema JSON are byte-identical to the pinned
  source artifacts.

### Public signatures and layouts — no findings

- `integrate_on_depth_grid(grid, values, *, surface_value)` and its float64
  `(D,)` behavior are represented correctly.
- `integrate_optical_depth(column_mass, extinction, surface_tau)` is correctly
  taught as `(D,)`, `(W,D)`, `(W,) -> (W,D)`, with depth on axis 1.
- The hand integral, two-row batch, strides, single-layer limit, and input
  non-mutation claims agree with the exact code and executed output.

### Device and dtype policy — no findings

- CUDA then MPS then CPU preference is exact.
- Omitted runtime dtype is float32 on MPS and float64 on CPU/CUDA.
- `ACCUMULATION_DTYPE` is correctly distinguished as float32.
- The canonical edit now displays the source precision of `torch32` as
  float32 even though the comparison copy is converted to float64.

### Numba, `njit`, and `prange` semantics — no findings beyond P2-01

- `njit`, `cache=True`, `nogil=True`, `parallel=True`, and `prange` are
  distinguished correctly.
- The ordered depth recurrence is not mislabeled as parallel.
- Parallel work is correctly placed around independent frequency chunks with a
  fixed-order reduction.

### Transfer precision wording — no findings

The chapter correctly overrides the stale source docstring: all nine shared and
chunk-private accumulators are float64. The line-opacity slab, transfer
operators, and source-grid work introduce float32 precision islands. The
serial/parallel discrepancy is therefore regrouped floating-point addition in
a float64 accumulation path influenced by float32 intermediates, not a
“float32 reduction.”

### Standard and direct abundances — no findings beyond P1-05

- The seven alpha atomic numbers are exactly
  `(8, 10, 12, 14, 16, 20, 22)`.
- Absolute element offsets correctly override the generic metal and alpha
  shifts.
- The standard `(97,)` absolute log10 number fractions, direct `(97,)`
  quantized `[X/H]` coordinates, and external-deck `(99,)` linear numbers are
  kept distinct.
- The 81 public direct labels, 16 non-public slots, centidex quantization, and
  support checks match the exact source.

### Source-catalog paths — no findings

The returned `source_line_paths()` mapping is described accurately, including
predicted, observed, high-excitation, diatomic, TiO, water, and detailed
transition catalogs. The H3+ nuance is correct: the docstring mentions it, the
mapping does not return it, and `h3plus_lines_path` is an optional explicit
input used only when supplied and present.

### Population meaning — no findings

`ion_stage_populations` is correctly identified as actual \(n\);
`partition_normalized_populations` is \(n/U\), not a bound-level population.
Their `(D,6,139)` layouts and density-like units are exact. The explanation
that bound-level LTE populations still require statistical weight and a
Boltzmann factor is correct.

### Schema names, shapes, units, and structural failures — no findings beyond P1-01 and P1-06

- All 25 required names, symbolic shapes, and declared units match the pinned
  schema.
- Depth ordering, positivity/non-negativity, `(D,6,139)`, `(D,2)`,
  `n_element >= 99`, and continuum-edge checks are represented correctly.
- The demonstrated missing field, repeated column mass, and short cube fail for
  the stated reasons.
- The positive same-shape temperature/gas-pressure swap is correctly accepted,
  illustrating that the structural validator cannot infer units.

### Full-closure versus fixed-\(n_e\) population routes — no findings

The new bridge paragraph matches the pinned implementation:
`solve_population_state` performs the charge-balance solve, while
`solve_population_state_at_electron_density` fills the full population state
at a supplied electron density. The atmosphere-to-synthesis bridge uses the
latter with the converged upstream atmosphere value.

### Golden asset correctness — no findings

The committed serial and fixed-two-chunk golden arrays match the current exact
local kernels array-for-array, and their embedded fixture/table identities
match the declared artifacts. P1-02 concerns only the missing visible
comparison in the chapter.
