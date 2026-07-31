# Chapter 2 Causal Outline — From Equations to Fast, Trustworthy Kernels and Explicit Data

This is the section-level narrative contract for the canonical Chapter 2 source. It is subordinate
to `BIBLE.md` and `design/global_chapter_contracts.md`. It does not authorize a second
implementation, a book-defined compatibility wrapper, or a detached exercise section.

## Central question and earned claim

**Question.** How can the same depth integral remain scientifically identical when it is expressed
as readable array code, compiled CPU work, a parallel transfer reduction, and a Torch device
tensor?

**Claim the reader must earn.**

> A numerical value is usable only when its physical meaning, units, axes, precision, data
> identity, and validation claim are all explicit.

This sentence is the chapter's durable spine. “The contract” should never mean only a function
signature. It means:

```text
equation + units + axes + dtype/device + data identity + validation claim
```

The chapter applies that same rule three times:

1. to a depth integral;
2. to an abundance and its source data;
3. to the structured atmosphere that crosses from atmosphere construction to synthesis.

Those are not three unrelated topics. The second and third reveal information that the first
cannot encode by itself.

## Neighbor handoff

Chapter 1 ends with four controlled-limit arrays on the exact 80-layer grid. Its last unresolved
problem is concrete: later calculations must integrate through those layers at many wavelengths,
and a smooth-looking output can still come from the wrong axis, unit, order, or precision.

Chapter 2 must not rederive optical depth, the Planck function, the grey law, or hydrostatic
balance. It reads:

| object | shape | unit | dtype/device | meaning retained from Chapter 1 |
| --- | --- | --- | --- | --- |
| `standard_rosseland_optical_depth` | `(D,)`, normally `D=80` | dimensionless | NumPy `float64`, CPU | outer-to-inner depth coordinate |
| `column_mass` | `(D,)` | g cm\(^{-2}\) | NumPy `float64`, CPU | strictly increases inward |
| one extinction column | `(D,)` | cm\(^2\) g\(^{-1}\) | NumPy `float64`, CPU | interaction strength per unit column mass |

It writes:

- exact depth-integral parity and timing evidence;
- abundance arrays in their exact, deliberately different representations;
- a provenance and checksum result for declared data;
- a schema-v4 representation-validation result;
- the fixed-\(n_e\) bridge promise, but not the populations that Chapter 3 will derive.

Agreement between numerical implementations is not physical convergence. A valid schema is not a
physically acceptable atmosphere. A checksum says which bytes were read, not whether the
oscillator strengths in them are correct.

## Movement I — A smooth result can still be wrong

### 2.1 The dangerous answer is the one that looks reasonable

**Opening stakes.** Begin with the Chapter 1 depth grid and a positive toy extinction field with 80
wavelength rows and 80 depth columns. Because the array is square, its transpose has the same
shape. Integrating the intended array and the transposed array can produce two positive, smooth,
inward-increasing optical-depth curves.

Do not begin with a library list, decorator syntax, a hardware diagram, or a speedup chart.

The opening one-panel figure should show the two curves for one deliberately chosen wavelength:

- dark navy: “depth is axis 1”;
- muted orange dashed: “same numbers, axes swapped”;
- \(x\)-axis: layer index or `column_mass` with its unit;
- \(y\)-axis: optical depth, dimensionless.

The code preceding the plot must make the separable toy field and its transpose explicit. It must
not hide the mistake in a helper.

**Prediction before execution.** Both curves can look physically plausible because positivity and
monotonicity do not identify what an axis means.

**Required output interpretation.** The paragraph after the plot must read the actual bottom-layer
values and their measured ratio. It should say, in substance:

> Both results are positive and monotone, so appearance alone does not choose between them. At the
> selected wavelength the bottom optical depth is `<actual correct value>` for the declared
> wavelength-by-depth layout and `<actual transposed value>` after the silent axis swap. The
> calculation needs a contract, not a prettier plot.

No hard-coded ratio should survive if the fixture changes.

**Central claim.** State the chapter's one-sentence claim here. The next section asks what must be
frozen before any optimization is meaningful.

### 2.2 Freeze the physical contract before changing the code

Reuse, without rederivation,

\[
\tau_\nu(m)
=
\tau_{\nu,0}
+
\int_{m_0}^{m}\chi_\nu(m')\,dm'.
\]

Define only the information newly needed:

- \(m\): `column_mass`, g cm\(^{-2}\), increasing inward;
- \(\chi_\nu\): mass extinction, cm\(^2\) g\(^{-1}\);
- \(\chi_\nu\,dm\): dimensionless optical-depth increment;
- \(\tau_{\nu,0}\): a wavelength-specific surface seed;
- recurrence: a step whose result is needed by the next step;
- independent work: one wavelength does not need another wavelength's result.

Use the first original schematic here: a wavelength-by-depth slab with separate rows, inward arrows
within each row, and no arrows between rows. The labels should be no more than `wavelength`,
`ordered depth`, `independent rows`, and `surface seed`.

The schematic establishes the chapter's key dependency:

```text
different wavelength rows: independent
within one row: ordered outer → inner recurrence
```

Now introduce the first exact production boundary, because the reader has earned the role:

| exact function | reads | writes | layout |
| --- | --- | --- | --- |
| `payne_zero_atmosphere.radiative_transfer.integrate_on_depth_grid` | `grid (D,)`, `values (D,)`, keyword-only `surface_value` | one NumPy `(D,)` integral | one depth column |
| `payne_zero_synthesis.radiative_transfer.integrate_optical_depth` | `column_mass (D,)`, `extinction (W,D)`, `surface_tau (W,)` | Torch `(W,D)` optical depth | wavelength, then depth |

Do not invent a shared wrapper to make these signatures look alike.

Use a compact “what can detect the mistake?” table:

| mistake | plausible output possible? | what exposes it? |
| --- | --- | --- |
| kg m\(^{-2}\) numbers passed as g cm\(^{-2}\) | yes | unit-scale or analytic check; the kernel cannot infer units |
| `(D,W)` passed where `(W,D)` is required | yes when dimensions coincide | named-axis test; a square shape is insufficient |
| inward-to-outward `column_mass` | sometimes | strict monotonic-depth check |
| float32 used where float64 policy applies | yes | declared dtype and tolerance-specific parity |

This table must not imply that either integral performs dimensional algebra. Later, the schema
validator can reject reversed depth and incompatible shapes; it still cannot discover an
undeclared physical unit or the semantic transpose of a square array.

**Transition.** A contract tells us what answer to preserve. We next need a case simple enough to
check without trusting any implementation.

### 2.3 Integrate one depth column that we can audit by hand

Use the four-point case

\[
m=[0,1,2,3]\ {\rm g\,cm^{-2}},
\qquad
\chi(m)=1+2m\ {\rm cm^2\,g^{-1}},
\qquad
\tau_0=0.25 .
\]

Before code, derive

\[
\tau(m)=0.25+\int_0^m(1+2m')\,dm'=0.25+m+m^2,
\]

and predict

\[
\tau=[0.25,\ 2.25,\ 6.25,\ 12.25].
\]

The visible implementation should be taught in two bite-size stages, not as one large source dump:

1. form the constant, linear, and quadratic interval coefficients;
2. advance the integral in outer-to-inner order.

Explain why the first near-surface intervals are deliberately linearized in the exact coefficient
construction. Do not replace the public method with a trapezoidal “teaching” function.

Execute the canonical function and print the analytic and computed arrays on adjacent lines.

**Required output interpretation.**

> The computed values are exactly `[0.25, 2.25, 6.25, 12.25]` for this
> exactly representable linear case. The increments grow inward because both the interval position
> and \(\chi=1+2m\) grow. This establishes the recurrence and normalization; it does not establish
> speed or broad dynamic-range accuracy.

Follow immediately with two inline limiting cases:

- constant \(\chi\): equal-width layers add equal optical-depth increments;
- one layer only: the result is the supplied surface seed and no interval is traversed.

These are part of the main text, not exercises.

**Transition.** One audited column is readable, but a spectrum needs many independent wavelength
rows. The next step changes the layout without changing the depth dependency.

## Movement II — Accelerate only the work that is independent

### 2.4 Batch wavelengths without erasing the depth axis

Introduce an array axis as a physical index, not merely a dimension number. Build a two-row
extinction tensor from two copies of the audited case with different normalizations.

Before code, give the shape ledger:

```text
extinction                     (W, D)
column_mass                    (D,)
constant, linear, quadratic    (W, D)
interval_width                 (D-1,)
interval_tau                   (W, D-1)
optical_depth                  (W, D)
```

Explain broadcasting as alignment of the shared `(D-1,)` interval factors with every wavelength
row; it is not a conceptual copy of the depth grid. Predict that both rows retain independent
surface seeds and that only depth axis 1 is scanned.

At the earned boundary, show only the focused exact lines that create `interval_tau` and use
`torch.cumsum(..., dim=1)`, then execute
`payne_zero_synthesis.radiative_transfer.integrate_optical_depth`. The full private coefficient
helper should not be pasted into Markdown.

**Required output interpretation.** Read the actual `(2,4)` output and say:

- row 0 reproduces the audited four-point result;
- changing row 1 changes no value in row 0;
- `cumsum(dim=1)` orders depth while keeping wavelength rows independent;
- the returned tensor retains the input dtype and device.

Add an explicit assertion that the input tensors were not mutated.

Then inspect one C-contiguous `(W,D)` NumPy analogue:

- with `float64` and `D=80`, the depth stride is 8 bytes;
- its transpose has the same values but a different stride order and is not C-contiguous.

Show the actual `shape`, `strides`, and contiguity flags. Interpret these as memory-access facts,
not correctness facts: contiguous wrong axes are still wrong.

State the actual architecture-specific layouts once:

- atmosphere opacity slabs are commonly depth-major `(depth, frequency)`;
- synthesis transfer tensors are wavelength-major `(wavelength, depth)`;
- `planck_bnu` from Chapter 1 has its own `(depth, wavelength)` contract.

There is no universal book layout and no general-purpose bridge that hides these differences.

### 2.5 Compile the ordered loop; do not pretend compilation means parallelism

The Python/NumPy recurrence is now understood. Introduce just-in-time compilation because the same
ordered numerical loop will be called many times.

Teach the options from separate needs:

| exact choice | what it changes | what it does **not** claim |
| --- | --- | --- |
| `njit` | compiles supported numerical Python to machine code | it does not choose independent work |
| `cache=True` | stores reusable compilation artifacts | a first call in every context is not free |
| `nogil=True` | releases Python's global interpreter lock during compiled work | it does not create threads |
| `parallel=True` | enables Numba's parallel transformation | ordered recurrences do not become independent |
| `prange` | marks a loop whose iterations may run independently | shared writes still require safe ownership/reduction |

Show the exact `_njit = numba.njit(cache=True, nogil=True)` assignment and execute the private
`_integrate_on_depth_grid_compiled` only as an implementation checkpoint. Its leading underscore
must remain visible: it is not a new public “fast integral” API.

Run the hand case and a smooth `D=80` case through the public NumPy and private compiled forms.

**Required output interpretation.**

- State the measured maximum error and worst depth.
- If the hand case is bit-equal, limit that claim to the hand case.
- If the smooth case differs, identify whether the difference is rounding or a changed operation.
- Say explicitly that the compiled depth recurrence remains serial.

No speed claim occurs until numerical agreement is established.

### 2.6 Parallelize frequency chunks with private state

Ask: if depth cannot be parallelized safely, where is the real independent work? Return to the
first schematic: frequency columns are independent.

Introduce:

- a thread as one executing worker;
- a race condition as two workers updating the same location without a defined ownership rule;
- a reduction as combining private partial results;
- reduction order as part of floating-point behavior.

Do not fabricate a `prange` version of the small optical-depth function. The exact parallel
production object is the enclosing atmosphere transfer workload:

- `transfer_chunk_count()`;
- `accumulate_transfer_range_compiled(...)`;
- `accumulate_transfer_range_parallel(...)`.

Use a declared **integration fixture** for the many required transfer inputs. Its manifest entry
must state provenance, shapes, units, dtype, and the fact that Chapter 12 owns the full physical
derivation. In the narrative, teach only the dependency structure:

```text
contiguous frequency ranges
    → one private accumulator set per chunk
    → `prange` over chunks
    → serial ordered work within each chunk
    → fixed chunk-order reduction
```

The code card should show this stage trace or a focused source excerpt, not the long transfer
kernel body.

**Predictions before execution.**

1. Repeating with the same thread/chunk policy should reproduce the same result.
2. Changing the grouping may alter a few last bits because floating-point addition is not
   associative.
3. A one-frequency workload should not be advertised as benefiting from many threads.

**Required output interpretation.** Report the actual fixed-policy repeat difference and the
serial-versus-parallel maximum absolute/relative difference with its worst accumulator and depth.
If regrouping changes no value for the compact fixture, say that this fixture did not expose the
allowed effect; do not convert the observation into a universal bit-identity claim.

Include the small-work limiting case in place: show the measured time at one or a few frequencies
and explain that scheduling/private-buffer overhead can dominate. This is evidence against a
blanket “more threads is faster” statement.

### 2.7 Put the wavelength batch on its actual device

Only after the tensor dependency is understood should the chapter define:

- tensor;
- host and device;
- CUDA GPU;
- Apple Metal/MPS;
- upload, device-resident work, synchronization, and return transfer.

Use the second original schematic here. It combines the acceleration ladder with the actual
architecture:

```text
one CPU column: NumPy → private njit
many CPU frequency chunks: exact prange transfer path
many synthesis wavelength rows: Torch on CUDA / MPS / CPU
```

Keep the physical atmosphere loop on multicore CPU. Do not draw a GPU atmosphere arrow. The
initializer may appear only as a small deferred Torch lane, not as part of this kernel.

Introduce the exact runtime boundary after this picture:

- `payne_zero_synthesis.device.device()`;
- `resolve_runtime(requested_device=None, requested_dtype=None)`;
- CUDA → MPS → CPU default preference;
- `float64` work on CUDA/CPU and `float32` work on MPS;
- `ACCUMULATION_DTYPE = torch.float32` for later synthesis line deposition on every backend;
- large synthesis tensors remain on device and completed spectra return once.

Execute the audited batch on every actually available requested device. The notebook must degrade
honestly to CPU rather than imply a GPU was tested.

**Required output interpretation.** Print a compact row per tested backend:

```text
device | dtype | shape | max_abs_error | tolerance | passed
```

The following prose must name the detected backend and actual dtype. Float32 and float64 results
must use separate policies. This proves one device kernel's contract, not a complete GPU spectrum.

**Transition.** We now have several fast realizations. Before comparing speed, we must define what
“the same result” and “the timed operation” mean.

## Movement III — Trust requires a declared comparison

### 2.8 Decide whether two answers are the same

Define absolute error, relative error, worst index, and tolerance from a concrete pair of numbers.
Avoid “machine precision” as a universal phrase.

Run three cases:

1. the exactly representable four-point linear integral;
2. a smooth, positive 80-layer physical-scale extinction;
3. a positive dynamic-range stress case whose increments span many orders of magnitude.

For every candidate/reference pair, create a **chapter result table**, not a public dataclass:

| case | candidate | reference dtype | candidate dtype/device | max abs | max rel | worst index | monotone | policy | status |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |

The table compares only genuinely corresponding outputs. The full parallel transfer path has
additional accumulators and must be reported as a separate stage comparison, not placed on a
misleading one-column speed/parity row.

**Required output interpretation.**

- Read the worst case, index, and scale from the executed table.
- Explain why exact equality is justified only where operation order and representation warrant it.
- State that monotonicity is a physical/numerical invariant but cannot by itself detect the opening
  axis error.
- State that parity means agreement with the declared reference under this policy, not radiative,
  hydrostatic, or chemical closure.

#### Constants are part of the numerical contract

Use the dynamic-range result to motivate, not interrupt with, the exact and parity-reference
constant tiers. Show one compact two-package table with the real names:

- synthesis unsuffixed exact constants and formula-specific `REFERENCE_*` constants;
- atmosphere `*_EXACT` and `*_REFERENCE` constants;
- distinct source-conversion names such as
  `LINE_CATALOG_WAVENUMBER_PER_EV` and `WAVENUMBER_PER_EV_REFERENCE`;
- `HYDROGEN_PROFILE_ATOMIC_MASS_GRAM` as a formula-specific value.

Do not invent a book-wide suffix convention and do not rederive the physics of these constants.
The point is numerical identity: substituting a newer literal can break parity even when the
physical dimension is unchanged.

### 2.9 Ask what the stopwatch included

Define separate timing boundaries:

- import/setup;
- first Numba compilation;
- warm same-process Numba call;
- fresh-process cache reload;
- device allocation;
- first synchronized device kernel;
- warm synchronized device call;
- host-to-device transfer;
- device-to-host transfer;
- end to end.

Use `time.perf_counter`, multiple repeats, and medians. Synchronize CUDA/MPS before stopping a
device timer. Do not time an asynchronous launch and call it execution time.

Prefer two professional one-claim figures:

1. warm same-work runtime versus workload for the public one-column NumPy path and private compiled
   one-column path;
2. measured full-transfer throughput or speedup versus fixed thread count for the exact
   chunk-parallel fixture.

Device timing should be a compact table unless sufficiently matched workloads make a separate
one-panel plot honest. Do not draw one omnibus bar chart that visually equates a single-column
integral, a full transfer accumulator, and a batched device scan.

**Required output interpretation.**

- Report the observed cold/warm ratio without generalizing it to another machine.
- Name the workload size at any measured crossover; if no crossover appears, say so.
- Explain whether transfer time was included.
- State hardware, dtype, thread count, and synchronization policy beside the result.
- Never say “GPU is \(N\times\) faster” without a matched boundary and tested GPU.

**Transition.** The integrator can now be fast and numerically accountable. But an optical-depth
number is still scientifically ambiguous if the opacity's chemical mixture or source bytes are
ambiguous. The same contract must now be applied to the values entering the kernel.

## Movement IV — Make the physical values and their interface explicit

### 2.10 A fast opacity integral still needs a defined mixture

Begin with hydrogen in four roles:

- H: element;
- protium and deuterium: isotopes;
- H I and H II: neutral and singly ionized stages;
- H\(_2\): molecule.

Use plain language before “species slot”: related objects can share a chemical symbol without being
interchangeable numerical quantities. An array position is an indexing convention, not a physical
definition.

Derive abundance notation from a number ratio:

\[
A(X)=\log_{10}(n_X/n_H)+12,
\]

\[
[X/H]=\log_{10}(n_X/n_H)_\star-\log_{10}(n_X/n_H)_{\rm ref},
\]

\[
[X/Fe]=[X/H]-[Fe/H].
\]

Before code, predict:

- \(A({\rm O})=8.69\) means \(n_{\rm O}/n_{\rm H}=10^{-3.31}\), about
  \(4.9\times10^{-4}\);
- \(+0.30\) dex is a factor \(10^{0.30}\), about \(2.0\);
- \([{\rm O/H}]=[{\rm O/Fe}]+[{\rm Fe/H}]\).

Execute only this small arithmetic first and interpret each printed number immediately.

Then define:

- `[M/H]` as a declared global metal-pattern coordinate, not one element;
- `[\alpha/M]` as the offset for exactly O, Ne, Mg, Si, S, Ca, and Ti;
- the exact
  `ALPHA_ELEMENT_ATOMIC_NUMBERS = (8, 10, 12, 14, 16, 20, 22)`.

At the earned implementation boundary, keep the three exact representations visibly separate:

| path | exact boundary | output | meaning |
| --- | --- | --- | --- |
| standard pattern | `compute_metal_log_number_abundances(...)` | `(97,)` | log10 metal number abundances for \(Z=3,\ldots,99\) |
| direct mode | `complete_direct_abundance_vector(...)` | `(97,)` | authoritative, 0.01-dex-quantized direct solver vector after all 81 public labels are complete |
| external deck | `linear_elemental_abundances(model)` | `(99,)` | positive linear values indexed by \(Z-1\), not silently normalized |

Use `compute_hydrogen_fraction` only after fixed He and metals have been assigned. Show one standard
pattern with `[M/H]`, `[\alpha/M]`, and one absolute element override; print H, Fe, one alpha
element, and the overridden element rather than dumping 97 entries.

In a second compact cell, inspect the direct `(97,)` output and the deck `(99,)` output side by
side. State explicitly:

- identical `(97,)` shape does not make the standard-pattern and direct coordinates the same mode;
- a sparse public `x_over_h` mapping is completed with `fe_over_h` only at the higher initializer
  boundary, not inside `complete_direct_abundance_vector`;
- `metallicity` and `alpha_enhancement` do not apply in direct-abundance mode;
- deck H and He are already linear, metals are decoded from stored logarithms, and no
  sum-normalization occurs.

**Required output interpretation.** Read the actual selected values, verify the seven alpha
indices, confirm both exact shapes, show the direct 0.01-dex lattice on a chosen coordinate, and
print the deck sum only to demonstrate that it is not forced to one.

Do not teach initializer routing, neural features, or physical abundance closure here; Chapter 14
owns those.

### 2.11 Bind each data claim to a role and exact bytes

The abundance calculation needs reference mixtures and later opacity needs large catalogs.
Introduce the authoritative four file roles with one concrete example each:

| repository role | meaning | allowed use |
| --- | --- | --- |
| `data/static` | immutable physical tables, schemas, checkpoints | may enter the reader calculation |
| `data/subsets` | checksum-bound small source-catalog slice | may enter only the declared teaching case |
| `data/fixtures` | computed upstream state supplied to isolate a lesson | may enter, but is never called “built here” |
| `data/golden/payne_zero` | pinned comparison output | loaded only after the reader's result |

Use the third original schematic: four separated trays, with arrows from static/subset/fixture into
declared lessons and no arrow from golden into the physical calculation. A generated runtime cache
is rebuildable output tied to source identity; it does not become a fifth kind of physical truth.

Introduce manifest, provenance, checksum, and SHA-256 from a familiar question: “How do we know
that two readers used the same bytes?” Open the exact source-catalog identity boundary only now:

- `source_catalog_root()`;
- `load_source_catalog_checksums()`;
- `source_line_paths()` and `atmosphere_source_catalog_paths()`;
- `verify_source_catalog_checksums(...)`.

The visible notebook uses a tiny staged textbook fixture and a temporary copy. It must not hash the
multi-gigabyte optional catalog during normal rendering and must not mutate a committed asset.

First print the successful exact result fields:

```text
status, root, checksum_manifest, checksum_manifest_sha256,
file_count, total_bytes, files[path, bytes, sha256]
```

Then flip one byte in the temporary copy and rerun the exact verifier.

**Required output interpretation.**

> The first result binds the declared relative path to `<actual byte count>` bytes and the printed
> SHA-256. After one byte changes, verification fails even though the file keeps its name and size.
> The hash establishes identity. It says nothing about whether the physical data are accurate.

Keep `source_data_files/runtime_data_manifest.json` distinct from
`source_data_files/source_catalogs/CHECKSUMS.sha256`; they answer different questions. The private
`_sha256` helper remains private.

**Transition.** We can now say what the numbers mean and which bytes supplied them. The final
problem is to carry the full depth state into synthesis without relying on shape alone.

### 2.12 Same shape is not the same physical quantity

Introduce the two population cubes through one numerical example, without deriving partition
functions:

\[
n=10^{12}\ {\rm cm^{-3}}, \qquad U=4,
\qquad n/U=2.5\times10^{11}\ {\rm cm^{-3}}.
\]

Both full arrays have shape `(D,6,139)`:

- `ion_stage_populations`: actual ion-stage number density \(n\);
- `partition_normalized_populations`: \(n/U\), population divided by a partition function.

Use the fourth original schematic: equal-size cubes labeled `actual n` and `n/U`, separated by a
small `divide by U` card and accompanied by `same shape ≠ same meaning`. Include the muted warning
`n/U is not a bound-level population`.

Execute the one-number example and a tiny synthetic cube only to demonstrate representation.

**Required output interpretation.**

> Dividing by \(U=4\) changes \(10^{12}\) to \(2.5\times10^{11}\) while preserving shape and
> density-like units. Shape and positivity therefore cannot distinguish the arrays. Later,
> `ion_stage_populations` supplies actual charge-weighted populations, while a bound-level LTE
> calculation begins from `partition_normalized_populations` and still needs statistical weight
> and a Boltzmann factor.

Do not derive \(U\), Boltzmann factors, Saha ionization, or charge closure. Those are the missing
physics that opens Chapter 3.

State the fixed-\(n_e\) bridge promise:

- a full charge solve determines `electron_density`;
- the synthesis bridge preserves a supplied `electron_density` while filling synthesis-shaped
  populations;
- it must not silently rerun charge closure and change the atmosphere.

No population field is claimed as physically computed in this chapter.

### 2.13 Seal the structured atmosphere boundary

Explain schema in ordinary language: a schema is a promise that named arrays carry declared
representations together. It is not a certificate that the star obeys the equations of physics.

Generate the complete schema-v4 inventory from the canonical declarative source or exact constants;
do not hand-maintain a second field list in chapter prose. Render it once, grouped by role:

**Thermodynamic depth columns**

```text
temperature
gas_pressure
electron_density
mass_density
column_mass
microturbulence
hc_over_kt
```

**Population and width fields**

```text
partition_normalized_populations
ion_stage_populations
fractional_doppler_widths
hydrogen_neutral_population
helium_neutral_population
helium_singly_ionized_population
molecular_hydrogen_population
hydrogen_partition_normalized_ion_stage_populations
carbon_partition_normalized_ion_stage_populations
magnesium_neutral_partition_normalized_population
aluminum_neutral_partition_normalized_population
silicon_neutral_partition_normalized_population
iron_neutral_partition_normalized_population
```

**Composition and continuum-edge fields**

```text
elemental_abundances
signed_continuum_edge_frequency_hz
continuum_edge_wavelength_nm
continuum_edge_midpoint_wavelength_nm
continuum_edge_interval_width_squared_over_two_nm2
```

The rendered inventory must include exact shape, unit, host dtype, and one-sentence meaning:

- scalar depth columns `(D,)`;
- populations `(D,6,139)` and species-specific smaller shapes;
- `elemental_abundances` positive one-dimensional with at least 99 entries, while the public
  builder path documents `(99,)`;
- edge arrays `(Q,)` and `(Q-1,)`;
- `atmosphere_schema_version = 4` stored as one `int32` value;
- host archive numerical fields as NumPy `float64` unless their exact type differs.

Future chapter ownership may appear in one compact table column. Do not repeatedly write “as we
will see” in prose and do not derive future fields here.

Use a tiny NPZ labeled **synthetic schema data, not a physical atmosphere**. At the exact boundary,
call:

- `validate_atmosphere_npz(path)`;
- `load_atmosphere_product_metadata(path)`.

The main-text failure gallery should be compact and executable:

1. valid representation;
2. reversed `column_mass`;
3. missing `ion_stage_populations`;
4. wrong population cube shape;
5. inconsistent `Q`/`Q-1` edge lengths.

Use temporary variants derived from the declared fixture. Print one line per case with the actual
accepted/rejected status and concise reason.

Then make the validator's limit visible: swap same-shaped, nonnegative actual and
partition-normalized values while retaining their canonical field names. If the exact structural
validator accepts the archive, interpret that acceptance honestly—it validates the representation,
not the physical relationship \(n/U\).

Read optional product metadata separately. An initializer product may report:

```text
atmosphere_product_role = learned_initializer_prediction
atmosphere_converged = False
atmosphere_closure_required = True
```

**Required output interpretation.**

- The valid fixture passes only the declared schema checks.
- Reversed depth and incompatible shapes fail at representation boundaries.
- A semantic population swap can remain structurally valid, proving why schema and physics are
  distinct.
- An unconverged, closure-required initializer can be schema-valid and still be unacceptable as a
  physical atmosphere.

End with the four claims, in increasing scope:

1. **checksum identity** — these are the named bytes;
2. **schema validity** — these arrays have the required representation;
3. **numerical parity** — a candidate agrees with a declared reference and policy;
4. **physical acceptance** — conservation, closure, convergence, and regime tests pass.

Mark the fourth as **not yet earned**.

## 2.14 Chapter summary

Return to the opening square array. The answer is now clear: shape and smoothness cannot protect a
scientific calculation. The reader has made the depth integral accountable by attaching units and
axes, preserving its ordered recurrence, parallelizing only independent frequency work, retaining
the exact CPU/device layouts, and comparing values under dtype-, device-, and grouping-specific
policies. Timing now distinguishes compilation, cache reload, synchronized execution, and data
transfer.

The same rule made abundance coordinates and file inputs explicit. The reader can distinguish
standard-pattern and direct `(97,)` log representations from a `(99,)` linear deck array, classify
every data file by role, and verify exact bytes without confusing identity with correctness. The
schema-v4 boundary now has complete names, shapes, units, ordering, and metadata, including the
noninterchangeable actual and partition-normalized population cubes.

**Exact outputs now available**

- parity and timing evidence for the real NumPy, compiled CPU, chunk-parallel transfer, and Torch
  depth-integration paths;
- exact abundance representations and selected-value checks;
- checksum-bound data identity evidence;
- a complete schema-v4 representation validation;
- an explicit fixed-\(n_e\) bridge contract.

**Important missing claim.** The population fields have names and shapes but have not been
physically filled. The chapter has not derived excitation, ionization, charge balance, or a
converged atmosphere.

### Next: determine how atoms share particles and electrons

At one depth, temperature and composition do not say directly how many particles are neutral, how
many are ionized, which bound levels they occupy, or how many free electrons must accompany them.
Those quantities are precisely the empty schema fields that opacity will later consume.

[Continue to Chapter 3: Atoms, Ions, and Electrons](/reader.html?ch=3)

## Visible code and output budget

Target at most 18 substantial visible code cells. A viable budget is:

| cells | purpose |
| ---: | --- |
| 1 | opening square-axis trap and one-panel plot |
| 2 | hand-checkable coefficients/integral and limiting cases |
| 1 | exact Torch batch and non-mutation check |
| 1 | shape/stride/contiguity inspection |
| 1 | private compiled checkpoint and parity |
| 1 | exact parallel transfer fixture and fixed-policy comparison |
| 1 | available-device runtime and dtype comparison |
| 1 | three-case parity result table |
| 2 | matched warm timing and full-transfer thread scaling |
| 1 | abundance notation arithmetic |
| 2 | standard/direct/deck exact abundance paths |
| 1 | data-role/manifest inspection |
| 1 | successful and one-byte-failure checksum verification |
| 1 | schema inventory and valid synthetic archive |
| 1 | population distinction, invalid variants, and product metadata |

Setup/style cells may be hidden. Do not add a separate visible source dump for every imported
helper. Long exact routines are taught as ordered conceptual stages with focused excerpts, while
the canonical implementation is the executed object.

## Quantitative figure plan

Every plot uses the shared paper-inspired style: white background, dark reference, color-safe
model trace, inward ticks where suitable, units on axes, no default palette, and immediate prose
interpretation.

1. **Opening axis trap:** one panel, two smooth optical-depth curves, one claim.
2. **Matched one-column warm timing:** one panel, workload on the horizontal axis, runtime with
   explicit unit on the vertical axis.
3. **Exact full-transfer thread scaling:** one panel, fixed fixture and declared thread counts.

Parity, device availability, cold/warm boundaries, and schema failures are clearer as compact
tables than as ornamental plots. Do not turn the chapter into a benchmark dashboard.

## Original schematic plan

1. **Independent wavelengths, ordered depth** — establishes the recurrence and independent axis.
2. **Acceleration ladder and hardware split** — NumPy/private `njit`, exact CPU chunk `prange`,
   and Torch device batch without a GPU atmosphere path.
3. **Four data roles** — static, subset, fixture, golden; no golden-to-physics arrow.
4. **Actual versus partition-normalized populations** — equal shapes, distinct quantities, and
   “not bound levels” warning.

Each must use a textbook-owned prompt, the restrained website-inspired notebook aesthetic, short
labels, alt text, an explicit conceptual caption, a hash, and a scientific review. The schema
inventory and four validation claims should use crisp native tables rather than adding decorative
art.

## Density, redundancy, and honesty risks

### Highest density risks

1. **Five apparent topics.** Acceleration, timing, abundances, data, and schema will feel like
   separate lectures unless every transition returns to the one usable-value rule. Keep the four
   movements and their causal transitions visible.
2. **The exact parallel kernel is too large to display.** Teach its ownership and reduction trace;
   execute it through a manifest-bound fixture. Never paste the full signature and body into the
   narrative.
3. **The schema inventory is intrinsically dense.** Generate one grouped table and explain only
   the distinctions needed now. Chapters 3–8 own the physical derivations.
4. **Abundance modes can swallow the chapter.** One standard example and one direct/deck
   comparison are enough here. Neural routing and physical closure belong to Chapter 14.
5. **Benchmarks can crowd out meaning.** Limit plots to the three declared one-claim figures and
   put unmatched workloads in separate evidence.

### Redundancy fences

- Do not rederive optical depth or the grey atmosphere from Chapter 1.
- Chapter 2 owns basic NumPy-axis, broadcasting, Numba, `prange`, Torch-device, dtype, and timing
  explanations. Later chapters give only kernel-specific reminders.
- Do not teach partition functions, Boltzmann factors, Saha ionization, or charge closure; Chapter
  3 owns them.
- Do not teach molecular equilibrium; Chapter 4 owns it.
- Do not teach continuum-edge interpolation; Chapter 5 owns it.
- Do not teach complete scattering transfer; Chapter 9 owns it.
- Do not imply that this one Torch integral is complete GPU synthesis; Chapter 10 owns composition
  and caching.
- Do not teach learned-initializer inference or direct-abundance safety policy; Chapter 14 owns it.
- Do not enumerate historical schema aliases or implementation archaeology.

### Exactness and output gates

- No invented common wrapper around the exact depth integrals.
- No synthetic `prange` optical-depth API.
- No GPU atmosphere claim and no Numba synthesis claim.
- No timing without synchronization and a declared boundary.
- No hard-coded performance number in explanatory prose.
- No “machine precision” without a dtype/backend-specific policy.
- No checksum called scientific validation.
- No synthetic schema archive called an atmosphere.
- No schema validity called convergence or physical acceptance.
- No forward reference in prose except the final causal Chapter 3 dependency; compact physical
  owner labels in the schema table are sufficient.
- No detached exercises. Every useful limit, failure, and debugging question is predicted,
  executed, and interpreted at the point where it changes the argument.

## Acceptance questions for the Chapter 2 draft

The draft is not ready for rendering until all answers are yes:

- Does the opening show why a smooth result can be wrong before naming acceleration tools?
- Can a reader explain why wavelengths are independent but depth remains ordered?
- Is the four-point analytic result predicted and then read from actual output?
- Are the NumPy, compiled, parallel, and Torch objects kept in their real layouts and workloads?
- Are `njit`, `cache=True`, `nogil=True`, `parallel=True`, and `prange` explained as distinct?
- Does every timing statement name cold/warm state, workload, device, dtype, threads, transfer,
  and synchronization as applicable?
- Are standard, direct, and deck abundance representations visibly distinct?
- Can a reader explain why the four data roles and a checksum are scientifically necessary but
  insufficient?
- Is every schema-v4 field present once with exact name, shape, unit, dtype, and meaning?
- Does the population example prevent \(n\), \(n/U\), and bound-level population from being
  conflated?
- Does the closing validation ladder prevent identity, representation, parity, and physical
  closure from substituting for one another?
- Does the summary introduce no new object and causally require Chapter 3?
