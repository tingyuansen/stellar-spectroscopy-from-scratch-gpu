# Chapter 2 independent pedagogical-flow audit

Audit date: 2026-07-30

Audited artifacts:

- canonical source: `book/chapters/chapter_02.py`;
- executed reader artifact: `content/Chapter02.ipynb`;
- binding standards: `BIBLE.md`,
  `design/pedagogical_flow_rubric.md`, and
  `design/global_chapter_contracts.md`.

Reader model: a final-year undergraduate or first-year graduate student who
knows ordinary Python, calculus, and introductory physics, but not Torch,
Numba, GPU programming, numerical reductions, stellar chemistry, or data
provenance practice.

The canonical source changed after the last notebook build to add the
fixed-electron-density bridge. The source is authoritative for this audit. The
executed notebook is still useful evidence for output values and visual
rendering; its pending rebuild is not counted as a chapter-content defect.

## Overall verdict

The chapter has a strong causal spine and is substantially closer to the
requested teaching standard than an API survey. The opening failure is
memorable, the six-part numerical contract recurs coherently, exact names
arrive at meaningful boundaries, and most code cells are predicted and then
read from their actual output. The abundance, data-role, and schema movements
all answer the opening claim that shape and plausible values do not preserve
meaning.

There is no P0 defect in the current canonical source. Chapter acceptance
should nevertheless wait for the P1 items below. The most important are not
stylistic: a novice meets Torch concepts before their definitions, the
parabolic interpolation and real `prange` structure remain hidden behind
calls, the pinned golden result is never compared, and two prose statements
contradict the executed output. The chapter also crosses the book's measured
density gate.

## P0 — blocking correctness failures

None found in the current canonical source.

## P1 — material acceptance issues

### P1.1 — Torch appears before the reader is told what a tensor, dtype, or device is

**Anchors:** §2.1, canonical lines 127–177 / executed cells 3–4; §2.2,
canonical lines 223–278 / executed cell 7; definition delayed until §2.7,
canonical lines 583–588 / executed cell 22.

The opening code uses `torch.as_tensor`, `torch.float64`, `.T.contiguous()`,
`.numpy()`, and the exact Torch integrator. The boxed contract and interface
table then use `dtype` and `device`. Only five sections later does the text say
that a tensor is a multidimensional array carrying a dtype and device. This
reverses the book's prerequisite floor and just-in-time machinery rule.

Define, in one compact paragraph before executed cell 4, a tensor, dtype,
device, and the host conversion performed by `.numpy()`. The later architecture
section can still deepen transfer and asynchronous-execution consequences.
Also define **kernel** at first use in the title/opening; the Bible explicitly
forbids treating that term as self-explanatory.

### P1.2 — `surface_tau` is numerically used but its physical seed convention is unresolved

**Anchors:** §2.1 cell 4 / canonical lines 157–170; §2.2 lines 232–240;
exact source cell 13 in the executed notebook.

The prose calls \(\tau_{\nu,0}\) a wavelength-specific surface seed, the
opening constructs it as `extinction[:, 0] * column_mass[0]`, and the hand case
uses an arbitrary `0.25`. The displayed exact `integrate_optical_depth`
docstring calls the same argument a “top half-cell seed.” A novice cannot tell
whether this is a boundary value, a half-cell approximation, or merely a
caller-supplied normalization.

State the production caller convention explicitly, then separate it from the
integrator's more general mathematical contract: the routine starts at the
supplied first-layer optical depth and does not derive that seed. Reconcile the
visible docstring wording rather than leaving three interpretations adjacent.

### P1.3 — the core parabolic step is asserted, not built

**Anchors:** §2.3, canonical lines 304–315 / executed cells 8–9; §2.4,
canonical lines 351–373 / executed cells 12–13.

The text announces a fit \(a+bm+cm^2\), says near-surface curvature is
suppressed, and calls the displayed `integrate_on_depth_grid` the “complete
exact public integration function.” That function immediately calls
`parabolic_coefficients`, whose interpolation, boundary linearization, and
curvature blending are neither shown nor derived. The Torch source likewise
calls the hidden `_parabolic_interval_coefficients`. These helpers are part of
the chapter's binding exact numerical spine, not incidental infrastructure.

The linear analytic case verifies integration of a line but cannot verify the
quadratic construction or blend. Add a bite-size conceptual derivation and an
executed coefficient/interval check before the complete recurrence. If the
full helper is too long, split its exact order into named visible stages and
test those stages; do not describe a caller of an unexplained helper as the
complete construction.

### P1.4 — the real `parallel=True`/`prange` implementation remains hidden

**Anchors:** §2.5 table, canonical lines 451–465; §2.6 dependency text and
cell, lines 508–564 / executed cells 20–21.

The explanation of threads, races, private buffers, and fixed-order reduction
is good, but the only visible execution is
`run_transfer_fixture(chunk_count=2)`. That book helper hides the exact
`@numba.njit(parallel=True, ...)`, `numba.prange(chunk_count)`, per-chunk
buffer indexing, and serial reduction loops. A reader can repeat the call but
cannot rebuild the safe pattern the section claims to teach. This is especially
important because Chapter 2 owns Numba/Torch syntax and the user explicitly
requested a visible, pedagogical comparison of `njit`, `prange`, and their
limits.

Expose the exact long routine as a few ordered source excerpts or conceptual
stages: allocation and bounds, the one-line `prange` ownership rule, and the
serial reduction. Keep the full physics kernel deferred to Chapter 12, but
make the concurrency structure executable and inspectable here.

### P1.5 — pinned golden parity is promised but never performed

**Anchors:** hidden import of `TRANSFER_GOLDEN_PATH`, canonical lines 69–73;
serial/parallel computation in §2.6, lines 536–564; data-role rule in §2.11,
lines 1103–1135; “later comparison” at lines 1153–1157.

The reader computes serial and two-chunk results before the golden file is
introduced, which is the correct order. The golden path is imported and its
file hash is checked, but no later cell loads its arrays or compares any of the
nine accumulators with the pinned oracle. The sentence “it exists for the
later comparison” has no fulfillment in the chapter. Only the unit test
performs exact golden equality, so the reader-facing production-convergence
claim remains hidden.

After the reader result exists and the manifest hash passes, load the golden
output and report equality/tolerance for all nine named accumulator families.
Keep the serial-versus-parallel comparison as a distinct claim; it does not
replace pinned-source parity.

### P1.6 — two post-output interpretations contradict the executed values

**Anchors:** §2.8 post-table prose, canonical line 765 / executed cell 32;
§2.9 post-plot prose, canonical lines 870–874 / executed cell 36.

- The parity table contains \(3\) cases × \(3\) candidates = **nine** result
  rows, but the prose says “All twelve rows.”
- The executed 1280-depth compiled median is
  `1.76250032e-05 s`, or about **17.6 microseconds**, but the prose says it
  remains below ten microseconds.

Correct the row count. For timing, avoid a brittle fixed threshold in static
prose; either render the measured value into output and say “tens of
microseconds,” or generate the interpretation from the current measurement.
Re-read every numeric sentence after the authoritative rebuild.

### P1.7 — the timing section does not yet satisfy the cold/warm/fresh-cache contract

**Anchors:** §2.5 lines 463–465; §2.9 cells 33–36 / canonical lines roughly
810–882.

The chapter correctly admits that its “first call” may be compilation or an
on-disk cache load. It therefore has not measured a true cold compile or a
fresh-process cached start. It also does not print the machine/software/thread
context that would make the warm curve reproducible. The binding Chapter 2
contract assigns cold, warm, and fresh-cache timing to this chapter.

Add a small subprocess-based three-state measurement, or narrow the claimed
scope and place the missing measured states in a clearly owned later
verification route. Report Python/NumPy/Numba version, CPU/thread policy, and
cache state beside the timings. A thread-scaling curve is not mandatory for
the four-frequency teaching fixture, but the text should explicitly defer it
to a sufficiently large exact workload rather than silently omit it.

### P1.8 — the ordered-depth schematic contradicts its own dependency claim

**Anchor:** §2.2 schematic, canonical lines 251–260 / executed cell 8;
`assets/schematics/textbook/ch02-ordered-depth-v1.png`.

The blue brace labeled “independent wavelengths” spans horizontally across
layers 1–8, exactly the same direction that the orange brace labels “ordered
depth.” Independence actually lies between the five horizontal rows. The
legend also shows a dashed “independent work” mark that does not appear in the
main drawing. The caption is correct, but the visual grammar contradicts it.

Redraw independence as a vertical bracket or separate row cards, retain the
horizontal ordered-depth arrows, and remove or use the dashed legend. This is
a scientific-content correction, not cosmetic polish.

### P1.9 — the chapter exceeds its binding density gate

**Anchors:** whole chapter; especially the transition from §2.9 to §2.10 and
the three abundance/data/schema movements in §§2.10–2.13.

The current notebook has approximately 4,600 prose words, 14 level-two
sections including the summary, and **30 visible code/source cells containing
about 666 visible code lines**. The global contract names more than 18
substantial visible code cells as a measured density-gate breach. Even at only
two minutes per visible cell, code inspection alone consumes an hour before
equations, schematics, outputs, and discussion.

Do not discard abundance, data, or schema material, and do not automatically
create a sixteenth chapter. First make two explicit movements or web routes
inside Chapter 2—trustworthy kernels, then explicit state/data—and remove
low-yield cell transitions. In particular, merge the declaration-only parity
cell with its comparison, avoid displaying a source function and then a
second cell that merely invokes it when one cell can carry the claim, and
move exact inventories into readable generated tables without re-explaining
them. Re-measure against the 18-cell/90-minute gate after the scientific
omissions above are fixed.

## P2 — valuable flow and polish improvements

### P2.1 — §2.8 repeats checks already earned

**Anchors:** §2.3 hand check, §2.5 smooth compiled check, §2.7 device check,
then §2.8 cells 29–31.

The formal tolerance policy and stress case are new and valuable. Recomputing
hand/smooth × compiled/Torch after those relations were already established
makes the middle feel like a verification rerun. Reuse the earlier named
results and let §2.8 add only the explicit policy, consolidated ledger, and
stress behavior. The declaration-only cell 29 (“declared cases”) does not by
itself change what the reader knows.

### P2.2 — the constants sidebar interrupts the parity-to-timing question

**Anchor:** “Constants belong to the numerical contract” inside executed cell
32 / canonical lines roughly 778–805.

The constant tiers matter globally, but this short inventory is not exercised
by the optical-depth example and creates no question that §2.9 answers. Fold
one sentence into §2.2's contract and move the exact name inventory to the
first chapter that compares a formula-specific reference constant.

### P2.3 — the H3+ note is premature implementation archaeology

**Anchor:** end of §2.11, canonical lines 1233–1237 / executed cell 59.

The stale-docstring distinction is correct but brings predicted/observed
catalogs, TiO, water, and H3+ into the narrative before molecular source
compilation is needed. It neither prepares the population representation nor
interprets the checksum output. Preserve the finding in the source/coverage
ledger and teach the executable `source_line_paths()` mapping when Chapter 8
actually consumes it.

### P2.4 — define schema and fixed point in ordinary language

**Anchors:** §2.13 opening, canonical lines 1294–1301; fixed-\(n_e\) bridge,
lines 1404–1411.

The reader can infer that a schema lists required arrays, but the chapter
should give the promised one-line definition: a schema is a machine-readable
contract for names, shapes, and representations, not a physics certificate.
“Charge fixed point” also appears before the course has defined a fixed point;
“iterative charge-balance solve” would preserve the bridge promise without
premature jargon.

### P2.5 — two schematics need small data-flow/axis clarifications

**Anchors:** §2.11 data-role schematic, executed cell 51; §2.12 population
schematic, executed cell 60.

- The data-role schematic correctly keeps golden data out of “calculate,” but
  it draws no arrow from `golden` to `compare`, even though the caption says
  golden output enters the comparison. Add that second comparison input.
- The population cubes show shape `(D, 6, 139)` while the visually dominant
  axes read species, ion stage, depth. Add `axis 0/1/2` labels so the drawing
  cannot be mistaken for `(139, 6, D)` in a chapter whose central lesson is
  axis meaning.

### P2.6 — the one-layer `ModelAtmosphere` demonstration exposes future fields at high cost

**Anchors:** §2.10, canonical source cell corresponding to executed cell 49.

The text honestly says the object is not physical, but the 32-line constructor
introduces `convective_flux`, `convective_velocity`, and other future state
solely to decode the abundance block. Keep the exact boundary, but visually
separate “required placeholder fields not used in this check” from the
abundance data under examination, or use a declared integration fixture and
show only the exact decoder logic. This would reduce cognitive load without
inventing a second API.

## What is already strong

- **Neighbor handoff:** Chapter 1 closes on the risk of a smooth wrong depth
  integral, and Chapter 2 opens by making exactly that failure visible.
- **Durable central claim:** meaning, units, axes, precision, identity, and
  validation recur from the opening transpose through abundance and schema
  validation rather than becoming disconnected mini-lectures.
- **Opening evidence:** the square-array trap is concrete, predicted before
  execution, plotted as one claim, and interpreted with actual bottom values.
- **Code/prose bridges:** every visible executable cell has prose before and
  after it; there are no adjacent unexplained code cells and almost every
  numerical output changes the reader's claim.
- **Honest acceleration:** the chapter never pretends the ordered depth
  recurrence is parallel, correctly separates compilation from independence,
  and accurately identifies float32 precision islands inside float64
  accumulators.
- **Actual backend reporting:** unavailable accelerators are not simulated;
  the executed result names CPU/MPS, dtype, error, and tolerance, while
  limiting the claim to one kernel.
- **Abundance representations:** element/isotope/ion/molecule, standard
  absolute log fractions, direct `[X/H]`, and external-deck linear values are
  separated with exact names and numerical closure checks.
- **Data honesty:** static inputs, subsets, fixtures, and goldens are
  distinguished; the one-byte checksum failure is a particularly effective
  in-flow debugging demonstration.
- **Schema honesty:** the 25-field inventory is generated from the canonical
  schema, the fixture is explicitly synthetic, structural failures are
  injected, and the semantic swap demonstrates the validator's limit.
- **Visual finish:** both quantitative plots are clean, one-panel,
  color-safe, unit-labeled, and legible. The architecture and population
  schematics are original and useful apart from the specific corrections
  above.
- **Self-containedness:** chapter execution uses repository-local source,
  tables, fixtures, schemas, and goldens; it does not require the external
  Payne Zero or paper checkout.
- **Product-name restraint:** the prose does not use “Payne Zero” as branding;
  production fidelity is conveyed through exact interfaces and parity claims.
- **No exercise detour:** useful limits, error injection, and variations are
  resolved inside the main causal sequence. There is no exercise section.
- **Close:** the summary returns to the opening failure, introduces no new
  physics beyond the already stated fixed-\(n_e\) bridge, distinguishes
  numerical trust from physical closure, and gives a genuine causal link to
  Chapter 3.

## Recommended acceptance order

1. Fix the two prose/output contradictions.
2. Repair the ordered-depth schematic.
3. Move tensor/dtype/device and seed semantics to first use.
4. Expose the parabolic and `prange` stages at bite-size depth.
5. Add the post-computation golden comparison.
6. Complete or explicitly re-home the cold/warm/fresh-cache timing contract.
7. Perform the density reduction without deleting scientific coverage.
8. Rebuild the notebook/HTML, then re-read every numeric sentence and both
   corrected schematics at reader width.

