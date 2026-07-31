# Chapter 3–4 neighboring-chapter audit: the overloaded public stage-5 lane

## Scope and evidence

This is a bounded, read-only audit of:

- `book/chapters/chapter_03.py`;
- `scripts/textbook_schematic_specs.py`;
- `assets/schematics/MANIFEST.json` and the rendered Chapter 3 and Chapter 4
  mapping schematics;
- `design/chapter03_exact_source_contract.md`;
- `design/chapter03_causal_outline.md`;
- `design/chapter04_pedagogical_sequence.md`;
- the pinned Payne Zero checkout at commit
  `9c44001feae40b85146630499e6f8a5fed42e5af`, especially
  `payne_zero_synthesis/pipeline.py`,
  `payne_zero_synthesis/molecular_equilibrium.py`, and the atmosphere
  synthesis bridge;
- the Chapter 3 layout tests and the current Chapter 4 synthesis-oracle
  worker/tests.

No canonical chapter, image, test, source file, or external repository was
modified in this audit.

## Verdict

The corrected Chapter 3 *atom-only* claim is scientifically sound:

- its public cube has shape `(D,6,139)`;
- columns 0–98 contain atomic values;
- columns 99–138 are unused in the atom-only route;
- the empty tail is explicitly not presented as the molecular address rule;
- later molecular destinations are described as selected stage-index-5 cells
  across the species axis;
- actual populations and populations divided by partition functions remain
  distinct.

The Chapter 3 causal exit is also strong. It ends with the precise limitation
that separate elemental ledgers fail once one molecule consumes several
elements, and the Chapter 4 opening begins from that failure without
re-teaching Boltzmann, Saha, or the atomic fixed point.

One important public-layout fact is still under-explained, however, and it
must be made explicit in Chapter 4 before canonical prose is written:

> The molecule-enabled synthesis builder does not add a separate molecular
> block to both public cubes. It leaves `ion_stage_populations` unchanged and
> overwrites selected cells only in
> `partition_normalized_populations[:, 5, :]`.

Consequently, at a mapped stage-5 coordinate, the actual cube and normalized
cube no longer describe the same physical species. The actual cube retains
the atomic sixth stored stage, while the normalized cube carries molecular
line support. This is an intentional interface overload, not a physical
ion-stage statement.

## Exact source result

The pinned synthesis pipeline first copies both public arrays from the atomic
population state:

- `partition_normalized_populations` is copied at pipeline lines 443–445;
- `ion_stage_populations` is copied at lines 446–448.

Only the normalized copy is later assigned at
`[:, 5, species_code // 6 - 1]` (lines 573–581). The actual cube is never
modified by the molecular mapping. The exact line-population helper is fed the
no-ground-floor neutral partitions and returns molecular line support, not
the raw molecular-equilibrium population vector.

The exact supported map has 54 distinct line-list species codes and 54
distinct zero-based public columns:

```text
39–73, 81, 82, 84–97, 129, 130, 131
```

Thus:

- 51 mapped columns lie inside the nominal atomic range 0–98;
- only 3 lie in the atom-only tail 99–138;
- CO follows `276 → 608 → [depth, 5, 45]`;
- describing columns 99–138 as a molecular reservation is false;
- describing all of stage index 5 as exclusively molecular is also false.

The overwrite is not merely hypothetical. The accepted pinned Chapter 3
atomic golden has nonzero atomic stage-5 values before molecular insertion
(111 nonzero cells within `[:,5,:99]` for each of the full and fixed public
states). The release mapping therefore repurposes selected occupied interface
coordinates; it does not simply fill a previously empty molecular plane.

## Findings and proposed edits

### F1 — Chapter 4 must teach normalized-only overwrite semantics

**Priority: required before Chapter 4 acceptance.**

Cell 17 currently says that molecular support occupies stage index 5 and asks
for “zero leakage into the other five lanes.” That is directionally correct
but incomplete and potentially misleading because all six lanes already
contain atomic values. “Zero leakage” must refer to the *molecular
before/after delta*, not to zero-valued public lanes.

Add the following claims to Cell 17:

1. The builder begins with atom-populated `before` cubes.
2. `ion_stage_populations` is unchanged everywhere.
3. Only
   `partition_normalized_populations[:,5,public_columns]` is overwritten.
4. Every unowned normalized cell is unchanged.
5. The molecular delta is zero in stage indices 0–4.
6. Of the 54 destinations, 51 are in columns 0–98 and 3 are in columns
   129–131.
7. At mapped cells, the actual and normalized cubes must not be paired as
   \(n\) and \(n/U\) for one species.

Suggested compact prose:

> Stage index 5 is an overloaded interface lane. The molecule-enabled release
> builder leaves the actual ion-stage cube atomic, but repurposes selected
> cells of the normalized cube for molecular line support. The molecular
> change is confined to those selected cells; it is not a new 40-column
> molecular block and not a sixth molecular ion stage.

The expected output should print the exact 51/3 split and compare the complete
pre/post arrays, not only the extracted 54 values.

### F2 — Scope Chapter 3's coordinate-pair claim explicitly to atom-only data

**Priority: small canonical edit, scientifically important.**

The body of Chapter 3 repeatedly says “in this atom-only route,” so its current
calculation is valid. The summary sentence saying that exact maps carry actual
and normalized populations between layouts is less tightly scoped. A reader
could carry the Chapter 2/3 “same coordinate, divide by \(U\)” picture into the
molecule-enabled public cube, where it fails at selected stage-5 cells.

Proposed edits:

- change “Molecular line slots remain unused here” to “No molecular line
  values have been inserted in this atom-only state”;
- after the public axes are introduced, add one boundary sentence:

  > Throughout this atom-only chapter, the actual and normalized cubes share
  > atomic coordinates. Chapter 4 will identify the selected normalized
  > stage-5 cells that the molecular-line interface later repurposes.

- in the summary, change “Exact maps carry actual and partition-normalized
  populations…” to “For the atom-only states used here, exact maps carry
  actual and partition-normalized populations…”.

This states the exception once at the representation boundary without
teaching molecular equilibrium prematurely.

### F3 — Replace the remaining “40-column molecular reservation” language

**Priority: required design-document cleanup.**

Two planning documents retain obsolete language that contradicts their own
corrected source contract:

- `design/chapter03_exact_source_contract.md` says “molecular 40-column
  reservation” in the Chapter 3 outline;
- `design/chapter03_causal_outline.md` says “pale empty 40-species molecular
  reservation” and “zero molecular reservation.”

Replace each with “unused atom-only tail 99–138.” Keep the already-correct
warning that the tail is not the molecular address rule.

### F4 — The Chapter 3 schematic is acceptable; its metadata can be sharper

**Priority: polish, no image regeneration required.**

The rendered Chapter 3 schematic is scientifically acceptable:

- the pale region is labeled “unused tail 99–138 in this atom-only route”;
- the future callout says “selected synthetic line cells”;
- the callout targets stage index 5 rather than the tail;
- the conversion is visibly an explicit gather-and-place map, not a reshape.

The prompt, image, and manifest therefore support the corrected atom-only
claim. The alt text and figure-spec caption still use the broader phrase
“synthetic line lane.” Change that phrase to “selected cells later overloaded
for synthetic molecular-line support.” The pixels already say “selected
synthetic line cells,” so this metadata correction does not require a new
asset.

### F5 — The Chapter 4 mapping schematic needs regeneration

**Priority: required visual correction.**

The Chapter 4 prompt asks for a “highlighted horizontal sheet” while also
labeling depth as the vertical axis and stage index as a horizontal axis. The
rendered image consequently highlights a constant-depth sheet but labels it
“stage index 5.” That geometry is inconsistent with its own axes. It also
colors a whole plane, which obscures the fact that only 54 selected cells are
owned by the molecular mapping.

Revise the prompt and regenerate the asset:

- make the fixed-stage-5 plane span **depth × species**, with stage index 5
  fixed;
- mark several discrete selected code-derived columns on that plane rather
  than coloring an undifferentiated sheet;
- label the destination “normalized cube only”;
- note “51 columns inside 0–98; 3 columns at 129–131” in a quiet callout;
- retain the CO trace `[depth,5,45]`;
- use the exact source expression
  `column = species_code // 6 - 1`;
- add “actual ion-stage cube unchanged.”

After regeneration, update the asset hash/dimensions and repeat the manual
scientific review. The current manifest's accepted review does not catch the
axis/plane mismatch visible in the raster.

### F6 — Reduce repeated Chapter 4 mapping promises in Chapter 3

**Priority: flow polish.**

The exact stage-5/code-derived promise appears in the Doppler section, the
layout section, the schematic caption, and the sentinel interpretation.
Retain the full statement once in Section 3.10, where the public axes have
earned it. Elsewhere, use shorter scoped reminders such as “the molecule-
enabled address rule remains deferred.” Keep the opening assumption, the
molecule-enabled handoff warning, and the final causal link; those serve
different scientific purposes and are not redundant.

### F7 — Freeze the overload in tests, not only in prose

**Priority: required acceptance gate.**

The current Chapter 3 tests correctly freeze the atom-only zero tail and
actual-versus-normalized identities. The Chapter 4 oracle worker already
constructs the right pre/post comparison internally, but the focused
acceptance tests should make the overload impossible to regress.

Add explicit gates for:

- 54 unique supported species codes and 54 unique columns;
- the exact column vector
  `39–73, 81, 82, 84–97, 129, 130, 131`;
- the 51-in-0–98 / 3-in-99–138 split;
- CO `276 → 608 → 45`;
- complete equality of the before/after actual cube;
- complete equality of every unowned normalized cell;
- molecular delta zero for stage indices 0–4;
- exact equality of mapped normalized cells to the independently computed
  no-ground-floor molecular line populations;
- a discriminating raw-CO-versus-normalized-CO check;
- schematic prompt text containing “selected cells,” “normalized cube only,”
  and “actual ion-stage cube unchanged.”

The test should never assert that stage indices 0–4 themselves are zero,
because they contain atomic populations. It should assert that their
*molecular delta* is zero.

## Neighbor-flow decision

With F1–F3 addressed, the best division of labor is:

- Chapter 3 teaches the atom-only axes, the zero tail in that route, and one
  narrowly scoped warning that molecule-enabled synthesis repurposes selected
  normalized stage-5 cells.
- Chapter 4 derives the chemistry, distinguishes raw molecular populations
  from normalized line support, proves the 54-code mapping, and exposes the
  normalized-only overwrite.
- Chapter 5 consumes the resulting structured fields without reopening either
  atomic closure or molecular mapping.

This keeps Chapter 3 honest without front-loading Chapter 4, gives Chapter 4
the one surprising interface fact it owns, and preserves the existing strong
causal transition from separate atomic ledgers to coupled molecular
equilibrium.
