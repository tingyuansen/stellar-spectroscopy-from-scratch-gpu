# Chapter 11 Acceptance Record

Acceptance date: 2026-07-31

Scope: chapter-level acceptance. Chapter 11 remains subject to the later
whole-book notation, redundancy, coverage, and capstone-parity audits.

## Outcome

Chapter 11 is accepted as the *Starting and Blanketing an Atmosphere*
construction slice. The reader builds the following in one causal sequence:

1. A seed is nine aligned columns plus explicit metadata, and validation
   answers whether the object is safe to interpret, not whether the atmosphere
   is correct.
2. The standard Rosseland array is a coordinate, and fixed-column text is a
   numerical operator.
3. The atmosphere sampling grid is distinguished from a synthesis grid.
4. One pass ends at `OpacityState`, before transfer.

There is no detached exercise section. Useful predictions, limiting cases,
timings, invalid inputs, and parity questions are resolved where they become
physically relevant.

## Gates passed

Measured 2026-07-31 against the executed notebook and the live suite.

- 6 Chapter 11 runtime tests pass.
- The complete executed notebook contains no error output, and every code cell
  carries an execution count.
- 18 visible code cells. The longest is 22 lines and the longest
  source line is 85 characters, within the 60-line soft ceiling and
  92-character limit of `BIBLE.md`.
- The chapter closes with a `## 11.N Chapter summary` heading and a literal
  `### Next:` link, as required by `design/global_chapter_contracts.md`.
- 1 quantitative figures and 3 original schematic
  references are present and render in the local reader.
- Exact source-fragment verification passes: all chapter-stage fragments match
  Payne Zero `9c44001feae40b85146630499e6f8a5fed42e5af`.
- The schematic-provenance and data-manifest suites pass.
- All `§N.M` cross-references in this chapter resolve
  (`scripts/check_section_references.py`).

## Pedagogical deepening, 2026-07-31

§11.12 now derives backwarming from the fixed-flux constraint through the
harmonic Rosseland mean, including the companion surface cooling, and names
blanketing as the reason Chapter 13's correction must exist. §11.9 now
explains why selection samples 344 reference coordinates rather than 30,000
frequencies, why the filter is deliberately conservative, and what the
duplicated entry plus sentinel buys.

## Neighbour handoff

Chapter 10 hands off a supplied schema-valid atmosphere; Chapter 11 closes by
letting radiation reshape it.

## Not evaluated

These are recorded as open rather than omitted, so this record cannot be read
as a stronger claim than the evidence supports.

- **No independent pedagogy audit has been run for this chapter.** Chapters 2-4
  carry one; Chapters 7-15 do not. This record is therefore a measured-gate
  acceptance, not an audited one.
- No chapter-level golden-parity record exists beyond the coverage already
  provided by the whole-book pinned-source and solar-oracle verification.
- Rendered typography and figure legibility were reviewed in the local reader
  during construction but are not separately certified here.
