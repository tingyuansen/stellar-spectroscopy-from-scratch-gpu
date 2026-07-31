# Chapter 5 Acceptance Record

Acceptance date: 2026-07-31

Scope: chapter-level acceptance. Chapter 5 remains subject to the later
whole-book notation, redundancy, coverage, and capstone-parity audits.

## Outcome

Chapter 5 is accepted as the *Continuous Opacity and Scattering* construction
slice. The reader builds the following in one causal sequence:

1. Continuum absorbers and scatterers are built one physical process at a
   time, with H\(^-\) established as the dominant optical source in solar-type
   and cooler stars.
2. Each source is checked against a limiting case before the production
   interface is introduced.
3. Scattering is kept distinct from true absorption, because Chapter 9 will
   need that division.

There is no detached exercise section. Useful predictions, limiting cases,
timings, invalid inputs, and parity questions are resolved where they become
physically relevant.

## Gates passed

Measured 2026-07-31 against the executed notebook and the live suite.

- 105 passed, 1 skipped across 6 test files.
- The complete executed notebook contains no error output, and every code cell
  carries an execution count.
- 16 visible code cells. The longest is 35 lines and the longest
  source line is 88 characters, within the 60-line soft ceiling and
  92-character limit of `BIBLE.md`.
- The chapter closes with a `## 5.N Chapter summary` heading and a literal `### Next:` link, as required by
  `design/global_chapter_contracts.md`.
- 6 quantitative figures and 4 original schematic
  references are present and render in the local reader.
- Exact source-fragment verification passes: all chapter-stage fragments match
  Payne Zero `9c44001feae40b85146630499e6f8a5fed42e5af`.
- The schematic-provenance and data-manifest suites pass.
- All `§N.M` cross-references in this chapter resolve
  (`scripts/check_section_references.py`).
- Published Chapter 5 golden artifacts verify through their read-only publisher,
  and the manifest prepublication suite passes.

## Neighbour handoff

Chapter 4 hands off the coupled molecular equilibrium; Chapter 5 closes by
giving one transition a strength and a shape.

## Not evaluated

These are recorded as open rather than omitted, so this record cannot be read
as a stronger claim than the evidence supports.

- **No independent pedagogy audit has been run for this chapter.** Chapters 2-4
  carry one; Chapters 5-15 do not. This record is therefore a measured-gate
  acceptance, not an audited one.
- No chapter-level golden-parity record exists beyond the coverage already
  provided by the whole-book pinned-source and solar-oracle verification.
- Rendered typography and figure legibility were reviewed in the local reader
  during construction but are not separately certified here.
