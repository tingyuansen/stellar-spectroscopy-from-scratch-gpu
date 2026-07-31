# Chapter 7 Acceptance Record

Acceptance date: 2026-07-31

Scope: chapter-level acceptance. Chapter 7 remains subject to the later
whole-book notation, redundancy, coverage, and capstone-parity audits.

## Outcome

Chapter 7 is accepted as the *Atomic Line Forests and Special Profiles*
construction slice. The reader builds the following in one causal sequence:

1. A catalog row becomes a kernel input with explicit physical meaning.
2. Route codes select physics rather than branching on convenience.
3. Selection happens at three distinct physical stages.
4. `njit` and `prange` accelerate independent decisions, with scatter-add for
   overlapping deposits and explicit ownership rather than hopeful scheduling.
5. Hydrogen and deuterium are treated as series, and dense plasma dissolves
   the top of a series.
6. Helium and autoionizing resonances keep their source-specific meanings.

There is no detached exercise section. Useful predictions, limiting cases,
timings, invalid inputs, and parity questions are resolved where they become
physically relevant.

## Gates passed

Measured 2026-07-31 against the executed notebook and the live suite.

- 10 Chapter 7 runtime tests pass.
- The complete executed notebook contains no error output, and every code cell
  carries an execution count.
- 18 visible code cells. The longest is 43 lines and the longest
  source line is 88 characters, within the 60-line soft ceiling and
  92-character limit of `BIBLE.md`.
- The chapter closes with a `## 7.N Chapter summary` heading and a literal
  `### Next:` link, as required by `design/global_chapter_contracts.md`.
- 6 quantitative figures and 3 original schematic
  references are present and render in the local reader.
- Exact source-fragment verification passes: all chapter-stage fragments match
  Payne Zero `9c44001feae40b85146630499e6f8a5fed42e5af`.
- The schematic-provenance and data-manifest suites pass.
- All `§N.M` cross-references in this chapter resolve
  (`scripts/check_section_references.py`).

## Pedagogical deepening, 2026-07-31

§7.15 now derives the Inglis-Teller exponent from Stark splitting against
level spacing, so the \(n_e^{-2/15}\) density dependence is earned rather than
presented as a fitted constant.

## Neighbour handoff

Chapter 6 hands off one trustworthy line; Chapter 7 closes by showing that
molecular sources do not share the atomic encoding.

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
