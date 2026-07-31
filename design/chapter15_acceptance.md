# Chapter 15 Acceptance Record

Acceptance date: 2026-07-31

Scope: chapter-level acceptance. Chapter 15 remains subject to the later
whole-book notation, redundancy, coverage, and capstone-parity audits.

## Outcome

Chapter 15 is accepted as the *From Stellar Labels to a Verified Spectrum*
construction slice. The reader builds the following in one causal sequence:

1. Exact types prevent an initializer proposal from being promoted to a
   converged product, and the two workflows are chosen by the question being
   asked rather than by convenience.
2. Four regime requests share one algorithm; only labels, the molecular
   switch, and the initializer family vary.
3. The exploratory acceptance matrix contains deliberate blanks, and
   unevaluated gates stay `None` rather than being promoted.
4. One exact solar solve closes the implementation loop, and reproducibility
   is demonstrated as an identity rather than a runtime anecdote.

There is no detached exercise section. Useful predictions, limiting cases,
timings, invalid inputs, and parity questions are resolved where they become
physically relevant.

## Gates passed

Measured 2026-07-31 against the executed notebook and the live suite.

- 17 passed across 2 test files.
- The complete executed notebook contains no error output, and every code cell
  carries an execution count.
- 15 visible code cells. The longest is 38 lines and the longest
  source line is 78 characters, within the 60-line soft ceiling and
  92-character limit of `BIBLE.md`.
- The chapter closes with a `## 15.N Chapter summary` heading and, as the terminal chapter, correctly omits a next link.
- 1 quantitative figure and 1 original schematic
  reference are present and render in the local reader.
- Exact source-fragment verification passes: all chapter-stage fragments match
  Payne Zero `9c44001feae40b85146630499e6f8a5fed42e5af`.
- The schematic-provenance and data-manifest suites pass.
- All `§N.M` cross-references in this chapter resolve
  (`scripts/check_section_references.py`).
- `scripts/verify_chapter15_artifacts.py` reports verified, with 12 identities
  across 4 requests.

## Neighbour handoff

Chapter 14 hands off a learned start that still requires closure; Chapter 15
is the terminal chapter and correctly omits a next link.

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
