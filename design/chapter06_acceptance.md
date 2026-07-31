# Chapter 6 Acceptance Record

Acceptance date: 2026-07-31

Scope: chapter-level acceptance. Chapter 6 remains subject to the later
whole-book notation, redundancy, coverage, and capstone-parity audits.

## Outcome

Chapter 6 is accepted as the *One Spectral Line* construction slice. The
reader builds the following in one causal sequence:

1. The integrated strength is established before any profile shape is chosen,
   so the \(gf\), population, and density scalings can be predicted and then
   tested.
2. The Gaussian core and Lorentzian wings are derived from thermal motion and
   finite lifetimes respectively, then convolved into the Voigt profile.
3. FASTEX and Harris tables replace the transparent calculation one piece at a
   time, with each replacement checked against what it changed.
4. The exact CPU-atmosphere and device-synthesis lanes are shown to agree at
   their declared boundaries.

There is no detached exercise section. Useful predictions, limiting cases,
timings, invalid inputs, and parity questions are resolved where they become
physically relevant.

## Gates passed

Measured 2026-07-31 against the executed notebook and the live suite.

- 215 passed, 1 skipped across 13 test files.
- The complete executed notebook contains no error output, and every code cell
  carries an execution count.
- 15 visible code cells. The longest is 40 lines and the longest
  source line is 92 characters, within the 60-line soft ceiling and
  92-character limit of `BIBLE.md`.
- The chapter closes with a `## 6.N Chapter summary` heading and a literal `### Next:` link, as required by
  `design/global_chapter_contracts.md`.
- 4 quantitative figures and 4 original schematic
  references are present and render in the local reader.
- Exact source-fragment verification passes: all chapter-stage fragments match
  Payne Zero `9c44001feae40b85146630499e6f8a5fed42e5af`.
- The schematic-provenance and data-manifest suites pass.
- All `§N.M` cross-references in this chapter resolve
  (`scripts/check_section_references.py`).
- Published Chapter 6 atmosphere and synthesis comparison products verify
  through read-only postpublication checks.
- Both published artifacts require single-link mode `0600`. Git cannot record
  that bit, so `scripts/restore_publication_modes.py` must be run after a fresh
  clone or any checkout that rewrites `data/`.

## Neighbour handoff

Chapter 5 hands off the continuum; Chapter 6 closes by moving from one
trustworthy line to an atomic forest.

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
