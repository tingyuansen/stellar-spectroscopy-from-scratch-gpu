# Chapter 14 Acceptance Record

Acceptance date: 2026-07-31

Scope: chapter-level acceptance. Chapter 14 remains subject to the later
whole-book notation, redundancy, coverage, and capstone-parity audits.

## Outcome

Chapter 14 is accepted as the *Learned Initializers and Mandatory Physical
Closure* construction slice. The reader builds the following in one causal
sequence:

1. A converged atmosphere is a fixed point; a learned model proposes a
   starting state and cannot certify one.
2. The three label families share a target shape but not a trained decoder,
   and equal array dimensions do not make learned arrays interchangeable.
3. The direct-abundance route remains experimental and carries a larger safety
   contract.

There is no detached exercise section. Useful predictions, limiting cases,
timings, invalid inputs, and parity questions are resolved where they become
physically relevant.

## Gates passed

Measured 2026-07-31 against the executed notebook and the live suite.

- 17 Chapter 14 runtime tests pass.
- The complete executed notebook contains no error output, and every code cell
  carries an execution count.
- 20 visible code cells. The longest is 27 lines and the longest
  source line is 75 characters, within the 60-line soft ceiling and
  92-character limit of `BIBLE.md`.
- The chapter closes with a `## 14.N Chapter summary` heading and a literal
  `### Next:` link, as required by `design/global_chapter_contracts.md`.
- 6 quantitative figures and 1 original schematic
  reference are present and render in the local reader.
- Exact source-fragment verification passes: all chapter-stage fragments match
  Payne Zero `9c44001feae40b85146630499e6f8a5fed42e5af`.
- The schematic-provenance and data-manifest suites pass.
- All `§N.M` cross-references in this chapter resolve
  (`scripts/check_section_references.py`).
- `scripts/verify_chapter14_artifacts.py` reports verified, with exact restart
  status `ready`.

## Pedagogical deepening, 2026-07-31

§14.2 now explains what PCA does and why atmosphere profiles compress at all,
closing on the caveat that carries the chapter: discarded directions are
negligible only for the training distribution, so an out-of-distribution
request returns a confident, smooth, wrong profile with no internal warning.
§14.5 explains why the in-memory route discards precision it holds, framed as
the same rule as Chapter 13's terminal \(Q\); §14.7 explains why element
responses are summed.

## Neighbour handoff

Chapter 13 hands off the exact iteration; Chapter 14 closes by carrying one
verified atmosphere into synthesis.

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
