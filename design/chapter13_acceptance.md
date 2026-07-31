# Chapter 13 Acceptance Record

Acceptance date: 2026-07-31

Scope: chapter-level acceptance. Chapter 13 remains subject to the later
whole-book notation, redundancy, coverage, and capstone-parity audits.

## Outcome

Chapter 13 is accepted as the *Correction and the Full Numba Iteration*
construction slice. The reader builds the following in one causal sequence:

1. A correction proposes a new native-grid temperature and column mass; a
   remap places every carried field on one standard grid. Only the complete
   unquantized state may begin another pass.
2. The deliberately unsafe opening proposal shows why "the residual decreased"
   is not sufficient.
3. Quantization is applied exactly once after the loop, because applying it
   inside would change the fixed-point map.

There is no detached exercise section. Useful predictions, limiting cases,
timings, invalid inputs, and parity questions are resolved where they become
physically relevant.

## Gates passed

Measured 2026-07-31 against the executed notebook and the live suite.

- 15 Chapter 13 runtime tests pass.
- The complete executed notebook contains no error output, and every code cell
  carries an execution count.
- 15 visible code cells. The longest is 23 lines and the longest
  source line is 71 characters, within the 60-line soft ceiling and
  92-character limit of `BIBLE.md`.
- The chapter closes with a `## 13.N Chapter summary` heading and a literal
  `### Next:` link, as required by `design/global_chapter_contracts.md`.
- 4 quantitative figures and 2 original schematic
  references are present and render in the local reader.
- Exact source-fragment verification passes: all chapter-stage fragments match
  Payne Zero `9c44001feae40b85146630499e6f8a5fed42e5af`.
- The schematic-provenance and data-manifest suites pass.
- All `§N.M` cross-references in this chapter resolve
  (`scripts/check_section_references.py`).
- `scripts/verify_chapter13_artifacts.py` reports verified.

## Pedagogical deepening, 2026-07-31

§§13.2-13.3 now derive the temperature correction: the Eddington closure and
\(dJ/d\tau_{\rm R}=3H\) give the deep integral form, showing why the flux term
must be an integral, and the failure of that argument near the surface earns
the lambda diagonal and the separate boundary repair. §13.5 now names the
failure each of the eight ordered safeguards prevents and explains why the
order is load-bearing; §13.8 explains the convergence slice, the max-norm, and
why convergence must be consecutive; §13.9 explains why the chunk reduction is
ordered.

## Neighbour handoff

Chapter 12 hands off an `IterationFinalization`; Chapter 13 closes by entering
the solver's basin without changing the solver.

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
