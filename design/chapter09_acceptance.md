# Chapter 9 Acceptance Record

Acceptance date: 2026-07-31

Scope: chapter-level acceptance. Chapter 9 remains subject to the later
whole-book notation, redundancy, coverage, and capstone-parity audits.

## Outcome

Chapter 9 is accepted as the *Radiative Transfer with Scattering* construction
slice. The reader builds the following in one causal sequence:

1. Equal extinction with different absorption/scattering division produces
   different emergent \(H_\nu\), proving geometry alone is insufficient.
2. The formal solution, angular moments, and contribution function are built
   before the fixed operator is introduced.
3. Source iteration solves the static feedback on one reusable 51-point grid.
4. Total and continuum rows share one launch, and zero line opacity returns
   normalized flux one.

There is no detached exercise section. Useful predictions, limiting cases,
timings, invalid inputs, and parity questions are resolved where they become
physically relevant.

## Gates passed

Measured 2026-07-31 against the executed notebook and the live suite.

- 11 Chapter 9 runtime tests pass.
- The complete executed notebook contains no error output, and every code cell
  carries an execution count.
- 18 visible code cells. The longest is 42 lines and the longest
  source line is 88 characters, within the 60-line soft ceiling and
  92-character limit of `BIBLE.md`.
- The chapter closes with a `## 9.N Chapter summary` heading and a literal
  `### Next:` link, as required by `design/global_chapter_contracts.md`.
- 5 quantitative figures and 4 original schematic
  references are present and render in the local reader.
- Exact source-fragment verification passes: all chapter-stage fragments match
  Payne Zero `9c44001feae40b85146630499e6f8a5fed42e5af`.
- The schematic-provenance and data-manifest suites pass.
- All `§N.M` cross-references in this chapter resolve
  (`scripts/check_section_references.py`).

## Pedagogical deepening, 2026-07-31

§9.2 now states the full source function \(S_\nu=(1-\alpha_\nu)S_{\nu,\rm
th}+\alpha_\nu J_\nu\), which is what makes the problem nonlocal and therefore
why §9.7 must iterate, and derives the thermalization depth and
\(\sqrt{1-\alpha_\nu}\) surface law. §9.14 adds equivalent width and the curve
of growth; its three regimes were verified numerically rather than asserted
(\(W/(\sqrt\pi\tau_0)\to1.0\) for \(\tau_0\le0.1\); \(W\) grows 3.4 to 9.4
while \(\tau_0\) spans \(10^1\) to \(10^3\); \(W/\sqrt{\tau_0}\approx0.26\)
for \(\tau_0\ge10^4\)).

## Neighbour handoff

Chapter 8 hands off compiled molecular sources; Chapter 9 closes by keeping
the whole spectrum on the right device.

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
