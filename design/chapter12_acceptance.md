# Chapter 12 Acceptance Record

Acceptance date: 2026-07-31

Scope: chapter-level acceptance. Chapter 12 remains subject to the later
whole-book notation, redundancy, coverage, and capstone-parity audits.

## Outcome

Chapter 12 is accepted as the *Radiation, Thermodynamics, and Convection*
construction slice. The reader builds the following in one causal sequence:

1. The 80-layer, 30,000-frequency `OpacityState` from Chapter 11 is executed
   at both ends of the chapter, with a six-layer microfixture between them
   that calls the same canonical routines.
2. Nine numerical deposits are kept distinct rather than collapsed into one
   quantity.
3. Grouping changes last bits, not the physical contract.
4. Four perturbed EOS closures form one transaction, and molecular closure is
   an explicit branch.

There is no detached exercise section. Useful predictions, limiting cases,
timings, invalid inputs, and parity questions are resolved where they become
physically relevant.

## Gates passed

Measured 2026-07-31 against the executed notebook and the live suite.

- 10 Chapter 12 runtime tests pass.
- The complete executed notebook contains no error output, and every code cell
  carries an execution count.
- 21 visible code cells. The longest is 27 lines and the longest
  source line is 74 characters, within the 60-line soft ceiling and
  92-character limit of `BIBLE.md`.
- The chapter closes with a `## 12.N Chapter summary` heading and a literal
  `### Next:` link, as required by `design/global_chapter_contracts.md`.
- 2 quantitative figures and 2 original schematic
  references are present and render in the local reader.
- Exact source-fragment verification passes: all chapter-stage fragments match
  Payne Zero `9c44001feae40b85146630499e6f8a5fed42e5af`.
- The schematic-provenance and data-manifest suites pass.
- All `§N.M` cross-references in this chapter resolve
  (`scripts/check_section_references.py`).
- `scripts/verify_chapter12_source.py` reports 4 exact files and 6 extracted
  symbols verified.

## Pedagogical deepening, 2026-07-31

§§12.19-12.21 now derive the Schwarzschild criterion from the parcel buoyancy
argument, the adiabatic gradient from the first law (including why partial
ionization lowers it below 2/5), and the \((\nabla-\nabla_{\rm ad})^{3/2}\)
enthalpy flux. A figure evaluates the criterion on the teaching fixture and
correctly reports 0 unstable layers of 6, which the following prose explains
rather than hides. This also supplies the causal link that was missing between
the perturbed-EOS machinery of §§12.14-12.18 and convection.

## Neighbour handoff

Chapter 11 hands off `OpacityState`; Chapter 12 closes by turning imbalance
into a safe new atmosphere.

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
