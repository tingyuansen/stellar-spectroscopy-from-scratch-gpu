# Chapter 1 Acceptance Record

Acceptance date: 2026-07-31

Scope: chapter-level acceptance. Chapter 1 remains subject to the later
whole-book notation, redundancy, coverage, and capstone-parity audits.

## Outcome

Chapter 1 is accepted as the *From Starlight to a First Grey Atmosphere*
scaffold slice. The reader builds the following in one causal sequence:

1. An observed spectrum poses the question, and the atmosphere is shown to
   come before the spectrum.
2. The photosphere becomes a stack of layers with explicit units, constants,
   and one optical photon to anchor them.
3. Thermal radiation supplies the Planck function; optical depth measures the
   difficulty of escape; LTE supplies a local source rather than a global
   blackbody.
4. Radiative equilibrium gives a grey temperature law, Rosseland depth puts
   every layer on one grid, and hydrostatic balance supplies a pressure scale.
5. The scaffold is then read back as a causal chain, with its own
   inconsistency stated plainly.

There is no detached exercise section. Useful predictions, limiting cases,
timings, invalid inputs, and parity questions are resolved where they become
physically relevant.

## The chapter's distinctive role

Chapter 1 is the only chapter that deliberately stages almost no pinned Payne
Zero source. It builds a transparent grey scaffold from first principles and
declares openly that the result is not self-consistent — a real atmosphere
needs an equation of state, real opacities, and iteration, none of which exist
yet. Every later chapter replaces one of its controlled assumptions. Judged as
a source-staging chapter it would look empty; judged as the chapter that earns
the reader's first vocabulary, it is doing exactly its job.

The controlled limit \(\kappa_{\rm Ross}=1\ {\rm cm^2\,g^{-1}}\) is the clearest
example: it is declared as a placeholder, used to attach a pressure scale to the
grey temperature, and retired once Chapter 5 supplies real continuous opacity.

## Gates passed

Measured 2026-07-31 against the executed notebook and the live suite.

- 12 Chapter 1 tests pass across 2 test files.
- The complete executed notebook contains no error output, and every code cell
  carries an execution count.
- 15 visible code cells across 12 numbered sections. The longest is 31 lines
  and the longest source line is 87 characters, within the 60-line soft ceiling
  and 92-character limit of `BIBLE.md`.
- **No fenced code block appears in any Markdown cell.** This was the central
  finding of `design/chapter01_rewrite_audit.md`, which measured 151 lines of
  source across five Markdown cells; the count is now zero.
- The chapter closes with a `## 1.12 Chapter summary` heading and a literal
  `### Next:` link, as required by `design/global_chapter_contracts.md`.
- 5 quantitative figures and 2 original schematic references are present and
  render in the local reader.
- The schematic-provenance and data-manifest suites pass.
- All `§N.M` cross-references in this chapter resolve
  (`scripts/check_section_references.py`).

## Audit history

Chapter 1 is the most heavily audited chapter in the book, and unusually its
audits are *older* than the text they describe:

- `chapter01_rewrite_audit.md` returned "executable but not yet acceptable as
  the style prototype", objecting to removed reasoning steps and to source code
  embedded in Markdown. The Markdown-code objection is measurably resolved.
- `chapter01_flow_audit.md` found the physical spine sound and listed flow
  problems, beginning with the chapter opening on a syllabus rather than on the
  observable. The current §1.1 opens on the observed spectrum.
- `chapter01_02_03_continuity_audit.md` covers the handoff into Chapters 2
  and 3 and is the most recent of the three.

## Neighbour handoff

Chapter 1 is the entry point and has no predecessor. It closes by making the
scaffold's inadequacy the motivation for Chapter 2: the calculation must be
made trustworthy and fast before more physics is added.

## Not evaluated

These are recorded as open rather than omitted, so this record cannot be read
as a stronger claim than the evidence supports.

- **The two Chapter 1 audits predate the current text and have not been re-run
  against it.** Their headline findings appear addressed, but "appears
  addressed" is not the same as an audit that returned ACCEPT. Treat them as
  historical, and re-run rather than cite them if a fresh verdict is needed.
- No chapter-level golden-parity record exists, and none is expected: this
  chapter stages no exact Payne Zero product to compare against.
- Rendered typography and figure legibility were reviewed in the local reader
  during construction but are not separately certified here.
