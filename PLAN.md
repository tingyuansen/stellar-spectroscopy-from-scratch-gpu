# Stellar Spectroscopy from Scratch — Live Construction Plan

`BIBLE.md` defines the quality standard. `COVERAGE.md` is the scientific
completeness ledger. `design/global_chapter_contracts.md` owns chapter
boundaries, notation, and handoffs. `PASSDOWN.md` records the current
executable state. This file contains only the live work sequence.

## Read-only authority

- Payne Zero: `/Users/ysting/payne-zero`
- pinned commit: `9c44001feae40b85146630499e6f8a5fed42e5af`
- paper source: `/Users/ysting/Source_Files_Not_For_Review`
- `main.tex` SHA-256:
  `e11507b9150550b246f6664debf22e540aa92d8261eb40daabb594da91bd8e0d`

Neither external tree may be modified. Required source fragments, teaching
data, figures, and comparison products are owned and verified inside this
repository.

## Course promise

The reader begins with basic mathematics, physics, and Python and constructs
the complete one-dimensional LTE atmosphere-and-synthesis calculation:

```text
stellar labels and abundances
        ↓
learned initializer used only as a starting proposal
        ↓
NumPy + Numba multicore physical atmosphere iteration
        ↓
schema-v4 structured atmosphere
        ↓
PyTorch CUDA / MPS / CPU spectral synthesis
        ↓
wavelength, total flux, continuum flux, normalized flux
```

Payne Zero supplies the exact implementation contracts and the parity oracle.
It is not the narrative protagonist. Each component is motivated by a physical
need, derived at the appropriate level, implemented in bite-sized cells, and
tested before the production interface is introduced.

## Fifteen-chapter architecture

1. From Starlight to a First Grey Atmosphere
2. From Equations to Fast, Trustworthy Kernels and Explicit Data
3. Atoms, Ions, and Electrons
4. Molecules and Coupled Equilibrium
5. Continuous Opacity and Scattering
6. One Spectral Line
7. Atomic Line Forests and Special Profiles
8. Molecular Bands and Source Compilation
9. Radiative Transfer with Scattering
10. GPU Synthesis from a Structured Atmosphere
11. Starting and Blanketing an Atmosphere
12. Radiation, Thermodynamics, and Convection
13. Correction and the Full Numba Iteration
14. Learned Initializers and Mandatory Physical Closure
15. From Stellar Labels to a Verified Spectrum

The chapter count changes only if a global flow audit shows that material
cannot be taught clearly without a split or merge. Coverage is never removed
merely to preserve the count.

## Cadence: complete book first, then global passes

The book is maintained as a complete executable object. Work does not stop at
one locally polished chapter while later chapters remain imaginary.

1. Keep all 15 canonical notebooks executable and rendered.
2. Read the whole course in order and repair prerequisites, notation,
   redundancy, pacing, and handoffs.
3. Reconcile every atmosphere, synthesis, initializer, and emulator behavior
   against the pinned source and paper.
4. Deepen under-explained physics and implementation narratives.
5. Standardize original schematics and professional one-panel plots.
6. Repeat execution, parity, browser, provenance, and cleanliness audits.

Scientific discrepancies still block the affected claim. Metadata ceremony
and historical audit notes do not block the construction of unrelated
chapters.

## Current state

Live state, verified evidence, open defects, and the immediate action sequence
are owned by `PASSDOWN.md`. Do not duplicate them here — this file holds only
the static architecture and the standing backlog.

## Per-chapter design contract backlog

Chapters 2–6 were built to a per-chapter contract standard under `design/`.
Chapters 7–15 were carried end-to-end first and still hold only a
`first_pass_contract`. Bringing them to the same standard is standing work, not
optional ceremony.

Each chapter should own:

- `chapterNN_exact_source_contract.md` — which pinned Payne Zero symbols the
  chapter stages, and the exact-match obligations on them;
- `chapterNN_causal_outline.md` — the question-before-machinery ordering, so
  prose flow can be audited without reading the notebook;
- `chapterNN_acceptance.md` — what must be true for the chapter to be accepted,
  with unevaluated gates left visibly `None`.

Use Chapter 3 as the template: it is the cleanest complete example at five
files. Chapter 6's 49 files are not the target — that is an artifact of
iterating hardest there, and it should be consolidated rather than imitated.

The current per-chapter counts and precise gaps are tabulated in `PASSDOWN.md`.

## Resolved source boundary

1. **Resolved boundary:** Chapter 8 implements and verifies H2O compiler parity,
   but the complete textbook pipeline preserves and documents the pinned
   standard synthesis runtime's omission of H2O. It must not confuse this
   synthesis-only compiler boundary with the standard atmosphere water
   selection/deposition path.

## Repeated whole-book passes

### Pass 1 — Executable spine

Status: complete.

Every chapter has causal prose, executable cells, a summary, and an explicit
handoff or final close. The reader and chapter registry contain no planned
placeholders.

### Pass 2 — Flow and ownership

Status: first pass complete; repeat after substantive chapter edits.

For each section, verify:

- every symbol and input has been introduced;
- the question precedes the machinery that answers it;
- physics is derived once and later chapters invoke rather than rederive it;
- supplied fixtures are named at the moment they enter;
- forward references name only the missing dependency;
- chapter summaries accurately describe what executed.

### Pass 3 — Exact coverage and parity

Status: in progress.

- reconcile all 1,501 public source records;
- ensure every nonredundant atmosphere, synthesis, initializer, emulator,
  catalog, cache, schema, and CLI responsibility has a chapter owner;
- bind claims to executable invariants, analytic limits, or pinned oracles;
- retain `None` for acceptance gates that were not evaluated;
- keep exploratory initializer products distinct from converged physical
  products.

### Pass 4 — Pedagogical depth

Status: in progress.

- expand terse causal transitions, especially in Chapters 10–15;
- keep code cells small enough to explain immediately;
- remove source-file tours and large code blocks from Markdown;
- teach Numba, `njit`, `prange`, PyTorch devices, batching, and caching where
  their consequences first matter;
- interpret every output before moving on.

### Pass 5 — Visual system

Status: pending global acceptance.

- use original website-aesthetic schematics generated from owned figure
  specifications;
- prefer one scientific claim per plot and one panel when possible;
- standardize typography, colors, labels, units, legends, and accessibility;
- verify full-resolution figures and mobile/desktop reader behavior.

### Pass 6 — Final execution and cleanup

Status: pending.

- execute all chapters from a clean process in order;
- run source, data, manifest, schema, parity, and full test suites;
- inspect every rendered chapter in the reader;
- remove obsolete candidate artifacts, superseded audit debris, and stale
  planning notes only after their canonical replacements pass;
- leave only owned source, data, tests, figures, and generated reader products.

## Integration rule for parallel chapter work

Subagents may work on bounded, nonoverlapping chapters or audits. The primary
agent owns final integration and must recheck:

- global notation and prerequisites;
- single ownership of repeated concepts;
- preceding and following chapter transitions;
- exact runtime claims;
- rendered output and repository cleanliness.

No subagent draft is accepted solely because its local tests pass.

## Definition of done

The project is complete only when:

- a student can read Chapters 1–15 in order without unexplained prerequisites;
- the student can rebuild the atmosphere and synthesis pathways from the
  textbook's owned code and data contracts;
- all nonredundant Payne Zero atmosphere, synthesis, and initializer behavior
  has an explicit pedagogical owner;
- retained reference calculations match the pinned implementation under their
  declared policies;
- every scientific claim has visible evidence and every absent test remains
  visibly absent;
- all notebooks execute, all pages render, and all plots/schematics pass visual
  review;
- the repository is minimal, self-contained for reading and compact execution,
  and free of obsolete construction debris.
