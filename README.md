# Stellar Spectroscopy from Scratch

This is a self-contained, executable textbook for building the atmosphere and spectral-synthesis
parts of Payne Zero from physical principles. It is written for a final-year undergraduate or
first-year graduate student who knows basic mathematics, physics, and Python but has not studied
stellar spectroscopy, radiative transfer, Numba, GPUs, or machine-learning initializers.

The book is organized as a half-semester course of fifteen substantial chapters. A lossless
29-unit internal map keeps every non-redundant physics branch, data boundary, implementation
choice, and public Payne Zero source object assigned while the reader sees a coherent course.

The implementation follows the working architecture:

- NumPy and Numba on multicore CPUs for physical atmosphere iteration;
- PyTorch on CUDA, Apple Metal, or CPU for broad spectral synthesis;
- Torch atmosphere initializers that provide starting structures, never physical acceptance.

Every important idea follows the same rhythm: physical question, concrete limit, derivation with
defined symbols, bite-sized canonical code, physical checks, and Payne Zero parity. Useful
variations and debugging checks are resolved inside that teaching sequence; there are no detached
exercise sets. Conceptual schematics use the hand-sketched visual language of the official Payne
Zero website; quantitative figures are generated from executed chapter code.

## Current construction status

All fifteen chapters exist, execute, and render; the reader publishes Chapters 1–15. The source
inventory maps 58 modules and 1,501 public exports, routines, classes, fields, and named source
objects to a chapter and verification gate. The exact solar atmosphere converges and reproduces
the pinned oracle bitwise. Remaining work is pedagogical depth and density balance rather than
missing machinery.

`PASSDOWN.md` is authoritative for live state and is the correct entry point for anyone — human
or agent — picking the project up.

Read these files in order:

1. [PASSDOWN.md](PASSDOWN.md) — live state, open defects, and the next concrete actions;
2. [BIBLE.md](BIBLE.md) — pedagogical, scientific, visual, and verification standard;
3. [PLAN.md](PLAN.md) — fifteen-chapter course and lossless detailed topic map;
4. [COVERAGE.md](COVERAGE.md) — package/module/feature ownership.

## Build and test

```bash
python3 -m pytest
python3 scripts/verify_pinned_source_fragments.py
python3 scripts/audit_paynezero_source.py
python3 scripts/build_symbol_coverage.py
python3 scripts/build_book.py
```

`scripts/build_book.py` is the canonical source → notebook → executed notebook → HTML build. The
same Python registry generates the web navigation. Available chapters are written to `content/`.
The source-fragment gate verifies that code displayed as Payne Zero is AST-identical—or, for
complete copied files, byte-identical—to the pinned read-only checkout.

To read the local site:

```bash
python3 -m http.server 8765
```

Then open `http://127.0.0.1:8765/`.

## Ground truth and scope

The book is checked against the pinned Payne Zero source at commit
`9c44001feae40b85146630499e6f8a5fed42e5af` and the pinned paper source recorded in `PLAN.md`.
Those source trees are read-only and are never runtime dependencies of the taught path.

Spectrum fitting is outside the main scope. The book covers the complete non-redundant atmosphere
solver, atmosphere initializers, structured atmosphere interface, and synthesis calculation,
including their data, precision, caching, thread/device, convergence, provenance, and failure-mode
contracts.

Yuan-Sen Ting — Max Planck Institute for Astronomy & The Ohio State University.
