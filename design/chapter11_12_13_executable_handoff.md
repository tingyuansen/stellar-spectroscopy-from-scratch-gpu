# Chapter 11 → 12 → 13 executable handoff

## Decision

Use Chapter 11's exact cached `OpacityState` directly. Do not invent a sliced
depth grid, partial-frequency finalization, or serialized surrogate.

Chapter 12 may retain its six-layer state as a teaching microfixture for
one-frequency deposits, reduction grouping, restoration, and convection. Its
chapter-level handoff, however, must separately run the complete Chapter 11
state through the canonical

```text
accumulate_transfer_state
→ finalize_transfer_state
→ IterationFinalization
```

Chapter 13 then consumes that same finalization through the canonical
`remap_finalized_iteration_state` boundary. Its analytic correction fixture
remains useful for isolating correction terms and exact golden parity, but it
is no longer presented as the Chapter 12 handoff.

## Measured envelope

Measurements used the repository's 80-layer, 30,000-frequency Chapter 11
state with molecules enabled and the exact staged runner:

| Cache state | Opacity | Transfer | Finalize | Remap | Total | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Existing Numba cache | 3.75 s | 0.44 s | 0.03 s | 0.01 s | 4.23 s | 809 MB |
| Empty Numba cache | 28.77 s | 6.58 s | 0.26 s | <0.01 s | 35.61 s | 919 MB |

The four dominant opacity slabs occupy 67.2 MB. The larger process peak is
already paid by Chapter 11's exact population, molecule, catalog, and compiled
kernel state. The composed transfer/remap adds no new prohibitive allocation.

This is comfortably below the book builder's 900-second chapter timeout.
Because the exact path is practical, a compact derived state would add
scientific ambiguity without solving a measured problem.

## Runtime seams

- `book.chapter12_runtime.chapter11_opacity_state()` returns Chapter 11's
  cached exact state.
- `book.chapter12_runtime.chapter11_iteration_finalization()` runs the two
  canonical Chapter 12 runner calls exactly once per process.
- Small immutable checkpoints expose shapes, dtypes, provenance, finiteness,
  and aliasing without copying opacity slabs.
- `book.chapter13_runtime.chapter12_handoff_checkpoint()` consumes the live
  `IterationFinalization`, calls the canonical remapper, and reports the
  corrected and remapped state invariants.

No package-runner, source-manifest, registry, or Chapter 1–10 change is
required.

## Notebook treatment

Keep cells bite-sized by replacing existing handoff/status cells rather than
adding a second derivation:

1. Chapter 12 opens with the exact 80-layer Chapter 11 handoff checkpoint.
   It labels the six-layer object as a local teaching microfixture.
2. Chapter 12's final handoff cell reports the exact 80-layer
   `IterationFinalization`.
3. Chapter 13 adds one compact early cell that opens that finalization and
   remaps it. The analytic fixture then follows as a controlled parity lens.

