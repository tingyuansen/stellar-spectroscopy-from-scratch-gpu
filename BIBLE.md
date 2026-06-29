# GPU Textbook Lecture Standard

This is the compact standard for every lecture. `PASSDOWN.md` is the live handoff; this file is the
quality bar.

## Scope

The GPU textbook is a standalone torch/MPS/CUDA reconstruction of the pykurucz/kgpu physics. A
reader should be able to understand and debug the production kernels from the lectures without
importing `kgpu`, `pykurucz`, or the NumPy textbook in the taught path.

Reference files are allowed only as:

- immutable physical/static input data;
- scoped fixtures explicitly marked as such;
- comparison targets loaded after the computation.

Do not present a loaded computed field as if the notebook computed it.

## Required Lecture Pattern

Each lecture should contain:

1. the physics goal and tensor shapes;
2. the torch implementation, with device selection once near the top;
3. comments/docstrings at nontrivial kernels, especially where scalar order or fixture boundaries
   matter;
4. a parity cell that computes first, then compares against the reference target;
5. a short statement of precision floor and remaining boundary.

## GPU-Native Standard

- Prefer torch tensors on the selected device for taught compute paths.
- Vectorize over depth, wavelength, line, or species axes where that preserves the algorithm.
- Use `torch.where`, masks, `gather`, `searchsorted`, reductions, and precomputed invariants instead
  of Python loops when the loop is only elementwise selection or arithmetic.
- Keep scalar loops only for structural recurrences, table parsing, exact-order accumulation,
  boundary stencils, or small fixed heterogeneous tables. State why the loop is scalar.
- Avoid `.cpu()`, `.numpy()`, `.item()`, and `.tolist()` inside compute paths. They are fine in
  comparison, plotting, reference-loading, or explicit host-boundary cells.
- Use NumPy only for reference/comparison cells, static table preparation, or boundary material that
  is explicitly not the taught GPU compute path.

## Pedagogy

- Break long kernels into readable helpers when the split does not hide shared state or change
  parity.
- Avoid consecutive dense code cells without a markdown bridge.
- Do not add marketing copy, “GPU edition” framing, or claims that exceed the computed path.
- Keep schematics and plots current with the actual implemented path.

## Verification

After any lecture edit:

```bash
python _pipeline/build_lecture<N>_gpu.py  # or the lecture's actual builder
python _pipeline/build.py <N>
```

Then run the smallest matching verifier. For shared or capstone changes, rerun the relevant full
ledger in `PASSDOWN.md`.

Before claiming handoff quality, run:

```bash
git diff --check
python _pipeline/build.py 1 2 3 4 5 6 7 8
python _pipeline/build.py 9 10 11 12 13 14 15 16
```

and the verifier suite listed in `PASSDOWN.md`.
