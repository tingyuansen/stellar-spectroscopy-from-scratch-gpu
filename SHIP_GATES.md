# GPU textbook ship gates

This file is the acceptance ledger for the GPU textbook. `PASSDOWN.md` explains the state in prose;
this file lists the gates a future agent must satisfy before claiming closure.

Priority order:

1. no hidden `kgpu`/`pykurucz` runtime dependence in taught lecture paths;
2. parity with shipped reference outputs to the documented CPU/GPU floor;
3. faithful coverage of the production `kgpu` implementation pieces;
4. optimization only after the first three are stable.

## Current audit snapshot

Generated from source inspection and local runs on 2026-06-29.

### Import and fixture boundary

- Executed lecture notebooks/builders do not import `kgpu` or `pykurucz`.
- Reference-generation and verification tools under `_pipeline/make_*`, `_pipeline/verify_*`, and
  `_pipeline/converge_fromscratch.py` may read production/reference code as offline provenance
  tools. They are not the taught notebook path.
- Checked-in fixture files are allowed when classified honestly as physical input, scoped fixture,
  or comparison target. Loaded fixtures must not masquerade as notebook-computed values.
- Physical/static tables and line-list metadata must stay distinct from computed answer/state
  fixtures.

### Lecture status

| Lectures | Status | Ship interpretation |
|---|---|---|
| L1-L7 | GPU builders, executed parity cells | Core microphysics and opacity primitives are torch-based and validated against shipped references. |
| L8 | torch-native builder; data-boundary debt | JOSH kernels run in torch/MPS and import no production solver, but the taught path still consumes scoped opacity slabs and fixed operator tables. |
| L9-L13 | GPU builders | Atmosphere primitives, molecular equilibrium, molecular bands, and molecular continuum are GPU-textbook form with named reference-input boundaries. |
| L14 | non-torch builder | Honest same-atmosphere synthesis capstone. It computes opacity/transfer but loads atmosphere/EOS/population/Doppler state. Not final stellar-parameters-to-spectrum closure. |
| L15 | GPU builder with in-notebook scalar LINOP1 gate | Teaching-window deposit is computed in the taught path. Strict boundary remains open because atmosphere/EOS/window/continuum/full-grid blanket fixtures are loaded. |
| L16 | GPU builder with helper-backed closure debt | EOS-derived state is recomputed for a loaded atmosphere fixture. PFSAHA/finite-difference/continuum/molecular helpers remain clean-room NumPy, not final all-torch lecture cells. |

## Required gates before calling the textbook passdown-ready

- Run the full notebook ledger after material lecture edits:
  `python _pipeline/build.py 1 2 3 4 5 6 7 8`
  and
  `python _pipeline/build.py 9 10 11 12 13 14 15 16`.
- Run the focused physics gates:
  `python _pipeline/verify_josh.py`,
  `python _pipeline/verify_leankurucz.py`,
  `python _pipeline/verify_lineblanket.py`,
  `python _pipeline/verify_molecules.py`,
  `python _pipeline/verify_nmolec.py`,
  `python _pipeline/verify_mol_continuum.py`,
  `python _pipeline/verify_convec_gaps.py`,
  `python _pipeline/verify_converged.py`.
- Run the code-cell hygiene audit: no generated code cell should start or end with a blank line.
- Run `git diff --check`.
- Keep `reference/MANIFEST_L14_L16.md` current whenever L14-L16 arrays are added, removed, or
  reclassified.
- For prose quality, review the lectures directly. API critics are optional aids only; they are not
  the authority for closure.

## Final closure work

- Close L8's remaining data boundary: feed opacity slabs from the Lecture 3-6 torch outputs and
  derive/generate or explicitly vendor-provenance the fixed JOSH operator tables.
- Promote L14 to a torch/MPS capstone and wire/inline the L15/L16 atmosphere + EOS-state path so
  the Sun state is regenerated rather than loaded.
- Extend non-solar capstone coverage from SYNTHE on warm-started structures to full
  atmosphere+spectrum once the corresponding `kgpu` four-star ATLAS12+SYNTHE gate exists.
- Replace L15's production-derived atmosphere/EOS/window/continuum/full-grid blanket fixtures and
  L16's loaded-atmosphere/helper-backed pieces with self-contained torch-native lecture cells where
  practical.
- Break dense L13/L14/L15 code cells into smaller commented/docstringed units with interleaved
  markdown, preserving parity after each change.

## Optimization role

After correctness/no-leakage gates are agreed, the textbook becomes the readable kernel source for
optimizing `kgpu`. Each speed change in `kgpu` should trace back to a lecture-level computation and
be checked against the matching lecture/reference parity target before adoption in production.

The production optimization target belongs to `kgpu`: full 4000-9000 Å, R=20,000, no leakage,
hot-start repeated evaluations, `<10 s` first target and `~1 s` stretch target on a large CUDA GPU.
