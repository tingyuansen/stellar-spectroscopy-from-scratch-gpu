# GPU textbook ship gates

This file tracks the concrete acceptance gates for the GPU textbook. The priority order is:

1. self-contained lecture computations, with no `kgpu` or `pykurucz` imports in the taught path;
2. parity with shipped reference outputs to the documented CPU/GPU float floor;
3. faithful pedagogical coverage of the production `kgpu` implementation pieces;
4. optimization only after the first three are true.

## Next round: optimization role

After the correctness/no-leakage gates are agreed, this textbook becomes the readable kernel source
for optimizing `kgpu`. Each speed change in `kgpu` should be traceable back to a lecture-level
computation and checked against the NumPy parity target before it is adopted in production. The
optimization target belongs to `kgpu`: full 4000-9000 Å, R=20,000, no leakage, hot-start repeated
evaluations, `<10 s` first target and `~1 s` stretch target on a large CUDA GPU.

## Current audit snapshot

Generated from source inspection on 2026-06-29.

### Import boundary

- Executed lecture notebooks/builders do not import `kgpu` or `pykurucz`.
- Reference-generation and verification utilities under `_pipeline/make_*`, `_pipeline/verify_*`,
  and `_pipeline/converge_fromscratch.py` may read `pykurucz` or `kgpu` as offline provenance tools.
  They are not the taught notebook path.
- L8 is now torch-native but still uses the legacy `_pipeline/build_lecture8.py` filename and still
  consumes scoped computed opacity slabs plus fixed JOSH operator tables in the taught path. L14
  still uses a non-`_gpu` builder. These are remaining GPU-textbook/data-closure gaps.

### Lecture state

| Lectures | Status | Ship interpretation |
|---|---|---|
| L1-L7 | GPU builders, executed parity cells | Core microphysics/synthesis primitives are torch-based and validated against shipped NumPy references. |
| L8 | torch-native builder; data-boundary debt | JOSH kernels run in torch/MPS and import no production solver, but the taught path still consumes `diag.npz` opacity slabs and `josh_tables.npz` operator tables. Source/flux arrays are comparison-only. Not strict self-contained closure. |
| L9-L13 | GPU builders | Atmosphere primitives, molecular equilibrium, and molecular continuum are GPU-textbook form, with reference-input boundaries named in notebooks. |
| L14 | non-torch builder | Numerically honest synthesis-half capstone. It computes opacity/transfer but loads atmosphere/EOS/population/Doppler state; not final stellar-parameters-to-spectrum closure. |
| L15 | GPU builder with in-notebook scalar LINOP1 gate | Teaching-window deposit is computed in the taught path. Strict boundary remains open: converged atmosphere, EOS/window state, continuum opacity/scattering/source arrays, and full-grid line blanket are production-derived computed fixtures. |
| L16 | GPU builder with clean-room helper modules | EOS-state blocker is reduced: populations/state are recomputed before comparison, but only for a loaded atmosphere fixture. Load-bearing PFSAHA/finite-difference/continuum/molecular helpers are still clean-room NumPy modules, not final all-torch closure. |

## Required gates before calling the GPU textbook passdown-ready

- Run the full notebook ledger after any material lecture edit:
  `python _pipeline/build.py 1 2 3 4 5 6 7 8` and
  `python _pipeline/build.py 9 10 11 12 13 14 15 16`.
- Run the focused physics gates:
  `python _pipeline/verify_leankurucz.py`,
  `python _pipeline/verify_lineblanket.py`,
  `python _pipeline/verify_molecules.py`,
  `python _pipeline/verify_nmolec.py`,
  `python _pipeline/verify_mol_continuum.py`,
  `python _pipeline/verify_convec_gaps.py`,
  `python _pipeline/verify_converged.py`.
- Run the full critic sweep before declaring prose/passdown closure:
  `python _pipeline/critic.py 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16`.
- Keep `reference/MANIFEST_L14_L16.md` current whenever L14-L16 reference arrays are added or
  reclassified.
- Do not upgrade any lecture claim from "comparison target" or "integration input" to
  "computed here" unless the notebook actually computes it in the taught path.
- Stored fixture files checked into this repo are allowed when classified honestly as physical
  input, scoped fixture, or comparison target. The hard ban is runtime dependence on external
  `pykurucz`/`leankurucz` code or peeking into external computed state. Loaded fixtures must not
  masquerade as computed notebook outputs. Physical/static tables and line-list metadata should
  stay distinct from computed answer/state fixtures.

## Final closure work

- Close L8's remaining data boundary: feed opacity slabs from the Lecture 3-6 torch outputs and
  derive/generate or explicitly vendor-provenance the fixed JOSH operator tables, instead of
  treating production-derived computed arrays as closure.
- Promote L14 to a true torch/MPS capstone and wire/inline the L15/L16 atmosphere + EOS-state path
  so the Sun state is regenerated rather than loaded.
- Extend the non-solar capstone from SYNTHE on warm-started structures to full atmosphere+spectrum
  once the corresponding kgpu four-star ATLAS12+SYNTHE gate exists.
- Replace L15's production-derived atmosphere/EOS/window/continuum/full-grid blanket fixtures and
  L16's loaded-atmosphere/helper-backed pieces with final self-contained torch-native lecture cells
  where practical, while preserving the scalar clean-room verifiers as regression oracles.
