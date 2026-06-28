# GPU textbook ship gates

This file tracks the concrete acceptance gates for the GPU textbook. The priority order is:

1. self-contained lecture computations, with no `kgpu` or `pykurucz` imports in the taught path;
2. parity with the NumPy textbook reference outputs to the documented CPU/GPU float floor;
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
- L8 and L14 still use non-`_gpu` builders. This is the clearest remaining GPU-textbook closure gap.

### Lecture state

| Lectures | Status | Ship interpretation |
|---|---|---|
| L1-L7 | GPU builders, executed parity cells | Core microphysics/synthesis primitives are torch-based and validated against shipped NumPy references. |
| L8 | NumPy-style builder | Self-contained JOSH lesson exists, but not final GPU-native textbook form. |
| L9-L13 | GPU builders | Atmosphere primitives, molecular equilibrium, and molecular continuum are GPU-textbook form, with reference-input boundaries named in notebooks. |
| L14 | NumPy-style builder | Numerically honest synthesis-half capstone. It computes opacity/transfer but loads atmosphere/EOS/population/Doppler state; not final stellar-parameters-to-spectrum closure. |
| L15 | GPU builder with helper verifier | Line-blanketing lesson is honest and useful. Teaching-window deposit is computed; accepted heavy fidelity still uses clean-room verifier/full-grid precomputed inputs. |
| L16 | GPU builder with clean-room helper modules | EOS-state blocker is repaired semantically: populations/state are recomputed before comparison. Load-bearing PFSAHA/continuum/molecular helpers are still clean-room NumPy modules, not final all-torch closure. |

## Required gates before calling the GPU textbook passdown-ready

- Run the full notebook ledger after any material lecture edit:
  `python _pipeline/build.py 1 2 3 4 5 6 7 8` and
  `python _pipeline/build.py 9 10 11 12 13 14 15 16`.
- Run the focused physics gates:
  `python _pipeline/verify_leankurucz.py`,
  `python _pipeline/verify_lineblanket.py`,
  `python _pipeline/verify_convec_gaps.py`,
  `python _pipeline/verify_converged.py`.
- Keep `reference/MANIFEST_L14_L16.md` current whenever L14-L16 reference arrays are added or
  reclassified.
- Do not upgrade any lecture claim from "comparison target" or "integration input" to
  "computed here" unless the notebook actually computes it in the taught path.

## Final closure work

- Promote L8 to a true torch/MPS builder.
- Promote L14 to a true torch/MPS capstone and wire/inline the L15/L16 atmosphere + EOS-state path
  so the Sun state is regenerated rather than loaded.
- Extend the non-solar capstone from SYNTHE on warm-started structures to full atmosphere+spectrum
  once the corresponding kgpu four-star ATLAS12+SYNTHE gate exists.
- Replace L15/L16 helper-backed accepted paths with final torch-native lecture cells where practical,
  while preserving the scalar clean-room verifier as a regression oracle.
