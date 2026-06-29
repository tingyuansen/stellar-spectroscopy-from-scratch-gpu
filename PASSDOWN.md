# GPU textbook passdown

Read this before touching the GPU textbook. This repo is the standalone torch/MPS textbook for
the `kgpu` implementation in `~/pykurucz_gpu`: a reader should be able to reconstruct the
production kernels from the lectures without importing `kgpu`, `pykurucz`, or the original NumPy
book at runtime.

The hard quality order is:

1. honest self-contained taught paths;
2. parity with shipped reference targets to the documented float floor;
3. faithful coverage of the production `kgpu` pieces;
4. optimization only after the first three stay green.

Reference files are allowed, but they must be classified honestly. A checked-in file can be a
physical/static input, a scoped fixture used to keep an integration lesson honest, or a comparison
target. It must not masquerade as a value computed by the notebook.

## Repository relationship

- `~/pykurucz` is read-only gold/reference code. Do not import it in taught lecture paths.
- `~/Stellar_Spectroscopy_From_Scratch` is the read-only NumPy reference textbook and parity oracle,
  not a dependency or framing device for this book.
- `~/pykurucz_gpu` is the production torch/MPS product. It is useful for implementation clues, but
  the textbook must remain independently readable and debuggable.
- This repo is the GPU textbook. Lecture code should be torch-native where practical, pedagogical,
  and explicit about precision floors, fixture boundaries, and parity checks.

## Current state

Lectures 1-16 build and render. The local and GitHub Pages sites are live.

### Mostly green

- L1-L7: torch/MPS builders and parity cells for core microphysics and opacity primitives.
- L8: torch-native JOSH transfer path, no production solver import, strong algorithmic parity.
  Remaining boundary: it still loads scoped opacity slabs (`diag.npz`) and fixed JOSH operator
  tables (`josh_tables.npz`).
- L9-L13: GPU builders for atmosphere primitives, molecular equilibrium, molecular bands, and
  molecular continuum. Parity checks are green.
- L15: GPU builder with in-notebook scalar LINOP1 teaching-window recurrence. The verifier reports
  teaching-window deposit max rel `2.877e-06`, one-step T `6.606e-09`, RHOX `5.850e-09`, and
  corrected Sun base `RHOX=12.152` versus `12.144`.
- L16: GPU builder that recomputes EOS-derived state for a loaded atmosphere fixture and documents
  the helper boundaries.

### Honest remaining blockers

- L8 is not strict closure until its opacity slabs come from the earlier lecture torch outputs and
  the fixed JOSH operator tables are either derived in-book or explicitly vendor-provenanced.
- L14 is numerically honest but not the final capstone. It computes opacity + JOSH spectra from a
  supplied atmosphere/EOS/population/Doppler state. It does not yet regenerate the stellar
  atmosphere and EOS state from stellar parameters inside the capstone.
- L15 still consumes production-derived fixtures for the converged atmosphere, per-iteration
  EOS/window state, continuum opacity/scattering/source arrays, and full-grid line blanket.
- L16 still starts from a loaded atmosphere/radiation fixture and uses clean-room NumPy helper
  modules (`eos_fromscratch.py`, `continuum_fromscratch.py`, `molecular_fromscratch.py`) for some
  load-bearing physics. These helpers are not runtime `kgpu`/`pykurucz` imports, but they are not
  final all-torch lecture closure.
- Dense pedagogical cells remain in L14. Break them only when the corresponding parity gate is
  rerun.

## L14 status

L14 has been repaired away from the stale continuum-only Sun bundle. Its current solar capstone
uses the Part-VI line-blanketed solar atmosphere as an input:

- base `RHOX = 12.1439331`
- base `T = 11425 K`

Be precise about this number. `12.1439331` is the pyk exact-LINOP solar atmosphere target currently
loaded into L14 as a solar atmosphere input. It does not prove that `kgpu` has already closed the
exact-LINOP atmosphere build. The old `RHOX=10.5357` bundle is stale and should remain only as a
guardrail against regression.

`python _pipeline/verify_leankurucz.py` passes:

- hot dwarf max rel `1.28e-07`
- Sun max rel `1.02e-08`
- giant max rel `2.15e-08`
- M dwarf max rel `1.06e-08`
- tamper check: Fe I population x1.01 moves the spectrum by `4.867e-03`

The final L14 target is stricter: inline or call the L15/L16 line-blanketed atmosphere + EOS-state
path so the Sun state is regenerated from stellar parameters, then feed that state into the L14
synthesis path.

## Checks just run

- Full notebook execution/render:
  `python _pipeline/build.py 1 2 3 4 5 6 7 8`
  and
  `python _pipeline/build.py 9 10 11 12 13 14 15 16`.
- Code-cell hygiene audit: `total bad=0` for leading/trailing blank code cells.
- Function-docstring audit: `lowdoc>=20 = 0` under the current rule (functions/classes at least
  20 lines must have useful docstrings).
- Remaining long-cell audit: four cells at `>=120` lines remain, all in L14. L15's exact LINOP1
  deposit has been split into scalar Voigt, FASTEX/wing, driver, and execution cells and reverified.
  The remaining L14 cells are structural capstone cells and should be split only with matching
  parity reruns.
- `git diff --check`: clean.
- `python _pipeline/verify_josh.py`: normalised spectrum max `8.94e-09`, median `1.13e-11`.
- `python _pipeline/verify_lineblanket.py`: PASS, values listed above.
- `python _pipeline/verify_leankurucz.py`: PASS, values listed above.
- `python _pipeline/verify_molecules.py`: TiO band spectrum max rel `1.063e-08`, median
  `1.679e-13`.
- `python _pipeline/verify_nmolec.py`: all molecular densities max rel `9.920e-14`.
- `python _pipeline/verify_mol_continuum.py`: molecular continuum max rel `0.000e+00`.
- `python _pipeline/verify_convec_gaps.py`: PASS with sabotage checks.
- `python _pipeline/verify_converged.py`: PASS; convection flux `2.376e-10`, one-step T
  `9.160e-09`, one-step RHOX `4.714e-08`.

## Editing standard

- Do not upgrade any claim from "loaded fixture" or "comparison target" to "computed here" unless
  the notebook actually computes it in the taught path.
- Every nontrivial function should have a useful docstring covering inputs, outputs, precision
  caveats, and fixture boundaries.
- Code cells should have enough comments for a reader to follow the physics and tensor shape
  changes without reading production code.
- Audit `for`/`while`/`if` blocks and `[i]`/`[i+1]` slicing. Prefer torch vectorization,
  `gather`, `searchsorted`, masks, and batched reductions where parity allows. Keep scalar loops
  only for structural recurrences, table parsers, boundary stencils, or exact-order parity paths,
  and say why.
- Avoid leading/trailing blank lines in generated code cells.
- After any readability/vectorization change, rerun the smallest matching parity gate; after
  builder/reference changes, rerun the full notebook ledger.

## Next useful work

1. Close L8's data boundary by feeding opacity from earlier torch lecture outputs and documenting or
   deriving the JOSH operator tables.
2. Promote L14 to a torch/MPS builder while keeping the current no-leakage/parity guardrails.
3. Wire L15/L16 into L14 so the solar atmosphere/EOS state is generated rather than loaded.
4. Replace L15's production-derived atmosphere/EOS/window/continuum/full-grid blanket fixtures with
   computed lecture cells in small parity-gated steps.
5. Move L16 helper-backed clean-room NumPy pieces into pedagogical torch cells where practical.
6. Break up the densest L14/L15 code cells with interleaved markdown and docstrings, rerunning
   parity after each chunk.
7. Use the exact-LINOP L15/L16 material to design the future `kgpu` squeeze from the 12.3-class
   coarse-deposit fixed point toward the pyk exact-LINOP `RHOX=12.1439331` target.
