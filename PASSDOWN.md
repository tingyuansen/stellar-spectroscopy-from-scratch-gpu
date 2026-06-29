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

Lectures 1-16 build and render. The local site is currently running at
`http://localhost:8081/`; GitHub Pages is pushed. The active GPU builders and generated pages no
longer carry stale collaborator/API/title-page framing.

### Per-lecture closure matrix

Legend: checked means the current lecture has been audited against the criterion and no known
blocking miss remains. Unchecked means it is the next fix target, not a vague concern.

| Lecture | NumPy/material coverage | Self-contained / honest inputs | Logical flow | Readable names | Dense-code pedagogy | GPU/vectorized taught path | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | [x] | [x] | [x] | [x] | [x] | [x] | [x] build |
| L2 | [x] | [x] | [x] | [x] | [x] | [x] | [x] build |
| L3 | [x] | [x] | [x] | [x] | [ ] 111-line NumPy comparison twin needs a reader-walkthrough split/comment pass | [x] | [x] `verify_kapp.py` |
| L4 | [x] | [x] | [x] | [x] | [x] | [x] | [x] build plus downstream line gates |
| L5 | [x] | [x] | [x] | [x] | [ ] two 100+ line oracle/helium cells need splitting or stronger inline comments | [x] | [x] `verify_full_lines.py`, `verify_linetypes.py` |
| L6 | [x] | [x] | [x] | [x] | [ ] 115-line hydrogen-opacity cell needs splitting/comment pass | [x] | [x] `verify_josh.py` |
| L7 | [x] | [x] | [x] | [x] | [x] | [x] | [x] `verify_josh.py` |
| L8 | [x] | [ ] loads scoped opacity slabs and fixed JOSH operator tables | [x] | [x] | [ ] two 100+ line transfer/helper cells need splitting/comment pass | [x] | [x] `verify_josh.py` |
| L9 | [x] | [x] | [x] | [x] | [x] | [x] | [x] build/downstream atmosphere gates |
| L10 | [x] | [x] | [x] | [x] | [ ] one 100-line convergence cell needs splitting/comment pass | [x] | [x] build/downstream atmosphere gates |
| L11 | [x] | [x] comparison oracles are local verifier modules, not taught inputs | [x] | [x] | [x] | [x] | [x] `verify_convec_gaps.py`, `verify_converged.py` |
| L12 | [x] | [x] | [x] | [x] | [x] | [x] | [x] `verify_molecules.py` |
| L13 | [x] | [x] | [x] | [x] | [ ] 100-line molecular-equilibrium scalar driver needs splitting/comment pass | [x] | [x] `verify_nmolec.py`, `verify_mol_continuum.py` |
| L14 | [x] | [ ] loads atmosphere/EOS/population/Doppler state; not yet stellar-parameters-to-spectrum | [x] after L16 -> L15 -> L14 public read-order repair | [x] partial | [ ] multiple 100+ line capstone cells remain | [ ] capstone is not yet torch/MPS throughout | [x] `verify_leankurucz.py` |
| L15 | [x] | [ ] still consumes production-derived atmosphere/EOS/window/continuum/full-grid blanket fixtures | [x] | [x] | [x] | [x] teaching-window deposit is torch; exact scalar recurrence retained for parity | [x] `verify_lineblanket.py` |
| L16 | [x] | [ ] starts from loaded atmosphere/radiation fixture and helper-backed state modules | [x] | [x] | [x] | [ ] several load-bearing helpers are clean-room NumPy, not final torch cells | [x] `build.py 16`, L15/L14 downstream gates |

### Honest remaining blockers

- Dense pedagogical code cells remain in L3, L5, L6, L8, L10, L13, and L14 under the current
  `>=100` line audit. Break or comment them only with the matching lecture build and verifier rerun.
- L8 is not strict closure until its opacity slabs come from the earlier lecture torch outputs and
  the fixed JOSH operator tables are either derived in-book or explicitly vendor-provenanced.
- L14 is numerically honest but not the final capstone. It computes opacity + JOSH spectra from a
  supplied atmosphere/EOS/population/Doppler state. It does not yet regenerate the stellar
  atmosphere and EOS state from stellar parameters inside the capstone, and it is not yet a
  torch/MPS builder throughout.
- L15 still consumes production-derived fixtures for the converged atmosphere, per-iteration
  EOS/window state, continuum opacity/scattering/source arrays, and full-grid line blanket.
- L16 still starts from a loaded atmosphere/radiation fixture and uses clean-room NumPy helper
  modules (`eos_fromscratch.py`, `continuum_fromscratch.py`, `molecular_fromscratch.py`) for some
  load-bearing physics. These helpers are not runtime `kgpu`/`pykurucz` imports, but they are not
  final all-torch lecture closure.
- Naming is now an explicit quality track. `BIBLE.md` contains the shared glossary; keep raw
  fixture/table keys stable at boundaries and translate to readable names in taught code. Continue
  aggressive local-name cleanup in small gated slices before calling readability closed.

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

L14's public title has been narrowed to **"A Spectrum from an Atmosphere, End to End"**. Do not
restore "from stellar parameters" until the atmosphere and EOS state are actually regenerated in
the capstone path. The public/site reading order now presents the finale in conceptual order while
preserving stable lecture numbers and URLs: L16 EOS/state first, L15 line blanketing and ATLAS12
correction second, L14 atmosphere-to-spectrum SYNTHE capstone last. If the capstone remains much
longer than the other lectures, split it naturally into synthesis ingredients/opacity and transfer
plus HR comparison.

## Checks just run

- Full notebook execution/render:
  `python _pipeline/build.py 1 2 3 4 5 6 7 8`
  and
  `python _pipeline/build.py 9 10 11 12 13 14 15 16`.
- Code-cell hygiene audit: `total bad=0` for leading/trailing blank code cells.
- Function-docstring audit: `lowdoc>=20 = 0` under the current rule (functions/classes at least
  20 lines must have useful docstrings).
- Remaining long-cell audit using the stricter `>=100` line threshold:
  L3 has 1, L5 has 2, L6 has 1, L8 has 2, L10 has 1, L13 has 1, and L14 has 9. L15 and L16 have
  no code cells at or above this threshold.
- `git diff --check`: clean.
- `python _pipeline/build_lecture3_gpu.py && python _pipeline/build.py 3 &&
  python _pipeline/verify_kapp.py`: L3 stricter KAPP naming pass is green; KAPP verifier reports
  continuum absorption max rel `9.015e-05`, scattering max rel `2.061e-07`, and photosphere max rel
  `0.000e+00`.
- `python _pipeline/build_lecture5_gpu.py && python _pipeline/build.py 5 &&
  python _pipeline/verify_full_lines.py && python _pipeline/verify_linetypes.py`: L5 stricter
  line-opacity naming pass is green; full-line max rel `2.254e-15`, special TYPE=1/81 records
  bit-exact.
- `python _pipeline/build_lecture6_gpu.py && python _pipeline/build.py 6 &&
  python _pipeline/verify_josh.py`: L6 stricter HPROF4 naming pass is green; inline Planck source
  max `2.62e-15`, normalized spectrum max `8.94e-09`, median `1.13e-11`.
- `python _pipeline/build_lecture13_gpu.py && python _pipeline/build.py 13 &&
  python _pipeline/verify_nmolec.py && python _pipeline/verify_mol_continuum.py`: L13 stricter
  molecular-equilibrium solver naming pass is green; NMOLEC max rel `9.920e-14`, molecular
  continuum max rel `0.000e+00`.
- `python _pipeline/build_lecture16.py && python _pipeline/build_lecture15_gpu.py &&
  python _pipeline/build_lecture14.py && python _pipeline/build.py 16 15 14 &&
  python _pipeline/verify_lineblanket.py && python _pipeline/verify_leankurucz.py &&
  python _pipeline/verify_converged.py`: finale read-order/prose restructure is green. L15 PASS;
  L14 four-star capstone PASS with Sun base `RHOX=12.1439`, base `T=11425.0 K`; L11 converged
  verifier PASS.
- `python _pipeline/verify_josh.py`: normalised spectrum max `8.94e-09`, median `1.13e-11`.
- `python _pipeline/verify_lineblanket.py`: PASS, values listed above.
- `python _pipeline/verify_leankurucz.py`: PASS, values listed above.
- `python _pipeline/verify_molecules.py`: TiO band spectrum max rel `1.063e-08`, median
  `1.679e-13`; after the L12 readability rename, molecular opacity max rel `4.069e-11`, spectrum
  max rel `1.063e-08`.
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

## Final-mile closure criteria

Do not delete the plan/passdown until each lecture has been audited against this checklist and any
misses have either been fixed with a passing gate or explicitly marked as a real boundary:

1. **Self-contained taught path:** the lecture imports no `kgpu`, `pykurucz`, or NumPy-textbook code;
   static inputs, fixtures, and comparison targets are named honestly.
2. **Logical flow:** the lecture connects from the previous lecture and toward the next one without
   stale numbering, outside-code framing, or unsupported closure claims.
3. **Readable code:** functions and local variables use physical names where possible; canonical
   Kurucz names remain only at fixture/table/reference boundaries.
4. **Pedagogical code cells:** dense code cells are split or interleaved with comments/docstrings
   enough for a reader to follow the physics, tensor shape, precision boundary, and loop reason.
5. **GPU-native/vectorized where feasible:** taught compute uses torch/device tensors and batched
   operations where the algorithm allows; scalar loops are justified as recurrences, exact-order
   parity paths, small heterogeneous tables, or host setup.
6. **Parity gate:** the lecture's build/render plus its narrow verifier still pass after edits; the
   comparison is self-contained inside this repo.

## Next useful work

1. Run the final-mile closure audit for every lecture (L1-L16) against the checklist above; fix
   misses in gated slices and record the result lecture by lecture.
2. Close L8's data boundary by feeding opacity from earlier torch lecture outputs and documenting or
   deriving the JOSH operator tables.
3. Promote L14 to a torch/MPS builder while keeping the current no-leakage/parity guardrails.
4. Wire L15/L16 into L14 so the solar atmosphere/EOS state is generated rather than loaded.
5. Replace L15's production-derived atmosphere/EOS/window/continuum/full-grid blanket fixtures with
   computed lecture cells in small parity-gated steps.
6. Move L16 helper-backed clean-room NumPy pieces into pedagogical torch cells where practical.
7. Break up the densest L14/L15 code cells with interleaved markdown and docstrings, rerunning
   parity after each chunk.
8. Continue the shared naming pass: use the `BIBLE.md` glossary, avoid destructive fixture/schema
   renames, and verify each pure readability slice with the touched lecture's builder/verifier.
9. Use the exact-LINOP L15/L16 material to design the future `kgpu` squeeze from the 12.3-class
   coarse-deposit fixed point toward the pyk exact-LINOP `RHOX=12.1439331` target.
