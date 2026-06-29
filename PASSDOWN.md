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

Public site numbering is now logical: **L14 EOS state -> L15 line blanketing -> L16 spectrum
capstone**. Some source filenames remain legacy-stable for git/history:
`content/Lecture16.*` is public L14, `content/Lecture15.*` is public L15, and
`content/Lecture14.*` is public L16. Keep this mapping in mind when running builders/verifiers.

| Lecture | NumPy/material coverage | Self-contained / honest inputs | Logical flow | Readable names | Dense-code pedagogy | GPU/vectorized taught path | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | [x] | [x] | [x] | [x] | [x] | [x] | [x] build |
| L2 | [x] | [x] | [x] | [x] | [x] | [x] | [x] build |
| L3 | [x] | [x] | [x] | [x] | [x] split fp64 twin from validation/reporting block | [x] | [x] `verify_kapp.py` |
| L4 | [x] | [x] | [x] | [x] | [x] | [x] | [x] build plus downstream line gates |
| L5 | [x] | [x] | [x] | [x] | [x] split Harris precision helper and helium merge/stop helper | [x] | [x] `verify_full_lines.py`, `verify_linetypes.py` |
| L6 | [x] | [x] | [x] | [x] | [x] split fp64 Stark-profile island from Balmer opacity driver | [x] | [x] `verify_josh.py` |
| L7 | [x] | [x] | [x] | [x] | [x] | [x] | [x] `verify_josh.py` |
| L8 | [x] | [x] opacity from `L6.npz`; fixed JOSH operator tables documented as static method constants | [x] | [x] | [x] split PARCOE/INTEG and JOSH/spectrum wrappers | [x] | [x] `verify_josh.py` |
| L9 | [x] | [x] | [x] | [x] | [x] | [x] | [x] build/downstream atmosphere gates |
| L10 | [x] | [x] | [x] | [x] | [x] split physical-grid thin-layer recurrence from JOSH profile driver | [x] | [x] `verify_converged.py` |
| L11 | [x] | [x] comparison oracles are local verifier modules, not taught inputs | [x] | [x] | [x] | [x] | [x] `verify_convec_gaps.py`, `verify_converged.py` |
| L12 | [x] | [x] | [x] | [x] | [x] | [x] | [x] `verify_molecules.py` |
| L13 | [x] | [x] | [x] | [x] | [x] split one-depth Newton solve from pressure-continuation driver | [x] | [x] `verify_nmolec.py`, `verify_mol_continuum.py` |
| L14 | [x] | [ ] loaded solar atmosphere fixture; helper-backed state cells remain | [x] public first finale lecture | [x] | [x] | [ ] helper-backed NumPy islands remain | [x] `build.py 14` via public/site mapping |
| L15 | [x] | [ ] still consumes production-derived atmosphere/EOS/window/continuum/full-grid blanket fixtures | [x] | [x] | [x] | [ ] exact accepted teaching-window recurrence is scalar for parity; torch approximation is not the accepted gate | [x] `verify_lineblanket.py` |
| L16 | [x] | [ ] loads atmosphere/EOS/population/Doppler state; not yet stellar-parameters-to-spectrum | [x] public capstone | [x] partial | [x] split continuum, line, molecular, JOSH, and TCORR helpers below 100-line threshold | [ ] capstone is not yet torch/MPS throughout | [x] `verify_leankurucz.py` |

### Honest remaining blockers

- Parked on `kgpu`: do not try to fully close public L14/L15 inside the textbook before the production
  line-blanketed atmosphere convergence path is corrected/squeezed. L15's exact scalar LINOP1
  teaching-window recurrence and public L14's `max_iter=1` live-smoke boundary are the honest
  current representation. The final all-local, GPU-native finale closure should follow the kgpu
  solver fix, then be ported back here with parity gates.
- Dense pedagogical code cells are closed under the current `>=100` line audit: no lecture has a
  code cell at or above this threshold. A stricter readability pass can still target 80-99 line
  cells, led by L14 and followed by L15/L13/L11/L9/L6/L5; this is not a current parity blocker.
- Public L16 is numerically honest but not the final capstone. It computes opacity + JOSH spectra from a
  supplied atmosphere/EOS/population/Doppler state. It does not yet regenerate the stellar
  atmosphere and EOS state from stellar parameters inside the capstone, and it is not yet a
  torch/MPS builder throughout. The stellar-parameters-to-spectrum version depends on the parked
  public L14/L15/`kgpu` convergence closure.
- L15 still consumes production-derived fixtures for the converged atmosphere, per-iteration
  EOS/window state, continuum opacity/scattering/source arrays, and full-grid line blanket.
- Public L14 is scientifically scoped but not final atmosphere closure. Keep the live atmosphere code, when
  added to the notebook, as a **runnable one-iteration smoke gate** (`max_iter=1`) so readers can
  execute it. The converged solar gate remains future work: target a 12.3-class / pyk exact-LINOP
  `RHOX=12.1439331` solar base after the kgpu squeeze makes the loop fast enough for pedagogy. When
  resumed, use `~/pykurucz_gpu/bench/converge_kgpu_port.py` and `kgpu.atlas_loop.converge_atmosphere`
  only as the reference design to rewrite locally; the textbook must not import `kgpu`.
- Naming is now an explicit quality track. `BIBLE.md` contains the shared glossary; keep raw
  fixture/table keys stable at boundaries and translate to readable names in taught code. Continue
  aggressive local-name cleanup in small gated slices before calling readability closed.

## Public L16 Capstone Status

Public L16 has been repaired away from the stale continuum-only Sun bundle. Its current solar capstone
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

The final public L16 target is stricter: inline or call the public L14/L15 line-blanketed atmosphere
and EOS-state path so the Sun state is regenerated from stellar parameters, then feed that state into the
synthesis path.

Public L16's title has been narrowed to **"A Spectrum from an Atmosphere, End to End"**. Do not
restore "from stellar parameters" until the atmosphere and EOS state are actually regenerated in
the capstone path. The public/site reading order is now normal `N-1/N+1`: L14 EOS state, L15 line
blanketing and ATLAS12 correction, L16 atmosphere-to-spectrum SYNTHE capstone. If the capstone
remains much longer than the other lectures, split it naturally into synthesis ingredients/opacity
and transfer plus HR comparison.

## Checks just run

- Full notebook execution/render:
  `python _pipeline/build.py 1 2 3 4 5 6 7 8`
  and
  `python _pipeline/build.py 9 10 11 12 13 14 15 16`.
- Code-cell hygiene audit: `total bad=0` for leading/trailing blank code cells.
- Function-docstring audit: `lowdoc>=20 = 0` under the current rule (functions/classes at least
  20 lines must have useful docstrings).
- Long-cell audit using the current blocking `>=100` line threshold: zero code cells at or above
  this threshold across L1-L16. A non-blocking stricter pass now tracks 80-99 line cells.
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

1. Textbook-only closeout before the parked kgpu round: run the final-mile closure audit for every
   lecture (L1-L16) against the checklist above; fix
   misses in gated slices and record the result lecture by lecture.
2. Break up or add reader-walkthrough comments to the remaining 80-99 line cells, starting with the
   capstone and L15, and rerun parity after each chunk.
3. Continue the shared naming pass: use the `BIBLE.md` glossary, avoid destructive fixture/schema
   renames, and verify each pure readability slice with the touched lecture's builder/verifier.
4. After kgpu convergence is fixed/squeezed, return to the parked finale closure:
   - promote public L16 to a torch/MPS builder and true stellar-parameters-to-spectrum capstone;
   - replace L15's production-derived atmosphere/EOS/window/continuum/full-grid blanket fixtures
     with computed lecture cells;
   - move public L14 helper-backed clean-room NumPy pieces into pedagogical torch cells where practical;
   - extend the local public L14 loop beyond the `max_iter=1` live-smoke gate, with no runtime
     `kgpu` import, targeting the 12.3-class / `RHOX=12.1439331` solar base before wiring public
     L14/L15 into public L16.
5. Use the exact-LINOP public L14/L15 material to design the future `kgpu` squeeze from the 12.3-class
   coarse-deposit fixed point toward the pyk exact-LINOP `RHOX=12.1439331` target.
