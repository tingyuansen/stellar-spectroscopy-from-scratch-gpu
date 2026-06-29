# GPU Textbook Plan

`PASSDOWN.md` is the live state. `BIBLE.md` is the lecture quality standard. This file is the
roadmap.

## Current Scope

The textbook is a standalone torch/MPS/CUDA reconstruction of the kgpu/pykurucz physics. The taught
paths must not import `kgpu`, `pykurucz`, or the NumPy textbook.

Current status:

- L1-L7 and L9-L13 are mostly green GPU-native lecture paths.
- L8 has a torch-native JOSH path but still consumes scoped opacity slabs and fixed operator tables.
- L14 is an honest same-atmosphere synthesis capstone: opacity and transfer are computed in the
  notebook, while atmosphere/EOS/population/Doppler state is loaded.
- L15 computes the exact LINOP1 teaching-window recurrence, but still loads atmosphere/EOS/window/
  continuum/full-grid blanket fixtures.
- L16 recomputes EOS-derived state for a loaded atmosphere fixture, with remaining helper-backed
  NumPy boundaries.

## Closure Roadmap

0. Complete the final-mile lecture closure audit:
   - audit every lecture L1-L16, not only high-value hotspots;
   - each lecture must be self-contained, logically connected to neighbors, honest about fixtures,
     readable in functions and local variables, pedagogically commented inside dense code, and
     GPU-native/vectorized where the algorithm allows;
   - after each fix, rerun the lecture's build/render and narrow parity verifier;
   - do not erase this roadmap until every lecture either passes the closure checklist or has a
     precise, honest boundary that remains as future work.
   - status: `PASSDOWN.md` now carries the per-lecture checkbox matrix. The remaining concrete
     lecture fixes are L8 fixed-operator provenance,
     L14 true capstone/torch promotion, L15 fixture reduction, and L16 helper/fixture reduction.

1. Establish the shared naming/glossary pass:
   - define readable names in `BIBLE.md` and use them consistently across lectures and kgpu;
   - translate legacy bundle/table fields at input boundaries, not throughout taught code;
   - keep canonical Kurucz table names only when changing them would obscure provenance or break
     parity;
   - rename in small builder/verifier-gated slices;
   - status: L1-L13 have now had the broad naming/line-break readability pass; L3/L5/L6/L13 also
     have stricter internal-function rename slices for KAPP, line opacity, HPROF4, and molecular
     equilibrium. Continue any remaining local-name cleanup in small builder/verifier-gated commits
     before claiming global readability closure. Leave remaining legacy names only where they are
     canonical table/routine labels, fixture keys, or scalar-reference transcriptions.

2. Rework the L14-L16 curriculum order before major capstone promotion:
   - collect the end-to-end ATLAS12+SYNTHE topics in `BIBLE.md`/`PLAN.md`;
   - make the sequence match the kgpu build: atmosphere/EOS state, line blanket/deposit,
     continuum+line synthesis, then capstone spectrum;
   - preferred logical order is EOS/state first, line blanketing and ATLAS12 correction second,
     atmosphere-to-spectrum SYNTHE capstone last;
   - if the capstone remains much longer than the other lectures, split it at a natural boundary:
     synthesis ingredients/opacities first, transfer plus HR comparison finale second;
   - preserve every current NumPy/reference parity check while moving topics;
   - keep `reference/MANIFEST_L14_L16.md` synchronized whenever arrays move between loaded,
     computed, and comparison-only roles.
   - status: first low-risk pass complete. Public/site read order is now L16 EOS/state, L15 line
     blanketing and atmosphere correction, then L14 atmosphere-to-spectrum capstone, with stable
     lecture numbers and URLs. L14/L15/L16 rebuilt and verified after the prose/order move.

3. Close L8's data boundary:
   - status: opacity slabs now feed from `reference/L6.npz`, the preceding lecture artifact;
   - remaining: derive, rebuild, or explicitly vendor-provenance fixed JOSH operator tables.

4. Promote L14 from same-atmosphere synthesis to true capstone:
   - wire or inline L15/L16 so the Sun state is regenerated from stellar parameters;
   - keep the four HR-window `verify_leankurucz.py` gate green;
   - do not claim non-solar atmosphere closure until kgpu has a corresponding four-star
     ATLAS12+SYNTHE gate.

5. Reduce L15 fixtures:
   - replace loaded atmosphere/EOS/window/continuum/full-grid blanket fixtures in small
     parity-gated steps;
   - keep the exact LINOP1 recurrence and float32 accumulation-order gate intact.

6. Reduce L16 helper boundaries:
   - port clean-room NumPy PFSAHA/NELECT/continuum/molecular helper paths into pedagogical torch
     cells where practical;
   - retain scalar loops only for true recurrences or fixed table logic.

7. Finish dense-code teaching cleanup after the L14-L16 restructure:
   - status: closed under the current `>=100` line audit; all L1-L16 code cells are below this
     threshold after the L14 helper split;
   - add reader-walkthrough comments/docstrings inside dense code blocks after the sequence is
     stable, so comments describe the final structure rather than a soon-to-move draft;
   - split or comment dense cells only when the corresponding verifier is rerun.

## GPU-Native Audit Rules

- Prefer torch on the selected device for taught compute.
- Vectorize large axes when this preserves the algorithm.
- Keep scalar loops only for exact recurrences, table parsers, boundary stencils, or small
  heterogeneous tables, and state why.
- Avoid host pulls inside compute paths.
- Use NumPy only for comparison, plotting, static table preparation, or explicitly named
  non-taught helper boundaries.

## Naming Audit Rules

- Prefer descriptive names in taught code even if the reference variable is terse.
- Translate legacy fixture keys once near load time, then use the readable local names.
- Keep compatibility names at serialization boundaries and public APIs unless a migration plan and
  tests cover the change.
- Avoid mass renames mixed with physics edits. Commit pure readability slices separately, with the
  same parity gate as the touched lecture/module.

## Verification Roadmap

After meaningful lecture changes, run the narrowest matching gate plus:

```bash
git diff --check
python _pipeline/build.py <N>
```

For capstone or shared-physics changes, also run:

```bash
python _pipeline/verify_josh.py
python _pipeline/verify_leankurucz.py
python _pipeline/verify_lineblanket.py
python _pipeline/verify_molecules.py
python _pipeline/verify_nmolec.py
python _pipeline/verify_mol_continuum.py
python _pipeline/verify_convec_gaps.py
python _pipeline/verify_converged.py
```

Before a handoff claim, rebuild all lectures:

```bash
python _pipeline/build.py 1 2 3 4 5 6 7 8
python _pipeline/build.py 9 10 11 12 13 14 15 16
```

## kgpu Feedback Loop

Only port textbook improvements into kgpu after the textbook verifier passes. A kgpu port then must
pass full pytest, the four-star gate, and the relevant solar/full-spectrum gate. Optimization is a
second-order goal until the correctness scope is explicit.
