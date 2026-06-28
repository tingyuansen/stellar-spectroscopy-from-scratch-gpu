# GPU textbook passdown

Read this before touching the GPU textbook. The goal of this repository is not merely to
"look like" the NumPy book: it is the self-contained, GPU-native decomposition of `kgpu`
(`~/pykurucz_gpu`). A reader should be able to reconstruct the production `kgpu` engine from
these lectures: each lecture is a clean pedagogical torch/MPS block, validates against the
NumPy textbook reference, and imports neither `kgpu` nor `pykurucz`.

For the current ship-gate ledger, read `SHIP_GATES.md` after this file. It records the practical
acceptance criteria: self-contained taught path, NumPy parity, kgpu coverage, and the named
remaining boundaries.

The final quality bar is the same as the NumPy textbook: reference files may provide physical
data tables and comparison answers, but a lecture should not teach by loading a completed
computed state that it claims to build. If a temporary bundle is used to keep a capstone honest,
the passdown must say so and name the integration work needed to remove it.

## Standing relationship

- `~/pykurucz` is the read-only production/reference implementation.
- `~/Stellar_Spectroscopy_From_Scratch` is the read-only NumPy textbook reference. It is the
  parity oracle for this GPU book.
- `~/pykurucz_gpu` is the production torch/MPS product. Its implementation is the source of
  useful algorithmic clues, but the textbook must not import it.
- This repo is the GPU textbook: self-contained torch/MPS lectures, built from the same physics
  pieces as `kgpu`, with comparison cells against the NumPy references.

## Current state

- Lectures 1-16 exist and build.
- L14 has been refreshed away from the stale continuum-only Sun bundle. Its solar capstone
  reference is now the Part-VI line-blanketed solar atmosphere:
  `base RHOX = 12.1439331`, `base T = 11425 K`.
- Be precise about the RHOX numbers. `12.1439331` is the pyk exact-LINOP solar atmosphere
  target currently loaded as the solar atmosphere input for this capstone. The production
  `kgpu` / prior oracle fixed point is
  12.3-class because its coarse opacity-sampled line deposit has a bounded deep-base residual
  against pyk exact LINOP. That is a useful future squeeze target, not a reason to regress L14
  to the old `RHOX = 10.5357` bundle.
- L14 still computes the full opacity in-notebook from populations and line data, then carries
  it through the inlined JOSH transfer. It does not import `_pipeline/verify_leankurucz`,
  `kgpu`, or `pykurucz`.
- But L14 is not yet the final "from stellar parameters to spectrum" GPU-native capstone. Today
  it is self-contained for the synthesis half given an atmosphere/EOS state, and it loads the
  solar line-blanketed atmosphere plus verified computed intermediate state (`population_per_ion`,
  `doppler_per_ion`, continuum absorber populations, and related depth tables) as reference
  inputs. The stricter end state is to wire or inline the L15/L16 atmosphere + state machinery so
  the Sun atmosphere and EOS state are regenerated inside the capstone from stellar parameters,
  then passed directly into the L14 synthesis path.
- Reference-bundle bookkeeping is explicit in `reference/MANIFEST_L14_L16.md`. It classifies
  L14-L16 arrays as physical inputs/static data, computed state currently loaded by a lecture,
  comparison-only targets, or provenance. Use it before changing L14-L16 so target arrays do not
  leak into the computed path.
- GPU-native status is mixed, and should stay honest: lectures 1-7, 9-13, 15-16 have
  `build_lecture<N>_gpu.py` torch/MPS builders; L8 and L14 still use NumPy-style builders.
  L14 is numerically honest now, but not yet the final GPU-native capstone.
- L15/L16 are not final GPU-native closures yet. L15 contains torch/vectorized line-record
  physics, but its accepted fidelity gate still uses the clean-room scalar `verify_lineblanket`
  recurrence for the LINOP1 window/full-step audit. L16 packs and audits the per-iteration state
  in torch, but the load-bearing PFSAHA/NELECT/continuum/molecular pieces still come through the
  clean-room NumPy helper modules (`eos_fromscratch.py`, `continuum_fromscratch.py`,
  `molecular_fromscratch.py`). These helpers are self-contained and do not import `kgpu`/`pykurucz`,
  but they are integration debts for the final GPU-native book.

## L14 checks just run

`python _pipeline/verify_leankurucz.py`:

- Sun spectrum max relative error: `1.02e-08`
- Hot dwarf: `1.28e-07` at the documented H-beta/deep-hot-continuum floor
- Giant: `2.15e-08`
- M dwarf: `1.06e-08`
- Tamper check: Fe I population x1.01 moves the spectrum by `4.867e-03`

`python _pipeline/build.py 14` executed and rendered `content/Lecture14.ipynb/html`.

Notebook audit:

- no `verify_leankurucz` import
- no `import kgpu`
- no unguarded stale `10.535` use; mentions are guardrails only
- prints the solar base `RHOX` guardrail

## Broader checks just run

- Full notebook execution/render:
  `python _pipeline/build.py 1 2 3 4 5 6 7 8` and
  `python _pipeline/build.py 9 10 11 12 13 14 15 16` both completed. All 16 notebooks
  executed and rendered.
- L11/L16 support checks:
  `python _pipeline/verify_convec_gaps.py` PASS, with GAP A/B bit-exact and sabotage checks OK.
  `python _pipeline/verify_converged.py` PASS: convection flux `2.376e-10`, one-step T
  `9.160e-09`, one-step RHOX `4.714e-08`.
- L15 line-blanketing check:
  `python _pipeline/verify_lineblanket.py` PASS. Teaching-window deposit max rel
  `2.877e-06`; one-step line-blanketed engine fidelity T `6.606e-09`, RHOX `5.850e-09`;
  corrected Sun base RHOX `12.152` vs sun `12.144`.
- Targeted API critic review:
  `_pipeline/critic_reports/GPU_textbook_selfcontained_gpu_native_gpt55.md` and
  `_pipeline/critic_reports/GPU_textbook_selfcontained_gpu_native_gemini.md` both say the repo is
  honest/numerically useful but not final-passdown-ready. They flag the same open gaps recorded
  above: L14 loads atmosphere/EOS/population/Doppler state, L8/L14 are NumPy-style builders, L15
  accepted LINOP1 fidelity still uses a clean-room scalar verifier path, and L16 still relies on
  clean-room NumPy helper modules for load-bearing EOS/continuum/molecular computation.

## Delegation policy

Use the API workers for bounded generation/review:

- `_pipeline/port_worker.py` for code-generation/squeeze jobs where the harness can parity-gate
  the result automatically.
- `_pipeline/critic.py` and `_pipeline/critic_visual.py` for prose/visual review.

The orchestrator must keep ownership of the hard gates:

- whether a from-scratch claim is honest;
- whether a reference file is an input or a computed answer;
- whether a parity number is meaningful;
- whether generated prose contradicts `kgpu` passdown;
- final diffs, test runs, and commit decisions.

## Next useful work

- Promote L8 and L14 to true torch/MPS GPU builders. L14 should keep the same honesty gates:
  no hidden computed spectra, no `kgpu`/`pykurucz` import, opacity computed from inputs, and
  parity to the same floors.
- Make the final L14 capstone stricter: call/inline the L15/L16 line-blanketed atmosphere path
  for the Sun instead of loading the finished solar atmosphere bundle as an input.
- Keep using targeted API critics to audit this standard: flag any lecture that loads a completed
  computed state instead of rebuilding it from physical inputs; separately flag any non-GPU-native
  implementation and any parity gap. The first targeted pass has already been run and is recorded
  above.
- Use `reference/MANIFEST_L14_L16.md` as the bookkeeping gate when promoting loaded L14/L15 state
  into computed-in-notebook code.
- Maintain the full lecture build ledger. The 2026-06-28 full build passed; rerun after any
  builder/reference changes.
- Use the L15/L16 exact-line-blanketing material to design a future kgpu squeeze that moves the
  atmosphere base from the coarse-deposit 12.3-class fixed point toward pyk's exact-LINOP
  `RHOX = 12.1439331`, while preserving the solar spectrum gate.
