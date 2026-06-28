# PLAN — the GPU-substitution roadmap

*Stellar Spectroscopy from Scratch — GPU Edition.* This file is the working map for turning the
NumPy edition, lecture by lecture, into a clean **torch/MPS** companion. It states the per-lecture
pattern, classifies which lectures can be ported **now** vs. which **wait**, and records the open
TODOs. Read it alongside `STRUCTURE.md` (the four-part arc) and `README.md` (the prose).

---

## The two design philosophies, stacked

The NumPy edition is a **clean pedagogical reduction of pykurucz** (ATLAS12 + SYNTHE): the same
formulas and constants, the production hardening stripped away. This edition adds a second reduction
on the same spine:

1. **GPU port = a clean pedagogical reduction of `kgpu`.** `kgpu` (`~/pykurucz_gpu`) is the
   production torch/MPS reimplementation of ATLAS12 + SYNTHE — depth-batched, GPU-resident, with a
   22-file parity test suite. We **read** it as the reference for how each computation vectorizes on
   the GPU, then write a *clean* version for the lecture: plain readable `torch`, bite-size cells,
   no production residency bookkeeping / custom kernels / caching / CLI. `kgpu` is **read-only**; we
   never import it into a notebook and never edit it. (The notebooks also never import pykurucz.)

2. **Each part is validated against its NumPy twin.** Every lecture ends with a **comparison cell**
   that runs the GPU computation and the NumPy edition's shipped `reference/*.npz` side by side and
   reports the **maximum relative deviation**, asserting parity to the documented float floor. This
   is the independent per-part check: the GPU code is correct iff its number matches the NumPy result
   the NumPy edition already proved bit-for-bit against pykurucz.

So the validation chain is: **pykurucz ⇄ NumPy edition** (machine precision, already done) and
**NumPy edition ⇄ GPU edition** (this book, the float floor). The NumPy reference is the gold
standard here and is never modified.

---

## The per-lecture pattern (the template)

Every ported lecture follows the same shape. The PoC (Lecture 2) is the worked example.

1. **Same prose, same physics.** Keep the NumPy edition's markdown — the derivations, the
   schematics, the learning objectives. The physics does not change; only the implementation does.
2. **Device + dtype preamble (one cell).** Pick the device once: MPS if available, else CUDA, else
   CPU. On CPU the working dtype is fp64 (machine-precision reference); on MPS/CUDA it is fp32 (MPS
   has no fp64). State the precision budget plainly.
3. **The clean torch computation, in bite-size cells.** Port the lecture's core routine to `torch`,
   **vectorized over the depth axis** (depth is the batch axis — every tensor op processes all
   atmospheric layers at once; no per-depth Python loop). Branchy per-element / per-regime logic is
   folded into `torch.where` masks so every depth lane does the same work. This mirrors `kgpu`'s
   structure but in readable form. A short note flags where fp32 needs care (e.g. a Saha ladder run
   in log-space so the running product never overflows fp32's exponent ceiling).
4. **The comparison cell (the per-part check).** Load the NumPy edition's `reference/*.npz`, move the
   GPU result to CPU/NumPy, and report `max |gpu − ref| / |ref|`. Assert it is below the lecture's
   documented float floor. Print the device used and the floor met.
5. **Build + render.** `_pipeline/build_lecture<N>_gpu.py` assembles the notebook;
   `_pipeline/build.py <N>` executes it (MPS if present, else CPU) and renders the `content/*.html`
   fragment.

**The float floors** (fp32 GPU vs. fp64 NumPy reference; tighter on a CPU fp64 run):

| computation kind | typical fp32 floor |
|---|---|
| single reduction (e.g. a charge-balance sum, a moment integral) | ~1e-7 |
| chained log/exp ladders, opacities (EOS per-ion, continuum, lines) | a few ×1e-6 on the opacity-bearing values |
| trace quantities tens of dex below the dominant value | larger (fp32 exponent floor; physically irrelevant) |

These are the same floors `kgpu`'s test suite asserts. The comparison cell reports the number; it is
quantified, not hidden.

---

## Classification — port NOW vs. WAIT

The dividing line is exactly `kgpu`'s own: the **SYNTHE-side microphysics** (atmosphere given →
opacity + transfer) is GPU-validated by `kgpu`'s component tests and ports now; the **ATLAS12-side
atmosphere convergence** depends on the in-flight **fp32-convergence-core fix** in `kgpu` (the
fully-MPS path currently diverges in fp32 until the precision-critical reductions are fp64-promoted),
so those lectures' GPU numbers are not final and they **wait**.

### Port NOW — microphysics validated by kgpu's test suite

| L | Lecture | kgpu module (read-only ref) | kgpu test | Validate vs (NumPy ref) |
|---|---------|------------------------------|-----------|-------------------------|
| **2** | The Equation of State | `kgpu/eos.py` | `test_eos`, `test_eos_nelect` | `reference/L2.npz` (n_e, H ionization) → **the PoC** |
| 3 | Continuous Opacity | `kgpu/continuum.py` | `test_continuum` (6.2e-15 photosphere) | `reference/L3.npz`, `kapp_tables.npz` |
| 4 | Line Opacity I: A Single Line | `kgpu/line_opacity.py` | `test_line_opacity` | `reference/L4.npz` |
| 5 | Line Opacity II: The Line List | `kgpu/lines.py`, `kgpu/special_lines.py` | `test_lines`, `test_special_lines` | `reference/L5.npz`, `full_lines_data.npz`, `linetypes.npz` |
| 6 | Hydrogen Lines: Stark Broadening | `kgpu/hydrogen.py` | `test_hydrogen` | `reference/L6.npz` |
| 7 | Radiative Transfer & the Emergent Spectrum | `kgpu/josh.py` (formal solution) | `test_josh` | `josh_tables.npz`, `josh_ck.npz` |
| 8 | The JOSH Solver | `kgpu/josh.py` | `test_josh` | `josh_tables.npz`, `josh_ck.npz` |
| 12 | Molecular Equilibrium & Bands | `kgpu/molecular.py` | `test_molecular` | `diag_tio.npz`, `mol_lines_tio.npz` |
| 13 | Molecular Chemistry (coupled NMOLEC + continuum) | `kgpu/nmolec.py`, `kgpu/mol_continuum.py` | `test_nmolec`, `test_mol_continuum` | `nmolec_*.npz`, `mol_continuum_*.npz` |

The Part IV atmosphere-*structure* primitives that do **not** depend on the convergence-loop fp32
core can also be ported in this batch as their `kgpu` components are individually green:

| L | Lecture | kgpu module | kgpu test | Note |
|---|---------|-------------|-----------|------|
| 9 | Hydrostatic Equilibrium & Temperature Structure | `kgpu/atlas_hydrostatic.py` | `test_atlas_hydrostatic` | TTAUP / pressure structure — single-pass, no convergence loop |
| 10 | Radiative Equilibrium & Temperature Correction | `kgpu/atlas_rt.py`, `kgpu/atlas_rosseland.py`, `kgpu/atlas_tcorr.py` | `test_atlas_rt`, `test_atlas_rosseland`, `test_atlas_tcorr` | the **components** are green on the CPU/fp64 gate; port them with the fp64-on-CPU reference path and flag the fp32 reductions (these are the ones the in-flight fix touches) |

> Note on L10/L11: the individual `kgpu` components (RT sweep, Rosseland fold, TCORR secant) pass
> their parity tests, but they contain exactly the reductions the fp32-convergence-core fix targets
> (the ΔRHOX/TTAUP secant, the Rosseland harmonic fold, the τ_Ross integral). Port L9 and the L10
> *components* now on the CPU/fp64 reference path; defer the L11 **converged loop** (below) until the
> fix lands, so the GPU convergence numbers are final.

### WAIT — the atmosphere-convergence finale (pending the kgpu fp32-core fix)

| L | Lecture | Why it waits |
|---|---------|--------------|
| 11 | Convection & the Converged Atmosphere | the converged continuum-only model runs the full convergence **loop** (`kgpu/atlas_loop.py`, `atlas_convec.py`); the fully-MPS fp32 path diverges (base RHOX → ~8.5) until the precision-critical reductions are fp64-promoted. Port the convection *physics* component now; defer the **converged-loop GPU number** until the fix validates by a full MPS convergence. |
| 14 | A Spectrum from Stellar Parameters, End to End | the **spectrum** assembly is all NOW-ported microphysics, but the Sun's *atmosphere* it runs on is the converged model from L11. Port the synthesis assembly now against a warm-started / NumPy-reference atmosphere; the from-scratch-atmosphere GPU path closes once L11 does. |
| 15 | Line Blanketing: the True Model Atmosphere | ~~reruns the L11 convergence engine with the line blanket on — the same fp32-convergence-core dependency, and the deepest. WAIT.~~ **DONE (as the divergence DIAGNOSTIC).** Ported the line deposit + the convergence-core reductions to depth-batched torch, and used the cell-by-cell fp32-vs-fp64 comparison to **localise** the divergence: per-eval physics (deposit, Rosseland fold, τ integral, each hydrostatic march) holds fp32 parity to the float floor; the divergence enters at the temperature-correction **secant** `(ptot2-ptot1)/ptot1` (catastrophic cancellation, ~41× amplification over its inputs on a realistic step). The lecture ships the **localization**, not a converged GPU number. |
| 16 | The Full Equation of State: Species Slots & Convective Heat Capacity | ~~feeds the L15 line-blanketed loop; its convergence numbers are downstream of the fp32-core fix. WAIT.~~ **DONE.** Ported the per-iteration EOS state (POPSALL species slots, `dopple`/`xnfdop`, `txnxn`, the ionization-energy heat capacity) to depth-batched torch; the cell-by-cell diagnostic **confirms** L15's prediction — every per-eval EOS-state cell holds fp32 parity at the float floor (worst 2.5e-7). Documented the one expected caveat (the fp32 *exponent* floor on trace `xnfdop` slots ~200 dex below the dominant, which underflow to zero — physically irrelevant). |

**Note on L15/L16 (the divergence diagnostic).** These were lifted from WAIT and built as a
**precision diagnostic** rather than a converged-atmosphere number: the user's insight was that the
cell-by-cell GPU(fp32)-vs-numpy(fp64) comparison would *localise* exactly where fp32 peels away.
It did — to one cell, the TCORR secant — confirming the surgical-fp64-promotion design the `kgpu`
fix targets. The lectures run live on MPS/fp32 with a CPU/fp64 twin per cell; no converged-loop GPU
number is claimed (that still waits on the `kgpu` fp32-core fix), only the diagnostic.

**Trigger to lift the remaining WAIT (L11/L14 converged numbers):** `kgpu`'s fp32-convergence-core
fix lands and is **validated by a full MPS convergence** reaching the scalar best-state (per
`~/pykurucz_gpu/PLAN.md §0` and `RESEARCH_LOG`). L11/L14 ship their GPU *components* but not the
final converged-atmosphere GPU number; the lecture text says so explicitly where it applies.

---

## Order of work

1. **PoC: Lecture 2 (EOS).** ✅ Done — see below. Establishes the pattern end to end.
2. **Part I–III microphysics: L3, L4, L5, L6, L7, L8.** The cleanest GPU wins (depth-batched
   opacity + a moment transfer solve); each validated against its `reference/*.npz`.
3. **Part V microphysics: L12, L13.** Molecular equilibrium + continuum.
4. **Part IV components: L9, L10.** Atmosphere-structure primitives on the CPU/fp64 reference path.
5. **WAIT batch: L11, L14, L15, L16.** Port their components; finalize the converged-atmosphere GPU
   numbers once the `kgpu` fp32-convergence-core fix validates.

---

## The proof-of-concept — Lecture 2 (the Equation of State)

The PoC ports the **electron-density solve** — the pedagogical heart of the EOS lecture: the
Boltzmann/partition setup, the Saha ratio, Debye (pressure-ionization) lowering, and the
charge-conservation fixed point that solves for n_e at every depth — to clean depth-batched `torch`,
and validates it against the NumPy edition's `reference/L2.npz`.

- **What is ported:** `saha_ratio`, `ionization_fractions` (the per-element Saha ladder), and
  `solve_electron_density` (the damped charge-balance fixed point), all in `torch`, depth-batched
  (the 80-layer atmosphere is the batch axis; the per-element loop folds into vectorized depth ops).
  The hydrogen-ionization Saha unit test and the H-vs-metal electron-donor attribution are ported
  too, so the lecture's two headline checks (n_e and the H II fraction) both run on the GPU.
- **GPU-native care:** the Saha ladder is accumulated so it cannot overflow fp32's exponent ceiling;
  the device + dtype are chosen once at the top (MPS→fp32, CPU→fp64). This mirrors `kgpu/eos.py`'s
  log-space ladder, stated plainly.
- **The comparison cell:** loads `reference/L2.npz`, compares the GPU `n_e` and `H II fraction` to
  the reference, and reports the max relative deviation, asserting the documented EOS float floor.
- **Deferred within L2 (a deeper continuation, not the PoC):** the full **PFSAHA** per-ion partition
  assembly (the iron-group grid, the hand-built light-element level sums, the occupation correction)
  is `kgpu/eos.py`'s `_build_part_for_element` — already GPU-validated by `test_eos`, and the natural
  next cell to port when L2 is filled out completely. The PoC covers the charge-balance core that
  every later lecture's populations rest on.

**Parity achieved (this PoC run):** recorded in the executed `content/Lecture2.html` comparison cell
(device + max relative deviation vs. `reference/L2.npz`, asserted below the EOS float floor).

---

## Open TODOs / decisions

- **TODO — sync `resources/figures/` from the NumPy edition.** A concurrent agent is adding the
  L14/L15/L16 conceptual schematics to the NumPy book's working tree. This clone is the committed
  snapshot and has the existing 14 schematics (`s1`–`s13` + capstone); once the new L14/L15/L16
  schematics are pushed to the NumPy edition's origin, copy them into `resources/figures/` here.
- **DECISION (flag to author):** whether the GPU edition should keep the identical *figure
  filenames* (`sN_*.png`) and reuse the NumPy edition's schematics verbatim (recommended — the
  physics they illustrate is unchanged), or eventually get GPU-specific schematics (e.g. a
  depth-batched-tensor diagram). Defaulting to **reuse** for now.
- **DEPENDENCY (external):** lifting the WAIT on L11/L14–L16 is gated on the `kgpu`
  fp32-convergence-core fix (read-only dependency; tracked in `~/pykurucz_gpu/PLAN.md §0`). Not a
  decision for this book — just the trigger to finalize those lectures' GPU numbers.
