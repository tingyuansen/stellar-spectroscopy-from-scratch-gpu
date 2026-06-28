# PLAN — the GPU-substitution roadmap

*Stellar Spectroscopy from Scratch — GPU Edition.* This file is the working map for maintaining the
clean **torch/MPS** companion to the NumPy edition. It states the per-lecture pattern, the current
status of the 16-lecture GPU arc, and the open TODOs. Read `PASSDOWN.md` first for the live handoff
state, then this file alongside `STRUCTURE.md` (the four-part arc) and `README.md` (the prose).

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

## Current status by lecture

The standing rule remains `kgpu`'s own: this textbook reads `kgpu` as the implementation clue source
but never imports it. The target state is one self-contained torch/MPS block per lecture, validated
against the NumPy reference to the documented floor. The current state is close but not final:
L1-7, L9-13, and L15-L16 have GPU builders; L8 and L14 are still NumPy-style builders, and the
accepted heavy paths in L15/L16 still rely on clean-room NumPy helper modules for some fidelity
gates. L14 was refreshed after the line-blanketed atmosphere work: the solar capstone now uses the
Part-VI solar target (`base RHOX = 12.1439331`, `base T = 11425 K`), not the stale continuum-only
`RHOX = 10.5357` bundle, but it still loads atmosphere/EOS-state intermediates and is therefore a
synthesis-half capstone until L15/L16 are wired directly into it.

### Microphysics validated by kgpu's test suite

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

The Part IV atmosphere-*structure* primitives are also ported as their `kgpu` components are green:

| L | Lecture | kgpu module | kgpu test | Note |
|---|---------|-------------|-----------|------|
| 9 | Hydrostatic Equilibrium & Temperature Structure | `kgpu/atlas_hydrostatic.py` | `test_atlas_hydrostatic` | TTAUP / pressure structure — single-pass, no convergence loop |
| 10 | Radiative Equilibrium & Temperature Correction | `kgpu/atlas_rt.py`, `kgpu/atlas_rosseland.py`, `kgpu/atlas_tcorr.py` | `test_atlas_rt`, `test_atlas_rosseland`, `test_atlas_tcorr` | components green; precision-critical reductions are named and promoted where needed |

### Atmosphere and capstone state

| L | Lecture | Current status |
|---|---------|----------------|
| 11 | Convection & the Converged Atmosphere | continuum-only convergence machinery and convection physics are ported; the lecture remains the scaffold for Part VI. |
| 14 | A Spectrum from Stellar Parameters, End to End | refreshed and verified as a synthesis-half capstone. The Sun uses the Part-VI line-blanketed solar target (`RHOX=12.1439331`, `T=11425 K`); hot/giant/M-dwarf structures are documented emulator warm-starts. Opacity and transfer are computed in the notebook with no `kgpu`/`pykurucz` import, but atmosphere/EOS/population/Doppler intermediates are still loaded integration inputs. |
| 15 | Line Blanketing: the True Model Atmosphere | built as the line-blanketing / precision-critical reduction lesson. The deposit, Rosseland fold, optical-depth integral, and hydrostatic pieces are checked at the fp32 floor; the TCORR secant is identified as a surgical promotion point. The accepted LINOP1 fidelity gate still uses the clean-room scalar verifier path, so this is not the final all-torch deposit closure. |
| 16 | The Full Equation of State: Species Slots & Convective Heat Capacity | built. The per-iteration state (`POPSALL` species slots, Doppler widths, `TABCONT`, molecular slots, ionization-energy heat capacity) is packed/audited in torch and checked against the reference floor, with the expected fp32 exponent-floor caveat on negligible trace slots. The PFSAHA/NELECT/continuum/molecular source computations still rely on clean-room NumPy helper modules and must be ported for the final GPU-native book. |

The production `kgpu` passdown now resolves the deep-base RHOX question: kgpu and the independent
from-scratch oracle agree in the 12.3-class fixed point, while pyk's exact line-blanketed solar
reference has base `RHOX=12.1439331`. L14 must never be regressed to the old continuum-only
`10.5357` solar bundle.

---

## Order of work

1. **PoC: Lecture 2 (EOS).** ✅ Done — see below. Establishes the pattern end to end.
2. **Part I–III microphysics: L3, L4, L5, L6, L7, L8.** The cleanest GPU wins (depth-batched
   opacity + a moment transfer solve); each validated against its `reference/*.npz`.
3. **Part V microphysics: L12, L13.** Molecular equilibrium + continuum.
4. **Part IV components: L9, L10.** Atmosphere-structure primitives on the CPU/fp64 reference path.
5. **Atmosphere/capstone pass:** keep L11-L16 synchronized with `kgpu/PASSDOWN.md`, especially
   the L14 solar-state guardrail and the L15/L16 precision-promotion narrative.

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
- **DEPENDENCY (read-only clue source):** continue reading `~/pykurucz_gpu/PASSDOWN.md` and
  `~/pykurucz_gpu/PLAN.md` before changing L14-L16. `kgpu` is stable and self-verified; this book
  should reflect its implementation pieces without importing it.
