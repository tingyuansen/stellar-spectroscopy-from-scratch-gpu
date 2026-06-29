# Textbook Build Bible

**The single spec every lecture follows.** It is fed verbatim to external code-generation workers
(GPT-5.5 / Gemini-3.1-pro, driven by `_pipeline/port_worker.py`) together with the parity reference
material and a per-lecture spec, so every generated lecture is consistent in standard, structure,
and voice. Claude orchestrates and owns the parity gate; the external APIs do bounded drafting work
against this bible.

---

## 0. What this book is
This is a standalone, GPU-native *Stellar Spectroscopy from Scratch*. Each lecture re-derives the
physics in clean, vectorized **torch/MPS/CUDA** and validates it cell-by-cell against shipped
reference data to the documented float floor. Two payoffs: a readable textbook and an independent
per-part check of the `kgpu` engine. It is a *pedagogical reduction of kgpu*: plain readable torch,
not production imports.

## 1. Prime directive — PARITY (non-negotiable), and the COMPLETENESS gate
Every computation ships with a **comparison cell** against the shipped reference data, printing
`max|rel|` deviation.
- **Acceptance gate:** `max|rel| ≤` the documented float floor for that component (fp32 ~1e-6; tighter where the component is tighter — see §4).
- A parity pass is **necessary but not sufficient**: the code must be genuinely vectorized (no hidden host loops), correct in general (not just on the test case), and read cleanly. Spot-review before accepting.
- **A residual ABOVE the float floor is a real ERROR — FIX it, do NOT just "honestly document" it.**
  The lecture must reproduce the shipped reference over the FULL computation (every depth/layer), not
  only the spectrum-forming photosphere. The ONLY acceptable documented residual is a genuine
  fp32-vs-fp64 float-floor difference. Distinguish carefully: kgpu/reference-vs-**pyk**
  coarse-OS deposit differences can be accepted *method* approximations when explicitly bounded, but
  taught-computation-vs-reference gaps above the float floor are porting or teaching-path bugs. The
  textbook must be a faithful, in-parity representation of `kgpu`'s implementation, not an
  approximation of it.

**COMPLETENESS GATE (hard — added after the truncation incident).** A lecture is not accepted until
it covers the planned **FULL structure**. This is a gate, exactly like parity, enforced by
`port_worker.py` (`completeness_gate` over `parse_structure`):
- **Every planned section is present**, *including the closing sections* — Synthesis / Summary /
  Practice (Exercises) / Further Reading. A lecture missing its closers FAILS the gate.
- **Comparable depth:** the lecture must have enough cells and prose to teach the whole component
  unless a deliberate, documented re-scope justifies fewer (e.g. L5 ports the metal scatter-add only;
  L15/L16 are the fp32-vs-fp64 diagnostics). A short lecture without that justification is treated as
  **truncated** and FAILS.
- **The parity comparison spans the WHOLE computation**, not an early subset — the full physics is ported and checked, so a parity pass on the back half is impossible to skip by stopping early.
- **Why this exists:** GPT-5.5 is a reasoning model that spends the output budget on hidden reasoning before emitting text, so a long single-call lecture hit the length cap and silently dropped its back half + closers while an early-subset parity still passed. `port_worker.py` now (a) **continues-on-length** (re-calls and concatenates when `finish_reason='length'`) and (b) generates **section-by-section / catch-and-fill** (each missing section is a bounded call appended to the validated early cells) so **a single API call can no longer cut a lecture off**.

## 2. Compute standard — full MPS, vectorized, branchless
1. **One device handle, chosen once:** MPS→fp32, CUDA→fp32, CPU→fp64. (Pattern:
   `build_lecture2_gpu.py`.) The book targets **GPU residency** — CPU is the fp64 reference path,
   not the primary path.
   - **Torch-native THROUGHOUT.** Write even the simple/scalar calculations in torch on the device.
     Do not drop to numpy just because a step is trivial. NumPy appears only as comparison/reference
     glue, plotting, or genuinely non-numerical setup.
2. **No Python loops** over depths / wavelengths / lines. Batch over the `(depth, wavelength)` axes with tensor ops.
   - **VECTORIZATION LINT (hard gate — enforced by `port_worker.py` `lint_builder`).** The accepted port is scanned, cell-aware, and a parity pass is **not trusted blindly**: it FLAGS, as a gate, (i) any python `for`/`while` over the big axes (depth/wavelength/line) and any `.item()`/`.tolist()` host-pull in a compute cell, and (ii) any **gratuitous numpy in shipped torch code**. numpy is allowed **ONLY** inside the comparison-reference cell (the parity oracle) and the matplotlib plot cells; everywhere else the shipped path is torch-native. A **small, JUSTIFIED, heterogeneous loop** (~≤ 30 fixed elements — the critics' L2 verdict) is permitted **iff** it carries a one-line justification comment (`# JUSTIFIED-LOOP: <why>`); an un-justified loop or stray numpy FAILS the gate, and the API runs further iterations to eliminate it until the shipped code is fully torch-native vectorized. (`.numpy()`/`.cpu()` at a genuine host/device boundary or in a plot cell is benign — spot-review distinguishes a boundary bounce from a host loop.)
3. **Branchless:** recast data-dependent `if/else` as boolean-mask × arithmetic, `torch.where`, or `gather`/one-hot selection. Reserve real control flow only for genuinely expensive branches.
4. **Precision:** fp32 is the default working dtype on MPS. Where a reduction suffers **catastrophic cancellation** (a secant/difference of two nearly-equal sums) or **fp32 accumulation drift** (a long harmonic/optical-depth accumulation), **fp64-promote just that reduction** via a CPU offload — keep it tiny (scalars / per-depth vectors), bulk stays MPS-resident.
5. **Metal kernels:** use a custom `torch.mps.compile_shader` kernel **only** for scatter-heavy hot paths where torch vectorization becomes a per-element dispatch storm (line-opacity accumulation, the atmosphere deposit). Everything else: plain batched torch. When unsure, **consult the critics**.

## 3. MPS gotchas (learned the hard way — always apply)
- `tensor.to("cpu", torch.float64)` **RAISES** on MPS (it casts on-device first). Use **`tensor.cpu().to(torch.float64)`** — move to CPU, *then* cast.
- Accumulators are fp32 on MPS; never reduce a wide-dynamic-range sum in fp16/bf16.
- `Math.random`-style nondeterminism and fp64 ops are unavailable on Metal — design around them.

## 4. Per-component parity floors (the gate values, from kgpu validation)
| Component (lecture) | Float floor to expect (fp32 on MPS) |
|---|---|
| EOS / Saha / PFSAHA (L2) | ~1.5e-6 |
| Continuum opacity (L3) | ~machine (fp32 ~1e-6) |
| Single line / Voigt (L4) | ~1e-6 |
| Line list accumulation (L5) | ~1e-6 (scatter-add — Metal-kernel candidate) |
| Hydrogen Stark (L6) | ~1e-6 |
| Formal RT / JOSH (L7–L8) | float32 JOSH floor ~1e-8…1e-6 |
| Molecular eq. + bands (L12–L13) | ~1e-6 |
| **Atmosphere convergence (L11/L14/L15/L16)** | **GATED** on the kgpu fp32-convergence-core fix (the secant + Rosseland fold need fp64-promotion); until then, expect the *components* at floor and the *convergence loop* to diverge in pure fp32 — document that honestly. |

## 5. Lecture format — house style
- **Bite-size code cells (~3–30 lines)** interleaved with pedagogical markdown + equations. **No long code blocks.**
- Anatomy per lecture: title cell → `**Learning objectives.**` callout → for each step: intuition → the equation → the vectorized-torch cell → **the reference comparison cell** (print `max|rel|`) → closing Synthesis / Summary / Practice / Further Reading.
- **Voice:** pedagogical, no hype, equation-complete, every input/parameter and its units explained.
- **Figures:** use the shipped `resources/figures/sN_*.png` schematics where they remain the clearest physics picture.
- **The GPU book's added pedagogy:** each lecture also teaches the *vectorization* — how the loop became a tensor op, why branchless, where fp32 is safe vs. needs fp64-promotion, when a Metal kernel earns its place. Make that explicit; it is the book's distinct value.
- **Standalone prose.** Preserve the proven physics explanations where they still fit, but do not frame a lecture as a port, companion, or translation. Change or add what the GPU story requires: loop→tensor narrative, precision budget, kernel choices, and any shared definitions needed for self-consistency.

## 6. References every lecture may use
- **Shipped reference data** (`reference/*.npz`, generated by pykurucz and the independent textbook checks) is the parity target. The lecture can load it for setup and comparison, but the taught computation must be implemented in the notebook itself.
- **kgpu is a read-only algorithm reference.** kgpu (`~/pykurucz_gpu`) has the product implementation. Read it to understand the validated algorithm, then write the lecture's own self-contained torch, reduced into bite-size cells. The lecture must **NEVER `import kgpu`**. Relationship: kgpu is an assembly of these building blocks, and improvements discovered here can port back to kgpu after parity tests.
- **This bible** + the per-lecture port spec.
- Never modify `~/Stellar_Spectroscopy_From_Scratch`, `~/pykurucz_gpu`, or `~/pykurucz`.

## 7. Orchestration (how a port is produced)
1. Claude (orchestrator) picks the lecture + writes the short port spec.
2. `_pipeline/port_worker.py` sends **this bible + shipped reference context + kgpu's existing torch implementation of the component + the spec** to the external API (GPT-5.5 or Gemini-3.1-pro), which drafts the full GPU lecture: standalone prose, vectorized torch code, and comparison cells.
3. `port_worker` auto-runs the comparison vs the shipped reference; on failure it feeds the API the traceback/diff and **retries** until parity (or a max-tries cap → surface for review).
4. Claude spot-reviews the accepted port (genuinely vectorized? correct generally?), then it is built into `content/LectureN.html` and committed.
- **Port order:** ground-up — microphysics first (L2→L3→L4→L5→L6→L7→L8→L12→L13), atmosphere lectures (L11/L14/L15/L16) gated per §4. See `PLAN.md`.

## 8. Git discipline (concurrent agents)
Multiple agents write this repo on **disjoint files**. Before every push: `git pull --rebase origin main`; retry on reject. Touch only your lecture's `build_lectureN_gpu.py` + `content/LectureN.html` (+ shared `_pipeline/` tooling when explicitly building it).

## 9. Optimization squeeze (per lecture — API-driven, parity-gated)
Once a lecture reaches parity, run an **optimization-squeeze pass** with the external API, lecture by lecture — this is the book's analogue of the kgpu vectorization audit. Ask GPT-5.5 / Gemini, iteratively, "what is the next bottleneck, and how do we remove it?" and apply: eliminate any remaining loop, fuse adjacent kernels, precompute and reuse invariants, replace a torch dispatch storm with a Metal kernel, pick a better memory layout / contiguity.
- **Every optimization is re-validated against the shipped reference (the §1 parity gate) AND timed** (record the before/after wall-clock or dispatch count). Adopt only those that hold parity.
- **Document each in the lecture** — the "why this is faster" is core pedagogy for a GPU book.
- Keep squeezing until the lecture is fully MPS-vectorized with **no un-justified loop left** and the hot path is at its kernel optimum. The external API drives the rewrites; Claude reviews each accepted change against parity + timing. (NOTE: the squeeze is a SEPARATE, Claude-reviewed pass after the port reaches parity — NOT inline — because the API's squeeze judgment is unreliable: it once regressed a vectorized kernel to a scalar host-loop just to pass parity. Ship the parity-passing vectorized port first; squeeze + review after.)

## 10. FINAL ACCEPTANCE CRITERIA — every lecture gated against ALL (user, definitive)
1. **No information drop.** Covers the original numpy lecture's FULL material — every physics piece, every section, the schematic; nothing dropped (the completeness gate, §1).
2. **As GPU-native as possible** — torch-native throughout (§2), including trivial calcs.
3. **Maximum squeezing** — no silly loops, no inefficient slicing/indexing, no numpy where a torch alternative exists (the vectorization lint, §2.2).
4. **Follow kgpu's implementation** where possible (clear 1:1 correspondence) AND flag anything learned here to port BACK to kgpu — bidirectional; these become kgpu's squeeze ideas.
5. **Parity, no breaking** — both the lecture AND any proposed kgpu back-port are parity-tested against the shipped references / pyk; nothing breaks.
6. **Format may change** where the original narrative is genuinely at odds with the GPU-narrative — there the GPU story (loop→tensor, the precision budget, kernel choices) WINS; don't force GPU code into a numpy-shaped narrative. Preserve the original where it fits.
7. **House style** still applies — bite-size cells, spacing between code, well-commented code, walk-through markdown between cells, the closing sections.
8. **Orchestration** — the external APIs do the expensive generation + iterations; Claude orchestrates + gates; a final Claude read-through at the very end (gates green + token-efficient) catches what the automated gates can't.
