# GPU Edition — Port Bible

**The single spec every lecture port follows.** It is fed verbatim to the external code-generation workers (GPT-5.5 / Gemini-3.1-pro, driven by `_pipeline/port_worker.py`) together with the numpy twin and a per-lecture port spec, so every generated port is consistent in standard, structure, and voice. Claude orchestrates and owns the parity gate; the external APIs do the heavy lifting against this bible.

---

## 0. What this book is
The GPU/vectorized companion to *Stellar Spectroscopy from Scratch*. Each lecture re-derives the **same physics, same numbers** as the numpy edition, but in clean, fully-vectorized **torch/MPS**, and **validates it cell-by-cell against the numpy twin** to the documented float floor. Two payoffs: a GPU-native textbook, and an independent per-part check of the kgpu engine. It is a *pedagogical reduction of kgpu* exactly as the numpy book is a pedagogical reduction of pykurucz — plain readable torch, not production imports.

## 1. Prime directive — PARITY (non-negotiable)
Every ported computation ships with a **comparison cell** against the numpy twin (the same lecture in `~/Stellar_Spectroscopy_From_Scratch`), printing `max|rel|` deviation.
- **Acceptance gate:** `max|rel| ≤` the documented float floor for that component (fp32 ~1e-6; tighter where the component is tighter — see §4).
- A parity pass is **necessary but not sufficient**: the code must be genuinely vectorized (no hidden host loops), correct in general (not just on the test case), and read cleanly. Spot-review before accepting.

## 2. Compute standard — full MPS, vectorized, branchless
1. **One device handle, chosen once:** MPS→fp32, CUDA→fp32, CPU→fp64. (Pattern: `build_lecture2_gpu.py`.) The book targets **full MPS/GPU residency** — CPU is the fp64 reference path, not the shipped path.
   - **Torch-native THROUGHOUT.** Write even the simple/scalar calculations in torch on the device — don't drop to numpy just because a step is trivial. This is the GPU edition; keep it uniformly GPU-native end to end. numpy appears ONLY as the comparison-cell reference (the parity oracle) and for genuinely non-numerical setup.
2. **No Python loops** over depths / wavelengths / lines. Batch over the `(depth, wavelength)` axes with tensor ops.
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

## 5. Lecture format — house style (match the numpy edition exactly)
- **Bite-size code cells (~3–30 lines)** interleaved with pedagogical markdown + equations. **No long code blocks.**
- Anatomy per lecture: title cell → `**Learning objectives.**` callout → for each step: intuition → the equation → the vectorized-torch cell → **the numpy-vs-GPU comparison cell** (print `max|rel|`) → closing Synthesis / Summary / Practice / Further Reading.
- **Voice:** pedagogical, no hype, equation-complete, every input/parameter and its units explained. Same register as the numpy book.
- **Figures:** reuse the numpy edition's `resources/figures/sN_*.png` schematics verbatim (physics unchanged).
- **The GPU book's added pedagogy:** each lecture also teaches the *vectorization* — how the loop became a tensor op, why branchless, where fp32 is safe vs. needs fp64-promotion, when a Metal kernel earns its place. Make that explicit; it is the book's distinct value.
- **Preserve the original pedagogical text + flavor as much as possible; do NOT gratuitously rewrite.** Keep the numpy edition's explanations, equations, and voice; change or ADD only what accuracy and the GPU story require — the loop→tensor narrative, the precision budget (fp32-safe vs needs fp64), the kernel choices. A *massive* rewrite is allowed where the structure genuinely changed, but the prose should still read as the trusted original with GPU-native adaptations woven in, not a fresh essay. The external API does the rewrite (NEVER Claude) and may iterate several passes; Claude gates accuracy + vectorization + that the original flavor survived. The whole book must stay self-consistent across lectures — if a port changes a shared definition, the API updates the dependent lectures' text too.

## 6. References every port consumes
- **The numpy twin** (the fp64 truth + parity target): `~/Stellar_Spectroscopy_From_Scratch/<lecture>`.
- **The torch starting point** (READ-ONLY, BORROW it): kgpu `~/pykurucz_gpu` already has a validated torch/MPS implementation of most components — *our partial success*. **Start from it** — pedagogically reduce/clean kgpu's working torch into bite-size readable cells; do NOT regenerate the physics from scratch. The numpy twin is the parity oracle; kgpu's torch is the working basis.
- **This bible** + the per-lecture port spec.
- Never modify `~/Stellar_Spectroscopy_From_Scratch`, `~/pykurucz_gpu`, or `~/pykurucz`.

## 7. Orchestration (how a port is produced)
1. Claude (orchestrator) picks the lecture + writes the short port spec.
2. `_pipeline/port_worker.py` sends **this bible + the numpy twin (prose + code, the parity oracle) + kgpu's existing torch implementation of the component (the partial-success basis to borrow + reduce) + the spec** to the external API (GPT-5.5 or Gemini-3.1-pro), which **generates the FULL GPU lecture** — the markdown prose rewritten to the GPU/vectorization narrative (§5), the vectorized torch code (reduced from kgpu's working torch, not from scratch), and the comparison cell — all adhering to this bible.
3. `port_worker` auto-runs the comparison vs the numpy twin; on failure it feeds the API the traceback/diff and **retries** until parity (or a max-tries cap → surface for review).
4. Claude spot-reviews the accepted port (genuinely vectorized? correct generally?), then it is built into `content/LectureN.html` and committed.
- **Port order:** ground-up — microphysics first (L2→L3→L4→L5→L6→L7→L8→L12→L13), atmosphere lectures (L11/L14/L15/L16) gated per §4. See `PLAN.md`.

## 8. Git discipline (concurrent agents)
Multiple agents write this repo on **disjoint files**. Before every push: `git pull --rebase origin main`; retry on reject. Touch only your lecture's `build_lectureN_gpu.py` + `content/LectureN.html` (+ shared `_pipeline/` tooling when explicitly building it).

## 9. Optimization squeeze (per lecture — API-driven, parity-gated)
Once a lecture reaches parity, run an **optimization-squeeze pass** with the external API, lecture by lecture — this is the book's analogue of the kgpu vectorization audit. Ask GPT-5.5 / Gemini, iteratively, "what is the next bottleneck, and how do we remove it?" and apply: eliminate any remaining loop, fuse adjacent kernels, precompute and reuse invariants, replace a torch dispatch storm with a Metal kernel, pick a better memory layout / contiguity.
- **Every optimization is re-validated against the numpy twin (the §1 parity gate) AND timed** (record the before/after wall-clock or dispatch count). Adopt only those that hold parity.
- **Document each in the lecture** — the "why this is faster" is core pedagogy for a GPU book.
- Keep squeezing until the lecture is fully MPS-vectorized with **no un-justified loop left** and the hot path is at its kernel optimum. The external API drives the rewrites; Claude reviews each accepted change against parity + timing.
