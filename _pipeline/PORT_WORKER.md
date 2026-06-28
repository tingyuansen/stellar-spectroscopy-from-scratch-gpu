# `port_worker.py` — the GPU-edition port worker (external API does the rewrites; Claude only validates)

The remaining lecture ports are **delegated to the external critic APIs as the code-GENERATION
worker** (GPT-5.5 via LiteLLM, Gemini-3.1-pro via google.genai — the same two models and call
interface as `~/pykurucz_gpu/critic/critic.py`). `port_worker.py` drives them and automates the
parity loop, so Claude tokens are spent **only reviewing the accepted port**. The external API does
ALL torch code generation and prose rewriting; this script generates → parity-gates → opt-squeezes.

This is the token-cheap path for batches L3, L5–L8, L12–L13 (and the components of L9–L11/L14–L16).
Read the [BIBLE](../BIBLE.md) first — the worker feeds it verbatim to the API.

---

## What it drives (per the bible)

For one lecture (a "port job") the worker runs, end to end:

1. **Borrow kgpu's working torch** (bible §6) — it feeds the API, alongside the bible + the numpy
   twin (the parity oracle), kgpu's *existing* validated torch for that component (read-only, sliced
   out of `~/pykurucz_gpu/kgpu/<module>.py`). The API's job is to **reduce/clean** that working
   torch into bite-size readable cells, **not** regenerate the physics from scratch.
2. **Generate** — one API call returns a complete importable torch module (and, for a full-lecture
   job, the rewritten GPU-narrative prose + the comparison cell).
3. **Parity-gate** (bible §1) — the worker runs an automated numpy-vs-torch check in a *subprocess*
   (isolates MPS state / segfaults), parses `PARITY <max_rel_dev>`, accepts iff `< float_floor`.
4. **Loop on failure** — feeds the API the traceback / parity diff and its own prior code, retries
   up to `--max-tries`; surfaces for human review if it never converges.
5. **Optimization-squeeze** (bible §9, `--squeeze N`) — after parity, asks the API "what's the next
   bottleneck?", applies it, and **re-validates parity AND times it** (the harness also prints
   `TIME <ms>`); adopts a change only if it holds parity and is not slower, else reverts.
6. **Vectorization lint** — flags `for`/`while`/`.item()`/`.tolist()`/illegal MPS casts in the
   accepted module, so a parity pass is **not trusted blindly** (a pass is necessary, not sufficient).

Claude then spot-reviews the accepted module (genuinely vectorized? correct in general?), transcribes
it into `build_lecture<N>_gpu.py`, builds + renders, and commits.

---

## Usage

```bash
# port + parity-gate a lecture, choosing the model (default gpt55)
python _pipeline/port_worker.py --job lecture4 --model gpt55  --max-tries 6
python _pipeline/port_worker.py --job lecture4 --model gemini --max-tries 6

# also run N optimization-squeeze rounds after parity (bible §9; parity-gated + timed)
python _pipeline/port_worker.py --job lecture4 --model gpt55 --max-tries 6 --squeeze 2
```

Importable too:

```python
import sys; sys.path.insert(0, "_pipeline")
from port_worker import run_job, JOBS
res = run_job(JOBS["lecture4"], model="gpt55", max_tries=6, squeeze_rounds=2)
# res: {passed, max_rel_dev, time_ms, api_iterations, module_path, lint, squeeze_applied, log}
```

Artifacts land in `_pipeline/_ports/`: `<job>_port.py` (the accepted module),
`<job>_try<k>.log` / `<job>_squeeze<k>.log` (each attempt's run log). `_ports/` is scratch —
the accepted kernel is transcribed into the lecture's `build_lecture<N>_gpu.py`, which is the
source of truth. (Concurrent runs share `_ports/<job>_port.py`; don't run two models on the same
job at once if you need the file preserved — read the per-try logs instead.)

Keys (`LITELLM_API_KEY`, `GOOGLE_API_KEY`) are read from `~/.env`, exactly like `critic/critic.py`.

---

## Registering a new lecture (the only per-lecture work)

Add a `PortJob` to `JOBS` in `port_worker.py`. Five pieces:

| field | what it is |
|---|---|
| `numpy_source` | the NumPy routine(s) to port — the parity oracle's code (paste from `~/Stellar_Spectroscopy_From_Scratch/_pipeline/build_lecture<N>.py`). |
| `spec` | a short port spec: WHAT to port + the entry-point names/shapes the harness will call. |
| `contract` | the exact module API (`port.fn(...) -> ...`) the harness imports and calls. |
| `harness` | python text — a runner body that imports `port` + the numpy twin, compares, and **prints `PARITY <max_rel_dev>`** (and optionally `TIME <ms>` for §9). |
| `kgpu_borrow` | `[(kgpu_module_filename, start_line, end_line)]` — the validated torch to BORROW + REDUCE (read-only). Find it by grepping `~/pykurucz_gpu/kgpu/` for the kernel. |
| `float_floor` | accept iff `max_rel_dev < this` (bible §4: fp32 ~1e-6, tighter where the component is). |

The harness is the contract that makes parity automatic and unfakeable: it rebuilds (or loads) the
**numpy-twin reference** and prints the single `PARITY` number the worker gates on. Keep the kgpu
borrow tight (one routine, not a whole module) so the prompt stays focused.

---

## The L4 test port — what we learned (read this before the next batch)

Ported Lecture 4 (the Voigt profile + single Fe I line-centre opacity) THROUGH the worker:

- **Parity:** `max|rel| = 7.85e-7` (Voigt grid) — under the 1e-6 fp32-MPS floor. Both models hit
  the **identical** number because both correctly borrowed kgpu's `harris_hav` (so the Voigt is
  bit-identical); the assembled single-line kappa lands at 4.31e-7.
- **API iterations:** GPT-5.5 **3**, Gemini-3.1-pro **2**. Both *self-corrected* from the same fp32
  trap on try 1 (4.4e-2 / 6.0e-2) once fed the parity miss — the fix loop works as designed.
- **Which model produced better torch:** roughly a tie, with a split decision. **Gemini** reached
  parity in fewer iterations. **GPT-5.5** produced the faster final kernel and the more
  GPU-meaningful squeeze (pack the 3 Harris tables → a single `index_select`: 6.47→4.53 ms, 30%);
  Gemini's squeeze was a scalar-fold (9.16→7.54 ms). Default to **gpt55**; try **gemini** if gpt55
  stalls.
- **The external APIs hit parity comfortably for this difficulty** (a branchless transliteration of
  a kernel kgpu already solved). They did NOT struggle — so **lean on them** for the microphysics
  batch (L3/L5/L6/L12/L13 are the same shape: borrow the kgpu kernel, reduce, parity-gate). Expect
  Claude to do the *harder* ports (the L7–L8 JOSH solve and the atmosphere-convergence reductions
  that need fp64-promotion, bible §4) where the physics judgement, not the transliteration, is the bottleneck.

### One trap the worker's parity gate caught — and the lecture now teaches it
The reduced frequency `v = (nu - nu0)/dnu_D` is a difference of two ~6e14 Hz numbers whose
difference is ~5e10 — **catastrophic cancellation in fp32** (~12% error in `v`). Both models' first
attempt fell into it; the parity loop rejected it. The fix (bible §2.4): **fp64-promote just that
scalar reduction** on the host, then move the conditioned `v` to the device. The L4 lecture now
teaches this explicitly as the GPU-precision lesson. **Lesson for the harness author:** make the
harness exercise the *assembled* path (not just the kernel in isolation) so cancellation traps
surface — the L4 harness checks both the Voigt grid AND the full single-line kappa.

### A spot-review note (don't trust the lint alone)
The accepted `line_opacity` does the *single-line scalar assembly* on the host (it is genuinely
scalar — one line, one depth) and only the 800-point profile on-device. That is correct and even
sensible for a one-line demo, but it means the lint's `.item()`/`.numpy()` hits are benign
host/device-boundary calls, **not** host loops. Always read the accepted module by eye: confirm the
*batched* part (here the Voigt grid) is truly branchless and loop-free, and that host work is only
genuinely-scalar setup.
