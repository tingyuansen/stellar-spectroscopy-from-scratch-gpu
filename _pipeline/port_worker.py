#!/usr/bin/env python
"""port_worker.py — the GPU-edition PORT WORKER.

Delegate the numpy->torch/MPS rewrites for the GPU edition to the EXTERNAL critic APIs
(GPT-5.5 via LiteLLM, Gemini-3.1-pro via google.genai) used as the code-GENERATION worker,
and spend Claude tokens only REVIEWING the accepted port. The external API does ALL code
generation; this script automates the parity loop.

The loop, per lecture (a "port job"):
  1. Build a generation prompt from the lecture's NumPy source + a short PORT SPEC + the
     fixed GPU-edition conventions + a strict OUTPUT CONTRACT (the function name + signature
     the parity harness will call, and the numpy reference it is checked against).
  2. Call the chosen external API -> extract the fenced ```python``` block -> write it to a
     candidate module file.
  3. Run an AUTOMATED numpy-vs-torch parity check in a SUBPROCESS (isolates segfaults / MPS
     state): import the candidate, call its entry point, compare max relative deviation vs the
     documented float floor (~1e-6 fp32 MPS / machine precision fp64 CPU).
  4. LOOP: on an import/run traceback OR a parity miss, feed the failure text back to the API
     ("here is your previous code and how it failed; return a corrected full module") and retry,
     up to --max-tries. On success, stop and surface the accepted module for human review.
  5. Print a VECTORIZATION LINT of the accepted module (flags `for`/`while` host loops, `.item()`
     in loops, `.tolist()`, python-level element indexing) so a parity pass is not trusted blindly.

Driveable two ways:
  CLI:        python _pipeline/port_worker.py --job lecture4 --model gpt55 --max-tries 6
  importable: from port_worker import run_job, PortJob ; run_job(PortJob(...))

A "job" is registered in JOBS below (a PortJob dataclass). To port a new lecture, add a job:
its numpy source slice, its port spec, and a tiny parity HARNESS function (pure python text)
that the worker writes into the runner — it imports the candidate torch module + the numpy twin,
calls both, and prints `PARITY <max_rel_dev>` (or raises). The worker reads that number.

Keys (LITELLM_API_KEY, GOOGLE_API_KEY) are read from ~/.env, exactly like critic/critic.py.
The external APIs are READ-ONLY consumers of source we paste; nothing here edits kgpu or the
NumPy edition.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BOOK = Path(__file__).resolve().parent.parent          # ~/Stellar_Spectroscopy_From_Scratch_GPU
PORTS = BOOK / "_pipeline" / "_ports"                   # candidate modules + run logs land here
PORTS.mkdir(exist_ok=True)

NUMPY_BOOK = Path.home() / "Stellar_Spectroscopy_From_Scratch"   # the numpy twin (parity oracle)
KGPU = Path.home() / "pykurucz_gpu" / "kgpu"                     # kgpu's working torch (BORROW basis)
BIBLE = (BOOK / "BIBLE.md")                                      # the GPU-edition port bible


def _slice(path: Path, start: int, end: int) -> str:
    """Read a line-range [start, end] (1-based, inclusive) from a file — used to BORROW the
    relevant routine out of kgpu's torch (READ-ONLY) without pasting an entire module."""
    lines = path.read_text().splitlines()
    return "\n".join(lines[start - 1:end])

# ── API keys (names only into the env; never printed) ────────────────────────
for _ln in (Path.home() / ".env").read_text().splitlines():
    if "=" in _ln and not _ln.strip().startswith("#"):
        _k, _v = _ln.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

GPT_MODEL = "gpt-5.5-2026-04-24"
GEM_MODEL = "gemini-3.1-pro-preview"


# ── the external code-GENERATION worker (same call interface as critic/critic.py) ──
def ask_gpt(messages: list[dict]) -> str:
    """messages: a chat list [{'role','content'}, ...]. Returns the assistant text."""
    from openai import OpenAI
    c = OpenAI(base_url="https://litellm.cloud.osu.edu", api_key=os.environ["LITELLM_API_KEY"])
    r = c.chat.completions.create(model=GPT_MODEL, max_tokens=16000, messages=messages)
    return r.choices[0].message.content or ""


def ask_gemini(messages: list[dict]) -> str:
    """google.genai has no roles list; flatten the chat into one contents string."""
    from google import genai
    g = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    flat = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
    return g.models.generate_content(model=GEM_MODEL, contents=flat).text or ""


MODELS: dict[str, Callable[[list[dict]], str]] = {"gpt55": ask_gpt, "gemini": ask_gemini}


# ── the fixed GPU-edition conventions every port must obey (the house style) ──
CONVENTIONS = r"""GPU-EDITION CONVENTIONS (obey ALL — these are non-negotiable house rules):
- Target torch on Apple MPS (fp32) with a CPU fp64 fallback. Pick the device + dtype ONCE:
      if torch.backends.mps.is_available(): DEVICE, DTYPE = torch.device("mps"), torch.float32
      elif torch.cuda.is_available():       DEVICE, DTYPE = torch.device("cuda"), torch.float32
      else:                                 DEVICE, DTYPE = torch.device("cpu"), torch.float64
- FULLY VECTORIZED, BRANCHLESS. NO python `for`/`while` over data, NO `.item()`/`.tolist()` in
  a loop, NO per-element python indexing. Regime branches (if a<0.2 ... elif ... else) MUST be
  folded into `torch.where` masks so every lane does the same work and all branches are computed
  then selected. A `for` over a tiny FIXED small set of distinct scalar parameter values (e.g. the
  6 damping values in the reference grid) is acceptable ONLY if each iteration is itself a fully
  vectorized tensor call; prefer broadcasting to a 2-D grid if you can.
- MPS GOTCHA (this RAISES): `tensor.to('cpu', torch.float64)`. You MUST instead do
  `tensor.detach().cpu().to(torch.float64)` — move to CPU FIRST, then cast (MPS has no fp64).
- Table lookups (e.g. Harris h0/h1/h2 tables) become `torch.index_select` / fancy indexing with a
  clamped integer index tensor, NOT a python loop.
- Keep the SAME numeric constants and the SAME branch thresholds as the numpy source bit-for-bit
  (this is a transliteration to torch, not a re-derivation). Reproduce the reference's seams.
- Pure compute. Do NOT import kgpu or pykurucz. Do NOT read/write files. Do NOT print. Only torch
  (+ numpy/math if you must) and the standard library.
"""

GEN_SYSTEM = (
    "You are an expert GPU-numerics engineer. You transliterate scientific NumPy into clean, "
    "FULLY-VECTORIZED, BRANCHLESS PyTorch that runs on Apple MPS (fp32) with a CPU fp64 fallback, "
    "reproducing the NumPy result to the float floor (~1e-6 fp32 / machine precision fp64). You "
    "return EXACTLY ONE fenced ```python ... ``` code block: a complete, self-contained, importable "
    "module — no prose, no commentary outside the block."
)


@dataclass
class PortJob:
    """One lecture's port task. The worker feeds the API the BIBLE + numpy twin (parity oracle)
    + kgpu's existing torch (the partial-success BASIS to reduce, per bible §6) + the spec, then
    runs `harness` (python text) which imports the generated module + checks parity vs the numpy
    twin, printing `PARITY <max_rel_dev>`.
    """
    name: str                       # job id, e.g. "lecture4"
    numpy_source: str               # the NumPy routine(s) to port (the parity oracle's code)
    spec: str                       # short port spec: WHAT to port, the entry-point contract
    contract: str                   # the exact module API the harness will call (names/signatures)
    harness: str                    # python text: a runner body that prints `PARITY <float>`
    float_floor: float = 1e-6       # accept if max_rel_dev < this (the documented fp32 floor)
    preamble: str = ""              # optional extra context (reference field names, etc.)
    kgpu_borrow: list[tuple[str, int, int]] = field(default_factory=list)
    # ^ [(kgpu_module_filename, start_line, end_line)] — the validated torch to BORROW + REDUCE.


def _kgpu_basis(job: PortJob) -> str:
    """Assemble kgpu's existing torch for this component — the partial-success basis the API
    REDUCES into clean bite-size cells (bible §6: borrow, don't regenerate). READ-ONLY."""
    if not job.kgpu_borrow:
        return ""
    chunks = []
    for fname, s, e in job.kgpu_borrow:
        src = _slice(KGPU / fname, s, e)
        chunks.append(f"# --- kgpu/{fname}  (lines {s}-{e}) ---\n{src}")
    return "\n\n".join(chunks)


def _bible_text() -> str:
    return BIBLE.read_text() if BIBLE.exists() else "(BIBLE.md not found)"


# ── prompt assembly ──────────────────────────────────────────────────────────
def build_gen_messages(job: PortJob) -> list[dict]:
    basis = _kgpu_basis(job)
    basis_block = (
        "--- kgpu's EXISTING torch for this component (THE PARTIAL-SUCCESS BASIS — BORROW + REDUCE "
        "this into clean bite-size form; do NOT regenerate the physics from scratch; it is already "
        "branchless/vectorized and validated) ---\n"
        f"{basis}\n--- END kgpu BASIS ---\n\n"
    ) if basis else ""
    user = f"""Produce a fully-vectorized torch/MPS port that REDUCES kgpu's working torch into clean,
readable form and reproduces the NumPy twin to the float floor.

=== THE GPU-EDITION PORT BIBLE (obey it) ===
{_bible_text()}
=== END BIBLE ===

{CONVENTIONS}

PORT SPEC:
{job.spec}

OUTPUT CONTRACT (the parity harness will import your module and call EXACTLY this):
{job.contract}

{('CONTEXT:\\n' + job.preamble + '\\n\\n') if job.preamble else ''}{basis_block}--- NUMPY TWIN — THE PARITY ORACLE (your output must reproduce its numbers) ---
{job.numpy_source}
--- END NUMPY TWIN ---

Return ONE ```python``` block: the complete importable module that satisfies the OUTPUT CONTRACT,
reduced from the kgpu basis above (not re-derived), matching the numpy twin bit-for-bit in constants
and branch thresholds."""
    return [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}]


def build_fix_message(prev_code: str, failure: str, floor: float) -> str:
    return f"""Your previous module FAILED the automated numpy-vs-torch parity check.

--- YOUR PREVIOUS MODULE ---
{prev_code}
--- END ---

--- HOW IT FAILED (traceback or parity miss; parity must be < {floor:.1e}) ---
{failure}
--- END ---

Fix the cause and return the COMPLETE corrected module as ONE ```python``` block (no prose).
Remember the house rules: fully vectorized, branchless, torch.where for regimes, MPS-safe
`.detach().cpu().to(torch.float64)` cast, same constants/thresholds as the numpy source."""


CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    """Pull the (last, largest) fenced python block; fall back to the whole reply."""
    blocks = CODE_RE.findall(reply or "")
    if not blocks:
        return (reply or "").strip()
    return max(blocks, key=len).strip()


# ── the parity runner (subprocess-isolated) ──────────────────────────────────
def run_parity(job: PortJob, module_path: Path) -> tuple[bool, float | None, float | None, str]:
    """Write a runner that imports the candidate module + runs job.harness, execute it in a
    SUBPROCESS, and parse `PARITY <float>` (plus optional `TIME <ms>` for the §9 squeeze).
    Returns (passed, max_rel_dev, time_ms, log)."""
    runner = PORTS / f"_run_{job.name}.py"
    runner.write_text(textwrap.dedent(f"""\
        import sys, importlib.util, pathlib
        BOOK = pathlib.Path({str(BOOK)!r})
        # import the candidate torch module under test as `port`
        spec = importlib.util.spec_from_file_location("port", {str(module_path)!r})
        port = importlib.util.module_from_spec(spec)
        sys.modules["port"] = port
        spec.loader.exec_module(port)
        # --- harness: imports the numpy twin + port, must print `PARITY <max_rel_dev>` ---
        # --- (optionally also `TIME <ms>` for the §9 optimization-squeeze timing) ---
    """) + job.harness)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True,
                          cwd=str(BOOK), timeout=600)
    dt = time.time() - t0
    log = f"[exit {proc.returncode}, {dt:.1f}s]\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    if proc.returncode != 0:
        return False, None, None, log
    m = re.search(r"PARITY\s+([0-9eE.+-]+)", proc.stdout)
    if not m:
        return False, None, None, "no `PARITY <float>` line in stdout\n" + log
    dev = float(m.group(1))
    mt = re.search(r"TIME\s+([0-9eE.+-]+)", proc.stdout)
    tms = float(mt.group(1)) if mt else None
    return (dev < job.float_floor), dev, tms, log


# ── vectorization lint (so a parity pass is not trusted blindly) ──────────────
LINT_PATTERNS = [
    (re.compile(r"^\s*for\b"), "python `for` loop over data (host loop?)"),
    (re.compile(r"^\s*while\b"), "python `while` loop (host loop?)"),
    (re.compile(r"\.item\(\)"), "`.item()` (host scalar pull — bad inside a loop)"),
    (re.compile(r"\.tolist\(\)"), "`.tolist()` (host materialization)"),
    (re.compile(r"\.to\(\s*['\"]cpu['\"]\s*,\s*torch\.float64"), "MPS-illegal `.to('cpu', float64)` cast"),
    (re.compile(r"\.numpy\(\)"), "`.numpy()` (host bounce — fine at the boundary, flag if mid-compute)"),
]


def lint(code: str) -> list[str]:
    hits = []
    for i, line in enumerate(code.splitlines(), 1):
        for pat, msg in LINT_PATTERNS:
            if pat.search(line):
                hits.append(f"  L{i}: {msg}  |  {line.strip()[:80]}")
    return hits


SQUEEZE_MSG = """The module above PASSES parity (max_rel_dev = {dev:.3e} < {floor:.1e}) and the
harness times it at {tms:.3f} ms. Now do ONE optimization-squeeze step (bible §9): identify the
SINGLE biggest remaining bottleneck (a leftover loop, an unfused pair of kernels, a recomputed
invariant, a host bounce, a non-contiguous layout, a dispatch storm that wants a Metal kernel) and
remove it — WITHOUT changing the result. The OUTPUT CONTRACT and every numeric constant/threshold
must stay identical; this must still pass the same parity gate. Return the COMPLETE optimized module
as ONE ```python``` block, with a one-line `# SQUEEZE:` comment at the top stating what you changed
and why it is faster. If the module is already at its kernel optimum and you cannot make it provably
faster without risking parity, return it UNCHANGED with a top comment `# SQUEEZE: none — already optimal: <reason>`."""


def optimization_squeeze(job, model, messages, last_reply, code, dev, tms, module_path, say, rounds):
    """Bible §9: after parity, iteratively ask the API for the next bottleneck, apply, and
    RE-VALIDATE parity + time. Adopt a change only if it holds parity AND is not slower.
    Returns (best_code, best_dev, best_tms, applied_changes)."""
    best_code, best_dev, best_tms = code, dev, tms
    applied = []
    convo = list(messages) + [{"role": "assistant", "content": last_reply}]
    for r in range(1, rounds + 1):
        if best_tms is None:
            say("  squeeze: harness emits no `TIME` line — skipping timed squeeze.")
            break
        say(f"\n[squeeze {r}/{rounds}] asking {model} for the next bottleneck "
            f"(current {best_tms:.3f} ms, dev {best_dev:.2e}) ...")
        convo = convo + [{"role": "user",
                          "content": SQUEEZE_MSG.format(dev=best_dev, floor=job.float_floor, tms=best_tms)}]
        try:
            reply = ask_of(model)(convo)
        except Exception as e:
            say(f"  squeeze API ERROR: {type(e).__name__}: {str(e)[:160]}")
            break
        convo = convo + [{"role": "assistant", "content": reply}]
        cand = extract_code(reply)
        note = next((ln for ln in cand.splitlines() if ln.strip().startswith("# SQUEEZE:")), "# SQUEEZE: (unlabeled)")
        if "none" in note.lower():
            say(f"  {note.strip()}  -> stop."); break
        module_path.write_text(cand)
        ok2, dev2, tms2, log2 = run_parity(job, module_path)
        (PORTS / f"{job.name}_squeeze{r}.log").write_text(log2)
        if ok2 and tms2 is not None and tms2 <= best_tms * 1.02:
            say(f"  ADOPT: {note.strip()}   {best_tms:.3f} -> {tms2:.3f} ms   dev {dev2:.2e}")
            best_code, best_dev, best_tms = cand, dev2, tms2
            applied.append(dict(round=r, note=note.strip(), ms_before=tms, ms_after=tms2, dev=dev2))
        else:
            why = (f"parity fail (dev {dev2})" if not ok2 else
                   f"not faster ({tms2:.3f} vs {best_tms:.3f} ms)" if tms2 is not None else "no TIME")
            say(f"  REJECT ({why}): {note.strip()}  -> revert.")
            module_path.write_text(best_code)   # revert to the last accepted
    return best_code, best_dev, best_tms, applied


def ask_of(model: str):
    return MODELS[model]


# ── the driver ───────────────────────────────────────────────────────────────
def run_job(job: PortJob, model: str = "gpt55", max_tries: int = 6,
            squeeze_rounds: int = 0, verbose: bool = True) -> dict:
    """Full generate -> parity-fix loop, then (optional) §9 optimization-squeeze. Returns a result dict."""
    ask = MODELS[model]
    module_path = PORTS / f"{job.name}_port.py"
    messages = build_gen_messages(job)
    code, dev, tms, ok, log, last_reply = "", None, None, False, "", ""

    def say(*a):
        if verbose:
            print(*a, flush=True)

    say(f"\n=== PORT JOB: {job.name}   model={model}   floor={job.float_floor:.1e}   max_tries={max_tries} ===")
    for attempt in range(1, max_tries + 1):
        say(f"\n[try {attempt}/{max_tries}] calling {model} ...")
        try:
            reply = ask(messages)
        except Exception as e:
            say(f"  API ERROR: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(3)
            continue
        last_reply = reply
        code = extract_code(reply)
        module_path.write_text(code)
        say(f"  generated {len(code)} chars -> {module_path.relative_to(BOOK)}")

        ok, dev, tms, log = run_parity(job, module_path)
        (PORTS / f"{job.name}_try{attempt}.log").write_text(log)
        if ok:
            tmsg = f", {tms:.3f} ms" if tms is not None else ""
            say(f"  PARITY PASS: max_rel_dev = {dev:.3e}  (< floor {job.float_floor:.1e}){tmsg}  in {attempt} API iteration(s)")
            break
        fail = f"parity max_rel_dev = {dev:.3e} (need < {job.float_floor:.1e})" if dev is not None else "run failed"
        say(f"  FAIL: {fail}")
        # feed the failure back; keep the conversation so the model sees its own prior code
        messages = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": build_fix_message(code, log[-6000:], job.float_floor)},
        ]
    else:
        say(f"\n!! {job.name}: did NOT reach parity in {max_tries} tries — surfacing for human review.")

    # §9 — the optimization-squeeze loop (parity-gated + timed), only after a clean parity pass
    squeeze_applied = []
    if ok and squeeze_rounds > 0:
        say(f"\n--- OPTIMIZATION SQUEEZE (bible §9): up to {squeeze_rounds} round(s) ---")
        code, dev, tms, squeeze_applied = optimization_squeeze(
            job, model, messages, last_reply, code, dev, tms, module_path, say, squeeze_rounds)

    lints = lint(code)
    say("\n--- VECTORIZATION LINT (review these by eye; a parity pass is not a vectorization proof) ---")
    if lints:
        for h in lints:
            say(h)
    else:
        say("  clean: no host-loop / host-pull patterns detected.")

    return dict(name=job.name, model=model, passed=ok, max_rel_dev=dev, time_ms=tms,
                api_iterations=attempt, module_path=str(module_path), lint=lints,
                squeeze_applied=squeeze_applied, log=log)


# ─────────────────────────────────────────────────────────────────────────────
#  JOB REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# Lecture 4 — the Voigt profile + single-line-centre opacity. The numpy twin lives in
# _pipeline/build_lecture4.py (the NumPy-edition notebook builder); we paste the two routines
# verbatim and check the torch port against reference/L4.npz and the numpy twin.

_L4_NUMPY = r'''
# Harris special-function tables (numerical math, not atomic data): h0tab,h1tab,h2tab (2001,)

def _voigt_wing(a, v):
    """Asymptotic (Lorentzian-wing) branch of Kurucz's Voigt approximation, valid far from centre."""
    aa, vv = a*a, v*v
    u = (aa + vv)*1.4142
    val = a*0.79788/u
    if a <= 100.0:
        aau, vvu, uu = aa/u, vv/u, u*u
        val = ((((aau - 10.0*vvu)*aau*3.0 + 15.0*vvu*vvu) + 3.0*vv - aa)/uu + 1.0)*val
    return val

def voigt_H(a, v):
    """Voigt H(a,v): Kurucz's Harris-table routine. a is a scalar; v is a 1-D array."""
    v = np.atleast_1d(np.asarray(v, float)); av = np.abs(v)
    iv = np.clip((av*200.0 + 0.5).astype(int), 0, h0tab.size-1)
    H0, H1, H2 = h0tab[iv], h1tab[iv], h2tab[iv]
    out = np.empty_like(v)
    if a < 0.2:                                    # weak damping: 2nd-order Harris series
        far = av > 10.0
        out[far]  = 0.5642*a/(v[far]*v[far])
        out[~far] = (H2[~far]*a + H1[~far])*a + H0[~far]
    elif a > 1.4:                                  # strong damping: pure asymptotic wing
        out[:] = _voigt_wing(a, v)
    else:                                          # intermediate: per-point split at a+|v|>3.2
        asy = (a + av) > 3.2
        out[asy] = _voigt_wing(a, v[asy])
        m = ~asy; vv = v[m]*v[m]; h0 = H0[m]; h1t = H1[m]; h2t = H2[m]
        h1 = h1t + h0*1.12838
        h2 = h2t + h1*1.12838 - h0
        h3 = (1.0 - h2t)*0.37613 - h1*0.66667*vv + h2*1.12838
        h4 = (3.0*h3 - h1)*0.37613 + h0*0.66667*vv*vv
        polyA = (((h4*a + h3)*a + h2)*a + h1)*a + h0
        polyB = ((-0.122727278*a + 0.532770573)*a - 0.96284325)*a + 0.979895032
        out[m] = polyA*polyB
    return out

# --- single Fe I line-centre opacity assembly at the photosphere layer jp ---
# constants: H_C,C,K (CGS Planck/c/Boltzmann), AMU; KEV = 1/11604.5 (eV per K)
# CLASSICAL = pi*(4.803204e-10)**2 / (9.1093837e-28 * C)
# line: lam0_nm=500.5, loggf=-1.0, chi_l=3.3, m_Fe=55.845*AMU, xi=2.0e5
# nu0 = C/(lam0_nm*1e-7)
# dnu_D = (nu0/C)*sqrt(2*K*T[jp]/m_Fe + xi**2)
# gamma = 2.0e8 + 1.0e-7*nHI[jp] + 1.0e-8*xne[jp]
# a_damp = gamma/(4*pi*dnu_D)
# lam = linspace(lam0_nm-0.04, lam0_nm+0.04, 800); nu = C/(lam*1e-7); v=(nu-nu0)/dnu_D
# phi = voigt_H(a_damp, v) / (sqrt(pi)*dnu_D)
# stim = 1 - exp(-H_C*nu/(K*T[jp]))
# nl_over_gl = (n_FeI[jp]/U_FeI[jp])*exp(-chi_l/(KEV*T[jp]))
# alpha_line = CLASSICAL * 10**loggf * nl_over_gl * phi * stim
# kappa_line = alpha_line / rho[jp]
'''

_L4_SPEC = """Port BOTH (a) the Voigt function H(a,v) [Kurucz's Harris-table routine, the three-regime
branch logic] and (b) the single Fe I line-centre opacity assembly, to fully-vectorized torch.

(a) `voigt_H_torch(a, v, h0tab, h1tab, h2tab)`: `a` is a 1-D tensor of damping values (shape [na]),
    `v` is a 1-D tensor of reduced frequencies (shape [nv]); the h-tables are 1-D tensors. Return a
    [na, nv] tensor H(a_i, v_j). The numpy twin's `voigt_H` takes a SCALAR a and a 1-D v; you must
    handle the WHOLE (a,v) GRID at once and BRANCHLESSLY: the three regimes (a<0.2 weak series,
    a>1.4 pure wing, else intermediate per-point split at a+|v|>3.2, plus the far av>10 fallback in
    the weak regime) must ALL be computed and combined with torch.where masks — no python branching
    on `a`, no loop over the na damping values. The table lookup
    `iv = clip((|v|*200 + 0.5).astype(int), 0, size-1)` becomes a clamped integer index tensor +
    torch.index_select. Match the numpy constants and thresholds bit-for-bit.

(b) `line_opacity(atm, line, voigt_tables)` -> a dict with key 'kappa_line' (a 1-D tensor over the
    800-point wavelength grid) and also 'a_damp','dnu_D' (scalars as 0-d tensors). Build it exactly
    as the commented assembly: Doppler width, the three broadening rates -> a_damp, the v-grid, the
    Voigt profile (reuse voigt_H_torch with a 1-element `a`), the stimulated-emission factor, the
    Boltzmann population, then alpha_line and kappa_line. The reference uses a SINGLE photosphere
    layer jp = argmin(|tau - 2/3|).

Provide a `DEVICE`/`DTYPE` picked once at module top and move all inputs onto it."""

_L4_CONTRACT = """Your module MUST define, importable as `port`:
  - port.DEVICE, port.DTYPE  (device + working dtype picked once)
  - port.voigt_H_torch(a, v, h0tab, h1tab, h2tab) -> tensor [len(a), len(v)]
        a,v,h0tab,h1tab,h2tab are array-likes; cast them onto DEVICE/DTYPE inside.
  - port.line_opacity(atm, line, voigt_tables) -> dict
        atm: dict with numpy arrays T,tk(unused),xne,rho,nHI,n_FeI,U_FeI,tau (each shape [80])
        line: dict with floats lam0_nm, loggf, chi_l, xi, m_Fe, gamma_rad, c_vdw, c_stark
              and constants H_C, C, K, KEV, CLASSICAL
        voigt_tables: dict with h0tab,h1tab,h2tab
        returns {'kappa_line': tensor[800], 'a_damp': 0-d tensor, 'dnu_D': 0-d tensor,
                 'jp': int}
All returned tensors may be on DEVICE; the harness moves them to CPU/fp64 itself."""

_L4_HARNESS = r'''
import numpy as np, torch, pathlib
REF = np.load(pathlib.Path("reference") / "L4.npz")

def to64(x):
    if torch.is_tensor(x):
        return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, float)

# --- (a) Voigt grid parity vs reference H(a,v) ---
H_torch = port.voigt_H_torch(REF["a"], REF["v"], REF["h0tab"], REF["h1tab"], REF["h2tab"])
H_torch = to64(H_torch)
H_ref = np.asarray(REF["H"], float)
denomH = np.where(H_ref != 0.0, np.abs(H_ref), 1.0)
dev_voigt = float(np.max(np.abs(H_torch - H_ref) / denomH))

# --- (b) single-line kappa parity vs an inline numpy twin (the L4 assembly) ---
C = 2.99792458e10; H_C = 6.62607015e-27; K = 1.380649e-16; AMU = 1.66053907e-24
KEV = 1.0/11604.5
CLASSICAL = np.pi*(4.803204e-10)**2 / (9.1093837e-28 * C)
T=REF["T"]; xne=REF["xne"]; rho=REF["rho"]; nHI=REF["nHI"]
n_FeI=REF["n_FeI"]; U_FeI=REF["U_FeI"]; tau=REF["tau"]; tk=REF["tk"]
jp = int(np.argmin(np.abs(tau - 2/3)))
lam0_nm=500.5; loggf=-1.0; chi_l=3.3; xi=2.0e5; m_Fe=55.845*AMU
gamma_rad=2.0e8; c_vdw=1.0e-7; c_stark=1.0e-8

# numpy twin (reuses the SCALAR-a numpy voigt for the reference kappa)
h0tab,h1tab,h2tab = REF["h0tab"],REF["h1tab"],REF["h2tab"]
def _wing(a,v):
    aa,vv=a*a,v*v; u=(aa+vv)*1.4142; val=a*0.79788/u
    if a<=100.0:
        aau,vvu,uu=aa/u,vv/u,u*u
        val=((((aau-10.0*vvu)*aau*3.0+15.0*vvu*vvu)+3.0*vv-aa)/uu+1.0)*val
    return val
def _voigtH(a,v):
    v=np.atleast_1d(np.asarray(v,float)); av=np.abs(v)
    iv=np.clip((av*200.0+0.5).astype(int),0,h0tab.size-1)
    H0,H1,H2=h0tab[iv],h1tab[iv],h2tab[iv]; out=np.empty_like(v)
    if a<0.2:
        far=av>10.0; out[far]=0.5642*a/(v[far]*v[far]); out[~far]=(H2[~far]*a+H1[~far])*a+H0[~far]
    elif a>1.4: out[:]=_wing(a,v)
    else:
        asy=(a+av)>3.2; out[asy]=_wing(a,v[asy])
        m=~asy; vv=v[m]*v[m]; h0=H0[m]; h1t=H1[m]; h2t=H2[m]
        h1=h1t+h0*1.12838; h2=h2t+h1*1.12838-h0
        h3=(1.0-h2t)*0.37613-h1*0.66667*vv+h2*1.12838; h4=(3.0*h3-h1)*0.37613+h0*0.66667*vv*vv
        polyA=(((h4*a+h3)*a+h2)*a+h1)*a+h0
        polyB=((-0.122727278*a+0.532770573)*a-0.96284325)*a+0.979895032
        out[m]=polyA*polyB
    return out
nu0=C/(lam0_nm*1e-7)
dnu_D=(nu0/C)*np.sqrt(2*K*T[jp]/m_Fe + xi**2)
gamma=gamma_rad + c_vdw*nHI[jp] + c_stark*xne[jp]
a_damp=gamma/(4*np.pi*dnu_D)
lam=np.linspace(lam0_nm-0.04,lam0_nm+0.04,800); nu=C/(lam*1e-7); vv=(nu-nu0)/dnu_D
phi=_voigtH(a_damp,vv)/(np.sqrt(np.pi)*dnu_D)
stim=1.0-np.exp(-H_C*nu/(K*T[jp]))
nl_over_gl=(n_FeI[jp]/U_FeI[jp])*np.exp(-chi_l/(KEV*T[jp]))
kappa_ref=(CLASSICAL*10**loggf*nl_over_gl*phi*stim)/rho[jp]

atm=dict(T=T,tk=tk,xne=xne,rho=rho,nHI=nHI,n_FeI=n_FeI,U_FeI=U_FeI,tau=tau)
line=dict(lam0_nm=lam0_nm,loggf=loggf,chi_l=chi_l,xi=xi,m_Fe=m_Fe,
          gamma_rad=gamma_rad,c_vdw=c_vdw,c_stark=c_stark,
          H_C=H_C,C=C,K=K,KEV=KEV,CLASSICAL=CLASSICAL)
res=port.line_opacity(atm, line, dict(h0tab=h0tab,h1tab=h1tab,h2tab=h2tab))
kappa_torch=to64(res["kappa_line"])
denomK=np.where(kappa_ref!=0.0,np.abs(kappa_ref),1.0)
dev_kappa=float(np.max(np.abs(kappa_torch-kappa_ref)/denomK))

max_dev=max(dev_voigt,dev_kappa)
print(f"voigt_grid_dev={dev_voigt:.3e}  kappa_dev={dev_kappa:.3e}  device={port.DEVICE}")
print(f"PARITY {max_dev:.6e}")

# --- §9 timing: the Voigt grid eval is the hot path; time it (warm, best-of-N) ---
import time as _t
a_big = np.repeat(REF["a"], 64); v_big = REF["v"]      # 384 x 481 grid, a realistic batch
for _ in range(3):                                     # warm up (MPS lazy alloc / compile)
    Hb = port.voigt_H_torch(a_big, v_big, REF["h0tab"], REF["h1tab"], REF["h2tab"])
    if torch.is_tensor(Hb) and Hb.device.type == "mps": torch.mps.synchronize()
best = 1e9
for _ in range(20):
    t0 = _t.perf_counter()
    Hb = port.voigt_H_torch(a_big, v_big, REF["h0tab"], REF["h1tab"], REF["h2tab"])
    if torch.is_tensor(Hb) and Hb.device.type == "mps": torch.mps.synchronize()
    best = min(best, _t.perf_counter() - t0)
print(f"TIME {best*1e3:.6f}")
'''

# ═════════════════════════════════════════════════════════════════════════════
# Lecture 5 — Line Opacity II: the line list (the SCATTER-ADD hot path).
# The numpy twin (build_lecture5.py) accumulates EVERY atomic metal line's Voigt
# wing onto a [80, 5941] log-lambda grid by walking outward from each line centre
# and += depositing (the per-line, per-depth, per-offset host loop). The GPU port
# REDUCES kgpu/line_opacity.py: evaluate Harris H(a,v) branchlessly on the whole
# (a,v) grid, and replace the thousands of tiny per-line += walks with ONE batched
# [depth, line, offset] scatter (`index_put_(accumulate=True)`). The bible flags
# this accumulation as the PRIME Metal-kernel candidate (§2.5), so the harness times
# the scatter so the §9 squeeze can evaluate a custom `torch.mps.compile_shader`
# kernel vs batched torch.
# ─────────────────────────────────────────────────────────────────────────────
_L5_NUMPY = r'''
# --- the metal-line scatter accumulation (build_lecture5.py, reduced to the core kernel) ---
# Harris H(a,v) (the L4 kernel): three branches by damping a, table iv = clip(int(|v|*200+0.5),0,N-1)
def voigt_profile(v, a, h0tab, h1tab, h2tab):
    iv = max(0, min(int(abs(v)*200.0+0.5), h0tab.size-1))
    if a < 0.2:
        if abs(v) > 10.0: return 0.5642*a/(v*v)
        return (h2tab[iv]*a + h1tab[iv])*a + h0tab[iv]
    elif a > 1.4 or (a+abs(v)) > 3.2:
        aa=a*a; vv=v*v; u=(aa+vv)*1.4142; val=a*0.79788/u
        if a <= 100.0:
            aau=aa/u; vvu=vv/u; uu=u*u
            val=((((aau-10.0*vvu)*aau*3.0+15.0*vvu*vvu)+3.0*vv-aa)/uu+1.0)*val
        return val
    else:
        vv=v*v; h0=h0tab[iv]; h1=h1tab[iv]+h0*1.12838; h2=h2tab[iv]+h1*1.12838-h0
        h3=(1.0-h2tab[iv])*0.37613-h1*0.66667*vv+h2*1.12838
        h4=(3.0*h3-h1)*0.37613+h0*0.66667*vv*vv
        pa=(((h4*a+h3)*a+h2)*a+h1)*a+h0
        pb=((-0.122727278*a+0.532770573)*a-0.96284325)*a+0.979895032
        return pa*pb

def voigt_h_at_zero(adamp, h0tab, h1tab, h2tab):    # vectorized H(a,0), floored at 1e-30
    h0_0,h1_0,h2_0=float(h0tab[0]),float(h1tab[0]),float(h2tab[0])
    h0v=h0_0; h1v=h1_0+h0v*1.12838; h2v=h2_0+h1v*1.12838-h0v
    h3v=(1.0-h2_0)*0.37613+h2v*1.12838; h4v=(3.0*h3v-h1v)*0.37613
    a=np.asarray(adamp,float)
    h_low=(h2_0*a+h1_0)*a+h0_0
    h_mid=((((h4v*a+h3v)*a+h2v)*a+h1v)*a+h0v)*(((-0.122727278*a+0.532770573)*a-0.96284325)*a+0.979895032)
    aa=a*a; u=np.maximum(aa*1.4142,1e-40); base=a*0.79788/u; aau=aa/u
    h_high=np.where(a<=100.0,((aau*aau*3.0-aa)/np.maximum(u*u,1e-40)+1.0)*base,base)
    return np.maximum(np.where(a<0.2,h_low,np.where((a>1.4)|(a>3.2),h_high,h_mid)),1e-30)

# FASTEX e^{-x}: the production-table Boltzmann factor (must match bit-for-bit)
def fast_ex(x):
    v=np.asarray(x,float); out=np.empty_like(v); out[v==0.0]=1.0
    neg=v<0.0; out[neg]=np.exp(-v[neg]); pos=v>0.0
    if np.any(pos):
        p=v[pos]; i=np.floor(p).astype(np.int64); tab=i<_EXTAB.size; po=np.empty_like(p)
        if np.any(tab):
            it=i[tab]; j=np.clip(np.floor((p[tab]-it)*1000.0+0.5).astype(np.int64),0,_EXTABF.size-1)
            po[tab]=_EXTAB[it]*_EXTABF[j]
        if np.any(~tab): po[~tab]=np.exp(-p[~tab])
        out[pos]=po
    return out

# grid index helpers (log grid): center pixel + wing anchor (a sub-pixel apart, both reproduced)
def nearest_grid_indices(grid, values):
    ratiolg=np.log(grid[1]/grid[0]); ix0=int(np.log(grid[0])/ratiolg+0.5)
    idx=(np.log(values)/ratiolg+0.5).astype(np.int64)-ix0
    idx[values<grid[0]]=-1; idx[values>grid[-1]]=grid.size; return idx
def nearest_grid_indices_raw(grid, values, origin_start):
    ratiolg=np.log(grid[1]/grid[0]); ixf=int(np.floor(np.log(origin_start)/ratiolg))
    wb=np.exp(ixf*ratiolg)
    if wb<origin_start: ixf+=1; wb=np.exp(ixf*ratiolg)
    return np.rint(np.log(values/wb)/ratiolg).astype(np.int64)

CUTOFF=1e-3; KAPMIN_FLOOR=1e-8; MAX_PROFILE_STEPS=1_000_000
CGF_CONSTANT=0.026538/1.77245; C_LIGHT_NM=2.99792458e17

def process_wing_pair(asynth_d, grid, center_idx, kappa0, adamp, doppler_width,
                      line_wavelength, kapmin_ref, resolu, h0tab, h1tab, h2tab):
    """One (line, depth) wing walk: stages 1-3 (near wing, far-wing reach, red+blue deposit)."""
    n_w=grid.size
    if doppler_width<=0.0: return
    dopple=doppler_width/line_wavelength if line_wavelength>0.0 else 1e-10
    n10dop=int(10.0*dopple*resolu); dvoigt=1.0/(dopple*resolu) if dopple>0.0 else 1.0
    nstep_cutoff=n10dop; profile_at_n10dop=0.0; tabstep=200.0*dvoigt; tabi=0.5; broke=False
    for nstep in range(1, n10dop+1):
        if adamp<0.2:
            tabi+=tabstep; idx=max(int(tabi),0); x=nstep*dvoigt
            if x>10.0: pv=kappa0*(0.5642*adamp/(x*x))
            else: pv=kappa0*(h0tab[min(idx,h0tab.size-1)]+adamp*h1tab[min(idx,h1tab.size-1)])
        else: pv=kappa0*voigt_profile(nstep*dvoigt, adamp, h0tab, h1tab, h2tab)
        if nstep==n10dop: profile_at_n10dop=pv
        if pv<kapmin_ref: nstep_cutoff=nstep; broke=True; break
    if not broke and n10dop>=1: nstep_cutoff=-1
    if nstep_cutoff!=-1: maxstep=nstep_cutoff; use_far=False; x_far=0.0
    else:
        use_far=True
        if n10dop>0 and profile_at_n10dop>0.0:
            x_far=profile_at_n10dop*float(n10dop)**2
            maxstep=int(np.sqrt(x_far/kapmin_ref)+1.0) if kapmin_ref>0.0 else MAX_PROFILE_STEPS
        else: x_far=0.0; maxstep=0
        maxstep=min(maxstep, MAX_PROFILE_STEPS)
    red=blue=True; offset=1; tabi=0.5
    while offset<=maxstep and (red or blue):
        if use_far and offset>n10dop: pv=x_far/float(offset)**2
        elif adamp<0.2:
            tabi+=tabstep; idx=max(int(tabi),0); x=offset*dvoigt
            if x>10.0: pv=kappa0*(0.5642*adamp/(x*x))
            else: pv=kappa0*(h0tab[min(idx,h0tab.size-1)]+adamp*h1tab[min(idx,h1tab.size-1)])
        else: pv=kappa0*voigt_profile(offset*dvoigt, adamp, h0tab, h1tab, h2tab)
        if pv==0.0: break
        if red:
            j=center_idx+offset
            if j>=n_w: red=False
            elif j>=0: asynth_d[j]+=pv
        if blue:
            j=center_idx-offset
            if j<0: blue=False
            elif j<n_w: asynth_d[j]+=pv
        offset+=1

def metal_accumulate_numpy(catalog, atm, grid, cont):
    """The numpy-twin metal-line accumulation -> metal_opacity[n_depths, n_w] (NO stim factor;
    applied once at the end). The scalar reference the torch scatter must reproduce."""
    lam=catalog['lam']; gf_lin=catalog['gf']; Elow=catalog['Elow']; idxwl=catalog['idxwl']
    Zc=catalog['Z']; ion=catalog['ion']; lt=catalog['lt']
    grad=catalog['grad']; gstark=catalog['gstark']; gvdw=catalog['gvdw']
    h0tab,h1tab,h2tab=catalog['h0tab'],catalog['h1tab'],catalog['h2tab']
    pop3=atm['pop3']; dop3=atm['dop3']; rho=atm['rho']; xne=atm['xne']; T=atm['T']
    hckt=atm['hckt']; txnxn=atm['txnxn']
    n_depths=T.size; n_w=grid.size
    freq_hz=C_LIGHT_NM/lam; cgf=CGF_CONSTANT*gf_lin/freq_hz
    elem_idx=Zc-1; ion_idx=ion-1
    n_ion_max,n_elem_max=pop3.shape[1],pop3.shape[2]
    center_idx=nearest_grid_indices(grid, idxwl)
    wing_idx=nearest_grid_indices_raw(grid, idxwl, float(grid[0]))
    ratio=grid[1]/grid[0]; resolu=1.0/(ratio-1.0) if ratio>1.0 else 300000.0
    line_ok=(lt==0)&(elem_idx>=0)&(elem_idx<n_elem_max)&(ion_idx>=0)&(ion_idx<n_ion_max)
    boltz=fast_ex(Elow[None,:]*hckt[:,None])
    M=MAX_PROFILE_STEPS
    center_valid=line_ok&(center_idx>=0)&(center_idx<n_w)
    wing_active=line_ok&(wing_idx>=-M)&(wing_idx<=n_w-1+M)
    metal_opacity=np.zeros((n_depths,n_w),dtype=np.float64)
    for i in np.where(line_ok)[0]:
        ci=int(center_idx[i]); wi=int(wing_idx[i]); wl_i=lam[i]; clamped=max(0,min(ci,n_w-1))
        pop=pop3[:,ion_idx[i],elem_idx[i]]; dop=dop3[:,ion_idx[i],elem_idx[i]]
        kapmin=cont[:,clamped]*CUTOFF; good=(pop>0.0)&(dop>0.0)&(rho>0.0)
        if not np.any(good): continue
        xnfdop=np.zeros(n_depths); xnfdop[good]=pop[good]/(rho[good]*dop[good])
        kappa0_pre=cgf[i]*xnfdop; post=kappa0_pre*boltz[:,i]
        passcut=good&(kappa0_pre>=kapmin)&(post>=kapmin)&(post>0.0)
        if not np.any(passcut): continue
        doppler_width=dop*wl_i; dopple=np.where(wl_i>0,doppler_width/wl_i,1e-6)
        gamma_total=grad[i]+gstark[i]*xne+gvdw[i]*txnxn
        adamp=np.where((doppler_width>0)&(dopple>0),gamma_total/dopple,0.0)
        kapcen=np.zeros(n_depths); cd=passcut&(adamp>=0.0)&(post>0.0)
        for d in np.where(cd)[0]:
            ad=adamp[d]
            kapcen[d]=post[d]*(1.0-1.128*ad) if ad<0.2 else post[d]*voigt_profile(0.0,ad,h0tab,h1tab,h2tab)
        if center_valid[i]:
            for d in np.where(cd)[0]: metal_opacity[d,ci]+=kapcen[d]
        if not wing_active[i]: continue
        wing_pairs=cd&(kapcen>0.0)
        if not np.any(wing_pairs): continue
        adamp_w=np.maximum(adamp,1e-12)
        kappa0_wing=np.where(kapcen>0.0,kapcen/voigt_h_at_zero(adamp_w,h0tab,h1tab,h2tab),0.0)
        ci_w=min(max(wi,0),n_w-1); kapmin_ref=np.maximum(cont[:,ci_w]*CUTOFF,cont[:,ci_w]*KAPMIN_FLOOR)
        for d in np.where(wing_pairs)[0]:
            process_wing_pair(metal_opacity[d],grid,wi,kappa0_wing[d],adamp_w[d],
                              doppler_width[d],wl_i,kapmin_ref[d],resolu,h0tab,h1tab,h2tab)
    return metal_opacity
'''

_L5_SPEC = """Port the LINE-LIST line-opacity accumulation — the SCATTER-ADD hot path — to fully
vectorized torch/MPS. Reproduce the numpy twin's metal-line accumulation onto the [n_depths, n_w]
log-lambda grid: every atomic metal line (type 0) contributes its Voigt wings, walked outward from
its centre pixel until the profile drops below 1e-3 x the local continuum, += deposited red and blue.

Provide TWO entry points (the contract spells out signatures):

(a) `voigt_H_grid(v, a, h0tab, h1tab, h2tab)` -> H(a,v) on the WHOLE (v,a) broadcast grid, branchless
    (the L4 Harris kernel, REDUCED from kgpu's `harris_hav`): three regimes (a<0.2 table series with the
    |v|>10 Lorentzian tail; a>1.4 OR a+|v|>3.2 far-wing asymptotic with the a<=100 correction; else the
    intermediate polynomial blend) ALL computed and selected with torch.where. Table lookup
    iv = clamp(int(|v|*200+0.5),0,N-1) is a clamped index_select. No python loop over a or v.

(b) `accumulate_metal(catalog, atm, grid, cont)` -> kappa_metal[n_depths, n_w] (a torch tensor on
    DEVICE; NO stimulated-emission factor — applied once at the very end, outside this routine). This is
    the hot path: REDUCE kgpu/line_opacity.py's BATCHED accumulation (`_wing_reach_batched` for the
    per-(depth,line) reach geometry, `_wing_walk_narrow_core` for the ONE batched [depth,line,offset]
    deposit, `_scatter_add_3d` = `index_put_(accumulate=True)`). Forbidden: a python `for` over the
    ~12k lines doing a per-line walk. The line axis MUST be a tensor batch axis; the deposit MUST be a
    single (or a few reach-tiered) batched scatter, NOT thousands of tiny launches.

    The accumulation, reproduced bit-for-bit from the numpy twin:
      - cgf_i = (0.026538/sqrt(pi)) * gf_i / (C_LIGHT_NM/lam_i)
      - center pixel = nearest_grid_indices(grid, idxwl); wing anchor = nearest_grid_indices_raw(...)
        (a log-grid round; the two anchors are a sub-pixel apart — reproduce BOTH)
      - resolu = 1/(grid[1]/grid[0] - 1)
      - per (depth, line): xnfdop = pop/(rho*dop); kappa0_pre = cgf*xnfdop; post = kappa0_pre*FASTEX_boltz;
        two-stage cutoff (kappa0_pre >= 1e-3*cont AND post >= 1e-3*cont, both at the centre pixel)
      - adamp = (grad + gstark*xne + gvdw*txnxn) / (doppler_width/lam)
      - center deposit kapcen = post*(1-1.128*a) for a<0.2 else post*H(a,0); deposited at the centre pixel
      - wing amplitude kappa0_wing = kapcen / H(a,0) (floor a at 1e-12); walk the near wing (steps up to
        10 Doppler widths) evaluating the CHEAP a<0.2 2-term form (h0+a*h1, with 0.5642*a/x^2 for x>10)
        or full H(a,v); set the far-wing reach analytically (x_far = profile_at_n10dop * n10dop^2;
        reach = sqrt(x_far/kapmin_ref)+1); deposit kappa0_wing*H or x_far/offset^2 at center±offset
      - the wing cutoff kapmin_ref = max(cont*1e-3, cont*1e-8) at the (clamped) wing anchor pixel
    Use FASTEX (the production tabulated e^{-x}, EXTAB[floor]*EXTABF[round(1000*frac)]) for the Boltzmann
    factor — NOT torch.exp — to match the numpy twin bit-for-bit. Match every constant and threshold.

DEVICE/DTYPE picked once at module top; cast inputs onto it. fp32 on MPS; the float floor is ~1e-6."""

_L5_CONTRACT = """Your module MUST define, importable as `port`:
  - port.DEVICE, port.DTYPE  (device + working dtype picked once)
  - port.voigt_H_grid(v, a, h0tab, h1tab, h2tab) -> tensor broadcast(v,a)
        v,a,h*tab are array-likes; cast onto DEVICE/DTYPE inside. v and a broadcast (e.g. v[nv], a[na,1]).
  - port.accumulate_metal(catalog, atm, grid, cont) -> tensor [n_depths, n_w]  (on DEVICE; NO stim factor)
        catalog: dict of numpy arrays lam[L],gf[L],Elow[L],idxwl[L],Z[L],ion[L],lt[L],
                 grad[L],gstark[L],gvdw[L]  (1-based Z and ion; lt is the line-type code, metals are lt==0)
                 and h0tab[N],h1tab[N],h2tab[N]
        atm: dict of numpy arrays pop3[D,n_ion,n_elem], dop3[D,n_ion,n_elem], rho[D], xne[D], T[D],
             hckt[D] (= h*c/kT per depth), txnxn[D] (the effective vdW perturber density)
        grid: numpy array [n_w] (the log-lambda grid, nm); cont: numpy array [D, n_w] (the cutoff continuum)
        returns kappa_metal[D, n_w] (the accumulated metal-line opacity BEFORE stimulated emission)
All returned tensors may be on DEVICE; the harness moves them to CPU/fp64 itself.
You MAY define helper functions (FASTEX, the grid-index helpers, the batched wing reach + scatter)."""

_L5_HARNESS = r'''
import numpy as np, torch, pathlib
REFDIR = pathlib.Path("reference")
cat = np.load(REFDIR/"full_lines_data.npz", allow_pickle=True)
atmz = np.load(REFDIR/"atmosphere.npz")
diag = np.load(REFDIR/"diag.npz")

def to64(x):
    if torch.is_tensor(x): return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, float)

_EXTAB  = np.exp(-np.arange(1001, dtype=np.float64))
_EXTABF = np.exp(-np.arange(1001, dtype=np.float64)*0.001)
''' + _L5_NUMPY + r'''

# --- assemble the inputs (same arrays the numpy twin reads) ---
grid = diag["wavelength"]
cont = diag["continuum_absorption"] + diag["continuum_scattering"]
T = atmz["temperature"]
txnxn = (atmz["xnf_h"] + 0.42*atmz["xnf_he1"] + 0.85*atmz["xnf_h2"]) * (T/1e4)**0.3
catalog = dict(lam=cat["cat_wl"], gf=cat["cat_gf"], Elow=cat["cat_elow"],
               idxwl=cat["cat_index_wl"], Z=cat["cat_Z"].astype(np.int64),
               ion=cat["cat_ion"].astype(np.int64), lt=cat["cat_line_types"].astype(np.int64),
               grad=cat["cat_grad"], gstark=cat["cat_gstark"], gvdw=cat["cat_gvdw"],
               h0tab=cat["h0tab"], h1tab=cat["h1tab"], h2tab=cat["h2tab"])
atm = dict(pop3=atmz["population_per_ion"], dop3=atmz["doppler_per_ion"],
           rho=atmz["mass_density"], xne=atmz["electron_density"], T=T,
           hckt=atmz["hckt"], txnxn=txnxn)

# --- (a) Voigt grid parity vs the numpy Harris kernel ---
v_grid = np.linspace(-12.0, 12.0, 481); a_grid = np.array([0.01,0.1,0.5,1.0,2.0,8.0])
H_ref = np.array([[voigt_profile(vv, aa, catalog["h0tab"], catalog["h1tab"], catalog["h2tab"])
                   for vv in v_grid] for aa in a_grid])
H_torch = to64(port.voigt_H_grid(v_grid, a_grid.reshape(-1,1), catalog["h0tab"],
                                 catalog["h1tab"], catalog["h2tab"]))
denomH = np.where(H_ref!=0.0, np.abs(H_ref), 1.0)
dev_voigt = float(np.max(np.abs(H_torch - H_ref)/denomH))

# --- (b) the full metal-line scatter accumulation vs the numpy twin ---
kappa_ref = metal_accumulate_numpy(catalog, atm, grid, cont)     # [80, 5941], no stim
kappa_torch = to64(port.accumulate_metal(catalog, atm, grid, cont))
big = np.abs(kappa_ref) > 1e-12     # the opacity-bearing pixels (the line cores+wings)
if big.sum() == 0:
    dev_metal = 1e9
else:
    dev_metal = float(np.max(np.abs(kappa_torch[big] - kappa_ref[big]) / np.abs(kappa_ref[big])))

max_dev = max(dev_voigt, dev_metal)
print(f"voigt_grid_dev={dev_voigt:.3e}  metal_scatter_dev={dev_metal:.3e}  "
      f"(nonzero px={int(big.sum())})  device={port.DEVICE}")
print(f"PARITY {max_dev:.6e}")

# --- §9 timing: the scatter accumulation is the hot path; time it (warm, best-of-N) ---
import time as _t
for _ in range(2):
    Kb = port.accumulate_metal(catalog, atm, grid, cont)
    if torch.is_tensor(Kb) and Kb.device.type=="mps": torch.mps.synchronize()
best = 1e9
for _ in range(6):
    t0=_t.perf_counter()
    Kb = port.accumulate_metal(catalog, atm, grid, cont)
    if torch.is_tensor(Kb) and Kb.device.type=="mps": torch.mps.synchronize()
    best=min(best, _t.perf_counter()-t0)
print(f"TIME {best*1e3:.6f}")
'''


# ═════════════════════════════════════════════════════════════════════════════
# Lecture 6 — Hydrogen Lines: Stark broadening (the HPROF4 profile).
# The numpy twin (build_lecture6.py) evaluates the HPROF4 hydrogen line profile —
# the quasi-static Holtsmark `sofbeta` + the electron-impact width + a Doppler core
# over fine-structure components + the non-Stark Lorentzian — SCALAR, point by point.
# The GPU port REDUCES kgpu/hydrogen.py: `_sofbeta` + `_hydrogen_profile_grid` evaluate
# the WHOLE [depth, wavelength] grid branchlessly (the three beta-regimes and the
# core/wing width-selection folded into torch.where). Note the cancellation-free
# del_freq factoring (the catastrophic-cancellation fix the GPU edition teaches).
# ─────────────────────────────────────────────────────────────────────────────
_L6_NUMPY = r'''
# --- the HPROF4 hydrogen profile (build_lecture6.py, scalar reference) ---
RYDH=3.2880515e15; C_LIGHT_AA=2.99792458e18
def _fast_ex(x): return 0.0 if x>80.0 else math.exp(-x)
def _vcse1f(x):
    if x<=0.0: return 0.0
    if x<=0.01: return -math.log(x)-0.577215+x
    if x<=1.0:
        return (-math.log(x)-0.57721566+x*(0.99999193+x*(-0.24991055+x*(0.05519968
                +x*(-0.00976004+x*0.00107857)))))
    if x>30.0: return 0.0
    num=x*(x+2.334733)+0.25062; den=(x*(x+3.330657)+1.681534)*x
    return num/den*math.exp(-x)
def _hf_nm(n,m):
    if m<=n: return 0.0
    xn,xm=float(n),float(m); ginf=0.2027/xn**0.71; gca=0.124/xn
    fkn=xn*1.9603; wtc=0.45-2.4/xn**3*(xn-1.0); xmn=xm-xn
    fk=fkn*(xm/(xmn*(xm+xn)))**3; xmn12=xmn**1.2; wt=(xmn12-1.0)/(xmn12+wtc)
    return fk*(1.0-wt*ginf-(0.222+gca/xm)*(1.0-wt))
def sofbeta(beta, p, n, m, propbm, c_arr, d_arr, pp_arr, beta_arr):
    if beta<=0.0: return 0.0
    b2=beta*beta; sb=math.sqrt(beta); corr=1.0
    if beta<=500.0:
        mmn=m-n; indx=2*(n-1)+mmn if (n<=3 and mmn<=2) else 7; indx=min(max(indx,1),7)
        im=min(int(5.0*p)+1,4); im=max(im,1); ip=im+1
        wtp=min(max(5.0*(p-pp_arr[im-1]),0.0),1.0); wtm=1.0-wtp
        if beta<=25.12:
            j=int(np.searchsorted(beta_arr,beta)); j=min(max(j,1),beta_arr.shape[0]-1)
            jm,jp=j-1,j; denom=beta_arr[jp]-beta_arr[jm]
            wtb=0.0 if denom<=0.0 else (beta-beta_arr[jm])/denom; wtbm=1.0-wtb
            cbp=propbm[indx-1,ip-1,jp]*wtp+propbm[indx-1,im-1,jp]*wtm
            cbm=propbm[indx-1,ip-1,jm]*wtp+propbm[indx-1,im-1,jm]*wtm
            corr=1.0+cbp*wtb+cbm*wtbm
            wt=min(max(0.5*(10.0-beta),0.0),1.0)
            pr1=8.0/(83.0+(2.0+0.95*b2)*beta) if beta<=10.0 else 0.0
            pr2=(1.5/sb+27.0/b2)/b2 if beta>=8.0 else 0.0
            return (pr1*wt+pr2*(1.0-wt))*corr
        cc=c_arr[im-1,indx-1]*wtp+c_arr[ip-1,indx-1]*wtm
        dd=d_arr[im-1,indx-1]*wtp+d_arr[ip-1,indx-1]*wtm
        denom2=cc+beta*sb
        if denom2==0.0: denom2=1e-30
        corr=1.0+dd/denom2
    return (1.5/sb+27.0/b2)/b2*corr
def hydrogen_line_profile(n, m, delta_lambda_nm, hyd, tabs, foff, fwt, n_fine):
    """HPROF4 profile phi(Delta-lambda) for transition n->m (scalar reference)."""
    t3nhe,t3nh2,fo=hyd["t3nhe"],hyd["t3nh2"],hyd["fo"]; dopph=hyd["dopph"]
    c1d,c2d,y1s,y1b=hyd["c1d"],hyd["c2d"],hyd["y1s"],hyd["y1b"]
    gcon1,gcon2,pp_val=hyd["gcon1"],hyd["gcon2"],hyd["pp"]
    xnfph_0,ne=hyd["xnfph_0"],hyd["ne"]
    asum=tabs["asum"]; y1wtm=tabs["y1wtm"]; xknmtb=tabs["xknmtb"]
    propbm=tabs["propbm"]; c_t=tabs["c"]; d_t=tabs["d"]; pp_t=tabs["pp"]; beta_t=tabs["beta"]
    mmn=m-n
    if mmn<=0: return 0.0
    xn,xm=float(n),float(m); xn2,xm2=xn*xn,xm*xm
    xm2mn2=xm2-xn2; xmn2=xm2*xn2; gnm=xm2mn2/xmn2
    xknm=xknmtb[n-1,mmn-1] if (n<=4 and mmn<=3) else 5.5e-5/gnm*xmn2/(1.0+0.13/mmn)
    freqnm=RYDH*gnm; wavenm=C_LIGHT_AA/freqnm; dbeta=C_LIGHT_AA/(freqnm*freqnm*xknm)
    c1con=xknm/wavenm*gnm*xm2mn2; c2con=(xknm/wavenm)**2
    n_a=asum.shape[0]
    radamp=(asum[n-1]+asum[m-1]) if (n<=n_a and m<=n_a) else (asum[n-1] if n<=n_a else 0.0)
    radamp=radamp/12.5664/freqnm
    resont=_hf_nm(1,m)/xm/(1.0-1.0/xm2)
    if n!=1: resont+=_hf_nm(1,n)/xn/(1.0-1.0/xn2)
    resont*=3.579e-24/gnm
    vdw=4.45e-26/gnm*(xm2*(7.0*xm2+5.0))**0.4
    hwvdw=vdw*t3nhe+2.0*vdw*t3nh2; hwrad=radamp; stark=1.6678e-18*freqnm*xknm
    hwres=resont*xnfph_0*2.0; hwstk=stark*fo; hwlor=hwres+hwvdw+hwrad
    wlA=wavenm+delta_lambda_nm*10.0
    if wlA<=0.0: return 0.0
    freq=C_LIGHT_AA/wlA; del_freq=abs(freq-freqnm)
    dop=freqnm*max(dopph,1e-40); hfwidth=freqnm*max(max(dopph,1e-40),hwlor,hwstk)
    ifcore=del_freq<=hfwidth; nwid=1
    if not (dopph>=hwstk and dopph>=hwlor):
        nwid=2
        if hwlor<hwstk: nwid=3
    core=0.0
    for fi in range(n_fine):
        dd=abs(freq-(freqnm+foff[fi]))/max(dop,1e-30)
        if dd<=7.0: core+=_fast_ex(dd*dd)*fwt[fi]
    hhw=freqnm*hwlor
    lorentz=(hhw/math.pi/(del_freq*del_freq+hhw*hhw)*1.77245*dop) if hhw>0.0 else 0.0
    y1num=320.0 if m>3 else (550.0 if m==2 else 380.0)
    y1wht=1.0e14 if mmn<=3 else 1.0e13
    if mmn<=2 and 1<=n<=2 and n<=y1wtm.shape[0] and mmn<=y1wtm.shape[1]: y1wht=y1wtm[n-1,mmn-1]
    wty1=1.0/(1.0+max(ne,0.0)/max(y1wht,1e-30)); y1_scal=y1num*y1s*wty1+y1b*(1.0-wty1)
    c1=c1d*c1con*y1_scal; c2=c2d*c2con
    beta=del_freq/max(fo,1e-30)*dbeta; y1=c1*beta; y2=c2*beta*beta
    g1=6.77*math.sqrt(max(c1,1e-30))
    ratio=math.sqrt(c2)/max(c1,1e-30) if (c1>0.0 and c2>0.0) else 0.0
    log_term=math.log(max(ratio,1e-30)) if ratio>0.0 else 0.0
    gamma=g1*max(0.0,0.2114+log_term)*(1.0-gcon1-gcon2)
    if y2>1e-4 and y1>1e-5:
        gamma=(g1*(0.5*_fast_ex(min(80.0,y1))+_vcse1f(y1)-0.5*_vcse1f(y2))
               *(1.0-gcon1/(1.0+(90.0*y1)**3)-gcon2/(1.0+2000.0*y1)))
    f=gamma/math.pi/(gamma*gamma+beta*beta) if gamma>0.0 else 0.0
    prqs=sofbeta(beta,pp_val,n,m,propbm,c_t,d_t,pp_t,beta_t)
    p1=(0.9*y1)**2; fns=(p1+0.03*math.sqrt(max(y1,0.0)))/(p1+1.0)
    stark_core=(prqs*(1.0+fns)+f)/max(fo,1e-30)*dbeta*1.77245*dop
    if ifcore:
        if nwid==1: return max(core,0.0)
        if nwid==2: return max(lorentz,0.0)
        return max(stark_core,0.0)
    return max(core+lorentz+stark_core,0.0)

def hydrogen_state_numpy(di, T, xne, xnf_he1, xnf_h2, xnfph, vturb_cms):
    KBOLTZ=1.380649e-16; AMU=1.66054e-24; C_CMS=2.99792458e10; C_KMS=299792.458; MASS_H=1.008
    temp=max(float(T[di]),1.0); ne_d=float(xne[di]); x16=ne_d**(1.0/6.0)
    fo_d=x16**4*1.25e-9; pp=x16*0.08989/math.sqrt(temp)
    y1b=2.0/(1.0+0.012/temp*math.sqrt(ne_d/temp)); t43=(temp/1.0e4)**0.3
    y1s=t43/x16; c1d=fo_d*78940.0/temp; c2d=fo_d**2/5.96e-23/ne_d
    gcon1=0.2+0.09*math.sqrt(max(temp/1e4,1e-12))/(1.0+ne_d/1.0e13); gcon2=0.2/(1.0+ne_d/1.0e15)
    vth=math.sqrt(2.0*KBOLTZ*temp/(MASS_H*AMU))/C_CMS; vtb=(float(vturb_cms[di])/1e5)/C_KMS
    dopph=math.sqrt(vth*vth+vtb*vtb)
    return dict(t3nhe=t43*float(xnf_he1[di]), t3nh2=t43*float(xnf_h2[di]),
                fo=fo_d, dopph=dopph, c1d=c1d, c2d=c2d, y1s=y1s, y1b=y1b,
                gcon1=gcon1, gcon2=gcon2, pp=pp, ne=ne_d, xnfph_0=float(xnfph[di,0]))
'''

_L6_SPEC = """Port the HYDROGEN STARK HPROF4 line profile to fully vectorized torch/MPS. The numpy
twin evaluates the profile phi(Delta-lambda) for a Balmer transition n->m SCALAR, point by point; the
GPU port evaluates the WHOLE [n_depths, n_w] grid (all depths x all wavelength offsets) branchlessly.

Provide TWO entry points (the contract spells out signatures):

(a) `sofbeta_grid(beta, p, n, m, propbm, c_arr, d_arr, pp_arr, beta_arr)` -> S(beta) on a [D, n_w]
    tensor `beta` with [D,1] `p`, REDUCED from kgpu/hydrogen.py `_sofbeta`. The three beta-regimes
    (beta<=25.12 bilinear-interp correction + near/asymptotic blend; 25.12<beta<=500 asymptotic *
    (c,d) correction; beta>500 bare Holtsmark tail) ALL computed and torch.where-selected. The (p,beta)
    bilinear table interpolation becomes flat-gather index math (bracket p in pp_arr, beta in beta_arr
    via searchsorted), NOT a python loop. Returns 0 where beta<=0.

(b) `hydrogen_profile_grid(n, m, delta_lambda_nm, hyd, tabs, foff, fwt, n_fine)` -> phi[D, n_w],
    REDUCED from kgpu/hydrogen.py `_hydrogen_profile_grid`. delta_lambda_nm is [D, n_w] (the wavelength
    offset of every grid pixel from line centre, per depth); `hyd` carries per-depth state as [D,1]
    tensors. The three profile pieces — (1) Doppler core summed over the n_fine fine-structure Gaussians,
    (2) the non-Stark Lorentzian, (3) the linear-Stark term (sofbeta_grid * (1+fns) + the electron-impact
    Lorentzian whose width gamma uses the vectorized E_1(x)) — ALL computed; then SELECTED by the dominant
    width: inside the dominant half-width return only the dominant piece (Doppler / Lorentz / Stark per
    the nwid rule), in the wing return the SUM — all folded into torch.where, NO python branch on depth.
    A short for-loop over the FIXED small n_fine fine-structure components is acceptable ONLY if each
    iteration is a fully vectorized [D,n_w] tensor op (kgpu does exactly this).

PRECISION TRAP (teach it): del_freq = |freq - freqnm| with freq = C_LIGHT_AA/wlA is a difference of two
~6e14 Hz numbers — catastrophic fp32 cancellation in the line core. Do NOT subtract them directly; factor
the wavelength difference out algebraically (freq - freqnm = -C_LIGHT_AA * (delta_lambda_nm*10) /
(wlA*wavenm), exact, no cancellation), exactly as kgpu's `_hydrogen_profile_grid` does. Apply the same
factoring to the fine-structure core term (freq - comp_freq = df_signed - foff). Match every numpy
constant and threshold bit-for-bit (RYDH, C_LIGHT_AA, the Holtsmark coefficients, the E_1 branch points).

You will also need a vectorized E_1(x) (`_vcse1f_tensor`, kgpu hydrogen.py) and a guarded exp(-x)
(`_fast_ex_gauss`). DEVICE/DTYPE picked once at module top. fp32 on MPS; the float floor is ~1e-6."""

_L6_CONTRACT = """Your module MUST define, importable as `port`:
  - port.DEVICE, port.DTYPE  (device + working dtype picked once)
  - port.sofbeta_grid(beta, p, n, m, propbm, c_arr, d_arr, pp_arr, beta_arr) -> tensor [D, n_w]
        beta is a [D,n_w] array-like, p a [D,1] array-like; n,m python ints; the tables are array-likes.
        Cast onto DEVICE/DTYPE inside.
  - port.hydrogen_profile_grid(n, m, delta_lambda_nm, hyd, tabs, foff, fwt, n_fine) -> tensor [D, n_w]
        n,m python ints; delta_lambda_nm a [D, n_w] array-like (offset in nm of each pixel from centre);
        hyd: dict of per-depth state — each value a [D] or [D,1] array-like:
             t3nhe, t3nh2, fo, dopph, c1d, c2d, y1s, y1b, gcon1, gcon2, pp, ne, xnfph_0
        tabs: dict of tables — asum[96], y1wtm[2,2], xknmtb[4,3], propbm[7,5,15], c[5,7], d[5,7],
              pp[5], beta[15] (array-likes)
        foff, fwt: array-likes [n_fine] (fine-structure offsets in Hz and weights); n_fine: python int
        returns phi[D, n_w]
All returned tensors may be on DEVICE; the harness moves them to CPU/fp64. You MAY define helpers
(the vectorized E_1, the guarded exp, the table-gather index math)."""

_L6_HARNESS = r'''
import numpy as np, torch, math, pathlib
REFDIR = pathlib.Path("reference")
C = np.load(REFDIR/"full_lines_data.npz", allow_pickle=True)
A = np.load(REFDIR/"atmosphere.npz")

def to64(x):
    if torch.is_tensor(x): return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, float)
''' + _L6_NUMPY + r'''

# --- per-depth hydrogen state + tables (same arrays the numpy twin reads) ---
T = A["temperature"]; xne = np.maximum(A["electron_density"], 1e-40)
n_depths = T.size
tabs = dict(asum=C["htab_asum"], y1wtm=C["htab_y1wtm"], xknmtb=C["htab_xknmtb"],
            propbm=C["htab_propbm"], c=C["htab_c"], d=C["htab_d"],
            pp=C["htab_pp"], beta=C["htab_beta"])
fkeys=C["fine_keys"]; foff_a=C["fine_offsets"]; fwt_a=C["fine_weights"]; fn_a=C["fine_n"]
fine_map={(int(fkeys[j,0]),int(fkeys[j,1])):(foff_a[j],fwt_a[j],int(fn_a[j])) for j in range(fkeys.shape[0])}

# H-beta (n=2, m=4): build the per-depth state for ALL depths, and the [D,n_w] offset grid
n, m = 2, 4
foff, fwt, nf = fine_map[(2, 4)]
RYDH=3.2880515e15; C_LIGHT_AA=2.99792458e18
wavenm = C_LIGHT_AA / (RYDH*((m*m-n*n)/(m*m*n*n)))   # H-beta centre wavelength in Angstrom
# a representative wavelength grid spanning the core to the far wing (+-30 nm of H-beta), per depth
wl_nm = np.linspace(wavenm/10.0 - 30.0, wavenm/10.0 + 30.0, 601)
line_wl = wavenm/10.0
dlam = (wl_nm - line_wl)[None, :].repeat(n_depths, 0)   # [D, n_w] offset in nm

states = [hydrogen_state_numpy(di, T, xne, A["xnf_he1"], A["xnf_h2"], A["xnfph"],
                               A["turbulent_velocity"]) for di in range(n_depths)]
hyd = {k: np.array([s[k] for s in states], float).reshape(-1, 1)
       for k in ("t3nhe","t3nh2","fo","dopph","c1d","c2d","y1s","y1b","gcon1","gcon2","pp","ne","xnfph_0")}

# --- numpy reference: scalar profile, every (depth, pixel) ---
phi_ref = np.array([[hydrogen_line_profile(n, m, float(dlam[di, j]), states[di], tabs, foff, fwt, nf)
                     for j in range(dlam.shape[1])] for di in range(n_depths)])   # [D, n_w]

# --- torch port over the whole grid ---
phi_torch = to64(port.hydrogen_profile_grid(n, m, dlam, hyd, tabs, foff, fwt, nf))

big = phi_ref > 1e-12     # the opacity-bearing pixels
if big.sum() == 0:
    dev_prof = 1e9
else:
    dev_prof = float(np.max(np.abs(phi_torch[big] - phi_ref[big]) / np.abs(phi_ref[big])))

# also a direct sofbeta_grid check over a wide beta range at a deep layer
di = n_depths - 1
beta_test = np.logspace(-1, 4, 400)[None, :]    # [1, 400]
p_test = np.array([[states[di]["pp"]]])
s_ref = np.array([[sofbeta(float(b), states[di]["pp"], n, m,
                           tabs["propbm"], tabs["c"], tabs["d"], tabs["pp"], tabs["beta"])
                   for b in beta_test[0]]])
s_torch = to64(port.sofbeta_grid(beta_test, p_test, n, m, tabs["propbm"], tabs["c"],
                                 tabs["d"], tabs["pp"], tabs["beta"]))
denomS = np.where(s_ref!=0.0, np.abs(s_ref), 1.0)
dev_sof = float(np.max(np.abs(s_torch - s_ref)/denomS))

max_dev = max(dev_prof, dev_sof)
print(f"hprof4_grid_dev={dev_prof:.3e}  sofbeta_dev={dev_sof:.3e}  "
      f"(nonzero px={int(big.sum())})  device={port.DEVICE}")
print(f"PARITY {max_dev:.6e}")

# --- §9 timing: the profile-grid eval is the hot path; time it (warm, best-of-N) ---
import time as _t
for _ in range(3):
    Pb = port.hydrogen_profile_grid(n, m, dlam, hyd, tabs, foff, fwt, nf)
    if torch.is_tensor(Pb) and Pb.device.type=="mps": torch.mps.synchronize()
best = 1e9
for _ in range(20):
    t0=_t.perf_counter()
    Pb = port.hydrogen_profile_grid(n, m, dlam, hyd, tabs, foff, fwt, nf)
    if torch.is_tensor(Pb) and Pb.device.type=="mps": torch.mps.synchronize()
    best=min(best, _t.perf_counter()-t0)
print(f"TIME {best*1e3:.6f}")
'''


JOBS: dict[str, PortJob] = {
    "lecture4": PortJob(
        name="lecture4",
        numpy_source=_L4_NUMPY,
        spec=_L4_SPEC,
        contract=_L4_CONTRACT,
        harness=_L4_HARNESS,
        float_floor=1e-6,
        preamble="reference/L4.npz fields: v[481], a[6], H[6,481], h0tab/h1tab/h2tab[2001], "
                 "T/tk/xne/rho/nHI/n_FeI/U_FeI/tau each [80]. The fp32-MPS float floor for this "
                 "lecture is ~1e-6; a CPU fp64 run should reach ~1e-12.",
        # BORROW kgpu's already-validated branchless Voigt (line_opacity.harris_hav) as the basis
        # to REDUCE (bible §6) — it broadcasts (a,v) and selects the 3 regimes with torch.where;
        # the API reshapes it to the [na,nv] grid contract, NOT regenerating the physics.
        kgpu_borrow=[("line_opacity.py", 116, 164)],
    ),
    "lecture5": PortJob(
        name="lecture5",
        numpy_source=_L5_NUMPY,
        spec=_L5_SPEC,
        contract=_L5_CONTRACT,
        harness=_L5_HARNESS,
        float_floor=1e-6,
        preamble="The hot path is the line-opacity ACCUMULATION (scatter-add). reference/ holds "
                 "full_lines_data.npz (cat_* line columns ~12.5k lines, h0/h1/h2tab[2001]), "
                 "atmosphere.npz (population_per_ion/doppler_per_ion [80,6,139], hckt[80], xnf_*), and "
                 "diag.npz (wavelength[5941], continuum_absorption/scattering[80,5941]). The metal lines "
                 "are line_type==0. The fp32-MPS float floor is ~1e-6 on the opacity-bearing pixels; the "
                 "scatter `index_put_(accumulate=True)` is add-order non-deterministic at ~1e-7 (within floor).",
        # BORROW kgpu's BATCHED scatter accumulation: the [D,L] reach geometry, the ONE batched
        # [D,L,offset] deposit, and the index_put_ scatter, plus the branchless Harris kernels.
        kgpu_borrow=[("line_opacity.py", 116, 164),   # harris_hav (the L4 Voigt kernel)
                     ("line_opacity.py", 167, 193),   # harris_h_at_zero (back-solve the wing amplitude)
                     ("line_opacity.py", 199, 226),   # FASTEX tables + branchless gather
                     ("line_opacity.py", 557, 585),   # harris_hav_walk (the near-wing H-form)
                     ("line_opacity.py", 602, 617),   # _scatter_add_3d (index_put_ accumulate)
                     ("line_opacity.py", 623, 691),   # _wing_reach_batched (Stage-1+2 reach geometry)
                     ("line_opacity.py", 733, 816)],  # narrow-line batched [D,L,W] deposit (THE hot path)
    ),
    "lecture6": PortJob(
        name="lecture6",
        numpy_source=_L6_NUMPY,
        spec=_L6_SPEC,
        contract=_L6_CONTRACT,
        harness=_L6_HARNESS,
        float_floor=1e-6,
        preamble="The HPROF4 hydrogen Stark profile, vectorized over [depth, wavelength]. reference/ holds "
                 "full_lines_data.npz (htab_* Stark tables: asum[96], y1wtm[2,2], xknmtb[4,3], "
                 "propbm[7,5,15], c[5,7], d[5,7], pp[5], beta[15]; fine_* fine-structure components) and "
                 "atmosphere.npz (temperature, electron_density, xnf_he1/xnf_h2, xnfph[80,*], "
                 "turbulent_velocity each [80]). The fp32-MPS float floor is ~1e-6 on the opacity-bearing "
                 "pixels. PRECISION TRAP: del_freq = |freq - freqnm| is fp32 catastrophic cancellation in "
                 "the core (~6e14 Hz numbers); kgpu factors the wavelength difference out (cancellation-free) "
                 "— reproduce that exactly. kgpu CPU-fp64-promotes the per-depth state setup; here the harness "
                 "passes the state in directly, so the port only needs the cancellation-free del_freq.",
        # BORROW kgpu's vectorized HPROF4: _sofbeta (the 3 beta-regimes), _hydrogen_profile_grid (the
        # 3-piece profile + the cancellation-free del_freq + the width-selection), and the vectorized E_1.
        kgpu_borrow=[("hydrogen.py", 267, 350),   # _sofbeta + _fast_ex_gauss (the quasi-static profile)
                     ("hydrogen.py", 369, 529),   # _hydrogen_profile_grid (the 3-piece HPROF4 profile)
                     ("hydrogen.py", 532, 554)],  # _vcse1f_tensor (the vectorized E_1)
    ),
}


def main():
    ap = argparse.ArgumentParser(description="GPU-edition port worker (external API generates; we validate).")
    ap.add_argument("--job", required=True, choices=sorted(JOBS), help="which lecture port job to run")
    ap.add_argument("--model", default="gpt55", choices=sorted(MODELS), help="external code-gen model")
    ap.add_argument("--max-tries", type=int, default=6, help="max API generate->fix iterations")
    ap.add_argument("--squeeze", type=int, default=0, metavar="N",
                    help="after parity, run N optimization-squeeze rounds (bible §9; parity-gated + timed)")
    args = ap.parse_args()
    res = run_job(JOBS[args.job], model=args.model, max_tries=args.max_tries, squeeze_rounds=args.squeeze)
    print("\n=== RESULT ===")
    print(f"  job={res['name']} model={res['model']} passed={res['passed']} "
          f"max_rel_dev={res['max_rel_dev']} api_iterations={res['api_iterations']} time_ms={res['time_ms']}")
    if res["squeeze_applied"]:
        print(f"  squeeze: {len(res['squeeze_applied'])} change(s) adopted")
        for s in res["squeeze_applied"]:
            print(f"    - {s['note']}  ({s['ms_before']:.3f} -> {s['ms_after']:.3f} ms)")
    print(f"  module: {res['module_path']}")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
