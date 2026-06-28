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
