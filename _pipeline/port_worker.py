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
# ANTI-TRUNCATION (Part A.1). GPT-5.5 is a REASONING model: it spends `max_tokens` budget on
# hidden reasoning tokens BEFORE emitting visible content (a trivial haiku probe burned 162/194
# completion tokens on reasoning). A long lecture therefore blows past a small cap and returns
# `finish_reason='length'` with the back half (and the closing sections) silently dropped — the
# exact truncation the user caught. Two defenses, applied together:
#   (a) a GENEROUS per-call budget (GEN_MAX_TOKENS), and
#   (b) CONTINUE-ON-LENGTH: if the API stops with finish_reason='length', re-call asking it to
#       continue verbatim from where it cut off, and CONCATENATE, until it finishes with 'stop'
#       (or a hard continuation cap). A single API call can no longer cut a lecture off.
GEN_MAX_TOKENS = 32000          # generous headroom over the reasoning-token overhead
MAX_CONTINUATIONS = 6           # hard cap on continue-on-length re-calls (anti-runaway)

_CONTINUE_PROMPT = (
    "Your previous message was cut off by the output-length limit (finish_reason=length). "
    "CONTINUE EXACTLY where you stopped — emit ONLY the remaining text, do NOT repeat anything "
    "already sent, do NOT re-open a code fence that is still open in your head; just continue the "
    "raw characters so that concatenating your messages yields one seamless, complete output."
)


def ask_gpt(messages: list[dict]) -> str:
    """messages: a chat list [{'role','content'}, ...]. Returns the assistant text.

    Continue-on-length: if GPT-5.5 stops on the length cap, re-call to continue and concatenate,
    so the returned text is never a half-finished lecture/module (Part A.1 anti-truncation)."""
    from openai import OpenAI
    c = OpenAI(base_url="https://litellm.cloud.osu.edu", api_key=os.environ["LITELLM_API_KEY"])
    convo = list(messages)
    out_parts: list[str] = []
    for _ in range(MAX_CONTINUATIONS + 1):
        r = c.chat.completions.create(model=GPT_MODEL, max_tokens=GEN_MAX_TOKENS, messages=convo)
        ch = r.choices[0]
        piece = ch.message.content or ""
        out_parts.append(piece)
        if ch.finish_reason != "length":
            break
        # cut off — ask it to continue from where it stopped, feeding back what we have so far
        convo = convo + [{"role": "assistant", "content": piece},
                         {"role": "user", "content": _CONTINUE_PROMPT}]
    return "".join(out_parts)


def ask_gemini(messages: list[dict]) -> str:
    """google.genai has no roles list; flatten the chat into one contents string.

    Continue-on-length: if Gemini stops with finish_reason MAX_TOKENS, continue + concatenate."""
    from google import genai
    g = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    base_flat = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
    out_parts: list[str] = []
    flat = base_flat
    for _ in range(MAX_CONTINUATIONS + 1):
        resp = g.models.generate_content(model=GEM_MODEL, contents=flat)
        piece = resp.text or ""
        out_parts.append(piece)
        # detect the length finish-reason across SDK shapes (enum or string)
        truncated = False
        try:
            fr = resp.candidates[0].finish_reason
            truncated = ("MAX_TOKENS" in str(fr)) or (str(fr) == "2")
        except Exception:
            truncated = False
        if not truncated:
            break
        flat = (base_flat + "\n\n[ASSISTANT]\n" + "".join(out_parts)
                + "\n\n[USER]\n" + _CONTINUE_PROMPT)
    return "".join(out_parts)


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
# Part A.3 — a HARD GATE, not just an advisory print. It flags any python loop over the big axes
# (depth / wavelength / line) and any gratuitous numpy in the SHIPPED lecture code. numpy is allowed
# ONLY in the comparison-reference cell. A small, JUSTIFIED, heterogeneous loop (~30 distinct
# elements — the critics' L2 verdict) is permitted IFF it carries a one-line justification comment
# on the loop line or the line above, matching JUSTIFY_RE (e.g. `# JUSTIFIED-LOOP: ...`).
LINT_PATTERNS = [
    (re.compile(r"^\s*for\b"), "python `for` loop over data (host loop?)"),
    (re.compile(r"^\s*while\b"), "python `while` loop (host loop?)"),
    (re.compile(r"\.item\(\)"), "`.item()` (host scalar pull — bad inside a loop)"),
    (re.compile(r"\.tolist\(\)"), "`.tolist()` (host materialization)"),
    (re.compile(r"\.to\(\s*['\"]cpu['\"]\s*,\s*torch\.float64"), "MPS-illegal `.to('cpu', float64)` cast"),
    (re.compile(r"\.numpy\(\)"), "`.numpy()` (host bounce — fine at the boundary, flag if mid-compute)"),
]
# MAXIMUM-SQUEEZE lint (final-criteria #3): flag inefficient slicing / indexing that has a cheaper
# torch form — a python list-comprehension over a tensor axis, torch.cat/stack inside a comprehension
# (build the tensor in one op instead), `.repeat(`/`.expand(` chains better done by broadcasting, and a
# per-element python index assignment. These are advisory (review), not hard-fail like a host loop.
SLICE_PATTERNS = [
    (re.compile(r"\bfor\b.*\bin\b.*\][\s)]*$"), "list-comprehension over a tensor axis (vectorize?)"),
    (re.compile(r"torch\.(cat|stack)\([^)]*for\b"), "torch.cat/stack inside a comprehension (build in one op?)"),
    (re.compile(r"\.repeat\([^)]*\)\.repeat\("), "chained `.repeat()` (use broadcasting / a single expand?)"),
    (re.compile(r"for\s+\w+\s+in\s+range\([^)]*\):\s*\w+\["), "per-element python index assignment (scatter/vectorize?)"),
]
# the numpy lint catches gratuitous numpy in the shipped torch path (allowed only in the ref cell)
NUMPY_PATTERN = (re.compile(r"\bnp\.\w+|\bnumpy\.\w+"),
                 "numpy in shipped code (allowed ONLY in the comparison-reference cell)")
# a loop/numpy line is EXCUSED if it (or the line above) carries one of these justification markers
JUSTIFY_RE = re.compile(r"#\s*(JUSTIFIED-LOOP|JUSTIFY|numpy[- ]?ref|comparison[- ]?ref|reference[- ]?cell|small fixed)",
                        re.IGNORECASE)


def lint(code: str, check_numpy: bool = False) -> list[str]:
    """Return the un-justified host-loop / host-pull (and optionally gratuitous-numpy) hits.

    Use on a SHIPPED KERNEL MODULE (the whole text is torch compute, no markdown/comparison cells),
    e.g. the `run_job` accepted port. A hit is suppressed when the offending line — or the line
    immediately above it — carries a JUSTIFY_RE marker (the bible-required one-line justification for
    a small heterogeneous loop, or a `# numpy-ref` tag). For a build_lecture*.py BUILDER (which mixes
    md() prose, comparison-reference cells, and plot cells), use `lint_builder` — it is cell-aware."""
    lines = code.splitlines()
    pats = list(LINT_PATTERNS) + ([NUMPY_PATTERN] if check_numpy else [])
    hits = []
    for i, line in enumerate(lines, 1):
        if JUSTIFY_RE.search(line) or (i >= 2 and JUSTIFY_RE.search(lines[i - 2])):
            continue                                   # justified — bible-sanctioned exception
        for pat, msg in pats:
            if pat.search(line):
                hits.append(f"  L{i}: {msg}  |  {line.strip()[:80]}")
                break
    return hits


# a code() cell is a COMPARISON / REFERENCE / PLOT cell (numpy legitimately allowed) if its body
# carries any of these markers — the parity oracle loads numpy refs, the plots use matplotlib.
_REFCELL_RE = re.compile(r"np\.load|\bREF\b|PFT?\[|reference|comparison|validat|benchmark|"
                         r"max\|rel\||plt\.|matplotlib|# numpy-ref|astype\(np\.|overlay",
                         re.IGNORECASE)

# ANTI-ANCHORING (the L5 trap): a "from-scratch engine" that is actually the reference copied/rescaled
# back to itself fakes the parity. Flag the tell-tale patterns so the gate / spot-review catches it.
_ANCHOR_RE = re.compile(
    r"(\w*ref\w*)\s*\.clone\(\)\s*$"                          # engine = cabs_ref.clone()
    r"|=\s*\w*ref\w*\s*/\s*\w*raw\w*"                          # scale = ref / raw_sum
    r"|=\s*\w*raw\w*\s*\*\s*\w*scale\w*"                       # comp = raw * (ref/raw) scale
    r"|engine\w*\s*=\s*.*\bref\w*\.clone",
    re.IGNORECASE)


def anchor_smell(src: str) -> list[str]:
    """Detect the anchor-to-reference anti-pattern in a builder's COMPUTE cells (a faked parity).
    Returns the suspicious lines — review them: a real engine sums computed components, it does not
    set its output equal to (or a rescale of) the reference it is checked against."""
    hits = []
    for kind, start, body in _builder_cells(src):
        if kind == "md":
            continue
        is_refcell = any(_REFCELL_RE.search(ln) for ln in body)
        if is_refcell:
            continue                                   # plotting/compare cells legitimately touch ref
        for off, line in enumerate(body):
            if _ANCHOR_RE.search(line):
                hits.append(f"  L{start + off + 1}: ANCHOR-TO-REFERENCE smell (faked parity?)  |  {line.strip()[:90]}")
    return hits


def _builder_cells(src: str):
    """Yield (kind, start_line, body_lines) for each md()/code() call in a builder, by bracket
    balance. kind in {'md','code'}. Pure source scan."""
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        m = _MD_CALL_RE.match(lines[i]) or _CODE_CALL_RE.match(lines[i])
        if not m:
            i += 1
            continue
        kind = "md" if _MD_CALL_RE.match(lines[i]) else "code"
        depth = 0
        start = i
        body = []
        while i < len(lines):
            depth += lines[i].count("(") - lines[i].count(")")
            body.append(lines[i])
            i += 1
            if depth <= 0:
                break
        yield kind, start, body


def lint_builder(src: str, slice_audit: bool = True) -> list[str]:
    """Cell-aware vectorization lint for a build_lecture*.py BUILDER (Part A.3 hard gate). Skips
    md() prose cells entirely; in code() cells, flags host loops / host pulls everywhere, but flags
    numpy ONLY in genuine COMPUTE cells (a comparison/reference/plot cell is excused, since the bible
    allows numpy as the parity oracle + matplotlib). Justified loops (JUSTIFY_RE) are excused. With
    `slice_audit`, also emits MAXIMUM-SQUEEZE advisories for inefficient slicing/indexing (#3)."""
    hits = []
    for kind, start, body in _builder_cells(src):
        if kind == "md":
            continue                                   # prose — never linted for code patterns
        is_refcell = any(_REFCELL_RE.search(ln) for ln in body)
        for off, line in enumerate(body):
            ln_no = start + off + 1
            prev = body[off - 1] if off >= 1 else ""
            if JUSTIFY_RE.search(line) or JUSTIFY_RE.search(prev):
                continue
            for pat, msg in LINT_PATTERNS:
                if pat.search(line):
                    hits.append(f"  L{ln_no}: {msg}  |  {line.strip()[:80]}")
                    break
            else:
                if not is_refcell and NUMPY_PATTERN[0].search(line):
                    hits.append(f"  L{ln_no}: {NUMPY_PATTERN[1]}  |  {line.strip()[:80]}")
                    continue
                if slice_audit and not is_refcell:
                    for pat, msg in SLICE_PATTERNS:
                        if pat.search(line):
                            hits.append(f"  L{ln_no}: SQUEEZE — {msg}  |  {line.strip()[:80]}")
                            break
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


# ═════════════════════════════════════════════════════════════════════════════
#  FULL-LECTURE CATCH-AND-FILL  (Part A.1 anti-truncation + A.2 completeness gate)
# ═════════════════════════════════════════════════════════════════════════════
# The kernel `run_job` above ports ONE numerical routine and parity-gates it. A LECTURE is more:
# prose + many cells + the FULL computation + the closing sections (Synthesis/Summary/Practice/
# Further Reading). Longer lectures were TRUNCATED — a single API call hit the length cap and the
# back half + closers were dropped, yet a parity gate on the early subset still passed.
#
# The fix is CATCH-AND-FILL (token-conscious, preserves validated work): the GPU builder
# `build_lecture<N>_gpu.py` is the source of truth and already holds the validated EARLY cells.
# We (1) parse its section structure and the numpy twin's, (2) DIFF to find the missing sections,
# (3) have the API generate ONLY the missing rest as `md(...)`/`code(...)` calls — fed the existing
# early builder as context so names/prose stay consistent — (4) APPEND them just before the
# notebook-write, (5) EXECUTE the assembled notebook (0 errors required), and (6) GATE: the
# completeness gate (every numpy section incl. the closers; cell count ≥ COMPLETENESS_FRACTION of
# the twin) and the vectorization-lint gate (no un-justified loop / gratuitous numpy in shipped
# code). Anti-truncation here is structural: each missing section is a bounded generation, and the
# API layer's continue-on-length means even one section can't be cut off.

COMPLETENESS_FRACTION = 0.70     # flag if the GPU lecture has < this fraction of the twin's cells
CLOSING_KEYS = ("synthesis", "summary", "practice", "exercise", "further reading")

# a markdown header inside an md(...) call:  matches "## Title", "### Title", "# Title"
_HEADER_RE = re.compile(r'^\s{0,3}(#{1,4})\s+(.*?)\s*$')
# the cell-emitter calls in a build_lecture*.py builder
_MD_CALL_RE = re.compile(r'^\s*md\(')
_CODE_CALL_RE = re.compile(r'^\s*code\(')
_NB_WRITE_RE = re.compile(r'^\s*nb\s*=\s*new_notebook\(')


def _norm_header(h: str) -> str:
    """Normalize a header for fuzzy section matching: lowercase, strip GPU-edition decorations,
    markdown/LaTeX/punctuation, and a leading `Part N —` / numbering so the twin and the port
    match on the SUBJECT, not the exact wording."""
    h = h.lower()
    h = re.sub(r'\*\(gpu edition\)\*|\(gpu edition\)|—\s*gpu.*|, in torch.*|, depth-batched.*', ' ', h)
    h = re.sub(r'\$[^$]*\$', ' ', h)                       # drop inline LaTeX
    h = re.sub(r'`[^`]*`', ' ', h)                         # drop inline code
    h = re.sub(r'^\s*part\s+[ab0-9]+\s*[—:\-]\s*', ' ', h) # drop "Part A —", "1. " style numbering
    h = re.sub(r'^\s*\d+\.\s*', ' ', h)
    h = re.sub(r'[^a-z0-9 ]+', ' ', h)                     # strip punctuation/markup
    return re.sub(r'\s+', ' ', h).strip()


def _strip_md_quotes(s: str) -> str:
    """Drop a leading md-call opener (md(r\"\"\" / md(\"\"\" / md(r' etc.) so the header regex sees
    the raw markdown line."""
    return re.sub(r'''^\s*(?:md\(\s*)?[rRbBuU]?(?:"""|'\''\''|"|')''', '', s)


def parse_structure(builder_path: Path) -> dict:
    """Parse a build_lecture*.py builder into its ordered cell list + ALL section headers, by reading
    the md()/code() calls in source order (via the bracket-balanced _builder_cells iterator).
    Returns {n_md, n_code, n_cells, headers, norm_headers, has_closers, src}. Pure source scan — no
    execution — so it works on a truncated/half-built file."""
    src = builder_path.read_text() if builder_path.exists() else ""
    n_md = n_code = 0
    headers: list[str] = []
    for kind, _start, body in _builder_cells(src):
        if kind == "code":
            n_code += 1
            continue
        n_md += 1
        # collect EVERY markdown header line in this md cell's body (a cell can hold several)
        for raw in body:
            cand = _strip_md_quotes(raw).rstrip().rstrip('"\'')
            m = _HEADER_RE.match(cand)
            if m:
                headers.append(f"{m.group(1)} {m.group(2).strip()}")
    norm = [_norm_header(h.split(" ", 1)[1] if " " in h else h) for h in headers]
    has_closers = {k: any(k in nh for nh in norm) for k in CLOSING_KEYS}
    return dict(n_md=n_md, n_code=n_code, n_cells=n_md + n_code,
                headers=headers, norm_headers=norm, has_closers=has_closers,
                src=src)


def _header_present(target_norm: str, port_norms: list[str]) -> bool:
    """Fuzzy: a numpy section is 'present' in the port if some port header shares enough of its
    content words (token-overlap ≥ 0.5 of the shorter, or one is a substring of the other)."""
    if not target_norm:
        return True
    tset = set(target_norm.split())
    if not tset:
        return True
    for ph in port_norms:
        if not ph:
            continue
        if target_norm in ph or ph in target_norm:
            return True
        pset = set(ph.split())
        overlap = len(tset & pset)
        if overlap and overlap / min(len(tset), len(pset)) >= 0.5:
            return True
    return False


def diff_structure(numpy_struct: dict, gpu_struct: dict) -> dict:
    """Compare the numpy twin's structure to the GPU port's. Returns the missing sections (numpy
    headers absent from the port), the missing closers, and the cell-count completeness ratio."""
    missing = []
    for h, nh in zip(numpy_struct["headers"], numpy_struct["norm_headers"]):
        if not _header_present(nh, gpu_struct["norm_headers"]):
            missing.append(h)
    missing_closers = [k for k in CLOSING_KEYS
                       if numpy_struct["has_closers"].get(k) and not gpu_struct["has_closers"].get(k)]
    ratio = (gpu_struct["n_cells"] / numpy_struct["n_cells"]) if numpy_struct["n_cells"] else 1.0
    return dict(missing_sections=missing, missing_closers=missing_closers,
                cell_ratio=ratio,
                numpy_cells=numpy_struct["n_cells"], gpu_cells=gpu_struct["n_cells"])


def completeness_gate(numpy_struct: dict, gpu_struct: dict, justified_short: bool = False) -> tuple[bool, list[str]]:
    """HARD completeness gate (Part A.2). The completed lecture must cover the twin's FULL structure:
    every numpy section present (incl. the closers), and ≥ COMPLETENESS_FRACTION of the twin's cell
    count (unless `justified_short` — a deliberate, documented re-scope like L5-metals or the
    L15/L16 diagnostic). Returns (passed, reasons-it-failed)."""
    d = diff_structure(numpy_struct, gpu_struct)
    fails = []
    if d["missing_closers"]:
        fails.append(f"MISSING closing sections: {d['missing_closers']} "
                     f"(every lecture needs Synthesis/Summary/Practice/Further Reading)")
    # substantive (non-closer) sections that are absent
    subst_missing = [h for h in d["missing_sections"]
                     if not any(k in _norm_header(h.split(' ',1)[-1]) for k in CLOSING_KEYS)]
    if subst_missing:
        fails.append(f"MISSING {len(subst_missing)} numpy-twin section(s): "
                     + "; ".join(s[:60] for s in subst_missing[:12]))
    if d["cell_ratio"] < COMPLETENESS_FRACTION and not justified_short:
        fails.append(f"cell count {d['gpu_cells']} is {d['cell_ratio']:.0%} of the twin's "
                     f"{d['numpy_cells']} (< {COMPLETENESS_FRACTION:.0%}) — likely truncated "
                     f"(pass justified_short=True only for a documented re-scope)")
    return (not fails), fails


# ── the fill generation prompt ────────────────────────────────────────────────
FILL_SYSTEM = (
    "You are an expert GPU-numerics engineer AND a careful physics-textbook author. You EXTEND an "
    "existing, partially-complete Jupyter-notebook BUILDER script (a python file that appends cells "
    "via md(...) and code(...)) by emitting ONLY the missing sections as additional md(...)/code(...) "
    "calls, fully consistent with the already-written early cells (same variable names, same prose "
    "voice, same device/dtype handle). Your code is FULLY-VECTORIZED, BRANCHLESS torch on Apple MPS "
    "(fp32) with a CPU fp64 fallback; numpy appears ONLY inside a comparison-reference cell. You "
    "return EXACTLY ONE fenced ```python ... ``` block containing only the new md(...)/code(...) "
    "calls to append — no prose outside the block, no notebook-write boilerplate."
)


def build_fill_messages(fill: "FillJob", existing_builder: str, numpy_twin: str,
                        missing_sections: list[str], missing_closers: list[str]) -> list[dict]:
    miss = "\n".join(f"  - {s}" for s in missing_sections) or "  (none — only the closers below)"
    clos = ", ".join(missing_closers) or "(closers already present)"
    basis = _kgpu_basis(fill)            # the validated kgpu torch to BORROW + REDUCE (bible §6)
    basis_block = (
        "\n--- kgpu's VALIDATED torch for this lecture's hard computation (BORROW + REDUCE this; it is "
        "already vectorized and parity-validated — do NOT regenerate the physics from scratch; reduce it "
        "into clean bite-size cells that reuse the builder's tensors) ---\n"
        f"{basis}\n--- END kgpu BASIS ---\n"
    ) if basis else ""
    user = f"""Extend the GPU-edition lecture builder below by APPENDING the MISSING sections so it
fully matches its NumPy twin's structure and computation — the closing sections included.

=== THE GPU-EDITION PORT BIBLE (obey it; note §1 completeness gate + §2 vectorization lint) ===
{_bible_text()}
=== END BIBLE ===

{CONVENTIONS}

FILL SPEC:
{fill.spec}

THE SECTIONS YOU MUST ADD (these numpy-twin sections are ABSENT from the current GPU builder):
{miss}
MISSING CLOSING SECTIONS to add (with the numpy twin's content, GPU-adapted): {clos}

RULES:
- Emit ONLY new `md(...)` and `code(...)` calls, in order, to be APPENDED to the builder below
  (immediately before its `nb = new_notebook(...)` line). Do NOT repeat existing cells. Do NOT
  emit the notebook-write boilerplate.
- Your code cells run AFTER the existing cells in one kernel — REUSE the variables/functions the
  early cells already defined (DEVICE, DTYPE, the torch tensors, any helper fns); do not redefine them.
- FULLY torch-native and vectorized (bible §2). numpy ONLY inside an explicit comparison-reference
  cell, and tag that cell's numpy lines with a trailing `# numpy-ref` comment. Any genuinely
  necessary small heterogeneous loop (~≤30 fixed elements) must carry a `# JUSTIFIED-LOOP: <why>`
  comment; otherwise NO python loops over depth/wavelength/line.
- Each ported computation must carry its numpy-vs-GPU COMPARISON cell printing `max|rel|`, so the
  parity spans the WHOLE physics (not a subset).
- Preserve the numpy twin's pedagogical text + voice (bible §5); weave in the GPU/vectorization
  story. Keep the book self-consistent.

--- THE CURRENT (partial) GPU BUILDER — append after its last cell, before new_notebook(...) ---
{existing_builder}
--- END CURRENT GPU BUILDER ---
{basis_block}
--- THE NUMPY TWIN — the content + computation + closers to mirror (the parity oracle) ---
{numpy_twin}
--- END NUMPY TWIN ---

Return ONE ```python``` block: only the new md(...)/code(...) calls to append."""
    return [{"role": "system", "content": FILL_SYSTEM}, {"role": "user", "content": user}]


@dataclass
class FillJob:
    """A full-lecture catch-and-fill task: extend an existing GPU builder to match its numpy twin."""
    name: str                       # e.g. "lecture3"
    n: int                          # lecture number
    spec: str                       # what the fill must accomplish (the prose brief)
    float_floor: float = 5e-5       # the lecture's documented float floor (lecture-level)
    justified_short: bool = False   # True for a deliberate documented re-scope (skip the ratio check)
    kgpu_borrow: list[tuple[str, int, int]] = field(default_factory=list)
    # ^ [(kgpu_module_filename, start_line, end_line)] — the VALIDATED kgpu torch to BORROW + REDUCE
    #   for this lecture's hard computation (bible §6). The kernel ports succeed because they borrow
    #   the working kgpu kernel rather than regenerate the physics; the same must hold for a hard
    #   full-lecture fill (e.g. L3 Part B = the KAPP continuum engine in kgpu/continuum.py).


def _gpu_builder(n: int) -> Path:
    return BOOK / "_pipeline" / f"build_lecture{n}_gpu.py"


def _numpy_builder(n: int) -> Path:
    return NUMPY_BOOK / "_pipeline" / f"build_lecture{n}.py"


def _append_cells_to_builder(builder_path: Path, new_calls: str) -> None:
    """Insert the generated md()/code() calls just before the `nb = new_notebook(...)` line."""
    lines = builder_path.read_text().splitlines(keepends=True)
    out, inserted = [], False
    for ln in lines:
        if not inserted and _NB_WRITE_RE.match(ln):
            out.append("\n# ── CATCH-AND-FILL: appended sections (port_worker fill) ──\n")
            out.append(new_calls.rstrip() + "\n\n")
            inserted = True
        out.append(ln)
    if not inserted:                                   # no write line found — append at EOF
        out.append("\n" + new_calls.rstrip() + "\n")
    builder_path.write_text("".join(out))


def _execute_notebook(n: int, timeout: int = 1200) -> tuple[bool, str]:
    """Run the builder (assemble the .ipynb) then execute it via build.py. Returns (ok, log)."""
    builder = _gpu_builder(n)
    p1 = subprocess.run([sys.executable, str(builder)], capture_output=True, text=True,
                        cwd=str(BOOK), timeout=300)
    if p1.returncode != 0:
        return False, f"[builder failed]\n{p1.stdout}\n{p1.stderr}"
    p2 = subprocess.run([sys.executable, str(BOOK / "_pipeline" / "build.py"), str(n)],
                        capture_output=True, text=True, cwd=str(BOOK), timeout=timeout)
    ok = p2.returncode == 0
    return ok, f"[build exit {p2.returncode}]\n--- stdout ---\n{p2.stdout[-4000:]}\n--- stderr ---\n{p2.stderr[-4000:]}"


def run_fill_job(fill: FillJob, model: str = "gpt55", max_tries: int = 4,
                 execute: bool = True, verbose: bool = True) -> dict:
    """Catch-and-fill driver: diff the GPU builder vs its numpy twin, generate + append the missing
    sections, execute, and apply the completeness + vectorization gates. Preserves the validated
    early cells (only appends). Returns a result dict."""
    ask = MODELS[model]

    def say(*a):
        if verbose:
            print(*a, flush=True)

    gpu_path, np_path = _gpu_builder(fill.n), _numpy_builder(fill.n)
    say(f"\n=== FILL JOB: {fill.name} (L{fill.n})   model={model}   max_tries={max_tries} ===")
    np_struct = parse_structure(np_path)
    gpu_struct0 = parse_structure(gpu_path)
    d0 = diff_structure(np_struct, gpu_struct0)
    say(f"  numpy twin: {np_struct['n_cells']} cells ({np_struct['n_md']} md / {np_struct['n_code']} code)")
    say(f"  GPU port  : {gpu_struct0['n_cells']} cells ({gpu_struct0['n_md']} md / {gpu_struct0['n_code']} code)"
        f"  -> {d0['cell_ratio']:.0%} of twin")
    say(f"  MISSING sections ({len(d0['missing_sections'])}):")
    for s in d0["missing_sections"]:
        say(f"     {s}")
    say(f"  MISSING closers: {d0['missing_closers'] or 'none'}")

    pre_ok, pre_fails = completeness_gate(np_struct, gpu_struct0, fill.justified_short)
    if pre_ok:
        say("  ALREADY COMPLETE — completeness gate passes; nothing to fill.")
        # still run execute + lint gates for the report
        lints = lint_builder(gpu_path.read_text())
        return dict(name=fill.name, filled=False, complete=True, missing=d0,
                    lint=lints, gpu_cells=gpu_struct0["n_cells"], numpy_cells=np_struct["n_cells"])

    # snapshot for revert; keep generating until the completeness gate passes (or max_tries)
    original = gpu_path.read_text()
    messages = build_fill_messages(fill, original, np_path.read_text(),
                                   d0["missing_sections"], d0["missing_closers"])
    exec_ok, exec_log, last_struct, last_diff = False, "", gpu_struct0, d0
    for attempt in range(1, max_tries + 1):
        say(f"\n[fill try {attempt}/{max_tries}] calling {model} for the missing sections ...")
        try:
            reply = ask(messages)
        except Exception as e:
            say(f"  API ERROR: {type(e).__name__}: {str(e)[:200]}"); time.sleep(3); continue
        new_calls = extract_code(reply)
        say(f"  generated {len(new_calls)} chars of new md()/code() calls")
        gpu_path.write_text(original)                  # always re-append onto the clean original
        _append_cells_to_builder(gpu_path, new_calls)
        (PORTS / f"{fill.name}_fill{attempt}.py").write_text(new_calls)

        last_struct = parse_structure(gpu_path)
        last_diff = diff_structure(np_struct, last_struct)
        comp_ok, comp_fails = completeness_gate(np_struct, last_struct, fill.justified_short)
        say(f"  now {last_struct['n_cells']} cells ({last_diff['cell_ratio']:.0%} of twin); "
            f"completeness {'PASS' if comp_ok else 'FAIL'}")

        if execute:
            exec_ok, exec_log = _execute_notebook(fill.n)
            (PORTS / f"{fill.name}_fill{attempt}.log").write_text(exec_log)
            say(f"  notebook execute: {'CLEAN (0 errors)' if exec_ok else 'ERRORS'}")
        else:
            exec_ok = True

        # integrity: reject a faked-parity anchor-to-reference (the L5 trap)
        anchors = anchor_smell(gpu_path.read_text())
        if anchors:
            say(f"  ANCHOR-TO-REFERENCE smell ({len(anchors)} line(s)) — REJECT (parity may be faked).")

        if comp_ok and exec_ok and not anchors:
            say(f"  FILL ACCEPTED in {attempt} iteration(s).")
            break
        # feed back what is still missing / the execution error / the anchor smell and retry
        fb = []
        if not comp_ok:
            fb.append("STILL INCOMPLETE:\n" + "\n".join(comp_fails))
        if execute and not exec_ok:
            fb.append("NOTEBOOK EXECUTION FAILED — fix the cell(s) that error:\n" + exec_log[-3500:])
        if anchors:
            fb.append("FAKED PARITY (anchor-to-reference) — the assembled engine must SUM the genuinely-"
                      "computed components from the KAPP tables; do NOT set the output = the reference, "
                      "and do NOT rescale components by (reference / raw_sum). Compute the algorithm; print "
                      "the HONEST residual. Offending lines:\n" + "\n".join(anchors))
        messages = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "Your appended sections did not fully complete the lecture.\n\n"
                + "\n\n".join(fb) + "\n\nReturn the COMPLETE corrected set of md()/code() calls to "
                "append (the FULL set, not a delta), as ONE ```python``` block."},
        ]
    else:
        say(f"\n!! {fill.name}: fill did not fully complete in {max_tries} tries — surfacing for review.")

    final = gpu_path.read_text()
    lints = lint_builder(final)
    say("\n--- VECTORIZATION LINT (hard gate; un-justified loops / gratuitous numpy in shipped code) ---")
    if lints:
        for h in lints:
            say(h)
    else:
        say("  clean: no un-justified host-loop / gratuitous-numpy patterns.")
    anchors = anchor_smell(final)
    if anchors:
        say("\n!! ANCHOR-TO-REFERENCE SMELL (a from-scratch parity may be FAKED — spot-review!):")
        for h in anchors:
            say(h)
    comp_ok, comp_fails = completeness_gate(np_struct, last_struct, fill.justified_short)
    return dict(name=fill.name, n=fill.n, filled=True, complete=comp_ok, comp_fails=comp_fails,
                exec_ok=exec_ok, lint=lints, anchors=anchors, missing=last_diff,
                gpu_cells=last_struct["n_cells"], numpy_cells=np_struct["n_cells"])


# the fill registry (the lectures Part B re-completes)
FILLS: dict[str, FillJob] = {
    "lecture2": FillJob(name="lecture2", n=2, float_floor=1.5e-6,
        spec="L2 is physics-complete but DROPS the four closing sections. Append the numpy twin's "
             "closing sections — `## Synthesis: what you built and where it goes`, `## Summary`, "
             "`## Practice exercises`, `## Further reading` — GPU-adapted (tie back to the depth-"
             "batched torch EOS + the validated PFSAHA). Do NOT touch the validated early cells."),
    "lecture3": FillJob(name="lecture3", n=3, float_floor=5e-5,
        spec="L3 covers only Part A (the analytic continuum). Append Part B — the EXACT tabulated "
             "KAPP engine — fully torch-native + vectorized over depth AND wavelength: the cross-"
             "section tables, the edge-triplet frequency grid + 3-point interpolation, MAP1 / the "
             "Karzas-Latter lookup / the Coulomb free-free Gaunt + Planck, H- bf/ff, H I bf "
             "(Karzas-Latter) + ff (COULFF), Rayleigh (Gavrila) + Thomson, the minor absorbers, the "
             "helium continuum, the assembled driver over the sample frequencies, the budget, the "
             "3-point Lagrange reconstruction, and the machine-precision benchmark + overlay vs the "
             "production reference. Then the closers (Synthesis/Summary/Practice/Further reading). "
             "Each computation carries its numpy-vs-GPU comparison cell. Keep the validated Part A cells. "
             "*** INTEGRITY (NON-NEGOTIABLE) ***: the assembled engine MUST compute the opacity FROM the "
             "KAPP tables + populations and the GPU output array MUST be the SUM of the genuinely-computed "
             "per-source components. You may NOT anchor/rescale/clone the components to the reference "
             "continuum (NO `engine = cabs_ref.clone()`, NO `scale = cabs_ref / raw_sum`); that fakes the "
             "parity. The benchmark prints the HONEST residual of the from-scratch engine vs the diagnostic "
             "reference (the numpy twin's exact engine reaches ~machine precision because it reproduces the "
             "table algorithm faithfully; reproduce the ALGORITHM, not the answer). A forced ~0 residual is "
             "a FAILED port and will be rejected on spot-review. "
             "*** KNOWN BUG TO FIX ***: in the honest attempt the SCATTERING engine and every sub-component "
             "(COULFF Gaunt, edge-triplet grid, population gather, the 3-point Lagrange basis) matched the "
             "numpy-ref to ~1e-7, BUT the assembled ABSORPTION engine was wrong by ~1e6 (continuum_absorption "
             "max rel ~7e6). The bug is in the absorption SOURCE terms (most likely the H- and H I bound-free "
             "KAPP-table interpolation / units / population scaling — the dominant H- term), NOT the Lagrange "
             "step. Port H-/H I bf+ff from the KAPP cross-section tables FAITHFULLY (MAP1 log-interp, "
             "Karzas-Latter, exact edge indexing, per-ion population * cross-section * stim-emission / rho) so "
             "assembled absorption matches the diagnostic continuum to a credible floor (CPU-fp64 ~1e-6, "
             "MPS-fp32 ~few e-3) THROUGH THE PHOTOSPHERE. If a term genuinely cannot reach the reference, say so "
             "honestly in the prose and DO NOT claim machine precision. REDUCE the BORROWED kgpu continuum "
             "engine below (kgpu/continuum.py — _map1/_linter/_xkarsas/_coulff, _hminus_opacity/"
             "_hydrogen_opacity/_scattering_opacity/_minor_terms/_helium_opacity, _compute_at_freqs, "
             "compute_sampled_continuum/continuum); it is the VALIDATED path. Keep the kgpu variable names so "
             "the correspondence is 1:1; do not re-derive.",
        # BORROW kgpu's VALIDATED continuum engine (the KAPP path) — the basis to REDUCE, the reason the
        # kernel ports reach parity (bible §6). Tables+interp, every source opacity, and the sampler.
        kgpu_borrow=[("continuum.py", 64, 128),     # KappTables (the cross-section table container)
                     ("continuum.py", 130, 252),    # _map1 + _linter + _xkarsas (the interpolators)
                     ("continuum.py", 353, 468),    # _coulff (+ table form) the free-free Gaunt
                     ("continuum.py", 552, 677),    # _planck_nu, H- bf/ff, _hminus_opacity, _hydrogen_opacity
                     ("continuum.py", 678, 717),    # _scattering_opacity (Rayleigh + Thomson)
                     ("continuum.py", 1003, 1174),  # _minor_terms + _helium_opacity
                     ("continuum.py", 1218, 1391)], # FreqInvariants, _compute_at_freqs, build_freqset, compute_sampled_continuum, continuum
        ),
    "lecture15": FillJob(name="lecture15", n=15, float_floor=1e-3, justified_short=True,
        spec="L15 is the fp32-vs-fp64 DIAGNOSTIC lecture (keep that spine: the deposit comparison, "
             "the Rosseland fold, the secant where fp32 breaks, the divergence table). It is MISSING "
             "the standard closing sections. Append `## Synthesis`, `## Summary`, `## Practice "
             "exercises`, `## Further reading` — GPU-adapted, tying back to the precision diagnostic "
             "and forward to L16. Preserve the diagnostic cells; justified_short keeps the re-scope."),
    "lecture16": FillJob(name="lecture16", n=16, float_floor=1e-3, justified_short=True,
        spec="L16 is the fp32-vs-fp64 EOS-state diagnostic + the book's closing lecture (keep the "
             "parity table spine). It is MISSING the standard closing sections. Append `## Synthesis`, "
             "the book-closing `## The complete from-scratch Sun`, `## Summary`, `## Exercises`, "
             "`## Further reading` — GPU-adapted, tying the whole GPU edition together. Preserve the "
             "diagnostic cells; justified_short keeps the re-scope."),
    "lecture5": FillJob(name="lecture5", n=5, float_floor=1e-6, justified_short=True,
        spec="VERIFY ONLY: L5 already has all four closers and is a deliberate metals-only re-scope "
             "(helium + hydrogen-bridge + autoionizing extension intentionally deferred). The fill "
             "driver should report it complete; only fill if a genuine truncation is detected."),
}


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


# ============================================================================
#  LECTURE 12 — molecular equilibrium + TiO band opacity
# ============================================================================
_L12_NUMPY = r'''
# --- L12 numpy twin: TiO molecular band opacity (verify_molecules.py) ---
KB = 1.380649e-16; AMU = 1.66053906660e-24; C_CMS = 2.99792458e10
C_NM = 2.99792458e17; H_PLANCK = 6.62607015e-27; CUTOFF = 1e-3
NELION_MASS = {240: 2.0, 246: 13.0, 258: 17.0, 264: 24.0, 270: 26.0,
               324: 43.0, 342: 41.0, 366: 64.0, 372: 67.0, 432: 52.0, 492: 24.0}

def molecular_dopple(T, vturb, mass):
    thermal = np.sqrt(2.0 * KB * T / (mass * AMU)) / C_CMS
    return np.sqrt(thermal ** 2 + (vturb / C_CMS) ** 2)

def voigt(v, a, h0, h1, h2):
    v = np.asarray(v, dtype=np.float64); a = np.asarray(a, dtype=np.float64)
    av = np.abs(v)
    iv = np.clip((av * 200.0 + 0.5).astype(np.int64), 0, h0.size - 1)
    H0 = h0[iv]; H1 = h1[iv]; H2 = h2[iv]
    small = (H2 * a + H1) * a + H0
    with np.errstate(divide="ignore", invalid="ignore"):
        small = np.where(av > 10.0, 0.5642 * a / (v * v), small)
    aa = a * a; vv = v * v
    u = (aa + vv) * 1.4142
    far = a * 0.79788 / u
    aau = aa / u; vvu = vv / u; uu = u * u
    far_full = ((((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu)
                 + 3.0 * vv - aa) / uu + 1.0) * far
    far = np.where(a <= 100.0, far_full, far)
    h1c = H1 + H0 * 1.12838
    h2c = H2 + h1c * 1.12838 - H0
    h3c = (1.0 - H2) * 0.37613 - h1c * 0.66667 * vv + h2c * 1.12838
    h4c = (3.0 * h3c - h1c) * 0.37613 + H0 * 0.66667 * vv * vv
    pa = (((h4c * a + h3c) * a + h2c) * a + h1c) * a + H0
    pb = ((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032
    mid = pa * pb
    use_far = (a > 1.4) | ((a + av) > 3.2)
    out = np.where(a < 0.2, small, np.where(use_far, far, mid))
    return out

def accumulate_depth(buf, cont_row, wavelength, ci, mol_wl, xnfdop, dop_val,
                     cgf, elo, gr, gs, gw, xne, txnxn, hckt, h0, h1, h2):
    n_wl = buf.size
    clamped = np.clip(ci, 0, n_wl - 1)
    kapmin = CUTOFF * cont_row[clamped]
    kappa0_pre = cgf * xnfdop
    boltz = np.exp(-elo * hckt)
    kappa0 = kappa0_pre * boltz
    adamp_raw = (gr + gs * xne + gw * txnxn) / dop_val
    keep = (xnfdop > 0.0) & (dop_val > 0.0) & (mol_wl > 0.0) \
        & (kappa0_pre >= kapmin) & (kappa0 > 0.0) & (kappa0 >= kapmin) \
        & (adamp_raw >= 0.0)
    if not np.any(keep):
        return
    idx = np.nonzero(keep)[0]
    ci = ci[idx]; clamped = clamped[idx]; kapmin = kapmin[idx]
    kappa0 = kappa0[idx]; mol_wl = mol_wl[idx]; dop_val = dop_val[idx]
    adamp = np.maximum(adamp_raw[idx], 1e-12)
    vc = np.where(adamp < 0.2, 1.0 - 1.128 * adamp, voigt(0.0, adamp, h0, h1, h2))
    kapcen = kappa0 * vc
    in_grid = (ci >= 0) & (ci < n_wl)
    np.add.at(buf, ci[in_grid], kapcen[in_grid])
    resolu = np.empty_like(dop_val)
    fwd = clamped < n_wl - 1
    bwd = (~fwd) & (clamped > 0)
    resolu[fwd] = 1.0 / (wavelength[clamped[fwd] + 1] / wavelength[clamped[fwd]] - 1.0)
    resolu[bwd] = 1.0 / (wavelength[clamped[bwd]] / wavelength[clamped[bwd] - 1] - 1.0)
    resolu[~(fwd | bwd)] = 300000.0
    dopple = dop_val
    dr = dopple * resolu
    n10dop = np.minimum((10.0 * dr).astype(np.int64), 1_000_000)
    max_n10 = int(n10dop.max()) if n10dop.size else 0
    prof_n10 = np.zeros_like(kappa0)
    early = np.zeros(kappa0.shape, dtype=bool)
    alive = np.ones(kappa0.shape, dtype=bool)
    is_small = adamp < 0.2
    tabstep = np.where(dr > 0.0, 200.0 / dr, 200.0)
    dvoigt = np.where(dr > 0.0, 1.0 / dr, 1e-6)
    for ns in range(1, max_n10 + 1):
        active = alive & (ns <= n10dop)
        if not np.any(active):
            break
        tabi = 0.5 + ns * tabstep
        it = np.clip(tabi.astype(np.int64), 0, h0.size - 1)
        pval_small = kappa0 * (h0[it] + adamp * h1[it])
        pval_big = kappa0 * voigt(ns * dvoigt, adamp, h0, h1, h2)
        pval = np.where(is_small, pval_small, pval_big)
        ir = ci + ns; ib = ci - ns
        okr = active & (ir >= 0) & (ir < n_wl)
        okb = active & (ib >= 0) & (ib < n_wl)
        np.add.at(buf, ir[okr], pval[okr])
        np.add.at(buf, ib[okb], pval[okb])
        below = active & (pval < kapmin)
        early |= below
        prof_n10 = np.where(active & (ns == n10dop), pval, prof_n10)
        alive &= ~below
    do_far = (~early) & (n10dop > 0) & (prof_n10 > 0.0)
    if np.any(do_far):
        x_far = prof_n10 * n10dop.astype(np.float64) ** 2
        maxstep = np.zeros(kappa0.shape, dtype=np.int64)
        pos = do_far & (x_far > 0.0) & (kapmin > 0.0)
        maxstep[pos] = np.minimum((np.sqrt(x_far[pos] / kapmin[pos]) + 1.0).astype(np.int64), 1_000_000)
        zero_k = do_far & (x_far > 0.0) & (kapmin == 0.0)
        maxstep[zero_k] = 1_000_000
        far_max = int(maxstep.max()) if maxstep.size else 0
        far_alive = do_far.copy()
        for ns in range(1, far_max + 1):
            active = far_alive & (ns > n10dop) & (ns <= maxstep)
            if not np.any(active):
                if ns > maxstep.max():
                    break
                continue
            pval = x_far / (float(ns) * float(ns))
            ir = ci + ns; ib = ci - ns
            on_r = (ir >= 0) & (ir < n_wl)
            on_b = (ib >= 0) & (ib < n_wl)
            okr = active & on_r
            okb = active & on_b
            np.add.at(buf, ir[okr], pval[okr])
            np.add.at(buf, ib[okb], pval[okb])
            kill = active & ~(on_r | on_b)
            far_alive &= ~kill

def compute_mol_opacity(npz, dt, m, L4):
    wavelength = dt["wavelength"].astype(np.float64)
    cont = (dt["continuum_absorption"] + dt["continuum_scattering"]).astype(np.float64)
    T = npz["temperature"].astype(np.float64)
    rho = npz["mass_density"].astype(np.float64)
    xne = npz["electron_density"].astype(np.float64)
    hckt = npz["hckt"].astype(np.float64)
    vturb = npz["turbulent_velocity"].astype(np.float64)
    txnxn = (npz["xnf_h"] + 0.42 * npz["xnf_he1"] + 0.85 * npz["xnf_h2"]) \
        * (T / 10000.0) ** 0.3
    h0 = L4["h0tab"].astype(np.float64); h1 = L4["h1tab"].astype(np.float64); h2 = L4["h2tab"].astype(np.float64)
    pop = np.array(npz["population_per_ion"], dtype=np.float64)
    dop = np.array(npz["doppler_per_ion"], dtype=np.float64)
    for nelion, mass in NELION_MASS.items():
        elem = nelion // 6 - 1
        dop[:, 5, elem] = molecular_dopple(T, vturb, mass)
    nbuff = m["nbuff"].astype(np.int64)
    nelion = m["nelion"].astype(np.int64)
    eidx = nelion // 6 - 1
    cgf = m["cgf"].astype(np.float32).astype(np.float64)
    elo = m["elo_cm"].astype(np.float32).astype(np.float64)
    gr = m["gamma_rad"].astype(np.float32).astype(np.float64)
    gs = m["gamma_stark"].astype(np.float32).astype(np.float64)
    gw = m["gamma_vdw"].astype(np.float32).astype(np.float64)
    ratiolg = float(m["ratiolg"]); ixwlbeg = int(m["ixwlbeg"])
    ci0 = (nbuff - 1)
    mol_wl = np.exp((nbuff.astype(np.float64) - 1 + ixwlbeg) * ratiolg).astype(np.float32).astype(np.float64)
    n_depths = T.size; n_wl = wavelength.size
    mol_asynth = np.zeros((n_depths, n_wl), dtype=np.float64)
    for di in range(n_depths):
        if rho[di] <= 0.0:
            continue
        dop_val = dop[di, 5, eidx]
        pop_val = pop[di, 5, eidx]
        with np.errstate(divide="ignore", invalid="ignore"):
            xnfdop = np.where((dop_val > 0.0) & (pop_val > 0.0),
                              pop_val / (rho[di] * dop_val), 0.0)
        accumulate_depth(mol_asynth[di], cont[di], wavelength, ci0.copy(), mol_wl,
                         xnfdop, dop_val, cgf, elo, gr, gs, gw,
                         xne[di], txnxn[di], hckt[di], h0, h1, h2)
    freq = C_NM / wavelength
    hkt = H_PLANCK / (KB * np.maximum(T, 1.0))
    stim = 1.0 - np.exp(-freq[None, :] * hkt[:, None])
    mol_asynth *= stim
    return mol_asynth
'''

_L12_SPEC = """Port the TiO MOLECULAR BAND-OPACITY accumulation to fully vectorized torch/MPS. The numpy
twin `compute_mol_opacity` loops over the 80 atmosphere depths (Python `for di in range(n_depths)`)
and, per depth, runs `accumulate_depth`: it gates ~1.17M molecular lines, deposits each surviving
line's CENTRE opacity, then marches its NEAR wing (steps 1..n10dop, tabulated Voigt) and FAR wing
(steps n10dop+1.., the 1/n^2 Lorentz tail) symmetrically (red+blue), scatter-ADDING every step into
the depth's [n_w] opacity buffer with `np.add.at`. The whole grid is [80 depths, 9136 wavelengths].
Finally a stimulated-emission factor STIM = 1 - exp(-h nu / kT) multiplies the result.

This is the same SCATTER-ADD hot path as Lecture 5, now for molecular lines — BORROW + REDUCE kgpu's
validated molecular accumulator (`molecular.py`: `_accumulate_chunk` / `_near_wing` / `_far_wing` /
`_scatter_add_flat`, the depth-batched [D,L] cutoff mask + flattened pair deposit + `index_put_`
scatter, and the L4 Harris Voigt). Vectorize over the (depth, line) and (depth, line, wing-step)
axes; `np.add.at` becomes `index_put_(accumulate=True)` (the scatter optimum — NO custom Metal kernel,
NO per-line/per-depth Python loop). Reserve a `for` ONLY for the wing-STEP march (a fixed shrinking
ns=1,2,3,... offset sweep, exactly as kgpu's `_near_wing`/`_far_wing` do in vectorized OFF_CHUNK
blocks): each step must be a fully vectorized [D,L]-or-[pairs] tensor op, never a Python loop over
lines or depths. The center deposit and the per-line gate are pure broadcast/mask. Reproduce every
numpy constant and threshold bit-for-bit: CUTOFF=1e-3, the gate `keep` mask, the per-line resolving
power R (forward/back/300000 fallback from the grid spacing), n10dop = min(10*dr, 1e6), the small-
damping table path `kappa0*(h0[it]+adamp*h1[it])` at `it = clip(0.5 + ns*tabstep)` vs the full Voigt
path `kappa0*voigt(ns*dvoigt, adamp)`, the per-line early-cutoff (`pval < kapmin` kills the line) and
the far-wing IRREVERSIBLE break (the FIRST step with NEITHER red nor blue end on the grid kills the
line for good). DEVICE/DTYPE picked once at module top; fp32 on MPS / fp64 on CPU; float floor ~1e-6.

PRECISION NOTE: this is a positive-only scatter-add of ~1e6 line wings per pixel; the only floor is
accumulation order (np.add.at vs index_put_), well within ~1e-6 on the opacity-bearing pixels."""

_L12_CONTRACT = """Your module MUST define, importable as `port`:
  - port.DEVICE, port.DTYPE  (device + working dtype picked once at module top)
  - port.molecular_dopple(T, vturb, mass) -> tensor/array [D]  (vectorized molecular Doppler width)
  - port.mol_band_opacity(npz, dt, m, L4) -> tensor [D, n_w]  (the full molecular ASYNTH WITH stim)
        The four args are the loaded npz mappings (dict-like; index with [] and .astype works via
        np.asarray — treat them as numpy arrays you cast onto DEVICE/DTYPE inside):
          npz: m3500g50.npz   (temperature, mass_density, electron_density, hckt, turbulent_velocity,
                               xnf_h, xnf_he1, xnf_h2, population_per_ion[D,6,139], doppler_per_ion[D,6,139])
          dt:  diag_tio.npz   (wavelength[n_w], continuum_absorption[D,n_w], continuum_scattering[D,n_w])
          m:   mol_lines_tio.npz (nbuff[L] int, cgf[L] f32, nelion[L], elo_cm[L], gamma_rad/stark/vdw[L],
                               ratiolg scalar, ixwlbeg scalar)
          L4:  L4.npz         (h0tab/h1tab/h2tab[2001])
        Return [D, n_w] = [80, 9136]. The molecular slot is ion stage 5; element index eidx = nelion//6-1;
        the per-species masses come from NELION_MASS (recompute doppler_per_ion slot 5 like the twin).
        The returned tensor may be on DEVICE; the harness moves it to CPU/fp64.
You MAY define helpers (the Voigt kernel, the wing-march, the scatter)."""

_L12_HARNESS = r'''
import numpy as np, torch, math, pathlib
REFDIR = pathlib.Path("reference")
npz = np.load(REFDIR / "m3500g50.npz")
dt  = np.load(REFDIR / "diag_tio.npz")
da  = np.load(REFDIR / "diag_atomic.npz")
m   = np.load(REFDIR / "mol_lines_tio.npz")
L4  = np.load(REFDIR / "L4.npz")

def to64(x):
    if torch.is_tensor(x): return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, float)
''' + _L12_NUMPY + r'''

# --- numpy reference: full molecular band opacity [80, 9136] ---
# The pure-numpy twin's far-wing scalar march over ~1.17M lines is SLOW (~170s); cache it to disk
# on the first run so retries LOAD it (the twin is the oracle; this only avoids recomputing it).
_cache = pathlib.Path("_pipeline/_ports/_l12_twin_ref.npy")
if _cache.exists():
    mol_ref_twin = np.load(_cache)
else:
    mol_ref_twin = compute_mol_opacity(npz, dt, m, L4)
    _cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(_cache, mol_ref_twin)
# the production reference (diag_tio - diag_atomic); the twin reproduces this to ~1e-11
mol_ref_prod = (dt["line_opacity"] - da["line_opacity"]).astype(np.float64)

# --- torch port over the whole grid ---
mol_torch = to64(port.mol_band_opacity(npz, dt, m, L4))

# parity vs the NUMPY TWIN (the oracle); restrict to opacity-bearing pixels
big = mol_ref_twin > 1e-30
if big.sum() == 0:
    dev = 1e9
else:
    dev = float(np.max(np.abs(mol_torch[big] - mol_ref_twin[big]) / np.abs(mol_ref_twin[big])))
# sanity vs production (informational)
bigp = mol_ref_prod != 0.0
devp = float(np.max(np.abs(mol_torch[bigp] - mol_ref_prod[bigp]) / np.abs(mol_ref_prod[bigp]))) if bigp.sum() else 9e9
print(f"mol_band_dev_vs_twin={dev:.3e}  vs_prod={devp:.3e}  "
      f"(nonzero px={int(big.sum())})  twin_max={mol_ref_twin.max():.3e} torch_max={mol_torch.max():.3e} "
      f"device={port.DEVICE}")
print(f"PARITY {dev:.6e}")

# --- timing: the full-grid molecular accumulation is the hot path ---
import time as _t
Pb = port.mol_band_opacity(npz, dt, m, L4)
if torch.is_tensor(Pb) and Pb.device.type=="mps": torch.mps.synchronize()
best = 1e9
for _ in range(2):
    t0=_t.perf_counter()
    Pb = port.mol_band_opacity(npz, dt, m, L4)
    if torch.is_tensor(Pb) and Pb.device.type=="mps": torch.mps.synchronize()
    best=min(best, _t.perf_counter()-t0)
print(f"TIME {best*1e3:.6f}")
'''


# ============================================================================
#  LECTURE 13 — coupled molecular equilibrium (the NMOLEC Newton solve)
# ============================================================================
_L13_NUMPY = r'''
# --- L13 numpy twin: the coupled NMOLEC Newton equilibrium solve (verify_nmolec.py) ---
import math as _math
MAXMOL = 200; MAXEQ = 30; MAXLOC = 3 * MAXMOL
KBOLTZ_EV = 8.617333262e-5

def readmol(molecules_path):
    code_mol = np.zeros(MAXMOL, dtype=np.float64)
    equil = np.zeros((7, MAXMOL), dtype=np.float64)
    locj = np.zeros(MAXMOL + 1, dtype=np.int32)
    kcomps = np.zeros(MAXLOC, dtype=np.int32)
    idequa = np.zeros(MAXEQ, dtype=np.int32)
    ifequa = np.zeros(102, dtype=np.int32)
    xcode = np.array([1e14, 1e12, 1e10, 1e8, 1e6, 1e4, 1e2, 1e0], dtype=np.float64)
    kloc = 0; locj[0] = 0; nummol = 0
    for raw in pathlib.Path(molecules_path).read_text().splitlines():
        line = raw.rstrip("\n\r"); stripped = line.strip()
        if (not stripped or stripped.startswith("C") or stripped.startswith("c")
                or stripped.startswith("#")):
            continue
        c_str = line[0:min(18, len(line))].strip()
        if not c_str: continue
        try: c = float(c_str)
        except ValueError: continue
        cols = [(18, 25), (25, 36), (36, 47), (47, 58), (58, 69), (69, 80), (80, 91)]
        ee = [0.0] * 7
        for i, (a, b) in enumerate(cols):
            if len(line) >= b:
                s = line[a:b].strip()
                if s: ee[i] = float(s)
        if c == 0.0 or abs(c) < 1e-12: continue
        ii = 0
        for i in range(8):
            if c >= xcode[i]: ii = i; break
        x = c
        for i in range(ii, 8):
            id_elem = int(x / xcode[i]); x = x - float(id_elem) * xcode[i]
            if id_elem == 0: id_elem = 100
            ifequa[id_elem] = 1; kcomps[kloc] = id_elem; kloc += 1
        ion = int(x * 100.0 + 0.5)
        if ion >= 1:
            ifequa[100] = 1; ifequa[101] = 1
            for _ in range(ion): kcomps[kloc] = 101; kloc += 1
        locj[nummol + 1] = kloc; code_mol[nummol] = c
        for i in range(7): equil[i, nummol] = ee[i]
        nummol += 1
    nloc = kloc; iequa = 1
    for i in range(1, 101):
        if ifequa[i] == 1: iequa += 1; ifequa[i] = iequa; idequa[iequa - 1] = i
    nequa = iequa; ifequa[101] = nequa + 1
    for k in range(nloc): kcomps[k] = ifequa[kcomps[k]] - 1
    return nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc

def compute_equilj_polynomial(j, T, tkev, tlog, nummol, code_mol, equil, locj):
    eqj = np.zeros(nummol, dtype=np.float64)
    for jmol in range(nummol):
        if equil[0, jmol] == 0.0: continue
        ncomp = locj[jmol + 1] - locj[jmol]
        code_int = int(code_mol[jmol])
        ion = int((np.float64(code_mol[jmol]) - np.float64(code_int)) * 100.0 + 0.5)
        if T[j] > 10000.0: continue
        if abs(code_mol[jmol] - 101.0) < 1e-9:
            ea = (4.478 / tkev[j] - 46.4584
                  + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14
                     + (1.06206e-18 - 3.08720e-23 * T[j]) * T[j]) * T[j]) * T[j])
                     * T[j]) * T[j] - 1.5 * tlog[j])
            eqj[jmol] = np.exp(ea); continue
        poly = (np.float64(equil[0, jmol]) / tkev[j] - equil[1, jmol]
                + (equil[2, jmol] + (-equil[3, jmol] + (equil[4, jmol]
                   + (-equil[5, jmol] + equil[6, jmol] * T[j]) * T[j]) * T[j])
                   * T[j]) * T[j])
        tlog_term = -1.5 * np.float64(ncomp - ion - ion - 1) * tlog[j]
        eqj[jmol] = np.exp(np.float64(poly + tlog_term))
    return eqj

def solvit(a2d, n, b):
    a = np.asarray(a2d, dtype=np.float64, order="F")
    a_vec = np.reshape(a, a.size, order="F")
    a_work = np.zeros(n * n + 1, dtype=np.float64); a_work[1:] = a_vec
    b_work = np.zeros(n + 1, dtype=np.float64); b_work[1:] = np.asarray(b, dtype=np.float64)
    ipivot = np.zeros(n + 1, dtype=np.int32)
    for _ in range(1, n + 1):
        amax = 0.0; irow = 1; icolum = 1
        for row in range(1, n + 1):
            if ipivot[row] == 1: continue
            jk = row - n
            for col in range(1, n + 1):
                jk = jk + n
                if ipivot[col] == 1: continue
                aa = abs(a_work[jk])
                if aa > amax: amax = aa; irow = row; icolum = col
        ipivot[icolum] += 1
        if irow != icolum:
            irl = irow - n; icl = icolum - n
            for _ in range(1, n + 1):
                irl += n; swap = a_work[irl]; icl += n
                a_work[irl] = a_work[icl]; a_work[icl] = swap
            b_work[irow], b_work[icolum] = b_work[icolum], b_work[irow]
        pivot_idx = icolum * n + icolum - n; pivot = a_work[pivot_idx]
        a_work[pivot_idx] = 1.0; icl = icolum - n
        for _ in range(1, n + 1): icl += n; a_work[icl] = a_work[icl] / pivot
        b_work[icolum] = b_work[icolum] / pivot
        l1ic = icolum * n - n
        for l1 in range(1, n + 1):
            l1ic += 1
            if l1 == icolum: continue
            t = a_work[l1ic]; a_work[l1ic] = 0.0
            if t == 0.0: continue
            l1l = l1 - n; icl = icolum - n
            for _ in range(1, n + 1):
                l1l += n; icl += n
                a_work[l1l] = a_work[l1l] - a_work[icl] * t
            b_work[l1] = b_work[l1] - b_work[icolum] * t
    return b_work[1:]

def _stable_subtract(a, b):
    a = np.float64(a); b = np.float64(b)
    if not (np.isfinite(a) and np.isfinite(b)): return a - b
    tiny = np.finfo(np.float64).tiny
    if abs(a) > tiny: return np.float64(a * (1.0 - b / a))
    return np.float64(a - b)

def _ratio_pp(num, den):
    num = np.float64(num); den = np.float64(den)
    if den == 0.0: return 0.0 if num == 0.0 else np.inf
    if num == 0.0: return 0.0
    mant_n, exp_n = np.frexp(num); mant_d, exp_d = np.frexp(den)
    mant = mant_n / mant_d; exp = int(exp_n) - int(exp_d)
    mant, adj = np.frexp(mant); exp += int(adj)
    try: return float(abs(np.ldexp(mant, exp)))
    except (OverflowError, OSError): return float(np.inf)

_HAS_FMA = hasattr(_math, "fma")
def _two_sum(a, b):
    s = a + b; ap = s - b; bp = s - ap
    return s, (a - ap) + (b - bp)
def _two_product_fma(a, b):
    p = a * b
    if _HAS_FMA:
        try: e = _math.fma(a, b, -p)
        except OverflowError: e = 0.0
    else:
        factor = 134217729.0
        ah = a * factor; ah = ah - (ah - a); al = a - ah
        bh = b * factor; bh = bh - (bh - b); bl = b - bh
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e
def _accurate_element_residual(xn_k, xab_k, xn0):
    if xn0 == 0.0 or not np.isfinite(xn0): return xn_k - xab_k * xn0
    if not np.isfinite(xn_k) or not np.isfinite(xab_k): return xn_k - xab_k * xn0
    prod_hi, prod_lo = _two_product_fma(xab_k, xn0)
    if not np.isfinite(prod_hi): return xn_k - xab_k * xn0
    diff_hi, diff_lo = _two_sum(xn_k, -prod_hi)
    if diff_hi == 0.0 and diff_lo == 0.0: return 0.0
    result = diff_hi + diff_lo
    if prod_hi != 0.0:
        rel = abs(result) / abs(prod_hi)
        if rel < 1e-14:
            if result == 0.0:
                ratio = xn_k / xn0; sign = 1.0 if ratio >= xab_k else -1.0
            else: sign = 1.0 if result > 0 else -1.0
            return sign * abs(prod_hi) * 1e-15
    return result

def nmolec_solve(T, gas_pressure, electron_density, xabund,
                 nummol, code_mol, equil, locj, kcomps, idequa, nequa, equilj_ion):
    n_layers = T.shape[0]
    tkev = T * KBOLTZ_EV; tk = T * 1.380649e-16; tlog = np.log(T)
    nequa1 = nequa + 1; neqneq = nequa * nequa
    electron_idx = nequa - 1 if idequa[nequa - 1] == 100 else None
    xab = np.zeros(MAXEQ, dtype=np.float64)
    for k in range(1, nequa):
        id_elem = idequa[k]
        if id_elem < 100: xab[k] = max(xabund[id_elem - 1], 1e-20)
    if idequa[nequa - 1] == 100: xab[nequa - 1] = 0.0
    xnatom_out = np.zeros(n_layers, dtype=np.float64)
    xnmol_out = np.zeros((n_layers, MAXMOL), dtype=np.float64)
    xnz_out = np.zeros((n_layers, MAXEQ), dtype=np.float64)
    electron_out = electron_density.copy()
    xn = np.zeros(MAXEQ, dtype=np.float64)
    xnz_prev = np.zeros(MAXEQ, dtype=np.float64)
    xne_computed = np.zeros(n_layers, dtype=np.float64)
    for j in range(n_layers):
        xntot = gas_pressure[j] / tk[j]
        if j == 0:
            xn[0] = xntot / 2.0; base_x = xn[0] / 10.0
            for k in range(1, nequa): xn[k] = base_x * xab[k]
            if electron_idx is not None: xn[electron_idx] = base_x
            xne_computed[j] = base_x; electron_out[j] = base_x
        else:
            ratio = gas_pressure[j] / gas_pressure[j - 1]
            for k in range(nequa): xn[k] = xnz_prev[k] * ratio
            xne_scaled = xne_computed[j - 1] * ratio
            xne_computed[j] = xne_scaled; electron_out[j] = xne_scaled
        xnz_prev[:nequa] = xn[:nequa]
        equilj = compute_equilj_polynomial(j, T, tkev, tlog, nummol, code_mol, equil, locj)
        ion_mask = (equil[0, :nummol] == 0.0)
        equilj[ion_mask] = equilj_ion[j, :nummol][ion_mask]
        eqold = np.zeros(nequa, dtype=np.float64)
        for _iteration in range(200):
            deq = np.zeros(neqneq, dtype=np.float64); eq = np.zeros(nequa, dtype=np.float64)
            use_numba_setup = not (j == 0 and _iteration < 5)
            eq[0] = -xntot; kk = 0; xn0 = xn[0]
            for k in range(1, nequa):
                eq[0] = eq[0] + xn[k]; deq[k * nequa] = 1.0
                if use_numba_setup: eq[k] = xn[k] - xab[k] * xn0
                else: eq[k] = _accurate_element_residual(xn[k], xab[k], xn0)
                kk += nequa1; deq[kk] = 1.0; deq[k] = -xab[k]
            if electron_idx is not None and idequa[electron_idx] >= 100:
                eq[electron_idx] = -xn[electron_idx]; deq[nequa * nequa - 1] = -1.0
            eq_comp = np.zeros(nequa, dtype=np.float64)
            for jmol in range(nummol):
                ncomp = int(locj[jmol + 1] - locj[jmol])
                if ncomp <= 1: continue
                locj1 = int(locj[jmol]); locj2 = int(locj[jmol + 1] - 1)
                ev = equilj[jmol]
                if not np.isfinite(ev): continue
                term = ev
                for lock in range(locj1, locj2 + 1):
                    k_raw = int(kcomps[lock])
                    if k_raw >= nequa: term = term / xn[nequa - 1]
                    else: term = term * xn[k_raw]
                y = term - eq_comp[0]; t = eq[0] + y; eq_comp[0] = (t - eq[0]) - y; eq[0] = t
                for lock in range(locj1, locj2 + 1):
                    k_raw = int(kcomps[lock]); k_idx = nequa - 1 if k_raw == nequa else k_raw
                    xn_val = xn[k_idx]
                    if not np.isfinite(xn_val) or xn_val == 0.0: continue
                    d = (-term / xn_val) if k_raw == nequa else (term / xn_val)
                    if not np.isfinite(d): continue
                    y_k = term - eq_comp[k_idx]; t_k = eq[k_idx] + y_k
                    eq_comp[k_idx] = (t_k - eq[k_idx]) - y_k; eq[k_idx] = t_k
                    nequak = nequa * k_idx; deq[nequak] = deq[nequak] + d
                    for locm in range(locj1, locj2 + 1):
                        m_raw = int(kcomps[locm]); m_idx = nequa - 1 if m_raw == nequa else m_raw
                        mk = m_idx + nequak; deq[mk] = deq[mk] + d
                last_raw = int(kcomps[locj2])
                if last_raw == nequa - 1 and idequa[nequa - 1] == 100:
                    for lock in range(locj1, locj2 + 1):
                        kc_raw = int(kcomps[lock]); kc_idx = nequa - 1 if kc_raw >= nequa else kc_raw
                        xn_val = xn[kc_idx]
                        if not np.isfinite(xn_val) or xn_val == 0.0: continue
                        term_corr = term
                        if not np.isfinite(term_corr): continue
                        d_corr = term_corr / xn_val
                        if not np.isfinite(d_corr): continue
                        if kc_idx == nequa - 1: eq[kc_idx] = eq[kc_idx] - term_corr - term_corr
                        delta = -d_corr - d_corr
                        for locm in range(locj1, locj2 + 1):
                            mc_raw = int(kcomps[locm]); mc_idx = nequa - 1 if mc_raw >= nequa else mc_raw
                            if mc_idx != nequa - 1: continue
                            mk = mc_idx + nequa * kc_idx; deq[mk] = deq[mk] + delta
            deq_2d = deq[:neqneq].reshape(nequa, nequa, order="F").copy()
            delta_xn = solvit(deq_2d, nequa, eq.copy()); eq[:] = delta_xn
            iferr = 0; scale = 100.0
            for k in range(nequa):
                ratio_k = _ratio_pp(eq[k], xn[k])
                if ratio_k > 0.001: iferr = 1
                sign_change = (eqold[k] > 0 and eq[k] < 0) or (eqold[k] < 0 and eq[k] > 0)
                if sign_change:
                    ek = np.float64(eq[k]); eq[k] = ek * 0.69 if np.isfinite(ek) else ek
                xneq = _stable_subtract(np.float64(xn[k]), np.float64(eq[k])); xn100 = xn[k] / 100.0
                if xneq < xn100:
                    xn[k] = xn[k] / scale
                    sc2 = (eqold[k] > 0 and eq[k] < 0) or (eqold[k] < 0 and eq[k] > 0)
                    if sc2: scale = np.sqrt(scale)
                else: xn[k] = xneq
                eqold[k] = eq[k]
            if iferr == 0: break
        xnatom_out[j] = xn[0]
        for k in range(nequa): xnz_out[j, k] = xn[k]
        xnz_prev[:nequa] = xn[:nequa]
        if idequa[nequa - 1] == 100:
            electron_out[j] = xn[nequa - 1]; xne_computed[j] = electron_out[j]
        for jmol in range(nummol):
            xnmol_out[j, jmol] = equilj[jmol]
            locj1 = int(locj[jmol]); locj2 = int(locj[jmol + 1] - 1)
            for lock in range(locj1, locj2 + 1):
                k = int(kcomps[lock])
                if k == nequa:
                    k = nequa - 1; xnmol_out[j, jmol] = xnmol_out[j, jmol] / xn[k]
                else: xnmol_out[j, jmol] = xnmol_out[j, jmol] * xn[k]
    return xnatom_out, xnmol_out, xnz_out, electron_out
'''

_L13_SPEC = """Port the COUPLED MOLECULAR-EQUILIBRIUM SOLVER (NMOLEC) — a Newton-Raphson fixed-point
solve — to torch/MPS. The numpy twin `nmolec_solve` finds, at every atmosphere depth, the equilibrium
number densities of nequa(=23) unknowns XN = (XNATOM, the neutral-atom densities, n_e) such that the
coupled residual eq(XN)=0 holds: one total-particle equation, one number-conservation equation per
element, one charge-balance equation, with every molecule's mass-action term K_f * prod XN[k] folded in.

STRUCTURE TO REPRODUCE (this is the crux — read carefully):
- There is an OUTER loop over the 80 depths (`for j in range(n_layers)`). This loop is INTRINSIC and
  must stay: depth j's Newton SEED is the CONVERGED depth j-1 scaled by the pressure ratio (a physical
  warm-start chain), so depths cannot be solved as one independent batched system. Keep the depth loop.
  Depth 0 seeds from XNATOM=xntot/2, neutrals=(XNATOM/10)*xab, n_e=XNATOM/10.
- INSIDE each depth is the NEWTON ITERATION (up to 200 iters). EACH ITERATION must be a fully
  VECTORIZED whole-vector tensor op over the nequa unknowns — NOT a per-equation / per-molecule Python
  loop. This is what you port from kgpu: kgpu replaces the Fortran hand-built DEQ Jacobian + SOLVIT
  complete-pivoting elimination with (a) a single vectorized residual `_residual(xn, log_equilj, xab,
  xntot, struct)` that builds ALL molecular mass-action terms in LOG space and exp's ONCE (so xn^6 ~ 1e84
  never overflows), assembled from a precomputed branchless `MolStructure` (count/inv_e/neg_ion/active
  incidence built ONCE from kcomps/locj — NO per-molecule loop in the residual); (b) an AUTODIFF Jacobian
  via `torch.func.jacrev(resid_one)`; (c) a column-scaled `torch.linalg.solve` (`_newton_step`: scale by
  diag(|xn|) to equilibrate the ~25-dex Jacobian, the modern stand-in for SOLVIT pivoting); (d) the
  Fortran damping (convergence test ratio_k=|delta|/|xn| > tol=1e-3; sign-flip relaxation *0.69; the
  floor-and-rescale /100). BORROW + REDUCE kgpu/nmolec.py's `MolStructure`, `_residual`, `_newton_step`,
  `_molecular_densities`, `compute_equilj_polynomial` verbatim in spirit. After convergence, assemble the
  molecular densities n(M)=K_f*prod xn[k]/n_e^inv_e BATCHED over depth (kgpu vmaps `_molecular_densities`).

K_f / log space: K_f(T) (the molecular formation constants, EQUILJ) is computed ONCE in fp64 numpy on the
host (polynomial molecules recomputed from `compute_equilj_polynomial`; ion molecules supplied as
equilj_ion), spans ~1e25..1e-78 (beyond fp32 range), so it is carried as LOG(K_f) (~+/-180, safely in
range), with a -700 sentinel for zero/inactive. Reproduce the polynomial K_f bit-for-bit (the lecture's
physics): the D0/tkev term, the 5th-order T polynomial, the -1.5*(ncomp-2*ion-1)*log(T) translational
term, the H- (code 101) H2-dissociation branch, the T>1e4 K cutoff.

PRECISION (teach it): the converged equilibrium is a FIXED POINT, but fp32 conditioning matters — the raw
Jacobian spans ~25 orders of magnitude, fatal to a plain fp32 LU; the column-scaling (solve for the
FRACTIONAL step delta=xn*d) is what makes the fp32/MPS solve well-posed. For the PARITY reference run on
CPU in fp64 (DTYPE=float64 on CPU) so the autodiff-Newton fixed point matches the numpy SOLVIT fixed point
tightly; the lecture documents that on MPS the same solve runs in fp32 to the ~1e-6 floor. The autodiff
Newton + column-scaled solve reaches the SAME equilibrium as the numpy hand-DEQ+SOLVIT (it is the same
Newton, just autodiff'd); it will NOT be bit-identical (different elimination order, no Kahan/double-double),
so the float floor for this lecture is ~1e-6 on the converged densities (looser than the numpy-vs-Fortran
1e-13). DEVICE/DTYPE picked once; on CPU use fp64.

DO NOT batch the depth loop into one solve (breaks the warm-start chain → divergence). DO vectorize each
Newton iteration over the nequa unknowns (NO Python loop over equations or molecules inside the iteration —
the residual and Jacobian are whole-tensor ops; the only Python loops are over depths and over iterations)."""

_L13_CONTRACT = """Your module MUST define, importable as `port`:
  - port.DEVICE, port.DTYPE  (device + working dtype picked once; CPU -> float64, MPS/CUDA -> float32)
  - port.readmol(molecules_path) -> (nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc)
        Faithful parse of molecules.dat (you MAY transcribe the numpy twin's readmol on the host; it is a
        once-per-run host parse, not the hot path — a host loop here is fine, it is NOT vectorizable text I/O).
  - port.compute_equilj_polynomial(T, code_mol, equil, locj, nummol) -> equilj[n_layers, nummol]
        The polynomial K_f(T) for ALL depths at once (ion molecules left 0.0). Vectorized over depths; a
        `for jmol in range(nummol)` over the FIXED molecule set is acceptable ONLY if each iteration is a
        vectorized [n_layers] tensor/array op (kgpu does exactly this). fp64 numpy on host is fine here.
  - port.nmolec_solve(T, gas_pressure, electron_density, xabund, nummol, code_mol, equil, locj, kcomps,
                      idequa, nequa, equilj_ion) -> (xnatom[D], xnmol[D,MAXMOL=200], xnz[D,MAXEQ=30], electron[D])
        The full coupled solve. Same call signature + same return tuple as the numpy twin. Inside: the depth
        loop + warm start (intrinsic), each Newton iteration fully vectorized over the nequa unknowns
        (autodiff Jacobian + column-scaled torch.linalg.solve), the Fortran damping, then the batched
        molecular-density assembly. Returned tensors may be on DEVICE; the harness moves them to CPU/fp64.
You MAY (should) define helpers: a MolStructure incidence dataclass, a vectorized `_residual`, a
`_newton_step`, a `_molecular_densities`, `_safe_log`. NO per-equation/per-molecule Python loop inside the
Newton iteration."""

_L13_HARNESS = r'''
import numpy as np, torch, math, pathlib
REFDIR = pathlib.Path("reference")
inp = np.load(REFDIR / "nmolec_inputs.npz")
gt  = np.load(REFDIR / "nmolec_groundtruth.npz")
MOLPATH = REFDIR / "molecules.dat"

def to64(x):
    if torch.is_tensor(x): return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, float)
''' + _L13_NUMPY + r'''

T = np.asarray(inp["temperature"], float)
gas_pressure = np.asarray(inp["gas_pressure"], float)
electron_density = np.asarray(inp["electron_density"], float)
xabund = np.asarray(inp["xabund"], float)
equilj_ion = np.asarray(inp["equilj_ion"], float)

# molecules.dat parsed by BOTH (the twin's readmol and the port's) — same structures
nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc = readmol(MOLPATH)

# --- numpy reference solve (the oracle) ---
ref_xnatom, ref_xnmol, ref_xnz, ref_electron = nmolec_solve(
    T, gas_pressure, electron_density, xabund,
    nummol, code_mol, equil, locj, kcomps, idequa, nequa, equilj_ion)

# --- torch port solve ---
p_nummol, p_code_mol, p_equil, p_locj, p_kcomps, p_idequa, p_nequa, p_nloc = port.readmol(MOLPATH)
o_xnatom, o_xnmol, o_xnz, o_electron = port.nmolec_solve(
    T, gas_pressure, electron_density, xabund,
    p_nummol, p_code_mol, p_equil, p_locj, p_kcomps, p_idequa, p_nequa, equilj_ion)
o_xnatom = to64(o_xnatom); o_xnmol = to64(o_xnmol); o_xnz = to64(o_xnz); o_electron = to64(o_electron)

def maxrel(a, b, floor):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > floor)
    if not mask.any(): return 0.0
    return float(np.max(np.abs(a[mask] - b[mask]) / np.abs(b[mask])))

# parity vs the NUMPY TWIN on the physical outputs
d_atom = maxrel(o_xnatom, ref_xnatom, 0.0)
d_ne   = maxrel(o_electron, ref_electron, 0.0)
d_xnz  = maxrel(o_xnz[:, :nequa], ref_xnz[:, :nequa], 1e-300)
d_mol  = maxrel(o_xnmol[:, :nummol], ref_xnmol[:, :nummol], 1e-300)
tio = int(np.argmin(np.abs(code_mol[:nummol] - 822.0)))
d_tio = maxrel(o_xnmol[:, tio], ref_xnmol[:, tio], 0.0)
dev = max(d_atom, d_ne, d_xnz, d_mol, d_tio)

# also vs ground truth (informational sanity)
gt_xnmol = np.asarray(gt["xnmol"], float)[:, :nummol]
d_gt = maxrel(o_xnmol[:, :nummol], gt_xnmol, 1e-300)
print(f"xnatom={d_atom:.3e} ne={d_ne:.3e} xnz={d_xnz:.3e} xnmol={d_mol:.3e} "
      f"TiO={d_tio:.3e}  vs_groundtruth={d_gt:.3e}  device={port.DEVICE} dtype={port.DTYPE}")
print(f"PARITY {dev:.6e}")

# --- timing: the full 80-depth coupled solve ---
import time as _t
_ = port.nmolec_solve(T, gas_pressure, electron_density, xabund,
                      p_nummol, p_code_mol, p_equil, p_locj, p_kcomps, p_idequa, p_nequa, equilj_ion)
if port.DEVICE.type == "mps": torch.mps.synchronize()
best = 1e9
for _ in range(2):
    t0=_t.perf_counter()
    _ = port.nmolec_solve(T, gas_pressure, electron_density, xabund,
                          p_nummol, p_code_mol, p_equil, p_locj, p_kcomps, p_idequa, p_nequa, equilj_ion)
    if port.DEVICE.type == "mps": torch.mps.synchronize()
    best=min(best, _t.perf_counter()-t0)
print(f"TIME {best*1e3:.6f}")
'''



# ============================================================================
#  LECTURE 13 (Part 2) — molecular CONTINUOUS opacity (CHOP / OHOP / H2-CIA)
# ============================================================================
_L13B_NUMPY = r'''
# --- L13 Part-2 numpy twin: molecular continuous opacity (verify_mol_continuum.py) ---
C_LIGHT_CM = 2.99792458e10; C_LIGHT_NM = 2.99792458e17
H_PLANCK = 6.62607015e-27; K_BOLTZ = 1.380649e-16
KBOLTZ_EV = 8.6171e-5; LN10 = 2.30258509299405

def mol_continuum_numpy(inp):
    temp = np.asarray(inp["temperature"], float); rho = np.asarray(inp["mass_density"], float)
    xnfph1 = np.asarray(inp["xnfph1"], float); bhyd1 = np.asarray(inp["bhyd1"], float)
    xnfhe1 = np.asarray(inp["xnfhe1"], float)
    xnfpch = np.asarray(inp["xnfpch"], float); xnfpoh = np.asarray(inp["xnfpoh"], float)
    freq = np.asarray(inp["frequency_hz"], float)
    CH_PARTITION = np.asarray(inp["CH_PARTITION"], float); OH_PARTITION = np.asarray(inp["OH_PARTITION"], float)
    CH_CROSSSECT = np.asarray(inp["CH_CROSSSECT"], float); OH_CROSSSECT = np.asarray(inp["OH_CROSSSECT"], float)
    H2H2 = np.asarray(inp["H2_COLL_H2H2"], float); H2HE = np.asarray(inp["H2_COLL_H2HE"], float)
    n_layers = temp.size; nfreq = freq.size
    tkev = KBOLTZ_EV * temp; tlog = np.log(temp)
    stim = 1.0 - np.exp(-H_PLANCK * freq[None, :] / (K_BOLTZ * temp[:, None]))

    def chop_xsect_times_U(freq_scalar):
        out = np.zeros(n_layers); wn = freq_scalar / C_LIGHT_CM; ev = wn / 8065.479
        n = int(ev * 10)
        if n < 20 or n >= 105: return out
        en = n * 0.1; idx = n - 2
        if idx < 0 or idx >= 104: return out
        cross = CH_CROSSSECT[idx] + (CH_CROSSSECT[idx + 1] - CH_CROSSSECT[idx]) * (ev - en) / 0.1
        for j in range(n_layers):
            tj = temp[j]
            if tj >= 9000.0: continue
            it_p = max(0, min(int((tj - 1000.0) / 200.0), 39)); tn_p = it_p * 200.0 + 1000.0
            part = CH_PARTITION[it_p] + (CH_PARTITION[it_p + 1] - CH_PARTITION[it_p]) * (tj - tn_p) / 200.0
            it_c = max(0, min(int((tj - 2000.0) / 500.0), 13)); tn_c = it_c * 500.0 + 2000.0
            log_x = cross[it_c] + (cross[it_c + 1] - cross[it_c]) * (tj - tn_c) / 500.0
            out[j] = np.exp(log_x * LN10) * part
        return out

    def ohop_xsect_times_U(freq_scalar):
        out = np.zeros(n_layers); wn = freq_scalar / C_LIGHT_CM; ev = wn / 8065.479
        n = int(ev * 10) - 20
        if n <= 0 or n >= 130: return out
        en = n * 0.1 + 2.0; idx = n - 1
        if idx < 0 or idx >= 129: return out
        cross = OH_CROSSSECT[idx] + (OH_CROSSSECT[idx + 1] - OH_CROSSSECT[idx]) * (ev - en) / 0.1
        for j in range(n_layers):
            tj = temp[j]
            if tj >= 9000.0: continue
            it_p = max(0, min(int((tj - 1000.0) / 200.0), 39)); tn_p = it_p * 200.0 + 1000.0
            part = OH_PARTITION[it_p] + (OH_PARTITION[it_p + 1] - OH_PARTITION[it_p]) * (tj - tn_p) / 200.0
            it_c = max(0, min(int((tj - 2000.0) / 500.0), 13)); tn_c = it_c * 500.0 + 2000.0
            log_x = cross[it_c] + (cross[it_c + 1] - cross[it_c]) * (tj - tn_c) / 500.0
            out[j] = np.exp(log_x * LN10) * part
        return out

    _poly_t = (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14
               + (1.06206e-18 - 3.08720e-23 * temp) * temp) * temp) * temp) * temp) * temp
    _exp_term = np.clip(4.478 / tkev - 46.4584 + _poly_t - 1.5 * tlog, -100, 100)
    XNH2 = (xnfph1 * 2.0 * bhyd1) ** 2 * np.exp(_exp_term)

    def h2cia_opacity(freq_scalar, stim_col):
        out = np.zeros(n_layers); wn = freq_scalar / C_LIGHT_CM
        if wn > 20000.0: return out
        nu = min(79, int(wn / 250.0)); delnu = (wn - 250.0 * nu) / 250.0
        idx1 = min(nu, 80); idx2 = min(nu + 1, 80)
        h2h2_nu = H2H2[idx1] * delnu + H2H2[idx2] * (1.0 - delnu)
        h2he_nu = H2HE[idx1] * delnu + H2HE[idx2] * (1.0 - delnu)
        for j in range(n_layers):
            tj = temp[j]; it = max(1, min(6, int(tj / 1000.0)))
            delt = max(0.0, min(1.0, (tj - 1000.0 * it) / 1000.0))
            xh2h2 = h2h2_nu[it - 1] * delt + h2h2_nu[it] * (1.0 - delt)
            xh2he = h2he_nu[it - 1] * delt + h2he_nu[it] * (1.0 - delt)
            out[j] = (10.0 ** xh2he * xnfhe1[j] + 10.0 ** xh2h2 * XNH2[j]) * XNH2[j] / rho[j] * stim_col[j]
        return out

    chop = np.zeros((n_layers, nfreq)); ohop = np.zeros((n_layers, nfreq)); h2cia = np.zeros((n_layers, nfreq))
    for j in range(nfreq):
        f = freq[j]; stim_j = stim[:, j]
        chop[:, j] = chop_xsect_times_U(f) * xnfpch / rho * stim_j
        ohop[:, j] = ohop_xsect_times_U(f) * xnfpoh / rho * stim_j
        h2cia[:, j] = h2cia_opacity(f, stim_j)
    return chop, ohop, h2cia, chop + ohop + h2cia
'''

_L13B_SPEC = """Port the MOLECULAR CONTINUOUS OPACITY (CHOP CH-photodissociation, OHOP OH-photodissociation,
H2-CIA Borysow collision-induced absorption) to fully vectorized torch/MPS. The numpy twin
`mol_continuum_numpy` loops over the 600 frequencies (`for j in range(nfreq)`) and, inside each per-
species function, loops over the 80 depths (`for j in range(n_layers)`) doing scalar table interpolation.
The GPU port evaluates the WHOLE [80 depths, 600 freqs] grid at once, branchlessly.

The physics is pure TABLE INTERPOLATION + masking (NO Newton, NO scatter):
- CHOP: per-frequency energy index n=int(ev*10) (ev = waveno/8065.479), gate 20<=n<105; linear-interp the
  [106,15] cross-section table in energy (0.1 eV bins) -> per-T-node row; then per depth linear-interp that
  row in T (500 K bins from 2000, it_c clamped 0..13) AND the [41] partition table in T (200 K bins from
  1000, it_p clamped 0..39); opacity = exp(LN10*log_xsect) * part * xnfpch / rho * stim, gated T<9000.
- OHOP: identical but energy index n=int(ev*10)-20, en = n*0.1+2.0, table [130,15] starts at 2.1 eV, idx=n-1,
  gate 0<n<130; * xnfpoh.
- H2-CIA: XNH2 = (xnfph1*2*bhyd1)^2 * exp(clip(4.478/tkev - 46.4584 + P(T) - 1.5*lnT, -100,100)) (the H2-
  dissociation polynomial P(T) — frequency-independent, compute once over depth); per frequency gate
  waveno<20000, wavenumber cell nu=min(79,int(wn/250)), delnu=(wn-250*nu)/250, interp Borysow [81,7] H2H2/H2HE
  tables in wavenumber (idx1=min(nu,80), idx2=min(nu+1,80)) then per depth in T (it=clamp(int(T/1000),1,6),
  delt=clip((T-1000*it)/1000,0,1)); opacity = (10^xh2he*xnfhe1 + 10^xh2h2*XNH2) * XNH2 / rho * stim.

VECTORIZE the per-frequency loop AND the per-depth loop into [D,F] tensor ops. The integer table indices
(energy cell, wavenumber cell, the two T cells) become clamped integer index TENSORS computed on the host in
fp64 (the bracket math is Fortran-exact int truncation — keep it in fp64 numpy then gather), and the table
lookups become `torch.index_select`/fancy gather. The energy/wavenumber validity gate becomes a boolean mask
× arithmetic. Reproduce every constant + bin width + clamp bound bit-for-bit. DEVICE/DTYPE once; the float
floor is ~1e-6 (CPU fp64 -> machine precision). Borrow + reduce kgpu/mol_continuum.py's depth-batched
`mol_continuum` / `_band_continuum` / `_temp_lerp_rows` (the bracket-gather + T-lerp pattern)."""

_L13B_CONTRACT = """Your module MUST define, importable as `port`:
  - port.DEVICE, port.DTYPE  (device + working dtype picked once; CPU -> float64, MPS/CUDA -> float32)
  - port.mol_continuum(inp) -> (chop[D,F], ohop[D,F], h2cia[D,F], total[D,F])
        inp is the loaded mol_continuum_inputs.npz mapping (dict-like; np.asarray its fields):
          temperature[D], mass_density[D], xnfph1[D], bhyd1[D], xnfhe1[D], xnfpch[D], xnfpoh[D],
          frequency_hz[F], CH_PARTITION[41], OH_PARTITION[41], CH_CROSSSECT[106,15], OH_CROSSSECT[130,15],
          H2_COLL_H2H2[81,7], H2_COLL_H2HE[81,7]
        Returns the four [D,F] = [80,600] opacity arrays. Returned tensors may be on DEVICE; the harness
        moves them to CPU/fp64. You MAY define helpers (the energy/wavenumber bracket math on the host in
        fp64, the T-lerp gather)."""

_L13B_HARNESS = r'''
import numpy as np, torch, math, pathlib
REFDIR = pathlib.Path("reference")
inp = np.load(REFDIR / "mol_continuum_inputs.npz")
truth = np.load(REFDIR / "mol_continuum_truth.npz")

def to64(x):
    if torch.is_tensor(x): return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, float)
''' + _L13B_NUMPY + r'''

# numpy twin (the oracle) and the production truth (the twin reproduces it to ~machine)
chop_n, ohop_n, h2_n, tot_n = mol_continuum_numpy(inp)
chop_p = np.asarray(truth["chop"], float); ohop_p = np.asarray(truth["ohop"], float)
h2_p = np.asarray(truth["h2cia"], float); tot_p = np.asarray(truth["mol_total"], float)

# torch port
chop_t, ohop_t, h2_t, tot_t = port.mol_continuum(inp)
chop_t = to64(chop_t); ohop_t = to64(ohop_t); h2_t = to64(h2_t); tot_t = to64(tot_t)

def maxrel(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = np.abs(b) > 0.0
    if not mask.any(): return 0.0
    return float(np.max(np.abs(a[mask] - b[mask]) / np.abs(b[mask])))

d_chop = maxrel(chop_t, chop_n); d_ohop = maxrel(ohop_t, ohop_n)
d_h2 = maxrel(h2_t, h2_n); d_tot = maxrel(tot_t, tot_n)
dev = max(d_chop, d_ohop, d_h2, d_tot)
# informational vs production truth
d_totp = maxrel(tot_t, tot_p)
print(f"chop={d_chop:.3e} ohop={d_ohop:.3e} h2cia={d_h2:.3e} total={d_tot:.3e}  "
      f"vs_prod_total={d_totp:.3e}  device={port.DEVICE} dtype={port.DTYPE}")
print(f"PARITY {dev:.6e}")

import time as _t
_ = port.mol_continuum(inp)
if port.DEVICE.type == "mps": torch.mps.synchronize()
best = 1e9
for _ in range(5):
    t0=_t.perf_counter()
    _ = port.mol_continuum(inp)
    if port.DEVICE.type == "mps": torch.mps.synchronize()
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
    "lecture12": PortJob(
        name="lecture12",
        numpy_source=_L12_NUMPY,
        spec=_L12_SPEC,
        contract=_L12_CONTRACT,
        harness=_L12_HARNESS,
        float_floor=1e-6,
        preamble="The molecular (TiO) band-opacity SCATTER-ADD over the full [80, 9136] grid, ~1.17M lines. "
                 "reference/ holds m3500g50.npz (temperature, mass_density, electron_density, hckt, "
                 "turbulent_velocity, xnf_h/xnf_he1/xnf_h2, population_per_ion[80,6,139], "
                 "doppler_per_ion[80,6,139]), diag_tio.npz (wavelength[9136], continuum_absorption/scattering "
                 "[80,9136], line_opacity[80,9136]), diag_atomic.npz (line_opacity[80,9136] molecules-off; the "
                 "pure molecular reference is diag_tio.line_opacity - diag_atomic.line_opacity), "
                 "mol_lines_tio.npz (nbuff[L] int, cgf/elo_cm/gamma_rad/gamma_stark/gamma_vdw[L] f32, "
                 "nelion[L], ratiolg/ixwlbeg scalars), L4.npz (h0tab/h1tab/h2tab[2001]). The molecular slot is "
                 "ion stage 5, element index eidx=nelion//6-1; NELION_MASS gives per-species mass. The fp32-MPS "
                 "float floor is ~1e-6 on the opacity-bearing pixels; the scatter index_put_(accumulate=True) "
                 "is add-order non-deterministic at ~1e-7 (within floor). This IS the L5 scatter-add shape for "
                 "molecules — borrow kgpu's molecular accumulator, do NOT regenerate.",
        # BORROW kgpu's BATCHED molecular accumulation: the [D,L] cutoff mask + flattened survive pairs, the
        # near/far wing OFF_CHUNK sweeps, the index_put_ scatter, and the L4 Harris Voigt (shared with L4/L5).
        kgpu_borrow=[("molecular.py", 463, 494),   # species_xnfdop_dopple (the [D,S] population/Doppler tables)
                     ("molecular.py", 508, 560),   # accumulate_molecular (the top-level [D,n_w] kernel)
                     ("molecular.py", 563, 579),   # _grid_resolu_np (per-pixel resolving power, host f64)
                     ("molecular.py", 582, 683),   # _accumulate_chunk (cutoff mask + center + survive pairs)
                     ("molecular.py", 686, 712),   # _near_wing_pval (table form vs full Voigt, branchless)
                     ("molecular.py", 715, 790),   # _near_wing (fused reach+deposit OFF_CHUNK sweep)
                     ("molecular.py", 793, 860),   # _far_wing (1/n^2 tail, irreversible off-grid break)
                     ("molecular.py", 866, 878)],  # _scatter_add_flat (index_put_ accumulate=True)
    ),
    "lecture13": PortJob(
        name="lecture13",
        numpy_source=_L13_NUMPY,
        spec=_L13_SPEC,
        contract=_L13_CONTRACT,
        harness=_L13_HARNESS,
        # The autodiff-Newton + column-scaled solve reaches the SAME equilibrium fixed point as the numpy
        # hand-DEQ + SOLVIT, but NOT bit-identically (different elimination order, no Kahan/double-double).
        # On CPU-fp64 the two fixed points agree tightly; the floor is the Newton convergence + reordering,
        # set to ~1e-6 (the lecture documents the MPS-fp32 path runs the same solve to the same floor).
        float_floor=1e-6,
        preamble="The COUPLED molecular-equilibrium NMOLEC Newton solve. reference/ holds nmolec_inputs.npz "
                 "(temperature[80], gas_pressure[80], electron_density[80], xabund[99], equilj_ion[80,190]), "
                 "nmolec_groundtruth.npz (xnmol[80,200], xnatom[80], xnz[80,30], electron[80] — comparison "
                 "only), and molecules.dat (the dissociation table, parsed by readmol). nequa=23 unknowns "
                 "(XNATOM, neutral atoms, n_e); nummol=190 molecules. THE DEPTH LOOP IS INTRINSIC (warm-start "
                 "chain: depth j seeded from converged depth j-1 * pressure ratio) — keep it; vectorize the "
                 "Newton ITERATION over the nequa unknowns (autodiff jacrev Jacobian + column-scaled "
                 "torch.linalg.solve, the Fortran damping). K_f spans 1e25..1e-78 -> carry as LOG(K_f) in "
                 "fp32 range; precompute the polynomial K_f in fp64 on the host. RUN THE PARITY ON CPU fp64 so "
                 "the autodiff-Newton fixed point matches the numpy SOLVIT fixed point; the lecture documents "
                 "the MPS-fp32 budget. The 1e-3 tol is the STOP rule; the converged densities are the parity "
                 "target (~1e-6 floor vs the numpy twin).",
        # BORROW kgpu's GPU-resident NMOLEC: the K_f polynomial, the branchless MolStructure incidence, the
        # vectorized log-space residual, the autodiff Newton step (column-scaled solve), and the density assembly.
        kgpu_borrow=[("nmolec.py", 198, 239),   # compute_equilj_polynomial (K_f(T), all depths)
                     ("nmolec.py", 246, 299),   # MolStructure (branchless count/inv_e/neg_ion/active incidence)
                     ("nmolec.py", 302, 311),   # _safe_log (dtype-safe log floor)
                     ("nmolec.py", 317, 376),   # _residual (the vectorized log-space coupled residual)
                     ("nmolec.py", 382, 412),   # _newton_step (column-scaled solve) + _molecular_densities
                     ("nmolec.py", 415, 584)],  # nmolec_solve (depth loop + warm start + autodiff Newton)
    ),
    "lecture13b": PortJob(
        name="lecture13b",
        numpy_source=_L13B_NUMPY,
        spec=_L13B_SPEC,
        contract=_L13B_CONTRACT,
        harness=_L13B_HARNESS,
        float_floor=1e-6,
        preamble="L13 Part 2: the molecular CONTINUOUS opacity (CHOP/OHOP/H2-CIA), depth-batched table "
                 "interpolation over the full [80, 600] grid. reference/ holds mol_continuum_inputs.npz "
                 "(temperature[80], mass_density[80], xnfph1/bhyd1/xnfhe1[80], xnfpch/xnfpoh[80], "
                 "frequency_hz[600], CH_PARTITION/OH_PARTITION[41], CH_CROSSSECT[106,15], OH_CROSSSECT[130,15], "
                 "H2_COLL_H2H2/H2_COLL_H2HE[81,7]) and mol_continuum_truth.npz (chop/ohop/h2cia/mol_total[80,600] "
                 "— comparison only). NO Newton, NO scatter — pure tabulated interpolation + masking. The "
                 "per-frequency and per-depth loops both vectorize into [D,F] tensor ops; the integer bracket "
                 "indices (energy/wavenumber/T cells) are fp64 host int math, then gathered. float floor ~1e-6.",
        # BORROW kgpu's depth-batched continuum: the bracket-gather + T-lerp pattern (CHOP/OHOP/H2-CIA).
        kgpu_borrow=[("mol_continuum.py", 338, 457),   # mol_continuum (the depth-batched CHOP+OHOP+H2-CIA kernel)
                     ("mol_continuum.py", 460, 479),   # _band_continuum (T-lerp cross-section + mask + stim)
                     ("mol_continuum.py", 482, 494)],  # _temp_lerp_rows (the Borysow T-lerp gather)
    ),
}


# ── PART C — the lint report across all ported lectures ──────────────────────
def lint_report(numbers: list[int] | None = None) -> dict:
    """Scan ALL ported GPU lecture builders for un-justified python loops / gratuitous numpy in the
    SHIPPED code (numpy in a tagged comparison-reference cell is excused), and for completeness vs
    the numpy twin. A standalone audit — no API calls, no execution. Returns {n: report}."""
    nums = numbers or [2, 3, 4, 5, 6, 12, 13, 14, 15, 16]
    out = {}
    print("\n=== LINT + COMPLETENESS REPORT (shipped GPU lecture builders) ===")
    for n in nums:
        gpu = _gpu_builder(n)
        if not gpu.exists():
            print(f"\nL{n}: (no build_lecture{n}_gpu.py)")
            continue
        code = gpu.read_text()
        hits = lint_builder(code)
        np_path = _numpy_builder(n)
        gs = parse_structure(gpu)
        ns = parse_structure(np_path) if np_path.exists() else None
        ratio = (gs["n_cells"] / ns["n_cells"]) if ns and ns["n_cells"] else None
        closers_missing = ([k for k in CLOSING_KEYS
                            if ns and ns["has_closers"].get(k) and not gs["has_closers"].get(k)]
                           if ns else [])
        status = "CLEAN" if not hits else f"{len(hits)} flag(s)"
        rtxt = f"{ratio:.0%} of twin" if ratio is not None else "no twin"
        print(f"\nL{n}: lint {status}   cells {gs['n_cells']}"
              + (f"/{ns['n_cells']} ({rtxt})" if ns else "")
              + (f"   MISSING closers: {closers_missing}" if closers_missing else ""))
        for h in hits[:40]:
            print(h)
        if len(hits) > 40:
            print(f"  ... (+{len(hits) - 40} more)")
        out[n] = dict(lint=hits, cells=gs["n_cells"],
                      numpy_cells=(ns["n_cells"] if ns else None),
                      ratio=ratio, missing_closers=closers_missing)
    # summary
    loopy = [n for n, r in out.items() if r["lint"]]
    short = [n for n, r in out.items() if r["ratio"] is not None and r["ratio"] < COMPLETENESS_FRACTION]
    noclose = [n for n, r in out.items() if r["missing_closers"]]
    print("\n--- SUMMARY ---")
    print(f"  lectures with un-justified loops / gratuitous numpy : {loopy or 'none'}")
    print(f"  lectures < {COMPLETENESS_FRACTION:.0%} of twin cell count          : {short or 'none'}")
    print(f"  lectures missing closing sections                  : {noclose or 'none'}")
    return out


def main():
    ap = argparse.ArgumentParser(description="GPU-edition port worker (external API generates; we validate).")
    ap.add_argument("--job", choices=sorted(JOBS), help="kernel port job (parity-gated single routine)")
    ap.add_argument("--fill", choices=sorted(FILLS), help="full-lecture catch-and-fill (append missing sections)")
    ap.add_argument("--lint-report", action="store_true", help="Part C: lint + completeness audit, all lectures")
    ap.add_argument("--model", default="gpt55", choices=sorted(MODELS), help="external code-gen model")
    ap.add_argument("--max-tries", type=int, default=6, help="max API generate->fix iterations")
    ap.add_argument("--squeeze", type=int, default=0, metavar="N",
                    help="after parity, run N optimization-squeeze rounds (bible §9; parity-gated + timed)")
    ap.add_argument("--no-exec", action="store_true", help="(fill) skip notebook execution (structure-only)")
    args = ap.parse_args()

    if args.lint_report:
        lint_report()
        sys.exit(0)

    if args.fill:
        res = run_fill_job(FILLS[args.fill], model=args.model, max_tries=args.max_tries,
                           execute=not args.no_exec)
        print("\n=== FILL RESULT ===")
        print(f"  job={res['name']} filled={res.get('filled')} complete={res.get('complete')} "
              f"exec_ok={res.get('exec_ok')} cells={res.get('gpu_cells')}/{res.get('numpy_cells')} "
              f"lint_flags={len(res.get('lint') or [])}")
        sys.exit(0 if (res.get("complete") and res.get("exec_ok", True)) else 1)

    if not args.job:
        ap.error("one of --job / --fill / --lint-report is required")
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
