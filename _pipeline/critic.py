#!/usr/bin/env python
"""Cross-check each lecture with two independent LLM critics — GPT-5.5 (via LiteLLM) and
Gemini-3.5-flash (via google.genai) — for ACCURACY, ACCESSIBILITY, and PEDAGOGY.

Writes one report per lecture+model to _pipeline/critic_reports/. It does NOT edit any
lecture — a human (or a follow-up pass) reads the reports and applies prose-only fixes.

HARD RULE baked into the prompt: the critic may NOT suggest changing any computed number,
constant, formula, or code logic — those are verified to machine precision against pykurucz.
Only prose / clarity / ordering / pedagogy.

Usage:  python _pipeline/critic.py 1 2 3 ...   (default: all built lectures)
Keys (GOOGLE_API_KEY, LITELLM_API_KEY) are read from ~/.env.
"""
import os, sys, json
from pathlib import Path
import nbformat

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "_pipeline" / "critic_reports"; OUT.mkdir(exist_ok=True)
for ln in (Path.home()/".env").read_text().splitlines():
    if "=" in ln and not ln.strip().startswith("#"):
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

GPT_MODEL, GEM_MODEL = "gpt-5.5-2026-04-24", "gemini-3.1-pro-preview"

PROMPT = """You are an expert reviewer of a graduate-level computational astrophysics textbook. \
This lecture rebuilds part of a stellar-spectrum synthesis pipeline from scratch in plain NumPy \
and benchmarks it, to machine precision, against the reference code pykurucz (Kurucz's ATLAS12 + \
SYNTHE). Review it on three axes:

1. SCIENTIFIC ACCURACY (MOST IMPORTANT) — scrutinise every physical statement, equation, definition, \
unit, and explanatory CLAIM in the descriptions for correctness. Flag anything wrong, imprecise, \
overstated, or misleading, however small — a mis-stated dependence, a wrong sign or limit, a term used \
loosely, a cause/effect reversed, a regime of validity omitted. Be a skeptical domain expert: if a \
sentence would make a stellar-atmospheres specialist wince, call it out with the precise correction.
2. ACCESSIBILITY — is it clear and well-paced for a first/second-year astronomy graduate student? \
Where might a reader get lost or want a sentence more?
3. PEDAGOGY — does it build logically, motivate each step, define terms before use, and flow well? \
Any gaps, jumps, or things referenced before they are introduced?

HARD CONSTRAINTS — read carefully:
- Do NOT suggest changing any computed NUMBER, physical CONSTANT, FORMULA, or CODE logic. Those are \
verified to machine precision against pykurucz and are correct by construction. If a number looks \
surprising, assume it is right and (at most) suggest a clarifying sentence.
- Restrict every suggestion to PROSE, EXPLANATION, CLARITY, ORDERING, NOTATION, and PEDAGOGICAL FLOW.

Be specific and concise. Output:
- STRENGTHS: 2-4 bullets.
- ISSUES: a numbered list, each tagged [HIGH]/[MED]/[LOW], with the location (quote a few words) and a concrete suggested fix.
- VERDICT: one line — is it at the quality bar of a polished published textbook chapter? what is the single most valuable change?

LECTURE (markdown + code cells, in order):
---
{body}
---"""

def lecture_text(n):
    nb = nbformat.read(BOOK/"content"/f"Lecture{n}.ipynb", as_version=4)
    parts = []
    for c in nb.cells:
        if c.cell_type == "markdown": parts.append(c.source)
        elif c.cell_type == "code":  parts.append("```python\n"+c.source+"\n```")
    return "\n\n".join(parts)

def ask_gpt(body):
    from openai import OpenAI
    c = OpenAI(base_url="https://litellm.cloud.osu.edu", api_key=os.environ["LITELLM_API_KEY"])
    r = c.chat.completions.create(model=GPT_MODEL, max_tokens=16000,
        messages=[{"role":"user","content":PROMPT.format(body=body)}])
    return r.choices[0].message.content

def ask_gemini(body):
    from google import genai
    g = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return g.models.generate_content(model=GEM_MODEL, contents=PROMPT.format(body=body)).text

if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(range(1, 10))
    for n in ns:
        body = lecture_text(n)
        for tag, fn in (("gpt55", ask_gpt), ("gemini", ask_gemini)):
            try:
                rep = fn(body)
                (OUT/f"Lecture{n}_{tag}.md").write_text(rep or "(empty)")
                head = (rep or "").strip().replace("\n", " ")[:140]
                print(f"L{n} [{tag}]  ok  {len(rep or '')} chars  | {head}")
            except Exception as e:
                print(f"L{n} [{tag}]  FAIL  {type(e).__name__}: {str(e)[:120]}")
