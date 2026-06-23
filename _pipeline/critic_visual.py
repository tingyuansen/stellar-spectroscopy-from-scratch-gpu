#!/usr/bin/env python
"""MULTIMODAL figure critic — cross-check each lecture's FIGURES with two independent
vision LLMs, GPT-5.5 (via LiteLLM) and Gemini-3.1-pro (via google.genai), for SCIENTIFIC
and VISUAL accuracy.

For each lecture it gathers, as raw PNG bytes:
  (a) the conceptual SCHEMATIC(s) under resources/figures/ that the lecture embeds
      (discovered by parsing content/Lecture{n}.html for resources/figures/*.png), and
  (b) the matplotlib OUTPUT PLOTS — the base64 image/png outputs stored in the code
      cells of content/Lecture{n}.ipynb.

Images are capped at MAX_IMAGES per lecture to stay within token limits: every schematic
is ALWAYS included; if there are too many plots they are evenly sub-sampled. The same
image set is sent to BOTH models with one shared accuracy-review prompt.

Writes one report per lecture+model to _pipeline/critic_reports/visual/. It is strictly
READ-ONLY on the lectures and figures — it only reports.

Usage:  python _pipeline/critic_visual.py 1 2 3 ...   (default: all 13 lectures)
Keys (GOOGLE_API_KEY, LITELLM_API_KEY) are read from ~/.env.
"""
import os, sys, re, base64
from pathlib import Path
import nbformat

BOOK = Path(__file__).resolve().parent.parent
FIGDIR = BOOK / "resources" / "figures"
OUT = BOOK / "_pipeline" / "critic_reports" / "visual"
OUT.mkdir(parents=True, exist_ok=True)

for ln in (Path.home() / ".env").read_text().splitlines():
    if "=" in ln and not ln.strip().startswith("#"):
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

GPT_MODEL, GEM_MODEL = "gpt-5.5-2026-04-24", "gemini-3.1-pro-preview"
LITELLM_BASE = "https://litellm.cloud.osu.edu"
MAX_IMAGES = 6  # per lecture; schematics always kept, plots sampled if needed

PROMPT = (
    "You are an expert stellar-atmospheres reviewer checking a figure for SCIENTIFIC and "
    "VISUAL accuracy. Is everything correct: axis labels and units, the direction/trend of "
    "each curve, monotonicity, physical positions (e.g. HR-diagram luminosity ordering — a "
    "red giant must be more luminous than any main-sequence dwarf), spectral-feature shapes "
    "(absorption dips DOWN, emission UP, band heads, line forests), legends matching curves, "
    "nothing mislabeled or reversed? List any figure that is wrong, misleading, or mislabeled, "
    "with the precise problem and fix. If all correct, say so."
)


def schematics_for(n):
    """PNG paths under resources/figures/ that Lecture{n}.html embeds (in order, de-duped)."""
    html = BOOK / "content" / f"Lecture{n}.html"
    if not html.exists():
        return []
    refs = re.findall(r"resources/figures/([A-Za-z0-9_]+\.png)", html.read_text())
    seen, out = set(), []
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        p = FIGDIR / r
        if p.exists():
            out.append(p)
    return out


def plot_pngs(n):
    """Decoded matplotlib PNG outputs from Lecture{n}.ipynb, as (label, bytes)."""
    nb = nbformat.read(BOOK / "content" / f"Lecture{n}.ipynb", as_version=4)
    out = []
    code_i = 0
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        code_i += 1
        for oi, o in enumerate(c.get("outputs", [])):
            b64 = o.get("data", {}).get("image/png")
            if b64:
                try:
                    raw = base64.b64decode(b64)
                except Exception:
                    continue
                out.append((f"cell{code_i}_out{oi}", raw))
    return out


def collect_images(n):
    """Return (images, notes): images is a list of (label, bytes); notes records what was
    included/sampled. Schematics always kept; plots sub-sampled to fit MAX_IMAGES."""
    notes = []
    schem = [(p.name, p.read_bytes()) for p in schematics_for(n)]
    plots = plot_pngs(n)
    notes.append(f"schematics={[s[0] for s in schem]}")
    notes.append(f"plots_found={len(plots)}")

    budget = MAX_IMAGES - len(schem)
    if budget < 0:  # more schematics than the cap (shouldn't happen) — keep all schematics
        budget = 0
    if len(plots) > budget:
        # evenly sub-sample the plots down to `budget`
        if budget == 0:
            kept = []
        else:
            idx = [round(i * (len(plots) - 1) / (budget - 1)) if budget > 1 else 0
                   for i in range(budget)]
            idx = sorted(set(idx))
            kept = [plots[i] for i in idx]
        notes.append(f"plots_sampled={[k[0] for k in kept]} (of {len(plots)})")
    else:
        kept = plots
        notes.append(f"plots_kept=all ({len(plots)})")

    return schem + kept, notes


def ask_gpt(images):
    """images: list of (label, bytes). OpenAI-compatible vision via LiteLLM."""
    from openai import OpenAI
    c = OpenAI(base_url=LITELLM_BASE, api_key=os.environ["LITELLM_API_KEY"])
    content = [{"type": "text", "text": PROMPT}]
    for label, raw in images:
        b64 = base64.b64encode(raw).decode()
        content.append({"type": "text", "text": f"\n[Figure: {label}]"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    r = c.chat.completions.create(model=GPT_MODEL, max_tokens=8000,
                                  messages=[{"role": "user", "content": content}])
    return r.choices[0].message.content


def ask_gemini(images):
    """images: list of (label, bytes). google.genai vision via Part.from_bytes."""
    from google import genai
    from google.genai import types
    g = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    parts = [types.Part.from_text(text=PROMPT)]
    for label, raw in images:
        parts.append(types.Part.from_text(text=f"\n[Figure: {label}]"))
        parts.append(types.Part.from_bytes(data=raw, mime_type="image/png"))
    r = g.models.generate_content(model=GEM_MODEL, contents=parts)
    return r.text


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(range(1, 14))
    for n in ns:
        images, notes = collect_images(n)
        manifest = f"Lecture {n} — {len(images)} images sent\n" + "\n".join(notes)
        (OUT / f"Lecture{n}_manifest.txt").write_text(manifest)
        print(f"L{n}: sending {len(images)} images | {'; '.join(notes)}")
        if not images:
            print(f"L{n}: SKIP — no images found")
            continue
        for tag, fn in (("gpt55", ask_gpt), ("gemini", ask_gemini)):
            try:
                rep = fn(images)
                (OUT / f"Lecture{n}_{tag}.md").write_text(rep or "(empty)")
                head = (rep or "").strip().replace("\n", " ")[:140]
                print(f"L{n} [{tag}]  ok  {len(rep or '')} chars  | {head}")
            except Exception as e:
                print(f"L{n} [{tag}]  FAIL  {type(e).__name__}: {str(e)[:160]}")
                (OUT / f"Lecture{n}_{tag}.ERROR.txt").write_text(
                    f"{type(e).__name__}: {e}")
