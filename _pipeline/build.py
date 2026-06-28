#!/usr/bin/env python
"""Build lectures: execute the notebook, then render it to a content/*.html fragment.

GPU EDITION: the per-lecture notebook is assembled by _pipeline/build_lecture<N>_gpu.py
(the GPU/torch source of truth); this script then EXECUTES it (on MPS/CUDA if present,
else CPU/fp64) and renders the HTML fragment. The notebooks select the device themselves.

  python _pipeline/build_lecture2_gpu.py  # assemble content/Lecture2.ipynb (the GPU PoC)
  python _pipeline/build.py            # build all registered lectures
  python _pipeline/build.py 2          # execute + render lecture 2
  python _pipeline/build.py 2 --no-exec  # re-render only (skip execution)

The lecture registry below mirrors assets/book-data.js (keep them in sync).
"""
import subprocess, sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
AFFIL = "Max Planck Institute for Astronomy &amp; The Ohio State University"

LECTURES = {
    1: dict(slug="Lecture1", title="Overview &amp; a First Model Atmosphere", lecturer="Yuan-Sen Ting", affil=AFFIL),
    2: dict(slug="Lecture2", title="The Equation of State", lecturer="Yuan-Sen Ting", affil=AFFIL),
    3: dict(slug="Lecture3", title="Continuous Opacity", lecturer="Yuan-Sen Ting", affil=AFFIL),
    4: dict(slug="Lecture4", title="Line Opacity I: A Single Line", lecturer="Yuan-Sen Ting", affil=AFFIL),
    5: dict(slug="Lecture5", title="Line Opacity II: The Line List", lecturer="Yuan-Sen Ting", affil=AFFIL),
    6: dict(slug="Lecture6", title="Hydrogen Lines: Stark Broadening", lecturer="Yuan-Sen Ting", affil=AFFIL),
    7: dict(slug="Lecture7", title="Radiative Transfer &amp; the Emergent Spectrum", lecturer="Yuan-Sen Ting", affil=AFFIL),
    8: dict(slug="Lecture8", title="The JOSH Solver: Production Radiative Transfer", lecturer="Yuan-Sen Ting", affil=AFFIL),
    9: dict(slug="Lecture9", title="Hydrostatic Equilibrium &amp; Temperature Structure", lecturer="Yuan-Sen Ting", affil=AFFIL),
    10: dict(slug="Lecture10", title="Radiative Equilibrium &amp; Temperature Correction", lecturer="Yuan-Sen Ting", affil=AFFIL),
    11: dict(slug="Lecture11", title="Convection &amp; the Converged Atmosphere", lecturer="Yuan-Sen Ting", affil=AFFIL),
    12: dict(slug="Lecture12", title="Molecular Equilibrium &amp; Molecular Bands", lecturer="Yuan-Sen Ting", affil=AFFIL),
    13: dict(slug="Lecture13", title="Molecular Chemistry: Coupled Equilibrium &amp; Continuous Opacity", lecturer="Yuan-Sen Ting", affil=AFFIL),
    14: dict(slug="Lecture14", title="A Spectrum from Stellar Parameters, End to End", lecturer="Yuan-Sen Ting", affil=AFFIL),
    15: dict(slug="Lecture15", title="Line Blanketing: the True Model Atmosphere", lecturer="Yuan-Sen Ting", affil=AFFIL),
    16: dict(slug="Lecture16", title="The Full Equation of State: Species Slots &amp; the Convective Heat Capacity", lecturer="Yuan-Sen Ting", affil=AFFIL),
}


def build(n: int, execute: bool = True) -> None:
    m = LECTURES[n]
    nb = BOOK / "content" / f"{m['slug']}.ipynb"
    if not nb.exists():
        raise FileNotFoundError(nb)
    if execute:
        print(f"[exec] {nb.name}")
        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", "--ExecutePreprocessor.timeout=900",
             "--ExecutePreprocessor.kernel_name=python3", str(nb)],
            check=True,
        )
    out = BOOK / "content" / f"{m['slug']}.html"
    print(f"[render] {out.name}")
    subprocess.run(
        ["node", str(BOOK / "_pipeline" / "render_fragment.js"), str(nb), str(out),
         str(n), m["title"], m["lecturer"], m["affil"]],
        check=True,
    )


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(LECTURES)
    no_exec = "--no-exec" in sys.argv
    for n in ns:
        build(n, execute=not no_exec)
