#!/usr/bin/env python
"""Generate the Lecture 13 (capstone) reference data — 4 stars across the HR diagram.

Run ONCE by the author against the READ-ONLY pykurucz tree.  For each star it:
  1. predicts a model atmosphere from (Teff, logg, [M/H], vturb) with the kurucz-a1
     emulator warm-start  ->  capstone_<slug>.atm  (the atmosphere is GIVEN, exactly
     as Lectures 3-12 took the solar / M-dwarf atmosphere as given);
  2. converts that .atm to .npz (runs the full EOS + molecular equilibrium);
  3. runs the production SYNTHE CLI over the star's representative window, with
     --diagnostics, to capture the synthesis diagnostics (continuum opacity, line
     opacity, line source/scatter, emergent + continuum flux) and the .spec file.

The capstone verifier and Lecture 13 notebook load ONLY the shipped
reference/capstone_<slug>.npz files (atmosphere depth scale + temperature + the
diagnostics) and reproduce the EMERGENT SPECTRUM from scratch with the book's own
JOSH solver (Lecture 8) — never importing pykurucz.  This is the end-to-end proof
at the spectrum level: production opacity + the book's radiative transfer must
reproduce the production spectrum to the documented float floor, on EVERY star.

The four stars and their windows (each chosen to SHOW that star's character):
  * hot dwarf  Teff=9000  logg=4.0   484-488 nm   broad Hbeta Stark wing (L6)
  * Sun        Teff=5777  logg=4.44  500-505 nm   metal-line forest (L4-5)
  * giant      Teff=4500  logg=2.0   516-519 nm   Mg b + low-gravity metals (L4-5,L9)
  * M dwarf    Teff=3500  logg=5.0   705-718 nm   TiO band head (L12)

Deterministic configuration: all thread pools pinned to 1; molecules ON for the
two cool stars (giant, M dwarf), molecular lines OFF for the two warm stars
(hot dwarf, Sun) where they are negligible — matching the canonical solar run.
"""
from __future__ import annotations

import os
# Pin every thread pool to 1 BEFORE any numba-backed import (byte determinism).
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import subprocess
import sys
from pathlib import Path

import numpy as np

PYK = Path("/Users/ysting/pykurucz")
sys.path.insert(0, str(PYK))            # read-only reference code
BOOK = Path(__file__).resolve().parent.parent
REF = BOOK / "reference"
REF.mkdir(exist_ok=True)

GFALL = PYK / "data" / "lines" / "gfallvac.latest"
if not GFALL.exists():
    GFALL = PYK / "lines" / "gfallvac.latest"
MOL_DIR = PYK / "data" / "molecules"

# Each star: (slug, Teff, logg, [M/H], vturb km/s, wl_start, wl_end, resolution, molecules)
STARS = [
    ("hot",    9000.0, 4.0,  0.0, 2.0, 484.0, 488.0, 200_000.0, False),
    ("sun",    5777.0, 4.44, 0.0, 2.0, 500.0, 505.0, 300_000.0, False),
    ("giant",  4500.0, 2.0,  0.0, 2.0, 516.0, 519.0, 300_000.0, True),
    ("mdwarf", 3500.0, 5.0,  0.0, 2.0, 705.0, 718.0, 500_000.0, True),
]

# Keys we keep from the converted .npz (the model atmosphere the spectrum is built on).
ATM_KEYS = ("depth", "temperature", "mass_density", "electron_density", "hckt",
            "turbulent_velocity", "teff", "glog")
# Keys we keep from the SYNTHE --diagnostics npz (the inputs to the JOSH spectrum solve).
DIAG_KEYS = ("wavelength", "continuum_absorption", "continuum_scattering",
             "line_opacity", "line_scattering", "line_source", "slinec",
             "flux_total", "flux_continuum")


def make_atmosphere(slug, teff, logg, mh, vturb):
    """Emulator warm-start .atm + convert to _full.npz (full EOS + molecular eq.)."""
    atm_path = REF / f"capstone_{slug}.atm"
    npz_path = REF / f"capstone_{slug}_full.npz"
    print(f"[atm:{slug}] emulator warm-start Teff={teff} logg={logg} [M/H]={mh} vturb={vturb}")
    from pykurucz import emulator_warmstart_atm
    emulator_warmstart_atm(atm_path, teff=teff, logg=logg, mh=mh, am=0.0, vturb=vturb)

    print(f"[atm:{slug}] convert .atm -> .npz")
    rc = subprocess.run(
        [sys.executable, "-m", "synthe_py.tools.convert_atm_to_npz",
         str(atm_path), str(npz_path)],
        cwd=str(PYK), env={**os.environ, "PYTHONPATH": str(PYK)},
    )
    if rc.returncode != 0:
        raise SystemExit(f"convert_atm_to_npz failed for {slug}")
    return atm_path, npz_path


def run_synthesis(slug, atm_path, npz_path, wl0, wl1, res, molecules):
    """Run the production SYNTHE CLI with --diagnostics; return the diag npz path."""
    diag_arg = REF / f"capstone_{slug}.diag"            # np.savez appends .npz
    diag_path = REF / f"capstone_{slug}.diag.npz"
    spec_path = REF / f"capstone_{slug}.spec"
    cmd = [
        sys.executable, "-m", "synthe_py.cli",
        str(atm_path), str(GFALL),
        "--npz", str(npz_path),
        "--wl-start", str(wl0), "--wl-end", str(wl1),
        "--resolution", str(res),
        "--spec", str(spec_path),
        "--diagnostics", str(diag_arg),
        "--n-workers", "1",
        "--log-level", "WARNING",
    ]
    if molecules:
        cmd += ["--molecules-dir", str(MOL_DIR), "--no-h2o"]
    else:
        cmd += ["--no-molecular-lines"]
    print(f"[synth:{slug}] {wl0}-{wl1} nm  R={res:.0f}  molecules={'ON' if molecules else 'OFF'}")
    rc = subprocess.run(cmd, cwd=str(PYK), env={**os.environ, "PYTHONPATH": str(PYK)})
    if rc.returncode != 0:
        raise SystemExit(f"synthesis failed for {slug}")
    if not diag_path.exists():
        # fall back to whatever name the CLI actually produced
        cand = list(REF.glob(f"capstone_{slug}.diag*"))
        if cand:
            diag_path = cand[0]
        else:
            raise SystemExit(f"diagnostics file not found for {slug}")
    return diag_path, spec_path


def assemble(slug, teff, logg, npz_path, diag_path):
    """Pack the atmosphere subset + the diagnostics into one capstone_<slug>.npz."""
    atm = np.load(str(npz_path))
    diag = np.load(str(diag_path))
    out = {}
    for k in ATM_KEYS:
        if k in atm.files:
            out[f"atm_{k}"] = atm[k]
    for k in DIAG_KEYS:
        out[k] = diag[k]
    out["teff"] = np.float64(teff)
    out["logg"] = np.float64(logg)
    dest = REF / f"capstone_{slug}.npz"
    np.savez_compressed(dest, **out)
    print(f"[pack:{slug}] wrote {dest}  ({dest.stat().st_size/1e6:.2f} MB)  "
          f"line_opacity {out['line_opacity'].shape}")
    return dest


def main():
    for slug, teff, logg, mh, vturb, wl0, wl1, res, mol in STARS:
        atm_path, npz_full = make_atmosphere(slug, teff, logg, mh, vturb)
        diag_path, spec_path = run_synthesis(slug, atm_path, npz_full, wl0, wl1, res, mol)
        assemble(slug, teff, logg, npz_full, diag_path)
        # tidy intermediate full atmosphere npz (keep .atm, .spec, .diag.npz, capstone_<slug>.npz)
        npz_full.unlink(missing_ok=True)
    print("\n[done] capstone reference data written for all 4 stars.")


if __name__ == "__main__":
    main()
