#!/usr/bin/env python
"""Generate the Lecture 12 (molecular equilibrium + molecular bands) reference.

Run ONCE by the author against the READ-ONLY pykurucz tree.  Produces, in
reference/:
  * m3500g50.atm / m3500g50.npz   — the cool M-dwarf model atmosphere (GIVEN).
                                     The npz carries population_per_ion (with the
                                     molecular slot 5 = XNFPMOL) and
                                     doppler_per_ion.
  * diag_tio.npz                   — synthesis diagnostics WITH molecules
                                     (line_opacity incl. molecular, continuum,
                                     wavelength, flux_total / flux_continuum).
  * diag_atomic.npz                — synthesis diagnostics with molecules OFF
                                     (atomic-only line_opacity).  The difference
                                     line_opacity(tio) - line_opacity(atomic) is
                                     the pure molecular component (stim applied),
                                     used to benchmark the band opacity in
                                     isolation to machine precision.
  * mol_lines_tio.npz              — the WINDOWED molecular line arrays for
                                     705-718 nm (nbuff / cgf / nelion / elo_cm /
                                     gamma_rad / gamma_stark / gamma_vdw, plus the
                                     reconstructed wavelength and center index),
                                     from compile_tio_schwenke + compile_molecular_ascii.
  * molecules.dat (verbatim copy)  — the dissociation-equilibrium table.

The lecture notebook + _pipeline/verify_molecules.py load these and NEVER import
pykurucz.

Deterministic configuration (matches the lecture's claims):
  * cool M-dwarf  : Teff=3500, logg=5.0, [M/H]=0, vturb=2 km/s
  * atmosphere    : emulator warm-start (kurucz-a1) -> .atm  (the atmosphere is
                    GIVEN, exactly as Lectures 3-11 took the solar atmosphere;
                    torch is fine HERE in the reference, never in the lecture)
  * window        : 705-718 nm (TiO gamma-system band heads dominate)
  * resolution    : 500000
  * molecules     : data/molecules, --no-h2o (Fortran reference skips H2O)
  * serial        : NUMBA/OMP pinned to 1 thread for byte-determinism
"""
from __future__ import annotations

import os
# Pin every thread pool to 1 BEFORE any numba-backed import (determinism).
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PYK = Path("/Users/ysting/pykurucz")
sys.path.insert(0, str(PYK))  # read-only reference code
BOOK = Path(__file__).resolve().parent.parent
REF = BOOK / "reference"
REF.mkdir(exist_ok=True)

TEFF = 3500.0
LOGG = 5.0
MH = 0.0
VTURB = 2.0
WL_START = 705.0
WL_END = 718.0
RESOLUTION = 500000.0

ATM_PATH = REF / "m3500g50.atm"
NPZ_PATH = REF / "m3500g50.npz"
DIAG_TIO = REF / "diag_tio.npz"
DIAG_ATOMIC = REF / "diag_atomic.npz"
MOL_LINES = REF / "mol_lines_tio.npz"
MOLECULES_DAT_SRC = PYK / "lines" / "molecules.dat"
MOLECULES_DAT_DST = REF / "molecules.dat"

GFALL = PYK / "data" / "lines" / "gfallvac.latest"
if not GFALL.exists():
    GFALL = PYK / "lines" / "gfallvac.latest"
MOL_DIR = PYK / "data" / "molecules"


# --------------------------------------------------------------------------
# 1. Cool M-dwarf model atmosphere (GIVEN) via the kurucz-a1 emulator warm-start.
# --------------------------------------------------------------------------
def make_atmosphere() -> None:
    print(f"[atm] emulator warm-start Teff={TEFF} logg={LOGG} [M/H]={MH} vturb={VTURB}")
    from pykurucz import emulator_warmstart_atm

    emulator_warmstart_atm(
        ATM_PATH,
        teff=TEFF,
        logg=LOGG,
        mh=MH,
        am=0.0,
        vturb=VTURB,
    )
    print(f"[atm] wrote {ATM_PATH}")

    # Convert to NPZ (carries population_per_ion incl. molecular slot 5, doppler).
    print("[atm] convert .atm -> .npz (runs full NMOLEC equilibrium internally)")
    rc = subprocess.run(
        [sys.executable, "-m", "synthe_py.tools.convert_atm_to_npz",
         str(ATM_PATH), str(NPZ_PATH)],
        cwd=str(PYK),
        env={**os.environ, "PYTHONPATH": str(PYK)},
    )
    if rc.returncode != 0:
        raise SystemExit("convert_atm_to_npz failed")
    print(f"[atm] wrote {NPZ_PATH}")


# --------------------------------------------------------------------------
# 2. Synthesis with the production CLI (molecules ON, then OFF for isolation).
# --------------------------------------------------------------------------
def run_synthesis(diag_path: Path, with_molecules: bool, spec_path: Path) -> None:
    cmd = [
        sys.executable, "-m", "synthe_py.cli",
        str(ATM_PATH), str(GFALL),
        "--npz", str(NPZ_PATH),
        "--wl-start", str(WL_START),
        "--wl-end", str(WL_END),
        "--resolution", str(RESOLUTION),
        "--spec", str(spec_path),
        "--diagnostics", str(diag_path),
        "--n-workers", "1",
        "--log-level", "WARNING",
    ]
    if with_molecules:
        cmd += ["--molecules-dir", str(MOL_DIR), "--no-h2o"]
    else:
        cmd += ["--no-molecular-lines"]
    print(f"[synth] molecules={'ON' if with_molecules else 'OFF'} -> {diag_path.name}")
    rc = subprocess.run(cmd, cwd=str(PYK),
                        env={**os.environ, "PYTHONPATH": str(PYK)})
    if rc.returncode != 0:
        raise SystemExit(f"synthesis failed (molecules={with_molecules})")


# --------------------------------------------------------------------------
# 3. Windowed molecular line arrays (TiO Schwenke + ASCII .dat/.asc lists).
#    Compiled EXACTLY as the production engine does (ifvac=1 throughout).
# --------------------------------------------------------------------------
def dump_molecular_lines() -> None:
    from synthe_py.io.lines import molecular_compiler as mc

    mol_dicts = []

    # ASCII .dat/.asc lists (root + vo/ subdir) — exactly what opacity.py globs.
    all_mol_files = []
    for ext in ("*.dat", "*.asc"):
        all_mol_files.extend(sorted(MOL_DIR.glob(ext)))
    for ext in ("*.dat", "*.asc"):
        all_mol_files.extend(sorted((MOL_DIR / "vo").glob(ext)))
    if all_mol_files:
        print(f"[mol] compiling {len(all_mol_files)} ASCII lists")
        mol_dicts.append(mc.compile_molecular_ascii(
            paths=all_mol_files,
            wlbeg=WL_START, wlend=WL_END, resolution=RESOLUTION, ifvac=1))

    # TiO Schwenke binary.
    tio_bin = MOL_DIR / "tio" / "schwenke.bin"
    print(f"[mol] compiling TiO Schwenke {tio_bin}")
    mol_dicts.append(mc.compile_tio_schwenke(
        bin_path=tio_bin,
        wlbeg=WL_START, wlend=WL_END, resolution=RESOLUTION, ifvac=1))

    combined = {}
    keys = mol_dicts[0].keys()
    for k in keys:
        combined[k] = np.concatenate([d[k] for d in mol_dicts])
    n = len(combined["nbuff"])
    print(f"[mol] combined molecular lines in window: {n}")

    # Reconstruct the grid parameters the engine uses for wavelength/center.
    ratio = 1.0 + 1.0 / RESOLUTION
    ratiolg = float(np.log(ratio))
    import math
    ixwlbeg = math.floor(math.log(WL_START) / ratiolg)
    if math.exp(ixwlbeg * ratiolg) < WL_START:
        ixwlbeg += 1

    np.savez(
        MOL_LINES,
        nbuff=combined["nbuff"],
        cgf=combined["cgf"],
        nelion=combined["nelion"],
        elo_cm=combined["elo_cm"],
        gamma_rad=combined["gamma_rad"],
        gamma_stark=combined["gamma_stark"],
        gamma_vdw=combined["gamma_vdw"],
        limb=combined.get("limb", np.zeros(n, dtype=np.int16)),
        ratiolg=np.float64(ratiolg),
        ixwlbeg=np.int64(ixwlbeg),
        wl_start=np.float64(WL_START),
        wl_end=np.float64(WL_END),
        resolution=np.float64(RESOLUTION),
    )
    print(f"[mol] wrote {MOL_LINES} ({MOL_LINES.stat().st_size/1e6:.2f} MB)")


def main() -> None:
    make_atmosphere()
    run_synthesis(DIAG_TIO, with_molecules=True, spec_path=REF / "m3500_tio.spec")
    run_synthesis(DIAG_ATOMIC, with_molecules=False, spec_path=REF / "m3500_atomic.spec")
    dump_molecular_lines()
    shutil.copyfile(MOLECULES_DAT_SRC, MOLECULES_DAT_DST)
    print(f"[done] molecules.dat copied to {MOLECULES_DAT_DST}")

    # Report shapes for the deliverable.
    d = np.load(DIAG_TIO)
    print("\n[diag_tio.npz keys]:", list(d.keys()))
    print("  line_opacity", d["line_opacity"].shape,
          "continuum_absorption", d["continuum_absorption"].shape,
          "wavelength", d["wavelength"].shape)
    print("  flux_total", d["flux_total"].shape,
          "flux_continuum", d["flux_continuum"].shape)


if __name__ == "__main__":
    main()
