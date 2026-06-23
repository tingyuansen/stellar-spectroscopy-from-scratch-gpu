#!/usr/bin/env python3
"""gen_nmolec_reference.py — regenerate the molecular-equilibrium reference inputs
and ground truth, READ-ONLY against pykurucz.

This is NOT part of the from-scratch verification (verify_nmolec.py is).  It only
exists to (re)produce the two shipped npz under reference/ by driving pykurucz's
own coupled solver `nmolec_exact(..., use_gibbs=False)` — the EXACT path that
mol_populations.py uses to fill the atmosphere's molecular populations.  pykurucz
is imported read-only; nothing in it is modified.

Outputs:
  reference/nmolec_inputs.npz       solver INPUTS (T, Pgas, n_e, abundances, and the
                                    PFSAHA atomic-ion equilibrium constants equilj_ion;
                                    poly_mask flags the polynomial molecules whose K_f
                                    verify_nmolec.py recomputes from scratch)
  reference/nmolec_groundtruth.npz  COMPARISON-ONLY ground truth (molecular densities
                                    xnmol, XNATOM, the converged XN vector, n_e)

Run from the book root with the book venv active:
    python _pipeline/gen_nmolec_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PYKURUCZ = "/Users/ysting/pykurucz"
ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"


def main():
    sys.path.insert(0, PYKURUCZ)
    from synthe_py.tools.readmol_exact import readmol_exact
    from synthe_py.tools.nmolec_exact import nmolec_exact
    from synthe_py.tools.pops_exact import pfsaha_exact, load_fortran_data
    import synthe_py.tools.pops_exact as pops
    from synthe_py.tools.departure_tables import initialize_departure_tables
    import synthe_py.tools.nmolec_exact as nm

    # --- atmosphere from the shipped cool M-dwarf model (m3500g50) ---
    d = np.load(REF / "m3500g50.npz", allow_pickle=True)
    n_layers = int(d["temperature"].shape[0])
    T = np.asarray(d["temperature"], float)
    P = np.asarray(d["gas_pressure"], float)
    ne = np.maximum(np.asarray(d["electron_density"], float), 1e-30)
    xa = np.asarray(d["xabund"], float)
    xabund99 = np.zeros(99)
    xabund99[: min(99, len(xa))] = xa[:99]

    tkev = T * 8.617333262e-5
    tk = T * 1.380649e-16
    tlog = np.log(T)
    xnatom_atomic = P / tk - ne
    hckt = np.asarray(d["hckt"], float) if "hckt" in d.files else 1.4388 / T

    nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc = readmol_exact(
        REF / "molecules.dat")

    if pops.POTION is None:
        load_fortran_data()
    dep = initialize_departure_tables(n_layers)
    electron_work = ne.copy()
    answer_full = np.zeros((n_layers, 31))

    def pfsaha_wrapper(j, iz, nion, mode, frac, nlte_on):
        try:
            pfsaha_exact(j=0, iz=iz, nion=nion, mode=mode, temperature=T, tkev=tkev,
                         tk=tk, hkt=6.6256e-27 / tk, hckt=hckt, tlog=tlog,
                         gas_pressure=P, electron_density=electron_work,
                         xnatom=xnatom_atomic, answer=answer_full,
                         departure_tables=dep, nlte_on=nlte_on)
            frac[j, :] = answer_full[j, :]
        except Exception:
            pass

    # capture the per-layer EQUILJ the solver actually uses (the ion ones are the
    # atomic-physics input we ship; the polynomial ones are recomputed in verify)
    captured = {}
    orig = nm._accumulate_molecules_atlas7

    def patched(**kw):
        li, it = kw["layer_index"], kw["iteration"]
        if it == 0 and li not in captured:
            captured[li] = np.array(kw["equilj"][:nummol]).copy()
        return orig(**kw)

    nm._accumulate_molecules_atlas7 = patched

    xnmol = np.zeros((n_layers, 200))
    ed = ne.copy()
    xnatom_out, xnmol_out, xnz_out = nmolec_exact(
        n_layers=n_layers, temperature=T, tkev=tkev, tk=tk, tlog=tlog,
        gas_pressure=P, electron_density=ed, xabund=xabund99,
        xnatom_atomic=xnatom_atomic, nummol=nummol, code_mol=code_mol, equil=equil,
        locj=locj, kcomps=kcomps, idequa=idequa, nequa=nequa,
        bhyd=dep["bhyd"], bc1=dep["bc1"], bo1=dep["bo1"], bmg1=dep["bmg1"],
        bal1=dep["bal1"], bsi1=dep["bsi1"], bca1=dep["bca1"],
        pfsaha_func=pfsaha_wrapper, xnmol=xnmol, use_gibbs=False, auto_gibbs=False)

    nm._accumulate_molecules_atlas7 = orig

    EQUILJ_ref = np.array([captured[j] for j in range(n_layers)])
    poly = equil[0, :nummol] != 0.0

    np.savez_compressed(REF / "nmolec_inputs.npz",
                        temperature=T, gas_pressure=P, electron_density=ne,
                        xabund=xabund99, equilj_ion=EQUILJ_ref, poly_mask=poly)
    np.savez_compressed(REF / "nmolec_groundtruth.npz",
                        xnmol=xnmol_out, xnatom=xnatom_out, xnz=xnz_out,
                        electron=ed, code_mol=code_mol[:nummol],
                        nequa=nequa, idequa=idequa[:nequa])
    print("Wrote reference/nmolec_inputs.npz and reference/nmolec_groundtruth.npz")
    print(f"  nummol={nummol} nequa={nequa} depths={n_layers}")


if __name__ == "__main__":
    main()
