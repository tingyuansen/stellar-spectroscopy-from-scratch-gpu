#!/usr/bin/env python
"""Generate the PFSAHA reference bundle for verify_pfsaha.py and Lecture 2.

READ-ONLY w.r.t. pykurucz: we import pykurucz solely to (a) read the atomic
DATA tables (NNN / POTION / PFIRON grids / special-element level blocks / the
PFGROUND ground-state correction) and (b) produce the COMPARISON TARGET, namely
pykurucz's own PFSAHA output (per-ion partition functions U and the per-ion
populations population_per_ion = n_ion/U), so the book can be checked bit-exact.

Two files are written:

  reference/pfsaha_inputs.npz   -- STATE + atomic-data TABLES only, NO answers.
      The depth state (T, tkev, tk, hckt, tlog, P, electron_density, xnatom,
      xabund) plus the measured atomic-data tables PFSAHA consumes.  These are
      exactly the quantities Lecture 2 already treats as given (Pe/Pg/T and the
      tabulated atomic data), reused the way the lecture already reuses U(T).

  reference/pfsaha_truth.npz    -- the COMPARISON TARGET (pykurucz PFSAHA
      output) computed by pykurucz's _pfsaha_exact_python engine, kept in a
      SEPARATE file so the verify script must COMPUTE its own answer and only
      then load this to compare.

WHY a fresh target instead of atmosphere.npz['population_per_ion']: that stored
array multiplies the PFSAHA fractions F/PART by the NMOLEC molecule-depleted
atomic density (atoms locked into H2/CO/... are removed before POPS multiplies
by XNATOM).  NMOLEC is a separate molecular-equilibrium engine (Lectures 7+),
NOT part of the PFSAHA ionization core.  To verify the PFSAHA core itself we
target pykurucz's PFSAHA output evaluated at the atmosphere's own XNATOM, which
is the genuine "per-ion populations from the Saha/partition physics".

Run:  python _pipeline/make_pfsaha_reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BOOK = Path(__file__).resolve().parent.parent
REF = BOOK / "reference"
PYK = Path("/Users/ysting/pykurucz")
sys.path.insert(0, str(PYK))

from synthe_py.tools import pops_exact as PE  # noqa: E402

PE.load_fortran_data()

# ---------------------------------------------------------------------------
# 1.  Read the atmosphere STATE (depth grid, thermodynamics, abundances).
#     These are the same inputs Lecture 2 already carries: temperature, the
#     pressure, the electron density, the atomic number density, abundances.
# ---------------------------------------------------------------------------
atm = np.load(REF / "atmosphere.npz", allow_pickle=True)
T = np.asarray(atm["temperature"], np.float64)
tkev = np.asarray(atm["tkev"], np.float64)        # kT [eV]
tk = np.asarray(atm["tk"], np.float64)            # kT [erg]
hckt = np.asarray(atm["hckt"], np.float64)        # hc/kT [cm]   (for E[cm^-1]*hckt)
tlog = np.asarray(atm["tlog"], np.float64)        # ln(T)
P = np.asarray(atm["gas_pressure"], np.float64)   # gas pressure [dyn/cm^2]
xne = np.asarray(atm["electron_density"], np.float64)
xnatom = np.asarray(atm["xnatm"], np.float64)     # atomic number density [cm^-3]
xabund = np.asarray(atm["xabund"], np.float64)    # number fraction over all atoms
nlay = T.size

# ---------------------------------------------------------------------------
# 2.  Read the atomic-DATA tables PFSAHA consumes (measured data, not physics).
# ---------------------------------------------------------------------------
fd = np.load(PYK / "synthe_py" / "data" / "fortran_data.npz")
NNN = np.asarray(fd["nnn"], np.int64)             # (6, 374) packed partition data
POTION = np.asarray(fd["potion"], np.float64)     # (999,)   ionization potentials [cm^-1]
pf = np.load(PYK / "synthe_py" / "data" / "pfiron_data.npz")
PFTAB = np.asarray(pf["pftab"], np.float64)       # (7,56,10,9) Fe-group partition grid
PFIRON_POTLO = np.asarray(pf["potlo"], np.float64)
PFIRON_POTLOLOG = np.asarray(pf["potlolog"], np.float64)
LOCZ = np.asarray(PE.LOCZ, np.int64)
SCALE = np.asarray(PE.SCALE, np.float64)

# special-element level energies / weights (atomic data DATA blocks)
SPECIAL = {}
for nm in ["EHYD", "GHYD", "EHE1", "GHE1", "EHE2", "GHE2", "EC1", "GC1",
           "EC2", "GC2", "EO1", "GO1", "EMG1", "GMG1", "EMG2", "GMG2",
           "EAL1", "GAL1", "ESI1", "GSI1", "ESI2", "GSI2", "ECA1", "GCA1",
           "ECA2", "GCA2", "ENA1", "GNA1", "EB1", "GB1", "EK1", "GK1"]:
    SPECIAL["sp_" + nm] = np.asarray(getattr(PE, nm), np.float64)

# PFGROUND ground-state correction, PRE-EVALUATED on this T grid (pure numbers).
# nelion = (Z-1)*6 + ion is the flat element/ion index PFGROUND is keyed on.
MAX_NELION = 99 * 6 + 10
pfground_tab = np.ones((MAX_NELION + 1, nlay), np.float64)   # row 0 unused
for nel in range(1, MAX_NELION + 1):
    for j in range(nlay):
        pfground_tab[nel, j] = PE.pfground(nel, float(T[j]))

# per-element ion-count code (xnfpelsyn.for CALL POPS codes -> NION per element)
CODE_NION = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5,
             11: 5, 12: 5, 13: 5, 14: 5, 15: 5, 16: 5, 17: 4, 18: 4, 19: 4,
             20: 9, 21: 9, 22: 9, 23: 9, 24: 9, 25: 9, 26: 9, 27: 9, 28: 9}
nion_per_Z = np.array([CODE_NION.get(z, 2) for z in range(1, 100)], np.int64)

# ---------------------------------------------------------------------------
# 3.  Build the COMPARISON TARGET with pykurucz's own PFSAHA engine.
#     mode 13 -> PART (the per-ion partition function U).
#     mode 11 -> F/PART (population fraction per ion = n_ion / (U * n_element)).
#     population_per_ion = (F/PART) * xnatom * xabund = n_ion / U.
# ---------------------------------------------------------------------------
NSTORE = 6                                          # ion stages stored per element
U_truth = np.zeros((nlay, 99, NSTORE), np.float64)        # partition functions
popfrac_truth = np.zeros((nlay, 99, NSTORE), np.float64)  # F/PART per ion
pop_truth = np.zeros((nlay, 99, NSTORE), np.float64)      # n_ion/U per ion

ap = np.zeros((nlay, 31), np.float64)
af = np.zeros((nlay, 31), np.float64)
for z in range(1, 100):
    nion = int(nion_per_Z[z - 1])
    ap.fill(0.0)
    af.fill(0.0)
    PE._pfsaha_exact_python(z, nion, 13, T, tkev, tk, atm["hkt"], hckt, tlog,
                            P, xne, xnatom, ap, None, 0)
    PE._pfsaha_exact_python(z, nion, 11, T, tkev, tk, atm["hkt"], hckt, tlog,
                            P, xne, xnatom, af, None, 0)
    for ion in range(NSTORE):
        U_truth[:, z - 1, ion] = ap[:, ion]
        popfrac_truth[:, z - 1, ion] = af[:, ion]
        pop_truth[:, z - 1, ion] = af[:, ion] * xnatom * xabund[z - 1]

# also capture pykurucz's NELECT electron density solved from scratch, as an
# independent charge-balance benchmark (target n_e the book's solver must hit).
xne_seed = xne.copy()
xnatom_seed = xnatom.copy()
PE.nelect_exact(temperature=T, tk=tk, tkev=tkev, hkt=np.asarray(atm["hkt"]),
                hckt=hckt, tlog=tlog, gas_pressure=P,
                electron_density=xne_seed, xnatom=xnatom_seed, xabund=xabund,
                departure_tables=None)
xne_nelect_truth = xne_seed.copy()

# ---------------------------------------------------------------------------
# 4.  Write the two files.
# ---------------------------------------------------------------------------
inputs = dict(
    T=T, tkev=tkev, tk=tk, hckt=hckt, tlog=tlog, gas_pressure=P,
    electron_density=xne, xnatom=xnatom, xabund=xabund,
    NNN=NNN, POTION=POTION, PFTAB=PFTAB,
    PFIRON_POTLO=PFIRON_POTLO, PFIRON_POTLOLOG=PFIRON_POTLOLOG,
    LOCZ=LOCZ, SCALE=SCALE, nion_per_Z=nion_per_Z,
    pfground_tab=pfground_tab,
)
inputs.update(SPECIAL)
np.savez_compressed(REF / "pfsaha_inputs.npz", **inputs)

np.savez_compressed(
    REF / "pfsaha_truth.npz",
    U=U_truth, population_fraction=popfrac_truth, population_per_ion=pop_truth,
    xne_nelect=xne_nelect_truth,
)

print(f"wrote {REF/'pfsaha_inputs.npz'}  (keys: {sorted(inputs)})")
print(f"wrote {REF/'pfsaha_truth.npz'}   (U, population_fraction, "
      f"population_per_ion, xne_nelect)")
print(f"layers={nlay}  elements=99  ion stages stored={NSTORE}")
