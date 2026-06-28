#!/usr/bin/env python
"""Molecular line-deposit populations from scratch (Stage-7 molecular remediation).

The from-scratch line deposit drops the ~440k molecular line records in 300-400 nm (nelion 841-940),
because the molecular population slots xnfdop[:, 841:940] are zero — leaving a ~6% deep-base
Rosseland-mean deficit (Stage-7: abross[base] 0.936 vs pyk 1.0, exactly kgpu's atomic-only ceiling).
This module fills those slots from the book's OWN Lecture-13 molecular equilibrium (NMOLEC):

  1. equilj_ion_from_saha  — the atomic-ion molecular formation constants from the Stage-1 EOS Saha.
  2. verify_nmolec.nmolec_solve — the book's L13 coupled molecular equilibrium (pure numpy, validated
     to 1e-13), driven by the Stage-1 EOS, giving the per-equation densities xnz.
  3. molecular_xnfpmol      — the XNFP/partition conversion xnfpmol = N_mol/U_mol per molecular slot.
  4. molecular doppler widths from the molecular masses + microturbulence.

(2) is the book's own L13.  (1),(3),(4) are faithful pure-NumPy transcriptions of kgpu/nmolec.py
(equilj_ion_from_saha / molecular_xnfpmol) + the molecular doppler — numpy-only, ZERO pyk.  The
molecule data is the honest reference/molecules.dat (the same file L13 ships).
"""
from __future__ import annotations

import numpy as np

import verify_nmolec as VN   # the book's own Lecture-13 NMOLEC solver + readmol

# the 16 molecular deposit slots (POPSALL mol_targets order): (code, nelion, mass[amu])
_MOL_SLOTS = (
    (101, 841, 2.0), (106, 846, 13.0), (107, 847, 15.0), (108, 848, 17.0),
    (112, 851, 25.0), (114, 853, 29.0), (120, 858, 41.0), (124, 862, 53.0),
    (126, 864, 57.0), (606, 868, 24.0), (607, 869, 26.0), (608, 870, 28.0),
    (814, 889, 44.0), (822, 895, 64.0), (823, 896, 67.0), (10108, 940, 18.0),
)
MOL_CODES = tuple(c for c, _n, _m in _MOL_SLOTS)
MOL_NELION = tuple(n for _c, n, _m in _MOL_SLOTS)
MOL_MASS = np.array([m for _c, _n, m in _MOL_SLOTS], np.float64)

_K_BOLTZ_TK = 1.38054e-16
_AMU_G = 1.660e-24
_C_LIGHT_CMS = 2.99792458e10

# NMOLEC standard atomic weights (kgpu/nmolec _ATMASS_NM, atlas12)
_ATMASS_NM = np.array([
    1.008, 4.003, 6.939, 9.013, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18, 22.99,
    24.31, 26.98, 28.09, 30.98, 32.07, 35.45, 39.95, 39.10, 40.08, 44.96, 47.90,
    50.94, 52.00, 54.94, 55.85, 58.94, 58.71, 63.55, 65.37, 69.72, 72.60, 74.92,
    78.96, 79.91, 83.80, 85.48, 87.63, 88.91, 91.22, 92.91, 95.95, 99.00, 101.1,
    102.9, 106.4, 107.9, 112.4, 114.8, 118.7, 121.8, 127.6, 126.9, 131.3, 132.9,
    137.4, 138.9, 140.1, 140.9, 144.3, 147.0, 150.4, 152.0, 157.3, 158.9, 162.5,
    164.9, 167.3, 168.9, 173.0, 175.0, 178.5, 181.0, 183.9, 186.3, 190.2, 192.2,
    195.1, 197.0, 200.6, 204.4, 207.2, 209.0, 210.0, 211.0, 222.0, 223.0, 226.1,
    227.1, 232.0, 231.0, 238.0, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 254.0],
    np.float64)


def _mol_struct(molecules_path):
    """Parse molecules.dat (the book's L13 readmol)."""
    nummol, code_mol, equil, locj, kcomps, idequa, nequa, _nloc = VN.readmol(molecules_path)
    # the atomic-ion "molecules" (equil[0]==0): (jmol, Z, ion) for equilj_ion_from_saha
    ion_meta = []
    for jmol in range(nummol):
        if equil[0, jmol] != 0.0:
            continue
        Z = int(code_mol[jmol]); ion = int(locj[jmol + 1] - locj[jmol]) - 1
        ion_meta.append((jmol, Z, ion))
    return nummol, code_mol, equil, locj, kcomps, idequa, nequa, ion_meta


def equilj_ion_from_saha(ion_meta, nummol, Fpop, xne):
    """Atomic-ion molecular formation constants from the EOS Saha (equilj_ion = F_ion/F_neutral * n_e^ion)."""
    nd = Fpop.shape[0]
    out = np.zeros((nd, nummol), np.float64)
    xne_s = np.maximum(np.asarray(xne, np.float64), 1e-300)
    for jmol, Z, ion in ion_meta:
        if ion >= Fpop.shape[2] or Z < 1 or Z > Fpop.shape[1]:
            continue
        f0 = Fpop[:, Z - 1, 0]; fion = Fpop[:, Z - 1, ion]
        good = f0 > 0.0
        out[good, jmol] = fion[good] / f0[good] * xne_s[good] ** ion
    return out


def molecular_xnfpmol(T, xnz, neutral_partition, code_mol, equil, locj, kcomps, idequa, nequa, nummol, codes):
    """xnfpmol = N_mol/U_mol per requested molecule code (the line-deposit population numerator).

    Faithful pure-NumPy transcription of kgpu.nmolec.molecular_xnfpmol (the XNFP/partition conversion
    pyk's POPSALL mode=1 does).  Step A: convert per-equation densities xnz[:,k] to N/U*translational.
    Step B: assemble the polynomial-molecule xnfpmol.  Returns (nd, len(codes)).
    """
    T = np.asarray(T, np.float64); nd = T.size
    xnz = np.array(xnz, np.float64, copy=True)
    Up = np.asarray(neutral_partition, np.float64)
    n_up = Up.shape[1]
    sqrtT = np.sqrt(np.maximum(T, 1e-300))
    # Step A
    for k in range(1, nequa):
        idk = int(idequa[k])
        if idk <= 0:
            continue
        if idk == 100:
            xnz[:, k] = xnz[:, k] / (2.0 * 2.4148e15 * T * sqrtT)
            continue
        amass = float(_ATMASS_NM[idk - 1]) if 1 <= idk <= _ATMASS_NM.size else float(idk)
        pf = Up[:, idk - 1] if 1 <= idk <= n_up else np.ones(nd)
        denom = np.maximum(pf * 1.8786e20 * np.sqrt(np.maximum((amass * T) ** 3, 1e-300)), 1e-300)
        xnz[:, k] = xnz[:, k] / denom
    # Step B
    xnfpmol_all = np.zeros((nd, nummol), np.float64)
    tkev = T / 11604.5
    for jmol in range(nummol):
        e1 = float(equil[0, jmol])
        if e1 == 0.0:
            continue
        loc1 = int(locj[jmol]); loc2 = int(locj[jmol + 1])
        amass = 0.0
        for loc in range(loc1, loc2):
            k = int(kcomps[loc])
            if k >= nequa:
                continue
            idk = int(idequa[k])
            if 1 <= idk <= _ATMASS_NM.size:
                amass += float(_ATMASS_NM[idk - 1])
        val = np.exp(e1 / np.maximum(tkev, 1e-300))
        for loc in range(loc1, loc2):
            k = int(kcomps[loc])
            if k >= nequa:
                val = val / np.maximum(xnz[:, nequa - 1], 1e-300)
            else:
                val = val * xnz[:, k]
        val = val * 1.8786e20 * np.sqrt(np.maximum((amass * T) ** 3, 1e-300))
        xnfpmol_all[:, jmol] = val
    out = np.zeros((nd, len(codes)), np.float64)
    for col, code in enumerate(codes):
        hits = np.where(np.abs(code_mol[:nummol] - float(code)) < 1e-3)[0]
        if hits.size:
            out[:, col] = xnfpmol_all[:, int(hits[0])]
    return out


def molecular_deposit_populations(T, P, pa, xabund, vturb, molecules_path):
    """Full from-scratch molecular line-deposit populations for the 16 molecular slots.

    Returns (xnfpmol[nd,16], dop_mol[nd,16], MOL_NELION): the deposit population numerators +
    Doppler widths for the molecular slots 841-940, from the Stage-1 EOS (pa) + the book's own
    Lecture-13 NMOLEC molecular equilibrium.
    """
    T = np.asarray(T, np.float64); P = np.asarray(P, np.float64)
    nummol, code_mol, equil, locj, kcomps, idequa, nequa, ion_meta = _mol_struct(molecules_path)
    Fpop = pa.popfrac * pa.U                                  # (nd,99,6) mode-12 F = popfrac*U
    equilj_ion = equilj_ion_from_saha(ion_meta, nummol, Fpop, pa.xne)
    # the book's L13 coupled molecular equilibrium -> per-equation densities xnz
    _xna, _xnmol, xnz, _e = VN.nmolec_solve(
        T, P, np.asarray(pa.xne, np.float64), np.asarray(xabund, np.float64),
        nummol, code_mol, equil, locj, kcomps, idequa, nequa, equilj_ion)
    # the deposit population numerator xnfpmol = N_mol/U_mol per molecular slot
    xnfpmol = molecular_xnfpmol(T, xnz, pa.U[:, :, 0], code_mol, equil, locj, kcomps, idequa,
                                nequa, nummol, MOL_CODES)
    # molecular Doppler widths v_D/c = sqrt(2kT/m + vturb^2)/c
    tk = _K_BOLTZ_TK * T
    vturb = np.asarray(vturb, np.float64)
    arg = 2.0 * tk[:, None] / (MOL_MASS[None, :] * _AMU_G) + (vturb ** 2)[:, None]
    dop_mol = np.sqrt(arg) / _C_LIGHT_CMS
    return xnfpmol, dop_mol, np.array(MOL_NELION, np.int64)


def fill_molecular_slots(xnfdop, dopple, rho, xnfpmol, dop_mol):
    """Write the molecular populations into the flat 1006-slot xnfdop/dopple (slots 841-940).

    XNFDOP[:, nelion-1] = xnfpmol / dop_mol / rho;  DOPPLE[:, nelion-1] = dop_mol.  Modifies the
    arrays in place and returns them.
    """
    rho = np.asarray(rho, np.float64)
    for col, nelion in enumerate(MOL_NELION):
        dop = dop_mol[:, col]
        dopple[:, nelion - 1] = dop
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = np.where((dop > 0) & (rho > 0), xnfpmol[:, col] / (dop * rho), 0.0)
        xnfdop[:, nelion - 1] = xn
    return xnfdop, dopple
