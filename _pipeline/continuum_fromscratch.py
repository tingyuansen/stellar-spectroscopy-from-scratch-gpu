#!/usr/bin/env python
"""From-scratch continuum + the TABCONT/IWAVETAB continuum-cutoff table (Stage 3).

Bridges the book's OWN from-scratch multi-element EOS (eos_fromscratch.popsall, Stage 1) to the
book's OWN from-scratch continuous opacity (verify_kapp.compute_kapp = the Lecture-3 KAPP physics,
proven bit-exact vs pyk), and builds the TABCONT continuum-cutoff table the line deposit keys off.

The continuum acont/sigmac is the Lecture-3 KAPP engine — nothing new is taught here; we only
ASSEMBLE its per-layer continuum-species `pops` dict from the Stage-1 EOS populations (the same
slices L3's verify_kapp builds), so the continuum is recomputed from the CURRENT structure rather
than read as a given.  TABCONT = (acont + sigmac)·1e-3 / stim at the 344 coarse KAPCONT
wavelengths; inactive columns (bluer than the OS-grid start) get the 1e10·1e-3/stim sentinel.
IWAVETAB is the static OS-pixel index of those 344 cutoff wavelengths (model-independent).

Faithful pure-NumPy transcription of kgpu/atmosphere_xlines.build_tabcont (numpy-only, zero pyk).
Honest inputs: reference/leankurucz_tables.npz (the L3 continuum tables) + the static wavetab grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import verify_kapp as VK   # the book's own Lecture-3 KAPP continuum (compute_kapp)

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"

_H_PLANCK = 6.6256e-27
_K_BOLTZ = 1.38054e-16
_C_LIGHT_NM = 2.99792458e17

# solar IFOP card (atlas12; the same one verify_kapp uses)
IFOP_SOLAR = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]


def build_continuum_pops(pa, T, rho, xne):
    """Assemble the L3 KAPP `pops` dict from the Stage-1 EOS PopsAllResult.

    Maps the continuum-species populations the Lecture-3 compute_kapp consumes: the H I/H II,
    He, C/Mg/Al/Si/Fe named vectors + the HOTOP and sum-of-q^2 metal slabs — exactly the slices
    verify_kapp builds in its main from `population_per_ion`.
    """
    pop = pa.population_per_ion                      # (nd, 6, 139)
    nd = T.size
    pops = dict(
        temperature=np.asarray(T, np.float64),
        mass_density=np.asarray(rho, np.float64),
        electron_density=np.asarray(xne, np.float64),
        xnfph=pa.xnfph, xnf_h=pa.xnf_h,
        he1_mode11=pop[:, 0, 1], he2_mode11=pop[:, 1, 1], he3_mode11=pop[:, 2, 1],
        he1_mode12=pa.xnf_he1, he2_mode12=pa.xnf_he2,
        xnfpc=pop[:, 0:2, 5].copy(), xnfpmg=pop[:, 0, 11], xnfpal=pop[:, 0, 12],
        xnfpsi=pop[:, 0, 13], xnfpfe=pop[:, 0, 25],
        xnfpn=pop[:, 0, 6], xnfpo=pop[:, 0, 7],
        xnfpmg2=pop[:, 1, 11], xnfpsi2=pop[:, 1, 13], xnfpca2=pop[:, 1, 19],
    )
    hotop_xnfp = np.zeros((nd, 21))
    hotop_xnfp[:, 0:4] = pop[:, 0:4, 5]
    hotop_xnfp[:, 4:9] = pop[:, 0:5, 6]
    hotop_xnfp[:, 9:15] = pop[:, 0:6, 7]
    hotop_xnfp[:, 15:21] = pop[:, 0:6, 9]
    xnf_sumqq = np.zeros((nd, 5))
    for elem in (5, 6, 7, 9, 11, 13, 15, 25):
        for iz in range(1, 6):
            xnf_sumqq[:, iz - 1] += (iz * iz) * pop[:, iz, elem]
    pops["hotop_xnfp"] = hotop_xnfp
    pops["xnf_sumqq"] = xnf_sumqq
    return pops


def compute_continuum(freq, pa, T, rho, xne, ifop=IFOP_SOLAR):
    """acont, sigmac at arbitrary frequencies from the Stage-1 EOS (the L3 KAPP continuum)."""
    pops = build_continuum_pops(pa, T, rho, xne)
    return VK.compute_kapp(np.asarray(freq, np.float64), pops, ifop)


def kapcont_wavetab():
    """The static 344-pt KAPCONT continuum-cutoff wavelengths + their OS-pixel indices."""
    d = np.load(REF / "wavetab_grid.npz")
    return np.asarray(d["wavetab"], np.float64), np.asarray(d["i_wavetab"], np.int64)


def build_tabcont(pa, T, rho, xne, os_start_nm, ifop=IFOP_SOLAR):
    """TABCONT(nd, 344) + i_wavetab from the from-scratch continuum (per-iteration).

    tabcont = (acont + sigmac)·1e-3 / stim at the active 344 coarse wavelengths; inactive
    columns (bluer than os_start_nm = the OS grid's first wavelength) get 1e10·1e-3/stim.
    """
    wavetab, i_wavetab = kapcont_wavetab()
    Td = np.asarray(T, np.float64)
    nd = Td.size
    hkt = _H_PLANCK / np.maximum(Td * _K_BOLTZ, 1e-300)
    os0 = float(os_start_nm)

    tabcont = np.zeros((nd, 344), dtype=np.float32)
    active = wavetab[:343] > os0
    act_idx = np.nonzero(active)[0]
    if act_idx.size:
        freq = _C_LIGHT_NM / np.maximum(wavetab[act_idx], 1e-300)
        acont, sigmac = compute_continuum(freq, pa, Td, rho, xne, ifop=ifop)
        stim = np.maximum(1.0 - np.exp(-np.outer(hkt, freq)), 1e-300)
        tabcont[:, act_idx] = ((acont + sigmac) * 1.0e-3 / stim).astype(np.float32)
    inact = np.nonzero(~active)[0]
    if inact.size:
        freq = _C_LIGHT_NM / np.maximum(wavetab[inact], 1e-300)
        stim = np.maximum(1.0 - np.exp(-np.outer(hkt, freq)), 1e-300)
        tabcont[:, inact] = (1.0e10 * 1.0e-3 / stim).astype(np.float32)
    tabcont[:, 343] = tabcont[:, 342]
    return tabcont, i_wavetab
