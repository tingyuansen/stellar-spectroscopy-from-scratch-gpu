#!/usr/bin/env python
"""Far-UV metal bound-free continuum forest (Stage-7 remediation).

The book's Lecture-3 KAPP continuum (verify_kapp.compute_kapp) was validated at 500-510 nm and
truncated each neutral-metal bound-free to its single OPTICAL edge (e.g. C1OP keeps only the
22006 cm^-1 edge).  At the hot deep base (11425 K) in the far-UV (91-150 nm) those single-edge
tails undershoot the true bound-free by 10-2000x, leaving a ~7% deep-base Rosseland-mean deficit
(Stage-7 diagnosis: C I far-UV bf carries 764 of 803 cm^2/g at 100 nm).  This module restores the
FULL far-UV metal bf forest — the multi-edge level series, Luo-Pradhan / Nahar-Pradhan resonances,
and Kramers-Gaunt n-series free-free — for C I, Mg I, Si I, Al I, Fe I.

Faithful pure-NumPy transcription of kgpu/continuum.py (_c1op/_mg1op/_si1op/_al1op/_fe1op_atlas,
numpy/torch only, ZERO pyk), which matches pykurucz's kapp_continuum bit-exact.  Reuses the book's
OWN Karzas-Latter cross-section verify_kapp.xkarsas (the same _xkarsas kgpu uses) + the karsas
tables already shipped in leankurucz_tables.npz.  The cross-section coefficients (_C1_ELEV etc.)
are atomic-data constants (Karzas-Latter / atlas12.for) — the same honest-input class as the L3
tables.  ATLAS12 ground-limit photoionisation; active only below the Lyman cutoff (3.28805e15 Hz).
"""
from __future__ import annotations

import numpy as np

import verify_kapp as VK   # the book's own xkarsas (Karzas-Latter) + karsas tables

_C_LIGHT_CM = 2.99792458e10
_MET_RYD = 109732.298
_LYMAN_CUT_HZ = 3.28805e15

# ── C I — atlas12 C1OP, 25 lower levels ──────────────────────────────────────
_C1_RYD = 109732.298
_C1_ELEV = np.array([
    79314.86, 78731.27, 78529.62, 78309.76, 78226.35, 77679.82, 73975.91, 72610.72,
    71374.90, 70743.95, 69722.00, 68856.33, 61981.82, 60373.00, 21648.01, 10192.63,
    43.42, 16.42, 0.00, 119878.0, 105798.7, 97878.0, 75254.93, 64088.85, 33735.20])
_C1_GLEV = np.array([
    9.0, 3.0, 7.0, 15.0, 21.0, 5.0, 1.0, 5.0, 9.0, 3.0, 15.0, 3.0, 3.0, 9.0, 1.0,
    5.0, 5.0, 3.0, 1.0, 3.0, 3.0, 5.0, 12.0, 15.0, 5.0])
_C1_ELIM1 = 90862.70
_C1_ELIM2 = 90820.42
_C1_ELIM2B = _C1_ELIM2 + 63.42
_C1_ELIM3 = _C1_ELIM2 + 43003.3
_C1_ELIM_FF = _C1_ELIM2


def _c1op(freq, hckt):
    """C I far-UV bound-free cross-section sum H(depth) per frequency.  (nfreq, ndepth)."""
    freq = np.atleast_1d(np.asarray(freq, np.float64))
    hckt = np.asarray(hckt, np.float64)
    nf = freq.size; nd = hckt.size
    out = np.zeros((nf, nd), np.float64)
    bolt = _C1_GLEV[:, None] * np.exp(-_C1_ELEV[:, None] * hckt[None, :])   # (25, nd)
    for jf in range(nf):
        f = float(freq[jf])
        if f > _LYMAN_CUT_HZ:
            continue
        wno = f / _C_LIGHT_CM
        x = np.zeros(25)
        for i in range(14):                      # group 1 -> C II ground limit
            thr = _C1_ELIM1 - _C1_ELEV[i]
            if wno < thr:
                continue
            ell = 2 if i < 6 else (1 if i < 12 else 0)
            x[i] = VK.xkarsas(f, 9.0 / _C1_RYD * thr, 3, ell)
        for elim_g2, weight in ((_C1_ELIM2, 1.0 / 3.0), (_C1_ELIM2B, 2.0 / 3.0)):   # Luo-Pradhan
            if wno >= elim_g2 - _C1_ELEV[14]:
                xs0 = 10.0 ** (-16.80 - (wno - elim_g2 + _C1_ELEV[14]) / 3.0 / _C1_RYD)
                eps = (wno - 97700.0) * 2.0 / 2743.0
                xs1 = (68e-18 * eps + 118e-18) / (eps * eps + 1.0)
                x[14] += (xs0 + xs1) * weight
            if wno >= elim_g2 - _C1_ELEV[15]:
                xd0 = 10.0 ** (-16.80 - (wno - elim_g2 + _C1_ELEV[15]) / 3.0 / _C1_RYD)
                eps1 = (wno - 93917.0) * 2.0 / 9230.0
                xd1 = (22e-18 * eps1 + 26e-18) / (eps1 * eps1 + 1.0)
                eps2 = (wno - 111130.0) * 2.0 / 2743.0
                xd2 = (-10.5e-18 * eps2 + 46e-18) / (eps2 * eps2 + 1.0)
                x[15] += (xd0 + xd1 + xd2) * weight
            for i in range(16, 19):
                if wno >= elim_g2 - _C1_ELEV[i]:
                    x[i] += 10.0 ** (-16.80 - (wno - elim_g2 + _C1_ELEV[i]) / 3.0 / _C1_RYD) * weight
        for i in range(19, 25):                  # group 3 -> higher limit
            thr = _C1_ELIM3 - _C1_ELEV[i]
            if wno < thr:
                continue
            x[i] = VK.xkarsas(f, 4.0 / _C1_RYD * thr, 2, 1) * 3.0
        freq3 = 2.815e29 / (f * f * f)           # Kramers-Gaunt n>=4 ff
        kthr = max(_C1_ELIM_FF - _C1_RYD / 16.0, _C1_ELIM_FF - wno)
        h_kram = freq3 * 6.0 / (_C1_RYD * hckt) * (np.exp(-kthr * hckt) - np.exp(-_C1_ELIM_FF * hckt))
        out[jf] = h_kram + x @ bolt
    return out


# ── Mg I — atlas12 MG1OP, 15 levels ──────────────────────────────────────────
_MG1_ELEV = np.array([
    54676.710, 54676.438, 54192.284, 53134.642, 49346.729, 47957.034, 47847.797,
    46403.065, 43503.333, 41197.043, 35051.264, 21919.178, 21870.464, 21850.405, 0.0])
_MG1_GLEV = np.array([21., 7., 15., 5., 3., 15., 9., 5., 1., 3., 3., 5., 3., 1., 1.])
_MG1_ELIM = 61671.02


def _mg1op(freq, hckt):
    freq = np.atleast_1d(np.asarray(freq, np.float64)); hckt = np.asarray(hckt, np.float64)
    nf = freq.size; nd = hckt.size; out = np.zeros((nf, nd))
    bolt = _MG1_GLEV[:, None] * np.exp(-_MG1_ELEV[:, None] * hckt[None, :])
    elim = _MG1_ELIM
    for jf in range(nf):
        f = float(freq[jf])
        if f > _LYMAN_CUT_HZ:
            continue
        wno = f / _C_LIGHT_CM
        x = np.zeros(15)
        for i, ell in ((0, 3), (1, 3), (2, 2), (3, 2), (4, 1)):
            thr = elim - _MG1_ELEV[i]
            if wno >= thr:
                x[i] = VK.xkarsas(f, 16.0 / _MET_RYD * thr, 4, ell)
        for i, (thr0, c0, p0) in ((5, (13713.986, 25e-18, 2.7)), (6, (13823.223, 33.8e-18, 2.8)),
                                  (7, (15267.955, 45e-18, 2.7)), (8, (18167.687, 0.43e-18, 2.6)),
                                  (9, (20473.617, 2.1e-18, 2.6))):
            if wno >= elim - _MG1_ELEV[i]:
                x[i] = c0 * (thr0 / wno) ** p0
        if wno >= elim - _MG1_ELEV[10]:
            x[10] = 16e-18 * (26619.756 / wno) ** 2.1 - 7.8e-18 * (26619.756 / wno) ** 9.5
        for i in range(11, 14):
            if wno >= elim - _MG1_ELEV[i]:
                x[i] = max(20e-18 * (39759.842 / wno) ** 2.7, 40e-18 * (39759.842 / wno) ** 14)
        if wno >= elim - _MG1_ELEV[14]:
            x[14] = 1.1e-18 * ((elim - _MG1_ELEV[14]) / wno) ** 10
        freq3 = 2.815e29 / (f * f * f)
        kthr = max(elim - _MET_RYD / 25.0, elim - wno)
        h_kram = freq3 * 2.0 / (_MET_RYD * hckt) * (np.exp(-kthr * hckt) - np.exp(-elim * hckt))
        out[jf] = h_kram + x @ bolt
    return out


# ── Si I — atlas12 SI1OP, 33 levels ──────────────────────────────────────────
_SI1_ELEV = np.array([
    59962.284, 59100., 59077.112, 58893.40, 58801.529, 58777., 57488.974, 56503.346,
    54225.621, 53387.34, 53362.24, 51612.012, 50533.424, 50189.389, 49965.894, 49399.670,
    49128.131, 48161.459, 47351.554, 47284.061, 40991.884, 39859.920, 15394.370, 6298.850,
    223.157, 77.115, 0.000, 94000., 79664.0, 72000., 56698.738, 45303.310, 33326.053])
_SI1_GLEV = np.array([
    9., 56., 15., 7., 3., 28., 21., 5., 15., 3., 7., 1., 9., 5., 21., 3., 9., 15.,
    5., 3., 3., 9., 1., 5., 5., 3., 1., 3., 3., 5., 12., 15., 5.])
_SI1_NL = ((4, 2), (4, 3), (4, 2), (4, 2), (4, 2), (4, 3), (4, 2), (4, 2),
           (3, 2), (3, 2), (3, 2), (4, 1), (3, 2), (4, 1), (3, 2), (4, 1),
           (4, 1), (4, 1), (3, 2), (4, 1), (4, 0), (4, 0))
_SI1_ZF = (16., 16., 16., 16., 16., 16., 16., 16., 9., 9., 9., 16., 9., 16., 9.,
           16., 16., 16., 9., 16., 16., 16.)
_SI1_ELIM1 = 65939.18
_SI1_ELIM2 = 65747.55
_SI1_ELIM3 = 65747.5 + 42824.35


def _si1op(freq, hckt):
    freq = np.atleast_1d(np.asarray(freq, np.float64)); hckt = np.asarray(hckt, np.float64)
    nf = freq.size; nd = hckt.size; out = np.zeros((nf, nd))
    bolt = _SI1_GLEV[:, None] * np.exp(-_SI1_ELEV[:, None] * hckt[None, :])
    for jf in range(nf):
        f = float(freq[jf])
        if f > _LYMAN_CUT_HZ:
            continue
        wno = f / _C_LIGHT_CM
        x = np.zeros(33)
        for i in range(22):
            thr = _SI1_ELIM1 - _SI1_ELEV[i]
            if wno >= thr:
                n_qn, l_qn = _SI1_NL[i]
                x[i] = VK.xkarsas(f, _SI1_ZF[i] / _MET_RYD * thr, n_qn, l_qn)
        for elim_g, weight in ((_SI1_ELIM2, 1.0 / 3.0), (_SI1_ELIM2 + 287.45, 2.0 / 3.0)):
            if wno >= elim_g - _SI1_ELEV[22]:
                eps = (wno - 70000.0) * 2.0 / 6500.0
                reson = (97e-18 * eps + 94e-18) / (eps * eps + 1.0)
                x[22] += (37e-18 * (50353.180 / wno) ** 2.40 + reson) * weight
            if wno >= elim_g - _SI1_ELEV[23]:
                eps = (wno - 78600.0) * 2.0 / 13000.0
                reson = (-10e-18 * eps + 77e-18) / (eps * eps + 1.0)
                x[23] += (24.5e-18 * (59448.700 / wno) ** 1.85 + reson) * weight
            for i in (24, 25, 26):
                if wno >= elim_g - _SI1_ELEV[i]:
                    ratio = 65524.393 / wno
                    eff_w = (2.0 / 3.0) if i == 25 else weight
                    x[i] += (72e-18 * ratio ** 1.90 if wno <= 74000.0
                             else 93e-18 * ratio ** 4.00) * eff_w
        for i in range(27, 33):
            thr = _SI1_ELIM3 - _SI1_ELEV[i]
            if wno >= thr:
                x[i] = VK.xkarsas(f, 9.0 / _MET_RYD * thr, 3, 1) * 3.0
        freq3 = 2.815e29 / (f * f * f)
        kthr = max(_SI1_ELIM2 - _MET_RYD / 25.0, _SI1_ELIM2 - wno)
        h_kram = freq3 * 6.0 / (_MET_RYD * hckt) * (np.exp(-kthr * hckt) - np.exp(-_SI1_ELIM2 * hckt))
        out[jf] = h_kram + x @ bolt
    return out


# ── Al I — atlas12 AL1OP (frequency-only twin 2P edge) ───────────────────────
_AL1_ELIM = 48278.37


def _al1op(freq, hckt):
    freq = np.atleast_1d(np.asarray(freq, np.float64)); hckt = np.asarray(hckt, np.float64)
    nf = freq.size; nd = hckt.size; out = np.zeros((nf, nd))
    for jf in range(nf):
        f = float(freq[jf])
        if f > _LYMAN_CUT_HZ:
            continue
        wno = f / _C_LIGHT_CM
        x = 0.0
        if wno >= _AL1_ELIM - 112.061:
            x += 6.5e-17 * ((_AL1_ELIM - 112.061) / wno) ** 5 * 4.0
        if wno >= _AL1_ELIM:
            x += 6.5e-17 * (_AL1_ELIM / wno) ** 5 * 2.0
        out[jf] = x   # frequency-only -> uniform over depth
    return out


# ── Fe I — atlas7v FE1OP (48-transition resonance forest) ────────────────────
_FE1_G = np.array([
    25., 35., 21., 15., 9., 35., 33., 21., 27., 49., 9., 21., 27., 9., 9., 25.,
    33., 15., 35., 3., 5., 11., 15., 13., 15., 9., 21., 15., 21., 25., 35., 9.,
    5., 45., 27., 21., 15., 21., 15., 25., 21., 35., 5., 15., 45., 35., 55., 25.])
_FE1_E = np.array([
    500., 7500., 12500., 17500., 19000., 19500., 19500., 21000., 22000., 23000.,
    23000., 24000., 24000., 24500., 24500., 26000., 26500., 26500., 27000., 27500.,
    28500., 29000., 29500., 29500., 29500., 30000., 31500., 31500., 33500., 33500.,
    34000., 34500., 34500., 35000., 35500., 37000., 37000., 37000., 38500., 40000.,
    40000., 41000., 41000., 43000., 43000., 43000., 43000., 44000.])
_FE1_WNO = np.array([
    63500., 58500., 53500., 59500., 45000., 44500., 44500., 43000., 58000., 41000.,
    54000., 40000., 40000., 57500., 55500., 38000., 57500., 57500., 37000., 54500.,
    53500., 55000., 34500., 34500., 34500., 34000., 32500., 32500., 32500., 32500.,
    32000., 29500., 29500., 31000., 30500., 29000., 27000., 54000., 27500., 24000.,
    47000., 23000., 44000., 42000., 42000., 21000., 42000., 42000.])


def _fe1op(freq, hckt):
    freq = np.atleast_1d(np.asarray(freq, np.float64)); hckt = np.asarray(hckt, np.float64)
    nf = freq.size; nd = hckt.size; out = np.zeros((nf, nd))
    for jf in range(nf):
        wno = float(freq[jf]) / _C_LIGHT_CM
        if wno < 21000.0:
            continue
        active = _FE1_WNO <= wno
        if not np.any(active):
            continue
        xsect = 3e-18 / (1.0 + ((_FE1_WNO[active] + 3000.0 - wno) / _FE1_WNO[active] / 0.1) ** 4)
        bolt = _FE1_G[active][:, None] * np.exp(-_FE1_E[active][:, None] * hckt[None, :])
        out[jf] = xsect @ bolt
    return out


def _l3_single_edge_metals(freq, pops, hckt, stim, ehvkt, rho):
    """The book's L3 single-edge metal bf (C1/Mg1/Al1/Si1; Fe1=0) that compute_kapp ADDED.

    We subtract these and add the full forest so the net REPLACES (kgpu's coulff_atlas=True path),
    exactly as ATLAS12 does — never double-counting the optical edge.  cm^2/g, shape (ndepth, nfreq).
    """
    freq = np.asarray(freq, np.float64); wno = freq / _C_LIGHT_CM
    nd = hckt.size; nf = freq.size
    out = np.zeros((nd, nf), np.float64)
    # C1: single visible edge 22006.370
    c1 = np.full((nd, nf), 1e-30)
    m = wno >= 22006.370
    c1[:, m] += (2.1e-18 * (22006.370 / wno[m]) ** 1.5 * 3.0)[None, :] * np.exp(-68856.33 * hckt)[:, None] * stim[:, m]
    out += c1 * pops["xnfpc"][:, 0][:, None] / rho[:, None]
    # Mg1
    MG1 = [(13713.986, 25e-18, 2.7, 15.0, 47957.034), (13823.223, 33.8e-18, 2.8, 9.0, 47847.797),
           (15267.955, 45e-18, 2.7, 5.0, 46403.065), (18167.687, 0.43e-18, 2.6, 1.0, 43503.333),
           (20473.617, 2.1e-18, 2.6, 3.0, 41197.043)]
    mg = np.full((nd, nf), 1e-30)
    for (thr, c0, p0, g, e) in MG1:
        m = wno >= thr
        mg[:, m] += (c0 * (thr / wno[m]) ** p0 * g)[None, :] * np.exp(-e * hckt)[:, None] * stim[:, m]
    out += mg * pops["xnfpmg"][:, None] / rho[:, None]
    # Al1
    AL1 = [(8002.467, 50e-18, 3, 6.0, 40275.903), (9346.231, 50e-18, 3, 10.0, 38932.139),
           (10588.957, 56.7e-18, 1.9, 2.0, 37689.413), (15318.007, 14.5e-18, 1, 6.0, 32960.363),
           (15842.129, 47e-18, 1.83, 10.0, 32436.241)]
    al = np.full((nd, nf), 1e-30)
    bal2 = np.exp(-48278.37 * hckt)
    for (thr, c0, p0, g, e) in AL1:
        m = wno >= thr
        al[:, m] += (c0 * (thr / wno[m]) ** p0 * g)[None, :] * np.exp(-e * hckt)[:, None] * (1.0 - bal2[:, None] * ehvkt[:, m])
    out += al * pops["xnfpal"][:, None] / rho[:, None]
    # Si1: single visible edge 17777.641
    si = np.full((nd, nf), 1e-30)
    m = wno >= 17777.641
    si[:, m] += (18e-18 * (17777.641 / wno[m]) ** 3 * 15.0)[None, :] * np.exp(-48161.459 * hckt)[:, None] * (1.0 - ehvkt[:, m])
    out += si * pops["xnfpsi"][:, None] / rho[:, None]
    return out


def add_faruv_metal_bf(acont, freq, pops):
    """Replace the book's L3 single-edge metal bf with the full far-UV forest.  Returns acont.

    net = acont - L3_single_edge_metals + forest  (kgpu coulff_atlas=True == ATLAS12 COOLOP).
    acont: (ndepth, nfreq).  Caller's H*pop*stim/rho contract is applied here.
    """
    temp = np.asarray(pops["temperature"], np.float64)
    rho = np.maximum(np.asarray(pops["mass_density"], np.float64), 1e-30)
    H_PLANCK = 6.62607015e-27; K_BOLTZ = 1.380649e-16
    hckt = (H_PLANCK * _C_LIGHT_CM) / (K_BOLTZ * temp)
    freq = np.asarray(freq, np.float64)
    ehvkt = np.exp(-H_PLANCK * freq[None, :] / (K_BOLTZ * temp[:, None]))
    stim = 1.0 - ehvkt
    forest = (
        (_c1op(freq, hckt), pops["xnfpc"][:, 0]),
        (_mg1op(freq, hckt), pops["xnfpmg"]),
        (_si1op(freq, hckt), pops["xnfpsi"]),
        (_al1op(freq, hckt), pops["xnfpal"]),
        (_fe1op(freq, hckt), pops["xnfpfe"]),
    )
    add = np.zeros_like(acont)
    for H, pop in forest:                      # H: (nfreq, ndepth)
        add += (H.T * pop[:, None] / rho[:, None]) * stim
    sub = _l3_single_edge_metals(freq, pops, hckt, stim, ehvkt, rho)
    return acont - sub + add
