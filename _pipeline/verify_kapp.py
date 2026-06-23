#!/usr/bin/env python
"""From-scratch NumPy reproduction of pykurucz's CONTINUUM opacity engine (KAPP).

Reproduces reference/diag.npz["continuum_absorption"] and ["continuum_scattering"]
(each (80, 5941), float64) on the diag wavelength grid (500-510 nm), from the atomic
CROSS-SECTION tables (Karsas H I bf, HMINOP H- bf/ff, Gavrila H Rayleigh, COULFF Gaunt)
plus the EOS populations stored in atmosphere.npz.

Two-stage chain, faithful to convert_atm_to_npz.py + physics/continuum.py:

  (1) compute_kapp_continuum() at an EDGE-triplet frequency grid (3 freqs per
      continuum-edge interval), store log10(acont)/log10(sigmac) reshaped to
      (80, n_edges-1, 3) as the per-edge coefficients.  Only the edge that brackets
      500-510 nm (edge 185, freqset indices 555,556,557) is needed to fill the grid.

  (2) build_depth_continuum()'s 3-point Lagrange interpolation per edge in wavelength,
      then 10**.

The KAPP physics is a clean reimplementation of synthe_py/physics/kapp.py
compute_kapp_continuum (the JIT/guard machinery stripped, verified per-element).
Tables are loaded from pykurucz here for verification; a lecture would ship them as
a data file (the exact array list is printed at the end).

ONE documented deviation from the literal current pykurucz source (see report):
HE1OP's free-free term is weighted by the He II population in POPS mode-11
(= mode-12 / U_HeII), which is what built the reference; the current source feeds the
mode-12 value there.  This only matters in the deep T>8000 K layers (below the
continuum-forming photosphere) and is required to reproduce the shipped coefficients.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYKURUCZ = "/Users/ysting/pykurucz"
sys.path.insert(0, PYKURUCZ)

# ── tables (loaded from pykurucz for verification; ship as data in the lecture) ──
_KARSAS = np.load(f"{PYKURUCZ}/synthe_py/data/karsas_tables.npz")
FREQ_LOG = _KARSAS["FREQ_LOG"]            # (29,15)  descending log10(freq/zeff2) grid
XN_LOG = _KARSAS["XN_LOG"]                # (29,15)  log10 cross-section (l-summed)
XL_LOG_ARRAY = _KARSAS["XL_LOG_ARRAY"]    # (6,6,29) l-resolved log10 cross-section
EKARSAS = _KARSAS["EKARSAS"]              # (29,)    excitation energies (n>15 branch)

_KAPP = np.load(f"{PYKURUCZ}/synthe_py/data/kapp_continuum_tables.npz")
HMINOP_WBF = _KAPP["HMINOP_WBF"]          # (85,) wavelength nm for H- bf MAP1
HMINOP_BF = _KAPP["HMINOP_BF"]            # (85,) H- bf cross-section (1e-18 cm^2)
HMINOP_WAVEK = _KAPP["HMINOP_WAVEK"]      # (22,) WAVEK grid for H- ff
HMINOP_THETAFF = _KAPP["HMINOP_THETAFF"]  # (11,) theta grid for H- ff
HMINOP_FFBEG = _KAPP["HMINOP_FFBEG"]      # (11,11) H- ff (first 11 wavek cols)
HMINOP_FFEND = _KAPP["HMINOP_FFEND"]      # (11,11) H- ff (last 11 wavek cols)
HRAYOP_GAVRILAM = _KAPP["HRAYOP_GAVRILAM"]            # (74,) visible Rayleigh G
HRAYOP_GAVRILAMAB = _KAPP["HRAYOP_GAVRILAMAB"]        # (27,)
HRAYOP_GAVRILAMBC = _KAPP["HRAYOP_GAVRILAMBC"]        # (24,)
HRAYOP_GAVRILAMCD = _KAPP["HRAYOP_GAVRILAMCD"]        # (22,)
HRAYOP_GAVRILALYMANCONT = _KAPP["HRAYOP_GAVRILALYMANCONT"]    # (64,)
HRAYOP_FGAVRILALYMANCONT = _KAPP["HRAYOP_FGAVRILALYMANCONT"]  # (64,)
COULFF_Z4LOG = _KAPP["COULFF_Z4LOG"]      # (6,)
COULFF_A_TABLE = _KAPP["COULFF_A_TABLE"]  # (12,11) Coulomb ff Gaunt-factor table
HOTOP_TRANSITIONS = _KAPP["HOTOP_TRANSITIONS"]  # (60,7) hot-star bf transitions
_SI2OP_PEACH = _KAPP["_SI2OP_PEACH"]      # (14,6) Si II Peach tables
_SI2OP_FREQSI = _KAPP["_SI2OP_FREQSI"]    # (7,)
_SI2OP_FLOG = _KAPP["_SI2OP_FLOG"]        # (9,)
_SI2OP_TLG = _KAPP["_SI2OP_TLG"]          # (6,)
H_ENERGY_CM = _KAPP["H_ENERGY_CM"]        # (6,) H level energies cm^-1 (partition fn)
H_STAT_WEIGHT = _KAPP["H_STAT_WEIGHT"]    # (6,) H statistical weights

# ── constants (Fortran values, matching kapp.py exactly) ──
C_LIGHT_CM = 2.99792458e10
C_LIGHT_NM = 2.99792458e17
H_PLANCK = 6.62607015e-27
K_BOLTZ = 1.380649e-16
KBOLTZ_EV = 8.6171e-5
RYDBERG_CM = 109677.576
LN10 = np.log(10.0)
H_ENERGY_EV = H_ENERGY_CM / 8065.479
H_MAX_LEVEL = 6


# ── MAP1: Fortran parabolic/weighted-3-pt interpolation (identical to verify_josh) ──
def map1(xold, fold, xnew):
    nold, nnew = xold.size, xnew.size
    fnew = np.zeros(nnew)
    if nold == 0 or nnew == 0:
        return fnew
    xo = np.empty(nold + 1); fo = np.empty(nold + 1)
    xo[1:] = xold; fo[1:] = fold
    l = 2; ll = 0
    cfor = bfor = afor = cbac = bbac = abac = a = b = c = 0.0
    for k in range(1, nnew + 1):
        xk = xnew[k - 1]
        while True:
            if xk < xo[l]:
                if l == ll:
                    break
                if l == 2 or l == 3:
                    l = min(nold, l); c = 0.0
                    b = (fo[l] - fo[l - 1]) / (xo[l] - xo[l - 1]); a = fo[l] - xo[l] * b; ll = l
                    break
                l1 = l - 1
                if l > ll + 1 or l == 3 or l == 4:
                    l2 = l - 2
                    d = (fo[l1] - fo[l2]) / (xo[l1] - xo[l2])
                    cbac = fo[l] / ((xo[l] - xo[l1]) * (xo[l] - xo[l2])) + \
                           (fo[l2] / (xo[l] - xo[l2]) - fo[l1] / (xo[l] - xo[l1])) / (xo[l1] - xo[l2])
                    bbac = d - (xo[l1] + xo[l2]) * cbac
                    abac = fo[l2] - xo[l2] * d + xo[l1] * xo[l2] * cbac
                    if l >= nold:
                        c, b, a, ll = cbac, bbac, abac, l
                        break
                else:
                    cbac, bbac, abac = cfor, bfor, afor
                    if l == nold:
                        c, b, a, ll = cbac, bbac, abac, l
                        break
                d = (fo[l] - fo[l1]) / (xo[l] - xo[l1])
                cfor = fo[l + 1] / ((xo[l + 1] - xo[l]) * (xo[l + 1] - xo[l1])) + \
                       (fo[l1] / (xo[l + 1] - xo[l1]) - fo[l] / (xo[l + 1] - xo[l])) / (xo[l] - xo[l1])
                bfor = d - (xo[l] + xo[l1]) * cfor
                afor = fo[l1] - xo[l1] * d + xo[l] * xo[l1] * cfor
                wt = abs(cfor) / (abs(cfor) + abs(cbac)) if abs(cfor) != 0.0 else 0.0
                a = afor + wt * (abac - afor); b = bfor + wt * (bbac - bfor); c = cfor + wt * (cbac - cfor)
                ll = l
                break
            l += 1
            if l > nold:
                l = min(nold, l); c = 0.0
                b = (fo[l] - fo[l - 1]) / (xo[l] - xo[l - 1]); a = fo[l] - xo[l] * b; ll = l
                break
        fnew[k - 1] = a + (b + c * xk) * xk
    return fnew


def linter(xold, yold, xnew):
    """Linear interpolation/EXTRAPOLATION (Fortran LINTER) - no clamping at the ends."""
    nold, nnew = xold.size, xnew.size
    ynew = np.zeros(nnew)
    iold = 1
    for inew in range(nnew):
        while iold < nold - 1 and xnew[inew] >= xold[iold]:
            iold += 1
        denom = xold[iold] - xold[iold - 1]
        if abs(denom) < 1e-40:
            ynew[inew] = yold[iold - 1]
        else:
            w = (xnew[inew] - xold[iold - 1]) / denom
            ynew[inew] = yold[iold - 1] + (yold[iold] - yold[iold - 1]) * w
    return ynew


def xkarsas(freq, zeff_squared, n, ell):
    """Karsas hydrogen bf cross-section coefficient (binary search, log10(freq) linear)."""
    if freq <= 0.0 or zeff_squared <= 0.0 or n <= 0:
        return 0.0
    if ell < 0:
        ell = 0
    freq_log = np.log10(freq / zeff_squared)
    if n <= 15:
        column = FREQ_LOG[:, n - 1]
        if freq_log < column[-1]:
            return 0.0
        if ell >= n or n > 6:
            values = XN_LOG[:, n - 1]
        else:
            values = XL_LOG_ARRAY[ell, n - 1, :]
            if np.isnan(values[0]):
                return 0.0
        left, right = 1, column.size - 1
        idx = column.size
        while left <= right:
            mid = (left + right) // 2
            if freq_log > column[mid]:
                idx = mid; right = mid - 1
            else:
                left = mid + 1
        if idx >= column.size:
            return float(np.exp(values[-1] * LN10) / zeff_squared)
        denom = column[idx - 1] - column[idx]
        if abs(denom) < 1e-15:
            return float(np.exp(values[idx - 1] * LN10) / zeff_squared)
        w = (freq_log - column[idx]) / denom
        x_val = (values[idx - 1] - values[idx]) * w + values[idx]
        return float(np.exp(x_val * LN10) / zeff_squared)
    # n > 15 branch
    inv_n2 = 1.0 / (n * n)
    ryd_c = 109677.576 * C_LIGHT_CM
    freqn15_last = np.log10(ryd_c * inv_n2)
    if freq_log < freqn15_last:
        return 0.0
    for idx in range(1, 28):
        fcur = np.log10((EKARSAS[idx] + inv_n2) * ryd_c)
        if freq_log > fcur:
            fprev = np.log10((EKARSAS[idx - 1] + inv_n2) * ryd_c) if idx - 1 >= 1 else freqn15_last
            denom = fprev - fcur
            if denom == 0.0:
                return 0.0
            w = (freq_log - fcur) / denom
            x_val = (XN_LOG[idx - 1, 14] - XN_LOG[idx, 14]) * w + XN_LOG[idx, 14]
            return float(np.exp(x_val * LN10) / zeff_squared)
    return float(np.exp(XN_LOG[28, 14] * LN10) / zeff_squared)


def coulff(nz, freq, freqlg, temp, tlog):
    """Coulomb free-free Gaunt factor, vectorised over layers (Fortran COULFF, bilinear)."""
    if nz < 1 or nz > 6:
        return np.ones_like(temp)
    z4log = COULFF_Z4LOG[nz - 1]
    gamlog = 10.39638 - tlog / 1.15129 + z4log
    hvktlg = (freqlg - tlog) / 1.15129 - 20.63764
    igam = np.clip((gamlog + 7.0).astype(np.int64), 1, 10)
    ihvkt = np.clip((hvktlg + 9.0).astype(np.int64), 1, 11)
    p = gamlog - (igam - 7.0)
    q = hvktlg - (ihvkt - 9.0)
    ig = igam - 1; ih = ihvkt - 1
    a00 = COULFF_A_TABLE[ig, ih]
    a01 = np.where(ihvkt < 11, COULFF_A_TABLE[ig, np.minimum(ih + 1, 10)], a00)
    a10 = np.where(igam < 10, COULFF_A_TABLE[np.minimum(ig + 1, 11), ih], a00)
    a11 = np.where((igam < 10) & (ihvkt < 11),
                   COULFF_A_TABLE[np.minimum(ig + 1, 11), np.minimum(ih + 1, 10)], a00)
    return (1.0 - p) * ((1.0 - q) * a00 + q * a01) + p * ((1.0 - q) * a10 + q * a11)


def planck_nu(freq, temp):
    """Planck B_nu(T) in erg/s/cm^2/Hz/sr (RJ limit for tiny hnu/kT, matching kapp.py)."""
    const = 2.0 * H_PLANCK / C_LIGHT_CM**2
    x = H_PLANCK * freq / (K_BOLTZ * temp)
    bnu = np.where(x < 1e-6, 2.0 * K_BOLTZ * temp * freq**2 / C_LIGHT_CM**2,
                   const * freq**3 / np.expm1(np.where(x < 1e-6, 1.0, x)))
    bnu = np.where(np.isfinite(bnu), bnu, 0.0)
    return bnu


def seaton(freq0, xsect, power, a, freq):
    if freq < freq0:
        return 0.0
    ratio = freq0 / freq
    return xsect * (a + (1.0 - a) * ratio) * np.sqrt(ratio ** int(2.0 * power + 0.01))


def si2op_vectorized(freq, freqlg, temp, tlog):
    """Si II opacity (Peach tables), returns cross-section * partition per layer."""
    n_layers = temp.size
    nt = np.clip((temp / 2000.0).astype(int) - 4, 1, 5)
    dt = (tlog - _SI2OP_TLG[nt - 1]) / (_SI2OP_TLG[nt] - _SI2OP_TLG[nt - 1])
    n = 0
    for i in range(7):
        if freq > _SI2OP_FREQSI[i]:
            n = i + 1
            break
    else:
        n = 8
    d = ((freqlg - _SI2OP_FLOG[n - 1]) / (_SI2OP_FLOG[n] - _SI2OP_FLOG[n - 1])
         if 0 < n < 9 else 0.0)
    if n > 2:
        n = 2 * n - 2
    n = min(n, 13)
    d1 = 1.0 - d
    if n < 14:
        x = _SI2OP_PEACH[n] * d + _SI2OP_PEACH[n - 1] * d1 if n > 0 else _SI2OP_PEACH[0]
    else:
        x = _SI2OP_PEACH[13]
    result = np.zeros(n_layers)
    for j in range(n_layers):
        nj = nt[j] - 1
        val = x[nj] * (1.0 - dt[j]) + x[nj + 1] * dt[j] if nj < 5 else x[5]
        result[j] = np.exp(val) * 6.0
    return result


def hydrogen_partition(temp):
    kt = KBOLTZ_EV * temp
    U = np.zeros_like(temp)
    for i in range(H_MAX_LEVEL):
        U += H_STAT_WEIGHT[i] * np.exp(-H_ENERGY_EV[i] / kt)
    return U


# ──────────────────────────────────────────────────────────────────────────────
# compute_kapp_continuum: clean port returning (acont, sigmac)
# ──────────────────────────────────────────────────────────────────────────────
def compute_kapp(freq, pops, ifop):
    """Reproduce kapp.compute_kapp_continuum for the given frequency array.

    pops: dict of per-layer atmosphere quantities (see __main__ assembly).
    Returns acont, sigmac  (each (n_layers, nfreq)).
    """
    temp = pops["temperature"]
    rho = np.maximum(pops["mass_density"], 1e-30)
    xne = pops["electron_density"]
    n_layers = temp.size
    nfreq = freq.size

    # b-tables are all unity (LTE) in this model; KAPP reads atlas_tables.get(...,ones)
    bhyd = np.ones((n_layers, 8)); bmin = np.ones(n_layers)
    bhe1 = np.ones((n_layers, 29)); bhe2 = np.ones((n_layers, 6))

    xnfph = pops["xnfph"]              # (n_layers,2) mode-11 H I ground, H II
    xnf_h = pops["xnf_h"]             # mode-12 total neutral H
    # HE populations from the POPS mode-11 grid
    he1_11 = pops["he1_mode11"]; he2_11 = pops["he2_mode11"]; he3_11 = pops["he3_mode11"]
    he1_12 = pops["he1_mode12"]; he2_12 = pops["he2_mode12"]

    hkt = H_PLANCK / (K_BOLTZ * temp)
    hckt = hkt * C_LIGHT_CM
    tlog = np.log(np.maximum(temp, 1e-10))
    tkev = temp * KBOLTZ_EV
    waveno = freq / C_LIGHT_CM
    ehvkt = np.exp(-H_PLANCK * freq[None, :] / (K_BOLTZ * temp[:, None]))
    stim = 1.0 - ehvkt
    bnu = np.zeros((n_layers, nfreq))
    for j in range(nfreq):
        bnu[:, j] = planck_nu(freq[j], temp)

    ahyd = np.zeros((n_layers, nfreq)); ahmin = np.zeros((n_layers, nfreq))
    ah2p = np.zeros((n_layers, nfreq)); ahemin = np.zeros((n_layers, nfreq))
    ahe1 = np.zeros((n_layers, nfreq)); ahe2 = np.zeros((n_layers, nfreq))
    ac1 = np.zeros((n_layers, nfreq)); amg1 = np.zeros((n_layers, nfreq))
    aal1 = np.zeros((n_layers, nfreq)); asi1 = np.zeros((n_layers, nfreq))
    afe1 = np.zeros((n_layers, nfreq)); aluke = np.zeros((n_layers, nfreq))
    ahot = np.zeros((n_layers, nfreq))
    sigh = np.zeros((n_layers, nfreq)); sighe = np.zeros((n_layers, nfreq))
    sigel = np.zeros((n_layers, nfreq)); sigh2 = np.zeros((n_layers, nfreq))

    # H I bf level thresholds (wavenumber, weight 2n^2, energy cm^-1)
    HOP_LEVELS = [
        (15, 487.456, 450.0, 109191.313), (14, 559.579, 392.0, 109119.188),
        (13, 648.980, 338.0, 109029.789), (12, 761.649, 288.0, 108917.117),
        (11, 906.426, 242.0, 108772.336), (10, 1096.776, 200.0, 108581.992),
        (9, 1354.044, 162.0, 108324.719), (8, 1713.713, 128.0, 107965.051),
        (7, 2238.320, 98.0, 107440.444),
    ]
    HOP_LEVELS_B = [  # n<=6 use departure coeff (b - ehvkt) instead of *stim
        (6, 3046.604, 72.0, 106632.160, 5), (5, 4387.113, 50.0, 105291.651, 4),
        (4, 6854.871, 32.0, 102823.893, 3), (3, 12186.462, 18.0, 97492.302, 2),
        (2, 27419.659, 8.0, 82259.105, 1),
    ]

    xnfph1 = xnfph[:, 0]; xnfph2 = xnfph[:, 1]
    for j in range(nfreq):
        f = freq[j]; wno = waveno[j]; ehv = ehvkt[:, j]; st = stim[:, j]; bn = bnu[:, j]
        freq3 = 2.815e29 / (f * f * f)
        # N=16 to infinity
        h = freq3 * 2.0 / 2.0 / (RYDBERG_CM * hckt) * (
            np.exp(-np.maximum(109250.336, 109678.764 - wno) * hckt)
            - np.exp(-109678.764 * hckt)) * st
        s = h * bn
        for (n, thr, wt, e) in HOP_LEVELS:
            if wno >= thr:
                a = xkarsas(f, 1.0, n, n) * wt * np.exp(-e * hckt) * st
                h = h + a; s = s + a * bn
        for (n, thr, wt, e, bi) in HOP_LEVELS_B:
            if wno >= thr:
                bh = bhyd[:, bi]
                a = xkarsas(f, 1.0, n, n) * wt * np.exp(-e * hckt) * (bh - ehv)
                h = h + a; s = s + a * bn * st / np.maximum(bh - ehv, 1e-40)
        if wno >= 109678.764:  # N=1
            bh = bhyd[:, 0]
            a = xkarsas(f, 1.0, 1, 1) * 2.0 * 1.0 * (bh - ehv)
            h = h + a; s = s + a * bn * st / np.maximum(bh - ehv, 1e-40)
        h = h * xnfph1 / rho; s = s * xnfph1 / rho
        # free-free
        cff = coulff(1, f, np.log(f), temp, tlog)
        a_ff = 3.6919e8 / np.sqrt(temp) * cff / f * xne / f * xnfph2 / f * st / rho
        h = h + a_ff; s = s + a_ff * bn
        ahyd[:, j] = h

    # H2PLOP (H2+); only for f <= 3.28805e15 (always true at visible)
    for j in range(nfreq):
        f = freq[j]
        if f > 3.28805e15:
            continue
        freqlg = np.log(f)
        freq15 = f / 1.0e15
        fr = -3.0233e3 + (3.7797e2 + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * freqlg) * freqlg) * freqlg) * freqlg
        es = -7.342e-3 + (-2.409e0 + (1.028e0 + (-4.230e-1 + (1.224e-1 - 1.351e-2 * freq15) * freq15) * freq15) * freq15) * freq15
        st = stim[:, j]
        ah2p[:, j] = (np.exp(-es / tkev + fr + np.log(np.maximum(xnfph1, 1e-40)))
                      * 2.0 * bhyd[:, 0] * xnfph2 / rho * st)

    # HMINOP (H- bf + ff)
    bhyd1 = bhyd[:, 0]
    xhmin = (np.exp(0.754209 / tkev) / (2.0 * 2.4148e15 * temp * np.sqrt(temp))
             * bmin * bhyd1 * xnfph1 * xne)
    theta = 5040.0 / temp
    nthetaff = HMINOP_THETAFF.size
    ff_full = np.zeros((nthetaff, 22))
    for it in range(nthetaff):
        for iw in range(22):
            ff_full[it, iw] = HMINOP_FFBEG[iw, it] if iw < 11 else HMINOP_FFEND[iw - 11, it]
    fflog = np.zeros((22, nthetaff))
    for iw in range(22):
        for it in range(nthetaff):
            fflog[iw, it] = np.log(ff_full[it, iw] / HMINOP_THETAFF[it] * 5040.0 * K_BOLTZ)
    wfflog = np.log(91.134 / HMINOP_WAVEK)
    for j in range(nfreq):
        f = freq[j]; ehv = ehvkt[:, j]; st = stim[:, j]; bn = bnu[:, j]
        wave = 2.99792458e17 / f
        wavelog = np.log(wave)
        fftheta = np.zeros(n_layers)
        fftt_for_theta = np.zeros(nthetaff)
        for it in range(nthetaff):
            fftt_for_theta[it] = np.exp(linter(wfflog, fflog[:, it], np.array([wavelog]))[0])
        for layer in range(n_layers):
            fftheta[layer] = linter(HMINOP_THETAFF, fftt_for_theta, np.array([theta[layer]]))[0]
        hminbf = map1(HMINOP_WBF, HMINOP_BF, np.array([wave]))[0] if f > 1.82365e14 else 0.0
        hminff = fftheta * xnfph1 * 2.0 * bhyd1 * xne / rho * 1e-26
        h_bf = hminbf * 1e-18 * (1.0 - ehv / np.maximum(bmin, 1e-40)) * xhmin / rho
        ahmin[:, j] = h_bf + hminff

    # HEMIOP (He- ff), gated by ifop[6]
    if ifop[6] == 1:
        for j in range(nfreq):
            f = freq[j]
            ac = 3.397e-01 + (-5.216e14 + 7.039e30 / f) / f
            bc = -4.116e03 + (1.067e19 + 8.135e34 / f) / f
            cc = 5.081e08 + (-8.724e22 - 5.659e37 / f) / f
            ahemin[:, j] = (ac * temp + bc + cc / temp) / 1.0e15 * xne / 1.0e15 * he1_12 / 1.0e15 / rho

    # ELECOP (Thomson) - no stim
    for j in range(nfreq):
        sigel[:, j] = 0.6653e-24 * xne / rho

    # HRAYOP (H Rayleigh) - ground-state H from xnf_h / U(T)
    xnfph1_ray = xnf_h / hydrogen_partition(temp)
    freq_lyman = 3.288051e15; freq_step = 3.288051e13
    for j in range(nfreq):
        f = freq[j]
        if f < freq_lyman * 0.01:
            g = HRAYOP_GAVRILAM[0] * (f / freq_step) ** 2
        elif f <= freq_lyman * 0.74:
            i = int(f / freq_step); i = min(i + 1, 74); i = max(1, i)
            if i >= len(HRAYOP_GAVRILAM):
                i = len(HRAYOP_GAVRILAM) - 1
            if i > 1:
                g = HRAYOP_GAVRILAM[i - 2] + (HRAYOP_GAVRILAM[i - 1] - HRAYOP_GAVRILAM[i - 2]) / freq_step * (f - (i - 1) * freq_step)
            else:
                g = HRAYOP_GAVRILAM[0]
        elif f < freq_lyman * 0.755:
            g = 15.57
        elif f <= freq_lyman * 0.885:
            step_ab = 1.644026e13
            i = int((f - freq_lyman * 0.755) / step_ab) + 1; i = min(i + 1, 27); i = max(1, i)
            if i >= len(HRAYOP_GAVRILAMAB):
                i = len(HRAYOP_GAVRILAMAB) - 1
            if i > 1:
                f1 = freq_lyman * 0.755 + ((i - 1) - 1) * 1.664026e13
                g = HRAYOP_GAVRILAMAB[i - 2] + (HRAYOP_GAVRILAMAB[i - 1] - HRAYOP_GAVRILAMAB[i - 2]) / step_ab * (f - f1)
            else:
                g = HRAYOP_GAVRILAMAB[0]
        elif f < freq_lyman * 0.890:
            g = 8.0
        elif f <= freq_lyman * 0.936:
            step_bc = 0.657610e13
            i = int((f - freq_lyman * 0.890) / step_bc) + 1; i = min(i + 1, 24); i = max(1, i)
            if i >= len(HRAYOP_GAVRILAMBC):
                i = len(HRAYOP_GAVRILAMBC) - 1
            if i > 1:
                f1 = freq_lyman * 0.890 + ((i - 1) - 1) * step_bc
                g = HRAYOP_GAVRILAMBC[i - 2] + (HRAYOP_GAVRILAMBC[i - 1] - HRAYOP_GAVRILAMBC[i - 2]) / step_bc * (f - f1)
            else:
                g = HRAYOP_GAVRILAMBC[0]
        elif f < freq_lyman * 0.938:
            g = 9.0
        elif f <= freq_lyman * 0.959:
            step_cd = 0.3288051e13
            i = int((f - freq_lyman * 0.938) / step_cd) + 1; i = min(i + 1, 22); i = max(1, i)
            if i >= len(HRAYOP_GAVRILAMCD):
                i = len(HRAYOP_GAVRILAMCD) - 1
            if i > 1:
                f1 = freq_lyman * 0.938 + ((i - 1) - 1) * step_cd
                g = HRAYOP_GAVRILAMCD[i - 2] + (HRAYOP_GAVRILAMCD[i - 1] - HRAYOP_GAVRILAMCD[i - 2]) / step_cd * (f - f1)
            else:
                g = HRAYOP_GAVRILAMCD[0]
        elif f <= freq_lyman:
            g = HRAYOP_GAVRILALYMANCONT[0]
        else:
            g = map1(HRAYOP_FGAVRILALYMANCONT, HRAYOP_GAVRILALYMANCONT, np.array([f / freq_lyman]))[0]
        xsect = 6.65e-25 * g**2
        sigh[:, j] = xsect * xnfph1_ray * 2.0 * bhyd1 / rho

    # HERAOP (He Rayleigh), gated by ifop[7]
    if ifop[7] == 1:
        for j in range(nfreq):
            f = freq[j]
            wave = 2.99792458e18 / min(f, 5.15e15)
            ww = wave**2
            sig = 5.484e-14 / (ww * ww) * (1.0 + (2.44e5 + 5.94e10 / max(ww - 2.90e5, 1e-10)) / ww) ** 2
            sighe[:, j] = sig * he1_12 / rho * bhe1[:, 0]

    # H2RAOP (H2 Rayleigh), gated by ifop[12]
    if ifop[12] == 1:
        poly_T = (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 + (1.06206e-18 - 3.08720e-23 * temp) * temp) * temp) * temp) * temp) * temp
        exp_term = np.clip(4.478 / tkev - 4.64584e1 + poly_T - 1.5 * tlog, -100, 100)
        xnh2 = (xnfph1_ray * 2.0 * bhyd1) ** 2 * np.exp(exp_term) / rho
        for j in range(nfreq):
            f = freq[j]
            wave = 2.99792458e18 / min(f, 2.922e15)
            ww = wave**2
            sig = (8.14e-13 + 1.28e-6 / ww + 1.61 / (ww * ww)) / (ww * ww)
            sigh2[:, j] = sig * xnh2

    # ── HE1OP ──
    # bound: weighted by He I mode-11.  free-free: weighted by He II mode-11
    #   (= mode-12 / U_HeII).  See module docstring / report for this deviation.
    rydberg_he = 109722.267
    HE1_N5 = [(4368.190, 3.0, 193942.57, 28), (4388.260, 9.0, 193922.5, 27),
              (4388.260, 27.0, 193922.5, 26), (4389.390, 7.0, 193921.37, 25),
              (4389.450, 15.0, 193921.31, 24), (4392.369, 5.0, 193918.391, 23),
              (4393.515, 15.0, 193917.245, 22), (4509.980, 9.0, 193800.78, 21),
              (4647.133, 1.0, 193663.627, 20), (4963.671, 3.0, 193347.089, 19)]
    HE1_N4 = [(6817.943, 3.0, 191492.817, 18), (6858.680, 7.0, 191452.08, 17),
              (6858.960, 21.0, 191451.80, 16), (6864.201, 5.0, 191446.559, 15),
              (6866.172, 15.0, 191444.588, 14), (7093.620, 9.0, 191217.14, 13),
              (7370.429, 1.0, 190940.331, 12), (8012.550, 3.0, 190298.210, 11)]
    HE1_N3 = [(12101.289, (58.81, -2.89), 3.0, 186209.471, 10),
              (12205.695, (85.20, -3.69), 5.0, 186105.065, 9),
              (12209.106, (85.20, -3.69), 15.0, 186101.654, 8),
              (12746.066, (49.30, -2.60), 9.0, 185564.694, 7),
              (13445.824, (23.85, -1.86), 1.0, 184864.936, 6),
              (15073.868, (12.69, -1.54), 3.0, 183236.892, 5)]
    HE1_N2 = [(27175.760, (81.35, -3.5, 0.0), 3.0, 171135.000, 4),
              (29223.753, (61.21, -2.9, 0.0), 9.0, 169087.007, 3),
              (32033.214, (26.83, -1.91, 0.0), 1.0, 166277.546, 2)]
    for j in range(nfreq):
        f = freq[j]; wno = waveno[j]; ehv = ehvkt[:, j]; st = stim[:, j]; bn = bnu[:, j]
        freqlg = np.log(f); freq3 = 2.815e29 / (f * f * f)
        bhe2_1 = bhe2[:, 0]
        h = (freq3 * 4.0 / 2.0 / (rydberg_he * hckt)
             * (np.exp(-np.maximum(195262.919, 198310.76 - wno) * hckt) - np.exp(-198310.76 * hckt))
             * st * bhe2_1)
        s = h * bn
        for (thr, g, e, bi) in HE1_N5 + HE1_N4:
            if wno >= thr:
                x = freq3 / (3125.0 if bi >= 19 else 1024.0)
                a = x * g * np.exp(-e * hckt) * (bhe1[:, bi] - bhe2_1 * ehv)
                h = h + a
                denom = bhe1[:, bi] / np.maximum(bhe2_1, 1e-40) - ehv
                s = s + a * bn * st / np.maximum(denom, 1e-40)
        for (thr, cf, g, e, bi) in HE1_N3:
            if wno >= thr:
                x = np.exp(cf[0] + cf[1] * freqlg)
                a = x * g * np.exp(-e * hckt) * (bhe1[:, bi] - bhe2_1 * ehv)
                h = h + a
                denom = bhe1[:, bi] / np.maximum(bhe2_1, 1e-40) - ehv
                s = s + a * bn * st / np.maximum(denom, 1e-40)
        for (thr, cf, g, e, bi) in HE1_N2:
            if wno >= thr:
                x = np.exp(cf[0] + cf[1] * freqlg)
                a = x * g * np.exp(-e * hckt) * (bhe1[:, bi] - bhe2_1 * ehv)
                h = h + a
                denom = bhe1[:, bi] / np.maximum(bhe2_1, 1e-40) - ehv
                s = s + a * bn * st / np.maximum(denom, 1e-40)
        if wno >= 38454.691:  # 2S 3S
            x = np.exp(-390.026 + (21.035 - 0.318 * freqlg) * freqlg)
            a = x * 3.0 * np.exp(-159856.069 * hckt) * (bhe1[:, 1] - bhe2_1 * ehv)
            h = h + a
            denom = bhe1[:, 1] / np.maximum(bhe2_1, 1e-40) - ehv
            s = s + a * bn * st / np.maximum(denom, 1e-40)
        if wno >= 198310.760:  # 1S 1S
            x = np.exp(33.32 - 2.0 * freqlg)
            a = x * 1.0 * 1.0 * (bhe1[:, 0] - bhe2_1 * ehv)
            h = h + a
            denom = bhe1[:, 0] / np.maximum(bhe2_1, 1e-40) - ehv
            s = s + a * bn * st / np.maximum(denom, 1e-40)
        h = h * he1_11 / rho; s = s * he1_11 / rho
        cff = coulff(1, f, freqlg, temp, tlog)
        # DEVIATION: He II population is mode-11 (he2_11), not mode-12.
        a_ff = 3.619e8 / np.sqrt(temp) * cff / f * xne / f * he2_11 / f * st / rho
        h = h + a_ff
        ahe1[:, j] = h

    # ── HE2OP ── uses mode-11 He II (bound) and mode-11 He III (free-free)
    rydberg_he2 = 438889.068
    HE2_LEVELS = [(5418.390, 162.0, 433490.46, 59049.0), (6857.660, 128.0, 432051.19, 32768.0),
                  (8956.950, 98.0, 429951.90, 16807.0)]
    HE2_B = [(12191.437, 72.0, 426717.413, 7776.0, 5, (1.0986, -2.704e13, 1.229e27)),
             (17555.715, 50.0, 421353.135, 3125.0, 4, (1.102, -3.909e13, 2.371e27)),
             (27430.925, 32.0, 411477.925, 1024.0, 3, (1.101, -5.765e13, 4.593e27)),
             (48766.491, 18.0, 390142.359, 243.0, 2, (1.101, -9.863e13, 1.035e28)),
             (109726.529, 8.0, 329182.321, 32.0, 1, (1.105, -2.375e14, 4.077e28)),
             (438908.850, 2.0, 0.0, 1.0, 0, (0.9916, 2.719e13, -2.268e30))]
    for j in range(nfreq):
        f = freq[j]; wno = waveno[j]; ehv = ehvkt[:, j]; st = stim[:, j]; bn = bnu[:, j]
        freq3 = 2.815e29 / (f * f * f)
        xnfprho = he2_11 / rho
        h = (freq3 * 16.0 * 2.0 / 2.0 / (rydberg_he2 * hckt)
             * (np.exp(-np.maximum(434519.959, 438908.85 - wno) * hckt) - np.exp(-438908.85 * hckt))
             * st * xnfprho)
        for (thr, wt, e, div) in HE2_LEVELS:
            if wno >= thr:
                x = freq3 * 16.0 / div
                a = x * wt * np.exp(-e * hckt) * st * xnfprho
                h = h + a
        for (thr, wt, e, div, bi, poly) in HE2_B:
            if wno >= thr:
                bb = bhe2[:, bi]
                x = freq3 * 16.0 / div * (poly[0] + (poly[1] + poly[2] / f) / f)
                if e == 0.0:
                    a = x * wt * 1.0 * (bb - ehv) * xnfprho
                else:
                    a = x * wt * np.exp(-e * hckt) * (bb - ehv) * xnfprho
                h = h + a
        cff = coulff(2, f, np.log(f), temp, tlog)
        a_ff = 3.6919e8 * 4.0 / np.sqrt(temp) * cff / f * xne / f * he3_11 / f * st / rho
        h = h + a_ff
        ahe2[:, j] = h

    # ── C1OP ── (active bf terms have x=0 placeholders below visible; h floor 1e-30)
    xnfpc = pops["xnfpc"]
    for j in range(nfreq):
        wno = waveno[j]; ehv = ehvkt[:, j]; st = stim[:, j]; bn = bnu[:, j]
        h = 1e-30 * np.ones(n_layers)
        if wno >= 22006.370:
            x = 2.1e-18 * (22006.370 / wno) ** 1.5
            a = x * 3.0 * np.exp(-68856.33 * hckt) * (np.ones(n_layers) - np.ones(n_layers) * ehv)
            h = h + a
        # other C1 edges are far-UV (>=28880); inactive at 500-510 nm.
        ac1[:, j] = h * xnfpc[:, 0] / rho

    # ── MG1OP ──
    xnfpmg = pops["xnfpmg"]
    MG1 = [(13713.986, 25e-18, 13713.986, 2.7, 15.0, 47957.034),
           (13823.223, 33.8e-18, 13823.223, 2.8, 9.0, 47847.797),
           (15267.955, 45e-18, 15267.955, 2.7, 5.0, 46403.065),
           (18167.687, 0.43e-18, 18167.687, 2.6, 1.0, 43503.333),
           (20473.617, 2.1e-18, 20473.617, 2.6, 3.0, 41197.043)]
    for j in range(nfreq):
        wno = waveno[j]; ehv = ehvkt[:, j]; st = stim[:, j]; bn = bnu[:, j]
        h = 1e-30 * np.ones(n_layers)
        for (thr, c0, w0, p0, g, e) in MG1:
            if wno >= thr:
                x = c0 * (w0 / wno) ** p0
                a = x * g * np.exp(-e * hckt) * (np.ones(n_layers) - np.ones(n_layers) * ehv)
                h = h + a
        # higher MG1 edges (>=26619) inactive at 500-510 nm.
        amg1[:, j] = h * xnfpmg / rho

    # ── AL1OP ── (Al I); bal2_1 = exp(-48278.37*hckt) is the ionization-limit factor.
    #   bound-free uses (bal1 - bal2_1*ehvkt); bal1 = 1 (LTE).
    xnfpal = pops["xnfpal"]; xnfpsi = pops["xnfpsi"]; xnfpfe = pops["xnfpfe"]
    bal1 = np.ones((n_layers, 9))
    AL1 = [(6958.993, None, 14.0, 41319.377, 8),           # 4F 2F (x=0)
           (8002.467, (50e-18, 8002.467, 3), 6.0, 40275.903, 7),
           (9346.231, (50e-18, 9346.231, 3), 10.0, 38932.139, 6),
           (10588.957, (56.7e-18, 10588.957, 1.9), 2.0, 37689.413, 5),
           (15318.007, (14.5e-18, 15318.007, 1), 6.0, 32960.363, 4),
           (15842.129, (47e-18, 15842.129, 1.83), 10.0, 32436.241, 3)]
    for j in range(nfreq):
        wno = waveno[j]; ehv = ehvkt[:, j]
        bal2_1 = np.exp(-48278.37 * hckt)
        h = 1e-30 * np.ones(n_layers)
        for (thr, xcoef, g, e, bi) in AL1:
            if wno >= thr:
                if xcoef is None:
                    x = 0.0
                else:
                    c0, w0, p0 = xcoef
                    x = c0 * (w0 / wno) ** p0
                a = x * g * np.exp(-e * hckt) * (bal1[:, bi] - bal2_1 * ehv)
                h = h + a
        # higher AL1 edges (>=22930) inactive at 500-510 nm.
        aal1[:, j] = h * xnfpal / rho

    # ── SI1OP ── (Si I); bsi2_1 = 1 (LTE). Active visible edge: PP 3D (17777, x!=0);
    #   16810/18587/18655 have x=0; higher edges (>=24947) far-UV.
    bsi1 = np.ones((n_layers, 11)); bsi2 = np.ones((n_layers, 10))
    SI1 = [(16810.969, None, 9.0, 49128.131, 10),                 # PP 3P (x=0)
           (17777.641, (18e-18, 17777.641, 3), 15.0, 48161.459, 9),  # PP 3D
           (18587.546, None, 5.0, 47351.554, 8),                  # PD 1D (x=0)
           (18655.039, None, 3.0, 47284.061, 7)]                  # PP 1P (x=0)
    for j in range(nfreq):
        wno = waveno[j]; ehv = ehvkt[:, j]
        bsi2_1 = bsi2[:, 0]
        h = 1e-30 * np.ones(n_layers)
        for (thr, xcoef, g, e, bi) in SI1:
            if wno >= thr:
                if xcoef is None:
                    x = 0.0
                else:
                    c0, w0, p0 = xcoef
                    x = c0 * (w0 / wno) ** p0
                a = x * g * np.exp(-e * hckt) * (bsi1[:, bi] - bsi2_1 * ehv)
                h = h + a
        asi1[:, j] = h * xnfpsi / rho

    # FE1OP: lowest transition edge fe1_wno=21000; at 500-510 nm (wno ~ 19600-19980)
    #   the FE1OP loop is skipped (wno<21000), so afe1 stays exactly 0.

    # ── LUKEOP ── (only Si II contributes a small constant at 500-510 nm)
    if ifop[9] == 1:
        xnfpn = pops["xnfpn"]; xnfpo = pops["xnfpo"]
        xnfpmg2 = pops["xnfpmg2"]; xnfpsi2 = pops["xnfpsi2"]; xnfpca2 = pops["xnfpca2"]
        for j in range(nfreq):
            f = freq[j]; freqlg = np.log(f); st = stim[:, j]
            n1op = np.zeros(n_layers); o1op = 0.0; mg2op = np.zeros(n_layers); ca2op = np.zeros(n_layers)
            si2op = si2op_vectorized(f, freqlg, temp, tlog)
            aluke[:, j] = (n1op * xnfpn + o1op * xnfpo + mg2op * xnfpmg2
                           + si2op * xnfpsi2 + ca2op * xnfpca2) * st / rho

    # ── HOTOP ── free-free + bound-free from the transition table
    if ifop[10] == 1:
        sumqq = pops["xnf_sumqq"]          # (n_layers,5)
        hotop_xnfp = pops["hotop_xnfp"]    # (n_layers,21)
        sqrt_t = np.sqrt(np.maximum(temp, 1e-30))
        exp_hot = np.exp(-HOTOP_TRANSITIONS[:, 5][None, :] / np.maximum(tkev[:, None], 1e-30))
        hid = np.clip(HOTOP_TRANSITIONS[:, 6].astype(np.int64) - 1, 0, 20)
        freqlg = np.log(freq)
        free = np.zeros((n_layers, nfreq))
        for q in range(1, 6):
            free += coulff(q, freq, freqlg, temp[:, None], tlog[:, None]) * sumqq[:, q - 1][:, None]
        ahot_v = free * (3.6919e8 / (freq[None, :] ** 3)) * (xne[:, None] / sqrt_t[:, None])
        for k in range(HOTOP_TRANSITIONS.shape[0]):
            f0, xs0, al0, pw0, mu0, _, _ = HOTOP_TRANSITIONS[k]
            use = freq >= f0
            if not np.any(use):
                continue
            ratio = f0 / freq[use]
            xsect = xs0 * (al0 + ratio - al0 * ratio) * np.sqrt(ratio ** int(pw0))
            xx = xsect[None, :] * hotop_xnfp[:, hid[k]][:, None] * mu0
            thr = ahot_v[:, use] / 100.0
            ahot_v[:, use] += np.where(xx > thr, xx * exp_hot[:, k][:, None], 0.0)
        ahot = ahot_v * stim / rho[:, None]

    acont = (ah2p + ahemin + ahot + aluke + ahyd + ahmin + ahe1 + ahe2
             + ac1 + amg1 + aal1 + asi1 + afe1)
    sigmac = sigh + sighe + sigel + sigh2
    return acont, sigmac


# ──────────────────────────────────────────────────────────────────────────────
# Assemble inputs, run physics at edge triplets, interpolate, compare to diag
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    a = np.load(ROOT / "reference/atmosphere.npz", allow_pickle=True)
    d = np.load(ROOT / "reference/diag.npz")
    wl = d["wavelength"]
    cabs_ref = d["continuum_absorption"]; cscat_ref = d["continuum_scattering"]
    n_layers = cabs_ref.shape[0]

    # IFOP from the .atm OPACITY line (HERAOP, H2RAOP, LUKEOP, HOTOP, HEMIOP all ON)
    ifop = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]

    # ── edge frequency grid (3 freqs per edge) ──
    frqedg = a["frqedg"]; wledge_signed = a["wledge"]
    n_edges = frqedg.size
    freqset = np.empty(3 * (n_edges - 1))
    for i in range(n_edges - 1):
        freqset[3 * i] = abs(frqedg[i]) / 1.0000001
        freqset[3 * i + 1] = C_LIGHT_NM / ((abs(wledge_signed[i]) + abs(wledge_signed[i + 1])) / 2.0)
        freqset[3 * i + 2] = abs(frqedg[i + 1]) * 1.0000001

    # ── locate the edge(s) the 500-510 nm grid uses (only edge 185 here) ──
    wledge = np.abs(wledge_signed)
    half_edge = a["half_edge"]; delta_edge = a["delta_edge"]
    edge_idx = np.clip(np.searchsorted(wledge, np.abs(wl), side="right") - 1, 0, wledge.size - 2)
    used_edges = np.unique(edge_idx)

    # ── populations (POPS mode-11 and mode-12) ──
    pop = a["population_per_ion"]  # (80,6,139): [layer, ion_stage, element]
    pops = dict(
        temperature=a["temperature"], mass_density=a["mass_density"],
        electron_density=a["electron_density"],
        xnfph=a["xnfph"], xnf_h=a["xnf_h"],
        he1_mode11=pop[:, 0, 1], he2_mode11=pop[:, 1, 1], he3_mode11=pop[:, 2, 1],
        he1_mode12=a["xnf_he1"], he2_mode12=a["xnf_he2"],
        xnfpc=a["xnfpc"], xnfpmg=a["xnfpmg"], xnfpal=a["xnfpal"],
        xnfpsi=a["xnfpsi"], xnfpfe=a["xnfpfe"],
        xnfpn=pop[:, 0, 6], xnfpo=pop[:, 0, 7],
        xnfpmg2=pop[:, 1, 11], xnfpsi2=pop[:, 1, 13], xnfpca2=pop[:, 1, 19],
    )
    # HOTOP population vectors (Fortran POPS layout)
    hotop_xnfp = np.zeros((n_layers, 21))
    hotop_xnfp[:, 0:4] = pop[:, 0:4, 5]
    hotop_xnfp[:, 4:9] = pop[:, 0:5, 6]
    hotop_xnfp[:, 9:15] = pop[:, 0:6, 7]
    hotop_xnfp[:, 15:21] = pop[:, 0:6, 9]
    xnf_sumqq = np.zeros((n_layers, 5))
    for elem in (5, 6, 7, 9, 11, 13, 15, 25):
        for iz in range(1, 6):
            xnf_sumqq[:, iz - 1] += (iz * iz) * pop[:, iz, elem]
    pops["hotop_xnfp"] = hotop_xnfp
    pops["xnf_sumqq"] = xnf_sumqq

    # ── compute KAPP physics only at the freqsets of the used edges (here edge 185) ──
    sel = np.concatenate([[3 * e, 3 * e + 1, 3 * e + 2] for e in used_edges])
    acont_sel, sigmac_sel = compute_kapp(freqset[sel], pops, ifop)

    # ── build per-edge log10 coefficients (only the used edges are filled) ──
    cabs_coeff = np.zeros((n_layers, wledge.size - 1, 3))
    cscat_coeff = np.zeros((n_layers, wledge.size - 1, 3))
    for k, e in enumerate(used_edges):
        cabs_coeff[:, e, :] = np.log10(np.maximum(acont_sel[:, 3 * k:3 * k + 3], 1e-30))
        cscat_coeff[:, e, :] = np.log10(np.maximum(sigmac_sel[:, 3 * k:3 * k + 3], 1e-30))

    # ── locked 3-point Lagrange interpolation in wavelength, then 10** ──
    absorption = np.zeros((n_layers, wl.size))
    scattering = np.zeros((n_layers, wl.size))
    for e in range(wledge.size - 1):
        m = edge_idx == e
        if not np.any(m):
            continue
        w = wl[m]; wl_l = wledge[e]; wl_r = wledge[e + 1]
        half = half_edge[e]; delta = delta_edge[e] if delta_edge[e] != 0.0 else 1e-20
        c1 = (w - half) * (w - wl_r) / delta
        c2 = (wl_l - w) * (w - wl_r) * 2.0 / delta
        c3 = (w - wl_l) * (w - half) / delta
        la = (cabs_coeff[:, e, 0][:, None] * c1[None, :]
              + cabs_coeff[:, e, 1][:, None] * c2[None, :]
              + cabs_coeff[:, e, 2][:, None] * c3[None, :])
        ls = (cscat_coeff[:, e, 0][:, None] * c1[None, :]
              + cscat_coeff[:, e, 1][:, None] * c2[None, :]
              + cscat_coeff[:, e, 2][:, None] * c3[None, :])
        absorption[:, m] = 10.0 ** la
        scattering[:, m] = 10.0 ** ls

    # ── compare to diag ──
    def rel(x, y):
        msk = np.abs(y) > 0
        return np.abs(x[msk] - y[msk]) / np.abs(y[msk])

    ra = rel(absorption, cabs_ref); rs = rel(scattering, cscat_ref)
    print(f"continuum_absorption : max rel = {ra.max():.3e}   median = {np.median(ra):.3e}")
    print(f"continuum_scattering : max rel = {rs.max():.3e}   median = {np.median(rs):.3e}")

    # where is the worst absorption error? (expected: deepest hot layers, not the photosphere)
    err = np.abs(absorption - cabs_ref) / np.maximum(np.abs(cabs_ref), 1e-300)
    li, wi = np.unravel_index(np.argmax(err), err.shape)
    T = a["temperature"]
    print(f"  worst abs error at layer {li} (T={T[li]:.0f} K), wl={wl[wi]:.3f} nm")
    cool = T < 8000.0
    print(f"  max abs rel over photosphere (T<8000 K, {cool.sum()} layers) = "
          f"{(np.abs(absorption[cool] - cabs_ref[cool]) / np.abs(cabs_ref[cool])).max():.3e}")

    TABLES_NEEDED = [
        "karsas_tables.npz: FREQ_LOG, XN_LOG, XL_LOG_ARRAY, EKARSAS",
        "kapp_continuum_tables.npz: HMINOP_WBF, HMINOP_BF, HMINOP_WAVEK, HMINOP_THETAFF,",
        "    HMINOP_FFBEG, HMINOP_FFEND, HRAYOP_GAVRILAM, HRAYOP_GAVRILAMAB, HRAYOP_GAVRILAMBC,",
        "    HRAYOP_GAVRILAMCD, HRAYOP_GAVRILALYMANCONT, HRAYOP_FGAVRILALYMANCONT, COULFF_Z4LOG,",
        "    COULFF_A_TABLE, HOTOP_TRANSITIONS, _SI2OP_PEACH, _SI2OP_FREQSI, _SI2OP_FLOG,",
        "    _SI2OP_TLG, H_ENERGY_CM, H_STAT_WEIGHT",
        "atmosphere.npz (EOS): temperature, mass_density, electron_density, xnfph, xnf_h,",
        "    xnf_he1, xnf_he2, xnfpc, xnfpmg, xnfpal, xnfpsi, xnfpfe, population_per_ion,",
        "    frqedg, wledge, half_edge, delta_edge",
    ]
    print("\nTable arrays needed to ship:")
    for t in TABLES_NEEDED:
        print("  " + t)

    # ── per-term diagnostic at a representative photosphere layer (tau~1, T~6300 K) ──
    layer = int(np.argmin(np.abs(T - 6300.0)))
    terms = {}
    import sys as _sys

    def _grab(frame, ev, arg):
        if ev == "return" and frame.f_code.co_name == "compute_kapp":
            for nm in ("ahyd", "ahmin", "ah2p", "ahemin", "ahe1", "ahe2", "ac1",
                       "amg1", "aal1", "asi1", "afe1", "aluke", "ahot",
                       "sigh", "sighe", "sigel", "sigh2"):
                if nm in frame.f_locals:
                    terms[nm] = np.array(frame.f_locals[nm])
        return _grab

    _sys.settrace(_grab)
    ac_dbg, sg_dbg = compute_kapp(freqset[sel[:3]], pops, ifop)
    _sys.settrace(None)
    print(f"\nPer-term contributions at layer {layer} (T={T[layer]:.0f} K), freq1 (cm^2/g):")
    tot_a = ac_dbg[layer, 0]
    for nm in ("ahyd", "ahmin", "ah2p", "ahemin", "ahe1", "ahe2",
               "ac1", "amg1", "aal1", "asi1", "afe1", "aluke", "ahot"):
        v = terms[nm][layer, 0]
        print(f"  ABS {nm:8s} = {v:.6e}  ({100 * v / tot_a:7.3f}%)")
    tot_s = sg_dbg[layer, 0]
    for nm in ("sigh", "sighe", "sigel", "sigh2"):
        v = terms[nm][layer, 0]
        print(f"  SCAT {nm:7s} = {v:.6e}  ({100 * v / tot_s:7.3f}%)")
