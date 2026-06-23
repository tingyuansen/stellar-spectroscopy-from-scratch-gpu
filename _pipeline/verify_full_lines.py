#!/usr/bin/env python
"""From-scratch NumPy reproduction of pykurucz's FULL line opacity, verified to
machine precision against reference/diag.npz["line_opacity"] (the ground-truth
float64 array; line_scattering is all zeros and fscat==0, so line_opacity is the
sum of three contributions):

    line_opacity = ASYNTH_metal(all Z>=3)        # metal Voigt kernel (TRANSP + wings)
                 + ahline(H, HPROF4)              # hydrogen linear-Stark wing engine
                 + helium_wings * STIM(He)        # helium Voigt-batch wing kernel

The metal part is the same recipe verified bit-exactly in verify_lines.py, here
extended from the Z<=30 subset to *all* metal lines (type-0, Z>=3, including the
754 in-window Z>30 lines).  The hydrogen part is a clean port of pykurucz's
HPROF4 / HLINOP linear-Stark line-broadening engine (synthe_py/physics/
hydrogen_jit.py).  The helium part is a port of the helium Voigt-batch wing
walk (synthe_py/physics/voigt_jit.accumulate_voigt_wings_and_source_jit) with the
fort.19 continuum-limit tapering.

No pykurucz import — only NumPy and the reference data files:
  reference/diag.npz            ground-truth line_opacity, wavelength, continua
  reference/atmosphere.npz      depth-state inputs (T, ne, xnf_*, populations,...)
  reference/L4.npz              Voigt Harris tables h0tab/h1tab/h2tab
  reference/full_lines_data.npz full atomic catalog (all Z incl >30, H, He),
                                hydrogen HPROF4 Stark tables, He kernel inputs,
                                and per-component ground truth (gt_ahline,
                                gt_helium_wings) for the per-component diagnostic.

Source faithfully ported from:
  synthe_py/physics/line_opacity.py    (metal TRANSP + ASYNTH wing kernel)
  synthe_py/physics/voigt_jit.py       (Voigt H(a,v); helium wing walk)
  synthe_py/physics/hydrogen_jit.py     (HPROF4 hydrogen linear-Stark profile)
  synthe_py/engine/opacity.py          (line routing; per-depth hydrogen state;
                                        wcon/wtail; STIM; component summation)
  synthe_py/physics/populations.py     (_hydrogen_state depth fields)
"""
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"
sys.path.insert(0, str(ROOT / "_pipeline"))

# Reuse the proven metal kernel + helpers from verify_lines.py.
import verify_lines as VL

C_LIGHT_NM = 2.99792458e17  # nm/s
H_PLANCK = 6.62607015e-27   # erg*s
K_BOLTZ = 1.380649e-16      # erg/K
CGF_CONSTANT = 0.026538 / 1.77245
CUTOFF = 1e-3
KAPMIN_FLOOR = 1e-8

# ── hydrogen engine constants (synthe_py/physics/hydrogen_jit.py) ───────────
RYDH = 3.2880515e15          # Hz  (Rydberg frequency)
C_LIGHT_AA = 2.99792458e18   # Å/s
C_LIGHT_CM = 2.99792458e10   # cm/s
LYMAN_ALPHA_CENTER_WN = 82259.10  # cm^-1
# EHYD levels (cm^-1) for n=1..8, with the n>=9 limit formula
_EHYD_CM = np.array([0.0, 82259.105, 97492.302, 102823.893, 105291.651,
                     106632.160, 107440.444, 107965.051], dtype=np.float64)
_HYD_RYD_CM = 109677.576   # cm^-1
_HYD_EINF_CM = 109678.764  # cm^-1


# ════════════════════════════════════════════════════════════════════════════
#  METAL: all-Z ASYNTH (same kernel as verify_lines, no Z<=30 filter)
# ════════════════════════════════════════════════════════════════════════════
def compute_metal_opacity(cat, atm, diag, L4):
    """All metal (type-0, Z>=3) lines through the ASYNTH Voigt kernel."""
    wl = cat["cat_wl"]
    loggf = cat["cat_loggf"]
    Elow = cat["cat_elow"]
    Z = cat["cat_Z"].astype(np.int64)
    ion = cat["cat_ion"].astype(np.int64)
    line_type = cat["cat_line_types"].astype(np.int64)
    grad = cat["cat_grad"]; gstark = cat["cat_gstark"]; gvdw = cat["cat_gvdw"]
    grid = diag["wavelength"]
    h0tab = L4["h0tab"]; h1tab = L4["h1tab"]; h2tab = L4["h2tab"]

    pop3 = atm["population_per_ion"]
    dop3 = atm["doppler_per_ion"]
    rho = atm["mass_density"]
    xne = atm["electron_density"]
    T = atm["temperature"]
    if "txnxn" in atm.files:
        txnxn = atm["txnxn"]
    else:
        txnxn = (atm["xnf_h"] + 0.42 * atm["xnf_he1"] + 0.85 * atm["xnf_h2"]) * (T / 1e4) ** 0.3

    cont = diag["continuum_absorption"] + diag["continuum_scattering"]

    n_depths = pop3.shape[0]
    n_lines = wl.size
    n_w = grid.size
    asynth = np.zeros((n_depths, n_w), dtype=np.float64)

    gf = 10.0 ** loggf
    freq_hz = C_LIGHT_NM / wl
    cgf = CGF_CONSTANT * gf / freq_hz

    elem_idx = Z - 1
    ion_idx = ion - 1
    # index_wavelength is what pykurucz uses for the metal center index.
    center_idx = VL.nearest_grid_indices(grid, cat["cat_index_wl"])
    wing_idx = VL.nearest_grid_indices_raw(grid, cat["cat_index_wl"], origin_start=float(grid[0]))

    ratio = grid[1] / grid[0]
    resolu = 1.0 / (ratio - 1.0) if ratio > 1.0 else 300000.0

    hckt = atm["hckt"]
    n_ion_max = pop3.shape[1]; n_elem_max = pop3.shape[2]
    # type-0 metal lines (all Z>=3; H/He are routed elsewhere)
    line_ok = (
        (line_type == 0)
        & (elem_idx >= 0) & (elem_idx < n_elem_max)
        & (ion_idx >= 0) & (ion_idx < n_ion_max)
    )

    boltz = VL.fast_ex_array(Elow[None, :] * hckt[:, None])
    center_valid = line_ok & (center_idx >= 0) & (center_idx < n_w)
    M = VL.MAX_PROFILE_STEPS
    wing_active = line_ok & (wing_idx >= -M) & (wing_idx <= n_w - 1 + M)

    for i in range(n_lines):
        if not line_ok[i]:
            continue
        wl_i = wl[i]; cgf_i = cgf[i]
        ci = int(center_idx[i]); wi = int(wing_idx[i])
        clamped = max(0, min(ci, n_w - 1))
        pop = pop3[:, ion_idx[i], elem_idx[i]]
        dop = dop3[:, ion_idx[i], elem_idx[i]]
        boltz_i = boltz[:, i]
        kapmin = cont[:, clamped] * CUTOFF

        good = (pop > 0.0) & (dop > 0.0) & (rho > 0.0)
        if not np.any(good):
            continue
        xnfdop = np.zeros(n_depths)
        xnfdop[good] = pop[good] / (rho[good] * dop[good])
        kappa0_pre = cgf_i * xnfdop
        post = kappa0_pre * boltz_i
        passcut = good & (kappa0_pre >= kapmin) & (post >= kapmin) & (post > 0.0)
        if not np.any(passcut):
            continue

        doppler_width = dop * wl_i
        dopple = np.where(wl_i > 0, doppler_width / wl_i, 1e-6)
        gamma_total = grad[i] + gstark[i] * xne + gvdw[i] * txnxn
        adamp = np.where((doppler_width > 0) & (dopple > 0), gamma_total / dopple, 0.0)
        kappa0 = post

        kapcen = np.zeros(n_depths)
        cd = passcut & (adamp >= 0.0) & (kappa0 > 0.0)
        for d in np.where(cd)[0]:
            ad = adamp[d]
            if ad < 0.2:
                kapcen[d] = kappa0[d] * (1.0 - 1.128 * ad)
            else:
                kapcen[d] = kappa0[d] * VL.voigt_profile(0.0, ad, h0tab, h1tab, h2tab)

        if center_valid[i]:
            for d in np.where(cd)[0]:
                asynth[d, ci] += kapcen[d]

        if not wing_active[i]:
            continue
        wing_pairs = cd & (kapcen > 0.0)
        if not np.any(wing_pairs):
            continue
        adamp_w = np.maximum(adamp, 1e-12)
        voigt_c = VL.voigt_h_at_zero(adamp_w, h0tab, h1tab, h2tab)
        kappa0_wing = np.where(kapcen > 0.0, kapcen / voigt_c, 0.0)
        ci_w = min(max(wi, 0), n_w - 1)
        kapmin_ref = np.maximum(cont[:, ci_w] * CUTOFF, cont[:, ci_w] * KAPMIN_FLOOR)
        for d in np.where(wing_pairs)[0]:
            VL.process_wing_pair(
                asynth[d], grid, wi, kappa0_wing[d], adamp_w[d],
                doppler_width[d], wl_i, kapmin_ref[d], True,
                resolu, h0tab, h1tab, h2tab,
            )

    # STIM applied last, to the whole array
    freq_grid = C_LIGHT_NM / grid
    hkt = H_PLANCK / (K_BOLTZ * T)
    stim_grid = 1.0 - np.exp(-freq_grid[None, :] * hkt[:, None])
    asynth *= stim_grid
    return asynth


# ════════════════════════════════════════════════════════════════════════════
#  HYDROGEN: HPROF4 / HLINOP linear-Stark line broadening engine
#  Pure-NumPy port of synthe_py/physics/hydrogen_jit.py (JIT/guards stripped).
# ════════════════════════════════════════════════════════════════════════════
import math

def _build_e1_table():
    values = np.empty(2000, dtype=np.float64)
    for i in range(2000):
        x = (i + 1) * 0.01
        values[i] = math.exp(-x) / x
    return values
_E1_TABLE = _build_e1_table()


def _vcse1f(x):
    if x <= 0.0:
        return 0.0
    if x <= 0.01:
        return -math.log(x) - 0.577215 + x
    if x <= 1.0:
        return (-math.log(x) - 0.57721566
                + x * (0.99999193 + x * (-0.24991055 + x * (0.05519968
                + x * (-0.00976004 + x * 0.00107857)))))
    if x > 30.0:
        return 0.0
    numerator = x * (x + 2.334733) + 0.25062
    denominator = (x * (x + 3.330657) + 1.681534) * x
    return numerator / denominator * math.exp(-x)


def _fast_ex(x):
    if x > 80.0:
        return 0.0
    return math.exp(-x)


def _hf_nm(n, m):
    if m <= n:
        return 0.0
    xn = float(n)
    ginf = 0.2027 / xn ** 0.71
    gca = 0.124 / xn
    fkn = xn * 1.9603
    wtc = 0.45 - 2.4 / xn ** 3 * (xn - 1.0)
    xm = float(m)
    xmn = xm - xn
    fk = fkn * (xm / (xmn * (xm + xn))) ** 3
    xmn12 = xmn ** 1.2
    wt = (xmn12 - 1.0) / (xmn12 + wtc)
    return fk * (1.0 - wt * ginf - (0.222 + gca / xm) * (1.0 - wt))


def _interpolate_cutoff(delta_wn, table, start, step):
    """Returns 1e300 sentinel when delta_wn > max_delta (== None in source)."""
    max_delta = start + step * (table.shape[0] - 1)
    if delta_wn > max_delta:
        return 1e300
    if delta_wn <= start:
        if table.shape[0] < 2:
            return table[0]
        frac = (delta_wn - start) / step
        return table[0] + (table[1] - table[0]) * frac
    position = (delta_wn - start) / step
    index = int(math.floor(position))
    frac = position - index
    if index >= table.shape[0] - 1:
        return float(table[-1])
    return float(table[index] + (table[index + 1] - table[index]) * frac)


def _sofbeta(beta, p, n, m, propbm, c_arr, d_arr, pp_arr, beta_arr):
    if beta <= 0.0:
        return 0.0
    b2 = beta * beta
    sb = math.sqrt(beta)
    corr = 1.0
    if beta <= 500.0:
        mmn = m - n
        if n <= 3 and mmn <= 2:
            indx = 2 * (n - 1) + mmn
        else:
            indx = 7
        if indx < 1:
            indx = 1
        if indx > 7:
            indx = 7
        im = min(int(5.0 * p) + 1, 4)
        if im < 1:
            im = 1
        ip = im + 1
        wtp = 5.0 * (p - pp_arr[im - 1])
        if wtp < 0.0:
            wtp = 0.0
        if wtp > 1.0:
            wtp = 1.0
        wtm = 1.0 - wtp
        if beta <= 25.12:
            j = int(np.searchsorted(beta_arr, beta))
            if j < 1:
                j = 1
            if j > beta_arr.shape[0] - 1:
                j = beta_arr.shape[0] - 1
            jm = j - 1
            jp = j
            denom = beta_arr[jp] - beta_arr[jm]
            if denom <= 0.0:
                wtb = 0.0
            else:
                wtb = (beta - beta_arr[jm]) / denom
            wtbm = 1.0 - wtb
            cbp = (propbm[indx - 1, ip - 1, jp] * wtp
                   + propbm[indx - 1, im - 1, jp] * wtm)
            cbm = (propbm[indx - 1, ip - 1, jm] * wtp
                   + propbm[indx - 1, im - 1, jm] * wtm)
            corr = 1.0 + cbp * wtb + cbm * wtbm
            wt = 0.5 * (10.0 - beta)
            if wt < 0.0:
                wt = 0.0
            if wt > 1.0:
                wt = 1.0
            pr1 = 0.0
            pr2 = 0.0
            if beta <= 10.0:
                pr1 = 8.0 / (83.0 + (2.0 + 0.95 * b2) * beta)
            if beta >= 8.0:
                pr2 = (1.5 / sb + 27.0 / b2) / b2
            return (pr1 * wt + pr2 * (1.0 - wt)) * corr
        cc = c_arr[im - 1, indx - 1] * wtp + c_arr[ip - 1, indx - 1] * wtm
        dd = d_arr[im - 1, indx - 1] * wtp + d_arr[ip - 1, indx - 1] * wtm
        denom2 = cc + beta * sb
        if denom2 == 0.0:
            denom2 = 1e-30
        corr = 1.0 + dd / denom2
    return (1.5 / sb + 27.0 / b2) / b2 * corr


def _lyman_alpha_lorentz(freq, freqnm, del_freq, dop, hwres, hwvdw, hwrad,
                         cutoff_h2, cutoff_h2_plus, xnfph_0, xnfph_1):
    if dop <= 0.0:
        return 0.0
    hwres_near = hwres * 4.0
    hwlor_near = hwres_near + hwvdw + hwrad
    hhw_near = freqnm * max(hwlor_near, 0.0)
    freq_threshold = (LYMAN_ALPHA_CENTER_WN - 4000.0) * C_LIGHT_CM
    wavenumber = freq / C_LIGHT_CM
    delta_wn = wavenumber - LYMAN_ALPHA_CENTER_WN
    if freq > freq_threshold and hhw_near > 0.0:
        hres_term = (hwres_near * freqnm / math.pi
                     / (del_freq * del_freq + hhw_near * hhw_near) * 1.77245 * dop)
        hhw_use = hhw_near
    else:
        cutoff_val = 0.0
        cutoff_log = _interpolate_cutoff(delta_wn, cutoff_h2, -22000.0, 200.0)
        if cutoff_log < 1e299 and xnfph_0 > 0.0:
            cutoff_val = (10.0 ** (cutoff_log - 14.0)) * xnfph_0 * 2.0 / C_LIGHT_CM
        hres_term = cutoff_val * 1.77245 * dop
        hwlor = hwres + hwvdw + hwrad
        hhw_use = freqnm * max(hwlor, 0.0)
    hrad_term = 0.0
    if hwrad > 0.0 and hhw_use > 0.0:
        freq_low = 2.4190611e15
        freq_high = 0.77 * RYDH
        if freq > freq_low and freq < freq_high:
            hrad_term = (hwrad * freqnm / math.pi
                         / (del_freq * del_freq + hhw_use * hhw_use) * 1.77245 * dop)
    hvdw_term = 0.0
    if hwvdw > 0.0 and hhw_use > 0.0:
        if freq >= 1.8e15:
            hvdw_term = (hwvdw * freqnm / math.pi
                         / (del_freq * del_freq + hhw_use * hhw_use) * 1.77245 * dop)
    return hres_term + hrad_term + hvdw_term


def _lyman_quasistatic_cutoff(freq, prqs, xnfph_0, xnfph_1, fo, dbeta, dop, n, m,
                              cutoff_h2_plus, propbm, c_arr, d_arr, pp_arr, beta_arr, pp_val):
    if fo <= 0.0:
        return 0.0
    wavenumber = freq / C_LIGHT_CM
    delta_wn = wavenumber - LYMAN_ALPHA_CENTER_WN
    if delta_wn < -20000.0:
        return 0.0
    extra = 0.0
    if delta_wn <= -4000.0:
        cutoff_log = _interpolate_cutoff(delta_wn, cutoff_h2_plus, -15000.0, 100.0)
        if cutoff_log < 1e299:
            cutoff_val = (10.0 ** (cutoff_log - 14.0)) * xnfph_1 / C_LIGHT_CM
            extra += cutoff_val * 1.77245 * dop
    else:
        fo_safe = max(fo, 1e-30)
        beta4000 = 4000.0 * C_LIGHT_CM / fo_safe * dbeta
        prqs4000 = _sofbeta(beta4000, pp_val, n, m, propbm, c_arr, d_arr, pp_arr, beta_arr) * 0.5
        normalization = prqs4000 / fo_safe * dbeta
        cutoff4000 = (10.0 ** (-11.07 - 14.0)) * xnfph_1 / C_LIGHT_CM
        if normalization > 0.0:
            extra += (cutoff4000 / normalization * (prqs / fo_safe * dbeta) * 1.77245 * dop)
    return extra


def _hydrogen_line_profile(n, m, delta_lambda_nm, hyd, tabs, fine_offsets, fine_weights, n_fine):
    """Port of _hydrogen_line_profile_jit. `hyd` is a dict of per-depth scalars;
    `tabs` is a dict of the HPROF4 Stark tables."""
    t3nhe = hyd["t3nhe"]; t3nh2 = hyd["t3nh2"]; fo = hyd["fo"]; dopph = hyd["dopph"]
    c1d = hyd["c1d"]; c2d = hyd["c2d"]; y1s = hyd["y1s"]; y1b = hyd["y1b"]
    gcon1 = hyd["gcon1"]; gcon2 = hyd["gcon2"]; pp_val = hyd["pp"]
    xnfph_0 = hyd["xnfph_0"]; xnfph_1 = hyd["xnfph_1"]; electron_density = hyd["ne"]
    asum = tabs["asum"]; asum_lyman = tabs["asum_lyman"]; y1wtm = tabs["y1wtm"]
    xknmtb = tabs["xknmtb"]; propbm = tabs["propbm"]; c_tbl = tabs["c"]; d_tbl = tabs["d"]
    pp_tbl = tabs["pp"]; beta_tbl = tabs["beta"]
    cutoff_h2_plus = tabs["cutoff_h2_plus"]; cutoff_h2 = tabs["cutoff_h2"]

    mmn = m - n
    if mmn <= 0:
        return 0.0
    xn = float(n); xm = float(m)
    xn2 = xn * xn; xm2 = xm * xm
    xm2mn2 = xm2 - xn2
    xmn2 = xm2 * xn2
    gnm = xm2mn2 / xmn2
    if n <= 4 and mmn <= 3 and n >= 1 and mmn >= 1:
        xknm = xknmtb[n - 1, mmn - 1]
    else:
        xknm = 5.5e-5 / gnm * xmn2 / (1.0 + 0.13 / float(mmn))
    freqnm = RYDH * gnm
    wavenm = C_LIGHT_AA / freqnm
    dbeta = C_LIGHT_AA / (freqnm * freqnm * xknm)
    c1con = xknm / wavenm * gnm * xm2mn2
    c2con = (xknm / wavenm) ** 2
    n_asum = asum.shape[0]; m_asum = asum.shape[0]
    if n <= n_asum and m <= m_asum:
        radamp = asum[n - 1] + asum[m - 1]
    elif n <= n_asum:
        radamp = asum[n - 1]
    elif m <= m_asum:
        radamp = asum[m - 1]
    else:
        radamp = 0.0
    n_asum_ly = asum_lyman.shape[0]
    if n == 1 and m <= n_asum_ly:
        radamp = asum_lyman[m - 1]
    radamp /= 12.5664
    radamp /= freqnm
    resont = _hf_nm(1, m) / xm / (1.0 - 1.0 / xm2)
    if n != 1:
        resont += _hf_nm(1, n) / xn / (1.0 - 1.0 / xn2)
    resont *= 3.579e-24 / gnm
    vdw = 4.45e-26 / gnm * (xm2 * (7.0 * xm2 + 5.0)) ** 0.4
    hwvdw = vdw * t3nhe + 2.0 * vdw * t3nh2
    hwrad = radamp
    stark = 1.6678e-18 * freqnm * xknm
    hwres = resont * xnfph_0 * 2.0
    hwstk = stark * fo
    hwlor = hwres + hwvdw + hwrad
    wl = wavenm + delta_lambda_nm * 10.0
    if wl <= 0.0:
        return 0.0
    freq = C_LIGHT_AA / wl
    del_freq = abs(freq - freqnm)
    dopph_safe = max(dopph, 1e-40)
    dop = freqnm * dopph_safe
    hfwidth = freqnm * max(dopph_safe, hwlor, hwstk)
    ifcore = del_freq <= hfwidth
    nwid = 1
    if not (dopph_safe >= hwstk and dopph_safe >= hwlor):
        nwid = 2
        if hwlor < hwstk:
            nwid = 3
    core = 0.0
    for fi in range(n_fine):
        component_freq = freqnm + fine_offsets[fi]
        dop_safe = max(dop, 1e-30)
        dd = abs(freq - component_freq) / dop_safe
        if dd <= 7.0:
            core += _fast_ex(dd * dd) * fine_weights[fi]
    lorentz = 0.0
    hhw = freqnm * hwlor
    if n == 1 and m == 2:
        lorentz = _lyman_alpha_lorentz(freq, freqnm, del_freq, dop, hwres, hwvdw, hwrad,
                                       cutoff_h2, cutoff_h2_plus, xnfph_0, xnfph_1)
    else:
        top = hhw
        if n == 1:
            freq_ratio = freq / RYDH
            if m == 3 and 0.885 <= freq_ratio <= 0.890:
                top = max(hhw - freqnm * hwrad, 0.0)
            elif m == 4 and 0.936 <= freq_ratio <= 0.938:
                top = max(hhw - freqnm * hwrad, 0.0)
            elif m == 5 and 0.959 <= freq_ratio <= 0.961:
                top = max(hhw - freqnm * hwrad, 0.0)
        if hhw > 0.0:
            lorentz = (top / math.pi / (del_freq * del_freq + hhw * hhw) * 1.77245 * dop)
    y1num = 320.0
    if m == 2:
        y1num = 550.0
    elif m == 3:
        y1num = 380.0
    y1wht = 1.0e13
    if mmn <= 3:
        y1wht = 1.0e14
    if mmn <= 2 and n <= 2 and n >= 1 and mmn >= 1:
        if n <= y1wtm.shape[0] and mmn <= y1wtm.shape[1]:
            y1wht = y1wtm[n - 1, mmn - 1]
    y1wht_safe = max(y1wht, 1e-30)
    elec_safe = max(electron_density, 0.0)
    wty1 = 1.0 / (1.0 + elec_safe / y1wht_safe)
    y1_scal = y1num * y1s * wty1 + y1b * (1.0 - wty1)
    c1 = c1d * c1con * y1_scal
    c2 = c2d * c2con
    fo_safe = max(fo, 1e-30)
    beta = del_freq / fo_safe * dbeta
    y1 = c1 * beta
    y2 = c2 * beta * beta
    g1 = 6.77 * math.sqrt(max(c1, 1e-30))
    ratio = 0.0
    if c1 > 0.0 and c2 > 0.0:
        ratio = math.sqrt(c2) / max(c1, 1e-30)
    log_term = 0.0
    if ratio > 0.0:
        log_term = math.log(max(ratio, 1e-30))
    gnot = g1 * max(0.0, 0.2114 + log_term) * (1.0 - gcon1 - gcon2)
    gamma = gnot
    if y2 > 1e-4 and y1 > 1e-5:
        gamma = (g1 * (0.5 * _fast_ex(min(80.0, y1)) + _vcse1f(y1) - 0.5 * _vcse1f(y2))
                 * (1.0 - gcon1 / (1.0 + (90.0 * y1) ** 3) - gcon2 / (1.0 + 2000.0 * y1)))
    f = 0.0
    if gamma > 0.0:
        f = gamma / math.pi / (gamma * gamma + beta * beta)
    prqs = _sofbeta(beta, pp_val, n, m, propbm, c_tbl, d_tbl, pp_tbl, beta_tbl)
    stark_extra = 0.0
    if m <= 2:
        prqs *= 0.5
        stark_extra = _lyman_quasistatic_cutoff(freq, prqs, xnfph_0, xnfph_1, fo, dbeta, dop,
                                                n, m, cutoff_h2_plus, propbm, c_tbl, d_tbl,
                                                pp_tbl, beta_tbl, pp_val)
    p1 = (0.9 * y1) ** 2
    fns = (p1 + 0.03 * math.sqrt(max(y1, 0.0))) / (p1 + 1.0)
    fo_safe2 = max(fo, 1e-30)
    stark_core = (prqs * (1.0 + fns) + f) / fo_safe2 * dbeta * 1.77245 * dop
    if ifcore:
        if nwid == 1:
            return max(core, 0.0)
        if nwid == 2:
            return max(lorentz, 0.0)
        return max(stark_core + stark_extra, 0.0)
    return max(core + lorentz + stark_core + stark_extra, 0.0)


def _accumulate_hyd_line_depth(buffer, continuum_row, stim_row, grid, center_index,
                               line_wavelength, kappa0, n_lower, n_upper, wcon, wtail,
                               wlminus1, wlminus2, wlplus1, wlplus2, redcut, bluecut, cutoff,
                               hyd, tabs, fine_offsets, fine_weights, n_fine):
    n_points = buffer.shape[0]
    simple_wings = n_upper <= n_lower + 2
    use_taper = (not simple_wings) and (wtail > wcon)
    upper_minus2 = max(n_upper - 2, n_lower + 1)
    upper_plus2 = n_upper + 2

    red_active = True
    blue_active = True
    offset = 1
    max_steps = center_index
    tmp = n_points - center_index - 1
    if tmp > max_steps:
        max_steps = tmp

    def prof(nl, nu, dl):
        return _hydrogen_line_profile(nl, nu, dl, hyd, tabs, fine_offsets, fine_weights, n_fine)

    if 0 <= center_index < n_points:
        wave_center = grid[center_index]
        if not (not simple_wings and wave_center < wcon):
            delta_center_nm = wave_center - line_wavelength
            profile_center = kappa0 * prof(n_lower, n_upper, delta_center_nm)
            value_center = profile_center * stim_row[center_index]
            if use_taper and wave_center < wtail:
                value_center *= (wave_center - wcon) / (wtail - wcon)
            if value_center >= continuum_row[center_index] * cutoff:
                buffer[center_index] += value_center
    else:
        if center_index >= n_points:
            red_active = False
            offset = max(1, center_index - (n_points - 1))
        else:
            blue_active = False
            offset = max(1, -center_index)

    while offset <= max_steps and (red_active or blue_active):
        if red_active:
            idx = center_index + offset
            if idx >= n_points:
                red_active = False
            else:
                wave = grid[idx]
                if not simple_wings:
                    if wave > wlminus1:
                        red_active = False
                    elif wave < wcon:
                        pass
                    else:
                        delta_nm = wave - line_wavelength
                        value = kappa0 * prof(n_lower, n_upper, delta_nm) * stim_row[idx]
                        if use_taper and wave < wtail:
                            value *= (wave - wcon) / (wtail - wcon)
                        if wave > redcut:
                            delta_minus2 = wave - wlminus2
                            value_minus2 = kappa0 * prof(n_lower, upper_minus2, delta_minus2) * stim_row[idx]
                            if use_taper and wave < wtail:
                                value_minus2 *= (wave - wcon) / (wtail - wcon)
                            if value_minus2 >= value:
                                red_active = False
                                value = 0.0
                        if value <= 0.0 or value < continuum_row[idx] * cutoff:
                            red_active = False
                        else:
                            buffer[idx] += value
                else:
                    delta_nm = wave - line_wavelength
                    value = kappa0 * prof(n_lower, n_upper, delta_nm) * stim_row[idx]
                    if value <= 0.0 or value < continuum_row[idx] * cutoff:
                        red_active = False
                    else:
                        buffer[idx] += value
        if blue_active:
            idx = center_index - offset
            if idx < 0:
                blue_active = False
            else:
                wave = grid[idx]
                if not simple_wings and (wave < wcon or wave < wlplus1):
                    blue_active = False
                else:
                    delta_nm = wave - line_wavelength
                    value = kappa0 * prof(n_lower, n_upper, delta_nm) * stim_row[idx]
                    if not simple_wings:
                        if use_taper and wave < wtail:
                            value *= (wave - wcon) / (wtail - wcon)
                        if wave < bluecut:
                            delta_plus2 = wave - wlplus2
                            value_plus2 = kappa0 * prof(n_lower, upper_plus2, delta_plus2) * stim_row[idx]
                            if use_taper and wave < wtail:
                                value_plus2 *= (wave - wcon) / (wtail - wcon)
                            if value_plus2 >= value:
                                blue_active = False
                                value = 0.0
                    if value <= 0.0 or value < continuum_row[idx] * cutoff:
                        blue_active = False
                    else:
                        buffer[idx] += value
        offset += 1


def _ehyd_cm(n):
    if n <= 0:
        return 0.0
    idx = n - 1
    if 0 <= idx < _EHYD_CM.size:
        return float(_EHYD_CM[idx])
    return _HYD_EINF_CM - _HYD_RYD_CM / float(n * n)


def compute_hydrogen_opacity(cat, atm, diag, L4):
    """Hydrogen line opacity (HPROF4) — full from-scratch port of the JIT engine."""
    grid = diag["wavelength"]
    cont = diag["continuum_absorption"] + diag["continuum_scattering"]
    T = atm["temperature"]

    # ── per-depth hydrogen state (populations._hydrogen_state + dopph fallback) ──
    KBOLTZ = 1.380649e-16; AMU = 1.66054e-24
    C_LIGHT_CMS = 2.99792458e10; C_LIGHT_KMS = 299792.458
    MASS_H = 1.008
    xne = np.maximum(atm["electron_density"], 1e-40)
    xnf_he1 = atm["xnf_he1"]; xnf_h2 = atm["xnf_h2"]
    xnfph = atm["xnfph"]
    vturb_cms = atm["turbulent_velocity"]   # cm/s
    n_depths = T.size

    # STIM grid (frequency quantities)
    freq_grid = C_LIGHT_NM / grid
    hkt_vec = H_PLANCK / (K_BOLTZ * T)
    ehvkt = np.exp(-freq_grid[None, :] * hkt_vec[:, None])
    stim = 1.0 - ehvkt

    # continuum-merge inputs (engine: inglis/nmerge/emerge_h, conth)
    inglis = 1600.0 / np.power(xne, 2.0 / 15.0)
    nmerge = np.maximum(inglis - 1.5, 1.0)
    emerge_h = _HYD_RYD_CM / np.maximum(nmerge * nmerge, 1e-12)
    conth = cat["conth"]

    tabs = dict(asum=cat["htab_asum"], asum_lyman=cat["htab_asum_lyman"],
                y1wtm=cat["htab_y1wtm"], xknmtb=cat["htab_xknmtb"], propbm=cat["htab_propbm"],
                c=cat["htab_c"], d=cat["htab_d"], pp=cat["htab_pp"], beta=cat["htab_beta"],
                cutoff_h2_plus=cat["htab_cutoff_h2_plus"], cutoff_h2=cat["htab_cutoff_h2"])

    # fine-structure lookup keyed by (nl,nu)
    fkeys = cat["fine_keys"]; foff = cat["fine_offsets"]; fwt = cat["fine_weights"]; fn = cat["fine_n"]
    fine_map = {}
    for j in range(fkeys.shape[0]):
        fine_map[(int(fkeys[j, 0]), int(fkeys[j, 1]))] = (foff[j], fwt[j], int(fn[j]))

    # ── select the H lines (type -1/-2, ion 1) ──
    lt = cat["cat_line_types"].astype(np.int64)
    ion = cat["cat_ion"].astype(np.int64)
    Z = cat["cat_Z"].astype(np.int64)
    hmask = np.isin(lt, [-1, -2]) & (ion == 1)
    hidx = np.where(hmask)[0]

    wl_all = cat["cat_wl"]; idxwl_all = cat["cat_index_wl"]; gf_all = cat["cat_gf"]
    nl_all = cat["cat_n_lower"].astype(np.int64); nu_all = cat["cat_n_upper"].astype(np.int64)
    elow_all = cat["cat_elow"]; hckt = atm["hckt"]

    pop3 = atm["population_per_ion"]; dop3 = atm["doppler_per_ion"]
    rho = atm["mass_density"]
    h_index = 0   # H is Z=1 -> element index 0
    pop_densities = pop3[:, :, h_index]   # (80,6)
    dop_velocity = dop3[:, :, h_index]

    center_idx_all = VL.nearest_grid_indices(grid, idxwl_all)

    ahline = np.zeros((n_depths, grid.size), dtype=np.float64)
    n_w = grid.size

    for line_idx in hidx:
        line_wavelength = float(wl_all[line_idx])
        center_idx = int(center_idx_all[line_idx])
        gf_linear = float(gf_all[line_idx])
        freq_hz = C_LIGHT_NM / line_wavelength
        cgf = CGF_CONSTANT * gf_linear / freq_hz
        nl = max(int(nl_all[line_idx]), 1)
        nu = max(int(nu_all[line_idx]), nl + 1)
        nl_eff = nl; nu_eff = nu
        ehyd_lower = _ehyd_cm(nl_eff)
        wlminus1 = (1.0e7 / (_ehyd_cm(nu_eff - 1) - ehyd_lower)) if nu_eff - 1 > nl_eff else line_wavelength
        wlminus2 = (1.0e7 / (_ehyd_cm(nu_eff - 2) - ehyd_lower)) if nu_eff - 2 > nl_eff else line_wavelength
        wlplus1 = 1.0e7 / (_ehyd_cm(nu_eff + 1) - ehyd_lower)
        wlplus2 = 1.0e7 / (_ehyd_cm(nu_eff + 2) - ehyd_lower)
        redcut = 1.0e7 / (conth[0] - _HYD_RYD_CM / (float(nu_eff) - 0.8) ** 2 - ehyd_lower)
        bluecut = 1.0e7 / (conth[0] - _HYD_RYD_CM / (float(nu_eff) + 0.8) ** 2 - ehyd_lower)
        ncon_idx = max(1, min(nl_eff, conth.size)) - 1
        conth_val = float(conth[ncon_idx])
        wshift = 1.0e7 / (conth_val - _HYD_RYD_CM / 81.0 ** 2)

        off, wt, n_fine = fine_map.get((nl_eff, nu_eff), (np.zeros(1), np.zeros(1), 0))

        # boltzmann factor per depth (FASTEX of Elow*hckt)
        boltz_line = VL.fast_ex_array(elow_all[line_idx] * hckt)

        for di in range(n_depths):
            pop_val = pop_densities[di, 0]
            dop_val = dop_velocity[di, 0]
            rho_d = float(rho[di])
            if pop_val <= 0.0 or dop_val <= 0.0 or rho_d <= 0.0:
                continue
            xnfdop = pop_val / (rho_d * dop_val)
            clamped = max(0, min(center_idx, n_w - 1))
            kapmin = cont[di, clamped] * CUTOFF
            kappa0_pre = cgf * xnfdop
            if kappa0_pre < kapmin:
                continue
            kappa0 = kappa0_pre * boltz_line[di]
            if kappa0 < kapmin:
                continue

            # per-depth hydrogen state
            temp = max(float(T[di]), 1.0)
            ne_d = float(xne[di])
            xne16 = ne_d ** (1.0 / 6.0)
            pp = xne16 * 0.08989 / math.sqrt(temp)
            fo = (xne16 ** 4) * 1.25e-9
            y1b = 2.0 / (1.0 + 0.012 / temp * math.sqrt(ne_d / temp))
            t4 = temp / 10000.0
            t43 = t4 ** 0.3
            y1s = t43 / xne16
            t3nhe = t43 * float(xnf_he1[di])
            t3nh2 = t43 * float(xnf_h2[di])
            c1d = fo * 78940.0 / temp
            c2d = (fo ** 2) / 5.96e-23 / ne_d
            gcon1 = 0.2 + 0.09 * math.sqrt(max(t4, 1e-12)) / (1.0 + ne_d / 1.0e13)
            gcon2 = 0.2 / (1.0 + ne_d / 1.0e15)
            # dopph fallback (no atmosphere.dopph): H thermal + turbulence
            thermal_vel_h = math.sqrt(2.0 * KBOLTZ * temp / (MASS_H * AMU)) / C_LIGHT_CMS
            vturb_model = (float(vturb_cms[di]) / 1e5) / C_LIGHT_KMS
            total_turb = vturb_model   # microturb_kms = 0
            dopph = math.sqrt(thermal_vel_h * thermal_vel_h + total_turb * total_turb)

            hyd = dict(t3nhe=t3nhe, t3nh2=t3nh2, fo=fo, dopph=dopph, c1d=c1d, c2d=c2d,
                       y1s=y1s, y1b=y1b, gcon1=gcon1, gcon2=gcon2, pp=pp,
                       xnfph_0=float(xnfph[di, 0]) if xnfph.shape[1] > 0 else 0.0,
                       xnfph_1=float(xnfph[di, 1]) if xnfph.shape[1] > 1 else 0.0,
                       ne=ne_d)

            # wcon/wtail (engine: per depth)
            denom = conth_val - emerge_h[di]
            wmerge = 1.0e7 / denom if denom > 0.0 else -1.0
            if wmerge < 0.0:
                wmerge = wshift + wshift
            wcon = max(wshift, wmerge)
            if wcon > 0.0:
                inner = 1.0e7 / wcon - 500.0
                wtail = 1.0e7 / inner if inner > 0.0 else wcon + wcon
            else:
                wtail = wcon + wcon
            wcon = min(wshift + wshift, wcon)
            if wtail < 0.0:
                wtail = wcon + wcon
            wtail = min(wcon + wcon, wtail)

            _accumulate_hyd_line_depth(
                ahline[di], cont[di], stim[di], grid, center_idx, line_wavelength,
                kappa0, nl_eff, nu_eff, wcon, wtail, wlminus1, wlminus2, wlplus1, wlplus2,
                redcut, bluecut, CUTOFF, hyd, tabs, off, wt, n_fine,
            )

    return ahline


# ════════════════════════════════════════════════════════════════════════════
#  HELIUM: Voigt-batch wing walk (voigt_jit.accumulate_voigt_wings_and_source_jit)
# ════════════════════════════════════════════════════════════════════════════
def _voigt_hav(x_val, adamp, h0tab, h1tab, h2tab):
    """Voigt H(a,v) exactly as in the helium kernel (voigt_jit, fastmath stripped)."""
    iv = int(x_val * 200.0 + 0.5)
    if iv > h0tab.shape[0] - 1:
        iv = h0tab.shape[0] - 1
    if adamp < 0.2:
        if x_val > 10.0:
            return 0.5642 * adamp / (x_val * x_val)
        return (h2tab[iv] * adamp + h1tab[iv]) * adamp + h0tab[iv]
    elif adamp > 1.4 or (adamp + x_val) > 3.2:
        aa = adamp * adamp
        vv = x_val * x_val
        u = (aa + vv) * 1.4142
        voigt_val = adamp * 0.79788 / u
        if adamp <= 100.0:
            aau = aa / u; vvu = vv / u; uu = u * u
            voigt_val = (
                (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa)
                / uu + 1.0
            ) * voigt_val
        return voigt_val
    else:
        vv = x_val * x_val
        h0 = h0tab[iv]
        h1 = h1tab[iv] + h0 * 1.12838
        h2 = h2tab[iv] + h1 * 1.12838 - h0
        h3 = (1.0 - h2tab[iv]) * 0.37613 - h1 * 0.66667 * vv + h2 * 1.12838
        h4 = (3.0 * h3 - h1) * 0.37613 + h0 * 0.66667 * vv * vv
        poly_a = (((h4 * adamp + h3) * adamp + h2) * adamp + h1) * adamp + h0
        poly_b = ((-0.122727278 * adamp + 0.532770573) * adamp - 0.96284325) * adamp + 0.979895032
        return poly_a * poly_b


def _accumulate_helium_line(wings_row, continuum_row, grid, center_index,
                            line_wavelength, kappa_eff, doppler, adamp, cutoff,
                            has_wcon, wcon_val, has_wtail, wtail_val, base_wave,
                            h0tab, h1tab, h2tab):
    n_points = grid.shape[0]
    clamped = max(0, min(center_index, n_points - 1))
    # red wing (clamped .. end)
    if line_wavelength <= grid[n_points - 1]:
        for idx in range(clamped, n_points):
            wave = grid[idx]
            if has_wcon and wave <= wcon_val:
                continue
            x_val = abs(wave - line_wavelength) / doppler
            voigt_val = _voigt_hav(x_val, adamp, h0tab, h1tab, h2tab)
            value = kappa_eff * voigt_val
            if has_wtail and wave < wtail_val:
                denom = wtail_val - base_wave
                if denom < 1e-12:
                    denom = 1e-12
                value = value * (wave - base_wave) / denom
            if value < continuum_row[idx] * cutoff:
                break
            wings_row[idx] += value
    # blue wing (clamped-1 .. 0)
    if clamped > 0 and line_wavelength >= grid[0]:
        for idx in range(clamped - 1, -1, -1):
            wave = grid[idx]
            if has_wcon and wave <= wcon_val:
                continue
            x_val = abs(line_wavelength - wave) / doppler
            voigt_val = _voigt_hav(x_val, adamp, h0tab, h1tab, h2tab)
            value = kappa_eff * voigt_val
            if has_wtail and wave < wtail_val:
                denom = wtail_val - base_wave
                if denom < 1e-12:
                    denom = 1e-12
                value = value * (wave - base_wave) / denom
            if value < continuum_row[idx] * cutoff:
                break
            wings_row[idx] += value


def compute_helium_opacity(cat, atm, diag, L4):
    """Helium I/II wing opacity via the pre-STIM Voigt-batch walk; STIM applied last.

    kappa0/adamp/doppler/center are recomputed from scratch with the standard
    metal recipe (verified bit-exact against pykurucz); only the wcon/wtail
    continuum-merge limits are taken from the shipped fort.19 metadata
    (he_wcon_2d/he_wtail_2d) since they derive from the fort.19 NCON/NELIONX
    records, not from the line catalog itself.
    """
    grid = diag["wavelength"]
    cont = diag["continuum_absorption"] + diag["continuum_scattering"]
    T = atm["temperature"]
    h0tab = L4["h0tab"]; h1tab = L4["h1tab"]; h2tab = L4["h2tab"]

    pop3 = atm["population_per_ion"]; dop3 = atm["doppler_per_ion"]
    rho = atm["mass_density"]; xne = atm["electron_density"]; hckt = atm["hckt"]
    if "txnxn" in atm.files:
        txnxn = atm["txnxn"]
    else:
        txnxn = (atm["xnf_h"] + 0.42 * atm["xnf_he1"] + 0.85 * atm["xnf_h2"]) * (T / 1e4) ** 0.3

    lt = cat["cat_line_types"].astype(np.int64)
    hemask = np.isin(lt, [-3, -4, -6])
    hidx = np.where(hemask)[0]
    n_he = hidx.size
    n_depths = T.size; n_w = grid.size

    # fort.19-derived taper limits (shipped data), aligned with hidx ordering
    wcon_2d = cat["he_wcon_2d"]; wtail_2d = cat["he_wtail_2d"]
    ltc = cat["he_ltc"].astype(np.int64)
    cutoff = float(cat["he_cutoff"])

    Z = cat["cat_Z"].astype(np.int64); ion = cat["cat_ion"].astype(np.int64)
    wl_all = cat["cat_wl"]; idxwl_all = cat["cat_index_wl"]; gf_all = cat["cat_gf"]
    elow_all = cat["cat_elow"]
    grad = cat["cat_grad"]; gstark = cat["cat_gstark"]; gvdw = cat["cat_gvdw"]

    # build per-line kappa0/doppler/adamp/center from scratch
    k0_2d = np.zeros((n_depths, n_he)); dop_2d = np.zeros((n_depths, n_he))
    ad_2d = np.zeros((n_depths, n_he)); center_idx = np.zeros(n_he, dtype=np.int64)
    line_wl = np.zeros(n_he)
    for col, i in enumerate(hidx):
        Zi = int(Z[i]); ii = int(ion[i]) - 1; ei = Zi - 1
        wl_i = float(wl_all[i]); line_wl[col] = wl_i
        cgf = CGF_CONSTANT * float(gf_all[i]) / (C_LIGHT_NM / wl_i)
        pop = pop3[:, ii, ei]; dop = dop3[:, ii, ei]
        boltz = VL.fast_ex_array(elow_all[i] * hckt)
        ci = int(VL.nearest_grid_indices(grid, np.array([idxwl_all[i]]))[0])
        ci = max(0, min(ci, n_w - 1))
        center_idx[col] = ci
        valid = (pop > 0.0) & (dop > 0.0) & (rho > 0.0)
        dop_safe = np.maximum(dop, 1e-40)
        xnfdop = np.where(valid, pop / (rho * dop_safe), 0.0)
        k0pre = cgf * xnfdop
        kmin = cont[:, ci] * cutoff
        valid = valid & (k0pre >= kmin)
        k0 = k0pre * boltz
        valid = valid & (k0 >= kmin)
        gtot = grad[i] + gstark[i] * xne + gvdw[i] * txnxn
        adamp = gtot / dop_safe
        k0_2d[:, col] = np.where(valid, k0, 0.0)
        dop_2d[:, col] = np.where(valid, dop * wl_i, 0.0)
        ad_2d[:, col] = np.where(valid, adamp, 0.0)
    he_wings = np.zeros((n_depths, n_w), dtype=np.float64)

    for d in range(n_depths):
        for l in range(n_he):
            k0 = k0_2d[d, l]
            if k0 <= 0.0:
                continue
            dop = dop_2d[d, l]
            if dop <= 0.0:
                continue
            lw = line_wl[l]; ci = int(center_idx[l])
            if ltc[l] == -4:           # 3He branch
                keff = k0 / 1.155
                deff = dop * 1.155
                ad = ad_2d[d, l] / 1.155
            else:
                keff = k0; deff = dop; ad = ad_2d[d, l]
            ad = max(ad, 1e-12)
            wcon_v = wcon_2d[d, l]; wtail_v = wtail_2d[d, l]
            has_wcon = wcon_v > 0.0; has_wtail = wtail_v > 0.0
            base = wcon_v if has_wcon else lw
            _accumulate_helium_line(
                he_wings[d], cont[d], grid, ci, lw, keff, deff, ad, cutoff,
                has_wcon, wcon_v, has_wtail, wtail_v, base, h0tab, h1tab, h2tab,
            )

    freq_grid = C_LIGHT_NM / grid
    hkt = H_PLANCK / (K_BOLTZ * T)
    stim_grid = 1.0 - np.exp(-freq_grid[None, :] * hkt[:, None])
    return he_wings * stim_grid


def main():
    diag = np.load(REF / "diag.npz")
    atm = np.load(REF / "atmosphere.npz")
    L4 = np.load(REF / "L4.npz")
    cat = np.load(REF / "full_lines_data.npz", allow_pickle=True)
    ref = diag["line_opacity"].astype(np.float64)

    t0 = time.perf_counter()
    metal = compute_metal_opacity(cat, atm, diag, L4)
    dt = time.perf_counter() - t0
    print(f"metal runtime: {dt:.1f}s")

    # per-component diagnostic vs the metal-only residual of diag
    grid = diag["wavelength"]; T = atm["temperature"]
    freq_grid = C_LIGHT_NM / grid
    hkt = H_PLANCK / (K_BOLTZ * T)
    stim_grid = 1.0 - np.exp(-freq_grid[None, :] * hkt[:, None])
    gt_ahline = cat["gt_ahline"]
    gt_he = cat["gt_helium_wings"] * stim_grid
    metal_ref = ref - gt_ahline - gt_he
    # Restrict to grid points where the metal term dominates (elsewhere metal_ref is
    # a tiny difference of two large H/He numbers, so its "rel error" is pure fp
    # cancellation noise, not a metal-kernel error).
    big = (metal_ref > 1e-10) & (metal_ref > 10.0 * (gt_ahline + gt_he))
    rel = np.abs(metal[big] - metal_ref[big]) / np.abs(metal_ref[big])
    print(f"[metal_allZ] vs (diag - H - He), metal-dominated: max rel={rel.max():.3e} median={np.median(rel):.3e}")

    t0 = time.perf_counter()
    he = compute_helium_opacity(cat, atm, diag, L4)
    print(f"helium runtime: {time.perf_counter()-t0:.1f}s")
    he_ref = gt_he
    big = he_ref > 1e-10
    relhe = np.abs(he[big] - he_ref[big]) / np.abs(he_ref[big])
    print(f"[He] vs gt_helium_wings*stim, |ref|>1e-10: max rel={relhe.max():.3e} median={np.median(relhe):.3e}")

    t0 = time.perf_counter()
    ah = compute_hydrogen_opacity(cat, atm, diag, L4)
    print(f"hydrogen runtime: {time.perf_counter()-t0:.1f}s")
    big = gt_ahline > 1e-10
    relh = np.abs(ah[big] - gt_ahline[big]) / np.abs(gt_ahline[big])
    print(f"[H] vs gt_ahline, |ref|>1e-10: max rel={relh.max():.3e} median={np.median(relh):.3e}")

    # ── full reconstruction ──
    mine = metal + he + ah
    resid = ref - mine
    nz = ref != 0.0
    rel_nz = np.abs(resid[nz]) / np.abs(ref[nz])
    print()
    print(f"reference nonzero elements: {nz.sum()} / {ref.size}")
    print(f"FULL max  relative error vs diag['line_opacity'] (nonzero): {rel_nz.max():.3e}")
    print(f"FULL median relative error vs diag['line_opacity'] (nonzero): {np.median(rel_nz):.3e}")
    print(f"residual (ref - mine):  min={resid.min():.3e}  max={resid.max():.3e}")
    if rel_nz.max() < 1e-9:
        print("PASS: max relative error < 1e-9 vs diag['line_opacity']")
    else:
        k = np.unravel_index(np.argmax(np.where(nz, np.abs(resid) / np.where(nz, np.abs(ref), 1.0), 0.0)), ref.shape)
        print(f"largest rel residual at depth={k[0]} grid={k[1]}: mine={mine[k]:.6e} ref={ref[k]:.6e}")


if __name__ == "__main__":
    main()
