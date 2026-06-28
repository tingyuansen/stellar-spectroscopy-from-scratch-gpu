#!/usr/bin/env python
"""From-scratch verification of the Lecture 15 line-blanketed atmosphere.

Loads ONLY reference/lineblanket_ref.npz (+ the shipped Lecture 8/11 tables) and,
using a pure-NumPy transcription of the LINOP1 deposit kernel and the Lecture 11
convergence engine (verify_converged.py), checks:

  (a) the teaching-window LINE DEPOSIT reproduces the production deposit
      xlines_window_ref to the float32 wing-accumulator floor (~1e-7);
  (b) ONE line-blanketed convergence step {JOSH, ROSS, RADIAP, CONVEC, TCORR},
      with the deposited blanket folded into the opacity, reproduces the
      production single step (T_step/rhox_step) to machine precision; and
  (c) that step lands on the real Sun's model atmosphere sun.npz to the
      documented deep-column residual (~1e-3 median).

No pykurucz import; the deposit kernel is the same pure-Python walk the notebook
runs (8-stride probe + fill-in, asymmetric wing-walk, full Voigt, cv<tabcont cut).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import verify_converged as VC

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"

RATIOLG = np.log(1.0 + 1.0 / 2_000_000.0)
TABLOG = 10.0 ** ((np.arange(1, 32769, dtype=np.float64) - 16384.0) * 0.001)
C_NM = 2.99792458e17
CGF_SCALE = 0.026538 / 1.77245 / C_NM
GAMMA_SCALE = 1.0 / 12.5664 / C_NM
SIGMA = 5.6697e-5
PLANCK = 6.6256e-27
KBOLTZ = 1.38054e-16


def _build_voigt(H0, H1, H2):
    def voigt_H(v, a):
        iv = int(v * 200.0 + 1.5); iv = min(max(iv, 1), 2001); i = iv - 1
        if a >= 0.2:
            if a > 1.4 or a + v > 3.2:
                aa = a * a; vv = v * v; u = (aa + vv) * 1.4142; out = a * 0.79788 / u
                if a > 100.0:
                    return out
                aau = aa / u; vvu = vv / u; uu = u * u
                return ((((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa) / uu + 1.0) * out
            vv = v * v
            hh1 = H1[i] + H0[i] * 1.12838
            hh2 = H2[i] + hh1 * 1.12838 - H0[i]
            hh3 = (1.0 - H2[i]) * 0.37613 - hh1 * 0.66667 * vv + hh2 * 1.12838
            hh4 = (3.0 * hh3 - hh1) * 0.37613 + H0[i] * 0.66667 * vv * vv
            return ((((hh4 * a + hh3) * a + hh2) * a + hh1) * a + H0[i]) \
                * (((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032)
        if v > 10.0:
            return 0.5642 * a / (v * v)
        return (H2[i] * a + H1[i]) * a + H0[i]
    return voigt_H


def _deposit_window(d, H0, H1, H2):
    """Pure-Python LINOP1 deposit of the teaching window (8-stride + fill-in, asymmetric wing-walk)."""
    voigt_H = _build_voigt(H0, H1, H2)
    f32 = np.float32
    waveset = d["waveset_nm"].astype(np.float64)
    iwavetab = d["iwavetab"].astype(np.int64)
    tabcont = d["tabcont"].astype(np.float64)
    hckt = d["win_hckt"].astype(np.float64); xne = d["win_xne"].astype(np.float64)
    txnxn = d["win_txnxn"].astype(np.float64)
    nelion_set = d["win_nelion_set"].astype(np.int64)
    xnfdop = d["win_xnfdop"].astype(np.float64); dopple = d["win_dopple"].astype(np.float64)
    nelion_to_col = {int(z): k for k, z in enumerate(nelion_set)}
    pix_lo, pix_hi = int(d["win_pix_lo"]), int(d["win_pix_hi"])
    iwl = d["win_iwl"]; ielion = d["win_ielion"]; ielo = d["win_ielo"]
    igflog = d["win_igflog"]; igr = d["win_igr"]; igs = d["win_igs"]; igw = d["win_igw"]

    extab = np.exp(-np.arange(1001, dtype=np.float64)).astype(np.float32)
    extabf = np.exp(-np.arange(1001, dtype=np.float64) * 0.001).astype(np.float32)

    def fastex(x):
        if not np.isfinite(x) or x < 0.0 or x >= 1001.0:
            return np.float32(0.0)
        i = int(x); j = int((x - i) * 1000.0 + 1.5); j = min(max(j, 1), 1001)
        return np.float32(extab[i] * extabf[j - 1])

    def accwings(xlines, j0, nu0, wlvac, center, adamp, dopwave, tabref, waveset):
        numnu = waveset.shape[0]
        if dopwave <= 0.0:
            return
        ired_max = 100; ired_hi = min(nu0 + ired_max + 1, numnu)
        if adamp <= 0.2:
            for iw in range(nu0, ired_hi):
                vv = f32(waveset[iw] - wlvac) / dopwave
                if vv > f32(10.0):
                    cv = f32(center * f32(0.5642) * adamp / (vv * vv))
                else:
                    iv = int(vv * f32(200.0) + f32(1.5)); iv = min(max(iv, 1), 2001)
                    cv = f32(center * ((H2[iv - 1] * adamp + H1[iv - 1]) * adamp + H0[iv - 1]))
                xlines[j0, iw] += cv
                if cv < tabref:
                    break
            for ired in range(1, ired_max + 1):
                iw = nu0 - ired
                if iw < 0:
                    break
                vv = f32(wlvac - waveset[iw]) / dopwave
                if vv > f32(10.0):
                    cv = f32(center * f32(0.5642) * adamp / (vv * vv))
                else:
                    iv = int(vv * f32(200.0) + f32(1.5)); iv = min(max(iv, 1), 2001)
                    cv = f32(center * ((H2[iv - 1] * adamp + H1[iv - 1]) * adamp + H0[iv - 1]))
                xlines[j0, iw] += cv
                if cv < tabref:
                    break
            return
        for iw in range(nu0, ired_hi):
            cv = f32(center * voigt_H(f32(waveset[iw] - wlvac) / dopwave, adamp))
            xlines[j0, iw] += cv
            if cv < tabref:
                break
        for ired in range(1, ired_max + 1):
            iw = nu0 - ired
            if iw < 0:
                break
            cv = f32(center * voigt_H(f32(wlvac - waveset[iw]) / dopwave, adamp))
            xlines[j0, iw] += cv
            if cv < tabref:
                break

    nrhox = hckt.size; numnu = waveset.size
    xlines = np.zeros((nrhox, numnu), dtype=np.float32)
    ifj = np.zeros(nrhox + 2, dtype=np.int32)
    start, stop = waveset[0] - 1.0, waveset[-1] + 1.0
    nu0 = 0; nucont0 = 0; iwlold = 0

    def deposit_one(j0, cgf, elo, gr, gs, gw, col, wl, wl4, nucont0):
        cen = cgf * xnfdop[j0, col]
        if cen < tabcont[j0, nucont0]:
            return False
        cen = cen * fastex(elo * hckt[j0])
        if cen < tabcont[j0, nucont0]:
            return False
        dop = dopple[j0, col]
        if dop <= 0.0:
            return True
        adamp = (gr + gs * xne[j0] + gw * txnxn[j0]) / dop
        accwings(xlines, j0, nu0, wl, cen, adamp, dop * wl4, tabcont[j0, nucont0], waveset)
        return True

    for k in range(iwl.size):
        iw = int(iwl[k])
        if iw < iwlold:
            nu0 = 0; nucont0 = 0
        while nucont0 < iwavetab.size and iw >= int(iwavetab[nucont0]):
            nucont0 += 1
        if nucont0 >= tabcont.shape[1]:
            iwlold = iw; continue
        nel = abs(int(ielion[k])) // 10
        if nel not in nelion_to_col:
            iwlold = iw; continue
        col = nelion_to_col[nel]
        wl = np.exp(float(iw) * RATIOLG); wl4 = np.float32(wl)
        if wl < start or wl > stop:
            iwlold = iw; continue
        while nu0 < numnu and wl >= waveset[nu0]:
            nu0 += 1
        if nu0 >= numnu:
            iwlold = iw; continue
        cgf = np.float32(CGF_SCALE) * wl4 * TABLOG[int(igflog[k]) - 1]
        elo = TABLOG[int(ielo[k]) - 1]
        gr = TABLOG[int(igr[k]) - 1] * wl4 * np.float32(GAMMA_SCALE)
        gs = TABLOG[int(igs[k]) - 1] * wl4 * np.float32(GAMMA_SCALE)
        gw = TABLOG[int(igw[k]) - 1] * wl4 * np.float32(GAMMA_SCALE)
        for j1 in range(8, nrhox + 1, 8):
            ifj[j1 + 1] = 0
            if deposit_one(j1 - 1, cgf, elo, gr, gs, gw, col, wl, wl4, nucont0):
                ifj[j1 + 1] = 1
        for k1 in range(8, nrhox + 1, 8):
            if ifj[k1 - 7] + ifj[k1 + 1] == 0:
                continue
            for j1 in range(k1 - 7, k1):
                deposit_one(j1 - 1, cgf, elo, gr, gs, gw, col, wl, wl4, nucont0)
        iwlold = iw
    return xlines[:, pix_lo:pix_hi]


def _blanketed_step(d):
    """ONE line-blanketed convergence step from the Sun, using the Lecture 11 engines."""
    jt = np.load(REF / "josh_tables.npz")
    xtau = jt["xtau"].astype(np.float64); ch = jt["ch"].astype(np.float64); coefj = jt["coefj"].astype(np.float64)
    VC.CH_MAT = np.load(REF / "converged_ref.npz")["josh_coefh"].astype(np.float64)
    VC.CK_W = np.load(REF / "josh_ck.npz")["ck"].astype(np.float32)

    T = d["T"].astype(np.float64); rhox = d["rhox"].astype(np.float64)
    P = d["P"].astype(np.float64); rho = d["rho"].astype(np.float64)
    ptotal = d["ptotal"].astype(np.float64); grav = float(d["gravity_cgs"]); teff = float(d["teff"])
    freq = d["freq_hz"].astype(np.float64); rco = d["rco"].astype(np.float64)
    acont = d["acont"]; sigmac = d["sigmac"]; scont = d["scont"]
    xlines = d["xlines_fullgrid"].astype(np.float64); ahline = d["ahline_fullgrid"].astype(np.float64)
    n = T.size; nf = freq.size
    hkt = PLANCK / np.maximum(T * KBOLTZ, 1e-300)
    flux = SIGMA / 12.5664 * teff ** 4; z = np.zeros(n)

    ross_acc = np.zeros(n); flxrad = np.zeros(n); rjmins = np.zeros(n)
    rdabh = np.zeros(n); rdiagj = np.zeros(n); accrad = np.zeros(n); pradk0_acc = 0.0
    for inu in range(nf):
        f = float(freq[inu]); w = float(rco[inu])
        ehvkt = np.exp(-f * hkt); stim = np.maximum(1.0 - ehvkt, 1e-300)
        bnu = 1.47439e-2 * ((f / 1e15) ** 3) * ehvkt / stim
        aline = ahline[:, inu] + xlines[:, inu] * stim
        taunu, hnu, jmins, abtot, alpha, knu0 = VC.josh_profiles(
            acont[:, inu].astype(np.float64), scont[:, inu].astype(np.float64), aline, bnu,
            sigmac[:, inu].astype(np.float64), z, rhox, bnu, xtau, ch, coefj)
        if np.any(hnu < 0.0):
            hnu = np.maximum(hnu, 1e-99)
        dbdt = bnu * f * hkt / np.maximum(T * stim, 1e-300)
        ross_acc += dbdt / np.maximum(abtot, 1e-300) * w
        rdabh += VC.deriv(rhox, abtot) / np.maximum(abtot, 1e-300) * hnu * w
        rjmins += abtot * jmins * w; flxrad += hnu * w; accrad += abtot * hnu * w; pradk0_acc += knu0 * w
        term2 = 0.0
        for j in range(n):
            term1 = term2
            dd = max(1e-10, float(taunu[j + 1] - taunu[j]) if j != n - 1 else 1e-10)
            if dd <= 0.01:
                term2 = (0.922784335098467 - np.log(dd)) * dd / 4.0 + dd * dd / 12.0 - dd ** 3 / 96.0 + dd ** 4 / 720.0
            else:
                ex = VC.expi3(dd) if dd < 10.0 else 0.0
                if teff <= 4250.0 and 0.005 < dd < 0.02:
                    ex = 0.0
                term2 = 0.5 * (dd + ex - 0.5) / dd
            diagj = term1 + term2; dbdtj = bnu[j] * f * hkt[j] / max(T[j] * stim[j], 1e-300)
            rdiagj[j] += abtot[j] * (diagj - 1.0) / max(1.0 - alpha[j] * diagj, 1e-300) * (1.0 - alpha[j]) * dbdtj * w

    abross, tauros = VC.ross_finalize(ross_acc, T, rhox)
    conv = 12.5664 / 2.99792458e10; accrad *= conv
    ratio = flxrad / max(flux, 1e-300); over = ratio > 1.0
    accrad[over] *= flux / np.maximum(flxrad[over], 1e-300)
    prad = VC.integ(rhox, accrad, accrad[0] * rhox[0])
    pradk0 = pradk0_acc * conv
    if float(np.max(ratio)) > 1.0:
        pradk0 /= float(np.max(ratio))
    pradk = prad + pradk0
    rosstab = VC.Rosstab(); rosstab.ingest(T, P, abross)
    dilut = 1.0 - np.exp(-tauros)
    r1, r2, r3, r4 = d["rho1"], d["rho2"], d["rho3"], d["rho4"]
    e1 = d["eint1"] + 3.0 * pradk / np.maximum(r1, 1e-300) * (1.0 + dilut * (1.001 ** 4 - 1.0))
    e2 = d["eint2"] + 3.0 * pradk / np.maximum(r2, 1e-300) * (1.0 + dilut * (0.999 ** 4 - 1.0))
    e3 = d["eint3"] + 3.0 * pradk / np.maximum(r3, 1e-300)
    e4 = d["eint4"] + 3.0 * pradk / np.maximum(r4, 1e-300)
    cv = VC.convec(rosstab, rhox, tauros, T, P, rho, abross, pradk, ptotal, grav, flux,
                   e1, e2, e3, e4, r1, r2, r3, r4, mixlth=1.25, nconv=36)
    cv["ptotal"] = ptotal; cv["rho"] = rho
    res = VC.tcorr_mode3(T, rhox, tauros, abross, flxrad, rjmins, rdabh, rdiagj,
                         flux, teff, prad, grav, rosstab, cv, mixlth=1.25)
    taustd = 10.0 ** (-6.875 + np.arange(n) * 0.125)
    T_out, _ = VC.map1(tauros, res["tnew"], taustd)
    rhox_out, _ = VC.map1(tauros, res["rhox_new"], taustd)
    return T_out, rhox_out, abross[-1]


def main() -> int:
    d = dict(np.load(REF / "lineblanket_ref.npz", allow_pickle=True))
    kt = np.load(REF / "leankurucz_tables.npz")
    H0 = kt["h0tab"].astype(np.float32); H1 = kt["h1tab"].astype(np.float32); H2 = kt["h2tab"].astype(np.float32)

    print("=" * 70)
    print("Lecture 15 verification: line-blanketed model atmosphere")
    print(f"  Teff={float(d['teff']):.0f}  logg={float(d['logg']):.2f}  "
          f"surf T={d['T'][0]:.1f}  base T={d['T'][-1]:.1f}  base RHOX={d['rhox'][-1]:.3f}")
    print("=" * 70)

    # (a) the deposit kernel on the teaching window
    xl = _deposit_window(d, H0, H1, H2)
    ref = d["xlines_window_ref"].astype(np.float64); got = xl.astype(np.float64)
    m = np.abs(ref) > 0
    amax = float(np.abs(got - ref).max())
    rdep = float((np.abs(got[m] - ref[m]) / np.abs(ref[m])).max()) if m.any() else 0.0
    npix = int(np.count_nonzero(ref.sum(0)))
    print(f"\n(a) line deposit, teaching window ({d['win_iwl'].size} lines, {npix} non-empty pixels):")
    print(f"    max|abs diff| = {amax:.3e}   max|rel| = {rdep:.3e}   <- float32 wing floor")

    # (b) one line-blanketed step + (c) landing on sun.npz
    T_out, rhox_out, base_abross = _blanketed_step(d)
    rT = float(np.max(np.abs(T_out - d["T_step"]) / np.maximum(np.abs(d["T_step"]), 1e-300)))
    rX = float(np.max(np.abs(rhox_out - d["rhox_step"]) / np.maximum(np.abs(d["rhox_step"]), 1e-300)))
    print(f"\n(b) one line-blanketed step vs production single step (engine fidelity):")
    print(f"    T max|rel| = {rT:.3e}   RHOX max|rel| = {rX:.3e}   base kappa_Ross = {base_abross:.1f} cm^2/g")

    sun_T = d["sun_T"]; sun_rhox = d["sun_rhox"]
    Ton = np.interp(np.log(sun_rhox), np.log(rhox_out), T_out)
    mrel = np.abs(Ton - sun_T) / np.abs(sun_T)
    print(f"\n(c) the corrected model vs the real Sun sun.npz:")
    print(f"    T median|rel| = {np.median(mrel):.3e}   max|rel| = {np.max(mrel):.3e}")
    print(f"    surf T={T_out[0]:.1f} (sun {sun_T[0]:.1f})  base T={T_out[-1]:.1f} (sun {sun_T[-1]:.1f})  "
          f"base RHOX={rhox_out[-1]:.3f} (sun {sun_rhox[-1]:.3f})")

    ok = (rdep < 1e-5) and (rT < 1e-5) and (rX < 1e-5) and (np.median(mrel) < 5e-3)
    print("\n" + "=" * 70)
    print("  PASS" if ok else "  FAIL")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
