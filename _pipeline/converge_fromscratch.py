#!/usr/bin/env python
"""Stage 8 — the FULL from-scratch line-blanketed convergence (NO pyk subprocess).

Replaces numpy_oracle.oracle_state (the per-iteration pyk subprocess) with the book's OWN
from-scratch per-iteration state:
  * EOS (eos_fromscratch.popsall)                   — multi-element populations + flat slots
  * Doppler (eos_fromscratch)                        — xnfdop/dopple/txnxn
  * continuum + tabcont (continuum_fromscratch)      — L3 KAPP + the far-UV metal-bf forest
  * EDENS FD (eos_fromscratch.convec_fd_samples)     — the ionization-energy heat capacity
  * molecular populations (molecular_fromscratch)    — L13 NMOLEC -> molecular slots 841-940
  * deposit (numpy_oracle.linop1_deposit + kgpu atmosphere_xlines/_hlines kernels — numba, NO pyk)

The structure update uses the book's OWN convergence engines (verify_converged: josh_profiles,
ross_finalize, convec, tcorr_mode3, map1, integ, Rosstab, deriv, expi3) — exactly the Lecture-11
machinery.  The deposit KERNELS are the verified bit-exact transcriptions (no pyk import).  Starts
from the emulator warm-start; converges to near-sun.

Run in the book .venv (has numpy/numba/torch).  pyk is NEVER imported.
"""
from __future__ import annotations

import os
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"
GPU = Path("/Users/ysting/pykurucz_gpu")
sys.path.insert(0, str(ROOT / "_pipeline"))
sys.path.insert(0, str(GPU / "bench"))   # numpy_oracle.linop1_deposit (numba, no pyk)
sys.path.insert(0, str(GPU))             # kgpu.atmosphere_xlines/_hlines (numba, no pyk)

import eos_fromscratch as EOS
import continuum_fromscratch as CF
import molecular_fromscratch as MF
import verify_converged as VC

# the deposit kernels (verified bit-exact transcriptions; NO pyk import on these paths)
import numpy_oracle as NO                      # linop1_deposit + read_fort12 (numba)
from kgpu import atmosphere_xlines as KX       # xlinop type-0 + fort19 decode (numba)
from kgpu import atmosphere_hlines as KH       # xlinop type-1 H HPROF4 (numba)

TEFF = 5777.0
WTMOLE = 1.2584297579180466
VTURB = 2.0e5
SIGMA = 5.6697e-5
FOURPI = 12.566370614359172
PLANCK = 6.6256e-27
KBOLTZ = 1.38054e-16
PRADK0_SUN = 1.47
MOLPATH = REF / "molecules.dat"


def _warm_start():
    """The emulator warm-start structure (the honest initial condition)."""
    sun = np.load(GPU / "bench" / "work" / "sun.npz", allow_pickle=True)
    # use the warm-start atm (RHOX base ~7.5) if present, else the sun structure scaled
    atm = GPU / "bench" / "work" / "sun_warmstart.atm"
    grav = 10.0 ** float(sun["glog"])
    xab = np.asarray(sun["xabund"], np.float64)
    # parse the warm-start .atm columns (RHOX, T, P, XNE, ...)
    lines = atm.read_text().splitlines()
    data = []
    started = False
    for ln in lines:
        if ln.strip().startswith("READ DECK6") or "RHOX" in ln:
            started = True; continue
        if started:
            parts = ln.split()
            if len(parts) >= 5 and parts[0].replace(".", "").replace("E", "").replace("+", "").replace("-", "").isdigit():
                try:
                    data.append([float(x) for x in parts[:7]])
                except ValueError:
                    break
    arr = np.array(data)
    if arr.shape[0] < 70:   # fall back to the sun structure if parse failed
        T = np.asarray(sun["temperature"], np.float64)
        rhox = np.asarray(sun["depth"], np.float64)
        P = np.asarray(sun["gas_pressure"], np.float64)
        return T, rhox, P, grav, xab
    rhox = arr[:, 0]; T = arr[:, 1]; P = arr[:, 2]
    return T, rhox, P, grav, xab


def _setup():
    VC.CH_MAT = np.load(REF / "converged_ref.npz")["josh_coefh"].astype(np.float64)
    VC.CK_W = np.load(REF / "josh_ck.npz")["ck"].astype(np.float32)
    jt = np.load(REF / "josh_tables.npz")
    lb = np.load(REF / "lineblanket_ref.npz", allow_pickle=True)
    wave = np.asarray(lb["waveset_nm"], np.float64)
    freq = np.asarray(lb["freq_hz"], np.float64)
    rco = np.asarray(lb["rco"], np.float64)
    os_start = float(wave[0])
    return (jt["xtau"].astype(np.float64), jt["ch"].astype(np.float64),
            jt["coefj"].astype(np.float64), wave, freq, rco, os_start)


def _state(T, rhox, P, grav, xab, tab, records, h0, h1, h2, recs0, h_records, contx,
           wave, freq, os_start):
    """The full from-scratch per-iteration state: EOS + continuum + EDENS + deposit."""
    n = T.size; nf = freq.size
    pa = EOS.popsall(T, P, xab, tab=tab, wtmole=WTMOLE)
    rho = pa.rho; xne = pa.xne
    vturb = np.full(n, VTURB)
    amass = np.load(REF / "atomic_masses.npz")["atmass"]
    dop3 = EOS.compute_doppler_per_ion(T, vturb, amass)
    xnfdop, dopple = EOS.build_xnfdop_dopple_per_nelion(pa.population_per_ion, dop3, rho)
    txnxn = EOS.compute_txnxn(T, pa.xnf_h, pa.xnf_he1)
    # molecular slots
    xnfpmol, dop_mol, _ = MF.molecular_deposit_populations(T, P, pa, xab, vturb, MOLPATH)
    xnfdop, dopple = MF.fill_molecular_slots(xnfdop, dopple, rho, xnfpmol, dop_mol)
    # continuum + tabcont (with the far-UV metal-bf forest)
    acont, sigmac = CF.compute_continuum(freq, pa, T, rho, xne, faruv=True)
    tabcont, iwt = CF.build_tabcont(pa, T, rho, xne, os_start)
    hckt = PLANCK * 2.99792458e10 / (KBOLTZ * T)
    # deposit: LINOP1 + XLINOP type-0 + type-1 H
    xl = NO.linop1_deposit(records, wave, iwt, tabcont, hckt, xne, xnfdop, dopple, txnxn,
                           nulo=1, nuhi=nf, n_chunks=8)
    xl = KX.xlinop_deposit(base_xlines=xl, records=recs0, waveset_nm=wave, iwavetab=iwt,
                           tabcont=tabcont, temperature_k=T, hckt=hckt, xne=xne,
                           xnfdop=xnfdop, dopple=dopple, txnxn=txnxn, h0tab=h0, h1tab=h1, h2tab=h2)
    nstark = 1600.0 / np.maximum(xne, 1e-300) ** (2.0 / 15.0)
    emerge = 109737.312 / np.maximum(nstark * nstark, 1e-300)
    xl = KH.xlinop_hydrogen_deposit(base_xlines=xl, records=h_records, waveset_nm=wave,
                                    iwavetab=iwt, tabcont=tabcont, temperature_k=T, hckt=hckt,
                                    xne=xne, xnf_h1=pa.xnf_h, xnf_h2=pa.xnfph[:, 1],
                                    xnfp_h1=pa.xnfph[:, 0], xnfp_he1=pa.xnf_he1,
                                    xnfdop_h1=xnfdop[:, 0], dopple_h1=dopple[:, 0],
                                    contx=contx, emerge=emerge)
    # EDENS FD raw gas internal energy + rho (radiation term added fresh each iteration)
    ei1, ei2, ei3, ei4, r1, r2, r3, r4 = EOS.convec_fd_raw(T, P, xab, WTMOLE, xne, tab)
    return dict(acont=acont.astype(np.float64), sigmac=sigmac.astype(np.float64),
                xlines=xl.astype(np.float64), rho=rho, xne=xne, pa=pa,
                ei=(ei1, ei2, ei3, ei4), rr=(r1, r2, r3, r4))


def converge(max_iters=30, step_damp=0.5, verbose=True):
    xtau, ch, coefj, wave, freq, rco, os_start = _setup()
    T, rhox, P, grav, xab = _warm_start()
    n = T.size; nf = freq.size
    tab = EOS.EOSTables.from_npz()
    print(f"loading fort12 catalog (honest input)...", flush=True)
    records = NO.read_fort12(NO.FORT12)
    h0, h1, h2 = KX.harris_tables()
    r0 = np.load(KX.fort19_records_npz()); recs0 = {k: r0[k] for k in r0.files}
    rh = np.load(KX.fort19_hydrogen_records_npz()); h_records = {k: rh[k] for k in rh.files}
    contx = KX.build_contx()
    sun = np.load(GPU / "bench" / "work" / "sun.npz", allow_pickle=True)
    Ts = np.asarray(sun["temperature"], np.float64); Rs = np.asarray(sun["depth"], np.float64)
    ptotal = grav * rhox + PRADK0_SUN
    flux = SIGMA / FOURPI * TEFF ** 4
    hkt = PLANCK / np.maximum(T * KBOLTZ, 1e-300)
    print(f"start: surf T={T[0]:.1f} base T={T[-1]:.1f} base RHOX={rhox[-1]:.3f}  target sun "
          f"3696/11425/12.14", flush=True)

    for it in range(max_iters):
        t0 = time.perf_counter()
        st = _state(T, rhox, P, grav, xab, tab, records, h0, h1, h2, recs0, h_records, contx,
                    wave, freq, os_start)
        acont = st["acont"]; sigmac = st["sigmac"]; xlines = st["xlines"]; rho = st["rho"]
        hkt = PLANCK / np.maximum(T * KBOLTZ, 1e-300)
        z = np.zeros(n)
        ross_acc = np.zeros(n); flxrad = np.zeros(n); rjmins = np.zeros(n)
        rdabh = np.zeros(n); rdiagj = np.zeros(n); accrad = np.zeros(n); pradk0_acc = 0.0
        for inu in range(nf):
            f = float(freq[inu]); rcowt = float(rco[inu])
            ehvkt = np.exp(-f * hkt); stim = np.maximum(1.0 - ehvkt, 1e-300)
            bnu = 1.47439e-2 * ((f / 1e15) ** 3) * ehvkt / stim
            aline = xlines[:, inu] * stim
            taunu, hnu, jmins, abtot, alpha, knu = VC.josh_profiles(
                acont[:, inu], bnu, aline, bnu, sigmac[:, inu], z, rhox, bnu, xtau, ch, coefj)
            if np.any(hnu < 0.0):
                hnu = np.maximum(hnu, 1e-99)
            dbdt = bnu * f * hkt / np.maximum(T * stim, 1e-300)
            ross_acc += dbdt / np.maximum(abtot, 1e-300) * rcowt
            rdabh += VC.deriv(rhox, abtot) / np.maximum(abtot, 1e-300) * hnu * rcowt
            rjmins += abtot * jmins * rcowt
            flxrad += hnu * rcowt
            accrad += abtot * hnu * rcowt
            pradk0_acc += knu * rcowt
            term2 = 0.0
            for j in range(n):
                term1 = term2
                dd = 1e-10 if j == n - 1 else (taunu[j + 1] - taunu[j])
                dd = max(1e-10, float(dd))
                if dd <= 0.01:
                    term2 = (0.922784335098467 - np.log(dd)) * dd / 4.0 + dd * dd / 12.0 - dd ** 3 / 96.0 + dd ** 4 / 720.0
                else:
                    ex = VC.expi3(dd) if dd < 10.0 else 0.0
                    if TEFF <= 4250.0 and 0.005 < dd < 0.02:
                        ex = 0.0
                    term2 = 0.5 * (dd + ex - 0.5) / dd
                diagj = term1 + term2
                dbdtj = bnu[j] * f * hkt[j] / max(T[j] * stim[j], 1e-300)
                rdiagj[j] += abtot[j] * (diagj - 1.0) / max(1.0 - alpha[j] * diagj, 1e-300) \
                    * (1.0 - alpha[j]) * dbdtj * rcowt
        abross, tauros = VC.ross_finalize(ross_acc, T, rhox)
        conv = 12.5664 / 2.99792458e10
        accrad *= conv
        ratio = flxrad / max(flux, 1e-300); over = ratio > 1.0
        accrad[over] *= flux / np.maximum(flxrad[over], 1e-300)
        prad = VC.integ(rhox, accrad, accrad[0] * rhox[0])
        errormax = float(np.max(ratio)); pradk0 = pradk0_acc * conv
        if errormax > 1.0:
            pradk0 /= errormax
        pradk = prad + pradk0
        rosstab = VC.Rosstab(); rosstab.ingest(T, P, abross)
        dilut = 1.0 - np.exp(-tauros)
        ei1, ei2, ei3, ei4 = st["ei"]; r1, r2, r3, r4 = st["rr"]
        e1 = ei1 + 3.0 * pradk / np.maximum(r1, 1e-300) * (1.0 + dilut * (1.001 ** 4 - 1.0))
        e2 = ei2 + 3.0 * pradk / np.maximum(r2, 1e-300) * (1.0 + dilut * (0.999 ** 4 - 1.0))
        e3 = ei3 + 3.0 * pradk / np.maximum(r3, 1e-300)
        e4 = ei4 + 3.0 * pradk / np.maximum(r4, 1e-300)
        cv = VC.convec(rosstab, rhox, tauros, T, P, rho, abross, pradk, ptotal, grav, flux,
                       e1, e2, e3, e4, r1, r2, r3, r4, mixlth=1.25, nconv=36)
        cv["ptotal"] = ptotal; cv["rho"] = rho
        res = VC.tcorr_mode3(T, rhox, tauros, abross, flxrad, rjmins, rdabh, rdiagj,
                             flux, TEFF, prad, grav, rosstab, cv, mixlth=1.25)
        taustd = 10.0 ** (-6.875 + np.arange(n) * 0.125)
        T_new, _ = VC.map1(tauros, res["tnew"], taustd)
        rhox_new, _ = VC.map1(tauros, res["rhox_new"], taustd)
        prad_new, _ = VC.map1(tauros, prad, taustd)
        T_step = T + step_damp * (T_new - T)
        rhox_step = rhox + step_damp * (rhox_new - rhox)
        dTmax = float(np.max(np.abs(T_step - T) / np.maximum(T, 1.0)))
        T, rhox, prad = T_step, rhox_step, prad_new
        P = np.maximum(grav * rhox - prad, 1e-4)
        ptotal = grav * rhox + PRADK0_SUN
        if verbose:
            Tk_on = np.interp(np.log(Rs), np.log(rhox), T)
            mrel = np.abs(Tk_on - Ts) / np.abs(Ts)
            print(f"  iter {it+1:2d}  dTmax={dTmax:.3e}  surfT={T[0]:.1f} baseT={T[-1]:.1f} "
                  f"baseRHOX={rhox[-1]:.3f}  abross[base]={abross[-1]:.3e}  Tmed={np.median(mrel):.3e} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        if dTmax < 5e-4:
            print("  CONVERGED", flush=True); break

    Tk_on = np.interp(np.log(Rs), np.log(rhox), T)
    mrel = np.abs(Tk_on - Ts) / np.abs(Ts)
    print("=" * 70, flush=True)
    print(f"FINAL: surfT={T[0]:.1f} (3696)  baseT={T[-1]:.1f} (11425)  baseRHOX={rhox[-1]:.3f} (12.14)", flush=True)
    print(f"  T vs sun.npz: median|rel|={np.median(mrel):.3e}  max|rel|={np.max(mrel):.3e}", flush=True)
    for lbl, j in [("surface", 0), ("mid40", 40), ("deep55", 55), ("deep63", 63), ("base", -1)]:
        print(f"    {lbl:9s} book T={Tk_on[j]:7.1f}  sun T={Ts[j]:7.1f}  rel={mrel[j]:.2e}", flush=True)
    np.savez("/tmp/converge_fromscratch.npz", T=T, rhox=rhox, Ts=Ts, Rs=Rs,
             Tmed=float(np.median(mrel)), base_T=T[-1], base_RHOX=rhox[-1])
    print("FROMSCRATCHDONE", flush=True)
    return T, rhox


if __name__ == "__main__":
    mi = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    converge(max_iters=mi)
