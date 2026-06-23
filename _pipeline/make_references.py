#!/usr/bin/env python
"""Precompute the reference values each lecture checks against, and save them as
data files under reference/.

This is the ONLY place pykurucz is used. It runs once (by the author) to produce
the small reference/*.npz data files that ship with the book; the lecture
notebooks then load those files and compare — they never import pykurucz, so a
reader needs only numpy + matplotlib to run everything.

pykurucz is imported read-only and never modified.

  python _pipeline/make_references.py        # build all lecture reference files
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/ysting/pykurucz")          # reference code, read-only
BOOK = Path(__file__).resolve().parent.parent
REF = BOOK / "reference"
REF.mkdir(exist_ok=True)

C_LIGHT = 2.99792458e10   # cm/s


def lecture1() -> None:
    """Reference values for Lecture 1: the Planck function and the grey atmosphere."""
    from synthe_py.engine.radiative import _planck_nu
    from atlas_py.physics.grey_start import generate_grey_atm

    # Grey model atmosphere of the Sun (Teff=5770, logg=4.44).
    g = generate_grey_atm(teff=5770.0, logg=4.44)

    # Planck B_nu at mid-window (505 nm) across a spread of layer temperatures.
    planck_T = np.array([3500.0, 4500.0, 5770.0, 7000.0, 9000.0, 12000.0])
    planck_freq = C_LIGHT / (505e-7)
    planck_B = _planck_nu(planck_freq, planck_T)

    out = REF / "L1.npz"
    np.savez(
        out,
        grey_tau=g.tauros, grey_T=g.temperature, grey_pgas=g.p_gas, grey_rhox=g.rhox,
        planck_freq=np.float64(planck_freq), planck_T=planck_T, planck_B=planck_B,
    )
    print(f"wrote {out}  ({', '.join(np.load(out).files)})")


def _grey_atmosphere(teff=5770.0, logg=4.44):
    g = 10.0**logg
    tau = 10.0**(-6.875 + 0.125*np.arange(80))
    q = 0.710 + tau - 0.1331*np.exp(-3.4488*tau)
    T = teff*(0.75*q)**0.25
    P_total = g*tau
    P_rad = 2.521e-15*np.maximum(T**4, teff**4/2.0); P_rad -= P_rad[0]
    P_gas = P_total - P_rad
    return tau, T, P_gas, P_total/g  # tau, T, P_gas, rhox


def _solar_xabund():
    """99-element linear number fractions over all atoms (Kurucz solar)."""
    xab = np.empty(99)
    xab[0] = float(pykurucz_compute_h())
    xab[1] = float(pykurucz_HE())
    xab[2:] = 10.0**np.asarray(pykurucz_compute_abund(), float)
    return xab


def lecture2() -> None:
    """Reference values for Lecture 2: the equation of state.

    Ships, on the grey solar atmosphere: the reference electron density n_e
    (a full charge-balance solve), the partition functions U and ionization
    potentials chi of the major electron donors (atomic data the lecture reuses),
    and the reference ionization fractions, so a clean from-scratch Saha + charge
    balance can be checked layer by layer.
    """
    from atlas_py.physics.pfsaha import pfsaha_depth, _start_and_nions
    from atlas_py.physics.nelect import _nion_for_atomic_number

    tau, T, P_gas, rhox = _grey_atmosphere()
    K = 1.380649e-16
    tk = K*T
    xab = _solar_xabund()
    n = T.size
    MAXION = 8
    nion_z = np.array([_nion_for_atomic_number(z) for z in range(1, 100)])
    # pfsaha normalises the Saha ladder over nion2 = min(nion+2, available) stages internally,
    # then reports only the first nion fractions. We must ship the +2 stages to reproduce it.
    nion2_z = np.array([min(int(nion_z[z-1]) + 2, _start_and_nions(z)[1], MAXION) for z in range(1, 100)])

    # --- reference electron density: nelect-style charge balance over all 99 elements ---
    xne = P_gas/tk/2.0
    chargesq = np.maximum(2.0*xne, 1e-30)
    for _ in range(300):
        xnatom = P_gas/tk - xne
        new = np.zeros(n); csq = np.zeros(n)
        for j in range(n):
            acc = cs = 0.0
            for z in range(1, 100):
                nion = int(nion_z[z-1])
                f = pfsaha_depth(temperature_k=float(T[j]), electron_density_cm3=float(xne[j]),
                                 xnatom_cm3=float(xnatom[j]), xabund_linear=float(xab[z-1]),
                                 atomic_number=z, nion=nion, mode=12,
                                 chargesq_cm3=float(max(chargesq[j], 1e-30)))
                ions = np.arange(nion)
                contrib = f[:nion]*xnatom[j]*xab[z-1]
                acc += np.sum(contrib*ions); cs += np.sum(contrib*ions*ions)
            new[j] = acc; csq[j] = cs
        new = np.maximum(new, xne*0.5); new = 0.5*(new + xne)
        err = np.max(np.abs((xne-new)/np.maximum(new, 1e-300)))
        xne = new; chargesq = csq + xne
        if err < 1e-8:
            break
    print(f"  L2 reference n_e: converged err={err:.1e}")

    # Debye lowering of the ionization potential (pressure ionization), per layer.
    debye = np.sqrt(tk/(12.5664*(4.801e-10**2)*np.maximum(chargesq, 1e-30)))
    potlow = np.minimum(1.0, 1.44e-7/np.maximum(debye, 1e-300))   # eV

    # --- partition functions U and ionization potentials chi for ALL elements ---
    # Padded arrays: U[layer, Z-1, ion], chi[Z-1, ion]; ion in 0..MAXION-1.
    U = np.ones((n, 99, MAXION)); chi = np.zeros((99, MAXION)); fracH = np.zeros((n, 2))
    for z in range(1, 100):
        ni2 = int(nion2_z[z-1])
        for j in range(n):
            U[j, z-1, :ni2] = pfsaha_depth(
                temperature_k=float(T[j]), electron_density_cm3=float(xne[j]),
                xnatom_cm3=1.0, xabund_linear=1.0, atomic_number=z, nion=ni2,
                mode=13, chargesq_cm3=float(chargesq[j]))[:ni2]
        cum = pfsaha_depth(temperature_k=float(T[40]), electron_density_cm3=float(xne[40]),
                           xnatom_cm3=1.0, xabund_linear=1.0, atomic_number=z, nion=ni2,
                           mode=15, chargesq_cm3=float(chargesq[40]))   # ionization potentials (T-independent)
        chi[z-1, :ni2] = np.diff(cum[31:31+ni2+1])[:ni2]
    for j in range(n):   # reference hydrogen ionization fractions for the benchmark
        fracH[j] = pfsaha_depth(temperature_k=float(T[j]), electron_density_cm3=float(xne[j]),
                                xnatom_cm3=1.0, xabund_linear=1.0, atomic_number=1, nion=2,
                                mode=12, chargesq_cm3=float(chargesq[j]))[:2]

    out = REF / "L2.npz"
    np.savez(out, tau=tau, T=T, P_gas=P_gas, rhox=rhox, tk=tk, xne=xne, chargesq=chargesq,
             potlow=potlow, xabund=xab, nion=nion_z, nion2=nion2_z, U=U, chi=chi, fracH=fracH)
    print(f"wrote {out}")

    # self-check: from-scratch charge balance (Saha ladder over nion2, charges summed over nion,
    # Debye from the evolving chargesq) reproduces the reference n_e — mirrors nelect exactly.
    SAHA, KEV = 2.4148e15, 1.0/11604.5
    xs = P_gas/tk/2.0; csq_it = np.maximum(2.0*xs, 1e-30)
    for _ in range(300):
        xnatom = P_gas/tk - xs
        pl = np.minimum(1.0, 1.44e-7/np.sqrt(tk/(12.5664*(4.801e-10**2)*np.maximum(csq_it, 1e-30))))
        ne_new = np.zeros(n); cs_new = np.zeros(n)
        for z in range(1, 100):
            ni = int(nion_z[z-1]); ni2 = int(nion2_z[z-1]); r = np.ones((n, ni2))
            for i in range(1, ni2):
                r[:, i] = r[:, i-1]*(2.0*U[:, z-1, i]/np.maximum(U[:, z-1, i-1], 1e-300)) \
                          * SAHA*T**1.5*np.exp(-(chi[z-1, i-1]-pl*i)/(KEV*T))/xs
            f = r/r.sum(axis=1, keepdims=True)
            contrib = f*xnatom[:, None]*xab[z-1]; ions = np.arange(ni2)
            ne_new += (contrib[:, :ni]*ions[:ni]).sum(axis=1)
            cs_new += (contrib[:, :ni]*ions[:ni]**2).sum(axis=1)
        ne_new = np.maximum(ne_new, xs*0.5); ne_new = 0.5*(ne_new+xs)
        e = np.max(np.abs((xs-ne_new)/np.maximum(ne_new, 1e-300))); xs = ne_new; csq_it = cs_new + xs
        if e < 1e-8: break
    print(f"  L2 self-check (nion2 ladder): n_e vs reference max|rel|={np.max(np.abs(xs-xne)/xne):.2e}")


# small import shims (kept local so the module top stays clean)
def pykurucz_compute_abund():
    import pykurucz; return pykurucz.compute_abundances(mh=0.0, am=0.0)
def pykurucz_compute_h():
    import pykurucz; return pykurucz.compute_h(mh=0.0, am=0.0)
def pykurucz_HE():
    import pykurucz; return pykurucz.HE_ABUNDANCE


def lecture3() -> None:
    """Reference continuum opacity (absorption + scattering, cm^2/g) on the canonical
    atmosphere, plus the inputs a from-scratch continuum needs (T, n_e, rho, n(H I))."""
    from synthe_py.io.atmosphere import load_cached
    from synthe_py.physics.continuum import build_depth_continuum
    atm = load_cached(REF / "atmosphere.npz")
    wl = np.linspace(500.0, 510.0, 200)
    absb, scat = build_depth_continuum(atm, wl)[:2]
    d = np.load(REF / "atmosphere.npz")
    out = REF / "L3.npz"
    np.savez(out, wl=wl, absorption=absb, scattering=scat,
             T=atm.temperature, n_e=atm.electron_density, rho=atm.mass_density,
             nHI=d["xnf_h"], tau=np.load(REF / "L2.npz")["tau"], rhox=np.load(REF / "L2.npz")["rhox"])
    print(f"wrote {out}  (continuum {absb.shape}, abs {absb[50,100]:.3e} scat {scat[50,100]:.3e} cm^2/g)")


def lecture4() -> None:
    """Reference Voigt profile H(a,v) over a grid, from pykurucz's tabulated approximation."""
    from synthe_py.physics.voigt_jit import voigt_profile_jit
    from synthe_py.physics.tables import VoigtTables
    vt = VoigtTables.build()
    v = np.linspace(0.0, 12.0, 481)
    a = np.array([0.001, 0.01, 0.05, 0.2, 0.5, 1.0])
    Hgrid = np.array([[voigt_profile_jit(float(vv), float(aa), vt.h0tab, vt.h1tab, vt.h2tab)
                       for vv in v] for aa in a])
    # photospheric Fe context for a single-line example (reuse the L2 equation of state)
    D = np.load(REF / "L2.npz"); T = D["T"]; tk = D["tk"]; ne = D["xne"]
    xab = D["xabund"]; U = D["U"]; chi = D["chi"]; nion = D["nion"]
    SAHA, KEV = 2.4148e15, 1.0/11604.5
    n_atom = D["P_gas"]/tk - ne
    Z = 26; ni = int(nion[Z-1]); r = np.ones((T.size, ni))
    cs = 2.0*ne; pl = np.minimum(1.0, 1.44e-7/np.sqrt(tk/(12.5664*(4.801e-10**2)*cs)))
    for i in range(1, ni):
        r[:, i] = r[:, i-1]*(2.0*U[:, Z-1, i]/U[:, Z-1, i-1])*SAHA*T**1.5*np.exp(-(chi[Z-1, i-1]-pl*i)/(KEV*T))/ne
    fFe = r/r.sum(axis=1, keepdims=True)
    n_FeI = fFe[:, 0]*xab[Z-1]*n_atom
    A = np.load(REF / "atmosphere.npz"); rho = A["mass_density"]; nHI = A["xnf_h"]
    np.savez(REF / "L4.npz", v=v, a=a, H=Hgrid, T=T, tk=tk, xne=ne, rho=rho, nHI=nHI,
             n_FeI=n_FeI, U_FeI=U[:, Z-1, 0], tau=D["tau"],
             h0tab=vt.h0tab, h1tab=vt.h1tab, h2tab=vt.h2tab)   # Voigt Harris tables (data)
    from scipy.special import wofz
    He = np.array([[wofz(vv + 1j*aa).real for vv in v] for aa in a])
    print(f"wrote {REF/'L4.npz'}  (Voigt grid {Hgrid.shape}); "
          f"exact-Faddeeva vs tabulated approx: max|abs|={np.max(np.abs(He-Hgrid)):.2e}")


_SYM = ["", "H","HE","LI","BE","B","C","N","O","F","NE","NA","MG","AL","SI","P","S","CL","AR","K","CA",
        "SC","TI","V","CR","MN","FE","CO","NI","CU","ZN"]  # Z=1..30 (enough for optical metals)


def lecture5() -> None:
    """Read the gfall line list, ship the 500-510 nm window, and benchmark a
    from-scratch total line opacity against the synthe diagnostics."""
    from synthe_py.io.lines.atomic import load_catalog
    cat = load_catalog(Path("/Users/ysting/pykurucz/lines/gfallvac.latest"))
    wl_line = cat.wavelength
    m = (wl_line >= 499.8) & (wl_line <= 510.2)
    print(f"  gfall: {wl_line.size} lines total, {m.sum()} in 499.8-510.2 nm")
    # element symbol -> Z
    sym2Z = {s: i for i, s in enumerate(_SYM)}
    Z = np.array([sym2Z.get(str(e).upper().strip(), 0) for e in cat.elements[m]], dtype=int)
    keep = Z > 0
    sub = dict(wl=wl_line[m][keep], loggf=cat.log_gf[m][keep],
               Elow_cm=cat.excitation_energy[m][keep], Z=Z[keep],
               ion=cat.ion_stages[m][keep].astype(int),
               grad=cat.gamma_rad[m][keep], gstark=cat.gamma_stark[m][keep], gvdw=cat.gamma_vdw[m][keep])
    # ship the reference per-depth line opacity + the atmosphere arrays the lecture needs
    d = np.load(REF / "diag.npz"); L2 = np.load(REF / "L2.npz")
    A = np.load(REF / "atmosphere.npz")
    sub.update(grid_wl=d["wavelength"], line_opacity=d["line_opacity"].astype(np.float32),
               T=L2["T"], tk=L2["tk"], xne=L2["xne"], rhox=L2["rhox"],
               rho=A["mass_density"], nHI=A["xnf_h"], P_gas=L2["P_gas"])
    np.savez(REF / "L5.npz", **sub)
    print(f"wrote {REF/'L5.npz'}  ({sub['wl'].size} atomic lines Z<=30; "
          f"ref line_opacity {d['line_opacity'].shape})")


def lecture6() -> None:
    """Reference for Lecture 6: the assembled total opacity (continuum + lines) per depth
    and wavelength, plus the reference emergent + continuum flux, for the formal solution."""
    d = np.load(REF / "diag.npz")
    L2 = np.load(REF / "L2.npz")
    total_abs = (d["continuum_absorption"] + d["line_opacity"]).astype(np.float32)
    total_scat = (d["continuum_scattering"] + d["line_scattering"]).astype(np.float32)
    np.savez(REF / "L6.npz", wl=d["wavelength"], total_abs=total_abs, total_scat=total_scat,
             cont_abs=d["continuum_absorption"].astype(np.float32),
             cont_scat=d["continuum_scattering"].astype(np.float32),
             flux_total=d["flux_total"], flux_continuum=d["flux_continuum"],
             T=L2["T"], rhox=L2["rhox"], tau=L2["tau"])
    print(f"wrote {REF/'L6.npz'}  (opacity {total_abs.shape}, flux {d['flux_total'].shape})")


def canonical_run() -> None:
    """Write our grey+EOS solar atmosphere as a .atm and run pykurucz convert+synthe on it.

    Produces the single consistent reference run that L3 (continuum), L5 (line opacity),
    and L6 (spectrum) all check against: reference/atmosphere.atm, .npz, .spec.
    Atomic lines only (molecules off) over the 500-510 nm window.
    """
    import pykurucz
    D = np.load(REF / "L2.npz")
    tau, T, P_gas, rhox, xne = D["tau"], D["T"], D["P_gas"], D["rhox"], D["xne"]
    n = T.size
    # 9-column READ DECK6 deck: RHOX, T, P, XNE, ABROSS, ACCRAD, VTURB, FLXCNV, VCONV
    deck = np.column_stack([
        rhox, T, P_gas, xne,
        np.ones(n),                 # ABROSS placeholder (synthe recomputes opacity from the EOS)
        np.zeros(n),                # ACCRAD
        np.full(n, 2.0e5),          # VTURB = 2 km/s in cm/s
        np.zeros(n), np.zeros(n),   # FLXCNV, VCONV
    ])
    atm = REF / "atmosphere.atm"
    pykurucz.write_atm_file(atm, 5770.0, 4.44, deck, 2.0, mh=0.0, am=0.0,
                            title="grey+EOS solar atmosphere (Stellar Spectroscopy from Scratch)")
    print(f"wrote {atm}")

    npz = REF / "atmosphere.npz"
    spec = REF / "spectrum_500_510.spec"
    log = REF / "canonical_run.log"
    pykurucz.run_synthe_py(atm, spec=spec, npz=npz, log_path=log,
                           wl_start=500.0, wl_end=510.0, resolution=300_000.0,
                           use_molecular_lines=False, npz_cache=False)
    sp = np.loadtxt(spec)
    print(f"wrote {npz} and {spec}  (spectrum: {len(sp)} points, "
          f"flux {sp[:,1].min():.3e}..{sp[:,1].max():.3e})")


if __name__ == "__main__":
    import sys as _sys
    which = _sys.argv[1:] or ["1", "2"]
    if "1" in which: lecture1()
    if "2" in which: lecture2()
    if "canonical" in which: canonical_run()
    if "3" in which: lecture3()
    if "4" in which: lecture4()
    if "5" in which: lecture5()
    if "6" in which: lecture6()
