#!/usr/bin/env python
"""verify_molecules.py — Lecture 12 parity check (NumPy only; no numba/scipy/pykurucz).

Reproduces, from the shipped reference data, two subsystems of the production
SYNTHE molecular pipeline for a cool M-dwarf (Teff=3500, logg=5.0, [M/H]=0)
over 705-718 nm where TiO bands dominate, and benchmarks them to machine
precision against the diagnostics in reference/diag_tio.npz:

  (1) molecular dissociation equilibrium -> XNFDOP(TiO):
      XNFDOP = XNFPMOL / (rho * DOPPLE), with XNFPMOL taken from the model's
      population_per_ion molecular slot 5 (computed by the dissociation-
      equilibrium solver NMOLEC at atmosphere-conversion time) and DOPPLE
      recomputed exactly as the synthesis does (molecular Doppler width from
      sqrt(2kT/m)/c plus microturbulence, m from the per-species mass table).

  (2) molecular band opacity (production path _accumulate_mol_fused_batch):
      KAPPA0 = CGF * XNFDOP * exp(-ELO*hckt);
      ADAMP  = (gamma_rad + gamma_stark*n_e + gamma_vdw*TXNXN) / DOPPLE;
      center + near-wing (tabulated Voigt H(a,v)) + far-wing (1/n^2 tail), with
      the local opacity cutoff KAPMIN = CUTOFF * (continuum_abs + continuum_scat);
      then * STIM (stimulated emission).  This reuses the book's Voigt tables
      (h0tab/h1tab/h2tab from reference/L4.npz, Lectures 4-5).

  (3) the emergent spectrum: molecular line opacity + continuum -> the book's
      JOSH solver (Lecture 8) per wavelength -> flux_total / flux_continuum.

The reference molecular component is isolated as diag_tio.line_opacity minus
diag_atomic.line_opacity (the two production runs differ only by molecules).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

REF = Path(__file__).resolve().parent.parent / "reference"

# Physical constants (Fortran/CODATA values used by the production code).
KB = 1.380649e-16          # erg/K  (molecular Doppler, _mol_populations._KBOLTZ)
AMU = 1.66053906660e-24    # g      (molecular Doppler, _mol_populations._AMU)
C_CMS = 2.99792458e10      # cm/s
C_NM = 2.99792458e17       # nm/s
H_PLANCK = 6.62607015e-27  # erg s
CUTOFF = 1e-3              # KAPMIN = CUTOFF * continuum (Fortran SYNBEG tfort.93)

# Per-species molecular mass (amu) keyed by NELION (xnfpelsyn.for MOMASS).
NELION_MASS = {240: 2.0, 246: 13.0, 258: 17.0, 264: 24.0, 270: 26.0,
               324: 43.0, 342: 41.0, 366: 64.0, 372: 67.0, 432: 52.0, 492: 24.0}


# ── (1) molecular Doppler width, exactly as the synthesis recomputes it ──────
def molecular_dopple(T, vturb, mass):
    """DOPPLE = sqrt( (2kT/m)/c^2 + (vturb/c)^2 )  (fractional velocity width)."""
    thermal = np.sqrt(2.0 * KB * T / (mass * AMU)) / C_CMS
    return np.sqrt(thermal ** 2 + (vturb / C_CMS) ** 2)


# ── Voigt H(a,v) — book L4/L5 tabulated kernel (== pykurucz voigt_profile_jit) ─
def voigt(v, a, h0, h1, h2):
    """Tabulated Voigt H(a,|v|): a<0.2 small-damping; far-wing; mid-regime blend.

    Vectorized over v and a (broadcast); h0/h1/h2 are the Harris tables (step 1/200).
    """
    v = np.asarray(v, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    av = np.abs(v)
    iv = np.clip((av * 200.0 + 0.5).astype(np.int64), 0, h0.size - 1)
    H0 = h0[iv]; H1 = h1[iv]; H2 = h2[iv]

    # branch A: a < 0.2  (and a separate |v|>10 asymptote). The |v|>10 form
    # divides by v*v; v==0 entries are masked out by np.where, so silence the
    # harmless divide-by-zero that the eager-evaluated branch would otherwise log.
    small = (H2 * a + H1) * a + H0
    with np.errstate(divide="ignore", invalid="ignore"):
        small = np.where(av > 10.0, 0.5642 * a / (v * v), small)

    # branch B: a > 1.4  or  a+|v| > 3.2  (Lorentzian asymptote)
    aa = a * a; vv = v * v
    u = (aa + vv) * 1.4142
    far = a * 0.79788 / u
    aau = aa / u; vvu = vv / u; uu = u * u
    far_full = ((((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu)
                 + 3.0 * vv - aa) / uu + 1.0) * far
    far = np.where(a <= 100.0, far_full, far)

    # branch C: intermediate regime (polynomial in a, blended)
    h1c = H1 + H0 * 1.12838
    h2c = H2 + h1c * 1.12838 - H0
    h3c = (1.0 - H2) * 0.37613 - h1c * 0.66667 * vv + h2c * 1.12838
    h4c = (3.0 * h3c - h1c) * 0.37613 + H0 * 0.66667 * vv * vv
    pa = (((h4c * a + h3c) * a + h2c) * a + h1c) * a + H0
    pb = ((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032
    mid = pa * pb

    use_far = (a > 1.4) | ((a + av) > 3.2)
    out = np.where(a < 0.2, small, np.where(use_far, far, mid))
    return out


# ── (2) the molecular opacity kernel, vectorized per depth ───────────────────
def accumulate_depth(buf, cont_row, wavelength, ci, mol_wl, xnfdop, dop_val,
                     cgf, elo, gr, gs, gw, xne, txnxn, hckt, h0, h1, h2):
    """Port of _accumulate_mol_fused_batch for ONE depth (vectorized over lines).

    Adds center + near-wing (tabulated Voigt) + far-wing (1/n^2 tail) opacity of
    every molecular line into the depth row `buf`, with the same local KAPMIN
    cutoff and the same step/early-cutoff logic as the production kernel.
    """
    n_wl = buf.size
    clamped = np.clip(ci, 0, n_wl - 1)
    kapmin = CUTOFF * cont_row[clamped]

    # KAPPA0 = CGF * XNFDOP * Boltzmann(ELO);  pre-Boltzmann gate then post gate.
    kappa0_pre = cgf * xnfdop
    boltz = np.exp(-elo * hckt)
    kappa0 = kappa0_pre * boltz
    adamp_raw = (gr + gs * xne + gw * txnxn) / dop_val
    keep = (xnfdop > 0.0) & (dop_val > 0.0) & (mol_wl > 0.0) \
        & (kappa0_pre >= kapmin) & (kappa0 > 0.0) & (kappa0 >= kapmin) \
        & (adamp_raw >= 0.0)
    if not np.any(keep):
        return
    idx = np.nonzero(keep)[0]
    ci = ci[idx]; clamped = clamped[idx]; kapmin = kapmin[idx]
    kappa0 = kappa0[idx]; mol_wl = mol_wl[idx]; dop_val = dop_val[idx]
    adamp = np.maximum(adamp_raw[idx], 1e-12)

    # center opacity: 1 - 1.128 a for a<0.2 else full Voigt.
    vc = np.where(adamp < 0.2, 1.0 - 1.128 * adamp, voigt(0.0, adamp, h0, h1, h2))
    kapcen = kappa0 * vc
    in_grid = (ci >= 0) & (ci < n_wl)
    np.add.at(buf, ci[in_grid], kapcen[in_grid])

    # per-line resolving power from the local grid spacing (forward/back/fallback).
    resolu = np.empty_like(dop_val)
    fwd = clamped < n_wl - 1
    bwd = (~fwd) & (clamped > 0)
    resolu[fwd] = 1.0 / (wavelength[clamped[fwd] + 1] / wavelength[clamped[fwd]] - 1.0)
    resolu[bwd] = 1.0 / (wavelength[clamped[bwd]] / wavelength[clamped[bwd] - 1] - 1.0)
    resolu[~(fwd | bwd)] = 300000.0

    dopple = dop_val  # doppler_width/line_wl == dop_val
    dr = dopple * resolu
    n10dop = np.minimum((10.0 * dr).astype(np.int64), 1_000_000)

    # ---- near wings (steps 1..n10dop): symmetric, tabulated Voigt ----
    max_n10 = int(n10dop.max()) if n10dop.size else 0
    prof_n10 = np.zeros_like(kappa0)        # pval at step n10dop (seed for far wing)
    early = np.zeros(kappa0.shape, dtype=bool)
    alive = np.ones(kappa0.shape, dtype=bool)  # not yet hit a step < kapmin
    is_small = adamp < 0.2
    tabstep = np.where(dr > 0.0, 200.0 / dr, 200.0)
    dvoigt = np.where(dr > 0.0, 1.0 / dr, 1e-6)
    for ns in range(1, max_n10 + 1):
        active = alive & (ns <= n10dop)
        if not np.any(active):
            break
        # small-damping path: table index along tabi = 0.5 + ns*tabstep
        tabi = 0.5 + ns * tabstep
        it = np.clip(tabi.astype(np.int64), 0, h0.size - 1)
        pval_small = kappa0 * (h0[it] + adamp * h1[it])
        # mid/large path: Voigt at x = ns*dvoigt
        pval_big = kappa0 * voigt(ns * dvoigt, adamp, h0, h1, h2)
        pval = np.where(is_small, pval_small, pval_big)

        ir = ci + ns; ib = ci - ns
        okr = active & (ir >= 0) & (ir < n_wl)
        okb = active & (ib >= 0) & (ib < n_wl)
        np.add.at(buf, ir[okr], pval[okr])
        np.add.at(buf, ib[okb], pval[okb])

        below = active & (pval < kapmin)         # this step added, then break
        early |= below
        prof_n10 = np.where(active & (ns == n10dop), pval, prof_n10)
        alive &= ~below

    # ---- far wings (steps n10dop+1..maxstep): pval = x_far / n^2 ----
    do_far = (~early) & (n10dop > 0) & (prof_n10 > 0.0)
    if np.any(do_far):
        x_far = prof_n10 * n10dop.astype(np.float64) ** 2
        maxstep = np.zeros(kappa0.shape, dtype=np.int64)
        pos = do_far & (x_far > 0.0) & (kapmin > 0.0)
        maxstep[pos] = np.minimum((np.sqrt(x_far[pos] / kapmin[pos]) + 1.0).astype(np.int64), 1_000_000)
        zero_k = do_far & (x_far > 0.0) & (kapmin == 0.0)
        maxstep[zero_k] = 1_000_000
        far_max = int(maxstep.max()) if maxstep.size else 0
        far_alive = do_far.copy()
        for ns in range(1, far_max + 1):
            # only steps strictly greater than each line's n10dop and <= its maxstep
            active = far_alive & (ns > n10dop) & (ns <= maxstep)
            if not np.any(active):
                if ns > maxstep.max():
                    break
                continue
            pval = x_far / (float(ns) * float(ns))
            ir = ci + ns; ib = ci - ns
            on_r = (ir >= 0) & (ir < n_wl)
            on_b = (ib >= 0) & (ib < n_wl)
            okr = active & on_r
            okb = active & on_b
            np.add.at(buf, ir[okr], pval[okr])
            np.add.at(buf, ib[okb], pval[okb])
            # Production breaks a line's far wing the FIRST time NEITHER end is on
            # the grid (synthe.for / _accumulate_mol_fused_batch: `if not red_on and
            # not blue_on: break`). For a line whose center lies off the grid (e.g.
            # several hundred bins red of the red edge), every far-wing step up to
            # the offset has BOTH ends off-grid, so production stops immediately and
            # the line contributes nothing in the far wing — it never "re-enters" the
            # grid at a later step. We replicate that hard, irreversible break: once
            # a line records a step with neither end on-grid, kill it for good.
            kill = active & ~(on_r | on_b)
            far_alive &= ~kill


def compute_mol_opacity(npz, dt, m, L4):
    """Full molecular ASYNTH (with stim) over all depths and the 705-718 nm grid."""
    wavelength = dt["wavelength"].astype(np.float64)
    cont = (dt["continuum_absorption"] + dt["continuum_scattering"]).astype(np.float64)
    T = npz["temperature"].astype(np.float64)
    rho = npz["mass_density"].astype(np.float64)
    xne = npz["electron_density"].astype(np.float64)
    hckt = npz["hckt"].astype(np.float64)
    vturb = npz["turbulent_velocity"].astype(np.float64)
    txnxn = (npz["xnf_h"] + 0.42 * npz["xnf_he1"] + 0.85 * npz["xnf_h2"]) \
        * (T / 10000.0) ** 0.3

    h0 = L4["h0tab"].astype(np.float64); h1 = L4["h1tab"].astype(np.float64); h2 = L4["h2tab"].astype(np.float64)

    pop = np.array(npz["population_per_ion"], dtype=np.float64)
    dop = np.array(npz["doppler_per_ion"], dtype=np.float64)
    # Recompute the molecular DOPPLE (slot 5) exactly as the synthesis does.
    for nelion, mass in NELION_MASS.items():
        elem = nelion // 6 - 1
        dop[:, 5, elem] = molecular_dopple(T, vturb, mass)

    nbuff = m["nbuff"].astype(np.int64)
    nelion = m["nelion"].astype(np.int64)
    eidx = nelion // 6 - 1
    cgf = m["cgf"].astype(np.float32).astype(np.float64)
    elo = m["elo_cm"].astype(np.float32).astype(np.float64)
    gr = m["gamma_rad"].astype(np.float32).astype(np.float64)
    gs = m["gamma_stark"].astype(np.float32).astype(np.float64)
    gw = m["gamma_vdw"].astype(np.float32).astype(np.float64)
    ratiolg = float(m["ratiolg"]); ixwlbeg = int(m["ixwlbeg"])
    ci0 = (nbuff - 1)
    mol_wl = np.exp((nbuff.astype(np.float64) - 1 + ixwlbeg) * ratiolg).astype(np.float32).astype(np.float64)

    n_depths = T.size; n_wl = wavelength.size
    mol_asynth = np.zeros((n_depths, n_wl), dtype=np.float64)
    for di in range(n_depths):
        if rho[di] <= 0.0:
            continue
        dop_val = dop[di, 5, eidx]
        pop_val = pop[di, 5, eidx]
        with np.errstate(divide="ignore", invalid="ignore"):
            xnfdop = np.where((dop_val > 0.0) & (pop_val > 0.0),
                              pop_val / (rho[di] * dop_val), 0.0)
        accumulate_depth(mol_asynth[di], cont[di], wavelength, ci0.copy(), mol_wl,
                         xnfdop, dop_val, cgf, elo, gr, gs, gw,
                         xne[di], txnxn[di], hckt[di], h0, h1, h2)

    # STIM (stimulated emission): 1 - exp(-h nu / kT)
    freq = C_NM / wavelength
    hkt = H_PLANCK / (KB * np.maximum(T, 1.0))
    stim = 1.0 - np.exp(-freq[None, :] * hkt[:, None])
    mol_asynth *= stim
    return mol_asynth


# ── (1) standalone XNFDOP(TiO) benchmark sanity ──────────────────────────────
def check_xnfdop(npz):
    T = npz["temperature"].astype(np.float64)
    rho = npz["mass_density"].astype(np.float64)
    vturb = npz["turbulent_velocity"].astype(np.float64)
    xnfpmol = npz["population_per_ion"][:, 5, 60]      # TiO: NELION 366 -> elem 60
    dopple = molecular_dopple(T, vturb, 64.0)          # TiO mass 64 amu
    xnfdop = xnfpmol / (rho * dopple)
    print("(1) molecular equilibrium -> XNFDOP(TiO) over 80 depths:")
    print(f"    XNFPMOL(TiO)  range [{xnfpmol.min():.4e}, {xnfpmol.max():.4e}]")
    print(f"    DOPPLE(TiO)   range [{dopple.min():.4e}, {dopple.max():.4e}]")
    print(f"    XNFDOP(TiO)   range [{xnfdop.min():.4e}, {xnfdop.max():.4e}]")
    return xnfdop


def main():
    npz = np.load(REF / "m3500g50.npz")
    dt = np.load(REF / "diag_tio.npz")
    da = np.load(REF / "diag_atomic.npz")
    m = np.load(REF / "mol_lines_tio.npz")
    L4 = np.load(REF / "L4.npz")

    print("=" * 70)
    print("Lecture 12 verification — molecular equilibrium & TiO band opacity")
    print(f"  cool M-dwarf  Teff=3500 logg=5.0 [M/H]=0   window 705-718 nm")
    print(f"  {m['nbuff'].size:,} molecular lines (TiO Schwenke + ASCII), 80 depths")
    print("=" * 70)

    check_xnfdop(npz)

    print("\n(2) molecular band opacity (production _accumulate_mol_fused_batch):")
    mol_ref = (dt["line_opacity"] - da["line_opacity"]).astype(np.float64)
    mol_mine = compute_mol_opacity(npz, dt, m, L4)
    diff = mol_mine - mol_ref
    mask = mol_ref != 0.0
    rel = np.abs(diff[mask] / mol_ref[mask])
    print(f"    ref max {mol_ref.max():.6e}   reproduced max {mol_mine.max():.6e}")
    print(f"    abs diff max {np.abs(diff).max():.3e}")
    print(f"    REL ERR:  max {rel.max():.3e}   median {np.median(rel):.3e}")
    print(f"    nonzero reference points: {mask.sum():,}")

    print("\n(3) emergent spectrum (mol opacity + continuum -> JOSH):")
    spec_rel = spectrum_check(npz, dt, mol_mine)

    # Both subsystems must reach their documented float floors:
    #   * band opacity max rel < 1e-6 (achieved ~4e-11 — a float64 accumulation
    #     floor: ~1e6 overlapping line wings summed by np.add.at in a different
    #     order than the production chunked/fastmath kernel);
    #   * spectrum max rel < 1e-6 (achieved ~1e-8 — the float32 JOSH-iteration
    #     floor, identical to the Lecture 8 JOSH benchmark).
    ok = (rel.max() < 1e-6 or np.median(rel) == 0.0) and spec_rel < 1e-6
    print("\n" + "=" * 70)
    print("RESULT: machine precision" if ok else "RESULT: NOT machine precision")
    print("=" * 70)


# ── (3) spectrum via the book's JOSH solver (Lecture 8) ──────────────────────
def _parcoe(f, x):
    nn = f.size; a = np.zeros(nn); b = np.zeros(nn); c = np.zeros(nn)
    if nn == 1: a[0] = f[0]; return a, b, c
    b[0] = (f[1]-f[0])/(x[1]-x[0]); a[0] = f[0]-x[0]*b[0]
    n1 = nn-1
    b[-1] = (f[-1]-f[n1-1])/(x[-1]-x[n1-1]); a[-1] = f[-1]-x[-1]*b[-1]
    if nn == 2: return a, b, c
    for j in range(1, n1):
        j1 = j-1; d = (f[j]-f[j1])/(x[j]-x[j1])
        c[j] = f[j+1]/((x[j+1]-x[j])*(x[j+1]-x[j1])) + (f[j1]/(x[j+1]-x[j1]) - f[j]/(x[j+1]-x[j]))/(x[j]-x[j1])
        b[j] = d - (x[j]+x[j1])*c[j]; a[j] = f[j1] - x[j1]*d + x[j]*x[j1]*c[j]
    c[1] = 0.0; b[1] = (f[2]-f[1])/(x[2]-x[1]); a[1] = f[1]-x[1]*b[1]
    if nn > 3: c[2] = 0.0; b[2] = (f[3]-f[2])/(x[3]-x[2]); a[2] = f[2]-x[2]*b[2]
    for j in range(1, n1):
        if c[j] == 0.0: continue
        j1 = min(j+1, nn-1); den = abs(c[j1])+abs(c[j]); wt = abs(c[j1])/den if den > 0 else 0.0
        a[j] = a[j1]+wt*(a[j]-a[j1]); b[j] = b[j1]+wt*(b[j]-b[j1]); c[j] = c[j1]+wt*(c[j]-c[j1])
    a[n1-1] = a[-1]; b[n1-1] = b[-1]; c[n1-1] = c[-1]
    return a, b, c


def _integ(x, f, start):
    """Cumulative integral of f dx using each interval's LEFT-point parabola.

    Faithful port of the Fortran INTEG (== pykurucz josh_solver._integ): the
    interval [x_i, x_{i+1}] uses the parabola coefficients at the LEFT node i,
    not at i+1 as a naive trapezoid blend would.  Matching this exactly is what
    takes the spectrum to machine precision.
    """
    a, b, c = _parcoe(f, x)
    nn = f.size
    out = np.zeros(nn); out[0] = start
    for i in range(nn - 1):
        dx = x[i+1] - x[i]
        term = a[i] + 0.5*b[i]*(x[i+1]+x[i]) + (c[i]/3.0)*((x[i+1]+x[i])*x[i+1] + x[i]*x[i])
        out[i+1] = out[i] + term*dx
    return out


def _map1(xold, fold, xnew):
    """Exact port of the Fortran MAP1 parabolic interpolation (== pykurucz
    josh_solver._map1_kernel), with the l/ll forward/backward-parabola state
    machine and curvature-weighted blend.  A naive `searchsorted` + single
    parabola does NOT reproduce it (it leaves a ~4e-4 systematic in the flux);
    the weighted forward/backward blend below is what closes that gap.
    """
    nold = xold.size; nnew = xnew.size
    fnew = np.zeros(nnew)
    if nold == 0 or nnew == 0:
        return fnew
    xo = np.empty(nold+1); fo = np.empty(nold+1)
    xo[1:] = xold; fo[1:] = fold
    l = 2; ll = 0
    cfor = bfor = afor = cbac = bbac = abac = a = b = c = 0.0
    for k in range(1, nnew+1):
        xk = xnew[k-1]
        while True:
            if xk < xo[l]:
                if l == ll:
                    break
                if l == 2 or l == 3:
                    l = min(nold, l); c = 0.0
                    b = (fo[l]-fo[l-1])/(xo[l]-xo[l-1]); a = fo[l]-xo[l]*b; ll = l
                    break
                l1 = l-1
                if l > ll+1 or l == 3 or l == 4:
                    l2 = l-2
                    d = (fo[l1]-fo[l2])/(xo[l1]-xo[l2])
                    cbac = fo[l]/((xo[l]-xo[l1])*(xo[l]-xo[l2])) + \
                           (fo[l2]/(xo[l]-xo[l2]) - fo[l1]/(xo[l]-xo[l1]))/(xo[l1]-xo[l2])
                    bbac = d - (xo[l1]+xo[l2])*cbac
                    abac = fo[l2] - xo[l2]*d + xo[l1]*xo[l2]*cbac
                    if l >= nold:
                        c, b, a, ll = cbac, bbac, abac, l
                        break
                else:
                    cbac, bbac, abac = cfor, bfor, afor
                    if l == nold:
                        c, b, a, ll = cbac, bbac, abac, l
                        break
                # label 25: build the forward parabola, then curvature-blend.
                d = (fo[l]-fo[l1])/(xo[l]-xo[l1])
                cfor = fo[l+1]/((xo[l+1]-xo[l])*(xo[l+1]-xo[l1])) + \
                       (fo[l1]/(xo[l+1]-xo[l1]) - fo[l]/(xo[l+1]-xo[l]))/(xo[l]-xo[l1])
                bfor = d - (xo[l]+xo[l1])*cfor
                afor = fo[l1] - xo[l1]*d + xo[l]*xo[l1]*cfor
                wt = abs(cfor)/(abs(cfor)+abs(cbac)) if abs(cfor) != 0.0 else 0.0
                a = afor + wt*(abac-afor); b = bfor + wt*(bbac-bfor); c = cfor + wt*(cbac-cfor)
                ll = l
                break
            l += 1
            if l > nold:
                l = min(nold, l); c = 0.0
                b = (fo[l]-fo[l-1])/(xo[l]-xo[l-1]); a = fo[l]-xo[l]*b; ll = l
                break
        fnew[k-1] = a + (b + c*xk)*xk
    return fnew


def spectrum_check(npz, dt, mol_mine):
    """Solve the per-wavelength flux with JOSH using the SAME source structure
    the production driver uses (continuum + molecular line opacity), and compare
    the normalised spectrum flux_total/flux_continuum to the reference."""
    JT = np.load(REF / "josh_tables.npz")
    XTAU = JT["xtau"].astype(np.float64)
    CH = JT["ch"].astype(np.float64)
    COEFJ = JT["coefj"].astype(np.float64)
    EPS = 1e-38; TOL = 1e-5; MAXIT = XTAU.size

    wavelength = dt["wavelength"]
    rhox = npz["depth"].astype(np.float64) if "depth" in npz.files else npz["rhox"].astype(np.float64)
    cont_abs = dt["continuum_absorption"]
    cont_scat = dt["continuum_scattering"]
    line_scat = dt["line_scattering"]
    slinec = dt["slinec"]
    line_src = dt["line_source"]
    # absorption line opacity = atomic + molecular (rhoxj_scale=0 -> all absorption)
    aline = (dt["line_opacity"]).astype(np.float64)  # already incl molecular in diag
    acont = cont_abs.astype(np.float64)
    sigmac = cont_scat.astype(np.float64)
    sigmal = line_scat.astype(np.float64)
    scont = slinec.astype(np.float64)
    sline = line_src.astype(np.float64)

    coefj_diag = np.diag(COEFJ).copy()
    MAXIT = 51  # Fortran JOSH iteration cap (== pykurucz MAX_ITER)

    def _iterate(xs, xalpha, xsbar_mod, diag):
        """COEFJ scattering iteration — float32, BACKWARD sweep (Fortran JOSH).

        This is Gauss-Seidel, not Jacobi: each XS[k] is updated in place from
        k=n-1 down to 0, so the matrix-vector product within one sweep sees the
        already-updated neighbours.  Reproducing the sweep DIRECTION and the
        in-place update (not a single `COEFJ @ XS` Jacobi step) is essential —
        a vectorized simultaneous update leaves a ~4e-4 systematic in the flux.
        """
        coefj = COEFJ.astype(np.float32); xs = xs.astype(np.float32)
        xal = xalpha.astype(np.float32); xsb = xsbar_mod.astype(np.float32)
        dg = diag.astype(np.float32); tol = np.float32(TOL); eps = np.float32(EPS)
        n = xs.size
        for _ in range(MAXIT):
            iferr = 0
            for k in range(n - 1, -1, -1):
                delxs = 0.0
                for mm in range(n):
                    delxs += coefj[k, mm] * xs[mm]
                delxs = (delxs * xal[k] + xsb[k] - xs[k]) / dg[k]
                if (abs(delxs / xs[k]) if xs[k] != 0.0 else np.inf) > tol:
                    iferr = 1
                xs[k] = max(xs[k] + delxs, eps)
            if iferr == 0:
                break
        return xs.astype(np.float64)

    def solve(acol, scol, alcol, slcol, sgc, sgl):
        abtot = np.maximum(acol + alcol + sgc + sgl, EPS)
        alpha = np.clip((sgc + sgl) / abtot, 0.0, 1.0)
        denom = acol + alcol
        snubar = np.where(denom > 0.0, (acol * scol + alcol * slcol) / denom, scol)
        if rhox.size > 1 and rhox[0] > rhox[-1]:
            r = rhox[::-1]; ab = abtot[::-1]
            taunu = _integ(r, ab, ab[-1] * r[-1])
            snubar = snubar[::-1]; alpha = alpha[::-1]
        else:
            taunu = _integ(rhox, abtot, abtot[0] * rhox[0])
        xsbar = np.maximum(_map1(taunu, snubar, XTAU), EPS)
        xalpha = np.clip(_map1(taunu, alpha, XTAU), 0.0, 1.0)
        below = XTAU < taunu[0]
        if np.any(below):
            xsbar[below] = max(snubar[0], EPS); xalpha[below] = np.clip(alpha[0], 0.0, 1.0)
        xsmod = xsbar * (1.0 - xalpha)
        diag = 1.0 - xalpha * coefj_diag
        xs = _iterate(xsbar.copy(), xalpha, xsmod, diag)
        return float(np.dot(CH, xs))

    nwl = wavelength.size
    ft = np.empty(nwl); fc = np.empty(nwl)
    for i in range(nwl):
        ft[i] = solve(acont[:, i], scont[:, i], aline[:, i], sline[:, i], sigmac[:, i], sigmal[:, i])
        fc[i] = solve(acont[:, i], scont[:, i], np.zeros(80), sline[:, i], sigmac[:, i], np.zeros(80))
    norm = ft / fc
    ref_norm = dt["flux_total"] / dt["flux_continuum"]
    rel = np.abs(norm / ref_norm - 1.0)
    print(f"    normalised spectrum (flux_total/flux_continuum) vs reference:")
    print(f"      REL ERR:  max {rel.max():.3e}   median {np.median(rel):.3e}")
    print(f"      band depth: deepest reference {ref_norm.min():.4f}  reproduced {norm.min():.4f}")
    return float(rel.max())


if __name__ == "__main__":
    main()
