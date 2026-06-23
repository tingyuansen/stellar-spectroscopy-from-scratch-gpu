#!/usr/bin/env python
"""From-scratch JOSH moment solver, verified two ways:
  (A) per-wavelength against pykurucz's solve_josh_flux (solver equivalence),
  (B) assembled normalised spectrum against the reference diag.flux_total/flux_continuum.

Clean reimplementation: same algorithm, same constants, same tables as pykurucz, with
the JIT/guard machinery stripped. Tables (XTAU_GRID, CH_WEIGHTS, COEFJ_MATRIX) are loaded
from pykurucz here for verification; the lecture will ship them as a data file.
"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/Users/ysting/pykurucz")
from synthe_py.physics.josh_tables import XTAU_GRID, CH_WEIGHTS, COEFJ_MATRIX
from synthe_py.physics.josh_solver import solve_josh_flux as pk_solve

EPS, ITER_TOL, MAX_ITER = 1e-38, 1e-5, 51
COEFJ_DIAG = np.diag(COEFJ_MATRIX).copy()

# ── 1. optical-depth integrator (parabolic): tau from column mass & extinction ──
def parcoe(f, x):
    """Parabolic coefficients a,b,c so f ≈ a + b x + c x² per interval (Fortran PARCOE).

    Each interior point gets a parabola through its three neighbours; the 2nd and 3rd
    points are forced linear; a curvature-weighted blend then averages each parabola with
    its right neighbour so the coefficients vary smoothly. (Faithful port of pykurucz.)
    """
    n = f.size
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    if n == 1:
        a[0] = f[0]; return a, b, c
    # linear endpoints
    b[0] = (f[1]-f[0])/(x[1]-x[0]); a[0] = f[0]-x[0]*b[0]
    n1 = n-1
    b[-1] = (f[-1]-f[n1-1])/(x[-1]-x[n1-1]); a[-1] = f[-1]-x[-1]*b[-1]
    if n == 2:
        return a, b, c
    # interior parabolae
    for j in range(1, n1):
        j1 = j-1
        d = (f[j]-f[j1])/(x[j]-x[j1])
        c[j] = f[j+1]/((x[j+1]-x[j])*(x[j+1]-x[j1])) + \
               (f[j1]/(x[j+1]-x[j1]) - f[j]/(x[j+1]-x[j]))/(x[j]-x[j1])
        b[j] = d - (x[j]+x[j1])*c[j]
        a[j] = f[j1] - x[j1]*d + x[j]*x[j1]*c[j]
    # force points 2 and 3 linear
    c[1] = 0.0; b[1] = (f[2]-f[1])/(x[2]-x[1]); a[1] = f[1]-x[1]*b[1]
    if n > 3:
        c[2] = 0.0; b[2] = (f[3]-f[2])/(x[3]-x[2]); a[2] = f[2]-x[2]*b[2]
    # curvature-weighted blend with the right neighbour
    for j in range(1, n1):
        if c[j] == 0.0:
            continue
        j1 = min(j+1, n-1)
        denom = abs(c[j1]) + abs(c[j])
        wt = abs(c[j1])/denom if denom > 0.0 else 0.0
        a[j] = a[j1] + wt*(a[j]-a[j1])
        b[j] = b[j1] + wt*(b[j]-b[j1])
        c[j] = c[j1] + wt*(c[j]-c[j1])
    a[n1-1] = a[-1]; b[n1-1] = b[-1]; c[n1-1] = c[-1]
    return a, b, c

def integ(x, f, start):
    """Cumulative integral of f dx using each interval's LEFT-point parabola (Fortran INTEG)."""
    a, b, c = parcoe(f, x)
    n = f.size
    out = np.zeros(n); out[0] = start
    for i in range(n-1):
        dx = x[i+1]-x[i]
        term = a[i] + 0.5*b[i]*(x[i+1]+x[i]) + (c[i]/3.0)*((x[i+1]+x[i])*x[i+1] + x[i]*x[i])
        out[i+1] = out[i] + term*dx
    return out

# ── 2. MAP1: parabolic interpolation onto the fixed Eddington τ-grid (Fortran MAP1) ──
def map1(xold, fold, xnew):
    nold, nnew = xold.size, xnew.size
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
                did_21 = False
                if l > ll+1 or l == 3 or l == 4:
                    l2 = l-2
                    d = (fo[l1]-fo[l2])/(xo[l1]-xo[l2])
                    cbac = fo[l]/((xo[l]-xo[l1])*(xo[l]-xo[l2])) + \
                           (fo[l2]/(xo[l]-xo[l2]) - fo[l1]/(xo[l]-xo[l1]))/(xo[l1]-xo[l2])
                    bbac = d - (xo[l1]+xo[l2])*cbac
                    abac = fo[l2] - xo[l2]*d + xo[l1]*xo[l2]*cbac
                    did_21 = True
                    if l >= nold:
                        c, b, a, ll = cbac, bbac, abac, l
                        break
                else:
                    cbac, bbac, abac = cfor, bfor, afor
                    if l == nold:
                        c, b, a, ll = cbac, bbac, abac, l
                        break
                # label 25 (forward branch + weighted blend)
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

# ── 3. COEFJ Λ-iteration (float32, backward sweep) (Fortran JOSH inner loop) ──
def josh_iterate(xs, xalpha, xsbar_mod, diag):
    coefj = COEFJ_MATRIX.astype(np.float32); xs = xs.astype(np.float32)
    xal = xalpha.astype(np.float32); xsb = xsbar_mod.astype(np.float32)
    dg = diag.astype(np.float32); tol = np.float32(ITER_TOL); eps = np.float32(EPS)
    n = xs.size
    for _ in range(MAX_ITER):
        iferr = 0
        for k in range(n-1, -1, -1):
            delxs = 0.0
            for m in range(n):
                delxs += coefj[k, m]*xs[m]
            delxs = (delxs*xal[k] + xsb[k] - xs[k])/dg[k]
            if (abs(delxs/xs[k]) if xs[k] != 0.0 else np.inf) > tol:
                iferr = 1
            xs[k] = max(xs[k] + delxs, eps)
        if iferr == 0:
            break
    return xs.astype(np.float64)

# ── 4. assemble the surface (Eddington) flux for one wavelength ──
def solve_josh(acont, scont, aline, sline, sigmac, sigmal, rhox):
    abtot = np.maximum(acont + aline + sigmac + sigmal, EPS)
    alpha = np.clip((sigmac + sigmal)/abtot, 0.0, 1.0)
    denom = acont + aline
    snubar = np.where(denom > 0, (acont*scont + aline*sline)/denom, scont)
    needs_reverse = rhox.size > 1 and rhox[0] > rhox[-1]
    if needs_reverse:
        r, ab = rhox[::-1], abtot[::-1]
        taunu = integ(r, ab, ab[-1]*r[-1])
        snubar, alpha = snubar[::-1], alpha[::-1]
    else:
        taunu = integ(rhox, abtot, abtot[0]*rhox[0])
    xsbar = np.maximum(map1(taunu, snubar, XTAU_GRID), EPS)
    xalpha = np.clip(map1(taunu, alpha, XTAU_GRID), 0.0, 1.0)
    mask = XTAU_GRID < taunu[0]
    if np.any(mask):
        xsbar[mask] = max(snubar[0], EPS); xalpha[mask] = np.clip(alpha[0], 0.0, 1.0)
    xsbar_mod = xsbar*(1.0 - xalpha)
    diag = 1.0 - xalpha*COEFJ_DIAG
    xs = josh_iterate(xsbar.copy(), xalpha, xsbar_mod, diag)
    return float(np.dot(CH_WEIGHTS, xs))

if __name__ == "__main__":
    d = np.load(ROOT/"reference/diag.npz")
    cabs, cscat = d["continuum_absorption"], d["continuum_scattering"]
    lop, lscat = d["line_opacity"], d["line_scattering"]
    slinec, lsrc = d["slinec"], d["line_source"]
    ftot, fcont = d["flux_total"], d["flux_continuum"]
    # column mass RHOX (g/cm^2) from the reference atmosphere
    rhox = np.array([float(line.split()[0]) for line in
                     (ROOT/"reference/atmosphere.atm").read_text().splitlines()
                     if line[:1].isdigit() or line[:2].strip().replace('.','').isdigit()][:cabs.shape[0]])
    if rhox.size != cabs.shape[0]:
        a = np.load(ROOT/"reference/atmosphere.npz"); rhox = a["depth"][:cabs.shape[0]]
    nw = cabs.shape[1]
    zero = np.zeros(cabs.shape[0])
    me_t = np.empty(nw); me_c = np.empty(nw); pk_t = np.empty(nw); pk_c = np.empty(nw)
    for k in range(nw):
        me_t[k] = solve_josh(cabs[:,k], slinec[:,k], lop[:,k], lsrc[:,k], cscat[:,k], lscat[:,k], rhox)
        me_c[k] = solve_josh(cabs[:,k], slinec[:,k], zero, slinec[:,k], cscat[:,k], zero, rhox)
        pk_t[k] = pk_solve(cabs[:,k], slinec[:,k], lop[:,k], lsrc[:,k], cscat[:,k], lscat[:,k], rhox)
        pk_c[k] = pk_solve(cabs[:,k], slinec[:,k], zero, slinec[:,k], cscat[:,k], zero, rhox)
    def rel(a, b): m = np.abs(b) > 0; return np.abs(a[m]-b[m])/np.abs(b[m])
    print(f"(A) my solve vs pk solve   total: max={rel(me_t,pk_t).max():.2e} median={np.median(rel(me_t,pk_t)):.2e}")
    print(f"    my solve vs pk solve   cont : max={rel(me_c,pk_c).max():.2e} median={np.median(rel(me_c,pk_c)):.2e}")
    sp_me = me_t/me_c; sp_ref = ftot/fcont
    r = rel(sp_me, sp_ref)
    print(f"(B) my normalised spectrum vs reference: max={r.max():.2e} median={np.median(r):.2e}")
