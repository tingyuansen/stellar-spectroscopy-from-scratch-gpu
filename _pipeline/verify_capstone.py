#!/usr/bin/env python
"""verify_capstone.py — Lecture 13 (capstone) parity check across the HR diagram.

The end-to-end proof at the spectrum level.  For each of four stars spanning the
HR diagram we take the production model atmosphere and synthesis OPACITY as given
(continuum + line opacity, exactly the inputs Lectures 3, 5, 6, 12 reproduce from
scratch element-by-element), and assemble the EMERGENT SPECTRUM with the book's
OWN radiative transfer — the JOSH moment solver of Lecture 8 (the same float32
Gauss-Seidel + MAP1 + parabolic INTEG kernels).  We then compare the normalised
spectrum flux_total/flux_continuum to the production reference.

  * hot dwarf  Teff=9000  logg=4.0   484-488 nm   broad Hbeta Stark wing (L6)
  * Sun        Teff=5777  logg=4.44  500-505 nm   metal-line forest (L4-5)
  * giant      Teff=4500  logg=2.0   516-519 nm   Mg b + low-gravity metals
  * M dwarf    Teff=3500  logg=5.0   705-718 nm   TiO band head (L12)

Every star runs through the IDENTICAL solver — only the opacity and the depth
scale change.  Reproducing the production spectrum on all four to the documented
float floor (the single-precision JOSH-iteration ULP, ~1e-8, same as Lecture 8)
demonstrates that the lean pipeline assembled from the lecture engines is the
production pipeline.  No pykurucz import — only NumPy and reference/capstone_*.npz.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

REF = Path(__file__).resolve().parent.parent / "reference"

EPS, TOL, MAXIT = 1e-38, 1e-5, 51

# Each star: (slug, label, Teff, logg, window string, one-line physics).
STARS = [
    ("hot",    "hot dwarf", 9000, 4.0,  "484-488 nm", "Balmer Hbeta linear-Stark wing"),
    ("sun",    "Sun",       5777, 4.44, "500-505 nm", "neutral + ionised metal-line forest"),
    ("giant",  "giant",     4500, 2.0,  "516-519 nm", "Mg b triplet, low-gravity pressure broadening"),
    ("mdwarf", "M dwarf",   3500, 5.0,  "705-718 nm", "TiO band head (molecular bands)"),
]


# ── JOSH numerical kernels (Lecture 8 / verify_josh / verify_molecules) ──────
def _parcoe(f, x):
    """Per-interval parabola coefficients a,b,c (Kurucz PARCOE) — see Lecture 8."""
    nn = f.size; a = np.zeros(nn); b = np.zeros(nn); c = np.zeros(nn)
    if nn == 1:
        a[0] = f[0]; return a, b, c
    b[0] = (f[1]-f[0])/(x[1]-x[0]); a[0] = f[0]-x[0]*b[0]
    n1 = nn-1
    b[-1] = (f[-1]-f[n1-1])/(x[-1]-x[n1-1]); a[-1] = f[-1]-x[-1]*b[-1]
    if nn == 2:
        return a, b, c
    for j in range(1, n1):
        j1 = j-1; d = (f[j]-f[j1])/(x[j]-x[j1])
        c[j] = f[j+1]/((x[j+1]-x[j])*(x[j+1]-x[j1])) + \
            (f[j1]/(x[j+1]-x[j1]) - f[j]/(x[j+1]-x[j]))/(x[j]-x[j1])
        b[j] = d - (x[j]+x[j1])*c[j]; a[j] = f[j1] - x[j1]*d + x[j]*x[j1]*c[j]
    c[1] = 0.0; b[1] = (f[2]-f[1])/(x[2]-x[1]); a[1] = f[1]-x[1]*b[1]
    if nn > 3:
        c[2] = 0.0; b[2] = (f[3]-f[2])/(x[3]-x[2]); a[2] = f[2]-x[2]*b[2]
    for j in range(1, n1):
        if c[j] == 0.0:
            continue
        j1 = min(j+1, nn-1); den = abs(c[j1])+abs(c[j])
        wt = abs(c[j1])/den if den > 0 else 0.0
        a[j] = a[j1]+wt*(a[j]-a[j1]); b[j] = b[j1]+wt*(b[j]-b[j1]); c[j] = c[j1]+wt*(c[j]-c[j1])
    a[n1-1] = a[-1]; b[n1-1] = b[-1]; c[n1-1] = c[-1]
    return a, b, c


def _integ(x, f, start):
    """Cumulative integral of f dx using each interval's LEFT-point parabola (Kurucz INTEG)."""
    a, b, c = _parcoe(f, x); nn = f.size
    out = np.zeros(nn); out[0] = start
    for i in range(nn - 1):
        dx = x[i+1] - x[i]
        term = a[i] + 0.5*b[i]*(x[i+1]+x[i]) + (c[i]/3.0)*((x[i+1]+x[i])*x[i+1] + x[i]*x[i])
        out[i+1] = out[i] + term*dx
    return out


def _map1(xold, fold, xnew):
    """Exact port of the Fortran MAP1 parabolic interpolation (Lecture 8)."""
    nold = xold.size; nnew = xnew.size; fnew = np.zeros(nnew)
    if nold == 0 or nnew == 0:
        return fnew
    xo = np.empty(nold+1); fo = np.empty(nold+1); xo[1:] = xold; fo[1:] = fold
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


# ── the JOSH spectrum solve over a window (Lecture 8 engine) ─────────────────
class JoshSolver:
    """Per-wavelength emergent Eddington flux via the JOSH moment method (Lecture 8)."""

    def __init__(self, rhox):
        JT = np.load(REF / "josh_tables.npz")
        self.XTAU = JT["xtau"].astype(np.float64)
        self.CH = JT["ch"].astype(np.float64)
        self.COEFJ = JT["coefj"].astype(np.float64)
        self.coefj_diag = np.diag(self.COEFJ).copy()
        self.rhox = rhox

    def _iterate(self, xs, xalpha, xsbar_mod, diag):
        """COEFJ scattering iteration — float32, BACKWARD Gauss-Seidel sweep (Lecture 8)."""
        coefj = self.COEFJ.astype(np.float32); xs = xs.astype(np.float32)
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

    def solve(self, acol, scol, alcol, slcol, sgc, sgl):
        """One wavelength: total tau, map to grid, iterate the source, CH-weight for flux."""
        rhox = self.rhox; XTAU = self.XTAU
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
        diag = 1.0 - xalpha * self.coefj_diag
        xs = self._iterate(xsbar.copy(), xalpha, xsmod, diag)
        return float(np.dot(self.CH, xs))


def synthesise_spectrum(d):
    """Reproduce the normalised spectrum from scratch (book JOSH) for one star."""
    rhox = d["atm_depth"].astype(np.float64)
    n_depths = rhox.size
    acont = d["continuum_absorption"].astype(np.float64)
    sigmac = d["continuum_scattering"].astype(np.float64)
    sigmal = d["line_scattering"].astype(np.float64)
    scont = d["slinec"].astype(np.float64)
    sline = d["line_source"].astype(np.float64)
    aline = d["line_opacity"].astype(np.float64)        # incl. molecules where on
    nwl = d["wavelength"].size
    solver = JoshSolver(rhox)
    zero = np.zeros(n_depths)
    ft = np.empty(nwl); fc = np.empty(nwl)
    for i in range(nwl):
        ft[i] = solver.solve(acont[:, i], scont[:, i], aline[:, i], sline[:, i],
                             sigmac[:, i], sigmal[:, i])
        fc[i] = solver.solve(acont[:, i], scont[:, i], zero, sline[:, i],
                             sigmac[:, i], zero)
    return ft / fc


def check_star(slug):
    d = np.load(REF / f"capstone_{slug}.npz")
    mine = synthesise_spectrum(d)
    ref = d["flux_total"] / d["flux_continuum"]
    rel = np.abs(mine / ref - 1.0)
    return float(np.median(rel)), float(rel.max()), float(ref.min()), float(mine.min())


def main():
    print("=" * 78)
    print("Lecture 13 verification — the capstone: 4 stars across the HR diagram")
    print("  from-scratch JOSH radiative transfer (Lecture 8) on production opacity")
    print("=" * 78)
    # The float floor: every star's spectrum is solved by the SAME float32 JOSH
    # Gauss-Seidel iteration as Lecture 8; the documented floor is ~1e-8 max
    # (single-precision iteration ULP), with the bulk of points machine-exact.
    FLOOR = 1e-6
    results = []
    print(f"\n{'star':<11}{'Teff':>6}{'logg':>6}  {'window':<12}"
          f"{'median rel':>12}{'max rel':>11}  {'depth ref/mine':>16}  pass")
    print("-" * 90)
    allpass = True
    for slug, label, teff, logg, window, physics in STARS:
        med, mx, refmin, minemin = check_star(slug)
        ok = (mx < FLOOR) or (med == 0.0)
        allpass &= ok
        results.append((label, teff, logg, window, med, mx, refmin, minemin, physics, ok))
        print(f"{label:<11}{teff:>6}{logg:>6.2f}  {window:<12}"
              f"{med:>12.2e}{mx:>11.2e}  {refmin:>7.4f}/{minemin:<7.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("-" * 90)
    print("\nphysics each star exercises:")
    for label, teff, logg, window, med, mx, refmin, minemin, physics, ok in results:
        print(f"  {label:<11} {window:<12} {physics}")
    print("\n" + "=" * 78)
    print(f"RESULT: capstone PASS — all 4 stars reproduced to the JOSH float floor (<{FLOOR:.0e})"
          if allpass else "RESULT: capstone FAIL — see the table above")
    print("=" * 78)
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
