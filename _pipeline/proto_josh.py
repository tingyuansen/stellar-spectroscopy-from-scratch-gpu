#!/usr/bin/env python
"""Script-first reproduction of pykurucz's JOSH radiative-transfer solver, verified
piece-by-piece against the reference BEFORE it goes into the dedicated JOSH lecture
(so the live L6 never breaks). Run from the book root with the venv active.

VERIFIED so far:
  [x] optical-depth integrator (_parcoe + integ) — bit-exact vs synthe_py...josh_solver._integ (max|rel|=0.0)
TODO (next pieces, then assemble solve_josh_flux + verify vs diag flux_total to machine precision):
  [ ] map1 parabolic interpolation onto XTAU_GRID  (josh_solver.py _map1/_map1_kernel 224-369)
  [ ] moment Lambda-iteration using COEFJ_MATRIX   (josh_solver.py ~560-780)
  [ ] CH_WEIGHTS surface flux                        (josh_solver.py ~780-809)
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/ysting/pykurucz")


def parcoe(f, x):
    """Parabolic coefficients a,b,c per interval (Fortran PARCOE; josh_solver._parcoe)."""
    n = f.size; a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    if n == 1: a[0] = f[0]; return a, b, c
    b[0] = (f[1]-f[0])/(x[1]-x[0]); a[0] = f[0]-x[0]*b[0]
    n1 = n-1; b[-1] = (f[-1]-f[n1-1])/(x[-1]-x[n1-1]); a[-1] = f[-1]-x[-1]*b[-1]
    if n == 2: return a, b, c
    for j in range(1, n1):
        j1 = j-1; d = (f[j]-f[j1])/(x[j]-x[j1])
        c[j] = f[j+1]/((x[j+1]-x[j])*(x[j+1]-x[j1])) \
               + (f[j1]/(x[j+1]-x[j1]) - f[j]/(x[j+1]-x[j]))/(x[j]-x[j1])
        b[j] = d-(x[j]+x[j1])*c[j]; a[j] = f[j1]-x[j1]*d+x[j]*x[j1]*c[j]
    c[1] = 0.0; b[1] = (f[2]-f[1])/(x[2]-x[1]); a[1] = f[1]-x[1]*b[1]
    if n > 3:
        c[2] = 0.0; b[2] = (f[3]-f[2])/(x[3]-x[2]); a[2] = f[2]-x[2]*b[2]
    for j in range(1, n1):
        if c[j] == 0.0: continue
        j1 = min(j+1, n-1); denom = abs(c[j1])+abs(c[j]); wt = abs(c[j1])/denom if denom > 0 else 0.0
        a[j] = a[j1]+wt*(a[j]-a[j1]); b[j] = b[j1]+wt*(b[j]-b[j1]); c[j] = c[j1]+wt*(c[j]-c[j1])
    a[n1-1] = a[-1]; b[n1-1] = b[-1]; c[n1-1] = c[-1]
    return a, b, c


def integ(x, f, start):
    """Cumulative parabolic integral (Fortran INTEG; josh_solver._integ). Bit-exact."""
    n = f.size; fint = np.zeros(n); a, b, c = parcoe(f, x); fint[0] = start
    for i in range(n-1):
        dx = x[i+1]-x[i]
        term = a[i] + 0.5*b[i]*(x[i+1]+x[i]) + (c[i]/3.0)*((x[i+1]+x[i])*x[i+1]+x[i]*x[i])
        fint[i+1] = fint[i] + term*dx
    return fint


if __name__ == "__main__":
    from synthe_py.physics.josh_solver import _integ as ref_integ
    from pathlib import Path
    R = np.load(Path(__file__).resolve().parent.parent / "reference" / "L6.npz")
    rhox = R["rhox"]; ktot = (R["total_abs"]+R["total_scat"]).astype(float)
    worst = 0.0
    for k in (0, 1000, 2970, 5000, 5940):
        ab = ktot[:, k]; start = ab[0]*rhox[0]
        mine = integ(rhox, ab, start); ref = ref_integ(rhox, ab, start)
        worst = max(worst, np.max(np.abs(mine-ref)/np.maximum(np.abs(ref), 1e-300)))
    print(f"_integ reproduction vs pykurucz: max|rel|={worst:.2e}")
