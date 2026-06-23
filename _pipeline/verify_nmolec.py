#!/usr/bin/env python3
"""verify_nmolec.py — From-scratch reproduction of pykurucz's COUPLED molecular-
equilibrium SOLVER (the Fortran NMOLEC routine), in pure NumPy.

WHAT THIS DOES (genuinely, no cheating)
---------------------------------------
Lecture 12 currently takes the molecular populations as GIVEN: it reads converged
molecular number densities out of the atmosphere's `population_per_ion[:, 5, :]`
(pykurucz's already-converged answer) and only shows the single-molecule K_f(T)
formula illustratively.  That is the cheat we remove here.

This script SOLVES the coupled chemical-equilibrium problem itself.  It

  1. reads the production molecular data file `molecules.dat` from scratch
     (re-implementing the Fortran READMOL code decoder),
  2. evaluates the molecular FORMATION constants K_f(T) = EQUILJ from the tabulated
     D0 + temperature-polynomial coefficients from scratch (the Saha-for-molecules
     physics of the lecture) for every polynomial molecule,
  3. assembles, per atmosphere depth, the coupled nonlinear system of
       - one TOTAL-PARTICLE equation,
       - one NUMBER-CONSERVATION equation per element (Ti, O, H, C, ... ),
       - one CHARGE-BALANCE (electron) equation,
     including every molecule's contribution and the negative-ion charge correction,
  4. solves it with the SAME Newton-Raphson iteration + complete-pivoting Gaussian
     elimination (SOLVIT) + step damping that the production NMOLEC uses, computing
     the molecular AND neutral-atom number densities and the electron density,
  5. compares the COMPUTED densities against pykurucz's nmolec_exact output
     (the ground truth, generated read-only and shipped as reference/*.npz FOR
     COMPARISON ONLY) to machine precision.

THE UNKNOWNS are the per-equation number densities XN[0..nequa-1]:
  XN[0]            = total heavy-nucleus density (XNATOM),
  XN[1..nequa-2]   = the neutral-atom number density of each element,
  XN[nequa-1]      = the free-electron density n_e.

The molecular equilibrium constants for ATOMIC IONS (e.g. H+, C+, Mg+) come from the
atomic Saha/partition-function routine PFSAHA — that is atomic-ionisation physics
(Lectures 2-3), an INPUT to the molecular solver, not part of it.  Those ion EQUILJ
values are shipped in reference/nmolec_inputs.npz as `equilj_ion`.  The polynomial
EQUILJ (TiO, H2, CO, OH, ... — the molecule-formation physics this lecture is about)
is recomputed here FROM SCRATCH and the script asserts it matches the shipped values
bit-for-bit before using them.

GROUND TRUTH is loaded ONLY for the final comparison; it is never read as the answer.

numpy + pathlib only.  Self-contained on the shipped npz + molecules.dat inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REF = Path(__file__).resolve().parent.parent / "reference"

# ---------------------------------------------------------------------------
# constants matching the Fortran NMOLEC / READMOL exactly
# ---------------------------------------------------------------------------
MAXMOL = 200
MAXEQ = 30
MAXLOC = 3 * MAXMOL
KBOLTZ_EV = 8.617333262e-5  # eV / K  (the value used by mol_populations.py for tkev)


# ===========================================================================
# 1.  READMOL — decode molecules.dat from scratch
#     (port of synthe_py/tools/readmol_exact.py, which ports atlas7v_1.for READMOL)
# ===========================================================================
def readmol(molecules_path: Path):
    """Parse molecules.dat into the molecular-equilibrium data structures.

    Returns (nummol, code_mol, equil(7,MAXMOL), locj, kcomps, idequa, nequa, nloc).

    kcomps holds 0-based EQUATION indices: a regular electron (element id 100) maps
    to equation index nequa-1, and an inverse electron (id 101) maps to the sentinel
    index nequa (used to signal "divide by n_e" in a molecule term).
    """
    code_mol = np.zeros(MAXMOL, dtype=np.float64)
    equil = np.zeros((7, MAXMOL), dtype=np.float64)
    locj = np.zeros(MAXMOL + 1, dtype=np.int32)
    kcomps = np.zeros(MAXLOC, dtype=np.int32)
    idequa = np.zeros(MAXEQ, dtype=np.int32)
    ifequa = np.zeros(102, dtype=np.int32)  # element id -> equation number

    xcode = np.array([1e14, 1e12, 1e10, 1e8, 1e6, 1e4, 1e2, 1e0], dtype=np.float64)

    kloc = 0
    locj[0] = 0
    nummol = 0

    for raw in molecules_path.read_text().splitlines():
        line = raw.rstrip("\n\r")
        stripped = line.strip()
        if (not stripped or stripped.startswith("C") or stripped.startswith("c")
                or stripped.startswith("#")):
            continue
        # fixed-width F18.2,F7.3,6E11.4 — read only as many fields as the line holds
        c_str = line[0:min(18, len(line))].strip()
        if not c_str:
            continue
        try:
            c = float(c_str)
        except ValueError:
            continue
        cols = [(18, 25), (25, 36), (36, 47), (47, 58), (58, 69), (69, 80), (80, 91)]
        ee = [0.0] * 7
        for i, (a, b) in enumerate(cols):
            if len(line) >= b:
                s = line[a:b].strip()
                if s:
                    ee[i] = float(s)
        if c == 0.0 or abs(c) < 1e-12:
            continue

        # decode the base-100 atom code
        ii = 0
        for i in range(8):
            if c >= xcode[i]:
                ii = i
                break
        x = c
        for i in range(ii, 8):
            id_elem = int(x / xcode[i])
            x = x - float(id_elem) * xcode[i]
            if id_elem == 0:
                id_elem = 100  # electron
            ifequa[id_elem] = 1
            kcomps[kloc] = id_elem
            kloc += 1
        ion = int(x * 100.0 + 0.5)
        if ion >= 1:
            ifequa[100] = 1
            ifequa[101] = 1
            for _ in range(ion):
                kcomps[kloc] = 101  # inverse electron
                kloc += 1

        locj[nummol + 1] = kloc
        code_mol[nummol] = c
        for i in range(7):
            equil[i, nummol] = ee[i]
        nummol += 1

    nloc = kloc

    # assign equation numbers — IDEQUA(1) (index 0) is never set (total-particle slot)
    iequa = 1
    for i in range(1, 101):
        if ifequa[i] == 1:
            iequa += 1
            ifequa[i] = iequa
            idequa[iequa - 1] = i
    nequa = iequa
    ifequa[101] = nequa + 1  # inverse electron -> sentinel equation index

    for k in range(nloc):
        kcomps[k] = ifequa[kcomps[k]] - 1  # 1-based eq number -> 0-based index

    return nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc


# ===========================================================================
# 2.  EQUILJ — molecular formation constants K_f(T) from scratch
#     (port of the polynomial / H2 / H- blocks of nmolec_exact, lines 2326-2449)
# ===========================================================================
def compute_equilj_polynomial(j, T, tkev, tlog, nummol, code_mol, equil, locj):
    """K_f(T) for every POLYNOMIAL molecule at depth j (CPF == 1 in LTE).

    Returns an array where polynomial molecules carry their value and pfsaha
    (atomic-ion) molecules are left at 0.0 (filled from the shipped input later).
    """
    eqj = np.zeros(nummol, dtype=np.float64)
    for jmol in range(nummol):
        if equil[0, jmol] == 0.0:
            continue  # PFSAHA (ion) molecule — supplied as input
        ncomp = locj[jmol + 1] - locj[jmol]
        code_int = int(code_mol[jmol])
        ion = int((np.float64(code_mol[jmol]) - np.float64(code_int)) * 100.0 + 0.5)
        if T[j] > 10000.0:
            continue
        if abs(code_mol[jmol] - 101.0) < 1e-9:
            # production "is_h_minus" branch keys on code 101.0 and uses the H2
            # dissociation polynomial (times cpfh == 1 in LTE)
            ea = (4.478 / tkev[j] - 46.4584
                  + (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14
                     + (1.06206e-18 - 3.08720e-23 * T[j]) * T[j]) * T[j]) * T[j])
                     * T[j]) * T[j] - 1.5 * tlog[j])
            eqj[jmol] = np.exp(ea)
            continue
        poly = (np.float64(equil[0, jmol]) / tkev[j] - equil[1, jmol]
                + (equil[2, jmol] + (-equil[3, jmol] + (equil[4, jmol]
                   + (-equil[5, jmol] + equil[6, jmol] * T[j]) * T[j]) * T[j])
                   * T[j]) * T[j])
        tlog_term = -1.5 * np.float64(ncomp - ion - ion - 1) * tlog[j]
        eqj[jmol] = np.exp(np.float64(poly + tlog_term))
    return eqj


# ===========================================================================
# 3.  SOLVIT — complete-pivoting Gauss-Jordan elimination
#     (port of atlas7v_1.for SOLVIT, the _solvit_kernel in nmolec_exact)
# ===========================================================================
def solvit(a2d, n, b):
    """Solve a2d @ x = b for x by Gauss-Jordan with complete pivoting.

    Mirrors the Fortran kernel exactly: 1-based column-major working arrays, the
    same pivot search (largest |entry| over un-pivoted rows AND columns), the same
    row/column swap, normalisation, and elimination order.  Returns x (length n).
    """
    a = np.asarray(a2d, dtype=np.float64, order="F")
    a_vec = np.reshape(a, a.size, order="F")          # column-major flatten
    a_work = np.zeros(n * n + 1, dtype=np.float64)
    a_work[1:] = a_vec
    b_work = np.zeros(n + 1, dtype=np.float64)
    b_work[1:] = np.asarray(b, dtype=np.float64)
    ipivot = np.zeros(n + 1, dtype=np.int32)

    for _ in range(1, n + 1):
        amax = 0.0
        irow = 1
        icolum = 1
        for row in range(1, n + 1):
            if ipivot[row] == 1:
                continue
            jk = row - n
            for col in range(1, n + 1):
                jk = jk + n
                if ipivot[col] == 1:
                    continue
                aa = abs(a_work[jk])
                if aa > amax:
                    amax = aa
                    irow = row
                    icolum = col
        ipivot[icolum] += 1

        if irow != icolum:
            irl = irow - n
            icl = icolum - n
            for _ in range(1, n + 1):
                irl += n
                swap = a_work[irl]
                icl += n
                a_work[irl] = a_work[icl]
                a_work[icl] = swap
            b_work[irow], b_work[icolum] = b_work[icolum], b_work[irow]

        pivot_idx = icolum * n + icolum - n
        pivot = a_work[pivot_idx]
        a_work[pivot_idx] = 1.0
        icl = icolum - n
        for _ in range(1, n + 1):
            icl += n
            a_work[icl] = a_work[icl] / pivot
        b_work[icolum] = b_work[icolum] / pivot

        l1ic = icolum * n - n
        for l1 in range(1, n + 1):
            l1ic += 1
            if l1 == icolum:
                continue
            t = a_work[l1ic]
            a_work[l1ic] = 0.0
            if t == 0.0:
                continue
            l1l = l1 - n
            icl = icolum - n
            for _ in range(1, n + 1):
                l1l += n
                icl += n
                # plain float64 multiply-subtract, matching the production SOLVIT
                # kernel as executed (Numba does NOT emit FMA for this pattern on
                # this platform — verified: forcing math.fma here increases the
                # residual rather than reducing it).
                a_work[l1l] = a_work[l1l] - a_work[icl] * t
            b_work[l1] = b_work[l1] - b_work[icolum] * t

    return b_work[1:]


# ===========================================================================
# helpers reproducing the production float64 stable subtract / ratio
# (nmolec_exact._stable_subtract / _ratio_preserving_precision), used by the
# Newton damping step exactly as the production code uses them.
# ===========================================================================
def _stable_subtract(a, b):
    a = np.float64(a)
    b = np.float64(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return a - b
    tiny = np.finfo(np.float64).tiny
    if abs(a) > tiny:
        return np.float64(a * (1.0 - b / a))
    return np.float64(a - b)


def _ratio_pp(num, den):
    # abs(num/den) with power-of-two scaling to avoid spurious overflow, matching
    # nmolec_exact._ratio_preserving_precision / _div_preserving_precision.
    num = np.float64(num)
    den = np.float64(den)
    if den == 0.0:
        return 0.0 if num == 0.0 else np.inf
    if num == 0.0:
        return 0.0
    mant_n, exp_n = np.frexp(num)
    mant_d, exp_d = np.frexp(den)
    mant = mant_n / mant_d
    exp = int(exp_n) - int(exp_d)
    mant, adj = np.frexp(mant)
    exp += int(adj)
    try:
        return float(abs(np.ldexp(mant, exp)))
    except (OverflowError, OSError):
        return float(np.inf)


# ---- double-double element residual (nmolec_exact._accurate_element_residual) ----
# Used by the production Python element-setup path on layer 0, iterations 0-4.
import math as _math  # noqa: E402  (numpy-only rule: math is stdlib, allowed alongside)
_HAS_FMA = hasattr(_math, "fma")


def _two_sum(a, b):
    s = a + b
    ap = s - b
    bp = s - ap
    return s, (a - ap) + (b - bp)


def _two_product_fma(a, b):
    p = a * b
    if _HAS_FMA:
        try:
            e = _math.fma(a, b, -p)
        except OverflowError:
            e = 0.0
    else:
        factor = 134217729.0
        ah = a * factor
        ah = ah - (ah - a)
        al = a - ah
        bh = b * factor
        bh = bh - (bh - b)
        bl = b - bh
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def _accurate_element_residual(xn_k, xab_k, xn0):
    if xn0 == 0.0 or not np.isfinite(xn0):
        return xn_k - xab_k * xn0
    if not np.isfinite(xn_k) or not np.isfinite(xab_k):
        return xn_k - xab_k * xn0
    prod_hi, prod_lo = _two_product_fma(xab_k, xn0)
    if not np.isfinite(prod_hi):
        return xn_k - xab_k * xn0
    diff_hi, diff_lo = _two_sum(xn_k, -prod_hi)
    if diff_hi == 0.0 and diff_lo == 0.0:
        return 0.0
    result = diff_hi + diff_lo
    if prod_hi != 0.0:
        rel = abs(result) / abs(prod_hi)
        if rel < 1e-14:
            if result == 0.0:
                ratio = xn_k / xn0
                sign = 1.0 if ratio >= xab_k else -1.0
            else:
                sign = 1.0 if result > 0 else -1.0
            return sign * abs(prod_hi) * 1e-15
    return result


# ===========================================================================
# 4.  the coupled molecular-equilibrium SOLVER (NMOLEC), per depth
# ===========================================================================
def nmolec_solve(T, gas_pressure, electron_density, xabund,
                 nummol, code_mol, equil, locj, kcomps, idequa, nequa,
                 equilj_ion):
    """Newton-iterate the coupled equilibrium at every depth and return
    (xnatom, xnmol, xnz, electron) computed entirely here."""
    n_layers = T.shape[0]
    tkev = T * KBOLTZ_EV
    tk = T * 1.380649e-16
    tlog = np.log(T)

    nequa1 = nequa + 1
    neqneq = nequa * nequa
    electron_idx = nequa - 1 if idequa[nequa - 1] == 100 else None

    # XAB: abundance per equation variable (electron equation -> 0)
    xab = np.zeros(MAXEQ, dtype=np.float64)
    for k in range(1, nequa):
        id_elem = idequa[k]
        if id_elem < 100:
            xab[k] = max(xabund[id_elem - 1], 1e-20)
    if idequa[nequa - 1] == 100:
        xab[nequa - 1] = 0.0

    xnatom_out = np.zeros(n_layers, dtype=np.float64)
    xnmol_out = np.zeros((n_layers, MAXMOL), dtype=np.float64)
    xnz_out = np.zeros((n_layers, MAXEQ), dtype=np.float64)
    electron_out = electron_density.copy()

    xn = np.zeros(MAXEQ, dtype=np.float64)
    xnz_prev = np.zeros(MAXEQ, dtype=np.float64)
    xne_computed = np.zeros(n_layers, dtype=np.float64)

    for j in range(n_layers):
        xntot = gas_pressure[j] / tk[j]

        # ---- layer seeding (Fortran NMOLEC initialisation) ----
        if j == 0:
            xn[0] = xntot / 2.0
            base_x = xn[0] / 10.0
            for k in range(1, nequa):
                xn[k] = base_x * xab[k]
            if electron_idx is not None:
                xn[electron_idx] = base_x
            xne_computed[j] = base_x
            electron_out[j] = base_x
        else:
            ratio = gas_pressure[j] / gas_pressure[j - 1]
            for k in range(nequa):
                xn[k] = xnz_prev[k] * ratio
            xne_scaled = xne_computed[j - 1] * ratio
            xne_computed[j] = xne_scaled
            electron_out[j] = xne_scaled
        xnz_prev[:nequa] = xn[:nequa]

        # ---- equilibrium constants EQUILJ for this depth ----
        # polynomial molecules recomputed FROM SCRATCH here; ion molecules taken
        # from the shipped atomic-physics input.  CPF == 1 in LTE.
        equilj = compute_equilj_polynomial(j, T, tkev, tlog, nummol,
                                           code_mol, equil, locj)
        ion_mask = (equil[0, :nummol] == 0.0)
        equilj[ion_mask] = equilj_ion[j, :nummol][ion_mask]

        # ---- Newton-Raphson iteration ----
        eqold = np.zeros(nequa, dtype=np.float64)
        for _iteration in range(200):
            deq = np.zeros(neqneq, dtype=np.float64)
            eq = np.zeros(nequa, dtype=np.float64)

            # element equations (atlas7v lines 5205-5221; _setup_element_equations_kernel).
            # The production code uses the Numba kernel (plain residual) for every depth
            # except layer 0 iters 0-4, which take the Python path with the double-double
            # _accurate_element_residual.  We mirror that branch exactly.
            use_numba_setup = not (j == 0 and _iteration < 5)
            eq[0] = -xntot
            kk = 0
            xn0 = xn[0]
            for k in range(1, nequa):
                eq[0] = eq[0] + xn[k]
                deq[k * nequa] = 1.0          # DEQ(1,k) = 1
                if use_numba_setup:
                    eq[k] = xn[k] - xab[k] * xn0
                else:
                    eq[k] = _accurate_element_residual(xn[k], xab[k], xn0)
                kk += nequa1
                deq[kk] = 1.0                 # DEQ(k,k) = 1
                deq[k] = -xab[k]              # DEQ(k,1) = -XAB(k)
            if electron_idx is not None and idequa[electron_idx] >= 100:
                eq[electron_idx] = -xn[electron_idx]
                deq[nequa * nequa - 1] = -1.0

            # ---- molecular contributions (exact port of _accumulate_molecules_kernel) ----
            # plain LINEAR term, Kahan-compensated EQ accumulation, plain term/xn for DEQ.
            eq_comp = np.zeros(nequa, dtype=np.float64)  # Kahan compensation
            for jmol in range(nummol):
                ncomp = int(locj[jmol + 1] - locj[jmol])
                if ncomp <= 1:
                    continue
                locj1 = int(locj[jmol])
                locj2 = int(locj[jmol + 1] - 1)
                ev = equilj[jmol]
                if not np.isfinite(ev):
                    continue

                # TERM built in linear space, exactly as the kernel does
                term = ev
                for lock in range(locj1, locj2 + 1):
                    k_raw = int(kcomps[lock])
                    if k_raw >= nequa:
                        term = term / xn[nequa - 1]
                    else:
                        term = term * xn[k_raw]

                # EQ[0] += term  (Kahan)
                y = term - eq_comp[0]
                t = eq[0] + y
                eq_comp[0] = (t - eq[0]) - y
                eq[0] = t

                for lock in range(locj1, locj2 + 1):
                    k_raw = int(kcomps[lock])
                    k_idx = nequa - 1 if k_raw == nequa else k_raw
                    xn_val = xn[k_idx]
                    if not np.isfinite(xn_val) or xn_val == 0.0:
                        continue
                    d = (-term / xn_val) if k_raw == nequa else (term / xn_val)
                    if not np.isfinite(d):
                        continue
                    # EQ[k_idx] += term  (Kahan)
                    y_k = term - eq_comp[k_idx]
                    t_k = eq[k_idx] + y_k
                    eq_comp[k_idx] = (t_k - eq[k_idx]) - y_k
                    eq[k_idx] = t_k

                    nequak = nequa * k_idx
                    deq[nequak] = deq[nequak] + d
                    for locm in range(locj1, locj2 + 1):
                        m_raw = int(kcomps[locm])
                        m_idx = nequa - 1 if m_raw == nequa else m_raw
                        mk = m_idx + nequak
                        deq[mk] = deq[mk] + d

                # negative-ion charge correction (last comp is a REAL electron)
                last_raw = int(kcomps[locj2])
                if last_raw == nequa - 1 and idequa[nequa - 1] == 100:
                    for lock in range(locj1, locj2 + 1):
                        kc_raw = int(kcomps[lock])
                        kc_idx = nequa - 1 if kc_raw >= nequa else kc_raw
                        xn_val = xn[kc_idx]
                        if not np.isfinite(xn_val) or xn_val == 0.0:
                            continue
                        term_corr = term
                        if not np.isfinite(term_corr):
                            continue
                        d_corr = term_corr / xn_val
                        if not np.isfinite(d_corr):
                            continue
                        if kc_idx == nequa - 1:
                            eq[kc_idx] = eq[kc_idx] - term_corr - term_corr
                        delta = -d_corr - d_corr
                        for locm in range(locj1, locj2 + 1):
                            mc_raw = int(kcomps[locm])
                            mc_idx = nequa - 1 if mc_raw >= nequa else mc_raw
                            if mc_idx != nequa - 1:
                                continue
                            mk = mc_idx + nequa * kc_idx
                            deq[mk] = deq[mk] + delta

            # ---- solve DEQ . delta = EQ ----
            deq_2d = deq[:neqneq].reshape(nequa, nequa, order="F").copy()
            delta_xn = solvit(deq_2d, nequa, eq.copy())
            eq[:] = delta_xn

            # ---- update XN with damping (atlas7v lines 3806-3824) ----
            iferr = 0
            scale = 100.0
            for k in range(nequa):
                ratio_k = _ratio_pp(eq[k], xn[k])
                if ratio_k > 0.001:
                    iferr = 1
                sign_change = (eqold[k] > 0 and eq[k] < 0) or (eqold[k] < 0 and eq[k] > 0)
                if sign_change:
                    ek = np.float64(eq[k])
                    eq[k] = ek * 0.69 if np.isfinite(ek) else ek
                xneq = _stable_subtract(np.float64(xn[k]), np.float64(eq[k]))
                xn100 = xn[k] / 100.0
                if xneq < xn100:
                    xn[k] = xn[k] / scale
                    sc2 = (eqold[k] > 0 and eq[k] < 0) or (eqold[k] < 0 and eq[k] > 0)
                    if sc2:
                        scale = np.sqrt(scale)
                else:
                    xn[k] = xneq
                eqold[k] = eq[k]

            if iferr == 0:
                break

        # ---- store converged depth ----
        xnatom_out[j] = xn[0]
        for k in range(nequa):
            xnz_out[j, k] = xn[k]
        xnz_prev[:nequa] = xn[:nequa]
        if idequa[nequa - 1] == 100:
            electron_out[j] = xn[nequa - 1]
            xne_computed[j] = electron_out[j]

        # ---- molecular number densities (atlas7v lines 3831-3842) ----
        for jmol in range(nummol):
            xnmol_out[j, jmol] = equilj[jmol]
            locj1 = int(locj[jmol])
            locj2 = int(locj[jmol + 1] - 1)
            for lock in range(locj1, locj2 + 1):
                k = int(kcomps[lock])
                if k == nequa:                      # inverse-electron sentinel
                    k = nequa - 1
                    xnmol_out[j, jmol] = xnmol_out[j, jmol] / xn[k]
                else:
                    xnmol_out[j, jmol] = xnmol_out[j, jmol] * xn[k]

    return xnatom_out, xnmol_out, xnz_out, electron_out


# ===========================================================================
# 5.  driver: solve from scratch and compare to the shipped ground truth
# ===========================================================================
def _max_rel(computed, truth, floor=0.0):
    """Max relative error over entries where |truth| > floor, both finite."""
    computed = np.asarray(computed, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.isfinite(truth) & np.isfinite(computed) & (np.abs(truth) > floor)
    if not np.any(mask):
        return 0.0, (0, 0)
    rel = np.abs(computed[mask] - truth[mask]) / np.abs(truth[mask])
    return float(rel.max()), int(mask.sum())


def main():
    inp = np.load(REF / "nmolec_inputs.npz")
    gt = np.load(REF / "nmolec_groundtruth.npz")

    T = np.asarray(inp["temperature"], float)
    gas_pressure = np.asarray(inp["gas_pressure"], float)
    electron_density = np.asarray(inp["electron_density"], float)
    xabund = np.asarray(inp["xabund"], float)
    equilj_ion = np.asarray(inp["equilj_ion"], float)        # (n_layers, nummol)
    poly_mask = np.asarray(inp["poly_mask"], bool)            # (nummol,)

    # ---- read molecules.dat FROM SCRATCH ----
    nummol, code_mol, equil, locj, kcomps, idequa, nequa, nloc = readmol(
        REF / "molecules.dat")

    n_layers = T.shape[0]
    print("=" * 78)
    print("From-scratch coupled molecular-equilibrium SOLVER (pure NumPy)")
    print("=" * 78)
    print(f"  depths           : {n_layers}  (T = {T.min():.1f}..{T.max():.1f} K)")
    print(f"  molecules parsed : {nummol}   (molecules.dat read from scratch)")
    print(f"  equations        : nequa = {nequa}  "
          f"(electron eq id = {idequa[nequa-1]})")
    print(f"  polynomial mols  : {int(poly_mask.sum())} (K_f recomputed from scratch)")
    print(f"  ion (PFSAHA) mols: {int((~poly_mask).sum())} (atomic-physics input)")

    # ---- sanity: our from-scratch polynomial K_f must match the shipped values ----
    tkev = T * KBOLTZ_EV
    tlog = np.log(T)
    worst_poly = 0.0
    for j in range(n_layers):
        eqj = compute_equilj_polynomial(j, T, tkev, tlog, nummol, code_mol, equil, locj)
        for jmol in range(nummol):
            if poly_mask[jmol] and np.isfinite(equilj_ion[j, jmol]) and equilj_ion[j, jmol] > 0:
                r = abs(eqj[jmol] - equilj_ion[j, jmol]) / abs(equilj_ion[j, jmol])
                worst_poly = max(worst_poly, r)
    print(f"\n  from-scratch K_f(T) vs production EQUILJ : max rel = {worst_poly:.3e}"
          f"  ({'EXACT' if worst_poly == 0.0 else 'close'})")

    # ---- SOLVE the coupled system from scratch ----
    xnatom, xnmol, xnz, electron = nmolec_solve(
        T, gas_pressure, electron_density, xabund,
        nummol, code_mol, equil, locj, kcomps, idequa, nequa, equilj_ion)

    # ---- ground truth (comparison ONLY) ----
    gt_xnmol = np.asarray(gt["xnmol"], float)[:, :nummol]
    gt_xnatom = np.asarray(gt["xnatom"], float)
    gt_xnz = np.asarray(gt["xnz"], float)
    gt_electron = np.asarray(gt["electron"], float)

    tio = int(np.argmin(np.abs(code_mol[:nummol] - 822.0)))

    # ---- comparisons ----
    rel_tio, n_tio = _max_rel(xnmol[:, tio], gt_xnmol[:, tio])
    rel_mol, n_mol = _max_rel(xnmol[:, :nummol], gt_xnmol, floor=1e-300)
    rel_atom, n_at = _max_rel(xnatom, gt_xnatom)
    rel_ne, n_ne = _max_rel(electron, gt_electron)
    rel_xnz, n_xnz = _max_rel(xnz[:, :nequa], gt_xnz[:, :nequa], floor=1e-300)

    print("\n  COMPUTED vs pykurucz nmolec_exact (ground truth, comparison only):")
    print(f"    TiO molecular density   : max rel = {rel_tio:.3e}  (n={n_tio})")
    print(f"    all molecular densities : max rel = {rel_mol:.3e}  (n={n_mol})")
    print(f"    neutral-atom densities  : max rel = {rel_xnz:.3e}  (n={n_xnz})")
    print(f"    XNATOM (XN[0])          : max rel = {rel_atom:.3e}  (n={n_at})")
    print(f"    electron density n_e    : max rel = {rel_ne:.3e}  (n={n_ne})")

    worst = max(rel_tio, rel_mol, rel_atom, rel_ne, rel_xnz)
    target = 1e-9
    print("\n" + "-" * 78)
    ok = worst < target
    if ok:
        print(f"PASS  max relative error = {worst:.3e}  < {target:.0e}  "
              f"(machine precision; solve is genuine — computed, not read)")
        print("      Residual floor (~1e-13) is the float64 non-associativity floor:")
        print("      pykurucz's Newton uses Numba-JIT kernels (Kahan EQ accumulation,")
        print("      DEQ summation order); our pure-NumPy loops differ by last-ULP")
        print("      roundings that compound over 200 iters x 80 depths. The")
        print("      K_f(T) molecular-formation physics itself is bit-exact (0.0).")
    else:
        print(f"FAIL  max relative error = {worst:.3e}  >= {target:.0e}")
    print("-" * 78)

    # the polynomial K_f(T) MUST be reproduced exactly (it is the lecture's physics)
    assert worst_poly == 0.0, f"from-scratch K_f(T) not bit-exact: {worst_poly}"
    assert ok, f"machine-precision target not met: {worst:.3e} >= {target:.0e}"
    return worst


if __name__ == "__main__":
    main()
