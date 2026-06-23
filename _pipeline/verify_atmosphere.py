#!/usr/bin/env python
"""From-scratch grey model-atmosphere builder, verified bit-exact against reference/L1.npz.

Reproduces the ATLAS12 grey cold start that Lectures 1-9 took as GIVEN:
  (Teff, logg)  ->  grey T(tau)  ->  hydrostatic P_gas(tau), column mass RHOX(tau)

The reference is computed by pykurucz's `generate_grey_atm(teff=5770, logg=4.44)` (pure NumPy,
no torch/emulator on this path -> deterministic). That routine:
  - builds the Hopf-fit grey temperature T(tau),
  - sets kappa_Ross == 1 on the cold start (empty ROSSTAB table, NROSS=0),
  - integrates hydrostatic equilibrium dP/dtau = g/kappa in LOG pressure with a
    predictor-corrector multistep (subroutine TTAUP),
  - returns P_gas = P_total - P_rad and RHOX = P_total / g.

This script is a clean reimplementation (no pykurucz import) verified to machine precision
(bit-exact, rel error 0) against reference/L1.npz: grey_tau, grey_T, grey_pgas, grey_rhox.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. The grey/Hopf temperature law  T(tau)  (recap of Lecture 1, exact).
# ---------------------------------------------------------------------------
def grey_temperature(teff, tau):
    """Eddington-Kurucz grey T(tau): T = Teff * [3/4 (0.710 + tau - 0.1331 e^{-3.4488 tau})]^{1/4}."""
    bracket = 0.75 * (0.710 + tau - 0.1331 * np.exp(-3.4488 * tau))
    return float(teff) * np.power(np.maximum(bracket, 1e-300), 0.25)


# ---------------------------------------------------------------------------
# 2. Hydrostatic equilibrium by predictor-corrector in log pressure (ATLAS12 TTAUP).
#    kappa == 1 at every layer on the cold start (ROSSTAB returns 1.0 with NROSS=0).
# ---------------------------------------------------------------------------
def ttaup(t, tau, prad, pturb, grav):
    """Integrate dP/dtau = g/kappa over the tau grid in log pressure (Kurucz TTAUP).

    Sequential, order-dependent in j (cannot vectorise across layers): a multistep
    predictor sets log P, then a corrector loop evaluates P_total / kappa / dplog and
    refines log P.  The "evaluate-then-check" ordering (write the stored values from the
    last predictor BEFORE testing convergence) is what makes this bit-exact with Fortran.
    """
    n = int(t.size)
    abstd = np.zeros(n)                 # kappa_Ross at each layer (== 1 on the cold start)
    ptotal = np.zeros(n)                # total pressure P_total = exp(log P)
    pgas = np.zeros(n)                  # gas pressure  P_gas = P_total - P_rad - P_turb

    # log-tau step (the grid is uniform in log tau, so this is a constant)
    dlg_tau = np.log(max(float(tau[1] / max(tau[0], 1e-300)), 1e-300)) if n > 1 else 0.0

    # rolling history of log P and its tau-derivative dplog (init 0)
    plog1 = plog2 = plog3 = plog4 = 0.0
    dplog1 = dplog2 = dplog3 = 0.0

    # seed opacity at the top layer
    abstd[0] = 0.1
    if prad[0] > 0.0:
        abstd[0] = min(0.1, grav * tau[0] / max(prad[0], 1e-300) / 2.0)

    for j in range(n):
        # ---- predictor: extrapolate log P from the rolling history ----
        if j == 0:
            plog = np.log(max(grav / max(abstd[0], 1e-300) * tau[0], 1e-300))
        elif j <= 3:
            plog = plog1 + dplog1
        else:
            plog = (3.0 * plog4 + 8.0 * dplog1 - 4.0 * dplog2 + 8.0 * dplog3) / 3.0

        # ---- corrector loop ----
        # Fortran sets ERROR=1, N=1, then GO TO label 21 -> it WRITES ptotal/pgas/abstd/dplog
        # from the current plog BEFORE computing pnew and BEFORE the convergence test.  We
        # match that "label-21-first" flow exactly so the stored values are bit-identical.
        error = 1.0
        dplog = 0.0
        itn = 1
        while True:
            # LABEL 21: evaluate P_total, gas pressure, opacity, and the derivative first.
            plog = min(plog, 709.78)                    # exp() overflow guard
            ptotal[j] = np.exp(plog)
            pgas[j] = ptotal[j] + (prad[0] - prad[j]) - pturb[j]
            if pgas[j] <= 0.0:                          # non-physical: clamp and stop
                pgas[j] = 1e-30
                abstd[j] = 0.1
                break
            abstd[j] = 1.0                              # ROSSTAB == 1 on the cold start
            dplog = grav / max(abstd[j], 1e-300) * tau[j] / max(ptotal[j], 1e-300) * dlg_tau
            itn += 1
            if itn > 1000 or error <= 5.0e-5:           # converged or out of iterations
                break
            # LABEL 20: corrector estimate of log P, using the just-updated dplog.
            if j == 0:
                pnew = np.log(max(grav / max(abstd[j], 1e-300) * tau[j], 1e-300))
            elif j <= 3:
                pnew = (plog + 2.0 * plog1 + dplog + dplog1) / 3.0
            else:
                pnew = (
                    126.0 * plog1 - 14.0 * plog3 + 9.0 * plog4
                    + 42.0 * dplog + 108.0 * dplog1 - 54.0 * dplog2 + 24.0 * dplog3
                ) / 121.0
            error = abs(pnew - plog)
            plog = 0.5 * (pnew + plog)

        # ---- shift the rolling history forward one layer ----
        plog4 = plog3
        plog3 = plog2
        plog2 = plog1
        plog1 = plog
        dplog3 = dplog2
        dplog2 = dplog1
        dplog1 = dplog

    return abstd, ptotal, pgas


def generate_grey_atm(teff=5770.0, logg=4.44, nrhox=80, tau1lg=-6.875, steplg=0.125):
    """Build the grey starting model: returns tau, T, P_gas, RHOX (column mass)."""
    grav = 10.0 ** float(logg)                          # surface gravity g [cm s^-2]
    j = np.arange(nrhox, dtype=np.float64)
    tau = np.power(10.0, tau1lg + j * steplg)           # 80 layers, 0.125 dex apart
    t = grey_temperature(teff, tau)

    # radiation pressure (Kurucz's floored form) and its surface-subtracted run
    pradk = 2.521e-15 * np.maximum(t ** 4, float(teff) ** 4 / 2.0)
    prad = pradk - pradk[0]
    pturb = np.zeros(nrhox)                              # no turbulence on the cold start

    abstd, ptotal, pgas = ttaup(t, tau, prad, pturb, grav)
    rhox = ptotal / grav                                # column mass [g cm^-2]
    return tau, t, pgas, rhox


def _check(name, got, ref):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    m = np.abs(ref) > 0
    rel = np.zeros_like(ref)
    rel[m] = np.abs(got[m] - ref[m]) / np.abs(ref[m])
    bit = np.array_equal(got, ref)
    print(f"  {name:10s}  max|rel|={rel.max():.3e}  median|rel|={np.median(rel):.3e}  "
          f"bit-exact={bit}")
    return rel.max(), bit


if __name__ == "__main__":
    ref = np.load(ROOT / "reference" / "L1.npz")
    tau, t, pgas, rhox = generate_grey_atm(teff=5770.0, logg=4.44)

    print("grey model atmosphere  (Teff=5770, logg=4.44)  vs reference/L1.npz:")
    r_tau, b_tau = _check("grey_tau", tau, ref["grey_tau"])
    r_t,   b_t   = _check("grey_T", t, ref["grey_T"])
    r_p,   b_p   = _check("grey_pgas", pgas, ref["grey_pgas"])
    r_x,   b_x   = _check("grey_rhox", rhox, ref["grey_rhox"])

    worst = max(r_tau, r_t, r_p, r_x)
    all_bit = b_tau and b_t and b_p and b_x
    print(f"\nworst max|rel| over all four arrays = {worst:.3e}")
    print(f"all four arrays bit-exact = {all_bit}")
    sys.exit(0 if all_bit else 1)
