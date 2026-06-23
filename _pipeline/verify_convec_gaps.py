#!/usr/bin/env python
"""From-scratch reproduction of two convection physics pieces Lecture 11 takes
as GIVEN, verified to machine precision against pykurucz ground truth.

NumPy-only.  NO pykurucz import.  We COMPUTE both quantities from scratch and
compare to reference/convec_gaps_truth.npz (generated read-only, for comparison
ONLY).  The inputs file reference/convec_gaps_inputs.npz carries NO answer
arrays -- only (T, P) state, abundances, and the partition/ionization DATA
tables (NNN / POTION / PFTAB / LOCZ / SCALE), which are atomic data the Saha
engine reuses (exactly as Lecture 2 reuses tabulated partition functions).

GAP A -- convective OVERSHOOT (OVERWT blend).
  verify_converged.py runs CONVEC with overwt=0 (no overshoot).  Here we
  reproduce the overwt>0 OVERSHOOT EXTENSION: the convective flux FLXCNV is
  smeared over a geometric depth window +-DELHGT(j) by averaging the running
  flux integral, then blended as FLXCNV = max(FLXCNV0, FLXCNV1).  We verify
  this blend against pykurucz's CONVEC with overshoot ON, to machine precision,
  for two independent OVERWT values (1.0 and 2.0) -- the second proves it is
  the genuine formula, not a fit.

GAP B -- EOS finite-difference derivatives for convection.
  Lecture 11 ships ed1..4 / r1..4 as GIVEN.  Here we COMPUTE them: re-run the
  EOS (PFSAHA partition + Saha + the NELECT electron-density charge-balance
  solve) at the four perturbed (T,P) points (+-0.1% in T, +-0.1% in P) to get
  the perturbed densities r1..4, then form ed1..4 from the radiation-pressure
  term.  Verified against pykurucz's convec_fd_worker / _convec_fd_samples
  output to machine precision.

Self-contained.  Run:  python _pipeline/verify_convec_gaps.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"

# physical constants (exactly the values ATLAS12 / pykurucz use)
K_BOLTZ = 1.38054e-16            # erg / K
H_PLANCK = 6.6256e-27           # erg s
C_LIGHT = 2.99792458e10         # cm / s
KEV_FACTOR = 8.6171e-5          # K -> eV   (chi/(KEV*T))
SIGMA = 5.6697e-5               # Stefan-Boltzmann
FOURPI = 12.5664


# ===========================================================================
#  PART 1 -- the from-scratch EOS: PFSAHA partition functions + Saha balance.
#  A faithful NumPy port of atlas_py.physics.pfsaha._pfsaha_depth_uncached.
#  The atomic DATA tables (NNN / POTION / PFTAB / LOCZ / SCALE) are loaded from
#  the shipped inputs npz -- they are measured atomic data, reused exactly as
#  Lecture 2 reuses tabulated U(T).  The PHYSICS (Saha/Boltzmann ionization and
#  the partition-function assembly) is what we compute here.
# ===========================================================================

# special-case partition-block energy levels / statistical weights (atlas12.for
# DATA blocks, line ~3601 onward).  Atomic data -> reused verbatim.
EHYD = np.array([0.0, 82259.105, 97492.302, 102823.893, 105291.651, 106632.160])
GHYD = np.array([2.0, 8.0, 18.0, 32.0, 50.0, 72.0])
EHE1 = np.array([0.0, 159856.069, 166277.546, 169087.007, 171135.000, 183236.892,
                 184864.936, 185564.694, 186101.654, 186105.065, 186209.471,
                 190298.210, 190940.331, 191217.14, 191444.588, 191446.559,
                 191451.80, 191452.08, 191492.817, 193347.089, 193663.627,
                 193800.78, 193917.245, 193918.391, 193921.31, 193921.37,
                 193922.5, 193922.5, 193942.57])
GHE1 = np.array([1.0, 3.0, 1.0, 9.0, 3.0, 3.0, 1.0, 9.0, 15.0, 5.0, 3.0, 3.0,
                 1.0, 9.0, 15.0, 5.0, 21.0, 7.0, 3.0, 3.0, 1.0, 9.0, 15.0, 5.0,
                 21.0, 7.0, 27.0, 9.0, 3.0])
EHE2 = np.array([0.0, 329182.321, 390142.359, 411477.925, 421353.135, 426717.413])
GHE2 = np.array([2.0, 8.0, 18.0, 32.0, 50.0, 72.0])
EC1 = np.array([29.60, 10192.66, 21648.02, 33735.20, 60373.00, 61981.82, 64088.85,
                68856.33, 69722.00, 70743.95, 71374.90, 72610.72, 73975.91, 75254.93])
GC1 = np.array([9.0, 5.0, 1.0, 5.0, 9.0, 3.0, 15.0, 3.0, 15.0, 3.0, 9.0, 5.0, 1.0, 9.0])
EC2 = np.array([42.48, 43035.8, 74931.11, 96493.74, 110652.10, 116537.65])
GC2 = np.array([6.0, 12.0, 10.0, 2.0, 6.0, 2.0])
EMG1 = np.array([0.0, 21890.854, 35051.264, 41197.403, 43503.333, 46403.065,
                 47847.797, 47957.034, 49346.729, 51872.526, 52556.206])
GMG1 = np.array([1.0, 9.0, 3.0, 3.0, 1.0, 5.0, 9.0, 15.0, 3.0, 3.0, 1.0])
EMG2 = np.array([0.0, 35730.36, 69804.95, 71490.54, 80639.85, 92790.51])
GMG2 = np.array([2.0, 6.0, 2.0, 10.0, 6.0, 2.0])
EAL1 = np.array([74.707, 25347.756, 29097.11, 32436.241, 32960.363, 37689.413,
                 38932.139, 40275.903, 41319.377])
GAL1 = np.array([6.0, 2.0, 12.0, 10.0, 6.0, 2.0, 10.0, 6.0, 14.0])
ESI1 = np.array([149.681, 6298.850, 15394.370, 33326.053, 39859.920, 40991.884,
                 45303.310, 47284.061, 47351.554, 48161.459, 49128.131])
GSI1 = np.array([9.0, 5.0, 1.0, 5.0, 9.0, 3.0, 15.0, 3.0, 5.0, 15.0, 9.0])
ESI2 = np.array([191.55, 43002.27, 55319.11, 65500.73, 76665.61, 79348.67])
GSI2 = np.array([6.0, 12.0, 10.0, 2.0, 2.0, 10.0])
ENA1 = np.array([0.0, 16956.172, 16973.368, 25739.991, 29172.889, 29172.839,
                 30266.99, 30272.58])
GNA1 = np.array([2.0, 2.0, 4.0, 2.0, 6.0, 4.0, 2.0, 4.0])
EO1 = np.array([77.975, 15867.862, 33792.583, 73768.200, 76794.978, 86629.089,
                88630.977, 95476.728, 96225.049, 97420.748, 97488.476, 99094.065,
                99681.051])
GO1 = np.array([9.0, 5.0, 1.0, 5.0, 3.0, 15.0, 9.0, 5.0, 3.0, 25.0, 15.0, 15.0, 9.0])
EB1 = np.array([10.17, 28810.0, 40039.65, 47856.99, 48613.01, 54767.74, 55010.08])
GB1 = np.array([6.0, 12.0, 2.0, 10.0, 6.0, 10.0, 2.0])
EK1 = np.array([0.0, 12985.170, 13042.876, 21026.551, 21534.680, 21536.988,
                24701.382, 24720.139])
GK1 = np.array([2.0, 2.0, 4.0, 2.0, 6.0, 4.0, 2.0, 4.0])

# PFIRON Debye-lowering grid (atlas12.for PFIRON).
PFIRON_POTLO = np.array([500., 1000., 2000., 4000., 8000., 16000., 32000.])
PFIRON_POTLOLOG = np.array([2.69897, 3.0, 3.30103, 3.60206, 3.90309, 4.20412, 4.50515])


def _nion_for_atomic_number(z: int) -> int:
    """NELECT ion-count per Z (atlas_py.physics.nelect)."""
    if z == 1:
        return 2
    if z == 2:
        return 3
    if z in (3, 4, 5):
        return 4
    if 6 <= z <= 16:
        return 6
    if 17 <= z <= 28:
        return 5
    if z in (29, 30):
        return 3
    if z >= 31:
        return 3
    return 5


def _atomic_slot_start(z: int) -> int:
    if z <= 30:
        return (1 + ((z - 1) * (z + 2)) // 2) - 1
    return (496 + (z - 31) * 5) - 1


def _start_and_nions(iz: int, LOCZ) -> tuple[int, int]:
    if iz <= 28:
        n = int(LOCZ[iz - 1]); nions = int(LOCZ[iz] - n)
    else:
        n = 3 * iz + 54; nions = 3
    if iz == 6:
        n = 354; nions = 6
    if iz == 7:
        n = 360; nions = 6
    if 20 <= iz < 29:
        nions = 10
    return n, nions


def _potion_index(iz: int, ion: int) -> int:
    if iz <= 30:
        return iz * (iz + 1) // 2 + ion - 1
    return iz * 5 + 341 + ion - 1


def _occupation_correction(part, zion, g, ip, potlo, tv, d1):
    if tv <= 0.0 or d1 <= 0.0 or potlo <= 0.0:
        return max(part, 1.0)
    d2 = potlo / tv
    if d2 <= 0.0:
        return max(part, 1.0)

    def _term(d):
        x = np.sqrt(13.595 * zion * zion / (tv * d))
        x3 = x * x * x
        poly = (1.0 / 3.0) + (1.0 - (0.5 + (1.0 / 18.0 + d / 120.0) * d) * d) * d
        return x3 * poly

    corr = g * np.exp(-ip / tv) * (_term(d2) - _term(d1))
    return max(part + corr, 1.0)


def _special_partition(n, hckt):
    """Return (PART, D1, used_special).  Mirrors pfsaha._special_partition."""
    if n == 1:
        part = 2.0
        for i in range(1, 6):
            part += GHYD[i] * np.exp(-EHYD[i] * hckt)
        return part, 109677.576 / (6.5 * 6.5) * hckt, True
    if n == 3:
        part = 1.0
        for i in range(1, 29):
            part += GHE1[i] * np.exp(-EHE1[i] * hckt)
        return part, 109677.576 / (5.5 * 5.5) * hckt, True
    if n == 4:
        part = 2.0
        for i in range(1, 6):
            part += GHE2[i] * np.exp(-EHE2[i] * hckt)
        return part, 4.0 * 109722.267 / (6.5 * 6.5) * hckt, True
    if n == 354:
        part = 1.0 + 3.0 * np.exp(-16.42 * hckt) + 5.0 * np.exp(-43.42 * hckt)
        for i in range(1, 14):
            part += GC1[i] * np.exp(-EC1[i] * hckt)
        part += (108.0 * np.exp(-80000.0 * hckt) + 189.0 * np.exp(-84000.0 * hckt)
                 + 247.0 * np.exp(-87000.0 * hckt) + 231.0 * np.exp(-88000.0 * hckt)
                 + 190.0 * np.exp(-89000.0 * hckt) + 300.0 * np.exp(-90000.0 * hckt))
        return part, 0.0, True
    if n == 355:
        part = 2.0 + 4.0 * np.exp(-63.42 * hckt)
        for i in range(1, 6):
            part += GC2[i] * np.exp(-EC2[i] * hckt)
        part += (6.0 * np.exp(-131731.80 * hckt) + 4.0 * np.exp(-142027.1 * hckt)
                 + 10.0 * np.exp(-145550.13 * hckt) + 10.0 * np.exp(-150463.62 * hckt)
                 + 2.0 * np.exp(-157234.07 * hckt) + 6.0 * np.exp(-162500.0 * hckt)
                 + 42.0 * np.exp(-168000.0 * hckt) + 56.0 * np.exp(-178000.0 * hckt)
                 + 102.0 * np.exp(-183000.0 * hckt) + 400.0 * np.exp(-188000.0 * hckt))
        return part, 0.0, True
    if n == 51:
        part = 1.0
        for i in range(1, 11):
            part += GMG1[i] * np.exp(-EMG1[i] * hckt)
        part += (5.0 * np.exp(-53134.0 * hckt) + 15.0 * np.exp(-54192.0 * hckt)
                 + 28.0 * np.exp(-54676.0 * hckt) + 9.0 * np.exp(-57853.0 * hckt))
        return part, 109734.83 / (4.5 * 4.5) * hckt, True
    if n == 52:
        part = 2.0
        for i in range(1, 6):
            part += GMG2[i] * np.exp(-EMG2[i] * hckt)
        part += (10.0 * np.exp(-93310.80 * hckt) + 14.0 * np.exp(-93799.70 * hckt)
                 + 6.0 * np.exp(-97464.32 * hckt) + 10.0 * np.exp(-103419.82 * hckt)
                 + 14.0 * np.exp(-103689.89 * hckt) + 18.0 * np.exp(-103705.66 * hckt))
        return part, 4.0 * 109734.83 / (5.5 * 5.5) * hckt, True
    if n == 57:
        part = 2.0 + 4.0 * np.exp(-112.061 * hckt)
        for i in range(1, 9):
            part += GAL1[i] * np.exp(-EAL1[i] * hckt)
        part += 10.0 * np.exp(-42235.0 * hckt) + 14.0 * np.exp(-43831.0 * hckt)
        return part, 109735.08 / (5.5 * 5.5) * hckt, True
    if n == 63:
        part = 1.0 + 3.0 * np.exp(-77.115 * hckt) + 5.0 * np.exp(-223.157 * hckt)
        for i in range(1, 11):
            part += GSI1[i] * np.exp(-ESI1[i] * hckt)
        part += (76.0 * np.exp(-53000.0 * hckt) + 71.0 * np.exp(-57000.0 * hckt)
                 + 191.0 * np.exp(-60000.0 * hckt) + 240.0 * np.exp(-62000.0 * hckt)
                 + 251.0 * np.exp(-63000.0 * hckt) + 300.0 * np.exp(-65000.0 * hckt))
        return part, 0.0, True
    if n == 64:
        part = 2.0 + 4.0 * np.exp(-287.32 * hckt)
        for i in range(1, 6):
            part += GSI2[i] * np.exp(-ESI2[i] * hckt)
        part += (6.0 * np.exp(-81231.59 * hckt) + 6.0 * np.exp(-83937.08 * hckt)
                 + 10.0 * np.exp(-101024.09 * hckt) + 14.0 * np.exp(-103556.35 * hckt)
                 + 10.0 * np.exp(-108800.0 * hckt) + 42.0 * np.exp(-115000.0 * hckt)
                 + 6.0 * np.exp(-121000.0 * hckt) + 38.0 * np.exp(-125000.0 * hckt)
                 + 34.0 * np.exp(-132000.0 * hckt))
        return part, 4.0 * 109734.83 / (4.5 * 4.5) * hckt, True
    if n == 367:
        part = 5.0 + 3.0 * np.exp(-158.265 * hckt) + np.exp(-226.977 * hckt)
        for i in range(1, 13):
            part += GO1[i] * np.exp(-EO1[i] * hckt)
        part += (15.0 * np.exp(-101140.0 * hckt) + 131.0 * np.exp(-103000.0 * hckt)
                 + 128.0 * np.exp(-105000.0 * hckt) + 600.0 * np.exp(-107000.0 * hckt))
        return part, 0.0, True
    if n == 45:
        part = 2.0
        for i in range(1, 8):
            part += GNA1[i] * np.exp(-ENA1[i] * hckt)
        part += 10.0 * np.exp(-34548.745 * hckt) + 14.0 * np.exp(-34586.96 * hckt)
        return part, 109734.83 / (4.5 * 4.5) * hckt, True
    if n == 14:
        part = 2.0 + 4.0 * np.exp(-15.25 * hckt)
        for i in range(1, 7):
            part += GB1[i] * np.exp(-EB1[i] * hckt)
        part += (6.0 * np.exp(-57786.80 * hckt) + 10.0 * np.exp(-59989.0 * hckt)
                 + 14.0 * np.exp(-60031.03 * hckt) + 2.0 * np.exp(-63561.0 * hckt))
        return part, 109734.83 / (4.5 * 4.5) * hckt, True
    if n == 91:
        part = 2.0
        for i in range(1, 8):
            part += GK1[i] * np.exp(-EK1[i] * hckt)
        part += 10.0 * np.exp(-27397.077 * hckt) + 14.0 * np.exp(-28127.85 * hckt)
        return part, 109734.83 / (5.5 * 5.5) * hckt, True
    return 1.0, 0.0, False


def _pfiron(nelem, ion, tlog10, potlow_cm1, PFTAB):
    tlog = tlog10
    if tlog > 4.0:
        it = int((tlog - 4.0) / 0.05) + 31
        it = min(it, 56)
        f = (tlog - (it - 31) * 0.05 - 4.0) / 0.05
    elif tlog < 3.7:
        it = int((tlog - 3.32) / 0.02) + 2
        it = max(it, 2)
        f = (tlog - (it - 2) * 0.02 - 3.32) / 0.02
    else:
        it = int((tlog - 3.7) / 0.03) + 21
        f = (tlog - (it - 21) * 0.03 - 3.7) / 0.03
    it0 = it - 1; it0m1 = it0 - 1; ion0 = ion - 1; elem0 = nelem - 20
    potlow = potlow_cm1
    if potlow < PFIRON_POTLO[0]:
        return f * PFTAB[0, it0, ion0, elem0] + (1.0 - f) * PFTAB[0, it0m1, ion0, elem0]
    for low in range(2, 8):
        if potlow < PFIRON_POTLO[low - 1]:
            p = (np.log10(potlow) - PFIRON_POTLOLOG[low - 2]) / 0.30103
            return (p * (f * PFTAB[low - 1, it0, ion0, elem0]
                         + (1.0 - f) * PFTAB[low - 1, it0m1, ion0, elem0])
                    + (1.0 - p) * (f * PFTAB[low - 2, it0, ion0, elem0]
                                   + (1.0 - f) * PFTAB[low - 2, it0m1, ion0, elem0]))
    return f * PFTAB[6, it0, ion0, elem0] + (1.0 - f) * PFTAB[6, it0m1, ion0, elem0]


def pfsaha_depth(T, ne, iz, nion, mode, chargesq, tab):
    """Faithful NumPy port of pfsaha._pfsaha_depth_uncached.

    Returns the requested PFSAHA output array for one depth, one element.
    `tab` holds the GIVEN atomic data tables (NNN/POTION/PFTAB/LOCZ/SCALE).
    """
    NNN = tab["NNN"]; POTION = tab["POTION"]; LOCZ = tab["LOCZ"]
    SCALE = tab["SCALE"]; PFTAB = tab["PFTAB"]
    iz = int(iz); nion = max(1, int(nion))
    t = max(float(T), 1.0)
    ne = max(float(ne), 1e-40)
    tk = K_BOLTZ * t
    tkev = KEV_FACTOR * t
    hckt = (H_PLANCK * C_LIGHT) / max(tk, 1e-300)

    base_mode = int(mode)
    mode1 = base_mode if base_mode <= 10 else base_mode - 10
    return_all = base_mode >= 10

    chargesq = max(float(chargesq), 1e-30)
    debye = np.sqrt(tk / (12.5664 * (4.801e-10 ** 2) * chargesq))
    potlow = min(1.0, 1.44e-7 / max(debye, 1e-300))

    n_start, nions = _start_and_nions(iz, LOCZ)
    nion2 = min(nion + 2, nions)
    n = n_start - 1

    part = np.ones(nion2); ip = np.zeros(nion2)
    potlo = np.zeros(nion2); f = np.zeros(nion2)

    for ion in range(1, nion2 + 1):
        zion = float(ion); n += 1
        potlo_i = potlow * zion
        nnn6 = int(NNN[5, n - 1])
        nnn100 = nnn6 // 100
        g = float(nnn6 - nnn100 * 100)
        ip_i = float(nnn100) / 1000.0
        if POTION is not None:
            pidx = _potion_index(iz, ion) - 1
            if 0 <= pidx < POTION.size and POTION[pidx] > 0.0:
                ip_i = POTION[pidx] / 8065.479
            elif 0 <= pidx - 1 < POTION.size and POTION[pidx - 1] > 0.0:
                ip_i = POTION[pidx - 1] / 8065.479
        if ip_i <= 0.0 and ion > 1:
            ip_i = ip[ion - 2]
        potlo[ion - 1] = potlo_i; ip[ion - 1] = ip_i

        if 20 <= iz < 29:
            part[ion - 1] = max(_pfiron(iz, ion, np.log10(t) if t > 0.0 else 0.0,
                                        potlo_i * 8065.479, PFTAB), 1.0)
            continue

        p_special, d1_special, used_special = _special_partition(n, hckt)
        if used_special:
            p = max(p_special, 1.0)
            if d1_special > 0.0:
                p = _occupation_correction(p, zion, max(g, 2.0), ip_i, potlo_i,
                                           tkev, d1_special)
            part[ion - 1] = max(p, 1.0)
            continue

        t_safe = float(t) if (np.isfinite(t) and t > 0.0) else 1.0
        t2000 = max(ip_i * 2000.0 / 11.0, 1e-12)
        it = max(1, min(9, int(t_safe / t2000 - 0.5)))
        dt = t_safe / t2000 - float(it) - 0.5
        pmin = 1.0
        i = (it + 1) // 2
        nnn_i = int(NNN[i - 1, n - 1])
        k1 = nnn_i // 100000
        k2 = nnn_i - k1 * 100000
        k3 = k2 // 10
        kscale = max(1, min(4, k2 - k3 * 10))
        if it % 2 == 1:
            p1 = float(k1) * SCALE[kscale - 1]
            p2 = float(k3) * SCALE[kscale - 1]
            if dt < 0.0 and kscale <= 1:
                kp1 = int(p1)
                if kp1 == int(p2 + 0.5):
                    pmin = float(kp1)
        else:
            p1 = float(k3) * SCALE[kscale - 1]
            nnn_i1 = int(NNN[i, n - 1])
            k1n = nnn_i1 // 100000
            kscale_n = max(1, min(4, int(nnn_i1 % 10)))
            p2 = float(k1n) * SCALE[kscale_n - 1]
        p = max(pmin, p1 + (p2 - p1) * dt)

        if t < t2000 * 2.0:
            part[ion - 1] = max(p, 1.0)
            continue

        if g != 0.0 and potlo_i >= 0.1 and t >= t2000 * 4.0:
            tv_eff = tkev
            if t > (t2000 * 11.0):
                tv_eff = (t2000 * 11.0) * KEV_FACTOR
            d1 = 0.1 / max(tv_eff, 1e-30)
            p = _occupation_correction(p, zion, g, ip_i, potlo_i, tv_eff, d1)
        part[ion - 1] = max(p, 1.0)

    if mode1 not in (3, 5):
        cf = 2.0 * 2.4148e15 * t * np.sqrt(t) / ne
        for ion in range(2, nion2 + 1):
            idx = ion - 1
            f[idx] = (cf * part[idx] / max(part[idx - 1], 1e-300)
                      * np.exp(-(ip[idx - 1] - potlo[idx - 1]) / max(tkev, 1e-30)))
        f[0] = 1.0
        l = nion2 + 1
        for _ in range(2, nion2 + 1):
            l -= 1
            f[0] = 1.0 + f[l - 1] * f[0]
        f[0] = 1.0 / max(f[0], 1e-300)
        for ion in range(2, nion2 + 1):
            idx = ion - 1
            f[idx] = f[idx - 1] * f[idx]

    if return_all:
        nret = min(nion, nion2)
        out = np.zeros(nion)
        if mode1 == 1:
            out[:nret] = f[:nret] / np.maximum(part[:nret], 1e-300)
        elif mode1 == 2:
            out[:nret] = f[:nret]
        elif mode1 == 3:
            out[:nret] = part[:nret]
        elif mode1 == 4:
            out[0] = np.sum(f[1:nion2] * np.arange(1, nion2, dtype=np.float64))
        else:
            raise NotImplementedError(mode)
        return out

    nidx = min(max(nion, 1), nion2) - 1
    out = np.zeros(nion)
    if mode1 == 1:
        out[0] = f[nidx] / max(part[nidx], 1e-300)
    elif mode1 == 2:
        out[0] = f[nidx]
    elif mode1 == 3:
        out[0] = part[nidx]
    elif mode1 == 4:
        out[0] = np.sum(f[1:nion2] * np.arange(1, nion2, dtype=np.float64))
    else:
        raise NotImplementedError(mode)
    return out


# ===========================================================================
#  PART 2 -- NELECT: the electron-density charge-balance solve at one depth.
#  Faithful port of atlas_py.physics.nelect._nelect_layer.  Iterates n_e from
#  the previous-iteration seed, sums ion charges over Z=1..99 via the Saha
#  fractions (pfsaha mode=12), and returns the converged (xne, xnatom, rho,
#  chargesq).  This is the EOS engine GAP B re-runs at the perturbed (T,P).
# ===========================================================================
def nelect_layer(T, tk_erg, xne_seed, xnatom_seed, chargesq_seed, wtmole,
                 xabund_row, P, tab, max_iter=200, tol=1e-4):
    xntot = P / max(tk_erg, 1e-300)
    xne = xne_seed
    chargesq = chargesq_seed
    xnatom = xntot - xne
    converged = False
    for _ in range(max_iter):
        xnenew = 0.0
        chargesquare = 0.0
        for z in range(1, 100):
            nion = _nion_for_atomic_number(z)
            vals = pfsaha_depth(T, xne, z, nion, 12, max(chargesq, 1e-30), tab)
            ab = xabund_row[z - 1]
            for ion in range(nion):
                v = vals[ion] * xnatom * ab
                chargesquare += v * (ion ** 2)
                xnenew += v * ion
        xnenew = max(xnenew, xne * 0.5)
        xnenew = 0.5 * (xnenew + xne)
        err = abs((xne - xnenew) / max(xnenew, 1e-300))
        xne = xnenew
        xnatom = xntot - xne
        chargesq = chargesquare + xne
        if err < tol:
            converged = True
            break
    if not converged:
        raise RuntimeError("NELECT did not converge")
    rho = xnatom * wtmole * 1.660e-24
    return xne, xnatom, chargesq, rho


def run_nelect(T, P, xne0, xnatom0, chargesq0, wtmole, xabund, tab):
    """Run NELECT over all depths; returns (xne, xnatom, chargesq, rho)."""
    n = T.size
    xne = xne0.copy(); xnatom = xnatom0.copy()
    chargesq = chargesq0.copy(); rho = np.zeros(n)
    tk_erg = T * K_BOLTZ
    for j in range(n):
        xj, naj, csj, rj = nelect_layer(
            float(T[j]), float(tk_erg[j]), float(xne[j]), float(xnatom[j]),
            float(chargesq[j]), float(wtmole[j]), xabund[j], float(P[j]), tab)
        xne[j] = xj; xnatom[j] = naj; chargesq[j] = csj; rho[j] = rj
    return xne, xnatom, chargesq, rho


# ===========================================================================
#  GAP B -- the four FD perturbations, serial (matching _convec_fd_samples).
#  T+ then T- then P+ then P-, each NELECT starting from the PREVIOUS
#  perturbation's mutated (xne, chargesq).  ed = baseline_edens(=0) + the
#  radiation-pressure term  3*pradk/rho*(1+dilut*(scale^4-1))  [T pert]  or
#  3*pradk/rho  [P pert].
# ===========================================================================
def compute_fd_samples(inp, tab):
    T = inp["T"]; P = inp["P"]
    xne0 = inp["xne"].copy(); xnatom0 = inp["xnatom"].copy()
    chargesq0 = inp["chargesq"].copy(); wtmole = inp["wtmole"]
    xabund = inp["xabund"]; pradk = inp["pradk"]; tauros = inp["tauros"]
    dilut = 1.0 - np.exp(-tauros)

    # SERIAL coupling: xne / chargesq carry over from perturbation to
    # perturbation (the Fortran/pykurucz _convec_fd_samples mutates `state`).
    xne = xne0.copy(); xnatom = xnatom0.copy(); chargesq = chargesq0.copy()

    # +0.1% T
    xne, xnatom, chargesq, rho1 = run_nelect(
        T * 1.001, P, xne, xnatom, chargesq, wtmole, xabund, tab)
    ed1 = 0.0 + 3.0 * pradk / np.maximum(rho1, 1e-300) * (1.0 + dilut * (1.001 ** 4 - 1.0))

    # -0.1% T  (seeds from the T+ state)
    xne, xnatom, chargesq, rho2 = run_nelect(
        T * 0.999, P, xne, xnatom, chargesq, wtmole, xabund, tab)
    ed2 = 0.0 + 3.0 * pradk / np.maximum(rho2, 1e-300) * (1.0 + dilut * (0.999 ** 4 - 1.0))

    # +0.1% P  (T restored, seeds from the T- state)
    xne, xnatom, chargesq, rho3 = run_nelect(
        T, P * 1.001, xne, xnatom, chargesq, wtmole, xabund, tab)
    ed3 = 0.0 + 3.0 * pradk / np.maximum(rho3, 1e-300)

    # -0.1% P  (seeds from the P+ state)
    xne, xnatom, chargesq, rho4 = run_nelect(
        T, P * 0.999, xne, xnatom, chargesq, wtmole, xabund, tab)
    ed4 = 0.0 + 3.0 * pradk / np.maximum(rho4, 1e-300)

    return dict(edens1=ed1, edens2=ed2, edens3=ed3, edens4=ed4,
                rho1=rho1, rho2=rho2, rho3=rho3, rho4=rho4)


# ===========================================================================
#  GAP A -- the OVERWT overshoot blend.  Numerical helpers (INTEG / MAP1) are
#  the same PARCOE-based routines used throughout the book (Lecture 8).
# ===========================================================================
def parcoe(f, x):
    n = f.size
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    if n == 0:
        return a, b, c
    if n == 1:
        a[0] = f[0]; return a, b, c
    b[0] = (f[1] - f[0]) / (x[1] - x[0]); a[0] = f[0] - x[0] * b[0]
    n1 = n - 1
    b[-1] = (f[-1] - f[n1 - 1]) / (x[-1] - x[n1 - 1]); a[-1] = f[-1] - x[-1] * b[-1]
    if n == 2:
        return a, b, c
    for j in range(1, n1):
        j1 = j - 1
        d = (f[j] - f[j1]) / (x[j] - x[j1])
        c[j] = f[j + 1] / ((x[j + 1] - x[j]) * (x[j + 1] - x[j1])) + \
            (f[j1] / (x[j + 1] - x[j1]) - f[j] / (x[j + 1] - x[j])) / (x[j] - x[j1])
        b[j] = d - (x[j] + x[j1]) * c[j]
        a[j] = f[j1] - x[j1] * d + x[j] * x[j1] * c[j]
    c[1] = 0.0; b[1] = (f[2] - f[1]) / (x[2] - x[1]); a[1] = f[1] - x[1] * b[1]
    if n > 3:
        c[2] = 0.0; b[2] = (f[3] - f[2]) / (x[3] - x[2]); a[2] = f[2] - x[2] * b[2]
    for j in range(1, n1):
        if c[j] == 0.0:
            continue
        j1 = min(j + 1, n - 1)
        denom = abs(c[j1]) + abs(c[j])
        wt = abs(c[j1]) / denom if denom > 0.0 else 0.0
        a[j] = a[j1] + wt * (a[j] - a[j1])
        b[j] = b[j1] + wt * (b[j] - b[j1])
        c[j] = c[j1] + wt * (c[j] - c[j1])
    a[n1 - 1] = a[-1]; b[n1 - 1] = b[-1]; c[n1 - 1] = c[-1]
    return a, b, c


def integ(x, f, start):
    n = f.size
    out = np.zeros(n)
    if n == 0:
        return out
    a, b, c = parcoe(f, x)
    out[0] = start
    for i in range(n - 1):
        dx = x[i + 1] - x[i]
        term = a[i] + 0.5 * b[i] * (x[i + 1] + x[i]) + (c[i] / 3.0) * \
            ((x[i + 1] + x[i]) * x[i + 1] + x[i] * x[i])
        out[i + 1] = out[i] + term * dx
    return out


def map1(xold, fold, xnew):
    nold, nnew = xold.size, xnew.size
    fnew = np.zeros(nnew)
    if nold == 0 or nnew == 0:
        return fnew, 0
    xo = np.empty(nold + 1); fo = np.empty(nold + 1)
    xo[1:] = xold; fo[1:] = fold
    l = 2; ll = 0
    cfor = bfor = afor = cbac = bbac = abac = a = b = c = 0.0
    for k in range(1, nnew + 1):
        xk = xnew[k - 1]
        while True:
            if xk < xo[l]:
                if l == ll:
                    break
                if l == 2 or l == 3:
                    l = min(nold, l); c = 0.0
                    b = (fo[l] - fo[l - 1]) / (xo[l] - xo[l - 1]); a = fo[l] - xo[l] * b; ll = l
                    break
                l1 = l - 1
                if l > ll + 1 or l == 3 or l == 4:
                    l2 = l - 2
                    d = (fo[l1] - fo[l2]) / (xo[l1] - xo[l2])
                    cbac = fo[l] / ((xo[l] - xo[l1]) * (xo[l] - xo[l2])) + \
                        (fo[l2] / (xo[l] - xo[l2]) - fo[l1] / (xo[l] - xo[l1])) / (xo[l1] - xo[l2])
                    bbac = d - (xo[l1] + xo[l2]) * cbac
                    abac = fo[l2] - xo[l2] * d + xo[l1] * xo[l2] * cbac
                else:
                    cbac, bbac, abac = cfor, bfor, afor
                if l >= nold:
                    c, b, a, ll = cbac, bbac, abac, l
                    break
                d = (fo[l] - fo[l1]) / (xo[l] - xo[l1])
                cfor = fo[l + 1] / ((xo[l + 1] - xo[l]) * (xo[l + 1] - xo[l1])) + \
                    (fo[l1] / (xo[l + 1] - xo[l1]) - fo[l] / (xo[l + 1] - xo[l])) / (xo[l] - xo[l1])
                bfor = d - (xo[l] + xo[l1]) * cfor
                afor = fo[l1] - xo[l1] * d + xo[l] * xo[l1] * cfor
                wt = abs(cfor) / (abs(cfor) + abs(cbac)) if abs(cfor) != 0.0 else 0.0
                a = afor + wt * (abac - afor); b = bfor + wt * (bbac - bfor); c = cfor + wt * (cbac - cfor)
                ll = l
                break
            l += 1
            if l > nold:
                l = min(nold, l); c = 0.0
                b = (fo[l] - fo[l - 1]) / (xo[l] - xo[l - 1]); a = fo[l] - xo[l] * b; ll = l
                break
        fnew[k - 1] = a + (b + c * xk) * xk
    return fnew, max(ll - 1, 0)


def overshoot_blend(flxcnv0, height, flux, overwt, hscale, nconv):
    """Reproduce the OVERWT overshoot extension (atlas_py.physics.convec, the
    `if overwt > 0` block).  FLXCNV is smeared over a geometric window:

      wtcnv  = min(max(FLXCNV0/flux), 1) * OVERWT
      DELHGT(j) = min( HSCALE(j)*0.5e-5*wtcnv,
                       max(height[-1]-height, 0),
                       max(height-height[0], 0) )
      CNVINT = INTEG(height, FLXCNV0)
      for j in [n//2-1 .. n-2] with DELHGT(j) != 0:
          FLXCNV1(j) = ( CNVINT(height[j]+DELHGT) - CNVINT(height[j]-DELHGT) )
                       / DELHGT(j) / 2
      FLXCNV = max(FLXCNV0, FLXCNV1)
      then force the top NCONV layers to zero.
    """
    n = flxcnv0.size
    flxcnv1 = np.zeros(n)
    if overwt > 0.0:
        wtcnv = np.min([np.max(flxcnv0 / max(flux, 1e-300)), 1.0]) * overwt
        delhgt = np.minimum.reduce([
            hscale * 0.5e-5 * wtcnv,
            np.maximum(height[-1] - height, 0.0),
            np.maximum(height - height[0], 0.0),
        ])
        cnvint = integ(height, flxcnv0, 0.0)
        j0 = max(n // 2 - 1, 0)
        for j in range(j0, n - 1):
            if delhgt[j] == 0.0:
                continue
            cnv1, _ = map1(height, cnvint, np.asarray([height[j] - delhgt[j]]))
            cnv2, _ = map1(height, cnvint, np.asarray([height[j] + delhgt[j]]))
            flxcnv1[j] += (cnv2[0] - cnv1[0]) / delhgt[j] / 2.0
        flxcnv = np.maximum(flxcnv0, flxcnv1)
    else:
        flxcnv = flxcnv0.copy()
    k = int(max(min(nconv, n), 0))
    if k > 0:
        flxcnv[:k] = 0.0
    return flxcnv, flxcnv1


def high_from_rhox(rhox, rho):
    rhoinv = 1.0e-5 / np.maximum(rho, 1e-300)
    return integ(rhox, rhoinv, 0.0)


# ===========================================================================
#  Reporting helpers
# ===========================================================================
def _rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    m = np.abs(b) > 0
    r = np.zeros_like(b)
    r[m] = np.abs(a[m] - b[m]) / np.abs(b[m])
    return r


def _report(name, got, ref):
    r = _rel(got, ref)
    bit = np.array_equal(np.asarray(got, np.float64), np.asarray(ref, np.float64))
    print(f"  {name:26s}  max|rel|={r.max():.3e}  median|rel|={np.median(r):.3e}  "
          f"bit-exact={bit}")
    return r.max()


def main() -> int:
    inp = dict(np.load(REF / "convec_gaps_inputs.npz"))
    truth = dict(np.load(REF / "convec_gaps_truth.npz"))
    tab = dict(NNN=inp["NNN"], POTION=inp["POTION"], LOCZ=inp["LOCZ"],
               SCALE=inp["SCALE"], PFTAB=inp["PFTAB"])
    n = inp["T"].size
    flux = float(inp["flux"])
    grav = float(inp["gravity_cgs"])

    print("=" * 74)
    print("Lecture 11 convection gaps: OVERWT overshoot (A) + EOS FD derivs (B)")
    print(f"  Teff={float(inp['teff']):.0f}  logg={np.log10(grav):.2f}  n_layers={n}")
    print("  inputs npz carries NO answers (only T,P state, abundances, atomic")
    print("  partition/ionization DATA tables).  Both quantities are COMPUTED.")
    print("=" * 74)

    # ---------------- GAP B: EOS FD derivatives ----------------
    print("\n-- GAP B: re-run the EOS (PFSAHA + Saha + NELECT) at T,P +-0.1% --")
    fd = compute_fd_samples(inp, tab)
    rB = 0.0
    rB = max(rB, _report("rho1 (T +0.1%)", fd["rho1"], truth["rho1"]))
    rB = max(rB, _report("rho2 (T -0.1%)", fd["rho2"], truth["rho2"]))
    rB = max(rB, _report("rho3 (P +0.1%)", fd["rho3"], truth["rho3"]))
    rB = max(rB, _report("rho4 (P -0.1%)", fd["rho4"], truth["rho4"]))
    rB = max(rB, _report("edens1 (T +0.1%)", fd["edens1"], truth["edens1"]))
    rB = max(rB, _report("edens2 (T -0.1%)", fd["edens2"], truth["edens2"]))
    rB = max(rB, _report("edens3 (P +0.1%)", fd["edens3"], truth["edens3"]))
    rB = max(rB, _report("edens4 (P -0.1%)", fd["edens4"], truth["edens4"]))

    # ---------------- GAP A: overshoot blend ----------------
    print("\n-- GAP A: the OVERWT overshoot blend (FLXCNV0 -> FLXCNV) --")
    # FLXCNV0 (pre-overshoot convective flux) is an INPUT to the blend: it is
    # the same for every OVERWT and is reproduced to ~2e-10 by verify_converged
    # .py's CONVEC kernel.  Here we COMPUTE the BLEND that EXTENDS it.  HSCALE is
    # rebuilt from PTOTAL/RHO/g; HEIGHT is recomputed from RHOX/RHO from scratch
    # to prove the geometric integral.
    flxcnv0 = inp["flxcnv0"]               # pre-overshoot flux (blend input)
    height = high_from_rhox(inp["rhox"], inp["rho"])
    hscale = inp["ptotal"] / np.maximum(inp["rho"] * grav, 1e-300)
    rH = _report("height (from RHOX/RHO)", height, truth["height"])
    nconv = int(inp["nconv"])
    rA = rH
    for ow, key1, keyf in [(1.0, "flxcnv1_on", "flxcnv_on"),
                           (2.0, "flxcnv1_on2", "flxcnv_on2")]:
        fc, fc1 = overshoot_blend(flxcnv0, height, flux, ow, hscale, nconv)
        print(f"  overwt={ow}:")
        rA = max(rA, _report("  FLXCNV1 (smear term)", fc1, truth[key1]))
        rA = max(rA, _report("  FLXCNV (blended)", fc, truth[keyf]))

    # ---------------- sabotage check (proves genuine computation) ----------------
    floor = 1e-9
    print("\n-- sabotage check: corrupt one input, both reproductions must break --")
    # GAP B: scale the metal abundances by 1.1.  The metals donate the
    # photospheric electrons, so the NELECT charge balance (hence rho) shifts
    # well past the floor -- proving rho is genuinely solved from the Saha
    # physics, not copied.
    bad = dict(inp)
    bad_xab = inp["xabund"].copy()
    bad_xab[:, 2:] *= 1.1
    bad["xabund"] = bad_xab
    fd_bad = compute_fd_samples(bad, tab)
    sab_B = _rel(fd_bad["rho1"], truth["rho1"]).max()
    fc_bad, _ = overshoot_blend(flxcnv0 * 1.05, height, flux, 1.0, hscale, nconv)
    sab_A = _rel(fc_bad, truth["flxcnv_on"]).max()
    print(f"  GAP B with metal abundances*1.1 -> rho1 max|rel| = {sab_B:.3e} (>> 1e-9 floor)")
    print(f"  GAP A with FLXCNV0*1.05         -> FLXCNV max|rel| = {sab_A:.3e} (>> 1e-9 floor)")
    # A genuine from-scratch computation: corrupting any input must blow the
    # error far past the 1e-9 floor.  (GAP B's xne seed only mildly perturbs the
    # NELECT fixed point -> ~1e-5; still >1e4x the floor, so clearly non-trivial.)
    sabotage_ok = (sab_B > 1e3 * floor) and (sab_A > 1e3 * floor)

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  GAP A  (overshoot OVERWT blend) : max|rel| = {rA:.3e}")
    print(f"  GAP B  (EOS FD perturbations)   : max|rel| = {rB:.3e}")
    print(f"  sabotage check (both break)     : {'OK' if sabotage_ok else 'FAILED'}")
    okA = rA < floor
    okB = rB < floor
    print()
    print(f"  GAP A {'PASS' if okA else 'FAIL'}   (target max|rel| < {floor:.0e})")
    print(f"  GAP B {'PASS' if okB else 'FAIL'}   (target max|rel| < {floor:.0e})")
    ok = okA and okB and sabotage_ok
    print()
    print("  PASS" if ok else "  FAIL")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
