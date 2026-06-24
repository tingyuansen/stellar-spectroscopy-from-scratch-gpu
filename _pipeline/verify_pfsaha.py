#!/usr/bin/env python
"""From-scratch reproduction of pykurucz's PFSAHA ionization core, verified to
machine precision against pykurucz's own PFSAHA output.

NumPy-only.  NO pykurucz import.  We COMPUTE, per depth and per element, the
per-ion partition functions U and the per-ion populations

    population_per_ion = n_ion / U = (F / PART) * xnatom * xabund

from scratch, and compare to reference/pfsaha_truth.npz, which holds pykurucz's
own PFSAHA output (the COMPARISON TARGET, loaded ONLY to compare).

The inputs file reference/pfsaha_inputs.npz carries NO answer arrays -- only the
depth STATE (T, Pe via electron_density, Pg via gas_pressure, xnatom, xabund)
and the atomic-data TABLES the Saha engine reuses (NNN packed partition data,
POTION ionization potentials, the PFIRON Fe-group grid, the special-element
level blocks for H/He/C/Mg/Al/Si/O/Na/B/K/Ca, the PFGROUND ground-state
correction pre-evaluated on the T grid, and LOCZ/SCALE).  These are measured
atomic data, reused exactly as Lecture 2 reuses tabulated U(T): the PHYSICS we
compute here is the partition-function assembly and the Saha/Boltzmann ladder.

What is reproduced bit-exact:
  * the per-ion partition functions U (PFSAHA mode 13),
  * the per-ion population fractions F/PART (mode 11),
  * hence population_per_ion = n_ion/U at the atmosphere's own XNATOM,
  * and the NELECT electron-density charge balance solved from scratch.

The stored atmosphere.npz['population_per_ion'] additionally folds in NMOLEC
molecular depletion (atoms locked into H2/CO/... removed from XNATOM before the
multiply).  NMOLEC is a separate molecular-equilibrium engine (Lectures 7+), NOT
part of the PFSAHA ionization core, so the PFSAHA-core target here is pykurucz's
PFSAHA output evaluated at the atmosphere's XNATOM -- the genuine per-ion
populations of the Saha/partition physics.

Self-contained.  Run:  python _pipeline/verify_pfsaha.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"

# physical constants (exactly the values ATLAS / pykurucz carry)
EV_TO_CM = 8065.479          # 1 eV in cm^-1
KEV_FACTOR = 8.6171e-5       # K -> eV   (kT[eV] = KEV*T)
SAHA = 2.4148e15             # (2*pi*m_e*k / h^2)^{3/2}  [cm^-3 K^-3/2]


# ===========================================================================
#  The Fe-group partition functions (PFIRON): Ca..Ni (Z=20-28) read their
#  partition functions from a measured (potlow, log10 T) grid PFTAB rather than
#  the packed NNN data.  Bilinear interpolation in log10(T) and log10(potlow).
# ===========================================================================
def pfiron(iz, ion, tlog8, potlow8, PFTAB, POTLO, POTLOLOG):
    elem_idx = iz - 20          # 0..8 for Ca..Ni
    ion_idx = ion - 1
    tlog = tlog8                # log10 T
    # locate the temperature bracket on PFIRON's three-piece log10 T grid
    if tlog > 4.0:
        it = max(1, min(56, int((tlog - 4.0) / 0.05 + 31.0)))
        f = (tlog - (it - 31) * 0.05 - 4.0) / 0.05
    elif tlog < 3.7:
        it = max(2, int((tlog - 3.32) / 0.02 + 2.0))
        f = (tlog - (it - 2) * 0.02 - 3.32) / 0.02
    else:
        it = int((tlog - 3.7) / 0.03 + 21.0)
        f = (tlog - (it - 21) * 0.03 - 3.7) / 0.03
    it = max(1, min(56, it))
    it_idx = it - 1
    itm1 = max(0, it_idx - 1)
    # below the lowest tabulated lowering: interpolate in temperature only
    if potlow8 < POTLO[0]:
        return float(f * PFTAB[0, it_idx, ion_idx, elem_idx]
                     + (1.0 - f) * PFTAB[0, itm1, ion_idx, elem_idx])
    # otherwise bracket the lowering and do the second (log-potlow) interpolation
    low = len(POTLO) - 1
    for i in range(1, len(POTLO)):
        if potlow8 < POTLO[i]:
            low = i
            break
    p = (np.log10(potlow8) - POTLOLOG[low - 1]) / 0.30103
    return float(p * (f * PFTAB[low, it_idx, ion_idx, elem_idx]
                      + (1.0 - f) * PFTAB[low, itm1, ion_idx, elem_idx])
                 + (1.0 - p) * (f * PFTAB[low - 1, it_idx, ion_idx, elem_idx]
                                + (1.0 - f) * PFTAB[low - 1, itm1, ion_idx, elem_idx]))


def potion_index(iz, ion):
    """Flat index of ion's ionization potential in the POTION table (1-based)."""
    if iz <= 30:
        return iz * (iz + 1) // 2 + ion - 1
    return iz * 5 + 341 + ion - 1


def occupation_term(d, zion, tv):
    """One side of the high-T occupation-probability correction (Hummer-Mihalas
    style): the extra effective states from the truncated Rydberg series."""
    x3 = np.sqrt(13.595 * zion * zion / tv / d) ** 3
    poly = 1.0 / 3.0 + (1.0 - (0.5 + (1.0 / 18.0 + d / 120.0) * d) * d) * d
    return x3 * poly


def pfsaha_layer(iz, nion, T, tkev, tk, hckt, tlog, P, xne, tab, layer):
    """Faithful NumPy port of pykurucz's pfsaha_exact, one depth, one element.

    Returns (U, popfrac, nion2): the per-ion partition functions U=PART, the
    per-ion population fractions popfrac = F/PART (= n_ion / (U*n_element)), and
    the number of ion stages computed.
    """
    NNN = tab["NNN"]; POTION = tab["POTION"]; LOCZ = tab["LOCZ"]; SCALE = tab["SCALE"]
    PFTAB = tab["PFTAB"]; PFIRON_POTLO = tab["PFIRON_POTLO"]
    PFIRON_POTLOLOG = tab["PFIRON_POTLOLOG"]; PFG = tab["pfground_tab"]
    sp = tab  # special-element blocks live under sp_* keys

    # Debye screening of the ionization potential.  The screening charge is
    # n_e + n_ion (~2 n_e), with a correction when 2 n_e exceeds n_total.
    CHARGE = xne * 2.0
    EXCESS = 2.0 * xne - P / tk
    if EXCESS > 0.0:
        CHARGE += 2.0 * EXCESS
    if CHARGE == 0.0:
        CHARGE = 1.0
    DEBYE = np.sqrt(tk / 2.8965e-18 / CHARGE)
    POTLOW = min(1.0, 1.44e-7 / DEBYE)            # lowering per unit charge [eV]

    # how many stages this element occupies in the packed tables, and where
    if iz <= 28:
        n = int(LOCZ[iz - 1])
        nions = int(LOCZ[iz] - LOCZ[iz - 1]) if iz < len(LOCZ) else 3
    else:
        n = 3 * iz + 54
        nions = 3
    if iz == 6:
        n = 354; nions = 6
    elif iz == 7:
        n = 360; nions = 7
    elif iz == 8:
        n = 367; nions = 8
    if 20 <= iz < 29:
        nions = 10
    n_start = n - 1
    nion2 = min(nion + 2, nions)                   # normalise over a slightly longer ladder

    IP = np.zeros(31); PART = np.ones(31); POTLO = np.zeros(31); F = np.zeros(31)

    for ion in range(1, nion2 + 1):
        zion = float(ion)
        POTLO[ion - 1] = POTLOW * zion             # lowering scales with the ion charge

        # ionization potential from the POTION table [cm^-1] -> eV
        pidx = potion_index(iz, ion) - 1
        if 0 <= pidx < len(POTION):
            IP[ion - 1] = POTION[pidx] / EV_TO_CM
            if IP[ion - 1] == 0.0 and pidx > 0:
                IP[ion - 1] = POTION[pidx - 1] / EV_TO_CM

        # ---- Fe-group elements read their partition function from PFIRON ----
        if 20 <= iz < 29:
            tlog8 = tlog / 2.30258509299405        # ln T -> log10 T
            PART[ion - 1] = pfiron(iz, ion, tlog8, POTLO[ion - 1] * EV_TO_CM,
                                   PFTAB, PFIRON_POTLO, PFIRON_POTLOLOG)
            continue

        col = n_start + ion                        # 1-based packed-table column
        c0 = col - 1
        G = 0.0; D1 = 0.0
        if c0 < NNN.shape[1]:
            nnn100 = NNN[5, c0] // 100
            G = float(NNN[5, c0] - nnn100 * 100)   # statistical weight for the high-T term
        handled = _special_partition(col, ion, T, hckt, PART, sp)
        if handled is not None:
            # special blocks set D1, and a few override G; G=None keeps the NNN value
            g_override, D1 = handled
            if g_override is not None:
                G = g_override

        # ---- standard packed-NNN partition function (non-special elements) ----
        if (handled is None and c0 < NNN.shape[1] and IP[ion - 1] > 0.0):
            T2000 = IP[ion - 1] * 2000.0 / 11.0
            IT = max(1, min(9, int(T / T2000 - 0.5)))
            DT = T / T2000 - float(IT) - 0.5
            PMIN = 1.0
            i_idx = max(0, min(NNN.shape[0] - 1, (IT + 1) // 2 - 1))
            nnn_i = int(NNN[i_idx, c0])
            K1 = nnn_i // 100000
            K2 = nnn_i - K1 * 100000
            K3 = K2 // 10
            KSCALE = K2 - K3 * 10
            s = max(0, min(len(SCALE) - 1, KSCALE - 1))
            if IT % 2 == 1:                        # odd bracket: both ends from this row
                P1 = float(K1) * SCALE[s]
                P2 = float(K3) * SCALE[s]
                if DT < 0.0 and s <= 0:
                    if P1 == float(int(P2 + 0.5)):
                        PMIN = P1
            else:                                  # even bracket: far end from the next row
                P1 = float(K3) * SCALE[s]
                nnn_i1 = int(NNN[min(i_idx + 1, NNN.shape[0] - 1), c0])
                sN = max(0, min(len(SCALE) - 1, (nnn_i1 % 10) - 1))
                P2 = float(nnn_i1 // 100000) * SCALE[sN]
            PART[ion - 1] = max(PMIN, P1 + (P2 - P1) * DT)
        elif handled is None:
            PART[ion - 1] = 1.0

        # ---- low-T ground-state correction (PFGROUND), pre-tabulated ----
        skip_high_t = False
        if IP[ion - 1] > 0.0:
            T2000 = IP[ion - 1] * 2000.0 / 11.0
            if T < T2000 * 2.0:
                nelion = (iz - 1) * 6 + ion
                pfg = PFG[nelion, layer] if nelion < PFG.shape[0] else 1.0
                if pfg > 0.0:
                    PART[ion - 1] = max(PART[ion - 1], pfg)
                skip_high_t = True                 # PFGROUND replaces the high-T term

        # ---- high-T occupation-probability correction ----
        special_bypass = D1 > 0.0
        if (not skip_high_t and (G > 0.0 or D1 > 0.0)
                and (special_bypass or POTLO[ion - 1] >= 0.1) and IP[ion - 1] > 0.0):
            T2000 = IP[ion - 1] * 2000.0 / 11.0
            if special_bypass or T >= T2000 * 4.0:
                tv = tkev
                if T > T2000 * 11.0:               # cap the effective temperature
                    tv = (T2000 * 11.0) * KEV_FACTOR
                d1 = (0.1 / tv) if D1 <= 0.0 else D1
                d2 = POTLO[ion - 1] / tv
                if G > 0.0:
                    PART[ion - 1] += (G * np.exp(-IP[ion - 1] / tv)
                                      * (occupation_term(d2, zion, tv)
                                         - occupation_term(d1, zion, tv)))

    # ---- the Saha ladder: chain stage-to-stage ratios, then normalise ----
    CF = 2.0 * SAHA * T * np.sqrt(T) / xne
    F[0] = 1.0
    for ion in range(2, nion2 + 1):
        if PART[ion - 2] > 0.0:
            F[ion - 1] = (CF * PART[ion - 1] / PART[ion - 2]
                          * np.exp(-(IP[ion - 2] - POTLO[ion - 2]) / tkev))
    L = nion2 + 1
    for _ in range(2, nion2 + 1):                  # build the normalising sum, top-down
        L -= 1
        F[0] = 1.0 + F[L - 1] * F[0]
    F[0] = 1.0 / F[0]
    for ion in range(2, nion2 + 1):                # cascade the neutral fraction up the ladder
        F[ion - 1] = F[ion - 2] * F[ion - 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        popfrac = np.where(PART[:nion2] > 0.0, F[:nion2] / PART[:nion2], 0.0)
    return PART[:nion2].copy(), popfrac, nion2


# ---------------------------------------------------------------------------
#  Special-element partition blocks (H/He/C/Mg/Al/Si/O/Na/B/K/Ca).  Each block
#  sums an explicit list of measured levels (E in cm^-1, weight g) instead of
#  the packed NNN data.  Returns (G, D1) when it handles the column, else None.
# ---------------------------------------------------------------------------
def _special_partition(col, ion, T, hckt, PART, sp):
    idx = ion - 1

    def block(part0, Ekey, Gkey, nlev):
        # part0 + sum_i g_i exp(-E_i hckt) over the explicit measured levels
        E = sp["sp_" + Ekey]; g = sp["sp_" + Gkey]
        s = part0
        for i in range(1, nlev):
            s += g[i] * np.exp(-E[i] * hckt)
        return s

    if col == 2:                                   # bare ion (H II): no excited states
        PART[idx] = 1.0
        return (None, 0.0)
    if col == 1:                                   # H I
        PART[idx] = 2.0
        if T >= 9000.0:
            PART[idx] = block(2.0, "EHYD", "GHYD", 6)
        return (None, 109677.576 / 6.5 / 6.5 * hckt)
    if col == 3:                                   # He I
        PART[idx] = 1.0
        if T >= 15000.0:
            PART[idx] = block(1.0, "EHE1", "GHE1", 29)
        return (None, 109677.576 / 5.5 / 5.5 * hckt)
    if col == 4:                                   # He II
        PART[idx] = 2.0
        if T >= 30000.0:
            PART[idx] = block(2.0, "EHE2", "GHE2", 6)
        return (None, 4.0 * 109722.267 / 6.5 / 6.5 * hckt)
    if col == 354:                                 # C I
        PART[idx] = block(1.0 + 3.0 * np.exp(-16.42 * hckt) + 5.0 * np.exp(-43.42 * hckt),
                          "EC1", "GC1", 14)
        PART[idx] += (108.0 * np.exp(-80000.0 * hckt) + 189.0 * np.exp(-84000.0 * hckt)
                      + 247.0 * np.exp(-87000.0 * hckt) + 231.0 * np.exp(-88000.0 * hckt)
                      + 190.0 * np.exp(-89000.0 * hckt) + 300.0 * np.exp(-90000.0 * hckt))
        return (None, 0.0)
    if col == 355:                                 # C II
        PART[idx] = block(2.0 + 4.0 * np.exp(-63.42 * hckt), "EC2", "GC2", 6)
        PART[idx] += (6.0 * np.exp(-131731.80 * hckt) + 4.0 * np.exp(-142027.1 * hckt)
                      + 10.0 * np.exp(-145550.13 * hckt) + 10.0 * np.exp(-150463.62 * hckt)
                      + 2.0 * np.exp(-157234.07 * hckt) + 6.0 * np.exp(-162500.0 * hckt)
                      + 42.0 * np.exp(-168000.0 * hckt) + 56.0 * np.exp(-178000.0 * hckt)
                      + 102.0 * np.exp(-183000.0 * hckt) + 400.0 * np.exp(-188000.0 * hckt))
        return (None, 0.0)
    if col == 51:                                  # Mg I
        PART[idx] = block(1.0, "EMG1", "GMG1", 11)
        PART[idx] += (5.0 * np.exp(-53134.0 * hckt) + 15.0 * np.exp(-54192.0 * hckt)
                      + 28.0 * np.exp(-54676.0 * hckt) + 9.0 * np.exp(-57853.0 * hckt))
        return (4.0, 109734.83 / 4.5 / 4.5 * hckt)
    if col == 52:                                  # Mg II
        PART[idx] = block(2.0, "EMG2", "GMG2", 6)
        PART[idx] += (10.0 * np.exp(-93310.80 * hckt) + 14.0 * np.exp(-93799.70 * hckt)
                      + 6.0 * np.exp(-97464.32 * hckt) + 10.0 * np.exp(-103419.82 * hckt)
                      + 14.0 * np.exp(-103689.89 * hckt) + 18.0 * np.exp(-103705.66 * hckt))
        return (2.0, 4.0 * 109734.83 / 5.5 / 5.5 * hckt)
    if col == 57:                                  # Al I
        PART[idx] = block(2.0 + 4.0 * np.exp(-112.061 * hckt), "EAL1", "GAL1", 9)
        PART[idx] += 10.0 * np.exp(-42235.0 * hckt) + 14.0 * np.exp(-43831.0 * hckt)
        return (2.0, 109735.08 / 5.5 / 5.5 * hckt)
    if col == 63:                                  # Si I
        PART[idx] = block(1.0 + 3.0 * np.exp(-77.115 * hckt) + 5.0 * np.exp(-223.157 * hckt),
                          "ESI1", "GSI1", 11)
        PART[idx] += (76.0 * np.exp(-53000.0 * hckt) + 71.0 * np.exp(-57000.0 * hckt)
                      + 191.0 * np.exp(-60000.0 * hckt) + 240.0 * np.exp(-62000.0 * hckt)
                      + 251.0 * np.exp(-63000.0 * hckt) + 300.0 * np.exp(-65000.0 * hckt))
        return (None, 0.0)
    if col == 64:                                  # Si II
        PART[idx] = block(2.0 + 4.0 * np.exp(-287.32 * hckt), "ESI2", "GSI2", 6)
        PART[idx] += (6.0 * np.exp(-81231.59 * hckt) + 6.0 * np.exp(-83937.08 * hckt)
                      + 10.0 * np.exp(-101024.09 * hckt) + 14.0 * np.exp(-103556.35 * hckt)
                      + 10.0 * np.exp(-108800.0 * hckt) + 42.0 * np.exp(-115000.0 * hckt)
                      + 6.0 * np.exp(-121000.0 * hckt) + 38.0 * np.exp(-125000.0 * hckt)
                      + 34.0 * np.exp(-132000.0 * hckt))
        return (2.0, 4.0 * 109734.83 / 4.5 / 4.5 * hckt)
    if col == 96:                                  # Ca I
        PART[idx] = block(1.0, "ECA1", "GCA1", 8)
        PART[idx] += (28.0 * np.exp(-37000.0 * hckt) + 67.0 * np.exp(-40000.0 * hckt)
                      + 21.0 * np.exp(-43000.0 * hckt) + 34.0 * np.exp(-48000.0 * hckt))
        return (4.0, 109734.82 / 4.5 / 4.5 * hckt)
    if col == 97:                                  # Ca II
        PART[idx] = block(2.0, "ECA2", "GCA2", 5)
        PART[idx] += 12.0 * np.exp(-68000.0 * hckt)
        return (2.0, 109734.83 / 4.5 / 4.5 * hckt)
    if col == 367:                                 # O I
        PART[idx] = block(5.0 + 3.0 * np.exp(-158.265 * hckt) + np.exp(-226.977 * hckt),
                          "EO1", "GO1", 13)
        PART[idx] += (15.0 * np.exp(-101140.0 * hckt) + 131.0 * np.exp(-103000.0 * hckt)
                      + 128.0 * np.exp(-105000.0 * hckt) + 600.0 * np.exp(-107000.0 * hckt))
        return (None, 0.0)
    if col == 45:                                  # Na I
        PART[idx] = block(2.0, "ENA1", "GNA1", 8)
        PART[idx] += 10.0 * np.exp(-34548.745 * hckt) + 14.0 * np.exp(-34586.96 * hckt)
        return (2.0, 109734.83 / 4.5 / 4.5 * hckt)
    if col == 14:                                  # B I
        PART[idx] = block(2.0 + 4.0 * np.exp(-15.25 * hckt), "EB1", "GB1", 7)
        PART[idx] += (6.0 * np.exp(-57786.80 * hckt) + 10.0 * np.exp(-59989.0 * hckt)
                      + 14.0 * np.exp(-60031.03 * hckt) + 2.0 * np.exp(-63561.0 * hckt))
        return (2.0, 109734.83 / 4.5 / 4.5 * hckt)
    if col == 91:                                  # K I
        PART[idx] = block(2.0, "EK1", "GK1", 8)
        PART[idx] += 10.0 * np.exp(-27397.077 * hckt) + 14.0 * np.exp(-28127.85 * hckt)
        return (2.0, 109734.83 / 5.5 / 5.5 * hckt)
    return None


# ===========================================================================
#  Drive PFSAHA over all elements / ions / depths -> U and population_per_ion.
# ===========================================================================
def run_pfsaha(inp, tab):
    T = inp["T"]; tkev = inp["tkev"]; tk = inp["tk"]; hckt = inp["hckt"]
    tlog = inp["tlog"]; P = inp["gas_pressure"]; xne = inp["electron_density"]
    xnatom = inp["xnatom"]; xabund = inp["xabund"]; nion_per_Z = inp["nion_per_Z"]
    nlay = T.size
    NSTORE = 6
    U = np.zeros((nlay, 99, NSTORE))
    popfrac = np.zeros((nlay, 99, NSTORE))
    pop = np.zeros((nlay, 99, NSTORE))
    for z in range(1, 100):
        nion = int(nion_per_Z[z - 1])
        for j in range(nlay):
            PART, fr, nion2 = pfsaha_layer(
                z, nion, float(T[j]), float(tkev[j]), float(tk[j]),
                float(hckt[j]), float(tlog[j]), float(P[j]), float(xne[j]),
                tab, j)
            ns = min(NSTORE, nion2)
            U[j, z - 1, :ns] = PART[:ns]
            popfrac[j, z - 1, :ns] = fr[:ns]
            pop[j, z - 1, :ns] = fr[:ns] * xnatom[j] * xabund[z - 1]
    return U, popfrac, pop


# ===========================================================================
#  NELECT: the electron-density charge balance solved from scratch.  Iterate
#  n_e from a half-ionized seed, summing the charge donated by ten key electron
#  donors (H, He, C, Na, Mg, Al, Si, K, Ca, Fe) through the Saha fractions.
# ===========================================================================
def electron_charge(iz, nion, T, tkev, tk, hckt, tlog, P, xne, tab, layer):
    """Electrons donated per atom of element iz = sum_i (i) * f_i (mode-4 sum)."""
    _, fr, nion2 = pfsaha_layer(iz, nion, T, tkev, tk, hckt, tlog, P, xne, tab, layer)
    # rebuild the un-normalised F = fraction (popfrac already F/PART so re-form F):
    # mode 4 sums F[ion]*(ion-1); we have popfrac=F/PART, but the electron sum
    # uses the NORMALISED F directly, so recompute F from the ladder once more.
    PART, _, _ = pfsaha_layer(iz, nion, T, tkev, tk, hckt, tlog, P, xne, tab, layer)
    F = fr * PART                                   # F = (F/PART) * PART
    return float(np.sum(F[1:nion2] * np.arange(1, nion2)))


def run_nelect(inp, tab, max_iter=200, tol=5e-4):
    T = inp["T"]; tkev = inp["tkev"]; tk = inp["tk"]; hckt = inp["hckt"]
    tlog = inp["tlog"]; P = inp["gas_pressure"]; xabund = inp["xabund"]
    nlay = T.size
    NELEMZ = [1, 2, 6, 11, 12, 13, 14, 19, 20, 26]   # the ten electron donors
    NIONZ = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    NZ = len(NELEMZ)
    xne = np.zeros(nlay)
    xne[0] = P[0] / tk[0] / 2.0                       # seed: half the particles are electrons
    for j in range(nlay):
        if j > 0:
            xne[j] = xne[j - 1] * P[j] / P[j - 1]     # extrapolate the seed across depth
        xntot = P[j] / tk[j]
        xnatom = xntot - xne[j]
        mask = [1] * NZ
        for _ in range(max_iter):
            xnenew = 0.0
            xcontrib = [0.0] * NZ
            for i in range(NZ):
                if mask[i] == 0:
                    continue
                ec = electron_charge(NELEMZ[i], NIONZ[i], float(T[j]), float(tkev[j]),
                                     float(tk[j]), float(hckt[j]), float(tlog[j]),
                                     float(P[j]), float(xne[j]), tab, j)
                xcontrib[i] = ec * xnatom * xabund[NELEMZ[i] - 1]
                xnenew += xcontrib[i]
            xnenew = (xnenew + xne[j]) / 2.0          # damp the fixed-point iteration
            err = abs((xne[j] - xnenew) / xnenew) if xnenew > 0 else abs(xne[j])
            xne[j] = xnenew
            xnatom = xntot - xne[j]
            if err < tol:
                break
            if j > 0:                                 # drop negligible donors to match NELECT
                x1 = 1e-5 * xne[j] * (10.0 if err < 0.05 else 1.0)
                for i in range(NZ):
                    if xcontrib[i] < x1:
                        mask[i] = 0
    return xne


# ===========================================================================
#  Reporting
# ===========================================================================
def report(name, got, ref):
    got = np.asarray(got, np.float64); ref = np.asarray(ref, np.float64)
    m = np.abs(ref) > 0
    rel = np.zeros_like(ref)
    rel[m] = np.abs(got[m] - ref[m]) / np.abs(ref[m])
    # tag: exact to the bit, or within the float64 round-off floor (a few ULP)
    if np.array_equal(got[m], ref[m]):
        tag = "bit-exact"
    elif rel.max() < 1e-14:
        tag = "exact (float floor)"
    else:
        tag = "CHECK"
    print(f"  {name:34s}  n={int(m.sum()):6d}  max|rel|={rel.max():.3e}  "
          f"median|rel|={np.median(rel[m]):.3e}  [{tag}]")
    return rel.max()


def main() -> int:
    inp = dict(np.load(REF / "pfsaha_inputs.npz"))
    truth = dict(np.load(REF / "pfsaha_truth.npz"))
    # the atomic-data tables (everything that is not depth STATE)
    tab = {k: inp[k] for k in (
        "NNN", "POTION", "PFTAB", "PFIRON_POTLO", "PFIRON_POTLOLOG",
        "LOCZ", "SCALE", "pfground_tab")}
    for k in inp:
        if k.startswith("sp_"):
            tab[k] = inp[k]
    nlay = inp["T"].size

    print("=" * 78)
    print("PFSAHA ionization core from scratch: per-ion U + per-ion populations")
    print(f"  n_layers={nlay}  elements=99  ion stages stored=6")
    print("  inputs npz carries NO answers (only depth state + atomic-data tables);")
    print("  the per-ion U, populations and n_e below are all COMPUTED from scratch.")
    print("=" * 78)

    print("\n-- per-ion partition functions U and per-ion populations (PFSAHA) --")
    U, popfrac, pop = run_pfsaha(inp, tab)
    r = 0.0
    r = max(r, report("partition function U", U, truth["U"]))
    r = max(r, report("population fraction F/PART", popfrac, truth["population_fraction"]))
    r = max(r, report("population_per_ion = n_ion/U", pop, truth["population_per_ion"]))

    print("\n-- electron density n_e (NELECT charge balance, from scratch) --")
    xne = run_nelect(inp, tab)
    r = max(r, report("electron density n_e", xne, truth["xne_nelect"]))

    # ---------------- tamper check: corrupt an input, answers must move -------
    print("\n-- tamper check: perturb the metal abundances, populations must move --")
    bad = dict(inp)
    bad_xab = inp["xabund"].copy()
    bad_xab[10] *= 1.1                                 # bump sodium (Z=11) by 10%
    bad["xabund"] = bad_xab
    _, _, pop_bad = run_pfsaha(bad, tab)
    # Na's own populations scale ~linearly with its abundance; and because Na is
    # a photospheric electron donor, re-solving n_e shifts every species too.
    na_shift = np.abs(pop_bad[:, 10, 0] - pop[:, 10, 0]) / np.abs(pop[:, 10, 0])
    xne_bad = run_nelect(bad, tab)
    ne_shift = np.abs(xne_bad - xne) / np.abs(xne)
    print(f"  Na I population shift (abundance*1.1) : max|rel| = {na_shift.max():.3e}")
    print(f"  n_e shift from re-solved charge balance: max|rel| = {ne_shift.max():.3e}")
    tamper_ok = na_shift.max() > 1e-3 and ne_shift.max() > 1e-4

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    floor = 1e-9
    ok = (r < floor) and tamper_ok
    print(f"  PFSAHA core + n_e : max|rel| = {r:.3e}   (target < {floor:.0e})")
    print(f"  tamper check      : {'OK' if tamper_ok else 'FAILED'}")
    print()
    print("  PASS" if ok else "  FAIL")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
