#!/usr/bin/env python
"""Pure-NumPy multi-element equation of state (POPSALL / NELECT), from scratch.

Stage 1 of the from-scratch line-blanketed atmosphere (PLAN_lineblanket_fromscratch.md).
This REPLACES the pyk-subprocess EOS borrow (numpy_oracle.oracle_state) with the book's
OWN numpy: the partition functions + Saha ladder (the Lecture-2 physics), the charge-balance
electron density, and the flat multi-element population slots the line deposit consumes.

It is a faithful pure-NumPy transcription of the verified self-contained kgpu EOS
(/Users/ysting/pykurucz_gpu/kgpu/eos.py — numpy/torch only, ZERO pykurucz), which is itself
the textbook Lecture-2 PFSAHA physics vectorised over depth.  kgpu is the READ-ONLY template;
nothing here imports kgpu or pykurucz.  The only data loaded is the honest atomic-data table
reference/pfsaha_inputs.npz (partition levels, ionization potentials, Fe-group grids) — the same
table pyk reads, already shipped with the book (Lecture 2).

Verified bit-exact vs the kgpu native EOS at the Sun's structure (see verify_eos_fromscratch.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── physical constants (Kurucz/pykurucz literals, exact) ──────────────────────
EV_TO_CM = 8065.479          # 1 eV in cm^-1
KEV_FACTOR = 8.6171e-5       # K -> eV   (kT[eV] = KEV*T)
SAHA = 2.4148e15             # (2 pi m_e k / h^2)^{3/2}  [cm^-3 K^-3/2]
_LN10 = 2.30258509299405
K_BOLTZ = 1.38054e-16        # erg / K   (TK = 1.38054e-16 * T)
H_PLANCK = 6.6256e-27        # erg s
C_LIGHT = 2.99792458e10      # cm / s
AMU = 1.660e-24              # g  (atomic mass unit, Fortran constant)
_FLOOR = 1e-300

_DEFAULT_INPUTS = Path(__file__).resolve().parent.parent / "reference" / "pfsaha_inputs.npz"


# ── per-Z ion-stage counts NELECT/POPSALL track ───────────────────────────────
def _nion_for_atomic_number(z: int) -> int:
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
    return 3   # z in (29, 30) or z >= 31


NION_PER_Z = np.array([_nion_for_atomic_number(z) for z in range(1, 100)], dtype=np.int64)
_ION_CHARGE = np.arange(6, dtype=np.float64)   # charge of ion index 0..5


def atomic_slot_start(z: int) -> int:
    """0-based flat-slot start for atomic number z (the POPSALL/LINOP nelion layout).

    Z<=30 packed triangular; Z>=31 starts at slot 496 (1-based) with stride 5.
    """
    if z <= 30:
        return (1 + ((z - 1) * (z + 2)) // 2) - 1
    return (496 + (z - 31) * 5) - 1


# ── atomic-data tables (the honest input) ─────────────────────────────────────
@dataclass
class EOSTables:
    """PFSAHA atomic-data tables, loaded once from reference/pfsaha_inputs.npz."""

    NNN_i: np.ndarray            # (6, 374) int64 packed partition data
    POTION: np.ndarray           # (999,) ionization potentials [cm^-1]
    PFTAB: np.ndarray            # (7, 56, 10, 9) Fe-group grid
    PFIRON_POTLO: np.ndarray     # (7,)
    PFIRON_POTLOLOG: np.ndarray  # (7,)
    LOCZ: np.ndarray             # (29,)
    SCALE: np.ndarray            # (4,)
    pfground_tab: np.ndarray     # (605, 80) ground-state correction, pre-evaluated on the L2 T grid
    sp: dict                     # special light-element level blocks {"sp_EHYD": ..., ...}

    @classmethod
    def from_npz(cls, path: Path = _DEFAULT_INPUTS) -> "EOSTables":
        d = np.load(path, allow_pickle=False)
        sp = {k: np.asarray(d[k], np.float64) for k in d.files if k.startswith("sp_")}
        return cls(
            NNN_i=np.asarray(d["NNN"]).astype(np.int64),
            POTION=np.asarray(d["POTION"], np.float64),
            PFTAB=np.asarray(d["PFTAB"], np.float64),
            PFIRON_POTLO=np.asarray(d["PFIRON_POTLO"], np.float64),
            PFIRON_POTLOLOG=np.asarray(d["PFIRON_POTLOLOG"], np.float64),
            LOCZ=np.asarray(d["LOCZ"]).astype(np.int64),
            SCALE=np.asarray(d["SCALE"], np.float64),
            pfground_tab=np.asarray(d["pfground_tab"], np.float64),
            sp=sp,
        )


# ── per-element dispatch ──────────────────────────────────────────────────────
def _potion_index_1based(iz: int, ion: int) -> int:
    if iz <= 30:
        return iz * (iz + 1) // 2 + ion - 1
    return iz * 5 + 341 + ion - 1


def _element_block(iz: int, locz: np.ndarray):
    if iz <= 28:
        n = int(locz[iz - 1])
        nions = int(locz[iz] - locz[iz - 1]) if iz < len(locz) else 3
    else:
        n = 3 * iz + 54
        nions = 3
    if iz == 6:
        n, nions = 354, 6
    elif iz == 7:
        n, nions = 360, 7
    elif iz == 8:
        n, nions = 367, 8
    if 20 <= iz < 29:
        nions = 10
    return n, nions


# ── Debye lowering + occupation correction ────────────────────────────────────
def _debye_lowering(xne, tk, Pg):
    charge = 2.0 * xne
    excess = 2.0 * xne - Pg / tk
    charge = np.where(excess > 0.0, charge + 2.0 * excess, charge)
    charge = np.where(charge == 0.0, 1.0, charge)
    debye = np.sqrt(tk / 2.8965e-18 / charge)
    return np.minimum(1.44e-7 / debye, 1.0)


def _occupation_term(d, zion, tv):
    x3 = np.sqrt(13.595 * zion * zion / tv / d) ** 3
    poly = 1.0 / 3.0 + (1.0 - (0.5 + (1.0 / 18.0 + d / 120.0) * d) * d) * d
    return x3 * poly


# ── iron-group partition grid (PFIRON) ────────────────────────────────────────
def _pfiron(tab: EOSTables, iz: int, ion: int, tlog10, potlow_cm1):
    PFTAB = tab.PFTAB
    POTLO = tab.PFIRON_POTLO
    POTLOLOG = tab.PFIRON_POTLOLOG
    elem_idx = iz - 20
    ion_idx = ion - 1
    plow = np.asarray(potlow_cm1, np.float64)
    nd = tlog10.shape[0]
    tl = tlog10

    it_hot = np.clip(((tl - 4.0) / 0.05 + 31.0).astype(np.int64), 1, 56)
    f_hot = (tl - (it_hot.astype(np.float64) - 31.0) * 0.05 - 4.0) / 0.05
    it_cool = np.maximum(((tl - 3.32) / 0.02 + 2.0).astype(np.int64), 2)
    f_cool = (tl - (it_cool.astype(np.float64) - 2.0) * 0.02 - 3.32) / 0.02
    it_mid = ((tl - 3.7) / 0.03 + 21.0).astype(np.int64)
    f_mid = (tl - (it_mid.astype(np.float64) - 21.0) * 0.03 - 3.7) / 0.03

    hot = tl > 4.0
    cool = tl < 3.7
    it = np.where(hot, it_hot, np.where(cool, it_cool, it_mid))
    f = np.where(hot, f_hot, np.where(cool, f_cool, f_mid))
    it = np.clip(it, 1, 56)
    it0 = it - 1
    itm1 = np.maximum(it0 - 1, 0)

    flat = PFTAB[:, :, ion_idx, elem_idx]   # (7, 56)
    val_weak = f * flat[0, it0] + (1.0 - f) * flat[0, itm1]

    n_lo = POTLO.shape[0]
    low = np.full(nd, n_lo - 1, dtype=np.int64)
    found = np.zeros(nd, dtype=bool)
    for i in range(1, n_lo):
        hit = (~found) & (plow < POTLO[i])
        low = np.where(hit, i, low)
        found = found | hit

    safe_pl = np.maximum(plow, 1e-30)
    p = (np.log10(safe_pl) - POTLOLOG[low - 1]) / 0.30103
    pl_it0 = flat[low, it0]
    pl_itm1 = flat[low, itm1]
    pl1_it0 = flat[low - 1, it0]
    pl1_itm1 = flat[low - 1, itm1]
    val_low = (p * (f * pl_it0 + (1.0 - f) * pl_itm1)
               + (1.0 - p) * (f * pl1_it0 + (1.0 - f) * pl1_itm1))
    return np.where(plow < POTLO[0], val_weak, val_low)


# ── ordinary-ion NNN unpacking ────────────────────────────────────────────────
def _nnn_partition(tab: EOSTables, c0, ip_val, T):
    NNN_i = tab.NNN_i
    nrows = NNN_i.shape[0]
    scale = tab.SCALE
    T2000 = ip_val * 2000.0 / 11.0
    IT = np.clip((T / T2000 - 0.5).astype(np.int64), 1, 9)
    DT = T / T2000 - IT.astype(np.float64) - 0.5
    i_idx = np.clip((IT + 1) // 2 - 1, 0, nrows - 1)

    nnn_i = NNN_i[i_idx, c0]
    K1 = nnn_i // 100000
    K2 = nnn_i - K1 * 100000
    K3 = K2 // 10
    KSCALE = K2 - K3 * 10
    s = np.clip(KSCALE - 1, 0, scale.shape[0] - 1)

    P1_odd = K1.astype(np.float64) * scale[s]
    P2_odd = K3.astype(np.float64) * scale[s]

    i_idx_next = np.clip(i_idx + 1, 0, nrows - 1)
    nnn_i1 = NNN_i[i_idx_next, c0]
    sN = np.clip((nnn_i1 % 10) - 1, 0, scale.shape[0] - 1)
    P1_even = K3.astype(np.float64) * scale[s]
    P2_even = (nnn_i1 // 100000).astype(np.float64) * scale[sN]

    odd = (IT % 2) == 1
    P1 = np.where(odd, P1_odd, P1_even)
    P2 = np.where(odd, P2_odd, P2_even)

    pmin_cond = odd & (DT < 0.0) & (s <= 0) & (P1 == np.floor(P2 + 0.5))
    PMIN = np.where(pmin_cond, P1, 1.0)
    return np.maximum(PMIN, P1 + (P2 - P1) * DT)


# ── special light-element partition sums ──────────────────────────────────────
def _block(tab, Ekey, Gkey, nlev, hckt):
    E = tab.sp["sp_" + Ekey]; g = tab.sp["sp_" + Gkey]
    s = np.zeros_like(hckt)
    for i in range(1, nlev):
        s = s + g[i] * np.exp(-E[i] * hckt)
    return s


def _block_p0(tab, part0, Ekey, Gkey, nlev, hckt):
    E = tab.sp["sp_" + Ekey]; g = tab.sp["sp_" + Gkey]
    s = np.array(part0, dtype=np.float64, copy=True)
    for i in range(1, nlev):
        s = s + g[i] * np.exp(-E[i] * hckt)
    return s


def _special_partition(tab, col, ion, T, hckt):
    """Returns (PART, g_override, D1) or None.  PART/D1 are (nd,) arrays."""
    z = np.zeros_like(hckt)
    if col == 2:
        return np.ones_like(hckt), None, z
    if col == 1:   # H I
        part = np.where(T >= 9000.0, _block(tab, "EHYD", "GHYD", 6, hckt) + 2.0, 2.0)
        return part, None, 109677.576 / 6.5 / 6.5 * hckt
    if col == 3:   # He I
        part = np.where(T >= 15000.0, _block(tab, "EHE1", "GHE1", 29, hckt) + 1.0, 1.0)
        return part, None, 109677.576 / 5.5 / 5.5 * hckt
    if col == 4:   # He II
        part = np.where(T >= 30000.0, _block(tab, "EHE2", "GHE2", 6, hckt) + 2.0, 2.0)
        return part, None, 4.0 * 109722.267 / 6.5 / 6.5 * hckt
    if col == 354:  # C I
        part0 = 1.0 + 3.0 * np.exp(-16.42 * hckt) + 5.0 * np.exp(-43.42 * hckt)
        part = _block_p0(tab, part0, "EC1", "GC1", 14, hckt)
        part = part + (108.0 * np.exp(-80000.0 * hckt) + 189.0 * np.exp(-84000.0 * hckt)
                       + 247.0 * np.exp(-87000.0 * hckt) + 231.0 * np.exp(-88000.0 * hckt)
                       + 190.0 * np.exp(-89000.0 * hckt) + 300.0 * np.exp(-90000.0 * hckt))
        return part, None, z
    if col == 355:  # C II
        part0 = 2.0 + 4.0 * np.exp(-63.42 * hckt)
        part = _block_p0(tab, part0, "EC2", "GC2", 6, hckt)
        part = part + (6.0 * np.exp(-131731.80 * hckt) + 4.0 * np.exp(-142027.1 * hckt)
                       + 10.0 * np.exp(-145550.13 * hckt) + 10.0 * np.exp(-150463.62 * hckt)
                       + 2.0 * np.exp(-157234.07 * hckt) + 6.0 * np.exp(-162500.0 * hckt)
                       + 42.0 * np.exp(-168000.0 * hckt) + 56.0 * np.exp(-178000.0 * hckt)
                       + 102.0 * np.exp(-183000.0 * hckt) + 400.0 * np.exp(-188000.0 * hckt))
        return part, None, z
    if col == 51:   # Mg I
        part = _block(tab, "EMG1", "GMG1", 11, hckt) + 1.0
        part = part + (5.0 * np.exp(-53134.0 * hckt) + 15.0 * np.exp(-54192.0 * hckt)
                       + 28.0 * np.exp(-54676.0 * hckt) + 9.0 * np.exp(-57853.0 * hckt))
        return part, 4.0, 109734.83 / 4.5 / 4.5 * hckt
    if col == 52:   # Mg II
        part = _block(tab, "EMG2", "GMG2", 6, hckt) + 2.0
        part = part + (10.0 * np.exp(-93310.80 * hckt) + 14.0 * np.exp(-93799.70 * hckt)
                       + 6.0 * np.exp(-97464.32 * hckt) + 10.0 * np.exp(-103419.82 * hckt)
                       + 14.0 * np.exp(-103689.89 * hckt) + 18.0 * np.exp(-103705.66 * hckt))
        return part, 2.0, 4.0 * 109734.83 / 5.5 / 5.5 * hckt
    if col == 57:   # Al I
        part0 = 2.0 + 4.0 * np.exp(-112.061 * hckt)
        part = _block_p0(tab, part0, "EAL1", "GAL1", 9, hckt)
        part = part + 10.0 * np.exp(-42235.0 * hckt) + 14.0 * np.exp(-43831.0 * hckt)
        return part, 2.0, 109735.08 / 5.5 / 5.5 * hckt
    if col == 63:   # Si I
        part0 = 1.0 + 3.0 * np.exp(-77.115 * hckt) + 5.0 * np.exp(-223.157 * hckt)
        part = _block_p0(tab, part0, "ESI1", "GSI1", 11, hckt)
        part = part + (76.0 * np.exp(-53000.0 * hckt) + 71.0 * np.exp(-57000.0 * hckt)
                       + 191.0 * np.exp(-60000.0 * hckt) + 240.0 * np.exp(-62000.0 * hckt)
                       + 251.0 * np.exp(-63000.0 * hckt) + 300.0 * np.exp(-65000.0 * hckt))
        return part, None, z
    if col == 64:   # Si II
        part0 = 2.0 + 4.0 * np.exp(-287.32 * hckt)
        part = _block_p0(tab, part0, "ESI2", "GSI2", 6, hckt)
        part = part + (6.0 * np.exp(-81231.59 * hckt) + 6.0 * np.exp(-83937.08 * hckt)
                       + 10.0 * np.exp(-101024.09 * hckt) + 14.0 * np.exp(-103556.35 * hckt)
                       + 10.0 * np.exp(-108800.0 * hckt) + 42.0 * np.exp(-115000.0 * hckt)
                       + 6.0 * np.exp(-121000.0 * hckt) + 38.0 * np.exp(-125000.0 * hckt)
                       + 34.0 * np.exp(-132000.0 * hckt))
        return part, 2.0, 4.0 * 109734.83 / 4.5 / 4.5 * hckt
    if col == 96:   # Ca I
        part = _block(tab, "ECA1", "GCA1", 8, hckt) + 1.0
        part = part + (28.0 * np.exp(-37000.0 * hckt) + 67.0 * np.exp(-40000.0 * hckt)
                       + 21.0 * np.exp(-43000.0 * hckt) + 34.0 * np.exp(-48000.0 * hckt))
        return part, 4.0, 109734.82 / 4.5 / 4.5 * hckt
    if col == 97:   # Ca II
        part = _block(tab, "ECA2", "GCA2", 5, hckt) + 2.0
        part = part + 12.0 * np.exp(-68000.0 * hckt)
        return part, 2.0, 109734.83 / 4.5 / 4.5 * hckt
    if col == 367:  # O I
        part0 = 5.0 + 3.0 * np.exp(-158.265 * hckt) + np.exp(-226.977 * hckt)
        part = _block_p0(tab, part0, "EO1", "GO1", 13, hckt)
        part = part + (15.0 * np.exp(-101140.0 * hckt) + 131.0 * np.exp(-103000.0 * hckt)
                       + 128.0 * np.exp(-105000.0 * hckt) + 600.0 * np.exp(-107000.0 * hckt))
        return part, None, z
    if col == 45:   # Na I
        part = _block(tab, "ENA1", "GNA1", 8, hckt) + 2.0
        part = part + 10.0 * np.exp(-34548.745 * hckt) + 14.0 * np.exp(-34586.96 * hckt)
        return part, 2.0, 109734.83 / 4.5 / 4.5 * hckt
    if col == 14:   # B I
        part0 = 2.0 + 4.0 * np.exp(-15.25 * hckt)
        part = _block_p0(tab, part0, "EB1", "GB1", 7, hckt)
        part = part + (6.0 * np.exp(-57786.80 * hckt) + 10.0 * np.exp(-59989.0 * hckt)
                       + 14.0 * np.exp(-60031.03 * hckt) + 2.0 * np.exp(-63561.0 * hckt))
        return part, 2.0, 109734.83 / 4.5 / 4.5 * hckt
    if col == 91:   # K I
        part = _block(tab, "EK1", "GK1", 8, hckt) + 2.0
        part = part + 10.0 * np.exp(-27397.077 * hckt) + 14.0 * np.exp(-28127.85 * hckt)
        return part, 2.0, 109734.83 / 5.5 / 5.5 * hckt
    return None


# ── the per-(Z,ion) partition builder ─────────────────────────────────────────
def _build_part_for_element(tab, iz, nion_track, st):
    T = st["T"]; tkev = st["tkev"]; hckt = st["hckt"]
    tlog = st["tlog"]; potlow = st["potlow"]
    pfg = tab.pfground_tab
    nd = T.shape[0]

    n_start_1, nions = _element_block(iz, tab.LOCZ)
    n_start = n_start_1 - 1
    nion2 = min(nion_track + 2, nions)

    PART = np.ones((nion2, nd), dtype=np.float64)
    IP = np.zeros((nion2, nd), dtype=np.float64)
    POTLO = np.zeros((nion2, nd), dtype=np.float64)

    for ion in range(1, nion2 + 1):
        zion = float(ion)
        k = ion - 1
        POTLO[k] = potlow * zion

        pidx = _potion_index_1based(iz, ion) - 1
        ip_val = 0.0
        if 0 <= pidx < tab.POTION.size:
            ip_val = tab.POTION[pidx] / EV_TO_CM
            if ip_val == 0.0 and pidx > 0:
                ip_val = tab.POTION[pidx - 1] / EV_TO_CM
        IP[k] = ip_val

        if 20 <= iz < 29:
            PART[k] = _pfiron(tab, iz, ion, tlog / _LN10, POTLO[k] * EV_TO_CM)
            continue

        col = n_start + ion
        c0 = col - 1
        G = 0.0
        D1 = np.zeros(nd, dtype=np.float64)
        if c0 < tab.NNN_i.shape[1]:
            v = int(tab.NNN_i[5, c0])
            G = float(v - (v // 100) * 100)

        handled = _special_partition(tab, col, ion, T, hckt)
        if handled is not None:
            part_sp, g_override, D1 = handled
            PART[k] = part_sp
            if g_override is not None:
                G = float(g_override)
        elif c0 < tab.NNN_i.shape[1] and ip_val > 0.0:
            PART[k] = _nnn_partition(tab, c0, ip_val, T)
        else:
            PART[k] = np.ones(nd, dtype=np.float64)

        if ip_val > 0.0:
            T2000 = ip_val * 2000.0 / 11.0
            low_t_mask = T < (T2000 * 2.0)
            nelion = (iz - 1) * 6 + ion
            if nelion < pfg.shape[0]:
                pfg_row = pfg[nelion]
                corrected = np.maximum(PART[k], pfg_row)
                apply_lowt = low_t_mask & (pfg_row > 0.0)
                PART[k] = np.where(apply_lowt, corrected, PART[k])
            skip_high_t = low_t_mask

            special_bypass = bool((D1 > 0.0).any())
            D1_pos = D1 > 0.0
            if G > 0.0 or special_bypass:
                gate_T = D1_pos | (T >= (T2000 * 4.0))
                gate_pot = D1_pos | ((potlow * float(ion)) >= 0.1)
                gate = gate_pot & gate_T & (~skip_high_t)
                tcap = T > (T2000 * 11.0)
                tv = np.where(tcap, (T2000 * 11.0) * KEV_FACTOR, tkev)
                d1c = np.where(D1 <= 0.0, 0.1 / tv, D1)
                d2c = POTLO[k] / tv
                if G > 0.0:
                    add = (G * np.exp(-ip_val / tv)
                           * (_occupation_term(d2c, zion, tv) - _occupation_term(d1c, zion, tv)))
                    PART[k] = np.where(gate, PART[k] + add, PART[k])

    return PART, IP, POTLO, nion2


def _saha_ladder(PART, IP, POTLO, T, tkev, xne):
    """Saha ladder in LOG space (overflow-free).  Returns popfrac (nion2, nd) = F/PART."""
    nion2, nd = PART.shape
    log_cf = math.log(2.0 * SAHA) + 1.5 * np.log(T) - np.log(xne)
    logF = np.zeros((nion2, nd), dtype=np.float64)
    eps = np.finfo(np.float64).tiny
    for ion in range(2, nion2 + 1):
        k = ion - 1
        logratio = (log_cf + np.log(np.clip(PART[k], eps, None))
                    - np.log(np.clip(PART[k - 1], eps, None))
                    - (IP[k - 1] - POTLO[k - 1]) / tkev)
        logF[k] = np.where(PART[k - 1] > 0.0, logratio, -np.inf)
    logout = np.cumsum(logF, axis=0)
    m = np.amax(logout, axis=0, keepdims=True)
    w = np.exp(logout - m)
    out = w / np.sum(w, axis=0, keepdims=True)
    return np.where(PART > 0.0, out / PART, 0.0)


# ── public API ────────────────────────────────────────────────────────────────
def derived_state(T, P, xne, *, tab: EOSTables) -> dict:
    """Depth-state dict from (T, P, n_e) — TKEV/TK/HCKT/TLOG, XNATOM = P/TK - n_e."""
    T = np.asarray(T, np.float64); P = np.asarray(P, np.float64); xne = np.asarray(xne, np.float64)
    tk = T * K_BOLTZ
    tkev = KEV_FACTOR * T
    hckt = (H_PLANCK * C_LIGHT) / np.maximum(tk, _FLOOR)
    tlog = np.log(np.maximum(T, _FLOOR))
    xnatom = np.maximum(P / np.maximum(tk, _FLOOR) - xne, _FLOOR)
    potlow = _debye_lowering(xne, tk, P)
    return dict(T=T, tkev=tkev, tk=tk, hckt=hckt, tlog=tlog, gas_pressure=P,
                electron_density=xne, xnatom=xnatom, potlow=potlow)


def populations(state: dict, tab: EOSTables, n_elem: int = 99, max_ion: int = 6):
    """PFSAHA per-ion U + populations for all elements, depth-batched.

    Returns (U, popion, popfrac), each (nd, n_elem, max_ion).
    popion is per-ion n_ion using the state's xnatom*xabund scale.
    """
    T = state["T"]; tkev = state["tkev"]; xne = state["electron_density"]
    xnatom = state["xnatom"]; xab = np.asarray(state["xabund"], np.float64)
    nd = T.shape[0]
    U = np.zeros((nd, n_elem, max_ion), np.float64)
    popion = np.zeros((nd, n_elem, max_ion), np.float64)
    popfrac = np.zeros((nd, n_elem, max_ion), np.float64)
    for Z in range(1, n_elem + 1):
        nion_track = int(NION_PER_Z[Z - 1])
        PART, IP, POTLO, nion2 = _build_part_for_element(tab, Z, nion_track, state)
        pf = _saha_ladder(PART, IP, POTLO, T, tkev, xne)   # (nion2, nd)
        ns = min(max_ion, nion2)
        U[:, Z - 1, :ns] = PART[:ns].T
        popfrac[:, Z - 1, :ns] = pf[:ns].T
        popion[:, Z - 1, :ns] = pf[:ns].T * (xnatom * xab[Z - 1])[:, None]
    return U, popion, popfrac


@dataclass
class PopsAllResult:
    xne: np.ndarray
    xnatom: np.ndarray
    rho: np.ndarray
    population_per_ion: np.ndarray   # (nd, 6, 139)
    U: np.ndarray                    # (nd, 99, 6)
    popfrac: np.ndarray              # (nd, 99, 6)
    xnf_h: np.ndarray
    xnfph: np.ndarray
    xnf_he1: np.ndarray
    xnf_he2: np.ndarray


def solve_nelect(T, P, xabund, *, tab: EOSTables, wtmole=None, xne_seed=None,
                 max_iter: int = 200, tol: float = 1e-4):
    """Self-consistent electron density via charge balance over Z=1..99, depth-batched.

    Iterate n_e from the Saha charge sum sum_Z sum_ion (ion * n_ion), with the Fortran
    max(., 0.5*xne) floor + 0.5*(new+old) damping, to the fixed point.
    """
    T = np.asarray(T, np.float64); P = np.asarray(P, np.float64)
    nd = T.size
    tk = T * K_BOLTZ
    xntot = P / np.maximum(tk, _FLOOR)
    xab = np.asarray(xabund, np.float64)
    if xab.ndim == 1:
        xab = np.broadcast_to(xab, (nd, 99))
    xne = (xntot * 0.5 if xne_seed is None else np.asarray(xne_seed, np.float64).copy())
    ion_charge = _ION_CHARGE[None, None, :]
    ones99 = np.ones(99, np.float64)

    U = popfrac = None
    for _ in range(max_iter):
        state = derived_state(T, P, xne, tab=tab)
        state["xabund"] = ones99   # Saha FRACTION is abundance-independent
        U, _popion, popfrac = populations(state, tab)
        Fpop = popfrac * U                                   # F = popfrac*U  (nd,99,6)
        xnatom = np.maximum(xntot - xne, _FLOOR)
        weighted = Fpop * (xnatom[:, None, None] * xab[:, :, None])
        xnenew = (weighted * ion_charge).sum(axis=(1, 2))
        xnenew = np.maximum(xnenew, xne * 0.5)
        xnenew = 0.5 * (xnenew + xne)
        err = np.abs((xne - xnenew) / np.maximum(xnenew, _FLOOR))
        xne = xnenew
        if np.all(err < tol):
            break
    else:
        raise RuntimeError(f"NELECT did not converge in {max_iter} iterations")

    xnatom = np.maximum(xntot - xne, _FLOOR)
    rho = xnatom.copy() if wtmole is None else xnatom * np.asarray(wtmole, np.float64) * AMU
    return xne, xnatom, rho, U, popfrac


def popsall(T, P, xabund, *, tab: EOSTables, wtmole=None, xne_seed=None,
            max_iter: int = 200, tol: float = 1e-4) -> PopsAllResult:
    """Full-slot EOS fill from (T, P, abundances) with self-consistent n_e."""
    xne, xnatom, rho, U, popfrac = solve_nelect(
        T, P, xabund, tab=tab, wtmole=wtmole, xne_seed=xne_seed, max_iter=max_iter, tol=tol)
    xab = np.asarray(xabund, np.float64)
    nd = np.asarray(T).size
    if xab.ndim == 1:
        xab = np.broadcast_to(xab, (nd, 99))
    scale = (xnatom[:, None] * xab)[:, :, None]
    popion = popfrac * scale          # mode=11 per-ion n_ion
    stage = popion * U                # mode=12 stage totals

    population_per_ion = np.zeros((nd, 6, 139), np.float64)
    population_per_ion[:, :, :99] = popion.transpose(0, 2, 1)
    return PopsAllResult(
        xne=xne, xnatom=xnatom, rho=rho,
        population_per_ion=population_per_ion, U=U, popfrac=popfrac,
        xnf_h=stage[:, 0, 0], xnfph=popion[:, 0, 0:2].copy(),
        xnf_he1=stage[:, 1, 0], xnf_he2=stage[:, 1, 1],
    )
