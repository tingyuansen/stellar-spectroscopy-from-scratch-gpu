#!/usr/bin/env python
"""From-scratch NumPy reproduction of pykurucz's MOLECULAR CONTINUOUS opacity.

Reproduces the CH (CHOP), OH (OHOP), and H2 collision-induced absorption
(H2-CIA / Borysow, Jorgensen & Zheng 1997) continuum opacity -- the dominant
continuum source in cool M dwarfs (T < 4000 K) -- to machine precision, using
ONLY numpy + the shipped coefficient/Borysow tables.

This closes the cool-star "cheat" in Lectures 12/13: the molecular continuum is
now COMPUTED from the atmosphere state + tables, not taken as a given input.

INPUTS (reference/mol_continuum_inputs.npz, all shipped):
  - atmosphere state:  T(depth), mass density, H ground-state pop xnfph1,
                       H ground departure bhyd1, He I pop xnfhe1
  - molecular equilibrium (taken as input):  CH and OH mode-1 number-density
                       populations xnfpch, xnfpoh
  - wavelength grid:   wavelength_nm / frequency_hz
  - coefficient TABLES: CH/OH cross-section + partition-function tables, and the
                       Borysow H2-H2 / H2-He CIA absorption-coefficient tables.

GROUND TRUTH (reference/mol_continuum_truth.npz, pykurucz output, COMPARISON
ONLY -- never used in the computation): per-species chop / ohop / h2cia.

The from-scratch code below COMPUTES the opacity from the inputs; it loads the
ground-truth file in a clearly separated block solely to measure precision.

  numpy / matplotlib / pathlib only.
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference"

# ── Physical constants (Fortran/pykurucz values) ─────────────────────────────
C_LIGHT_CM = 2.99792458e10      # cm/s
C_LIGHT_NM = 2.99792458e17      # nm/s
H_PLANCK = 6.62607015e-27       # erg s
K_BOLTZ = 1.380649e-16          # erg/K
KBOLTZ_EV = 8.6171e-5           # eV/K  (Fortran TKEV constant, 8.6171D-5)
LN10 = 2.30258509299405

# =============================================================================
# LOAD SHIPPED INPUTS (atmosphere state + equilibrium pops + coefficient tables)
# =============================================================================
inp = np.load(REF / "mol_continuum_inputs.npz")
temp = inp["temperature"]            # (n_layers,)  T per depth [K]
rho = inp["mass_density"]            # (n_layers,)  mass density [g/cm^3]
xnfph1 = inp["xnfph1"]               # (n_layers,)  H ground-state pop (mode-11)
bhyd1 = inp["bhyd1"]                 # (n_layers,)  H ground departure coeff (=1 LTE)
xnfhe1 = inp["xnfhe1"]               # (n_layers,)  He I number density
xnfpch = inp["xnfpch"]              # (n_layers,)  CH mode-1 population (N_CH/U_CH)
xnfpoh = inp["xnfpoh"]              # (n_layers,)  OH mode-1 population (N_OH/U_OH)
wl_nm = inp["wavelength_nm"]         # (nfreq,)
freq = inp["frequency_hz"]           # (nfreq,)  Hz

# Coefficient / Borysow tables (shipped, self-contained)
CH_PARTITION = inp["CH_PARTITION"]   # (40+,)  CH partition fn, 200K grid from 1000K
OH_PARTITION = inp["OH_PARTITION"]   # OH partition fn, same grid
CH_CROSSSECT = inp["CH_CROSSSECT"]   # (105,15) log10(xsect*U) energy x temp
OH_CROSSSECT = inp["OH_CROSSSECT"]   # (130,15) log10(xsect*U) energy x temp
H2H2 = inp["H2_COLL_H2H2"]           # (81,7)  log10 Borysow H2-H2 coeff [cm^5]
H2HE = inp["H2_COLL_H2HE"]           # (81,7)  log10 Borysow H2-He coeff

n_layers = temp.size
nfreq = freq.size

# Common derived quantities
tkev = KBOLTZ_EV * temp
tlog = np.log(temp)
stim = 1.0 - np.exp(-H_PLANCK * freq[None, :] / (K_BOLTZ * temp[:, None]))   # (L,F)
waveno = freq / C_LIGHT_CM           # cm^-1
evolt = waveno / 8065.479            # eV


# =============================================================================
# CH (CHOP) molecular continuous opacity -- COMPUTED from tables
# atlas7v.for FUNCTION CHOP; active for E in [2.0, 10.5] eV and T < 9000 K.
#   crosssect = log10(xsect*U), linear-interpolated in E (0.1 eV bins from 0.2 eV)
#               and in T (500 K bins from 2000 K).
#   partition = linear-interpolated in T (200 K bins from 1000 K).
#   opacity = exp(LN10 * log_xsect) * U  -> cross-section*U per molecule,
#             then * xnfpch / rho * stim.
# =============================================================================
def chop_xsect_times_U(freq_scalar):
    """Return CHOP cross-section * partition function per layer (cm^2)."""
    out = np.zeros(n_layers)
    wn = freq_scalar / C_LIGHT_CM
    ev = wn / 8065.479
    n = int(ev * 10)
    if n < 20 or n >= 105:
        return out
    en = n * 0.1
    idx = n - 2
    if idx < 0 or idx >= 104:
        return out
    # interpolate cross-section in energy at each of the 15 temperature nodes
    cross = CH_CROSSSECT[idx] + (CH_CROSSSECT[idx + 1] - CH_CROSSSECT[idx]) * (ev - en) / 0.1
    for j in range(n_layers):
        tj = temp[j]
        if tj >= 9000.0:
            continue
        it_p = max(0, min(int((tj - 1000.0) / 200.0), 39))
        tn_p = it_p * 200.0 + 1000.0
        part = CH_PARTITION[it_p] + (CH_PARTITION[it_p + 1] - CH_PARTITION[it_p]) * (tj - tn_p) / 200.0
        it_c = max(0, min(int((tj - 2000.0) / 500.0), 13))
        tn_c = it_c * 500.0 + 2000.0
        log_x = cross[it_c] + (cross[it_c + 1] - cross[it_c]) * (tj - tn_c) / 500.0
        out[j] = np.exp(log_x * LN10) * part
    return out


# =============================================================================
# OH (OHOP) molecular continuous opacity -- COMPUTED from tables
# atlas7v.for FUNCTION OHOP; active for E in [2.1, 15.0] eV and T < 9000 K.
#   Energy index shifted by 2.0 eV (table starts at 2.1 eV).
# =============================================================================
def ohop_xsect_times_U(freq_scalar):
    out = np.zeros(n_layers)
    wn = freq_scalar / C_LIGHT_CM
    ev = wn / 8065.479
    n = int(ev * 10) - 20
    if n <= 0 or n >= 130:
        return out
    en = n * 0.1 + 2.0
    idx = n - 1
    if idx < 0 or idx >= 129:
        return out
    cross = OH_CROSSSECT[idx] + (OH_CROSSSECT[idx + 1] - OH_CROSSSECT[idx]) * (ev - en) / 0.1
    for j in range(n_layers):
        tj = temp[j]
        if tj >= 9000.0:
            continue
        it_p = max(0, min(int((tj - 1000.0) / 200.0), 39))
        tn_p = it_p * 200.0 + 1000.0
        part = OH_PARTITION[it_p] + (OH_PARTITION[it_p + 1] - OH_PARTITION[it_p]) * (tj - tn_p) / 200.0
        it_c = max(0, min(int((tj - 2000.0) / 500.0), 13))
        tn_c = it_c * 500.0 + 2000.0
        log_x = cross[it_c] + (cross[it_c + 1] - cross[it_c]) * (tj - tn_c) / 500.0
        out[j] = np.exp(log_x * LN10) * part
    return out


# =============================================================================
# H2 collision-induced absorption (Borysow, Jorgensen & Zheng 1997) -- COMPUTED
# atlas7v.for SUBROUTINE H2COLLOP; active for waveno < 20000 cm^-1.
#   H2 number density from the molecular-equilibrium polynomial fit:
#     XNH2 = (xnfph1 * 2 * bhyd1)^2 * exp(4.478/tkev - 46.4584 + P(T) - 1.5*lnT)
#     P(T) = (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14
#             + (1.06206e-18 - 3.08720e-23 T) T) T) T) T) * T
#   Tables H2H2 / H2HE are log10(absorption coefficient cm^5); bilinear interp
#   in (waveno, T): waveno bins 250 cm^-1, T bins 1000 K (clamped IT in 1..6).
#   opacity = (10^XH2HE * n_HeI + 10^XH2H2 * XNH2) * XNH2 / rho * stim.
# =============================================================================
# XNH2 is frequency-independent -- compute once.
_poly_t = (
    1.63660e-3
    + (-4.93992e-7
       + (1.11822e-10
          + (-1.49567e-14
             + (1.06206e-18 - 3.08720e-23 * temp) * temp) * temp) * temp) * temp
) * temp
_exp_term = np.clip(4.478 / tkev - 46.4584 + _poly_t - 1.5 * tlog, -100, 100)
XNH2 = (xnfph1 * 2.0 * bhyd1) ** 2 * np.exp(_exp_term)


def h2cia_opacity(freq_scalar, stim_col):
    """Return H2-CIA opacity per layer (cm^2/g) at one frequency."""
    out = np.zeros(n_layers)
    wn = freq_scalar / C_LIGHT_CM
    if wn > 20000.0:
        return out
    nu = min(79, int(wn / 250.0))
    delnu = (wn - 250.0 * nu) / 250.0
    idx1 = min(nu, 80)
    idx2 = min(nu + 1, 80)
    # interpolate tables in wavenumber first, at each of 7 temperature nodes.
    # (matches synthe_py FUNCTION ordering: H2H2(idx1)*delnu + H2H2(idx2)*(1-delnu))
    h2h2_nu = H2H2[idx1] * delnu + H2H2[idx2] * (1.0 - delnu)
    h2he_nu = H2HE[idx1] * delnu + H2HE[idx2] * (1.0 - delnu)
    for j in range(n_layers):
        tj = temp[j]
        it = max(1, min(6, int(tj / 1000.0)))
        delt = max(0.0, min(1.0, (tj - 1000.0 * it) / 1000.0))
        xh2h2 = h2h2_nu[it - 1] * delt + h2h2_nu[it] * (1.0 - delt)
        xh2he = h2he_nu[it - 1] * delt + h2he_nu[it] * (1.0 - delt)
        out[j] = (10.0 ** xh2he * xnfhe1[j] + 10.0 ** xh2h2 * XNH2[j]) * XNH2[j] / rho[j] * stim_col[j]
    return out


# =============================================================================
# COMPUTE the molecular continuum, per species, over the full grid
# =============================================================================
chop = np.zeros((n_layers, nfreq))
ohop = np.zeros((n_layers, nfreq))
h2cia = np.zeros((n_layers, nfreq))

for j in range(nfreq):
    f = freq[j]
    stim_j = stim[:, j]
    chop[:, j] = chop_xsect_times_U(f) * xnfpch / rho * stim_j
    ohop[:, j] = ohop_xsect_times_U(f) * xnfpoh / rho * stim_j
    h2cia[:, j] = h2cia_opacity(f, stim_j)

mol_total = chop + ohop + h2cia

# =============================================================================
# COMPARISON ONLY -- load pykurucz ground truth (NOT used above)
# =============================================================================
truth = np.load(REF / "mol_continuum_truth.npz")
chop_t = truth["chop"]
ohop_t = truth["ohop"]
h2cia_t = truth["h2cia"]
total_t = truth["mol_total"]


def max_rel(a, b, floor=0.0):
    """Max relative error over genuinely-nonzero points (|b| > floor)."""
    a = np.asarray(a); b = np.asarray(b)
    mask = np.abs(b) > floor
    if not np.any(mask):
        return 0.0, 0
    rel = np.abs(a[mask] - b[mask]) / np.abs(b[mask])
    return float(rel.max()), int(mask.sum())


print("=" * 72)
print("MOLECULAR CONTINUOUS OPACITY -- from-scratch (numpy) vs pykurucz")
print("=" * 72)
print(f"grid: {n_layers} layers x {nfreq} wavelengths, "
      f"{wl_nm.min():.0f}-{wl_nm.max():.0f} nm; T = {temp.min():.0f}-{temp.max():.0f} K")
print()

results = {}
for name, comp, ref in [("CHOP   (CH)", chop, chop_t),
                        ("OHOP   (OH)", ohop, ohop_t),
                        ("H2-CIA (Borysow)", h2cia, h2cia_t),
                        ("MOL TOTAL", mol_total, total_t)]:
    nz = int((np.abs(ref) > 0).sum())
    floor = np.abs(ref[ref != 0]).max() * 1e-300 if nz else 0.0  # ~0; keep all nonzeros
    mr, npts = max_rel(comp, ref)
    results[name] = mr
    print(f"  {name:<18} max|rel| = {mr:.3e}   over {npts:>6d} nonzero pts "
          f"(peak {np.abs(ref).max():.3e} cm^2/g)")

print()
# ── Dominance demonstration: H2-CIA in the cool-star near-IR continuum ────────
# Use a representative cool photospheric layer (T ~ 3000 K, the M-dwarf regime
# where H2 forms and CIA dominates the continuum).  jdeep kept for the figure.
jdeep = n_layers - 1
jcool = int(np.argmin(np.abs(temp - 3000.0)))
print(f"H2-CIA dominance (cool photospheric layer, T={temp[jcool]:.0f} K):")
for target in (450.0, 1000.0, 1500.0, 2200.0):
    i = int(np.argmin(np.abs(wl_nm - target)))
    tot = mol_total[jcool, i]
    frac = (h2cia[jcool, i] / tot * 100.0) if tot > 0 else 0.0
    print(f"  {wl_nm[i]:6.0f} nm : CHOP={chop[jcool,i]:.3e} OHOP={ohop[jcool,i]:.3e} "
          f"H2CIA={h2cia[jcool,i]:.3e}  -> H2-CIA = {frac:5.1f}% of mol continuum")

print()
worst = max(results.values())
TARGET = 1e-9
status = "PASS" if worst < TARGET else "FAIL"
print("=" * 72)
print(f"{status}: molecular continuous opacity reproduced to max rel = {worst:.3e} "
      f"(target < {TARGET:.0e})")
print("   CHOP/OHOP/H2-CIA all COMPUTED from atmosphere state + shipped tables;")
print("   pykurucz output loaded for COMPARISON ONLY.")
print("=" * 72)

# ── optional figure (matplotlib only) ────────────────────────────────────────
if __name__ == "__main__":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(wl_nm, chop[jdeep], label="CH (CHOP)")
        ax.plot(wl_nm, ohop[jdeep], label="OH (OHOP)")
        ax.plot(wl_nm, h2cia[jdeep], label="H2-CIA (Borysow)")
        ax.plot(wl_nm, mol_total[jdeep], "k--", lw=0.8, label="molecular total")
        ax.set_yscale("log")
        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel(r"molecular continuum opacity [cm$^2$/g]")
        ax.set_title(f"Cool-dwarf molecular continuum, T={temp[jdeep]:.0f} K (from scratch)")
        ax.legend()
        ax.set_ylim(1e-12, None)
        fig.tight_layout()
        out = ROOT / "_pipeline" / "mol_continuum_check.png"
        fig.savefig(out, dpi=110)
        print(f"figure -> {out}")
    except Exception as e:
        print(f"(figure skipped: {e})")
